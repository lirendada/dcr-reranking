#!/usr/bin/env python3
"""Compute per-query request-change features (topic vs full) with the project's
historical tokenizer: lowercase, strip ASCII punctuation, whitespace split."""
import argparse, csv, json, re, string

NEG_CUES = ["not", "no", "without", "except", "excluding", "exclude", "other than",
            "but not", "rather than", "instead of", "besides", "apart from", "non",
            "never", "neither", "nor", "unrelated", "outside"]
_NEG_RE = re.compile(r"\b(" + "|".join(re.escape(c) for c in NEG_CUES) + r")\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def tokenize(text):
    return text.lower().translate(_PUNCT_TABLE).split()

def features(topic, full):
    t, r = tokenize(topic), tokenize(full)
    st, sr = set(t), set(r)
    inter, union = len(st & sr), len(st | sr)
    t_norm, r_norm = " ".join(t), " ".join(r)
    return {
        "len_topic": len(t), "len_full": len(r),
        "jaccard": inter / union if union else 0.0,
        "topic_retention": inter / len(st) if st else 0.0,
        "added_tokens": len(sr - st), "removed_tokens": len(st - sr),
        "literal_containment": int(bool(t_norm) and t_norm in r_norm),
        "neg_cue": int(bool(_NEG_RE.search(full.lower()))),
    }

FIELDS = ["query_id", "len_topic", "len_full", "jaccard", "topic_retention",
          "added_tokens", "removed_tokens", "literal_containment", "neg_cue"]

def run_excluir(path, out):
    data = json.load(open(path, encoding="utf-8"))
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader()
        for i, q in enumerate(data):
            row = features(q["question0"], q["RQ_rewrite"]); row["query_id"] = f"q{i:06d}"
            w.writerow(row)
    return len(data)

def run_pairs(path, out):
    """Generic: JSONL/JSON list of {query_id, topic, full} (used for IFIR)."""
    rows = []
    if path.endswith(".jsonl"):
        for line in open(path, encoding="utf-8"):
            if line.strip(): rows.append(json.loads(line))
    else:
        rows = json.load(open(path, encoding="utf-8"))
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader()
        for q in rows:
            row = features(q["topic"], q["full"]); row["query_id"] = q["query_id"]; w.writerow(row)
    return len(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["excluir", "pairs"], required=True)
    ap.add_argument("--input", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = run_excluir(a.input, a.out) if a.mode == "excluir" else run_pairs(a.input, a.out)
    print(json.dumps({"rows": n, "out": a.out, "tokenizer": "lowercase; delete ASCII punctuation chars; whitespace split"}))
