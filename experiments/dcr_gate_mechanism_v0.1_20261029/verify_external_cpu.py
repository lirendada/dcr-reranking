"""Independent real retrieval audit: all saved scores, fixed-hash full-search checks and coverage."""
from collections import Counter
import csv
import json
import math
from pathlib import Path
import hashlib
import string
import time
from adapter import sha256, write_json

ROOT = Path(__file__).resolve().parent


def tokens(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).split()


def main():
    started = time.monotonic()
    folder = ROOT / 'external_retrieval'
    bundle = ROOT / 'external_inputs'
    manifest = json.loads((folder / 'retrieval_manifest.json').read_text())
    assert manifest['candidate_sha256'] == sha256(folder / 'candidates.csv')
    assert manifest['run_sha256'] == sha256(folder / 'bm25_top100.csv')
    queries = {r['query_id']: r['text'] for r in map(json.loads, (bundle / 'queries_RQ_rewrite.jsonl').open())}
    corpus = list(map(json.loads, (bundle / 'corpus.jsonl').open()))
    tfs = [Counter(tokens(d['text'])) for d in corpus]
    df = Counter(term for tf in tfs for term in tf)
    n = len(tfs)
    avg = sum(sum(tf.values()) for tf in tfs) / n
    norm = [1.5 * (.25 + .75 * sum(tf.values()) / avg) for tf in tfs]
    idf = {t: math.log1p((n - f + .5) / (f + .5)) for t, f in df.items()}
    def score(q, i):
        tf = tfs[i]
        return sum(idf[t] * (2.5 * tf[t]) / (tf[t] + norm[i]) for t in tokens(q) if tf[t])
    runs, total, maxerr = {}, 0, 0
    for r in csv.DictReader((folder / 'bm25_top100.csv').open()):
        qid, did, value = r['query_id'], r['document_id'], float(r['score'])
        entries = runs.setdefault(qid, [])
        assert did not in {d for d, _ in entries}
        assert math.isfinite(value)
        err = abs(value - score(queries[qid], int(did[1:])))
        maxerr = max(maxerr, err)
        assert err < 1e-10
        entries.append((did, value))
        total += 1
    assert set(runs) == set(queries) and total == 345200
    for entries in runs.values():
        assert len(entries) == 100 and entries == sorted(entries, key=lambda r: (-r[1], r[0]))
    candidates = [(r['query_id'], r['document_id']) for r in csv.DictReader((folder / 'candidates.csv').open())]
    assert candidates == [(q, d) for q, docs in runs.items() for d, _ in docs]
    selected = sorted(queries, key=lambda q: hashlib.sha256(('dcr-external-bm25-audit-20261029-v1:' + q).encode()).hexdigest())[:8]
    import numpy as np
    # Independent document-by-document scoring over full90406 for eight hash-selected queries.
    # Use precomputed postings-free TF records, not the production retrieval's arrays/partition.
    for qid in selected:
        qt = tokens(queries[qid])
        weights = [(t, idf[t]) for t in qt if t in idf]
        values = []
        for i, tf in enumerate(tfs):
            values.append(sum(w * (2.5 * tf[t]) / (tf[t] + norm[i]) for t, w in weights if tf[t]))
        order = sorted(range(n), key=lambda i: (-values[i], i))[:100]
        assert [d for d, _ in runs[qid]] == [f'd{i:06d}' for i in order]
    refs = json.loads((bundle / 'references.json').read_text())
    official = {r['query_id']: r for r in csv.DictReader((folder / 'evaluation/per_query.csv').open())}
    for qid, pair in refs.items():
        ranks = {did: i + 1 for i, (did, _) in enumerate(runs[qid])}
        pr, nr = ranks.get(pair['positive_id']), ranks.get(pair['negative_id'])
        row = official[qid]
        assert row['positive_rank'] == (str(pr) if pr else '')
        assert row['negative_rank'] == (str(nr) if nr else '')
        for k in (1, 5, 10, 100):
            assert float(row[f'positive_recall@{k}']) == float(pr is not None and pr <= k)
            assert float(row[f'negative_entry@{k}']) == float(nr is not None and nr <= k)
        assert float(row['author_right_rank@10']) == float(pr is not None and pr <= 10 and (nr is None or nr > 10 or pr < nr))
    write_json(folder / 'independent_verification.json', {'status': 'passed', 'queries': len(runs),
        'all_saved_pair_scores_recomputed': total, 'maximum_absolute_score_error': maxerr,
        'independent_full_corpus_top100_queries': selected, 'full_corpus_documents': n,
        'all_candidate_ids_bound': True, 'all_reference_rank_and_hit_rows_recomputed': 3452,
        'seconds': time.monotonic() - started, 'numpy_version': np.__version__})
    print('Independent external retrieval score/top100/coverage audit passed.', flush=True)


if __name__ == '__main__':
    main()
