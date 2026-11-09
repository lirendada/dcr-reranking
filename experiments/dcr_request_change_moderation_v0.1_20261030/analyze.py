#!/usr/bin/env python3
"""Stratified moderation analysis of input effects with grouped bootstrap.
Pure Python (no numpy). Resampling spec (documented for independent reproduction):
  rng = random.Random(seed); for draw in range(draws): for fold in sorted folds:
    comps = components of that fold sorted by representative id; n = len(comps);
    for _ in range(n): idx = rng.randrange(n); count[comps[idx]] += 1
  query weight = count of its component. All statistics in a draw share the weights.
Percentile CI: sorted draws, linear interpolation at (p/100)*(n-1)."""
import argparse, csv, gzip, json, math, random, statistics

def load_perquery(path, key_col, cells, filt=None):
    P = {c: {} for c in cells}; N = {c: {} for c in cells}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if filt and not filt(r): continue
            c = r[key_col]
            if c in cells:
                P[c][r["query_id"]] = float(r["P1"]); N[c][r["query_id"]] = float(r.get("N1", "nan") or "nan")
    return P, N

def tertile_cuts(vals):
    s = sorted(vals); n = len(s); return s[n // 3], s[2 * n // 3]

def tertile(v, t1, t2):
    return "low" if v <= t1 else ("mid" if v <= t2 else "high")

def wmean(ids, vals, w):
    sw = sum(w[q] for q in ids)
    return sum(w[q] * vals[q] for q in ids) / sw if sw else float("nan")

def wslope(ids, x, y, w):
    sw = sum(w[q] for q in ids)
    if sw == 0: return float("nan")
    mx = sum(w[q] * x[q] for q in ids) / sw; my = sum(w[q] * y[q] for q in ids) / sw
    sxx = sum(w[q] * (x[q] - mx) ** 2 for q in ids)
    return sum(w[q] * (x[q] - mx) * (y[q] - my) for q in ids) / sxx if sxx else float("nan")

def pct(sorted_vals, p):
    n = len(sorted_vals); pos = (p / 100.0) * (n - 1); lo = int(math.floor(pos)); hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--perquery", required=True)
    ap.add_argument("--groups", help="ExcluIR groups.json (components+component_fold)")
    ap.add_argument("--cluster-col", help="IFIR: cluster bootstrap by this per_query column (e.g. base_id)")
    ap.add_argument("--dataset", help="IFIR: filter per_query dataset column")
    ap.add_argument("--key-col", default="cell"); ap.add_argument("--weak", default="MiniLM"); ap.add_argument("--strong", default="Qwen")
    ap.add_argument("--strat-primary", default="jaccard"); ap.add_argument("--strat-secondary", default="topic_retention")
    ap.add_argument("--primary-direction", default="low_minus_high", choices=["low_minus_high", "high_minus_low"])
    ap.add_argument("--draws", type=int, default=1000); ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    cells = [f"{a.weak}_topic", f"{a.weak}_full", f"{a.strong}_topic", f"{a.strong}_full"]
    filt = (lambda r: r["dataset"] == a.dataset) if a.dataset else None
    P, N = load_perquery(a.perquery, a.key_col, cells, filt)
    feats = {r["query_id"]: {k: float(v) for k, v in r.items() if k != "query_id"} for r in csv.DictReader(open(a.features, encoding="utf-8"))}
    qids = sorted(q for q in feats if all(q in P[c] for c in cells))
    assert len(qids) == len(feats), f"feature/perquery mismatch: {len(qids)} vs {len(feats)}"
    dW = {q: P[cells[1]][q] - P[cells[0]][q] for q in qids}; dS = {q: P[cells[3]][q] - P[cells[2]][q] for q in qids}
    dWN = {q: N[cells[1]][q] - N[cells[0]][q] for q in qids}; dSN = {q: N[cells[3]][q] - N[cells[2]][q] for q in qids}
    # clusters
    if a.groups:
        g = json.load(open(a.groups, encoding="utf-8")); comps = g["components"]; cfold = g["component_fold"]
        q2c = {q: rep for rep, mem in comps.items() for q in mem}
        folds = {}
        for rep in comps: folds.setdefault(cfold[rep], []).append(rep)
        cluster_desc = {"unit": "components", "n_clusters": len(comps), "strata": {str(k): len(v) for k, v in sorted(folds.items())}}
    else:
        col = a.cluster_col; q2c = {}
        with gzip.open(a.perquery, "rt", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if (not filt or filt(r)) and r["query_id"] in feats: q2c[r["query_id"]] = r[col]
        comps = {}
        for q, c in q2c.items(): comps.setdefault(c, []).append(q)
        folds = {0: list(comps.keys())}
        cluster_desc = {"unit": col, "n_clusters": len(comps), "strata": {"0": len(comps)}}
    for k in folds: folds[k].sort()
    assert set(q2c) >= set(qids), "every query must belong to a cluster"
    # strata (fixed)
    strata = {}
    for var in [a.strat_primary, a.strat_secondary]:
        t1, t2 = tertile_cuts([feats[q][var] for q in qids])
        strata[var] = {"cuts": [t1, t2], "groups": {}}
        for q in qids: strata[var]["groups"].setdefault(tertile(feats[q][var], t1, t2), []).append(q)
    strata["neg_cue"] = {"cuts": None, "groups": {}}
    for q in qids: strata["neg_cue"]["groups"].setdefault("neg_cue=1" if feats[q]["neg_cue"] else "neg_cue=0", []).append(q)
    if a.cluster_col:  # IFIR: also by level
        lv = {}
        with gzip.open(a.perquery, "rt", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if (not filt or filt(r)) and r["query_id"] in feats: lv[r["query_id"]] = r.get("level", "")
        strata["level"] = {"cuts": None, "groups": {}}
        for q in qids: strata["level"]["groups"].setdefault(f"level={lv[q]}", []).append(q)
    one = {q: 1 for q in qids}
    # point stratum table
    rows = []
    for var, S in strata.items():
        for gname, ids in sorted(S["groups"].items()):
            rows.append({"variable": var, "stratum": gname, "n": len(ids),
                **{f"P1_{c}": 100 * wmean(ids, P[c], one) for c in cells},
                f"dP1_{a.weak}": 100 * wmean(ids, dW, one), f"dP1_{a.strong}": 100 * wmean(ids, dS, one),
                f"dN1_{a.weak}": 100 * wmean(ids, dWN, one), f"dN1_{a.strong}": 100 * wmean(ids, dSN, one),
                f"mean_{var}" if S["cuts"] else "mean_var": statistics.mean(feats[q][var] for q in ids) if S["cuts"] else float("nan")})
    with open(f"{a.outdir}/strata_results.csv", "w", newline="", encoding="utf-8") as fh:
        keys = []; [keys.append(k) for r in rows for k in r if k not in keys]
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    # statistics per draw
    pv = a.strat_primary; sv = a.strat_secondary
    lo_p, hi_p = strata[pv]["groups"].get("low", []), strata[pv]["groups"].get("high", [])
    lo_s, hi_s = strata[sv]["groups"].get("low", []), strata[sv]["groups"].get("high", [])
    sign = 1 if a.primary_direction == "low_minus_high" else -1
    xj = {q: feats[q][pv] for q in qids}
    def stats(w):
        pW = sign * 100 * (wmean(lo_p, dW, w) - wmean(hi_p, dW, w)); pS = sign * 100 * (wmean(lo_p, dS, w) - wmean(hi_p, dS, w))
        return {
            f"PRIMARY_dP1_{a.weak}_{pv}_{a.primary_direction}": pW,
            f"dP1_{a.strong}_{pv}_{a.primary_direction}": pS,
            f"interaction_{a.strong}_minus_{a.weak}": pS - pW,
            f"slope_dP1_{a.weak}_on_{pv}_pp_per_unit": 100 * wslope(qids, xj, dW, w),
            f"slope_dP1_{a.strong}_on_{pv}_pp_per_unit": 100 * wslope(qids, xj, dS, w),
            f"dP1_{a.weak}_{sv}_{a.primary_direction}": sign * 100 * (wmean(lo_s, dW, w) - wmean(hi_s, dW, w)),
            f"dP1_{a.strong}_{sv}_{a.primary_direction}": sign * 100 * (wmean(lo_s, dS, w) - wmean(hi_s, dS, w)),
            f"topicP1_{a.weak}_{pv}_{a.primary_direction}": sign * 100 * (wmean(lo_p, P[cells[0]], w) - wmean(hi_p, P[cells[0]], w)),
            f"dP1_{a.weak}_all": 100 * wmean(qids, dW, w), f"dP1_{a.strong}_all": 100 * wmean(qids, dS, w),
        }
    point = stats(one)
    rng = random.Random(a.seed); draws = []
    for d in range(a.draws):
        cnt = {}
        for f in sorted(folds):
            cl = folds[f]; n = len(cl)
            for _ in range(n):
                rep = cl[rng.randrange(n)]; cnt[rep] = cnt.get(rep, 0) + 1
        w = {q: cnt.get(q2c[q], 0) for q in qids}
        draws.append(stats(w))
    with gzip.open(f"{a.outdir}/bootstrap_draws.csv.gz", "wt", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=["draw"] + list(point.keys())); wtr.writeheader()
        for i, s in enumerate(draws): wtr.writerow({"draw": i, **s})
    intervals = []
    for k in point:
        vals = sorted(s[k] for s in draws if not math.isnan(s[k]))
        intervals.append({"statistic": k, "point": point[k], "ci95_low": pct(vals, 2.5), "ci95_high": pct(vals, 97.5), "draws_used": len(vals), "unit": "pp"})
    with open(f"{a.outdir}/bootstrap_intervals.csv", "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(intervals[0].keys())); wtr.writeheader(); wtr.writerows(intervals)
    summary = {"n_queries": len(qids), "cells": cells, "clusters": cluster_desc, "draws": a.draws, "seed": a.seed,
               "strata_cuts": {k: v["cuts"] for k, v in strata.items()}, "strata_sizes": {k: {g: len(i) for g, i in v["groups"].items()} for k, v in strata.items()},
               "primary": intervals[0], "intervals": intervals,
               "resampling_spec": "per draw: for fold in sorted(folds): clusters sorted by id, n=len; n times rep=clusters[rng.randrange(n)], count+=1; query weight=count of its cluster; random.Random(seed) consumed sequentially across folds and draws",
               "percentile_spec": "sorted draws; linear interpolation at (p/100)*(n-1)"}
    json.dump(summary, open(f"{a.outdir}/summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps({"n": len(qids), "primary": intervals[0]}, ensure_ascii=False))

if __name__ == "__main__":
    main()
