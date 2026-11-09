"""Independent reconstruction from saved gates/rankings and official qrels; no model."""
import csv
import gzip
import json
import math
from pathlib import Path
import random
from adapter import write_json
from gate_cpu import ROOT, PREV, ev, verify


def read_csv(path):
    return list(csv.DictReader(path.open()))


def metric(order, refs):
    positives = {did for did, rel in refs.items() if rel > 0}
    hits, total = 0, 0
    for rank, did in enumerate(order, 1):
        if did in positives:
            hits += 1
            total += hits / rank
    ideal = sum(rel / math.log2(i + 2) for i, rel in enumerate(sorted(refs.values(), reverse=True)[:10]))
    actual = sum(refs.get(did, 0) / math.log2(i + 2) for i, did in enumerate(order[:10]))
    return total / len(positives) if positives else 0, actual / ideal if ideal else 0


def quantile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def main():
    verify()
    out = ROOT / 'robust_results'
    qrels = {}
    for side in ('og', 'changed'):
        refs = {}
        for line in (ev.DATA / ('qrels_' + side) / 'test.jsonl').open():
            row = json.loads(line)
            refs.setdefault(str(row['query-id']), {})[str(row['corpus-id'])] = int(row['score'])
        qrels[side] = refs
    gates = {}
    with gzip.open(out / 'gates.jsonl.gz', 'rt') as stream:
        for line in stream:
            r = json.loads(line)
            assert (r['qid'], r['side'], r['pid']) not in gates
            assert r['Qwen_gate'] == (r['no_logit'] > r['yes_logit'])
            gates[r['qid'], r['side'], r['pid']] = r
    assert len(gates) == 104000
    # Compare stored components against the original source files independently.
    with gzip.open(PREV / 'reuse_results/scores.jsonl.gz', 'rt') as stream:
        for line in stream:
            r = json.loads(line)
            g = gates[r['qid'], r['side'], r['pid']]
            assert g['lexical_gate'] == r['violates'] and g['base_score'] == r['ce_query_score']
    with gzip.open(PREV / 'qwen_results/scores.jsonl.gz', 'rt') as stream:
        for line in stream:
            r = json.loads(line)
            if r['role'] != 'query_only':
                g = gates[r['qid'], r['role'][7:], r['pid']]
                assert g['yes_logit'] == r['yes_logit'] and g['no_logit'] == r['no_logit']
    pools = {}
    for line in (ev.DATA / 'top_ranked.jsonl').open():
        r = json.loads(line)
        pools.setdefault(str(r['qid']), []).append(str(r['pid']))
    calculated = {}
    with gzip.open(out / 'rankings.jsonl.gz', 'rt') as stream:
        records = [json.loads(line) for line in stream]
    assert len(records) == 208
    for r in records:
        qid, method = r['qid'], r['method']
        for side, field in [('og', 'original'), ('changed', 'changed')]:
            pool = pools[qid]
            g = {d: gates[qid, side, d] for d in pool}
            if method == 'Qwen_direct':
                expected = sorted(pool, key=lambda d: -(g[d]['yes_logit'] - g[d]['no_logit']))
            else:
                key = {'CE_no_gate': None, 'CE_lexical_gate': 'lexical_gate', 'CE_Qwen_gate': 'Qwen_gate'}[method]
                expected = sorted(pool, key=lambda d: (g[d][key] if key else False, -g[d]['base_score']))
            assert r[field] == expected
        before = {d: i + 1 for i, d in enumerate(r['original'])}
        after = {d: i + 1 for i, d in enumerate(r['changed'])}
        flips = [d for d, rel in qrels['og'][qid].items() if rel > 0 and qrels['changed'][qid].get(d, 0) == 0]
        movements = []
        for d in flips:
            b, a = before[d], after[d]
            movements.append((a - b) / a if a > b else (a - b) / b)
        ap, ndcg = metric(r['changed'], qrels['changed'][qid])
        calculated[method, qid] = {'p_mrr': sum(movements) / len(movements),
                                 'ap_changed_1000': ap, 'ndcg_changed_10': ndcg}
    rows = read_csv(out / 'per_query.csv')
    for r in rows:
        for key, value in calculated[r['method'], r['qid']].items():
            assert abs(value - float(r[key])) < 1e-12
    for r in read_csv(out / 'aggregate.csv'):
        for key in ('p_mrr', 'ap_changed_1000', 'ndcg_changed_10'):
            values = [v[key] for (method, _), v in calculated.items() if method == r['method']]
            assert len(values) == 52
            assert abs(100 * sum(values) / 52 - float(r[key + '_pct'])) < 1e-10
    qids = list(pools)
    rng = random.Random(20261027)
    draws = [[rng.randrange(52) for _ in range(52)] for _ in range(5000)]
    for row in read_csv(out / 'paired_bootstrap.csv'):
        a, b = row['comparison'].split(' - ')
        diff = [calculated[a, q][row['metric']] - calculated[b, q][row['metric']] for q in qids]
        samples = [100 * sum(diff[i] for i in draw) / 52 for draw in draws]
        for key, value in [('difference_pp', 100 * sum(diff) / 52),
                           ('ci_low_pp', quantile(samples, .025)), ('ci_high_pp', quantile(samples, .975))]:
            assert abs(float(row[key]) - value) < 1e-10, (row, key, value)
    verify()
    write_json(out / 'independent_verification.json', {'status': 'passed', 'queries': 52,
        'method_queries': len(rows), 'source_bound_gates': len(gates), 'all_rankings_reconstructed': True,
        'metrics_independently_recomputed': True, 'bootstrap_contrasts': 12,
        'frozen_sources_verified_before_after': True})
    print('Independent score/rank/source/bootstrap verification passed.')


if __name__ == '__main__':
    main()
