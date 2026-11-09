"""Released-reference metrics and frozen paired-topic bootstrap after sealing."""
from collections import Counter
from datetime import datetime, timezone
import csv
import gzip
import json
import math
import os
import random
import time
import numpy as np

from config import ROOT, QRELS, N, METHODS, METRICS, CONTRASTS, rows, read, write, sha, check_release


def evaluate(order, reference):
    positive = {did for did, grade in reference.items() if grade > 0}
    first = next((i for i, did in enumerate(order[:20], 1) if did in positive), 0)
    dcg = sum(reference.get(did, 0) / math.log2(i + 1) for i, did in enumerate(order[:20], 1))
    ideal = sum(grade / math.log2(i + 1) for i, grade in enumerate(sorted((v for v in reference.values() if v > 0), reverse=True)[:20], 1))
    return [float(order[0] in positive), dcg / ideal if ideal else 0., 1 / first if first else 0.,
            len(set(order[:10]) & positive) / len(positive) if positive else 0.]


def bootstrap(values, bases, draws=10000):
    unique = sorted(set(bases))
    generator = random.Random(20261102)
    counts = np.zeros((draws, len(unique)), dtype=np.int32)
    for draw in range(draws):
        for _ in unique:
            counts[draw, generator.randrange(len(unique))] += 1
    mapping = {base: i for i, base in enumerate(unique)}
    weights = counts[:, [mapping[base] for base in bases]]
    result = np.einsum('dq,qmk->dmk', weights, values) / weights.sum(axis=1)[:, None, None]
    return counts, result


def main() -> None:
    start = time.monotonic()
    assert os.environ.get('DCR_SCORING_PHASE') == '1'
    check_release()
    seal = read(ROOT / 'predictions_sealed.json')
    assert seal['qrels_read'] is False
    for name, expected in seal['files'].items():
        assert sha(ROOT / name) == expected, name
    ref_receipt = read(ROOT / 'reference_receipt.json')
    assert sha(QRELS) == ref_receipt['sha256']
    with gzip.open(ROOT / 'corpus_ids.json.gz', 'rt') as stream:
        corpus_ids = set(json.load(stream))
    all_queries = {str(q['_id']) for q in rows(ROOT / 'queries/cds-query.jsonl')}
    references = {}
    with QRELS.open() as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        assert {'query-id', 'corpus-id', 'score'} <= set(reader.fieldnames)
        for row in reader:
            qid, did, grade = row['query-id'], row['corpus-id'], float(row['score'])
            assert qid in all_queries and did in corpus_ids and math.isfinite(grade) and grade >= 0
            assert did not in references.setdefault(qid, {})
            references[qid][did] = grade
    meta = read(ROOT / 'ranking_metadata.json')
    qids = [m['query_id'] for m in meta]
    values = np.zeros((N, len(METHODS), len(METRICS)), dtype=np.float64)
    rankings = {(r['query_id'], r['method']): r['order'] for r in rows(ROOT / 'rankings.jsonl.gz')}
    per_query, coverage = [], []
    zero_reference = 0
    for i, q in enumerate(meta):
        ref = references.get(q['query_id'], {})
        positive = {did for did, grade in ref.items() if grade > 0}
        zero_reference += not bool(positive)
        coverage.append({'query_id': q['query_id'], 'known_positive': len(positive),
            'positive_in_top100': len(set(q['base']) & positive), 'positive_in_head20': len(set(q['head']) & positive)})
        for j, method in enumerate(METHODS):
            order = rankings[q['query_id'], method]
            values[i, j] = evaluate(order, ref)
            per_query.append({'query_id': q['query_id'], 'base_id': q['base_id'], 'method': method,
                **dict(zip(METRICS, values[i, j].tolist())), 'top1_unjudged': order[0] not in ref})
    assert values.shape == (43, 7, 4) and np.isfinite(values).all()
    counts, draws = bootstrap(values, [q['base_id'] for q in meta])
    np.savez_compressed(ROOT / 'bootstrap_arrays.npz', topic_counts=counts, draws=draws, values=values)
    point = values.mean(axis=0)
    contrasts, transitions = [], []
    for a, b in CONTRASTS:
        ai, bi = METHODS.index(a), METHODS.index(b)
        distribution = (draws[:, ai, 0] - draws[:, bi, 0]) * 100
        low, high = np.percentile(distribution, [2.5, 97.5], method='linear')
        contrasts.append({'comparison': f'{a}-{b}', 'primary': a == 'G_same12' and b == 'QL5',
            'difference_pp': float((point[ai, 0] - point[bi, 0]) * 100),
            'ci95_low_pp': float(low), 'ci95_high_pp': float(high)})
        first, second = values[:, ai, 0], values[:, bi, 0]
        transitions.append({'comparison': f'{a}-{b}', 'rescues': int(((first == 1) & (second == 0)).sum()),
                            'losses': int(((first == 0) & (second == 1)).sum())})
    effect = (draws[:, 3, 0] - draws[:, 2, 0] - draws[:, 1, 0] + draws[:, 0, 0]) * 100
    low, high = np.percentile(effect, [2.5, 97.5], method='linear')
    contrasts.append({'comparison': 'model_by_input_interaction', 'primary': False,
        'difference_pp': float((point[3, 0] - point[2, 0] - point[1, 0] + point[0, 0]) * 100),
        'ci95_low_pp': float(low), 'ci95_high_pp': float(high)})
    with gzip.open(ROOT / 'per_query_metrics.jsonl.gz', 'xt') as stream:
        for row in per_query:
            stream.write(json.dumps(row) + '\n')
    write(ROOT / 'coverage.json', coverage)
    write(ROOT / 'summary.json', {'finished_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': 'IFIR-CDS', 'queries': N, 'original_topics': len(set(q['base_id'] for q in meta)),
        'point_metrics_percent': {m: dict(zip(METRICS, (point[i] * 100).tolist())) for i, m in enumerate(METHODS)},
        'contrasts': contrasts, 'transitions': transitions,
        'zero_positive_reference_queries': int(zero_reference),
        'coverage': {'positive_in_top100_queries': sum(r['positive_in_top100'] > 0 for r in coverage),
                     'positive_in_head20_queries': sum(r['positive_in_head20'] > 0 for r in coverage)},
        'bootstrap': {'unit': 'original topic', 'draws': 10000, 'seed': 20261102,
                      'statistic': 'request-weighted mean', 'refit': False, 'intervals': 'pointwise percentile linear'},
        'predictions_sealed_sha256': sha(ROOT / 'predictions_sealed.json'),
        'reference_sha256': sha(QRELS), 'new_model_pairs': 6880,
        'scope': 'Additional published-reference transfer evaluation conditional on frozen source models and technical-compatibility selection.',
        'score_seconds': time.monotonic() - start})


if __name__ == '__main__':
    main()
