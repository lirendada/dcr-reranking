"""All populations, all fixed contrasts; no fitting or model selection in scoring."""
import csv
import gzip
import json
import math
import random
import time
import numpy as np
from config import (ROOT, POST, SAME, IFIR, CDS, REF, METHODS, COMPARISONS,
                    SOURCE_METRICS, TARGET_METRICS, TARGETS, read, write, rows, sha, check, target_spec)


def evaluate(order: list, reference: dict) -> list:
    positives = {document for document, grade in reference.items() if grade > 0}
    first = next((rank for rank, document in enumerate(order[:20], 1) if document in positives), 0)
    gain = sum(reference.get(document, 0) / math.log2(rank + 1) for rank, document in enumerate(order[:20], 1))
    ideal = sum(grade / math.log2(rank + 1) for rank, grade in enumerate(sorted((g for g in reference.values() if g > 0), reverse=True)[:20], 1))
    return [float(order[0] in positives), gain / ideal if ideal else 0., 1. / first if first else 0.,
            len(set(order[:10]) & positives) / len(positives) if positives else 0.]


def indicators(order, positive, negative):
    hits, negatives = order[:, 0] == positive, order[:, 0] == negative
    return np.stack([hits, negatives, ~(hits | negatives), (order[:, :10] == positive[:, None]).any(1)], axis=1)


def interval(population, metric, a, b, point, samples, primary=False):
    diff = samples[:, METHODS.index(a)] - samples[:, METHODS.index(b)]
    lo, hi = np.quantile(diff * 100, [.025, .975])
    return {'population': population, 'metric': metric, 'comparison': a + '-' + b,
            'primary': primary, 'difference_pp': 100 * float(point[METHODS.index(a)] - point[METHODS.index(b)]),
            'ci_low_pp': float(lo), 'ci_high_pp': float(hi), 'draws': len(diff)}


def main() -> None:
    start = time.monotonic()
    check()
    seal = read(ROOT / 'predictions_sealed.json')
    assert not seal['target_qrels_read']
    for name, expected in seal['files'].items():
        assert sha(ROOT / name) == expected, name
    meta, refs = read(POST / 'head_and_tail.json'), read(REF)
    pos, neg = [], []
    for qid, pool in zip(meta['qids'], meta['ranked_pools']):
        head = pool[:20]
        pos.append(head.index(refs[qid]['positive_id']) if refs[qid]['positive_id'] in head else -1)
        neg.append(head.index(refs[qid]['negative_id']) if refs[qid]['negative_id'] in head else -1)
    pos, neg = np.array(pos), np.array(neg)
    with np.load(POST / 'prepared/plan.npz', allow_pickle=False) as plan:
        weights = plan['weights']
    with np.load(SAME / 'metrics_draws.npz', allow_pickle=False) as old:
        legacy = {name: old['values'][:, i] for i, name in enumerate(old['methods'].tolist())}
    folds = []
    for fold in range(5):
        with np.load(ROOT / 'fitted' / f'fold_{fold}.npz', allow_pickle=False) as data:
            folds.append({'indices': data['test_indices'], 'order': data['order']})
    values = np.empty((1001, len(METHODS), 4))
    point_order = None
    for draw in range(1001):
        order = np.empty((3452, 20), dtype=np.uint8)
        for fold in folds:
            order[fold['indices']] = fold['order'][draw]
        w = np.ones(3452) if not draw else weights[draw - 1].astype(float)
        for j, method in enumerate(METHODS):
            values[draw, j] = np.sum(w[:, None] * indicators(order, pos, neg), axis=0) / w.sum() if method == 'Q7' else legacy[method][draw]
        if not draw:
            point_order = order.copy()
    np.savez_compressed(ROOT / 'source_metrics.npz', values=values, Q7_order=point_order, methods=np.array(METHODS))
    contrasts = []
    for a, b in COMPARISONS:
        for j, metric in enumerate(SOURCE_METRICS[:2]):
            contrasts.append(interval('ExcluIR', metric, a, b, values[0, :, j], values[1:, :, j],
                                      a == 'G_same12' and b == 'Q7' and metric == 'P1'))
    source = {'queries': 3452, 'reports_pct': {m: dict(zip(SOURCE_METRICS, (values[0, j] * 100).tolist())) for j, m in enumerate(METHODS)}}
    with np.load(SAME / 'point_orders.npz', allow_pickle=False) as old:
        source_points = {m: indicators(old['orders'][i], pos, neg)[:, 0] for i, m in enumerate(old['methods'].tolist())}
    source_points['Q7'] = indicators(point_order, pos, neg)[:, 0]
    source['transitions'] = {a + '-' + b: {'rescues': int((source_points[a] & ~source_points[b]).sum()),
                                          'losses': int((~source_points[a] & source_points[b]).sum())} for a, b in COMPARISONS}
    targets, arrays, metadata, points = {}, {}, {}, []
    for dataset in TARGETS:
        spec = target_spec(dataset)
        metadata[dataset] = queries = read(ROOT / 'targets' / f'{dataset}_metadata.json')
        judgments = {q['query_id']: {} for q in queries}
        with spec['qrels'].open() as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                if row['query_id'] in judgments:
                    target = judgments[row['query_id']]
                    assert row['corpus_id'] not in target
                    target[row['corpus_id']] = float(row['score'])
        orders = {(row['query_id'], row['method']): row['order'] for row in rows(spec['rankings'])}
        orders.update({(row['query_id'], 'Q7'): row['order'] for row in rows(ROOT / 'targets' / f'{dataset}_rankings.jsonl.gz')})
        aliases = {'fixed': 'Qwen_full'}
        array = np.array([[evaluate(orders[q['query_id'], aliases.get(m, m)], judgments[q['query_id']]) for m in METHODS] for q in queries])
        assert np.isfinite(array).all()
        old_summary = read(spec['summary'])
        old_points = old_summary['point_metrics_percent'] if dataset == 'cds' else old_summary['datasets'][dataset]['reports_pct']
        for method, old_metrics in old_points.items():
            old_array = np.array([evaluate(orders[q['query_id'], method], judgments[q['query_id']]) for q in queries])
            assert np.allclose(old_array.mean(0) * 100, [old_metrics[k] for k in TARGET_METRICS], atol=1e-10, rtol=0), (dataset, method)
        report = {m: dict(zip(TARGET_METRICS, (array[:, j].mean(0) * 100).tolist())) for j, m in enumerate(METHODS)}
        targets[dataset] = {'queries': len(queries), 'original_topics': len({q['base_id'] for q in queries}),
                            'zero_positive_requests': sum(not any(g > 0 for g in ref.values()) for ref in judgments.values()),
                            'reports_pct': report, 'old_all_method_point_metrics_reproduced': True,
                            'transitions': {a + '-' + b: {'rescues': int(((array[:, METHODS.index(a), 0] == 1) & (array[:, METHODS.index(b), 0] == 0)).sum()),
                                                         'losses': int(((array[:, METHODS.index(a), 0] == 0) & (array[:, METHODS.index(b), 0] == 1)).sum())} for a, b in COMPARISONS}}
        arrays[dataset] = array
        for i, query in enumerate(queries):
            for j, method in enumerate(METHODS):
                points.append({'dataset': dataset, 'query_id': query['query_id'], 'base_id': query['base_id'],
                               'method': method, **dict(zip(TARGET_METRICS, array[i, j].tolist()))})
    # Preserve the original two-dataset draw ordering; CDS has its own fixed RNG.
    counts, base_lists = {}, {}
    for dataset in TARGETS:
        base_lists[dataset] = sorted({q['base_id'] for q in metadata[dataset]})
        counts[dataset] = np.zeros((10000, len(base_lists[dataset])), dtype=np.int16)
    rng = random.Random(20261029)
    for draw in range(10000):
        for dataset in ['fiqa', 'nfcorpus']:
            n = len(base_lists[dataset])
            for _ in range(n):
                counts[dataset][draw, rng.randrange(n)] += 1
    rng = random.Random(20261102)
    for draw in range(10000):
        n = len(base_lists['cds'])
        for _ in range(n):
            counts['cds'][draw, rng.randrange(n)] += 1
    means = {}
    for dataset in TARGETS:
        bases = base_lists[dataset]
        totals = np.zeros((len(bases), len(METHODS)))
        sizes = np.zeros(len(bases))
        mapping = {base: i for i, base in enumerate(bases)}
        for i, query in enumerate(metadata[dataset]):
            totals[mapping[query['base_id']]] += arrays[dataset][i, :, 0]
            sizes[mapping[query['base_id']]] += 1
        means[dataset] = (counts[dataset] @ totals) / (counts[dataset] @ sizes)[:, None]
        np.savez_compressed(ROOT / f'{dataset}_metrics.npz', values=arrays[dataset], draws=means[dataset], topic_counts=counts[dataset])
    macro_point = (arrays['fiqa'].mean(0) + arrays['nfcorpus'].mean(0)) / 2
    macro_draws = (means['fiqa'] + means['nfcorpus']) / 2
    targets['equal_weight_macro'] = {'collections': ['fiqa', 'nfcorpus'], 'reports_pct': {m: dict(zip(TARGET_METRICS, (macro_point[j] * 100).tolist())) for j, m in enumerate(METHODS)}}
    for dataset in TARGETS + ['equal_weight_macro']:
        point = macro_point[:, 0] if dataset == 'equal_weight_macro' else arrays[dataset].mean(0)[:, 0]
        draws = macro_draws if dataset == 'equal_weight_macro' else means[dataset]
        for a, b in COMPARISONS:
            contrasts.append(interval(dataset, 'P1', a, b, point, draws))
    assert len(contrasts) == 18 and sum(c['primary'] for c in contrasts) == 1
    with gzip.open(ROOT / 'per_query_metrics.jsonl.gz', 'xt') as stream:
        for row in points:
            stream.write(json.dumps(row) + '\n')
    write(ROOT / 'summary.json', {'source': source, 'targets': targets, 'contrasts': contrasts,
          'source_draws': 1000, 'target_draws': 10000, 'new_neural_pairs': 0,
          'observed_data': True, 'transfer_model': read(ROOT / 'transfer_model.json'),
          'seconds': time.monotonic() - start, 'prediction_seal_sha256': sha(ROOT / 'predictions_sealed.json')})


if __name__ == '__main__':
    main()
