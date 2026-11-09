"""Separate numerical implementation. Does not import production fitting or scoring."""
from collections import Counter
import csv
import math
import random
import time
import numpy as np
from config import (ROOT, POST, ATTR, SAME, Q7, M5, REF, METHODS, COMPARISONS,
                    TARGETS, read, write, rows, sha, check, target_spec, now)

LAMBDAS = [.001, .01, .1, 1.]


def ridge(x, positive, negative, weights, lam):
    indices = np.flatnonzero((positive >= 0) & (negative >= 0) & (weights > 0))
    assert len(indices)
    d = x[indices, positive[indices]] - x[indices, negative[indices]]
    w = weights[indices].astype(float)
    count = int(w.sum())
    scale = np.sqrt(np.sum(w[:, None] * d * d, axis=0) / count)
    scale[scale == 0] = 1.
    z = d / scale
    coefficient = np.linalg.solve(z.T @ (w[:, None] * z) + count * lam * np.eye(x.shape[-1]), z.T @ w)
    return scale, coefficient, count


def ordered(x, model, canonical):
    scores = np.matmul(x / model[0], model[1])
    canonical_scores = np.take_along_axis(scores, canonical, axis=1)
    position = np.broadcast_to(np.arange(x.shape[1]), canonical.shape)
    return np.take_along_axis(canonical, np.lexsort((position, -canonical_scores), axis=1), axis=1)


def metric(order, positive, negative):
    p = np.equal(order[:, 0], positive)
    n = np.equal(order[:, 0], negative)
    r = np.sum(order[:, :10] == positive[:, None], axis=1) > 0
    return np.column_stack((p, n, ~(p | n), r))


def target_metrics(order, reference):
    positives = {d for d in reference if reference[d] > 0}
    reciprocal = 0.
    gains = []
    for i, document in enumerate(order[:20]):
        if not reciprocal and document in positives:
            reciprocal = 1. / (i + 1)
        gains.append(reference.get(document, 0) / math.log2(i + 2))
    ideal = [r / math.log2(i + 2) for i, r in enumerate(sorted((r for r in reference.values() if r > 0), reverse=True)[:20])]
    return [int(order[0] in positives), sum(gains) / sum(ideal) if sum(ideal) else 0., reciprocal,
            sum(d in positives for d in order[:10]) / len(positives) if positives else 0.]


def percentile(values, fraction):
    sorted_values = sorted(float(v) for v in values)
    index = fraction * (len(sorted_values) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    return sorted_values[lower] + (index - lower) * (sorted_values[upper] - sorted_values[lower])


def main() -> None:
    start = time.monotonic()
    check()
    manifest = read(ROOT / 'frozen_manifest.json')
    summary = read(ROOT / 'summary.json')
    seal = read(ROOT / 'predictions_sealed.json')
    assert seal['frozen_manifest_sha256'] == sha(ROOT / 'frozen_manifest.json')
    for filename, expected in seal['files'].items():
        assert sha(ROOT / filename) == expected, filename
    with np.load(POST / 'features.npz', allow_pickle=False) as data:
        original, canonical = data['X'][:, :, :12], data['canonical']
    with np.load(SAME / 'features.npz', allow_pickle=False) as data:
        assert np.array_equal(original[:, :, Q7], data['X'][:, :, Q7])
    assert sorted(Q7 + M5) == list(range(12))
    x = np.ascontiguousarray(original[:, :, Q7])
    with np.load(POST / 'prepared/plan.npz', allow_pickle=False) as data:
        folds, weights = data['fold'], data['weights']
    meta, references = read(POST / 'head_and_tail.json'), read(REF)
    positive, negative = [], []
    for qid, pool in zip(meta['qids'], meta['ranked_pools']):
        positive.append(next((i for i, d in enumerate(pool[:20]) if d == references[qid]['positive_id']), -1))
        negative.append(next((i for i, d in enumerate(pool[:20]) if d == references[qid]['negative_id']), -1))
    positive, negative = np.array(positive), np.array(negative)
    with np.load(ROOT / 'source_metrics.npz', allow_pickle=False) as data:
        produced_values = data['values']
    with np.load(SAME / 'point_orders.npz', allow_pickle=False) as data:
        fixed = data['orders'][data['methods'].tolist().index('fixed')]
    fixed_indicators = metric(fixed, positive, negative)
    rebuilt = np.empty_like(produced_values)
    verified_models = 0
    for fold in range(5):
        test = np.flatnonzero(folds == fold)
        view = read(POST / 'prepared' / f'view_{fold}.json')
        assert np.array_equal(view['test_indices'], test)
        with np.load(view['labels_path'], allow_pickle=False) as data:
            assert np.array_equal(positive[data['train_indices']], data['positive'])
            assert np.array_equal(negative[data['train_indices']], data['negative'])
        with np.load(ROOT / 'fitted' / f'fold_{fold}.npz', allow_pickle=False) as data:
            saved = {k: data[k] for k in data.files}
        for draw in range(1001):
            w = np.ones(3452, dtype=np.int16) if not draw else weights[draw - 1]
            counts = np.zeros(4, dtype=np.int64)
            for inner in range(5):
                if inner == fold:
                    continue
                validation = np.flatnonzero((folds == inner) & (w > 0))
                training = w * ((folds != fold) & (folds != inner))
                for j, lam in enumerate(LAMBDAS):
                    model = ridge(x, positive, negative, training, lam)
                    order = ordered(x[validation], model, canonical[validation])
                    counts[j] += np.sum(w[validation] * (order[:, 0] == positive[validation]))
            lam = LAMBDAS[int(np.argmax(counts))]
            model = ridge(x, positive, negative, w * (folds != fold), lam)
            order = ordered(x[test], model, canonical[test])
            assert saved['lambda'][draw] == lam and np.array_equal(saved['inner_correct_counts'][draw], counts)
            assert np.allclose(saved['scale'][draw], model[0], atol=1e-12, rtol=1e-12)
            assert np.allclose(saved['coef'][draw], model[1], atol=1e-11, rtol=1e-11)
            assert saved['pair_instances'][draw] == model[2]
            assert np.array_equal(saved['order'][draw], order), (fold, draw)
            verified_models += 1
        print(f'verified fold {fold}: all1001 fits/orders', flush=True)
    # Rebuild old and new source metrics from full source orders and original draws.
    old_folds = {}
    for method, directory, key in [('G12', POST, 'G_order'), ('QL5', ATTR, 'QL5_order'), ('G_same12', SAME, 'order'), ('Q7', ROOT, 'order')]:
        old_folds[method] = []
        for fold in range(5):
            with np.load(directory / 'fitted' / f'fold_{fold}.npz', allow_pickle=False) as data:
                old_folds[method].append((data['test_indices'], data[key]))
    source_points = {}
    for draw in range(1001):
        w = np.ones(3452, dtype=np.int16) if not draw else weights[draw - 1]
        for j, method in enumerate(METHODS):
            if method == 'fixed':
                indicators = fixed_indicators
            else:
                order = np.empty((3452, 20), dtype=np.uint8)
                for indices, orders in old_folds[method]:
                    order[indices] = orders[draw]
                indicators = metric(order, positive, negative)
            rebuilt[draw, j] = np.sum(w[:, None].astype(float) * indicators, axis=0) / int(w.sum())
            if draw == 0:
                source_points[method] = indicators[:, 0]
    assert np.allclose(rebuilt, produced_values, atol=1e-14, rtol=0)
    counts = np.zeros(4, dtype=np.int64)
    for fold in range(5):
        test = np.flatnonzero(folds == fold)
        for j, lam in enumerate(LAMBDAS):
            model = ridge(x, positive, negative, (folds != fold).astype(np.int16), lam)
            counts[j] += np.sum(ordered(x[test], model, canonical[test])[:, 0] == positive[test])
    selected = LAMBDAS[int(np.argmax(counts))]
    fitted = ridge(x, positive, negative, np.ones(3452), selected)
    model = read(ROOT / 'transfer_model.json')
    assert model['lambda'] == selected and model['CV_correct_by_lambda'] == counts.tolist()
    assert fitted[2] == model['pair_instances'] == 2901
    assert np.allclose(fitted[0], model['scale'], atol=1e-12, rtol=1e-12)
    assert np.allclose(fitted[1], model['coef'], atol=1e-11, rtol=1e-11)
    target_arrays, target_means, target_meta = {}, {}, {}
    rankings_checked = 3452
    for dataset in TARGETS:
        spec = target_spec(dataset)
        metadata = read(ROOT / 'targets' / f'{dataset}_metadata.json')
        if dataset == 'cds':
            assert metadata == read(spec['meta'])
        else:
            old_meta = read(spec['meta'])
            assert [q['query_id'] for q in metadata] == old_meta['qids']
            assert [q['head'] for q in metadata] == old_meta['heads']
        target_meta[dataset] = metadata
        with np.load(spec['features'], allow_pickle=False) as data:
            tx = data[spec['feature_key']][:, :, Q7]
        references = {q['query_id']: {} for q in metadata}
        with spec['qrels'].open() as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                if row['query_id'] in references:
                    references[row['query_id']][row['corpus_id']] = float(row['score'])
        all_orders = {(r['query_id'], r['method']): r['order'] for r in rows(spec['rankings'])}
        produced_orders = {r['query_id']: r['order'] for r in rows(ROOT / 'targets' / f'{dataset}_rankings.jsonl.gz')}
        tc = np.array([sorted(range(20), key=lambda i: q['head'][i]) for q in metadata])
        q7order = ordered(tx, fitted, tc)
        arr = []
        for i, q in enumerate(metadata):
            qid = q['query_id']
            order = [q['head'][j] for j in q7order[i]] + q['base'][20:]
            assert order == produced_orders[qid] and set(order) == set(q['base'])
            all_orders[qid, 'Q7'] = order
            arr.append([target_metrics(all_orders[qid, 'Qwen_full' if m == 'fixed' else m], references[qid]) for m in METHODS])
        arr = np.array(arr)
        with np.load(ROOT / f'{dataset}_metrics.npz', allow_pickle=False) as data:
            assert np.allclose(arr, data['values'], atol=1e-14, rtol=0)
            target_means[dataset] = data['draws']
        target_arrays[dataset] = arr
        rankings_checked += len(metadata)
    # Independently regenerate the topic counts and weighted request means.
    generators = {'fiqa': random.Random(20261029), 'cds': random.Random(20261102)}
    generators['nfcorpus'] = generators['fiqa']
    unique = {d: sorted({q['base_id'] for q in target_meta[d]}) for d in TARGETS}
    loaded_counts = {}
    for dataset in TARGETS:
        with np.load(ROOT / f'{dataset}_metrics.npz', allow_pickle=False) as data:
            loaded_counts[dataset] = data['topic_counts']
    for draw in range(10000):
        for dataset in TARGETS:
            topics = unique[dataset]
            rng = generators[dataset]
            count = Counter(topics[rng.randrange(len(topics))] for _ in topics)
            assert [count[t] for t in topics] == loaded_counts[dataset][draw].tolist()
            w = np.array([count[q['base_id']] for q in target_meta[dataset]])
            means = np.sum(w[:, None] * target_arrays[dataset][:, :, 0], axis=0) / int(w.sum())
            assert np.allclose(means, target_means[dataset][draw], atol=1e-14, rtol=0)
    for contrast in summary['contrasts']:
        population, metric = contrast['population'], contrast['metric']
        a, b = contrast['comparison'].split('-')
        if population == 'ExcluIR':
            index = ['P1', 'N1', 'U1', 'R10'].index(metric)
            point, samples = rebuilt[0, :, index], rebuilt[1:, :, index]
        elif population == 'equal_weight_macro':
            point = (target_arrays['fiqa'][:, :, 0].mean(0) + target_arrays['nfcorpus'][:, :, 0].mean(0)) / 2
            samples = (target_means['fiqa'] + target_means['nfcorpus']) / 2
        else:
            point, samples = target_arrays[population][:, :, 0].mean(0), target_means[population]
        diff = 100 * (samples[:, METHODS.index(a)] - samples[:, METHODS.index(b)])
        expected = [100 * (point[METHODS.index(a)] - point[METHODS.index(b)]), percentile(diff, .025), percentile(diff, .975)]
        assert np.allclose(expected, [contrast['difference_pp'], contrast['ci_low_pp'], contrast['ci_high_pp']], atol=1e-10, rtol=0)
        assert contrast['primary'] == (population == 'ExcluIR' and metric == 'P1' and a == 'G_same12' and b == 'Q7')
    assert len(summary['contrasts']) == 18
    check()
    write(ROOT / 'verification.json', {'status': 'PASS', 'finished_utc': now(),
          'source_fold_draw_models_checked': verified_models, 'source_transfer_model_checked': True,
          'Q7_rankings_checked': rankings_checked,
          'ranking_scope': '3452 source head20 orders and 1847 complete target100 orders; original tails preserved.',
          'source_metric_draw_fields_checked': int(rebuilt.size),
          'target_metric_fields_checked': sum(int(a.size) for a in target_arrays.values()),
          'target_topic_draws_checked_per_collection': 10000, 'contrasts_checked': 18,
          'production_functions_imported': False, 'same_agent_authored_both_implementations': True,
          'neural_forward_or_BM25_repeated': False, 'seconds': time.monotonic() - start,
          'code_sha256': sha(ROOT / 'verify.py'), 'source_hashes_after': 'PASS'})


if __name__ == '__main__':
    main()
