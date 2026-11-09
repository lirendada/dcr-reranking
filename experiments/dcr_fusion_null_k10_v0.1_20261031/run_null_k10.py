#!/usr/bin/env python3
"""Frozen CPU null fits use one training view per process; only score reads gold."""
import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = Path('/path/to/remote/lw_experiments/runs')
POST = RUNS / 'dcr_post_fusion_v0.1_20261029'
CELLS = RUNS / 'dcr_model_input_v0.1_20261029'
MATCHED = RUNS / 'dcr_input_matched_fusion_v0.1_20261029'
ATTR = RUNS / 'dcr_fusion_attribution_v0.1_20261029'
REF = RUNS / 'dcr_gate_mechanism_v0.1_20261029/external_inputs/references.json'
VARIANTS = ('null_G12', 'null_same12')
SEEDS = (20261031, 20261032)
S_NULL = 200
METRICS = ('P1', 'N1', 'U1', 'R10')
CONTRASTS = {'MiniLM_full-MiniLM_topic': (-1, 1, 0, 0),
             'Qwen_full-Qwen_topic': (0, 0, -1, 1),
             'Qwen_topic-MiniLM_topic': (-1, 0, 1, 0),
             'Qwen_full-MiniLM_full': (0, -1, 0, 1),
             'input_effect_interaction': (1, -1, -1, 1)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def verify(phase, fold=None):
    manifest = read(ROOT / 'frozen_manifest.json')
    categories = ['code', 'common', 'score'] if phase == 'score' else ['code', 'common', f'fold_{fold}']
    for category in categories:
        for path, digest in manifest[category].items():
            assert sha(path) == digest, f'Frozen source changed: {path}'
    return manifest


def core():
    spec = importlib.util.spec_from_file_location('frozen_core', ATTR / 'math_core.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rebuild(old, scores, canonical):
    result = old.copy()
    for i, row in enumerate(scores):
        std = row.std(ddof=0)
        z = (row-row.mean()) / std if std else np.zeros_like(row)
        order = canonical[i][np.argsort(-row[canonical[i]], kind='stable')]
        result[i, :, 0] = z
        for rank, column in enumerate(order):
            result[i, column, 2] = (rank+1)/20
        gap_m = np.sort(z)[-1]-np.sort(z)[-2]
        q = old[i, :, 1]
        gap_q = np.sort(q)[-1]-np.sort(q)[-2]
        result[i, :, 8] = z*gap_m
        result[i, :, 9] = z*gap_q
        result[i, :, 10] = q*gap_m
    assert np.array_equal(result[:, :, [1, 3, 4, 5, 6, 7, 11]], old[:, :, [1, 3, 4, 5, 6, 7, 11]])
    return result


def fit_fold(fold_id, precheck):
    started = time.monotonic()
    verify('fit', fold_id)
    view = read(POST / 'prepared' / f'view_{fold_id}.json')
    with np.load(POST / 'features.npz', allow_pickle=False) as data:
        old = data['X'][:, :, :12].copy()
        canonical = data['canonical']
        mini_topic = data['raw_scores'][:, :, 0]
    with np.load(MATCHED / 'features.npz', allow_pickle=False) as data:
        same = data['X']
        mini_full = data['MiniLM_full_scores']
        assert np.array_equal(canonical, data['canonical'])
    with np.load(POST / 'prepared/plan.npz', allow_pickle=False) as data:
        folds = data['fold']
    with np.load(view['labels_path'], allow_pickle=False) as data:
        train = data['train_indices']
        positive = np.full(len(old), -2, dtype=np.int8)
        negative = positive.copy()
        positive[train] = data['positive']
        negative[train] = data['negative']
    test = np.asarray(view['test_indices'])
    assert np.array_equal(train, np.flatnonzero(folds != fold_id))
    assert np.array_equal(test, np.flatnonzero(folds == fold_id))
    assert np.all(positive[test] == -2) and np.all(negative[test] == -2)
    assert np.array_equal(rebuild(old, mini_topic, canonical), old)
    assert np.array_equal(rebuild(old, mini_full, canonical), same)
    maths = core()
    ones = np.ones(len(old), dtype=np.int16)

    def fitted(features):
        lam, counts = maths.select(features, positive, negative, canonical, folds, fold_id, ones)
        model = maths.solve(maths.statistics(features, positive, negative, ones*(folds != fold_id)), lam)
        return maths.rank(maths.predict(features[test], model), canonical[test]), lam, model, counts

    baseline = []
    for variant, features, directory, order_key, lambda_key in [
        ('G12', old, POST, 'G_order', 'G_lambda'),
        ('G_same12', same, MATCHED, 'order', 'lambda'),
    ]:
        order, lam, model, counts = fitted(features)
        with np.load(directory / 'fitted' / f'fold_{fold_id}.npz', allow_pickle=False) as saved:
            assert np.array_equal(saved['test_indices'], test)
            assert np.array_equal(order, saved[order_key][0]), f'{variant}: frozen order mismatch'
            assert lam == saved[lambda_key][0], f'{variant}: frozen lambda mismatch'
        baseline.append(order)
    destination = ROOT / ('precheck' if precheck else 'fits')
    destination.mkdir(exist_ok=True)
    target = destination / f'fold_{fold_id}.npz'
    assert not target.exists(), str(target)
    if precheck:
        np.savez_compressed(target, test_indices=test, baseline_order=np.asarray(baseline))
    else:
        orders = np.empty((2, S_NULL, len(test), 20), dtype=np.uint8)
        lambdas = np.empty((2, S_NULL))
        coefficients = np.empty((2, S_NULL, 12))
        scales = np.empty_like(coefficients)
        counts_all = np.empty((2, S_NULL, 4), dtype=np.int32)
        for variant, (scores, seed) in enumerate(zip((mini_topic, mini_full), SEEDS)):
            rng = np.random.default_rng(seed)
            for draw in range(S_NULL):
                permutations = np.stack([rng.permutation(20) for _ in range(len(old))])
                features = rebuild(old, np.take_along_axis(scores, permutations, axis=1), canonical)
                order, lam, model, counts = fitted(features)
                orders[variant, draw] = order
                lambdas[variant, draw] = lam
                scales[variant, draw] = model[0]
                coefficients[variant, draw] = model[1]
                counts_all[variant, draw] = counts
                if draw % 50 == 0:
                    print(json.dumps({'fold': fold_id, 'variant': VARIANTS[variant], 'draw': draw,
                                      'elapsed_seconds': time.monotonic()-started}), flush=True)
        np.savez_compressed(target, test_indices=test, baseline_order=np.asarray(baseline),
                            order=orders, lambdas=lambdas, scale=scales, coefficient=coefficients,
                            inner_correct_counts=counts_all)
    verify('fit', fold_id)
    write(target.with_suffix('.json'), {'fold': fold_id, 'precheck': precheck, 'sha256': sha(target),
        'frozen_manifest_sha256': sha(ROOT/'frozen_manifest.json'),
        'training_label_sha256': sha(view['labels_path']), 'references_opened': False,
        'heldout_labels_masked': True, 'frozen_baseline_orders_identical': True,
        'elapsed_seconds': time.monotonic()-started, 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def indicators(order, positive, negative):
    p = order[..., 0] == positive
    n = order[..., 0] == negative
    return np.stack([p, n, ~(p | n), (order[..., :10] == positive[..., None]).any(-1)], axis=-1).astype(float)


def part_a(orders, cells, positive, negative, plan, ks):
    reports = {}
    values_20 = None
    for k in ks:
        current = orders.copy()
        if k < 20:
            for c in range(4):
                for i, row in enumerate(orders[c]):
                    current[c, i] = np.concatenate([row[row < k], np.arange(k, 20, dtype=row.dtype)])
        ind = np.stack([indicators(order, positive, negative) for order in current])
        values = np.empty((1001, 4, 4))
        values[0] = ind.mean(1)
        for iteration, weights in enumerate(plan, 1):
            values[iteration] = (ind*weights[None, :, None]).sum(1)/weights.sum()
        comparisons = []
        for name, vector in CONTRASTS.items():
            for j, metric in enumerate(METRICS[:2]):
                diff = (values[:, :, j] @ np.asarray(vector))*100
                low, high = np.quantile(diff[1:], [.025, .975])
                comparisons.append({'K': k, 'comparison': name, 'metric': metric,
                    'primary': name == 'input_effect_interaction' and metric == 'P1',
                    'difference_pp': float(diff[0]), 'ci_low_pp': float(low), 'ci_high_pp': float(high)})
        reports[str(k)] = {'cells': {cell: {metric: float(values[0, c, j]*100)
            for j, metric in enumerate(METRICS)} for c, cell in enumerate(cells)}, 'contrasts': comparisons}
        if k == 20:
            values_20 = values
    with np.load(CELLS/'bootstrap_metrics.npz', allow_pickle=False) as original:
        assert np.array_equal(values_20, original['values']), 'All original K20 draws must match'
    old = read(CELLS/'summary.json')
    expected = {(v['comparison'], v['metric']): v for v in old['comparisons']}
    for row in reports['20']['contrasts']:
        reference = expected[row['comparison'], row['metric']]
        for field in ['difference_pp', 'ci_low_pp', 'ci_high_pp']:
            assert row[field] == reference[field], (row['comparison'], field)
    return reports


def score(precheck):
    started = time.monotonic()
    verify('score')
    meta = read(POST/'head_and_tail.json')
    refs = read(REF)
    qids, pools = meta['qids'], meta['ranked_pools']
    positive = np.array([pool[:20].index(refs[q]['positive_id']) if refs[q]['positive_id'] in pool[:20] else -1
                         for q, pool in zip(qids, pools)], dtype=np.int8)
    negative = np.array([pool[:20].index(refs[q]['negative_id']) if refs[q]['negative_id'] in pool[:20] else -1
                         for q, pool in zip(qids, pools)], dtype=np.int8)
    with np.load(CELLS/'four_cell_arrays.npz', allow_pickle=False) as data:
        orders, cells = data['orders'], list(data['cells'])
        assert list(data['qids']) == qids
    with np.load(POST/'prepared/plan.npz', allow_pickle=False) as data:
        plan = data['weights']
    baseline = np.empty((2, len(qids), 20), dtype=np.uint8)
    folder = ROOT/('precheck' if precheck else 'fits')
    all_orders = None if precheck else np.empty((2, S_NULL, len(qids), 20), dtype=np.uint8)
    all_lambdas = None if precheck else np.empty((2, S_NULL, 5))
    seen = np.zeros(len(qids), dtype=int)
    for fold in range(5):
        source = folder/f'fold_{fold}.npz'
        receipt = read(source.with_suffix('.json'))
        assert receipt['sha256'] == sha(source)
        assert receipt['frozen_manifest_sha256'] == sha(ROOT/'frozen_manifest.json')
        with np.load(source, allow_pickle=False) as data:
            test = data['test_indices']
            baseline[:, test] = data['baseline_order']
            seen[test] += 1
            if not precheck:
                all_orders[:, :, test] = data['order']
                all_lambdas[:, :, fold] = data['lambdas']
    assert np.all(seen == 1)
    observed_counts = indicators(baseline, positive, negative).sum(1).astype(int)
    old_same = {r['method']: r for r in read(MATCHED/'summary.json')['reports']}
    old_attr = {r['method']: r for r in read(ATTR/'summary.json')['reports']}
    for i, name in enumerate(('G12', 'G_same12')):
        reference = old_attr[name] if i == 0 else old_same[name]
        for j, metric in enumerate(METRICS):
            assert abs(observed_counts[i, j]*100/len(qids)-reference[metric+'_pct']) < 1e-10
    a = part_a(orders, cells, positive, negative, plan, (20,) if precheck else (20, 10))
    if precheck:
        verify('score')
        write(ROOT/'precheck/result.json', {'status': 'PASS', 'all_K20_draws_identical': True,
            'all_K20_contrasts_identical': True, 'G12_Same12_all_frozen_orders_identical': True,
            'observed_counts': observed_counts.tolist(), 'seconds': time.monotonic()-started})
        return
    seal = read(ROOT/'fits_sealed.json')
    for path, digest in seal['files'].items():
        assert sha(ROOT/path) == digest
    draws, nulls = [], {}
    for variant, name in enumerate(VARIANTS):
        counts = indicators(all_orders[variant], positive, negative).sum(1).astype(int)
        values = counts*100/len(qids)
        observed = observed_counts[variant, 0]*100/len(qids)
        for draw in range(S_NULL):
            draws.append({'variant': name, 'draw': draw, 'P1_correct_count': int(counts[draw, 0]),
                **{metric: float(values[draw, j]) for j, metric in enumerate(METRICS)},
                'lambdas': '|'.join(map(str, all_lambdas[variant, draw]))})
        nulls[name] = {'S': S_NULL, 'observed_correct_count': int(observed_counts[variant, 0]),
            'observed_P1': float(observed), 'null_P1_mean': float(values[:, 0].mean()),
            'null_P1_sd': float(values[:, 0].std(ddof=1)),
            'null_P1_p2.5': float(np.quantile(values[:, 0], .025)),
            'null_P1_p97.5': float(np.quantile(values[:, 0], .975)),
            'observed_minus_null_mean_pp': float(observed-values[:, 0].mean()),
            'fraction_null_ge_observed': float(np.mean(counts[:, 0] >= observed_counts[variant, 0])),
            'QL5_minus_null_mean_pp': float(old_attr['QL5']['P1_pct']-values[:, 0].mean()),
            'fixed_minus_null_mean_pp': float(old_attr['fixed']['P1_pct']-values[:, 0].mean()),
            'lambda_counts': {str(lam): int(np.sum(all_lambdas[variant] == lam)) for lam in (.001, .01, .1, 1.)}}
    with (ROOT/'null_draws.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(draws[0])); writer.writeheader(); writer.writerows(draws)
    for name, rows in [('k10_contrasts.csv', [r for k in a.values() for r in k['contrasts']]),
                       ('k10_cell_metrics.csv', [{'K': int(k), 'cell': cell, **vals}
                        for k, report in a.items() for cell, vals in report['cells'].items()])]:
        with (ROOT/name).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    verify('score')
    write(ROOT/'summary.json', {'part_A': a, 'part_B': nulls, 'new_neural_pairs': 0,
        'frozen_manifest_sha256': sha(ROOT/'frozen_manifest.json'), 'fit_seal_sha256': sha(ROOT/'fits_sealed.json'),
        'score_seconds': time.monotonic()-started,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__},
        'interpretation': 'Post-hoc finite permutation diagnostic; failure to distinguish is not proof of no information.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['fit', 'score'])
    parser.add_argument('--fold', type=int, choices=range(5))
    parser.add_argument('--precheck', action='store_true')
    args = parser.parse_args()
    if args.stage == 'fit':
        if args.fold is None:
            parser.error('--fold is required for fit')
        fit_fold(args.fold, args.precheck)
    else:
        score(args.precheck)
