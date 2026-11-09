"""Independent matched-input fusion audit; numerical execution is server-only.

Preparing this file does not read scores or fit models. The final runner will
require production completion and its frozen SHA. No new production learner,
feature or scoring module is imported by the independent implementation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent
POST = RUNS / 'dcr_post_fusion_v0.1_20261029'
ATTR = RUNS / 'dcr_fusion_attribution_v0.1_20261029'
FOUR = RUNS / 'dcr_model_input_v0.1_20261029'
GROUPS = RUNS / 'dcr_constraint_router_v0.1_20261029/prepared/groups.json'
LAMBDAS = (.001, .01, .1, 1.)
METHODS = ('fixed', 'G12', 'G_same12', 'QL5')
METRICS = ('P1', 'N1', 'U1', 'R10')
QL_COLUMNS = [1, 4, 5, 6, 7]
NUMERIC = {'comparisons': 0, 'maximum_absolute_difference': 0., 'outer_models': 0,
           'inner_candidate_solves': 0, 'exact_order_matrices': 0}


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def csv_rows(path: Path) -> list:
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', newline='') as stream:
        return list(csv.DictReader(stream))


def json_rows(path: Path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def close(actual, expected, label: str, tolerance: float = 1e-10) -> None:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), label
    error = float(np.max(np.abs(a-b))) if a.size else 0.
    NUMERIC['comparisons'] += int(a.size)
    NUMERIC['maximum_absolute_difference'] = max(NUMERIC['maximum_absolute_difference'], error)
    assert error <= tolerance, (label, error, tolerance)


def ranks(scores: np.ndarray, canonical: np.ndarray) -> np.ndarray:
    id_rank = np.empty_like(canonical)
    np.put_along_axis(id_rank, canonical, np.broadcast_to(np.arange(20), canonical.shape), axis=1)
    return np.lexsort((id_rank, -scores), axis=1).astype(np.uint8)


def zscore(scores: np.ndarray) -> np.ndarray:
    std = scores.std(axis=1, keepdims=True, ddof=0)
    return np.divide(scores-scores.mean(axis=1, keepdims=True), std,
                     out=np.zeros(scores.shape), where=std != 0)


def independent_features(mini_full: np.ndarray, qwen_full: np.ndarray,
                         old_x: np.ndarray, canonical: np.ndarray) -> np.ndarray:
    """Rows keep old candidate IDs; rank_M is recomputed, not replaced by row IDs."""
    assert mini_full.shape == qwen_full.shape == old_x.shape[:2]
    mz, qz = zscore(mini_full), zscore(qwen_full)
    mrank, qrank = np.empty(mini_full.shape), np.empty(qwen_full.shape)
    positions = np.broadcast_to(np.arange(1, 21)/20, mini_full.shape)
    np.put_along_axis(mrank, ranks(mini_full, canonical), positions, axis=1)
    np.put_along_axis(qrank, ranks(qwen_full, canonical), positions, axis=1)
    sm, sq = np.sort(mz, axis=1), np.sort(qz, axis=1)
    mgap, qgap = (sm[:, -1]-sm[:, -2])[:, None], (sq[:, -1]-sq[:, -2])[:, None]
    x = old_x[:, :, :12].copy()
    close(qz, old_x[:, :, 1], 'raw-source Qwen z invariant')
    close(qrank, old_x[:, :, 3], 'raw-source Qwen rank invariant')
    close(qz*qgap, old_x[:, :, 11], 'raw-source Qwen gap invariant')
    x[:, :, 0], x[:, :, 2] = mz, mrank
    x[:, :, 8], x[:, :, 9] = mz*mgap, mz*qgap
    x[:, :, 10] = old_x[:, :, 1]*mgap
    assert np.array_equal(x[:, :, 4:8], old_x[:, :, 4:8])
    assert np.array_equal(x[:, :, [1, 3, 11]], old_x[:, :, [1, 3, 11]])
    return x


def statistics(x: np.ndarray, positive: np.ndarray, negative: np.ndarray, weights: np.ndarray) -> tuple:
    active = (positive >= 0) & (negative >= 0) & (weights > 0)
    index = np.flatnonzero(active)
    assert len(index)
    difference = x[index, positive[index]]-x[index, negative[index]]
    w = weights[index].astype(float); count = int(w.sum())
    scale = np.sqrt(np.sum(w[:, None]*difference*difference, axis=0)/count)
    scale[scale == 0] = 1.
    d = difference/scale
    return scale, d.T @ (w[:, None]*d), d.T @ w, count


def solve(stats: tuple, regularization: float) -> tuple:
    scale, gram, rhs, count = stats
    matrix = gram+count*regularization*np.eye(len(scale))
    coefficient = np.linalg.solve(matrix, rhs)
    assert np.max(np.abs(matrix @ coefficient-rhs)) < 1e-8*max(1., np.max(np.abs(rhs)))
    return scale, coefficient, count


def predict(x: np.ndarray, model: tuple) -> np.ndarray:
    return (x/model[0]) @ model[1]


def choose(x: np.ndarray, canonical: np.ndarray, positive: np.ndarray, negative: np.ndarray,
           folds: np.ndarray, outer: int, weights: np.ndarray) -> tuple:
    totals = [0]*4
    for held in range(5):
        if held == outer:
            continue
        train = weights*((folds != outer) & (folds != held))
        stats = statistics(x, positive, negative, train)
        check = np.flatnonzero((folds == held) & (weights > 0))
        for j, regularization in enumerate(LAMBDAS):
            scores = predict(x[check], solve(stats, regularization))
            canonical_scores = np.take_along_axis(scores, canonical[check], axis=1)
            winner = canonical[check][np.arange(len(check)), np.argmax(canonical_scores, axis=1)]
            totals[j] += int((winner == positive[check]) @ weights[check])
            NUMERIC['inner_candidate_solves'] += 1
    return LAMBDAS[totals.index(max(totals))], totals


def verify_plan(qids: list, saved: dict) -> None:
    groups = read(GROUPS)
    assert groups['qids'] == qids and len(groups['components']) == 3239
    assert np.array_equal(saved['fold'], [groups['query_fold'][q] for q in qids])
    index = {q: i for i, q in enumerate(qids)}
    members = {c: [index[q] for q in qq] for c, qq in groups['components'].items()}
    rng = random.Random(20261029)
    for draw in range(1000):
        weights = np.zeros(len(qids), dtype=np.int16)
        for fold in range(5):
            components = groups['fold_component_IDs'][str(fold)]
            assert components == sorted(components)
            for _ in components:
                weights[members[components[rng.randrange(len(components))]]] += 1
        assert np.array_equal(weights, saved['weights'][draw]), ('component plan', draw)


def source_inputs() -> tuple:
    certificates = {
        POST: ('0c3d920df1cf5e180bb402bd8b7decf4a35fece2982ba63ad07276493ca60336',
               'PASS_INDEPENDENT_POST_FUSION_VERIFICATION'),
        ATTR: ('5d60e87a3cc7980cfb17bde93be0d611bd7035ba4020668fdea1fbc0986ff5c6',
               'PASS_INDEPENDENT_FUSION_ATTRIBUTION_VERIFICATION'),
        FOUR: ('1ac9797f1d82e2403a3e5d2e50d44d12ffecbf343b245482484c79da3078dc68',
               'PASS_INDEPENDENT_MODEL_INPUT_FACTORIAL_VERIFICATION')}
    for directory, (digest, status) in certificates.items():
        assert sha(directory / 'independent_verification.json') == digest
        assert read(directory / 'independent_verification.json')['status'] == status
    meta = read(POST / 'head_and_tail.json'); qids, pools = meta['qids'], meta['ranked_pools']
    assert len(qids) == len(pools) == 3452 and qids == sorted(qids) and len(set(qids)) == 3452
    assert all(len(p) == len(set(p)) == 100 for p in pools)
    old = arrays(POST / 'features.npz'); four = arrays(FOUR / 'four_cell_arrays.npz')
    assert four['qids'].tolist() == qids
    assert four['cells'].tolist() == ['MiniLM_topic', 'MiniLM_full', 'Qwen_topic', 'Qwen_full']
    close(four['scores'][0], old['raw_scores'][:, :, 0], 'source MiniLM topic', tolerance=0.)
    close(four['scores'][3], old['raw_scores'][:, :, 1], 'source Qwen full', tolerance=0.)
    # Bind convenience arrays to the actual score streams frozen as A inputs.
    score_certificate = read(FOUR / 'independent_verification.json')
    positions = {(q, d): (i, j) for i, (q, pool) in enumerate(zip(qids, pools)) for j, d in enumerate(pool[:20])}
    records = {}
    for cell, cell_index in [('MiniLM_full', 1), ('Qwen_full', 3)]:
        path = FOUR / 'cells' / cell / 'scores.jsonl.gz'
        assert sha(path) == score_certificate['cell_inventory'][cell]['scores_sha256']
        records[cell] = {}; values = np.empty((3452, 20))
        for r in json_rows(path):
            key = r['query_id'], r['document_id']; assert key in positions and key not in records[cell]
            records[cell][key] = r; i, j = positions[key]; values[i, j] = r['score']
        assert set(records[cell]) == set(positions)
        close(values, four['scores'][cell_index], (cell, 'score stream versus convenience array'), tolerance=0.)
    assert all(records['MiniLM_full'][key]['doc_prefix_end_char'] == records['Qwen_full'][key]['doc_prefix_end_char'] for key in positions)
    for name, digest in read(POST / 'features_sealed.json')['files'].items():
        assert sha(POST / name) == digest
    canonical = np.array([sorted(range(20), key=lambda i: pool[i]) for pool in pools])
    assert np.array_equal(canonical, old['canonical'])
    x = independent_features(four['scores'][1], four['scores'][3], old['X'], canonical)
    for column in (1, 3, 11):
        close(x[:, :, column], old['X'][:, :, column], ('unchanged Qwen feature', column))
    plan = arrays(POST / 'prepared/plan.npz'); verify_plan(qids, plan)
    saved = arrays(ROOT / 'features.npz')
    close(saved['X'], x, 'new matched feature matrix')
    assert np.array_equal(saved['canonical'], canonical)
    assert np.array_equal(saved['MiniLM_full_scores'], four['scores'][1])
    assert np.array_equal(saved['X'][:, :, [1, 3, 11]], old['X'][:, :, [1, 3, 11]])
    assert np.array_equal(saved['X'][:, :, 4:8], old['X'][:, :, 4:8])
    return x, old, meta, plan


def bit10(order: np.ndarray) -> np.ndarray:
    return np.sum(np.left_shift(np.uint32(1), order[:, :10].astype(np.uint32)), axis=1).astype(np.uint32)


def verify_folds(x: np.ndarray, old: dict, plan: dict) -> tuple:
    canonical, fold = old['canonical'], plan['fold']
    fixed = ranks(old['raw_scores'][:, :, 1], canonical)
    point = {name: np.empty((3452, 20), dtype=np.uint8) for name in METHODS}; point['fixed'] = fixed
    tops = {name: np.empty((1000, 3452), dtype=np.uint8) for name in METHODS}
    masks = {name: np.empty((1000, 3452), dtype=np.uint32) for name in METHODS}
    tops['fixed'][:] = fixed[:, 0]; masks['fixed'][:] = bit10(fixed)
    old_x = {'G12': old['X'][:, :, :12], 'QL5': old['X'][:, :, QL_COLUMNS].copy()}
    for outer in range(5):
        view = read(POST / 'prepared' / f'view_{outer}.json')
        for path, digest in view['allowed_inputs'].items():
            assert sha(Path(path)) == digest
        labels = arrays(Path(view['labels_path']))
        train, test = np.flatnonzero(fold != outer), np.flatnonzero(fold == outer)
        assert np.array_equal(labels['train_indices'], train) and view['test_indices'] == test.tolist()
        p, n = np.full(3452, -2), np.full(3452, -2)
        p[train], n[train] = labels['positive'], labels['negative']
        saved = arrays(ROOT / 'fitted' / f'fold_{outer}.npz')
        assert np.array_equal(saved['test_indices'], test)
        baseline = {}
        for name, directory, prefix in [('G12', POST, 'G'), ('QL5', ATTR, 'QL5')]:
            with np.load(directory / 'fitted' / f'fold_{outer}.npz', allow_pickle=False) as archive:
                assert np.array_equal(archive['test_indices'], test)
                baseline[name] = {key: archive[prefix+'_'+key] for key in ('scale', 'coef', 'order', 'point_scores')}
        for iteration in range(1001):
            weights = np.ones(3452, dtype=np.int32) if iteration == 0 else plan['weights'][iteration-1].astype(np.int32)
            regularization, counts = choose(x, canonical, p, n, fold, outer, weights)
            assert saved['lambda'][iteration] == regularization
            assert np.array_equal(saved['inner_correct_counts'][iteration], counts)
            model = solve(statistics(x, p, n, weights*(fold != outer)), regularization)
            close(saved['scale'][iteration], model[0], (outer, iteration, 'RMS'))
            close(saved['coef'][iteration], model[1], (outer, iteration, 'coefficient'))
            assert saved['pair_instances'][iteration] == model[2]
            score = predict(x[test], model); order = ranks(score, canonical[test])
            assert np.array_equal(saved['order'][iteration], order), (outer, iteration, 'new complete order')
            NUMERIC['outer_models'] += 1; NUMERIC['exact_order_matrices'] += 1
            if iteration == 0:
                close(saved['point_scores'], score, (outer, 'point scores')); point['G_same12'][test] = order
            else:
                tops['G_same12'][iteration-1, test] = order[:, 0]; masks['G_same12'][iteration-1, test] = bit10(order)
            for name, cached in baseline.items():
                score = predict(old_x[name][test], (cached['scale'][iteration], cached['coef'][iteration]))
                order = ranks(score, canonical[test])
                assert np.array_equal(order, cached['order'][iteration]), (outer, iteration, name, 'reused order')
                if iteration == 0:
                    close(cached['point_scores'], score, (outer, name, 'reused point scores')); point[name][test] = order
                else:
                    tops[name][iteration-1, test] = order[:, 0]; masks[name][iteration-1, test] = bit10(order)
            if iteration % 200 == 0:
                print(json.dumps({'phase': 'independent new fit and old reuse', 'fold': outer, 'iteration': iteration}), flush=True)
    saved = arrays(ROOT / 'point_orders.npz')
    assert saved['methods'].tolist() == list(METHODS)
    assert np.array_equal(saved['orders'], np.array([point[n] for n in METHODS]))
    return point, tops, masks


def verify_transfer(x: np.ndarray, old: dict, plan: dict, positive: np.ndarray, negative: np.ndarray) -> dict:
    saved = read(ROOT / 'transfer_models.json')
    assert saved['source_only'] == 'ExcluIR' and saved['IFIR_files_opened'] is False
    assert saved['lambda_tie'] == 'smallest'
    p = np.where((positive >= 0) & (positive < 20), positive, -1)
    n = np.where((negative >= 0) & (negative < 20), negative, -1)
    assert int(((p >= 0) & (n >= 0)).sum()) == 2901
    for name, features in [('G_same12', x), ('QL5', old['X'][:, :, QL_COLUMNS].copy())]:
        counts = [0]*4
        for held in range(5):
            train = (plan['fold'] != held).astype(np.int32); test = np.flatnonzero(plan['fold'] == held)
            stats = statistics(features, p, n, train)
            for j, regularization in enumerate(LAMBDAS):
                score = predict(features[test], solve(stats, regularization))
                winner = ranks(score, old['canonical'][test])[:, 0]
                counts[j] += int((winner == p[test]).sum())
        regularization = LAMBDAS[counts.index(max(counts))]
        model = solve(statistics(features, p, n, np.ones(3452, dtype=np.int32)), regularization)
        observed = saved['models'][name]
        assert observed['lambda'] == regularization and observed['CV_correct_by_lambda'] == counts
        assert observed['lambda_grid'] == list(LAMBDAS) and observed['CV_query_denominator'] == 3452
        assert observed['pair_instances'] == model[2] == 2901 and observed['dimensions'] == features.shape[2]
        close(observed['scale'], model[0], (name, 'full source RMS'))
        close(observed['coef'], model[1], (name, 'full source coefficient'))
    seal = read(ROOT / 'transfer_models_sealed.json')
    assert seal['models_sha256'] == sha(ROOT / 'transfer_models.json')
    assert seal['manifest_sha256'] == sha(ROOT / 'frozen_manifest.json')
    assert seal['before_A_score_and_any_new_IFIR_prediction'] is True
    return saved['models']


def verify_scores(meta: dict, point: dict, tops: dict, masks: dict, plan: dict) -> tuple:
    references = read(RUNS / 'dcr_gate_mechanism_v0.1_20261029/external_inputs/references.json')
    positive, negative = [], []
    for q, pool in zip(meta['qids'], meta['ranked_pools']):
        p, n = references[q]['positive_id'], references[q]['negative_id']
        positive.append(pool.index(p) if p in pool else -1); negative.append(pool.index(n) if n in pool else -1)
    positive, negative = np.array(positive), np.array(negative)
    phead = np.where((positive >= 0) & (positive < 20), positive, -1)
    nhead = np.where((negative >= 0) & (negative < 20), negative, -1)
    for f in range(5):
        label = arrays(POST / 'prepared' / f'labels_{f}.npz'); train = np.flatnonzero(plan['fold'] != f)
        assert np.array_equal(label['train_indices'], train)
        assert np.array_equal(label['positive'], phead[train]) and np.array_equal(label['negative'], nhead[train])
    values = np.empty((1001, 4, 4)); indicators = {}
    pbit = np.where((positive >= 0) & (positive < 20),
        np.left_shift(np.uint32(1), np.clip(positive, 0, 19).astype(np.uint32)), np.uint32(0))
    for j, name in enumerate(METHODS):
        order = point[name]; p, n = order[:, 0] == positive, order[:, 0] == negative
        indicator = np.column_stack([p, n, ~(p | n), np.any(order[:, :10] == positive[:, None], axis=1)])
        indicators[name] = indicator; values[0, j] = indicator.mean(axis=0)
        for draw, w in enumerate(plan['weights']):
            p, n = tops[name][draw] == positive, tops[name][draw] == negative
            hit10 = (masks[name][draw] & pbit) != 0
            values[draw+1, j] = np.array([w @ p, w @ n, w @ ~(p | n), w @ hit10])/w.sum()
    observed = arrays(ROOT / 'metrics_draws.npz')
    assert observed['methods'].tolist() == list(METHODS) and observed['metrics'].tolist() == list(METRICS)
    close(observed['values'], values, 'all point and bootstrap metrics')
    previous = arrays(ATTR / 'metrics_draws.npz')
    for name in ('fixed', 'G12', 'QL5'):
        assert np.array_equal(values[:, METHODS.index(name)], previous['values'][:, previous['methods'].tolist().index(name)])
    summary = read(ROOT / 'summary.json')
    assert [r['method'] for r in summary['reports']] == list(METHODS)
    def transition(before, after):
        return {'rescued': int((~before & after).sum()), 'new_losses': int((before & ~after).sum()),
                'net': int(after.sum()-before.sum())}
    for j, report in enumerate(summary['reports']):
        assert report['queries'] == 3452
        close([report[k+'_pct'] for k in METRICS], 100*values[0, j], 'point reports')
        assert report['vs_fixed'] == transition(indicators['fixed'][:, 0], indicators[report['method']][:, 0])
    expected_ci = {}
    for baseline in ('QL5', 'G12', 'fixed'):
        a, b = METHODS.index('G_same12'), METHODS.index(baseline)
        name = 'G_same12-'+baseline
        assert summary['transitions'][name] == transition(indicators[baseline][:, 0], indicators['G_same12'][:, 0])
        for j, metric in enumerate(METRICS[:2]):
            difference = 100*(values[:, a, j]-values[:, b, j]); bounds = np.quantile(difference[1:], [.025, .975])
            expected_ci[name, metric] = [difference[0], *bounds]
    assert len(summary['comparisons']) == 6
    assert {(r['comparison'], r['metric']) for r in summary['comparisons']} == set(expected_ci)
    for r in summary['comparisons']:
        close([r[k] for k in ('difference_pp', 'ci_low_pp', 'ci_high_pp')], expected_ci[r['comparison'], r['metric']], 'paired interval')
        assert r['draws'] == 1000 and r['primary'] == (r['comparison'] == 'G_same12-QL5' and r['metric'] == 'P1')
    intervals = csv_rows(ROOT / 'paired_bootstrap.csv')
    assert len(intervals) == 6
    assert {(r['comparison'], r['metric']) for r in intervals} == set(expected_ci)
    for r in intervals:
        close([float(r[k]) for k in ('difference_pp', 'ci_low_pp', 'ci_high_pp')], expected_ci[r['comparison'], r['metric']], 'interval CSV')
    rows = csv_rows(ROOT / 'per_query.csv.gz'); assert len(rows) == 4*3452
    index = {q: i for i, q in enumerate(meta['qids'])}; seen = set()
    for r in rows:
        key = r['method'], r['qid']; assert key not in seen; seen.add(key)
        i, name = index[r['qid']], r['method']
        assert r['top_document'] == meta['ranked_pools'][i][point[name][i, 0]]
        assert [int(r[k]) for k in METRICS] == indicators[name][i].astype(int).tolist()
    assert summary['old_G_QL5_all1001_draws_exact_reproduction'] is True and summary['new_neural_pairs'] == 0
    assert summary['source_model_sealed_before_A_score'] is True
    assert summary['transfer_models_sha256'] == sha(ROOT / 'transfer_models.json')
    assert summary['predictions_sealed_sha256'] == sha(ROOT / 'predictions_sealed.json')
    close(summary['point_lambdas'], [arrays(ROOT / 'fitted' / f'fold_{f}.npz')['lambda'][0] for f in range(5)], 'point lambda summary')
    return summary, positive, negative


def provenance(expected: str) -> dict:
    assert sha(ROOT / 'frozen_manifest.json') == expected
    manifest = read(ROOT / 'frozen_manifest.json')
    assert 'verify_matched.py' not in manifest['code']
    for name, digest in manifest['code'].items():
        assert sha(ROOT / name) == digest
    count = 0
    for group in ('feature_sources', 'kernel_sources', 'reuse_sources', 'transfer_sources', 'scoring_sources', 'view_manifests'):
        for name, digest in manifest[group].items():
            assert sha(Path(name)) == digest, name
            count += 1
    assert manifest['config']['primary'] == 'G_same12-QL5:P1' and manifest['config']['new_neural_pairs'] == 0
    assert manifest['config']['bootstrap_draws'] == 1000 and manifest['config']['CPU_threads'] == 1
    assert manifest['fixtures_sha256'] == sha(ROOT / 'fixtures.json')
    assert manifest['timing_sha256'] == sha(ROOT / 'synthetic_timing.json')
    assert read(ROOT / 'fixtures.json')['status'] == 'PASS'
    feature_seal = read(ROOT / 'features_sealed.json')
    assert feature_seal['features_sha256'] == sha(ROOT / 'features.npz') and feature_seal['pairs_aligned'] == 69040
    assert feature_seal['references_opened'] is False and feature_seal['head_row_identity_unchanged'] is True
    assert feature_seal['Q_columns_bitwise_unchanged'] == [1, 3, 11] and feature_seal['text_columns_bitwise_unchanged'] == [4, 5, 6, 7]
    sealed = read(ROOT / 'predictions_sealed.json')
    assert sealed['frozen_manifest_sha256'] == expected and sealed['scoring_references_opened'] is False
    assert sealed['reused_sources'] == manifest['reuse_sources']
    expected_files = {f'fitted/fold_{f}.{suffix}' for f in range(5) for suffix in ('npz', 'json')} | {
        'features.npz', 'features_sealed.json', 'transfer_models.json', 'transfer_models_sealed.json'}
    assert set(sealed['files']) == expected_files
    for name, digest in sealed['files'].items():
        assert sha(ROOT / name) == digest
    for f in range(5):
        receipt = read(ROOT / 'fitted' / f'fold_{f}.json')
        assert receipt['heldout_labels_read'] is False
        assert receipt['sha256'] == sha(ROOT / 'fitted' / f'fold_{f}.npz')
        assert receipt['training_label_sha256'] == sha(POST / 'prepared' / f'labels_{f}.npz')
    finished = read(ROOT / 'production_finished.json')
    assert finished['exit_code'] == 0 and finished['task_retries'] == 0 and finished['new_neural_pairs'] == 0
    launcher = read(ROOT / 'launcher_receipt.json')
    assert launcher['exit_code'] == 0 and launcher['timeout_seconds'] == 14400 and launcher['timeout_or_kill'] is False
    return {'frozen_code_files': len(manifest['code']), 'phase_source_entries': count,
            'sealed_artifacts': len(sealed['files']), 'production_wall_seconds': finished['wall_seconds']}


def fixtures() -> dict:
    canonical = np.tile(np.arange(20), (2, 1))
    # Fixed row identity differs from the new full-input score rank.
    mini = np.tile(np.arange(20.), (2, 1)); mini[1] = 0.
    qwen = np.tile(np.arange(20., 0., -1), (2, 1))
    old_x = np.zeros((2, 20, 12))
    old_x[:, :, 1] = zscore(qwen)
    old_x[:, :, 3] = np.arange(1, 21)/20
    old_x[:, :, 11] = old_x[:, :, 1]*(old_x[:, 0, 1]-old_x[:, 1, 1])[:, None]
    old_x[:, :, 4:8] = np.arange(4)
    new_x = independent_features(mini, qwen, old_x, canonical)
    close(new_x[0, :, 2], np.arange(20, 0, -1)/20, 'fixture changed MiniLM rank')
    close(new_x[1, :, 2], np.arange(1, 21)/20, 'fixture tied MiniLM canonical rank')
    assert np.array_equal(new_x[:, :, 4:8], old_x[:, :, 4:8])
    return {'new_rank_independent_of_old_row_position': True,
            'tied_MiniLM_canonical_rank': True, 'text_block_exactly_unchanged': True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--production-complete', action='store_true', required=True)
    parser.add_argument('--freeze-sha256', required=True)
    args = parser.parse_args()
    assert args.production_complete and str(ROOT).startswith('/path/to/remote/lw_experiments/runs/')
    started = time.monotonic(); source_report = provenance(args.freeze_sha256)
    fixture_report = fixtures(); x, old, meta, plan = source_inputs()
    point, tops, masks = verify_folds(x, old, plan)
    summary, positive, negative = verify_scores(meta, point, tops, masks, plan)
    transfer = verify_transfer(x, old, plan, positive, negative)
    assert source_report == provenance(args.freeze_sha256)
    certificate = {'status': 'PASS_INDEPENDENT_INPUT_MATCHED_FUSION_VERIFICATION',
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'wall_seconds': time.monotonic()-started,
        'frozen_manifest_sha256': args.freeze_sha256, 'verifier_sha256': sha(Path(__file__)),
        'provenance': source_report, 'fixtures': fixture_report, 'numeric': NUMERIC,
        'queries': 3452, 'components': 3239, 'draws': 1000, 'methods': list(METHODS),
        'point_and_bootstrap_metric_cells': 1001*4*4, 'per_query_rows': 3452*4, 'paired_intervals': 6,
        'source_transfer_models': transfer, 'reports': summary['reports'], 'comparisons': summary['comparisons'],
        'transitions': summary['transitions'], 'limits': ['Previously exposed development data.',
            'Old G12/QL5 fits and model scores reuse pinned prior independent certificates.',
            'Source transfer models use only ExcluIR; they are not used to produce A OOF predictions.',
            'No neural inference or IFIR evaluation is repeated by this CPU auditor.']}
    with (ROOT / 'independent_verification.json').open('x') as stream:
        json.dump(certificate, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(certificate, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
