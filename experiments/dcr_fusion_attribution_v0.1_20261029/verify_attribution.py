"""Independent CPU reconstruction for the frozen attribution diagnostic.

No production mathematics/feature/scoring module is imported. Source features,
groups, reference semantics and the previously independently verified G fits
are reused as pinned protocol inputs. New feature-arm fits are rebuilt from
each fold's own training-label view, including all 1000 inner selections.
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

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parent
POST = ROOT.parent / 'dcr_post_fusion_v0.1_20261029'
GATE = ROOT.parent / 'dcr_gate_mechanism_v0.1_20261029'
GROUPS = ROOT.parent / 'dcr_constraint_router_v0.1_20261029/prepared/groups.json'
LAMBDAS = (.001, .01, .1, 1.)
COLUMNS = {'SR4': [0, 1, 2, 3], 'SR8': [0, 1, 2, 3, 8, 9, 10, 11],
           'QL5': [1, 4, 5, 6, 7]}
METHODS = ['fixed', 'RRF', 'SR4', 'SR8', 'QL5', 'G12', 'G12_guard', 'SR8_guard']
METRICS = ['P1', 'N1', 'U1', 'R10']
CONTRASTS = [('G12', 'SR8'), ('SR8', 'SR4'), ('SR8', 'fixed'), ('QL5', 'fixed'),
             ('G12_guard', 'G12'), ('SR8_guard', 'SR8'), ('RRF', 'fixed'),
             ('G12', 'QL5'), ('G12', 'fixed')]
NUMERIC = {'comparisons': 0, 'maximum_absolute_difference': 0., 'new_outer_models': 0,
           'new_inner_candidate_solves': 0, 'exact_order_matrices': 0, 'reused_G_models': 0}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def csv_rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', newline='') as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, label, tolerance=1e-10):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), label
    error = float(np.max(np.abs(a-b))) if a.size else 0.
    NUMERIC['comparisons'] += int(a.size)
    NUMERIC['maximum_absolute_difference'] = max(NUMERIC['maximum_absolute_difference'], error)
    assert error <= tolerance, (label, error, tolerance)


def ranks(score, canonical):
    # Sort by score then canonical document ID, independently of production argsort.
    id_rank = np.empty_like(canonical)
    np.put_along_axis(id_rank, canonical, np.broadcast_to(np.arange(20), canonical.shape), axis=1)
    return np.lexsort((id_rank, -score), axis=1).astype(np.uint8)


def wins(score, canonical, positive):
    arranged = np.take_along_axis(score, canonical, axis=1)
    top = canonical[np.arange(len(score)), np.argmax(arranged, axis=1)]
    return top == positive


def statistics(x, positive, negative, weights):
    eligible = (positive >= 0) & (negative >= 0) & (weights > 0)
    index = np.flatnonzero(eligible)
    assert len(index), 'No known endpoint pair in this training view'
    difference = x[index, positive[index]]-x[index, negative[index]]
    multiplicity = weights[index].astype(float)
    count = int(multiplicity.sum())
    scale = np.sqrt(np.sum(multiplicity[:, None]*difference*difference, axis=0)/count)
    scale[scale == 0] = 1.
    normalized = difference/scale
    return scale, normalized.T @ (multiplicity[:, None]*normalized), normalized.T @ multiplicity, count


def solve(stats, regularization):
    scale, gram, rhs, count = stats
    matrix = gram+count*regularization*np.eye(len(scale))
    coefficient = np.linalg.solve(matrix, rhs)
    assert np.max(np.abs(matrix @ coefficient-rhs)) < 1e-8*max(1., np.max(np.abs(rhs)))
    return scale, coefficient, count


def predict(x, model):
    return (x/model[0]) @ model[1]


def choose(x, canonical, positive, negative, folds, outer, weights):
    counts = [0]*4
    for heldout in range(5):
        if heldout == outer:
            continue
        train_weights = weights*((folds != outer) & (folds != heldout))
        stats = statistics(x, positive, negative, train_weights)
        check = np.flatnonzero((folds == heldout) & (weights > 0))
        for j, regularization in enumerate(LAMBDAS):
            model = solve(stats, regularization)
            counts[j] += int(wins(predict(x[check], model), canonical[check], positive[check]) @ weights[check])
            NUMERIC['new_inner_candidate_solves'] += 1
    # Ascending grid and first maximum give the prescribed exact small-lambda tie.
    return LAMBDAS[counts.index(max(counts))], counts


def guard(order, agreement, common_document):
    protected = order.copy()
    for i in np.flatnonzero(agreement):
        first = int(common_document[i])
        protected[i] = [first]+[int(d) for d in order[i] if d != first]
    return protected


def top10_mask(order):
    return np.sum(np.left_shift(np.uint32(1), order[:, :10].astype(np.uint32)), axis=1).astype(np.uint32)


def fixtures():
    # Guard must treat common positive/negative/unjudged candidates identically.
    order = np.tile(np.arange(19, -1, -1), (3, 1))
    common = np.array([0, 1, 2])
    protected = guard(order, np.array([True, True, False]), common)
    assert protected[0].tolist() == [0]+list(range(19, 0, -1))
    assert protected[1].tolist() == [1]+[i for i in range(19, -1, -1) if i != 1]
    assert np.array_equal(protected[2], order[2])
    canonical = np.array([np.arange(19, -1, -1)])
    assert np.array_equal(ranks(np.zeros((1, 20)), canonical), canonical)
    # Unknown candidate does not enter the pair loss; duplicate weights are exact counts.
    x = np.array([[[2., 0.], [0., 1.], [1e6, -1e6]]])
    a = solve(statistics(x, np.array([0]), np.array([1]), np.array([3])), .1)
    b = solve(statistics(np.repeat(x, 3, axis=0), np.zeros(3, dtype=int), np.ones(3, dtype=int), np.ones(3, dtype=int)), .1)
    close(a[0], b[0], 'fixture repeat scale'); close(a[1], b[1], 'fixture repeat coefficient')
    x[0, 2] = [-1e30, 1e30]
    c = solve(statistics(x, np.array([0]), np.array([1]), np.array([3])), .1)
    close(a[1], c[1], 'fixture unjudged exclusion')
    return {'canonical_ties': True, 'stable_guard_no_labels': True,
            'weighted_pair_equivalence': True, 'unknown_candidate_excluded_from_loss': True}


def source_inputs():
    # Prior independent certificate covers original feature construction and G fits.
    expected = '0c3d920df1cf5e180bb402bd8b7decf4a35fece2982ba63ad07276493ca60336'
    assert sha(POST / 'independent_verification.json') == expected
    assert read(POST / 'independent_verification.json')['status'] == 'PASS_INDEPENDENT_POST_FUSION_VERIFICATION'
    source = load(POST / 'features.npz')
    meta = read(POST / 'head_and_tail.json')
    qids, pools = meta['qids'], meta['ranked_pools']
    assert len(qids) == len(pools) == 3452 and qids == sorted(qids) and len(set(qids)) == 3452
    assert source['X'].shape == (3452, 20, 18)
    assert all(len(p) == len(set(p)) == 100 for p in pools)
    canonical = np.array([sorted(range(20), key=lambda j: p[j]) for p in pools])
    assert np.array_equal(canonical, source['canonical'])
    plan = load(POST / 'prepared/plan.npz'); groups = read(GROUPS)
    assert groups['qids'] == qids and len(groups['components']) == 3239
    assert np.array_equal(plan['fold'], [groups['query_fold'][q] for q in qids])
    rng = random.Random(20261029); index = {q: i for i, q in enumerate(qids)}
    members = {c: [index[q] for q in qq] for c, qq in groups['components'].items()}
    for draw in range(1000):
        weights = np.zeros(3452, dtype=np.int16)
        for fold in range(5):
            components = groups['fold_component_IDs'][str(fold)]
            assert components == sorted(components)
            for _ in components:
                chosen = components[rng.randrange(len(components))]
                weights[members[chosen]] += 1
        assert np.array_equal(weights, plan['weights'][draw]), ('component plan', draw)
    return source, meta, plan


def reconstruct(source, plan):
    x, canonical, fold = source['X'], source['canonical'], plan['fold']
    fixed = ranks(source['raw_scores'][:, :, 1], canonical)
    qrank = np.empty(fixed.shape, dtype=np.int16)
    np.put_along_axis(qrank, fixed, np.broadcast_to(np.arange(1, 21), fixed.shape), axis=1)
    rrf = ranks(1/(60+np.arange(1, 21))+1/(60+qrank), canonical)
    agreement = fixed[:, 0] == 0  # X rows are in canonical-tie MiniLM order.
    point = {name: np.empty((3452, 20), dtype=np.uint8) for name in METHODS}
    point['fixed'], point['RRF'] = fixed, rrf
    tops = {name: np.empty((1000, 3452), dtype=np.uint8) for name in METHODS}
    membership = {name: np.empty((1000, 3452), dtype=np.uint32) for name in METHODS}
    tops['fixed'][:] = fixed[:, 0]; tops['RRF'][:] = rrf[:, 0]
    membership['fixed'][:] = top10_mask(fixed); membership['RRF'][:] = top10_mask(rrf)
    point_models = {}
    for outer in range(5):
        view = read(POST / 'prepared' / f'view_{outer}.json')
        for path, expected in view['allowed_inputs'].items():
            assert sha(Path(path)) == expected, path
        label = load(Path(view['labels_path']))
        train, test = np.flatnonzero(fold != outer), np.flatnonzero(fold == outer)
        assert label['train_indices'].tolist() == train.tolist() and view['test_indices'] == test.tolist()
        positive, negative = np.full(3452, -2), np.full(3452, -2)
        positive[train], negative[train] = label['positive'], label['negative']
        saved = load(ROOT / 'fitted' / f'fold_{outer}.npz')
        assert saved['test_indices'].tolist() == test.tolist()
        with np.load(POST / 'fitted' / f'fold_{outer}.npz', allow_pickle=False) as old:
            old_g = {k: old['G_'+k] for k in ('scale', 'coef', 'order', 'point_scores')}
            assert np.array_equal(old['test_indices'], test)
        point_models[outer] = old_g['scale'][0], old_g['coef'][0]
        selected_x = {name: x[:, :, columns].copy() for name, columns in COLUMNS.items()}
        for iteration in range(1001):
            weights = np.ones(3452, dtype=np.int32) if iteration == 0 else plan['weights'][iteration-1].astype(np.int32)
            for name, arm_x in selected_x.items():
                regularization, counts = choose(arm_x, canonical, positive, negative, fold, outer, weights)
                assert regularization == saved[name+'_lambda'][iteration]
                assert np.array_equal(counts, saved[name+'_inner_correct_counts'][iteration])
                model = solve(statistics(arm_x, positive, negative, weights*(fold != outer)), regularization)
                close(saved[name+'_scale'][iteration], model[0], (outer, iteration, name, 'RMS'))
                close(saved[name+'_coef'][iteration], model[1], (outer, iteration, name, 'coefficient'))
                assert saved[name+'_pair_instances'][iteration] == model[2]
                score = predict(arm_x[test], model)
                order = ranks(score, canonical[test])
                assert np.array_equal(order, saved[name+'_order'][iteration]), (outer, iteration, name, 'order')
                NUMERIC['exact_order_matrices'] += 1; NUMERIC['new_outer_models'] += 1
                if iteration == 0:
                    close(saved[name+'_point_scores'], score, (outer, name, 'point scores'))
                    point[name][test] = order
                else:
                    tops[name][iteration-1, test] = order[:, 0]
                    membership[name][iteration-1, test] = top10_mask(order)
                if name == 'SR8':
                    protected = guard(order, agreement[test], fixed[test, 0])
                    if iteration == 0:
                        point['SR8_guard'][test] = protected
                    else:
                        tops['SR8_guard'][iteration-1, test] = protected[:, 0]
                        membership['SR8_guard'][iteration-1, test] = top10_mask(protected)
            g_scores = (x[test, :, :12]/old_g['scale'][iteration]) @ old_g['coef'][iteration]
            g_order = ranks(g_scores, canonical[test])
            assert np.array_equal(g_order, old_g['order'][iteration]), ('old G internally consistent', outer, iteration)
            NUMERIC['reused_G_models'] += 1
            protected = guard(g_order, agreement[test], fixed[test, 0])
            if iteration == 0:
                close(old_g['point_scores'], g_scores, ('old G point scores', outer))
                point['G12'][test], point['G12_guard'][test] = g_order, protected
            else:
                tops['G12'][iteration-1, test] = g_order[:, 0]
                tops['G12_guard'][iteration-1, test] = protected[:, 0]
                membership['G12'][iteration-1, test] = top10_mask(g_order)
                membership['G12_guard'][iteration-1, test] = top10_mask(protected)
            if iteration % 200 == 0:
                print(json.dumps({'fold': outer, 'iteration': iteration, 'phase': 'independent models'}), flush=True)
    saved_point = load(ROOT / 'point_orders.npz')
    assert saved_point['methods'].tolist() == METHODS
    assert np.array_equal(saved_point['orders'], np.array([point[n] for n in METHODS]))
    return point, tops, membership, point_models, agreement


def endpoint_indices(meta, references):
    p, n = [], []
    for q, pool in zip(meta['qids'], meta['ranked_pools']):
        p.append(pool.index(references[q]['positive_id']) if references[q]['positive_id'] in pool else -1)
        n.append(pool.index(references[q]['negative_id']) if references[q]['negative_id'] in pool else -1)
    return np.array(p), np.array(n)


def verify_label_views(plan, positive, negative):
    phead = np.where((positive >= 0) & (positive < 20), positive, -1)
    nhead = np.where((negative >= 0) & (negative < 20), negative, -1)
    assert int(((phead >= 0) & (nhead >= 0)).sum()) == 2901
    for outer in range(5):
        labels = load(POST / 'prepared' / f'labels_{outer}.npz')
        train = np.flatnonzero(plan['fold'] != outer)
        assert np.array_equal(labels['train_indices'], train)
        assert np.array_equal(labels['positive'], phead[train])
        assert np.array_equal(labels['negative'], nhead[train])


def scores(point, tops, membership, plan, positive, negative):
    result = np.empty((1001, 8, 4))
    positive_bit = np.where((positive >= 0) & (positive < 20),
        np.left_shift(np.uint32(1), np.clip(positive, 0, 19).astype(np.uint32)), np.uint32(0))
    for i, method in enumerate(METHODS):
        order = point[method]
        top = order[:, 0]
        p, n = top == positive, top == negative
        point_recall = np.any(order[:, :10] == positive[:, None], axis=1)
        result[0, i] = np.mean([p, n, ~(p | n), point_recall], axis=1)
        for draw in range(1000):
            weight = plan['weights'][draw]
            first = tops[method][draw]
            bp, bn = first == positive, first == negative
            result[draw+1, i, :3] = np.array([weight @ bp, weight @ bn, weight @ ~(bp | bn)])/weight.sum()
            result[draw+1, i, 3] = (weight @ ((membership[method][draw] & positive_bit) != 0))/weight.sum()
    return result


def verify_scoring(point, tops, membership, plan, positive, negative):
    reconstructed = scores(point, tops, membership, plan, positive, negative)
    artifact = load(ROOT / 'metrics_draws.npz')
    assert artifact['methods'].tolist() == METHODS and artifact['metrics'].tolist() == METRICS
    saved = artifact['values']
    assert saved.shape == reconstructed.shape == (1001, 8, 4)
    close(saved[0], reconstructed[0], 'all point metrics')
    close(saved[1:], reconstructed[1:], 'all bootstrap P/N/U/R10')
    summary = read(ROOT / 'summary.json')
    assert [r['method'] for r in summary['reports']] == METHODS
    for i, r in enumerate(summary['reports']):
        assert r['queries'] == 3452
        close([r[k+'_pct'] for k in METRICS], 100*reconstructed[0, i], 'report metrics')
        baseline, current = point['fixed'][:, 0] == positive, point[r['method']][:, 0] == positive
        rescued, lost = int((current & ~baseline).sum()), int((baseline & ~current).sum())
        assert r['rescued_vs_fixed'] == rescued and r['new_losses_vs_fixed'] == lost
        assert r['net_rescued_vs_fixed'] == rescued-lost
    comparisons = summary['comparisons']
    assert len(comparisons) == 18
    lookup = {(r['comparison'], r['metric']): r for r in comparisons}
    assert len(lookup) == 18
    for a, b in CONTRASTS:
        for j, metric in enumerate(METRICS[:2]):
            r = lookup[a+'-'+b, metric]
            samples = 100*(saved[1:, METHODS.index(a), j]-saved[1:, METHODS.index(b), j])
            expected = [100*(saved[0, METHODS.index(a), j]-saved[0, METHODS.index(b), j]), *np.quantile(samples, [.025, .975])]
            close([r[k] for k in ('difference_pp', 'ci_low_pp', 'ci_high_pp')], expected, 'paired CI')
            assert r['draws'] == 1000 and r['primary'] == (a == 'G12' and b == 'SR8' and metric == 'P1')
    csv_intervals = csv_rows(ROOT / 'paired_bootstrap.csv')
    assert len(csv_intervals) == 18
    for row, r in zip(csv_intervals, comparisons):
        assert row['comparison'] == r['comparison'] and row['metric'] == r['metric']
        assert row['primary'] == str(r['primary']) and int(row['draws']) == 1000
        close([float(row[k]) for k in ('difference_pp', 'ci_low_pp', 'ci_high_pp')],
              [r[k] for k in ('difference_pp', 'ci_low_pp', 'ci_high_pp')], 'interval CSV')
    for name in COLUMNS:
        close(summary['point_lambdas'][name],
            [load(ROOT / 'fitted' / f'fold_{f}.npz')[name+'_lambda'][0] for f in range(5)], 'reported point lambdas')
    assert summary['new_neural_pairs'] == 0 and summary['reused_Qwen_pairs'] == 69040
    assert summary['head_size'] == 20 and summary['tail_size'] == 80
    assert summary['diagnostic_not_confirmatory'] is True
    assert summary['predictions_sealed_sha256'] == sha(ROOT / 'predictions_sealed.json')
    return summary


def contribution_reference(source, meta, plan, point, models, agreement, positive):
    expected = {}
    common_correct = agreement & (point['fixed'][:, 0] == positive)
    for i, q in enumerate(meta['qids']):
        winner, fixed = int(point['G12'][i, 0]), int(point['fixed'][i, 0])
        if winner == fixed:
            continue
        scale, coef = models[int(plan['fold'][i])]
        difference = source['X'][i, winner, :12]-source['X'][i, fixed, :12]
        contribution = difference/scale*coef
        total = float(contribution.sum())
        direct = float((source['X'][i, winner, :12]/scale) @ coef-(source['X'][i, fixed, :12]/scale) @ coef)
        close(total, direct, ('contribution sum', q))
        assert total >= -1e-10
        expected[q] = {'columns': contribution, 'total_diff': total,
            'sr4_diff': float(contribution[:4].sum()), 'interaction_diff': float(contribution[8:12].sum()),
            'text_diff': float(contribution[4:8].sum()), 'shared_top': bool(agreement[i]),
            'common_correct': bool(common_correct[i]), 'G_winner_index': winner, 'fixed_winner_index': fixed,
            'no_text_pair_diff': total-float(contribution[4:8].sum())}
    losses = common_correct & (point['G12'][:, 0] != positive)
    assert int(losses.sum()) == 93, 'Old independently verified result consistency only'
    margins = [r['no_text_pair_diff'] for r in expected.values() if r['common_correct']]
    assert len(margins) == 93
    return expected, {'old_common_correct_losses': len(margins),
        'no_text_two_candidate_difference_positive': sum(v > 0 for v in margins),
        'no_text_two_candidate_difference_zero': sum(v == 0 for v in margins),
        'no_text_two_candidate_difference_negative': sum(v < 0 for v in margins)}


def consensus_reference(point, agreement, positive, negative):
    common = point['fixed'][:, 0]
    kinds = {'positive': agreement & (common == positive),
             'negative': agreement & (common == negative),
             'unjudged': agreement & (common != positive) & (common != negative)}
    result = {'shared_top_queries': int(agreement.sum()),
        'shared_top_categories': {k: int(v.sum()) for k, v in kinds.items()}, 'guards': {}}
    for name in ('G12', 'SR8'):
        before, after = point[name][:, 0], point[name+'_guard'][:, 0]
        assert np.array_equal(point[name][~agreement], point[name+'_guard'][~agreement])
        assert np.array_equal(after[agreement], common[agreement])
        bh, ah = before == positive, after == positive
        rescued, lost = ~bh & ah, bh & ~ah
        result['guards'][name] = {'rescued': int(rescued.sum()), 'new_losses': int(lost.sum()),
            'net_correct': int(ah.sum()-bh.sum()),
            'by_common_category': {kind: {'rescued': int((rescued & mask).sum()),
                'new_losses': int((lost & mask).sum())} for kind, mask in kinds.items()}}
    return result


def verify_contributions(source, meta, plan, point, models, agreement, positive, summary):
    expected, direction = contribution_reference(source, meta, plan, point, models, agreement, positive)
    rows = csv_rows(ROOT / 'attribution.csv.gz')
    assert len(rows) == len(expected)
    assert [r['qid'] for r in rows] == list(expected)
    index = {q: i for i, q in enumerate(meta['qids'])}
    for row in rows:
        q = row['qid']; r = expected[q]; i = index[q]
        assert int(row['fold']) == int(plan['fold'][i])
        assert row['fixed_winner'] == meta['ranked_pools'][i][r['fixed_winner_index']]
        assert row['G_winner'] == meta['ranked_pools'][i][r['G_winner_index']]
        assert row['shared_top'] == str(r['shared_top']) and row['common_correct'] == str(r['common_correct'])
        close([float(row[f'c{j}']) for j in range(12)], r['columns'], ('column contributions', q))
        close([float(row[k]) for k in ('total_diff', 'sr4_diff', 'interaction_diff', 'text_diff')],
              [r[k] for k in ('total_diff', 'sr4_diff', 'interaction_diff', 'text_diff')], ('block contributions', q))
    cc = [r for r in expected.values() if r['common_correct']]
    assert summary['attribution_rows'] == len(rows) and summary['common_correct_changed'] == len(cc) == 93
    assert summary['common_correct_SR8_partial_positive'] == sum(r['sr4_diff']+r['interaction_diff'] > 0 for r in cc)
    assert summary['common_correct_SR4_partial_positive'] == sum(r['sr4_diff'] > 0 for r in cc)
    return {'changed_queries_expected': len(expected), 'rows': len(rows), **direction}


def verify_consensus(point, agreement, positive, negative):
    expected = consensus_reference(point, agreement, positive, negative)
    saved = read(ROOT / 'consensus.json')
    assert saved['shared_total'] == expected['shared_top_queries']
    assert saved['shared_reference'] == {short: expected['shared_top_categories'][kind]
        for short, kind in [('P', 'positive'), ('N', 'negative'), ('U', 'unjudged')]}
    for name in ('G12', 'SR8'):
        value = saved['guards'][name]; r = expected['guards'][name]
        assert value['rescued'] == r['rescued'] and value['new_losses'] == r['new_losses']
        assert value['net'] == r['net_correct']
        assert value['shared_correct_restored'] == r['by_common_category']['positive']['rescued']
        assert value['shared_wrong_rescues_revoked'] == sum(r['by_common_category'][k]['new_losses'] for k in ('negative', 'unjudged'))
        assert value['changed_top1'] == int((point[name][:, 0] != point[name+'_guard'][:, 0]).sum())
    assert saved['original_G_shared_correct_losses'] == 93 and saved['old_93_consistency_check'] is True
    return expected


def verify_per_query(meta, point, agreement, positive, negative):
    rows = csv_rows(ROOT / 'per_query.csv.gz')
    assert len(rows) == 3452*8
    index = {q: i for i, q in enumerate(meta['qids'])}
    seen = set()
    for r in rows:
        key = r['qid'], r['method']; assert key not in seen; seen.add(key)
        i, method = index[r['qid']], r['method']; order = point[method][i]
        assert r['top_document'] == meta['ranked_pools'][i][order[0]]
        assert r['shared_top'] == str(bool(agreement[i]))
        p, n = bool(order[0] == positive[i]), bool(order[0] == negative[i])
        expected = [p, n, not (p or n), bool(positive[i] in order[:10])]
        assert [int(r[k]) for k in METRICS] == [int(v) for v in expected]
    return len(rows)


def provenance(expected_freeze):
    assert sha(ROOT / 'frozen_manifest.json') == expected_freeze
    manifest = read(ROOT / 'frozen_manifest.json')
    assert manifest['model_calls'] == 0 and manifest['IFIR_inputs_used'] is False
    assert 'verify_attribution.py' not in manifest['code']
    for name, digest in manifest['code'].items():
        assert sha(ROOT / name) == digest, name
    for path, digest in manifest['sources'].items():
        assert sha(Path(path)) == digest, path
    assert sha(POST / 'frozen_manifest.json') == '65e1a77a41ce26652e4b7faa8365e17b7e4425a36035e713605ba702ea091ea7'
    seal = read(ROOT / 'predictions_sealed.json')
    assert seal['frozen_manifest_sha256'] == expected_freeze
    assert seal['scoring_references_opened'] is False
    assert seal['reused_G_routes_sha256'] == sha(POST / 'routes_sealed.json')
    assert set(seal['files']) == {f'fitted/fold_{f}.{suffix}' for f in range(5) for suffix in ('npz', 'json')}
    for name, digest in seal['files'].items():
        assert sha(ROOT / name) == digest
    for f in range(5):
        receipt = read(ROOT / 'fitted' / f'fold_{f}.json')
        assert receipt['heldout_labels_read'] is False
        assert receipt['file_sha256'] == sha(ROOT / 'fitted' / f'fold_{f}.npz')
    finished = read(ROOT / 'production_finished.json')
    assert finished['exit_code'] == 0 and finished['task_retries'] == 0
    assert finished['foundation_model_calls'] == 0 and finished['GPU_used'] is False
    assert finished['sources_before_after'] == 'matched'
    return {'frozen_sources': len(manifest['sources']), 'frozen_code_files': len(manifest['code']),
            'sealed_fit_artifacts': len(seal['files']), 'production_wall_seconds': finished['wall_seconds']}


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--production-complete', action='store_true', required=True)
    cli.add_argument('--freeze-sha256', required=True)
    args = cli.parse_args()
    assert args.production_complete and str(ROOT).startswith('/path/to/remote/lw_experiments/runs/')
    started = time.monotonic()
    source_report = provenance(args.freeze_sha256)
    fixture_report = fixtures()
    source, meta, plan = source_inputs()
    point, tops, membership, models, agreement = reconstruct(source, plan)
    references = read(GATE / 'external_inputs/references.json')
    positive, negative = endpoint_indices(meta, references)
    verify_label_views(plan, positive, negative)
    summary = verify_scoring(point, tops, membership, plan, positive, negative)
    contributions = verify_contributions(source, meta, plan, point, models, agreement, positive, summary)
    consensus = verify_consensus(point, agreement, positive, negative)
    per_query = verify_per_query(meta, point, agreement, positive, negative)
    # End-of-audit hashes catch any concurrent change to frozen inputs/outputs.
    assert source_report == provenance(args.freeze_sha256)
    certificate = {'status': 'PASS_INDEPENDENT_FUSION_ATTRIBUTION_VERIFICATION',
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'wall_seconds': time.monotonic()-started,
        'frozen_manifest_sha256': args.freeze_sha256, 'verifier_sha256': sha(Path(__file__)),
        'provenance': source_report, 'fixtures': fixture_report, 'numeric': NUMERIC,
        'queries': 3452, 'components': 3239, 'bootstrap_draws': 1000, 'methods': METHODS,
        'point_and_draw_metric_cells': 1001*8*4, 'per_query_rows': per_query, 'paired_intervals': 18,
        'consensus': consensus, 'attribution': contributions, 'reports': summary['reports'],
        'comparisons': summary['comparisons'],
        'limits': ['All data are previously exposed development material.',
            'Original features and G fits reuse the pinned prior independent certificate.',
            'Guard is a label-free postprocessing rule; column margins are arithmetic, not causal effects.',
            'No neural inference is rerun by this CPU auditor.']}
    with (ROOT / 'independent_verification.json').open('x') as stream:
        json.dump(certificate, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
    print(json.dumps(certificate, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
