"""Open full references only after prediction sealing; preserve all comparisons."""
import csv
import gzip
import time
import numpy as np
from paths import ROOT, PREV, REF, read, write, sha, check_code, check_sources
from math_core import METHODS, METRICS, COMPARISONS, rank, guard


def indicators(order, positive, negative):
    p = order[:, 0] == positive
    n = order[:, 0] == negative
    return np.stack((p, n, ~(p | n), (order[:, :10] == positive[:, None]).any(axis=1)), axis=1)


def transition(before, after):
    rescued = int(np.sum(~before & after))
    lost = int(np.sum(before & ~after))
    return {'rescued': rescued, 'new_losses': lost, 'net': rescued-lost}


def save_csv(path, rows):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'xt', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    started = time.monotonic()
    manifest = check_code()
    check_sources(manifest)
    seal = read(ROOT/'predictions_sealed.json')
    assert seal['frozen_manifest_sha256'] == sha(ROOT/'frozen_manifest.json')
    assert not seal['scoring_references_opened']
    for name, digest in seal['files'].items():
        assert sha(ROOT/name) == digest
    assert seal['reused_G_routes_sha256'] == sha(PREV/'routes_sealed.json')
    meta = read(PREV/'head_and_tail.json')
    qids, pools = meta['qids'], meta['ranked_pools']
    references = read(REF)
    positive, negative = [], []
    for qid, pool in zip(qids, pools):
        p, n = references[qid]['positive_id'], references[qid]['negative_id']
        positive.append(pool[:20].index(p) if p in pool[:20] else -1)
        negative.append(pool[:20].index(n) if n in pool[:20] else -1)
    positive, negative = np.asarray(positive), np.asarray(negative)
    with np.load(PREV/'features.npz', allow_pickle=False) as z:
        x, canonical, raw = z['X'][:, :, :12], z['canonical'], z['raw_scores']
    with np.load(PREV/'prepared/plan.npz', allow_pickle=False) as z:
        plan, folds = z['weights'], z['fold']
    fitted = [{k: z[k] for k in z.files} for z in
              [np.load(ROOT/'fitted'/f'fold_{f}.npz', allow_pickle=False) for f in range(5)]]
    old = []
    for fold in range(5):
        with np.load(PREV/'fitted'/f'fold_{fold}.npz', allow_pickle=False) as z:
            old.append({k: z[k] for k in z.files if k.startswith('G_') or k == 'test_indices'})
        assert np.array_equal(old[-1]['test_indices'], fitted[fold]['test_indices'])
    fixed = rank(raw[:, :, 1], canonical)
    rrf = rank(1/(60+x[:, :, 2]*20) + 1/(60+x[:, :, 3]*20), canonical)
    shared = fixed[:, 0] == 0
    values = np.empty((1001, len(METHODS), len(METRICS)))
    point = None
    for iteration in range(1001):
        orders = {'fixed': fixed, 'RRF': rrf}
        for name in ('SR4', 'SR8', 'QL5', 'G12'):
            orders[name] = np.empty_like(fixed)
            for fold in range(5):
                source = old[fold] if name == 'G12' else fitted[fold]
                key = 'G_order' if name == 'G12' else name+'_order'
                orders[name][source['test_indices']] = source[key][iteration]
        orders['G12_guard'] = guard(orders['G12'], shared)
        orders['SR8_guard'] = guard(orders['SR8'], shared)
        w = np.ones(len(qids)) if iteration == 0 else plan[iteration-1].astype(float)
        for i, name in enumerate(METHODS):
            indicator = indicators(orders[name], positive, negative)
            values[iteration, i] = np.sum(w[:, None]*indicator, axis=0)/w.sum()
        if iteration == 0:
            point = {name: order.copy() for name, order in orders.items()}
    np.savez_compressed(ROOT/'metrics_draws.npz', values=values, methods=np.array(METHODS), metrics=np.array(METRICS))
    np.savez_compressed(ROOT/'point_orders.npz', orders=np.stack([point[n] for n in METHODS]), methods=np.array(METHODS))
    point_ind = {n: indicators(point[n], positive, negative) for n in METHODS}
    reports = []
    for i, name in enumerate(METHODS):
        tr = transition(point_ind['fixed'][:, 0], point_ind[name][:, 0])
        reports.append({'method': name, 'queries': len(qids),
                        **{metric+'_pct': float(values[0, i, j]*100) for j, metric in enumerate(METRICS)},
                        'rescued_vs_fixed': tr['rescued'], 'new_losses_vs_fixed': tr['new_losses'],
                        'net_rescued_vs_fixed': tr['net']})
    comparisons = []
    for a, b in COMPARISONS:
        for metric in ('P1', 'N1'):
            diff = 100*(values[:, METHODS.index(a), METRICS.index(metric)] -
                        values[:, METHODS.index(b), METRICS.index(metric)])
            lo, hi = np.quantile(diff[1:], [.025, .975])
            comparisons.append({'comparison': a+'-'+b, 'metric': metric,
                'primary': (a, b, metric) == ('G12', 'SR8', 'P1'), 'difference_pp': float(diff[0]),
                'ci_low_pp': float(lo), 'ci_high_pp': float(hi), 'draws': 1000})
    save_csv(ROOT/'paired_bootstrap.csv', comparisons)
    old_summary = read(PREV/'summary.json')
    for name, old_name in [('fixed', 'fixed_K20'), ('G12', 'Post_G')]:
        previous = next(r for r in old_summary['reports'] if r['method'] == old_name)
        now = next(r for r in reports if r['method'] == name)
        for key in ['P1_pct', 'N1_pct', 'rescued_vs_fixed', 'new_losses_vs_fixed']:
            assert abs(previous[key]-now[key]) < 1e-10, (name, key)
        assert abs(previous['positive_recall@10_pct']-now['R10_pct']) < 1e-10
    for metric in ('P1', 'N1'):
        before = next(r for r in old_summary['paired_bootstrap'] if
                      r['method'] == 'Post_G' and r['reference'] == 'fixed_K20' and r['metric'] == metric)
        after = next(r for r in comparisons if r['comparison'] == 'G12-fixed' and r['metric'] == metric)
        for key in ['difference_pp', 'ci_low_pp', 'ci_high_pp']:
            assert abs(before[key]-after[key]) < 1e-10, key
    consensus = {'shared_total': int(shared.sum()),
                 'shared_reference': {category: int(np.sum(shared & point_ind['fixed'][:, j]))
                                      for j, category in enumerate(('P', 'N', 'U'))}, 'guards': {}}
    for name in ('G12', 'SR8'):
        before, after = point_ind[name][:, 0], point_ind[name+'_guard'][:, 0]
        consensus['guards'][name] = {**transition(before, after),
             'shared_correct_restored': int(np.sum(shared & point_ind['fixed'][:, 0] & ~before & after)),
             'shared_wrong_rescues_revoked': int(np.sum(shared & ~point_ind['fixed'][:, 0] & before & ~after)),
             'changed_top1': int(np.sum(point[name][:, 0] != point[name+'_guard'][:, 0]))}
    consensus['original_G_shared_correct_losses'] = int(np.sum(shared & point_ind['fixed'][:, 0] & ~point_ind['G12'][:, 0]))
    consensus['old_93_consistency_check'] = consensus['original_G_shared_correct_losses'] == 93
    assert consensus['old_93_consistency_check']
    write(ROOT/'consensus.json', consensus)
    per_query = []
    for name in METHODS:
        for i, qid in enumerate(qids):
            per_query.append({'qid': qid, 'method': name, 'top_document': pools[i][point[name][i, 0]],
                              'shared_top': bool(shared[i]),
                              **{m: int(point_ind[name][i, j]) for j, m in enumerate(METRICS)}})
    save_csv(ROOT/'per_query.csv.gz', per_query)
    attribution = []
    for i in np.flatnonzero(point['G12'][:, 0] != fixed[:, 0]):
        fold = int(folds[i])
        a, b = int(point['G12'][i, 0]), int(fixed[i, 0])
        c = (x[i, a]-x[i, b])/old[fold]['G_scale'][0]*old[fold]['G_coef'][0]
        local = int(np.flatnonzero(old[fold]['test_indices'] == i)[0])
        expected = old[fold]['G_point_scores'][local, a]-old[fold]['G_point_scores'][local, b]
        assert abs(c.sum()-expected) < 1e-10
        attribution.append({'qid': qids[i], 'fold': fold, 'fixed_winner': pools[i][b], 'G_winner': pools[i][a],
                'shared_top': bool(shared[i]), 'common_correct': bool(shared[i] and point_ind['fixed'][i, 0]),
                **{f'c{j}': float(c[j]) for j in range(12)},
                'total_diff': float(c.sum()), 'sr4_diff': float(c[:4].sum()),
                'interaction_diff': float(c[8:].sum()), 'text_diff': float(c[4:8].sum())})
    save_csv(ROOT/'attribution.csv.gz', attribution)
    cc = [r for r in attribution if r['common_correct']]
    summary = {'reports': reports, 'comparisons': comparisons, 'feature_columns': meta['G_columns'],
               'fixed_G_reproduction': 'PASS', 'attribution_rows': len(attribution),
               'common_correct_changed': len(cc),
               'common_correct_SR8_partial_positive': sum(r['sr4_diff']+r['interaction_diff'] > 0 for r in cc),
               'common_correct_SR4_partial_positive': sum(r['sr4_diff'] > 0 for r in cc),
               'point_lambdas': {n: [float(f[n+'_lambda'][0]) for f in fitted] for n in ('SR4', 'SR8', 'QL5')},
               'predictions_sealed_sha256': sha(ROOT/'predictions_sealed.json'),
               'wall_seconds': time.monotonic()-started, 'new_neural_pairs': 0, 'reused_Qwen_pairs': 69040,
               'head_size': 20, 'tail_size': 80, 'diagnostic_not_confirmatory': True}
    write(ROOT/'summary.json', summary)
    print('Scoring finished; fixed and G reproduced, including old bootstrap intervals.', flush=True)


if __name__ == '__main__':
    main()
