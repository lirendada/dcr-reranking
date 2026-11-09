"""Paired fixed-prediction input contrasts, using the old component draw plan."""
import csv
import gzip
import time
import numpy as np
from config import (ROOT, SOURCE_POST, HEAD_PATH, REFERENCE_PATH, PLAN_PATH, CELLS,
                    METRICS, CONTRASTS, read, write, rows, sha, verify_code, verify_sources)


def metrics(order, positive, negative):
    p, n = order[:, 0] == positive, order[:, 0] == negative
    return np.stack((p, n, ~(p | n), (order[:, :10] == positive[:, None]).any(1)), axis=1)


def main():
    started = time.monotonic()
    manifest = verify_code()
    verify_sources(manifest, 'scoring')
    seal = read(ROOT/'predictions_sealed.json')
    assert seal['frozen_manifest_sha256'] == sha(ROOT/'frozen_manifest.json')
    assert seal['scoring_references_opened'] is False
    for name, digest in seal['files'].items():
        assert sha(ROOT/name) == digest
    meta, reference = read(HEAD_PATH), read(REFERENCE_PATH)
    qids, pools = meta['qids'], meta['ranked_pools']
    pos, neg = [], []
    for qid, pool in zip(qids, pools):
        p, n = reference[qid]['positive_id'], reference[qid]['negative_id']
        pos.append(pool[:20].index(p) if p in pool[:20] else -1)
        neg.append(pool[:20].index(n) if n in pool[:20] else -1)
    pos, neg = np.asarray(pos), np.asarray(neg)
    with np.load(ROOT/'four_cell_arrays.npz', allow_pickle=False) as z:
        assert list(z['cells']) == list(CELLS) and list(z['qids']) == qids
        orders = z['orders']
    with np.load(PLAN_PATH, allow_pickle=False) as z:
        plan = z['weights']
    indicator = np.stack([metrics(order, pos, neg) for order in orders])
    values = np.empty((1001, 4, 4))
    values[0] = indicator.mean(axis=1)
    for iteration, weights in enumerate(plan, start=1):
        values[iteration] = np.sum(weights[None, :, None]*indicator, axis=1)/weights.sum()
    assert not (ROOT/'bootstrap_metrics.npz').exists()
    np.savez_compressed(ROOT/'bootstrap_metrics.npz', values=values, cells=np.array(CELLS), metrics=np.array(METRICS))
    reports = [{'cell': cell, 'queries': len(qids),
                **{metric+'_pct': float(values[0, ci, j]*100) for j, metric in enumerate(METRICS)}}
               for ci, cell in enumerate(CELLS)]
    comparison = []
    for name, contrast in CONTRASTS.items():
        for j, metric in enumerate(METRICS[:2]):
            diff = (values[:, :, j] @ np.asarray(contrast))*100
            low, high = np.quantile(diff[1:], [.025, .975])
            comparison.append({'comparison': name, 'metric': metric,
                'primary': name == 'input_effect_interaction' and metric == 'P1',
                'difference_pp': float(diff[0]), 'ci_low_pp': float(low), 'ci_high_pp': float(high), 'draws': 1000})
    with (ROOT/'paired_bootstrap.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
        writer.writeheader(); writer.writerows(comparison)
    per_query = []
    for ci, cell in enumerate(CELLS):
        for i, qid in enumerate(qids):
            per_query.append({'cell': cell, 'query_id': qid, 'top_document': pools[i][orders[ci, i, 0]],
                              **{metric: int(indicator[ci, i, j]) for j, metric in enumerate(METRICS)}})
    with gzip.open(ROOT/'per_query.csv.gz', 'xt', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_query[0]))
        writer.writeheader(); writer.writerows(per_query)
    transitions = {}
    for model, before, after in [('MiniLM', 0, 1), ('Qwen', 2, 3)]:
        source, target = indicator[before, :, 0], indicator[after, :, 0]
        rescue, lost = int(np.sum(~source & target)), int(np.sum(source & ~target))
        classes = ('P', 'N', 'U')
        transitions[model] = {'topic_to_full': True, 'rescued': rescue, 'new_losses': lost, 'net': rescue-lost,
            'changed_top1': int(np.sum(orders[before, :, 0] != orders[after, :, 0])),
            'category_transitions': {a+'->'+b: int(np.sum(indicator[before, :, j] & indicator[after, :, k]))
                for j, a in enumerate(classes) for k, b in enumerate(classes)}}
    write(ROOT/'transitions.json', transitions)
    old_meta=read(SOURCE_POST/'head_and_tail.json')
    coverage={}
    for label,pp in [('topic_retrieval',pools),('full_retrieval',old_meta['ranked_pools'])]:
        assert old_meta['qids']==qids
        coverage[label]={'positive_in_top100':sum(reference[q]['positive_id'] in pool for q,pool in zip(qids,pp)),
                         'positive_in_head20':sum(reference[q]['positive_id'] in pool[:20] for q,pool in zip(qids,pp)),
                         'queries':len(qids)}
    write(ROOT/'candidate_coverage.json',coverage)
    write(ROOT/'summary.json', {'reports': reports, 'comparisons': comparison,
        'predictions_sealed_sha256': sha(ROOT/'predictions_sealed.json'),
        'bootstrap': {'draws': 1000, 'components': 3239, 'strata': 5, 'refit': False,
                      'plan_sha256': sha(PLAN_PATH), 'unit': 'shared endpoints/duplicate-query components'},
        'candidate_coverage':coverage,
        'original_full_retrieval_reports':read(ROOT.parent/'dcr_model_input_v0.1_20261029/summary.json')['reports'],
        'positive_in_head': int(np.sum(pos >= 0)), 'negative_in_head': int(np.sum(neg >= 0)),
        'cost': {'new_model_pairs': 552320, 'head_pairs_per_cell': 69040, 'cell_inventory': seal['cell_inventory'],
                 'new_MiniLM_topic_selection_pairs': 345200},
        'CPU_scoring_wall_seconds': time.monotonic()-started, 'judgment_scope': 'Published endpoints; U remains unjudged',
        'scope': 'Observed-data retrieval-input control; new question0 BM25 candidates, topic MiniLM head20 and shared body prefixes'})
    print('All four cells, ten intervals and input transitions scored.', flush=True)


if __name__ == '__main__':
    main()
