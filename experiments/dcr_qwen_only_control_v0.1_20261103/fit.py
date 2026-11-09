"""Q7 fits reuse exactly the source folds, legal labels and ridge procedure."""
import argparse
import time
from pathlib import Path
import numpy as np
from config import ROOT, POST, ATTR, REF, Q7, read, write, sha, check, load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, choices=range(5))
    parser.add_argument('--transfer', action='store_true')
    args = parser.parse_args()
    assert args.transfer != (args.fold is not None)
    start = time.monotonic()
    check()
    math = load(ATTR / 'math_core.py', 'original_ridge')
    with np.load(POST / 'features.npz', allow_pickle=False) as data:
        x = np.ascontiguousarray(data['X'][:, :, Q7])
        canonical = data['canonical']
    with np.load(POST / 'prepared/plan.npz', allow_pickle=False) as data:
        folds, draws = data['fold'], data['weights']
    assert x.shape == (3452, 20, 7) and np.isfinite(x).all()
    if args.transfer:
        meta, references = read(POST / 'head_and_tail.json'), read(REF)
        pos, neg = [], []
        for qid, pool in zip(meta['qids'], meta['ranked_pools']):
            head = pool[:20]
            pos.append(head.index(references[qid]['positive_id']) if references[qid]['positive_id'] in head else -1)
            neg.append(head.index(references[qid]['negative_id']) if references[qid]['negative_id'] in head else -1)
        pos, neg = np.array(pos), np.array(neg)
        counts = np.zeros(4, dtype=np.int64)
        for fold in range(5):
            test = np.flatnonzero(folds == fold)
            stats = math.statistics(x, pos, neg, (folds != fold).astype(np.int16))
            for j, lam in enumerate(math.LAMBDAS):
                order = math.rank(math.predict(x[test], math.solve(stats, lam)), canonical[test])
                counts[j] += np.sum(order[:, 0] == pos[test])
        lam = math.LAMBDAS[int(np.argmax(counts))]
        model = math.solve(math.statistics(x, pos, neg, np.ones(len(x), dtype=np.int16)), lam)
        assert model[2] == 2901
        write(ROOT / 'transfer_model.json', {'lambda': lam, 'CV_correct_by_lambda': counts.tolist(),
              'scale': model[0].tolist(), 'coef': model[1].tolist(), 'pair_instances': model[2],
              'columns': Q7, 'source_only': True, 'target_references_read': False,
              'seconds': time.monotonic() - start})
        return
    fold = args.fold
    view = read(POST / 'prepared' / f'view_{fold}.json')
    for name, expected in view['allowed_inputs'].items():
        assert sha(Path(name)) == expected, name
    with np.load(view['labels_path'], allow_pickle=False) as labels:
        pos = np.full(len(x), -2, dtype=np.int8)
        neg = pos.copy()
        pos[labels['train_indices']], neg[labels['train_indices']] = labels['positive'], labels['negative']
    test = np.array(view['test_indices'])
    assert np.array_equal(test, np.flatnonzero(folds == fold))
    result = {'test_indices': test, 'lambda': np.empty(1001),
              'inner_correct_counts': np.empty((1001, 4), dtype=np.int32),
              'scale': np.empty((1001, 7)), 'coef': np.empty((1001, 7)),
              'pair_instances': np.empty(1001, dtype=np.int32),
              'order': np.empty((1001, len(test), 20), dtype=np.uint8)}
    for iteration in range(1001):
        weights = np.ones(len(x), dtype=np.int16) if not iteration else draws[iteration - 1]
        lam, counts = math.select(x, pos, neg, canonical, folds, fold, weights)
        model = math.solve(math.statistics(x, pos, neg, weights * (folds != fold)), lam)
        for key, value in [('lambda', lam), ('inner_correct_counts', counts), ('scale', model[0]),
                           ('coef', model[1]), ('pair_instances', model[2]),
                           ('order', math.rank(math.predict(x[test], model), canonical[test]))]:
            result[key][iteration] = value
        if iteration % 250 == 0:
            print(f'fold={fold} iteration={iteration}/1000 seconds={time.monotonic()-start:.3f}', flush=True)
    path = ROOT / 'fitted' / f'fold_{fold}.npz'
    assert not path.exists()
    np.savez_compressed(path, **result)
    write(path.with_suffix('.json'), {'fold': fold, 'sha256': sha(path), 'heldout_labels_read': False,
                                    'seconds': time.monotonic() - start})


if __name__ == '__main__':
    main()
