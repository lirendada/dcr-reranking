"""One fold, training-side labels only, same selection and draws as original G."""
import argparse
from pathlib import Path
import time
import numpy as np
from paths import ROOT, PREV, read, write, sha, check_code
from math_core import BLOCKS, select, statistics, solve, predict, rank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    outer = parser.parse_args().fold
    started = time.monotonic()
    check_code()
    view = read(PREV/'prepared'/f'view_{outer}.json')
    for path, digest in view['allowed_inputs'].items():
        assert sha(Path(path)) == digest, path
    with np.load(PREV/'features.npz', allow_pickle=False) as data:
        x, canonical = data['X'][:, :, :12], data['canonical']
    with np.load(PREV/'prepared/plan.npz', allow_pickle=False) as data:
        plan, folds = data['weights'], data['fold']
    with np.load(view['labels_path'], allow_pickle=False) as data:
        train, p, n = data['train_indices'], data['positive'], data['negative']
    pos = np.full(len(x), -2, dtype=np.int8)
    neg = pos.copy()
    pos[train], neg[train] = p, n
    test = np.asarray(view['test_indices'], dtype=np.int32)
    assert np.array_equal(test, np.flatnonzero(folds == outer))
    result = {'test_indices': test}
    blocks = {name: np.ascontiguousarray(x[:, :, columns]) for name, columns in BLOCKS.items()}
    for name, block in blocks.items():
        size = block.shape[2]
        result[name+'_lambda'] = np.empty(1001)
        result[name+'_inner_correct_counts'] = np.empty((1001, 4), dtype=np.int32)
        result[name+'_scale'] = np.empty((1001, size))
        result[name+'_coef'] = np.empty((1001, size))
        result[name+'_pair_instances'] = np.empty(1001, dtype=np.int32)
        result[name+'_order'] = np.empty((1001, len(test), 20), dtype=np.uint8)
        result[name+'_point_scores'] = np.empty((len(test), 20))
    for iteration in range(1001):
        weights = np.ones(len(x), dtype=np.int16) if iteration == 0 else plan[iteration-1]
        training = weights * (folds != outer)
        for name, block in blocks.items():
            lam, counts = select(block, pos, neg, canonical, folds, outer, weights)
            model = solve(statistics(block, pos, neg, training), lam)
            scores = predict(block[test], model)
            for suffix, value in [('lambda', lam), ('inner_correct_counts', counts),
                                  ('scale', model[0]), ('coef', model[1]), ('pair_instances', model[2]),
                                  ('order', rank(scores, canonical[test]))]:
                result[name+'_'+suffix][iteration] = value
            if iteration == 0:
                result[name+'_point_scores'][:] = scores
        if iteration % 100 == 0:
            print(f'fold={outer} iteration={iteration}/1000 elapsed={time.monotonic()-started:.2f}', flush=True)
    path = ROOT/'fitted'/f'fold_{outer}.npz'
    assert not path.exists()
    np.savez_compressed(path, **result)
    write(path.with_suffix('.json'), {'fold': outer, 'file_sha256': sha(path),
          'heldout_labels_read': False, 'wall_seconds': time.monotonic()-started})


if __name__ == '__main__':
    main()
