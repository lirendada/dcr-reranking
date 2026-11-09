"""Synthetic numerical and protocol checks; never loads experimental references."""
import importlib.util
import numpy as np
from paths import ROOT, PREV, write, sha
from math_core import BLOCKS, LAMBDAS, rank, guard, statistics, solve, predict, select


def main():
    assert str(ROOT).startswith('/path/to/remote/lw_experiments/')
    rng = np.random.default_rng(4217)
    x = rng.normal(size=(250, 20, 12))
    pos, neg = rng.integers(0, 10, 250), rng.integers(10, 20, 250)
    folds = np.arange(250) % 5
    canonical = np.tile(rng.permutation(20), (250, 1))
    weights = rng.integers(0, 4, 250)
    spec = importlib.util.spec_from_file_location('old_math', PREV/'post_math.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old_lambda, old_counts = old.select_lambda(x, pos, neg, canonical, folds, 0, weights, 12)
    lam, counts = select(x, pos, neg, canonical, folds, 0, weights)
    assert old_lambda == lam and np.array_equal(counts, old_counts)
    a = solve(statistics(x, pos, neg, weights), lam)
    b = old.ridge_fit(x, pos, neg, weights, lam, 12)
    np.testing.assert_allclose(a[0], b[0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(a[1], b[1], rtol=0, atol=1e-14)
    assert a[2] == b[2]
    assert np.array_equal(rank(predict(x, a), canonical), old.rank_positions(old.ridge_scores(x, b), canonical))
    # Compare weighted pairs to an independent duplicated-row augmented least-squares fit.
    idx = np.repeat(np.arange(250), weights)
    d = x[idx, pos[idx]] - x[idx, neg[idx]]
    scale = np.sqrt(np.mean(d*d, axis=0))
    z = d/scale
    augmented = np.vstack((z, np.sqrt(len(d)*lam)*np.eye(12)))
    target = np.concatenate((np.ones(len(d)), np.zeros(12)))
    coef = np.linalg.lstsq(augmented, target, rcond=None)[0]
    np.testing.assert_allclose(a[0], scale, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a[1], coef, rtol=1e-12, atol=1e-12)
    # Outer-test labels cannot affect selection or final training.
    other_pos, other_neg = pos.copy(), neg.copy()
    other_pos[folds == 0], other_neg[folds == 0] = -2, -2
    other_lambda, other_counts = select(x, other_pos, other_neg, canonical, folds, 0, weights)
    assert other_lambda == lam and np.array_equal(other_counts, counts)
    training = weights*(folds != 0)
    c = solve(statistics(x, pos, neg, training), lam)
    e = solve(statistics(x, other_pos, other_neg, training), lam)
    np.testing.assert_array_equal(c[1], e[1])
    # Stable canonical ties and label-free protection, including a shared wrong winner.
    tied = np.zeros((2, 20))
    assert np.array_equal(rank(tied, canonical[:2]), canonical[:2])
    order = np.array([[2, 0, 1], [1, 2, 0], [2, 1, 0]])
    guarded = guard(order, np.array([True, True, False]))
    assert np.array_equal(guarded, [[0, 2, 1], [0, 1, 2], [2, 1, 0]])
    assert np.array_equal(guard(guarded, np.array([True, True, False])), guarded)
    assert BLOCKS['SR8'] == [0, 1, 2, 3, 8, 9, 10, 11]
    assert BLOCKS['QL5'] == [1, 4, 5, 6, 7]
    for columns in BLOCKS.values():
        block = x[:, :, columns].copy()
        block[:, :, -1] = 7.0
        model = solve(statistics(block, pos, neg, weights), LAMBDAS[0])
        assert model[0][-1] == 1 and model[1][-1] == 0
        assert np.isfinite(predict(block, model)).all()
    write(ROOT/'fixtures.json', {'passed': True, 'synthetic_only': True,
         'old_kernel_sha256': sha(PREV/'post_math.py'), 'checks': ['old_G12_kernel_equivalence',
         'weighted_ridge_vs_duplicated_augmented_lstsq', 'outer_label_invariance',
         'canonical_ties', 'label_free_guard_and_idempotence', 'block_membership_and_zero_RMS']})
    print('PASS synthetic fixtures', flush=True)


if __name__ == '__main__':
    main()
