"""Frozen feature-block comparisons and label-free consensus postprocessing."""
import numpy as np

BLOCKS = {'SR4': [0, 1, 2, 3], 'SR8': [0, 1, 2, 3, 8, 9, 10, 11],
          'QL5': [1, 4, 5, 6, 7]}
LAMBDAS = (.001, .01, .1, 1.)
METHODS = ('fixed', 'RRF', 'SR4', 'SR8', 'QL5', 'G12', 'G12_guard', 'SR8_guard')
METRICS = ('P1', 'N1', 'U1', 'R10')
COMPARISONS = (('G12', 'SR8'), ('SR8', 'SR4'), ('SR8', 'fixed'), ('QL5', 'fixed'),
               ('G12_guard', 'G12'), ('SR8_guard', 'SR8'), ('RRF', 'fixed'),
               ('G12', 'QL5'), ('G12', 'fixed'))


def rank(scores: np.ndarray, canonical: np.ndarray) -> np.ndarray:
    values = np.take_along_axis(scores, canonical, axis=1)
    return np.take_along_axis(canonical, np.argsort(-values, axis=1, kind='stable'), axis=1)


def guard(order: np.ndarray, shared: np.ndarray) -> np.ndarray:
    result = order.copy()
    for i in np.flatnonzero(shared):
        result[i] = np.concatenate(([0], order[i][order[i] != 0]))
    return result


def statistics(x: np.ndarray, pos: np.ndarray, neg: np.ndarray, weights: np.ndarray) -> tuple:
    eligible = (pos >= 0) & (neg >= 0) & (weights > 0)
    index = np.flatnonzero(eligible)
    count = int(weights[eligible].sum())
    if not count:
        raise ValueError('No eligible training pairs')
    d = x[index, pos[index]] - x[index, neg[index]]
    w = weights[index].astype(float)
    scale = np.sqrt(np.sum(w[:, None] * d*d, axis=0) / count)
    scale[scale == 0] = 1.
    z = d / scale
    return scale, z.T @ (w[:, None] * z), z.T @ w, count


def solve(stats: tuple, lam: float) -> tuple:
    scale, gram, rhs, count = stats
    coefficient = np.linalg.solve(gram + count*lam*np.eye(len(scale)), rhs)
    assert np.isfinite(coefficient).all()
    return scale, coefficient, count


def predict(x: np.ndarray, model: tuple) -> np.ndarray:
    return (x/model[0]) @ model[1]


def select(x: np.ndarray, pos: np.ndarray, neg: np.ndarray, canonical: np.ndarray,
           folds: np.ndarray, outer: int, weights: np.ndarray) -> tuple:
    counts = np.zeros(4, dtype=np.int64)
    for held in range(5):
        if held == outer:
            continue
        training = weights * ((folds != outer) & (folds != held))
        validation = np.flatnonzero((folds == held) & (weights > 0))
        stats = statistics(x, pos, neg, training)
        for j, lam in enumerate(LAMBDAS):
            winners = rank(predict(x[validation], solve(stats, lam)), canonical[validation])[:, 0]
            counts[j] += np.sum(weights[validation] * (winners == pos[validation]))
    return LAMBDAS[int(np.argmax(counts))], counts
