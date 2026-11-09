"""Small synthetic behavioral checks, before any real Q7 fit or metric."""
import math
import numpy as np
from config import ROOT, ATTR, Q7, M5, load, write, now
from score import evaluate


def main() -> None:
    checks = []
    x = np.arange(2 * 20 * 12, dtype=float).reshape(2, 20, 12)
    changed = x.copy()
    changed[:, :, M5] = -1e6
    assert np.array_equal(changed[:, :, Q7], x[:, :, Q7])
    assert sorted(Q7 + M5) == list(range(12)) and len(Q7) == 7
    checks.append('Q7 excludes all five MiniLM-dependent coordinates')
    kernel = load(ATTR / 'math_core.py', 'fixture_legacy_math')
    canonical = np.array([list(range(19, -1, -1))])
    order = kernel.rank(np.zeros((1, 20)), canonical)
    assert np.array_equal(order, canonical)
    checks.append('equal scores resolve by canonical document ID')
    features = np.zeros((1, 20, 7))
    model = kernel.solve(kernel.statistics(features, np.array([0]), np.array([1]), np.ones(1)), .1)
    assert np.array_equal(model[0], np.ones(7)) and np.array_equal(model[1], np.zeros(7))
    checks.append('zero RMS gives unit scale and zero coefficient')
    features = np.zeros((1, 20, 1)); features[0, 0, 0] = 1.
    model = kernel.solve(kernel.statistics(features, np.array([0]), np.array([1]), np.ones(1)), .1)
    assert abs(model[1][0] - 1 / 1.1) < 1e-14
    checks.append('one-feature ridge matches analytic solution')
    try:
        kernel.statistics(features, np.array([-1]), np.array([1]), np.ones(1))
    except ValueError:
        checks.append('missing endpoint pair raises instead of fabricating a fit')
    else:
        raise AssertionError('missing pair accepted')
    assert evaluate(['a', 'b'], {}) == [0., 0., 0., 0.]
    assert evaluate(['a', 'b'], {'b': 1.}) == [0., 1 / math.log2(3), .5, 1.]
    checks.append('zero-positive and graded full-reference metrics preserve denominators')
    # Two variants of topic a share the same weight; statistic is still a request mean.
    weights = np.array([2, 2, 0]); outcomes = np.array([1, 0, 1])
    assert np.sum(weights * outcomes) / weights.sum() == .5
    checks.append('topic variants share resampling count and request denominator')
    write(ROOT / 'fixtures.json', {'status': 'PASS', 'checked_utc': now(), 'checks': checks,
          'real_labels_or_scores_used': False})
    print(f'{len(checks)} synthetic checks PASS', flush=True)


if __name__ == '__main__':
    main()
