"""Same-dimension synthetic timing; no experimental labels or metric selection."""
import time
import numpy as np
from paths import ROOT, write
from math_core import BLOCKS, select, statistics, solve, predict, rank


def main():
    assert str(ROOT).startswith('/path/to/remote/lw_experiments/')
    rng = np.random.default_rng(619)
    x = rng.normal(size=(3452, 20, 12))
    canonical = np.tile(np.arange(20), (len(x), 1))
    folds = np.arange(len(x)) % 5
    pos = rng.integers(0, 10, len(x))
    neg = rng.integers(10, 20, len(x))
    pos[:551] = -1
    timings = []
    blocks = [np.ascontiguousarray(x[:, :, cols]) for cols in BLOCKS.values()]
    for iteration in range(4):
        weights = np.ones(len(x), dtype=int) if iteration == 0 else rng.poisson(1, len(x))
        started = time.monotonic()
        for block in blocks:
            lam, _ = select(block, pos, neg, canonical, folds, 0, weights)
            model = solve(statistics(block, pos, neg, weights*(folds != 0)), lam)
            rank(predict(block[folds == 0], model), canonical[folds == 0])
        timings.append(time.monotonic()-started)
    estimate = max(timings)*5005*1.5 + 120
    write(ROOT/'synthetic_timing.json', {'synthetic_only': True, 'iterations_seconds': timings,
          'estimated_production_seconds': estimate, 'safety_multiplier': 1.5,
          'scoring_and_io_allowance_seconds': 120, 'GPU_used': False, 'CPU_threads': 1})
    print(f'Synthetic estimated production wall={estimate:.1f}s', flush=True)


if __name__ == '__main__':
    main()
