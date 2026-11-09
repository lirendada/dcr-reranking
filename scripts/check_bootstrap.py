"""Reconstruct selected source intervals from the archived fitted draw arrays."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ['P1', 'N1', 'U1', 'R10']


def main():
    checks = 0
    largest_error = 0.0
    for name in ['source_attribution', 'source_input_matched']:
        summary = json.loads((ROOT / 'data/summaries' / (name + '.json')).read_text())
        with np.load(ROOT / 'data/arrays' / (name + '_metrics_draws.npz'), allow_pickle=False) as stored:
            methods = stored['methods'].tolist()
            values = stored['values']
        if values.shape != (1001, len(methods), 4) or not np.isfinite(values).all():
            raise ValueError('Unexpected draw array in ' + name)
        for comparison in summary['comparisons']:
            a, b = comparison['comparison'].split('-')
            metric = METRICS.index(comparison['metric'])
            delta = (values[:, methods.index(a), metric] - values[:, methods.index(b), metric]) * 100
            point = float(delta[0])
            low, high = np.quantile(delta[1:], [.025, .975]).tolist()
            for field, actual in [('difference_pp', point), ('ci_low_pp', low), ('ci_high_pp', high)]:
                error = abs(actual - comparison[field])
                if error > 1e-8:
                    raise ValueError(name + ' / ' + comparison['comparison'] + ' / ' + field)
                largest_error = max(largest_error, error)
                checks += 1
    print(json.dumps({'status': 'PASS', 'numeric_checks': checks, 'maximum_absolute_error_pp': largest_error, 'numpy': np.__version__, 'scope': 'Selected source intervals recomputed from archived draws. Each original draw already includes its recorded refit.'}, indent=2))


if __name__ == '__main__':
    main()
