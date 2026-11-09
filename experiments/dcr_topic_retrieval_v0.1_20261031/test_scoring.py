"""Small protocol fixtures; no model or experimental reference access."""
import numpy as np
from config import ROOT, CONTRASTS, write
from score import metrics


def main():
    assert str(ROOT).startswith('/path/to/remote/lw_experiments/')
    order = np.tile(np.arange(20), (3, 1))
    actual = metrics(order, np.array([0, 3, -1]), np.array([1, 0, -1]))
    assert np.array_equal(actual, [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0]])
    # Input interaction must equal the difference between the two input gains.
    points = np.array([.20, .25, .40, .55])
    assert abs(points @ CONTRASTS['input_effect_interaction'] - .10) < 1e-14
    assert abs(points @ CONTRASTS['input_effect_interaction'] -
               (points @ CONTRASTS['Qwen_full-Qwen_topic'] -
                points @ CONTRASTS['MiniLM_full-MiniLM_topic'])) < 1e-14
    # Duplicating a component is equivalent to weighting all its queries together.
    values = np.array([[0, 1, 1, 1], [1, 1, 0, 1], [1, 0, 1, 0]])
    w = np.array([2, 2, 1])
    np.testing.assert_allclose((w[:, None]*values).sum(0)/w.sum(), values[[0, 0, 1, 1, 2]].mean(0))
    write(ROOT/'scoring_fixtures.json', {'passed': True, 'synthetic_only': True,
         'checks': ['positive_negative_unjudged_and_out_of_head', 'interaction_direction', 'component_weight_equivalence']})
    print('PASS scoring protocol fixtures', flush=True)


if __name__ == '__main__':
    main()
