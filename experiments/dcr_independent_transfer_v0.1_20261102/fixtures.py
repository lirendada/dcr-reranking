"""Behavioral checks for metric denominators, canonical ties and matched columns."""
import math
import numpy as np
from config import ROOT, QRELS, write, source_math, source_matched
from score import evaluate


def main() -> None:
    order = [str(i) for i in range(1, 101)]
    result = evaluate(order, {'2': 3., '21': 2., '200': 1.})
    expected = [0., (3 / math.log2(3)) / (3 + 2 / math.log2(3) + 1 / math.log2(4)), .5, 1 / 3]
    assert np.allclose(result, expected, atol=1e-15, rtol=0)
    assert evaluate(order, {}) == [0., 0., 0., 0.]
    assert evaluate(order, {'21': 1.}) == [0., 0., 0., 0.]
    math_kernel = source_math()
    m = np.ones(20)
    q = np.arange(20, dtype=np.float64)
    x = math_kernel.doc_features(m, q, [[] for _ in range(20)], [], [], [])[:, :12]
    x[:, 2] = np.arange(1, 21) / 20
    x[:, 3] = np.arange(20, 0, -1) / 20
    canonical = np.array([list(range(19, -1, -1))])
    same = source_matched()(x[None], np.ones((1, 20)), canonical)
    assert np.isfinite(same).all() and np.array_equal(same[:, :, 0], np.zeros((1, 20)))
    assert np.array_equal(same[0][:, [1, 3, 11]], x[:, [1, 3, 11]])
    assert np.array_equal(same[:, :, 4:8], x[None, :, 4:8])
    positions = math_kernel.rank_positions(np.zeros((1, 20)), canonical)
    assert positions[0].tolist() == list(range(19, -1, -1))
    denied = False
    try:
        QRELS.open()
    except PermissionError:
        denied = True
    assert denied
    write(ROOT / 'fixtures.json', {'status': 'PASS', 'checks': [
        'full_reference_recall_and_graded_IDCG_denominators', 'MRR_rank20_boundary',
        'zero_reference_query_retained', 'zero_variance_score_finite',
        'matched_Q_and_text_columns_unchanged', 'canonical_score_ties',
        'reference_access_denied_in_prediction_phase'], 'model_or_target_evaluation': False})


if __name__ == '__main__':
    main()
