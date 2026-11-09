"""Tie-preserving gate exports and complete-candidate scoring, artificial data only."""
import unittest
from scorer import evaluate
from score_external_neural import ordinal_gate_scores


class ExternalGateTests(unittest.TestCase):
    def test_tie_credit_and_group_override(self):
        refs = {'q': {'positive_id': 'd0', 'negative_id': 'd1'}}
        pools = {'q': {'d0': None, 'd1': None, 'd2': None}}
        relevance = {'d0': 1.0, 'd1': 1.0, 'd2': 20.0}
        scores = ordinal_gate_scores(relevance, {'d2'})
        rows, summary = evaluate(refs, {'q': scores}, [1, 10], pools)
        self.assertEqual(rows[0]['pairwise_credit'], .5)
        self.assertEqual(rows[0]['positive_rank'], 1)
        self.assertEqual(rows[0]['negative_rank'], 2)
        self.assertEqual(scores['d0'], scores['d1'])
        self.assertGreater(scores['d1'], scores['d2'])
        scores = ordinal_gate_scores(relevance, {'d0'})
        rows, _ = evaluate(refs, {'q': scores}, [1, 10], pools)
        self.assertEqual(rows[0]['pairwise_credit'], 0)
        self.assertEqual(rows[0]['positive_rank'], 3)

    def test_missing_competitor_withholds_formal_metrics(self):
        refs = {'q': {'positive_id': 'd0', 'negative_id': 'd1'}}
        rows, summary = evaluate(refs, {'q': {'d0': 1}}, [1, 10],
                                 {'q': {'d0': None, 'd1': None}})
        self.assertEqual(summary['evaluation_status'], 'incomplete_candidate_scores')
        self.assertIsNone(summary['ranking_metrics_all_planned_queries'])
        self.assertIsNone(rows[0]['positive_rank'])


if __name__ == '__main__':
    unittest.main()
