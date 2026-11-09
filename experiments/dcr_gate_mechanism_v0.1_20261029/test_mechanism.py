"""Small independent arithmetic/order fixtures; no formal labels or model."""
from collections import Counter
import math
import unittest


class MechanismTests(unittest.TestCase):
    def test_stable_gate_and_tie(self):
        from gate_cpu import hard
        order = ['b', 'a', 'd', 'c']
        self.assertEqual(hard(order, {'b', 'd'}), ['a', 'c', 'b', 'd'])
        self.assertEqual(hard(order, set()), order)
        no_yes = [(1, 1), (2, 1), (0, 1)]
        self.assertEqual([n > y for n, y in no_yes], [False, True, False])

    def test_postings_full_sort_and_formula(self):
        from retrieve_external import build_index, score_query, top_indices, tokenize
        docs = ['red red blue', 'red green', 'blue', 'absent']
        index, stats = build_index(docs)
        self.assertEqual(stats['N'], 4)
        query = 'red red blue'
        scores = score_query(query, index, 4)
        expected = []
        for text in docs:
            tf = Counter(tokenize(text))
            value = 0
            for term in tokenize(query):
                df = sum(term in tokenize(d) for d in docs)
                if tf[term]:
                    value += math.log(1 + (4 - df + .5) / (df + .5)) * (2.5 * tf[term]) / (tf[term] + 1.5 * (.25 + .75 * len(tokenize(text)) / stats['avgdl']))
            expected.append(value)
        for a, b in zip(scores, expected):
            self.assertAlmostEqual(a, b, places=12)
        self.assertEqual(top_indices(scores, 3), sorted(range(4), key=lambda i: (-expected[i], i))[:3])
        self.assertEqual(top_indices(score_query('unseen', index, 4), 2), [0, 1])

    def test_linear_gain_ndcg(self):
        # Reversed rel4/2: (2+4/log2(3))/(4+2/log2(3)).
        from gate_cpu import ev
        ap, ndcg = ev.frozen_metric().metrics(['r2', 'r4'], {'r2': 2, 'r4': 4})
        self.assertEqual(ap, 1)
        self.assertAlmostEqual(ndcg, (2 + 4 / math.log2(3)) / (4 + 2 / math.log2(3)))


if __name__ == '__main__':
    unittest.main()
