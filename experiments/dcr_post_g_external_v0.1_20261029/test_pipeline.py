"""Small fixtures: synthetic math/adapter checks, no actual fit/model inference."""
import math
import unittest
import numpy as np
from prepare_inputs import adapt_queries
from prepare_neural import prefix_end
from metrics import evaluate,coverage
from paths import load_source_math


class Tests(unittest.TestCase):
    def test_topic_suffix(self):
        topics,q=adapt_queries([{'_id':'a_b_0','level':0,'text':'topic'},{'_id':'a_b_2','level':2,'text':'topic request'}])
        self.assertEqual(q[0]['base_id'],'a_b');self.assertEqual(q[0]['topic'],'topic')
        with self.assertRaises(AssertionError):adapt_queries([{'_id':'a_1','level':2,'text':'x'}])
        with self.assertRaises(AssertionError):adapt_queries([{'_id':'a_1','level':1,'text':'x'}])

    def test_linear_gain_and_full_qrels(self):
        r=evaluate(['b','a'],{'a':4,'b':2})
        expected=(2+4/math.log2(3))/(4+2/math.log2(3))
        self.assertAlmostEqual(r['nDCG20'],expected)
        r=evaluate(['a'],{'a':1,'b':1,'c':1})
        self.assertEqual(r['R10'],1/3)
        self.assertEqual(coverage(['a'],{'a':1,'b':1,'c':1},100)['positive_document_recall'],1/3)
        self.assertEqual(evaluate(['x'],{'a':1}),{'P1':0.,'nDCG20':0.,'MRR20':0.,'R10':0.})
        self.assertEqual(evaluate(['x'],{}),{'P1':0.,'nDCG20':0.,'MRR20':0.,'R10':0.})

    def test_cluster_query_weight(self):
        # A has2 requests, B has1: sampling A,B gives query mean2/3, not mean-of-base1/2.
        base=np.array([0,0,1]);count=np.array([1,1]);v=np.array([1.,1.,0.])
        weights=count[base];self.assertAlmostEqual(float(np.sum(weights*v)/weights.sum()),2/3)
        count=np.array([2,0]);weights=count[base]
        self.assertEqual(float(np.sum(weights*v)/weights.sum()),1.)

    def test_empty_body_G_and_score_ties(self):
        m=load_source_math();x=m.doc_features(np.zeros(20),np.zeros(20),[[]]*20,['topic'],['request'],[])[:,:12]
        np.testing.assert_array_equal(x,np.zeros((20,12)))
        self.assertEqual(prefix_end('',[],[],4),0)
        self.assertEqual(prefix_end('  ',[],[],4),2)
        with self.assertRaises(AssertionError):prefix_end('',[],[],0)
        canonical=np.array([list(range(19,-1,-1))]);order=m.rank_positions(np.zeros((1,20)),canonical)
        np.testing.assert_array_equal(order,canonical)


if __name__=='__main__':unittest.main()
