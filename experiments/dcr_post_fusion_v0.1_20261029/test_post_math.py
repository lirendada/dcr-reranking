"""Synthetic-only fixtures; must execute on authorized server, not local fitting."""
import unittest
import numpy as np
from score import pair_credit
from post_math import doc_features,zscore,ridge_fit,ridge_scores,select_alpha,rank_positions


class Tests(unittest.TestCase):
    def test_pair_no_intercept_and_weighted_duplicates(self):
        rng=np.random.default_rng(12);x=rng.normal(size=(12,20,18))
        p=np.zeros(12,dtype=int);n=np.ones(12,dtype=int);w=np.array([1,2,3]*4)
        model=ridge_fit(x,p,n,w,.01,18)
        indices=np.repeat(np.arange(12),w)
        repeated=ridge_fit(x[indices],p[indices],n[indices],np.ones(len(indices),int),.01,18)
        np.testing.assert_allclose(model[0],repeated[0],atol=1e-14)
        np.testing.assert_allclose(model[1],repeated[1],atol=1e-14)
        self.assertGreater(np.linalg.norm(model[1]),0)
        changed=x.copy();changed[11]+=1000;w[11]=0
        a=ridge_fit(x,p,n,w,.01,18);b=ridge_fit(changed,p,n,w,.01,18)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])

    def test_features_and_alpha_endpoint(self):
        m=np.arange(20,0,-1,dtype=float);q=np.roll(m,8);docs=[['x','x','a']]+[['b']]*19
        f=doc_features(m,q,docs,['a'],['b'],['x'])
        self.assertEqual(f.shape,(20,18));self.assertEqual(f[0,4],1.)
        self.assertEqual(f[0,12],1.);self.assertEqual(f[0,13],1.)
        self.assertAlmostEqual(f[0,17],f[0,1])
        gap=np.sort(zscore(q))[-1]-np.sort(zscore(q))[-2]
        self.assertAlmostEqual(f[0,9],f[0,0]*gap)
        canonical=np.tile(np.arange(19,-1,-1),(1,1))
        baseline=rank_positions(q[None,:],canonical)
        fusion=rank_positions(f[None,:,1],canonical)
        np.testing.assert_array_equal(baseline,fusion)
        zeros=np.zeros((1,20));np.testing.assert_array_equal(rank_positions(zeros,canonical),canonical)

    def test_pair_true_ties_and_segments(self):
        head=np.zeros(20);mini=np.arange(100.)
        self.assertEqual(pair_credit(0,1,head,mini),.5)
        self.assertEqual(pair_credit(0,99,head,mini),1.)
        self.assertEqual(pair_credit(99,0,head,mini),0.)
        self.assertIsNone(pair_credit(-1,0,head,mini))

    def test_unjudged_not_negative(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(3,20,18))
        p=np.array([0,-1,0]);n=np.array([1,-1,1]);w=np.array([1,4,1])
        a=ridge_fit(x,p,n,w,.1,12)
        x[1]=100000
        b=ridge_fit(x,p,n,w,.1,12)
        np.testing.assert_array_equal(a[1],b[1])


if __name__=='__main__':unittest.main()
