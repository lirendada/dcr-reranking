"""Synthetic fixed-row rank and exact invariant fixtures; no real labels/grades."""
import time
import numpy as np
from config import ROOT,write,math_core
from features import matched


def main():
 old=np.zeros((1,20,12));old[0,:,1]=np.arange(20)/7;old[0,:,3]=np.arange(1,21)/20
 old[0,:,4:8]=np.arange(80).reshape(20,4);old[0,:,11]=old[0,:,1]*(old[0,-1,1]-old[0,-2,1])
 canonical=np.arange(20)[None,:];m=np.arange(20.)[None,:]
 x=matched(old,m,canonical)
 assert x[0,19,2]==1/20 and x[0,0,2]==1.0
 assert np.array_equal(x[:,:,4:8],old[:,:,4:8]) and np.array_equal(x[:,:,[1,3,11]],old[:,:,[1,3,11]])
 tied=matched(old,np.zeros((1,20)),canonical[:,::-1]);assert tied[0,19,2]==1/20
 rng=np.random.default_rng(20261029);core=math_core();a=rng.normal(size=(12,20,12));p=np.zeros(12,int);n=np.ones(12,int);w=np.array([1,2,3]*4)
 stats=core.statistics(a,p,n,w);expanded=np.repeat(np.arange(12),w);other=core.statistics(a[expanded],p[expanded],n[expanded],np.ones(len(expanded)))
 for u,v in zip(stats[:3],other[:3]):np.testing.assert_allclose(u,v,atol=1e-13)
 assert core.solve(stats,.1)[2]==int(w.sum())
 write(ROOT/'fixtures.json',{'status':'PASS','fixed_row_full_score_rank_reversal':True,'canonical_ties':True,'Q_and_text_bitwise_invariant':True,'weighted_RMS_equivalent':True})
 # Resource-only one full synthetic fivefold kernel, repeated3; no real outcomes used.
 a=rng.normal(size=(3452,20,12));p=np.zeros(3452,int);n=np.ones(3452,int);can=np.tile(np.arange(20),(3452,1));fold=np.repeat(np.arange(5),[691,691,690,690,690]);times=[]
 for _ in range(3):
  start=time.monotonic();w=rng.poisson(1,3452)
  for f in range(5):
   lam,_=core.select(a,p,n,can,fold,f,w);model=core.solve(core.statistics(a,p,n,w*(fold!=f)),lam);core.rank(core.predict(a[fold==f],model),can[fold==f])
  times.append(time.monotonic()-start)
 write(ROOT/'synthetic_timing.json',{'synthetic_only':True,'kernels':3,'seconds_each':times,'projected1000_kernel_seconds':sum(times)/3*1000,'not_actual_production':True})


if __name__=='__main__':main()
