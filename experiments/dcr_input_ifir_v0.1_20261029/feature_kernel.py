"""Verbatim pure matched() from frozen A; no A execution/data imports."""
import numpy as np

def matched(old,full_scores,canonical):
 assert old.shape[:2]==full_scores.shape==canonical.shape and old.shape[2]==12
 assert np.isfinite(full_scores).all()
 x=old.copy()
 for i,m in enumerate(full_scores):
  std=m.std(ddof=0);z=(m-m.mean())/std if std else np.zeros_like(m)
  order=canonical[i][np.argsort(-m[canonical[i]],kind='stable')]
  x[i,:,0]=z
  for rank,j in enumerate(order):x[i,j,2]=(rank+1)/20
  gm=np.sort(z)[-1]-np.sort(z)[-2];q=old[i,:,1];gq=np.sort(q)[-1]-np.sort(q)[-2]
  x[i,:,8]=z*gm;x[i,:,9]=z*gq;x[i,:,10]=q*gm
 # Q, rankQ, Q×gapQ and all four lexical/text coordinates must be bitwise invariant.
 assert np.array_equal(x[:,:,[1,3,11]],old[:,:,[1,3,11]])
 assert np.array_equal(x[:,:,4:8],old[:,:,4:8])
 return x
