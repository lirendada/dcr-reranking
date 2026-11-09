"""Rebuild only MiniLM-dependent coordinates; source head rows never move."""
from datetime import datetime,timezone
import gzip,json,time
import numpy as np
from config import ROOT,POST,CELLS,read,write,sha,check_phase


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


def main():
 started=time.monotonic();check_phase('feature')
 meta=read(POST/'head_and_tail.json');qids=meta['qids'];pools=meta['ranked_pools']
 with np.load(POST/'features.npz',allow_pickle=False) as a:old=a['X'][:,:,:12];canonical=a['canonical'];raw=a['raw_scores']
 assert len(qids)==len(pools)==3452 and qids==sorted(qids) and old.shape==(3452,20,12)
 scores={}
 for cell in ['MiniLM_full','Qwen_full']:
  p=CELLS/'cells'/cell/'scores.jsonl.gz';r=read(p.parent/'receipt.json');assert r['exit_code']==0 and r['pairs']==69040 and sha(p)==r['score_sha256']
  scores[cell]={}
  with gzip.open(p,'rt') as f:
   for line in f:
    a=json.loads(line);key=(a['query_id'],a['document_id']);assert key not in scores[cell];scores[cell][key]=a
  assert len(scores[cell])==69040
 full=np.empty((3452,20));aligned=0
 for i,(q,pool) in enumerate(zip(qids,pools)):
  for j,d in enumerate(pool[:20]):
   m=scores['MiniLM_full'][q,d];w=scores['Qwen_full'][q,d]
   assert m['doc_prefix_end_char']==w['doc_prefix_end_char']
   assert w['score']==raw[i,j,1]
   full[i,j]=m['score'];aligned+=1
 x=matched(old,full,canonical)
 np.savez_compressed(ROOT/'features.npz',X=x,canonical=canonical,MiniLM_full_scores=full)
 write(ROOT/'features_sealed.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'pairs_aligned':aligned,
  'features_sha256':sha(ROOT/'features.npz'),'Q_columns_bitwise_unchanged':[1,3,11],'text_columns_bitwise_unchanged':[4,5,6,7],
  'head_row_identity_unchanged':True,'references_opened':False,'wall_seconds':time.monotonic()-started})
 check_phase('feature')


if __name__=='__main__':main()
