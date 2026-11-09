"""ExcluIR-only fivefold source lambda selection, then two complete2901-pair fits."""
import time
import numpy as np
from config import ROOT,POST,REF,QL_COLUMNS,read,write,sha,math_core,check_phase


def main():
 start=time.monotonic();check_phase('transfer');m=math_core();meta=read(POST/'head_and_tail.json');refs=read(REF)
 with np.load(ROOT/'features.npz',allow_pickle=False) as a:same=a['X'];canonical=a['canonical']
 with np.load(POST/'features.npz',allow_pickle=False) as a:old=a['X'][:,:,:12]
 with np.load(POST/'prepared/plan.npz',allow_pickle=False) as a:fold=a['fold']
 pos=[];neg=[]
 for q,pool in zip(meta['qids'],meta['ranked_pools']):
  head=pool[:20];p,n=refs[q]['positive_id'],refs[q]['negative_id'];pos.append(head.index(p) if p in head else -1);neg.append(head.index(n) if n in head else -1)
 pos,neg=np.array(pos),np.array(neg);models={}
 for name,x in [('G_same12',same),('QL5',np.ascontiguousarray(old[:,:,QL_COLUMNS]))]:
  counts=np.zeros(4,dtype=np.int64)
  for f in range(5):
   train=(fold!=f).astype(np.int16);test=np.flatnonzero(fold==f);stats=m.statistics(x,pos,neg,train)
   for j,lam in enumerate(m.LAMBDAS):counts[j]+=int(np.sum(m.rank(m.predict(x[test],m.solve(stats,lam)),canonical[test])[:,0]==pos[test]))
  lam=m.LAMBDAS[int(np.argmax(counts))];model=m.solve(m.statistics(x,pos,neg,np.ones(len(x),dtype=np.int16)),lam)
  assert model[2]==2901
  models[name]={'lambda':lam,'CV_correct_by_lambda':counts.tolist(),'lambda_grid':list(m.LAMBDAS),'CV_query_denominator':3452,
    'scale':model[0].tolist(),'coef':model[1].tolist(),'pair_instances':model[2],'dimensions':x.shape[2]}
 write(ROOT/'transfer_models.json',{'models':models,'source_only':'ExcluIR','IFIR_files_opened':False,'lambda_tie':'smallest','wall_seconds':time.monotonic()-start})
 write(ROOT/'transfer_models_sealed.json',{'models_sha256':sha(ROOT/'transfer_models.json'),'manifest_sha256':sha(ROOT/'frozen_manifest.json'),'before_A_score_and_any_new_IFIR_prediction':True})


if __name__=='__main__':main()
