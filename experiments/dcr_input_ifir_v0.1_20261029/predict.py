"""All seven prespecified methods, no qrels; original G/fixed rankings exactly retained."""
from datetime import datetime,timezone
import gzip,json,time
import numpy as np
from config import ROOT,OLD,A,DATASETS,CELLS,METHODS,read,write,rows,sha,check_A,check_phase
from feature_kernel import matched


def main():
 start=time.monotonic();check_phase('input');seal=check_A();models=read(A/'transfer_models.json')['models'];files={}
 for ds in DATASETS:
  queries=list(rows(OLD/ds/'inputs/queries.jsonl'));meta=read(OLD/ds/'outputs/feature_metadata.json');qids=meta['qids'];heads=meta['heads']
  assert qids==[r['query_id'] for r in queries]
  with np.load(OLD/ds/'outputs/features.npz',allow_pickle=False) as z:xold=z['X']
  scores={}
  for c in CELLS:
   p=ROOT/ds/'cells'/c/'scores.jsonl.gz';receipt=read(p.parent/'receipt.json');assert receipt['exit_code']==0 and sha(p)==receipt['score_sha256']
   files[str(p.relative_to(ROOT))]=sha(p);files[str((p.parent/'receipt.json').relative_to(ROOT))]=sha(p.parent/'receipt.json')
   scores[c]={}
   for r in rows(p):
    pool=scores[c].setdefault(r['query_id'],{});assert r['document_id'] not in pool;pool[r['document_id']]=r
   assert set(scores[c])==set(qids) and all(set(scores[c][q])==set(head) for q,head in zip(qids,heads))
  previous={m:{} for m in ['MiniLM','fixed_Qwen20','Post_G']}
  for r in rows(OLD/ds/'outputs/rankings.jsonl.gz'):previous[r['method']][r['query_id']]=r['order']
  mf=np.array([[scores['MiniLM_full'][q][d]['score'] for d in head] for q,head in zip(qids,heads)])
  can=np.array([sorted(range(20),key=lambda j:head[j]) for head in heads]);x=matched(xold,mf,can)
  values={}
  for name,features in [('G_same12',x),('QL5',np.ascontiguousarray(xold[:,:,[1,4,5,6,7]]))]:
   m=models[name];assert m['pair_instances']==2901
   values[name]=(features/np.array(m['scale']))@np.array(m['coef']);assert np.isfinite(values[name]).all()
  out=ROOT/ds/'outputs';out.mkdir(exist_ok=False)
  with gzip.open(out/'rankings.jsonl.gz','xt') as f:
   for i,query in enumerate(queries):
    q=query['query_id'];head=heads[i];base=previous['MiniLM'][q];assert base[:20]==head
    orders={'G12':previous['Post_G'][q]}
    for c in CELLS:
     order=sorted(head,key=lambda d:(-scores[c][q][d]['score'],d));orders[c]=order+base[20:]
    assert orders['MiniLM_topic']==base and orders['Qwen_full']==previous['fixed_Qwen20'][q]
    for name in ['G_same12','QL5']:orders[name]=[head[j] for j in sorted(range(20),key=lambda j:(-values[name][i,j],head[j]))]+base[20:]
    for name in METHODS:
     order=orders[name];assert len(order)==100 and len(set(order))==100 and set(order)==set(base) and order[20:]==base[20:]
     f.write(json.dumps({'query_id':q,'base_id':query['base_id'],'level':query['level'],'method':name,'order':order})+'\n')
  np.savez_compressed(out/'features.npz',G_same12=x)
  for n in ['rankings.jsonl.gz','features.npz']:files[f'{ds}/outputs/{n}']=sha(out/n)
 write(ROOT/'predictions_sealed.json',{'sealed_utc':datetime.now(timezone.utc).isoformat(),'files':files,'qrels_opened':False,
    'source_models_sha256':seal['models_sha256'],'old_G_source_model_sha256':sha(OLD/'source_model.json'),
    'frozen_manifest_sha256':sha(ROOT/'frozen_manifest.json'),'wall_seconds':time.monotonic()-start})


if __name__=='__main__':main()
