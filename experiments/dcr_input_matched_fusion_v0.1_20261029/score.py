"""All-query score after seal; old G/QL5 point+1000draw scores rebuilt and matched."""
import csv,gzip,time
import numpy as np
from config import ROOT,POST,ATTR,REF,METHODS,METRICS,COMPARISONS,read,write,sha,check_phase,math_core


def indicators(order,pos,neg):
 p=order[:,0]==pos;n=order[:,0]==neg
 return np.stack([p,n,~(p|n),(order[:,:10]==pos[:,None]).any(1)],axis=1)


def transitions(a,b):
 return {'rescued':int((~a&b).sum()),'new_losses':int((a&~b).sum()),'net':int(b.sum()-a.sum())}


def main():
 started=time.monotonic();check_phase('scoring');check_phase('reuse');seal=read(ROOT/'predictions_sealed.json')
 assert seal['frozen_manifest_sha256']==sha(ROOT/'frozen_manifest.json') and not seal['scoring_references_opened']
 for n,h in seal['files'].items():assert sha(ROOT/n)==h,n
 meta=read(POST/'head_and_tail.json');qids=meta['qids'];pools=meta['ranked_pools'];refs=read(REF);pos=[];neg=[]
 for q,pool in zip(qids,pools):
  head=pool[:20];p,n=refs[q]['positive_id'],refs[q]['negative_id'];pos.append(head.index(p) if p in head else -1);neg.append(head.index(n) if n in head else -1)
 pos,neg=np.array(pos),np.array(neg);m=math_core()
 with np.load(POST/'features.npz',allow_pickle=False) as z:canonical=z['canonical'];raw=z['raw_scores']
 with np.load(POST/'prepared/plan.npz',allow_pickle=False) as z:plan=z['weights']
 with np.load(ATTR/'metrics_draws.npz',allow_pickle=False) as z:old_values=z['values'];old_methods=z['methods'].tolist();assert z['metrics'].tolist()==list(METRICS)
 fixed=m.rank(raw[:,:,1],canonical);fitted=[];g=[];ql=[]
 for f in range(5):
  with np.load(ROOT/'fitted'/f'fold_{f}.npz',allow_pickle=False) as z:fitted.append({k:z[k] for k in ['test_indices','order','lambda']})
  with np.load(POST/'fitted'/f'fold_{f}.npz',allow_pickle=False) as z:g.append(z['G_order']);assert np.array_equal(z['test_indices'],fitted[-1]['test_indices'])
  with np.load(ATTR/'fitted'/f'fold_{f}.npz',allow_pickle=False) as z:ql.append(z['QL5_order']);assert np.array_equal(z['test_indices'],fitted[-1]['test_indices'])
 values=np.empty((1001,4,4));point={}
 for iteration in range(1001):
  orders={'fixed':fixed}
  for name in ['G12','G_same12','QL5']:
   order=np.empty_like(fixed)
   for f in range(5):order[fitted[f]['test_indices']]=(g[f] if name=='G12' else ql[f] if name=='QL5' else fitted[f]['order'])[iteration]
   orders[name]=order
  weights=np.ones(len(qids)) if iteration==0 else plan[iteration-1].astype(float)
  for j,name in enumerate(METHODS):
   values[iteration,j]=np.sum(weights[:,None]*indicators(orders[name],pos,neg),axis=0)/weights.sum()
   if name in ['fixed','G12','QL5']:assert np.array_equal(values[iteration,j],old_values[iteration,old_methods.index(name)]),(name,iteration)
  if iteration==0:point={n:o.copy() for n,o in orders.items()}
 np.savez_compressed(ROOT/'metrics_draws.npz',values=values,methods=np.array(METHODS),metrics=np.array(METRICS))
 np.savez_compressed(ROOT/'point_orders.npz',orders=np.stack([point[n] for n in METHODS]),methods=np.array(METHODS))
 reports=[];points={n:indicators(point[n],pos,neg) for n in METHODS}
 for j,name in enumerate(METHODS):reports.append({'method':name,'queries':3452,**{k+'_pct':100*float(values[0,j,c]) for c,k in enumerate(METRICS)},'vs_fixed':transitions(points['fixed'][:,0],points[name][:,0])})
 ci=[];transition={}
 for a,b in COMPARISONS:
  transition[a+'-'+b]=transitions(points[b][:,0],points[a][:,0])
  for metric in ['P1','N1']:
   j=METRICS.index(metric);v=100*(values[:,METHODS.index(a),j]-values[:,METHODS.index(b),j]);lo,hi=np.quantile(v[1:],[.025,.975])
   ci.append({'comparison':a+'-'+b,'metric':metric,'primary':(a,b,metric)==('G_same12','QL5','P1'),'difference_pp':float(v[0]),'ci_low_pp':float(lo),'ci_high_pp':float(hi),'draws':1000})
 with (ROOT/'paired_bootstrap.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(ci[0]));w.writeheader();w.writerows(ci)
 rows=[]
 for name in METHODS:
  for i,q in enumerate(qids):rows.append({'qid':q,'method':name,'top_document':pools[i][point[name][i,0]],**{k:int(points[name][i,j]) for j,k in enumerate(METRICS)}})
 with gzip.open(ROOT/'per_query.csv.gz','xt',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 write(ROOT/'summary.json',{'reports':reports,'comparisons':ci,'transitions':transition,'old_G_QL5_all1001_draws_exact_reproduction':True,
  'point_lambdas':[float(z['lambda'][0]) for z in fitted],'new_neural_pairs':0,'wall_seconds':time.monotonic()-started,
  'transfer_models_sha256':sha(ROOT/'transfer_models.json'),'source_model_sealed_before_A_score':True,'predictions_sealed_sha256':sha(ROOT/'predictions_sealed.json')})


if __name__=='__main__':main()
