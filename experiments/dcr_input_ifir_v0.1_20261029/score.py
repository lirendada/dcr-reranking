"""Observed IFIR replay: fixed-model24 P1 intervals; other metrics complete point tables."""
import csv,gzip,importlib.util,time,random
import numpy as np
from config import ROOT,OLD,DATASETS,METHODS,METRICS,CELLS,CONTRASTS,FUSION,read,write,rows,sha,check_phase


def save_csv(path,records):
 with (gzip.open(path,'xt',newline='') if path.suffix=='.gz' else path.open('x',newline='')) as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def main():
 started=time.monotonic();check_phase('scoring');seal=read(ROOT/'predictions_sealed.json')
 assert not seal['qrels_opened'] and seal['frozen_manifest_sha256']==sha(ROOT/'frozen_manifest.json')
 for p,h in seal['files'].items():assert sha(ROOT/p)==h
 spec=importlib.util.spec_from_file_location('frozen_IFIR_metrics',OLD/'metrics.py');metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)
 oldsummary=read(OLD/'summary.json');result={};points={};meta={};per=[];outcomes=[]
 for ds in DATASETS:
  queries=list(rows(OLD/ds/'inputs/queries.jsonl'));qids=[q['query_id'] for q in queries];qr={q:{} for q in qids}
  with (OLD/'raw'/ds/f'{ds}-qrel.tsv').open() as f:
   for r in csv.DictReader(f,delimiter='\t'):
    if r['query_id'] in qr:
     pool=qr[r['query_id']];assert r['corpus_id'] not in pool;pool[r['corpus_id']]=float(r['score'])
  assert all(any(v>0 for v in qr[q].values()) for q in qids)
  orders={m:{} for m in METHODS}
  for r in rows(ROOT/ds/'outputs/rankings.jsonl.gz'):
   assert r['query_id'] not in orders[r['method']];orders[r['method']][r['query_id']]=r['order']
  arrays=np.empty((len(METHODS),len(qids),4))
  for j,m in enumerate(METHODS):
   assert set(orders[m])==set(qids)
   for i,q in enumerate(qids):
    v=metric.evaluate(orders[m][q],qr[q]);arrays[j,i]=[v[k] for k in METRICS]
    per.append({'dataset':ds,'method':m,'query_id':q,'base_id':queries[i]['base_id'],'level':queries[i]['level'],**v})
  oldrows={}
  for r in rows(OLD/ds/'outputs/rankings.jsonl.gz'):oldrows[r['method'],r['query_id']]=r['order']
  for new,old in [('MiniLM_topic','MiniLM'),('Qwen_full','fixed_Qwen20'),('G12','Post_G')]:
   for i,q in enumerate(qids):
    assert orders[new][q]==oldrows[old,q]
    prior=metric.evaluate(oldrows[old,q],qr[q]);assert np.array_equal(arrays[METHODS.index(new),i],[prior[k] for k in METRICS])
   for k in METRICS:assert abs(float(arrays[METHODS.index(new),:,METRICS.index(k)].mean()*100)-oldsummary['datasets'][ds]['reports_pct'][old][k])<1e-10
  report={m:{k:100*float(arrays[j,:,c].mean()) for c,k in enumerate(METRICS)} for j,m in enumerate(METHODS)}
  level={}
  for l in sorted({q['level'] for q in queries}):
   ii=[i for i,q in enumerate(queries) if q['level']==l];level[str(l)]={'queries':len(ii),'reports_pct':{m:{k:100*float(arrays[j,ii,c].mean()) for c,k in enumerate(METRICS)} for j,m in enumerate(METHODS)}}
  comparisons=[('MiniLM_full','MiniLM_topic'),('Qwen_full','Qwen_topic'),*FUSION];trans={}
  for a,b in comparisons:
   v=arrays[METHODS.index(a),:,0];u=arrays[METHODS.index(b),:,0];res=(v==1)&(u==0);loss=(v==0)&(u==1)
   trans[a+'-'+b]={'rescued':int(res.sum()),'new_losses':int(loss.sum()),'net':int(res.sum()-loss.sum())}
   for i,q in enumerate(qids):outcomes.append({'dataset':ds,'comparison':a+'-'+b,'query_id':q,'base_id':queries[i]['base_id'],'rescued':int(res[i]),'new_loss':int(loss[i])})
  result[ds]={'queries':len(qids),'reports_pct':report,'level_reports':level,'transitions':trans,'coverage':oldsummary['datasets'][ds]['coverage'],'old_G_fixed_all_query_reproduction':True}
  points[ds]=arrays;meta[ds]=queries
 # Frozen draw order: draw outer; FiQA then NF; each sorted base list sampled its original size.
 names=list(CONTRASTS)+[a+'-'+b for a,b in FUSION];samples={d:np.empty((10000,len(names))) for d in DATASETS};rng=random.Random(20261029)
 for draw in range(10000):
  for ds in DATASETS:
   qs=meta[ds];bases=sorted({q['base_id'] for q in qs});index={b:i for i,b in enumerate(bases)};count=np.zeros(len(bases),dtype=np.int16)
   for _ in bases:count[rng.randrange(len(bases))]+=1
   weights=np.array([count[index[q['base_id']]] for q in qs]);means=np.sum(points[ds][:,:,0]*weights[None,:],axis=1)/int(weights.sum())
   vals=[sum(v*means[i] for i,v in enumerate(factor)) for factor in CONTRASTS.values()]+[means[METHODS.index(a)]-means[METHODS.index(b)] for a,b in FUSION]
   samples[ds][draw]=100*np.array(vals)
 ci=[]
 for ds in list(DATASETS)+['equal_weight_macro']:
  arrays=(points['fiqa'][:,:,0].mean(1)+points['nfcorpus'][:,:,0].mean(1))/2 if ds=='equal_weight_macro' else points[ds][:,:,0].mean(1)
  point=[sum(v*arrays[i] for i,v in enumerate(factor)) for factor in CONTRASTS.values()]+[arrays[METHODS.index(a)]-arrays[METHODS.index(b)] for a,b in FUSION]
  draws=(samples['fiqa']+samples['nfcorpus'])/2 if ds=='equal_weight_macro' else samples[ds]
  for j,n in enumerate(names):
   lo,hi=np.quantile(draws[:,j],[.025,.975]);ci.append({'dataset':ds,'comparison':n,'metric':'P1','primary':ds=='equal_weight_macro' and n=='input_effect_interaction','difference_pp':100*float(point[j]),'ci_low_pp':float(lo),'ci_high_pp':float(hi),'draws':10000})
 assert len(ci)==24
 save_csv(ROOT/'paired_bootstrap.csv',ci);save_csv(ROOT/'per_query.csv.gz',per);save_csv(ROOT/'paired_outcomes.csv',outcomes)
 write(ROOT/'summary.json',{'datasets':result,'paired_bootstrap':ci,'equal_weight_macro_reports_pct':{m:{k:float(np.mean([result[d]['reports_pct'][m][k] for d in DATASETS])) for k in METRICS} for m in METHODS},
    'bootstrap':{'draws':10000,'seed':20261029,'RNG':'Python random.Random','fixed_model_no_refit':True,'order':'draw then FiQA,NF then sorted-base randrange'},
    'wall_seconds':time.monotonic()-started,'judgment_scope':'published positives; unjudged not confirmed violations','observed_data_role':'development-exposed cross-collection replay'})


if __name__=='__main__':main()
