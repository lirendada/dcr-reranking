"""All predictions must be sealed before this process reads IFIR qrels."""
import csv
import gzip
import json
import platform
import random
import time
import numpy as np
from paths import ROOT,DATASETS,read,write,rows,sha,verify_phase
from metrics import evaluate,coverage
METHODS=['MiniLM','fixed_Qwen20','Post_G'];METRICS=['P1','nDCG20','MRR20','R10']


def main():
    started=time.monotonic();verify_phase('scoring');seal=read(ROOT/'predictions_sealed.json')
    assert sha(ROOT/'frozen_manifest.json')==seal['frozen_manifest_sha256']
    for p,h in seal['files'].items():assert sha(ROOT/p)==h
    assert sha(ROOT/'source_model.json')==seal['source_model_sha256']
    result={};all_rows=[];outcomes=[];boot_input={}
    for ds in DATASETS:
        queries=list(rows(ROOT/ds/'inputs/queries.jsonl'));qids=[r['query_id'] for r in queries]
        qrels={q:{} for q in qids}
        with (ROOT/'raw'/ds/f'{ds}-qrel.tsv').open() as f:
            for r in csv.DictReader(f,delimiter='\t'):
                q,d,rel=r['query_id'],r['corpus_id'],float(r['score'])
                # Public file also includes level0; evaluation is level>0 only.
                if q not in qrels:continue
                assert d not in qrels[q] and rel>=0;qrels[q][d]=rel
        assert all(any(rel>0 for rel in qrels[q].values()) for q in qids)
        query_metadata={r['query_id']:(r['base_id'],r['level']) for r in queries}
        pools={q:[] for q in qids}
        with (ROOT/ds/'retrieval/candidates.csv').open() as f:
            for r in csv.DictReader(f):pools[r['query_id']].append(r['document_id'])
        assert all(len(pool)==len(set(pool))==100 for pool in pools.values())
        orders={m:{} for m in METHODS}
        for r in rows(ROOT/ds/'outputs/rankings.jsonl.gz'):
            q=r['query_id'];assert (r['base_id'],r['level'])==query_metadata[q]
            assert q not in orders[r['method']]
            assert len(r['order'])==len(set(r['order']))==100 and set(r['order'])==set(pools[q])
            orders[r['method']][q]=r['order']
        assert all(set(orders[m])==set(qids) for m in METHODS)
        arrays={m:np.empty((len(qids),4)) for m in METHODS};coverage_rows=[]
        for i,r in enumerate(queries):
            q=r['query_id']
            for m in METHODS:
                order=orders[m][q];v=evaluate(order,qrels[q]);arrays[m][i]=[v[k] for k in METRICS]
                all_rows.append({'dataset':ds,'method':m,'query_id':q,'base_id':r['base_id'],'level':r['level'],**v})
            assert set(orders['Post_G'][q][:20])==set(orders['MiniLM'][q][:20])==set(orders['fixed_Qwen20'][q][:20])
            assert orders['Post_G'][q][20:]==orders['MiniLM'][q][20:]==orders['fixed_Qwen20'][q][20:]
            for k in [100,20]:coverage_rows.append({'query_id':q,'base_id':r['base_id'],'pool':'BM25_top100' if k==100 else 'MiniLM_top20',**coverage(orders['MiniLM'][q],qrels[q],k)})
            fixed_hit=arrays['fixed_Qwen20'][i,0];post_hit=arrays['Post_G'][i,0]
            outcomes.append({'dataset':ds,'query_id':q,'base_id':r['base_id'],'level':r['level'],
                'fixed_top_document':orders['fixed_Qwen20'][q][0],'post_g_top_document':orders['Post_G'][q][0],
                'fixed_P1':int(fixed_hit),'post_g_P1':int(post_hit),
                'outcome':'rescued' if post_hit>fixed_hit else 'new_loss' if post_hit<fixed_hit else 'both_positive' if post_hit else 'neither_positive'})
        fixed=arrays['fixed_Qwen20'][:,0];post=arrays['Post_G'][:,0]
        reports={m:{k:100*float(arrays[m][:,j].mean()) for j,k in enumerate(METRICS)} for m in METHODS}
        level_reports={}
        for level in sorted({r['level'] for r in queries}):
            idx=np.array([i for i,r in enumerate(queries) if r['level']==level])
            level_reports[str(level)]={'queries':len(idx),'reports':{m:{k:100*float(arrays[m][idx,j].mean()) for j,k in enumerate(METRICS)} for m in METHODS}}
        cov={}
        for pool in ['BM25_top100','MiniLM_top20']:
            rr=[r for r in coverage_rows if r['pool']==pool]
            cov[pool]={'query_mean_positive_document_recall_pct':100*sum(r['positive_document_recall'] for r in rr)/len(rr),
                      'queries_with_positive':sum(r['at_least_one_positive'] for r in rr),'query_positive_hit_pct':100*sum(r['at_least_one_positive'] for r in rr)/len(rr),
                      'positive_documents_retrieved':sum(r['positive_documents_retrieved'] for r in rr),'known_positive_documents':sum(r['known_positive_documents'] for r in rr)}
        result[ds]={'queries':len(qids),'bases':len({r['base_id'] for r in queries}),'reports_pct':reports,'level_reports':level_reports,'coverage':cov,
                    'rescued':int(((post==1)&(fixed==0)).sum()),'new_losses':int(((post==0)&(fixed==1)).sum()),'net_correct':int((post-fixed).sum())}
        bases=sorted({r['base_id'] for r in queries});bidx={b:i for i,b in enumerate(bases)}
        boot_input[ds]={'base_ids':bases,'query_base_indices':np.array([bidx[r['base_id']] for r in queries]),'difference':arrays['Post_G']-arrays['fixed_Qwen20']}
        with (ROOT/ds/'coverage_per_query.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(coverage_rows[0]));w.writeheader();w.writerows(coverage_rows)
    rng=random.Random(20261029);draws={ds:np.empty((10000,4)) for ds in DATASETS}
    plans={ds:np.zeros((10000,len(boot_input[ds]['base_ids'])),dtype=np.uint16) for ds in DATASETS}
    for draw in range(10000):
        for ds in DATASETS:
            b=boot_input[ds];count=plans[ds][draw];n=len(count)
            for _ in range(n):count[rng.randrange(n)]+=1
            weight=count[b['query_base_indices']]
            draws[ds][draw]=100*np.sum(weight[:,None]*b['difference'],axis=0)/int(weight.sum())
    macro=(draws['fiqa']+draws['nfcorpus'])/2
    ci=[]
    for ds in list(DATASETS)+['equal_weight_macro']:
        samples=macro if ds=='equal_weight_macro' else draws[ds]
        point=np.mean([100*boot_input[d]['difference'].mean(0) for d in DATASETS],axis=0) if ds=='equal_weight_macro' else 100*boot_input[ds]['difference'].mean(0)
        for j,k in enumerate(METRICS):ci.append({'dataset':ds,'metric':k,'primary':ds=='equal_weight_macro' and k=='P1','difference_pp':float(point[j]),
               'ci_low_pp':float(np.quantile(samples[:,j],.025)),'ci_high_pp':float(np.quantile(samples[:,j],.975)),'draws':10000})
    np.savez_compressed(ROOT/'bootstrap_plan.npz',**plans)
    with gzip.open(ROOT/'per_query.csv.gz','wt',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    with (ROOT/'paired_outcomes.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(outcomes[0]));w.writeheader();w.writerows(outcomes)
    with (ROOT/'paired_bootstrap.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ci[0]));w.writeheader();w.writerows(ci)
    write(ROOT/'summary.json',{'datasets':result,'paired_bootstrap':ci,'cross_dataset_equal_weight_reports_pct':{m:{k:float(np.mean([result[ds]['reports_pct'][m][k] for ds in DATASETS])) for k in METRICS} for m in METHODS},
      'bootstrap':{'RNG':'Python random.Random','seed':20261029,'draws':10000,'order':'draw outer, fiqa then nfcorpus, sorted base IDs randrange; all member queries weighted','fixed_model_no_refit':True,'plan_sha256':sha(ROOT/'bootstrap_plan.npz')},
      'units':{'per_query.csv.gz':'fractions','reports_pct':'percent','paired_bootstrap':'percentage points'},
      'judgment_scope':'Published positives; unjudged documents are not confirmed violations.',
      'CPU_scoring_wall_seconds':time.monotonic()-started,'Python':platform.python_version(),'source_model_sha256':sha(ROOT/'source_model.json')})


if __name__=='__main__':main()
