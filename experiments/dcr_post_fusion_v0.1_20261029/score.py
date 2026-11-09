"""After route seal: all-query point outcomes and paired full-refit intervals."""
import csv
import gzip
import time
from pathlib import Path
import numpy as np
from paths import ROOT,GATE,BUDGET,read,write,sha,verify_code
from post_math import rank_positions
METHODS=['fixed_K20','linear','Post_G','Post_GC']
CONTRASTS=[('Post_GC','Post_G'),('Post_G','fixed_K20'),('Post_GC','fixed_K20'),('linear','fixed_K20'),('Post_GC','linear')]


def verify():
    m=verify_code();s=read(ROOT/'routes_sealed.json')
    assert sha(ROOT/'frozen_manifest.json')==s['frozen_manifest_sha256']
    for n,h in s['files'].items():assert sha(ROOT/n)==h
    assert sha(ROOT/'features_sealed.json')==s['features_sealed_sha256']
    assert sha(ROOT/'fixed_source_check.json')==s['fixed_source_check_sha256']
    for n,h in read(ROOT/'features_sealed.json')['files'].items():assert sha(ROOT/n)==h
    for n,h in m['scoring_sources'].items():assert sha(Path(n))==h


def pair_credit(positive,negative,head_scores,mini_scores):
    if positive<0 or negative<0:return None
    ph,nh=positive<20,negative<20
    if ph!=nh:return float(ph)
    a,b=(head_scores[positive],head_scores[negative]) if ph else (mini_scores[positive],mini_scores[negative])
    return 1. if a>b else 0. if a<b else .5


def old_subsets(qids,own_p1,mini_p1):
    old={}
    sources={}
    for m in ['MiniLM','topK_20']:
        path=BUDGET/'ExcluIR/evaluation'/m/'per_query.csv';sources[m]=sha(path)
        old[m]={r['query_id']:float(r['positive_recall@1']) for r in csv.DictReader(path.open())}
    for i,q in enumerate(qids):assert old['MiniLM'][q]==mini_p1[i] and old['topK_20'][q]==own_p1[i]
    groups={'194_MiniLM_only':[],'1006_fixed_only':[],'both_correct':[],'both_wrong':[]}
    for q in qids:
        a,b=old['MiniLM'][q],old['topK_20'][q]
        name='both_correct' if a and b else '194_MiniLM_only' if a else '1006_fixed_only' if b else 'both_wrong'
        groups[name].append(q)
    assert len(groups['194_MiniLM_only'])==194 and len(groups['1006_fixed_only'])==1006
    return groups,sources


def main():
    started=time.monotonic();verify()
    meta=read(ROOT/'head_and_tail.json');qids=meta['qids'];pools=meta['ranked_pools'];n=len(qids)
    refs=read(GATE/'external_inputs/references.json')
    with np.load(ROOT/'features.npz',allow_pickle=False) as a:x=a['X'];raw=a['raw_scores'];canonical=a['canonical'];mini=a['MiniLM_all_scores'];tokens=a['Qwen_tokens']
    with np.load(ROOT/'prepared/plan.npz',allow_pickle=False) as a:weights=a['weights']
    fixed=rank_positions(raw[:,:,1],canonical)
    # Complete point heads and true point scores; all 1000 bootstrap top1s for all original query IDs.
    point_orders={'fixed_K20':fixed};point_scores={'fixed_K20':raw[:,:,1]}
    boot_top={name:np.empty((1000,n),dtype=np.uint8) for name in METHODS[1:]}
    parameter_counts={}
    for name in METHODS[1:]:point_orders[name]=np.empty((n,20),dtype=np.uint8);point_scores[name]=np.empty((n,20))
    for f in range(5):
        with np.load(ROOT/'fitted'/f'fold_{f}.npz',allow_pickle=False) as a:
            test=a['test_indices'];alpha=a['alpha']
            point_scores['linear'][test]=alpha[0]*x[test,:,0]+(1-alpha[0])*x[test,:,1]
            parameter_counts[str(f)]={'point_alpha':float(alpha[0])}
            for key,name in [('linear','linear'),('G','Post_G'),('GC','Post_GC')]:
                order=a[key+'_order'];point_orders[name][test]=order[0];boot_top[name][:,test]=order[1:,:,0]
                if key!='linear':
                    point_scores[name][test]=a[key+'_point_scores']
                    parameter_counts[str(f)]['point_'+key+'_lambda']=float(a[key+'_lambda'][0])
    pos=np.array([pools[i].index(refs[q]['positive_id']) if refs[q]['positive_id'] in pools[i] else -1 for i,q in enumerate(qids)])
    neg=np.array([pools[i].index(refs[q]['negative_id']) if refs[q]['negative_id'] in pools[i] else -1 for i,q in enumerate(qids)])
    point={};rows=[]
    for name in METHODS:
        order=point_orders[name];winner=order[:,0]
        p1=winner==pos;n1=winner==neg
        ranks=np.empty((n,100),dtype=np.int16)
        for i in range(n):
            full=list(order[i])+list(range(20,100));ranks[i,full]=np.arange(1,101)
            pc=pair_credit(pos[i],neg[i],point_scores[name][i],mini[i])
            rows.append({'method':name,'qid':qids[i],'positive_recall@1':int(p1[i]),'negative_entry@1':int(n1[i]),
                 'positive_recall@10':int(pos[i]>=0 and ranks[i,pos[i]]<=10),
                 'pairwise_credit':pc if pc is not None else '',
                 'positive_rank':int(ranks[i,pos[i]]) if pos[i]>=0 else 0,'negative_rank':int(ranks[i,neg[i]]) if neg[i]>=0 else 0})
        point[name]=np.stack([p1,n1],axis=1).astype(float)
    groups,sources=old_subsets(qids,point['fixed_K20'][:,0],(pos==0).astype(float))
    write(ROOT/'auxiliary_groups.json',{'groups':groups,'source_hashes':sources,'scoring_only':True})
    reports=[];fixed_p=point['fixed_K20'][:,0]
    for name in METHODS:
        rr=[r for r in rows if r['method']==name];p=point[name][:,0]
        rescued=int(((p==1)&(fixed_p==0)).sum());lost=int(((p==0)&(fixed_p==1)).sum())
        reports.append({'method':name,'queries':n,'P1_pct':100*float(p.mean()),'N1_pct':100*float(point[name][:,1].mean()),
          'positive_recall@10_pct':100*sum(r['positive_recall@10'] for r in rr)/n,
          'pairwise_credit_pct':100*float(np.mean([r['pairwise_credit'] for r in rr if r['pairwise_credit']!=''])),
          'pairwise_covered':sum(r['pairwise_credit']!='' for r in rr),'rescued_vs_fixed':rescued,'new_losses_vs_fixed':lost,'net_rescued_vs_fixed':rescued-lost,
          'Qwen_pairs':n*20,'Qwen_input_tokens':int(tokens.sum()),
          'auxiliary_P1_pct':{g:100*float(p[[qids.index(q) for q in ids]].mean()) for g,ids in groups.items()}})
    samples=np.empty((1000,len(CONTRASTS),2))
    for draw in range(1000):
        w=weights[draw];den=int(w.sum());means={}
        means['fixed_K20']=np.sum(w[:,None]*point['fixed_K20'],axis=0)/den
        for name in METHODS[1:]:
            winner=boot_top[name][draw];v=np.stack([winner==pos,winner==neg],axis=1)
            means[name]=np.sum(w[:,None]*v,axis=0)/den
        for j,(a,b) in enumerate(CONTRASTS):samples[draw,j]=100*(means[a]-means[b])
    ci=[]
    for j,(a,b) in enumerate(CONTRASTS):
        for k,metric in enumerate(['P1','N1']):
            ci.append({'method':a,'reference':b,'metric':metric,'primary':a=='Post_GC' and b=='Post_G' and metric=='P1',
               'difference_pp':100*float(point[a][:,k].mean()-point[b][:,k].mean()),
               'ci_low_pp':float(np.quantile(samples[:,j,k],.025)),'ci_high_pp':float(np.quantile(samples[:,j,k],.975)), 'draws':1000})
    with gzip.open(ROOT/'per_query.csv.gz','wt',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (ROOT/'paired_bootstrap.csv').open('x',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(ci[0]));writer.writeheader();writer.writerows(ci)
    write(ROOT/'summary.json',{'reports':reports, 'primary_GC_vs_G_transition':{
       'rescued':int(((point['Post_GC'][:,0]==1)&(point['Post_G'][:,0]==0)).sum()),
       'new_losses':int(((point['Post_GC'][:,0]==0)&(point['Post_G'][:,0]==1)).sum())},'paired_bootstrap':ci,'point_selected_parameters':parameter_counts,
       'positive_in_pool':int((pos>=0).sum()),'negative_in_pool':int((neg>=0).sum()),'positive_in_head':int(((pos>=0)&(pos<20)).sum()),
       'negative_in_head':int(((neg>=0)&(neg<20)).sum()),'both_in_head':int(((pos>=0)&(pos<20)&(neg>=0)&(neg<20)).sum()),
       'routes_sealed_sha256':sha(ROOT/'routes_sealed.json'),'CPU_score_wall_seconds':time.monotonic()-started})
    verify()


if __name__=='__main__':main()
