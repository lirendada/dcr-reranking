"""No-label feature calculation and three full rankings, then prediction seal."""
from datetime import datetime,timezone
import gzip
import json
import time
import numpy as np
from paths import ROOT,DATASETS,read,write,rows,sha,verify_phase,load_source_math,parser_module


def main():
    started=time.monotonic();verify_phase('input')
    sealed=read(ROOT/'model_sealed.json');assert sha(ROOT/'source_model.json')==sealed['model_sha256']
    saved=read(ROOT/'source_model.json');assert saved['lambda']==.1 and saved['dimensions']==12
    model=(np.array(saved['scale']),np.array(saved['coef']),saved['qualified_pairs'])
    math=load_source_math();parser=parser_module();files={}
    for ds in DATASETS:
        queries=list(rows(ROOT/ds/'inputs/queries.jsonl'));corpus={r['document_id']:r['text'] for r in rows(ROOT/ds/'inputs/corpus.jsonl')}
        sources={};score={}
        for name in ['MiniLM','Qwen']:
            p=ROOT/ds/name/'scores.jsonl.gz';manifest=read(p.parent/'manifest.json');assert sha(p)==manifest['scores_sha256']
            score[name]={}
            for r in rows(p):
                q,d=r['query_id'],r['document_id'];pool=score[name].setdefault(q,{})
                assert d not in pool;pool[d]=r
            sources[name]=manifest['scores_sha256']
        output=ROOT/ds/'outputs';output.mkdir(exist_ok=False)
        xs=[];heads=[];qids=[]
        with gzip.open(output/'rankings.jsonl.gz','wt') as f:
            for query in queries:
                q=query['query_id'];mini=score['MiniLM'][q]
                # Only100 already scored candidates; never expand/inject candidate endpoints.
                assert len(mini)==100
                base=sorted(mini,key=lambda d:(-mini[d]['score'],d));head=base[:20]
                qw=score['Qwen'][q];assert set(qw)==set(head)
                assert all(qw[d]['doc_prefix_end_char']==mini[d]['doc_prefix_end_char'] for d in head)
                tokens=[parser.tokenize(corpus[d][:mini[d]['doc_prefix_end_char']],remove_stop=True) for d in head]
                ms=np.array([mini[d]['score'] for d in head]);qs=np.array([qw[d]['score'] for d in head])
                x=math.doc_features(ms,qs,tokens,parser.tokenize(query['topic'],remove_stop=True),parser.tokenize(query['request'],remove_stop=True),[])[:,:12]
                x[:,2]=np.arange(1,21)/20
                fixed=sorted(range(20),key=lambda i:(-qs[i],head[i]))
                for rank,i in enumerate(fixed):x[i,3]=(rank+1)/20
                post=math.ridge_scores(x[None,:,:],model)[0]
                assert np.isfinite(post).all()
                final=sorted(range(20),key=lambda i:(-post[i],head[i]))
                orders={'MiniLM':base,'fixed_Qwen20':[head[i] for i in fixed]+base[20:],'Post_G':[head[i] for i in final]+base[20:]}
                for name,order in orders.items():
                    assert len(order)==100 and set(order)==set(base)
                    if name!='MiniLM':assert order[20:]==base[20:]
                    f.write(json.dumps({'query_id':q,'base_id':query['base_id'],'level':query['level'],'method':name,'order':order})+'\n')
                xs.append(x);heads.append(head);qids.append(q)
        np.savez_compressed(output/'features.npz',X=np.array(xs))
        write(output/'feature_metadata.json',{'qids':qids,'heads':heads,'columns':math.G_COLUMNS,'source_model_sha256':sealed['model_sha256'],
           'source_score_hashes':sources,'C_features_used':False,'reference_inputs_opened':False})
        for name in ['rankings.jsonl.gz','features.npz','feature_metadata.json']:files[f'{ds}/outputs/{name}']=sha(output/name)
        for name in ['inputs/queries.jsonl','retrieval/candidates.csv']:files[f'{ds}/{name}']=sha(ROOT/ds/name)
    write(ROOT/'predictions_sealed.json',{'sealed_utc':datetime.now(timezone.utc).isoformat(),'files':files,
       'source_model_sha256':sha(ROOT/'source_model.json'),'frozen_manifest_sha256':sha(ROOT/'frozen_manifest.json'),
       'qrels_opened':False,'CPU_wall_seconds':time.monotonic()-started})


if __name__=='__main__':main()
