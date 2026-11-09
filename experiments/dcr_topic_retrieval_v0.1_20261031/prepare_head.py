"""Select topic MiniLM head20 and align all four inputs without opening references."""
import csv
import gzip
import hashlib
import json
import time
from config import (ROOT,QUERY_PATHS,CORPUS_PATH,CE_MODEL,STRONG_QWEN,CELLS,CELL_CONFIG,
                    MAX_LENGTH,EXPECTED_QUERIES,HEAD_SIZE,sha,read,write,rows,verify_code,verify_sources)
from model_contract import original_template,length_statistics


def main():
    started=time.monotonic();manifest=verify_code();verify_sources(manifest,'input')
    score_path=ROOT/'selection/MiniLM_topic/scores.jsonl.gz'
    receipt=read(score_path.parent/'receipt.json')
    assert receipt['exit_code']==0 and receipt['pairs']==345200 and receipt['score_sha256']==sha(score_path)
    pools={};original={}
    for r in rows(score_path):
        q,d=r['query_id'],r['document_id'];assert (q,d) not in original
        original[q,d]=r;pools.setdefault(q,[]).append(d)
    candidates={}
    for r in csv.DictReader((ROOT/'external_retrieval/candidates.csv').open()):candidates.setdefault(r['query_id'],[]).append(r['document_id'])
    qids=sorted(pools);assert len(qids)==EXPECTED_QUERIES and set(pools)==set(candidates)
    for q in qids:
        assert len(pools[q])==len(set(pools[q]))==100 and set(pools[q])==set(candidates[q])
        pools[q].sort(key=lambda d:(-original[q,d]['score'],d))
    write(ROOT/'head_and_tail.json',{'qids':qids,'ranked_pools':[pools[q] for q in qids],'selection_scores_sha256':sha(score_path),'reference_files_opened':[]})
    query={v:{r['query_id']:r['text'] for r in rows(path)} for v,path in QUERY_PATHS.items()}
    selected={d for q in qids for d in pools[q][:20]}
    corpus={r['document_id']:r['text'] for r in rows(CORPUS_PATH) if r['document_id'] in selected}
    from transformers import AutoTokenizer
    ce=AutoTokenizer.from_pretrained(CE_MODEL,local_files_only=True,trust_remote_code=False)
    qw=AutoTokenizer.from_pretrained(read(STRONG_QWEN/'model_assets.json')['directory'],local_files_only=True,trust_remote_code=False)
    template=original_template();out=ROOT/'prepared';out.mkdir(exist_ok=False)
    lengths={c:[] for c in CELLS};selected_rows=[];records=[];overflow=[]
    for n,q in enumerate(qids,1):
        bodies=[];metadata=[]
        for j,d in enumerate(pools[q][:20],1):
            r=original[q,d];end=r['doc_prefix_end_char'];body=corpus[d][:end]
            assert 0<=end<=len(corpus[d])
            bodies.append(body);metadata.append({'query_id':q,'document_id':d,'head_rank':j,'doc_prefix_end_char':end,'body_prefix_sha256':hashlib.sha256(body.encode()).hexdigest()})
            selected_rows.append(r)
        cet={v:[len(ids) for ids in ce([query[v][q]]*20,bodies,truncation=False,padding=False)['input_ids']] for v in ['topic','full']}
        qwt={v:[len(ids) for ids in qw([template(query[v][q],b) for b in bodies],add_special_tokens=False,truncation=False,padding=False)['input_ids']] for v in ['topic','full']}
        for j,r in enumerate(metadata):
            tokens={c:(cet if cfg['model']=='MiniLM' else qwt)[cfg['input_variant']][j] for c,cfg in CELL_CONFIG.items()}
            assert tokens['MiniLM_topic']==original[q,r['document_id']]['input_tokens']
            for c,count in tokens.items():
                lengths[c].append(count)
                if count>MAX_LENGTH[CELL_CONFIG[c]['model']]:overflow.append({'cell':c,'query_id':q,'document_id':r['document_id'],'tokens':count})
            r['token_lengths']=tokens;records.append(r)
        if n%250==0:print(json.dumps({'queries':n,'seconds':time.monotonic()-started}),flush=True)
    write(out/'token_preflight.json',{'token_statistics':{c:length_statistics(v) for c,v in lengths.items()},'overflows':overflow,'model_loaded':False,'references_opened':False,'prefix_rule':'Exact selection-stage shared prefix; no retruncation.'})
    assert not overflow,overflow[:3]
    with gzip.open(out/'pairs.jsonl.gz','xt') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    target=ROOT/'cells/MiniLM_topic';target.mkdir(parents=True,exist_ok=False)
    with gzip.open(target/'scores.jsonl.gz','xt') as f:
        for r in selected_rows:f.write(json.dumps(r)+'\n')
    write(target/'receipt.json',{'cell':'MiniLM_topic','pairs':69040,'queries':3452,'score_sha256':sha(target/'scores.jsonl.gz'),'source_score_sha256':sha(score_path),'input_tokens_total':sum(lengths['MiniLM_topic']),'exit_code':0,'new_model_pairs':0,'inference_rerun':False,'selection_pairs':345200,'references_read':False})
    files=['prepared/pairs.jsonl.gz','prepared/token_preflight.json','head_and_tail.json','cells/MiniLM_topic/scores.jsonl.gz','cells/MiniLM_topic/receipt.json']
    write(out/'summary.json',{'queries':3452,'pairs':69040,'lengths':{c:length_statistics(v) for c,v in lengths.items()},'outputs':{f:sha(ROOT/f) for f in files},'config_sha256':sha(ROOT/'config.py'),'references_opened':False,'new_model_calls':0,'CPU_wall_seconds':time.monotonic()-started})
    verify_code();verify_sources(manifest,'input')


if __name__=='__main__':main()
