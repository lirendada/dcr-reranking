"""CPU tokenizer-only four-cell alignment on the exact historical selected body strings."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from config import (ROOT,SOURCE_POST,GATE,STRONG_QWEN,CE_MODEL,QUERY_PATHS,CORPUS_PATH,HEAD_PATH,OLD_SCORES,
                    EXPECTED_QUERIES,HEAD_SIZE,EXPECTED_NEW_PAIRS,CELL_CONFIG,CELLS,MAX_LENGTH,read,write,sha,rows)


def validate_historical_pair(mini: dict, qwen: dict, body: str) -> tuple[int, str]:
    assert (mini['query_id'],mini['document_id'])==(qwen['query_id'],qwen['document_id'])
    end=mini['doc_prefix_end_char']
    assert isinstance(end,int) and not isinstance(end,bool) and 0<=end<=len(body)
    assert qwen['doc_prefix_end_char']==end
    assert all(math.isfinite(r['score']) and isinstance(r['input_tokens'],int) and r['input_tokens']>0 for r in [mini,qwen])
    return end,hashlib.sha256(body[:end].encode()).hexdigest()


def original_template():
    sys.path.insert(0,str(GATE))
    spec=importlib.util.spec_from_file_location('old_frozen_preparation',GATE/'prepare_external_neural.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.qwen_text


def verify_model_assets() -> dict:
    verified={}
    ce_manifest=read(CE_MODEL/'download_manifest.json')
    ce_files=[]
    for item in ce_manifest['files']:
        path=CE_MODEL/item['name'];actual=sha(path)
        assert actual==item['sha256'],str(path)
        ce_files.append({'name':item['name'],'bytes':path.stat().st_size,'algorithm':'sha256','digest':actual})
    verified['MiniLM']={'directory':str(CE_MODEL),'revision':'233902d25c440f23af6f7d6e94d2946bac0bee0a',
        'download_manifest_sha256':sha(CE_MODEL/'download_manifest.json'),'files':ce_files}
    assets=read(STRONG_QWEN/'model_assets.json');official=read(STRONG_QWEN/'official_model_metadata.json')
    assert assets['revision']=='22e683669bc0f0bd69640a1354a6d0aebcfeede5'
    entries={r['rfilename']:r for r in official['siblings']};files=[]
    for item in assets['files']:
        path=Path(assets['directory'])/item['name'];entry=entries[item['name']]
        size=path.stat().st_size;assert size==entry['size']
        if entry.get('lfs'):
            algorithm='sha256';actual=sha(path);expected=entry['lfs']['sha256']
        else:
            algorithm='git_blob_sha1';h=hashlib.sha1(b'blob '+str(size).encode()+b'\0')
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1048576),b''):h.update(block)
            actual=h.hexdigest();expected=entry['blobId']
        assert actual==expected,str(path)
        files.append({'name':item['name'],'bytes':size,'algorithm':algorithm,'digest':actual})
    verified['Qwen']={'directory':assets['directory'],'revision':assets['revision'],
        'official_metadata_sha256':sha(STRONG_QWEN/'official_model_metadata.json'),'files':files}
    return verified


def length_statistics(values: list[int]) -> dict:
    ordered=sorted(values)
    return {'pairs':len(values),'min':ordered[0],'max':ordered[-1],'total':sum(values),
            'p50':ordered[len(ordered)//2],'p95':ordered[int(len(ordered)*.95)]}


def prepare() -> None:
    assert str(ROOT).startswith('/path/to/remote/lw_experiments/')
    assert os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    started=time.monotonic()
    source_paths=[HEAD_PATH,CORPUS_PATH,*QUERY_PATHS.values(),*OLD_SCORES.values(),GATE/'prepare_external_neural.py',STRONG_QWEN/'model_assets.json',STRONG_QWEN/'official_model_metadata.json',CE_MODEL/'download_manifest.json']
    source_hashes={str(p):sha(p) for p in source_paths}
    meta=read(HEAD_PATH);qids=meta['qids'];pools=meta['ranked_pools']
    assert len(qids)==EXPECTED_QUERIES and qids==sorted(qids) and len(set(qids))==len(qids)
    assert all(len(p)==100 and len(set(p))==100 for p in pools)
    head={q:p[:HEAD_SIZE] for q,p in zip(qids,pools)}
    query={variant:{r['query_id']:r['text'] for r in rows(path)} for variant,path in QUERY_PATHS.items()}
    assert all(set(r)==set(qids) for r in query.values())
    mini=defaultdict(dict)
    for r in rows(OLD_SCORES['MiniLM_topic']):
        q,d=r['query_id'],r['document_id'];assert d not in mini[q];mini[q][d]=r
    assert set(mini)==set(qids)
    for q,pool in zip(qids,pools):
        assert set(mini[q])==set(pool)
        assert sorted(pool,key=lambda d:(-mini[q][d]['score'],d))==pool
    selected={(q,d) for q in qids for d in head[q]}
    qwen={}
    for r in rows(OLD_SCORES['Qwen_full']):
        key=(r['query_id'],r['document_id'])
        if key not in selected:continue
        assert key not in qwen;qwen[key]=r
    assert len(qwen)==EXPECTED_NEW_PAIRS
    needed={d for q,d in selected}
    corpus={r['document_id']:r['text'] for r in rows(CORPUS_PATH) if r['document_id'] in needed}
    assert set(corpus)==needed
    model_assets=verify_model_assets()
    from transformers import AutoTokenizer
    ce=AutoTokenizer.from_pretrained(CE_MODEL,local_files_only=True,trust_remote_code=False)
    qw=AutoTokenizer.from_pretrained(read(STRONG_QWEN/'model_assets.json')['directory'],local_files_only=True,trust_remote_code=False)
    qwen_text=original_template()
    out=ROOT/'prepared';out.mkdir(exist_ok=False)
    for cell in OLD_SCORES:(ROOT/'cells'/cell).mkdir(parents=True,exist_ok=False)
    reused={cell:[] for cell in OLD_SCORES};lengths={cell:[] for cell in CELLS};overflows=[]
    aligned=0;records=[];alignment_rows=[]
    # No truncation/cropping operation exists here: every body is sliced only at saved end.
    for q in qids:
        bodies=[];base_records=[]
        for rank,d in enumerate(head[q],1):
            mr,qr=mini[q][d],qwen[q,d];end,digest=validate_historical_pair(mr,qr,corpus[d])
            bodies.append(corpus[d][:end]);base_records.append({'query_id':q,'document_id':d,'head_rank':rank,
                'doc_prefix_end_char':end,'body_prefix_sha256':digest})
        ce_tokens={variant:[len(ids) for ids in ce([query[variant][q]]*HEAD_SIZE,bodies,truncation=False,padding=False)['input_ids']] for variant in ['topic','full']}
        qw_tokens={variant:[len(ids) for ids in qw([qwen_text(query[variant][q],body) for body in bodies],add_special_tokens=False,truncation=False,padding=False)['input_ids']] for variant in ['topic','full']}
        for i,r in enumerate(base_records):
            d=r['document_id'];tokens={cell:(ce_tokens if cfg['model']=='MiniLM' else qw_tokens)[cfg['input_variant']][i] for cell,cfg in CELL_CONFIG.items()}
            assert tokens['MiniLM_topic']==mini[q][d]['input_tokens'],(q,d,'old MiniLM length mismatch')
            assert tokens['Qwen_full']==qwen[q,d]['input_tokens'],(q,d,'old Qwen length mismatch')
            for cell,count in tokens.items():
                lengths[cell].append(count)
                if count>MAX_LENGTH[CELL_CONFIG[cell]['model']]:overflows.append({'cell':cell,'query_id':q,'document_id':d,'tokens':count})
            r['token_lengths']=tokens;records.append(r)
            for cell,row in [('MiniLM_topic',mini[q][d]),('Qwen_full',qwen[q,d])]:reused[cell].append(row)
            alignment_rows.append({'query_id':q,'document_id':d,'head_rank':r['head_rank'],'doc_prefix_end_char':r['doc_prefix_end_char'],
                'body_prefix_sha256':r['body_prefix_sha256'],**tokens})
            aligned+=1
        if aligned%(HEAD_SIZE*200)==0:print(json.dumps({'prepared_queries':aligned//HEAD_SIZE,'pairs':aligned,'wall_seconds':time.monotonic()-started}),flush=True)
    assert aligned==EXPECTED_NEW_PAIRS
    # Persist the complete overflow observation before refusing GPU preparation.
    write(out/'token_preflight.json',{'token_statistics':{c:length_statistics(v) for c,v in lengths.items()},
       'overflows':overflows,'silent_retruncation':False,'old_cells_pair_aligned':aligned,'model_loaded':False,'references_opened':False})
    assert not overflows,('Input overflow; preserve old body and revise protocol explicitly',len(overflows))
    with gzip.open(out/'pairs.jsonl.gz','xt') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    with gzip.open(out/'old_pair_alignment.csv.gz','wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(alignment_rows[0]));writer.writeheader();writer.writerows(alignment_rows)
    outputs={}
    for cell,records in reused.items():
        target=ROOT/'cells'/cell/'scores.jsonl.gz'
        with gzip.open(target,'xt') as f:
            for r in records:f.write(json.dumps(r)+'\n')
        receipt={'cell':cell,**CELL_CONFIG[cell],'pairs':len(records),'queries':EXPECTED_QUERIES,'input_tokens_total':sum(r['input_tokens'] for r in records),
            'score_sha256':sha(target),'source_score_sha256':source_hashes[str(OLD_SCORES[cell])],
            'exit_code':0,'wall_seconds':None,'new_model_pairs':0,'model_inference_rerun':False,
            'original_fields_preserved':True,'prefix_and_token_alignment_verified':True}
        write(target.parent/'receipt.json',receipt)
        for p in [target,target.parent/'receipt.json']:outputs[str(p.relative_to(ROOT))]=sha(p)
    for name in ['pairs.jsonl.gz','old_pair_alignment.csv.gz','token_preflight.json']:outputs['prepared/'+name]=sha(out/name)
    assert all(sha(p)==h for p,h in [(path,source_hashes[str(path)]) for path in source_paths])
    import transformers,tokenizers
    write(out/'summary.json',{'prepared_utc':datetime.now(timezone.utc).isoformat(),'queries':EXPECTED_QUERIES,'pairs':aligned,'head_size':HEAD_SIZE,
       'cells':CELL_CONFIG,'lengths':{c:length_statistics(v) for c,v in lengths.items()},'old_cells_alignment':{'MiniLM_topic':aligned,'Qwen_full':aligned},
       'prefix_source':'historical MiniLM_topic doc_prefix_end_char on corpus.text; exactly equal to historical Qwen_full',
       'sources':source_hashes,'outputs':outputs,'config_sha256':sha(ROOT/'config.py'),
       'code':{name:sha(ROOT/name) for name in ['prepare_inputs.py','resource_preflight.py']},'assets':model_assets,
       'tokenizer_environment':{'transformers':transformers.__version__,'tokenizers':tokenizers.__version__},
       'references_opened':False,'new_model_calls':0,'CPU_wall_seconds':time.monotonic()-started})


if __name__=='__main__':prepare()
