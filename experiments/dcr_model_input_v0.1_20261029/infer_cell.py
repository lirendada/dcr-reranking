"""One released new cell, exact historical body strings and unchanged model loader."""
import argparse
from datetime import datetime,timezone
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from config import (ROOT,GATE,STRONG_QWEN,CE_MODEL,CE_LOADER,GPU_LOCK,QUERY_PATHS,CORPUS_PATH,CELL_CONFIG,NEW_CELLS,
    EXPECTED_QUERIES,HEAD_SIZE,EXPECTED_NEW_PAIRS,BATCH_SIZE,MAX_LENGTH,read,write,sha,rows,verify_code,verify_sources)
from prepare_inputs import original_template


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--cell',choices=NEW_CELLS,required=True);args=parser.parse_args()
    manifest=verify_code();verify_sources(manifest,'input');verify_sources(manifest,'loader')
    release=read(ROOT/'parent_release.json');assert release['frozen_manifest_sha256']==sha(ROOT/'frozen_manifest.json')
    assert os.environ.get('DCR_GPU_LEASE')=='1' and Path('/proc/self/fd/9').resolve()==GPU_LOCK
    assert os.environ.get('TOKENIZERS_PARALLELISM')=='false' and os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
    summary=read(ROOT/'prepared/summary.json');assert summary['config_sha256']==sha(ROOT/'config.py')
    for rel,h in summary['outputs'].items():assert sha(ROOT/rel)==h,rel
    cfg=CELL_CONFIG[args.cell];model_name=cfg['model'];variant=cfg['input_variant']
    queries={r['query_id']:r['text'] for r in rows(QUERY_PATHS[variant])}
    corpus={r['document_id']:r['text'] for r in rows(CORPUS_PATH)}
    out=ROOT/'cells'/args.cell;out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();started_utc=datetime.now(timezone.utc).isoformat()
    if model_name=='MiniLM':
        spec=importlib.util.spec_from_file_location('original_ce_loader',CE_LOADER);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        torch,tokenizer,model,load_seconds=old.load_model(CE_MODEL,'cuda',4)
    else:
        sys.path.insert(0,str(STRONG_QWEN));import qwen as old
        torch,tokenizer,model,load_seconds=old.load_model()
    import transformers
    expected_transformers='4.49.0' if model_name=='MiniLM' else '4.51.3'
    assert transformers.__version__==expected_transformers
    qwen_text=original_template();batch_size=BATCH_SIZE[model_name]
    yes_id=no_id=None
    if model_name=='Qwen':
        yes_id,no_id=[tokenizer.convert_tokens_to_ids(t) for t in ['yes','no']];assert yes_id!=no_id
    count=0;tokens_total=0;stats=[];inference_started=time.monotonic();max_actual_batch=0
    with gzip.open(out/'scores.jsonl.gz','xt') as target:
        def infer_query(batch_rows):
            nonlocal count,tokens_total,max_actual_batch
            assert len(batch_rows)==HEAD_SIZE and len({r['document_id'] for r in batch_rows})==HEAD_SIZE
            assert len({r['query_id'] for r in batch_rows})==1
            tick=time.monotonic()
            for offset in range(0,len(batch_rows),batch_size):
                batch=batch_rows[offset:offset+batch_size];max_actual_batch=max(max_actual_batch,len(batch))
                bodies=[]
                for r in batch:
                    body=corpus[r['document_id']][:r['doc_prefix_end_char']]
                    assert hashlib.sha256(body.encode()).hexdigest()==r['body_prefix_sha256']
                    bodies.append(body)
                qtexts=[queries[r['query_id']] for r in batch]
                if model_name=='MiniLM':
                    encoded=tokenizer(qtexts,bodies,truncation=False,padding='max_length',max_length=MAX_LENGTH[model_name],return_tensors='pt')
                    assert encoded['input_ids'].shape[1]==512
                else:
                    encoded=tokenizer([qwen_text(q,body) for q,body in zip(qtexts,bodies)],add_special_tokens=False,truncation=False,
                        padding=True,pad_to_multiple_of=8,return_tensors='pt')
                    assert encoded['input_ids'].shape[1]<=MAX_LENGTH[model_name]
                lengths=encoded['attention_mask'].sum(1).tolist();assert lengths==[r['token_lengths'][args.cell] for r in batch]
                with torch.inference_mode():
                    tensors={k:v.to('cuda') for k,v in encoded.items()}
                    if model_name=='MiniLM':values=model(**tensors).logits.squeeze(-1).cpu().tolist();extras=[{} for _ in batch]
                    else:
                        logits=model(**tensors,use_cache=False,logits_to_keep=1).logits[:,-1,:].float()
                        yes,no=logits[:,yes_id].cpu().tolist(),logits[:,no_id].cpu().tolist()
                        values=[y-n for y,n in zip(yes,no)];extras=[{'yes_logit':y,'no_logit':n} for y,n in zip(yes,no)]
                for r,value,length,extra in zip(batch,values,lengths,extras):
                    assert math.isfinite(value) and all(math.isfinite(v) for v in extra.values())
                    target.write(json.dumps({'query_id':r['query_id'],'document_id':r['document_id'],'score':value,
                       'doc_prefix_end_char':r['doc_prefix_end_char'],'input_tokens':length,**extra})+'\n')
                    count+=1;tokens_total+=length
            torch.cuda.synchronize();target.flush()
            stat={'query_id':batch_rows[0]['query_id'],'pairs':len(batch_rows),'query_wall_seconds':time.monotonic()-tick,
                'pairs_completed':count,'inference_wall_seconds':time.monotonic()-inference_started,
                'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            stats.append(stat)
            if len(stats)%50==0:print(json.dumps(stat),flush=True)
        pending=[];previous=None
        for r in rows(ROOT/'prepared/pairs.jsonl.gz'):
            if previous is not None and r['query_id']!=previous:infer_query(pending);pending=[]
            pending.append(r);previous=r['query_id']
        if pending:infer_query(pending)
    assert count==EXPECTED_NEW_PAIRS and len(stats)==EXPECTED_QUERIES
    assert tokens_total==summary['lengths'][args.cell]['total']
    with (out/'time_and_memory.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(stats[0]));writer.writeheader();writer.writerows(stats)
    verify_sources(manifest,'input');verify_sources(manifest,'loader');verify_code()
    write(out/'receipt.json',{'cell':args.cell,**cfg,'pairs':count,'queries':len(stats),'input_tokens_total':tokens_total,
       'score_sha256':sha(out/'scores.jsonl.gz'),'exit_code':0,'wall_seconds':time.monotonic()-started,'inference_seconds':time.monotonic()-inference_started,
       'load_seconds':load_seconds,'started_utc':started_utc,'finished_utc':datetime.now(timezone.utc).isoformat(),
       'batch_size_configured':batch_size,'max_actual_batch_size':max_actual_batch,'precision':'float32' if model_name=='MiniLM' else 'bfloat16',
       'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
       'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'environment':{'torch':torch.__version__,'transformers':transformers.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name()},
       'prepared_summary_sha256':sha(ROOT/'prepared/summary.json'),'frozen_manifest_sha256':sha(ROOT/'frozen_manifest.json'),'references_read':False,'task_retries':0})


if __name__=='__main__':main()
