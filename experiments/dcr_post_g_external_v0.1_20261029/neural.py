"""Frozen loaders and templates, new selected pairs; GPU lease mandatory, no references."""
import argparse
from datetime import datetime,timezone
import gzip
import importlib.util
import json
import math
import os
import resource
import sys
import time
from paths import ROOT,GATE,STRONG,BUNDLE,DATASETS,CE_MODEL,rows,read,write,sha,verify_phase


def main():
    a=argparse.ArgumentParser();a.add_argument('--model',choices=['MiniLM','Qwen'],required=True);args=a.parse_args()
    verify_phase('input');verify_phase('loader')
    assert os.environ.get('DCR_GPU_LEASE')=='1'
    from pathlib import Path
    assert Path('/proc/self/fd/9').resolve()==Path('/path/to/remote/lw_experiments/logs/haccp_first_seed.lock')
    assert os.environ.get('TOKENIZERS_PARALLELISM')=='false'
    started=time.monotonic();started_utc=datetime.now(timezone.utc).isoformat()
    if args.model=='MiniLM':
        p=BUNDLE/'research/dcr_error_analysis_v0.1_20261027/ce_baseline.py'
        spec=importlib.util.spec_from_file_location('original_ce',p);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        torch,tokenizer,model,load_seconds=old.load_model(CE_MODEL,'cuda',4);batch_size=32
    else:
        sys.path.insert(0,str(STRONG));import qwen as old
        torch,tokenizer,model,load_seconds=old.load_model();batch_size=16
    import transformers
    assert transformers.__version__==('4.49.0' if args.model=='MiniLM' else '4.51.3')
    sys.path.insert(0,str(GATE));from prepare_external_neural import qwen_text
    yes_id=no_id=None
    if args.model=='Qwen':
        yes_id,no_id=[tokenizer.convert_tokens_to_ids(t) for t in ['yes','no']];assert yes_id!=no_id
    for ds in DATASETS:
        query={r['query_id']:r for r in rows(ROOT/ds/'inputs/queries.jsonl')}
        corpus={r['document_id']:r['text'] for r in rows(ROOT/ds/'inputs/corpus.jsonl')}
        prepared=ROOT/ds/'prepared'/(args.model+'_pairs.jsonl.gz')
        pm=read(ROOT/(args.model+'_preparation.json'))['datasets'][ds];assert sha(prepared)==pm['prepared_sha256']
        out=ROOT/ds/args.model;out.mkdir(exist_ok=False)
        count=0;tick=time.monotonic()
        with gzip.open(out/'scores.jsonl.gz','wt') as target:
            buffer=[]
            def flush():
                nonlocal count
                if not buffer:return
                for start in range(0,len(buffer),batch_size):
                    batch=buffer[start:start+batch_size];bodies=[corpus[r['document_id']][:r['doc_prefix_end_char']] for r in batch]
                    if args.model=='MiniLM':
                        encoded=tokenizer([query[r['query_id']]['topic'] for r in batch],bodies,truncation=False,padding='max_length',max_length=512,return_tensors='pt')
                        assert encoded['input_ids'].shape[1]==512
                    else:
                        encoded=tokenizer([qwen_text(query[r['query_id']]['request'],body) for r,body in zip(batch,bodies)],add_special_tokens=False,truncation=False,padding=True,pad_to_multiple_of=8,return_tensors='pt')
                        assert encoded['input_ids'].shape[1]<=2048
                    lengths=encoded['attention_mask'].sum(1).tolist();assert lengths==[r['input_tokens'] for r in batch]
                    with torch.inference_mode():
                        tensors={k:v.to('cuda') for k,v in encoded.items()}
                        if args.model=='MiniLM':
                            values=model(**tensors).logits.squeeze(-1).cpu().tolist();extra=[{}]*len(batch)
                        else:
                            logits=model(**tensors,use_cache=False,logits_to_keep=1).logits[:,-1,:].float()
                            yes,no=logits[:,yes_id].cpu().tolist(),logits[:,no_id].cpu().tolist();values=[y-n for y,n in zip(yes,no)]
                            extra=[{'yes_logit':y,'no_logit':n} for y,n in zip(yes,no)]
                    for r,value,length,ext in zip(batch,values,lengths,extra):
                        assert math.isfinite(value)
                        target.write(json.dumps({**r,'score':value,'input_tokens':length,**ext})+'\n');count+=1
                target.flush();buffer.clear()
            previous=None
            for r in rows(prepared):
                if previous is not None and previous!=r['query_id']:
                    flush()
                    if count%(100*50 if args.model=='MiniLM' else 20*50)==0:print(json.dumps({'dataset':ds,'model':args.model,'pairs':count,'elapsed_seconds':time.monotonic()-tick}),flush=True)
                buffer.append(r);previous=r['query_id']
            flush()
        torch.cuda.synchronize();assert count==pm['pairs']
        write(out/'manifest.json',{'model':args.model,'dataset':ds,'pairs':count,'queries':len(query),'load_seconds_shared_across_datasets':load_seconds,
          'inference_seconds':time.monotonic()-tick,'scores_sha256':sha(out/'scores.jsonl.gz'),'batch_size':batch_size,'precision':'float32' if args.model=='MiniLM' else 'bfloat16',
          'sum_input_tokens':pm['sum_input_tokens'],'prepared_sha256':pm['prepared_sha256'],
          'candidate_sha256':read(ROOT/'retrieval_receipt.json')['datasets'][ds]['candidate_sha256'],
          'model_revision':'233902d25c440f23af6f7d6e94d2946bac0bee0a' if args.model=='MiniLM' else '22e683669bc0f0bd69640a1354a6d0aebcfeede5',
          'environment':{'torch':torch.__version__,'transformers':transformers.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name()},
          'query_input':'topic' if args.model=='MiniLM' else 'full_request','finished_utc':datetime.now(timezone.utc).isoformat(),'started_utc':started_utc,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
          'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'qrels_read':False,'task_retries':0})
    write(ROOT/(args.model+'_execution.json'),{'load_seconds':load_seconds,'total_model_phase_wall_seconds':time.monotonic()-started,'datasets':list(DATASETS)})


if __name__=='__main__':main()
