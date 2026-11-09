"""Tokenizer-only exact body prefixes; MiniLM all100 then Qwen head20 only."""
import argparse
import csv
import gzip
import importlib.util
import json
import time
from paths import ROOT,GATE,DATASETS,CE_MODEL,rows,read,write,sha,verify_phase


def old_preparation():
    import sys;sys.path.insert(0,str(GATE))
    spec=importlib.util.spec_from_file_location('old_preparation',GATE/'prepare_external_neural.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def prefix_end(body,ids,offsets,budget):
    assert budget>0
    return len(body) if len(ids)<=budget else offsets[budget-1][1]


def main():
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['MiniLM','Qwen'],required=True);args=a.parse_args()
    verify_phase('input');started=time.monotonic();old=old_preparation()
    import os;os.environ['TOKENIZERS_PARALLELISM']='false'
    from transformers import AutoTokenizer
    if args.stage=='MiniLM':
        tokenizer=AutoTokenizer.from_pretrained(CE_MODEL,local_files_only=True,trust_remote_code=False)
    else:
        assets=read(old.PREV/'model_assets.json')
        tokenizer=AutoTokenizer.from_pretrained(assets['directory'],local_files_only=True,trust_remote_code=False)
    summaries={}
    for ds in DATASETS:
        corpus={r['document_id']:r['text'] for r in rows(ROOT/ds/'inputs/corpus.jsonl')}
        queries={r['query_id']:r for r in rows(ROOT/ds/'inputs/queries.jsonl')}
        out=ROOT/ds/'prepared';out.mkdir(parents=True,exist_ok=True)
        assert not (out/(args.stage+'_pairs.jsonl.gz')).exists()
        lengths=[];truncated=0;count=0
        if args.stage=='MiniLM':
            cache={};budgets={}
            for q,r in queries.items():
                maximum=max(len(tokenizer(r[k],add_special_tokens=False)['input_ids']) for k in ['topic','request'])
                budgets[q]=512-3-maximum
                assert budgets[q]>0,('complete query overflow',ds,q,maximum)
            with gzip.open(out/'MiniLM_pairs.jsonl.gz','wt') as f:
                for r in csv.DictReader((ROOT/ds/'retrieval/candidates.csv').open()):
                    q,d=r['query_id'],r['document_id'];body=corpus[d]
                    if d not in cache:
                        enc=tokenizer(body,add_special_tokens=False,return_offsets_mapping=True)
                        cache[d]=(enc['input_ids'],enc['offset_mapping'])
                    ids,offsets=cache[d];budget=budgets[q]
                    end=prefix_end(body,ids,offsets,budget)
                    length=len(tokenizer(queries[q]['topic'],body[:end],truncation=False)['input_ids'])
                    assert length<=512 and 0<=end<=len(body)
                    f.write(json.dumps({'query_id':q,'document_id':d,'doc_prefix_end_char':end,'input_tokens':length,'shared_document_budget':budget})+'\n')
                    count+=1;lengths.append(length);truncated+=end<len(body)
            assert count==len(queries)*100
        else:
            mini={}
            source=ROOT/ds/'MiniLM/scores.jsonl.gz'
            assert sha(source)==read(source.parent/'manifest.json')['scores_sha256']
            for r in rows(source):mini.setdefault(r['query_id'],{})[r['document_id']]=r
            selected=[]
            for q in queries:
                assert len(mini[q])==100
                docs=sorted(mini[q],key=lambda d:(-mini[q][d]['score'],d))[:20]
                for d in docs:selected.append(mini[q][d])
            with gzip.open(out/'Qwen_pairs.jsonl.gz','wt') as f:
                for offset in range(0,len(selected),128):
                    batch=selected[offset:offset+128]
                    texts=[old.qwen_text(queries[r['query_id']]['request'],corpus[r['document_id']][:r['doc_prefix_end_char']]) for r in batch]
                    encoded=tokenizer(texts,add_special_tokens=False,truncation=False)['input_ids']
                    for r,ids in zip(batch,encoded):
                        assert len(ids)<=2048,('Qwen overflow',ds,r['query_id'],r['document_id'],len(ids))
                        f.write(json.dumps({'query_id':r['query_id'],'document_id':r['document_id'],'doc_prefix_end_char':r['doc_prefix_end_char'],'input_tokens':len(ids)})+'\n')
                        count+=1;lengths.append(len(ids))
            assert count==len(queries)*20
        target=out/(args.stage+'_pairs.jsonl.gz')
        summaries[ds]={'pairs':count,'queries':len(queries),'min_input_tokens':min(lengths),'max_input_tokens':max(lengths),
          'sum_input_tokens':sum(lengths),'truncated_document_pairs':truncated if args.stage=='MiniLM' else None,'prepared_sha256':sha(target)}
    write(ROOT/(args.stage+'_preparation.json'),{'datasets':summaries,'model_loaded':False,'qrels_opened':False,'CPU_wall_seconds':time.monotonic()-started})


if __name__=='__main__':main()
