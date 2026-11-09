"""Label-free extraction. Only selected top20 Qwen fields enter any numerical state."""
import csv
import gzip
import importlib.util
import json
import time
import numpy as np
from paths import ROOT,GATE,PARSER,sha,read,write,verify_code
from post_math import doc_features,G_COLUMNS,C_COLUMNS


def rows(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt') as f:
        for line in f:yield json.loads(line)


def main():
    start=time.monotonic();m=verify_code()
    for p,h in m['feature_sources'].items():
        from pathlib import Path
        assert sha(Path(p))==h
    pools={}
    for row in csv.DictReader((GATE/'external_retrieval/candidates.csv').open()):
        pools.setdefault(row['query_id'],[]).append(row['document_id'])
    mini={}
    for row in rows(GATE/'external_MiniLM/scores.jsonl.gz'):
        key=(row['query_id'],row['document_id']);assert key not in mini
        mini[key]=(row['score'],row['doc_prefix_end_char'])
    qids=sorted(pools);head={};ranked={}
    for q in qids:
        assert len(pools[q])==len(set(pools[q]))==100
        ranked[q]=sorted(pools[q],key=lambda d:(-mini[q,d][0],d));head[q]=set(ranked[q][:20])
    qwen={}
    for row in rows(GATE/'external_Qwen/scores.jsonl.gz'):
        q,d=row['query_id'],row['document_id']
        if d not in head[q]:continue  # No score/no_logit/token access for tail rows.
        assert (q,d) not in qwen and row['doc_prefix_end_char']==mini[q,d][1]
        qwen[q,d]=(row['score'],row['input_tokens'])
    assert len(qwen)==len(qids)*20
    needed={d for q in qids for d in head[q]}
    corpus={r['document_id']:r['text'] for r in rows(GATE/'external_inputs/corpus.jsonl') if r['document_id'] in needed}
    assert set(corpus)==needed
    topics={r['query_id']:r['text'] for r in rows(GATE/'external_inputs/queries_question0.jsonl')}
    requests={r['query_id']:r['text'] for r in rows(GATE/'external_inputs/queries_RQ_rewrite.jsonl')}
    spec=importlib.util.spec_from_file_location('frozen_parser',PARSER);parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
    features=[];canon=[];raw=[];tokens=[];parsed=[]
    for q in qids:
        docs=ranked[q][:20]
        body_tokens=[parser.tokenize(corpus[d][:mini[q,d][1]],remove_stop=True) for d in docs]
        assert all(0<=mini[q,d][1]<=len(corpus[d]) for d in docs)
        neg=parser.standalone_parse_instruction(topics[q],requests[q])['c_neg']
        ms=np.asarray([mini[q,d][0] for d in docs]);qs=np.asarray([qwen[q,d][0] for d in docs])
        x=doc_features(ms,qs,body_tokens,parser.tokenize(topics[q],remove_stop=True),parser.tokenize(requests[q],remove_stop=True),neg)
        qrank=sorted(range(20),key=lambda i:(-qs[i],docs[i]))
        x[:,2]=np.arange(1,21)/20
        for r,i in enumerate(qrank):x[i,3]=(r+1)/20
        features.append(x);canon.append(sorted(range(20),key=lambda i:docs[i]));raw.append(np.stack([ms,qs],axis=1))
        tokens.append(sum(qwen[q,d][1] for d in docs));parsed.append({'qid':q,'c_neg':neg})
    np.savez_compressed(ROOT/'features.npz',X=np.asarray(features),canonical=np.asarray(canon,dtype=np.uint8),raw_scores=np.asarray(raw),MiniLM_all_scores=np.asarray([[mini[q,d][0] for d in ranked[q]] for q in qids]),Qwen_tokens=np.asarray(tokens,dtype=np.int64))
    write(ROOT/'head_and_tail.json',{'qids':qids,'ranked_pools':[ranked[q] for q in qids],'G_columns':G_COLUMNS,'C_columns':C_COLUMNS})
    write(ROOT/'parsed_constraints.json',parsed)
    write(ROOT/'features_sealed.json',{'files':{n:sha(ROOT/n) for n in ['features.npz','head_and_tail.json','parsed_constraints.json']},
      'Qwen_rows_retained':len(qwen),'Qwen_tail_scores_used':False,'reference_inputs_opened':False,'CPU_task_wall_seconds':time.monotonic()-start})


if __name__=='__main__':main()
