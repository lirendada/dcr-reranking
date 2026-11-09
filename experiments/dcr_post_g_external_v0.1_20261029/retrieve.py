"""Topic full-corpus BM25 top100; same-base pools reused, no labels opened."""
from collections import Counter
import csv
import importlib.util
import time
from paths import ROOT,GATE,DATASETS,rows,read,write,sha,verify_phase


def main():
    verify_phase('input');started=time.monotonic()
    spec=importlib.util.spec_from_file_location('old_exact_retrieval',GATE/'retrieve_external.py');old=importlib.util.module_from_spec(spec)
    import sys;sys.path.insert(0,str(GATE));spec.loader.exec_module(old)
    summaries={}
    for ds in DATASETS:
        queries=list(rows(ROOT/ds/'inputs/queries.jsonl'));corpus=list(rows(ROOT/ds/'inputs/corpus.jsonl'))
        # Preserve original author corpus row order: old BM25 cutoff ties use corpus index.
        docs=[r['document_id'] for r in corpus];assert len(docs)==len(set(docs))>=100
        postings,stats=old.build_index([r['text'] for r in corpus])
        topics={r['base_id']:r['topic'] for r in queries};pools={};pool_scores={};no_positive_score=0
        for base in sorted(topics):
            scores=old.score_query(topics[base],postings,len(docs));indices=old.top_indices(scores,100)
            no_positive_score+=int(not (scores>0).any());pools[base]=[docs[i] for i in indices];pool_scores[base]=[float(scores[i]) for i in indices]
        out=ROOT/ds/'retrieval';out.mkdir(parents=True,exist_ok=False)
        with (out/'candidates.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['query_id','base_id','document_id','BM25_rank','BM25_score']);w.writeheader()
            for q in queries:
                for rank,d in enumerate(pools[q['base_id']],1):w.writerow({'query_id':q['query_id'],'base_id':q['base_id'],'document_id':d,'BM25_rank':rank,'BM25_score':repr(pool_scores[q['base_id']][rank-1])})
        summaries[ds]={'BM25_k1':1.5,'BM25_b':.75,'tf':'old tokenize default, query frequency counted',
          'IDF':'log(1+(N-df+.5)/(df+.5))','tie':'original corpus row index ascending','topics_retrieved':len(topics),
          'queries':len(queries),'candidate_pairs':len(queries)*100,'all_zero_score_topics':no_positive_score,
          'statistics':stats,'candidate_sha256':sha(out/'candidates.csv')}
        write(out/'base_pools.json',pools)
        del corpus,postings
    write(ROOT/'retrieval_receipt.json',{'datasets':summaries,'reference_inputs_opened':False,'CPU_wall_seconds':time.monotonic()-started})


if __name__=='__main__':main()
