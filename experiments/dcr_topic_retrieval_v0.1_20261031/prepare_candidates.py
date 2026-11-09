"""Question0 full-corpus BM25 using the unchanged historical retrieval functions."""
import csv
import importlib.util
import json
import platform
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GATE = ROOT.parent / 'dcr_gate_mechanism_v0.1_20261029'
sys.path.insert(0, str(GATE))
from adapter import sha256, write_json
import retrieve_external as old


def main():
    assert str(ROOT).startswith('/path/to/remote/lw_experiments/')
    started = time.monotonic()
    bundle = GATE / 'external_inputs'
    meta = json.loads((bundle / 'metadata.json').read_text())
    sources = [bundle / f for f in ['metadata.json', 'queries_question0.jsonl', 'queries_RQ_rewrite.jsonl', 'corpus.jsonl']]
    sources += [GATE/'retrieve_external.py', GATE/'adapter.py', old.OLD/'frozen_parser.py']
    frozen = json.loads((GATE/'frozen_manifest.json').read_text())
    assert sha256(GATE/'retrieve_external.py') == frozen['code']['retrieve_external.py']
    for p in sources[:4]:
        if p.name != 'metadata.json':
            assert sha256(p) == meta['export_sha256'][p.name]
    hashes = {str(p): sha256(p) for p in sources}
    out = ROOT/'external_retrieval'; out.mkdir(exist_ok=False)
    write_json(out/'input_freeze.json', {'sources': hashes, 'code_sha256': sha256(Path(__file__)), 'references_opened': False})
    queries = [json.loads(line) for line in (bundle/'queries_question0.jsonl').open()]
    corpus = [json.loads(line) for line in (bundle/'corpus.jsonl').open()]
    assert len(queries)==3452 and len(corpus)==90406
    assert all(set(q)=={'query_id','text'} for q in queries)
    assert all(set(d)=={'document_id','text'} for d in corpus)
    assert [d['document_id'] for d in corpus]==[f'd{i:06d}' for i in range(90406)]
    index, statistics = old.build_index([d['text'] for d in corpus]); indexed=time.monotonic()
    times=[]; zeros=0
    with (out/'bm25_top100.csv').open('x',newline='') as rf, (out/'candidates.csv').open('x',newline='') as cf:
        writer,pool=csv.writer(rf),csv.writer(cf)
        writer.writerow(['query_id','document_id','score']);pool.writerow(['query_id','document_id'])
        for number,q in enumerate(queries,1):
            tick=time.monotonic();scores=old.score_query(q['text'],index,len(corpus));selected=old.top_indices(scores,100)
            zeros+=int(scores.max()==0)
            for i in selected:
                did=corpus[i]['document_id'];writer.writerow([q['query_id'],did,repr(float(scores[i]))]);pool.writerow([q['query_id'],did])
            times.append({'query_id':q['query_id'],'seconds':time.monotonic()-tick,'returned':len(selected)})
            if number%250==0:print(json.dumps({'queries':number,'seconds':time.monotonic()-started}),flush=True)
    assert all(sha256(Path(p))==h for p,h in hashes.items())
    write_json(out/'candidates_frozen.json', {'candidate_sha256':sha256(out/'candidates.csv'),'run_sha256':sha256(out/'bm25_top100.csv'),'queries':3452,'documents_per_query':100,'frozen_before_reference_scoring':True})
    write_json(out/'retrieval_manifest.json', {'query_variant':'question0','sources':hashes,'queries':3452,'pairs_returned':345200,'search_space_document_count':90406,'statistics':statistics,'all_zero_score_queries':zeros,'index_seconds':indexed-started,'retrieval_seconds':time.monotonic()-indexed,'total_seconds':time.monotonic()-started,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'references_opened':False,'model_loaded':False,'GPU_used':False,'code_sha256':sha256(Path(__file__))})
    with (out/'retrieval_times.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(times[0]));w.writeheader();w.writerows(times)
    print('CANDIDATES_FROZEN',flush=True)


if __name__=='__main__': main()
