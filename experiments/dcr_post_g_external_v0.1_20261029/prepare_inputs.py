"""Model-facing query/body adapter; qrels and test_data.json are never opened."""
import json
import re
import time
from paths import ROOT,DATASETS,rows,write,sha,verify_phase


def adapt_queries(records):
    topics={};requests=[];seen=set()
    for row in records:
        q=str(row['_id']);text=row['text'];level=row['level']
        assert isinstance(text,str) and isinstance(level,int) and not isinstance(level,bool) and level>=0
        match=re.fullmatch(r'(.+)_(\d+)',q);assert match and int(match[2])==level,q
        base=match[1];assert q not in seen;seen.add(q)
        if level==0:
            assert base not in topics;topics[base]=text
        else:requests.append({'query_id':q,'base_id':base,'level':level,'request':text})
    for row in requests:
        assert row['base_id'] in topics;row['topic']=topics[row['base_id']]
    requests.sort(key=lambda r:r['query_id'])
    return topics,requests


def main():
    verify_phase('input');started=time.monotonic();inventory={}
    for ds in DATASETS:
        topics,queries=adapt_queries(rows(ROOT/'raw'/ds/f'{ds}-query.jsonl'))
        expected=(639,1718) if ds=='fiqa' else (86,86)
        assert (len(topics),len(queries))==expected,(ds,len(topics),len(queries))
        out=ROOT/ds/'inputs';out.mkdir(parents=True,exist_ok=False)
        with (out/'queries.jsonl').open('x') as f:
            for q in queries:f.write(json.dumps(q)+'\n')
        with (out/'corpus.jsonl').open('x') as f:
            seen=set();empty=0
            for r in rows(ROOT/'raw'/ds/f'{ds}-corpus.jsonl'):
                d=str(r['_id']);body=r['text'];assert d not in seen and isinstance(body,str);seen.add(d)
                empty+=not bool(body.strip());f.write(json.dumps({'document_id':d,'text':body})+'\n')
        inventory[ds]={'topics':len(topics),'instruction_queries':len(queries),'corpus_documents':len(seen),'empty_body':empty,
          'level_counts':{str(k):sum(q['level']==k for q in queries) for k in sorted({q['level'] for q in queries})},
          'files':{n:sha(out/n) for n in ['queries.jsonl','corpus.jsonl']}}
    write(ROOT/'inputs_sealed.json',{'datasets':inventory,'qrels_or_test_data_opened':False,'title_used':False,'CPU_wall_seconds':time.monotonic()-started})
    verify_phase('input')


if __name__=='__main__':main()
