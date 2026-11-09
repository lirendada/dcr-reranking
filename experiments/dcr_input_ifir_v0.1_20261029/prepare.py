"""No-label tokenizer preflight and exact reuse of original two cells, per instruction."""
from datetime import datetime,timezone
from pathlib import Path
import gzip,hashlib,json,os,time
from config import ROOT,OLD,EXCL_FOUR,CE_MODEL,CE_LOADER,GATE,STRONG,DATASETS,COUNTS,CELLS,MODEL,VARIANT,read,write,rows,sha,check_A,qwen_text


def validate_assets():
 # These expected digests were computed/officially checked in the frozen ExcluIR four-cell prep.
 approved=read(EXCL_FOUR/'prepared/summary.json')['assets'];result={}
 for model,asset in approved.items():
  for r in asset['files']:
   p=Path(asset['directory'])/r['name'];assert p.stat().st_size==r['bytes']
   if r['algorithm']=='sha256':actual=sha(p)
   else:
    h=hashlib.sha1(b'blob '+str(r['bytes']).encode()+b'\0')
    with p.open('rb') as f:
     for block in iter(lambda:f.read(1048576),b''):h.update(block)
    actual=h.hexdigest()
   assert actual==r['digest'],str(p)
  result[model]=asset
 return result


def create_context(ds,out,queries):
    # Reuse the tested inference body with dataset constants, plus a pre-load time gate only.
    source=(EXCL_FOUR/'infer_cell.py').read_text()
    needle="    if model_name=='MiniLM':\n        spec="
    replacement="    from config import assert_gpu_start_allowed\n    assert_gpu_start_allowed()\n    if model_name=='MiniLM':\n        spec="
    assert source.count(needle)==1
    patched=source.replace(needle,replacement,1)
    patched=patched.replace("old.load_model(CE_MODEL,'cuda',4)","old.load_model(CE_MODEL,'cuda',1)")
    patched=patched.replace("torch,tokenizer,model,load_seconds=old.load_model()","from prepare_inputs import load_qwen_single_thread\n        torch,tokenizer,model,load_seconds=load_qwen_single_thread(old)")
    with (out/'infer_cell.py').open('x') as f:f.write(patched)
    cfg="""from pathlib import Path
from datetime import datetime
import hashlib,json,gzip
ROOT=Path(__file__).resolve().parent;B_ROOT=ROOT.parent
GATE=Path(GATE_VALUE);STRONG_QWEN=Path(STRONG_VALUE);CE_MODEL=Path(CE_MODEL_VALUE);CE_LOADER=Path(CE_LOADER_VALUE)
GPU_LOCK=Path('/path/to/remote/lw_experiments/logs/haccp_first_seed.lock')
QUERY_PATHS={'topic':ROOT/'prepared/queries_topic.jsonl','full':ROOT/'prepared/queries_full.jsonl'}
CORPUS_PATH=Path(CORPUS_VALUE)
CELL_CONFIG={'MiniLM_topic':{'model':'MiniLM','input_variant':'topic','new_or_reuse':'reuse'},'MiniLM_full':{'model':'MiniLM','input_variant':'full','new_or_reuse':'new'},'Qwen_topic':{'model':'Qwen','input_variant':'topic','new_or_reuse':'new'},'Qwen_full':{'model':'Qwen','input_variant':'full','new_or_reuse':'reuse'}}
NEW_CELLS=('MiniLM_full','Qwen_topic');EXPECTED_QUERIES=COUNT_VALUE;HEAD_SIZE=20;EXPECTED_NEW_PAIRS=20*EXPECTED_QUERIES
BATCH_SIZE={'MiniLM':32,'Qwen':16};MAX_LENGTH={'MiniLM':512,'Qwen':2048}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\\n')
def rows(p):
 with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:
  for line in f:yield json.loads(line)
def verify_code():
 m=read(B_ROOT/'frozen_manifest.json')
 for n,h in m['code'].items():assert sha(B_ROOT/n)==h,n
 return m
def verify_sources(m,phase):
 for p,h in m[phase+'_sources'].items():assert sha(Path(p))==h,p
def assert_gpu_start_allowed():
 assert datetime.now().astimezone()<datetime.fromisoformat('2026-10-30T04:00:00+08:00'),'After nightly GPU start cutoff; do not load model'
"""
    values={'GATE_VALUE':repr(str(GATE)),'STRONG_VALUE':repr(str(STRONG)),'CE_MODEL_VALUE':repr(str(CE_MODEL)),
       'CE_LOADER_VALUE':repr(str(CE_LOADER)),'CORPUS_VALUE':repr(str(OLD/ds/'inputs/corpus.jsonl')),'COUNT_VALUE':str(COUNTS[ds])}
    for key,value in values.items():cfg=cfg.replace(key,value)
    with (out/'config.py').open('x') as f:f.write(cfg)
    template="""import importlib.util,sys
from config import GATE
def original_template():
 sys.path.insert(0,str(GATE));spec=importlib.util.spec_from_file_location('old_preparation',GATE/'prepare_external_neural.py')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.qwen_text
def load_qwen_single_thread(original):
 import time,torch,transformers
 from transformers import AutoModelForCausalLM
 original.require_gpu_lease();c=original.read(original.ROOT/'qwen_config.json')
 assert transformers.__version__==c['transformers'] and torch.cuda.is_available()
 torch.set_num_threads(1);started=time.monotonic()
 model=AutoModelForCausalLM.from_pretrained(original.model_path(),local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,low_cpu_mem_usage=False,attn_implementation='sdpa').eval().to('cuda')
 tokenizer=original.tokenizer_only();torch.cuda.synchronize()
 return torch,tokenizer,model,time.monotonic()-started
"""
    with (out/'prepare_inputs.py').open('x') as f:f.write(template)
    (out/'prepared').mkdir()
    for name,field in [('topic','topic'),('full','request')]:
        with (out/'prepared'/f'queries_{name}.jsonl').open('x') as f:
            for q in queries:f.write(json.dumps({'query_id':q['query_id'],'text':q[field]})+'\n')

def main():
 assert str(ROOT).startswith('/path/to/remote/lw_experiments/');check_A();os.environ['TOKENIZERS_PARALLELISM']='false'
 started=time.monotonic();assets=validate_assets()
 from transformers import AutoTokenizer
 ce=AutoTokenizer.from_pretrained(CE_MODEL,local_files_only=True,trust_remote_code=False)
 qw=AutoTokenizer.from_pretrained(assets['Qwen']['directory'],local_files_only=True,trust_remote_code=False)
 summary={};outputs={};sources={str(EXCL_FOUR/'prepared/summary.json'):sha(EXCL_FOUR/'prepared/summary.json')}
 for ds in DATASETS:
  inputs=read(OLD/'inputs_sealed.json')['datasets'][ds]['files']
  for n,h in inputs.items():assert sha(OLD/ds/'inputs'/n)==h
  queries=list(rows(OLD/ds/'inputs/queries.jsonl'));assert len(queries)==COUNTS[ds]
  featuremeta=read(OLD/ds/'outputs/feature_metadata.json');assert featuremeta['qids']==[q['query_id'] for q in queries]
  source_paths=[OLD/ds/'inputs/queries.jsonl',OLD/ds/'inputs/corpus.jsonl',OLD/ds/'outputs/features.npz',OLD/ds/'outputs/feature_metadata.json']
  corpus={r['document_id']:r['text'] for r in rows(source_paths[1])};score={}
  for model in ['MiniLM','Qwen']:
   p=OLD/ds/model/'scores.jsonl.gz';r=read(p.parent/'manifest.json');assert sha(p)==r['scores_sha256'];source_paths += [p,p.parent/'manifest.json']
   score[model]={}
   for row in rows(p):
    q,d=row['query_id'],row['document_id'];pool=score[model].setdefault(q,{})
    assert d not in pool;pool[d]=row
  out=ROOT/ds;out.mkdir(exist_ok=False);(out/'cells').mkdir();create_context(ds,out,queries);records=[];reuse={'MiniLM_topic':[],'Qwen_full':[]};lengths={c:[] for c in CELLS};overflow=[]
  for i,query in enumerate(queries):
   q=query['query_id'];mini=score['MiniLM'][q];head=sorted(mini,key=lambda d:(-mini[d]['score'],d))[:20]
   assert len(mini)==100 and head==featuremeta['heads'][i] and set(score['Qwen'][q])==set(head)
   bodies=[];batch=[]
   for rank,d in enumerate(head,1):
    mr,qr=mini[d],score['Qwen'][q][d];end=mr['doc_prefix_end_char'];assert end==qr['doc_prefix_end_char'] and 0<=end<=len(corpus[d])
    body=corpus[d][:end];bodies.append(body);batch.append({'query_id':q,'document_id':d,'head_rank':rank,'doc_prefix_end_char':end,'body_prefix_sha256':hashlib.sha256(body.encode()).hexdigest()})
   mlen={v:[len(s) for s in ce([query[v]]*20,bodies,truncation=False,padding=False)['input_ids']] for v in ['topic','request']}
   qlen={v:[len(s) for s in qw([qwen_text(query[v],b) for b in bodies],add_special_tokens=False,truncation=False,padding=False)['input_ids']] for v in ['topic','request']}
   for j,r in enumerate(batch):
    d=r['document_id'];tok={c:(mlen if MODEL[c]=='MiniLM' else qlen)[VARIANT[c]][j] for c in CELLS}
    assert tok['MiniLM_topic']==mini[d]['input_tokens'] and tok['Qwen_full']==score['Qwen'][q][d]['input_tokens']
    for c,t in tok.items():
     lengths[c].append(t)
     if t>(512 if MODEL[c]=='MiniLM' else 2048):overflow.append({'query_id':q,'document_id':d,'cell':c,'tokens':t})
    r['token_lengths']=tok;records.append(r);reuse['MiniLM_topic'].append(mini[d]);reuse['Qwen_full'].append(score['Qwen'][q][d])
  write(out/'token_preflight.json',{'overflows':overflow,'pairs':len(records),'old_pairs_aligned':len(records),'no_retruncation':True});assert not overflow,overflow[:3]
  with gzip.open(out/'prepared/pairs.jsonl.gz','xt') as f:
   for r in records:f.write(json.dumps(r)+'\n')
  for cell,rs in reuse.items():
   folder=out/'cells'/cell;folder.mkdir()
   with gzip.open(folder/'scores.jsonl.gz','xt') as f:
    for r in rs:f.write(json.dumps(r)+'\n')
   write(folder/'receipt.json',{'cell':cell,'pairs':len(rs),'score_sha256':sha(folder/'scores.jsonl.gz'),'reused':True,'new_neural_pairs':0,'input_tokens_total':sum(r['input_tokens'] for r in rs),'exit_code':0})
   for p in [folder/'scores.jsonl.gz',folder/'receipt.json']:outputs[str(p.relative_to(ROOT))]=sha(p)
  for p in [out/'prepared/pairs.jsonl.gz',out/'token_preflight.json']:outputs[str(p.relative_to(ROOT))]=sha(p)
  local_outputs={}
  for p in [out/'prepared/pairs.jsonl.gz',out/'token_preflight.json']+[out/'cells'/c/n for c in reuse for n in ['scores.jsonl.gz','receipt.json']]+[out/'prepared/queries_topic.jsonl',out/'prepared/queries_full.jsonl']:
   local_outputs[str(p.relative_to(out))]=sha(p)
  local_summary={'config_sha256':sha(out/'config.py'),'outputs':local_outputs,'lengths':{c:{'total':sum(v),'max':max(v),'min':min(v)} for c,v in lengths.items()},'pairs':len(records),'queries':len(queries)}
  write(out/'prepared/summary.json',local_summary)
  for n in ['infer_cell.py','config.py','prepare_inputs.py','prepared/queries_topic.jsonl','prepared/queries_full.jsonl','prepared/summary.json']:outputs[f'{ds}/{n}']=sha(out/n)
  summary[ds]={'queries':len(queries),'pairs_per_cell':len(records),'lengths':{c:{'min':min(v),'max':max(v),'total':sum(v)} for c,v in lengths.items()}}
  for p in source_paths:sources[str(p)]=sha(p)
  print(json.dumps({'dataset':ds,'prepared_queries':len(queries),'new_pairs_per_cell':len(records)}),flush=True)
 write(ROOT/'preparation.json',{'prepared_utc':datetime.now(timezone.utc).isoformat(),'datasets':summary,'outputs':outputs,'sources':sources,'assets':assets,
   'model_loaded':False,'qrels_opened':False,'source_models_sha256':sha(ROOT.parent/'dcr_input_matched_fusion_v0.1_20261029/transfer_models.json'),'wall_seconds':time.monotonic()-started})


if __name__=='__main__':main()
