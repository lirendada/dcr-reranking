"""Reviewed code/source freeze; phase lists keep external labels out of model processes."""
from datetime import datetime,timezone
import importlib.metadata as metadata
from paths import ROOT,PREV,GATE,STRONG,BUNDLE,PARSER,CE_MODEL,DATASETS,REVISION,sha,write,read
CODE=['PROTOCOL.md','paths.py','prepare_inputs.py','fit_source.py','retrieve.py','prepare_neural.py','neural.py',
      'apply_post.py','metrics.py','score.py','resource_gate.py','run_gpu.sh','run_pipeline.py','freeze.py','test_pipeline.py']
training=[PREV/n for n in ['features.npz','head_and_tail.json','post_math.py','features_sealed.json','frozen_manifest.json']]+[GATE/'external_inputs/references.json']
inputs=[ROOT/'raw'/ds/f'{ds}-{kind}' for ds in DATASETS for kind in ['query.jsonl','corpus.jsonl']]+[
    ROOT/'hf_metadata.json',PARSER,GATE/'retrieve_external.py',GATE/'adapter.py',GATE/'prepare_external_neural.py',PREV/'post_math.py']
loaders=[BUNDLE/'research/dcr_error_analysis_v0.1_20261027/ce_baseline.py',CE_MODEL/'download_manifest.json']+[
    STRONG/n for n in ['qwen.py','qwen_config.json','model_assets.json','official_model_metadata.json','evaluation.py','assets.py']]
scoring=[ROOT/'raw'/ds/f'{ds}-qrel.tsv' for ds in DATASETS]
assert read(ROOT/'hf_metadata.json')['sha']==REVISION
pre=read(ROOT/'tokenizer_input_preflight.json');assert all(not pre[ds]['overflow'] for ds in DATASETS)
write(ROOT/'frozen_manifest.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'revision':REVISION,
  'code':{n:sha(ROOT/n) for n in CODE},'training_sources':{str(p):sha(p) for p in training},'input_sources':{str(p):sha(p) for p in inputs},
  'loader_sources':{str(p):sha(p) for p in loaders},'scoring_sources':{str(p):sha(p) for p in scoring},
  'config':{'head':20,'candidates':100,'G_dimensions':12,'lambda':.1,'source_pairs':2901,'bootstrap_draws':10000,'seed':20261029,
    'dataset_order':list(DATASETS),'prefix':'per-instruction original512-3-max(topic_tokens,request_tokens)','gain':'raw_rel',
    'MiniLM_precision':'float32','MiniLM_batch':32,'Qwen_precision':'bfloat16','Qwen_batch':16,'Qwen_max_tokens':2048},
  'tokenizer_preflight_sha256':sha(ROOT/'tokenizer_input_preflight.json'),'runtime':{'numpy':metadata.version('numpy')}})
