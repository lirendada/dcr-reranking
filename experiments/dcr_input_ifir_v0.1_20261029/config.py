"""Frozen observed-IFIR replay; input scoring labels live only in scoring phase."""
from pathlib import Path
import hashlib,json,gzip,importlib.util
from functools import cache
ROOT=Path(__file__).resolve().parent;RUNS=ROOT.parent
A=RUNS/'dcr_input_matched_fusion_v0.1_20261029';OLD=RUNS/'dcr_post_g_external_v0.1_20261029'
EXCL_FOUR=RUNS/'dcr_model_input_v0.1_20261029';GATE=RUNS/'dcr_gate_mechanism_v0.1_20261029'
STRONG=RUNS/'dcr_stronger_baselines_v0.1_20261028';BUNDLE=RUNS/'dot_experiment_bundle'
CE_MODEL=BUNDLE/'research/model_cache/cross-encoder_ms-marco-MiniLM-L6-v2_233902d25c440f23af6f7d6e94d2946bac0bee0a'
CE_LOADER=BUNDLE/'research/dcr_error_analysis_v0.1_20261027/ce_baseline.py'
LOCK=Path('/path/to/remote/lw_experiments/logs/haccp_first_seed.lock')
DATASETS=('fiqa','nfcorpus');COUNTS={'fiqa':1718,'nfcorpus':86}
CELLS=('MiniLM_topic','MiniLM_full','Qwen_topic','Qwen_full');NEW_CELLS=('MiniLM_full','Qwen_topic')
MODEL={'MiniLM_topic':'MiniLM','MiniLM_full':'MiniLM','Qwen_topic':'Qwen','Qwen_full':'Qwen'}
VARIANT={'MiniLM_topic':'topic','MiniLM_full':'request','Qwen_topic':'topic','Qwen_full':'request'}
METHODS=CELLS+('G12','G_same12','QL5');METRICS=('P1','nDCG20','MRR20','R10')
CONTRASTS={'MiniLM_full-MiniLM_topic':(-1,1,0,0),'Qwen_full-Qwen_topic':(0,0,-1,1),'Qwen_topic-MiniLM_topic':(-1,0,1,0),'Qwen_full-MiniLM_full':(0,-1,0,1),'input_effect_interaction':(1,-1,-1,1)}
FUSION=(('G_same12','QL5'),('G_same12','G12'),('G_same12','Qwen_full'))
CUTOFF='2026-10-30T04:00:00+08:00'


def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()


def read(p):return json.loads(p.read_text())


def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')


def rows(p):
 with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:
  for line in f:yield json.loads(line)


def check_code():
 m=read(ROOT/'frozen_manifest.json')
 for n,h in m['code'].items():assert sha(ROOT/n)==h,n
 return m


def check_phase(phase):
 m=check_code()
 for p,h in m[phase+'_sources'].items():assert sha(Path(p))==h,p
 return m


def check_A():
 status=read(A/'independent_verification.json')['status'];assert status.startswith('PASS'),status
 sealed=read(A/'transfer_models_sealed.json');assert sha(A/'transfer_models.json')==sealed['models_sha256']
 return sealed


@cache
def qwen_template():
 import sys;sys.path.insert(0,str(GATE))
 spec=importlib.util.spec_from_file_location('original_external_preparation',GATE/'prepare_external_neural.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return m.qwen_text


def qwen_text(query,body):return qwen_template()(query,body)
