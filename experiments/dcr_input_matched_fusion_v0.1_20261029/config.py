"""Read-only source graph and frozen same-input comparison contract."""
from pathlib import Path
import hashlib,json,importlib.util
ROOT=Path(__file__).resolve().parent;RUNS=ROOT.parent
POST=RUNS/'dcr_post_fusion_v0.1_20261029';ATTR=RUNS/'dcr_fusion_attribution_v0.1_20261029';CELLS=RUNS/'dcr_model_input_v0.1_20261029'
REF=RUNS/'dcr_gate_mechanism_v0.1_20261029/external_inputs/references.json'
METHODS=('fixed','G12','G_same12','QL5');METRICS=('P1','N1','U1','R10')
COMPARISONS=(('G_same12','QL5'),('G_same12','G12'),('G_same12','fixed'))
QL_COLUMNS=[1,4,5,6,7]


def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()


def read(p):return json.loads(p.read_text())


def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')


def math_core():
 spec=importlib.util.spec_from_file_location('frozen_attribution_math',ATTR/'math_core.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def check_code():
 m=read(ROOT/'frozen_manifest.json')
 for n,h in m['code'].items():assert sha(ROOT/n)==h,n
 return m


def check_phase(phase):
 m=check_code()
 for p,h in m[phase+'_sources'].items():assert sha(Path(p))==h,p
 return m
