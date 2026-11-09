"""Fixed project-local paths and phase-specific guards; importing opens no data."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUNS=ROOT.parent
PREV=RUNS/'dcr_post_fusion_v0.1_20261029'
GATE=RUNS/'dcr_gate_mechanism_v0.1_20261029'
STRONG=RUNS/'dcr_stronger_baselines_v0.1_20261028'
BUNDLE=RUNS/'dot_experiment_bundle'
PARSER=BUNDLE/'research/dcr_public_baselines_v0.1_20261027/frozen_parser.py'
CE_MODEL=BUNDLE/'research/model_cache/cross-encoder_ms-marco-MiniLM-L6-v2_233902d25c440f23af6f7d6e94d2946bac0bee0a'
DATASETS=('fiqa','nfcorpus')
REVISION='0e1bd15954eed1414023961a9f79038cf2aad2c6'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(path.read_text())


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def rows(path):
    import gzip
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt') as f:
        for line in f:yield json.loads(line)


def verify_code():
    m=read(ROOT/'frozen_manifest.json')
    for n,h in m['code'].items():assert sha(ROOT/n)==h,n
    return m


def verify_phase(phase):
    m=verify_code()
    for p,h in m[phase+'_sources'].items():assert sha(Path(p))==h,p


def load_source_math():
    import importlib.util
    p=PREV/'post_math.py'
    spec=importlib.util.spec_from_file_location('frozen_post_math',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def parser_module():
    import importlib.util
    spec=importlib.util.spec_from_file_location('original_frozen_parser',PARSER);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
