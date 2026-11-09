"""Explicit sources; importing this module never opens data."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUNS=ROOT.parent
GATE=RUNS/'dcr_gate_mechanism_v0.1_20261029'
PREVIOUS=RUNS/'dcr_constraint_router_v0.1_20261029'
ORACLE=RUNS/'dcr_oracle_v0.1_20261029'
PARSER=RUNS/'dot_experiment_bundle/research/dcr_public_baselines_v0.1_20261027/frozen_parser.py'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(path.read_text())


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def feature_sources():
    return [GATE/'external_inputs'/n for n in ['queries_question0.jsonl','queries_RQ_rewrite.jsonl','corpus.jsonl']]+[
      GATE/'external_retrieval/candidates.csv',GATE/'external_MiniLM/scores.jsonl.gz',GATE/'external_Qwen/scores.jsonl.gz',PARSER]


def reference_sources():
    return [GATE/'external_inputs/references.json',PREVIOUS/'prepared/groups.json']


def verify_code():
    m=read(ROOT/'frozen_manifest.json')
    for name,h in m['code'].items():assert sha(ROOT/name)==h,name
    return m

BUDGET=RUNS/'dcr_budget_v0.1_20261029'


def baseline_sources():return [BUDGET/'ExcluIR/rankings.jsonl.gz']


def scoring_sources():
    return [GATE/'external_inputs/references.json']+[BUDGET/'ExcluIR/evaluation'/m/'per_query.csv' for m in ['MiniLM','topK_20']]
