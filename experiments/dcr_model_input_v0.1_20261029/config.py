"""Frozen four-cell contract; importing opens no data or model."""
from pathlib import Path
import gzip
import hashlib
import json

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent
SOURCE_POST = RUNS/'dcr_post_fusion_v0.1_20261029'
GATE = RUNS/'dcr_gate_mechanism_v0.1_20261029'
STRONG_QWEN = RUNS/'dcr_stronger_baselines_v0.1_20261028'
BUNDLE = RUNS/'dot_experiment_bundle'
CE_MODEL = BUNDLE/'research/model_cache/cross-encoder_ms-marco-MiniLM-L6-v2_233902d25c440f23af6f7d6e94d2946bac0bee0a'
CE_LOADER = BUNDLE/'research/dcr_error_analysis_v0.1_20261027/ce_baseline.py'
GPU_LOCK = Path('/path/to/remote/lw_experiments/logs/haccp_first_seed.lock')
QUERY_PATHS = {'topic': GATE/'external_inputs/queries_question0.jsonl',
               'full': GATE/'external_inputs/queries_RQ_rewrite.jsonl'}
CORPUS_PATH = GATE/'external_inputs/corpus.jsonl'
HEAD_PATH = SOURCE_POST/'head_and_tail.json'
OLD_SCORES = {'MiniLM_topic': GATE/'external_MiniLM/scores.jsonl.gz',
              'Qwen_full': GATE/'external_Qwen/scores.jsonl.gz'}
EXPECTED_QUERIES, HEAD_SIZE, EXPECTED_NEW_PAIRS = 3452, 20, 69040
CELL_CONFIG = {
    'MiniLM_topic': {'model': 'MiniLM', 'input_variant': 'topic', 'new_or_reuse': 'reuse'},
    'MiniLM_full': {'model': 'MiniLM', 'input_variant': 'full', 'new_or_reuse': 'new'},
    'Qwen_topic': {'model': 'Qwen', 'input_variant': 'topic', 'new_or_reuse': 'new'},
    'Qwen_full': {'model': 'Qwen', 'input_variant': 'full', 'new_or_reuse': 'reuse'},
}
CELLS = tuple(CELL_CONFIG)
NEW_CELLS = ('MiniLM_full', 'Qwen_topic')
BATCH_SIZE = {'MiniLM': 32, 'Qwen': 16}
MAX_LENGTH = {'MiniLM': 512, 'Qwen': 2048}
METRICS = ('P1', 'N1', 'U1', 'R10')
# Each vector is applied to the four cells in CELLS order, in percentage points.
CONTRASTS = {
    'MiniLM_full-MiniLM_topic': (-1, 1, 0, 0),
    'Qwen_full-Qwen_topic': (0, 0, -1, 1),
    'Qwen_topic-MiniLM_topic': (-1, 0, 1, 0),
    'Qwen_full-MiniLM_full': (0, -1, 0, 1),
    'input_effect_interaction': (1, -1, -1, 1),
}
REFERENCE_PATH = GATE/'external_inputs/references.json'
GROUPS_PATH = RUNS/'dcr_constraint_router_v0.1_20261029/prepared/groups.json'
PLAN_PATH = SOURCE_POST/'prepared/plan.npz'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def rows(path: Path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as f:
        for line in f:
            yield json.loads(line)


def verify_code() -> dict:
    manifest = read(ROOT/'frozen_manifest.json')
    for name, digest in manifest['code'].items():
        assert sha(ROOT/name) == digest, name
    return manifest


def verify_sources(manifest: dict, phase: str) -> None:
    for name, digest in manifest[phase+'_sources'].items():
        assert sha(Path(name)) == digest, name
