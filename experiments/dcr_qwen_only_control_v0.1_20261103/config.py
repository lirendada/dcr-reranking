"""Explicit read-only dependency graph for the Qwen-only cached-score control."""
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent
POST = RUNS / 'dcr_post_fusion_v0.1_20261029'
ATTR = RUNS / 'dcr_fusion_attribution_v0.1_20261029'
SAME = RUNS / 'dcr_input_matched_fusion_v0.1_20261029'
IFIR = RUNS / 'dcr_input_ifir_v0.1_20261029'
OLD = RUNS / 'dcr_post_g_external_v0.1_20261029'
CDS = RUNS / 'dcr_independent_transfer_v0.1_20261102'
REF = RUNS / 'dcr_gate_mechanism_v0.1_20261029/external_inputs/references.json'
Q7 = [1, 3, 4, 5, 6, 7, 11]
M5 = [0, 2, 8, 9, 10]
METHODS = ['fixed', 'QL5', 'Q7', 'G12', 'G_same12']
COMPARISONS = [('G_same12', 'Q7'), ('G12', 'Q7'), ('Q7', 'QL5')]
SOURCE_METRICS = ['P1', 'N1', 'U1', 'R10']
TARGET_METRICS = ['P1', 'nDCG20', 'MRR20', 'R10']
TARGETS = ['fiqa', 'nfcorpus', 'cds']


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def rows(path: Path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        for line in stream:
            yield json.loads(line)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check() -> dict:
    assert str(ROOT).startswith('/path/to/remote/lw_experiments/runs/')
    manifest = read(ROOT / 'frozen_manifest.json')
    for group, base in [('code', ROOT), ('sources', Path('/'))]:
        for name, expected in manifest[group].items():
            assert sha(base / name) == expected, name
    return manifest


def target_spec(dataset: str) -> dict:
    assert dataset in TARGETS
    if dataset == 'cds':
        return {'features': CDS / 'features.npz', 'feature_key': 'G12',
                'meta': CDS / 'ranking_metadata.json', 'rankings': CDS / 'rankings.jsonl.gz',
                'qrels': CDS / 'references/cds-qrel.tsv', 'summary': CDS / 'summary.json'}
    return {'features': OLD / dataset / 'outputs/features.npz', 'feature_key': 'X',
            'queries': OLD / dataset / 'inputs/queries.jsonl',
            'meta': OLD / dataset / 'outputs/feature_metadata.json',
            'rankings': IFIR / dataset / 'outputs/rankings.jsonl.gz',
            'qrels': OLD / 'raw' / dataset / f'{dataset}-qrel.tsv',
            'summary': IFIR / 'summary.json'}
