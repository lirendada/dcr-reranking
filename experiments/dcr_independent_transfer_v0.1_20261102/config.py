"""Fixed paths, frozen-source loaders and reference-access guard."""
import ast
from functools import cache
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent
GATE = RUNS / 'dcr_gate_mechanism_v0.1_20261029'
STRONG_QWEN = RUNS / 'dcr_stronger_baselines_v0.1_20261028'
BUNDLE = RUNS / 'dot_experiment_bundle'
CE_MODEL = BUNDLE / 'research/model_cache/cross-encoder_ms-marco-MiniLM-L6-v2_233902d25c440f23af6f7d6e94d2946bac0bee0a'
CE_LOADER = BUNDLE / 'research/dcr_error_analysis_v0.1_20261027/ce_baseline.py'
PARSER = BUNDLE / 'research/dcr_public_baselines_v0.1_20261027/frozen_parser.py'
A = RUNS / 'dcr_input_matched_fusion_v0.1_20261029'
OLD = RUNS / 'dcr_post_g_external_v0.1_20261029'
GPU_LOCK = Path('/path/to/remote/lw_experiments/logs/haccp_first_seed.lock')
CORPUS = ROOT / 'raw/cds/cds-corpus.jsonl'
QRELS = ROOT / 'references/cds-qrel.tsv'
N = 43
CELLS = ('MiniLM_topic', 'MiniLM_full', 'Qwen_topic', 'Qwen_full')
METHODS = CELLS + ('G12', 'G_same12', 'QL5')
METRICS = ('P1', 'nDCG20', 'MRR20', 'R10')
CONTRASTS = (
    ('G_same12', 'QL5'), ('G_same12', 'G12'), ('G_same12', 'Qwen_full'),
    ('QL5', 'Qwen_full'), ('G12', 'QL5'),
    ('MiniLM_full', 'MiniLM_topic'), ('Qwen_full', 'Qwen_topic'),
    ('Qwen_topic', 'MiniLM_topic'), ('Qwen_full', 'MiniLM_full'),
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def rows(path: Path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def documents():
    for row in rows(CORPUS):
        assert isinstance(row['text'], str)
        yield {'document_id': str(row['_id']), 'text': row['text']}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@cache
def parser_module():
    return load_module(PARSER, 'frozen_transfer_parser')


@cache
def source_math():
    return load_module(RUNS / 'dcr_post_fusion_v0.1_20261029/post_math.py', 'frozen_transfer_math')


@cache
def source_matched():
    return load_module(RUNS / 'dcr_input_ifir_v0.1_20261029/feature_kernel.py', 'frozen_transfer_match').matched


@cache
def qwen_text_function():
    path = GATE / 'prepare_external_neural.py'
    tree = ast.parse(path.read_text())
    namespace = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {'DEFAULT', 'PREFIX', 'SUFFIX'}:
                namespace[target.id] = ast.literal_eval(node.value)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'qwen_text')
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['qwen_text']


def check_sources() -> None:
    for name, expected in read(ROOT / 'source_contract.json').items():
        assert sha(Path(name)) == expected, name


def check_frozen() -> dict:
    freeze = read(ROOT / 'frozen_manifest.json')
    for name, expected in freeze['code'].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in freeze['inputs'].items():
        assert sha(ROOT / name) == expected, name
    check_sources()
    return freeze


def check_release() -> None:
    check_frozen()
    assert read(ROOT / 'parent_release.json')['freeze_sha256'] == sha(ROOT / 'frozen_manifest.json')


def _reference_guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        if path == QRELS.resolve() or ROOT / 'references' in path.parents:
            if os.environ.get('DCR_SCORING_PHASE') != '1':
                raise PermissionError('Reference access is prohibited before prediction sealing')


sys.addaudithook(_reference_guard)
