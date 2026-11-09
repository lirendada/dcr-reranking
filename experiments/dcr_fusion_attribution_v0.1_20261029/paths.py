"""Task paths; all production sources remain read-only."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
PREV = ROOT.parent / 'dcr_post_fusion_v0.1_20261029'
REF = ROOT.parent / 'dcr_gate_mechanism_v0.1_20261029/external_inputs/references.json'
GROUPS = ROOT.parent / 'dcr_constraint_router_v0.1_20261029/prepared/groups.json'


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def check_code() -> dict:
    manifest = read(ROOT / 'frozen_manifest.json')
    for name, digest in manifest['code'].items():
        assert sha(ROOT/name) == digest, name
    return manifest


def check_sources(manifest: dict) -> None:
    for path, digest in manifest['sources'].items():
        assert sha(Path(path)) == digest, path
