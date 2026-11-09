"""Seal prepared inputs, legacy model assets and executable code before scoring."""
from datetime import datetime, timezone
import time
from config import ROOT, read, write, sha, check_sources
from model_contract import verify_model_assets


def main() -> None:
    start = time.monotonic()
    check_sources()
    assert not (ROOT / 'cells').exists() and not (ROOT / 'production_started.json').exists()
    assert read(ROOT / 'fixtures.json')['status'] == 'PASS'
    preparation = read(ROOT / 'preparation.json')
    assert preparation['queries'] == 43 and preparation['qrels_read'] is False
    for name, expected in preparation['files'].items():
        assert sha(ROOT / name) == expected
    assets = verify_model_assets()
    write(ROOT / 'model_assets_sealed.json', assets)
    names = ['config.py', 'prepare.py', 'infer.py', 'rank.py', 'score.py', 'verify.py',
             'fixtures.py', 'freeze.py', 'model_contract.py', 'pipeline.py', 'launch.sh',
             'PROTOCOL.md', 'SCREENING_PROTOCOL.md', 'SOURCE_REVIEW.md']
    inputs = dict(preparation['files'])
    for name in ['candidate_manifest.json', 'preflight_summary.json', 'source_contract.json',
                 'reference_receipt.json', 'preparation.json', 'fixtures.json', 'model_assets_sealed.json',
                 'queries/cds-query.jsonl', 'raw/cds/cds-corpus.jsonl']:
        inputs[name] = sha(ROOT / name)
    write(ROOT / 'frozen_manifest.json', {'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'code': {name: sha(ROOT / name) for name in names}, 'inputs': inputs,
        'primary': 'G_same12-QL5 P1 on all 43 CDS requests', 'new_model_pairs': 6880,
        'timeout_seconds': 14400, 'source_refit': False, 'target_scores_exist': False,
        'reference_semantic_content_read': False, 'asset_check_seconds': time.monotonic() - start})
    print(sha(ROOT / 'frozen_manifest.json'), flush=True)


if __name__ == '__main__':
    main()
