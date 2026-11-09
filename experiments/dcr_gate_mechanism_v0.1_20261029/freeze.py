"""Bind exact new code and all reused artifacts before new scores."""
import json
from pathlib import Path
from adapter import sha256, write_json

ROOT = Path(__file__).resolve().parent
PREV = ROOT.parent / 'dcr_stronger_baselines_v0.1_20261028'


def main():
    assert str(ROOT).startswith('/path/to/remote/lw_experiments/')
    sources = json.loads((PREV / 'source_hashes.json').read_text())
    for suffix in ('reuse_results/scores.jsonl.gz', 'reuse_results/per_query.csv',
                   'qwen_results/scores.jsonl.gz', 'qwen_results/per_query.csv',
                   'evaluation.py', 'frozen_manifest.json'):
        path = PREV / suffix
        sources[str(path)] = sha256(path)
    sources.update({str(ROOT / 'source_data' / name): expected for name, expected in {
        'corpus.json': '429224106113e7d5eb3938ae1da0e7c0de24a8762685adfda2ba11891b8d2b32',
        'test_manual_final.json': '28a5f3fcf935e8ad52802d21a56262fc690505cf43f953458e45d03a45400e78'}.items()})
    for name, expected in sources.items():
        assert sha256(Path(name)) == expected, name
    names = ['protocol.md', 'gate_cpu.py', 'retrieve_external.py', 'adapter.py',
             'scorer.py', 'test_mechanism.py', 'freeze.py']
    write_json(ROOT / 'frozen_manifest.json', {'code': {name: sha256(ROOT / name) for name in names},
        'sources': sources, 'frozen_before_new_ranking_metrics': True})
    print(sha256(ROOT / 'frozen_manifest.json'))


if __name__ == '__main__':
    main()
