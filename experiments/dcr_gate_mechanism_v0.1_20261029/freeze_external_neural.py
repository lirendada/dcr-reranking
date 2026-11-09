"""Second freeze binds candidate pool/shared offsets/neural execution before GPU scoring."""
import json
from pathlib import Path
from adapter import sha256, write_json
from prepare_external_neural import ROOT, PREV, CE_MODEL


def main():
    assert str(ROOT).startswith('/path/to/remote/lw_experiments/')
    assert not (ROOT / 'external_MiniLM').exists() and not (ROOT / 'external_Qwen').exists()
    prepared = json.loads((ROOT / 'external_prepared/summary.json').read_text())
    assert prepared['pairs'] == 345200 and prepared['model_loaded'] is False
    assert prepared['lengths']['ce']['max'] <= 512 and prepared['lengths']['qwen']['max'] <= 2048
    assert sha256(ROOT / 'external_prepared/pairs.jsonl.gz') == prepared['prepared_sha256']
    sources = [ROOT / 'frozen_manifest.json', ROOT / 'external_inputs/metadata.json',
        ROOT / 'external_inputs/queries_question0.jsonl', ROOT / 'external_inputs/queries_RQ_rewrite.jsonl',
        ROOT / 'external_inputs/corpus.jsonl', ROOT / 'external_retrieval/candidates.csv',
        ROOT / 'external_retrieval/candidates_frozen.json', ROOT / 'external_prepared/pairs.jsonl.gz',
        ROOT / 'external_prepared/summary.json', PREV / 'qwen.py', PREV / 'qwen_config.json',
        PREV / 'model_assets.json', PREV / 'official_model_metadata.json', PREV / 'evaluation.py',
        CE_MODEL / 'download_manifest.json',
        PREV.parent / 'dot_experiment_bundle/research/dcr_error_analysis_v0.1_20261027/ce_baseline.py']
    names = ['protocol.md', 'external_neural_protocol.md', 'prepare_external_neural.py',
             'external_infer.py', 'score_external_neural.py', 'freeze_external_neural.py',
             'run_external_gpu.sh', 'adapter.py', 'scorer.py', 'external_neural_tests.py']
    write_json(ROOT / 'external_neural_frozen.json', {'code': {n: sha256(ROOT / n) for n in names},
        'sources': {str(p): sha256(p) for p in sources},
        'neural_scores_exist': False, 'GPU_started': False,
        'model_pairs': {'MiniLM': 345200, 'Qwen': 345200}, 'candidate_sha256': prepared['candidate_sha256'],
        'after_cpu_retrieval_before_external_neural_scores': True})
    print(sha256(ROOT / 'external_neural_frozen.json'), flush=True)


if __name__ == '__main__':
    main()
