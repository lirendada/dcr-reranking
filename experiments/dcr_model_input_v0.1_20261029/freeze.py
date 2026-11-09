"""Pin the prepared four-cell experiment before either new score stream exists."""
from datetime import datetime, timezone
from pathlib import Path
from config import (ROOT, SOURCE_POST, GATE, STRONG_QWEN, CE_MODEL, CE_LOADER,
                    REFERENCE_PATH, GROUPS_PATH, PLAN_PATH, NEW_CELLS, MAX_LENGTH,
                    CELL_CONFIG, read, write, sha)


def main():
    assert str(ROOT).startswith('/path/to/remote/lw_experiments/')
    assert all(not (ROOT/'cells'/cell).exists() for cell in NEW_CELLS)
    prepared = read(ROOT/'prepared/summary.json')
    assert prepared['queries'] == 3452 and prepared['pairs'] == 69040
    assert prepared['old_cells_alignment'] == {'MiniLM_topic': 69040, 'Qwen_full': 69040}
    assert not prepared['references_opened'] and prepared['new_model_calls'] == 0
    assert prepared['config_sha256'] == sha(ROOT/'config.py')
    for name, digest in prepared['code'].items():
        assert sha(ROOT/name) == digest, name
    for cell, stats in prepared['lengths'].items():
        assert stats['pairs'] == 69040 and stats['max'] <= MAX_LENGTH[CELL_CONFIG[cell]['model']]
    assert read(ROOT/'scoring_fixtures.json')['passed']
    assert read(ROOT/'resource_preflight.json')['passed']
    assert sha(SOURCE_POST/'frozen_manifest.json') == '65e1a77a41ce26652e4b7faa8365e17b7e4425a36035e713605ba702ea091ea7'
    assert read(SOURCE_POST/'independent_verification.json')['status'] == 'PASS_INDEPENDENT_POST_FUSION_VERIFICATION'
    inputs = dict(prepared['sources'])
    for name, digest in prepared['outputs'].items():
        assert sha(ROOT/name) == digest, name
        inputs[str(ROOT/name)] = digest
    for name in ('head_and_tail.json', 'features.npz', 'features_sealed.json', 'frozen_manifest.json', 'independent_verification.json'):
        inputs[str(SOURCE_POST/name)] = sha(SOURCE_POST/name)
    for name in ('prepared/summary.json', 'scoring_fixtures.json', 'resource_preflight.json'):
        inputs[str(ROOT/name)] = sha(ROOT/name)
    for name, digest in inputs.items():
        assert sha(Path(name)) == digest, name
    for name, digest in read(SOURCE_POST/'features_sealed.json')['files'].items():
        assert sha(SOURCE_POST/name) == digest, name
    loaders = [CE_LOADER, CE_MODEL/'download_manifest.json', GATE/'prepare_external_neural.py', GATE/'adapter.py',
               STRONG_QWEN/'qwen.py', STRONG_QWEN/'qwen_config.json', STRONG_QWEN/'assets.py',
               STRONG_QWEN/'evaluation.py', STRONG_QWEN/'model_assets.json', STRONG_QWEN/'official_model_metadata.json',
               Path('/path/to/remote/lw_experiments/runtime.sh')]
    code = ['config.py', 'protocol.md', 'prepare_inputs.py', 'infer_cell.py', 'resource_preflight.py',
            'build_rankings.py', 'score.py', 'freeze.py', 'run_pipeline.py', 'run_gpu.sh', 'launch.sh',
            'test_cells.py', 'test_scoring.py']
    write(ROOT/'frozen_manifest.json', {'frozen_utc': datetime.now(timezone.utc).isoformat(),
          'code': {name: sha(ROOT/name) for name in code}, 'input_sources': inputs,
          'loader_sources': {str(p): sha(p) for p in loaders},
          'scoring_sources': {str(p): sha(p) for p in [REFERENCE_PATH, GROUPS_PATH, PLAN_PATH,
                                                     SOURCE_POST/'prepared/receipt.json', SOURCE_POST/'summary.json']},
          'model_assets_checked_before_freeze': prepared['assets'],
          'primary_contrast': 'input_effect_interaction:P1', 'new_neural_pairs': 138080,
          'timeout_seconds': 14400, 'new_scores_exist': False, 'IFIR_used': False})
    print('FROZEN '+sha(ROOT/'frozen_manifest.json'), flush=True)


if __name__ == '__main__':
    main()
