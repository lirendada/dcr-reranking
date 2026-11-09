"""Run once after synthetic checks; pin old arrays and all new executable sources."""
from datetime import datetime, timezone
from paths import ROOT, PREV, REF, GROUPS, read, write, sha


def main() -> None:
    assert str(ROOT).startswith('/path/to/remote/lw_experiments/')
    assert read(ROOT/'fixtures.json')['passed']
    assert read(ROOT/'synthetic_timing.json')['estimated_production_seconds'] <= 1800
    assert sha(PREV/'frozen_manifest.json') == '65e1a77a41ce26652e4b7faa8365e17b7e4425a36035e713605ba702ea091ea7'
    sources = [PREV/n for n in ['features.npz', 'head_and_tail.json', 'features_sealed.json',
               'routes_sealed.json', 'frozen_manifest.json', 'independent_verification.json', 'post_math.py',
               'summary.json', 'paired_bootstrap.csv', 'prepared/plan.npz', 'prepared/receipt.json']]
    for fold in range(5):
        sources += [PREV/'prepared'/f'{kind}_{fold}.{ext}' for kind, ext in [('labels', 'npz'), ('view', 'json')]]
        sources += [PREV/'fitted'/f'fold_{fold}.{ext}' for ext in ['npz', 'json']]
    sources += [REF, GROUPS]
    for name, digest in read(PREV/'features_sealed.json')['files'].items():
        assert sha(PREV/name) == digest
    for name, digest in read(PREV/'routes_sealed.json')['files'].items():
        assert sha(PREV/name) == digest
    write(ROOT/'frozen_manifest.json', {'frozen_utc': datetime.now(timezone.utc).isoformat(),
          'code': {n: sha(ROOT/n) for n in ['math_core.py', 'paths.py', 'fit_fold.py', 'score.py',
                   'freeze.py', 'run_production.py', 'fixtures.py', 'benchmark.py', 'protocol.md']},
          'sources': {str(p): sha(p) for p in sources}, 'model_calls': 0, 'IFIR_inputs_used': False})


if __name__ == '__main__':
    main()
