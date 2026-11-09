"""Hash every input/code dependency and check Q7 invariance before launch."""
from pathlib import Path
import platform
import shutil
import sys
import numpy as np
from config import ROOT, POST, ATTR, SAME, IFIR, CDS, REF, Q7, M5, TARGETS, read, write, sha, target_spec, now


def main() -> None:
    assert str(ROOT).startswith('/path/to/remote/lw_experiments/runs/')
    assert read(ROOT / 'fixtures.json')['status'] == 'PASS'
    assert not (ROOT / 'production_started.json').exists()
    assert shutil.disk_usage(ROOT).free > 1024 ** 3
    files = [REF, POST / 'features.npz', POST / 'head_and_tail.json', POST / 'prepared/plan.npz',
             ATTR / 'math_core.py', SAME / 'features.npz', SAME / 'metrics_draws.npz', SAME / 'point_orders.npz',
             SAME / 'transfer_models.json', SAME / 'transfer_models_sealed.json']
    for fold in range(5):
        view_path = POST / 'prepared' / f'view_{fold}.json'
        view = read(view_path)
        files += [view_path, Path(view['labels_path'])]
        for name, expected in view['allowed_inputs'].items():
            assert sha(Path(name)) == expected, name
            files.append(Path(name))
        for directory in [POST, ATTR, SAME]:
            files.append(directory / 'fitted' / f'fold_{fold}.npz')
    with np.load(POST / 'features.npz', allow_pickle=False) as old, np.load(SAME / 'features.npz', allow_pickle=False) as same:
        assert np.array_equal(old['X'][:, :, Q7], same['X'][:, :, Q7])
        assert old['X'].shape == (3452, 20, 18)
    for dataset in TARGETS:
        spec = target_spec(dataset)
        files.extend(spec[k] for k in ['features', 'meta', 'rankings', 'qrels', 'summary'])
        if dataset != 'cds':
            files += [spec['queries'], IFIR / dataset / 'outputs/features.npz']
        assert spec['qrels'].read_text().splitlines()[0].split('\t') == ['query_id', 'corpus_id', 'score']
    for directory in [POST, ATTR, SAME, IFIR, CDS]:
        for name in ['frozen_manifest.json', 'independent_verification.json', 'verification.json', 'production_finished.json',
                     'completion_receipt.json', 'source_contract.json', 'predictions_sealed.json']:
            if (directory / name).is_file():
                files.append(directory / name)
    files += [CDS / 'header_alias_recovery_20261103/recovery_receipt.json',
              CDS / 'header_alias_recovery_20261103/repair_manifest.json']
    for directory in [ATTR, SAME, IFIR]:
        status = read(directory / 'independent_verification.json')['status']
        assert status.startswith('PASS'), (str(directory), status)
    assert read(CDS / 'verification.json')['status'] == 'PASS'
    source = {str(p): sha(p) for p in sorted(set(files))}
    code = {p.name: sha(p) for p in sorted(ROOT.iterdir()) if p.suffix in {'.py', '.sh'} or p.name == 'PROTOCOL.md'}
    write(ROOT / 'frozen_manifest.json', {'frozen_utc': now(), 'code': code, 'sources': source,
          'fixtures_sha256': sha(ROOT / 'fixtures.json'),
          'Q7_columns': Q7, 'MiniLM_dependent_columns': M5,
          'primary': 'ExcluIR:G_same12-Q7:P1', 'interval_count': 18, 'new_neural_pairs': 0,
          'environment': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
                          'threads': 1, 'timeout_seconds': 14400, 'available_disk_bytes': shutil.disk_usage(ROOT).free}})
    print(f'frozen {len(code)} code/protocol and {len(source)} read-only inputs', flush=True)


if __name__ == '__main__':
    main()
