"""Serial CPU analysis. Exclusive outputs and preserved failure receipt."""
from datetime import datetime, timezone
import os
import subprocess
import sys
import time
from paths import ROOT, PREV, read, write, sha, check_code, check_sources


def main() -> None:
    assert str(ROOT).startswith('/path/to/remote/lw_experiments/')
    assert read(ROOT/'parent_release.json')['frozen_manifest_sha256'] == sha(ROOT/'frozen_manifest.json')
    assert not (ROOT/'status').exists()
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        assert os.environ[key] == '1'
    assert os.environ['PYTHONDONTWRITEBYTECODE'] == '1'
    manifest = check_code()
    check_sources(manifest)
    started = time.monotonic()
    write(ROOT/'production_started.json', {'PID': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat()})
    (ROOT/'fitted').mkdir(exist_ok=False)
    try:
        for fold in range(5):
            (ROOT/'status').write_text(f'running fold_{fold}\n')
            subprocess.run([sys.executable, '-B', 'fit_fold.py', '--fold', str(fold)], cwd=ROOT, check=True)
        write(ROOT/'predictions_sealed.json', {'sealed_utc': datetime.now(timezone.utc).isoformat(),
              'files': {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/'fitted').glob('*'))},
              'reused_G_routes_sha256': sha(PREV/'routes_sealed.json'),
              'frozen_manifest_sha256': sha(ROOT/'frozen_manifest.json'), 'scoring_references_opened': False})
        (ROOT/'status').write_text('running scoring\n')
        subprocess.run([sys.executable, '-B', 'score.py'], cwd=ROOT, check=True)
        check_code()
        check_sources(manifest)
        write(ROOT/'production_finished.json', {'finished_utc': datetime.now(timezone.utc).isoformat(),
              'wall_seconds': time.monotonic()-started, 'exit_code': 0, 'task_retries': 0,
              'foundation_model_calls': 0, 'GPU_used': False, 'sources_before_after': 'matched'})
        (ROOT/'status').write_text('produced_pending_verification\n')
    except BaseException as exc:
        write(ROOT/'production_failed.json', {'finished_utc': datetime.now(timezone.utc).isoformat(),
              'wall_seconds': time.monotonic()-started, 'error': repr(exc)})
        (ROOT/'status').write_text('failed\n')
        raise


if __name__ == '__main__':
    main()
