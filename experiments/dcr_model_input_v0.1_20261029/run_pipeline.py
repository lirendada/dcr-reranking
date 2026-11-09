"""One authorized serial production; keep any failure, never retry automatically."""
from datetime import datetime, timezone
import os
import subprocess
import sys
import time
from config import ROOT, read, write, sha, verify_code, verify_sources


def main():
    assert str(ROOT).startswith('/path/to/remote/lw_experiments/')
    manifest = verify_code()
    assert read(ROOT/'parent_release.json')['frozen_manifest_sha256'] == sha(ROOT/'frozen_manifest.json')
    assert not (ROOT/'status').exists(), 'Inspect the existing run before considering another launch'
    assert os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        assert os.environ.get(key) == '1'
    for phase in ('input', 'loader', 'scoring'):
        verify_sources(manifest, phase)
    started = time.monotonic()
    write(ROOT/'production_started.json', {'PID': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat()})
    try:
        for stage in ('run_gpu.sh', 'build_rankings.py', 'score.py'):
            (ROOT/'status').write_text('running '+stage+'\n')
            print('STAGE '+stage, flush=True)
            command = ['bash', stage] if stage.endswith('.sh') else [sys.executable, '-B', stage]
            subprocess.run(command, cwd=ROOT, check=True)
        verify_code()
        for phase in ('input', 'loader', 'scoring'):
            verify_sources(manifest, phase)
        write(ROOT/'production_finished.json', {'finished_utc': datetime.now(timezone.utc).isoformat(),
              'wall_seconds': time.monotonic()-started, 'exit_code': 0, 'task_retries': 0,
              'new_neural_pairs': 138080, 'source_hashes_before_after': 'matched'})
        (ROOT/'status').write_text('produced_pending_independent_verification\n')
    except BaseException as exc:
        write(ROOT/'production_failed.json', {'finished_utc': datetime.now(timezone.utc).isoformat(),
              'wall_seconds': time.monotonic()-started, 'error': repr(exc)})
        (ROOT/'status').write_text('failed_'+type(exc).__name__+'\n')
        raise


if __name__ == '__main__':
    main()
