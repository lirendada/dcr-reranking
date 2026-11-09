"""Bounded CPU-only analysis and same-author numerical verification."""
import os
import resource
import subprocess
import sys
import time
from config import ROOT, write, read, sha, check, now


def main() -> None:
    for key in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
        assert os.environ[key] == '1'
    assert os.environ['PYTHONDONTWRITEBYTECODE'] == '1'
    assert read(ROOT / 'parent_release.json')['manifest_sha256'] == sha(ROOT / 'frozen_manifest.json')
    check()
    started = time.monotonic()
    write(ROOT / 'production_started.json', {'started_utc': now(), 'pid': os.getpid(), 'new_neural_pairs': 0})
    (ROOT / 'fitted').mkdir(exist_ok=False)
    stages = []
    try:
        jobs = [(f'fit_fold_{f}', ['fit.py', '--fold', str(f)]) for f in range(5)]
        jobs += [('source_transfer_fit', ['fit.py', '--transfer']), ('seal_target_predictions', ['predict.py']),
                 ('score', ['score.py']), ('numerical_verification', ['verify.py'])]
        for stage, command in jobs:
            (ROOT / 'status').write_text(stage + '\n')
            begin = time.monotonic()
            process = subprocess.run([sys.executable, '-B', *command], cwd=ROOT, check=False)
            record = {'stage': stage, 'exit_code': process.returncode, 'seconds': time.monotonic() - begin}
            stages.append(record)
            write(ROOT / 'stages' / f'{stage}.json', record)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        assert read(ROOT / 'verification.json')['status'] == 'PASS'
        check()
        write(ROOT / 'completion_receipt.json', {'status': 'PASS', 'finished_utc': now(), 'exit_code': 0,
              'wall_seconds': time.monotonic() - started, 'stages': stages, 'new_neural_pairs': 0,
              'GPU_used': False, 'source_hashes_before_after': 'PASS', 'task_retries': 0,
              'peak_child_RSS_KiB': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
              'same_agent_authored_verification': True})
        (ROOT / 'status').write_text('complete_verified\n')
    except BaseException as error:
        write(ROOT / 'production_failed.json', {'finished_utc': now(), 'seconds': time.monotonic() - started,
              'error': repr(error), 'stages': stages})
        (ROOT / 'status').write_text('failed\n')
        raise


if __name__ == '__main__':
    main()
