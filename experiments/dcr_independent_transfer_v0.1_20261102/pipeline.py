"""Serial child stages with real exits and resource checks; no automatic retries."""
from datetime import datetime, timezone
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time

from config import ROOT, GPU_LOCK, CELLS, read, write, sha, check_release, check_sources

RESEARCH = '/path/to/remote/lw_experiments/envs/research/bin/python'
QWEN = '/path/to/remote/lw_experiments/envs/qwen3-reranker/bin/python'


def resource_gate() -> dict:
    assert Path('/proc/self/fd/9').resolve() == GPU_LOCK
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,memory.total',
                                      '--format=csv,noheader,nounits'], text=True).splitlines()
    free, total = [int(x.strip()) for x in result[0].split(',')]
    own = {int(x) for x in subprocess.check_output(['ps', '-u', str(os.getuid()), '-o', 'pid='], text=True).split()}
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    own_gpu = [int(x.strip()) for x in active.splitlines() if x.strip().isdigit() and int(x.strip()) in own]
    assert not own_gpu, ('Existing GPU process under this account', own_gpu)
    assert free >= 20480, ('Insufficient free GPU memory', free)
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    assert available >= 16 * 1024 ** 3
    assert shutil.disk_usage(ROOT).free >= 2 * 1024 ** 3
    return {'checked_utc': datetime.now(timezone.utc).isoformat(), 'free_MiB': free,
            'total_MiB': total, 'RAM_available_bytes': available, 'disk_free_bytes': shutil.disk_usage(ROOT).free,
            'own_GPU_PIDs_before_model': own_gpu, 'other_users_not_modified': True}


def run(python: str, script: str, *arguments, scoring=False) -> dict:
    env = os.environ.copy()
    env['DCR_SCORING_PHASE'] = '1' if scoring else '0'
    start = time.monotonic()
    print('STAGE ' + script + ' ' + ' '.join(arguments), flush=True)
    result = subprocess.run([python, '-B', str(ROOT / script), *arguments], env=env, pass_fds=(9,))
    receipt = {'stage': script, 'arguments': list(arguments), 'exit_code': result.returncode,
               'seconds': time.monotonic() - start}
    print(str(receipt), flush=True)
    if result.returncode:
        raise RuntimeError(f'Stage failed: {receipt}')
    return receipt


def main() -> None:
    check_release()
    start = time.monotonic()
    write(ROOT / 'production_started.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
          'freeze_sha256': sha(ROOT / 'frozen_manifest.json'), 'pid': os.getpid()})
    stages, gates = [], []
    try:
        for cell in CELLS:
            gates.append(resource_gate())
            stages.append(run(RESEARCH if cell.startswith('MiniLM') else QWEN, 'infer.py', '--cell', cell))
        stages.append(run(RESEARCH, 'rank.py'))
        stages.append(run(RESEARCH, 'score.py', scoring=True))
        stages.append(run(RESEARCH, 'verify.py', scoring=True))
        from model_contract import verify_model_assets
        assert verify_model_assets() == read(ROOT / 'model_assets_sealed.json')
        write(ROOT / 'model_assets_after.json', {'status': 'PASS', 'all_asset_hashes_match_before': True})
        check_sources()
        write(ROOT / 'execution_receipt.json', {'exit_code': 0, 'stages': stages, 'GPU_gates': gates,
            'finished_utc': datetime.now(timezone.utc).isoformat(), 'wall_seconds': time.monotonic() - start,
            'new_model_pairs': 6880, 'task_retries': 0, 'source_refit': False,
            'verification_status': read(ROOT / 'verification.json')['status']})
    except BaseException as error:
        write(ROOT / 'production_failed.json', {'error_type': type(error).__name__, 'error': str(error),
            'finished_utc': datetime.now(timezone.utc).isoformat(), 'completed_stages': stages,
            'seconds': time.monotonic() - start})
        raise


if __name__ == '__main__':
    main()
