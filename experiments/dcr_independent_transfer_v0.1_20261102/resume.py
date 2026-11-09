"""Resume only after the original resource wait timed out without production."""
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys

from config import ROOT, GPU_LOCK, read, write, sha, check_frozen
from model_contract import verify_model_assets


def gate() -> dict:
    gpu = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.free,memory.total',
         '--format=csv,noheader,nounits'], text=True).splitlines()
    free, total = [int(value.strip()) for value in gpu[0].split(',')]
    own = {int(value) for value in subprocess.check_output(
        ['ps', '-u', str(os.getuid()), '-o', 'pid='], text=True).split()}
    active = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid',
         '--format=csv,noheader,nounits'], text=True)
    own_gpu = [int(value.strip()) for value in active.splitlines()
               if value.strip().isdigit() and int(value.strip()) in own]
    available = int(next(line.split()[1] for line in
                        Path('/proc/meminfo').read_text().splitlines()
                        if line.startswith('MemAvailable:'))) * 1024
    record = {'checked_utc': datetime.now(timezone.utc).isoformat(),
              'free_MiB': free, 'total_MiB': total,
              'own_GPU_PIDs': own_gpu, 'RAM_available_bytes': available,
              'disk_free_bytes': shutil.disk_usage(ROOT).free}
    print(record, flush=True)
    if (free < 20480 or own_gpu or available < 16 * 1024 ** 3
            or record['disk_free_bytes'] < 2 * 1024 ** 3):
        raise SystemExit(75)
    return record


def main() -> int:
    assert Path('/path/to/remote/lw_experiments/runs') == ROOT.parent
    assert read(ROOT / 'resource_wait_timeout.json')['production_started'] is False
    for name in ['.launch_once', 'parent_release.json', 'cells',
                 'production_started.json', 'production_failed.json']:
        assert not (ROOT / name).exists(), f'Existing execution state: {name}'
    expected = '7e2bfc4ad2d97ba58fd0df924ac2436051fabbc6f79d6873cc572340323e40fb'
    assert sha(ROOT / 'frozen_manifest.json') == expected
    gate()
    check_frozen()
    assert read(ROOT / 'fixtures.json')['status'] == 'PASS'
    assert verify_model_assets() == read(ROOT / 'model_assets_sealed.json')
    assert Path('/path/to/remote/lw_experiments/logs/haccp_remaining.status').read_text().strip() == 'completed'
    with GPU_LOCK.open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resources = gate()
        write(ROOT / 'parent_release.json', {
            'released_utc': datetime.now(timezone.utc).isoformat(),
            'freeze_sha256': expected,
            'source_review_sha256': sha(ROOT / 'SOURCE_REVIEW.md'),
            'resume_script_sha256': sha(Path(__file__)),
            'authorization': 'User authorized continuing the post-v0.10 plan on 2026-11-02.',
            'resource_gate': resources,
            'new_model_pairs': 6880,
            'primary': 'G_same12 minus QL5 P1, all 43 CDS requests',
            'human_peer_review': False,
            'review_kind': 'Same-assistant source review and fixtures; separate numerical reconstruction follows scoring.'})
    return subprocess.run(['bash', str(ROOT / 'launch.sh')]).returncode


if __name__ == '__main__':
    sys.exit(main())
