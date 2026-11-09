#!/usr/bin/env python3
"""Sequential CPU orchestration with real child statuses and one launch per stage."""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from run_null_k10 import ROOT, read, write, sha

parser = argparse.ArgumentParser()
parser.add_argument('--precheck', action='store_true')
args = parser.parse_args()
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    assert os.environ.get(key) == '1', key
phase = 'precheck' if args.precheck else 'production'
folder = ROOT/'precheck' if args.precheck else ROOT
folder.mkdir(exist_ok=True)
if not args.precheck:
    approval = read(ROOT/'root_approval.json')
    assert approval['status'] == 'PASS'
    assert approval['frozen_manifest_sha256'] == sha(ROOT/'frozen_manifest.json')
    assert read(ROOT/'precheck/result.json')['status'] == 'PASS'
started = time.monotonic()
write(folder/f'{phase}_started.json', {'pid': os.getpid(), 'utc': datetime.now(timezone.utc).isoformat(),
      'frozen_manifest_sha256': sha(ROOT/'frozen_manifest.json')})
statuses = []
exit_code = 0
for fold in range(5):
    command = [sys.executable, '-B', str(ROOT/'run_null_k10.py'), 'fit', '--fold', str(fold)]
    if args.precheck:
        command.append('--precheck')
    with (folder/f'{phase}_fold_{fold}.log').open('x') as log:
        status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    statuses.append({'fold': fold, 'exit_code': status})
    if status:
        exit_code = status
        break
if not exit_code:
    if not args.precheck:
        files = [p for p in (ROOT/'fits').iterdir() if p.suffix in ('.json', '.npz')]
        write(ROOT/'fits_sealed.json', {'utc': datetime.now(timezone.utc).isoformat(),
              'frozen_manifest_sha256': sha(ROOT/'frozen_manifest.json'),
              'files': {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}})
    command = [sys.executable, '-B', str(ROOT/'run_null_k10.py'), 'score']
    if args.precheck:
        command.append('--precheck')
    with (folder/f'{phase}_score.log').open('x') as log:
        exit_code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    statuses.append({'score_exit_code': exit_code})
write(folder/'execution_receipt.json', {'stage': phase, 'exit_code': exit_code,
      'finished_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic()-started,
      'children': statuses, 'new_neural_pairs': 0, 'cpu_threads': 1,
      'frozen_manifest_sha256': sha(ROOT/'frozen_manifest.json')})
print(phase, 'exit', exit_code, flush=True)
sys.exit(exit_code)
