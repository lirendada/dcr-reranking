#!/usr/bin/env python3
"""Seal execution code and phase-specific original assets once, before precheck."""
from datetime import datetime, timezone
from run_null_k10 import ROOT, POST, CELLS, MATCHED, ATTR, REF, read, write, sha

code = ['PROTOCOL.md', 'EXECUTION_NOTES.md', 'run_null_k10.py', 'freeze.py', 'run_pipeline.py', 'run_cpu.sh']
common = [POST/'features.npz', POST/'prepared/plan.npz', MATCHED/'features.npz', ATTR/'math_core.py']
score = [REF, POST/'head_and_tail.json', CELLS/'four_cell_arrays.npz', CELLS/'bootstrap_metrics.npz',
         CELLS/'summary.json', MATCHED/'summary.json', ATTR/'summary.json']
manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
            'code': {str(ROOT/p): sha(ROOT/p) for p in code},
            'common': {str(p): sha(p) for p in common}, 'score': {str(p): sha(p) for p in score}}
for fold in range(5):
    view_path = POST/'prepared'/f'view_{fold}.json'
    view = read(view_path)
    paths = [view_path, view['labels_path'], POST/'fitted'/f'fold_{fold}.npz', MATCHED/'fitted'/f'fold_{fold}.npz']
    manifest[f'fold_{fold}'] = {str(p): sha(p) for p in paths}
write(ROOT/'frozen_manifest.json', manifest)
print(sha(ROOT/'frozen_manifest.json'))
