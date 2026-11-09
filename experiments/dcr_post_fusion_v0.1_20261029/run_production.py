"""Serial CPU execution after authorized preflight; no task retries."""
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import time
from paths import ROOT,read,write,sha,verify_code


def main():
    assert str(ROOT).startswith('/path/to/remote/lw_experiments/')
    release=read(ROOT/'parent_release.json')
    assert release['frozen_manifest_sha256']==sha(ROOT/'frozen_manifest.json')
    assert read(ROOT/'fixed_source_check.json')['all_complete_head_and_tail_identical']
    verify_code();assert not (ROOT/'status').exists(),'Do not duplicate or blindly restart'
    assert os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:assert os.environ.get(key)=='1'
    started=time.monotonic();status=ROOT/'status'
    write(ROOT/'production_started.json',{'PID':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat()})
    stages=[['prepare_labels.py']]+[['fit_fold.py','--fold',str(f)] for f in range(5)]+[['seal_routes.py'],['score.py']]
    try:
        for args in stages:
            status.write_text('running '+' '.join(args)+'\n');print('STAGE '+' '.join(args),flush=True)
            subprocess.run([sys.executable,'-B',*args],cwd=ROOT,check=True)
        m=verify_code()
        for group in ['feature_sources','reference_sources','baseline_sources','scoring_sources']:
            for p,h in m[group].items():assert sha(Path(p))==h
        status.write_text('produced_pending_independent_verification\n')
        write(ROOT/'production_finished.json',{'finished_utc':datetime.now(timezone.utc).isoformat(),'exit_code':0,
           'CPU_task_wall_seconds':time.monotonic()-started,'foundation_model_calls':0,'GPU_used':False,
           'source_hashes_before_after':'matched'})
    except BaseException as exc:
        status.write_text('failed_'+type(exc).__name__+'\n')
        write(ROOT/'production_failed.json',{'finished_utc':datetime.now(timezone.utc).isoformat(),'error':repr(exc),
           'CPU_task_wall_seconds':time.monotonic()-started})
        raise


if __name__=='__main__':main()
