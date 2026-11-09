"""Single authorized serial pipeline; no automatic model/task retry."""
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import time
from paths import ROOT,read,write,sha,verify_code


def main():
    assert str(ROOT).startswith('/path/to/remote/lw_experiments/')
    m=verify_code();release=read(ROOT/'parent_release.json')
    assert release['frozen_manifest_sha256']==sha(ROOT/'frozen_manifest.json')
    assert not (ROOT/'status').exists(),'Existing production: inspect it, never duplicate'
    pre=read(ROOT/'tokenizer_input_preflight.json');assert all(not pre[d]['overflow'] for d in ['fiqa','nfcorpus'])
    assert os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:assert os.environ.get(key)=='1'
    started=time.monotonic();status=ROOT/'status'
    write(ROOT/'production_started.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'PID':os.getpid()})
    stages=[['fit_source.py'],['prepare_inputs.py'],['retrieve.py'],['prepare_neural.py','--stage','MiniLM'],
            ['run_gpu.sh'],['apply_post.py'],['score.py']]
    try:
        for stage in stages:
            status.write_text('running '+' '.join(stage)+'\n');print('STAGE '+' '.join(stage),flush=True)
            cmd=['bash',stage[0]] if stage[0].endswith('.sh') else [sys.executable,'-B',*stage]
            subprocess.run(cmd,cwd=ROOT,check=True)
        for key in ['training_sources','input_sources','loader_sources','scoring_sources']:
            for path,h in m[key].items():assert sha(Path(path))==h,path
        status.write_text('produced_pending_independent_verification\n')
        write(ROOT/'production_finished.json',{'finished_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.monotonic()-started,
            'exit_code':0,'task_retries':0,'source_hashes_before_after':'matched'})
    except BaseException as exc:
        status.write_text('failed_'+type(exc).__name__+'\n')
        write(ROOT/'production_failed.json',{'finished_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.monotonic()-started,'error':repr(exc)})
        raise


if __name__=='__main__':main()
