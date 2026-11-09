"""Unique serial CPU production, released after review/preflight; no retry."""
from datetime import datetime,timezone
import os,subprocess,sys,time
from pathlib import Path
from config import ROOT,read,write,sha,check_code


def main():
 assert str(ROOT).startswith('/path/to/remote/lw_experiments/')
 manifest=check_code();assert read(ROOT/'parent_release.json')['manifest_sha256']==sha(ROOT/'frozen_manifest.json')
 assert not (ROOT/'status').exists()
 for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:assert os.environ.get(key)=='1'
 start=time.monotonic();status=ROOT/'status';write(ROOT/'production_started.json',{'PID':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat()})
 stages=[['features.py']]+[['fit_fold.py','--fold',str(f)] for f in range(5)]+[['fit_transfer.py'],['seal.py'],['score.py']]
 try:
  for stage in stages:
   status.write_text('running '+' '.join(stage)+'\n');print('STAGE '+' '.join(stage),flush=True)
   subprocess.run([sys.executable,'-B',*stage],cwd=ROOT,check=True)
  check_code()
  for group in ['feature_sources','kernel_sources','reuse_sources','transfer_sources','scoring_sources']:
   for p,h in manifest[group].items():assert sha(Path(p))==h,p
  status.write_text('produced_pending_independent_verification\n')
  write(ROOT/'production_finished.json',{'exit_code':0,'finished_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.monotonic()-start,'new_neural_pairs':0,'task_retries':0})
 except BaseException as e:
  status.write_text('failed_'+type(e).__name__+'\n');write(ROOT/'production_failed.json',{'error':repr(e),'wall_seconds':time.monotonic()-start,'recorded_utc':datetime.now(timezone.utc).isoformat()});raise


if __name__=='__main__':main()
