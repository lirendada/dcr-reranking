"""Bounded new process groups:4h overall,3h GPU-phase upper bound incl waiting."""
from datetime import datetime,timezone
import os,signal,subprocess,sys,time
from config import ROOT,read,write,sha,check_code,check_A
active=None


def stop_owned():
 global active
 if active is None:return
 try:os.killpg(active.pid,signal.SIGTERM)
 except ProcessLookupError:pass
 try:active.wait(timeout=30)
 except subprocess.TimeoutExpired:
  try:os.killpg(active.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  active.wait()


def interrupted(signum,frame):
 stop_owned();raise SystemExit(128+signum)


def main():
 global active
 assert str(ROOT).startswith('/path/to/remote/lw_experiments/');check_A();check_code()
 assert read(ROOT/'parent_release.json')['frozen_manifest_sha256']==sha(ROOT/'frozen_manifest.json')
 assert not (ROOT/'status').exists()
 for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:assert os.environ.get(key)=='1'
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 started=time.monotonic();deadline=started+14400;gpu_seconds=0
 write(ROOT/'production_started.json',{'PID':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),'overall_seconds_limit':14400,'GPU_stage_limit_seconds':10800})
 try:
  for label,cmd,limit in [('gpu',['bash','run_gpu.sh'],10800),('predict',[sys.executable,'-B','predict.py'],14400),('score',[sys.executable,'-B','score.py'],14400)]:
   (ROOT/'status').write_text('running '+label+'\n');print('STAGE '+label,flush=True);tick=time.monotonic()
   active=subprocess.Popen(cmd,cwd=ROOT,start_new_session=True)
   try:code=active.wait(timeout=min(limit,max(0.,deadline-time.monotonic())))
   except subprocess.TimeoutExpired:
    stop_owned();raise TimeoutError(label+' budget exceeded; owned group stopped')
   if label=='gpu':gpu_seconds=time.monotonic()-tick
   assert code==0,(label,code);active=None
  check_code()
  (ROOT/'status').write_text('produced_pending_independent_verification\n')
  write(ROOT/'production_finished.json',{'exit_code':0,'finished_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.monotonic()-started,'GPU_stage_wall_seconds_including_wait_and_load':gpu_seconds,'new_neural_pairs':72160,'task_retries':0})
 except BaseException as e:
  stop_owned();(ROOT/'status').write_text('failed_'+type(e).__name__+'\n')
  write(ROOT/'production_failed.json',{'error':repr(e),'recorded_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.monotonic()-started,'new_tasks_retried':0});raise


if __name__=='__main__':main()
