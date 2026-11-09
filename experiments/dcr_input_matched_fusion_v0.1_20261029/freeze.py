"""Freeze all phases; isolated fits later verify only own view and kernel."""
from datetime import datetime,timezone
from config import ROOT,POST,ATTR,CELLS,REF,sha,write
CODE=['PROTOCOL.md','config.py','features.py','fit_fold.py','fit_transfer.py','seal.py','score.py','fixtures.py','freeze.py','run_pipeline.py','run_cpu.sh']
feature=[POST/'features.npz',POST/'head_and_tail.json',POST/'features_sealed.json',POST/'frozen_manifest.json']+[
 CELLS/'cells'/c/n for c in ['MiniLM_full','Qwen_full'] for n in ['scores.jsonl.gz','receipt.json']]+[CELLS/'frozen_manifest.json']
kernel=[ATTR/'math_core.py']
reuse=[POST/'routes_sealed.json',ATTR/'predictions_sealed.json',ATTR/'metrics_draws.npz',ATTR/'independent_verification.json']+[
 p/'fitted'/f'fold_{f}.npz' for p in [POST,ATTR] for f in range(5)]
transfer=[REF,POST/'head_and_tail.json',POST/'features.npz',POST/'prepared/plan.npz',ATTR/'math_core.py']
scoring=[REF,POST/'features.npz',POST/'head_and_tail.json',POST/'prepared/plan.npz',ATTR/'metrics_draws.npz']
write(ROOT/'frozen_manifest.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'code':{n:sha(ROOT/n) for n in CODE},
 'feature_sources':{str(p):sha(p) for p in feature},'kernel_sources':{str(p):sha(p) for p in kernel},
 'reuse_sources':{str(p):sha(p) for p in reuse},'transfer_sources':{str(p):sha(p) for p in transfer},'scoring_sources':{str(p):sha(p) for p in scoring},
 'view_manifests':{str(POST/'prepared'/f'view_{f}.json'):sha(POST/'prepared'/f'view_{f}.json') for f in range(5)},
 'fixtures_sha256':sha(ROOT/'fixtures.json'),'timing_sha256':sha(ROOT/'synthetic_timing.json'),
 'config':{'outer_folds':5,'bootstrap_draws':1000,'source_lambda_grid':[.001,.01,.1,1.],'new_neural_pairs':0,'CPU_threads':1,'primary':'G_same12-QL5:P1'}})
