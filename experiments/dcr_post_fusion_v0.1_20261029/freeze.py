"""Freeze reviewed code and source hashes before source preflight and fitting."""
from datetime import datetime,timezone
import platform
import numpy as np
from paths import ROOT,feature_sources,reference_sources,baseline_sources,scoring_sources,sha,write
FILES=['protocol.md','paths.py','post_math.py','prepare_features.py','prepare_labels.py','check_fixed.py','fit_fold.py',
       'seal_routes.py','score.py','freeze.py','run_production.py','benchmark.py','test_post_math.py']
write(ROOT/'frozen_manifest.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),
  'code':{n:sha(ROOT/n) for n in FILES},
  'feature_sources':{str(p):sha(p) for p in feature_sources()},'reference_sources':{str(p):sha(p) for p in reference_sources()},
  'baseline_sources':{str(p):sha(p) for p in baseline_sources()},'scoring_sources':{str(p):sha(p) for p in scoring_sources()},
  'synthetic_timing_sha256':sha(ROOT/'synthetic_timing.json'),
  'runtime':{'Python':platform.python_version(),'NumPy':np.__version__,'BLAS_threads':1},
  'config':{'head':20,'G_dimensions':12,'C_dimensions':6,'draws':1000,'seed':20261029,
    'lambda_grid':[.001,.01,.1,1.],'alpha_grid':[i/20 for i in range(21)],
    'weighted_training_order':'queryID ascending; integer component replicate counts','intercept':False}})
