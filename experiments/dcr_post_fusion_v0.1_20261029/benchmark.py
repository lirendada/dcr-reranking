"""Ten full same-size synthetic refit kernels, no real inputs opened."""
import json
import platform
import resource
from pathlib import Path
import time
import numpy as np
from post_math import select_alpha,select_lambda,ridge_fit,ridge_scores,rank_positions
ROOT=Path(__file__).resolve().parent
assert str(ROOT).startswith('/path/to/remote/lw_experiments/')
rng=np.random.default_rng(20261029)
x=rng.normal(size=(3452,20,18));canonical=np.tile(np.arange(20),(3452,1))
fold=np.repeat(np.arange(5),[691,691,690,690,690])
positive=np.where(rng.random(3452)<.8,rng.integers(0,20,3452),-1)
negative=np.where(rng.random(3452)<.8,rng.integers(0,20,3452),-1)
negative=np.where(negative==positive,(negative+1)%20,negative)
times=[]
for d in range(10):
    start=time.monotonic();weights=rng.poisson(1,size=3452).astype(np.int16)
    for f in range(5):
        train=weights*(fold!=f);test=np.flatnonzero(fold==f)
        select_alpha(x,positive,canonical,train)
        for p in [12,18]:
            lam,_=select_lambda(x,positive,negative,canonical,fold,f,weights,p)
            model=ridge_fit(x,positive,negative,train,lam,p)
            rank_positions(ridge_scores(x[test],model),canonical[test])
    times.append(time.monotonic()-start)
    print(json.dumps({'kernel':d+1,'wall_seconds':times[-1]}),flush=True)
with (ROOT/'synthetic_timing.json').open('x') as out:
    json.dump({'synthetic_only':True,'real_inputs_opened':False,'seconds_each':times,
      'projected_1000_draw_seconds':sum(times)*100,'Python':platform.python_version(),'NumPy':np.__version__,
      'BLAS_threads':1,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},out,indent=2)
