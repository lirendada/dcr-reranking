"""Small point-parameter export after production; never changes scores or routes."""
import numpy as np
from paths import ROOT,read,write,sha
qids=read(ROOT/'head_and_tail.json')['qids']
result={}
for f in range(5):
    path=ROOT/'fitted'/f'fold_{f}.npz'
    with np.load(path,allow_pickle=False) as a:
        record={'array_sha256':sha(path),'test_qids':[qids[i] for i in a['test_indices']],
                'point_alpha':float(a['alpha'][0]),'models':{}}
        for key in ['G','GC']:
            record['models'][key]={'lambda':float(a[key+'_lambda'][0]),
                'scale':a[key+'_scale'][0].tolist(),'coef':a[key+'_coef'][0].tolist(),
                'inner_correct_counts_by_lambda':a[key+'_inner_correct_counts'][0].tolist(),
                'pair_instances':int(a[key+'_pair_instances'][0])}
        result[str(f)]=record
write(ROOT/'point_models.json',result)
