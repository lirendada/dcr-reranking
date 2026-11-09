"""Only permutations and hashes: no reference labels are read."""
from datetime import datetime,timezone
import numpy as np
from paths import ROOT,read,write,sha,verify_code


def main():
    verify_code();assert read(ROOT/'fixed_source_check.json')['all_complete_head_and_tail_identical']
    files={}
    for fold in range(5):
        p=ROOT/'fitted'/f'fold_{fold}.npz'
        receipt=read(p.with_suffix('.json'));assert receipt['file_sha256']==sha(p)
        with np.load(p,allow_pickle=False) as a:
            for key in ['linear_order','G_order','GC_order']:
                order=a[key];assert np.all(np.sort(order,axis=2)==np.arange(20))
        for f in [p,p.with_suffix('.json')]:files[str(f.relative_to(ROOT))]=sha(f)
    write(ROOT/'routes_sealed.json',{'sealed_utc':datetime.now(timezone.utc).isoformat(),'files':files,
        'frozen_manifest_sha256':sha(ROOT/'frozen_manifest.json'),'features_sealed_sha256':sha(ROOT/'features_sealed.json'),
        'fixed_source_check_sha256':sha(ROOT/'fixed_source_check.json'),'scoring_references_opened':False})


if __name__=='__main__':main()
