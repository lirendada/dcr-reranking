"""Seal new and reused prediction provenance before any all-query reference scoring."""
from datetime import datetime,timezone
from pathlib import Path
from config import ROOT,read,write,sha,check_code,check_phase


def main():
 manifest=check_code();check_phase('reuse');files={}
 for f in range(5):
  p=ROOT/'fitted'/f'fold_{f}.npz';r=read(p.with_suffix('.json'));assert sha(p)==r['sha256']
  for item in [p,p.with_suffix('.json')]:files[str(item.relative_to(ROOT))]=sha(item)
 for p in ['features.npz','features_sealed.json','transfer_models.json','transfer_models_sealed.json']:files[p]=sha(ROOT/p)
 write(ROOT/'predictions_sealed.json',{'sealed_utc':datetime.now(timezone.utc).isoformat(),'files':files,
    'reused_sources':manifest['reuse_sources'],'frozen_manifest_sha256':sha(ROOT/'frozen_manifest.json'),'scoring_references_opened':False})


if __name__=='__main__':main()
