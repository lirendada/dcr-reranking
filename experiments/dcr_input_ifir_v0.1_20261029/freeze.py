"""Freeze root+generated dataset contexts; control copies are same global bytes, not hash inputs."""
from datetime import datetime,timezone
from pathlib import Path
from config import ROOT,A,OLD,EXCL_FOUR,GATE,STRONG,CE_LOADER,CE_MODEL,DATASETS,read,write,sha,check_A
check_A();prep=read(ROOT/'preparation.json');assert prep['model_loaded'] is False and not prep['qrels_opened']
CODE=['PROTOCOL.md','config.py','prepare.py','feature_kernel.py','predict.py','score.py','night_gate.py','run_gpu.sh','run_pipeline.py','freeze.py','fixtures.py']
CODE += [f'{d}/{n}' for d in DATASETS for n in ['config.py','prepare_inputs.py','infer_cell.py']]
inputs=[A/'transfer_models.json',A/'transfer_models_sealed.json',A/'independent_verification.json',OLD/'source_model.json',OLD/'model_sealed.json',ROOT/'preparation.json']
inputs += [ROOT/n for n in prep['outputs']]+[Path(p) for p in prep['sources']]
inputs += [OLD/d/'outputs/rankings.jsonl.gz' for d in DATASETS]
inputs += [OLD/'inputs_sealed.json',OLD/'retrieval_receipt.json',OLD/'predictions_sealed.json']+[OLD/d/'retrieval/candidates.csv' for d in DATASETS]
loaders=[Path('/path/to/remote/lw_experiments/runtime.sh'),CE_LOADER,CE_MODEL/'download_manifest.json',EXCL_FOUR/'infer_cell.py',GATE/'prepare_external_neural.py']+[
 STRONG/n for n in ['qwen.py','qwen_config.json','model_assets.json','official_model_metadata.json','evaluation.py','assets.py']]
scoring=[OLD/'raw'/d/f'{d}-qrel.tsv' for d in DATASETS]+[OLD/'summary.json',OLD/'metrics.py',OLD/'predictions_sealed.json']
write(ROOT/'frozen_manifest.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'code':{n:sha(ROOT/n) for n in CODE},
 'input_sources':{str(p):sha(p) for p in inputs},'loader_sources':{str(p):sha(p) for p in loaders},'scoring_sources':{str(p):sha(p) for p in scoring},
 'preparation_sha256':sha(ROOT/'preparation.json'),'resource_preflight_sha256':sha(ROOT/'resource_preflight.json'),'fixtures_sha256':sha(ROOT/'fixtures.json'),
 'config':{'new_neural_pairs':72160,'P1_CIs':24,'bootstrap':10000,'seed':20261029,'cutoff_Beijing':'2026-10-30T04:00:00+08:00','CPU_threads':1}})
# The mature child infer reads these local controls; byte identity is asserted.
for d in DATASETS:
 with (ROOT/d/'frozen_manifest.json').open('xb') as f:f.write((ROOT/'frozen_manifest.json').read_bytes())
 assert sha(ROOT/d/'frozen_manifest.json')==sha(ROOT/'frozen_manifest.json')
