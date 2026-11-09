"""Freeze the complete topic-retrieval factorial before any new neural score exists."""
from datetime import datetime,timezone
from pathlib import Path
from config import ROOT,GATE,STRONG_QWEN,CE_LOADER,CE_MODEL,SOURCE_POST,PLAN_PATH,GROUPS_PATH,REFERENCE_PATH,read,write,sha
from model_contract import verify_model_assets


def main():
    assert str(ROOT).startswith('/path/to/remote/lw_experiments/')
    assert not (ROOT/'selection').exists() and not (ROOT/'cells').exists()
    preparation=read(ROOT/'external_prepared/summary.json')
    assert preparation['pairs']==345200 and sha(ROOT/'external_prepared/pairs.jsonl.gz')==preparation['prepared_sha256']
    assert read(ROOT/'resource_preflight.json')['passed']
    assert read(ROOT/'scoring_fixtures.json')['passed']
    historical=read(ROOT/'historical_contract.json')['historical_hashes']
    for path,digest in historical.items():assert sha(Path(path))==digest,path
    assets=verify_model_assets()
    code=['historical_contract.json','PROTOCOL.md','config.py','model_contract.py','prepare_candidates.py','prepare_prefixes.py','prepare_head.py','infer_cell.py','build_rankings.py','score.py','freeze.py','resource_preflight.py','run_cpu_prepare.sh','run_gpu.sh','run_pipeline.py','launch.sh','test_scoring.py']
    inputs=[GATE/'external_inputs'/f for f in ['metadata.json','corpus.jsonl','queries_question0.jsonl','queries_RQ_rewrite.jsonl']]
    inputs += [ROOT/f for f in ['external_retrieval/candidates.csv','external_retrieval/candidates_frozen.json','external_retrieval/bm25_top100.csv','external_prepared/pairs.jsonl.gz','external_prepared/summary.json','prefix_preparation_receipt.json']]
    loaders=[GATE/'retrieve_external.py',ROOT.parent/'dot_experiment_bundle/research/dcr_public_baselines_v0.1_20261027/frozen_parser.py',CE_LOADER,CE_MODEL/'download_manifest.json',GATE/'prepare_external_neural.py',GATE/'adapter.py',STRONG_QWEN/'qwen.py',STRONG_QWEN/'qwen_config.json',STRONG_QWEN/'assets.py',STRONG_QWEN/'evaluation.py',STRONG_QWEN/'model_assets.json',STRONG_QWEN/'official_model_metadata.json',Path('/path/to/remote/lw_experiments/runtime.sh')]
    scoring=[REFERENCE_PATH,GROUPS_PATH,PLAN_PATH,SOURCE_POST/'head_and_tail.json',ROOT.parent/'dcr_model_input_v0.1_20261029/summary.json']
    write(ROOT/'frozen_manifest.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'code':{f:sha(ROOT/f) for f in code},'input_sources':{str(p):sha(p) for p in inputs},'loader_sources':{str(p):sha(p) for p in loaders},'scoring_sources':{str(p):sha(p) for p in scoring},'historical_contract_sha256':sha(ROOT/'historical_contract.json'),'model_assets_checked_before_freeze':assets,'primary_contrast':'input_effect_interaction:P1','new_neural_pairs':552320,'timeout_seconds':14400,'new_scores_exist':False,'IFIR_used':False})
    print('FROZEN '+sha(ROOT/'frozen_manifest.json'),flush=True)


if __name__=='__main__':main()
