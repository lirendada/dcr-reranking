"""Run the unchanged historical tokenizer-only prefix preparation in the new directory."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
GATE=ROOT.parent/'dcr_gate_mechanism_v0.1_20261029'
sys.path.insert(0,str(GATE))
from adapter import sha256,write_json
import prepare_external_neural as old


def main():
    assert str(ROOT).startswith('/path/to/remote/lw_experiments/')
    link=ROOT/'external_inputs'
    if not link.exists():link.symlink_to(GATE/'external_inputs',target_is_directory=True)
    assert link.resolve()==(GATE/'external_inputs').resolve()
    source=GATE/'prepare_external_neural.py';before=sha256(source)
    old.ROOT=ROOT
    old.main()
    assert sha256(source)==before
    # Original cost projections described full-Qwen top100; they do not describe this run.
    p=ROOT/'external_prepared/summary.json';summary=json.loads(p.read_text())
    summary['historical_preparer_projection_only']= {k:summary.pop(k) for k in ['planned_model_pairs','prior_Qwen_pairs_per_second','projected_Qwen_seconds_at_prior_speed','projection_not_actual_external_inference']}
    summary['planned_new_pairs']={'MiniLM_topic':345200,'MiniLM_full':69040,'Qwen_topic':69040,'Qwen_full':69040}
    summary['preparer_source_sha256']=before
    p.write_text(json.dumps(summary,indent=2)+'\n')
    write_json(ROOT/'prefix_preparation_receipt.json',{'exit_code':0,'historical_code_sha256':before,'summary_sha256':sha256(p),'references_read':False,'model_loaded':False})


if __name__=='__main__':main()
