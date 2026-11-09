"""Synthetic code/feature/cutoff fixtures. No real model or external qrels read."""
from datetime import datetime
from pathlib import Path
import ast,importlib.util,json,tempfile
import numpy as np
from config import ROOT,A,write
from feature_kernel import matched
from prepare import create_context
from night_gate import allowed


def main():
 old=np.zeros((1,20,12));old[0,:,1]=np.arange(20)/3;old[0,:,4:8]=np.arange(80).reshape(20,4);old[0,:,11]=old[0,:,1]/3
 x=matched(old,np.arange(20.)[None,:],np.arange(20)[None,:]);assert x[0,19,2]==.05 and np.array_equal(x[:,:,4:8],old[:,:,4:8])
 assert allowed(datetime.fromisoformat('2026-10-30T03:59:59+08:00')) and not allowed(datetime.fromisoformat('2026-10-30T04:00:00+08:00'))
 # Dataset context generation/receipt syntax; no inference function called.
 with tempfile.TemporaryDirectory(dir=ROOT,prefix='fixture_') as temp:
  out=Path(temp);create_context('nfcorpus',out,[{'query_id':'q','topic':'topic','request':'full'}])
  for name in ['config.py','prepare_inputs.py','infer_cell.py']:ast.parse((out/name).read_text())
  spec=importlib.util.spec_from_file_location('fixture_child_context',out/'config.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  p=out/'test.json';m.write(p,{'ok':True});assert json.loads(p.read_text())=={'ok':True}
  assert m.EXPECTED_NEW_PAIRS==1720 and m.QUERY_PATHS['full'].name=='queries_full.jsonl'
  copied=(out/'infer_cell.py').read_text();assert "old.load_model(CE_MODEL,'cuda',1)" in copied and 'assert_gpu_start_allowed()' in copied
 write(ROOT/'fixtures.json',{'status':'PASS','fixed_row_matched_features':True,'cutoff_exact04':True,'generated_context_AST_and_JSON':True,'no_model_loaded':True})


if __name__=='__main__':main()
