"""Single full-source fit: no IFIR input, qrels or evaluation files are read."""
import time
import numpy as np
from paths import ROOT,PREV,GATE,read,write,sha,load_source_math,verify_phase


def main():
    started=time.monotonic();verify_phase('training')
    m=load_source_math();meta=read(PREV/'head_and_tail.json');refs=read(GATE/'external_inputs/references.json')
    with np.load(PREV/'features.npz',allow_pickle=False) as f:x=f['X'][:,:,:12]
    assert meta['qids']==sorted(meta['qids']) and len(x)==3452
    pos=[];neg=[]
    for q,pool in zip(meta['qids'],meta['ranked_pools']):
        head=pool[:20];p,n=refs[q]['positive_id'],refs[q]['negative_id']
        pos.append(head.index(p) if p in head else -1);neg.append(head.index(n) if n in head else -1)
    scale,coef,count=m.ridge_fit(x,np.array(pos),np.array(neg),np.ones(len(x),dtype=np.int16),.1,12)
    assert count==2901
    write(ROOT/'source_model.json',{'lambda':.1,'dimensions':12,'intercept':False,'scale':scale.tolist(),'coef':coef.tolist(),
      'qualified_pairs':count,'source_queries':len(x),'training_qid_order':'ascending','feature_columns':m.G_COLUMNS,
      'IFIR_inputs_or_labels_opened':False,'CPU_wall_seconds':time.monotonic()-started})
    verify_phase('training')
    write(ROOT/'model_sealed.json',{'model_sha256':sha(ROOT/'source_model.json'),'frozen_manifest_sha256':sha(ROOT/'frozen_manifest.json')})


if __name__=='__main__':main()
