"""Feature seal first; endpoint/group preparation isolated from fit processes."""
import random
from pathlib import Path
import numpy as np
from paths import ROOT,GATE,PREVIOUS,read,write,sha,verify_code


def main():
    m=verify_code();seal=read(ROOT/'features_sealed.json')
    for n,h in seal['files'].items():assert sha(ROOT/n)==h
    for p,h in m['reference_sources'].items():assert sha(Path(p))==h
    meta=read(ROOT/'head_and_tail.json');groups=read(PREVIOUS/'prepared/groups.json')
    assert meta['qids']==groups['qids']
    qids=meta['qids'];index={q:i for i,q in enumerate(qids)}
    fold=np.array([groups['query_fold'][q] for q in qids],dtype=np.uint8)
    refs=read(GATE/'external_inputs/references.json')
    positive=[];negative=[]
    for q,docs in zip(qids,meta['ranked_pools']):
        positive.append(docs[:20].index(refs[q]['positive_id']) if refs[q]['positive_id'] in docs[:20] else -1)
        negative.append(docs[:20].index(refs[q]['negative_id']) if refs[q]['negative_id'] in docs[:20] else -1)
    positive,negative=np.array(positive,dtype=np.int8),np.array(negative,dtype=np.int8)
    out=ROOT/'prepared';out.mkdir(exist_ok=False)
    rng=random.Random(20261029);weights=np.zeros((1000,len(qids)),dtype=np.int16)
    for d in range(1000):
        for f in range(5):
            cs=groups['fold_component_IDs'][str(f)]
            for _ in cs:
                c=cs[rng.randrange(len(cs))]
                for q in groups['components'][c]:weights[d,index[q]]+=1
    np.savez_compressed(out/'plan.npz',weights=weights,fold=fold)
    for f in range(5):
        train=np.flatnonzero(fold!=f);test=np.flatnonzero(fold==f)
        np.savez_compressed(out/f'labels_{f}.npz',train_indices=train,positive=positive[train],negative=negative[train])
        files=[ROOT/'features.npz',ROOT/'head_and_tail.json',ROOT/'features_sealed.json',out/'plan.npz',out/f'labels_{f}.npz']
        write(out/f'view_{f}.json',{'fold':f,'test_indices':test.tolist(),'allowed_inputs':{str(p):sha(p) for p in files},
              'labels_path':str(out/f'labels_{f}.npz'),'training_queries':len(train),
              'training_pair_queries':int(((positive[train]>=0)&(negative[train]>=0)).sum()),
              'training_positive_in_head':int((positive[train]>=0).sum()),'training_negative_in_head':int((negative[train]>=0).sum())})
    write(out/'receipt.json',{'groups_source_sha256':sha(PREVIOUS/'prepared/groups.json'),'queries':len(qids),
        'qualified_pair_queries':int(((positive>=0)&(negative>=0)).sum()),'fold_query_counts':np.bincount(fold,minlength=5).tolist(),
        'all_queries_evaluated_later':True,'unjudged_candidates_given_negative_labels':False})


if __name__=='__main__':main()
