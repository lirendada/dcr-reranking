"""One heldout fold process: no references, curves, oracle allocations or heldout labels."""
import argparse
from pathlib import Path
import time
import numpy as np
from paths import ROOT,read,write,sha,verify_code
from post_math import select_alpha,select_lambda,ridge_fit,ridge_scores,rank_positions


def main():
    a=argparse.ArgumentParser();a.add_argument('--fold',type=int,choices=range(5),required=True);args=a.parse_args()
    start=time.monotonic();verify_code();v=read(ROOT/'prepared'/f'view_{args.fold}.json')
    for p,h in v['allowed_inputs'].items():assert sha(Path(p))==h
    with np.load(ROOT/'features.npz',allow_pickle=False) as z:x=z['X'];canonical=z['canonical']
    with np.load(ROOT/'prepared/plan.npz',allow_pickle=False) as z:plan=z['weights'];fold=z['fold']
    with np.load(v['labels_path'],allow_pickle=False) as z:train=z['train_indices'];pos=z['positive'];neg=z['negative']
    positive=np.full(len(x),-2,dtype=np.int8);negative=positive.copy();positive[train]=pos;negative[train]=neg
    test=np.array(v['test_indices'],dtype=np.int32)
    results={'test_indices':test,'alpha':np.empty(1001),'linear_order':np.empty((1001,len(test),20),dtype=np.uint8)}
    for name,p in [('G',12),('GC',18)]:
        results[name+'_lambda']=np.empty(1001);results[name+'_inner_correct_counts']=np.empty((1001,4),dtype=np.int32)
        results[name+'_scale']=np.empty((1001,p));results[name+'_coef']=np.empty((1001,p))
        results[name+'_pair_instances']=np.empty(1001,dtype=np.int32)
        results[name+'_order']=np.empty((1001,len(test),20),dtype=np.uint8)
        results[name+'_point_scores']=np.empty((len(test),20))
    for iteration in range(1001):
        weights=np.ones(len(x),dtype=np.int16) if iteration==0 else plan[iteration-1]
        training=weights*(fold!=args.fold)
        alpha,count=select_alpha(x,positive,canonical,training)
        results['alpha'][iteration]=alpha
        scores=alpha*x[test,:,0]+(1-alpha)*x[test,:,1]
        results['linear_order'][iteration]=rank_positions(scores,canonical[test])
        for name,p in [('G',12),('GC',18)]:
            lam,counts=select_lambda(x,positive,negative,canonical,fold,args.fold,weights,p)
            model=ridge_fit(x,positive,negative,training,lam,p)
            scores=ridge_scores(x[test],model)
            results[name+'_lambda'][iteration]=lam;results[name+'_inner_correct_counts'][iteration]=counts
            results[name+'_scale'][iteration]=model[0];results[name+'_coef'][iteration]=model[1]
            results[name+'_pair_instances'][iteration]=model[2]
            results[name+'_order'][iteration]=rank_positions(scores,canonical[test])
            if iteration==0:results[name+'_point_scores'][:]=scores
        if iteration%100==0:print(f'fold={args.fold} iteration={iteration}/1000 wall={time.monotonic()-start:.3f}',flush=True)
    out=ROOT/'fitted';out.mkdir(exist_ok=True)
    path=out/f'fold_{args.fold}.npz';assert not path.exists();np.savez_compressed(path,**results)
    write(out/f'fold_{args.fold}.json',{'fold':args.fold,'training_labels_file_sha256':sha(Path(v['labels_path'])),
        'heldout_labels_read':False,'file_sha256':sha(path),'CPU_task_wall_seconds':time.monotonic()-start})


if __name__=='__main__':main()
