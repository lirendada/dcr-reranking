"""Only a single fold's existing training view; no references or heldout labels."""
import argparse,time
from pathlib import Path
import numpy as np
from config import ROOT,POST,read,write,sha,check_code,check_phase,math_core


def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True);f=p.parse_args().fold
 start=time.monotonic();check_code();check_phase('kernel');m=math_core()
 seal=read(ROOT/'features_sealed.json');assert sha(ROOT/'features.npz')==seal['features_sha256']
 view_path=POST/'prepared'/f'view_{f}.json';assert sha(view_path)==check_code()['view_manifests'][str(view_path)]
 view=read(view_path)
 for path,h in view['allowed_inputs'].items():assert sha(Path(path))==h,path
 with np.load(ROOT/'features.npz',allow_pickle=False) as a:x=a['X'];canonical=a['canonical']
 with np.load(POST/'prepared/plan.npz',allow_pickle=False) as a:plan=a['weights'];fold=a['fold']
 with np.load(view['labels_path'],allow_pickle=False) as a:train=a['train_indices'];pp=a['positive'];nn=a['negative']
 pos=np.full(len(x),-2,dtype=np.int8);neg=pos.copy();pos[train]=pp;neg[train]=nn
 test=np.asarray(view['test_indices']);assert np.array_equal(test,np.flatnonzero(fold==f))
 out={'test_indices':test,'lambda':np.empty(1001),'inner_correct_counts':np.empty((1001,4),dtype=np.int32),
  'scale':np.empty((1001,12)),'coef':np.empty((1001,12)),'pair_instances':np.empty(1001,dtype=np.int32),
  'order':np.empty((1001,len(test),20),dtype=np.uint8),'point_scores':np.empty((len(test),20))}
 for iteration in range(1001):
  weights=np.ones(len(x),dtype=np.int16) if iteration==0 else plan[iteration-1]
  lam,counts=m.select(x,pos,neg,canonical,fold,f,weights);model=m.solve(m.statistics(x,pos,neg,weights*(fold!=f)),lam)
  values=m.predict(x[test],model)
  for key,value in [('lambda',lam),('inner_correct_counts',counts),('scale',model[0]),('coef',model[1]),('pair_instances',model[2]),('order',m.rank(values,canonical[test]))]:out[key][iteration]=value
  if iteration==0:out['point_scores'][:]=values
  if iteration%200==0:print(f'fold={f} iteration={iteration}/1000 wall={time.monotonic()-start:.3f}',flush=True)
 folder=ROOT/'fitted';folder.mkdir(exist_ok=True);path=folder/f'fold_{f}.npz';assert not path.exists()
 np.savez_compressed(path,**out)
 write(path.with_suffix('.json'),{'fold':f,'sha256':sha(path),'wall_seconds':time.monotonic()-start,'heldout_labels_read':False,'training_label_sha256':sha(Path(view['labels_path']))})


if __name__=='__main__':main()
