"""Before fitting: compare complete alpha0 and fixedK20 rankings against frozen old artifacts."""
import gzip
import json
import numpy as np
from paths import ROOT,BUDGET,read,write,sha,verify_code
from post_math import rank_positions


def main():
    manifest=verify_code();meta=read(ROOT/'head_and_tail.json');qids=meta['qids'];index={q:i for i,q in enumerate(qids)}
    with np.load(ROOT/'features.npz',allow_pickle=False) as a:raw=a['raw_scores'];x=a['X'];canonical=a['canonical']
    fixed=rank_positions(raw[:,:,1],canonical);endpoint=rank_positions(x[:,:,1],canonical)
    assert np.array_equal(fixed,endpoint),'alpha0 z-normalization changed order: do not silently change ties'
    source=BUDGET/'ExcluIR/rankings.jsonl.gz';seen=set()
    assert sha(source)==manifest['baseline_sources'][str(source)]
    with gzip.open(source,'rt') as f:
        for line in f:
            r=json.loads(line)
            if r['method']!='topK_20':continue
            q=r['qid'];i=index[q];assert q not in seen;seen.add(q)
            base=meta['ranked_pools'][i]
            expected=[base[j] for j in fixed[i]]+base[20:]
            assert r['orders']['changed']==expected,q
    assert seen==set(qids)
    write(ROOT/'fixed_source_check.json',{'queries':len(seen),'all_complete_head_and_tail_identical':True,
       'alpha0_identical_to_raw_Qwen_order':True,'old_ranking_source_sha256':sha(source),'references_opened':False})


if __name__=='__main__':main()
