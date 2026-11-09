"""Post-seal descriptive counts only; no fitting, selection or revised evaluation."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
source=ROOT/'per_query.csv.gz'
with gzip.open(source,'rt') as f:rows=list(csv.DictReader(f))
groups=json.loads((ROOT/'auxiliary_groups.json').read_text())['groups']
by={}
for r in rows:by.setdefault(r['method'],{})[r['qid']]=r
states=['P','N','U']
def state(row):
    p,n=int(row['positive_recall@1']),int(row['negative_entry@1'])
    assert p+n<=1
    return 'P' if p else 'N' if n else 'U'
base=by['fixed_K20'];out={}
for name,queries in by.items():
    assert set(queries)==set(base)
    matrix=[[0]*3 for _ in states];counts={s:0 for s in states}
    for q,r in queries.items():
        a,b=state(base[q]),state(r);matrix[states.index(a)][states.index(b)]+=1;counts[b]+=1
    group_counts={}
    for g,qs in groups.items():
        old=sum(state(base[q])=='P' for q in qs);new=sum(state(queries[q])=='P' for q in qs)
        group_counts[g]={'queries':len(qs),'fixed_correct':old,'method_correct':new,'net_correct':new-old}
    out[name]={'top1_states':counts,'transition_from_fixed_rows_to_method_columns':matrix,'group_correct':group_counts}
result={'analysis_type':'post-seal descriptive only','states':states,'U_definition':'top1 is neither designated positive nor designated negative; unjudged, not established irrelevant',
 'source_per_query_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'group_source_sha256':hashlib.sha256((ROOT/'auxiliary_groups.json').read_bytes()).hexdigest(),'methods':out}
with (ROOT/'descriptive_transitions.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(out,indent=2))
