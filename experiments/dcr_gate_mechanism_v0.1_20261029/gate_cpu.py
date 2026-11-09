"""Frozen-score gate intervention; remote CPU only."""
import gzip
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
PREV = ROOT.parent / 'dcr_stronger_baselines_v0.1_20261028'
sys.path.insert(0, str(PREV))
import evaluation as ev

METHODS = ('CE_no_gate', 'CE_lexical_gate', 'CE_Qwen_gate', 'Qwen_direct')


def hard(order, gate):
    return [d for d in order if d not in gate] + [d for d in order if d in gate]


def verify():
    manifest = ev.read(ROOT / 'frozen_manifest.json')
    for name, expected in manifest['code'].items():
        assert ev.sha(ROOT / name) == expected, name
    for name, expected in manifest['sources'].items():
        assert ev.sha(Path(name)) == expected, name
    ev.verify()


def main():
    verify()
    started = time.monotonic()
    out = ROOT / 'robust_results'
    out.mkdir(exist_ok=False)
    queries, pools, _, relevance, ends = ev.inputs()
    lexical = {}
    with gzip.open(PREV / 'reuse_results/scores.jsonl.gz', 'rt') as stream:
        for line in stream:
            r = json.loads(line)
            key = r['qid'], r['side'], r['pid']
            assert key not in lexical
            assert r['ce_query_score'] == relevance[r['qid']][r['pid']]
            lexical[key] = r['violates']
    assert len(lexical) == 104000
    neural = {}
    with gzip.open(PREV / 'qwen_results/scores.jsonl.gz', 'rt') as stream:
        for line in stream:
            r = json.loads(line)
            assert r['doc_prefix_end_char'] == ends[r['qid'], r['pid']]
            assert all(math.isfinite(r[k]) for k in ('yes_logit', 'no_logit', 'score'))
            assert r['score'] == r['yes_logit'] - r['no_logit']
            key = r['qid'], r['role'], r['pid']
            assert key not in neural
            neural[key] = r
    assert len(neural) == 156000
    records, stats = [], []
    with gzip.open(out / 'gates.jsonl.gz', 'wt') as stream:
        for q in queries:
            qid = str(q['_id'])
            pids = pools[qid]
            order = sorted(pids, key=relevance[qid].get, reverse=True)
            sides = {}
            for side in ('og', 'changed'):
                lex = {d for d in pids if lexical[qid, side, d]}
                qw = {d: neural[qid, 'concat_' + side, d] for d in pids}
                ng = {d for d in pids if qw[d]['no_logit'] > qw[d]['yes_logit']}
                ties = sum(qw[d]['no_logit'] == qw[d]['yes_logit'] for d in pids)
                sides[side] = {'CE_no_gate': order, 'CE_lexical_gate': hard(order, lex),
                               'CE_Qwen_gate': hard(order, ng),
                               'Qwen_direct': sorted(pids, key=lambda d: qw[d]['score'], reverse=True)}
                stats.append({'qid': qid, 'side': side, 'lexical_demoted': len(lex),
                              'Qwen_demoted': len(ng), 'Qwen_tie_retained': ties,
                              'gate_disagreement': len(lex ^ ng)})
                for d in pids:
                    stream.write(json.dumps({'qid': qid, 'side': side, 'pid': d,
                        'lexical_gate': d in lex, 'Qwen_gate': d in ng,
                        'no_logit': qw[d]['no_logit'], 'yes_logit': qw[d]['yes_logit'],
                        'base_score': relevance[qid][d]}) + '\n')
            records += [{'qid': qid, 'method': m, 'original': sides['og'][m],
                         'changed': sides['changed'][m]} for m in METHODS]
    with gzip.open(out / 'rankings.jsonl.gz', 'wt') as stream:
        for record in records:
            stream.write(json.dumps(record) + '\n')
    result = ev.evaluate_rankings(records, out, ev.frozen_metric(),
        [('CE_Qwen_gate', 'CE_lexical_gate'), ('CE_Qwen_gate', 'CE_no_gate'),
         ('CE_Qwen_gate', 'Qwen_direct'), ('CE_lexical_gate', 'CE_no_gate')])
    ev.write_csv(out / 'gate_counts.csv', stats)
    result.update(elapsed_seconds=time.monotonic() - started, new_model_pairs=0,
                  gate_interpretation='conditional relevance, not independently labelled violation',
                  frozen_manifest_sha256=ev.sha(ROOT / 'frozen_manifest.json'))
    ev.write_json(out / 'summary.json', result)
    verify()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
