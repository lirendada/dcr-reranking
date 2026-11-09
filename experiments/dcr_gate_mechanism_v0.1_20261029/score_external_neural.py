"""Four fixed methods, complete real candidate scores, frozen ExcluIR metrics."""
import csv
import gzip
import json
from pathlib import Path
import random
import sys
import time
from adapter import sha256, write_json
from scorer import evaluate
from prepare_external_neural import ROOT

OLD = ROOT.parent / 'dot_experiment_bundle/research/dcr_public_baselines_v0.1_20261027'
sys.path.insert(0, str(OLD))
from frozen_parser import tokenize, standalone_parse_instruction

METHODS = ('CE_no_gate', 'CE_lexical_gate', 'CE_Qwen_gate', 'Qwen_direct')


def ordinal_gate_scores(relevance, gate):
    composite = {d: (d not in gate, value) for d, value in relevance.items()}
    levels = {value: i + 1 for i, value in enumerate(sorted(set(composite.values())))}
    return {d: float(levels[composite[d]]) for d in relevance}


def rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def main():
    started = time.monotonic()
    bundle = ROOT / 'external_inputs'
    candidates = ROOT / 'external_retrieval/candidates.csv'
    frozen = json.loads((ROOT / 'external_neural_frozen.json').read_text())
    for name, expected in frozen['code'].items():
        assert sha256(ROOT / name) == expected, name
    meta = json.loads((bundle / 'metadata.json').read_text())
    assert sha256(bundle / 'references.json') == meta['export_sha256']['references.json']
    queries = {k: {r['query_id']: r['text'] for r in map(json.loads, (bundle / f'queries_{k}.jsonl').open())}
               for k in ('question0', 'RQ_rewrite')}
    corpus = {r['document_id']: r['text'] for r in map(json.loads, (bundle / 'corpus.jsonl').open())}
    refs = json.loads((bundle / 'references.json').read_text())
    pools = {qid: {} for qid in refs}
    for r in csv.DictReader(candidates.open()):
        assert r['document_id'] not in pools[r['query_id']]
        pools[r['query_id']][r['document_id']] = None
    scores = {}
    for model in ('MiniLM', 'Qwen'):
        folder = ROOT / ('external_' + model)
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert manifest['pairs'] == 345200 and manifest['queries'] == 3452
        assert manifest['candidate_sha256'] == sha256(candidates)
        assert manifest['scores_sha256'] == sha256(folder / 'scores.jsonl.gz')
        score = {}
        for r in rows(folder / 'scores.jsonl.gz'):
            key = r['query_id'], r['document_id']
            assert key not in score
            score[key] = r
        assert set(score) == {(q, d) for q, docs in pools.items() for d in docs}
        scores[model] = score
    out = ROOT / 'external_neural_results'
    out.mkdir(exist_ok=False)
    outputs, counts = {m: {q: {} for q in pools} for m in METHODS}, []
    with gzip.open(out / 'gates.jsonl.gz', 'wt') as stream:
        for qid, docs in pools.items():
            r = {d: scores['MiniLM'][qid, d] for d in docs}
            qw = {d: scores['Qwen'][qid, d] for d in docs}
            assert all(r[d]['doc_prefix_end_char'] == qw[d]['doc_prefix_end_char'] for d in docs)
            base = sorted(docs, key=lambda d: (-r[d]['score'], d))
            negative = set(standalone_parse_instruction(queries['question0'][qid], queries['RQ_rewrite'][qid])['c_neg'])
            lex = {d for d in docs if negative & set(tokenize(corpus[d][:r[d]['doc_prefix_end_char']]))}
            neural = {d for d in docs if qw[d]['no_logit'] > qw[d]['yes_logit']}
            order = {'CE_no_gate': base,
                     'CE_lexical_gate': [d for d in base if d not in lex] + [d for d in base if d in lex],
                     'CE_Qwen_gate': [d for d in base if d not in neural] + [d for d in base if d in neural],
                     'Qwen_direct': sorted(docs, key=lambda d: (-qw[d]['score'], d))}
            counts.append({'query_id': qid, 'lexical_terms': json.dumps(sorted(negative)),
                           'lexical_demoted': len(lex), 'Qwen_demoted': len(neural),
                           'Qwen_tie_retained': sum(qw[d]['no_logit'] == qw[d]['yes_logit'] for d in docs),
                           'gate_disagreement': len(lex ^ neural)})
            for d in docs:
                stream.write(json.dumps({'query_id': qid, 'document_id': d,
                    'lexical_gate': d in lex, 'Qwen_gate': d in neural,
                    'base_score': r[d]['score'], 'Qwen_score': qw[d]['score']}) + '\n')
            outputs['CE_no_gate'][qid] = {d: r[d]['score'] for d in docs}
            outputs['Qwen_direct'][qid] = {d: qw[d]['score'] for d in docs}
            for method, gate in [('CE_lexical_gate', lex), ('CE_Qwen_gate', neural)]:
                # Ordinal of (retained group, base relevance) preserves true score ties.
                # Distinct document IDs only break ties when the scorer builds a ranking.
                outputs[method][qid] = ordinal_gate_scores({d: r[d]['score'] for d in docs}, gate)
            for method in METHODS:
                assert sorted(docs, key=lambda d: (-outputs[method][qid][d], d)) == order[method]
    summaries, all_records = {}, {}
    for method in METHODS:
        records, aggregate = evaluate(refs, outputs[method], [1, 5, 10, 100], pools)
        assert aggregate['evaluation_status'] == 'ranking_metrics_available'
        folder = out / method
        folder.mkdir()
        with (folder / 'run.csv').open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['query_id', 'document_id', 'score'])
            for qid, docs in outputs[method].items():
                writer.writerows((qid, d, value) for d, value in docs.items())
        with (folder / 'per_query.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        summaries[method] = aggregate
        all_records[method] = {r['query_id']: r for r in records}
        write_json(folder / 'summary.json', {'aggregate': aggregate, 'candidate_sha256': sha256(candidates),
            'stage': 'candidate_reranking', 'query_variant': 'RQ_rewrite',
            'score_representation': 'no_gate/direct use raw score; gates use ordinal of (retained_group,raw_CE_score), preserving actual score ties; canonical ID breaks rank ties',
            'run_sha256': sha256(folder / 'run.csv'), 'references_used_for_candidates': False,
            'engine': method, 'source_sha256': meta['source_sha256']})
    qids = sorted(refs)
    import numpy as np
    # Python RNG fixes identical draws; NumPy only accelerates sums, not random generation.
    rng = random.Random(20261027)
    draws = np.asarray([[rng.randrange(len(qids)) for _ in qids] for _ in range(5000)], dtype=np.int32)
    paired = []
    keys = ['positive_recall@1', 'positive_recall@10', 'negative_entry@1', 'negative_entry@10',
            'delta_recall@1', 'delta_recall@10', 'author_right_rank@10', 'pairwise_credit']
    for a, b in [('CE_Qwen_gate', 'CE_lexical_gate'), ('CE_Qwen_gate', 'CE_no_gate'),
                 ('CE_Qwen_gate', 'Qwen_direct'), ('CE_lexical_gate', 'CE_no_gate')]:
        for key in keys:
            valid = np.asarray([all_records[a][q][key] is not None and all_records[b][q][key] is not None for q in qids])
            diff = np.asarray([(all_records[a][q][key] or 0) - (all_records[b][q][key] or 0) for q in qids])
            totals = (diff[draws] * valid[draws]).sum(1)
            denominators = valid[draws].sum(1)
            assert (denominators > 0).all()
            samples = 100 * totals / denominators
            paired.append({'comparison': a + ' - ' + b, 'metric': key,
                'difference_pp': 100 * float(diff[valid].mean()), 'ci_low_pp': float(np.quantile(samples, .025)),
                'ci_high_pp': float(np.quantile(samples, .975)), 'queries_in_metric': int(valid.sum()),
                'planned_queries': len(qids), 'draws': 5000, 'seed': 20261027})
    for name, records in [('paired_bootstrap.csv', paired), ('gate_counts.csv', counts)]:
        with (out / name).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    write_json(out / 'summary.json', {'aggregate': summaries, 'paired_bootstrap': paired,
        'queries': 3452, 'candidate_pairs': 345200, 'candidate_sha256': sha256(candidates),
        'elapsed_scoring_seconds': time.monotonic() - started,
        'references_not_exhaustive': True, 'conditional_gate_not_violation_gold': True,
        'frozen_manifest_sha256': sha256(ROOT / 'external_neural_frozen.json')})
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
