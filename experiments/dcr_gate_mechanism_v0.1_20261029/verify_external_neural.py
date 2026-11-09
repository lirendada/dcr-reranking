"""Independent source-to-order and endpoint-metric audit after complete external scoring."""
import csv
import gzip
import json
import math
from pathlib import Path
import random
import sys
from adapter import sha256, write_json
from prepare_external_neural import ROOT


def read_scores(name):
    folder = ROOT / ('external_' + name)
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert sha256(folder / 'scores.jsonl.gz') == manifest['scores_sha256']
    result = {}
    with gzip.open(folder / 'scores.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            key = row['query_id'], row['document_id']
            assert key not in result and math.isfinite(row['score'])
            if name == 'Qwen':
                assert row['score'] == row['yes_logit'] - row['no_logit']
                assert row['demote'] == (row['no_logit'] > row['yes_logit'])
            result[key] = row
    assert len(result) == manifest['pairs'] == 345200
    return result


def main():
    out = ROOT / 'external_neural_results'
    frozen = json.loads((ROOT / 'external_neural_frozen.json').read_text())
    for name, expected in frozen['code'].items():
        assert sha256(ROOT / name) == expected
    for name, expected in frozen['sources'].items():
        assert sha256(Path(name)) == expected
    ce, qw = read_scores('MiniLM'), read_scores('Qwen')
    meta = json.loads((ROOT / 'external_inputs/metadata.json').read_text())
    refs_file = ROOT / 'external_inputs/references.json'
    assert sha256(refs_file) == meta['export_sha256']['references.json']
    refs = json.loads(refs_file.read_text())
    pools = {}
    for row in csv.DictReader((ROOT / 'external_retrieval/candidates.csv').open()):
        pools.setdefault(row['query_id'], []).append(row['document_id'])
    expected_pairs = {(q, d) for q, docs in pools.items() for d in docs}
    assert set(ce) == set(qw) == expected_pairs
    with gzip.open(ROOT / 'external_prepared/pairs.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            key = row['query_id'], row['document_id']
            for score, length in [(ce[key], 'ce_input_tokens'), (qw[key], 'qwen_input_tokens')]:
                assert row['doc_prefix_end_char'] == score['doc_prefix_end_char']
                assert row[length] == score['input_tokens']
    corpus = {r['document_id']: r['text'] for r in map(json.loads, (ROOT / 'external_inputs/corpus.jsonl').open())}
    queries = {k: {r['query_id']: r['text'] for r in map(json.loads, (ROOT / 'external_inputs' / f'queries_{k}.jsonl').open())}
               for k in ('question0', 'RQ_rewrite')}
    sys.path.insert(0, str(ROOT.parent / 'dot_experiment_bundle/research/dcr_public_baselines_v0.1_20261027'))
    from frozen_parser import tokenize, standalone_parse_instruction
    methods = ('CE_no_gate', 'CE_lexical_gate', 'CE_Qwen_gate', 'Qwen_direct')
    checked, all_rows = 0, {}
    for method in methods:
        imported = {}
        for row in csv.DictReader((out / method / 'run.csv').open()):
            imported.setdefault(row['query_id'], []).append((row['document_id'], float(row['score'])))
        rows = {r['query_id']: r for r in csv.DictReader((out / method / 'per_query.csv').open())}
        all_rows[method] = rows
        for qid, docs in pools.items():
            lexical = set(standalone_parse_instruction(queries['question0'][qid], queries['RQ_rewrite'][qid])['c_neg'])
            def key(did):
                pair = qid, did
                if method == 'Qwen_direct':
                    return -qw[pair]['score'], did
                if method == 'CE_lexical_gate':
                    gate = bool(lexical & set(tokenize(corpus[did][:ce[pair]['doc_prefix_end_char']])))
                else:
                    gate = method == 'CE_Qwen_gate' and qw[pair]['no_logit'] > qw[pair]['yes_logit']
                return gate, -ce[pair]['score'], did
            expected = sorted(docs, key=key)
            actual = [d for d, _ in sorted(imported[qid], key=lambda pair: (-pair[1], pair[0]))]
            assert actual == expected and len(actual) == 100
            ranks = {d: i + 1 for i, d in enumerate(expected)}
            p, n = ranks.get(refs[qid]['positive_id']), ranks.get(refs[qid]['negative_id'])
            row = rows[qid]
            assert row['positive_rank'] == (str(p) if p else '') and row['negative_rank'] == (str(n) if n else '')
            for k in (1, 5, 10, 100):
                for name, rank in [('positive_recall', p), ('negative_entry', n)]:
                    assert float(row[f'{name}@{k}']) == float(rank is not None and rank <= k)
            assert float(row['author_right_rank@10']) == float(p is not None and p <= 10 and (n is None or n > 10 or p < n))
            pmrr = 1 / p if p is not None and p <= 10 else 0
            nmrr = 1 / n if n is not None and n <= 10 else 0
            assert abs(float(row['positive_mrr@10']) - pmrr) < 1e-12
            assert abs(float(row['negative_mrr@10']) - nmrr) < 1e-12
            assert abs(float(row['delta_mrr@10']) - (pmrr - nmrr)) < 1e-12
            if p and n:
                values = dict(imported[qid])
                pv, nv = values[refs[qid]['positive_id']], values[refs[qid]['negative_id']]
                assert float(row['pairwise_credit']) == (1.0 if pv > nv else 0.5 if pv == nv else 0.0)
            else:
                assert row['pairwise_credit'] == ''
            checked += 1
        summary = json.loads((out / method / 'summary.json').read_text())['aggregate']
        assert summary['planned_queries'] == len(rows) == 3452
        for key, observed in summary['ranking_metrics_all_planned_queries'].items():
            assert abs(sum(float(row[key]) for row in rows.values()) / len(rows) - observed) < 1e-12
        pair_values = [float(row['pairwise_credit']) for row in rows.values() if row['pairwise_credit'] != '']
        assert len(pair_values) == summary['both_scored_queries']
        assert abs(sum(pair_values) / len(pair_values) - summary['pairwise_credit_given_both_scored']) < 1e-12
    # Different resampling arithmetic: count matrix rather than production's indexed arrays.
    import numpy as np
    qids = sorted(refs)
    rng = random.Random(20261027)
    weights = np.zeros((5000, len(qids)), dtype=np.uint16)
    for i in range(5000):
        draw = [rng.randrange(len(qids)) for _ in qids]
        weights[i] = np.bincount(draw, minlength=len(qids))
    paired_rows = list(csv.DictReader((out / 'paired_bootstrap.csv').open()))
    assert len(paired_rows) == 32
    for row in paired_rows:
        a, b = row['comparison'].split(' - ')
        key = row['metric']
        valid = np.asarray([all_rows[a][q][key] != '' and all_rows[b][q][key] != '' for q in qids])
        diff = np.asarray([float(all_rows[a][q][key] or 0) - float(all_rows[b][q][key] or 0) for q in qids])
        samples = 100 * (weights @ (diff * valid)) / (weights @ valid.astype(np.float64))
        expected = {'difference_pp': 100 * float(diff[valid].mean()),
                    'ci_low_pp': float(np.quantile(samples, .025)),
                    'ci_high_pp': float(np.quantile(samples, .975))}
        for key, value in expected.items():
            assert abs(float(row[key]) - value) < 1e-10
    write_json(out / 'independent_verification.json', {'status': 'passed', 'candidate_pairs': 345200,
        'matched_visible_prefix_pairs': 345200, 'raw_score_count': len(ce) + len(qw),
        'all_method_query_rankings_and_endpoint_metrics_recomputed': checked,
        'source_hashes_verified': True, 'conditional_pair_coverage_not_imputed': True,
        'all_aggregate_ranking_metrics_recomputed': True, 'independent_bootstrap_contrasts': len(paired_rows)})
    print('Independent neural source/order/endpoint audit passed.', flush=True)


if __name__ == '__main__':
    main()
