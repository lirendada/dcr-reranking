"""Independent CPU audit; run only on server after production has completed.

No production feature, learner, ranking or metric module is imported. The
unchanged frozen lexical parser is a declared shared protocol dependency.
Model inference is outside this auditor's scope; saved scores are checked for
coverage, selection, prefixes, cost accounting, and sealed provenance.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import time

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent
PREV = RUNS / 'dcr_post_fusion_v0.1_20261029'
GATE = RUNS / 'dcr_gate_mechanism_v0.1_20261029'
PARSER = RUNS / 'dot_experiment_bundle/research/dcr_public_baselines_v0.1_20261027/frozen_parser.py'
DATASETS = ('fiqa', 'nfcorpus')
METHODS = ('MiniLM', 'fixed_Qwen20', 'Post_G')
METRICS = ('P1', 'nDCG20', 'MRR20', 'R10')
G_COLUMNS = ['z_M', 'z_Q', 'rank_M_over20', 'rank_Q_over20', 'topic_cov', 'request_cov',
             'topic_request_jaccard', 'log1p_body_tokens', 'zM_gapM', 'zM_gapQ', 'zQ_gapM', 'zQ_gapQ']
RAW_HASHES = {
    'fiqa/fiqa-query.jsonl': '7b5cb08704bccc436e5fc60e398ca7f34f52bd2c7900096c04bda7c0026b9b91',
    'fiqa/fiqa-qrel.tsv': '026ead0aefa118ff014dc7f6c78e1945ce6c940259e15fae065ce80ee4d657d4',
    'fiqa/fiqa-corpus.jsonl': 'ff593e4df9933955dc3af83be0c3fa28ac7465f627e08c2e53593e734d506517',
    'nfcorpus/nfcorpus-query.jsonl': 'bffc93cb48a3e8009e4e275c1b48a142928ca4206f0eded5aa7ef7ed32e549c5',
    'nfcorpus/nfcorpus-qrel.tsv': 'b63ab04eb4b960b9ee32e22d51c4547f4f43802ee157087dfc0db2deb459661b',
    'nfcorpus/nfcorpus-corpus.jsonl': '10cc83ef1826b1425e6a87090b5140b39b27755d5a27e48215a88611c899991f'}
NUMERIC = {'scalar_comparisons': 0, 'maximum_absolute_difference': 0.}
CHECKS = {'BM25_base_pools': 0, 'exact_prefix_pairs': 0, 'exact_token_lengths': 0}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def csv_rows(path, delimiter=','):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', newline='') as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def close(actual, expected, label, tolerance=1e-10):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), label
    error = float(np.max(np.abs(a-b))) if a.size else 0.
    NUMERIC['scalar_comparisons'] += a.size
    NUMERIC['maximum_absolute_difference'] = max(NUMERIC['maximum_absolute_difference'], error)
    assert error <= tolerance, (label, error, tolerance)


def source_fit():
    meta = read(PREV / 'head_and_tail.json')
    qids, pools = meta['qids'], meta['ranked_pools']
    with np.load(PREV / 'features.npz', allow_pickle=False) as archive:
        x = archive['X'][:, :, :12]
    assert len(qids) == len(pools) == len(x) == 3452
    assert qids == sorted(qids) and len(set(qids)) == 3452
    refs = read(GATE / 'external_inputs/references.json')
    pairs = []
    for i, (q, pool) in enumerate(zip(qids, pools)):
        assert len(pool) == len(set(pool)) == 100
        p, n = refs[q]['positive_id'], refs[q]['negative_id']
        if p in pool[:20] and n in pool[:20]:
            pairs.append(x[i, pool.index(p)] - x[i, pool.index(n)])
    d = np.array(pairs)
    assert d.shape == (2901, 12)
    scale = np.sqrt(np.mean(d*d, axis=0))
    scale[scale == 0] = 1.
    z = d / scale
    coefficient = np.linalg.solve(z.T @ z + len(d)*.1*np.eye(12), z.sum(axis=0))
    model = read(ROOT / 'source_model.json')
    assert model['lambda'] == .1 and model['dimensions'] == 12
    assert model['intercept'] is False and model['qualified_pairs'] == 2901
    assert model['source_queries'] == 3452 and model['training_qid_order'] == 'ascending'
    assert model['IFIR_inputs_or_labels_opened'] is False
    assert model['feature_columns'] == G_COLUMNS
    close(model['scale'], scale, 'source RMS')
    close(model['coef'], coefficient, 'source coefficient')
    return scale, coefficient


def standardize(values):
    sigma = values.std(ddof=0)
    return (values-values.mean())/sigma if sigma else np.zeros_like(values)


def features(topic, request, head, mini, qwen, bodies, parser):
    mz, qz = standardize(mini), standardize(qwen)
    mgap, qgap = np.sort(mz)[-1]-np.sort(mz)[-2], np.sort(qz)[-1]-np.sort(qz)[-2]
    qranks = np.empty(20)
    qranks[sorted(range(20), key=lambda i: (-qwen[i], head[i]))] = np.arange(1, 21)/20
    topic, request = set(parser.tokenize(topic, remove_stop=True)), set(parser.tokenize(request, remove_stop=True))
    combined = topic | request
    result = []
    for i, body in enumerate(bodies):
        toks = parser.tokenize(body, remove_stop=True)
        terms = set(toks)
        result.append([mz[i], qz[i], (i+1)/20, qranks[i],
            len(topic & terms)/len(topic) if topic else 0.,
            len(request & terms)/len(request) if request else 0.,
            len(combined & terms)/len(combined | terms) if combined | terms else 0.,
            np.log1p(len(toks)), mz[i]*mgap, mz[i]*qgap, qz[i]*mgap, qz[i]*qgap])
    return np.array(result)


def metric(order, relevance):
    positive = {d for d, value in relevance.items() if value > 0}
    reciprocal = next((1/rank for rank, d in enumerate(order[:20], 1) if d in positive), 0.)
    dcg = sum(relevance.get(d, 0)/math.log2(rank+1) for rank, d in enumerate(order[:20], 1))
    ideal = sum(value/math.log2(rank+1) for rank, value in enumerate(sorted(relevance.values(), reverse=True)[:20], 1))
    return np.array([float(order[0] in positive), dcg/ideal if ideal else 0., reciprocal,
                     len(positive & set(order[:10]))/len(positive) if positive else 0.])


def bm25(corpus, queries, pools, parser):
    """Independent full-corpus BM25; only the approved tokenizer is shared."""
    ids = list(corpus)
    postings = defaultdict(list)
    lengths = []
    for i, text in enumerate(corpus.values()):
        counts = Counter(parser.tokenize(text))
        lengths.append(sum(counts.values()))
        for term, count in counts.items():
            postings[term].append((i, count))
    length = np.array(lengths, dtype=float)
    normalizer = 1.5*(.25+.75*length/length.mean())
    for term, pairs in list(postings.items()):
        indices, counts = np.array(pairs).T
        idf = math.log(1+(len(ids)-len(indices)+.5)/(len(indices)+.5))
        postings[term] = indices, idf*counts*2.5/(counts+normalizer[indices])
    observed = {}
    for query in queries:
        base = query['base_id']
        if base not in observed:
            score = np.zeros(len(ids))
            for term, count in Counter(parser.tokenize(query['topic'])).items():
                if term in postings:
                    indices, values = postings[term]
                    score[indices] += count*values
            # Python's sorted explicitly preserves corpus-row ties.
            selected = sorted(range(len(ids)), key=lambda i: (-score[i], i))[:100]
            observed[base] = [ids[i] for i in selected]
            CHECKS['BM25_base_pools'] += 1
        assert pools[query['query_id']] == observed[base], ('BM25', query['query_id'])


def prefix_checks(queries, corpus, scores, tokenizers):
    ce, qw = tokenizers
    cache = {}
    query = {r['query_id']: r for r in queries}
    prompt_prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    prompt_suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    batch = []
    def flush():
        if not batch:
            return
        texts = [query[q]['topic'] for q, d in batch]
        bodies = [corpus[d][:scores['MiniLM'][q][d]['doc_prefix_end_char']] for q, d in batch]
        lengths = ce(texts, bodies, truncation=False)['input_ids']
        for (q, d), encoded in zip(batch, lengths):
            assert len(encoded) == scores['MiniLM'][q][d]['input_tokens'] <= 512
            CHECKS['exact_token_lengths'] += 1
        batch.clear()
    for q, records in scores['MiniLM'].items():
        maximum = max(len(ce(query[q][k], add_special_tokens=False)['input_ids']) for k in ('topic', 'request'))
        budget = 512-3-maximum
        assert budget > 0
        for d, r in records.items():
            if d not in cache:
                cache[d] = ce(corpus[d], add_special_tokens=False, return_offsets_mapping=True)
            encoded = cache[d]
            end = len(corpus[d]) if len(encoded['input_ids']) <= budget else encoded['offset_mapping'][budget-1][1]
            assert r['doc_prefix_end_char'] == end and r['shared_document_budget'] == budget
            CHECKS['exact_prefix_pairs'] += 1
            batch.append((q, d))
            if len(batch) == 128:
                flush()
    flush()
    for q, records in scores['Qwen'].items():
        docs = list(records)
        texts = [prompt_prefix+'<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: '+query[q]['request']+'\n<Document>: '+corpus[d][:records[d]['doc_prefix_end_char']]+prompt_suffix for d in docs]
        encoded = qw(texts, add_special_tokens=False, truncation=False)['input_ids']
        for d, tokens in zip(docs, encoded):
            assert len(tokens) == records[d]['input_tokens'] <= 2048
            CHECKS['exact_token_lengths'] += 1


def dataset(ds, scale, coefficient, parser, tokenizers):
    raw_queries = list(rows(ROOT / 'raw' / ds / f'{ds}-query.jsonl'))
    topics = {r['_id'].rsplit('_', 1)[0]: r['text'] for r in raw_queries if r['level'] == 0}
    expected = []
    for r in raw_queries:
        base, level = r['_id'].rsplit('_', 1)
        assert int(level) == r['level']
        if r['level'] > 0:
            expected.append({'query_id': r['_id'], 'base_id': base, 'level': r['level'],
                             'topic': topics[base], 'request': r['text']})
    expected.sort(key=lambda r: r['query_id'])
    queries = list(rows(ROOT / ds / 'inputs/queries.jsonl'))
    assert queries == expected and len(queries) == (1718 if ds == 'fiqa' else 86)
    qids = [r['query_id'] for r in queries]
    corpus = {r['_id']: r['text'] for r in rows(ROOT / 'raw' / ds / f'{ds}-corpus.jsonl')}
    adapted = list(rows(ROOT / ds / 'inputs/corpus.jsonl'))
    assert adapted == [{'document_id': k, 'text': v} for k, v in corpus.items()]
    pools = {q: [] for q in qids}
    for r in csv_rows(ROOT / ds / 'retrieval/candidates.csv'):
        assert int(r['BM25_rank']) == len(pools[r['query_id']])+1
        pools[r['query_id']].append(r['document_id'])
    assert all(len(p) == len(set(p)) == 100 and set(p) <= corpus.keys() for p in pools.values())
    bm25(corpus, queries, pools, parser)
    score = {}
    costs = {}
    for name, count in [('MiniLM', 100), ('Qwen', 20)]:
        score[name] = {q: {} for q in qids}
        for r in rows(ROOT / ds / name / 'scores.jsonl.gz'):
            q, d = r['query_id'], r['document_id']
            assert d not in score[name][q] and math.isfinite(r['score'])
            assert 0 <= r['doc_prefix_end_char'] <= len(corpus[d])
            score[name][q][d] = r
            if name == 'Qwen':
                close(r['score'], r['yes_logit']-r['no_logit'], 'yes-no score')
        assert all(len(p) == count for p in score[name].values())
        prep = {(r['query_id'], r['document_id']): r for r in rows(ROOT / ds / 'prepared' / f'{name}_pairs.jsonl.gz')}
        assert len(prep) == count*len(qids)
        for q, records in score[name].items():
            for d, r in records.items():
                for key, value in prep[q, d].items():
                    assert r[key] == value, (name, q, d, key)
        manifest = read(ROOT / ds / name / 'manifest.json')
        tokens = sum(r['input_tokens'] for records in score[name].values() for r in records.values())
        assert manifest['pairs'] == count*len(qids) and manifest['sum_input_tokens'] == tokens
        assert manifest['scores_sha256'] == sha(ROOT / ds / name / 'scores.jsonl.gz')
        costs[name] = {'pairs': count*len(qids), 'input_tokens': tokens}
    prefix_checks(queries, corpus, score, tokenizers)
    actual_orders = {m: {} for m in METHODS}
    for r in rows(ROOT / ds / 'outputs/rankings.jsonl.gz'):
        assert r['query_id'] not in actual_orders[r['method']]
        actual_orders[r['method']][r['query_id']] = r
    assert all(set(actual_orders[m]) == set(qids) for m in METHODS)
    metadata = read(ROOT / ds / 'outputs/feature_metadata.json')
    assert metadata['qids'] == qids and len(metadata['heads']) == len(qids)
    assert metadata['columns'] == G_COLUMNS and metadata['C_features_used'] is False
    assert metadata['reference_inputs_opened'] is False
    with np.load(ROOT / ds / 'outputs/features.npz', allow_pickle=False) as archive:
        saved_x = archive['X']
    # Feature and prediction reconstruction occurs before opening IFIR qrels.
    predicted = {m: {} for m in METHODS}
    for i, query in enumerate(queries):
        q = query['query_id']; mini = score['MiniLM'][q]; qw = score['Qwen'][q]
        assert set(mini) == set(pools[q])
        base = sorted(mini, key=lambda d: (-mini[d]['score'], d)); head = base[:20]
        assert set(qw) == set(head) and metadata['heads'][i] == head
        assert all(qw[d]['doc_prefix_end_char'] == mini[d]['doc_prefix_end_char'] for d in head)
        x = features(query['topic'], query['request'], head,
            np.array([mini[d]['score'] for d in head]), np.array([qw[d]['score'] for d in head]),
            [corpus[d][:mini[d]['doc_prefix_end_char']] for d in head], parser)
        close(saved_x[i], x, 'external G12')
        post = (x/scale) @ coefficient
        fixed = sorted(head, key=lambda d: (-qw[d]['score'], d))
        post_order = [head[j] for j in sorted(range(20), key=lambda j: (-post[j], head[j]))]
        expected_orders = {'MiniLM': base, 'fixed_Qwen20': fixed+base[20:], 'Post_G': post_order+base[20:]}
        for name, order in expected_orders.items():
            r = actual_orders[name][q]
            assert r['order'] == order and r['base_id'] == query['base_id'] and r['level'] == query['level']
            predicted[name][q] = order
    qrels = {q: {} for q in qids}
    for r in csv_rows(ROOT / 'raw' / ds / f'{ds}-qrel.tsv', '\t'):
        if r['query_id'] in qrels:
            assert r['corpus_id'] not in qrels[r['query_id']]
            qrels[r['query_id']][r['corpus_id']] = float(r['score'])
    values = {m: np.array([metric(predicted[m][q], qrels[q]) for q in qids]) for m in METHODS}
    reports = {m: {k: 100*float(values[m][:, j].mean()) for j, k in enumerate(METRICS)} for m in METHODS}
    bases = sorted({r['base_id'] for r in queries})
    base_indices = np.array([bases.index(r['base_id']) for r in queries])
    return {'queries': queries, 'values': values, 'reports': reports, 'costs': costs,
            'base_indices': base_indices, 'bases': bases, 'orders': predicted, 'qrels': qrels}


def check_auxiliary(ds, result, summary):
    queries = result['queries']; values = result['values']
    point = summary['datasets'][ds]
    assert point['queries'] == len(queries) and point['bases'] == len(result['bases'])
    delta = values['Post_G'][:, 0]-values['fixed_Qwen20'][:, 0]
    assert point['rescued'] == int((delta == 1).sum())
    assert point['new_losses'] == int((delta == -1).sum())
    assert point['net_correct'] == int(delta.sum())
    for level in sorted({r['level'] for r in queries}):
        selected = [i for i, r in enumerate(queries) if r['level'] == level]
        saved = point['level_reports'][str(level)]
        assert saved['queries'] == len(selected)
        for method in METHODS:
            close([saved['reports'][method][k] for k in METRICS],
                  100*values[method][selected].mean(axis=0), 'level metrics')
    recorded = csv_rows(ROOT / ds / 'coverage_per_query.csv')
    assert len(recorded) == 2*len(queries)
    assert {(r['query_id'], r['pool']) for r in recorded} == {
        (q['query_id'], pool) for q in queries for pool in ('BM25_top100', 'MiniLM_top20')}
    counts = {pool: [] for pool in ('BM25_top100', 'MiniLM_top20')}
    for r in recorded:
        q, pool = r['query_id'], r['pool']; k = 100 if pool == 'BM25_top100' else 20
        positive = {d for d, rel in result['qrels'][q].items() if rel > 0}
        hits = len(positive & set(result['orders']['MiniLM'][q][:k]))
        expected = [hits/len(positive) if positive else 0., float(hits > 0), hits, len(positive)]
        close([float(r[n]) for n in ('positive_document_recall', 'at_least_one_positive',
              'positive_documents_retrieved', 'known_positive_documents')], expected, 'query coverage')
        counts[pool].append(expected)
    for pool, records in counts.items():
        v = np.array(records); saved = point['coverage'][pool]
        close([saved[n] for n in ('query_mean_positive_document_recall_pct', 'queries_with_positive',
              'query_positive_hit_pct', 'positive_documents_retrieved', 'known_positive_documents')],
              [100*v[:, 0].mean(), v[:, 1].sum(), 100*v[:, 1].mean(), v[:, 2].sum(), v[:, 3].sum()], 'aggregate coverage')


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--production-complete', action='store_true', required=True)
    cli.add_argument('--freeze-sha256', required=True)
    args = cli.parse_args()
    assert args.production_complete and str(ROOT).startswith('/path/to/remote/lw_experiments/runs/')
    started = time.monotonic()
    assert sha(ROOT / 'frozen_manifest.json') == args.freeze_sha256
    manifest = read(ROOT / 'frozen_manifest.json')
    assert manifest['revision'] == '0e1bd15954eed1414023961a9f79038cf2aad2c6'
    for name, expected in RAW_HASHES.items():
        assert sha(ROOT / 'raw' / name) == expected, name
    for name, expected in manifest['code'].items():
        assert sha(ROOT / name) == expected, name
    for group in ('training_sources', 'input_sources', 'loader_sources', 'scoring_sources'):
        for name, expected in manifest[group].items():
            assert sha(Path(name)) == expected, (group, name)
    sealed = read(ROOT / 'predictions_sealed.json')
    assert sealed['frozen_manifest_sha256'] == sha(ROOT / 'frozen_manifest.json')
    assert sealed['source_model_sha256'] == sha(ROOT / 'source_model.json')
    for name, expected in sealed['files'].items():
        assert sha(ROOT / name) == expected, name
    model_seal = read(ROOT / 'model_sealed.json')
    assert model_seal['model_sha256'] == sealed['source_model_sha256']
    assert model_seal['frozen_manifest_sha256'] == args.freeze_sha256
    assert sealed['qrels_opened'] is False
    scale, coefficient = source_fit()
    spec = importlib.util.spec_from_file_location('protocol_lexical_parser', PARSER)
    parser = importlib.util.module_from_spec(spec); spec.loader.exec_module(parser)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoTokenizer
    ce_path = RUNS / 'dot_experiment_bundle/research/model_cache/cross-encoder_ms-marco-MiniLM-L6-v2_233902d25c440f23af6f7d6e94d2946bac0bee0a'
    qw_path = read(RUNS / 'dcr_stronger_baselines_v0.1_20261028/model_assets.json')['directory']
    tokenizers = [AutoTokenizer.from_pretrained(p, local_files_only=True, trust_remote_code=False) for p in (ce_path, qw_path)]
    result = {ds: dataset(ds, scale, coefficient, parser, tokenizers) for ds in DATASETS}
    summary = read(ROOT / 'summary.json')
    for ds in DATASETS:
        for method in METHODS:
            close([summary['datasets'][ds]['reports_pct'][method][k] for k in METRICS],
                  [result[ds]['reports'][method][k] for k in METRICS], (ds, method, 'point'))
        check_auxiliary(ds, result[ds], summary)
    for method in METHODS:
        close([summary['cross_dataset_equal_weight_reports_pct'][method][k] for k in METRICS],
              np.mean([[result[d]['reports'][method][k] for k in METRICS] for d in DATASETS], axis=0), 'macro reports')
    per_query = csv_rows(ROOT / 'per_query.csv.gz')
    seen = set()
    for r in per_query:
        ds, name, q = r['dataset'], r['method'], r['query_id']
        key = ds, name, q; assert key not in seen; seen.add(key)
        queries = result[ds]['queries']; index = next(i for i, v in enumerate(queries) if v['query_id'] == q)
        assert r['base_id'] == queries[index]['base_id'] and int(r['level']) == queries[index]['level']
        close([float(r[k]) for k in METRICS], result[ds]['values'][name][index], 'per query')
    assert len(seen) == 3*1804
    outcomes = csv_rows(ROOT / 'paired_outcomes.csv')
    assert len(outcomes) == 1804
    outcome_keys = set()
    for r in outcomes:
        ds, q = r['dataset'], r['query_id']; assert (ds, q) not in outcome_keys; outcome_keys.add((ds, q))
        fixed, post = result[ds]['orders']['fixed_Qwen20'][q][0], result[ds]['orders']['Post_G'][q][0]
        rel = result[ds]['qrels'][q]; fh, ph = int(rel.get(fixed, 0) > 0), int(rel.get(post, 0) > 0)
        assert r['fixed_top_document'] == fixed and r['post_g_top_document'] == post
        assert int(r['fixed_P1']) == fh and int(r['post_g_P1']) == ph
        assert r['outcome'] == ('rescued' if ph > fh else 'new_loss' if ph < fh else 'both_positive' if ph else 'neither_positive')
    differences = {ds: result[ds]['values']['Post_G']-result[ds]['values']['fixed_Qwen20'] for ds in DATASETS}
    draws = {ds: np.empty((10000, 4)) for ds in DATASETS}
    rng = random.Random(20261029)
    with np.load(ROOT / 'bootstrap_plan.npz', allow_pickle=False) as archive:
        plans = {ds: archive[ds] for ds in DATASETS}
    for draw in range(10000):
        for ds in DATASETS:
            count = np.zeros(len(result[ds]['bases']), dtype=np.uint16)
            for _ in range(len(count)):
                count[rng.randrange(len(count))] += 1
            assert np.array_equal(count, plans[ds][draw]), (ds, draw, 'RNG')
            weight = count[result[ds]['base_indices']]
            draws[ds][draw] = 100*(weight[:, None]*differences[ds]).sum(axis=0)/weight.sum()
    draws['equal_weight_macro'] = (draws['fiqa']+draws['nfcorpus'])/2
    intervals = csv_rows(ROOT / 'paired_bootstrap.csv')
    assert len(intervals) == len(summary['paired_bootstrap']) == 12
    for r, sr in zip(intervals, summary['paired_bootstrap']):
        ds, k = r['dataset'], r['metric']; j = METRICS.index(k)
        assert int(r['draws']) == 10000 and sr['draws'] == 10000
        assert sr['primary'] == (ds == 'equal_weight_macro' and k == 'P1')
        point = np.mean([100*differences[d][:, j].mean() for d in DATASETS]) if ds == 'equal_weight_macro' else 100*differences[ds][:, j].mean()
        expected = [point, *np.quantile(draws[ds][:, j], [.025, .975])]
        close([float(r[n]) for n in ('difference_pp', 'ci_low_pp', 'ci_high_pp')], expected, 'CI')
        close([sr[n] for n in ('difference_pp', 'ci_low_pp', 'ci_high_pp')], expected, 'summary CI')
    certificate = {'status': 'PASS_INDEPENDENT_IFIR_EXTERNAL_VERIFICATION',
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'wall_seconds': time.monotonic()-started,
        'frozen_manifest_sha256': sha(ROOT / 'frozen_manifest.json'), 'verifier_sha256': sha(Path(__file__)),
        'numeric': NUMERIC, 'checks': CHECKS, 'source_training_pairs': 2901, 'external_queries': 1804,
        'per_query_rows': len(per_query), 'bootstrap_draws': 10000, 'intervals': 12,
        'datasets': {ds: {'reports_pct': result[ds]['reports'], 'costs': result[ds]['costs']} for ds in DATASETS},
        'limits': ['Does not rerun neural inference.', 'Conditional fixed-model cluster bootstrap; no source-fit uncertainty.',
                   'Published positives are not exhaustive violation judgments.']}
    target = ROOT / 'independent_verification.json'
    with target.open('x') as stream:
        json.dump(certificate, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(certificate, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
