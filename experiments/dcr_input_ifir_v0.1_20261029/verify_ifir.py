"""Independent IFIR appended-input/fusion CPU audit, prepared before new scores.

No production ranking/metric/feature imports or neural inference. Numerical
execution is allowed only after this separate stage's production completes.
The final schema adapter is intentionally fail-closed while code is prepared.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'dcr_post_g_external_v0.1_20261029'
SOURCE = ROOT.parent / 'dcr_input_matched_fusion_v0.1_20261029'
DATASETS = ('fiqa', 'nfcorpus')
CELLS = ('MiniLM_topic', 'MiniLM_full', 'Qwen_topic', 'Qwen_full')
METRICS = ('P1', 'nDCG20', 'MRR20', 'R10')
METHODS = CELLS+('G12', 'G_same12', 'QL5')
CONTRASTS = {
    'MiniLM_full-MiniLM_topic': [-1, 1, 0, 0, 0, 0, 0],
    'Qwen_full-Qwen_topic': [0, 0, -1, 1, 0, 0, 0],
    'Qwen_topic-MiniLM_topic': [-1, 0, 1, 0, 0, 0, 0],
    'Qwen_full-MiniLM_full': [0, -1, 0, 1, 0, 0, 0],
    'input_effect_interaction': [1, -1, -1, 1, 0, 0, 0],
    'G_same12-QL5': [0, 0, 0, 0, 0, 1, -1],
    'G_same12-G12': [0, 0, 0, 0, -1, 1, 0],
    'G_same12-Qwen_full': [0, 0, 0, -1, 0, 1, 0]}
NUMERIC = {'comparisons': 0, 'maximum_absolute_difference': 0.}
A_CERTIFICATE_SHA = '05515d972b28256754dcd9b3bd9b8b9dc297c8cf064613bf92a6bb89db44194b'
OLD_CERTIFICATE_SHA = '7ad8aab1b9a2cd5d57372cdf7207855438527880e50b8c35bb6a104aa8795525'


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def rows(path: Path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def csv_rows(path: Path, delimiter: str = ',') -> list:
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', newline='') as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def close(actual, expected, label: str, tolerance: float = 1e-10) -> None:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), label
    error = float(np.max(np.abs(a-b))) if a.size else 0.
    NUMERIC['comparisons'] += int(a.size)
    NUMERIC['maximum_absolute_difference'] = max(NUMERIC['maximum_absolute_difference'], error)
    assert error <= tolerance, (label, error, tolerance)


def source_registry() -> tuple:
    assert sha(SOURCE / 'independent_verification.json') == A_CERTIFICATE_SHA
    a = read(SOURCE / 'independent_verification.json')
    assert a['status'] == 'PASS_INDEPENDENT_INPUT_MATCHED_FUSION_VERIFICATION'
    assert sha(SOURCE / 'frozen_manifest.json') == a['frozen_manifest_sha256']
    transfer = read(SOURCE / 'transfer_models.json')
    assert transfer['models'] == a['source_transfer_models']
    sealed = read(SOURCE / 'transfer_models_sealed.json')
    assert sealed['models_sha256'] == sha(SOURCE / 'transfer_models.json')
    assert sealed['before_A_score_and_any_new_IFIR_prediction'] is True
    a_predictions = read(SOURCE / 'predictions_sealed.json')
    assert a_predictions['files']['transfer_models.json'] == sha(SOURCE / 'transfer_models.json')
    assert a_predictions['files']['transfer_models_sealed.json'] == sha(SOURCE / 'transfer_models_sealed.json')
    assert sha(OLD / 'independent_verification.json') == OLD_CERTIFICATE_SHA
    old = read(OLD / 'independent_verification.json')
    assert old['status'] == 'PASS_INDEPENDENT_IFIR_EXTERNAL_VERIFICATION'
    assert sha(OLD / 'frozen_manifest.json') == old['frozen_manifest_sha256']
    for group in ('input_sources', 'scoring_sources'):
        for path, digest in read(OLD / 'frozen_manifest.json')[group].items():
            assert sha(Path(path)) == digest
    old_predictions = read(OLD / 'predictions_sealed.json')
    assert sha(OLD / 'source_model.json') == old_predictions['source_model_sha256']
    for name, digest in old_predictions['files'].items():
        assert sha(OLD / name) == digest, name
    return transfer['models'], read(OLD / 'source_model.json'), a_predictions['sealed_utc']


def historical_dataset(ds: str, old_model: dict) -> dict:
    queries = list(rows(OLD / ds / 'inputs/queries.jsonl'))
    qids = [r['query_id'] for r in queries]
    assert len(qids) == (1718 if ds == 'fiqa' else 86) and len(set(qids)) == len(qids)
    assert qids == sorted(qids) and all(r['level'] > 0 for r in queries)
    assert all(r['query_id'].rsplit('_', 1)[0] == r['base_id'] for r in queries)
    pools = {q: [] for q in qids}
    for r in csv_rows(OLD / ds / 'retrieval/candidates.csv'):
        assert int(r['BM25_rank']) == len(pools[r['query_id']])+1
        pools[r['query_id']].append(r['document_id'])
    assert all(len(p) == len(set(p)) == 100 for p in pools.values())
    score_rows = {}
    for name, count in [('MiniLM', 100), ('Qwen', 20)]:
        score_rows[name] = {q: {} for q in qids}
        path = OLD / ds / name / 'scores.jsonl.gz'
        assert sha(path) == read(path.parent / 'manifest.json')['scores_sha256']
        for r in rows(path):
            q, d = r['query_id'], r['document_id']
            assert d not in score_rows[name][q] and math.isfinite(r['score'])
            score_rows[name][q][d] = r
        assert all(len(p) == count for p in score_rows[name].values())
    meta = read(OLD / ds / 'outputs/feature_metadata.json')
    assert meta['qids'] == qids
    old_x = arrays(OLD / ds / 'outputs/features.npz')['X']
    assert old_x.shape == (len(qids), 20, 12)
    base_orders, heads, mini, qwen = {}, [], [], []
    for i, q in enumerate(qids):
        m, w = score_rows['MiniLM'][q], score_rows['Qwen'][q]
        assert set(m) == set(pools[q])
        base = sorted(m, key=lambda d: (-m[d]['score'], d)); head = base[:20]
        assert head == meta['heads'][i] and set(w) == set(head)
        assert all(m[d]['doc_prefix_end_char'] == w[d]['doc_prefix_end_char'] for d in head)
        base_orders[q] = base; heads.append(head)
        mini.append([m[d]['score'] for d in head]); qwen.append([w[d]['score'] for d in head])
    mini, qwen = np.array(mini), np.array(qwen)
    fixed = ranks(qwen, heads)
    old_g = ranks((old_x/np.array(old_model['scale'])) @ np.array(old_model['coef']), heads)
    seen = set()
    for r in rows(OLD / ds / 'outputs/rankings.jsonl.gz'):
        q, method = r['query_id'], r['method']; assert (q, method) not in seen; seen.add((q, method))
        i = qids.index(q); base = base_orders[q]
        if method == 'MiniLM':
            expected = base
        else:
            local = fixed[i] if method == 'fixed_Qwen20' else old_g[i]
            assert method in ('fixed_Qwen20', 'Post_G')
            expected = [heads[i][j] for j in local]+base[20:]
        assert r['order'] == expected
    assert len(seen) == 3*len(qids)
    return {'queries': queries, 'qids': qids, 'pools': pools, 'heads': heads, 'base_orders': base_orders,
            'old_X': old_x, 'old_score_rows': score_rows, 'MiniLM_topic': mini, 'Qwen_full': qwen,
            'old_G_order': old_g, 'fixed_order': fixed}


def ranks(scores: np.ndarray, heads: list) -> np.ndarray:
    assert scores.shape == (len(heads), 20) and np.isfinite(scores).all()
    return np.array([sorted(range(20), key=lambda j: (-float(s[j]), heads[i][j]))
                     for i, s in enumerate(scores)], dtype=np.uint8)


def matched_features(mini: np.ndarray, qwen: np.ndarray, old_x: np.ndarray, heads: list) -> np.ndarray:
    """Fixed original candidate rows; independently recompute changed score features."""
    mean = mini.mean(axis=1, keepdims=True); std = mini.std(axis=1, keepdims=True, ddof=0)
    z = np.divide(mini-mean, std, out=np.zeros(mini.shape), where=std != 0)
    rank = np.empty(mini.shape)
    np.put_along_axis(rank, ranks(mini, heads), np.broadcast_to(np.arange(1, 21)/20, mini.shape), axis=1)
    stdq = qwen.std(axis=1, keepdims=True, ddof=0)
    zq = np.divide(qwen-qwen.mean(axis=1, keepdims=True), stdq, out=np.zeros(qwen.shape), where=stdq != 0)
    close(zq, old_x[:, :, 1], 'unchanged Qwen z')
    sm, sq = np.sort(z, axis=1), np.sort(old_x[:, :, 1], axis=1)
    mgap, qgap = (sm[:, -1]-sm[:, -2])[:, None], (sq[:, -1]-sq[:, -2])[:, None]
    result = old_x.copy()
    result[:, :, 0], result[:, :, 2] = z, rank
    result[:, :, 8], result[:, :, 9], result[:, :, 10] = z*mgap, z*qgap, old_x[:, :, 1]*mgap
    assert np.array_equal(result[:, :, [1, 3, 11]], old_x[:, :, [1, 3, 11]])
    assert np.array_equal(result[:, :, 4:8], old_x[:, :, 4:8])
    return result


def independent_orders(historical: dict, mini_full: np.ndarray, qwen_topic: np.ndarray,
                       models: dict) -> tuple:
    heads, x = historical['heads'], historical['old_X']
    matched = matched_features(mini_full, historical['Qwen_full'], x, heads)
    same, ql = models['G_same12'], models['QL5']
    assert same['dimensions'] == 12 and ql['dimensions'] == 5
    same_score = (matched/np.array(same['scale'])) @ np.array(same['coef'])
    ql_score = (x[:, :, [1, 4, 5, 6, 7]].copy()/np.array(ql['scale'])) @ np.array(ql['coef'])
    order = {'MiniLM_topic': np.tile(np.arange(20, dtype=np.uint8), (len(heads), 1)),
        'MiniLM_full': ranks(mini_full, heads), 'Qwen_topic': ranks(qwen_topic, heads),
        'Qwen_full': historical['fixed_order'], 'G12': historical['old_G_order'],
        'G_same12': ranks(same_score, heads), 'QL5': ranks(ql_score, heads)}
    return matched, order


def stage_cells(ds: str, historical: dict, source_sealed_utc: str) -> tuple:
    qids, heads = historical['qids'], historical['heads']
    positions = {(q, d): (i, j) for i, (q, head) in enumerate(zip(qids, heads)) for j, d in enumerate(head)}
    pairs = list(rows(ROOT / ds / 'prepared/pairs.jsonl.gz'))
    assert len(pairs) == 20*len(qids) and {(r['query_id'], r['document_id']) for r in pairs} == set(positions)
    prepared = {(r['query_id'], r['document_id']): r for r in pairs}
    for key, r in prepared.items():
        q, d = key; i, j = positions[key]
        assert r['head_rank'] == j+1
        assert r['doc_prefix_end_char'] == historical['old_score_rows']['MiniLM'][q][d]['doc_prefix_end_char']
    for variant, field in [('topic', 'topic'), ('full', 'request')]:
        query_rows = list(rows(ROOT / ds / 'prepared' / f'queries_{variant}.jsonl'))
        assert query_rows == [{'query_id': r['query_id'], 'text': r[field]} for r in historical['queries']]
    local = read(ROOT / ds / 'prepared/summary.json')
    assert local['pairs'] == len(positions) and local['queries'] == len(qids)
    assert local['config_sha256'] == sha(ROOT / ds / 'config.py')
    for name, digest in local['outputs'].items():
        assert sha(ROOT / ds / name) == digest
    preflight = read(ROOT / ds / 'token_preflight.json')
    assert preflight['overflows'] == [] and preflight['no_retruncation'] is True
    assert preflight['pairs'] == preflight['old_pairs_aligned'] == len(positions)
    scores, records, inventory = {}, {}, {}
    for cell in CELLS:
        path = ROOT / ds / 'cells' / cell / 'scores.jsonl.gz'
        receipt = read(path.parent / 'receipt.json')
        assert receipt['exit_code'] == 0 and receipt['pairs'] == len(positions) and receipt['score_sha256'] == sha(path)
        scores[cell] = np.empty((len(qids), 20)); records[cell] = {}; tokens = []
        for r in rows(path):
            key = r['query_id'], r['document_id']; assert key in positions and key not in records[cell]
            assert math.isfinite(r['score']) and r['doc_prefix_end_char'] == prepared[key]['doc_prefix_end_char']
            assert r['input_tokens'] == prepared[key]['token_lengths'][cell]
            assert 0 < r['input_tokens'] <= (512 if cell.startswith('MiniLM') else 2048)
            if cell.startswith('Qwen'):
                close(r['score'], r['yes_logit']-r['no_logit'], (ds, cell, 'yes-no score'), tolerance=0.)
            i, j = positions[key]; scores[cell][i, j] = r['score']; records[cell][key] = r; tokens.append(r['input_tokens'])
            if cell in ('MiniLM_topic', 'Qwen_full'):
                old_name = 'MiniLM' if cell == 'MiniLM_topic' else 'Qwen'
                assert r == historical['old_score_rows'][old_name][key[0]][key[1]]
        assert set(records[cell]) == set(positions)
        length_stat = {'total': sum(tokens), 'max': max(tokens), 'min': min(tokens)}
        assert local['lengths'][cell] == length_stat and receipt['input_tokens_total'] == sum(tokens)
        if cell in ('MiniLM_topic', 'Qwen_full'):
            assert receipt['reused'] is True and receipt['new_neural_pairs'] == 0
        else:
            assert receipt['queries'] == len(qids) and receipt['references_read'] is False and receipt['task_retries'] == 0
            assert receipt['frozen_manifest_sha256'] == sha(ROOT / 'frozen_manifest.json')
            assert receipt['prepared_summary_sha256'] == sha(ROOT / ds / 'prepared/summary.json')
            assert receipt['precision'] == ('float32' if cell == 'MiniLM_full' else 'bfloat16')
            assert receipt['batch_size_configured'] == (32 if cell == 'MiniLM_full' else 16)
            assert receipt['max_actual_batch_size'] == (20 if cell == 'MiniLM_full' else 16)
            assert datetime.fromisoformat(receipt['started_utc']) >= datetime.fromisoformat(source_sealed_utc)
            assert datetime.fromisoformat(receipt['started_utc']) < datetime.fromisoformat('2026-10-30T04:00:00+08:00')
        inventory[cell] = {'pairs': len(positions), 'input_tokens': sum(tokens), 'score_sha256': sha(path)}
    close(scores['MiniLM_topic'], historical['MiniLM_topic'], (ds, 'old MiniLM reuse'), tolerance=0.)
    close(scores['Qwen_full'], historical['Qwen_full'], (ds, 'old Qwen reuse'), tolerance=0.)
    return scores, records, prepared, inventory


def verify_lengths(ds: str, historical: dict, records: dict, prepared: dict, tokenizers: tuple) -> dict:
    ce, qw = tokenizers
    query = {r['query_id']: r for r in historical['queries']}
    needed = {d for q, d in prepared}
    corpus = {r['document_id']: r['text'] for r in rows(OLD / ds / 'inputs/corpus.jsonl') if r['document_id'] in needed}
    assert set(corpus) == needed
    for (q, d), r in prepared.items():
        body = corpus[d]; end = r['doc_prefix_end_char']
        assert isinstance(end, int) and not isinstance(end, bool) and 0 <= end <= len(body)
        assert hashlib.sha256(body[:end].encode()).hexdigest() == r['body_prefix_sha256']
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    checked = {}
    for cell in ('MiniLM_full', 'Qwen_topic'):
        entries = list(records[cell].values()); checked[cell] = 0
        for start in range(0, len(entries), 128):
            batch = entries[start:start+128]
            bodies = [corpus[r['document_id']][:r['doc_prefix_end_char']] for r in batch]
            if cell == 'MiniLM_full':
                texts = [query[r['query_id']]['request'] for r in batch]
                encoded = ce(texts, bodies, truncation=False)['input_ids']
            else:
                texts = [prefix+'<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: '+query[r['query_id']]['topic']+'\n<Document>: '+body+suffix for r, body in zip(batch, bodies)]
                encoded = qw(texts, add_special_tokens=False, truncation=False)['input_ids']
            for r, encoded_ids in zip(batch, encoded):
                assert len(encoded_ids) == r['input_tokens']
                checked[cell] += 1
    print(json.dumps({'phase': 'independent input lengths', 'dataset': ds, 'pairs': checked}), flush=True)
    return checked


def evaluate_dataset(ds: str, historical: dict, order: dict, methods: tuple) -> dict:
    qids = historical['qids']; qrels = {q: {} for q in qids}
    for r in csv_rows(OLD / 'raw' / ds / f'{ds}-qrel.tsv', '\t'):
        if r['query_id'] in qrels:
            q, d = r['query_id'], r['corpus_id']
            assert d not in qrels[q]; qrels[q][d] = float(r['score'])
    full_orders = {m: {} for m in methods}
    values = {m: np.empty((len(qids), 4)) for m in methods}
    coverage = []
    for i, q in enumerate(qids):
        positive = {d for d, rel in qrels[q].items() if rel > 0}
        base = historical['base_orders'][q]
        for method in methods:
            full = [historical['heads'][i][j] for j in order[method][i]]+base[20:]
            assert len(full) == len(set(full)) == 100 and full[20:] == base[20:]
            full_orders[method][q] = full; values[method][i] = metrics(full, qrels[q])
        for k in (100, 20):
            found = len(positive & set(base[:k]))
            coverage.append({'query_id': q, 'k': k, 'positive_recall': found/len(positive) if positive else 0.,
                'hit': int(found > 0), 'found': found, 'total': len(positive)})
    bases = sorted({r['base_id'] for r in historical['queries']}); bindex = {b: i for i, b in enumerate(bases)}
    bootstrap_input = {'bases': bases, 'query_base_index': np.array([bindex[r['base_id']] for r in historical['queries']]),
        'P1_by_query_method': np.column_stack([values[m][:, 0] for m in methods])}
    return {'qrels': qrels, 'values': values, 'full_orders': full_orders, 'coverage': coverage,
            'bootstrap_input': bootstrap_input}


def verify_rankings(ds: str, historical: dict, matched: np.ndarray, order: dict) -> None:
    saved = arrays(ROOT / ds / 'outputs/features.npz')
    assert set(saved) == {'G_same12'}
    close(saved['G_same12'], matched, (ds, 'matched feature matrix'))
    assert np.array_equal(saved['G_same12'][:, :, [1, 3, 11]], historical['old_X'][:, :, [1, 3, 11]])
    assert np.array_equal(saved['G_same12'][:, :, 4:8], historical['old_X'][:, :, 4:8])
    index = {q: i for i, q in enumerate(historical['qids'])}; seen = set()
    for r in rows(ROOT / ds / 'outputs/rankings.jsonl.gz'):
        q, method = r['query_id'], r['method']; i = index[q]
        assert (q, method) not in seen and method in METHODS; seen.add((q, method))
        query = historical['queries'][i]
        assert r['base_id'] == query['base_id'] and r['level'] == query['level']
        base = historical['base_orders'][q]
        expected = [historical['heads'][i][j] for j in order[method][i]]+base[20:]
        assert r['order'] == expected and len(set(r['order'])) == 100, (ds, q, method)
    assert len(seen) == len(METHODS)*len(index)


def verify_reports(historical: dict, evaluated: dict, plans: dict, draws: dict) -> dict:
    summary = read(ROOT / 'summary.json'); old_summary = read(OLD / 'summary.json')
    for ds in DATASETS:
        result = evaluated[ds]; queries = historical[ds]['queries']; saved = summary['datasets'][ds]
        assert saved['queries'] == len(queries) and saved['old_G_fixed_all_query_reproduction'] is True
        for method in METHODS:
            close([saved['reports_pct'][method][k] for k in METRICS], 100*result['values'][method].mean(axis=0), (ds, method, 'point report'))
        for method, prior in [('MiniLM_topic', 'MiniLM'), ('Qwen_full', 'fixed_Qwen20'), ('G12', 'Post_G')]:
            close([saved['reports_pct'][method][k] for k in METRICS],
                [old_summary['datasets'][ds]['reports_pct'][prior][k] for k in METRICS], (ds, method, 'old point reproduction'))
        for level in sorted({q['level'] for q in queries}):
            chosen = [i for i, q in enumerate(queries) if q['level'] == level]; observed = saved['level_reports'][str(level)]
            assert observed['queries'] == len(chosen)
            for method in METHODS:
                close([observed['reports_pct'][method][k] for k in METRICS],
                    100*result['values'][method][chosen].mean(axis=0), (ds, level, method, 'descriptive level point'))
        for pool, k in [('BM25_top100', 100), ('MiniLM_top20', 20)]:
            rr = [r for r in result['coverage'] if r['k'] == k]; cov = saved['coverage'][pool]
            close([cov[n] for n in ('query_mean_positive_document_recall_pct', 'queries_with_positive',
                'query_positive_hit_pct', 'positive_documents_retrieved', 'known_positive_documents')],
                [100*np.mean([r['positive_recall'] for r in rr]), sum(r['hit'] for r in rr),
                 100*np.mean([r['hit'] for r in rr]), sum(r['found'] for r in rr), sum(r['total'] for r in rr)], (ds, pool, 'coverage'))
    for method in METHODS:
        close([summary['equal_weight_macro_reports_pct'][method][k] for k in METRICS],
            100*np.mean([evaluated[ds]['values'][method].mean(axis=0) for ds in DATASETS], axis=0), (method, 'macro point'))
    per_query = csv_rows(ROOT / 'per_query.csv.gz'); assert len(per_query) == 1804*len(METHODS)
    index = {ds: {q: i for i, q in enumerate(historical[ds]['qids'])} for ds in DATASETS}; seen = set()
    for r in per_query:
        ds, method, q = r['dataset'], r['method'], r['query_id']; key = ds, method, q
        assert key not in seen; seen.add(key); i = index[ds][q]; query = historical[ds]['queries'][i]
        assert r['base_id'] == query['base_id'] and int(r['level']) == query['level']
        close([float(r[k]) for k in METRICS], evaluated[ds]['values'][method][i], 'per-query metrics')
    # The new stage uses exactly the old fixed, dataset-independent cluster schedule.
    with np.load(OLD / 'bootstrap_plan.npz', allow_pickle=False) as old_plan:
        for ds in DATASETS:
            assert np.array_equal(plans[ds], old_plan[ds]), (ds, 'frozen old cluster plan')
    expected = {}
    for ds in DATASETS+('equal_weight_macro',):
        point = np.mean([evaluated[d]['bootstrap_input']['P1_by_query_method'].mean(axis=0) for d in DATASETS], axis=0) if ds == 'equal_weight_macro' else evaluated[ds]['bootstrap_input']['P1_by_query_method'].mean(axis=0)
        for name, coefficient in CONTRASTS.items():
            coefficient = np.array(coefficient)
            samples = 100*(draws[ds] @ coefficient)
            expected[ds, name] = [100*float(point @ coefficient), *np.quantile(samples, [.025, .975])]
    assert len(expected) == len(summary['paired_bootstrap']) == 24
    assert {(r['dataset'], r['comparison']) for r in summary['paired_bootstrap']} == set(expected)
    for r in summary['paired_bootstrap']:
        assert r['metric'] == 'P1' and r['draws'] == 10000
        assert r['primary'] == (r['dataset'] == 'equal_weight_macro' and r['comparison'] == 'input_effect_interaction')
        close([r[k] for k in ('difference_pp', 'ci_low_pp', 'ci_high_pp')], expected[r['dataset'], r['comparison']], 'paired P1 interval')
    interval_csv = csv_rows(ROOT / 'paired_bootstrap.csv'); assert len(interval_csv) == 24
    assert {(r['dataset'], r['comparison']) for r in interval_csv} == set(expected)
    for r in interval_csv:
        assert r['metric'] == 'P1' and int(r['draws']) == 10000
        assert r['primary'] == str(r['dataset'] == 'equal_weight_macro' and r['comparison'] == 'input_effect_interaction')
        close([float(r[k]) for k in ('difference_pp', 'ci_low_pp', 'ci_high_pp')], expected[r['dataset'], r['comparison']], 'paired interval CSV')
    assert summary['bootstrap']['draws'] == 10000 and summary['bootstrap']['seed'] == 20261029
    assert summary['bootstrap']['fixed_model_no_refit'] is True
    return summary


def verify_transitions(historical: dict, evaluated: dict, summary: dict) -> int:
    pairs = [('MiniLM_full', 'MiniLM_topic'), ('Qwen_full', 'Qwen_topic'),
             ('G_same12', 'QL5'), ('G_same12', 'G12'), ('G_same12', 'Qwen_full')]
    expected = {}
    for ds in DATASETS:
        for a, b in pairs:
            before, after = evaluated[ds]['values'][b][:, 0], evaluated[ds]['values'][a][:, 0]
            rescue, loss = (after == 1) & (before == 0), (after == 0) & (before == 1)
            name = a+'-'+b; saved = summary['datasets'][ds]['transitions'][name]
            assert saved == {'rescued': int(rescue.sum()), 'new_losses': int(loss.sum()), 'net': int(rescue.sum()-loss.sum())}
            for i, q in enumerate(historical[ds]['queries']):
                expected[ds, name, q['query_id']] = q['base_id'], int(rescue[i]), int(loss[i])
    rows = csv_rows(ROOT / 'paired_outcomes.csv'); assert len(rows) == len(expected) == 5*1804
    seen = set()
    for r in rows:
        key = r['dataset'], r['comparison'], r['query_id']; assert key not in seen; seen.add(key)
        assert (r['base_id'], int(r['rescued']), int(r['new_loss'])) == expected[key]
    return len(rows)


def metrics(order: list, relevance: dict) -> np.ndarray:
    positive = {d for d, value in relevance.items() if value > 0}
    first = next((rank for rank, d in enumerate(order[:20], 1) if d in positive), 0)
    gain = sum(relevance.get(d, 0)/math.log2(rank+1) for rank, d in enumerate(order[:20], 1))
    ideal = sum(value/math.log2(rank+1) for rank, value in enumerate(sorted(
        (value for value in relevance.values() if value > 0), reverse=True)[:20], 1))
    return np.array([float(order[0] in positive), gain/ideal if ideal else 0.,
        1/first if first else 0., len(positive & set(order[:10]))/len(positive) if positive else 0.])


def bootstrap(input_by_dataset: dict, methods: tuple) -> tuple:
    """Fixed models; draw outer, fiqa then nfcorpus, sorted-base randrange."""
    rng = random.Random(20261029)
    plans = {ds: np.empty((10000, len(input_by_dataset[ds]['bases'])), dtype=np.uint16) for ds in DATASETS}
    draws = {ds: np.empty((10000, len(methods))) for ds in DATASETS}
    for draw in range(10000):
        for ds in DATASETS:
            source = input_by_dataset[ds]; n = len(source['bases'])
            count = np.zeros(n, dtype=np.uint16)
            for _ in range(n):
                count[rng.randrange(n)] += 1
            plans[ds][draw] = count
            weight = count[source['query_base_index']]
            # P1 only has intervals. Other metrics remain complete point tables.
            draws[ds][draw] = (weight[:, None]*source['P1_by_query_method']).sum(axis=0)/weight.sum()
    draws['equal_weight_macro'] = (draws['fiqa']+draws['nfcorpus'])/2
    return plans, draws


def fixtures() -> dict:
    assert metrics(['x'], {'a': 1}).tolist() == [0., 0., 0., 0.]
    assert metrics(['x'], {}).tolist() == [0., 0., 0., 0.]
    r = metrics(['b', 'a'], {'a': 4, 'b': 2})
    close(r[1], (2+4/math.log2(3))/(4+2/math.log2(3)), 'raw rel gain fixture')
    assert metrics(['a'], {'a': 1, 'b': 1, 'c': 1})[3] == 1/3
    # Sampling one two-request family and one one-request family yields query mean.
    p = np.array([1., 1., 0.]); weight = np.array([1, 1])[np.array([0, 0, 1])]
    close(weight @ p/weight.sum(), 2/3, 'query mean not family mean')
    return {'all_queries_full_qrels_denominator': True, 'raw_rel_gain': True,
            'fixed_prediction_query_weighted_family_sampling': True}


def provenance(expected_freeze: str) -> dict:
    assert sha(ROOT / 'frozen_manifest.json') == expected_freeze
    manifest = read(ROOT / 'frozen_manifest.json')
    assert 'verify_ifir.py' not in manifest['code']
    for name, digest in manifest['code'].items():
        assert sha(ROOT / name) == digest, name
    source_entries = 0
    for group in ('input_sources', 'loader_sources', 'scoring_sources'):
        for name, digest in manifest[group].items():
            assert sha(Path(name)) == digest, name
            source_entries += 1
    config = manifest['config']
    assert config['new_neural_pairs'] == 72160 and config['P1_CIs'] == 24
    assert config['bootstrap'] == 10000 and config['seed'] == 20261029 and config['CPU_threads'] == 1
    assert config['cutoff_Beijing'] == '2026-10-30T04:00:00+08:00'
    assert manifest['preparation_sha256'] == sha(ROOT / 'preparation.json')
    assert manifest['fixtures_sha256'] == sha(ROOT / 'fixtures.json') and read(ROOT / 'fixtures.json')['status'] == 'PASS'
    assert manifest['resource_preflight_sha256'] == sha(ROOT / 'resource_preflight.json')
    assert read(ROOT / 'resource_preflight.json')['passed'] is True and read(ROOT / 'resource_gate.json')['passed'] is True
    preparation = read(ROOT / 'preparation.json')
    assert preparation['model_loaded'] is False and preparation['qrels_opened'] is False
    assert preparation['source_models_sha256'] == sha(SOURCE / 'transfer_models.json')
    for name, digest in preparation['sources'].items():
        assert sha(Path(name)) == digest
    for name, digest in preparation['outputs'].items():
        assert sha(ROOT / name) == digest
    asset_files = 0
    for model, asset in preparation['assets'].items():
        assert model in ('MiniLM', 'Qwen')
        for r in asset['files']:
            path = Path(asset['directory']) / r['name']; assert path.stat().st_size == r['bytes']
            if r['algorithm'] == 'sha256':
                digest = sha(path)
            else:
                assert r['algorithm'] == 'git_blob_sha1'
                h = hashlib.sha1(b'blob '+str(r['bytes']).encode()+b'\0')
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1048576), b''):
                        h.update(block)
                digest = h.hexdigest()
            assert digest == r['digest'], (model, r['name'])
            asset_files += 1
    release = read(ROOT / 'parent_release.json'); assert release['frozen_manifest_sha256'] == expected_freeze
    for ds in DATASETS:
        assert sha(ROOT / ds / 'frozen_manifest.json') == expected_freeze
        assert sha(ROOT / ds / 'parent_release.json') == sha(ROOT / 'parent_release.json')
    seal = read(ROOT / 'predictions_sealed.json')
    assert seal['frozen_manifest_sha256'] == expected_freeze and seal['qrels_opened'] is False
    assert seal['source_models_sha256'] == sha(SOURCE / 'transfer_models.json')
    assert seal['old_G_source_model_sha256'] == sha(OLD / 'source_model.json')
    expected = {f'{ds}/outputs/{name}' for ds in DATASETS for name in ('rankings.jsonl.gz', 'features.npz')} | {
        f'{ds}/cells/{cell}/{name}' for ds in DATASETS for cell in CELLS for name in ('scores.jsonl.gz', 'receipt.json')}
    assert set(seal['files']) == expected
    for name, digest in seal['files'].items():
        assert sha(ROOT / name) == digest
    finished = read(ROOT / 'production_finished.json')
    assert finished['exit_code'] == 0 and finished['task_retries'] == 0 and finished['new_neural_pairs'] == 72160
    assert finished['wall_seconds'] <= 14400 and finished['GPU_stage_wall_seconds_including_wait_and_load'] <= 10800
    begin = read(ROOT / 'production_started.json')
    assert begin['overall_seconds_limit'] == 14400 and begin['GPU_stage_limit_seconds'] == 10800
    assert datetime.fromisoformat(begin['started_utc']) >= datetime.fromisoformat(read(SOURCE / 'independent_verification.json')['completed_utc'])
    return {'frozen_code_files': len(manifest['code']), 'phase_source_entries': source_entries,
        'model_asset_files': asset_files, 'prediction_sealed_files': len(seal['files']),
        'production_wall_seconds': finished['wall_seconds'],
        'GPU_stage_wall_seconds_including_wait_and_load': finished['GPU_stage_wall_seconds_including_wait_and_load']}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--production-complete', action='store_true', required=True)
    parser.add_argument('--freeze-sha256', required=True)
    args = parser.parse_args()
    assert args.production_complete and str(ROOT).startswith('/path/to/remote/lw_experiments/runs/')
    started = time.monotonic(); source_report = provenance(args.freeze_sha256)
    fixture_report = fixtures(); models, old_model, a_sealed_utc = source_registry()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoTokenizer
    cache = ROOT.parent / 'dot_experiment_bundle/research/model_cache/cross-encoder_ms-marco-MiniLM-L6-v2_233902d25c440f23af6f7d6e94d2946bac0bee0a'
    qwen_path = read(ROOT.parent / 'dcr_stronger_baselines_v0.1_20261028/model_assets.json')['directory']
    tokenizers = tuple(AutoTokenizer.from_pretrained(p, local_files_only=True, trust_remote_code=False) for p in (cache, qwen_path))
    historical, evaluated, inventory, checked = {}, {}, {}, {}
    for ds in DATASETS:
        historical[ds] = historical_dataset(ds, old_model)
        score, records, prepared, inventory[ds] = stage_cells(ds, historical[ds], a_sealed_utc)
        checked[ds] = verify_lengths(ds, historical[ds], records, prepared, tokenizers)
        matched, order = independent_orders(historical[ds], score['MiniLM_full'], score['Qwen_topic'], models)
        verify_rankings(ds, historical[ds], matched, order)
        evaluated[ds] = evaluate_dataset(ds, historical[ds], order, METHODS)
    plans, draws = bootstrap({ds: evaluated[ds]['bootstrap_input'] for ds in DATASETS}, METHODS)
    summary = verify_reports(historical, evaluated, plans, draws)
    outcomes = verify_transitions(historical, evaluated, summary)
    assert source_report == provenance(args.freeze_sha256)
    certificate = {'status': 'PASS_INDEPENDENT_IFIR_INPUT_FUSION_VERIFICATION',
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'wall_seconds': time.monotonic()-started,
        'frozen_manifest_sha256': args.freeze_sha256, 'verifier_sha256': sha(Path(__file__)),
        'source_A_certificate_sha256': A_CERTIFICATE_SHA, 'source_old_IFIR_certificate_sha256': OLD_CERTIFICATE_SHA,
        'provenance': source_report, 'fixtures': fixture_report, 'numeric': NUMERIC, 'methods': list(METHODS),
        'queries': 1804, 'bases': 725, 'bootstrap_draws': 10000, 'paired_P1_intervals': 24,
        'new_neural_score_rows_audited': 72160, 'four_cell_score_rows': 4*36080,
        'new_input_lengths_checked': checked, 'complete_rankings': 7*1804,
        'per_query_metric_rows': 7*1804, 'paired_outcome_rows': outcomes, 'cell_inventory': inventory,
        'reports': {ds: summary['datasets'][ds]['reports_pct'] for ds in DATASETS},
        'macro_reports': summary['equal_weight_macro_reports_pct'], 'paired_bootstrap': summary['paired_bootstrap'],
        'transitions': {ds: summary['datasets'][ds]['transitions'] for ds in DATASETS},
        'limits': ['Previously observed IFIR data; appended replay, not a pristine test.',
            'Source models are fixed from ExcluIR; no IFIR fitting or training RMS recomputation.',
            'No independent GPU inference; score provenance/use, input lengths and statistics are verified.',
            'Candidate/body conditioning and different historical BM25 inputs remain.',
            'Same needs extra MiniLM-full20 for deployment, despite sharing Qwen-full20.',
            'Published qrels do not label all unjudged documents as violations.']}
    with (ROOT / 'independent_verification.json').open('x') as stream:
        json.dump(certificate, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(certificate, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
