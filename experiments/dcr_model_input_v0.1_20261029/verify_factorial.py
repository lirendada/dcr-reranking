"""Independent fixed-candidate model-by-input audit (CPU, after production).

No production mathematics/scoring imports. Neural scores are not regenerated;
selection, prefix/prompt/length provenance, canonical ranking, per-query metrics
and all paired contrasts are independently checked. The estimand is conditional
on the original MiniLM-topic head20, not full-corpus retrieval performance.
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
POST = ROOT.parent / 'dcr_post_fusion_v0.1_20261029'
GATE = ROOT.parent / 'dcr_gate_mechanism_v0.1_20261029'
CELLS = ('MiniLM_topic', 'MiniLM_full', 'Qwen_topic', 'Qwen_full')
METRICS = ('P1', 'N1', 'U1', 'R10')
# Ordered as topic/full for MiniLM then Qwen. Sign convention is frozen here.
CONTRASTS = {
    'MiniLM_full-MiniLM_topic': np.array([-1., 1., 0., 0.]),
    'Qwen_full-Qwen_topic': np.array([0., 0., -1., 1.]),
    'Qwen_topic-MiniLM_topic': np.array([-1., 0., 1., 0.]),
    'Qwen_full-MiniLM_full': np.array([0., -1., 0., 1.]),
    'input_effect_interaction': np.array([1., -1., -1., 1.])}
NUMERIC = {'comparisons': 0, 'maximum_absolute_difference': 0.}


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def rows(path: Path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def csv_rows(path: Path) -> list:
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', newline='') as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, label: str, tolerance: float = 1e-10) -> None:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), label
    error = float(np.max(np.abs(a-b))) if a.size else 0.
    NUMERIC['comparisons'] += int(a.size)
    NUMERIC['maximum_absolute_difference'] = max(NUMERIC['maximum_absolute_difference'], error)
    assert error <= tolerance, (label, error, tolerance)


def rank(scores: np.ndarray, heads: list) -> np.ndarray:
    assert scores.ndim == 2 and scores.shape[1] == 20 and len(heads) == len(scores)
    assert np.isfinite(scores).all()
    return np.array([sorted(range(20), key=lambda j: (-row[j], heads[i][j]))
                     for i, row in enumerate(scores)], dtype=np.uint8)


def source_inputs() -> tuple:
    certificate = POST / 'independent_verification.json'
    assert sha(certificate) == '0c3d920df1cf5e180bb402bd8b7decf4a35fece2982ba63ad07276493ca60336'
    assert read(certificate)['status'] == 'PASS_INDEPENDENT_POST_FUSION_VERIFICATION'
    meta = read(POST / 'head_and_tail.json')
    qids, pools = meta['qids'], meta['ranked_pools']
    assert len(qids) == len(pools) == 3452 and qids == sorted(qids) and len(set(qids)) == 3452
    assert all(len(pool) == len(set(pool)) == 100 for pool in pools)
    source = arrays(POST / 'features.npz')
    assert source['raw_scores'].shape == (3452, 20, 2)
    plan = arrays(POST / 'prepared/plan.npz')
    assert plan['weights'].shape == (1000, 3452)
    groups = read(ROOT.parent / 'dcr_constraint_router_v0.1_20261029/prepared/groups.json')
    assert groups['qids'] == qids and len(groups['components']) == 3239
    assert np.array_equal(plan['fold'], [groups['query_fold'][q] for q in qids])
    index = {q: i for i, q in enumerate(qids)}
    members = {c: [index[q] for q in qq] for c, qq in groups['components'].items()}
    rng = random.Random(20261029)
    for draw in range(1000):
        weights = np.zeros(3452, dtype=np.int16)
        for fold in range(5):
            components = groups['fold_component_IDs'][str(fold)]
            assert components == sorted(components)
            for _ in components:
                weights[members[components[rng.randrange(len(components))]]] += 1
        assert np.array_equal(weights, plan['weights'][draw]), ('component RNG', draw)
    return source, meta, plan


def indicators(order: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    p, n = order[:, 0] == positive, order[:, 0] == negative
    return np.column_stack([p, n, ~(p | n), np.any(order[:, :10] == positive[:, None], axis=1)])


def evaluate(order: dict, plan: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> tuple:
    per_query = np.array([indicators(order[cell], positive, negative) for cell in CELLS])
    values = np.empty((1001, 4, 4))
    values[0] = per_query.mean(axis=1)
    # Fixed prediction resampling: no fit, threshold selection or cross-query allocation.
    for draw, weights in enumerate(plan, 1):
        values[draw] = np.sum(per_query*weights[None, :, None], axis=1)/weights.sum()
    comparisons = []
    for name, coefficient in CONTRASTS.items():
        difference = 100*np.einsum('imk,m->ik', values, coefficient)
        for j, metric in enumerate(METRICS[:2]):
            lo, hi = np.quantile(difference[1:, j], [.025, .975])
            comparisons.append({'comparison': name, 'metric': metric,
                'primary': name == 'input_effect_interaction' and metric == 'P1',
                'difference_pp': float(difference[0, j]), 'ci_low_pp': float(lo),
                'ci_high_pp': float(hi), 'draws': 1000})
    return per_query, values, comparisons


def old_scores(meta: dict, source: dict) -> tuple:
    qids, pools = meta['qids'], meta['ranked_pools']
    query_index = {q: i for i, q in enumerate(qids)}
    head_sets = {q: set(pool[:20]) for q, pool in zip(qids, pools)}
    mini, qwen = {q: {} for q in qids}, {q: {} for q in qids}
    for r in rows(GATE / 'external_MiniLM/scores.jsonl.gz'):
        q, d = r['query_id'], r['document_id']
        assert d not in mini[q] and math.isfinite(r['score'])
        mini[q][d] = r
    for r in rows(GATE / 'external_Qwen/scores.jsonl.gz'):
        q, d = r['query_id'], r['document_id']
        if d not in head_sets[q]:
            continue
        # Never use a Qwen score outside the fixed source head.
        assert d not in qwen[q] and math.isfinite(r['score'])
        assert r['doc_prefix_end_char'] == mini[q][d]['doc_prefix_end_char']
        qwen[q][d] = r
    scores = {'MiniLM_topic': np.empty((3452, 20)), 'Qwen_full': np.empty((3452, 20))}
    selected = []
    for q, pool in zip(qids, pools):
        i = query_index[q]
        assert len(mini[q]) == 100 and set(mini[q]) == set(pool)
        assert sorted(mini[q], key=lambda d: (-mini[q][d]['score'], d)) == pool
        assert set(qwen[q]) == head_sets[q]
        for j, d in enumerate(pool[:20]):
            scores['MiniLM_topic'][i, j] = mini[q][d]['score']
            scores['Qwen_full'][i, j] = qwen[q][d]['score']
            selected.append({'query_id': q, 'document_id': d,
                             'doc_prefix_end_char': mini[q][d]['doc_prefix_end_char']})
    close(scores['MiniLM_topic'], source['raw_scores'][:, :, 0], 'source MiniLM head scores')
    close(scores['Qwen_full'], source['raw_scores'][:, :, 1], 'source Qwen head scores')
    old_selected = {'MiniLM_topic': {(q, d): mini[q][d] for q in qids for d in pools[query_index[q]][:20]},
                    'Qwen_full': {(q, d): qwen[q][d] for q in qids for d in pools[query_index[q]][:20]}}
    return scores, selected, old_selected


def cell_scores(meta: dict, original: dict, selected: list, reused: dict) -> tuple:
    qids, pools = meta['qids'], meta['ranked_pools']
    positions = {(q, d): (i, j) for i, (q, pool) in enumerate(zip(qids, pools)) for j, d in enumerate(pool[:20])}
    source_prefix = {(r['query_id'], r['document_id']): r['doc_prefix_end_char'] for r in selected}
    prepared = list(rows(ROOT / 'prepared/pairs.jsonl.gz'))
    assert len(prepared) == 69040
    assert {(r['query_id'], r['document_id']) for r in prepared} == set(positions)
    assert all(r['doc_prefix_end_char'] == source_prefix[r['query_id'], r['document_id']] for r in prepared)
    prepared_map = {(r['query_id'], r['document_id']): r for r in prepared}
    assert all(r['head_rank'] == positions[key][1]+1 for key, r in prepared_map.items())
    alignment = csv_rows(ROOT / 'prepared/old_pair_alignment.csv.gz')
    assert len(alignment) == 69040
    assert {(r['query_id'], r['document_id']) for r in alignment} == set(positions)
    for r in alignment:
        expected = prepared_map[r['query_id'], r['document_id']]
        assert int(r['head_rank']) == expected['head_rank']
        assert int(r['doc_prefix_end_char']) == expected['doc_prefix_end_char']
        assert r['body_prefix_sha256'] == expected['body_prefix_sha256']
        assert all(int(r[cell]) == expected['token_lengths'][cell] for cell in CELLS)
    scores, inventory, records = {}, {}, {}
    for cell in CELLS:
        scores[cell] = np.empty((3452, 20)); seen = set(); tokens = 0; records[cell] = {}
        path = ROOT / 'cells' / cell / 'scores.jsonl.gz'
        for r in rows(path):
            key = r['query_id'], r['document_id']
            assert key in positions and key not in seen; seen.add(key)
            assert math.isfinite(r['score']) and r['doc_prefix_end_char'] == source_prefix[key]
            assert isinstance(r['input_tokens'], int) and 0 < r['input_tokens'] <= (512 if cell.startswith('MiniLM') else 2048)
            assert r['input_tokens'] == prepared_map[key]['token_lengths'][cell]
            i, j = positions[key]; scores[cell][i, j] = r['score']; records[cell][key] = r
            tokens += r['input_tokens']
            if cell.startswith('Qwen'):
                close(r['score'], r['yes_logit']-r['no_logit'], 'yes-no score', tolerance=0.)
            if cell in reused:
                expected = reused[cell][key]
                assert r['score'] == expected['score'] and r['input_tokens'] == expected['input_tokens']
        assert seen == set(positions)
        if cell in original:
            close(scores[cell], original[cell], 'exact reused cell', tolerance=0.)
        inventory[cell] = {'pairs': len(seen), 'input_tokens_total': tokens, 'scores_sha256': sha(path)}
        receipt = read(path.parent / 'receipt.json')
        assert receipt['cell'] == cell and receipt['pairs'] == 69040 and receipt['queries'] == 3452
        assert receipt['exit_code'] == 0 and receipt['score_sha256'] == sha(path)
        assert receipt['input_tokens_total'] == tokens
        if cell in reused:
            assert receipt['new_model_pairs'] == 0 and receipt['model_inference_rerun'] is False
            assert receipt['original_fields_preserved'] is True
            assert receipt['prefix_and_token_alignment_verified'] is True
            old_name = 'external_MiniLM' if cell == 'MiniLM_topic' else 'external_Qwen'
            assert receipt['source_score_sha256'] == sha(GATE / old_name / 'scores.jsonl.gz')
        else:
            assert receipt['references_read'] is False and receipt['task_retries'] == 0
            assert receipt['frozen_manifest_sha256'] == sha(ROOT / 'frozen_manifest.json')
            assert receipt['prepared_summary_sha256'] == sha(ROOT / 'prepared/summary.json')
            assert receipt['batch_size_configured'] == (32 if cell.startswith('MiniLM') else 16)
            assert receipt['max_actual_batch_size'] == (20 if cell.startswith('MiniLM') else 16)
            assert receipt['precision'] == ('float32' if cell.startswith('MiniLM') else 'bfloat16')
            assert receipt['environment']['transformers'] == ('4.49.0' if cell.startswith('MiniLM') else '4.51.3')
    preflight = read(ROOT / 'prepared/token_preflight.json')
    assert preflight['overflows'] == [] and preflight['silent_retruncation'] is False
    assert preflight['model_loaded'] is False and preflight['references_opened'] is False
    for cell in CELLS:
        lengths = sorted(r['input_tokens'] for r in records[cell].values())
        statistics = {'pairs': len(lengths), 'min': lengths[0], 'max': lengths[-1], 'total': sum(lengths),
                      'p50': lengths[len(lengths)//2], 'p95': lengths[int(len(lengths)*.95)]}
        assert preflight['token_statistics'][cell] == statistics
    return scores, records, inventory


def token_lengths(records: dict) -> dict:
    """Tokenizer-only reconstruction of complete new input strings; no model loads."""
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoTokenizer
    ce_path = ROOT.parent / 'dot_experiment_bundle/research/model_cache/cross-encoder_ms-marco-MiniLM-L6-v2_233902d25c440f23af6f7d6e94d2946bac0bee0a'
    qw_path = read(ROOT.parent / 'dcr_stronger_baselines_v0.1_20261028/model_assets.json')['directory']
    ce, qw = [AutoTokenizer.from_pretrained(p, local_files_only=True, trust_remote_code=False) for p in (ce_path, qw_path)]
    query = {variant: {r['query_id']: r['text'] for r in rows(GATE / 'external_inputs' / filename)}
        for variant, filename in [('topic', 'queries_question0.jsonl'), ('full', 'queries_RQ_rewrite.jsonl')]}
    documents = {d for q, d in records['MiniLM_full']}
    corpus = {r['document_id']: r['text'] for r in rows(GATE / 'external_inputs/corpus.jsonl') if r['document_id'] in documents}
    for r in rows(ROOT / 'prepared/pairs.jsonl.gz'):
        body = corpus[r['document_id']]; end = r['doc_prefix_end_char']
        assert 0 <= end <= len(body)
        assert hashlib.sha256(body[:end].encode()).hexdigest() == r['body_prefix_sha256']
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    checked = {}
    for cell in ('MiniLM_full', 'Qwen_topic'):
        entries = list(records[cell].values()); checked[cell] = 0
        for offset in range(0, len(entries), 128):
            batch = entries[offset:offset+128]
            bodies = []
            for r in batch:
                body = corpus[r['document_id']]; end = r['doc_prefix_end_char']
                assert 0 <= end <= len(body); bodies.append(body[:end])
            if cell == 'MiniLM_full':
                encoded = ce([query['full'][r['query_id']] for r in batch], bodies, truncation=False)['input_ids']
            else:
                texts = [prefix+'<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: '+query['topic'][r['query_id']]+'\n<Document>: '+body+suffix for r, body in zip(batch, bodies)]
                encoded = qw(texts, add_special_tokens=False, truncation=False)['input_ids']
            for r, ids in zip(batch, encoded):
                assert len(ids) == r['input_tokens'] <= (512 if cell == 'MiniLM_full' else 2048)
                checked[cell] += 1
        print(json.dumps({'phase': 'independent token lengths', 'cell': cell, 'pairs': checked[cell]}), flush=True)
    return checked


def verify_outputs(meta: dict, scores: dict, orders: dict, plan: dict) -> dict:
    qids, pools = meta['qids'], meta['ranked_pools']
    index = {q: i for i, q in enumerate(qids)}
    recorded = {cell: set() for cell in CELLS}
    for r in rows(ROOT / 'rankings.jsonl.gz'):
        cell, q = r['cell'], r['query_id']; i = index[q]
        assert q not in recorded[cell]; recorded[cell].add(q)
        expected = [pools[i][j] for j in orders[cell][i]]+pools[i][20:]
        assert r['order'] == expected and len(set(r['order'])) == 100
    assert all(seen == set(qids) for seen in recorded.values())
    saved_arrays = arrays(ROOT / 'four_cell_arrays.npz')
    assert saved_arrays['cells'].tolist() == list(CELLS) and saved_arrays['qids'].tolist() == qids
    close(saved_arrays['scores'], np.array([scores[cell] for cell in CELLS]), 'all four scores')
    assert np.array_equal(saved_arrays['orders'], np.array([orders[cell] for cell in CELLS]))
    references = read(GATE / 'external_inputs/references.json')
    positive, negative = [], []
    for q, pool in zip(qids, pools):
        p, n = references[q]['positive_id'], references[q]['negative_id']
        positive.append(pool.index(p) if p in pool else -1)
        negative.append(pool.index(n) if n in pool else -1)
    positive, negative = np.array(positive), np.array(negative)
    per_query, values, comparisons = evaluate(orders, plan['weights'], positive, negative)
    saved = arrays(ROOT / 'bootstrap_metrics.npz')
    assert saved['cells'].tolist() == list(CELLS) and saved['metrics'].tolist() == list(METRICS)
    close(saved['values'], values, 'all four-cell metrics')
    seen = set()
    for r in csv_rows(ROOT / 'per_query.csv.gz'):
        cell, q = r['cell'], r['query_id']; key = cell, q
        assert key not in seen; seen.add(key)
        assert r['top_document'] == pools[index[q]][orders[cell][index[q], 0]]
        close([float(r[k]) for k in METRICS], per_query[CELLS.index(cell), index[q]], 'per-query metrics')
    assert len(seen) == 4*3452
    summary = read(ROOT / 'summary.json')
    assert [r['cell'] for r in summary['reports']] == list(CELLS)
    for j, r in enumerate(summary['reports']):
        assert r['queries'] == 3452
        close([r[k+'_pct'] for k in METRICS], 100*values[0, j], 'four cell report')
    expected_ci = {(r['comparison'], r['metric']): r for r in comparisons}
    assert len(summary['comparisons']) == len(expected_ci) == 10
    assert {(r['comparison'], r['metric']) for r in summary['comparisons']} == set(expected_ci)
    for r in summary['comparisons']:
        expected = expected_ci[r['comparison'], r['metric']]
        assert r['primary'] == expected['primary'] and r['draws'] == 1000
        close([r[k] for k in ('difference_pp', 'ci_low_pp', 'ci_high_pp')],
              [expected[k] for k in ('difference_pp', 'ci_low_pp', 'ci_high_pp')], 'paired contrast')
    interval_csv = csv_rows(ROOT / 'paired_bootstrap.csv')
    assert len(interval_csv) == 10
    seen_ci = set()
    for r in interval_csv:
        key = r['comparison'], r['metric']; assert key not in seen_ci; seen_ci.add(key)
        expected = expected_ci[key]
        assert r['primary'] == str(expected['primary']) and int(r['draws']) == 1000
        close([float(r[k]) for k in ('difference_pp', 'ci_low_pp', 'ci_high_pp')],
              [expected[k] for k in ('difference_pp', 'ci_low_pp', 'ci_high_pp')], 'paired interval CSV')
    assert summary['bootstrap']['draws'] == 1000 and summary['bootstrap']['components'] == 3239
    assert summary['bootstrap']['strata'] == 5 and summary['bootstrap']['refit'] is False
    assert summary['bootstrap']['plan_sha256'] == sha(POST / 'prepared/plan.npz')
    assert summary['positive_in_head'] == int(((positive >= 0) & (positive < 20)).sum())
    assert summary['negative_in_head'] == int(((negative >= 0) & (negative < 20)).sum())
    assert summary['cost']['new_model_pairs'] == 138080 and summary['cost']['head_pairs_per_cell'] == 69040
    assert summary['cost']['shared_historical_MiniLM_topic_selection_pairs'] == 345200
    assert summary['predictions_sealed_sha256'] == sha(ROOT / 'predictions_sealed.json')
    transitions = read(ROOT / 'transitions.json')
    for model, before, after in [('MiniLM', 0, 1), ('Qwen', 2, 3)]:
        r = transitions[model]
        source, target = per_query[before, :, 0], per_query[after, :, 0]
        rescued, lost = int((~source & target).sum()), int((source & ~target).sum())
        assert r['topic_to_full'] is True and r['rescued'] == rescued and r['new_losses'] == lost
        assert r['net'] == rescued-lost
        assert r['changed_top1'] == int((orders[CELLS[before]][:, 0] != orders[CELLS[after]][:, 0]).sum())
        expected_transition = {a+'->'+b: int((per_query[before, :, j] & per_query[after, :, k]).sum())
            for j, a in enumerate(('P', 'N', 'U')) for k, b in enumerate(('P', 'N', 'U'))}
        assert r['category_transitions'] == expected_transition
    return {'values': values, 'summary': summary, 'per_query': per_query,
            'positive': positive, 'negative': negative, 'per_query_rows': len(seen), 'transitions': transitions}


def fixtures() -> dict:
    # The interaction is the difference of input effects, equivalently model effects.
    values = np.array([.2, .4, .3, .8])
    close(CONTRASTS['input_effect_interaction'] @ values, (.8-.3)-(.4-.2), 'interaction sign')
    close(CONTRASTS['input_effect_interaction'] @ values,
          CONTRASTS['Qwen_full-MiniLM_full'] @ values-CONTRASTS['Qwen_topic-MiniLM_topic'] @ values, 'interaction identity')
    # Positive outside head must remain a zero rather than disappear from the mean.
    order = np.tile(np.arange(20), (3, 1))
    result = indicators(order, np.array([0, 30, -1]), np.array([1, 0, -1]))
    assert result.tolist() == [[True, False, False, True], [False, True, False, False], [False, False, True, False]]
    head = [f'd{i:02}' for i in range(19, -1, -1)]
    assert rank(np.zeros((1, 20)), [head]).tolist() == [list(range(19, -1, -1))]
    return {'interaction_sign_and_identity': True, 'outside_head_queries_retained': True,
            'raw_score_ties_use_canonical_ID': True}


def provenance(expected_freeze: str) -> dict:
    assert sha(ROOT / 'frozen_manifest.json') == expected_freeze
    manifest = read(ROOT / 'frozen_manifest.json')
    assert 'verify_factorial.py' not in manifest['code']
    assert manifest['primary_contrast'] == 'input_effect_interaction:P1'
    assert manifest['new_neural_pairs'] == 138080 and manifest['timeout_seconds'] == 14400
    assert manifest['new_scores_exist'] is False and manifest['IFIR_used'] is False
    for name, digest in manifest['code'].items():
        assert sha(ROOT / name) == digest, name
    source_count = 0
    for phase in ('input', 'loader', 'scoring'):
        for path, digest in manifest[phase+'_sources'].items():
            assert Path(path).is_absolute() and sha(Path(path)) == digest, (phase, path)
            source_count += 1
    preparation = read(ROOT / 'prepared/summary.json')
    assert preparation['config_sha256'] == sha(ROOT / 'config.py')
    assert preparation['references_opened'] is False and preparation['new_model_calls'] == 0
    assert preparation['queries'] == 3452 and preparation['pairs'] == 69040 and preparation['head_size'] == 20
    assert preparation['old_cells_alignment'] == {'MiniLM_topic': 69040, 'Qwen_full': 69040}
    assert preparation['lengths'] == read(ROOT / 'prepared/token_preflight.json')['token_statistics']
    for path, digest in preparation['sources'].items():
        assert sha(Path(path)) == digest, path
    for path, digest in preparation['outputs'].items():
        assert sha(ROOT / path) == digest, path
    for name, digest in preparation['code'].items():
        assert sha(ROOT / name) == digest, name
    assets = preparation['assets']
    assert assets == manifest['model_assets_checked_before_freeze']
    assert assets['MiniLM']['revision'] == '233902d25c440f23af6f7d6e94d2946bac0bee0a'
    assert assets['Qwen']['revision'] == '22e683669bc0f0bd69640a1354a6d0aebcfeede5'
    asset_files = 0
    for model in ('MiniLM', 'Qwen'):
        for item in assets[model]['files']:
            path = Path(assets[model]['directory']) / item['name']
            assert path.stat().st_size == item['bytes']
            if item['algorithm'] == 'sha256':
                digest = sha(path)
            else:
                assert item['algorithm'] == 'git_blob_sha1'
                h = hashlib.sha1(b'blob '+str(path.stat().st_size).encode()+b'\0')
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1048576), b''):
                        h.update(block)
                digest = h.hexdigest()
            assert digest == item['digest'], (model, item['name'])
            asset_files += 1
    seal = read(ROOT / 'predictions_sealed.json')
    assert seal['frozen_manifest_sha256'] == expected_freeze
    assert seal['scoring_references_opened'] is False and seal['old_scores_exactly_reproduced'] is True
    assert seal['new_neural_pairs'] == 138080
    expected = {'rankings.jsonl.gz', 'four_cell_arrays.npz'} | {
        f'cells/{cell}/scores.jsonl.gz' for cell in CELLS} | {
        f'cells/{cell}/receipt.json' for cell in ('MiniLM_full', 'Qwen_topic')}
    assert set(seal['files']) == expected
    for path, digest in seal['files'].items():
        assert sha(ROOT / path) == digest, path
    finished = read(ROOT / 'production_finished.json')
    assert finished['exit_code'] == 0 and finished['task_retries'] == 0 and finished['new_neural_pairs'] == 138080
    assert finished['source_hashes_before_after'] == 'matched'
    assert (ROOT / 'wrapper_exit_code').read_text().strip() == '0'
    assert read(ROOT / 'resource_preflight.json')['passed'] is True
    assert read(ROOT / 'resource_gate.json')['passed'] is True
    return {'frozen_code_files': len(manifest['code']), 'phase_source_entries': source_count, 'model_asset_files': asset_files,
            'sealed_prediction_files': len(seal['files']), 'production_wall_seconds': finished['wall_seconds']}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--production-complete', action='store_true', required=True)
    parser.add_argument('--freeze-sha256', required=True)
    args = parser.parse_args()
    assert args.production_complete and str(ROOT).startswith('/path/to/remote/lw_experiments/runs/')
    started = time.monotonic()
    source_report = provenance(args.freeze_sha256)
    fixture_report = fixtures()
    source, meta, plan = source_inputs()
    original, selected, reused = old_scores(meta, source)
    scores, records, inventory = cell_scores(meta, original, selected, reused)
    checked_lengths = token_lengths(records)
    orders = {cell: rank(scores[cell], [pool[:20] for pool in meta['ranked_pools']]) for cell in CELLS}
    assert np.array_equal(orders['MiniLM_topic'], np.tile(np.arange(20), (3452, 1)))
    assert read(ROOT / 'predictions_sealed.json')['cell_inventory'] == inventory
    result = verify_outputs(meta, scores, orders, plan)
    assert result['summary']['cost']['cell_inventory'] == inventory
    assert source_report == provenance(args.freeze_sha256)
    certificate = {'status': 'PASS_INDEPENDENT_MODEL_INPUT_FACTORIAL_VERIFICATION',
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'wall_seconds': time.monotonic()-started,
        'frozen_manifest_sha256': args.freeze_sha256, 'verifier_sha256': sha(Path(__file__)),
        'provenance': source_report, 'fixtures': fixture_report, 'numeric': NUMERIC,
        'queries': 3452, 'components': 3239, 'bootstrap_draws': 1000, 'cells': list(CELLS),
        'head_score_rows': 4*69040, 'new_input_token_lengths_checked': checked_lengths,
        'old_input_score_prefix_lengths_exactly_reused': 2*69040,
        'complete_rankings': 4*3452, 'per_query_rows': result['per_query_rows'],
        'point_and_bootstrap_metric_cells': 1001*4*4, 'paired_intervals': 10,
        'cell_inventory': inventory, 'reports': result['summary']['reports'],
        'comparisons': result['summary']['comparisons'], 'transitions': result['transitions'],
        'limits': ['Scores are used and audited, not regenerated with independent GPU inference.',
            'The estimand is conditional on the historical MiniLM-topic head20 and saved body prefixes.',
            'Fixed-prediction paired bootstrap; no refitting or new feature/threshold selection.',
            'The full request changes several aspects of text, not isolated negation.',
            'Previously exposed development data; no claim of fresh external validation.']}
    with (ROOT / 'independent_verification.json').open('x') as stream:
        json.dump(certificate, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(certificate, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
