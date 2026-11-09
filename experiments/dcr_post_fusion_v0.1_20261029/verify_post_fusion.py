"""Independent post-fusion audit, prepared before production is authorized.

No production learning/feature/scoring imports. All numerical execution is
server-only, after a separate production-completion authorization. References
are kept separate from source feature construction and heldout training views.
"""
from __future__ import annotations

import argparse
import ast
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
import unicodedata

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import numpy as np


ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent
GATE = RUNS / 'dcr_gate_mechanism_v0.1_20261029'
OLD = RUNS / 'dot_experiment_bundle/research'
LAMBDAS = (.001, .01, .1, 1.)
ALPHAS = tuple(i / 20 for i in range(21))
EXPECTED_FREEZE = '65e1a77a41ce26652e4b7faa8365e17b7e4425a36035e713605ba702ea091ea7'
METHODS = ('fixed_K20', 'linear', 'Post_G', 'Post_GC')
CONTRASTS = (('Post_GC', 'Post_G'), ('Post_G', 'fixed_K20'),
             ('Post_GC', 'fixed_K20'), ('linear', 'fixed_K20'), ('Post_GC', 'linear'))
NUMERIC = {'comparisons': 0, 'maximum_absolute_difference': {}, 'exact_rank_matrices': 0}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def compare(actual, expected, label: str, kind: str = 'ordinary') -> None:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape, (label, a.shape, b.shape)
    assert np.array_equal(np.isfinite(a), np.isfinite(b)), label
    assert np.array_equal(np.isnan(a), np.isnan(b)), label
    finite = np.isfinite(a) & np.isfinite(b)
    difference = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.
    tolerance = 1e-6 if kind == 'large_cost' else 1e-10
    NUMERIC['comparisons'] += int(finite.sum())
    maxima = NUMERIC['maximum_absolute_difference']
    maxima[kind] = max(maxima.get(kind, 0.), difference)
    assert difference <= tolerance, (label, difference, tolerance)


def jsonl(path: Path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def zscores(values: np.ndarray) -> np.ndarray:
    std = float(values.std(ddof=0))
    return np.zeros(values.shape) if std == 0 else (values - values.mean()) / std


def ordered_head(scores: np.ndarray, ids: list[str]) -> np.ndarray:
    assert scores.shape == (20,) and len(set(ids)) == 20
    assert np.isfinite(scores).all()
    return np.array(sorted(range(20), key=lambda i: (-float(scores[i]), ids[i])))


def independent_doc_features(topic: str, request: str, ids: list[str],
                              mini: np.ndarray, qwen: np.ndarray,
                              bodies: list[str], parser) -> tuple[np.ndarray, dict]:
    """Exactly twenty documents/scores; unselected Qwen values are not accepted."""
    assert mini.shape == qwen.shape == (20,) and len(bodies) == len(ids) == 20
    assert ordered_head(mini, ids).tolist() == list(range(20))
    mz, qz = zscores(mini), zscores(qwen)
    qorder = ordered_head(qwen, ids)
    mrank, qrank = np.empty(20), np.empty(20)
    mrank[np.arange(20)] = np.arange(1, 21) / 20
    qrank[qorder] = np.arange(1, 21) / 20
    msorted, qsorted = np.sort(mz), np.sort(qz)
    mgap, qgap = msorted[-1] - msorted[-2], qsorted[-1] - qsorted[-2]
    topic_set = set(parser.tokenize(topic, remove_stop=True))
    request_set = set(parser.tokenize(request, remove_stop=True))
    negative = set(parser.standalone_parse_instruction(topic, request)['c_neg'])
    combined = topic_set | request_set
    rows = []
    for i, text in enumerate(bodies):
        tokens = parser.tokenize(text, remove_stop=True)
        words, length = set(tokens), len(tokens)
        topic_cov = len(topic_set & words) / len(topic_set) if topic_set else 0.
        request_cov = len(request_set & words) / len(request_set) if request_set else 0.
        union = combined | words
        jaccard = len(combined & words) / len(union) if union else 0.
        hit_count = len(negative & words)
        exclusion_cov = hit_count / len(negative) if negative else 0.
        generic = [mz[i], qz[i], mrank[i], qrank[i], topic_cov, request_cov,
                   jaccard, np.log1p(length), mz[i] * mgap, mz[i] * qgap,
                   qz[i] * mgap, qz[i] * qgap]
        constraint = [float(hit_count > 0), exclusion_cov, np.log1p(hit_count),
                      hit_count / max(length, 1), exclusion_cov * mz[i], exclusion_cov * qz[i]]
        rows.append(generic + constraint)
    result = np.array(rows, dtype=float)
    assert result.shape == (20, 18) and np.isfinite(result).all()
    return result, {'MiniLM_z': mz, 'Qwen_z': qz, 'Qwen_order': qorder,
                    'MiniLM_gap_z': float(mgap), 'Qwen_gap_z': float(qgap),
                    'negative_terms': sorted(negative)}


def reconstruct_source_features() -> dict:
    """Label-free source reconstruction; no old Oracle outcomes or references."""
    parser_path = OLD / 'dcr_public_baselines_v0.1_20261027/frozen_parser.py'
    assert sha(parser_path) == '0bdb82cad89a2e1c852e3bdd4b5f209a17ad9ec4ccdcd24ce6c93fc06e1d0fd9'
    spec = importlib.util.spec_from_file_location('independent_frozen_parser', parser_path)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    pools = {}
    with (GATE / 'external_retrieval/candidates.csv').open() as stream:
        for row in csv.DictReader(stream):
            pools.setdefault(row['query_id'], []).append(row['document_id'])
    mini = {}
    for row in jsonl(GATE / 'external_MiniLM/scores.jsonl.gz'):
        key = row['query_id'], row['document_id']
        assert key not in mini and math.isfinite(row['score'])
        mini[key] = row['score'], row['doc_prefix_end_char']
    qids = sorted(pools)
    assert len(qids) == 3452
    assert len(mini) == 3452 * 100
    ranked, head = {}, {}
    for q in qids:
        assert len(pools[q]) == len(set(pools[q])) == 100
        ranked[q] = sorted(pools[q], key=lambda d: (-mini[q, d][0], d))
        head[q] = set(ranked[q][:20])
    qwen = {}
    for row in jsonl(GATE / 'external_Qwen/scores.jsonl.gz'):
        q, d = row['query_id'], row['document_id']
        if d not in head[q]:
            continue
        # No numeric Qwen field is accessed before the selected-head membership guard.
        assert (q, d) not in qwen
        assert row['doc_prefix_end_char'] == mini[q, d][1]
        assert math.isfinite(row['score']) and row['input_tokens'] > 0
        qwen[q, d] = row['score'], row['input_tokens']
    assert len(qwen) == len(qids) * 20
    documents = set().union(*head.values())
    corpus = {row['document_id']: row['text'] for row in jsonl(GATE / 'external_inputs/corpus.jsonl')
              if row['document_id'] in documents}
    assert set(corpus) == documents
    topics = {row['query_id']: row['text'] for row in jsonl(GATE / 'external_inputs/queries_question0.jsonl')}
    requests = {row['query_id']: row['text'] for row in jsonl(GATE / 'external_inputs/queries_RQ_rewrite.jsonl')}
    x, canonical, raw, token_totals, parsed, all_mini = [], [], [], [], [], []
    for q in qids:
        selected = ranked[q][:20]
        bodies = []
        for d in selected:
            end = mini[q, d][1]
            assert 0 <= end <= len(corpus[d])
            bodies.append(corpus[d][:end])
        m = np.array([mini[q, d][0] for d in selected])
        s = np.array([qwen[q, d][0] for d in selected])
        feature, meta = independent_doc_features(topics[q], requests[q], selected, m, s, bodies, parser)
        x.append(feature)
        canonical.append(sorted(range(20), key=lambda i: selected[i]))
        raw.append(np.column_stack([m, s]))
        token_totals.append(sum(qwen[q, d][1] for d in selected))
        parsed.append(meta['negative_terms'])
        all_mini.append([mini[q, d][0] for d in ranked[q]])
    return {'qids': qids, 'ranked_pools': [ranked[q] for q in qids],
            'X': np.stack(x), 'canonical': np.array(canonical), 'raw_scores': np.stack(raw),
            'Qwen_tokens': np.array(token_totals), 'MiniLM_all_scores': np.array(all_mini),
            'parsed_negative_terms': parsed}


def independent_pair_ridge(features: np.ndarray, positive: np.ndarray,
                           negative: np.ndarray, weights: np.ndarray,
                           regularization: float) -> dict:
    """Only known endpoint pairs with both members in head receive supervision.

    Weights count bootstrap query multiplicities, including pair multiplicities.
    No centering or intercept. Weighted arithmetic here is the mathematical
    reference; production's frozen accumulation order will be audited separately.
    """
    statistics = independent_pair_statistics(features, positive, negative, weights)
    return solve_pair_statistics(statistics, regularization)


def independent_pair_statistics(features: np.ndarray, positive: np.ndarray,
                                negative: np.ndarray, weights: np.ndarray) -> dict:
    eligible = (positive >= 0) & (positive < 20) & (negative >= 0) & (negative < 20) & (weights > 0)
    indices = np.flatnonzero(eligible)
    if not len(indices):
        raise ValueError('No eligible known endpoint pair in this training view.')
    d = features[indices, positive[indices]] - features[indices, negative[indices]]
    w = weights[indices].astype(float)
    pair_count = int(w.sum())
    assert pair_count > 0 and float(pair_count) == w.sum()
    scale = np.sqrt(np.sum(w[:, None] * d * d, axis=0) / pair_count)
    scale[scale == 0] = 1.
    scaled = d / scale
    gram = scaled.T @ (w[:, None] * scaled)
    rhs = scaled.T @ w
    return {'scale': scale, 'gram': gram, 'rhs': rhs, 'pair_count': pair_count,
            'eligible_unique_queries': len(indices)}


def solve_pair_statistics(statistics: dict, regularization: float) -> dict:
    scale = statistics['scale']
    gram = statistics['gram'] + statistics['pair_count'] * regularization * np.eye(len(scale))
    rhs = statistics['rhs']
    coefficients = np.linalg.solve(gram, rhs)
    assert np.isfinite(coefficients).all()
    residual = float(np.max(np.abs(gram @ coefficients - rhs)))
    assert residual < 1e-8 * max(1., float(np.max(np.abs(rhs))))
    return {**statistics, 'W': coefficients, 'normal_equation_residual': residual}


def post_scores(features: np.ndarray, model: dict) -> np.ndarray:
    return (features / model['scale']) @ model['W']


def rank_matrix(scores: np.ndarray, canonical: np.ndarray) -> np.ndarray:
    id_rank = np.empty(canonical.shape, dtype=np.int16)
    np.put_along_axis(id_rank, canonical, np.broadcast_to(np.arange(20), canonical.shape), axis=1)
    return np.lexsort((id_rank, -scores), axis=1)


def selected_success(scores: np.ndarray, canonical: np.ndarray, positive: np.ndarray) -> np.ndarray:
    # Only winner computation is needed during inner selection; sorting all twenty
    # is reserved for artifact verification. Equal maxima take the first ID.
    reordered = np.take_along_axis(scores, canonical, axis=1)
    first = np.argmax(reordered, axis=1)
    winners = canonical[np.arange(len(scores)), first]
    return winners == positive


def choose_alpha(mini_z: np.ndarray, qwen_z: np.ndarray, canonical: np.ndarray,
                  positive: np.ndarray, training_weights: np.ndarray) -> tuple[float, list[int]]:
    counts = []
    active = np.flatnonzero(training_weights > 0)
    for alpha in ALPHAS:
        scores = alpha * mini_z[active] + (1 - alpha) * qwen_z[active]
        hit = selected_success(scores, canonical[active], positive[active])
        counts.append(int(hit @ training_weights[active]))
    # Integer success counts and ascending grid give exact specified tie-breaking.
    return ALPHAS[counts.index(max(counts))], counts


def choose_lambda(features: np.ndarray, canonical: np.ndarray, positive: np.ndarray,
                   negative: np.ndarray, query_fold: np.ndarray, heldout: int,
                   draw_weights: np.ndarray) -> tuple[float, list[int]]:
    totals = [0] * len(LAMBDAS)
    for validation in range(5):
        if validation == heldout:
            continue
        train_weights = draw_weights * ((query_fold != heldout) & (query_fold != validation))
        statistics = independent_pair_statistics(features, positive, negative, train_weights)
        check = np.flatnonzero((query_fold == validation) & (draw_weights > 0))
        for j, regularization in enumerate(LAMBDAS):
            model = solve_pair_statistics(statistics, regularization)
            scores = post_scores(features[check], model)
            hit = selected_success(scores, canonical[check], positive[check])
            totals[j] += int(hit @ draw_weights[check])
    return LAMBDAS[totals.index(max(totals))], totals


def pairwise_credit(positive: str, negative: str, head_ids: list[str],
                     head_scores: np.ndarray, base_ids: list[str],
                     mini_scores: dict[str, float]) -> float:
    """Segment order precedes incomparable score scales; true ties get 0.5."""
    if positive not in mini_scores or negative not in mini_scores:
        return float('nan')
    selected = {document: float(value) for document, value in zip(head_ids, head_scores, strict=True)}
    assert set(selected) == set(base_ids[:20])
    def key(document):
        return (1, selected[document]) if document in selected else (0, mini_scores[document])
    a, b = key(positive), key(negative)
    return 1. if a > b else .5 if a == b else 0.


def stratified_weights(rng: random.Random, fold_components: list[list[str]],
                        component_rows: dict[str, list[int]], n: int) -> np.ndarray:
    weights = np.zeros(n, dtype=np.int32)
    for components in fold_components:
        for _ in components:
            chosen = components[rng.randrange(len(components))]
            weights[component_rows[chosen]] += 1
    return weights


def independent_fixtures() -> dict:
    ids = [f'd{i:06}' for i in range(20)]
    m = np.arange(20., 0., -1)
    q = np.array([0., 1., 9., 9.] + [-1.] * 16)
    assert ordered_head(q, ids).tolist()[:2] == [2, 3]
    assert np.array_equal(ordered_head(0 * zscores(m) + 1 * zscores(q), ids), ordered_head(q, ids))
    assert np.all(zscores(np.ones(20)) == 0)
    # Three documents, only endpoints 0 and 1 supervised; changing doc2 must not affect fit.
    feature = np.array([[[2., 0.], [0., 1.], [100., -999.]]])
    p, n, w = np.array([0]), np.array([1]), np.array([3])
    first = independent_pair_ridge(feature, p, n, w, .1)
    feature[0, 2] = [-1e9, 1e9]
    second = independent_pair_ridge(feature, p, n, w, .1)
    assert np.array_equal(first['W'], second['W']) and np.any(first['W'] != 0)
    expanded = independent_pair_ridge(np.repeat(feature, 3, axis=0), np.zeros(3, dtype=int),
        np.ones(3, dtype=int), np.ones(3, dtype=int), .1)
    assert np.allclose(first['scale'], expanded['scale'], rtol=0, atol=1e-14)
    assert np.allclose(first['W'], expanded['W'], rtol=0, atol=1e-14)
    baseline = ids + ['tail1', 'tail2']
    mini = {document: float(22 - i) for i, document in enumerate(baseline)}
    assert pairwise_credit(ids[2], ids[3], ids, q, baseline, mini) == .5
    assert pairwise_credit(ids[19], 'tail1', ids, q, baseline, mini) == 1.
    return {'alpha0_exact_fixed_ties': True, 'Qwen_gap_requires_score_sort': True,
            'zero_std_zero_z': True, 'unjudged_document_excluded_from_loss': True,
            'no_intercept_nondegenerate': True, 'weighted_repeat_equivalence': True,
            'true_score_tie_half_credit': True, 'head_before_tail_separate_scales': True}


def provenance(expected_freeze: str) -> dict:
    assert len(expected_freeze) == 64 and sha(ROOT / 'frozen_manifest.json') == expected_freeze
    frozen = read(ROOT / 'frozen_manifest.json')
    assert frozen['config']['intercept'] is False and frozen['config']['draws'] == 1000
    assert frozen['runtime']['BLAS_threads'] == 1
    all_sources = {}
    for section in ('feature_sources', 'reference_sources', 'baseline_sources', 'scoring_sources'):
        all_sources.update(frozen[section])
    for name, digest in frozen['code'].items():
        assert sha(ROOT / name) == digest
    for name, digest in all_sources.items():
        assert sha(Path(name)) == digest
    features = read(ROOT / 'features_sealed.json')
    for name, digest in features['files'].items():
        assert sha(ROOT / name) == digest
    assert features['Qwen_rows_retained'] == 3452 * 20
    assert not features['Qwen_tail_scores_used'] and not features['reference_inputs_opened']
    seal = read(ROOT / 'routes_sealed.json')
    assert seal['frozen_manifest_sha256'] == expected_freeze
    assert seal['features_sealed_sha256'] == sha(ROOT / 'features_sealed.json')
    assert seal['fixed_source_check_sha256'] == sha(ROOT / 'fixed_source_check.json')
    assert not seal['scoring_references_opened']
    for name, digest in seal['files'].items():
        assert sha(ROOT / name) == digest
    fixed = read(ROOT / 'fixed_source_check.json')
    assert fixed['queries'] == 3452 and fixed['all_complete_head_and_tail_identical']
    assert fixed['alpha0_identical_to_raw_Qwen_order'] and not fixed['references_opened']
    assert read(ROOT / 'parent_release.json')['frozen_manifest_sha256'] == expected_freeze
    assert (ROOT / 'status').read_text().strip() == 'produced_pending_independent_verification'
    finished = read(ROOT / 'production_finished.json')
    assert finished['exit_code'] == 0 and finished['foundation_model_calls'] == 0 and not finished['GPU_used']
    assert (ROOT / 'features_sealed.json').stat().st_mtime_ns < (ROOT / 'fixed_source_check.json').stat().st_mtime_ns
    assert (ROOT / 'fixed_source_check.json').stat().st_mtime_ns < (ROOT / 'parent_release.json').stat().st_mtime_ns
    assert (ROOT / 'parent_release.json').stat().st_mtime_ns < (ROOT / 'routes_sealed.json').stat().st_mtime_ns
    assert (ROOT / 'routes_sealed.json').stat().st_mtime_ns < (ROOT / 'summary.json').stat().st_mtime_ns
    fit_text = (ROOT / 'fit_fold.py').read_text()
    tree = ast.parse(fit_text)
    opens = [ast.unparse(n.args[0]) for n in ast.walk(tree) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == 'np.load']
    assert set(opens) == {"ROOT / 'features.npz'", "ROOT / 'prepared/plan.npz'", "v['labels_path']"}
    assert not any(word in fit_text for word in ('references.json', 'curves.npz', 'Oracle'))
    prepare = (ROOT / 'prepare_features.py').read_text()
    assert prepare.index('if d not in head[q]:') < prepare.index("row['score'],row['input_tokens']")
    assert not any(word in prepare for word in ('references.json', 'curves.npz', 'Oracle'))
    assert 'reference_sources' not in prepare and 'scoring_sources' not in prepare
    return {'frozen': frozen, 'sources': all_sources}


def verify_features(source: dict) -> dict:
    saved = arrays(ROOT / 'features.npz')
    assert set(saved) == {'X', 'canonical', 'raw_scores', 'MiniLM_all_scores', 'Qwen_tokens'}
    meta = read(ROOT / 'head_and_tail.json')
    assert meta['qids'] == source['qids'] and meta['ranked_pools'] == source['ranked_pools']
    assert len(meta['G_columns']) == 12 and len(meta['C_columns']) == 6
    for name in saved:
        compare(source[name], saved[name], 'source feature ' + name)
    assert np.array_equal(source['canonical'], saved['canonical'])
    assert np.array_equal(source['Qwen_tokens'], saved['Qwen_tokens'])
    parsed = read(ROOT / 'parsed_constraints.json')
    assert len(parsed) == len(source['qids'])
    for i, row in enumerate(parsed):
        assert row['qid'] == source['qids'][i]
        assert sorted(row['c_neg']) == source['parsed_negative_terms'][i]
    fixed = rank_matrix(source['raw_scores'][:, :, 1], source['canonical'])
    alpha0 = rank_matrix(source['X'][:, :, 1], source['canonical'])
    assert np.array_equal(fixed, alpha0)
    # Independently compare the actual complete ranking source, not just its receipt.
    previous = RUNS / 'dcr_budget_v0.1_20261029/ExcluIR/rankings.jsonl.gz'
    index = {q: i for i, q in enumerate(source['qids'])}
    seen = set()
    for row in jsonl(previous):
        if row['method'] != 'topK_20':
            continue
        q = row['qid']
        assert q not in seen
        seen.add(q)
        i = index[q]
        base = source['ranked_pools'][i]
        assert row['orders']['changed'] == [base[j] for j in fixed[i]] + base[20:]
    assert seen == set(source['qids'])
    return {'feature_rows': len(source['qids']), 'documents_per_query': 20,
            'dimensions': 18, 'selected_Qwen_rows': 3452 * 20,
            'complete_fixed_and_alpha0_rankings_checked': len(seen)}


def verify_groups_plan(source: dict, refs: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    previous = read(RUNS / 'dcr_constraint_router_v0.1_20261029/prepared/groups.json')
    qids = source['qids']
    assert qids == previous['qids']
    fold = np.array([previous['query_fold'][q] for q in qids])
    assert np.bincount(fold, minlength=5).tolist() == [691, 691, 690, 690, 690]
    index = {q: i for i, q in enumerate(qids)}
    # Recheck all shared-key isolation from original text/reference inputs.
    topics = {r['query_id']: r['text'] for r in jsonl(GATE / 'external_inputs/queries_question0.jsonl')}
    requests = {r['query_id']: r['text'] for r in jsonl(GATE / 'external_inputs/queries_RQ_rewrite.jsonl')}
    normalized = lambda value: ' '.join(unicodedata.normalize('NFKC', value).casefold().split())
    seen, first, graph = {}, {}, {q: set() for q in qids}
    for i, q in enumerate(qids):
        keys = [('endpoint', refs[q]['positive_id']), ('endpoint', refs[q]['negative_id']),
                ('topic', normalized(topics[q])), ('request', normalized(requests[q]))]
        for key in keys:
            if key in seen:
                assert seen[key] == fold[i], 'shared endpoint/text crosses folds'
            seen[key] = fold[i]
            if key in first:
                other = first[key]
                graph[q].add(other)
                graph[other].add(q)
            else:
                first[key] = q
    visited, rebuilt_components = set(), {}
    for q in qids:
        if q in visited:
            continue
        stack, members_q = [q], []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            members_q.append(current)
            stack.extend(graph[current] - visited)
        members_q.sort()
        rebuilt_components[members_q[0]] = members_q
    assert previous['components'] == rebuilt_components
    ordered = sorted(rebuilt_components, key=lambda c: (-len(rebuilt_components[c]),
        hashlib.sha256(('dcr-router-v1|' + c).encode()).hexdigest(), c))
    loads, assignment = [0] * 5, {}
    for c in ordered:
        selected = min(range(5), key=lambda f: (loads[f], f))
        assignment[c] = selected
        loads[selected] += len(rebuilt_components[c])
    assert previous['component_fold'] == assignment
    ids = [previous['fold_component_IDs'][str(f)] for f in range(5)]
    members = {c: [index[q] for q in queries] for c, queries in previous['components'].items()}
    assert len(members) == 3239 and max(map(len, members.values())) == 3
    assert all(cells == sorted(cells) for cells in ids)
    plan = arrays(ROOT / 'prepared/plan.npz')
    assert np.array_equal(plan['fold'], fold)
    rng = random.Random(20261029)
    for draw in range(1000):
        expected = stratified_weights(rng, ids, members, len(qids))
        assert np.array_equal(expected, plan['weights'][draw])
    return fold, plan['weights'], {'components': len(members), 'maximum_component_queries': 3,
        'fold_queries': np.bincount(fold, minlength=5).tolist(),
        'batch_queries_min': int(plan['weights'].sum(axis=1).min()),
        'batch_queries_max': int(plan['weights'].sum(axis=1).max()),
        'endpoint_topic_request_cross_fold_overlap': 0,
        'previous_groups_sha256': sha(RUNS / 'dcr_constraint_router_v0.1_20261029/prepared/groups.json')}


def verify_views(source: dict, refs: dict, fold: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    n = len(source['qids'])
    p = np.array([docs.index(refs[q]['positive_id']) if refs[q]['positive_id'] in docs else -1
                  for q, docs in zip(source['qids'], source['ranked_pools'], strict=True)])
    neg = np.array([docs.index(refs[q]['negative_id']) if refs[q]['negative_id'] in docs else -1
                    for q, docs in zip(source['qids'], source['ranked_pools'], strict=True)])
    phead, nhead = np.where(p < 20, p, -1), np.where(neg < 20, neg, -1)
    workers = {}
    for heldout in range(5):
        view = read(ROOT / 'prepared' / f'view_{heldout}.json')
        train, test = np.flatnonzero(fold != heldout), np.flatnonzero(fold == heldout)
        label_file = ROOT / 'prepared' / f'labels_{heldout}.npz'
        label = arrays(label_file)
        assert label['train_indices'].tolist() == train.tolist()
        assert label['positive'].tolist() == phead[train].tolist()
        assert label['negative'].tolist() == nhead[train].tolist()
        assert view['test_indices'] == test.tolist() and view['labels_path'] == str(label_file)
        allowed = {str(ROOT / name) for name in ('features.npz', 'head_and_tail.json', 'features_sealed.json',
                   'prepared/plan.npz', f'prepared/labels_{heldout}.npz')}
        assert set(view['allowed_inputs']) == allowed
        for path, digest in view['allowed_inputs'].items():
            assert sha(Path(path)) == digest
        assert view['training_queries'] == len(train)
        assert view['training_pair_queries'] == int(((phead[train] >= 0) & (nhead[train] >= 0)).sum())
        assert view['training_positive_in_head'] == int((phead[train] >= 0).sum())
        assert view['training_negative_in_head'] == int((nhead[train] >= 0).sum())
        saved = arrays(ROOT / 'fitted' / f'fold_{heldout}.npz')
        receipt = read(ROOT / 'fitted' / f'fold_{heldout}.json')
        assert not receipt['heldout_labels_read']
        assert receipt['file_sha256'] == sha(ROOT / 'fitted' / f'fold_{heldout}.npz')
        assert receipt['training_labels_file_sha256'] == sha(label_file)
        assert saved['test_indices'].tolist() == test.tolist()
        # Supply only each actual training-label view to independent fitting.
        pos_view, neg_view = np.full(n, -2), np.full(n, -2)
        pos_view[train], neg_view[train] = label['positive'], label['negative']
        workers[heldout] = {'saved': saved, 'positive': pos_view, 'negative': neg_view, 'test': test}
    receipt = read(ROOT / 'prepared/receipt.json')
    assert receipt['qualified_pair_queries'] == int(((phead >= 0) & (nhead >= 0)).sum()) == 2901
    assert receipt['all_queries_evaluated_later'] and not receipt['unjudged_candidates_given_negative_labels']
    return workers, p, neg


def verify_models(source: dict, workers: dict, fold: np.ndarray,
                  plan: np.ndarray) -> tuple[dict, dict, dict]:
    x, canonical = source['X'], source['canonical']
    alpha_orders = np.array([rank_matrix(alpha * x[:, :, 0] + (1 - alpha) * x[:, :, 1], canonical)
                             for alpha in ALPHAS], dtype=np.uint8)
    point_order = {'fixed_K20': rank_matrix(source['raw_scores'][:, :, 1], canonical)}
    point_score = {'fixed_K20': source['raw_scores'][:, :, 1]}
    bootstrap_top = {name: np.empty(plan.shape, dtype=np.uint8) for name in METHODS[1:]}
    for name in METHODS[1:]:
        point_order[name] = np.empty((len(x), 20), dtype=np.uint8)
        point_score[name] = np.empty((len(x), 20))
    point_parameters = {}
    for outer in range(5):
        worker = workers[outer]
        saved, positive, negative, test = (worker[key] for key in ('saved', 'positive', 'negative', 'test'))
        # Unlabeled alpha winners are cached; only this view's training endpoint
        # positions are ever compared with those winners during parameter choice.
        alpha_hit = alpha_orders[:, :, 0] == positive[None, :]
        point_parameters[str(outer)] = {}
        for iteration in range(1001):
            weights = np.ones(len(x), dtype=np.int32) if iteration == 0 else plan[iteration - 1].astype(np.int32)
            training = weights * (fold != outer)
            alpha_counts = np.sum(alpha_hit * training[None, :], axis=1)
            alpha_index = int(np.argmax(alpha_counts))
            alpha = ALPHAS[alpha_index]
            assert alpha == float(saved['alpha'][iteration])
            linear_order = alpha_orders[alpha_index, test]
            assert np.array_equal(linear_order, saved['linear_order'][iteration]), ('linear order', outer, iteration)
            NUMERIC['exact_rank_matrices'] += 1
            if iteration == 0:
                point_order['linear'][test] = linear_order
                point_score['linear'][test] = alpha * x[test, :, 0] + (1 - alpha) * x[test, :, 1]
                point_parameters[str(outer)]['point_alpha'] = alpha
            else:
                bootstrap_top['linear'][iteration - 1, test] = linear_order[:, 0]
            for arm, name, dimension in [('G', 'Post_G', 12), ('GC', 'Post_GC', 18)]:
                feature = x[:, :, :dimension]
                lam, counts = choose_lambda(feature, canonical, positive, negative, fold, outer, weights)
                assert lam == float(saved[arm + '_lambda'][iteration]), ('lambda choice', outer, iteration, arm)
                assert counts == saved[arm + '_inner_correct_counts'][iteration].tolist()
                model = independent_pair_ridge(feature, positive, negative, training, lam)
                assert model['pair_count'] == int(saved[arm + '_pair_instances'][iteration])
                compare(model['scale'], saved[arm + '_scale'][iteration], 'outer pair RMS')
                compare(model['W'], saved[arm + '_coef'][iteration], 'outer pair coefficient')
                scores = post_scores(feature[test], model)
                order = rank_matrix(scores, canonical[test])
                assert np.array_equal(order, saved[arm + '_order'][iteration]), ('Post order', outer, iteration, arm)
                NUMERIC['exact_rank_matrices'] += 1
                if iteration == 0:
                    compare(scores, saved[arm + '_point_scores'], 'point Post score')
                    point_order[name][test], point_score[name][test] = order, scores
                    point_parameters[str(outer)]['point_' + arm + '_lambda'] = lam
                else:
                    bootstrap_top[name][iteration - 1, test] = order[:, 0]
            if iteration % 100 == 0:
                print(json.dumps({'independent_outer_fold': outer, 'iteration': iteration, 'stage': 'reselect_refit_rank'}), flush=True)
    return point_order, point_score, {'bootstrap_top': bootstrap_top, 'point_parameters': point_parameters}


def verify_point(source: dict, refs: dict, positive: np.ndarray, negative: np.ndarray,
                  orders: dict, scores: dict, parameters: dict) -> dict:
    qids, pools = source['qids'], source['ranked_pools']
    n = len(qids)
    summary = read(ROOT / 'summary.json')
    assert summary['point_selected_parameters'] == parameters
    own = {}
    point = {}
    for name in METHODS:
        order = orders[name]
        assert np.all(np.sort(order, axis=1) == np.arange(20))
        winners = order[:, 0]
        p1, n1 = (winners == positive).astype(int), (winners == negative).astype(int)
        ranks = np.empty((n, 100), dtype=np.int16)
        for i in range(n):
            all_order = np.concatenate([order[i], np.arange(20, 100)])
            assert set(all_order.tolist()) == set(range(100))
            ranks[i, all_order] = np.arange(1, 101)
        rows = []
        for i, q in enumerate(qids):
            mini = dict(zip(pools[i], source['MiniLM_all_scores'][i], strict=True))
            credit = pairwise_credit(refs[q]['positive_id'], refs[q]['negative_id'], pools[i][:20], scores[name][i], pools[i], mini)
            rows.append({'positive_recall@1': int(p1[i]), 'negative_entry@1': int(n1[i]),
                         'positive_recall@10': int(positive[i] >= 0 and ranks[i, positive[i]] <= 10),
                         'pairwise_credit': credit,
                         'positive_rank': int(ranks[i, positive[i]]) if positive[i] >= 0 else 0,
                         'negative_rank': int(ranks[i, negative[i]]) if negative[i] >= 0 else 0})
        own[name] = rows
        point[name] = np.column_stack([p1, n1])
    auxiliary = {'194_MiniLM_only': [], '1006_fixed_only': [], 'both_correct': [], 'both_wrong': []}
    fixed_p1 = point['fixed_K20'][:, 0]
    for i, q in enumerate(qids):
        mini, fixed = positive[i] == 0, bool(fixed_p1[i])
        category = 'both_correct' if mini and fixed else '194_MiniLM_only' if mini else '1006_fixed_only' if fixed else 'both_wrong'
        auxiliary[category].append(q)
    assert len(auxiliary['194_MiniLM_only']) == 194 and len(auxiliary['1006_fixed_only']) == 1006
    auxiliary_file = read(ROOT / 'auxiliary_groups.json')
    assert auxiliary_file['groups'] == auxiliary and auxiliary_file['scoring_only']
    old_budget = RUNS / 'dcr_budget_v0.1_20261029/ExcluIR/evaluation'
    qposition = {q: i for i, q in enumerate(qids)}
    for name in ('MiniLM', 'topK_20'):
        path = old_budget / name / 'per_query.csv'
        assert auxiliary_file['source_hashes'][name] == sha(path)
        with path.open() as stream:
            for row in csv.DictReader(stream):
                i = qposition[row['query_id']]
                expected = positive[i] == 0 if name == 'MiniLM' else fixed_p1[i]
                assert float(row['positive_recall@1']) == float(expected)
    reports = {row['method']: row for row in summary['reports']}
    assert set(reports) == set(METHODS)
    for name in METHODS:
        report, rows, p = reports[name], own[name], point[name][:, 0]
        assert report['queries'] == n and report['Qwen_pairs'] == n * 20
        assert report['Qwen_input_tokens'] == int(source['Qwen_tokens'].sum())
        compare(100 * float(p.mean()), report['P1_pct'], 'P1 point')
        compare(100 * float(point[name][:, 1].mean()), report['N1_pct'], 'N1 point')
        compare(100 * sum(row['positive_recall@10'] for row in rows) / n, report['positive_recall@10_pct'], 'P10 point')
        finite = [row['pairwise_credit'] for row in rows if math.isfinite(row['pairwise_credit'])]
        compare(100 * float(np.mean(finite)), report['pairwise_credit_pct'], 'true score-tie pairwise')
        assert report['pairwise_covered'] == len(finite)
        rescued, losses = int(((p == 1) & (fixed_p1 == 0)).sum()), int(((p == 0) & (fixed_p1 == 1)).sum())
        assert report['rescued_vs_fixed'] == rescued and report['new_losses_vs_fixed'] == losses
        assert report['net_rescued_vs_fixed'] == rescued - losses
        for category, queries in auxiliary.items():
            compare(100 * float(p[[qposition[q] for q in queries]].mean()), report['auxiliary_P1_pct'][category], 'auxiliary P1')
    transition = summary['primary_GC_vs_G_transition']
    assert transition['rescued'] == int(((point['Post_GC'][:, 0] == 1) & (point['Post_G'][:, 0] == 0)).sum())
    assert transition['new_losses'] == int(((point['Post_GC'][:, 0] == 0) & (point['Post_G'][:, 0] == 1)).sum())
    for key, value in [('positive_in_pool', int((positive >= 0).sum())), ('negative_in_pool', int((negative >= 0).sum())),
                       ('positive_in_head', int(((positive >= 0) & (positive < 20)).sum())),
                       ('negative_in_head', int(((negative >= 0) & (negative < 20)).sum())),
                       ('both_in_head', int(((positive >= 0) & (positive < 20) & (negative >= 0) & (negative < 20)).sum()))]:
        assert summary[key] == value
    seen = set()
    with gzip.open(ROOT / 'per_query.csv.gz', 'rt') as stream:
        for row in csv.DictReader(stream):
            method, qid = row['method'], row['qid']
            assert (method, qid) not in seen
            seen.add((method, qid))
            expected = own[method][qposition[qid]]
            for key, value in expected.items():
                compare(float(row[key]) if row[key] else np.nan, value, 'per-query ' + key)
    assert len(seen) == 4 * n
    return {'point': point, 'reports': 4, 'per_query_rows': len(seen),
            'known_pairwise_coverage': reports['fixed_K20']['pairwise_covered'],
            'Qwen_pairs_each': n * 20, 'Qwen_input_tokens_each': int(source['Qwen_tokens'].sum()),
            'qualified_pair_queries': summary['both_in_head'], 'auxiliary_group_sizes': {k: len(v) for k, v in auxiliary.items()}}


def percentile(values: np.ndarray, probability: float) -> float:
    ordered = np.sort(values)
    location = (len(ordered) - 1) * probability
    lo = math.floor(location)
    hi = min(lo + 1, len(ordered) - 1)
    return float(ordered[lo] + (location - lo) * (ordered[hi] - ordered[lo]))


def verify_bootstrap(plan: np.ndarray, positive: np.ndarray, negative: np.ndarray,
                      point: dict, bootstrap_top: dict) -> dict:
    samples = np.empty((1000, 5, 2))
    for draw, weights in enumerate(plan):
        denominator = int(weights.sum())
        means = {'fixed_K20': np.sum(weights[:, None] * point['fixed_K20'], axis=0) / denominator}
        for name in METHODS[1:]:
            winners = bootstrap_top[name][draw]
            values = np.column_stack([winners == positive, winners == negative])
            means[name] = np.sum(weights[:, None] * values, axis=0) / denominator
        for j, (left, right) in enumerate(CONTRASTS):
            samples[draw, j] = 100 * (means[left] - means[right])
    summary = read(ROOT / 'summary.json')
    rows = list(csv.DictReader((ROOT / 'paired_bootstrap.csv').open()))
    assert len(rows) == len(summary['paired_bootstrap']) == 10
    for row, mirror in zip(rows, summary['paired_bootstrap'], strict=True):
        contrast = CONTRASTS.index((row['method'], row['reference']))
        metric = ('P1', 'N1').index(row['metric'])
        value = samples[:, contrast, metric]
        compare(percentile(value, .025), float(row['ci_low_pp']), 'complete-refit percentile low')
        compare(percentile(value, .975), float(row['ci_high_pp']), 'complete-refit percentile high')
        compare(float(row['ci_low_pp']), mirror['ci_low_pp'], 'CI low mirror')
        compare(float(row['ci_high_pp']), mirror['ci_high_pp'], 'CI high mirror')
        difference = 100 * float(point[row['method']][:, metric].mean() - point[row['reference']][:, metric].mean())
        compare(difference, float(row['difference_pp']), 'CI point difference')
        primary = row['method'] == 'Post_GC' and row['reference'] == 'Post_G' and row['metric'] == 'P1'
        assert (row['primary'] == 'True') == mirror['primary'] == primary
        assert int(row['draws']) == mirror['draws'] == 1000
    assert summary['routes_sealed_sha256'] == sha(ROOT / 'routes_sealed.json')
    return {'draws': 1000, 'CI_rows': 10,
            'fixed_folds_and_component_multiplicities': True,
            'reselected_alpha_lambda_and_refitted_each_draw': True}


def full_verification(expected_freeze: str) -> dict:
    started = time.monotonic()
    proof = provenance(expected_freeze)
    fixtures = independent_fixtures()
    source = reconstruct_source_features()
    feature_check = verify_features(source)
    print('Independent head-only source features and complete fixed/alpha0 rankings passed.', flush=True)
    refs = read(GATE / 'external_inputs/references.json')
    fold, plan, grouping = verify_groups_plan(source, refs)
    workers, positive, negative = verify_views(source, refs, fold)
    orders, scores, fitted = verify_models(source, workers, fold, plan)
    point = verify_point(source, refs, positive, negative, orders, scores, fitted['point_parameters'])
    bootstrap = verify_bootstrap(plan, positive, negative, point['point'], fitted['bootstrap_top'])
    for path, digest in proof['sources'].items():
        assert sha(Path(path)) == digest
    for name, digest in proof['frozen']['code'].items():
        assert sha(ROOT / name) == digest
    return {'status': 'PASS_INDEPENDENT_POST_FUSION_VERIFICATION',
            'verified_utc': datetime.now(timezone.utc).isoformat(),
            'verifier_sha256': sha(Path(__file__)), 'frozen_manifest_sha256': expected_freeze,
            'source_hashes_before_after': proof['sources'], 'fixtures': fixtures,
            'features': feature_check, 'grouping': grouping,
            'point_results': {key: value for key, value in point.items() if key != 'point'},
            'bootstrap': bootstrap, 'numeric': NUMERIC,
            'recomputed_outer_pair_models': 10010, 'recomputed_inner_candidate_solves': 160160,
            'checked_outer_alpha_choices': 5005, 'foundation_model_calls': 0, 'GPU_used': False,
            'BLAS_threads': 1, 'verification_seconds': time.monotonic() - started,
            'role': 'independent AI technical reconstruction, not human expert or unseen evaluation',
            'arithmetic_boundary': 'Strict hashes/indices/parameters and all complete head permutations; float arrays atol1e-10, large costs atol1e-6; no decision epsilon.',
            'supervision_boundary': 'Only known positive/negative endpoint pairs both inside head20 enter pair loss; inner P1 uses every validation query, unjudged docs never assigned negative labels.',
            'score_access_boundary': 'Only selected MiniLM top20 Qwen values enter features, normalization, training or ranking; head ranks plus untouched MiniLM tail; source-file hashing may read full bytes.',
            'statistical_boundary': 'Previously seen ExcluIR development; fixed-fold component bootstrap includes parameter reselection/refitting, excludes developer/split selection uncertainty. Sparse labels prevent full precision or causal claims.'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixtures-only', action='store_true')
    parser.add_argument('--expected-freeze', default=EXPECTED_FREEZE)
    args = parser.parse_args()
    assert str(ROOT).startswith('/path/to/remote/lw_experiments/'), 'No local numerical experiments.'
    if args.fixtures_only:
        started = time.monotonic()
        print(json.dumps({'fixtures': independent_fixtures(), 'wall_seconds': time.monotonic() - started}))
        return
    assert not (ROOT / 'independent_verification.json').exists(), 'Never overwrite an earlier certificate.'
    certificate = full_verification(args.expected_freeze)
    with (ROOT / 'independent_verification.json').open('x') as stream:
        json.dump(certificate, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': certificate['status'], 'wall_seconds': certificate['verification_seconds'], 'numeric': NUMERIC}), flush=True)


if __name__ == '__main__':
    main()
