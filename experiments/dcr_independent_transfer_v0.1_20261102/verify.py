"""Separate reconstruction: no imports of production ranking/metric/bootstrap functions."""
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent
METHODS = ('MiniLM_topic', 'MiniLM_full', 'Qwen_topic', 'Qwen_full', 'G12', 'G_same12', 'QL5')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def norm(values):
    values = np.asarray(values, dtype=np.float64)
    deviation = np.sqrt(np.mean((values - values.mean()) ** 2))
    return (values - values.mean()) / deviation if deviation else values * 0


def percentile(values, probability):
    data = sorted(map(float, values))
    position = probability * (len(data) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return data[lo] + (data[hi] - data[lo]) * (position - lo)


def main():
    started = time.monotonic()
    assert str(ROOT).startswith('/path/to/remote/lw_experiments/runs/')
    freeze = read(ROOT / 'frozen_manifest.json')
    for name, expected in {**freeze['code'], **freeze['inputs']}.items():
        assert digest(ROOT / name) == expected, name
    for name, expected in read(ROOT / 'source_contract.json').items():
        assert digest(Path(name)) == expected, name
    seal = read(ROOT / 'predictions_sealed.json')
    for name, expected in seal['files'].items():
        assert digest(ROOT / name) == expected, name
    meta = read(ROOT / 'ranking_metadata.json')
    queries = [json.loads(line) for line in (ROOT / 'model_queries.jsonl').read_text().splitlines()]
    assert len(meta) == len(queries) == 43
    assert [q['query_id'] for q in queries] == [q['query_id'] for q in meta]
    body = {r['document_id']: r['text'] for r in [json.loads(line) for line in (ROOT / 'candidate_bodies.jsonl').read_text().splitlines()]}
    # Verify the selected bodies against the pinned full corpus, without rerunning BM25.
    remaining = set(body)
    with (ROOT / 'raw/cds/cds-corpus.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            did = str(row['_id'])
            if did in remaining:
                assert row['text'] == body[did]
                remaining.remove(did)
    assert not remaining
    pairs = {(r['query_id'], r['document_id']): r for r in rows(ROOT / 'prepared_pairs.jsonl.gz')}
    scores, raw_scores = {}, {}
    for method in METHODS[:4]:
        source = rows(ROOT / 'cells' / method / 'scores.jsonl.gz')
        table = {}
        for row in source:
            qid, did = row['query_id'], row['document_id']
            assert did not in table.setdefault(qid, {})
            table[qid][did] = row['score']
            assert row['input_tokens'] == pairs[qid, did]['token_lengths'][method]
            if method.startswith('Qwen'):
                assert row['score'] == row['yes_logit'] - row['no_logit']
        scores[method], raw_scores[method] = table, source
    model = read(RUNS / 'dcr_post_g_external_v0.1_20261029/source_model.json')
    additional = read(RUNS / 'dcr_input_matched_fusion_v0.1_20261029/transfer_models.json')['models']
    parser_path = RUNS / 'dot_experiment_bundle/research/dcr_public_baselines_v0.1_20261027/frozen_parser.py'
    spec = importlib.util.spec_from_file_location('verification_frozen_tokenizer', parser_path)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    observed_orders = {(r['query_id'], r['method']): r['order'] for r in rows(ROOT / 'rankings.jsonl.gz')}
    saved = np.load(ROOT / 'features.npz', allow_pickle=False)
    rank_checks = 0
    for qi, query in enumerate(queries):
        qid = query['query_id']
        base = sorted(scores['MiniLM_topic'][qid], key=lambda did: (-scores['MiniLM_topic'][qid][did], did))
        head = base[:20]
        assert base == meta[qi]['base'] and head == meta[qi]['head']
        for method in METHODS[1:4]:
            assert set(scores[method][qid]) == set(head)
        t, r = set(parser.tokenize(query['topic'], remove_stop=True)), set(parser.tokenize(query['request'], remove_stop=True))
        zt = norm([scores['MiniLM_topic'][qid][did] for did in head])
        zr = norm([scores['MiniLM_full'][qid][did] for did in head])
        zq = norm([scores['Qwen_full'][qid][did] for did in head])
        gap_t, gap_r, gap_q = [np.sort(x)[-1] - np.sort(x)[-2] for x in [zt, zr, zq]]
        rank_q = {did: i + 1 for i, did in enumerate(sorted(head, key=lambda did: (-scores['Qwen_full'][qid][did], did)))}
        rank_r = {did: i + 1 for i, did in enumerate(sorted(head, key=lambda did: (-scores['MiniLM_full'][qid][did], did)))}
        original, matched = [], []
        for di, did in enumerate(head):
            pair = pairs[qid, did]
            prefix = body[did][:pair['doc_prefix_end_char']]
            assert hashlib.sha256(prefix.encode()).hexdigest() == pair['body_prefix_sha256']
            tokens = parser.tokenize(prefix, remove_stop=True)
            doc = set(tokens)
            text = [len(t & doc) / len(t) if t else 0., len(r & doc) / len(r) if r else 0.,
                    len((t | r) & doc) / len(t | r | doc) if t | r | doc else 0., math.log1p(len(tokens))]
            original.append([zt[di], zq[di], (di + 1) / 20, rank_q[did] / 20, *text,
                             zt[di] * gap_t, zt[di] * gap_q, zq[di] * gap_t, zq[di] * gap_q])
            matched.append([zr[di], zq[di], rank_r[did] / 20, rank_q[did] / 20, *text,
                            zr[di] * gap_r, zr[di] * gap_q, zq[di] * gap_r, zq[di] * gap_q])
        original, matched = np.asarray(original), np.asarray(matched)
        ql = original[:, [1, 4, 5, 6, 7]]
        for name, features in [('G12', original), ('G_same12', matched), ('QL5', ql)]:
            assert np.allclose(features, saved[name][qi], atol=1e-12, rtol=0), (qid, name)
        values = {}
        for name, features, parameter in [('G12', original, model), ('G_same12', matched, additional['G_same12']), ('QL5', ql, additional['QL5'])]:
            values[name] = np.sum(features / np.asarray(parameter['scale']) * np.asarray(parameter['coef']), axis=1)
        for method in METHODS:
            if method == 'MiniLM_topic':
                expected = base
            elif method in METHODS[:4]:
                expected = sorted(head, key=lambda did: (-scores[method][qid][did], did)) + base[20:]
            else:
                expected = [head[j] for j in sorted(range(20), key=lambda j: (-values[method][j], head[j]))] + base[20:]
            assert expected == observed_orders[qid, method], (qid, method)
            rank_checks += 1
    # Independent token lengths for every prepared pair: no model is loaded.
    from transformers import AutoTokenizer
    ce = AutoTokenizer.from_pretrained(str(RUNS / 'dot_experiment_bundle/research/model_cache/cross-encoder_ms-marco-MiniLM-L6-v2_233902d25c440f23af6f7d6e94d2946bac0bee0a'), local_files_only=True)
    prefix_checks = 0
    for query in queries:
        qid = query['query_id']
        budget = 509 - max(len(ce.encode(query[k], add_special_tokens=False)) for k in ['topic', 'request'])
        assert budget > 0
        for (pair_qid, did), pair in pairs.items():
            if pair_qid != qid:
                continue
            encoded = ce(body[did], add_special_tokens=False, return_offsets_mapping=True, truncation=False)
            stop = len(body[did]) if len(encoded['input_ids']) <= budget else encoded['offset_mapping'][budget - 1][1]
            assert stop == pair['doc_prefix_end_char']
            for method in METHODS[:2]:
                field = 'topic' if method.endswith('_topic') else 'request'
                assert len(ce(query[field], body[did][:stop], truncation=False)['input_ids']) == pair['token_lengths'][method]
            prefix_checks += 1
    references = {}
    reference_file = ROOT / 'references/cds-qrel.tsv'
    assert digest(reference_file) == read(ROOT / 'reference_receipt.json')['sha256']
    with reference_file.open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            references.setdefault(row['query-id'], {})[row['corpus-id']] = float(row['score'])
    metric_values = np.zeros((43, 7, 4))
    for qi, query in enumerate(queries):
        ref = references.get(query['query_id'], {})
        positive = {did for did, grade in ref.items() if grade > 0}
        ideal_values = sorted([grade for grade in ref.values() if grade > 0], reverse=True)[:20]
        ideal = sum(grade / math.log2(i + 2) for i, grade in enumerate(ideal_values))
        for mi, method in enumerate(METHODS):
            order = observed_orders[query['query_id'], method]
            ranks = [i + 1 for i, did in enumerate(order[:20]) if did in positive]
            dcg = sum(ref.get(did, 0) / math.log2(i + 2) for i, did in enumerate(order[:20]))
            metric_values[qi, mi] = [int(order[0] in positive), dcg / ideal if ideal else 0.,
                1 / min(ranks) if ranks else 0., sum(did in positive for did in order[:10]) / len(positive) if positive else 0.]
    arrays = np.load(ROOT / 'bootstrap_arrays.npz', allow_pickle=False)
    assert np.allclose(metric_values, arrays['values'], atol=1e-12, rtol=0)
    base_order = sorted(range(43), key=lambda i: queries[i]['base_id'])
    assert len({q['base_id'] for q in queries}) == 43
    generator = random.Random(20261102)
    counts = np.zeros((10000, 43), dtype=np.int32)
    for draw in range(10000):
        sampled = [generator.randrange(43) for _ in range(43)]
        for index in sampled:
            counts[draw, index] += 1
    assert np.array_equal(counts, arrays['topic_counts'])
    draws = (counts @ metric_values[base_order].reshape(43, -1) / 43).reshape(10000, 7, 4)
    assert np.allclose(draws, arrays['draws'], atol=1e-12, rtol=0)
    summary = read(ROOT / 'summary.json')
    assert sum(c['primary'] for c in summary['contrasts']) == 1
    for contrast in summary['contrasts']:
        if contrast['comparison'] == 'model_by_input_interaction':
            point = metric_values[:, 3, 0] - metric_values[:, 2, 0] - metric_values[:, 1, 0] + metric_values[:, 0, 0]
            distribution = draws[:, 3, 0] - draws[:, 2, 0] - draws[:, 1, 0] + draws[:, 0, 0]
        else:
            a, b = contrast['comparison'].split('-')
            ai, bi = METHODS.index(a), METHODS.index(b)
            point = metric_values[:, ai, 0] - metric_values[:, bi, 0]
            distribution = draws[:, ai, 0] - draws[:, bi, 0]
        for actual, expected in [(point.mean() * 100, contrast['difference_pp']),
            (percentile(distribution * 100, .025), contrast['ci95_low_pp']),
            (percentile(distribution * 100, .975), contrast['ci95_high_pp'])]:
            assert abs(actual - expected) < 1e-10, contrast['comparison']
    result = {'status': 'PASS', 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'rankings_checked': rank_checks, 'all_prepared_prefixes_and_MiniLM_lengths_checked': prefix_checks,
        'metric_fields_checked': 43 * 7 * 4, 'draws_checked': 10000, 'contrasts_checked': 10,
        'production_functions_imported': False, 'model_weights_loaded': False,
        'same_agent_authored_both_implementations': True,
        'scope': 'Full feature/ranking/metric/bootstrap reconstruction, candidate bodies against corpus and all MiniLM prefixes/lengths. Frozen tokenizer reused; no BM25 or neural forward recomputation. Qwen logits and recorded lengths checked, not independently retokenized.',
        'seconds': time.monotonic() - started, 'code_sha256': digest(Path(__file__)),
        'prediction_seal_sha256': digest(ROOT / 'predictions_sealed.json')}
    with (ROOT / 'verification.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
