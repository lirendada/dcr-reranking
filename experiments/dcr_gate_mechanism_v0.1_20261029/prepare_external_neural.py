"""CPU tokenizer-only shared-prefix preparation and cost inventory; no model or references."""
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time
from adapter import sha256, write_json

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT.parent / 'dot_experiment_bundle'
CE_MODEL = BUNDLE / 'research/model_cache/cross-encoder_ms-marco-MiniLM-L6-v2_233902d25c440f23af6f7d6e94d2946bac0bee0a'
PREV = ROOT.parent / 'dcr_stronger_baselines_v0.1_20261028'
DEFAULT = 'Given a web search query, retrieve relevant passages that answer the query'
PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def qwen_text(query, body):
    return PREFIX + f'<Instruct>: {DEFAULT}\n<Query>: {query}\n<Document>: {body}' + SUFFIX


def main():
    assert str(ROOT).startswith('/path/to/remote/lw_experiments/')
    started = time.monotonic()
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoTokenizer
    ce = AutoTokenizer.from_pretrained(CE_MODEL, local_files_only=True, trust_remote_code=False)
    assets = json.loads((PREV / 'model_assets.json').read_text())
    qw = AutoTokenizer.from_pretrained(assets['directory'], local_files_only=True, trust_remote_code=False)
    bundle = ROOT / 'external_inputs'
    # Only model-facing files; never open references.json or raw labelled query JSON.
    queries = {k: {r['query_id']: r['text'] for r in map(json.loads, (bundle / f'queries_{k}.jsonl').open())}
               for k in ('question0', 'RQ_rewrite')}
    corpus = {r['document_id']: r['text'] for r in map(json.loads, (bundle / 'corpus.jsonl').open())}
    candidate = ROOT / 'external_retrieval/candidates.csv'
    expected = json.loads((ROOT / 'external_retrieval/candidates_frozen.json').read_text())['candidate_sha256']
    assert sha256(candidate) == expected
    rows = list(csv.DictReader(candidate.open()))
    assert len(rows) == 345200
    out = ROOT / 'external_prepared'
    out.mkdir(exist_ok=False)
    cache, lengths, stats = {}, {'ce': [], 'qwen': [], 'body_original': []}, Counter()
    max_query, budgets = {}, {}
    for qid in queries['question0']:
        lens = [len(ce(queries[k][qid], add_special_tokens=False)['input_ids']) for k in queries]
        budget = 512 - 3 - max(lens)
        assert budget > 0, ('Query too long: preserve complete query', qid)
        budgets[qid], max_query[qid] = budget, max(lens)
    buffered = []
    def flush(stream):
        if not buffered:
            return
        encoded = qw([qwen_text(queries['RQ_rewrite'][r['query_id']], corpus[r['document_id']][:r['doc_prefix_end_char']])
                      for r in buffered], add_special_tokens=False, truncation=False)['input_ids']
        for row, ids in zip(buffered, encoded):
            row['qwen_input_tokens'] = len(ids)
            assert len(ids) <= 2048, ('Qwen overflow: no silent truncation', row['query_id'], row['document_id'])
            lengths['qwen'].append(len(ids))
            stream.write(json.dumps(row) + '\n')
        buffered.clear()
    with gzip.open(out / 'pairs.jsonl.gz', 'wt') as stream:
        for i, row in enumerate(rows):
            qid, did = row['query_id'], row['document_id']
            body = corpus[did]
            if did not in cache:
                doc = ce(body, add_special_tokens=False, return_offsets_mapping=True)
                cache[did] = (doc['input_ids'], doc['offset_mapping'])
            ids, offsets = cache[did]
            budget = budgets[qid]
            end = len(body) if len(ids) <= budget else offsets[budget - 1][1]
            stats['truncated_pairs'] += int(end < len(body))
            stats['full_body_pairs'] += int(end == len(body))
            # Re-tokenize exact visible substring for the true model pair length.
            ce_length = len(ce(queries['question0'][qid], body[:end], truncation=False)['input_ids'])
            assert ce_length <= 512
            lengths['ce'].append(ce_length)
            lengths['body_original'].append(len(ids))
            buffered.append({'query_id': qid, 'document_id': did, 'doc_prefix_end_char': end,
                             'ce_input_tokens': ce_length, 'shared_document_budget': budget})
            if len(buffered) == 128:
                flush(stream)
            if (i + 1) % 25000 == 0:
                print(json.dumps({'prepared_pairs': i + 1, 'seconds': time.monotonic() - started}), flush=True)
        flush(stream)
    summary = {'pairs': len(rows), 'unique_candidate_documents': len(cache),
        'candidate_sha256': expected, 'prepared_sha256': sha256(out / 'pairs.jsonl.gz'),
        'query_maximum_MiniLM_tokens': max(max_query.values()), 'minimum_body_budget': min(budgets.values()),
        'maximum_body_budget': max(budgets.values()), 'document_coverage': dict(stats),
        'model_loaded': False, 'GPU_used': False, 'seconds': time.monotonic() - started,
        'lengths': {k: {'min': min(v), 'max': max(v), 'mean': sum(v) / len(v),
                        'p50': sorted(v)[len(v)//2], 'p95': sorted(v)[int(len(v)*.95)]} for k, v in lengths.items()},
        'planned_model_pairs': {'MiniLM': 345200, 'Qwen': 345200},
        'prior_Qwen_pairs_per_second': 24.5908240513,
        'projected_Qwen_seconds_at_prior_speed': 345200 / 24.5908240513,
        'projection_not_actual_external_inference': True,
        'query_variant_Qwen': 'full RQ_rewrite in Query, official default Instruct',
        'query_variant_MiniLM': 'question0', 'reference_files_opened': []}
    write_json(out / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
