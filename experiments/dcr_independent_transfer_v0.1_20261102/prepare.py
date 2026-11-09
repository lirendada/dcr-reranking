"""Exact full-corpus topic retrieval and shared-prefix/token preparation."""
from collections import Counter
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import os
import resource
import sys
import time

from config import (ROOT, GATE, STRONG_QWEN, CE_MODEL, CORPUS, N, CELLS, rows,
                    documents, read, write, sha, load_module, check_sources, qwen_text_function)


def main() -> None:
    start = time.monotonic()
    assert str(ROOT).startswith('/path/to/remote/lw_experiments/runs/')
    check_sources()
    assert not (ROOT / 'preparation.json').exists()
    assert read(ROOT / 'preflight_summary.json')['provisional_selected'] == 'cds'
    assert sha(CORPUS) == '320db2e9f79a87393e2998d088c4de6692b78bafb06f62d57c291cb1484c8fad'
    resource.setrlimit(resource.RLIMIT_AS, (24 * 1024 ** 3, 24 * 1024 ** 3))
    topics, queries = {}, []
    for row in rows(ROOT / 'queries/cds-query.jsonl'):
        qid, text, level = str(row['_id']), row['text'], row['level']
        base = qid.rsplit('_', 1)[0]
        if level == 0:
            assert base not in topics
            topics[base] = text
        else:
            queries.append({'query_id': qid, 'base_id': base, 'level': level, 'request': text})
    for row in queries:
        row['topic'] = topics[row['base_id']]
    queries.sort(key=lambda x: x['query_id'])
    assert len(queries) == len(topics) == N and len({q['query_id'] for q in queries}) == N
    with (ROOT / 'model_queries.jsonl').open('x') as stream:
        for row in queries:
            stream.write(json.dumps(row) + '\n')
    corpus = list(documents())
    ids = [d['document_id'] for d in corpus]
    assert len(ids) == len(set(ids)) >= 100
    empty = sum(not d['text'].strip() for d in corpus)
    with gzip.open(ROOT / 'corpus_ids.json.gz', 'xt') as stream:
        json.dump(ids, stream)
    sys.path.insert(0, str(GATE))
    old = load_module(GATE / 'retrieve_external.py', 'frozen_cds_retrieval')
    tick = time.monotonic()
    postings, statistics = old.build_index([d['text'] for d in corpus])
    print(json.dumps({'stage': 'index_built', 'statistics': statistics,
                      'seconds': time.monotonic() - tick}), flush=True)
    pools, values = {}, {}
    for base in sorted(topics):
        scores = old.score_query(topics[base], postings, len(ids))
        indices = old.top_indices(scores, 100)
        pools[base] = [ids[i] for i in indices]
        values[base] = [float(scores[i]) for i in indices]
    with (ROOT / 'candidates.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['query_id', 'document_id', 'BM25_rank', 'BM25_score'])
        for q in queries:
            for rank, (did, score) in enumerate(zip(pools[q['base_id']], values[q['base_id']]), 1):
                writer.writerow([q['query_id'], did, rank, repr(score)])
    retrieval_seconds = time.monotonic() - tick
    del postings
    needed = {did for pool in pools.values() for did in pool}
    selected = {d['document_id']: d['text'] for d in corpus if d['document_id'] in needed}
    del corpus
    with (ROOT / 'candidate_bodies.jsonl').open('x') as stream:
        for did in sorted(selected):
            stream.write(json.dumps({'document_id': did, 'text': selected[did]}) + '\n')
    from transformers import AutoTokenizer
    mini = AutoTokenizer.from_pretrained(str(CE_MODEL), local_files_only=True, trust_remote_code=False)
    qdir = read(STRONG_QWEN / 'model_assets.json')['directory']
    qwen = AutoTokenizer.from_pretrained(qdir, local_files_only=True, trust_remote_code=False)
    qtext = qwen_text_function()
    cache, lengths, truncated = {}, {c: [] for c in CELLS}, 0
    with gzip.open(ROOT / 'prepared_pairs.jsonl.gz', 'xt') as stream:
        for q in queries:
            maximum = max(len(mini.encode(q[k], add_special_tokens=False)) for k in ['topic', 'request'])
            budget = 512 - 3 - maximum
            assert budget > 0
            for did in pools[q['base_id']]:
                body = selected[did]
                if did not in cache:
                    enc = mini(body, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
                    cache[did] = (enc['input_ids'], enc['offset_mapping'])
                tokens, offsets = cache[did]
                end = len(body) if len(tokens) <= budget else offsets[budget - 1][1]
                prefix = body[:end]
                size = {}
                for cell in CELLS:
                    text = q['topic' if cell.endswith('_topic') else 'request']
                    if cell.startswith('MiniLM'):
                        size[cell] = len(mini(text, prefix, truncation=False)['input_ids'])
                        assert size[cell] <= 512
                    else:
                        size[cell] = len(qwen.encode(qtext(text, prefix), add_special_tokens=False, truncation=False))
                        assert size[cell] <= 2048
                    lengths[cell].append(size[cell])
                stream.write(json.dumps({'query_id': q['query_id'], 'document_id': did,
                    'doc_prefix_end_char': end, 'body_prefix_sha256': hashlib.sha256(prefix.encode()).hexdigest(),
                    'token_lengths': size}) + '\n')
                truncated += end < len(body)
    paths = ['model_queries.jsonl', 'candidates.csv', 'candidate_bodies.jsonl', 'prepared_pairs.jsonl.gz', 'corpus_ids.json.gz']
    result = {'finished_utc': datetime.now(timezone.utc).isoformat(),
        'queries': N, 'topics': len(topics), 'corpus_documents': len(ids), 'empty_bodies': empty,
        'BM25_statistics': statistics, 'retrieval_seconds': retrieval_seconds,
        'candidate_pairs': N * 100, 'unique_candidate_documents': len(selected),
        'truncated_pairs': truncated, 'lengths': {c: {'min': min(v), 'max': max(v), 'total': sum(v)} for c, v in lengths.items()},
        'files': {name: sha(ROOT / name) for name in paths},
        'corpus_sha256': sha(CORPUS), 'source_contract_sha256': sha(ROOT / 'source_contract.json'),
        'code_sha256': sha(ROOT / 'prepare.py'), 'qrels_read': False, 'model_weights_loaded': False,
        'title_used': False, 'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'seconds': time.monotonic() - start}
    check_sources()
    write(ROOT / 'preparation.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
