"""True full-corpus ExcluIR BM25; inputs deliberately omit all reference fields."""
import argparse
from array import array
from collections import Counter
import csv
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time

from adapter import sha256, write_json

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'dot_experiment_bundle/research/dcr_public_baselines_v0.1_20261027'
sys.path.insert(0, str(OLD))
from frozen_parser import tokenize


def build_index(texts):
    import numpy as np
    postings, lengths = {}, []
    for index, text in enumerate(texts):
        tf = Counter(tokenize(text))
        lengths.append(sum(tf.values()))
        for term, count in tf.items():
            if term not in postings:
                postings[term] = (array('I'), array('I'))
            postings[term][0].append(index)
            postings[term][1].append(count)
    lengths = np.asarray(lengths, dtype=np.float64)
    avgdl = float(lengths.mean())
    assert avgdl > 0
    norm = 1.5 * (.25 + .75 * lengths / avgdl)
    for term, (indices, counts) in list(postings.items()):
        ids = np.asarray(indices, dtype=np.int64)
        tf = np.asarray(counts, dtype=np.float64)
        idf = math.log(1 + (len(texts) - len(ids) + .5) / (len(ids) + .5))
        postings[term] = (ids, idf * tf * 2.5 / (tf + norm[ids]))
    return postings, {'N': len(texts), 'avgdl': avgdl, 'tokens': int(lengths.sum()),
                      'vocabulary': len(postings), 'postings': sum(len(v[0]) for v in postings.values())}


def score_query(text, postings, count):
    import numpy as np
    scores = np.zeros(count, dtype=np.float64)
    for term, freq in Counter(tokenize(text)).items():
        if term in postings:
            ids, values = postings[term]
            scores[ids] += freq * values
    assert np.isfinite(scores).all() and (scores >= 0).all()
    return scores


def top_indices(scores, k):
    import numpy as np
    k = min(k, len(scores))
    cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
    candidates = np.flatnonzero(scores >= cutoff)
    return sorted(candidates.tolist(), key=lambda i: (-float(scores[i]), i))[:k]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert str(ROOT).startswith('/path/to/remote/lw_experiments/')
    frozen = json.loads((ROOT / 'frozen_manifest.json').read_text())
    for name, expected in frozen['code'].items():
        assert sha256(ROOT / name) == expected, name
    for name, expected in frozen['sources'].items():
        assert sha256(Path(name)) == expected, name
    meta = json.loads((args.bundle / 'metadata.json').read_text())
    # Deliberately never open references.json or raw query JSON with pair labels.
    queries_file = args.bundle / 'queries_RQ_rewrite.jsonl'
    corpus_file = args.bundle / 'corpus.jsonl'
    for file in (queries_file, corpus_file):
        assert sha256(file) == meta['export_sha256'][file.name]
    queries = [json.loads(line) for line in queries_file.open()]
    corpus = [json.loads(line) for line in corpus_file.open()]
    assert len(queries) == meta['queries'] == 3452
    assert len(corpus) == meta['documents'] == 90406
    assert all(set(q) == {'query_id', 'text'} for q in queries)
    assert all(set(d) == {'document_id', 'text'} for d in corpus)
    assert [d['document_id'] for d in corpus] == [f'd{i:06d}' for i in range(90406)]
    args.output.mkdir(exist_ok=False)
    start = time.monotonic()
    index, statistics = build_index([d['text'] for d in corpus])
    indexed = time.monotonic()
    times, zero_queries = [], 0
    run, candidates = args.output / 'bm25_top100.csv', args.output / 'candidates.csv'
    with run.open('x', newline='') as rf, candidates.open('x', newline='') as cf:
        writer, pool = csv.writer(rf), csv.writer(cf)
        writer.writerow(['query_id', 'document_id', 'score'])
        pool.writerow(['query_id', 'document_id'])
        for position, q in enumerate(queries):
            tick = time.monotonic()
            scores = score_query(q['text'], index, len(corpus))
            selected = top_indices(scores, 100)
            zero_queries += int(scores.max() == 0)
            for i in selected:
                did = corpus[i]['document_id']
                writer.writerow([q['query_id'], did, repr(float(scores[i]))])
                pool.writerow([q['query_id'], did])
            times.append({'query_id': q['query_id'], 'seconds': time.monotonic() - tick,
                          'documents_scored': len(corpus), 'returned': len(selected)})
            if (position + 1) % 250 == 0:
                print(json.dumps({'queries_completed': position + 1, 'seconds': time.monotonic() - start}), flush=True)
    import numpy as np
    manifest = {'stage': 'full_corpus_retrieval', 'query_variant': 'RQ_rewrite',
        'engine': 'exact float64 postings BM25 k1=1.5 b=.75; historical lowercase/ASCII-punctuation-strip/whitespace tokenizer; repeated query tokens; full corpus collection statistics',
        'environment': f'{platform.platform()} Python{platform.python_version()} numpy{np.__version__}',
        'source_sha256': meta['source_sha256'], 'search_space_document_count': len(corpus),
        'references_used_for_candidates': False, 'reference_files_opened_by_retriever': [],
        'queries': len(queries), 'pairs_returned': len(queries) * 100, 'statistics': statistics,
        'index_seconds': indexed - start, 'retrieval_and_write_seconds': time.monotonic() - indexed,
        'total_seconds': time.monotonic() - start,
        'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'all_zero_score_queries': zero_queries, 'topk': 100,
        'candidate_sha256': sha256(candidates), 'run_sha256': sha256(run),
        'tokenizer_sha256': sha256(OLD / 'frozen_parser.py'),
        'frozen_manifest_sha256': sha256(ROOT / 'frozen_manifest.json')}
    write_json(args.output / 'retrieval_manifest.json', manifest)
    with (args.output / 'retrieval_times.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(times[0]))
        writer.writeheader()
        writer.writerows(times)
    write_json(args.output / 'candidates_frozen.json', {'candidate_sha256': sha256(candidates),
        'run_sha256': sha256(run), 'queries': 3452, 'documents_per_query': 100,
        'frozen_before_reference_scoring': True})
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
