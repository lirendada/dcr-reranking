"""Seven frozen methods, complete rankings and prediction seal; no references."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
import time
import numpy as np

from config import (ROOT, A, OLD, CELLS, METHODS, N, rows, read, write, sha,
                    check_release, source_math, source_matched, parser_module)


def main() -> None:
    start = time.monotonic()
    check_release()
    queries = list(rows(ROOT / 'model_queries.jsonl'))
    heads = read(ROOT / 'head.json')['queries']
    corpus = {d['document_id']: d['text'] for d in rows(ROOT / 'candidate_bodies.jsonl')}
    pairs = {(p['query_id'], p['document_id']): p for p in rows(ROOT / 'prepared_pairs.jsonl.gz')}
    scores, files = {}, {}
    for cell in CELLS:
        path = ROOT / 'cells' / cell / 'scores.jsonl.gz'
        receipt = read(path.parent / 'receipt.json')
        assert receipt['exit_code'] == 0 and sha(path) == receipt['score_sha256']
        bucket = {}
        for row in rows(path):
            qid, did = row['query_id'], row['document_id']
            assert did not in bucket.setdefault(qid, {})
            assert np.isfinite(row['score'])
            assert row['doc_prefix_end_char'] == pairs[qid, did]['doc_prefix_end_char']
            bucket[qid][did] = row['score']
        assert set(bucket) == set(heads)
        scores[cell] = bucket
        files[str(path.relative_to(ROOT))] = sha(path)
    seal = read(A / 'transfer_models_sealed.json')
    assert sha(A / 'transfer_models.json') == seal['models_sha256']
    matched_models = read(A / 'transfer_models.json')['models']
    gmodel = read(OLD / 'source_model.json')
    assert gmodel['qualified_pairs'] == 2901 and gmodel['dimensions'] == 12 and gmodel['lambda'] == .1
    for name in ['G_same12', 'QL5']:
        assert matched_models[name]['pair_instances'] == 2901 and matched_models[name]['lambda'] == .1
    parser, math = parser_module(), source_math()
    x_old, mini_full, metadata = [], [], []
    for query in queries:
        qid = query['query_id']
        head, base = heads[qid]['head'], heads[qid]['base']
        assert len(base) == len(set(base)) == 100 and head == base[:20]
        assert set(scores['MiniLM_topic'][qid]) == set(base)
        for cell in CELLS[1:]:
            assert set(scores[cell][qid]) == set(head)
        ms = np.array([scores['MiniLM_topic'][qid][did] for did in head])
        qs = np.array([scores['Qwen_full'][qid][did] for did in head])
        tokens = []
        for did in head:
            pair = pairs[qid, did]
            body = corpus[did][:pair['doc_prefix_end_char']]
            assert hashlib.sha256(body.encode()).hexdigest() == pair['body_prefix_sha256']
            tokens.append(parser.tokenize(body, remove_stop=True))
        x = math.doc_features(ms, qs, tokens, parser.tokenize(query['topic'], remove_stop=True),
                              parser.tokenize(query['request'], remove_stop=True), [])[:, :12]
        x[:, 2] = np.arange(1, 21) / 20
        for rank, i in enumerate(sorted(range(20), key=lambda j: (-qs[j], head[j]))):
            x[i, 3] = (rank + 1) / 20
        x_old.append(x)
        mini_full.append([scores['MiniLM_full'][qid][did] for did in head])
        metadata.append({'query_id': qid, 'base_id': query['base_id'], 'level': query['level'],
                         'head': head, 'base': base})
    original = np.asarray(x_old)
    canonical = np.array([sorted(range(20), key=lambda i: q['head'][i]) for q in metadata])
    same = source_matched()(original, np.asarray(mini_full), canonical)
    ql = np.ascontiguousarray(original[:, :, [1, 4, 5, 6, 7]])
    values = {'G12': (original / np.array(gmodel['scale'])) @ np.array(gmodel['coef'])}
    for name, features in [('G_same12', same), ('QL5', ql)]:
        model = matched_models[name]
        values[name] = (features / np.array(model['scale'])) @ np.array(model['coef'])
    assert all(v.shape == (N, 20) and np.isfinite(v).all() for v in values.values())
    with gzip.open(ROOT / 'rankings.jsonl.gz', 'xt') as stream:
        for i, meta in enumerate(metadata):
            qid, head, base = meta['query_id'], meta['head'], meta['base']
            for method in METHODS:
                if method == 'MiniLM_topic':
                    order = base
                elif method in CELLS:
                    order = sorted(head, key=lambda did: (-scores[method][qid][did], did)) + base[20:]
                else:
                    order = [head[j] for j in sorted(range(20), key=lambda j: (-values[method][i, j], head[j]))] + base[20:]
                assert len(order) == len(set(order)) == 100 and set(order) == set(base) and order[20:] == base[20:]
                stream.write(json.dumps({'query_id': qid, 'base_id': meta['base_id'], 'level': meta['level'],
                                         'method': method, 'order': order}) + '\n')
    np.savez_compressed(ROOT / 'features.npz', G12=original, G_same12=same, QL5=ql)
    write(ROOT / 'ranking_metadata.json', metadata)
    for name in ['rankings.jsonl.gz', 'features.npz', 'ranking_metadata.json', 'head.json']:
        files[name] = sha(ROOT / name)
    write(ROOT / 'predictions_sealed.json', {'sealed_utc': datetime.now(timezone.utc).isoformat(),
        'files': files, 'rankings': N * len(METHODS), 'new_model_pairs': N * 160,
        'source_G_sha256': sha(OLD / 'source_model.json'),
        'source_matched_models_sha256': sha(A / 'transfer_models.json'),
        'frozen_manifest_sha256': sha(ROOT / 'frozen_manifest.json'),
        'qrels_read': False, 'seconds': time.monotonic() - start})


if __name__ == '__main__':
    main()
