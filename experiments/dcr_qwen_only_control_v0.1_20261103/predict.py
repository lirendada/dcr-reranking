"""Score cached target features without references, then seal every Q7 prediction."""
import gzip
import json
import time
import numpy as np
from config import ROOT, POST, SAME, IFIR, Q7, TARGETS, read, write, rows, sha, check, target_spec, now


def main() -> None:
    start = time.monotonic()
    check()
    with np.load(POST / 'features.npz', allow_pickle=False) as old, np.load(SAME / 'features.npz', allow_pickle=False) as same:
        assert np.array_equal(old['X'][:, :, Q7], same['X'][:, :, Q7])
    model = read(ROOT / 'transfer_model.json')
    assert model['columns'] == Q7 and model['pair_instances'] == 2901
    (ROOT / 'targets').mkdir(exist_ok=False)
    for dataset in TARGETS:
        spec = target_spec(dataset)
        with np.load(spec['features'], allow_pickle=False) as data:
            original = data[spec['feature_key']][:, :, :12]
            x = np.ascontiguousarray(original[:, :, Q7])
        if dataset == 'cds':
            metadata = read(spec['meta'])
        else:
            meta = read(spec['meta'])
            queries = list(rows(spec['queries']))
            bases = {r['query_id']: r['order'] for r in rows(spec['rankings']) if r['method'] == 'MiniLM_topic'}
            assert meta['qids'] == [q['query_id'] for q in queries]
            metadata = [{'query_id': q['query_id'], 'base_id': q['base_id'], 'level': q['level'],
                         'head': head, 'base': bases[q['query_id']]} for q, head in zip(queries, meta['heads'])]
            with np.load(IFIR / dataset / 'outputs/features.npz', allow_pickle=False) as same:
                assert np.array_equal(x, same['G_same12'][:, :, Q7])
        if dataset == 'cds':
            with np.load(spec['features'], allow_pickle=False) as same:
                assert np.array_equal(x, same['G_same12'][:, :, Q7])
        assert len(x) == len(metadata) == {'fiqa': 1718, 'nfcorpus': 86, 'cds': 43}[dataset]
        scores = (x / np.asarray(model['scale'])) @ np.asarray(model['coef'])
        assert np.isfinite(scores).all()
        with gzip.open(ROOT / 'targets' / f'{dataset}_rankings.jsonl.gz', 'xt') as stream:
            for i, meta in enumerate(metadata):
                head, base = meta['head'], meta['base']
                assert len(base) == len(set(base)) == 100 and head == base[:20]
                order = [head[j] for j in sorted(range(20), key=lambda j: (-scores[i, j], head[j]))] + base[20:]
                assert order[20:] == base[20:] and set(order) == set(base)
                stream.write(json.dumps({'query_id': meta['query_id'], 'base_id': meta['base_id'],
                                         'method': 'Q7', 'order': order}) + '\n')
        write(ROOT / 'targets' / f'{dataset}_metadata.json', metadata)
    files = [ROOT / 'transfer_model.json'] + sorted((ROOT / 'fitted').glob('*')) + sorted((ROOT / 'targets').glob('*'))
    write(ROOT / 'predictions_sealed.json', {'sealed_utc': now(), 'files': {str(p.relative_to(ROOT)): sha(p) for p in files},
          'target_qrels_read': False, 'fold_training_views_used': True,
          'transfer_fit_source_labels_used': True, 'frozen_manifest_sha256': sha(ROOT / 'frozen_manifest.json'),
          'new_neural_pairs': 0, 'seconds': time.monotonic() - start})


if __name__ == '__main__':
    main()
