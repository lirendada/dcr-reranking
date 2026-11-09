"""Assemble all four fixed-head rankings before any reference access."""
from datetime import datetime, timezone
import gzip
import json
import math
import numpy as np
from config import (ROOT, SOURCE_POST, HEAD_PATH, CELLS, NEW_CELLS, EXPECTED_QUERIES,
                    EXPECTED_NEW_PAIRS, read, write, rows, sha, verify_code, verify_sources)


def main():
    manifest = verify_code()
    verify_sources(manifest, 'input')
    meta = read(HEAD_PATH)
    qids, pools = meta['qids'], meta['ranked_pools']
    assert len(qids) == EXPECTED_QUERIES
    selected = {(q, did): (i, j) for i, (q, pool) in enumerate(zip(qids, pools))
                for j, did in enumerate(pool[:20])}
    prefix = {(r['query_id'], r['document_id']): r['doc_prefix_end_char']
              for r in rows(ROOT/'prepared/pairs.jsonl.gz')}
    assert set(prefix) == set(selected)
    canonical = np.asarray([sorted(range(20), key=lambda j: pool[j]) for pool in pools])
    scores = np.empty((4, len(qids), 20))
    orders = np.empty((4, len(qids), 20), dtype=np.uint8)
    inventory = {}
    for ci, cell in enumerate(CELLS):
        path = ROOT/'cells'/cell/'scores.jsonl.gz'
        if cell in NEW_CELLS:
            receipt = read(ROOT/'cells'/cell/'receipt.json')
            assert receipt['exit_code'] == 0 and receipt['pairs'] == EXPECTED_NEW_PAIRS
            assert receipt['score_sha256'] == sha(path)
        seen, tokens = set(), 0
        for row in rows(path):
            key = row['query_id'], row['document_id']
            assert key in selected and key not in seen and math.isfinite(row['score'])
            assert row['doc_prefix_end_char'] == prefix[key]
            seen.add(key)
            i, j = selected[key]
            scores[ci, i, j] = row['score']
            tokens += row['input_tokens']
        assert seen == set(selected)
        arranged = np.take_along_axis(scores[ci], canonical, axis=1)
        orders[ci] = np.take_along_axis(canonical, np.argsort(-arranged, axis=1, kind='stable'), axis=1)
        inventory[cell] = {'pairs': len(seen), 'input_tokens_total': tokens, 'scores_sha256': sha(path)}
    with np.load(SOURCE_POST/'features.npz', allow_pickle=False) as z:
        assert np.array_equal(scores[0], z['raw_scores'][:, :, 0])
        assert np.array_equal(scores[3], z['raw_scores'][:, :, 1])
        assert np.array_equal(canonical, z['canonical'])
    assert np.array_equal(orders[0], np.tile(np.arange(20), (len(qids), 1)))
    with gzip.open(ROOT/'rankings.jsonl.gz', 'xt') as f:
        for ci, cell in enumerate(CELLS):
            for i, qid in enumerate(qids):
                full = [pools[i][j] for j in orders[ci, i]] + pools[i][20:]
                assert len(full) == len(set(full)) == 100
                f.write(json.dumps({'cell': cell, 'query_id': qid, 'order': full})+'\n')
    assert not (ROOT/'four_cell_arrays.npz').exists()
    np.savez_compressed(ROOT/'four_cell_arrays.npz', scores=scores, orders=orders,
                        cells=np.asarray(CELLS), qids=np.asarray(qids))
    files = ['rankings.jsonl.gz', 'four_cell_arrays.npz']
    files += [f'cells/{c}/scores.jsonl.gz' for c in CELLS]
    files += [f'cells/{c}/receipt.json' for c in NEW_CELLS]
    write(ROOT/'predictions_sealed.json', {'sealed_utc': datetime.now(timezone.utc).isoformat(),
        'frozen_manifest_sha256': sha(ROOT/'frozen_manifest.json'),
        'files': {name: sha(ROOT/name) for name in files},
        'cell_inventory': inventory, 'old_scores_exactly_reproduced': True,
        'scoring_references_opened': False, 'new_neural_pairs': EXPECTED_NEW_PAIRS*2})
    print('Four complete rankings sealed; reused raw scores match old features exactly.', flush=True)


if __name__ == '__main__':
    main()
