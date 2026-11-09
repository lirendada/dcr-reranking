"""Frozen candidate neural scoring. Remote GPU only with project lock; no reference reads."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from adapter import sha256, write_json
from prepare_external_neural import ROOT, CE_MODEL, PREV, qwen_text


def verify():
    frozen = json.loads((ROOT / 'external_neural_frozen.json').read_text())
    for name, expected in frozen['code'].items():
        assert sha256(ROOT / name) == expected, name
    for name, expected in frozen['sources'].items():
        assert sha256(Path(name)) == expected, name
    assert str(ROOT).startswith('/path/to/remote/lw_experiments/')
    assert os.environ.get('DCR_GPU_LEASE') == '1'
    assert Path('/proc/self/fd/9').resolve() == Path('/path/to/remote/lw_experiments/logs/haccp_first_seed.lock')


def main():
    started_utc = datetime.now(timezone.utc).isoformat()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['MiniLM', 'Qwen'], required=True)
    args = parser.parse_args()
    verify()
    import torch
    import transformers
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    assert os.environ.get('TOKENIZERS_PARALLELISM') == 'false'
    inputs = ROOT / 'external_inputs'
    corpus = {r['document_id']: r['text'] for r in map(json.loads, (inputs / 'corpus.jsonl').open())}
    variant = 'question0' if args.model == 'MiniLM' else 'RQ_rewrite'
    queries = {r['query_id']: r['text'] for r in map(json.loads, (inputs / f'queries_{variant}.jsonl').open())}
    if args.model == 'MiniLM':
        path = PREV.parent / 'dot_experiment_bundle/research/dcr_error_analysis_v0.1_20261027/ce_baseline.py'
        spec = importlib.util.spec_from_file_location('original_ce_loader', path)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        torch, tokenizer, model, load_seconds = original.load_model(CE_MODEL, 'cuda', 4)
        batch_size = 32
        yes_id = no_id = None
    else:
        sys.path.insert(0, str(PREV))
        import qwen as original
        torch, tokenizer, model, load_seconds = original.load_model()
        yes_id, no_id = (tokenizer.convert_tokens_to_ids(token) for token in ('yes', 'no'))
        assert yes_id != no_id
        batch_size = 16
    folder = ROOT / ('external_' + args.model)
    folder.mkdir(exist_ok=False)
    stats, count = [], 0
    prepared_path = ROOT / 'external_prepared/pairs.jsonl.gz'
    prepared_meta = json.loads((ROOT / 'external_prepared/summary.json').read_text())
    assert sha256(prepared_path) == prepared_meta['prepared_sha256']
    start = time.monotonic()
    def infer_query(query_rows, stream):
        nonlocal count
        qid = query_rows[0]['query_id']
        assert len(query_rows) == 100 and len({r['document_id'] for r in query_rows}) == 100
        tick = time.monotonic()
        for offset in range(0, 100, batch_size):
            batch = query_rows[offset:offset + batch_size]
            bodies = [corpus[r['document_id']][:r['doc_prefix_end_char']] for r in batch]
            if args.model == 'MiniLM':
                encoded = tokenizer([queries[qid]] * len(batch), bodies, truncation=False,
                                    padding='max_length', max_length=512, return_tensors='pt')
                expected_lengths = [r['ce_input_tokens'] for r in batch]
                assert encoded['input_ids'].shape[1] == 512
            else:
                encoded = tokenizer([qwen_text(queries[qid], body) for body in bodies],
                                    add_special_tokens=False, truncation=False, padding=True,
                                    pad_to_multiple_of=8, return_tensors='pt')
                expected_lengths = [r['qwen_input_tokens'] for r in batch]
                assert encoded['input_ids'].shape[1] <= 2048
            lengths = encoded['attention_mask'].sum(1).tolist()
            assert lengths == expected_lengths
            with torch.inference_mode():
                tensors = {k: v.to('cuda') for k, v in encoded.items()}
                if args.model == 'MiniLM':
                    values = model(**tensors).logits.squeeze(-1).cpu().tolist()
                    extra = [{} for _ in batch]
                else:
                    logits = model(**tensors, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
                    yes, no = logits[:, yes_id].cpu().tolist(), logits[:, no_id].cpu().tolist()
                    values = [y - n for y, n in zip(yes, no)]
                    extra = [{'yes_logit': y, 'no_logit': n, 'demote': n > y} for y, n in zip(yes, no)]
            for row, value, ext, length in zip(batch, values, extra, lengths):
                assert math.isfinite(value) and all(math.isfinite(ext[k]) for k in ('yes_logit', 'no_logit') if k in ext)
                stream.write(json.dumps({'query_id': qid, 'document_id': row['document_id'],
                    'score': value, 'doc_prefix_end_char': row['doc_prefix_end_char'],
                    'input_tokens': length, **ext}) + '\n')
                count += 1
        torch.cuda.synchronize()
        row = {'query_id': qid, 'pairs': len(query_rows), 'seconds': time.monotonic() - tick,
               'pairs_completed': count, 'total_seconds': time.monotonic() - start,
               'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
               'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        stats.append(row)
        print(json.dumps(row), flush=True)
    with gzip.open(prepared_path, 'rt') as source, gzip.open(folder / 'scores.jsonl.gz', 'wt') as target:
        buffered, previous = [], None
        for line in source:
            r = json.loads(line)
            if previous is not None and r['query_id'] != previous:
                infer_query(buffered, target)
                target.flush()
                buffered = []
            buffered.append(r)
            previous = r['query_id']
        if buffered:
            infer_query(buffered, target)
    assert count == 345200 and len(stats) == 3452
    with (folder / 'time_and_memory.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    write_json(folder / 'manifest.json', {'model': args.model, 'query_variant': variant,
        'started_utc': started_utc, 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'pairs': count, 'queries': len(stats), 'load_seconds': load_seconds,
        'inference_seconds': time.monotonic() - start, 'batch_size': batch_size,
        'precision': 'float32' if args.model == 'MiniLM' else 'bfloat16',
        'environment': {'torch': torch.__version__, 'transformers': transformers.__version__,
                        'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name()},
        'scores_sha256': sha256(folder / 'scores.jsonl.gz'),
        'candidate_sha256': prepared_meta['candidate_sha256'], 'prepared_sha256': sha256(prepared_path),
        'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'references_read': [], 'task_retries': 0,
        'frozen_manifest_sha256': sha256(ROOT / 'external_neural_frozen.json')})
    verify()


if __name__ == '__main__':
    main()
