"""One frozen model-input cell; reference reads are prohibited by config."""
import argparse
import ast
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

from config import (ROOT, CE_MODEL, CE_LOADER, STRONG_QWEN, GPU_LOCK, N, CELLS,
                    rows, read, write, sha, load_module, check_release, qwen_text_function)


def qwen_loader():
    sys.path.insert(0, str(STRONG_QWEN))
    old = load_module(STRONG_QWEN / 'qwen.py', 'frozen_cds_qwen')
    tree = ast.parse((STRONG_QWEN / 'qwen.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_model')
    changed = 0
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'set_num_threads':
            assert len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and node.args[0].value == 4
            node.args[0].value = 1
            changed += 1
    assert changed == 1
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(STRONG_QWEN / 'qwen.py'), 'exec'), old.__dict__)
    return old.load_model()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', choices=CELLS, required=True)
    cell = parser.parse_args().cell
    check_release()
    assert os.environ.get('DCR_GPU_LEASE') == '1'
    assert Path('/proc/self/fd/9').resolve() == GPU_LOCK
    assert os.environ.get('TOKENIZERS_PARALLELISM') == 'false'
    queries = {q['query_id']: q for q in rows(ROOT / 'model_queries.jsonl')}
    bodies = {d['document_id']: d['text'] for d in rows(ROOT / 'candidate_bodies.jsonl')}
    prepared = {}
    for row in rows(ROOT / 'prepared_pairs.jsonl.gz'):
        bucket = prepared.setdefault(row['query_id'], {})
        assert row['document_id'] not in bucket
        bucket[row['document_id']] = row
    selection = cell == 'MiniLM_topic'
    heads = None if selection else read(ROOT / 'head.json')['queries']
    out = ROOT / 'cells' / cell
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    is_mini = cell.startswith('MiniLM')
    if is_mini:
        old = load_module(CE_LOADER, 'frozen_cds_ce')
        torch, tokenizer, model, load_seconds = old.load_model(CE_MODEL, 'cuda', 1)
    else:
        torch, tokenizer, model, load_seconds = qwen_loader()
    assert not model.training
    torch.set_num_threads(1)
    import transformers
    assert transformers.__version__ == ('4.49.0' if is_mini else '4.51.3')
    batch_size = 32 if is_mini else 16
    qtext = qwen_text_function()
    yes_id, no_id = (None, None) if is_mini else tuple(tokenizer.convert_tokens_to_ids(x) for x in ['yes', 'no'])
    total, token_total, maximum_batch = 0, 0, 0
    times, selection_heads = [], {}
    variant = 'topic' if cell.endswith('_topic') else 'request'
    score_start = time.monotonic()
    with gzip.open(out / 'scores.jsonl.gz', 'xt') as stream:
        for qid, query in queries.items():
            pool = prepared[qid]
            ids = list(pool) if selection else heads[qid]['head']
            assert len(ids) == (100 if selection else 20)
            tick = time.monotonic()
            values_for_query = {}
            for offset in range(0, len(ids), batch_size):
                dids = ids[offset:offset + batch_size]
                records = [pool[did] for did in dids]
                prefixes = [bodies[did][:row['doc_prefix_end_char']] for did, row in zip(dids, records)]
                assert all(hashlib.sha256(body.encode()).hexdigest() == row['body_prefix_sha256'] for body, row in zip(prefixes, records))
                if is_mini:
                    encoded = tokenizer([query[variant]] * len(dids), prefixes, truncation=False,
                                        padding='max_length', max_length=512, return_tensors='pt')
                    assert encoded['input_ids'].shape[1] == 512
                else:
                    encoded = tokenizer([qtext(query[variant], body) for body in prefixes],
                                        add_special_tokens=False, truncation=False, padding=True,
                                        pad_to_multiple_of=8, return_tensors='pt')
                    assert encoded['input_ids'].shape[1] <= 2048
                lengths = encoded['attention_mask'].sum(1).tolist()
                assert lengths == [row['token_lengths'][cell] for row in records]
                with torch.inference_mode():
                    tensors = {k: v.to('cuda') for k, v in encoded.items()}
                    if is_mini:
                        scores = model(**tensors).logits.squeeze(-1).cpu().tolist()
                        extras = [{} for _ in dids]
                    else:
                        logits = model(**tensors, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
                        yes, no = logits[:, yes_id].cpu().tolist(), logits[:, no_id].cpu().tolist()
                        scores = [a - b for a, b in zip(yes, no)]
                        extras = [{'yes_logit': a, 'no_logit': b} for a, b in zip(yes, no)]
                maximum_batch = max(maximum_batch, len(dids))
                for did, row, value, length, extra in zip(dids, records, scores, lengths, extras):
                    assert math.isfinite(value)
                    stream.write(json.dumps({'query_id': qid, 'document_id': did, 'score': value,
                        'doc_prefix_end_char': row['doc_prefix_end_char'], 'input_tokens': length, **extra}) + '\n')
                    values_for_query[did] = value
                    total += 1
                    token_total += length
            stream.flush()
            if selection:
                order = sorted(ids, key=lambda did: (-values_for_query[did], did))
                selection_heads[qid] = {'head': order[:20], 'base': order}
            times.append({'query_id': qid, 'pairs': len(ids), 'seconds': time.monotonic() - tick})
    torch.cuda.synchronize()
    assert total == N * (100 if selection else 20)
    receipt = {'cell': cell, 'pairs': total, 'queries': N, 'tokens': token_total,
        'score_sha256': sha(out / 'scores.jsonl.gz'), 'load_seconds': load_seconds,
        'scoring_wall_seconds': time.monotonic() - score_start, 'wall_seconds': time.monotonic() - start,
        'batch_size_configured': batch_size, 'max_actual_batch': maximum_batch,
        'CPU_threads_override': 1, 'precision': 'float32' if is_mini else 'bfloat16',
        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'environment': {'torch': torch.__version__, 'transformers': transformers.__version__,
                        'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name()},
        'started_utc': started_utc, 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'exit_code': 0, 'qrels_read': False, 'task_retries': 0}
    write(out / 'receipt.json', receipt)
    write(out / 'query_times.json', times)
    if selection:
        write(ROOT / 'head.json', {'queries': selection_heads, 'selection_score_sha256': receipt['score_sha256']})
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
