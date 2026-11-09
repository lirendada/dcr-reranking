#!/usr/bin/env python3
"""Frozen CPU-only IFIR continuation; preserve the ExcluIR analysis unchanged."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import random
import resource
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from features import FIELDS, features

ROOT = Path(__file__).resolve().parent
RUNS = Path('/path/to/remote/lw_experiments/runs')
OLD = RUNS / 'dcr_post_g_external_v0.1_20261029'
FOUR = RUNS / 'dcr_input_ifir_v0.1_20261029'
CELLS = ('MiniLM_topic', 'MiniLM_full', 'Qwen_topic', 'Qwen_full')
DATASETS = {'fiqa': (1718, 639), 'nfcorpus': (86, 86)}
SEED, DRAWS = 20261030, 1000
FREEZE = ROOT / 'ifir_frozen_manifest.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def jsonlines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def csvwrite(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inputs():
    paths = [ROOT / 'ifir_analysis.py', ROOT / 'features.py', ROOT / 'PROTOCOL.md',
             ROOT / 'IFIR_EXECUTION_SPEC.md', FOUR / 'per_query.csv.gz']
    for dataset in DATASETS:
        paths += [OLD / dataset / 'inputs/queries.jsonl',
                  OLD / 'raw' / dataset / f'{dataset}-query.jsonl']
        paths += [FOUR / dataset / 'prepared' / f'queries_{variant}.jsonl'
                  for variant in ('topic', 'full')]
    return paths


def pct(values, fraction):
    ordered = sorted(values)
    at = fraction * (len(ordered) - 1)
    lo = math.floor(at)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def load_dataset(dataset):
    queries = jsonlines(OLD / dataset / 'inputs/queries.jsonl')
    qids = [row['query_id'] for row in queries]
    assert len(qids) == len(set(qids)) == DATASETS[dataset][0]
    assert len({q['base_id'] for q in queries}) == DATASETS[dataset][1]
    original = {row['_id']: row for row in jsonlines(OLD / 'raw' / dataset / f'{dataset}-query.jsonl')}
    for variant, field in [('topic', 'topic'), ('full', 'request')]:
        prepared = jsonlines(FOUR / dataset / 'prepared' / f'queries_{variant}.jsonl')
        assert prepared == [{'query_id': q['query_id'], 'text': q[field]} for q in queries]
    for q in queries:
        assert q['level'] > 0
        assert original[q['base_id'] + '_0']['level'] == 0
        assert q['topic'] == original[q['base_id'] + '_0']['text']
        assert q['request'] == original[q['query_id']]['text']
        assert q['level'] == original[q['query_id']]['level']
    scores = {cell: {} for cell in CELLS}
    with gzip.open(FOUR / 'per_query.csv.gz', 'rt') as stream:
        for row in csv.DictReader(stream):
            if row['dataset'] == dataset and row['method'] in CELLS:
                cell, qid = row['method'], row['query_id']
                assert qid not in scores[cell]
                scores[cell][qid] = (float(row['P1']), row['base_id'], int(row['level']))
    records = []
    for q in queries:
        record = {'dataset': dataset, 'query_id': q['query_id'], 'base_id': q['base_id'],
                  'level': q['level'], **features(q['topic'], q['request'])}
        for cell in CELLS:
            value, base_id, level = scores[cell][q['query_id']]
            assert (base_id, level) == (q['base_id'], q['level'])
            assert value in (0.0, 1.0)
            record['P1_' + cell] = value
        record['dP1_MiniLM'] = record['P1_MiniLM_full'] - record['P1_MiniLM_topic']
        record['dP1_Qwen'] = record['P1_Qwen_full'] - record['P1_Qwen_topic']
        records.append(record)
    assert all(set(scores[cell]) == set(qids) for cell in CELLS)
    return sorted(records, key=lambda row: row['query_id'])


def summarize(records):
    count = len(records)
    result = {'n': count, 'bases': len({row['base_id'] for row in records})}
    for key in [*('P1_' + cell for cell in CELLS), 'dP1_MiniLM', 'dP1_Qwen']:
        result[key + '_pct_or_pp'] = 100 * statistics.mean(row[key] for row in records)
    return result


def analyze(dataset, output):
    records = load_dataset(dataset)
    values = sorted(row['added_tokens'] for row in records)
    cuts = [values[len(values) // 3], values[2 * len(values) // 3]]
    for row in records:
        row['added_stratum'] = 'low' if row['added_tokens'] <= cuts[0] else (
            'mid' if row['added_tokens'] <= cuts[1] else 'high')
    strata = []
    for variable in ['added_stratum', 'level']:
        for group in sorted({row[variable] for row in records}):
            subset = [row for row in records if row[variable] == group]
            strata.append({'dataset': dataset, 'variable': variable, 'stratum': group,
                           **summarize(subset),
                           'mean_added_tokens': statistics.mean(row['added_tokens'] for row in subset)})
    feature_summary = {}
    for key in FIELDS[1:]:
        vals = [row[key] for row in records]
        feature_summary[key] = {'min': min(vals), 'median': statistics.median(vals),
                                'mean': statistics.mean(vals), 'max': max(vals)}
    summary = {'dataset': dataset, 'all': summarize(records), 'feature_summary': feature_summary,
               'added_tokens_cuts': cuts, 'strata': strata,
               'literal_containment_count': sum(row['literal_containment'] for row in records),
               'zero_removed_tokens_count': sum(row['removed_tokens'] == 0 for row in records),
               'source_bindings_verified': True, 'bootstrap_intervals': []}
    if dataset == 'fiqa':
        low = [r for r in records if r['added_stratum'] == 'low']
        high = [r for r in records if r['added_stratum'] == 'high']
        assert low and high, 'Primary high/low comparison requires nonempty fixed groups'
        bases = sorted({r['base_id'] for r in records})
        rng = random.Random(SEED)

        def contrasts(weights):
            out = {}
            for model in ['MiniLM', 'Qwen']:
                means = []
                for subset in [high, low]:
                    total = sum(weights[r['base_id']] for r in subset)
                    if total == 0:
                        means.append(None)
                    else:
                        means.append(sum(weights[r['base_id']] * r['dP1_' + model]
                                         for r in subset) / total)
                out[model] = None if None in means else 100 * (means[0] - means[1])
            return out

        point = contrasts(dict.fromkeys(bases, 1))
        draws = []
        for draw in range(DRAWS):
            weights = dict.fromkeys(bases, 0)
            for _ in bases:
                weights[bases[rng.randrange(len(bases))]] += 1
            draws.append({'draw': draw, **contrasts(weights)})
        for model in ['MiniLM', 'Qwen']:
            valid = [r[model] for r in draws if r[model] is not None]
            assert valid, 'No valid resamples'
            summary['bootstrap_intervals'].append({
                'model': model, 'comparison': 'added_tokens_high_minus_low_dP1',
                'primary': model == 'MiniLM', 'point_pp': point[model],
                'ci95_low_pp': pct(valid, .025), 'ci95_high_pp': pct(valid, .975),
                'draws_used': len(valid), 'undefined_draws': DRAWS - len(valid)})
        with gzip.open(output / 'bootstrap_draws.csv.gz', 'wt', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['draw', 'MiniLM', 'Qwen'])
            writer.writeheader()
            writer.writerows(draws)
        csvwrite(output / 'bootstrap_intervals.csv', summary['bootstrap_intervals'])
        summary['resampling'] = {'unit': 'base_id', 'estimand': 'query-weighted mean',
                                  'seed': SEED, 'draws': DRAWS,
                                  'generator': 'random.Random; sorted base IDs; randrange(n_base) repeated n_base times per draw',
                                  'percentiles': 'linear interpolation at p*(n-1)'}
    csvwrite(output / 'per_query_features.csv', records)
    csvwrite(output / 'strata_results.csv', strata)
    dump(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--run-sha256')
    args = parser.parse_args()
    if args.freeze:
        assert not FREEZE.exists(), 'Freeze already exists; do not overwrite'
        dump(FREEZE, {'created_utc': utc(), 'sources': {str(p): sha(p) for p in inputs()},
                      'seed': SEED, 'draws': DRAWS})
        print(sha(FREEZE))
        return
    assert args.run_sha256 and sha(FREEZE) == args.run_sha256
    manifest = json.loads(FREEZE.read_text())
    assert all(sha(Path(p)) == h for p, h in manifest['sources'].items())
    output = ROOT / 'ifir'
    output.mkdir(exist_ok=False)
    started = utc()
    t0 = time.perf_counter()
    summaries = {}
    for dataset in DATASETS:
        folder = output / dataset
        folder.mkdir()
        summaries[dataset] = analyze(dataset, folder)
    assert all(sha(Path(p)) == h for p, h in manifest['sources'].items())
    dump(output / 'summary.json', {'datasets': summaries, 'new_neural_pairs': 0,
                                  'interpretation': 'Observed-set post-hoc diagnostic; no cross-dataset significance test.'})
    dump(output / 'execution_receipt.json', {'started_utc': started, 'finished_utc': utc(),
         'elapsed_seconds': time.perf_counter() - t0, 'process_exit_code': 0,
         'cpu_user_seconds': resource.getrusage(resource.RUSAGE_SELF).ru_utime,
         'cpu_system_seconds': resource.getrusage(resource.RUSAGE_SELF).ru_stime,
         'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         'python': platform.python_version(), 'platform': platform.platform(),
         'freeze_sha256': args.run_sha256, 'sources_unchanged': True,
         'new_neural_pairs': 0, 'gpu_used': False})
    dump(output / 'output_hashes.json', {str(p.relative_to(output)): sha(p)
         for p in sorted(output.rglob('*')) if p.is_file()})
    print(json.dumps({'completed': True, 'datasets': list(summaries),
                      'fiqa_intervals': summaries['fiqa']['bootstrap_intervals']}))


if __name__ == '__main__':
    main()
