"""Reconstruct point estimates from the released per-query outcomes."""

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path, file_format):
    with gzip.open(path, 'rt') as stream:
        if file_format == 'csv.gz':
            yield from csv.DictReader(stream)
        elif file_format == 'jsonl.gz':
            for line in stream:
                yield json.loads(line)
        else:
            raise ValueError('Unsupported format ' + file_format)


def aggregate(spec):
    observations = defaultdict(lambda: defaultdict(list))
    seen = set()
    count = 0
    path = ROOT / spec['path']
    for row in read_rows(path, spec['format']):
        group = '|'.join(str(row[field]) for field in spec['group_fields'])
        query_id = str(row[spec['query_id_field']])
        identity = (group, query_id)
        if not query_id or identity in seen:
            raise ValueError('Missing or repeated query/method unit in ' + spec['name'])
        seen.add(identity)
        for field, metric in spec['metric_fields'].items():
            value = float(row[field])
            if not math.isfinite(value) or not -1e-12 <= value <= 1 + 1e-12:
                raise ValueError('Invalid ' + field + ' in ' + spec['name'])
            observations[group][metric].append(value)
        count += 1
    if count != spec['rows']:
        raise ValueError('Unexpected row count in ' + spec['name'])
    result = {}
    for group, metrics in observations.items():
        result[group] = {metric: math.fsum(values) / len(values) * 100 for metric, values in metrics.items()}
    return result, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    catalog = json.loads((ROOT / 'data/catalog.json').read_text())
    expected = json.loads((ROOT / 'data/expected_point_metrics.json').read_text())
    results = {}
    comparisons = 0
    total_rows = 0
    largest_error = 0.0
    for spec in catalog:
        actual, count = aggregate(spec)
        reference = expected[spec['name']]
        if actual.keys() != reference.keys():
            raise ValueError('Metric groups differ in ' + spec['name'])
        for group, metrics in reference.items():
            if actual[group].keys() != metrics.keys():
                raise ValueError('Metric columns differ in ' + spec['name'])
            for metric, value in metrics.items():
                error = abs(actual[group][metric] - value)
                if error > 1e-8:
                    raise ValueError(f'Metric mismatch in {spec["name"]} / {group} / {metric}')
                largest_error = max(largest_error, error)
                comparisons += 1
        results[spec['name']] = actual
        total_rows += count

    macro = {}
    for analysis in ['original_transfer', 'input_transfer', 'q7_targets']:
        points = results[analysis]
        fiqa = {name.split('|', 1)[1]: values for name, values in points.items() if name.startswith('fiqa|')}
        nfcorpus = {name.split('|', 1)[1]: values for name, values in points.items() if name.startswith('nfcorpus|')}
        if fiqa.keys() != nfcorpus.keys():
            raise ValueError('Original transfer methods differ between collections')
        macro[analysis] = {method: {metric: (fiqa[method][metric] + nfcorpus[method][metric]) / 2 for metric in fiqa[method]} for method in fiqa}

    expected_macro = json.loads((ROOT / 'data/expected_macro_metrics.json').read_text())
    for analysis, methods in expected_macro.items():
        for method, metrics in methods.items():
            for metric, value in metrics.items():
                error = abs(macro[analysis][method][metric] - value)
                if error > 1e-8:
                    raise ValueError('Equal-collection mean mismatch in ' + analysis)
                largest_error = max(largest_error, error)
                comparisons += 1

    highlights = {
        'source_MiniLM_request_minus_topic': results['source_input']['MiniLM_full']['P1'] - results['source_input']['MiniLM_topic']['P1'],
        'source_G_same12_minus_G12': results['source_input_matched']['G_same12']['P1'] - results['source_input_matched']['G12']['P1'],
        'original_macro_G_same12_minus_Q7': macro['q7_targets']['G_same12']['P1'] - macro['q7_targets']['Q7']['P1'],
    }
    output = {'status': 'PASS', 'rows': total_rows, 'metric_checks': comparisons, 'maximum_absolute_error_pp': largest_error, 'units': 'percent for point metrics, percentage points for differences', 'per_collection_points': results, 'FiQA_NFCorpus_equal_collection_means': macro, 'key_P1_differences_pp': highlights, 'scope': 'Point reconstruction from saved outcomes. CDS is separate from the original macro mean.'}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: output[key] for key in ['status', 'rows', 'metric_checks', 'maximum_absolute_error_pp', 'key_P1_differences_pp', 'scope']}, indent=2))


if __name__ == '__main__':
    main()
