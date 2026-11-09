"""Score imported ExcluIR rankings; never construct candidates from references."""
import argparse
import csv
import json
import math
from pathlib import Path
from adapter import document_id, load, query_id, sha256, write_json


def read_rows(path: Path, fields: list) -> list:
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise ValueError(f'{path.name}: expected CSV header {fields}')
        rows = list(reader)
    if any(set(row) != set(fields) or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f'{path.name}: malformed CSV row')
    return rows


def read_run(path: Path, format_name: str) -> list:
    if format_name == 'csv':
        return read_rows(path, ['query_id', 'document_id', 'score'])
    rows = []
    with path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 6 or fields[1] != 'Q0' or not fields[3].isdigit() or int(fields[3]) < 1:
                raise ValueError(f'Invalid TREC row {line_number}')
            rows.append(dict(zip(('query_id', 'document_id', 'score'),
                                 (fields[0], fields[2], fields[4]))))
    return rows


def bind_rows(rows: list, query_ids: set, corpus_count: int, scored: bool) -> dict:
    result = {qid: {} for qid in query_ids}
    for row in rows:
        qid, did = row['query_id'], row['document_id']
        if qid not in query_ids:
            raise ValueError(f'Unknown query ID: {qid}')
        if not isinstance(did, str) or len(did) < 2 or not did[1:].isdigit():
            raise ValueError(f'Invalid document ID: {did}')
        index = int(did[1:])
        if index >= corpus_count or did != document_id(index):
            raise ValueError(f'Unknown/noncanonical document ID: {did}')
        if did in result[qid]:
            raise ValueError(f'Duplicate query/document row: {qid}/{did}')
        if scored:
            score = float(row['score'])
            if not math.isfinite(score):
                raise ValueError(f'Nonfinite score: {qid}/{did}')
            result[qid][did] = score
        else:
            result[qid][did] = None
    return result


def evaluate(refs: dict, scores: dict, ks: list, candidates=None) -> tuple:
    if not refs or set(scores) != set(refs):
        raise ValueError('Reference/run planned query IDs differ')
    records = []
    for qid in sorted(refs):
        row = refs[qid]
        if set(row) != {'negative_id', 'positive_id'} or row['negative_id'] == row['positive_id']:
            raise ValueError(f'Invalid reference pair {qid}')
        positive, negative = row['positive_id'], row['negative_id']
        query_scores = scores[qid]
        if candidates is not None and not set(query_scores).issubset(candidates[qid]):
            raise ValueError(f'Reranking run contains documents outside actual candidate pool: {qid}')
        ordered = sorted(query_scores, key=lambda did: (-query_scores[did], did))
        positions = {did: i + 1 for i, did in enumerate(ordered)}
        p, n = positions.get(positive), positions.get(negative)
        both = p is not None and n is not None
        entry = {'query_id': qid, 'returned_documents': len(ordered),
                 'query_returned': bool(ordered), 'positive_scored': p is not None,
                 'negative_scored': n is not None, 'positive_rank': p, 'negative_rank': n,
                 'both_scored': both,
                 'pairwise_strict_win': (query_scores[positive] > query_scores[negative]) if both else None,
                 'pairwise_score_tie': (query_scores[positive] == query_scores[negative]) if both else None,
                 'pairwise_credit': ((1.0 if query_scores[positive] > query_scores[negative] else
                                      0.5 if query_scores[positive] == query_scores[negative] else 0.0)
                                     if both else None)}
        for k in ks:
            entry[f'positive_recall@{k}'] = float(p is not None and p <= k)
            entry[f'negative_entry@{k}'] = float(n is not None and n <= k)
            entry[f'delta_recall@{k}'] = entry[f'positive_recall@{k}'] - entry[f'negative_entry@{k}']
            entry[f'returned_at_least_{k}'] = len(ordered) >= k
            phit, nhit = p is not None and p <= k, n is not None and n <= k
            entry[f'topk_both@{k}'] = phit and nhit
            entry[f'topk_positive_only@{k}'] = phit and not nhit
            entry[f'topk_negative_only@{k}'] = nhit and not phit
            entry[f'topk_neither@{k}'] = not phit and not nhit
        entry['positive_mrr@10'] = 1 / p if p is not None and p <= 10 else 0.0
        entry['negative_mrr@10'] = 1 / n if n is not None and n <= 10 else 0.0
        entry['delta_mrr@10'] = entry['positive_mrr@10'] - entry['negative_mrr@10']
        # Exact author compute_right_rank semantics applied to the returned top10.
        entry['author_right_rank@10'] = float(p is not None and p <= 10
                                             and (n is None or n > 10 or p < n))
        if candidates is not None:
            pool = candidates[qid]
            entry.update(candidate_documents=len(pool), candidate_query_present=bool(pool),
                         positive_in_candidates=positive in pool, negative_in_candidates=negative in pool,
                         both_in_candidates=positive in pool and negative in pool,
                         scored_candidate_documents=len(query_scores),
                         unscored_candidate_documents=len(pool) - len(query_scores),
                         ranking_complete=len(pool) == len(query_scores),
                         observed_scored_subset_positive_rank=p,
                         observed_scored_subset_negative_rank=n)
            if not entry['ranking_complete']:
                # A missing competitor can inflate rank: never publish this as a complete rerank.
                entry['positive_rank'] = None
                entry['negative_rank'] = None
                for key in list(entry):
                    if key.startswith(('positive_recall@', 'negative_entry@', 'delta_recall@',
                                       'positive_mrr@', 'negative_mrr@', 'delta_mrr@',
                                       'author_right_rank@', 'pairwise_', 'topk_')):
                        entry[key] = None
        records.append(entry)
    count = len(records)
    def mean(key):
        return sum(row[key] for row in records) / count
    metric_keys = [key for key in records[0] if key.startswith(
        ('positive_recall@', 'negative_entry@', 'delta_recall@', 'positive_mrr@',
         'negative_mrr@', 'delta_mrr@', 'author_right_rank@'))]
    both_count = sum(row['both_scored'] for row in records)
    incomplete_count = sum(not row.get('ranking_complete', True) for row in records)
    aggregate = {'planned_queries': count, 'queries_with_any_scores': sum(r['query_returned'] for r in records),
                 'missing_query_results': sum(not r['query_returned'] for r in records),
                 'query_result_coverage': mean('query_returned'),
                 'positive_scored_coverage': mean('positive_scored'),
                 'negative_scored_coverage': mean('negative_scored'),
                 'both_scored_queries': both_count, 'both_scored_coverage': mean('both_scored'),
                 'pairwise_credit_given_both_scored': (sum(r['pairwise_credit'] or 0 for r in records)
                                                      / both_count if both_count and not incomplete_count else None),
                 'pairwise_strict_win_given_both_scored': (sum(bool(r['pairwise_strict_win']) for r in records)
                                                          / both_count if both_count and not incomplete_count else None),
                 'pairwise_ties_given_both_scored': (sum(bool(r['pairwise_score_tie']) for r in records)
                                                     if not incomplete_count else None),
                 'evaluation_status': 'incomplete_candidate_scores' if incomplete_count else 'ranking_metrics_available',
                 'incomplete_candidate_queries': incomplete_count,
                 'ranking_metrics_all_planned_queries': ({key: mean(key) for key in metric_keys}
                                                          if not incomplete_count else None),
                 'queries_returning_fewer_than_k': {str(k): sum(r['returned_documents'] < k for r in records)
                                                   for k in ks}}
    aggregate['topk_reference_coverage_counts'] = (
        {str(k): {state: sum(row[f'topk_{state}@{k}'] for row in records)
                  for state in ('both', 'positive_only', 'negative_only', 'neither')}
         for k in ks} if not incomplete_count else None)
    if candidates is not None:
        total = sum(r['candidate_documents'] for r in records)
        scored = sum(r['scored_candidate_documents'] for r in records)
        aggregate['candidate_pool'] = {'queries_with_candidates': sum(r['candidate_query_present'] for r in records),
                                      'documents': total, 'scored_documents': scored,
                                      'missing_scores': total - scored,
                                      'score_coverage': scored / total if total else None,
                                      'positive_recall_ceiling': mean('positive_in_candidates'),
                                      'negative_candidate_entry_rate': mean('negative_in_candidates'),
                                      'both_reference_candidate_coverage': mean('both_in_candidates')}
    return records, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--format', choices=('csv', 'trec'), default='csv')
    parser.add_argument('--run-manifest', type=Path, required=True)
    parser.add_argument('--candidates', type=Path)
    parser.add_argument('--ks', type=int, nargs='+', default=[1, 5, 10, 100])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ks = sorted(set(args.ks))
    if not ks or any(k < 1 for k in ks):
        raise ValueError('All k must be positive')
    meta = load(args.bundle / 'metadata.json')
    refs_path = args.bundle / 'references.json'
    if sha256(refs_path) != meta['export_sha256']['references.json']:
        raise ValueError('Reference SHA256 differs from bundle metadata')
    refs = load(refs_path)
    expected = {query_id(i) for i in range(meta['queries'])}
    if set(refs) != expected:
        raise ValueError('Missing or unexpected reference query IDs')
    # Verify both reference IDs are legal without producing candidates from them.
    bind_rows([{'query_id': qid, 'document_id': value}
               for qid, row in refs.items() for value in row.values()], expected, meta['documents'], False)
    manifest = load(args.run_manifest)
    if manifest.get('stage') not in ('full_corpus_retrieval', 'candidate_reranking'):
        raise ValueError('Declare stage full_corpus_retrieval or candidate_reranking')
    if manifest.get('query_variant') not in ('question0', 'RQ_rewrite'):
        raise ValueError('Declare the exact query variant')
    for key in ('engine', 'environment'):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f'Missing provenance {key}')
    if manifest.get('source_sha256') != meta['source_sha256']:
        raise ValueError('Run source hashes differ from data bundle')
    if manifest.get('references_used_for_candidates') is not False:
        raise ValueError('Run must declare that references were not used to inject candidates')
    pool = None
    if manifest['stage'] == 'candidate_reranking':
        if args.candidates is None:
            raise ValueError('Reranking requires the actual pre-rerank candidate CSV')
        if manifest.get('candidate_sha256') != sha256(args.candidates):
            raise ValueError('Candidate CSV differs from declared SHA256')
        pool = bind_rows(read_rows(args.candidates, ['query_id', 'document_id']), expected,
                         meta['documents'], False)
    elif args.candidates is not None or manifest.get('search_space_document_count') != meta['documents']:
        raise ValueError('Full-corpus stage requires declared complete corpus search space, no candidate file')
    scores = bind_rows(read_run(args.run, args.format), expected, meta['documents'], True)
    records, aggregate = evaluate(refs, scores, ks, pool)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'per_query.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {'stage': manifest['stage'], 'query_variant': manifest['query_variant'],
               'aggregate': aggregate, 'run_manifest': manifest,
               'input_sha256': {'run': sha256(args.run), 'run_manifest': sha256(args.run_manifest),
                                'bundle_metadata': sha256(args.bundle / 'metadata.json'),
                                'references': sha256(refs_path)},
               'ordering': 'score descending, canonical document ID ascending for score ties; TREC rank ignored',
               'pairwise_ties': '0.5 credit when both scores available; missing pair excluded with coverage reported',
               'provenance_boundary': 'search space/no-reference-injection are producer declarations, not provable from ranking file',
               'interpretation': 'one known positive and negative per query; unjudged corpus docs are not labelled irrelevant; no FollowIR p-MRR/MAP/nDCG inferred'}
    write_json(args.output / 'summary.json', summary)
    print(json.dumps({'output': str(args.output), 'planned_queries': len(records),
                      'queries_with_any_scores': aggregate['queries_with_any_scores']}))


if __name__ == '__main__':
    main()
