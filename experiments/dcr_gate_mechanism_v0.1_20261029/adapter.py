"""ExcluIR published-schema adapter. No retrieval, model or reference injection."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, value) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def query_id(index: int) -> str:
    return f'q{index:06d}'


def document_id(index: int) -> str:
    return f'd{index:06d}'


def validate(queries: list, corpus: list) -> dict:
    if not isinstance(queries, list) or not queries:
        raise ValueError('Queries must be a nonempty JSON list')
    if not isinstance(corpus, list) or not corpus:
        raise ValueError('Corpus must be a nonempty JSON list')
    if any(not isinstance(text, str) or not text.strip() for text in corpus):
        raise ValueError('Every corpus item must be a nonempty string')
    fields = Counter()
    for index, row in enumerate(queries):
        if not isinstance(row, dict) or set(row) != {'question0', 'RQ_rewrite', 'corpus_sub_index'}:
            raise ValueError(f'Unexpected query schema at row {index}')
        fields[tuple(sorted(row))] += 1
        if any(not isinstance(row[key], str) or not row[key].strip()
               for key in ('question0', 'RQ_rewrite')):
            raise ValueError(f'Empty/nonstring query at row {index}')
        pair = row['corpus_sub_index']
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(i) is not int or i < 0 or i >= len(corpus) for i in pair)
                or pair[0] == pair[1]):
            raise ValueError(f'Invalid negative/positive pair at row {index}')
    return {'queries': len(queries), 'documents': len(corpus),
            'query_fields': [{'keys': list(keys), 'count': n} for keys, n in fields.items()],
            'index_base': 0, 'pair_order': ['negative', 'positive'],
            'query_id_scheme': 'q + zero-based row index padded to 6 digits',
            'document_id_scheme': 'd + zero-based corpus index padded to 6 digits',
            'inspection': 'schema/index/string-validity only; no task text printed or predicted'}


def export(queries: list, corpus: list, output: Path, source_hashes: dict) -> dict:
    audit = validate(queries, corpus)
    output.mkdir(parents=True, exist_ok=False)
    # Model-facing products contain no reference pair/label/stratum fields.
    for variant in ('question0', 'RQ_rewrite'):
        with (output / f'queries_{variant}.jsonl').open('x', encoding='utf-8') as stream:
            for i, row in enumerate(queries):
                stream.write(json.dumps({'query_id': query_id(i), 'text': row[variant]},
                                        ensure_ascii=False) + '\n')
    with (output / 'corpus.jsonl').open('x', encoding='utf-8') as stream:
        for i, text in enumerate(corpus):
            stream.write(json.dumps({'document_id': document_id(i), 'text': text},
                                    ensure_ascii=False) + '\n')
    refs = {query_id(i): {'negative_id': document_id(row['corpus_sub_index'][0]),
                         'positive_id': document_id(row['corpus_sub_index'][1])}
            for i, row in enumerate(queries)}
    write_json(output / 'references.json', refs)
    files = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    meta = {'schema_version': 1, **audit, 'source_sha256': source_hashes,
            'export_sha256': files, 'model_inputs': ['queries_question0.jsonl',
                                                   'queries_RQ_rewrite.jsonl', 'corpus.jsonl'],
            'scorer_only': ['references.json']}
    write_json(output / 'metadata.json', meta)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('audit', 'export'))
    parser.add_argument('--queries', required=True, type=Path)
    parser.add_argument('--corpus', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    queries, corpus = load(args.queries), load(args.corpus)
    hashes = {'test_manual_final.json': sha256(args.queries), 'corpus.json': sha256(args.corpus)}
    if args.action == 'audit':
        result = {**validate(queries, corpus), 'source_sha256': hashes}
        write_json(args.output, result)
    else:
        result = export(queries, corpus, args.output, hashes)
    print(json.dumps({'action': args.action, 'queries': result['queries'],
                      'documents': result['documents'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
