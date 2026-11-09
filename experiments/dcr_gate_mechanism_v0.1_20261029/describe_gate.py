"""Posthoc all-query gate saturation/response description; never change the frozen gate."""
import csv
import gzip
import json
from pathlib import Path
from adapter import write_json

ROOT = Path(__file__).resolve().parent


def main():
    rows = {}
    with gzip.open(ROOT / 'robust_results/gates.jsonl.gz', 'rt') as stream:
        for line in stream:
            r = json.loads(line)
            rows.setdefault(r['qid'], {}).setdefault(r['side'], []).append(r)
    output = []
    for qid, sides in rows.items():
        record = {'qid': qid}
        for name in ('lexical_gate', 'Qwen_gate'):
            sets = {side: {r['pid'] for r in values if r[name]} for side, values in sides.items()}
            record.update({name + '_og_demoted': len(sets['og']), name + '_changed_demoted': len(sets['changed']),
                           name + '_membership_changes': len(sets['og'] ^ sets['changed']),
                           name + '_identical_on_sides': sets['og'] == sets['changed']})
        output.append(record)
    with (ROOT / 'robust_results/gate_response_by_query.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    summary = {'queries': len(output), 'analysis': 'posthoc descriptive, not threshold tuning; all52 included',
        'gate_target': 'conditional relevance, absolute no>yes threshold fixed before this analysis',
        'source': 'robust_results/gates.jsonl.gz',
        'statistics': {name: {'identical_membership_between_sides_queries': sum(r[name + '_identical_on_sides'] for r in output),
            'all_demoted_og_queries': sum(r[name + '_og_demoted'] == 1000 for r in output),
            'all_demoted_changed_queries': sum(r[name + '_changed_demoted'] == 1000 for r in output),
            'none_demoted_og_queries': sum(r[name + '_og_demoted'] == 0 for r in output),
            'none_demoted_changed_queries': sum(r[name + '_changed_demoted'] == 0 for r in output),
            'mean_og_demoted_fraction': sum(r[name + '_og_demoted'] for r in output) / 52000,
            'mean_changed_demoted_fraction': sum(r[name + '_changed_demoted'] for r in output) / 52000}
            for name in ('lexical_gate', 'Qwen_gate')},
        'interpretation': 'All demoted/none demoted leaves R unchanged in that condition. Equal counts alone do not imply equal membership. Absolute relevance score sign is not calibrated violation probability; no threshold adjustment follows this observation.'}
    write_json(ROOT / 'robust_results/gate_response_summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
