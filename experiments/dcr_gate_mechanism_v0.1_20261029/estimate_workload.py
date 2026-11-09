"""CPU arithmetic projection from full prior run and actual complete token inventory."""
import gzip
import json
from pathlib import Path
from adapter import write_json
from prepare_external_neural import ROOT, PREV


def inventory(path, group_key, length_key, batch_size):
    count, batches, padded_tokens, attention_units = 0, 0, 0, 0
    previous, pending = None, []
    def add(values):
        nonlocal batches, padded_tokens, attention_units
        if not values:
            return
        maximum = ((max(values) + 7) // 8) * 8
        batches += 1
        padded_tokens += len(values) * maximum
        attention_units += len(values) * maximum * maximum
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            group = tuple(row[k] for k in group_key)
            if previous is not None and group != previous:
                add(pending)
                pending = []
            previous = group
            pending.append(row[length_key])
            count += 1
            if len(pending) == batch_size:
                add(pending)
                pending = []
        add(pending)
    return {'pairs': count, 'batches': batches, 'padded_tokens': padded_tokens,
            'attention_length_squared_units': attention_units, 'batch_size': batch_size}


def main():
    old = inventory(PREV / 'qwen_results/scores.jsonl.gz', ['qid', 'role'], 'input_tokens', 16)
    new = inventory(ROOT / 'external_prepared/pairs.jsonl.gz', ['query_id'], 'qwen_input_tokens', 16)
    observed = json.loads((PREV / 'qwen_results/run_manifest.json').read_text())['inference_wall_seconds']
    linear = new['padded_tokens'] / old['padded_tokens']
    quadratic = new['attention_length_squared_units'] / old['attention_length_squared_units']
    write_json(ROOT / 'external_budget.json', {'prior_full_Qwen': old, 'external_Qwen': new,
        'prior_full_actual_seconds': observed, 'padded_token_ratio': linear,
        'attention_units_ratio': quadratic, 'projected_seconds_linear_scenario': observed * linear,
        'projected_seconds_attention_scenario': observed * quadratic,
        'projected_seconds_unchanged_pairs_speed': observed * new['pairs'] / old['pairs'],
        'interpretation': 'Arithmetic scenarios from full token inventory, not elapsed external inference; per-batch/query overhead and GPU scheduling can dominate. Actual full execution logs supersede all projections.',
        'models_loaded': False, 'GPU_used': False})
    print(json.dumps({'padded_token_ratio': linear, 'attention_ratio': quadratic,
                      'linear_scenario_seconds': observed * linear}), flush=True)


if __name__ == '__main__':
    main()
