"""A lossless, allowlisted verifier-only name-shadow repair; no refitting/scoring."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    assert str(ROOT).startswith('/path/to/remote/lw_experiments/runs/')
    assert json.loads((ROOT / 'launcher_receipt.json').read_text())['exit_code'] == 1
    assert not (ROOT / 'verification.json').exists()
    directory = ROOT / 'verification_repair_20261103'
    directory.mkdir(exist_ok=False)
    (directory / 'original_status.txt').write_bytes((ROOT / 'status').read_bytes())
    source = (ROOT / 'verify.py').read_text()
    changes = [
        ("population, metric = contrast['population'], contrast['metric']", "population, metric_name = contrast['population'], contrast['metric']"),
        ("['P1', 'N1', 'U1', 'R10'].index(metric)", "['P1', 'N1', 'U1', 'R10'].index(metric_name)"),
        ("population == 'ExcluIR' and metric == 'P1'", "population == 'ExcluIR' and metric_name == 'P1'"),
        ("'code_sha256': sha(ROOT / 'verify.py')", "'code_sha256': sha(ROOT / 'verification_repair_20261103/verify_fixed.py')"),
    ]
    fixed = source
    for old, new in changes:
        assert fixed.count(old) == 1, old
        fixed = fixed.replace(old, new, 1)
    restored = fixed
    for old, new in reversed(changes):
        restored = restored.replace(new, old, 1)
    assert restored == source and ast.dump(ast.parse(restored)) == ast.dump(ast.parse(source))
    destination = directory / 'verify_fixed.py'
    destination.write_text(fixed)
    sealed = json.loads((ROOT / 'predictions_sealed.json').read_text())
    unchanged = {name: sha(ROOT / name) for name in sealed['files']}
    unchanged.update({name: sha(ROOT / name) for name in ['summary.json', 'frozen_manifest.json', 'verify.py', 'production_failed.json', 'launcher_receipt.json']})
    write(directory / 'repair_manifest.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
          'reason': 'Verifier-local metric variable shadowed the global metric function.',
          'allowed_text_changes': changes, 'reverse_bytes_and_AST_exact': True,
          'original_verifier_sha256': sha(ROOT / 'verify.py'), 'actual_verifier_sha256': sha(destination),
          'unchanged_before': unchanged, 'new_fits_or_scoring_runs': 0})
    started = time.monotonic()
    # Preserve the original ROOT via config; only sys.path gains the original code directory.
    command = [sys.executable, '-B', '-c',
               "import runpy,sys;sys.path.insert(0,sys.argv[1]);runpy.run_path(sys.argv[2],run_name='__main__')",
               str(ROOT), str(destination)]
    with (directory / 'verification.log').open('x') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    same = all(sha(ROOT / name) == expected for name, expected in unchanged.items())
    status = 'PASS' if result.returncode == 0 and same else 'FAILED'
    write(directory / 'recovery_receipt.json', {'status': status, 'exit_code': result.returncode,
          'original_pipeline_exit_code': 1, 'seconds': time.monotonic() - started,
          'finished_utc': datetime.now(timezone.utc).isoformat(), 'frozen_predictions_and_summary_unchanged': same,
          'actual_verifier_sha256': sha(destination), 'new_fits_or_scoring_runs': 0, 'new_neural_pairs': 0})
    if status == 'PASS':
        assert json.loads((ROOT / 'verification.json').read_text())['status'] == 'PASS'
        (ROOT / 'status').write_text('complete_verified_after_verifier_repair\n')
    raise SystemExit(result.returncode if result.returncode else (0 if same else 1))


if __name__ == '__main__':
    main()
