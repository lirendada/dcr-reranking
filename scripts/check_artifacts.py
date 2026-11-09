"""Check the complete published file set against its release manifest."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'provenance/release_manifest.json').read_text())
    total_bytes = 0
    for name, expected in manifest['files'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError('Missing or invalid release file ' + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('SHA-256 mismatch for ' + name)
        total_bytes += path.stat().st_size
    print(json.dumps({'status': 'PASS', 'files': len(manifest['files']), 'bytes': total_bytes}, indent=2))


if __name__ == '__main__':
    main()
