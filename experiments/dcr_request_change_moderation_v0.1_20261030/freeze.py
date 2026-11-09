#!/usr/bin/env python3
"""Hash inputs and code before analysis; write frozen_manifest.json."""
import hashlib, json, sys, os, datetime
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""): h.update(ch)
    return h.hexdigest()
if __name__ == "__main__":
    out = sys.argv[1]; files = sys.argv[2:]
    man = {"frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "files": {os.path.relpath(p, os.path.dirname(out)) if p.startswith(os.path.dirname(out)) else p:
                     {"sha256": sha(p), "bytes": os.path.getsize(p)} for p in files}}
    json.dump(man, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps({"frozen": len(files), "manifest": out}))
