"""Download the third-party files listed in references/SOURCES.json and verify sha256.

  python3 scripts/fetch_sources.py

Needed for offline scoring: data/raw/mathvista_testmini.json (MathVista, CC BY-SA 4.0) and the official MathVista
answer-extraction prompt (references/sources/mathvista_official_r4/evaluation/prompts/ext_ans.py). The other files are
the cited reference code used as evidence in docs/SPEC_LEDGER.md.
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
bad = 0
for rel, v in json.loads((ROOT / 'references' / 'SOURCES.json').read_text()).items():
    dst = ROOT / rel
    if not (dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() == v['sha256']):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(urllib.request.urlopen(v['url'], timeout=60).read())
    ok = hashlib.sha256(dst.read_bytes()).hexdigest() == v['sha256']
    bad += not ok
    print(('ok      ' if ok else 'MISMATCH'), rel)
sys.exit(1 if bad else 0)
