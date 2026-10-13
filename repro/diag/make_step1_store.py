"""Store the main run's step-1 thinking for run.py --step1-from (think experiments, EF 2026-10-13).

In the main run (b_main_s1..s9) every method of a question shares one step-1 thinking z_1 (forked from the ST job),
so the ST row holds it: 'thinking' = decode(z_1) + separator, 'think_tokens' = len(z_1), steps[0]['think_end'].
This writes those three fields per pid; methods.stored_step1 turns them back into token ids.

  python3 diag/make_step1_store.py [--out diag/step1_store_bmain.json]
"""
import argparse
import glob
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = 'kaggle/runs/b_main_s[0-9]/output/results/results_*.jsonl'


def build(files):
    pids = {}
    for f in files:
        for line in Path(f).read_text().splitlines():
            r = json.loads(line)
            if r['method'] != 'st':
                continue
            assert r['pid'] not in pids, f"duplicate ST row for pid {r['pid']}"
            assert r['status'] == 'ok' and len(r['steps']) == 1, r['pid']
            assert r['think_tokens'] == r['steps'][0]['think_tokens'], r['pid']
            pids[r['pid']] = {'thinking': r['thinking'], 'think_tokens': r['think_tokens'],
                              'think_end': r['steps'][0]['think_end']}
    return pids


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=RUNS, help='glob of the main run result files, relative to repro/')
    ap.add_argument('--out', default='diag/step1_store_bmain.json')
    a = ap.parse_args(argv)
    files = sorted(glob.glob(str(ROOT / a.runs)))
    assert files, a.runs
    pids = build(files)
    doc = {'source': 'ST rows of the main run (step 1 is shared by all methods there)',
           'files_sha256': {str(Path(f).relative_to(ROOT)): hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
           'n': len(pids), 'pids': dict(sorted(pids.items(), key=lambda kv: int(kv[0])))}
    out = ROOT / a.out
    out.write_text(json.dumps(doc, ensure_ascii=False, separators=(',', ':')) + '\n')
    ends = {}
    for s in pids.values():
        ends[s['think_end']] = ends.get(s['think_end'], 0) + 1
    print(json.dumps({'out': a.out, 'n': len(pids), 'think_end': ends,
                      'sha256': hashlib.sha256(out.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
