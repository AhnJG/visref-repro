"""Question list of the 2026-10-17 format check (EF "동작 점검 계획"): the first N questions of the main run's seed-0
order whose stored step 1 the model closed itself (think_end == 'close'), the group where A6's empty answers occurred.
Chosen from the order and the stored step 1 only (no correctness).

  cd repro && python3 diag/format_check_ids.py            # writes diag/format_check_ids.json
  cd repro && python3 diag/format_check_ids.py --check    # exits 1 if the file differs from a fresh build
"""
import argparse
import json
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parent.parent
ORDER = REPRO / 'kaggle/runs/b_main_hs/output/results/ids.json'
STORE = REPRO / 'diag/step1_store_bmain.json'
OUT = REPRO / 'diag/format_check_ids.json'
N = 100


def build(order, store, n=N):
    """First n pids of order (as strings) whose stored step 1 ended with the model's own </think>."""
    pids = [str(p) for p in order if store[str(p)]['think_end'] == 'close'][:n]
    assert len(pids) == n, f'only {len(pids)} closed first steps'
    return pids


def text(pids):
    return json.dumps(pids) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args(argv)
    pids = build(json.loads(ORDER.read_text()), json.loads(STORE.read_text())['pids'])
    if a.check:
        ok = OUT.read_text() == text(pids)
        print('ok' if ok else 'differs')
        return 0 if ok else 1
    OUT.write_text(text(pids))
    print(f'{len(pids)} pids -> {OUT.relative_to(REPRO)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
