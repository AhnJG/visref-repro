"""Question list of the 2026-10-17 format check (EF "동작 점검 계획"): the first N questions of the main run's seed-0
order, whatever way their stored step 1 ended (closed, cut or EOS), because A6's extra </think> came mostly after cut
steps. Chosen from the order only (no correctness).

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
N = 200


def build(order, store, n=N):
    """First n pids of order (as strings); every one must have a stored step 1."""
    pids = [str(p) for p in order][:n]
    assert len(pids) == n and all(p in store for p in pids), 'order too short or a stored step 1 missing'
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
