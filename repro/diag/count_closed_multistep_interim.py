"""Interim counter of `closed_multistep` (plan section 5, "중간 멈춤"; committed before any result).

Reads the result rows of part 1 (the first 112 questions of diag/closed_multistep_ids.json) and prints ONLY
  - the number of questions on which V and T differ in their 3-sample-mean accuracy (the count of correct answers among the 3
    sampled answers differs), with no direction and no accuracy, and
  - the stop decision of the plan: at most 1 such question -> stop (the other 222 questions are not run, verdict 3 with the
    sentence "중간 멈춤 규칙으로 112문항에서 멈춤"); 2 or more -> run part 2.
It refuses (exit 2, printing only how many part-1 rows are missing or not complete) when a part-1 question has no complete ok row,
and refuses a pid that appears twice. Nothing else is computed or printed.

  cd repro && python3 diag/count_closed_multistep_interim.py 'results/closed_multistep_*.jsonl' [--ids diag/closed_multistep_ids.json]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parent.parent
STOP_AT_MOST = 1


def load_rows(patterns):
    files = sorted({g for p in patterns for g in (glob.glob(p) or ([p] if Path(p).exists() else []))})
    if not files:
        raise SystemExit('no result files match')
    rows = {}
    for f in files:
        for line in Path(f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                pid = str(r['pid'])
                if pid in rows:
                    raise SystemExit(f'duplicate pid {pid}')
                rows[pid] = r
    return rows


def n_correct(r, c):
    return sum(int(a['correct']) for a in r['answers'][c] if not a.get('greedy'))


def usable(r):
    try:
        n = int(r['cfg']['n_t1'])
        return r.get('status') == 'ok' and all(len(r['answers'][c]) == n + 1 for c in ('T', 'V'))
    except (KeyError, TypeError):
        return False


def count(rows, part1):
    """(number of part-1 questions on which V and T differ, number of part-1 questions without a usable row)."""
    bad = sum(not (p in rows and usable(rows[p])) for p in part1)
    changed = sum(n_correct(rows[p], 'V') != n_correct(rows[p], 'T') for p in part1 if p in rows and usable(rows[p]))
    return changed, bad


def decision(changed):
    return 'stop' if changed <= STOP_AT_MOST else 'run part 2'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='+')
    ap.add_argument('--ids', default=str(REPRO / 'diag/closed_multistep_ids.json'))
    a = ap.parse_args(argv)
    part1 = [str(p) for p in json.loads(Path(a.ids).read_text())['part1']]
    changed, bad = count(load_rows(a.results), part1)
    if bad:
        print(f'part 1 incomplete: {bad} of {len(part1)} questions have no usable row')
        return 2
    print(f'questions with V and T different: {changed}')
    print(f'decision: {decision(changed)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
