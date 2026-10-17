"""Question list of step1_think (EXPERIMENT_FLOW 2026-10-18 "실행 전 계획: 첫 단계에서 멈춘 문항을 더 생각하게 하기").

pids where, in the main run (kaggle/runs/b_main_s*/output/results/results_*.jsonl), BOTH 'tsr' and 'visref_raw_rank'
stopped after the first step (k == 1). Chosen from the stopping decision only (no correctness). Ordered by the seed-0
order of the main run (kaggle/runs/b_main_hs/output/results/ids.json, the order diag/format_check_ids.py uses), so
that a run cut short still holds a random prefix.

Output diag/step1_think_ids.json (a dict; step1_branch-style readers take its 'ids'):
  {'rule', 'order_file', 'n', 'ids': [pid str...], 'strata_counts': {...},
   'stratum': {pid: think_end of the ST row's steps[0] ('close' | 'cap' | 'eos')}}
'stratum' is the end of the shared step 1 (= diag/step1_store_bmain.json think_end; asserted equal).

  cd repro && python3 diag/step1_think_ids.py            # writes diag/step1_think_ids.json
  cd repro && python3 diag/step1_think_ids.py --check    # exits 1 if the file differs from a fresh build
"""
import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

REPRO = Path(__file__).resolve().parent.parent
RUNS = 'kaggle/runs/b_main_s*/output/results/results_*.jsonl'
ORDER = REPRO / 'kaggle/runs/b_main_hs/output/results/ids.json'
STORE = REPRO / 'diag/step1_store_bmain.json'
OUT = REPRO / 'diag/step1_think_ids.json'
RULE = "pids with k == 1 for both 'tsr' and 'visref_raw_rank' in the main run, in seed-0 order"


def load_rows(repro=REPRO, pattern=RUNS):
    """{pid: {method: row}} over the main-run result files; a (pid, method) pair seen twice is an error."""
    rows = {}
    files = sorted(glob.glob(str(repro / pattern)))
    assert files, 'no main-run result files'
    for f in files:
        for line in Path(f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                d = rows.setdefault(str(r['pid']), {})
                assert r['method'] not in d, f"duplicate row {r['pid']} {r['method']}"
                d[r['method']] = r
    return rows


def build(order, rows, store):
    """(ids, stratum): the pids of `order` where tsr and visref_raw_rank both have k == 1, and each one's step-1 end."""
    order = [str(p) for p in order]
    assert len(set(order)) == len(order), 'duplicate pid in order'
    assert set(order) == set(rows), 'order and main-run rows cover different pids'
    ids, stratum = [], {}
    for p in order:
        d = rows[p]
        if not (int(d['tsr']['k']) == 1 and int(d['visref_raw_rank']['k']) == 1):
            continue
        assert p in store, f'no stored step 1 for {p}'
        end = d['st']['steps'][0]['think_end']
        assert end == store[p]['think_end'], (p, end, store[p]['think_end'])
        assert end in ('close', 'cap', 'eos'), (p, end)
        ids.append(p)
        stratum[p] = end
    return ids, stratum


def text(ids, stratum, order_file=ORDER, repro=REPRO):
    try:
        of = str(Path(order_file).relative_to(repro))
    except ValueError:
        of = str(order_file)
    return json.dumps({'rule': RULE, 'order_file': of, 'n': len(ids), 'ids': ids,
                       'strata_counts': dict(sorted(Counter(stratum.values()).items())), 'stratum': stratum},
                      ensure_ascii=False) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args(argv)
    store = json.loads(STORE.read_text())['pids']
    ids, stratum = build(json.loads(ORDER.read_text()), load_rows(), store)
    new = text(ids, stratum)
    if a.check:
        ok = OUT.exists() and OUT.read_text() == new
        print('ok' if ok else 'differs')
        return 0 if ok else 1
    OUT.write_text(new)
    print(f'{len(ids)} pids {dict(sorted(Counter(stratum.values()).items()))} -> {OUT.relative_to(REPRO)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
