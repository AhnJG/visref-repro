"""Question lists of `blank_think` (plan docs/EXPERIMENT_FLOW.md 2026-10-24 "실행 전 계획: 빈 그림 생성 양성 대조", written before any
result). Chosen from the main run's seed-0 order and the image-token counts only (no correctness).

  eligible  the 1,000 questions of the main run's order (kaggle/runs/b_main_hs/output/results/ids.json) that have at least one
            other question with exactly the same number of image tokens (n_visual, llm/final/merged.jsonl); the others are
            listed in 'dropped'
  trial     the last 5 eligible questions of the order; when none of them has n_visual 3328, the 5th is replaced by the last
            eligible question with n_visual 3328 not yet in the list (memory check of the longest context)
  main      eligible minus trial, in order. The run is sharded by index (i % 4) over two kernels of two GPUs and stops starting
            new groups at its time limit, so the analysed questions are the ones finished in this fixed order
  donor     the "other" image of condition X: the next eligible question in order (cyclic scan from the question's own
            position) with the same n_visual (recover_check_ids.partners' first donor, over the eligible list)

  cd repro && python3 diag/blank_think_ids.py            # writes diag/blank_think_ids.json
  cd repro && python3 diag/blank_think_ids.py --check    # exits 1 if the file differs from a fresh build
"""
import argparse
import json
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO / 'diag'))
from recover_check_ids import n_visual_of  # noqa: E402

ORDER = REPRO / 'kaggle/runs/b_main_hs/output/results/ids.json'
OUT = REPRO / 'diag/blank_think_ids.json'
N_TRIAL, BIG = 5, 3328


def donors(lst, nvis):
    """{pid: donor} for the pids of lst with another same-size pid in lst: the next one in a cyclic scan from the pid."""
    out, n = {}, len(lst)
    for i, p in enumerate(lst):
        q = next((lst[(i + k) % n] for k in range(1, n) if nvis[lst[(i + k) % n]] == nvis[p]), None)
        if q is not None:
            out[p] = q
    return out


def build(order, nvis):
    order = [str(p) for p in order]
    assert len(order) == len(set(order)) and all(p in nvis for p in order), 'order has repeats or missing n_visual'
    don = donors(order, nvis)
    elig = [p for p in order if p in don]
    dropped = [{'pid': p, 'n_visual': nvis[p], 'reason': f'no other question with {nvis[p]} image tokens'}
               for p in order if p not in don]
    trial = elig[-N_TRIAL:]
    if not any(nvis[p] == BIG for p in trial):
        big = next(p for p in reversed(elig) if nvis[p] == BIG and p not in trial)
        trial = trial[:-1] + [big]
    main = [p for p in elig if p not in set(trial)]
    return {'experiment': 'blank_think', 'main': main, 'trial': trial, 'dropped': dropped,
            'memcheck': next(p for p in trial if nvis[p] == BIG),
            'n_visual': {p: nvis[p] for p in elig}, 'donor': {p: don[p] for p in elig}}


def text(doc):
    return json.dumps(doc, separators=(',', ':')) + '\n'


def fresh():
    return build(json.loads(ORDER.read_text()), n_visual_of())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args(argv)
    t = text(fresh())
    if a.check:
        ok = OUT.exists() and OUT.read_text() == t
        print('ok' if ok else 'differs')
        return 0 if ok else 1
    OUT.write_text(t)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
