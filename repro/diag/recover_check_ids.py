"""Question lists of the recovery check `recover_check` (plan /home/claude/plan_recover.md, "회복 시험", written before any
result). Chosen from the main run's seed-0 order and the image-token counts only (no correctness).

  main     inject_sensitivity_ids.json 'part1' (the first 200 questions of the order, same order) minus the questions that
           do not have two other questions of exactly the same number of image tokens (n_visual, llm/final/merged.jsonl)
           in that list; the left-out ones are listed in 'dropped' with their reason
  donor1   the picture shown first (B, C, D conditions): the next question in list order (cyclic) with the same n_visual
  donor2   the picture re-inserted as the "other" one: after donor1, the next question in the same cyclic scan with the
           same n_visual, distinct from the question itself and from donor1
  trial    6 questions OUTSIDE the 200: the pool is the main-run order from position 201 on (0-based index >= 200); within
           the pool a question is eligible when two other pool questions have its n_visual; the trial list is the first 4
           eligible pool questions, then the first eligible one with n_visual 3328 and the first eligible one with
           n_visual 256 that are not yet in the list. Donors of a trial question come from the eligible part of the pool
           (same cyclic rule, scan order = pool order); n_visual 3328 and 256 have hundreds of questions, so the pool is
           never short of partners for them
  subset   the random 30% of the image tokens that the Cn/Co/Ce and Dn/Do/De conditions re-insert: for every main and
           trial question floor(0.3 * n_visual) distinct indices of range(n_visual), sorted, drawn by
           random.Random(seed_of(42, pid, 'recover_sub')).sample (the rule of step1_think.random_sel with the label
           'recover_sub'); stored here so that the run can assert it recomputes the same ones

  cd repro && python3 diag/recover_check_ids.py            # writes diag/recover_check_ids.json
  cd repro && python3 diag/recover_check_ids.py --check    # exits 1 if the file differs from a fresh build
"""
import argparse
import json
import random
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO / 'diag'))
from step1_think import seed_of  # noqa: E402

ORDER = REPRO / 'kaggle/runs/b_main_hs/output/results/ids.json'
PART1 = REPRO / 'diag/inject_sensitivity_ids.json'
MERGED = REPRO / 'llm/final/merged.jsonl'
OUT = REPRO / 'diag/recover_check_ids.json'
SEED, FRAC, LABEL = 42, 0.3, 'recover_sub'
N_MAIN_ORDER = 200
TRIAL_FIRST, TRIAL_SIZES = 4, (3328, 256)


def n_visual_of(path=MERGED):
    """{pid: number of image tokens} from the main run's merged rows (the three methods agree for every pid)."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get('n_visual') is not None:
            assert out.setdefault(str(r['pid']), r['n_visual']) == r['n_visual'], r['pid']
    return out


def subset_size(n_visual):
    """floor(0.3 * n_visual), in integer arithmetic (no float rounding at multiples of 10)."""
    return 3 * n_visual // 10


def subset_of(pid, n_visual, seed=SEED):
    """The re-inserted subset of pid's image tokens: sorted, distinct, a function of (seed, pid, n_visual) only."""
    return sorted(random.Random(seed_of(seed, pid, LABEL)).sample(range(n_visual), subset_size(n_visual)))


def partners(lst, nvis):
    """{pid: (donor1, donor2)} for the pids of lst that have two other same-size pids in lst, scanning lst cyclically
    from the pid's own position; the second donor is the next same-size pid after the first one."""
    out, n = {}, len(lst)
    for i, p in enumerate(lst):
        same = [q for q in (lst[(i + k) % n] for k in range(1, n)) if nvis[q] == nvis[p]]
        if len(same) >= 2:
            out[p] = (same[0], same[1])
    return out


def build(order, part1, nvis):
    order, part1 = [str(p) for p in order], [str(p) for p in part1]
    assert order[:len(part1)] == part1 and len(part1) == N_MAIN_ORDER, 'part1 is not the start of the order'
    assert all(p in nvis for p in order), 'n_visual missing'
    pr = partners(part1, nvis)
    main = [p for p in part1 if p in pr]
    dropped = [{'pid': p, 'n_visual': nvis[p],
                'reason': f'fewer than two other questions with {nvis[p]} image tokens among the 200'}
               for p in part1 if p not in pr]
    pool = order[len(part1):]
    tp = partners(pool, nvis)
    elig = [p for p in pool if p in tp]
    trial = elig[:TRIAL_FIRST]
    for size in TRIAL_SIZES:
        trial.append(next(p for p in elig if nvis[p] == size and p not in trial))
    donors = {**pr, **tp}
    keep = main + trial
    return {'experiment': 'recover_check', 'seed': SEED, 'subset_frac': FRAC, 'subset_label': LABEL,
            'main': main, 'dropped': dropped, 'trial': trial,
            'n_visual': {p: nvis[p] for p in keep},
            'donor1': {p: donors[p][0] for p in keep}, 'donor2': {p: donors[p][1] for p in keep},
            'subset': {p: subset_of(p, nvis[p]) for p in keep}}


def text(doc):
    return json.dumps(doc, separators=(',', ':')) + '\n'


def fresh():
    return build(json.loads(ORDER.read_text()), json.loads(PART1.read_text())['part1'], n_visual_of())


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
