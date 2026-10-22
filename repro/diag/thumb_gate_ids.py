"""Question lists of the thumbnail gate `thumb_gate` (plan: docs/EXPERIMENT_FLOW.md 2026-10-22 "실행 전 계획: 썸네일 관문 `thumb_gate`",
written before any result). Chosen from the main run's seed-0 order, the image-token counts and the stored step-1 ending only
(no correctness).

  main     inject_sensitivity_ids.json 'part1' (the first 200 questions of the order, same order)
  trial    5 questions OUTSIDE the 200 (main-run seed-0 order from position 201 on), in this order:
             (a) the one with the largest n_visual; ties: the longest stored think_tokens (diag/step1_store_bmain.json), then
                 the earlier position in the order
             (b) the first with n_visual == 256 (one tile, no thumbnail: T- input equals T+ input)
             (c)(d)(e) the first not yet chosen multi-tile question (n_visual > 256) whose stored think_end is 'cap', 'close',
                 'eos' respectively; when none is left for an ending, the next not yet chosen multi-tile question in order
  donor    main pids: the donor of inject_sensitivity_ids.json; trial pids: inject_sensitivity_ids.donors(order[200:], nvis)
           (the next question of order[200:], cyclic, with the same n_visual, else the next with more)
  n_visual image tokens of every main / trial / donor pid (the TSR rows of the main run, as inject_sensitivity_ids.tsr_probes
           reads them)
  multi    {pid: n_visual > 256}: the image has a thumbnail tile (the last 256 tokens) only when it has more than one tile
  stratum  {pid: stored think_end} of every main / trial pid
  rules    this selection as text

  cd repro && python3 diag/thumb_gate_ids.py            # writes diag/thumb_gate_ids.json
  cd repro && python3 diag/thumb_gate_ids.py --check    # exits 1 if the file differs from a fresh build
"""
import argparse
import glob
import json
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO / 'diag'))
from inject_sensitivity_ids import ORDER, RUNS, STORE, donors, tsr_probes  # noqa: E402

PART1 = REPRO / 'diag/inject_sensitivity_ids.json'
OUT = REPRO / 'diag/thumb_gate_ids.json'
N_MAIN = 200
TILE = 256
ENDINGS = ('cap', 'close', 'eos')
RULES = ('main = inject_sensitivity_ids.json part1 (first 200 of the main-run seed-0 order, same order). trial = 5 questions of '
         'the order from position 201 on: (a) largest n_visual, ties by the longest stored think_tokens then by order; (b) the '
         'first with n_visual 256; (c)(d)(e) the first not yet chosen multi-tile question (n_visual > 256) whose stored '
         "think_end is 'cap', 'close', 'eos' (when none is left: the next not yet chosen multi-tile question in order). "
         'donor = inject_sensitivity_ids donor for main, inject_sensitivity_ids.donors(order[200:], n_visual) for trial. '
         'multi = n_visual > 256 (a thumbnail tile of 256 tokens exists only for multi-tile images). Chosen from order, '
         'n_visual and the stored step-1 ending only, no correctness.')


def pick_trial(pool, nvis, store):
    """The 5 trial pids (see the module docstring) from pool = order[200:]."""
    pos = {p: i for i, p in enumerate(pool)}
    a = min(pool, key=lambda p: (-nvis[p], -int(store[p]['think_tokens']), pos[p]))
    trial = [a]
    b = next(p for p in pool if nvis[p] == TILE and p not in trial)
    trial.append(b)
    for end in ENDINGS:
        multi = [p for p in pool if nvis[p] > TILE and p not in trial]
        same = [p for p in multi if store[p]['think_end'] == end]
        trial.append((same or multi)[0])
    return trial


def build(order, part1, donor_main, nvis, store):
    order, part1 = [str(p) for p in order], [str(p) for p in part1]
    assert order[:N_MAIN] == part1 and len(part1) == N_MAIN, 'part1 is not the start of the order'
    assert all(p in nvis and p in store for p in order), 'n_visual or stored step 1 missing'
    assert all(nvis[p] % TILE == 0 for p in order), 'n_visual is not a multiple of 256'
    pool = order[N_MAIN:]
    trial = pick_trial(pool, nvis, store)
    assert len(trial) == len(set(trial)) == 5
    don_trial = donors(pool, {p: nvis[p] for p in pool})
    donor = {**{p: str(donor_main[p]) for p in part1}, **{p: don_trial[p] for p in trial}}
    keep = part1 + trial
    allp = list(dict.fromkeys(keep + sorted(set(donor.values()), key=lambda p: order.index(p))))
    assert all(nvis[donor[p]] >= nvis[p] for p in keep), 'a donor has fewer image tokens than its question'
    return {'experiment': 'thumb_gate', 'main': part1, 'trial': trial, 'donor': donor,
            'n_visual': {p: nvis[p] for p in allp}, 'multi': {p: nvis[p] > TILE for p in allp},
            'stratum': {p: store[p]['think_end'] for p in keep}, 'rules': RULES}


def text(doc):
    return json.dumps(doc, separators=(',', ':')) + '\n'


def fresh():
    order = json.loads(ORDER.read_text())
    part1 = json.loads(PART1.read_text())
    nvis = {p: v['n_visual'] for p, v in tsr_probes(glob.glob(str(REPRO / RUNS))).items()}
    return build(order, part1['part1'], part1['donor'], nvis, json.loads(STORE.read_text())['pids'])


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
