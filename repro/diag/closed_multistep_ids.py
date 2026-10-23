"""Question lists of `closed_multistep` (plan: closed_multistep_plan.md, sections 3 and 5; written before any result).

  main    the 334 MathVista testmini pids whose stored first thinking step closed by itself (diag/step1_store_bmain.json
          think_end == 'close'), in the main run's seed-0 order (kaggle/runs/b_main_hs/output/results/ids.json, the order
          diag/step1_think_ids.py and diag/thumb_gate_ids.py use). Chosen from the stored step-1 ending only (no correctness).
  part1   the first 112 of `main` (interim stop rule: diag/count_closed_multistep_interim.py)
  part2   the other 222 of `main`
  trial   the first 5 of `main` + the first pid of `main` with n_visual 3328 when none is among those 5 (the trial pids are
          ordinary members of `main`: their rows count in the main analysis)
  memcheck the first trial pid with n_visual 3328 (the memory check of the trial run)
  n_visual image tokens of every main pid (the TSR rows of the main run, as inject_sensitivity_ids.tsr_probes reads them)
  stratum {pid: stored think_end} of every main pid (all 'close')
  rules   this selection as text

  cd repro && python3 diag/closed_multistep_ids.py            # writes diag/closed_multistep_ids.json
  cd repro && python3 diag/closed_multistep_ids.py --check    # exits 1 if the file differs from a fresh build
"""
import argparse
import glob
import json
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO / 'diag'))
from inject_sensitivity_ids import ORDER, RUNS, STORE, tsr_probes  # noqa: E402

OUT = REPRO / 'diag/closed_multistep_ids.json'
N_MAIN, N_PART1, N_TRIAL, BIG = 334, 112, 5, 3328
RULES = ('main = the pids whose stored first thinking step closed by itself (think_end == close in diag/step1_store_bmain.json), '
         'in the main-run seed-0 order; no correctness is used. part1 = the first 112 of main, part2 = the other 222 (interim stop '
         'rule). trial = the first 5 of main, plus the first pid of main with 3328 image tokens when none of the 5 has that many; '
         'the trial pids stay in main. memcheck = the first trial pid with 3328 image tokens.')


def build(order, store, nvis, n_main=N_MAIN):
    """The ids document from the main-run order, the stored first steps {pid: {think_end}} and {pid: n_visual}."""
    order = [str(p) for p in order]
    assert len(set(order)) == len(order), 'duplicate pid in order'
    assert all(p in store and p in nvis for p in order), 'stored step 1 or n_visual missing'
    main = [p for p in order if store[p]['think_end'] == 'close']
    assert n_main is None or len(main) == n_main, (len(main), n_main)
    trial = main[:N_TRIAL]
    if not any(nvis[p] == BIG for p in trial):
        extra = [p for p in main if nvis[p] == BIG]
        assert extra, f'no question with {BIG} image tokens'
        trial.append(extra[0])
    memcheck = next(p for p in trial if nvis[p] == BIG)
    return {'experiment': 'closed_multistep', 'main': main, 'part1': main[:N_PART1], 'part2': main[N_PART1:], 'trial': trial,
            'memcheck': memcheck, 'n_visual': {p: int(nvis[p]) for p in main}, 'stratum': {p: store[p]['think_end'] for p in main},
            'rules': RULES}


def text(doc):
    return json.dumps(doc, separators=(',', ':')) + '\n'


def fresh():
    nvis = {p: v['n_visual'] for p, v in tsr_probes(glob.glob(str(REPRO / RUNS))).items()}
    return build(json.loads(ORDER.read_text()), json.loads(STORE.read_text())['pids'], nvis)


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
