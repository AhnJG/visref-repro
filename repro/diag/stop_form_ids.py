"""Question lists of `stop_form` (plan docs/EXPERIMENT_FLOW.md 2026-10-24 "실행 전 계획: 멈춤 엔트로피 추정 방식 교차 진단", written
before any result). Chosen from the main run's seed-0 order and how its stored first step ended only (no correctness).

  self      questions whose stored first step (diag/step1_store_bmain.json) ended with the model's </think> ('close') or EOS
            ('eos'), in the main run's order (kaggle/runs/b_main_hs/output/results/ids.json)
  cut       questions whose stored first step was cut at the step cap, in order
  trial     the last 5 'self' questions and cut[100:103] (8 questions: one full group of 4 per GPU)
  main      'self' minus trial, then cut[:100]; the run is sharded by index (i % 2) and stops starting new groups at its
            time limit, so the analysed questions are a prefix of this order on each GPU (fixed before the main run from the
            trial's timing, diag/analyze_blank_think.make_analysis_ids)
  stratum   {pid: 'close' | 'eos' | 'cap'}
  stored_H  {pid: the main run's TSR step-1 stopping entropy (llm/final/merged.jsonl, steps[0]['H']), None when it was
            infinite (an invalid probe, which continues)}

  cd repro && python3 diag/stop_form_ids.py            # writes diag/stop_form_ids.json
  cd repro && python3 diag/stop_form_ids.py --check    # exits 1 if the file differs from a fresh build
"""
import argparse
import json
import math
from pathlib import Path

REPRO = Path(__file__).resolve().parent.parent
ORDER = REPRO / 'kaggle/runs/b_main_hs/output/results/ids.json'
STORE = REPRO / 'diag/step1_store_bmain.json'
MERGED = REPRO / 'llm/final/merged.jsonl'
OUT = REPRO / 'diag/stop_form_ids.json'
N_CUT, N_TRIAL_SELF, N_TRIAL_CUT = 100, 5, 3  # trial: 4 questions per GPU, the main run's group size


def stored_H():
    """{pid: TSR step-1 H or None (infinite)} from the main run's rows."""
    out = {}
    for line in MERGED.read_text().splitlines():
        r = json.loads(line)
        if r['method'] == 'tsr' and r['status'] == 'ok':
            h = float(r['steps'][0]['H'])
            out[str(r['pid'])] = None if math.isinf(h) else h
    return out


def build(order, ends, H):
    order = [str(p) for p in order]
    assert len(order) == len(set(order)) and all(p in ends and p in H for p in order), 'order has repeats or missing pids'
    self_ = [p for p in order if ends[p] in ('close', 'eos')]
    cut = [p for p in order if ends[p] == 'cap']
    trial = self_[-N_TRIAL_SELF:] + cut[N_CUT:N_CUT + N_TRIAL_CUT]
    main = self_[:-N_TRIAL_SELF] + cut[:N_CUT]
    return {'experiment': 'stop_form', 'main': main, 'trial': trial, 'n_self_main': len(self_) - N_TRIAL_SELF,
            'stratum': {p: ends[p] for p in main + trial}, 'stored_H': {p: H[p] for p in main + trial}}


def text(doc):
    return json.dumps(doc, separators=(',', ':')) + '\n'


def fresh():
    store = json.loads(STORE.read_text())
    store = store.get('pids', store)
    return build(json.loads(ORDER.read_text()), {p: s['think_end'] for p, s in store.items()}, stored_H())


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
