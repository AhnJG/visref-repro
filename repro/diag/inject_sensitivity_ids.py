"""Question lists of the injection-sensitivity check (EF 2026-10-18 "실행 전 계획: 이미지 민감도 점검"), chosen from the
main run's seed-0 order and the stored step-1 ending only (no correctness).

  part1  the first 200 questions of the order (= diag/format_check_ids.json), any step-1 ending
  part2  the first 100 questions of the order whose stored step 1 the model closed itself ('close')
  tsr_probe  for part1: the token ids of the main-run TSR's step-1 sampled answer after '\\n\\n\\boxed' (steps[0].probe_ids)
             and whether it was valid; it is the answer the model gave in the context without re-inserted image tokens

  cd repro && python3 diag/inject_sensitivity_ids.py            # writes diag/inject_sensitivity_ids.json
  cd repro && python3 diag/inject_sensitivity_ids.py --check    # exits 1 if the file differs from a fresh build
"""
import argparse
import glob
import json
from pathlib import Path

REPRO = Path(__file__).resolve().parent.parent
ORDER = REPRO / 'kaggle/runs/b_main_hs/output/results/ids.json'
STORE = REPRO / 'diag/step1_store_bmain.json'
OUT = REPRO / 'diag/inject_sensitivity_ids.json'
RUNS = 'kaggle/runs/b_main_s*/output/results/results_*.jsonl'
N1, N2 = 200, 100


def tsr_probes(files):
    out = {}
    for f in sorted(files):
        for line in open(f):
            r = json.loads(line)
            if r.get('method') == 'tsr' and r.get('steps'):
                s = r['steps'][0]
                out[str(r['pid'])] = {'ids': s.get('probe_ids') or [], 'valid': bool(s.get('probe_valid'))}
    return out


def build(order, store, probes=None, n1=N1, n2=N2):
    order = [str(p) for p in order]
    assert all(p in store for p in order), 'a stored step 1 is missing'
    part1 = order[:n1]
    part2 = [p for p in order if store[p]['think_end'] == 'close'][:n2]
    assert len(part1) == n1 and len(part2) == n2
    doc = {'part1': part1, 'part2': part2, 'stratum': {p: store[p]['think_end'] for p in sorted(set(part1 + part2), key=int)}}
    if probes is not None:
        doc['tsr_probe'] = {p: probes[p] for p in part1}
    return doc


def text(doc):
    return json.dumps(doc) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args(argv)
    doc = build(json.loads(ORDER.read_text()), json.loads(STORE.read_text())['pids'],
                tsr_probes(glob.glob(str(REPRO / RUNS))))
    if a.check:
        ok = OUT.read_text() == text(doc)
        print('ok' if ok else 'differs')
        return 0 if ok else 1
    OUT.write_text(text(doc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
