"""Question lists of the injection-sensitivity check (EF 2026-10-18 "실행 전 계획: 이미지 민감도 점검"), chosen from the
main run's seed-0 order and the stored step-1 ending only (no correctness).

  part1  the first 200 questions of the order (= diag/format_check_ids.json), any step-1 ending
  part2  within part1, in part-1 order: the first 50 whose stored main-run TSR step-1 stopping entropy (steps[0].H) is
         finite and above 0.25 and the first 50 whose stored value is at most 0.25 (changed before the main run, see
         EF 2026-10-18 "5문항 시험과 본 실행 전 고침": the first version took 100 questions the model had closed itself,
         of which only 1 had a stored value above 0.25)
  part2_group  'above' / 'below' for each part-2 question
  donor  for part1: the length-matched control's question, the next question in part-1 order (cyclic) with the same
         number of image tokens; when there is none, the next with more image tokens (its tokens are cut, never repeated)
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
N1, N2 = 200, 50
DELTA = 0.25


def tsr_probes(files):
    """{pid: {'ids', 'valid', 'H', 'n_visual'}} of the main-run TSR's step 1."""
    out = {}
    for f in sorted(files):
        for line in open(f):
            r = json.loads(line)
            if r.get('method') == 'tsr' and r.get('steps'):
                s = r['steps'][0]
                out[str(r['pid'])] = {'ids': s.get('probe_ids') or [], 'valid': bool(s.get('probe_valid')),
                                      'H': s.get('H'), 'n_visual': r.get('n_visual')}
    return out


def donors(part1, nvis):
    """Next question in part-1 order (cyclic) with the same number of image tokens, else the next with more."""
    out, n = {}, len(part1)
    for i, p in enumerate(part1):
        nxt = [part1[(i + k) % n] for k in range(1, n)]
        same = [q for q in nxt if nvis[q] == nvis[p]]
        more = [q for q in nxt if nvis[q] > nvis[p]]
        out[p] = (same or more)[0]
    return out


def build(order, store, probes=None, n1=N1, n2=N2):
    order = [str(p) for p in order]
    assert all(p in store for p in order), 'a stored step 1 is missing'
    part1 = order[:n1]
    assert len(part1) == n1
    doc = {'part1': part1, 'stratum': {p: store[p]['think_end'] for p in sorted(part1, key=int)}}
    if probes is not None:
        h = {p: probes[p]['H'] for p in part1}
        fin = lambda x: x is not None and x != float('inf')  # noqa: E731
        above = [p for p in part1 if fin(h[p]) and h[p] > DELTA][:n2]
        below = [p for p in part1 if fin(h[p]) and h[p] <= DELTA][:n2]
        assert len(above) == n2 and len(below) == n2
        doc['part2'] = [p for p in part1 if p in set(above + below)]
        doc['part2_group'] = {p: 'above' if p in above else 'below' for p in doc['part2']}
        doc['donor'] = donors(part1, {p: probes[p]['n_visual'] for p in part1})
        doc['tsr_probe'] = {p: {'ids': probes[p]['ids'], 'valid': probes[p]['valid']} for p in part1}
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
