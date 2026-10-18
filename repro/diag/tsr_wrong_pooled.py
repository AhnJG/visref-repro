"""The questions the main-run TSR got wrong (399 of 1,000), pooled over step1_branch (A3) and step1_think (A7): D - C.
A look after the results (2026-10-18), descriptive; it answers "does the image help where TSR was wrong?" from the two
runs that already compared, from the same stored step 1, more thinking without (C) and with (D) the main-run DPP image.
A3 covers the questions where a method continued at step 1, A7 those where both stopped; together they hold every
TSR-wrong question once. Per question: (correct T = 1 answers under D - under C) / answers per condition (A3: 3, A7: its
own count), so each question weighs 1. Correctness: each run's own stored scoring (as in analyze_step1_branch /
analyze_step1_think); TSR-wrong: llm/final/merged.jsonl. 95% range by question bootstrap (2,000, seed 0).
Greedy D - C (one answer per condition) as a second view.

Run from repro/: python diag/tsr_wrong_pooled.py  ->  diag/tsr_wrong_pooled.json
"""
import glob
import json
import random
from pathlib import Path

A3 = ['kaggle/runs/step1_branch_pilot/output/results/step1_branch_*.jsonl',
      'kaggle/runs/step1_branch_main/output/results/step1_branch_*.jsonl']   # as in the Makefile
A7 = ['kaggle/runs/step1_think_bmain/output/results/step1_think_*.jsonl']   # A7 main reading (main-run injection)


def rows(patterns):
    R = {}
    for f in sorted({f for p in patterns for f in glob.glob(p)}):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                R[str(r['pid'])] = r
    return R


def frac(r, c):
    a = [x for x in r['answers'][c] if not x.get('greedy')]
    return sum(int(x['correct']) for x in a) / len(a)


def boot(d, b=2000, seed=0):
    rng, n = random.Random(seed), len(d)
    m = sorted(100 * sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return [round(m[int(.025 * b)], 2), round(m[int(.975 * b)], 2)]


def main():
    wrong = {str(r['pid']) for r in map(json.loads, open('llm/final/merged.jsonl')) if r['method'] == 'tsr' and not r['correct']}
    ids3 = json.loads(Path('diag/step1_branch_ids.json').read_text())
    ids3 = {str(x) for x in (ids3['ids'] if isinstance(ids3, dict) else ids3)}
    ids7 = json.loads(Path('diag/step1_think_ids.json').read_text())
    ids7 = {str(x) for x in (ids7['ids'] if isinstance(ids7, dict) else ids7)}
    r3 = {p: r for p, r in rows(A3).items() if p in ids3 and all(c in r['acc_t1'] for c in 'ABCD')}
    r7 = {p: r for p, r in rows(A7).items() if p in ids7 and all(c in r['acc_t1'] for c in 'CDR')}
    assert not set(r3) & set(r7)
    out = {'label': 'a look after the results, descriptive', 'n_tsr_wrong': len(wrong)}
    pooled_d, pooled_g = [], []
    for name, R in (('A3', r3), ('A7', r7)):
        ps = sorted(set(R) & wrong, key=int)
        d = [frac(R[p], 'D') - frac(R[p], 'C') for p in ps]
        g = [int(R[p]['greedy']['D']) - int(R[p]['greedy']['C']) for p in ps]
        pooled_d += d
        pooled_g += g
        out[name] = {'n': len(ps), 'diff_pp': round(100 * sum(d) / len(d), 2), 'range95_pp': boot(d),
                     'up': sum(x > 0 for x in d), 'down': sum(x < 0 for x in d),
                     'greedy_diff_pp': round(100 * sum(g) / len(g), 2), 'greedy_up_down': [g.count(1), g.count(-1)]}
    out['pooled'] = {'n': len(pooled_d), 'diff_pp': round(100 * sum(pooled_d) / len(pooled_d), 2) + 0.0, 'range95_pp': boot(pooled_d),
                     'up': sum(x > 0 for x in pooled_d), 'down': sum(x < 0 for x in pooled_d),
                     'greedy_diff_pp': round(100 * sum(pooled_g) / len(pooled_g), 2),
                     'greedy_up_down': [pooled_g.count(1), pooled_g.count(-1)]}
    out['tsr_wrong_not_covered'] = len(wrong - set(r3) - set(r7))
    Path('diag/tsr_wrong_pooled.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
