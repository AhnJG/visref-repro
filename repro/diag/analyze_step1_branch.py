"""Pre-registered analysis of step1_branch (EXPERIMENT_FLOW 2026-10-11 "실행 전 계획: step1_branch"), fixed before results.

Item-level accuracy = mean over the n_t1 T=1 answers. Co-primary paired comparisons B-A (immediate answer) and D-C
(after free thinking): bootstrap 95% CI (2,000 resamples, seed 0) and two-sided bootstrap p with Holm over the
two; a rule counts as met when the CI lower bound > 0 and the Holm p < 0.05; greedy-answer McNemar as a
secondary check. Secondary: D-B, C-A, (D-C)-(B-A), strata by b_main step-1 end. Mechanism (C vs D): free-thinking
length, how it ended, image-word mentions. Reconstruction: token-count and DPP-count agreement with b_main.

Arithmetic (fix of 2026-10-12, after results): paired differences are computed on integer counts of correct T=1
answers (0..n_t1 per item) and divided only at the end. The earlier float version summed thirds, so a bootstrap
mean that is exactly 0 came out as about +-1e-15 and fell into one tail or the other depending on the Python
version (sum() changed in 3.12): p_boot 0.289 / 0.575 (Holm 0.578) on Python <= 3.11 vs 0.281 / 0.583 on 3.12+.
The rule itself is unchanged (two-sided p = 2 * min(P(b <= 0), P(b >= 0)), the same resamples as the CI), so an
exact 0 now counts in both tails as the rule intends. The float result is kept in
diag/step1_branch_analysis_float_py311.json.

  python3 diag/analyze_step1_branch.py RESULTS.jsonl... --ids diag/step1_branch_ids.json --out diag/step1_branch_analysis.json
"""
import argparse
import json
import os
import random
import re
import statistics as st
import sys
from math import comb
from pathlib import Path

IMG = re.compile(r'(?i)\b(image|picture|figure|diagram|graph|chart|look(ing)? (at|again)|shown|see)\b')


def boot_sums(d, n=2000, seed=0):
    """Bootstrap resample sums of the integer differences d (exact; same resamples as before the fix)."""
    rng = random.Random(seed)
    return [sum(rng.choice(d) for _ in d) for _ in range(n)]


def boot(d, scale, n=2000, seed=0):
    """95% CI in percentage points of mean(d) / scale, d integer differences of correct counts."""
    bs = sorted(boot_sums(d, n, seed))
    return [round(100 * bs[int(.025 * n)] / (scale * len(d)), 2), round(100 * bs[int(.975 * n)] / (scale * len(d)), 2)]


def p_boot(d, n=2000, seed=0):
    """Two-sided bootstrap p = 2 * min(P(b <= 0), P(b >= 0)); an exact 0 counts in both tails."""
    bs = boot_sums(d, n, seed)
    return round(min(1.0, 2 * min(sum(b <= 0 for b in bs), sum(b >= 0 for b in bs)) / n), 4)


def n_correct(r, c):
    """Number of correct T=1 answers of row r under condition c (the greedy answer is excluded)."""
    k = sum(int(a['correct']) for a in r['answers'][c] if not a.get('greedy'))
    m = sum(1 for a in r['answers'][c] if not a.get('greedy'))
    assert abs(k / m - r['acc_t1'][c]) < 1e-12, (r['pid'], c)
    return k


def mcnemar(a, b):
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def n_t1(R, pids):
    ns = {sum(1 for a in R[p]['answers'][c] if not a.get('greedy')) for p in pids for c in 'ABCD'}
    assert len(ns) == 1, ns
    return ns.pop()


def pair(R, x, y, pids):
    d = [n_correct(R[p], x) - n_correct(R[p], y) for p in pids]
    m = n_t1(R, pids)
    gx, gy = sum(R[p]['greedy'][x] > R[p]['greedy'][y] for p in pids), sum(R[p]['greedy'][x] < R[p]['greedy'][y] for p in pids)
    return {'n': len(pids), 'diff_pp': round(100 * sum(d) / (m * len(d)), 2), 'ci95_pp': boot(d, m),
            'greedy_only': [gx, gy], 'p_mcnemar_greedy': round(mcnemar(gx, gy), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='+')
    ap.add_argument('--ids', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ids = json.loads(Path(a.ids).read_text())
    ids = [str(x) for x in (ids['ids'] if isinstance(ids, dict) else ids)]
    R = {}
    for f in a.results:
        for line in Path(f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                R[r['pid']] = r
    pids = [p for p in ids if p in R and all(c in R[p]['acc_t1'] for c in 'ABCD')]
    out = {'n': len(pids), 'n_rows': len(R), 'code_versions': sorted({R[p]['code_version'] for p in pids}),
           'acc_t1': {c: round(100 * st.mean(R[p]['acc_t1'][c] for p in pids), 2) for c in 'ABCD'},
           'acc_greedy': {c: round(100 * st.mean(R[p]['greedy'][c] for p in pids), 2) for c in 'ABCD'},
           'valid_rate': {c: round(st.mean(st.mean(x['valid'] for x in R[p]['answers'][c]) for p in pids), 4) for c in 'ABCD'}}
    prim = {'B-A': pair(R, 'B', 'A', pids), 'D-C': pair(R, 'D', 'C', pids)}
    for k in prim:  # two-sided bootstrap p (same resamples as the CI), then Holm over the two co-primary comparisons
        prim[k]['p_boot'] = p_boot([n_correct(R[p], k[0]) - n_correct(R[p], k[2]) for p in pids])
    order = sorted(prim, key=lambda k: prim[k]['p_boot'])
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, prim[k]['p_boot'] * (len(order) - i)))
        prim[k]['p_holm'] = round(run, 4)
    out['primary'] = prim
    nc = {p: {c: n_correct(R[p], c) for c in 'ABCD'} for p in pids}
    did = [(nc[p]['D'] - nc[p]['C']) - (nc[p]['B'] - nc[p]['A']) for p in pids]
    m = n_t1(R, pids)
    out['secondary'] = {'D-B': pair(R, 'D', 'B', pids), 'C-A': pair(R, 'C', 'A', pids),
                        'DiD_(D-C)-(B-A)': {'diff_pp': round(100 * sum(did) / (m * len(did)), 2), 'ci95_pp': boot(did, m)}}
    out['by_step1_end'] = {}
    for e in ('cap', 'eos', 'close'):
        ps_e = [p for p in pids if R[p]['think_end_bmain'] == e]
        if ps_e:
            out['by_step1_end'][e] = {'n': len(ps_e), 'acc_t1': {c: round(100 * st.mean(R[p]['acc_t1'][c] for p in ps_e), 1) for c in 'ABCD'}}
    mech = {}
    for c in 'CD':
        ex = [R[p]['extra'][c] for p in pids if 'end' in R[p]['extra'].get(c, {})]
        L = sorted(x['len'] for x in ex)
        mech[c] = {'n': len(ex), 'end': {e: sum(x['end'] == e for x in ex) for e in ('close', 'eos', 'cap')},
                   'len_q10_50_90': [L[int(f * (len(L) - 1))] for f in (.1, .5, .9)] if L else None,
                   'image_word_rate': round(st.mean(bool(IMG.search(x['text'])) for x in ex), 4) if ex else None}
    out['mechanism'] = mech
    out['reconstruction'] = {'tokens_match': sum(R[p]['recon']['tokens_match'] for p in pids),
                             'sel_match_bmain': sum(bool(R[p]['recon']['sel_match']) for p in pids),
                             'n_sel_minus_bmain_q': (lambda v: [v[int(f * (len(v) - 1))] for f in (.1, .5, .9)])(
                                 sorted(R[p]['n_sel'] - R[p]['recon']['n_dpp_bmain'] for p in pids))}
    pos = {k: prim[k]['ci95_pp'][0] > 0 and prim[k]['p_holm'] < 0.05 for k in prim}
    ba, dc = pos['B-A'], pos['D-C']
    if ba and dc:
        j = '①② 즉답 재주입 이득 재현 + 이미지 뒤 생각에서도 유지 → 주입 뒤 생각 변형을 다음 VisRef 실험 후보로'
    elif ba:
        j = '③ 즉답 이득 재현, 생각 후에는 확인 안 됨 → 이어가기 구간 조사'
    elif dc:
        j = '② 단독: 즉답 이득은 재현 안 됐으나 이미지 뒤 생각에서 이득 → 해석 주의(사전 규칙에 없는 조합, 기술만)'
    else:
        j = '④ 즉답 이득 재현 안 됨(b_main probe 관찰은 잡음 가능); 5–6%p 이상 효과는 검출되지 않음'
    out['judgment'] = j
    if os.path.exists(a.out):
        sys.exit(f'refusing to overwrite {a.out}')
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
