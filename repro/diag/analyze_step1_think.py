"""Pre-registered analysis of step1_think (EXPERIMENT_FLOW 2026-10-18 "실행 전 계획: 첫 단계에서 멈춘 문항을 더 생각하게 하기"),
fixed before results. Inputs: the result JSONL files of diag/step1_think.py (glob patterns, one per GPU) and the ids file
(diag/step1_think_ids.json); nothing else.

Item-level accuracy = mean over the n_t1 T=1 answers, kept as integer counts of correct answers (0..n_t1) so that sums are
exact (the 2026-10-12 fix of analyze_step1_branch.py). Paired difference of two conditions = mean of count differences / n_t1.
Bootstrap = question resampling, 20,000 resamples, numpy default_rng(0) (a fresh generator per comparison), CI = the
2.5% / 97.5% order statistics sorted(sums)[int(.025 n)], [int(.975 n)] (as analyze_step1_branch.boot), two-sided p =
min(1, 2 * min(P(b <= 0), P(b >= 0))) over the same resamples (an exact 0 counts in both tails; analyze_step1_branch.p_boot).

Primary (the only verdict): D - C in the strata 'close' (step 1 closed itself) and 'cap' (step 1 cut), Holm over the two
(sorted ascending, running max of min(1, p * (m - i))). Verdict per stratum: 'gain' if Holm p < 0.05 and CI low > 0;
otherwise 'no_gain', with the flag 'below_paper_size' when the CI high < 3.5 pp (no gain of +3.5 pp size).
Secondary (descriptive): D - R and R - C per stratum, in all analysed rows and in 'eos'; greedy-answer differences with an
exact McNemar p; per condition: extra-thinking end-type counts, mean length, rows with </think> inside the extra thinking,
valid-answer rate over all of a row's answers (flag below 95%).
Completeness: duplicate pids are refused (exit 1); pids of the ids file without a row are reported as missing; the analysis
uses the complete prefix of the ids order (the longest prefix whose pids all have a row), n reported. Rows of that prefix
that lack a condition (non-finite logits) are excluded and listed.

  python3 diag/analyze_step1_think.py 'results/step1_think_bmain_*.jsonl' --ids diag/step1_think_ids.json [--out FILE]
"""
import argparse
import glob
import json
import os
import statistics as st
import sys
from math import comb
from pathlib import Path

import numpy as np

CONDS = ('C', 'D', 'R')
STRATA = ('close', 'cap')
B, MARGIN_PP, ALPHA, VALID_MIN = 20000, 3.5, 0.05, 0.95


def boot_sums(d, n=B, seed=0, chunk=2000):
    """Sums of n bootstrap resamples of the integer differences d (exact int64), in resample order."""
    d = np.asarray(d, dtype=np.int64)
    rng = np.random.default_rng(seed)
    out = []
    for s in range(0, n, chunk):
        idx = rng.integers(0, len(d), size=(min(chunk, n - s), len(d)))
        out.append(d[idx].sum(1))
    return np.concatenate(out)


def boot_ci_p(d, scale, n=B, seed=0):
    """(ci_low_pp, ci_high_pp, p) from one set of resamples; d integer count differences, scale = n_t1."""
    raw = boot_sums(d, n, seed)
    bs = np.sort(raw)
    lo, hi = (float(100 * bs[int(f * n)] / (scale * len(d))) for f in (.025, .975))
    p = min(1.0, 2 * min(int((raw <= 0).sum()), int((raw >= 0).sum())) / n)
    return lo, hi, p


def mcnemar(a, b):
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def holm(ps):
    """Holm-adjusted p for a dict {name: p}."""
    out, run = {}, 0.0
    for i, k in enumerate(sorted(ps, key=ps.get)):
        run = max(run, min(1.0, ps[k] * (len(ps) - i)))
        out[k] = run
    return out


def n_correct(r, c):
    """Correct T=1 answers of row r under condition c (greedy excluded); checked against the stored acc_t1."""
    a = [x for x in r['answers'][c] if not x.get('greedy')]
    k = sum(int(x['correct']) for x in a)
    assert abs(k / len(a) - r['acc_t1'][c]) < 1e-12, (r['pid'], c)
    return k


def n_samples(rows):
    ns = {sum(1 for x in r['answers'][c] if not x.get('greedy')) for r in rows for c in CONDS}
    assert len(ns) == 1, ns
    return ns.pop()


def pair(rows, x, y):
    """x - y over rows (paired by question)."""
    m = n_samples(rows)
    d = [n_correct(r, x) - n_correct(r, y) for r in rows]
    lo, hi, p = boot_ci_p(d, m)
    gx = sum(r['greedy'][x] > r['greedy'][y] for r in rows)
    gy = sum(r['greedy'][x] < r['greedy'][y] for r in rows)
    return {'n': len(rows), 'diff_pp': round(100 * sum(d) / (m * len(d)), 2), 'ci95_pp': [round(lo, 2), round(hi, 2)],
            'p_boot': round(p, 4), '_lo': lo, '_hi': hi, 'greedy_diff_pp': round(100 * (sum(r['greedy'][x] for r in rows) - sum(r['greedy'][y] for r in rows)) / len(rows), 2),
            'greedy_only': [gx, gy], 'p_mcnemar_greedy': round(mcnemar(gx, gy), 4)}


def clean(d):
    return {k: v for k, v in d.items() if not k.startswith('_')}


def load(patterns):
    """{pid: row} from all files matching the patterns; a pid seen twice is an error."""
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        raise SystemExit('no result files match')
    R = {}
    for f in files:
        for line in Path(f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                pid = str(r['pid'])
                if pid in R:
                    raise SystemExit(f'duplicate pid {pid} (second occurrence in {f})')
                R[pid] = r
    return R, files


def prefix_of(ids, R):
    """(complete prefix of ids, missing pids of ids, rows that are not in ids)."""
    n = 0
    while n < len(ids) and ids[n] in R:
        n += 1
    return ids[:n], [p for p in ids if p not in R], sorted(set(R) - set(ids))


def mechanism(rows, c):
    ex = [r['extra'][c] for r in rows if 'end' in r['extra'].get(c, {})]
    valid = [st.mean(x['valid'] for x in r['answers'][c]) for r in rows if c in r['answers']]
    return {'n': len(ex), 'end': {e: sum(x['end'] == e for x in ex) for e in ('close', 'eos', 'cap')},
            'mean_len': round(st.mean(x['len'] for x in ex), 1) if ex else None,
            'rows_with_think_end_inside': sum(x.get('n_think_end_inside', 0) > 0 for x in ex),
            'valid_answer_rate': round(st.mean(valid), 4) if valid else None}


def analyse(R, ids):
    pre, missing, extra_rows = prefix_of(ids, R)
    inj = sorted({R[p].get('inject') for p in pre})
    vers = sorted({(R[p]['code_version'], R[p]['scorer_version']) for p in pre})
    assert len(inj) <= 1, f'rows of several --inject modes: {inj}'
    ok = [p for p in pre if all(c in R[p]['acc_t1'] for c in CONDS)]
    rows = [R[p] for p in ok]
    by = {s: [r for r in rows if r['stratum'] == s] for s in ('close', 'cap', 'eos')}
    out = {'n_ids': len(ids), 'n_prefix': len(pre), 'n_analysed': len(rows), 'missing_n': len(missing),
           'missing_first': missing[:10], 'n_after_prefix': len(ids) - len(pre), 'rows_not_in_ids': extra_rows, 'rows_lacking_a_condition': [p for p in pre if p not in set(ok)],
           'inject': inj[0] if inj else None, 'code_version_scorer_version': vers,
           'n_by_stratum': {s: len(v) for s, v in by.items()}}
    if not rows:
        return out
    m = n_samples(rows)
    out['acc_t1_pp'] = {s: {c: round(100 * sum(n_correct(r, c) for r in v) / (m * len(v)), 2) for c in CONDS}
                        for s, v in {**by, 'all': rows}.items() if v}
    prim = {s: pair(by[s], 'D', 'C') for s in STRATA if by[s]}
    h = holm({s: prim[s]['p_boot'] for s in prim})
    for s in prim:
        r = prim[s]
        r['p_holm'] = round(h[s], 4)
        gain = h[s] < ALPHA and r['_lo'] > 0
        r['verdict'] = 'gain' if gain else 'no_gain'
        r['flags'] = ['below_paper_size'] if not gain and r['_hi'] < MARGIN_PP else []
    out['primary_D_minus_C'] = {s: clean(v) for s, v in prim.items()}
    sec = {}
    for name, (x, y) in {'D-R': ('D', 'R'), 'R-C': ('R', 'C'), 'D-C': ('D', 'C')}.items():
        sec[name] = {s: clean(pair(v, x, y)) for s, v in {**by, 'all': rows}.items() if v}
    out['secondary_pairs'] = sec
    out['conditions'] = {c: {s: mechanism(v, c) for s, v in {**by, 'all': rows}.items() if v} for c in CONDS}
    rate = {c: out['conditions'][c]['all']['valid_answer_rate'] for c in CONDS}
    out['flags'] = [f'valid_answer_rate<{VALID_MIN} for {c}' for c in CONDS if rate[c] is not None and rate[c] < VALID_MIN]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='+', help='result JSONL files or glob patterns')
    ap.add_argument('--ids', required=True)
    ap.add_argument('--out', default='')
    a = ap.parse_args(argv)
    ids = json.loads(Path(a.ids).read_text())
    ids = [str(x) for x in (ids['ids'] if isinstance(ids, dict) else ids)]
    R, files = load(a.results)
    out = analyse(R, ids)
    out['files'] = files
    text = json.dumps(out, indent=1, ensure_ascii=False)
    if a.out:
        if os.path.exists(a.out):
            sys.exit(f'refusing to overwrite {a.out}')
        Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
