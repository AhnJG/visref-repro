"""Analysis of `stop_form` (diag/stop_form.py; plan docs/EXPERIMENT_FLOW.md 2026-10-24 "실행 전 계획: 멈춤 엔트로피 추정 방식 교차
진단", written before any result). Implements the plan's decision rule; the Korean sentences are the plan's, kept as constants.

Data: the result JSONL files and the analysis list (diag/stop_form_analysis_ids.json 'analysis', fixed from the trial's timing
before the main run with analyze_blank_think.make_analysis_ids). A pid seen twice is merged only when exactly one of its rows
is ok (analyze_blank_think.load). If any analysis question has no row or an error row: no verdict (HOLD).

Estimators per question (continue when the value is >= DELTA = 0.25, the main run's rule; infinite = continue):
  stored  the main run's TSR step-1 H (row 'stored_H', None = infinite)
  H1      the main run's H of short answer j = 0 (a fresh single sample of the same estimator)
  S_tok   mean of the short answers' H; infinite when any short answer is invalid by the main run's probe rule (H None; the main
          run continues on an invalid probe). H1 and S_tok follow that rule, not the value rule of S_val / F_val
  S_val   entropy (nats) of the empirical distribution of the short answers' values
  F_val   entropy of the empirical distribution of the free answers' values -- our value-frequency estimator (main)
  F_sum   mean over the free answers of the summed token entropies (EOS included; answers cut at the cap kept as they are)
  F_tok   mean over the free answers of the per-token mean entropy
Invalid answers (value None) are never dropped. S_val and F_val are computed twice: 'hi' gives every invalid answer its own new
value (largest entropy), 'lo' gives them all the most common valid value (smallest entropy; 0 when no answer is valid).
Decision (questions whose stored first step ended on its own, stratum close or eos; n of them):
  r_lo = share with F_val_lo >= DELTA, r_hi = share with F_val_hi >= DELTA, exact (Clopper-Pearson) 95% ranges.
  Change against the main run: b = stored stops and F_val_lo continues, c = stored continues and F_val_lo stops; exact
  two-sided sign test on b vs c (= exact McNemar).
  MOVE    lower end of r_lo > 0.10 and b > c and p < 0.05
  STAY    upper end of r_hi < 0.10
  else    UNDECIDED
  0.10 is a practical criterion for "the operating point changes" (plan), not a bound on any accuracy gain.
Described (no sentence): the same counts for every estimator and for the cut questions, 2 x 2 tables against stored for
F_val lo / hi, invalid and cap shares, free-answer length, F_val from the first 8 free answers only, the majority value's
correctness, multiple choice / free form.

  cd repro && python3 diag/analyze_stop_form.py 'kaggle/runs/stop_form_main/output/results/stop_form_*.jsonl' \\
      [--ids diag/stop_form_analysis_ids.json] --out diag/stop_form_analysis.json [--force]
"""
import argparse
import json
import math
import statistics
import sys
from collections import Counter
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_blank_think import load  # noqa: E402
from analyze_closed_multistep import rel, sign_p  # noqa: E402

DELTA, R_MIN, ALPHA = 0.25, 0.10, 0.05
SELF = ('close', 'eos')
INF = float('inf')
MOVE = ('스스로 끝낸 첫 step에서, 자유 답 여러 개의 값 빈도로 잰 엔트로피로는 본 실험의 멈춤 엔트로피보다 뚜렷이 많은 문항이 '
        '이어 간다(운영점이 바뀐다).')
STAY = '자유 답 여러 개의 값 빈도로 잰 엔트로피로도, 스스로 끝낸 첫 step은 대부분(이어 가는 비율 10% 미만) 멈춘다.'
UNDECIDED = '이 표본으로는 자유 답 값 빈도 추정량으로 이어 가는 비율이 10%를 넘는지 정하지 못했다.'
HOLD = '고정한 분석 문항 중 결과가 없거나 오류인 문항이 있어 판정을 보류한다.'
ESTIMATORS = ('stored', 'H1', 'S_tok', 'S_val_lo', 'S_val_hi', 'F_val_lo', 'F_val_hi', 'F_sum', 'F_tok')


def value_entropy(values):
    """(lo, hi) entropy in nats of the empirical value distribution; None = invalid (see the module docstring)."""
    valid = [v for v in values if v is not None]
    n, k = len(values), len(values) - len(valid)
    if not n:
        return None, None
    c = Counter(valid)
    hi_counts = list(c.values()) + [1] * k
    lo = Counter(c)
    if valid:
        lo[c.most_common(1)[0][0]] += k
    else:
        lo['_'] = k
    h = lambda counts: -sum(x / n * math.log(x / n) for x in counts if x) + 0.0  # noqa: E731
    return h(lo.values()), h(hi_counts)


def estimates(row, n_free=None):
    """{estimator: value} for one row; n_free limits the free answers to the first n (the 8-answer description)."""
    sh, fr = row['short'], row['free'][:n_free]
    s_lo, s_hi = value_entropy([a['value'] for a in sh])
    f_lo, f_hi = value_entropy([a['value'] for a in fr])
    hs = [a['H'] for a in sh]
    return {'stored': INF if row['stored_H'] is None else row['stored_H'],
            'H1': INF if hs[0] is None else hs[0],
            'S_tok': INF if any(h is None for h in hs) else sum(hs) / len(hs),
            'S_val_lo': s_lo, 'S_val_hi': s_hi, 'F_val_lo': f_lo, 'F_val_hi': f_hi,
            'F_sum': statistics.mean(a['ent_sum'] for a in fr),
            'F_tok': statistics.mean(a['ent_mean'] if a['ent_mean'] is not None else 0.0 for a in fr)}


def cont(x):
    return x >= DELTA


def cp_range(k, n, alpha=ALPHA):
    """Exact (Clopper-Pearson) two-sided range of a binomial share k / n, by bisection on the binomial tails."""
    def tail_ge(p):  # P(X >= k)
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))

    def tail_le(p):  # P(X <= k)
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))

    def root(f, target, rising):
        a, b = 0.0, 1.0
        for _ in range(80):
            m = (a + b) / 2
            if (f(m) < target) == rising:
                a = m
            else:
                b = m
        return (a + b) / 2
    lo = 0.0 if k == 0 else root(tail_ge, alpha / 2, True)
    hi = 1.0 if k == n else root(tail_le, alpha / 2, False)
    return lo, hi


def table(E, a, b):
    """2 x 2 counts of continue / stop under estimators a (rows) and b."""
    t = Counter((cont(e[a]), cont(e[b])) for e in E)
    return {'stop_stop': t[(False, False)], 'stop_cont': t[(False, True)], 'cont_stop': t[(True, False)],
            'cont_cont': t[(True, True)]}


def decide(E):
    n = len(E)
    k_lo = sum(cont(e['F_val_lo']) for e in E)
    k_hi = sum(cont(e['F_val_hi']) for e in E)
    r_lo, r_hi = cp_range(k_lo, n), cp_range(k_hi, n)
    t = table(E, 'stored', 'F_val_lo')
    b, c = t['stop_cont'], t['cont_stop']
    p = sign_p(b, c)
    if r_lo[0] > R_MIN and b > c and p < ALPHA:
        s = MOVE
    elif r_hi[1] < R_MIN:
        s = STAY
    else:
        s = UNDECIDED
    rd = lambda x: [round(x[0], 4), round(x[1], 4)]  # noqa: E731
    return {'n': n, 'k_lo': k_lo, 'k_hi': k_hi, 'r_lo': round(k_lo / n, 4), 'r_lo_ci95': rd(r_lo), 'r_hi': round(k_hi / n, 4),
            'r_hi_ci95': rd(r_hi), 'stored_vs_F_val_lo': t, 'b_new_continue': b, 'c_new_stop': c, 'p_sign': round(p, 6),
            'sentence': s}


def describe(rows):
    E = [estimates(r) for r in rows]
    E8 = [estimates(r, 8) for r in rows]
    fr = [a for r in rows for a in r['free']]
    sh = [a for r in rows for a in r['short']]
    maj = []
    for r in rows:
        c = Counter(a['value'] for a in r['free'] if a['value'] is not None)
        if c:
            v = c.most_common(1)[0][0]
            maj.append(next(a['correct'] for a in r['free'] if a['value'] == v))
    return {'n': len(rows),
            'n_continue': {k: sum(cont(e[k]) for e in E) for k in ESTIMATORS},
            'n_continue_first8_free': {k: sum(cont(e[k]) for e in E8) for k in ('F_val_lo', 'F_val_hi', 'F_sum', 'F_tok')},
            'stored_vs': {k: table(E, 'stored', k) for k in ('H1', 'S_tok', 'F_val_lo', 'F_val_hi', 'F_sum', 'F_tok')},
            'median': {k: round(statistics.median(e[k] for e in E), 5) for k in ESTIMATORS if k != 'stored'},
            'short_invalid': sum(not a['valid'] for a in sh), 'n_short': len(sh),
            'free_invalid': sum(not a['valid'] for a in fr), 'free_cap': sum(a['finish'] == 'cap' for a in fr),
            'free_no_box': sum(not a['boxed'] for a in fr), 'n_free': len(fr),
            'free_len_median': statistics.median(a['n_tokens'] for a in fr),
            'n_questions_all_free_invalid': sum(all(not a['valid'] for a in r['free']) for r in rows),
            'majority_free_value_correct': {'n': len(maj), 'correct': sum(maj)}}


def analyze(R, ids, files=None, ids_file=None):
    want = [str(p) for p in ids['analysis']]
    used = [R[p] for p in want if p in R and R[p].get('status') == 'ok']
    cov = {'n_analysis': len(want), 'n_used': len(used), 'n_rows': len(R), 'n_outside_analysis': sum(p not in set(want) for p in R),
           'errors': sorted(p for p in want if p in R and R[p].get('status') != 'ok'), 'n_missing': sum(p not in R for p in want),
           'code_versions': sorted({x['code_version'] for x in R.values() if 'code_version' in x})}
    out = {'files': [rel(f) for f in (files or [])], 'ids_file': rel(ids_file) if ids_file else None, 'coverage': cov}
    selfr = [r for r in used if r['stratum'] in SELF]
    if not selfr or cov['errors'] or cov['n_missing']:
        out.update(decision=None, sentence=HOLD)
        return out
    d = decide([estimates(r) for r in selfr])
    out.update(decision=d, sentence=d['sentence'])
    cut = [r for r in used if r['stratum'] == 'cap']
    out['descriptive'] = {'self': describe(selfr), 'cut': describe(cut) if cut else None}
    out['by_question_type'] = {qt: describe(sub) for qt in sorted({r['question_type'] for r in selfr})
                               for sub in [[r for r in selfr if r['question_type'] == qt]]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='+')
    ap.add_argument('--ids', default=str(HERE / 'stop_form_analysis_ids.json'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and not a.force:
        sys.exit(f'{out} exists (use --force)')
    R, files = load(a.results)
    res = analyze(R, json.loads(Path(a.ids).read_text()), files, a.ids)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'sentence': res['sentence'], 'coverage': res['coverage']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
