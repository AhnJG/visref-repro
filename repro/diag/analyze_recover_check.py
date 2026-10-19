"""Analysis of the recovery check `recover_check` (plan /home/claude/plan_recover.md section 5; written before any result).
Implements the plan's decision steps exactly; the Korean verdict sentences are the plan's, copied as constants.

Data: rows of diag/recover_check.py (status 'ok', 14 conditions). Left out and counted: trial rows (pid in the ids file's
'trial' list or row set == 'trial'), error rows, rows of pids that are not in the ids file's 'main' list, rows that lack a
condition. Everything is a paired question comparison on p_gt (probability units); the bootstrap resamples questions
(20,000 resamples, seed 0, one resample matrix shared by all statistics of a group, 95% range = sorted resample sums at
int(0.025 B) and int(0.975 B), as diag/analyze_step1_think.boot_ci_p). p_gt is stored with 6 decimals, so every
difference is an integer count of millionths and every resample sum is exact int64 arithmetic; ratios are formed per
resample from those exact sums.

Step 1  AB = mean(p_A - p_Bb). Information present = lower end of AB's 95% range > 0 AND AB >= 0.05. If not: stop, the
        'no information' sentence, nothing else is decided (only the means of A and Bb are written).
Step 2  (information present) CB = mean(p_Cb - p_Bb), CD = mean(p_Cb - p_Db); the first that holds:
          1 'barely passes'  upper end of CB's range < 0.25 * AB (AB = the observed mean)
          2 'channel works'  lower ends of CB and CD both > 0
          3 'undecided'
        note: CD's lower end > 0 and CB's range contains 0 -> 'cannot tell from disturbance by the other image'
        (read as: CB's 95% range contains 0).
        described: CB / AB and CB / mean(p_A+ - p_Bb) (each question-bootstrap 95% range), DB = mean(p_Db - p_Bb), A+ - A.
Step 2-aux  mean(p_C - p_B), mean(p_C - p_D), mean(p_B - p_Bb) (description only).
Step 3  (information present; exploratory) S_x = p_Cx - p_Dx for x in n, o, e; compare mean(S_o - S_n): lower end > 0 ->
        'more recovered at the original spacing', upper end < 0 -> 'less recovered', else 'not confirmed'. Described:
        S_e - S_n, S_o - S_e, each S_x, S_x / CD.
Descriptive: means of p_gt and log p_gt per condition, share of questions whose most likely first answer token is the
gold answer's first token, median KL per pair, median context length per condition, and every difference above split
into multiple choice / free form (own bootstrap per group).

  cd repro && python3 diag/analyze_recover_check.py ROWS.jsonl [ROWS2.jsonl ...] [--ids diag/recover_check_ids.json]
      --out diag/recover_check_analysis.json
"""
import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPRO = Path(__file__).resolve().parent.parent
B, SEED = 20000, 0
INFO_MIN, BARELY = 0.05, 0.25
CONDS = ('A', 'Bb', 'Cb', 'Db', 'A+', 'Cn', 'Co', 'Ce', 'Dn', 'Do', 'De', 'B', 'C', 'D')
PAIRS = ('Cb||Db', 'Cn||Dn', 'Co||Do', 'Ce||De', 'C||D')
MICRO = 1_000_000

# ---- the plan's sentences (verbatim) ------------------------------------------------------------------------------------
V1_NO_INFO = '이 문항들과 이 앞글에서는 그림을 빈 그림으로 바꿔도 정답 확률이 뚜렷이 바뀌지 않아, 회복 시험은 정보가 없다.'
V1 = {True: '정보 있음', False: '정보 없음'}
V2 = {'barely': '거의 전달 못 함', 'works': '통로 작동', 'undecided': '가르지 못함'}
V2_NOTE = '다른 그림의 방해와 구분 못 함'
V3 = {'more': '원래 간격 번호에서 다시 넣은 조각이 정답 확률을 더 되살렸다(탐색 결과).',
      'less': '원래 간격 번호가 오히려 덜 되살렸다(탐색 결과).',
      'unclear': '번호 방식 차이 확인 안 됨.'}
V3_NOISE = '네 조건의 차이라 잡음이 크고, 차이를 못 볼 가능성이 높다(추론).'
NEXT = {'works_no_pos_diff': '결과를 작성자에게 보고하고 다음 실험 후보를 함께 정한다. 해석 후보는 "통로는 작동하고, 지금까지의 결과는 '
                             '원래 그림과 겹치는 정보 때문일 가능성". 정리 글은 VisRef의 이득이 확인되기 전까지 쓰지 않는다(작성자 결정, 10-19).',
        'barely': '원인 후보(번호, 태그 없는 그림 값, 생각 블록 안의 자리)를 작성자와 정한다. 3단계에서 원래 간격이 이득이면 그것이 첫 후보다.',
        'pos_gain': '309문항 정답률 실험(그림 없음 / 이어 붙인 번호 / 원래 간격, 더 생각 1,024)의 계획을 작성자와 정한다.'}


class Boot:
    """One matrix of question resamples shared by all statistics over n questions."""

    def __init__(self, n, b=B, seed=SEED):
        self.n, self.b = n, b
        self.idx = np.random.default_rng(seed).integers(0, n, size=(b, n)) if n else None

    def sums(self, d):
        return np.asarray(d, dtype=np.int64)[self.idx].sum(1)

    def mean_ci(self, d):
        """(observed mean, [lo, hi]) of the integer vector d (millionths), in probability units."""
        d = np.asarray(d, dtype=np.int64)
        if not self.n:
            return None, None
        s = np.sort(self.sums(d))
        k = self.n * MICRO
        return int(d.sum()) / k, [int(s[int(.025 * self.b)]) / k, int(s[int(.975 * self.b)]) / k]

    def ratio_ci(self, num, den):
        """(observed ratio, [lo, hi], n resamples with a non-positive denominator left out) of sum(num)/sum(den)."""
        num, den = np.asarray(num, dtype=np.int64), np.asarray(den, dtype=np.int64)
        sn, sd = self.sums(num), self.sums(den)
        ok = sd > 0
        r = np.sort(sn[ok] / sd[ok])
        obs = int(num.sum()) / int(den.sum()) if int(den.sum()) > 0 else None
        ci = [float(r[int(.025 * len(r))]), float(r[int(.975 * len(r))])] if len(r) >= 100 else None
        return obs, ci, int((~ok).sum())


def micro(p):
    return int(round(p * MICRO))


def rnd(x, k=5):
    return None if x is None else round(x, k)


def r5(pair):
    m, ci = pair
    return {'mean': rnd(m), 'ci95': None if ci is None else [rnd(ci[0]), rnd(ci[1])]}


def vec(rows, c):
    return np.array([micro(r['cond'][c]['p_gt']) for r in rows], dtype=np.int64)


def contrasts(P):
    """The integer difference vectors of the plan, from P = {cond: int64 vector}."""
    d = {'AB': P['A'] - P['Bb'], 'CB': P['Cb'] - P['Bb'], 'CD': P['Cb'] - P['Db'], 'DB': P['Db'] - P['Bb'],
         'A+ - A': P['A+'] - P['A'], 'A+ - Bb': P['A+'] - P['Bb'],
         'C - B': P['C'] - P['B'], 'C - D': P['C'] - P['D'], 'B - Bb': P['B'] - P['Bb']}
    for x in 'noe':
        d['S_' + x] = P['C' + x] - P['D' + x]
    d['S_o - S_n'] = d['S_o'] - d['S_n']
    d['S_e - S_n'] = d['S_e'] - d['S_n']
    d['S_o - S_e'] = d['S_o'] - d['S_e']
    return d


def decide(bt, P):
    """The plan's steps 1-3 on the integer vectors P; returns the verdict block."""
    d = contrasts(P)
    ab, ab_ci = bt.mean_ci(d['AB'])
    info = bool(ab_ci[0] > 0 and ab >= INFO_MIN)
    out = {'step1': {'AB': r5((ab, ab_ci)), 'info_present': info, 'label': V1[info], 'threshold_min_AB': INFO_MIN}}
    if not info:
        out['stopped_after_step1'] = True
        out['verdict'] = V1_NO_INFO
        return out, d
    cb, cb_ci = bt.mean_ci(d['CB'])
    cd, cd_ci = bt.mean_ci(d['CD'])
    if cb_ci[1] < BARELY * ab:
        key = 'barely'
    elif cb_ci[0] > 0 and cd_ci[0] > 0:
        key = 'works'
    else:
        key = 'undecided'
    out['step2'] = {'CB': r5((cb, cb_ci)), 'CD': r5((cd, cd_ci)), 'key': key, 'label': V2[key],
                    'barely_bound_0.25_AB': rnd(BARELY * ab), 'note': V2_NOTE if (cd_ci[0] > 0 and cb_ci[0] <= 0 <= cb_ci[1]) else None}
    diff, diff_ci = bt.mean_ci(d['S_o - S_n'])
    k3 = 'more' if diff_ci[0] > 0 else 'less' if diff_ci[1] < 0 else 'unclear'
    out['step3'] = {'S_o - S_n': r5((diff, diff_ci)), 'key': k3, 'label': V3[k3], 'exploratory': True, 'noise_note': V3_NOISE,
                    'read_with_step2': V2[key]}
    nxt = []
    if key == 'works' and k3 == 'unclear':
        nxt.append(NEXT['works_no_pos_diff'])
    if key == 'barely':
        nxt.append(NEXT['barely'])
    if k3 == 'more':
        nxt.append(NEXT['pos_gain'])
    out['next_actions'] = nxt
    out['verdict'] = f'{V1[True]} / {V2[key]} / {V3[k3]}'
    return out, d


def describe(bt, P, d):
    """Descriptive items of the plan (step 2 description, 2-aux, step 3 description)."""
    out = {}
    for k in ('DB', 'A+ - A', 'C - B', 'C - D', 'B - Bb', 'S_e - S_n', 'S_o - S_e'):
        out[k] = r5(bt.mean_ci(d[k]))
    for x in 'noe':
        out['S_' + x] = r5(bt.mean_ci(d['S_' + x]))
    for name, num, den in (('CB / AB', 'CB', 'AB'), ('CB / (A+ - Bb), length matched', 'CB', 'A+ - Bb'),
                           ('S_n / CD', 'S_n', 'CD'), ('S_o / CD', 'S_o', 'CD'), ('S_e / CD', 'S_e', 'CD')):
        o, ci, bad = bt.ratio_ci(d[num], d[den])
        out[name] = {'ratio': rnd(o), 'ci95': None if ci is None else [rnd(ci[0]), rnd(ci[1])], 'resamples_with_nonpositive_denominator': bad}
    return out


def condition_table(rows):
    t = {}
    for c in CONDS:
        v = [r['cond'][c] for r in rows]
        t[c] = {'mean_p_gt': rnd(statistics.mean(x['p_gt'] for x in v), 5) if v else None,
                'mean_logp_gt': rnd(statistics.mean(x['logp_gt'] for x in v), 4) if v else None,
                'top1_is_gt_first_share': rnd(sum(bool(x['top1_is_gt_first']) for x in v) / len(v), 4) if v else None,
                'median_ctx_len': statistics.median(x['ctx_len'] for x in v) if v else None,
                'median_entropy_first': rnd(statistics.median(x['entropy_first'] for x in v), 4) if v else None}
    return t


def split(rows):
    """Descriptive: every contrast for multiple choice and free form questions, own bootstrap per group."""
    out = {}
    for g in ('multi_choice', 'free_form'):
        rg = [r for r in rows if r.get('question_type') == g]
        o = {'n': len(rg)}
        if rg:
            P = {c: vec(rg, c) for c in CONDS}
            d = contrasts(P)
            bt = Boot(len(rg))
            o['means_p_gt'] = {c: rnd(float(P[c].sum()) / (len(rg) * MICRO)) for c in CONDS}
            o['contrasts'] = {k: r5(bt.mean_ci(d[k])) for k in ('AB', 'CB', 'CD', 'DB', 'A+ - A', 'C - B', 'C - D', 'B - Bb',
                                                                 'S_n', 'S_o', 'S_e', 'S_o - S_n', 'S_e - S_n', 'S_o - S_e')}
        out[g] = o
    return out


def select(rows, ids=None):
    """(rows used, counts of what was left out)."""
    trial = set(ids['trial']) if ids else set()
    main = set(ids['main']) if ids else None
    c = {'n_rows': len(rows), 'n_trial_excluded': 0, 'n_error': 0, 'n_outside_main_list': 0, 'n_incomplete': 0}
    used = []
    for r in rows:
        if r.get('set') == 'trial' or r['pid'] in trial:
            c['n_trial_excluded'] += 1
        elif r.get('status') != 'ok':
            c['n_error'] += 1
        elif main is not None and r['pid'] not in main:
            c['n_outside_main_list'] += 1
        elif any(k not in r.get('cond', {}) for k in CONDS):
            c['n_incomplete'] += 1
        else:
            used.append(r)
    c['n_used'] = len(used)
    c['n_main_expected'] = len(main) if main is not None else None
    c['n_main_missing'] = len(main - {r['pid'] for r in used}) if main is not None else None
    return used, c


def analyze(rows, ids=None):
    used, counts = select(rows, ids)
    res = {'label': 'pre-registered recovery check; rules fixed before the results (plan_recover.md section 5)',
           'code_versions': sorted({r['code_version'] for r in rows if 'code_version' in r}), 'counts': counts,
           'bootstrap': {'resamples': B, 'seed': SEED, 'level': 0.95}}
    if not used:
        res['verdict'] = None
        return res
    bt = Boot(len(used))
    P = {c: vec(used, c) for c in CONDS}
    verdict, d = decide(bt, P)
    res.update(verdict)
    res['means_p_gt'] = {c: rnd(float(P[c].sum()) / (len(used) * MICRO)) for c in CONDS}
    if not verdict['step1']['info_present']:
        res['split'] = {g: {'n': sum(r.get('question_type') == g for r in used)} for g in ('multi_choice', 'free_form')}
        return res
    res['described'] = describe(bt, P, d)
    res['conditions'] = condition_table(used)
    res['kl_first_position'] = {k: {'median': rnd(statistics.median(r['kl_pairs'][k] for r in used), 5),
                                    'mean': rnd(statistics.mean(r['kl_pairs'][k] for r in used), 5)} for k in PAIRS}
    res['split'] = split(used)
    return res


def load(files):
    rows, seen = [], set()
    for f in files:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                assert r['pid'] not in seen, f'duplicate row {r["pid"]}'
                seen.add(r['pid'])
                rows.append(r)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('rows', nargs='+')
    ap.add_argument('--ids', default=str(REPRO / 'diag/recover_check_ids.json'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    files = [g for p in a.rows for g in (sorted(glob.glob(p)) or [p])]
    res = analyze(load(files), json.loads(Path(a.ids).read_text()))
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'counts': res['counts'], 'verdict': res.get('verdict')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
