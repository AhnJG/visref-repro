"""Analysis of `slot_cross` (diag/slot_cross.py; plan docs/EXPERIMENT_FLOW.md 2026-10-24 "실행 전 계획: 그림 자리 교차 진단", written
before any result). Implements the plan's decision rule; the Korean sentences are the plan's, kept as constants.

Data: the result JSONL files and the analysis list (diag/slot_cross_analysis_ids.json 'analysis', fixed from the trial's timing
before the main run with analyze_blank_think.make_analysis_ids). A pid seen twice exits 1. If any analysis question has no row
or an error row: no verdict (HOLD).
Statistics: p_gt is stored with 6 decimals, so differences are exact integer counts of millionths; paired question bootstrap,
20,000 resamples, numpy default_rng(0) per comparison (analyze_step1_think.boot_sums), 95% range.
  G   = p(o-) - p(b-)       gate: lower end > 0 AND mean >= 0.05, else VERDICT_NO_INFO and nothing else is decided
  R_S = p(ob) - p(bb)       recovery through the prompt's own slots (after the generated thinking, same length as R_P)
  R_P = p(bo) - p(bb)       recovery through the block put back after the thinking
  D   = p(ob) - p(bo)       same image information and length, swapped places
Sentences (all three are written): R_S / R_P lower end > 0 -> REC[path][True], else REC[path][False]; D lower end > 0 ->
POS['slot'], upper end < 0 -> POS['post'], else POS['none']. "되살렸다" is only written for a path whose recovery is confirmed.
Described: R_P / R_S (only when R_S is confirmed; per-resample ratio of sums, 95% range), oo - o-, the cells' mean p, and every
quantity split into multiple choice / free form.

  cd repro && python3 diag/analyze_slot_cross.py 'kaggle/runs/slot_cross_main/output/results/slot_cross_*.jsonl' \\
      [--ids diag/slot_cross_analysis_ids.json] --out diag/slot_cross_analysis.json [--force]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_blank_think import load  # noqa: E402
from analyze_closed_multistep import rel  # noqa: E402
from analyze_step1_think import B, boot_sums  # noqa: E402

CELLS = ('o-', 'b-', 'oo', 'ob', 'bo', 'bb')
MICRO, GATE_MIN = 1_000_000, 0.05
VERDICT_NO_INFO = '생각 뒤 문맥에서 그림을 빈 그림으로 바꿔도 정답 확률이 뚜렷이 바뀌지 않아, 이 진단은 정보가 부족하다.'
HOLD = '고정한 분석 문항 중 결과가 없거나 오류인 문항이 있어 판정을 보류한다.'
REC = {'S': {True: '생각 뒤 문맥에서도 원래 자리의 그림은 정답 확률을 되살렸다.',
             False: '생각 뒤 문맥에서 원래 자리의 그림이 정답 확률을 되살리는 것은 확인하지 못했다.'},
       'P': {True: '생각 뒤에 넣은 그림이 정답 확률을 되살렸다.',
             False: '생각 뒤에 넣은 그림이 정답 확률을 되살리는 것은 확인하지 못했다.'}}
POS = {'slot': '같은 그림·같은 길이에서, 원래 자리에 둔 배치가 생각 뒤에 둔 배치보다 정답 확률이 높았다.',
       'post': '같은 그림·같은 길이에서, 생각 뒤에 둔 배치가 원래 자리에 둔 배치보다 정답 확률이 높았다.',
       'none': '같은 그림·같은 길이에서 자리에 따른 정답 확률 차이를 확인하지 못했다.'}


def micro(p):
    return int(round(p * MICRO))


def vec(rows, c):
    return np.array([micro(x['cell'][c]['p_gt']) for x in rows], dtype=np.int64)


def mean_ci(d):
    """(mean, [lo, hi]) in probability units of an integer (millionths) difference vector."""
    n = len(d)
    s = np.sort(boot_sums(d))
    return int(d.sum()) / n / MICRO, [int(s[int(.025 * B)]) / n / MICRO, int(s[int(.975 * B)]) / n / MICRO]


def ratio_ci(num, den):
    """R_P / R_S as ratio of sums with a per-resample 95% range (same resample indices for both)."""
    rng = np.random.default_rng(0)
    n = len(num)
    rs = []
    for s in range(0, B, 2000):
        idx = rng.integers(0, n, size=(min(2000, B - s), n))
        a, b = num[idx].sum(1), den[idx].sum(1)
        rs.append(np.where(b != 0, a / np.where(b == 0, 1, b), np.nan))
    rs = np.sort(np.concatenate(rs)[~np.isnan(np.concatenate(rs))])
    return float(num.sum() / den.sum()), [float(rs[int(.025 * len(rs))]), float(rs[int(.975 * len(rs))])]


def rd(x, k=5):
    return None if x is None else round(x, k)


def block(m_ci):
    m, ci = m_ci
    return {'mean': rd(m), 'ci95': [rd(ci[0]), rd(ci[1])]}


def decide(rows):
    P = {c: vec(rows, c) for c in CELLS}
    g = mean_ci(P['o-'] - P['b-'])
    out = {'G': block(g), 'gate': bool(g[1][0] > 0 and g[0] >= GATE_MIN)}
    if not out['gate']:
        out['sentence'] = VERDICT_NO_INFO
        return out
    rs, rp, dd = mean_ci(P['ob'] - P['bb']), mean_ci(P['bo'] - P['bb']), mean_ci(P['ob'] - P['bo'])
    ok_s, ok_p = rs[1][0] > 0, rp[1][0] > 0
    pos = 'slot' if dd[1][0] > 0 else 'post' if dd[1][1] < 0 else 'none'
    out.update(R_S=block(rs), R_P=block(rp), D=block(dd), R_S_confirmed=bool(ok_s), R_P_confirmed=bool(ok_p), position=pos,
               sentence=' '.join([REC['S'][ok_s], REC['P'][ok_p], POS[pos]]))
    if ok_s:
        m, ci = ratio_ci(P['bo'] - P['bb'], P['ob'] - P['bb'])
        out['R_P_over_R_S'] = {'mean': rd(m, 4), 'ci95': [rd(ci[0], 4), rd(ci[1], 4)]}
    return out


def describe(rows):
    P = {c: vec(rows, c) for c in CELLS}
    return {'mean_p': {c: rd(float(P[c].mean()) / MICRO) for c in CELLS},
            'oo - o-': block(mean_ci(P['oo'] - P['o-'])), 'bo - b-': block(mean_ci(P['bo'] - P['b-'])),
            'step1_end': {e: sum(x['step1']['end'] == e for x in rows) for e in ('close', 'cap', 'eos')}}


def analyze(R, ids, files=None, ids_file=None):
    want = [str(p) for p in ids['analysis']]
    used = [R[p] for p in want if p in R and R[p].get('status') == 'ok']
    cov = {'n_analysis': len(want), 'n_used': len(used), 'n_rows': len(R), 'n_outside_analysis': sum(p not in set(want) for p in R),
           'errors': sorted(p for p in want if p in R and R[p].get('status') != 'ok'), 'n_missing': sum(p not in R for p in want),
           'code_versions': sorted({x['code_version'] for x in R.values() if 'code_version' in x})}
    out = {'files': [rel(f) for f in (files or [])], 'ids_file': rel(ids_file) if ids_file else None, 'coverage': cov}
    if not used or cov['errors'] or cov['n_missing']:
        out.update(decision=None, sentence=HOLD)
        return out
    d = decide(used)
    out.update(decision=d, sentence=d['sentence'], descriptive=describe(used))
    out['by_question_type'] = {qt: {'n': len(sub), 'decision_quantities': decide(sub), **describe(sub)}
                               for qt in sorted({x['question_type'] for x in used})
                               for sub in [[x for x in used if x['question_type'] == qt]]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='+')
    ap.add_argument('--ids', default=str(HERE / 'slot_cross_analysis_ids.json'))
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
