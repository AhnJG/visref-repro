"""Analysis of `cap_pair` (plan docs/EXPERIMENT_FLOW.md 2026-10-24 "실행 전 계획: 단계 상한 1,024 대 4,096", written before any result).

Data: run.py rows of the two settings, results_1024_*.jsonl and results_4096_*.jsonl of one run folder (kaggle/build_cap_pair.py),
and diag/cap_pair_ids.json (42 questions). Key = (cap, pid, method); a key seen twice exits 1. Every (cap, pid, method) of the
42 x 3 x 2 = 252 must have a row with status 'ok', else there is no verdict (HOLD). Correctness = run.py's rule-scored `correct`.

(a) ST(4096) - ST(1024): paired by question, percentile bootstrap (20,000 resamples, numpy default_rng(0),
    analyze_step1_think.boot_sums) and the exact two-sided sign test on the changed questions (= exact McNemar). The ST
    sentence: lower end > 0 and p < 0.05 -> UP; upper end < 0 and p < 0.05 -> DOWN; else NOT_CONFIRMED (+ range, up : down).
(b) [VisRef - TSR](4096) - [VisRef - TSR](1024): per-question integer in -2..2; mean, bootstrap range, counts of questions
    with a positive / negative value. Descriptive only (no sentence).
(c) Descriptive: per cap and method accuracy, share of first steps cut at the cap, mean number of steps k, share of rows
    whose final answer needed answer forcing, stop reasons, mean thinking tokens; TSR - ST and VisRef - TSR within each cap.

  cd repro && python3 diag/analyze_cap_pair.py kaggle/runs/cap_pair_main/output/results --out diag/cap_pair_analysis.json [--force]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_closed_multistep import pair_stats, pub, rel  # noqa: E402

CAPS = (1024, 4096)
METHODS = ('st', 'tsr', 'visref_raw_rank')
ALPHA = 0.05
UP = '단계 상한을 4,096으로 올리면 ST 정답률이 올랐다(탐색, 42문항).'
DOWN = '단계 상한을 4,096으로 올리면 ST 정답률이 내려갔다(탐색, 42문항).'
NOT_CONFIRMED = '42문항으로는 상한에 따른 ST 정답률 변화를 확인하지 못했다.'
HOLD = '고정한 42문항 × 3방법 × 2설정 중 결과가 없거나 오류인 행이 있어 판정을 보류한다.'


def load(folder):
    """{(cap, pid, method): row}; a key seen twice exits 1."""
    R, files = {}, []
    for cap in CAPS:
        for f in sorted(Path(folder).glob(f'results_{cap}_*.jsonl')):
            files.append(str(f))
            for line in f.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    key = (cap, str(r['pid']), r['method'])
                    if key in R:
                        sys.exit(f'{key} seen twice ({f})')
                    R[key] = r
    return R, files


def complete(R, ids):
    keys = [(c, p, m) for c in CAPS for p in ids for m in METHODS]
    missing = [k for k in keys if k not in R]
    bad = [k for k in keys if k in R and R[k].get('status') != 'ok']
    return missing, bad


def corr(R, ids, cap, m):
    return np.array([int(R[(cap, p, m)]['correct']) for p in ids], dtype=np.int64)


def st_sentence(st):
    if st['ci95_pp'][0] > 0 and st['p_sign'] < ALPHA:
        return UP
    if st['ci95_pp'][1] < 0 and st['p_sign'] < ALPHA:
        return DOWN
    return NOT_CONFIRMED + f" (95% 범위 {st['ci95_pp'][0]:+.1f} ~ {st['ci95_pp'][1]:+.1f} pp, 좋아짐 {st['up']} : 나빠짐 {st['down']})"


def diff_of_diffs(R, ids):
    d = {cap: corr(R, ids, cap, 'visref_raw_rank') - corr(R, ids, cap, 'tsr') for cap in CAPS}
    x = d[4096] - d[1024]
    st = pair_stats(x, np.zeros_like(x), 1)  # bootstrap of the mean of x; its sign test counts questions with x > 0 vs < 0
    return {'mean_pp': st['mean_pp'], 'ci95_pp': st['ci95_pp'], 'n_pos': int((x > 0).sum()), 'n_neg': int((x < 0).sum())}


def describe(R, ids):
    out = {}
    for cap in CAPS:
        o = {}
        for m in METHODS:
            rows = [R[(cap, p, m)] for p in ids]
            o[m] = {'acc_pct': round(100 * sum(int(r['correct']) for r in rows) / len(rows), 2),
                    'first_step_cut': sum(r['steps'][0].get('think_end') == 'cap' for r in rows if r.get('steps')),
                    'mean_k': round(statistics.mean(r['k'] for r in rows), 3),
                    'forced_answer': sum(bool(r.get('final_forced_answer')) for r in rows),
                    'mean_think_tokens': round(statistics.mean(r['think_tokens'] for r in rows), 1),
                    'stop_reason': {s: sum(r['stop_reason'] == s for r in rows) for s in sorted({r['stop_reason'] for r in rows})}}
        o['tsr-st'] = pub(pair_stats(corr(R, ids, cap, 'tsr'), corr(R, ids, cap, 'st'), 1))
        o['visref-tsr'] = pub(pair_stats(corr(R, ids, cap, 'visref_raw_rank'), corr(R, ids, cap, 'tsr'), 1))
        out[str(cap)] = o
    return out


def analyze(R, ids, files=None):
    missing, bad = complete(R, ids)
    out = {'files': [rel(f) for f in files or []], 'n_ids': len(ids), 'n_rows': len(R), 'missing': [list(k) for k in missing],
           'errors': [list(k) for k in bad], 'code_versions': sorted({r.get('code_version') for r in R.values()} - {None})}
    if missing or bad:
        out['sentence'] = HOLD
        return out
    st = pair_stats(corr(R, ids, 4096, 'st'), corr(R, ids, 1024, 'st'), 1)
    out.update(st_4096_minus_1024=pub(st), sentence=st_sentence(st), visref_tsr_change=diff_of_diffs(R, ids),
               descriptive=describe(R, ids))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--ids', default=str(HERE / 'cap_pair_ids.json'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and not a.force:
        sys.exit(f'{out} exists (use --force)')
    R, files = load(a.folder)
    res = analyze(R, [str(p) for p in json.loads(Path(a.ids).read_text())['ids']], files)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'sentence': res['sentence'], 'n_rows': res['n_rows']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
