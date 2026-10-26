"""Analysis of tally_force diagnostic.

Measures T=1 accuracy differences between condition D (DPP-selected image tokens re-inserted)
and condition N (no tokens re-inserted).

Inputs:
  Result JSONL files + ids file (diag/tally_force_ids.json) + --set (main|trial).
Analysis questions = the first n_analysis items of the set in the ids file.
If any analysis question is missing or has an error status, verdict is HOLD.

Primary metric:
  Per question count of correct answers among 3 sampled T=1 answers (excluding greedy).
  Difference D - N in pp = mean(count_D - count_N) / 3 * 100.
  Paired bootstrap 20,000 resamples (np.random.default_rng(0)), 95% range.
  Exact two-sided sign test on changed questions (sign_p from analyze_closed_multistep).
Verdict:
  UP = lower CI > 0 and p < 0.05
  DOWN = upper CI < 0 and p < 0.05
  NOT = otherwise
Outputs up : down counts.

Descriptive (not for decision):
  - Greedy accuracy N and D
  - Step 1 end type (cap/close/eos) distribution
  - D - N by step 1 end type
  - Invalid answer (valid == False) rate per condition
  - n_sel distribution
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
sys.path.insert(0, str(HERE))

from analyze_closed_multistep import sign_p  # noqa: E402


def expand(patterns):
    return sorted({g for p in patterns for g in (glob.glob(p) or ([p] if Path(p).exists() else []))})


def load(patterns):
    """{pid: row}. A pid seen twice (a re-run of an error row) is kept only when exactly one of its rows is 'ok' (that row is
    used); otherwise exit 1."""
    seen, files = {}, expand(patterns)
    if not files:
        raise SystemExit('no result files match')
    for f in files:
        for line in Path(f).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                seen.setdefault(str(row['pid']), []).append(row)
    R = {}
    for pid, rows in seen.items():
        ok = [x for x in rows if x.get('status') == 'ok']
        if len(rows) > 1 and len(ok) != 1:
            sys.exit(f'pid {pid} seen {len(rows)} times with {len(ok)} ok rows')
        R[pid] = ok[0] if ok else rows[0]
    return R, files


def boot_sums(d, n=20000, seed=0, chunk=2000):
    """Sums of n bootstrap resamples of integer differences d (exact int64)."""
    d = np.asarray(d, dtype=np.int64)
    rng = np.random.default_rng(seed)
    out = []
    for s in range(0, n, chunk):
        idx = rng.integers(0, len(d), size=(min(chunk, n - s), len(d)))
        out.append(d[idx].sum(1))
    return np.concatenate(out)


def analyze(R, target_pids):
    """Run analysis over target_pids using result rows R."""
    missing = [p for p in target_pids if p not in R]
    errors = [p for p in target_pids if p in R and R[p].get('status') != 'ok']

    if missing or errors:
        return {
            'verdict': 'HOLD',
            'n_target': len(target_pids),
            'n_used': 0,
            'missing': missing,
            'errors': errors,
            'sentence': f"고정한 분석 문항 중 결과가 없거나(누락 {len(missing)}) 오류인 문항(오류 {len(errors)})이 있어 판정을 보류한다."
        }

    rows = [R[p] for p in target_pids]
    n = len(rows)
    if n == 0:
        return {'verdict': 'HOLD', 'n_target': 0, 'n_used': 0, 'missing': [], 'errors': [], 'sentence': '분석 문항이 없습니다.'}

    counts_n = np.array([r['n_correct']['N'] for r in rows], dtype=np.int64)
    counts_d = np.array([r['n_correct']['D'] for r in rows], dtype=np.int64)
    d = counts_d - counts_n

    scale = 3 * n
    diff_mean_pp = float(100.0 * d.sum() / scale)
    raw = boot_sums(d, 20000, seed=0)
    bs = np.sort(raw)
    ci_lo = float(100.0 * bs[int(0.025 * 20000)] / scale)
    ci_hi = float(100.0 * bs[int(0.975 * 20000)] / scale)

    up = int((d > 0).sum())
    down = int((d < 0).sum())
    same = int((d == 0).sum())
    p = sign_p(up, down)

    if ci_lo > 0 and p < 0.05:
        verdict = 'UP'
    elif ci_hi < 0 and p < 0.05:
        verdict = 'DOWN'
    else:
        verdict = 'NOT'

    # Descriptive statistics
    greedy_acc = {c: float(np.mean([r['greedy'][c] for r in rows])) for c in ('N', 'D')}
    step1_ends = {k: sum(r['step1']['end'] == k for r in rows) for k in ('cap', 'close', 'eos')}

    diff_by_end = {}
    for end_k in ('cap', 'close', 'eos'):
        sub = [r for r in rows if r['step1']['end'] == end_k]
        if sub:
            sub_d = [r['n_correct']['D'] - r['n_correct']['N'] for r in sub]
            diff_by_end[end_k] = float(np.mean(sub_d) / 3 * 100)
        else:
            diff_by_end[end_k] = None

    invalid_rate = {}
    for c in ('N', 'D'):
        all_ans = [a for r in rows for a in r.get('answers', {}).get(c, [])]
        invalid_rate[c] = float(sum(not a.get('valid', False) for a in all_ans) / len(all_ans)) if all_ans else 0.0

    n_sels = [r['n_sel']['own'] if isinstance(r.get('n_sel'), dict) else r.get('n_sel', 0) for r in rows]
    n_sel_dist = {
        'median': float(np.median(n_sels)) if n_sels else None,
        'mean': float(np.mean(n_sels)) if n_sels else None,
        'min': int(np.min(n_sels)) if n_sels else None,
        'max': int(np.max(n_sels)) if n_sels else None,
    }

    return {
        'verdict': verdict,
        'n_target': len(target_pids),
        'n_used': n,
        'missing': [],
        'errors': [],
        'up': up,
        'down': down,
        'same': same,
        'changed_ratio': f'{up}:{down}',
        'p_sign': p,
        'diff_D_minus_N_pp': diff_mean_pp,
        'ci95_pp': [ci_lo, ci_hi],
        'descriptive': {
            'greedy_acc': greedy_acc,
            'step1_ends': step1_ends,
            'diff_by_step1_end_pp': diff_by_end,
            'invalid_rate': invalid_rate,
            'n_sel_dist': n_sel_dist,
        }
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='*')
    ap.add_argument('--ids', default=str(REPRO / 'diag/tally_force_ids.json'))
    ap.add_argument('--set', default='main', choices=('main', 'trial'))
    ap.add_argument('--n-analysis', type=int, default=0, help='number of items to analyze (0 = all)')
    ap.add_argument('--out', required=True)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    doc = json.loads(Path(a.ids).read_text())
    target_list = doc[a.set]
    if a.n_analysis > 0:
        target_pids = [str(p) for p in target_list[:a.n_analysis]]
    else:
        target_pids = [str(p) for p in target_list]

    R, files = load(a.results)
    out = analyze(R, target_pids)
    out['files'] = files

    out_p = Path(a.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'verdict': out['verdict'], 'up': out.get('up'), 'down': out.get('down'),
                      'p_sign': out.get('p_sign'), 'diff_pp': out.get('diff_D_minus_N_pp')}))


if __name__ == '__main__':
    main()
