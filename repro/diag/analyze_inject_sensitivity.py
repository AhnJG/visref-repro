"""Analysis of the injection-sensitivity check (EF 2026-10-18 "실행 전 계획: 이미지 민감도 점검", readings as changed
before the main run in "5문항 시험과 본 실행 전 고침"; both written before the results).

Descriptive diagnostic (no verdict on VisRef). Part 1 per group (all / step 1 cut / step 1 closed / multiple choice /
free form / right at none (p_gt >= 0.5 in the context none) / not right at none; the readings use 'all' only): for each
context c in dpp, rand, all, other, dpp_tag against none: mean change of p(right answer) and of log p(right answer),
mean change of log p(TSR's answer; rows whose stored TSR answer was not valid are left out), median and mean
KL(p_none || p_c) at the first answer position, share of questions whose most likely first answer token changes.
Paired contrasts with a question bootstrap 95% CI (2,000 resamples, seed 0): all - other, dpp - rand, dpp_tag - dpp on
the change of p(right answer) and of log p(right answer); the pairwise KLs kl(all || other), kl(dpp || rand),
kl(dpp_tag || dpp) (median, mean) and the share of questions whose most likely first answer token differs between the
two contexts. Rows with status 'error' are counted and left out.
Readings fixed before the results (they steer the next step, they do not test the paper):
  moves     the model reads the content of the re-inserted image tokens: in 'all' the most likely first answer token
            differs from none in >= 5% of questions AND it differs between all and other (another question's image with
            as many tokens) in >= 5% of questions. Direction-free: another image confusing the model also counts.
            Whether this question's image helps or another image hurts is described from all - none and other - none.
  selection the selection matters: the CI of dpp - rand on the change of p(right answer) lies above 0
Part 2: for each question, the mean of the finite stopping entropies of its 8 sampled answers (the main run's estimator,
averaged over 8 samples instead of 1), in the contexts none and dpp; per half ('above' / 'below' 0.25 by the stored
main-run TSR single-sample value) the number of questions whose 8-sample decision (mean above delta = 0.25) differs from
the stored single-sample one. Description only (the first version's 'continue' reading was dropped: on the questions it
used, the stored value was above 0.25 for 1 of 100).

  cd repro && python3 diag/analyze_inject_sensitivity.py ROWS.jsonl [ROWS2.jsonl ...] --out diag/inject_sensitivity_analysis.json
"""
import argparse
import glob
import json
import random
import statistics
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parent.parent
DELTA, B, SEED = 0.25, 2000, 0
CONDS = ('dpp', 'rand', 'all', 'other', 'dpp_tag')
PAIRS = {'all-other': ('all', 'other'), 'dpp-rand': ('dpp', 'rand'), 'dpp_tag-dpp': ('dpp_tag', 'dpp')}
KL_OF = {'all-other': 'all||other', 'dpp-rand': 'dpp||rand', 'dpp_tag-dpp': 'dpp_tag||dpp'}
RUNS = 'kaggle/runs/b_main_s*/output/results/results_*.jsonl'


def mean(xs):
    return statistics.mean(xs) if xs else None


def median(xs):
    return statistics.median(xs) if xs else None


def rnd(x, k):
    return None if x is None else round(x, k)


def boot_ci(xs, b=B, seed=SEED):
    if not xs:
        return None
    rng, n = random.Random(seed), len(xs)
    m = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return [round(m[int(.025 * b)], 4), round(m[int(.975 * b)], 4)]


def summary(rows):
    out = {'n': len(rows)}
    cond = lambda r, c: r['part1']['cond'][c]  # noqa: E731
    for c in CONDS:
        d_p = [cond(r, c)['p_gt'] - cond(r, 'none')['p_gt'] for r in rows]
        d_gt = [cond(r, c)['logp_gt'] - cond(r, 'none')['logp_gt'] for r in rows]
        d_ts = [cond(r, c)['logp_tsr'] - cond(r, 'none')['logp_tsr'] for r in rows
                if cond(r, c)['logp_tsr'] is not None and r['part1'].get('tsr_valid', True)]
        k = [cond(r, c)['kl_from_none'] for r in rows]
        top = [cond(r, c)['top1'] != cond(r, 'none')['top1'] for r in rows]
        out[c] = {'mean_dp_gt': rnd(mean(d_p), 5), 'ci_dp_gt': boot_ci(d_p),
                  'mean_dlogp_gt': rnd(mean(d_gt), 4), 'ci_dlogp_gt': boot_ci(d_gt),
                  'mean_dlogp_tsr': rnd(mean(d_ts), 4),
                  'median_kl': rnd(median(k), 5), 'mean_kl': rnd(mean(k), 5),
                  'top1_changed': sum(top), 'top1_changed_share': round(sum(top) / len(top), 4) if top else None}
    for name, (a, b) in PAIRS.items():
        x = [cond(r, a)['p_gt'] - cond(r, b)['p_gt'] for r in rows]
        y = [cond(r, a)['logp_gt'] - cond(r, b)['logp_gt'] for r in rows]
        kk = [r['part1']['kl_pairs'][KL_OF[name]] for r in rows]
        t = [cond(r, a)['top1'] != cond(r, b)['top1'] for r in rows]
        out[name] = {'mean_p_gt_diff': rnd(mean(x), 5), 'ci_p_gt_diff': boot_ci(x),
                     'mean_logp_gt_diff': rnd(mean(y), 4), 'ci_logp_gt_diff': boot_ci(y),
                     'median_kl': rnd(median(kk), 5), 'mean_kl': rnd(mean(kk), 5),
                     'top1_differs': sum(t), 'top1_differs_share': round(sum(t) / len(t), 4) if t else None}
    s1 = out['all']['top1_changed_share'] or 0
    s2 = out['all-other']['top1_differs_share'] or 0
    out['moves'] = bool(rows) and s1 >= 0.05 and s2 >= 0.05
    cs = out['dpp-rand']['ci_p_gt_diff']
    out['selection'] = cs is not None and cs[0] > 0
    return out


def stored_H(files):
    h = {}
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if r.get('method') in ('tsr', 'visref_raw_rank') and r.get('steps'):
                h[(str(r['pid']), r['method'])] = r['steps'][0].get('H')
    return h


def part2(rows, H):
    out = {'n': len(rows)}
    for g in ('above', 'below'):
        rg = [r for r in rows if r.get('part2_group') == g]
        o = {'n': len(rg)}
        for c in ('none', 'dpp'):
            h = [r['part2'][c]['H_mean'] for r in rg if r['part2'][c]['H_mean'] is not None]
            o[c] = {'n': len(h), 'no_valid_answer': len(rg) - len(h), 'above_delta': sum(x > DELTA for x in h),
                    'median_H_mean': rnd(median(h), 6),
                    'answer_strings_entropy_above_delta': sum(x > DELTA for x in
                                                              [r['part2'][c]['entropy'] for r in rg if r['part2'][c]['entropy'] is not None]),
                    'n_valid_mean': rnd(mean([r['part2'][c]['n_valid'] for r in rg]), 3)}
        flips = [r for r in rg if r['part2']['none']['H_mean'] is not None and H.get((r['pid'], 'tsr')) is not None
                 and (r['part2']['none']['H_mean'] > DELTA) != (H[(r['pid'], 'tsr')] > DELTA)]
        o['decision_differs_from_stored'] = len(flips)
        out[g] = o
    return out


def analyze(rows, H):
    ok = [r for r in rows if r.get('status', 'ok') == 'ok']
    p1 = [r for r in ok if 'part1' in r and all(c in r['part1']['cond'] for c in ('none',) + CONDS)]
    p2 = [r for r in ok if 'part2' in r]
    groups = {'all': p1, 'cap': [r for r in p1 if r['stratum'] == 'cap'], 'close': [r for r in p1 if r['stratum'] == 'close'],
              'multi_choice': [r for r in p1 if r['part1'].get('question_type') == 'multi_choice'],
              'free_form': [r for r in p1 if r['part1'].get('question_type') == 'free_form'],
              'right_at_none': [r for r in p1 if r['part1']['cond']['none']['p_gt'] >= 0.5],
              'not_right_at_none': [r for r in p1 if r['part1']['cond']['none']['p_gt'] < 0.5]}
    versions = sorted({r['code_version'] for r in rows})
    return {'label': 'descriptive diagnostic; readings fixed before the results', 'code_versions': versions,
            'n_rows': len(rows), 'n_error': len(rows) - len(ok), 'n_part1_incomplete': sum('part1' in r for r in ok) - len(p1),
            'part1': {g: summary(v) for g, v in groups.items()}, 'part2': part2(p2, H)}


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
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    res = analyze(load(a.rows), stored_H(glob.glob(str(REPRO / RUNS))))
    Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({g: {k: res['part1'][g][k] for k in ('n', 'moves', 'selection')} for g in res['part1']}))
    print(json.dumps({g: res['part2'][g].get('decision_differs_from_stored') for g in ('above', 'below')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
