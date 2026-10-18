"""Analysis of the injection-sensitivity check (EF 2026-10-18 "실행 전 계획: 이미지 민감도 점검"; written before the results).

Descriptive diagnostic (no verdict on VisRef). Part 1 per group (all / step 1 cut / step 1 closed / multiple choice /
free form; the readings use 'all' only): for each context c in dpp, rand, all, other, dpp_tag against none: mean change
of log p(right answer) and of log p(TSR's answer; rows whose stored TSR answer was not valid are left out), median and
mean KL(p_none || p_c) at the first answer position, share of questions whose most likely first answer token changes.
Paired contrasts with a question bootstrap 95% CI (2,000 resamples, seed 0) on the change of log p(right answer), KL
alongside as description: all - other (image content against a length-matched control: another question's image tokens
of the same number), dpp - rand (the selection against random tokens of the same number), dpp_tag - dpp (the tags).
Readings fixed before the results (they steer the next step, they do not test the paper):
  (moves asks whether the model reads the content of re-inserted image tokens at all; another question's image confusing
  the model also counts, so the direction is not used. Whether this question's image helps or another image hurts is
  described from the per-context changes against none: all - none and other - none.)
  moves     the image content moves the answer: in 'all', the most likely first token changes in >= 5% of questions
            AND the CI of all - other on the change of log p(right answer) excludes 0
  selection the selection matters: the CI of dpp - rand on the change of log p(right answer) lies above 0
Part 2: for each closed question, the mean of the finite stopping entropies of its 8 sampled answers (the main run's
estimator, averaged over 8 samples instead of 1), in the contexts none and dpp; share of questions whose mean is above
delta = 0.25, next to the share whose stored single-sample stopping entropy of the main run (steps[0].H of tsr /
visref_raw_rank) was above 0.25. The empirical entropy of the 8 answer strings is reported as description.
  continue  >= 20% of the closed questions have a mean above 0.25 in the context none

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
RUNS = 'kaggle/runs/b_main_s*/output/results/results_*.jsonl'


def boot_ci(xs, b=B, seed=SEED):
    if not xs:
        return None
    rng, n = random.Random(seed), len(xs)
    m = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return [round(m[int(.025 * b)], 4), round(m[int(.975 * b)], 4)]


def summary(rows):
    out = {'n': len(rows)}
    for c in CONDS:
        d_gt = [r['part1']['cond'][c]['logp_gt'] - r['part1']['cond']['none']['logp_gt'] for r in rows]
        d_ts = [r['part1']['cond'][c]['logp_tsr'] - r['part1']['cond']['none']['logp_tsr'] for r in rows
                if r['part1']['cond'][c]['logp_tsr'] is not None and r['part1'].get('tsr_valid', True)]
        k = [r['part1']['cond'][c]['kl_from_none'] for r in rows]
        top = [r['part1']['cond'][c]['top1'] != r['part1']['cond']['none']['top1'] for r in rows]
        out[c] = {'mean_dlogp_gt': round(statistics.mean(d_gt), 4) if d_gt else None, 'ci_dlogp_gt': boot_ci(d_gt),
                  'mean_dlogp_tsr': round(statistics.mean(d_ts), 4) if d_ts else None,
                  'median_kl': round(statistics.median(k), 5) if k else None, 'mean_kl': round(statistics.mean(k), 5) if k else None,
                  'top1_changed': sum(top), 'top1_changed_share': round(sum(top) / len(top), 4) if top else None}
    pair = lambda a, b, f: [f(r, a) - f(r, b) for r in rows]  # noqa: E731
    kl = lambda r, c: r['part1']['cond'][c]['kl_from_none']  # noqa: E731
    gt = lambda r, c: r['part1']['cond'][c]['logp_gt']  # noqa: E731
    for name, (a, b) in {'all-other': ('all', 'other'), 'dpp-rand': ('dpp', 'rand'), 'dpp_tag-dpp': ('dpp_tag', 'dpp')}.items():
        x, y = pair(a, b, kl), pair(a, b, gt)
        out[name] = {'mean_kl_diff': round(statistics.mean(x), 5) if x else None, 'ci_kl_diff': boot_ci(x),
                     'mean_logp_gt_diff': round(statistics.mean(y), 4) if y else None, 'ci_logp_gt_diff': boot_ci(y)}
    share = out['all']['top1_changed_share'] or 0
    ci = out['all-other']['ci_logp_gt_diff']
    out['moves'] = bool(rows) and share >= 0.05 and ci is not None and (ci[0] > 0 or ci[1] < 0)
    cs = out['dpp-rand']['ci_logp_gt_diff']
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
    for c in ('none', 'dpp'):
        h = [r['part2'][c]['H_mean'] for r in rows if r['part2'][c]['H_mean'] is not None]
        e = [r['part2'][c]['entropy'] for r in rows if r['part2'][c]['entropy'] is not None]
        out[c] = {'n': len(h), 'no_valid_answer': len(rows) - len(h), 'above_delta': sum(x > DELTA for x in h),
                  'share': round(sum(x > DELTA for x in h) / len(h), 4) if h else None,
                  'median_H_mean': round(statistics.median(h), 6) if h else None,
                  'answer_strings_entropy_above_delta': sum(x > DELTA for x in e),
                  'n_valid_mean': round(statistics.mean(r['part2'][c]['n_valid'] for r in rows), 3) if rows else None}
    for m in ('tsr', 'visref_raw_rank'):
        v = [H.get((r['pid'], m)) for r in rows]
        v = [x for x in v if x is not None]
        out['stored_single_' + m] = {'n': len(v), 'above_delta': sum(x > DELTA for x in v)}
    out['continue'] = (out['none']['share'] or 0) >= 0.20
    return out


def analyze(rows, H):
    p1 = [r for r in rows if 'part1' in r]
    p2 = [r for r in rows if 'part2' in r]
    groups = {'all': p1, 'cap': [r for r in p1 if r['stratum'] == 'cap'], 'close': [r for r in p1 if r['stratum'] == 'close'],
              'multi_choice': [r for r in p1 if r['part1'].get('question_type') == 'multi_choice'],
              'free_form': [r for r in p1 if r['part1'].get('question_type') == 'free_form']}
    versions = sorted({r['code_version'] for r in rows})
    return {'label': 'descriptive diagnostic; readings fixed before the results', 'code_versions': versions,
            'n_rows': len(rows), 'part1': {g: summary(v) for g, v in groups.items()}, 'part2': part2(p2, H)}


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
    print(json.dumps({k: res['part2'][k] for k in ('n', 'continue')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
