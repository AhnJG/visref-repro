"""Analysis of the thumbnail gate `thumb_gate` (diag/thumb_gate.py; plan docs/EXPERIMENT_FLOW.md 2026-10-22 "실행 전 계획: 썸네일 관문
`thumb_gate`", written before any result). Implements the plan's decision rule; the Korean sentences are the plan's, kept as constants.

Data: the result JSONL files of diag/thumb_gate.py (glob patterns allowed, one per GPU) and the ids file (diag/thumb_gate_ids.json).
A pid seen twice is refused (exit 1). Only rows with set 'main' (and a pid of the ids file's 'main' list) count; rows of the trial
set and of other pids are left out and counted; error rows are listed; main pids without a row are reported as missing; the
code_version(s) of all rows are reported. Probabilities p_gt are stored with 6 decimals, so every difference is an exact integer
count of millionths. Bootstrap = the one of diag/analyze_recover_check.Boot: question resampling, 20,000 resamples, numpy
default_rng(0), ONE resample matrix shared by all statistics of a comparison (a group of questions), 95% range = the sorted resample
sums at int(0.025 B) and int(0.975 B).

MAIN (decision). Multi-tile questions (multi True), the NEW text of stage 3. Delta_s(c) = p_gt_s(c) - p_gt_s(none) per question,
averaged; s = tplus / tminus. d_minus = Delta_tminus(dpp); I = Delta_tminus(dpp) - Delta_tplus(dpp) per question.
pass = lower end of d_minus > 0 AND lower end of I > 0 (PASS_SENTENCE, else NOT_PASS_SENTENCE). Always also given:
Delta_tminus(dpp) - Delta_tminus(rand) with its range.

DESCRIPTIVE (never used for the decision; for the stages 'new' and 'stored'): Delta_s(c) with range for every setting and condition
(dpp, rand, all, other, thumb), for the multi-tile and the single-tile questions; all - other per setting and its difference
(minus - plus); the splits by question type and by stored step-1 ending; p_gt(none) tminus - tplus; Delta_s(dpp) by p_gt(none) bin
(< 0.5, >= 0.5, of the setting's own 'none'); top-1 changes per setting and condition against 'none' (improved = the most likely
first answer token becomes the first gold content token gt_ids[1], worsened = it stops being it); mean n_sel per setting and the T+
thumbnail share of the picked tokens (sum n_sel_thumb / sum n_sel) next to the uniform share (sum 256 / sum N); end counts and mean
length of the generated first step per setting.

PART 2 (multi-tile, the generation answers acc_t1 = mean of the n_t1 samples; integer counts of correct answers): tminus - tplus
mean and range; sentence by rule (upper end < 0 -> PART2_DOWN with the size, else PART2_NONE). Also the greedy difference
(descriptive) and the single-tile noise floor (accuracy difference of the single-tile questions, whose two inputs are identical,
and how many of them got identical generated text in both settings).

IMPLEMENTATION CHECKS: single-tile questions: max |logp_gt tplus - tminus| over all stored-stage conditions (flag at >= 1e-4) and the
number with same_input True; multi-tile: n_visual tminus == tplus - 256 for all. Optional --ref FILES (the 4.11 results,
inject_sensitivity_*.jsonl): per pid and condition (none, dpp, rand, all, other) |stored tplus logp_gt - 4.11 part1 logp_gt|, max and
median (descriptive).

  cd repro && python3 diag/analyze_thumb_gate.py 'results/thumb_gate_*.jsonl' [--ids diag/thumb_gate_ids.json]
      --out diag/thumb_gate_analysis.json [--force] [--ref 'inject_sensitivity_*.jsonl']
"""
import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
sys.path.insert(0, str(HERE))
from analyze_recover_check import B, MICRO, Boot, micro, rnd  # noqa: E402  (the same bootstrap)

TILE = 256
SETTINGS = ('tplus', 'tminus')
CONDS = ('dpp', 'rand', 'all', 'other', 'thumb')
REF_CONDS = ('none', 'dpp', 'rand', 'all', 'other')
FLAG_LOGP = 1e-4
BINS = (('p_none<0.5', lambda p: p < 0.5), ('p_none>=0.5', lambda p: p >= 0.5))

# ---- the plan's sentences (verbatim) -----------------------------------------------------------------------------------
PASS_SENTENCE = '썸네일을 뺀 입력에서 DPP reinject의 정답 확률 이득이 썸네일이 있을 때보다 컸다(탐색 결과, 정답률이 아님).'
NOT_PASS_SENTENCE = '썸네일을 빼도 reinject의 정답 확률 이득이 커진다는 근거는 보이지 않았다.'
PART2_DOWN = '썸네일을 빼면 이 답 읽기(첫 단계 뒤 바로 받은 짧은 답)의 정답률이 내려갔다(크기 {x} pp).'
PART2_NONE = '이 표본으로는 약 10 pp보다 작은 변화를 가를 수 없다.'


# ---- loading ---------------------------------------------------------------------------------------------------------
def expand(patterns):
    return sorted({g for p in patterns for g in (glob.glob(p) or ([p] if Path(p).exists() else []))})


def load(patterns):
    """({pid: row}, files) of all rows (ok and error) of the files; a pid seen twice is refused (SystemExit)."""
    files = expand(patterns)
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


def select(R, ids):
    """(rows used, counts and lists of what was left out): ok rows of set 'main' whose pid is in the ids file's main list."""
    main = [str(p) for p in ids['main']]
    inset = set(main)
    used = [R[p] for p in main if p in R and R[p].get('status') == 'ok' and R[p].get('set') == 'main']
    err = [R[p] for p in main if p in R and R[p].get('status') != 'ok']
    c = {'n_rows': len(R), 'n_main_expected': len(main), 'n_used': len(used),
         'n_trial_rows_left_out': sum(r.get('set') == 'trial' for r in R.values()),
         'rows_outside_main_list': sorted(p for p, r in R.items() if p not in inset and r.get('set') != 'trial'),
         'n_wrong_set_in_main_list': sum(1 for p in main if p in R and R[p].get('status') == 'ok' and R[p].get('set') != 'main'),
         'missing': [p for p in main if p not in R], 'n_error': len(err), 'error_pids': [r['pid'] for r in err],
         'error_messages': [r.get('error') for r in err][:5],
         'code_versions': sorted({r['code_version'] for r in R.values() if 'code_version' in r}),
         'scorer_versions': sorted({r['scorer_version'] for r in used if 'scorer_version' in r})}
    c['n_missing'] = len(c['missing'])
    return used, c


# ---- statistics ------------------------------------------------------------------------------------------------------
def stats(bt, d, scale=MICRO):
    """{'n', 'mean', 'ci95'} of the integer vector d (units of 1/scale) over the questions of bt, in natural units."""
    d = np.asarray(d, dtype=np.int64)
    n = len(d)
    if not n:
        return {'n': 0, 'mean': None, 'ci95': None}
    s = np.sort(bt.sums(d))
    k = n * scale
    return {'n': n, 'mean': rnd(int(d.sum()) / k), 'ci95': [rnd(int(s[int(.025 * bt.b)]) / k), rnd(int(s[int(.975 * bt.b)]) / k)],
            '_lo': int(s[int(.025 * bt.b)]) / k, '_hi': int(s[int(.975 * bt.b)]) / k, '_mean': int(d.sum()) / k}


def pub(st):
    """The stats without the private full-precision fields."""
    return {k: v for k, v in st.items() if not k.startswith('_')}


def has(r, stage, s, *conds):
    st = r.get(stage, {}).get(s, {}).get('cond', {})
    return all(c in st for c in conds)


def pvec(rows, stage, s, c):
    return np.array([micro(r[stage][s]['cond'][c]['p_gt']) for r in rows], dtype=np.int64)


def delta(rows, stage, s, c):
    """Delta_s(c) = p_gt(c) - p_gt(none) per question, in millionths."""
    return pvec(rows, stage, s, c) - pvec(rows, stage, s, 'none')


def rows_with(rows, stage, spec):
    """The rows that have, for every (setting, condition) of spec, that condition at that stage."""
    return [r for r in rows if all(has(r, stage, s, *cs) for s, cs in spec)]


# ---- the decision and its core quantities -----------------------------------------------------------------------------
CORE_SPEC = (('tplus', ('none', 'dpp')), ('tminus', ('none', 'dpp', 'rand')))


def core(rows, stage):
    """d_minus, Delta_tplus(dpp), I and Delta_tminus(dpp) - Delta_tminus(rand) over the rows that have the conditions."""
    rows = rows_with(rows, stage, CORE_SPEC)
    if not rows:
        return {'n': 0}, None
    bt = Boot(len(rows))
    dm, dp = delta(rows, stage, 'tminus', 'dpp'), delta(rows, stage, 'tplus', 'dpp')
    out = {'n': len(rows), 'd_minus': stats(bt, dm), 'delta_tplus_dpp': stats(bt, dp), 'I': stats(bt, dm - dp),
           'dpp_minus_rand_tminus': stats(bt, dm - delta(rows, stage, 'tminus', 'rand'))}
    return out, rows


def main_judgment(multi):
    """The plan's decision on the multi-tile rows at the new-text stage."""
    c, rows = core(multi, 'new')
    if rows is None:
        return {'n': 0, 'pass': None, 'sentence': None}
    ok = bool(c['d_minus']['_lo'] > 0 and c['I']['_lo'] > 0)
    out = {k: (pub(v) if isinstance(v, dict) else v) for k, v in c.items()}
    out.update({'pass': ok, 'rule': 'pass = lower(d_minus) > 0 and lower(I) > 0', 'sentence': PASS_SENTENCE if ok else NOT_PASS_SENTENCE,
                'exploratory_label': 'search result (탐색 결과), not an accuracy'})
    return out


# ---- descriptive ----------------------------------------------------------------------------------------------------
def delta_table(rows, stage):
    out = {}
    for s in SETTINGS:
        for c in CONDS:
            rs = rows_with(rows, stage, ((s, ('none', c)),))
            if rs:
                out[f'{s}:{c}'] = pub(stats(Boot(len(rs)), delta(rs, stage, s, c)))
    return out


def all_minus_other(rows, stage):
    rs = rows_with(rows, stage, (('tplus', ('all', 'other')), ('tminus', ('all', 'other'))))
    if not rs:
        return {'n': 0}
    bt = Boot(len(rs))
    ao = {s: pvec(rs, stage, s, 'all') - pvec(rs, stage, s, 'other') for s in SETTINGS}
    return {'n': len(rs), 'tplus': pub(stats(bt, ao['tplus'])), 'tminus': pub(stats(bt, ao['tminus'])),
            'tminus_minus_tplus': pub(stats(bt, ao['tminus'] - ao['tplus']))}


def none_diff(rows, stage):
    rs = rows_with(rows, stage, (('tplus', ('none',)), ('tminus', ('none',))))
    if not rs:
        return {'n': 0}
    return pub(stats(Boot(len(rs)), pvec(rs, stage, 'tminus', 'none') - pvec(rs, stage, 'tplus', 'none')))


def by_none_bins(rows, stage):
    out = {}
    for s in SETTINGS:
        rs = rows_with(rows, stage, ((s, ('none', 'dpp')),))
        for name, f in BINS:
            sub = [r for r in rs if f(r[stage][s]['cond']['none']['p_gt'])]
            out[f'{s}:{name}'] = pub(stats(Boot(len(sub)), delta(sub, stage, s, 'dpp'))) if sub else {'n': 0}
    return out


def top1_changes(rows, stage):
    out = {}
    for s in SETTINGS:
        for c in CONDS:
            rs = rows_with(rows, stage, ((s, ('none', c)),))
            if not rs:
                continue
            imp = wor = chg = 0
            for r in rs:
                g1 = r['gt_ids'][1]
                a, b = r[stage][s]['cond']['none']['top1'], r[stage][s]['cond'][c]['top1']
                chg += a != b
                imp += b == g1 and a != g1
                wor += a == g1 and b != g1
            out[f'{s}:{c}'] = {'n': len(rs), 'changed': chg, 'improved': imp, 'worsened': wor}
    return out


def selection(rows, stage):
    rs = [r for r in rows if all(s in r.get(stage, {}) for s in SETTINGS)]
    if not rs:
        return {'n': 0}
    n_sel = {s: statistics.mean(r[stage][s]['n_sel'] for r in rs) for s in SETTINGS}
    tot = sum(r[stage]['tplus']['n_sel'] for r in rs)
    thumb = sum(r[stage]['tplus'].get('n_sel_thumb', 0) for r in rs)
    return {'n': len(rs), 'mean_n_sel': {s: rnd(v, 2) for s, v in n_sel.items()},
            'tplus_thumbnail_share_of_picked': rnd(thumb / tot, 4) if tot else None,
            'uniform_share_sum256_over_sumN': rnd(len(rs) * TILE / sum(r['n_visual']['tplus'] for r in rs), 4)}


def splits(rows, stage):
    """The core quantities by question type and by stored ending (each group its own bootstrap)."""
    out = {}
    for name, key in (('question_type', lambda r: r.get('question_type')), ('stratum', lambda r: r.get('stratum'))):
        groups = {}
        for r in rows:
            groups.setdefault(str(key(r)), []).append(r)
        out[name] = {g: {k: (pub(v) if isinstance(v, dict) else v) for k, v in core(v, stage)[0].items()}
                     for g, v in sorted(groups.items())}
    return out


def descriptive(multi, single, stage):
    return {'delta_multi': delta_table(multi, stage), 'delta_single': delta_table(single, stage),
            'all_minus_other_multi': all_minus_other(multi, stage), 'all_minus_other_single': all_minus_other(single, stage),
            'core_multi': {k: (pub(v) if isinstance(v, dict) else v) for k, v in core(multi, stage)[0].items()},
            'splits_multi': splits(multi, stage), 'p_none_tminus_minus_tplus_multi': none_diff(multi, stage),
            'dpp_by_p_none_bin_multi': by_none_bins(multi, stage), 'top1_changes_multi': top1_changes(multi, stage),
            'selection_multi': selection(multi, stage)}


def gen_summary(rows):
    out = {}
    for s in SETTINGS:
        g = [r['gen'][s] for r in rows if s in r.get('gen', {})]
        out[s] = {'n': len(g), 'end': {e: sum(x['end'] == e for x in g) for e in ('close', 'eos', 'cap')},
                  'mean_len': rnd(statistics.mean(x['len'] for x in g), 1) if g else None}
    return out


# ---- part 2 ---------------------------------------------------------------------------------------------------------
def counts(rows, s, key):
    """Integer vector: number of correct sampled answers (key 'acc_t1', scale n_t1) or the greedy answer (scale 1)."""
    if key == 'greedy':
        return np.array([int(r['gen'][s]['greedy']) for r in rows], dtype=np.int64), 1
    ns = {r['cfg']['n_t1'] for r in rows}
    assert len(ns) == 1, ns
    n = ns.pop()
    return np.array([round(r['gen'][s]['acc_t1'] * n) for r in rows], dtype=np.int64), n


def acc_diff(rows, key='acc_t1'):
    rows = [r for r in rows if all(key in r.get('gen', {}).get(s, {}) for s in SETTINGS)]
    if not rows:
        return {'n': 0, 'mean': None, 'ci95': None}, None
    bt = Boot(len(rows))
    (a, n), (b, _) = counts(rows, 'tminus', key), counts(rows, 'tplus', key)
    st = stats(bt, a - b, n)
    st['mean_pp'] = rnd(100 * st['_mean'], 2)
    st['ci95_pp'] = [rnd(100 * st['_lo'], 2), rnd(100 * st['_hi'], 2)]
    st['acc_tminus_pp'] = rnd(100 * int(a.sum()) / (n * len(rows)), 2)
    st['acc_tplus_pp'] = rnd(100 * int(b.sum()) / (n * len(rows)), 2)
    return st, rows


def same_text(r):
    g = r['gen']
    return g['tplus']['text'] == g['tminus']['text'] and g['tplus']['z_ids'] == g['tminus']['z_ids']


def part2(multi, single):
    st, rs = acc_diff(multi)
    out = {'n': st['n'], 'acc_t1_tminus_minus_tplus': pub(st)}
    if rs is None:
        out['sentence'] = None
    else:
        out['sentence'] = PART2_DOWN.format(x=f"{abs(st['_mean']) * 100:.1f}") if st['_hi'] < 0 else PART2_NONE
        out['acc_t1_tminus_minus_tplus'].update(mean_pp=st['mean_pp'], ci95_pp=st['ci95_pp'], acc_tminus_pp=st['acc_tminus_pp'],
                                                acc_tplus_pp=st['acc_tplus_pp'])
    g, _ = acc_diff(multi, 'greedy')
    out['greedy_tminus_minus_tplus_descriptive'] = pub(g)
    sg, srows = acc_diff(single)
    out['single_tile_noise_floor'] = {'acc_t1_tminus_minus_tplus': pub(sg), 'n_single_tile': len(single),
                                      'n_identical_generated_text': sum(same_text(r) for r in single if 'gen' in r)}
    out['multi_identical_generated_text_descriptive'] = sum(same_text(r) for r in multi if 'gen' in r)
    out['generated_first_step_multi'] = gen_summary(multi)
    out['generated_first_step_single'] = gen_summary(single)
    return out


# ---- checks ---------------------------------------------------------------------------------------------------------
def checks(multi, single):
    diffs = []
    for r in single:
        for c in set(r['stored']['tplus']['cond']) & set(r['stored']['tminus']['cond']):
            diffs.append((abs(r['stored']['tplus']['cond'][c]['logp_gt'] - r['stored']['tminus']['cond'][c]['logp_gt']), r['pid'], c))
    mx = max(diffs)[0] if diffs else None
    bad_multi = [r['pid'] for r in multi if r['n_visual']['tminus'] != r['n_visual']['tplus'] - TILE]
    bad_single = [r['pid'] for r in single if r['n_visual']['tminus'] != r['n_visual']['tplus']]
    return {'single_tile': {'n': len(single), 'n_same_input': sum(bool(r['same_input']) for r in single),
                            'max_abs_dlogp_gt_stored': mx, 'flag_ge_1e-4': bool(mx is not None and mx >= FLAG_LOGP),
                            'worst': list(max(diffs)) if diffs else None, 'n_visual_differs': bad_single},
            'multi_tile': {'n': len(multi), 'n_visual_tminus_is_tplus_minus_256_for_all': not bad_multi, 'violations': bad_multi,
                           'n_same_input': sum(bool(r['same_input']) for r in multi)}}


def ref_check(used, ref_rows):
    """Stored T+ logp_gt vs the 4.11 part-1 values per pid and condition (descriptive)."""
    by = {}
    n_pid = n_sel = 0
    for r in used:
        q = ref_rows.get(r['pid'])
        if q is None or 'part1' not in q:
            continue
        n_pid += 1
        n_sel += q.get('sel') == r['stored']['tplus']['sel']
        for c in REF_CONDS:
            a, b = r['stored']['tplus']['cond'].get(c), q['part1']['cond'].get(c)
            if a is not None and b is not None:
                by.setdefault(c, []).append(abs(a['logp_gt'] - b['logp_gt']))
    return {'n_pids_compared': n_pid, 'n_same_sel': n_sel,
            'abs_dlogp_gt': {c: {'n': len(v), 'max': rnd(max(v), 5), 'median': rnd(statistics.median(v), 5)} for c, v in by.items()}}


def load_ref(patterns):
    rows = {}
    for f in expand(patterns):
        for line in Path(f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get('status') == 'ok':
                    rows[str(r['pid'])] = r
    return rows


# ---- run ------------------------------------------------------------------------------------------------------------
def analyze(R, ids, ref=None):
    used, cnt = select(R, ids)
    res = {'label': 'pre-registered thumbnail gate; rule fixed before the results (docs/EXPERIMENT_FLOW.md 2026-10-22 "실행 전 계획: 썸네일 관문 `thumb_gate`")',
           'counts': cnt, 'bootstrap': {'resamples': B, 'seed': 0, 'level': 0.95}}
    multi, single = [r for r in used if r['multi']], [r for r in used if not r['multi']]
    res['n_multi'], res['n_single'] = len(multi), len(single)
    res['main_judgment_new_text_multi_tile'] = main_judgment(multi)
    res['part2_generation_answers'] = part2(multi, single)
    res['descriptive_not_for_decision'] = {'new': descriptive(multi, single, 'new'), 'stored': descriptive(multi, single, 'stored')}
    res['checks'] = checks(multi, single)
    if ref is not None:
        res['ref_4_11'] = ref_check(used, ref)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('rows', nargs='+', help='result JSONL files or glob patterns')
    ap.add_argument('--ids', default=str(REPRO / 'diag/thumb_gate_ids.json'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--force', action='store_true', help='overwrite an existing --out')
    ap.add_argument('--ref', nargs='*', default=None, help='the 4.11 results (inject_sensitivity_*.jsonl) for the consistency check')
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and not a.force:
        sys.exit(f'refusing to overwrite {out} (pass --force)')
    R, files = load(a.rows)
    res = analyze(R, json.loads(Path(a.ids).read_text()), load_ref(a.ref) if a.ref else None)
    res['files'] = files
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
    j, p2, ck = res['main_judgment_new_text_multi_tile'], res['part2_generation_answers'], res['checks']
    cnt = res['counts']
    print(json.dumps({'n_used': cnt['n_used'], 'n_multi': res['n_multi'], 'n_single': res['n_single'], 'missing': cnt['n_missing'],
                      'error': cnt['n_error'], 'code_versions': cnt['code_versions']}, ensure_ascii=False))
    if j.get('pass') is not None:
        print('d_minus', j['d_minus']['mean'], j['d_minus']['ci95'], '| I', j['I']['mean'], j['I']['ci95'], '| dpp-rand(T-)',
              j['dpp_minus_rand_tminus']['mean'], j['dpp_minus_rand_tminus']['ci95'])
        print(j['sentence'])
    if p2.get('sentence'):
        print(p2['sentence'])
    print('single-tile max |dlogp_gt|', ck['single_tile']['max_abs_dlogp_gt_stored'], 'flag', ck['single_tile']['flag_ge_1e-4'],
          '| multi n_visual rule', ck['multi_tile']['n_visual_tminus_is_tplus_minus_256_for_all'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
