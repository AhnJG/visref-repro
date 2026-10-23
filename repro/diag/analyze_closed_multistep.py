"""Analysis of `closed_multistep` (diag/closed_multistep.py; plan closed_multistep_plan.md, section 5, written before any result).
Implements the plan's decision rule; the Korean sentences are the plan's, kept as constants.

Data: the result JSONL files of diag/closed_multistep.py (glob patterns allowed, one per GPU) and the ids file
(diag/closed_multistep_ids.json). A pid seen twice is refused (exit 1). Rows of pids in the ids file's 'main' list count, whatever
their set ('trial' pids are ordinary members of main); rows of other pids are left out and counted; error rows and rows that lack
an answer are listed; main pids without a row are reported as missing; the code_version(s) of all rows are reported.

DECISION. Item accuracy = the number of correct answers among the n_t1 (3) T=1 answers (an integer 0..3, so every sum is exact);
a difference of two conditions is the mean of the integer count differences / n_t1, in pp. Paired bootstrap by question: 20,000
resamples, numpy default_rng(0) (a fresh generator per comparison, analyze_step1_think.boot_sums), 95% range = sorted resample sums
at int(.025 B) and int(.975 B). Changed question = the two conditions' counts differ; "up : down" = more : fewer correct. Exact
two-sided sign test on the changed questions (binomial 1/2). Comparisons: V-T, V-S (primary), T-S, V-Vminus (descriptive).
The first that holds (verdict 1/2 need BOTH the lower end of the range > 0 AND the sign test p < 0.05):
  1  V-T and V-S             -> VERDICT_1
  2  V-T                     -> VERDICT_2
  3  otherwise               -> VERDICT_3, plus the scale sentence (V-T upper end < 10.5 pp: SCALE_CONC; < 3.5 pp: SCALE_EVEN)
The sentence always carries the changed-question counts (up : down). An invalid answer counts as wrong; a condition whose share of
valid answers (all n_t1 + 1 answers of all questions) is below 95% is named in the sentence. --interim-stop (the plan's interim
rule, 112 questions): requires that V and T differ in at most 1 question and writes verdict 3 with INTERIM_SENTENCE.

DESCRIPTIVE (never used for the decision): the greedy-answer comparisons (same statistics); per-step probe accuracy curve (steps
1-4, T and V); step lengths, end types and cap hits; the share of steps whose token ids equal the previous step's (repetition);
V's picked tokens n_sel vs m; the 94 questions that st, tsr and visref_raw_rank of the main run (llm/final/merged.jsonl `correct`)
all got wrong vs the rest; multiple choice vs free form; the main comparisons without the questions whose stored step 1 did not
re-tokenise to its stored length (recon); a post-hoc simulation of stopping at the first step whose probe entropy H < 0.25.

  cd repro && python3 diag/analyze_closed_multistep.py 'results/closed_multistep_*.jsonl' [--ids diag/closed_multistep_ids.json]
      [--merged llm/final/merged.jsonl] --out diag/closed_multistep_analysis.json [--force] [--interim-stop]
"""
import argparse
import glob
import json
import statistics
import sys
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
sys.path.insert(0, str(HERE))
from analyze_step1_think import B, boot_sums  # noqa: E402  (the same bootstrap)

CONDS = ('S', 'T', 'V', 'Vminus')
NAME = {'S': 'S', 'T': 'T', 'V': 'V', 'Vminus': 'V−'}
STEP_CONDS = ('T', 'V')
N_STEPS = 4
ALPHA, VALID_MIN, DELTA = 0.05, 0.95, 0.25
SCALE_CONC_PP, SCALE_EVEN_PP, N_QUESTIONS = 10.5, 3.5, 334
PAIRS = {'V-T': ('V', 'T'), 'V-S': ('V', 'S'), 'T-S': ('T', 'S'), 'V-Vminus': ('V', 'Vminus')}
METHODS_ALL_WRONG = ('st', 'tsr', 'visref_raw_rank')

# ---- the plan's sentences (verbatim) -----------------------------------------------------------------------------------
VERDICT = {1: '스스로 닫은 첫 step 뒤 4단계까지 이어 가는 운영점에서, VisRef가 TSR과 바로 답보다 모두 정답이 많았다(탐색 결과).',
           2: 'V가 T보다 높았으나 바로 답(S)을 넘는 차이는 확인하지 못했다.',
           3: '이 운영점(4단계까지)에서도 VisRef의 이득을 확인하지 못했다.'}
SCALE_CONC = '논문 1B의 +3.5 pp가 이 334문항에 몰렸다는 가정 아래서도 그 크기의 이득은 보이지 않았다.'
SCALE_EVEN = '논문 1B의 +3.5 pp가 이 334문항에 고르게 퍼졌다는 가정 아래서도 그 크기의 이득은 보이지 않았다.'
INTERIM_SENTENCE = '중간 멈춤 규칙으로 112문항에서 멈춤.'


def rel(path):
    """A path relative to repro/ when it lies inside it, else its file name (no absolute paths in the output)."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPRO))
    except ValueError:
        return p.name


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


def complete(r):
    """True when the ok row r has all four conditions' answers (n_t1 sampled + 1 greedy) and the step / probe records."""
    if r.get('status') != 'ok':
        return False
    n = int(r['cfg']['n_t1'])
    try:
        return (all(len(r['answers'][c]) == n + 1 for c in CONDS)
                and all(len(r['steps'][c]) == N_STEPS and len(r['probes'][c]) == N_STEPS for c in ('T', 'V')))
    except (KeyError, TypeError):
        return False


def select(R, ids):
    """(rows used in main-list order, counts and lists of what was left out)."""
    main = [str(p) for p in ids['main']]
    inset = set(main)
    ok = [R[p] for p in main if p in R and R[p].get('status') == 'ok']
    used = [r for r in ok if complete(r)]
    err = [R[p] for p in main if p in R and R[p].get('status') != 'ok']
    c = {'n_rows': len(R), 'n_main_expected': len(main), 'n_used': len(used),
         'n_trial_rows_in_main': sum(R[p].get('set') == 'trial' for p in main if p in R),
         'rows_outside_main_list': sorted(p for p in R if p not in inset),
         'missing': [p for p in main if p not in R], 'n_error': len(err), 'error_pids': [r['pid'] for r in err],
         'error_messages': [r.get('error') for r in err][:5],
         'rows_lacking_answers': [r['pid'] for r in ok if not complete(r)],
         'code_versions': sorted({r['code_version'] for r in R.values() if 'code_version' in r}),
         'scorer_versions': sorted({r['scorer_version'] for r in used if 'scorer_version' in r})}
    c['n_missing'] = len(c['missing'])
    return used, c


# ---- statistics ------------------------------------------------------------------------------------------------------
def sign_p(up, down):
    """Exact two-sided sign test (binomial 1/2) on the changed questions."""
    n, k = up + down, min(up, down)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def pair_stats(x, y, scale):
    """Paired comparison x - y of two integer vectors (counts of correct answers out of `scale`): mean and 95% bootstrap range in
    pp, changed questions up : down, exact sign test p."""
    d = np.asarray(x, dtype=np.int64) - np.asarray(y, dtype=np.int64)
    n = len(d)
    if not n:
        return {'n': 0, 'mean_pp': None, 'ci95_pp': None, 'up': 0, 'down': 0, 'same': 0, 'p_sign': None}
    s = np.sort(boot_sums(d))
    k = scale * n
    up, down = int((d > 0).sum()), int((d < 0).sum())
    return {'n': n, 'mean_pp': 100 * int(d.sum()) / k, 'ci95_pp': [100 * int(s[int(.025 * B)]) / k, 100 * int(s[int(.975 * B)]) / k],
            'up': up, 'down': down, 'same': n - up - down, 'p_sign': sign_p(up, down)}


def counts(rows, c):
    return np.array([int(r['n_correct'][c]) if 'n_correct' in r else _n_correct(r, c) for r in rows], dtype=np.int64)


def _n_correct(r, c):
    return sum(int(a['correct']) for a in r['answers'][c] if not a.get('greedy'))


def greedy_of(rows, c):
    return np.array([int(next(a['correct'] for a in r['answers'][c] if a.get('greedy'))) for r in rows], dtype=np.int64)


def comparisons(rows, scale, pairs=PAIRS):
    """{pair: stats} of the accuracy (counts of the sampled answers) comparisons over rows."""
    return {name: pair_stats(counts(rows, a), counts(rows, b), scale) for name, (a, b) in pairs.items()}


def greedy_comparisons(rows, pairs=PAIRS):
    return {name: pair_stats(greedy_of(rows, a), greedy_of(rows, b), 1) for name, (a, b) in pairs.items()}


def r(x, k=4):
    return None if x is None else round(x, k)


def pub(st):
    """The stats with rounded numbers for the output file."""
    if not st or st.get('mean_pp') is None:
        return st
    return {**st, 'mean_pp': r(st['mean_pp']), 'ci95_pp': [r(v) for v in st['ci95_pp']], 'p_sign': r(st['p_sign'], 6)}


# ---- the decision ----------------------------------------------------------------------------------------------------
def passes(st):
    """Lower end of the 95% range > 0 AND exact sign test p < 0.05."""
    return bool(st['n'] and st['ci95_pp'][0] > 0 and st['p_sign'] < ALPHA)


def verdict(vt, vs, n_questions=N_QUESTIONS, valid_rates=None, interim=False):
    """The plan's verdict from the V-T and V-S comparison stats (pair_stats). Returns {'id', 'sentence', ...}."""
    pt, ps = passes(vt), passes(vs)
    vid = 1 if (pt and ps) else 2 if pt else 3
    parts = [VERDICT[vid], f"(바뀐 문항 좋아짐 : 나빠짐 — V−T {vt['up']} : {vt['down']}, V−S {vs['up']} : {vs['down']})"]
    flags = []
    if vid == 3:
        hi = vt['ci95_pp'][1] if vt['n'] else None
        if n_questions != N_QUESTIONS:
            flags.append('scale_sentence_omitted: not 334 questions')
        elif hi is not None and hi < SCALE_EVEN_PP:
            parts.append(SCALE_EVEN)
        elif hi is not None and hi < SCALE_CONC_PP:
            parts.append(SCALE_CONC)
    low = sorted(c for c, v in (valid_rates or {}).items() if v < VALID_MIN)
    if low:
        parts.append('유효한 답 비율이 95% 미만인 조건: ' + ', '.join(f"{NAME[c]} {100 * valid_rates[c]:.1f}%" for c in low) + '.')
    if interim:
        parts.append(INTERIM_SENTENCE)
    return {'id': vid, 'sentence': ' '.join(parts), 'V-T_pass': pt, 'V-S_pass': ps, 'flags': flags,
            'conditions_below_95pct_valid': low}


def validity(rows, n_t1):
    """{cond: {valid_rate (all n_t1 + 1 answers), valid_rate_sampled, valid_rate_greedy}} over all questions."""
    out = {}
    for c in CONDS:
        al = [a for r_ in rows for a in r_['answers'][c]]
        sa = [a for a in al if not a.get('greedy')]
        gr = [a for a in al if a.get('greedy')]
        f = lambda xs: (sum(bool(a['valid']) for a in xs) / len(xs)) if xs else None  # noqa: E731
        out[c] = {'valid_rate': f(al), 'valid_rate_sampled': f(sa), 'valid_rate_greedy': f(gr), 'n_answers': len(al)}
    return out


# ---- descriptive -----------------------------------------------------------------------------------------------------
def probe_curve(rows):
    """Per condition and step 1..4: n, accuracy of the single probe answer, valid rate, mean finite H, share with H < DELTA."""
    out = {}
    for c in STEP_CONDS:
        out[c] = []
        for k in range(N_STEPS):
            ps = [r_['probes'][c][k] for r_ in rows]
            fin = [p['H'] for p in ps if p.get('H') is not None]
            out[c].append({'k': k + 1, 'n': len(ps), 'acc': r(sum(int(p['correct']) for p in ps) / len(ps)) if ps else None,
                           'valid_rate': r(sum(bool(p['valid']) for p in ps) / len(ps)) if ps else None,
                           'mean_H_finite': r(statistics.fmean(fin)) if fin else None,
                           'share_H_below_delta': r(sum(p.get('H') is not None and p['H'] < DELTA for p in ps) / len(ps)) if ps else None})
    same1 = sum(r_['probes']['T'][0].get('text') == r_['probes']['V'][0].get('text') for r_ in rows)
    return {'curve': out, 'step1_probe_text_equal_T_V': same1, 'n': len(rows)}


def step_table(rows):
    out = {}
    for c in STEP_CONDS:
        out[c] = []
        for k in range(N_STEPS):
            st = [r_['steps'][c][k] for r_ in rows]
            lens = [s['len'] for s in st]
            ends = {e: sum(s['end'] == e for s in st) for e in ('close', 'eos', 'cap')}
            out[c].append({'k': k + 1, 'n': len(st), 'mean_len': r(statistics.fmean(lens), 2) if lens else None,
                           'median_len': statistics.median(lens) if lens else None, 'end': ends, 'n_cap_hits': ends['cap'],
                           'n_empty': sum(l == 0 for l in lens)})
    return out


def repetition(rows):
    out = {}
    for c in STEP_CONDS:
        per, anyrep = [], 0
        for k in range(1, N_STEPS):
            n = len(rows)
            rep = sum(r_['steps'][c][k]['ids'] == r_['steps'][c][k - 1]['ids'] for r_ in rows)
            per.append({'k': k + 1, 'n': n, 'n_same_as_previous': rep, 'share': r(rep / n) if n else None})
        for r_ in rows:
            anyrep += any(r_['steps'][c][k]['ids'] == r_['steps'][c][k - 1]['ids'] for k in range(1, N_STEPS))
        out[c] = {'per_step': per, 'n_questions_with_a_repeated_step': anyrep}
    return out


def selection_table(rows):
    out = []
    for k in range(N_STEPS):
        st = [r_['steps']['V'][k] for r_ in rows]
        blocks = [s for s in st if s.get('block')]
        out.append({'k': k + 1, 'n': len(st), 'n_blocks': len(blocks), 'n_no_block_empty_step': len(st) - len(blocks),
                    'median_n_sel': statistics.median([s['n_sel'] for s in blocks]) if blocks else None,
                    'median_m': statistics.median([s['m'] for s in blocks]) if blocks else None,
                    'mean_n_sel_over_m': r(statistics.fmean([s['n_sel'] / s['m'] for s in blocks if s['m']])) if any(s['m'] for s in blocks) else None,
                    'n_sel_zero': sum(s['n_sel'] == 0 for s in blocks), 'n_sel_equals_m': sum(s['n_sel'] == s['m'] for s in blocks)})
    return out


def delta_simulation(rows, scale):
    """Post hoc: stop at the first step k whose probe H < DELTA (an invalid probe, H None, never stops), else at step 4. The answer
    read is the single probe answer of the stopping step (for step 4 the first final answer of T / Vminus)."""
    out, stop_acc = {}, {}
    for c in STEP_CONDS:
        stops, corr = [], []
        for r_ in rows:
            hs = [p.get('H') for p in r_['probes'][c]]
            k = next((i for i, h in enumerate(hs[:N_STEPS - 1]) if h is not None and h < DELTA), N_STEPS - 1)
            stops.append(k)
            corr.append(int(r_['probes'][c][k]['correct']))
        stop_acc[c] = np.array(corr, dtype=np.int64)
        n = len(rows)
        out[c] = {'stop_step_counts': {str(k + 1): sum(s == k for s in stops) for k in range(N_STEPS)},
                  'acc_at_stop': r(float(stop_acc[c].mean())) if n else None,
                  'acc_if_always_step1': r(statistics.fmean(int(r_['probes'][c][0]['correct']) for r_ in rows)) if n else None,
                  'acc_if_always_step4': r(statistics.fmean(int(r_['probes'][c][N_STEPS - 1]['correct']) for r_ in rows)) if n else None,
                  'n_continue_after_step1': sum(s > 0 for s in stops)}
    out['V_stop-T_stop'] = pub(pair_stats(stop_acc['V'], stop_acc['T'], 1))
    out['note'] = ('single probe answers (not 3-sample means); the probe of steps 1-3 is read on the context BEFORE the image '
                   'block, the one of step 4 on the final-answer context (T) / V- (V); post hoc, not a decision')
    return out


def all_wrong_pids(merged_path):
    """{pid: True} for the pids whose st, tsr and visref_raw_rank rows are all wrong in the main run (merged.jsonl `correct`)."""
    got = {}
    for line in Path(merged_path).read_text().splitlines():
        if line.strip():
            x = json.loads(line)
            if x['method'] in METHODS_ALL_WRONG:
                got.setdefault(str(x['pid']), {})[x['method']] = int(x['correct'])
    return {p for p, d in got.items() if len(d) == len(METHODS_ALL_WRONG) and not any(d.values())}


def subgroup(rows, scale, pairs=None):
    pairs = pairs or {k: PAIRS[k] for k in ('V-T', 'V-S', 'T-S')}
    return {'n': len(rows), **{name: pub(pair_stats(counts(rows, a), counts(rows, b), scale)) for name, (a, b) in pairs.items()},
            'acc_pp': {NAME[c]: r(100 * float(counts(rows, c).sum()) / (scale * len(rows))) if rows else None for c in CONDS}}


def analyze(R, ids, merged_path=None, interim=False, files=None, ids_file=None):
    """The whole analysis document from {pid: row} and the ids dict."""
    used, cnt = select(R, ids)
    n_t1s = {int(r_['cfg']['n_t1']) for r_ in used}
    if len(n_t1s) > 1:
        raise SystemExit(f'rows with different n_t1: {sorted(n_t1s)}')
    scale = n_t1s.pop() if n_t1s else 3
    out = {'experiment': 'closed_multistep', 'files': [rel(f) for f in (files or [])], 'ids_file': rel(ids_file) if ids_file else None,
           'counts': cnt, 'n_analysed': len(used), 'n_t1': scale, 'bootstrap': {'B': B, 'seed': 0}, 'flags': []}
    if cnt['n_missing'] or cnt['n_error'] or cnt['rows_lacking_answers']:
        out['flags'].append('incomplete: not every main question has a complete ok row')
    if len(set(cnt['code_versions'])) > 1:
        out['flags'].append('several code_versions among the rows')
    if not used:
        out['flags'].append('no usable rows')
        return out
    comp = comparisons(used, scale)
    valid = validity(used, scale)
    rates = {c: v['valid_rate'] for c, v in valid.items()}
    if interim and comp['V-T']['up'] + comp['V-T']['down'] > 1:
        raise SystemExit('--interim-stop: V and T differ in more than 1 question; the interim rule does not stop here')
    v = verdict(comp['V-T'], comp['V-S'], len(used), rates, interim=interim)  # <= 1 changed question can only give verdict 3
    out['verdict'] = v
    out['accuracy_pp'] = {NAME[c]: r(100 * float(counts(used, c).sum()) / (scale * len(used))) for c in CONDS}
    out['primary'] = {k: pub(comp[k]) for k in ('V-T', 'V-S')}
    out['secondary'] = {k: pub(comp[k]) for k in ('T-S', 'V-Vminus')}
    out['validity'] = {NAME[c]: {k: r(x) if isinstance(x, float) else x for k, x in d.items()} for c, d in valid.items()}
    desc = {'greedy': {k: pub(x) for k, x in greedy_comparisons(used).items()},
            'greedy_accuracy_pp': {NAME[c]: r(100 * float(greedy_of(used, c).mean())) for c in CONDS},
            'probes': probe_curve(used), 'steps': step_table(used), 'repetition': repetition(used),
            'v_selection': selection_table(used), 'delta_0.25_post_hoc': delta_simulation(used, scale)}
    if merged_path and Path(merged_path).exists():
        wrong = all_wrong_pids(merged_path)
        a, b = [r_ for r_ in used if r_['pid'] in wrong], [r_ for r_ in used if r_['pid'] not in wrong]
        desc['all_wrong_in_main_run'] = {'merged_file': rel(merged_path), 'n_all_wrong_in_analysed': len(a),
                                         'all_wrong': subgroup(a, scale) if a else None, 'rest': subgroup(b, scale) if b else None}
    else:
        desc['all_wrong_in_main_run'] = {'error': 'merged file not found'}
    desc['question_type'] = {t: subgroup([r_ for r_ in used if r_.get('question_type') == t], scale)
                             for t in sorted({r_.get('question_type') for r_ in used if r_.get('question_type')})}
    ok_recon = [r_ for r_ in used if r_.get('recon', {}).get('tokens_match', True)]
    desc['recon_mismatch_excluded'] = {'n_excluded': len(used) - len(ok_recon), 'n': len(ok_recon),
                                       **{k: pub(x) for k, x in comparisons(ok_recon, scale).items()}} if ok_recon else None
    desc['end_of_step1'] = {e: sum(r_.get('stratum') == e for r_ in used) for e in ('close', 'cap', 'eos')}
    out['descriptive'] = desc
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='+', help='result JSONL files or glob patterns')
    ap.add_argument('--ids', default=str(REPRO / 'diag/closed_multistep_ids.json'))
    ap.add_argument('--merged', default=str(REPRO / 'llm/final/merged.jsonl'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--interim-stop', action='store_true', help='the interim rule stopped the run after part 1')
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and not a.force:
        raise SystemExit(f'{out} exists (use --force)')
    R, files = load(a.results)
    ids = json.loads(Path(a.ids).read_text())
    if a.interim_stop:
        ids = {**ids, 'main': ids['part1']}
    doc = analyze(R, ids, a.merged, a.interim_stop, files, a.ids)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + '\n')
    if 'verdict' in doc:
        print(doc['verdict']['sentence'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
