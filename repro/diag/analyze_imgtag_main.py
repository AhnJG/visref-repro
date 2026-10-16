"""Analysis of the image-tag injection run (EF 2026-10-15 "실행 전 계획: 이미지 태그 주입 1,000문항 비교"),
written before the results.

Run (runpod/run_imgtag.py, possibly several restarts, possibly several folders): methods 'visref_imgtag' (VisRef whose picked
image tokens are wrapped in <img>...</img> and placed after </think> when the model answers, last-layer hidden states as the
DPP kernel's Z and V), 'tsr_answer_prompt' (TSR whose every final answer starts after the forced prompt '\\nFinal Answer:',
the answer-prompt control) and 'tsr' (the paper's baseline). All three start from the main run's stored step 1
(run.py --step1-from diag/step1_store_bmain.json) with one shared sampler state after it, so the three are paired with
common random numbers from step 2 on; the dual probe is on (logging only).

Sample: the complete prefix of the seed-0 order (--shards 1) in which all three methods are finished, the rows of every
folder put together (a (pid, method) pair in two files is an error; all rows must share one code_version and one stored
step-1 sha). Fewer than 300 questions -> the same rule, flagged "검출력 부족".
Scoring (main): every row re-scored with the current rule scorer (v4), then merged with the LLM fallback (llm_extract.py,
official MathVista prompt and guards; `--llm merged.jsonl`). The rules-only result (no LLM) is reported next to it.
EXTRA RULE (fixed before the results): if `final_forced_answer` is true and the text after the forced prompt
'\\nFinal Answer:' is empty after stripping whitespace, a directly following DEER answer-forcing prompt
('\\n**Final Answer**\\n\\boxed') and leftover box braces (empty_after_prompt), the row is wrong (correct = 0) in both
readings, whatever the rules or the LLM would extract; such rows are left out of the LLM request rows (`rows`), and are
counted and reported per method.
Main comparison (one, no multiplicity correction): visref_imgtag - tsr_answer_prompt (the answer prompt is the same, so the
difference is the injected tagged image tokens), paired bootstrap 95% CI (2,000 resamples, seed 0, the same resamples as
summarize.paired) and the exact McNemar test. Verdict from the CI only, bounds compared as integers (as analyze_think):
  1 lower bound > 0                         -> the image-tag injection gains over the same-prompt TSR (+ "smaller than
                                               the paper" when the upper bound is < +3.5)
  2 CI contains 0 and upper bound >= +3.5   -> undecided (a paper-size effect is not ruled out)
  3 upper bound < +3.5 (and not 1)          -> no paper-size (+3.5 pp) gain from the image-tag injection
The realised discordance, the 80%-power minimum detectable effect and the power at +3.5 pp are reported with it.
Secondary (no verdict of their own, none of the main comparison's wording): visref_imgtag - tsr (the paper's definition of
the comparison; CI, McNemar and only the number 1/2/3 of the same rule, labelled "same rule applied, secondary"),
tsr_answer_prompt - tsr (effect of the answer prompt alone; CI and McNemar only).
Reference only (decides nothing, no verdict): this run's tsr - the main run's TSR on the same questions (main-run rows as
scored in llm/final/merged.jsonl, or the raw kaggle/runs/b_main_s* rows when that file is absent; skipped with a note
when neither exists). Output key 'reference_only_tsr_vs_main_run_tsr'. It cannot separate causes: the main-run TSR rows
span several code versions, started from the generated step 1 with a different sampler state, and differ in other settings,
GPU and torch.
Same thinking in the TSR pair (no correctness involved; output key 'tsr_pair_same_thinking'): tsr and tsr_answer_prompt
differ only in the final answer (the forced prompt), thinking and its random stream are the same. For every question of the
sample with both rows not 'error': the thinking is identical if step count, stop reason, per-step (think_tokens, think_end)
and the recorded thinking text ('thinking', the most specific per-row record: token ids and per-step text are not stored)
are equal. Reported overall and per step-1 end type, with up to 5 differing example pids.
Source clusters (sensitivity check, descriptive, added before the results; output key 'source_cluster'): the main and the
visref_imgtag - tsr differences with whole MathVista source datasets (metadata.source, 31 in testmini) resampled instead of
single questions (2,000 resamples, seed 0), leave-one-source-out lowest and highest, per-source discordant counts.
Strata (descriptive): by how the stored step 1 ended (close / cap / eos): n, accuracy per method, the main difference.
Mechanism (descriptive, per method): k distribution, stop reasons, injected/m quantiles (visref_imgtag), probe-pair
disagreement with / without V (steps with probe_noV_crn and both probes valid), second </think> in the final answer,
empty-after-prompt count, rows whose rule-extracted answer is itself a "Final Answer" phrase, DEER answer forcing,
final_len quantiles, final_context_len maximum, stored step-1 re-tokenisation match, rows with status != 'ok'.

Usage (from repro/):
  python3 diag/analyze_imgtag_main.py rows    --out llm/imgtag_main/rows_v4.jsonl   # input for llm_extract.py prepare
  python3 diag/analyze_imgtag_main.py analyze [--llm llm/imgtag_main/merged.jsonl] [--out diag/analyze_imgtag_main.json]
Options: --run-dirs (default: every kaggle/runs/imgtag_main*/output/results), --shards (1), --data, --bmain, --bmain-dirs."""
import argparse
import collections
import glob
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import summarize as S  # noqa: E402
from analyze_hs import MIN_N, pair, quart, rescored  # noqa: E402
from analyze_imgtag import summarize_group  # noqa: E402  (mechanism counts of the pilot read-out)
from analyze_think import PAPER_GAIN_OVER_TSR, load_runs, power, verdict as _verdict  # noqa: E402
from visref.scoring import SCORER_VERSION  # noqa: E402

METHODS = ('visref_imgtag', 'tsr_answer_prompt', 'tsr')
MAIN = ('visref_imgtag', 'tsr_answer_prompt')
FORCED_PROMPT = '\nFinal Answer:'  # run.py DEFAULTS forced_prompt
AF_PROMPT = '\n**Final Answer**\n\\boxed'  # run.py DEFAULTS af_prompt (DEER answer forcing)
FINAL_ANSWER_PHRASE = re.compile(r'final\s*answer', re.I)
SECONDARY_LABEL = 'same rule applied, secondary'
REF_LABEL = ('reference only: cannot separate causes (main-run TSR rows span several code versions, started from the '
             'generated step 1 with a different sampler state, other settings; different GPU/torch)')
STRATA = ('close', 'cap', 'eos')
SOURCE_LABEL = 'sensitivity check, descriptive: whole source datasets resampled; the verdict uses the question-level CI only'
VERDICT = {1: '이미지 태그 주입이 같은 답 문구의 TSR보다 정확도 향상',
           2: '판단 보류(논문 크기 효과를 배제하지 못함)',
           3: '이미지 태그 주입의 논문 크기(+3.5 pp) 이득은 재현 안 됨'}


def verdict(lo_sum, hi_sum, n):
    """Rule fixed before the results (analyze_think.verdict) with this experiment's texts."""
    return _verdict(lo_sum, hi_sum, n, PAPER_GAIN_OVER_TSR, VERDICT)


def empty_after_prompt(r):
    """Forced final answer with nothing following the forced prompt '\\nFinal Answer:' once whitespace, a directly following
    DEER answer-forcing prompt ('\\n**Final Answer**\\n\\boxed') and leftover box braces are removed."""
    if not r.get('final_forced_answer'):
        return False
    a = r.get('final_answer') or ''
    if a.startswith(FORCED_PROMPT):
        a = a[len(FORCED_PROMPT):]
    a = a.strip()
    af = AF_PROMPT.strip()
    if a.startswith(af):
        a = a[len(af):]
    return not re.sub(r'[\s{}]', '', a)


def check_consistent(rows):
    """One code_version and one stored step-1 sha over every loaded row."""
    cv = sorted({str(r.get('code_version')) for r in rows.values()})
    assert len(cv) == 1, f'rows come from several code versions: {cv}'
    sha = sorted({str(r['step1_info'].get('step1_store_sha')) for r in rows.values() if isinstance(r.get('step1_info'), dict)})
    assert len(sha) <= 1, f'rows come from several stored step-1 files: {sha}'
    return cv, sha


def rule_scored(r, q):
    """Row re-scored with the current rule scorer, then the empty-after-prompt rule (correct = 0)."""
    out = rescored(r, q)
    out['empty_after_prompt'] = empty_after_prompt(r)
    if out['empty_after_prompt']:
        out['correct_before_empty_rule'] = out['correct']
        out['correct'] = 0
    return out


def step1_end(r):
    return (r.get('steps') or [{}])[0].get('think_end')


def acc(rows, pids, m):
    return round(100 * sum(int(rows[(p, m)]['correct']) for p in pids) / len(pids), 2) if pids else None


def strata(rows, pids):
    """By how the stored step 1 ended (same for every method of a question; read from visref_imgtag's row)."""
    out = {}
    for e in STRATA:
        ps = [p for p in pids if step1_end(rows[(p, MAIN[0])]) == e]
        d = [int(rows[(p, MAIN[0])]['correct']) - int(rows[(p, MAIN[1])]['correct']) for p in ps]
        out[e] = {'n': len(ps), 'acc': {m: acc(rows, ps, m) for m in METHODS},
                  'main_diff_pp': round(100 * sum(d) / len(d), 2) if d else None,
                  'a_only': sum(x == 1 for x in d), 'b_only': sum(x == -1 for x in d)}
    out['other'] = sum(step1_end(rows[(p, MAIN[0])]) not in STRATA for p in pids)
    return out


def first_thinking_diff(a, b):
    """None if the thinking of rows a and b is identical, else {'step': first differing step index or None, 'what': ...}.
    Compared in this order: step count, stop reason, per-step (think_tokens, think_end), the recorded thinking text. The text
    is one string per row (no per-step text or token ids are stored), so when only the text differs 'step' is None and
    'char' is the first differing character offset."""
    sa, sb = a.get('steps') or [], b.get('steps') or []
    if len(sa) != len(sb):
        return {'step': min(len(sa), len(sb)), 'what': 'step_count'}
    for i, (x, y) in enumerate(zip(sa, sb)):
        if (x.get('think_tokens'), x.get('think_end')) != (y.get('think_tokens'), y.get('think_end')):
            return {'step': i, 'what': 'step_tokens_or_end'}
    if a.get('stop_reason') != b.get('stop_reason'):
        return {'step': None, 'what': 'stop_reason'}
    ta, tb = a.get('thinking'), b.get('thinking')
    if ta != tb:
        ta, tb = ta or '', tb or ''
        return {'step': None, 'what': 'thinking_text',
                'char': next((i for i, (c, d) in enumerate(zip(ta, tb)) if c != d), min(len(ta), len(tb)))}
    return None


def tsr_pair_same_thinking(rows, pids, n_examples=5):
    """tsr and tsr_answer_prompt share their thinking (they differ only in the final answer). Over the questions of the
    sample with both rows not 'error': {'same', 'total', 'rate'} overall and per step-1 end type, example pids that differ."""
    ps = [p for p in pids if all(rows[(p, m)].get('status') != 'error' for m in ('tsr', 'tsr_answer_prompt'))]
    diffs = {p: first_thinking_diff(rows[(p, 'tsr')], rows[(p, 'tsr_answer_prompt')]) for p in ps}

    def tally(sub):
        same = sum(diffs[p] is None for p in sub)
        return {'same': same, 'total': len(sub), 'rate': round(same / len(sub), 4) if sub else None}
    out = tally(ps)
    out['by_step1_end'] = {e: tally([p for p in ps if step1_end(rows[(p, 'tsr')]) == e]) for e in STRATA}
    out['by_step1_end']['other'] = tally([p for p in ps if step1_end(rows[(p, 'tsr')]) not in STRATA])
    out['n_skipped_error'] = len(pids) - len(ps)
    out['n_pairs_without_thinking_text'] = sum(rows[(p, 'tsr')].get('thinking') is None or rows[(p, 'tsr_answer_prompt')].get('thinking') is None for p in ps)
    out['differing_examples'] = [dict(pid=p, **diffs[p]) for p in ps if diffs[p] is not None][:n_examples]
    return out


def reemit_vs_answer_cap(rs):
    """Descriptive (added 2026-10-16, before the results): does re-emitting </think> in the final answer go with answers
    that hit the answer cap? Rows split by a second </think> in the final answer (final_y_start > 0); per group the row
    count, answers that hit the answer cap (final_finish == 'cap') and DEER answer forcing (final_af). No correctness."""
    out = {}
    for name, grp in (('reemit', [r for r in rs if (r.get('final_y_start') or 0) > 0]),
                      ('no_reemit', [r for r in rs if not (r.get('final_y_start') or 0) > 0])):
        out[name] = {'n': len(grp), 'answer_cap': sum(r.get('final_finish') == 'cap' for r in grp),
                     'answer_forced_deer': sum(bool(r.get('final_af')) for r in grp)}
    return out


DELTA = 0.25  # stopping threshold of the run (run.py DEFAULTS delta); used here only to describe the recorded probes
NEAR = 0.1     # "near the threshold": |H - DELTA| <= NEAR


def cost(rows, pids):
    """Descriptive (added 2026-10-16, before the results), no correctness: generated tokens (gen_tokens, every token the
    method generated for the question), thinking tokens, final context length and per-row wall time (wall_s; rows run in
    shared batches, so wall time is noisy and only a reference) per method, and the ratios of visref_imgtag to the two
    TSR conditions (total generated tokens; median wall time), to set against the paper's Table 5 scale (VisRef / TSR =
    8.2 s / 7.7 s on InternVL3.5-8B, one H100)."""
    out = {}
    for m in METHODS:
        rs = [rows[(p, m)] for p in pids if rows[(p, m)].get('status') != 'error']
        g = [r.get('gen_tokens') or 0 for r in rs]
        ws = [r['wall_s'] for r in rs if r.get('wall_s') is not None]
        out[m] = {'n': len(rs), 'gen_tokens_total': sum(g), 'gen_tokens_q': quart(g),
                  'think_tokens_q': quart([r.get('think_tokens') for r in rs]),
                  'final_context_len_q': quart([r.get('final_context_len') for r in rs]),
                  'wall_s_median': round(statistics.median(ws), 1) if ws else None}
    for b in ('tsr_answer_prompt', 'tsr'):
        a, c = out['visref_imgtag'], out[b]
        out[f'visref_imgtag_over_{b}'] = {
            'gen_tokens_total': round(a['gen_tokens_total'] / c['gen_tokens_total'], 3) if c['gen_tokens_total'] else None,
            'wall_s_median': round(a['wall_s_median'] / c['wall_s_median'], 3) if a['wall_s_median'] and c['wall_s_median'] else None}
    return out


def stop_entropy(rows, pids):
    """Descriptive (added 2026-10-16, before the results), no correctness: the recorded stopping statistic H of every
    valid probe step per method: quantiles, share below DELTA, share near DELTA (|H - DELTA| <= NEAR). For visref_imgtag
    steps whose logging-only probe without the image shares the random state (probe_noV_crn): mean H - H_noV and the
    number of steps whose stop decision (H < DELTA) differs without the image. A free estimator variant: the entropy of
    the first token after the opening brace (probe_ent[1], recorded rounded) instead of the sum, and how often its
    stop decision differs from the recorded one."""
    out = {}
    for m in METHODS:
        st = [s for p in pids for s in (rows[(p, m)].get('steps') or []) if s.get('probe_valid') and s.get('H') is not None]
        hs = [s['H'] for s in st]
        g = {'n_steps': len(st), 'H_q': quart(hs),
             'below_delta': sum(h < DELTA for h in hs), 'near_delta': sum(abs(h - DELTA) <= NEAR for h in hs)}
        fc = [(s['H'] < DELTA, s['probe_ent'][1] < DELTA) for s in st if len(s.get('probe_ent') or []) >= 2]
        g['first_content_token'] = {'n': len(fc), 'decision_differs': sum(a != b for a, b in fc)}
        if m == 'visref_imgtag':
            pr = [(s['H'], s['H_noV']) for s in st if s.get('probe_noV_crn') and s.get('H_noV') is not None]
            # H_noV is inf when the probe without the image did not close its answer (it can never stop there): such steps
            # count for the decision but not for the mean (fixed 2026-10-16 after the results: the mean was -inf)
            fin = [(a, b) for a, b in pr if math.isfinite(a) and math.isfinite(b)]
            g['with_vs_without_image'] = {
                'n': len(pr), 'n_nonfinite': len(pr) - len(fin),
                'mean_H_minus_H_noV': round(statistics.mean(a - b for a, b in fin), 4) if fin else None,
                'decision_differs': sum((a < DELTA) != (b < DELTA) for a, b in pr)}
        out[m] = g
    return out


def source_of(q):
    return (q.get('metadata') or {}).get('source') or 'unknown'


def source_cluster(rs, pids, data, a, b, n_boot=2000, seed=0):
    """Sensitivity check (descriptive, decides nothing): the a - b difference with whole source datasets (MathVista
    metadata.source) resampled instead of single questions. Ratio estimator: sum of differences / number of questions over
    the drawn sources; percentile 95% CI. Also leave-one-source-out (the difference with each source left out: lowest and
    highest, with that source) and per-source counts."""
    by = collections.defaultdict(list)
    for p in pids:
        by[source_of(data[p])].append(int(rs[(p, a)]['correct']) - int(rs[(p, b)]['correct']))
    srcs = sorted(by)
    tot_d, tot_n = sum(sum(v) for v in by.values()), len(pids)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        pick = [rng.choice(srcs) for _ in srcs]
        boots.append(100 * sum(sum(by[s]) for s in pick) / sum(len(by[s]) for s in pick))
    boots.sort()
    loo = sorted((round(100 * (tot_d - sum(by[s])) / (tot_n - len(by[s])), 2), s) for s in srcs if tot_n > len(by[s]))
    return {'n_sources': len(srcs), 'diff_pp': round(100 * tot_d / tot_n, 2),
            'ci95_pp': [round(boots[int(.025 * n_boot)], 2), round(boots[int(.975 * n_boot)], 2)],
            'leave_one_source_out': {'lowest': loo[0], 'highest': loo[-1]} if loo else None,
            'per_source': {s: {'n': len(by[s]), 'a_only': sum(x == 1 for x in by[s]), 'b_only': sum(x == -1 for x in by[s])}
                           for s in srcs}}


def mechanism(rows, pids):
    out = {}
    for m in METHODS:
        rs = [rows[(p, m)] for p in pids]
        g = summarize_group(rs, m)
        inner = g['final_y_start_gt0']
        steps = [s for r in rs for s in (r.get('steps') or [])]
        s1 = [r['step1_info'] for r in rs if isinstance(r.get('step1_info'), dict)]
        g.update(second_think_end_rate=round(inner / len(rs), 4) if rs else None,
                 empty_after_prompt=sum(bool(r.get('empty_after_prompt')) for r in rs),
                 final_af=sum(bool(r.get('final_af')) for r in rs),
                 extracted_final_answer_phrase=sum(bool(FINAL_ANSWER_PHRASE.search(str(r.get('extracted') or ''))) for r in rs),
                 final_context_len_max=max((r.get('final_context_len') or 0 for r in rs), default=None),
                 step1_tokens_match=sum(bool(i.get('step1_tokens_match')) for i in s1), n_step1_info=len(s1),
                 inject_over_m_q25_50_75=(quart([s['n_inject'] / s['m'] for s in steps if s.get('m') and s.get('n_inject') is not None])
                                          if m.startswith('visref') else None),
                 mean_k=round(statistics.mean(r['k'] for r in rs), 2) if rs else None,
                 reemit_vs_answer_cap=reemit_vs_answer_cap(rs))
        out[m] = g
    return out


def reference_tsr_vs_main_run(rs, pids, bmain_tsr, data, reading):
    """This run's tsr - the main run's TSR on the same questions (reference only, decides nothing, no verdict). bmain_tsr: pid -> main-run 'tsr' row
    as scored in llm/final/merged.jsonl; the rules-only reading re-scores it with the current rule scorer (as analyze_think)."""
    if not bmain_tsr:
        return {'skipped': 'main-run TSR rows not found', 'label': REF_LABEL}
    ps = [p for p in pids if p in bmain_tsr]
    if not ps:
        return {'skipped': 'main-run TSR rows cover none of the prefix questions', 'label': REF_LABEL}
    two = {}
    for p in ps:
        two[(p, 'tsr')] = rs[(p, 'tsr')]
        two[(p, 'bmain_tsr')] = rescored(bmain_tsr[p], data[p]) if reading == 'rules_only' else bmain_tsr[p]
    out = pair(two, 'tsr', 'bmain_tsr', ps)
    return dict(out, label=REF_LABEL, n_prefix_covered=len(ps),
                acc={'tsr': acc(two, ps, 'tsr'), 'main_run_tsr': acc(two, ps, 'bmain_tsr')})


def load_bmain_tsr(merged_path, dirs):
    """(pid -> main-run tsr row, source note). merged.jsonl (main scoring) if present, else the raw b_main_s* rows (rules scorer
    applies in the rules-only reading; in the main reading they are used as recorded), else (None, note)."""
    if merged_path and Path(merged_path).exists():
        rows = [json.loads(l) for l in open(merged_path) if l.strip()]
        return {str(r['pid']): r for r in rows if r['method'] == 'tsr'}, f'{merged_path}'
    got = {}
    for d in dirs or []:
        for f in sorted(glob.glob(str(Path(d) / 'results_*.jsonl'))):
            for l in open(f):
                if l.strip():
                    r = json.loads(l)
                    if r['method'] == 'tsr':
                        got[str(r['pid'])] = r
    return (got or None), ('raw b_main_s* rows (no merged.jsonl)' if got else 'main-run TSR rows not found: skipped')


def analyze(rows, order, data, llm=None, shards=1, bmain_tsr=None):
    """rows: (pid, method) -> raw row of the run; llm: (pid, method) -> merged row (rules v4 + LLM) or None while pending;
    bmain_tsr: pid -> main-run tsr row (main scoring) for the reference-only comparison, or None."""
    order = [str(x) for x in order]
    cv, sha = check_consistent(rows)
    rows_rule = {k: rule_scored(r, data[k[0]]) for k, r in rows.items()}
    pids = S.complete_prefix(rows_rule, METHODS, order, shards)
    assert pids, 'no complete question yet'
    out = {'n_prefix': len(pids), 'n_rows': {m: sum(k[1] == m for k in rows) for m in METHODS},
           'n_error': {m: sum(rows[(p, m)].get('status') != 'ok' for p in pids) for m in METHODS},
           'code_version': cv, 'step1_store_sha': sha, 'shards': shards,
           'power_flag': '검출력 부족' if len(pids) < MIN_N else None, 'scorer': SCORER_VERSION,
           'empty_after_prompt': {m: sum(rows_rule[(p, m)]['empty_after_prompt'] for p in pids) for m in METHODS},
           'empty_after_prompt_all_rows': {m: sum(r['empty_after_prompt'] for k, r in rows_rule.items() if k[1] == m)
                                           for m in METHODS}}
    readings = {'rules_only': rows_rule}
    if llm is not None:
        missing = [(p, m) for p in pids for m in METHODS if not rows_rule[(p, m)]['empty_after_prompt'] and (p, m) not in llm]
        assert not missing, f'merged LLM file lacks {len(missing)} prefix rows, e.g. {missing[:3]}'
        merged = {}
        for p in pids:
            for m in METHODS:
                r = rows_rule[(p, m)]
                merged[(p, m)] = r if r['empty_after_prompt'] else dict(llm[(p, m)])
        out['llm_used'] = {m: sum(bool(merged[(p, m)].get('llm_used')) for p in pids) for m in METHODS}
        readings = {'main': merged, 'rules_only': rows_rule}
    out['main_reading'] = 'main' if llm is not None else 'rules_only (LLM extraction pending: provisional)'
    n = len(pids)
    for name, rs in readings.items():
        prim = pair(rs, MAIN[0], MAIN[1], pids)
        vt = pair(rs, MAIN[0], 'tsr', pids)
        out[name] = {'acc': {m: acc(rs, pids, m) for m in METHODS},
                     'primary': prim, 'verdict': verdict(*prim['ci95_sum'], n),
                     'power': power(prim['a_only'], prim['b_only'], n),
                     # no verdict wording: the verdict belongs to the main comparison only
                     'secondary': {'visref_imgtag-tsr': dict(vt, rule_code=verdict(*vt['ci95_sum'], n)['code'], label=SECONDARY_LABEL),
                                   'tsr_answer_prompt-tsr': pair(rs, MAIN[1], 'tsr', pids)},
                     'reference_only_tsr_vs_main_run_tsr': reference_tsr_vs_main_run(rs, pids, bmain_tsr, data, name),
                     'strata_step1_end': strata(rs, pids),
                     'source_cluster': {'label': SOURCE_LABEL,
                                        'visref_imgtag-tsr_answer_prompt': source_cluster(rs, pids, data, *MAIN),
                                        'visref_imgtag-tsr': source_cluster(rs, pids, data, MAIN[0], 'tsr')}}
    out['verdict'] = out['main' if llm is not None else 'rules_only']['verdict']
    out['mechanism'] = mechanism(rows_rule, pids)
    out['tsr_pair_same_thinking'] = tsr_pair_same_thinking(rows, pids)
    out['cost'] = cost(rows, pids)
    out['stop_entropy'] = stop_entropy(rows, pids)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['rows', 'analyze'])
    ap.add_argument('--run-dirs', nargs='*', default=None)
    ap.add_argument('--llm', help='merged rows from llm_extract.py merge (omit while pending)')
    ap.add_argument('--data', default='../data/raw/mathvista_testmini.json')
    ap.add_argument('--bmain', default='llm/final/merged.jsonl', help='main-run rows (main scoring); only its tsr rows are used')
    ap.add_argument('--bmain-dirs', nargs='*', default=None, help='fallback: raw main-run folders (default kaggle/runs/b_main_s*/output/results)')
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--out')
    a = ap.parse_args(argv)
    dirs = a.run_dirs if a.run_dirs is not None else sorted(glob.glob('kaggle/runs/imgtag_main*/output/results'))
    assert dirs, 'no run folders'
    data = json.load(open(a.data))
    rows, order = load_runs(dirs, METHODS)
    assert rows, f'no rows in {dirs}'
    if a.cmd == 'rows':
        assert a.out, '--out is required'
        n_skip = 0
        with open(a.out, 'w') as f:
            for k in sorted(rows, key=lambda k: (int(k[0]), METHODS.index(k[1]))):
                if empty_after_prompt(rows[k]):
                    n_skip += 1  # wrong by rule: not sent to the LLM
                    continue
                f.write(json.dumps(rescored(rows[k], data[k[0]]), ensure_ascii=False) + '\n')
        print(json.dumps({'rows': len(rows) - n_skip, 'left_out_empty_after_prompt': n_skip, 'scorer': SCORER_VERSION,
                          'out': a.out}))
        return
    llm = None
    if a.llm:
        llm = {(str(r['pid']), r['method']): r for r in map(json.loads, open(a.llm)) if r['method'] in METHODS}
    bdirs = a.bmain_dirs if a.bmain_dirs is not None else sorted(glob.glob('kaggle/runs/b_main_s*/output/results'))
    bmain_tsr, note = load_bmain_tsr(a.bmain, bdirs)
    out = analyze(rows, order, data, llm, a.shards, bmain_tsr)
    out['run_dirs'], out['bmain_tsr_source'] = dirs, note
    t = out['tsr_pair_same_thinking']
    print('tsr_pair_same_thinking: same {same}/{total} (rate {rate}); by step-1 end: {by}'.format(
        by=', '.join(f'{e} {v["same"]}/{v["total"]}' for e, v in t['by_step1_end'].items()), **t))
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
