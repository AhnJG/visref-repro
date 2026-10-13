"""Pre-registered analysis of the think experiments (EF 2026-10-13 "사전등록: 생각 이어가기 실험"), written before the
results.

Methods (one new run, possibly several Kaggle sessions): 'tsr_think', 'visref_think', 'visref_hs_think' = TSR, VisRef
and VisRef with last-layer Z, each followed, once the method decides to stop (any reason), by the paper's cue
"Wait, think more" and up to 512 more thinking tokens before the final answer (for VisRef after the injected V_k).
All three start from the main run's stored step 1 (run.py --step1-from diag/step1_store_bmain.json) with one shared
sampler state after it, so within the new run the methods are paired with common random numbers from step 2 on.

Sample: the complete prefix of the seed-0 shuffled order (2 shards) in which all three new methods are finished, the
rows of every session put together (a (pid, method) pair in two sessions is an error). The main-run ST / TSR / VisRef
rows of the same questions are used for the secondary comparisons.
Scoring (main): main-run rows as recorded in llm/final/merged.jsonl; new rows re-scored with the current rule scorer
(v4), then merged with the LLM fallback (llm_extract.py, official MathVista prompt and guards; the extractor used is
recorded in the responses file). The rules-only result (every row re-scored with the rule scorer, no LLM) is reported
next to it.
Main comparison (one, no multiplicity correction): visref_think - tsr_think, paired bootstrap 95% CI (2,000 resamples,
seed 0, the same resamples as summarize.paired) and the exact McNemar test. Verdict from the CI only, bounds compared as
integers:
  1 lower bound > 0                         -> extra gain of VisRef over TSR when both think on (+ "smaller than the
                                               paper" when the upper bound is < +3.5)
  2 CI contains 0 and upper bound >= +3.5   -> undecided (a paper-size effect is not ruled out)
  3 upper bound < +3.5 (and not 1)          -> no paper-size (+3.5 pp) extra gain even when the model thinks on
Fewer than 300 questions -> the same rule, flagged "검출력 부족". The realised discordance, the 80%-power minimum
detectable effect and the power at +3.5 pp are reported with it.
Secondary (no verdict of their own): visref_hs_think - tsr_think (the same rule applied, reported as secondary),
visref_hs_think - visref_think, visref_think - VisRef and tsr_think - TSR (main-run rows: same step 1, but a different
sampler state after it), and the 2 x 2 contrast (visref_think - tsr_think) - (VisRef - TSR) with a bootstrap CI.
Mechanism (descriptive): how the extra thinking ended (</think>, EOS, 512 cap, skipped for lack of context), its length,
the share of extra-thinking texts with image words (the step1_branch regex), stop reasons, k, injected/m, second
</think> in the final answer, answer forcing, and how often re-tokenising the stored step 1 gave its stored length.

Usage (from repro/):
  python3 diag/analyze_think.py rows    --out llm/think/rows_v4.jsonl         # input for llm_extract.py prepare
  python3 diag/analyze_think.py analyze [--llm llm/think/merged.jsonl] [--out diag/analyze_think.json]
Options: --think-dirs (default: every kaggle/runs/b_main_think_s*/output/results), --shards (2)."""
import argparse
import collections
import glob
import json
import math
import statistics
import sys
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import summarize as S  # noqa: E402
from analyze_hs import MIN_N, boot_bounds, pair, quart, rescored, v4_changes  # noqa: E402
from analyze_step1_branch import IMG  # noqa: E402  (image-word regex of the step1_branch analysis)
from report import inner_think_diag  # noqa: E402
from visref.scoring import SCORER_VERSION  # noqa: E402

PAPER_GAIN_OVER_TSR = Fraction('3.5')  # paper Table 6, InternVL3.5-1B: VisRef 52.0 - TSR 48.5 (docs/IMPLEMENTATION_CHOICES.md A-5)
THINK = ('tsr_think', 'visref_think', 'visref_hs_think')
BMAIN = ('st', 'tsr', 'visref_raw_rank')
EXTRA_CAP = 512
VERDICT = {1: '생각을 이어가면 VisRef가 TSR보다 추가 이득',
           2: '판단 보류(논문 크기 효과를 배제하지 못함)',
           3: '생각을 이어가도 논문 크기(+3.5 pp)의 VisRef 추가 이득은 재현 안 됨'}
Z = NormalDist()


def verdict(lo_sum, hi_sum, n, paper=PAPER_GAIN_OVER_TSR, texts=VERDICT):
    """Pre-registered rule on integer bounds (the same rule as analyze_hs.verdict): the CI in pp is 100 * sum / n."""
    hi_ge_paper = 100 * hi_sum >= paper * n
    if lo_sum > 0:
        return {'code': 1, 'text': texts[1] + ('' if hi_ge_paper else ' (논문보다 작음: CI 상한 < +3.5)')}
    return {'code': 2, 'text': texts[2]} if hi_ge_paper else {'code': 3, 'text': texts[3]}


def power(a_only, b_only, n, delta_pp=PAPER_GAIN_OVER_TSR):
    """Normal approximation with the realised discordance q: SE = sqrt(q / n); two-sided 5% test."""
    q = (a_only + b_only) / n
    se = math.sqrt(q / n)
    za, zb = Z.inv_cdf(0.975), Z.inv_cdf(0.8)
    return {'discordant': a_only + b_only, 'discordant_rate': round(q, 4), 'se_pp': round(100 * se, 2),
            'mde80_pp': round(100 * (za + zb) * se, 2),
            'power_at_paper': round(Z.cdf(float(delta_pp) / 100 / se - za), 3) if se else None}


def contrast(rows, pids):
    """(visref_think - tsr_think) - (VisRef - TSR) per question, bootstrap CI on integer sums."""
    c = lambda p, m: int(rows[(p, m)]['correct'])  # noqa: E731
    d = [(c(p, 'visref_think') - c(p, 'tsr_think')) - (c(p, 'visref_raw_rank') - c(p, 'tsr')) for p in pids]
    lo, hi = boot_bounds(d)
    return {'n': len(d), 'diff_pp': round(100 * sum(d) / len(d), 2),
            'ci95_pp': [round(100 * lo / len(d), 2), round(100 * hi / len(d), 2)]}


def load_runs(dirs, methods=THINK):
    """(pid, method) -> row over every session; also the question order (the same ids.json in every session)."""
    rows, src, order = {}, {}, None
    for d in dirs:
        o = json.load(open(Path(d) / 'ids.json'))
        assert order is None or o == order, f'{d}: question order differs from the first session'
        order = o
        for f in sorted(glob.glob(str(Path(d) / 'results_*.jsonl'))):
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r['method'] in methods:
                    k = (str(r['pid']), r['method'])
                    assert k not in rows, f'{k} finished in two sessions: {src[k]}, {f}'
                    rows[k], src[k] = r, f
    return rows, order


def extra_text(r):
    """Text of the extra thinking (after '<EXTRA>' + the cue), or None if it was skipped."""
    t = r.get('thinking') or ''
    if '<EXTRA>' not in t:
        return None
    return t.split('<EXTRA>', 1)[1][len('Wait, think more'):]


def mechanism(rows, pids):
    by = {m: [rows[(p, m)] for p in pids] for m in BMAIN + THINK}
    out = {'extra_think_end': {}, 'extra_think_tokens_q25_50_75': {}, 'extra_cap_below_512': {},
           'image_word_rate_extra': {}, 'step1_tokens_match': {}, 'step1_store_sha': {}}
    for m in THINK:
        rs = by[m]
        out['extra_think_end'][m] = dict(collections.Counter(r.get('extra_think_end') for r in rs))
        out['extra_think_tokens_q25_50_75'][m] = quart([r.get('extra_think_tokens') for r in rs])
        out['extra_cap_below_512'][m] = sum(r.get('extra_think_cap', EXTRA_CAP) < EXTRA_CAP for r in rs)
        texts = [x for x in map(extra_text, rs) if x is not None]
        out['image_word_rate_extra'][m] = round(statistics.mean(bool(IMG.search(x)) for x in texts), 4) if texts else None
        out['step1_tokens_match'][m] = sum(bool(r.get('step1_info', {}).get('step1_tokens_match')) for r in rs)
        out['step1_store_sha'][m] = sorted({str(r.get('step1_info', {}).get('step1_store_sha')) for r in rs})
    inj = lambda m: [s['n_inject'] / s['m'] for r in by[m] for s in r['steps'] if s.get('m')]  # noqa: E731
    out['inject_over_m_q25_50_75'] = {m: quart(inj(m)) for m in ('visref_raw_rank', 'visref_think', 'visref_hs_think')}
    out['n_meaningful_q25_50_75'] = quart([s.get('n_meaningful') for r in by['visref_hs_think'] for s in r['steps'] if s.get('m')])
    out['stop_reason'] = {m: dict(collections.Counter(r['stop_reason'] for r in rs)) for m, rs in by.items()}
    out['mean_k'] = {m: round(statistics.mean(r['k'] for r in rs), 2) for m, rs in by.items()}
    out['mean_think_tokens'] = {m: round(statistics.mean(r['think_tokens'] for r in rs)) for m, rs in by.items()}
    out['second_think_end_in_final_answer'] = {m: inner_think_diag(rs) for m, rs in by.items()}
    out['answer_forced'] = {m: sum(bool(r.get('final_af')) for r in rs) for m, rs in by.items()}
    out['s1_forced_answer'] = {m: sum(bool(r.get('final_forced_answer')) for r in rs) for m, rs in by.items()}
    return out


def analyze(bmain, think, order, data, llm=None, shards=2):
    """bmain: (pid, method) -> main-run main-scoring row; think: (pid, method) -> raw new row;
    llm: (pid, method) -> merged new row (rules v4 + LLM) or None while the extraction is pending."""
    assert sorted(order, key=int) == sorted({p for p, _ in bmain}, key=int), 'order must cover the main-run questions'
    rows_rule = {k: rescored(r, data[k[0]]) for k, r in {**bmain, **think}.items()}
    pids = S.complete_prefix(rows_rule, BMAIN + THINK, order, shards)
    assert pids, 'no complete question yet'
    out = {'n_prefix': len(pids), 'n_rows': {m: sum(k[1] == m for k in think) for m in THINK},
           'n_error': {m: sum(think[(p, m)].get('status') == 'error' for p in pids) for m in THINK},
           'code_version': sorted({str(r.get('code_version')) for r in think.values()}),
           'power_flag': '검출력 부족' if len(pids) < MIN_N else None, 'scorer': SCORER_VERSION,
           'bmain_rows_changed_by_current_scorer': sum(v4_changes(bmain[(p, m)], data[p]) for p in pids for m in BMAIN)}
    readings = {'rules_only': rows_rule}
    if llm is not None:
        missing = [(p, m) for p in pids for m in THINK if (p, m) not in llm]
        assert not missing, f'merged LLM file lacks {len(missing)} prefix rows, e.g. {missing[:3]}'
        out['llm_used'] = {m: sum(bool(llm[(p, m)].get('llm_used')) for p in pids) for m in THINK}
        readings = {'main': {**bmain, **{(p, m): llm[(p, m)] for p in pids for m in THINK}}, **readings}
    out['main_reading'] = 'main' if llm is not None else 'rules_only (LLM extraction pending: provisional)'
    for name, rows in readings.items():
        prim = pair(rows, 'visref_think', 'tsr_think', pids)
        hs = pair(rows, 'visref_hs_think', 'tsr_think', pids)
        res = {'acc': {m: round(100 * sum(int(rows[(p, m)]['correct']) for p in pids) / len(pids), 2)
                       for m in BMAIN + THINK},
               'primary': prim, 'verdict': verdict(*prim['ci95_sum'], len(pids)),
               'power': power(prim['a_only'], prim['b_only'], len(pids)),
               'secondary': {'visref_hs_think-tsr_think': dict(hs, rule=verdict(*hs['ci95_sum'], len(pids))),
                             'visref_hs_think-visref_think': pair(rows, 'visref_hs_think', 'visref_think', pids),
                             'visref_think-visref_raw_rank': pair(rows, 'visref_think', 'visref_raw_rank', pids),
                             'tsr_think-tsr': pair(rows, 'tsr_think', 'tsr', pids),
                             'contrast_(visref_think-tsr_think)-(visref-tsr)': contrast(rows, pids)}}
        out[name] = res
    out['verdict'] = out['main' if llm is not None else 'rules_only']['verdict']
    out['mechanism'] = mechanism(readings['rules_only'], pids)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['rows', 'analyze'])
    ap.add_argument('--think-dirs', nargs='*', default=None)
    ap.add_argument('--llm', help='merged rows from llm_extract.py merge (omit while pending)')
    ap.add_argument('--bmain', default='llm/final/merged.jsonl')
    ap.add_argument('--data', default='../data/raw/mathvista_testmini.json')
    ap.add_argument('--shards', type=int, default=2)
    ap.add_argument('--out')
    a = ap.parse_args(argv)
    dirs = a.think_dirs if a.think_dirs is not None else sorted(glob.glob('kaggle/runs/b_main_think_s*/output/results'))
    assert dirs, 'no session folders'
    data = json.load(open(a.data))
    think, order = load_runs(dirs)
    assert think, f'no rows in {dirs}'
    if a.cmd == 'rows':
        assert a.out, '--out is required'
        with open(a.out, 'w') as f:
            for k in sorted(think, key=lambda k: (int(k[0]), THINK.index(k[1]))):
                f.write(json.dumps(rescored(think[k], data[k[0]]), ensure_ascii=False) + '\n')
        print(json.dumps({'rows': len(think), 'scorer': SCORER_VERSION, 'out': a.out}))
        return
    bmain = {(r['pid'], r['method']): r for r in map(json.loads, open(a.bmain)) if r['method'] in BMAIN}
    llm = None
    if a.llm:
        llm = {(r['pid'], r['method']): r for r in map(json.loads, open(a.llm)) if r['method'] in THINK}
    out = analyze(bmain, think, order, data, llm, a.shards)
    out['sessions'] = dirs
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
