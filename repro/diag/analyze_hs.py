"""Pre-registered visref_hs analysis (EF 2026-10-12 "후속 실험 3 `visref_hs` 사전등록"), written before the results.

visref_hs changes one thing against VisRef (visref_raw_rank): Z, the vectors of the thinking step, are the model's
last-layer hidden states instead of the input embeddings. Everything else is as in b_main.

Sample: the complete prefix of the seed-0 shuffled order (2 shards) among finished visref_hs rows; the comparison rows
are the b_main ST / TSR / VisRef rows of the same questions.
Scoring (main): b_main rows as recorded in llm/final/merged.jsonl (rules + LLM fallback; scorer v4 changes none of
them, checked below); visref_hs rows re-scored with the current rule scorer (v4) and then merged with the LLM fallback
(llm_extract.py, same official MathVista prompt and guards). The rules-only result is reported next to it: every row
of every method re-scored with the current rule scorer, no LLM.
Main comparison (one, no multiplicity correction): hs - TSR, paired bootstrap 95% CI (2,000 resamples, seed 0, the
same resamples as summarize.paired) and the exact McNemar test. Verdict from the CI only:
  1 lower bound > 0                         -> extra gain over TSR with last-layer Z (+ "smaller than the paper"
                                               when the upper bound is < +3.5)
  2 CI contains 0 and upper bound >= +3.5   -> undecided (a paper-size effect is not ruled out)
  3 upper bound < +3.5 (and not 1)          -> no paper-size (+3.5 pp) extra gain with last-layer Z
The bounds are compared as integers (sums of paired differences), so a bound of exactly +3.5 is not lost to rounding.
Fewer than 300 questions -> the verdict is computed by the same rule and flagged "검출력 부족".
Secondary (descriptive, no verdict): hs - VisRef (Z differs by design; step 1 is regenerated in the new session), hs - ST; injected/m; n_meaningful; rank correlation
of late picks with relevance; second `</think>` in the final answer; k and stop reasons; agreement of the first-step
end type with b_main.

Usage (from repro/):
  python3 diag/analyze_hs.py rows    --out llm/hs/rows_v4.jsonl          # input for llm_extract.py prepare
  python3 diag/analyze_hs.py analyze [--llm llm/hs/merged.jsonl] [--out diag/analyze_hs.json]
Options: --hs-dir (default kaggle/runs/b_main_hs/output/results), --method (default visref_hs), --shards (2)."""
import argparse
import collections
import glob
import json
import random
import statistics
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import summarize as S  # noqa: E402
from report import inner_think_diag, mcnemar_exact  # noqa: E402
from visref.scoring import SCORER_VERSION, score  # noqa: E402

PAPER_GAIN_OVER_TSR = Fraction('3.5')  # paper Table 6, InternVL3.5-1B: VisRef 52.0 - TSR 48.5 (docs/IMPLEMENTATION_CHOICES.md A-5)
MIN_N = 300
N_BOOT, SEED = 2000, 0
BMAIN = ('st', 'tsr', 'visref_raw_rank')
VERDICT = {1: '마지막 층 Z에서는 VisRef가 TSR보다 추가 이득',
           2: '판단 보류(논문 크기 효과를 배제하지 못함)',
           3: '마지막 층 Z로도 논문 크기(+3.5 pp)의 추가 이득은 재현 안 됨'}


def boot_bounds(d, n_boot=N_BOOT, seed=SEED):
    """2.5% and 97.5% bootstrap resample SUMS of the integer differences d (same resamples as summarize.paired)."""
    rng = random.Random(seed)
    sums = sorted(sum(rng.choice(d) for _ in d) for _ in range(n_boot))
    return sums[int(.025 * n_boot)], sums[int(.975 * n_boot)]


def verdict(lo_sum, hi_sum, n, paper=PAPER_GAIN_OVER_TSR):
    """Pre-registered rule on integer bounds: the CI in pp is 100 * sum / n."""
    hi_ge_paper = 100 * hi_sum >= paper * n
    if lo_sum > 0:
        return {'code': 1, 'text': VERDICT[1] + ('' if hi_ge_paper else ' (논문보다 작음: CI 상한 < +3.5)')}
    return {'code': 2, 'text': VERDICT[2]} if hi_ge_paper else {'code': 3, 'text': VERDICT[3]}


def pair(rows, a, b, pids):
    d = [int(rows[(p, a)]['correct']) - int(rows[(p, b)]['correct']) for p in pids]
    lo, hi = boot_bounds(d)
    ao, bo = sum(x == 1 for x in d), sum(x == -1 for x in d)
    out = {'n': len(d), 'diff_pp': round(100 * sum(d) / len(d), 2),
           'ci95_pp': [round(100 * lo / len(d), 2), round(100 * hi / len(d), 2)], 'ci95_sum': [lo, hi],
           'a_only': ao, 'b_only': bo, 'p_mcnemar': round(mcnemar_exact(ao, bo), 4)}
    ref = S.paired(rows, a, b, pids)  # cross-check: same numbers as the b_main / fill analyses
    assert all(abs(x - y) < 0.0101 for x, y in zip(ref['ci95_pp'] + [ref['diff_pp']], out['ci95_pp'] + [out['diff_pp']]))
    return out


def load_hs(hs_dir, method):
    rows = {}
    for f in sorted(glob.glob(str(Path(hs_dir) / 'results_*.jsonl'))):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if r['method'] == method:
                    rows[(str(r['pid']), method)] = r
    return rows


def rescored(r, q):
    """Copy of row r re-scored with the current rule scorer (no LLM)."""
    n = score(r['final_answer'], q)
    out = dict(r, correct_before_rescore=r['correct'], scorer_rescored=SCORER_VERSION)
    for k in ('correct', 'extracted', 'prediction', 'rule', 'extract_failed'):
        out[k] = n[k]
    return out


def v4_changes(r, q):
    """1 if the current scorer would change the recorded correctness of a b_main main-scoring row (as final_judgment)."""
    ext = r.get('llm_extraction') if r.get('llm_used') else r.get('extracted')
    return int(q['question_type'] == 'multi_choice' and r['correct'] == 1 and (ext is None or not str(ext).strip()))


def quart(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 4:
        return sorted(xs)
    return [round(x, 3) for x in statistics.quantiles(xs, n=4)]


def mechanism(rows, pids, method):
    hr = [rows[(p, method)] for p in pids]
    vr = [rows[(p, 'visref_raw_rank')] for p in pids]
    steps = [s for r in hr for s in r['steps'] if s.get('m')]
    long_ = [s for s in steps if s['think_tokens'] >= s['m']]
    vsteps = [s for r in vr for s in r['steps'] if s.get('m')]
    by = {m: [rows[(p, m)] for p in pids] for m in (method,) + BMAIN}
    return {
        'n_steps_with_injection': len(steps),
        'inject_over_m_q25_50_75': quart([s['n_inject'] / s['m'] for s in steps]),
        'steps_len_ge_m': len(long_), 'steps_len_ge_m_inject_eq_m': sum(s['n_inject'] == s['m'] for s in long_),
        'visref_inject_over_m_q25_50_75': quart([s['n_inject'] / s['m'] for s in vsteps]),
        'n_meaningful_q25_50_75': quart([s.get('n_meaningful') for s in steps]),
        'n_meaningful_over_m_q25_50_75': quart([s['n_meaningful'] / s['m'] for s in steps if s.get('n_meaningful') is not None]),
        'tail_rel_spearman_q25_50_75': quart([s.get('tail_rel_spearman') for s in steps]),
        'n_unique_z_q25_50_75': quart([s.get('n_unique_z') for s in steps]),
        'second_think_end_in_final_answer': {m: inner_think_diag(rs) for m, rs in by.items()},
        'k_counts': {m: dict(sorted(collections.Counter(r['k'] for r in rs).items())) for m, rs in by.items()},
        'mean_k': {m: round(statistics.mean(r['k'] for r in rs), 2) for m, rs in by.items()},
        'stop_reason': {m: dict(collections.Counter(r['stop_reason'] for r in rs)) for m, rs in by.items()},
        'step1_end_agree': {f'{method}-{m}': sum(rows[(p, method)]['steps'][0]['think_end'] == rows[(p, m)]['steps'][0]['think_end']
                                                 for p in pids) for m in BMAIN},
        'answer_forced': {m: sum(bool(r.get('final_af')) for r in rs) for m, rs in by.items()},
        'mean_think_tokens': {m: round(statistics.mean(r['think_tokens'] for r in rs)) for m, rs in by.items()},
    }


def analyze(bmain, hs, order, method, data, llm=None, shards=2, fill_order=None):
    """bmain: (pid, method) -> b_main main-scoring row; hs: (pid, method) -> raw visref_hs row;
    llm: (pid, method) -> merged visref_hs row (rules v4 + LLM) or None while the extraction is pending."""
    assert sorted(order, key=int) == sorted({p for p, _ in bmain}, key=int), 'order must cover the b_main questions'
    rows_rule = {k: rescored(r, data[k[0]]) for k, r in {**bmain, **hs}.items()}
    pids = S.complete_prefix(rows_rule, BMAIN + (method,), order, shards)
    out = {'method': method, 'n_prefix': len(pids), 'n_hs_rows': len(hs),
           'n_hs_error': sum(hs[(p, method)].get('status') == 'error' for p in pids),
           'hs_code_version': sorted({str(r.get('code_version')) for r in hs.values()}),
           'order_same_as_fill': None if fill_order is None else list(order) == list(fill_order),
           'power_flag': '검출력 부족' if len(pids) < MIN_N else None,
           'scorer': SCORER_VERSION,
           'bmain_rows_changed_by_current_scorer': sum(v4_changes(bmain[(p, m)], data[p]) for p in pids for m in BMAIN)}
    readings = {'rules_only': rows_rule}
    if llm is not None:
        missing = [p for p in pids if (p, method) not in llm]
        assert not missing, f'merged LLM file lacks {len(missing)} prefix rows, e.g. {missing[:3]}'
        out['hs_llm_rows_scorer'] = sorted({str(llm[(p, method)].get('scorer_rescored')) for p in pids})
        out['hs_llm_used'] = sum(bool(llm[(p, method)].get('llm_used')) for p in pids)
        readings = {'main': {**bmain, **{(p, method): llm[(p, method)] for p in pids}}, **readings}
    out['main_reading'] = 'main' if llm is not None else 'rules_only (LLM extraction pending: provisional)'
    for name, rows in readings.items():
        res = {'acc': {m: round(100 * sum(int(rows[(p, m)]['correct']) for p in pids) / len(pids), 2)
                       for m in BMAIN + (method,)},
               'primary': pair(rows, method, 'tsr', pids),
               'secondary': {f'{method}-{b}': pair(rows, method, b, pids) for b in ('visref_raw_rank', 'st')}}
        res['verdict'] = verdict(*res['primary']['ci95_sum'], len(pids))
        out[name] = res
    out['verdict'] = out['main' if llm is not None else 'rules_only']['verdict']
    out['mechanism'] = mechanism(readings['rules_only'], pids, method)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['rows', 'analyze'])
    ap.add_argument('--hs-dir', default='kaggle/runs/b_main_hs/output/results')
    ap.add_argument('--method', default='visref_hs')
    ap.add_argument('--llm', help='merged visref_hs rows from llm_extract.py merge (omit while pending)')
    ap.add_argument('--bmain', default='llm/final/merged.jsonl')
    ap.add_argument('--data', default='../data/raw/mathvista_testmini.json')
    ap.add_argument('--fill-order', default='kaggle/runs/b_main_fill/output/results/ids.json')
    ap.add_argument('--shards', type=int, default=2)
    ap.add_argument('--out')
    a = ap.parse_args(argv)
    data = json.load(open(a.data))
    hs = load_hs(a.hs_dir, a.method)
    assert hs, f'no {a.method} rows in {a.hs_dir}'
    if a.cmd == 'rows':
        assert a.out, '--out is required'
        with open(a.out, 'w') as f:
            for k in sorted(hs, key=lambda k: int(k[0])):
                f.write(json.dumps(rescored(hs[k], data[k[0]]), ensure_ascii=False) + '\n')
        print(json.dumps({'rows': len(hs), 'scorer': SCORER_VERSION, 'out': a.out}))
        return
    bmain = {(r['pid'], r['method']): r for r in map(json.loads, open(a.bmain)) if r['method'] in BMAIN}
    llm = None
    if a.llm:
        llm = {(r['pid'], r['method']): r for r in map(json.loads, open(a.llm)) if r['method'] == a.method}
    order = json.load(open(Path(a.hs_dir) / 'ids.json'))
    fill_order = json.load(open(a.fill_order)) if a.fill_order and Path(a.fill_order).exists() else None
    out = analyze(bmain, hs, order, a.method, data, llm, a.shards, fill_order)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
