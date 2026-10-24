"""Analysis of `sel_think` (diag/sel_think.py; plan docs/EXPERIMENT_FLOW.md 2026-10-24 "실행 전 계획: 선택 토큰 생성 대조", written
before any result). Implements the plan's decision rule; the Korean sentences are the plan's, kept as constants.

Data: the result JSONL files and the analysis list (diag/sel_think_analysis_ids.json 'analysis', fixed from the trial's timing
before the main run with analyze_blank_think.make_analysis_ids). A pid seen twice is merged only when exactly one of its rows is
ok (analyze_blank_think.load). If any analysis question has no row or an error row: no verdict (HOLD).

Item accuracy = correct answers among the n_t1 (3) T=1 answers; a difference = mean count difference / n_t1, in pp. Paired
bootstrap by question (20,000 resamples, numpy default_rng(0)), 95% range; exact two-sided sign test on the changed questions.
  HOLD   any analysis question missing or with an error row
  GATE   F_b - N_b: lower end > 0, sign p < 0.05 and mean >= 5 pp; otherwise GATE_FAIL and no sentence for P1-P2
  P1 D_b - N_b, P2 D_b - R_b: Holm over the two sign-test p; UP = lower end > 0 and Holm p < 0.05, DOWN = upper end < 0 and
         Holm p < 0.05, else NOT (each comparison gets its own sentence). The own-prompt comparison P3 was dropped before any run.
Described (no sentence): accuracy per condition; F_b - D_b, R_b - N_b, X_b - N_b, F_b - X_b; greedy answers; multiple choice /
free form; selection sizes; extra-thinking ends; agreement of N_b, F_b, X_b with blank_think's N, O, X on the same first steps.

  cd repro && python3 diag/analyze_sel_think.py 'kaggle/runs/sel_think_main/output/results/sel_think_*.jsonl' \\
      [--ids diag/sel_think_analysis_ids.json] [--blank-think 'kaggle/runs/blank_think_main/output/results/*.jsonl'] \\
      --out diag/sel_think_analysis.json [--force]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
sys.path.insert(0, str(HERE))
from analyze_blank_think import load, select  # noqa: E402
from analyze_closed_multistep import pair_stats, pub, r, rel  # noqa: E402

CONDS = ('N_b', 'D_b', 'R_b', 'F_b', 'X_b')
GATE_PP, ALPHA = 5.0, 0.05
GATE = ('F_b', 'N_b')
PRIMARY = {'P1': ('D_b', 'N_b'), 'P2': ('D_b', 'R_b')}
DESCRIPTIVE = {'F_b-D_b': ('F_b', 'D_b'), 'R_b-N_b': ('R_b', 'N_b'), 'X_b-N_b': ('X_b', 'N_b'), 'F_b-X_b': ('F_b', 'X_b')}
BLANK_THINK = {'N_b': 'N', 'F_b': 'O', 'X_b': 'X'}
HOLD = '고정한 분석 문항 중 결과가 없거나 오류인 문항이 있어 판정을 보류한다.'
GATE_FAIL = '빈 그림 문맥에서 그림 전체를 다시 넣은 이득(F_b − N_b)이 미리 정한 관문(뚜렷한 양의 차이, 평균 5 pp 이상)에 못 미쳐, 선택 비교는 정보가 부족하다.'
SENTENCES = {
    'P1': {'UP': '빈 그림 문맥에서, VisRef 선택(DPP)으로 고른 토큰만 다시 넣어도 생성을 거친 정답률이 올랐다(탐색).',
           'DOWN': '빈 그림 문맥에서, DPP로 고른 토큰을 다시 넣으면 정답률이 내려갔다(탐색).',
           'NOT': '빈 그림 문맥에서 DPP로 고른 토큰의 정답률 변화는 확인하지 못했다.'},
    'P2': {'UP': '빈 그림 문맥에서, DPP로 고른 토큰을 넣은 정답률이 같은 수의 무작위 토큰을 넣은 정답률보다 높았다(탐색).',
           'DOWN': '빈 그림 문맥에서, DPP로 고른 토큰을 넣은 정답률이 같은 수의 무작위 토큰을 넣은 정답률보다 낮았다(탐색).',
           'NOT': '빈 그림 문맥에서 DPP 선택과 같은 수 무작위 선택의 정답률 차이는 확인하지 못했다.'}}


def counts(rows, c):
    return [int(x['n_correct'][c]) for x in rows]


def greedy(rows, c):
    return [int(x['greedy'][c]) for x in rows]


def stats(rows, pairs, n_t1, use_greedy=False):
    f = greedy if use_greedy else counts
    return {k: pair_stats(f(rows, a), f(rows, b), 1 if use_greedy else n_t1) for k, (a, b) in pairs.items()}


def holm(ps):
    """Holm step-down adjusted p values, in the input order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def gate_passes(st):
    return bool(st['n'] and st['ci95_pp'][0] > 0 and st['p_sign'] < ALPHA and st['mean_pp'] >= GATE_PP)


def decide(st):
    """{name: 'UP' | 'DOWN' | 'NOT'} and Holm p for P1-P2."""
    names = list(PRIMARY)
    adj = dict(zip(names, holm([st[k]['p_sign'] for k in names])))
    out = {}
    for k in names:
        lo, hi = st[k]['ci95_pp']
        out[k] = 'UP' if lo > 0 and adj[k] < ALPHA else 'DOWN' if hi < 0 and adj[k] < ALPHA else 'NOT'
    return out, adj


def accuracy(rows, n_t1):
    return {c: r(100 * sum(counts(rows, c)) / (n_t1 * len(rows))) if rows else None for c in CONDS}


def sizes(rows):
    blank = [x['n_sel']['blank'] for x in rows]
    return {'median_m': statistics.median(x['m'] for x in rows), 'median_n_sel_blank': statistics.median(blank),
            'n_sel_below_m_blank': sum(x['n_sel']['blank'] < x['m'] for x in rows),
            'median_share_of_V_blank': r(statistics.median(x['n_sel']['blank'] / x['n_visual'] for x in rows))}


def ends(rows):
    out = {}
    for c in CONDS:
        e = [x['extra'][c]['end'] for x in rows]
        out[c] = {k: e.count(k) for k in sorted(set(e))} | {'median_len': statistics.median(x['extra'][c]['len'] for x in rows)}
    return out


def agreement(rows, bt):
    """Per condition: questions with a blank_think row, equal n_correct, and both accuracies (same first steps, new run)."""
    out = {}
    for c, b in BLANK_THINK.items():
        pairs = [(x['n_correct'][c], bt[x['pid']]['n_correct'][b]) for x in rows if x['pid'] in bt]
        if pairs:
            out[f'{c}~{b}'] = {'n': len(pairs), 'equal': sum(u == v for u, v in pairs), 'sum_here': sum(u for u, _ in pairs),
                               'sum_blank_think': sum(v for _, v in pairs)}
    return out


def analyze(R, ids, files=None, ids_file=None, bt=None):
    used, cov = select(R, ids)
    n_t1 = used[0]['cfg']['n_t1'] if used else 3
    out = {'files': [rel(f) for f in (files or [])], 'ids_file': rel(ids_file) if ids_file else None, 'coverage': cov, 'n_t1': n_t1,
           'gate_pp': GATE_PP}
    if not used or cov['errors'] or cov['n_missing']:
        out.update(gate=None, decision=None, sentences=[HOLD])
        return out
    gate = pair_stats(counts(used, GATE[0]), counts(used, GATE[1]), n_t1)
    st = stats(used, PRIMARY, n_t1)
    out['gate'] = {'F_b-N_b': pub(gate), 'passes': gate_passes(gate)}
    out['primary'] = {k: pub(x) for k, x in st.items()}
    if out['gate']['passes']:
        d, adj = decide(st)
        out['decision'] = {k: {'result': d[k], 'p_holm': r(adj[k], 6)} for k in PRIMARY}
        out['sentences'] = [SENTENCES[k][d[k]] + f" ({k} {st[k]['up']} : {st[k]['down']})" for k in PRIMARY]
    else:
        out['decision'] = None
        out['sentences'] = [GATE_FAIL + f" (F_b − N_b {gate['up']} : {gate['down']})"]
    desc = {'accuracy_pct': accuracy(used, n_t1), 'pairs': {k: pub(x) for k, x in stats(used, DESCRIPTIVE, n_t1).items()},
            'greedy': {k: pub(x) for k, x in stats(used, {'F_b-N_b': GATE, **PRIMARY, **DESCRIPTIVE}, n_t1, True).items()},
            'selection': sizes(used), 'extra_ends': ends(used)}
    by = {}
    for qt in sorted({x['question_type'] for x in used}):
        sub = [x for x in used if x['question_type'] == qt]
        by[qt] = {'n': len(sub), 'accuracy_pct': accuracy(sub, n_t1),
                  'pairs': {k: pub(x) for k, x in stats(sub, {'F_b-N_b': GATE, **PRIMARY}, n_t1).items()}}
    desc['by_question_type'] = by
    if bt:
        desc['blank_think_agreement'] = agreement(used, bt)
    out['descriptive'] = desc
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='+')
    ap.add_argument('--ids', default=str(HERE / 'sel_think_analysis_ids.json'))
    ap.add_argument('--blank-think', default=str(REPRO / 'kaggle/runs/blank_think_main/output/results/blank_think_*.jsonl'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and not a.force:
        sys.exit(f'{out} exists (use --force)')
    R, files = load(a.results)
    bt = load([a.blank_think])[0] if a.blank_think else None
    res = analyze(R, json.loads(Path(a.ids).read_text()), files, a.ids, bt)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'sentences': res['sentences'], 'coverage': res['coverage']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
