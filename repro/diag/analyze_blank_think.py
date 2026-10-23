"""Analysis of `blank_think` (diag/blank_think.py; plan docs/EXPERIMENT_FLOW.md 2026-10-24 "실행 전 계획: 빈 그림 생성 양성 대조", written
before any result). Implements the plan's decision rule; the Korean sentences are the plan's, kept as constants.

Data: the result JSONL files (glob patterns allowed) and the analysis list (diag/blank_think_analysis_ids.json: 'analysis', the
questions fixed from the trial's timing BEFORE the main run, see make_analysis_ids). A pid seen twice is refused (exit 1). Only
rows of the analysis list count; rows of other pids are left out and counted. If any question of the list has no row or an
error row, there is NO verdict (verdict None, HOLD sentence): failed questions are never dropped to reach verdicts 0-3.

DECISION. Item accuracy = correct answers among the n_t1 (3) T=1 answers (integer 0..3); a difference = mean count difference /
n_t1, in pp. Paired bootstrap by question (20,000 resamples, numpy default_rng(0), analyze_step1_think.boot_sums), 95% range;
changed question = the two counts differ; exact two-sided sign test on the changed questions; "passes" = lower end > 0 AND
sign test p < 0.05 (analyze_closed_multistep.pair_stats / passes). The first that holds:
  -  any analysis question missing or with an error row            -> HOLD (no verdict)
  0  gate: A - N does not pass, or its mean < GATE_PP (5 pp)        -> VERDICT[0] (the test has too little to recover)
  1  O - N and O - X both pass                                       -> VERDICT[1]
  2  O - N passes                                                    -> VERDICT[2]
  3  otherwise                                                       -> VERDICT[3]
The sentence carries the changed-question counts (up : down) of O - N and O - X. An invalid answer counts as wrong; a condition
whose share of valid answers is below 95% is named.

DESCRIPTIVE (never used for the decision): greedy-answer comparisons; A0 - N0, A - A0, N - N0, O - A, X - N; the share of the
blank loss won back, (O - N) / (A - N); multiple choice vs free form; step-1 and extra-thinking end types and lengths.

  cd repro && python3 diag/analyze_blank_think.py 'kaggle/runs/blank_think_main*/output/results/blank_think_*.jsonl' \\
      [--ids diag/blank_think_ids.json] --out diag/blank_think_analysis.json [--force]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
sys.path.insert(0, str(HERE))
from analyze_closed_multistep import expand, pair_stats, passes, pub, r, rel  # noqa: E402

CONDS = ('A0', 'N0', 'A', 'N', 'O', 'X')
FREE = ('A', 'N', 'O', 'X')
GATE_PP, VALID_MIN = 5.0, 0.95
PRIMARY = {'A-N': ('A', 'N'), 'O-N': ('O', 'N'), 'O-X': ('O', 'X')}
DESCRIPTIVE = {'A0-N0': ('A0', 'N0'), 'A-A0': ('A', 'A0'), 'N-N0': ('N', 'N0'), 'O-A': ('O', 'A'), 'X-N': ('X', 'N')}

VERDICT = {0: '빈 그림으로 잃은 정답률이 이 표본에서 뚜렷하지 않아, 이 시험은 정보가 부족하다.',
           1: '이 빈 그림·그림 전체·강제 한 단계 경로에서, 다시 넣은 그림이 생성을 거쳐 정답률을 올렸다(탐색 결과).',
           2: '다시 넣은 그림 뒤 정답률이 올랐지만, 그림 내용 때문인지 길이·형식 때문인지 가르지 못했다.',
           3: '이 경로에서 정답률 수준의 전달은 확인하지 못했다(관문은 통과).'}
HOLD = '고정한 분석 문항 중 결과가 없거나 오류인 문항이 있어 판정을 보류한다.'
MARGIN = 0.7  # make_analysis_ids: use 70% of the time budget (30% margin)


def make_analysis_ids(main, shard_total, n_per_shard):
    """The analysis list: for every shard i (pids at index j with j % shard_total == i of `main`), its first n_per_shard pids,
    returned in main order. n_per_shard = floor(MARGIN * time budget per GPU / measured seconds per question), fixed from the
    trial before the main run."""
    keep = set()
    for i in range(shard_total):
        keep |= set([p for j, p in enumerate(main) if j % shard_total == i][:n_per_shard])
    return [p for p in main if p in keep]


def load(patterns):
    """{pid: row}; a pid seen twice exits 1."""
    R, files = {}, expand(patterns)
    for f in files:
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row['pid'] in R:
                sys.exit(f'pid {row["pid"]} seen twice ({f})')
            R[row['pid']] = row
    return R, files


def select(R, ids):
    main = [str(p) for p in ids['analysis']]
    ms = set(main)
    used = [R[p] for p in main if p in R and R[p].get('status') == 'ok']
    c = {'n_analysis': len(main), 'n_used': len(used), 'n_rows': len(R),
         'n_outside_analysis': sum(p not in ms for p in R),
         'errors': sorted(p for p in main if p in R and R[p].get('status') != 'ok'),
         'n_missing': sum(p not in R for p in main),
         'code_versions': sorted({x['code_version'] for x in R.values() if 'code_version' in x}),
         'scorer_versions': sorted({x['scorer_version'] for x in used if 'scorer_version' in x})}
    return used, c


def counts(rows, c):
    return [int(x['n_correct'][c]) for x in rows]


def greedy(rows, c):
    return [int(x['greedy'][c]) for x in rows]


def stats(rows, pairs, n_t1, use_greedy=False):
    f = greedy if use_greedy else counts
    return {k: pair_stats(f(rows, a), f(rows, b), 1 if use_greedy else n_t1) for k, (a, b) in pairs.items()}


def validity(rows):
    out = {}
    for c in CONDS:
        ans = [a for x in rows for a in x['answers'][c]]
        out[c] = sum(bool(a['valid']) for a in ans) / len(ans) if ans else None
    return out


def verdict(st, valid_rates=None):
    """(number, sentence) by the plan's order."""
    an = st['A-N']
    if not passes(an) or an['mean_pp'] < GATE_PP:
        v = 0
    elif passes(st['O-N']) and passes(st['O-X']):
        v = 1
    elif passes(st['O-N']):
        v = 2
    else:
        v = 3
    s = VERDICT[v]
    if v:
        s += f" (O−N {st['O-N']['up']} : {st['O-N']['down']}, O−X {st['O-X']['up']} : {st['O-X']['down']})"
    low = [c for c, x in (valid_rates or {}).items() if x is not None and x < VALID_MIN]
    if low:
        s += f" 유효한 답이 95% 미만인 조건: {', '.join(low)}."
    return v, s


def accuracy(rows, n_t1):
    return {c: r(100 * sum(counts(rows, c)) / (n_t1 * len(rows))) if rows else None for c in CONDS}


def ends(rows):
    out = {'step1': {}, 'extra': {}}
    for s in ('own', 'blank'):
        e = [x['step1'][s]['end'] for x in rows]
        out['step1'][s] = {k: e.count(k) for k in sorted(set(e))} | {'median_len': statistics.median(x['step1'][s]['len'] for x in rows)}
    for c in FREE:
        e = [x['extra'][c]['end'] for x in rows]
        out['extra'][c] = {k: e.count(k) for k in sorted(set(e))} | {'median_len': statistics.median(x['extra'][c]['len'] for x in rows)}
    return out


def analyze(R, ids, files=None, ids_file=None):
    used, cov = select(R, ids)
    n_t1 = used[0]['cfg']['n_t1'] if used else 3
    st = stats(used, PRIMARY, n_t1)
    val = validity(used) if used else {}
    complete = bool(used) and not cov['errors'] and not cov['n_missing']
    v, sentence = verdict(st, val) if complete else (None, HOLD)
    an, on = st['A-N'], st['O-N']
    won = r(on['mean_pp'] / an['mean_pp']) if used and an['mean_pp'] else None
    out = {'files': [rel(f) for f in (files or [])], 'ids_file': rel(ids_file) if ids_file else None, 'coverage': cov,
           'n_t1': n_t1, 'gate_pp': GATE_PP, 'verdict': v, 'sentence': sentence,
           'primary': {k: pub(x) for k, x in st.items()}, 'valid_rate': {c: r(x) for c, x in val.items()},
           'descriptive': {'accuracy_pct': accuracy(used, n_t1),
                           'pairs': {k: pub(x) for k, x in stats(used, DESCRIPTIVE, n_t1).items()},
                           'greedy': {k: pub(x) for k, x in stats(used, {**PRIMARY, **DESCRIPTIVE}, n_t1, True).items()},
                           'share_won_back': won}}
    if used:
        by = {}
        for qt in sorted({x['question_type'] for x in used}):
            sub = [x for x in used if x['question_type'] == qt]
            by[qt] = {'n': len(sub), 'accuracy_pct': accuracy(sub, n_t1),
                      'pairs': {k: pub(x) for k, x in stats(sub, PRIMARY, n_t1).items()}}
        out['descriptive']['by_question_type'] = by
        out['descriptive']['ends'] = ends(used)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='+')
    ap.add_argument('--ids', default=str(HERE / 'blank_think_analysis_ids.json'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and not a.force:
        sys.exit(f'{out} exists (use --force)')
    R, files = load(a.results)
    res = analyze(R, json.loads(Path(a.ids).read_text()), files, a.ids)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({k: res[k] for k in ('verdict', 'sentence', 'coverage')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
