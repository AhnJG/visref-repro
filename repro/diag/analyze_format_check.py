"""Format check of 2026-10-17 (EF "동작 점검 계획"): does keeping the image block inside the thinking block when answering
cut the extra </think> in answers? Counts only; correctness is never read.

Rows: kaggle/runs/fmtcheck_main/output/results/results_0.jsonl (runpod/run_imgtag.py --plan format_check), methods
visref_hs_imgtag_in and tsr_answer_prompt, on the 200 questions of diag/format_check_ids.json (first 200 of the seed-0
order). Reference: A6 rows (kaggle/runs/imgtag_main) on the same questions.

Counts per method (defined before the run):
  reemit   -- the final answer contains a </think> the model generated (final_y_start > 0)
  empty    -- nothing after '\\nFinal Answer:' (analyze_imgtag_main.empty_after_prompt, the A6 rule); recorded only
  restated -- 'Question:' in the generated answer (final_raw if </think> was re-emitted, else final_answer); recorded only
Rule (fixed before the run): pass when reemit(in) - reemit(TSR with answer prompt) <= LIMIT = 8, half of A6's excess on
the same 200 questions (visref_imgtag 38 - tsr_answer_prompt 21 = 17; 17 / 2 = 8.5, rounded down).

  cd repro && python3 diag/analyze_format_check.py --rows kaggle/runs/fmtcheck_main_kg/output/results/results_*.jsonl \\
      --a6 kaggle/runs/imgtag_main/output/results/results_0.jsonl --out diag/analyze_format_check.json

Additional observations (fixed before the results, never used for the decision; `--observe`, needs the MathVista data
file): correctness is re-scored with the rule scorer (v4) and the empty-after-prompt rule for both runs alike, with no
LLM extraction (rows the rules cannot parse count as wrong and are counted per method). Reported:
  hardware -- this run's TSR with answer prompt against A6's on the same questions (same method, stored step 1 and seed;
              only GPU, software and batching differ): questions with a different prediction, the 2 x 2 correctness
              table and the accuracy difference; this is the size of differences that the machine alone produces
  vs_a6    -- the same for the new format against A6's visref_imgtag
  within   -- new format against TSR in this run, and A6's visref_imgtag against its TSR on the same questions
  reemit   -- correct answers among rows with and without a re-emitted </think>, per method
  by_step1 -- re-emission and accuracy per way the stored step 1 ended (cut, close, eos)
  behaviour-- mean steps, stop reasons, generated tokens, stopping-entropy quartiles, injected / m and informative picks
Accuracy differences on 200 questions are wide (95% range about +-4 pp); they are descriptions, not conclusions.
"""
import argparse
import collections
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_imgtag_main import empty_after_prompt  # noqa: E402

IN, TSR = 'visref_hs_imgtag_in', 'tsr_answer_prompt'
METHODS = (IN, TSR)
A6 = ('visref_imgtag', 'tsr_answer_prompt')
COUNTS = ('reemit', 'empty', 'restated')
A6_EXCESS = 17  # reemit(visref_imgtag) - reemit(tsr_answer_prompt) of A6 on the 200 questions, counted before the run
LIMIT = A6_EXCESS // 2
IDS = Path(__file__).resolve().parent / 'format_check_ids.json'


def flags(r):
    raw = r.get('final_raw') if (r.get('final_y_start') or 0) > 0 else r.get('final_answer')
    return {'reemit': (r.get('final_y_start') or 0) > 0, 'empty': empty_after_prompt(r), 'restated': 'Question:' in (raw or '')}


def counts(rows, pids, methods):
    by = {(str(r['pid']), r['method']): r for r in rows if r['method'] in methods}
    missing = [(p, m) for p in pids for m in methods if (p, m) not in by]
    assert not missing, f'{len(missing)} rows missing, e.g. {missing[:3]}'
    return ({m: {c: sum(flags(by[(p, m)])[c] for p in pids) for c in COUNTS} for m in methods},
            {m: sum(by[(p, m)].get('status') != 'ok' for p in pids) for m in methods})


def analyze(rows, pids, a6_rows=None):
    c, errors = counts(rows, pids, METHODS)
    excess = c[IN]['reemit'] - c[TSR]['reemit']
    out = {'n': len(pids), 'errors': errors, 'counts': c, 'reemit_excess': excess, 'limit': LIMIT,
           'pass': excess <= LIMIT, 'rule': f'reemit(in) - reemit(TSR with answer prompt) <= {LIMIT} (half of A6 excess {A6_EXCESS})'}
    if a6_rows is not None:  # reference: A6 on the same questions; its excess must be the constant fixed before the run
        a, _ = counts(a6_rows, pids, A6)
        out['a6_reference'] = a
        assert a[A6[0]]['reemit'] - a[A6[1]]['reemit'] == A6_EXCESS, 'A6 excess differs from the constant'
    return out


def _pair(a, b, pids):
    """Paired description of two {pid: scored row} maps: prediction agreement, 2 x 2 correctness, accuracy difference
    (a - b, pp) with a normal-approximation 95% range for paired proportions."""
    n = len(pids)
    same = sum(str(a[p].get('prediction')).strip() == str(b[p].get('prediction')).strip() for p in pids)
    t = {'both': 0, 'a_only': 0, 'b_only': 0, 'neither': 0}
    for p in pids:
        ca, cb = int(a[p]['correct']), int(b[p]['correct'])
        t['both' if ca and cb else 'a_only' if ca else 'b_only' if cb else 'neither'] += 1
    d = (t['a_only'] - t['b_only']) / n
    var = ((t['a_only'] + t['b_only']) / n - d * d) / n
    half = 1.96 * max(var, 0) ** 0.5
    return {'n': n, 'same_prediction': same, **t, 'acc_a': round(100 * (t['both'] + t['a_only']) / n, 1),
            'acc_b': round(100 * (t['both'] + t['b_only']) / n, 1), 'diff_pp': round(100 * d, 1),
            'range95_pp': [round(100 * (d - half), 1), round(100 * (d + half), 1)]}


def _quart(xs):
    xs = sorted(x for x in xs if x is not None and x == x and abs(x) != float('inf'))
    if not xs:
        return []
    return [round(xs[int(q * (len(xs) - 1))], 3) for q in (0.25, 0.5, 0.75)]


def observe(rows, a6_rows, pids, score):
    """Additional observations (never used for the decision). score(row) -> row with 'correct', 'prediction',
    'extract_failed' (rule scorer + empty-after-prompt rule)."""
    def scored(rs, ms):
        by = {(str(r['pid']), r['method']): r for r in rs if r['method'] in ms}
        return {m: {p: score(by[(p, m)]) for p in pids} for m in ms}
    new, old = scored(rows, METHODS), scored(a6_rows, A6)
    end = {p: (new[TSR][p].get('steps') or [{}])[0].get('think_end') for p in pids}
    groups = {'new_in': ('new', IN), 'tsr': ('new', TSR), 'a6_imgtag': ('a6', A6[0]), 'a6_tsr': ('a6', A6[1])}
    src = {'new': new, 'a6': old}
    out = {'scoring': 'rules v4 + empty-after-prompt rule, no LLM extraction, both runs alike',
           'extract_failed': {k: sum(bool(src[s][m][p].get('extract_failed')) for p in pids) for k, (s, m) in groups.items()},
           'hardware': _pair(new[TSR], old[A6[1]], pids),
           'vs_a6': _pair(new[IN], old[A6[0]], pids),
           'within': {'new_in_minus_tsr': _pair(new[IN], new[TSR], pids),
                      'a6_imgtag_minus_tsr': _pair(old[A6[0]], old[A6[1]], pids)},
           'reemit': {}, 'by_step1': {}, 'behaviour': {}}
    for k, (s, m) in groups.items():
        rs = src[s][m]
        re_ = [p for p in pids if flags(rs[p])['reemit']]
        out['reemit'][k] = {'reemit': len(re_), 'reemit_correct': sum(int(rs[p]['correct']) for p in re_),
                            'other': len(pids) - len(re_),
                            'other_correct': sum(int(rs[p]['correct']) for p in pids if p not in re_)}
        out['by_step1'][k] = {e: {'n': len(ps), 'reemit': sum(flags(rs[p])['reemit'] for p in ps),
                                  'correct': sum(int(rs[p]['correct']) for p in ps)}
                              for e in ('cap', 'close', 'eos') for ps in [[p for p in pids if end[p] == e]]}
        steps = [st for p in pids for st in (rs[p].get('steps') or [])]
        inj = [st['n_inject'] / st['m'] for st in steps if st.get('m')]
        out['behaviour'][k] = {
            'mean_k': round(sum(rs[p].get('k', 0) for p in pids) / len(pids), 2),
            'stop_reason': dict(sorted(collections.Counter(str(rs[p].get('stop_reason')) for p in pids).items())),
            'gen_tokens': sum(rs[p].get('gen_tokens', 0) for p in pids),
            'H_quartiles': _quart([st.get('H') for st in steps]),
            'inject_over_m_quartiles': _quart(inj), 'n_meaningful_quartiles': _quart([st.get('n_meaningful') for st in steps if st.get('m')])}
    return out


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', required=True, nargs='+', help='result files of the run (one per GPU)')
    ap.add_argument('--a6', default=None)
    ap.add_argument('--observe', action='store_true', help='add the additional observations (needs --a6 and --data)')
    ap.add_argument('--data', default='../data/raw/mathvista_testmini.json')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    rows = [r for pat in a.rows for f in sorted(glob.glob(pat)) for r in load(f)]
    pids = json.loads(IDS.read_text())
    a6 = load(a.a6) if a.a6 else None
    res = analyze(rows, pids, a6)
    if a.observe:
        assert a6 is not None, '--observe needs --a6'
        from analyze_imgtag_main import rule_scored
        data = json.loads(Path(a.data).read_text())
        res['observations'] = observe(rows, a6, pids, lambda r: rule_scored(r, data[str(r['pid'])]))
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'n': res['n'], 'reemit_excess': res['reemit_excess'], 'pass': res['pass']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
