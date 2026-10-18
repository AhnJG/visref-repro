"""Pre-registered analysis of full_think (diag/full_think.py), fixed before results. Inputs: the result JSONL files of
diag/full_think.py (glob patterns, one per GPU) and the ids file (diag/full_think_ids.json, --set, default cap_all);
optionally --types FILE (JSON {pid: question_type}) for rows that lack 'question_type'. Nothing else.

Question-level accuracy = mean of the n_t1 T=1 answers, kept as integer counts of correct answers (0..n_t1) so that sums are
exact. Paired difference x - y = mean of count differences / n_t1 (in percentage points). Bootstrap = question resampling,
20,000 resamples, numpy default_rng(0) (a fresh generator per comparison), 95% CI = the 2.5% / 97.5% order statistics,
two-sided p = min(1, 2 * min(P(b <= 0), P(b >= 0))) over the same resamples (diag/analyze_step1_think.boot_ci_p, the same
convention).

Main comparison (the only one for the decision): F - C over all non-error questions of the set that have both conditions.
Everything else is DESCRIPTIVE (not for the decision): F - A0, C - A0, L - C and F - L (when L was run), C_cut - C,
F_cut - F, F_cut - C_cut (L_cut - L); per condition the end types of the extra thinking (close / eos / cap), mean and median
length, valid-answer rate (flag below 95%), T=1 and greedy accuracy; per pair the number of questions whose
question-level accuracy is higher / lower ('better' / 'worse', x relative to y) and the greedy answers that changed
(exact McNemar p); the F - C comparison split by origin set (continued / stopped), by question type (multiple choice /
free form), by whether F's extra thinking ended by itself (</think> or EOS) or at the cap; the F - C comparison without the
questions whose stored step 1 re-tokenised to another length (recon.tokens_match false); the error rows.
'X_cut' = the cut-512 answers of condition X; for a question whose extra thinking was at most `cut` tokens long the cut
equals the full result and X_cut takes X's answers.
Completeness: a pid seen twice is refused (exit 1); pids of the set without a row are reported as missing; rows whose pid is
not in the set are left out and listed. There is no prefix rule: all non-error rows of the set count.

  python3 diag/analyze_full_think.py 'results/full_think_*.jsonl' --ids diag/full_think_ids.json [--set cap_all] [--out FILE]
"""
import argparse
import glob
import json
import os
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_step1_think as a1  # noqa: E402  (bootstrap, McNemar: the same convention)

FULL = ('A0', 'C', 'F', 'L')
FREE = ('C', 'F', 'L')
VALID_MIN = 0.95
PAIRS = {'F-A0': ('F', 'A0'), 'C-A0': ('C', 'A0'), 'L-C': ('L', 'C'), 'F-L': ('F', 'L'), 'C_cut-C': ('C_cut', 'C'),
         'F_cut-F': ('F_cut', 'F'), 'L_cut-L': ('L_cut', 'L'), 'F_cut-C_cut': ('F_cut', 'C_cut')}
NAMES = FULL + tuple(c + '_cut' for c in FREE)


def is_ok(r):
    return r.get('status', 'ok') == 'ok'


def load(patterns):
    """{pid: row} (ok and error rows) from all files matching the patterns; a pid seen twice is an error."""
    files = sorted({f for p in patterns for f in glob.glob(p)})
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


def answers(r, name):
    """The answer records behind condition `name` of row r (None when the row has none): X's own answers, or for 'X_cut'
    the cut-512 answers, or X's when the cut equals the full result."""
    if name.endswith('_cut'):
        rec = r.get('cut512', {}).get(name[:-4])
        if rec is None:
            return None
        return answers(r, name[:-4]) if rec['same_as_full'] else rec.get('answers')
    return r.get('answers', {}).get(name)


def n_correct(r, name):
    """Correct T=1 answers (greedy excluded) of row r under `name`; the stored accuracy is checked for full conditions."""
    a = [x for x in answers(r, name) if not x.get('greedy')]
    k = sum(int(x['correct']) for x in a)
    if name in FULL:
        assert abs(k / len(a) - r['acc_t1'][name]) < 1e-12, (r['pid'], name)
    return k


def greedy_of(r, name):
    return next(int(x['correct']) for x in answers(r, name) if x.get('greedy'))


def n_samples(rows, names):
    ns = {sum(1 for x in answers(r, c) if not x.get('greedy')) for r in rows for c in names}
    assert len(ns) <= 1, ns
    return ns.pop() if ns else 0


def has(r, *names):
    return all(answers(r, c) is not None for c in names)


def pair(rows, x, y):
    """x - y over the rows that have both conditions (paired by question)."""
    rows = [r for r in rows if has(r, x, y)]
    if not rows:
        return {'n': 0}
    m = n_samples(rows, (x, y))
    d = [n_correct(r, x) - n_correct(r, y) for r in rows]
    lo, hi, p = a1.boot_ci_p(d, m)
    gx = sum(greedy_of(r, x) > greedy_of(r, y) for r in rows)
    gy = sum(greedy_of(r, x) < greedy_of(r, y) for r in rows)
    n = len(rows)
    return {'n': n, 'acc_x_pp': round(100 * sum(n_correct(r, x) for r in rows) / (m * n), 2),
            'acc_y_pp': round(100 * sum(n_correct(r, y) for r in rows) / (m * n), 2),
            'diff_pp': round(100 * sum(d) / (m * n), 2), 'ci95_pp': [round(lo, 2), round(hi, 2)],
            'p_boot': round(p, 4), 'ci_excludes_zero': bool(lo > 0 or hi < 0),
            'better': sum(v > 0 for v in d), 'worse': sum(v < 0 for v in d), 'same': sum(v == 0 for v in d),
            'greedy_diff_pp': round(100 * (sum(greedy_of(r, x) for r in rows) - sum(greedy_of(r, y) for r in rows)) / n, 2),
            'greedy_better': gx, 'greedy_worse': gy, 'p_mcnemar_greedy': round(a1.mcnemar(gx, gy), 4)}


def condition(rows, name):
    rows = [r for r in rows if has(r, name)]
    out = {'n': len(rows)}
    if not rows:
        return out
    m = n_samples(rows, (name,))
    valid = [st.mean(bool(a['valid']) for a in answers(r, name)) for r in rows]
    out.update(acc_t1_pp=round(100 * sum(n_correct(r, name) for r in rows) / (m * len(rows)), 2),
               greedy_acc_pp=round(100 * sum(greedy_of(r, name) for r in rows) / len(rows), 2),
               valid_answer_rate=round(st.mean(valid), 4), valid_rate_flag=bool(st.mean(valid) < VALID_MIN))
    base = name[:-4] if name.endswith('_cut') else name
    if base in FREE and not name.endswith('_cut'):
        ex = [r['extra'][name] for r in rows if 'end' in r.get('extra', {}).get(name, {})]
        lens = [x['len'] for x in ex]
        z = [x['zlib_ratio_tail3000'] for x in ex if x.get('zlib_ratio_tail3000') is not None]
        out.update(end={e: sum(x['end'] == e for x in ex) for e in ('close', 'eos', 'cap')},
                   mean_len=round(st.mean(lens), 1) if lens else None, median_len=st.median(lens) if lens else None,
                   mean_zlib_ratio_tail3000=round(st.mean(z), 4) if z else None)
    if name.endswith('_cut'):
        recs = [r['cut512'][base] for r in rows]
        out.update(n_same_as_full=sum(x['same_as_full'] for x in recs), n_new_answers=sum(not x['same_as_full'] for x in recs))
    return out


def f_ended_by_itself(r):
    e = r.get('extra', {}).get('F', {}).get('end')
    return None if e is None else ('self_ended' if e in ('close', 'eos') else 'cap')


def qtype(r, types):
    return r.get('question_type') or types.get(str(r['pid'])) or 'unknown'


def split_table(rows, key):
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    return {g: {'n': len(v), 'F-C': pair(v, 'F', 'C'),
                'acc_t1_pp': {c: condition(v, c).get('acc_t1_pp') for c in FULL if any(has(r, c) for r in v)}}
            for g, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def analyse(R, ids, set_name='cap_all', types=None):
    types = types or {}
    ids = [str(p) for p in ids]
    inset = set(ids)
    missing = [p for p in ids if p not in R]
    outside = sorted(p for p in R if p not in inset)
    err = [R[p] for p in ids if p in R and not is_ok(R[p])]
    ok = [R[p] for p in ids if p in R and is_ok(R[p])]
    main = [r for r in ok if has(r, 'C', 'F')]
    out = {'set': set_name, 'n_ids': len(ids), 'n_rows': sum(p in R for p in ids), 'n_ok': len(ok), 'n_error': len(err),
           'missing_n': len(missing), 'missing_first': missing[:10], 'rows_not_in_set': outside,
           'n_main': len(main), 'rows_lacking_C_or_F': [r['pid'] for r in ok if not has(r, 'C', 'F')],
           'code_version_scorer_version': sorted({(r['code_version'], r['scorer_version']) for r in ok}),
           'conds_run': sorted({c for r in ok for c in r.get('conds', [])}),
           'errors': {'n': len(err), 'pids': [r['pid'] for r in err][:50],
                      'by_stage': {s: sum(r.get('stage') == s for r in err) for s in sorted({r.get('stage') for r in err}, key=str)},
                      'first_messages': [r.get('error') for r in err][:5]}}
    if not main:
        return out
    prim = dict(a1.clean(pair(main, 'F', 'C')))
    out['primary_F_minus_C'] = prim
    out['secondary_descriptive_not_for_decision'] = {k: pair(ok, x, y) for k, (x, y) in PAIRS.items() if any(has(r, x, y) for r in ok)}
    out['conditions_not_for_decision'] = {c: condition(ok, c) for c in NAMES if any(has(r, c) for r in ok)}
    out['flags'] = [f'valid_answer_rate<{VALID_MIN} for {c}' for c, v in out['conditions_not_for_decision'].items()
                    if v.get('valid_answer_rate') is not None and v['valid_answer_rate'] < VALID_MIN]
    out['splits_F_minus_C_not_for_decision'] = {
        'origin': split_table(main, lambda r: r.get('origin')),
        'question_type': split_table(main, lambda r: qtype(r, types)),
        'F_extra_thinking_ended': split_table(main, f_ended_by_itself)}
    kept = [r for r in main if r.get('recon', {}).get('tokens_match', True)]
    out['sensitivity_excluding_recon_mismatch'] = {'n_excluded': len(main) - len(kept), 'F-C': pair(kept, 'F', 'C')}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='+', help='result JSONL files or glob patterns')
    ap.add_argument('--ids', required=True)
    ap.add_argument('--set', default='cap_all')
    ap.add_argument('--types', default='', help='JSON {pid: question_type} for rows without question_type')
    ap.add_argument('--out', default='')
    a = ap.parse_args(argv)
    doc = json.loads(Path(a.ids).read_text())
    ids = doc['sets'][a.set] if isinstance(doc, dict) and 'sets' in doc else (doc['ids'] if isinstance(doc, dict) else doc)
    types = json.loads(Path(a.types).read_text()) if a.types else {}
    R, files = load(a.results)
    out = analyse(R, ids, a.set, types)
    out['files'] = files
    text = json.dumps(out, indent=1, ensure_ascii=False)
    if a.out:
        if os.path.exists(a.out):
            sys.exit(f'refusing to overwrite {a.out}')
        Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
