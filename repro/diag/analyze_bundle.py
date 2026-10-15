"""Mechanism readout for visref_bundle vs visref_raw_rank result rows (2026-10-15). No scoring: it only looks at what the
runs did, per method and per group of rows (all / close / cap / eos by how step 1 ended).

  (i)   probe-pair disagreement: over steps where both probes are valid (probe_valid and probe_noV_finish == 'end'),
        how often the box contents differ with and without V_k. Interpretable only if the no-V probe replayed the
        decision probe's random state (probe_noV_crn, common random numbers); an older run without it is flagged.
  (ii)  '</think>' re-emitted in the final answer (final_y_start > 0, or the literal string in final_raw).
  (iii) final answer length (final_len quantiles), answer cap hits, DEER-forced answers.
  run problems: empty final answer (a forced answer counts as empty when nothing follows its forced prompt, or when
        final_len == 0), status != 'ok', VisRef steps that injected nothing, immediate close (a step that closed after
        <= 2 thinking tokens).
  mechanism: n_inject / m quantiles and the inject_format values seen.

  python3 diag/analyze_bundle.py RESULTS.jsonl... [--out diag/analyze_bundle.json]
"""
import argparse
import glob
import json
import sys
from collections import Counter

GROUPS = ('all', 'close', 'cap', 'eos')
QS = (.1, .25, .5, .75, .9)
FORCED_PROMPT = '\nFinal Answer:'  # run.py DEFAULTS forced_prompt: a forced final_answer starts with it


def quantiles(xs, qs=QS):
    """Linear-interpolation quantiles of xs (None when empty)."""
    xs = sorted(xs)
    if not xs:
        return None
    out = []
    for q in qs:
        p = q * (len(xs) - 1)
        lo = int(p)
        hi = min(lo + 1, len(xs) - 1)
        out.append(round(xs[lo] + (xs[hi] - xs[lo]) * (p - lo), 4))
    return out


def empty_final_answer(r):
    """No answer text: nothing but whitespace after an optional leading forced prompt, or no generated answer token."""
    if r.get('final_len') == 0:
        return True
    a = r.get('final_answer') or ''
    if a.startswith(FORCED_PROMPT):
        a = a[len(FORCED_PROMPT):]
    return not a.strip()


def step1_end(r):
    return (r.get('steps') or [{}])[0].get('think_end')


def group_rows(rows):
    """{'all': rows, 'close': rows whose first step closed, ...}."""
    g = {'all': list(rows)}
    for e in GROUPS[1:]:
        g[e] = [r for r in rows if step1_end(r) == e]
    return g


def summarize_group(rows, method=''):
    """Counts and quantiles over one method's rows (pure function of the row dicts)."""
    steps = [s for r in rows for s in (r.get('steps') or [])]
    visref = method.startswith('visref')
    out = {'n_rows': len(rows), 'n_steps': len(steps),
           'k_dist': dict(sorted(Counter(r.get('k') for r in rows).items(), key=lambda kv: (kv[0] is None, kv[0]))),
           'stop_reason': dict(sorted(Counter(r.get('stop_reason') for r in rows).items(), key=lambda kv: str(kv[0]))),
           'status_not_ok': sum(r.get('status', 'ok') != 'ok' for r in rows)}
    # (i) probe pair
    dual = [s for s in steps if 'probe_noV_finish' in s]
    pairs = [s for s in dual if s.get('probe_valid') is True and s['probe_noV_finish'] == 'end']
    dis = sum((s.get('probe_inner') or '').strip() != (s.get('probe_noV_inner') or '').strip() for s in pairs)
    # a failed no-V probe ('error', nov_failed) records no crn flag by design
    no_crn = sum(not s.get('probe_noV_crn') for s in dual if s['probe_noV_finish'] != 'error')
    out.update(n_dual_steps=len(dual), n_crn_steps=sum(bool(s.get('probe_noV_crn')) for s in dual),
               n_pairs_valid=len(pairs), n_pairs_differ=dis,
               pair_differ_rate=round(dis / len(pairs), 4) if pairs else None, pairs_not_interpretable=no_crn > 0)
    # (ii) </think> in the final answer
    raws = [r['final_raw'] for r in rows if r.get('final_raw') is not None]
    out.update(final_y_start_gt0=sum((r.get('final_y_start') or 0) > 0 for r in rows),
               final_raw_has_think_end=sum('</think>' in x for x in raws), n_final_raw=len(raws))
    # (iii) final answer length
    out.update(final_len_q=quantiles([r['final_len'] for r in rows if r.get('final_len') is not None]),
               final_finish_cap=sum(r.get('final_finish') == 'cap' for r in rows),
               final_forced_answer=sum(bool(r.get('final_forced_answer')) for r in rows))
    # run problems
    out.update(empty_final_answer=sum(empty_final_answer(r) for r in rows),
               steps_n_inject_0=sum(s.get('n_inject', 0) == 0 for s in steps) if visref else None,  # missing = none injected
               immediate_close_steps=sum(s.get('think_end') == 'close' and (s.get('think_tokens') or 0) <= 2 for s in steps))
    # mechanism
    out.update(inject_frac_q=quantiles([s['n_inject'] / s['m'] for s in steps if s.get('m') and s.get('n_inject') is not None]),
               inject_format=dict(sorted(Counter(s['inject_format'] for s in steps if 'inject_format' in s).items())))
    s1 = [r['step1_info'] for r in rows if isinstance(r.get('step1_info'), dict)]
    if s1:
        out['step1_source'] = dict(sorted(Counter(i.get('step1_source') for i in s1).items(), key=lambda kv: str(kv[0])))
        out['step1_tokens_mismatch'] = sum(i.get('step1_tokens_match') is False for i in s1)
    return out


def summarize(rows):
    """{method: {group: stats}} plus the list of warnings."""
    by = {}
    for r in rows:
        by.setdefault(r['method'], []).append(r)
    res, warn = {}, []
    for m in sorted(by):
        res[m] = {g: summarize_group(rs, m) for g, rs in group_rows(by[m]).items()}
        if res[m]['all']['pairs_not_interpretable']:
            warn.append(f'{m}: some probe pairs lack probe_noV_crn (run predates common random numbers): '
                        f'(i) probe-pair disagreement is not interpretable')
    return {'methods': res, 'warnings': warn}


def fmt(v):
    if v is None:
        return '-'
    if isinstance(v, float):
        return f'{v:g}'
    if isinstance(v, (list, tuple)):
        return '/'.join(f'{x:g}' for x in v)
    if isinstance(v, dict):
        return ' '.join(f'{k}:{x}' for k, x in v.items()) or '-'
    return str(v)


def table(res, group):
    """Side-by-side text table, methods as columns."""
    ms = list(res['methods'])
    keys = [k for k in res['methods'][ms[0]][group]] if ms else []
    for m in ms:
        keys += [k for k in res['methods'][m][group] if k not in keys]
    cells = [[k] + [fmt(res['methods'][m][group].get(k)) for m in ms] for k in keys]
    w = [max(len(c[i]) for c in cells + [['metric'] + ms]) for i in range(len(ms) + 1)]
    lines = ['  '.join(c.ljust(w[i]) for i, c in enumerate(['metric'] + ms))]
    lines += ['  '.join(c.ljust(w[i]) for i, c in enumerate(row)) for row in cells]
    return '\n'.join(lines)


def load(paths):
    files = sorted({f for p in paths for f in (glob.glob(p) or [p])})
    rows = []
    for f in files:
        rows += [json.loads(line) for line in open(f) if line.strip()]
    return files, rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+', help='result JSONL paths or globs')
    ap.add_argument('--out', default='diag/analyze_bundle.json')
    a = ap.parse_args(argv)
    files, rows = load(a.files)
    assert rows, f'no rows in {a.files}'
    res = summarize(rows)
    res['files'] = files
    for g in GROUPS:
        print(f'== {g} ==')
        print(table(res, g))
    for w in res['warnings']:
        print('WARNING:', w, file=sys.stderr)
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
