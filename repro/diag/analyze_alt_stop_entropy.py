"""Alternative stopping-entropy estimators on the SAVED per-token entropies (descriptive; 2026-10-20).

Question (counts only, no accuracy, no decision): the paper does not specify how its stopping entropy is estimated. In
the main run, after a step the model closed itself (think_end 'close' or 'eos'), H is almost always tiny, so TSR and
VisRef nearly never continue. Under plausible alternative estimators, computed from the values already saved in the
rows (no re-generation, no GPU), how many such steps would have had value >= DELTA (= would have continued)?

What the saved fields are (visref/methods.py):
  probe_ent       -- `_probe_entropy` (methods.py:355-363): one value per generated probe token, where the token is at box
                     depth 1 (inside the braces; the closing-brace-like tokens, those whose first non-space char is '}',
                     are merged into one outcome) use the merged entropy `ent_l`; every other token (the opening '{'
                     token at depth 0, tokens in nested braces at depth >= 2) uses the full-vocabulary entropy `ent`.
  H               -- `sum(probe_ent) if valid else inf` (methods.py:362); valid = the probe ended on its own, the box
                     closed, and the inside has an alphanumeric character (methods.py:359). The decision is
                     `H < delta` (methods.py `_decide`, 0.25), so an invalid probe (H = inf) always continues.
  probe_ent_full  -- `[round(e, 4) for e in self.ent]` (methods.py:348): full-vocabulary entropy of ALL generated probe
                     tokens, nothing merged.
  Both lists are rounded to 4 decimals when saved; H is not rounded.

Rows: llm/final/merged.jsonl, methods tsr and visref_raw_rank, status ok. Unit = step (a dict in row['steps']).
Groups (reported separately per method):
  (a) first steps (k == 1) with think_end in {close, eos}
  (b) first steps (k == 1) with think_end == cap
  (c) all steps with k >= 2, by think_end (close, eos, cap)
Per-question view for (a): number of questions whose first step closed and whose stop decision flips from stop under the
stored H (E0 < DELTA) to continue under the estimator (value >= DELTA). Also reported: questions that already continue
under E0.

Estimators (fixed list; every one uses saved values only):
  E0 stored H. Sanity: recompute `sum(probe_ent)` for valid probes (inf for invalid) and compare with H; report the
     number of steps that differ by more than the rounding bound 0.00005 * len(probe_ent) (+1e-9), the largest
     difference, and the number of steps where the stop decision (< DELTA) would differ.
  E1 mean of probe_ent.
  E2 entropy of the first content token only. Defined from the code's brace scan: the box is opened by the first token
     that contains '{' (the prompt ends in '\\boxed', depth 0, methods.py `_brace_scan`). If probe_ids[0] == BRACE_ID (90,
     a token that is the bare '{'), the first content token is the next one, probe_ent[1] (depth 1, merged-close
     entropy). If the first token is not the bare '{' (in this run '{-' or '{\\' style tokens fused with content), the
     fused token already makes the first content choice, so E2 = probe_ent[0] (full-vocabulary entropy, depth 0).
     The number of steps using each branch is reported.
  E3 max of probe_ent.
  E4 sum of probe_ent_full (all generated probe tokens, no merging).
  E5 mean of probe_ent_full.
Invalid probes (probe_valid false): H = inf in the code, i.e. continue. The same rule is kept for every estimator
(value = inf). They are counted separately ('n_invalid'); statistics (median, 90th percentile, near-threshold) use valid
steps only; 'n_ge_valid' counts valid steps with value >= DELTA, 'n_ge_incl_invalid' adds the invalid ones.
Per estimator and group: n_total, n_invalid, n_valid, n_ge_valid, share_ge_valid, n_ge_incl_invalid, share_ge_of_total,
median and 90th percentile (linear interpolation) over valid steps. E0 also: n with 0.15 <= H < 0.35.
Rounding: probe_ent values are rounded to 4 decimals, so a sum/mean can move by up to 0.00005 * len; the number of valid
steps within +-0.001 of DELTA is counted for every estimator ('n_near_025').

Usage: cd repro && python3 diag/analyze_alt_stop_entropy.py   (writes diag/analyze_alt_stop_entropy.json)
"""
import json
import math
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parent.parent
IN_REL = 'llm/final/merged.jsonl'
OUT_REL = 'diag/analyze_alt_stop_entropy.json'
DELTA = 0.25
NEAR = 0.001
BAND = (0.15, 0.35)
BRACE_ID = 90  # token id of the bare '{' in the InternVL3.5 tokenizer, as seen in probe_ids[0] of the saved rows
METHODS = ('tsr', 'visref_raw_rank')
ESTIMATORS = ('E0', 'E1', 'E2', 'E3', 'E4', 'E5')


def valid(step):
    return bool(step.get('probe_valid'))


def recompute_H(step):
    """The way the code forms H from probe_ent (methods.py:362)."""
    return sum(step['probe_ent']) if valid(step) else math.inf


def first_content_index(step):
    """Index into probe_ent of the first content token (see docstring, E2)."""
    return 1 if step['probe_ids'][0] == BRACE_ID else 0


def estimate(step, name):
    """Value of estimator `name` on one step; inf for an invalid probe (the code's rule), for every estimator."""
    if not valid(step):
        return math.inf
    pe, pf = step['probe_ent'], step['probe_ent_full']
    if name == 'E0':
        return step['H']
    if name == 'E1':
        return sum(pe) / len(pe)
    if name == 'E2':
        return pe[first_content_index(step)]
    if name == 'E3':
        return max(pe)
    if name == 'E4':
        return sum(pf)
    if name == 'E5':
        return sum(pf) / len(pf)
    raise ValueError(name)


def percentile(xs, q):
    """Linear interpolation between order statistics (numpy's default); None for an empty list."""
    if not xs:
        return None
    s = sorted(xs)
    pos = (len(s) - 1) * q / 100
    lo = math.floor(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summarize(steps, name):
    vals = [estimate(s, name) for s in steps]
    ok = [v for v, s in zip(vals, steps) if valid(s)]
    n, nv = len(steps), len(ok)
    ge = sum(v >= DELTA for v in ok)
    out = {'n_total': n, 'n_invalid': n - nv, 'n_valid': nv, 'n_ge_valid': ge,
           'share_ge_valid': round(ge / nv, 4) if nv else None,
           'n_ge_incl_invalid': ge + (n - nv), 'share_ge_of_total': round((ge + n - nv) / n, 4) if n else None,
           'median': None if not ok else round(percentile(ok, 50), 4),
           'p90': None if not ok else round(percentile(ok, 90), 4),
           'n_near_025': sum(abs(v - DELTA) <= NEAR for v in ok)}
    if name == 'E0':
        out['n_in_015_035'] = sum(BAND[0] <= v < BAND[1] for v in ok)
    return out


def groups(rows):
    """{method: {group: [steps]}} for the groups of the docstring."""
    out = {}
    for m in METHODS:
        g = {'a_first_close_eos': [], 'b_first_cap': [], 'c_k2plus_close': [], 'c_k2plus_eos': [], 'c_k2plus_cap': []}
        for r in rows:
            if r['method'] != m or r.get('status') != 'ok':
                continue
            for s in r['steps']:
                if s['k'] == 1:
                    g['a_first_close_eos' if s['think_end'] in ('close', 'eos') else 'b_first_cap'].append(s)
                else:
                    g['c_k2plus_' + s['think_end']].append(s)
        out[m] = g
    return out


def sanity_e0(rows):
    """E0 sanity over all tsr / visref_raw_rank ok steps: stored H against the sum recomputed from probe_ent."""
    n = bad = dec = n_inf = 0
    maxdiff = 0.0
    for r in rows:
        if r['method'] not in METHODS or r.get('status') != 'ok':
            continue
        for s in r['steps']:
            n += 1
            h, rc = s['H'], recompute_H(s)
            if math.isinf(h) or math.isinf(rc):
                n_inf += 1
                if not (math.isinf(h) and math.isinf(rc)):
                    bad += 1
                continue
            d = abs(h - rc)
            maxdiff = max(maxdiff, d)
            if d > 0.00005 * len(s['probe_ent']) + 1e-9:
                bad += 1
            if (h < DELTA) != (rc < DELTA):
                dec += 1
    return {'n_steps': n, 'n_both_inf': n_inf, 'n_mismatch_beyond_rounding': bad, 'max_abs_diff_finite': round(maxdiff, 6),
            'n_decision_differs': dec}


def per_question_a(rows):
    """For each method and estimator: questions whose first step closed (close/eos): stop under E0 -> continue (flip),
    already continue under E0, and stop under both."""
    out = {}
    for m in METHODS:
        qs = [r for r in rows if r['method'] == m and r.get('status') == 'ok' and r['steps']
              and r['steps'][0]['think_end'] in ('close', 'eos')]
        d = {'n_questions': len(qs), 'e0_already_continue': sum(estimate(r['steps'][0], 'E0') >= DELTA for r in qs)}
        for e in ESTIMATORS[1:]:
            d[e] = {'flip_stop_to_continue': sum(estimate(r['steps'][0], 'E0') < DELTA <= estimate(r['steps'][0], e) for r in qs),
                    'continue_to_stop': sum(estimate(r['steps'][0], 'E0') >= DELTA > estimate(r['steps'][0], e) for r in qs)}
        out[m] = d
    return out


def e2_branches(rows):
    c = {'bare_brace_index1': 0, 'fused_index0': 0}
    for r in rows:
        if r['method'] in METHODS and r.get('status') == 'ok':
            for s in r['steps']:
                if valid(s):
                    c['bare_brace_index1' if first_content_index(s) else 'fused_index0'] += 1
    return c


def analyze(rows):
    g = groups(rows)
    return {'rules_date': '2026-10-20', 'input': IN_REL, 'delta': DELTA, 'near_window': NEAR,
            'e0_sanity': sanity_e0(rows), 'e2_branches_valid_steps': e2_branches(rows),
            'groups': {m: {gn: {e: summarize(st, e) for e in ESTIMATORS} for gn, st in gg.items()} for m, gg in g.items()},
            'per_question_a': per_question_a(rows)}


def main():
    rows = [json.loads(l) for l in open(REPRO / IN_REL) if l.strip()]
    res = analyze(rows)
    with open(REPRO / OUT_REL, 'w') as f:
        json.dump(res, f, indent=1)
    print('wrote', OUT_REL)
    return 0


if __name__ == '__main__':
    sys.exit(main())
