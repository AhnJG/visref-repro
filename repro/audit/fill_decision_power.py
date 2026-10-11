"""Independent re-analysis for the visref_fill go/no-go review (2026-10-11).
Reads b_main s1-s8 result rows (read-only), rescores with visref/scoring.py (same call as report.rescore_rows;
report.py itself imports summarize.py which is not in the upload, so the 10-line loop is re-implemented here),
and computes: subgroup sizes, fill-can-differ fractions, dose-response of injection amount within existing data,
discordance rates, and paired-binary power / MDE under several discordance scenarios.
Usage: python3 analyze_fill_power.py  (writes analyze_fill_power.json next to itself)"""
import json, glob, sys, math, random, statistics as st, collections
from pathlib import Path

ROOT = '/mnt/user-data/uploads/VisRef'
sys.path.insert(0, f'{ROOT}/repro')
from visref.scoring import score  # noqa: E402

OUT = Path(__file__).with_suffix('.json')
data = json.load(open(f'{ROOT}/data/raw/mathvista_testmini.json'))
rows = {}
for s in range(1, 9):
    for f in sorted(glob.glob(f'{ROOT}/repro/kaggle/runs/b_main_s{s}/output/results/results_*.jsonl')):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                rows[(str(r['pid']), r['method'])] = r
changed = 0
for (p, m), r in rows.items():  # == report.rescore_rows
    n = score(r['final_answer'], data[str(p)])
    r['correct_recorded'] = r['correct']
    changed += n['correct'] != r['correct']
    for k in ('correct', 'extracted', 'prediction', 'rule', 'extract_failed'):
        r[k] = n[k]
M = ('st', 'tsr', 'visref_raw_rank')
pids = sorted({p for p, m in rows if all((p, mm) in rows for mm in M)}, key=int)
R = {m: {p: rows[(p, m)] for p in pids} for m in M}
out = {'n_rows': len(rows), 'n_pairs': len(pids), 'rescore_changed': changed}

# ---------------------------------------------------------------- headline + paired stats
random.seed(0)
def paired(a, b, ps, B=2000):
    xa = [R[a][p]['correct'] for p in ps]; xb = [R[b][p]['correct'] for p in ps]
    d = [x - y for x, y in zip(xa, xb)]
    n = len(ps)
    a_only = sum(x == 1 and y == 0 for x, y in zip(xa, xb)); b_only = sum(x == 0 and y == 1 for x, y in zip(xa, xb))
    boots = []
    for _ in range(B):
        idx = [random.randrange(n) for _ in range(n)]
        boots.append(100 * sum(d[i] for i in idx) / n)
    boots.sort()
    k = min(a_only, b_only); nn = a_only + b_only
    p = min(1.0, 2 * sum(math.comb(nn, i) for i in range(k + 1)) / 2 ** nn) if nn else 1.0
    return {'n': n, 'diff': 100 * sum(d) / n, 'ci': [boots[int(.025 * B)], boots[int(.975 * B) - 1]],
            'a_only': a_only, 'b_only': b_only, 'discordance': (a_only + b_only) / n, 'p_mcnemar': p}
out['acc'] = {m: 100 * sum(R[m][p]['correct'] for p in pids) / len(pids) for m in M}
out['pairs_all'] = {f'{a}-{b}': paired(a, b, pids) for a, b in
                    (('tsr', 'st'), ('visref_raw_rank', 'st'), ('visref_raw_rank', 'tsr'))}

# ---------------------------------------------------------------- subgroups (step-1 end is pre-treatment: shared fork)
def s1end(p):
    return R['st'][p]['steps'][0]['think_end']
grp = collections.defaultdict(list)
for p in pids:
    grp[s1end(p)].append(p)
out['step1_end_n'] = {k: len(v) for k, v in grp.items()}
V = R['visref_raw_rank']; T = R['tsr']; S_ = R['st']

def sub_stats(ps, name):
    d = {'n': len(ps)}
    for m in M:
        d[f'acc_{m}'] = 100 * sum(R[m][p]['correct'] for p in ps) / max(1, len(ps))
    d['k_gt1_visref'] = sum(V[p]['k'] > 1 for p in ps); d['k_gt1_tsr'] = sum(T[p]['k'] > 1 for p in ps)
    d['k_gt1_any'] = sum(V[p]['k'] > 1 or T[p]['k'] > 1 for p in ps)
    for a, b in (('visref_raw_rank', 'tsr'), ('visref_raw_rank', 'st'), ('tsr', 'st')):
        if ps:
            d[f'{a}-{b}'] = paired(a, b, ps, B=500)
    return d
out['sub'] = {g: sub_stats(ps, g) for g, ps in grp.items()}
out['sub']['all'] = sub_stats(pids, 'all')
# k>1 (either VisRef or TSR continued) vs both k=1
k1both = [p for p in pids if V[p]['k'] == 1 and T[p]['k'] == 1]
kgt1 = [p for p in pids if V[p]['k'] > 1 or T[p]['k'] > 1]
out['sub']['both_k1'] = sub_stats(k1both, 'both_k1'); out['sub']['any_kgt1'] = sub_stats(kgt1, 'any_kgt1')
vk1 = [p for p in pids if V[p]['k'] == 1]
out['sub']['visref_k1'] = sub_stats(vk1, 'visref_k1')
out['sub']['visref_k1_close'] = sub_stats([p for p in vk1 if s1end(p) == 'close'], 'visref_k1_close')
out['sub']['visref_k1_cap'] = sub_stats([p for p in vk1 if s1end(p) == 'cap'], 'visref_k1_cap')
out['sub']['visref_kgt1'] = sub_stats([p for p in pids if V[p]['k'] > 1], 'visref_kgt1')

# ---------------------------------------------------------------- where can fill differ from visref_raw_rank?
def fill_changes_step1(p):
    s = V[p]['steps'][0]
    return s.get('m', 0) > 0 and s.get('n_inject', 0) < s['m']
def fill_changes_any(p):
    return any(s.get('m', 0) > 0 and s.get('n_inject', 0) < s['m'] for s in V[p]['steps'])
fc = {'step1_changes': sum(fill_changes_step1(p) for p in pids), 'any_step_changes': sum(fill_changes_any(p) for p in pids),
      'no_step_changes_(fill==visref_by_construction)': sum(not fill_changes_any(p) for p in pids)}
for g, ps in grp.items():
    fc[f'step1_changes_{g}'] = sum(fill_changes_step1(p) for p in ps)
    fc[f'n_{g}'] = len(ps)
# steps reaching m
steps = [s for p in pids for s in V[p]['steps'] if s.get('m')]
fc['steps_total'] = len(steps); fc['steps_at_m'] = sum(s['n_inject'] >= s['m'] for s in steps)
fc['inject_over_m_median_steps'] = st.median(s['n_inject'] / s['m'] for s in steps)
fc['inject_over_V_median_steps'] = st.median(s['n_inject'] / V[p]['n_visual'] for p in pids for s in V[p]['steps'] if s.get('m'))
# extra tokens fill would add at step 1 (m - n_inject)
fc['extra_tokens_step1_q'] = sorted(V[p]['steps'][0]['m'] - V[p]['steps'][0]['n_inject'] for p in pids if V[p]['steps'][0].get('m'))
q = fc['extra_tokens_step1_q']; fc['extra_tokens_step1_q'] = [q[int(f * (len(q) - 1))] for f in (.1, .25, .5, .75, .9)]
fc['m_step1_q'] = [sorted(V[p]['steps'][0]['m'] for p in pids)[int(f * (len(pids) - 1))] for f in (.1, .25, .5, .75, .9)]
out['fill_can_differ'] = fc

# ---------------------------------------------------------------- dose-response inside existing data (k=1 VisRef rows)
# At k=1, VisRef differs from ST only by V_1 (shared step-1 fork, shared main RNG state). Bin by n_inject/m and by n_inject.
def bins(ps, key, edges):
    res = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [p for p in ps if lo <= key(p) < hi]
        if not sel:
            res.append({'bin': [lo, hi], 'n': 0}); continue
        res.append({'bin': [lo, hi], 'n': len(sel),
                    'pred_differs_vs_st': sum(str(V[p]['prediction']) != str(S_[p]['prediction']) for p in sel),
                    'pred_differs_tsr_vs_st': sum(str(T[p]['prediction']) != str(S_[p]['prediction']) for p in sel if T[p]['k'] == 1),
                    'n_tsr_k1': sum(T[p]['k'] == 1 for p in sel),
                    'visref_only': sum(V[p]['correct'] and not S_[p]['correct'] for p in sel),
                    'st_only': sum(S_[p]['correct'] and not V[p]['correct'] for p in sel),
                    'reemit_visref': sum(bool(V[p].get('final_y_start')) for p in sel),
                    'reemit_st': sum(bool(S_[p].get('final_y_start')) for p in sel),
                    'n_visual_median': st.median(V[p]['n_visual'] for p in sel),
                    'n_inject_median': st.median(V[p]['steps'][0]['n_inject'] for p in sel)})
    return res
ratio = lambda p: V[p]['steps'][0]['n_inject'] / V[p]['steps'][0]['m']
ninj = lambda p: V[p]['steps'][0]['n_inject']
dr = {}
for g in ('close', 'cap'):
    ps = [p for p in vk1 if s1end(p) == g]
    dr[f'{g}_by_ratio'] = bins(ps, ratio, [0, .15, .3, .6, 1.01])
    dr[f'{g}_by_ninject'] = bins(ps, ninj, [0, 80, 120, 160, 10000])
out['dose_response_k1'] = dr

# step-1 probe entropy shift: VisRef H_1 vs TSR H_1 on the same question (same z_1; only V_1 differs; probe RNG same seed)
def H(r):
    h = r['steps'][0].get('H'); return None if h is None else (float('inf') if h == 'inf' or h == 'Infinity' else float(h))
hs = []
for p in pids:
    hv, ht = H(V[p]), H(T[p])
    if hv is None or ht is None: continue
    hs.append((p, hv, ht))
def hstats(sel):
    fin = [(hv, ht) for p, hv, ht in sel if math.isfinite(hv) and math.isfinite(ht)]
    d = {'n': len(sel), 'both_finite': len(fin),
         'median_H_visref': st.median(hv for hv, ht in fin) if fin else None,
         'median_H_tsr': st.median(ht for hv, ht in fin) if fin else None,
         'visref_H_gt_tsr': sum(hv > ht for hv, ht in fin), 'visref_H_lt_tsr': sum(hv < ht for hv, ht in fin),
         'visref_ge_delta': sum(hv >= .25 for p, hv, ht in sel), 'tsr_ge_delta': sum(ht >= .25 for p, hv, ht in sel),
         'decision_flips': sum((hv >= .25) != (ht >= .25) for p, hv, ht in sel),
         'visref_continue_tsr_stop': sum(hv >= .25 and ht < .25 for p, hv, ht in sel),
         'tsr_continue_visref_stop': sum(hv < .25 and ht >= .25 for p, hv, ht in sel)}
    return d
ph = {}
for g in ('close', 'cap', 'eos'):
    sel = [(p, hv, ht) for p, hv, ht in hs if s1end(p) == g]
    ph[g] = hstats(sel)
    # dose response of decision flips vs ratio
    for lo, hi in ((0, .3), (.3, .6), (.6, 1.01)):
        ss = [(p, hv, ht) for p, hv, ht in sel if lo <= ratio(p) < hi]
        ph[f'{g}_ratio_{lo}_{hi}'] = {'n': len(ss), 'flips': sum((hv >= .25) != (ht >= .25) for p, hv, ht in ss),
                                      'visref_ge_delta': sum(hv >= .25 for p, hv, ht in ss), 'tsr_ge_delta': sum(ht >= .25 for p, hv, ht in ss),
                                      'median_H_visref': st.median(hv for p, hv, ht in ss if math.isfinite(hv)) if any(math.isfinite(hv) for p, hv, ht in ss) else None,
                                      'median_H_tsr': st.median(ht for p, hv, ht in ss if math.isfinite(ht)) if any(math.isfinite(ht) for p, hv, ht in ss) else None}
out['probe_H_step1'] = ph
# step transitions after close vs cap (TSR, VisRef), all steps
tr = {}
for m in ('tsr', 'visref_raw_rank'):
    c = collections.Counter()
    for p in pids:
        r = R[m][p]; stp = r['steps']
        for i, s in enumerate(stp):
            cont = i < len(stp) - 1
            c[(s['think_end'], 'continue' if cont else 'stop')] += 1
    tr[m] = {f'{a}->{b}': v for (a, b), v in sorted(c.items())}
out['transitions'] = tr

# ---------------------------------------------------------------- discordance (correctness) tables for power scenarios
disc = {}
for name, ps in (('all', pids), ('close', grp['close']), ('cap', grp['cap']), ('eos', grp.get('eos', [])),
                 ('visref_k1', vk1), ('visref_k1_close', [p for p in vk1 if s1end(p) == 'close']),
                 ('visref_k1_cap', [p for p in vk1 if s1end(p) == 'cap']), ('visref_kgt1', [p for p in pids if V[p]['k'] > 1]),
                 ('tsr_k1', [p for p in pids if T[p]['k'] == 1])):
    if not ps: continue
    d = {'n': len(ps)}
    for a, b in (('visref_raw_rank', 'tsr'), ('visref_raw_rank', 'st'), ('tsr', 'st')):
        xa = [R[a][p]['correct'] for p in ps]; xb = [R[b][p]['correct'] for p in ps]
        d[f'{a}-{b}'] = {'disc': sum(x != y for x, y in zip(xa, xb)), 'rate': sum(x != y for x, y in zip(xa, xb)) / len(ps),
                         'pred_differs': sum(str(R[a][p]['prediction']) != str(R[b][p]['prediction']) for p in ps)}
    disc[name] = d
out['discordance'] = disc

# ---------------------------------------------------------------- power / MDE (paired binary, normal approx; two-sided alpha .05, power .8)
def mde(n, disc_rate, z=1.959964 + 0.841621):
    # Delta = z * sqrt((disc - Delta^2)/n)  ->  Delta = z*sqrt(disc/(n+z^2))
    return 100 * z * math.sqrt(disc_rate / (n + z * z))
def power(n, disc_rate, delta_pp):
    d = delta_pp / 100
    se = math.sqrt(max(1e-12, disc_rate - d * d) / n)
    from math import erf, sqrt
    Phi = lambda x: .5 * (1 + erf(x / sqrt(2)))
    return 1 - Phi(1.959964 - d / se) + Phi(-1.959964 - d / se)
pw = {}
n_total = 1000
# share of subgroups scaled to 1000 from 868
share = {g: len(grp[g]) / len(pids) for g in grp}
pw['share_step1_end_868'] = share
pw['projected_n_1000'] = {g: round(1000 * share[g]) for g in grp}
# discordance scenarios for fill vs TSR (primary) and fill vs VisRef
scen = {
    'A_preregistered_8.8%': .088,
    'B_visref_vs_tsr_observed_all': disc['all']['visref_raw_rank-tsr']['rate'],
    'C_visref_vs_st_observed_all': disc['all']['visref_raw_rank-st']['rate'],
    'D_main_session_view(close=0, cap/kgt1 as observed)': None,
    'E_fill_changes_only_at_cap_half_rate': None,
}
# D: close questions contribute discordance 0 (main-session view); cap/eos contribute observed VisRef-vs-TSR rate in that stratum
d_close = disc['close']['visref_raw_rank-tsr']['rate']; d_cap = disc['cap']['visref_raw_rank-tsr']['rate']
d_eos = disc['eos']['visref_raw_rank-tsr']['rate'] if 'eos' in disc else 0
scen['D_main_session_view(close=0, cap/kgt1 as observed)'] = share['cap'] * d_cap + share.get('eos', 0) * d_eos
scen['E_fill_changes_only_at_cap_half_rate'] = share['cap'] * d_cap / 2 + share.get('eos', 0) * d_eos / 2
# F: close contributes TSR-vs-ST(k=1) noise-level discordance
d_noise = disc['tsr_k1']['tsr-st']['rate']
scen['F_close_noise_only'] = share['close'] * d_noise + share['cap'] * d_cap + share.get('eos', 0) * d_eos
pw['scenarios_disc_rate'] = scen
pw['mde_n1000'] = {k: mde(1000, v) for k, v in scen.items()}
pw['power_n1000_delta3.5'] = {k: power(1000, v, 3.5) for k, v in scen.items()}
pw['power_n1000_delta2.0'] = {k: power(1000, v, 2.0) for k, v in scen.items()}
# Effect dilution: if the true effect lives only on cap+eos questions (share s), a subgroup effect of Δ_sub becomes s*Δ_sub overall
s_cap = share['cap'] + share.get('eos', 0)
pw['dilution_share_cap_eos'] = s_cap
pw['mde_cap_eos_subgroup_only'] = {
    'n_sub': round(1000 * s_cap),
    'disc_cap_observed': d_cap,
    'mde_subgroup_pp': mde(round(1000 * s_cap), d_cap),
    'equivalent_overall_pp': s_cap * mde(round(1000 * s_cap), d_cap),
    'mde_if_all_1000_run_but_effect_only_in_subgroup(overall_scale)': mde(1000, scen['D_main_session_view(close=0, cap/kgt1 as observed)'])}
# subgroup designs
pw['designs'] = {}
for n in (392, 500, 600, 700, 1000):
    pw['designs'][f'all_n{n}'] = {k: round(mde(n, v), 2) for k, v in scen.items()}
pw['designs']['cap_eos_only_n%d' % round(1000 * s_cap)] = {'disc=cap_observed': round(mde(round(1000 * s_cap), d_cap), 2)}
pw['designs']['cap_eos_only_n%d_868base' % (len(grp['cap']) + len(grp.get('eos', [])))] = {
    'disc=cap_observed': round(mde(len(grp['cap']) + len(grp.get('eos', [])), d_cap), 2)}
pw['discordance_inputs'] = {'close_visref_tsr': d_close, 'cap_visref_tsr': d_cap, 'eos_visref_tsr': d_eos, 'tsr_k1_vs_st_noise': d_noise}
out['power'] = pw

# ---------------------------------------------------------------- wall time by subgroup (VisRef rows; wall_s includes batch waiting)
wt = {}
for g, ps in list(grp.items()) + [('all', pids)]:
    w = [V[p]['wall_s'] for p in ps]
    g_tok = [V[p]['gen_tokens'] for p in ps]
    wt[g] = {'n': len(ps), 'wall_sum_visref_s': round(sum(w)), 'wall_median_s': st.median(w), 'gen_tokens_sum': sum(g_tok),
             'gen_tokens_median': st.median(g_tok), 'ctx_final_median': st.median(V[p]['final_context_len'] for p in ps),
             'k_mean_visref': st.mean(V[p]['k'] for p in ps)}
tot_tok = sum(V[p]['gen_tokens'] for p in pids)
for g in wt:
    wt[g]['gen_tokens_share'] = wt[g]['gen_tokens_sum'] / tot_tok
out['time_by_group'] = wt
# all-method tokens, to size a fill-only session relative to b_main sessions
out['gen_tokens_by_method'] = {m: sum(R[m][p]['gen_tokens'] for p in pids) for m in M}

json.dump(out, open(OUT, 'w'), indent=1, default=str)
print(json.dumps({k: out[k] for k in ('n_pairs', 'rescore_changed', 'acc', 'pairs_all', 'step1_end_n', 'fill_can_differ')}, indent=1, default=str))
print(json.dumps(out['discordance'], indent=1, default=str))
print(json.dumps(out['power'], indent=1, default=str))
