"""Independent recomputation for the cause review (2026-10-11).
Reads b_main merged.jsonl (3,000 rows) and fill merged.jsonl (994 rows); writes analyze_cause.json.
Run: python3 -I analyze_cause.py
"""
import json, re, sys, random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/mnt/user-data/uploads/VisRef/repro')
OUT = Path(__file__).resolve().parent / 'analyze_cause.json'

rows = [json.loads(l) for l in open(ROOT / 'llm/final/merged.jsonl')]
fill = [json.loads(l) for l in open(ROOT / 'llm/fill/merged.jsonl')]
by = defaultdict(dict)
for r in rows:
    by[r['pid']][r['method']] = r
fill_by = {r['pid']: r for r in fill}
pids = sorted(by, key=int)
assert len(pids) == 1000 and all(len(by[p]) == 3 for p in pids)
M = ('st', 'tsr', 'visref_raw_rank')
res = {}

def acc(ms, ps):
    return {m: round(100 * sum(by[p][m]['correct'] for p in ps) / len(ps), 2) for m in ms}

def paired(a, b, ps, B=2000, seed=0):
    d = [by[p][a]['correct'] - by[p][b]['correct'] for p in ps]
    n = len(d); mean = sum(d) / n
    rng = random.Random(seed); bs = []
    for _ in range(B):
        s = sum(d[rng.randrange(n)] for _ in range(n)) / n
        bs.append(s)
    bs.sort()
    return {'n': n, 'diff_pp': round(100 * mean, 2), 'ci95': [round(100 * bs[int(0.025 * B)], 2), round(100 * bs[int(0.975 * B) - 1], 2)],
            'win': sum(x > 0 for x in d), 'lose': sum(x < 0 for x in d)}

res['acc_all'] = acc(M, pids)
res['paired_all'] = {'visref-tsr': paired('visref_raw_rank', 'tsr', pids), 'visref-st': paired('visref_raw_rank', 'st', pids), 'tsr-st': paired('tsr', 'st', pids)}

# first-step end (pre-treatment: shared across methods)
fe = {p: by[p]['st']['steps'][0]['think_end'] for p in pids}
assert all(by[p][m]['steps'][0]['think_end'] == fe[p] for p in pids for m in M)
res['first_step_end'] = dict(Counter(fe.values()))
res['by_first_end'] = {}
for e in ('close', 'cap', 'eos'):
    ps = [p for p in pids if fe[p] == e]
    res['by_first_end'][e] = {'n': len(ps), 'acc': acc(M, ps), 'visref-tsr': paired('visref_raw_rank', 'tsr', ps), 'tsr-st': paired('tsr', 'st', ps)}

# k distribution and continuation transitions
kd = {m: dict(Counter(by[p][m]['k'] for p in pids)) for m in M}
res['k_dist'] = kd
trans = {m: Counter() for m in ('tsr', 'visref_raw_rank')}
for m in trans:
    for p in pids:
        st = by[p][m]['steps']
        for i, s in enumerate(st):
            cont = i < len(st) - 1
            trans[m][f"{s['think_end']}->{'continue' if cont else 'stop'}"] += 1
res['step_transitions'] = {m: dict(c) for m, c in trans.items()}

# k groups (post-treatment, exploratory)
both1 = [p for p in pids if by[p]['tsr']['k'] == 1 and by[p]['visref_raw_rank']['k'] == 1]
any_gt1 = [p for p in pids if by[p]['tsr']['k'] > 1 or by[p]['visref_raw_rank']['k'] > 1]
res['k_groups'] = {'both_k1': {'n': len(both1), 'acc': acc(M, both1), 'visref-tsr': paired('visref_raw_rank', 'tsr', both1)},
                   'any_k>1': {'n': len(any_gt1), 'acc': acc(M, any_gt1), 'visref-tsr': paired('visref_raw_rank', 'tsr', any_gt1), 'tsr-st': paired('tsr', 'st', any_gt1)}}
vk1 = [p for p in pids if by[p]['visref_raw_rank']['k'] == 1]
res['visref_k1_vs_st'] = {'n': len(vk1), 'acc': acc(('st', 'visref_raw_rank'), vk1), 'visref-st': paired('visref_raw_rank', 'st', vk1),
                          'pred_differs': sum(by[p]['visref_raw_rank']['prediction'] != by[p]['st']['prediction'] for p in vk1)}
tk1 = [p for p in pids if by[p]['tsr']['k'] == 1]
res['tsr_k1_vs_st_noise'] = {'n': len(tk1), 'pred_differs': sum(by[p]['tsr']['prediction'] != by[p]['st']['prediction'] for p in tk1)}

# injection amounts
inj = []
for p in pids:
    for s in by[p]['visref_raw_rank']['steps']:
        if 'n_inject' in s:
            inj.append((s['n_inject'], s['m'], by[p]['visref_raw_rank']['n_visual'], s.get('n_unique_z'), s.get('rank_V'), s['think_end'], s['think_tokens']))
def q(xs, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    xs = sorted(xs); return {str(qq): round(xs[min(len(xs) - 1, int(qq * len(xs)))], 3) for qq in qs}
res['injection'] = {'n_steps': len(inj), 'n_inject_over_m': q([a / b for a, b, *_ in inj]), 'n_inject_over_V': q([a / c for a, b, c, *_ in inj]),
                    'steps_below_m': sum(a < b for a, b, *_ in inj), 'n_inject_eq_min(m,unique_z,rankV)': sum(a == min(b, u, rv) for a, b, c, u, rv, *_ in inj),
                    'n_inject': q([a for a, *_ in inj]), 'n_visual_dist': dict(Counter(by[p]['st']['n_visual'] for p in pids))}

# dose-response (observational): per-question mean n_inject/|V| vs correct, within VisRef, stratified by first end and |V|
dose = defaultdict(lambda: [0, 0])
for p in pids:
    r = by[p]['visref_raw_rank']; st = [s for s in r['steps'] if 'n_inject' in s]
    if not st: continue
    frac = sum(s['n_inject'] for s in st) / (len(st) * r['n_visual'])
    b = 'lt5%' if frac < 0.05 else 'lt10%' if frac < 0.10 else 'lt20%' if frac < 0.2 else 'ge20%'
    key = f"{fe[p]}|V={r['n_visual']}|{b}"
    dose[key][0] += r['correct'] - by[p]['tsr']['correct']; dose[key][1] += 1
res['dose_visref_minus_tsr_by_stratum'] = {k: {'n': v[1], 'sum_diff': v[0]} for k, v in sorted(dose.items())}

# </think> re-emission and answer behaviour by method and last-step end
reem = {}
for m in M:
    c = Counter()
    for p in pids:
        r = by[p][m]; last = r['steps'][-1]['think_end']
        c[f'{last}|reemit'] += int(r['final_y_start'] > 0); c[f'{last}|n'] += 1
        c['af'] += int(bool(r.get('final_af'))); c['final_cap'] += int(r['final_finish'] == 'cap')
    reem[m] = dict(c)
res['reemission'] = reem

# text right after injection: what does z_{k+1} start with? (VisRef vs TSR continuation openings)
def openings(m):
    c = Counter(); ex = []
    for p in pids:
        r = by[p][m]
        if r['k'] < 2: continue
        t = r['thinking']
        parts = re.split(r'Wait, think more', t)
        for seg in parts[1:]:
            s = seg.lstrip()[:60].replace('\n', '\\n')
            head = s[:25]
            c[head] += 1
            if len(ex) < 8: ex.append(s)
    return c.most_common(15), ex
res['continuation_openings'] = {m: {'top': openings(m)[0], 'examples': openings(m)[1]} for m in ('tsr', 'visref_raw_rank')}
# mentions of image/figure/picture in continuation segments
def img_mentions(m):
    n = 0; hit = 0
    for p in pids:
        r = by[p][m]
        if r['k'] < 2: continue
        for seg in re.split(r'Wait, think more', r['thinking'])[1:]:
            n += 1; seg = re.sub(r'<VISREF[^>]*>', '', seg)[:400].lower()
            hit += bool(re.search(r'\b(image|figure|picture|diagram|graph|chart|look at|looking at)\b', seg))
    return {'segments': n, 'mention_in_first_400_chars': hit}
res['img_mentions_in_continuations'] = {m: img_mentions(m) for m in ('tsr', 'visref_raw_rank')}

# probe H: step-1 H distribution for TSR vs VisRef (same z_1) -- does V raise H?
h = {'tsr': [], 'visref_raw_rank': []}
flip = Counter()
for p in pids:
    a, b = by[p]['tsr']['steps'][0]['H'], by[p]['visref_raw_rank']['steps'][0]['H']
    h['tsr'].append(a); h['visref_raw_rank'].append(b)
    flip[('tsr_cont' if a >= 0.25 else 'tsr_stop') + '|' + ('vis_cont' if b >= 0.25 else 'vis_stop')] += 1
fin = lambda xs: [x for x in xs if x != float('inf')]
res['step1_H'] = {m: {'median': q(fin(v), (0.5,))['0.5'], 'ge_delta': sum(x >= 0.25 for x in v), 'inf': sum(x == float('inf') for x in v)} for m, v in h.items()}
res['step1_decision_cross'] = dict(flip)

# fill: dual probe H - H_noV, tail spearman, n_fill
fs = [s for r in fill for s in r['steps'] if 'H_noV' in s]
dH = [s['H'] - s['H_noV'] for s in fs if s['H'] != float('inf') and s['H_noV'] not in (None, float('inf'))]
res['fill'] = {'rows': len(fill), 'steps_with_dual': len(fs), 'H_minus_HnoV': q(dH),
               'flip': dict(Counter(('V_cont' if s['H'] >= 0.25 else 'V_stop') + '|' + ('noV_cont' if (s['H_noV'] is None or s['H_noV'] >= 0.25) else 'noV_stop') for s in fs)),
               'tail_spearman': q([s['tail_rel_spearman'] for s in fs if s.get('tail_rel_spearman') is not None]),
               'n_fill_over_m': q([s['n_fill'] / s['m'] for s in fs if s.get('m')])}
# fill vs visref paired on complete prefix (as recorded)
fp = [p for p in pids if p in fill_by]
d = [fill_by[p]['correct'] - by[p]['visref_raw_rank']['correct'] for p in fp]
d2 = [fill_by[p]['correct'] - by[p]['tsr']['correct'] for p in fp]
res['fill_paired'] = {'n': len(fp), 'fill-visref_pp': round(100 * sum(d) / len(d), 2), 'fill-tsr_pp': round(100 * sum(d2) / len(d2), 2),
                      'fill_acc': round(100 * sum(fill_by[p]['correct'] for p in fp) / len(fp), 2)}

# paper-scale sanity: median unique tokens per 1024-step vs m; fraction of steps where rank could reach m under hidden-state Z (rank<=T)
T_ge_m = sum(1 for a, b, c, u, rv, e, T in inj if T >= b)
res['steps_T_ge_m'] = {'n': len(inj), 'T_ge_m': T_ge_m, 'unique_z_median': q([u for *_, u, rv, e, T in [(a, b, c, u, rv, e, T) for a, b, c, u, rv, e, T in inj]], (0.5,))}

json.dump(res, open(OUT, 'w'), indent=1, ensure_ascii=False, default=str)
print(json.dumps(res, indent=1, ensure_ascii=False, default=str))
