"""Supplementary calculations for the cause review: reading (c), |V|-stratum natural experiment, k>1 cross-tab,
answer-change direction at k=1, cost proxy and power for the proposed oracle experiment. Writes extra_calcs.json."""
import json, math, random
from collections import Counter
from pathlib import Path

ROOT = Path('/mnt/user-data/uploads/VisRef/repro')
OUT = Path(__file__).resolve().parent / 'extra_calcs.json'
rows = [json.loads(l) for l in open(ROOT / 'llm/final/merged.jsonl')]
by = {}
for r in rows:
    by.setdefault(r['pid'], {})[r['method']] = r
ids = json.load(open(ROOT / 'kaggle/runs/b_main_s1/output/results/ids.json'))
ids = ids['ids'] if isinstance(ids, dict) else ids
fe = {p: by[p]['st']['steps'][0]['think_end'] for p in by}
M = ('st', 'tsr', 'visref_raw_rank')
res = {}

def paired(a, b, ps, B=2000):
    d = [by[p][a]['correct'] - by[p][b]['correct'] for p in ps]; n = len(d); rng = random.Random(0)
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return dict(n=n, diff_pp=round(100 * sum(d) / n, 2), ci95=[round(100 * bs[int(0.025 * B)], 2), round(100 * bs[int(0.975 * B) - 1], 2)],
                win=sum(x > 0 for x in d), lose=sum(x < 0 for x in d))

res['reading_c_af_rows_wrong'] = {m: round(sum(by[p][m]['correct'] * (not by[p][m].get('final_af')) for p in by) / 10, 1) for m in M}

# natural experiment by |V|
nat = {}
for N in (256, 768, 1792, 3328):
    ps = [p for p in by if by[p]['st']['n_visual'] == N]
    steps = [s for p in ps for s in by[p]['visref_raw_rank']['steps'] if 'n_inject' in s]
    nat[N] = {'n': len(ps), 'steps': len(steps), 'steps_at_m': sum(s['n_inject'] == s['m'] for s in steps),
              'median_inject_over_V': round(sorted(s['n_inject'] / N for s in steps)[len(steps) // 2], 3),
              'acc': {m: round(100 * sum(by[p][m]['correct'] for p in ps) / len(ps), 1) for m in M},
              'visref-tsr': paired('visref_raw_rank', 'tsr', ps),
              'visref-tsr_cap_only': paired('visref_raw_rank', 'tsr', [p for p in ps if fe[p] == 'cap'])}
res['natural_experiment_by_V'] = nat

# k>1 cross tab and paired within cells
cross = Counter((by[p]['visref_raw_rank']['k'] > 1, by[p]['tsr']['k'] > 1) for p in by)
res['k_gt1_cross_visref_tsr'] = {f'vis>{int(a)}_tsr>{int(b)}': v for (a, b), v in cross.items()}
both = [p for p in by if by[p]['visref_raw_rank']['k'] > 1 and by[p]['tsr']['k'] > 1]
res['both_k_gt1_visref-tsr'] = paired('visref_raw_rank', 'tsr', both)

# direction of answer changes at k=1 (VisRef vs ST) and TSR noise baseline
def changes(m):
    c = Counter()
    for p in by:
        v, s = by[p][m], by[p]['st']
        if v['k'] == 1 and v['prediction'] != s['prediction']:
            c[f"{fe[p]}|{'m_right' if v['correct'] else ('st_right' if s['correct'] else 'both_wrong')}"] += 1
    return dict(c)
res['k1_answer_changes'] = {'visref_vs_st': changes('visref_raw_rank'), 'tsr_vs_st_noise': changes('tsr')}

# cost proxy (prefill-dominated: each generate call re-prefills the full context) and oracle experiment estimates
def cost(r, inj_fn):
    L = r['prompt_len']; tot = 0
    for s in r['steps']:
        tt = s['think_tokens']; nchunks = max(1, -(-tt // 1024))
        tot += nchunks * L + tt; L += tt; tot += L; L += inj_fn(s, r); L += 4
    return tot + L
V = {p: by[p]['visref_raw_rank'] for p in by}
tot_rank = sum(cost(V[p], lambda s, r: s.get('n_inject', 0)) for p in by)
cap = [p for p in ids if fe[p] != 'close']
cap_only = [p for p in ids if fe[p] == 'cap']
res['cost_proxy'] = {'tsr/rank': round(sum(cost(by[p]['tsr'], lambda s, r: 0) for p in by) / tot_rank, 2),
                     'fill(m)/rank': round(sum(cost(V[p], lambda s, r: s.get('m', 0)) for p in by) / tot_rank, 2),
                     'allN/rank': round(sum(cost(V[p], lambda s, r: r['n_visual']) for p in by) / tot_rank, 2),
                     'share_rank_cost_in_cap_eos': round(sum(cost(V[p], lambda s, r: s.get('n_inject', 0)) for p in cap) / tot_rank, 2),
                     'max_final_ctx_visref': max(r['final_context_len'] for r in V.values()),
                     'rows_over_40960_under_allN': sum(1 for p in by if V[p]['prompt_len'] + sum(s['think_tokens'] + V[p]['n_visual'] + 4 for s in V[p]['steps']) + 1134 > 40960)}
H_rank_1000 = 9.5  # assumed: fill session 11.3h at 1.19x cost incl. dual probe -> rank-equivalent ~9.5h (estimate)
est = {}
for n in (100, 150, 200, 300, 400, len(cap_only)):
    sub = cap_only[:n]
    est[n] = {'allN_h': round(H_rank_1000 * sum(cost(V[p], lambda s, r: r['n_visual']) for p in sub) / tot_rank, 1),
              'm_fill_h': round(H_rank_1000 * sum(cost(V[p], lambda s, r: s.get('m', 0)) for p in sub) / tot_rank, 1)}
res['gpu_hours_estimate_cap_prefix'] = est
d = sum(by[p]['visref_raw_rank']['correct'] != by[p]['tsr']['correct'] for p in cap_only)
res['discordance_cap'] = {'n': len(cap_only), 'discordant': d, 'rate': round(d / len(cap_only), 3)}
res['power_cap_prefix'] = {n: {'SE_pp': round(100 * math.sqrt(d / len(cap_only) / n), 2), 'MDE80_pp': round(100 * math.sqrt(d / len(cap_only) / n) * 2.8, 2)} for n in (100, 150, 200, 300, 400, len(cap_only))}
res['paper_implied_stratum_effect_pp'] = round(3.5 / (len(cap_only) / 1000), 2)
json.dump(res, open(OUT, 'w'), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
