"""Pre-registered visref_fill analysis (EF 2026-10-10 "원인 분리 실험 사전등록 수정", amended 2026-10-11).
Sample: complete prefix of the seed-0 shuffled order (2 shards) among finished fill rows.
Comparisons: fill-TSR, fill-VisRef(raw_rank) paired difference, bootstrap 95% CI (2,000, seed 0), McNemar exact.
b_main rows: llm/final/merged.jsonl (v3 + LLM). fill rows: rescored with the current rule scorer (v3 when this was
written, v4 since 2026-10-11), then replaced by the LLM merge file if given (llm/fill/merged.jsonl: v3 rows + LLM).
Strata (pre-specified): b_main ST first-step end (close/cap/eos). k strata: exploratory.
Usage: cd repro && python3 diag/analyze_fill.py [fill_llm_merged.jsonl]"""
import json, glob, sys, statistics, collections
sys.path.insert(0, '.')
import summarize as S
from report import mcnemar_exact, rescore_rows

data = json.load(open('../data/raw/mathvista_testmini.json'))
B = {(r['pid'], r['method']): r for r in map(json.loads, open('llm/final/merged.jsonl'))}
F = {}
for f in sorted(glob.glob('kaggle/runs/b_main_fill/output/results/results_*.jsonl')):
    for l in open(f):
        if l.strip():
            r = json.loads(l); F[(str(r['pid']), r['method'])] = r
rescore_rows(F, data)
reading = 'current rule scorer only (LLM extraction pending)'
if len(sys.argv) > 1:
    for r in map(json.loads, open(sys.argv[1])):
        F[(r['pid'], r['method'])] = r
    reading = 'v3 + LLM'
rows = dict(B); rows.update(F)
order = json.load(open('kaggle/runs/b_main_fill/output/results/ids.json'))
pids = S.complete_prefix(rows, ('st', 'tsr', 'visref_raw_rank', 'visref_fill'), order, 2)
out = {'reading_fill': reading, 'n_prefix': len(pids), 'n_fill_rows': len(F), 'acc': {}, 'pairs': {}, 'strata': {}, 'diag': {}}
for m in ('st', 'tsr', 'visref_raw_rank', 'visref_fill'):
    out['acc'][m] = round(100 * sum(rows[(p, m)]['correct'] for p in pids) / len(pids), 2)
for a, b in (('visref_fill', 'tsr'), ('visref_fill', 'visref_raw_rank'), ('visref_fill', 'st'), ('visref_raw_rank', 'tsr')):
    pr = S.paired(rows, a, b, pids)
    out['pairs'][f'{a}-{b}'] = {'diff_pp': round(pr['diff_pp'], 2), 'ci95_pp': [round(x, 2) for x in pr['ci95_pp']],
                                'a_only': pr['a_only'], 'b_only': pr['b_only'], 'p_mcnemar': round(mcnemar_exact(pr['a_only'], pr['b_only']), 4)}
for e in ('close', 'cap', 'eos'):
    ps = [p for p in pids if rows[(p, 'st')]['steps'][0]['think_end'] == e]
    if not ps:
        continue
    d = {'n': len(ps)}
    for a, b in (('visref_fill', 'tsr'), ('visref_fill', 'visref_raw_rank')):
        pr = S.paired(rows, a, b, ps)
        d[f'{a}-{b}'] = [round(pr['diff_pp'], 2), [round(x, 2) for x in pr['ci95_pp']], pr['a_only'], pr['b_only']]
    out['strata'][e] = d
fr = [rows[(p, 'visref_fill')] for p in pids]
vr = [rows[(p, 'visref_raw_rank')] for p in pids]
steps = [s for r in fr for s in r['steps'] if s.get('m')]
q = lambda xs: [round(statistics.quantiles(xs, n=4)[i], 3) for i in range(3)] if len(xs) > 3 else xs
out['diag'] = {
    'inject_over_m_q25_50_75': q([s['n_inject'] / s['m'] for s in steps]),
    'steps_inject_lt_m': sum(s['n_inject'] < s['m'] for s in steps), 'n_steps': len(steps),
    'n_fill_q': q([s.get('n_fill', 0) for s in steps]),
    'tail_rel_spearman_median': round(statistics.median([s['tail_rel_spearman'] for s in steps if s.get('tail_rel_spearman') is not None]), 3),
    'mean_k': {'fill': round(statistics.mean(r['k'] for r in fr), 2), 'visref': round(statistics.mean(r['k'] for r in vr), 2),
               'tsr': round(statistics.mean(rows[(p, 'tsr')]['k'] for p in pids), 2)},
    'k1': {'fill': sum(r['k'] == 1 for r in fr), 'visref': sum(r['k'] == 1 for r in vr)},
    'stop': {'fill': dict(collections.Counter(r['stop_reason'] for r in fr)), 'visref': dict(collections.Counter(r['stop_reason'] for r in vr))},
    'reemit_think_end': {'fill': sum(bool(r.get('final_y_start')) for r in fr), 'visref': sum(bool(r.get('final_y_start')) for r in vr)},
    'af': {'fill': sum(bool(r.get('final_af')) for r in fr), 'visref': sum(bool(r.get('final_af')) for r in vr)},
    'step1_end_agree_with_b_main': sum(rows[(p, 'visref_fill')]['steps'][0]['think_end'] == rows[(p, 'st')]['steps'][0]['think_end'] for p in pids),
    'mean_think_tokens': {'fill': round(statistics.mean(r['think_tokens'] for r in fr)), 'visref': round(statistics.mean(r['think_tokens'] for r in vr))},
}
# dual probe: H with V vs without V; decision flips (H<delta vs H_noV<delta)
dp = [s for r in fr for s in r['steps'] if 'H_noV' in s and 'H' in s]
fin = [s for s in dp if s['H'] != float('inf') and s['H_noV'] is not None and s['H_noV'] != float('inf')]
out['dual_probe'] = {
    'n_steps': len(dp), 'both_finite': len(fin),
    'H_minus_HnoV_median': round(statistics.median([s['H'] - s['H_noV'] for s in fin]), 4) if fin else None,
    'stop_withV_continue_noV': sum(1 for s in dp if s['H'] < 0.25 and not (s['H_noV'] is not None and s['H_noV'] < 0.25)),
    'continue_withV_stop_noV': sum(1 for s in dp if not s['H'] < 0.25 and (s['H_noV'] is not None and s['H_noV'] < 0.25)),
    'by_end': {e: {'n': sum(1 for s in dp if s['think_end'] == e),
                   'flip_any': sum(1 for s in dp if s['think_end'] == e and (s['H'] < 0.25) != (s['H_noV'] is not None and s['H_noV'] < 0.25))}
               for e in ('close', 'cap', 'eos')},
}
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open('diag/analyze_fill.json', 'w'), indent=1, ensure_ascii=False)
