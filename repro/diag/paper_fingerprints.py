"""Offline (no GPU) consistency checks of b_main against side facts reported in the VisRef paper (2026-10-11).

These do not identify the authors' choices; they show which readings/configs are consistent with the paper's numbers.
  1. Visual tokens: paper "~1,772 visual tokens" (InternVL3.5-8B, MathVista; VisRef_plain.txt L383-385, ledger X-6).
     1B and 8B share InternViT-300M, so |V| depends only on preprocessing. Tile counts are recomputed from the image
     sizes in the MathVista metadata with InternVL dynamic_preprocess (448 px, 256 tokens/tile); the recomputation
     must reproduce the recorded n_visual of every b_main row (max_num=12, thumbnail) before alternatives are shown.
  2. Answer handling: readings of the recorded b_main rows vs Table 6 1B (46.1 / 48.5 / 52.0; VisRef_plain.txt L1545).
  3. Thinking volume: generated-token ratios vs Table 5 latency ratios (ST 7.1 / TSR 7.7 / VisRef 8.2 s, 8B, H100;
     ledger A-4). Different model size: indicative only.

  cd repro && python3 diag/paper_fingerprints.py   -> diag/paper_fingerprints.json
"""
import glob
import json
import statistics as st

data = json.load(open('../data/raw/mathvista_testmini.json'))
rows = [json.loads(l) for l in open('llm/final/merged.jsonl')]
M = ('st', 'tsr', 'visref_raw_rank')
by = {m: {str(r['pid']): r for r in rows if r['method'] == m} for m in M}
pids = sorted(by['st'], key=int)
PAPER_V = 1772
PAPER_T6 = {'st': 46.1, 'tsr': 48.5, 'visref_raw_rank': 52.0}
PAPER_T5 = {'st': 7.1, 'tsr': 7.7, 'visref_raw_rank': 8.2}


def closest_ratio(ar, ratios, w, h, size):  # InternVL find_closest_aspect_ratio (model card)
    best_diff, best = float('inf'), (1, 1)
    for r in ratios:
        d = abs(ar - r[0] / r[1])
        if d < best_diff:
            best_diff, best = d, r
        elif d == best_diff and w * h > 0.5 * size * size * r[0] * r[1]:
            best = r
    return best


def n_tiles(w, h, max_num=12, min_num=1, size=448, thumbnail=True):
    ratios = sorted({(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
                     if min_num <= i * j <= max_num}, key=lambda x: x[0] * x[1])
    i, j = closest_ratio(w / h, ratios, w, h, size)
    return i * j + (1 if thumbnail and i * j != 1 else 0)


def vis(p, **kw):
    md = data[p]['metadata']
    return 256 * n_tiles(md['img_width'], md['img_height'], **kw)


out = {}
rec = {p: by['st'][p]['n_visual'] for p in pids}
match = sum(vis(p) == rec[p] for p in pids)
assert match == len(pids), f'tile recomputation does not reproduce recorded n_visual ({match}/{len(pids)})'
cfg = {}
for mx in (6, 8, 10, 12):
    for th in (True, False):
        v = [vis(p, max_num=mx, thumbnail=th) for p in pids]
        cfg[f'max_num={mx},thumbnail={th}'] = {'mean': round(st.mean(v), 1), 'median': st.median(v),
                                               'rel_to_paper': round(st.mean(v) / PAPER_V - 1, 3)}
out['visual_tokens'] = {'paper_mean_8B': PAPER_V, 'recorded_mean': round(st.mean(rec.values()), 1),
                        'recompute_matches_recorded': f'{match}/{len(pids)}', 'configs': cfg}

readings = {
    'a_as_scored': lambda r: r['correct'],
    'c_af_rows_wrong': lambda r: 0 if r.get('final_af') else r['correct'],
    'e_forced_answer_rows_wrong': lambda r: 0 if r.get('final_forced_answer') else r['correct'],
    'f_forced_or_af_wrong': lambda r: 0 if (r.get('final_forced_answer') or r.get('final_af')) else r['correct'],
}
ans = {}
for name, f in readings.items():
    acc = {m: round(100 * sum(f(by[m][p]) for p in pids) / len(pids), 1) for m in M}
    ans[name] = {'acc': acc, 'tsr-st': round(acc['tsr'] - acc['st'], 1),
                 'visref-tsr': round(acc['visref_raw_rank'] - acc['tsr'], 1),
                 'abs_gap_to_paper': {m: round(acc[m] - PAPER_T6[m], 1) for m in M}}
out['answer_handling'] = {'paper_table6_1B': PAPER_T6, 'readings': ans}

gen = {m: st.mean(by[m][p]['gen_tokens'] for p in pids) for m in M}
out['thinking_volume'] = {
    'gen_tokens_mean': {m: round(gen[m]) for m in M},
    'gen_ratio_vs_st': {m: round(gen[m] / gen['st'], 2) for m in M},
    'paper_latency_ratio_vs_st_8B': {m: round(PAPER_T5[m] / PAPER_T5['st'], 2) for m in M},
    'k1_share': {m: round(sum(by[m][p]['k'] == 1 for p in pids) / len(pids), 3) for m in M},
    'st_step1_cap_share': round(sum(by['st'][p]['steps'][0]['think_end'] == 'cap' for p in pids) / len(pids), 3)}
json.dump(out, open('diag/paper_fingerprints.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
