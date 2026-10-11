"""Re-check the claim "at a 4096 per-step cap about half of ST first steps are cut, about half of those are loops".
Data: kaggle/runs/{main_s0,full_s1} (step cap 4096). Loop heuristic: zlib compression ratio of the last W chars of the
thinking text < 0.2 (W = 3000 and 4000). Heuristic only; prints counts, no thresholds tuned on outcomes.
Usage: cd repro && python3 diag/check_4096_cap_loops.py"""
import json, glob, zlib

def load(name):
    rows = []
    for f in sorted(glob.glob(f'kaggle/runs/{name}/output/results/results_*.jsonl')):
        for l in open(f):
            try:
                rows.append(json.loads(l))
            except ValueError:
                pass
    return rows

def zr(t, w):
    t = t[-w:].encode()
    return len(zlib.compress(t, 9)) / max(1, len(t))

out = {}
for name in ('main_s0', 'full_s1'):
    rows = load(name)
    st = [r for r in rows if r['method'] == 'st']
    cap = [r for r in st if r['steps'][0]['think_end'] == 'cap']
    d = {'n_st': len(st), 'st_step1_cap': len(cap), 'cap_tokens': sorted({r['steps'][0]['think_tokens'] for r in cap}),
         'cap_correct': sum(r['correct'] for r in cap), 'noncap_correct': sum(r['correct'] for r in st if r not in cap),
         'all_steps_end': {}}
    for r in rows:
        for s in r['steps']:
            d['all_steps_end'][s['think_end']] = d['all_steps_end'].get(s['think_end'], 0) + 1
    for w in (3000, 4000):
        d[f'cap_zlib_lt_0.2_w{w}'] = sum(zr(r['thinking'], w) < 0.2 for r in cap)
        d[f'noncap_len_ge_w_zlib_lt_0.2_w{w}'] = sum(zr(r['thinking'], w) < 0.2 for r in st if r not in cap and len(r['thinking']) >= w)
    out[name] = d
a = {str(r['pid']) for r in load('main_s0') if r['method'] == 'st'}
b = {str(r['pid']) for r in load('full_s1') if r['method'] == 'st'}
out['pid_overlap'] = {'shared': len(a & b), 'union': len(a | b)}
print(json.dumps(out, indent=1))
json.dump(out, open('diag/check_4096_cap_loops.json', 'w'), indent=1)
