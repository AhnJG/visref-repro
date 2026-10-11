"""Post-hoc exploratory (2026-10-11, not pre-registered): score the saved step-1 stopping-probe answers of b_main.

The probe is `</think>` + "\n\n\\boxed" right after z_1 (TSR) or z_1 + V_1 (VisRef), one answer at T=1 from the probe-only
RNG stream (common per pid across methods, run.py). Both methods share z_1 (fork), so the pair differs only by V_1.
Scored text = "\\boxed" + probe_answer with the current scorer; invalid probes (H = inf) count as wrong.
  cd repro && python3 diag/analyze_step1_probe.py  -> diag/step1_probe_analysis.json
"""
import json
import random
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from visref.scoring import SCORER_VERSION, score

data = json.load(open('../data/raw/mathvista_testmini.json'))
rows = [json.loads(l) for l in open('llm/final/merged.jsonl')]
by = {m: {str(r['pid']): r for r in rows if r['method'] == m} for m in ('st', 'tsr', 'visref_raw_rank')}
S, T, V = by['st'], by['tsr'], by['visref_raw_rank']
pids = sorted(S, key=int)


def pc(step, p):
    return score('\\boxed' + step['probe_answer'], data[p])['correct'] if step.get('probe_valid') else 0


def mcnemar(a, b):
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def boot(d, n=2000, seed=0):
    rng = random.Random(seed)
    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return [round(100 * bs[int(.025 * n)], 2), round(100 * bs[int(.975 * n)], 2)]


def pair(ps, a, b):
    w, l = sum(a[p] > b[p] for p in ps), sum(a[p] < b[p] for p in ps)
    return {'n': len(ps), 'acc_visref': round(100 * sum(a[p] for p in ps) / len(ps), 1),
            'acc_tsr': round(100 * sum(b[p] for p in ps) / len(ps), 1), 'visref_only': w, 'tsr_only': l,
            'p_mcnemar': round(mcnemar(w, l), 4), 'diff_ci95': boot([a[p] - b[p] for p in ps])}


pv = {p: pc(V[p]['steps'][0], p) for p in pids}
pt = {p: pc(T[p]['steps'][0], p) for p in pids}
fv = {p: V[p]['correct'] for p in pids}
ft = {p: T[p]['correct'] for p in pids}
cont = [p for p in pids if T[p]['k'] > 1 or V[p]['k'] > 1]
both = [p for p in pids if T[p]['k'] > 1 and V[p]['k'] > 1]
k1 = [p for p in pids if T[p]['k'] == 1 and V[p]['k'] == 1]
out = {'scorer_version': SCORER_VERSION, 'note': 'post-hoc exploratory; single T=1 sample per probe',
       'valid_probes': {'tsr': sum(bool(T[p]['steps'][0].get('probe_valid')) for p in pids),
                        'visref': sum(bool(V[p]['steps'][0].get('probe_valid')) for p in pids)},
       'identical_probe_strings': sum(T[p]['steps'][0]['probe_answer'] == V[p]['steps'][0]['probe_answer'] for p in pids),
       'probe_all': pair(pids, pv, pt), 'probe_either_continued': pair(cont, pv, pt),
       'probe_both_continued': pair(both, pv, pt), 'probe_both_k1': pair(k1, pv, pt),
       'final_either_continued': pair(cont, fv, ft), 'final_both_continued': pair(both, fv, ft),
       'probe_by_st_step1_end': {e: pair([p for p in pids if S[p]['steps'][0]['think_end'] == e], pv, pt)
                                 for e in ('close', 'cap', 'eos')}}
json.dump(out, open('diag/step1_probe_analysis.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
