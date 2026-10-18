"""Subgroup reanalysis of the three paired runs (EF 2026-10-18 "다음 계획 초안: 묶음별 재분석"), fixed before the results.

Hypothesis generation only: the same 1,000 testmini questions are split again, so a "candidate" is a hypothesis, not a
finding. No GPU.

Runs (each as scored by its own pre-registered analysis, main reading = rules v4 + LLM fallback):
  b_main      visref_raw_rank - tsr             llm/final/merged.jsonl (1,000)
  imgtag_main visref_imgtag   - tsr_answer_prompt  kaggle/runs/imgtag_main + llm/imgtag_main/merged.jsonl (1,000)
  visref_hs   visref_hs       - tsr (b_main)    kaggle/runs/b_main_hs + llm/hs/merged.jsonl (839, complete prefix)
The overall differences are checked against diag/analyze_imgtag_main.json and diag/analyze_hs.json.

Subgroups (16, fixed): question type 2, task 5, image tokens (n_visual) 3 (<=256 / 257-1,792 / >1,792),
b_main first-step end 3 (close / cap / eos, from the b_main tsr row), b_main TSR first-step H terciles 3.
Every subgroup is defined on the b_main rows, so it is the same set of questions in all three runs.

Statistic: pooled = mean over the three runs of (VisRef - TSR) in the subgroup (each run on its own questions).
Bootstrap: 10,000 resamples of the 1,000 questions (seed 0), one resample shared by the three runs and all subgroups
(questions are resampled jointly, so the run-to-run correlation is kept). Two-sided bootstrap p = 2 * min(P*(<=0), P*(>=0)),
Holm over the 16 subgroups. Candidate rule: n >= 100 in all three runs, all three run differences > 0, Holm p < 0.05
(equivalently the bootstrap CI at the Holm level excludes 0 from below).

Usage (from repro/): python3 diag/subgroup_reanalysis.py [--out diag/subgroup_reanalysis.json]"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize as S  # noqa: E402
import analyze_hs as AH  # noqa: E402
import analyze_imgtag_main as AI  # noqa: E402

N_BOOT, SEED, ALPHA, MIN_N = 10_000, 0, 0.05, 100
TASKS = ['figure question answering', 'geometry problem solving', 'math word problem', 'visual question answering',
         'textbook question answering']


def diffs_bmain(bmain):
    pids = sorted({p for p, _ in bmain}, key=int)
    return {p: int(bmain[(p, 'visref_raw_rank')]['correct']) - int(bmain[(p, 'tsr')]['correct']) for p in pids}


def diffs_imgtag(data):
    rows, order = AI.load_runs(['kaggle/runs/imgtag_main/output/results'], AI.METHODS)
    order = [str(x) for x in order]
    rule = {k: AI.rule_scored(r, data[k[0]]) for k, r in rows.items()}
    pids = S.complete_prefix(rule, AI.METHODS, order, 1)
    llm = {(str(r['pid']), r['method']): r for r in map(json.loads, open('llm/imgtag_main/merged.jsonl'))
           if r['method'] in AI.METHODS}
    main = {}
    for p in pids:
        for m in AI.MAIN:
            r = rule[(p, m)]
            main[(p, m)] = r if r['empty_after_prompt'] else llm[(p, m)]
    return {p: int(main[(p, AI.MAIN[0])]['correct']) - int(main[(p, AI.MAIN[1])]['correct']) for p in pids}


def diffs_hs(bmain, data):
    hs_dir = 'kaggle/runs/b_main_hs/output/results'
    hs = AH.load_hs(hs_dir, 'visref_hs')
    order = json.load(open(Path(hs_dir) / 'ids.json'))
    rule = {k: AH.rescored(r, data[k[0]]) for k, r in {**bmain, **hs}.items()}
    pids = S.complete_prefix(rule, AH.BMAIN + ('visref_hs',), order, 2)
    llm = {(r['pid'], r['method']): r for r in map(json.loads, open('llm/hs/merged.jsonl')) if r['method'] == 'visref_hs'}
    return {p: int(llm[(p, 'visref_hs')]['correct']) - int(bmain[(p, 'tsr')]['correct']) for p in pids}


def subgroups(bmain, data):
    tsr = {p: r for (p, m), r in bmain.items() if m == 'tsr'}
    H = {p: r['steps'][0]['H'] for p, r in tsr.items()}
    q1, q2 = statistics.quantiles(list(H.values()), n=3)
    g = {}
    g['type: multi_choice'] = {p for p in tsr if data[p]['question_type'] == 'multi_choice'}
    g['type: free_form'] = {p for p in tsr if data[p]['question_type'] == 'free_form'}
    for t in TASKS:
        g[f'task: {t}'] = {p for p in tsr if data[p]['metadata']['task'] == t}
    nv = {p: r['n_visual'] for p, r in tsr.items()}
    g['image tokens: <=256'] = {p for p in tsr if nv[p] <= 256}
    g['image tokens: 257-1792'] = {p for p in tsr if 256 < nv[p] <= 1792}
    g['image tokens: >1792'] = {p for p in tsr if nv[p] > 1792}
    for e in ('close', 'cap', 'eos'):
        g[f'step-1 end: {e}'] = {p for p, r in tsr.items() if r['steps'][0]['think_end'] == e}
    g['TSR step-1 H: low'] = {p for p in tsr if H[p] <= q1}
    g['TSR step-1 H: mid'] = {p for p in tsr if q1 < H[p] <= q2}
    g['TSR step-1 H: high'] = {p for p in tsr if H[p] > q2}
    assert len(g) == 16
    for fam in ('type', 'task', 'image tokens', 'step-1 end', 'TSR step-1 H'):
        parts = [s for k, s in g.items() if k.startswith(fam + ':')]
        assert sum(map(len, parts)) == len(tsr) and len(set().union(*parts)) == len(tsr), fam
    return g, [round(q1, 4), round(q2, 4)]


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj, {i: rank for rank, i in enumerate(order)}


def analyze(runs, groups):
    pids = sorted(set().union(*groups.values()), key=int)
    idx = {p: i for i, p in enumerate(pids)}
    n = len(pids)
    D, M = {}, {}
    for name, d in runs.items():
        D[name] = np.zeros(n)
        M[name] = np.zeros(n, dtype=bool)
        for p, v in d.items():
            D[name][idx[p]], M[name][idx[p]] = v, True
    rng = np.random.default_rng(SEED)
    W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(N_BOOT)]).astype(np.float64)
    res, ps = {}, []
    for gname, s in groups.items():
        gm = np.array([p in s for p in pids])
        per, boot = {}, []
        for name in runs:
            mk = gm & M[name]
            per[name] = {'n': int(mk.sum()), 'diff_pp': round(100 * D[name][mk].mean(), 2),
                         'visref_only': int((D[name][mk] == 1).sum()), 'tsr_only': int((D[name][mk] == -1).sum())}
            cnt = W @ mk
            boot.append(np.where(cnt > 0, (W @ (D[name] * mk)) / np.maximum(cnt, 1), np.nan))
        pooled = statistics.mean(v['diff_pp'] for v in per.values()) / 100
        b = np.nanmean(np.stack(boot), axis=0)
        p = min(1.0, 2 * min(np.mean(b <= 0), np.mean(b >= 0)))
        ps.append(p)
        res[gname] = {'runs': per, 'pooled_pp': round(100 * pooled, 2),
                      'ci95_pp': [round(100 * x, 2) for x in np.quantile(b, [.025, .975])],
                      'p_boot': round(float(p), 4), '_boot': b}
    adj, rank = holm(ps)
    for i, (gname, r) in enumerate(res.items()):
        b = r.pop('_boot')
        a = ALPHA / (len(ps) - rank[i])  # Holm level for this subgroup's rank
        r['ci_holm_level_pp'] = [round(100 * x, 2) for x in np.quantile(b, [a / 2, 1 - a / 2])]
        r['p_holm'] = round(adj[i], 4)
        r['n_min_ok'] = all(v['n'] >= MIN_N for v in r['runs'].values())
        r['all_positive'] = all(v['diff_pp'] > 0 for v in r['runs'].values())
        r['candidate'] = r['n_min_ok'] and r['all_positive'] and adj[i] < ALPHA and r['pooled_pp'] > 0
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../data/raw/mathvista_testmini.json')
    ap.add_argument('--out')
    a = ap.parse_args(argv)
    data = json.load(open(a.data))
    bmain = {(r['pid'], r['method']): r for r in map(json.loads, open('llm/final/merged.jsonl')) if r['method'] in AH.BMAIN}
    runs = {'b_main': diffs_bmain(bmain), 'imgtag_main': diffs_imgtag(data), 'visref_hs': diffs_hs(bmain, data)}
    # cross-check against the recorded pre-registered analyses
    ri = json.load(open('diag/analyze_imgtag_main.json'))['main']['primary']
    rh = json.load(open('diag/analyze_hs.json'))['main']['primary']
    for name, ref in (('imgtag_main', ri), ('visref_hs', rh)):
        d = runs[name]
        assert len(d) == ref['n'] and sum(d.values()) * 100 / len(d) - ref['diff_pp'] < 0.006, (name, len(d), sum(d.values()))
    groups, h_cuts = subgroups(bmain, data)
    out = {'overall': {k: {'n': len(d), 'diff_pp': round(100 * sum(d.values()) / len(d), 2)} for k, d in runs.items()},
           'h_tercile_cuts': h_cuts, 'n_boot': N_BOOT, 'seed': SEED,
           'subgroups': analyze(runs, groups)}
    out['candidates'] = [g for g, r in out['subgroups'].items() if r['candidate']]
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
