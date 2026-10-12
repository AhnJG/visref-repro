# Read-only recomputation used by the independent review of 2026-10-13 (review.md). Run from repro/: python3 -I audit/independent_review_20261013/recompute.py
"""Read-only recomputation used for the review (run from repro/: python3 -I <this file>).
Writes nothing into the repository."""
import collections, glob, json, math, os, statistics

rows = {}
for line in open('llm/final/merged.jsonl'):
    r = json.loads(line); rows[(r['pid'], r['method'])] = r
pids = sorted({p for p, m in rows}, key=int)


def mcnemar(a, b):
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


print('main acc', {m: 100 * sum(rows[(p, m)]['correct'] for p in pids) / 1000 for m in ('st', 'tsr', 'visref_raw_rank')})
for a, b in [('visref_raw_rank', 'st'), ('visref_raw_rank', 'tsr'), ('tsr', 'st')]:
    d = [rows[(p, a)]['correct'] - rows[(p, b)]['correct'] for p in pids]
    ao, bo = sum(x == 1 for x in d), sum(x == -1 for x in d)
    print(a, b, ao, bo, round(mcnemar(ao, bo), 4))

# hs run: step-1 text identity with the main run's shared step 1
hs = {}
for f in sorted(glob.glob('kaggle/runs/b_main_hs/output/results/results_*.jsonl')):
    for line in open(f):
        if line.strip():
            r = json.loads(line); hs[r['pid']] = r
hs_m = {json.loads(l)['pid']: json.loads(l) for l in open('llm/hs/merged.jsonl')}
order = json.load(open('kaggle/runs/b_main_hs/output/results/ids.json'))
keep = []
for si in range(2):
    for pid in order[si::2]:
        if pid not in hs:
            break
        keep.append(pid)
s1 = lambda r: r['thinking'].split('<VISREF')[0].split('</think>')[0]
ident = [p for p in keep if s1(hs[p]) == s1(rows[(p, 'st')])]
diff = [p for p in keep if p not in set(ident)]
print('hs prefix', len(keep), 'step-1 text identical to main run', len(ident))
pref = [len(os.path.commonprefix([s1(hs[p]), s1(rows[(p, 'st')])])) for p in diff]
print('common prefix chars (different ones) median', statistics.median(pref))
for name, ps in (('identical', ident), ('different', diff)):
    d = [hs_m[p]['correct'] - rows[(p, 'tsr')]['correct'] for p in ps]
    print(name, len(ps), 'hs-TSR only-first/only-second', sum(x == 1 for x in d), sum(x == -1 for x in d),
          'discordance %', round(100 * (sum(x != 0 for x in d)) / len(ps), 1))
d = [hs_m[p]['correct'] - rows[(p, 'tsr')]['correct'] for p in keep]
disc = sum(x != 0 for x in d) / len(d)
se = 100 * math.sqrt(disc / len(d))
print('hs-TSR discordance', round(100 * disc, 1), 'SE pp', round(se, 2), 'MDE80 pp', round(2.8 * se, 2),
      'power for +3.5', round(1 - 0.5 * (1 + math.erf((1.96 - 3.5 / se) / math.sqrt(2))), 2))

# hs mechanism
steps = [s for p in keep for s in hs[p]['steps'] if s.get('m')]
long_ = [s for s in steps if s['think_tokens'] >= s['m']]
below = [s for s in long_ if s['n_inject'] < s['m']]
print('steps', len(steps), 'len>=m', len(long_), 'eq m', sum(s['n_inject'] == s['m'] for s in long_),
      'below m: m values', collections.Counter(s['m'] for s in below))
m998 = [s for s in steps if s['m'] == 998]
print('m=998 steps', len(m998), 'reached m', sum(s['n_inject'] == s['m'] for s in m998),
      'shorter than m', sum(s['think_tokens'] < 998 for s in m998))

# fill run step-1 identity
fill = {}
for f in sorted(glob.glob('kaggle/runs/b_main_fill/output/results/results_*.jsonl')):
    for line in open(f):
        if line.strip():
            r = json.loads(line); fill[r['pid']] = r
print('fill rows', len(fill), 'step-1 text identical', sum(s1(r) == s1(rows[(p, 'st')]) for p, r in fill.items()))

# TSR vs VisRef final-answer discordance outside the 309 step1_branch questions
ids = json.load(open('diag/step1_branch_ids.json'))
ids = set(str(x) for x in (ids['ids'] if isinstance(ids, dict) else ids))
both = [p for p in pids if rows[(p, 'tsr')]['k'] == 1 and rows[(p, 'visref_raw_rank')]['k'] == 1]
d = [rows[(p, 'visref_raw_rank')]['correct'] - rows[(p, 'tsr')]['correct'] for p in both]
print('both k=1', len(both), 'in 309?', sum(p in ids for p in both), 'VisRef-only', sum(x == 1 for x in d), 'TSR-only', sum(x == -1 for x in d))
