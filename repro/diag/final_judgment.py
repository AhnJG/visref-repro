"""Pre-registered final judgment for b_main (EF 2026-10-10 "재현 판정 기준" and the analysis plan).
Reading (a) = scorer v3 + LLM fallback extraction (guards 1-2), from llm/final/merged.jsonl.
Paired difference + bootstrap 95% CI (summarize.paired: 2,000 resamples, seed 0), McNemar exact p, Holm over the
two main comparisons (VisRef-ST, VisRef-TSR); TSR-ST is supplementary. Judgment by the CI rule only.
Sensitivity (reported, not judged): rule-only v3, (b) af rows re-scored before the af prompt, (c) af rows wrong.
Scorer (2026-10-11): view (b) re-scores with the current scorer (v4: an empty multiple-choice extraction is wrong, the
official MathVista --ignore_empty_extractions mode); the v3 value of (b) is reported next to it ('b_af_stripped_v3').
Reading (a) keeps the recorded per-row correctness; 'v4_changes_in_a' counts rows v4 would change there.
Usage: cd repro && python3 diag/final_judgment.py"""
import json, sys
sys.path.insert(0, '.')
import summarize as S
from report import mcnemar_exact, AF_PROMPT
from visref.scoring import SCORER_V3, SCORER_VERSION, score

PAPER = {'st': 46.1, 'tsr': 48.5, 'visref_raw_rank': 52.0}
PAIRS = [('visref_raw_rank', 'st', 'main'), ('visref_raw_rank', 'tsr', 'main'), ('tsr', 'st', 'supplementary')]
data = json.load(open('../data/raw/mathvista_testmini.json'))
rows = {(r['pid'], r['method']): r for r in map(json.loads, open('llm/final/merged.jsonl'))}
pids = sorted({p for p, m in rows if all((p, mm) in rows for mm in PAPER)}, key=int)
assert len(pids) == 1000


def judge(ci, paper):
    lo, hi = ci
    has0, hasp = lo <= 0 <= hi, lo <= paper <= hi
    if not has0 and hasp:
        return '재현됨'
    if not hasp:
        return '효과 크기 재현 안 됨' + (' (방향은 같으나 크기 다름)' if not has0 and (lo > 0) == (paper > 0) else '')
    return '판정 불가'


def view(fn):
    return {k: dict(r, correct=fn(r)) for k, r in rows.items()}


def af_b(r, version=SCORER_VERSION):
    fa = r['final_answer']; i = fa.rfind(AF_PROMPT) if r.get('final_af') else -1
    return r['correct'] if i < 0 else score(fa[:i], data[str(r['pid'])], version=version)['correct']


def v4_changes(r):
    """1 if v4 would change the recorded correctness of this row: a correct MC row whose scored extraction is empty."""
    q = data[str(r['pid'])]
    ext = r.get('llm_extraction') if r.get('llm_used') else r.get('extracted')
    return int(q['question_type'] == 'multi_choice' and r['correct'] == 1 and (ext is None or not str(ext).strip()))


views = {'a_v3_llm': rows,
         'rule_only_v3': view(lambda r: r.get('correct_rule', r['correct'])),
         'b_af_stripped': view(af_b),
         'b_af_stripped_v3': view(lambda r: af_b(r, SCORER_V3)),
         'c_af_wrong': view(lambda r: 0 if r.get('final_af') else r['correct'])}
out = {'n': len(pids), 'scorer_version_rescored_views': SCORER_VERSION,
       'v4_changes_in_a': sum(v4_changes(rows[(p, m)]) for p in pids for m in PAPER),
       'acc': {}, 'pairs': {}, 'sensitivity': {}}
for m in PAPER:
    out['acc'][m] = round(100 * sum(rows[(p, m)]['correct'] for p in pids) / len(pids), 1)
ps = {}
for a, b, kind in PAIRS:
    pr = S.paired(rows, a, b, pids)
    p = mcnemar_exact(pr['a_only'], pr['b_only'])
    paper = round(PAPER[a] - PAPER[b], 1)
    ps[(a, b)] = p
    out['pairs'][f'{a}-{b}'] = {'kind': kind, 'diff_pp': round(pr['diff_pp'], 2),
                                'ci95_pp': [round(x, 2) for x in pr['ci95_pp']], 'a_only': pr['a_only'],
                                'b_only': pr['b_only'], 'p_mcnemar': round(p, 4), 'paper_pp': paper,
                                'judgment': judge(pr['ci95_pp'], paper)}
main = sorted([(ps[(a, b)], f'{a}-{b}') for a, b, k in PAIRS if k == 'main'])
for i, (p, name) in enumerate(main):  # Holm, 2 tests
    adj = min(1.0, max(p * (2 - j) for j, (pp, _) in enumerate(main[:i + 1]) for p in [pp]))
    out['pairs'][name]['p_holm'] = round(adj, 4)
for v, rr in views.items():
    out['sensitivity'][v] = {'acc': {m: round(100 * sum(rr[(p, m)]['correct'] for p in pids) / len(pids), 1) for m in PAPER},
                             'pairs': {f'{a}-{b}': (lambda q: [round(q['diff_pp'], 2), [round(x, 2) for x in q['ci95_pp']]])(S.paired(rr, a, b, pids))
                                       for a, b, _ in PAIRS}}
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open('diag/final_judgment.json', 'w'), indent=1, ensure_ascii=False)
