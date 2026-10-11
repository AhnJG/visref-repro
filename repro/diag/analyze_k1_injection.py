"""Why does VisRef's injection at k=1 (V_1 placed right before </think> and the final answer) not change accuracy?
Uses b_main s1-s8 (complete-prefix not required: pairs are per pid). CRN: ST, TSR and VisRef share step-1 thinking and
the main RNG state at the final answer, so for k=1 jobs the only difference VisRef vs ST is the injected V_1.
Usage: cd repro && python3 diag/analyze_k1_injection.py [n_sessions]"""
import json, glob, sys, collections, statistics as st
sys.path.insert(0, '.')
import report

N = int(sys.argv[1]) if len(sys.argv) > 1 else 9
data = json.load(open('../data/raw/mathvista_testmini.json'))
rows = {}
for s in range(1, N + 1):
    for f in glob.glob(f'kaggle/runs/b_main_s{s}/output/results/results_*.jsonl'):
        for l in open(f):
            if l.strip():
                r = json.loads(l); rows[(str(r['pid']), r['method'])] = r
report.rescore_rows(rows, data)
pids = sorted({p for p, m in rows if all((p, mm) in rows for mm in ('st', 'tsr', 'visref_raw_rank'))}, key=int)

def q(xs):
    xs = sorted(xs)
    return [xs[int(f * (len(xs) - 1))] for f in (.1, .25, .5, .75, .9)] if xs else None

out = {'n_pairs': len(pids)}
# CRN sanity: TSR k=1 vs ST
t1 = [p for p in pids if rows[(p, 'tsr')]['k'] == 1]
out['tsr_k1'] = {'n': len(t1), 'same_final_text': sum(rows[(p, 'tsr')]['final_answer'] == rows[(p, 'st')]['final_answer'] for p in t1)}
v1 = [p for p in pids if rows[(p, 'visref_raw_rank')]['k'] == 1]
res = {}
for grp in ('all', 'close', 'cap', 'eos'):
    ps = [p for p in v1 if grp == 'all' or rows[(p, 'st')]['steps'][0]['think_end'] == grp]
    V = [rows[(p, 'visref_raw_rank')] for p in ps]; S = [rows[(p, 'st')] for p in ps]
    same_text = sum(v['final_answer'] == s['final_answer'] for v, s in zip(V, S))
    same_pred = sum(str(v['prediction']) == str(s['prediction']) for v, s in zip(V, S))
    res[grp] = {
        'n': len(ps), 'st_correct': sum(s['correct'] for s in S), 'visref_correct': sum(v['correct'] for v in V),
        'visref_only': sum(v['correct'] and not s['correct'] for v, s in zip(V, S)),
        'st_only': sum(s['correct'] and not v['correct'] for v, s in zip(V, S)),
        'same_final_text': same_text, 'same_prediction': same_pred,
        'final_len_q_st': q([s['final_len'] for s in S]), 'final_len_q_visref': q([v['final_len'] for v in V]),
        'reemit_think_end_st': sum(bool(s.get('final_y_start')) for s in S),
        'reemit_think_end_visref': sum(bool(v.get('final_y_start')) for v in V),
        'answer_cap_st': sum(s['final_finish'] == 'cap' for s in S),
        'answer_cap_visref': sum(v['final_finish'] == 'cap' for v in V),
        'n_inject_q': q([v['steps'][0].get('n_inject', 0) for v in V]),
        'inject_over_m_q': q([round(v['steps'][0].get('n_inject', 0) / v['steps'][0]['m'], 3) for v in V if v['steps'][0].get('m')]),
        'n_visual_q': q([v['n_visual'] for v in V]),
    }
    # among pairs whose prediction differs: who is right
    diff = [(v, s) for v, s in zip(V, S) if str(v['prediction']) != str(s['prediction'])]
    res[grp]['pred_differs'] = {'n': len(diff), 'visref_right': sum(v['correct'] for v, s in diff),
                                'st_right': sum(s['correct'] for v, s in diff), 'neither': sum(not v['correct'] and not s['correct'] for v, s in diff)}
out['visref_k1'] = res
# by question type, k=1 all
bt = collections.defaultdict(lambda: [0, 0, 0, 0])
for p in v1:
    t = data[p]['question_type'] + '/' + data[p]['answer_type']
    v, s = rows[(p, 'visref_raw_rank')], rows[(p, 'st')]
    bt[t][0] += 1; bt[t][1] += s['correct']; bt[t][2] += v['correct']; bt[t][3] += str(v['prediction']) != str(s['prediction'])
out['visref_k1_by_type'] = {k: dict(zip(('n', 'st', 'visref', 'pred_differs'), v)) for k, v in sorted(bt.items())}
print(json.dumps(out, indent=1))
json.dump(out, open('diag/analyze_k1_injection.json', 'w'), indent=1)
