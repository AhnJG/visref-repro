"""Pre-registered analysis of diag_st_plain (EXPERIMENT_FLOW 2026-10-10, fixed before results).

diag = ST with --forced-prompt= (rule 1 off: a capped thinking step gets only '</think>'), rerun on 150 pids whose
b_main ST first step hit the cap. Paired against the b_main ST row of the same pid. Three readings, both sides,
scorer = current visref/scoring.py (v3 when this was run, v4 since 2026-10-11):
  (a) as scored                      -> decision reading
  (b) af rows re-scored on the text before the af prompt (rule 2 removed too, approx.)
  (c) af rows counted wrong          -> lower bound only
Overall ST estimate (per reading) = (b_main ST correct on the non-capped pids of s1-s4 + n_cap * diag acc) / n.

  python3 diag/analyze_st_plain.py DIAG_RESULTS.jsonl... --main MAIN_RESULTS.jsonl... --order IDS.json \
      --data ../data/raw/mathvista_testmini.json --ids diag/st_plain_ids.json --out diag/st_plain_analysis.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import summarize as S  # noqa: E402
from report import AF_PROMPT, mcnemar_exact  # noqa: E402
from visref.scoring import SCORER_VERSION, score  # noqa: E402

READINGS = ('a', 'b', 'c')


def readings(r, prob):
    a = score(r['final_answer'], prob)['correct']
    fa = r['final_answer']
    i = fa.rfind(AF_PROMPT) if r.get('final_af') else -1
    if i < 0:
        return {'a': a, 'b': a, 'c': a, 'af': 0}
    return {'a': a, 'b': score(fa[:i], prob)['correct'], 'c': 0, 'af': 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('diag', nargs='+')
    ap.add_argument('--main', nargs='+', required=True, help='b_main s1-s4 results_*.jsonl (the 392-pid base)')
    ap.add_argument('--order', required=True)
    ap.add_argument('--shards', type=int, default=2)
    ap.add_argument('--data', required=True)
    ap.add_argument('--ids', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    data = json.loads(Path(a.data).read_text())
    ids = set(json.loads(Path(a.ids).read_text()))
    main_rows = S.load(a.main, ['st', 'tsr', 'visref_raw_rank'])
    base = S.complete_prefix(main_rows, ('st', 'tsr', 'visref_raw_rank'), json.loads(Path(a.order).read_text()), a.shards)
    st = {p: main_rows[(p, 'st')] for p in base}
    cap = [p for p in st if st[p]['steps'][0]['think_end'] == 'cap']
    rest = [p for p in st if p not in set(cap)]
    diag = {}
    for path in a.diag:
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            if r['method'] == 'st':
                diag[r['pid']] = r
    errors = [p for p, r in diag.items() if r.get('status') == 'error']
    done = sorted((p for p in diag if p not in errors), key=int)
    assert set(done) <= ids, 'diag pids outside st_plain_ids'
    assert ids <= set(cap), 'st_plain_ids not all capped in b_main ST'
    R = {p: {'diag': readings(diag[p], data[str(p)]), 'main': readings(st[p], data[str(p)])} for p in done}
    rest_r = [readings(st[p], data[str(p)]) for p in rest]
    n = max(1, len(done))
    out = {'scorer_version': SCORER_VERSION, 'n_ids': len(ids), 'n_done': len(done), 'n_error': len(errors),
           'n_base': len(base), 'n_cap_base': len(cap),
           'diag_step1_end': {e: sum(diag[p]['steps'][0]['think_end'] == e for p in done) for e in ('cap', 'close', 'eos')},
           'af_rows': {s: sum(R[p][s]['af'] for p in done) for s in ('diag', 'main')},
           'readings': {}}
    for k in READINGS:
        d = [R[p]['diag'][k] for p in done]
        m = [R[p]['main'][k] for p in done]
        b = sum(x > y for x, y in zip(d, m))
        c = sum(x < y for x, y in zip(d, m))
        acc_d, acc_m = 100 * sum(d) / n, 100 * sum(m) / n
        rest_k = sum(x[k] for x in rest_r)
        out['readings'][k] = {
            'diag_acc': acc_d, 'main_acc': acc_m, 'diff_pp': acc_d - acc_m, 'diag_only': b, 'main_only': c,
            'p_mcnemar': mcnemar_exact(b, c),
            'af_acc_diag': [sum(R[p]['diag'][k] for p in done if R[p]['diag']['af']), out['af_rows']['diag']],
            'af_acc_main': [sum(R[p]['main'][k] for p in done if R[p]['main']['af']), out['af_rows']['main']],
            'st_estimate_plain': 100 * (rest_k + len(cap) * acc_d / 100) / len(base),
            'st_estimate_forced': 100 * (rest_k + len(cap) * acc_m / 100) / len(base),
        }
    ra = out['readings']['a']
    out['criterion_a_drop_ge_10pp'] = ra['diff_pp'] <= -10
    out['paper_st'] = 46.1  # VisRef App. Table 6, 1B ST (SPEC_LEDGER)
    if os.path.exists(a.out):
        sys.exit(f'refusing to overwrite {a.out}')
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
