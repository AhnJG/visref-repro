"""Re-apply the current rule scorer to saved final answers (no inference).

Writes NEW files only: rows.jsonl (per-row old/new) and summary.json (per-method accuracy on
the common pid set). Inputs are never modified.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visref.scoring import SCORER_VERSION, score  # noqa: E402

METHODS = ('st', 'tsr', 'visref_raw_rank')


def _failed(e):
    return e is None or str(e).strip() == ''


def side_by_side(a, rows, rows_p, sum_p):
    by = {}
    for r in rows:
        by.setdefault(r['method'], {}).setdefault(r['pid'], r)
    common = sorted(set.intersection(*[set(by[m]) for m in a.methods]), key=int)
    acc = {}
    for m in a.methods:
        acc[m] = {v: [by[m][p][f'{v}_correct'] for p in common] for v in ('v1', 'v2', 'v3')}
    summ = {'scorer_version': SCORER_VERSION, 'n_common_pids': len(common), 'methods': {}, 'paired_diff_pp': {}}
    for m in a.methods:
        summ['methods'][m] = {'n': len(common), **{f'{v}_correct': sum(acc[m][v]) for v in ('v1', 'v2', 'v3')},
                              **{f'{v}_accuracy_pct': round(100 * sum(acc[m][v]) / len(common), 3) for v in ('v1', 'v2', 'v3')},
                              'n_extract_failed_v2': sum(_failed(by[m][p]['v2_extracted']) for p in common),
                              'n_extract_failed_v3': sum(_failed(by[m][p]['v3_extracted']) for p in common)}
    for x, y in (('visref_raw_rank', 'st'), ('tsr', 'st'), ('visref_raw_rank', 'tsr')):
        if x in acc and y in acc:
            d = {}
            for v in ('v2', 'v3'):
                a_only = sum(i > j for i, j in zip(acc[x][v], acc[y][v]))
                b_only = sum(i < j for i, j in zip(acc[x][v], acc[y][v]))
                d[v] = {'diff_pp': round(100 * (sum(acc[x][v]) - sum(acc[y][v])) / len(common), 3), 'a_only': a_only, 'b_only': b_only}
            summ['paired_diff_pp'][f'{x}-{y}'] = d
    keep = ('pid', 'method', 'session', 'v2_correct', 'v3_correct', 'v2_extracted', 'v3_extracted', 'v3_rule', '_tail')
    ch = lambda r: r['v2_correct'] != r['v3_correct'] or r['v2_extracted'] != r['v3_extracted']
    summ['changed_v3_vs_v2_common'] = [{k: r[k] for k in keep} for m in a.methods for p in common for r in [by[m][p]] if ch(r)]
    summ['changed_v3_vs_v2_other_methods'] = [{k: r[k] for k in keep} for r in rows if r['method'] not in a.methods and ch(r)]
    summ['n_rows_other_methods'] = {m: len(d) for m, d in by.items() if m not in a.methods}
    with open(rows_p, 'x') as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if not k.startswith('_')}, ensure_ascii=False) + '\n')
    with open(sum_p, 'x') as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if not k.startswith('changed')}, ensure_ascii=False, indent=1))
    for k in ('changed_v3_vs_v2_common', 'changed_v3_vs_v2_other_methods'):
        print(k, len(summ[k]))
        for x in summ[k]:
            print(json.dumps(x, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True, help='results_*.jsonl paths (session = b_main_sN in path, else file stem)')
    ap.add_argument('--data', required=True, help='mathvista_testmini.json (dict keyed by pid)')
    ap.add_argument('--out', required=True, help='output directory (created; existing files are not overwritten)')
    ap.add_argument('--methods', nargs='+', default=list(METHODS))
    ap.add_argument('--v2-rows', default='', help='earlier rows.jsonl (scorer v2); switches to v1/v2/v3 side-by-side output')
    a = ap.parse_args()
    data = json.load(open(a.data))
    os.makedirs(a.out, exist_ok=True)
    rows_p, sum_p = os.path.join(a.out, 'rows.jsonl'), os.path.join(a.out, 'summary.json')
    for p in (rows_p, sum_p):
        if os.path.exists(p):
            sys.exit(f'refusing to overwrite {p}')
    v2 = {}
    if a.v2_rows:
        for line in open(a.v2_rows):
            x = json.loads(line)
            v2[(str(x['pid']), x['method'], x['session'])] = x
    rows = []
    for path in a.inputs:
        m = re.search(r'_(s\d+)(?:/|$)', path)
        session = m[1] if m else os.path.splitext(os.path.basename(path))[0]
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            prob = data[str(r['pid'])]
            n = score(r['final_answer'], prob)
            if a.v2_rows:
                o = v2[(str(r['pid']), r['method'], session)]
                rows.append({'pid': r['pid'], 'method': r['method'], 'session': session,
                             'v1_correct': r['correct'], 'v2_correct': o['new_correct'], 'v2_extracted': o['new_extracted'],
                             'v3_correct': n['correct'], 'v3_extracted': n['extracted'], 'v3_rule': n['rule'],
                             'scorer_version': SCORER_VERSION, '_tail': r['final_answer'][-200:]})
                continue
            rows.append({'pid': r['pid'], 'method': r['method'], 'session': session,
                         'old_correct': r['correct'], 'new_correct': n['correct'],
                         'old_extracted': r['extracted'], 'new_extracted': n['extracted'],
                         'old_rule': r['rule'], 'new_rule': n['rule'],
                         'old_extract_failed': r['extract_failed'], 'new_extract_failed': n['extract_failed'],
                         'scorer_version': SCORER_VERSION,
                         '_tail': r['final_answer'][-200:]})
    if a.v2_rows:
        return side_by_side(a, rows, rows_p, sum_p)
    by = {}
    for r in rows:
        by.setdefault(r['method'], {}).setdefault(r['pid'], []).append(r)
    dup = {m: sum(len(v) > 1 for v in d.values()) for m, d in by.items()}
    common = set.intersection(*[set(by.get(m, {})) for m in a.methods])
    summ = {'scorer_version': SCORER_VERSION, 'n_common_pids': len(common), 'duplicate_pids_per_method': dup, 'methods': {}}
    changed = []
    for m in a.methods:
        rs = [x for p in sorted(common) for x in by[m][p]]
        n = len(rs)
        oc, nc = sum(x['old_correct'] for x in rs), sum(x['new_correct'] for x in rs)
        summ['methods'][m] = {'n': n, 'old_correct': oc, 'new_correct': nc,
                              'old_accuracy_pct': round(100 * oc / n, 3), 'new_accuracy_pct': round(100 * nc / n, 3),
                              'n_extract_failed_old': sum(bool(x['old_extract_failed']) for x in rs),
                              'n_extract_failed_new': sum(bool(x['new_extract_failed']) for x in rs)}
        changed += [x for x in rs if x['old_correct'] != x['new_correct'] or x['old_extracted'] != x['new_extracted']]
    summ['changed_rows'] = [{k: v for k, v in x.items() if not k.startswith('old_extract_failed') and not k.startswith('new_extract_failed')} for x in changed]
    with open(rows_p, 'x') as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if not k.startswith('_') and 'extract_failed' not in k}, ensure_ascii=False) + '\n')
    with open(sum_p, 'x') as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != 'changed_rows'}, ensure_ascii=False, indent=1))
    print('changed rows:', len(changed))
    for x in changed:
        print(json.dumps(x, ensure_ascii=False))


if __name__ == '__main__':
    main()
