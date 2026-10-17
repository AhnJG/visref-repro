"""Format check of 2026-10-17 (EF "동작 점검 계획"): does keeping the image block inside the thinking block when answering
cut the extra </think> in answers? Counts only; correctness is never read.

Rows: kaggle/runs/fmtcheck_main/output/results/results_0.jsonl (runpod/run_imgtag.py --plan format_check), methods
visref_hs_imgtag_in and tsr_answer_prompt, on the 200 questions of diag/format_check_ids.json (first 200 of the seed-0
order). Reference: A6 rows (kaggle/runs/imgtag_main) on the same questions.

Counts per method (defined before the run):
  reemit   -- the final answer contains a </think> the model generated (final_y_start > 0)
  empty    -- nothing after '\\nFinal Answer:' (analyze_imgtag_main.empty_after_prompt, the A6 rule); recorded only
  restated -- 'Question:' in the generated answer (final_raw if </think> was re-emitted, else final_answer); recorded only
Rule (fixed before the run): pass when reemit(in) - reemit(TSR with answer prompt) <= LIMIT = 8, half of A6's excess on
the same 200 questions (visref_imgtag 38 - tsr_answer_prompt 21 = 17; 17 / 2 = 8.5, rounded down).

  cd repro && python3 diag/analyze_format_check.py --rows kaggle/runs/fmtcheck_main/output/results/results_0.jsonl \\
      --a6 kaggle/runs/imgtag_main/output/results/results_0.jsonl --out diag/analyze_format_check.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_imgtag_main import empty_after_prompt  # noqa: E402

IN, TSR = 'visref_hs_imgtag_in', 'tsr_answer_prompt'
METHODS = (IN, TSR)
A6 = ('visref_imgtag', 'tsr_answer_prompt')
COUNTS = ('reemit', 'empty', 'restated')
A6_EXCESS = 17  # reemit(visref_imgtag) - reemit(tsr_answer_prompt) of A6 on the 200 questions, counted before the run
LIMIT = A6_EXCESS // 2
IDS = Path(__file__).resolve().parent / 'format_check_ids.json'


def flags(r):
    raw = r.get('final_raw') if (r.get('final_y_start') or 0) > 0 else r.get('final_answer')
    return {'reemit': (r.get('final_y_start') or 0) > 0, 'empty': empty_after_prompt(r), 'restated': 'Question:' in (raw or '')}


def counts(rows, pids, methods):
    by = {(str(r['pid']), r['method']): r for r in rows if r['method'] in methods}
    missing = [(p, m) for p in pids for m in methods if (p, m) not in by]
    assert not missing, f'{len(missing)} rows missing, e.g. {missing[:3]}'
    return ({m: {c: sum(flags(by[(p, m)])[c] for p in pids) for c in COUNTS} for m in methods},
            {m: sum(by[(p, m)].get('status') != 'ok' for p in pids) for m in methods})


def analyze(rows, pids, a6_rows=None):
    c, errors = counts(rows, pids, METHODS)
    excess = c[IN]['reemit'] - c[TSR]['reemit']
    out = {'n': len(pids), 'errors': errors, 'counts': c, 'reemit_excess': excess, 'limit': LIMIT,
           'pass': excess <= LIMIT, 'rule': f'reemit(in) - reemit(TSR with answer prompt) <= {LIMIT} (half of A6 excess {A6_EXCESS})'}
    if a6_rows is not None:  # reference: A6 on the same questions; its excess must be the constant fixed before the run
        a, _ = counts(a6_rows, pids, A6)
        out['a6_reference'] = a
        assert a[A6[0]]['reemit'] - a[A6[1]]['reemit'] == A6_EXCESS, 'A6 excess differs from the constant'
    return out


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', required=True)
    ap.add_argument('--a6', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    res = analyze(load(a.rows), json.loads(IDS.read_text()), load(a.a6) if a.a6 else None)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'n': res['n'], 'reemit_excess': res['reemit_excess'], 'pass': res['pass']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
