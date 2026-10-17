"""Check 1 of the 2026-10-17 format check (EF "동작 점검 계획"): answer-format counts per method, no correctness.

Rows: kaggle/runs/fmtcheck_main/output/results/results_0.jsonl (runpod/run_imgtag.py --plan format_check), methods
visref_hs_imgtag_in (block inside the thinking block when answering), visref_hs_imgtag_out (A6 format, same selection)
and tsr_answer_prompt, on the 100 questions of diag/format_check_ids.json.

Counts per method (defined before the run):
  reemit   -- the final answer contains a </think> the model generated (final_y_start > 0)
  empty    -- nothing after '\\nFinal Answer:' (analyze_imgtag_main.empty_after_prompt, the A6 rule)
  restated -- 'Question:' appears in the generated answer (final_raw if the model re-emitted </think>, else final_answer)
Rule (fixed before the run): the format passes when for each count, in <= max(ceil(1.5 * T), T + 3), T = the count of
tsr_answer_prompt; the '_out' counts are reported only. Correctness is never read.

  cd repro && python3 diag/analyze_format_check.py --rows kaggle/runs/fmtcheck_main/output/results/results_0.jsonl \\
      --out diag/analyze_format_check.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_imgtag_main import empty_after_prompt  # noqa: E402

IN, OUT, TSR = 'visref_hs_imgtag_in', 'visref_hs_imgtag_out', 'tsr_answer_prompt'
METHODS = (IN, OUT, TSR)
COUNTS = ('reemit', 'empty', 'restated')
IDS = Path(__file__).resolve().parent / 'format_check_ids.json'


def flags(r):
    raw = r.get('final_raw') if (r.get('final_y_start') or 0) > 0 else r.get('final_answer')
    return {'reemit': (r.get('final_y_start') or 0) > 0, 'empty': empty_after_prompt(r), 'restated': 'Question:' in (raw or '')}


def limit(t):
    return max(math.ceil(1.5 * t), t + 3)


def analyze(rows, pids):
    by = {(str(r['pid']), r['method']): r for r in rows if r['method'] in METHODS}
    missing = [(p, m) for p in pids for m in METHODS if (p, m) not in by]
    assert not missing, f'{len(missing)} rows missing, e.g. {missing[:3]}'
    errors = {m: sum(by[(p, m)].get('status') != 'ok' for p in pids) for m in METHODS}
    counts = {m: {c: sum(flags(by[(p, m)])[c] for p in pids) for c in COUNTS} for m in METHODS}
    checks = {c: {'in': counts[IN][c], 'tsr': counts[TSR][c], 'limit': limit(counts[TSR][c]),
                  'pass': counts[IN][c] <= limit(counts[TSR][c])} for c in COUNTS}
    return {'n': len(pids), 'errors': errors, 'counts': counts, 'check1': checks,
            'check1_pass': all(v['pass'] for v in checks.values()),
            'rule': 'in <= max(ceil(1.5*T), T+3) for each count; T = tsr_answer_prompt; out reported only'}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    rows = [json.loads(l) for l in Path(a.rows).read_text().splitlines() if l.strip()]
    res = analyze(rows, json.loads(IDS.read_text()))
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'n': res['n'], 'check1_pass': res['check1_pass']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
