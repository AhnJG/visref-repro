"""Main run (untagged injection) on the 200 questions of the format check (B8): another </think> in the answer and
accuracy, VisRef - TSR. A look after the results (2026-10-18), descriptive; it puts the main run next to A6 and B8 in
REPORT Section 2.6 (B8). Re-emission is counted as in analyze_format_check.flags (scored answer starts after a later
</think>); correctness is the main scoring (llm/final/merged.jsonl); 95% range by question bootstrap (2,000, seed 0).
Also: the B8 excess (new format − TSR with answer prompt) minus the A6 excess on the same questions, with its range.

Run from repro/: python diag/format_compare_200.py  ->  diag/format_compare_200.json
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_format_check import IDS, flags  # noqa: E402

A, B = 'visref_raw_rank', 'tsr'


def main():
    ids = json.loads(IDS.read_text())
    ids = [str(x) for x in (ids['ids'] if isinstance(ids, dict) else ids)]
    rs = {}
    for line in open('llm/final/merged.jsonl'):
        r = json.loads(line)
        rs[(str(r['pid']), r['method'])] = r
    c = lambda p, m: int(rs[(p, m)]['correct'])  # noqa: E731
    d = [c(p, A) - c(p, B) for p in ids]
    rng, n = random.Random(0), len(d)
    boots = sorted(100 * sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
    out = {'label': 'a look after the results, descriptive', 'n': n,
           'reemit': {m: sum(flags(rs[(p, m)])['reemit'] for p in ids) for m in (A, B)},
           'acc_pct': {m: round(100 * sum(c(p, m) for p in ids) / n, 1) for m in (A, B)},
           'diff_pp': round(100 * sum(d) / n, 1), 'range95_pp': [round(boots[50], 1), round(boots[1949], 1)],
           'a_only': sum(x == 1 for x in d), 'b_only': sum(x == -1 for x in d)}
    # excess of another </think> over the partner: this run's B8 pair minus A6's pair on the same questions
    b8, a6 = {}, {}
    for f in ['colab/runs/fmtcheck_main_cb/output/results_0.jsonl', 'kaggle/runs/fmtcheck_rest_kg/output/results/results_0.jsonl',
              'kaggle/runs/fmtcheck_rest_kg/output/results/results_1.jsonl']:
        for line in open(f):
            r = json.loads(line)
            b8[(str(r['pid']), r['method'])] = r
    for line in open('kaggle/runs/imgtag_main/output/results/results_0.jsonl'):
        r = json.loads(line)
        a6[(str(r['pid']), r['method'])] = r
    fl = lambda d, p, m: int(flags(d[(p, m)])['reemit'])  # noqa: E731
    e = [(fl(b8, p, 'visref_hs_imgtag_in') - fl(b8, p, 'tsr_answer_prompt'))
         - (fl(a6, p, 'visref_imgtag') - fl(a6, p, 'tsr_answer_prompt')) for p in ids]
    rng = random.Random(0)
    eb = sorted(sum(e[rng.randrange(n)] for _ in range(n)) for _ in range(2000))
    out['b8_minus_a6_excess'] = {'diff': sum(e), 'range95': [eb[50], eb[1949]]}
    Path('diag/format_compare_200.json').write_text(json.dumps(out, indent=1) + '\n')
    print(out)


if __name__ == '__main__':
    main()
