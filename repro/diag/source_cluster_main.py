"""Main run (A1): paired differences with whole MathVista source datasets resampled instead of single questions.

Sensitivity check, decided after the results (2026-10-18), descriptive, decides nothing: the verdict of the main run uses
the question-level CI of final_judgment.py only. Same estimator and resampling as the A6 check
(analyze_imgtag_main.source_cluster: 31 source datasets, 2,000 resamples, seed 0, leave-one-source-out). Rows: the main
scoring (llm/final/merged.jsonl).

Run from repro/: python diag/source_cluster_main.py  ->  diag/source_cluster_main.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_imgtag_main import source_cluster  # noqa: E402

PAIRS = [('visref_raw_rank', 'st'), ('visref_raw_rank', 'tsr'), ('tsr', 'st')]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--merged', default='llm/final/merged.jsonl')
    ap.add_argument('--data', default='../data/raw/mathvista_testmini.json')
    ap.add_argument('--out', default='diag/source_cluster_main.json')
    a = ap.parse_args(argv)
    data = json.load(open(a.data))
    rs = {}
    for line in open(a.merged):
        r = json.loads(line)
        rs[(r['pid'], r['method'])] = r
    pids = sorted({p for p, _ in rs}, key=int)
    assert len(pids) == 1000 and all((p, m) in rs for p in pids for pair in PAIRS for m in pair)
    out = {'label': 'sensitivity check decided after the results, descriptive: whole source datasets resampled; '
                    'the verdict uses the question-level CI only',
           'rows': a.merged, 'n_questions': len(pids)}
    for x, y in PAIRS:
        out[f'{x}-{y}'] = source_cluster(rs, pids, data, x, y)
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')
    for x, y in PAIRS:
        c = out[f'{x}-{y}']
        print(f'{x} - {y}: {c["diff_pp"]:+.2f} pp, source-resampled 95% CI {c["ci95_pp"]}, '
              f'leave one out {c["leave_one_source_out"]["lowest"]} .. {c["leave_one_source_out"]["highest"]}')


if __name__ == '__main__':
    main()
