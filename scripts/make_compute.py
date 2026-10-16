"""Build results/compute.csv: the GPU and the running time of every experiment, from the run records (standard library).

  python3 scripts/make_compute.py        # run from the repository root; part of `make results`

For each run folder under repro/kaggle/runs/: the GPUs from output/results/env.json ('gpus') and the generation time from
the logs output/results/log_*.txt (the last "elapsed" value of each start of run.py; a log with several starts, as after
a resume, adds the starts). One log per GPU. Times cover run.py only (generation), not the setup before it (package
install, model download and loading). Per experiment:
  wall_h      sum over its run folders of the slowest GPU of that folder (the run's wall time; sessions add up)
  gpu_h       sum over all logs (GPU-hours)
  rows        lines of the result files (results_*.jsonl, or step1_branch_*.jsonl)
GPUs and software differ between the Kaggle runs (2 x T4) and the A6 run (1 x RTX A5000), so times are not comparable
across them. An experiment whose run folders are not in the repository yet is listed with status 'running'.
"""
import csv
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / 'repro' / 'kaggle' / 'runs'
OUT = ROOT / 'results' / 'compute.csv'
EXPERIMENTS = [  # (label, description, run folders)
    ('A1', 'main run: ST, TSR, VisRef (9 sessions)', [f'b_main_s{i}' for i in range(1, 10)]),
    ('A2', 'VisRef-fill', ['b_main_fill']),
    ('A3', 'step1_branch (pilot and main)', ['step1_branch_pilot', 'step1_branch_main']),
    ('A4', 'VisRef-hs', ['b_main_hs']),
    ('A6', 'VisRef-imgtag, TSR with answer prompt, TSR (check run and main run)', ['imgtag_check', 'imgtag_main']),
    ('B1-B6', 'setting checks', ['smoke', 'smoke2', 'smoke3', 'main_s0', 'full_s1', 'b_test', 'diag_st_plain']),
    ('B7', 'two 10-question trials of the image-tag variant', ['bundle_pilot', 'bundle_pilot2']),
]


def log_seconds(path):
    """Generation seconds of one log: per start of run.py, the last 'elapsed' value; starts add up."""
    total, last = 0, 0
    for line in open(path, encoding='utf-8', errors='replace'):
        if '"event": "start"' in line:
            total, last = total + last, 0
        m = re.search(r'"elapsed": (\d+)', line)
        if m:
            last = int(m.group(1))
    return total + last


def gpus(run):
    env = run / 'output' / 'results' / 'env.json'
    if not env.exists():
        return None
    e = json.loads(env.read_text())
    g = e.get('gpus') or e.get('gpu')
    return g if isinstance(g, list) else [g] if g else None


def rows(run):
    n = 0
    for pat in ('results_*.jsonl', 'step1_branch_*.jsonl'):
        for f in glob.glob(str(run / 'output' / 'results' / pat)):
            n += sum(1 for line in open(f) if line.strip())
    return n


def gpu_text(names):
    names = sorted(set(names), key=names.index)
    return '; '.join(names) if names else ''


def main():
    out = []
    for label, desc, folders in EXPERIMENTS:
        present = [RUNS / f for f in folders if (RUNS / f).is_dir()]
        if not present:
            out.append({'experiment': label, 'description': desc, 'runs': ' '.join(folders), 'gpu': '', 'gpus_per_run': '',
                        'wall_h': '', 'gpu_h': '', 'rows': '', 'status': 'running'})
            continue
        wall = gpu_s = n = 0
        kinds, per = [], set()
        for run in present:
            secs = [log_seconds(f) for f in sorted(glob.glob(str(run / 'output' / 'results' / 'log_*.txt')))]
            wall += max(secs, default=0)
            gpu_s += sum(secs)
            n += rows(run)
            g = gpus(run) or []
            kinds += g
            per.add(len(g))
        out.append({'experiment': label, 'description': desc, 'runs': ' '.join(r.name for r in present),
                    'gpu': gpu_text(kinds), 'gpus_per_run': '/'.join(str(x) for x in sorted(per)),
                    'wall_h': round(wall / 3600, 1), 'gpu_h': round(gpu_s / 3600, 1), 'rows': n,
                    'status': 'done' if len(present) == len(folders) else 'partial'})
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(r)


if __name__ == '__main__':
    main()
