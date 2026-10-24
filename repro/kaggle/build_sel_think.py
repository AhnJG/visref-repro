"""Pack the sel_think diagnostic (diag/sel_think.py; plan docs/EXPERIMENT_FLOW.md 2026-10-24) into a single-cell Kaggle GPU
notebook. Adapted from build_stop_form.py (same pins, same notebook shape). Only builds the run folder runs/NAME (notebook,
kernel-metadata.json, run_config.json with the sha256 of every bundled file); nothing is pushed.

usage: python3 repro/kaggle/build_sel_think.py NAME [--set main|trial] [--limit 5] [--group 4] [--timeout 19800]
           [--shard-offset 0] [--shard-total 2] [--skip-from runs/EARLIER/output/results/sel_think_*.jsonl] [--owner USER]
One diag/sel_think.py process per visible GPU i, with --shard (shard_offset + i)/shard_total.
The bundled diag/sel_think_ids.json keeps every list (main, trial, n_visual, donor) but only the stored first steps of the questions
this kernel runs (two GPUs: shards shard_offset and shard_offset + 1; the trial adds the memcheck question), so the notebook stays
small enough for the Kaggle API (a 1.34 MB notebook was refused on 2026-10-24). run_config.json records the sha256 of the repository
file and of the bundled copy. Outputs
results/sel_think_{shard}.jsonl and log_{shard}.txt. Push / pull with kaggle_ctl.py.
"""
import argparse
import base64
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
ROOT = REPRO.parent
sys.path.insert(0, str(HERE))
from build import DATA_REV, MODEL, MODEL_REV, PARQUET, new_run_dir  # noqa: E402  same pins as the main runs

MIN_TIMEOUT_MAIN = 3600
IDS = 'diag/sel_think_ids.json'
GPUS = 2  # T4 x 2


def needed_pids(doc, set_name, limit, shard_offset, shard_total, gpus=GPUS):
    """The pids this kernel runs: index j of the set's list (first `limit`) with j % shard_total in its shards, + memcheck in trial."""
    ids = [str(p) for p in doc[set_name]]
    ids = ids[:limit] if limit else ids
    shards = {(shard_offset + i) % shard_total for i in range(gpus)}
    keep = {p for j, p in enumerate(ids) if j % shard_total in shards}
    if set_name == 'trial' and doc.get('memcheck'):
        keep.add(str(doc['memcheck']))
    return keep


def slim_ids(raw, set_name, limit, shard_offset, shard_total):
    """The ids file with step1 kept only for this kernel's pids (lists unchanged, so shard indices are the same)."""
    doc = json.loads(raw)
    keep = needed_pids(doc, set_name, limit, shard_offset, shard_total)
    doc['step1'] = {p: v for p, v in doc['step1'].items() if p in keep}
    return (json.dumps(doc, separators=(',', ':')) + '\n').encode(), len(doc['step1'])
CELL = r'''
import os, sys, io, json, time, base64, zipfile, pathlib, subprocess
T0 = time.time()
W = pathlib.Path('/kaggle/working'); SRC = W / 'src'; OUT = W / 'results'; OUT.mkdir(exist_ok=True)
CFG = json.loads(base64.b64decode('{cfg}').decode())
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.55.0'], check=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode('{bundle}'))).extractall(SRC)
from huggingface_hub import hf_hub_download, snapshot_download
pq = hf_hub_download('AI4Math/MathVista', CFG['parquet'], repo_type='dataset', revision=CFG['data_rev'])
snapshot_download(CFG['model'], revision=CFG['model_rev'])
import torch, transformers
n = torch.cuda.device_count()
env = {{'torch': torch.__version__, 'transformers': transformers.__version__, 'python': sys.version,
       'gpus': [torch.cuda.get_device_name(i) for i in range(n)], 'cfg': CFG, 'setup_s': time.time() - T0}}
(OUT / 'env.json').write_text(json.dumps(env, indent=1)); print(json.dumps(env), flush=True)
skip = OUT / 'skip.json'; skip.write_text(json.dumps(CFG['skip']))
procs = []
sh = [CFG['shard_offset'] + i for i in range(n)]
for i in range(n):
    cmd = [sys.executable, 'diag/sel_think.py', '--parquet', pq, '--model', CFG['model'],
           '--revision', CFG['model_rev'], '--ids', 'diag/sel_think_ids.json',
           '--shard', f"{{sh[i]}}/{{CFG['shard_total']}}", '--out', str(OUT / f'sel_think_{{sh[i]}}.jsonl'), '--skip', str(skip),
           '--time-limit', str(CFG['time_limit'] - (time.time() - T0))] + CFG['run_args']
    procs.append(subprocess.Popen(cmd, cwd=SRC, env={{**os.environ, 'CUDA_VISIBLE_DEVICES': str(i),
                                                       'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}},
                                  stdout=open(OUT / f'log_{{sh[i]}}.txt', 'w'), stderr=subprocess.STDOUT))
while any(p.poll() is None for p in procs):
    time.sleep(60)
    for i in range(n):
        lines = (OUT / f'log_{{sh[i]}}.txt').read_text().splitlines()
        print(f'[gpu{{i}} {{time.time()-T0:.0f}}s]', lines[-1][:300] if lines else '', flush=True)
for i, p in enumerate(procs):
    print('gpu', i, 'exit', p.returncode, '\n', '\n'.join((OUT / f'log_{{sh[i]}}.txt').read_text().splitlines()[-12:]))
print('total_s', round(time.time() - T0))
'''

# diag/sel_think.py imports these diag helpers; the ids file (with blank_think's stored first steps) is its data
FILES = ['run.py', 'diag/sel_think.py', 'diag/blank_think.py', 'diag/closed_multistep.py', 'diag/step1_think.py',
         'diag/sel_think_ids.json']


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--set', default='main', choices=('main', 'trial'))
    ap.add_argument('--limit', type=int, default=0, help='first N items of the ordered list (smoke: 5)')
    ap.add_argument('--group', type=int, default=4)
    ap.add_argument('--shard-offset', type=int, default=0)
    ap.add_argument('--shard-total', type=int, default=2, help='shards over all kernels; each kernel runs one per GPU')
    ap.add_argument('--timeout', type=int, default=19800, help='Kaggle session timeout (s); main needs >= 3600')
    ap.add_argument('--accelerator', default='NvidiaTeslaT4')
    ap.add_argument('--skip-from', nargs='*', default=[], help='earlier sel_think JSONL files; their pids are skipped')
    ap.add_argument('--owner', default='', help='Kaggle username (default: read from kaggle.json at the repo root)')
    ap.add_argument('--overwrite', action='store_true', help='rebuild an existing, not yet pushed run folder')
    a = ap.parse_args(argv)
    if a.set == 'main' and a.timeout < MIN_TIMEOUT_MAIN:
        sys.exit(f'--timeout {a.timeout} < {MIN_TIMEOUT_MAIN}: refused for --set main')
    if a.timeout <= 900:
        sys.exit('--timeout must exceed 900 s (the time limit leaves 900 s for the last group and the copy-out)')
    if not 0 <= a.shard_offset < a.shard_total or a.shard_total < 1:
        sys.exit('--shard-offset must be in [0, --shard-total)')
    files = sorted(FILES + [str(p.relative_to(REPRO)) for p in (REPRO / 'visref').glob('*.py')])
    buf, sha = io.BytesIO(), {}
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            data = (REPRO / f).read_bytes()
            if f == IDS:
                sha[f + ' (repository)'] = hashlib.sha256(data).hexdigest()
                data, n_step1 = slim_ids(data, a.set, a.limit, a.shard_offset, a.shard_total)
                sha[f + ' (bundled, step1 of this kernel only)'] = hashlib.sha256(data).hexdigest()
                sha[f + ' n_step1'] = n_step1
            z.writestr(f, data)
            sha.setdefault(f, hashlib.sha256(data).hexdigest())
    skip = sorted({r['pid'] for f in a.skip_from for l in Path(f).read_text().splitlines() if l.strip()
                   for r in [json.loads(l)] if r.get('status') == 'ok'})  # error rows are re-run, never skipped
    run_args = ['--set', a.set, '--group', str(a.group)]
    run_args += ['--limit', str(a.limit)] if a.limit else []
    cfg = {'model': MODEL, 'model_rev': MODEL_REV, 'data_rev': DATA_REV, 'parquet': PARQUET, 'run_args': run_args,
           'skip': skip, 'time_limit': a.timeout - 900, 'shard_offset': a.shard_offset, 'shard_total': a.shard_total}
    owner = a.owner or json.loads((ROOT / 'kaggle.json').read_text())['username']
    slug = f'visref-{a.name}'.replace('_', '-').lower()
    out = new_run_dir(HERE, a.name, a.overwrite)
    cell = CELL.format(cfg=base64.b64encode(json.dumps(cfg).encode()).decode(), bundle=base64.b64encode(buf.getvalue()).decode())
    nb = {'cells': [{'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                     'source': cell.strip().splitlines(keepends=True)}],
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
          'nbformat': 4, 'nbformat_minor': 4}
    (out / 'notebook.ipynb').write_text(json.dumps(nb))
    meta = {'id': f'{owner}/{slug}', 'title': slug, 'code_file': 'notebook.ipynb', 'language': 'python',
            'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
            'enable_internet': True, 'dataset_sources': [], 'competition_sources': [], 'kernel_sources': []}
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=1))
    rc = {**{k: v for k, v in cfg.items() if k != 'skip'}, 'n_skip': len(skip), 'skip_from': a.skip_from, 'set': a.set,
          'limit': a.limit, 'group': a.group, 'kernel': meta['id'],
          'accelerator': a.accelerator, 'timeout': a.timeout, 'source_sha256': sha,
          'notebook_sha256': hashlib.sha256((out / 'notebook.ipynb').read_bytes()).hexdigest(),
          'built_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    (out / 'run_config.json').write_text(json.dumps(rc, indent=1))
    print(json.dumps({k: rc[k] for k in ('kernel', 'set', 'limit', 'group', 'n_skip', 'timeout', 'accelerator', 'shard_offset',
                                         'shard_total')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
