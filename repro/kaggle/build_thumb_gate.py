"""Pack the thumb_gate diagnostic (diag/thumb_gate.py; plan docs/EXPERIMENT_FLOW.md 2026-10-22 "실행 전 계획: 썸네일 관문 `thumb_gate`") into a
single-cell Kaggle GPU notebook. Adapted from build_full_think.py (same pins, same notebook shape). Only builds the run folder
runs/NAME (notebook, kernel-metadata.json, run_config.json); nothing is pushed.

usage: python3 repro/kaggle/build_thumb_gate.py NAME [--set main|trial] [--limit 5] [--group 4] [--timeout 14400]
           [--reencode-check | --no-reencode-check] [--skip-from runs/EARLIER/output/results/thumb_gate_*.jsonl]
One diag/thumb_gate.py process per visible GPU (items sharded by index, --shard i/n), outputs results/thumb_gate_{i}.jsonl and
log_{i}.txt. The plan asks for a Kaggle session limit of at least 4 hours: --timeout below 14400 is refused for --set main.
--reencode-check defaults to on for the trial and off for main. Push / pull with kaggle_ctl.py (by the user).
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

MIN_TIMEOUT_MAIN = 14400
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
for i in range(n):
    cmd = [sys.executable, 'diag/thumb_gate.py', '--parquet', pq, '--model', CFG['model'],
           '--revision', CFG['model_rev'], '--samples', 'diag/step1_store_bmain.json', '--ids', 'diag/thumb_gate_ids.json',
           '--shard', f'{{i}}/{{n}}', '--out', str(OUT / f'thumb_gate_{{i}}.jsonl'), '--skip', str(skip),
           '--time-limit', str(CFG['time_limit'] - (time.time() - T0))] + CFG['run_args']
    procs.append(subprocess.Popen(cmd, cwd=SRC, env={{**os.environ, 'CUDA_VISIBLE_DEVICES': str(i),
                                                       'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}},
                                  stdout=open(OUT / f'log_{{i}}.txt', 'w'), stderr=subprocess.STDOUT))
while any(p.poll() is None for p in procs):
    time.sleep(60)
    for i in range(n):
        lines = (OUT / f'log_{{i}}.txt').read_text().splitlines()
        print(f'[gpu{{i}} {{time.time()-T0:.0f}}s]', lines[-1][:300] if lines else '', flush=True)
for i, p in enumerate(procs):
    print('gpu', i, 'exit', p.returncode, '\n', '\n'.join((OUT / f'log_{{i}}.txt').read_text().splitlines()[-12:]))
print('total_s', round(time.time() - T0))
'''

# diag/thumb_gate.py imports diag/step1_think.py and diag/inject_sensitivity.py; the ids / store files are its data
FILES = ['run.py', 'diag/thumb_gate.py', 'diag/step1_think.py', 'diag/inject_sensitivity.py', 'diag/step1_store_bmain.json',
         'diag/thumb_gate_ids.json']


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--set', default='main', choices=('main', 'trial'))
    ap.add_argument('--limit', type=int, default=0, help='first N items of the ordered list (smoke: 5)')
    ap.add_argument('--group', type=int, default=4)
    ap.add_argument('--timeout', type=int, default=MIN_TIMEOUT_MAIN, help='Kaggle session timeout (s); main needs >= 14400')
    ap.add_argument('--reencode-check', action=argparse.BooleanOptionalAction, default=None,
                    help='re-encode multi-tile images without the thumbnail (default: on for trial, off for main)')
    ap.add_argument('--accelerator', default='NvidiaTeslaT4')
    ap.add_argument('--skip-from', nargs='*', default=[], help='earlier thumb_gate JSONL files; their pids are skipped')
    ap.add_argument('--owner', default='', help='Kaggle username (default: read from kaggle.json at the repo root)')
    ap.add_argument('--overwrite', action='store_true', help='rebuild an existing, not yet pushed run folder')
    a = ap.parse_args(argv)
    if a.set == 'main' and a.timeout < MIN_TIMEOUT_MAIN:
        sys.exit(f'--timeout {a.timeout} < {MIN_TIMEOUT_MAIN}: refused for --set main (the plan asks for a Kaggle limit of at '
                 'least 4 hours)')
    if a.timeout <= 900:
        sys.exit('--timeout must exceed 900 s (the time limit leaves 900 s for the last group and the copy-out)')
    reencode = a.set == 'trial' if a.reencode_check is None else a.reencode_check
    files = sorted(FILES + [str(p.relative_to(REPRO)) for p in (REPRO / 'visref').glob('*.py')])
    buf, sha = io.BytesIO(), {}
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            data = (REPRO / f).read_bytes()
            z.writestr(f, data)
            sha[f] = hashlib.sha256(data).hexdigest()
    skip = sorted({json.loads(l)['pid'] for f in a.skip_from for l in Path(f).read_text().splitlines() if l.strip()})
    run_args = ['--set', a.set, '--group', str(a.group)]
    run_args += (['--limit', str(a.limit)] if a.limit else []) + (['--reencode-check'] if reencode else [])
    cfg = {'model': MODEL, 'model_rev': MODEL_REV, 'data_rev': DATA_REV, 'parquet': PARQUET, 'run_args': run_args,
           'skip': skip, 'time_limit': a.timeout - 900}
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
          'limit': a.limit, 'group': a.group, 'reencode_check': reencode, 'kernel': meta['id'], 'accelerator': a.accelerator,
          'timeout': a.timeout, 'source_sha256': sha,
          'notebook_sha256': hashlib.sha256((out / 'notebook.ipynb').read_bytes()).hexdigest(),
          'built_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    (out / 'run_config.json').write_text(json.dumps(rc, indent=1))
    print(json.dumps({k: rc[k] for k in ('kernel', 'set', 'limit', 'group', 'reencode_check', 'n_skip', 'timeout', 'accelerator')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
