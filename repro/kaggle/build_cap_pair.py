"""Pack the cap_pair comparison (plan docs/EXPERIMENT_FLOW.md 2026-10-24 "실행 전 계획: 단계 상한 1,024 대 4,096") into a single-cell
Kaggle notebook. Same pins and bundle as build.py; only run.py and visref/ run. Builds runs/NAME only; nothing is pushed.

On every GPU i one thread runs run.py TWICE, in this order, on the same questions (diag/cap_pair_ids.json, --shard i/2):
  cap 1024: --think-cap 1024, out results_1024_{i}.jsonl, log_1024_{i}.txt, deadline = F1 * (time left until END)
  cap 4096: --think-cap 4096, out results_4096_{i}.jsonl, log_4096_{i}.txt, deadline = time left until END
END = timeout - 600 s after the notebook start (copy-out margin); each call's --time-limit = its deadline - 300. A call whose
deadline would be below 600 s is not started and is recorded in calls.json. All other run.py arguments are the main run's
(methods st,tsr,visref_raw_rank, total think cap 10,240, kmax 10, seed 42). No --skip: the two settings never share an output.

usage: python3 repro/kaggle/build_cap_pair.py NAME [--timeout 13500] [--f1 0.35] [--owner USER]
"""
import argparse
import base64
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
sys.path.insert(0, str(HERE))
from build import DATA_REV, MODEL, MODEL_REV, PARQUET, bundle, new_run_dir  # noqa: E402

CAPS = (1024, 4096)
BASE_ARGS = ['--methods', 'st,tsr,visref_raw_rank', '--total-think-cap', '10240']
CELL = r'''
import os, sys, io, json, time, base64, zipfile, pathlib, subprocess, threading
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
ids = OUT / 'ids.json'; ids.write_text(json.dumps(CFG['ids']))
calls, lock = [], threading.Lock()

def gpu(i):
    for k, cap in enumerate(CFG['caps']):
        left = CFG['end_s'] - (time.time() - T0)
        dl = left * CFG['f1'] if k == 0 else left
        rec = {{'gpu': i, 'cap': cap, 'start_s': round(time.time() - T0), 'deadline_s': round(dl)}}
        if dl < 600:
            rec['skipped'] = True
        else:
            cmd = [sys.executable, 'run.py', '--parquet', pq, '--model', CFG['model'], '--revision', CFG['model_rev'],
                   '--ids', str(ids), '--shard', f'{{i}}/{{n}}', '--out', str(OUT / f'results_{{cap}}_{{i}}.jsonl'),
                   '--think-cap', str(cap), '--time-limit', str(dl - 300), '--deadline', str(dl)] + CFG['run_args']
            rec['argv'] = cmd[2:]
            with open(OUT / f'log_{{cap}}_{{i}}.txt', 'w') as log:
                rec['exit'] = subprocess.run(cmd, cwd=SRC, env={{**os.environ, 'CUDA_VISIBLE_DEVICES': str(i),
                                             'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}},
                                             stdout=log, stderr=subprocess.STDOUT).returncode
        rec['end_s'] = round(time.time() - T0)
        with lock:
            calls.append(rec); (OUT / 'calls.json').write_text(json.dumps(calls, indent=1))
        print(json.dumps(rec)[:400], flush=True)

th = [threading.Thread(target=gpu, args=(i,)) for i in range(n)]
for t in th: t.start()
while any(t.is_alive() for t in th):
    time.sleep(120)
    for f in sorted(OUT.glob('log_*.txt')):
        lines = f.read_text().splitlines()
        print(f'[{{f.name}} {{time.time()-T0:.0f}}s]', lines[-1][:200] if lines else '', flush=True)
print('total_s', round(time.time() - T0))
'''


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--timeout', type=int, default=13500)
    ap.add_argument('--f1', type=float, default=0.35, help='share of the time left that the 1,024 call may use')
    ap.add_argument('--accelerator', default='NvidiaTeslaT4')
    ap.add_argument('--owner', default='')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args(argv)
    if a.timeout <= 1800 or not 0 < a.f1 < 1:
        sys.exit('--timeout must exceed 1800 s and 0 < --f1 < 1')
    ids = json.loads((REPRO / 'diag/cap_pair_ids.json').read_text())['ids']
    data, hashes = bundle()
    cfg = {'model': MODEL, 'model_rev': MODEL_REV, 'data_rev': DATA_REV, 'parquet': PARQUET, 'ids': ids, 'caps': list(CAPS),
           'run_args': BASE_ARGS, 'end_s': a.timeout - 600, 'f1': a.f1}
    owner = a.owner or json.loads((REPRO.parent / 'kaggle.json').read_text())['username']
    slug = f'visref-{a.name}'.replace('_', '-').lower()
    out = new_run_dir(HERE, a.name, a.overwrite)
    cell = CELL.format(cfg=base64.b64encode(json.dumps(cfg).encode()).decode(), bundle=base64.b64encode(data).decode())
    nb = {'cells': [{'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                     'source': cell.strip().splitlines(keepends=True)}],
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
          'nbformat': 4, 'nbformat_minor': 4}
    (out / 'notebook.ipynb').write_text(json.dumps(nb))
    meta = {'id': f'{owner}/{slug}', 'title': slug, 'code_file': 'notebook.ipynb', 'language': 'python',
            'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
            'enable_internet': True, 'dataset_sources': [], 'competition_sources': [], 'kernel_sources': []}
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=1))
    rc = {**{k: v for k, v in cfg.items() if k != 'ids'}, 'n_ids': len(ids), 'kernel': meta['id'], 'accelerator': a.accelerator,
          'timeout': a.timeout, 'source_sha256': hashes,
          'notebook_sha256': hashlib.sha256((out / 'notebook.ipynb').read_bytes()).hexdigest(),
          'built_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    (out / 'run_config.json').write_text(json.dumps(rc, indent=1))
    print(json.dumps({k: rc[k] for k in ('kernel', 'n_ids', 'caps', 'end_s', 'f1', 'timeout')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
