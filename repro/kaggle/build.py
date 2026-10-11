"""Pack repro/ into a single-cell Kaggle notebook (one GPU kernel; one run.py process per visible GPU).

usage: python3 repro/kaggle/build.py NAME [--ids PATH|LIST] [--limit N] [--timeout S] [--run-args "--methods st,tsr"]
Writes repro/kaggle/runs/NAME/{notebook.ipynb,kernel-metadata.json,run_config.json}.
"""
import argparse
import base64
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
ROOT = REPRO.parent
MODEL, MODEL_REV = 'OpenGVLab/InternVL3_5-1B', '2f71cf52542334823e48a46ffba0e2bc9add3446'
DATA_REV = '2b6ad69445fbb5695c9b165475e8decdbeb97747'
PARQUET = 'data/testmini-00000-of-00001-725687bf7a18d64b.parquet'

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
(OUT / 'env.json').write_text(json.dumps(env, indent=1)); print(json.dumps(env))
ids = OUT / 'ids.json'; ids.write_text(json.dumps(CFG['ids']))
skip = OUT / 'skip.json'; skip.write_text(json.dumps(CFG.get('skip', [])))
procs = []
for i in range(n):
    cmd = [sys.executable, 'run.py', '--parquet', pq, '--model', CFG['model'], '--revision', CFG['model_rev'],
           '--ids', str(ids), '--shard', f'{{i}}/{{n}}', '--out', str(OUT / f'results_{{i}}.jsonl'),
           '--time-limit', str(CFG['time_limit'] - (time.time() - T0)),
           '--deadline', str(CFG['deadline'] - (time.time() - T0)), '--skip', str(skip)] + CFG['run_args']
    procs.append(subprocess.Popen(cmd, cwd=SRC, env={{**os.environ, 'CUDA_VISIBLE_DEVICES': str(i),
                                                       'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}},
                                  stdout=open(OUT / f'log_{{i}}.txt', 'w'), stderr=subprocess.STDOUT))
while any(p.poll() is None for p in procs):
    time.sleep(120)
    for i in range(n):
        lines = (OUT / f'log_{{i}}.txt').read_text().splitlines()
        print(f'[gpu{{i}} {{time.time()-T0:.0f}}s]', lines[-1][:300] if lines else '', flush=True)
for i, p in enumerate(procs):
    print('gpu', i, 'exit', p.returncode, '\n', '\n'.join((OUT / f'log_{{i}}.txt').read_text().splitlines()[-15:]))
res = [str(p) for p in OUT.glob('results_*.jsonl') if p.stat().st_size]  # *.partial holds unfinished jobs
if res:
    subprocess.run([sys.executable, 'summarize.py', *res, '--out', str(OUT / 'summary.json'),
                    '--order', str(ids), '--shards', str(n)], cwd=SRC)
'''


def bundle():
    buf = io.BytesIO()
    files = sorted([REPRO / 'run.py', REPRO / 'summarize.py', *(REPRO / 'visref').glob('*.py')])
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.writestr(str(f.relative_to(REPRO)), f.read_bytes())
    return buf.getvalue(), {str(f.relative_to(REPRO)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--ids', default='', help='manifest path, JSON list path, or comma list; default all 1000')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--timeout', type=int, default=7200, help='Kaggle session timeout (s)')
    ap.add_argument('--shuffle-seed', type=int, default=0,
                    help='permute ids with this seed so any completed prefix is a random subset')
    ap.add_argument('--accelerator', default='NvidiaTeslaT4')
    ap.add_argument('--skip-from', nargs='*', default=[], help='earlier result JSONL files; their (pid, method) pairs are skipped')
    ap.add_argument('--run-args', default='', help='extra run.py arguments, space separated')
    a = ap.parse_args()
    if a.ids and Path(a.ids).exists():
        ids = json.loads(Path(a.ids).read_text())
        ids = ids['ids'] if isinstance(ids, dict) else ids
    elif a.ids:
        ids = a.ids.split(',')
    else:
        ids = [str(i) for i in range(1, 1001)]
    ids = [str(i) for i in ids]
    if a.shuffle_seed is not None:
        import random
        random.Random(a.shuffle_seed).shuffle(ids)
    ids = ids[: a.limit or None]
    skip = sorted({(json.loads(l)['pid'], json.loads(l)['method']) for f in a.skip_from
                   for l in Path(f).read_text().splitlines() if l.strip()})
    data, hashes = bundle()
    cfg = {'model': MODEL, 'model_rev': MODEL_REV, 'data_rev': DATA_REV, 'parquet': PARQUET, 'ids': ids,
           'time_limit': a.timeout - min(2700, a.timeout // 3), 'deadline': a.timeout - 600, 'run_args': a.run_args.split(),
           'skip': skip}
    owner = json.loads((ROOT / 'kaggle.json').read_text())['username']
    slug = f'visref-repro-{a.name}'.replace('_', '-').lower()  # Kaggle slugs: lowercase, digits, hyphens
    out = HERE / 'runs' / a.name
    out.mkdir(parents=True, exist_ok=True)
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
    rc_extra = {'shuffle_seed': a.shuffle_seed}
    rc = {**rc_extra, **{k: v for k, v in cfg.items() if k != 'skip'}, 'n_skip': len(skip), 'skip_from': a.skip_from, 'n_ids': len(ids), 'kernel': meta['id'], 'accelerator': a.accelerator, 'timeout': a.timeout,
          'source_sha256': hashes, 'notebook_sha256': hashlib.sha256((out / 'notebook.ipynb').read_bytes()).hexdigest(),
          'built_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    (out / 'run_config.json').write_text(json.dumps(rc, indent=1))
    print(json.dumps({k: rc[k] for k in ('kernel', 'n_ids', 'run_args', 'timeout')}))


if __name__ == '__main__':
    main()
