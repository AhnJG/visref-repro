"""Pack the tally_force diagnostic (diag/tally_force.py) into a single-cell Kaggle GPU notebook.

Adapted from build_sel_think.py. Builds the run folder runs/NAME (notebook.ipynb,
kernel-metadata.json, run_config.json with the sha256 of every bundled file).
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

from build import MODEL, MODEL_REV, new_run_dir  # noqa: E402

TALLY_REPO = 'vikhyatk/tallyqa-test'
TALLY_REV = '2636200d9845bf54d545ed9e66b2957532c5bf13'
MIN_TIMEOUT_MAIN = 3600
IDS = 'diag/tally_force_ids.json'
GPUS = 2

FILES = [
    'run.py',
    'diag/tally_force.py',
    'diag/run_common.py',
    'diag/blank_think.py',
    'diag/closed_multistep.py',
    'diag/step1_think.py',
    'diag/tally_force_ids.json',
]

CELL = r'''
import os, sys, io, json, time, base64, zipfile, pathlib, subprocess
T0 = time.time()
W = pathlib.Path('/kaggle/working'); SRC = W / 'src'; OUT = W / 'results'; OUT.mkdir(exist_ok=True)
CFG = json.loads(base64.b64decode('{cfg}').decode())
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.55.0'], check=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode('{bundle}'))).extractall(SRC)
from huggingface_hub import hf_hub_download, snapshot_download
DATA_DIR = W / 'data'
for shard in CFG['needed_shards']:
    hf_hub_download('vikhyatk/tallyqa-test', f'data/test-{{shard:05d}}-of-00019.parquet', repo_type='dataset', revision=CFG['tally_rev'], local_dir=DATA_DIR)
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
    cmd = [sys.executable, 'diag/tally_force.py', '--parquet-dir', str(DATA_DIR), '--model', CFG['model'],
           '--revision', CFG['model_rev'], '--ids', 'diag/tally_force_ids.json',
           '--shard', f"{{sh[i]}}/{{CFG['shard_total']}}", '--out', str(OUT / f'tally_force_{{sh[i]}}.jsonl'), '--skip', str(skip),
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


def needed_pids(doc, set_name, limit, shard_offset, shard_total, gpus=GPUS):
    """The pids this kernel runs: index j of the set's list (first `limit`) with j % shard_total in its shards, + memcheck in trial."""
    ids = [str(p) for p in doc[set_name]]
    ids = ids[:limit] if limit else ids
    shards = {(shard_offset + i) % shard_total for i in range(gpus)}
    keep = {p for j, p in enumerate(ids) if j % shard_total in shards}
    if set_name == 'trial' and doc.get('memcheck'):
        keep.add(str(doc['memcheck']))
    return keep


def needed_shards(doc, set_name, limit, shard_offset, shard_total, gpus=GPUS):
    pids = needed_pids(doc, set_name, limit, shard_offset, shard_total, gpus)
    return sorted({doc['loc'][p]['shard'] for p in pids if p in doc['loc']})


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--set', default='main', choices=('main', 'trial'))
    ap.add_argument('--limit', type=int, default=0, help='first N items of the ordered list')
    ap.add_argument('--group', type=int, default=4)
    ap.add_argument('--shard-offset', type=int, default=0)
    ap.add_argument('--shard-total', type=int, default=2, help='shards over all kernels')
    ap.add_argument('--timeout', type=int, default=19800, help='Kaggle session timeout (s)')
    ap.add_argument('--accelerator', default='NvidiaTeslaT4')
    ap.add_argument('--skip-from', nargs='*', default=[], help='earlier tally_force JSONL files')
    ap.add_argument('--owner', default='', help='Kaggle username (default: read from kaggle.json at repo root)')
    ap.add_argument('--overwrite', action='store_true', help='rebuild an existing run folder')
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.set == 'main' and a.timeout < MIN_TIMEOUT_MAIN:
        sys.exit(f'--timeout {a.timeout} < {MIN_TIMEOUT_MAIN}: refused for --set main')
    if a.timeout <= 900:
        sys.exit('--timeout must exceed 900 s')
    if not 0 <= a.shard_offset < a.shard_total or a.shard_total < 1:
        sys.exit('--shard-offset must be in [0, --shard-total)')

    if a.owner:
        owner = a.owner
    elif (ROOT / 'kaggle.json').exists():
        owner = json.loads((ROOT / 'kaggle.json').read_text())['username']
    else:
        sys.exit('--owner is required when kaggle.json is not present at repository root')

    doc = json.loads((REPRO / IDS).read_text())
    shards = needed_shards(doc, a.set, a.limit, a.shard_offset, a.shard_total)

    files = sorted(FILES + [str(p.relative_to(REPRO)) for p in (REPRO / 'visref').glob('*.py')])
    buf, sha = io.BytesIO(), {}
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            data = (REPRO / f).read_bytes()
            z.writestr(f, data)
            sha[f] = hashlib.sha256(data).hexdigest()

    skip = sorted({r['pid'] for f in a.skip_from for l in Path(f).read_text().splitlines() if l.strip()
                   for r in [json.loads(l)] if r.get('status') == 'ok'})

    run_args = ['--set', a.set, '--group', str(a.group)]
    run_args += ['--limit', str(a.limit)] if a.limit else []
    cfg = {
        'model': MODEL, 'model_rev': MODEL_REV, 'tally_rev': TALLY_REV, 'needed_shards': shards,
        'run_args': run_args, 'skip': skip, 'time_limit': a.timeout - 900,
        'shard_offset': a.shard_offset, 'shard_total': a.shard_total
    }
    slug = f'visref-{a.name}'.replace('_', '-').lower()
    out = new_run_dir(HERE, a.name, a.overwrite)
    cell = CELL.format(cfg=base64.b64encode(json.dumps(cfg).encode()).decode(), bundle=base64.b64encode(buf.getvalue()).decode())
    nb = {
        'cells': [{'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                   'source': cell.strip().splitlines(keepends=True)}],
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'nbformat': 4, 'nbformat_minor': 4
    }
    (out / 'notebook.ipynb').write_text(json.dumps(nb))
    meta = {
        'id': f'{owner}/{slug}', 'title': slug, 'code_file': 'notebook.ipynb', 'language': 'python',
        'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
        'enable_internet': True, 'dataset_sources': [], 'competition_sources': [], 'kernel_sources': []
    }
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=1))
    rc = {
        **{k: v for k, v in cfg.items() if k != 'skip'}, 'n_skip': len(skip), 'skip_from': a.skip_from, 'set': a.set,
        'limit': a.limit, 'group': a.group, 'kernel': meta['id'],
        'accelerator': a.accelerator, 'timeout': a.timeout, 'source_sha256': sha,
        'notebook_sha256': hashlib.sha256((out / 'notebook.ipynb').read_bytes()).hexdigest(),
        'built_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')
    }
    (out / 'run_config.json').write_text(json.dumps(rc, indent=1))
    print(json.dumps({k: rc[k] for k in ('kernel', 'set', 'limit', 'group', 'n_skip', 'timeout', 'accelerator', 'shard_offset',
                                         'shard_total')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
