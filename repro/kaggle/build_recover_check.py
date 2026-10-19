"""Pack the recovery check `recover_check` (plan /home/claude/plan_recover.md "회복 시험") into a single-cell Kaggle GPU
notebook. Adapted from build_inject_sensitivity.py (same pins, same notebook shape).

usage: python3 repro/kaggle/build_recover_check.py NAME [--mode trial|main] [--limit N] [--timeout 10800]
  trial  the 6 questions outside the 200 (ids file 'trial'): per GPU first `recover_check.py --check` (numeric check, plan
         section 7), then the 14 conditions on the same questions (rows are excluded by the analysis)
  main   the 199 questions of the ids file 'main', the 14 conditions
One diag/recover_check.py process per visible GPU (questions sharded by index); no new group of questions starts after
timeout - 900 s. Refuses to overwrite an existing run folder (build.new_run_dir). Push / pull with kaggle_ctl.py.
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

CELL = r'''
import os, sys, io, json, time, base64, zipfile, pathlib, subprocess, shlex
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
procs = []
for i in range(n):
    base = [sys.executable, 'diag/recover_check.py', '--parquet', pq, '--model', CFG['model'], '--revision', CFG['model_rev'],
            '--ids', 'diag/recover_check_ids.json', '--set', CFG['set'], '--shard', f'{{i}}/{{n}}']
    left = lambda: str(CFG['time_limit'] - (time.time() - T0))
    cmds = []
    if CFG['mode'] == 'trial':
        cmds.append(base + ['--check', '--out', str(OUT / f'recover_check_numeric_{{i}}.jsonl'), '--time-limit', left()] + CFG['run_args'])
    cmds.append(base + ['--out', str(OUT / f'recover_check_{{i}}.jsonl'), '--time-limit', left()] + CFG['run_args'])
    sh = ' ; '.join(shlex.join(c) for c in cmds)
    procs.append(subprocess.Popen(['sh', '-c', sh], cwd=SRC, env={{**os.environ, 'CUDA_VISIBLE_DEVICES': str(i),
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

FILES = ['run.py', 'diag/step1_think.py', 'diag/inject_sensitivity.py', 'diag/recover_check.py', 'diag/recover_check_ids.py',
         'diag/recover_check_ids.json']


def main(argv=None, here=HERE):
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--mode', default='main', choices=('trial', 'main'))
    ap.add_argument('--limit', type=int, default=0, help='first N questions of the list (default all)')
    ap.add_argument('--group', type=int, default=4, help='questions between two flushes / time-limit checks')
    ap.add_argument('--timeout', type=int, default=10800, help='Kaggle session timeout (s); no new group starts after timeout - 900')
    ap.add_argument('--accelerator', default='NvidiaTeslaT4')
    ap.add_argument('--owner', default='', help='Kaggle username (default: read from kaggle.json at the repo root)')
    ap.add_argument('--overwrite', action='store_true', help='rebuild an existing, not yet pushed run folder')
    a = ap.parse_args(argv)
    repro = REPRO
    files = sorted(FILES + [str(p.relative_to(repro)) for p in (repro / 'visref').glob('*.py')])
    out = new_run_dir(here, a.name, a.overwrite)  # refuses before anything is written
    buf, sha = io.BytesIO(), {}
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            data = (repro / f).read_bytes()
            z.writestr(f, data)
            sha[f] = hashlib.sha256(data).hexdigest()
    run_args = ['--group', str(a.group)] + (['--limit', str(a.limit)] if a.limit else [])
    cfg = {'model': MODEL, 'model_rev': MODEL_REV, 'data_rev': DATA_REV, 'parquet': PARQUET, 'run_args': run_args,
           'mode': a.mode, 'set': a.mode, 'time_limit': a.timeout - 900}
    owner = a.owner or json.loads((ROOT / 'kaggle.json').read_text())['username']
    slug = f'visref-{a.name}'.replace('_', '-').lower()
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
    rc = {**cfg, 'limit': a.limit, 'group': a.group, 'kernel': meta['id'], 'accelerator': a.accelerator,
          'timeout': a.timeout, 'source_sha256': sha,
          'notebook_sha256': hashlib.sha256((out / 'notebook.ipynb').read_bytes()).hexdigest(),
          'built_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    (out / 'run_config.json').write_text(json.dumps(rc, indent=1))
    print(json.dumps({k: rc[k] for k in ('kernel', 'mode', 'limit', 'timeout', 'accelerator')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
