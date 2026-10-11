"""Pack the hidden-state rank diagnostic into a single-cell Kaggle CPU notebook (no GPU quota).

usage: python3 repro/kaggle/build_hidden_check.py [NAME]
Writes repro/kaggle/runs/NAME/{notebook.ipynb,kernel-metadata.json,run_config.json}; push with kaggle_ctl.py.
"""
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
sys.path.insert(0, str(HERE))
from build import DATA_REV, MODEL, MODEL_REV, PARQUET  # noqa: E402  same pins as the main runs

FILES = ['visref/__init__.py', 'visref/model.py', 'visref/dpp.py', 'diag/hidden_rank_check.py',
         'diag/hidden_rank_samples.json']

CELL = r'''
import sys, io, json, time, base64, zipfile, pathlib, subprocess
T0 = time.time()
W = pathlib.Path('/kaggle/working'); SRC = W / 'src'; OUT = W / 'results'; OUT.mkdir(exist_ok=True)
CFG = json.loads(base64.b64decode('{cfg}').decode())
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.55.0'], check=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode('{bundle}'))).extractall(SRC)
from huggingface_hub import hf_hub_download, snapshot_download
pq = hf_hub_download('AI4Math/MathVista', CFG['parquet'], repo_type='dataset', revision=CFG['data_rev'])
snapshot_download(CFG['model'], revision=CFG['model_rev'])
import torch, transformers, os
env = {{'torch': torch.__version__, 'transformers': transformers.__version__, 'python': sys.version,
       'cpus': os.cpu_count(), 'threads': torch.get_num_threads(), 'cfg': CFG, 'setup_s': time.time() - T0}}
(OUT / 'env.json').write_text(json.dumps(env, indent=1)); print(json.dumps(env), flush=True)
p = subprocess.run([sys.executable, 'diag/hidden_rank_check.py', '--samples', 'diag/hidden_rank_samples.json',
                    '--parquet', pq, '--model', CFG['model'], '--revision', CFG['model_rev'],
                    '--out', str(OUT / 'hidden_rank_check.json')], cwd=SRC,
                   stdout=open(OUT / 'log.txt', 'w'), stderr=subprocess.STDOUT)
print('exit', p.returncode, time.time() - T0)
print((OUT / 'log.txt').read_text()[-6000:])
'''


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'hidden_rank_check'
    run = HERE / 'runs' / name
    run.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    sha = {}
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            p = REPRO / f
            data = p.read_bytes() if p.exists() else b''  # visref/__init__.py may be absent
            z.writestr(f, data)
            sha[f] = hashlib.sha256(data).hexdigest()
    kernel = f'arki13/visref-{name.replace("_", "-")}'
    cfg = {'model': MODEL, 'model_rev': MODEL_REV, 'data_rev': DATA_REV, 'parquet': PARQUET}
    src = CELL.format(cfg=base64.b64encode(json.dumps(cfg).encode()).decode(),
                      bundle=base64.b64encode(buf.getvalue()).decode())
    nb = {'cells': [{'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                     'source': src.strip('\n')}],
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
          'nbformat': 4, 'nbformat_minor': 5}
    (run / 'notebook.ipynb').write_text(json.dumps(nb))
    (run / 'kernel-metadata.json').write_text(json.dumps({
        'id': kernel, 'title': kernel.split('/')[1], 'code_file': 'notebook.ipynb', 'language': 'python',
        'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': False, 'enable_tpu': False,
        'enable_internet': True, 'dataset_sources': [], 'competition_sources': [], 'kernel_sources': []}, indent=1))
    (run / 'run_config.json').write_text(json.dumps({
        **cfg, 'kernel': kernel, 'accelerator': None, 'timeout': 10800, 'source_sha256': sha,
        'built_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}, indent=1))
    print(json.dumps({'kernel': kernel, 'files': sha}, indent=1))


if __name__ == '__main__':
    main()
