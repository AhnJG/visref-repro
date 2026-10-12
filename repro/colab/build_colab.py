"""Build a Google Colab notebook that runs repro/run.py on one Colab GPU, resumable across sessions.

  python3 repro/colab/build_colab.py NAME --ids-first N --run-args="--methods st ..."

The notebook embeds the same source bundle as kaggle/build.py (sha256 checked inside the notebook) and keeps
results on Google Drive (MyDrive/VisRef_colab/NAME). run.py skips (pid, method) rows already in its output file,
so after a disconnect the user re-opens the notebook and runs all cells again; only in-flight questions are redone.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import random
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
spec = importlib.util.spec_from_file_location('kbuild', REPRO / 'kaggle' / 'build.py')
kb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kb)  # reuse MODEL, MODEL_REV, DATA_REV, PARQUET, bundle()


def code_version():  # same as run.code_version() without importing torch
    h = hashlib.sha256()
    for f in sorted([REPRO / 'run.py', *(REPRO / 'visref').glob('*.py')]):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:12]


MD = """# VisRef {name} (Colab)
**실행 방법**: 메뉴 런타임 → 런타임 유형 변경 → **T4 GPU** 선택 → 런타임 → **모두 실행**. 처음 한 번 Google Drive 접근 허용.
- 결과는 `MyDrive/VisRef_colab/{name}/`에 1분마다 저장됩니다. 연결이 끊기면 이 노트북을 다시 열고 **모두 실행**만 하면 끝난 문항은 건너뛰고 이어서 돕니다.
- 마지막 셀이 실행 중인 동안 탭을 닫아도 되지만, 무료 등급은 최대 12시간이고 GPU 할당이 보장되지 않습니다.
- 진행 상황은 마지막 셀 출력의 `done N/{n}`입니다. 모두 끝나면 `ALL DONE`이 출력됩니다.
"""

CELL_MOUNT = """from google.colab import drive
drive.mount('/content/drive')
import os, sys, io, json, time, base64, zipfile, hashlib, shutil, subprocess, pathlib
NAME = {name!r}
RUN = pathlib.Path('/content/drive/MyDrive/VisRef_colab') / NAME
RUN.mkdir(parents=True, exist_ok=True)
LOCAL = pathlib.Path('/content/out'); LOCAL.mkdir(exist_ok=True)
print('Drive folder:', RUN)"""

CELL_SETUP = """CFG = json.loads(base64.b64decode('{cfg}').decode())
BUNDLE = '{bundle}'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.55.0', 'timm', 'einops'], check=True)  # timm/einops: InternViT remote code; skipped if present
SRC = pathlib.Path('/content/src')
zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))).extractall(SRC)
for k, v in CFG['source_sha256'].items():
    assert hashlib.sha256((SRC / k).read_bytes()).hexdigest() == v, 'source mismatch: ' + k
from huggingface_hub import hf_hub_download, snapshot_download
PQ = hf_hub_download('AI4Math/MathVista', CFG['parquet'], repo_type='dataset', revision=CFG['data_rev'])
snapshot_download(CFG['model'], revision=CFG['model_rev'])
print('setup ok, code_version', CFG['code_version'], 'n_ids', len(CFG['ids']))"""

CELL_RUN = """import torch, transformers
assert torch.cuda.is_available(), 'GPU 런타임이 아닙니다: 런타임 유형을 T4 GPU로 바꾸세요'
out, drive_out = LOCAL / 'results_0.jsonl', RUN / 'results_0.jsonl'
if drive_out.exists() and not out.exists():
    shutil.copy(drive_out, out)
if out.exists():  # drop a truncated last line left by a disconnect
    lines = out.read_text().splitlines(); good = []
    for l in lines:
        try:
            json.loads(l); good.append(l)
        except Exception:
            pass
    if len(good) != len(lines):
        out.write_text(''.join(x + '\\n' for x in good)); print('dropped broken lines:', len(lines) - len(good))
ts = time.strftime('%Y%m%d-%H%M%S')
env = {{'torch': torch.__version__, 'transformers': transformers.__version__, 'python': sys.version,
       'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], 'started': ts,
       'code_version': CFG['code_version'], 'run_args': CFG['run_args'], 'built_at': CFG['built_at']}}
(RUN / f'env_{{ts}}.json').write_text(json.dumps(env, indent=1)); print(env)
ids = LOCAL / 'ids.json'; ids.write_text(json.dumps(CFG['ids'])); shutil.copy(ids, RUN / 'ids.json')
log = LOCAL / f'log_{{ts}}.txt'
cmd = [sys.executable, 'run.py', '--parquet', PQ, '--model', CFG['model'], '--revision', CFG['model_rev'],
       '--ids', str(ids), '--shard', '0/1', '--out', str(out)] + CFG['run_args']
p = subprocess.Popen(cmd, cwd=SRC, stdout=open(log, 'w'), stderr=subprocess.STDOUT,
                     env={{**os.environ, 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}})
def sync(state='running'):
    for f in (out, log):
        if f.exists():
            shutil.copy(f, RUN / f.name)
    n = len(out.read_text().splitlines()) if out.exists() else 0
    (RUN / 'status.json').write_text(json.dumps({{'state': state, 'done': n, 'total': len(CFG['ids']),
        'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'session': ts}}))
    return n
sync()
while p.poll() is None:
    time.sleep(60); n = sync()
    last = log.read_text().splitlines()[-1:] if log.exists() else []
    print(time.strftime('%H:%M:%S'), f'done {{n}}/{{len(CFG["ids"])}}', (last[0][:160] if last else ''), flush=True)
n = len(out.read_text().splitlines()) if out.exists() else 0
sync('finished' if n >= len(CFG['ids']) else 'stopped')
print('exit', p.returncode); print('\\n'.join(log.read_text().splitlines()[-15:]))
print('ALL DONE' if n >= len(CFG['ids']) else f'stopped at {{n}}/{{len(CFG["ids"])}} - re-run all cells to continue')"""


def cell(src, kind='code'):
    c = {'cell_type': kind, 'metadata': {}, 'source': src.strip().splitlines(keepends=True)}
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--ids-first', type=int, required=True, help='first N pids of the seed-0 shuffled order (same as b_main)')
    ap.add_argument('--shuffle-seed', type=int, default=0)
    ap.add_argument('--run-args', default='')
    ap.add_argument('--overwrite', action='store_true', help='rebuild an existing, not yet started run folder')
    a = ap.parse_args()
    ids = [str(i) for i in range(1, 1001)]
    random.Random(a.shuffle_seed).shuffle(ids)
    ids = ids[: a.ids_first]
    data, hashes = kb.bundle()
    cfg = {'model': kb.MODEL, 'model_rev': kb.MODEL_REV, 'data_rev': kb.DATA_REV, 'parquet': kb.PARQUET, 'ids': ids,
           'run_args': a.run_args.split(), 'source_sha256': hashes, 'code_version': code_version(),
           'built_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    nb = {'cells': [cell(MD.format(name=a.name, n=len(ids)), 'markdown'),
                    cell(CELL_MOUNT.format(name=a.name)),
                    cell(CELL_SETUP.format(cfg=base64.b64encode(json.dumps(cfg).encode()).decode(),
                                           bundle=base64.b64encode(data).decode())),
                    cell(CELL_RUN.format())],
          'metadata': {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'},
                       'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}},
          'nbformat': 4, 'nbformat_minor': 0}
    out = HERE / 'runs' / a.name
    if not a.overwrite and any((out / f).exists() for f in ('run_config.json', f'{a.name}.ipynb', 'output')):
        raise SystemExit(f'refusing to overwrite the existing run record {out}: choose a new run name '
                         '(or pass --overwrite to rebuild a run that has not been started yet)')
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{a.name}.ipynb').write_text(json.dumps(nb))
    rc = {k: v for k, v in cfg.items()}
    rc.update(n_ids=len(ids), shuffle_seed=a.shuffle_seed,
              notebook_sha256=hashlib.sha256((out / f'{a.name}.ipynb').read_bytes()).hexdigest())
    (out / 'run_config.json').write_text(json.dumps(rc, indent=1))
    print(json.dumps({'name': a.name, 'n_ids': len(ids), 'code_version': cfg['code_version'], 'run_args': cfg['run_args'],
                      'notebook': str(out / f'{a.name}.ipynb'), 'bytes': (out / f'{a.name}.ipynb').stat().st_size}))


if __name__ == '__main__':
    main()
