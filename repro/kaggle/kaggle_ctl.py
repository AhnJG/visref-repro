"""Push / status / pull a built run via the Kaggle API (run on a machine that can reach kaggle.com).

  python3 repro/kaggle/kaggle_ctl.py push NAME
  python3 repro/kaggle/kaggle_ctl.py status NAME
  python3 repro/kaggle/kaggle_ctl.py pull NAME
Credentials: <repo>/kaggle.json (never printed).
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def api():
    os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT)
    from kaggle.api.kaggle_api_extended import KaggleApi
    k = KaggleApi()
    k.authenticate()
    return k


def d(x):
    return x.to_dict() if hasattr(x, 'to_dict') else str(x)


def main():
    action, name = sys.argv[1], sys.argv[2]
    run = HERE / 'runs' / name
    rc = json.loads((run / 'run_config.json').read_text())
    k = api()
    if action == 'push':
        log = run / 'push.json'
        if log.exists():
            raise SystemExit(f'{log} exists: already pushed; check status instead of re-pushing')
        r = k.kernels_push(str(run), timeout=str(rc['timeout']), acc=rc['accelerator'])
        log.write_text(json.dumps({'pushed_at': time.time(), 'response': d(r)}, indent=1, default=str))
        print(d(r))
    elif action == 'status':  # the response object has no usable to_dict(); read its fields
        r = k.kernels_status(rc['kernel'])
        s = {'status': str(getattr(r, 'status', None)), 'failure_message': getattr(r, 'failure_message', None) or ''}
        (run / 'status.json').write_text(json.dumps(s, indent=1, default=str))
        print(s)
    elif action == 'pull':
        out = run / 'output'
        out.mkdir(exist_ok=True)
        k.kernels_output(rc['kernel'], str(out))
        print(sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()))


if __name__ == '__main__':
    main()
