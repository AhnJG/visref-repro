"""Run the 1,000-question image-tag injection comparison on one rented GPU machine (e.g. RunPod).

Does what the Kaggle notebook cell (repro/kaggle/build.py CELL) does for this experiment, reproducibly and resumably, on
a plain Linux GPU machine: pin transformers, download the MathVista parquet and the model at the pinned revisions, write
env.json / ids.json / run_config.json, run `run.py` (one process, shard 0/1, log appended to log_0.txt), print a progress
line every 120 s, then run summarize.py (`--order ids --shards 1`).

Experiment (EF 2026-10-15 "실행 전 계획: 이미지 태그 주입 1,000문항 비교"): methods visref_imgtag, tsr_answer_prompt,
tsr on the seed-0 question order of the main run, every method started from the main run's stored step 1
(diag/step1_store_bmain.json), dual probe on. All settings are in the module-level dict RUN below (single source of
truth); nothing about the experiment is taken from the command line except the name, the output root, --limit and
--hours. Not passed on purpose, so run.py / model.py defaults hold: dtype (fp16), batch (16), batch_tokens (72000),
chunk (1024); RUN['run_py_defaults'] records them and they are checked against run.py at start.

Plans (--plan, default imgtag_main): PLANS holds one settings dict per experiment and the chosen one becomes RUN.
imgtag_main is the A6 run above; its code_version pin is the one of commit 8f5c09e, so it runs only from that commit.
format_check (EF 2026-10-17 "동작 점검 계획"): methods visref_hs_imgtag_in, visref_hs_imgtag_out, tsr_answer_prompt on the
100 questions of diag/format_check_ids.json, other settings as imgtag_main; summarize.py is never run (no accuracy).
Its main run is named fmtcheck_main; a check run takes another name, e.g. fmtcheck_try --limit 3:
  python3 repro/runpod/run_imgtag.py --plan format_check --name fmtcheck_main --out-root /workspace/runs

Results land in <out-root>/<name>/output/results/ (the layout of the Kaggle runs): run_config.json is one level up at
<out-root>/<name>/run_config.json; results/ holds env.json, ids.json, results_0.jsonl, log_0.txt, summary.json.

Summary only when complete: summarize.py runs only if results_0.jsonl holds every expected (pid, method) pair
((--limit or 1000 questions) x 3 methods, unique pairs); otherwise the runner prints "incomplete: N/expected rows; summary
not written" and creates no summary.json (no accuracies are ever computed on partial data).
Failure handling: --max-errors N (default 10) terminates run.py as soon as more than N rows with status 'error' are in
results_0.jsonl (0 = stop at the first one) and exits with code 3 without a summary; --redo-errors moves the 'error' rows
out of results_0.jsonl into results_0.errors_<UTC timestamp>.jsonl before run.py starts, so run.py redoes those pairs (the
moved count is recorded in run_config.json 'restarts'); --stop-pod runs `runpodctl stop pod $RUNPOD_POD_ID` when the runner
is done, whether or not a summary was written (not after SIGINT/SIGTERM/SIGHUP, and not when another runner holds the lock).
With RUNPOD_POD_ID set, an --out-root outside /workspace prints a warning (results may be lost when the pod stops).
The progress line shows the number of error rows (counts only, never correctness).

Re-running the same command RESUMES: run.py skips every (pid, method) pair already in results_0.jsonl, the log is
appended, and run_config.json gets one more entry in 'restarts' (HEAD, code_version, ids and step-1 store sha must be the
same as in the first start, otherwise the runner refuses). Jobs in flight when a run is killed are redone.
Correctness is never printed (progress lines and the log tail carry row counts / status lines only; summarize.py output
goes to summary_stdout.txt).

How to use on RunPod
  1. Start a pod with a PyTorch (CUDA) image (torch >= 2.5; the main run used torch 2.11) and a network volume mounted
     at /workspace; one GPU is enough.
  2. git clone https://github.com/AhnJG/visref-repro && cd visref-repro && git checkout <commit>
     (the tree must be clean: the runner refuses a dirty tree unless --allow-dirty, and records HEAD).
  3. Keep outputs on the volume, OUTSIDE the clone (--out-root inside the repository is refused: the files written there
     would make the clean-tree check fail on every restart), and run detached so a dropped SSH session does not stop it:
       tmux new -s imgtag
       python3 repro/runpod/run_imgtag.py --name imgtag_main --out-root /workspace/runs
     (or: nohup python3 repro/runpod/run_imgtag.py --name imgtag_main --out-root /workspace/runs > /workspace/runner.log 2>&1 &)
     The model/data download goes to <out-root>/hf_cache (HF_HOME, unless HF_HOME is already set), so a restarted pod on the
     same volume reuses it. Missing/mismatched packages of requirements.txt are installed (torch is only checked).
     Add --dry-run first to see the exact run.py command and run_config without downloading or running anything.
     --hours H makes run.py stop admitting new jobs and dump in-flight ones before H hours are over (the same rule as the
     Kaggle cell: time_limit = T - min(2700, T/3), deadline = T - min(600, T/10) with T = H * 3600 minus the setup time
     already spent). It only stops run.py: the POD KEEPS RUNNING (and billing) until you stop it after the run (or pass --stop-pod).
  4. A short check run must use a name that does NOT match `imgtag_main*`, because the analysis collects
     kaggle/runs/imgtag_main*/output/results (the runner refuses --limit together with an imgtag_main name):
       python3 repro/runpod/run_imgtag.py --name imgtag_check --out-root /workspace/runs --limit 3
     Look at: rows with status 'error' (none expected), the fast_kv_check line in log_0.txt (ok: true), the last 'round'
     line's peak_alloc_gb / peak_reserved_gb against the GPU memory, and rows/hour in the progress line (3 questions x 3
     methods = 9 rows). Nothing about correctness is printed by the runner on purpose.
  5. Re-running the same command resumes (see above); stop the pod when it is finished, then copy
     <out-root>/<name> back to the analysis machine as repro/kaggle/runs/<name> (same layout), then
       cd repro && python3 diag/analyze_imgtag_main.py rows --out llm/imgtag_main/rows_v4.jsonl

Safety: an exclusive lock on <out-root>/<name>/.lock is held for the whole run (a second runner on the same name refuses
to start); SIGTERM / SIGINT / SIGHUP terminate run.py (30 s grace, then kill) before the runner exits; a torn last line of
results_0.jsonl (a write cut by a kill) is moved to results_0.jsonl.torn.<timestamp> at the next start and noted in
run_config.json 'restarts'. No GPU memory is touched by the runner itself (GPU name from nvidia-smi).
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
ROOT = REPRO.parent
sys.path.insert(0, str(REPRO / 'kaggle'))
import build  # noqa: E402  (constants only; build.main() is not executed)

MODEL, MODEL_REV, DATA_REV, PARQUET = build.MODEL, build.MODEL_REV, build.DATA_REV, build.PARQUET

_METHODS = 'visref_imgtag,tsr_answer_prompt,tsr'
RUN = {
    'plan': 'imgtag_main',
    'experiment': 'EF 2026-10-15 실행 전 계획: 이미지 태그 주입 1,000문항 비교',
    'methods': _METHODS,
    # run.py arguments after the fixed ones (--parquet --model --revision --ids --shard --out [--time-limit --deadline --limit])
    'run_args': ['--methods', _METHODS, '--think-cap', '1024', '--total-think-cap', '10240',
                 '--step1-from', 'diag/step1_store_bmain.json', '--dual-probe', '1'],
    'ids_file': 'kaggle/runs/b_main_hs/output/results/ids.json',  # relative to repro/: seed-0 order of the main run
    'ids_sha256': 'a5ad8da9d10e7991abad84ca75dd108d60f94b890c40600d9dfa5ae2eb31b030',
    'n_ids': 1000,
    'shard': '0/1',
    'store_file': 'diag/step1_store_bmain.json',  # relative to repro/ (run.py's cwd)
    'store_sha256_prefix': 'f937478f3b26',
    'code_version': '6e218887d2b5',  # run.code_version(): sha256 over run.py and visref/*.py (A6: commit 8f5c09e)
    'main_name': 'imgtag_main',  # analysis collects <main_name>*; --limit is refused with such a name
    'check_name': 'imgtag_check',  # example name of a short check run
    'summarize': True,
    'transformers': '4.55.0',
    'min_torch': '2.5',  # run.py uses torch.OutOfMemoryError
    'dtype': 'float16 (visref.model.load_model default; not passed)',
    'run_py_defaults': {'batch': 16, 'batch_tokens': 72000, 'chunk': 1024},  # run.py DEFAULTS, left untouched
    'cuda_alloc_conf': 'expandable_segments:True',
    'progress_every_s': 120,
}
# 2026-10-17 format check (EF "동작 점검 계획"): image block inside vs after the thinking block, same selection
# (Z = last-layer hidden states, kernel V = input image embeddings), with the same-prompt TSR; the first 100 questions of
# the seed-0 order whose stored step 1 closed itself (diag/format_check_ids.py). No accuracy: summarize.py is not run.
_FMT_METHODS = 'visref_hs_imgtag_in,visref_hs_imgtag_out,tsr_answer_prompt'
FORMAT_CHECK = dict(RUN, **{
    'plan': 'format_check',
    'experiment': 'EF 2026-10-17 동작 점검 계획: 이미지 블록 위치 두 형식(첫 단계를 스스로 닫은 100문항, 정답률 사용 안 함)',
    'methods': _FMT_METHODS,
    'run_args': ['--methods', _FMT_METHODS, '--think-cap', '1024', '--total-think-cap', '10240',
                 '--step1-from', 'diag/step1_store_bmain.json', '--dual-probe', '1'],
    'ids_file': 'diag/format_check_ids.json',
    'ids_sha256': '386538750286bf25cb5bc7e9c8b09635613d5f1a915c88b7519c56342719b161',
    'n_ids': 100,
    'code_version': '06435ae748c8',
    'main_name': 'fmtcheck_main',
    'check_name': 'fmtcheck_try',
    'summarize': False,
})
PLANS = {'imgtag_main': RUN, 'format_check': FORMAT_CHECK}
REQUIREMENTS = ROOT / 'requirements.txt'
SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
POLL_S = 5  # how often the progress loop looks at the new rows (the progress line itself: RUN['progress_every_s'])
EXIT_MAX_ERRORS = 3


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def time_budget(hours, elapsed_s=0.0):
    """(time_limit, deadline) in seconds for run.py: the Kaggle cell's rule, with the margins capped for short runs."""
    total = hours * 3600
    return total - min(2700, total // 3) - elapsed_s, total - min(600, total // 10) - elapsed_s


def build_cmd(parquet, ids_path, out_path, hours=None, limit=0, elapsed_s=0.0, python=None):
    """argv of the run.py process (cwd=repro/). Everything experiment-specific comes from RUN."""
    cmd = [python or sys.executable, 'run.py', '--parquet', str(parquet), '--model', MODEL, '--revision', MODEL_REV,
           '--ids', str(ids_path), '--shard', RUN['shard'], '--out', str(out_path)]
    if hours:
        tl, dl = time_budget(hours, elapsed_s)
        cmd += ['--time-limit', f'{tl:.0f}', '--deadline', f'{dl:.0f}']
    if limit:
        cmd += ['--limit', str(limit)]
    return cmd + list(RUN['run_args'])


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def repo_state():
    return {'head': git('rev-parse', 'HEAD'), 'dirty_files': git('status', '--porcelain').splitlines()}


def source_hashes():
    files = sorted([REPRO / 'run.py', *(REPRO / 'visref').glob('*.py')])
    return {str(f.relative_to(REPRO)): sha256_file(f) for f in files}


def probe_run_py():
    """code_version and DEFAULTS as run.py itself reports them (fresh process: sees the pinned transformers)."""
    code = 'import json, run; print(json.dumps({"code_version": run.code_version(), "defaults": run.DEFAULTS}))'
    out = subprocess.run([sys.executable, '-c', code], cwd=REPRO, capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def probe_torch():
    """torch and CUDA build versions from a fresh process (importing torch does not initialise CUDA; the runner stays clean)."""
    code = 'import json, torch; print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda}))'
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def version_tuple(v):
    m = re.match(r'(\d+)\.(\d+)', v)
    return (int(m[1]), int(m[2])) if m else (0, 0)


def check_torch(version):
    assert version_tuple(version) >= version_tuple(RUN['min_torch']), \
        f'torch {version} < {RUN["min_torch"]} (run.py needs torch.OutOfMemoryError)'


def installed(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def parse_requirements(path=REQUIREMENTS):
    """[(name, pinned version or None)] from requirements.txt (comments and blank lines skipped; only 'name' / 'name==ver')."""
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.split('#')[0].strip()
        if not line:
            continue
        m = re.fullmatch(r'([A-Za-z0-9_.-]+)\s*(?:==\s*([^\s]+))?', line)
        assert m, f'unsupported requirement line: {line!r}'
        out.append((m[1], m[2]))
    return out


def required_packages(reqs=None):
    """{name: pinned version or None}; transformers is pinned by RUN (and must agree with requirements.txt if pinned there)."""
    pins = dict(reqs if reqs is not None else parse_requirements())
    assert pins.get('transformers') in (None, RUN['transformers']), 'requirements.txt and RUN disagree on transformers'
    pins['transformers'] = RUN['transformers']
    return pins


def missing_packages(pins):
    """pip specs for packages that are absent or differ from a pin. torch is never installed (the image's build is kept)."""
    specs = []
    for name, ver in pins.items():
        if name.lower() == 'torch':
            continue
        cur = installed(name)
        if cur is None or (ver and cur != ver):
            specs.append(f'{name}=={ver}' if ver else name)
    return specs


def ensure_packages(pins, dry_run=False):
    """Install only missing/mismatched packages; returns ({name: version}, specs installed or that would be)."""
    specs = missing_packages(pins)
    if specs and not dry_run:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *specs], check=True)
    return {n: installed(n) for n in pins}, specs


def gpu_name():
    """GPU name and memory without initialising CUDA in this process: nvidia-smi, else a throw-away python process."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True,
                             text=True, check=True).stdout.strip().splitlines()
        if out and out[0].strip():
            return out[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        code = 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")'
        out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True).stdout.strip()
        return out.splitlines()[-1] if out else None
    except (OSError, subprocess.SubprocessError):
        return None


def check_inputs():
    """ids file and stored step-1 file must be the ones the plan names; returns (ids, ids_sha, store_sha)."""
    ids_path, store_path = REPRO / RUN['ids_file'], REPRO / RUN['store_file']
    ids_sha, store_sha = sha256_file(ids_path), sha256_file(store_path)
    assert ids_sha == RUN['ids_sha256'], f'ids sha256 {ids_sha} != {RUN["ids_sha256"]}'
    ids = [str(i) for i in json.loads(ids_path.read_text())]
    assert len(ids) == RUN['n_ids'], f'{len(ids)} ids, expected {RUN["n_ids"]}'
    assert store_sha.startswith(RUN['store_sha256_prefix']), f'step-1 store sha256 {store_sha[:12]} != {RUN["store_sha256_prefix"]}'
    return ids, ids_sha, store_sha


def check_out_root(out_root):
    """Resolved out-root; refuse one inside the git repository (its files would make the clean-tree check fail on restart)."""
    r, repo = Path(out_root).resolve(), ROOT.resolve()
    if r == repo or repo in r.parents:
        sys.exit(f'refusing --out-root {r}: it is inside the repository {repo}; the run files written there would make the '
                 'clean-tree check fail when the run is restarted. Use a folder outside the clone, e.g. /workspace/runs')
    return r


def select_plan(plan):
    """Make PLANS[plan] the module-level RUN that every function reads."""
    global RUN
    RUN = PLANS[plan]
    return RUN


def check_name(name, limit):
    mn = RUN['main_name']
    if limit and name.startswith(mn):
        sys.exit(f'refusing --limit with --name {name}: the analysis collects {mn}*; use a check name such as {RUN["check_name"]}')


def acquire_lock(path):
    """Exclusive non-blocking flock for the whole run; returns the open file (keep it referenced)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, 'a+')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        sys.exit(f'refusing to start: another runner holds {path}')
    f.seek(0)
    f.truncate()
    f.write(f'pid {os.getpid()} started {time.strftime("%Y-%m-%dT%H:%M:%S%z")}\n')
    f.flush()
    return f


def repair_torn_line(path, stamp=None):
    """A kill can cut the last JSON line. Move an invalid last line to <path>.torn.<stamp> and truncate it from the file; a
    valid last line missing its newline gets one (run.py appends). Returns a note dict or None."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    data = path.read_bytes()
    body = data[:-1] if data.endswith(b'\n') else data
    cut = body.rfind(b'\n') + 1
    last = body[cut:]
    try:
        json.loads(last)
        ok = True
    except ValueError:
        ok = False
    if ok:
        if not data.endswith(b'\n'):
            with open(path, 'ab') as f:
                f.write(b'\n')
            return {'newline_added': True}
        return None
    torn = Path(f'{path}.torn.{stamp or time.strftime("%Y%m%dT%H%M%S")}')
    torn.write_bytes(last)
    with open(path, 'r+b') as f:
        f.truncate(cut)
    return {'torn_file': torn.name, 'torn_bytes': len(last)}


def terminate_child(proc, grace=30):
    """terminate, wait up to `grace` s, then kill."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def expected_rows(limit=0):
    """Rows a complete run holds: (questions) x (methods); questions = --limit (at most n_ids) or all n_ids."""
    n = min(limit, RUN['n_ids']) if limit else RUN['n_ids']
    return n * len(RUN['methods'].split(','))


def summary_decision(n_rows, limit=0):
    """(run summarize.py?, message). Only a complete result file gets a summary."""
    exp = expected_rows(limit)
    if n_rows == exp:
        return True, f'complete: {n_rows}/{exp} rows'
    return False, f'incomplete: {n_rows}/{exp} rows; summary not written'


def too_many_errors(n_errors, max_errors):
    """True when run.py must be stopped: more error rows than allowed (max_errors 0: the first error row)."""
    return n_errors > max_errors


def move_error_rows(path, stamp=None):
    """Move rows with status 'error' out of `path` into <dir>/<stem>.errors_<UTC stamp>.jsonl so that run.py (which skips
    the (pid, method) pairs already present) redoes them. The kept rows are written to a temp file and put in place with
    os.replace. Unparsable lines are kept. Returns {'moved': n, 'file': name or None}."""
    path = Path(path)
    if not path.exists():
        return {'moved': 0, 'file': None}
    keep, moved = [], []
    for line in path.read_bytes().splitlines(keepends=True):
        try:
            err = json.loads(line).get('status') == 'error'
        except (ValueError, AttributeError):
            err = False
        (moved if err else keep).append(line if line.endswith(b'\n') else line + b'\n')
    if not moved:
        return {'moved': 0, 'file': None}
    dest = path.with_name(f'{path.stem}.errors_{stamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}.jsonl')
    dest.write_bytes(b''.join(moved))  # the moved rows are safe before the original is rewritten
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(b''.join(keep))
    os.replace(tmp, path)
    return {'moved': len(moved), 'file': dest.name}


def workspace_warning(out_root, env=None):
    """Warning text when running on a RunPod pod with results outside /workspace (the volume), else None."""
    env = os.environ if env is None else env
    if not env.get('RUNPOD_POD_ID'):
        return None
    r, ws = Path(out_root).resolve(), Path('/workspace')
    if r == ws or ws in r.parents:
        return None
    return (f'WARNING: RUNPOD_POD_ID is set but --out-root {r} is not under /workspace: the results may be lost when the '
            'pod stops')


def stop_pod(env=None):
    """`runpodctl stop pod $RUNPOD_POD_ID` (no shell). Prints a warning and returns False if it cannot be done."""
    env = os.environ if env is None else env
    pod = env.get('RUNPOD_POD_ID')
    if not pod:
        print('WARNING: --stop-pod given but RUNPOD_POD_ID is not set; the pod was NOT stopped', flush=True)
        return False
    if shutil.which('runpodctl') is None:
        print('WARNING: --stop-pod given but runpodctl was not found on PATH; the pod was NOT stopped', flush=True)
        return False
    try:
        r = subprocess.run(['runpodctl', 'stop', 'pod', pod], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        print(f'WARNING: runpodctl stop pod {pod} failed: {e!r}; the pod was NOT stopped', flush=True)
        return False
    print(f'runpodctl stop pod {pod}: exit {r.returncode} {(r.stdout + r.stderr).strip()[-300:]}', flush=True)
    return r.returncode == 0


def status_lines(lines):
    """Log lines that are safe to print: row events (they carry 'correct') are dropped."""
    return [l for l in lines if '"correct"' not in l and '"event": "row"' not in l]


def download():
    """(parquet path, model snapshot path) at the pinned revisions (cache under HF_HOME)."""
    from huggingface_hub import hf_hub_download, snapshot_download
    pq = hf_hub_download('AI4Math/MathVista', PARQUET, repo_type='dataset', revision=DATA_REV)
    return pq, snapshot_download(MODEL, revision=MODEL_REV)


_ROW = re.compile(r'^\{"pid": "([^"]*)", "method": "([^"]+)"')


class RowCounter:
    """Rows in a growing JSONL file (rows start with {"pid": ..., "method": ...}); reads only new bytes, complete lines only.
    counts: rows per method; status: (pid, method) -> status of its last row (unique pairs; only 'error' is looked at)."""

    def __init__(self, path):
        self.path, self.pos, self.counts, self.status = Path(path), 0, {}, {}

    def update(self):
        if not self.path.exists():
            return self.counts
        with open(self.path, 'rb') as f:
            f.seek(self.pos)
            data = f.read()
        end = data.rfind(b'\n') + 1  # only complete lines
        for line in data[:end].decode('utf-8', 'replace').split('\n'):
            m = _ROW.match(line)
            if m:
                self.counts[m[2]] = self.counts.get(m[2], 0) + 1
                try:
                    st = json.loads(line).get('status')
                except ValueError:
                    st = None
                self.status[(m[1], m[2])] = st
        self.pos += end
        return self.counts

    def n_pairs(self):
        """Unique (pid, method) pairs of the run's methods."""
        ms = set(RUN['methods'].split(','))
        return sum(k[1] in ms for k in self.status)

    def errors(self):
        """Unique pairs whose row has status 'error', per method."""
        out = {}
        for (_, m), st in self.status.items():
            if st == 'error':
                out[m] = out.get(m, 0) + 1
        return out


def progress_line(counts, start_counts, t_start, now, errors=None):
    """Row counts per method, error rows and rows/hour only (never anything about the answers)."""
    total, base = sum(counts.values()), sum(start_counts.values())
    hours = max(now - t_start, 1e-9) / 3600
    per = ' '.join(f'{m}={counts.get(m, 0)}' for m in RUN['methods'].split(','))
    errors = errors or {}
    err = f'errors {sum(errors.values())}' + (' (' + ' '.join(f'{m}={n}' for m, n in sorted(errors.items())) + ')' if errors else '')
    return f'[{time.strftime("%H:%M:%S")} +{now - t_start:.0f}s] rows {total}: {per} | {err} | {(total - base) / hours:.1f} rows/h this start'


def make_config(state, probe, tinfo, packages, ids_sha, store_sha, allow_dirty, hf_home, gpu):
    return {'run': RUN, 'head': state['head'], 'dirty_files': state['dirty_files'], 'allow_dirty': allow_dirty,
            'code_version': probe['code_version'], 'run_py_defaults_seen': {k: probe['defaults'][k] for k in RUN['run_py_defaults']},
            'source_sha256': source_hashes(), 'ids_sha256': ids_sha, 'store_sha256': store_sha,
            'versions': {'python': sys.version, 'torch': tinfo['torch'], 'cuda_build': tinfo['cuda'], 'packages': packages},
            'hf_home': hf_home, 'gpu': gpu, 'model': MODEL, 'model_rev': MODEL_REV, 'data_rev': DATA_REV, 'parquet': PARQUET}


def same_experiment(old, new):
    keys = ('run', 'head', 'code_version', 'ids_sha256', 'store_sha256', 'source_sha256')
    return [k for k in keys if old.get(k) != new.get(k)]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--plan', choices=sorted(PLANS), default='imgtag_main', help='which settings dict to run (default imgtag_main)')
    ap.add_argument('--name', required=True, help='run folder name, e.g. imgtag_main (a short check run: imgtag_check)')
    ap.add_argument('--out-root', required=True, help='parent folder of <name>, outside the git clone (e.g. /workspace/runs)')
    ap.add_argument('--limit', type=int, default=0, help="only the first N questions of the order (check run; not with a name "
                    "starting with the plan's main_name)")
    ap.add_argument('--hours', type=float, default=0, help='stop admitting jobs / dump in-flight jobs before this many hours')
    ap.add_argument('--allow-dirty', action='store_true', help='run on a tree with uncommitted changes (recorded)')
    ap.add_argument('--dry-run', action='store_true', help='print the run.py command and run_config; download and run nothing')
    ap.add_argument('--max-errors', type=int, default=10, help="stop run.py (exit 3, no summary) once more than N rows have status "
                    "'error'; 0 = stop at the first error row (default 10)")
    ap.add_argument('--redo-errors', action='store_true', help="before starting, move the 'error' rows of results_0.jsonl to "
                    'results_0.errors_<UTC timestamp>.jsonl so that run.py redoes them')
    ap.add_argument('--stop-pod', action='store_true', help='when the runner is done (summary or not), run '
                    '`runpodctl stop pod $RUNPOD_POD_ID` (default off)')
    a = ap.parse_args(argv)
    assert a.max_errors >= 0, '--max-errors must be >= 0'
    return a


def main(argv=None):
    a = parse_args(argv)
    ctx = {}  # locked: this runner got the lock (so a pod stop cannot kill another runner); signal: killed by a signal
    try:
        return run(a, ctx)
    finally:
        if a.stop_pod and not a.dry_run and ctx.get('locked') and not ctx.get('signal'):
            stop_pod()


def run(a, ctx):
    t0 = time.time()
    select_plan(a.plan)
    check_name(a.name, a.limit)
    out_root = check_out_root(a.out_root)
    warn = workspace_warning(out_root)
    if warn:
        print(warn, flush=True)
    run_dir = out_root / a.name
    res = run_dir / 'output' / 'results'
    hf_src = 'env' if os.environ.get('HF_HOME') else 'default <out-root>/hf_cache'
    os.environ.setdefault('HF_HOME', str(out_root / 'hf_cache'))  # before huggingface_hub is imported (here and in the child)
    hf_home = os.environ['HF_HOME']
    ids, ids_sha, store_sha = check_inputs()
    state = repo_state()
    if state['dirty_files'] and not (a.allow_dirty or a.dry_run):
        sys.exit(f'refusing to run on a dirty tree ({len(state["dirty_files"])} changed files, e.g. {state["dirty_files"][:3]}); '
                 'commit/stash, or pass --allow-dirty')
    pins = required_packages()
    lock = child = None
    out_jsonl = res / 'results_0.jsonl'
    if not a.dry_run:
        lock = acquire_lock(run_dir / '.lock')  # held until the process exits
        ctx['locked'] = True

        def on_signal(signum, frame):
            ctx['signal'] = True
            terminate_child(child)
            sys.exit(128 + signum)
        for sg in SIGNALS:
            signal.signal(sg, on_signal)
    packages, specs = ensure_packages(pins, a.dry_run)
    if a.dry_run and specs:
        print(f'[dry-run] would pip install: {" ".join(specs)}')
    tinfo = probe_torch()
    check_torch(tinfo['torch'])
    probe = probe_run_py()
    assert probe['code_version'] == RUN['code_version'], f'code_version {probe["code_version"]} != pinned {RUN["code_version"]}'
    for k, v in RUN['run_py_defaults'].items():
        assert probe['defaults'][k] == v, f'run.py default {k}={probe["defaults"][k]} != recorded {v}'
    cfg = make_config(state, probe, tinfo, packages, ids_sha, store_sha, a.allow_dirty,
                      {'path': hf_home, 'source': hf_src}, gpu_name())
    if a.dry_run:
        cmd = build_cmd('<parquet at DATA_REV>', res / 'ids.json', out_jsonl, a.hours or None, a.limit)
        print('[dry-run] cwd:', REPRO)
        print(f'[dry-run] env: PYTORCH_CUDA_ALLOC_CONF={RUN["cuda_alloc_conf"]} HF_HOME={hf_home}')
        print('[dry-run] command:', shlex.join(cmd))
        print('[dry-run] run_config:', json.dumps(cfg, indent=1, ensure_ascii=False))
        return 0

    pq, _ = download()
    res.mkdir(parents=True, exist_ok=True)
    cfg['started_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    cfg['setup_s'] = round(time.time() - t0, 1)
    torn = repair_torn_line(out_jsonl)
    rc_path = run_dir / 'run_config.json'
    old = json.loads(rc_path.read_text()) if rc_path.exists() else None
    if old is not None:  # restart: must be the same experiment (checked before any row is moved)
        diff = same_experiment(old, cfg)
        assert not diff, f'{rc_path} was written by a different setup ({diff}); use a new --name'
    redo = move_error_rows(out_jsonl) if a.redo_errors else None
    if redo:
        print(f'--redo-errors: moved {redo["moved"]} error rows to {redo["file"]}', flush=True)
    notes = {k: v for k, v in (('torn_last_line', torn), ('redo_errors', redo)) if v}
    if old is not None:  # add an entry
        entry = {k: cfg[k] for k in ('started_at', 'versions', 'hf_home', 'gpu', 'setup_s')} | {
            'limit': a.limit, 'hours': a.hours, 'max_errors': a.max_errors} | notes
        old.setdefault('restarts', []).append(entry)
        rc_path.write_text(json.dumps(old, indent=1, ensure_ascii=False))
    else:
        cfg.update(limit=a.limit, hours=a.hours, max_errors=a.max_errors, restarts=[])
        if notes:
            cfg['restarts'].append({'started_at': cfg['started_at']} | notes)
        rc_path.write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
    env = {'torch': tinfo['torch'], 'cuda_build': tinfo['cuda'], 'transformers': packages['transformers'], 'packages': packages,
           'python': sys.version, 'gpus': [cfg['gpu']], 'hf_home': hf_home,
           'cfg': {'model': MODEL, 'model_rev': MODEL_REV, 'data_rev': DATA_REV, 'parquet': PARQUET,
                   'ids': ids, 'run_args': RUN['run_args'], 'skip': []}, 'setup_s': cfg['setup_s']}
    if not (res / 'env.json').exists():
        (res / 'env.json').write_text(json.dumps(env, indent=1))
    (res / 'ids.json').write_text(json.dumps(ids))
    cmd = build_cmd(pq, res / 'ids.json', out_jsonl, a.hours or None, a.limit, elapsed_s=time.time() - t0)
    print('run.py command:', shlex.join(cmd), flush=True)

    counter = RowCounter(out_jsonl)
    start_counts = dict(counter.update())
    t_start = last_print = time.time()
    aborted = None
    with open(res / 'log_0.txt', 'a') as log:
        log.write(f'\n=== start {cfg["started_at"]} HEAD {state["head"]} ===\n')
        log.flush()
        child = subprocess.Popen(cmd, cwd=REPRO, stdout=log, stderr=subprocess.STDOUT,
                                 env={**os.environ, 'PYTORCH_CUDA_ALLOC_CONF': RUN['cuda_alloc_conf']})
        try:
            while True:
                try:
                    child.wait(timeout=POLL_S)
                    break
                except subprocess.TimeoutExpired:
                    counter.update()
                    n_err = sum(counter.errors().values())
                    if too_many_errors(n_err, a.max_errors):
                        aborted = f'{n_err} error rows > --max-errors {a.max_errors}'
                        terminate_child(child)
                        break
                    if time.time() - last_print >= RUN['progress_every_s']:
                        last_print = time.time()
                        print(progress_line(counter.counts, start_counts, t_start, last_print, counter.errors()), flush=True)
        except BaseException:
            terminate_child(child)  # rows are flushed one by one: a rerun resumes
            raise
    print(progress_line(counter.update(), start_counts, t_start, time.time(), counter.errors()), flush=True)
    n_err = sum(counter.errors().values())
    if not aborted and too_many_errors(n_err, a.max_errors):  # errors written in the last poll interval
        aborted = f'{n_err} error rows > --max-errors {a.max_errors} (found after run.py exited)'
    tail = status_lines((res / 'log_0.txt').read_text().splitlines())[-15:]
    print('run.py exit', child.returncode, '\n', '\n'.join(tail), flush=True)
    if aborted:
        print(f'STOPPED: {aborted}; run.py was terminated, no summary written. Look at the error rows (status "error", field '
              f'"error") and the log, fix the cause, then re-run with --redo-errors to redo them.', flush=True)
        return EXIT_MAX_ERRORS
    ready, msg = summary_decision(counter.n_pairs(), a.limit)
    print(msg, flush=True)
    if ready and not RUN['summarize']:
        print('this plan does not use accuracy: summary not written', flush=True)
    elif ready and out_jsonl.exists() and out_jsonl.stat().st_size:
        with open(res / 'summary_stdout.txt', 'w') as so:  # summarize prints accuracies: keep them off the screen
            r = subprocess.run([sys.executable, 'summarize.py', str(out_jsonl), '--out', str(res / 'summary.json'),
                                '--order', str(res / 'ids.json'), '--shards', '1'], cwd=REPRO, stdout=so, stderr=subprocess.STDOUT)
        print('summary written' if r.returncode == 0 else f'summarize.py failed (exit {r.returncode}); see summary_stdout.txt')
    return child.returncode


if __name__ == '__main__':
    sys.exit(main())
