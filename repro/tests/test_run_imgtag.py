"""runpod/run_imgtag.py: fixed settings, input files, and the run.py command (CPU, no network, no subprocess run)."""
import hashlib
import importlib.util
import inspect
import json
import pathlib
import sys

import pytest

REPRO = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location('run_imgtag', REPRO / 'runpod' / 'run_imgtag.py')
ri = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ri)
sys.path.insert(0, str(REPRO))
import run  # noqa: E402

MAIN_PIN = ri.RUN['code_version']  # A6's pin (commit 8f5c09e), read before the fixture below changes it


@pytest.fixture(autouse=True)
def _current_code_version(monkeypatch):
    """The flow tests exercise the runner on the current code: the imgtag_main pin is set to the current code_version
    for each test (restored afterwards), and RUN is restored after main() selected another plan."""
    monkeypatch.setattr(ri, 'RUN', ri.RUN)
    monkeypatch.setitem(ri.PLANS['imgtag_main'], 'code_version', run.code_version())


def test_run_dict_values():
    R = ri.RUN
    assert R['methods'] == 'visref_imgtag,tsr_answer_prompt,tsr'
    args = R['run_args']
    assert args[:2] == ['--methods', R['methods']]
    pairs = {args[i]: args[i + 1] for i in range(0, len(args), 2)}
    assert pairs == {'--methods': R['methods'], '--think-cap': '1024', '--total-think-cap': '10240',
                     '--step1-from': 'diag/step1_store_bmain.json', '--dual-probe': '1'}
    assert R['shard'] == '0/1' and R['transformers'] == '4.55.0'
    assert not {'--dtype', '--batch', '--batch-tokens', '--chunk', '--fast-kv', '--seed'} & set(args)


def test_constants_come_from_build():
    import build
    assert (ri.MODEL, ri.MODEL_REV, ri.DATA_REV, ri.PARQUET) == (build.MODEL, build.MODEL_REV, build.DATA_REV, build.PARQUET)


def test_ids_file_and_store_sha():
    ids_path, store_path = REPRO / ri.RUN['ids_file'], REPRO / ri.RUN['store_file']
    ids = json.loads(ids_path.read_text())
    assert len(ids) == 1000 == ri.RUN['n_ids']
    assert hashlib.sha256(ids_path.read_bytes()).hexdigest() == \
        'a5ad8da9d10e7991abad84ca75dd108d60f94b890c40600d9dfa5ae2eb31b030' == ri.RUN['ids_sha256']
    assert hashlib.sha256(store_path.read_bytes()).hexdigest().startswith('f937478f3b26')
    assert ri.RUN['store_sha256_prefix'] == 'f937478f3b26'
    assert ri.check_inputs()[0] == [str(i) for i in ids]


def test_pinned_code_version_and_defaults_match_run_py():
    assert MAIN_PIN == '6e218887d2b5'  # A6, commit 8f5c09e: that plan runs only from that commit
    assert ri.FORMAT_CHECK['code_version'] == run.code_version()  # the format check runs on the current code
    for k, v in ri.RUN['run_py_defaults'].items():
        assert run.DEFAULTS[k] == v
    assert run.DEFAULTS['dual_probe'] == 0  # the runner must pass --dual-probe 1 itself


def test_dtype_default_is_fp16():
    import torch
    from visref.model import load_model
    assert inspect.signature(load_model).parameters['dtype'].default == torch.float16


def test_build_cmd_dry_run_argv():
    cmd = ri.build_cmd('/pq/x.parquet', '/o/ids.json', '/o/results_0.jsonl', python='python3')
    assert cmd == ['python3', 'run.py', '--parquet', '/pq/x.parquet', '--model', 'OpenGVLab/InternVL3_5-1B',
                   '--revision', '2f71cf52542334823e48a46ffba0e2bc9add3446', '--ids', '/o/ids.json', '--shard', '0/1',
                   '--out', '/o/results_0.jsonl', '--methods', 'visref_imgtag,tsr_answer_prompt,tsr',
                   '--think-cap', '1024', '--total-think-cap', '10240', '--step1-from', 'diag/step1_store_bmain.json',
                   '--dual-probe', '1']


def test_build_cmd_limit_and_hours():
    cmd = ri.build_cmd('p', 'i', 'o', hours=8, limit=20, elapsed_s=100, python='python3')
    assert cmd[cmd.index('--time-limit') + 1] == '26000' and cmd[cmd.index('--deadline') + 1] == '28100'
    assert cmd[cmd.index('--limit') + 1] == '20'
    assert cmd[-len(ri.RUN['run_args']):] == ri.RUN['run_args']
    short = ri.build_cmd('p', 'i', 'o', hours=0.1, python='python3')  # margins shrink for a short run
    assert 0 < float(short[short.index('--time-limit') + 1]) < float(short[short.index('--deadline') + 1]) < 360


def test_argv_is_accepted_by_run_py_parser():
    """Every flag in the command is one run.py defines (checked on its source, no model loaded)."""
    src = (REPRO / 'run.py').read_text()
    cmd = ri.build_cmd('p', 'i', 'o', hours=1, limit=5)
    for flag in [c for c in cmd if c.startswith('--')]:
        name = flag[2:].replace('-', '_')
        assert f"'{flag}'" in src or name in run.DEFAULTS, flag


def test_row_counter_and_progress(tmp_path):
    f = tmp_path / 'r.jsonl'
    f.write_text('{"pid": "1", "method": "tsr", "x": 1}\n{"pid": "1", "method": "visref_imgtag", "x": 1}\n{"pid": "2", "meth')
    c = ri.RowCounter(f)
    assert c.update() == {'tsr': 1, 'visref_imgtag': 1}  # the half-written last line is not counted
    with open(f, 'a') as h:
        h.write('od": "tsr"}\n')
    assert c.update() == {'tsr': 2, 'visref_imgtag': 1}
    line = ri.progress_line(c.counts, {'tsr': 1}, 0, 1800)
    assert 'rows 3' in line and 'tsr=2' in line and '4.0 rows/h' in line


def test_dry_run_prints_command_without_side_effects(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv('HF_HOME', 'x')
    monkeypatch.delenv('HF_HOME')  # main() sets it; monkeypatch removes it again afterwards
    rc = ri.main(['--name', 'imgtag_check', '--out-root', str(tmp_path), '--dry-run', '--limit', '7'])
    out = capsys.readouterr().out
    assert rc == 0 and 'run.py --parquet' in out and '--dual-probe 1' in out and '--limit 7' in out
    assert f'"code_version": "{run.code_version()}"' in out
    assert list(tmp_path.iterdir()) == [] and f'HF_HOME={tmp_path.resolve()}/hf_cache' in out


def test_restart_must_be_same_experiment():
    base = {'run': ri.RUN, 'head': 'a', 'code_version': 'v', 'ids_sha256': 'i', 'store_sha256': 's', 'source_sha256': {}}
    assert ri.same_experiment(base, dict(base)) == []
    assert ri.same_experiment(base, dict(base, head='b')) == ['head']
    assert ri.same_experiment(base, dict(base, run=dict(ri.RUN, methods='tsr'))) == ['run']


# ---- robustness (review fixes) ----

def test_torn_last_line_is_moved_aside(tmp_path):
    f = tmp_path / 'results_0.jsonl'
    f.write_text('{"pid": "1", "method": "tsr"}\n{"pid": "2", "method": "tsr"}\n{"pid": "3", "meth')
    note = ri.repair_torn_line(f, stamp='T1')
    assert note == {'torn_file': 'results_0.jsonl.torn.T1', 'torn_bytes': len('{"pid": "3", "meth')}
    assert f.read_text() == '{"pid": "1", "method": "tsr"}\n{"pid": "2", "method": "tsr"}\n'
    assert (tmp_path / 'results_0.jsonl.torn.T1').read_text() == '{"pid": "3", "meth'
    assert ri.repair_torn_line(f, stamp='T2') is None  # now clean
    # a single torn line empties the file; a valid last line without newline gets one; missing / empty file: nothing
    g = tmp_path / 'g.jsonl'
    g.write_text('{"pid": "1", "met')
    assert ri.repair_torn_line(g, stamp='T')['torn_bytes'] == 17 and g.read_text() == ''
    h = tmp_path / 'h.jsonl'
    h.write_text('{"pid": "1"}\n{"pid": "2"}')
    assert ri.repair_torn_line(h) == {'newline_added': True} and h.read_text() == '{"pid": "1"}\n{"pid": "2"}\n'
    assert ri.repair_torn_line(tmp_path / 'none.jsonl') is None and ri.repair_torn_line(tmp_path / 'g.jsonl') is None


def test_out_root_inside_the_repository_is_refused(tmp_path, capsys):
    inside = ri.ROOT / 'zz_runs_do_not_create'
    with pytest.raises(SystemExit, match='inside the repository'):
        ri.check_out_root(inside)
    with pytest.raises(SystemExit):
        ri.check_out_root(ri.ROOT)
    with pytest.raises(SystemExit):
        ri.main(['--name', 'imgtag_check', '--out-root', str(inside / 'x'), '--dry-run'])
    assert not inside.exists()
    link = tmp_path / 'link'
    link.symlink_to(ri.REPRO)  # resolved before the comparison
    with pytest.raises(SystemExit):
        ri.check_out_root(link / 'runs')
    assert ri.check_out_root(tmp_path / 'ok') == (tmp_path / 'ok').resolve()


def test_second_runner_on_the_same_lock_refuses(tmp_path):
    lock = ri.acquire_lock(tmp_path / 'run' / '.lock')
    with pytest.raises(SystemExit, match='another runner holds'):
        ri.acquire_lock(tmp_path / 'run' / '.lock')
    lock.close()  # released with the file
    ri.acquire_lock(tmp_path / 'run' / '.lock').close()


def test_check_run_name_guard():
    with pytest.raises(SystemExit, match='imgtag_check'):
        ri.check_name('imgtag_main_2', 3)
    ri.check_name('imgtag_check', 3)
    ri.check_name('imgtag_main', 0)


def test_log_tail_filter_drops_correctness_lines():
    lines = ['{"event": "start", "todo": 3000}', '{"event": "row", "pid": "1", "method": "tsr", "correct": 1, "k": 2}',
             '{"event": "round", "elapsed": 5, "tokens": 100}', 'x "correct": 0 y', '{"event": "row", "pid": "2"}',
             'Traceback (most recent call last):']
    got = ri.status_lines(lines)
    assert got == [lines[0], lines[2], lines[5]] and not any('correct' in l for l in got)


def test_progress_line_has_only_counts_and_rate():
    line = ri.progress_line({'tsr': 2, 'visref_imgtag': 3}, {}, 0, 3600)
    assert 'correct' not in line and 'acc' not in line and 'rows 5' in line and '5.0 rows/h' in line


def test_gpu_name_does_not_touch_torch_in_process(monkeypatch):
    calls = []

    class Done:
        def __init__(self, out):
            self.stdout = out

    def fake(argv, **kw):
        calls.append(argv)
        if argv[0] == 'nvidia-smi':
            assert argv[1:] == ['--query-gpu=name,memory.total', '--format=csv,noheader']
            return Done('NVIDIA A100 80GB, 81920 MiB\n')
        return Done('Fallback GPU\n')

    monkeypatch.setitem(sys.modules, 'torch', None)  # an in-process `import torch` would raise
    monkeypatch.setattr(ri.subprocess, 'run', fake)
    assert ri.gpu_name() == 'NVIDIA A100 80GB, 81920 MiB' and len(calls) == 1

    def no_smi(argv, **kw):
        if argv[0] == 'nvidia-smi':
            raise FileNotFoundError
        calls.append(argv)
        return Done('Fallback GPU\n')
    monkeypatch.setattr(ri.subprocess, 'run', no_smi)
    assert ri.gpu_name() == 'Fallback GPU' and calls[-1][1] == '-c' and 'torch.cuda.get_device_name(0)' in calls[-1][2]
    src = (REPRO / 'runpod' / 'run_imgtag.py').read_text()
    assert not [l for l in src.splitlines() if l.startswith(('import torch', 'from torch'))]  # only inside `-c` strings


def test_terminate_child_terminates_then_kills(tmp_path):
    import subprocess
    p = subprocess.Popen(['sleep', '60'])
    ri.terminate_child(p, grace=5)
    assert p.poll() is not None
    ready = tmp_path / 'ready'
    stub = ('import signal, time, pathlib; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
            f'pathlib.Path({str(ready)!r}).write_text("x"); time.sleep(60)')
    q = subprocess.Popen([sys.executable, '-c', stub])
    import time
    for _ in range(100):
        if ready.exists():
            break
        time.sleep(0.1)
    ri.terminate_child(q, grace=0.5)  # ignores SIGTERM -> killed
    assert q.poll() is not None
    ri.terminate_child(q)  # already gone: no error
    ri.terminate_child(None)


def test_requirements_and_package_check(monkeypatch):
    reqs = ri.parse_requirements()
    names = dict(reqs)
    assert names['torch'] == '2.11.0' and names['transformers'] == '4.55.0' and 'timm' in names and 'einops' in names
    pins = ri.required_packages()
    assert pins['transformers'] == ri.RUN['transformers']
    have = {'torch': '2.4.0', 'transformers': '4.50.0', 'timm': None, 'pandas': '3.0.5'}
    monkeypatch.setattr(ri, 'installed', lambda n: have.get(n, '1.0'))
    specs = ri.missing_packages({'torch': '2.11.0', 'transformers': '4.55.0', 'timm': None, 'pandas': None, 'einops': None})
    assert specs == ['transformers==4.55.0', 'timm']  # torch is never installed; unpinned present packages are kept
    with pytest.raises(AssertionError, match='disagree'):
        ri.required_packages([('transformers', '4.0.0')])


def test_torch_version_floor():
    ri.check_torch('2.5.0')
    ri.check_torch('2.11.0+cu128')
    with pytest.raises(AssertionError, match='OutOfMemoryError'):
        ri.check_torch('2.4.1+cu121')


def _stub_rows_code(pairs):
    """Child stub: appends the given (pid, method[, status]) rows like run.py does, prints a row event and a round event."""
    return ("import sys, json; f = open(sys.argv[1], 'a'); "
            f"[f.write(json.dumps(dict(zip(('pid', 'method', 'status'), r), correct=1)) + '\\n') for r in {pairs!r}]; f.close(); "
            "print('{\"event\": \"row\", \"pid\": \"1\", \"correct\": 1}'); print('{\"event\": \"round\", \"tokens\": 3}')")


def _flow_setup(monkeypatch, code):
    monkeypatch.setenv('HF_HOME', 'x')
    monkeypatch.delenv('HF_HOME')
    monkeypatch.setattr(ri, 'download', lambda: ('pq.parquet', 'model'))
    monkeypatch.setattr(ri, 'ensure_packages', lambda pins, dry_run=False: ({'transformers': '4.55.0'}, []))
    monkeypatch.setattr(ri, 'gpu_name', lambda: 'Fake GPU, 1 MiB')
    monkeypatch.setattr(ri, 'build_cmd', lambda pq, ids, out, *a, **k: [sys.executable, '-c', code, str(out)])


ALL6 = [(p, m, 'ok') for p in ('1', '2') for m in ('visref_imgtag', 'tsr_answer_prompt', 'tsr')]


def test_main_flow_with_a_stub_child(tmp_path, capsys, monkeypatch):
    """Whole non-dry flow with download/pip/child replaced: files, torn-line repair, restart entry, nothing about correctness printed."""
    _flow_setup(monkeypatch, _stub_rows_code(ALL6[1:]))  # (1, visref_imgtag) is already in the file below: 6 unique pairs
    res = tmp_path / 'imgtag_check' / 'output' / 'results'
    res.mkdir(parents=True)
    (res / 'results_0.jsonl').write_text('{"pid": "1", "method": "visref_imgtag"}\n{"pid": "8", "me')
    argv = ['--name', 'imgtag_check', '--out-root', str(tmp_path), '--allow-dirty', '--limit', '2']
    assert ri.main(argv) == 0
    out = '\n'.join(l for l in capsys.readouterr().out.splitlines() if not l.startswith('run.py command'))  # the stub's code
    assert '"correct"' not in out and 'accuracy' not in out and '"event": "row"' not in out
    assert '"event": "round"' in out and 'complete: 6/6 rows' in out and 'incomplete' not in out and 'errors 0' in out
    assert len(list(res.glob('results_0.jsonl.torn.*'))) == 1
    cfg = json.loads((tmp_path / 'imgtag_check' / 'run_config.json').read_text())
    assert cfg['restarts'][0]['torn_last_line']['torn_bytes'] == len('{"pid": "8", "me') and cfg['gpu'] == 'Fake GPU, 1 MiB'
    assert cfg['hf_home']['path'] == str(tmp_path.resolve() / 'hf_cache') and cfg['versions']['torch'] and 'cuda_build' in cfg['versions']
    assert json.loads((res / 'ids.json').read_text()) == [str(i) for i in json.loads((REPRO / ri.RUN['ids_file']).read_text())]
    assert (res / 'env.json').exists() and (res / 'summary_stdout.txt').exists() and (tmp_path / 'imgtag_check' / '.lock').exists()
    assert '=== start' in (res / 'log_0.txt').read_text()
    assert ri.main(argv) == 0  # restart: same setup accepted, one more entry, lock was released (duplicate rows count once)
    assert len(json.loads((tmp_path / 'imgtag_check' / 'run_config.json').read_text())['restarts']) == 2


def test_incomplete_run_writes_no_summary(tmp_path, capsys, monkeypatch):
    _flow_setup(monkeypatch, _stub_rows_code(ALL6[:4]))
    argv = ['--name', 'imgtag_check', '--out-root', str(tmp_path), '--allow-dirty', '--limit', '2']
    assert ri.main(argv) == 0
    out = capsys.readouterr().out
    res = tmp_path / 'imgtag_check' / 'output' / 'results'
    assert 'incomplete: 4/6 rows; summary not written' in out
    assert not (res / 'summary.json').exists() and not (res / 'summary_stdout.txt').exists()


def test_max_errors_terminates_child_and_skips_summary(tmp_path, capsys, monkeypatch):
    code = _stub_rows_code([('1', 'tsr', 'ok'), ('1', 'visref_imgtag', 'error')]).replace('f.close();', 'f.close(); sys.stdout.flush(); import time; time.sleep(120);')
    _flow_setup(monkeypatch, code)
    monkeypatch.setattr(ri, 'POLL_S', 0.2)
    argv = ['--name', 'imgtag_check', '--out-root', str(tmp_path), '--allow-dirty', '--limit', '2', '--max-errors', '0']
    t = __import__('time').time()
    assert ri.main(argv) == ri.EXIT_MAX_ERRORS == 3
    assert __import__('time').time() - t < 60  # the sleeping child was terminated
    out = capsys.readouterr().out
    assert 'STOPPED: 1 error rows > --max-errors 0' in out and 'errors 1 (visref_imgtag=1)' in out
    assert not (tmp_path / 'imgtag_check' / 'output' / 'results' / 'summary.json').exists()


def test_redo_errors_in_main_flow(tmp_path, capsys, monkeypatch):
    _flow_setup(monkeypatch, _stub_rows_code(ALL6[1:]))  # (1, tsr_answer_prompt) is redone, (1, visref_imgtag) kept
    res = tmp_path / 'imgtag_check' / 'output' / 'results'
    res.mkdir(parents=True)
    (res / 'results_0.jsonl').write_text('{"pid": "1", "method": "visref_imgtag", "status": "ok"}\n'
                                         '{"pid": "1", "method": "tsr_answer_prompt", "status": "error"}\n')
    argv = ['--name', 'imgtag_check', '--out-root', str(tmp_path), '--allow-dirty', '--limit', '2', '--redo-errors']
    assert ri.main(argv) == 0  # the stub writes the 5 pairs that are missing once the error row is gone
    cfg = json.loads((tmp_path / 'imgtag_check' / 'run_config.json').read_text())
    note = cfg['restarts'][0]['redo_errors']
    assert note['moved'] == 1 and (res / note['file']).exists() and note['file'].startswith('results_0.errors_')
    assert 'complete: 6/6 rows' in capsys.readouterr().out and (res / 'summary_stdout.txt').exists()  # summarize.py was started


def test_expected_rows_and_summary_decision():
    assert ri.expected_rows(0) == 3000 and ri.expected_rows(20) == 60 and ri.expected_rows(5000) == 3000
    assert ri.summary_decision(60, 20) == (True, 'complete: 60/60 rows')
    ok, msg = ri.summary_decision(59, 20)
    assert not ok and msg == 'incomplete: 59/60 rows; summary not written'
    assert not ri.summary_decision(3001, 0)[0] and ri.summary_decision(3000)[0]


def test_row_counter_unique_pairs_errors_and_torn_line(tmp_path):
    f = tmp_path / 'r.jsonl'
    f.write_text('{"pid": "1", "method": "tsr", "status": "ok"}\n{"pid": "1", "method": "tsr", "status": "ok"}\n'
                 '{"pid": "2", "method": "visref_imgtag", "status": "error", "error": "x"}\n'
                 '{"pid": "3", "method": "tsr", "status": "er')
    c = ri.RowCounter(f)
    c.update()
    assert c.counts == {'tsr': 2, 'visref_imgtag': 1} and c.n_pairs() == 2  # duplicate counts once, torn last line ignored
    assert c.errors() == {'visref_imgtag': 1}
    with open(f, 'a') as h:
        h.write('ror"}\n')
    c.update()
    assert c.errors() == {'visref_imgtag': 1, 'tsr': 1} and c.n_pairs() == 3
    line = ri.progress_line(c.counts, {}, 0, 3600, c.errors())
    assert 'errors 2 (tsr=1 visref_imgtag=1)' in line and 'correct' not in line
    assert 'errors 0' in ri.progress_line({}, {}, 0, 60)


@pytest.mark.parametrize('n, mx, want', [(0, 0, False), (1, 0, True), (10, 10, False), (11, 10, True), (3, 10, False)])
def test_too_many_errors(n, mx, want):
    assert ri.too_many_errors(n, mx) is want


def test_move_error_rows(tmp_path):
    f = tmp_path / 'results_0.jsonl'
    ok1, bad, ok2 = ('{"pid": "1", "method": "tsr", "status": "ok"}', '{"pid": "2", "method": "tsr", "status": "error"}',
                     '{"pid": "3", "method": "tsr"}')
    f.write_text('\n'.join([ok1, bad, 'not json', ok2]) + '\n')
    note = ri.move_error_rows(f, stamp='20261015T000000Z')
    assert note == {'moved': 1, 'file': 'results_0.errors_20261015T000000Z.jsonl'}
    assert f.read_text() == '\n'.join([ok1, 'not json', ok2]) + '\n'
    assert (tmp_path / note['file']).read_text() == bad + '\n' and not (tmp_path / 'results_0.jsonl.tmp').exists()
    assert ri.move_error_rows(f) == {'moved': 0, 'file': None} and f.read_text().count('\n') == 3  # nothing to move: untouched
    assert ri.move_error_rows(tmp_path / 'none.jsonl') == {'moved': 0, 'file': None}
    c = ri.RowCounter(f)
    c.update()
    assert c.errors() == {}  # run.py would now see (2, tsr) as missing and redo it


def test_workspace_warning():
    assert ri.workspace_warning('/workspace/runs', {'RUNPOD_POD_ID': 'p'}) is None
    assert ri.workspace_warning('/workspace', {'RUNPOD_POD_ID': 'p'}) is None
    assert ri.workspace_warning('/tmp/runs', {}) is None  # not on a pod: no warning
    w = ri.workspace_warning('/root/runs', {'RUNPOD_POD_ID': 'p'})
    assert 'not under /workspace' in w and 'lost' in w and ri.workspace_warning('/workspacex/runs', {'RUNPOD_POD_ID': 'p'})


def test_stop_pod_uses_runpodctl_without_shell(monkeypatch, capsys):
    calls = []

    class Done:
        returncode, stdout, stderr = 0, 'stopping\n', ''

    def fake_run(argv, **kw):
        calls.append((argv, kw))
        return Done()
    monkeypatch.setattr(ri.subprocess, 'run', fake_run)
    monkeypatch.setattr(ri.shutil, 'which', lambda n: '/usr/bin/runpodctl')
    assert ri.stop_pod({'RUNPOD_POD_ID': 'abc123'}) is True
    assert calls[0][0] == ['runpodctl', 'stop', 'pod', 'abc123'] and not calls[0][1].get('shell')
    # no pod id / no runpodctl: warning, no call
    calls.clear()
    assert ri.stop_pod({}) is False and 'RUNPOD_POD_ID is not set' in capsys.readouterr().out
    monkeypatch.setattr(ri.shutil, 'which', lambda n: None)
    assert ri.stop_pod({'RUNPOD_POD_ID': 'abc123'}) is False and 'runpodctl was not found' in capsys.readouterr().out
    assert calls == []


def test_stop_pod_flag_in_main(tmp_path, monkeypatch):
    _flow_setup(monkeypatch, _stub_rows_code(ALL6))
    stopped = []
    monkeypatch.setattr(ri, 'stop_pod', lambda *a, **k: stopped.append(1))
    base = ['--name', 'imgtag_check', '--out-root', str(tmp_path), '--allow-dirty', '--limit', '2']
    assert ri.main(base) == 0 and stopped == []  # default off
    assert ri.main(base + ['--stop-pod']) == 0 and stopped == [1]
    assert ri.main(base + ['--stop-pod', '--dry-run']) == 0 and stopped == [1]  # dry run never stops the pod
    lock = ri.acquire_lock(tmp_path / 'imgtag_check' / '.lock')
    with pytest.raises(SystemExit, match='another runner holds'):
        ri.main(base + ['--stop-pod'])
    assert stopped == [1]  # another runner owns the pod: not stopped
    lock.close()


def test_new_flags_defaults_and_help(capsys):
    a = ri.parse_args(['--name', 'n', '--out-root', 'o'])
    assert (a.max_errors, a.redo_errors, a.stop_pod) == (10, False, False)
    with pytest.raises(SystemExit):
        ri.parse_args(['-h'])
    h = capsys.readouterr().out
    assert '--max-errors' in h and '--redo-errors' in h and '--stop-pod' in h
    with pytest.raises(AssertionError):
        ri.parse_args(['--name', 'n', '--out-root', 'o', '--max-errors', '-1'])


# ---- format check plan (2026-10-17) ----------------------------------------------------------------------------------
def test_format_check_plan_values():
    F, R = ri.FORMAT_CHECK, ri.PLANS['imgtag_main']
    assert ri.PLANS['format_check'] is F and F['plan'] == 'format_check' and R['plan'] == 'imgtag_main'
    assert F['methods'] == 'visref_hs_imgtag_in,tsr_answer_prompt'
    assert F['run_args'] == ['--methods', F['methods']] + R['run_args'][2:]  # other run.py arguments as A6
    assert F['n_ids'] == 200 and F['summarize'] is False and F['main_name'] == 'fmtcheck_main' and F['check_name'] == 'fmtcheck_try'
    same = ('store_file', 'store_sha256_prefix', 'shard', 'transformers', 'min_torch', 'run_py_defaults', 'cuda_alloc_conf')
    assert all(F[k] == R[k] for k in same)
    for m in F['methods'].split(','):
        assert m in __import__('visref.methods', fromlist=['METHODS']).METHODS


def test_format_check_ids_file():
    sys.path.insert(0, str(REPRO / 'diag'))
    import format_check_ids as fci
    F = ri.FORMAT_CHECK
    path = REPRO / F['ids_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == F['ids_sha256']
    order = json.loads((REPRO / ri.RUN['ids_file']).read_text())
    store = json.loads((REPRO / F['store_file']).read_text())['pids']
    ids = json.loads(path.read_text())
    assert ids == fci.build(order, store) == [str(p) for p in order[:200]] and len(set(ids)) == 200
    assert fci.main(['--check']) == 0
    ri.select_plan('format_check')
    assert ri.check_inputs()[0] == ids


def test_format_check_build_cmd_and_names():
    ri.select_plan('format_check')
    cmd = ri.build_cmd('p', 'i', 'o', python='python3')
    assert cmd[cmd.index('--methods') + 1] == 'visref_hs_imgtag_in,tsr_answer_prompt'
    assert ri.expected_rows(0) == 400 and ri.expected_rows(3) == 6
    with pytest.raises(SystemExit):
        ri.check_name('fmtcheck_main', 3)
    ri.check_name('fmtcheck_try', 3)
    ri.check_name('imgtag_main', 3)  # only the plan's own main name is refused


def test_format_check_complete_run_writes_no_summary(tmp_path, capsys, monkeypatch):
    order = json.loads((REPRO / ri.FORMAT_CHECK['ids_file']).read_text())
    rows = [(p, m, 'ok') for p in order[:2] for m in ri.FORMAT_CHECK['methods'].split(',')]
    _flow_setup(monkeypatch, _stub_rows_code(rows))
    argv = ['--plan', 'format_check', '--name', 'fmtcheck_try', '--out-root', str(tmp_path), '--allow-dirty', '--limit', '2']
    assert ri.main(argv) == 0
    out = capsys.readouterr().out
    res = tmp_path / 'fmtcheck_try' / 'output' / 'results'
    assert 'complete: 4/4 rows' in out and 'this plan does not use accuracy: summary not written' in out
    assert not (res / 'summary.json').exists() and not (res / 'summary_stdout.txt').exists()
    cfg = json.loads((tmp_path / 'fmtcheck_try' / 'run_config.json').read_text())
    assert cfg['run']['plan'] == 'format_check' and json.loads((res / 'ids.json').read_text())[:2] == order[:2]
