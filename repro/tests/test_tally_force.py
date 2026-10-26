"""CPU tests for diag/run_common.py, diag/tally_force.py, diag/analyze_tally_force.py,
and kaggle/build_tally_force.py (tiny LM / stub model)."""
import ast
import base64
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

REPRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO / 'diag'))
sys.path.insert(0, str(REPRO / 'kaggle'))

import analyze_tally_force as an  # noqa: E402
import build_tally_force as bt  # noqa: E402
import run_common as rc  # noqa: E402
import tally_force as tf  # noqa: E402
from test_repro import lm  # noqa: E402,F401 (pytest fixture)
from test_step1_think import SP as SP0, TOK  # noqa: E402

from visref.engine import Engine, Request  # noqa: E402

SP = {**SP0, 'close_ids': frozenset(i for i, s in enumerate(TOK.vocab) if s.startswith('}'))}
CFG = tf.resolve_cfg({**tf.DEFAULTS, 'think_extra': 5, 'probe_cap': 4, 'n_t1': 2, 'batch_tokens': 4000})


# ---- (a) & (b): Contexts and Seeds ------------------------------------------------------------------
def fake_item(lm, pid='0-0-0', n=6, seed=0, z_str='abc', sep_str='\n'):
    emb = lm.get_input_embeddings()
    g = torch.Generator().manual_seed(seed)
    prompt = emb(torch.randint(5, 90, (n + 4,), generator=g)).detach()
    vis = torch.randn(n, emb.embedding_dim, generator=g)
    z = TOK.encode(z_str) if z_str else []
    sep = TOK.encode(sep_str) if (z_str and sep_str) else []
    it = {
        'pid': pid,
        'shard': 0,
        'row_idx': 0,
        'qi': 0,
        'row': {'question_type': 'free_form', 'answer_type': 'integer', 'answer': '7', 'query': 'q'},
        'prompt': prompt,
        'vis': vis,
        'z': z,
        'sep': sep,
        'sep_text': sep_str if (z_str and sep_str) else '',
        'step1': {'end': 'cap' if sep_str else 'close', 'ids': z, 'sep_text': sep_str if (z_str and sep_str) else ''}
    }
    return it


def test_seed_and_contexts_with_and_without_block(lm):
    """(a) 같은 pid·시드면 N과 D의 이어 생각 시드가 같고, D 문맥에만 블록이 있음
       (b) z가 비면 D에 블록 없음"""
    eng = Engine(lm)
    # Case 1: non-empty z -> D has v_block, N does not
    it = fake_item(lm, pid='0-1-0', n=8, z_str='hello', sep_str='\n')
    m, sel = tf.selections(it, eng, CFG)
    it['m'], it['sel'] = m, sel
    ctxs = tf.contexts(it, eng, TOK, SP)

    assert set(ctxs) == {'N', 'D'}
    pre = torch.cat([it['prompt'], eng.embed(it['z'] + it['sep'])], 0)
    wait = eng.embed(SP['wait'])
    # N is [pre, wait]
    assert torch.equal(ctxs['N'], torch.cat([pre, wait], 0))
    # D has v_block inserted before wait
    block = tf.v_block(eng, TOK, SP, it['vis'], sel['own'], TOK.decode(it['z']), it['sep_text'])
    assert torch.equal(ctxs['D'], torch.cat([pre, block, wait], 0))
    assert not torch.equal(ctxs['N'], ctxs['D'])
    # Both N and D share seed_of(cfg['seed'], pid, 'think')
    s_n = tf.seed_of(CFG['seed'], it['pid'], 'think')
    s_d = tf.seed_of(CFG['seed'], it['pid'], 'think')
    assert s_n == s_d

    # Case 2: (b) empty z -> D has no block (equal to N)
    it_empty = fake_item(lm, pid='0-2-0', n=8, z_str='', sep_str='')
    m_e, sel_e = tf.selections(it_empty, eng, CFG)
    it_empty['m'], it_empty['sel'] = m_e, sel_e
    assert sel_e['own'] == []
    ctxs_empty = tf.contexts(it_empty, eng, TOK, SP)
    assert torch.equal(ctxs_empty['N'], ctxs_empty['D'])


# ---- (f) & (g): Query Format and Integer String Validation ------------------------------------------
def test_query_hint_format():
    """(f) 질문 문장이 위 Hint 형식 그대로"""
    question = 'How many chairs are visible?'
    want = ('Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.\n'
            'Question: How many chairs are visible?')
    got = ('Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.\n'
           'Question: ' + question)
    assert got == want


def test_integer_string_validation_and_load_error():
    """(g) 정답이 정수 문자열이 아니면 load 오류 행"""
    assert tf.is_integer_string('0') is True
    assert tf.is_integer_string('42') is True
    assert tf.is_integer_string('-5') is True
    assert tf.is_integer_string(' 12 ') is True
    assert tf.is_integer_string('3.14') is False
    assert tf.is_integer_string('two') is False
    assert tf.is_integer_string(42) is False
    assert tf.is_integer_string(None) is False
    assert tf.is_integer_string('') is False

    # In process_group: an item raising ValueError in make_item produces an error row with stage 'load'
    def bad_item(pid):
        ans = 'not_int'
        if not tf.is_integer_string(ans):
            raise ValueError(f"answer {ans!r} is not an integer string")

    rows = rc.process_group(['bad_pid'], bad_item, None, None, None, CFG, None, 'v', 'main', None, None)
    assert len(rows) == 1
    assert rows[0]['status'] == 'error'
    assert rows[0]['stage'] == 'load'
    assert 'not an integer string' in rows[0]['error']


# ---- (d) & (e): Analysis Decision, Bootstrap and HOLD -----------------------------------------------
def make_fake_row(pid, n_c_n, n_c_d, status='ok'):
    if status != 'ok':
        return {'pid': pid, 'status': 'error', 'stage': 'load', 'error': 'bad'}
    return {
        'pid': pid,
        'status': 'ok',
        'set': 'main',
        'shard': 0,
        'row': 0,
        'qi': 0,
        'n_visual': 10,
        'm': 3,
        'sel': {'own': [0, 1, 2]},
        'n_sel': {'own': 3},
        'step1': {'end': 'close', 'len': 5, 'ids': [1, 2, 3, 4, 5], 'sep_text': ''},
        'extra': {'N': {'end': 'close', 'len': 5}, 'D': {'end': 'close', 'len': 5}},
        'answers': {
            'N': [{'valid': True, 'correct': int(i < n_c_n), 'greedy': False} for i in range(3)] + [{'valid': True, 'correct': 1, 'greedy': True}],
            'D': [{'valid': True, 'correct': int(i < n_c_d), 'greedy': False} for i in range(3)] + [{'valid': True, 'correct': 1, 'greedy': True}],
        },
        'n_correct': {'N': n_c_n, 'D': n_c_d},
        'greedy': {'N': 1, 'D': 1},
        'answer_gt': '3',
        'code_version': 'v',
        'scorer_version': 'v4_mc_empty_wrong',
        'cfg': CFG,
        'timing': {}
    }


def test_analysis_4_changed_sign_p_0125_and_verdict_not():
    """(d) 분석: 합성 자료에서 4문항만 D 3/3·N 0/3이고 나머지 동률이면 부호 p = 0.125이고 판정 NOT"""
    pids = [f'p{i}' for i in range(20)]
    R = {}
    for i, p in enumerate(pids):
        if i < 4:
            R[p] = make_fake_row(p, n_c_n=0, n_c_d=3)
        else:
            R[p] = make_fake_row(p, n_c_n=2, n_c_d=2)

    res = an.analyze(R, pids)
    assert res['up'] == 4
    assert res['down'] == 0
    assert res['same'] == 16
    assert res['p_sign'] == pytest.approx(0.125)
    assert res['verdict'] == 'NOT'


def test_analysis_all_present_gives_verdict_one_missing_or_error_gives_hold():
    """(e) 분석: 모두 있으면 판정, 하나 빠지면 HOLD"""
    pids = [f'p{i}' for i in range(10)]
    R = {p: make_fake_row(p, n_c_n=1, n_c_d=1) for p in pids}

    # All present and ok -> verdict is evaluated
    res_ok = an.analyze(R, pids)
    assert res_ok['verdict'] in ('UP', 'DOWN', 'NOT')
    assert len(res_ok['missing']) == 0
    assert len(res_ok['errors']) == 0

    # One missing -> HOLD
    R_missing = dict(R)
    del R_missing[pids[-1]]
    res_missing = an.analyze(R_missing, pids)
    assert res_missing['verdict'] == 'HOLD'
    assert pids[-1] in res_missing['missing']

    # One error -> HOLD
    R_error = dict(R)
    R_error[pids[0]] = make_fake_row(pids[0], 0, 0, status='error')
    res_error = an.analyze(R_error, pids)
    assert res_error['verdict'] == 'HOLD'
    assert pids[0] in res_error['errors']


# ---- (h): code_version -------------------------------------------------------------------------------
def test_code_version_tracks_run_and_visref(monkeypatch, tmp_path):
    """(h) code_version이 run.py·visref/ 내용으로 바뀜"""
    v0 = tf.code_version()
    assert len(v0) == 12
    assert all(c in '0123456789abcdef' for c in v0)

    # a copy of the hashed files under tmp_path: changing the copy's run.py or visref/ changes code_version (the repository is never written)
    for f in ['run.py', 'diag/run_common.py', 'diag/blank_think.py', 'diag/closed_multistep.py', 'diag/step1_think.py',
              *[str(p.relative_to(REPRO)) for p in (REPRO / 'visref').glob('*.py')]]:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_bytes((REPRO / f).read_bytes())
    monkeypatch.setattr(tf, 'ROOT', tmp_path)
    base = tf.code_version()  # the file order differs from the repository's, so compare with the copy's own value
    assert tf.code_version() == base
    (tmp_path / 'run.py').write_bytes((REPRO / 'run.py').read_bytes() + b'\n# test comment\n')
    v1 = tf.code_version()
    assert v1 != base
    (tmp_path / 'visref/dpp.py').write_bytes((REPRO / 'visref/dpp.py').read_bytes() + b'\n# test comment\n')
    assert tf.code_version() not in (base, v1)


# ---- (c): Stub End-to-End and Resume Skips OK Rows Only ---------------------------------------------
def make_stub_parquet(pq_path, n_rows=5):
    buf = io.BytesIO()
    Image.new('RGB', (4, 4), color='green').save(buf, format='PNG')
    img_bytes = buf.getvalue()

    rows = []
    for i in range(n_rows):
        rows.append({
            'image': {'bytes': img_bytes},
            'qa': [{'question': f'How many objects in item {i}?', 'answer': str(i % 5), 'is_simple': False, 'question_id': i}]
        })
    table = pa.Table.from_pylist(rows)
    pq_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, pq_path)


def stub_environment(lm, tmp_path, monkeypatch):
    import run as run_mod
    import visref.model as vm

    pq_dir = tmp_path / 'parquet_data'
    make_stub_parquet(pq_dir / 'test-00000-of-00019.parquet', n_rows=5)

    emb = lm.get_input_embeddings()

    def encode_sample(model, tok, image, query, max_num, with_positions=False):
        g = torch.Generator().manual_seed(99)
        vis = torch.randn(6, emb.embedding_dim, generator=g)
        pre = emb(torch.randint(5, 90, (4,), generator=g)).detach()
        e = torch.cat([pre, vis, pre.flip(0)])
        return (e, vis)

    monkeypatch.setattr(vm, 'load_model', lambda m, r: (types.SimpleNamespace(language_model=lm), TOK))
    monkeypatch.setattr(vm, 'encode_sample', encode_sample)
    monkeypatch.setattr(vm, 'special_ids', lambda tok, w, p: {**SP, 'wait': TOK.encode(w), 'probe_prompt': TOK.encode(p)})
    monkeypatch.setattr(run_mod, 'fast_kv_selfcheck', lambda lm_, tok: (True, {}))
    monkeypatch.setattr(tf, 'Engine', lambda lm_, t, k, p, fast_kv=True: Engine(lm_, t, k, p, fast_kv=False))

    doc = {
        'source': {'repo': 'vikhyatk/tallyqa-test'},
        'trial': ['0-4-0'],
        'main': ['0-0-0', '0-1-0', '0-2-0', '0-3-0'],
        'memcheck': '0-4-0',
        'loc': {
            f'0-{i}-0': {'shard': 0, 'row': i, 'qi': 0} for i in range(5)
        }
    }
    ids_path = tmp_path / 'tally_force_ids.json'
    ids_path.write_text(json.dumps(doc))

    out_path = tmp_path / 'results' / 'out.jsonl'
    args = [
        '--parquet-dir', str(pq_dir),
        '--ids', str(ids_path),
        '--out', str(out_path),
        '--think-cap', '6',
        '--think-extra', '6',
        '--probe-cap', '4',
        '--n-t1', '2',
        '--batch-tokens', '4000',
        '--group', '2'
    ]
    return out_path, args, ids_path


def test_main_end_to_end_and_resume_skips_ok_rows_only(lm, tmp_path, monkeypatch):
    """(c) 행 재개 시 ok 행만 건너뜀"""
    out, args, ids_path = stub_environment(lm, tmp_path, monkeypatch)
    torch.set_grad_enabled(False)
    try:
        # First complete run
        tf.main(args)
        lines = out.read_text().splitlines()
        rows = [json.loads(l) for l in lines]
        assert len(rows) == 4
        assert [r['pid'] for r in rows] == ['0-0-0', '0-1-0', '0-2-0', '0-3-0']
        assert all(r['status'] == 'ok' for r in rows)
        assert all(set(r['n_correct']) == {'N', 'D'} for r in rows)

        # Resume with all ok: nothing more is added
        tf.main(args)
        assert len(out.read_text().splitlines()) == 4

        # Inject an error row for '0-1-0'
        bad = {'pid': '0-1-0', 'status': 'error', 'stage': 'generate', 'error': 'simulated failure'}
        out.write_text('\n'.join([lines[0], json.dumps(bad)]) + '\n')
        # On resume, '0-0-0' is skipped because it's 'ok', but '0-1-0' is re-run (and 0-2, 0-3 also run)
        tf.main(args)
        got = [json.loads(l) for l in out.read_text().splitlines()]
        pids_and_status = [(r['pid'], r['status']) for r in got]
        assert pids_and_status == [
            ('0-0-0', 'ok'),
            ('0-1-0', 'error'),
            ('0-1-0', 'ok'),
            ('0-2-0', 'ok'),
            ('0-3-0', 'ok')
        ]

        # Trial mode: memcheck
        out_trial = tmp_path / 'results' / 'trial.jsonl'
        args_trial = [x if x != str(out) else str(out_trial) for x in args] + ['--set', 'trial']
        tf.main(args_trial)
        mem_file = Path(str(out_trial) + '.memcheck.json')
        assert mem_file.exists()
        mem = json.loads(mem_file.read_text())
        assert mem['pid'] == '0-4-0'
        assert mem['last_hidden_n'] == 6
    finally:
        torch.set_grad_enabled(True)


# ---- Builder Tests -----------------------------------------------------------------------------------
def test_builder(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, 'HERE', tmp_path)
    assert bt.main(['test_run', '--owner', 'test_user', '--limit', '5', '--timeout', '1800', '--set', 'trial']) == 0
    nb_path = tmp_path / 'runs/test_run/notebook.ipynb'
    assert nb_path.exists()
    assert nb_path.stat().st_size < 1_000_000  # < 1 MB

    nb = json.loads(nb_path.read_text())
    src = ''.join(nb['cells'][0]['source'])
    ast.parse(src)  # valid python syntax
    assert "'diag/tally_force.py'" in src
    assert "'vikhyatk/tallyqa-test'" in src
    assert "'diag/tally_force_ids.json'" in src
    assert "DATA_DIR = pathlib.Path.home() / 'tallyqa_data'" in src  # data stays out of the kernel output (/kaggle/working)

    # only the data files of this kernel's questions (first 5 of the trial list, both GPU shards, + memcheck) are downloaded
    doc = json.loads((REPRO / 'diag/tally_force_ids.json').read_text())
    want = sorted({doc['loc'][p]['shard'] for p in doc['trial'][:5] + [doc['memcheck']]})
    rc_ = json.loads((tmp_path / 'runs/test_run/run_config.json').read_text())
    assert rc_['needed_shards'] == want and len(want) < 19

    # Inspect zip bundle inside notebook
    m = re.search(r"b64decode\('([^']+)'\)\)\)\.extractall", src)
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(m.group(1))))
    for f in ('diag/tally_force.py', 'diag/run_common.py', 'diag/closed_multistep.py', 'diag/step1_think.py',
              'diag/blank_think.py', 'diag/tally_force_ids.json', 'run.py', 'visref/dpp.py'):
        assert f in z.namelist(), f


def test_no_absolute_sandbox_paths_in_new_files():
    bad = ('/ho' + 'me/', '/t' + 'mp/', 'scratch' + 'pad', '/Us' + 'ers/')
    new_files = (
        'diag/run_common.py',
        'diag/tally_force.py',
        'diag/analyze_tally_force.py',
        'kaggle/build_tally_force.py',
        'tests/test_tally_force.py'
    )
    for f in new_files:
        content = (REPRO / f).read_text()
        for b in bad:
            assert b not in content, f"{b} found in {f}"
