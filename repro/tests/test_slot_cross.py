"""CPU tests for diag/slot_cross.py, diag/analyze_slot_cross.py and kaggle/build_slot_cross.py (tiny LM / stub model)."""
import ast
import base64
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import pytest
import torch

REPRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO / 'diag'))
sys.path.insert(0, str(REPRO / 'kaggle'))
import analyze_slot_cross as an  # noqa: E402
import build_slot_cross as bs  # noqa: E402
import recover_check as rc  # noqa: E402
import slot_cross as sc  # noqa: E402
from test_blank_think import ROW, SP, item, stub  # noqa: E402,F401
from test_repro import lm  # noqa: E402,F401
from test_step1_think import TOK  # noqa: E402

from run import pack  # noqa: E402
from visref.engine import Engine  # noqa: E402

TE = SP['think_end']
CFG = sc.resolve_cfg({**sc.DEFAULTS, 'think_cap': 6, 'batch_tokens': 4000})


def sitem(lm, pid='1', seed=0):
    it = item(lm, pid, seed=seed)
    return {**it, 'vis_of': {'own': it['vis'], 'blank': it['blank']}, 'gt_ids': TOK.encode('{7}')}


def test_cell_contexts_follow_the_plan(lm):
    eng = Engine(lm)
    it = sitem(lm)
    it['z'], it['sep'] = TOK.encode('xy'), TOK.encode('\n')
    tail = eng.embed([TE] + SP['probe_prompt'])
    zz = eng.embed(TOK.encode('xy\n'))
    P = it['prompt']
    want = {'o-': [P['own'], zz, tail], 'b-': [P['blank'], zz, tail], 'oo': [P['own'], zz, it['vis'], tail],
            'ob': [P['own'], zz, it['blank'], tail], 'bo': [P['blank'], zz, it['vis'], tail], 'bb': [P['blank'], zz, it['blank'], tail]}
    for name, (slot, post) in sc.CELLS.items():
        assert torch.equal(sc.cell_context(it, eng, SP, slot, post), torch.cat(want[name])), name
    c = {n: sc.cell_context(it, eng, SP, *sc.CELLS[n]) for n in ('ob', 'bo')}
    assert c['ob'].shape == c['bo'].shape  # same length, swapped places


def test_run_group_scores_every_cell_as_recover_check_does(lm):
    torch.set_grad_enabled(False)
    try:
        eng = Engine(lm)
        sp = {**SP, 'close_ids': frozenset(i for i, s in enumerate(TOK.vocab) if s.startswith('}'))}
        its = [sitem(lm, '1'), sitem(lm, '2', seed=1)]
        sc.run_group(its, eng, TOK, sp, CFG, sc.make_generate(eng, CFG, pack))
        for it in its:
            assert not it['nonfinite'] and set(it['cell']) == set(sc.CELLS)
            ctx = sc.cell_context(it, eng, sp, *sc.CELLS['bo'])
            full, logp = rc.score_string_pos(eng, ctx, it['gt_ids'], torch.arange(ctx.shape[0]))
            pos, close = rc.content_positions(it['gt_ids'], sp['brace_text'])
            a, p, _ = rc.read_cond(logp, it['gt_ids'], pos, close, torch.tensor(sorted(sp['close_ids'])))
            assert it['cell']['bo']['p_gt'] == round(p, 6) and it['cell']['bo']['logp_gt_full'] == round(full, 4)
            row = sc.row_out(it, CFG, 'v', 'main')
            json.dumps(row)
            assert row['status'] == 'ok' and row['step1']['ids'] == it['z']
        # step 1 equals blank_think's (same seed label, same prompt, same step-end rule)
        import blank_think as bt
        cfg_bt = bt.resolve_cfg({**bt.DEFAULTS, 'think_cap': 6, 'think_extra': 6, 'probe_cap': 4, 'n_t1': 1, 'batch_tokens': 4000})
        b = sitem(lm, '1')
        bt.run_group([b], eng, TOK, sp, cfg_bt, bt.make_generate(eng, cfg_bt, pack))
        assert b['step1']['blank']['ids'] == its[0]['step1']['ids']
    finally:
        torch.set_grad_enabled(True)


def test_main_end_to_end_with_a_stub_model(lm, tmp_path, monkeypatch):
    out, args = stub(lm, tmp_path, monkeypatch)
    monkeypatch.setattr(sc, 'Engine', lambda lm_, t, k, p, fast_kv=True: Engine(lm_, t, k, p, fast_kv=False))
    drop = {'--think-extra', '--probe-cap', '--n-t1'}
    args = [a for i, a in enumerate(args) if a not in drop and (i == 0 or args[i - 1] not in drop)]
    try:
        sc.main(args)
        res = [json.loads(l) for l in out.read_text().splitlines()]
        assert [r['pid'] for r in res] == ['1', '2', '3', '4'] and all(r['status'] == 'ok' for r in res)
        assert all(r['code_version'] == sc.code_version() and set(r['cell']) == set(sc.CELLS) for r in res)
        sc.main(args)
        assert len(out.read_text().splitlines()) == 4
    finally:
        torch.set_grad_enabled(True)


def frow(pid, p):
    return {'pid': pid, 'status': 'ok', 'question_type': 'free_form', 'code_version': 'v', 'step1': {'end': 'cap'},
            'cell': {c: {'p_gt': p[c]} for c in an.CELLS}}


def rows_from(n, **means):
    """n rows with p per cell = means (default 0.1) plus a small per-question spread."""
    R = {}
    for i in range(n):
        e = (i % 5 - 2) * 0.01
        R[str(i)] = frow(str(i), {c: max(0.0, means.get(c.replace('-', '_'), 0.1) + e) for c in an.CELLS})
    return R


def test_gate_counterexample():
    R = rows_from(100, o_=0.12, b_=0.10)  # gate difference 0.02 < 0.05
    assert an.analyze(R, {'analysis': list(R)})['sentence'] == an.VERDICT_NO_INFO
    # the review's counterexample: gate passes, D > 0, but both recoveries are negative -> never "되살렸다"
    R = rows_from(100, o_=0.30, b_=0.10, ob=0.04, bb=0.08, bo=0.02)
    d = an.analyze(R, {'analysis': list(R)})['decision']
    assert d['gate'] and d['position'] == 'slot' and not d['R_S_confirmed'] and not d['R_P_confirmed']
    assert d['sentence'] == ' '.join([an.REC['S'][False], an.REC['P'][False], an.POS['slot']]) and 'R_P_over_R_S' not in d


def test_recoveries_and_position_sentences():
    R = rows_from(100, o_=0.30, b_=0.10, ob=0.25, bb=0.10, bo=0.15)
    d = an.analyze(R, {'analysis': list(R)})['decision']
    assert d['R_S_confirmed'] and d['R_P_confirmed'] and d['position'] == 'slot'
    assert abs(d['R_P_over_R_S']['mean'] - 1 / 3) < 1e-3
    R = rows_from(100, o_=0.30, b_=0.10, ob=0.15, bb=0.10, bo=0.15)
    assert an.analyze(R, {'analysis': list(R)})['decision']['position'] == 'none'


def test_hold_on_missing_or_error_rows():
    R = rows_from(10, o_=0.3)
    R['e'] = {'pid': 'e', 'status': 'error'}
    assert an.analyze(R, {'analysis': list(R)})['sentence'] == an.HOLD
    assert an.analyze(rows_from(10), {'analysis': [str(i) for i in range(11)]})['sentence'] == an.HOLD


def test_builder(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, 'HERE', tmp_path)
    assert bs.main(['s', '--owner', 'me']) == 0
    src = ''.join(json.loads((tmp_path / 'runs/s/notebook.ipynb').read_text())['cells'][0]['source'])
    ast.parse(src)
    assert "'diag/slot_cross.py'" in src and "f'slot_cross_{sh[i]}.jsonl'" in src
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(re.search(r"b64decode\('([^']+)'\)\)\)\.extractall", src).group(1))))
    for f in ('diag/slot_cross.py', 'diag/blank_think.py', 'diag/recover_check.py', 'diag/recover_check_ids.py',
              'diag/inject_sensitivity.py', 'diag/closed_multistep.py', 'diag/step1_think.py', 'diag/blank_think_ids.json'):
        assert f in z.namelist(), f


def test_no_absolute_sandbox_paths_in_the_new_files():
    bad = ('/ho' + 'me/', '/t' + 'mp/', 'scratch' + 'pad', '/Us' + 'ers/')
    for f in ('diag/slot_cross.py', 'diag/analyze_slot_cross.py', 'kaggle/build_slot_cross.py', 'tests/test_slot_cross.py'):
        assert not any(b in (REPRO / f).read_text() for b in bad), f
