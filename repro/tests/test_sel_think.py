"""CPU tests for diag/sel_think_ids.py, diag/sel_think.py, diag/analyze_sel_think.py and kaggle/build_sel_think.py
(tiny LM / stub model)."""
import ast
import base64
import io
import json
import re
import sys
import types
import zipfile

import pytest
import torch

from pathlib import Path

REPRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO / 'diag'))
sys.path.insert(0, str(REPRO / 'kaggle'))
import analyze_sel_think as an  # noqa: E402
import build_sel_think as bs  # noqa: E402
import sel_think as st  # noqa: E402
import sel_think_ids as ids_mod  # noqa: E402
import step1_think as t1  # noqa: E402
from test_repro import lm  # noqa: E402,F401  (pytest fixture)
from test_step1_think import SP as SP0, TOK  # noqa: E402

from run import pack  # noqa: E402
from visref.dpp import budget, select_tokens  # noqa: E402
from visref.engine import Engine, Request  # noqa: E402

SP = {**SP0, 'close_ids': frozenset(i for i, s in enumerate(TOK.vocab) if s.startswith('}'))}
TE, EOS = SP['think_end'], SP['eos']
ROW = {'question_type': 'free_form', 'answer_type': 'integer', 'answer': '7', 'precision': None, 'choices': None}
CFG = st.resolve_cfg({**st.DEFAULTS, 'think_extra': 5, 'probe_cap': 4, 'n_t1': 2, 'batch_tokens': 4000})


# ---- ids ------------------------------------------------------------------------------------------------------------
def test_ids_file_equals_a_fresh_build_and_keeps_blank_thinks_lists_and_first_steps():
    assert ids_mod.main(['--check']) == 0
    d = json.loads(ids_mod.OUT.read_text())
    a = json.loads(ids_mod.ANALYSIS.read_text())
    b = json.loads(ids_mod.IDS.read_text())
    rows = ids_mod.stored_rows(ids_mod.ROWS)
    assert d['main'] == [p for p in b['main'] if p in rows] and len(d['main']) == 724 and d['trial'] == b['trial']
    assert set(a['analysis']) <= set(d['main'])
    assert d['memcheck'] == b['memcheck'] == '865' and d['memcheck'] in d['trial'] and d['n_visual']['865'] == 3328
    for p in d['main'][:3] + d['main'][-3:] + d['trial']:
        for s in ('own', 'blank'):
            assert d['step1'][p][s] == {k: rows[p]['step1'][s][k] for k in ('end', 'ids', 'sep_text')}
            assert len(d['step1'][p][s]['ids']) == rows[p]['step1'][s]['len']
        assert d['donor'][p] == b['donor'][p] and d['n_visual'][p] == b['n_visual'][p] == rows[p]['n_visual']


# ---- contexts -------------------------------------------------------------------------------------------------------
def item(lm, pid='1', n=6, seed=0, z_own='abc', z_blank='xy', sep_blank='\n'):
    emb = lm.get_input_embeddings()
    g = torch.Generator().manual_seed(seed)
    base = emb(torch.randint(5, 90, (n + 4,), generator=g)).detach()
    pos = torch.arange(2, 2 + n)
    vis = torch.randn(n, base.shape[1], generator=g)
    blank = torch.randn(n, base.shape[1], generator=g)
    step1 = {'own': {'end': 'close', 'ids': TOK.encode(z_own), 'sep_text': ''},
             'blank': {'end': 'cap', 'ids': TOK.encode(z_blank), 'sep_text': sep_blank}}
    return {'pid': pid, 'row': ROW, 'prompt': {'own': base, 'blank': st.fill_slots(base, pos, blank)}, 'vis': vis,
            'vis_donor': torch.randn(n, base.shape[1], generator=g), 'donor': 'd' + pid, 'step1': step1}


def prepared(lm, eng, **kw):
    it = item(lm, **kw)
    it['z'], it['sep'], it['sep_text'] = {}, {}, {}
    for s in st.SIDES:
        it['z'][s], it['sep'][s], it['sep_text'][s] = st.stored_step(TOK, it['step1'][s])
    it['m'], it['sel'] = st.selections(it, eng, CFG)
    return it


def test_selections_follow_the_plan(lm):
    eng = Engine(lm)
    it = prepared(lm, eng, n=10)
    m = budget(10, 0.3)
    assert it['m'] == m == 3
    # Z = the last-layer states of the stored step in its own context (EF 1525, closed_multistep), not its input embeddings
    z = TOK.encode('xy')
    Z = eng.last_hidden(torch.cat([it['prompt']['blank'], eng.embed(z)]), len(z))
    assert Z.shape[0] == len(z) and it['sel']['blank'] == select_tokens(it['vis'], Z, m) and set(it['sel']) == {'blank', 'rand'}
    assert it['sel']['rand'] == t1.random_sel('1', 42, 10, len(it['sel']['blank'])) == sorted(it['sel']['rand'])
    # the random tokens depend on the pid only, not on the image or the step text
    other = prepared(lm, eng, n=10, seed=5, z_blank='qrstuv')
    if len(other['sel']['blank']) == len(it['sel']['blank']):
        assert other['sel']['rand'] == it['sel']['rand']
    # an empty step text picks nothing (no crash), and R then has no tokens either
    empty = prepared(lm, eng, z_blank='')
    assert empty['sel']['blank'] == [] and empty['sel']['rand'] == []


def test_contexts_follow_the_plan(lm):
    eng = Engine(lm)
    it = prepared(lm, eng, n=10)
    c = st.contexts(it, eng, TOK, SP)
    pb = torch.cat([it['prompt']['blank'], eng.embed(TOK.encode('xy\n'))])
    w, v, s, e, nl = eng.embed(SP['wait']), it['vis'], it['sel'], eng.embed, TOK.encode('\n')
    # the closed_multistep block: a newline first only when the step text (+ separator) does not end with one
    blk_b = lambda x: torch.cat([e([SP['img_start']]), x, e([SP['img_end']] + nl)])  # noqa: E731  'xy' + '\n' ends with one
    assert torch.equal(st.v_block(eng, TOK, SP, v, [0], 'abc', ''), torch.cat([e(nl), blk_b(v[[0]])]))  # no trailing \n
    want = {'N_b': torch.cat([pb, w]), 'D_b': torch.cat([pb, blk_b(v[s['blank']]), w]),
            'R_b': torch.cat([pb, blk_b(v[s['rand']]), w]), 'F_b': torch.cat([pb, blk_b(v), w]),
            'X_b': torch.cat([pb, blk_b(it['vis_donor']), w])}
    assert set(c) == set(st.CONDS)
    for k in st.CONDS:
        assert torch.equal(c[k], want[k]), k
    # the same block as closed_multistep.v_block (the ② format of 2026-10-23)
    assert torch.equal(st.v_block(eng, TOK, SP, v, s['blank'], 'xy', '\n'), blk_b(v[s['blank']]))


def test_stored_step_keeps_the_ids_and_rebuilds_the_separator():
    z, sep, text = st.stored_step(TOK, {'end': 'cap', 'ids': [5, 6, 7], 'sep_text': '\n'})
    assert z == [5, 6, 7] and sep == TOK.encode('\n') and text == '\n'
    assert st.stored_step(TOK, {'end': 'close', 'ids': [5], 'sep_text': ''})[1] == []


# ---- tiny LM end to end ---------------------------------------------------------------------------------------------
def tiny(lm, cfg=CFG):
    eng = Engine(lm, cfg['temperature'], cfg['top_k'], cfg['top_p'])
    items = [item(lm, '1', n=10, seed=0), item(lm, '2', n=7, seed=1)]
    st.run_group(items, eng, TOK, SP, cfg, st.make_generate(eng, cfg, pack))
    return eng, items


def test_run_group_is_deterministic_complete_and_seeded_as_planned(lm):
    torch.set_grad_enabled(False)
    try:
        eng, a = tiny(lm)
        _, b = tiny(lm)
        gen = st.make_generate(eng, CFG, pack)
        for x, y in zip(a, b):
            assert not x['nonfinite'] and x['answers'] == y['answers'] and x['extra'] == y['extra'] and x['sel'] == y['sel']
            row = st.row_out(x, CFG, 'v', 'main')
            json.dumps(row)
            assert row['status'] == 'ok' and set(row['n_correct']) == set(st.CONDS) == set(row['extra'])
            assert all(len(row['answers'][c]) == CFG['n_t1'] + 1 for c in st.CONDS)
            assert row['n_sel']['rand'] == row['n_sel']['blank'] == len(row['sel']['blank'])
            assert row['recon']['blank'] == {'n_z': 2, 'n_sep': 1, 'end': 'cap'}
            # extra thinking of D_b: its context, generator seed_of(seed, pid, 'think')
            blk = torch.cat([eng.embed([SP['img_start']]), x['vis'][x['sel']['blank']], eng.embed([SP['img_end']] + TOK.encode('\n'))])
            ctx = torch.cat([x['prompt']['blank'], eng.embed(TOK.encode('xy\n')), blk, eng.embed(SP['wait'])])
            g = torch.Generator().manual_seed(t1.seed_of(42, x['pid'], 'think'))
            r = gen([Request(ctx, frozenset({TE, EOS}), CFG['think_extra'], g)])[0]
            assert list(r.ids) == x['extra']['D_b']['ids']
    finally:
        torch.set_grad_enabled(True)


def test_a_non_finite_generation_gives_an_error_row(lm):
    torch.set_grad_enabled(False)
    try:
        eng = Engine(lm)
        inner = st.make_generate(eng, CFG, pack)
        calls = []

        def gen(reqs):
            out = inner(reqs)
            calls.append(len(reqs))
            return [None if len(calls) == 1 and i == 0 else o for i, o in enumerate(out)]  # first extra row of question 1
        its = [item(lm, '1'), item(lm, '2', seed=1)]
        st.run_group(its, eng, TOK, SP, CFG, gen)
        r1, r2 = st.row_out(its[0], CFG, 'v', 'main'), st.row_out(its[1], CFG, 'v', 'main')
        assert r1['status'] == 'error' and r1['nonfinite'] == ['N_b'] and r2['status'] == 'ok'
    finally:
        torch.set_grad_enabled(True)


def test_chunk_below_the_cap_is_refused():
    with pytest.raises(SystemExit):
        st.resolve_cfg({**st.DEFAULTS, 'chunk': 512})
    assert st.resolve_cfg(st.DEFAULTS)['chunk'] == 1024


# ---- main with a stub model -----------------------------------------------------------------------------------------
def stub(lm, tmp_path, monkeypatch):
    import run as run_mod
    import visref.model as vm
    pids = ['1', '2', '3', '4', '5']
    nvis = {'1': 6, '2': 4, '3': 6, '4': 4, '5': 6}
    emb = lm.get_input_embeddings()

    def encode_sample(model, tok, image, query, max_num, with_positions=False):
        g = torch.Generator().manual_seed(getattr(image, 'seed', 99))
        vis = torch.randn(nvis[query], emb.embedding_dim, generator=g)
        pre = emb(torch.randint(5, 90, (3,), generator=g)).detach()
        e = torch.cat([pre, vis, pre.flip(0)])
        return (e, vis, torch.arange(3, 3 + vis.shape[0])) if with_positions else (e, vis)

    rows = {p: {**ROW, 'pid': p, 'query': p, 'image': (lambda p=p: types.SimpleNamespace(size=(4, 4), seed=int(p)))}
            for p in pids}
    monkeypatch.setattr(vm, 'load_mathvista', lambda path: rows)
    monkeypatch.setattr(vm, 'load_model', lambda m, r: (types.SimpleNamespace(language_model=lm), TOK))
    monkeypatch.setattr(vm, 'encode_sample', encode_sample)
    monkeypatch.setattr(vm, 'special_ids', lambda tok, w, p: {**SP, 'wait': TOK.encode(w), 'probe_prompt': TOK.encode(p)})
    monkeypatch.setattr(run_mod, 'fast_kv_selfcheck', lambda lm_, tok: (True, {}))
    monkeypatch.setattr(st, 'Engine', lambda lm_, t, k, p, fast_kv=True: Engine(lm_, t, k, p, fast_kv=False))
    step1 = {p: {'own': {'end': 'close', 'ids': TOK.encode('ab'), 'sep_text': ''},
                 'blank': {'end': 'cap', 'ids': TOK.encode('abc'), 'sep_text': '\n'}} for p in pids}
    doc = {'main': ['1', '2', '3', '4'], 'trial': ['5'], 'memcheck': '5', 'n_visual': nvis,
           'donor': {'1': '3', '2': '4', '3': '5', '4': '2', '5': '1'}, 'step1': step1}
    (tmp_path / 'ids.json').write_text(json.dumps(doc))
    out = tmp_path / 'res' / 'out.jsonl'
    args = ['--parquet', 'x', '--ids', str(tmp_path / 'ids.json'), '--out', str(out), '--think-extra', '5',
            '--probe-cap', '4', '--n-t1', '2', '--batch-tokens', '4000', '--group', '2', '--step1-cap', '6']
    return out, args


def test_main_end_to_end_with_a_stub_model_and_resume_skips_ok_rows_only(lm, tmp_path, monkeypatch):
    out, args = stub(lm, tmp_path, monkeypatch)
    try:
        st.main(args)
        res = [json.loads(l) for l in out.read_text().splitlines()]
        assert [r['pid'] for r in res] == ['1', '2', '3', '4'] and all(r['status'] == 'ok' for r in res)
        assert all(r['code_version'] == st.code_version() and r['set'] == 'main' for r in res)
        assert res[0]['donor'] == '3' and res[0]['n_visual'] == 6 and res[0]['cfg']['chunk'] == 5 and res[0]['m'] == 1
        st.main(args)  # resume: nothing left
        assert len(out.read_text().splitlines()) == 4
        # trial: the memory check on the memcheck question (own prompt 12 + 6 padded + block of all 6 tokens + Wait + 5)
        o2 = tmp_path / 'res' / 'trial.jsonl'
        st.main([x if x != str(out) else str(o2) for x in args] + ['--set', 'trial'])
        mem = json.loads(Path(str(o2) + '.memcheck.json').read_text())
        assert mem['pid'] == '5' and mem['last_hidden_n'] == 6 and mem['n_sel'] >= 1
        assert mem['ctx_len'] == 12 + 6 + (1 + 1 + 6 + 1 + 1) + len(SP['wait']) + 5 and mem['generated_tokens'] in range(1, 9)
        # an error row in --out is run again on resume
        lines = out.read_text().splitlines()
        bad = {'pid': '2', 'status': 'error', 'stage': 'generate', 'error': 'x'}
        out.write_text('\n'.join([lines[0], json.dumps(bad)]) + '\n')
        st.main(args)
        got = [json.loads(l) for l in out.read_text().splitlines()]
        assert [(r['pid'], r['status']) for r in got] == [('1', 'ok'), ('2', 'error'), ('2', 'ok'), ('3', 'ok'), ('4', 'ok')]
        # a wrong n_visual in the ids file gives an error row, the run goes on
        d = json.loads((tmp_path / 'ids.json').read_text())
        d['n_visual']['1'] = 5
        (tmp_path / 'ids.json').write_text(json.dumps(d))
        o3 = tmp_path / 'res' / 'bad.jsonl'
        st.main([x if x != str(out) else str(o3) for x in args] + ['--limit', '2'])
        got = [json.loads(l) for l in o3.read_text().splitlines()]
        assert [r['status'] for r in got] == ['error', 'ok'] and got[0]['stage'] == 'load'
    finally:
        torch.set_grad_enabled(True)


# ---- analysis -------------------------------------------------------------------------------------------------------
def fake_row(pid, n, qt='free_form'):
    return {'pid': pid, 'status': 'ok', 'question_type': qt, 'code_version': 'v', 'scorer_version': 4, 'cfg': {'n_t1': 3},
            'n_visual': 100, 'm': 30, 'n_sel': {'blank': 25, 'rand': 25}, 'n_correct': n,
            'greedy': {c: int(n[c] >= 2) for c in an.CONDS}, 'extra': {c: {'end': 'cap', 'len': 5} for c in an.CONDS}}


def rows_with(k_gate, k1, k2, k3=0, n=100, down=()):
    """n rows; the first k_gate questions F_b > N_b, k1 D_b > N_b, k2 D_b > R_b (3 vs 0 correct; k3 is unused since P3 was
    dropped); a name in
    `down` flips that comparison's direction."""
    out = {}
    for i in range(n):
        v = {c: 0 for c in an.CONDS}
        v['F_b'] = 3 if i < k_gate else 0
        hi, lo = (0, 3) if 'P1' in down else (3, 0)
        if i < k1:
            v['D_b'], v['N_b'] = hi, lo
            v['R_b'] = v['D_b']  # no change for P2 on these questions
        hi2, lo2 = (0, 3) if 'P2' in down else (3, 0)
        if i >= 50 and i - 50 < k2:  # separate questions so P1 and P2 do not share changed questions
            v['D_b'], v['R_b'] = hi2, lo2
            v['N_b'] = v['F_b'] = v['D_b']  # no change for P1 or the gate on these questions
        out[str(i)] = fake_row(str(i), v)
    return out


def run_an(R):
    return an.analyze(R, {'analysis': list(R)})


def test_gate_failure_gives_no_selection_sentence():
    res = run_an(rows_with(3, 20, 20, 20))
    assert res['gate']['passes'] is False and res['decision'] is None and res['sentences'][0].startswith(an.GATE_FAIL)


def test_decisions_up_down_not_with_holm():
    res = run_an(rows_with(30, 20, 20, 0))
    assert res['gate']['passes'] and [res['decision'][k]['result'] for k in an.PRIMARY] == ['UP', 'UP']
    assert res['sentences'][0].startswith(an.SENTENCES['P1']['UP']) and '(P1 20 : 0)' in res['sentences'][0]
    res = run_an(rows_with(30, 20, 20, 20, down=('P2',)))
    assert [res['decision'][k]['result'] for k in an.PRIMARY] == ['UP', 'DOWN']
    # 6 : 0 has p = 0.03125 unadjusted; Holm over two comparisons makes the smallest count twice: 0.0625, not a pass
    res = run_an(rows_with(30, 6, 0, 0))
    assert res['primary']['P1']['p_sign'] < 0.05 and res['decision']['P1']['p_holm'] == pytest.approx(2 * 2 / 2 ** 6, abs=1e-6)
    assert res['decision']['P1']['result'] == 'NOT'


def test_p2_speaks_of_the_ranking_only():
    # review counterexample: both D_b and R_b below N_b, D_b below R_b -> "lower than random", never "raised"
    R = {str(i): fake_row(str(i), {**{c: 0 for c in an.CONDS}, 'N_b': 2, 'D_b': 0, 'R_b': 1, 'F_b': 3}) for i in range(100)}
    res = run_an(R)
    assert res['gate']['passes'] and res['decision']['P2']['result'] == 'DOWN' and res['decision']['P1']['result'] == 'DOWN'
    assert res['sentences'][1].startswith(an.SENTENCES['P2']['DOWN']) and '낮았다' in res['sentences'][1]
    assert all('올렸다' not in x for x in an.SENTENCES['P2'].values())


def test_holm():
    assert an.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert an.holm([0.5, 0.001, 1.0]) == pytest.approx([1.0, 0.003, 1.0])


def test_hold_on_missing_or_error_rows():
    R = rows_with(30, 20, 20, 20)
    ids = {'analysis': list(R) + ['x']}
    assert an.analyze(R, ids)['sentences'] == [an.HOLD]
    R['0'] = {'pid': '0', 'status': 'error'}
    assert an.analyze(R, {'analysis': list(R)})['sentences'] == [an.HOLD]


def test_agreement_with_blank_think():
    R = rows_with(30, 0, 0, 0, n=4)
    bt = {p: {'n_correct': {'N': 0, 'O': 3 if int(p) < 2 else 0, 'X': 0}} for p in R}
    res = an.analyze(R, {'analysis': list(R)}, bt=bt)
    ag = res['descriptive']['blank_think_agreement']
    assert ag['F_b~O'] == {'n': 4, 'equal': 2, 'sum_here': 12, 'sum_blank_think': 6}
    assert 'N_o~A' not in ag and ag['N_b~N']['equal'] == 4


# ---- builder --------------------------------------------------------------------------------------------------------
def test_builder(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, 'HERE', tmp_path)
    assert bs.main(['s', '--owner', 'me']) == 0
    src = ''.join(json.loads((tmp_path / 'runs/s/notebook.ipynb').read_text())['cells'][0]['source'])
    ast.parse(src)
    assert "'diag/sel_think.py'" in src and "f'sel_think_{sh[i]}.jsonl'" in src and "'diag/sel_think_ids.json'" in src
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(re.search(r"b64decode\('([^']+)'\)\)\)\.extractall", src).group(1))))
    for f in ('diag/sel_think.py', 'diag/blank_think.py', 'diag/closed_multistep.py', 'diag/step1_think.py',
              'diag/sel_think_ids.json', 'visref/dpp.py', 'run.py'):
        assert f in z.namelist(), f


def test_no_absolute_sandbox_paths_in_the_new_files():
    bad = ('/ho' + 'me/', '/t' + 'mp/', 'scratch' + 'pad', '/Us' + 'ers/')
    for f in ('diag/sel_think.py', 'diag/sel_think_ids.py', 'diag/analyze_sel_think.py', 'kaggle/build_sel_think.py',
              'tests/test_sel_think.py'):
        assert not any(b in (REPRO / f).read_text() for b in bad), f
