"""CPU tests for diag/blank_think_ids.py, diag/blank_think.py, diag/analyze_blank_think.py and kaggle/build_blank_think.py
(tiny LM / stub model)."""
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
import analyze_blank_think as an  # noqa: E402
import blank_think as bt  # noqa: E402
import blank_think_ids as ids_mod  # noqa: E402
import build_blank_think as bb  # noqa: E402
import step1_think as t1  # noqa: E402
from test_repro import lm  # noqa: E402,F401  (pytest fixture)
from test_step1_think import SP as SP0, TOK  # noqa: E402

from run import pack  # noqa: E402
from visref.engine import Engine, Request  # noqa: E402

SP = {**SP0, 'close_ids': frozenset(i for i, s in enumerate(TOK.vocab) if s.startswith('}'))}
TE, EOS = SP['think_end'], SP['eos']
ROW = {'question_type': 'free_form', 'answer_type': 'integer', 'answer': '7', 'precision': None, 'choices': None}
CFG = bt.resolve_cfg({**bt.DEFAULTS, 'think_cap': 6, 'think_extra': 5, 'probe_cap': 4, 'n_t1': 2, 'batch_tokens': 4000})


# ---- ids ------------------------------------------------------------------------------------------------------------
def test_ids_file_equals_a_fresh_build_and_follows_the_plan():
    assert ids_mod.main(['--check']) == 0
    d = json.loads(ids_mod.OUT.read_text())
    order = [str(p) for p in json.loads(ids_mod.ORDER.read_text())]
    elig = [p for p in order if p in d['donor']]
    assert len(order) == 1000 and len(elig) + len(d['dropped']) == 1000
    assert [p for p in elig if p not in d['trial']] == d['main'] and len(d['trial']) == 5
    assert set(d['trial']) <= set(elig[-6:]) and d['n_visual'][d['memcheck']] == 3328 and d['memcheck'] in d['trial']
    for p, q in d['donor'].items():
        assert q != p and d['n_visual'][q] == d['n_visual'][p]
    for x in d['dropped']:
        assert sum(d['n_visual'].get(p) == x['n_visual'] for p in elig) == 0


def test_ids_donor_is_the_next_same_size_question_cyclically():
    nvis = {'a': 1, 'b': 2, 'c': 1, 'd': 3, 'e': 2, 'f': 1}
    assert ids_mod.donors(list('abcdef'), nvis) == {'a': 'c', 'b': 'e', 'c': 'f', 'e': 'b', 'f': 'a'}


# ---- contexts -------------------------------------------------------------------------------------------------------
def item(lm, pid='1', n=5, seed=0):
    emb = lm.get_input_embeddings()
    g = torch.Generator().manual_seed(seed)
    base = emb(torch.randint(5, 90, (10,), generator=g)).detach()
    pos = torch.arange(2, 2 + n)
    vis = torch.randn(n, base.shape[1], generator=g)
    blank = torch.randn(n, base.shape[1], generator=g)
    return {'pid': pid, 'row': ROW, 'prompt': {'own': base, 'blank': bt.fill_slots(base, pos, blank)}, 'vis': vis,
            'vis_donor': torch.randn(n, base.shape[1], generator=g), 'donor': 'd' + pid, 'base': base, 'pos': pos, 'blank': blank}


def test_fill_slots_replaces_only_the_image_rows_and_checks_the_count(lm):
    it = item(lm)
    b, p = it['prompt']['blank'], it['pos']
    assert torch.equal(b[p], it['blank']) and torch.equal(b[:2], it['base'][:2]) and torch.equal(b[7:], it['base'][7:])
    with pytest.raises(ValueError):
        bt.fill_slots(it['base'], p, it['blank'][:4])


def test_contexts_follow_the_plan(lm):
    eng = Engine(lm)
    it = item(lm)
    it['z'] = {'own': TOK.encode('abc'), 'blank': TOK.encode('xy')}
    it['sep'] = {'own': [], 'blank': TOK.encode('\n')}
    c = bt.contexts(it, eng, SP)
    po = torch.cat([it['prompt']['own'], eng.embed(TOK.encode('abc'))])
    pb = torch.cat([it['prompt']['blank'], eng.embed(TOK.encode('xy\n'))])
    w = eng.embed(SP['wait'])
    assert torch.equal(c['A0'], torch.cat([po, eng.embed([TE])])) and torch.equal(c['N0'], torch.cat([pb, eng.embed([TE])]))
    assert torch.equal(c['A'], torch.cat([po, w])) and torch.equal(c['N'], torch.cat([pb, w]))
    assert torch.equal(c['O'], torch.cat([pb, it['vis'], w])) and torch.equal(c['X'], torch.cat([pb, it['vis_donor'], w]))
    # the O block is A8's F format: all image tokens, untagged (step1_think.inject_embeds 'bmain')
    assert torch.equal(t1.inject_embeds(eng, TOK, SP, 'bmain', it['vis'], list(range(5)), 'xy', '\n'), it['vis'])


# ---- tiny LM end to end ---------------------------------------------------------------------------------------------
def tiny(lm, cfg=CFG):
    eng = Engine(lm, cfg['temperature'], cfg['top_k'], cfg['top_p'])
    items = [item(lm, '1', seed=0), item(lm, '2', n=7, seed=1)]
    bt.run_group(items, eng, TOK, SP, cfg, bt.make_generate(eng, cfg, pack))
    return eng, items


def test_run_group_is_deterministic_complete_and_seeded_as_planned(lm):
    torch.set_grad_enabled(False)
    try:
        eng, a = tiny(lm)
        _, b = tiny(lm)
        gen = bt.make_generate(eng, CFG, pack)
        for x, y in zip(a, b):
            assert not x['nonfinite'] and x['answers'] == y['answers'] and x['step1'] == y['step1'] and x['extra'] == y['extra']
            row = bt.row_out(x, CFG, 'v', 'main')
            json.dumps(row)
            assert row['status'] == 'ok' and set(row['n_correct']) == set(bt.CONDS) and set(row['extra']) == set(bt.FREE)
            assert all(len(row['answers'][c]) == CFG['n_t1'] + 1 for c in bt.CONDS)
            assert all(row['step1'][s]['len'] <= CFG['think_cap'] for s in ('own', 'blank'))
            # step 1 of the blank prompt: generator seed_of(seed, pid, 'step1'), stops </think> / EOS
            g = torch.Generator().manual_seed(t1.seed_of(42, x['pid'], 'step1'))
            r = gen([Request(x['prompt']['blank'], frozenset({TE, EOS}), CFG['think_cap'], g)])[0]
            assert bt.split_generated(TOK, SP, r.ids, '\n')[0] == x['step1']['blank']['ids']
            # extra thinking of O: its context, generator seed_of(seed, pid, 'think')
            z, sep = x['step1']['blank']['ids'], TOK.encode(x['step1']['blank']['sep_text'])
            ctx = torch.cat([x['prompt']['blank'], eng.embed(z + sep), x['vis'], eng.embed(SP['wait'])])
            g = torch.Generator().manual_seed(t1.seed_of(42, x['pid'], 'think'))
            r = gen([Request(ctx, frozenset({TE, EOS}), CFG['think_extra'], g)])[0]
            assert list(r.ids) == x['extra']['O']['ids']
    finally:
        torch.set_grad_enabled(True)


def test_a_non_finite_generation_gives_an_error_row(lm, monkeypatch):
    torch.set_grad_enabled(False)
    try:
        eng = Engine(lm)
        inner = bt.make_generate(eng, CFG, pack)
        calls = []

        def gen(reqs):
            out = inner(reqs)
            calls.append(len(reqs))
            return [None if len(calls) == 2 and i == 0 else o for i, o in enumerate(out)]  # first extra row of question 1
        its = [item(lm, '1'), item(lm, '2', seed=1)]
        bt.run_group(its, eng, TOK, SP, CFG, gen)
        r1, r2 = bt.row_out(its[0], CFG, 'v', 'main'), bt.row_out(its[1], CFG, 'v', 'main')
        assert r1['status'] == 'error' and r1['nonfinite'] == ['A'] and r2['status'] == 'ok'
    finally:
        torch.set_grad_enabled(True)


def test_chunk_below_the_caps_is_refused():
    with pytest.raises(SystemExit):
        bt.resolve_cfg({**bt.DEFAULTS, 'chunk': 512})
    assert bt.resolve_cfg(bt.DEFAULTS)['chunk'] == 1024


def test_select_pids():
    doc = {'main': list('abcdefgh'), 'trial': ['x']}
    assert bt.select_pids(doc, 'main', shard='1/4') == ['b', 'f']
    assert bt.select_pids(doc, 'main', limit=5, shard='0/4', done={'a'}) == ['e']
    assert bt.select_pids(doc, 'trial') == ['x']


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
    monkeypatch.setattr(bt, 'Engine', lambda lm_, t, k, p, fast_kv=True: Engine(lm_, t, k, p, fast_kv=False))
    doc = {'main': ['1', '2', '3', '4'], 'trial': ['5'], 'memcheck': '5', 'n_visual': nvis,
           'donor': {'1': '3', '2': '4', '3': '5', '4': '2', '5': '1'}}
    (tmp_path / 'ids.json').write_text(json.dumps(doc))
    out = tmp_path / 'res' / 'out.jsonl'
    args = ['--parquet', 'x', '--ids', str(tmp_path / 'ids.json'), '--out', str(out), '--think-cap', '6', '--think-extra', '5',
            '--probe-cap', '4', '--n-t1', '2', '--batch-tokens', '4000', '--group', '2']
    return out, args


def test_main_end_to_end_with_a_stub_model(lm, tmp_path, monkeypatch, capsys):
    out, args = stub(lm, tmp_path, monkeypatch)
    try:
        bt.main(args)
        res = [json.loads(l) for l in out.read_text().splitlines()]
        assert [r['pid'] for r in res] == ['1', '2', '3', '4'] and all(r['status'] == 'ok' for r in res)
        assert all(r['code_version'] == bt.code_version() and r['set'] == 'main' for r in res)
        assert res[0]['donor'] == '3' and res[0]['n_visual'] == 6 and res[0]['cfg']['chunk'] == 6
        bt.main(args)  # resume
        assert len(out.read_text().splitlines()) == 4
        o2 = tmp_path / 'res' / 'trial.jsonl'
        bt.main([x if x != str(out) else str(o2) for x in args] + ['--set', 'trial'])
        assert [json.loads(l)['set'] for l in o2.read_text().splitlines()] == ['trial']
        mem = json.loads(Path(str(o2) + '.memcheck.json').read_text())
        assert mem['pid'] == '5' and mem['ctx_len'] == 12 + 6 + 6 + len(SP['wait']) + 6
        # a wrong n_visual in the ids file gives an error row, the run goes on
        d = json.loads((tmp_path / 'ids.json').read_text())
        d['n_visual']['1'] = 5
        (tmp_path / 'ids.json').write_text(json.dumps(d))
        o3 = tmp_path / 'res' / 'bad.jsonl'
        bt.main([x if x != str(out) else str(o3) for x in args] + ['--limit', '2'])
        got = [json.loads(l) for l in o3.read_text().splitlines()]
        assert [r['status'] for r in got] == ['error', 'ok'] and got[0]['stage'] == 'load'
    finally:
        torch.set_grad_enabled(True)


# ---- analysis -------------------------------------------------------------------------------------------------------
def fake_row(pid, n, qt='free_form'):
    return {'pid': pid, 'status': 'ok', 'question_type': qt, 'code_version': 'v', 'scorer_version': 4, 'cfg': {'n_t1': 3},
            'n_correct': n, 'greedy': {c: int(n[c] >= 2) for c in an.CONDS},
            'answers': {c: [{'valid': True, 'correct': 0, 'greedy': False}] * 3 + [{'valid': True, 'correct': 0, 'greedy': True}]
                        for c in an.CONDS},
            'step1': {s: {'end': 'cap', 'len': 6} for s in ('own', 'blank')}, 'extra': {c: {'end': 'cap', 'len': 5} for c in an.FREE}}


def rows_with(k_an, k_on, k_ox, n=100):
    """n rows; the first k_an questions A > N, the first k_on O > N, the first k_ox O > X (3 vs 0 correct)."""
    out = {}
    for i in range(n):
        v = {c: 0 for c in an.CONDS}
        v['A'] = 3 if i < k_an else 0
        v['O'] = 3 if i < max(k_on, k_ox) else 0
        v['N'] = 0 if i < k_on else v['O']
        v['X'] = 0 if i < k_ox else v['O']
        out[str(i)] = fake_row(str(i), v)
    return out


@pytest.mark.parametrize('k_an,k_on,k_ox,want', [(3, 10, 10, 0), (20, 10, 10, 1), (20, 10, 0, 2), (20, 4, 4, 3), (20, 0, 0, 3)])
def test_verdict_order(k_an, k_on, k_ox, want):
    R = rows_with(k_an, k_on, k_ox)
    res = an.analyze(R, {'analysis': list(R)})
    assert res['verdict'] == want and res['sentence'].startswith(an.VERDICT[want])
    assert res['coverage']['n_used'] == 100 and res['primary']['A-N']['up'] == k_an


def test_gate_needs_five_points_even_when_significant():
    R = rows_with(8, 10, 10, n=200)  # A - N = 4 pp, 8 : 0, sign test passes
    res = an.analyze(R, {'analysis': list(R)})
    assert res['primary']['A-N']['mean_pp'] == 4.0 and res['primary']['A-N']['p_sign'] < 0.05 and res['verdict'] == 0


def test_analysis_counts_trial_errors_missing_and_refuses_duplicates(tmp_path):
    R = rows_with(20, 10, 10, n=50)
    R['t'] = fake_row('t', {c: 0 for c in an.CONDS})
    R['e'] = {'pid': 'e', 'status': 'error', 'code_version': 'v'}
    res = an.analyze(R, {'analysis': [str(i) for i in range(50)] + ['e', 'm']})
    c = res['coverage']
    assert c['n_used'] == 50 and c['n_outside_analysis'] == 1 and c['errors'] == ['e'] and c['n_missing'] == 1
    assert res['verdict'] is None and res['sentence'] == an.HOLD  # an error or a missing question holds the verdict
    assert an.analyze(R, {'analysis': [str(i) for i in range(50)] + ['m']})['verdict'] is None
    assert an.analyze(R, {'analysis': [str(i) for i in range(50)]})['verdict'] == 1
    f = tmp_path / 'r.jsonl'
    f.write_text(json.dumps(R['0']) + '\n' + json.dumps(R['0']) + '\n')
    with pytest.raises(SystemExit):
        an.load([str(f)])
    ids = tmp_path / 'ids.json'
    ids.write_text(json.dumps({'analysis': ['0']}))
    g = tmp_path / 'g.jsonl'
    g.write_text(json.dumps(R['0']) + '\n')
    o = tmp_path / 'a.json'
    assert an.main([str(g), '--ids', str(ids), '--out', str(o)]) == 0
    with pytest.raises(SystemExit):
        an.main([str(g), '--ids', str(ids), '--out', str(o)])


# ---- builder --------------------------------------------------------------------------------------------------------
def test_builder_shards_and_files(tmp_path, monkeypatch):
    monkeypatch.setattr(bb, 'HERE', tmp_path)
    assert bb.main(['k2', '--owner', 'me', '--shard-offset', '2', '--shard-total', '4']) == 0
    rc = json.loads((tmp_path / 'runs/k2/run_config.json').read_text())
    assert rc['shard_offset'] == 2 and rc['shard_total'] == 4 and rc['run_args'] == ['--set', 'main', '--group', '4']
    assert rc['timeout'] == 19800 and rc['time_limit'] == 18900
    src = ''.join(json.loads((tmp_path / 'runs/k2/notebook.ipynb').read_text())['cells'][0]['source'])
    ast.parse(src)
    assert "'diag/blank_think.py'" in src and "f\"{sh[i]}/{CFG['shard_total']}\"" in src and "f'blank_think_{sh[i]}.jsonl'" in src
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(re.search(r"b64decode\('([^']+)'\)\)\)\.extractall", src).group(1))))
    assert {'diag/blank_think.py', 'diag/closed_multistep.py', 'diag/step1_think.py', 'diag/blank_think_ids.json', 'run.py',
            'visref/engine.py'} <= set(z.namelist())
    assert (REPRO / 'diag/blank_think.py').read_bytes() == z.read('diag/blank_think.py')
    with pytest.raises(SystemExit):
        bb.main(['k3', '--owner', 'me', '--shard-offset', '4', '--shard-total', '4'])
    with pytest.raises(SystemExit):
        bb.main(['k3', '--owner', 'me', '--timeout', '3000'])
    assert bb.main(['t', '--owner', 'me', '--set', 'trial', '--timeout', '3000']) == 0


def test_the_pinned_code_version_of_run_py_is_untouched():
    import run
    assert run.code_version() == '06435ae748c8'


def test_no_absolute_sandbox_paths_in_the_new_files():
    bad = ('/ho' + 'me/', '/t' + 'mp/', 'scratch' + 'pad', '/Us' + 'ers/')
    for f in ('diag/blank_think_ids.py', 'diag/blank_think.py', 'diag/analyze_blank_think.py', 'kaggle/build_blank_think.py',
              'tests/test_blank_think.py', 'diag/blank_think_ids.json'):
        assert not any(b in (REPRO / f).read_text() for b in bad), f


def test_make_analysis_ids_takes_the_first_n_of_every_shard():
    main = list('abcdefghij')
    assert an.make_analysis_ids(main, 4, 2) == list('abcdefgh')
    assert an.make_analysis_ids(main, 2, 2) == list('abcd')
    assert an.make_analysis_ids(main, 4, 1) == list('abcd') and an.make_analysis_ids(main, 4, 9) == main
