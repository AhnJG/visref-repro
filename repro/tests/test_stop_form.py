"""CPU tests for diag/stop_form_ids.py, diag/stop_form.py, diag/analyze_stop_form.py and kaggle/build_stop_form.py
(tiny LM / stub model)."""
import ast
import base64
import io
import json
import math
import re
import sys
import zipfile
from pathlib import Path

import torch

REPRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO / 'diag'))
sys.path.insert(0, str(REPRO / 'kaggle'))
import analyze_stop_form as an  # noqa: E402
import build_stop_form as bs  # noqa: E402
import inject_sensitivity as isn  # noqa: E402
import stop_form as sf  # noqa: E402
import stop_form_ids as ids_mod  # noqa: E402
from test_blank_think import SP, stub  # noqa: E402
from test_repro import lm  # noqa: E402,F401  (pytest fixture)
from test_step1_think import TOK  # noqa: E402

from run import pack  # noqa: E402
from visref.engine import Engine, Result  # noqa: E402

TE, EOS = SP['think_end'], SP['eos']
CFG = sf.resolve_cfg({**sf.DEFAULTS, 'n_short': 3, 'n_free': 4, 'probe_cap': 4, 'answer_cap': 6, 'batch_tokens': 4000})
MC = {'question_type': 'multi_choice', 'answer_type': 'text', 'choices': ['10', '20', '30'], 'answer': '20', 'precision': None}
NUM = {'question_type': 'free_form', 'answer_type': 'integer', 'choices': None, 'answer': '2', 'precision': None}


# ---- ids ------------------------------------------------------------------------------------------------------------
def test_ids_file_equals_a_fresh_build_and_follows_the_plan():
    assert ids_mod.main(['--check']) == 0
    d = json.loads(ids_mod.OUT.read_text())
    order = [str(p) for p in json.loads(ids_mod.ORDER.read_text())]
    selfs = [p for p in order if d['stratum'].get(p) in ('close', 'eos') or p in d['trial'][:5]]
    assert len(d['main']) == 454 and len(set(d['main'])) == 454
    n = d['n_self_main']
    assert n == 354 and all(d['stratum'][p] in ('close', 'eos') for p in d['main'][:n])
    assert all(d['stratum'][p] == 'cap' for p in d['main'][n:]) and len(d['main'][n:]) == 100
    assert not set(d['trial']) & set(d['main']) and len(d['trial']) == 8
    assert [p for p in order if p in set(d['main'][:n])] == d['main'][:n]  # main order kept
    assert [p for p in order if p in set(selfs)][-5:] == d['trial'][:5]
    # the stored decision reproduces the report's count: 9 of the self-ended questions continue (REPORT 4.8)
    assert sum(d['stored_H'][p] is None or d['stored_H'][p] >= 0.25 for p in d['main'][:n] + d['trial'][:5]) == 9


# ---- values ---------------------------------------------------------------------------------------------------------
def test_value_rules_refuse_the_reviews_counterexamples():
    assert sf.value_of('I cannot determine the answer.', MC)[0] is None  # 'I' is not a choice letter
    assert sf.value_of('\\boxed{B}', MC)[0] == '20' and sf.value_of('\\boxed{(C)}', MC)[0] == '30'
    assert sf.value_of('\\boxed{20}', MC)[0] == '20' and sf.value_of('\\boxed{maybe}', MC)[0] is None
    assert sf.value_of('\\boxed{D}', MC)[0] is None and sf.value_of('\\boxed{}', NUM)[0] is None


def result(text, finish='stop', ent=None):
    ids = TOK.encode(text) + ([EOS] if finish == 'stop' else [])
    return Result(ids=ids, ent=ent if ent is not None else [0.5] * len(ids), finish=finish)


def test_free_record_validity():
    r = sf.free_record(TOK, SP, result('so \\boxed{2}'), NUM)
    assert r['valid'] and r['value'] == '2' and r['correct'] == 1 and r['finish'] == 'eos'
    assert r['ent_sum'] == round(0.5 * r['n_tokens'], 5) and r['ent_mean'] == 0.5
    assert not sf.free_record(TOK, SP, result('so \\boxed{2}', 'cap'), NUM)['valid']  # cut at the cap: invalid
    no_box = sf.free_record(TOK, SP, result('The answer is 1/2.'), NUM)
    assert not no_box['valid'] and no_box['value'] is None and no_box['prediction'] == '2'  # the scorer would say 2
    # the scored text starts after the last </think> the answer writes (methods.py)
    ids = TOK.encode('\\boxed{3}') + [TE] + TOK.encode('\\boxed{2}') + [EOS]
    r = sf.free_record(TOK, SP, Result(ids=ids, ent=[0.0] * len(ids), finish='stop'), NUM)
    assert r['value'] == '2' and r['n_think_end'] == 1


def test_short_record_keeps_the_main_runs_probe_rule_for_H():
    # '{D}' with choices A-C: the main run's probe accepts it (finite H, stops); its value is refused (S_val / F_val count it
    # as invalid). H1 and S_tok follow the main run's rule, as the plan says.
    ids = TOK.encode('{D}')
    r = sf.short_record(TOK, SP, Result(ids=ids, ent=[0.001] * len(ids), ent_lump=[0.001] * len(ids), finish='stop'), MC)
    assert r['probe_valid'] and not r['valid'] and r['value'] is None and r['H'] is not None and r['H'] < 0.25


# ---- contexts and run ----------------------------------------------------------------------------------------------
def fitem(lm, pid='1', seed=0, end='close'):
    g = torch.Generator().manual_seed(seed)
    base = lm.get_input_embeddings()(torch.randint(5, 90, (10,), generator=g)).detach()
    sep = TOK.encode('\n') if end == 'cap' else []
    return {'pid': pid, 'row': NUM, 'base': base, 'z': TOK.encode('xy'), 'sep': sep, 'recon': {}}


def test_contexts_and_requests_follow_the_plan(lm):
    eng = Engine(lm)
    sp = {**SP, 'close_ids': frozenset(i for i, s in enumerate(TOK.vocab) if s.startswith('}'))}
    it = fitem(lm, end='cap')
    ctx = sf.answer_context(it, eng, sp)
    assert torch.equal(ctx, torch.cat([it['base'], eng.embed(TOK.encode('xy\n') + [TE])]))
    reqs = sf.requests_of(it, eng, sp, CFG)
    assert [(k, j) for k, j, _ in reqs] == [('short', j) for j in range(3)] + [('free', j) for j in range(4)]
    for k, j, r in reqs:
        assert r.sampling == isn.T1 and r.record_entropy and r.stop_ids == frozenset({EOS})
        if k == 'short':
            assert torch.equal(r.ctx, torch.cat([ctx, eng.embed(sp['probe_prompt'])])) and r.max_new == 4
            assert r.ent_lump == sp['close_ids'] and r.stop_check is not None
        else:
            assert torch.equal(r.ctx, ctx) and r.max_new == 6 and r.stop_check is None and not r.ent_lump
    seeds = {(k, j): r.gen.initial_seed() for k, j, r in reqs}
    assert len(set(seeds.values())) == 7 and seeds[('free', 0)] == sf.seed_of(42, '1', 'free', 0)


def test_run_group_records_every_answer(lm):
    torch.set_grad_enabled(False)
    try:
        eng = Engine(lm)
        sp = {**SP, 'close_ids': frozenset(i for i, s in enumerate(TOK.vocab) if s.startswith('}'))}
        its = [fitem(lm, '1'), fitem(lm, '2', seed=1, end='cap')]
        sf.run_group(its, eng, TOK, sp, CFG, sf.make_generate(eng, CFG, pack))
        doc = {'stratum': {'1': 'close', '2': 'cap'}, 'stored_H': {'1': 0.01, '2': None}}
        for it in its:
            assert not it['nonfinite'] and len(it['short']) == 3 and len(it['free']) == 4
            assert all(a['n_tokens'] <= 4 for a in it['short']) and all(a['n_tokens'] <= 6 for a in it['free'])
            row = sf.row_out(it, CFG, 'v', 'main', doc)
            json.dumps(row)
            assert row['status'] == 'ok' and row['stratum'] == doc['stratum'][it['pid']]
        # one generate call per kind: run.pack budgets a group by its last request, so max_new must not mix
        calls = []
        gen = sf.make_generate(eng, CFG, pack)
        sf.run_group([fitem(lm, '3')], eng, TOK, sp, CFG, lambda reqs: calls.append({r.max_new for r in reqs}) or gen(reqs))
        assert calls == [{4}, {6}]
        # same answers on a second run (fixed seeds per question, kind and index)
        again = [fitem(lm, '1')]
        sf.run_group(again, eng, TOK, sp, CFG, sf.make_generate(eng, CFG, pack))
        assert again[0]['free'] == its[0]['free'] and again[0]['short'] == its[0]['short']
    finally:
        torch.set_grad_enabled(True)


def test_main_end_to_end_with_a_stub_model(lm, tmp_path, monkeypatch):
    _, args0 = stub(lm, tmp_path, monkeypatch)
    monkeypatch.setattr(sf, 'Engine', lambda lm_, t, k, p, fast_kv=True: Engine(lm_, t, k, p, fast_kv=False))
    store = {p: {'thinking': 'ab', 'think_tokens': 2, 'think_end': e} for p, e in
             (('1', 'close'), ('2', 'eos'), ('3', 'cap'), ('4', 'close'), ('5', 'close'))}
    doc = {'main': ['1', '2', '3', '4'], 'trial': ['5'], 'stratum': {p: s['think_end'] for p, s in store.items()},
           'stored_H': {p: 0.01 for p in store}}
    (tmp_path / 'store.json').write_text(json.dumps({'pids': store}))
    (tmp_path / 'sf_ids.json').write_text(json.dumps(doc))
    out = tmp_path / 'res' / 'sf.jsonl'
    args = ['--parquet', 'x', '--ids', str(tmp_path / 'sf_ids.json'), '--samples', str(tmp_path / 'store.json'), '--out', str(out),
            '--n-short', '2', '--n-free', '2', '--probe-cap', '4', '--answer-cap', '5', '--batch-tokens', '4000', '--group', '2']
    try:
        sf.main(args)
        res = [json.loads(l) for l in out.read_text().splitlines()]
        assert [r['pid'] for r in res] == ['1', '2', '3', '4'] and all(r['status'] == 'ok' for r in res)
        assert all(r['code_version'] == sf.code_version() and r['cfg']['chunk'] == 5 for r in res)
        assert [r['stratum'] for r in res] == ['close', 'eos', 'cap', 'close']
        sf.main(args)  # resume
        assert len(out.read_text().splitlines()) == 4
        # an error row is run again on resume; ok rows are not
        lines = out.read_text().splitlines()
        out.write_text('\n'.join(lines[:3] + [json.dumps({'pid': '4', 'status': 'error', 'stage': 'generate', 'error': 'x'})]) + '\n')
        sf.main(args)
        res = [json.loads(l) for l in out.read_text().splitlines()]
        assert [(r['pid'], r['status']) for r in res] == [('1', 'ok'), ('2', 'ok'), ('3', 'ok'), ('4', 'error'), ('4', 'ok')]
        R, _ = an.load([str(out)])
        assert R['4']['status'] == 'ok'
    finally:
        torch.set_grad_enabled(True)


# ---- analysis -------------------------------------------------------------------------------------------------------
def test_value_entropy_bounds_from_the_review():
    assert an.value_entropy([None] * 16) == (0.0, math.log(16))
    lo, hi = an.value_entropy(['1'] * 15 + [None])
    assert lo == 0.0 and abs(hi - 0.23379) < 1e-5  # one invalid never makes the value entropy look surer than lo
    assert abs(an.value_entropy(['1'] * 14 + ['2'] * 2)[0] - 0.37677) < 1e-5
    assert an.value_entropy(['1'] * 15 + ['2'])[0] < 0.25 < an.value_entropy(['1'] * 7 + ['2'])[0]


def test_clopper_pearson_matches_known_values():
    lo, hi = an.cp_range(36, 356)
    assert abs(lo - 0.071836) < 1e-5 and abs(hi - 0.137244) < 1e-5
    assert an.cp_range(0, 50)[0] == 0.0 and an.cp_range(50, 50)[1] == 1.0


def arow(pid, free_vals, stored=0.01, stratum='close', short_vals=None, qt='free_form'):
    fr = [{'value': v, 'valid': v is not None, 'correct': int(v == '7'), 'finish': 'eos', 'boxed': True, 'n_tokens': 20,
           'ent_sum': 1.0, 'ent_mean': 0.05} for v in free_vals]
    sh = [{'value': v, 'valid': v is not None, 'correct': 0, 'H': 0.01 if v is not None else None, 'n_tokens': 4}
          for v in (short_vals or ['7'] * 16)]
    return {'pid': pid, 'status': 'ok', 'stratum': stratum, 'stored_H': stored, 'question_type': qt, 'code_version': 'v',
            'short': sh, 'free': fr}


def rows(n_cont, n=200, invalid_cont=False):
    """n self-ended rows, the first n_cont with mixed free values (14 : 2), the rest unanimous."""
    R = {}
    for i in range(n):
        if i < n_cont:
            fv = ['7'] * 15 + [None] if invalid_cont else ['7'] * 14 + ['8'] * 2
        else:
            fv = ['7'] * 16
        R[str(i)] = arow(str(i), fv)
    return R


def test_decisions():
    R = rows(60)
    d = an.analyze(R, {'analysis': list(R)})
    assert d['sentence'] == an.MOVE and d['decision']['b_new_continue'] == 60 and d['decision']['c_new_stop'] == 0
    R = rows(5)
    assert an.analyze(R, {'analysis': list(R)})['sentence'] == an.STAY
    R = rows(22)
    assert an.analyze(R, {'analysis': list(R)})['sentence'] == an.UNDECIDED
    # invalid answers: lo assignment cannot make MOVE, hi assignment keeps STAY from being claimed
    R = rows(60, invalid_cont=True)
    d = an.analyze(R, {'analysis': list(R)})
    assert d['decision']['k_lo'] == 0 and d['decision']['k_hi'] == 0  # 15 : 1 invalid stays below 0.25 either way
    R = {str(i): arow(str(i), [None] * 16 if i < 60 else ['7'] * 16) for i in range(200)}
    d = an.analyze(R, {'analysis': list(R)})
    assert d['decision']['k_lo'] == 0 and d['decision']['k_hi'] == 60 and d['sentence'] == an.UNDECIDED


def test_stored_continuing_questions_count_against_move():
    R = rows(60)
    for i in range(60, 120):
        R[str(i)]['stored_H'] = None  # stored continues, F_val stops
    d = an.analyze(R, {'analysis': list(R)})['decision']
    assert d['b_new_continue'] == 60 and d['c_new_stop'] == 60 and d['sentence'] != an.MOVE


def test_hold_and_cut_rows():
    R = rows(5, n=20)
    R['c1'] = arow('c1', ['7'] * 16, stratum='cap')
    out = an.analyze(R, {'analysis': list(R)})
    assert out['decision']['n'] == 20 and out['descriptive']['cut']['n'] == 1  # cut rows are described, never decided
    R['e'] = {'pid': 'e', 'status': 'error'}
    assert an.analyze(R, {'analysis': list(R)})['sentence'] == an.HOLD
    assert an.analyze(rows(5, n=10), {'analysis': [str(i) for i in range(11)]})['sentence'] == an.HOLD


def test_builder(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, 'HERE', tmp_path)
    assert bs.main(['s', '--owner', 'me']) == 0
    src = ''.join(json.loads((tmp_path / 'runs/s/notebook.ipynb').read_text())['cells'][0]['source'])
    ast.parse(src)
    assert "'diag/stop_form.py'" in src and "f'stop_form_{sh[i]}.jsonl'" in src and "'diag/stop_form_ids.json'" in src
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(re.search(r"b64decode\('([^']+)'\)\)\)\.extractall", src).group(1))))
    for f in ('diag/stop_form.py', 'diag/blank_think.py', 'diag/inject_sensitivity.py', 'diag/closed_multistep.py',
              'diag/step1_think.py', 'diag/stop_form_ids.json', 'diag/step1_store_bmain.json'):
        assert f in z.namelist(), f


def test_no_absolute_sandbox_paths_in_the_new_files():
    bad = ('/ho' + 'me/', '/t' + 'mp/', 'scratch' + 'pad', '/Us' + 'ers/')
    for f in ('diag/stop_form.py', 'diag/stop_form_ids.py', 'diag/analyze_stop_form.py', 'kaggle/build_stop_form.py',
              'tests/test_stop_form.py'):
        assert not any(b in (REPRO / f).read_text() for b in bad), f
