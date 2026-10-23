"""CPU tests for diag/closed_multistep_ids.py, diag/closed_multistep.py, diag/analyze_closed_multistep.py,
diag/count_closed_multistep_interim.py and kaggle/build_closed_multistep.py (fake engine / tiny LM)."""
import ast
import base64
import io
import json
import re
import shutil
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

REPRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO / 'diag'))
sys.path.insert(0, str(REPRO / 'kaggle'))
import analyze_closed_multistep as an  # noqa: E402
import build_closed_multistep as bc  # noqa: E402
import closed_multistep as cm  # noqa: E402
import closed_multistep_ids as ids_mod  # noqa: E402
import count_closed_multistep_interim as ci  # noqa: E402
import step1_think as t1  # noqa: E402
import thumb_gate as tg  # noqa: E402
from test_hs import HSEngine  # noqa: E402
from test_repro import make_job, lm  # noqa: E402,F401  (lm is a pytest fixture)
from test_step1_think import NL, SP as SP0, TOK, FakeEng  # noqa: E402

from run import pack  # noqa: E402
from visref.dpp import budget, select_tokens  # noqa: E402
from visref.engine import Engine, Request, Result  # noqa: E402
from visref.methods import Job  # noqa: E402

SP = {**SP0, 'close_ids': frozenset(i for i, s in enumerate(TOK.vocab) if s.startswith('}'))}
TE, EOS = SP['think_end'], SP['eos']
ROW = {'question_type': 'free_form', 'answer_type': 'integer', 'answer': '7', 'precision': None, 'choices': None}
CFG = cm.resolve_cfg({**cm.DEFAULTS, 'think_cap': 6, 'probe_cap': 4, 'n_t1': 2, 'batch_tokens': 4000, 'memcheck_tokens': 40})
WAIT, PROMPT = SP['wait'], SP['probe_prompt']


# ---- ids list ---------------------------------------------------------------------------------------------------
def test_ids_file_equals_a_fresh_build_and_follows_the_plan():
    assert ids_mod.main(['--check']) == 0
    assert ids_mod.OUT.read_text() == ids_mod.text(ids_mod.fresh()) == ids_mod.text(ids_mod.fresh())
    d = json.loads(ids_mod.OUT.read_text())
    order = [str(p) for p in json.loads(ids_mod.ORDER.read_text())]
    store = json.loads(ids_mod.STORE.read_text())['pids']
    want = [p for p in order if store[p]['think_end'] == 'close']
    assert d['main'] == want and len(want) == len(set(want)) == 334
    assert d['part1'] == want[:112] and d['part2'] == want[112:] and len(d['part2']) == 222
    assert d['part1'] + d['part2'] == d['main']
    assert d['trial'] == want[:5] and d['memcheck'] in d['trial'] and d['n_visual'][d['memcheck']] == 3328
    assert set(d['n_visual']) == set(d['stratum']) == set(want) and set(d['stratum'].values()) == {'close'}
    assert sum(v == 3328 for v in d['n_visual'].values()) == 97  # the plan's count of 3,328-token questions among the 334


def test_ids_build_adds_a_big_question_to_the_trial_only_when_the_first_five_have_none():
    order = [str(i) for i in range(10)]
    store = {p: {'think_end': 'close' if int(p) != 3 else 'cap'} for p in order}
    nvis = {p: 256 for p in order}
    with pytest.raises(AssertionError):
        ids_mod.build(order, store, nvis, n_main=9)  # no 3,328-image question anywhere
    nvis['8'] = 3328
    d = ids_mod.build(order, store, nvis, n_main=9)
    assert d['trial'] == ['0', '1', '2', '4', '5', '8'] and d['memcheck'] == '8'
    nvis['1'] = 3328
    d = ids_mod.build(order, store, nvis, n_main=9)
    assert d['trial'] == ['0', '1', '2', '4', '5'] and d['memcheck'] == '1'
    with pytest.raises(AssertionError):
        ids_mod.build(order, store, nvis, n_main=334)


# ---- the V block equals what methods.py TAG_IN builds -----------------------------------------------------------------
class RecJob:
    """Drive a visref_hs_imgtag_in Job with scripted steps and record its contexts per step."""

    def __init__(self, steps, nvis=10, zero_vis=False):
        job0, _, self.tok = make_job('visref_hs_imgtag_in', [], kmax=len(steps), nvis=nvis)
        self.eng = HSEngine(self.tok, [])
        self.base = self.eng.embed(self.tok.encode('Q?'))
        vis = torch.zeros_like(job0.vis) if zero_vis else job0.vis
        self.job = Job('visref_hs_imgtag_in', job0.row, self.base, vis, job0.cfg, self.tok, self.eng, job0.sp,
                       torch.Generator(), torch.Generator())
        self.vis, self.sp, self.cfg = vis, job0.sp, job0.cfg
        self.seen = []  # per step: dict(step_ctx, answer_ctx, next_start (None after the last step), sel, m, hs_calls)
        for k, (ids, finish) in enumerate(steps):
            self._consume(ids, finish, 0.0)
            j = self.job
            rec = {'step_ctx': j.step_ctx, 'answer_ctx': j.answer_ctx, 'ctx_nov': j.ctx_nov, 'sel': j.steps[-1].get('sel'),
                   'm': j.steps[-1].get('m')}
            self._consume(self.tok.encode('{7}'), 'stop', 5.0)  # a high-entropy probe: the job goes on
            rec['next_start'] = j.request().ctx if j.phase == 'think' else None
            self.seen.append(rec)

    def _consume(self, ids, finish, h):
        req = self.job.request()
        self.job.consume(Result(ids=list(ids), ent=[h] * len(ids) if req.record_entropy else [],
                                ent_lump=[h] * len(ids) if req.ent_lump else [], finish=finish))


def mine(tok, sp, base, vis, steps, m, eng):
    """closed_multistep's V pipeline on the same scripted steps: [(step_ctx, answer_ctx(V), start of the next step)] per step."""
    s = {'start': base, 'steps': [], 'probes': []}
    it = {'pid': '1', 'base': base, 'vis': vis, 'm': m, 'st': {'V': s}}
    out = []
    for k, (ids, finish) in enumerate(steps, 1):
        z, sep, sep_text, end = cm.split_generated(tok, sp, ids, '\n')
        cm.close_step(it, 'V', k, (z, sep, sep_text, end, 'generated'), 'h', eng, tok, sp)
        step_ctx, end_ctx = s['step_ctx'], s['end_ctx']
        s['start'] = torch.cat([step_ctx, eng.embed(sp['wait'])], 0)
        out.append({'step_ctx': step_ctx, 'end_ctx': end_ctx, 'answer_ctx': torch.cat([step_ctx, eng.embed([sp['think_end']])], 0),
                    'next_start': s['start'], 'rec': s['steps'][-1]})
    return out


def scripted_steps(tok):
    a = tok.encode
    return [(a('aaaaaaaa') + [1], 'stop'),    # closed, text without a trailing newline: one '\n' before <img>
            (a('bbbb\n') + [1], 'stop'),      # closed, text ends with a newline: none added
            (a('cccccccc'), 'cap'),           # cut at the cap: separator '\n' supplies the newline
            (a('dd') + [0], 'stop')]          # EOS ends the step, EOS dropped


@pytest.mark.parametrize('zero_vis', [False, True])
def test_v_block_layout_equals_methods_tag_in(zero_vis):
    tok = TOK
    steps = scripted_steps(tok)
    rj = RecJob(steps, zero_vis=zero_vis)  # zero_vis: DPP picks nothing; methods.py still writes the tags
    eng2 = HSEngine(tok, [])
    got = mine(tok, rj.sp, rj.base, rj.vis, steps, budget(rj.vis.shape[0], 0.3), eng2)
    assert len(got) == len(rj.job.steps) == 4
    for k, (g, ref) in enumerate(zip(got, rj.seen)):
        assert torch.equal(g['step_ctx'], ref['step_ctx']), k   # context after step k with its block
        assert torch.equal(g['answer_ctx'], ref['answer_ctx']), k  # block, then </think>
        assert torch.equal(g['end_ctx'], ref['ctx_nov']), k     # text without the block
        if ref['next_start'] is not None:
            assert torch.equal(g['next_start'], ref['next_start']), k  # + 'Wait'
        assert g['rec']['sel'] == ref['sel'] and g['rec']['m'] == ref['m'], k
        assert g['rec']['n_sel'] == len(ref['sel']) and g['rec']['block'] is True
    assert eng2.hs_calls == rj.eng.hs_calls and len(eng2.hs_calls) == 4  # Z: hidden states of z_k in its own context
    if zero_vis:
        assert all(g['rec']['sel'] == [] for g in got)
    else:
        assert all(len(g['rec']['sel']) > 0 for g in got)
    # the layout in tokens: one newline before <img> only after steps 1 (no trailing newline) and none after 2-4
    ids = lambda c: [int(i) for i in c.argmax(1)]  # noqa: E731
    n1 = len(got[0]['rec']['sel'])
    blk = got[0]['step_ctx'][rj.base.shape[0] + 8:]
    assert ids(blk[:2]) == [NL[0], 2] and ids(blk[2 + n1:]) == [3, NL[0]]
    blk2 = got[1]['step_ctx'][got[0]['next_start'].shape[0] + 5:]
    assert ids(blk2[:1]) == [2] and ids(blk2[1 + len(got[1]['rec']['sel']):]) == [3, NL[0]]


def test_v_block_for_an_empty_step_is_not_built_like_methods():
    tok = TOK
    steps = [(tok.encode('aaaaaaaa') + [1], 'stop'), ([1], 'stop'), (tok.encode('ccc') + [1], 'stop')]
    rj = RecJob(steps)
    got = mine(tok, rj.sp, rj.base, rj.vis, steps, budget(rj.vis.shape[0], 0.3), HSEngine(tok, []))
    assert got[1]['rec'] == {**got[1]['rec'], 'block': False, 'n_sel': 0, 'sel': [], 'len': 0, 'block_len': 0}
    assert torch.equal(got[1]['step_ctx'], got[1]['end_ctx']) and got[1]['end_ctx'].shape[0] == got[0]['next_start'].shape[0]
    for g, ref in zip(got, rj.seen):
        assert torch.equal(g['step_ctx'], ref['step_ctx'])
        if ref['next_start'] is not None:
            assert torch.equal(g['next_start'], ref['next_start'])
    assert 'sel' not in rj.job.steps[1]  # methods.py: no selection after an empty step


def test_step_end_split_equals_thumb_gate_and_methods_rules():
    a = TOK.encode
    for ids in (a('abc') + [TE], a('abc') + [EOS], a('abc'), a('ab\n'), [], [TE], [EOS], a('x\n') + [TE]):
        assert cm.split_generated(TOK, SP, ids, '\n') == tg.split_generated(TOK, SP, ids, '\n')
    z, sep, sep_text, end = cm.split_generated(TOK, SP, a('abc') + [TE], '\n')
    assert (z, sep, sep_text, end) == (a('abc'), [], '', 'close')     # </think> dropped, thinking goes on
    assert cm.split_generated(TOK, SP, a('abc'), '\n') == (a('abc'), NL, '\n', 'cap')   # cut: cut_sep appended
    assert cm.split_generated(TOK, SP, a('ab\n'), '\n') == (a('ab\n'), [], '', 'cap')   # ... unless the text ends with one
    assert cm.split_generated(TOK, SP, a('abc') + [EOS], '\n') == (a('abc'), [], '', 'eos')  # EOS dropped


# ---- run_group with a scripted generate -----------------------------------------------------------------------------------
def item(eng, pid='5', z='think abc', n_vis=10, seed=0, end='close'):
    g = torch.Generator().manual_seed(seed)
    zi = TOK.encode(z)
    sep_text = '\n' if end == 'cap' and not z.endswith('\n') else ''
    return {'pid': pid, 'row': ROW, 'sample': {'think_end': end}, 'base': eng.embed(torch.randint(5, 90, (6,), generator=g)),
            'vis': torch.randn(n_vis, eng.emb.embedding_dim, generator=g), 'z': zi, 'sep': TOK.encode(sep_text),
            'sep_text': sep_text, 'recon': {'tokens_match': True, 'think_end_stored': end}}


class Spy:
    """Scripted generate: thinking requests (max_new == think_cap) get script[(item index, cond, k)] (default 'ok' + </think>);
    probes and final answers get '{7}' with fixed entropies. calls = [('think' | 'ans', reqs)] in order."""

    def __init__(self, script=None, answer='{7}'):
        self.script, self.answer, self.calls = script or {}, answer, []

    def __call__(self, reqs):
        kind = 'think' if reqs[0].max_new == CFG['think_cap'] else 'ans'
        self.calls.append((kind, reqs))
        k = sum(c[0] == 'think' for c in self.calls) + 1
        out = []
        for idx, r in enumerate(reqs):
            if kind == 'think':
                i, c = divmod(idx, 2)
                ids, fin = self.script.get((i, 'TV'[c], k), (TOK.encode('ok') + [TE], 'stop'))
                out.append(Result(ids=list(ids), finish=fin))
            else:
                ids = TOK.encode(self.answer)
                out.append(Result(ids=ids, ent=[1.0] * len(ids) if r.record_entropy else [],
                                  ent_lump=[0.5] * len(ids) if r.ent_lump else [], finish='stop'))
        return out


def manual_contexts(eng, it, zs, sep_texts, m):
    """Independent re-implementation of the contexts of T and V for the texts zs (list of id lists) and separators."""
    out = {'T': [], 'V': []}
    nl = TOK.encode('\n')
    for c in ('T', 'V'):
        ctx = it['base']
        for k, (z, st) in enumerate(zip(zs[c], sep_texts[c])):
            end = torch.cat([ctx, eng.embed(z + TOK.encode(st))])
            nxt = end
            if c == 'V' and z:
                Z = eng.last_hidden(torch.cat([ctx, eng.embed(z)]), len(z))
                sel = select_tokens(it['vis'], Z, m)
                pre = [] if (TOK.decode(z) + st).endswith('\n') else nl
                nxt = torch.cat([end, eng.embed(pre), eng.embed([SP['img_start']]), it['vis'][sel], eng.embed([SP['img_end']] + nl)])
            out[c].append({'start': ctx, 'end': end, 'step': nxt})
            ctx = torch.cat([nxt, eng.embed(WAIT)])
    return out


def test_run_group_contexts_steps_probes_and_final_answers():
    eng = FakeEng()
    its = [item(eng, pid=p, seed=k) for k, p in enumerate(('3', '9'))]
    a = TOK.encode
    script = {(0, 'T', 2): (a('abc') + [TE], 'stop'), (0, 'V', 2): (a('abc\n') + [TE], 'stop'),
              (0, 'T', 3): (a('xyz'), 'cap'), (0, 'V', 3): (a('uvw'), 'cap'),
              (0, 'T', 4): (a('q') + [EOS], 'stop'), (0, 'V', 4): (a('p\n'), 'cap')}
    spy = Spy(script)
    cm.run_group(its, eng, TOK, SP, CFG, spy)
    kinds = [k for k, _ in spy.calls]
    assert kinds == ['ans', 'think', 'ans', 'think', 'ans', 'think', 'ans']  # probe1, step2, probe2, step3, probe3, step4, finals
    n = len(its)
    assert [len(r) for _, r in spy.calls] == [2 * n, 2 * n, 2 * n, 2 * n, 2 * n, 2 * n, 4 * n * (CFG['n_t1'] + 1)]
    it = its[0]
    m = budget(10, CFG['vis_frac'])
    zs = {'T': [it['z'], a('abc'), a('xyz'), a('q')], 'V': [it['z'], a('abc\n'), a('uvw'), a('p\n')]}
    sts = {'T': ['', '', '\n', ''], 'V': ['', '', '\n', '']}  # 'abc\n' / 'p\n' end with a newline: no separator
    man = manual_contexts(eng, it, zs, sts, m)
    probe_calls = [r for k, r in spy.calls if k == 'ans'][:3]
    think_calls = [r for k, r in spy.calls if k == 'think']
    for k in range(3):  # probes read the context BEFORE the block; thinking requests start after the block and the cue
        for ci_, c in enumerate('TV'):
            assert torch.equal(probe_calls[k][ci_].ctx, torch.cat([man[c][k]['end'], eng.embed([TE] + PROMPT)])), (k, c)
            assert torch.equal(think_calls[k][ci_].ctx, man[c][k + 1]['start']), (k, c)
    assert man['V'][1]['step'].shape[0] > man['V'][1]['end'].shape[0]
    final = spy.calls[-1][1]
    n1 = CFG['n_t1'] + 1
    expect = {'S': torch.cat([it['base'], eng.embed(it['z'])]), 'T': man['T'][3]['end'], 'V': man['V'][3]['step'],
              'Vminus': man['V'][3]['end']}
    for ci_, c in enumerate(cm.CONDS):
        for r in final[ci_ * n1:(ci_ + 1) * n1]:
            assert torch.equal(r.ctx, torch.cat([expect[c], eng.embed([TE] + PROMPT)])), c
    # rows: exactly 4 steps, 4 probes, answers of all four conditions
    for x in its:
        assert not x['nonfinite']
        for c in 'TV':
            assert [s['k'] for s in x['st'][c]['steps']] == [1, 2, 3, 4] and [p['k'] for p in x['st'][c]['probes']] == [1, 2, 3, 4]
        assert set(x['answers']) == set(cm.CONDS) and all(len(v) == n1 for v in x['answers'].values())
    st = it['st']
    assert [s['end'] for s in st['T']['steps']] == ['close', 'close', 'cap', 'eos']
    assert [s['end'] for s in st['V']['steps']] == ['close', 'close', 'cap', 'cap']
    assert [s['sep_text'] for s in st['T']['steps']] == ['', '', '\n', ''] and st['T']['steps'][0]['source'] == 'stored'
    assert [s['len'] for s in st['T']['steps']] == [9, 3, 3, 1] and st['T']['steps'][2]['ids'] == a('xyz')
    assert [s['block'] for s in st['V']['steps']] == [True] * 4 and 'block' not in st['T']['steps'][0]
    assert st['V']['steps'][1]['m'] == m and st['V']['steps'][1]['n_sel'] == len(st['V']['steps'][1]['sel'])
    # the step-4 probe is the first sampled final answer of T / Vminus, with its entropies
    assert st['T']['probes'][3]['source'] == 'answer0_of_T' and st['V']['probes'][3]['source'] == 'answer0_of_Vminus'
    assert st['T']['probes'][3]['H'] == it['answers']['T'][0]['H'] == 2.0 and 'greedy' not in st['T']['probes'][3]
    assert st['T']['probes'][0]['H'] == 2.0 and st['T']['probes'][0]['ent'] == [1.0, 0.5, 0.5] and st['T']['probes'][0]['correct'] == 1
    assert it['answers']['S'][-1]['greedy'] is True and it['answers']['S'][0]['greedy'] is False


def test_empty_generated_step_gets_no_block():
    eng = FakeEng()
    its = [item(eng)]
    spy = Spy({(0, 'V', 3): ([TE], 'stop'), (0, 'T', 3): ([TE], 'stop')})
    cm.run_group(its, eng, TOK, SP, CFG, spy)
    v = its[0]['st']['V']['steps']
    assert v[2]['len'] == 0 and v[2]['block'] is False and v[2]['n_sel'] == 0 and v[2]['sel'] == []
    assert v[1]['block'] is True and v[3]['block'] is True
    think = [r for k, r in spy.calls if k == 'think']
    # the request of step 4 follows step 3's empty text directly: [step-3 start][cue]
    prev = think[1][1].ctx  # V, step 3 start
    assert torch.equal(think[2][1].ctx, torch.cat([prev, eng.embed(WAIT)]))
    assert its[0]['st']['T']['steps'][2]['len'] == 0


def test_seeds_are_shared_across_conditions_and_one_generator_serves_all_steps():
    eng = FakeEng()
    its = [item(eng, pid=p, seed=k) for k, p in enumerate(('3', '9'))]
    spy = Spy()
    cm.run_group(its, eng, TOK, SP, CFG, spy)
    thinks = [r for k, r in spy.calls if k == 'think']
    assert len(thinks) == 3
    for i, x in enumerate(its):
        for ci_, c in enumerate('TV'):
            gens = [t[2 * i + ci_].gen for t in thinks]
            assert gens[0] is gens[1] is gens[2]                                  # one generator over all steps
            assert gens[0].initial_seed() == cm.seed_of(42, x['pid'], 'think')    # the same seed for T and V
            assert all(t[2 * i + ci_].stop_ids == frozenset({TE, EOS}) and t[2 * i + ci_].max_new == CFG['think_cap'] for t in thinks)
        assert thinks[0][2 * i].gen is not thinks[0][2 * i + 1].gen
    probes = [r for k, r in spy.calls if k == 'ans'][:3]
    for kk, call in enumerate(probes, 1):
        for i, x in enumerate(its):
            for ci_ in range(2):
                r = call[2 * i + ci_]
                assert r.gen.initial_seed() == cm.seed_of(42, x['pid'], 'probe', kk) and r.sampling == t1.T1
                assert r.record_entropy and r.max_new == CFG['probe_cap']
    final = spy.calls[-1][1]
    n1 = CFG['n_t1'] + 1
    for i, x in enumerate(its):
        for ci_ in range(4):
            chunk = final[(i * 4 + ci_) * n1:(i * 4 + ci_ + 1) * n1]
            seeds = [r.gen.initial_seed() if r.gen is not None else None for r in chunk]
            assert seeds == [cm.seed_of(42, x['pid'], 'ans', j) for j in range(CFG['n_t1'])] + [None]  # same in S, T, V, V-
            assert chunk[0].sampling == t1.T1 and chunk[-1].sampling == t1.GREEDY and chunk[0].max_new == CFG['probe_cap']
    # the recorded state hashes: step 1 and 2 (before any draw) are equal for T and V
    for x in its:
        h = {c: [s['rng_hash'] for s in x['st'][c]['steps']] for c in 'TV'}
        assert h['T'] == h['V'] and all(len(v) == 12 for v in h['T'])


def test_non_finite_logits_give_an_error_row_and_other_questions_go_on():
    eng = FakeEng()
    its = [item(eng, pid=p, seed=k) for k, p in enumerate(('3', '9'))]
    inner = Spy()

    def spy(reqs):
        out = inner(reqs)
        if inner.calls[-1][0] == 'think' and len(inner.calls) == 2:
            out[1] = None  # item 0, condition V, step 2
        return out

    cm.run_group(its, eng, TOK, SP, CFG, spy)
    assert its[0]['nonfinite'] == ['V'] and its[1]['nonfinite'] == []
    row0, row1 = cm.row_out(its[0], CFG, 'v', 'main'), cm.row_out(its[1], CFG, 'v', 'main')
    assert row0['status'] == 'error' and row0['nonfinite'] == ['V'] and 'non-finite' in row0['error'] and row0['code_version'] == 'v'
    assert row1['status'] == 'ok' and row1['n_correct'] == {c: CFG['n_t1'] for c in cm.CONDS} and row1['acc_t1']['V'] == 1.0
    json.dumps(row1)


# ---- chunk guard ----------------------------------------------------------------------------------------------------------
class GenEng(FakeEng):
    """Engine.generate's finish rules (stop > cap > chunk) with a constant token that never stops by itself."""

    def generate(self, reqs, chunk, stop_at=None):
        out = []
        for r in reqs:
            o = Result()
            while True:
                o.ids.append(30)
                if 30 in r.stop_ids or (r.stop_check is not None and r.stop_check(o.ids)):
                    o.finish = 'stop'
                elif len(o.ids) >= r.max_new:
                    o.finish = 'cap'
                elif len(o.ids) >= chunk:
                    o.finish = 'chunk'
                else:
                    continue
                break
            out.append(o)
        return out


def test_chunk_is_refused_below_think_cap_and_a_chunk_finish_raises():
    d = cm.DEFAULTS
    assert d['think_cap'] == 1024 and d['chunk'] == 0 and d['n_t1'] == 3 and d['probe_cap'] == 16 and d['vis_frac'] == 0.3
    assert d['temperature'] == 0.6 and d['top_k'] == 50 and d['top_p'] == 0.95 and d['seed'] == 42 and d['group'] == 4
    assert d['probe_prompt'] == '\n\n\\boxed' and d['wait_text'] == 'Wait, think more' and d['cut_sep'] == '\n'
    assert cm.N_STEPS == 4 and cm.resolve_cfg(d)['chunk'] == 1024
    assert cm.resolve_cfg({**d, 'think_cap': 2048})['chunk'] == 2048
    with pytest.raises(SystemExit):
        cm.resolve_cfg({**d, 'chunk': 512})
    base = ['--parquet', 'p.parquet', '--out', 'o.jsonl']
    a = cm.parse_args(base)
    assert a.cfg['chunk'] == 1024 and a.set == 'main' and a.pids_part == 'all' and a.shard == '0/1'
    with pytest.raises(SystemExit):
        cm.parse_args(base + ['--chunk', '512'])
    with pytest.raises(SystemExit):
        cm.parse_args(base + ['--pids-part', '3'])
    eng = GenEng()
    bad = {**CFG, 'think_cap': 40, 'chunk': 16}  # not through resolve_cfg
    mk = lambda: Request(torch.randn(6, 8), frozenset({TE, EOS}), 40, None)  # noqa: E731
    assert t1.make_generate(eng, bad, pack)([mk()])[0].finish == 'chunk'
    with pytest.raises(cm.ChunkError):
        cm.make_generate(eng, bad, pack)([mk()])
    good = cm.resolve_cfg({**CFG, 'think_cap': 40, 'chunk': 0})
    r = cm.make_generate(eng, good, pack)([mk()])[0]
    assert good['chunk'] == 40 and r.finish == 'cap' and len(r.ids) == 40


def test_a_chunk_error_stops_the_run_even_in_the_error_handling():
    eng = FakeEng()
    its = {p: item(eng, pid=p, seed=int(p)) for p in ('1', '2')}

    def spy(reqs):
        raise cm.ChunkError('boom')

    with pytest.raises(cm.ChunkError):  # not turned into per-question error rows
        cm.process_group(['1', '2'], its.__getitem__, eng, TOK, SP, CFG, spy, 'v', 'main')


# ---- process_group, select_pids, code_version -----------------------------------------------------------------------------
def test_error_rows_and_per_question_retry(monkeypatch):
    eng = FakeEng()
    its = {p: item(eng, pid=p, seed=int(p)) for p in ('1', '2', '3', '4')}

    def mk(pid):
        if pid == '2':
            raise KeyError('no image for 2')
        return its[pid]

    bad = {cm.seed_of(42, '3', 'ans', 0)}
    orig, sizes, state = cm.run_group, [], {}

    def run_group(items, eng_, tok, sp, cfg, generate, timing=None):  # a fresh scripted generate for every run of the steps
        sizes.append(len(items))
        state['spy'] = Spy()

        def gen(reqs):
            if any(r.gen is not None and r.gen.initial_seed() in bad for r in reqs):
                raise RuntimeError('boom')
            return state['spy'](reqs)
        return orig(items, eng_, tok, sp, cfg, gen, timing)

    monkeypatch.setattr(cm, 'run_group', run_group)
    rows = cm.process_group(['1', '2', '3', '4'], mk, eng, TOK, SP, CFG, None, 'v', 'main')
    assert [r['pid'] for r in rows] == ['1', '2', '3', '4'] and [r['status'] for r in rows] == ['ok', 'error', 'error', 'ok']
    assert rows[1]['stage'] == 'load' and 'KeyError' in rows[1]['error'] and rows[1]['set'] == 'main'
    assert rows[2]['stage'] == 'generate' and 'boom' in rows[2]['error'] and rows[2]['code_version'] == 'v'
    assert sizes == [3, 1, 1, 1]  # the group raised, each question was run alone
    assert rows[0]['n_correct']['V'] == CFG['n_t1'] and rows[3]['code_version'] == 'v' and rows[0]['timing']['encode_s'] is not None
    json.dumps(rows)


def test_select_pids():
    doc = {'main': ['a', 'b', 'c', 'd', 'e', 'f'], 'part1': ['a', 'b', 'c'], 'part2': ['d', 'e', 'f'], 'trial': ['x', 'y']}
    sp_ = cm.select_pids
    assert sp_(doc, 'main') == doc['main'] and sp_(doc, 'trial') == ['x', 'y']
    assert sp_(doc, 'main', '1') == ['a', 'b', 'c'] and sp_(doc, 'main', '2') == ['d', 'e', 'f']
    assert sp_(doc, 'main', 'all', limit=3, done=['b']) == ['a', 'c']
    assert sp_(doc, 'main', '1', extra=['z', 'a']) == ['a', 'b', 'c', 'z']
    assert sp_(doc, 'main', 'all', shard='1/2', done=['d']) == ['b', 'f']
    assert sp_(doc, 'main', shard='0/2') + sp_(doc, 'main', shard='1/2') == ['a', 'c', 'e', 'b', 'd', 'f']
    for bad in (dict(set_name='nope'), dict(set_name='main', part='7'), dict(set_name='trial', part='1')):
        with pytest.raises(SystemExit):
            sp_(doc, **bad)


def test_code_version_includes_the_imported_files(tmp_path, monkeypatch):
    v = cm.code_version()
    assert len(v) == 12 and v == cm.code_version() and v != t1.code_version() and v != tg.code_version()
    root = tmp_path / 'repro'
    for f in ('diag/step1_think.py', 'run.py', 'visref/engine.py', 'visref/model.py'):
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPRO / f, root / f)
    monkeypatch.setattr(cm, 'ROOT', root)
    v0 = cm.code_version()
    for f in ('diag/step1_think.py', 'run.py', 'visref/engine.py', 'visref/model.py'):
        p = root / f
        old = p.read_bytes()
        p.write_bytes(old + b'\n# changed\n')
        assert cm.code_version() != v0, f
        p.write_bytes(old)
        assert cm.code_version() == v0
    (root / 'visref/new_file.py').write_text('x = 1\n')
    assert cm.code_version() != v0


def test_the_pinned_code_version_of_run_py_is_untouched():
    import run
    assert run.code_version() == '06435ae748c8'


# ---- the tiny LM end to end -----------------------------------------------------------------------------------------------
def lm_items(lm, pids=('1', '2'), n_vis=10, end='close'):
    emb = lm.get_input_embeddings()
    out = []
    for k, p in enumerate(pids):
        g = torch.Generator().manual_seed(k)
        base = emb(torch.randint(5, 90, (12,), generator=g)).detach()
        out.append({'pid': p, 'row': ROW, 'sample': {'think_end': end}, 'base': base,
                    'vis': torch.randn(n_vis, base.shape[1], generator=g), 'z': TOK.encode('think abc'), 'sep': [], 'sep_text': '',
                    'recon': {'tokens_match': True, 'think_end_stored': end}})
    return out


def tiny_run(lm, monkeypatch, force_cap=False, cfg=CFG):
    eng = Engine(lm, cfg['temperature'], cfg['top_k'], cfg['top_p'])
    if force_cap:  # the tiny random LM stops early now and then: forbid the stop tokens of the thinking requests
        orig = cm.Request
        monkeypatch.setattr(cm, 'Request', lambda ctx, stop, mx, g, **kw: orig(ctx, stop if mx == cfg['probe_cap'] else frozenset(), mx, g, **kw))
    items = lm_items(lm)
    cm.run_group(items, eng, TOK, SP, cfg, cm.make_generate(eng, cfg, pack))
    return items


def test_run_group_on_the_tiny_lm_is_deterministic_and_complete(lm, monkeypatch):
    torch.set_grad_enabled(False)
    try:
        a, b = tiny_run(lm, monkeypatch), tiny_run(lm, monkeypatch)
        for x, y in zip(a, b):
            assert not x['nonfinite']
            assert x['answers'] == y['answers']
            for c in 'TV':
                assert x['st'][c]['steps'] == y['st'][c]['steps'] and x['st'][c]['probes'] == y['st'][c]['probes']
                assert [s['k'] for s in x['st'][c]['steps']] == [1, 2, 3, 4]
                assert all(s['len'] <= CFG['think_cap'] for s in x['st'][c]['steps'][1:])  # the stored step 1 is longer
            h = {c: [s['rng_hash'] for s in x['st'][c]['steps']] for c in 'TV'}
            assert h['T'][:2] == h['V'][:2]  # same generator state until the first generated step has drawn
            row = cm.row_out(x, CFG, 'v', 'main')
            json.dumps(row)
            assert row['status'] == 'ok' and set(row['acc_t1']) == set(cm.CONDS) and set(row['steps']) == {'T', 'V'}
            assert row['steps']['T'][0]['source'] == 'stored' and row['steps']['T'][1]['source'] == 'generated'
        # S answers do not depend on the continuation: T-less check, S context is base + stored step + </think>
        eng = Engine(lm, CFG['temperature'], CFG['top_k'], CFG['top_p'])
        x = a[0]
        ctx = torch.cat([x['base'], eng.embed(x['z'] + [TE] + PROMPT)])
        r = cm.make_generate(eng, CFG, pack)([Request(ctx, frozenset({EOS}), CFG['probe_cap'], torch.Generator().manual_seed(cm.seed_of(42, x['pid'], 'ans', 0)),
                                                      sampling=t1.T1, stop_check=lambda ids: t1.box_close_index(ids, SP['brace_text']) is not None)])[0]
        assert x['answers']['S'][0]['text'] == t1.answer_record(TOK, SP, r.ids, ROW)['text']
    finally:
        torch.set_grad_enabled(True)


def test_forced_cap_steps_get_the_separator_on_the_tiny_lm(lm, monkeypatch):
    torch.set_grad_enabled(False)
    try:
        cfg = cm.resolve_cfg({**CFG, 'think_cap': 7, 'chunk': 0})
        its = tiny_run(lm, monkeypatch, force_cap=True, cfg=cfg)
        for x in its:
            for c in 'TV':
                for s in x['st'][c]['steps'][1:]:
                    assert s['len'] == 7 and s['end'] == 'cap'
                    assert s['sep_text'] == ('' if TOK.decode(s['ids']).endswith('\n') else '\n')
    finally:
        torch.set_grad_enabled(True)


# ---- main() end to end with a stubbed model ----------------------------------------------------------------------------------
def stub_main(lm, tmp_path, monkeypatch):
    import run as run_mod
    import visref.model as vm
    pids = ['1', '2', '3', '4', '5', '6']
    nvis = {'1': 12, '2': 8, '3': 12, '4': 8, '5': 12, '6': 8}
    emb = lm.get_input_embeddings()

    def encode_sample(model, tok, image, query, max_num, with_positions=False):
        g = torch.Generator().manual_seed(int(image))
        vis = torch.randn(nvis[query], emb.embedding_dim, generator=g)
        pre = emb(torch.randint(5, 90, (3,), generator=g)).detach()
        return torch.cat([pre, vis, pre.flip(0)]), vis

    rows = {p: {**ROW, 'pid': p, 'query': p, 'image': (lambda p=p: p)} for p in pids if p != '6'}  # '6' has no row
    monkeypatch.setattr(vm, 'load_mathvista', lambda path: rows)
    monkeypatch.setattr(vm, 'load_model', lambda m, r: (types.SimpleNamespace(language_model=lm), TOK))
    monkeypatch.setattr(vm, 'encode_sample', encode_sample)
    monkeypatch.setattr(vm, 'special_ids', lambda tok, w, p: {**SP, 'wait': TOK.encode(w), 'probe_prompt': TOK.encode(p)})
    monkeypatch.setattr(run_mod, 'fast_kv_selfcheck', lambda lm_, tok: (True, {}))
    monkeypatch.setattr(cm, 'Engine', lambda lm_, t, k, p, fast_kv=True: Engine(lm_, t, k, p, fast_kv=False))
    store = {'pids': {p: {'thinking': 'think abc', 'think_tokens': '9', 'think_end': 'close'} for p in pids}}
    doc = {'main': pids, 'part1': pids[:3], 'part2': pids[3:], 'trial': ['2', '1'], 'memcheck': '1',
           'stratum': {p: 'close' for p in pids}}
    (tmp_path / 'store.json').write_text(json.dumps(store))
    (tmp_path / 'ids.json').write_text(json.dumps(doc))
    out = tmp_path / 'res' / 'out.jsonl'
    args = ['--parquet', 'x', '--samples', str(tmp_path / 'store.json'), '--ids', str(tmp_path / 'ids.json'), '--out', str(out),
            '--think-cap', '6', '--probe-cap', '4', '--n-t1', '2', '--batch-tokens', '4000', '--group', '2',
            '--memcheck-tokens', '40']
    return pids, out, args


def test_main_end_to_end_with_a_stub_model(lm, tmp_path, monkeypatch, capsys):
    pids, out, args = stub_main(lm, tmp_path, monkeypatch)
    try:
        cm.main(args)
        res = [json.loads(l) for l in out.read_text().splitlines()]
        assert [r['pid'] for r in res] == pids and [r['status'] for r in res] == ['ok'] * 5 + ['error']
        assert res[5]['stage'] == 'load' and res[0]['cfg']['chunk'] == 6
        assert all(r['code_version'] == cm.code_version() and r['set'] == 'main' for r in res)
        ok = res[0]
        assert ok['n_visual'] == 12 and ok['m'] == budget(12, 0.3) and ok['stratum'] == 'close' and ok['recon']['tokens_match'] is True
        assert set(ok) == {'pid', 'status', 'set', 'stratum', 'question_type', 'n_visual', 'm', 'code_version', 'scorer_version',
                           'cfg', 'recon', 'steps', 'probes', 'answers', 'n_correct', 'acc_t1', 'greedy', 'answer_gt', 'timing'}
        assert [s['k'] for s in ok['steps']['V']] == [1, 2, 3, 4] and len(ok['probes']['T']) == 4
        assert {'encode_s', 'group_steps_s', 'group_probes_s', 'group_answers_s'} <= set(ok['timing'])
        ev = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith('{')]
        kinds = [e['event'] for e in ev]
        assert kinds[0] == 'start' and kinds[-1] == 'end' and kinds.count('group') == 3 and kinds.count('error') == 1
        g = [e for e in ev if e['event'] == 'group']
        assert all({'group_s', 'peak_alloc_gb', 'max_rss_gb', 'n_error', 'left', 'n', 'steps_s', 'probes_s', 'answers_s'} <= set(e) for e in g)
        assert not any(e['event'] == 'memcheck' for e in ev) and not Path(str(out) + '.memcheck.json').exists()
        cm.main(args)  # resume: everything is done (error rows included)
        assert len(out.read_text().splitlines()) == 6
        assert json.loads(capsys.readouterr().out.splitlines()[0])['todo'] == 0
        # part 1 / part 2 / limit / skip / extra pids
        o1, o2 = tmp_path / 'res' / 'p1.jsonl', tmp_path / 'res' / 'p2.jsonl'
        sw = lambda o: [x if x != str(out) else str(o) for x in args]  # noqa: E731
        (tmp_path / 'skip.json').write_text(json.dumps(['2']))
        cm.main(sw(o1) + ['--pids-part', '1', '--skip', str(tmp_path / 'skip.json'), '--extra-pids', '5'])
        assert [json.loads(l)['pid'] for l in o1.read_text().splitlines()] == ['1', '3', '5']
        cm.main(sw(o2) + ['--pids-part', '2', '--limit', '2'])
        assert [json.loads(l)['pid'] for l in o2.read_text().splitlines()] == ['4', '5']
        o3 = tmp_path / 'res' / 'shard.jsonl'
        cm.main(sw(o3) + ['--shard', '1/2'])
        assert [json.loads(l)['pid'] for l in o3.read_text().splitlines()] == ['2', '4', '6']
        # trial: rows tagged 'trial' and the memory check written next to the rows
        o4 = tmp_path / 'res' / 'trial.jsonl'
        capsys.readouterr()
        cm.main(sw(o4) + ['--set', 'trial'])
        got = [json.loads(l) for l in o4.read_text().splitlines()]
        assert [r['pid'] for r in got] == ['2', '1'] and all(r['set'] == 'trial' and r['status'] == 'ok' for r in got)
        mem = json.loads(Path(str(o4) + '.memcheck.json').read_text())
        assert mem['pid'] == '1' and mem['ctx_len'] == 40 and mem['n_visual'] == 12 and mem['generated_tokens'] >= 1
        assert mem['last_hidden_n'] == 6 and mem['n_sel'] > 0 and {'peak_alloc_gb', 'peak_reserved_gb'} <= set(mem)
        ev = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith('{')]
        assert [e['event'] for e in ev if e['event'] == 'memcheck'] == ['memcheck']
        with pytest.raises(SystemExit):
            cm.main(args + ['--chunk', '3'])  # refused before anything is loaded
        with pytest.raises(SystemExit):
            cm.main(sw(tmp_path / 'res' / 'z.jsonl') + ['--set', 'trial', '--pids-part', '1'])
    finally:
        torch.set_grad_enabled(True)


def test_main_stops_at_the_time_limit_and_checks_the_ids_file(lm, tmp_path, monkeypatch, capsys):
    pids, out, args = stub_main(lm, tmp_path, monkeypatch)
    try:
        cm.main(args + ['--time-limit', '0.000001', '--limit', '4'])
        ev = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith('{')]
        assert any(e['event'] == 'time_limit' for e in ev) and (not out.exists() or out.read_text() == '')
        doc = json.loads((tmp_path / 'ids.json').read_text())
        doc['stratum']['1'] = 'cap'  # the ids file disagrees with the stored step 1
        (tmp_path / 'ids.json').write_text(json.dumps(doc))
        with pytest.raises(AssertionError):
            cm.main(args)
    finally:
        torch.set_grad_enabled(True)


def test_memory_check_context_and_record(lm):
    torch.set_grad_enabled(False)
    try:
        eng = Engine(lm, 0.6, 50, 0.95)
        it = lm_items(lm, ('1',), n_vis=12)[0]
        cfg = cm.resolve_cfg({**CFG, 'think_cap': 9, 'chunk': 0})
        rec = cm.memory_check(it, eng, TOK, SP, cfg, cm.make_generate(eng, cfg, pack), 60)
        assert rec['ctx_len'] == 60 and rec['last_hidden_n'] == 9 and rec['pid'] == '1' and rec['generated_tokens'] in range(1, 9)
    finally:
        torch.set_grad_enabled(True)


# ---- analysis --------------------------------------------------------------------------------------------------------------
def make_row(pid, nc, n_t1=3, valid=True, qtype='free_form', tokens_match=True, stratum='close', set_='main', code='v',
             probe_ok=(1, 1, 1, 1), H=(0.1, 0.1, 0.1, 0.1), ids_rep=False):
    """A complete ok row; nc = {cond: number of correct sampled answers}; greedy correct iff all sampled are."""
    def ans(c):
        return ([{'text': '{7}', 'valid': valid, 'correct': int(j < nc[c]), 'greedy': False, 'H': 0.1, 'ent': [0.1]} for j in range(n_t1)]
                + [{'text': '{7}', 'valid': valid, 'correct': int(nc[c] == n_t1), 'greedy': True}])

    def steps(c):
        return [{'k': k + 1, 'source': 'stored' if k == 0 else 'generated', 'end': 'close' if k % 2 == 0 else 'cap', 'len': 100 + k,
                 'ids': [1, 2, 3] if (ids_rep and k) else [k, 5, 6], 'sep_text': '', 'rng_hash': 'h',
                 **({'m': 10, 'n_sel': 4 + k, 'sel': list(range(4 + k)), 'block': True, 'block_len': 8} if c == 'V' else {})}
                for k in range(4)]

    def probes(c):
        return [{'k': k + 1, 'text': '{7}', 'valid': True, 'correct': int(probe_ok[k]), 'H': H[k]} for k in range(4)]

    return {'pid': pid, 'status': 'ok', 'set': set_, 'stratum': stratum, 'question_type': qtype, 'code_version': code,
            'scorer_version': 's', 'cfg': {'n_t1': n_t1}, 'recon': {'tokens_match': tokens_match},
            'answers': {c: ans(c) for c in cm.CONDS}, 'n_correct': dict(nc),
            'acc_t1': {c: nc[c] / n_t1 for c in cm.CONDS}, 'steps': {c: steps(c) for c in 'TV'}, 'probes': {c: probes(c) for c in 'TV'}}


def synth(n=334, changes=(), base=1, **kw):
    """n questions with every condition at `base` correct answers except `changes` = [(index, {cond: count})]."""
    rows = {}
    ch = dict(changes)
    for i in range(n):
        nc = {c: base for c in cm.CONDS}
        nc.update(ch.get(i, {}))
        rows[str(i)] = make_row(str(i), nc, **kw)
    return rows, {'main': [str(i) for i in range(n)], 'part1': [str(i) for i in range(min(n, 112))]}


def test_sign_test_is_exact_two_sided():
    assert an.sign_p(0, 0) == 1.0 and an.sign_p(4, 0) == 0.125 and an.sign_p(5, 0) == 0.0625 and an.sign_p(0, 6) == 0.03125
    assert an.sign_p(3, 3) == 1.0 and an.sign_p(1, 0) == 1.0 and an.sign_p(8, 2) == pytest.approx(2 * 56 / 1024)


def test_bootstrap_reproduces_the_plans_4_0_and_5_0_lower_end():
    z = np.zeros(334, dtype=np.int64)
    for k in (4, 5):
        d = z.copy()
        d[:k] = 3
        st = an.pair_stats(d, z, 3)
        assert st['up'] == k and st['down'] == 0 and st['ci95_pp'][0] > 0           # the plan: lower end about +0.30 pp
        assert st['ci95_pp'][0] == pytest.approx(0.2994, abs=1e-3)
    assert an.pair_stats([0], [0], 3)['mean_pp'] == 0.0
    st = an.pair_stats([3] * 10, [0] * 10, 3)
    assert st['mean_pp'] == 100.0 and st['ci95_pp'] == [100.0, 100.0] and an.pair_stats([], [], 3)['n'] == 0


def run_verdict(changes, **kw):
    rows, ids = synth(changes=changes, **kw)
    return an.analyze(rows, ids)


def test_four_to_zero_fails_the_sign_test_but_six_to_zero_passes():
    four = run_verdict([(i, {'V': 3, 'T': 0, 'S': 0}) for i in range(4)])
    assert four['primary']['V-T']['ci95_pp'][0] > 0 and four['primary']['V-T']['p_sign'] == 0.125
    assert four['verdict']['id'] == 3 and '4 : 0' in four['verdict']['sentence']
    five = run_verdict([(i, {'V': 3, 'T': 0, 'S': 0}) for i in range(5)])
    assert five['primary']['V-T']['p_sign'] == 0.0625 and five['verdict']['id'] == 3
    six = run_verdict([(i, {'V': 3, 'T': 0, 'S': 0}) for i in range(6)])
    assert six['primary']['V-T']['p_sign'] == 0.03125 and six['primary']['V-T']['ci95_pp'][0] > 0
    assert six['verdict']['id'] == 1 and six['verdict']['sentence'].startswith(an.VERDICT[1]) and '6 : 0' in six['verdict']['sentence']
    # V above T but S as high as V: verdict 2
    two = run_verdict([(i, {'V': 3, 'T': 0, 'S': 3}) for i in range(6)])
    assert two['verdict']['id'] == 2 and two['verdict']['sentence'].startswith(an.VERDICT[2])
    assert two['primary']['V-S']['up'] == 0 and two['primary']['V-S']['down'] == 0
    # a significant change that goes the wrong way is verdict 3
    down = run_verdict([(i, {'V': 0, 'T': 3, 'S': 3}) for i in range(8)])
    assert down['verdict']['id'] == 3 and '0 : 8' in down['verdict']['sentence']
    # many changes both ways: a small gain is not enough when the sign test fails
    mixed = run_verdict([(i, {'V': 2, 'T': 1}) for i in range(10)] + [(i, {'V': 1, 'T': 2}) for i in range(10, 18)])
    assert mixed['primary']['V-T']['up'] == 10 and mixed['primary']['V-T']['down'] == 8 and mixed['verdict']['id'] == 3


def test_verdict_function_directly_and_the_scale_sentences():
    ok = {'n': 334, 'mean_pp': 2.0, 'ci95_pp': [0.5, 3.0], 'up': 8, 'down': 0, 'same': 326, 'p_sign': 0.0078}
    no_sign = {**ok, 'up': 4, 'p_sign': 0.125}
    no_ci = {**ok, 'ci95_pp': [-0.1, 3.0]}
    assert an.verdict(ok, ok)['id'] == 1 and an.verdict(ok, no_sign)['id'] == 2 and an.verdict(ok, no_ci)['id'] == 2
    assert an.verdict(no_sign, ok)['id'] == 3 and an.verdict(no_ci, ok)['id'] == 3
    v = an.verdict({**no_sign, 'ci95_pp': [0.3, 3.4]}, no_sign)
    assert v['id'] == 3 and an.SCALE_EVEN in v['sentence'] and an.SCALE_CONC not in v['sentence']        # upper end < 3.5
    v = an.verdict({**no_sign, 'ci95_pp': [0.3, 9.0]}, no_sign)
    assert an.SCALE_CONC in v['sentence'] and an.SCALE_EVEN not in v['sentence']                          # 3.5 <= upper < 10.5
    v = an.verdict({**no_sign, 'ci95_pp': [0.3, 12.0]}, no_sign)
    assert an.SCALE_CONC not in v['sentence'] and an.SCALE_EVEN not in v['sentence']
    v = an.verdict({**no_sign, 'ci95_pp': [0.3, 2.0]}, no_sign, n_questions=112)
    assert an.SCALE_EVEN not in v['sentence'] and v['flags']
    v = an.verdict(ok, ok, valid_rates={'S': 1.0, 'T': 0.9, 'V': 0.949, 'Vminus': 0.95})
    assert v['conditions_below_95pct_valid'] == ['T', 'V'] and 'T 90.0%' in v['sentence'] and 'V 94.9%' in v['sentence']
    assert 'V− ' not in v['sentence']
    assert an.verdict(ok, ok, interim=True)['sentence'].endswith(an.INTERIM_SENTENCE)
    assert an.VERDICT[3] == '이 운영점(4단계까지)에서도 VisRef의 이득을 확인하지 못했다.'


def test_validity_flag_and_no_change_scale_sentence():
    out = run_verdict([], valid=False)
    assert out['verdict']['id'] == 3 and set(out['verdict']['conditions_below_95pct_valid']) == set(cm.CONDS)
    assert 'V− 0.0%' in out['verdict']['sentence']
    assert out['validity']['V']['valid_rate'] == 0.0
    ok = run_verdict([])
    assert ok['primary']['V-T']['ci95_pp'] == [0.0, 0.0] and an.SCALE_EVEN in ok['verdict']['sentence']  # upper end 0 < 3.5
    assert ok['verdict']['conditions_below_95pct_valid'] == [] and 'V− ' not in ok['verdict']['sentence']


def test_interim_stop_rule_in_the_analysis():
    rows, ids = synth(n=112, changes=[(0, {'V': 3, 'T': 0})])
    out = an.analyze(rows, {'main': ids['part1']}, interim=True)
    assert out['verdict']['id'] == 3 and out['verdict']['sentence'].endswith('중간 멈춤 규칙으로 112문항에서 멈춤.')
    rows, ids = synth(n=112, changes=[(0, {'V': 3, 'T': 0}), (1, {'V': 0, 'T': 3})])
    with pytest.raises(SystemExit):
        an.analyze(rows, {'main': ids['part1']}, interim=True)


def test_selection_missing_error_trial_and_extra_rows():
    rows, ids = synth(n=6)
    rows['1']['set'] = 'trial'                     # trial pids count in the main analysis
    del rows['2']
    rows['3'] = {'pid': '3', 'status': 'error', 'error': 'boom', 'code_version': 'v', 'set': 'main'}
    del rows['4']['answers']['V'][3]               # an incomplete row
    rows['x'] = make_row('x', {c: 1 for c in cm.CONDS})
    rows['5']['code_version'] = 'w'
    out = an.analyze(rows, ids)
    c = out['counts']
    assert out['n_analysed'] == 3 and c['missing'] == ['2'] and c['error_pids'] == ['3'] and c['rows_lacking_answers'] == ['4']
    assert c['rows_outside_main_list'] == ['x'] and c['n_trial_rows_in_main'] == 1 and c['code_versions'] == ['v', 'w']
    assert any('incomplete' in f for f in out['flags']) and any('code_versions' in f for f in out['flags'])


def test_descriptive_outputs_on_a_synthetic_run(tmp_path):
    rows, ids = synth(n=20, qtype='multi_choice', ids_rep=False)
    for i in range(10):
        rows[str(i)]['question_type'] = 'free_form'
    rows['0']['recon']['tokens_match'] = False
    rows['1']['probes']['T'][0]['H'] = None                                   # an invalid probe never stops
    rows['2']['steps']['V'][1]['ids'] = rows['2']['steps']['V'][0]['ids']      # a repeated step
    rows['3']['probes']['T'][0]['H'] = 5.0
    rows['3']['probes']['T'][1]['H'] = 0.2
    rows['3']['probes']['T'][1]['correct'] = 0
    merged = tmp_path / 'merged.jsonl'
    merged.write_text('\n'.join(json.dumps({'pid': str(i), 'method': m, 'correct': 0 if i < 4 else int(m == 'st')})
                                for i in range(20) for m in ('st', 'tsr', 'visref_raw_rank')) + '\n')
    out = an.analyze(rows, ids, str(merged))
    d = out['descriptive']
    assert d['all_wrong_in_main_run']['n_all_wrong_in_analysed'] == 4 and d['all_wrong_in_main_run']['all_wrong']['n'] == 4
    assert d['all_wrong_in_main_run']['rest']['n'] == 16 and d['all_wrong_in_main_run']['merged_file'] == 'merged.jsonl'
    assert set(d['question_type']) == {'free_form', 'multi_choice'} and d['question_type']['free_form']['n'] == 10
    assert d['recon_mismatch_excluded']['n_excluded'] == 1 and d['recon_mismatch_excluded']['n'] == 19
    assert d['probes']['curve']['T'][0]['acc'] == 1.0 and len(d['probes']['curve']['V']) == 4 and d['probes']['step1_probe_text_equal_T_V'] == 20
    assert d['repetition']['V']['per_step'][0]['n_same_as_previous'] == 1 and d['repetition']['T']['n_questions_with_a_repeated_step'] == 0
    assert d['steps']['T'][0]['mean_len'] == 100.0 and d['steps']['T'][1]['end'] == {'close': 0, 'eos': 0, 'cap': 20}
    assert d['steps']['T'][1]['n_cap_hits'] == 20
    sel = d['v_selection']
    assert sel[0]['n_blocks'] == 20 and sel[0]['median_m'] == 10 and sel[1]['median_n_sel'] == 5 and sel[0]['n_sel_equals_m'] == 0
    sim = d['delta_0.25_post_hoc']['T']
    # H = 0.1 < 0.25 stops at step 1 for most; the invalid probe (H None) and H=5.0 go on to step 2 (H 0.1 stops; q3: H 0.2 stops)
    assert sim['stop_step_counts'] == {'1': 18, '2': 2, '3': 0, '4': 0} and sim['n_continue_after_step1'] == 2
    assert 0 < sim['acc_at_stop'] < 1 and d['greedy']['V-T']['n'] == 20 and out['accuracy_pp']['S'] == pytest.approx(33.3333, abs=1e-3)
    json.dumps(out)
    # no merged file: reported, not a crash
    assert an.analyze(rows, ids, str(tmp_path / 'nope.jsonl'))['descriptive']['all_wrong_in_main_run'] == {'error': 'merged file not found'}


def test_analysis_cli_writes_json_with_relative_paths_and_refuses_overwrite(tmp_path, capsys):
    rows, ids = synth(n=12, changes=[(i, {'V': 3, 'T': 0, 'S': 0}) for i in range(7)])
    res, idf, out = tmp_path / 'r.jsonl', tmp_path / 'ids.json', tmp_path / 'o.json'
    res.write_text(''.join(json.dumps(rows[p]) + '\n' for p in ids['main']))
    idf.write_text(json.dumps(ids))
    assert an.main([str(res), '--ids', str(idf), '--out', str(out), '--merged', str(tmp_path / 'none.jsonl')]) == 0
    doc = json.loads(out.read_text())
    assert doc['n_analysed'] == 12 and doc['files'] == ['r.jsonl'] and doc['ids_file'] == 'ids.json'
    assert not any(str(p).startswith('/') for p in doc['files'] + [doc['ids_file']])
    assert doc['verdict']['id'] == 1 and capsys.readouterr().out.strip() == doc['verdict']['sentence']
    with pytest.raises(SystemExit):
        an.main([str(res), '--ids', str(idf), '--out', str(out)])
    assert an.main([str(res), '--ids', str(idf), '--out', str(out), '--force', '--merged', 'nope']) == 0
    dup = tmp_path / 'dup.jsonl'
    dup.write_text(json.dumps(rows['0']) + '\n')
    with pytest.raises(SystemExit):
        an.load([str(res), str(dup)])


# ---- interim counter --------------------------------------------------------------------------------------------------------
def write_part1(tmp_path, changes, n=112):
    rows, ids = synth(n=n, changes=changes)
    f, idf = tmp_path / 'p1.jsonl', tmp_path / 'ids.json'
    f.write_text(''.join(json.dumps(rows[p]) + '\n' for p in ids['main']))
    idf.write_text(json.dumps(ids))
    return f, idf


def test_interim_counter_prints_only_the_count_and_the_decision(tmp_path, capsys):
    f, idf = write_part1(tmp_path, [(0, {'V': 3, 'T': 0}), (1, {'V': 0, 'T': 2}), (2, {'S': 3})])
    assert ci.main([str(f), '--ids', str(idf)]) == 0
    text = capsys.readouterr().out
    assert text.splitlines() == ['questions with V and T different: 2', 'decision: run part 2']
    f2, idf2 = write_part1(tmp_path, [(0, {'V': 0, 'T': 3}), (5, {'S': 3})])  # one changed question, the other direction: same kind of output
    assert ci.main([str(f2), '--ids', str(idf2)]) == 0
    assert capsys.readouterr().out.splitlines() == ['questions with V and T different: 1', 'decision: stop']
    f3, idf3 = write_part1(tmp_path, [])
    assert ci.main([str(f3), '--ids', str(idf3)]) == 0
    assert capsys.readouterr().out.splitlines() == ['questions with V and T different: 0', 'decision: stop']
    assert ci.decision(1) == 'stop' and ci.decision(2) == 'run part 2'
    # only part 1 counts; rows of part 2 are ignored
    rows, ids = synth(n=130, changes=[(120, {'V': 3, 'T': 0})])
    f4, idf4 = tmp_path / 'all.jsonl', tmp_path / 'ids4.json'
    f4.write_text(''.join(json.dumps(rows[p]) + '\n' for p in ids['main']))
    idf4.write_text(json.dumps(ids))
    assert ci.main([str(f4), '--ids', str(idf4)]) == 0
    assert capsys.readouterr().out.splitlines()[0] == 'questions with V and T different: 0'


def test_interim_counter_refuses_incomplete_part_1_and_duplicates(tmp_path, capsys):
    f, idf = write_part1(tmp_path, [], n=112)
    lines = f.read_text().splitlines()
    f.write_text('\n'.join(lines[:100]) + '\n')
    assert ci.main([str(f), '--ids', str(idf)]) == 2
    out = capsys.readouterr().out
    assert out.strip() == 'part 1 incomplete: 12 of 112 questions have no usable row' and 'decision' not in out
    f.write_text('\n'.join(lines + lines[:1]) + '\n')
    with pytest.raises(SystemExit):
        ci.main([str(f), '--ids', str(idf)])


# ---- the Kaggle builder -----------------------------------------------------------------------------------------------------
def test_builder_makes_a_folder_and_refuses_bad_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(bc, 'HERE', tmp_path)
    assert bc.main(['m1', '--owner', 'me']) == 0
    run = tmp_path / 'runs' / 'm1'
    rc = json.loads((run / 'run_config.json').read_text())
    assert rc['set'] == 'main' and rc['timeout'] == 21600 and rc['time_limit'] == 21600 - 900 and rc['group'] == 4
    assert rc['kernel'] == 'me/visref-m1' and rc['run_args'] == ['--set', 'main', '--group', '4', '--pids-part', 'all']
    for f in ('diag/closed_multistep.py', 'diag/step1_think.py', 'diag/closed_multistep_ids.json', 'diag/step1_store_bmain.json',
              'run.py', 'visref/engine.py', 'visref/model.py', 'visref/methods.py'):
        assert f in rc['source_sha256']
    nb = json.loads((run / 'notebook.ipynb').read_text())
    src = ''.join(nb['cells'][0]['source'])
    ast.parse(src)
    assert "'diag/closed_multistep.py'" in src and "'--ids', 'diag/closed_multistep_ids.json'" in src
    assert "f'closed_multistep_{i}.jsonl'" in src and "f'log_{i}.txt'" in src and "f'{i}/{n}'" in src
    cfg = json.loads(base64.b64decode(re.search(r"CFG = json.loads\(base64.b64decode\('([^']+)'\)", src).group(1)))
    assert cfg['run_args'] == rc['run_args'] and cfg['time_limit'] == 20700
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(re.search(r"b64decode\('([^']+)'\)\)\)\.extractall", src).group(1))))
    assert (REPRO / 'diag/closed_multistep.py').read_bytes() == z.read('diag/closed_multistep.py')
    assert {'diag/step1_think.py', 'diag/closed_multistep_ids.json', 'run.py'} <= set(z.namelist())
    assert 'diag/inject_sensitivity.py' not in z.namelist() and 'diag/thumb_gate.py' not in z.namelist()
    meta = json.loads((run / 'kernel-metadata.json').read_text())
    assert meta['id'] == 'me/visref-m1' and meta['enable_gpu'] is True and meta['is_private'] is True
    with pytest.raises(SystemExit):
        bc.main(['m1', '--owner', 'me'])  # an existing run record is never overwritten
    with pytest.raises(SystemExit):
        bc.main(['m2', '--owner', 'me', '--timeout', '21599'])  # main needs at least 6 hours
    with pytest.raises(SystemExit):
        bc.main(['m2', '--owner', 'me', '--timeout', '14400', '--pids-part', '1'])
    assert bc.main(['m3', '--owner', 'me', '--pids-part', '1', '--limit', '8', '--extra-pids', '5']) == 0
    ra = json.loads((tmp_path / 'runs/m3/run_config.json').read_text())['run_args']
    assert ra == ['--set', 'main', '--group', '4', '--pids-part', '1', '--limit', '8', '--extra-pids', '5']
    assert bc.main(['t1', '--owner', 'me', '--set', 'trial', '--timeout', '3600']) == 0   # the trial may be short
    t = json.loads((tmp_path / 'runs/t1/run_config.json').read_text())
    assert t['run_args'] == ['--set', 'trial', '--group', '4'] and t['time_limit'] == 2700
    with pytest.raises(SystemExit):
        bc.main(['t2', '--owner', 'me', '--set', 'trial', '--timeout', '3600', '--pids-part', '2'])
    sk = tmp_path / 'done.jsonl'
    sk.write_text(json.dumps({'pid': '12'}) + '\n' + json.dumps({'pid': '3'}) + '\n')
    assert bc.main(['m4', '--owner', 'me', '--skip-from', str(sk)]) == 0
    assert json.loads((tmp_path / 'runs/m4/run_config.json').read_text())['n_skip'] == 2
    assert not (REPRO / 'kaggle/runs/m1').exists()  # the repo's own runs folder is untouched


def test_no_absolute_sandbox_paths_in_the_new_files():
    bad = ('/ho' + 'me/', '/t' + 'mp/', 'scratch' + 'pad')
    files = ['diag/closed_multistep_ids.py', 'diag/closed_multistep.py', 'diag/analyze_closed_multistep.py',
             'diag/count_closed_multistep_interim.py', 'kaggle/build_closed_multistep.py', 'tests/test_closed_multistep.py',
             'diag/closed_multistep_ids.json']
    for f in files:
        text = (REPRO / f).read_text()
        assert not any(b in text for b in bad), f
