"""CPU tests for diag/full_think_ids.py, diag/full_think.py, diag/analyze_full_think.py and kaggle/build_full_think.py
(fake engine / tiny LM)."""
import ast
import base64
import io
import json
import random
import re
import string
import sys
import types
import zipfile
from pathlib import Path

import pytest
import torch

REPRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO / 'diag'))
sys.path.insert(0, str(REPRO / 'kaggle'))
import analyze_full_think as an  # noqa: E402
import build_full_think as fb  # noqa: E402
import full_think as ft  # noqa: E402
import full_think_ids as ids_mod  # noqa: E402
import step1_branch as sb  # noqa: E402
import step1_think as t1  # noqa: E402
from test_repro import lm  # noqa: E402,F401  (pytest fixture)
from test_step1_think import NL, SP, TOK, FakeEng, item, tiny_items  # noqa: E402

from run import pack  # noqa: E402
from visref.engine import Engine, Request, Result  # noqa: E402

TE, EOS = SP['think_end'], SP['eos']
ROW = {'question_type': 'free_form', 'answer_type': 'integer', 'answer': '7', 'precision': None, 'choices': None}
CFG = {**ft.DEFAULTS, 'think_extra': 8, 'cut': 3, 'probe_cap': 4, 'n_t1': 2, 'chunk': 8, 'batch_tokens': 4000}


# ---- ids list ---------------------------------------------------------------------------------------------------
def test_ids_file_equals_a_fresh_build():
    assert ids_mod.main(['--check']) == 0
    assert ids_mod.OUT.read_text() == ids_mod.text(ids_mod.fresh())
    assert ids_mod.text(ids_mod.fresh()) == ids_mod.text(ids_mod.fresh())  # deterministic


def test_ids_sets_on_the_real_data():
    d = json.loads(ids_mod.OUT.read_text())
    store = json.loads(ids_mod.STORE.read_text())['pids']
    order = [str(p) for p in json.loads(ids_mod.ORDER.read_text())]
    S = d['sets']
    assert d['set_sizes'] == {'cap_all': 641, 'cap_continued': 297, 'cap_stopped': 344, 'pilot_long': 2}
    assert d['ids'] == S['cap_all'] and d['n'] == 641
    assert 297 + 344 == 641 and not set(S['cap_continued']) & set(S['cap_stopped'])
    assert set(S['cap_continued']) | set(S['cap_stopped']) == set(S['cap_all'])
    assert set(S['cap_all']) == {p for p in order if store[p]['think_end'] == 'cap'}
    pos = {p: i for i, p in enumerate(order)}
    for s in ('cap_all', 'cap_continued', 'cap_stopped'):  # seed-0 order
        assert [pos[p] for p in S[s]] == sorted(pos[p] for p in S[s])
    branch = set(json.loads(ids_mod.BRANCH.read_text())['ids'])
    think = set(json.loads(ids_mod.THINK.read_text())['ids'])
    assert set(S['cap_continued']) == set(S['cap_all']) & branch and set(S['cap_stopped']) == set(S['cap_all']) & think
    assert all(d['origin'][p] == 'continued' for p in S['cap_continued'])
    assert all(d['origin'][p] == 'stopped' for p in S['cap_stopped'])
    for p in S['cap_all']:
        assert d['est_ctx_F'][p] == d['prompt_len'][p] + d['think_tokens'][p] + d['n_visual'][p]
        assert d['think_tokens'][p] == 1024 and d['stratum'][p] == 'cap'
    top = sorted(S['cap_all'], key=lambda p: (-d['est_ctx_F'][p], pos[p]))[:2]
    assert S['pilot_long'] == top and d['est_ctx_F'][top[0]] >= d['est_ctx_F'][top[1]]
    assert set(d['source_sha256']) == {'order', 'store', 'step1_branch_ids', 'step1_think_ids', 'merged'}
    assert 'seed-0' in d['rule'] and 'donor' in d['rule']


def test_ids_donors_are_cyclic_same_size_and_within_the_set():
    d = json.loads(ids_mod.OUT.read_text())
    nv = d['n_visual']
    for s in ('cap_all', 'cap_continued', 'cap_stopped'):
        order, don = d['sets'][s], d['donor'][s]
        assert list(don) == order
        for i, p in enumerate(order):
            q = don[p]
            assert q in order and q != p and nv[q] >= nv[p]
            nxt = [order[(i + k) % len(order)] for k in range(1, len(order))]
            same = [x for x in nxt if nv[x] == nv[p]]
            assert q == (same or [x for x in nxt if nv[x] > nv[p]])[0]  # next in the set's order, cyclic
    assert d['donor']['pilot_long'] == {p: d['donor']['cap_all'][p] for p in d['sets']['pilot_long']}


def test_ids_build_checks_and_synthetic_rule():
    order = ['a', 'b', 'c', 'd', 'e']
    store = {p: {'think_end': 'cap' if p != 'e' else 'close', 'think_tokens': 1024} for p in order}
    sizes = {p: (n, 100) for p, n in zip(order, (2, 3, 2, 3, 2))}
    doc = ids_mod.build(order, store, ['b', 'd', 'x'], ['a', 'c'], sizes, expect=None)
    assert doc['sets']['cap_all'] == ['a', 'b', 'c', 'd'] and doc['sets']['cap_continued'] == ['b', 'd']
    assert doc['sets']['cap_stopped'] == ['a', 'c']
    assert doc['donor']['cap_all'] == {'a': 'c', 'b': 'd', 'c': 'a', 'd': 'b'}
    assert doc['donor']['cap_continued'] == {'b': 'd', 'd': 'b'} and doc['donor']['cap_stopped'] == {'a': 'c', 'c': 'a'}
    assert doc['est_ctx_F']['a'] == 100 + 1024 + 2 and doc['sets']['pilot_long'] == ['b', 'd']  # est 1127 vs 1126
    assert doc['origin'] == {'a': 'stopped', 'b': 'continued', 'c': 'stopped', 'd': 'continued'}
    with pytest.raises(AssertionError):
        ids_mod.build(order, store, ['a', 'b'], ['a', 'c', 'd'], sizes, expect=None)  # overlap
    with pytest.raises(AssertionError):
        ids_mod.build(order, store, ['b'], ['a', 'c'], sizes, expect=None)  # d in neither set
    with pytest.raises(AssertionError):
        ids_mod.build(order, store, ['b', 'd'], ['a', 'c'], sizes)  # not 641 / 297 / 344
    with pytest.raises(AssertionError):
        ids_mod.build(order, store, ['b', 'd'], ['a', 'c'], {p: v for p, v in sizes.items() if p != 'a'}, expect=None)
    with pytest.raises(AssertionError):  # a set of one question has no donor
        ids_mod.build(order, store, ['b'], ['a', 'c', 'd'], sizes, expect=None)


# ---- contexts and their order -----------------------------------------------------------------------------------
def a0_item(eng, n_vis=10, n_other=14, **kw):
    it = item(eng, n_vis=n_vis, **kw)
    it.update(sample={'think_end': 'cap'}, recon={'tokens_match': True})
    it['vis_other'] = torch.randn(n_other, eng.emb.embedding_dim, generator=torch.Generator().manual_seed(9))
    return it


def test_contexts_a0_c_f_l_and_their_order():
    eng = FakeEng()
    it = a0_item(eng)
    ft.build_contexts(it, eng, TOK, SP, CFG, ft.ALL_CONDS)
    e, n = eng.embed, it['vis'].shape[0]
    prefix = torch.cat([it['base'], e(it['z'] + it['sep'])])
    wait = e(SP['wait'])
    assert list(it['ctx']) == ['A0', 'C', 'F', 'L']
    assert torch.equal(it['ctx']['A0'], torch.cat([prefix, e([TE])]))
    assert torch.equal(it['ctx']['C'], torch.cat([prefix, wait]))
    assert torch.equal(it['ctx']['F'], torch.cat([prefix, it['vis'], wait]))
    assert torch.equal(it['ctx']['L'], torch.cat([prefix, it['vis_other'][:n], wait]))  # cropped to the first n tokens
    P = prefix.shape[0]
    assert it['ctx']['A0'].shape[0] == P + 1 and it['ctx']['C'].shape[0] == P + wait.shape[0]
    assert it['ctx']['F'].shape[0] == it['ctx']['L'].shape[0] == P + n + wait.shape[0]
    assert torch.equal(it['ctx']['F'][P:P + n], it['vis'])  # every image token, original order, before the cue
    assert eng.calls == []  # no hidden states: untagged 'bmain' injection


def test_f_is_inject_bmain_with_all_indices_and_has_no_tags():
    eng = FakeEng()
    it = a0_item(eng)
    ft.build_contexts(it, eng, TOK, SP, CFG, ('F',))
    inj = t1.inject_embeds(eng, TOK, SP, 'bmain', it['vis'], list(range(it['vis'].shape[0])), TOK.decode(it['z']), it['sep_text'])
    prefix = torch.cat([it['base'], eng.embed(it['z'] + it['sep'])])
    assert torch.equal(it['ctx']['F'], torch.cat([prefix, inj, eng.embed(SP['wait'])]))
    for tag in ('img_start', 'img_end'):
        assert not (it['ctx']['F'] == eng.embed([SP[tag]])).all(-1).any()


def test_l_uses_the_donor_tokens_same_count_and_needs_them():
    eng = FakeEng()
    it = a0_item(eng, n_other=10)  # same size: the donor's tokens as they are
    ft.build_contexts(it, eng, TOK, SP, CFG, ('L',))
    assert torch.equal(it['ctx']['L'][-(it['vis'].shape[0] + len(SP['wait'])):-len(SP['wait'])], it['vis_other'])
    it['vis_other'] = it['vis_other'][:5]  # a donor with fewer tokens is a fault, never padded or repeated
    with pytest.raises(AssertionError):
        ft.build_contexts(it, eng, TOK, SP, CFG, ('L',))
    del it['vis_other']
    with pytest.raises(ValueError):
        ft.build_contexts(it, eng, TOK, SP, CFG, ('L',))
    ft.build_contexts(it, eng, TOK, SP, CFG, ('A0', 'C', 'F'))  # without L no donor is needed
    assert list(it['ctx']) == ['A0', 'C', 'F']


def test_a0_is_step1_branch_condition_a_plus_the_probe():
    """A0 = the context step1_branch gives its condition A, probe included."""
    eng = FakeEng()
    it = a0_item(eng)
    seen = []

    def spy(reqs):
        seen.append(reqs)
        if len(seen) == 1:
            return [Result(ids=TOK.encode('ok') + [TE], finish='stop') for _ in reqs]
        return [Result(ids=TOK.encode('{7}'), finish='stop') for _ in reqs]

    it1 = {**it, 'row': ROW, 'pid': '5'}
    sb.run_group([it1], eng, TOK, SP, {**sb.DEFAULTS, 'think_extra': 8, 'n_t1': 2, 'probe_cap': 4}, spy)
    a_reqs = seen[1][:3]  # step1_branch answers: A first, n_t1 + 1 requests each
    it2 = {**it, 'row': ROW, 'pid': '5'}
    seen.clear()
    ft.run_group([it2], eng, TOK, SP, CFG, spy, ('A0',))
    mine = seen[0]  # no free thinking in A0: the only generate call holds the answers
    assert len(mine) == 3 and len(a_reqs) == 3 and all(torch.equal(x.ctx, y.ctx) for x, y in zip(mine, a_reqs))
    key = lambda r: (r.max_new, r.stop_ids, r.sampling, None if r.gen is None else r.gen.initial_seed())  # noqa: E731
    assert [key(r) for r in mine] == [key(r) for r in a_reqs]


# ---- seeds ---------------------------------------------------------------------------------------------------------
def scripted(think_ids):
    """A generate spy: the first call (free thinking) returns think_ids per request, the next calls answers."""
    seen = []

    def spy(reqs):
        seen.append(reqs)
        if len(seen) == 1:
            return [Result(ids=list(think_ids), finish='stop' if think_ids[-1] in (TE, EOS) else 'cap') for _ in reqs]
        return [Result(ids=TOK.encode('{7}'), finish='stop') for _ in reqs]
    spy.seen = seen
    return spy


def test_seeds_are_shared_across_conditions_incl_a0_and_cut():
    eng = FakeEng()
    items = [{**a0_item(eng, pid=p), 'row': ROW} for p in ('3', '9')]
    spy = scripted(TOK.encode('abcdef') + [TE])  # 6 thinking tokens > cut 3 -> every free condition has a cut context
    ft.run_group(items, eng, TOK, SP, CFG, spy, ft.ALL_CONDS)
    think, ans = spy.seen
    assert len(think) == 2 * 3
    n1 = CFG['n_t1'] + 1
    per_item = (4 + 3) * n1  # A0, C, F, L and C_cut, F_cut, L_cut
    assert len(ans) == 2 * per_item
    for k, it in enumerate(items):
        gs = [r.gen.initial_seed() for r in think[3 * k:3 * k + 3]]
        assert gs == [ft.seed_of(42, it['pid'], 'think')] * 3
        for ci in range(7):
            chunk = ans[k * per_item + ci * n1:k * per_item + (ci + 1) * n1]
            seeds = [r.gen.initial_seed() if r.gen is not None else None for r in chunk]
            assert seeds == [ft.seed_of(42, it['pid'], 'ans', j) for j in range(CFG['n_t1'])] + [None]
            assert chunk[-1].sampling == t1.GREEDY and chunk[0].sampling == t1.T1 and chunk[0].max_new == CFG['probe_cap']
        assert set(it['answers']) == {'A0', 'C', 'F', 'L'} and set(it['extra']) == {'C', 'F', 'L'}
        assert set(it['cut']) == {'C', 'F', 'L'} and all(len(v['answers']) == n1 for v in it['cut'].values())
    assert ft.seed_of is t1.seed_of


# ---- cut-512 logic (a small cut) ------------------------------------------------------------------------------------
def answer_ctxs(spy, n_cond_full, n_cut, n1):
    """Answer-request contexts of one item: (full per condition, cut per condition) in order."""
    reqs = spy.seen[1]
    firsts = [reqs[i * n1].ctx for i in range(n_cond_full + n_cut)]
    return firsts[:n_cond_full], firsts[n_cond_full:]


@pytest.mark.parametrize('text, cut, expect_same', [('abcdef', 3, False), ('ab', 3, True), ('abc', 3, True), ('abcd', 3, False),
                                                    ('ab\ncdef', 3, False)])
def test_cut_logic_with_a_small_cut(text, cut, expect_same):
    eng = FakeEng()
    it = {**a0_item(eng), 'row': ROW}
    cfg = {**CFG, 'cut': cut}
    ids = TOK.encode(text) + [TE]
    spy = scripted(ids)
    ft.run_group([it], eng, TOK, SP, cfg, spy, ('C', 'F'))
    n1 = cfg['n_t1'] + 1
    for c in 'CF':
        rec = it['cut'][c]
        assert rec['same_as_full'] is expect_same and rec['len_full'] == len(text) and rec['cut'] == cut
    assert it['extra']['C']['len'] == len(text) and it['extra']['C']['end'] == 'close'
    full, cuts = answer_ctxs(spy, 2, 0 if expect_same else 2, n1)
    assert len(spy.seen[1]) == (2 + (0 if expect_same else 2)) * n1  # no new answers when the cut equals the full result
    probe = eng.embed(SP['probe_prompt'])
    for k, c in enumerate('CF'):
        pre = it['pre'][c]
        assert torch.equal(full[k], torch.cat([pre, eng.embed(ids), probe]))  # the thinking ended by itself: </think> as is
        if not expect_same:
            first = TOK.encode(text)[:cut]
            sep = [] if TOK.decode(first).endswith('\n') else NL
            assert it['cut'][c]['ids'] == first and it['cut'][c]['text'] == text[:cut] and it['cut'][c]['end'] == 'cap'
            assert torch.equal(cuts[k], torch.cat([pre, eng.embed(first + sep + [TE]), probe]))  # close_thinking(ids, 'cap')
    row = ft.row_out(it, cfg, 'v', ('C', 'F'), 'cap_all')
    json.dumps(row)
    for c in 'CF':
        rc = row['cut512'][c]
        assert rc['same_as_full'] is expect_same
        if expect_same:
            assert 'answers' not in rc and rc['acc_t1'] == row['acc_t1'][c] and rc['greedy'] == row['greedy'][c]
        else:
            assert len(rc['answers']) == n1 and 'acc_t1' in rc and 'greedy' in rc


def test_cut_of_a_capped_thinking_and_cut_not_below_think_extra():
    eng = FakeEng()
    cap_ids = TOK.encode('abcdefgh')  # 8 = think_extra tokens, no </think>: ended at the cap
    for cut, same in ((5, False), (8, True), (50, True)):
        it = {**a0_item(eng), 'row': ROW}
        spy = scripted(cap_ids)
        ft.run_group([it], eng, TOK, SP, {**CFG, 'cut': cut}, spy, ('C',))
        assert it['extra']['C']['end'] == 'cap' and it['extra']['C']['len'] == 8
        assert it['cut']['C']['same_as_full'] is same
        full, cuts = answer_ctxs(spy, 1, 0 if same else 1, 3)
        assert torch.equal(full[0], torch.cat([it['pre']['C'], eng.embed(cap_ids + NL + [TE]), eng.embed(SP['probe_prompt'])]))
        if not same:
            assert torch.equal(cuts[0], torch.cat([it['pre']['C'], eng.embed(cap_ids[:5] + NL + [TE]), eng.embed(SP['probe_prompt'])]))


# ---- chunk fix ----------------------------------------------------------------------------------------------------------
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


def test_old_make_generate_silently_stops_at_chunk_and_the_new_one_does_not():
    eng = GenEng()
    it = {**a0_item(eng), 'row': ROW}
    old_cfg = {**CFG, 'think_extra': 40, 'chunk': 16}  # the old default shape: chunk below the thinking length
    old = t1.make_generate(eng, old_cfg, pack)([Request(it['base'], frozenset({TE, EOS}), 40, None)])[0]
    assert old.finish == 'chunk' and len(old.ids) == 16  # the bug: unfinished ...
    assert t1.close_thinking(TOK, SP, old.ids, old.finish, '\n')[1] == 'cap'  # ... and read as a finished cap
    with pytest.raises(ft.ChunkError):  # new: never read as cap
        ft.make_generate(eng, old_cfg, pack)([Request(it['base'], frozenset({TE, EOS}), 40, None)])
    with pytest.raises(ft.ChunkError):
        ft.run_group([it], eng, TOK, SP, old_cfg, ft.make_generate(eng, old_cfg, pack), ('C',))
    cfg = ft.resolve_cfg({**CFG, 'think_extra': 40, 'chunk': 0, 'cut': 20})
    assert cfg['chunk'] == 40
    ft.run_group([it], eng, TOK, SP, cfg, ft.make_generate(eng, cfg, pack), ('A0', 'C', 'F'))
    for c in 'CF':
        assert it['extra'][c]['len'] == 40 and it['extra'][c]['end'] == 'cap'  # reaches think_extra, ended at the cap
        assert it['cut'][c]['same_as_full'] is False and len(it['cut'][c]['ids']) == 20


def test_real_engine_chunk_semantics_and_think_extra_above_the_old_chunk(lm):
    eng = Engine(lm, 0.6, 50, 0.95)
    base = tiny_items(lm)[0]['base']
    mk = lambda: Request(base, frozenset(), 40, torch.Generator().manual_seed(1))  # noqa: E731
    short, full = eng.generate([mk()], 16)[0], eng.generate([mk()], 40)[0]
    assert (short.finish, len(short.ids)) == ('chunk', 16) and (full.finish, len(full.ids)) == ('cap', 40)
    assert full.ids[:16] == short.ids
    with pytest.raises(ft.ChunkError):
        ft.make_generate(eng, {**CFG, 'chunk': 16, 'batch_tokens': 4000}, pack)([mk()])
    assert ft.make_generate(eng, {**CFG, 'chunk': 40, 'batch_tokens': 4000}, pack)([mk()])[0].finish == 'cap'


def test_chunk_below_think_extra_is_refused_and_the_default_follows_think_extra():
    assert ft.DEFAULTS['think_extra'] == 2048 and ft.DEFAULTS['cut'] == 512 and ft.DEFAULTS['group'] == 4
    assert ft.DEFAULTS['n_t1'] == 3 and ft.DEFAULTS['probe_cap'] == 16 and ft.DEFAULTS['chunk'] == 0
    assert ft.resolve_cfg(ft.DEFAULTS)['chunk'] == 2048 == ft.resolve_cfg(ft.DEFAULTS)['think_extra']
    assert ft.resolve_cfg({**ft.DEFAULTS, 'think_extra': 4096})['chunk'] == 4096
    assert ft.resolve_cfg({**ft.DEFAULTS, 'chunk': 3000})['chunk'] == 3000
    with pytest.raises(SystemExit):
        ft.resolve_cfg({**ft.DEFAULTS, 'chunk': 1024})
    base = ['--parquet', 'p.parquet', '--out', 'o.jsonl']
    assert ft.parse_args(base).cfg['chunk'] == 2048 and ft.parse_args(base).conds == ('A0', 'C', 'F')
    assert ft.parse_args(base + ['--conds', 'A0,C,F,L']).conds == ft.ALL_CONDS
    with pytest.raises(SystemExit):
        ft.parse_args(base + ['--chunk', '1024'])  # 1024 < 2048
    with pytest.raises(SystemExit):
        ft.parse_args(base + ['--think-extra', '4096', '--chunk', '2048'])
    with pytest.raises(SystemExit):
        ft.parse_args(base + ['--conds', 'A0,X'])
    for bad in ('Fx', 'C,C', ''):
        with pytest.raises(SystemExit):
            ft.parse_args(base + ['--conds', bad])


def test_unfinished_rows_stop_the_run_even_in_the_error_handling():
    eng = GenEng()
    bad_cfg = {**CFG, 'think_extra': 40, 'chunk': 16}
    gen = ft.make_generate(eng, bad_cfg, pack)
    mk = lambda pid: a0_item(eng, pid=pid)  # noqa: E731
    with pytest.raises(ft.ChunkError):  # not turned into per-question error rows
        ft.process_group(['1', '2'], mk, eng, TOK, SP, bad_cfg, gen, ('C',), 'v', 'cap_all')


# ---- errors ------------------------------------------------------------------------------------------------------------
def test_error_rows_and_nonfinite_questions():
    eng = FakeEng()
    its = {p: {**a0_item(eng, pid=p), 'row': ROW} for p in ('1', '2', '3', '4')}

    def mk(pid):
        if pid == '2':
            raise KeyError('no image for 2')
        return its[pid]

    bad = {ft.seed_of(42, '3', 'ans', 0)}

    def spy(reqs):
        if any(r.gen is not None and r.gen.initial_seed() in bad for r in reqs):
            raise RuntimeError('boom')
        if reqs[0].max_new == CFG['think_extra']:
            return [Result(ids=TOK.encode('ok') + [TE], finish='stop') for _ in reqs]
        return [Result(ids=TOK.encode('{7}'), finish='stop') for _ in reqs]

    rows = ft.process_group(['1', '2', '3', '4'], mk, eng, TOK, SP, CFG, spy, ('C', 'F'), 'v', 'cap_all')
    assert [r['pid'] for r in rows] == ['1', '2', '3', '4']
    assert [r['status'] for r in rows] == ['ok', 'error', 'error', 'ok']
    assert rows[1]['stage'] == 'load' and 'KeyError' in rows[1]['error'] and rows[1]['set'] == 'cap_all'
    assert rows[2]['stage'] == 'generate' and 'boom' in rows[2]['error'] and rows[2]['code_version'] == 'v'
    assert set(rows[0]) >= {'pid', 'stratum', 'origin', 'set', 'code_version', 'scorer_version', 'cfg', 'n_visual', 'donor',
                            'z_ids', 'sep_ids', 'recon', 'extra', 'answers', 'acc_t1', 'greedy', 'cut512', 'answer_gt'}
    assert rows[0]['stratum'] == 'cap' and rows[0]['n_visual'] == 10 and rows[0]['cfg']['think_extra'] == CFG['think_extra']
    for k in ('zlib_ratio_tail3000', 'text', 'ids', 'end', 'len'):
        assert k in rows[0]['extra']['C']
    # non-finite logits in one condition: the question is an error row, the others are not affected
    def spy2(reqs):
        if reqs[0].max_new == CFG['think_extra']:
            return [None if i == 1 else Result(ids=TOK.encode('ok') + [TE], finish='stop') for i, _ in enumerate(reqs)]
        return [Result(ids=TOK.encode('{7}'), finish='stop') for _ in reqs]

    rows = ft.process_group(['1'], lambda p: its[p], eng, TOK, SP, CFG, spy2, ('C', 'F'), 'v', 'cap_all')
    assert rows[0]['status'] == 'error' and rows[0]['nonfinite'] == ['F'] and 'non-finite' in rows[0]['error']

    def spy3(reqs):  # non-finite only in an answer of C
        if reqs[0].max_new == CFG['think_extra']:
            return [Result(ids=TOK.encode('ok') + [TE], finish='stop') for _ in reqs]
        return [None if i == 1 else Result(ids=TOK.encode('{7}'), finish='stop') for i, _ in enumerate(reqs)]

    rows = ft.process_group(['1'], lambda p: its[p], eng, TOK, SP, CFG, spy3, ('C', 'F'), 'v', 'cap_all')
    assert rows[0]['status'] == 'error' and rows[0]['nonfinite'] == ['C']


# ---- tiny LM end to end ----------------------------------------------------------------------------------------------------
def tiny_group(lm, conds=ft.ALL_CONDS, cfg=None):
    cfg = cfg or CFG
    eng = Engine(lm, cfg['temperature'], cfg['top_k'], cfg['top_p'])
    its = tiny_items(lm)
    for k, it in enumerate(its):
        it['origin'] = 'continued' if k % 2 else 'stopped'
        it['vis_other'] = torch.randn(10, it['vis'].shape[1], generator=torch.Generator().manual_seed(50 + k))
        it['donor_pid'], it['donor_n_visual'] = f'd{k}', 10
    return ft.process_group([it['pid'] for it in its], {it['pid']: it for it in its}.__getitem__, eng, TOK, SP, cfg,
                            ft.make_generate(eng, cfg, pack), conds, 'v', 'cap_all')


def test_tiny_lm_rows_and_determinism(lm):
    a, b = tiny_group(lm), tiny_group(lm)
    assert a == b
    for r in a:
        assert r['status'] == 'ok'
        json.dumps(r)
        assert set(r['answers']) == {'A0', 'C', 'F', 'L'} == set(r['acc_t1']) == set(r['greedy'])
        assert all(len(v) == CFG['n_t1'] + 1 for v in r['answers'].values())
        assert set(r['extra']) == {'C', 'F', 'L'} and all(r['extra'][c]['len'] <= CFG['think_extra'] for c in 'CFL')
        assert r['donor'] and r['donor_n_visual'] == 10 and r['origin'] in ('continued', 'stopped')
        for c in 'CFL':
            rec = r['cut512'][c]
            assert rec['same_as_full'] == (r['extra'][c]['len'] <= CFG['cut'])
            assert ('ids' in rec) == (not rec['same_as_full'])


def test_tiny_lm_think_extra_beyond_the_old_chunk_reaches_think_extra(lm):
    """With chunk = think_extra, thinking that does not stop by itself reaches think_extra and ends at the cap."""
    cfg = {**CFG, 'think_extra': 60, 'cut': 20, 'chunk': 60}
    eng = Engine(lm, 0.6, 50, 0.95)
    its = tiny_items(lm)
    # the tiny random LM stops early now and then: forbid the stop tokens by routing them through a Request without stops
    orig = ft.Request
    ft.Request = lambda ctx, stop, mx, g, **kw: orig(ctx, stop if mx == cfg['probe_cap'] else frozenset(), mx, g, **kw)
    try:
        ft.run_group(its[:2], eng, TOK, SP, cfg, ft.make_generate(eng, cfg, pack), ('C', 'F'))
    finally:
        ft.Request = orig
    for it in its[:2]:
        for c in 'CF':
            assert len(it['extra'][c]['ids']) == 60 and it['extra'][c]['end'] == 'cap'  # (len excludes any </think> / EOS drawn)
            assert len(it['cut'][c]['ids']) == 20 and not it['cut'][c]['same_as_full']


# ---- main() end to end with a stubbed model ---------------------------------------------------------------------------------
def test_main_end_to_end_with_a_stub_model(lm, tmp_path, monkeypatch, capsys):
    import run as run_mod
    import visref.model as vm
    pids = ['1', '2', '3', '4', '5', '6']
    nvis = {'1': 6, '2': 6, '3': 4, '4': 4, '5': 6, '6': 6}
    emb = lm.get_input_embeddings()

    def encode_sample(model, tok, image, query, max_num):
        g = torch.Generator().manual_seed(int(image))
        return emb(torch.randint(5, 90, (8,), generator=g)).detach(), torch.randn(nvis[query], emb.embedding_dim, generator=g)

    rows = {p: {**ROW, 'pid': p, 'query': p, 'image': (lambda p=p: p)} for p in pids if p != '6'}  # '6' has no row: load error
    monkeypatch.setattr(vm, 'load_mathvista', lambda path: rows)
    monkeypatch.setattr(vm, 'load_model', lambda m, r: (types.SimpleNamespace(language_model=lm), TOK))
    monkeypatch.setattr(vm, 'encode_sample', encode_sample)
    monkeypatch.setattr(vm, 'special_ids', lambda tok, w, p: {**SP, 'wait': TOK.encode(w), 'probe_prompt': TOK.encode(p)})
    monkeypatch.setattr(run_mod, 'fast_kv_selfcheck', lambda lm_, tok: (True, {}))
    monkeypatch.setattr(ft, 'Engine', lambda lm_, t, k, p, fast_kv=True: Engine(lm_, t, k, p, fast_kv=False))
    store = {'pids': {p: {'thinking': 'think abc\n', 'think_tokens': '9', 'think_end': 'cap'} for p in pids}}
    doc = {'ids': pids, 'sets': {'cap_all': pids, 'cap_stopped': pids[:3], 'pilot_long': ['1']},
           'donor': {'cap_all': {'1': '2', '2': '1', '3': '4', '4': '3', '5': '1', '6': '1'}, 'cap_stopped': {'1': '2'}},
           'origin': {p: 'stopped' for p in pids}}
    (tmp_path / 'store.json').write_text(json.dumps(store))
    (tmp_path / 'ids.json').write_text(json.dumps(doc))
    out = tmp_path / 'res' / 'out.jsonl'
    args = ['--parquet', 'x', '--samples', str(tmp_path / 'store.json'), '--ids', str(tmp_path / 'ids.json'), '--out', str(out),
            '--think-extra', '6', '--cut', '3', '--probe-cap', '4', '--n-t1', '2', '--batch-tokens', '4000', '--group', '2',
            '--conds', 'A0,C,F,L']
    try:
        ft.main(args)
        res = [json.loads(l) for l in out.read_text().splitlines()]
        assert [r['pid'] for r in res] == pids and [r['status'] for r in res] == ['ok'] * 5 + ['error']
        assert res[5]['stage'] == 'load' and res[0]['donor'] == '2' and res[0]['cfg']['chunk'] == 6
        assert res[2]['donor'] == '4' and res[4]['donor'] == '1' and res[4]['donor_n_visual'] == 6
        assert all(r['code_version'] == ft.code_version() and r['set'] == 'cap_all' for r in res)
        ev = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith('{')]
        kinds = [e['event'] for e in ev]
        assert kinds[0] == 'start' and kinds[-1] == 'end' and kinds.count('group') == 3 and kinds.count('error') == 1
        g = [e for e in ev if e['event'] == 'group']
        assert all({'group_s', 'peak_alloc_gb', 'max_rss_gb', 'n_error', 'left'} <= set(e) for e in g) and g[-1]['left'] == 0
        assert {'peak_alloc_gb', 'elapsed'} <= set(ev[-1]) and ev[0]['conds'] == ['A0', 'C', 'F', 'L']
        ft.main(args)  # resume: everything is done (error rows included)
        assert len(out.read_text().splitlines()) == 6
        assert json.loads(capsys.readouterr().out.splitlines()[0])['todo'] == 0
        out2 = tmp_path / 'res' / 'out2.jsonl'  # set, extra pids, limit, shard, skip
        a2 = [x if x != str(out) else str(out2) for x in args]
        (tmp_path / 'skip.json').write_text(json.dumps(['2']))
        ft.main(a2 + ['--set', 'cap_stopped', '--extra-pids', '5', '--limit', '2', '--shard', '0/1', '--skip', str(tmp_path / 'skip.json')])
        got = [json.loads(l)['pid'] for l in out2.read_text().splitlines()]
        assert got == ['1', '5']  # first 2 of the set, plus 5, without the skipped 2
        assert json.loads(out2.read_text().splitlines()[1])['donor'] == '1'  # extra pid: donor from cap_all's map
        with pytest.raises(SystemExit):
            ft.main(args + ['--chunk', '3'])  # refused before anything is loaded
    finally:
        torch.set_grad_enabled(True)


def test_select_pids():
    doc = {'sets': {'s': ['a', 'b', 'c', 'd', 'e']}}
    assert ft.select_pids(doc, 's') == ['a', 'b', 'c', 'd', 'e']
    assert ft.select_pids(doc, 's', limit=3, extra=['z', 'a']) == ['a', 'b', 'c', 'z']
    assert ft.select_pids(doc, 's', shard='1/2', done=['d']) == ['b']
    assert ft.select_pids(doc, 's', shard='0/2') + ft.select_pids(doc, 's', shard='1/2') == ['a', 'c', 'e', 'b', 'd']
    with pytest.raises(SystemExit):
        ft.select_pids(doc, 'nope')


def test_zlib_ratio_and_code_version():
    assert ft.zlib_ratio('') is None
    rnd = random.Random(0)
    assert ft.zlib_ratio('ab' * 5000) < 0.05 < 0.5 < ft.zlib_ratio(''.join(rnd.choice(string.ascii_letters) for _ in range(3000)))
    assert ft.zlib_ratio('x' * 10 + 'ab' * 5000) == ft.zlib_ratio('ab' * 5000)  # only the last 3,000 characters
    v = ft.code_version()
    assert len(v) == 12 and v == ft.code_version() and v != t1.code_version()


# ---- analysis -----------------------------------------------------------------------------------------------------------------
def make_row(pid, origin='continued', qtype='free_form', counts=None, cut=None, ends=None, greedy=None, n_t1=3, valid=True,
             match=True, conds=('A0', 'C', 'F')):
    counts = counts or {}
    greedy = greedy or {}
    ends = ends or {}

    def ans(k, g):
        return [{'text': '{7}', 'valid': valid, 'correct': int(j < k), 'greedy': False} for j in range(n_t1)] \
            + [{'text': '{7}', 'valid': valid, 'correct': g, 'greedy': True}]
    answers = {c: ans(k, greedy.get(c, int(k == n_t1))) for c, k in counts.items()}
    row = {'pid': pid, 'status': 'ok', 'stratum': 'cap', 'origin': origin, 'set': 'cap_all', 'code_version': 'v',
           'scorer_version': 's', 'conds': list(conds), 'question_type': qtype, 'recon': {'tokens_match': match},
           'answers': answers, 'acc_t1': {c: k / n_t1 for c, k in counts.items()},
           'greedy': {c: answers[c][-1]['correct'] for c in answers},
           'extra': {c: {'end': ends.get(c, 'cap'), 'len': 100 if ends.get(c, 'cap') == 'cap' else 40, 'text': 'x',
                         'zlib_ratio_tail3000': 0.3} for c in counts if c in ('C', 'F', 'L')}, 'cut512': {}}
    for c, k in (cut or {}).items():
        if k is None:
            row['cut512'][c] = {'cut': 512, 'same_as_full': True, 'len_full': 40, 'acc_t1': row['acc_t1'][c], 'greedy': row['greedy'][c]}
        else:
            a = ans(k, int(k == n_t1))
            row['cut512'][c] = {'cut': 512, 'same_as_full': False, 'len_full': 900, 'answers': a, 'acc_t1': k / n_t1,
                                'greedy': a[-1]['correct']}
    return row


def synth(n=40, effect=lambda i: (1, 1, 1), **kw):
    """effect(i) -> (A0, C, F) counts."""
    R, ids = {}, []
    for i in range(n):
        a, c, f = effect(i)
        pid = f'q{i}'
        R[pid] = make_row(pid, origin='continued' if i % 2 else 'stopped', qtype='multi_choice' if i % 3 == 0 else 'free_form',
                          counts={'A0': a, 'C': c, 'F': f}, **kw)
        ids.append(pid)
    return R, ids


def test_identical_f_and_c_give_zero_difference():
    R, ids = synth(effect=lambda i: (i % 4, i % 4, i % 4))
    out = an.analyse(R, ids)
    p = out['primary_F_minus_C']
    assert out['n_main'] == 40 and p['diff_pp'] == 0 and p['ci95_pp'] == [0.0, 0.0] and p['p_boot'] == 1.0
    assert p['better'] == p['worse'] == 0 and p['same'] == 40 and p['ci_excludes_zero'] is False
    assert p['greedy_better'] == p['greedy_worse'] == 0 and p['p_mcnemar_greedy'] == 1.0


def test_known_differences_give_known_results():
    # A0 = 0, C = 1, F = 2 on every question (counts of 3): F - C = +33.33 pp, C - A0 = +33.33, F - A0 = +66.67
    R, ids = synth(effect=lambda i: (0, 1, 2))
    out = an.analyse(R, ids)
    p = out['primary_F_minus_C']
    assert p['diff_pp'] == 33.33 and p['acc_x_pp'] == 66.67 and p['acc_y_pp'] == 33.33
    assert p['ci95_pp'] == [33.33, 33.33] and p['p_boot'] == 0.0 and p['ci_excludes_zero'] is True
    assert (p['better'], p['worse'], p['same']) == (40, 0, 0)
    sec = out['secondary_descriptive_not_for_decision']
    assert sec['F-A0']['diff_pp'] == 66.67 and sec['C-A0']['diff_pp'] == 33.33
    assert set(sec) == {'F-A0', 'C-A0'}  # no L, no cut records: those pairs are absent
    # mixed: F - C = +1 on 12 questions, -1 on 4, 0 on 24 -> (12 - 4) / (40 * 3) = 6.67 pp
    R, ids = synth(effect=lambda i: (1, 1, 2 if i < 12 else 0 if i < 16 else 1))
    p = an.analyse(R, ids)['primary_F_minus_C']
    assert p['diff_pp'] == 6.67 and (p['better'], p['worse'], p['same']) == (12, 4, 24)
    lo, hi, pb = an.a1.boot_ci_p([1] * 12 + [-1] * 4 + [0] * 24, 3)
    assert (p['ci95_pp'], p['p_boot']) == ([round(lo, 2), round(hi, 2)], round(pb, 4))  # the step1_think convention
    # greedy: F right where C wrong on k questions
    R, ids = synth(effect=lambda i: (0, 0, 3 if i < 5 else 0))
    p = an.analyse(R, ids)['primary_F_minus_C']
    assert (p['greedy_better'], p['greedy_worse']) == (5, 0) and p['p_mcnemar_greedy'] == 0.0625


def test_l_and_cut_pairs_and_condition_summaries():
    R, ids = {}, []
    for i in range(30):
        pid = f'q{i}'
        counts = {'A0': 0, 'C': 1, 'F': 2, 'L': 1}
        ends = {'C': 'cap', 'F': 'close' if i % 2 else 'cap', 'L': 'eos'}
        cut = {'C': None if i < 10 else 0, 'F': None if i < 10 else 3, 'L': None}
        R[pid] = make_row(pid, counts=counts, cut=cut, ends=ends, conds=('A0', 'C', 'F', 'L'))
        ids.append(pid)
    out = an.analyse(R, ids)
    sec = out['secondary_descriptive_not_for_decision']
    assert sec['L-C']['diff_pp'] == 0.0 and sec['F-L']['diff_pp'] == 33.33
    assert sec['C_cut-C']['diff_pp'] == round(100 * (-20) / 90, 2) == -22.22 and (sec['C_cut-C']['worse'], sec['C_cut-C']['same']) == (20, 10)
    assert sec['F_cut-F']['diff_pp'] == round(100 * 20 / 90, 2) and sec['F_cut-C_cut']['n'] == 30
    assert sec['F_cut-C_cut']['diff_pp'] == round(100 * (10 * 1 + 20 * 3) / 90, 2)  # (F_cut - C_cut): +1 x10, +3 x20
    assert sec['L_cut-L']['diff_pp'] == 0.0
    cond = out['conditions_not_for_decision']
    assert cond['F']['end'] == {'close': 15, 'eos': 0, 'cap': 15} and cond['L']['end'] == {'close': 0, 'eos': 30, 'cap': 0}
    assert cond['C']['mean_len'] == 100.0 and cond['F']['mean_len'] == 70.0 and cond['F']['median_len'] == 70.0
    assert cond['C_cut']['n_same_as_full'] == 10 and cond['C_cut']['n_new_answers'] == 20 and 'A0' in cond
    assert cond['A0']['acc_t1_pp'] == 0.0 and cond['F']['greedy_acc_pp'] == 0.0 and cond['F']['mean_zlib_ratio_tail3000'] == 0.3
    assert cond['A0']['valid_answer_rate'] == 1.0 and out['flags'] == []
    assert out['conds_run'] == ['A0', 'C', 'F', 'L']


def test_valid_rate_flag():
    R, ids = synth(n=6, valid=False)
    out = an.analyse(R, ids)
    assert out['flags'] == [f'valid_answer_rate<0.95 for {c}' for c in ('A0', 'C', 'F')]
    assert out['conditions_not_for_decision']['F']['valid_rate_flag'] is True


def test_splits_by_origin_type_and_f_end_and_recon_sensitivity():
    R, ids = {}, []
    for i in range(24):
        pid = f'q{i}'
        origin, qt = ('continued' if i < 12 else 'stopped'), ('multi_choice' if i % 2 else 'free_form')
        f = 2 if origin == 'continued' else 1  # F - C = +1 only in the continued origin
        R[pid] = make_row(pid, origin, qt, counts={'A0': 0, 'C': 1, 'F': f}, ends={'F': 'close' if i % 4 == 0 else 'cap'},
                          match=i != 0)
        ids.append(pid)
    out = an.analyse(R, ids)
    sp = out['splits_F_minus_C_not_for_decision']
    assert sp['origin']['continued']['F-C']['diff_pp'] == 33.33 and sp['origin']['stopped']['F-C']['diff_pp'] == 0.0
    assert sp['origin']['continued']['n'] == sp['origin']['stopped']['n'] == 12
    assert set(sp['question_type']) == {'multi_choice', 'free_form'} and sp['question_type']['multi_choice']['n'] == 12
    assert sp['F_extra_thinking_ended']['self_ended']['n'] == 6 and sp['F_extra_thinking_ended']['cap']['n'] == 18
    assert sp['origin']['continued']['acc_t1_pp']['F'] == 66.67
    s = out['sensitivity_excluding_recon_mismatch']
    assert s['n_excluded'] == 1 and s['F-C']['n'] == 23 and out['primary_F_minus_C']['n'] == 24


def test_question_type_from_the_types_file_and_unknown():
    R, ids = synth(n=6, effect=lambda i: (0, 1, 2))
    for r in R.values():
        del r['question_type']
    out = an.analyse(R, ids, types={'q0': 'multi_choice', 'q1': 'free_form'})
    assert {k: v['n'] for k, v in out['splits_F_minus_C_not_for_decision']['question_type'].items()} == \
        {'multi_choice': 1, 'free_form': 1, 'unknown': 4}


def test_errors_missing_rows_outside_the_set_and_lacking_conditions():
    R, ids = synth(n=10, effect=lambda i: (0, 1, 2))
    R['q3'] = {'pid': 'q3', 'status': 'error', 'stage': 'generate', 'error': 'boom', 'code_version': 'v', 'set': 'cap_all'}
    R['q4'] = {'pid': 'q4', 'status': 'error', 'stage': 'load', 'error': 'KeyError', 'code_version': 'v', 'set': 'cap_all',
               'nonfinite': ['F']}
    del R['q9']
    del R['q5']['answers']['F'], R['q5']['acc_t1']['F']
    R['zz'] = make_row('zz', counts={'A0': 0, 'C': 1, 'F': 2})
    out = an.analyse(R, ids)
    assert out['n_ids'] == 10 and out['n_rows'] == 9 and out['n_ok'] == 7 and out['n_error'] == 2 and out['missing_n'] == 1
    assert out['missing_first'] == ['q9'] and out['rows_not_in_set'] == ['zz'] and out['rows_lacking_C_or_F'] == ['q5']
    assert out['n_main'] == 6 and out['primary_F_minus_C']['n'] == 6
    assert out['errors'] == {'n': 2, 'pids': ['q3', 'q4'], 'by_stage': {'generate': 1, 'load': 1}, 'first_messages': ['boom', 'KeyError']}
    assert out['secondary_descriptive_not_for_decision']['F-A0']['n'] == 6  # q5 has no F
    assert out['secondary_descriptive_not_for_decision']['C-A0']['n'] == 7  # q5 still has A0 and C
    assert an.analyse({}, ids)['n_main'] == 0 and 'primary_F_minus_C' not in an.analyse({}, ids)


def test_duplicate_pids_are_refused(tmp_path):
    r = make_row('1', counts={'C': 1, 'F': 1})
    a, b = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'
    a.write_text(json.dumps(r) + '\n')
    b.write_text(json.dumps(r) + '\n')
    with pytest.raises(SystemExit):
        an.load([str(tmp_path / '*.jsonl')])
    b.write_text(json.dumps(make_row('2', counts={'C': 1, 'F': 1})) + '\n')
    R, files = an.load([str(tmp_path / '*.jsonl')])
    assert set(R) == {'1', '2'} and len(files) == 2
    with pytest.raises(SystemExit):
        an.load([str(tmp_path / 'none*.jsonl')])


def test_stored_accuracy_is_checked_against_the_answers():
    R, ids = synth(n=4, effect=lambda i: (0, 1, 2))
    R['q0']['acc_t1']['F'] = 0.0  # disagrees with the three answers
    with pytest.raises(AssertionError):
        an.analyse(R, ids)


def test_analysis_runs_on_rows_written_by_the_run_script(lm):
    rows = tiny_group(lm)
    ids = [r['pid'] for r in rows]
    R = {r['pid']: r for r in rows}
    out = an.analyse(R, ids)
    assert out['n_main'] == 3 and out['n_error'] == 0
    assert set(out['secondary_descriptive_not_for_decision']) >= {'F-A0', 'C-A0', 'L-C', 'F-L', 'F_cut-C_cut'}
    json.dumps(out)


def test_cli_prints_and_refuses_overwrite(tmp_path, capsys):
    R, ids = synth(n=6, effect=lambda i: (0, 1, 2))
    res, idf, out = tmp_path / 'r.jsonl', tmp_path / 'ids.json', tmp_path / 'o.json'
    res.write_text(''.join(json.dumps(R[p]) + '\n' for p in ids))
    idf.write_text(json.dumps({'ids': ids, 'sets': {'cap_all': ids, 'cap_stopped': ids[:2]}}))
    assert an.main([str(res), '--ids', str(idf), '--out', str(out)]) == 0
    assert json.loads(out.read_text())['n_main'] == 6
    with pytest.raises(SystemExit):
        an.main([str(res), '--ids', str(idf), '--out', str(out)])
    capsys.readouterr()
    assert an.main([str(res), '--ids', str(idf), '--set', 'cap_stopped']) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed['set'] == 'cap_stopped' and printed['n_ids'] == 2 and printed['n_main'] == 2


# ---- the Kaggle builder ----------------------------------------------------------------------------------------------------------
def test_builder_makes_a_folder_and_refuses_bad_settings(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fb, 'HERE', tmp_path)
    assert fb.main(['t1', '--owner', 'me', '--conds', 'A0,C,F,L', '--limit', '8', '--extra-pids', '12,34']) == 0
    run = tmp_path / 'runs' / 't1'
    rc = json.loads((run / 'run_config.json').read_text())
    assert rc['think_extra'] == 2048 and rc['chunk'] == 2048 and rc['cut'] == 512 and rc['group'] == 4 and rc['set'] == 'cap_all'
    assert rc['conds'] == ['A0', 'C', 'F', 'L'] and rc['kernel'] == 'me/visref-t1'
    ra = rc['run_args']
    assert ra[ra.index('--chunk') + 1] == '2048' and ra[ra.index('--conds') + 1] == 'A0,C,F,L'
    assert ra[ra.index('--limit') + 1] == '8' and ra[ra.index('--extra-pids') + 1] == '12,34'
    for f in ('diag/full_think.py', 'diag/step1_think.py', 'diag/inject_sensitivity.py', 'diag/full_think_ids.json',
              'diag/step1_store_bmain.json', 'run.py', 'visref/engine.py'):
        assert f in rc['source_sha256']
    nb = json.loads((run / 'notebook.ipynb').read_text())
    src = ''.join(nb['cells'][0]['source'])
    ast.parse(src)
    assert 'diag/full_think.py' in src and "'--ids', 'diag/full_think_ids.json'" in src
    cfg = json.loads(base64.b64decode(re.search(r"CFG = json.loads\(base64.b64decode\('([^']+)'\)", src).group(1)))
    assert cfg['run_args'] == ra
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(re.search(r"b64decode\('([^']+)'\)\)\)\.extractall", src).group(1))))
    assert (REPRO / 'diag/full_think.py').read_bytes() == z.read('diag/full_think.py')
    assert 'diag/inject_sensitivity.py' in z.namelist() and 'diag/step1_think.py' in z.namelist()
    meta = json.loads((run / 'kernel-metadata.json').read_text())
    assert meta['id'] == 'me/visref-t1' and meta['enable_gpu'] is True and meta['is_private'] is True
    with pytest.raises(SystemExit):
        fb.main(['t1', '--owner', 'me'])  # an existing run record is never overwritten
    for bad in (['--think-extra', '4096', '--chunk', '1024'], ['--conds', 'A0,Q'], ['--conds', 'C,C']):
        with pytest.raises(SystemExit):
            fb.main(['t2', '--owner', 'me'] + bad)
    assert fb.main(['t3', '--owner', 'me', '--think-extra', '1024']) == 0
    assert json.loads((tmp_path / 'runs/t3/run_config.json').read_text())['chunk'] == 1024
    assert not (REPRO / 'kaggle/runs/t1').exists()  # the repo's own runs folder is untouched
