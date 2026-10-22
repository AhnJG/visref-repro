"""CPU tests for diag/thumb_gate_ids.py, diag/thumb_gate.py, diag/analyze_thumb_gate.py and kaggle/build_thumb_gate.py
(fake engine / tiny LM). The tile size (256 image tokens) is patched to 4 for the tiny-LM tests."""
import ast
import base64
import io
import json
import math
import re
import shutil
import sys
import types
import zipfile
from pathlib import Path

import pytest
import torch

REPRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO / 'diag'))
sys.path.insert(0, str(REPRO / 'kaggle'))
import analyze_thumb_gate as an  # noqa: E402
import build_thumb_gate as bt  # noqa: E402
import inject_sensitivity as isn  # noqa: E402
import inject_sensitivity_ids as isn_ids  # noqa: E402
import step1_think as t1  # noqa: E402
import thumb_gate as tg  # noqa: E402
import thumb_gate_ids as ids_mod  # noqa: E402
from test_repro import lm  # noqa: E402,F401  (pytest fixture)
from test_step1_think import NL, SP as SP0, TOK, FakeEng  # noqa: E402

from run import pack  # noqa: E402
from visref.engine import Engine, Request, Result  # noqa: E402

SP = {**SP0, 'close_ids': frozenset(i for i, s in enumerate(TOK.vocab) if s.startswith('}'))}
TE, EOS = SP['think_end'], SP['eos']
ROW = {'question_type': 'free_form', 'answer_type': 'integer', 'answer': '7', 'precision': None, 'choices': None}
CFG = tg.resolve_cfg({**tg.DEFAULTS, 'think_cap': 6, 'probe_cap': 4, 'n_t1': 2, 'batch_tokens': 4000})
SMALL = 4  # the patched tile size of the tiny-LM tests


@pytest.fixture
def small_tile(monkeypatch):
    monkeypatch.setattr(tg, 'TILE', SMALL)
    return SMALL


def make_item(d, pid='5', n_vis=12, n_other=None, seed=0, emb=None, vis_other=True):
    """An item as thumb_gate.main builds it: base = [3 rows][image rows][3 rows], pos = the image slots."""
    g = torch.Generator().manual_seed(seed)
    emb = emb or (lambda ids: torch.randn(len(ids), d, generator=g))
    vis = torch.randn(n_vis, d, generator=g)
    base = torch.cat([emb(torch.randint(5, 90, (3,), generator=g).tolist()), vis, emb(torch.randint(5, 90, (3,), generator=g).tolist())])
    pos = torch.arange(3, 3 + n_vis)
    other = torch.randn(n_other or n_vis, d, generator=g) if vis_other else None
    it = {'pid': pid, 'row': ROW, 'sample': {'think_end': 'cap'}, 'S': tg.make_settings(base, vis, pos, other), 'pos': pos,
          'n_vis': n_vis, 'multi': n_vis > tg.TILE, 'gt_ids': tg.gt_ids_of(TOK, ROW), 'z': TOK.encode('think abc'), 'sep': NL,
          'sep_text': '\n', 'recon': {'tokens_match': True}, 'other_pid': '9', 'other_n_visual': int(other.shape[0]) if vis_other else None}
    it['base'], it['vis'], it['vis_other'] = base, vis, other
    return it


def lm_item(lm, pid='5', n_vis=12, seed=0, **kw):
    emb = lm.get_input_embeddings()
    return make_item(lm.config.hidden_size, pid, n_vis, seed=seed, emb=lambda ids: emb(torch.as_tensor(ids)).detach(), **kw)


# ---- ids list ---------------------------------------------------------------------------------------------------
def test_ids_file_equals_a_fresh_build():
    assert ids_mod.main(['--check']) == 0
    assert ids_mod.OUT.read_text() == ids_mod.text(ids_mod.fresh())
    assert ids_mod.text(ids_mod.fresh()) == ids_mod.text(ids_mod.fresh())  # deterministic


def test_ids_trial_rules_on_the_real_data():
    d = json.loads(ids_mod.OUT.read_text())
    order = [str(p) for p in json.loads(ids_mod.ORDER.read_text())]
    store = json.loads(ids_mod.STORE.read_text())['pids']
    import glob
    nv = {p: v['n_visual'] for p, v in isn_ids.tsr_probes(glob.glob(str(REPRO / isn_ids.RUNS))).items()}
    part1 = json.loads(ids_mod.PART1.read_text())
    assert d['main'] == part1['part1'] == order[:200] and len(d['main']) == 200
    pool = order[200:]
    t = d['trial']
    assert len(t) == len(set(t)) == 5 and all(p in pool for p in t) and not set(t) & set(d['main'])
    # (a) the largest n_visual, ties: the longest stored think_tokens, then the earlier position
    best = max(pool, key=lambda p: (nv[p], int(store[p]['think_tokens']), -pool.index(p)))
    assert t[0] == best and nv[t[0]] == max(nv[p] for p in pool)
    # (b) the first with n_visual == 256
    assert t[1] == next(p for p in pool if nv[p] == 256)
    # (c)(d)(e) the first not yet chosen multi-tile question of each stored ending
    chosen = t[:2]
    for p, end in zip(t[2:], ('cap', 'close', 'eos')):
        want = next(q for q in pool if nv[q] > 256 and q not in chosen and store[q]['think_end'] == end)
        assert p == want and store[p]['think_end'] == end and nv[p] > 256
        chosen.append(p)
    # donors: main = the 4.11 donor, trial = donors(order[200:])
    for p in d['main']:
        assert d['donor'][p] == part1['donor'][p]
    don = isn_ids.donors(pool, {p: nv[p] for p in pool})
    for p in t:
        assert d['donor'][p] == don[p]
    keep = d['main'] + t
    assert set(d['donor']) == set(keep)
    assert set(d['n_visual']) == set(keep) | set(d['donor'].values())
    assert all(d['n_visual'][p] == nv[p] for p in d['n_visual'])
    assert all(d['multi'][p] == (nv[p] > 256) for p in d['n_visual'])
    assert d['stratum'] == {p: store[p]['think_end'] for p in keep}
    assert all(d['n_visual'][d['donor'][p]] >= d['n_visual'][p] for p in keep)
    assert 'trial' in d['rules'] and 'multi' in d['rules']


def test_ids_pick_trial_synthetic_rule_and_fallback():
    pool = [f'p{i}' for i in range(8)]
    nvis = dict(zip(pool, (768, 256, 3328, 3328, 768, 1792, 256, 768)))
    store = {p: {'think_end': e, 'think_tokens': tt} for p, e, tt in zip(
        pool, ('cap', 'cap', 'close', 'cap', 'cap', 'cap', 'cap', 'close'), (10, 10, 10, 50, 10, 10, 10, 10))}
    # (a) p3 (3328, longer think), (b) p1 (first 256), (c) cap -> p0, (d) close -> p2, (e) no 'eos' left -> the next multi-tile: p4
    assert ids_mod.pick_trial(pool, nvis, store) == ['p3', 'p1', 'p0', 'p2', 'p4']
    store['p2']['think_tokens'] = 50  # a tie on both: the earlier position wins
    assert ids_mod.pick_trial(pool, nvis, store)[0] == 'p2'
    with pytest.raises(AssertionError):
        ids_mod.build(pool, pool[:3], {}, nvis, store)


# ---- the two inputs ---------------------------------------------------------------------------------------------------
def test_strip_thumbnail():
    one, three, four = torch.randn(256, 4), torch.randn(768, 4), torch.randn(1024, 4)
    assert tg.strip_thumbnail(one) is one
    assert torch.equal(tg.strip_thumbnail(three), three[:512]) and tg.strip_thumbnail(three).shape[0] == 512
    assert torch.equal(tg.strip_thumbnail(four), four[:768])
    for bad in (300, 100, 0, 257):
        with pytest.raises(ValueError):
            tg.strip_thumbnail(torch.randn(bad, 4))


def test_minus_base_is_the_plus_base_without_the_last_256_slot_rows():
    n = 768
    base = torch.randn(n + 30, 6)
    pos = torch.arange(11, 11 + n)
    mb = tg.minus_base(base, pos, n)
    assert mb.shape[0] == base.shape[0] - 256
    assert torch.equal(mb, torch.cat([base[:11 + 512], base[11 + 768:]]))  # prompt before, first 512 slots, prompt after
    assert tg.minus_base(base[:256 + 30], torch.arange(11, 11 + 256), 256) is not None
    one = torch.randn(256 + 5, 6)
    assert tg.minus_base(one, torch.arange(2, 258), 256) is one  # single tile: the same tensor
    with pytest.raises(ValueError):
        tg.minus_base(base, pos[:-1], n)
    with pytest.raises(ValueError):
        tg.minus_base(base, torch.cat([pos[:100], pos[101:] + 5, torch.tensor([0])])[:n], n)  # not contiguous


def test_slot_removal_check_on_fake_id_lists():
    pos = torch.arange(3, 3 + 512)
    plus = [1, 2, 3] + [7] * 512 + [9, 9]
    minus = [1, 2, 3] + [7] * 256 + [9, 9]
    tg.check_slot_removal(plus, minus, pos)
    for bad in ([1, 2, 3] + [7] * 257 + [9, 9], [1, 2, 4] + [7] * 256 + [9, 9], [1, 2, 3] + [7] * 256 + [9]):
        with pytest.raises(AssertionError):
            tg.check_slot_removal(plus, bad, pos)


def test_verify_prompt_ids_uses_the_tokenizer(monkeypatch):
    class Tok:
        def __call__(self, text, return_tensors=None):
            ids = [{'<': 1, '>': 2}.get(ch, 3) for ch in text]
            return types.SimpleNamespace(input_ids=torch.tensor([ids]))
    monkeypatch.setattr(tg, 'build_prompt', lambda q, n: 'a' + 'x' * n + 'b' + q)
    pos = torch.arange(1, 1 + 512)
    tg.verify_prompt_ids(Tok(), 'hi', 512, pos)  # char tokenizer: removing 256 slots gives the shorter prompt
    monkeypatch.setattr(tg, 'build_prompt', lambda q, n: 'a' + 'x' * n + ('<' if n < 512 else '') + 'b' + q)
    with pytest.raises(AssertionError):
        tg.verify_prompt_ids(Tok(), 'hi', 512, pos)


def test_make_settings_multi_and_single_tile():
    g = torch.Generator().manual_seed(1)
    base = torch.randn(768 + 10, 4, generator=g)
    vis, pos = base[5:5 + 768], torch.arange(5, 5 + 768)
    donor = torch.randn(1024, 4, generator=g)
    S = tg.make_settings(base, vis, pos, donor)
    assert torch.equal(S['tplus']['vis'], vis) and S['tplus']['base'] is base and S['tplus']['thumb'] is None
    assert torch.equal(S['tminus']['vis'], vis[:512]) and torch.equal(S['tminus']['thumb'], vis[512:])
    assert torch.equal(S['tplus']['other'], donor[:768])                       # T+: the donor's first N tokens
    assert torch.equal(S['tminus']['other'], donor[:768][:512]) and S['tminus']['other'].shape[0] == 512  # T-: stripped donor
    assert torch.equal(S['tminus']['base'][5:5 + 512], vis[:512]) and S['tminus']['base'].shape[0] == base.shape[0] - 256
    sbase = torch.randn(256 + 10, 4, generator=g)
    svis, spos = sbase[5:261], torch.arange(5, 261)
    for d in (torch.randn(256, 4, generator=g), torch.randn(768, 4, generator=g)):  # a single-tile question, any donor
        S = tg.make_settings(sbase, svis, spos, d)
        assert torch.equal(S['tplus']['base'], S['tminus']['base']) and torch.equal(S['tplus']['vis'], S['tminus']['vis'])
        assert S['tminus']['thumb'] is None
        assert torch.equal(S['tplus']['other'], S['tminus']['other'])          # identical input, control included
    assert tg.make_settings(sbase, svis, spos, None)['tminus']['other'] is None


# ---- contexts -----------------------------------------------------------------------------------------------------------
def test_tplus_contexts_equal_inject_sensitivity_contexts():
    eng = FakeEng()
    for n_vis in (768, 256):
        it = make_item(eng.emb.embedding_dim, n_vis=n_vis, n_other=n_vis + 256, emb=eng.embed)
        ref = isn.contexts(dict(it), eng, TOK, SP, {**isn.DEFAULTS})
        mine, meta = tg.contexts(it['pid'], it['S']['tplus'], it['z'], it['sep'], eng, SP, CFG)
        assert list(mine) == ['none', 'dpp', 'rand', 'all', 'other']
        for c in mine:
            assert torch.equal(mine[c], ref[c]), (n_vis, c)
        r = dict(it)
        isn.contexts(r, eng, TOK, SP, {**isn.DEFAULTS})
        assert meta['sel'] == r['sel'] and meta['sel_rand'] == r['sel_rand'] and meta['m'] == r['m'] and len(meta['sel']) > 0


def test_empty_thinking_gives_no_selection(small_tile):
    eng = FakeEng()
    it = make_item(eng.emb.embedding_dim, n_vis=12, emb=eng.embed)
    ctxs, meta = tg.contexts('5', it['S']['tplus'], [], [], eng, SP, CFG)
    assert meta['sel'] == [] and meta['sel_rand'] == [] and torch.equal(ctxs['dpp'], ctxs['none'])


def test_tminus_contexts_thumb_other_and_single_tile(small_tile):
    eng = FakeEng()
    it = make_item(eng.emb.embedding_dim, n_vis=12, n_other=16, emb=eng.embed)  # 2 tiles + thumbnail
    S = it['S']
    cp, mp = tg.contexts('5', S['tplus'], it['z'], it['sep'], eng, SP, CFG)
    cm, mm = tg.contexts('5', S['tminus'], it['z'], it['sep'], eng, SP, CFG)
    e = eng.embed
    tail = e([TE] + SP['probe_prompt'])
    prefix_m = torch.cat([S['tminus']['base'], e(it['z'] + it['sep'])])
    vm = it['vis'][:8]
    assert list(cm) == ['none', 'dpp', 'rand', 'all', 'other', 'thumb'] and 'thumb' not in cp
    assert torch.equal(cm['thumb'], torch.cat([prefix_m, it['vis'][-4:], tail]))  # prefix_minus + vis_plus[-256:] + tail
    assert torch.equal(cm['all'], torch.cat([prefix_m, vm, tail])) and torch.equal(cm['none'], torch.cat([prefix_m, tail]))
    sel = tg.select_tokens(vm, e(it['z']), tg.budget(8, 0.3))
    assert mm['sel'] == sel and mm['m'] == tg.budget(8, 0.3) and mm['m'] != mp['m'] and max(sel) < 8
    assert torch.equal(cm['dpp'], torch.cat([prefix_m, vm[sel], tail]))
    assert torch.equal(cm['rand'], torch.cat([prefix_m, vm[mm['sel_rand']], tail])) and len(mm['sel_rand']) == len(sel)
    assert torch.equal(cm['other'], torch.cat([prefix_m, it['vis_other'][:12][:8], tail]))  # the stripped donor, cut to |V_minus|
    assert cm['other'].shape == cm['all'].shape
    assert cm['thumb'].shape[0] == cm['none'].shape[0] + 4
    # a single-tile question: T- contexts are the T+ contexts, and there is no thumb context
    one = make_item(eng.emb.embedding_dim, n_vis=4, n_other=4, emb=eng.embed)
    p1, _ = tg.contexts('5', one['S']['tplus'], one['z'], one['sep'], eng, SP, CFG)
    m1, _ = tg.contexts('5', one['S']['tminus'], one['z'], one['sep'], eng, SP, CFG)
    assert list(p1) == list(m1) == ['none', 'dpp', 'rand', 'all', 'other']
    assert all(torch.equal(p1[c], m1[c]) for c in p1)


def test_read_values_equal_inject_sensitivity_part1_on_the_tiny_lm(lm, small_tile):
    torch.set_grad_enabled(False)
    try:
        eng = Engine(lm)
        it = lm_item(lm, n_vis=12, n_other=16)
        it['row'] = ROW
        p1 = isn.part1({**it, 'sep_text': '\n', 'base': it['base'], 'vis': it['vis'], 'vis_other': it['vis_other']}, eng, TOK, SP,
                       isn.DEFAULTS, {'ids': TOK.encode('{8}'), 'valid': True})
        ctxs, meta = tg.contexts(it['pid'], it['S']['tplus'], it['z'], it['sep'], eng, SP, CFG)
        got = tg.read_contexts(eng, SP, ctxs, tg.gt_ids_of(TOK, ROW))
        assert p1['gt_ids'] == tg.gt_ids_of(TOK, ROW) == TOK.encode('{7}')
        assert list(got['cond']) == ['none', 'dpp', 'rand', 'all', 'other']
        for c, v in got['cond'].items():
            ref = p1['cond'][c]
            for k in ('logp_gt', 'p_gt', 'logp_gt_full', 'top32', 'kl_from_none', 'top1', 'entropy_first', 'ctx_len'):
                assert v[k] == ref[k], (c, k)
            assert len(v['top32']) == 32 and abs(v['p_gt'] - math.exp(v['logp_gt'])) < 1e-5
        assert got['cond']['none']['kl_from_none'] == 0
        assert got['kl_pairs'] == {k: p1['kl_pairs'][k] for k in ('dpp||rand', 'all||other')}
        assert set(got) == {'cond', 'kl_pairs'} and not any('tsr' in k for v in got['cond'].values() for k in v)
    finally:
        torch.set_grad_enabled(True)


# ---- generation: chunk, cap, seeds, the rule for the new text ---------------------------------------------------------------
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


def test_chunk_is_refused_below_think_cap_and_zero_means_think_cap():
    assert tg.DEFAULTS['think_cap'] == 1024 and tg.DEFAULTS['chunk'] == 0 and tg.DEFAULTS['group'] == 4
    assert tg.DEFAULTS['n_t1'] == 3 and tg.DEFAULTS['probe_cap'] == 16 and tg.DEFAULTS['vis_frac'] == 0.3
    assert tg.DEFAULTS['probe_prompt'] == '\n\n\\boxed' and tg.DEFAULTS['seed'] == 42 and tg.DEFAULTS['batch_tokens'] == 72000
    assert tg.DEFAULTS['max_num'] == 12 and tg.DEFAULTS['temperature'] == 0.6 and tg.DEFAULTS['top_k'] == 50
    assert tg.resolve_cfg(tg.DEFAULTS)['chunk'] == 1024 == tg.resolve_cfg(tg.DEFAULTS)['think_cap']
    assert tg.resolve_cfg({**tg.DEFAULTS, 'think_cap': 2048})['chunk'] == 2048
    assert tg.resolve_cfg({**tg.DEFAULTS, 'chunk': 3000})['chunk'] == 3000
    with pytest.raises(SystemExit):
        tg.resolve_cfg({**tg.DEFAULTS, 'chunk': 512})  # think_cap 1024 larger than a small chunk
    base = ['--parquet', 'p.parquet', '--out', 'o.jsonl']
    a = tg.parse_args(base)
    assert a.cfg['chunk'] == 1024 and a.set == 'main' and a.reencode_check is False and a.shard == '0/1'
    assert tg.parse_args(base + ['--set', 'trial', '--reencode-check', '--limit', '5']).reencode_check is True
    with pytest.raises(SystemExit):
        tg.parse_args(base + ['--chunk', '512'])
    with pytest.raises(SystemExit):
        tg.parse_args(base + ['--set', 'other'])
    with pytest.raises(SystemExit):
        tg.parse_args(base + ['--think-cap', '2048', '--chunk', '1024'])


def test_a_chunk_finish_raises_chunk_error_and_cap_does_not():
    eng = GenEng()
    bad = {**CFG, 'think_cap': 40, 'chunk': 16}  # not through resolve_cfg: the old shape
    mk = lambda: Request(torch.randn(6, 8), frozenset({TE, EOS}), 40, None)  # noqa: E731
    old = t1.make_generate(eng, bad, pack)([mk()])[0]
    assert old.finish == 'chunk' and len(old.ids) == 16  # step1_think's generate returns the unfinished row
    with pytest.raises(tg.ChunkError):
        tg.make_generate(eng, bad, pack)([mk()])
    good = tg.resolve_cfg({**CFG, 'think_cap': 40, 'chunk': 0})
    assert good['chunk'] == 40
    r = tg.make_generate(eng, good, pack)([mk()])[0]
    assert r.finish == 'cap' and len(r.ids) == 40


def test_the_real_engine_reaches_think_cap_with_chunk_zero(lm, small_tile, monkeypatch):
    cfg = tg.resolve_cfg({**CFG, 'think_cap': 60, 'chunk': 0, 'batch_tokens': 4000})
    eng = Engine(lm, 0.6, 50, 0.95)
    torch.set_grad_enabled(False)
    orig = tg.Request  # the tiny random LM stops early now and then: forbid the stop tokens of the thinking requests
    monkeypatch.setattr(tg, 'Request', lambda ctx, stop, mx, g, **kw: orig(ctx, stop if mx == cfg['probe_cap'] else frozenset(), mx, g, **kw))
    try:
        its = [lm_item(lm, pid=str(k), n_vis=12 if k else 4, seed=k) for k in range(2)]
        tg.run_group(its, eng, TOK, SP, cfg, tg.make_generate(eng, cfg, pack))
        for it in its:
            for s in tg.SETTINGS:
                g = it['gen'][s]  # len leaves out any </think> / EOS the unstopped random model happens to draw
                assert len(g['z_ids']) == 60 and g['end'] == 'cap' and g['len'] + g['n_think_end_inside'] <= 60
        with pytest.raises(tg.ChunkError):  # chunk below think_cap never becomes 'cap' silently
            tg.run_group(its[:1], eng, TOK, SP, {**cfg, 'chunk': 16}, tg.make_generate(eng, {**cfg, 'chunk': 16}, pack))
    finally:
        torch.set_grad_enabled(True)


def test_split_generated_follows_the_split_stored_rule():
    ids = TOK.encode('abc')
    z, sep, sep_text, end = tg.split_generated(TOK, SP, ids + [TE], '\n')
    assert (z, sep, sep_text, end) == (ids, [], '', 'close')
    z, sep, sep_text, end = tg.split_generated(TOK, SP, ids + [EOS], '\n')
    assert (z, sep, sep_text, end) == (ids, [], '', 'eos')
    z, sep, sep_text, end = tg.split_generated(TOK, SP, ids, '\n')
    assert (z, sep, sep_text, end) == (ids, NL, '\n', 'cap')
    z, sep, sep_text, end = tg.split_generated(TOK, SP, TOK.encode('ab\n'), '\n')  # the text already ends with a newline
    assert (z, sep, sep_text, end) == (TOK.encode('ab\n'), [], '', 'cap')
    assert tg.split_generated(TOK, SP, [], '\n')[3] == 'cap'
    # the same pieces as split_stored gives for the equivalent stored step: (stored text, stored length, ending, generated ids)
    for text, n, end, gen_ids in (('abc', 3, 'close', TOK.encode('abc') + [TE]), ('abc', 3, 'eos', TOK.encode('abc') + [EOS]),
                                  ('abc\n', 3, 'cap', TOK.encode('abc')), ('ab\n\n', 4, 'cap', TOK.encode('ab\n\n'))):
        stored = t1.split_stored(TOK, SP, {'thinking': text, 'think_tokens': n, 'think_end': end}, '\n')
        assert tg.split_generated(TOK, SP, gen_ids, '\n')[:3] == stored[:3], (text, end)
    rec = tg.gen_record(TOK, SP, ids + [TE], '\n')
    assert 'ids' not in rec and rec['z_ids'] == ids and rec['end'] == 'close' and rec['len'] == 3 and rec['text'] == 'abc'
    assert rec['n_think_end_inside'] == 0 and rec['sep_ids'] == [] and rec['sep_text'] == ''


def scripted_spy(think_ids, answer='{7}', seen=None):
    """A generate spy: requests of max_new == think_cap are thinking, the others answers."""
    seen = [] if seen is None else seen

    def spy(reqs):
        seen.append(reqs)
        if reqs[0].max_new == CFG['think_cap']:
            return [Result(ids=list(think_ids), finish='stop' if think_ids[-1] in (TE, EOS) else 'cap') for _ in reqs]
        return [Result(ids=TOK.encode(answer), finish='stop') for _ in reqs]
    spy.seen = seen
    return spy


def test_seeds_and_requests_of_the_generation_stages(lm, small_tile):
    torch.set_grad_enabled(False)
    try:
        eng = Engine(lm)
        its = [lm_item(lm, pid=p, n_vis=n, seed=k) for k, (p, n) in enumerate((('3', 12), ('9', 4)))]
        spy = scripted_spy(TOK.encode('abcd') + [TE])
        tg.run_group(its, eng, TOK, SP, CFG, spy)
        think, ans = spy.seen
        assert len(think) == 4 and len(ans) == 2 * 2 * (CFG['n_t1'] + 1)
        for k, it in enumerate(its):
            for j, s in enumerate(tg.SETTINGS):
                r = think[2 * k + j]
                assert r.gen.initial_seed() == tg.seed_of(42, it['pid'], 'think')           # same seed in both settings
                assert r.stop_ids == frozenset({TE, EOS}) and r.max_new == CFG['think_cap']
                assert torch.equal(r.ctx, it['S'][s]['base'])                                # from base_s
        n1 = CFG['n_t1'] + 1
        for k, it in enumerate(its):
            for j, s in enumerate(tg.SETTINGS):
                chunk = ans[(2 * k + j) * n1:(2 * k + j + 1) * n1]
                want = torch.cat([it['S'][s]['base'], eng.embed(TOK.encode('abcd') + [TE] + SP['probe_prompt'])])
                assert all(torch.equal(r.ctx, want) for r in chunk)
                seeds = [r.gen.initial_seed() if r.gen is not None else None for r in chunk]
                assert seeds == [tg.seed_of(42, it['pid'], 'ans', jj) for jj in range(CFG['n_t1'])] + [None]
                assert chunk[0].sampling == t1.T1 and chunk[-1].sampling == t1.GREEDY and chunk[0].max_new == CFG['probe_cap']
            g = it['gen']['tplus']
            assert g['end'] == 'close' and g['z_ids'] == TOK.encode('abcd') and g['sep_ids'] == [] and len(g['answers']) == n1
            assert g['acc_t1'] == 1.0 and g['greedy'] == 1 and 'ids' not in g and g['answers'][-1]['greedy'] is True
            assert set(it['stored']) == set(it['new']) == set(tg.SETTINGS)
            # the new-text read is on the generated text: its contexts end the thinking with that z
            assert it['new']['tplus']['cond']['none']['ctx_len'] == it['S']['tplus']['base'].shape[0] + 4 + 0 + 1 + len(SP['probe_prompt'])
            assert it['stored']['tplus']['cond']['none']['ctx_len'] == it['S']['tplus']['base'].shape[0] + 9 + 1 + 1 + len(SP['probe_prompt'])
    finally:
        torch.set_grad_enabled(True)


def test_single_tile_item_gives_identical_inputs_text_and_readings(lm, small_tile):
    torch.set_grad_enabled(False)
    try:
        eng = Engine(lm, 0.6, 50, 0.95)
        cfg = tg.resolve_cfg({**CFG, 'think_cap': 12, 'chunk': 0})
        it = lm_item(lm, pid='7', n_vis=4, seed=3)
        tg.run_group([it], eng, TOK, SP, cfg, tg.make_generate(eng, cfg, pack))
        assert torch.equal(it['S']['tplus']['base'], it['S']['tminus']['base'])
        a, b = it['gen']['tplus'], it['gen']['tminus']
        assert a['z_ids'] == b['z_ids'] and a['text'] == b['text'] and a['end'] == b['end']  # same seed, same input, same ids
        assert a['answers'] == b['answers'] and a['acc_t1'] == b['acc_t1']
        for stage in ('stored', 'new'):
            x, y = it[stage]['tplus'], it[stage]['tminus']
            assert x['cond'] == y['cond'] and x['sel'] == y['sel'] and x['kl_pairs'] == y['kl_pairs']
            assert 'thumb' not in x['cond'] and x['n_sel_thumb'] == 0 and 'n_sel_thumb' not in y
        # a multi-tile item: T- has fewer tokens, a thumb condition, and its own selection
        m = lm_item(lm, pid='8', n_vis=12, seed=4)
        tg.run_group([m], eng, TOK, SP, cfg, tg.make_generate(eng, cfg, pack))
        assert 'thumb' in m['stored']['tminus']['cond'] and 'thumb' not in m['stored']['tplus']['cond']
        assert m['stored']['tplus']['cond']['all']['ctx_len'] - m['stored']['tminus']['cond']['all']['ctx_len'] == 2 * SMALL
        assert m['stored']['tplus']['n_sel_thumb'] == sum(i >= 8 for i in m['stored']['tplus']['sel'])
    finally:
        torch.set_grad_enabled(True)


# ---- errors and retry -------------------------------------------------------------------------------------------------------
def test_error_rows_and_per_question_retry(lm, small_tile):
    torch.set_grad_enabled(False)
    try:
        eng = Engine(lm)
        its = {p: lm_item(lm, pid=p, n_vis=12 if p != '4' else 4, seed=int(p)) for p in ('1', '2', '3', '4')}

        def mk(pid):
            if pid == '2':
                raise KeyError('no image for 2')
            return its[pid]

        bad = {tg.seed_of(42, '3', 'ans', 0)}
        calls = []

        def spy(reqs):
            calls.append(len(reqs))
            if any(r.gen is not None and r.gen.initial_seed() in bad for r in reqs):
                raise RuntimeError('boom')
            return scripted_spy(TOK.encode('ok') + [TE])(reqs)

        rows = tg.process_group(['1', '2', '3', '4'], mk, eng, TOK, SP, CFG, spy, 'v', 'main')
        assert [r['pid'] for r in rows] == ['1', '2', '3', '4'] and [r['status'] for r in rows] == ['ok', 'error', 'error', 'ok']
        assert rows[1]['stage'] == 'load' and 'KeyError' in rows[1]['error'] and rows[1]['set'] == 'main'
        assert rows[2]['stage'] == 'generate' and 'boom' in rows[2]['error'] and rows[2]['code_version'] == 'v'
        assert len(calls) > 3  # the group raised, each question was run alone
        ok = rows[0]
        assert set(ok) == {'pid', 'status', 'set', 'stratum', 'question_type', 'multi', 'n_tiles', 'n_visual', 'other_pid',
                           'other_n_visual', 'same_input', 'code_version', 'scorer_version', 'cfg', 'recon', 'gt_ids',
                           'answer_gt', 'stored', 'gen', 'new', 'timing'}
        assert ok['multi'] is True and ok['n_tiles'] == 2 and ok['n_visual'] == {'tplus': 12, 'tminus': 8} and ok['same_input'] is False
        assert ok['stratum'] == 'cap' and ok['question_type'] == 'free_form' and ok['other_pid'] == '9' and ok['set'] == 'main'
        assert ok['gt_ids'] == TOK.encode('{7}') and ok['answer_gt'] == '7' and ok['cfg'] == CFG and ok['scorer_version']
        assert set(ok['stored']['tplus']) == {'cond', 'kl_pairs', 'm', 'sel', 'sel_rand', 'n_sel', 'n_sel_thumb'}
        assert set(ok['stored']['tminus']) == {'cond', 'kl_pairs', 'm', 'sel', 'sel_rand', 'n_sel'}
        assert set(ok['gen']['tplus']) == {'end', 'len', 'text', 'n_think_end_inside', 'z_ids', 'sep_ids', 'sep_text', 'answers',
                                           'acc_t1', 'greedy'}
        assert {'encode_s', 'stored_read_s', 'new_read_s'} <= set(ok['timing'])
        assert rows[3]['multi'] is False and rows[3]['n_tiles'] == 1 and rows[3]['same_input'] is True
        assert rows[3]['n_visual'] == {'tplus': 4, 'tminus': 4}
        json.dumps(rows)
        # non-finite logits in one generation: the question is an error row, the others are not affected
        def spy2(reqs):
            out = scripted_spy(TOK.encode('ok') + [TE])(reqs)
            return [None if (reqs[0].max_new == CFG['think_cap'] and i == 1) else r for i, r in enumerate(out)]

        rows = tg.process_group(['1'], lambda p: its[p], eng, TOK, SP, CFG, spy2, 'v', 'main')
        assert rows[0]['status'] == 'error' and rows[0]['nonfinite'] == ['tminus'] and 'non-finite' in rows[0]['error']

        def spy3(reqs):  # non-finite only in an answer of T+
            out = scripted_spy(TOK.encode('ok') + [TE])(reqs)
            return [None if (reqs[0].max_new != CFG['think_cap'] and i == 1) else r for i, r in enumerate(out)]

        rows = tg.process_group(['1'], lambda p: its[p], eng, TOK, SP, CFG, spy3, 'v', 'main')
        assert rows[0]['status'] == 'error' and rows[0]['nonfinite'] == ['tplus']
    finally:
        torch.set_grad_enabled(True)


def test_a_chunk_error_stops_the_run_even_in_the_error_handling(lm, small_tile, monkeypatch):
    torch.set_grad_enabled(False)
    try:
        eng = Engine(lm, 0.6, 50, 0.95)
        bad = {**CFG, 'think_cap': 40, 'chunk': 16, 'batch_tokens': 4000}
        orig = tg.Request
        monkeypatch.setattr(tg, 'Request', lambda ctx, stop, mx, g, **kw: orig(ctx, stop if mx == bad['probe_cap'] else frozenset(), mx, g, **kw))
        its = {p: lm_item(lm, pid=p, seed=int(p)) for p in ('1', '2')}
        with pytest.raises(tg.ChunkError):  # not turned into per-question error rows
            tg.process_group(['1', '2'], its.__getitem__, eng, TOK, SP, bad, tg.make_generate(eng, bad, pack), 'v', 'main')
    finally:
        torch.set_grad_enabled(True)


def test_select_pids():
    doc = {'main': ['a', 'b', 'c', 'd', 'e'], 'trial': ['x', 'y']}
    assert tg.select_pids(doc, 'main') == ['a', 'b', 'c', 'd', 'e'] and tg.select_pids(doc, 'trial') == ['x', 'y']
    assert tg.select_pids(doc, 'main', limit=3, done=['b']) == ['a', 'c']
    assert tg.select_pids(doc, 'main', shard='1/2', done=['d']) == ['b']
    assert tg.select_pids(doc, 'main', shard='0/2') + tg.select_pids(doc, 'main', shard='1/2') == ['a', 'c', 'e', 'b', 'd']
    with pytest.raises(SystemExit):
        tg.select_pids(doc, 'nope')


# ---- code version -------------------------------------------------------------------------------------------------------------
def test_code_version_includes_the_imported_files(tmp_path, monkeypatch):
    v = tg.code_version()
    assert len(v) == 12 and v == tg.code_version() and v != t1.code_version() and v != isn.code_version()
    root = tmp_path / 'repro'
    for f in ('diag/step1_think.py', 'diag/inject_sensitivity.py', 'run.py', 'visref/engine.py', 'visref/model.py'):
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPRO / f, root / f)
    monkeypatch.setattr(tg, 'ROOT', root)
    v0 = tg.code_version()
    for f in ('diag/step1_think.py', 'diag/inject_sensitivity.py', 'run.py', 'visref/engine.py', 'visref/model.py'):
        p = root / f
        old = p.read_bytes()
        p.write_bytes(old + b'\n# changed\n')
        assert tg.code_version() != v0, f
        p.write_bytes(old)
        assert tg.code_version() == v0
    (root / 'visref/new_file.py').write_text('x = 1\n')  # any visref/*.py counts
    assert tg.code_version() != v0


# ---- re-encoding check ---------------------------------------------------------------------------------------------------------
class TileModel:
    """A stub model whose extract_feature works tile by tile (as the vision encoder does)."""

    def __init__(self, lm, k=4):
        self.language_model, self.k = lm, k
        torch.manual_seed(0)
        self.proj = torch.nn.Linear(3 * 8 * 8, k * lm.config.hidden_size)

    def extract_feature(self, pv):
        return self.proj(pv.reshape(pv.shape[0], -1)).reshape(pv.shape[0], self.k, -1)


def stub_transform(im):
    import numpy as np
    return torch.from_numpy(np.asarray(im.convert('RGB').resize((8, 8)), dtype='float32').copy()).permute(2, 0, 1) / 255


def test_reencode_minus_with_a_tilewise_stub_model(lm, small_tile):
    from PIL import Image
    from visref.model import dynamic_preprocess
    torch.set_grad_enabled(False)
    try:
        g = torch.Generator().manual_seed(2)
        img = Image.effect_noise((900, 450), 60).convert('RGB')  # 2 x 1 tiles of 448 -> two tiles + a thumbnail
        model = TileModel(lm)
        lm_dtype = lm.dtype
        model.language_model = types.SimpleNamespace(device=torch.device('cpu'), dtype=lm_dtype, config=lm.config)
        re = tg.reencode_minus(model, img, 12, transform=stub_transform)
        tiles_plus = dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=12)
        assert len(tiles_plus) == 3
        plus = model.extract_feature(torch.stack([stub_transform(t) for t in tiles_plus])).reshape(-1, lm.config.hidden_size)
        assert re.shape == (2 * model.k, lm.config.hidden_size) and plus.shape[0] == 3 * model.k
        assert torch.equal(re, plus[:-model.k])  # encoding tile by tile: re-encoding equals the stripped tokens
        # the record
        it = lm_item(lm, n_vis=12, seed=5)
        it['vis_re'] = it['S']['tminus']['vis'] + 1e-3 * torch.randn(8, lm.config.hidden_size, generator=g)
        eng = Engine(lm)
        rec = tg.reencode_record(it, eng, SP, CFG)
        d = (it['vis_re'] - it['S']['tminus']['vis']).abs()
        assert abs(rec['max_abs_diff'] - float(d.max())) < 1e-9 and abs(rec['mean_abs_diff'] - float(d.mean())) < 1e-9
        assert abs(rec['max_abs_vis_minus'] - float(it['S']['tminus']['vis'].abs().max())) < 1e-9
        assert abs(rec['abs_dlogp_gt_none'] - abs(rec['logp_gt_none_minus'] - rec['logp_gt_none_reencoded'])) < 1e-3
        it['vis_re'] = it['S']['tminus']['vis'].clone()  # identical tokens: no difference at all
        rec0 = tg.reencode_record(it, eng, SP, CFG)
        assert rec0['max_abs_diff'] == 0 and rec0['abs_dlogp_gt_none'] == 0
        it['vis_re'] = it['S']['tminus']['vis'][:-1]
        assert 'error' in tg.reencode_record(it, eng, SP, CFG)
    finally:
        torch.set_grad_enabled(True)


# ---- main() end to end with a stubbed model --------------------------------------------------------------------------------------------
def stub_main(lm, tmp_path, monkeypatch, verify_calls=None):
    import run as run_mod
    import visref.model as vm
    pids = ['1', '2', '3', '4', '5', '6']
    nvis = {'1': 12, '2': 12, '3': 4, '4': 4, '5': 12, '6': 12, '7': 12}
    emb = lm.get_input_embeddings()
    monkeypatch.setattr(tg, 'TILE', SMALL)

    def encode_sample(model, tok, image, query, max_num, with_positions=False):
        g = torch.Generator().manual_seed(int(image))
        vis = torch.randn(nvis[query], emb.embedding_dim, generator=g)
        pre = emb(torch.randint(5, 90, (3,), generator=g)).detach()
        base = torch.cat([pre, vis, pre.flip(0)])
        pos = torch.arange(3, 3 + nvis[query])
        return (base, vis, pos) if with_positions else (base, vis)

    rows = {p: {**ROW, 'pid': p, 'query': p, 'image': (lambda p=p: p)} for p in pids + ['7'] if p != '6'}  # '6' has no row
    monkeypatch.setattr(vm, 'load_mathvista', lambda path: rows)
    monkeypatch.setattr(vm, 'load_model', lambda m, r: (types.SimpleNamespace(language_model=lm), TOK))
    monkeypatch.setattr(vm, 'encode_sample', encode_sample)
    monkeypatch.setattr(vm, 'special_ids', lambda tok, w, p: {**SP, 'wait': TOK.encode(w), 'probe_prompt': TOK.encode(p)})
    monkeypatch.setattr(run_mod, 'fast_kv_selfcheck', lambda lm_, tok: (True, {}))
    monkeypatch.setattr(tg, 'Engine', lambda lm_, t, k, p, fast_kv=True: Engine(lm_, t, k, p, fast_kv=False))
    monkeypatch.setattr(tg, 'verify_prompt_ids', lambda tok, q, n, pos: (verify_calls.append((q, n)) if verify_calls is not None else None))
    monkeypatch.setattr(tg, 'reencode_minus', lambda model, image, max_num: encode_sample(None, None, image, image, max_num)[1][:-SMALL] + 1e-3)
    store = {'pids': {p: {'thinking': 'think abc\n', 'think_tokens': '9', 'think_end': 'cap'} for p in pids}}
    doc = {'main': pids, 'trial': ['3', '1'], 'donor': {'1': '2', '2': '1', '3': '4', '4': '5', '5': '7', '6': '1'},
           'stratum': {p: 'cap' for p in pids}}
    (tmp_path / 'store.json').write_text(json.dumps(store))
    (tmp_path / 'ids.json').write_text(json.dumps(doc))
    out = tmp_path / 'res' / 'out.jsonl'
    args = ['--parquet', 'x', '--samples', str(tmp_path / 'store.json'), '--ids', str(tmp_path / 'ids.json'), '--out', str(out),
            '--think-cap', '6', '--probe-cap', '4', '--n-t1', '2', '--batch-tokens', '4000', '--group', '2']
    return pids, out, args


def test_main_end_to_end_with_a_stub_model(lm, tmp_path, monkeypatch, capsys):
    calls = []
    pids, out, args = stub_main(lm, tmp_path, monkeypatch, calls)
    try:
        tg.main(args)
        res = [json.loads(l) for l in out.read_text().splitlines()]
        assert [r['pid'] for r in res] == pids and [r['status'] for r in res] == ['ok'] * 5 + ['error']
        assert res[5]['stage'] == 'load' and res[0]['other_pid'] == '2' and res[0]['cfg']['chunk'] == 6
        assert res[3]['other_pid'] == '5' and res[3]['other_n_visual'] == 12 and res[2]['multi'] is False and res[0]['multi'] is True
        assert res[4]['other_n_visual'] == 12 and 'reencode' not in res[0]
        assert [r['same_input'] for r in res[:5]] == [False, False, True, True, False]
        assert all(r['code_version'] == tg.code_version() and r['set'] == 'main' for r in res)
        # the tokenizer check ran for the multi-tile questions (and their donors) only
        assert {q for q, n in calls} == {'1', '2', '5', '7'} and all(n == 12 for q, n in calls)
        ev = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith('{')]
        kinds = [e['event'] for e in ev]
        assert kinds[0] == 'start' and kinds[-1] == 'end' and kinds.count('group') == 3 and kinds.count('error') == 1
        g = [e for e in ev if e['event'] == 'group']
        assert all({'group_s', 'peak_alloc_gb', 'max_rss_gb', 'n_error', 'left', 'n', 'encode_s', 'stored_read_s', 'generation_s',
                    'answers_s', 'new_read_s', 'elapsed'} <= set(e) for e in g) and g[-1]['left'] == 0
        assert {'peak_alloc_gb', 'elapsed', 'n_error', 'max_rss_gb'} <= set(ev[-1]) and ev[0]['set'] == 'main'
        tg.main(args)  # resume: everything is done (error rows included)
        assert len(out.read_text().splitlines()) == 6
        assert json.loads(capsys.readouterr().out.splitlines()[0])['todo'] == 0
        out2 = tmp_path / 'res' / 'out2.jsonl'  # trial set, re-encoding check, limit, skip
        a2 = [x if x != str(out) else str(out2) for x in args]
        (tmp_path / 'skip.json').write_text(json.dumps(['3']))
        tg.main(a2 + ['--set', 'trial', '--reencode-check', '--skip', str(tmp_path / 'skip.json')])
        got = [json.loads(l) for l in out2.read_text().splitlines()]
        assert [r['pid'] for r in got] == ['1'] and got[0]['set'] == 'trial'
        re = got[0]['reencode']
        assert abs(re['max_abs_diff'] - 1e-3) < 1e-5 and re['max_abs_vis_minus'] > 0 and 'abs_dlogp_gt_none' in re
        out3 = tmp_path / 'res' / 'out3.jsonl'
        tg.main([x if x != str(out) else str(out3) for x in args] + ['--set', 'trial', '--reencode-check'])
        got = [json.loads(l) for l in out3.read_text().splitlines()]
        assert [r['pid'] for r in got] == ['3', '1'] and got[0]['reencode'] is None and got[1]['reencode'] is not None
        with pytest.raises(SystemExit):
            tg.main(args + ['--chunk', '3'])  # refused before anything is loaded
    finally:
        torch.set_grad_enabled(True)


def test_main_stops_at_the_time_limit_and_checks_the_ids_file(lm, tmp_path, monkeypatch, capsys):
    pids, out, args = stub_main(lm, tmp_path, monkeypatch)
    try:
        tg.main(args + ['--time-limit', '0.000001', '--limit', '4'])
        ev = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith('{')]
        assert any(e['event'] == 'time_limit' for e in ev) and (not out.exists() or out.read_text() == '')
        doc = json.loads((tmp_path / 'ids.json').read_text())
        doc['stratum']['1'] = 'close'  # the ids file disagrees with the stored step 1
        (tmp_path / 'ids.json').write_text(json.dumps(doc))
        with pytest.raises(AssertionError):
            tg.main(args)
    finally:
        torch.set_grad_enabled(True)


# ---- analysis ------------------------------------------------------------------------------------------------------------------------
GT1 = 11


def cond(p, top1=3):
    return {'logp_gt': round(math.log(p), 4), 'p_gt': round(p, 6), 'top1': top1}


def stage(p_none, deltas, n_sel=10, thumb=None, top1=None):
    top1 = top1 or {}
    c = {'none': cond(p_none, top1.get('none', 3))}
    for k, d in deltas.items():
        c[k] = cond(p_none + d, top1.get(k, 3))
    out = {'cond': c, 'kl_pairs': {}, 'm': 3, 'sel': list(range(n_sel)), 'sel_rand': list(range(n_sel)), 'n_sel': n_sel}
    if thumb is not None:
        out['n_sel_thumb'] = thumb
    return out


def gen(k, g, text='a', end='cap', n=3, ln=100):
    return {'end': end, 'len': ln, 'text': text, 'n_think_end_inside': 0, 'z_ids': [1, 2, ord(text[0])], 'sep_ids': [], 'sep_text': '',
            'answers': [], 'acc_t1': k / n, 'greedy': g}


def srow(pid, i, multi=True, dm=0.10, dp=0.02, drand=0.0, qtype='free_form', stratum='cap', p_none=(0.2, 0.2), top1=None,
         kp=3, km=3, gp=1, gm=1, text=('a', 'a'), set_='main', n_vis=768, dall=(0.3, 0.3), dother=(0.1, 0.1), version='v1'):
    e = 0.01 * (-1) ** i
    nv = {'tplus': n_vis, 'tminus': n_vis - 256 if multi else n_vis}

    def st(s, d, pn, thumb):
        deltas = {'dpp': d + e, 'rand': drand, 'all': dall[s == 'tminus'], 'other': dother[s == 'tminus']}
        if s == 'tminus' and multi:
            deltas['thumb'] = 0.05
        return stage(pn, deltas, thumb=thumb, top1=top1 if s == 'tminus' else None)

    def block(dm_, dp_):
        return {'tplus': st('tplus', dp_, p_none[0], 5 if multi else 0), 'tminus': st('tminus', dm_, p_none[1], None)}

    if not multi:
        dm = dp
    return {'pid': pid, 'status': 'ok', 'set': set_, 'stratum': stratum, 'question_type': qtype, 'multi': multi,
            'n_tiles': 2 if multi else 1, 'n_visual': nv, 'other_pid': 'd', 'other_n_visual': n_vis, 'same_input': not multi,
            'code_version': version, 'scorer_version': 's', 'cfg': {'n_t1': 3}, 'recon': {'tokens_match': True},
            'gt_ids': [10, GT1, 12], 'answer_gt': '7', 'stored': block(dm, dp), 'new': block(dm, dp),
            'gen': {'tplus': gen(kp, gp, text[0]), 'tminus': gen(km, gm, text[1])}, 'timing': {}}


def synth(n=40, n_single=0, **kw):
    R = {}
    for i in range(n):
        R[f'q{i}'] = srow(f'q{i}', i, qtype='multi_choice' if i % 3 == 0 else 'free_form', stratum=('cap', 'close', 'eos')[i % 3], **kw)
    for j in range(n_single):
        R[f's{j}'] = srow(f's{j}', j, multi=False, n_vis=256, **{k: v for k, v in kw.items() if k not in ('n_vis',)})
    return R, {'main': list(R)}


def test_known_effects_give_known_means_and_the_pass_rule():
    R, ids = synth(dm=0.10, dp=0.02)
    out = an.analyze(R, ids)
    j = out['main_judgment_new_text_multi_tile']
    assert out['n_multi'] == 40 and j['n'] == 40
    assert j['d_minus']['mean'] == 0.1 and j['delta_tplus_dpp']['mean'] == 0.02 and j['I']['mean'] == 0.08
    assert j['d_minus']['ci95'][0] > 0.09 and j['I']['ci95'] == [0.08, 0.08]
    assert j['dpp_minus_rand_tminus']['mean'] == 0.1 and j['dpp_minus_rand_tminus']['ci95'][0] > 0
    assert j['pass'] is True and j['sentence'] == an.PASS_SENTENCE
    assert an.PASS_SENTENCE == '썸네일을 뺀 입력에서 DPP reinject의 정답 확률 이득이 썸네일이 있을 때보다 컸다(탐색 결과, 정답률이 아님).'
    assert an.NOT_PASS_SENTENCE == '썸네일을 빼도 reinject의 정답 확률 이득이 커진다는 근거는 보이지 않았다.'


def test_pass_needs_both_lower_bounds():
    # no extra gain without the thumbnail: I = 0 -> lower(I) = 0, not > 0
    j = an.analyze(*synth(dm=0.10, dp=0.10))['main_judgment_new_text_multi_tile']
    assert j['d_minus']['ci95'][0] > 0 and j['I']['mean'] == 0 and j['I']['ci95'] == [0.0, 0.0]
    assert j['pass'] is False and j['sentence'] == an.NOT_PASS_SENTENCE
    # a gain only with the thumbnail: I < 0
    j = an.analyze(*synth(dm=0.0, dp=0.10))['main_judgment_new_text_multi_tile']
    assert j['pass'] is False and j['I']['ci95'][1] < 0
    # I clearly positive but d_minus not distinguishable from 0 (mean 0.001, noise +-0.01)
    j = an.analyze(*synth(dm=0.001, dp=-0.10))['main_judgment_new_text_multi_tile']
    assert j['I']['ci95'][0] > 0 and j['d_minus']['ci95'][0] <= 0 and j['pass'] is False
    # both clear
    assert an.analyze(*synth(dm=0.05, dp=-0.05))['main_judgment_new_text_multi_tile']['pass'] is True
    # the decision reads the NEW text, not the stored one
    R, ids = synth(dm=0.10, dp=0.02)
    for r in R.values():
        r['new'], r['stored'] = r['stored'], r['new']
        r['new']['tminus']['cond']['dpp'] = dict(r['new']['tminus']['cond']['none'])
        r['new']['tplus']['cond']['dpp'] = dict(r['new']['tplus']['cond']['none'])
    out = an.analyze(R, ids)
    assert out['main_judgment_new_text_multi_tile']['pass'] is False
    assert out['descriptive_not_for_decision']['stored']['core_multi']['d_minus']['mean'] == 0.1  # the stored stage is described


def test_dpp_minus_rand_and_all_minus_other_and_none_difference():
    R, ids = synth(dm=0.10, dp=0.02, drand=0.03, dall=(0.30, 0.50), dother=(0.10, 0.12), p_none=(0.2, 0.25))
    out = an.analyze(R, ids)
    assert out['main_judgment_new_text_multi_tile']['dpp_minus_rand_tminus']['mean'] == 0.07
    d = out['descriptive_not_for_decision']
    for stg in ('new', 'stored'):
        x = d[stg]
        assert x['delta_multi']['tminus:dpp']['mean'] == 0.1 and x['delta_multi']['tplus:rand']['mean'] == 0.03
        assert x['delta_multi']['tminus:thumb']['mean'] == 0.05 and 'tplus:thumb' not in x['delta_multi']
        assert x['delta_multi']['tplus:all']['mean'] == 0.3 and x['delta_multi']['tminus:other']['mean'] == 0.12
        ao = x['all_minus_other_multi']
        assert ao['tplus']['mean'] == 0.2 and ao['tminus']['mean'] == 0.38 and ao['tminus_minus_tplus']['mean'] == 0.18
        assert x['p_none_tminus_minus_tplus_multi']['mean'] == 0.05
        assert x['delta_single'] == {}


def test_splits_bins_top1_selection_and_generation_summary():
    R, ids = synth(n=40, top1={'dpp': GT1}, p_none=(0.2, 0.6))
    for i, r in enumerate(R.values()):
        r['gen']['tminus'].update(end='close' if i % 2 else 'cap', len=40 if i % 2 else 100)
    out = an.analyze(R, ids)
    x = out['descriptive_not_for_decision']['new']
    sp = x['splits_multi']
    assert {k: v['n'] for k, v in sp['stratum'].items()} == {'cap': 14, 'close': 13, 'eos': 13}  # 40 rows, stratum i % 3
    assert {k: v['n'] for k, v in sp['question_type'].items()} == {'multi_choice': 14, 'free_form': 26}
    assert all(v['d_minus']['mean'] is not None for v in sp['stratum'].values())
    b = x['dpp_by_p_none_bin_multi']
    assert b['tplus:p_none<0.5']['n'] == 40 and b['tplus:p_none>=0.5']['n'] == 0
    assert b['tminus:p_none>=0.5']['n'] == 40 and b['tminus:p_none<0.5']['n'] == 0 and b['tminus:p_none>=0.5']['mean'] == 0.1
    t = x['top1_changes_multi']
    assert t['tminus:dpp'] == {'n': 40, 'changed': 40, 'improved': 40, 'worsened': 0}
    assert t['tplus:dpp'] == {'n': 40, 'changed': 0, 'improved': 0, 'worsened': 0}
    sel = x['selection_multi']
    assert sel['mean_n_sel'] == {'tplus': 10.0, 'tminus': 10.0} and sel['tplus_thumbnail_share_of_picked'] == 0.5
    assert sel['uniform_share_sum256_over_sumN'] == round(256 / 768, 4)
    g = out['part2_generation_answers']['generated_first_step_multi']
    assert g['tminus']['end'] == {'close': 20, 'eos': 0, 'cap': 20} and g['tminus']['mean_len'] == 70.0 and g['tplus']['mean_len'] == 100.0
    # worsened: the old first token was the gold one and is not after reinsertion
    for r in R.values():
        r['new']['tminus']['cond']['none']['top1'] = GT1
        r['new']['tminus']['cond']['dpp']['top1'] = 4
    assert an.analyze(R, ids)['descriptive_not_for_decision']['new']['top1_changes_multi']['tminus:dpp']['worsened'] == 40


def test_part2_sentence_rule():
    R, ids = synth(kp=3, km=1)  # sample accuracy 100 % -> 33.3 %: -66.7 pp
    p2 = an.analyze(R, ids)['part2_generation_answers']
    assert p2['acc_t1_tminus_minus_tplus']['mean_pp'] == -66.67 and p2['acc_t1_tminus_minus_tplus']['ci95_pp'] == [-66.67, -66.67]
    assert p2['sentence'] == '썸네일을 빼면 이 답 읽기(첫 단계 뒤 바로 받은 짧은 답)의 정답률이 내려갔다(크기 66.7 pp).'
    assert p2['acc_t1_tminus_minus_tplus']['acc_tplus_pp'] == 100.0 and p2['acc_t1_tminus_minus_tplus']['acc_tminus_pp'] == 33.33
    assert p2['greedy_tminus_minus_tplus_descriptive']['mean'] == 0.0
    # up: no sentence about a drop
    p2 = an.analyze(*synth(kp=1, km=3))['part2_generation_answers']
    assert p2['sentence'] == an.PART2_NONE == '이 표본으로는 약 10 pp보다 작은 변화를 가를 수 없다.'
    # mixed: -1 on a quarter of the questions, +1 on a quarter, 0 on the rest: mean 0, upper end > 0
    R, ids = synth(kp=2, km=2)
    for i, r in enumerate(R.values()):
        r['gen']['tminus']['acc_t1'] = [1 / 3, 1.0, 2 / 3, 2 / 3][i % 4]
        r['gen']['tminus']['greedy'] = 0 if i % 4 == 0 else 1
    p2 = an.analyze(R, ids)['part2_generation_answers']
    assert p2['acc_t1_tminus_minus_tplus']['mean'] == 0.0 and p2['sentence'] == an.PART2_NONE
    assert p2['greedy_tminus_minus_tplus_descriptive']['mean'] == -0.25
    # the sentence size is |mean| with one decimal
    R, ids = synth(kp=3, km=3)
    for i, r in enumerate(R.values()):
        r['gen']['tminus']['acc_t1'] = 2 / 3 if i % 4 else 1.0
    p2 = an.analyze(R, ids)['part2_generation_answers']
    assert p2['acc_t1_tminus_minus_tplus']['mean_pp'] == -25.0
    assert p2['sentence'] == '썸네일을 빼면 이 답 읽기(첫 단계 뒤 바로 받은 짧은 답)의 정답률이 내려갔다(크기 25.0 pp).'


def test_single_tile_noise_floor_and_checks_flag():
    R, ids = synth(n=20, n_single=6, kp=2, km=1)
    for r in R.values():
        if not r['multi']:
            r['gen']['tminus'] = dict(r['gen']['tplus'])
    out = an.analyze(R, ids)
    nf = out['part2_generation_answers']['single_tile_noise_floor']
    assert nf['n_single_tile'] == 6 and nf['n_identical_generated_text'] == 6 and nf['acc_t1_tminus_minus_tplus']['mean'] == 0.0
    ck = out['checks']
    assert ck['single_tile']['n'] == 6 and ck['single_tile']['n_same_input'] == 6 and ck['single_tile']['max_abs_dlogp_gt_stored'] == 0
    assert ck['single_tile']['flag_ge_1e-4'] is False and ck['multi_tile']['n_visual_tminus_is_tplus_minus_256_for_all'] is True
    assert out['n_single'] == 6 and out['n_multi'] == 20 and out['descriptive_not_for_decision']['new']['delta_single']['tminus:dpp']['n'] == 6
    # a single-tile question whose two readings differ by 2e-4 is flagged
    R['s3']['stored']['tminus']['cond']['all']['logp_gt'] += 2e-4
    R['s4']['gen']['tminus']['text'] = 'zz'
    out = an.analyze(R, ids)
    ck = out['checks']['single_tile']
    assert ck['flag_ge_1e-4'] is True and abs(ck['max_abs_dlogp_gt_stored'] - 2e-4) < 1e-9 and ck['worst'][1:] == ['s3', 'all']
    assert out['part2_generation_answers']['single_tile_noise_floor']['n_identical_generated_text'] == 5
    R['s3']['stored']['tminus']['cond']['all']['logp_gt'] -= 1.5e-4  # 0.5e-4 < 1e-4: no flag
    assert an.analyze(R, ids)['checks']['single_tile']['flag_ge_1e-4'] is False
    # n_visual rule of the multi-tile questions
    R['q2']['n_visual']['tminus'] = 600
    ck = an.analyze(R, ids)['checks']['multi_tile']
    assert ck['n_visual_tminus_is_tplus_minus_256_for_all'] is False and ck['violations'] == ['q2']


def test_selection_of_rows_missing_error_trial_and_versions():
    R, ids = synth(n=10)
    ids['main'] = list(R) + ['gone']
    R['q3'] = {'pid': 'q3', 'status': 'error', 'stage': 'generate', 'error': 'boom', 'set': 'main', 'code_version': 'v2'}
    R['t1'] = srow('t1', 0, set_='trial')
    R['x9'] = srow('x9', 0)
    out = an.analyze(R, ids)
    c = out['counts']
    assert c['n_used'] == 9 and c['n_missing'] == 1 and c['missing'] == ['gone'] and c['n_error'] == 1 and c['error_pids'] == ['q3']
    assert c['n_trial_rows_left_out'] == 1 and c['rows_outside_main_list'] == ['x9'] and c['error_messages'] == ['boom']
    assert c['code_versions'] == ['v1', 'v2'] and c['scorer_versions'] == ['s'] and out['n_multi'] == 9
    assert out['main_judgment_new_text_multi_tile']['n'] == 9
    empty = an.analyze({}, {'main': ['a']})
    assert empty['main_judgment_new_text_multi_tile']['pass'] is None and empty['counts']['n_missing'] == 1


def test_duplicate_pids_are_refused(tmp_path):
    r = srow('1', 0)
    a, b = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'
    a.write_text(json.dumps(r) + '\n')
    b.write_text(json.dumps(r) + '\n')
    with pytest.raises(SystemExit):
        an.load([str(tmp_path / '*.jsonl')])
    b.write_text(json.dumps(srow('2', 0)) + '\n')
    R, files = an.load([str(tmp_path / '*.jsonl')])
    assert set(R) == {'1', '2'} and len(files) == 2
    with pytest.raises(SystemExit):
        an.load([str(tmp_path / 'none*.jsonl')])


def test_ref_comparison_with_the_4_11_results():
    R, ids = synth(n=8)
    ref = {}
    for i, (pid, r) in enumerate(R.items()):
        cd = {c: {'logp_gt': v['logp_gt'] + (1e-5 if c == 'dpp' else 0.0)} for c, v in r['stored']['tplus']['cond'].items()}
        ref[pid] = {'pid': pid, 'status': 'ok', 'sel': r['stored']['tplus']['sel'] if i else [99], 'part1': {'cond': cd}}
    out = an.analyze(R, ids, ref)['ref_4_11']
    assert out['n_pids_compared'] == 8 and out['n_same_sel'] == 7
    assert out['abs_dlogp_gt']['none'] == {'n': 8, 'max': 0.0, 'median': 0.0}
    assert out['abs_dlogp_gt']['dpp']['max'] == 1e-5 and set(out['abs_dlogp_gt']) == {'none', 'dpp', 'rand', 'all', 'other'}
    assert 'ref_4_11' not in an.analyze(R, ids)


def test_cli_prints_and_refuses_overwrite(tmp_path, capsys):
    R, ids = synth(n=12, n_single=2, kp=3, km=1)
    res, idf, out = tmp_path / 'r.jsonl', tmp_path / 'ids.json', tmp_path / 'o.json'
    res.write_text(''.join(json.dumps(r) + '\n' for r in R.values()))
    idf.write_text(json.dumps(ids))
    assert an.main([str(res), '--ids', str(idf), '--out', str(out)]) == 0
    printed = capsys.readouterr().out
    assert json.loads(out.read_text())['n_multi'] == 12 and an.PASS_SENTENCE in printed and '66.7' in printed
    with pytest.raises(SystemExit):
        an.main([str(res), '--ids', str(idf), '--out', str(out)])
    assert an.main([str(res), '--ids', str(idf), '--out', str(out), '--force']) == 0
    assert out.exists() and json.loads(out.read_text())['bootstrap'] == {'resamples': 20000, 'seed': 0, 'level': 0.95}


def test_bootstrap_is_the_shared_recover_check_one():
    import analyze_recover_check as rc
    assert an.Boot is rc.Boot and an.B == 20000 and rc.SEED == 0
    R, ids = synth(dm=0.10, dp=0.02)
    j = an.analyze(R, ids)['main_judgment_new_text_multi_tile']
    d = an.delta(list(R.values()), 'new', 'tminus', 'dpp')
    mean, ci = rc.Boot(40).mean_ci(d)
    assert j['d_minus']['mean'] == round(mean, 5) and j['d_minus']['ci95'] == [round(ci[0], 5), round(ci[1], 5)]


def test_analysis_runs_on_rows_written_by_the_run_script(lm, small_tile):
    torch.set_grad_enabled(False)
    try:
        eng = Engine(lm, 0.6, 50, 0.95)
        cfg = tg.resolve_cfg({**CFG, 'think_cap': 8, 'chunk': 0})
        its = {str(k): lm_item(lm, pid=str(k), n_vis=12 if k % 2 else 4, seed=k) for k in range(4)}
        rows = tg.process_group(list(its), its.__getitem__, eng, TOK, SP, cfg, tg.make_generate(eng, cfg, pack), 'v', 'main')
        assert [r['status'] for r in rows] == ['ok'] * 4
        again = tg.process_group(list(its), its.__getitem__, eng, TOK, SP, cfg, tg.make_generate(eng, cfg, pack), 'v', 'main')
        strip = lambda rs: [{k: v for k, v in r.items() if k != 'timing'} for r in rs]  # noqa: E731
        assert strip(rows) == strip(again)  # determinism (timing aside)
        R = {r['pid']: r for r in rows}
        out = an.analyze(R, {'main': list(its)})
        assert out['n_multi'] == 2 and out['n_single'] == 2 and out['counts']['n_error'] == 0
        assert out['main_judgment_new_text_multi_tile']['n'] == 2 and out['main_judgment_new_text_multi_tile']['pass'] in (True, False)
        assert out['checks']['single_tile']['max_abs_dlogp_gt_stored'] == 0.0 and out['checks']['single_tile']['n_same_input'] == 2
        assert out['checks']['multi_tile']['n_visual_tminus_is_tplus_minus_256_for_all'] is False  # the tiny tile is 4, not 256
        assert out['part2_generation_answers']['single_tile_noise_floor']['n_identical_generated_text'] == 2
        json.dumps(out)
    finally:
        torch.set_grad_enabled(True)


# ---- the Kaggle builder ----------------------------------------------------------------------------------------------------------------
def test_builder_makes_a_folder_and_refuses_bad_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, 'HERE', tmp_path)
    assert bt.main(['m1', '--owner', 'me']) == 0
    run = tmp_path / 'runs' / 'm1'
    rc = json.loads((run / 'run_config.json').read_text())
    assert rc['set'] == 'main' and rc['timeout'] == 14400 and rc['time_limit'] == 14400 - 900 and rc['group'] == 4
    assert rc['reencode_check'] is False and rc['kernel'] == 'me/visref-m1' and rc['run_args'] == ['--set', 'main', '--group', '4']
    for f in ('diag/thumb_gate.py', 'diag/step1_think.py', 'diag/inject_sensitivity.py', 'diag/thumb_gate_ids.json',
              'diag/step1_store_bmain.json', 'run.py', 'visref/engine.py', 'visref/model.py'):
        assert f in rc['source_sha256']
    nb = json.loads((run / 'notebook.ipynb').read_text())
    src = ''.join(nb['cells'][0]['source'])
    ast.parse(src)
    assert "'diag/thumb_gate.py'" in src and "'--ids', 'diag/thumb_gate_ids.json'" in src
    assert "f'thumb_gate_{i}.jsonl'" in src and "f'log_{i}.txt'" in src and "f'{i}/{n}'" in src
    cfg = json.loads(base64.b64decode(re.search(r"CFG = json.loads\(base64.b64decode\('([^']+)'\)", src).group(1)))
    assert cfg['run_args'] == rc['run_args'] and cfg['time_limit'] == 13500
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(re.search(r"b64decode\('([^']+)'\)\)\)\.extractall", src).group(1))))
    assert (REPRO / 'diag/thumb_gate.py').read_bytes() == z.read('diag/thumb_gate.py')
    assert {'diag/inject_sensitivity.py', 'diag/step1_think.py', 'diag/thumb_gate_ids.json', 'run.py'} <= set(z.namelist())
    meta = json.loads((run / 'kernel-metadata.json').read_text())
    assert meta['id'] == 'me/visref-m1' and meta['enable_gpu'] is True and meta['is_private'] is True
    with pytest.raises(SystemExit):
        bt.main(['m1', '--owner', 'me'])  # an existing run record is never overwritten
    with pytest.raises(SystemExit):
        bt.main(['m2', '--owner', 'me', '--timeout', '14399'])  # main needs at least 4 hours
    assert bt.main(['m3', '--owner', 'me', '--timeout', '20000', '--reencode-check', '--limit', '8']) == 0
    ra = json.loads((tmp_path / 'runs/m3/run_config.json').read_text())['run_args']
    assert ra == ['--set', 'main', '--group', '4', '--limit', '8', '--reencode-check']
    # the trial: short timeout allowed, the re-encoding check on by default, switchable off
    assert bt.main(['t1', '--owner', 'me', '--set', 'trial', '--timeout', '3600']) == 0
    t = json.loads((tmp_path / 'runs/t1/run_config.json').read_text())
    assert t['reencode_check'] is True and t['run_args'] == ['--set', 'trial', '--group', '4', '--reencode-check'] and t['time_limit'] == 2700
    assert bt.main(['t2', '--owner', 'me', '--set', 'trial', '--timeout', '3600', '--no-reencode-check']) == 0
    assert '--reencode-check' not in json.loads((tmp_path / 'runs/t2/run_config.json').read_text())['run_args']
    # skip-from
    sk = tmp_path / 'done.jsonl'
    sk.write_text(json.dumps({'pid': '12'}) + '\n' + json.dumps({'pid': '3'}) + '\n')
    assert bt.main(['m4', '--owner', 'me', '--skip-from', str(sk)]) == 0
    assert json.loads((tmp_path / 'runs/m4/run_config.json').read_text())['n_skip'] == 2
    assert not (REPRO / 'kaggle/runs/m1').exists()  # the repo's own runs folder is untouched


def test_no_absolute_sandbox_paths_in_the_new_files():
    bad = ('/home/' + 'claude', '/t' + 'mp')
    files = ['diag/thumb_gate_ids.py', 'diag/thumb_gate.py', 'diag/analyze_thumb_gate.py', 'kaggle/build_thumb_gate.py',
             'tests/test_thumb_gate.py', 'diag/thumb_gate_ids.json']
    for f in files:
        text = (REPRO / f).read_text()
        assert not any(b in text for b in bad), f
