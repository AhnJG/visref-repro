"""CPU tests for diag/recover_check_ids.py, diag/recover_check.py, diag/analyze_recover_check.py and
kaggle/build_recover_check.py (tiny random model, plan /home/claude/plan_recover.md section 6)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO / 'diag'))
sys.path.insert(0, str(REPRO / 'kaggle'))
import analyze_recover_check as an  # noqa: E402
import build_recover_check as bld  # noqa: E402
import recover_check as rc  # noqa: E402
import recover_check_ids as ids_mod  # noqa: E402
from test_repro import CharTok, lm  # noqa: E402,F401  (lm is a pytest fixture)

from visref.engine import Engine  # noqa: E402

TOK = CharTok()


def embeds(lm, n, seed):
    g = torch.Generator().manual_seed(seed)
    return lm.get_input_embeddings()(torch.randint(5, 90, (n,), generator=g)).detach()


@torch.no_grad()
def reference(lm, ctx, pos, n):
    """Ground truth: one forward of the whole sequence with an explicit all-ones 2-D attention mask (no cache), so that
    transformers 4.55.0 does not read the skipped position ids as packed sequences (masking_utils.py:717)."""
    L = ctx.shape[0]
    o = lm.get_decoder()(inputs_embeds=ctx[None], attention_mask=torch.ones(1, L, dtype=torch.long), position_ids=pos[None],
                         use_cache=False)
    return o.last_hidden_state[0, -n:]


def gappy(L, seed=0):
    """Strictly increasing position ids with gaps (like the 'o' scheme)."""
    g = torch.Generator().manual_seed(seed)
    return torch.cumsum(1 + (torch.rand(L, generator=g) < 0.3).long() * torch.randint(1, 9, (L,), generator=g), 0) - 1


@pytest.fixture
def restore_sdpa():
    from transformers.modeling_utils import AttentionInterface
    prev = AttentionInterface._global_mapping.get('sdpa')
    yield
    AttentionInterface.register('sdpa', prev)


# ---- last_hidden_pos / score_string_pos ------------------------------------------------------------------------------
def test_arange_positions_equal_engine_last_hidden(lm):
    eng = Engine(lm, attn_budget=2000)
    ctx = embeds(lm, 45, 1)
    want = eng.last_hidden(ctx, 7)
    assert torch.equal(rc.last_hidden_pos(eng, ctx, 7), want)
    assert torch.equal(rc.last_hidden_pos(eng, ctx, 7, torch.arange(45)), want)
    assert rc.last_hidden_pos(eng, ctx, 0).shape[0] == 0
    for bad in (torch.arange(44), torch.arange(45).float(), torch.arange(45)[None], list(range(45))):
        with pytest.raises(AssertionError):
            rc.last_hidden_pos(eng, ctx, 7, bad)


@pytest.mark.parametrize('L,budget', [(33, 1), (48, 1), (70, 1500)])  # chunks of 16: 16+16+1, 16*3, a larger chunk
@pytest.mark.parametrize('fastkv', [False, True])
def test_skipped_positions_chunked_equal_oneshot_with_mask(lm, restore_sdpa, monkeypatch, L, budget, fastkv):
    import visref.fastkv as fk
    from visref.fastkv import set_enabled
    one_token = []
    real = fk.gqa_attention
    monkeypatch.setattr(fk, 'gqa_attention', lambda m, q, *a, **kw: (one_token.append(q.shape[2] == 1), real(m, q, *a, **kw))[1])
    eng = Engine(lm, attn_budget=budget)
    q = max(16, budget // (lm.config.num_attention_heads * L))
    assert (L % q == 1) == (L == 33)  # the 33-token case ends with a 1-token chunk (the fast path of visref/fastkv.py)
    ctx, pos = embeds(lm, L, 2), gappy(L)
    assert int(pos[-1]) > L - 1 and bool((pos[1:] > pos[:-1]).all())
    set_enabled(fastkv)
    got = rc.last_hidden_pos(eng, ctx, 9, pos)
    set_enabled(False)
    assert any(one_token) == (fastkv and L == 33)        # the fast decode path really ran on the final 1-token chunk
    assert torch.allclose(got, reference(lm, ctx, pos, 9), atol=1e-4)
    assert not torch.allclose(got, eng.last_hidden(ctx, 9), atol=1e-3)  # the gaps matter


def test_constant_shift_of_all_positions_changes_nothing_in_float32(lm):
    eng = Engine(lm, attn_budget=2000)
    ctx, pos = embeds(lm, 40, 3), gappy(40, 1)
    a = rc.last_hidden_pos(eng, ctx, 6, pos)
    b = rc.last_hidden_pos(eng, ctx, 6, pos + 120)
    assert torch.allclose(a, b, atol=1e-4)


def test_answer_positions_continue_from_the_last_context_id(lm, monkeypatch):
    eng = Engine(lm, attn_budget=2000)
    ctx, pos = embeds(lm, 30, 4), gappy(30, 2)
    ids = TOK.encode('{7}')
    seen = []
    real = rc.last_hidden_pos
    monkeypatch.setattr(rc, 'last_hidden_pos', lambda e, c, n, p=None: (seen.append(p), real(e, c, n, p))[1])
    total, logp = rc.score_string_pos(eng, ctx, ids, pos)
    got = seen[0]
    assert got.tolist() == pos.tolist() + [int(pos[-1]) + 1 + i for i in range(len(ids))] and int(pos[-1]) > 30
    full = torch.cat([ctx, lm.get_input_embeddings()(torch.tensor(ids)).detach()])
    ref = torch.log_softmax(lm.lm_head(reference(lm, full, got, len(ids) + 1)[:-1]).float(), -1).detach()
    assert torch.allclose(logp, ref, atol=1e-4)
    assert abs(total - sum(float(ref[i, t]) for i, t in enumerate(ids))) < 1e-4
    # starting the answer at L instead (the context length) would be a different input
    wrong = torch.cat([pos, 30 + torch.arange(len(ids))])
    ref_wrong = torch.log_softmax(lm.lm_head(reference(lm, full, wrong, len(ids) + 1)[:-1]).float(), -1)
    assert not torch.allclose(logp, ref_wrong, atol=1e-3)


# ---- contexts and position ids ----------------------------------------------------------------------------------------
D = 8


def fake(n_slot=20, seed=0, prompt=7, fixed=5, tail=4):
    g = torch.Generator().manual_seed(seed)
    img_pos = torch.arange(2, 2 + n_slot)
    base = torch.randn(prompt + n_slot, D, generator=g)
    vs = [torch.randn(n_slot, D, generator=g) for _ in range(4)]
    sub = ids_mod.subset_of('7', n_slot)
    return base, img_pos, vs, sub, torch.randn(fixed, D, generator=g), torch.randn(tail, D, generator=g)


def test_positions_whole_image_subset_makes_original_spacing_equal_consecutive():
    P, n, T = 12, 15, 4
    pn, t_n = rc.make_positions(P, n, T, 'n', list(range(n)))
    po, t_o = rc.make_positions(P, n, T, 'o', list(range(n)))
    pe, t_e = rc.make_positions(P, n, T, 'e', list(range(n)))
    assert torch.equal(pn, po) and torch.equal(pn, pe) and t_n == t_o == t_e == P + n
    assert torch.equal(pn, torch.arange(P + n + T))


def test_positions_follow_the_plan_rules():
    P, T, s = 12, 3, [2, 3, 7, 20]
    n = len(s)
    po, to = rc.make_positions(P, n, T, 'o', s)
    pe, te = rc.make_positions(P, n, T, 'e', s)
    pn, tn = rc.make_positions(P, n, T, 'n', s)
    assert po[P:P + n].tolist() == [P, P + 1, P + 5, P + 18] and to == P + 18 + 1
    assert pe[P:P + n].tolist() == [P + 15, P + 16, P + 17, P + 18] and te == to       # last subset id = Co's, consecutive
    assert pn[P:P + n].tolist() == [P, P + 1, P + 2, P + 3] and tn == P + n
    assert po[-T:].tolist() == pe[-T:].tolist() == [to, to + 1, to + 2]
    assert torch.equal(po[:P], torch.arange(P)) and torch.equal(pe[:P], torch.arange(P))
    pz, tz = rc.make_positions(P, 0, T, None)
    assert torch.equal(pz, torch.arange(P + T)) and tz == P


def test_build_contexts_equalities():
    base, img_pos, (vis, blank, d1, d2), sub, fixed, tail = fake()
    c = rc.build_contexts(base, img_pos, vis, blank, d1, d2, sub, fixed, tail)
    assert list(c) == list(rc.CONDS) and len(c) == 14
    P = base.shape[0] + fixed.shape[0]
    n = len(sub)
    assert n == 6
    for x in 'noe':   # D has Cx's ids and length
        assert torch.equal(c['C' + x][1], c['D' + x][1]) and c['C' + x][0].shape == c['D' + x][0].shape
    assert torch.equal(c['Cn'][0], c['Co'][0]) and torch.equal(c['Cn'][0], c['Ce'][0])   # same embeddings, ids differ
    assert not torch.equal(c['Cn'][1], c['Co'][1])
    assert c['Co'][1][P + n - 1] == c['Ce'][1][P + n - 1] and c['Co'][2]['tail_pos0'] == c['Ce'][2]['tail_pos0']
    assert torch.equal(c['Cn'][0][P:P + n], vis[sub]) and torch.equal(c['Dn'][0][P:P + n], d2[sub])
    assert torch.equal(c['Cb'][0][base.shape[0] + fixed.shape[0]:][:20], vis)
    assert torch.equal(c['Db'][0][P:P + 20], d2) and torch.equal(c['A+'][0][P:P + 20], vis)
    # slots: A and A+ keep the own image, Bb/Cb/Db/Cn.. the gray one, B/C/D donor1's; nothing else of the prompt changes
    assert torch.equal(c['A'][0][:base.shape[0]], base) and torch.equal(c['A+'][0][:base.shape[0]], base)
    for k in ('Bb', 'Cb', 'Db', 'Cn', 'Co', 'Ce', 'Dn', 'Do', 'De'):
        assert torch.equal(c[k][0][img_pos], blank) and torch.equal(c[k][0][:2], base[:2])
    for k in ('B', 'C', 'D'):
        assert torch.equal(c[k][0][img_pos], d1)
    for k in ('A', 'Bb', 'B'):
        assert c[k][0].shape[0] == P + tail.shape[0] and torch.equal(c[k][0][-4:], tail)
    assert c['Cb'][0].shape[0] == c['Db'][0].shape[0] == c['A+'][0].shape[0] == c['C'][0].shape[0] == P + 20 + 4
    for k, (ctx, pos, info) in c.items():
        assert info['ctx_len'] == ctx.shape[0] == pos.shape[0] and info['max_pos'] == int(pos[-1])
        assert torch.equal(pos[:P], torch.arange(P))
    for k in ('A', 'Bb', 'Cb', 'Db', 'A+', 'B', 'C', 'D'):
        assert torch.equal(c[k][1], torch.arange(c[k][0].shape[0]))   # no special ids
    assert float((base - c['Bb'][0][:base.shape[0]]).abs().sum()) > 0   # the slots really were overwritten


def test_size_mismatch_raises():
    base, img_pos, (vis, blank, d1, d2), sub, fixed, tail = fake()
    for bad in ('blank', 'd1', 'd2', 'vis'):
        v = {'blank': blank, 'd1': d1, 'd2': d2, 'vis': vis}
        v[bad] = torch.randn(21, D)
        with pytest.raises(ValueError):
            rc.build_contexts(base, img_pos, v['vis'], v['blank'], v['d1'], v['d2'], sub, fixed, tail)
    with pytest.raises(ValueError):
        rc.fill_slots(base, img_pos, torch.randn(19, D))


def test_subset_rule():
    s = ids_mod.subset_of('5', 3328)
    assert s == ids_mod.subset_of('5', 3328) and s == sorted(set(s)) and len(s) == 998 and max(s) < 3328
    assert s != ids_mod.subset_of('6', 3328) and s != ids_mod.subset_of('5', 3328, seed=43)
    assert [ids_mod.subset_size(n) for n in (256, 768, 1280, 3328)] == [76, 230, 384, 998]


# ---- tiny end-to-end: the 14 conditions -------------------------------------------------------------------------------
def test_score_question_on_a_tiny_model(lm):
    eng = Engine(lm, attn_budget=3000)
    sp = {'close_ids': frozenset(i for i, s in enumerate(TOK.vocab) if s.startswith('}')),
          'brace_text': {i: s for i, s in enumerate(TOK.vocab) if '{' in s or '}' in s}}
    emb = lm.get_input_embeddings()
    g = torch.Generator().manual_seed(9)
    n_slot, prompt = 20, 7
    base = emb(torch.randint(5, 90, (prompt + n_slot,), generator=g)).detach()
    vs = [torch.randn(n_slot, lm.config.hidden_size, generator=g) for _ in range(4)]
    fixed, tail = eng.embed(TOK.encode('ab\n')), eng.embed(TOK.encode('\n\\boxed'))
    ctxs = rc.build_contexts(base, torch.arange(2, 2 + n_slot), *vs, ids_mod.subset_of('7', n_slot), fixed, tail)
    gt = TOK.encode('{7}')
    rec, kls = rc.score_question(eng, ctxs, gt, sp)
    assert list(rec) == list(rc.CONDS) and set(kls) == set(rc.KL_PAIRS) and min(kls.values()) >= 0
    for c, v in rec.items():
        assert abs(v['p_gt'] - math.exp(v['logp_gt'])) < 1e-5 and v['logp_gt'] >= v['logp_gt_full'] - 1e-4
        assert v['ctx_len'] == ctxs[c][0].shape[0] and isinstance(v['top1_is_gt_first'], bool)
    # Cn and Co hold the same embeddings, so with the whole image as the subset their reads are identical
    assert rec['Cn']['p_gt'] != rec['Co']['p_gt'] or rec['Cn']['top1'] == rec['Co']['top1']
    # the numeric check (comparison 1) agrees on a float32 tiny model; comparison 2 runs without error
    chk = rc.numeric_check(eng, ctxs, gt, sp)
    assert chk['cmp1_pass'] is True and all(v['abs_dlogp'] < 1e-3 and v['kl_first'] < 1e-5 for v in chk['cmp1'].values())
    assert all('error' not in v for v in chk['cmp2'].values()) and set(chk['cmp2']) == {'fastkv_on', 'fastkv_off'}
    rc.numeric_check(eng, ctxs, gt, sp)
    from visref.fastkv import set_enabled
    set_enabled(False)


def test_code_version_covers_the_helpers():
    v = rc.code_version()
    assert len(v) == 12 and v == rc.code_version()


# ---- ids -------------------------------------------------------------------------------------------------------------
def test_partners_and_dropped():
    nv = {'a': 2, 'b': 3, 'c': 2, 'd': 2, 'e': 3, 'f': 9, 'g': 2, 'h': 3}
    lst = list(nv)
    assert ids_mod.partners(lst, nv) == {'a': ('c', 'd'), 'b': ('e', 'h'), 'c': ('d', 'g'), 'd': ('g', 'a'), 'e': ('h', 'b'),
                                         'g': ('a', 'c'), 'h': ('b', 'e')}   # 'f' has no partner; cyclic; donor2 after donor1


def test_build_drops_questions_without_two_partners_and_picks_trial_outside():
    part1 = [str(i) for i in range(200)]
    nv = {p: 256 if int(p) % 2 else 768 for p in part1}
    nv['0'], nv['1'] = 2560, 2048    # singletons
    nv['2'], nv['4'] = 3328, 3328    # a pair: both dropped
    order = part1 + [str(i) for i in range(200, 260)]
    nv.update({p: (3328 if int(p) % 3 == 0 else 256 if int(p) % 3 == 1 else 768) for p in order[200:]})
    d = ids_mod.build(order, part1, nv)
    assert [x['pid'] for x in d['dropped']] == ['0', '1', '2', '4'] and all('reason' in x for x in d['dropped'])
    assert d['main'] == [p for p in part1 if p not in ('0', '1', '2', '4')]
    assert all(int(p) >= 200 for p in d['trial']) and len(d['trial']) == 6 and len(set(d['trial'])) == 6
    assert d['trial'][:4] == ['200', '201', '202', '203'] and nv[d['trial'][4]] == 3328 and nv[d['trial'][5]] == 256
    for p in d['main'] + d['trial']:
        q1, q2 = d['donor1'][p], d['donor2'][p]
        assert len({p, q1, q2}) == 3 and nv[q1] == nv[q2] == nv[p] and len(d['subset'][p]) == 3 * nv[p] // 10
    assert all(int(d['donor1'][p]) >= 200 and int(d['donor2'][p]) >= 200 for p in d['trial'])
    assert all(int(d['donor1'][p]) < 200 and int(d['donor2'][p]) < 200 for p in d['main'])
    with pytest.raises(AssertionError):
        ids_mod.build(order[1:], part1, nv)


def test_ids_file_is_current_and_follows_the_rules():
    assert ids_mod.main(['--check']) == 0
    d = json.loads(ids_mod.OUT.read_text())
    part1 = json.loads((REPRO / 'diag/inject_sensitivity_ids.json').read_text())['part1']
    order = json.loads(ids_mod.ORDER.read_text())
    assert order[:200] == part1 and len(d['main']) + len(d['dropped']) == 200 and len(d['trial']) == 6
    assert [p for p in part1 if p in set(d['main'])] == d['main']              # same order as part1
    assert set(d['main']) | {x['pid'] for x in d['dropped']} == set(part1) and not set(d['trial']) & set(part1)
    nv = ids_mod.n_visual_of()
    assert [x['pid'] for x in d['dropped']] == ['353'] and d['dropped'][0]['n_visual'] == 2560
    pos = {p: i for i, p in enumerate(order)}
    assert sorted(pos[p] for p in d['trial'][:4]) == list(range(200, 204))
    for p in d['main'] + d['trial']:
        assert d['n_visual'][p] == nv[p] and len({p, d['donor1'][p], d['donor2'][p]}) == 3
        assert nv[d['donor1'][p]] == nv[d['donor2'][p]] == nv[p]
        assert d['subset'][p] == ids_mod.subset_of(p, nv[p])
    assert all(pos[d['donor1'][p]] >= 200 and pos[d['donor2'][p]] >= 200 for p in d['trial'])


def test_ids_check_fails_on_a_changed_file(tmp_path, monkeypatch):
    f = tmp_path / 'ids.json'
    f.write_text(ids_mod.OUT.read_text().replace('"main"', '"mainX"'))
    monkeypatch.setattr(ids_mod, 'OUT', f)
    assert ids_mod.main(['--check']) == 1


# ---- analysis ---------------------------------------------------------------------------------------------------------
def make_row(pid, ps, qtype='free_form', status='ok', rset='main'):
    cond = {c: {'p_gt': ps[c], 'logp_gt': math.log(max(ps[c], 1e-6)), 'top1_is_gt_first': ps[c] > 0.5, 'ctx_len': 100,
                'entropy_first': 1.0} for c in an.CONDS}
    return {'pid': pid, 'status': status, 'set': rset, 'code_version': 'x', 'question_type': qtype, 'cond': cond,
            'kl_pairs': {k: 0.01 for k in an.PAIRS}}


def rows_for(ps_fn, n=60):
    return [make_row(str(i), ps_fn(i), 'multi_choice' if i % 2 else 'free_form') for i in range(n)]


def base_ps(i, a=0.8, b=0.1, cb=0.1, db=0.1, aplus=0.8, cn=0.1, co=0.1, ce=0.1, dn=0.1, do=0.1, de=0.1):
    j = 0.002 * (i % 5)    # a little spread between questions
    v = {'A': a + j, 'Bb': b + j, 'Cb': cb + j, 'Db': db + j, 'A+': aplus + j, 'Cn': cn + j, 'Co': co + j, 'Ce': ce + j,
         'Dn': dn + j, 'Do': do + j, 'De': de + j, 'B': b + j, 'C': cb + j, 'D': db + j}
    return {k: min(max(x, 0.0), 1.0) for k, x in v.items()}


def test_analysis_no_information_stops_at_step_1():
    r = an.analyze(rows_for(lambda i: base_ps(i, a=0.12)))      # AB = 0.02 < 0.05
    assert r['step1']['info_present'] is False and r['verdict'] == an.V1_NO_INFO and r['stopped_after_step1']
    assert 'step2' not in r and 'step3' not in r and 'described' not in r
    r = an.analyze(rows_for(lambda i: base_ps(i, a=0.5, b=0.5)))   # AB = 0 : lower end not above 0
    assert r['verdict'] == an.V1_NO_INFO


def test_analysis_channel_works_and_step3_unclear():
    r = an.analyze(rows_for(lambda i: base_ps(i, cb=0.6, db=0.15, aplus=0.85, cn=0.3, co=0.3, ce=0.3, dn=0.1, do=0.1, de=0.1)))
    assert r['step1']['label'] == '정보 있음' and r['step1']['AB']['mean'] == pytest.approx(0.7, abs=1e-4)
    assert r['step2']['key'] == 'works' and r['step2']['label'] == an.V2['works']
    assert r['step3']['key'] == 'unclear' and r['step3']['label'] == an.V3['unclear'] and r['step3']['noise_note'] == an.V3_NOISE
    assert r['next_actions'] == [an.NEXT['works_no_pos_diff']]
    d = r['described']
    assert d['CB / AB']['ratio'] == pytest.approx(0.5 / 0.7, abs=1e-3) and d['DB']['mean'] == pytest.approx(0.05, abs=1e-4)
    assert d['A+ - A']['mean'] == pytest.approx(0.05, abs=1e-4) and d['S_n']['mean'] == pytest.approx(0.2, abs=1e-4)
    assert r['split']['multi_choice']['n'] == r['split']['free_form']['n'] == 30
    assert r['counts']['n_used'] == 60 and r['counts']['n_error'] == 0 and r['means_p_gt']['A'] == pytest.approx(0.804, abs=1e-3)


def test_analysis_barely_transfers_wins_over_works_by_precedence():
    # Cb is above Bb and Db with a lower end > 0 (so 'works' would hold), but CB < 0.25 * AB: rule 1 comes first
    r = an.analyze(rows_for(lambda i: base_ps(i, a=0.8, b=0.1, cb=0.15, db=0.1)))
    assert r['step2']['CB']['ci95'][0] > 0 and r['step2']['CD']['ci95'][0] > 0
    assert r['step2']['CB']['ci95'][1] < 0.25 * r['step1']['AB']['mean']
    assert r['step2']['key'] == 'barely' and r['step2']['label'] == an.V2['barely']
    assert an.NEXT['barely'] in r['next_actions']


def test_analysis_undecided_and_the_disturbance_note():
    # CB averages 0.02 but varies +0.22 / -0.18 (range contains 0, upper end above 0.25 * AB = 0.0625), Db far below
    # Bb: CD > 0 only -> neither rule 1 nor rule 2 holds; the note applies
    def ps(i):
        return base_ps(i, a=0.65, b=0.4, cb=0.4 + (0.22 if i % 2 else -0.18), db=0.1, aplus=0.65)
    r = an.analyze(rows_for(ps, 40))
    assert r['step1']['info_present'] is True
    assert r['step2']['CB']['ci95'][0] < 0 < r['step2']['CB']['ci95'][1] and r['step2']['CD']['ci95'][0] > 0
    assert r['step2']['key'] == 'undecided' and r['step2']['note'] == an.V2_NOTE and r['next_actions'] == []


def test_analysis_step3_branches():
    more = an.analyze(rows_for(lambda i: base_ps(i, cb=0.6, db=0.15, cn=0.3, co=0.5, ce=0.3, dn=0.1, do=0.1, de=0.1)))
    assert more['step3']['key'] == 'more' and more['step3']['label'] == an.V3['more']
    assert an.NEXT['pos_gain'] in more['next_actions'] and an.NEXT['works_no_pos_diff'] not in more['next_actions']
    assert more['described']['S_o - S_e']['mean'] == pytest.approx(0.2, abs=1e-4)
    assert more['described']['S_e - S_n']['mean'] == pytest.approx(0.0, abs=1e-4)
    less = an.analyze(rows_for(lambda i: base_ps(i, cb=0.6, db=0.15, cn=0.5, co=0.3, ce=0.4, dn=0.1, do=0.1, de=0.1)))
    assert less['step3']['key'] == 'less' and less['step3']['label'] == an.V3['less']


def test_analysis_excludes_trial_error_incomplete_and_counts_them():
    ids = {'main': [str(i) for i in range(10)], 'trial': ['t1', 't2']}
    ps = base_ps(0, cb=0.6, db=0.15)
    rows = [make_row(str(i), ps) for i in range(8)]
    rows.append({'pid': '8', 'status': 'error', 'error': 'x', 'code_version': 'x'})
    rows.append(make_row('t1', base_ps(0, a=0.0, b=0.9)))                      # listed trial question
    rows.append(make_row('t2', base_ps(0, a=0.0, b=0.9), rset='trial'))
    rows.append(make_row('99', ps))                                            # not in the main list
    bad = make_row('9', ps)
    del bad['cond']['Do']
    rows.append(bad)
    r = an.analyze(rows, ids)
    assert r['counts'] == {'n_rows': 13, 'n_trial_excluded': 2, 'n_error': 1, 'n_outside_main_list': 1, 'n_incomplete': 1,
                           'n_used': 8, 'n_main_expected': 10, 'n_main_missing': 2}
    assert r['means_p_gt']['A'] == pytest.approx(0.8, abs=1e-4)                # the trial rows (A = 0) are not in it
    assert an.analyze([], ids)['verdict'] is None


def test_bootstrap_is_exact_integer_arithmetic_and_paired():
    bt = an.Boot(5, b=2000)
    d = np.array([100000, 200000, 10000, 300000, 50000])
    m, ci = bt.mean_ci(d)
    assert m == pytest.approx(660000 / 5e6) and ci[0] <= m <= ci[1]
    assert bt.sums(d).dtype == np.int64 and np.array_equal(bt.sums(d), bt.sums(d))
    assert np.array_equal(bt.sums(d) + bt.sums(d), bt.sums(2 * d))             # one shared resample matrix
    o, rci, bad = bt.ratio_ci(d, 2 * d)
    assert o == 0.5 and rci == [0.5, 0.5] and bad == 0
    assert an.Boot(7).idx.tolist() == an.Boot(7).idx.tolist()                  # seed 0 every time


# ---- builder ----------------------------------------------------------------------------------------------------------
def test_builder_modes_and_refusal(tmp_path):
    here = tmp_path / 'kaggle'
    here.mkdir()
    for mode in ('trial', 'main'):
        assert bld.main([f'rc_{mode}', '--mode', mode, '--owner', 'someone'], here=here) == 0
        run = here / 'runs' / f'rc_{mode}'
        cfg = json.loads((run / 'run_config.json').read_text())
        assert cfg['mode'] == cfg['set'] == mode and cfg['timeout'] == 10800 and cfg['time_limit'] == 9900
        assert cfg['kernel'] == f'someone/visref-rc-{mode}'
        nb = json.loads((run / 'notebook.ipynb').read_text())
        src = ''.join(nb['cells'][0]['source'])
        compile(src, 'cell', 'exec')                                           # the single cell is valid Python
        assert 'diag/recover_check.py' in src and "CFG['mode'] == 'trial'" in src and '--check' in src
        assert {'diag/recover_check.py', 'diag/recover_check_ids.json', 'run.py', 'visref/engine.py'} <= set(cfg['source_sha256'])
    with pytest.raises(SystemExit):
        bld.main(['rc_main', '--owner', 'someone'], here=here)                  # an existing run record is never overwritten
    assert bld.main(['rc_main', '--owner', 'someone', '--overwrite', '--timeout', '7200'], here=here) == 0


# ---- the run loop with a stubbed model ----------------------------------------------------------------------------------
def test_main_runs_end_to_end_with_stubs(lm, restore_sdpa, monkeypatch, tmp_path):
    import types

    from PIL import Image

    import run as run_mod
    import visref.model as vm
    n_by_w = {100: 10, 200: 30}
    pids = ['p0', 'p1', 'p2', 'p3', 'q0']          # p*: 10 tokens (4 of them), q0 has 30 tokens and no partner
    width = {p: 200 if p == 'q0' else 100 for p in pids}
    nv = {p: n_by_w[width[p]] for p in pids}
    ids_doc = {'main': ['p0', 'p1', 'p2', 'q0'], 'trial': ['p3'], 'n_visual': nv,
               'donor1': {'p0': 'p1', 'p1': 'p2', 'p2': 'p3', 'p3': 'p0', 'q0': 'p0'},
               'donor2': {'p0': 'p2', 'p1': 'p3', 'p2': 'p0', 'p3': 'p1', 'q0': 'p1'},
               'subset': {p: ids_mod.subset_of(p, nv[p]) for p in pids}}
    (tmp_path / 'ids.json').write_text(json.dumps(ids_doc))
    rows = {p: {'pid': p, 'query': 'q ' + p, 'answer': '7', 'question_type': 'free_form', 'choices': None,
                'image': (lambda p=p: Image.new('RGB', (width[p], 40), (10 * int(p[1]) + 5, 20, 30)))} for p in pids}
    d = lm.config.hidden_size
    emb = lm.get_input_embeddings()

    def encode_sample(model, tok, image, question, max_num=12, with_positions=False):
        n = n_by_w[image.size[0]]
        g = torch.Generator().manual_seed(sum(image.getpixel((0, 0))) * 7 + n)
        vis = torch.randn(n, d, generator=g)
        ids = torch.randint(5, 90, (6,), generator=torch.Generator().manual_seed(len(question)))
        base = torch.cat([emb(ids[:3]).detach(), torch.zeros(n, d), emb(ids[3:]).detach()])
        pos = torch.arange(3, 3 + n)
        base[pos] = vis
        return (base, vis, pos) if with_positions else (base, vis)

    sp = {'think_end': TOK.vocab.index('</think>'), 'eos': 0, 'probe_prompt': TOK.encode('\n\n\\boxed'),
          'close_ids': frozenset(i for i, s in enumerate(TOK.vocab) if s.startswith('}')),
          'brace_text': {i: s for i, s in enumerate(TOK.vocab) if '{' in s or '}' in s}}
    monkeypatch.setattr(vm, 'load_mathvista', lambda path: rows)
    monkeypatch.setattr(vm, 'load_model', lambda *a, **k: (types.SimpleNamespace(language_model=lm), TOK))
    monkeypatch.setattr(vm, 'encode_sample', encode_sample)
    monkeypatch.setattr(vm, 'special_ids', lambda *a, **k: sp)
    monkeypatch.setattr(run_mod, 'fast_kv_selfcheck', lambda lm_, tok_: (True, {}))
    out = tmp_path / 'main.jsonl'
    argv = ['--parquet', 'x', '--ids', str(tmp_path / 'ids.json'), '--out', str(out), '--group', '2']
    monkeypatch.setattr(sys, 'argv', ['recover_check.py'] + argv)
    rc.main()
    got = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r['pid'] for r in got] == ['p0', 'p1', 'p2', 'q0'] and [r['status'] for r in got] == ['ok', 'ok', 'ok', 'error']
    r = got[0]
    assert r['set'] == 'main' and r['n_visual'] == 10 and r['donor1'] == 'p1' and r['donor2'] == 'p2' and r['subset'] == ids_doc['subset']['p0']
    assert list(r['cond']) == list(rc.CONDS) and set(r['kl_pairs']) == set(rc.KL_PAIRS) and r['gt_ids'] == TOK.encode('{7}')
    assert r['fixed_ids'] == TOK.encode(rc.THINK_OPEN + rc.SENTENCE) and r['code_version'] == rc.code_version()
    c = r['cond']
    assert c['Co']['max_pos'] >= c['Cn']['max_pos'] and c['Co']['max_pos'] == c['Ce']['max_pos']
    assert c['Cb']['ctx_len'] == c['Db']['ctx_len'] == c['A+']['ctx_len'] and c['Cb']['ctx_len'] - c['A']['ctx_len'] == 10
    assert c['Cn']['n_ins'] == 3 and c['Cn']['ctx_len'] - c['A']['ctx_len'] == 3 and c['A']['max_pos'] == c['A']['ctx_len'] - 1
    assert c['Cn']['tail_pos0'] == c['Cn']['ctx_len'] - 1 - len(sp['probe_prompt'])
    assert got[3]['error'].startswith('ValueError: donor1 image has 10 tokens, the own image 30')   # q0 vs 10-token donors
    # analysis ignores error rows; the same rows resume without being repeated
    rc.main()
    assert len(out.read_text().splitlines()) == 4
    # trial set + numeric check mode
    out2, out3 = tmp_path / 'trial.jsonl', tmp_path / 'check.jsonl'
    for extra, o in (([], out2), (['--check'], out3)):
        monkeypatch.setattr(sys, 'argv', ['recover_check.py', '--parquet', 'x', '--ids', str(tmp_path / 'ids.json'), '--out', str(o),
                                          '--set', 'trial'] + extra)
        rc.main()
    t = [json.loads(l) for l in out2.read_text().splitlines()]
    k = [json.loads(l) for l in out3.read_text().splitlines()]
    assert t[0]['set'] == 'trial' and t[0]['status'] == 'ok' and k[0]['status'] == 'check' and k[0]['cmp1_pass'] is True
    res = an.analyze(got + t, ids_doc)
    assert res['counts']['n_trial_excluded'] == 1 and res['counts']['n_error'] == 1 and res['counts']['n_used'] == 3
