"""CPU tests for diag/inject_sensitivity_ids.py and diag/inject_sensitivity.py (fake engine / tiny LM)."""
import json
import math
import sys
from pathlib import Path

import torch

REPRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO / 'diag'))
import inject_sensitivity as isn  # noqa: E402
import inject_sensitivity_ids as ids_mod  # noqa: E402
from test_repro import lm  # noqa: E402,F401  (pytest fixture)
from test_step1_think import SP as SP0, TOK, FakeEng, item  # noqa: E402

from visref.dpp import select_tokens  # noqa: E402
from visref.engine import Engine  # noqa: E402

SP = {**SP0, 'close_ids': frozenset(i for i, s in enumerate(TOK.vocab) if s.startswith('}'))}
CFG = {**isn.DEFAULTS, 'probe_cap': 4, 'n_msh': 3, 'chunk': 64, 'batch_tokens': 4000}


def test_ids_file_is_current_and_parts_follow_the_rule():
    assert ids_mod.main(['--check']) == 0
    d = json.loads(ids_mod.OUT.read_text())
    assert d['part1'] == json.loads((REPRO / 'diag/format_check_ids.json').read_text())
    assert len(d['part2']) == 100 and all(d['stratum'][p] == 'close' for p in d['part2'])
    order = [str(p) for p in json.loads(ids_mod.ORDER.read_text())]
    pos = {p: i for i, p in enumerate(order)}
    assert [pos[p] for p in d['part2']] == sorted(pos[p] for p in d['part2'])
    assert set(d['tsr_probe']) == set(d['part1'])


def test_gt_text():
    assert isn.gt_text({'question_type': 'multi_choice', 'choices': ['x', 'y', 'z'], 'answer': 'y'}) == 'B'
    assert isn.gt_text({'question_type': 'free_form', 'choices': None, 'answer': '1.5'}) == '1.5'


def test_answer_entropy():
    assert isn.answer_entropy(['A'] * 8) == 0
    assert abs(isn.answer_entropy(['A'] * 7 + [' B']) - 0.3768) < 1e-3
    assert abs(isn.answer_entropy(['A', 'A ', 'B', 'B']) - math.log(2)) < 1e-9


def test_contexts_order():
    eng = FakeEng()
    it = item(eng)
    it['vis_other'] = torch.randn(4, eng.emb.embedding_dim)
    c = isn.contexts(it, eng, TOK, SP, CFG)
    e = eng.embed
    prefix = torch.cat([it['base'], e(it['z'] + it['sep'])])
    tail = e([SP['think_end']] + SP['probe_prompt'])
    sel = select_tokens(it['vis'], e(it['z']), 3)
    assert it['sel'] == sel and len(sel) > 0 and len(it['sel_rand']) == len(sel)
    assert torch.equal(c['none'], torch.cat([prefix, tail]))
    assert torch.equal(c['dpp'], torch.cat([prefix, it['vis'][sel], tail]))
    assert torch.equal(c['rand'], torch.cat([prefix, it['vis'][it['sel_rand']], tail]))
    assert torch.equal(c['all'], torch.cat([prefix, it['vis'], tail]))
    other = torch.cat([it['vis_other']] * 3)[:10]  # 4 tokens repeated to |V| = 10
    assert torch.equal(c['other'], torch.cat([prefix, other, tail])) and c['other'].shape == c['all'].shape
    del it['vis_other']
    assert 'other' not in isn.contexts(it, eng, TOK, SP, CFG)  # part-2-only questions have no control
    tag = torch.cat([e([SP['img_start']]), it['vis'][sel], e([SP['img_end']] + TOK.encode('\n'))])
    assert torch.equal(c['dpp_tag'], torch.cat([prefix, tag, tail]))  # the cut step's sep already ends with '\n'


def test_score_string_matches_a_full_forward(lm):
    eng = Engine(lm)
    eng.lm = lm
    g = torch.Generator().manual_seed(1)
    ctx = lm.get_input_embeddings()(torch.randint(5, 90, (12,), generator=g)).detach()
    ids = TOK.encode('{7}')
    total, logp = isn.score_string(eng, ctx, ids)
    full = torch.cat([ctx, lm.get_input_embeddings()(torch.tensor(ids)).detach()])
    ref = torch.log_softmax(lm(inputs_embeds=full[None]).logits[0].float(), -1)
    want = sum(float(ref[len(ctx) - 1 + i, t]) for i, t in enumerate(ids))
    assert abs(total - want) < 1e-4
    assert torch.allclose(logp, ref[len(ctx) - 1:len(ctx) - 1 + len(ids)], atol=1e-4)


def test_kl_zero_for_same_distribution_and_positive_otherwise():
    p = torch.log_softmax(torch.tensor([1.0, 2.0, 0.5]), -1)
    q = torch.log_softmax(torch.tensor([0.0, 2.0, 1.5]), -1)
    assert isn.kl(p, p) == 0 and isn.kl(p, q) > 0


def test_part1_and_part2_run_on_a_tiny_model(lm):
    from run import pack
    from step1_think import make_generate
    eng = Engine(lm)
    torch.set_grad_enabled(False)
    try:
        g = torch.Generator().manual_seed(0)
        it = {'pid': '5', 'base': eng.embed(torch.randint(5, 90, (6,), generator=g).tolist()),
              'vis': torch.randn(10, lm.config.hidden_size, generator=g), 'z': TOK.encode('think abc'),
              'sep': TOK.encode('\n'), 'sep_text': '\n',
              'row': {'question_type': 'free_form', 'answer_type': 'integer', 'answer': '7', 'precision': None, 'choices': None}}
        it['vis_other'], it['other_pid'] = torch.randn(3, lm.config.hidden_size, generator=g), '9'
        p1 = isn.part1(it, eng, TOK, SP, CFG, {'ids': TOK.encode('{8}'), 'valid': True})
        assert set(p1['cond']) == set(isn.CONDS1) and p1['cond']['none']['kl_from_none'] == 0
        assert p1['tsr_brace_first'] and p1['other_pid'] == '9' and p1['question_type'] == 'free_form'
        n = it['base'].shape[0] + len(it['z']) + 1 + 1 + len(SP['probe_prompt'])
        assert p1['cond']['none']['ctx_len'] == n and p1['cond']['all']['ctx_len'] == n + 10
        assert p1['gt_ids'] == TOK.encode('{7}') and p1['first_pos'] == 1
        isn.part2([it], eng, TOK, SP, CFG, make_generate(eng, CFG, pack))
        assert set(it['msh']) == set(isn.CONDS2) and all(len(v) == CFG['n_msh'] for v in it['msh'].values())
        for v in it['msh'].values():
            assert all((a['H'] is None) == (not a['valid']) for a in v)
    finally:
        torch.set_grad_enabled(True)


def _fake_row(pid, stratum, shift, sel_extra=0.0, ent=0.0):
    cond = {'none': {'logp_gt': -2.0, 'logp_tsr': -0.5, 'kl_from_none': 0.0, 'top1': 1}}
    for c, k in (('dpp', shift + sel_extra), ('rand', shift), ('all', 2 * shift), ('other', 0.0), ('dpp_tag', shift + sel_extra)):
        cond[c] = {'logp_gt': -2.0 + k, 'logp_tsr': -0.5 - k, 'kl_from_none': k, 'top1': 2 if k > 0.5 else 1}
    return {'pid': pid, 'stratum': stratum, 'code_version': 'x', 'part1': {'cond': cond, 'question_type': 'free_form', 'tsr_valid': True},
            'part2': {c: {'H_mean': ent, 'entropy': 0.0, 'n_valid': 8} for c in ('none', 'dpp')}}


def test_analysis_readings():
    import analyze_inject_sensitivity as an
    flat = [_fake_row(str(i), 'cap' if i % 2 else 'close', 0.0) for i in range(40)]
    r = an.analyze(flat, {})
    assert r['part1']['all']['moves'] is False and r['part1']['all']['selection'] is False
    assert r['part2']['continue'] is False and r['part2']['none']['above_delta'] == 0
    moved = [_fake_row(str(i), 'cap', 0.4 + 0.01 * (i % 3), sel_extra=0.2, ent=0.3 if i < 10 else 0.0) for i in range(40)]
    r = an.analyze(moved, {(str(i), 'tsr'): 0.3 for i in range(5)})
    a = r['part1']['all']
    assert a['moves'] is True and a['all']['top1_changed_share'] == 1.0 and a['selection'] is True
    assert a['dpp-rand']['mean_logp_gt_diff'] == 0.2 and r['part1']['close']['n'] == 0 and r['part1']['free_form']['n'] == 40
    assert r['part2']['none']['share'] == 0.25 and r['part2']['continue'] is True
    assert r['part2']['stored_single_tsr']['above_delta'] == 5


def test_moves_needs_the_length_matched_control():
    import analyze_inject_sensitivity as an
    rows = [_fake_row(str(i), 'cap', 0.4 + 0.01 * (i % 3)) for i in range(40)]
    for r in rows:  # the control moves log p(right answer) exactly as much as all image tokens: no content effect
        r['part1']['cond']['other'] = dict(r['part1']['cond']['all'])
    a = an.analyze(rows, {})['part1']['all']
    assert a['all']['top1_changed_share'] == 1.0 and a['moves'] is False


def test_box_depths_and_sample_H():
    from types import SimpleNamespace
    ids = TOK.encode('{7}')
    assert isn.box_depths(ids, SP['brace_text']) == [0, 1, 1]
    r = SimpleNamespace(ids=ids, ent=[0.5, 0.2, 0.3], ent_lump=[0.4, 0.1, 0.05], finish='stop')
    assert abs(isn.sample_H(r, {'valid': True}, SP) - (0.5 + 0.1 + 0.05)) < 1e-9  # merged entropy inside the box
    assert isn.sample_H(r, {'valid': False}, SP) == float('inf')
    assert isn.mean_finite([0.1, None, 0.3]) == 0.2 and isn.mean_finite([None]) is None
