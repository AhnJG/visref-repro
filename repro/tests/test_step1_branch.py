"""CPU tests for diag/step1_branch.py with the tiny Qwen3 LM and character tokenizer of test_repro."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'diag'))
from step1_branch import (CONDS, DEFAULTS, box_close_index, close_thinking, make_generate, run_group,  # noqa: E402
                          split_z)
from test_repro import CharTok, lm  # noqa: E402,F401  (lm is a pytest fixture)

from run import pack  # noqa: E402
from visref.engine import Engine  # noqa: E402

TOK = CharTok()


def sp_for(tok):
    texts = tok.vocab
    return {'think_end': tok.vocab.index('</think>'), 'eos': tok.vocab.index('<|im_end|>'),
            'probe_prompt': tok.encode('\n\n\\boxed'),
            'brace_text': {i: s for i, s in enumerate(texts) if '{' in s or '}' in s}}


def test_split_z_uses_think_tokens_to_place_the_separator():
    assert split_z(TOK, 'abc\n', 3, 'cap', '\n')[:2] == (TOK.encode('abc'), TOK.encode('\n'))
    z, sep, info = split_z(TOK, 'ab\n\n', 3, 'cap', '\n')  # text already ended with '\n': b_main added no separator?
    assert (z, sep, info['tokens_match']) == (TOK.encode('ab\n'), TOK.encode('\n'), True)
    z, sep, info = split_z(TOK, 'abc\n', 4, 'cap', '\n')  # no separator was added
    assert (z, sep, info['tokens_match']) == (TOK.encode('abc\n'), [], True)
    z, sep, info = split_z(TOK, 'abc', 3, 'close', '\n')
    assert (z, sep, info['tokens_match']) == (TOK.encode('abc'), [], True)
    assert split_z(TOK, 'abc\n', 9, 'cap', '\n')[2]['tokens_match'] is False


def test_box_close_index():
    bt = sp_for(TOK)['brace_text']
    assert box_close_index(TOK.encode('{12}'), bt) == 3
    assert box_close_index(TOK.encode('{a{b}c} x'), bt) == 6
    assert box_close_index(TOK.encode('}{7}'), bt) == 3  # a '}' before the box opens is ignored
    assert box_close_index(TOK.encode('{12'), bt) is None


def test_close_thinking():
    sp = sp_for(TOK)
    te, eos = sp['think_end'], sp['eos']
    ids = TOK.encode('ok')
    assert close_thinking(TOK, sp, ids + [te], 'stop', '\n') == (ids + [te], 'close')
    assert close_thinking(TOK, sp, ids + [eos], 'stop', '\n') == (ids + [te], 'eos')
    assert close_thinking(TOK, sp, ids, 'cap', '\n') == (ids + TOK.encode('\n') + [te], 'cap')
    nl = TOK.encode('ok\n')
    assert close_thinking(TOK, sp, nl, 'cap', '\n') == (nl + [te], 'cap')


def make_items(lm, zero_vis=False, n=3):
    emb = lm.get_input_embeddings()
    items = []
    for k in range(n):
        g = torch.Generator().manual_seed(k)
        base = emb(torch.randint(5, 90, (12,), generator=g)).detach()
        vis = torch.zeros(10, base.shape[1]) if zero_vis else torch.randn(10, base.shape[1], generator=g)
        row = {'question_type': 'free_form', 'answer_type': 'integer', 'answer': '7', 'precision': None, 'choices': None}
        items.append({'pid': str(k), 'row': row, 'sample': {'think_end': 'cap', 'n_dpp_bmain': 3}, 'base': base,
                      'vis': vis, 'z': TOK.encode('think abc'), 'sep': TOK.encode('\n'), 'recon': {}})
    return items


def cfg_small():
    return {**DEFAULTS, 'think_extra': 8, 'probe_cap': 4, 'n_t1': 2, 'chunk': 64, 'batch_tokens': 4000}


def run_once(lm, zero_vis=False):
    cfg = cfg_small()
    eng = Engine(lm, cfg['temperature'], cfg['top_k'], cfg['top_p'])
    items = make_items(lm, zero_vis)
    run_group(items, eng, TOK, sp_for(TOK), cfg, make_generate(eng, cfg, pack))
    return items, cfg


def test_run_group_shapes_and_determinism(lm):
    a, cfg = run_once(lm)
    b, _ = run_once(lm)
    for x, y in zip(a, b):
        assert set(x['answers']) == set(CONDS)
        assert all(len(x['answers'][c]) == cfg['n_t1'] + 1 for c in CONDS)
        assert sum(r['greedy'] for r in x['answers']['A']) == 1
        assert x['extra']['C']['end'] in ('close', 'eos', 'cap') and x['extra']['C']['len'] <= cfg['think_extra']
        assert x['answers'] == y['answers'] and x['extra'] == y['extra'] and x['sel'] == y['sel']


def test_zero_visual_tokens_make_B_equal_A_and_D_equal_C(lm):
    items, _ = run_once(lm, zero_vis=True)  # DPP picks nothing from an all-zero V -> identical contexts and seeds
    for it in items:
        assert it['sel'] == []
        assert it['answers']['B'] == it['answers']['A']
        assert it['answers']['D'] == it['answers']['C'] and it['extra']['D'] == it['extra']['C']
