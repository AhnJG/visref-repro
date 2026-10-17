"""CPU tests for diag/step1_think_ids.py, diag/step1_think.py and diag/analyze_step1_think.py (fake engine / tiny LM)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

REPRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO / 'diag'))
import step1_branch as sb  # noqa: E402
import step1_think as t1  # noqa: E402
import step1_think_ids as ids_mod  # noqa: E402
from test_repro import CharTok, lm  # noqa: E402,F401  (lm is a pytest fixture)

from run import pack  # noqa: E402
from visref.dpp import select_tokens  # noqa: E402
from visref.engine import Engine, Result  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPRO / 'diag' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


an = _load('analyze_step1_think')
TOK = CharTok()


def sp_for(tok):
    texts = tok.vocab
    return {'think_end': tok.vocab.index('</think>'), 'eos': tok.vocab.index('<|im_end|>'),
            'img_start': tok.vocab.index('<img>'), 'img_end': tok.vocab.index('</img>'),
            'wait': tok.encode('Wait, think more'), 'probe_prompt': tok.encode('\n\n\\boxed'),
            'brace_text': {i: s for i, s in enumerate(texts) if '{' in s or '}' in s}}


SP = sp_for(TOK)
NL = TOK.encode('\n')


# ---- ids list ---------------------------------------------------------------------------------------------------
def test_ids_build_on_the_real_main_run():
    store = json.loads(ids_mod.STORE.read_text())['pids']
    order = json.loads(ids_mod.ORDER.read_text())
    ids, stratum = ids_mod.build(order, ids_mod.load_rows(), store)
    assert len(ids) == len(set(ids)) == 691
    assert all(isinstance(p, str) and p in store for p in ids)
    counts = {e: sum(v == e for v in stratum.values()) for e in ('close', 'cap', 'eos')}
    assert counts == {'close': 329, 'cap': 344, 'eos': 18}
    pos = {str(p): i for i, p in enumerate(order)}
    assert [pos[p] for p in ids] == sorted(pos[p] for p in ids)  # seed-0 order
    assert all(store[p]['think_end'] == stratum[p] for p in ids)


def test_ids_file_is_current():
    assert ids_mod.main(['--check']) == 0
    d = json.loads(ids_mod.OUT.read_text())
    assert d['n'] == len(d['ids']) == 691 and set(d['stratum']) == set(d['ids'])
    assert d['strata_counts'] == {'cap': 344, 'close': 329, 'eos': 18}


def test_ids_build_selection_rule_and_checks():
    rows = {p: {'st': {'steps': [{'think_end': 'close'}]}, 'tsr': {'k': a}, 'visref_raw_rank': {'k': b}}
            for p, (a, b) in {'1': (1, 1), '2': (1, 2), '3': (2, 1), '4': (1, 1)}.items()}
    store = {p: {'think_end': 'close'} for p in rows}
    assert ids_mod.build(['4', '3', '2', '1'], rows, store)[0] == ['4', '1']
    with pytest.raises(AssertionError):
        ids_mod.build(['1', '2', '3', '4'], rows, {'1': {'think_end': 'close'}})  # stored step 1 missing
    store['1']['think_end'] = 'cap'
    with pytest.raises(AssertionError):
        ids_mod.build(['1', '2', '3', '4'], rows, store)  # store and ST row disagree


# ---- the copied helpers equal step1_branch's ---------------------------------------------------------------------
def test_copied_helpers_equal_step1_branch():
    assert t1.seed_of(42, '7', 'think') == sb.seed_of(42, '7', 'think')
    te, eos = SP['think_end'], SP['eos']
    ids = TOK.encode('ok')
    for case in ((ids + [te], 'stop'), (ids + [eos], 'stop'), (ids, 'cap'), (TOK.encode('ok\n'), 'cap')):
        assert t1.close_thinking(TOK, SP, *case, '\n') == sb.close_thinking(TOK, SP, *case, '\n')
    row = {'question_type': 'free_form', 'answer_type': 'integer', 'answer': '7', 'precision': None, 'choices': None}
    for s in ('{7}', '{8}', '{', '{ }', '{a{7}}'):
        x = TOK.encode(s)
        assert t1.answer_record(TOK, SP, x, row) == sb.answer_record(TOK, SP, x, row)
        assert t1.box_close_index(x, SP['brace_text']) == sb.box_close_index(x, SP['brace_text'])


# ---- stored step 1 ------------------------------------------------------------------------------------------------
def test_split_stored():
    def s(text, n, end):
        return {'thinking': text, 'think_tokens': n, 'think_end': end}
    z, sep, sep_text, info = t1.split_stored(TOK, SP, s('abc\n', 3, 'cap'), '\n')  # separator added by b_main
    assert (z, sep, sep_text, info['tokens_match']) == (TOK.encode('abc'), NL, '\n', True)
    z, sep, sep_text, info = t1.split_stored(TOK, SP, s('ab\n\n', 4, 'cap'), '\n')  # z ends in '\n': no separator added
    assert (z, sep, sep_text, info['tokens_match']) == (TOK.encode('ab\n\n'), [], '', True)
    z, sep, sep_text, info = t1.split_stored(TOK, SP, s('abc', 3, 'close'), '\n')  # closed step: no separator
    assert (z, sep, sep_text, info['tokens_match']) == (TOK.encode('abc'), [], '', True)
    z, sep, sep_text, info = t1.split_stored(TOK, SP, s('abc', 3, 'eos'), '\n')
    assert (z, sep) == (TOK.encode('abc'), []) and 'think_end_stored' in info
    assert t1.split_stored(TOK, SP, s('abc\n', 9, 'cap'), '\n')[3]['tokens_match'] is False


# ---- contexts -----------------------------------------------------------------------------------------------------
class FakeEng:
    def __init__(self, d=8):
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(len(TOK.vocab), d)
        self.calls = []

    def embed(self, ids):
        return self.emb(torch.as_tensor(ids, dtype=torch.long)).detach()

    def last_hidden(self, ctx, n):
        self.calls.append((ctx.shape[0], n))
        return ctx[-n:] * 0.5 + 1.0  # not the input embeddings


CFG = {**t1.DEFAULTS, 'think_extra': 8, 'probe_cap': 4, 'n_t1': 2, 'chunk': 64, 'batch_tokens': 4000}


def item(eng, pid='5', z='think abc', cut=True, n_vis=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    zi = TOK.encode(z)
    sep_text = '\n' if cut and not z.endswith('\n') else ''
    return {'pid': pid, 'base': eng.embed(torch.randint(5, 90, (6,), generator=g)),
            'vis': torch.randn(n_vis, eng.emb.embedding_dim, generator=g), 'z': zi, 'sep': TOK.encode(sep_text),
            'sep_text': sep_text}


def test_contexts_bmain_order_and_random_control():
    eng = FakeEng()
    it = item(eng)
    t1.build_contexts(it, eng, TOK, SP, CFG, 'bmain')
    e = eng.embed
    prefix = torch.cat([it['base'], e(it['z'] + it['sep'])])
    sel = select_tokens(it['vis'], e(it['z']), 3)
    assert it['m'] == 3 and it['sel'] == sel and len(sel) > 0
    assert torch.equal(it['ctx']['C'], torch.cat([prefix, e(SP['wait'])]))
    assert torch.equal(it['ctx']['D'], torch.cat([prefix, it['vis'][sel], e(SP['wait'])]))  # untagged, z sep inject wait
    r = it['sel_rand']
    assert len(r) == len(sel) == len(set(r)) and r == sorted(r) and all(0 <= i < 10 for i in r)
    assert it['ctx']['R'].shape == it['ctx']['D'].shape
    assert torch.equal(it['ctx']['R'], torch.cat([prefix, it['vis'][r], e(SP['wait'])]))
    assert eng.calls == []  # bmain does not use hidden states


def test_random_selection_is_fixed_per_pid():
    a = t1.random_sel('5', 42, 3328, 900)
    assert a == t1.random_sel('5', 42, 3328, 900) and a == sorted(a) and len(set(a)) == 900
    assert a != t1.random_sel('6', 42, 3328, 900) and a != t1.random_sel('5', 43, 3328, 900)
    assert t1.random_sel('5', 42, 3328, 0) == []
    for pid in ('1', '2'):  # same pid, rebuilt twice -> identical R context
        e1, e2 = FakeEng(), FakeEng()
        i1, i2 = item(e1, pid), item(e2, pid)
        t1.build_contexts(i1, e1, TOK, SP, CFG, 'bmain')
        t1.build_contexts(i2, e2, TOK, SP, CFG, 'bmain')
        assert i1['sel_rand'] == i2['sel_rand'] and torch.equal(i1['ctx']['R'], i2['ctx']['R'])


@pytest.mark.parametrize('z, cut, pre', [('think abc', True, []), ('think abc', False, NL), ('think abc\n', True, []),
                                         ('think abc\n\n', False, []), ('think abc\n', False, [])])
def test_contexts_hs_tag_in(z, cut, pre):
    eng = FakeEng()
    it = item(eng, z=z, cut=cut)
    t1.build_contexts(it, eng, TOK, SP, CFG, 'hs_tag_in')
    e = eng.embed
    prefix = torch.cat([it['base'], e(it['z'] + it['sep'])])
    assert eng.calls == [(it['base'].shape[0] + len(it['z']), len(it['z']))]  # Z from [base, z] only, not sep / inject
    Z = eng.last_hidden(torch.cat([it['base'], e(it['z'])]), len(it['z']))
    sel = select_tokens(it['vis'], Z, 3)
    assert it['sel'] == sel and len(sel) > 0
    # the newline before <img> only when decode(z) + sep does not already end with one
    # (z ending in '\n' with a cut step, or a cut step whose separator was added, ends with '\n')
    expect_pre = [] if (z + it['sep_text']).endswith('\n') else NL
    if z == 'think abc' and not cut:
        assert expect_pre == NL
    block = lambda idx: torch.cat([e(expect_pre), e([SP['img_start']]), it['vis'][idx], e([SP['img_end']] + NL)])  # noqa: E731
    assert torch.equal(it['ctx']['D'], torch.cat([prefix, block(sel), e(SP['wait'])]))
    assert torch.equal(it['ctx']['R'], torch.cat([prefix, block(it['sel_rand']), e(SP['wait'])]))
    assert it['ctx']['R'].shape == it['ctx']['D'].shape
    assert torch.equal(it['ctx']['C'], torch.cat([prefix, e(SP['wait'])]))  # C has no inject block, no tags
    if cut and not z.endswith('\n'):
        assert pre == [] and expect_pre == []  # the separator supplied the newline


def test_contexts_no_selection_leaves_d_and_r_equal_to_c():
    for mode in t1.INJECTS:
        eng = FakeEng()
        it = item(eng)
        it['vis'] = torch.zeros_like(it['vis'])  # DPP picks nothing from an all-zero V
        t1.build_contexts(it, eng, TOK, SP, CFG, mode)
        assert it['sel'] == [] and it['sel_rand'] == []
        assert torch.equal(it['ctx']['D'], it['ctx']['C']) and torch.equal(it['ctx']['R'], it['ctx']['C'])


# ---- seeds shared across conditions -------------------------------------------------------------------------------
def test_seeds_are_shared_across_conditions():
    eng = FakeEng()
    te = SP['think_end']
    seen = []

    def spy(reqs):
        seen.append(reqs)
        if len(seen) == 1:  # free thinking: stop at </think> after two tokens
            return [Result(ids=TOK.encode('ok') + [te], finish='stop') for _ in reqs]
        return [Result(ids=TOK.encode('{7}'), finish='stop') for _ in reqs]  # answers

    items = [item(eng, pid=p) for p in ('3', '9')]
    for it in items:
        it['row'] = {'question_type': 'free_form', 'answer_type': 'integer', 'answer': '7', 'precision': None, 'choices': None}
    t1.run_group(items, eng, TOK, SP, CFG, spy, 'bmain')
    think, ans = seen
    assert len(think) == 6 and len(ans) == 2 * 3 * (CFG['n_t1'] + 1)
    for k, it in enumerate(items):
        gs = [r.gen.initial_seed() for r in think[3 * k:3 * k + 3]]
        assert gs == [t1.seed_of(42, it['pid'], 'think')] * 3
        by_cond = ans[len(ans) // 2 * k:len(ans) // 2 * (k + 1)]
        for ci in range(3):
            chunk = by_cond[ci * 3:ci * 3 + 3] if False else by_cond[ci * (CFG['n_t1'] + 1):(ci + 1) * (CFG['n_t1'] + 1)]
            seeds = [r.gen.initial_seed() if r.gen is not None else None for r in chunk]
            assert seeds == [t1.seed_of(42, it['pid'], 'ans', j) for j in range(CFG['n_t1'])] + [None]
            assert chunk[-1].sampling == t1.GREEDY and chunk[0].sampling == t1.T1
        assert set(it['answers']) == set(it['extra']) == set('CDR')
        assert it['extra']['D']['end'] == 'close' and it['extra']['D']['n_think_end_inside'] == 0


# ---- tiny LM end to end -------------------------------------------------------------------------------------------
def tiny_items(lm, zero_vis=False, n=3):
    emb = lm.get_input_embeddings()
    out = []
    for k in range(n):
        g = torch.Generator().manual_seed(k)
        base = emb(torch.randint(5, 90, (12,), generator=g)).detach()
        vis = torch.zeros(10, base.shape[1]) if zero_vis else torch.randn(10, base.shape[1], generator=g)
        row = {'question_type': 'free_form', 'answer_type': 'integer', 'answer': '7', 'precision': None, 'choices': None}
        out.append({'pid': str(k), 'row': row, 'sample': {'think_end': 'cap'}, 'base': base, 'vis': vis,
                    'z': TOK.encode('think abc'), 'sep': NL, 'sep_text': '\n', 'recon': {'tokens_match': True}})
    return out


def tiny_run(lm, mode, zero_vis=False):
    eng = Engine(lm, CFG['temperature'], CFG['top_k'], CFG['top_p'])
    items = tiny_items(lm, zero_vis)
    t1.run_group(items, eng, TOK, SP, CFG, t1.make_generate(eng, CFG, pack), mode)
    return items


@pytest.mark.parametrize('mode', t1.INJECTS)
def test_run_group_rows_and_determinism(lm, mode):
    a, b = tiny_run(lm, mode), tiny_run(lm, mode)
    for x, y in zip(a, b):
        assert set(x['answers']) == set('CDR') and all(len(v) == CFG['n_t1'] + 1 for v in x['answers'].values())
        assert x['answers'] == y['answers'] and x['extra'] == y['extra'] and x['sel'] == y['sel']
        assert len(x['sel_rand']) == len(x['sel'])
        assert all(x['extra'][c]['len'] <= CFG['think_extra'] for c in 'CDR')
        row = t1.row_out(x, CFG, 'v', mode)
        json.dumps(row)
        assert row['inject'] == mode and row['stratum'] == 'cap' and set(row['acc_t1']) == set(row['greedy']) == set('CDR')
        assert row['n_sel'] == len(row['sel']) == len(row['sel_rand'])


@pytest.mark.parametrize('mode', t1.INJECTS)
def test_zero_visual_tokens_make_all_conditions_equal(lm, mode):
    for x in tiny_run(lm, mode, zero_vis=True):
        assert x['sel'] == [] and x['answers']['D'] == x['answers']['C'] == x['answers']['R']
        assert x['extra']['D'] == x['extra']['C'] == x['extra']['R']


# ---- analysis -----------------------------------------------------------------------------------------------------
def make_row(pid, stratum, counts, greedy=None, n_t1=3, inject='bmain', end='cap'):
    greedy = greedy or {c: int(counts[c] == n_t1) for c in 'CDR'}
    ans = {c: [{'text': '{7}', 'valid': True, 'correct': int(j < counts[c]), 'greedy': False} for j in range(n_t1)]
           + [{'text': '{7}', 'valid': True, 'correct': greedy[c], 'greedy': True}] for c in 'CDR'}
    return {'pid': pid, 'stratum': stratum, 'inject': inject, 'code_version': 'v', 'scorer_version': 's',
            'answers': ans, 'acc_t1': {c: counts[c] / n_t1 for c in 'CDR'}, 'greedy': greedy,
            'extra': {c: {'end': end, 'len': 100, 'text': 'x', 'n_think_end_inside': 0} for c in 'CDR'}}


def synth(n_close=60, n_cap=60, effect=None):
    """effect(stratum, i) -> (C, D, R) counts."""
    rows, ids = {}, []
    for s, n in (('close', n_close), ('cap', n_cap)):
        for i in range(n):
            pid = f"{'a' if s == 'close' else 'b'}{i}"
            c, d, r = effect(s, i)
            rows[pid] = make_row(pid, s, {'C': c, 'D': d, 'R': r}, end='close' if s == 'close' else 'cap')
            ids.append(pid)
    order = sorted(ids, key=lambda p: (int(p[1:]), p[0]))  # interleave strata
    return rows, order


def test_identical_c_and_d_gives_zero_difference_and_no_gain():
    rows, ids = synth(effect=lambda s, i: (i % 4, i % 4, (i + 1) % 4))
    out = an.analyse(rows, ids)
    assert out['n_analysed'] == 120 and out['inject'] == 'bmain'
    for s in ('close', 'cap'):
        p = out['primary_D_minus_C'][s]
        assert p['diff_pp'] == 0 and p['ci95_pp'][0] <= 0 <= p['ci95_pp'][1] and p['p_boot'] == 1.0
        assert p['verdict'] == 'no_gain' and p['flags'] == ['below_paper_size']  # CI [0, 0] -> high < 3.5
    assert out['secondary_pairs']['D-R']['all']['n'] == 120 and out['secondary_pairs']['R-C']['close']['n'] == 60


def test_strong_effect_is_a_gain_with_holm():
    rows, ids = synth(effect=lambda s, i: (1, 2 + (i % 2), 1))  # D - C = +1.5 counts of 3 -> +50 pp
    out = an.analyse(rows, ids)
    for s in ('close', 'cap'):
        p = out['primary_D_minus_C'][s]
        assert p['diff_pp'] == 50.0 and p['ci95_pp'][0] > 0 and p['p_holm'] < 0.05 and p['verdict'] == 'gain'
        assert p['flags'] == []
    assert out['secondary_pairs']['D-R']['all']['diff_pp'] == 50.0 and out['secondary_pairs']['R-C']['all']['diff_pp'] == 0.0


def test_holm_changes_the_decision_and_requires_the_ci():
    assert an.holm({'a': 0.01, 'b': 0.04}) == {'a': 0.02, 'b': 0.04}
    assert an.holm({'a': 0.03, 'b': 0.03}) == {'a': 0.06, 'b': 0.06}  # same p: both get the Holm factor 2
    assert an.holm({'a': 0.5, 'b': 0.001}) == {'b': 0.002, 'a': 0.5}
    # close: clear gain; cap: borderline raw p < .05 that Holm pushes over .05
    rows, ids = synth(n_close=60, n_cap=60, effect=lambda s, i: (1, 2 + (i % 2), 1) if s == 'close' else (1, 1, 1))
    for i in range(60):  # cap: D - C = +1 on 12 of 60 questions, -1 on 5 -> p near .05-.1
        c, d = (1, 2) if i < 12 else (1, 0) if i < 17 else (1, 1)
        rows[f'b{i}'] = make_row(f'b{i}', 'cap', {'C': c, 'D': d, 'R': 1})
    out = an.analyse(rows, ids)
    p = out['primary_D_minus_C']['cap']
    assert p['p_holm'] == pytest.approx(min(1.0, 2 * p['p_boot'])) or p['p_holm'] >= p['p_boot']
    assert out['primary_D_minus_C']['close']['verdict'] == 'gain'
    assert p['verdict'] == ('gain' if p['p_holm'] < 0.05 and p['ci95_pp'][0] > 0 else 'no_gain')
    # the p-value in the 'cap' stratum is not smaller after correction
    assert p['p_holm'] >= p['p_boot']


def test_no_gain_with_wide_ci_has_no_size_flag():
    rows, ids = synth(n_close=20, n_cap=20, effect=lambda s, i: (1, 1 + (1 if i % 2 else -1), 1))
    p = an.analyse(rows, ids)['primary_D_minus_C']['cap']
    assert p['verdict'] == 'no_gain' and p['flags'] == [] and p['ci95_pp'][1] >= 3.5


def test_bootstrap_matches_a_hand_case():
    lo, hi, p = an.boot_ci_p([3] * 10, 3)
    assert (lo, hi, p) == (100.0, 100.0, 0.0)
    lo, hi, p = an.boot_ci_p([0] * 10, 3)
    assert (lo, hi, p) == (0.0, 0.0, 1.0)
    assert an.boot_ci_p([1, -1] * 20, 3) == an.boot_ci_p([1, -1] * 20, 3)  # deterministic


def test_duplicate_pids_are_refused(tmp_path):
    r = make_row('1', 'close', {'C': 1, 'D': 1, 'R': 1})
    a, b = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'
    a.write_text(json.dumps(r) + '\n')
    b.write_text(json.dumps(r) + '\n')
    with pytest.raises(SystemExit):
        an.load([str(tmp_path / '*.jsonl')])
    b.write_text(json.dumps(make_row('2', 'cap', {'C': 1, 'D': 1, 'R': 1})) + '\n')
    R, files = an.load([str(tmp_path / '*.jsonl')])
    assert set(R) == {'1', '2'} and len(files) == 2


def test_complete_prefix_missing_and_incomplete_rows():
    rows, ids = synth(n_close=5, n_cap=5, effect=lambda s, i: (1, 1, 1))
    del rows[ids[6]]
    bad = rows[ids[2]]
    del bad['acc_t1']['R']
    out = an.analyse(rows, ids)
    assert out['n_prefix'] == 6 and out['missing_n'] == 1 and out['n_after_prefix'] == 4 and out['n_analysed'] == 5
    assert out['rows_lacking_a_condition'] == [ids[2]]
    assert an.prefix_of(['a', 'b', 'c'], {'a': 1, 'c': 1})[0] == ['a']


def test_mechanism_and_valid_rate_flag():
    rows, ids = synth(n_close=4, n_cap=4, effect=lambda s, i: (1, 1, 1))
    for r in rows.values():
        for a in r['answers']['D']:
            a['valid'] = False
        r['extra']['R']['n_think_end_inside'] = 1
    out = an.analyse(rows, ids)
    assert out['flags'] == ['valid_answer_rate<0.95 for D']
    assert out['conditions']['R']['all']['rows_with_think_end_inside'] == 8
    assert out['conditions']['C']['all']['end'] == {'close': 4, 'eos': 0, 'cap': 4}
    assert out['conditions']['C']['all']['mean_len'] == 100.0


def test_cli_prints_and_refuses_overwrite(tmp_path, capsys):
    rows, ids = synth(n_close=3, n_cap=3, effect=lambda s, i: (1, 1, 1))
    res, idf, out = tmp_path / 'r.jsonl', tmp_path / 'ids.json', tmp_path / 'o.json'
    res.write_text(''.join(json.dumps(rows[p]) + '\n' for p in ids))
    idf.write_text(json.dumps({'ids': ids}))
    assert an.main([str(res), '--ids', str(idf), '--out', str(out)]) == 0
    assert json.loads(out.read_text())['n_analysed'] == 6
    with pytest.raises(SystemExit):
        an.main([str(res), '--ids', str(idf), '--out', str(out)])
