import itertools
import math

import pytest
import torch

from visref.dpp import select_tokens
from visref.engine import Engine, Request, Result
from visref.methods import Job, PROBE_SAMPLING
from visref.scoring import SCORER_V3, SCORER_VERSION, extract, score


# ---------------------------------------------------------------- DPP ---------
def naive_greedy(V, Z, m):
    L = (V.double() @ Z.double().T) @ (V.double() @ Z.double().T).T
    sel = []
    for _ in range(m):
        best, bj = -math.inf, None
        for j in range(V.shape[0]):
            if j in sel:
                continue
            idx = sel + [j]
            val = torch.logdet(L[idx][:, idx]).item()
            if val > best:
                best, bj = val, j
        sel.append(bj)
    return sorted(sel)


def test_dpp_matches_naive_greedy():
    g = torch.Generator().manual_seed(0)
    V, Z = torch.randn(12, 8, generator=g), torch.randn(10, 8, generator=g)
    assert select_tokens(V, Z, 5) == naive_greedy(V, Z, 5)


def test_dpp_rank_deficient_returns_only_greedy_picks():
    g = torch.Generator().manual_seed(1)
    V, Z = torch.randn(20, 8, generator=g), torch.randn(3, 8, generator=g)  # rank(L) <= 3 < m
    assert select_tokens(V, Z, 6) == naive_greedy(V, Z, 3)
    assert select_tokens(V, Z, 2) == naive_greedy(V, Z, 2)


def test_dpp_scale_invariant():
    g = torch.Generator().manual_seed(2)
    V, Z = torch.randn(30, 16, generator=g), torch.randn(25, 16, generator=g)
    assert select_tokens(V, Z, 9) == select_tokens(V * 7.5, Z * 0.01, 9)


# ------------------------------------------------------------- scoring --------
MC = {'question_type': 'multi_choice', 'answer_type': 'text', 'choices': ['3', '5', '7'], 'answer': '5', 'precision': None}
FL = {'question_type': 'free_form', 'answer_type': 'float', 'choices': None, 'answer': '1.2', 'precision': 1.0}
IN = {'question_type': 'free_form', 'answer_type': 'integer', 'choices': None, 'answer': '1000', 'precision': None}


@pytest.mark.parametrize('text,row,ok', [
    ('\n\nThe answer is (B).', MC, 1),
    ('So we pick \\boxed{B}', MC, 1),
    ('Answer: 7', MC, 0),
    ('B', MC, 1),
    ('The distance is about 1.22 m.\n', FL, 1),
    ('\\boxed{\\frac{6}{5}}', FL, 1),
    ('Total: **1,000** apples', IN, 1),
    ('The answer is 999', IN, 0),
])
def test_score(text, row, ok):
    assert score(text, row)['correct'] == ok


def test_extract_span_points_at_answer():
    t = 'blah 3 blah. Final answer: 42 units'
    v, (s, e), _ = extract(t, IN)
    assert v == '42' and t[s:e] == '42'


def _mc(choices, gold):
    return {'question_type': 'multi_choice', 'answer_type': 'text', 'choices': choices, 'answer': gold, 'precision': None}


def test_markdown_answer_label_takes_next_line_pid836():
    row = _mc(['chinlea', 'arganodus', 'semionotus', 'xenacanthus'], 'semionotus')
    t = 'reasoning...\n\nThe correct answer is C.  \n\nAnswering as per the question and options:  \nThe answer is C.  \n\n**Answer:**  \nC'
    r = score(t, row)
    assert (r['extracted'], r['prediction'], r['correct'], r['extract_failed']) == ('C', 'semionotus', 1, False)
    assert r['scorer_version'] == SCORER_VERSION == 'v4_mc_empty_wrong'
    v, (s, e), _ = extract(t, row)
    assert t[s:e] == 'C'


@pytest.mark.parametrize('text,choices,gold', [
    ('', ['Yes', 'No'], 'No'),  # v3: '' -> Levenshtein-closest = shortest option
    ('   \n', ['larger than', 'smaller than', 'equal to'], 'equal to'),
    # b_main reading (b) rows cut before the af prompt (pid 122 st, pid 117 tsr): label without content
    ('the answer is **B**. Wait, no, the answer is **', ['Yes', 'No'], 'No'),
    ('Wait, no. Wait, the correct answer is **', ['[0, 2]', '[3, 2]', '[2, 4]', '[-3, 4]'], '[0, 2]'),
])
def test_v4_empty_multiple_choice_extraction_is_wrong(text, choices, gold):
    row = _mc(choices, gold)
    v4, v3 = score(text, row), score(text, row, version=SCORER_V3)
    assert v4['extract_failed'] and v4['prediction'] is None and v4['correct'] == 0
    assert v4['scorer_version'] == SCORER_VERSION
    assert v3['prediction'] == gold and v3['correct'] == 1 and v3['scorer_version'] == SCORER_V3  # old behaviour kept


@pytest.mark.parametrize('text,gold,ok', [('The answer is B.', 'No', 1), ('**Answer:** yes', 'Yes', 1),
                                          ('Final answer: (A)', 'No', 0)])
def test_v4_equals_v3_when_extraction_nonempty(text, gold, ok):
    row = _mc(['Yes', 'No'], gold)
    v4, v3 = score(text, row), score(text, row, version=SCORER_V3)
    assert (v4['prediction'], v4['correct']) == (v3['prediction'], v3['correct']) and v4['correct'] == ok


def test_unknown_scorer_version_rejected():
    with pytest.raises(AssertionError):
        score('A', _mc(['x', 'y'], 'x'), version='v2')


def test_markdown_answer_label_takes_next_line_pid768():
    row = _mc(['December, January, and February', 'July and August', 'March and April'], 'July and August')
    t = ('...\n\n**Answer:**  \nB. July and August  \n**Explanation:** July and August have the highest average '
         'temperatures, making this the correct answer. \n\n**Answer:**  \nB. July and August')
    r = score(t, row)
    assert (r['extracted'], r['correct'], r['extract_failed']) == ('B', 1, False)
    assert r['rule'] == 'answer_phrase:letter'


@pytest.mark.parametrize('text', ['Answer: C', '**Answer:** C', 'thinking\n**Answer:** C  \n'])
def test_answer_label_same_line_still_extracts(text):
    row = _mc(['3', '5', '7'], '7')
    r = score(text, row)
    assert r['extracted'] == 'C' and r['correct'] == 1 and not r['extract_failed']


@pytest.mark.parametrize('text', ['The answer is:\n\n**B) y**', 'Final answer: **B**.', 'The answer is: B', 'Answer is **B.**'])
def test_v3_colon_after_is_and_emphasis_before_period(text):
    row = _mc(['x', 'y', 'z', 'w'], 'y')
    r = score(text, row)
    assert (r['extracted'], r['prediction'], r['correct'], r['extract_failed']) == ('B', 'y', 1, False)


def test_v3_leaves_plain_integer_and_text_cleaning_unchanged():
    row5 = {'question_type': 'free_form', 'answer_type': 'integer', 'choices': None, 'answer': '5', 'precision': None}
    r = score('The answer is 5', row5)
    assert (r['extracted'], r['correct'], r['rule']) == ('5', 1, 'answer_phrase:number')
    assert score('The answer is: 5', row5)['extracted'] == '5'
    txt = {'question_type': 'free_form', 'answer_type': 'text', 'choices': None, 'answer': 'red apple', 'precision': None}
    for s in ('The answer is red apple.', 'Final answer: **red apple**.', 'Answer: red apple'):
        r = score(s, txt)
        assert (r['extracted'], r['correct']) == ('red apple', 1), s


# ------------------------------------------------------- tiny LM engine ------
class CharTok:
    specials = ['<|im_end|>', '</think>', '<img>', '</img>', '<IMG_CONTEXT>']

    def __init__(self):
        self.vocab = self.specials + [chr(c) for c in range(32, 127)] + ['\n']

    def encode(self, s, add_special_tokens=False):
        return [self.vocab.index(c) for c in s]

    def decode(self, ids):
        return ''.join(self.vocab[i] for i in ids)


@pytest.fixture(scope='module')
def lm():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(CharTok().vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=512)
    return Qwen3ForCausalLM(cfg).eval()


def ctx(lm, n, seed):
    g = torch.Generator().manual_seed(seed)
    return lm.get_input_embeddings()(torch.randint(5, 90, (n,), generator=g)).detach()


def run(eng, reqs, chunk):
    """Drive chunked generation to completion like the job loop does."""
    outs = [Result() for _ in reqs]
    live = list(range(len(reqs)))
    while live:
        res = eng.generate([reqs[i] for i in live], chunk)
        nxt = []
        for i, r in zip(live, res):
            outs[i].ids += r.ids
            outs[i].ent += r.ent
            outs[i].finish = r.finish
            rq = reqs[i]
            reqs[i] = Request(torch.cat([rq.ctx, eng.embed(r.ids)]), rq.stop_ids, rq.max_new - len(r.ids), rq.gen, rq.record_entropy)
            if r.finish == 'chunk':
                nxt.append(i)
        live = nxt
    return outs


@pytest.mark.parametrize('temperature,attn_budget', [(0.0, 2 ** 28), (0.8, 2 ** 28), (0.8, 2000)])
def test_batched_chunked_equals_single(lm, temperature, attn_budget):
    eng = Engine(lm, temperature=temperature, top_k=20, top_p=0.9, attn_budget=attn_budget)
    lens = [7, 40, 25]

    def reqs():
        return [Request(ctx(lm, n, s), frozenset({0}), 25 + s, torch.Generator().manual_seed(100 + s), True)
                for s, n in enumerate(lens)]
    single = [run(eng, [r], 1000)[0] for r in reqs()]
    batched = run(eng, reqs(), 1000)
    chunked = run(eng, reqs(), 4)
    for a, b, c in zip(single, batched, chunked):
        assert a.ids == b.ids == c.ids
        assert torch.allclose(torch.tensor(a.ent), torch.tensor(b.ent), atol=1e-4)


def test_entropy_matches_full_forward(lm):
    eng = Engine(lm, temperature=0.0)
    c = ctx(lm, 9, 3)
    r = run(eng, [Request(c, frozenset(), 6, torch.Generator(), True)], 1000)[0]
    full = torch.cat([c, eng.embed(r.ids[:-1])])[None]
    logp = torch.log_softmax(lm(inputs_embeds=full).logits[0, -6:].float(), -1)
    H = -(logp.exp() * logp).sum(-1)
    assert torch.allclose(H, torch.tensor(r.ent), atol=1e-4)


def test_sampler_topk_first_respects_top_p_and_k():
    from visref.engine import sample_row
    logits = torch.tensor([5.0, 4.9, 1.0, 0.0, -1.0])
    g = torch.Generator().manual_seed(0)
    draws = {sample_row(logits, g, 1.0, 2, 1.0) for _ in range(200)}
    assert draws == {0, 1}
    draws = {sample_row(logits, g, 1.0, 5, 0.3) for _ in range(200)}
    assert draws == {0}


def test_generate_stop_at_returns_chunk(lm):
    eng = Engine(lm, temperature=0.8)
    r = Request(ctx(lm, 9, 1), frozenset(), 100, torch.Generator().manual_seed(1), True)
    out = eng.generate([r], 1000, stop_at=0.0)
    assert out[0].finish == 'chunk' and len(out[0].ids) == 1


@pytest.mark.parametrize('temperature', [0.0, 0.8])
def test_fast_kv_matches_stock(lm, temperature):
    from visref.fastkv import set_enabled
    lens = [7, 40, 25]

    def reqs():
        return [Request(ctx(lm, n, s), frozenset({0}), 25 + s, torch.Generator().manual_seed(100 + s), True)
                for s, n in enumerate(lens)]
    stock = run(Engine(lm, temperature=temperature, top_k=20, top_p=0.9), reqs(), 6)
    set_enabled(True)
    try:
        fast = run(Engine(lm, temperature=temperature, top_k=20, top_p=0.9, fast_kv=True, attn_budget=2000), reqs(), 6)
    finally:
        set_enabled(False)
    for a, b in zip(stock, fast):
        assert a.ids == b.ids
        assert torch.allclose(torch.tensor(a.ent), torch.tensor(b.ent), atol=1e-4)


# ------------------------------------------------------- \\boxed{ probe ---------
def test_brace_ids_close_set():
    from visref.model import brace_ids
    b = brace_ids(['}', ' }', '}\n\n', 'a}', '{', 'x', '}}'])
    assert b['close_ids'] == frozenset({0, 1, 2, 6}) and set(b['brace_text']) == {0, 1, 2, 3, 4, 6}




def test_engine_lumped_entropy_and_stop_check(lm):
    eng = Engine(lm, temperature=0.0)
    S = frozenset({5, 9, 40})
    r = Request(ctx(lm, 9, 4), frozenset(), 50, None, True, stop_check=lambda ids: len(ids) >= 4, ent_lump=S)
    out = run_lump(eng, r)
    assert len(out.ids) == 4 and out.finish == 'stop' and len(out.ent_lump) == 4
    full = torch.cat([r.ctx, eng.embed(out.ids[:-1])])[None]
    logp = torch.log_softmax(lm(inputs_embeds=full).logits[0, -4:].float(), -1)
    p = logp.exp()
    idx = torch.tensor(sorted(S))
    PS = p[:, idx].sum(-1)
    rest = torch.ones(p.shape[-1], dtype=torch.bool)
    rest[idx] = False
    H = -(p[:, rest] * logp[:, rest]).sum(-1) - PS * PS.log()
    assert torch.allclose(H, torch.tensor(out.ent_lump), atol=1e-4)
    assert all(a >= b - 1e-6 for a, b in zip(out.ent, out.ent_lump))  # merging outcomes never adds entropy


def run_lump(eng, r):
    return eng.generate([r], 1000)[0]


def test_engine_sampling_override_runs_to_cap(lm):
    eng = Engine(lm, temperature=0.0)
    r2 = Request(ctx(lm, 9, 2), frozenset(), 50, torch.Generator().manual_seed(0), False, sampling=(1.0, 0, 1.0))
    assert len(run(eng, [r2], 1000)[0].ids) == 50


# ------------------------------------------------------- method logic --------
class ScriptEngine:
    """Returns scripted outputs per phase; embeddings are one-hot rows."""

    def __init__(self, tok, scripts):
        self.tok, self.scripts, self.calls = tok, scripts, []

    def embed(self, ids):
        return torch.nn.functional.one_hot(torch.as_tensor(ids, dtype=torch.long), len(self.tok.vocab)).float()

    def step(self, job):
        req = job.request()
        self.calls.append((job.phase, req.ctx.shape[0]))
        ids, h, *hl = self.scripts.pop(0)
        job.consume(Result(ids=ids, ent=[h] * len(ids) if req.record_entropy else [],
                           ent_lump=[hl[0] if hl else h] * len(ids) if req.ent_lump else [], finish='stop'))


BOX = '\n\n\\boxed'
S1 = '\nFinal Answer:'
AF = '\n**Final Answer**\n\\boxed'
P = len(BOX)


def make_job(method, scripts, kmax=3, nvis=10, vis_hs=None, **over):
    from visref.model import brace_ids
    tok = CharTok()
    sp = {'think_end': 1, 'eos': 0, 'img_start': 2, 'img_end': 3, 'wait': tok.encode('Wait'),
          'probe_prompt': tok.encode(BOX), **brace_ids(tok.vocab)}
    cfg = dict(delta=0.25, kmax=kmax, vis_frac=0.3, think_cap=100, total_think_cap=1000, answer_cap=50,
               max_context=5000, min_step_tokens=1, wait_text='Wait', probe_cap=16, probe_prompt=BOX,
               forced_prompt=S1, af_prompt=AF, af_cap=8, cut_sep='\n')
    cfg.update(over)
    eng = ScriptEngine(tok, scripts)
    base = eng.embed(tok.encode('Q?'))
    vis = torch.randn(nvis, len(tok.vocab))
    job = Job(method, IN, base, vis, cfg, tok, eng, sp, torch.Generator(), torch.Generator(), vis_hs=vis_hs)
    return job, eng, tok


def think(tok, s):
    return tok.encode(s) + [1]


def ans(tok, s):
    return tok.encode(s) + [0]


def box(tok, s):
    return tok.encode('{' + s + '}')


def test_tsr_probe_sums_entropy_and_continues_until_confident():
    tok = CharTok()
    job, eng, _ = make_job('tsr', [(think(tok, 'abc'), 0), (box(tok, '7'), 0.5),
                                   (think(tok, 'de'), 0), (box(tok, '7'), 0.05), (ans(tok, '\n1000'), 0)])
    while not job.done:
        eng.step(job)
    assert [c[0] for c in eng.calls] == ['think', 'probe', 'think', 'probe', 'final']
    # probe = Q? + abc + </think> + "\n\n\boxed" ; step 2 = Q? + abc + Wait ; final = step2 + de + </think>
    assert eng.calls[1][1] == 2 + 3 + 1 + P
    assert eng.calls[2][1] == 2 + 3 + 4 and eng.calls[3][1] == 2 + 3 + 4 + 2 + 1 + P
    assert eng.calls[4][1] == 2 + 3 + 4 + 2 + 1
    s0, s1 = job.out['steps']
    assert s0['H'] == pytest.approx(1.5) and s1['H'] == pytest.approx(0.15)  # '{' full + '7', '}' merged
    assert s0['probe_answer'] == '{7}' and s0['probe_finish'] == 'end' and s0['probe_inner'] == '7'
    assert job.out['stop_reason'] == 'entropy' and job.out['k'] == 2 and job.out['thinking'] == 'abcWaitde'
    assert job.out['final_answer'] == '\n1000' and job.out['final_finish'] == 'eos'


def test_brace_probe_request():
    tok = CharTok()
    job, eng, _ = make_job('tsr', [(think(tok, 'abc'), 0)])
    assert job.request().sampling is None
    eng.step(job)
    req = job.request()
    assert req.sampling == PROBE_SAMPLING and req.record_entropy and req.gen is job.probe_gen
    assert req.ent_lump == frozenset({tok.vocab.index('}')}) and req.max_new == 16
    assert req.ctx.shape[0] == 2 + 3 + 1 + P
    assert not req.stop_check(tok.encode('{\\text{7}')) and req.stop_check(tok.encode('{\\text{7}}'))
    assert not req.stop_check(tok.encode('7}'))  # box never opened


def test_brace_probe_merges_close_only_at_box_depth():
    tok = CharTok()
    job, eng, _ = make_job('tsr', [(think(tok, 'abc'), 0)])
    eng.step(job)
    ids = tok.encode('{\\text{7}}')  # depth before each token: 0 ('{'), 1 x6 ('\\text{'), 2 ('7'), 2, 1 ('}')
    full = [0.1 * (i + 1) for i in range(len(ids))]
    lump = [0.01 * (i + 1) for i in range(len(ids))]
    job.consume(Result(ids=ids, ent=full, ent_lump=lump, finish='stop'))
    st = job.steps[0]
    assert st['probe_valid'] and st['probe_finish'] == 'end' and st['probe_inner'] == '\\text{7}'
    assert st['H'] == pytest.approx(full[0] + sum(lump[1:7]) + full[7] + full[8] + lump[9])


@pytest.mark.parametrize('ids_text,finish,label', [('{}', 'stop', 'invalid'),     # empty box
                                                    ('{  }', 'stop', 'invalid'),  # whitespace only
                                                    ('{7', 'cap', 'cap')])
def test_brace_probe_not_confident(ids_text, finish, label):
    tok = CharTok()
    job, eng, _ = make_job('tsr', [(think(tok, 'abc'), 0)])
    eng.step(job)
    ids = tok.encode(ids_text)
    job.consume(Result(ids=ids, ent=[0.0] * len(ids), ent_lump=[0.0] * len(ids), finish=finish))
    st = job.steps[0]
    assert st['H'] == float('inf') and st['probe_finish'] == label and job.phase == 'think' and job.k == 2


def test_brace_probe_eos_before_close_is_not_confident():
    tok = CharTok()
    job, eng, _ = make_job('tsr', [(think(tok, 'abc'), 0)])
    eng.step(job)
    job.consume(Result(ids=tok.encode('{7') + [0], ent=[0.0] * 3, ent_lump=[0.0] * 3, finish='stop'))
    assert job.steps[0]['H'] == float('inf') and job.steps[0]['probe_finish'] == 'invalid'


def test_brace_probe_stops_when_confident():
    tok = CharTok()
    job, eng, _ = make_job('tsr', [(think(tok, 'abc'), 0), (tok.encode('{7}'), 0.05, 0.04), (ans(tok, '\n1000'), 0)])
    while not job.done:
        eng.step(job)
    assert job.out['stop_reason'] == 'entropy' and job.out['k'] == 1
    assert job.out['steps'][0]['H'] == pytest.approx(0.13) and job.out['final_answer'] == '\n1000'


def test_brace_probe_with_open_brace_prompt():
    tok = CharTok()
    job, eng, _ = make_job('tsr', [(think(tok, 'abc'), 0)], probe_prompt=BOX + '{')  # box open: depth starts at 1
    eng.step(job)
    job.consume(Result(ids=tok.encode('12}'), ent=[0.3] * 3, ent_lump=[0.01] * 3, finish='stop'))
    st = job.steps[0]
    assert st['probe_valid'] and st['probe_inner'] == '12' and st['H'] == pytest.approx(0.03)


def test_kmax_and_visref_injection_of_greedy_picks():
    tok = CharTok()
    job, eng, _ = make_job('visref_raw_rank', [(think(tok, 'ab'), 0), (box(tok, '7'), 0.9),
                                               (think(tok, 'c'), 0), (box(tok, '7'), 0.9), (ans(tok, '7'), 0)], kmax=2)
    while not job.done:
        eng.step(job)
    n1, n2 = 2, 1  # rank(V Z^T) = #distinct tokens of z_k (< m = floor(0.3 * 10) = 3)
    assert eng.calls[1][1] == 2 + 2 + n1 + 1 + P  # Q? + ab + bare V_1 + </think> + probe prompt
    assert eng.calls[2][1] == 2 + 2 + n1 + 4
    assert eng.calls[4][1] == 2 + 2 + n1 + 4 + 1 + n2 + 1  # final: no answer prompt (step 2 closed)
    assert job.out['stop_reason'] == 'kmax' and job.out['k'] == 2
    st = job.out['steps']
    assert [s['m'] for s in st] == [3, 3] and [s['n_inject'] for s in st] == [n1, n2] == [s['n_dpp'] for s in st]
    assert [s['n_unique_z'] for s in st] == [2, 1] and st[0]['rank_V'] == 10


def test_visref_cut_step_separator_comes_before_injected_tokens():
    tok = CharTok()
    job, eng, _ = make_job('visref_raw_rank', [], kmax=2)
    job.consume(Result(ids=tok.encode('abc'), finish='cap'))
    n = job.steps[0]['n_inject']
    assert n == 3 and job.request().ctx.shape[0] == 2 + 3 + 1 + n + 1 + P  # Q? + abc + '\n' + V + </think> + probe
    assert job.steps[0]['think_tokens'] == 3 and job.trace[0] == 'abc\n'


def test_st_cut_step_gets_s1_forced_answer():
    tok = CharTok()
    job, eng, _ = make_job('st', [])
    job.consume(Result(ids=tok.encode('xyz'), finish='cap'))
    req = job.request()
    assert job.phase == 'final' and job.forced_answer and req.gen is job.gen
    assert req.ctx.shape[0] == 2 + 3 + 1 + 1 + len(S1) and req.max_new == 50 and req.stop_check is None
    job.consume(Result(ids=ans(tok, ' 1000'), finish='stop'))
    o = job.out
    assert o['final_answer'] == S1 + ' 1000' and o['final_forced_valid'] and o['final_finish'] == 'eos'
    assert o['final_forced_close'] and o['stop_reason'] == 'st' and score(o['final_answer'], IN)['correct'] == 1


def test_s1_newline_only_when_cut_text_lacks_one_and_no_forcing_after_eos():
    tok = CharTok()
    job, eng, _ = make_job('st', [])
    job.consume(Result(ids=tok.encode('ab\n'), finish='cap'))  # already ends with a newline: nothing added
    assert job.forced_answer and job.request().ctx.shape[0] == 2 + 3 + 1 + len(S1)
    job2, _, _ = make_job('st', [])
    job2.consume(Result(ids=tok.encode('ab') + [0], finish='stop'))  # EOS without </think>: natural answer
    req = job2.request()
    assert job2.forced and not job2.forced_answer and req.ctx.shape[0] == 2 + 2 + 1 and req.max_new == 50
    job2.consume(Result(ids=ans(tok, '\n1000'), finish='stop'))
    assert job2.out['final_answer'] == '\n1000' and job2.out['final_forced_close'] and not job2.out['final_forced_answer']


def test_natural_close_gets_natural_answer():
    tok = CharTok()
    job, eng, _ = make_job('st', [(think(tok, 'a'), 0)])
    eng.step(job)
    req = job.request()
    assert not job.forced_answer and req.ctx.shape[0] == 2 + 1 + 1 and req.max_new == 50 and req.stop_check is None
    job.consume(Result(ids=ans(tok, '\n1000'), finish='stop'))
    assert job.out['final_answer'] == '\n1000' and not job.out['final_forced_answer']


def test_final_text_after_last_generated_think_end():
    tok = CharTok()
    job, eng, _ = make_job('st', [(think(tok, 'a'), 0)])
    eng.step(job)
    job.consume(Result(ids=tok.encode('draft') + [1] + ans(tok, '\n1000'), finish='stop'))
    assert job.out['final_answer'] == '\n1000' and job.out['final_y_start'] == 6 and job.out['final_raw']


def test_s1_later_think_end_is_respected():
    tok = CharTok()
    job, eng, _ = make_job('st', [])
    job.consume(Result(ids=tok.encode('xyz'), finish='cap'))
    job.consume(Result(ids=tok.encode(' more') + [1] + ans(tok, '\n1000'), finish='stop'))
    assert job.out['final_answer'] == '\n1000' and job.out['final_y_start'] == 6


@pytest.mark.parametrize('text,ok', [(S1 + ' B', 1), (S1 + ' (B)', 1), (S1 + ' 5', 1), (S1 + ' C', 0)])
def test_s1_final_answer_scoring_multiple_choice(text, ok):
    assert score(text, MC)['correct'] == ok


def test_s1_answer_hitting_cap_gets_deer_answer_forcing():
    tok = CharTok()
    job, eng, _ = make_job('st', [])
    job.consume(Result(ids=tok.encode('xyz'), finish='cap'))                   # thinking cut
    job.consume(Result(ids=tok.encode('B... wait ' * 10)[:50], finish='cap'))  # s1 answer hits answer_cap (50)
    req = job.request()
    assert job.phase == 'final_af' and not job.done and req.max_new == 8 and req.stop_check is None
    assert req.ctx.shape[0] == 2 + 3 + 1 + 1 + len(S1) + 50 + len(AF)
    job.consume(Result(ids=tok.encode('{1000}') + [0], finish='stop'))
    o = job.out
    assert o['final_af'] and o['final_af_valid'] and o['final_af_finish'] == 'eos' and o['final_af_len'] == 7
    assert o['final_finish'] == 'cap' and o['final_len'] == 50 and not o['final_forced_valid']
    assert o['final_answer'].endswith(AF + '{1000}') and o['final_answer'].startswith(S1)
    assert score(o['final_answer'], IN)['correct'] == 1


def test_natural_answer_hitting_cap_also_gets_answer_forcing_and_eos_answer_does_not():
    tok = CharTok()
    job, eng, _ = make_job('st', [(think(tok, 'a'), 0)])
    eng.step(job)
    job.consume(Result(ids=tok.encode('9' * 50), finish='cap'))
    assert job.phase == 'final_af'
    job.consume(Result(ids=tok.encode('{1000}'), finish='cap'))
    assert job.out['final_af'] and job.out['final_af_valid'] and job.out['final_af_finish'] == 'cap'
    job2, eng2, _ = make_job('st', [(think(tok, 'a'), 0)])
    eng2.step(job2)
    job2.consume(Result(ids=ans(tok, ' 1000'), finish='stop'))
    assert job2.done and not job2.out.get('final_af')


def test_answer_forcing_after_later_think_end_keeps_scored_tail():
    tok = CharTok()
    job, eng, _ = make_job('st', [(think(tok, 'a'), 0)])
    eng.step(job)
    job.consume(Result(ids=tok.encode('xx') + [1] + tok.encode('7' * 47), finish='cap'))
    job.consume(Result(ids=tok.encode('{1000}') + [0], finish='stop'))
    assert job.out['final_y_start'] == 3 and job.out['final_answer'] == '7' * 47 + AF + '{1000}'


def test_cut_step_gets_separator_before_wait_then_natural_answer_at_kmax():
    tok = CharTok()
    job, eng, _ = make_job('tsr', [], kmax=2)
    job.consume(Result(ids=tok.encode('abc'), finish='cap'))  # step 1 cut by the cap
    assert job.phase == 'probe'
    job.consume(Result(ids=tok.encode('{7}'), ent=[0.5] * 3, ent_lump=[0.5] * 3, finish='stop'))  # H=1.5
    assert job.phase == 'think' and job.k == 2
    assert job.request().ctx.shape[0] == 2 + 3 + 1 + len('Wait')  # Q? + abc + '\n' + Wait
    job.consume(Result(ids=think(tok, 'de'), finish='stop'))  # step 2 closes by itself
    job.consume(Result(ids=tok.encode('{7}'), ent=[0.5] * 3, ent_lump=[0.5] * 3, finish='stop'))
    assert job.phase == 'final' and not job.forced_answer and job.reason == 'kmax'
    assert job.trace[:3] == ['abc\n', 'Wait', 'de']


def test_cut_step_at_kmax_gets_forced_answer():
    tok = CharTok()
    job, eng, _ = make_job('tsr', [], kmax=1)
    job.consume(Result(ids=tok.encode('abc'), finish='cap'))
    job.consume(Result(ids=tok.encode('{7}'), ent=[0.5] * 3, ent_lump=[0.5] * 3, finish='stop'))
    assert job.phase == 'final' and job.forced_answer and job.request().ctx.shape[0] == 2 + 3 + 1 + 1 + len(S1)


def test_closed_step_continues_without_separator():
    tok = CharTok()
    job, eng, _ = make_job('tsr', [(think(tok, 'abc'), 0), (box(tok, '7'), 0.5)])
    eng.step(job)
    eng.step(job)
    assert job.phase == 'think' and job.request().ctx.shape[0] == 2 + 3 + len('Wait')
    assert job.trace == ['abc', 'Wait']


def test_probe_uses_its_own_generator():
    tok = CharTok()
    job, eng, _ = make_job('tsr', [(think(tok, 'abc'), 0), (box(tok, '7'), 0.0)])
    g, pg = job.gen, job.probe_gen
    assert job.request().gen is g
    eng.step(job)
    assert job.phase == 'probe' and job.request().gen is pg
    eng.step(job)
    assert job.phase == 'final' and job.request().gen is g


def test_fork_from_step1_matches_running_step1():
    tok = CharTok()
    scripts = [(think(tok, 'abc'), 0), (box(tok, '1'), 0.01), (ans(tok, '\n1000'), 0)]
    a, ea, _ = make_job('visref_raw_rank', list(scripts))
    while not a.done:
        ea.step(a)
    leader, el, _ = make_job('st', [(think(tok, 'abc'), 0)])
    el.step(leader)
    b, eb, _ = make_job('visref_raw_rank', list(scripts[1:]))
    b2 = Job('visref_raw_rank', IN, b.ctx, b.vis, b.cfg, tok, eb, b.sp, torch.Generator(), torch.Generator(),
             step1=leader.step1)
    while not b2.done:
        eb.step(b2)
    assert b2.out['thinking'].split('<VISREF')[0] == a.out['thinking'].split('<VISREF')[0] == 'abc'
    assert b2.out['steps'][0]['think_tokens'] == a.out['steps'][0]['think_tokens'] == 3 and b2.forked


def _real_job(lm, method, seed, probe_seed, kmax=1):
    from visref.model import brace_ids
    tok = CharTok()
    eng = Engine(lm, temperature=0.8)
    sp = {'think_end': 1, 'eos': 0, 'wait': tok.encode('Wait'), 'probe_prompt': tok.encode(BOX), **brace_ids(tok.vocab)}
    cfg = dict(delta=0.25, kmax=kmax, vis_frac=0.3, think_cap=12, total_think_cap=100, answer_cap=10,
               max_context=5000, min_step_tokens=1, wait_text='Wait', probe_cap=6, probe_prompt=BOX,
               forced_prompt=S1, af_prompt=AF, af_cap=6, cut_sep='\n')
    gen = torch.Generator().manual_seed(seed)
    pg = gen if probe_seed is None else torch.Generator().manual_seed(probe_seed)  # None: shared (pre-CRN)
    job = Job(method, IN, eng.embed(tok.encode('Q?')), ctx(lm, 10, 7), cfg, tok, eng, sp, gen, pg)
    return job, eng


def _drive(job, eng):
    while not job.done:
        job.consume(eng.generate([job.request()], 1000)[0])
    return job.out


@pytest.mark.parametrize('seed', [5, 11, 23])
def test_crn_tsr_stopping_at_k1_reproduces_st_final_answer(lm, seed):
    st = _drive(*_real_job(lm, 'st', seed, seed + 1))
    tsr = _drive(*_real_job(lm, 'tsr', seed, seed + 1))
    assert tsr['k'] == 1 and tsr['steps'][0]['think_tokens'] == st['steps'][0]['think_tokens']
    assert tsr['final_answer'] == st['final_answer'] and tsr['final_len'] == st['final_len']
    assert tsr['final_forced_answer'] == st['final_forced_answer']
    old = _drive(*_real_job(lm, 'tsr', seed, None))  # pre-CRN: the probe used the answer's draws
    assert old['final_answer'] != st['final_answer']
