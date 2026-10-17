"""visref_hs_imgtag_in / visref_hs_imgtag_out (2026-10-17 format check): visref_hs selection (Z = last-layer hidden states,
kernel V = input image embeddings) with image tags; '_in' keeps the block inside the thinking block when answering
(answer / probe context = continuation context + </think>), '_out' places it as visref_imgtag does."""
import pytest
import torch

import visref.methods as M
from test_hs import HSEngine
from test_imgtag import NL, NVIS, TOK, ids_of
from test_repro import BOX, P, S1, Result, ans, box, make_job, think  # noqa: F401
from visref.methods import Job

IN, OUT, A6 = 'visref_hs_imgtag_in', 'visref_hs_imgtag_out', 'visref_imgtag'
END, IMG, IMG_END = 1, 2, 3


class RecEngine(HSEngine):
    """HSEngine that also keeps every request's context."""

    def __init__(self, tok, scripts):
        super().__init__(tok, scripts)
        self.ctxs = []

    def step(self, job):
        self.ctxs.append((job.phase, job.request().ctx))
        super().step(job)


def job_of(method, scripts, vis=None, vis_hs=None, step1=None, **over):
    tmpl, _, _ = make_job('tsr', [], **over)
    eng = RecEngine(TOK, scripts)
    vis = torch.randn(NVIS, len(TOK.vocab), generator=torch.Generator().manual_seed(3)) if vis is None else vis
    if method == A6 and vis_hs is None:
        vis_hs = torch.randn(NVIS, len(TOK.vocab), generator=torch.Generator().manual_seed(5))
    dual = bool(tmpl.cfg.get('dual_probe'))
    j = Job(method, tmpl.row, eng.embed(TOK.encode('Q?')), vis, tmpl.cfg, TOK, eng, tmpl.sp, torch.Generator(),
            torch.Generator(), step1=step1, probe2_gen=torch.Generator() if dual else None, vis_hs=vis_hs)
    return j, eng


@pytest.fixture
def fixed_sel(monkeypatch):
    """select_tokens returns [1, 4] and records the kernel V it was given (same picks for every method)."""
    seen = []

    def sel(V, Z, m, trace=None, **kw):
        seen.append(V)
        if trace is not None:
            trace.append({'k': 0, 'res': 1.0, 'rel': 1.0, 'rel_rank': 0})
            trace.append({'k': 1, 'res': 0.5, 'rel': 1.0, 'rel_rank': 1})
        return [1, 4]

    monkeypatch.setattr(M, 'select_tokens', sel)
    monkeypatch.setattr(M, 'pick_stats', lambda tr: {'n_meaningful': len(tr)})
    return seen


def block(j):
    return [IMG] + [-1, -1] + [IMG_END, NL]


def test_tuples_keep_other_methods_unchanged():
    assert IN in M.TAG_IN and OUT not in M.TAG_IN and A6 not in M.TAG_IN
    assert {IN, OUT} <= set(M.HS) and IN not in M.IMGTAG and OUT not in M.IMGTAG  # no vis_hs, no hidden_at in run.py
    assert M.IMGTAG == (A6,) and set(M.TAG) == {A6, IN, OUT}
    assert {IN, OUT, A6, 'tsr_answer_prompt'} == set(M.ANSWER_PROMPT) and {IN, OUT} <= set(M.VISREF)
    assert not {IN, OUT} & set(M.THINK)
    import run
    assert run.IMGTAG == (A6,)


@pytest.mark.parametrize('method', [IN, OUT])
def test_kernel_is_input_embeddings_and_no_vis_hs_needed(method, fixed_sel):
    j, eng = job_of(method, [(think(TOK, 'aaaaaaaa'), 0), (box(TOK, '7'), 0.0), (ans(TOK, '7'), 0)])
    while not j.done:
        eng.step(j)
    assert fixed_sel[0] is j.vis and j.vis_hs is None
    assert eng.hs_calls == [(2 + 8, 8)]  # Z = last-layer hidden states of z_1 in its own context
    s = j.out['steps'][0]
    assert s['inject_format'] == ('img_tag_inside_think' if method == IN else 'img_tag_outside_think')
    assert s['v_space'] == 'input_embed' and s['sel'] == [1, 4] and s['z_ids'] == TOK.encode('aaaaaaaa')
    assert s['n_inject'] == 2 and s['n_meaningful'] == 2 and j.trace[1] == '<VISREF_IMG m=3 dpp=2>'
    assert j.out['final_ids'] == ans(TOK, '7') and 'final_af_ids' not in j.out
    assert j.out['final_answer'] == S1 + '7' and j.out['final_forced_answer']


def test_in_contexts_closed_step(fixed_sel):
    j, eng = job_of(IN, [(think(TOK, 'aaaa'), 0), (box(TOK, '7'), 0.0), (ans(TOK, '7'), 0)])
    while not j.done:
        eng.step(j)
    ctx = dict(eng.ctxs)
    want = TOK.encode('Q?aaaa') + [NL] + block(j) + [END]
    assert ids_of(j.answer_ctx) == want and ids_of(j.step_ctx) == want[:-1]
    assert torch.equal(j.answer_ctx[:-1], j.step_ctx)
    assert torch.equal(j.answer_ctx[8:10], j.vis[[1, 4]])
    assert torch.equal(ctx['probe'][:len(want)], j.answer_ctx) and ids_of(ctx['probe'][len(want):]) == TOK.encode(BOX)
    assert torch.equal(ctx['final'][:len(want)], j.answer_ctx) and ids_of(ctx['final'][len(want):]) == TOK.encode(S1)
    assert j.out['final_context_len'] == len(want)


def test_out_matches_visref_imgtag_contexts_with_the_same_picks(fixed_sel):
    scripts = lambda: [(think(TOK, 'aaaa'), 0), (box(TOK, '7'), 0.5), (think(TOK, 'bb'), 0), (box(TOK, '7'), 0.0),
                       (ans(TOK, '7'), 0)]
    runs = {}
    for m in (OUT, A6, IN):
        j, eng = job_of(m, scripts())
        while not j.done:
            eng.step(j)
        runs[m] = (j, eng.ctxs)
    (jo, co), (ja, ca), (ji, ci) = runs[OUT], runs[A6], runs[IN]
    assert [p for p, _ in co] == [p for p, _ in ca] == [p for p, _ in ci] == ['think', 'probe', 'think', 'probe', 'final']
    for (_, x), (_, y) in zip(co, ca):  # '_out' = A6's format exactly
        assert torch.equal(x, y)
    assert torch.equal(co[2][1], ci[2][1])  # step 2 starts from the same continuation context in both formats
    assert ids_of(ci[2][1])[-4:] == TOK.encode('Wait') and END not in ids_of(ci[2][1])
    assert not torch.equal(co[1][1], ci[1][1])  # the stopping probe differs: block before vs after </think>
    assert ids_of(co[1][1])[6:9] == [END, NL, IMG] and ids_of(ci[1][1])[6:8] == [NL, IMG]


@pytest.mark.parametrize('text,finish,pre', [('aaaa', 'close', [NL]), ('aaaa\n', 'close', []), ('abc', 'cap', [NL]),
                                             ('ab\n', 'cap', []), ('abc', 'eos', [NL])])
def test_in_newlines_and_answer_context(text, finish, pre, fixed_sel):
    j, eng = job_of(IN, [(box(TOK, '7'), 0.0), (ans(TOK, '7'), 0)])
    ids = TOK.encode(text) + ([END] if finish == 'close' else [0] if finish == 'eos' else [])
    j.consume(Result(ids=ids, finish='cap' if finish == 'cap' else 'stop'))
    want = TOK.encode('Q?' + text) + pre + block(j)
    assert ids_of(j.step_ctx) == want and ids_of(j.answer_ctx) == want + [END]
    eng.step(j)  # confident probe
    req = j.request()
    assert j.phase == 'final' and j.forced_answer  # every final answer of the format check follows S1
    assert torch.equal(req.ctx[:len(want) + 1], j.answer_ctx) and ids_of(req.ctx[len(want) + 1:]) == TOK.encode(S1)


@pytest.mark.parametrize('kmax,total,reason', [(2, 1000, 'kmax'), (5, 7, 'budget')])
def test_in_kmax_and_budget_stop_answer_from_the_probe_context(kmax, total, reason, fixed_sel):
    over = dict(kmax=kmax, total_think_cap=total, min_step_tokens=4 if reason == 'budget' else 1)
    scripts = [(think(TOK, 'aaaa'), 0), (box(TOK, '7'), 0.5)]
    if reason == 'kmax':
        scripts += [(think(TOK, 'bb'), 0), (box(TOK, '7'), 0.5)]
    j, eng = job_of(IN, scripts + [(ans(TOK, '7'), 0)], **over)
    while not j.done:
        eng.step(j)
    assert j.out['stop_reason'] == reason
    probe, final = [c for p, c in eng.ctxs if p == 'probe'][-1], [c for p, c in eng.ctxs if p == 'final'][0]
    L = probe.shape[0] - P
    assert torch.equal(probe[:L], final[:L]) and torch.equal(probe[:L], j.answer_ctx)
    assert ids_of(final[L:]) == TOK.encode(S1) and ids_of(j.answer_ctx)[-1] == END


def test_in_step_two_z_context_holds_the_first_block(fixed_sel):
    j, eng = job_of(IN, [(think(TOK, 'aaaa'), 0), (box(TOK, '7'), 0.5), (think(TOK, 'bbb'), 0), (box(TOK, '7'), 0.0),
                         (ans(TOK, '7'), 0)])
    while not j.done:
        eng.step(j)
    step2_start = [c for p, c in eng.ctxs if p == 'think'][1].shape[0]
    assert eng.hs_calls == [(2 + 4, 4), (step2_start + 3, 3)]  # Q? z_1 \n block Wait z_2: causal, block included
    assert [s['z_ids'] for s in j.out['steps']] == [TOK.encode('aaaa'), TOK.encode('bbb')]


@pytest.mark.parametrize('end', ['close', 'eos', 'cap'])
def test_in_from_stored_step1(end, fixed_sel):
    text = 'abc'
    ids = TOK.encode(text) + ([END] if end == 'close' else [0] if end == 'eos' else [])
    step1 = (ids, 'cap' if end == 'cap' else 'stop', torch.Generator().get_state())
    j, eng = job_of(IN, [(box(TOK, '7'), 0.0), (ans(TOK, '7'), 0)], step1=step1)
    assert ids_of(j.answer_ctx) == TOK.encode('Q?' + text) + [NL] + block(j) + [END]
    assert torch.equal(j.answer_ctx[:-1], j.step_ctx) and j.forked
    while not j.done:
        eng.step(j)
    assert j.out['steps'][0]['think_end'] == end and j.out['final_answer'].startswith(S1)


def test_in_dual_probe_context_and_same_decision(fixed_sel):
    j, eng = job_of(IN, [(think(TOK, 'aaaa'), 0), (box(TOK, '7'), 0.0), (box(TOK, '8'), 0.9), (ans(TOK, '7'), 0)],
                    dual_probe=1)
    while not j.done:
        eng.step(j)
    assert [p for p, _ in eng.ctxs] == ['think', 'probe', 'probe_nov', 'final']
    nov = dict(eng.ctxs)['probe_nov']
    assert ids_of(nov) == TOK.encode('Q?aaaa') + [END] + TOK.encode(BOX)  # z_k without the block
    s = j.out['steps'][0]
    assert s['H'] == 0.0 and s['H_noV'] > 0.25 and j.out['stop_reason'] == 'entropy' and j.out['k'] == 1


def test_in_reemitted_think_end_and_deer_are_recorded(fixed_sel):
    # the model writes an answer, then </think> again, then more: scored text = after the generated </think>
    j, eng = job_of(IN, [(think(TOK, 'aaaa'), 0), (box(TOK, '7'), 0.0), (TOK.encode(' 7\n') + [END] + TOK.encode('8') + [0], 0)])
    while not j.done:
        eng.step(j)
    assert j.out['final_y_start'] == 4 and j.out['final_answer'] == '8' and j.out['final_raw'].startswith(' 7')
    assert j.out['final_ids'] == TOK.encode(' 7\n') + [END] + TOK.encode('8') + [0]
    # an answer that reaches answer_cap: DEER answer forcing, its continuation recorded apart
    j, eng = job_of(IN, [(think(TOK, 'aaaa'), 0), (box(TOK, '7'), 0.0)], answer_cap=5)
    eng.step(j)
    eng.step(j)
    j.consume(Result(ids=TOK.encode('12345'), finish='cap'))
    assert j.phase == 'final_af'
    j.consume(Result(ids=TOK.encode('{9}') + [0], finish='stop'))
    assert j.done and j.out['final_ids'] == TOK.encode('12345') and j.out['final_af_ids'] == TOK.encode('{9}') + [0]
    assert j.out['final_af'] and j.out['final_answer'].endswith('{9}')


def test_room_is_the_same_for_both_formats_and_large_enough(fixed_sel):
    rooms = {}
    for m in (IN, OUT, A6):
        j, eng = job_of(m, [(box(TOK, '7'), 0.0)])
        rooms[m] = (j._vis_room(), j._think_cap(j.ctx.shape[0]))
        start = j.ctx.shape[0]
        j.consume(Result(ids=TOK.encode('abc'), finish='cap'))
        # answer context <= start + z + room (incl. the cut separator) + 1 (</think>), the reserve _think_cap keeps
        assert j.answer_ctx.shape[0] <= start + 3 + j._vis_room() + 1
    assert rooms[IN] == rooms[OUT] == rooms[A6]


def test_other_methods_rows_have_no_new_fields(fixed_sel):
    for m in (A6, 'visref_hs', 'tsr_answer_prompt'):
        j, eng = job_of(m, [(think(TOK, 'aaaa'), 0), (box(TOK, '7'), 0.0), (ans(TOK, '7'), 0)])
        while not j.done:
            eng.step(j)
        assert 'final_ids' not in j.out and 'final_af_ids' not in j.out
        assert all('sel' not in s and 'z_ids' not in s for s in j.out['steps'])
