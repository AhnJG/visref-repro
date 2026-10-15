"""visref_bundle (2026-10-15 exploratory variant): V and Z are last-layer hidden states, the picked input embeddings
are wrapped in <img>...</img> and placed after </think> when answering, inside the thinking block when continuing."""
import pytest
import torch

import visref.methods as M
from test_hs import HSEngine
from test_repro import BOX, P, S1, Result, ans, box, make_job, think, CharTok, lm  # noqa: F401 (lm is a fixture)
from visref.engine import Engine
from visref.methods import Job
from visref.model import image_positions

TOK = CharTok()
NL = TOK.encode('\n')[0]
NVIS = 10


def bundle_job(scripts, **over):
    """A visref_bundle Job on an HSEngine (distinct Z rows), with vis_hs different from vis."""
    tok = TOK
    vis_hs = torch.randn(NVIS, len(tok.vocab), generator=torch.Generator().manual_seed(5))
    job, _, _ = make_job('visref_bundle', [], vis_hs=vis_hs, **over)
    eng = HSEngine(tok, scripts)
    j = Job('visref_bundle', job.row, eng.embed(tok.encode('Q?')), job.vis, job.cfg, tok, eng, job.sp,
            torch.Generator(), torch.Generator(), vis_hs=vis_hs)
    return j, eng, vis_hs


def ids_of(ctx):
    """Token ids of one-hot rows; -1 for any other row (injected visual embeddings)."""
    one = (ctx.sum(1) == 1) & (ctx.max(1).values == 1)
    return [int(i) if o else -1 for i, o in zip(ctx.argmax(1), one)]


@pytest.mark.parametrize('attn_budget', [2 ** 28, 300])
def test_hidden_at_matches_full_forward(lm, attn_budget):
    torch.manual_seed(4)
    L, d = 40, lm.config.hidden_size
    ctx = torch.randn(L, d)
    eng = Engine(lm, attn_budget=attn_budget)  # 300 forces a chunked prefill
    with torch.no_grad():
        ref = lm(inputs_embeds=ctx[None], output_hidden_states=True).hidden_states[-1][0]
    for idx in ([0, 3, 17, 39], list(range(12, 29)), torch.tensor([39, 5, 5, 0]), []):
        got = eng.hidden_at(ctx, idx)
        want = ref[torch.as_tensor(idx, dtype=torch.long)]
        assert got.shape == (len(idx), d) and torch.allclose(got, want, atol=1e-5)
    for n in (1, 7, L):
        assert torch.allclose(eng.hidden_at(ctx, range(L - n, L)), eng.last_hidden(ctx, n), atol=1e-6)


def test_image_positions_helper():
    ids = torch.tensor([7, 9, 5, 5, 5, 8, 5, 1])
    pos = image_positions(ids, 5)
    assert pos.dtype == torch.long and pos.tolist() == [2, 3, 4, 6]
    assert image_positions(ids, 99).numel() == 0


def test_bundle_requires_vis_hs():
    with pytest.raises(AssertionError):
        make_job('visref_bundle', [])
    with pytest.raises(AssertionError):
        make_job('visref_bundle', [], vis_hs=torch.randn(NVIS + 1, len(TOK.vocab)))
    make_job('visref_hs', [])  # other methods do not need it


def test_bundle_contexts_closed_step(monkeypatch):
    tok = TOK
    j, eng, vis_hs = bundle_job([(think(tok, 'aaaaaaaa'), 0), (box(tok, '7'), 0.0), (ans(tok, '7'), 0)])
    seen = []
    real = M.select_tokens

    def spy(V, Z, m, **kw):
        seen.append((V, real(V, Z, m, **kw)))
        return seen[-1][1]

    monkeypatch.setattr(M, 'select_tokens', spy)
    ctxs = {}
    while not j.done:
        ctxs.setdefault(j.phase, j.request().ctx)
        eng.step(j)
    assert eng.hs_calls == [(2 + 8, 8)]
    assert len(seen) == 1 and seen[0][0] is vis_hs and seen[0][0] is not j.vis  # kernel V = hidden states
    sel = seen[0][1]
    s = j.out['steps'][0]
    n = len(sel)
    assert s['inject_format'] == 'img_tag_outside_think' and s['v_space'] == 'last_hidden'
    assert s['n_inject'] == s['n_dpp'] == n and s['m'] == 3 and 'n_meaningful' in s and 'n_unique_z' in s
    base = 2 + 8 + 1 + 1 + 1 + n + 1 + 1  # Q? z </think> '\n' <img> V </img> '\n'
    assert ctxs['probe'].shape[0] == base + P
    assert ctxs['final'].shape[0] == base == j.out['final_context_len']
    c = ctxs['probe']
    assert ids_of(c[:10]) == tok.encode('Q?') + tok.encode('a' * 8)
    assert ids_of(c[10:13]) == [1, NL, 2]
    assert torch.equal(c[13:13 + n], j.vis[sel]) and not torch.equal(c[13:13 + n], vis_hs[sel])  # input embeddings
    assert ids_of(c[13 + n:]) == [3, NL] + tok.encode(BOX)
    assert torch.equal(ctxs['final'], c[:base])
    assert j.trace[1] == f'<VISREF_IMG m=3 dpp={n}>'


def test_bundle_contexts_continue_step():
    tok = TOK
    j, eng, _ = bundle_job([(think(tok, 'aaaaaaaa'), 0), (box(tok, '7'), 0.5)])
    eng.step(j)
    eng.step(j)
    assert j.phase == 'think' and j.k == 2
    n = j.steps[0]['n_inject']
    c = j.request().ctx
    assert c.shape[0] == 2 + 8 + 1 + 1 + n + 1 + 1 + len('Wait')
    ids = ids_of(c)
    assert 1 not in ids  # no </think>: the block sits inside the thinking block
    assert ids[:10] == tok.encode('Q?') + tok.encode('a' * 8)
    assert ids[10:12] == [NL, 2] and ids[12 + n:] == [3, NL] + tok.encode('Wait')


def test_bundle_newline_not_doubled():
    tok = TOK
    # answering: always ctx_z </think> '\n' <img>, whatever z ends with; continuing: one newline before <img>
    # closed step whose text already ends with '\n'
    j, eng, _ = bundle_job([(think(tok, 'aaaa\n'), 0), (box(tok, '7'), 0.0)])
    eng.step(j)
    n = j.steps[0]['n_inject']
    ids = ids_of(j.answer_ctx)
    assert ids[:10] == tok.encode('Q?aaaa\n') + [1, NL, 2] and ids[10 + n:] == [3, NL] and ids.count(NL) == 3
    ids = ids_of(j.step_ctx)
    assert ids[:7] == tok.encode('Q?aaaa\n') and ids[7] == 2 and ids[8 + n:] == [3, NL] and ids.count(NL) == 2
    # closed step without a trailing newline
    j, eng, _ = bundle_job([(think(tok, 'aaaa'), 0), (box(tok, '7'), 0.0)])
    eng.step(j)
    ids = ids_of(j.answer_ctx)
    assert ids[:9] == tok.encode('Q?aaaa') + [1, NL, 2] and ids.count(NL) == 2
    ids = ids_of(j.step_ctx)
    assert ids[:8] == tok.encode('Q?aaaa') + [NL, 2] and ids.count(NL) == 2
    # cut step: cut_sep is the newline before <img> when continuing and before </think> when answering
    j, eng, _ = bundle_job([(box(tok, '7'), 0.0)])
    j.consume(Result(ids=tok.encode('abc'), finish='cap'))
    n = j.steps[0]['n_inject']
    assert ids_of(j.step_ctx)[:7] == tok.encode('Q?abc') + [NL, 2] and ids_of(j.step_ctx).count(NL) == 2
    assert ids_of(j.answer_ctx)[:9] == tok.encode('Q?abc') + [NL, 1, NL, 2] and ids_of(j.answer_ctx).count(NL) == 3
    eng.step(j)  # confident probe -> forced answer after the cut
    req = j.request()
    assert j.phase == 'final' and j.forced_answer
    assert req.ctx.shape[0] == j.answer_ctx.shape[0] + len(S1) == 2 + 3 + 1 + 1 + 1 + 1 + n + 1 + 1 + len(S1)
    assert torch.equal(req.ctx[:j.answer_ctx.shape[0]], j.answer_ctx)
    # cut step already ending with '\n': no separator added
    j, eng, _ = bundle_job([])
    j.consume(Result(ids=tok.encode('ab\n'), finish='cap'))
    assert ids_of(j.step_ctx)[:6] == tok.encode('Q?ab\n') + [2] and ids_of(j.step_ctx).count(NL) == 2
    assert ids_of(j.answer_ctx)[:8] == tok.encode('Q?ab\n') + [1, NL, 2] and ids_of(j.answer_ctx).count(NL) == 3
    assert j.trace[0] == 'ab\n'
