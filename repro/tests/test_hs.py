"""visref_hs: Z = last-layer hidden states of the thinking step (EF 2026-10-12)."""
import pytest
import torch

from test_repro import P, ScriptEngine, ans, box, make_job, think, CharTok, lm  # noqa: F401 (lm is a fixture)
from visref.dpp import pick_stats, select_tokens
from visref.engine import Engine
from visref.methods import Job


@pytest.mark.parametrize('attn_budget', [2 ** 28, 300])
def test_last_hidden_matches_full_forward(lm, attn_budget):
    torch.manual_seed(3)
    ctx = torch.randn(40, lm.config.hidden_size)
    eng = Engine(lm, attn_budget=attn_budget)  # 300 forces a chunked prefill
    with torch.no_grad():
        ref = lm(inputs_embeds=ctx[None], output_hidden_states=True).hidden_states[-1][0]  # after the final norm
    for n in (1, 7, 40):
        got = eng.last_hidden(ctx, n)
        assert got.shape == (n, lm.config.hidden_size)
        assert torch.allclose(got, ref[-n:], atol=1e-5)


def test_trace_does_not_change_the_selection():
    torch.manual_seed(0)
    V, Z = torch.randn(200, 32), torch.randn(25, 32)
    for m in (5, 25, 60):
        tr = []
        assert select_tokens(V, Z, m, trace=tr) == select_tokens(V, Z, m)
        assert len(tr) == len(select_tokens(V, Z, m))
        s = pick_stats(tr)
        assert 0 <= s['n_meaningful'] <= len(tr)
        assert s['tail_rel_spearman'] is None or -1 <= s['tail_rel_spearman'] <= 1


def test_contextual_z_lifts_the_rank_limit_of_repeated_tokens():
    """Input embeddings of a step that repeats few distinct tokens give rank(L) <= #distinct tokens; contextual
    (position-dependent) vectors of the same step do not have that limit."""
    torch.manual_seed(1)
    d, T, m = 64, 40, 30
    V = torch.randn(300, d)
    vocab = torch.randn(4, d)
    z_in = vocab[torch.randint(0, 4, (T,))]   # 4 distinct tokens
    z_hs = z_in + torch.randn(T, d)           # context makes every position different
    assert len(select_tokens(V, z_in, m)) <= 4
    assert len(select_tokens(V, z_hs, m)) == m


class HSEngine(ScriptEngine):
    """ScriptEngine with a last_hidden that records its inputs and returns distinct position-dependent rows."""

    def __init__(self, tok, scripts):
        super().__init__(tok, scripts)
        self.hs_calls = []

    def last_hidden(self, ctx, n):
        self.hs_calls.append((ctx.shape[0], n))
        g = torch.Generator().manual_seed(ctx.shape[0])
        return ctx[-n:] + torch.randn(n, ctx.shape[1], generator=g)


def test_hs_job_takes_z_from_last_hidden_and_records_stats():
    tok = CharTok()
    job, _, _ = make_job('visref_hs', [], nvis=10)
    eng = HSEngine(tok, [(think(tok, 'aaaaaaaa'), 0), (box(tok, '7'), 0.0), (ans(tok, '7'), 0)])
    j = Job('visref_hs', job.row, eng.embed(tok.encode('Q?')), job.vis, job.cfg, tok, eng, job.sp,
            torch.Generator(), torch.Generator())
    while not j.done:
        eng.step(j)
    s = j.out['steps'][0]
    assert eng.hs_calls == [(2 + 8, 8)]  # context Q? + z_1 (8 tokens), hidden states of the 8 thinking tokens
    assert s['m'] == 3 and s['n_inject'] == 3  # 8 identical input tokens: input embeddings would give rank 1
    assert s['n_unique_z'] == 1
    assert 'n_meaningful' in s and 'tail_rel_spearman' in s
    assert eng.calls[1][1] == 2 + 8 + 3 + 1 + P  # Q? + z_1 + V(m=3) + </think> + probe prompt


def test_raw_rank_with_the_same_step_is_rank_limited():
    tok = CharTok()
    job, eng, _ = make_job('visref_raw_rank', [(think(tok, 'aaaaaaaa'), 0), (box(tok, '7'), 0.0), (ans(tok, '7'), 0)])
    while not job.done:
        eng.step(job)
    assert job.out['steps'][0]['n_inject'] == 1
