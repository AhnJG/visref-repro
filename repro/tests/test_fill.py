"""visref_fill and the logging-only dual probe (EF 2026-10-11)."""
import pytest
import torch

from test_repro import BOX, IN, P, ScriptEngine, ans, box, make_job, think, CharTok
from visref.dpp import select_tokens, select_tokens_fill
from visref.methods import Job


def make_dual(method, scripts, kmax=3, nvis=10, **over):
    job, eng, tok = make_job(method, [], kmax=kmax, nvis=nvis, **over)  # reuse cfg/sp/vis construction
    cfg = dict(job.cfg, dual_probe=1)
    eng = ScriptEngine(tok, scripts)
    j = Job(method, IN, eng.embed(tok.encode('Q?')), job.vis, cfg, tok, eng, job.sp, torch.Generator(),
            torch.Generator(), probe2_gen=torch.Generator())
    return j, eng, tok


def test_fill_keeps_greedy_head_and_reaches_m():
    torch.manual_seed(0)
    V, Z = torch.randn(300, 48), torch.randn(20, 48)  # rank(V Z^T) <= 20 < m
    for m in (5, 20, 90):
        head = select_tokens(V, Z, m)
        sel, info = select_tokens_fill(V, Z, m)
        assert set(head) <= set(sel) and len(sel) == m and len(set(sel)) == m and sel == sorted(sel)
        assert info['n_head'] == len(head) and info['n_head'] + info['n_fill'] == m
    assert info['ridge'] > 0 and -1 <= info['tail_rel_spearman'] <= 1


def test_fill_with_repeated_tokens_and_tiny_image():
    torch.manual_seed(1)
    V = torch.randn(30, 16)
    Z = torch.randn(1, 16).repeat(50, 1)  # one distinct token: rank 1
    sel, info = select_tokens_fill(V, Z, 9)
    assert info['n_head'] == 1 and len(sel) == 9


def test_fill_job_injects_m_after_head():
    tok = CharTok()
    job, eng, _ = make_job('visref_fill', [(think(tok, 'ab'), 0), (box(tok, '7'), 0.0), (ans(tok, '7'), 0)])
    while not job.done:
        eng.step(job)
    s = job.out['steps'][0]
    assert s['m'] == 3 and s['n_head'] == 2 and s['n_fill'] == 1 and s['n_inject'] == 3
    assert eng.calls[1][1] == 2 + 2 + 3 + 1 + P  # Q? + ab + V(m=3) + </think> + probe prompt


@pytest.mark.parametrize('method', ['visref_raw_rank', 'visref_fill'])
@pytest.mark.parametrize('h_main,h_nov', [(0.9, 0.0), (0.0, 0.9)])
def test_dual_probe_logs_without_v_and_never_changes_the_decision(method, h_main, h_nov):
    tok = CharTok()
    nov = (box(tok, '3'), h_nov)
    if h_main > 0.25:  # continue to step 2, then stop
        plain = [(think(tok, 'ab'), 0), (box(tok, '7'), h_main), (think(tok, 'c'), 0), (box(tok, '7'), 0.0),
                 (ans(tok, '7'), 0)]
        dual = [plain[0], plain[1], nov, plain[2], plain[3], nov, plain[4]]
    else:  # stop at k = 1
        plain = [(think(tok, 'ab'), 0), (box(tok, '7'), h_main), (ans(tok, '7'), 0)]
        dual = [plain[0], plain[1], nov, plain[2]]
    a, ea, _ = make_job(method, list(plain))
    while not a.done:
        ea.step(a)
    b, eb, _ = make_dual(method, list(dual))
    b.vis = a.vis  # same image tokens
    while not b.done:
        eb.step(b)
    hn = [s.pop('H_noV') for s in b.out['steps'] if 'H_noV' in s]
    for s in b.out['steps']:
        for k in [k for k in s if k.startswith('probe_noV')]:
            s.pop(k)
    gen_nov = b.out.pop('gen_tokens_noV')
    assert a.out == b.out  # decisions, contexts, answers identical
    assert len(hn) == a.out['k'] and gen_nov == 3 * a.out['k']
    assert hn[0] == pytest.approx(2 * h_nov + h_nov)  # '{', '3' full + '}' merged (same h)
    # the second probe sees z_k without V_k: Q? + ab + </think> + probe prompt
    calls_nov = [c for c in eb.calls if c[0] == 'probe_nov']
    assert calls_nov[0][1] == 2 + 2 + 1 + P


def test_dual_probe_uses_its_own_generator_and_is_off_for_tsr():
    tok = CharTok()
    j, eng, _ = make_dual('visref_fill', [(think(tok, 'ab'), 0), (box(tok, '7'), 0.0)])
    eng.step(j)
    assert j.phase == 'probe' and j.request().gen is j.probe_gen
    eng.step(j)
    assert j.phase == 'probe_nov' and j.request().gen is j.probe2_gen and j.probe2_gen is not j.probe_gen
    t, _, _ = make_dual('tsr', [])
    assert not t.dual


def test_failed_dual_probe_keeps_the_decision():
    tok = CharTok()
    j, eng, _ = make_dual('visref_fill', [(think(tok, 'ab'), 0), (box(tok, '7'), 0.0), (ans(tok, '7'), 0)])
    eng.step(j)
    eng.step(j)
    assert j.phase == 'probe_nov'
    j.nov_failed()
    assert j.phase == 'final' and j.steps[-1]['probe_noV_finish'] == 'error'
    eng.step(j)
    assert j.done and j.out['stop_reason'] == 'entropy'
