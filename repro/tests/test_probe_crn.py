"""Common random numbers for the two stopping probes: the logging-only no-V probe replays the decision probe's state."""
import torch

from test_repro import IN, ScriptEngine, ans, box, ctx, make_job, think, CharTok, lm  # noqa: F401 (lm is a fixture)
from visref.engine import Engine, Request
from visref.methods import Job, PROBE_SAMPLING


class DrawEngine(ScriptEngine):
    """ScriptEngine whose steps consume draws from the request's generator (a real sampler would), and which records
    the state of probe_gen at the 'probe' request and of probe2_gen at the 'probe_nov' request."""

    def __init__(self, tok, scripts):
        super().__init__(tok, scripts)
        self.states = []  # (phase, state of the request's generator before the draws)

    def step(self, job):
        req = job.request()
        if req.gen is not None:
            self.states.append((job.phase, req.gen.get_state()))
            torch.rand(7, generator=req.gen)
        super().step(job)


def make(method, scripts, dual, vis, seed=3, kmax=3):
    job, _, tok = make_job(method, [], kmax=kmax)
    eng = DrawEngine(tok, scripts)
    cfg = dict(job.cfg, dual_probe=int(dual))
    gens = [torch.Generator().manual_seed(s) for s in (seed, seed + 1, seed + 2)]
    j = Job(method, IN, eng.embed(tok.encode('Q?')), vis, cfg, tok, eng, job.sp, gens[0], gens[1],
            probe2_gen=gens[2] if dual else None)
    return j, eng, tok


def scripts_for(tok, dual):
    nov = (box(tok, '3'), 0.5)
    s = [(think(tok, 'ab'), 0), (box(tok, '7'), 0.9)] + ([nov] if dual else []) \
        + [(think(tok, 'c'), 0), (box(tok, '7'), 0.0)] + ([nov] if dual else []) + [(ans(tok, '7'), 0)]
    return s


def test_second_probe_replays_the_decision_probe_state():
    tok = CharTok()
    vis = torch.randn(10, len(tok.vocab))
    j, eng, _ = make('visref_raw_rank', scripts_for(tok, True), True, vis)
    while not j.done:
        eng.step(j)
    probe = [s for ph, s in eng.states if ph == 'probe']
    nov = [s for ph, s in eng.states if ph == 'probe_nov']
    assert len(probe) == len(nov) == 2
    for a, b in zip(probe, nov):
        assert torch.equal(a, b)  # the no-V probe started from the state the decision probe started from
    assert not torch.equal(probe[0], probe[1])  # the decision probe's own state moved on between the steps
    assert j.probe2_gen is not j.probe_gen
    assert all(s['probe_noV_crn'] is True for s in j.out['steps'])


def test_decision_path_is_identical_with_and_without_the_second_probe():
    tok = CharTok()
    vis = torch.randn(10, len(tok.vocab))
    outs, pstates = {}, {}
    for dual in (0, 1):
        j, eng, _ = make('visref_raw_rank', scripts_for(tok, dual), dual, vis)
        while not j.done:
            eng.step(j)
        outs[dual], pstates[dual] = j.out, j.probe_gen.get_state()
    a, b = outs[0], outs[1]
    assert a['k'] == b['k'] == 2
    for key in ('stop_reason', 'final_answer', 'k'):
        assert a[key] == b[key]
    for sa, sb in zip(a['steps'], b['steps']):
        for key in ('H', 'probe_answer'):
            assert sa[key] == sb[key]
        assert 'probe_noV_crn' not in sa and sb['probe_noV_crn'] is True
    assert torch.equal(pstates[0], pstates[1])  # the second probe never consumed probe_gen


def _req(c, g):
    return Request(c, frozenset(), 8, g, True, sampling=PROBE_SAMPLING)


def test_real_sampling_same_state_same_context_gives_same_ids(lm):
    eng = Engine(lm, temperature=0.8)
    g1 = torch.Generator().manual_seed(11)
    torch.rand(5, generator=g1)  # some earlier draws: S is not a fresh seed state
    S = g1.get_state()
    g2 = torch.Generator()
    g2.set_state(S)
    c = ctx(lm, 12, 1)
    a = eng.generate([_req(c, g1)], 1000)[0]
    b = eng.generate([_req(c, g2)], 1000)[0]
    assert len(a.ids) == 8 and a.ids == b.ids
    assert torch.allclose(torch.tensor(a.ent), torch.tensor(b.ent), atol=1e-6)
