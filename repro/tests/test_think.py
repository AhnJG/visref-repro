"""Thinking after the stop decision ('tsr_think', 'visref_think', 'visref_hs_think') and the stored step 1
(EF 2026-10-13: cue 'Wait, think more', up to think_extra = 512 tokens outside the thinking budget, any stop reason)."""
import json
import sys
import types

import pytest
import torch

import run
from test_hs import HSEngine
from test_repro import IN, P, S1, CharTok, ans, box, make_job, think
from visref.engine import Result
from visref.methods import THINK, Job, stored_step1
from visref.model import brace_ids
from visref.scoring import score

W = len('Wait')  # make_job's wait_text


def drive(job, eng):
    while not job.done:
        eng.step(job)
    return [c[0] for c in eng.calls]


def test_tsr_think_thinks_once_more_after_an_entropy_stop():
    tok = CharTok()
    job, eng, _ = make_job('tsr_think', [(think(tok, 'abc'), 0), (box(tok, '7'), 0.05),
                                         (think(tok, 'de'), 0), (ans(tok, '\n1000'), 0)], think_extra=40)
    assert drive(job, eng) == ['think', 'probe', 'extra', 'final']
    assert eng.calls[2][1] == 2 + 3 + W          # Q? + abc + cue (the stopping probe is not in the context)
    assert eng.calls[3][1] == 2 + 3 + W + 2 + 1  # ... + de + </think>, natural answer (no answer prompt)
    o = job.out
    assert o['stop_reason'] == 'entropy' and o['k'] == 1 and o['thinking'] == 'abc<EXTRA>Waitde'
    assert (o['extra_think_cap'], o['extra_think_tokens'], o['extra_think_end']) == (40, 2, 'close')
    assert o['think_tokens'] == 3  # the extra thinking is outside the thinking budget
    assert o['final_answer'] == '\n1000' and not o['final_forced_answer'] and score(o['final_answer'], IN)['correct']


def test_extra_request_is_ordinary_thinking():
    tok = CharTok()
    job, eng, _ = make_job('tsr_think', [(think(tok, 'abc'), 0), (box(tok, '7'), 0.05)], think_extra=40)
    eng.step(job)
    eng.step(job)
    req = job.request()
    assert job.phase == 'extra' and req.gen is job.gen and req.sampling is None and not req.record_entropy
    assert req.max_new == 40 and req.stop_ids == frozenset({0, 1})  # EOS or </think>


def test_visref_think_cue_comes_after_the_injected_tokens():
    tok = CharTok()
    job, eng, _ = make_job('visref_think', [(think(tok, 'ab'), 0), (box(tok, '7'), 0.05),
                                            (think(tok, 'c'), 0), (ans(tok, '7'), 0)], think_extra=40)
    assert drive(job, eng) == ['think', 'probe', 'extra', 'final']
    n1 = job.out['steps'][0]['n_inject']
    assert n1 == 2 and eng.calls[2][1] == 2 + 2 + n1 + W  # Q? + ab + V_1 + cue
    assert eng.calls[3][1] == 2 + 2 + n1 + W + 1 + 1       # ... + c + </think>; no injection after the extra thinking
    assert job.out['thinking'] == 'ab<VISREF m=3 dpp=2><EXTRA>Waitc'


def test_extra_cut_by_its_cap_gets_separator_and_the_s1_answer_prompt():
    tok = CharTok()
    job, eng, _ = make_job('tsr_think', [(think(tok, 'abc'), 0), (box(tok, '7'), 0.05)], think_extra=5)
    eng.step(job)
    eng.step(job)
    job.consume(Result(ids=tok.encode('vwxyz'), finish='cap'))
    req = job.request()
    assert job.phase == 'final' and job.forced_answer
    assert req.ctx.shape[0] == 2 + 3 + W + 5 + 1 + 1 + len(S1)  # ... + vwxyz + '\n' + </think> + 'Final Answer:'
    job.consume(Result(ids=ans(tok, ' 1000'), finish='stop'))
    o = job.out
    assert o['extra_think_end'] == 'cap' and o['extra_think_tokens'] == 5 and o['final_answer'] == S1 + ' 1000'
    assert o['thinking'] == 'abc<EXTRA>Waitvwxyz\n'


def test_extra_ending_with_eos_gets_a_natural_answer():
    tok = CharTok()
    job, eng, _ = make_job('tsr_think', [(think(tok, 'abc'), 0), (box(tok, '7'), 0.05)], think_extra=40)
    eng.step(job)
    eng.step(job)
    job.consume(Result(ids=tok.encode('xy') + [0], finish='stop'))
    req = job.request()
    assert job.phase == 'final' and not job.forced_answer and req.ctx.shape[0] == 2 + 3 + W + 2 + 1
    job.consume(Result(ids=ans(tok, '7'), finish='stop'))
    assert job.out['extra_think_end'] == 'eos' and job.out['extra_think_tokens'] == 2


@pytest.mark.parametrize('reason, over', [('kmax', dict(kmax=1)), ('budget', dict(min_step_tokens=5, total_think_cap=6))])
def test_every_stop_reason_gets_the_extra_thinking(reason, over):
    tok = CharTok()
    job, eng, _ = make_job('tsr_think', [(think(tok, 'abc'), 0), (box(tok, '7'), 0.9),  # H 1.8 > delta: not confident
                                         (think(tok, 'd'), 0), (ans(tok, '7'), 0)], think_extra=40, **over)
    assert drive(job, eng) == ['think', 'probe', 'extra', 'final']
    assert job.out['stop_reason'] == reason and job.out['extra_think_end'] == 'close'


def test_cut_last_step_keeps_its_separator_before_the_cue():
    tok = CharTok()
    job, eng, _ = make_job('tsr_think', [], kmax=1, think_extra=40)
    job.consume(Result(ids=tok.encode('abc'), finish='cap'))  # cut step: 'abc' + '\n'
    job.consume(Result(ids=box(tok, '7'), ent=[0.9] * 3, ent_lump=[0.9] * 3, finish='stop'))
    assert job.phase == 'extra' and job.request().ctx.shape[0] == 2 + 3 + 1 + W
    job.consume(Result(ids=think(tok, 'de'), finish='stop'))  # the extra thinking closes: natural answer
    assert job.phase == 'final' and not job.forced_answer and job.request().ctx.shape[0] == 2 + 3 + 1 + W + 2 + 1


def test_no_room_in_the_context_skips_the_extra_thinking():
    tok = CharTok()
    job, eng, _ = make_job('tsr_think', [(think(tok, 'abc'), 0), (box(tok, '7'), 0.05), (ans(tok, '7'), 0)],
                           think_extra=40)
    room = job._answer_room()
    job, eng, _ = make_job('tsr_think', [(think(tok, 'abc'), 0), (box(tok, '7'), 0.05), (ans(tok, '7'), 0)],
                           think_extra=40, max_context=2 + 3 + W + room + 1)
    assert drive(job, eng) == ['think', 'probe', 'final']
    assert job.out['extra_think_end'] == 'skipped' and job.out['extra_think_cap'] == 0
    assert eng.calls[2][1] == 2 + 3 + 1  # answered as TSR would


def test_base_methods_have_no_extra_thinking():
    tok = CharTok()
    for m in ('tsr', 'visref_raw_rank'):
        job, eng, _ = make_job(m, [(think(tok, 'abc'), 0), (box(tok, '7'), 0.05), (ans(tok, '7'), 0)], think_extra=40)
        assert drive(job, eng) == ['think', 'probe', 'final'] and 'extra_think_end' not in job.out
    assert THINK == ('tsr_think', 'visref_think', 'visref_hs_think')


def test_visref_hs_think_uses_last_hidden_only_for_the_steps():
    tok = CharTok()
    job, _, _ = make_job('visref_hs_think', [], nvis=10, think_extra=40)
    eng = HSEngine(tok, [(think(tok, 'aaaaaaaa'), 0), (box(tok, '7'), 0.0), (think(tok, 'b'), 0), (ans(tok, '7'), 0)])
    j = Job('visref_hs_think', job.row, eng.embed(tok.encode('Q?')), job.vis, job.cfg, tok, eng, job.sp,
            torch.Generator(), torch.Generator())
    assert drive(j, eng) == ['think', 'probe', 'extra', 'final']
    assert eng.hs_calls == [(2 + 8, 8)] and j.out['steps'][0]['n_inject'] == 3
    assert eng.calls[2][1] == 2 + 8 + 3 + W


# ---------------------------------------------------------------- stored step 1 --
SP = {'think_end': 1, 'eos': 0}


@pytest.mark.parametrize('s, z, finish, match', [
    ({'thinking': 'abc', 'think_tokens': 3, 'think_end': 'close'}, 'abc', 'stop', True),
    ({'thinking': 'abc\n', 'think_tokens': 3, 'think_end': 'cap'}, 'abc', 'cap', True),    # separator split off
    ({'thinking': 'ab\n', 'think_tokens': 3, 'think_end': 'cap'}, 'ab\n', 'cap', True),    # text ended with '\n'
    ({'thinking': 'xy', 'think_tokens': 2, 'think_end': 'eos'}, 'xy', 'stop', True),
    ({'thinking': 'abc', 'think_tokens': 5, 'think_end': 'close'}, 'abc', 'stop', False),  # length differs: recorded
])
def test_stored_step1_ids(s, z, finish, match):
    tok = CharTok()
    ids, fin, info = stored_step1(tok, SP, s, '\n')
    tail = [1] if s['think_end'] == 'close' else [0] if s['think_end'] == 'eos' else []
    assert ids == tok.encode(z) + tail and fin == finish
    assert info == {'step1_source': 'stored', 'step1_tokens_match': match, 'step1_n_z': len(z),
                    'step1_think_tokens_stored': s['think_tokens']}


class NLTok(CharTok):
    """CharTok with one merged token for '\\n\\n' (as real tokenizers have), to test the cut-step separator."""

    def __init__(self):
        super().__init__()
        self.vocab = self.vocab + ['\n\n']

    def encode(self, s, add_special_tokens=False):
        out, i = [], 0
        while i < len(s):
            if s[i:i + 2] == '\n\n':
                out.append(len(self.vocab) - 1)
                i += 2
            else:
                out.append(self.vocab.index(s[i]))
                i += 1
        return out


def test_stored_cut_step_ending_in_two_newlines_keeps_both():
    """2026-10-13 review: z_1 = 'ab' + '\\n\\n' (3 tokens) cut at the cap got no separator (it ends with '\\n'); splitting
    off the last '\\n' as a separator would give 'ab\\n', also 3 tokens, and lose a newline."""
    tok = NLTok()
    ids, fin, info = stored_step1(tok, SP, {'thinking': 'ab\n\n', 'think_tokens': 3, 'think_end': 'cap'}, '\n')
    assert tok.decode(ids) == 'ab\n\n' and fin == 'cap' and info['step1_tokens_match']
    ids, _, info = stored_step1(tok, SP, {'thinking': 'ab\n', 'think_tokens': 2, 'think_end': 'cap'}, '\n')
    assert tok.decode(ids) == 'ab' and info['step1_tokens_match']  # separator added after 'ab' is split off


@pytest.mark.parametrize('end, text', [('close', 'abc'), ('cap', 'abc'), ('eos', 'abc'), ('cap', 'ab\n\n'), ('cap', 'a\n')])
def test_job_from_stored_step1_equals_job_that_generated_it(end, text):
    tok = NLTok()
    gen_ids = tok.encode(text) + {'close': [1], 'eos': [0], 'cap': []}[end]
    fin = 'cap' if end == 'cap' else 'stop'
    ref, eng, _ = make_job('visref_think', [], kmax=2, think_extra=40)
    ref.tok, eng.tok = tok, tok  # one-hot width = len(NLTok.vocab)
    ref = Job('visref_think', ref.row, eng.embed(tok.encode('Q?')), torch.randn(10, len(tok.vocab)), ref.cfg, tok, eng,
              ref.sp, torch.Generator(), torch.Generator())
    ref.consume(Result(ids=gen_ids, finish=fin))
    stored = {'thinking': ref.trace[0], 'think_tokens': ref.steps[0]['think_tokens'], 'think_end': end}
    ids, f, info = stored_step1(tok, ref.sp, stored, '\n')
    j = Job('visref_think', ref.row, ref.ctx, ref.vis, ref.cfg, tok, eng, ref.sp, torch.Generator(), torch.Generator(),
            step1=(ids, f, torch.Generator().get_state()))
    assert info['step1_tokens_match'] and j.forked and j.phase == ref.phase == 'probe'
    assert torch.equal(j.request().ctx, ref.request().ctx) and j.steps == ref.steps and j.trace == ref.trace


# ---------------------------------------------------------------- run.py --step1-from --
def _patch(monkeypatch, tok, lm):
    rows = {str(p): {'pid': str(p), 'query': 'q' * p, 'answer': '5', 'question_type': 'free_form',
                     'answer_type': 'integer', 'choices': None, 'precision': None, 'unit': None,
                     'image': lambda: None} for p in range(1, 5)}

    def enc(model_, tok_, image, q, max_num):
        g = torch.Generator().manual_seed(len(q))
        return lm.get_input_embeddings()(torch.randint(5, 90, (12,), generator=g)).detach(), torch.randn(20, 32, generator=g)

    monkeypatch.setattr(run, 'load_mathvista', lambda p: rows)
    monkeypatch.setattr(run, 'load_model', lambda m, r: (types.SimpleNamespace(language_model=lm), tok))
    monkeypatch.setattr(run, 'encode_sample', enc)
    monkeypatch.setattr(run, 'special_ids', lambda t, w, p=None: {'think_end': 1, 'eos': 0, 'wait': tok.encode(w),
                                                                  'probe_prompt': tok.encode(p), **brace_ids(tok.vocab)})


def test_run_from_stored_step1(tmp_path, monkeypatch):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    sys.path.insert(0, str(run.Path(run.__file__).resolve().parent / 'diag'))
    import make_step1_store
    tok = CharTok()
    torch.manual_seed(0)
    lm = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok.vocab), hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                      head_dim=8, max_position_embeddings=4096)).eval()
    _patch(monkeypatch, tok, lm)
    base = ['run.py', '--parquet', 'x', '--think-cap', '20', '--total-think-cap', '45', '--answer-cap', '8',
            '--kmax', '3', '--chunk', '7', '--min-step-tokens', '5', '--batch-tokens', '80', '--batch', '6',
            '--think-extra', '9']
    first = tmp_path / 'main.jsonl'
    monkeypatch.setattr(sys, 'argv', base + ['--out', str(first), '--methods', 'st', '--limit', '2'])
    run.main()
    store = tmp_path / 'store.json'
    store.write_text(json.dumps({'pids': make_step1_store.build([first])}))
    st = {r['pid']: r for r in map(json.loads, first.read_text().splitlines())}

    out = tmp_path / 'think.jsonl'
    monkeypatch.setattr(sys, 'argv', base + ['--out', str(out), '--methods', ','.join(THINK), '--limit', '2',
                                             '--step1-from', str(store)])
    run.main()
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert sorted((r['pid'], r['method']) for r in recs) == sorted((p, m) for p in '12' for m in THINK)
    for r in recs:
        s = st[r['pid']]
        assert r['status'] == 'ok' and r['forked'] and r['step1_info']['step1_tokens_match']
        assert r['step1_info']['step1_store_sha'] == run.hashlib.sha256(store.read_bytes()).hexdigest()[:12]
        assert r['steps'][0]['think_tokens'] == s['steps'][0]['think_tokens']
        assert r['steps'][0]['think_end'] == s['steps'][0]['think_end'] and r['thinking'].startswith(s['thinking'])
        assert r['stop_reason'] in {'entropy', 'kmax', 'budget'} and r['extra_think_cap'] <= 9
        assert r['extra_think_end'] in {'close', 'eos', 'cap', 'skipped'} and '<EXTRA>Wait, think more' in r['thinking']

    monkeypatch.setattr(sys, 'argv', base + ['--out', str(tmp_path / 'x.jsonl'), '--methods', 'tsr_think',
                                             '--limit', '3', '--step1-from', str(store)])
    with pytest.raises(AssertionError, match='no stored step 1'):
        run.main()
