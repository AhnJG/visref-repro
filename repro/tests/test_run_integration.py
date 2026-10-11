import json
import sys
import types

import torch

import run
from test_repro import CharTok
from visref.model import brace_ids


def test_run_end_to_end_and_resume(tmp_path, monkeypatch):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    tok = CharTok()
    torch.manual_seed(0)
    lm = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok.vocab), hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                      head_dim=8, max_position_embeddings=4096)).eval()
    model = types.SimpleNamespace(language_model=lm)
    rows = {str(p): {'pid': str(p), 'query': 'q', 'answer': '5', 'question_type': 'free_form',
                     'answer_type': 'integer', 'choices': None, 'precision': None, 'unit': None,
                     'image': lambda: None} for p in range(1, 5)}

    def enc(model_, tok_, image, q, max_num):
        g = torch.Generator().manual_seed(len(q))
        e = lm.get_input_embeddings()
        return e(torch.randint(5, 90, (12,), generator=g)).detach(), torch.randn(20, 32, generator=g)

    monkeypatch.setattr(run, 'load_mathvista', lambda p: rows)
    monkeypatch.setattr(run, 'load_model', lambda m, r: (model, tok))
    monkeypatch.setattr(run, 'encode_sample', enc)
    monkeypatch.setattr(run, 'special_ids', lambda t, w, p=None: {'think_end': 1, 'eos': 0, 'wait': tok.encode(w),
                                                          'probe_prompt': tok.encode(p),
                                                          **brace_ids(tok.vocab)})
    out = tmp_path / 'r.jsonl'
    base = ['run.py', '--parquet', 'x', '--out', str(out), '--think-cap', '20', '--total-think-cap', '45',
            '--answer-cap', '8', '--kmax', '3', '--batch', '4', '--chunk', '7', '--min-step-tokens', '5', '--batch-tokens', '80', '--methods', 'st,tsr,visref_raw_rank', '--batch', '5']
    monkeypatch.setattr(sys, 'argv', base + ['--limit', '2'])
    run.main()
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert sorted((r['pid'], r['method']) for r in recs) == sorted((p, m) for p in '12' for m in ('st', 'tsr', 'visref_raw_rank'))
    for r in recs:
        assert r['status'] == 'ok' and r['forked'] == (r['method'] != 'st')
        assert r['think_tokens'] <= 45 and 1 <= r['k'] <= 3
        assert r['stop_reason'] in ({'st'} if r['method'] == 'st' else {'entropy', 'kmax', 'budget'})
        if r['method'].startswith('visref'):
            assert all(s['m'] == 6 for s in r['steps'] if s['think_tokens'])  # empty z_k: nothing to select from
        if r['method'] != 'st':  # default probe is the \\boxed{ format
            assert all('probe_valid' in s and len(s['probe_ent']) == s['probe_len'] for s in r['steps'])
    st = {r['pid']: r for r in recs if r['method'] == 'st'}
    tsr = {r['pid']: r for r in recs if r['method'] == 'tsr'}
    for p in st:  # step-1 thinking is shared: identical across all methods
        for r in recs:
            if r['pid'] == p:
                assert r['steps'][0]['think_tokens'] == st[p]['steps'][0]['think_tokens']
                assert r['thinking'].startswith(st[p]['thinking'])
    monkeypatch.setattr(sys, 'argv', base + ['--limit', '3'])
    run.main()
    recs2 = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(recs2) == 9 and recs2[:6] == recs
    # new session: skip pairs finished elsewhere, write to a fresh file
    skip = tmp_path / 'skip.json'
    skip.write_text(json.dumps([[r['pid'], r['method']] for r in recs2]))
    out2 = tmp_path / 'r2.jsonl'
    argv = [x if x != str(out) else str(out2) for x in base]
    monkeypatch.setattr(sys, 'argv', argv + ['--limit', '4', '--skip', str(skip)])
    run.main()
    recs3 = [json.loads(l) for l in out2.read_text().splitlines()]
    assert sorted((r['pid'], r['method']) for r in recs3) == sorted(('4', m) for m in ('st', 'tsr', 'visref_raw_rank'))


def test_deadline_dumps_partial(tmp_path, monkeypatch):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    tok = CharTok()
    lm = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok.vocab), hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                      head_dim=8, max_position_embeddings=4096)).eval()
    rows = {'1': {'pid': '1', 'query': 'q', 'answer': '5', 'question_type': 'free_form', 'answer_type': 'integer',
                  'choices': None, 'precision': None, 'unit': None, 'image': lambda: None}}
    monkeypatch.setattr(run, 'load_mathvista', lambda p: rows)
    monkeypatch.setattr(run, 'load_model', lambda m, r: (types.SimpleNamespace(language_model=lm), tok))
    monkeypatch.setattr(run, 'encode_sample', lambda *a: (lm.get_input_embeddings()(torch.arange(5, 17)).detach(),
                                                          torch.randn(20, 32)))
    monkeypatch.setattr(run, 'special_ids', lambda t, w, p=None: {'think_end': 1, 'eos': 0, 'wait': tok.encode(w),
                                                          'probe_prompt': tok.encode(p),
                                                          **brace_ids(tok.vocab)})
    out = tmp_path / 'r.jsonl'
    monkeypatch.setattr(sys, 'argv', ['run.py', '--parquet', 'x', '--out', str(out), '--think-cap', '400',
                                      '--chunk', '3', '--deadline', '1e-9'])
    run.main()
    assert not out.exists() or not out.read_text()
    part = [json.loads(l) for l in (tmp_path / 'r.jsonl.partial').read_text().splitlines()]
    assert part and all(p['pid'] == '1' for p in part)


def test_pack_respects_padded_budget():
    from visref.engine import Request
    reqs = [Request(torch.zeros(n, 2), frozenset(), 100, None) for n in [50, 10, 30, 20, 90]]
    groups = run.pack(reqs, 10, 100)
    assert sorted(i for g in groups for i in g) == list(range(5))
    for g in groups:
        assert len(g) == 1 or len(g) * max(reqs[i].ctx.shape[0] + 10 for i in g) <= 100
