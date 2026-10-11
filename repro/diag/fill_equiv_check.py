"""CPU equivalence check for the visref_fill / dual-probe change (EF 2026-10-11).

Runs run.main() end to end with a tiny random Qwen3 and fake encoder (as tests/test_run_integration.py) from a
given code root, and writes the result rows. Compare roots / options with --compare.

  python3 diag/fill_equiv_check.py run ROOT OUT.jsonl METHODS [extra run.py args...]
  python3 diag/fill_equiv_check.py all ORIG_ROOT NEW_ROOT WORKDIR
"""
import json
import subprocess
import sys
import types
from pathlib import Path

VOLATILE = {'code_version', 'wall_s', 'gen_tokens_noV'}
NOV = ('H_noV', 'probe_noV_finish', 'probe_noV_answer', 'probe_noV_inner', 'probe_noV_ent')
FILL = ('n_head', 'n_fill', 'ridge', 'tail_rel_spearman')


def run_one(root, out, methods, extra):
    sys.path[:0] = [str(root), str(Path(root) / 'tests')]
    import torch
    import run
    from test_repro import CharTok
    from visref.model import brace_ids
    from transformers import Qwen3Config, Qwen3ForCausalLM
    tok = CharTok()
    torch.manual_seed(0)
    lm = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok.vocab), hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                      head_dim=8, max_position_embeddings=4096)).eval()
    model = types.SimpleNamespace(language_model=lm)
    rows = {str(p): {'pid': str(p), 'query': 'q' * p, 'answer': '5', 'question_type': 'free_form',
                     'answer_type': 'integer', 'choices': None, 'precision': None, 'unit': None,
                     'image': lambda: None} for p in range(1, 9)}

    def enc(model_, tok_, image, q, max_num):
        g = torch.Generator().manual_seed(len(q))
        e = lm.get_input_embeddings()
        return e(torch.randint(5, 90, (12,), generator=g)).detach(), torch.randn(100, 32, generator=g)

    run.load_mathvista = lambda p: rows
    run.load_model = lambda m, r: (model, tok)
    run.encode_sample = enc
    run.special_ids = lambda t, w, p=None: {'think_end': 1, 'eos': 0, 'wait': tok.encode(w),
                                            'probe_prompt': tok.encode(p), **brace_ids(tok.vocab)}
    sys.argv = ['run.py', '--parquet', 'x', '--out', str(out), '--think-cap', '40', '--total-think-cap', '120',
                '--answer-cap', '12', '--kmax', '4', '--batch', '6', '--chunk', '9', '--min-step-tokens', '5',
                '--batch-tokens', '400', '--delta', '2.0', '--methods', methods, '--limit', '8'] + extra
    run.main()


def load(p):
    return {(r['pid'], r['method']): r for r in map(json.loads, Path(p).read_text().splitlines())}


def strip(r, drop):
    r = {k: v for k, v in r.items() if k not in VOLATILE}
    r['steps'] = [{k: v for k, v in s.items() if k not in drop} for s in r['steps']]
    return r


def same(a, b, drop=()):
    A, B = load(a), load(b)
    assert A.keys() == B.keys(), (sorted(A), sorted(B))
    bad = [k for k in A if strip(A[k], drop) != strip(B[k], drop)]
    return len(A), bad


def main():
    if sys.argv[1] == 'run':
        run_one(Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5:])
        return
    orig, new, wd = map(Path, sys.argv[2:5])
    wd.mkdir(parents=True, exist_ok=True)

    def go(root, name, methods, *extra):
        out = wd / f'{name}.jsonl'
        out.unlink(missing_ok=True)
        subprocess.run([sys.executable, __file__, 'run', str(root), str(out), methods, *extra], check=True,
                       capture_output=True)
        return out

    three = 'st,tsr,visref_raw_rank'
    res = {}
    o = go(orig, 'orig_3', three)
    n = go(new, 'new_3', three)
    res['A_existing_methods_orig_vs_new'] = same(o, n)
    nd = go(new, 'new_3_dual', three, '--dual-probe', '1')
    res['B_existing_methods_dual_probe_on_vs_off(ignoring H_noV fields)'] = same(n, nd, NOV)
    f = go(new, 'new_fill', 'visref_fill')
    fd = go(new, 'new_fill_dual', 'visref_fill', '--dual-probe', '1')
    res['C_fill_dual_probe_on_vs_off(ignoring H_noV fields)'] = same(f, fd, NOV)
    # D: fill forked from the same step 1 as raw_rank: head = raw picks, exactly m injected
    both = go(new, 'new_raw_fill', 'visref_raw_rank,visref_fill')
    B = load(both)
    d = {'n': 0, 'head_eq_raw_n_dpp': 0, 'inject_eq_m': 0, 'fill_added': 0}
    for (p, m), r in B.items():
        if m != 'visref_fill':
            continue
        raw = B[(p, 'visref_raw_rank')]
        s, sr = r['steps'][0], raw['steps'][0]
        if not s.get('m'):
            continue
        d['n'] += 1
        d['head_eq_raw_n_dpp'] += s['n_head'] == sr['n_dpp']
        d['inject_eq_m'] += s['n_inject'] == s['m']
        d['fill_added'] += s['n_fill'] > 0
    res['D_fill_step1_vs_raw'] = d
    dual = load(fd)
    res['E_dual_rows_with_H_noV_all_steps'] = sum(all('H_noV' in s for s in r['steps'] if 'H' in s) for r in dual.values())
    res['E_dual_rows'] = len(dual)
    print(json.dumps(res, indent=1))
    ok = (not res['A_existing_methods_orig_vs_new'][1] and not res['B_existing_methods_dual_probe_on_vs_off(ignoring H_noV fields)'][1]
          and not res['C_fill_dual_probe_on_vs_off(ignoring H_noV fields)'][1] and d['n'] and d['head_eq_raw_n_dpp'] == d['n']
          and d['inject_eq_m'] == d['n'] and res['E_dual_rows_with_H_noV_all_steps'] == res['E_dual_rows'])
    print('PASS' if ok else 'FAIL')


if __name__ == '__main__':
    main()
