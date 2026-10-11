"""step1_branch -- pre-registered diagnostic (EXPERIMENT_FLOW 2026-10-11 "사전등록: step1_branch").

From the saved b_main step-1 thinking z_1 (shared by ST/TSR/VisRef through the fork), four conditions per item; each
ends with the stopping-probe style answer ("\n\n\\boxed", stops when the box closes, <= probe_cap tokens):
  A  [z_1][sep] </think> probe                     no reinjection, immediate answer
  B  [z_1][sep][V_1] </think> probe                reinjection, immediate answer
  C  [z_1][sep] + free thinking (<= think_extra), then </think> (the model's own if emitted) + probe
  D  [z_1][sep][V_1] + the same free thinking, then </think> + probe
V_1 = visref.dpp.select_tokens(V, embed(z_1), floor(vis_frac*|V|)) recomputed with the b_main code. Answers: n_t1
samples at T=1 (no top-k/p) + 1 greedy, the same seeds in every condition; the free thinking of C and D shares one seed
(engine defaults T 0.6 / top-k 50 / top-p 0.95). Rows keep z ids, selected indices and reconstruction checks against
b_main (think_tokens, n_dpp) so the inputs can be replayed exactly later.

  python3 diag/step1_branch.py --parquet PQ --samples diag/step1_branch_samples.json --ids diag/step1_branch_ids.json \
      --out results/step1_branch_0.jsonl [--limit 20] [--shard 0/1] [--think-extra 512]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from visref.dpp import budget, select_tokens  # noqa: E402
from visref.engine import Engine, Request  # noqa: E402
from visref.scoring import SCORER_VERSION, _boxed, score  # noqa: E402

CONDS = ('A', 'B', 'C', 'D')
DEFAULTS = dict(vis_frac=0.3, think_extra=512, probe_prompt='\n\n\\boxed', probe_cap=16, n_t1=3, temperature=0.6,
                top_k=50, top_p=0.95, seed=42, cut_sep='\n', chunk=1024, batch_tokens=72000, group=8, max_num=12)
T1, GREEDY = (1.0, 0, 1.0), (0.0, 0, 1.0)


def code_version():
    h = hashlib.sha256()
    for f in sorted([Path(__file__).resolve(), ROOT / 'run.py', *(ROOT / 'visref').glob('*.py')]):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:12]


def seed_of(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str, parts)).encode()).digest()[:8], 'little') & ((1 << 63) - 1)


def split_z(tok, thinking, think_tokens, think_end, cut_sep):
    """b_main stored decode(z_1) + sep (sep only after a cut step whose text did not already end with it). Return
    (z ids, sep ids, info), choosing the split whose token count equals b_main's think_tokens when one does."""
    full = tok.encode(thinking, add_special_tokens=False)
    cands = []
    if think_end == 'cap' and cut_sep and thinking.endswith(cut_sep):
        cands.append((tok.encode(thinking[:-len(cut_sep)], add_special_tokens=False),
                      tok.encode(cut_sep, add_special_tokens=False)))
    cands.append((full, []))
    pick = next(((z, s) for z, s in cands if len(z) == think_tokens), cands[0])
    z, sep = pick
    return list(z), list(sep), {'n_full': len(full), 'n_z': len(z), 'n_sep': len(sep), 'tokens_match': len(z) == think_tokens}


def box_close_index(ids, brace_text):
    """Index of the token that closes the box opened after a prompt ending in '\\boxed' (same rule as methods.py)."""
    depth, opened = 0, False
    for n, t in enumerate(ids):
        for ch in brace_text.get(t, ''):
            if ch == '{':
                depth, opened = depth + 1, True
            elif ch == '}' and opened:
                depth -= 1
            if opened and depth == 0:
                return n
    return None


def answer_record(tok, sp, ids, row):
    ids = [t for t in ids if t != sp['eos']]
    close = box_close_index(ids, sp['brace_text'])
    if close is not None:
        ids = ids[:close + 1]
    text = '\\boxed' + tok.decode(ids)
    b = _boxed(text)
    inner = text[b[0]:b[1]] if b else ''
    valid = close is not None and any(c.isalnum() for c in inner)
    return {'text': tok.decode(ids), 'valid': valid, 'correct': int(valid and score(text, row)['correct'])}


def close_thinking(tok, sp, ids, finish, cut_sep):
    """Ids to append after free thinking so the answer follows a </think> (the model's own when it emitted one)."""
    te, eos = sp['think_end'], sp['eos']
    if finish == 'stop' and ids and ids[-1] == te:
        return list(ids), 'close'
    if finish == 'stop' and ids and ids[-1] == eos:
        return list(ids[:-1]) + [te], 'eos'
    sep = tok.encode(cut_sep, add_special_tokens=False) if cut_sep and not tok.decode(ids).endswith(cut_sep) else []
    return list(ids) + sep + [te], 'cap'


def make_generate(eng, cfg, pack):
    def gen_safe(rs):
        states = [r.gen.get_state() if r.gen is not None else None for r in rs]
        try:
            return eng.generate(rs, cfg['chunk'])
        except (torch.OutOfMemoryError, FloatingPointError) as e:
            if len(rs) == 1:
                if isinstance(e, FloatingPointError):
                    return [None]
                raise
            for r, s in zip(rs, states):
                if s is not None:
                    r.gen.set_state(s)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            h = len(rs) // 2
            return gen_safe(rs[:h]) + gen_safe(rs[h:])

    def generate(reqs):
        out = [None] * len(reqs)
        for g in pack(reqs, cfg['chunk'], cfg['batch_tokens']):
            if g:
                for i, r in zip(g, gen_safe([reqs[i] for i in g])):
                    out[i] = r
        return out
    return generate


def run_group(items, eng, tok, sp, cfg, generate):
    """items: dicts with pid, row, base [L,d], vis [N,d], z, sep. Adds the per-condition records in place."""
    te, eos = sp['think_end'], sp['eos']
    dev = eng.emb.weight.device
    for it in items:
        zc = torch.cat([it['base'], eng.embed(it['z'] + it['sep'])], 0)
        it['m'] = budget(it['vis'].shape[0], cfg['vis_frac'])
        it['sel'] = select_tokens(it['vis'], eng.embed(it['z']), it['m'])
        it['ctx'] = {'A': zc, 'B': torch.cat([zc, it['vis'][it['sel']]], 0)}
    reqs, keys = [], []
    for it in items:  # free thinking: C continues A's prefix, D continues B's prefix, one shared seed
        for c, src in (('C', 'A'), ('D', 'B')):
            g = torch.Generator(device=dev).manual_seed(seed_of(cfg['seed'], it['pid'], 'think'))
            reqs.append(Request(it['ctx'][src], frozenset({te, eos}), cfg['think_extra'], g))
            keys.append((it, c, src))
    for (it, c, src), r in zip(keys, generate(reqs)):
        if r is None:
            it.setdefault('extra', {})[c] = {'error': 'non-finite logits'}
            continue
        tail, end = close_thinking(tok, sp, r.ids, r.finish, cfg['cut_sep'])
        body = [t for t in r.ids if t not in (te, eos)]
        it.setdefault('extra', {})[c] = {'end': end, 'len': len(body), 'text': tok.decode(body), 'ids': list(r.ids)}
        it['ctx'][c] = torch.cat([it['ctx'][src], eng.embed(tail)], 0)
    for it in items:
        for c in ('A', 'B'):
            it['ctx'][c] = torch.cat([it['ctx'][c], eng.embed([te])], 0)
    reqs, keys = [], []
    stop = lambda ids, bt=sp['brace_text']: box_close_index(ids, bt) is not None  # noqa: E731
    for it in items:
        for c in CONDS:
            if c not in it['ctx']:
                continue
            ctx = torch.cat([it['ctx'][c], eng.embed(sp['probe_prompt'])], 0)
            for j in range(cfg['n_t1'] + 1):
                greedy = j == cfg['n_t1']
                g = None if greedy else torch.Generator(device=dev).manual_seed(seed_of(cfg['seed'], it['pid'], 'ans', j))
                reqs.append(Request(ctx, frozenset({eos}), cfg['probe_cap'], g, sampling=GREEDY if greedy else T1,
                                    stop_check=stop))
                keys.append((it, c, j))
    for (it, c, j), r in zip(keys, generate(reqs)):
        rec = answer_record(tok, sp, r.ids, it['row']) if r is not None else {'text': '', 'valid': False, 'correct': 0, 'error': True}
        rec['greedy'] = j == cfg['n_t1']
        it.setdefault('answers', {}).setdefault(c, []).append(rec)


def row_out(it, cfg, version):
    acc = {c: sum(a['correct'] for a in it['answers'][c] if not a['greedy']) / cfg['n_t1'] for c in it.get('answers', {})}
    greedy = {c: next(a['correct'] for a in it['answers'][c] if a['greedy']) for c in it.get('answers', {})}
    return {'pid': it['pid'], 'code_version': version, 'scorer_version': SCORER_VERSION, 'think_extra': cfg['think_extra'],
            'n_t1': cfg['n_t1'], 'n_visual': int(it['vis'].shape[0]), 'm': it['m'], 'n_sel': len(it['sel']),
            'sel': it['sel'], 'z_ids': it['z'], 'sep_ids': it['sep'],
            'recon': {**it['recon'], 'n_dpp_bmain': it['sample'].get('n_dpp_bmain'),
                      'sel_match': len(it['sel']) == it['sample'].get('n_dpp_bmain')},
            'think_end_bmain': it['sample']['think_end'], 'extra': it.get('extra', {}), 'answers': it.get('answers', {}),
            'acc_t1': acc, 'greedy': greedy, 'answer_gt': it['row']['answer']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--model', default='OpenGVLab/InternVL3_5-1B')
    ap.add_argument('--revision', default='2f71cf52542334823e48a46ffba0e2bc9add3446')
    ap.add_argument('--samples', required=True)
    ap.add_argument('--ids', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--skip', default='', help='JSON list of pids finished earlier (e.g. the pilot)')
    ap.add_argument('--time-limit', type=float, default=0, help='seconds; stop starting new groups after this')
    for k, v in DEFAULTS.items():
        ap.add_argument('--' + k.replace('_', '-'), type=type(v), default=v)
    a = ap.parse_args()
    cfg = {k: getattr(a, k) for k in DEFAULTS}
    t0 = time.time()
    from run import fast_kv_selfcheck, pack
    from visref.model import encode_sample, load_mathvista, load_model, special_ids
    samples = json.loads(Path(a.samples).read_text())
    ids = json.loads(Path(a.ids).read_text())
    ids = [str(x) for x in (ids['ids'] if isinstance(ids, dict) else ids)]
    if a.limit:
        ids = ids[:a.limit]
    si, sn = map(int, a.shard.split('/'))
    ids = [p for i, p in enumerate(ids) if i % sn == si]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)['pid'] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    if a.skip:
        done |= {str(x) for x in json.loads(Path(a.skip).read_text())}
    todo = [p for p in ids if p not in done]
    version = code_version()
    print(json.dumps({'event': 'start', 'code_version': version, 'todo': len(todo), 'done': len(done), 'cfg': cfg}), flush=True)
    rows = load_mathvista(a.parquet)
    torch.manual_seed(cfg['seed'])
    torch.set_grad_enabled(False)
    model, tok = load_model(a.model, a.revision)
    lm = model.language_model
    eng = Engine(lm, cfg['temperature'], cfg['top_k'], cfg['top_p'], fast_kv=True)
    ok, info = fast_kv_selfcheck(lm, tok)
    print(json.dumps({'event': 'fast_kv_check', 'ok': ok, **info}), flush=True)
    if not ok:
        from visref.fastkv import set_enabled
        set_enabled(False)
        eng.fast_kv = False
    sp = special_ids(tok, 'Wait, think more', cfg['probe_prompt'])
    generate = make_generate(eng, cfg, pack)
    with out.open('a') as f:
        for gi in range(0, len(todo), cfg['group']):
            if a.time_limit and time.time() - t0 > a.time_limit:
                print(json.dumps({'event': 'time_limit', 'left': len(todo) - gi}), flush=True)
                break
            tg = time.time()
            items = []
            for pid in todo[gi:gi + cfg['group']]:
                s, row = samples[pid], rows[pid]
                base, vis = encode_sample(model, tok, row['image'](), row['query'], cfg['max_num'])
                z, sep, recon = split_z(tok, s['thinking'], s['think_tokens'], s['think_end'], cfg['cut_sep'])
                items.append({'pid': pid, 'row': row, 'sample': s, 'base': base, 'vis': vis, 'z': z, 'sep': sep,
                              'recon': recon})
            run_group(items, eng, tok, sp, cfg, generate)
            for it in items:
                f.write(json.dumps(row_out(it, cfg, version), ensure_ascii=False) + '\n')
            f.flush()
            print(json.dumps({'event': 'group', 'n': len(items), 'group_s': round(time.time() - tg, 1),
                              'elapsed': round(time.time() - t0), 'left': len(todo) - gi - len(items),
                              'peak_alloc_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else 0}),
                  flush=True)
    print(json.dumps({'event': 'end', 'elapsed': round(time.time() - t0)}), flush=True)


if __name__ == '__main__':
    main()
