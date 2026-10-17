"""step1_think -- pre-registered diagnostic (EXPERIMENT_FLOW 2026-10-18 "실행 전 계획: 첫 단계에서 멈춘 문항을 더 생각하게 하기").

For the questions on which TSR and VisRef both stopped after step 1 in the main run (diag/step1_think_ids.py), start from
the stored step-1 thinking z_1 (diag/step1_store_bmain.json: thinking, think_tokens, think_end; re-tokenised exactly as
methods.stored_step1 does, z excludes </think>, the separator cut_sep '\\n' follows a cut step whose text does not end
with one) and run three conditions, each: free thinking (<= think_extra tokens, stops at </think> / EOS), then the
thinking is closed (close_thinking) and the stopping-probe style answer follows ('\\n\\n\\boxed', stops when the box
closes, <= probe_cap tokens; n_t1 samples at T=1 without top-k/top-p + 1 greedy):
  C  [base][z_1][sep] + wait                      wait = 'Wait, think more' (methods.py: after the thinking, after the image block)
  D  [base][z_1][sep] + INJECT(sel) + wait        sel = the picked visual tokens, m = floor(vis_frac * |V|)
  R  [base][z_1][sep] + INJECT(rand) + wait       rand = random subset of the image-token indices, |rand| = |sel|,
                                                  drawn with seed_of(seed, pid, 'rand'), sorted ascending, same format
INJECT, --inject (required):
  bmain       methods.py non-TAG branch of visref_raw_rank: sel = select_tokens(V, embed(z_1), m); INJECT = V[sel], untagged.
  hs_tag_in   methods.py L257-268 of visref_hs_imgtag_in: Z = last-layer hidden states of z_1 in its own context
              (Engine.last_hidden), kernel V = the input image embeddings; INJECT = [\\n unless decoded(z + sep) ends with
              \\n] <img> V[sel] </img> \\n, inside the thinking block, before wait.
The free thinking of C, D and R shares one generator seed per pid (seed_of(seed, pid, 'think'); engine defaults T 0.6 /
top-k 50 / top-p 0.95); the answer samples j use seed_of(seed, pid, 'ans', j) in every condition. Rows keep z ids,
selected indices and reconstruction checks so the inputs can be replayed. No accuracy is computed here (see
diag/analyze_step1_think.py).

  python3 diag/step1_think.py --inject bmain --parquet PQ --out results/step1_think_bmain_0.jsonl \\
      [--ids diag/step1_think_ids.json] [--samples diag/step1_store_bmain.json] [--limit 20] [--shard 0/2] \\
      [--skip DONE.json] [--time-limit SECONDS] [--think-extra 512]
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from visref.dpp import budget, select_tokens  # noqa: E402
from visref.engine import Engine, Request  # noqa: E402
from visref.methods import stored_step1  # noqa: E402
from visref.scoring import SCORER_VERSION, _boxed, score  # noqa: E402

CONDS = ('C', 'D', 'R')
INJECTS = ('bmain', 'hs_tag_in')
DEFAULTS = dict(vis_frac=0.3, think_extra=512, probe_prompt='\n\n\\boxed', wait_text='Wait, think more', probe_cap=16,
                n_t1=3, temperature=0.6, top_k=50, top_p=0.95, seed=42, cut_sep='\n', chunk=1024, batch_tokens=72000,
                group=8, max_num=12)
T1, GREEDY = (1.0, 0, 1.0), (0.0, 0, 1.0)


def code_version():
    h = hashlib.sha256()
    for f in sorted([Path(__file__).resolve(), ROOT / 'run.py', *(ROOT / 'visref').glob('*.py')]):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:12]


def seed_of(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str, parts)).encode()).digest()[:8], 'little') & ((1 << 63) - 1)


def split_stored(tok, sp, s, cut_sep):
    """Stored step 1 -> (z ids, sep ids, sep text, info), as methods.py rebuilds it: stored_step1 re-tokenises the stored
    text (closed / EOS steps get </think> / EOS back, which the step end strips again); a cut step gets cut_sep unless
    its text already ends with it (Job._end_think)."""
    ids, _, info = stored_step1(tok, sp, s, cut_sep)
    end = s['think_end']
    z = list(ids[:-1]) if end in ('close', 'eos') else list(ids)
    sep_text = cut_sep if end == 'cap' and cut_sep and not tok.decode(z).endswith('\n') else ''
    sep = tok.encode(sep_text, add_special_tokens=False) if sep_text else []
    return z, sep, sep_text, {'tokens_match': bool(info['step1_tokens_match']), 'n_z': len(z), 'n_sep': len(sep),
                              'think_tokens_stored': info['step1_think_tokens_stored'], 'think_end_stored': end}


def random_sel(pid, seed, n_visual, n):
    """The R condition's tokens: n distinct indices of range(n_visual), sorted, a function of (seed, pid) only."""
    return sorted(random.Random(seed_of(seed, pid, 'rand')).sample(range(n_visual), n))


def inject_embeds(eng, tok, sp, mode, vis, idx, z_text, sep_text):
    """The embeddings between the step-1 context and the wait cue for the picked indices idx."""
    if len(idx) == 0:
        return vis[:0]
    if mode == 'bmain':
        return vis[idx]
    assert mode == 'hs_tag_in', mode
    nl = tok.encode('\n', add_special_tokens=False)
    pre = [] if (z_text + sep_text).endswith('\n') else nl  # continuing: exactly one newline before <img>
    return torch.cat([eng.embed(pre), eng.embed([sp['img_start']]), vis[idx], eng.embed([sp['img_end']] + nl)], 0)


def build_contexts(it, eng, tok, sp, cfg, mode):
    """Fills it['m'], it['sel'], it['sel_rand'], it['ctx'] = {'C','D','R'} from it['base'], it['vis'], it['z'], it['sep']."""
    base, vis, z, sep = it['base'], it['vis'], it['z'], it['sep']
    prefix = torch.cat([base, eng.embed(z + sep)], 0)
    wait = eng.embed(sp['wait'])
    m = budget(vis.shape[0], cfg['vis_frac'])
    if not z:  # methods.py injects only after a non-empty step
        sel = []
    elif mode == 'bmain':
        sel = select_tokens(vis, eng.embed(z), m)
    else:
        assert mode == 'hs_tag_in', mode
        Z = eng.last_hidden(torch.cat([base, eng.embed(z)], 0), len(z))
        sel = select_tokens(vis, Z, m)
    sel = [int(i) for i in sel]
    rand = random_sel(it['pid'], cfg['seed'], int(vis.shape[0]), len(sel))
    zt = tok.decode(z)
    inj = {c: inject_embeds(eng, tok, sp, mode, vis, idx, zt, it['sep_text']) for c, idx in (('D', sel), ('R', rand))}
    it.update(m=m, sel=sel, sel_rand=rand,
              ctx={'C': torch.cat([prefix, wait], 0), 'D': torch.cat([prefix, inj['D'], wait], 0),
                   'R': torch.cat([prefix, inj['R'], wait], 0)})


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


def extra_record(tok, sp, ids, end):
    """Record of one condition's free thinking. n_think_end_inside counts </think> tokens (or literal '</think>' text)
    before the end of the thinking; generation stops at the first </think>, so it is 0 by construction (a check)."""
    te, eos = sp['think_end'], sp['eos']
    core = list(ids[:-1]) if ids and ids[-1] in (te, eos) else list(ids)
    body = [t for t in ids if t not in (te, eos)]
    text = tok.decode(body)
    return {'end': end, 'len': len(body), 'text': text,
            'n_think_end_inside': sum(t == te for t in core) + text.count('</think>'), 'ids': list(ids)}


def run_group(items, eng, tok, sp, cfg, generate, mode):
    """items: dicts with pid, row, base [L,d], vis [N,d], z, sep, sep_text. Adds the per-condition records in place."""
    te, eos = sp['think_end'], sp['eos']
    dev = eng.emb.weight.device
    for it in items:
        build_contexts(it, eng, tok, sp, cfg, mode)
    reqs, keys = [], []
    for it in items:  # free thinking: C, D, R share one seed
        for c in CONDS:
            g = torch.Generator(device=dev).manual_seed(seed_of(cfg['seed'], it['pid'], 'think'))
            reqs.append(Request(it['ctx'][c], frozenset({te, eos}), cfg['think_extra'], g))
            keys.append((it, c))
    for (it, c), r in zip(keys, generate(reqs)):
        if r is None:
            it.setdefault('extra', {})[c] = {'error': 'non-finite logits'}
            del it['ctx'][c]
            continue
        tail, end = close_thinking(tok, sp, r.ids, r.finish, cfg['cut_sep'])
        it.setdefault('extra', {})[c] = extra_record(tok, sp, r.ids, end)
        it['ctx'][c] = torch.cat([it['ctx'][c], eng.embed(tail)], 0)
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


def row_out(it, cfg, version, mode):
    ans = it.get('answers', {})
    acc = {c: sum(a['correct'] for a in ans[c] if not a['greedy']) / cfg['n_t1'] for c in ans}
    greedy = {c: next(a['correct'] for a in ans[c] if a['greedy']) for c in ans}
    return {'pid': it['pid'], 'stratum': it['sample']['think_end'], 'inject': mode, 'code_version': version,
            'scorer_version': SCORER_VERSION, 'think_extra': cfg['think_extra'], 'n_t1': cfg['n_t1'],
            'n_visual': int(it['vis'].shape[0]), 'm': it['m'], 'n_sel': len(it['sel']), 'sel': it['sel'],
            'sel_rand': it['sel_rand'], 'z_ids': it['z'], 'sep_ids': it['sep'], 'recon': it['recon'],
            'extra': it.get('extra', {}), 'answers': ans, 'acc_t1': acc, 'greedy': greedy, 'answer_gt': it['row']['answer']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inject', required=True, choices=INJECTS)
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--model', default='OpenGVLab/InternVL3_5-1B')
    ap.add_argument('--revision', default='2f71cf52542334823e48a46ffba0e2bc9add3446')
    ap.add_argument('--samples', default=str(ROOT / 'diag/step1_store_bmain.json'))
    ap.add_argument('--ids', default=str(ROOT / 'diag/step1_think_ids.json'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--skip', default='', help='JSON list of pids finished earlier')
    ap.add_argument('--time-limit', type=float, default=0, help='seconds; stop starting new groups after this')
    for k, v in DEFAULTS.items():
        ap.add_argument('--' + k.replace('_', '-'), type=type(v), default=v)
    a = ap.parse_args()
    cfg = {k: getattr(a, k) for k in DEFAULTS}
    t0 = time.time()
    from run import fast_kv_selfcheck, pack
    from visref.model import encode_sample, load_mathvista, load_model, special_ids
    samples = json.loads(Path(a.samples).read_text())
    samples = samples.get('pids', samples)
    ids = json.loads(Path(a.ids).read_text())
    strata = ids.get('stratum', {}) if isinstance(ids, dict) else {}
    ids = [str(x) for x in (ids['ids'] if isinstance(ids, dict) else ids)]
    if a.limit:
        ids = ids[:a.limit]
    si, sn = map(int, a.shard.split('/'))
    ids = [p for i, p in enumerate(ids) if i % sn == si]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    old = [json.loads(l) for l in out.read_text().splitlines() if l.strip()] if out.exists() else []
    if any(r.get('inject') != a.inject for r in old):
        sys.exit(f'{out} holds rows of another --inject mode')
    done = {r['pid'] for r in old}
    if a.skip:
        done |= {str(x) for x in json.loads(Path(a.skip).read_text())}
    todo = [p for p in ids if p not in done]
    for p in todo:
        assert p in samples, f'no stored step 1 for {p}'
        assert not strata or strata[p] == samples[p]['think_end'], (p, strata.get(p), samples[p]['think_end'])
    version = code_version()
    print(json.dumps({'event': 'start', 'code_version': version, 'inject': a.inject, 'todo': len(todo), 'done': len(done),
                      'cfg': cfg}), flush=True)
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
    sp = special_ids(tok, cfg['wait_text'], cfg['probe_prompt'])
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
                z, sep, sep_text, recon = split_stored(tok, sp, s, cfg['cut_sep'])
                items.append({'pid': pid, 'row': row, 'sample': s, 'base': base, 'vis': vis, 'z': z, 'sep': sep,
                              'sep_text': sep_text, 'recon': recon})
            run_group(items, eng, tok, sp, cfg, generate, a.inject)
            for it in items:
                f.write(json.dumps(row_out(it, cfg, version, a.inject), ensure_ascii=False) + '\n')
            f.flush()
            print(json.dumps({'event': 'group', 'n': len(items), 'group_s': round(time.time() - tg, 1),
                              'elapsed': round(time.time() - t0), 'left': len(todo) - gi - len(items),
                              'peak_alloc_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else 0}),
                  flush=True)
    print(json.dumps({'event': 'end', 'elapsed': round(time.time() - t0)}), flush=True)


if __name__ == '__main__':
    main()
