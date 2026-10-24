"""sel_think -- pre-registered diagnostic (plan docs/EXPERIMENT_FLOW.md 2026-10-24 "실행 전 계획: 선택 토큰 생성 대조"; this file is
code only, written before any result).

Question: blank_think showed that ALL of the question's image tokens, put back after the first step when the prompt's image slots
are blank, raise accuracy through generation. Do the tokens the current selection picks (greedy MAP dpp.select_tokens with
Z = the last-layer hidden states of the step, V = the input image embeddings, m = floor(vis_frac * |V|), no filling; the author's
selection of 2026-10-17 / 10-23, EF "동작 점검 계획" and closed_multistep) do the same, more than as many random tokens -- and does
the same insertion do anything when the prompt holds the real image?

Step 1 is NOT generated here: the own / blank first steps blank_think generated and used (diag/sel_think_ids.json 'step1', ids
and separator text as stored) are rebuilt as they were. prefix_s = [prompt_s][z_1,s][sep_s]; prompt_blank = the prompt with the
image slots overwritten by the gray image's tokens (blank_think.fill_slots).
Conditions (all: prefix_s [block] Wait, think more [z_2] -> close; block = closed_multistep.v_block: [\\n unless decode(z_1,s) +
sep_s ends with one] <img> tokens </img> \\n, the format of every insertion here (EF 1905); V = the question's own image tokens,
in their order):
  N_o  own    nothing                   N_b  blank  nothing
  D_o  own    V[sel_own]                D_b  blank  V[sel_blank]
  F_o  own    V (all)                   R_b  blank  V[random, |sel_blank| tokens]  (step1_think.random_sel, a function of pid)
                                        F_b  blank  V (all)
                                        X_b  blank  the donor's tokens (all; same n_visual, ids file)
sel_s = select_tokens(V, Z_s, budget(|V|, vis_frac)), Z_s = Engine.last_hidden([prompt_s][z_1,s], |z_1,s|): the step's last-layer
states in its own context, as methods.py 'visref_hs' and closed_multistep do (no filling below m).
The extra thinking z_2 (<= think_extra tokens, stops at </think> / EOS) of every condition shares seed_of(seed, pid, 'think');
close_thinking appends the </think>. Answers: '\\n\\n\\boxed', n_t1 samples at T=1 (seed_of(seed, pid, 'ans', j)) + 1 greedy,
<= probe_cap tokens, stopped when the box closes, rule scorer v4 (as blank_think). No accuracy and no decision is computed here
(diag/analyze_sel_think.py).
Rows: pid, status 'ok', set, question_type, n_visual, donor, m, sel {own, blank: indices}, n_sel {own, blank}, recon, extra
{cond: extra_record}, answers, n_correct, acc_t1, greedy, answer_gt, code_version, cfg, timing. A question that raises or returns
non-finite logits gets {pid, status: 'error', ...}. A ChunkError stops the run. Resume: only 'ok' rows of --out are skipped.
Trial mode (--set trial) also writes OUT.memcheck.json for the ids file's memcheck question (the largest image): last_hidden over
its own prompt + step1_cap padded tokens (the selection's prefill), and a few tokens generated after own prompt + step1_cap + the
block of ALL its image tokens + Wait + think_extra (the longest planned context); peak memory. Not part of any result.

  python3 diag/sel_think.py --parquet PQ --out results/sel_think_0.jsonl [--set main|trial] [--ids diag/sel_think_ids.json]
      [--limit N] [--shard 0/2] [--skip DONE.json] [--time-limit SECONDS] [--think-extra 1024] [--group 4]
"""
import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'diag'))
from blank_think import GRAY, device_of, fill_slots, select_pids  # noqa: E402
from closed_multistep import ChunkError, error_row, make_generate, read_rows, v_block  # noqa: E402
from step1_think import GREEDY, T1, answer_record, box_close_index, close_thinking, extra_record, random_sel, seed_of  # noqa: E402
from visref.dpp import budget, select_tokens  # noqa: E402
from visref.engine import Engine, Request  # noqa: E402
from visref.scoring import SCORER_VERSION  # noqa: E402

CONDS = ('N_o', 'D_o', 'F_o', 'N_b', 'D_b', 'R_b', 'F_b', 'X_b')
SLOT = {c: 'own' if c.endswith('_o') else 'blank' for c in CONDS}
DEFAULTS = dict(think_extra=1024, vis_frac=0.3, probe_prompt='\n\n\\boxed', wait_text='Wait, think more', probe_cap=16, n_t1=3,
                temperature=0.6, top_k=50, top_p=0.95, seed=42, cut_sep='\n', chunk=0, batch_tokens=72000, group=4, max_num=12,
                step1_cap=1024)  # step1_cap: the stored first steps' cap (blank_think think_cap), memory check only


def code_version():
    """sha256 over this script, the diag helpers it imports, run.py and visref/*.py."""
    h = hashlib.sha256()
    files = [Path(__file__).resolve(), ROOT / 'diag/blank_think.py', ROOT / 'diag/closed_multistep.py', ROOT / 'diag/step1_think.py',
             ROOT / 'run.py', *(ROOT / 'visref').glob('*.py')]
    for f in sorted(files):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:12]


def resolve_cfg(cfg):
    """chunk 0 -> think_extra; refuse a chunk below it (unfinished rows would be read as 'cap')."""
    cfg = dict(cfg)
    if cfg['chunk'] == 0:
        cfg['chunk'] = cfg['think_extra']
    if cfg['chunk'] < cfg['think_extra']:
        raise SystemExit(f"--chunk {cfg['chunk']} < {cfg['think_extra']}: refused")
    if min(cfg['think_extra'], cfg['probe_cap'], cfg['n_t1']) < 1 or cfg['probe_cap'] > cfg['chunk']:
        raise SystemExit('--think-extra, --probe-cap and --n-t1 must be >= 1 and --probe-cap <= --chunk')
    if not 0 < cfg['vis_frac'] <= 1:
        raise SystemExit('--vis-frac must be in (0, 1]')
    return cfg


def stored_step(tok, s):
    """Stored first step -> (z ids, sep ids, sep text): ids as generated without </think> / EOS, separator as stored."""
    sep_text = s['sep_text']
    return list(s['ids']), tok.encode(sep_text, add_special_tokens=False) if sep_text else [], sep_text


def selections(it, eng, cfg):
    """{'own', 'blank', 'rand'}: indices into V. sel_s from the last-layer states of z_1,s in its own context; rand = as many as
    sel_blank, a function of pid only."""
    vis = it['vis']
    m = budget(vis.shape[0], cfg['vis_frac'])
    sel = {}
    for s in ('own', 'blank'):
        z = it['z'][s]
        Z = eng.last_hidden(torch.cat([it['prompt'][s], eng.embed(z)], 0), len(z)) if z else None
        sel[s] = [int(i) for i in select_tokens(vis, Z, m)] if z else []
    sel['rand'] = random_sel(it['pid'], cfg['seed'], int(vis.shape[0]), len(sel['blank']))
    return m, sel


def contexts(it, eng, tok, sp):
    """{cond: context before the extra thinking} from it['prompt'], it['z'], it['sep'], it['sep_text'], it['vis'], it['vis_donor'],
    it['sel']. Every insertion is a v_block; no block when the step text is empty (methods.py: `if ... and z`)."""
    pre = {s: torch.cat([it['prompt'][s], eng.embed(it['z'][s] + it['sep'][s])], 0) for s in ('own', 'blank')}
    vis, sel = it['vis'], it['sel']
    every = list(range(int(vis.shape[0])))
    src = {'D_o': (vis, sel['own']), 'F_o': (vis, every), 'D_b': (vis, sel['blank']), 'R_b': (vis, sel['rand']), 'F_b': (vis, every),
           'X_b': (it['vis_donor'], list(range(int(it['vis_donor'].shape[0]))))}
    wait = eng.embed(sp['wait'])
    out = {}
    for c in CONDS:
        s = SLOT[c]
        block = pre[s][:0]
        if c in src and it['z'][s]:
            v, idx = src[c]
            block = v_block(eng, tok, sp, v, idx, tok.decode(it['z'][s]), it['sep_text'][s])
        out[c] = torch.cat([pre[s], block, wait], 0)
    return out


def run_group(items, eng, tok, sp, cfg, generate, timing=None):
    """items: dicts with pid, row, prompt {own, blank} [L,d], vis, vis_donor [N,d], step1 {own, blank} (stored). Fills z, sep, m,
    sel, extra, answers, nonfinite in place."""
    te, eos = sp['think_end'], sp['eos']
    dev = device_of(eng)
    timing = {} if timing is None else timing
    t = time.time()
    for it in items:
        it.update(z={}, sep={}, sep_text={}, extra={}, answers={}, nonfinite=[])
        for s in ('own', 'blank'):
            it['z'][s], it['sep'][s], it['sep_text'][s] = stored_step(tok, it['step1'][s])
        it['m'], it['sel'] = selections(it, eng, cfg)
        it['ctx'] = contexts(it, eng, tok, sp)
    timing['select'] = time.time() - t
    t = time.time()
    reqs, keys = [], []
    for it in items:  # extra thinking: every condition shares one seed
        for c in CONDS:
            g = torch.Generator(device=dev).manual_seed(seed_of(cfg['seed'], it['pid'], 'think'))
            reqs.append(Request(it['ctx'][c], frozenset({te, eos}), cfg['think_extra'], g))
            keys.append((it, c))
    for (it, c), r in zip(keys, generate(reqs) if reqs else []):
        if r is None:
            it['nonfinite'].append(c)
            continue
        tail, end = close_thinking(tok, sp, r.ids, r.finish, cfg['cut_sep'])
        it['extra'][c] = extra_record(tok, sp, r.ids, end)
        it['ctx'][c] = torch.cat([it['ctx'][c], eng.embed(tail)], 0)
    timing['extra'] = time.time() - t
    t = time.time()
    stop = lambda ids, bt=sp['brace_text']: box_close_index(ids, bt) is not None  # noqa: E731
    reqs, keys = [], []
    for it in items:
        for c in CONDS:
            if c in it['nonfinite']:
                continue
            ctx = torch.cat([it['ctx'][c], eng.embed(sp['probe_prompt'])], 0)
            for j in range(cfg['n_t1'] + 1):
                greedy = j == cfg['n_t1']
                g = None if greedy else torch.Generator(device=dev).manual_seed(seed_of(cfg['seed'], it['pid'], 'ans', j))
                reqs.append(Request(ctx, frozenset({eos}), cfg['probe_cap'], g, sampling=GREEDY if greedy else T1,
                                    stop_check=stop))
                keys.append((it, c, j))
    for (it, c, j), r in zip(keys, generate(reqs) if reqs else []):
        if r is None:
            it['nonfinite'].append(f'{c}_answer')
            continue
        rec = answer_record(tok, sp, r.ids, it['row'])
        rec.update(greedy=j == cfg['n_t1'], n_tok=len(r.ids))
        it['answers'].setdefault(c, []).append(rec)
    timing['answers'] = time.time() - t
    for it in items:
        it['ctx'] = {}
        it['timing'] = {f'group_{k}_s': round(timing.get(k, 0.0), 2) for k in ('select', 'extra', 'answers')}


def row_out(it, cfg, version, set_name):
    if it['nonfinite']:
        row = error_row(it['pid'], 'generate', 'non-finite logits in ' + ','.join(sorted(set(it['nonfinite']))), version, set_name)
        row['nonfinite'] = sorted(set(it['nonfinite']))
        return row
    ans = it['answers']
    n_correct = {c: sum(a['correct'] for a in ans[c] if not a['greedy']) for c in CONDS}
    return {'pid': it['pid'], 'status': 'ok', 'set': set_name, 'question_type': it['row'].get('question_type'),
            'n_visual': int(it['vis'].shape[0]), 'donor': it['donor'], 'm': it['m'],
            'sel': {s: [int(i) for i in it['sel'][s]] for s in ('own', 'blank')},
            'n_sel': {s: len(it['sel'][s]) for s in ('own', 'blank', 'rand')},
            'recon': {s: {'n_z': len(it['z'][s]), 'n_sep': len(it['sep'][s]), 'end': it['step1'][s]['end']} for s in ('own', 'blank')},
            'code_version': version, 'scorer_version': SCORER_VERSION, 'cfg': dict(cfg), 'extra': it['extra'], 'answers': ans,
            'n_correct': n_correct, 'acc_t1': {c: n_correct[c] / cfg['n_t1'] for c in CONDS},
            'greedy': {c: next(a['correct'] for a in ans[c] if a['greedy']) for c in CONDS}, 'answer_gt': it['row']['answer'],
            'timing': {'encode_s': it.get('encode_s'), **it['timing']}}


def process_group(pids, make_item, eng, tok, sp, cfg, generate, version, set_name, timing=None):
    """Rows (ok or error) for pids, in order; when the group raises, every question is re-run alone. A ChunkError stops the run."""
    timing = {} if timing is None else timing
    rows, items = {}, []
    t = time.time()
    for pid in pids:
        t0 = time.time()
        try:
            it = make_item(pid)
            it['encode_s'] = round(time.time() - t0, 2)
            items.append(it)
        except ChunkError:
            raise
        except Exception as e:
            rows[pid] = error_row(pid, 'load', f'{type(e).__name__}: {e}', version, set_name)
    timing['encode'] = time.time() - t
    try:
        run_group(items, eng, tok, sp, cfg, generate, timing)
    except ChunkError:
        raise
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for it in items:
            try:
                run_group([it], eng, tok, sp, cfg, generate)
            except ChunkError:
                raise
            except Exception as e:
                rows[it['pid']] = error_row(it['pid'], 'generate', f'{type(e).__name__}: {e}', version, set_name)
    for it in items:
        if it['pid'] not in rows:
            rows[it['pid']] = row_out(it, cfg, version, set_name)
    return [rows[p] for p in pids]


def memory_check(it, eng, tok, sp, cfg, generate):
    """Peak memory of the selection prefill (own prompt + step1_cap) and of the longest planned context (own prompt + step1_cap +
    block of all image tokens + Wait + think_extra), padded with text."""
    pad_ids = tok.encode('a b c ', add_special_tokens=False)
    pad = lambda n: (pad_ids * (n // len(pad_ids) + 1))[:n]  # noqa: E731
    z1, z2 = pad(cfg['step1_cap']), pad(cfg['think_extra'])
    vis = it['vis']
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    Z = eng.last_hidden(torch.cat([it['prompt']['own'], eng.embed(z1)], 0), len(z1))
    sel = select_tokens(vis, Z, budget(vis.shape[0], cfg['vis_frac']))
    t_hidden = time.time() - t
    block = v_block(eng, tok, sp, vis, list(range(int(vis.shape[0]))), tok.decode(z1), '')
    ctx = torch.cat([it['prompt']['own'], eng.embed(z1), block, eng.embed(sp['wait']), eng.embed(z2)], 0)
    t = time.time()
    g = torch.Generator(device=device_of(eng)).manual_seed(seed_of(cfg['seed'], it['pid'], 'memcheck'))
    r = generate([Request(ctx, frozenset({sp['eos']}), min(8, cfg['chunk']), g)])[0]
    if cuda:
        torch.cuda.synchronize()
    return {'pid': it['pid'], 'n_visual': int(vis.shape[0]), 'last_hidden_n': len(z1), 'n_sel': len(sel), 'ctx_len': int(ctx.shape[0]),
            'generated_tokens': None if r is None else len(r.ids), 'last_hidden_s': round(t_hidden, 2),
            'generate_s': round(time.time() - t, 2),
            'peak_alloc_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if cuda else 0,
            'peak_reserved_gb': round(torch.cuda.max_memory_reserved() / 2 ** 30, 2) if cuda else 0}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--model', default='OpenGVLab/InternVL3_5-1B')
    ap.add_argument('--revision', default='2f71cf52542334823e48a46ffba0e2bc9add3446')
    ap.add_argument('--ids', default=str(ROOT / 'diag/sel_think_ids.json'))
    ap.add_argument('--set', default='main', choices=('main', 'trial'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--skip', default='', help='JSON list of pids finished earlier (ok rows)')
    ap.add_argument('--time-limit', type=float, default=0, help='seconds; stop starting new groups after this')
    for k, v in DEFAULTS.items():
        ap.add_argument('--' + k.replace('_', '-'), type=type(v), default=v)
    a = ap.parse_args(argv)
    a.cfg = resolve_cfg({k: getattr(a, k) for k in DEFAULTS})
    return a


def main(argv=None):
    a = parse_args(argv)
    cfg = a.cfg
    t0 = time.time()
    from PIL import Image
    from run import fast_kv_selfcheck, pack
    from visref.model import encode_sample, load_mathvista, load_model, special_ids
    doc = json.loads(Path(a.ids).read_text())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r['pid'] for r in read_rows(out) if r.get('status') == 'ok'}
    n_done = len(done)
    if a.skip:
        done |= {str(x) for x in json.loads(Path(a.skip).read_text())}
    todo = select_pids(doc, a.set, a.limit, a.shard, done)
    version = code_version()
    print(json.dumps({'event': 'start', 'code_version': version, 'set': a.set, 'todo': len(todo), 'done': n_done, 'cfg': cfg}),
          flush=True)
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

    def make_item(pid):
        row = rows[pid]
        image = row['image']()
        base, vis, pos = encode_sample(model, tok, image, row['query'], cfg['max_num'], with_positions=True)
        n = int(vis.shape[0])
        if n != doc['n_visual'][pid]:
            raise ValueError(f'n_visual {n} differs from the ids file {doc["n_visual"][pid]}')
        blank = encode_sample(model, tok, Image.new('RGB', image.size, GRAY), row['query'], cfg['max_num'])[1]
        d = doc['donor'][pid]
        assert d != pid
        dr = rows[d]
        vd = encode_sample(model, tok, dr['image'](), dr['query'], cfg['max_num'])[1]
        if vd.shape[0] != n:
            raise ValueError(f'donor {d} gives {vd.shape[0]} image tokens, the question {n}')
        return {'pid': pid, 'row': row, 'prompt': {'own': base, 'blank': fill_slots(base, pos, blank)}, 'vis': vis,
                'vis_donor': vd, 'donor': d, 'step1': doc['step1'][pid]}

    peak_all, n_err = 0.0, 0
    with out.open('a') as f:
        for gi in range(0, len(todo), cfg['group']):
            if a.time_limit and time.time() - t0 > a.time_limit:
                print(json.dumps({'event': 'time_limit', 'left': len(todo) - gi}), flush=True)
                break
            tg = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            tm = {}
            out_rows = process_group(todo[gi:gi + cfg['group']], make_item, eng, tok, sp, cfg, generate, version, a.set, tm)
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                if r['status'] != 'ok':
                    n_err += 1
                    print(json.dumps({'event': 'error', 'pid': r['pid'], 'error': r['error']}), flush=True)
            f.flush()
            peak = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else 0
            peak_all = max(peak_all, peak)
            print(json.dumps({'event': 'group', 'n': len(out_rows), 'n_error': sum(r['status'] != 'ok' for r in out_rows),
                              'encode_s': round(tm.get('encode', 0), 1), 'select_s': round(tm.get('select', 0), 1),
                              'extra_s': round(tm.get('extra', 0), 1), 'answers_s': round(tm.get('answers', 0), 1),
                              'group_s': round(time.time() - tg, 1), 'elapsed': round(time.time() - t0),
                              'left': len(todo) - gi - len(out_rows), 'peak_alloc_gb': peak,
                              'max_rss_gb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)}), flush=True)
    if a.set == 'trial' and doc.get('memcheck') in todo + list(done):
        mem = memory_check(make_item(doc['memcheck']), eng, tok, sp, cfg, generate)
        Path(str(out) + '.memcheck.json').write_text(json.dumps(mem, indent=1) + '\n')
        print(json.dumps({'event': 'memcheck', **mem}), flush=True)
    print(json.dumps({'event': 'end', 'elapsed': round(time.time() - t0), 'n_error': n_err, 'peak_alloc_gb': peak_all,
                      'max_rss_gb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)}), flush=True)


if __name__ == '__main__':
    main()
