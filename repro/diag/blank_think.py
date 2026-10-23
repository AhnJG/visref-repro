"""blank_think -- pre-registered diagnostic (plan docs/EXPERIMENT_FLOW.md 2026-10-24 "실행 전 계획: 빈 그림 생성 양성 대조"; this file
is code only, written before any result).

Question: when the prompt's image slots hold no information (a same-size gray image, as in diag/recover_check.py), does putting the
question's own image tokens back after the first thinking step, then thinking on, raise the ACCURACY of the answer -- more than
putting back another question's image, and more than thinking on without an image? A positive control of the generation path of
re-insertion: recover_check (A9) read probabilities only and generated nothing.

Step 1 (generated here, not stored): from the prompt with its own image (own) and from the same prompt with the image slots
overwritten by the gray image's tokens (blank), the model thinks up to think_cap tokens (stops at </think> / EOS), one generator
seed_of(seed, pid, 'step1') for both. Step end as methods.Job._end_think (closed_multistep.split_generated): </think> and EOS are
dropped, a step cut at the cap gets cut_sep '\\n' unless its text ends with one.
Conditions (prefix_s = [prompt_s][z_1,s][sep_s]):
  A0  prefix_own   </think>                                  answer at once, own image (descriptive)
  N0  prefix_blank </think>                                  answer at once, blank (descriptive)
  A   prefix_own   Wait, think more  [z_2] -> close          own image, no re-insertion (gate: A - N)
  N   prefix_blank Wait, think more  [z_2] -> close          blank, no re-insertion
  O   prefix_blank V_own   Wait, think more [z_2] -> close   blank + ALL the question's image tokens, untagged (A8's F format)
  X   prefix_blank V_donor Wait, think more [z_2] -> close   blank + all tokens of the donor (same n_visual, ids file)
The extra thinking z_2 (<= think_extra tokens, stops at </think> / EOS) of A, N, O, X shares seed_of(seed, pid, 'think');
close_thinking (step1_think) appends the </think> (the model's own when it emitted one). Answers of every condition: '\\n\\n\\boxed',
n_t1 samples at T=1 without top-k/top-p (seed_of(seed, pid, 'ans', j)) + 1 greedy, <= probe_cap tokens, stopped when the box
closes, answer_record = rule scorer v4. No accuracy and no decision is computed here (diag/analyze_blank_think.py).
Rows: pid, status 'ok', set, question_type, n_visual, donor, step1 {own, blank: end, len, ids, sep_text}, extra {A, N, O, X:
extra_record}, answers, n_correct, acc_t1, greedy, answer_gt, code_version, cfg, timing. A question that raises or returns
non-finite logits gets {pid, status: 'error', ...}. A ChunkError (a generate row that is neither 'stop' nor 'cap') stops the run.
Trial mode (--set trial) also writes OUT.memcheck.json: the longest planned context (blank prompt of the memcheck question +
think_cap + donor tokens + think_extra, padded with its own text) and a few generated tokens, peak memory.

  python3 diag/blank_think.py --parquet PQ --out results/blank_think_0.jsonl [--set main|trial] [--ids diag/blank_think_ids.json]
      [--limit N] [--shard 0/4] [--skip DONE.json] [--time-limit SECONDS] [--think-cap 1024] [--think-extra 1024] [--group 4]
  Questions already in --out are skipped (resume).
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
from closed_multistep import ChunkError, error_row, make_generate, read_rows, split_generated  # noqa: E402
from step1_think import GREEDY, T1, answer_record, box_close_index, close_thinking, extra_record, seed_of  # noqa: E402
from visref.engine import Engine, Request  # noqa: E402
from visref.scoring import SCORER_VERSION  # noqa: E402

GRAY = (128, 128, 128)
CONDS = ('A0', 'N0', 'A', 'N', 'O', 'X')
FREE = ('A', 'N', 'O', 'X')  # conditions that think on
SLOT = {'A0': 'own', 'N0': 'blank', 'A': 'own', 'N': 'blank', 'O': 'blank', 'X': 'blank'}
INSERT = {'O': 'vis', 'X': 'vis_donor'}
DEFAULTS = dict(think_cap=1024, think_extra=1024, probe_prompt='\n\n\\boxed', wait_text='Wait, think more', probe_cap=16, n_t1=3,
                temperature=0.6, top_k=50, top_p=0.95, seed=42, cut_sep='\n', chunk=0, batch_tokens=72000, group=4,
                max_num=12)  # chunk 0 = max(think_cap, think_extra)


def code_version():
    """sha256 over this script, the diag helpers it imports, run.py and visref/*.py."""
    h = hashlib.sha256()
    files = [Path(__file__).resolve(), ROOT / 'diag/closed_multistep.py', ROOT / 'diag/step1_think.py', ROOT / 'run.py',
             *(ROOT / 'visref').glob('*.py')]
    for f in sorted(files):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:12]


def resolve_cfg(cfg):
    """chunk 0 -> max(think_cap, think_extra); refuse a chunk below either (unfinished rows would be read as 'cap')."""
    cfg = dict(cfg)
    need = max(cfg['think_cap'], cfg['think_extra'])
    if cfg['chunk'] == 0:
        cfg['chunk'] = need
    if cfg['chunk'] < need:
        raise SystemExit(f"--chunk {cfg['chunk']} < {need}: refused (rows that return finish 'chunk' are not re-submitted)")
    if min(cfg['think_cap'], cfg['think_extra'], cfg['probe_cap'], cfg['n_t1']) < 1 or cfg['probe_cap'] > cfg['chunk']:
        raise SystemExit('--think-cap, --think-extra, --probe-cap and --n-t1 must be >= 1 and --probe-cap <= --chunk')
    return cfg


def fill_slots(base, img_pos, vis):
    """Copy of the prompt embeddings with the <IMG_CONTEXT> rows replaced by vis [N, d] (recover_check.fill_slots)."""
    if vis.shape[0] != img_pos.numel():
        raise ValueError(f'image gives {vis.shape[0]} tokens, the prompt has {img_pos.numel()} image slots')
    out = base.clone()
    out[img_pos] = vis.to(out.dtype)
    return out


def device_of(eng):
    emb = getattr(eng, 'emb', None)
    return emb.weight.device if emb is not None else torch.device('cpu')


def contexts(it, eng, sp):
    """{cond: context before the extra thinking (FREE) or the answer (A0, N0)} from it['prompt'], it['z'], it['sep']."""
    pre = {s: torch.cat([it['prompt'][s], eng.embed(it['z'][s] + it['sep'][s])], 0) for s in ('own', 'blank')}
    wait = eng.embed(sp['wait'])
    out = {}
    for c in CONDS:
        p = pre[SLOT[c]]
        if c in FREE:
            ins = it[INSERT[c]] if c in INSERT else p[:0]
            out[c] = torch.cat([p, ins, wait], 0)
        else:
            out[c] = torch.cat([p, eng.embed([sp['think_end']])], 0)
    return out


def reset(it):
    for k in ('z', 'sep', 'step1', 'ctx', 'extra', 'answers', 'nonfinite'):
        it.pop(k, None)
    it.update(z={}, sep={}, step1={}, ctx={}, extra={}, answers={}, nonfinite=[])


def run_group(items, eng, tok, sp, cfg, generate, timing=None):
    """items: dicts with pid, row, prompt {own, blank} [L,d], vis, vis_donor [N,d]. Fills step1, extra, answers, nonfinite in place."""
    te, eos = sp['think_end'], sp['eos']
    dev = device_of(eng)
    timing = {} if timing is None else timing
    for it in items:
        reset(it)
    t = time.time()
    reqs, keys = [], []
    for it in items:  # step 1: own and blank share one seed
        for s in ('own', 'blank'):
            g = torch.Generator(device=dev).manual_seed(seed_of(cfg['seed'], it['pid'], 'step1'))
            reqs.append(Request(it['prompt'][s], frozenset({te, eos}), cfg['think_cap'], g))
            keys.append((it, s))
    for (it, s), r in zip(keys, generate(reqs) if reqs else []):
        if r is None:
            it['nonfinite'].append(f'step1_{s}')
            continue
        z, sep, sep_text, end = split_generated(tok, sp, r.ids, cfg['cut_sep'])
        it['z'][s], it['sep'][s] = z, sep
        it['step1'][s] = {'end': end, 'len': len(z), 'ids': [int(x) for x in z], 'sep_text': sep_text}
    live = [it for it in items if not it['nonfinite']]
    for it in live:
        it['ctx'] = contexts(it, eng, sp)
    timing['step1'] = time.time() - t
    t = time.time()
    reqs, keys = [], []
    for it in live:  # extra thinking: A, N, O, X share one seed
        for c in FREE:
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
    for it in live:
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
        it['timing'] = {f'group_{k}_s': round(timing.get(k, 0.0), 2) for k in ('step1', 'extra', 'answers')}


def row_out(it, cfg, version, set_name):
    if it['nonfinite']:
        row = error_row(it['pid'], 'generate', 'non-finite logits in ' + ','.join(sorted(set(it['nonfinite']))), version, set_name)
        row['nonfinite'] = sorted(set(it['nonfinite']))
        return row
    ans = it['answers']
    n_correct = {c: sum(a['correct'] for a in ans[c] if not a['greedy']) for c in CONDS}
    return {'pid': it['pid'], 'status': 'ok', 'set': set_name, 'question_type': it['row'].get('question_type'),
            'n_visual': int(it['vis'].shape[0]), 'donor': it['donor'], 'code_version': version, 'scorer_version': SCORER_VERSION,
            'cfg': dict(cfg), 'step1': it['step1'], 'extra': it['extra'], 'answers': ans, 'n_correct': n_correct,
            'acc_t1': {c: n_correct[c] / cfg['n_t1'] for c in CONDS},
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


def select_pids(doc, set_name, limit=0, shard='0/1', done=()):
    """The set's order, first `limit`, the shard's share by index (i % n), without the done ones."""
    if set_name not in ('main', 'trial'):
        raise SystemExit(f'no set {set_name!r}')
    ids = [str(p) for p in doc[set_name]]
    if limit:
        ids = ids[:limit]
    si, sn = map(int, shard.split('/'))
    done = {str(x) for x in done}
    return [p for i, p in enumerate(ids) if i % sn == si and p not in done]


def memory_check(it, eng, tok, sp, cfg, generate):
    """Peak memory of the longest planned context: blank prompt + think_cap + donor tokens + think_extra (padded text)."""
    pad_ids = tok.encode('a b c ', add_special_tokens=False)
    pad = (pad_ids * (cfg['think_cap'] // len(pad_ids) + 1))[:cfg['think_cap']]
    ctx = torch.cat([it['prompt']['blank'], eng.embed(pad), it['vis_donor'], eng.embed(sp['wait']), eng.embed(pad)], 0)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    g = torch.Generator(device=device_of(eng)).manual_seed(seed_of(cfg['seed'], it['pid'], 'memcheck'))
    r = generate([Request(ctx, frozenset({sp['eos']}), 8, g)] * 1)[0]
    if cuda:
        torch.cuda.synchronize()
    return {'pid': it['pid'], 'n_visual': int(it['vis'].shape[0]), 'ctx_len': int(ctx.shape[0]),
            'generated_tokens': None if r is None else len(r.ids), 'generate_s': round(time.time() - t, 2),
            'peak_alloc_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if cuda else 0,
            'peak_reserved_gb': round(torch.cuda.max_memory_reserved() / 2 ** 30, 2) if cuda else 0}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--model', default='OpenGVLab/InternVL3_5-1B')
    ap.add_argument('--revision', default='2f71cf52542334823e48a46ffba0e2bc9add3446')
    ap.add_argument('--ids', default=str(ROOT / 'diag/blank_think_ids.json'))
    ap.add_argument('--set', default='main', choices=('main', 'trial'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--skip', default='', help='JSON list of pids finished earlier')
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
    done = {r['pid'] for r in read_rows(out)}
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
                'vis_donor': vd, 'donor': d}

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
                              'encode_s': round(tm.get('encode', 0), 1), 'step1_s': round(tm.get('step1', 0), 1),
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
