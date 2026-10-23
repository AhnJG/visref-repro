"""slot_cross -- pre-registered, probability-level diagnostic (plan docs/EXPERIMENT_FLOW.md 2026-10-24 "실행 전 계획: 그림 자리 교차
진단"; this file is code only, written before any result).

Question: after a GENERATED first thinking step, is the same image read less when it is put back after the thinking than when
it sits in the prompt's own image slots? recover_check (A9) put the image back after one short fixed sentence; here the first
step is generated from the blank prompt (diag/blank_think.py: same seed_of(seed, pid, 'step1'), same step-end rule) and its
token ids are then held fixed in every cell.

Cells (slot S x post block P), context = [prompt with image slots S][z_1][sep][P] </think> '\\n\\n\\boxed', then the gold answer
'{' + answer + '}' is teacher-forced and its probability read exactly as recover_check does (score_string_pos with consecutive
position ids, read_cond: the box closing merged):
  S in {own, blank}: the prompt's <IMG_CONTEXT> rows hold the question's image tokens / the same-size gray image's tokens
  P in {'' none, own: all the question's image tokens, blank: all the gray image's tokens}, untagged (blank_think O format)
  names: S letter + P letter, '-' for none: o-, b-, oo, ob, bo, bb.  ob and bo hold the same image information at the same
  length, at swapped places.
Rows (JSONL): pid, status 'ok', set, question_type, n_visual, answer_gt, gt_ids, step1 {end, len, ids, sep_text}, cell {name:
logp_gt, p_gt, logp_gt_full, ctx_len}, code_version, cfg, timing. Errors: {pid, status 'error', stage, error}. No decision here
(diag/analyze_slot_cross.py).

  python3 diag/slot_cross.py --parquet PQ --out results/slot_cross_0.jsonl [--set main|trial] [--ids diag/blank_think_ids.json]
      [--limit N] [--shard 0/2] [--skip DONE.json] [--time-limit SECONDS] [--think-cap 1024] [--group 4]
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'diag'))
from blank_think import GRAY, device_of, fill_slots, select_pids  # noqa: E402
from closed_multistep import ChunkError, error_row, make_generate, read_rows, split_generated  # noqa: E402
from inject_sensitivity import content_positions  # noqa: E402
from recover_check import gold_ids, read_cond, score_string_pos  # noqa: E402
from step1_think import seed_of  # noqa: E402
from visref.engine import Engine, Request  # noqa: E402
from visref.scoring import SCORER_VERSION  # noqa: E402

CELLS = {'o-': ('own', None), 'b-': ('blank', None), 'oo': ('own', 'own'), 'ob': ('own', 'blank'), 'bo': ('blank', 'own'),
         'bb': ('blank', 'blank')}
DEFAULTS = dict(think_cap=1024, probe_prompt='\n\n\\boxed', wait_text='Wait, think more', temperature=0.6, top_k=50, top_p=0.95,
                seed=42, cut_sep='\n', chunk=0, batch_tokens=72000, group=4, max_num=12, probe_cap=16)  # chunk 0 = think_cap


def code_version():
    """sha256 over this script, the diag helpers it imports, run.py and visref/*.py."""
    h = hashlib.sha256()
    files = [Path(__file__).resolve(), *(ROOT / 'diag' / f for f in ('blank_think.py', 'closed_multistep.py', 'step1_think.py',
                                                                    'recover_check.py', 'recover_check_ids.py',
                                                                    'inject_sensitivity.py')),
             ROOT / 'run.py', *(ROOT / 'visref').glob('*.py')]
    for f in sorted(files):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:12]


def resolve_cfg(cfg):
    cfg = dict(cfg)
    if cfg['chunk'] == 0:
        cfg['chunk'] = cfg['think_cap']
    if cfg['chunk'] < cfg['think_cap'] or cfg['think_cap'] < 1:
        raise SystemExit(f"--chunk {cfg['chunk']} must be >= --think-cap {cfg['think_cap']} >= 1")
    return cfg


def cell_context(it, eng, sp, slot, post):
    """[prompt_slot][z][sep][post block] </think> + probe prompt."""
    block = it['vis_of'][post] if post else it['vis_of']['own'][:0]
    return torch.cat([it['prompt'][slot], eng.embed(it['z'] + it['sep']), block,
                      eng.embed([sp['think_end']] + sp['probe_prompt'])], 0)


def score_cells(it, eng, sp):
    """{cell: record} with the gold answer's probability read as recover_check.score_question does (consecutive ids)."""
    close_idx = torch.tensor(sorted(sp['close_ids']))
    gt_pos, gt_close = content_positions(it['gt_ids'], sp['brace_text'])
    out = {}
    for name, (slot, post) in CELLS.items():
        ctx = cell_context(it, eng, sp, slot, post)
        pos = torch.arange(ctx.shape[0], device=ctx.device)
        full, logp = score_string_pos(eng, ctx, it['gt_ids'], pos)
        a, p, _ = read_cond(logp, it['gt_ids'], gt_pos, gt_close, close_idx)
        out[name] = {'logp_gt': round(a, 4), 'p_gt': round(p, 6), 'logp_gt_full': round(full, 4), 'ctx_len': int(ctx.shape[0])}
    return out


def run_group(items, eng, tok, sp, cfg, generate):
    """Step 1 from the blank prompt for every item (batched), then the six cells per item. Fills it['step1'], it['cell']."""
    te, eos = sp['think_end'], sp['eos']
    dev = device_of(eng)
    reqs = [Request(it['prompt']['blank'], frozenset({te, eos}), cfg['think_cap'],
                    torch.Generator(device=dev).manual_seed(seed_of(cfg['seed'], it['pid'], 'step1'))) for it in items]
    t = time.time()
    for it, r in zip(items, generate(reqs) if reqs else []):
        it['nonfinite'] = r is None
        if r is None:
            continue
        z, sep, sep_text, end = split_generated(tok, sp, r.ids, cfg['cut_sep'])
        it['z'], it['sep'] = z, sep
        it['step1'] = {'end': end, 'len': len(z), 'ids': [int(x) for x in z], 'sep_text': sep_text}
    ts = time.time() - t
    for it in items:
        if it['nonfinite']:
            continue
        t = time.time()
        it['cell'] = score_cells(it, eng, sp)
        if any(not math.isfinite(c['logp_gt']) for c in it['cell'].values()):
            it['nonfinite'] = True
        it['timing'] = {'group_step1_s': round(ts, 2), 'cells_s': round(time.time() - t, 2)}


def row_out(it, cfg, version, set_name):
    if it['nonfinite']:
        return error_row(it['pid'], 'generate', 'non-finite logits', version, set_name)
    return {'pid': it['pid'], 'status': 'ok', 'set': set_name, 'question_type': it['row'].get('question_type'),
            'n_visual': int(it['vis_of']['own'].shape[0]), 'answer_gt': it['row']['answer'], 'gt_ids': it['gt_ids'],
            'step1': it['step1'], 'cell': it['cell'], 'code_version': version, 'scorer_version': SCORER_VERSION, 'cfg': dict(cfg),
            'timing': {'encode_s': it.get('encode_s'), **it['timing']}}


def process_group(pids, make_item, eng, tok, sp, cfg, generate, version, set_name):
    rows, items = {}, []
    for pid in pids:
        t0 = time.time()
        try:
            it = make_item(pid)
            it['encode_s'] = round(time.time() - t0, 2)
            items.append(it)
        except Exception as e:
            rows[pid] = error_row(pid, 'load', f'{type(e).__name__}: {e}', version, set_name)
    try:
        run_group(items, eng, tok, sp, cfg, generate)
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
        if int(vis.shape[0]) != doc['n_visual'][pid]:
            raise ValueError(f'n_visual {vis.shape[0]} differs from the ids file {doc["n_visual"][pid]}')
        blank = encode_sample(model, tok, Image.new('RGB', image.size, GRAY), row['query'], cfg['max_num'])[1]
        return {'pid': pid, 'row': row, 'prompt': {'own': base, 'blank': fill_slots(base, pos, blank)},
                'vis_of': {'own': vis, 'blank': blank}, 'gt_ids': gold_ids(tok, row)}

    n_err = 0
    with out.open('a') as f:
        for gi in range(0, len(todo), cfg['group']):
            if a.time_limit and time.time() - t0 > a.time_limit:
                print(json.dumps({'event': 'time_limit', 'left': len(todo) - gi}), flush=True)
                break
            tg = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            out_rows = process_group(todo[gi:gi + cfg['group']], make_item, eng, tok, sp, cfg, generate, version, a.set)
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                if r['status'] != 'ok':
                    n_err += 1
                    print(json.dumps({'event': 'error', 'pid': r['pid'], 'error': r['error']}), flush=True)
            f.flush()
            print(json.dumps({'event': 'group', 'n': len(out_rows), 'n_error': sum(r['status'] != 'ok' for r in out_rows),
                              'group_s': round(time.time() - tg, 1), 'elapsed': round(time.time() - t0),
                              'left': len(todo) - gi - len(out_rows),
                              'peak_alloc_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else 0}),
                  flush=True)
    print(json.dumps({'event': 'end', 'elapsed': round(time.time() - t0), 'n_error': n_err}), flush=True)


if __name__ == '__main__':
    main()
