"""thumb_gate -- pre-registered diagnostic (plan: docs/EXPERIMENT_FLOW.md 2026-10-22 "실행 전 계획: 썸네일 관문 `thumb_gate`"; this file is
code only, written before any result).

Question: with the thumbnail tile removed from the image input (setting "tminus", T-), does re-inserting DPP-selected image
tokens raise the probability of the correct answer string more than with the thumbnail present (setting "tplus", T+ = the
current input)? Image tiles: InternVL cuts the image into 448x448 tiles of 256 image tokens each and, only when there are at
least two tiles, appends one more 448x448 tile (the thumbnail, the whole image shrunk) as the LAST 256 tokens
(visref/model.py dynamic_preprocess / load_image). One-tile images have no thumbnail.

Settings
  T+  exactly the current input: base, vis = encode_sample(..., with_positions=True) (N image tokens, thumbnail included when
      N > 256).
  T-  vis_minus = vis[:-256] when N > 256 (multi-tile; N % 256 == 0 is asserted), else vis unchanged (single tile: the T- input
      is identical to the T+ input, 'same_input' is True). The T- prompt embedding = the T+ prompt embedding with the LAST 256
      image-slot rows deleted (the image slots are contiguous: <img> + N x <IMG_CONTEXT> + </img>). For every multi-tile
      question the run asserts with the tokenizer that tok(build_prompt(q, N-256)).input_ids equals the T+ ids with those 256
      slot positions removed (verify_prompt_ids). Thumbnail tokens = vis[-256:] (multi-tile only).

Contexts (per setting s, after a step-1 thinking text z and its separator sep, untagged; the layout of
diag/inject_sensitivity.contexts, and for T+ tensor-identical to it):
  prefix = [base_s][embed(z + sep)];  tail = embed([</think>] + '\\n\\n\\boxed')
  none   prefix + tail
  dpp    prefix + V_s[sel] + tail      sel = select_tokens(V_s, embed(z), budget(|V_s|, vis_frac)); empty when z is empty
  rand   prefix + V_s[rand] + tail     rand = random_sel(pid, seed, |V_s|, len(sel))
  all    prefix + V_s + tail
  other  prefix + O_s + tail           T+: O = length_matched(donor_vis, N); T-: O = length_matched(strip_thumbnail(donor_vis), |V_minus|)
  thumb  prefix_minus + vis_plus[-256:] + tail      T- and multi-tile only
Reading = ONLY the gold answer string: gt_ids = '{' + gt_text(row) + '}' tokens exactly as inject_sensitivity.part1 builds them
(not the TSR string, not dpp_tag). Per condition: logp_gt (score_at with content_positions, the closing tokens merged), p_gt,
logp_gt_full, top32 (top_k at the first answer position, position 1 after '{'), kl_from_none, top1, entropy_first, ctx_len;
plus kl_pairs {'dpp||rand', 'all||other'}. Per setting and stage also m, sel, sel_rand, n_sel and, for T+, n_sel_thumb (the
number of sel indices >= N-256; 0 for a single tile).

Stages per group of questions, in this order
  1 stored read (descriptive): z, sep, sep_text, recon = split_stored(tok, sp, store[pid], cut_sep); read all contexts of both
    settings on the stored step-1 text.
  2 generation: for each setting a fresh first thinking step from base_s: Request(base_s, {</think>, EOS}, think_cap,
    torch.Generator seeded with seed_of(seed, pid, 'think')); the same seed in both settings. The result becomes
    (z, sep, sep_text, end) by the rule of split_stored: end 'close' when the last id is </think>, 'eos' when it is EOS (z drops
    that id), else 'cap' (z = all ids; sep = cut_sep unless decode(z) ends with a newline). Then the answers: ctx = base_s +
    embed(z + sep + [</think>] + '\\n\\n\\boxed'); n_t1 samples at T=1 without top-k/top-p with seed_of(seed, pid, 'ans', j) and
    one greedy answer, <= probe_cap tokens, stopped when the box closes (answer_record, scorer v4).
  3 new-text read (the main judgment of the plan): for each setting every context read on that setting's OWN new z / sep of
    stage 2.
Chunk: --chunk 0 means chunk = think_cap; a chunk below think_cap is refused at start (SystemExit), and every generate call
raises ChunkError if a result is neither 'stop' nor 'cap' (an unfinished row would be read as a capped thinking). A ChunkError is
a setup fault and stops the run. A question that raises, or whose generation returns non-finite logits, gets a row
{pid, status: 'error', stage, error, set, code_version} and the rest goes on (per-question retry: when a group raises, each
question is run alone). No accuracy and no decision is computed here (see diag/analyze_thumb_gate.py).

Optional --reencode-check (the 5-question trial): for each multi-tile question the image is encoded again WITHOUT the thumbnail
(reencode_minus: dynamic_preprocess(use_thumbnail=False) -> build_transform -> stack -> model.extract_feature) and the row
records max/mean |re-encoded - vis_minus|, max |vis_minus| and the |logp_gt| difference of the stored-text 'none' context built
from the re-encoded tokens vs from vis_minus.

Row layout (JSONL, one row per question): pid, status 'ok', set, stratum (stored think_end), question_type, multi, n_tiles
(448x448 tiles without the thumbnail), n_visual {tplus, tminus}, other_pid, other_n_visual, same_input, code_version,
scorer_version, cfg, recon, gt_ids, answer_gt, stored / new {tplus|tminus: {cond: {...}, kl_pairs, m, sel, sel_rand, n_sel
[, n_sel_thumb]}}, gen {tplus|tminus: {end, len, text, n_think_end_inside, z_ids, sep_ids, sep_text, answers, acc_t1, greedy}},
reencode (only with the flag), timing. Seeds: think seed_of(seed, pid, 'think') (both settings), answers seed_of(seed, pid,
'ans', j), random control seed_of(seed, pid, 'rand') (step1_think.random_sel).

  python3 diag/thumb_gate.py --parquet PQ --out results/thumb_gate_0.jsonl [--set main|trial] [--ids diag/thumb_gate_ids.json]
      [--samples diag/step1_store_bmain.json] [--limit N] [--shard 0/2] [--skip DONE.json] [--time-limit SECONDS]
      [--reencode-check] [--think-cap 1024] [--chunk 0] [--group 4]
  Questions already in --out are skipped (resume).
"""
import argparse
import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'diag'))
import step1_think as t1  # noqa: E402
from inject_sensitivity import (content_positions, entropy, gt_text, kl, length_matched, score_at,  # noqa: E402
                                score_string, top_k)
from step1_think import GREEDY, T1, answer_record, box_close_index, extra_record, random_sel, seed_of, split_stored  # noqa: E402
from visref.dpp import budget, select_tokens  # noqa: E402
from visref.engine import Engine, Request  # noqa: E402
from visref.model import build_prompt  # noqa: E402
from visref.scoring import SCORER_VERSION  # noqa: E402

TILE = 256  # image tokens of one 448x448 tile (and of the thumbnail)
SETTINGS = ('tplus', 'tminus')
CONDS = ('none', 'dpp', 'rand', 'all', 'other', 'thumb')
PAIRS = {'dpp||rand': ('dpp', 'rand'), 'all||other': ('all', 'other')}
FIRST = 1  # index in gt_ids of the first answer position (after the opening brace)
DEFAULTS = dict(vis_frac=0.3, think_cap=1024, probe_prompt='\n\n\\boxed', wait_text='Wait, think more', probe_cap=16, n_t1=3,
                temperature=0.6, top_k=50, top_p=0.95, seed=42, cut_sep='\n', chunk=0, batch_tokens=72000, group=4,
                max_num=12)  # chunk 0 = think_cap (see resolve_cfg)


class ChunkError(RuntimeError):
    """A generate call returned a row that is neither finished ('stop') nor at its cap ('cap'), e.g. finish 'chunk'."""


def code_version():
    """sha256 over this script, the step1_think / inject_sensitivity helpers it imports, run.py and visref/*.py."""
    h = hashlib.sha256()
    files = [Path(__file__).resolve(), ROOT / 'diag/step1_think.py', ROOT / 'diag/inject_sensitivity.py', ROOT / 'run.py',
             *(ROOT / 'visref').glob('*.py')]
    for f in sorted(files):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:12]


def resolve_cfg(cfg):
    """chunk 0 -> think_cap; refuse a chunk below think_cap (rows would come back unfinished and be read as 'cap')."""
    cfg = dict(cfg)
    if cfg['chunk'] == 0:
        cfg['chunk'] = cfg['think_cap']
    if cfg['chunk'] < cfg['think_cap']:
        raise SystemExit(f"--chunk {cfg['chunk']} < --think-cap {cfg['think_cap']}: refused (rows that return finish "
                         "'chunk' are not re-submitted, so the thinking would silently stop at the chunk size)")
    if cfg['think_cap'] < 1 or cfg['probe_cap'] < 1 or cfg['probe_cap'] > cfg['chunk']:
        raise SystemExit('--think-cap and --probe-cap must be >= 1 and --probe-cap <= --chunk')
    return cfg


def make_generate(eng, cfg, pack):
    """step1_think.make_generate plus the check that no result is unfinished: every row must end 'stop' or 'cap'."""
    inner = t1.make_generate(eng, cfg, pack)

    def generate(reqs):
        out = inner(reqs)
        bad = sorted({r.finish for r in out if r is not None and r.finish not in ('stop', 'cap')})
        if bad:
            raise ChunkError(f"generate returned unfinished rows (finish {bad}) with chunk={cfg['chunk']}: refusing to treat "
                             'them as cap')
        return out
    return generate


# ---- the two inputs ---------------------------------------------------------------------------------------------------
def strip_thumbnail(vis):
    """vis without its thumbnail: the first N-256 tokens when N > 256 (N must be a multiple of 256); a single tile (N == 256)
    has no thumbnail and is returned unchanged."""
    n = int(vis.shape[0])
    if n < TILE or n % TILE:
        raise ValueError(f'{n} image tokens: not a positive multiple of {TILE}')
    return vis if n == TILE else vis[:-TILE]


def minus_base(base, pos, n):
    """The T- prompt embedding: base (T+, [L, d]) without the LAST 256 image-slot rows. pos = the positions of the n image
    slots (encode_sample(..., with_positions=True)), which must be contiguous. A single tile (n == 256) returns base itself."""
    if n < TILE or n % TILE or int(pos.numel()) != n:
        raise ValueError(f'{n} image tokens, {int(pos.numel())} image slots')
    if n == TILE:
        return base
    pos = pos.to(torch.long)
    if not bool((pos[1:] - pos[:-1] == 1).all()):
        raise ValueError('image slots are not contiguous')
    keep = torch.ones(base.shape[0], dtype=torch.bool, device=base.device)
    keep[pos[-TILE:]] = False
    return base[keep]


def check_slot_removal(plus_ids, minus_ids, pos):
    """Assert that minus_ids equals plus_ids with the last 256 image-slot positions (pos) removed."""
    drop = {int(i) for i in pos[-TILE:]}
    want = [t for i, t in enumerate(plus_ids) if i not in drop]
    assert list(minus_ids) == want, (len(plus_ids), len(minus_ids), len(want))


def verify_prompt_ids(tok, query, n, pos):
    """The tokenizer check of the plan, for a multi-tile question: tok(build_prompt(q, n-256)).input_ids == the ids of
    build_prompt(q, n) without the last 256 image-slot positions."""
    plus = tok(build_prompt(query, n), return_tensors='pt').input_ids[0].tolist()
    minus = tok(build_prompt(query, n - TILE), return_tensors='pt').input_ids[0].tolist()
    check_slot_removal(plus, minus, pos)


def make_settings(base, vis, pos, vis_other=None):
    """{'tplus': {...}, 'tminus': {...}}: base, V, the donor's control tokens O and (T-, multi-tile) the thumbnail tokens."""
    n = int(vis.shape[0])
    vm = strip_thumbnail(vis)
    s = {'tplus': {'base': base, 'vis': vis, 'thumb': None,
                   'other': None if vis_other is None else length_matched(vis_other, n)},
         'tminus': {'base': minus_base(base, pos, n), 'vis': vm, 'thumb': vis[-TILE:] if n > TILE else None,
                    'other': None if vis_other is None else length_matched(strip_thumbnail(vis_other), int(vm.shape[0]))}}
    return s


# ---- reading ---------------------------------------------------------------------------------------------------------
def contexts(pid, st, z, sep, eng, sp, cfg):
    """{cond: context ending with the probe prompt} for one setting st (an entry of make_settings) and the thinking z + sep,
    and the selections {'m', 'sel', 'sel_rand'}. For T+ identical to inject_sensitivity.contexts (none, dpp, rand, all, other)."""
    base, vis = st['base'], st['vis']
    prefix = torch.cat([base, eng.embed(z + sep)], 0)
    m = budget(vis.shape[0], cfg['vis_frac'])
    sel = [int(i) for i in select_tokens(vis, eng.embed(z), m)] if z else []
    rand = random_sel(pid, cfg['seed'], int(vis.shape[0]), len(sel))
    tail = eng.embed([sp['think_end']] + sp['probe_prompt'])
    inj = {'none': vis[:0], 'dpp': vis[sel], 'rand': vis[rand], 'all': vis, 'other': st['other'], 'thumb': st['thumb']}
    return ({c: torch.cat([prefix, inj[c], tail], 0) for c in CONDS if inj[c] is not None},
            {'m': m, 'sel': sel, 'sel_rand': rand})


def gt_ids_of(tok, row):
    """'{' + gold answer string + '}' as inject_sensitivity.part1 builds it."""
    brace = tok.encode('{', add_special_tokens=False)
    assert len(brace) == 1, brace
    return brace + tok.encode(gt_text(row) + '}', add_special_tokens=False)


def read_contexts(eng, sp, ctxs, gt_ids):
    """The reading of the plan for every context of ctxs ('none' must be among them): {'cond': {c: {...}}, 'kl_pairs': {...}}.
    The values are those of inject_sensitivity.part1 for the gold string."""
    close_idx = torch.tensor(sorted(sp['close_ids']))
    pos, close = content_positions(gt_ids, sp['brace_text'])
    rec, dist, ref = {}, {}, None
    for c in CONDS:
        if c not in ctxs:
            continue
        lg, logp = score_string(eng, ctxs[c], gt_ids)
        d = logp[FIRST]
        dist[c] = d
        if c == 'none':
            ref = d
        a = score_at(logp, gt_ids, pos, close, close_idx)
        rec[c] = {'logp_gt': round(a, 4), 'p_gt': round(math.exp(a), 6), 'logp_gt_full': round(lg, 4), 'top32': top_k(d),
                  'kl_from_none': round(kl(ref, d), 5), 'top1': int(d.argmax()), 'entropy_first': round(entropy(d), 4),
                  'ctx_len': int(ctxs[c].shape[0])}
    return {'cond': rec, 'kl_pairs': {k: round(kl(dist[x], dist[y]), 5) for k, (x, y) in PAIRS.items() if x in dist and y in dist}}


def read_stage(it, s, z, sep, eng, sp, cfg):
    """One stage's record of setting s on the thinking z + sep: the reading plus m, sel, sel_rand, n_sel (T+: n_sel_thumb)."""
    ctxs, meta = contexts(it['pid'], it['S'][s], z, sep, eng, sp, cfg)
    rec = read_contexts(eng, sp, ctxs, it['gt_ids'])
    rec.update(meta, n_sel=len(meta['sel']))
    if s == 'tplus':
        rec['n_sel_thumb'] = sum(i >= it['n_vis'] - TILE for i in meta['sel']) if it['multi'] else 0
    return rec


# ---- re-encoding check -----------------------------------------------------------------------------------------------
def reencode_minus(model, image, max_num=12, transform=None, size=448):
    """The image encoded again WITHOUT the thumbnail tile: dynamic_preprocess(use_thumbnail=False) -> transform -> stack ->
    model.extract_feature -> [N-256, d]. transform defaults to visref.model.build_transform (needs torchvision)."""
    from visref.model import build_transform, dynamic_preprocess
    lm = model.language_model
    transform = transform or build_transform(size)
    tiles = dynamic_preprocess(image, image_size=size, use_thumbnail=False, max_num=max_num)
    pv = torch.stack([transform(t) for t in tiles]).to(lm.device, dtype=lm.dtype)
    return model.extract_feature(pv).reshape(-1, lm.config.hidden_size)


def gt_logp(eng, sp, ctx, gt_ids):
    """Unrounded log p of the gold string (score_at) after the context ctx."""
    pos, close = content_positions(gt_ids, sp['brace_text'])
    _, logp = score_string(eng, ctx, gt_ids)
    return score_at(logp, gt_ids, pos, close, torch.tensor(sorted(sp['close_ids'])))


def reencode_record(it, eng, sp, cfg):
    """Re-encoded vs stored T- image tokens, and what it does to log p(gold) of the stored-text 'none' context."""
    vr, vm = it['vis_re'], it['S']['tminus']['vis']
    if vr.shape != vm.shape:
        return {'error': f'shape {tuple(vr.shape)} != {tuple(vm.shape)}'}
    d = (vr.float() - vm.float()).abs()
    base = it['S']['tminus']['base']
    base_re = base.clone()
    base_re[it['pos'][:it['n_vis'] - TILE]] = vr.to(base.dtype)
    tail = eng.embed([sp['think_end']] + sp['probe_prompt'])
    pre = eng.embed(it['z'] + it['sep'])
    a = gt_logp(eng, sp, torch.cat([base, pre, tail], 0), it['gt_ids'])
    b = gt_logp(eng, sp, torch.cat([base_re, pre, tail], 0), it['gt_ids'])
    return {'max_abs_diff': float(d.max()), 'mean_abs_diff': float(d.mean()), 'max_abs_vis_minus': float(vm.float().abs().max()),
            'logp_gt_none_minus': round(a, 4), 'logp_gt_none_reencoded': round(b, 4), 'abs_dlogp_gt_none': abs(a - b)}


# ---- generation ------------------------------------------------------------------------------------------------------
def split_generated(tok, sp, ids, cut_sep):
    """Free first thinking step -> (z, sep, sep_text, end), the rule of step1_think.split_stored: 'close' when the last id is
    </think>, 'eos' when it is EOS (z drops it), else 'cap'; a cut step gets cut_sep unless its text ends with a newline."""
    ids = list(ids)
    if ids and ids[-1] == sp['think_end']:
        end, z = 'close', ids[:-1]
    elif ids and ids[-1] == sp['eos']:
        end, z = 'eos', ids[:-1]
    else:
        end, z = 'cap', ids
    sep_text = cut_sep if end == 'cap' and cut_sep and not tok.decode(z).endswith('\n') else ''
    sep = tok.encode(sep_text, add_special_tokens=False) if sep_text else []
    return z, sep, sep_text, end


def gen_record(tok, sp, ids, cut_sep):
    z, sep, sep_text, end = split_generated(tok, sp, ids, cut_sep)
    rec = extra_record(tok, sp, ids, end)
    del rec['ids']  # z_ids holds the same tokens (without a final </think> / EOS)
    rec.update(z_ids=z, sep_ids=sep, sep_text=sep_text)
    return rec


def reset(it):
    for k in ('stored', 'gen', 'new', 'reencode', 'nonfinite', 'timing'):
        it.pop(k, None)
    it.update(stored={}, gen={}, new={}, nonfinite=[], timing={})


def run_group(items, eng, tok, sp, cfg, generate, reencode=False, timing=None):
    """items: dicts with pid, row, S (make_settings), pos, n_vis, multi, gt_ids, z, sep, sep_text, recon (and vis_re with
    reencode). Adds in place: stored[s], reencode, gen[s], new[s] (s in SETTINGS), nonfinite. timing (dict) gets the group's
    seconds of 'stored', 'generation', 'answers', 'new'."""
    te, eos = sp['think_end'], sp['eos']
    dev = eng.emb.weight.device
    timing = {} if timing is None else timing
    for k in ('stored', 'generation', 'answers', 'new'):
        timing[k] = 0.0
    for it in items:
        reset(it)
    for it in items:  # stage 1: stored read
        t = time.time()
        for s in SETTINGS:
            it['stored'][s] = read_stage(it, s, it['z'], it['sep'], eng, sp, cfg)
        if reencode and it.get('vis_re') is not None:
            it['reencode'] = reencode_record(it, eng, sp, cfg)
        it['timing']['stored_read_s'] = round(time.time() - t, 2)
        timing['stored'] += time.time() - t
    t = time.time()  # stage 2: first thinking step of each setting, same seed
    reqs, keys = [], []
    for it in items:
        for s in SETTINGS:
            g = torch.Generator(device=dev).manual_seed(seed_of(cfg['seed'], it['pid'], 'think'))
            reqs.append(Request(it['S'][s]['base'], frozenset({te, eos}), cfg['think_cap'], g))
            keys.append((it, s))
    for (it, s), r in zip(keys, generate(reqs)):
        if r is None:
            it['nonfinite'].append(s)
        else:
            it['gen'][s] = gen_record(tok, sp, r.ids, cfg['cut_sep'])
    timing['generation'] = time.time() - t
    t = time.time()  # answers after the closed first step
    live = [it for it in items if not it['nonfinite']]
    stop = lambda ids, bt=sp['brace_text']: box_close_index(ids, bt) is not None  # noqa: E731
    reqs, keys = [], []
    for it in live:
        for s in SETTINGS:
            g_ = it['gen'][s]
            ctx = torch.cat([it['S'][s]['base'], eng.embed(g_['z_ids'] + g_['sep_ids'] + [te] + sp['probe_prompt'])], 0)
            for j in range(cfg['n_t1'] + 1):
                greedy = j == cfg['n_t1']
                g = None if greedy else torch.Generator(device=dev).manual_seed(seed_of(cfg['seed'], it['pid'], 'ans', j))
                reqs.append(Request(ctx, frozenset({eos}), cfg['probe_cap'], g, sampling=GREEDY if greedy else T1, stop_check=stop))
                keys.append((it, s, j))
    for (it, s, j), r in zip(keys, generate(reqs) if reqs else []):
        if r is None:
            if s not in it['nonfinite']:
                it['nonfinite'].append(s)
            continue
        rec = answer_record(tok, sp, r.ids, it['row'])
        rec['greedy'] = j == cfg['n_t1']
        it['gen'][s].setdefault('answers', []).append(rec)
    for it in live:
        for s in SETTINGS:
            ans = it['gen'][s].get('answers', [])
            if s in it['nonfinite'] or len(ans) != cfg['n_t1'] + 1:
                continue
            it['gen'][s]['acc_t1'] = sum(a['correct'] for a in ans if not a['greedy']) / cfg['n_t1']
            it['gen'][s]['greedy'] = next(a['correct'] for a in ans if a['greedy'])
    timing['answers'] = time.time() - t
    for it in live:  # stage 3: new-text read
        if it['nonfinite']:
            continue
        t = time.time()
        for s in SETTINGS:
            g_ = it['gen'][s]
            it['new'][s] = read_stage(it, s, g_['z_ids'], g_['sep_ids'], eng, sp, cfg)
        it['timing']['new_read_s'] = round(time.time() - t, 2)
        timing['new'] += time.time() - t


# ---- rows ------------------------------------------------------------------------------------------------------------
def error_row(pid, stage, msg, version, set_name):
    return {'pid': pid, 'status': 'error', 'stage': stage, 'error': str(msg)[:500], 'set': set_name, 'code_version': version}


def row_out(it, cfg, version, set_name, reencode=False):
    """The result row of one finished item; an error row when any generation of it returned non-finite logits."""
    if it['nonfinite']:
        row = error_row(it['pid'], 'generate', 'non-finite logits in ' + ','.join(sorted(set(it['nonfinite']))), version, set_name)
        row['nonfinite'] = sorted(set(it['nonfinite']))
        return row
    n = it['n_vis']
    row = {'pid': it['pid'], 'status': 'ok', 'set': set_name, 'stratum': it['sample']['think_end'],
           'question_type': it['row'].get('question_type'), 'multi': it['multi'],
           'n_tiles': n // TILE - 1 if it['multi'] else 1,
           'n_visual': {s: int(it['S'][s]['vis'].shape[0]) for s in SETTINGS},
           'other_pid': it.get('other_pid'), 'other_n_visual': it.get('other_n_visual'),
           'same_input': bool(torch.equal(it['S']['tplus']['base'], it['S']['tminus']['base'])),
           'code_version': version, 'scorer_version': SCORER_VERSION, 'cfg': dict(cfg), 'recon': it['recon'],
           'gt_ids': it['gt_ids'], 'answer_gt': it['row']['answer'], 'stored': it['stored'], 'gen': it['gen'], 'new': it['new']}
    if reencode:
        row['reencode'] = it.get('reencode')
    row['timing'] = {'encode_s': it.get('encode_s'), **it['timing']}
    return row


def process_group(pids, make_item, eng, tok, sp, cfg, generate, version, set_name, reencode=False, timing=None):
    """Rows (ok or error) for pids, in order. A question that raises gets an error row and the rest goes on: when the group
    raises, every question is re-run alone. A ChunkError stops the run."""
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
        run_group(items, eng, tok, sp, cfg, generate, reencode, timing)
    except ChunkError:
        raise
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for it in items:
            try:
                run_group([it], eng, tok, sp, cfg, generate, reencode)
            except ChunkError:
                raise
            except Exception as e:
                rows[it['pid']] = error_row(it['pid'], 'generate', f'{type(e).__name__}: {e}', version, set_name)
    for it in items:
        if it['pid'] not in rows:
            rows[it['pid']] = row_out(it, cfg, version, set_name, reencode)
    return [rows[p] for p in pids]


def select_pids(doc, set_name, limit=0, shard='0/1', done=()):
    """The pids this process runs: the set's list, first `limit`, the shard's share of that list by index, without the done ones."""
    if set_name not in ('main', 'trial') or set_name not in doc:
        raise SystemExit(f"no set {set_name!r} in the ids file")
    ids = [str(p) for p in doc[set_name]]
    if limit:
        ids = ids[:limit]
    si, sn = map(int, shard.split('/'))
    ids = [p for i, p in enumerate(ids) if i % sn == si]
    done = {str(x) for x in done}
    return [p for p in ids if p not in done]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--model', default='OpenGVLab/InternVL3_5-1B')
    ap.add_argument('--revision', default='2f71cf52542334823e48a46ffba0e2bc9add3446')
    ap.add_argument('--samples', default=str(ROOT / 'diag/step1_store_bmain.json'))
    ap.add_argument('--ids', default=str(ROOT / 'diag/thumb_gate_ids.json'))
    ap.add_argument('--set', default='main', choices=('main', 'trial'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--skip', default='', help='JSON list of pids finished earlier')
    ap.add_argument('--time-limit', type=float, default=0, help='seconds; stop starting new groups after this')
    ap.add_argument('--reencode-check', action='store_true', help='re-encode multi-tile images without the thumbnail (trial)')
    for k, v in DEFAULTS.items():
        ap.add_argument('--' + k.replace('_', '-'), type=type(v), default=v)
    a = ap.parse_args(argv)
    a.cfg = resolve_cfg({k: getattr(a, k) for k in DEFAULTS})
    return a


def read_rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def encode_item(model, tok, row, cfg):
    """Prompt embedding, image tokens and image-slot positions of one question, plus the tokenizer check of the T- prompt for a
    multi-tile image."""
    from visref.model import encode_sample
    base, vis, pos = encode_sample(model, tok, row['image'](), row['query'], cfg['max_num'], with_positions=True)
    if vis.shape[0] > TILE:
        verify_prompt_ids(tok, row['query'], int(vis.shape[0]), pos)
    return base, vis, pos


def main(argv=None):
    a = parse_args(argv)
    cfg = a.cfg
    t0 = time.time()
    from run import fast_kv_selfcheck, pack
    from visref.model import load_mathvista, load_model, special_ids
    samples = json.loads(Path(a.samples).read_text())
    samples = samples.get('pids', samples)
    doc = json.loads(Path(a.ids).read_text())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r['pid'] for r in read_rows(out)}
    n_done = len(done)
    if a.skip:
        done |= {str(x) for x in json.loads(Path(a.skip).read_text())}
    todo = select_pids(doc, a.set, a.limit, a.shard, done)
    for p in todo:
        assert p in samples, f'no stored step 1 for {p}'
        assert p in doc['donor'], f'no donor for {p}'
        assert doc['stratum'][p] == samples[p]['think_end'], (p, doc['stratum'][p], samples[p]['think_end'])
    version = code_version()
    print(json.dumps({'event': 'start', 'code_version': version, 'set': a.set, 'todo': len(todo), 'done': n_done,
                      'reencode_check': a.reencode_check, 'cfg': cfg}), flush=True)
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
        s, row = samples[pid], rows[pid]
        base, vis, pos = encode_item(model, tok, row, cfg)
        d = doc['donor'][pid]
        o = rows[d]
        vis_other = encode_item(model, tok, o, cfg)[1]
        z, sep, sep_text, recon = split_stored(tok, sp, s, cfg['cut_sep'])
        n = int(vis.shape[0])
        it = {'pid': pid, 'row': row, 'sample': s, 'S': make_settings(base, vis, pos, vis_other), 'pos': pos, 'n_vis': n,
              'multi': n > TILE, 'gt_ids': gt_ids_of(tok, row), 'z': z, 'sep': sep, 'sep_text': sep_text, 'recon': recon,
              'other_pid': d, 'other_n_visual': int(vis_other.shape[0])}
        if a.reencode_check and n > TILE:
            it['vis_re'] = reencode_minus(model, row['image'](), cfg['max_num'])
        return it

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
            out_rows = process_group(todo[gi:gi + cfg['group']], make_item, eng, tok, sp, cfg, generate, version, a.set,
                                     a.reencode_check, tm)
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                if r['status'] != 'ok':
                    n_err += 1
                    print(json.dumps({'event': 'error', 'pid': r['pid'], 'error': r['error']}), flush=True)
            f.flush()
            peak = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else 0
            peak_all = max(peak_all, peak)
            print(json.dumps({'event': 'group', 'n': len(out_rows), 'n_error': sum(r['status'] != 'ok' for r in out_rows),
                              'encode_s': round(tm.get('encode', 0), 1), 'stored_read_s': round(tm.get('stored', 0), 1),
                              'generation_s': round(tm.get('generation', 0), 1), 'answers_s': round(tm.get('answers', 0), 1),
                              'new_read_s': round(tm.get('new', 0), 1), 'group_s': round(time.time() - tg, 1),
                              'elapsed': round(time.time() - t0), 'left': len(todo) - gi - len(out_rows), 'peak_alloc_gb': peak,
                              'max_rss_gb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)}), flush=True)
    print(json.dumps({'event': 'end', 'elapsed': round(time.time() - t0), 'n_error': n_err, 'peak_alloc_gb': peak_all,
                      'max_rss_gb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)}), flush=True)


if __name__ == '__main__':
    main()
