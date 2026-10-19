"""recover_check -- pre-registered, generation-free diagnostic (plan docs/EXPERIMENT_FLOW.md 2026-10-19 "실행 전 계획: 회복 시험", written
before any result; this file is code only).

Question: does a re-inserted image block hand information to the model, i.e. when the image slots of the prompt hold no
information (same-size gray image) or another question's image, does re-inserting the right image after a short fixed
thinking sentence bring back the probability of the gold answer string? And does the position-id scheme of a re-inserted
subset matter? Nothing is generated; only next-token probabilities after '</think>' + '\\n\\n\\boxed' are read, exactly as
diag/inject_sensitivity.py reads them (score_at, content_positions, gt_text, kl, entropy are imported from there).

Context of every condition:  [prompt up to the end of '<|im_start|>assistant\\n', its image slots filled per condition]
  + FIXED (THINK_OPEN + SENTENCE; the model writes '<think>\\n' itself, it is part of the generated step-1 text and not of the
    prompt: visref/model.py:48-52 build_prompt ends with '<|im_start|>assistant\\n', diag/step1_store_bmain.json 'thinking'
    starts with '<think>\\n' in 975 of 1000 questions, visref/methods.py:82-96 stored_step1 re-tokenises that text as one
    string; so the fixed sentence starts with the same '<think>\\n')
  + RE-INSERTED image tokens (none / image / subset, see CONDS)
  + '</think>' + '\\n\\n\\boxed'  then the gold answer '{' + answer + '}' is teacher-forced.
Image slots (the <IMG_CONTEXT> embedding rows, visref.model.encode_sample(..., with_positions=True)) are overwritten with
the embeddings of: the question's own image (A, A+), a same-size gray image Image.new('RGB', size, (128, 128, 128))
(blank), or donor1's image. Re-inserted: the question's own image (all tokens or the random 30% subset) or donor2's image
(all tokens, or the tokens with the same indices). Every image must give as many tokens as the question's own, else
ValueError.

Position ids: prompt + FIXED = 0..P-1; the subset s_1 < ... < s_n of N image tokens (n = floor(0.3 N)):
  n  (Cn, Dn)  consecutive P..P+n-1, tail P+n...
  o  (Co, Do)  token j at P + (s_j - s_1), tail from P + (s_n - s_1) + 1
  e  (Ce, De)  consecutive, shifted so that the last subset token has Co's last id; tail as Co
  all other contexts (including the full re-insertions Cb, Db, A+, C, D) consecutive 0, 1, 2, ...
The answer tokens continue from the last context position + 1 (not from the context length).

  python3 diag/recover_check.py --parquet PQ --out results/recover_check_0.jsonl [--set main|trial] [--limit N]
      [--shard 0/2] [--time-limit SECONDS] [--check]
  --check (plan: "계산 점검", on the trial questions by default): numeric comparison 1 (chunked last_hidden_pos vs a one-shot
  forward with an explicit all-ones attention mask and the same position ids, contexts Cb and Co; pass = |d logp_gt| < 0.1
  and KL at the first answer position < 0.01 nats) and the descriptive comparison 2 (Engine.generate batched with a shorter
  dummy row, max_new 1, fastkv on and off, entropy vs one-shot); writes 'check' rows, runs no condition.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import DynamicCache

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'diag'))
from inject_sensitivity import content_positions, entropy, gt_text, kl, score_at  # noqa: E402
from recover_check_ids import subset_of, subset_size  # noqa: E402
from step1_think import seed_of  # noqa: E402,F401  (the subset rule: recover_check_ids.subset_of uses it)
from visref.engine import Engine, Request  # noqa: E402
from visref.scoring import SCORER_VERSION  # noqa: E402

THINK_OPEN = '<think>\n'
SENTENCE = 'Let me look at the image again.\n'
GRAY = (128, 128, 128)
# name: (what fills the prompt's image slots, what is re-inserted, position scheme)
CONDS = {'A': ('own', None, None), 'Bb': ('blank', None, None), 'Cb': ('blank', 'own', None), 'Db': ('blank', 'd2', None),
         'A+': ('own', 'own', None),
         'Cn': ('blank', 'own_sub', 'n'), 'Co': ('blank', 'own_sub', 'o'), 'Ce': ('blank', 'own_sub', 'e'),
         'Dn': ('blank', 'd2_sub', 'n'), 'Do': ('blank', 'd2_sub', 'o'), 'De': ('blank', 'd2_sub', 'e'),
         'B': ('d1', None, None), 'C': ('d1', 'own', None), 'D': ('d1', 'd2', None)}
KL_PAIRS = {'Cb||Db': ('Cb', 'Db'), 'Cn||Dn': ('Cn', 'Dn'), 'Co||Do': ('Co', 'Do'), 'Ce||De': ('Ce', 'De'), 'C||D': ('C', 'D')}
DEFAULTS = dict(probe_prompt='\n\n\\boxed', wait_text='Wait, think more', temperature=0.6, top_k=50, top_p=0.95, seed=42,
                max_num=12, group=4)
CHECK_LOGP, CHECK_KL = 0.1, 0.01
FIRST = 1  # index (in the teacher-forced answer) of the first answer position after the opening brace


def code_version():
    """sha256 over this script, the diag helpers it imports, run.py and visref/*.py."""
    h = hashlib.sha256()
    files = [Path(__file__).resolve(), ROOT / 'diag/inject_sensitivity.py', ROOT / 'diag/step1_think.py',
             ROOT / 'diag/recover_check_ids.py', ROOT / 'run.py', *(ROOT / 'visref').glob('*.py')]
    for f in sorted(files):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:12]


# ---- reading with explicit position ids -------------------------------------------------------------------------------
@torch.no_grad()
def last_hidden_pos(eng, ctx, n, pos=None):
    """Engine.last_hidden (visref/engine.py:82-100: same DynamicCache, same chunk size) with explicit position ids.
    pos: 1-D LongTensor of length L (default arange). The cache is always passed, so transformers 4.55.0 never runs its
    packed-sequence detection on skipped position ids (masking_utils.py:717 needs past_key_values is None)."""
    L = ctx.shape[0]
    if pos is None:
        pos = torch.arange(L, device=ctx.device)
    assert torch.is_tensor(pos) and pos.dtype == torch.long and pos.dim() == 1 and pos.shape[0] == L, \
        (getattr(pos, 'dtype', None), getattr(pos, 'shape', None), L)
    pos = pos.to(ctx.device)[None]
    n = min(int(n), L)
    if n <= 0:
        return ctx[:0]
    dec = eng.lm.get_decoder()
    cache = DynamicCache()
    q = max(16, eng.attn_budget // (eng.lm.config.num_attention_heads * L))
    outs = []
    for s in range(0, L, q):
        e = min(L, s + q)
        o = dec(inputs_embeds=ctx[None, s:e], position_ids=pos[:, s:e], past_key_values=cache, use_cache=True)
        if e > L - n:
            outs.append(o.last_hidden_state[0, max(s, L - n) - s:])
    return torch.cat(outs, 0)


def answer_positions(pos_ctx, n_ids):
    """Position ids of the n_ids teacher-forced answer tokens: they continue from the last context id + 1."""
    return pos_ctx[-1] + 1 + torch.arange(n_ids, device=pos_ctx.device)


@torch.no_grad()
def score_string_pos(eng, ctx, ids, pos_ctx):
    """inject_sensitivity.score_string with position ids: (sum of log p over ids, log-prob vector at each position)."""
    assert pos_ctx.shape[0] == ctx.shape[0]
    full = torch.cat([ctx, eng.embed(ids)], 0)
    pos = torch.cat([pos_ctx, answer_positions(pos_ctx, len(ids))])
    h = last_hidden_pos(eng, full, len(ids) + 1, pos)[:-1]      # h[i] predicts ids[i]
    logp = torch.log_softmax(eng.lm.lm_head(h).float(), -1)
    return float(sum(logp[i, t] for i, t in enumerate(ids))), logp


# ---- contexts ---------------------------------------------------------------------------------------------------------
def fill_slots(base, img_pos, vis):
    """Copy of the prompt embeddings with the <IMG_CONTEXT> rows replaced by vis [N, d]; N must match the slots."""
    if vis.shape[0] != img_pos.numel():
        raise ValueError(f'image gives {vis.shape[0]} tokens, the prompt has {img_pos.numel()} image slots')
    out = base.clone()
    out[img_pos] = vis.to(out.dtype)
    return out


def make_positions(P, n_ins, tail_len, mode=None, s=None):
    """(position ids [P + n_ins + tail_len], first tail position). mode None/'n': everything consecutive; 'o': inserted
    token j at P + (s_j - s_1), tail after the last; 'e': inserted tokens consecutive and ending at Co's last id."""
    head = torch.arange(P)
    if mode in (None, 'n') or n_ins == 0:
        ins = P + torch.arange(n_ins)
    else:
        assert mode in ('o', 'e')
        s = torch.as_tensor(s, dtype=torch.long)
        assert s.numel() == n_ins and bool((s[1:] > s[:-1]).all())
        last = P + int(s[-1] - s[0])
        ins = (P + (s - s[0])) if mode == 'o' else last - (n_ins - 1) + torch.arange(n_ins)
    t0 = int(ins[-1]) + 1 if n_ins else P
    return torch.cat([head, ins, t0 + torch.arange(tail_len)]), t0


def build_contexts(base, img_pos, vis, blank, d1, d2, sub, fixed, tail):
    """{name: (context embeddings [L, d], position ids [L], info)} for the 14 conditions. base/img_pos: the question's
    prompt embeddings and its <IMG_CONTEXT> positions; vis/blank/d1/d2: own, gray, donor1 and donor2 image tokens [N, d]
    (each must have N = len(img_pos) tokens, else ValueError); sub: sorted subset indices; fixed/tail: embeddings of the
    fixed sentence (with the think-open token) and of '</think>' + probe prompt."""
    n_slot = img_pos.numel()
    for nm, v in (('own', vis), ('blank', blank), ('donor1', d1), ('donor2', d2)):
        if v.shape[0] != n_slot:
            raise ValueError(f'{nm} image has {v.shape[0]} tokens, the own image {n_slot}')
    assert len(sub) == subset_size(n_slot) and list(sub) == sorted(set(sub)) and 0 <= sub[0] and sub[-1] < n_slot
    slot = {'own': base, 'blank': fill_slots(base, img_pos, blank), 'd1': fill_slots(base, img_pos, d1)}
    ins = {None: vis[:0], 'own': vis, 'd2': d2, 'own_sub': vis[sub], 'd2_sub': d2[sub]}
    P = base.shape[0] + fixed.shape[0]
    out = {}
    for name, (sl, re, mode) in CONDS.items():
        ctx = torch.cat([slot[sl], fixed, ins[re], tail], 0)
        pos, t0 = make_positions(P, ins[re].shape[0], tail.shape[0], mode, sub)
        assert pos.shape[0] == ctx.shape[0] and bool((pos[1:] > pos[:-1]).all())
        out[name] = (ctx, pos, {'ctx_len': int(ctx.shape[0]), 'max_pos': int(pos[-1]), 'tail_pos0': t0,
                                'n_ins': int(ins[re].shape[0])})
    for x in 'noe':      # D has the same ids and length as C; Cn/Co/Ce hold the same embeddings
        assert torch.equal(out['C' + x][1], out['D' + x][1]) and out['C' + x][0].shape == out['D' + x][0].shape
    assert torch.equal(out['Cn'][0], out['Co'][0]) and torch.equal(out['Cn'][0], out['Ce'][0])
    assert out['Co'][2]['tail_pos0'] == out['Ce'][2]['tail_pos0']
    assert torch.equal(out['Co'][1][-tail.shape[0]:], out['Ce'][1][-tail.shape[0]:])  # same tail ids
    assert torch.equal(out['Cb'][1], out['Db'][1]) and out['Cb'][0].shape == out['A+'][0].shape == out['Db'][0].shape
    return out


# ---- reading ----------------------------------------------------------------------------------------------------------
def read_cond(logp, gt_ids, gt_pos, gt_close, close_idx):
    """(log p of the gold answer string with the box closing merged, p, distribution at the first answer position)."""
    a = score_at(logp, gt_ids, gt_pos, gt_close, close_idx)
    return a, math.exp(a), logp[FIRST]


def score_question(eng, ctxs, gt_ids, sp):
    """{cond: record} and {pair: KL at the first answer position} for every context in ctxs."""
    close_idx = torch.tensor(sorted(sp['close_ids']))
    gt_pos, gt_close = content_positions(gt_ids, sp['brace_text'])
    rec, dist = {}, {}
    for c, (ctx, pos, info) in ctxs.items():
        full, logp = score_string_pos(eng, ctx, gt_ids, pos)
        a, p, d = read_cond(logp, gt_ids, gt_pos, gt_close, close_idx)
        dist[c] = d
        top1 = int(d.argmax())
        rec[c] = {'logp_gt': round(a, 4), 'p_gt': round(p, 6), 'logp_gt_full': round(full, 4), 'top1': top1,
                  'top1_is_gt_first': top1 == gt_ids[FIRST], 'entropy_first': round(entropy(d), 4), **info}
    kls = {k: round(kl(dist[x], dist[y]), 5) for k, (x, y) in KL_PAIRS.items() if x in dist and y in dist}
    return rec, kls


def gold_ids(tok, row):
    brace = tok.encode('{', add_special_tokens=False)
    assert len(brace) == 1, brace
    return brace + tok.encode(gt_text(row) + '}', add_special_tokens=False)


# ---- numeric check (plan: "계산 점검") -------------------------------------------------------------------------------------
@torch.no_grad()
def oneshot_logp(eng, ctx, ids, pos_ctx):
    """One forward of the whole sequence without a cache but with an explicit all-ones 2-D attention mask (so that the
    skipped position ids are not read as packed sequences), same position ids; log-probs as score_string_pos."""
    full = torch.cat([ctx, eng.embed(ids)], 0)
    pos = torch.cat([pos_ctx, answer_positions(pos_ctx, len(ids))])
    L = full.shape[0]
    o = eng.lm.get_decoder()(inputs_embeds=full[None], attention_mask=torch.ones(1, L, dtype=torch.long, device=full.device),
                             position_ids=pos[None].to(full.device), use_cache=False)
    h = o.last_hidden_state[0, L - len(ids) - 1:L - 1]
    return torch.log_softmax(eng.lm.lm_head(h).float(), -1)


def numeric_check(eng, ctxs, gt_ids, sp, conds=('Cb', 'Co')):
    """Comparison 1 (decides) and 2 (descriptive) for one question; never raises."""
    from visref.fastkv import set_enabled
    close_idx = torch.tensor(sorted(sp['close_ids']))
    gt_pos, gt_close = content_positions(gt_ids, sp['brace_text'])
    out = {'cmp1': {}, 'cmp2': {}}
    for c in conds:
        ctx, pos, _ = ctxs[c]
        try:
            _, lc = score_string_pos(eng, ctx, gt_ids, pos)
            lo = oneshot_logp(eng, ctx, gt_ids, pos)
            a, _, dc = read_cond(lc, gt_ids, gt_pos, gt_close, close_idx)
            b, _, do = read_cond(lo, gt_ids, gt_pos, gt_close, close_idx)
            d_logp, d_kl = abs(a - b), kl(do, dc)
            out['cmp1'][c] = {'logp_chunked': round(a, 4), 'logp_oneshot': round(b, 4), 'abs_dlogp': round(d_logp, 5),
                              'kl_first': round(d_kl, 6), 'ctx_len': int(ctx.shape[0]),
                              'pass': bool(d_logp < CHECK_LOGP and d_kl < CHECK_KL)}
        except Exception as e:  # cmp 1 not computable counts as not passed, the run goes on
            out['cmp1'][c] = {'pass': False, 'error': f'{type(e).__name__}: {e}'[:300]}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    out['cmp1_pass'] = all(v['pass'] for v in out['cmp1'].values()) and len(out['cmp1']) == len(conds)
    ctx, pos, _ = ctxs['Cb']
    try:
        ref = oneshot_logp(eng, ctx, gt_ids[:1], pos)[0]          # distribution at the last context position
    except Exception as e:
        ref, out['cmp2']['error'] = None, f'{type(e).__name__}: {e}'[:300]
    prev = eng.fast_kv
    for fast in (True, False):
        key = 'fastkv_on' if fast else 'fastkv_off'
        try:
            if ref is None:
                raise RuntimeError('one-shot reference missing')
            set_enabled(fast)
            e2 = Engine(eng.lm, 0.0, 0, 1.0, fast_kv=fast)
            e2.attn_budget = eng.attn_budget
            dummy = ctx[ctx.shape[0] // 3:]                       # shorter row -> the first row is left-padded
            reqs = [Request(ctx, frozenset(), 1, None, record_entropy=True, sampling=(0.0, 0, 1.0)),
                    Request(dummy, frozenset(), 1, None, record_entropy=True, sampling=(0.0, 0, 1.0))]
            r = e2.generate(reqs, 1)[0]
            h_ref = entropy(ref)
            out['cmp2'][key] = {'entropy_generate': round(r.ent[0], 5), 'entropy_oneshot': round(h_ref, 5),
                                'abs_dentropy': round(abs(r.ent[0] - h_ref), 5), 'top1_same': r.ids[0] == int(ref.argmax())}
        except Exception as e:
            out['cmp2'][key] = {'error': f'{type(e).__name__}: {e}'[:300]}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    set_enabled(prev)
    return out


# ---- run --------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--model', default='OpenGVLab/InternVL3_5-1B')
    ap.add_argument('--revision', default='2f71cf52542334823e48a46ffba0e2bc9add3446')
    ap.add_argument('--ids', default=str(ROOT / 'diag/recover_check_ids.json'))
    ap.add_argument('--set', default='main', choices=['main', 'trial'], dest='qset')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--time-limit', type=float, default=0, help='no new group of questions starts after this many seconds')
    ap.add_argument('--check', action='store_true', help='numeric check (plan: "계산 점검") instead of the conditions')
    for k, v in DEFAULTS.items():
        ap.add_argument('--' + k.replace('_', '-'), type=type(v), default=v)
    a = ap.parse_args()
    cfg = {k: getattr(a, k) for k in DEFAULTS}
    t0 = time.time()
    from PIL import Image
    from run import fast_kv_selfcheck
    from visref.model import encode_sample, load_mathvista, load_model, special_ids
    doc = json.loads(Path(a.ids).read_text())
    order = doc[a.qset][:a.limit or None]
    si, sn = map(int, a.shard.split('/'))
    order = [p for i, p in enumerate(order) if i % sn == si]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)['pid'] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    todo = [p for p in order if p not in done]
    version = code_version()
    print(json.dumps({'event': 'start', 'code_version': version, 'set': a.qset, 'check': a.check, 'todo': len(todo),
                      'done': len(done), 'cfg': cfg}), flush=True)
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
    fixed_ids = tok.encode(THINK_OPEN + SENTENCE, add_special_tokens=False)
    tail_ids = [sp['think_end']] + sp['probe_prompt']
    print(json.dumps({'event': 'fixed', 'fixed_ids': fixed_ids, 'fixed_text': tok.decode(fixed_ids), 'tail_ids': tail_ids}),
          flush=True)
    fixed, tail = eng.embed(fixed_ids), eng.embed(tail_ids)

    def image_tokens(pid):
        r = rows[pid]
        return encode_sample(model, tok, r['image'](), r['query'], cfg['max_num'])[1]

    with out.open('a') as f:
        for gi in range(0, len(todo), cfg['group']):
            if a.time_limit and time.time() - t0 > a.time_limit:
                print(json.dumps({'event': 'time_limit', 'left': len(todo) - gi}), flush=True)
                break
            tg = time.time()
            n_ok = 0
            for pid in todo[gi:gi + cfg['group']]:
                tq = time.time()
                try:
                    row = rows[pid]
                    image = row['image']()
                    base, vis, img_pos = encode_sample(model, tok, image, row['query'], cfg['max_num'], with_positions=True)
                    n = int(vis.shape[0])
                    if n != doc['n_visual'][pid]:
                        raise ValueError(f'n_visual {n} differs from the ids file {doc["n_visual"][pid]}')
                    blank = encode_sample(model, tok, Image.new('RGB', image.size, GRAY), row['query'], cfg['max_num'])[1]
                    d1p, d2p = doc['donor1'][pid], doc['donor2'][pid]
                    assert len({pid, d1p, d2p}) == 3
                    d1, d2 = image_tokens(d1p), image_tokens(d2p)
                    sub = doc['subset'][pid]
                    if sub != subset_of(pid, n, cfg['seed']):
                        raise ValueError('stored subset differs from the recomputed one')
                    ctxs = build_contexts(base, img_pos, vis, blank, d1, d2, sub, fixed, tail)
                    gt_ids = gold_ids(tok, row)
                    r = {'pid': pid, 'set': a.qset, 'code_version': version, 'scorer_version': SCORER_VERSION, 'n_visual': n,
                         'question_type': row['question_type'], 'answer_gt': row['answer'], 'gt_ids': gt_ids,
                         'donor1': d1p, 'donor2': d2p, 'subset': sub, 'fixed_ids': fixed_ids, 'prompt_len': int(base.shape[0])}
                    if a.check:
                        r.update(status='check', **numeric_check(eng, ctxs, gt_ids, sp))
                        print(json.dumps({'event': 'check', 'pid': pid, 'cmp1_pass': r['cmp1_pass'], 'cmp1': r['cmp1'],
                                          'cmp2': r['cmp2']}), flush=True)
                    else:
                        rec, kls = score_question(eng, ctxs, gt_ids, sp)
                        r.update(status='ok', cond=rec, kl_pairs=kls, question_s=round(time.time() - tq, 2))
                    n_ok += 1
                except Exception as e:  # one question's failure must not lose the rest of the shard
                    r = {'pid': pid, 'set': a.qset, 'status': 'error', 'error': f'{type(e).__name__}: {e}'[:500],
                         'code_version': version}
                    print(json.dumps({'event': 'error', 'pid': pid, 'error': r['error']}), flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                f.flush()
            print(json.dumps({'event': 'group', 'n': n_ok, 'group_s': round(time.time() - tg, 1),
                              'elapsed': round(time.time() - t0), 'left': max(0, len(todo) - gi - cfg['group']),
                              'peak_alloc_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else 0}),
                  flush=True)
    print(json.dumps({'event': 'end', 'elapsed': round(time.time() - t0)}), flush=True)


if __name__ == '__main__':
    main()
