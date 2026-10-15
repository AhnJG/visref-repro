"""Batched sampling over embedding contexts with a HF causal LM (KV cache per call).

Each request is a context of input embeddings (text ids already embedded, visual
tokens inserted as raw embeddings), so reinjected visual tokens and text share one
code path. A call decodes at most `chunk` tokens per row; unfinished rows are
re-submitted by the caller with the generated ids appended. This keeps the batch
full (simple continuous batching) at the cost of one re-prefill per chunk.

Per-token entropy is computed from the raw (T=1) full-vocabulary distribution;
sampling uses temperature / top-k / top-p with one torch.Generator per job so a
job's draws do not depend on batch composition.
"""
from dataclasses import dataclass, field

import math
import time

import torch
from transformers import DynamicCache


@dataclass
class Request:
    ctx: torch.Tensor  # [L, d] input embeddings
    stop_ids: frozenset
    max_new: int
    gen: torch.Generator
    record_entropy: bool = False
    sampling: tuple = None          # (temperature, top_k, top_p) override; None = engine defaults
    stop_check: object = None       # callable(ids) -> bool, evaluated after every sampled token
    ent_lump: frozenset = frozenset()  # also record entropy with these ids merged into one outcome


@dataclass
class Result:
    ids: list = field(default_factory=list)
    ent: list = field(default_factory=list)
    ent_lump: list = field(default_factory=list)
    finish: str = ''  # 'stop' | 'cap' | 'chunk'


def lumped_entropy(logp, idx, H):
    """Entropy of the distribution with the ids `idx` merged into a single outcome:
    H - sum_{i in S} -p_i log p_i + (-P_S log P_S)."""
    lp = logp[idx]
    p = lp.exp()
    PS = float(p.sum())
    return H + float((p * lp).sum()) - (PS * math.log(PS) if PS > 0 else 0.0)


def sample_row(logits, gen, temperature, top_k, top_p):
    """Temperature -> top-k -> top-p sampling. With top-k only the k candidates are sorted (no full-vocab sort)."""
    if temperature <= 0:
        return int(torch.argmax(logits))
    x = logits / temperature
    if top_k and top_k < x.shape[-1]:
        vals, idx = torch.topk(x, top_k)  # sorted descending
    else:
        vals, idx = torch.sort(x, descending=True)
    if top_p < 1.0:
        p = torch.softmax(vals, -1)
        drop = p.cumsum(-1) - p > top_p  # keep the token that crosses top_p
        vals = vals.masked_fill(drop, float('-inf'))
    j = int(torch.multinomial(torch.softmax(vals, -1), 1, generator=gen))
    return int(idx[j])


class Engine:
    def __init__(self, lm, temperature=0.6, top_k=50, top_p=0.95, attn_budget=2 ** 28, fast_kv=False):
        self.lm = lm
        self.fast_kv = fast_kv  # preallocated KV + GQA decode attention (visref/fastkv.py)
        self.attn_budget = attn_budget  # max B*heads*q*L elements per prefill chunk (bounds attention memory)
        self.emb = lm.get_input_embeddings()
        self.t, self.k, self.p = temperature, top_k, top_p

    @torch.no_grad()
    def embed(self, ids):
        dev = self.emb.weight.device
        return self.emb(torch.as_tensor(ids, dtype=torch.long, device=dev))

    @torch.no_grad()
    def last_hidden(self, ctx, n):
        """Last-layer hidden states (after the final norm, i.e. the decoder's last_hidden_state that feeds lm_head) of
        the last n positions of one context ctx [L, d]. Chunked prefill with a KV cache bounds attention memory as in
        generate(); positions 0..L-1 as for an unpadded row. Used by 'visref_hs' for Z (EF 2026-10-12)."""
        L = ctx.shape[0]
        n = min(int(n), L)
        if n <= 0:
            return ctx[:0]
        dec = self.lm.get_decoder()
        cache = DynamicCache()
        pos = torch.arange(L, device=ctx.device)[None]
        q = max(16, self.attn_budget // (self.lm.config.num_attention_heads * L))
        outs = []
        for s in range(0, L, q):
            e = min(L, s + q)
            o = dec(inputs_embeds=ctx[None, s:e], position_ids=pos[:, s:e], past_key_values=cache, use_cache=True)
            if e > L - n:
                outs.append(o.last_hidden_state[0, max(s, L - n) - s:])
        return torch.cat(outs, 0)

    @torch.no_grad()
    def hidden_at(self, ctx, idx):
        """Last-layer hidden states (after the final norm, as last_hidden) of one context ctx [L, d] at the positions
        idx (LongTensor or list of ints), returned [len(idx), d] in the order of idx. Same chunked prefill as
        last_hidden. Used by 'visref_imgtag' for V: the prompt's hidden states at the image-token positions."""
        idx = torch.as_tensor(list(idx) if not torch.is_tensor(idx) else idx, dtype=torch.long, device=ctx.device)
        L = ctx.shape[0]
        if idx.numel() == 0:
            return ctx[:0]
        assert int(idx.min()) >= 0 and int(idx.max()) < L
        dec = self.lm.get_decoder()
        cache = DynamicCache()
        pos = torch.arange(L, device=ctx.device)[None]
        q = max(16, self.attn_budget // (self.lm.config.num_attention_heads * L))
        out = None
        for s in range(0, L, q):
            e = min(L, s + q)
            o = dec(inputs_embeds=ctx[None, s:e], position_ids=pos[:, s:e], past_key_values=cache, use_cache=True)
            hit = (idx >= s) & (idx < e)
            if hit.any():
                h = o.last_hidden_state[0]
                if out is None:
                    out = h.new_empty(idx.numel(), h.shape[-1])
                out[hit] = h[idx[hit] - s]
        return out

    @torch.no_grad()
    def generate(self, reqs, chunk, stop_at=None):
        """stop_at: wall-clock time after which all unfinished rows return early with finish='chunk'."""
        B = len(reqs)
        lens = [r.ctx.shape[0] for r in reqs]
        L = max(lens)
        first = reqs[0].ctx
        emb = torch.zeros(B, L, first.shape[1], dtype=first.dtype, device=first.device)
        mask = torch.zeros(B, L, dtype=torch.long, device=first.device)
        for i, r in enumerate(reqs):
            emb[i, L - lens[i]:] = r.ctx
            mask[i, L - lens[i]:] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        if self.fast_kv:
            from .fastkv import PreCache
            cache = PreCache(L + min(chunk, max(r.max_new for r in reqs)))
        else:
            cache = DynamicCache()
        q = max(16, self.attn_budget // (B * self.lm.config.num_attention_heads * L))
        for s in range(0, L, q):  # chunked prefill: SDPA with a padding mask may materialize B*H*q*L scores
            e = min(L, s + q)
            out = self.lm(inputs_embeds=emb[:, s:e], attention_mask=mask[:, :e], position_ids=pos[:, s:e],
                          past_key_values=cache, use_cache=True, logits_to_keep=1)
        del emb
        logits = out.logits[:, -1].float()
        nxt = pos[:, -1] + 1
        res = [Result() for _ in reqs]
        alive = list(range(B))
        lump = {}  # frozenset -> index tensor (built once per call)
        for r in reqs:
            if r.record_entropy and r.ent_lump and r.ent_lump not in lump:
                lump[r.ent_lump] = torch.tensor(sorted(r.ent_lump), dtype=torch.long, device=logits.device)
        while True:
            if not torch.isfinite(logits).all():
                raise FloatingPointError('non-finite logits (fp16 overflow?)')
            logp = torch.log_softmax(logits, -1)
            H = (-(logp.exp() * logp).nan_to_num(0.0)).sum(-1).tolist()
            toks, keep = [], []
            for j, i in enumerate(alive):
                r, o = reqs[i], res[i]
                t = sample_row(logits[j], r.gen, *(r.sampling or (self.t, self.k, self.p)))
                o.ids.append(t)
                if r.record_entropy:
                    o.ent.append(H[j])
                    if r.ent_lump:
                        o.ent_lump.append(lumped_entropy(logp[j], lump[r.ent_lump], H[j]))
                if t in r.stop_ids or (r.stop_check is not None and r.stop_check(o.ids)):
                    o.finish = 'stop'
                elif len(o.ids) >= r.max_new:
                    o.finish = 'cap'
                elif len(o.ids) >= chunk:
                    o.finish = 'chunk'
                else:
                    keep.append(j)
                toks.append(t)
            if not keep:
                return res
            if stop_at is not None and time.time() > stop_at:
                for j in keep:
                    res[alive[j]].finish = 'chunk'
                return res
            if len(keep) < len(alive):
                idx = torch.tensor(keep, device=mask.device)
                cache.batch_select_indices(idx)
                mask, nxt = mask[idx], nxt[idx]
                toks = [toks[j] for j in keep]
                alive = [alive[j] for j in keep]
            mask = torch.cat([mask, mask.new_ones(len(alive), 1)], 1)
            out = self.lm(inputs_embeds=self.embed(toks)[:, None], attention_mask=mask,
                          position_ids=nxt[:, None], past_key_values=cache, use_cache=True)
            logits = out.logits[:, -1].float()
            nxt = nxt + 1
