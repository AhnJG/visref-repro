"""Faster decode for long contexts: preallocated KV buffers + GQA decode attention without repeat_kv.

transformers 4.55 DynamicCache copies the whole KV cache with torch.cat on every decode step, and the SDPA
path with a padding mask materializes repeat_kv copies. Prefill keeps the stock SDPA path. Numerically the
decode attention matches SDPA up to rounding; run.py checks this on the GPU at startup and falls back if not.
"""
import torch
from transformers.cache_utils import Cache, DynamicLayer
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import AttentionInterface


class PreLayer(DynamicLayer):
    cap = 0

    def update(self, k, v, cache_kwargs=None):
        t = k.shape[-2]
        if self.keys is None:
            self.n = 0
            self.kb = k.new_empty(k.shape[0], k.shape[1], max(self.cap, t), k.shape[3])
            self.vb = torch.empty_like(self.kb)
        if self.n + t > self.kb.shape[-2]:  # grow if the preallocation was too small
            pad = self.kb.new_empty(*self.kb.shape[:2], self.n + t - self.kb.shape[-2], self.kb.shape[3])
            self.kb, self.vb = torch.cat([self.kb, pad], 2), torch.cat([self.vb, torch.empty_like(pad)], 2)
        self.kb[:, :, self.n:self.n + t] = k
        self.vb[:, :, self.n:self.n + t] = v
        self.n += t
        self.keys, self.values = self.kb[:, :, :self.n], self.vb[:, :, :self.n]
        return self.keys, self.values

    def batch_select_indices(self, idx):
        self.kb, self.vb = self.kb[idx], self.vb[idx]
        self.keys, self.values = self.kb[:, :, :self.n], self.vb[:, :, :self.n]


class PreCache(Cache):
    def __init__(self, cap):
        super().__init__(layer_classes=type('PreLayerN', (PreLayer,), {'cap': cap}))


def gqa_attention(module, q, k, v, mask, dropout=0.0, scaling=None, is_causal=None, **kw):
    if q.shape[2] != 1:  # prefill: stock SDPA
        return sdpa_attention_forward(module, q, k, v, mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kw)
    B, H, _, D = q.shape
    Hk = k.shape[1]
    qg = (q * scaling).reshape(B, Hk, H // Hk, D)
    s = torch.matmul(qg, k.transpose(-1, -2)).float()  # [B, Hk, G, L]: K read once
    if mask is not None:
        m = mask[:, :, -1:, :k.shape[-2]]
        s = s.masked_fill(~m, float('-inf')) if m.dtype == torch.bool else s + m
    o = torch.matmul(s.softmax(-1).to(v.dtype), v)  # [B, Hk, G, D]
    return o.reshape(B, 1, H, D), None


def set_enabled(on):
    AttentionInterface.register('sdpa', gqa_attention if on else sdpa_attention_forward)
