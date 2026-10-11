"""Diagnostic (CPU, no GPU): how many visual tokens can the VisRef greedy DPP pick *meaningfully* when the
text side Z of the kernel L = (V Z^T)(V Z^T)^T is a hidden state instead of the input embedding?

For each sample (pid, step-1 thinking of the b_main run) we rebuild the exact prompt (R1 system prompt, official
tiling), append the step-1 tokens, run one forward pass of the language model and take Z from several layers.
Greedy MAP (Chen et al. 2018 recursion, same as visref/dpp.py, float64) is run past the usual stop and every pick
records its normalized residual d_j^2 / r_j^2 (the diversity factor of Eq. 10's marginal gain; 1 = orthogonal to
everything picked so far, 0 = inside their span).

Outputs results/hidden_rank_check.json and prints one line per (pid, Z, V) setting.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repro/ root: visref package

THRESH = (1e-1, 1e-2, 1e-3, 1e-6)
PRIMARY = 1e-2  # pre-registered: a pick is "meaningful" while its normalized residual >= 1e-2
REL_EPS = 1e-9  # stop rule of visref/dpp.py (relative to max_i L_ii)


def greedy_trace(V, Z, cap):
    """Greedy MAP on L = A A^T, A = V Z^T, without early stop (up to cap picks).
    Returns picks, normalized residual of each pick, and the pick count at which visref/dpp.py would stop."""
    A = V @ Z.T
    d2 = np.einsum('ij,ij->i', A, A)
    r2 = d2.copy()
    floor = REL_EPS * float(d2.max())
    N = A.shape[0]
    cap = min(cap, N)
    c = np.zeros((N, cap))
    chosen = np.zeros(N, bool)
    picks, nres, stop_at = [], [], None
    for k in range(cap):
        g = np.where(chosen, -np.inf, d2)
        j = int(np.argmax(g))
        if stop_at is None and not g[j] > floor:
            stop_at = k
        if not g[j] > 0:  # numerically exhausted
            break
        picks.append(j)
        nres.append(float(g[j] / r2[j]))
        chosen[j] = True
        e = (A @ A[j] - c[:, :k] @ c[j, :k]) / math.sqrt(g[j])
        c[:, k] = e
        d2 = d2 - e * e
    if stop_at is None:
        stop_at = len(picks)
    return picks, nres, stop_at


def spectrum_stats(Z):
    s = np.linalg.svd(Z, compute_uv=False)
    s = s / s[0]
    pr = float((s ** 2).sum() ** 2 / (s ** 4).sum())  # participation ratio of the energy spectrum
    return {'sv_gt_1e-2': int((s > 1e-2).sum()), 'sv_gt_1e-4': int((s > 1e-4).sum()), 'participation_ratio': pr}


def summarize_trace(nres, stop_at, n_vis, m):
    nres = np.asarray(nres)

    def count(t):  # picks made before the normalized residual first drops below t
        below = np.nonzero(nres < t)[0]
        return int(below[0]) if len(below) else len(nres)

    out = {f'n_nres_ge_{t:g}': count(t) for t in THRESH}
    out.update(n_dpp_stop_rule=int(stop_at), m=m, n_visual=n_vis,
               frac_primary=count(PRIMARY) / n_vis, primary_ge_m=bool(count(PRIMARY) >= m),
               nres_at_m=float(nres[m - 1]) if len(nres) >= m else None)
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True)
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--model', default='OpenGVLab/InternVL3_5-1B')
    ap.add_argument('--revision', default='2f71cf52542334823e48a46ffba0e2bc9add3446')
    ap.add_argument('--out', required=True)
    ap.add_argument('--vis-frac', type=float, default=0.3)
    a = ap.parse_args()
    from visref.model import build_prompt, encode_sample, load_mathvista, load_model
    model, tok = load_model(a.model, a.revision, dtype=torch.float32, device='cpu')
    lm = model.language_model
    base = lm.model  # Qwen3Model: decoder layers + final norm, no vocab projection
    n_layers = lm.config.num_hidden_layers
    chosen = sorted({n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers})  # 1-based layer outputs
    caps = {}

    def grab(name):
        def hook(mod, inp, out):
            caps[name] = (out[0] if isinstance(out, (tuple, list)) else out).detach()
        return hook

    for i in chosen:
        base.layers[i - 1].register_forward_hook(grab(f'layer{i}'))
    base.norm.register_forward_hook(grab('final_norm'))
    zones = ['input_embedding'] + [f'layer{i}' for i in chosen] + ['final_norm']
    data = load_mathvista(a.parquet)
    samples = json.load(open(a.samples))
    results = []
    for s in samples:
        t0 = time.time()
        row = data[s['pid']]
        emb, vis = encode_sample(model, tok, row['image'](), row['query'])
        text = s['st_thinking']
        ids = tok.encode(text, add_special_tokens=False)
        n_tok_raw = len(ids)
        if len(ids) > s['think_tokens']:  # a cut step carries the '\n' separator in the trace text
            ids = ids[:s['think_tokens']]
        z_ids = torch.tensor(ids)
        z_emb = lm.get_input_embeddings()(z_ids)
        x = torch.cat([emb, z_emb], 0)[None]
        caps.clear()
        base(inputs_embeds=x, use_cache=False)
        caps['input_embedding'] = x
        P = emb.shape[0]
        # image token positions inside the prompt (same rule as encode_sample)
        pids_t = tok(build_prompt(row['query'], vis.shape[0]), return_tensors='pt').input_ids[0]
        img_pos = torch.nonzero(pids_t == tok.convert_tokens_to_ids('<IMG_CONTEXT>')).flatten()
        n_vis = vis.shape[0]
        m = int(math.floor(a.vis_frac * n_vis))
        V_in = vis.double().numpy()
        rec = {'pid': s['pid'], 'n_visual': n_vis, 'n_visual_recorded': s['n_visual'], 'T': len(ids),
               'T_recorded': s['think_tokens'], 'n_tok_raw': n_tok_raw, 'think_end': s['think_end'],
               'distinct_z': len(set(ids)), 'n_unique_z_recorded': s['n_unique_z'],
               'n_dpp_recorded': s['n_dpp'], 'm': m, 'settings': []}
        cap = min(n_vis, max(n_vis // 2, len(ids) + 64))  # enough to see the fall-off past m = 0.3|V|
        for zone in zones:
            H = caps[zone][0]
            Z = H[P:].double().numpy()
            spec = spectrum_stats(Z)
            variants = [('V_input', V_in)]
            if zone != 'input_embedding':
                variants.append(('V_same_layer', H[img_pos].double().numpy()))
            for vname, V in variants:
                picks, nres, stop_at = greedy_trace(V, Z, cap)
                summ = summarize_trace(nres, stop_at, n_vis, m)
                st = {'Z': zone, 'V': vname, **spec, **summ,
                      'nres_curve': [round(float(v), 6) for v in nres[:: max(1, len(nres) // 64)]]}
                rec['settings'].append(st)
                print(json.dumps({'pid': s['pid'], **{k: st[k] for k in ('Z', 'V', 'sv_gt_1e-2', 'n_nres_ge_0.01',
                                                                          'n_dpp_stop_rule', 'm', 'frac_primary')}}),
                      flush=True)
        rec['seconds'] = round(time.time() - t0, 1)
        results.append(rec)
        json.dump({'zones': zones, 'n_layers': n_layers, 'primary_threshold': PRIMARY, 'results': results},
                  open(a.out, 'w'), indent=1)
        print(f"pid {s['pid']} done in {rec['seconds']}s (T={len(ids)}, |V|={n_vis}, m={m})", flush=True)


if __name__ == '__main__':
    main()
