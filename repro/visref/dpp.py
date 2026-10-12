"""VisRef visual-token selection (paper Sec. 4.1.1, Eq. 6-11).

Kernel: L_k(v_i, v_j) = v_i^T M_k v_j with M_k = sum_t z_t z_t^T, i.e. L = A A^T where A = V Z^T (N x T_k).
Greedy MAP as in Chen et al. 2018 (paper ref [7]; references/sources/cited_code/chen_fast-map-dpp/dpp.py):
pick argmax of the residual d_j^2 = log-det gain until m tokens are chosen or the gain vanishes. L has rank
<= min(T_k, d) (<= #distinct tokens of z_k with input embeddings), so fewer than m tokens may be returned --
the cited code stops there too (ε test); here the threshold is relative, 1e-9 * max_i L_ii.
"""
import torch


def select_tokens(V, Z, m, rel_eps=1e-9, trace=None):
    """Sorted indices of the visual tokens chosen by greedy MAP (at most m).

    V: [N, d] visual token embeddings, Z: [T, d] text token embeddings of step z_k.
    trace: optional list; receives (index, residual / relevance, relevance) per pick in pick order (diagnostic only,
    the selection is the same with or without it).
    """
    N = V.shape[0]
    m = max(0, min(int(m), N))
    if m == 0:
        return []
    A = V.double() @ Z.double().T  # [N, T]; phi_k(v) = Z v (same Gram as M_k^{1/2} v)
    d2 = (A * A).sum(1)  # relevance r_i^2 = L_ii
    r2 = d2.clone() if trace is not None else None
    floor = rel_eps * float(d2.max()) if float(d2.max()) > 0 else 0.0
    c = torch.zeros(N, m, dtype=A.dtype, device=A.device)
    chosen = torch.zeros(N, dtype=torch.bool, device=A.device)
    sel = []
    for k in range(m):
        gains = d2.masked_fill(chosen, float('-inf'))
        j = int(torch.argmax(gains))
        if not float(gains[j]) > floor:
            break
        sel.append(j)
        if trace is not None:
            trace.append((j, float(d2[j] / r2[j]), float(r2[j])))
        chosen[j] = True
        Lj = A @ A[j]
        e = (Lj - c[:, :k] @ c[j, :k]) / torch.sqrt(d2[j])
        c[:, k] = e
        d2 = d2 - e * e
    return sorted(sel)


def budget(n_visual, frac):
    """m = floor(frac * |V|) (paper Sec. 5)."""
    return int(n_visual * frac)


def select_tokens_fill(V, Z, m, rel_eps=1e-9, ridge_rel=1e-6):
    """`visref_fill` (EF 2026-10-10/11 pre-registration): keep exactly the greedy MAP picks of select_tokens (the
    "head", which stops at rank(L)), then add eps*I to the kernel, eps = ridge_rel * mean_i L_ii, and continue the
    same greedy (Chen et al. incremental Cholesky, float64) on L + eps*I until exactly m tokens are chosen. The head
    is entered first as fixed picks (in index order; the residual variances after the whole head do not depend on
    the order), so the head is identical to visref_raw_rank by construction (asserted).

    Returns (sorted indices, info) with info = n_head, n_fill, ridge, tail_rel_spearman (Spearman correlation between
    the order in which tail tokens were picked and their relevance r_i^2 = L_ii; negative = more relevant first).
    """
    N = V.shape[0]
    m = max(0, min(int(m), N))
    head = select_tokens(V, Z, m, rel_eps)
    info = {'n_head': len(head), 'n_fill': 0, 'ridge': None, 'tail_rel_spearman': None}
    if len(head) >= m:
        return head, info
    A = V.double() @ Z.double().T
    r2 = (A * A).sum(1)
    eps = ridge_rel * float(r2.mean())
    if not eps > 0:
        eps = ridge_rel
    info['ridge'] = eps
    d2 = r2 + eps
    c = torch.zeros(N, m, dtype=A.dtype, device=A.device)
    chosen = torch.zeros(N, dtype=torch.bool, device=A.device)
    order = []
    for k in range(m):
        if k < len(head):
            j = head[k]
        else:
            j = int(torch.argmax(d2.masked_fill(chosen, float('-inf'))))
        if not float(d2[j]) > 0:  # numerically exhausted even with the ridge: stop (recorded via n_fill < m - n_head)
            break
        order.append(j)
        chosen[j] = True
        Lj = A @ A[j]
        Lj[j] += eps
        e = (Lj - c[:, :k] @ c[j, :k]) / torch.sqrt(d2[j])
        c[:, k] = e
        d2 = d2 - e * e
    assert order[:len(head)] == head and len(set(order)) == len(order)
    tail = order[len(head):]
    info['n_fill'] = len(tail)
    if len(tail) >= 3:
        rel = r2[tail]
        pos_rank = torch.arange(len(tail), dtype=torch.float64)
        rel_rank = torch.empty(len(tail), dtype=torch.float64)
        rel_rank[torch.argsort(rel)] = torch.arange(len(tail), dtype=torch.float64)
        pr, rr = pos_rank - pos_rank.mean(), rel_rank - rel_rank.mean()
        info['tail_rel_spearman'] = round(float((pr * rr).sum() / torch.sqrt((pr * pr).sum() * (rr * rr).sum())), 4)
    return sorted(order), info


def pick_stats(trace, thr=1e-2):
    """Diagnostics of a select_tokens trace (EF 2026-10-12, visref_hs): n_meaningful = picks made while the normalized
    residual (residual / relevance at pick time) was still >= thr (the measure of the 2026-10-08 hidden_rank_check);
    tail_rel_spearman = Spearman correlation between pick order and relevance over the later picks (negative = more
    relevant first), None if fewer than 3 later picks."""
    n_meaningful = 0
    for _, ratio, _ in trace:
        if ratio < thr:
            break
        n_meaningful += 1
    tail = [rel for _, _, rel in trace[n_meaningful:]]
    rho = None
    if len(tail) >= 3:
        rel = torch.tensor(tail, dtype=torch.float64)
        pos_rank = torch.arange(len(tail), dtype=torch.float64)
        rel_rank = torch.empty(len(tail), dtype=torch.float64)
        rel_rank[torch.argsort(rel)] = torch.arange(len(tail), dtype=torch.float64)
        pr, rr = pos_rank - pos_rank.mean(), rel_rank - rel_rank.mean()
        rho = round(float((pr * rr).sum() / torch.sqrt((pr * pr).sum() * (rr * rr).sum())), 4)
    return {'n_meaningful': n_meaningful, 'tail_rel_spearman': rho}
