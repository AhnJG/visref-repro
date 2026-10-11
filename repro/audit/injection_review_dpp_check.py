"""CPU check: numpy port of repro/visref/dpp.py:select_tokens (line-by-line) vs Chen et al. dpp() and naive log-det greedy."""
import math, sys
import numpy as np
sys.path.insert(0, '/mnt/user-data/uploads/VisRef/references/sources/cited_code/chen_fast-map-dpp')
from dpp import dpp as chen_dpp

def select_tokens_np(V, Z, m, rel_eps=1e-9):
    # faithful port of dpp.py L17-38
    N = V.shape[0]
    m = max(0, min(int(m), N))
    if m == 0:
        return []
    A = V.astype(np.float64) @ Z.astype(np.float64).T
    d2 = (A * A).sum(1)
    floor = rel_eps * float(d2.max()) if float(d2.max()) > 0 else 0.0
    c = np.zeros((N, m))
    chosen = np.zeros(N, dtype=bool)
    sel = []
    for k in range(m):
        gains = np.where(chosen, -np.inf, d2)
        j = int(np.argmax(gains))
        if not float(gains[j]) > floor:
            break
        sel.append(j)
        chosen[j] = True
        Lj = A @ A[j]
        e = (Lj - c[:, :k] @ c[j, :k]) / math.sqrt(d2[j])
        c[:, k] = e
        d2 = d2 - e * e
    return sorted(sel)

def naive_greedy(V, Z, m):
    A = V.astype(np.float64) @ Z.astype(np.float64).T
    L = A @ A.T
    sel = []
    for _ in range(m):
        best, bj = -math.inf, None
        for j in range(V.shape[0]):
            if j in sel: continue
            idx = sel + [j]
            sign, val = np.linalg.slogdet(L[np.ix_(idx, idx)])
            val = val if sign > 0 else -math.inf
            if val > best: best, bj = val, j
        sel.append(bj)
    return sorted(sel)

rng = np.random.default_rng(0)
ok = True
# 1) full rank: matches Chen (same order of picks) and naive
for trial in range(5):
    V = rng.standard_normal((40, 16)); Z = rng.standard_normal((30, 16))
    A = V @ Z.T; L = A @ A.T
    m = 10
    a = select_tokens_np(V, Z, m); b = sorted(chen_dpp(L, m)); c = naive_greedy(V, Z, m)
    ok &= (a == b == c)
    print('full-rank trial', trial, a == b == c, len(a))
# 2) rank deficient: Z has 3 rows -> rank(L)<=3; Chen stops by eps, repro by rel floor
for trial in range(5):
    V = rng.standard_normal((40, 16)) * 10 ** rng.uniform(-3, 3)  # random scale
    Z = rng.standard_normal((3, 16)) * 10 ** rng.uniform(-3, 3)
    A = V @ Z.T; L = A @ A.T
    a = select_tokens_np(V, Z, 10); b = sorted(chen_dpp(L, 10)); c = naive_greedy(V, Z, 3)
    print('rank-def trial', trial, 'scale Lmax=%.2e' % L.diagonal().max(), 'repro', len(a), 'chen', len(b), 'repro==naive3', a == c, 'repro==chen', a == b)
# 3) repeated tokens in Z: rank = #unique rows
Zu = rng.standard_normal((5, 16)); Z = Zu[rng.integers(0, 5, 50)]
V = rng.standard_normal((60, 16))
a = select_tokens_np(V, Z, 18)
print('repeated tokens: n_unique=5, n_sel=', len(a))
# 4) duplicate visual rows never both chosen
V = rng.standard_normal((20, 16)); V[7] = V[3]
a = select_tokens_np(V, rng.standard_normal((30, 16)), 20)
print('dup V rows: 3 in', 3 in a, '7 in', 7 in a, 'n=', len(a))
# 5) budget() floor check for all tile counts
for t in range(1, 14):
    n = 256 * t
    print('N=%d  int(N*0.3)=%d  floor=%d' % (n, int(n * 0.3), math.floor(n * 3 / 10)), end=' | ')
print()
