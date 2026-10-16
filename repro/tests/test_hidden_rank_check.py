"""CPU tests for the numeric parts of diag/hidden_rank_check.py (no model needed)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'diag'))
from hidden_rank_check import greedy_trace, spectrum_stats, summarize_trace  # noqa: E402
from visref.dpp import select_tokens  # noqa: E402


def test_matches_production_greedy_and_rank_stop():
    rng = np.random.default_rng(0)
    N, d, U, T = 300, 64, 25, 120
    V = rng.standard_normal((N, d))
    Z = rng.standard_normal((U, d))[rng.integers(0, U, T)]  # input-embedding-like: duplicated rows
    m = 90
    picks, nres, stop_at = greedy_trace(V, Z, cap=m)
    prod = select_tokens(torch.tensor(V), torch.tensor(Z), m)
    assert stop_at == len(prod) == len(set(map(tuple, Z))) == 25
    assert sorted(picks[:stop_at]) == prod
    assert nres[0] == 1.0 and nres[stop_at - 1] > 1e-6 and max(nres[stop_at:]) < 1e-9


def test_full_rank_text_side_keeps_picking():
    rng = np.random.default_rng(1)
    V = rng.standard_normal((300, 64))
    Z = rng.standard_normal((120, 64))  # hidden-state-like: all rows distinct, rank = d = 64
    picks, nres, stop_at = greedy_trace(V, Z, cap=100)
    assert stop_at == 64
    s = summarize_trace(nres, stop_at, n_vis=300, m=90)
    assert s['n_dpp_stop_rule'] == 64 and s['n_nres_ge_0.01'] <= 64 and s['primary_ge_m'] is False
    sp = spectrum_stats(Z)
    assert sp['sv_gt_1e-4'] == 64
