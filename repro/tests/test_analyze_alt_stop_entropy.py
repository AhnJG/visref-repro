"""diag/analyze_alt_stop_entropy.py: estimator functions on hand-made step dicts (counts only, no accuracy)."""
import math
import sys
from pathlib import Path

import pytest

REPRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO / 'diag'))
import analyze_alt_stop_entropy as A  # noqa: E402


def step(ent, full=None, ids=None, k=1, end='close', valid=True, H=None):
    full = ent if full is None else full
    ids = ids if ids is not None else [A.BRACE_ID] + [5] * (len(ent) - 1)
    return {'k': k, 'think_end': end, 'probe_ent': ent, 'probe_ent_full': full, 'probe_ids': ids,
            'probe_valid': valid, 'H': (sum(ent) if valid else math.inf) if H is None else H}


# '{' (0.01), content '7' (0.30), merged close (0.02); the unmerged close token has full entropy 0.50
S = step([0.01, 0.30, 0.02], full=[0.01, 0.30, 0.50])


def test_estimators_on_hand_made_step():
    assert A.estimate(S, 'E0') == pytest.approx(0.33)
    assert A.estimate(S, 'E1') == pytest.approx(0.11)
    assert A.estimate(S, 'E2') == pytest.approx(0.30)  # index 1: the token after the bare '{'
    assert A.estimate(S, 'E3') == pytest.approx(0.30)
    assert A.estimate(S, 'E4') == pytest.approx(0.81)
    assert A.estimate(S, 'E5') == pytest.approx(0.27)


def test_e2_fused_brace_token_uses_index_zero():
    fused = step([0.40, 0.01], ids=[19999, 5])  # a '{-' style token: the first content choice is token 0
    assert A.first_content_index(fused) == 0
    assert A.estimate(fused, 'E2') == pytest.approx(0.40)


def test_invalid_probe_is_inf_for_every_estimator():
    bad = step([0.1, 0.1, 0.1], valid=False)
    assert all(math.isinf(A.estimate(bad, e)) for e in A.ESTIMATORS)
    assert A.recompute_H(bad) == math.inf


def test_recompute_H_matches_code_rule():
    assert A.recompute_H(S) == pytest.approx(0.33)


def test_percentile_linear():
    assert A.percentile([1, 2, 3, 4], 50) == 2.5
    assert A.percentile([0, 10], 90) == pytest.approx(9.0)
    assert A.percentile([], 50) is None


def test_summarize_counts_threshold_invalid_and_near():
    steps = [step([0.0, 0.25, 0.0]),            # E0 == 0.25 exactly: counts as >= delta and as near
             step([0.0, 0.2495, 0.0]),          # within +-0.001, below delta
             step([0.0, 0.0, 0.0]),
             step([0.0, 0.0, 0.0], valid=False)]  # invalid: continue, kept apart
    s = A.summarize(steps, 'E0')
    assert (s['n_total'], s['n_invalid'], s['n_valid']) == (4, 1, 3)
    assert s['n_ge_valid'] == 1 and s['n_ge_incl_invalid'] == 2
    assert s['n_near_025'] == 2
    assert s['median'] == pytest.approx(0.2495)
    assert s['n_in_015_035'] == 2


def test_groups_and_flip_count():
    def row(m, steps):
        return {'method': m, 'status': 'ok', 'steps': steps}
    rows = [row('tsr', [step([0.0, 0.0, 0.0], end='close')]),                 # stops under E0 and E4
            row('tsr', [step([0.0, 0.0, 0.0], full=[0.0, 0.0, 0.9], end='eos')]),  # E4 = 0.9 -> flip
            row('tsr', [step([0.0, 0.0, 0.0], end='cap'), step([0.0, 0.5, 0.0], k=2, end='close')]),
            row('st', [step([0.0, 0.0, 0.0])]),
            {'method': 'tsr', 'status': 'error', 'steps': [step([0.0, 0.0, 0.0])]}]
    g = A.groups(rows)
    assert len(g['tsr']['a_first_close_eos']) == 2
    assert len(g['tsr']['b_first_cap']) == 1
    assert len(g['tsr']['c_k2plus_close']) == 1
    assert len(g['visref_raw_rank']['a_first_close_eos']) == 0
    pq = A.per_question_a(rows)['tsr']
    assert pq['n_questions'] == 2
    assert pq['E4']['flip_stop_to_continue'] == 1
    assert pq['E1']['flip_stop_to_continue'] == 0


def test_sanity_e0_detects_a_mismatch():
    ok = step([0.05, 0.05, 0.05])
    off = step([0.05, 0.05, 0.05], H=0.5)  # stored H far from the sum
    rows = [{'method': 'tsr', 'status': 'ok', 'steps': [ok, off]}]
    r = A.sanity_e0(rows)
    assert r['n_steps'] == 2 and r['n_mismatch_beyond_rounding'] == 1 and r['n_decision_differs'] == 1
