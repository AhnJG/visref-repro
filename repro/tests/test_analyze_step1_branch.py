"""step1_branch analysis arithmetic (2026-10-12 fix): integer differences make the bootstrap p independent of
float summation order, and an exact zero resample mean counts in both tails."""
import importlib.util
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location(
    'analyze_step1_branch', pathlib.Path(__file__).resolve().parent.parent / 'diag' / 'analyze_step1_branch.py')
asb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(asb)


def test_all_zero_differences_give_p_one():
    assert asb.p_boot([0] * 50) == 1.0


def test_zero_resample_counts_in_both_tails():
    d = [1, -1] * 20  # many resamples sum to exactly 0
    sums = asb.boot_sums(d)
    zeros = sum(s == 0 for s in sums)
    assert zeros > 0
    lo, hi = sum(s <= 0 for s in sums), sum(s >= 0 for s in sums)
    assert lo + hi == len(sums) + zeros
    assert asb.p_boot(d) == round(min(1.0, 2 * min(lo, hi) / len(sums)), 4)


def test_ci_scale_matches_mean_of_thirds():
    d = [3, 0, -3, 3]  # counts out of 3 -> differences of +1, 0, -1, +1 in accuracy
    lo, hi = asb.boot(d, 3)
    assert -100 <= lo <= hi <= 100


def test_n_correct_excludes_greedy_and_checks_acc():
    r = {'pid': 'x', 'acc_t1': {'A': 2 / 3},
         'answers': {'A': [{'correct': True}, {'correct': False}, {'correct': True}, {'correct': False, 'greedy': True}]}}
    assert asb.n_correct(r, 'A') == 2
    r['acc_t1']['A'] = 1.0
    with pytest.raises(AssertionError):
        asb.n_correct(r, 'A')
