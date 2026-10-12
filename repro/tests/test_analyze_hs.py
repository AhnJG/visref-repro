"""diag/analyze_hs.py (visref_hs pre-registered analysis, written before the results): verdict rule on integer
bounds, the same bootstrap resamples as summarize.paired, complete prefix, and which rows each reading uses."""
import importlib.util
import pathlib
import random

import pytest

import summarize as S

_spec = importlib.util.spec_from_file_location(
    'analyze_hs', pathlib.Path(__file__).resolve().parent.parent / 'diag' / 'analyze_hs.py')
ah = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ah)


@pytest.mark.parametrize('lo, hi, n, code, note', [
    (1, 21, 600, 1, False),    # upper bound exactly +3.5 pp: not "smaller than the paper"
    (1, 20, 600, 1, True),     # lower bound > 0, upper < +3.5
    (0, 21, 600, 2, False),    # CI touches 0, upper exactly +3.5 -> undecided
    (-5, 30, 600, 2, False),
    (-5, 20, 600, 3, False),   # upper < +3.5
    (-30, -1, 600, 3, False),  # CI entirely below 0
])
def test_verdict_rule(lo, hi, n, code, note):
    v = ah.verdict(lo, hi, n)
    assert v['code'] == code
    assert ('논문보다 작음' in v['text']) == note


def test_bound_of_exactly_3_5_is_not_lost_to_float_rounding():
    assert 100 * (21 / 600) != 3.5  # what a float comparison of the rounded-to-pp bound would see
    assert ah.verdict(-1, 21, 600)['code'] == 2


def test_bounds_are_the_summarize_paired_resamples():
    rng = random.Random(5)
    for n in (37, 300, 976):
        d = [rng.choice([-1, 0, 0, 0, 1]) for _ in range(n)]
        rows = {(str(i), 'a'): {'correct': int(x == 1)} for i, x in enumerate(d)}
        rows.update({(str(i), 'b'): {'correct': int(x == -1)} for i, x in enumerate(d)})
        lo, hi = ah.boot_bounds(d)
        ref = S.paired(rows, 'a', 'b', [str(i) for i in range(n)])
        assert abs(100 * lo / n - ref['ci95_pp'][0]) < 1e-9 and abs(100 * hi / n - ref['ci95_pp'][1]) < 1e-9


Q = {'question_type': 'free_form', 'answer_type': 'integer', 'answer': '7', 'precision': None, 'choices': None}


def row(pid, method, ok, end='cap', k=1):
    r = {'pid': pid, 'method': method, 'final_answer': '\\boxed{7}' if ok else '\\boxed{3}', 'correct': int(ok),
         'k': k, 'stop_reason': 'entropy', 'final_y_start': 0, 'final_af': False, 'think_tokens': 1024,
         'steps': [{'think_end': end, 'think_tokens': 1024}]}
    if method in ('visref_raw_rank', 'visref_hs'):
        r['steps'][0].update(m=600, n_inject=600 if method == 'visref_hs' else 120, n_unique_z=150,
                             **({'n_meaningful': 80, 'tail_rel_spearman': -0.1} if method == 'visref_hs' else {}))
    return r


def make(n=40, hs_ok=lambda i: True, tsr_ok=lambda i: i % 4 != 0, drop=()):
    pids = [str(i) for i in range(1, n + 1)]
    order = pids[:]
    random.Random(0).shuffle(order)
    bmain = {}
    for i, p in enumerate(pids):
        for m, ok in (('st', i % 2 == 0), ('tsr', tsr_ok(i)), ('visref_raw_rank', tsr_ok(i))):
            bmain[(p, m)] = row(p, m, ok)
    hs = {(p, 'visref_hs'): row(p, 'visref_hs', hs_ok(i)) for i, p in enumerate(pids) if p not in drop}
    data = {p: Q for p in pids}
    return bmain, hs, order, data


def test_complete_prefix_stops_at_the_first_unfinished_question_of_each_shard():
    bmain, hs, order, data = make()
    gap = order[0::2][5]  # 6th question of shard 0
    bmain, hs, order, data = make(drop=(gap,))
    out = ah.analyze(bmain, hs, order, 'visref_hs', data)
    assert out['n_prefix'] == 5 + 20
    assert out['power_flag'] == '검출력 부족'
    assert out['main_reading'].startswith('rules_only')


def test_main_reading_uses_the_merged_llm_rows_and_rules_only_rescores():
    bmain, hs, order, data = make()
    # rules-only: hs right everywhere, TSR wrong on 10/40 -> +25 pp
    out = ah.analyze(bmain, hs, order, 'visref_hs', data)
    assert out['rules_only']['primary']['diff_pp'] == 25.0 and out['verdict']['code'] == 1
    # merged LLM rows say hs is wrong on 20 rows -> main reading differs from the rules-only one
    llm = {k: dict(r, correct=int(int(k[0]) % 2 == 0), llm_used=True, scorer_rescored=ah.SCORER_VERSION)
           for k, r in hs.items()}
    out = ah.analyze(bmain, hs, order, 'visref_hs', data, llm=llm)
    assert out['main_reading'] == 'main'
    assert out['main']['acc']['visref_hs'] == 50.0 and out['rules_only']['acc']['visref_hs'] == 100.0
    assert out['verdict'] == out['main']['verdict']
    assert out['hs_llm_used'] == 40 and out['hs_llm_rows_scorer'] == [ah.SCORER_VERSION]
    mech = out['mechanism']
    assert mech['steps_len_ge_m_inject_eq_m'] == mech['steps_len_ge_m'] == 40
    assert mech['visref_inject_over_m_q25_50_75'] == [0.2, 0.2, 0.2]


def test_missing_llm_rows_are_an_error():
    bmain, hs, order, data = make()
    llm = {k: r for k, r in list(hs.items())[1:]}
    with pytest.raises(AssertionError, match='lacks'):
        ah.analyze(bmain, hs, order, 'visref_hs', data, llm=llm)


def test_rescoring_overrides_a_recorded_correct_flag():
    bmain, hs, order, data = make()
    k = next(iter(hs))
    hs[k] = dict(hs[k], correct=0)  # recorded wrong, but the final answer is right
    out = ah.analyze(bmain, hs, order, 'visref_hs', data)
    assert out['rules_only']['acc']['visref_hs'] == 100.0
