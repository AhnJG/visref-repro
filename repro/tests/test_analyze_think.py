"""diag/analyze_think.py (think experiments, pre-registered analysis written before the results)."""
import importlib.util
import json
import pathlib
import random

import pytest

DIAG = pathlib.Path(__file__).resolve().parent.parent / 'diag'


def _load(name):
    spec = importlib.util.spec_from_file_location(name, DIAG / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


at, ah = _load('analyze_think'), _load('analyze_hs')

Q = {'question_type': 'free_form', 'answer_type': 'integer', 'answer': '7', 'precision': None, 'choices': None}


@pytest.mark.parametrize('lo, hi, n', [(1, 21, 600), (1, 20, 600), (0, 21, 600), (-5, 30, 600), (-5, 20, 600),
                                       (-30, -1, 600), (-1, 21, 600)])
def test_verdict_rule_is_the_hs_rule(lo, hi, n):
    v, h = at.verdict(lo, hi, n), ah.verdict(lo, hi, n)
    assert v['code'] == h['code'] and ('논문보다 작음' in v['text']) == ('논문보다 작음' in h['text'])
    assert v['text'].startswith(at.VERDICT[v['code']])


def test_power_matches_the_numbers_used_in_the_plan():
    hs = at.power(68, 66, 839)  # visref_hs - TSR, 134 of 839 discordant (EF 2026-10-13 review entry)
    assert (hs['se_pp'], hs['mde80_pp'], hs['power_at_paper']) == (1.38, 3.87, 0.718)
    main = at.power(45, 44, 1000)  # main-run VisRef - TSR, 89 of 1,000 discordant
    assert (main['mde80_pp'], main['power_at_paper']) == (2.64, 0.96)


def row(pid, method, ok, extra='close', text=' look at the figure'):
    r = {'pid': pid, 'method': method, 'final_answer': '\\boxed{7}' if ok else '\\boxed{3}', 'correct': int(ok),
         'k': 2, 'stop_reason': 'entropy', 'final_y_start': 0, 'final_af': False, 'final_forced_answer': False,
         'think_tokens': 900, 'status': 'ok', 'code_version': 'abc',
         'steps': [{'think_end': 'cap', 'think_tokens': 600}, {'think_end': 'close', 'think_tokens': 300}],
         'thinking': 'z1\n'}
    if 'visref' in method:
        for s in r['steps']:
            s.update(m=500, n_inject=500 if 'hs' in method else 100)
            if 'hs' in method:
                s['n_meaningful'] = 70
    if method in at.THINK:
        r.update(extra_think_cap=512, extra_think_end=extra, extra_think_tokens=0 if extra == 'skipped' else 40,
                 step1_info={'step1_tokens_match': True, 'step1_store_sha': 'f937478f3b26'})
        if extra != 'skipped':
            r['thinking'] += '<EXTRA>Wait, think more' + text
    return r


def make(n=40, ok=None, drop=()):
    """ok(method, i) -> correct; default: TSR 3 of 4, VisRef as TSR, the think methods all right."""
    ok = ok or (lambda m, i: True if m in at.THINK else (i % 2 == 0 if m == 'st' else i % 4 != 0))
    pids = [str(i) for i in range(1, n + 1)]
    order = pids[:]
    random.Random(0).shuffle(order)
    bmain = {(p, m): row(p, m, ok(m, i)) for i, p in enumerate(pids) for m in at.BMAIN}
    think = {(p, m): row(p, m, ok(m, i)) for i, p in enumerate(pids) for m in at.THINK if (p, m) not in drop}
    return bmain, think, order, {p: Q for p in pids}


def test_primary_comparison_and_contrast():
    # visref_think right on every question, tsr_think wrong on every 5th; main run VisRef = TSR
    bmain, think, order, data = make(n=40, ok=lambda m, i: (i % 5 != 0) if m == 'tsr_think' else
                                     (True if m in at.THINK else i % 4 != 0))
    out = at.analyze(bmain, think, order, data)
    r = out['rules_only']
    assert out['n_prefix'] == 40 and out['power_flag'] == '검출력 부족' and out['main_reading'].startswith('rules_only')
    assert r['primary']['diff_pp'] == 20.0 and (r['primary']['a_only'], r['primary']['b_only']) == (8, 0)
    assert r['verdict']['code'] == 1 and r['power']['discordant'] == 8
    assert r['secondary']['contrast_(visref_think-tsr_think)-(visref-tsr)']['diff_pp'] == 20.0
    assert r['secondary']['tsr_think-tsr']['diff_pp'] == 5.0  # TSR wrong on i % 4 == 0 (10), tsr_think on i % 5 == 0 (8)
    assert r['acc']['visref_think'] == 100.0


def test_complete_prefix_needs_all_three_methods():
    bmain, think, order, data = make()
    gap = order[0::2][5]
    bmain, think, order, data = make(drop=((gap, 'visref_hs_think'),))
    assert at.analyze(bmain, think, order, data)['n_prefix'] == 5 + 20


def test_main_reading_uses_llm_rows_and_requires_all_of_them():
    bmain, think, order, data = make()
    llm = {k: dict(r, correct=0, llm_used=True) if k[1] == 'visref_think' else dict(r) for k, r in think.items()}
    out = at.analyze(bmain, think, order, data, llm)
    assert out['main_reading'] == 'main' and out['verdict'] == out['main']['verdict']
    assert out['main']['acc']['visref_think'] == 0.0 and out['rules_only']['acc']['visref_think'] == 100.0
    assert out['llm_used']['visref_think'] == 40
    del llm[(order[0], 'tsr_think')]
    with pytest.raises(AssertionError, match='lacks 1 prefix rows'):
        at.analyze(bmain, think, order, data, llm)


def test_rules_only_rescores_with_the_current_scorer():
    bmain, think, order, data = make()
    k = (order[0], 'tsr_think')
    think[k] = dict(think[k], correct=0)  # recorded wrong, but the text has the right box
    out = at.analyze(bmain, think, order, data)
    assert out['rules_only']['acc']['tsr_think'] == 100.0


def test_mechanism_counts():
    bmain, think, order, data = make()
    for p in order[:3]:
        think[(p, 'tsr_think')] = row(p, 'tsr_think', True, extra='skipped')
    think[(order[3], 'visref_think')] = row(order[3], 'visref_think', True, text=' so 7 is right')
    m = at.analyze(bmain, think, order, data)['mechanism']
    assert m['extra_think_end']['tsr_think'] == {'close': 37, 'skipped': 3}
    assert m['image_word_rate_extra']['visref_think'] == round(39 / 40, 4)
    assert m['step1_tokens_match']['visref_hs_think'] == 40 and m['step1_store_sha']['tsr_think'] == ['f937478f3b26']
    assert m['inject_over_m_q25_50_75']['visref_hs_think'] == [1.0, 1.0, 1.0]
    assert m['inject_over_m_q25_50_75']['visref_think'] == [0.2, 0.2, 0.2]


def test_extra_text():
    assert at.extra_text({'thinking': 'abc\n<EXTRA>Wait, think more so x'}) == ' so x'
    assert at.extra_text({'thinking': 'abc'}) is None


def _session(tmp, name, order, rows):
    d = tmp / name
    d.mkdir()
    (d / 'ids.json').write_text(json.dumps(order))
    (d / 'results_0.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return d


def test_sessions_are_put_together_and_never_overlap(tmp_path):
    bmain, think, order, data = make(n=8)
    rows = list(think.values())
    d1 = _session(tmp_path, 's1', order, rows[:10])
    d2 = _session(tmp_path, 's2', order, rows[10:])
    got, o = at.load_runs([d1, d2])
    assert o == order and got == think
    d3 = _session(tmp_path, 's3', order, rows[:1])
    with pytest.raises(AssertionError, match='finished in two sessions'):
        at.load_runs([d1, d3])
    d4 = _session(tmp_path, 's4', order[::-1], [])
    with pytest.raises(AssertionError, match='question order differs'):
        at.load_runs([d1, d4])
