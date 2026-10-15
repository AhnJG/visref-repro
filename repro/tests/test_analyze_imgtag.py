"""diag/analyze_imgtag.py: mechanism readout over synthetic result rows."""
import importlib.util
import json
import pathlib

_spec = importlib.util.spec_from_file_location(
    'analyze_imgtag', pathlib.Path(__file__).resolve().parent.parent / 'diag' / 'analyze_imgtag.py')
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)


def step(k=1, end='close', tt=50, inner='3', nov='3', crn=True, n_inject=4, m=8, fmt=None, **kw):
    s = {'k': k, 'think_end': end, 'think_tokens': tt, 'H': 0.1, 'probe_answer': '\\boxed{%s}' % inner,
         'probe_inner': inner, 'probe_valid': True, 'm': m, 'n_inject': n_inject}
    if nov is not None:
        s.update(H_noV=0.2, probe_noV_answer='\\boxed{%s}' % nov, probe_noV_inner=nov, probe_noV_finish='end')
        if crn:
            s['probe_noV_crn'] = True
    if fmt:
        s['inject_format'] = fmt
    s.update(kw)
    return s


def row(pid, method, steps, ans='7', **kw):
    r = {'pid': pid, 'method': method, 'steps': steps, 'final_answer': ans, 'final_raw': None, 'final_y_start': 0,
         'final_len': 20, 'final_finish': 'stop', 'final_forced_answer': False, 'k': len(steps),
         'stop_reason': 'entropy', 'status': 'ok', 'step1_info': None}
    r.update(kw)
    return r


ROWS = [
    row('1', 'visref_raw_rank', [step(inner='3', nov='4')]),                                  # close, probes differ
    row('2', 'visref_raw_rank', [step(end='cap', inner='5', nov=' 5 ', crn=False, n_inject=1, m=8)],  # cap, no CRN
        final_forced_answer=True, final_finish='cap', final_len=900),
    row('3', 'visref_imgtag', [step(fmt='img_tag_outside_think'), step(k=2, tt=1, n_inject=0, fmt='img_tag_outside_think')],
        final_raw='x </think> y', final_y_start=5, final_len=40),                              # re-emitted </think>
    row('4', 'visref_imgtag', [step(end='cap', nov=None, fmt='img_tag_outside_think')], ans='  ', status='error',
        stop_reason='kmax'),                                                                    # empty answer
    row('5', 'visref_imgtag', [step(fmt='img_tag_outside_think')], final_raw='clean', final_len=10),
    row('6', 'visref_raw_rank', [step(end='eos')], step1_info={'step1_source': 'stored', 'step1_tokens_match': False}),
]


def test_quantiles_and_empty():
    assert ab.quantiles([]) is None
    assert ab.quantiles([1, 2, 3, 4, 5]) == [1.4, 2.0, 3.0, 4.0, 4.6]


def test_summarize_groups_counts_and_warning():
    res = ab.summarize(ROWS)
    assert list(res['methods']) == ['visref_imgtag', 'visref_raw_rank']
    raw, bun = res['methods']['visref_raw_rank'], res['methods']['visref_imgtag']
    a = raw['all']
    assert a['n_rows'] == 3 and a['n_steps'] == 3 and a['k_dist'] == {1: 3} and a['stop_reason'] == {'entropy': 3}
    assert raw['close']['n_rows'] == 1 and raw['cap']['n_rows'] == 1 and raw['eos']['n_rows'] == 1
    # (i) rows 1, 2, 6: pairs valid for all three; row 1 differs ('3' vs '4'), row 2 equal after strip, row 6 equal
    assert a['n_pairs_valid'] == 3 and a['n_pairs_differ'] == 1 and a['pair_differ_rate'] == round(1 / 3, 4)
    assert a['n_dual_steps'] == 3 and a['n_crn_steps'] == 2 and a['pairs_not_interpretable'] is True
    assert raw['close']['pairs_not_interpretable'] is False and raw['cap']['pairs_not_interpretable'] is True
    assert raw['cap']['final_finish_cap'] == 1 and raw['cap']['final_forced_answer'] == 1
    assert raw['cap']['final_len_q'] == [900.0] * 5
    assert a['step1_source'] == {'stored': 1} and a['step1_tokens_mismatch'] == 1
    assert len(res['warnings']) == 1 and 'visref_raw_rank' in res['warnings'][0]
    b = bun['all']
    assert b['n_rows'] == 3 and b['n_steps'] == 4 and b['k_dist'] == {1: 2, 2: 1}
    assert b['status_not_ok'] == 1 and b['stop_reason'] == {'entropy': 2, 'kmax': 1}
    # (ii) re-emit, run problems, mechanism
    assert b['final_y_start_gt0'] == 1 and b['final_raw_has_think_end'] == 1 and b['n_final_raw'] == 2
    assert b['empty_final_answer'] == 1 and b['steps_n_inject_0'] == 1 and b['immediate_close_steps'] == 1
    assert b['n_dual_steps'] == 3 and b['pairs_not_interpretable'] is False and b['pair_differ_rate'] == 0.0
    assert b['inject_format'] == {'img_tag_outside_think': 4}
    assert b['inject_frac_q'] is not None and b['inject_frac_q'][2] == 0.5 and bun['cap']['n_rows'] == 1


def test_forced_answer_with_nothing_after_the_forced_prompt_counts_as_empty():
    assert ab.FORCED_PROMPT == '\nFinal Answer:'
    rows = [row('1', 'st', [step()], ans='\nFinal Answer:', final_len=0, final_forced_answer=True),
            row('2', 'st', [step()], ans='\nFinal Answer: 7', final_len=3, final_forced_answer=True),
            row('3', 'st', [step()], ans='\nFinal Answer:  \n', final_len=2, final_forced_answer=True),
            row('4', 'st', [step()], ans='7', final_len=0)]
    assert [ab.empty_final_answer(r) for r in rows] == [True, False, True, True]
    assert ab.summarize_group(rows[:2], 'st')['empty_final_answer'] == 1
    assert ab.summarize_group(rows, 'st')['empty_final_answer'] == 3


def test_non_visref_methods_skip_inject_checks_and_missing_flags_are_tolerated():
    res = ab.summarize([row('1', 'tsr', [{'k': 1, 'think_end': 'close', 'think_tokens': 5}], final_len=None)])
    a = res['methods']['tsr']['all']
    assert a['steps_n_inject_0'] is None and a['n_dual_steps'] == 0 and a['pair_differ_rate'] is None
    assert a['final_len_q'] is None and a['inject_frac_q'] is None and res['warnings'] == []


def test_table_and_main_write_json(tmp_path, capsys):
    res = ab.summarize(ROWS)
    t = ab.table(res, 'all').splitlines()
    assert t[0].split() == ['metric', 'visref_imgtag', 'visref_raw_rank']
    assert any(line.startswith('n_rows') for line in t)
    p = tmp_path / 'r.jsonl'
    p.write_text('\n'.join(json.dumps(r) for r in ROWS) + '\n\n')
    out = tmp_path / 'o.json'
    ab.main([str(tmp_path / '*.jsonl'), '--out', str(out)])
    cap = capsys.readouterr()
    assert '== close ==' in cap.out and 'WARNING' in cap.err
    j = json.loads(out.read_text())
    assert j['methods']['visref_imgtag']['all']['n_rows'] == 3 and len(j['warnings']) == 1
