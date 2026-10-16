"""diag/analyze_imgtag_main.py (image-tag injection run, analysis written before the results)."""
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


am, ah, at = _load('analyze_imgtag_main'), _load('analyze_hs'), _load('analyze_think')

Q = {'question_type': 'free_form', 'answer_type': 'integer', 'answer': '7', 'precision': None, 'choices': None}
FP = '\nFinal Answer:'
M1, M2, M3 = am.METHODS


def row(pid, method, ok, end='close', cv='abc', **kw):
    r = {'pid': pid, 'method': method, 'final_answer': FP + (' 7' if ok else ' 3'), 'correct': int(ok), 'k': 2,
         'stop_reason': 'entropy', 'final_y_start': 0, 'final_af': False, 'final_forced_answer': True, 'final_len': 6,
         'final_context_len': 1500, 'final_raw': None, 'think_tokens': 900, 'status': 'ok', 'code_version': cv,
         'steps': [{'think_end': end, 'think_tokens': 600}, {'think_end': 'close', 'think_tokens': 300}],
         'step1_info': {'step1_tokens_match': True, 'step1_store_sha': 'f937478f3b26'}}
    if method == M1:
        for s in r['steps']:
            s.update(m=500, n_inject=100, probe_valid=True, probe_inner='7', probe_noV_inner='7', probe_noV_finish='end',
                     probe_noV_crn=True)
    r.update(kw)
    return r


def make(n=40, ok=None, drop=(), ends=None):
    """ok(method, i) -> correct; default: all right except tsr_answer_prompt/tsr wrong on every 4th question."""
    ok = ok or (lambda m, i: True if m == M1 else i % 4 != 0)
    pids = [str(i) for i in range(1, n + 1)]
    order = pids[:]
    random.Random(0).shuffle(order)
    rows = {(p, m): row(p, m, ok(m, i), end=(ends or (lambda i: 'close'))(i)) for i, p in enumerate(pids) for m in am.METHODS
            if (p, m) not in drop}
    return rows, order, {p: Q for p in pids}


@pytest.mark.parametrize('r, want', [
    ({'final_forced_answer': True, 'final_answer': FP}, True),
    ({'final_forced_answer': True, 'final_answer': FP + '  \n'}, True),
    ({'final_forced_answer': True, 'final_answer': FP + ' 7'}, False),
    ({'final_forced_answer': False, 'final_answer': FP}, False),   # not a forced row
    ({'final_forced_answer': False, 'final_answer': ''}, False),
    ({'final_forced_answer': True, 'final_answer': FP + '\n**Final Answer**\n\\boxed'}, True),    # DEER prompt, nothing after
    ({'final_forced_answer': True, 'final_answer': FP + '\n**Final Answer**\n\\boxed{}'}, True),
    ({'final_forced_answer': True, 'final_answer': FP + '\n**Final Answer**\n\\boxed{ \n }'}, True),
    ({'final_forced_answer': True, 'final_answer': FP + '\n\n'}, True),
    ({'final_forced_answer': True, 'final_answer': FP + '\n**Final Answer**\n\\boxed{7}'}, False),
    ({'final_forced_answer': True, 'final_answer': FP + ' 7\n**Final Answer**\n\\boxed{}'}, False),  # an answer precedes it
    ({'final_forced_answer': False, 'final_answer': FP + '\n**Final Answer**\n\\boxed{}'}, False),
    ({'final_forced_answer': True, 'final_answer': ''}, True),
    ({'final_forced_answer': True}, True),
])
def test_empty_after_prompt(r, want):
    assert am.empty_after_prompt(r) is want


def test_verdict_rule_is_the_think_rule_with_own_texts():
    for lo, hi, n in [(1, 21, 600), (1, 20, 600), (0, 21, 600), (-5, 20, 600), (-30, -1, 600)]:
        v, t = am.verdict(lo, hi, n), at.verdict(lo, hi, n)
        assert v['code'] == t['code'] and v['text'].startswith(am.VERDICT[v['code']])
    assert '논문보다 작음' in am.verdict(1, 20, 600)['text'] and '논문보다 작음' not in am.verdict(1, 21, 600)['text']


def test_primary_comparison_secondary_strata():
    rows, order, data = make(n=40, ends=lambda i: ('close', 'cap', 'eos', 'close')[i % 4])
    out = am.analyze(rows, order, data)
    r = out['rules_only']
    assert out['n_prefix'] == 40 and out['power_flag'] == '검출력 부족' and out['main_reading'].startswith('rules_only')
    assert r['acc'] == {M1: 100.0, M2: 75.0, M3: 75.0}
    assert r['primary']['diff_pp'] == 25.0 and (r['primary']['a_only'], r['primary']['b_only']) == (10, 0)
    assert r['verdict']['code'] == 1 and r['power']['discordant'] == 10
    assert set(r['secondary']) == {'visref_imgtag-tsr', 'tsr_answer_prompt-tsr'}
    sec = r['secondary']['visref_imgtag-tsr']
    assert sec['rule_code'] == 1 and sec['label'] == 'same rule applied, secondary' and sec['diff_pp'] == 25.0
    assert 'verdict' not in sec and 'rule' not in sec and 'text' not in sec  # no verdict wording outside the main comparison
    ap_tsr = r['secondary']['tsr_answer_prompt-tsr']
    assert ap_tsr['diff_pp'] == 0.0 and not {'verdict', 'rule', 'rule_code', 'label'} & set(ap_tsr)
    assert 'text' in r['verdict']
    st = r['strata_step1_end']
    assert set(st) == {'close', 'cap', 'eos', 'other'} and st['other'] == 0
    assert (st['close']['n'], st['cap']['n'], st['eos']['n']) == (20, 10, 10)
    assert st['close']['acc'][M1] == 100.0 and st['close']['main_diff_pp'] == 50.0 and st['close']['a_only'] == 10
    assert st['cap']['main_diff_pp'] == 0.0 and st['eos']['acc'][M2] == 100.0


def test_verdicts_undecided_and_negative():
    same, order, data = make(n=40, ok=lambda m, i: i % 2 == 0)  # identical in all methods: CI [0, 0]
    assert am.analyze(same, order, data)['verdict']['code'] == 3
    # M1 wrong on 3 questions where M2 is right and the reverse on 3 others: CI [-5, 5] contains 0 and reaches +3.5 pp
    noisy, order, data = make(n=40, ok=lambda m, i: (not i < 3) if m == M1 else (not 3 <= i < 6))
    out = am.analyze(noisy, order, data)['rules_only']
    assert (out['primary']['a_only'], out['primary']['b_only']) == (3, 3) and out['verdict']['code'] == 2
    worse, order, data = make(n=40, ok=lambda m, i: False if m == M1 else True)
    assert am.analyze(worse, order, data)['verdict']['code'] == 3  # CI entirely below 0


def test_complete_prefix_with_one_shard_needs_all_three():
    rows, order, data = make(drop=((make()[1][5], 'tsr_answer_prompt'),))
    out = am.analyze(rows, order, data, shards=1)
    assert out['n_prefix'] == 5 and out['n_rows'][M2] == 39 and out['shards'] == 1
    rows, order, data = make(drop=((make()[1][5], 'tsr'),))
    assert am.analyze(rows, order, data)['n_prefix'] == 5


def test_power_flag_off_with_300():
    rows, order, data = make(n=300)
    assert am.analyze(rows, order, data)['power_flag'] is None


def test_code_version_and_store_sha_must_be_unique():
    rows, order, data = make(n=8)
    k = (order[0], M3)
    rows[k] = dict(rows[k], code_version='other')
    with pytest.raises(AssertionError, match='several code versions'):
        am.analyze(rows, order, data)
    rows, order, data = make(n=8)
    rows[k] = dict(rows[k], step1_info={'step1_tokens_match': True, 'step1_store_sha': 'deadbeef0000'})
    with pytest.raises(AssertionError, match='several stored step-1 files'):
        am.analyze(rows, order, data)
    rows, order, data = make(n=8)
    for r in rows.values():
        r.pop('step1_info')  # absent everywhere: fine
    assert am.analyze(rows, order, data)['step1_store_sha'] == []


def test_empty_after_prompt_is_wrong_in_both_readings_and_not_sent_to_llm():
    rows, order, data = make(n=40, ok=lambda m, i: True)
    bad = (order[2], M1)
    rows[bad] = dict(rows[bad], final_answer=FP + '  \n', correct=1)  # would be 'right' if the recorded flag were trusted
    out = am.analyze(rows, order, data)
    assert out['empty_after_prompt'] == {M1: 1, M2: 0, M3: 0} and out['rules_only']['acc'][M1] == 97.5
    assert out['mechanism'][M1]['empty_after_prompt'] == 1
    # an LLM file without that row is fine; one that contains it (and says right) is overridden
    llm = {k: dict(r, llm_used=False) for k, r in rows.items() if k != bad}
    assert am.analyze(rows, order, data, llm)['main']['acc'][M1] == 97.5
    llm[bad] = dict(rows[bad], correct=1, llm_used=True)
    main = am.analyze(rows, order, data, llm)
    assert main['main']['acc'][M1] == 97.5 and main['llm_used'][M1] == 0
    del llm[(order[0], M2)]
    with pytest.raises(AssertionError, match='lacks 1 prefix rows'):
        am.analyze(rows, order, data, llm)


def test_llm_reading_replaces_rule_rows():
    rows, order, data = make(n=40)
    llm = {k: dict(r, correct=0, llm_used=True) if k[1] == M3 else dict(r) for k, r in rows.items()}
    out = am.analyze(rows, order, data, llm)
    assert out['main_reading'] == 'main' and out['verdict'] == out['main']['verdict']
    assert out['main']['acc'][M3] == 0.0 and out['rules_only']['acc'][M3] == 75.0 and out['llm_used'][M3] == 40


def test_rules_only_rescores_with_the_current_scorer():
    rows, order, data = make(n=8, ok=lambda m, i: True)
    k = (order[0], M2)
    rows[k] = dict(rows[k], correct=0, final_answer=FP + ' 7')  # recorded wrong, text right
    assert am.analyze(rows, order, data)['rules_only']['acc'][M2] == 100.0


def test_mechanism_counts():
    rows, order, data = make(n=40)
    a, b, c = order[:3]
    rows[(a, M1)]['steps'][0]['probe_noV_inner'] = '3'                # probe pair differs
    rows[(b, M1)]['final_y_start'] = 12                               # second </think>
    rows[(c, M2)] = dict(rows[(c, M2)], status='error', final_af=True, final_len=0)
    rows[(c, M3)]['step1_info']['step1_tokens_match'] = False
    rows[(c, M3)]['final_context_len'] = 4000
    m = am.analyze(rows, order, data)['mechanism']
    assert m[M1]['n_pairs_valid'] == 80 and m[M1]['n_pairs_differ'] == 1 and m[M1]['final_y_start_gt0'] == 1
    assert m[M1]['second_think_end_rate'] == 0.025 and m[M1]['k_dist'] == {2: 40}
    assert m[M1]['inject_over_m_q25_50_75'] == [0.2, 0.2, 0.2] and m[M2]['inject_over_m_q25_50_75'] is None
    assert m[M2]['status_not_ok'] == 1 and m[M2]['final_af'] == 1 and m[M3]['final_context_len_max'] == 4000
    assert m[M3]['step1_tokens_match'] == 39 and m[M3]['n_step1_info'] == 40
    assert m[M1]['stop_reason'] == {'entropy': 40} and m[M1]['final_len_q'] == [6.0] * 5


def _session(tmp, name, order, rows):
    d = tmp / name
    d.mkdir()
    (d / 'ids.json').write_text(json.dumps(order))
    (d / 'results_0.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return d


def test_duplicate_pair_in_two_files_is_an_error(tmp_path):
    rows, order, data = make(n=8)
    vals = list(rows.values())
    d1, d2 = _session(tmp_path, 'a', order, vals[:10]), _session(tmp_path, 'b', order, vals[10:])
    got, o = am.load_runs([d1, d2], am.METHODS)
    assert got == rows and o == order
    d3 = _session(tmp_path, 'c', order, vals[:1])
    with pytest.raises(AssertionError, match='finished in two sessions'):
        am.load_runs([d1, d3], am.METHODS)


def test_cli_rows_and_analyze(tmp_path, capsys):
    rows, order, data = make(n=12)
    bad = (order[0], M1)
    rows[bad] = dict(rows[bad], final_answer=FP)
    d = _session(tmp_path, 'imgtag_main_x', order, list(rows.values()))
    dp = tmp_path / 'data.json'
    dp.write_text(json.dumps(data))
    out_rows = tmp_path / 'rows.jsonl'
    am.main(['rows', '--run-dirs', str(d), '--data', str(dp), '--out', str(out_rows)])
    got = [json.loads(l) for l in out_rows.read_text().splitlines()]
    assert len(got) == 35 and (bad[0], bad[1]) not in {(g['pid'], g['method']) for g in got}
    assert all(g['scorer_rescored'] for g in got)
    out_json = tmp_path / 'a.json'
    am.main(['analyze', '--run-dirs', str(d), '--data', str(dp), '--out', str(out_json)])
    res = json.loads(out_json.read_text())
    assert res['n_prefix'] == 12 and res['empty_after_prompt'][M1] == 1 and res['code_version'] == ['abc']


def test_final_answer_phrase_matcher():
    assert am.FINAL_ANSWER_PHRASE.search('Final   Answer') and am.FINAL_ANSWER_PHRASE.search('FINALANSWER')
    assert not am.FINAL_ANSWER_PHRASE.search('7')


def test_extracted_phrase_counted_from_rule_extraction():
    rows, order, data = make(n=8, ok=lambda m, i: True)
    q = dict(Q, question_type='free_form', answer_type='text', answer='Final Answer')
    data[order[0]] = q
    rows[(order[0], M3)] = dict(rows[(order[0], M3)], final_answer='\\boxed{Final Answer}', final_forced_answer=False)
    assert am.analyze(rows, order, data)['mechanism'][M3]['extracted_final_answer_phrase'] == 1


def bmain_rows(order, ok):
    return {p: {'pid': p, 'method': 'tsr', 'final_answer': '\\boxed{%s}' % ('7' if ok(i) else '3'), 'correct': int(ok(i))}
            for i, p in enumerate(sorted(order, key=int))}


def test_reference_only_against_main_run_tsr():
    rows, order, data = make(n=40)  # this run's tsr: wrong on every 4th question (i % 4 == 0 -> 10 wrong)
    bm = bmain_rows(order, lambda i: i % 2 == 0)  # main-run tsr: wrong on odd i (20 wrong)
    out = am.analyze(rows, order, data, bmain_tsr=bm)
    ec = out['rules_only']['reference_only_tsr_vs_main_run_tsr']
    assert ec['n'] == 40 == ec['n_prefix_covered'] and ec['label'].startswith('reference only: cannot separate causes')
    assert not any(w in ec['label'].lower() for w in ('alarm', 'check'))
    assert ec['acc'] == {'tsr': 75.0, 'main_run_tsr': 50.0} and ec['diff_pp'] == 25.0
    assert not {'verdict', 'rule', 'rule_code'} & set(ec)
    assert out['verdict']['code'] and 'reference_only_tsr_vs_main_run_tsr' not in out['rules_only']['secondary']
    # main-run rows are re-scored by the current scorer in the rules-only reading, used as recorded in the main reading
    bm2 = {p: dict(r, correct=0) for p, r in bm.items()}
    llm = {k: dict(r) for k, r in rows.items()}
    both = am.analyze(rows, order, data, llm, bmain_tsr=bm2)
    assert both['rules_only']['reference_only_tsr_vs_main_run_tsr']['acc']['main_run_tsr'] == 50.0   # re-scored from the text
    assert both['main']['reference_only_tsr_vs_main_run_tsr']['acc']['main_run_tsr'] == 0.0          # as recorded


def test_reference_only_skipped_without_main_run_rows():
    rows, order, data = make(n=8)
    ec = am.analyze(rows, order, data)['rules_only']['reference_only_tsr_vs_main_run_tsr']
    assert ec['skipped'] == 'main-run TSR rows not found'
    partial = bmain_rows(order, lambda i: True)
    for p in order[3:]:
        del partial[p]
    ec = am.analyze(rows, order, data, bmain_tsr=partial)['rules_only']['reference_only_tsr_vs_main_run_tsr']
    assert ec['n'] == 3 == ec['n_prefix_covered']


def test_load_bmain_tsr_sources(tmp_path):
    merged = tmp_path / 'merged.jsonl'
    merged.write_text(json.dumps({'pid': '1', 'method': 'tsr', 'correct': 1}) + '\n' +
                      json.dumps({'pid': '1', 'method': 'st', 'correct': 0}) + '\n')
    got, note = am.load_bmain_tsr(merged, [])
    assert list(got) == ['1'] and note == str(merged)
    d = tmp_path / 'b_main_s1'
    d.mkdir()
    (d / 'results_0.jsonl').write_text(json.dumps({'pid': '2', 'method': 'tsr', 'correct': 0}) + '\n')
    got, note = am.load_bmain_tsr(tmp_path / 'absent.jsonl', [d])
    assert list(got) == ['2'] and 'raw' in note
    assert am.load_bmain_tsr(None, [tmp_path / 'nothing']) == (None, 'main-run TSR rows not found: skipped')


def test_cli_reference_only_skip_note(tmp_path):
    rows, order, data = make(n=12)
    d = _session(tmp_path, 'imgtag_main_x', order, list(rows.values()))
    dp = tmp_path / 'data.json'
    dp.write_text(json.dumps(data))
    out_json = tmp_path / 'a.json'
    am.main(['analyze', '--run-dirs', str(d), '--data', str(dp), '--bmain', str(tmp_path / 'no.jsonl'), '--bmain-dirs',
             str(tmp_path / 'none'), '--out', str(out_json)])
    res = json.loads(out_json.read_text())
    assert res['bmain_tsr_source'].startswith('main-run TSR rows not found')
    assert res['rules_only']['reference_only_tsr_vs_main_run_tsr']['skipped']


def test_tsr_pair_same_thinking():
    rows, order, data = make(n=40, ends=lambda i: ('close', 'cap', 'eos', 'close')[i % 4])
    for (p, m), r in rows.items():
        r['thinking'] = f'think {p} '  # same thinking in tsr / tsr_answer_prompt
    out = am.analyze(rows, order, data)['tsr_pair_same_thinking']
    assert (out['same'], out['total'], out['rate']) == (40, 40, 1.0) and out['differing_examples'] == []
    assert out['by_step1_end']['close'] == {'same': 20, 'total': 20, 'rate': 1.0} and out['by_step1_end']['other']['total'] == 0
    assert out['by_step1_end']['other']['rate'] is None and out['n_skipped_error'] == 0
    # differing step count, differing step tokens, differing text only, differing stop reason; one pair with an error row
    a, b, c, d, e = order[:5]
    rows[(a, M2)] = dict(rows[(a, M2)], steps=rows[(a, M2)]['steps'][:1])
    rows[(b, M2)]['steps'][1]['think_tokens'] = 301
    rows[(c, M2)]['thinking'] = 'think ' + c + ' X'
    rows[(d, M2)]['stop_reason'] = 'cap'
    rows[(e, M2)] = dict(rows[(e, M2)], status='error')
    out = am.analyze(rows, order, data)['tsr_pair_same_thinking']
    assert (out['same'], out['total'], out['n_skipped_error']) == (35, 39, 1) and out['rate'] == round(35 / 39, 4)
    ex = {x['pid']: x for x in out['differing_examples']}
    assert set(ex) == {a, b, c, d} and ex[a]['step'] == 1 and ex[a]['what'] == 'step_count' and ex[b]['step'] == 1
    assert ex[c] == {'pid': c, 'step': None, 'what': 'thinking_text', 'char': len('think ' + c + ' ')}
    assert ex[d]['what'] == 'stop_reason'
    assert sum(v['total'] for v in out['by_step1_end'].values()) == 39
    # at most 5 examples
    for p in order[5:12]:
        rows[(p, M3)]['thinking'] = 'other'
    assert len(am.analyze(rows, order, data)['tsr_pair_same_thinking']['differing_examples']) == 5


def test_cli_prints_one_line_for_tsr_pair(tmp_path, capsys):
    rows, order, data = make(n=12)
    d = _session(tmp_path, 'imgtag_main_x', order, list(rows.values()))
    dp = tmp_path / 'data.json'
    dp.write_text(json.dumps(data))
    am.main(['analyze', '--run-dirs', str(d), '--data', str(dp), '--bmain', str(tmp_path / 'no.jsonl'), '--bmain-dirs', str(tmp_path / 'none')])
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith('tsr_pair_same_thinking:')]
    assert len(lines) == 1 and 'same 12/12' in lines[0]


def test_reemit_vs_answer_cap():
    rs = [{'final_y_start': 5, 'final_finish': 'cap', 'final_af': True},
          {'final_y_start': 3, 'final_finish': 'stop'},
          {'final_y_start': 0, 'final_finish': 'cap'},
          {'final_finish': 'stop'}]
    out = am.reemit_vs_answer_cap(rs)
    assert out == {'reemit': {'n': 2, 'answer_cap': 1, 'answer_forced_deer': 1},
                   'no_reemit': {'n': 2, 'answer_cap': 1, 'answer_forced_deer': 0}}


def _row(m, steps, gen=10, wall=1.0, think=5):
    return {'method': m, 'steps': steps, 'gen_tokens': gen, 'wall_s': wall, 'think_tokens': think, 'final_context_len': 100}


def test_cost_ratios():
    rows = {('1', 'visref_imgtag'): _row('visref_imgtag', [], gen=30, wall=3.0),
            ('1', 'tsr_answer_prompt'): _row('tsr_answer_prompt', [], gen=20, wall=2.0),
            ('1', 'tsr'): _row('tsr', [], gen=10, wall=1.0)}
    out = am.cost(rows, ['1'])
    assert out['visref_imgtag']['gen_tokens_total'] == 30
    assert out['visref_imgtag_over_tsr_answer_prompt'] == {'gen_tokens_total': 1.5, 'wall_s_median': 1.5}
    assert out['visref_imgtag_over_tsr'] == {'gen_tokens_total': 3.0, 'wall_s_median': 3.0}


def test_stop_entropy():
    s1 = {'probe_valid': True, 'H': 0.1, 'probe_ent': [0.0, 0.3], 'probe_noV_crn': True, 'H_noV': 0.4}
    s2 = {'probe_valid': True, 'H': 0.3, 'probe_ent': [0.0, 0.3], 'probe_noV_crn': True, 'H_noV': 0.35}
    s3 = {'probe_valid': False, 'H': 9.0}
    rows = {('1', 'visref_imgtag'): _row('visref_imgtag', [s1, s2, s3]),
            ('1', 'tsr_answer_prompt'): _row('tsr_answer_prompt', [s1]),
            ('1', 'tsr'): _row('tsr', [])}
    out = am.stop_entropy(rows, ['1'])
    v = out['visref_imgtag']
    assert v['n_steps'] == 2 and v['below_delta'] == 1 and v['near_delta'] == 1
    assert v['first_content_token'] == {'n': 2, 'decision_differs': 1}
    assert v['with_vs_without_image'] == {'n': 2, 'mean_H_minus_H_noV': -0.175, 'decision_differs': 1}
    assert out['tsr']['n_steps'] == 0
