"""diag/analyze_format_check.py: counts, the fixed rule (excess re-emission <= half of A6's), no correctness."""
import json
import sys
from pathlib import Path

import pytest

REPRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO / 'diag'))
import analyze_format_check as A  # noqa: E402

S1 = '\nFinal Answer:'


def row(pid, m, ans=S1 + ' 7', y0=0, raw=None, forced=True, status='ok', correct=1):
    return {'pid': pid, 'method': m, 'final_answer': ans, 'final_y_start': y0, 'final_raw': raw,
            'final_forced_answer': forced, 'status': status, 'correct': correct}


def reemit(pid, m):
    return row(pid, m, ans='', y0=3, raw='7\n</think>')


def test_flags():
    assert A.flags(row('1', A.IN)) == {'reemit': False, 'empty': False, 'restated': False}
    assert A.flags(reemit('1', A.IN)) == {'reemit': True, 'empty': True, 'restated': False}
    assert A.flags(row('1', A.IN, ans=S1 + ' Question: what is x?'))['restated']
    assert A.flags(row('1', A.IN, ans='8', y0=9, raw='Question: x\n</think>8'))['restated']
    assert A.flags(row('1', A.IN, ans=S1 + '\n**Final Answer**\n\\boxed{}'))['empty']


def test_limit_is_half_of_a6_excess():
    assert A.A6_EXCESS == 17 and A.LIMIT == 8


@pytest.mark.parametrize('n_in,n_tsr,ok', [(8, 0, True), (9, 0, False), (12, 4, True), (13, 4, False)])
def test_rule(n_in, n_tsr, ok):
    pids = [str(i) for i in range(20)]
    rows = [reemit(p, A.IN) if i < n_in else row(p, A.IN) for i, p in enumerate(pids)]
    rows += [reemit(p, A.TSR) if i < n_tsr else row(p, A.TSR) for i, p in enumerate(pids)]
    res = A.analyze(rows, pids)
    assert res['reemit_excess'] == n_in - n_tsr and res['pass'] is ok
    for r in rows:
        r['correct'] = 0
    assert A.analyze(rows, pids) == res  # correctness is never read


def test_missing_rows_refused():
    with pytest.raises(AssertionError):
        A.analyze([row('1', A.IN)], ['1'])


def test_a6_constant_matches_the_stored_a6_rows():
    pids = json.loads(A.IDS.read_text())
    a6 = A.load(REPRO / 'kaggle/runs/imgtag_main/output/results/results_0.jsonl')
    c, _ = A.counts(a6, pids, A.A6)
    assert c['visref_imgtag']['reemit'] == 38 and c['tsr_answer_prompt']['reemit'] == 21
    rows = [row(p, m) for p in pids for m in A.METHODS]
    res = A.analyze(rows, pids, a6)
    assert res['a6_reference']['visref_imgtag']['reemit'] == 38 and res['pass']


def test_main_writes_json(tmp_path):
    pids = json.loads(A.IDS.read_text())
    f = tmp_path / 'r.jsonl'
    f.write_text(''.join(json.dumps(row(p, m)) + '\n' for p in pids for m in A.METHODS))
    assert A.main(['--rows', str(f), '--out', str(tmp_path / 'o.json')]) == 0
    out = json.loads((tmp_path / 'o.json').read_text())
    assert out['n'] == 200 and out['pass'] and out['errors'] == {m: 0 for m in A.METHODS}


# ---- additional observations (never used for the decision) -----------------------------------------------------------
def test_pair_counts_and_range():
    a = {str(i): {'correct': int(i < 6), 'prediction': 'x' if i < 8 else 'y'} for i in range(10)}
    b = {str(i): {'correct': int(2 <= i < 9), 'prediction': 'x'} for i in range(10)}
    r = A._pair(a, b, [str(i) for i in range(10)])
    assert (r['both'], r['a_only'], r['b_only'], r['neither']) == (4, 2, 3, 1) and r['same_prediction'] == 8
    assert r['acc_a'] == 60.0 and r['acc_b'] == 70.0 and r['diff_pp'] == -10.0
    assert r['range95_pp'][0] < -10.0 < r['range95_pp'][1]


def _as_new_run(a6, pids):
    """A6 rows renamed as if they came from the new run (visref_imgtag -> the new method, TSR unchanged)."""
    out = []
    for r in a6:
        if str(r['pid']) in pids and r['method'] in A.A6:
            out.append(dict(r, method=A.IN if r['method'] == A.A6[0] else A.TSR))
    return out


def test_observe_on_identical_runs_shows_no_difference():
    pids = json.loads(A.IDS.read_text())
    a6 = A.load(REPRO / 'kaggle/runs/imgtag_main/output/results/results_0.jsonl')
    fake = lambda r: dict(r, correct=int(r['pid']) % 2, prediction=str(r['pid']), extract_failed=False)  # noqa: E731
    o = A.observe(_as_new_run(a6, set(pids)), a6, pids, fake)
    for k in ('hardware', 'vs_a6'):
        assert o[k]['same_prediction'] == 200 and o[k]['diff_pp'] == 0.0 and o[k]['a_only'] == o[k]['b_only'] == 0
    assert o['reemit']['new_in']['reemit'] == o['reemit']['a6_imgtag']['reemit'] == 38
    assert o['reemit']['tsr']['reemit'] == 21
    by = o['by_step1']['a6_imgtag']
    assert {e: v['n'] for e, v in by.items()} == {'cap': 131, 'close': 64, 'eos': 5}
    assert o['behaviour']['new_in'] == o['behaviour']['a6_imgtag'] and o['extract_failed']['new_in'] == 0


def test_main_observe_with_the_rule_scorer(tmp_path):
    pids = json.loads(A.IDS.read_text())
    a6p = REPRO / 'kaggle/runs/imgtag_main/output/results/results_0.jsonl'
    rows = _as_new_run(A.load(a6p), set(pids))
    f0, f1 = tmp_path / 'results_0.jsonl', tmp_path / 'results_1.jsonl'  # one file per GPU, as on Kaggle
    f0.write_text(''.join(json.dumps(r) + '\n' for r in rows[::2]))
    f1.write_text(''.join(json.dumps(r) + '\n' for r in rows[1::2]))
    (tmp_path / 'results_0.jsonl.partial').write_text('not json\n')  # unfinished jobs are not read
    out = tmp_path / 'o.json'
    assert A.main(['--rows', str(tmp_path / 'results_*.jsonl'), '--a6', str(a6p), '--observe',
                   '--data', str(REPRO.parent / 'data/raw/mathvista_testmini.json'), '--out', str(out)]) == 0
    res = json.loads(out.read_text())
    assert res['reemit_excess'] == 17 and not res['pass']  # A6 itself fails the rule
    ob = res['observations']
    assert ob['hardware']['diff_pp'] == 0.0 and ob['hardware']['same_prediction'] == 200
    assert ob['within']['new_in_minus_tsr'] == ob['within']['a6_imgtag_minus_tsr']
