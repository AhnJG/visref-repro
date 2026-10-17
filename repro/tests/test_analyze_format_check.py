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
