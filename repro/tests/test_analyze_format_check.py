"""diag/analyze_format_check.py: check-1 counts and the fixed rule (no correctness)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'diag'))
import analyze_format_check as A  # noqa: E402

S1 = '\nFinal Answer:'


def row(pid, m, ans=S1 + ' 7', y0=0, raw=None, forced=True, status='ok', correct=1):
    return {'pid': pid, 'method': m, 'final_answer': ans, 'final_y_start': y0, 'final_raw': raw,
            'final_forced_answer': forced, 'status': status, 'correct': correct}


def test_flags():
    assert A.flags(row('1', A.IN)) == {'reemit': False, 'empty': False, 'restated': False}
    f = A.flags(row('1', A.IN, ans='', y0=4, raw=' 7\n</think>'))
    assert f == {'reemit': True, 'empty': True, 'restated': False}
    assert A.flags(row('1', A.IN, ans=S1 + ' Question: what is x?'))['restated']
    assert A.flags(row('1', A.IN, ans='8', y0=9, raw='Question: x\n</think>8'))['restated']  # restated before the </think>
    assert A.flags(row('1', A.IN, ans=S1 + '\n**Final Answer**\n\\boxed{}'))['empty']


def test_limit():
    assert [A.limit(t) for t in (0, 1, 2, 6, 7, 10)] == [3, 4, 5, 9, 11, 15]


def test_analyze_rule_and_no_correctness():
    pids = ['1', '2', '3', '4']
    rows = [row(p, m) for p in pids for m in A.METHODS]
    for p in pids:  # '_in' re-emits on all 4, TSR on none: 4 > 3 fails; '_out' is reported only
        rows.append(row(p, A.IN, ans='', y0=3, raw='7\n</think>'))
    by = {}
    for r in rows:
        by[(r['pid'], r['method'])] = r  # later rows win, as in analyze()
    res = A.analyze(list(by.values()), pids)
    assert res['counts'][A.IN] == {'reemit': 4, 'empty': 4, 'restated': 0}
    assert res['counts'][A.TSR] == {'reemit': 0, 'empty': 0, 'restated': 0}
    assert res['check1']['reemit'] == {'in': 4, 'tsr': 0, 'limit': 3, 'pass': False} and not res['check1_pass']
    for r in by.values():
        r['correct'] = 0
    assert A.analyze(list(by.values()), pids) == res  # correctness is never read
    with pytest.raises(AssertionError):
        A.analyze([r for r in by.values() if r['pid'] != '4'], pids)


def test_main_writes_json(tmp_path, monkeypatch):
    pids = json.loads(A.IDS.read_text())
    rows = [row(p, m) for p in pids for m in A.METHODS]
    f = tmp_path / 'r.jsonl'
    f.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    assert A.main(['--rows', str(f), '--out', str(tmp_path / 'o.json')]) == 0
    out = json.loads((tmp_path / 'o.json').read_text())
    assert out['n'] == 100 and out['check1_pass'] and out['errors'] == {m: 0 for m in A.METHODS}
