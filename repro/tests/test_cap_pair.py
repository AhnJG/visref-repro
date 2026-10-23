"""CPU tests for diag/analyze_cap_pair.py, kaggle/build_cap_pair.py and diag/cap_pair_ids.json."""
import ast
import base64
import json
import re
import sys
from pathlib import Path

import pytest

REPRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO / 'diag'))
sys.path.insert(0, str(REPRO / 'kaggle'))
import analyze_cap_pair as an  # noqa: E402
import build_cap_pair as bc  # noqa: E402

IDS = [str(i) for i in range(42)]


def rows(up, down, base=0):
    """R with ST(4096) better on the first `up`, worse on the next `down` questions; everything else 0 correct."""
    R = {}
    for cap in an.CAPS:
        for i, p in enumerate(IDS):
            for m in an.METHODS:
                c = 0
                if m == 'st':
                    c = int(cap == 4096) if i < up else int(cap == 1024) if i < up + down else base
                R[(cap, p, m)] = {'pid': p, 'method': m, 'status': 'ok', 'correct': c, 'k': 1, 'think_tokens': 10,
                                  'final_forced_answer': False, 'stop_reason': 'st', 'steps': [{'think_end': 'cap'}],
                                  'code_version': 'v'}
    return R


@pytest.mark.parametrize('up,down,want', [(4, 0, 'nc'), (0, 4, 'nc'), (6, 0, 'up'), (0, 6, 'down'), (0, 0, 'nc')])
def test_st_sentence_needs_the_exact_test(up, down, want):
    res = an.analyze(rows(up, down), IDS)
    s = res['st_4096_minus_1024']
    assert (s['up'], s['down']) == (up, down)
    assert res['sentence'].startswith({'up': an.UP, 'down': an.DOWN, 'nc': an.NOT_CONFIRMED}[want])
    if (up, down) == (4, 0):
        assert s['ci95_pp'][0] > 0 and abs(s['p_sign'] - 0.125) < 1e-9  # the review's case: CI above 0, p = 0.125
    if (up, down) == (6, 0):
        assert abs(s['p_sign'] - 0.03125) < 1e-9


def test_hold_and_duplicates(tmp_path):
    R = rows(0, 0)
    del R[(4096, '5', 'tsr')]
    assert an.analyze(R, IDS)['sentence'] == an.HOLD
    R = rows(0, 0)
    R[(1024, '3', 'st')]['status'] = 'error'
    assert an.analyze(R, IDS)['sentence'] == an.HOLD
    (tmp_path / 'results_1024_0.jsonl').write_text(json.dumps({'pid': '1', 'method': 'st'}) + '\n')
    (tmp_path / 'results_1024_1.jsonl').write_text(json.dumps({'pid': '1', 'method': 'st'}) + '\n')
    with pytest.raises(SystemExit):
        an.load(tmp_path)
    (tmp_path / 'results_1024_1.jsonl').write_text(json.dumps({'pid': '1', 'method': 'tsr'}) + '\n')
    (tmp_path / 'results_4096_0.jsonl').write_text(json.dumps({'pid': '1', 'method': 'st'}) + '\n')
    R, _ = an.load(tmp_path)
    assert set(R) == {(1024, '1', 'st'), (1024, '1', 'tsr'), (4096, '1', 'st')}  # settings are never merged


def test_ids_are_the_first_42_of_the_main_order():
    order = [str(p) for p in json.loads((REPRO / 'kaggle/runs/b_main_hs/output/results/ids.json').read_text())]
    assert json.loads((REPRO / 'diag/cap_pair_ids.json').read_text())['ids'] == order[:42]


def test_builder_two_calls_per_gpu_with_separate_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(bc, 'HERE', tmp_path)
    assert bc.main(['c', '--owner', 'me', '--timeout', '13500']) == 0
    rc = json.loads((tmp_path / 'runs/c/run_config.json').read_text())
    assert rc['caps'] == [1024, 4096] and rc['end_s'] == 12900 and rc['n_ids'] == 42 and rc['f1'] == 0.35
    assert rc['run_args'] == ['--methods', 'st,tsr,visref_raw_rank', '--total-think-cap', '10240']
    src = ''.join(json.loads((tmp_path / 'runs/c/notebook.ipynb').read_text())['cells'][0]['source'])
    ast.parse(src)
    assert "f'results_{cap}_{i}.jsonl'" in src and "f'log_{cap}_{i}.txt'" in src and "'--skip'" not in src
    assert "'--deadline', str(dl)" in src and "'--time-limit', str(dl - 300)" in src and 'if dl < 600' in src
    cfg = json.loads(base64.b64decode(re.search(r"CFG = json.loads\(base64.b64decode\('([^']+)'\)", src).group(1)))
    assert cfg['ids'] == json.loads((REPRO / 'diag/cap_pair_ids.json').read_text())['ids']
    # the time plan: first call gets f1 of the time left, the second the rest until END; both end before END
    left = cfg['end_s'] - 600  # e.g. 600 s of setup
    d1 = left * cfg['f1']
    d2 = left - d1
    assert d1 < d2 and d1 + d2 <= cfg['end_s']
    with pytest.raises(SystemExit):
        bc.main(['d', '--owner', 'me', '--f1', '1.2'])


def test_run_py_code_version_is_untouched():
    import run
    assert run.code_version() == '06435ae748c8'
