"""Synthetic tests for diag/step1_stop_table.py (no real data)."""
import importlib.util
import json
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location(
    'step1_stop_table', pathlib.Path(__file__).resolve().parent.parent / 'diag' / 'step1_stop_table.py')
sst = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sst)
INF = float('inf')


def test_auroc_perfect_and_reversed():
    assert sst.auroc([5.0, 6.0], [1.0, 2.0]) == 1.0
    assert sst.auroc([1.0, 2.0], [5.0, 6.0]) == 0.0


def test_auroc_constant_is_half():
    assert sst.auroc([1.0, 1.0, 1.0], [1.0, 1.0]) == 0.5


def test_auroc_inf_handling():
    assert sst.auroc([INF], [100.0, 5.0]) == 1.0          # inf larger than any finite
    assert sst.auroc([INF], [INF]) == 0.5                  # inf == inf is a tie
    # explicit: pairs (INF,INF)=.5 (INF,0)=1 (1,INF)=0 (1,0)=1 -> 2.5/4
    assert sst.auroc([INF, 1.0], [INF, 0.0]) == 0.625


def test_auroc_empty_class_is_none():
    assert sst.auroc([], [1.0]) is None and sst.auroc([1.0], []) is None


def _rec(k, pc, st_correct, H=1.0, qtype='multi_choice', end='cap'):
    return {'qtype': qtype, 'think_end': end, 'st_correct': st_correct,
            'tsr': {'k': k, 'H': H, 'pc': pc}, 'visref_raw_rank': {'k': k, 'H': H, 'pc': pc}}


def test_cell_counts_and_max_gain():
    recs = [_rec(1, 0, 0), _rec(1, 0, 1), _rec(1, 1, 1), _rec(2, 0, 0), _rec(2, 1, 0), _rec(1, 0, 0)]
    c = sst.cell(recs, 'tsr')
    assert c['n'] == 6 and c['n_stopped_k1'] == 4 and c['base_rate_k1'] == round(4 / 6, 4)
    assert c['probe_2x2'] == {'stopped_correct': 1, 'stopped_wrong': 3, 'continued_correct': 1, 'continued_wrong': 1}
    assert c['st_final_2x2'] == {'stopped_correct': 2, 'stopped_wrong': 2, 'continued_correct': 0, 'continued_wrong': 2}
    assert c['n_k1_probe_wrong'] == 3 and c['n_k1_probe_wrong_and_st_wrong'] == 2
    assert c['max_gain_pp'] == round(2 / 1000 * 100, 2)  # 0.2
    v = sst.cell(recs, 'visref_raw_rank')
    assert v['max_gain_pp'] == round(3 / 1000 * 100, 2) and 'st_final_2x2' not in v


def test_auroc_wrong_on_records_and_ci_skip():
    recs = [_rec(1, 0, 0, H=9.0), _rec(1, 0, 0, H=INF), _rec(1, 1, 1, H=1.0), _rec(1, 1, 1, H=2.0)]
    a = sst.auroc_wrong(recs, 'tsr', n_boot=50)
    assert a['auroc'] == 1.0 and a['n_inf'] == 1 and a['n_wrong'] == 2 and a['ci95'] is not None
    only_wrong = sst.auroc_wrong([_rec(1, 0, 0, H=1.0)], 'tsr', n_boot=10)
    assert only_wrong['auroc'] is None and only_wrong['ci95'] is None


def test_refuses_duplicate_rows(tmp_path):
    row = {'pid': 5, 'method': 'tsr', 'steps': []}
    f1, f2 = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'
    f1.write_text(json.dumps(row) + '\n')
    f2.write_text(json.dumps(dict(row, pid='5')) + '\n')
    with pytest.raises(ValueError):
        sst.load_rows([str(f1), str(f2)])
    # other methods ignored, even if repeated
    g = tmp_path / 'c.jsonl'
    g.write_text((json.dumps({'pid': 1, 'method': 'visref_raw', 'steps': []}) + '\n') * 2)
    assert sst.load_rows([str(g)]) == {}
