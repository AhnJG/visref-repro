"""Decision tests (2026-10-12 decision check, docs/IMPLEMENTATION_CHOICES.md (formerly SPEC_LEDGER) "핵심 판단 요약" J table, column "테스트").

The behaviour tests in test_repro.py use their own small configs, so they do not pin the values that were decided for
the runs. These tests do: if a decided value changes, the change has to be made here too, on purpose.
"entry <date> <title>" = the dated entry of that title in docs/EXPERIMENT_FLOW.md."""
import glob
import hashlib
import json
import pathlib
import sys

import pytest

import llm_extract as LX
import run
from visref.dpp import budget
from visref.methods import METHODS, PROBE_SAMPLING, THINK
from visref.scoring import SCORER_VERSION

REPRO = pathlib.Path(__file__).resolve().parent.parent
RUNS = REPRO / 'kaggle' / 'runs'
STORE = 'diag/step1_store_bmain.json'

DECIDED = {
    'delta': 0.25, 'kmax': 10, 'vis_frac': 0.3,              # J-4, paper (A)
    'temperature': 0.6, 'top_k': 50, 'top_p': 0.95,          # J-3, official model page (D); entry 2026-10-07 "원문 명세표·독립 감사 → b_test를 파일럿으로 재정의"
    'wait_text': 'Wait, think more',                         # J-5
    'think_cap': 1024, 'total_think_cap': 10240,             # J-10, entry 2026-10-07 "결정(작성자): s1 + DEER af 폴백, 정지 통계 현행 유지, 파일럿 없이 본 실험 시작" (default fixed 2026-10-13)
    'think_extra': 512,                                      # J-23, think experiments (user decision 2026-10-13)
    'answer_cap': 1024,                                      # J-10
    'probe_prompt': '\n\n\\boxed', 'probe_cap': 16,          # J-9, entry 2026-10-07 "smoke3 결과 → probe를 `\boxed` 형식으로 교체" (user agreed)
    'forced_prompt': '\nFinal Answer:',                      # J-11, entry 2026-10-07 "b_test 재빌드: s1 강제 답변 + ridge DPP" -> entry 2026-10-07 "원문 명세표·독립 감사 → b_test를 파일럿으로 재정의" (s1 string)
    'af_prompt': '\n**Final Answer**\n\\boxed', 'af_cap': 100,  # J-12, entry 2026-10-07 "결정(작성자): s1 + DEER af 폴백, 정지 통계 현행 유지, 파일럿 없이 본 실험 시작" (user decision)
    'seed': 42, 'max_num': 12, 'dual_probe': 0,
}


@pytest.mark.parametrize('key', sorted(DECIDED))
def test_run_defaults_hold_the_decided_values(key):
    assert run.DEFAULTS[key] == DECIDED[key]


def test_probe_sampling_and_method_names():
    assert PROBE_SAMPLING == (1.0, 0, 1.0)  # J-9: T = 1, no top-k/top-p (entry 2026-10-07 "정지 통계 H를 논문 정의 쪽으로 변경")
    assert {'visref_imgtag', 'tsr_answer_prompt'} <= set(METHODS)  # 2026-10-15 exploratory variant and its control
    assert {'st', 'tsr', 'visref_raw_rank'} <= set(METHODS)  # J-7: main VisRef = visref_raw_rank (entry 2026-10-07 "계획 수정(작성자 결정, s2 결과 전): fill 제외, 주 결과 VisRef = visref_raw_rank")
    assert THINK == ('tsr_think', 'visref_think', 'visref_hs_think') and set(THINK) <= set(METHODS)  # J-23
    assert budget(1961, 0.3) == 588  # m = floor(0.3 |V|)


def test_scorer_is_v4():
    assert SCORER_VERSION == 'v4_mc_empty_wrong'  # J-14, entry 2026-10-11 "scorer v4 + st_official 분모 수정" (user instruction)


def generation_runs():
    out = []
    for f in sorted(glob.glob(str(RUNS / '*' / 'run_config.json'))):
        name = pathlib.Path(f).parent.name
        if name.startswith('b_main') and 'rescore' not in name:
            out.append((name, json.load(open(f))))
    return out


def test_generation_runs_were_built_with_the_decided_caps_and_order():
    """Every generation run passed the decided caps 1,024 / 10,240 explicitly (J-10): until 2026-10-13 run.py's own
    defaults were the older 4,096 / 8,192 (entry 2026-10-07 "smoke2 결과와 본 실험 결정", replaced by entry 2026-10-07 "결정(작성자): s1 + DEER af 폴백, 정지 통계 현행 유지, 파일럿 없이 본 실험 시작"). Runs also use the seed-0 question order (J-19)."""
    runs = generation_runs()
    assert len(runs) >= 11  # b_main_s1..s9, b_main_fill, b_main_hs
    for name, c in runs:
        a = c['run_args']
        assert a[a.index('--think-cap') + 1] == '1024', name
        assert a[a.index('--total-think-cap') + 1] == '10240', name
        assert c['shuffle_seed'] == 0, name


def test_think_runs_start_from_the_stored_main_run_step1():
    """J-23: the think runs pass --step1-from with the store bundled, the extra cap 512 and the three methods."""
    for name, c in generation_runs():
        if not name.startswith('b_main_think'):
            continue
        a = c['run_args']
        assert a[a.index('--step1-from') + 1] == STORE and a[a.index('--think-extra') + 1] == '512', name
        assert a[a.index('--methods') + 1] == ','.join(THINK), name
        assert c['source_sha256'][STORE] == hashlib.sha256((REPRO / STORE).read_bytes()).hexdigest(), name


def test_step1_store_is_the_main_run_st_rows():
    """diag/step1_store_bmain.json = make_step1_store.build over the b_main_s1..s9 result files."""
    sys.path.insert(0, str(REPRO / 'diag'))
    import make_step1_store as M
    files = sorted(glob.glob(str(REPRO / M.RUNS)))
    doc = json.loads((REPRO / STORE).read_text())
    assert doc['pids'] == M.build(files) and doc['n'] == 1000
    assert doc['files_sha256'] == {str(pathlib.Path(f).relative_to(REPRO)): hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
                                   for f in files}


@pytest.mark.parametrize('pred, text, ok', [
    ('B', 'so the answer is (B).', True),
    ('B', '**B**', True),
    ('B', 'The answer is \\boxed{B}', True),
    ('7 cm', 'it is 7 cm long', True),
    ('A', 'Considering all of this, I pick (B).', False),
    ('B', '', False),
])
def test_llm_guard_2_multiple_choice(pred, text, ok):
    prob = {'question_type': 'multi_choice', 'answer_type': 'text', 'choices': ['3 cm', '7 cm', '9 cm']}
    letter = {'A': '3 cm', 'B': '7 cm'}.get(pred, pred)
    assert LX.grounded(letter, text, prob) == ok  # J-21: an extraction counts only if it is in the scored text


def test_llm_guard_2_numbers():
    prob = {'question_type': 'free_form', 'answer_type': 'integer', 'choices': None}
    assert LX.grounded('1200', 'total is 1,200', prob)
    assert LX.grounded('12', 'the result: 12.', prob)
    assert not LX.grounded('13', 'the result: 12.', prob)


def test_llm_fallback_only_for_rows_the_rules_cannot_parse():
    assert LX.needs_llm({'rule': 'last_line', 'status': 'ok'})       # J-14
    assert LX.needs_llm({'rule': 'boxed', 'extract_failed': True})
    assert not LX.needs_llm({'rule': 'boxed', 'status': 'ok'})
    assert not LX.needs_llm({'rule': 'empty', 'status': 'error'})
    assert LX.empty_text({'final_answer': '  \n'})                    # J-21 guard 1: such rows are not sent
