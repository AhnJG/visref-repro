"""diag/analyze_probe_final.py: normalization and category rules on hand-made cases (no data file needed)."""
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO / 'diag'))
import analyze_probe_final as A  # noqa: E402

YN = ['Yes', 'No']


def cat(p, f, qt='free_form', ch=None):
    return A.compare(p, f, qt, ch)[0]


def test_normalize():
    assert A.normalize('  {\\text{Yes}} ') == 'yes'
    assert A.normalize('\\text{Yes}.') == 'yes'
    assert A.normalize('{yes}.') == '{yes}'  # braces are stripped before the period is dropped (rule order)
    assert A.normalize('\\text{1,234}') == '1234'
    assert A.normalize('1,234,567') == '1234567'
    assert A.normalize('{a}{b}') == '{a}{b}'  # not one surrounding pair
    assert A.normalize('3 . 5') == '3.5'
    assert A.normalize('1,23') == '1,23'  # not a thousands group


def test_exact_and_numeric():
    assert cat('4', '4') == 'exact'
    assert cat('1.00', '1') == 'numeric_equal'
    assert cat('1,000', '1000.0') == 'numeric_equal'
    assert cat('0.012', '0.0') == 'different'
    assert cat('inf', 'inf') == 'exact'
    assert cat('nan', '1') == 'different'
    assert cat('40%', '40') == 'different'  # no unit / percent handling


def test_multiple_choice_mapping():
    assert cat('B', 'No', 'multi_choice', YN) == 'mc_mapped'
    assert cat('(b)', '{\\text{No}}', 'multi_choice', YN) == 'mc_mapped'
    assert cat('A', 'No', 'multi_choice', YN) == 'different'
    assert cat('B', 'B', 'multi_choice', YN) == 'exact'  # exact comes first
    assert cat('C', 'No', 'multi_choice', YN) == 'undecidable'  # letter beyond the choices
    assert cat('Nope', 'No', 'multi_choice', YN) == 'undecidable'  # no fuzzy matching
    assert cat('B', 'No', 'free_form') == 'different'  # mapping only for multiple choice


def test_multiple_choice_ambiguity():
    ch = ['(c)', '(d)', '(a)', '(b)', '(e)']  # letter 'b' and the text '(b)' point at different choices
    assert cat('B', '(b)', 'multi_choice', ch) == 'undecidable'
    assert cat('A', 'A', 'multi_choice', ch) == 'exact'
    dup = ['20', '30', '20']
    assert cat('A', '20', 'multi_choice', dup) == 'undecidable'
    same = ['A', 'B']  # letter and text point at the same choice: fine
    assert cat('B', 'A', 'multi_choice', same) == 'different'


def test_classify_row():
    probs = {'1': {'question_type': 'multi_choice', 'choices': YN, 'answer': 'No', 'answer_type': 'text'}}
    base = {'pid': '1', 'prediction': 'No', 'extract_failed': False,
            'steps': [{'probe_valid': True, 'probe_inner': 'B'}]}
    assert A.classify_row(base, probs)[0] == 'used'
    assert A.classify_row(dict(base, extract_failed=True), probs) == ('excluded', 'extract_failed')
    assert A.classify_row(dict(base, steps=[{'probe_valid': False, 'probe_inner': 'B'}]), probs)[1] == 'probe_invalid_empty'
    assert A.classify_row(dict(base, steps=[{'probe_valid': True, 'probe_inner': ' '}]), probs)[1] == 'probe_invalid_empty'
    assert A.classify_row(dict(base, steps=[{'k': 1}]), probs)[1] == 'missing_field'
    assert A.classify_row(dict(base, pid='9'), probs)[1] == 'missing_field'
    assert A.classify_row(dict(base, prediction=None), probs)[1] == 'final_empty'
    llm = dict(base, prediction='Yes', llm_used=True, llm_prediction='No')
    assert A.classify_row(llm, probs)[1][0] == 'mc_mapped'  # graded final = llm_prediction
