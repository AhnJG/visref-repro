"""CPU tests for diag/stop_form_unboxed.py (no model)."""
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO / 'diag'))
import stop_form_unboxed as su  # noqa: E402

NUM = {'question_type': 'free_form', 'answer_type': 'integer', 'choices': None, 'precision': None, 'answer': '10'}
MC = {'question_type': 'multi_choice', 'answer_type': 'text', 'choices': ['10', '20', '30'], 'precision': None, 'answer': '20'}


def ans(text, finish='eos', valid=False, value=None):
    return {'text': text, 'finish': finish, 'valid': valid, 'value': value, 'boxed': '\\boxed' in text, 'n_tokens': 5,
            'ent_sum': 0.0, 'ent_mean': 0.0, 'correct': 0}


def test_unboxed_answers_are_read_and_the_guards_stay():
    assert su.reread(ans('\n\nThe age gap is 10 years. \n\nAnswer: 10'), NUM)['value'] == '10'
    assert su.reread(ans('\n\nAnswer: 10', finish='cap'), NUM)['valid'] is False          # must end with EOS
    assert su.reread(ans('I cannot determine the answer.'), MC)['value'] is None          # 'I' is not a choice letter
    assert su.reread(ans('The answer is B.'), MC)['value'] == '20'
    r = su.reread(ans('\\boxed{10}', valid=True, value='10'), NUM)                       # old valid keeps its value
    assert r['valid'] and r['value'] == r['old_value'] == '10' and r['correct'] == 1


def free16(values):
    return [ans(f'Answer: {v}') if v is not None else ans('', finish='cap') for v in values]


def test_recompute_counts_and_decides_on_the_self_rows():
    R = {}
    for i in range(20):
        vals = ['10'] * 16 if i < 18 else ['10'] * 8 + ['20'] * 8
        R[str(i)] = {'pid': str(i), 'status': 'ok', 'stratum': 'close', 'question_type': 'free_form', 'answer_gt': '10',
                     'stored_H': 0.0, 'short': [{'value': '10', 'H': 0.0, 'valid': True}] * 16, 'free': free16(vals)}
    data = {p: NUM for p in R}
    res = su.recompute(R, {'analysis': list(R)}, data)
    assert res['counts']['valid_old'] == 0 and res['counts']['valid_new'] == 320 and res['check_old_valid_value_changed'] == 0
    assert res['n_continue']['F_val_lo'] == 2 and res['decision_exploratory']['n'] == 20
    assert res['data_mismatch'] == []
