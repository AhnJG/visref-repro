"""LLM fallback answer extraction for rows the rule-based scorer could not parse (MathVista official style).

The official MathVista pipeline (references/sources/mathvista_official_r4/evaluation/extract_answer.py) sends
create_test_prompt(demo_prompt, query, response) to an LLM; VLMEvalKit tries rules first and falls back to an LLM.
Here the rule-based scorer runs first (visref/scoring.py) and only rows whose rule is 'last_line' / 'empty' or whose
extraction failed are sent to an LLM extractor. Ground truth is never put in the requests.

Guards (2026-10-10, set after the pre-check and before the final extraction; EXPERIMENT_FLOW):
  1. a row whose scored text is empty is not sent and stays as scored by the rule (nothing to extract from);
  2. an LLM extraction counts only if it is grounded in the scored text: the option text or a letter form
     ("(B)", "B)", "B.", "**B**", "\\boxed{B}", "answer is B") for multiple choice, the same number for numeric
     answers, the same string otherwise. Ungrounded extractions are recorded but the rule result is kept.

  python3 repro/llm_extract.py prepare RESULTS.jsonl... --out requests.jsonl [--control 20] [--items items.jsonl]
  python3 repro/llm_extract.py merge RESULTS.jsonl... --responses responses.jsonl --out rescored.jsonl

`query` is rebuilt with the official hint templates (evaluation_extra/build_query.py, zero-shot 'solution' hints,
hint first as in the HF `query` field) from data/raw/mathvista_testmini.json, which has no `query` field.
"""
import argparse
import ast
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from visref.scoring import score  # noqa: E402

EXT = ROOT / 'references/sources/mathvista_official_r4/evaluation/prompts/ext_ans.py'
DATA = ROOT / 'data/raw/mathvista_testmini.json'
FALLBACK_RULES = ('last_line', 'empty')


def demo_prompt():
    return re.search(r'demo_prompt = """(.*?)"""', EXT.read_text(), re.S).group(1)


def create_test_prompt(demo, query, response):  # verbatim logic of the official create_test_prompt
    demo = demo.strip()
    test_prompt = f"{query}\n\n{response}"
    return f"{demo}\n\n{test_prompt}\n\nExtracted answer: "


def _val(v):
    if v in (None, 'None', ''):
        return None
    try:
        return ast.literal_eval(v) if isinstance(v, str) and v[:1] in '[(' else v
    except Exception:
        return v


def problems():
    out = {}
    for pid, r in json.loads(DATA.read_text()).items():
        choices, unit, prec = _val(r['choices']), _val(r['unit']), _val(r['precision'])
        prec = float(prec) if prec is not None else None
        qt, at = r['question_type'], r['answer_type']
        if qt == 'multi_choice':
            hint = 'Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.'
        elif at == 'integer':
            hint = 'Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.'
        elif at == 'float' and prec == 1:
            hint = 'Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.'
        elif at == 'float' and prec == 2:
            hint = 'Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.'
        else:
            hint = 'Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.'
        q = f"Question: {r['question']}" + (f" (Unit: {unit})" if unit else '')
        if choices:
            q += '\nChoices:\n' + '\n'.join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
        out[str(pid)] = {'query': f'{hint}\n{q}', 'question_type': qt, 'answer_type': at,
                         'choices': [str(c) for c in choices] if choices else None, 'answer': r['answer'],
                         'precision': prec, 'unit': unit}
    return out


def load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    return rows


def rid(r):
    return f"{r['pid']}|{r['method']}"


def needs_llm(r):
    return r.get('status') != 'error' and (r['rule'].split(':')[0] in FALLBACK_RULES or r.get('extract_failed'))


def empty_text(r):
    return not str(r.get('final_answer') or '').strip()


NUM = re.compile(r'-?\d+(?:,\d{3})*(?:\.\d+)?')


def grounded(pred, text, prob):
    """Guard 2: does the normalized LLM extraction actually occur in the scored response text?"""
    if pred is None or not text.strip():
        return False
    flat = ' '.join(text.split()).lower()
    if prob['question_type'] == 'multi_choice':
        choices = prob['choices'] or []
        if str(pred) in choices:
            if ' '.join(str(pred).split()).lower() in flat:
                return True
            L = chr(65 + choices.index(str(pred)))
            pats = [rf'\(\s*{L}\s*\)', rf'(?<![A-Za-z0-9]){L}\s*[\).:]', rf'\\boxed\{{\s*\(?{L}\)?\s*\}}',
                    rf'\*\*\s*\(?{L}\)?\s*\*\*', rf'(?:answer|option|choice)\s*(?:is)?\s*:?\s*\(?{L}(?![A-Za-z])']
            return any(re.search(p_, text, re.I if p_.startswith('(?:') else 0) for p_ in pats)
        return False
    if prob['answer_type'] in ('integer', 'float'):
        try:
            v = float(pred)
        except (TypeError, ValueError):
            return False
        return any(abs(float(x.replace(',', '')) - v) < 1e-9 for x in NUM.findall(text))
    return ' '.join(str(pred).split()).lower() in flat


def prepare(a):
    probs, demo = problems(), demo_prompt()
    rows = load(a.results)
    pick = [r for r in rows if needs_llm(r) and not empty_text(r)]  # guard 1
    n_empty = sum(needs_llm(r) and empty_text(r) for r in rows)
    ctrl = [r for r in rows if r.get('status') != 'error' and not needs_llm(r)]
    random.Random(0).shuffle(ctrl)
    with open(a.out, 'w') as f:
        for r, is_ctrl in [(r, False) for r in pick] + [(r, True) for r in ctrl[:a.control]]:
            f.write(json.dumps({'id': rid(r), 'control': is_ctrl,
                                'prompt': create_test_prompt(demo, probs[r['pid']]['query'], r['final_answer'])},
                               ensure_ascii=False) + '\n')
    if a.items:  # for an agent that reads items itself: question and response split, no control flag, examples apart
        sel = pick + ctrl[:a.control]
        with open(a.items, 'w') as f:
            for r in sel:
                f.write(json.dumps({'id': rid(r), 'question': probs[r['pid']]['query'], 'response': r['final_answer']},
                                   ensure_ascii=False) + '\n')
        Path(a.items).with_name('extract_examples.txt').write_text(demo.strip() + '\n')
    print(json.dumps({'fallback_rows': len(pick), 'skipped_empty_text': n_empty, 'control_rows': min(a.control, len(ctrl)),
                      'total_rows': len(rows)}))


def merge(a):
    probs = problems()
    resp = {x['id']: x['extraction'] for x in load([a.responses])}
    rows, out = load(a.results), []
    stats = {'fallback': 0, 'skipped_empty_text': 0, 'ungrounded_rejected': 0, 'fixed': 0, 'broke': 0,
             'control': 0, 'control_agree': 0, 'control_grounded': 0}
    for r in rows:
        r = dict(r)
        e = resp.get(rid(r))
        if e is not None:
            s = score(e if e.strip() else '', probs[r['pid']])
            g = grounded(s['prediction'], str(r.get('final_answer') or ''), probs[r['pid']])
            if needs_llm(r) and empty_text(r):  # guard 1 (also for responses produced before the guard existed)
                stats['skipped_empty_text'] += 1
                r.update(llm_extraction=e, llm_prediction=s['prediction'], llm_grounded=False, llm_used=False)
            elif needs_llm(r):
                stats['fallback'] += 1
                if not e.strip():  # no extraction returned: nothing to use, the rule result stays
                    stats['empty_extraction'] = stats.get('empty_extraction', 0) + 1
                    r.update(llm_extraction=e, llm_prediction=None, llm_grounded=False, llm_used=False)
                elif not g:  # guard 2
                    stats['ungrounded_rejected'] += 1
                    r.update(llm_extraction=e, llm_prediction=s['prediction'], llm_grounded=False, llm_used=False)
                else:
                    stats['fixed'] += int(s['correct'] and not r['correct'])
                    stats['broke'] += int(r['correct'] and not s['correct'])
                    r.update(llm_extraction=e, llm_prediction=s['prediction'], llm_grounded=True, llm_used=True,
                             correct_rule=r['correct'], correct=s['correct'], rule=r['rule'] + '+llm')
            else:  # control: rule result kept, agreement recorded
                stats['control'] += 1
                stats['control_agree'] += int(s['prediction'] == r.get('prediction'))
                stats['control_grounded'] += int(g)
                r.update(llm_extraction=e, llm_prediction=s['prediction'], llm_grounded=g, llm_used=False)
        out.append(r)
    with open(a.out, 'w') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    by = {}
    for r in out:
        m = by.setdefault(r['method'], [0, 0, 0])
        m[0] += 1
        m[1] += r.get('correct_rule', r['correct'])
        m[2] += r['correct']
    stats['accuracy_rule_vs_rule_llm'] = {k: [v[0], round(100 * v[1] / v[0], 1), round(100 * v[2] / v[0], 1)]
                                          for k, v in by.items()}
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prepare', 'merge'])
    ap.add_argument('results', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--responses')
    ap.add_argument('--control', type=int, default=0)
    ap.add_argument('--items', help='prepare: also write question/response items for an agent extractor')
    a = ap.parse_args()
    prepare(a) if a.cmd == 'prepare' else merge(a)
