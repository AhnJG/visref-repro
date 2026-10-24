"""`stop_form` recomputed with free answers read without a box (exploratory, after the result; docs/EXPERIMENT_FLOW.md 2026-10-24
"stop_form 사후 탐색: 상자 없는 자유 답도 값으로 읽기"). The rule below was written and committed before it was run on the stored rows.

The pre-registered rule (diag/stop_form.py free_record) made a free answer valid only when it ended with EOS, its last \\boxed{}
closed with a letter or digit inside, and a value could be read. About 30% of the model's free answers end without a box ("Answer:
10"), so the "mostly stop" verdict could not be reached (r_hi). Here a free answer is valid when
  - it ended with EOS (finish 'eos'), and
  - stop_form.value_of(text, row) reads a value: the rule scorer v4 prediction (the main run's reader of final answers: \\boxed,
    "answer is", the last line ...), with the same multiple-choice guard (a choice letter, or exactly one of the choices).
No box is required. Everything else (F_val lo / hi, the decision function, the 126 analysis questions) is analyze_stop_form's.
Not a verdict: the result is reported as an exploration after the result, next to the pre-registered "undecided".
Check built in: every answer that was valid under the old rule keeps its value (the old rule is a subset of this one).

  cd repro && python3 diag/stop_form_unboxed.py 'kaggle/runs/stop_form_main/output/results/stop_form_*.jsonl' \\
      --data MATHVISTA_TESTMINI.json --out diag/stop_form_unboxed.json
  MATHVISTA_TESTMINI.json: {pid: {question_type, answer_type, choices, precision, answer, ...}} (the official testmini json);
  each row's answer and question type are checked against the stored row.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_blank_think import load  # noqa: E402
from analyze_closed_multistep import rel  # noqa: E402
from analyze_stop_form import SELF, cont, decide, describe, estimates  # noqa: E402
from stop_form import value_of  # noqa: E402

KEYS = ('question_type', 'answer_type', 'choices', 'precision', 'answer')


def reread(a, row):
    """The free answer a with value / valid under the box-free rule; old_valid kept for the check."""
    value, sc = value_of(a['text'], row)
    valid = a['finish'] == 'eos' and value is not None
    return {**a, 'old_valid': a['valid'], 'old_value': a['value'], 'valid': valid, 'value': value if valid else None,
            'correct': int(valid and bool(sc['correct']))}


def recompute(R, ids, data):
    """{'coverage', 'check', 'decision', 'descriptive', 'counts'} on the analysis list, with free answers re-read."""
    want = [str(p) for p in ids['analysis']]
    rows, mismatch, changed_old = [], [], 0
    for p in want:
        r = R[p]
        d = data[p]
        if str(d['answer']) != str(r['answer_gt']) or d['question_type'] != r['question_type']:
            mismatch.append(p)
        row = {k: d.get(k) for k in KEYS}
        free = [reread(a, row) for a in r['free']]
        changed_old += sum(a['old_valid'] and a['value'] != a['old_value'] for a in free)
        rows.append({**r, 'free': free})
    selfr = [r for r in rows if r['stratum'] in SELF]
    E = [estimates(r) for r in selfr]
    fr = [a for r in selfr for a in r['free']]
    return {'n_analysis': len(want), 'data_mismatch': mismatch,
            'check_old_valid_value_changed': changed_old,
            'counts': {'n_free': len(fr), 'valid_old': sum(a['old_valid'] for a in fr), 'valid_new': sum(a['valid'] for a in fr),
                       'invalid_new_cap': sum(a['finish'] == 'cap' for a in fr),
                       'invalid_new_no_value': sum(a['finish'] == 'eos' and not a['valid'] for a in fr),
                       'questions_all_free_invalid': sum(all(not a['valid'] for a in r['free']) for r in selfr)},
            'decision_exploratory': decide(E),
            'n_continue': {k: sum(cont(e[k]) for e in E) for k in ('stored', 'F_val_lo', 'F_val_hi')},
            'descriptive': describe(selfr)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='+')
    ap.add_argument('--ids', default=str(HERE / 'stop_form_analysis_ids.json'))
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and not a.force:
        sys.exit(f'{out} exists (use --force)')
    R, files = load(a.results)
    data = {str(k): v for k, v in json.loads(Path(a.data).read_text()).items()}
    ids = json.loads(Path(a.ids).read_text())
    want = [str(p) for p in ids['analysis']]
    if any(p not in R or R[p].get('status') != 'ok' for p in want):
        sys.exit('an analysis question has no ok row')
    res = {'files': [rel(f) for f in files], 'ids_file': rel(a.ids), **recompute(R, ids, data)}
    if res['data_mismatch'] or res['check_old_valid_value_changed']:
        sys.exit(f"check failed: {res['data_mismatch'][:5]} / {res['check_old_valid_value_changed']}")
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'counts': res['counts'], 'sentence': res['decision_exploratory']['sentence']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
