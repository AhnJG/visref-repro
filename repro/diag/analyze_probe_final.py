"""Probe answer at the stopping step vs the graded final answer (2026-10-19). Descriptive only: not used for any decision.

Question: when TSR / VisRef (visref_raw_rank) stopped because the probe entropy was low (stop_reason == 'entropy'), is the
LAST step's probe answer (steps[-1].probe_inner, the text inside \\boxed{...} of the short answer sampled at T=1 after
'</think>\\n\\n\\boxed') the same answer as the graded final answer?

Data: llm/final/merged.jsonl (final grading, 3,000 rows). MathVista question_type / answer_type / choices / answer come
from data/raw/mathvista_testmini.json (sha256-pinned in references/SOURCES.json; python3 scripts/fetch_sources.py), read
through llm_extract.problems() (the same file and the same choice conversion the offline scorer uses).

Graded final answer (rules written BEFORE counting):
  The scorer (visref/scoring.py score(), L161) grades `prediction` (correct = pred == answer), not `extracted`
  (`extracted` is the string before normalization). Rows whose extraction was recovered by the LLM fallback
  (llm_extract.py merge: llm_used True, r.update(... correct=s['correct']) at L177-L178) keep the rule's `prediction`
  but their `correct` comes from `llm_prediction`. So: graded final = llm_prediction if llm_used else prediction.
  (Checked in the output: graded final == answer_gt exactly when `correct` == 1.)

Rows: method in {tsr, visref_raw_rank} (st is listed for reference; it has no probe steps) with stop_reason == 'entropy'.
Exclusions (each counted separately; a row gets the first reason that applies, in this order):
  extract_failed       -- row['extract_failed'] is true
  missing_field        -- steps empty, or last step has no 'probe_valid'/'probe_inner' key, or no graded final answer
                          field at all (`prediction` absent), or the pid is not in the MathVista file
  probe_invalid_empty  -- last step probe_valid is false, or probe_inner is not a string / empty after strip
  final_empty          -- the graded final answer is None or empty after strip (reported apart from the three above)

Normalization (both strings, in this order):
  1. strip whitespace and surrounding braces (balanced '{...}' pairs, repeatedly)
  2. unwrap \\text{...} (also \\textbf/\\mathrm/\\mathbf are NOT touched; only \\text, innermost first)
  3. remove all whitespace
  4. drop one trailing period
  5. lowercase
  6. remove thousands commas (a comma between a digit and exactly three digits, e.g. 1,234,567 -> 1234567)

Categories (mutually exclusive, tested in this order):
  1 exact          -- normalized strings are equal
  2 numeric_equal  -- both parse as plain decimal floats (no nan/inf, no units/percent) and |a-b| <= 1e-9*max(1,|a|)
  3 mc_mapped      -- multi_choice question only: each side is mapped to a choice index ONLY if it is exactly a choice
                      letter (A..last choice, optional parentheses, after lowercasing) or exactly equals a normalized
                      choice text; both mapped and same index -> 3. A side that matches more than one index (letter
                      and text collide, or duplicate choice texts) is not mapped.
  4 different      -- the comparison was possible and the answers differ: non-multiple-choice not in 1/2, or
                      multiple-choice with both sides mapped to different indices
  5 undecidable    -- multiple-choice question where at least one side cannot be mapped exactly (no fuzzy / near-string
                      matching is ever used)

Probe correctness (for the 2x2 table): the probe answer is graded by the same scorer as the final answer,
visref.scoring.score('\\\\boxed{<probe_inner>}', problem) ; final correctness is merged.jsonl `correct`.

  cd repro && python3 diag/analyze_probe_final.py [--merged llm/final/merged.jsonl] [--out diag/analyze_probe_final.json]
"""
import argparse
import collections
import json
import math
import random
import re
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO))

DATE = '2026-10-19'
METHODS = ('tsr', 'visref_raw_rank')
REFERENCE = ('st',)
CATS = ('exact', 'numeric_equal', 'mc_mapped', 'different', 'undecidable')
EXCL = ('extract_failed', 'missing_field', 'probe_invalid_empty', 'final_empty')
_NUM = re.compile(r'^[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?$')
_THOUSANDS = re.compile(r'(?<=\d),(?=\d{3}(?!\d))')
_TEXT = re.compile(r'\\text\{([^{}]*)\}')


def normalize(s):
    s = s.strip()
    while len(s) >= 2 and s[0] == '{' and s[-1] == '}' and _balanced(s[1:-1]):
        s = s[1:-1].strip()
    prev = None
    while prev != s:
        prev, s = s, _TEXT.sub(r'\1', s)
    s = re.sub(r'\s+', '', s)
    if s.endswith('.'):
        s = s[:-1]
    s = s.lower()
    return _THOUSANDS.sub('', s)


def _balanced(s):
    d = 0
    for ch in s:
        d += ch == '{'
        d -= ch == '}'
        if d < 0:
            return False
    return d == 0


def to_float(s):
    if not _NUM.match(s):
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def choice_index(norm, norm_choices):
    """Index of the choice that `norm` is exactly (a letter A.. or the choice text), or None if none / ambiguous."""
    cand = set()
    m = re.fullmatch(r'\(?([a-z])\)?', norm)
    if m and ord(m[1]) - 97 < len(norm_choices):
        cand.add(ord(m[1]) - 97)
    cand.update(i for i, c in enumerate(norm_choices) if c == norm)
    return next(iter(cand)) if len(cand) == 1 else None


def compare(probe, final, question_type, choices):
    """Return (category, normalized_probe, normalized_final); category is one of CATS."""
    a, b = normalize(probe), normalize(final)
    if a == b:
        return 'exact', a, b
    x, y = to_float(a), to_float(b)
    if x is not None and y is not None and abs(x - y) <= 1e-9 * max(1.0, abs(x)):
        return 'numeric_equal', a, b
    if question_type == 'multi_choice':
        nc = [normalize(str(c)) for c in (choices or [])]
        i, j = choice_index(a, nc), choice_index(b, nc)
        if i is None or j is None:
            return 'undecidable', a, b
        return ('mc_mapped' if i == j else 'different'), a, b
    return 'different', a, b


def graded_final(r):
    """The string the scorer graded: llm_prediction for LLM-recovered rows, else prediction (see docstring)."""
    return r.get('llm_prediction') if r.get('llm_used') else r.get('prediction')


def classify_row(r, probs):
    """Return ('excluded', reason) or ('used', (category, probe, final, np, nf))."""
    if r.get('extract_failed') is True:
        return 'excluded', 'extract_failed'
    steps = r.get('steps')
    if (not steps or 'prediction' not in r or 'probe_valid' not in steps[-1] or 'probe_inner' not in steps[-1]
            or str(r.get('pid')) not in probs):
        return 'excluded', 'missing_field'
    last = steps[-1]
    inner = last['probe_inner']
    if last['probe_valid'] is not True or not isinstance(inner, str) or not inner.strip():
        return 'excluded', 'probe_invalid_empty'
    fin = graded_final(r)
    if not isinstance(fin, str) or not fin.strip():
        return 'excluded', 'final_empty'
    p = probs[str(r['pid'])]
    cat, np_, nf = compare(inner, fin, p['question_type'], p['choices'])
    return 'used', (cat, inner, fin, np_, nf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--merged', default=str(REPRO / 'llm/final/merged.jsonl'))
    ap.add_argument('--out', default=str(REPRO / 'diag/analyze_probe_final.json'))
    a = ap.parse_args()
    from llm_extract import problems
    from visref.scoring import score
    probs = problems()
    rows = [json.loads(l) for l in open(a.merged) if l.strip()]
    out = {'date': DATE, 'descriptive_only': True, 'merged': Path(a.merged).resolve().relative_to(REPRO).as_posix() if Path(a.merged).resolve().is_relative_to(REPRO) else Path(a.merged).name, 'n_rows_file': len(rows),
           'graded_field': 'llm_prediction if llm_used else prediction (visref/scoring.py L161 grades `prediction`)',
           'question_metadata': 'data/raw/mathvista_testmini.json via llm_extract.problems()', 'methods': {}}
    bad_consistency = 0
    for r in rows:  # graded final == answer exactly when correct == 1
        g = graded_final(r)
        bad_consistency += int((g is not None and g == probs[r['pid']]['answer']) != bool(r['correct']))
    out['graded_final_vs_correct_mismatch_rows'] = bad_consistency
    rng = random.Random(0)
    for m in METHODS + REFERENCE:
        ent = [r for r in rows if r['method'] == m and r.get('stop_reason') == 'entropy']
        info = {'rows_method': sum(r['method'] == m for r in rows), 'rows_entropy_stop': len(ent),
                'rows_with_probe_steps': sum(any('probe_inner' in s for s in r.get('steps') or []) for r in rows
                                             if r['method'] == m)}
        excl, cats = collections.Counter(), collections.Counter()
        used, ex = [], collections.defaultdict(list)
        by_qt = collections.defaultdict(collections.Counter)
        t2 = collections.Counter()
        t2_same = collections.defaultdict(collections.Counter)
        for r in ent:
            kind, v = classify_row(r, probs)
            if kind == 'excluded':
                excl[v] += 1
                continue
            cat, inner, fin, np_, nf = v
            p = probs[str(r['pid'])]
            cats[cat] += 1
            by_qt[p['question_type']][cat] += 1
            used.append(r)
            pc = score('\\boxed{' + inner + '}', p)['correct']
            key = f'probe_{"right" if pc else "wrong"}__final_{"right" if r["correct"] else "wrong"}'
            t2[key] += 1
            t2_same['same(1-3)' if cat in CATS[:3] else 'not_same(4-5)'][key] += 1
            if cat in ('different', 'undecidable'):
                ex[cat].append({'pid': r['pid'], 'question_type': p['question_type'], 'answer_type': p['answer_type'],
                                'choices': p['choices'], 'answer_gt': p['answer'], 'probe_inner': inner,
                                'final': fin, 'probe_norm': np_, 'final_norm': nf, 'probe_correct': pc,
                                'final_correct': r['correct']})
        info.update({'excluded': {k: excl.get(k, 0) for k in EXCL}, 'n_excluded': sum(excl.values()),
                     'n_used': len(used), 'counts': {k: cats.get(k, 0) for k in CATS},
                     'n_same_1to3': sum(cats[k] for k in CATS[:3]),
                     'by_question_type': {q: {k: c.get(k, 0) for k in CATS} for q, c in by_qt.items()},
                     'table_2x2_probe_vs_final_correct': dict(t2),
                     'table_2x2_split_by_same': {k: dict(v) for k, v in t2_same.items()},
                     'examples': {c: rng.sample(ex[c], min(10, len(ex[c]))) for c in ('different', 'undecidable')},
                     'examples_note': 'seeded random sample (random.Random(0)), up to 10 per category'})
        out['methods'][m] = info
    json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)
    for m, i in out['methods'].items():
        print(m, i['rows_entropy_stop'], 'used', i['n_used'], 'excl', i['excluded'], i['counts'], 'same', i['n_same_1to3'])
        print('  2x2', i['table_2x2_probe_vs_final_correct'])
    print('mismatch graded-final vs correct:', bad_consistency)


if __name__ == '__main__':
    main()
