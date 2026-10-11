"""MathVista scoring: rule-based extraction + official normalization/comparison.

The official MathVista pipeline extracts answers with GPT-4; here extraction is
rule-based (no external API) and applied identically to every method; rows the
rules cannot parse are re-extracted offline with an LLM (llm_extract.py). Matching
uses the official `normalize_extracted_answer` / `safe_equal` logic
(references/sources/mathvista_official_r4/evaluation/calculate_score.py).

`extract` also returns the character span of the answer (recorded as a diagnostic).
"""
import re

# v1: same-line remainder only ('**Answer:**\nC' -> ''); v2: markdown label line -> next line;
# v3: 'answer is:' colon consumed, trailing '.' and emphasis stripped in any order ('**B**.' -> 'B')
# v4 (2026-10-11): extraction identical to v3; an empty multiple-choice extraction is no prediction (wrong) instead of
#     the Levenshtein-closest option -- the official MathVista `--ignore_empty_extractions` mode
#     (calculate_score.py L28, L44, L121). v3 kept reproducible: score(..., version=SCORER_V3).
SCORER_V3 = 'v3_md_colon_emphasis'
SCORER_VERSION = 'v4_mc_empty_wrong'
_VERSIONS = (SCORER_V3, SCORER_VERSION)
NUM = r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-+]?\.\d+'


def _boxed(text):
    i = text.rfind('\\boxed{')
    if i < 0:
        return None
    s = i + len('\\boxed{')
    depth = 1
    for j in range(s, len(text)):
        if text[j] == '{':
            depth += 1
        elif text[j] == '}':
            depth -= 1
            if depth == 0:
                return s, j
    return None


def _clean(s):
    s = re.sub(r'\\text\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\(?:mathrm|textbf|mathbf)\{([^{}]*)\}', r'\1', s)
    s = s.replace('$', '').replace('\\%', '%').replace('\\,', '').replace('\\!', '')
    prev = None
    while s != prev:
        prev, s = s, s.strip().strip('*_').rstrip('.')
    return s.strip()


def _frac(s):
    m = re.search(r'\\d?frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}', s)
    if m and float(m[2]) != 0:
        return m.start(), m.end(), str(float(m[1]) / float(m[2]))
    return None


def _candidate(text):
    """Locate the answer region: last \\boxed{}, else 'answer is/:' line, else last line."""
    b = _boxed(text)
    if b:
        return b[0], b[1], 'boxed'
    ms = list(re.finditer(r'(?i)(?:final\s+answer|answer)\s*(?:is\s*[:：]?|[:：])\s*', text))
    if ms:
        s = ms[-1].end()
        e = text.find('\n', s)
        e = len(text) if e < 0 else e
        if text[s:e].strip() and not re.sub(r'[\s*_]', '', text[s:e]):  # '**Answer:**  \nC': label only
            nxt = re.search(r'[^\n]*\S[^\n]*', text[e:])
            if nxt:
                s, e = e + nxt.start(), e + nxt.end()
        if text[s:e].strip():
            return s, e, 'answer_phrase'
    lines = [m for m in re.finditer(r'[^\n]+', text) if m.group().strip()]
    if lines:
        return lines[-1].start(), lines[-1].end(), 'last_line'
    return None


def extract(text, row):
    """Return (extracted string or None, (start, end) span in text or None, rule)."""
    cand = _candidate(text)
    if cand is None:
        return None, None, 'empty'
    s, e, rule = cand
    seg = text[s:e]
    if row['question_type'] == 'multi_choice':
        c = _clean(seg)
        m = re.match(r'^\(?([A-Z])\)?(?:[.:)\s]|$)', c)
        if m:
            off = seg.find(m[1])
            return m[1], (s + off, s + off + 1), rule + ':letter'
        m = list(re.finditer(r'\(([A-Z])\)', seg))
        if m:
            return m[-1][1], (s + m[-1].start(1), s + m[-1].end(1)), rule + ':paren_letter'
        return c, (s, e), rule + ':text'
    if row['answer_type'] in ('integer', 'float'):
        f = _frac(seg)
        if f:
            return f[2], (s + f[0], s + f[1]), rule + ':frac'
        nums = list(re.finditer(NUM, seg))
        if not nums and rule != 'boxed':  # fall back to the last number in the whole response
            nums = list(re.finditer(NUM, text))
            s = 0
        if nums:
            n = nums[-1]
            return n.group().replace(',', ''), (s + n.start(), s + n.end()), rule + ':number'
        return None, (s, e), rule + ':no_number'
    if row['answer_type'] == 'list':
        m = list(re.finditer(r'\[[^\[\]]*\]', seg))
        if m:
            return m[-1].group(), (s + m[-1].start(), s + m[-1].end()), rule + ':list'
    return _clean(seg), (s, e), rule + ':text'


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize(extraction, row, ignore_empty=False):
    """Official MathVista normalize_extracted_answer; ignore_empty = its `ignore_empty_extractions` flag."""
    qt, at, choices = row['question_type'], row['answer_type'], row.get('choices')
    if qt == 'multi_choice':
        extraction = '' if extraction is None else str(extraction).strip()
        if ignore_empty and not extraction:
            return None
        letter = re.findall(r'\(([a-zA-Z])\)', extraction)
        if letter:
            extraction = letter[0].upper()
        seq = [chr(ord('A') + i) for i in range(len(choices))]
        if extraction in seq:
            return choices[seq.index(extraction)]
        d = [_lev(extraction, c) for c in choices]
        return choices[d.index(min(d))]
    try:
        if at == 'integer':
            return str(int(float(extraction)))
        if at == 'float':
            return str(round(float(extraction), int(row['precision'])))
        if at == 'list':
            return str(extraction)
    except Exception:
        return None
    return extraction


def score(answer_text, row, version=SCORER_VERSION):
    """version=SCORER_V3 reproduces the earlier scorer (empty MC extraction -> closest option)."""
    assert version in _VERSIONS, version
    ext, span, rule = extract(answer_text, row)
    ignore_empty = version != SCORER_V3
    pred = (normalize(ext, row, ignore_empty) if ext is not None or row['question_type'] == 'multi_choice'
            else None)
    return {'extracted': ext, 'prediction': pred, 'rule': rule, 'span': span,
            'extract_failed': ext is None or str(ext).strip() == '',
            'correct': int(pred is not None and pred == row['answer']),
            'scorer_version': version}
