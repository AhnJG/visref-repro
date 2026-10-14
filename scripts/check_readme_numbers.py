"""Check that README.md, README.ko.md and REPORT.md put the numbers of results/ in the right table cells.

  python3 scripts/check_readme_numbers.py            # run from the repository root; part of `make check`
  python3 scripts/check_readme_numbers.py --write    # rewrite the generated tables in place from results/

Tables that carry results are wrapped in markers:

  <!-- check:KIND -->
  | ... the table ... |
  <!-- /check -->

Two kinds of check (standard library only):
- Generated tables (accuracy, paired, accuracy_report, paired_report): the table between the markers must equal,
  character for character, the table this script builds from results/*.csv. A swapped or stale value fails.
- The "experiments at a glance" table (glance): each row A1-A4 must show its own result in the result column
  (header "Result" or "결과"), so a value cannot sit in another row's cell.
Each document must contain the kinds listed in REQUIRED. Numbers use "+" or the Unicode minus "−".
"""
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MINUS = '−'
BLOCK = re.compile(r'(<!-- check:(\w+) -->\n)(.*?)(\n<!-- /check -->)', re.S)
REQUIRED = {
    'README.md': {'accuracy', 'paired', 'glance'},
    'README.ko.md': {'accuracy', 'paired', 'glance'},
    'REPORT.md': {'accuracy_report', 'paired_report', 'glance'},
}
NAMES = {'VisRef - ST': 'VisRef − ST', 'VisRef - TSR': 'VisRef − TSR', 'TSR - ST': 'TSR − ST'}
VERDICT_KO = {
    'effect size not reproduced (same direction, different size)': '방향은 같으나 효과 크기 재현 안 됨',
    'effect size not reproduced': '논문 크기의 효과 재현 안 됨',
    'consistent with the paper value': '논문 수치와 양립',
}


def signed(v, nd):
    return f'{float(v):+.{nd}f}'.replace('-', MINUS)


def plain(v, nd):
    return f'{float(v):.{nd}f}'


def rows(name):
    return list(csv.DictReader((ROOT / 'results' / name).open(encoding='utf-8')))


def role(r, ko):
    if ko:
        return '주 비교' if r['kind'] == 'main' else '보조'
    return 'main' if r['kind'] == 'main' else 'supplementary'


def table(header, body):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(r) + ' |' for r in body]
    return '\n'.join(lines)


def render(kind, ko):
    acc = {r['method']: r for r in rows('accuracy.csv')}
    pairs = rows('paired_differences.csv')
    methods = ('ST', 'TSR', 'VisRef')
    if kind == 'accuracy':
        return table(['정확도 (%)' if ko else 'accuracy (%)', *methods], [
            ['논문 (Table 6)' if ko else 'paper (Table 6)', *(plain(acc[m]['paper_pct'], 1) for m in methods)],
            ['**우리**' if ko else '**ours**', *(f"**{plain(acc[m]['ours_pct'], 1)}**" for m in methods)]])
    if kind == 'paired':
        head = (['비교', '우리, %p (95% 신뢰구간)', '논문, %p', '판정'] if ko
                else ['comparison', 'ours, pp (95% CI)', 'paper, pp', 'verdict'])
        body = []
        for r in pairs:
            verdict = VERDICT_KO[r['classification']] if ko else r['classification']
            body.append([f"{NAMES[r['comparison']]} ({role(r, ko)})",
                         f"{signed(r['diff_pp'], 1)} ({signed(r['ci95_lo'], 1)}, {signed(r['ci95_hi'], 1)})",
                         signed(r['paper_pp'], 1), verdict])
        return table(head, body)
    if kind == 'accuracy_report':
        return table(['Method', 'Ours %', 'Paper %', 'Ours − paper'], [
            [m, plain(acc[m]['ours_pct'], 1), plain(acc[m]['paper_pct'], 1), signed(acc[m]['ours_minus_paper_pp'], 1)]
            for m in methods])
    if kind == 'paired_report':
        head = ['Comparison', 'Ours (pp)', '95% CI', 'Only first / only second correct', 'McNemar p', 'Holm p',
                'Paper (pp)', 'Classification']
        body = [[f"{NAMES[r['comparison']]} ({'main' if r['kind'] == 'main' else 'suppl.'})", signed(r['diff_pp'], 1),
                 f"{signed(r['ci95_lo'], 1)} to {signed(r['ci95_hi'], 1)}",
                 f"{r['only_first_correct']} / {r['only_second_correct']}", plain(r['p_mcnemar'], 3),
                 plain(r['p_holm'], 3) if r['p_holm'] else 'n/a', signed(r['paper_pp'], 1), r['classification']]
                for r in pairs]
        return table(head, body)
    raise KeyError(kind)


def glance_expected():
    """Per row, the values that must appear in its result cell."""
    pair = {r['comparison']: r for r in rows('paired_differences.csv')}
    fu = rows('followups.csv')

    def one(prefix, comparison):
        hits = [r for r in fu if r['experiment'].startswith(prefix) and r['comparison'] == comparison]
        if len(hits) != 1:
            raise SystemExit(f'followups.csv: expected one row for {prefix!r} / {comparison!r}, found {len(hits)}')
        return hits[0]

    co = [r for r in fu if 'co-primary' in r['experiment']]
    if len(co) != 2:
        raise SystemExit(f'followups.csv: expected two step1_branch co-primary rows, found {len(co)}')
    return {
        'A1': [signed(pair['VisRef - TSR']['diff_pp'], 1)],
        'A2': [signed(one('visref_fill', 'VisRef-fill - TSR')['diff_pp'], 1).lstrip('+')],
        'A3': [signed(r['diff_pp'], 2) for r in co],
        'A4': [signed([r for r in fu if r['experiment'].startswith('visref_hs primary')][0]['diff_pp'], 2)],
    }


def cells(line):
    return [c.strip() for c in line.strip().strip('|').split('|')]


def check_glance(text, want):
    lines = [l for l in text.splitlines() if l.strip().startswith('|')]
    head = cells(lines[0])
    col = next((i for i, h in enumerate(head) if h.startswith(('Result', '결과'))), None)
    if col is None:
        return ['glance: no "Result"/"결과" column']
    errors, seen = [], set()
    for line in lines[2:]:
        c = cells(line)
        key = c[0].strip('*')
        if key in want:
            seen.add(key)
            missing = [v for v in want[key] if v not in c[col]]
            if missing:
                errors.append(f'glance {key}: result cell {c[col]!r} lacks {missing}')
    errors += [f'glance: row {k} missing' for k in want if k not in seen]
    return errors


def main():
    write = '--write' in sys.argv[1:]
    want_glance, bad = glance_expected(), 0
    for name, required in REQUIRED.items():
        path = ROOT / name
        text = path.read_text(encoding='utf-8')
        ko = name.endswith('.ko.md')
        errors, found = [], set()

        def fix(m):
            kind, body = m.group(2), m.group(3)
            found.add(kind)
            if kind == 'glance':
                errors.extend(check_glance(body, want_glance))
                return m.group(0)
            expected = render(kind, ko)
            if body != expected:
                errors.append(f'{kind}: table differs from results/ (expected below)\n{expected}')
                return m.group(1) + expected + m.group(4)
            return m.group(0)

        new = BLOCK.sub(fix, text)
        errors += [f'missing table: <!-- check:{k} -->' for k in sorted(required - found)]
        if write and new != text:
            path.write_text(new, encoding='utf-8')
            errors = [e for e in errors if 'differs' not in e]
        bad += bool(errors)
        print(f"{'ok      ' if not errors else 'FAIL    '} {name}" + ''.join(f'\n  - {e}' for e in errors))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
