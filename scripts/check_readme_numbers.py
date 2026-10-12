"""Check that README.md and README.ko.md quote the current numbers in results/ (standard library only).

  python3 scripts/check_readme_numbers.py      # run from the repository root; part of `make check`

The two READMEs are maintained by hand. This check fails when a value in results/ (accuracies, paired differences,
CIs, p-values, follow-up results) is missing from either README, for example after a correction was applied to only
one of them. Numbers are formatted as the READMEs write them: signed with "+" or the Unicode minus "−".
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MINUS = '−'


def signed(v, nd):
    return f'{float(v):+.{nd}f}'.replace('-', MINUS)


def plain(v, nd):
    return f'{float(v):.{nd}f}'


def rows(name):
    return list(csv.DictReader((ROOT / 'results' / name).open()))


def expected():
    want = set()
    for r in rows('accuracy.csv'):
        want |= {plain(r['ours_pct'], 1), plain(r['paper_pct'], 1)}
    for r in rows('paired_differences.csv'):
        want |= {signed(r['diff_pp'], 1), signed(r['ci95_lo'], 1), signed(r['ci95_hi'], 1),
                 plain(r['p_mcnemar'], 3), signed(r['paper_pp'], 1)}
        if r['p_holm']:
            want.add(plain(r['p_holm'], 3))
    for r in rows('followups.csv'):
        if r['experiment'].startswith('visref_fill') and r['comparison'] in ('VisRef-fill - TSR', 'VisRef-fill - VisRef'):
            want |= {signed(r['ci95_lo'], 2), signed(r['ci95_hi'], 2)}
        if r['experiment'].startswith('visref_hs primary'):
            want |= {signed(r['diff_pp'], 2), signed(r['ci95_lo'], 2), signed(r['ci95_hi'], 2)}
        if 'co-primary' in r['experiment']:
            want |= {signed(r['diff_pp'], 2), signed(r['ci95_lo'], 2), signed(r['ci95_hi'], 2), plain(r['p_holm'], 3)}
    return sorted(want)


def main():
    want, bad = expected(), 0
    for name in ('README.md', 'README.ko.md'):
        text = (ROOT / name).read_text(encoding='utf-8')
        missing = [w for w in want if w not in text]
        bad += bool(missing)
        print(f"{'ok      ' if not missing else 'MISSING '} {name}" + (f": {missing}" if missing else ''))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
