"""Draw figures/paired_differences.svg from results/paired_differences.csv (standard library only).

  python3 scripts/make_figures.py        # run from the repository root, after scripts/make_results.py

One row per comparison: our paired difference with its 95% bootstrap CI (dot and whisker) under the paper's
difference (diamond, VisRef Appendix Table 6). The SVG carries its own light and dark colours (prefers-color-scheme).
figures/method_overview.svg is drawn by hand and is not generated here.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'results' / 'paired_differences.csv'
OUT = ROOT / 'figures' / 'paired_differences.svg'

W, H = 840, 386
X0, X1 = 176, 476            # plot area (x)
LO, HI = -3.0, 7.0           # domain in percentage points
TICKS = [-2, 0, 2, 4, 6]
TOP, ROW = 142, 52           # centre of the first row, row pitch
TX = 500                     # x of the text column
MINUS, DOT, SECT = '−', '·', '§'
ROLE = {'main': 'main comparison', 'supplementary': 'supplementary'}
SHORT = {'effect size not reproduced (same direction, different size)': 'effect size not reproduced, same direction'}
STYLE = """
  .surface{fill:#fcfcfb} .ink{fill:#0b0b0b} .ink2{fill:#52514e} .muted{fill:#6f6d68}
  .grid{stroke:#e1e0d9} .base{stroke:#9a9890} .frame{stroke:rgba(11,11,11,0.10);fill:none}
  .ours{fill:#2a78d6} .ours-l{stroke:#2a78d6} .paper{fill:#eb6834} .ring{stroke:#fcfcfb}
  @media (prefers-color-scheme: dark){
    .surface{fill:#1a1a19} .ink{fill:#f0efec} .ink2{fill:#c3c2b7} .muted{fill:#a19f97}
    .grid{stroke:#2c2c2a} .base{stroke:#6b6a64} .frame{stroke:rgba(255,255,255,0.10)}
    .ours{fill:#3987e5} .ours-l{stroke:#3987e5} .paper{fill:#d95926} .ring{stroke:#1a1a19}
  }
  text{font-family:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
  .num{font-variant-numeric:tabular-nums}
"""


def sx(v):
    return round(X0 + (v - LO) / (HI - LO) * (X1 - X0), 1)


def pp(v):
    return f'{v:+.1f}'.replace('-', MINUS)


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def text(x, y, s, cls, size=12, anchor='start', weight=None):
    w = f' font-weight="{weight}"' if weight else ''
    return f'<text x="{x}" y="{y}" class="{cls}" font-size="{size}" text-anchor="{anchor}"{w}>{esc(s)}</text>'


def diamond(x, y, r=6.5):
    return f'{x},{round(y - r, 1)} {round(x + r, 1)},{y} {x},{round(y + r, 1)} {round(x - r, 1)},{y}'


def line(x1, y1, x2, y2, cls='ours-l'):
    return (f'<line class="{cls}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke-width="2" '
            f'stroke-linecap="round"/>')


def main():
    rows = list(csv.DictReader(SRC.open()))
    plot_top, plot_bot = TOP - ROW // 2, TOP - ROW // 2 + len(rows) * ROW
    desc = '; '.join(f"{r['comparison']}: ours {pp(float(r['diff_pp']))} pp (95% CI {pp(float(r['ci95_lo']))} to "
                     f"{pp(float(r['ci95_hi']))}), paper {pp(float(r['paper_pp']))} pp, {r['classification']}"
                     for r in rows)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'aria-labelledby="t d">',
         '<title id="t">Paired accuracy differences on MathVista testmini: our reproduction vs. the VisRef paper</title>',
         f'<desc id="d">{esc(desc)}</desc>', f'<style>{STYLE}</style>',
         f'<rect class="surface" x="0" y="0" width="{W}" height="{H}" rx="10"/>',
         f'<rect class="frame" x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="10"/>',
         text(24, 34, 'Paired accuracy differences: our reproduction vs. the paper', 'ink', 16, weight='600'),
         text(24, 56, 'InternVL3.5-1B on MathVista testmini, n = 1,000 questions, pre-specified reading (a)', 'ink2', 12)]
    ly = 84                                                     # legend
    o += [line(24, ly, 48, ly), f'<circle class="ours ring" cx="36" cy="{ly}" r="5" stroke-width="2"/>',
          text(56, ly + 4, 'Ours: paired difference with 95% bootstrap CI', 'ink2', 12),
          f'<polygon class="paper ring" points="{diamond(366, ly)}" stroke-width="2"/>',
          text(378, ly + 4, 'Paper: difference in Appendix Table 6', 'ink2', 12)]
    for t in TICKS:                                             # grid, zero line, axis
        o.append(f'<line class="{"base" if t == 0 else "grid"}" x1="{sx(t)}" y1="{plot_top}" x2="{sx(t)}" '
                 f'y2="{plot_bot}" stroke-width="1"/>')
        o.append(text(sx(t), plot_bot + 18, '0' if t == 0 else pp(t).replace('.0', ''), 'muted num', 11, 'middle'))
    o.append(text((X0 + X1) / 2, plot_bot + 38, 'difference in accuracy (percentage points)', 'muted', 11, 'middle'))
    o.append(text(TX, plot_top - 6, 'values (pp) and classification', 'muted', 11))
    for i, r in enumerate(rows):
        y = TOP + i * ROW
        d, lo, hi, paper = (float(r[k]) for k in ('diff_pp', 'ci95_lo', 'ci95_hi', 'paper_pp'))
        yo = y + 7                                              # ours just below the row centre, paper just above
        o += [text(24, y - 2, r['comparison'].replace(' - ', f' {MINUS} '), 'ink', 13, weight='600'),
              text(24, y + 14, ROLE.get(r['kind'], r['kind']), 'muted', 11),
              f'<polygon class="paper ring" points="{diamond(sx(paper), y - 9)}" stroke-width="2"/>',
              line(sx(lo), yo, sx(hi), yo), line(sx(lo), yo - 5, sx(lo), yo + 5), line(sx(hi), yo - 5, sx(hi), yo + 5),
              f'<circle class="ours ring" cx="{sx(d)}" cy="{yo}" r="5.5" stroke-width="2"/>',
              text(TX, y - 2, f'ours {pp(d)} ({pp(lo)} to {pp(hi)}) {DOT} paper {pp(paper)}', 'ink num', 12),
              text(TX, y + 14, SHORT.get(r['classification'], r['classification']), 'ink2', 12)]
    o.append(text(24, H - 36, 'Consistent: the CI excludes 0 and contains the paper value. Not reproduced: the CI '
                              'excludes the paper value.', 'muted', 11))
    o.append(text(24, H - 20, f'The classification rule was fixed after an interim look at 759 questions (REPORT.md '
                              f'{SECT}2.4); the paper reports no per-question results.', 'muted', 11))
    o.append('</svg>')
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text('\n'.join(o) + '\n', encoding='utf-8')
    print('wrote', OUT.relative_to(ROOT))


if __name__ == '__main__':
    main()
