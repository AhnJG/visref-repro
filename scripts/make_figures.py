"""Draw figures/paired_differences.svg and figures/main_run_steps.svg (standard library only).

  python3 scripts/make_figures.py        # run from the repository root, after scripts/make_results.py

paired_differences.svg, from results/paired_differences.csv: one row per comparison, our paired difference with its
95% bootstrap CI (dot and whisker) under the paper's difference (diamond, VisRef Appendix Table 6).
main_run_steps.svg, from the main run records repro/kaggle/runs/b_main_s1..s9 (rows of ST, TSR and VisRef, 1,000
questions each): the number of thinking steps, how each step ended, and the injected tokens relative to m. No accuracy.
Both SVGs carry their own light and dark colours (prefers-color-scheme).
figures/method_overview.svg is drawn by hand and is not generated here.
"""
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'results' / 'paired_differences.csv'
OUT = ROOT / 'figures' / 'paired_differences.svg'
RUNS = ROOT / 'repro' / 'kaggle' / 'runs'
OUT_RUN = ROOT / 'figures' / 'main_run_steps.svg'
METHODS = [('st', 'ST'), ('tsr', 'TSR'), ('visref_raw_rank', 'VisRef')]

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
         text(24, 56, 'InternVL3.5-1B on MathVista testmini, n = 1,000 questions, main scoring', 'ink2', 12)]
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


# ---- figures/main_run_steps.svg: how the main run ran (counts from the recorded rows, no accuracy) ----------------
RW, RH = 1020, 476
PW, PY0 = 300, 128                      # panel width; top of the plot area
PX = [24, 360, 696]                     # left edge of the three panels
STYLE_RUN = STYLE + """
  .m-st{fill:#c3c2b7} .m-tsr{fill:#8a887f} .m-vis{fill:#2a78d6}
  .e-close{fill:#52514e} .e-cut{fill:#9a9890} .e-eos{fill:#e1e0d9}
  .tick{stroke:#9a9890}
  @media (prefers-color-scheme: dark){
    .m-st{fill:#4a4944} .m-tsr{fill:#8d8b83} .m-vis{fill:#3987e5}
    .e-close{fill:#c3c2b7} .e-cut{fill:#6b6a64} .e-eos{fill:#34342f}
    .tick{stroke:#6b6a64}
  }
"""
K_BINS = [('1', 1, 1), ('2', 2, 2), ('3–4', 3, 4), ('5–9', 5, 9), ('10', 10, 10)]
END_KEYS = [('close', 'closed by the model', 'e-close'), ('cap', 'cut at the token limit', 'e-cut'),
            ('eos', 'end of sequence', 'e-eos')]


def main_run_rows():
    """Rows of the main run per method (the early 'visref_raw' rows of session 1 are not part of the main run)."""
    out = {m: [] for m, _ in METHODS}
    for f in sorted(RUNS.glob('b_main_s[1-9]/output/results/results_*.jsonl')):
        for ln in f.open(encoding='utf-8'):
            if ln.strip():
                r = json.loads(ln)
                if r['method'] in out:
                    out[r['method']].append(r)
    return out


def rect(x, y, w, h, cls, extra=''):
    return f'<rect class="{cls}" x="{round(x, 1)}" y="{round(y, 1)}" width="{round(w, 1)}" height="{round(h, 1)}"{extra}/>'


def fmt(n):
    return f'{n:,}'


def panel_k(data, px):
    """(a) number of questions by number of thinking steps k, one bar per method (horizontal, grouped)."""
    o = [text(px, PY0 - 22, '(a) Thinking steps per question', 'ink', 13, weight='600'),
         text(px, PY0 - 6, 'number of questions, by number of steps k', 'muted', 11)]
    left, scale, bh, gap, row = px + 46, 200 / 1000, 11, 2, 48
    for i, (label, lo, hi) in enumerate(K_BINS):
        y = PY0 + 10 + i * row
        o.append(text(left - 8, y + 20, f'k = {label}', 'ink2', 11, 'end'))
        for j, (m, _) in enumerate(METHODS):
            n = sum(1 for r in data[m] if lo <= r['k'] <= hi)
            by = y + j * (bh + gap)
            o.append(rect(left, by, max(n * scale, 0.5), bh, ('m-st', 'm-tsr', 'm-vis')[j]))
            o.append(text(left + n * scale + 5, by + 9.5, fmt(n), 'ink2 num', 10))
    o.append(f'<line class="base" x1="{left}" y1="{PY0}" x2="{left}" y2="{PY0 + 10 + len(K_BINS) * row - 8}" '
             f'stroke-width="1"/>')
    ly = PY0 + 10 + len(K_BINS) * row + 8                       # legend
    for j, (_, name) in enumerate(METHODS):
        o += [rect(left + j * 70, ly, 10, 10, ('m-st', 'm-tsr', 'm-vis')[j]), text(left + j * 70 + 15, ly + 9, name, 'ink2', 11)]
    return o


def panel_end(data, px):
    """(b) how each thinking step ended, per method (one 100% bar per method, counts under the bar)."""
    o = [text(px, PY0 - 22, '(b) How each thinking step ended', 'ink', 13, weight='600'),
         text(px, PY0 - 6, 'all steps of a method; the bar is 100%', 'muted', 11)]
    for i, (m, name) in enumerate(METHODS):
        steps = Counter(s['think_end'] for r in data[m] for s in r['steps'])
        total = sum(steps.values())
        y = PY0 + 14 + i * 76
        o.append(text(px, y, f'{name}: {fmt(total)} steps', 'ink', 12, weight='600'))
        x = px
        for key, _, cls in END_KEYS:
            w = PW * steps[key] / total
            o.append(rect(x, y + 8, w, 26, cls, ' stroke="#9a9890" stroke-width="0.5"'))
            x += w
        o.append(text(px, y + 50, f"closed {fmt(steps['close'])} · cut {fmt(steps['cap'])} · "
                                  f"end of sequence {fmt(steps['eos'])}", 'ink2 num', 10.5))
    ly = PY0 + 14 + 3 * 76 - 10                                 # legend
    for i, (_, lab, cls) in enumerate(END_KEYS):
        o += [rect(px, ly + i * 18, 12, 12, cls, ' stroke="#9a9890" stroke-width="0.5"'),
              text(px + 18, ly + i * 18 + 10, lab, 'ink2', 11)]
    return o


def panel_inject(data, px):
    """(c) VisRef steps by injected tokens / m (ten bins of width 0.1, and exactly m)."""
    q = [(s['n_inject'], s['m']) for r in data['visref_raw_rank'] for s in r['steps']]
    ratio = [a / b for a, b in q]
    bins = [sum(1 for v in ratio if 0.1 * i <= v < 0.1 * (i + 1)) for i in range(10)] + [sum(1 for v in ratio if v >= 1)]
    med = statistics.median(ratio)
    o = [text(px, PY0 - 22, '(c) Injected image tokens per VisRef step', 'ink', 13, weight='600'),
         text(px, PY0 - 6, f'number of steps ({fmt(len(q))} in all), injected tokens / m', 'muted', 11)]
    left, bot, top = px + 34, PY0 + 250, PY0 + 6
    ymax = 1000
    bw = (PX[2] + PW - left) / len(bins)
    for t in (0, 250, 500, 750, 1000):
        y = bot - (bot - top) * t / ymax
        o.append(f'<line class="{"base" if t == 0 else "grid"}" x1="{left}" y1="{round(y, 1)}" x2="{px + PW}" '
                 f'y2="{round(y, 1)}" stroke-width="1"/>')
        o.append(text(left - 6, y + 4, fmt(t), 'muted num', 10, 'end'))
    for i, n in enumerate(bins):
        h = (bot - top) * n / ymax
        x = left + i * bw
        o.append(rect(x + 2, bot - h, bw - 4, h, 'm-vis'))
        o.append(text(x + bw / 2, bot - h - 4, fmt(n), 'ink2 num', 9.5, 'middle'))
    for v, lab in ((0, '0'), (5, '0.5'), (10, '1.0')):
        o.append(text(left + v * bw, bot + 16, lab, 'muted num', 10, 'middle'))
    o.append(text(left + 10.5 * bw, bot + 30, '= m', 'muted', 10, 'middle'))
    mx = left + med * 10 * bw
    o += [f'<line class="ink2-l" x1="{round(mx, 1)}" y1="{top}" x2="{round(mx, 1)}" y2="{bot}" stroke-width="1.5" '
          f'stroke-dasharray="4 3"/>',
          text(mx + 6, top + 10, f'median {med:.3f}', 'ink2 num', 11)]
    o.append(text(left + (PX[2] + PW - left) / 2, bot + 48, 'injected tokens / m', 'muted', 11, 'middle'))
    return o


def main_run():
    data = main_run_rows()
    n = {m: len(data[m]) for m, _ in METHODS}
    assert set(n.values()) == {1000}, n
    desc = ('Left: number of questions by number of thinking steps for ST, TSR and VisRef. Middle: how the thinking '
            'steps ended (closed by the model, cut at the token limit, end of sequence). Right: VisRef steps by the '
            'number of injected image tokens relative to m.')
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {RW} {RH}" width="{RW}" height="{RH}" role="img" '
         f'aria-labelledby="t d">',
         '<title id="t">How the main run ran: thinking steps, how steps ended, injected image tokens</title>',
         f'<desc id="d">{esc(desc)}</desc>', f'<style>{STYLE_RUN}  .ink2-l{{stroke:#52514e}} '
         f'@media (prefers-color-scheme: dark){{.ink2-l{{stroke:#c3c2b7}}}}</style>',
         f'<rect class="surface" x="0" y="0" width="{RW}" height="{RH}" rx="10"/>',
         f'<rect class="frame" x="0.5" y="0.5" width="{RW - 1}" height="{RH - 1}" rx="10"/>',
         text(24, 34, 'How the main run ran', 'ink', 16, weight='600'),
         text(24, 56, 'InternVL3.5-1B on MathVista testmini, 1,000 questions per method; counts from the recorded '
                      'rows, no accuracy', 'ink2', 12)]
    o += panel_k(data, PX[0]) + panel_end(data, PX[1]) + panel_inject(data, PX[2])
    o.append(text(24, RH - 36, 'k = 1: the model answered after the first thinking step; k = 10 is the limit on steps. '
                               'Each thinking step is limited to 1,024 tokens.', 'muted', 11))
    o.append(text(24, RH - 20, 'm = 30% of the image tokens of the question. ST has one step per question; TSR and VisRef '
                               'share the first step.', 'muted', 11))
    o.append('</svg>')
    OUT_RUN.parent.mkdir(exist_ok=True)
    OUT_RUN.write_text('\n'.join(o) + '\n', encoding='utf-8')
    print('wrote', OUT_RUN.relative_to(ROOT))


if __name__ == '__main__':
    main()
    main_run()
