"""Build the b_main report page (self-contained HTML + JSON numbers) from result JSONL files.

  python3 repro/report.py RESULTS.jsonl... --order IDS.json --label "b_main s1" \
      --out repro/reports/b_main_report.html [--llm RESCORED.jsonl]

Uses summarize.py's loading, complete-prefix and paired bootstrap so the page and summarize.py agree.
--llm: output of `llm_extract.py merge` (rows with correct_rule / correct after LLM extraction).
--rescore-data: mathvista_testmini.json; re-applies the current rule scorer (visref/scoring.py, same call as
  rescore.py) to every row's saved final_answer before aggregating, so sessions recorded under older scorer
  versions are counted under one version. Also enables the answer-handling sensitivity table (a)/(b)/(c).
"""
import argparse
import html
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import summarize as S
from visref.scoring import SCORER_VERSION, score

AF_PROMPT = '\n**Final Answer**\n\\boxed'  # run.py af_prompt (DEER answer forcing, rule 2)

METHODS = ('st', 'tsr', 'visref_raw_rank')
NAME = {'st': 'ST', 'tsr': 'TSR', 'visref_raw_rank': 'VisRef'}
SERIES = {'st': 's1', 'tsr': 's2', 'visref_raw_rank': 's3'}
PAPER = {'st': 46.1, 'tsr': 48.5, 'visref_raw_rank': 52.0}
PAIRS = [('tsr', 'st'), ('visref_raw_rank', 'st'), ('visref_raw_rank', 'tsr')]
END_NAME = {'close': '자연 종료 (</think>)', 'cap': '상한 1024에서 잘림', 'eos': 'EOS로 끝남'}


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def inner_think_diag(rs):
    def cell(xs):
        k = sum(bool(r.get('final_y_start')) for r in xs)
        return {'inner': k, 'n': len(xs), 'rate': k / len(xs) if xs else None}
    k1 = [r for r in rs if r['k'] == 1]
    return {'all': cell(rs), 'k1_close': cell([r for r in k1 if r['steps'][-1]['think_end'] == 'close']),
            'k1_cap': cell([r for r in k1 if r['steps'][-1]['think_end'] == 'cap'])}


def rescore_rows(rows, data):
    """Overwrite rule-scoring fields in place with the current scorer; keep the recorded value."""
    changed = 0
    for (p, m), r in rows.items():
        n = score(r['final_answer'], data[str(p)])
        r['correct_recorded'] = r['correct']
        changed += n['correct'] != r['correct']
        for k in ('correct', 'extracted', 'prediction', 'rule', 'extract_failed'):
            r[k] = n[k]
    return changed


def sensitivity(rs, data):
    """(a) as scored, (b) af rows re-scored on the text before the af prompt (no answer rules 1 and 2,
    approx.), (c) af rows counted wrong (lower bound), strict: only answers extracted from \\boxed."""
    a = b = c = strict = n_af = af_a = af_b = 0
    for r in rs:
        a += r['correct']
        strict += r['correct'] * r['rule'].startswith('boxed')
        fa = r['final_answer']
        i = fa.rfind(AF_PROMPT) if r.get('final_af') else -1
        if i < 0:
            b += r['correct']; c += r['correct']
            continue
        n_af += 1; af_a += r['correct']
        cb = score(fa[:i], data[str(r['pid'])])['correct']
        b += cb; af_b += cb
    n = max(1, len(rs))
    return {'a': 100 * a / n, 'b': 100 * b / n, 'c': 100 * c / n, 'strict_boxed': 100 * strict / n,
            'n_af': n_af, 'af_correct_a': af_a, 'af_correct_b': af_b}


def compute(rows, order, shards, llm_rows=None, data=None):
    pids = sorted(S.complete_prefix(rows, METHODS, order, shards), key=int) if order else \
        sorted({p for p, _ in rows if all((p, m) in rows for m in METHODS)}, key=int)
    llm = {}
    if llm_rows:
        llm = {(r['pid'], r['method']): r for r in llm_rows}
    out = {'n': len(pids), 'methods': {}, 'pairs': {}, 'step1': {}, 'k_hist': {}, 'health': {}}
    for m in METHODS:
        rs = [rows[(p, m)] for p in pids]
        k = sum(r['correct'] for r in rs)
        lo, hi = wilson(k, len(rs))
        rec = {'n': len(rs), 'correct': k, 'acc': 100 * k / max(1, len(rs)), 'ci': [lo, hi], 'paper': PAPER[m]}
        if llm:
            kl = sum(llm[(p, m)]['correct'] for p in pids if (p, m) in llm)
            rec['acc_llm'] = 100 * kl / max(1, len(rs))
            rec['ci_llm'] = list(wilson(kl, len(rs)))
        steps = [s for r in rs for s in r['steps']]
        # Alg. 1: the DPP pick of every step is added to the trajectory before the stop check, so the last
        # step's pick is injected right before the final answer (also when k = 1)
        cont = [s for r in rs for s in r['steps'] if s.get('m')]
        rec.update(
            mean_k=statistics.mean(r['k'] for r in rs) if rs else 0,
            n_continued=sum(r['k'] > 1 for r in rs),
            stop=dict(Counter(r['stop_reason'] for r in rs)),
            mean_think=statistics.mean(r['think_tokens'] for r in rs) if rs else 0,
            forced=sum(bool(r.get('final_forced_answer')) for r in rs),
            forced_eos=sum(bool(r.get('final_forced_valid')) for r in rs if r.get('final_forced_answer')),
            af=sum(bool(r.get('final_af')) for r in rs),
            af_valid=sum(bool(r.get('final_af_valid')) for r in rs),
            forced_unusable=sum(bool(r.get('final_forced_answer')) and not r.get('final_forced_valid')
                                and not r.get('final_af_valid') for r in rs),
            rules=dict(Counter(r['rule'].split(':')[0] for r in rs)),
            extract_fallback=sum(r['rule'].split(':')[0] in ('last_line', 'empty') or bool(r.get('extract_failed'))
                                 for r in rs),
            errors=sum(r.get('status') == 'error' for r in rs),
            inner_think_end=sum(bool(r.get('final_y_start')) for r in rs),
            inject=[s['n_inject'] for s in cont],
            inject_m=[s['n_inject'] / s['m'] for s in cont],
            m_list=[s['m'] for s in cont],
            uz_list=[s['n_unique_z'] for s in cont],
            inject_lt_m=sum(s['n_inject'] < s['m'] for s in cont),
            inject_eq_uz=sum(s['n_inject'] == s['n_unique_z'] for s in cont),
            close_probe_H_ge=sum(s['think_end'] == 'close' and 'H' in s and s['H'] >= S.DELTA for s in steps),
            close_probes=sum(s['think_end'] == 'close' and 'H' in s for s in steps),
        )
        if data is not None:
            rec['sens'] = sensitivity(rs, data)
        out['methods'][m] = rec
        out['k_hist'][m] = dict(Counter(r['k'] for r in rs))
    st_think = out['methods']['st']['mean_think'] or 1
    for m in METHODS:
        out['methods'][m]['think_ratio'] = out['methods'][m]['mean_think'] / st_think
    for a, b in PAIRS:
        pr = S.paired(rows, a, b, pids)
        out['pairs'][f'{a}-{b}'] = dict(pr, p_mcnemar=mcnemar_exact(pr['a_only'], pr['b_only']),
                                        paper=PAPER[a] - PAPER[b])
    # step 1 is shared by all methods (forked from ST), so its end type is a property of the question
    for e in ('close', 'cap', 'eos'):
        ps = [p for p in pids if rows[(p, 'st')]['steps'][0]['think_end'] == e]
        if ps:
            out['step1'][e] = {'n': len(ps), **{m: 100 * sum(rows[(p, m)]['correct'] for p in ps) / len(ps)
                                                 for m in METHODS},
                               'mean_k': {m: statistics.mean(rows[(p, m)]['k'] for p in ps) for m in METHODS}}
    k1 = [p for p in pids if rows[(p, 'tsr')]['k'] == 1]
    out['health'] = {
        'errors': sum(out['methods'][m]['errors'] for m in METHODS),
        'code_versions': dict(Counter(str(r.get('code_version')) for (p, m), r in rows.items() if m in METHODS)),
        # CRN: TSR stopped at k=1 answers from the same context and seed as ST; full text can still
        # diverge from fp16 batch noise, so the check is on the extracted answer and on correctness
        'crn_text': sum(rows[(p, 'tsr')]['final_answer'] == rows[(p, 'st')]['final_answer'] for p in k1),
        'crn_same': sum(str(rows[(p, 'tsr')].get('prediction')) == str(rows[(p, 'st')].get('prediction'))
                        for p in k1),
        'crn_correct': sum(rows[(p, 'tsr')]['correct'] == rows[(p, 'st')]['correct'] for p in k1),
        'crn_n': len(k1),
        'rows': sum(1 for (p, m) in rows if m in METHODS),
        'scorer': SCORER_VERSION if data is not None else 'recorded (per-row, may mix versions)',
        'rescored_changed': sum(r['correct'] != r['correct_recorded'] for (p, m), r in rows.items()
                                if m in METHODS and p in set(pids) and 'correct_recorded' in r),
        # answers that contain an inner </think> (final_y_start = index after the last one; 0/None = none),
        # overall and, for k == 1 rows, split by how the single (= last) step ended
        'inner_think': {m: inner_think_diag([rows[(p, m)] for p in pids]) for m in METHODS},
    }
    return out


# ---------------------------------------------------------------- rendering helpers
def esc(x):
    return html.escape(str(x))


def f1(x):
    return f'{x:.1f}'


def signed(x):
    return f'{x:+.1f}'


def pct(v, lo, hi):
    return 100 * (v - lo) / (hi - lo)


def ci_chart(items, lo, hi, step, zero=None, unit='%'):
    """Rows of (label, series, value, ci_lo, ci_hi, paper) drawn in HTML so text keeps its size at any width.
    One scale (pct) places grid lines, tick labels and marks."""
    ticks = []
    t = lo
    while t <= hi + 1e-9:
        ticks.append(t)
        t += step
    grid = ''.join(f'<span class="ci-grid" style="left:{pct(t, lo, hi):.2f}%"></span>' for t in ticks)
    if zero is not None:
        grid += f'<span class="ci-zero" style="left:{pct(zero, lo, hi):.2f}%"></span>'
    o = [f'<div class="ci" role="img" aria-label="점 추정치와 95% 신뢰구간">']
    for label, ser, v, a, b, paper in items:
        tip = f'{label}: {v:.1f}{unit} (95% CI {a:.1f}~{b:.1f})' + (f' · 논문 {paper:.1f}{unit}' if paper is not None else '')
        o.append(f'<div class="ci-row"><div class="ci-lab">{esc(label)}</div>'
                 f'<div class="ci-track tip" data-tip="{esc(tip)}">{grid}'
                 f'<span class="ci-line {ser}" style="left:{pct(a, lo, hi):.2f}%;width:{pct(b, lo, hi) - pct(a, lo, hi):.2f}%"></span>'
                 + (f'<span class="ci-paper" style="left:{pct(paper, lo, hi):.2f}%"></span>' if paper is not None else '')
                 + f'<span class="ci-dot {ser}" style="left:{pct(v, lo, hi):.2f}%"></span>'
                 f'<span class="ci-val" style="left:{pct(v, lo, hi):.2f}%">{v:.1f}</span></div></div>')
    o.append('<div class="ci-row ci-axis"><div></div><div class="ci-ticks">'
             + ''.join(f'<span style="left:{pct(t, lo, hi):.2f}%">{t:g}{unit}</span>' for t in ticks)
             + '</div></div></div>')
    return ''.join(o)


def k_chart(hist, n):
    """Grouped bars in HTML: share of questions by number of steps k (TSR, VisRef)."""
    ms = [m for m in METHODS if m != 'st']
    ymax = max([100 * c / max(1, n) for m in ms for c in hist[m].values()] + [10])
    ymax = math.ceil(ymax / 20) * 20
    ys = range(0, ymax + 1, 20)
    o = ['<div class="kc" role="img" aria-label="단계 수 분포"><div class="kc-plot"><div class="kc-y">'
         + ''.join(f'<span style="bottom:{100 * t / ymax:.2f}%">{t}%</span>' for t in ys)
         + '</div><div class="kc-area">'
         + ''.join(f'<span class="kc-gl" style="bottom:{100 * t / ymax:.2f}%"></span>' for t in ys)
         + '<div class="kc-cols">']
    for k in range(1, 11):
        o.append('<div class="kc-col">')
        for m in ms:
            c = hist[m].get(k, 0)
            v = 100 * c / max(1, n)
            o.append(f'<span class="kc-bar {SERIES[m]} tip" data-tip="{esc(NAME[m])} · k={k}: {c}문항 ({v:.1f}%)" '
                     f'style="height:{100 * v / ymax:.2f}%"></span>')
        o.append('</div>')
    o.append('</div></div></div><div class="kc-x"><span></span>'
             + ''.join(f'<span>{k}</span>' for k in range(1, 11))
             + '</div><div class="kc-xt">단계 수 k (K_max = 10)</div></div>')
    return ''.join(o)


def pill(ok, text):
    return f'<span class="pill {"ok" if ok else "bad"}">{"통과" if ok else "확인 필요"}</span> {esc(text)}'


# ---------------------------------------------------------------- page
CSS = """
/* Layout: one reading column; numbers sit in tables beside each chart, status strip on top. */
:root {
  --bg: #f6f7f9; --panel: #ffffff; --fg: #15191f; --fg2: #4a5361; --muted: #7c8594;
  --rule: #dfe3ea; --grid: #e8ebf0; --accent: #2a5bd7; --hover: rgba(42, 91, 215, .05);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a;
  --ok: #0ca30c; --bad: #d03b3b; --okbg: #e6f4e6; --badbg: #fbe9e9; --warnbg: #fff5dc; --warnfg: #7a5200;
  --display: "Hahmlet", "Noto Serif KR", "Apple SD Gothic Neo", serif;
  --body: "IBM Plex Sans KR", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #121519; --panel: #1a1e24; --fg: #eef1f5; --fg2: #b4bcc8; --muted: #8790a0;
  --rule: #2c323b; --grid: #262b33; --accent: #7aa2ff; --hover: rgba(122, 162, 255, .07);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70;
  --okbg: #183018; --badbg: #3a1c1c; --warnbg: #3a2f12; --warnfg: #f2cf7a; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #121519; --panel: #1a1e24; --fg: #eef1f5; --fg2: #b4bcc8; --muted: #8790a0;
  --rule: #2c323b; --grid: #262b33; --accent: #7aa2ff; --hover: rgba(122, 162, 255, .07);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70;
  --okbg: #183018; --badbg: #3a1c1c; --warnbg: #3a2f12; --warnfg: #f2cf7a; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font: 15px/1.6 var(--body); }
.wrap { max-width: 980px; margin: 0 auto; padding-inline: 16px; padding-block: 28px 56px; display: grid; gap: 28px; }
header { display: grid; gap: 8px; }
.eyebrow { font: 600 12px/1.4 var(--mono); letter-spacing: .06em; color: var(--muted); text-transform: uppercase; }
h1 { font: 700 30px/1.25 var(--display); margin: 0; text-wrap: balance; }
h2 { font: 700 20px/1.3 var(--display); margin: 0; text-wrap: balance; }
h3 { font: 600 15px/1.4 var(--body); margin: 0; color: var(--fg2); }
p { margin: 0; max-width: 68ch; color: var(--fg2); }
.meta { font: 13px/1.6 var(--body); color: var(--muted); max-width: 80ch; }
.banner { background: var(--warnbg); color: var(--warnfg); border-radius: 6px; padding: 10px 14px; font-size: 14px; }
section { display: grid; gap: 14px; }
.panel { background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 18px; display: grid; gap: 14px; min-width: 0; }
.status { display: flex; flex-wrap: wrap; gap: 8px 22px; font-size: 14px; }
.pill { font: 600 12px/1 var(--body); padding: 4px 8px; border-radius: 999px; }
.pill.ok { background: var(--okbg); color: var(--ok); }
.pill.bad { background: var(--badbg); color: var(--bad); }
.tablewrap { overflow-x: auto; min-width: 0; }
table { border-collapse: collapse; width: 100%; font-size: 14px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--rule); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead th { font: 600 12px/1.4 var(--body); color: var(--muted); letter-spacing: .03em; }
td.num, .num { font-family: var(--mono); }
.key { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 7px; vertical-align: 0; }
.key.s1 { background: var(--s1); } .key.s2 { background: var(--s2); } .key.s3 { background: var(--s3); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: var(--fg2); }
.s1 { --c: var(--s1); } .s2 { --c: var(--s2); } .s3 { --c: var(--s3); }
/* interval chart: HTML rows, one percentage scale for grid, ticks and marks */
.ci { display: grid; max-width: 760px; }
.ci-row { display: grid; grid-template-columns: 92px minmax(0, 1fr); align-items: stretch; height: 50px; }
.ci-lab { align-self: center; text-align: right; padding-right: 14px; font-weight: 600; font-size: 13px; white-space: nowrap; }
.ci-track { position: relative; margin-inline: 4px 22px; }
.ci-track:hover { background: var(--hover); }
.ci-grid { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--grid); }
.ci-zero { position: absolute; top: 0; bottom: 0; border-left: 1px dashed var(--fg2); }
.ci-line { position: absolute; top: 50%; height: 3px; margin-top: -1.5px; border-radius: 2px; background: var(--c); }
.ci-dot { position: absolute; top: 50%; width: 12px; height: 12px; margin: -6px 0 0 -6px; border-radius: 50%;
  background: var(--c); box-shadow: 0 0 0 2px var(--panel); }
.ci-paper { position: absolute; top: 50%; width: 9px; height: 9px; margin: -6px 0 0 -6px; background: var(--panel);
  border: 1.5px solid var(--fg2); transform: rotate(45deg); }
.ci-val { position: absolute; bottom: calc(50% + 9px); transform: translateX(-50%); font: 12px/1 var(--mono); color: var(--fg2);
  background: var(--panel); padding: 1px 3px; border-radius: 3px; }
.ci-axis { height: 24px; }
.ci-ticks { position: relative; margin-inline: 4px 22px; }
.ci-ticks span { position: absolute; top: 6px; transform: translateX(-50%); font-size: 12px; color: var(--muted); white-space: nowrap; }
/* step-count bars */
.kc { max-width: 760px; display: grid; gap: 4px; }
.kc-plot { display: grid; grid-template-columns: 40px minmax(0, 1fr); height: 200px; margin-top: 8px; }
.kc-y { position: relative; }
.kc-y span { position: absolute; right: 8px; transform: translateY(50%); font-size: 12px; color: var(--muted); }
.kc-area { position: relative; }
.kc-gl { position: absolute; left: 0; right: 0; height: 1px; background: var(--grid); }
.kc-cols { position: absolute; inset: 0; display: grid; grid-template-columns: repeat(10, minmax(0, 1fr)); }
.kc-col { display: flex; align-items: flex-end; justify-content: center; gap: 2px; }
.kc-col:hover { background: var(--hover); }
.kc-bar { position: relative; width: min(18px, 40%); border-radius: 3px 3px 0 0; background: var(--c); }
.kc-x, .kc-xt { font-size: 12px; color: var(--muted); }
.kc-x { display: grid; grid-template-columns: 40px repeat(10, minmax(0, 1fr)); text-align: center; }
.kc-xt { text-align: center; padding-left: 40px; }
/* hover tooltip */
.tip:hover::after { content: attr(data-tip); position: absolute; z-index: 2; bottom: calc(100% + 6px); left: 50%;
  transform: translateX(-50%); white-space: nowrap; background: var(--fg); color: var(--bg); font: 12px/1.4 var(--body);
  padding: 5px 8px; border-radius: 4px; pointer-events: none; }
.ci-track.tip:hover::after { bottom: auto; top: calc(100% - 4px); }
.diamond { display: inline-block; width: 9px; height: 9px; border: 1.5px solid var(--fg2); transform: rotate(45deg); margin-right: 8px; }
ul.notes { margin: 0; padding-left: 18px; color: var(--fg2); display: grid; gap: 6px; max-width: 75ch; }
code { font-family: var(--mono); font-size: .92em; }
"""


def page(d, label, generated, final=False, judgment=None):
    M = d['methods']
    n = d['n']
    h = d['health']
    forced_tot = sum(M[m]['forced'] for m in METHODS)
    unusable = sum(M[m]['forced_unusable'] for m in METHODS)
    ok_n = n >= 40
    decided = [k for k, p in d['pairs'].items() if p['ci95_pp'][0] > 0 or p['ci95_pp'][1] < 0]
    items = [(NAME[m], SERIES[m], M[m]['acc'], *M[m]['ci'], M[m]['paper']) for m in METHODS]
    lo = max(0, math.floor((min(min(i[3] for i in items), min(PAPER.values())) - 3) / 10) * 10)
    hi = min(100, math.ceil((max(max(i[4] for i in items), max(PAPER.values())) + 3) / 10) * 10)
    pairs = d['pairs']
    pitems = []
    for key, pr in pairs.items():
        a, b = key.split('-', 1)
        pitems.append((f'{NAME[a]}−{NAME[b]}', SERIES[a], pr['diff_pp'], pr['ci95_pp'][0], pr['ci95_pp'][1], pr['paper']))
    span = max(10, math.ceil(max(max(abs(p[3]), abs(p[4]), abs(p[5])) for p in pitems) / 5) * 5)
    llm_col = 'acc_llm' in M['st']

    acc_rows = ''.join(
        f'<tr><td><span class="key {SERIES[m]}"></span>{NAME[m]}</td><td class="num">{M[m]["correct"]}/{M[m]["n"]}</td>'
        f'<td class="num">{f1(M[m]["acc"])}</td><td class="num">{f1(M[m]["ci"][0])}–{f1(M[m]["ci"][1])}</td>'
        f'<td class="num">{f1(M[m]["acc_llm"]) if llm_col else "대기"}</td>'
        f'<td class="num">{M[m]["paper"]:.1f}</td><td class="num">{signed(M[m]["acc"] - M[m]["paper"])}</td></tr>'
        for m in METHODS)
    pair_rows = ''.join(
        f'<tr><td>{esc(lab)}</td><td class="num">{signed(pr["diff_pp"])}</td>'
        f'<td class="num">{signed(pr["ci95_pp"][0])} ~ {signed(pr["ci95_pp"][1])}</td>'
        f'<td class="num">{pr["a_only"]} / {pr["b_only"]}</td><td class="num">{pr["p_mcnemar"]:.2f}</td>'
        f'<td class="num">{signed(pr["paper"])}</td></tr>'
        for (lab, *_), pr in zip(pitems, pairs.values()))
    s1_rows = ''.join(
        f'<tr><td>{esc(END_NAME[e])}</td><td class="num">{v["n"]} ({100 * v["n"] / n:.0f}%)</td>'
        + ''.join(f'<td class="num">{f1(v[m])}</td>' for m in METHODS)
        + ''.join(f'<td class="num">{v["mean_k"][m]:.2f}</td>' for m in METHODS if m != 'st') + '</tr>'
        for e, v in d['step1'].items())
    stop_rows = ''.join(
        f'<tr><td><span class="key {SERIES[m]}"></span>{NAME[m]}</td><td class="num">{M[m]["mean_k"]:.2f}</td>'
        f'<td class="num">{M[m]["n_continued"]} ({100 * M[m]["n_continued"] / n:.0f}%)</td>'
        f'<td class="num">{M[m]["stop"].get("entropy", 0)}</td><td class="num">{M[m]["stop"].get("kmax", 0)}</td>'
        f'<td class="num">{M[m]["stop"].get("budget", 0)}</td>'
        f'<td class="num">{M[m]["mean_think"]:.0f}</td><td class="num">{M[m]["think_ratio"]:.2f}×</td></tr>'
        for m in METHODS)
    k_rows = ''.join(
        f'<tr><td><span class="key {SERIES[m]}"></span>{NAME[m]}</td>'
        + ''.join(f'<td class="num">{d["k_hist"][m].get(k, 0) or "·"}</td>' for k in range(1, 11)) + '</tr>'
        for m in METHODS if m != 'st')
    v = M['visref_raw_rank']
    inj = sorted(v['inject'])
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0
    inj_m = sorted(v['inject_m'])
    ans_rows = ''.join(
        f'<tr><td><span class="key {SERIES[m]}"></span>{NAME[m]}</td><td class="num">{M[m]["forced"]}</td>'
        f'<td class="num">{M[m]["forced_eos"]}</td><td class="num">{M[m]["af"]}</td><td class="num">{M[m]["af_valid"]}</td>'
        f'<td class="num">{M[m]["forced_unusable"]}</td><td class="num">{M[m]["rules"].get("boxed", 0)}</td>'
        f'<td class="num">{M[m]["rules"].get("answer_phrase", 0)}</td><td class="num">{M[m]["extract_fallback"]}</td><td class="num">{M[m]["inner_think_end"]}</td></tr>'
        for m in METHODS)
    def _it(c):
        return f'{c["inner"]}/{c["n"]} ({100 * c["rate"]:.0f}%)' if c['n'] else '·'
    inner_rows = ''.join(
        f'<tr><td><span class="key {SERIES[m]}"></span>{NAME[m]}</td><td class="num">{_it(h["inner_think"][m]["all"])}</td>'
        f'<td class="num">{_it(h["inner_think"][m]["k1_close"])}</td><td class="num">{_it(h["inner_think"][m]["k1_cap"])}</td></tr>'
        for m in METHODS)
    sens_rows = ''.join(
        f'<tr><td><span class="key {SERIES[m]}"></span>{NAME[m]}</td>'
        + ''.join(f'<td class="num">{f1(M[m]["sens"][k])}</td>' for k in ('a', 'b', 'c', 'strict_boxed'))
        + f'<td class="num">{M[m]["sens"]["n_af"]}</td><td class="num">{M[m]["sens"]["af_correct_a"]} / {M[m]["sens"]["af_correct_b"]}</td></tr>'
        for m in METHODS) if 'sens' in M['st'] else ''
    sens_section = f"""
<section>
  <h2>답 처리 규칙 민감도</h2>
  <p>논문은 토큰 상한·잘림·답 강제를 기술하지 않습니다. 우리 구현의 두 규칙(잘린 생각 뒤 <code>Final Answer:</code>, 답이 상한에 닿으면 DEER af)이 결과를 얼마나 바꾸는지 세 읽기로 봅니다. 주 검정은 (a)만 씁니다.</p>
  <div class="panel"><div class="tablewrap"><table>
    <thead><tr><th>방법</th><th>(a) 그대로 %</th><th>(b) af 앞에서 잘라 재채점 %</th><th>(c) af 행 오답 %</th><th>\\boxed만 인정 %</th><th>af 행</th><th>af 행 정답 (a) / (b)</th></tr></thead>
    <tbody>{sens_rows}</tbody></table></div>
    <p class="meta">(b)는 두 규칙이 모두 없을 때의 근사, (c)는 하한입니다(답 상한에 닿았어도 앞부분에 답이 있는 행까지 오답 처리). \\boxed만 인정: 규칙 추출이 \\boxed에서 답을 찾은 정답만 셉니다.</p>
  </div>
</section>""" if sens_rows else ''
    versions = ', '.join(f'<code>{esc(k)}</code> ({c}행)' for k, c in h['code_versions'].items())
    judgment_section = ''
    if judgment:
        JN = {'visref_raw_rank-st': 'VisRef−ST (주)', 'visref_raw_rank-tsr': 'VisRef−TSR (주)', 'tsr-st': 'TSR−ST (보조)'}
        holm = lambda v: f'{v["p_holm"]:.3f}' if 'p_holm' in v else '·'
        jrows = ''.join(
            f'<tr><td>{JN[k]}</td><td class="num">{signed(v["diff_pp"])}</td>'
            f'<td class="num">{signed(v["ci95_pp"][0])} ~ {signed(v["ci95_pp"][1])}</td>'
            f'<td class="num">{v["a_only"]} / {v["b_only"]}</td><td class="num">{v["p_mcnemar"]:.3f}</td>'
            f'<td class="num">{holm(v)}</td>'
            f'<td class="num">{signed(v["paper_pp"])}</td><td>{esc(v["judgment"])}</td></tr>'
            for k, v in judgment['pairs'].items())
        SN = {'a_v3_llm': '(a) v3 + LLM (판정)', 'rule_only_v3': '규칙만 v3', 'b_af_stripped': '(b) af 앞에서 재채점',
              'c_af_wrong': '(c) af 행 오답'}
        srows = ''.join(
            f'<tr><td>{SN[k]}</td>' + ''.join(f'<td class="num">{f1(v["acc"][m])}</td>' for m in METHODS)
            + ''.join(f'<td class="num">{signed(v["pairs"][p][0])} ({signed(v["pairs"][p][1][0])}~{signed(v["pairs"][p][1][1])})</td>'
                      for p in ('visref_raw_rank-st', 'visref_raw_rank-tsr', 'tsr-st')) + '</tr>'
            for k, v in judgment['sensitivity'].items())
        acc = judgment['acc']
        judgment_section = f"""
<section>
  <h2>사전등록 판정 (최종, 읽기 (a) = 채점 v3 + LLM 추출)</h2>
  <p>현재 InternVL3.5-1B 운영 구현(ST ≈ 58%)의 MathVista testmini 1,000문항 평가입니다. 정확도 ST {acc['st']:.1f} / TSR {acc['tsr']:.1f} / VisRef {acc['visref_raw_rank']:.1f} (논문 46.1 / 48.5 / 52.0).</p>
  <div class="panel"><div class="tablewrap"><table>
    <thead><tr><th>비교</th><th>차이 %p</th><th>95% CI</th><th>앞만 / 뒤만 정답</th><th>McNemar p</th><th>Holm p (주 비교 2개)</th><th>논문 %p</th><th>사전등록 분류</th></tr></thead>
    <tbody>{jrows}</tbody></table></div>
    <ul class="notes">
      <li>분류 정의(사전등록): "재현됨" = 같은 문항 짝 차이의 95% 부트스트랩 CI가 0을 제외하고 논문 값을 포함함 — 논문 수치와 <b>양립</b>한다는 뜻이며 효과 크기의 동등성이나 메커니즘·구현의 충실성을 입증하지 않습니다. "효과 크기 재현 안 됨" = CI가 논문 값을 포함하지 않음. "판정 불가" = CI가 0과 논문 값을 모두 포함.</li>
      <li>논문 값이 우리 CI 밖이라는 것은 논문 쪽 불확실성을 함께 고려한 두 결과 간 차이 검정이 아닙니다(논문 문항별 결과 없음). 따라서 논문과 우리 결과의 차이는 정식으로 검정하지 못했습니다.</li>
      <li>시점: 주 비교와 검정(McNemar + Holm)은 결과 전에, CI 분류 기준은 759문항 중간 결과를 확인한 뒤 최종 분석 전에 고정했습니다. 최종 1,000문항에 그 759문항이 포함되므로 이 분류는 결과 노출 전 사전등록 수준의 확증적 판정이 아닙니다. VisRef−TSR의 CI는 논문 +3.5는 배제하지만 작은 양의 이득까지 배제하지는 않고, VisRef−ST는 Holm 보정 후 유의하지 않습니다.</li>
      <li>범위: TSR 이득에는 생각 예산 연장 효과가 섞여 있습니다(ST 첫 단계 641/1,000이 상한에서 잘림, 이어가기 대부분이 절단 뒤). VisRef는 주입 수/m 중앙값 약 0.2로, DPP 잔차 임계값에 따른 조기 종료 때문에 논문의 |V_k| = m과 다른 확인된 절차 차이가 있습니다. 정지 엔트로피 추정과 추론 텍스트 토큰 표현 Z는 논문에 없어 운영 선택입니다. 따라서 이 결과만으로 논문의 TSR 메커니즘 재현이나 VisRef 자체의 효과 부재를 확정하지 않습니다.</li>
    </ul>
  </div>
  <div class="panel"><h3>채점 읽기별 민감도 (판정에 쓰지 않음)</h3><div class="tablewrap"><table>
    <thead><tr><th>읽기</th><th>ST %</th><th>TSR %</th><th>VisRef %</th><th>VisRef−ST</th><th>VisRef−TSR</th><th>TSR−ST</th></tr></thead>
    <tbody>{srows}</tbody></table></div>
    <p class="meta">세 읽기 모두 CI 기반 분류는 같습니다. 유의성은 같지 않습니다: (c)에서 VisRef−ST의 Holm p는 약 0.042로 (a)의 0.067과 다릅니다. (c)의 절대값이 논문 ST·TSR에 가깝다는 관찰은 사후 가설입니다(미확인).</p>
  </div>
</section>"""

    return f"""<title>VisRef 1B 재현 결과</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Hahmlet:wght@600;700&family=IBM+Plex+Sans+KR:wght@400;600&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
  <div class="eyebrow">MathVista testmini · InternVL3.5-1B · Appendix Table 6</div>
  <h1>VisRef 1B 재현 결과</h1>
  <p class="meta">포함 세션: {esc(label)} · 세 방법이 모두 끝난 문항 {n}개 (셔플 순서의 완료된 앞부분) · 생성 {esc(generated)}</p>
  {'' if final else f'<div class="banner">중간 결과입니다(문항 {n}개, LLM 재채점 전). ' + ('방법 간 차이의 95% 신뢰구간이 모두 0을 포함해 아직 차이를 판정할 수 없습니다. ' if not decided else '') + '최종 집계는 남은 세션과 재채점을 합친 뒤 같은 스크립트로 다시 만듭니다.</div>'}
</header>
{judgment_section}
<section class="panel" aria-label="실행 점검">
  <h3>실행 점검</h3>
  <div class="status">
    <div>{pill(h['errors'] == 0, f"오류 {h['errors']}행")}</div>
    <div>{pill(ok_n, f"완료 문항 {n}개 (첫 세션 기준 40개 이상)")}</div>
    <div>{pill(unusable <= 0.1 * max(1, forced_tot), f"쓸 수 없는 강제 답 {unusable}/{forced_tot}")}</div>
    <div>{pill(h['crn_n'] == 0 or h['crn_correct'] >= 0.8 * h['crn_n'], f"CRN: TSR 1단계 정지 {h['crn_n']}문항 중 ST와 정오 일치 {h['crn_correct']}, 추출 답 일치 {h['crn_same']}")}</div>
  </div>
  <p class="meta">CRN 점검: 1단계에서 멈춘 TSR은 ST와 같은 문맥·같은 시드로 답합니다. 답 전문이 같은 경우는 {h['crn_text']}/{h['crn_n']}이고, 나머지는 fp16 배치 구성 차이로 샘플링이 중간에 갈라진 것입니다.</p>
  <p class="meta">code_version: {versions} · 79eef648ebd1과 f2422d0e72ff는 세 방법 출력이 같음을 확인(810/810), d0095f79e691은 채점 규칙만 다름(생성 코드 해시 동일)</p>
  <p class="meta">채점: <code>{esc(h['scorer'])}</code>{f" · 기록된 채점과 정오가 달라진 행 {h['rescored_changed']}" if 'rescored_changed' in h and h['scorer'].startswith('v') else ''}</p>
</section>

<section>
  <h2>정확도</h2>
  <p>규칙 추출 기준 정확도와 Wilson 95% 신뢰구간입니다. 마름모는 논문 Table 6의 값입니다. 우리 ST는 공식 InternVL3.5-1B 점수(59.3, 보고서 문장상 thinking 모드 여부는 모호)에 가깝고 논문 ST(46.1)보다 높아서, 절대값보다 방법 간 차이를 비교합니다.</p>
  <div class="panel">
    <div class="legend"><span><span class="key s1"></span>ST</span><span><span class="key s2"></span>TSR</span><span><span class="key s3"></span>VisRef</span><span><span class="diamond"></span>논문 Table 6</span></div>
    {ci_chart(items, lo, hi, 10)}
    <div class="tablewrap"><table>
      <thead><tr><th>방법</th><th>정답/문항</th><th>정확도 %</th><th>95% CI</th><th>규칙+LLM %</th><th>논문 %</th><th>논문 대비</th></tr></thead>
      <tbody>{acc_rows}</tbody></table></div>
  </div>
</section>

<section>
  <h2>방법 간 차이</h2>
  <p>같은 문항끼리 짝지은 차이(%p)와 부트스트랩 95% 신뢰구간입니다. 불일치 쌍은 한쪽만 맞힌 문항 수이고, p는 McNemar 정확 검정입니다. 마름모는 논문의 차이입니다.</p>
  <div class="panel">
    {ci_chart(pitems, -span, span, 5 if span <= 15 else 10, zero=0, unit='')}
    <div class="tablewrap"><table>
      <thead><tr><th>비교</th><th>차이 %p</th><th>95% CI</th><th>앞만 정답 / 뒤만 정답</th><th>McNemar p</th><th>논문 %p</th></tr></thead>
      <tbody>{pair_rows}</tbody></table></div>
  </div>
</section>

{sens_section}

<section>
  <h2>첫 단계가 끝난 방식별 결과</h2>
  <p>첫 단계 생각은 세 방법이 같은 것을 나눠 쓰므로, 끝난 방식은 문항의 성질입니다. TSR·VisRef의 이어가기는 대부분 상한에서 잘린 문항에서 일어납니다.</p>
  <div class="panel"><div class="tablewrap"><table>
    <thead><tr><th>첫 단계 종료</th><th>문항</th><th>ST %</th><th>TSR %</th><th>VisRef %</th><th>TSR 평균 k</th><th>VisRef 평균 k</th></tr></thead>
    <tbody>{s1_rows}</tbody></table></div></div>
</section>

<section>
  <h2>단계 수와 생각 토큰</h2>
  <p>TSR과 VisRef가 몇 단계에서 멈췄는지의 분포입니다. 정지 통계 H가 δ = 0.25보다 작으면 멈추고, 최대 10단계 또는 생각 토큰 합계 10,240에서 멈춥니다.</p>
  <div class="panel">
    <div class="legend"><span><span class="key s2"></span>TSR</span><span><span class="key s3"></span>VisRef</span></div>
    {k_chart(d['k_hist'], n)}
    <div class="tablewrap"><table>
      <thead><tr><th>문항 수</th>{''.join(f'<th>k={k}</th>' for k in range(1, 11))}</tr></thead>
      <tbody>{k_rows}</tbody></table></div>
    <div class="tablewrap"><table>
      <thead><tr><th>방법</th><th>평균 k</th><th>2단계 이상</th><th>엔트로피 정지</th><th>최대 단계</th><th>예산</th><th>평균 생각 토큰</th><th>ST 대비</th></tr></thead>
      <tbody>{stop_rows}</tbody></table></div>
    <p class="meta">논문 Fig. 3 판독: VisRef 생각 토큰은 ST의 약 3.5배.</p>
  </div>
</section>

<section>
  <h2>VisRef 주입, 강제 답, 채점</h2>
  <div class="panel">
    <h3>다시 넣은 시각 토큰 (VisRef, 단계마다 — 마지막 단계의 선택은 최종 답 직전에 들어감)</h3>
    <div class="tablewrap"><table>
      <thead><tr><th>주입 횟수</th><th>m 중앙값</th><th>Z 고유 토큰 중앙값</th><th>주입 수 중앙값 (25–75%)</th><th>주입 수 / m 중앙값</th><th>m보다 적게 넣음</th><th>주입 = Z 고유 토큰 수</th><th>자연 종료 뒤 H ≥ δ</th></tr></thead>
      <tbody><tr><td class="num">{len(inj)}</td><td class="num">{q(sorted(v['m_list']), .5)}</td><td class="num">{q(sorted(v['uz_list']), .5)}</td>
        <td class="num">{q(inj, .5)} ({q(inj, .25)}–{q(inj, .75)})</td><td class="num">{q(inj_m, .5):.2f}</td>
        <td class="num">{v['inject_lt_m']}/{len(inj)}</td><td class="num">{v['inject_eq_uz']}/{len(inj)}</td><td class="num">{v['close_probe_H_ge']}/{v['close_probes']}</td></tr></tbody>
    </table></div>
    <p class="meta">V는 문항의 시각 토큰, Z는 직전 생각 단계의 토큰 임베딩이고 m = ⌊0.3·|V|⌋입니다. 커널 L = (VZᵀ)(VZᵀ)ᵀ의 계수는 Z의 고유 토큰 수를 넘지 못하므로, DPP greedy가 고른 토큰만 넣으면(인용한 Chen 2018 코드와 같은 동작) m보다 적게 들어갈 수 있습니다. 마지막 열은 생각을 스스로 닫은 단계 중 정지 통계가 δ 이상이라 이어간 경우입니다.</p>
  </div>
  <div class="panel">
    <h3>최종 답과 추출</h3>
    <div class="tablewrap"><table>
      <thead><tr><th>방법</th><th>강제 답</th><th>EOS 종료</th><th>af 폴백</th><th>af 유효</th><th>쓸 수 없음</th><th>\\boxed 추출</th><th>답 문구 추출</th><th>LLM 추출 대상</th><th>답 안 &lt;/think&gt;</th></tr></thead>
      <tbody>{ans_rows}</tbody></table></div>
    <div class="tablewrap"><table>
      <thead><tr><th>방법</th><th>답 안 &lt;/think&gt; 전체</th><th>k=1, 마지막 단계 자연 종료</th><th>k=1, 마지막 단계 상한 잘림</th></tr></thead>
      <tbody>{inner_rows}</tbody></table></div>
    <p class="meta">강제 답: 마지막 단계가 상한에서 잘려 <code>Final Answer:</code>로 답하게 한 경우(s1). af 폴백: 답이 EOS 없이 1024토큰에 닿아 <code>**Final Answer** \\boxed</code>로 마무리한 경우(DEER). LLM 추출 대상: 규칙으로 답을 못 찾아 마지막 줄을 쓴 행으로, 공식 MathVista 추출 프롬프트로 다시 채점합니다. 답 안 &lt;/think&gt;: 최종 답 안에서 모델이 생각을 다시 열고 닫은 행으로, 채점은 마지막 &lt;/think&gt; 뒤 텍스트로 합니다.</p>
  </div>
</section>

<section>
  <h2>설정과 해석 시 주의</h2>
  <div class="panel">
    <ul class="notes">
      <li>논문 값: thinking 모드, δ = 0.25, m = ⌊0.3·|V|⌋, K_max = 10. 모델 카드 값(논문에 없음): R1 thinking 프롬프트, T = 0.6, top-k 50, top-p 0.95.</li>
      <li>논문에 없는 운영 선택: 단계 상한 1024 / 합계 10,240 / 답 1024, 정지 통계는 <code>\\boxed</code> 답 1개의 토큰 엔트로피 합(T = 1), 잘린 단계의 s1 강제 답과 DEER af 폴백, rank 이상은 채우지 않는 DPP 주입, 규칙 추출 채점.</li>
      <li>문항 수가 n일 때 짝지은 차이의 오차는 대략 ±{196 * math.sqrt(0.08 / max(1, n)):.1f}%p(불일치율 8% 가정)입니다.{'' if judgment else ' 판정은 사전등록된 최종 분석에서만 합니다.'}</li>
      <li>s1에 섞인 이전 판 <code>visref_raw</code>(rank 이후를 관련성 r²로 채움, 74문항)는 분석에서 뺐습니다. 새 <code>visref_fill</code>(εI로 m까지 채움)은 별도 사전등록 실험입니다.</li>
    </ul>
  </div>
</section>
</div>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='+')
    ap.add_argument('--order', required=True)
    ap.add_argument('--shards', type=int, default=2)
    ap.add_argument('--label', default='')
    ap.add_argument('--llm', default='', help='rescored JSONL from llm_extract.py merge')
    ap.add_argument('--out', required=True)
    ap.add_argument('--rescore-data', default='', help='mathvista_testmini.json: re-score all rows with the current scorer')
    ap.add_argument('--final', action='store_true', help='drop the interim banner (all sessions and rescoring merged)')
    ap.add_argument('--judgment', default='', help='diag/final_judgment.json: add the pre-registered judgment section')
    a = ap.parse_args()
    rows = S.load(a.results, list(METHODS))
    llm_rows = [json.loads(l) for l in Path(a.llm).read_text().splitlines()] if a.llm else None
    data = json.loads(Path(a.rescore_data).read_text()) if a.rescore_data else None
    if data is not None:
        print('rescored rows with changed correctness:', rescore_rows(rows, data), f'(scorer {SCORER_VERSION})')
    d = compute(rows, json.loads(Path(a.order).read_text()), a.shards, llm_rows, data)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    judgment = json.loads(Path(a.judgment).read_text()) if a.judgment else None
    out.write_text(page(d, a.label, time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()), final=a.final, judgment=judgment))
    out.with_suffix('.json').write_text(json.dumps(d, indent=1, ensure_ascii=False))
    print(json.dumps({'n': d['n'], 'acc': {m: round(d['methods'][m]['acc'], 1) for m in METHODS},
                      'pairs': {k: [round(v['diff_pp'], 1), [round(x, 1) for x in v['ci95_pp']]] for k, v in d['pairs'].items()}}))


if __name__ == '__main__':
    main()
