"""Build the final tables in results/ from the analysis outputs in repro/diag/ (CPU only, standard library).

  python3 scripts/make_results.py        # run from the repository root

Inputs (committed; regenerate them with `make analysis`):
  repro/diag/final_judgment.json        main verdict (main scoring = rules + LLM fallback) and the other scoring variants
  repro/diag/analyze_fill.json          follow-up visref_fill (pre-registered)
  repro/diag/step1_branch_analysis.json follow-up step1_branch (pre-registered)
  repro/diag/analyze_hs.json            follow-up visref_hs, last-layer Z (pre-registered)
Outputs: results/accuracy.csv, results/paired_differences.csv, results/sensitivity.csv, results/followups.csv,
results/summary.json. Paper values: VisRef (arXiv 2603.00207) Appendix Table 6, row InternVL-1B.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / 'repro' / 'diag'
OUT = ROOT / 'results'
PAPER = {'st': 46.1, 'tsr': 48.5, 'visref_raw_rank': 52.0}
NAME = {'st': 'ST', 'tsr': 'TSR', 'visref_raw_rank': 'VisRef', 'visref_fill': 'VisRef-fill', 'visref_hs': 'VisRef-hs'}
LABEL = {'재현됨': 'consistent with the paper value',
         '효과 크기 재현 안 됨': 'effect size not reproduced',
         '효과 크기 재현 안 됨 (방향은 같으나 크기 다름)': 'effect size not reproduced (same direction, different size)'}
READING = {'a_v3_llm': 'main scoring: rules + LLM fallback (fixed in advance)', 'rule_only_v3': 'rules only, scorer v3',
           'b_af_stripped': 'scored before answer forcing, scorer v4',
           'b_af_stripped_v3': 'scored before answer forcing, scorer v3',
           'c_af_wrong': 'unfinished answers counted wrong'}


def pair_name(k):
    a, b = k.split('-')
    return f'{NAME.get(a, a)} - {NAME.get(b, b)}'


def write(name, header, rows):
    with open(OUT / name, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)


def main():
    fj = json.loads((DIAG / 'final_judgment.json').read_text())
    fill = json.loads((DIAG / 'analyze_fill.json').read_text())
    sb = json.loads((DIAG / 'step1_branch_analysis.json').read_text())
    hs = json.loads((DIAG / 'analyze_hs.json').read_text())
    OUT.mkdir(exist_ok=True)

    write('accuracy.csv', ['method', 'ours_pct', 'paper_pct', 'ours_minus_paper_pp'],
          [[NAME[m], a, PAPER[m], round(a - PAPER[m], 1)] for m, a in fj['acc'].items()])
    pairs = []
    for k, p in fj['pairs'].items():
        pairs.append([pair_name(k), p['kind'], p['diff_pp'], *p['ci95_pp'], p['a_only'], p['b_only'], p['p_mcnemar'],
                      p.get('p_holm', ''), p['paper_pp'], LABEL[p['judgment']]])
    write('paired_differences.csv', ['comparison', 'kind', 'diff_pp', 'ci95_lo', 'ci95_hi', 'only_first_correct',
                                     'only_second_correct', 'p_mcnemar', 'p_holm', 'paper_pp', 'classification'], pairs)
    sens = []
    for r, v in fj['sensitivity'].items():
        for k, (d, ci) in v['pairs'].items():
            sens.append([READING[r], pair_name(k), d, *ci, v['acc']['st'], v['acc']['tsr'], v['acc']['visref_raw_rank']])
    write('sensitivity.csv', ['reading', 'comparison', 'diff_pp', 'ci95_lo', 'ci95_hi', 'st_pct', 'tsr_pct',
                              'visref_pct'], sens)
    fol = []
    for k, p in fill['pairs'].items():
        fol.append(['visref_fill (pre-registered)', fill['n_prefix'], pair_name(k), p['diff_pp'], *p['ci95_pp'],
                    p['p_mcnemar'], ''])
    for grp in ('primary', 'secondary'):
        for k, p in sb[grp].items():
            role = 'co-primary' if grp == 'primary' else 'secondary'
            fol.append([f'step1_branch {role} (pre-registered)', sb['n'], k, p['diff_pp'], *p['ci95_pp'],
                        p.get('p_mcnemar_greedy', ''), p.get('p_holm', '')])
    for role, prs in (('primary', {'visref_hs-tsr': hs['main']['primary']}), ('secondary', hs['main']['secondary'])):
        for k, p in prs.items():
            fol.append([f'visref_hs {role} (pre-registered)', hs['n_prefix'], pair_name(k), p['diff_pp'], *p['ci95_pp'],
                        p['p_mcnemar'], ''])
    write('followups.csv', ['experiment', 'n', 'comparison', 'diff_pp', 'ci95_lo', 'ci95_hi',
                            'p_mcnemar (step1_branch: greedy answers)', 'p_holm'], fol)
    summary = {
        'target': 'VisRef (arXiv 2603.00207) Appendix Table 6, InternVL-1B, MathVista testmini (n = 1,000)',
        'paper_pct': {NAME[m]: v for m, v in PAPER.items()},
        'main': {'reading': READING['a_v3_llm'], 'n': fj['n'], 'acc_pct': {NAME[m]: a for m, a in fj['acc'].items()},
                 'pairs': {pair_name(k): {**{x: p[x] for x in ('kind', 'diff_pp', 'ci95_pp', 'a_only', 'b_only',
                                                              'p_mcnemar', 'paper_pp')},
                                          'p_holm': p.get('p_holm'), 'classification': LABEL[p['judgment']]}
                           for k, p in fj['pairs'].items()}},
        'visref_fill': {'n': fill['n_prefix'], 'acc_pct': {NAME[m]: a for m, a in fill['acc'].items()},
                        'pairs': {pair_name(k): p for k, p in fill['pairs'].items()}},
        'step1_branch': {'n': sb['n'], 'acc_t1_pct': sb['acc_t1'], 'acc_greedy_pct': sb['acc_greedy'],
                         'primary': sb['primary'], 'secondary': sb['secondary'],
                         'verdict': 'rule 4: immediate-answer gain from step-1 re-injection not reproduced'},
        'visref_hs': {'n': hs['n_prefix'], 'acc_pct': {NAME[m]: a for m, a in hs['main']['acc'].items()},
                      'primary': {pair_name('visref_hs-tsr'): hs['main']['primary']},
                      'secondary': {pair_name(k): p for k, p in hs['main']['secondary'].items()},
                      'verdict': {1: 'rule 1: extra gain over TSR with last-layer Z',
                                  2: 'rule 2: undecided (a paper-size effect is not ruled out)',
                                  3: 'rule 3: no paper-size (+3.5 pp) extra gain over TSR with last-layer Z'}[hs['verdict']['code']],
                      'rules_only': {'acc_pct': {NAME[m]: a for m, a in hs['rules_only']['acc'].items()},
                                     'primary': hs['rules_only']['primary']}},
        'sources': ['repro/diag/final_judgment.json', 'repro/diag/analyze_fill.json',
                    'repro/diag/step1_branch_analysis.json', 'repro/diag/analyze_hs.json'],
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n')
    print('wrote', sorted(p.name for p in OUT.iterdir()))


if __name__ == '__main__':
    main()
