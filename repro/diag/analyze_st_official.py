"""Pre-registered analysis of st_official (EXPERIMENT_FLOW 2026-10-10, fixed before results).

st_official = ST with the InternVL3.5 official Thinking budget (65,536; in practice up to max_context), no forced
answer prompt. Paired with the b_main ST row of the same pid (current scorer, v4 since 2026-10-11).
  - denominator (2026-10-11 fix): every pid of the complete shuffled-order prefix that has a b_main ST row; official rows
    with status 'error' stay in it as wrong (as run.py records them). The error-excluded readings are reported too
    ('excluding_errors'), never used for the verdict. Before the fix, error rows were dropped from the denominator.
  - main: paired accuracy difference (official - b_main), bootstrap 95% CI, McNemar exact p; by b_main ST step-1 end
  - descriptive: official accuracy with Wilson 95% CI next to 59.3 (InternVL3.5 official) and 46.1 (VisRef paper)
    rule: CI contains 46.1 and not 59.3 -> "large thinking budget can explain the paper's low ST";
          contains 59.3 and not 46.1 -> "b_main ST represents the official baseline"; both/neither -> undecided
  - check: on pids where b_main ST step 1 closed naturally, correctness agreement >= 90% (CRN), else comparison invalid
  - also: thinking length distribution, share hitting the context cap and their accuracy, readings (b) af removed,
    (d) context-cap rows counted wrong (lower bound)

  python3 diag/analyze_st_official.py OFFICIAL_results.jsonl... --main MAIN_results.jsonl... --order IDS.json \
      --data ../data/raw/mathvista_testmini.json --out diag/st_official_analysis.json
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import summarize as S  # noqa: E402
from report import AF_PROMPT, mcnemar_exact, wilson  # noqa: E402
from visref.scoring import SCORER_VERSION, score  # noqa: E402

OFFICIAL_59, PAPER_46 = 59.3, 46.1  # SPEC_LEDGER X-1, Table 6 (VisRef_layout L900)


def sc(r, prob):
    if r.get('status') == 'error':  # failed job: wrong in every reading (run.py L220-221)
        return 0, 0
    a = score(r['final_answer'], prob)['correct']
    fa = r['final_answer']
    i = fa.rfind(AF_PROMPT) if r.get('final_af') else -1
    b = a if i < 0 else score(fa[:i], prob)['correct']
    return a, b


def boot(d, n=2000, seed=0):
    rng = random.Random(seed)
    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return [100 * bs[int(.025 * n)], 100 * bs[int(.975 * n)]]


def step1_end(r):
    return r['steps'][0]['think_end'] if r.get('steps') else 'error'


def readings(R, pids):
    out = {}
    n = len(pids)
    for k, idx in (('a', 0), ('b', 1)):
        d = [R[p]['off'][idx] - R[p]['main'][idx] for p in pids]
        ko, km = sum(R[p]['off'][idx] for p in pids), sum(R[p]['main'][idx] for p in pids)
        lo, hi = wilson(ko, n)
        out[f'reading_{k}'] = {'official_acc': 100 * ko / n, 'official_ci': [lo, hi], 'main_acc': 100 * km / n,
                               'diff_pp': 100 * sum(d) / n, 'diff_ci': boot(d),
                               'official_only': sum(x == 1 for x in d), 'main_only': sum(x == -1 for x in d),
                               'p_mcnemar': mcnemar_exact(sum(x == 1 for x in d), sum(x == -1 for x in d))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('official', nargs='+')
    ap.add_argument('--main', nargs='+', required=True)
    ap.add_argument('--order', required=True, help='b_main ids.json (shuffled order)')
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    data = json.loads(Path(a.data).read_text())
    order = json.loads(Path(a.order).read_text())
    main_rows = S.load(a.main, ['st'])
    off = {}
    for p in a.official:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                off[r['pid']] = r
    # complete prefix of the single-shard Colab run in shuffled order
    pids = []
    for p in order:
        if p not in off:
            break
        pids.append(p)
    pids = [p for p in pids if (p, 'st') in main_rows]  # error rows stay (wrong); see docstring
    n = len(pids)
    if not n:
        sys.exit('no complete prefix with b_main ST rows')
    R = {p: {'off': sc(off[p], data[p]), 'main': sc(main_rows[(p, 'st')], data[p])} for p in pids}
    err = [p for p in pids if off[p].get('status') == 'error']
    out = {'scorer_version': SCORER_VERSION, 'n': n, 'n_rows_file': len(off),
           'n_error': len(err), 'n_error_file': sum(r.get('status') == 'error' for r in off.values()),
           'denominator': 'complete prefix incl. error rows (scored wrong)',
           'code_versions': sorted({str(r.get('code_version')) for r in off.values()})}
    out.update(readings(R, pids))
    ok = [p for p in pids if p not in set(err)]
    out['excluding_errors'] = dict(n=len(ok), **readings(R, ok)) if err and ok else None
    ra = out['reading_a']
    has59, has46 = ra['official_ci'][0] <= OFFICIAL_59 <= ra['official_ci'][1], ra['official_ci'][0] <= PAPER_46 <= ra['official_ci'][1]
    out['verdict'] = ('large thinking budget can explain paper ST' if has46 and not has59 else
                      'b_main ST represents official baseline' if has59 and not has46 else 'undecided')
    strata = {}
    for e in ('close', 'cap', 'eos'):
        ps = [p for p in pids if step1_end(main_rows[(p, 'st')]) == e]
        if ps:
            strata[e] = {'n': len(ps), 'official_acc': 100 * sum(R[p]['off'][0] for p in ps) / len(ps),
                         'main_acc': 100 * sum(R[p]['main'][0] for p in ps) / len(ps),
                         'agree': sum(R[p]['off'][0] == R[p]['main'][0] for p in ps)}
    out['by_main_step1_end'] = strata
    c = strata.get('close')
    out['crn_check_close_agree_ge_90pct'] = bool(c and c['agree'] >= 0.9 * c['n'])
    tt = sorted(off[p]['think_tokens'] for p in ok)
    q = lambda f: tt[min(len(tt) - 1, int(f * len(tt)))] if tt else None
    ctx = [p for p in ok if step1_end(off[p]) == 'cap']  # think cap 65,536 -> only the context room
    out['think_tokens_q'] = {f'q{int(f*100)}': q(f) for f in (.1, .25, .5, .75, .9)}
    out['official_step1_end'] = {e: sum(step1_end(off[p]) == e for p in pids) for e in ('close', 'cap', 'eos', 'error')}
    out['context_cap'] = {'n': len(ctx), 'acc': 100 * sum(R[p]['off'][0] for p in ctx) / len(ctx) if ctx else None,
                          'think_tokens_min': min((off[p]['think_tokens'] for p in ctx), default=None)}
    out['reading_d_context_cap_wrong'] = 100 * sum(R[p]['off'][0] for p in pids if p not in set(ctx)) / max(1, n)
    out['af_rows'] = {'official': sum(bool(off[p].get('final_af')) for p in pids), 'main': sum(bool(main_rows[(p, 'st')].get('final_af')) for p in pids)}
    if os.path.exists(a.out):
        sys.exit(f'refusing to overwrite {a.out}')
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
