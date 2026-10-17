"""Descriptive step-1 stopping table for b_main (st / tsr / visref_raw_rank), stratified by the ST step-1 end type.

Descriptive only; decided after probe correctness by step-1 end type was already seen on 2026-10-11; no decision rule.
  cd repro && python3 diag/step1_stop_table.py  -> diag/step1_stop_table.json
Probe correctness uses the same call as diag/analyze_step1_probe.py (score('\\boxed' + probe_answer, data[pid])['correct'],
invalid probe = wrong), from visref.scoring. ST final correctness = the final judged 'correct' in llm/final/merged.jsonl
(the main-run judgement, as analyze_step1_probe.py uses for final answers; the Kaggle rows' own 'correct' differs in 2 ST rows).
"""
import glob
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from visref.scoring import SCORER_VERSION, score  # noqa: E402

METHODS = ('st', 'tsr', 'visref_raw_rank')
PROBE_METHODS = ('tsr', 'visref_raw_rank')
QTYPES = ('all', 'multi_choice', 'free_form')
STRATA = ('cap', 'close', 'eos')
TOTAL_Q = 1000
NOTE = ('descriptive, decided after probe correctness by step-1 end type was already seen on 2026-10-11; '
        'no decision rule')


def load_rows(paths):
    """Rows keyed by (str(pid), method) for METHODS only; duplicate keys are refused."""
    rows = {}
    for p in paths:
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if r['method'] not in METHODS:
                continue
            key = (str(r['pid']), r['method'])
            if key in rows:
                raise ValueError('duplicate (pid, method) row: %r' % (key,))
            rows[key] = r
    return rows


def probe_correct(step, d):
    """Same as pc() in analyze_step1_probe.py: invalid probe = wrong."""
    return int(score('\\boxed' + step['probe_answer'], d)['correct']) if step.get('probe_valid') else 0


def auroc(h_pos, h_neg):
    """P(H_pos > H_neg) + 0.5 P(tie); inf is larger than any finite value (inf == inf is a tie). None if a class is empty."""
    if not h_pos or not h_neg:
        return None
    allv = sorted([(h, 1) for h in h_pos] + [(h, 0) for h in h_neg], key=lambda t: t[0])
    n = len(allv)
    rank_sum, i = 0.0, 0
    while i < n:
        j = i
        while j < n and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0  # average 1-based rank of the tie group
        rank_sum += avg * sum(t[1] for t in allv[i:j])
        i = j
    npos, nneg = len(h_pos), len(h_neg)
    return (rank_sum - npos * (npos + 1) / 2.0) / (npos * nneg)


def auroc_wrong(recs, method, n_boot=2000, seed=0):
    """AUROC of steps[0]['H'] for 'probe answer wrong' over records, with a question-bootstrap 95% CI."""
    hs = [(r[method]['H'], 1 - r[method]['pc']) for r in recs]
    pos = [h for h, w in hs if w]
    neg = [h for h, w in hs if not w]
    out = {'n_inf': sum(h == float('inf') for h, _ in hs), 'n_wrong': len(pos), 'n_right': len(neg),
           'auroc': None, 'ci95': None, 'n_boot_used': 0}
    a = auroc(pos, neg)
    if a is None:
        return out
    out['auroc'] = round(a, 4)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        s = [hs[rng.randrange(len(hs))] for _ in hs]
        v = auroc([h for h, w in s if w], [h for h, w in s if not w])
        if v is not None:
            vals.append(v)
    vals.sort()
    out['n_boot_used'] = len(vals)
    if vals:
        out['ci95'] = [round(vals[int(.025 * len(vals))], 4), round(vals[min(len(vals) - 1, int(.975 * len(vals)))], 4)]
    return out


def two_by_two(pairs):
    """pairs: iterable of (stopped_at_step1: bool, correct: bool-ish)."""
    t = {'stopped_correct': 0, 'stopped_wrong': 0, 'continued_correct': 0, 'continued_wrong': 0}
    for s, c in pairs:
        t[('stopped' if s else 'continued') + ('_correct' if c else '_wrong')] += 1
    return t


def cell(recs, method):
    n = len(recs)
    k1 = sum(r[method]['k'] == 1 for r in recs)
    out = {'n': n, 'n_stopped_k1': k1, 'base_rate_k1': round(k1 / n, 4) if n else None,
           'probe_2x2': two_by_two((r[method]['k'] == 1, r[method]['pc']) for r in recs)}
    if method == 'tsr':
        out['st_final_2x2'] = two_by_two((r['tsr']['k'] == 1, r['st_correct']) for r in recs)
        wrong_both = sum(r['tsr']['k'] == 1 and not r['tsr']['pc'] and not r['st_correct'] for r in recs)
        out['n_k1_probe_wrong'] = sum(r['tsr']['k'] == 1 and not r['tsr']['pc'] for r in recs)
        out['n_k1_probe_wrong_and_st_wrong'] = wrong_both
        out['max_gain_pp'] = round(wrong_both / TOTAL_Q * 100, 2)
    else:
        wrong = sum(r[method]['k'] == 1 and not r[method]['pc'] for r in recs)
        out['n_k1_probe_wrong'] = wrong
        out['max_gain_pp'] = round(wrong / TOTAL_Q * 100, 2)
    out['auroc_H_predicts_probe_wrong'] = auroc_wrong(recs, method)
    return out


def build_records(rows, data):
    pids = sorted({p for p, _ in rows}, key=int)
    recs = []
    for p in pids:
        for m in METHODS:
            assert (p, m) in rows, 'missing %r' % ((p, m),)
        s, t, v = (rows[(p, m)] for m in METHODS)
        rec = {'pid': p, 'qtype': data[p]['question_type'], 'think_end': s['steps'][0]['think_end'],
               'st_correct': int(bool(s['correct']))}
        for m, r in (('tsr', t), ('visref_raw_rank', v)):
            st0 = r['steps'][0]
            rec[m] = {'k': r['k'], 'H': float(st0['H']), 'pc': probe_correct(st0, data[p])}
        recs.append(rec)
    return recs


def build_tables(recs):
    out = {}
    for strat in STRATA:
        sub = [r for r in recs if r['think_end'] == strat]
        out[strat] = {'n_questions': len(sub), 'methods': {}}
        for m in PROBE_METHODS:
            out[strat]['methods'][m] = {
                q: cell([r for r in sub if q == 'all' or r['qtype'] == q], m) for q in QTYPES}
    return out


def main():
    paths = sorted(glob.glob(str(ROOT / 'kaggle/runs/b_main_s*/output/results/results_*.jsonl')))
    rows = load_rows(paths)
    for m in METHODS:
        assert sum(k[1] == m for k in rows) == TOTAL_Q, m
    data = json.load(open(ROOT / '../data/raw/mathvista_testmini.json'))
    final = {}
    for line in open(ROOT / 'llm/final/merged.jsonl'):
        if line.strip():
            r = json.loads(line)
            final[(str(r['pid']), r['method'])] = r['correct']
    for key in rows:  # the main-run judgement replaces the Kaggle rows' own field
        rows[key] = dict(rows[key], correct=final[key])
    recs = build_records(rows, data)
    assert len(recs) == TOTAL_Q
    out = {'note': NOTE, 'scorer_version': SCORER_VERSION,
           'probe_scoring': "score('\\\\boxed' + probe_answer, data[pid])['correct'] (visref.scoring.score, as in "
                            "diag/analyze_step1_probe.py); invalid probe = wrong",
           'st_final_correctness': "'correct' of the ST row in llm/final/merged.jsonl (main-run judgement, as analyze_step1_probe.py uses)",
           'source_files': [str(Path(p).relative_to(ROOT)) for p in paths],
           'stratified_by': "ST row steps[0]['think_end']; stopped at step 1 = row 'k' == 1",
           'max_gain_pp_def': 'questions stopped at step 1 and wrong (tsr: probe-wrong and ST-final-wrong; '
                              'visref_raw_rank: probe-wrong) / 1000 * 100',
           'auroc_def': "AUROC of steps[0]['H'] for 'probe wrong' (higher H -> wrong), ties 0.5, inf > finite; "
                        "95% CI = 2000 question-bootstrap resamples, seed 0 (resamples with an empty class skipped)",
           'strata': build_tables(recs)}
    json.dump(out, open(ROOT / 'diag/step1_stop_table.json', 'w'), indent=1, allow_nan=False)
    for strat, sv in out['strata'].items():
        print('[%s] n=%d' % (strat, sv['n_questions']))
        for m, qs in sv['methods'].items():
            for q, c in qs.items():
                a = c['auroc_H_predicts_probe_wrong']
                print('  %-16s %-12s n=%3d k1=%3d (%s) probe2x2=%s maxgain=%spp auroc=%s ci=%s n_inf=%d' % (
                    m, q, c['n'], c['n_stopped_k1'], c['base_rate_k1'], list(c['probe_2x2'].values()),
                    c['max_gain_pp'], a['auroc'], a['ci95'], a['n_inf']))


if __name__ == '__main__':
    main()
