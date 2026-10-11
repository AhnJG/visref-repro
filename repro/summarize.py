"""Aggregate result JSONL files: accuracy per method, paired differences, run diagnostics."""
import argparse
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

DELTA = 0.25


def median(xs):
    return statistics.median(xs) if xs else None


def quantiles(xs, qs=(.1, .25, .5, .75, .9)):
    xs = sorted(xs)
    return [round(xs[min(len(xs) - 1, int(q * len(xs)))], 3) for q in qs] if xs else None


PAPER_1B = {'st': 46.1, 'tsr': 48.5, 'visref_raw_rank': 52.0}  # Appendix Table 6, InternVL-1B, MathVista
ORDER = ('st', 'tsr', 'visref_raw_rank')


def load(paths, methods=None):
    """methods: keep only these methods (e.g. a method dropped after some sessions must not cut the
    complete prefix of the others)."""
    rows = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            if methods and r['method'] not in methods:
                continue
            rows[(r['pid'], r['method'])] = r
    return rows


def paired(rows, a, b, pids, n_boot=2000, seed=0):
    d = [rows[(p, a)]['correct'] - rows[(p, b)]['correct'] for p in pids]
    if not d:
        return {'diff_pp': None, 'ci95_pp': None, 'n': 0}
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    disc = {'a_only': sum(x == 1 for x in d), 'b_only': sum(x == -1 for x in d)}  # McNemar discordant pairs
    return {**disc, 'diff_pp': 100 * sum(d) / len(d), 'ci95_pp': [100 * boots[int(.025 * n_boot)], 100 * boots[int(.975 * n_boot)]],
            'n': len(d)}


def complete_prefix(rows, methods, order, shards):
    """pids in the shuffled order, per shard, up to the first pid not finished by every method.
    Deadline-cut pids are the long/hard ones, so only an unbroken prefix is an unbiased random subset."""
    keep = []
    for si in range(shards):
        for pid in order[si::shards]:
            if not all((pid, m) in rows for m in methods):
                break
            keep.append(pid)
    return keep


def summarize(rows, order=None, shards=2):
    methods = sorted({m for _, m in rows}, key=lambda m: (ORDER.index(m) if m in ORDER else len(ORDER), m))
    common = sorted({p for p, _ in rows if all((p, m) in rows for m in methods)}, key=int)
    if order:
        common = sorted(complete_prefix(rows, methods, order, shards), key=int)
    out = {'basis': 'shuffled-order complete prefix' if order else 'pids finished by all methods',
           'n_common': len(common), 'methods': {}}
    for m in methods:
        rs = [rows[(p, m)] for p in common]
        allr = [r for (p, mm), r in rows.items() if mm == m]
        steps = [s for r in rs for s in r['steps']]
        probes = [s for s in steps if 'H' in s]
        out['methods'][m] = {
            'n': len(rs), 'accuracy': round(100 * sum(r['correct'] for r in rs) / max(1, len(rs)), 2),
            'accuracy_all_rows': round(100 * sum(r['correct'] for r in allr) / len(allr), 2), 'n_all_rows': len(allr),
            'n_error': sum(r.get('status') == 'error' for r in rs),
            'paper_1b': PAPER_1B.get(m),
            'mean_k': round(sum(r['k'] for r in rs) / max(1, len(rs)), 2),
            'mean_think_tokens': round(sum(r['think_tokens'] for r in rs) / max(1, len(rs)), 1),
            'stop_reason': dict(Counter(r['stop_reason'] for r in rs)),
            'step_end': dict(Counter(s['think_end'] for s in steps)),
            'final_forced_close': sum(r['final_forced_close'] for r in rs),
            'n_final_forced_answer': sum(bool(r.get('final_forced_answer')) for r in rs),
            'n_final_forced_invalid': sum(bool(r.get('final_forced_answer')) and not r.get('final_forced_valid')
                                          for r in rs),
            'n_stop_at_k1': sum(r['k'] == 1 for r in rs),
            'extract_rule': dict(Counter(r['rule'].split(':')[0] for r in rs)),
            'dpp_frac_mean': (round(sum(s['n_dpp'] / s['m'] for s in steps if s.get('m')) /
                                    max(1, sum(1 for s in steps if s.get('m'))), 3) if m.startswith('visref') else None),
            'mean_n_visual': round(sum(r['n_visual'] for r in rs) / max(1, len(rs)), 1),
            'mean_wall_s': round(sum(r['wall_s'] for r in rs) / max(1, len(rs)), 1),
            'probe_H_first_step': sorted(round(r['steps'][0]['H'], 3) for r in rs if r['steps'] and 'H' in r['steps'][0])[:40],
            'probe_finish': dict(Counter(s.get('probe_finish') for s in steps if 'H' in s)),
            'final_finish': dict(Counter(r.get('final_finish') for r in rs)),
            'n_extract_failed': sum(bool(r.get('extract_failed')) for r in rs),
            'code_versions': sorted({str(r.get('code_version')) for r in allr}),
            'n_continued': sum(r['k'] > 1 for r in rs),
            'k_hist': dict(sorted(Counter(r['k'] for r in rs).items())),
            'mean_gen_tokens': round(sum(r.get('gen_tokens', 0) for r in rs) / max(1, len(rs)), 1),
            'n_final_inner_think_end': sum(bool(r.get('final_y_start')) for r in rs),
            'n_steps': len(steps),
            'n_final_question_text': sum('Question' in (r.get('final_raw') or r['final_answer']) for r in rs),
            # stopping-probe diagnostics over all steps
            'n_probes': len(probes),
            'probe_H_q10_25_50_75_90': quantiles([s['H'] for s in probes if math.isfinite(s['H'])]),
            'n_probe_H_below_delta': sum(s['H'] < DELTA for s in probes),
            'n_entropy_stop_at_k1': sum(r['k'] == 1 and r['stop_reason'] == 'entropy' for r in rs),
            'probe_inner_step1': [r['steps'][0].get('probe_inner', r['steps'][0].get('probe_answer'))
                                  for r in rs if r['steps'] and 'H' in r['steps'][0]][:20],
            'median_tokens_k2plus_step': median([s['think_tokens'] for r in rs for s in r['steps'][1:]]),
            'median_rank_V': median([s['rank_V'] for s in steps if 'rank_V' in s]),
            'median_n_unique_z': median([s['n_unique_z'] for s in steps if 'n_unique_z' in s]),
            'median_n_dpp': median([s['n_dpp'] for s in steps if 'n_dpp' in s]),
            'median_n_inject': median([s.get('n_inject', s['m']) for s in steps if 'm' in s]),
            'median_inject_over_m': median([round(s.get('n_inject', s['m']) / s['m'], 3) for s in steps if s.get('m')]),
            # pilot diagnostics (b_test): stratify by how step 1 ended; natural-close probes with H >= delta
            'by_step1_end': {e: {'n': len(g), 'acc': round(100 * sum(r['correct'] for r in g) / len(g), 1),
                                 'mean_k': round(sum(r['k'] for r in g) / len(g), 2)}
                             for e in ('close', 'cap', 'eos')
                             for g in [[r for r in rs if r['steps'] and r['steps'][0]['think_end'] == e]] if g},
            'n_close_probe_H_ge_delta': sum(s['think_end'] == 'close' and 'H' in s and s['H'] >= DELTA for s in steps),
            'n_close_probes': sum(s['think_end'] == 'close' and 'H' in s for s in steps),
            'n_loop_steps': sum(s['think_tokens'] >= 256 and s.get('n_unique_z', s['think_tokens']) / s['think_tokens'] < 0.1
                                for s in steps if 'n_unique_z' in s),
            'n_final_forced_valid': sum(bool(r.get('final_forced_valid')) for r in rs if r.get('final_forced_answer')),
            # forced answer usable = s1 answer ended with EOS, or DEER answer forcing produced a box with content
            'n_final_forced_unusable': sum(bool(r.get('final_forced_answer')) and not r.get('final_forced_valid')
                                           and not r.get('final_af_valid') for r in rs),
            'n_final_af': sum(bool(r.get('final_af')) for r in rs),
            'n_final_af_valid': sum(bool(r.get('final_af_valid')) for r in rs),
            'median_m': median([s['m'] for s in steps if 'm' in s]),
        }
    if 'st' in out['methods'] and 'tsr' in out['methods']:  # CRN check: TSR stopped at k=1 gives ST's final answer
        same = [p for p in common if rows[(p, 'tsr')]['k'] == 1]
        out['crn_tsr_k1_same_final_as_st'] = [sum(rows[(p, 'tsr')]['final_answer'] == rows[(p, 'st')]['final_answer']
                                                  for p in same), len(same)]
    if 'st' in out['methods']:  # thinking-token ratio vs ST (Fig. 3 readings differ: ~3.5x per trace vs 4.2-7.3x)
        st_tok = out['methods']['st']['mean_think_tokens']
        out['think_token_ratio_vs_st'] = {m: round(v['mean_think_tokens'] / st_tok, 2) if st_tok else None
                                          for m, v in out['methods'].items()}
    pairs = [(a, b) for i, b in enumerate(methods) for a in methods[i + 1:]]
    out['paired'] = {f'{a}-{b}': paired(rows, a, b, common) for a, b in pairs}
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='+')
    ap.add_argument('--out')
    ap.add_argument('--order', help='ids.json of the run (shuffled order) to use the unbiased complete prefix')
    ap.add_argument('--shards', type=int, default=2)
    ap.add_argument('--methods', default='', help='comma list: summarize only these methods')
    a = ap.parse_args()
    s = summarize(load(a.results, [m for m in a.methods.split(',') if m]),
                  json.loads(Path(a.order).read_text()) if a.order else None, a.shards)
    txt = json.dumps(s, indent=1, ensure_ascii=False)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + '\n')
