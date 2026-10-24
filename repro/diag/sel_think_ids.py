"""Question lists and stored first steps of `sel_think` (plan docs/EXPERIMENT_FLOW.md 2026-10-24 "실행 전 계획: 선택 토큰 생성
대조", written before any result). Chosen from blank_think's fixed lists only (no correctness).

  main      every question of blank_think's main order (diag/blank_think_ids.json 'main') that has an ok blank_think_main row
            (724; its 396 analysis questions and the 328 it ran beyond them), in that order (EF 2026-10-24 second pre-run revision:
            more questions in place of the own-prompt conditions). The run is sharded by index (i % shard_total) and the analysed
            questions are a prefix of each shard, fixed from the trial's timing before the main run
            (diag/analyze_blank_think.make_analysis_ids)
  trial     blank_think's 5 trial questions (diag/blank_think_ids.json 'trial')
  memcheck  blank_think's memory-check question (the largest image), in trial
  n_visual, donor   from diag/blank_think_ids.json
  step1     {pid: {own, blank: {end, ids, sep_text}}}: the first steps blank_think generated and used (its result rows); sel_think
            does not generate step 1 again
  source    blank_think code_version of the rows

  cd repro && python3 diag/sel_think_ids.py            # writes diag/sel_think_ids.json
  cd repro && python3 diag/sel_think_ids.py --check    # exits 1 if the file differs from a fresh build
"""
import argparse
import json
from pathlib import Path

REPRO = Path(__file__).resolve().parent.parent
IDS = REPRO / 'diag/blank_think_ids.json'
ANALYSIS = REPRO / 'diag/blank_think_analysis_ids.json'
ROWS = [REPRO / f'kaggle/runs/blank_think_{s}/output/results/blank_think_{i}.jsonl' for s in ('main', 'trial') for i in (0, 1)]
OUT = REPRO / 'diag/sel_think_ids.json'


def stored_rows(paths):
    """{pid: ok row}; a pid with two ok rows is refused."""
    out = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            if r.get('status') == 'ok':
                assert r['pid'] not in out, f"pid {r['pid']} has two ok rows"
                out[r['pid']] = r
    return out


def build(ids, analysis, rows):
    main, trial = [str(p) for p in ids['main'] if str(p) in rows], [str(p) for p in ids['trial']]
    assert set(analysis['analysis']) <= set(main), 'blank_think analysis questions missing'
    pids = main + trial
    assert len(set(pids)) == len(pids), 'main and trial overlap'
    missing = [p for p in pids if p not in rows]
    assert not missing, f'no stored first step for {missing[:5]}'
    step1 = {p: {s: {k: rows[p]['step1'][s][k] for k in ('end', 'ids', 'sep_text')} for s in ('own', 'blank')} for p in pids}
    return {'experiment': 'sel_think', 'main': main, 'trial': trial, 'n_visual': {p: ids['n_visual'][p] for p in pids},
            'donor': {p: ids['donor'][p] for p in pids}, 'step1': step1,
            'memcheck': ids['memcheck'], 'source': sorted({rows[p]['code_version'] for p in pids})}


def text(doc):
    return json.dumps(doc, separators=(',', ':')) + '\n'


def fresh():
    return build(json.loads(IDS.read_text()), json.loads(ANALYSIS.read_text()), stored_rows(ROWS))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args(argv)
    t = text(fresh())
    if a.check:
        ok = OUT.exists() and OUT.read_text() == t
        print('ok' if ok else 'differs')
        return 0 if ok else 1
    OUT.write_text(t)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
