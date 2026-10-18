"""Question lists of full_think (pre-registered diagnostic; see diag/full_think.py).

Questions whose stored step 1 (diag/step1_store_bmain.json) was cut at the 1,024-token cap (think_end == 'cap'), chosen from
the stored step-1 ending only (no correctness), in the main run's seed-0 order (kaggle/runs/b_main_hs/output/results/ids.json,
the order diag/step1_think_ids.py uses), so that a run cut short still holds a random prefix.

Sets (each a sub-sequence of the seed-0 order):
  cap_all        every 'cap' question (expected 641)
  cap_continued  cap_all and diag/step1_branch_ids.json 'ids' (main run: TSR k > 1 or VisRef k > 1; expected 297)
  cap_stopped    cap_all and diag/step1_think_ids.json 'ids' (main run: TSR and VisRef both stopped at k == 1; expected 344)
                 (297 + 344 == 641, no overlap: asserted)
  pilot_long     the 2 questions of cap_all with the largest est_ctx_F (memory pilot), largest first
est_ctx_F = prompt_len + think_tokens + n_visual: the length of the F context before the extra thinking
(prompt_len, n_visual: the main run's 'tsr' rows in llm/final/merged.jsonl; think_tokens: the stored step 1).

donor[set][pid]  the length-matched control's question for condition L, computed within that set's order by
inject_sensitivity_ids.donors: the next question in the set's order (cyclic) with the same number of image tokens, else the
next with more image tokens (its tokens are cut, never repeated). pilot_long has two questions only, so its donors are
cap_all's.

Output diag/full_think_ids.json (a dict; step1_branch-style readers take its 'ids' = cap_all):
  {'rule', 'order_file', 'source_sha256', 'n', 'ids', 'sets', 'set_sizes', 'donor', 'origin', 'stratum', 'n_visual',
   'prompt_len', 'think_tokens', 'est_ctx_F'}

  cd repro && python3 diag/full_think_ids.py            # writes diag/full_think_ids.json
  cd repro && python3 diag/full_think_ids.py --check    # exits 1 if the file differs from a fresh build
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO / 'diag'))
from inject_sensitivity_ids import donors  # noqa: E402

ORDER = REPRO / 'kaggle/runs/b_main_hs/output/results/ids.json'
STORE = REPRO / 'diag/step1_store_bmain.json'
BRANCH = REPRO / 'diag/step1_branch_ids.json'
THINK = REPRO / 'diag/step1_think_ids.json'
MERGED = REPRO / 'llm/final/merged.jsonl'
OUT = REPRO / 'diag/full_think_ids.json'
SOURCES = {'order': ORDER, 'store': STORE, 'step1_branch_ids': BRANCH, 'step1_think_ids': THINK, 'merged': MERGED}
EXPECT = {'cap_all': 641, 'cap_continued': 297, 'cap_stopped': 344, 'pilot_long': 2}
N_PILOT = 2
RULE = ("pids with think_end == 'cap' in the stored step 1 (diag/step1_store_bmain.json), in seed-0 order; cap_continued = "
        "cap and step1_branch_ids, cap_stopped = cap and step1_think_ids; pilot_long = the 2 cap pids with the largest "
        "est_ctx_F = prompt_len + think_tokens + n_visual; donor = next in the set's order (cyclic) with the same number of "
        "image tokens, else the next with more")


def tsr_sizes(merged_lines):
    """{pid: (n_visual, prompt_len)} of the main run's 'tsr' rows (llm/final/merged.jsonl)."""
    out = {}
    for line in merged_lines:
        if line.strip():
            r = json.loads(line)
            if r.get('method') == 'tsr':
                assert str(r['pid']) not in out, f"duplicate tsr row {r['pid']}"
                out[str(r['pid'])] = (int(r['n_visual']), int(r['prompt_len']))
    return out


def build(order, store, branch_ids, think_ids, sizes, expect=EXPECT, sources=None):
    order = [str(p) for p in order]
    assert len(set(order)) == len(order), 'duplicate pid in order'
    assert all(p in store for p in order), 'a stored step 1 is missing'
    cap = [p for p in order if store[p]['think_end'] == 'cap']
    branch, think = {str(p) for p in branch_ids}, {str(p) for p in think_ids}
    cont = [p for p in cap if p in branch]
    stop = [p for p in cap if p in think]
    assert not (set(cont) & set(stop)), 'cap_continued and cap_stopped overlap'
    assert set(cont) | set(stop) == set(cap), 'cap_continued + cap_stopped do not cover cap_all'
    assert len(cont) + len(stop) == len(cap)
    assert all(p in sizes for p in cap), 'a cap question has no tsr row in merged.jsonl'
    nv = {p: sizes[p][0] for p in cap}
    pl = {p: sizes[p][1] for p in cap}
    tt = {p: int(store[p]['think_tokens']) for p in cap}
    est = {p: pl[p] + tt[p] + nv[p] for p in cap}
    pos = {p: i for i, p in enumerate(cap)}
    pilot = sorted(cap, key=lambda p: (-est[p], pos[p]))[:N_PILOT]
    sets = {'cap_all': cap, 'cap_continued': cont, 'cap_stopped': stop, 'pilot_long': pilot}
    if expect:
        got = {k: len(v) for k, v in sets.items()}
        assert got == expect, f'unexpected set sizes {got} (expected {expect})'
    try:
        don = {k: donors(v, nv) for k, v in sets.items() if k != 'pilot_long'}
    except IndexError:
        raise AssertionError('a question has no donor: no other question of its set has as many or more image tokens')
    don['pilot_long'] = {p: don['cap_all'][p] for p in pilot}
    return {'rule': RULE, 'order_file': str(ORDER.relative_to(REPRO)) if ORDER.is_relative_to(REPRO) else str(ORDER),
            'source_sha256': sources or {}, 'n': len(cap), 'ids': cap, 'sets': sets,
            'set_sizes': {k: len(v) for k, v in sets.items()}, 'donor': don,
            'origin': {p: 'continued' if p in set(cont) else 'stopped' for p in cap},
            'stratum': {p: 'cap' for p in cap}, 'n_visual': nv, 'prompt_len': pl, 'think_tokens': tt, 'est_ctx_F': est}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fresh():
    store = json.loads(STORE.read_text())['pids']
    sizes = tsr_sizes(MERGED.read_text().splitlines())
    return build(json.loads(ORDER.read_text()), store, json.loads(BRANCH.read_text())['ids'],
                 json.loads(THINK.read_text())['ids'], sizes,
                 sources={k: sha(v) for k, v in SOURCES.items()})


def text(doc):
    return json.dumps(doc, ensure_ascii=False) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args(argv)
    new = text(fresh())
    if a.check:
        ok = OUT.exists() and OUT.read_text() == new
        print('ok' if ok else 'differs')
        return 0 if ok else 1
    OUT.write_text(new)
    print(json.dumps(json.loads(new)['set_sizes']), '->', OUT.relative_to(REPRO))
    return 0


if __name__ == '__main__':
    sys.exit(main())
