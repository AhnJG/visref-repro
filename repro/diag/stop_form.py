"""stop_form -- pre-registered diagnostic of the stopping-entropy estimator (plan docs/EXPERIMENT_FLOW.md 2026-10-24 "실행 전 계획:
멈춤 엔트로피 추정 방식 교차 진단"; this file is code only, written before any result).

Question: after the main run's stored first thinking step, would other ways of estimating the stopping entropy make the
questions whose first step ended on its own continue? Only answers are generated; no thinking, no image re-insertion.

Context (TSR's, image in the prompt only): [prompt][z_1][sep] </think>, z_1 / sep = the stored step 1 re-tokenised as the main
run does (diag/step1_think.split_stored; sep = cut_sep after a cut step).
  short  + '\\n\\n\\boxed' (the main run's probe), n_short answers at T = 1 without top-k/top-p, ending when the box closes
         (<= probe_cap tokens); per answer the main run's H (inject_sensitivity.sample_H: summed token entropies, the closing
         tokens merged inside the box; None = invalid by the main run's probe rule ('probe_valid') = infinite). H follows the
         main run's probe validity, not the value rule below: a probe the main run would accept keeps its finite H.
  free   the natural answer right after </think>, n_free answers at T = 1 without top-k/top-p, ending at EOS (<= answer_cap
         tokens); no answer forcing. Every token's full-vocabulary entropy is recorded (ent_sum over all generated positions,
         EOS included; ent_mean = ent_sum / n_tokens).
Random numbers: answer j of kind k uses seed_of(seed, pid, k, j).
Value of an answer (what the analysis counts): the prediction of the main run's scorer (visref/scoring.score, v4) on the
answer text ('\\boxed' + text for a short answer), kept only when it is not None and, for multiple choice, the extraction is a
letter of one of the choices (rule '...letter') or exactly one of the choices (case and outer spaces ignored): the scorer would otherwise map any
text to the closest choice (value_of). Validity:
  short  the box closes and holds a letter or digit (step1_think.answer_record, the main run's probe validity), and a value
  free   ended at EOS, its last '\\boxed{...}' closes and holds a letter or digit, and a value
An invalid answer has value None; the analysis never drops it (diag/analyze_stop_form.py).
Rows (JSONL): pid, status 'ok', set, stratum, stored_H, question_type, answer_type, answer_gt, recon, short [..], free [..],
code_version, scorer_version, cfg, timing. Errors: {pid, status 'error', stage, error}. No decision here.

  python3 diag/stop_form.py --parquet PQ --out results/stop_form_0.jsonl [--set main|trial] [--ids diag/stop_form_ids.json]
      [--limit N] [--shard 0/2] [--skip DONE.json] [--time-limit SECONDS] [--group 4]
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'diag'))
from blank_think import device_of, select_pids  # noqa: E402
from closed_multistep import ChunkError, error_row, make_generate, read_rows  # noqa: E402
from inject_sensitivity import T1, sample_H  # noqa: E402
from step1_think import answer_record, box_close_index, seed_of, split_stored  # noqa: E402
from visref.engine import Engine, Request  # noqa: E402
from visref.scoring import SCORER_VERSION, _boxed, score  # noqa: E402

KINDS = ('short', 'free')
DEFAULTS = dict(n_short=16, n_free=16, probe_cap=16, answer_cap=1024, probe_prompt='\n\n\\boxed', wait_text='Wait, think more',
                temperature=0.6, top_k=50, top_p=0.95, seed=42, cut_sep='\n', chunk=0, batch_tokens=72000, group=4,
                max_num=12)  # chunk 0 = answer_cap


def code_version():
    """sha256 over this script, the diag helpers it imports, run.py and visref/*.py."""
    h = hashlib.sha256()
    files = [Path(__file__).resolve(), *(ROOT / 'diag' / f for f in ('blank_think.py', 'closed_multistep.py', 'step1_think.py',
                                                                    'inject_sensitivity.py')),
             ROOT / 'run.py', *(ROOT / 'visref').glob('*.py')]
    for f in sorted(files):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:12]


def resolve_cfg(cfg):
    cfg = dict(cfg)
    if cfg['chunk'] == 0:
        cfg['chunk'] = max(cfg['answer_cap'], cfg['probe_cap'])
    if cfg['chunk'] < max(cfg['answer_cap'], cfg['probe_cap']):
        raise SystemExit(f"--chunk {cfg['chunk']} must be >= --answer-cap {cfg['answer_cap']} and --probe-cap")
    return cfg


def answer_context(it, eng, sp):
    """[prompt][z][sep] </think>: the context the main run answers from after step 1 (TSR, no injection)."""
    return torch.cat([it['base'], eng.embed(it['z'] + it['sep'] + [sp['think_end']])], 0)


def value_of(text, row):
    """(value or None, scorer record): the scorer's prediction, None for a multiple-choice extraction that is neither a letter
    nor exactly one of the choices."""
    sc = score(text, row)
    pred = sc['prediction']
    if pred is None:
        return None, sc
    if row['question_type'] == 'multi_choice':
        ext = '' if sc['extracted'] is None else str(sc['extracted']).strip()
        letters = {chr(ord('A') + i) for i in range(len(row['choices']))}
        if not ((str(sc['rule']).endswith('letter') and ext in letters)
                or ext.lower() in {str(c).strip().lower() for c in row['choices']}):
            return None, sc
    return str(pred), sc


def short_record(tok, sp, r, row):
    rec = answer_record(tok, sp, r.ids, row)
    h = sample_H(r, rec, sp)
    value = value_of('\\boxed' + rec['text'], row)[0] if rec['valid'] else None
    return {'text': rec['text'], 'probe_valid': rec['valid'], 'valid': value is not None, 'value': value, 'correct': rec['correct'],
            'H': None if math.isinf(h) else round(h, 6), 'n_tokens': len(r.ids)}


def free_record(tok, sp, r, row):
    eos, te = sp['eos'], sp['think_end']
    ids = [t for t in r.ids if t != eos]
    cut = [i for i, t in enumerate(ids) if t == te]
    text = tok.decode(ids[cut[-1] + 1:] if cut else ids)  # scored text: after the last </think> it writes, as methods.py
    b = _boxed(text)
    boxed = bool(b) and any(c.isalnum() for c in text[b[0]:b[1]])
    value, sc = value_of(text, row)
    ended = r.finish == 'stop'
    valid = ended and boxed and value is not None
    ent = [float(e) for e in r.ent]
    return {'text': text, 'finish': 'eos' if ended else 'cap', 'boxed': boxed, 'extracted': sc['extracted'], 'rule': sc['rule'],
            'prediction': None if sc['prediction'] is None else str(sc['prediction']), 'valid': valid,
            'value': value if valid else None, 'correct': int(valid and bool(sc['correct'])),
            'n_tokens': len(r.ids), 'ent_sum': round(sum(ent), 5), 'ent_mean': round(sum(ent) / len(ent), 6) if ent else None,
            'n_think_end': len(cut)}


def requests_of(it, eng, sp, cfg):
    """[(kind, j, Request)] for one item: n_short probe answers and n_free natural answers."""
    dev = device_of(eng)
    ctx = answer_context(it, eng, sp)
    short_ctx = torch.cat([ctx, eng.embed(sp['probe_prompt'])], 0)
    stop = lambda ids, bt=sp['brace_text']: box_close_index(ids, bt) is not None  # noqa: E731
    g = lambda k, j: torch.Generator(device=dev).manual_seed(seed_of(cfg['seed'], it['pid'], k, j))  # noqa: E731
    out = [('short', j, Request(short_ctx, frozenset({sp['eos']}), cfg['probe_cap'], g('short', j), record_entropy=True,
                                sampling=T1, stop_check=stop, ent_lump=sp['close_ids'])) for j in range(cfg['n_short'])]
    out += [('free', j, Request(ctx, frozenset({sp['eos']}), cfg['answer_cap'], g('free', j), record_entropy=True, sampling=T1))
            for j in range(cfg['n_free'])]
    return out


def run_group(items, eng, tok, sp, cfg, generate):
    """All answers of all items, one batched call per kind (run.pack budgets a group by its last request's context + max_new,
    so short and free requests are never packed together). Fills it['short'], it['free'] (lists in j order), it['nonfinite']."""
    by_kind = {k: ([], []) for k in KINDS}
    for it in items:
        it.update(nonfinite=False, short=[None] * cfg['n_short'], free=[None] * cfg['n_free'])
        for k, j, r in requests_of(it, eng, sp, cfg):
            by_kind[k][0].append((it, j))
            by_kind[k][1].append(r)
    t = time.time()
    for k, (keys, reqs) in by_kind.items():
        for (it, j), r in zip(keys, generate(reqs) if reqs else []):
            if r is None:
                it['nonfinite'] = True
                continue
            it[k][j] = (short_record if k == 'short' else free_record)(tok, sp, r, it['row'])
    for it in items:
        it['timing'] = {'group_answers_s': round(time.time() - t, 2)}


def row_out(it, cfg, version, set_name, doc):
    if it['nonfinite']:
        return error_row(it['pid'], 'generate', 'non-finite logits', version, set_name)
    row = it['row']
    return {'pid': it['pid'], 'status': 'ok', 'set': set_name, 'stratum': doc['stratum'][it['pid']],
            'stored_H': doc['stored_H'][it['pid']], 'question_type': row.get('question_type'),
            'answer_type': row.get('answer_type'), 'answer_gt': row['answer'], 'recon': it['recon'],
            'short': it['short'], 'free': it['free'], 'code_version': version, 'scorer_version': SCORER_VERSION, 'cfg': dict(cfg),
            'timing': {'encode_s': it.get('encode_s'), **it['timing']}}


def process_group(pids, make_item, eng, tok, sp, cfg, generate, version, set_name, doc):
    rows, items = {}, []
    for pid in pids:
        t0 = time.time()
        try:
            it = make_item(pid)
            it['encode_s'] = round(time.time() - t0, 2)
            items.append(it)
        except Exception as e:
            rows[pid] = error_row(pid, 'load', f'{type(e).__name__}: {e}', version, set_name)
    try:
        run_group(items, eng, tok, sp, cfg, generate)
    except ChunkError:
        raise
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for it in items:
            try:
                run_group([it], eng, tok, sp, cfg, generate)
            except ChunkError:
                raise
            except Exception as e:
                rows[it['pid']] = error_row(it['pid'], 'generate', f'{type(e).__name__}: {e}', version, set_name)
    for it in items:
        if it['pid'] not in rows:
            rows[it['pid']] = row_out(it, cfg, version, set_name, doc)
    return [rows[p] for p in pids]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--model', default='OpenGVLab/InternVL3_5-1B')
    ap.add_argument('--revision', default='2f71cf52542334823e48a46ffba0e2bc9add3446')
    ap.add_argument('--samples', default=str(ROOT / 'diag/step1_store_bmain.json'))
    ap.add_argument('--ids', default=str(ROOT / 'diag/stop_form_ids.json'))
    ap.add_argument('--set', default='main', choices=('main', 'trial'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--skip', default='', help='JSON list of pids finished earlier')
    ap.add_argument('--time-limit', type=float, default=0, help='seconds; stop starting new groups after this')
    for k, v in DEFAULTS.items():
        ap.add_argument('--' + k.replace('_', '-'), type=type(v), default=v)
    a = ap.parse_args(argv)
    a.cfg = resolve_cfg({k: getattr(a, k) for k in DEFAULTS})
    return a


def main(argv=None):
    a = parse_args(argv)
    cfg = a.cfg
    t0 = time.time()
    from run import fast_kv_selfcheck, pack
    from visref.model import encode_sample, load_mathvista, load_model, special_ids
    samples = json.loads(Path(a.samples).read_text())
    samples = samples.get('pids', samples)
    doc = json.loads(Path(a.ids).read_text())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r['pid'] for r in read_rows(out) if r.get('status') == 'ok'}  # error rows are run again
    n_done = len(done)
    if a.skip:
        done |= {str(x) for x in json.loads(Path(a.skip).read_text())}
    todo = select_pids(doc, a.set, a.limit, a.shard, done)
    version = code_version()
    print(json.dumps({'event': 'start', 'code_version': version, 'set': a.set, 'todo': len(todo), 'done': n_done, 'cfg': cfg}),
          flush=True)
    rows = load_mathvista(a.parquet)
    torch.manual_seed(cfg['seed'])
    torch.set_grad_enabled(False)
    model, tok = load_model(a.model, a.revision)
    lm = model.language_model
    eng = Engine(lm, cfg['temperature'], cfg['top_k'], cfg['top_p'], fast_kv=True)
    ok, info = fast_kv_selfcheck(lm, tok)
    print(json.dumps({'event': 'fast_kv_check', 'ok': ok, **info}), flush=True)
    if not ok:
        from visref.fastkv import set_enabled
        set_enabled(False)
        eng.fast_kv = False
    sp = special_ids(tok, cfg['wait_text'], cfg['probe_prompt'])
    generate = make_generate(eng, cfg, pack)

    def make_item(pid):
        row = rows[pid]
        base, _ = encode_sample(model, tok, row['image'](), row['query'], cfg['max_num'])
        z, sep, _, recon = split_stored(tok, sp, samples[pid], cfg['cut_sep'])
        if recon['think_end_stored'] != doc['stratum'][pid]:
            raise ValueError(f"stored step end {recon['think_end_stored']} differs from the ids file {doc['stratum'][pid]}")
        return {'pid': pid, 'row': row, 'base': base, 'z': z, 'sep': sep, 'recon': recon}

    n_err = 0
    with out.open('a') as f:
        for gi in range(0, len(todo), cfg['group']):
            if a.time_limit and time.time() - t0 > a.time_limit:
                print(json.dumps({'event': 'time_limit', 'left': len(todo) - gi}), flush=True)
                break
            tg = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            out_rows = process_group(todo[gi:gi + cfg['group']], make_item, eng, tok, sp, cfg, generate, version, a.set, doc)
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                if r['status'] != 'ok':
                    n_err += 1
                    print(json.dumps({'event': 'error', 'pid': r['pid'], 'error': r['error']}), flush=True)
            f.flush()
            print(json.dumps({'event': 'group', 'n': len(out_rows), 'n_error': sum(r['status'] != 'ok' for r in out_rows),
                              'group_s': round(time.time() - tg, 1), 'elapsed': round(time.time() - t0),
                              'left': len(todo) - gi - len(out_rows),
                              'peak_alloc_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else 0}),
                  flush=True)
    print(json.dumps({'event': 'end', 'elapsed': round(time.time() - t0), 'n_error': n_err}), flush=True)


if __name__ == '__main__':
    main()
