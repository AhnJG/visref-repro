"""full_think -- pre-registered diagnostic: do the image tokens help a question whose first step was cut at the cap?

Questions: the stored step 1 (diag/step1_store_bmain.json) of the question was cut at 1,024 tokens (think_end == 'cap');
lists and donors in diag/full_think_ids.json (diag/full_think_ids.py; sets cap_all 641, cap_continued, cap_stopped,
pilot_long). Start from the stored z_1 exactly as diag/step1_think.py does (split_stored: same base, same z, same separator
sep). Conditions (--conds, default A0,C,F; L only when asked):
  A0  [base][z_1][sep] </think> probe                 answer at once: step1_branch condition A (diag/step1_branch.py L133, L136, L151-L153, L160:
                                                      zc = [base][embed(z + sep)], then + embed([think_end]), then the probe)
  C   [base][z_1][sep] + wait                         wait = 'Wait, think more'; free thinking <= think_extra tokens (stops at
                                                      </think> / EOS), then close_thinking, then the probe
  F   [base][z_1][sep] + V[all image tokens, original order, untagged] + wait         (= inject 'bmain' with idx = range(n_visual))
  L   [base][z_1][sep] + V_donor + wait               V_donor = the donor question's image tokens (same number n_visual; the donor
                                                      rule is diag/inject_sensitivity_ids.donors; cropped to the first n_visual
                                                      tokens when the donor has more)
The probe = the stopping-probe style answer: '\\n\\n\\boxed', stops when the box closes, <= probe_cap tokens; n_t1 samples at
T=1 without top-k/top-p + 1 greedy (answer_record, box_close_index of step1_think).
cut-512 (for each free-thinking condition): when its extra thinking is longer than `cut` tokens, the first `cut` tokens are
closed with close_thinking(ids, 'cap') (a separator unless the text ends with one, then </think>) and the probe answers are
drawn on that context; when it was at most `cut` tokens long, the cut equals the full result (recorded, no new answers).
Seeds: the free thinking of C, F and L shares seed_of(seed, pid, 'think'); the answers of every context (A0, C, F, L and the
cut-512 contexts) use seed_of(seed, pid, 'ans', j).
Chunk fix: diag/step1_think.py calls Engine.generate(reqs, chunk) once and never re-submits rows that return finish == 'chunk';
close_thinking then treats such a row as 'cap', so a 2,048-token run with chunk 1,024 would silently stop at 1,024 tokens.
Here chunk defaults to think_extra (--chunk 0), a chunk below think_extra is refused at start, and every generate call raises
(ChunkError) if a result has finish 'chunk' (or anything but 'stop' / 'cap').
Errors: a question that raises gets a row {pid, status: 'error', error}; a question with non-finite logits in any condition
gets status 'error' as well (the analysis leaves it out). A ChunkError is a setup fault and stops the run. No accuracy
claim is made here (see diag/analyze_full_think.py).

  python3 diag/full_think.py --parquet PQ --out results/full_think_0.jsonl [--ids diag/full_think_ids.json] [--set cap_all]
      [--conds A0,C,F,L] [--extra-pids 12,34] [--samples diag/step1_store_bmain.json] [--limit 20] [--shard 0/2]
      [--skip DONE.json] [--time-limit SECONDS] [--think-extra 2048] [--cut 512] [--chunk 0] [--group 4]
"""
import argparse
import hashlib
import json
import resource
import sys
import time
import zlib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'diag'))
import step1_think as t1  # noqa: E402
from inject_sensitivity import length_matched  # noqa: E402
from step1_think import (DEFAULTS as T1_DEFAULTS, GREEDY, T1, answer_record, box_close_index, close_thinking,  # noqa: E402,F401
                         extra_record, inject_embeds, seed_of, split_stored)
from visref.engine import Engine, Request  # noqa: E402
from visref.scoring import SCORER_VERSION  # noqa: E402

ALL_CONDS = ('A0', 'C', 'F', 'L')
FREE = ('C', 'F', 'L')  # the conditions with free thinking
DEFAULT_CONDS = 'A0,C,F'
DEFAULTS = {**{k: v for k, v in T1_DEFAULTS.items() if k not in ('vis_frac', 'think_extra', 'chunk', 'group')},
            'think_extra': 2048, 'cut': 512, 'chunk': 0, 'group': 4}  # chunk 0 = think_extra (see resolve_cfg)
TAIL_CHARS = 3000


class ChunkError(RuntimeError):
    """A generate call returned a row that is neither finished ('stop') nor at its cap ('cap'), e.g. finish 'chunk'."""


def code_version():
    """sha256 over this script, the step1_think / inject_sensitivity helpers it imports, run.py and visref/*.py."""
    h = hashlib.sha256()
    files = [Path(__file__).resolve(), ROOT / 'diag/step1_think.py', ROOT / 'diag/inject_sensitivity.py', ROOT / 'run.py',
             *(ROOT / 'visref').glob('*.py')]
    for f in sorted(files):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:12]


def parse_conds(s):
    conds = [c.strip() for c in s.split(',') if c.strip()]
    bad = [c for c in conds if c not in ALL_CONDS]
    if bad or not conds or len(set(conds)) != len(conds):
        raise SystemExit(f'--conds must be a list without repeats of {ALL_CONDS}, got {s!r}')
    return tuple(c for c in ALL_CONDS if c in conds)


def resolve_cfg(cfg):
    """chunk 0 -> think_extra; refuse a chunk below think_extra (rows would come back unfinished and be read as 'cap')."""
    cfg = dict(cfg)
    if cfg['chunk'] == 0:
        cfg['chunk'] = cfg['think_extra']
    if cfg['chunk'] < cfg['think_extra']:
        raise SystemExit(f"--chunk {cfg['chunk']} < --think-extra {cfg['think_extra']}: refused (rows that return finish "
                         "'chunk' are not re-submitted, so the extra thinking would silently stop at the chunk size)")
    if cfg['cut'] < 1 or cfg['think_extra'] < 1 or cfg['probe_cap'] > cfg['chunk']:
        raise SystemExit('--cut and --think-extra must be >= 1 and --probe-cap <= --chunk')
    return cfg


def zlib_ratio(text, n=TAIL_CHARS):
    """zlib compressed size / size (utf-8 bytes) of the last n characters of text; None for an empty text. A low value
    means a repeating text."""
    tail = text[-n:]
    b = tail.encode('utf-8')
    return round(len(zlib.compress(b)) / len(b), 4) if b else None


def make_generate(eng, cfg, pack):
    """step1_think.make_generate plus the check that no result is unfinished: every row must end 'stop' or 'cap'."""
    inner = t1.make_generate(eng, cfg, pack)

    def generate(reqs):
        out = inner(reqs)
        bad = sorted({r.finish for r in out if r is not None and r.finish not in ('stop', 'cap')})
        if bad:
            raise ChunkError(f"generate returned unfinished rows (finish {bad}) with chunk={cfg['chunk']}: refusing to treat "
                             'them as cap')
        return out
    return generate


def build_contexts(it, eng, tok, sp, cfg, conds):
    """it['ctx'][c] (without the probe prompt) for c in conds from it['base'], it['vis'], it['z'], it['sep'] (and
    it['vis_other'] for L). F injects every image token in the original order, untagged (inject 'bmain', idx = range(n))."""
    base, vis, z, sep = it['base'], it['vis'], it['z'], it['sep']
    prefix = torch.cat([base, eng.embed(z + sep)], 0)
    wait = eng.embed(sp['wait'])
    n = int(vis.shape[0])
    ctx = {}
    if 'A0' in conds:
        ctx['A0'] = torch.cat([prefix, eng.embed([sp['think_end']])], 0)
    if 'C' in conds:
        ctx['C'] = torch.cat([prefix, wait], 0)
    if 'F' in conds:
        inj = inject_embeds(eng, tok, sp, 'bmain', vis, list(range(n)), tok.decode(z), it['sep_text'])
        ctx['F'] = torch.cat([prefix, inj, wait], 0)
    if 'L' in conds:
        if it.get('vis_other') is None:
            raise ValueError('condition L needs the donor image tokens (it["vis_other"])')
        ctx['L'] = torch.cat([prefix, length_matched(it['vis_other'], n), wait], 0)
    it['ctx'] = ctx


def reset(it):
    for k in ('ctx', 'pre', 'ctx_cut', 'extra', 'answers', 'cut', 'nonfinite'):
        it.pop(k, None)
    it.update(extra={}, answers={}, cut={}, nonfinite=[], pre={}, ctx_cut={})


def run_group(items, eng, tok, sp, cfg, generate, conds):
    """items: dicts with pid, row, base [L,d], vis [N,d], z, sep, sep_text (and vis_other for L). Adds in place: extra[c]
    (free-thinking record), cut[c] (cut-512 record), answers[c], nonfinite (the conditions whose generation returned None)."""
    te, eos = sp['think_end'], sp['eos']
    dev = eng.emb.weight.device
    cut = cfg['cut']
    for it in items:
        reset(it)
        build_contexts(it, eng, tok, sp, cfg, conds)
        it['pre'] = dict(it['ctx'])
    reqs, keys = [], []
    for it in items:  # free thinking: C, F, L share one seed per pid
        for c in FREE:
            if c in it['ctx']:
                g = torch.Generator(device=dev).manual_seed(seed_of(cfg['seed'], it['pid'], 'think'))
                reqs.append(Request(it['ctx'][c], frozenset({te, eos}), cfg['think_extra'], g))
                keys.append((it, c))
    for (it, c), r in zip(keys, generate(reqs) if reqs else []):
        if r is None:
            it['nonfinite'].append(c)
            it['extra'][c] = {'error': 'non-finite logits'}
            del it['ctx'][c]
            continue
        tail, end = close_thinking(tok, sp, r.ids, r.finish, cfg['cut_sep'])
        rec = extra_record(tok, sp, r.ids, end)
        rec['zlib_ratio_tail3000'] = zlib_ratio(rec['text'])
        it['extra'][c] = rec
        it['ctx'][c] = torch.cat([it['ctx'][c], eng.embed(tail)], 0)
        if rec['len'] <= cut:  # the thinking ended at or before the cut: the cut equals the full result
            it['cut'][c] = {'cut': cut, 'same_as_full': True, 'len_full': rec['len']}
        else:
            cids = list(r.ids[:cut])  # all thinking tokens: te / eos can only be the last token of a finished row
            ctail, _ = close_thinking(tok, sp, cids, 'cap', cfg['cut_sep'])
            it['cut'][c] = {'cut': cut, 'same_as_full': False, 'len_full': rec['len'], 'end': 'cap', 'ids': cids,
                            'text': tok.decode(cids)}
            it['ctx_cut'][c] = torch.cat([it['pre'][c], eng.embed(ctail)], 0)
    reqs, keys = [], []
    stop = lambda ids, bt=sp['brace_text']: box_close_index(ids, bt) is not None  # noqa: E731
    for it in items:
        todo = [(c, 'full', ctx) for c, ctx in it['ctx'].items()] + [(c, 'cut', ctx) for c, ctx in it['ctx_cut'].items()]
        for c, kind, ctx in todo:
            ctx = torch.cat([ctx, eng.embed(sp['probe_prompt'])], 0)
            for j in range(cfg['n_t1'] + 1):
                greedy = j == cfg['n_t1']
                g = None if greedy else torch.Generator(device=dev).manual_seed(seed_of(cfg['seed'], it['pid'], 'ans', j))
                reqs.append(Request(ctx, frozenset({eos}), cfg['probe_cap'], g, sampling=GREEDY if greedy else T1,
                                    stop_check=stop))
                keys.append((it, c, kind, j))
    for (it, c, kind, j), r in zip(keys, generate(reqs) if reqs else []):
        if r is None:
            rec = {'text': '', 'valid': False, 'correct': 0, 'error': True}
            it['nonfinite'].append(c if kind == 'full' else f'{c}_cut')
        else:
            rec = answer_record(tok, sp, r.ids, it['row'])
        rec['greedy'] = j == cfg['n_t1']
        if kind == 'full':
            it['answers'].setdefault(c, []).append(rec)
        else:
            it['cut'][c].setdefault('answers', []).append(rec)


def acc_greedy(ans, n_t1):
    return sum(a['correct'] for a in ans if not a['greedy']) / n_t1, next(a['correct'] for a in ans if a['greedy'])


def error_row(pid, stage, msg, version, set_name):
    return {'pid': pid, 'status': 'error', 'stage': stage, 'error': str(msg)[:500], 'set': set_name, 'code_version': version}


def row_out(it, cfg, version, conds, set_name):
    """The result row of one finished item; an error row when any generation of it returned non-finite logits."""
    if it['nonfinite']:
        row = error_row(it['pid'], 'generate', 'non-finite logits in ' + ','.join(sorted(set(it['nonfinite']))), version, set_name)
        row['nonfinite'] = sorted(set(it['nonfinite']))
        return row
    acc, greedy = {}, {}
    for c, ans in it['answers'].items():
        acc[c], greedy[c] = acc_greedy(ans, cfg['n_t1'])
    cut = {}
    for c, rec in it['cut'].items():
        rec = dict(rec)
        if rec['same_as_full']:  # no new answers: the cut equals the full result
            rec['acc_t1'], rec['greedy'] = acc[c], greedy[c]
        else:
            rec['acc_t1'], rec['greedy'] = acc_greedy(rec['answers'], cfg['n_t1'])
        cut[c] = rec
    return {'pid': it['pid'], 'status': 'ok', 'stratum': it['sample']['think_end'], 'origin': it.get('origin'),
            'set': set_name, 'code_version': version, 'scorer_version': SCORER_VERSION, 'conds': list(conds),
            'cfg': {k: cfg[k] for k in ('think_extra', 'cut', 'chunk', 'n_t1', 'probe_cap', 'seed', 'temperature', 'top_k',
                                        'top_p', 'cut_sep', 'wait_text', 'probe_prompt', 'max_num')},
            'n_visual': int(it['vis'].shape[0]), 'donor': it.get('donor_pid'),
            'donor_n_visual': it.get('donor_n_visual'), 'question_type': it['row'].get('question_type'),
            'z_ids': it['z'], 'sep_ids': it['sep'], 'recon': it['recon'], 'extra': it['extra'], 'answers': it['answers'],
            'acc_t1': acc, 'greedy': greedy, 'cut512': cut, 'answer_gt': it['row']['answer']}


def process_group(pids, make_item, eng, tok, sp, cfg, generate, conds, version, set_name):
    """Rows (ok or error) for pids, in order. A question that raises gets an error row and the rest goes on: when the
    group raises, every question is re-run alone. A ChunkError stops the run."""
    rows, items = {}, []
    for pid in pids:
        try:
            items.append(make_item(pid))
        except ChunkError:
            raise
        except Exception as e:
            rows[pid] = error_row(pid, 'load', f'{type(e).__name__}: {e}', version, set_name)
    try:
        run_group(items, eng, tok, sp, cfg, generate, conds)
    except ChunkError:
        raise
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for it in items:
            try:
                run_group([it], eng, tok, sp, cfg, generate, conds)
            except ChunkError:
                raise
            except Exception as e:
                rows[it['pid']] = error_row(it['pid'], 'generate', f'{type(e).__name__}: {e}', version, set_name)
    for it in items:
        if it['pid'] not in rows:
            rows[it['pid']] = row_out(it, cfg, version, conds, set_name)
    return [rows[p] for p in pids]


def select_pids(doc, set_name, limit=0, shard='0/1', extra=(), done=()):
    """The pids this process runs: the set's order, first `limit`, plus the extra pids (not already in), the shard's share
    of that list by index, without the done ones."""
    if set_name not in doc['sets']:
        raise SystemExit(f"no set {set_name!r} in the ids file (has {sorted(doc['sets'])})")
    ids = [str(p) for p in doc['sets'][set_name]]
    if limit:
        ids = ids[:limit]
    ids = ids + [p for p in dict.fromkeys(str(x) for x in extra) if p not in set(ids)]
    si, sn = map(int, shard.split('/'))
    ids = [p for i, p in enumerate(ids) if i % sn == si]
    done = {str(x) for x in done}
    return [p for p in ids if p not in done]


def donor_of(doc, set_name, pid):
    """The donor pid of pid for condition L: the set's map, else cap_all's (extra pids, pilot_long)."""
    for s in (set_name, 'cap_all'):
        if pid in doc['donor'].get(s, {}):
            return doc['donor'][s][pid]
    return None


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--model', default='OpenGVLab/InternVL3_5-1B')
    ap.add_argument('--revision', default='2f71cf52542334823e48a46ffba0e2bc9add3446')
    ap.add_argument('--samples', default=str(ROOT / 'diag/step1_store_bmain.json'))
    ap.add_argument('--ids', default=str(ROOT / 'diag/full_think_ids.json'))
    ap.add_argument('--set', default='cap_all', help='cap_all | cap_continued | cap_stopped | pilot_long')
    ap.add_argument('--conds', default=DEFAULT_CONDS, help='comma list of A0,C,F,L')
    ap.add_argument('--extra-pids', default='', help='comma list of pids run in addition to the set')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--skip', default='', help='JSON list of pids finished earlier')
    ap.add_argument('--time-limit', type=float, default=0, help='seconds; stop starting new groups after this')
    for k, v in DEFAULTS.items():
        ap.add_argument('--' + k.replace('_', '-'), type=type(v), default=v)
    a = ap.parse_args(argv)
    a.conds = parse_conds(a.conds)
    a.cfg = resolve_cfg({k: getattr(a, k) for k in DEFAULTS})
    return a


def read_rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def main(argv=None):
    a = parse_args(argv)
    cfg, conds = a.cfg, a.conds
    t0 = time.time()
    from run import fast_kv_selfcheck, pack
    from visref.model import encode_sample, load_mathvista, load_model, special_ids
    samples = json.loads(Path(a.samples).read_text())
    samples = samples.get('pids', samples)
    doc = json.loads(Path(a.ids).read_text())
    extra = [p for p in a.extra_pids.split(',') if p.strip()]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r['pid'] for r in read_rows(out)}
    n_done = len(done)
    if a.skip:
        done |= {str(x) for x in json.loads(Path(a.skip).read_text())}
    todo = select_pids(doc, a.set, a.limit, a.shard, extra, done)
    for p in todo:
        assert p in samples, f'no stored step 1 for {p}'
        assert samples[p]['think_end'] == 'cap', (p, samples[p]['think_end'])
        assert 'L' not in conds or donor_of(doc, a.set, p) is not None, f'no donor for {p}'
    version = code_version()
    print(json.dumps({'event': 'start', 'code_version': version, 'set': a.set, 'conds': list(conds), 'todo': len(todo),
                      'done': n_done, 'cfg': cfg}), flush=True)
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
        s, row = samples[pid], rows[pid]
        base, vis = encode_sample(model, tok, row['image'](), row['query'], cfg['max_num'])
        z, sep, sep_text, recon = split_stored(tok, sp, s, cfg['cut_sep'])
        it = {'pid': pid, 'row': row, 'sample': s, 'base': base, 'vis': vis, 'z': z, 'sep': sep, 'sep_text': sep_text,
              'recon': recon, 'origin': doc.get('origin', {}).get(pid)}
        if 'L' in conds:
            d = donor_of(doc, a.set, pid)
            o = rows[d]
            it['vis_other'] = encode_sample(model, tok, o['image'](), o['query'], cfg['max_num'])[1]
            assert it['vis_other'].shape[0] >= vis.shape[0], (pid, d, it['vis_other'].shape[0], vis.shape[0])
            it['donor_pid'], it['donor_n_visual'] = d, int(it['vis_other'].shape[0])
        return it

    peak_all, n_err = 0.0, 0
    with out.open('a') as f:
        for gi in range(0, len(todo), cfg['group']):
            if a.time_limit and time.time() - t0 > a.time_limit:
                print(json.dumps({'event': 'time_limit', 'left': len(todo) - gi}), flush=True)
                break
            tg = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            out_rows = process_group(todo[gi:gi + cfg['group']], make_item, eng, tok, sp, cfg, generate, conds, version, a.set)
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                if r['status'] != 'ok':
                    n_err += 1
                    print(json.dumps({'event': 'error', 'pid': r['pid'], 'error': r['error']}), flush=True)
            f.flush()
            peak = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else 0
            peak_all = max(peak_all, peak)
            print(json.dumps({'event': 'group', 'n': len(out_rows), 'n_error': sum(r['status'] != 'ok' for r in out_rows),
                              'group_s': round(time.time() - tg, 1), 'elapsed': round(time.time() - t0),
                              'left': len(todo) - gi - len(out_rows), 'peak_alloc_gb': peak,
                              'max_rss_gb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)}), flush=True)
    print(json.dumps({'event': 'end', 'elapsed': round(time.time() - t0), 'n_error': n_err, 'peak_alloc_gb': peak_all,
                      'max_rss_gb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)}), flush=True)


if __name__ == '__main__':
    main()
