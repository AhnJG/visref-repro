"""closed_multistep -- pre-registered diagnostic (plan: closed_multistep_plan.md; this file is code only, written before any result).

Question: for the 334 questions whose stored first thinking step closed by itself (diag/closed_multistep_ids.json 'main'), does
a VisRef-style continuation (after every step the DPP-picked image tokens are put back, Algorithm 1 with H_k always above
delta) answer better than a text-only continuation (TSR) and than answering at once, when every question is continued to
exactly N_STEPS = 4 thinking steps (the stored first step + 3 generated ones)?

Start: the stored first step re-tokenised exactly as diag/step1_think.py (A7) does it (split_stored: z_1, separator, recon flag).
Conditions (all answer with '</think>' + '\\n\\n\\boxed': n_t1 samples at T=1 without top-k/top-p + 1 greedy, <= probe_cap
tokens, stopped when the box closes, answer_record = rule scorer v4):
  S       [base][z_1][sep] </think>                                         answer at once (the ST place)
  T       [base][z_1][sep] Wait, think more [z_2][sep] Wait, think more [z_3][sep] Wait, think more [z_4][sep] </think>
  V       like T, but after every step (also the 4th) the image block of methods.py `visref_hs_imgtag_in` (TAG_IN) is put in:
          [\\n unless decode(z)+sep ends with \\n] <img> V[sel] </img> \\n, sel = select_tokens(vis, Z, m); Z = last-layer
          hidden states of z_k in its own context (Engine.last_hidden), V = the input image embeddings (NOT hidden states),
          m = floor(vis_frac * |V|); no block after an empty step text. Steps 1-3 continue with 'Wait, think more'; after the
          4th block comes '</think>'. Earlier blocks stay in the context.
  Vminus  (descriptive) V's 4-step text WITHOUT the last block, then '</think>' (no generation; also V's step-4 probe).
Step ends (methods.Job._end_think): </think> closes the step (dropped, thinking goes on); a step cut at think_cap (1,024) gets
cut_sep '\\n' unless its text ends with one; EOS ends the step (dropped). chunk >= think_cap is required (resolve_cfg) and a
generate result that is neither 'stop' nor 'cap' raises ChunkError (as full_think / thumb_gate).
Seeds: per condition ONE generator seed_of(seed, pid, 'think') used across all its steps (T and V share the seed; the hash of its
state at the start of every step is recorded: steps[c][k-1]['rng_hash']); the probe after step k (k = 1..3, T and V, context
before the block + '</think>' + '\\n\\n\\boxed', 1 sample at T=1, entropies recorded) seed_of(seed, pid, 'probe', k); final answer j
seed_of(seed, pid, 'ans', j). The step-4 probe is not sampled apart: it is the first sampled final answer of T / Vminus
(probes[c][3], source 'answer0', with its entropies).
Rows (JSONL, one per question): pid, status 'ok', set, stratum, question_type, n_visual, m, recon, code_version, scorer_version, cfg,
steps {T, V: 4 records {k, source, end, len, ids, sep_text, rng_hash [, V: m, n_sel, sel, block, block_len]}}, probes {T, V: 4
records}, answers {S, T, V, Vminus: n_t1 sampled + 1 greedy}, acc_t1, n_correct, greedy, answer_gt, timing. A question that
raises, or whose generation returns non-finite logits, gets a row {pid, status: 'error', stage, error, set, code_version}.
No accuracy and no decision is computed here (see diag/analyze_closed_multistep.py). Trial mode (--set trial) also runs the
memory check: one 3,328-image question, a ~12,000-token context (the question's base + its stored text repeated), Engine.last_hidden
+ select_tokens + a few generated tokens; the peak memory is written to OUT.memcheck.json.

  python3 diag/closed_multistep.py --parquet PQ --out results/closed_multistep_0.jsonl [--set main|trial] [--pids-part 1|2|all]
      [--ids diag/closed_multistep_ids.json] [--samples diag/step1_store_bmain.json] [--limit N] [--extra-pids 12,34]
      [--shard 0/2] [--skip DONE.json] [--time-limit SECONDS] [--think-cap 1024] [--chunk 0] [--group 4]
  Questions already in --out are skipped (resume).
"""
import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'diag'))
import step1_think as t1  # noqa: E402
from step1_think import GREEDY, T1, answer_record, box_close_index, seed_of, split_stored  # noqa: E402
from visref.dpp import budget, select_tokens  # noqa: E402
from visref.engine import Engine, Request  # noqa: E402
from visref.scoring import SCORER_VERSION  # noqa: E402

N_STEPS = 4  # the stored first step + 3 generated ones; fixed by the plan
STEP_CONDS = ('T', 'V')  # the conditions that think on
CONDS = ('S', 'T', 'V', 'Vminus')  # the conditions that answer
DEFAULTS = dict(vis_frac=0.3, think_cap=1024, probe_prompt='\n\n\\boxed', wait_text='Wait, think more', probe_cap=16, n_t1=3,
                temperature=0.6, top_k=50, top_p=0.95, seed=42, cut_sep='\n', chunk=0, batch_tokens=72000, group=4,
                max_num=12, memcheck_tokens=12000)  # chunk 0 = think_cap (see resolve_cfg)


class ChunkError(RuntimeError):
    """A generate call returned a row that is neither finished ('stop') nor at its cap ('cap'), e.g. finish 'chunk'."""


def code_version():
    """sha256 over this script, the step1_think helpers it imports, run.py and visref/*.py."""
    h = hashlib.sha256()
    files = [Path(__file__).resolve(), ROOT / 'diag/step1_think.py', ROOT / 'run.py', *(ROOT / 'visref').glob('*.py')]
    for f in sorted(files):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:12]


def resolve_cfg(cfg):
    """chunk 0 -> think_cap; refuse a chunk below think_cap (rows would come back unfinished and be read as 'cap')."""
    cfg = dict(cfg)
    if cfg['chunk'] == 0:
        cfg['chunk'] = cfg['think_cap']
    if cfg['chunk'] < cfg['think_cap']:
        raise SystemExit(f"--chunk {cfg['chunk']} < --think-cap {cfg['think_cap']}: refused (rows that return finish "
                         "'chunk' are not re-submitted, so a step would silently stop at the chunk size)")
    if cfg['think_cap'] < 1 or cfg['probe_cap'] < 1 or cfg['probe_cap'] > cfg['chunk'] or cfg['n_t1'] < 1:
        raise SystemExit('--think-cap, --probe-cap and --n-t1 must be >= 1 and --probe-cap <= --chunk')
    return cfg


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


# ---- pieces ----------------------------------------------------------------------------------------------------------
def rng_hash(g):
    """Short hash of a torch.Generator's state."""
    return hashlib.sha256(g.get_state().numpy().tobytes()).hexdigest()[:12]


def split_generated(tok, sp, ids, cut_sep):
    """One generated thinking step -> (z, sep, sep_text, end): 'close' when the last id is </think>, 'eos' when it is EOS (z
    drops it), else 'cap' (z = all ids; sep = cut_sep unless the text ends with a newline). methods.Job._end_think."""
    ids = list(ids)
    if ids and ids[-1] == sp['think_end']:
        end, z = 'close', ids[:-1]
    elif ids and ids[-1] == sp['eos']:
        end, z = 'eos', ids[:-1]
    else:
        end, z = 'cap', ids
    sep_text = cut_sep if end == 'cap' and cut_sep and not tok.decode(z).endswith('\n') else ''
    sep = tok.encode(sep_text, add_special_tokens=False) if sep_text else []
    return z, sep, sep_text, end


def v_block(eng, tok, sp, vis, sel, z_text, sep_text):
    """The image block of methods.Job._end_think for TAG methods: [\\n unless decode(z)+sep ends with \\n] <img> vis[sel] </img>
    \\n. Built also for an empty selection (the tags stay), as methods.py does; the caller skips an empty step text."""
    nl = tok.encode('\n', add_special_tokens=False)
    pre = [] if (z_text + sep_text).endswith('\n') else nl  # continuing: exactly one newline before <img>
    picked = vis[sel] if len(sel) else vis[:0]
    return torch.cat([eng.embed(pre), eng.embed([sp['img_start']]), picked, eng.embed([sp['img_end']] + nl)], 0)


def box_depths(ids, brace_text):
    """Brace depth before each token of a box answer after a prompt ending in '\\boxed' (Job._brace_scan)."""
    depth, opened, before = 0, False, []
    for t in ids:
        before.append(depth)
        for ch in brace_text.get(t, ''):
            if ch == '{':
                depth, opened = depth + 1, True
            elif ch == '}' and opened:
                depth -= 1
            if opened and depth == 0:
                return before
    return before


def entropy_of(sp, r, valid):
    """(H, per-token entropies) of a sampled answer: the close-merged entropy at box depth 1, the full one elsewhere, summed
    (Job._probe_entropy); H is None for an invalid answer (the main run's inf)."""
    if not r.ent or len(r.ent) != len(r.ids) or len(r.ent_lump) != len(r.ids):
        return None, []
    depth = box_depths(r.ids, sp['brace_text'])
    ent = [l if d == 1 else f for f, l, d in zip(r.ent, r.ent_lump, depth)]
    return (round(sum(ent), 5) if valid else None), [round(e, 4) for e in ent]


def read_answer(tok, sp, r, row, greedy):
    """answer_record of one answer request result plus greedy flag, token count and (sampled answers) H and entropies."""
    if r is None:
        return {'text': '', 'valid': False, 'correct': 0, 'error': True, 'greedy': greedy}
    rec = answer_record(tok, sp, r.ids, row)
    rec.update(greedy=greedy, n_tok=len(r.ids))
    if not greedy:
        rec['H'], rec['ent'] = entropy_of(sp, r, rec['valid'])
    return rec


def device_of(eng):
    emb = getattr(eng, 'emb', None)
    return emb.weight.device if emb is not None else torch.device('cpu')


# ---- one group of questions ------------------------------------------------------------------------------------------
def reset(it, eng, cfg):
    """(Re)start the per-question state of it from its base / vis / stored first step."""
    for k in ('st', 'ans_ctx', 'answers', 'nonfinite', 'timing', 'm', 'ctx1'):
        it.pop(k, None)
    dev = device_of(eng)
    it['m'] = budget(int(it['vis'].shape[0]), cfg['vis_frac'])
    it['nonfinite'], it['timing'], it['answers'], it['ans_ctx'] = [], {}, {}, {}
    it['st'] = {c: {'start': it['base'], 'steps': [], 'probes': [],
                    'gen': torch.Generator(device=dev).manual_seed(seed_of(cfg['seed'], it['pid'], 'think'))} for c in STEP_CONDS}


def close_step(it, c, k, parts, rng_h, eng, tok, sp):
    """Condition c ends its step k with the text parts = (z, sep, sep_text, end, source): the context after the step
    (s['end_ctx']), for V the picked image tokens and the context with the block (s['step_ctx']), and the step record."""
    s = it['st'][c]
    z, sep, sep_text, end, source = parts
    end_ctx = torch.cat([s['start'], eng.embed(z + sep)], 0)
    s['end_ctx'], s['step_ctx'] = end_ctx, end_ctx
    rec = {'k': k, 'source': source, 'end': end, 'len': len(z), 'ids': [int(t) for t in z], 'sep_text': sep_text, 'rng_hash': rng_h}
    if c == 'V':
        rec['m'] = it['m']
        if z:  # methods.py: `if self.method in VISREF and z`
            Z = eng.last_hidden(torch.cat([s['start'], eng.embed(z)], 0), len(z))  # z in its own context
            sel = [int(i) for i in select_tokens(it['vis'], Z, it['m'])]
            block = v_block(eng, tok, sp, it['vis'], sel, tok.decode(z), sep_text)
            s['step_ctx'] = torch.cat([end_ctx, block], 0)
            rec.update(n_sel=len(sel), sel=sel, block=True, block_len=int(block.shape[0]))
        else:
            rec.update(n_sel=0, sel=[], block=False, block_len=0)
    s['steps'].append(rec)


def run_group(items, eng, tok, sp, cfg, generate, timing=None):
    """items: dicts with pid, row, base [L,d], vis [N,d], z, sep, sep_text, recon. Runs the four steps of T and V, the probes and
    the final answers of S, T, V, Vminus in place (it['st'][c]['steps' | 'probes'], it['answers'], it['nonfinite']).
    timing (dict) gets the group's seconds of 'steps', 'probes', 'answers'."""
    te, eos = sp['think_end'], sp['eos']
    dev = device_of(eng)
    timing = {} if timing is None else timing
    for k in ('steps', 'probes', 'answers'):
        timing[k] = 0.0
    stop = lambda ids, bt=sp['brace_text']: box_close_index(ids, bt) is not None  # noqa: E731
    probe_prompt = sp['probe_prompt']
    for it in items:
        reset(it, eng, cfg)
        it['ctx1'] = torch.cat([it['base'], eng.embed(it['z'] + it['sep'])], 0)
    live = lambda: [it for it in items if not it['nonfinite']]  # noqa: E731
    for k in range(1, N_STEPS + 1):
        t = time.time()
        got = {}  # (pid, c) -> (parts, rng_hash)
        if k == 1:  # the stored step, the same text for T and V
            for it in live():
                for c in STEP_CONDS:
                    parts = (it['z'], it['sep'], it['sep_text'], it['recon'].get('think_end_stored', 'close'), 'stored')
                    got[(it['pid'], c)] = (parts, rng_hash(it['st'][c]['gen']))
        else:
            reqs, keys = [], []
            for it in live():
                for c in STEP_CONDS:
                    s = it['st'][c]
                    reqs.append(Request(s['start'], frozenset({te, eos}), cfg['think_cap'], s['gen']))
                    keys.append((it, c, rng_hash(s['gen'])))
            for (it, c, h), r in zip(keys, generate(reqs) if reqs else []):
                if r is None:
                    if c not in it['nonfinite']:
                        it['nonfinite'].append(c)
                    continue
                got[(it['pid'], c)] = ((*split_generated(tok, sp, r.ids, cfg['cut_sep']), 'generated'), h)
        for it in live():
            for c in STEP_CONDS:
                parts, h = got[(it['pid'], c)]
                close_step(it, c, k, parts, h, eng, tok, sp)
        timing['steps'] += time.time() - t
        t = time.time()
        if k < N_STEPS:  # the probe the model would give if it stopped now: context before the block
            reqs, keys = [], []
            for it in live():
                for c in STEP_CONDS:
                    ctx = torch.cat([it['st'][c]['end_ctx'], eng.embed([te] + probe_prompt)], 0)
                    g = torch.Generator(device=dev).manual_seed(seed_of(cfg['seed'], it['pid'], 'probe', k))
                    reqs.append(Request(ctx, frozenset({eos}), cfg['probe_cap'], g, record_entropy=True, sampling=T1,
                                        stop_check=stop, ent_lump=sp['close_ids']))
                    keys.append((it, c))
            for (it, c), r in zip(keys, generate(reqs) if reqs else []):
                if r is None:
                    it['nonfinite'].append(f'{c}_probe{k}')
                    continue
                rec = read_answer(tok, sp, r, it['row'], False)
                rec.update(k=k, source='probe')
                it['st'][c]['probes'].append(rec)
            for it in live():  # the next step starts after the block (V) and the continuation cue
                for c in STEP_CONDS:
                    s = it['st'][c]
                    s['start'] = torch.cat([s['step_ctx'], eng.embed(sp['wait'])], 0)
                    del s['end_ctx'], s['step_ctx']
        timing['probes'] += time.time() - t
    t = time.time()  # final answers of S, T, V, Vminus
    for it in live():
        T, V = it['st']['T'], it['st']['V']
        it['ans_ctx'] = {'S': it['ctx1'], 'T': T['end_ctx'], 'V': V['step_ctx'], 'Vminus': V['end_ctx']}
    reqs, keys = [], []
    for it in live():
        for c in CONDS:
            ctx = torch.cat([it['ans_ctx'][c], eng.embed([te] + probe_prompt)], 0)
            for j in range(cfg['n_t1'] + 1):
                greedy = j == cfg['n_t1']
                g = None if greedy else torch.Generator(device=dev).manual_seed(seed_of(cfg['seed'], it['pid'], 'ans', j))
                reqs.append(Request(ctx, frozenset({eos}), cfg['probe_cap'], g, record_entropy=not greedy,
                                    sampling=GREEDY if greedy else T1, stop_check=stop,
                                    ent_lump=frozenset() if greedy else sp['close_ids']))
                keys.append((it, c, j))
    for (it, c, j), r in zip(keys, generate(reqs) if reqs else []):
        if r is None:
            if c not in it['nonfinite']:
                it['nonfinite'].append(c)
            continue
        it['answers'].setdefault(c, []).append(read_answer(tok, sp, r, it['row'], j == cfg['n_t1']))
    timing['answers'] = time.time() - t
    for it in items:
        it['timing'] = {f'group_{k}_s': round(timing[k], 2) for k in ('steps', 'probes', 'answers')}
    for it in live():  # the step-4 probe is the first sampled final answer of T / Vminus
        for c, src in (('T', 'T'), ('V', 'Vminus')):
            a = it['answers'][src][0]
            it['st'][c]['probes'].append({**{kk: vv for kk, vv in a.items() if kk != 'greedy'}, 'k': N_STEPS,
                                          'source': f'answer0_of_{src}'})
        for c in STEP_CONDS:
            it['st'][c].pop('end_ctx', None)
            it['st'][c].pop('step_ctx', None)
        it['ans_ctx'] = {}


# ---- rows ------------------------------------------------------------------------------------------------------------
def error_row(pid, stage, msg, version, set_name):
    return {'pid': pid, 'status': 'error', 'stage': stage, 'error': str(msg)[:500], 'set': set_name, 'code_version': version}


def row_out(it, cfg, version, set_name):
    """The result row of one finished item; an error row when any generation of it returned non-finite logits."""
    if it['nonfinite']:
        row = error_row(it['pid'], 'generate', 'non-finite logits in ' + ','.join(sorted(set(it['nonfinite']))), version, set_name)
        row['nonfinite'] = sorted(set(it['nonfinite']))
        return row
    n_t1 = cfg['n_t1']
    ans = it['answers']
    n_correct = {c: sum(a['correct'] for a in ans[c] if not a['greedy']) for c in CONDS}
    return {'pid': it['pid'], 'status': 'ok', 'set': set_name, 'stratum': it['sample']['think_end'],
            'question_type': it['row'].get('question_type'), 'n_visual': int(it['vis'].shape[0]), 'm': it['m'],
            'code_version': version, 'scorer_version': SCORER_VERSION, 'cfg': dict(cfg), 'recon': it['recon'],
            'steps': {c: it['st'][c]['steps'] for c in STEP_CONDS}, 'probes': {c: it['st'][c]['probes'] for c in STEP_CONDS},
            'answers': ans, 'n_correct': n_correct, 'acc_t1': {c: n_correct[c] / n_t1 for c in CONDS},
            'greedy': {c: next(a['correct'] for a in ans[c] if a['greedy']) for c in CONDS}, 'answer_gt': it['row']['answer'],
            'timing': {'encode_s': it.get('encode_s'), **it['timing']}}


def process_group(pids, make_item, eng, tok, sp, cfg, generate, version, set_name, timing=None):
    """Rows (ok or error) for pids, in order. A question that raises gets an error row and the rest goes on: when the group
    raises, every question is re-run alone. A ChunkError stops the run."""
    timing = {} if timing is None else timing
    rows, items = {}, []
    t = time.time()
    for pid in pids:
        t0 = time.time()
        try:
            it = make_item(pid)
            it['encode_s'] = round(time.time() - t0, 2)
            items.append(it)
        except ChunkError:
            raise
        except Exception as e:
            rows[pid] = error_row(pid, 'load', f'{type(e).__name__}: {e}', version, set_name)
    timing['encode'] = time.time() - t
    try:
        run_group(items, eng, tok, sp, cfg, generate, timing)
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
            rows[it['pid']] = row_out(it, cfg, version, set_name)
    return [rows[p] for p in pids]


def select_pids(doc, set_name, part='all', limit=0, shard='0/1', extra=(), done=()):
    """The pids this process runs: the set's list (main: restricted to part1 / part2 / all), first `limit`, plus the extra pids
    (not already in), the shard's share of that list by index, without the done ones."""
    if set_name not in ('main', 'trial') or set_name not in doc:
        raise SystemExit(f'no set {set_name!r} in the ids file')
    if part not in ('1', '2', 'all'):
        raise SystemExit(f'--pids-part must be 1, 2 or all, got {part!r}')
    if set_name == 'trial' and part != 'all':
        raise SystemExit('--pids-part applies to --set main only')
    ids = [str(p) for p in doc['part1' if part == '1' else 'part2' if part == '2' else set_name]]
    if limit:
        ids = ids[:limit]
    ids = ids + [p for p in dict.fromkeys(str(x) for x in extra) if p not in set(ids)]
    si, sn = map(int, shard.split('/'))
    ids = [p for i, p in enumerate(ids) if i % sn == si]
    done = {str(x) for x in done}
    return [p for p in ids if p not in done]


def memory_check(it, eng, tok, sp, cfg, generate, target):
    """Peak memory of the longest planned work: the question's base + its stored text repeated up to `target` tokens, then
    Engine.last_hidden over a step's worth (think_cap) of the end, select_tokens with m, and a few generated tokens."""
    z = list(it['z']) or tok.encode('a', add_special_tokens=False)
    n_pad = max(0, target - int(it['base'].shape[0]))
    pad = (z * (n_pad // len(z) + 1))[:n_pad]
    ctx = torch.cat([it['base'], eng.embed(pad)], 0)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    n = min(cfg['think_cap'], int(ctx.shape[0]))
    Z = eng.last_hidden(ctx, n)
    t_hidden = time.time() - t
    sel = select_tokens(it['vis'], Z, budget(int(it['vis'].shape[0]), cfg['vis_frac']))
    g = torch.Generator(device=device_of(eng)).manual_seed(seed_of(cfg['seed'], it['pid'], 'memcheck'))
    t = time.time()
    r = generate([Request(ctx, frozenset({sp['think_end'], sp['eos']}), min(8, cfg['chunk']), g)])[0]
    if cuda:
        torch.cuda.synchronize()
    return {'pid': it['pid'], 'n_visual': int(it['vis'].shape[0]), 'ctx_len': int(ctx.shape[0]), 'last_hidden_n': n,
            'n_sel': len(sel), 'generated_tokens': None if r is None else len(r.ids),
            'last_hidden_s': round(t_hidden, 2), 'generate_s': round(time.time() - t, 2),
            'peak_alloc_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if cuda else 0,
            'peak_reserved_gb': round(torch.cuda.max_memory_reserved() / 2 ** 30, 2) if cuda else 0}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--model', default='OpenGVLab/InternVL3_5-1B')
    ap.add_argument('--revision', default='2f71cf52542334823e48a46ffba0e2bc9add3446')
    ap.add_argument('--samples', default=str(ROOT / 'diag/step1_store_bmain.json'))
    ap.add_argument('--ids', default=str(ROOT / 'diag/closed_multistep_ids.json'))
    ap.add_argument('--set', default='main', choices=('main', 'trial'))
    ap.add_argument('--pids-part', default='all', choices=('1', '2', 'all'), help='main only: 1 = first 112, 2 = the other 222')
    ap.add_argument('--extra-pids', default='', help='comma list of pids run in addition to the set')
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


def read_rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def main(argv=None):
    a = parse_args(argv)
    cfg = a.cfg
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
    todo = select_pids(doc, a.set, a.pids_part, a.limit, a.shard, extra, done)
    for p in todo:
        assert p in samples, f'no stored step 1 for {p}'
        assert doc['stratum'].get(p, samples[p]['think_end']) == samples[p]['think_end'], (p, doc['stratum'].get(p), samples[p]['think_end'])
    version = code_version()
    print(json.dumps({'event': 'start', 'code_version': version, 'set': a.set, 'pids_part': a.pids_part, 'todo': len(todo),
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
        return {'pid': pid, 'row': row, 'sample': s, 'base': base, 'vis': vis, 'z': z, 'sep': sep, 'sep_text': sep_text,
                'recon': recon}

    peak_all, n_err = 0.0, 0
    with out.open('a') as f:
        for gi in range(0, len(todo), cfg['group']):
            if a.time_limit and time.time() - t0 > a.time_limit:
                print(json.dumps({'event': 'time_limit', 'left': len(todo) - gi}), flush=True)
                break
            tg = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            tm = {}
            out_rows = process_group(todo[gi:gi + cfg['group']], make_item, eng, tok, sp, cfg, generate, version, a.set, tm)
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                if r['status'] != 'ok':
                    n_err += 1
                    print(json.dumps({'event': 'error', 'pid': r['pid'], 'error': r['error']}), flush=True)
            f.flush()
            peak = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else 0
            peak_all = max(peak_all, peak)
            print(json.dumps({'event': 'group', 'n': len(out_rows), 'n_error': sum(r['status'] != 'ok' for r in out_rows),
                              'encode_s': round(tm.get('encode', 0), 1), 'steps_s': round(tm.get('steps', 0), 1),
                              'probes_s': round(tm.get('probes', 0), 1), 'answers_s': round(tm.get('answers', 0), 1),
                              'group_s': round(time.time() - tg, 1), 'elapsed': round(time.time() - t0),
                              'left': len(todo) - gi - len(out_rows), 'peak_alloc_gb': peak,
                              'max_rss_gb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)}), flush=True)
    if a.set == 'trial' and doc.get('memcheck') and doc['memcheck'] in samples:
        mem = memory_check(make_item(doc['memcheck']), eng, tok, sp, cfg, generate, cfg['memcheck_tokens'])
        Path(str(out) + '.memcheck.json').write_text(json.dumps(mem, indent=1) + '\n')
        print(json.dumps({'event': 'memcheck', **mem}), flush=True)
    print(json.dumps({'event': 'end', 'elapsed': round(time.time() - t0), 'n_error': n_err, 'peak_alloc_gb': peak_all,
                      'max_rss_gb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)}), flush=True)


if __name__ == '__main__':
    main()
