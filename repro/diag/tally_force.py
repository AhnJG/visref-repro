"""tally_force -- diagnostic comparing conditions N and D on TallyQA with own image context.

Step 1 is generated from the question's own image prompt (up to think_cap tokens, stops at </think> / EOS).
Selection:
  Z = Engine.last_hidden([prompt][z], |z|)
  sel = select_tokens(vis, Z, budget(|vis|, vis_frac)), side 'own'. Empty z gives empty sel.
Conditions:
  N = [prompt][z][sep] + Wait, think more
  D = [prompt][z][sep] + v_block(vis, sel) + Wait, think more  (no block if z is empty)
Extra thinking:
  Both N and D think up to think_extra tokens, sharing seed_of(seed, pid, 'think'); close_thinking appends </think>.
Answers:
  probe_prompt '\\n\\n\\boxed', n_t1 samples at T=1 (seed_of(seed, pid, 'ans', j)) + 1 greedy, <= probe_cap tokens,
  stopped when box closes, answer_record = rule scorer v4.
No accuracy or decision is computed here (see diag/analyze_tally_force.py).

  python3 diag/tally_force.py --parquet-dir DIR --out results/tally_force_0.jsonl [--set main|trial]
      [--ids diag/tally_force_ids.json] [--limit N] [--shard 0/2] [--skip DONE.json] [--time-limit SECONDS]
"""
import argparse
import hashlib
import io
import json
import resource
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'diag'))

from blank_think import device_of  # noqa: E402
from closed_multistep import ChunkError, error_row, make_generate, split_generated, v_block  # noqa: E402
from run_common import read_done, run_loop, select_pids  # noqa: E402
from step1_think import GREEDY, T1, answer_record, box_close_index, close_thinking, extra_record, seed_of  # noqa: E402
from visref.dpp import budget, select_tokens  # noqa: E402
from visref.engine import Engine, Request  # noqa: E402
from visref.scoring import SCORER_VERSION  # noqa: E402

CONDS = ('N', 'D')
SIDES = ('own',)
DEFAULTS = dict(
    think_cap=1024, think_extra=1024, vis_frac=0.3, probe_prompt='\n\n\\boxed',
    wait_text='Wait, think more', probe_cap=16, n_t1=3, temperature=0.6, top_k=50, top_p=0.95,
    seed=42, cut_sep='\n', chunk=0, batch_tokens=72000, group=4, max_num=12
)


def code_version():
    """sha256 over this script, run_common.py, imported diag helpers, run.py and visref/*.py."""
    h = hashlib.sha256()
    files = [
        Path(__file__).resolve(),
        ROOT / 'diag/run_common.py',
        ROOT / 'diag/blank_think.py',
        ROOT / 'diag/closed_multistep.py',
        ROOT / 'diag/step1_think.py',
        ROOT / 'run.py',
        *(ROOT / 'visref').glob('*.py')
    ]
    for f in sorted(files):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:12]


def resolve_cfg(cfg):
    """chunk 0 -> max(think_cap, think_extra); refuse a chunk below either (unfinished rows would be read as 'cap')."""
    cfg = dict(cfg)
    need = max(cfg['think_cap'], cfg['think_extra'])
    if cfg['chunk'] == 0:
        cfg['chunk'] = need
    if cfg['chunk'] < need:
        raise SystemExit(f"--chunk {cfg['chunk']} < {need}: refused")
    if min(cfg['think_cap'], cfg['think_extra'], cfg['probe_cap'], cfg['n_t1']) < 1 or cfg['probe_cap'] > cfg['chunk']:
        raise SystemExit('--think-cap, --think-extra, --probe-cap and --n-t1 must be >= 1 and --probe-cap <= --chunk')
    if not 0 < cfg['vis_frac'] <= 1:
        raise SystemExit('--vis-frac must be in (0, 1]')
    return cfg


def is_integer_string(s):
    if not isinstance(s, str):
        return False
    s = s.strip()
    if s.startswith(('-', '+')):
        s = s[1:]
    return bool(s and s.isdigit())


def find_shard_file(parquet_dir, shard):
    name = f'test-{shard:05d}-of-00019.parquet'
    p = Path(parquet_dir) / name
    if p.exists():
        return p
    p_data = Path(parquet_dir) / 'data' / name
    if p_data.exists():
        return p_data
    return p


_PF_CACHE = {}


def get_parquet_file(shard_path):
    p = str(shard_path)
    if p not in _PF_CACHE:
        _PF_CACHE[p] = pq.ParquetFile(shard_path)
    return _PF_CACHE[p]


def read_parquet_row(shard_path, target_row):
    pf = get_parquet_file(shard_path)
    accum = 0
    for rg_idx in range(pf.num_row_groups):
        rg_meta = pf.metadata.row_group(rg_idx)
        n = rg_meta.num_rows
        if accum <= target_row < accum + n:
            offset_in_rg = target_row - accum
            table = pf.read_row_group(rg_idx)
            return {col: table[col][offset_in_rg].as_py() for col in table.column_names}
        accum += n
    raise IndexError(f"Row {target_row} out of range in {shard_path} (total rows {accum})")


def selections(it, eng, cfg):
    vis = it['vis']
    m = budget(vis.shape[0], cfg['vis_frac'])
    z = it['z']
    if z:
        Z = eng.last_hidden(torch.cat([it['prompt'], eng.embed(z)], 0), len(z))
        sel = [int(i) for i in select_tokens(vis, Z, m)]
    else:
        sel = []
    return m, {'own': sel}


def contexts(it, eng, tok, sp):
    z, sep, sep_text = it['z'], it['sep'], it['sep_text']
    pre = torch.cat([it['prompt'], eng.embed(z + sep)], 0)
    vis = it['vis']
    sel = it['sel']['own']
    block = pre[:0]
    if z:
        block = v_block(eng, tok, sp, vis, sel, tok.decode(z), sep_text)
    wait = eng.embed(sp['wait'])
    return {
        'N': torch.cat([pre, wait], 0),
        'D': torch.cat([pre, block, wait], 0),
    }


def run_group(items, eng, tok, sp, cfg, generate, timing=None):
    te, eos = sp['think_end'], sp['eos']
    dev = device_of(eng)
    timing = {} if timing is None else timing
    for it in items:
        it.update(z=None, sep=None, sep_text='', step1=None, sel={}, m=0, ctx={}, extra={}, answers={}, nonfinite=[])

    # 1. Step 1 generation
    t = time.time()
    reqs, keys = [], []
    for it in items:
        g = torch.Generator(device=dev).manual_seed(seed_of(cfg['seed'], it['pid'], 'step1'))
        reqs.append(Request(it['prompt'], frozenset({te, eos}), cfg['think_cap'], g))
        keys.append(it)
    for it, r in zip(keys, generate(reqs) if reqs else []):
        if r is None:
            it['nonfinite'].append('step1')
            continue
        z, sep, sep_text, end = split_generated(tok, sp, r.ids, cfg['cut_sep'])
        it['z'], it['sep'], it['sep_text'] = z, sep, sep_text
        it['step1'] = {'end': end, 'len': len(z), 'ids': [int(x) for x in z], 'sep_text': sep_text}
    timing['step1'] = time.time() - t

    # 2. Selections and Contexts
    t = time.time()
    for it in items:
        if it['nonfinite']:
            continue
        it['m'], it['sel'] = selections(it, eng, cfg)
        it['ctx'] = contexts(it, eng, tok, sp)
    timing['select'] = time.time() - t

    # 3. Extra thinking for N and D
    t = time.time()
    reqs, keys = [], []
    for it in items:
        if it['nonfinite']:
            continue
        for c in CONDS:
            g = torch.Generator(device=dev).manual_seed(seed_of(cfg['seed'], it['pid'], 'think'))
            reqs.append(Request(it['ctx'][c], frozenset({te, eos}), cfg['think_extra'], g))
            keys.append((it, c))
    for (it, c), r in zip(keys, generate(reqs) if reqs else []):
        if r is None:
            it['nonfinite'].append(c)
            continue
        tail, end = close_thinking(tok, sp, r.ids, r.finish, cfg['cut_sep'])
        it['extra'][c] = extra_record(tok, sp, r.ids, end)
        it['ctx'][c] = torch.cat([it['ctx'][c], eng.embed(tail)], 0)
    timing['extra'] = time.time() - t

    # 4. Answers for N and D
    t = time.time()
    stop = lambda ids, bt=sp['brace_text']: box_close_index(ids, bt) is not None  # noqa: E731
    reqs, keys = [], []
    for it in items:
        if 'step1' in it['nonfinite']:
            continue
        for c in CONDS:
            if c in it['nonfinite']:
                continue
            ctx = torch.cat([it['ctx'][c], eng.embed(sp['probe_prompt'])], 0)
            for j in range(cfg['n_t1'] + 1):
                greedy = j == cfg['n_t1']
                g = None if greedy else torch.Generator(device=dev).manual_seed(seed_of(cfg['seed'], it['pid'], 'ans', j))
                reqs.append(Request(ctx, frozenset({eos}), cfg['probe_cap'], g, sampling=GREEDY if greedy else T1, stop_check=stop))
                keys.append((it, c, j))
    for (it, c, j), r in zip(keys, generate(reqs) if reqs else []):
        if r is None:
            it['nonfinite'].append(f'{c}_answer')
            continue
        rec = answer_record(tok, sp, r.ids, it['row'])
        rec.update(greedy=j == cfg['n_t1'], n_tok=len(r.ids))
        it['answers'].setdefault(c, []).append(rec)
    timing['answers'] = time.time() - t

    for it in items:
        it['ctx'] = {}
        it['timing'] = {f'group_{k}_s': round(timing.get(k, 0.0), 2) for k in ('step1', 'select', 'extra', 'answers')}


def row_out(it, cfg, version, set_name):
    if it.get('nonfinite'):
        row = error_row(it['pid'], 'generate', 'non-finite logits in ' + ','.join(sorted(set(it['nonfinite']))), version, set_name)
        row['nonfinite'] = sorted(set(it['nonfinite']))
        return row
    ans = it['answers']
    n_correct = {c: sum(a['correct'] for a in ans[c] if not a['greedy']) for c in CONDS}
    return {
        'pid': it['pid'],
        'status': 'ok',
        'set': set_name,
        'shard': it['shard'],
        'row': it['row_idx'],
        'qi': it['qi'],
        'n_visual': int(it['vis'].shape[0]),
        'm': it['m'],
        'sel': {'own': [int(i) for i in it['sel']['own']]},
        'n_sel': {'own': len(it['sel']['own'])},
        'step1': it['step1'],
        'extra': it['extra'],
        'answers': ans,
        'n_correct': n_correct,
        'acc_t1': {c: n_correct[c] / cfg['n_t1'] for c in CONDS},
        'greedy': {c: next(a['correct'] for a in ans[c] if a['greedy']) for c in CONDS},
        'answer_gt': it['row']['answer'],
        'code_version': version,
        'scorer_version': SCORER_VERSION,
        'cfg': dict(cfg),
        'timing': {'encode_s': it.get('encode_s'), **it['timing']},
    }


def memory_check(it, eng, tok, sp, cfg, generate):
    """Peak memory of the selection prefill (prompt + think_cap) and of the longest planned context (prompt + think_cap +
    block of all image tokens + Wait + think_extra), padded with text."""
    pad_ids = tok.encode('a b c ', add_special_tokens=False)
    pad = lambda n: (pad_ids * (n // len(pad_ids) + 1))[:n]  # noqa: E731
    z1, z2 = pad(cfg['think_cap']), pad(cfg['think_extra'])
    vis = it['vis']
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    Z = eng.last_hidden(torch.cat([it['prompt'], eng.embed(z1)], 0), len(z1))
    sel = select_tokens(vis, Z, budget(vis.shape[0], cfg['vis_frac']))
    t_hidden = time.time() - t
    block = v_block(eng, tok, sp, vis, list(range(int(vis.shape[0]))), tok.decode(z1), '')
    ctx = torch.cat([it['prompt'], eng.embed(z1), block, eng.embed(sp['wait']), eng.embed(z2)], 0)
    t = time.time()
    g = torch.Generator(device=device_of(eng)).manual_seed(seed_of(cfg['seed'], it['pid'], 'memcheck'))
    r = generate([Request(ctx, frozenset({sp['eos']}), min(8, cfg['chunk']), g)])[0]
    if cuda:
        torch.cuda.synchronize()
    return {
        'pid': it['pid'], 'n_visual': int(vis.shape[0]), 'last_hidden_n': len(z1), 'n_sel': len(sel), 'ctx_len': int(ctx.shape[0]),
        'generated_tokens': None if r is None else len(r.ids), 'last_hidden_s': round(t_hidden, 2),
        'generate_s': round(time.time() - t, 2),
        'peak_alloc_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if cuda else 0,
        'peak_reserved_gb': round(torch.cuda.max_memory_reserved() / 2 ** 30, 2) if cuda else 0
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet-dir', required=True)
    ap.add_argument('--model', default='OpenGVLab/InternVL3_5-1B')
    ap.add_argument('--revision', default='2f71cf52542334823e48a46ffba0e2bc9add3446')
    ap.add_argument('--ids', default=str(ROOT / 'diag/tally_force_ids.json'))
    ap.add_argument('--set', default='main', choices=('main', 'trial'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--skip', default='', help='JSON list of pids finished earlier (ok rows)')
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
    from visref.model import encode_sample, load_model, special_ids

    doc = json.loads(Path(a.ids).read_text())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = read_done(out, a.skip)
    n_done = len(done)
    todo = select_pids(doc, a.set, a.limit, a.shard, done)
    version = code_version()
    print(json.dumps({'event': 'start', 'code_version': version, 'set': a.set, 'todo': len(todo), 'done': n_done, 'cfg': cfg}),
          flush=True)

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
        loc = doc['loc'][pid]
        shard, row_idx, qi = loc['shard'], loc['row'], loc['qi']
        shard_file = find_shard_file(a.parquet_dir, shard)
        raw_row = read_parquet_row(shard_file, row_idx)
        img_data = raw_row['image']
        if isinstance(img_data, Image.Image):
            image = img_data
        elif hasattr(img_data, 'convert'):
            image = img_data.convert('RGB')
        else:
            img_bytes = img_data['bytes'] if isinstance(img_data, dict) else img_data
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        qa = raw_row['qa']
        if qi < 0 or qi >= len(qa):
            raise IndexError(f"qi {qi} out of range (qa length {len(qa)})")
        q_item = qa[qi]
        ans = q_item['answer']
        if not is_integer_string(ans):
            raise ValueError(f"answer {ans!r} is not an integer string")
        query = ('Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.\nQuestion: '
                 + q_item['question'])
        score_row = {
            'answer': str(ans),
            'question_type': 'free_form',
            'answer_type': 'integer',
            'precision': None,
            'unit': None,
            'choices': None,
            'query': query
        }
        base, vis = encode_sample(model, tok, image, query, cfg['max_num'])
        return {
            'pid': pid,
            'shard': shard,
            'row_idx': row_idx,
            'qi': qi,
            'row': score_row,
            'prompt': base,
            'vis': vis
        }

    peak_all, n_err = run_loop(
        todo, out, make_item, eng, tok, sp, cfg, generate, version, a.set,
        run_group, row_out, time_limit=a.time_limit, t0=t0, skip=a.skip
    )

    if a.set == 'trial' and doc.get('memcheck') in (todo + list(done)):
        mem = memory_check(make_item(doc['memcheck']), eng, tok, sp, cfg, generate)
        Path(str(out) + '.memcheck.json').write_text(json.dumps(mem, indent=1) + '\n')
        print(json.dumps({'event': 'memcheck', **mem}), flush=True)

    print(json.dumps({'event': 'end', 'elapsed': round(time.time() - t0), 'n_error': n_err, 'peak_alloc_gb': peak_all,
                      'max_rss_gb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)}), flush=True)


if __name__ == '__main__':
    main()
