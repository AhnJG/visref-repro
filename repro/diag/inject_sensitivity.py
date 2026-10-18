"""Injection-sensitivity check (EF 2026-10-18 "실행 전 계획: 이미지 민감도 점검"; written before any run).

Question: do re-inserted image tokens move the model's answer distribution at all, and do the DPP-picked tokens move it
more than random ones? No answer is generated in part 1; only next-token probabilities are read.

Part 1 (diag/inject_sensitivity_ids.json 'part1', 200 questions). From the stored step-1 thinking z_1 (re-tokenised as in
diag/step1_think.split_stored), five contexts, each followed by </think> and the stopping-probe prompt '\\n\\n\\boxed' (the
main run's probe context, methods.py):
  none      [base][z_1][sep]                                  </think> \\n\\n\\boxed
  dpp       [base][z_1][sep] V[sel]                           </think> ...   sel = select_tokens(V, embed(z_1), m)  (main run)
  rand      [base][z_1][sep] V[rand]                          </think> ...   |rand| = |sel|, seed_of(seed, pid, 'rand')
  all       [base][z_1][sep] V                                </think> ...   every image token, original order
  other     [base][z_1][sep] V_other                          </think> ...   length-matched control: the image tokens of
                                                                             another question (the next one in part 1, cyclic),
                                                                             repeated or cut to |V| tokens
  dpp_tag   [base][z_1][sep] [\\n] <img> V[sel] </img> \\n      </think> ...   same sel, InternVL's image tags, before </think>
V = the image's input embeddings (vision encoder + projector, what the model reads), untagged unless stated.
Read per context, teacher-forcing two answer strings after '\\boxed':
  gt   '{' + answer + '}' (multiple choice: the letter of the right choice), with '{' as its own token as the model
       writes it in the stored answers                                                -> logp_gt (sum over its tokens)
  tsr  the main-run TSR's step-1 sampled answer ids (inject_sensitivity_ids.json)      -> logp_tsr
and the full next-token distribution at the first answer position (after the '{'): kl_from_none =
KL(p_none || p_ctx) in nats, top1 token, entropy. No accuracy and no decision is computed here.

Part 2 ('part2', 100 questions whose step 1 the model closed itself): in the contexts none and dpp, n_msh = 8 short answers
after the probe prompt at T = 1 without top-k/top-p (the main run's probe sampling), stopped when the box closes
(<= probe_cap tokens); sample j uses seed_of(seed, pid, 'msh', j) in both contexts. Each sample's stopping entropy is
computed exactly as the main run's H (methods._probe_entropy: summed token entropies, the closing-brace tokens merged
into one outcome inside the box, infinite when the box does not close with an alphanumeric answer); the analysis
averages the finite values of the 8 samples.

  python3 diag/inject_sensitivity.py --parquet PQ --out results/inject_sensitivity_0.jsonl [--part 1|2|12]
      [--limit 5] [--shard 0/2] [--time-limit SECONDS]
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
from step1_think import answer_record, box_close_index, inject_embeds, make_generate, random_sel, seed_of, split_stored  # noqa: E402
from visref.dpp import budget, select_tokens  # noqa: E402
from visref.engine import Engine, Request  # noqa: E402
from visref.scoring import SCORER_VERSION  # noqa: E402

CONDS1 = ('none', 'dpp', 'rand', 'all', 'other', 'dpp_tag')
CONDS2 = ('none', 'dpp')
DEFAULTS = dict(vis_frac=0.3, probe_prompt='\n\n\\boxed', wait_text='Wait, think more', probe_cap=16, n_msh=8,
                temperature=0.6, top_k=50, top_p=0.95, seed=42, cut_sep='\n', chunk=1024, batch_tokens=72000,
                group=8, max_num=12)
T1 = (1.0, 0, 1.0)


def code_version():
    """sha256 over this script, the step1_think helpers it imports, run.py and visref/*.py."""
    h = hashlib.sha256()
    for f in sorted([Path(__file__).resolve(), ROOT / 'diag/step1_think.py', ROOT / 'run.py', *(ROOT / 'visref').glob('*.py')]):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:12]


def gt_text(row):
    """The answer string inside the box: the letter for multiple choice, the answer otherwise."""
    if row['question_type'] == 'multi_choice' and row.get('choices'):
        return 'ABCDEFGH'[row['choices'].index(row['answer'])]
    return str(row['answer'])


def kl(p_log, q_log):
    """KL(p || q) in nats from two log-probability vectors over the same vocabulary."""
    return float((p_log.exp() * (p_log - q_log)).sum())


def entropy(p_log):
    return float(-(p_log.exp() * p_log).sum())


def length_matched(vis_other, n):
    """The control's tokens: another question's image tokens repeated (cyclically) or cut to n tokens."""
    reps = -(-n // vis_other.shape[0])
    return vis_other.repeat(reps, 1)[:n]


def contexts(it, eng, tok, sp, cfg):
    """Fills it['m'], it['sel'], it['sel_rand'] and returns {cond: context embeddings ending with the probe prompt}."""
    base, vis, z, sep = it['base'], it['vis'], it['z'], it['sep']
    prefix = torch.cat([base, eng.embed(z + sep)], 0)
    m = budget(vis.shape[0], cfg['vis_frac'])
    sel = [int(i) for i in select_tokens(vis, eng.embed(z), m)] if z else []
    rand = random_sel(it['pid'], cfg['seed'], int(vis.shape[0]), len(sel))
    tail = eng.embed([sp['think_end']] + sp['probe_prompt'])
    zt = tok.decode(z)
    other = length_matched(it['vis_other'], vis.shape[0]) if it.get('vis_other') is not None else None
    inj = {'none': vis[:0], 'dpp': vis[sel], 'rand': vis[rand], 'all': vis, 'other': other,
           'dpp_tag': inject_embeds(eng, tok, sp, 'hs_tag_in', vis, sel, zt, it['sep_text'])}
    it.update(m=m, sel=sel, sel_rand=rand)
    return {c: torch.cat([prefix, inj[c], tail], 0) for c in CONDS1 if inj[c] is not None}


def score_string(eng, ctx, ids):
    """Teacher-forced log-probabilities of ids after ctx: (sum of log p over ids, log-prob vector at each position)."""
    full = torch.cat([ctx, eng.embed(ids)], 0)
    h = eng.last_hidden(full, len(ids) + 1)[:-1]          # h[i] predicts ids[i]
    logp = torch.log_softmax(eng.lm.lm_head(h).float(), -1)
    return float(sum(logp[i, t] for i, t in enumerate(ids))), logp


def part1(it, eng, tok, sp, cfg, probe):
    ctxs = it['ctx1'] = contexts(it, eng, tok, sp, cfg)
    brace = tok.encode('{', add_special_tokens=False)
    assert len(brace) == 1, brace
    gt_ids = brace + tok.encode(gt_text(it['row']) + '}', add_special_tokens=False)
    tsr_ids = list(probe.get('ids') or [])
    first = 1                                             # position after the opening brace
    rec, ref = {}, None
    for c in CONDS1:
        if c not in ctxs:
            continue
        lg, logp = score_string(eng, ctxs[c], gt_ids)
        lt = score_string(eng, ctxs[c], tsr_ids)[0] if tsr_ids else None
        d = logp[first]
        if c == 'none':
            ref = d
        rec[c] = {'logp_gt': round(lg, 4), 'logp_tsr': None if lt is None else round(lt, 4),
                  'kl_from_none': round(kl(ref, d), 5), 'top1': int(d.argmax()), 'entropy_first': round(entropy(d), 4),
                  'ctx_len': int(ctxs[c].shape[0])}
    return {'gt_ids': gt_ids, 'tsr_ids': tsr_ids, 'tsr_valid': bool(probe.get('valid')),
            'tsr_brace_first': bool(tsr_ids) and tsr_ids[0] == brace[0], 'first_pos': first,
            'question_type': it['row']['question_type'], 'other_pid': it.get('other_pid'), 'cond': rec}


def box_depths(ids, brace_text):
    """Brace depth before each token of an answer after a prompt ending in '\\boxed' (methods._brace_scan)."""
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


def sample_H(r, rec, sp):
    """The main run's stopping entropy of one sampled answer (methods._probe_entropy), inf when not valid."""
    ids = [t for t in r.ids if t != sp['eos']]
    depth = box_depths(ids, sp['brace_text'])
    ent = [l if d == 1 else f for f, l, d in zip(r.ent, r.ent_lump, depth)]
    return sum(ent) if rec['valid'] and r.finish == 'stop' else float('inf')


def part2(items, eng, tok, sp, cfg, generate):
    eos = sp['eos']
    dev = eng.emb.weight.device
    stop = lambda ids, bt=sp['brace_text']: box_close_index(ids, bt) is not None  # noqa: E731
    reqs, keys = [], []
    for it in items:
        ctxs = it['ctx1'] if 'ctx1' in it else contexts(it, eng, tok, sp, cfg)
        for c in CONDS2:
            for j in range(cfg['n_msh']):
                g = torch.Generator(device=dev).manual_seed(seed_of(cfg['seed'], it['pid'], 'msh', j))
                reqs.append(Request(ctxs[c], frozenset({eos}), cfg['probe_cap'], g, record_entropy=True, sampling=T1,
                                    stop_check=stop, ent_lump=sp['close_ids']))
                keys.append((it, c))
    for (it, c), r in zip(keys, generate(reqs)):
        if r is None:
            rec = {'text': '', 'valid': False, 'correct': 0, 'error': True, 'H': None}
        else:
            rec = answer_record(tok, sp, r.ids, it['row'])
            h = sample_H(r, rec, sp)
            rec['H'] = None if math.isinf(h) else round(h, 6)
        it.setdefault('msh', {}).setdefault(c, []).append(rec)


def mean_finite(hs):
    """Mean of the samples' stopping entropies that are finite (valid answers); None when there is none."""
    v = [h for h in hs if h is not None]
    return round(sum(v) / len(v), 6) if v else None


def answer_entropy(texts):
    """Empirical entropy (nats) of a list of answer strings, whitespace removed; one differing answer of 8 gives 0.377."""
    keys = [''.join(t.split()) for t in texts]
    n = len(keys)
    return -sum(k / n * math.log(k / n) for k in (keys.count(x) for x in set(keys))) if n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--model', default='OpenGVLab/InternVL3_5-1B')
    ap.add_argument('--revision', default='2f71cf52542334823e48a46ffba0e2bc9add3446')
    ap.add_argument('--samples', default=str(ROOT / 'diag/step1_store_bmain.json'))
    ap.add_argument('--ids', default=str(ROOT / 'diag/inject_sensitivity_ids.json'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--part', default='12', choices=['1', '2', '12'])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--time-limit', type=float, default=0)
    for k, v in DEFAULTS.items():
        ap.add_argument('--' + k.replace('_', '-'), type=type(v), default=v)
    a = ap.parse_args()
    cfg = {k: getattr(a, k) for k in DEFAULTS}
    t0 = time.time()
    from run import fast_kv_selfcheck, pack
    from visref.model import encode_sample, load_mathvista, load_model, special_ids
    samples = json.loads(Path(a.samples).read_text())
    samples = samples.get('pids', samples)
    doc = json.loads(Path(a.ids).read_text())
    want = {p: set() for p in doc['part1'] + doc['part2']}
    if '1' in a.part:
        for p in doc['part1'][:a.limit or None]:
            want[p].add(1)
    if '2' in a.part:
        for p in doc['part2'][:a.limit or None]:
            want[p].add(2)
    order = [p for p in dict.fromkeys(doc['part1'] + doc['part2']) if want[p]]
    si, sn = map(int, a.shard.split('/'))
    order = [p for i, p in enumerate(order) if i % sn == si]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)['pid'] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    todo = [p for p in order if p not in done]
    version = code_version()
    print(json.dumps({'event': 'start', 'code_version': version, 'todo': len(todo), 'done': len(done), 'cfg': cfg}), flush=True)
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
    p1 = doc['part1']
    donor = {p: p1[(i + 1) % len(p1)] for i, p in enumerate(p1)}  # length-matched control: the next question's image
    with out.open('a') as f:
        for gi in range(0, len(todo), cfg['group']):
            if a.time_limit and time.time() - t0 > a.time_limit:
                print(json.dumps({'event': 'time_limit', 'left': len(todo) - gi}), flush=True)
                break
            tg = time.time()
            items = []
            for pid in todo[gi:gi + cfg['group']]:
                s, row = samples[pid], rows[pid]
                base, vis = encode_sample(model, tok, row['image'](), row['query'], cfg['max_num'])
                z, sep, sep_text, recon = split_stored(tok, sp, s, cfg['cut_sep'])
                it = {'pid': pid, 'row': row, 'sample': s, 'base': base, 'vis': vis, 'z': z, 'sep': sep,
                      'sep_text': sep_text, 'recon': recon}
                if 1 in want[pid]:
                    o = rows[donor[pid]]
                    it['vis_other'] = encode_sample(model, tok, o['image'](), o['query'], cfg['max_num'])[1]
                    it['other_pid'] = donor[pid]
                items.append(it)
            for it in items:
                if 1 in want[it['pid']]:
                    it['p1'] = part1(it, eng, tok, sp, cfg, doc['tsr_probe'][it['pid']])
            p2 = [it for it in items if 2 in want[it['pid']]]
            if p2:
                part2(p2, eng, tok, sp, cfg, generate)
            for it in items:
                r = {'pid': it['pid'], 'stratum': it['sample']['think_end'], 'parts': sorted(want[it['pid']]),
                     'code_version': version, 'scorer_version': SCORER_VERSION, 'n_visual': int(it['vis'].shape[0]),
                     'm': it.get('m'), 'n_sel': len(it.get('sel', [])), 'sel': it.get('sel'), 'sel_rand': it.get('sel_rand'),
                     'z_ids': it['z'], 'sep_ids': it['sep'], 'recon': it['recon'], 'answer_gt': it['row']['answer']}
                if 'p1' in it:
                    r['part1'] = it['p1']
                if 'msh' in it:
                    r['part2'] = {c: {'answers': v, 'n_valid': sum(x['valid'] for x in v),
                                      'H_mean': mean_finite([x['H'] for x in v]),
                                      'entropy': answer_entropy([x['text'] for x in v if x['valid']])}
                                  for c, v in it['msh'].items()}
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
            f.flush()
            print(json.dumps({'event': 'group', 'n': len(items), 'group_s': round(time.time() - tg, 1),
                              'elapsed': round(time.time() - t0), 'left': len(todo) - gi - len(items),
                              'peak_alloc_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else 0}),
                  flush=True)
    print(json.dumps({'event': 'end', 'elapsed': round(time.time() - t0)}), flush=True)


if __name__ == '__main__':
    main()
