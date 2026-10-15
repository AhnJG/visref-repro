"""Run ST / TSR / VisRef on MathVista testmini with InternVL3.5 (one process per GPU).

Resumable: (pid, method) pairs already in --out are skipped. Rows are appended as
JSON lines with raw thinking, per-step probe records and the scored final answer.

Step 1: by default the first method of a pid (the leader) generates the step-1 thinking and the other methods are
forked from it (same context and seed). With --step1-from STORE (diag/make_step1_store.py) every method instead starts
from the step-1 thinking stored by an earlier run (methods.stored_step1), all with the same sampler state after it
(seeded with job_seed(seed, f'{pid}:stored_step1')), and the row records step1_info.

'visref_imgtag' (2026-10-15): V in the kernel is the prompt's last-layer hidden states at the image positions, computed once per pid (Engine.hidden_at) and passed to the Job as vis_hs.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from visref.engine import Engine
from visref.methods import IMGTAG, METHODS, Job, stored_step1
from visref.model import encode_sample, load_mathvista, load_model, special_ids
from visref.scoring import score

DEFAULTS = dict(delta=0.25, kmax=10, vis_frac=0.3, think_cap=1024, total_think_cap=10240, think_extra=512, answer_cap=1024,
                max_context=40960, min_step_tokens=256, probe_cap=16, probe_prompt='\n\n\\boxed', wait_text='Wait, think more',
                forced_prompt='\nFinal Answer:', af_prompt='\n**Final Answer**\n\\boxed', af_cap=100, cut_sep='\n', temperature=0.6, top_k=50,
                top_p=0.95, seed=42, batch=16, chunk=1024, batch_tokens=72000, max_num=12, fast_kv=1, dual_probe=0)


def code_version():
    """sha256 over the source that determines results; stored in every row so sessions are never mixed silently."""
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for f in sorted([root / 'run.py', *(root / 'visref').glob('*.py')]):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:12]


def job_seed(seed, pid):
    # Same seed for all methods of a pid: common random numbers for paired comparison.
    return int.from_bytes(hashlib.sha256(f'{seed}:{pid}'.encode()).digest()[:8], 'little') & ((1 << 63) - 1)


def fast_kv_selfcheck(lm, tok, n=32):
    """Greedy-decode two padded rows with stock and fast decode attention; require identical tokens."""
    from visref.engine import Engine, Request
    from visref.fastkv import set_enabled
    ids = tok.encode('<|im_start|>user\nA box holds 12 red and 7 blue balls. How many balls in total? '
                     'Explain step by step.<|im_end|>\n<|im_start|>assistant\n<think>\n', add_special_tokens=False)
    emb = lm.get_input_embeddings()(torch.tensor(ids, device=lm.device)).detach()
    outs = {}
    for fast in (False, True):
        set_enabled(fast)
        e = Engine(lm, temperature=0.0, fast_kv=fast)
        reqs = [Request(emb, frozenset(), n, None), Request(emb[5:], frozenset(), n // 2, None)]
        outs[fast] = [r.ids for r in e.generate(reqs, 1000)]
    set_enabled(True)
    ok = outs[False] == outs[True]
    return ok, {'match_rows': [a == b for a, b in zip(outs[False], outs[True])]}


def pack(reqs, chunk, budget):
    """Split requests into groups whose padded KV size (rows x max length) stays under budget tokens."""
    groups, cur = [], []
    for i in sorted(range(len(reqs)), key=lambda i: reqs[i].ctx.shape[0]):
        n = reqs[i].ctx.shape[0] + min(chunk, reqs[i].max_new)
        if cur and (len(cur) + 1) * n > budget:
            groups.append(cur)
            cur = []
        cur.append(i)
    return groups + [cur]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--model', default='OpenGVLab/InternVL3_5-1B')
    ap.add_argument('--revision', default='2f71cf52542334823e48a46ffba0e2bc9add3446')
    ap.add_argument('--ids', default='', help='comma list or path to a JSON list / manifest with "ids"')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--methods', default='st,tsr,visref_raw_rank')
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--out', required=True)
    ap.add_argument('--time-limit', type=float, default=0, help='seconds; stop admitting jobs after this')
    ap.add_argument('--skip', default='', help='JSON list of [pid, method] pairs finished in earlier sessions')
    ap.add_argument('--deadline', type=float, default=0, help='seconds; dump in-flight jobs to <out>.partial and exit')
    ap.add_argument('--step1-from', default='', help='stored step-1 thinking of an earlier run (diag/make_step1_store.py); '
                                                     'every method starts from it instead of generating step 1')
    for k, v in DEFAULTS.items():
        ap.add_argument('--' + k.replace('_', '-'), type=type(v), default=v)
    a = ap.parse_args()
    assert a.probe_prompt.endswith(('{', '\\boxed'))
    cfg = {k: getattr(a, k) for k in DEFAULTS}
    t0 = time.time()

    rows = load_mathvista(a.parquet)
    if a.ids:
        p = Path(a.ids)
        ids = (json.loads(p.read_text()) if p.exists() else a.ids.split(','))
        ids = ids['ids'] if isinstance(ids, dict) else ids
    else:
        ids = sorted(rows, key=int)
    if a.limit:
        ids = ids[:a.limit]
    si, sn = map(int, a.shard.split('/'))
    ids = [str(x) for i, x in enumerate(ids) if i % sn == si]
    methods = [m for m in a.methods.split(',') if m]
    assert all(m in METHODS for m in methods)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r['pid'], r['method']))
    if a.skip:
        done |= {tuple(x) for x in json.loads(Path(a.skip).read_text())}
    todo = [(pid, m) for pid in ids for m in methods if (pid, m) not in done]
    store, store_sha = None, None
    if a.step1_from:
        raw = Path(a.step1_from).read_bytes()
        store_sha = hashlib.sha256(raw).hexdigest()
        store = json.loads(raw)['pids']
        missing = sorted({pid for pid, _ in todo} - set(store))
        assert not missing, f'no stored step 1 for {len(missing)} pids, e.g. {missing[:5]}'
    print(json.dumps({'event': 'start', 'code_version': code_version(), 'shard': a.shard, 'todo': len(todo), 'done': len(done), 'cfg': cfg,
                      'step1_from': a.step1_from or None, 'step1_store_sha256': store_sha}), flush=True)

    torch.manual_seed(cfg['seed'])
    version = code_version()
    torch.set_grad_enabled(False)
    model, tok = load_model(a.model, a.revision)
    lm = model.language_model
    eng = Engine(lm, cfg['temperature'], cfg['top_k'], cfg['top_p'], fast_kv=bool(cfg['fast_kv']))
    if eng.fast_kv:
        ok, info = fast_kv_selfcheck(lm, tok)
        print(json.dumps({'event': 'fast_kv_check', 'ok': ok, **info}), flush=True)
        if not ok:
            from visref.fastkv import set_enabled
            set_enabled(False)
            eng.fast_kv = False
    sp = special_ids(tok, cfg['wait_text'], cfg['probe_prompt'])
    enc = {}  # pid -> [base, vis, remaining jobs, vis_hs]
    pending = {}  # pid -> methods waiting for the leader's step-1 thinking
    s1info = {}  # pid -> step1_info of a stored step 1 (--step1-from)
    queue = []
    for pid, m in todo:
        if not queue or queue[-1][0] != pid:
            queue.append((pid, []))
        queue[-1][1].append(m)
    active = []

    def new_job(pid, m, step1=None):
        gen = torch.Generator(device=lm.device).manual_seed(job_seed(cfg['seed'], pid))
        # the probe has its own stream, so the final answer's draws do not depend on how many probes ran (CRN)
        pgen = torch.Generator(device=lm.device).manual_seed(job_seed(cfg['seed'], f'{pid}:probe'))
        # logging-only second probe without V_k (cfg dual_probe, VisRef methods): its own generator object, never used for
        # decisions; its seed is irrelevant (methods.py replaces its state by the decision probe's: common random numbers)
        p2gen = (torch.Generator(device=lm.device).manual_seed(job_seed(cfg['seed'], f'{pid}:probe_noV'))
                 if cfg['dual_probe'] else None)
        j = Job(m, rows[pid], enc[pid][0], enc[pid][1], cfg, tok, eng, sp, gen, step1=step1, probe_gen=pgen,
                probe2_gen=p2gen, vis_hs=enc[pid][3])
        j.pid = pid
        return j

    def gen_safe(rs):
        # OOM -> split the group and retry; sampler state is restored so the retry draws the same numbers
        states = [r.gen.get_state() for r in rs]
        try:
            return eng.generate(rs, cfg['chunk'], stop_at=t0 + a.deadline if a.deadline else None)
        except (torch.OutOfMemoryError, FloatingPointError) as e:
            if len(rs) == 1:
                if isinstance(e, FloatingPointError):
                    return [None]  # isolate the job with non-finite logits instead of killing the shard
                raise
            for r, st in zip(rs, states):
                r.gen.set_state(st)
            torch.cuda.empty_cache()
            h = len(rs) // 2
            print(json.dumps({'event': 'split', 'reason': type(e).__name__, 'rows': len(rs)}), flush=True)
            return gen_safe(rs[:h]) + gen_safe(rs[h:])

    with out.open('a') as f:
        while active or queue or any(pending.values()):
            while queue and (not active or len(active) + sum(map(len, pending.values())) + len(queue[0][1]) <= cfg['batch']) and not (a.time_limit and time.time() - t0 > a.time_limit):
                pid, ms = queue.pop(0)
                row = rows[pid]
                vis_hs = None
                if any(m in IMGTAG for m in ms):  # V of the image-tag kernel: hidden states at the image positions, once per pid
                    base, vis, pos = encode_sample(model, tok, row['image'](), row['query'], cfg['max_num'], with_positions=True)
                    vis_hs = eng.hidden_at(base, pos)
                else:
                    base, vis = encode_sample(model, tok, row['image'](), row['query'], cfg['max_num'])
                enc[pid] = [base, vis, len(ms), vis_hs]
                if store is not None:  # every method from the stored step 1, with one shared sampler state after it
                    ids1, fin1, info = stored_step1(tok, sp, store[pid], cfg['cut_sep'])
                    state = torch.Generator(device=lm.device).manual_seed(
                        job_seed(cfg['seed'], f'{pid}:stored_step1')).get_state()
                    s1info[pid] = {**info, 'step1_store_sha': store_sha[:12]}
                    t = time.time()
                    for m in ms:
                        j = new_job(pid, m, step1=(ids1, fin1, state))
                        j.t0 = t
                        active.append(j)
                    pending[pid] = []
                    continue
                j = new_job(pid, ms[0])
                j.t0 = time.time()
                pending[pid] = ms[1:]
                active.append(j)
            if not active:
                break
            reqs = [j.request() for j in active]
            ntok, groups, gstats = 0, pack(reqs, cfg['chunk'], cfg['batch_tokens']), []
            for g in groups:
                tg = time.time()
                res = gen_safe([reqs[i] for i in g])
                gtok = 0
                for i, r in zip(g, res):
                    if r is None:
                        if active[i].phase == 'probe_nov':  # logging-only probe: never fail the job for it
                            active[i].nov_failed()
                            continue
                        active[i].fail('non-finite logits')
                        continue
                    active[i].consume(r)
                    gtok += len(r.ids)
                ntok += gtok
                gstats.append([len(g), max(reqs[i].ctx.shape[0] for i in g), gtok, round(time.time() - tg, 1)])
            print(json.dumps({'event': 'round', 'elapsed': round(time.time() - t0), 'active': len(active), 'groups': gstats,
                              'tokens': ntok, 'queue': len(queue), 'peak_alloc_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else 0,
                              'peak_reserved_gb': round(torch.cuda.max_memory_reserved() / 2 ** 30, 2) if torch.cuda.is_available() else 0}), flush=True)
            # Step-1 thinking is identical for every method (same context and seed): fork followers from the leader.
            for j in list(active):
                if j.step1 is not None and pending.get(j.pid):
                    for m in pending.pop(j.pid):
                        fj = new_job(j.pid, m, step1=j.step1)
                        fj.t0 = j.t0
                        active.append(fj)
            for j in [j for j in active if j.done and j.out.get('status') == 'error' and pending.get(j.pid)]:
                for m in pending.pop(j.pid):
                    fj = new_job(j.pid, m)
                    fj.t0 = j.t0
                    fj.fail('leader failed before step 1')
                    active.append(fj)
            for j in [j for j in active if j.done]:
                active.remove(j)
                rec = {'pid': j.pid, 'method': j.method, 'answer_gt': j.row['answer'], **j.out,
                       **score(j.out['final_answer'], j.row), 'wall_s': round(time.time() - j.t0, 2),
                       'n_visual': int(j.vis.shape[0]), 'prompt_len': int(enc[j.pid][0].shape[0]),
                       'code_version': version}
                if j.pid in s1info:
                    rec['step1_info'] = s1info[j.pid]
                if rec['status'] == 'error':
                    rec['correct'] = 0
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                f.flush()
                print(json.dumps({'event': 'row', 'pid': j.pid, 'method': j.method, 'correct': rec['correct'],
                                  'k': rec['k'], 'think': rec['think_tokens'], 'stop': rec['stop_reason'],
                                  'wall': rec['wall_s'], 'elapsed': round(time.time() - t0)}), flush=True)
                enc[j.pid][2] -= 1
                if enc[j.pid][2] == 0:
                    del enc[j.pid]
                    s1info.pop(j.pid, None)
            if a.deadline and time.time() - t0 > a.deadline:
                with open(str(out) + '.partial', 'a') as pf:
                    for j in active:
                        pf.write(json.dumps({'pid': j.pid, 'method': j.method, 'k': j.k, 'phase': j.phase, 'steps': j.steps,
                                             'think_used': j.think_used, 'gen_tokens': j.gen_tokens}, ensure_ascii=False) + '\n')
                print(json.dumps({'event': 'deadline', 'active': len(active), 'queued': len(queue)}), flush=True)
                break
    print(json.dumps({'event': 'end', 'elapsed': round(time.time() - t0), 'left_in_queue': len(queue)}), flush=True)


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    main()
