"""ST / TSR / VisRef as per-job state machines (paper Eq. 1, Sec. 3, Algorithm 1).

Methods: 'st' (one thinking step, then the answer), 'tsr' (textual self-reflection [34]) and 'visref_raw_rank'
(VisRef: after each step the visual tokens chosen by DPP greedy MAP are appended as bare embeddings; only the
greedy picks, <= rank(L), as the cited Chen et al. code does) and 'visref_fill' (same, but after the greedy stops at
rank(L) a ridge eps*I continues it to exactly m tokens: dpp.select_tokens_fill; pre-registered variant, EF 2026-10-10/11)
and 'visref_hs' (same as visref_raw_rank, but Z = the last-layer hidden states of z_k (after the final norm) instead
of its input embeddings, Engine.last_hidden; pre-registered variant, EF 2026-10-12).

Dual probe (cfg dual_probe, VisRef methods only, logging only): after the stopping probe a second probe is sampled on
z_k without V_k (`... z_k </think>` + probe prompt) from its own generator; its H_noV is recorded on the step and is
never used for the decision, so every decision, context and answer is the same as without it.

Step k: sample z_k until the model emits </think> (or EOS / the step cap). A step cut by the cap gets 'cut_sep'
('\n') right after its text unless it already ends with one -- before V_k, </think> and the continuation text --
as s1's official eval does (s1/vllm_causallms.py L401-409); the separator is context only.

Stopping probe (paper Sec. 4.1.2): after z_k (and V_k) append </think> + probe_prompt ("\n\n\\boxed", the answer
format InternVL3.5 uses by itself) and sample ONE answer y at T=1 without top-k/top-p; y ends at the box's own
closing brace (brace depth back to 0, <= probe_cap tokens). The prompt stops before '{' so the model emits the
brace itself. H_k = sum of token entropies; at box depth 1 all tokens whose first non-space character is '}'
(close the box, nothing added) are merged into one outcome, elsewhere the full-vocabulary entropy is used. By
the chain rule E_y[H_k] = H(Y), so this is a single-sample estimate of H_k = -E_y[log pi(y | x, tau_1:k)].
A box with no letter/digit inside, EOS before the box closes, or probe_cap reached gives H = inf. The probe text
is discarded. If H_k < delta, k = Kmax or the thinking budget is exhausted, the final answer is sampled after
</think> (Algorithm 1 line 17); otherwise 'Wait, think more' is appended and step k+1 starts.

Final answer: after a natural </think> the model answers freely (answer_cap, EOS). After a step cut by the cap,
</think> + forced_prompt ('\nFinal Answer:', s1 budget forcing, L254) is appended first and the scored text is
forced_prompt + the answer. A step that ended with EOS gets </think> and a natural answer. Any final answer that
reaches answer_cap without EOS is followed by af_prompt ('\n**Final Answer**\n\\boxed') and at most af_cap more
tokens (EOS stop), as DEER's official code does with af=1 (deer/vllm-deer.py L485-491, L655-656); the scored text
then ends with that continuation, so its last \\boxed{} is scored. Scored text = what follows the last </think>
the model itself generated in the answer, if any.

Common random numbers: the probe draws from its own generator (probe_gen), so the main generator's state when the
final answer starts is its state right after the last thinking step; a TSR job that stops at k=1 then samples the
same final answer as ST (same context, same draws). The distribution of each method is unchanged.
"""
import torch

from .dpp import budget, pick_stats, select_tokens, select_tokens_fill
from .engine import Request

METHODS = ('st', 'tsr', 'visref_raw_rank', 'visref_fill', 'visref_hs')
VISREF = ('visref_raw_rank', 'visref_fill', 'visref_hs')
PROBE_SAMPLING = (1.0, 0, 1.0)  # T=1, no top-k/top-p: sampled from the distribution whose entropy is summed


class Job:
    def __init__(self, method, row, base, vis, cfg, tok, engine, special, gen, probe_gen, step1=None, probe2_gen=None):
        """step1=(ids, finish, gen_state): reuse another job's step-1 thinking (identical context and seed).
        probe_gen: generator used only by the stopping probe.
        probe2_gen: generator of the logging-only second probe without V_k (cfg dual_probe, VisRef methods only)."""
        assert method in METHODS
        self.method, self.row, self.cfg, self.tok, self.eng, self.sp = method, row, cfg, tok, engine, special
        self.vis, self.gen, self.probe_gen = vis, gen, probe_gen
        self.dual = bool(cfg.get('dual_probe')) and method in VISREF
        assert not self.dual or probe2_gen is not None
        self.probe2_gen = probe2_gen
        self.gen_tokens_noV = 0
        self.forced_answer = False
        self.ctx = base  # committed context [L, d]
        self.k, self.think_used, self.gen_tokens = 1, 0, 0
        self.steps, self.trace = [], []  # trace: decoded thinking text with markers
        self.done, self.out, self.step1 = False, None, None
        self.forked = step1 is not None
        self.forced_ids = tok.encode(cfg['forced_prompt'], add_special_tokens=False)
        self.af_ids = tok.encode(cfg['af_prompt'], add_special_tokens=False)
        self._start_think()
        if step1 is not None:
            ids, finish, state = step1
            self.gen.set_state(state)
            self.partial = list(ids)
            self.gen_tokens += len(ids)
            self._end_think(finish)

    # ---- phases -------------------------------------------------------------
    def _think_cap(self, ctx_len):
        answer_room = max(self.cfg['answer_cap'] + len(self.af_ids) + self.cfg['af_cap'] + len(self.forced_ids),
                          self.cfg['probe_cap'] + len(self.sp['probe_prompt']))
        room = self.cfg['max_context'] - ctx_len - answer_room - self._vis_room() - 1
        return min(self.cfg['think_cap'], self.cfg['total_think_cap'] - self.think_used, room)

    def _vis_room(self):
        if self.method not in VISREF:
            return 0
        return budget(self.vis.shape[0], self.cfg['vis_frac']) + 2  # + 2: margin (kept so caps match earlier runs)

    def _start_think(self):
        cap = self._think_cap(self.ctx.shape[0])
        assert cap > 0, 'prompt exceeds max_context'
        self.phase, self.cur, self.partial, self.cap = 'think', self.ctx, [], cap
        self.stop = frozenset({self.sp['think_end'], self.sp['eos']})

    def _start_answer(self, phase, forced, reason=None):
        """phase 'final': answer after </think>; phase 'probe': stopping-probe answer after the probe prompt."""
        self.phase, self.ent, self.ent_l = phase, [], []
        self.stop_check, self.ent_lump = None, frozenset()
        if phase == 'probe':
            self.cur = torch.cat([self.answer_ctx, self.eng.embed(self.sp['probe_prompt'])], 0)
            self.cap = self.cfg['probe_cap']
            self.stop_check = lambda ids: self._brace_scan(self.partial + ids, decode=False)[0] is not None
            self.ent_lump = self.sp['close_ids']
        elif forced and self.cut:  # s1 budget forcing: the thinking was cut, so prompt the answer
            self.cur = torch.cat([self.answer_ctx, self.eng.embed(self.forced_ids)], 0)
            self.cap = self.cfg['answer_cap']
        else:
            self.cur, self.cap = self.answer_ctx, self.cfg['answer_cap']
        self.forced_answer = phase == 'final' and bool(forced) and self.cut
        self.partial = []
        self.stop = frozenset({self.sp['eos']})
        self.forced, self.reason = forced, reason or getattr(self, 'reason', None)

    def request(self):
        if self.phase == 'think':
            return Request(self.cur, self.stop, self.cap - len(self.partial), self.gen)
        probe = self.phase in ('probe', 'probe_nov')
        pgen = self.probe2_gen if self.phase == 'probe_nov' else self.probe_gen
        return Request(self.cur, self.stop, self.cap - len(self.partial), pgen if probe else self.gen,
                       record_entropy=probe, sampling=PROBE_SAMPLING if probe else None,
                       stop_check=self.stop_check, ent_lump=self.ent_lump)

    def _brace_scan(self, ids, decode=True, prompt=None):
        """Brace depth over a box answer. A prompt ending in '\\boxed' leaves the '{' to the model (depth 0, box not
        yet open); a prompt ending in '{' has opened it (depth 1). Returns (index of the token that closes the box
        or None, text inside the box, depth before each token)."""
        prompt = self.cfg['probe_prompt'] if prompt is None else prompt
        bt, before = self.sp['brace_text'], []
        depth = 1 if prompt.endswith('{') else 0
        opened = depth > 0
        for n, t in enumerate(ids):
            before.append(depth)
            for ch in bt.get(t, ''):
                if ch == '{':
                    depth, opened = depth + 1, True
                elif ch == '}' and opened:  # a '}' before the box opens is ignored
                    depth -= 1
                if opened and depth == 0:
                    return n, (self._box_inner(ids[:n + 1], prompt) if decode else None), before
        return None, (self.tok.decode(ids) if decode else None), before

    def _box_inner(self, ids, prompt):
        """Text between the box's opening and closing braces in the decoded answer."""
        text = self.tok.decode(ids)
        depth = 1 if prompt.endswith('{') else 0
        start = 0 if depth else None
        for i, ch in enumerate(text):
            if ch == '{':
                depth += 1
                if start is None:
                    start = i + 1
            elif ch == '}' and start is not None:
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start:i]
        return text[start or 0:]

    def consume(self, res):
        self.partial += res.ids
        if self.phase == 'probe_nov':  # logging-only probe: counted apart so gen_tokens stays the decision path's
            self.gen_tokens_noV += len(res.ids)
        else:
            self.gen_tokens += len(res.ids)
        if self.phase in ('probe', 'probe_nov'):
            self.ent += res.ent
            self.ent_l += res.ent_lump
        self.cur = torch.cat([self.cur, self.eng.embed(res.ids)], 0)
        if res.finish == 'chunk':
            return
        if self.phase == 'think':
            self._end_think(res.finish)
        else:
            self._end_answer(res.finish)

    def _end_think(self, finish):
        sp, ids = self.sp, self.partial
        if self.k == 1 and self.step1 is None:  # shareable with other methods of the same pid
            self.step1 = (list(ids), finish, self.gen.get_state())
        closed = finish == 'stop' and ids[-1] == sp['think_end']
        z = ids[:-1] if finish == 'stop' else ids
        self.think_used += len(z)
        step = {'k': self.k, 'think_tokens': len(z),
                'think_end': 'close' if closed else ('eos' if finish == 'stop' else 'cap')}
        self.cut = finish == 'cap'
        zt = self.tok.decode(z)
        sep = self.cfg['cut_sep'] if self.cut and not zt.endswith('\n') else ''
        sep_ids = self.tok.encode(sep, add_special_tokens=False) if sep else []
        ctx = torch.cat([self.ctx, self.eng.embed(list(z) + sep_ids)], 0) if (z or sep_ids) else self.ctx
        self.trace.append(zt + sep)
        self.ctx_nov = ctx  # z_k (+ separator) without V_k: context of the logging-only second probe
        if self.method in VISREF and z:
            m = budget(self.vis.shape[0], self.cfg['vis_frac'])
            if self.method == 'visref_fill':
                sel, fill_info = select_tokens_fill(self.vis, self.eng.embed(z), m)  # asserts head = raw_rank picks
                step.update(fill_info)
            elif self.method == 'visref_hs':  # Z = last-layer hidden states of z_k in its own context (causal: the
                tr = []                      # separator / V_k appended after z_k do not change them)
                Z = self.eng.last_hidden(torch.cat([self.ctx, self.eng.embed(z)], 0), len(z))
                sel = select_tokens(self.vis, Z, m, trace=tr)
                step.update(pick_stats(tr))
            else:
                sel = select_tokens(self.vis, self.eng.embed(z), m)
            ctx = torch.cat([ctx, self.vis[sel]], 0)
            # rank diagnostics: n_dpp <= rank(V Z^T) <= min(rank V, #distinct tokens in z_k)
            if not hasattr(self, 'rank_V'):  # diagnostic only: CPU, never fatal; 1e-4 is above fp16 noise
                try:
                    s = torch.linalg.svdvals(self.vis.float().cpu())
                    self.rank_V = int((s > 1e-4 * s.max()).sum())
                except Exception:
                    self.rank_V = -1
            step.update(m=m, n_dpp=len(sel), n_inject=len(sel), rank_V=self.rank_V, n_unique_z=len(set(z)))
            self.trace.append(f'<VISREF m={m} dpp={len(sel)}>')
        self.steps.append(step)
        self.step_ctx = ctx
        self.answer_ctx = torch.cat([ctx, self.eng.embed([sp['think_end']])], 0)
        phase = 'final' if self.method == 'st' else 'probe'
        self._start_answer(phase, forced=not closed, reason='st' if self.method == 'st' else None)

    def _end_answer(self, finish):
        eos = self.sp['eos']
        if self.phase == 'final_af':  # DEER answer forcing finished
            text, raw, y0, finish0, extra = self.af_pre
            ap = self.cfg['af_prompt']
            close, inner, _ = self._brace_scan(self.partial, prompt=ap)
            text = text + ap + self.tok.decode([t for t in self.partial if t != eos])
            extra = dict(extra, final_af=True, final_af_len=len(self.partial),
                         final_af_finish='eos' if finish == 'stop' else 'cap',
                         final_af_valid=close is not None and any(ch.isalnum() for ch in inner))
            self.partial = self.partial_answer
            self._finish(text, raw, y0, finish0, **extra)
            return
        if self.phase == 'final':
            te = [i for i, t in enumerate(self.partial) if t == self.sp['think_end']]
            y0 = te[-1] + 1 if te else 0
            text = self.tok.decode([t for t in self.partial[y0:] if t != eos])
            extra = {}
            if self.forced_answer:  # the prompt 'Final Answer:' is part of the answer being scored
                if not y0:
                    text = self.cfg['forced_prompt'] + text
                extra = {'final_forced_valid': finish == 'stop'}
            raw = self.tok.decode([t for t in self.partial if t != eos]) if y0 else None
            if finish != 'stop':  # answer hit the cap without EOS: DEER answer forcing
                self.af_pre = (text, raw, y0, finish, extra)
                self.partial_answer = self.partial
                self.phase, self.partial, self.cap = 'final_af', [], self.cfg['af_cap']
                self.cur = torch.cat([self.cur, self.eng.embed(self.af_ids)], 0)
                self.stop, self.stop_check = frozenset({eos}), None
                return
            self._finish(text, raw, y0, finish, **extra)
            return
        if self.phase == 'probe_nov':  # logging-only probe without V_k: record, then act on the stored decision
            H, valid, ended, inner, ent = self._probe_entropy(finish)
            self.steps[-1].update(H_noV=H, probe_noV_finish='end' if valid else ('cap' if not ended else 'invalid'),
                                  probe_noV_answer=self.tok.decode([t for t in self.partial if t != eos]),
                                  probe_noV_inner=inner, probe_noV_ent=[round(e, 4) for e in ent])
            self._decide(self._pending_H)
            return
        # stopping probe
        H, valid, ended, inner, ent = self._probe_entropy(finish)
        self.steps[-1].update(H=H, probe_finish='end' if valid else ('cap' if not ended else 'invalid'),
                              probe_len=len(self.partial),
                              probe_answer=self.tok.decode([t for t in self.partial if t != eos]),
                              forced_close=self.forced, probe_ids=list(self.partial),
                              probe_ent=[round(e, 4) for e in ent], probe_inner=inner, probe_valid=valid,
                              probe_ent_full=[round(e, 4) for e in self.ent])
        if self.dual:  # decision uses H only; the second probe has its own generator and never feeds the context
            self._pending_H = H
            self._start_probe_nov()
            return
        self._decide(H)

    def _probe_entropy(self, finish):
        ended = finish == 'stop'
        close, inner, depth = self._brace_scan(self.partial)
        assert len(self.ent) == len(self.ent_l) == len(self.partial) and len(depth) <= len(self.ent)
        valid = ended and close is not None and any(ch.isalnum() for ch in inner)
        # merged-close entropy where the box is at depth 1, full-vocabulary entropy elsewhere
        ent = [l if d == 1 else f for f, l, d in zip(self.ent, self.ent_l, depth)]
        H = sum(ent) if valid else float('inf')
        return H, valid, ended, inner, ent

    def nov_failed(self):
        """The logging-only probe hit non-finite logits: record it and go on with the stored decision."""
        self.steps[-1].update(H_noV=None, probe_noV_finish='error')
        self._decide(self._pending_H)

    def _start_probe_nov(self):
        """Second stopping probe on z_k (+ separator) without V_k, then </think> + probe prompt (logging only)."""
        self.phase, self.ent, self.ent_l, self.partial = 'probe_nov', [], [], []
        self.cur = torch.cat([self.ctx_nov, self.eng.embed([self.sp['think_end']]), self.eng.embed(self.sp['probe_prompt'])], 0)
        self.cap = self.cfg['probe_cap']
        self.stop = frozenset({self.sp['eos']})
        self.stop_check = lambda ids: self._brace_scan(self.partial + ids, decode=False)[0] is not None
        self.ent_lump = self.sp['close_ids']

    def _decide(self, H):
        cfg = self.cfg
        nxt = torch.cat([self.step_ctx, self.eng.embed(self.sp['wait'])], 0)  # a cut step already ends with cut_sep
        reason = ('entropy' if H < cfg['delta'] else 'kmax' if self.k >= cfg['kmax']
                  else 'budget' if self._think_cap(nxt.shape[0]) < cfg['min_step_tokens'] else None)
        if reason:
            self._start_answer('final', self.forced, reason)
            return
        self.ctx = nxt
        self.trace.append(self.cfg['wait_text'])
        self.k += 1
        self._start_think()

    def _finish(self, text, raw, y0, finish, **extra):
        self.done = True
        self.out = {'steps': self.steps, 'k': self.k, 'stop_reason': self.reason,
                    'final_forced_close': self.forced, 'final_answer': text, 'final_raw': raw, 'final_y_start': y0,
                    'final_finish': 'eos' if finish == 'stop' else 'cap', 'final_len': len(self.partial),
                    'think_tokens': self.think_used, 'final_context_len': int(self.answer_ctx.shape[0]),
                    'gen_tokens': self.gen_tokens, 'forked': self.forked, 'status': 'ok',
                    'thinking': ''.join(self.trace), 'final_forced_answer': self.forced_answer}
        if self.dual:
            self.out['gen_tokens_noV'] = self.gen_tokens_noV
        self.out.update(extra)

    def fail(self, why):
        """Record a failed job (scored as wrong; counted separately in the summary)."""
        self.done = True
        self.out = {'steps': self.steps, 'k': self.k, 'stop_reason': 'error', 'final_forced_close': False,
                    'final_answer': '', 'final_raw': None, 'final_y_start': 0, 'final_finish': 'error',
                    'final_len': 0, 'think_tokens': self.think_used, 'final_context_len': 0,
                    'gen_tokens': self.gen_tokens, 'forked': self.forked,
                    'status': 'error', 'error': why, 'thinking': ''.join(self.trace),
                    'final_forced_answer': False}
