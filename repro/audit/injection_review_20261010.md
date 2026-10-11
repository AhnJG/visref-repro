# VisRef injection-path review (InternVL3.5-1B reproduction)

Scope: implementation bugs / unintended behaviour in the visual-token re-injection path
(methods.py, engine.py, model.py, dpp.py, fastkv.py) vs. the paper's Algorithm 1 / Eq. 6-11 and the
official InternVL code. Documented design choices (Z = input embeddings, greedy stops at rank(L),
bare embeddings, `</think>` re-appended after V_k) are taken as given and only checked for internal
consistency. torch is not installed in the review container, so engine/fastkv behaviour is reasoned
from the code and the existing CPU tests; the DPP was re-checked numerically with a line-by-line
numpy port (`/tmp/claude-0/injection_review/dpp_check.py`). Anything I could not execute is marked
"unverified".

Paths are abbreviated: `repro/` = `/mnt/user-data/uploads/VisRef/repro/`,
`paper` = `/mnt/user-data/uploads/VisRef/references/papers/VisRef_fulltext/VisRef_plain.txt`,
`internvl` = `/mnt/user-data/uploads/VisRef/references/sources/internvl_modeling_internvl_chat.py`,
`chen` = `/mnt/user-data/uploads/VisRef/references/sources/cited_code/chen_fast-map-dpp/dpp.py`,
`rows` = `repro/audit/review_inputs/visref_sample_rows.jsonl`.

## Summary

**No BUG found in the injection path.** The tensor that is injected is exactly the projector output
the model sees for the image, the DPP is computed on the same tensor, the selected rows are placed
where the code says they are (z_k, [cut_sep], V_k, then `</think>`+probe / `Wait, think more`), the
step text used for the kernel is exactly the step just generated, positions/masks/KV are consistent
because every call re-prefills from the full embedding context, and the greedy matches Chen et al.
and the naive log-det greedy exactly. The four sample rows are arithmetically consistent with the
code (final context length reproduces to the token in all four rows).

What remains are RISKs — behaviours that are faithful to the documented design but are plausible
reasons VisRef does not beat TSR here, with cheap diagnostics to confirm or rule them out. None of
them is proven to cost accuracy.

---

## RISK findings (plausible, not proven)

### R1. `</think>` is appended immediately after a block of bare visual embeddings (design-adjacent)

Evidence. `repro/visref/methods.py:172-189`: the step context is built as
`ctx = [committed ctx] + embed(z + sep) + vis[sel]` (L172, L177) and then
`answer_ctx = ctx + embed([</think>])` (L189). For a step the model closed itself, its own
`</think>` is dropped from `z` (L164: `z = ids[:-1] if finish == 'stop'`) and a fresh `</think>` is
re-appended *after* V_k. For TSR the same `</think>` directly follows z_k. So VisRef's final answer and
every stopping probe are conditioned on a token sequence `... z_k [\n] <N raw visual vectors> </think>`
that the model has never seen in training; in the prompt, visual vectors only ever appear inside
`<img> ... </img>\n` followed by a *user* turn (`repro/visref/model.py:113-115`;
`internvl:244,295`). This is the paper's ordering (Fig. 2 text: z1 -> V1 -> "Wait, think more" -> ...
-> Vk -> "Answer:", `paper:319-350`; Alg. 1 L9/L17, `paper:520,529`), so it is not an
implementation error, but it is the most likely mechanism behind the reported 47 % `</think>`
re-emission inside VisRef answers: after a bare V-block the model's prior is "a user turn follows",
not "thinking is over". In the four sample rows there is no re-emission (`final_y_start == 0`,
`final_answer` contains no `</think>` in all four), so the 47 % figure could not be checked here
(unverified).

Expected effect if real: the final answer (and the probe used for stopping) is generated from an
out-of-distribution state; answers get longer/rambling (cf. pid 466: 1024-token answer after
`\nFinal Answer:` consisting of "Wait, no. Wait, wait..." that only resolves via DEER forcing), which
lowers VisRef accuracy relative to TSR independent of whether the injected tokens help.

Minimal fix (keeps the paper's order, changes only the "bare" choice, which is a documented design
knob): wrap the injected block in the model's native image delimiters, i.e.
`ctx = cat([ctx, embed([<img>]), vis[sel], embed([</img>])])` at `methods.py:177`, and account for
+2 tokens in `_vis_room` (`methods.py:75` already reserves `+2`). Alternatively (closer to "Answer:"
in Fig. 2) place V_k *before* the cut separator so the `</think>` follows text. Any variant should be
run as an explicit ablation, not silently.

Test that would catch it / diagnostic: from the full result file compute
`P(final_y_start > 0)` and `P(final_finish == 'cap')` by method x last-step `think_end`
(`close` vs `cap`). If TSR's re-emission rate is near zero and VisRef's is ~47 % for *closed* last
steps too, the bare V-block before `</think>` is the cause (not the cap). A unit test for the fix:
`test_visref_injected_block_is_delimited` asserting `answer_ctx[-2-n:-2]` equals `vis[sel]` and the
two surrounding rows equal `embed(<img>)`/`embed(</img>)`.

### R2. Step-1 and later steps are almost always cut by the 1024-token cap, so V_k is injected at an arbitrary truncation point

Evidence. The four rows have 11 of 13 steps with `think_end == 'cap'` and `think_tokens == 1024`
(e.g. `rows` pid 589 steps 1-6; pid 270 both steps), i.e. the run used `--think-cap 1024` (DEFAULTS in
`repro/run.py:20` say 4096, so the cfg differed from defaults; the per-row record stores
`code_version` but not `cfg`, so this is inferred, unverified). A cut step ends mid-sentence
("...angle at C (angle ACB) is also \n" then `<VISREF>`, pid 270), gets `cut_sep='\n'`
(`methods.py:170-172`) and then V_k and `Wait, think more`. The paper's z_k is a self-terminated
reasoning segment (`paper:293-297, 567`). This affects TSR identically, but it changes what
"re-focusing between reasoning steps" means and makes the final answer go through the s1 forced
`\nFinal Answer:` path (`methods.py:92-94`) in 2 of 4 rows (pid 270, 466, `final_forced_answer: true`).

Minimal fix: none needed in code; record `cfg` in each row (`run.py:209-212`) so the cap is auditable,
and report VisRef-TSR separately for jobs whose steps all closed naturally vs. cut.

Test: `rec` should include `cfg` (or at least `think_cap`, `total_think_cap`); a test that a row's
`steps[*].think_tokens <= cfg.think_cap` is trivially available once stored.

### R3. `_think_cap` reserves `m + 2` extra tokens only for VisRef

Evidence. `methods.py:66-75`: `room = max_context - ctx_len - answer_room - _vis_room() - 1`, with
`_vis_room() = budget(|V|, vis_frac) + 2` for VisRef and 0 otherwise. Also VisRef's committed context
grows by `n_inject` per step, so `room` shrinks faster. With `max_context = 40960` and the longest
sample context 11383 (pid 589), `room` is far from binding, so the effective cap is the same 1024 for
both methods here. It could bias the comparison only in runs with small `max_context`.

Fix: none required; a one-line assert/log when `room < think_cap` would make any future asymmetry
visible. Test: construct a Job with `max_context` such that `room` binds and assert TSR and VisRef
get the same `cap` only if that is intended.

### R4. Relative DPP stopping floor (1e-9 x max L_ii) vs Chen's absolute 1e-10

Evidence. `repro/visref/dpp.py:23,30` vs `chen:29`. In the current regime (Z = input embeddings,
rank(L) = #distinct tokens) the floor only triggers at rank exhaustion: in all 9 steps of the 4 rows
`n_dpp == min(m, n_unique_z)` exactly (pid 785: 194/194, 94/94; pid 466: 122/122, 130/130; pid 589:
156,126,134,138,131,154,130 all equal to `n_unique_z`; pid 270: 76 == m). Numerically (numpy port,
random V/Z, kernel scales 1e-7..1e5, rank-3 Z) the repro and Chen's code return the same 3 items and
the same picks in full-rank cases (see dpp_check.py output). A difference would appear only if a
genuine gain fell below 1e-9 of the largest diagonal, which never occurred here. Low risk.

Fix: none; if the DPP is later moved to hidden states (full rank), re-check that `n_dpp == m`.
Test: `test_dpp_rank_deficient_returns_only_greedy_picks` already covers this
(`repro/tests/test_repro.py:36-40`).

---

## OK findings (checked, fine)

1. **Injected tensor = what the model sees for the image.** `model.py:122-123`: `vis =
   model.extract_feature(pv).reshape(-1, hidden)`; `extract_feature` is ViT -> pixel-shuffle -> `mlp1`
   projector (`internvl:195-213`), the same function `generate()` uses before writing into the
   IMG_CONTEXT slots (`internvl:337,345`, no scaling). The prompt gets `emb[sel] = vis.to(emb.dtype)`
   (`model.py:126-128`, with an `assert` that #IMG_CONTEXT == |V| at L127) and the *same* `vis` object
   is returned and later indexed for injection (`methods.py:177`) and used for the DPP
   (`methods.py:176`). dtype: `pv` is cast to `lm.dtype` (fp16, `run.py`/`model.py:82,122`), the vision
   tower and `mlp1` are in the same dtype, so `vis` is fp16 like the text embeddings; no normalization.
2. **Tile set / thumbnail.** `load_image` uses `dynamic_preprocess(..., use_thumbnail=True, max_num=12)`
   (`model.py:75-78`) and the ImageNet transform (`model.py:21-28`), matching the official model-card
   preprocessing (unverified against the model card text, which is not among the given files; the
   row `n_visual` values 256 / 768 / 3328 = 1, 2+1, 12+1 tiles x 256 are consistent with it). V thus
   contains all tiles plus the thumbnail, exactly the rows that occupy the prompt's IMG_CONTEXT slots.
3. **z_k boundaries.** `partial` is reset at `_start_think` (`methods.py:80`) and only generated ids are
   appended (`methods.py:146`); the stop token (`</think>` or EOS) is stripped (`methods.py:164`); the
   `Wait, think more` ids are appended to the committed ctx (`methods.py:240`) but never to `partial`;
   the `cut_sep` ids are in ctx but not in `Z` (`methods.py:171-176`). Forked jobs copy the leader's
   step-1 ids and re-run `_end_think` (`methods.py:58-63`). Hence `Z = embed(z_k)` is exactly the
   current step's tokens (Eq. 6, `paper:447-458`) with no prompt/previous-step/marker leakage. The
   `<think>` opening tag generated by the model is part of z_1 (sample rows start with `"<think>\n"`),
   which is harmless.
4. **Insertion point and order.** Sequence per step: `... z_k [\n if cut and no trailing \n] V_k`
   then either `</think> + "\n\n\boxed"` (probe, `methods.py:88,189`) / `</think>` (+ `\nFinal Answer:`
   if cut) (final, `methods.py:92-96`) or `Wait, think more` -> z_{k+1} (`methods.py:240-249`). V_k rows
   are injected in ascending index order (`dpp.py:38` returns `sorted(sel)`), i.e. spatial/tile order.
   This matches the methods.py docstring (L7-11) and Fig. 2 / Alg. 1 of the paper. The trace markers
   in all 4 rows sit exactly between the step text and `Wait, think more`, with the last marker followed
   by nothing (final answer).
5. **Injected tokens really are in the probe/final context.** `final_context_len` = `answer_ctx.shape[0]`
   (`methods.py:256`) reproduces exactly in all 4 rows from `prompt_len + think_tokens + sum(n_inject)
   + #cut_seps + 4 x (k-1) [Wait, think more = 4 tokens] + 1 [</think>]`: 496+2048+152+2+4+1 = 2703;
   1003+1234+288+1+4+1 = 2531; 3507+2048+252+2+4+1 = 5814; 3512+6871+969+6+24+1 = 11383.
6. **Positions / masks / padding (engine).** Left padding (`engine.py:90-92`), positions
   `cumsum(mask)-1` so real tokens get 0..len-1 regardless of padding (`engine.py:93`), chunked prefill
   passes `attention_mask[:, :e]` of length past+q and `position_ids` for the chunk (`engine.py:100-103`),
   decode continues at `pos[:, -1]+1` and extends the mask by one column per step (`engine.py:106,148-152`).
   Injected rows are ordinary positions in `r.ctx`, exactly like the IMG_CONTEXT positions in the prompt
   (InternVL uses the LM's 1-D RoPE, no special image positions: `internvl:142-151,351-358`). Batched ==
   single == chunked is covered by `test_batched_chunked_equals_single`
   (`tests/test_repro.py:170-184`, including a 16-token chunk size that produces fully-padded query
   chunks) and fast-KV == stock by `test_fast_kv_matches_stock` (L213-229) plus the GPU self-check at
   startup (`run.py:40-55,122-128`). HF's handling of fully-masked pad query rows for torch >= 2.5 is
   relied upon (unverified here; the engine raises on non-finite logits, `engine.py:114-115`).
7. **KV cache after injection.** Each `Engine.generate` call builds a fresh cache from the full embedding
   context (`engine.py:94-103`); nothing is carried across calls, so the cache for the next step, the
   probe and the final answer always contains V_k (`methods.py:88,93,96,240`). `PreLayer` grows if the
   preallocation is too small (`fastkv.py:22-24`) and `batch_select_indices` keeps `n` (`fastkv.py:31-33`).
8. **TSR vs VisRef symmetry.** Same step-1 (forked from the leader, `run.py:194-199`), same `gen` /
   `probe_gen` seeds (`run.py:139-145`), same `Wait, think more`, `cut_sep`, probe prompt, forced prompt,
   answer caps; the only differences are V_k and R3's `_vis_room`. The probe for both is
   `answer_ctx + "\n\n\boxed"` with T=1 sampling (`methods.py:39,88,106`).
9. **DPP vs Eq. 9-11 and Chen.** `A = V Z^T`, `L = A A^T` (= v_i^T M_k v_j with M_k = sum z z^T,
   `dpp.py:21`); relevance `r_i^2 = L_ii = sum_j (v_i . z_j)^2` (`dpp.py:22`, Eq. 9 `paper:563`); greedy
   residual update `e = (L_j - c c_j)/sqrt(d2_j)`, `d2 -= e^2` (`dpp.py:34-37`) is Chen's `eis`/`di2s`
   update (`chen:21-27`) with `cis` transposed; stopping before adding an item whose residual is below
   the floor is equivalent to Chen's post-update check (`chen:28-31`). Numpy port: identical picks to
   Chen and to naive log-det greedy on 5 full-rank and 5 rank-deficient trials; repeated tokens in Z
   give n_sel = #unique rows; duplicate rows of V are never both selected (0/200). `sorted(sel)` maps
   directly to rows of the same `vis` tensor. `budget()` = `int(N*0.3)` equals floor(0.3 N) for all
   N = 256..3328 step 256 (checked, no float rounding issue).
10. **Sample-row internal consistency.** `n_inject == n_dpp <= m` everywhere; `n_dpp == min(m, n_unique_z)`
    in all 9 steps (rank(V Z^T) bound is tight, as the docstring `dpp.py:5-7` predicts);
    `rank_V` = 256/768 for 1-/3-tile images and 1009-1012 (< d = 1024) for 13-tile images; probe
    `probe_len == 3` ('{', X, '}') with merged-close entropy; marker text and `Wait, think more`
    continuation ("...carefully.") are as the code produces them.
11. **No in-place aliasing.** `base`/`vis` are shared across the three jobs of a pid (`run.py:169-170`),
    but every context update uses `torch.cat` (new tensors) and `encode_sample` clones the prompt
    embeddings before writing visual rows (`model.py:125`).

---

## Not checked / out of scope

- Exact text of the InternVL3.5 R1 system prompt and the `internvl2_5` conversation template
  (`model.py:6-14,111-115`): consistent with my recollection of the model card, but the model-card
  and `conversation.py` sources are not among the provided files (unverified).
- fp16 (`model.py:82`) vs the model card's bf16: affects prompt and injection identically; not an
  injection-path issue (unverified numerically).
- The 47 % `</think>` re-emission statistic itself (not reproducible from the 4 rows; see R1 for the
  diagnostic that would attribute it).
