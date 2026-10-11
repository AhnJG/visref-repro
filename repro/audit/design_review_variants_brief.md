# Brief for an independent design review (facts only)

Reproduction target: VisRef (arXiv:2603.00207), Appendix Table 6 row InternVL-1B on MathVista testmini (1,000 q):
paper ST 46.1 / TSR 48.5 / VisRef 52.0 (VisRef−ST +5.9, VisRef−TSR +3.5).

Main run (b_main, pre-registered, NOT to be changed): methods st / tsr / visref_raw_rank (code: methods.py).
- Thinking step cap 1024 tokens (total 10,240), answer cap 1024. Stop statistic H_k = token-entropy sum of one short
  answer sampled after `</think>` + "\n\n\boxed" (T=1). delta = 0.25, K_max = 10, m = floor(0.3|V|).
- VisRef injection: greedy DPP (Chen et al.) on kernel L = (V Z^T)(V Z^T)^T with Z = input embeddings of the step's
  text tokens; the greedy stops at the kernel rank, so in practice median n_inject/m = 0.21 (≈6% of |V|); only 16% of
  steps reach m. Selected visual embeddings are appended bare (no <img> tags) right after z_k. Following Algorithm 1,
  V_k is appended before the stop check, so the stop probe AND the final answer are generated after
  `... z_k V_k </think>`.
Interim results (759/1,000 questions, scorer v3; paired):
- ST 58.0 / TSR 60.2 / VisRef 59.6. TSR−ST +2.2 (95% CI 0.3..4.3); VisRef−ST +1.6 (−0.7..3.8); VisRef−TSR −0.7 (−2.8..1.6).
- Questions where any method continued (k>1), n=230: ST 33.5 / TSR 41.7 / VisRef 38.7. Both k=1, n=529: 68.6/68.2/68.6.
- `</think>` re-emitted inside the final answer: last step closed naturally ST 6.5% / TSR 7.3% / VisRef 45.1%;
  last step capped 17.0 / 13.8 / 50.0%. On k=1 & naturally-closed questions (n=244) correct: TSR 176 vs VisRef 175.
- An independent code review found no implementation bug in the injection path (tensor = projector output the model
  sees, positions/KV consistent, DPP equals Chen's greedy). It flagged as main risk that `</think>` directly follows a
  block of bare visual embeddings, a sequence the model never saw in training.

Proposed follow-up experiments (to run after b_main completes; VisRef-only, same 392 question ids and seeds as the
first sessions, paired with the existing ST/TSR/VisRef rows):
1. `visref_fill`: after the greedy stops at the kernel rank, keep selecting with a tiny ridge L + eps*I
   (eps = 1e-6 * mean diag) until exactly m tokens are selected (pre-rank picks unchanged; post-rank picks ≈ relevance
   order). Motivation: the paper says the selection enforces |V_k| = m and terminates after m iterations.
2. originally `visref_nolast` (inject V_k only when continuing; stop probe and final answer without V_k). Withdrawn
   because it changes the stop decisions (the entropy is computed on a different context), confounding two effects.
   Revised proposal `visref_noansinj`: identical to visref_raw_rank for every step and for the stop probe (probe still
   sees V_k, so k and the stop decisions are the same as VisRef up to fp16 noise); only the final answer is generated
   from the context without the last V_k block (`... z_k </think>`). Purpose: isolate whether injecting V_k right
   before the answer hurts the answer.
Not pre-registered: wrapping injected tokens in <img>...</img> (an early smoke test showed the model then hallucinated
a new question and a second </think>).
