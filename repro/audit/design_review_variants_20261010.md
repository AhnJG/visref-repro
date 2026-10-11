# Design review: `visref_fill` and `visref_noansinj`

Reviewer role: independent methodologist. Sources: `brief.md` (B:line), `VisRef_plain.txt` (P:line), `methods.py` (M:line). Facts are cited; everything else is marked **Judgment**. Numbers from the brief are used as given (B) and were not re-derived from logs. Chen et al. 2018's exact stopping rule was not available in this container: 미확인, taken from B:10.

## 0. Verdict summary

| Variant | Label w.r.t. paper | Verdict | Run order |
|---|---|---|---|
| `visref_fill` | faithful reinterpretation of an under-specified case (the size-m requirement when rank(L_k) < m) — NOT the paper's intended "m relevant and diverse tokens" | KEEP with changes (§6) | 1st |
| `visref_noansinj` | deliberate deviation / diagnostic ablation (paper conditions y on V_k unambiguously) | KEEP with changes: run as *replay* and analyse the k>1 subset; k=1 rows are already known from existing data | 2nd (or fold into free analysis) |
| dual-probe logging (new, §5) | diagnostic only, decisions unchanged | ADD to the next run | with 1st |
| hidden-state Z (new, §3) | alternative reading of P:450-451 | PROPOSE conditionally, after a rank check | 3rd |

## 1. Paper fidelity

### 1(a) Is the final answer conditioned on the last V_k?
Facts.
- Eq. 3: τ_{1:k} = ((z_1,V_1),…,(z_k,V_k)) (P:244-245). §3: "generates the final answer as y ∼ πθ(· | x_input, τ_{1:k})" (P:273-274).
- Algorithm 1: line "τ_{1:k} ← τ_{1:k−1} ∪ {(z_k, Ṽ_k)}" (P:520) precedes the stop check (P:521-525); "16: // Generate final answer 17: Sample y ∼ πθ(· | x_input, τ_{1:k})" (P:528-529).
- §4.1.2: "After termination of the reasoning trajectory, the final answer is generated as: y ∼ πθ(· | x_input, τ_k)" (P:728). Figure 2 places "Selected Visual Tokens Vk" immediately before "Answer:" (P:346-350).
- Code: `answer_ctx = cat([ctx(with V_k), </think>])` (M:177, M:189); final answer starts from `answer_ctx` (M:96, M:244).

Judgment. The paper is unambiguous: y is sampled from the trajectory that includes V_k. The main run is faithful. `visref_noansinj` samples y from a context the paper never defines (`… z_k </think>` with V_1..V_{k−1} still present for k>1). It is a **deliberate deviation**, to be labelled a diagnostic ablation, not a reading of an ambiguity. It is also internally inconsistent with the paper's logic: the stop test certifies low entropy of π(y | x, τ_{1:k}) *with* V_k (P:605-606), and noansinj then samples from a different distribution whose entropy was never checked. That is acceptable for a diagnostic, but it must not be presented as a candidate implementation of VisRef.

### 1(b) Is the stop entropy computed after V_k is added?
Facts. Algorithm 1 order: select Ṽ_k (P:517) → update τ (P:520) → "H_k ← −E_y[log πθ(y | x_input, τ_{1:k})]" (P:523) → break if H_k < δ (P:524-525). §4.1.2 defines H_k "given the current visual-integrated trajectory τ_{1:k} = {(z_1,V_1),…,(z_k,V_k)}" (P:602-606). Code: probe context = `answer_ctx` (with V_k) + probe prompt (M:88-89).

Judgment. Faithful. Both proposed variants keep this; the withdrawn `visref_nolast` would have deviated here, and the withdrawal was correct (it would have changed k and the stop decisions, B:28-29).

### 1(c) |V_k| = m and termination after m iterations
Facts. "We introduce a token budget m … enforcing |V_k| = m" (P:534-536); Algorithm 1 "Ṽ_k ← Greedy selection of m tokens via Eq. (11)" (P:517); "We terminate after m iterations, setting Ṽ_k ← V_k^{(m)}" (P:578-581); App. G writes V_k = {v_1,…,v_m} and D = diag(r_1,…,r_m) (P:1344, P:1498). m = ⌊0.3|V|⌋ (P:738, P:1287). The greedy gain is log det(L^{S∪{v}})/det(L^S) (P:548-555); for rank-deficient L this is −∞ for every candidate once |S| = rank(L_k), so Eq. 11's argmax is undefined past the rank. The paper never mentions rank deficiency. Main run: greedy stops at the kernel rank, median n_inject/m = 0.21, only 16% of steps reach m (B:10-11).

Judgment. The text promises m tokens; the formula cannot deliver more than rank(L_k) tokens with Z = input embeddings of z_k (rank ≤ #distinct tokens in z_k, M:178). This is an **under-specification**, not an ambiguity the authors acknowledged. `visref_raw_rank` resolves it the way the cited greedy code does (B:10; Chen et al. rule 미확인); `visref_fill` resolves it by continuity (L + εI → L). Both are legitimate resolutions. Neither realises "m relevant AND diverse tokens": past the rank the diversity term log det(L̄) (P:587-594) is identically degenerate, so fill's tail picks carry no diversity information (see §3). Label: **faithful reinterpretation of an under-specified case**.

One inference worth stating in the report (Judgment, not fact): the paper's own 8B numbers — ~1,772 visual vs ~615 text tokens per step (P:385), m = 0.3·1772 ≈ 531 — mean the authors could only have obtained |V_k| = m if their kernel had rank ≥ 531 per step, which input embeddings of a ~615-token step (many repeated tokens) will rarely provide, while per-position hidden states (all distinct) would. This favours a hidden-state reading of "text token embeddings in reasoning state z_k" (P:450-451), but the text does not settle it.

## 2. Causal validity of `visref_noansinj`

Facts from the code.
- The probe draws from `probe_gen`; the main generator's state at the start of the final answer is its state right after the last thinking step (M:29-31, M:106). The final answer uses `answer_ctx` (M:96); noansinj would use the same context minus the V_k block.
- Step-1 thinking is shared across methods of the same question (`step1` fork, M:44-45, M:58-63, M:161-162).
- Forced path: a capped last step gets `</think>` + `\nFinal Answer:` (M:92-94); an answer hitting `answer_cap` gets the af continuation (M:216-222). Both are rules applied identically to any variant.

Judgment on isolation. Within a question, noansinj vs VisRef differ only in the presence of the last V_k block in the answer context (same z_1..z_k, same V_1..V_{k−1}, same probe, same k, same generator state). Given identical logits, the answer tokens are identical until the context difference changes a draw. This is as clean as a single-factor design gets. Remaining confounds and caveats:

1. **fp16 batch-composition noise** (B:18, flagged by proposer). If the trajectory is re-generated, a single changed token in any step changes k, the stop decision and the whole trajectory for that question; "identical up to fp16 noise" then becomes "identical for most questions, unknown for the rest". Fix: run noansinj as a **replay** — reconstruct the stored VisRef trajectory (token ids of z_1..z_k, cut separators, V_k selections) and generate only the final answer. `_finish` does not currently store step token ids (M:251-259 store decoded text only); add them (or re-encode the trace and assert the round-trip). If replay is impossible, report the fraction of questions whose thinking trace is byte-identical to the main run and restrict the paired analysis to them.
2. **The k=1 rows are already known.** For a question where VisRef stops at k=1, noansinj's answer context is `base + z_1 (+sep) + </think>` with the generator in the post-step-1 state — exactly ST's context and state (M:29-31, M:189-191). So noansinj ≡ ST on every VisRef-k=1 question (up to fp16 noise). The brief already contains this comparison: on both-k=1 questions ST 68.6 vs VisRef 68.6 (n=529, B:16), and on k=1 & naturally-closed TSR 176 vs VisRef 175 (n=244, B:18). The net answer-injection effect at k=1 is ≈ 0 and can be computed exactly from existing logs for the whole VisRef-k=1 subset (≈70% of questions) without any GPU time. The only new information noansinj can produce is on the k>1 subset (≈30% of questions, B:16 → ≈115 of 392).
3. **The stop probe itself sees the bare V block.** noansinj keeps this (correctly, §1(b)), so it cannot tell whether H_k is distorted by the V block (e.g., over- or under-confidence causing early/late stops). That question needs the dual-probe diagnostic (§5), not a decision-changing variant.
4. **Forced/af paths** are not confounds (same rule both arms) but are plausible *mechanisms*: VisRef re-emits `</think>` inside 45.1% / 50.0% of answers vs 7.3% / 13.8% for TSR (B:17-18); longer or restarted answers hit `answer_cap` more often and get the af continuation (M:216-222), which changes which `\boxed{}` is scored. Pre-register as secondary outcomes: re-emission rate, `final_af` rate, `final_len`, and accuracy conditional on re-emission.
5. **Context-length effect** is tiny here (median n_inject ≈ 0.21·m, B:11) and is part of the factor under test, not a confound.

Does it answer "does injecting V_k right before the answer hurt?" Yes, for the *last* block only and for the k>1 subset only in a non-trivial way; for k=1 the answer is already "no net effect" (B:16). Misreading risk: a positive noansinj−VisRef difference will be read as "VisRef's injection hurts" (it only speaks to the final block; steps still had V_k) and a null as "VisRef works" (it says nothing about the gap to TSR, which lives on the continued questions: TSR 41.7 vs VisRef 38.7, n=230, B:16). Wording is given in §6.

## 3. `visref_fill`

Facts. Proposal: after the greedy stops at the kernel rank, continue on L + εI (ε = 1e-6·mean diag) until exactly m picks; pre-rank picks unchanged (B:25-27). Paper's amount ablation (8B, MathVista): 76.1% → 79.2% from m = 20% to 30%, plateau at ≥ 30% (P:794-795, P:930-932). Selection-strategy ablation (8B): Random 67.3 ≈ ST 68.1, Relevance-only 75.6, DPP 79.3 on MathVista (P:1579-1589, P:1547). Main run's effective fraction ≈ 6% of |V| (B:11), below the lowest value the paper tested.

Judgment on defensibility. L + εI is a valid, full-rank DPP kernel (an isotropic diversity term added to L), and the ε → 0 limit of its greedy MAP coincides with the paper's greedy on the first rank(L_k) picks, so "run Eq. 11 to m iterations on L + εI" is a defensible operationalisation of P:534-536/578-581. What it is not: a realisation of "relevant and diverse" past the rank. For a candidate v in the span of the selected set S, the Schur complement under the ridge is ≈ ε(1 + ‖L_S^{-1} L_{Sv}‖²) (first-order in ε), so the tail is ordered by the norm of v's coordinates in the selected basis — correlated with relevance r_i² (P:563) but not equal to it, and with no coverage/diversity content. Verify empirically (Spearman of tail order vs r_i²) rather than asserting "≈ relevance order" (B:26). In the paper's vocabulary, fill = DPP head (rank(L_k) tokens) + a relevance-like tail to m; Table 7 suggests such a tail is still far better than random at 8B (75.6 vs 67.3, P:1583-1589), so if the paper's mechanism transfers, fill should show a sizeable gain.

What fill tests: "does injecting ~m tokens instead of ~0.2m change accuracy, holding the DPP head fixed?" — the injection-amount factor. What it cannot test: (i) whether m *diverse* tokens (the paper's intent) would do better than this tail; (ii) whether Z should be hidden states; (iii) amount separately from the format risk — more bare embeddings before `</think>` may change the re-emission behaviour (B:17-21), so report re-emission and `final_af` rates per arm; (iv) longer contexts shrink `_think_cap` room faster (M:66-70) and may increase 'budget' stops at large k — report `stop_reason` and k distributions per arm.

Required technical checks (KEEP-with-changes items): assert the first n_dpp picks equal `visref_raw_rank`'s on the same step; do the ridge Cholesky/Schur updates in fp32 or fp64 (condition number ~1e6 at ε = 1e-6·mean diag; fp16 will not resolve the tail order); log n_inject/m (should be exactly 1.0 unless |V| < m).

Better single variant for "amount"? Judgment: for isolating *amount*, fill is the cleaner design (one factor, head fixed). Hidden-state Z is the better candidate for *the paper's intended method* (rank up to min(T_k, d) → DPP semantics survive to m on most steps, and §1(c)'s inference favours it), but it changes two things at once (representation and amount). Recommended sequencing: fill now; before any hidden-state run, compute offline from existing traces the fraction of steps with rank(hidden Z) ≥ m (zero GPU generation needed) and only run it if that fraction is high. A random-fill control (DPP head + random tail to m; the paper's Table 7 Random analog) is the cheapest way to separate "more tokens" from "more relevant tokens" if fill shows a gain.

## 4. Power of the 392-question paired design

Facts. VisRef vs TSR discordance ≈ 67/759 ≈ 8.8% (task statement); the brief's CIs are consistent with this (VisRef−TSR −0.7, CI −2.8..1.6 at n=759 → SE ≈ 1.1 pp, B:15). Paper target VisRef−TSR +3.5 pp (P:1545, P:1534; B:4).

Computation (paired difference of correctness indicators; Var per pair ≈ discordance − Δ²; two-sided α = 0.05):

| n | discordance | SE of Δ | power for Δ = 3.5 pp | power for Δ = 2 pp | MDE (80%) |
|---|---|---|---|---|---|
| 392 | 8.8% | 1.49 pp | 0.65 | 0.27 | 4.2 pp |
| 759 | 8.8% | 1.07 pp | 0.91 | 0.46 | 3.0 pp |
| 1000 | 8.8% | 0.93 pp | 0.96 | 0.57 | 2.6 pp |
| k>1 subset, 115 | 15% (assumed) | 3.6 pp | — | — | 10.1 pp |
| k>1 subset, 300 | 15% (assumed) | 2.2 pp | — | — | 6.3 pp |

(McNemar on ≈34.5 expected discordant pairs gives the same ≈0.66 for 3.5 pp.)

Judgment. For `visref_fill` vs `visref_raw_rank` the discordance should be of the same order as VisRef vs TSR (trajectories diverge from the first probe on most questions), so 392 questions detect only effects ≥ ~4 pp at 80% power. That is adequate for the paper-implied effect (Fig. 4b/Table 7 suggest several pp if the mechanism transfers) and inadequate for a 1–2 pp effect; a null at 392 must be reported as "no effect ≥ 4 pp", not "no effect". If budget allows, run fill on all 1,000 (MDE 2.6 pp). For `visref_noansinj` the informative sample is the k>1 subset (~115 of 392), where the MDE is ~8–10 pp; it can only detect a gross effect, and the deficit it would need to explain is −3.0 pp on continued questions (B:16: 41.7 vs 38.7, n=230, i.e. ~7 questions). Replaying noansinj on all k>1 questions of the 1,000 (~300) still gives MDE ~6 pp. Pre-specify that noansinj is a mechanism check, not a hypothesis test.

## 5. Ordering, priority, missing pieces

Judgment, justified by the brief's numbers:
- The gap to the paper is not located at k=1: on both-k=1 questions ST/TSR/VisRef = 68.6/68.2/68.6 (B:16), and the answer-injection effect at k=1 is ≈ 0 (§2.2). It is located on continued questions (33.5/41.7/38.7, n=230, B:16), where VisRef trails TSR by ~7 questions. The leading explanations consistent with the brief are: (H1) effective injection ≈ 6% of |V| (B:11) vs the paper's 30%, below the lowest tested value (P:794-795); (H2) format: bare embeddings directly followed by `</think>` (B:20-21), visible as 45–50% `</think>` re-emission (B:17-18); (H3) kernel representation (input embeddings vs hidden states, §1(c)). `visref_fill` addresses H1 directly; noansinj addresses only the answer-side part of H2 and mostly on a small subset.
- Also note for the report (not a variant): the reproduction's baselines are ~12 pp above the paper's (ST 58.0 vs 46.1, TSR 60.2 vs 48.5; B:4, B:15), so the paper's margins come from a different operating point and need not transfer.

Recommended order:
1. **Free analyses from existing logs (no GPU):** (a) VisRef−ST on all VisRef-k=1 questions = exact answer-injection effect at k=1 (§2.2); (b) k distribution and H_k distribution for TSR vs VisRef — if VisRef stops earlier because the V block lowers probe entropy, that is a mechanism noansinj cannot see; (c) accuracy conditional on `</think>` re-emission and on `final_af` within VisRef.
2. **`visref_fill`** on the 392 (all 1,000 if affordable), with the §3 checks, plus **dual-probe logging**: at every step compute a second probe on `… z_k </think>` (without V_k) with its own generator and log H_k^{noV} alongside H_k; decisions keep using H_k. This measures how much the bare V block shifts the model's answer distribution and its confidence, for every step and every question, with zero effect on the run — it answers the question the withdrawn `visref_nolast` was after, without the confound.
3. **`visref_noansinj` as replay** of stored VisRef trajectories, analysed on the k>1 subset; k=1 rows serve only as an fp16-noise check (they must reproduce ST's answers).
4. Conditional: hidden-state Z (after the offline rank check) or random-fill control (if fill shows a gain), one of them, not both.

Not recommended: `<img>…</img>` wrapping (B:34-35) stays out as agreed; any variant that moves the probe before V_k (changes k; deviates from P:520-523).

## 6. Verdicts and reporting language

**`visref_fill` — KEEP with changes.** Changes: (1) assert head identity with `visref_raw_rank`; (2) fp32/fp64 ridge updates; (3) log n_inject/m, stop_reason, k, re-emission and final_af rates; (4) measure tail-vs-relevance correlation instead of asserting it; (5) pre-register the MDE (4.2 pp at n=392). Reporting: "A greedy MAP on L_k + εI (ε = 1e-6·mean diag) that runs to exactly m picks, as the text of §4.1.1 prescribes (|V_k| = m), differs from VisRef-rank by [Δ ± CI]. The first rank(L_k) picks are identical to VisRef-rank; the remaining picks are relevance-like and carry no diversity information because the kernel is rank-deficient. This estimates the effect of the injection amount, not of the paper's intended 'm diverse tokens'. The paper does not specify behaviour when rank(L_k) < m." A null: "no effect of injection amount ≥ 4 pp detected"; a gain: "consistent with the paper's m ablation (P:794-795); does not identify whether relevance-tail or amount drives it".

**`visref_noansinj` — KEEP with changes (as a diagnostic), otherwise DROP.** Changes: (1) implement as replay of stored VisRef trajectories (store step token ids); (2) primary analysis on VisRef-k>1 questions; report that k=1 rows reproduce ST's answers (fp16 check); (3) pre-register as mechanism check with secondary outcomes (re-emission, final_af, final_len); (4) do not present it as a VisRef implementation. Reporting: "Deliberate deviation from Algorithm 1 line 17 (P:529) and §4.1.2 (P:728): the final answer is sampled without the last V_k block, with steps and stop decisions identical to VisRef. On VisRef-k=1 questions this equals ST's answer by construction; the brief's data already show no net effect there (68.6 vs 68.6, n=529). On continued questions (n≈[..]) the difference is [Δ ± CI]; the design can detect only effects ≥ ~8–10 pp on that subset." Avoid: "VisRef's injection hurts/does not hurt" (it concerns one block in one position), and "VisRef works" (the gap to TSR on continued questions is not addressed by this variant).

**Dual-probe logging — ADD.** Reporting: "H_k with and without V_k differ by [..]; fraction of steps where the stop decision would flip: [..]". This is observational; it does not alter the method.

**Hidden-state Z — PROPOSE, conditional.** Reporting if run: "alternative reading of 'text token embeddings in reasoning state z_k' (P:450-451); changes representation and achievable |V_k| simultaneously."
