# Reproducibility report: VisRef, Appendix Table 6 row "InternVL-1B" on MathVista testmini

*Unofficial reproduction, MLRC / ReScience style. The main run and four follow-up experiments are finished; the decision rules of the follow-ups were fixed before their results. One more set of experiments (A5, three runs) and a confirmatory run are planned (Section 5.4).*

**Paper.** Ghosal, Kim, Li, Chaudhry, Xu, Zhang, Zablocki, Xing, Zhang. *VisRef: Visual Refocusing while Thinking Improves Test-Time Scaling in Multi-Modal Large Reasoning Models.* arXiv 2603.00207, CVPR 2026. No official code was released (the CVPR 2026 supplementary material was checked).

## Summary

We reimplemented the three methods of Table 6 (standard thinking ST, textual self-reflection TSR, and VisRef) for InternVL3.5-1B and evaluated each once on all 1,000 MathVista testmini questions, with the same questions, the same seeds and a shared first thinking step. Accuracies were ST 57.8, TSR 60.1 and VisRef 60.0 % (paper: 46.1, 48.5, 52.0). The TSR−ST difference (+2.3 pp, 95% CI +0.6 to +4.0) is consistent with the paper's +2.4; this is not proof of equivalence. VisRef−ST was +2.2 pp (CI +0.3 to +4.1): the CI excludes 0 but also the paper's +5.9, and the difference is not significant after the Holm correction fixed in advance (p = 0.067). VisRef−TSR was −0.1 pp (CI −2.0 to +1.7), which excludes the paper's +3.5: we did not confirm an additional gain over TSR in our InternVL3.5-1B implementation. Four follow-ups with decision rules fixed before their results (filling the injection up to m tokens, which turned out to add tokens in an order nearly unrelated to relevance; re-testing step-1 re-injection in a four-condition design; using the model's last-layer vectors for the thinking text, with which the DPP injected m tokens; and wrapping m tokens chosen with last-layer vectors in InternVL's image tags after `</think>`, which gave −1.0 pp against TSR with the same answer prompt, 95% CI −3.1 to +1.0) did not change this picture. We look for the cause of the missing gain in our implementation first: it has a confirmed procedural difference (about one fifth of the paper's token budget m is injected; injecting m tokens with last-layer vectors brought no detectable gain either), many settings we could not find in the paper were chosen by us, our absolute accuracies are 8–12 pp above the paper's, and we cover one model size, one benchmark and one seed.

## 1. Scope of reproducibility

| # | Claim (paper Table 6, 1B row: ST 46.1 / TSR 48.5 / VisRef 52.0) | Paper delta | Role here |
|---|---|---|---|
| C1 | TSR > ST | +2.4 pp | supplementary |
| C2 | VisRef > ST | +5.9 pp | main comparison 1 |
| C3 | VisRef > TSR | +3.5 pp | main comparison 2 |

The paper's headline claim is that VisRef improves over both ST and TSR, so C2 and C3 are the main comparisons. We target the paired differences, not the absolute accuracies: the paper's ST (46.1) is far below the official InternVL3.5-1B MathVista (mini) score of 59.3 (InternVL3.5 report, Table 3; the report says its Section 3 results were obtained without test-time scaling, which it defines to include the thinking mode, so this is most likely a non-thinking score; which evaluation setting produced it is not confirmed), and the paper's 8B ST (68.1) is also more than 10 pp below the official 78.4. Table 6 labels the row "InternVL-1B". We use InternVL3.5-1B: the only InternVL generation the paper names is InternVL3.5, it runs its models in thinking mode, it compares the 1B, 2B and 8B models "under the same decoding setup and token budget", and its 8B row equals the InternVL3.5-8B row of Table 1.

### 1.1 Experiments at a glance

The work has two kinds of experiment. **Result experiments** (A1–A6) test the paper's claim or look for the reason it was not reproduced; their decision rules were fixed before their results (for A1 with the limitation described in Section 2.4), and the conclusions come from them. **Setting checks** (B1–B7, [Section 2.6](#26-setting-checks-how-the-unstated-settings-were-chosen)) are small runs that chose settings the paper does not state, such as how to compute the stopping signal or how to re-insert the image. They gave no conclusions, but the settings they chose shape every result. The follow-ups A2–A6 are exploratory: each has its own rule fixed before its results, but there is no correction across them; a single confirmatory run with the setting closest to the paper's is planned next ([Section 5.4](#54-next-experiments-planned)).

<!-- check:glance -->
| | Question | Questions | Result | Paper | Main limitation | Section |
|---|---|---|---|---|---|---|
| A1 | Does VisRef (rank-limited selection) beat TSR and ST? | 1,000 | VisRef − TSR: −0.1 pp | +3.5 pp | only about one fifth of the paper's m image tokens were re-inserted | [3.1](#31-main-result-main-scoring-n--1000) |
| A2 | Does filling the re-inserted image tokens up to m help? | 976 | VisRef-fill − TSR: 0.0 pp | – | the added tokens were nearly unrelated to the thinking | [4.2](#42-visref-fill-filling-the-injection-up-to-m-tokens-rule-fixed-before-the-results) |
| A3 | Does re-inserting the image after step 1 help by itself? | 309 | with − without the image: +1.40 pp (answer at once), +0.97 pp (answer after more thinking) | – | only the questions where a method continued after step 1 | [4.3](#43-step1_branch-re-injection-at-step-1-rule-fixed-before-the-results) |
| A4 | Does choosing image tokens with the model's last-layer vectors help? | 839 | VisRef-hs − TSR: +0.24 pp | – | the first step was generated again (193 of 839 match the main run) | [4.5](#45-visref-hs-last-layer-z-rule-fixed-before-the-results) |
| A5 | Does thinking more after the re-inserted image help? | planned | not run yet | – | changes the paper's stopping behaviour | [5.4](#54-next-experiments-planned) |
| A6 | Does a tagged image block after `</think>` (tokens chosen with last-layer vectors) beat TSR with the same answer prompt? | 1,000 | VisRef-imgtag − TSR with answer prompt: −1.0 pp | +3.5 pp | changes four unstated settings at once; run on a different GPU, so compared only within the run | [4.6](#46-visref-imgtag-tagged-image-block-after-think-rule-fixed-before-the-results) |
<!-- /check -->

| | Setting to decide | What we chose | Evidence | Where the setting comes from |
|---|---|---|---|---|
| B1 | When to stop thinking | the entropy of one short answer in `\boxed{…}` | 4 checks, 10–60 questions each | paper: the stopping rule<br>cited work (Ghosal et al.): several answers<br>ours: one answer |
| B2 | How to re-insert the image tokens | without InternVL's image tags, right before `</think>` | 2 checks, 10 and 30 questions | paper: the order of the steps<br>official (InternVL): the image tags<br>ours: leaving them out |
| B3 | Thinking length limit, and how to get an answer after a cut | 1,024 tokens per step, 10,240 in total; a prompt for the answer | 5 checks, 18–150 questions | official (InternVL): 1,024 is its example value<br>cited work (s1): the answer prompt<br>related work (DEER, not cited): the last-resort answer |
| B4 | How many image tokens to pick | as many as the selection can pick, usually fewer than m | 74 questions and a calculation check | paper: m tokens<br>cited code (Chen et al. 2018): stops when no more can be picked<br>→ **conflict; we follow the code** |
| B5 | How to extract and score answers | rules first; an LLM only where the rules fail | a pre-check on 49 rows | official (VLMEvalKit): which text is scored<br>official (MathVista): the extraction prompt<br>ours: which LLM, the grounding check |
| B6 | Do the same inputs give the same outputs? | compare on the same questions from the same first step | 730–1,000 questions | ours |
| B7 | Where to put a tagged image block, and how to start the answer (exploratory, after the main run) | after `</think>` when answering, inside the thinking when continuing; every answer after `\nFinal Answer:` | 2 trial runs, 10 questions each | official (InternVL): the image tags<br>cited code (s1): the answer prompt<br>ours: the position, and using the prompt for every answer |

The raw outputs of every check are in `repro/kaggle/runs/`, except for the 10-answer check of B1 (8B model, earlier code) and the official-setting run of B3.

**The setting checks used evaluation questions.** They ran on part of a 100-question development split drawn from MathVista testmini and on the first questions of the main run's order. All of these questions are also among the 1,000 evaluated questions (`docs/EXPERIMENT_FLOW.md`, the opening entry and the entry of 2026-10-07 on the lean code), so the settings may be tuned to them to some degree. Most settings were chosen from stopping rates and broken answer formats, and several checks stated in advance that accuracy would not be used for the decision. The questions used by the setting-check runs (the run folders named in Section 2.6; 222 distinct questions, counting all 30 planned questions of `smoke2`, 6 of which have no recorded answer) are listed in `results/setting_check_questions.csv`; the earlier development split of the old code is not part of this repository.

### 1.2 Terms

| Term | Meaning |
|---|---|
| ST (standard thinking) | The model thinks once and answers. The baseline. |
| TSR (textual self-reflection) | When a thinking step ends, "Wait, think more" is appended and the model thinks again. The image is not shown again. |
| VisRef (code `visref_raw_rank`) | Like TSR, but just before each continuation a selected part of the image is inserted again. The paper's method, here with the selection stopping where the kernel rank ends ("rank-limited": usually fewer than m tokens, Section 4.1). |
| VisRef-fill (`visref_fill`) | VisRef with the selection filled up to m tokens (Section 4.2). |
| VisRef-hs (`visref_hs`) | VisRef with the model's last-layer vectors for the thinking text (Section 4.5). |
| VisRef-imgtag (`visref_imgtag`) | VisRef with the selected tokens wrapped in image tags, placed after `</think>` when answering and inside the thinking when continuing, and last-layer vectors for both text and image (B7, A6). |
| TSR with answer prompt (`tsr_answer_prompt`) | TSR in which every answer starts after `\nFinal Answer:`; the comparison partner of VisRef-imgtag. |
| answer forcing | When an answer reaches the length limit, `\n**Final Answer**\n\boxed` is appended and up to 100 more tokens are allowed. |
| thinking step | One stretch of thinking, until the model closes it (`</think>`) or hits the length limit. TSR and VisRef can run several steps. |
| token | The unit the model reads and writes, roughly a piece of a word. |
| image tokens | The pieces the model cuts an image into (visual tokens); 1,961.5 per question on average here. |
| m | The number of image tokens the paper re-inserts at each step: 30% of all image tokens of the question. |
| DPP selection | The calculation that picks image tokens that are relevant to the current thinking and do not repeat each other; we follow the code the paper cites. |
| stopping entropy H | How uncertain the answer would be if the model answered now. Thinking stops when H is below 0.25. |
| trial answer | A short answer the model writes after `</think>` and `\n\n\boxed` only to compute the stopping entropy; it is then discarded and is not the final answer. |
| pp | Percentage points: the difference between two accuracies (60.0% − 57.8% = 2.2 pp). |
| 95% CI | The range in which the difference would fall in most repetitions of the experiment. If it contains 0, we cannot say there is a difference. |
| paired comparison | Two methods compared on the same questions, which removes differences in question difficulty. |
| decision rule fixed before the results | The rule for reading a result was written down before the result was seen. |
| raw outputs | The text the model generated and the scored rows, from which anyone can recompute the numbers. |

## 2. Methodology

### 2.1 Setup

![ST, TSR and VisRef as run in this reproduction](figures/method_overview.svg)

*Figure 1. The loop of Algorithm 1 as implemented here. Grey boxes are steps the paper specifies, the blue box is the step only VisRef has, and orange notes are settings we could not find in the paper (our choices, listed in the table below).*

| Item | Value | Where it comes from |
|---|---|---|
| Model, data | `OpenGVLab/InternVL3_5-1B` @ `2f71cf52…`; `AI4Math/MathVista` testmini (1,000 questions) @ `2b6ad694…` | The paper (see Section 1) |
| Question order, seeds | Order shuffled with seed 0; generation seed 42, derived per question and shared by the three methods; first thinking step shared (copied) | Our own choice |
| Methods | ST; TSR (cue "Wait, think more"); VisRef (greedy MAP DPP over visual tokens) | The paper; the selection follows the DPP code of Chen et al. 2018, which the paper cites |
| VisRef values | δ = 0.25, m = ⌊0.3·\|V\|⌋, K_max = 10 | The paper |
| DPP selection size | Greedy stops when the residual gain falls below 1e-9 × max L_ii (the cited Chen code stops at an absolute threshold, 1e-10; the relative threshold is our choice), so most steps inject fewer than m tokens. The paper states \|V_k\| = m: **confirmed procedural difference** | The code the paper cites (Chen et al. 2018); this conflicts with the paper's text |
| Decoding | R1 thinking system prompt, T 0.6, top-k 50, top-p 0.95, fp16, Kaggle 2×T4 | Official InternVL3.5 material (the official model page on Hugging Face and VLMEvalKit). Hardware: the paper used 4×A10G; T4 has no bf16, so we use fp16 |
| Software | transformers 4.55.0 (as in the paper); torch from the Kaggle image (recorded per session in `env.json`; paper: 2.7.0) | transformers version: the paper. torch version: our environment (Kaggle image) |
| Token limits | 1,024 per thinking step, 10,240 total thinking, 1,024 answer, 100 for answer forcing | Our own choice (1,024 is the example value on the official model page) |
| Truncated thinking, capped answers | After truncated thinking: `</think>` + `\nFinal Answer:` (s1-style budget forcing). Answer reaching the cap without EOS: `\n**Final Answer**\n\boxed` + ≤100 tokens (answer forcing as in DEER) | Truncated thinking: the s1 method, which the paper cites; applying it here is our interpretation. Capped answers: DEER, which the paper does not cite (our own choice) |
| Stopping statistic H | Sum of token entropies of one answer sampled after `</think>` + `\n\n\boxed` (T = 1, ≤16 tokens) | Definition (entropy of the answer distribution): the paper. How it is estimated: our own choice |
| Reasoning-token representation Z | Input embeddings of the thinking-step tokens | Our own choice (the paper names no layer); a last-layer variant is in Section 4.5 |
| Injection format | Selected visual embeddings appended without `<img>` tags directly before `</think>`; order z_k → Ṽ_k → H_k → "Wait, think more"; Ṽ_k ⊆ V | Order and subset: the paper (Algorithm 1, Figure 2). Tags, separators, `</think>` handling, position ids and answer prompt were not found in the paper, so they are our own choice, including the order `[Ṽ_k] </think>` |
| Visual tokens | Dynamic tiling, max_num = 12 plus thumbnail tile; mean \|V\| = 1,961.5 | Official InternVL3.5 material (the official model page); see Section 4.4 |

### 2.2 Scoring

The text after the last `</think>` is scored (the same split rule as VLMEvalKit `SPLIT_THINK`; whether the InternVL3.5 evaluation enabled it is unconfirmed). Answers are extracted by rules (boxed, then "answer is", then last line) and compared with the MathVista normalisation. Rows on which the rules fail (36 of 3,000) go to an LLM fallback (official MathVista extraction prompt, gemini-3.5-flash-lite, fixed before the final extraction). An LLM answer is accepted only if it is grounded in the response text. Empty responses (13) are never sent and count as wrong. Scorer v4 (2026-10-11, after the results) counts an empty multiple-choice extraction as wrong, like the official `--ignore_empty_extractions`; it changes 0 rows of the main scoring. This main scoring (rules plus LLM fallback) was fixed in advance; the other scoring variants in Section 3.2 are sensitivity checks.

### 2.3 Statistics

The unit is the question, compared pairwise. We report paired bootstrap 95% CIs of accuracy differences (2,000 resamples, seed 0), exact McNemar tests, and a Holm correction over the two main comparisons (VisRef−ST, VisRef−TSR). TSR−ST is supplementary and outside the Holm family. Classification against the paper's delta: **"consistent with the paper's value"** means the CI excludes 0 and contains the paper's delta (not proof of equivalence); **"effect size not reproduced"** means the CI excludes the paper's delta. This is not a test of the difference between our result and the paper's: the paper gives no per-question outputs.

### 2.4 When the decision rules were fixed

The main comparisons, the tests (McNemar + Holm) and the scoring variants of Section 3.2 were fixed on 2026-10-10 before the final 1,000-question analysis. The choice of main comparisons and tests followed the paper's claim but was made after we looked at partial results for the first 392 questions (sessions 1–4; VisRef−TSR was about +1.0 pp then), so the decision rules of the main run were not fixed before all results in the strict sense. The **CI classification rule was fixed after we had seen the partial result on 759 questions** (sessions 1–7), and the final 1,000 include those 759, so the classification is not confirmatory in the strict sense. For the three follow-ups (Sections 4.2, 4.3 and 4.5) the decision rules were written down before their results. Scorer v4 was a change after results. Every other analysis is labelled as decided after the results and has no correction for multiple tests.

### 2.5 Code versions and compute

Three code versions produced the 1,000 questions: `79eef648ebd1` (session 1, 74 questions), `f2422d0e72ff` (sessions 2–4, 318) and `d0095f79e691` (sessions 5–9, 608). The follow-ups ran later versions (VisRef-fill `39b3fb1b4121`, VisRef-hs `f1376774850d`); changing `run.py`'s default token caps on 2026-10-13 (every run passed the caps explicitly) made the version `cf46b2e6222e`; the planned thinking-continuation code and its stored-first-step fix (2026-10-13) made it `86a5fc0f3cd9`, and the exploratory VisRef-imgtag method (first named `visref_bundle`) with the shared random state for the second trial answer that is only logged (2026-10-15) made `aa11cdaa52a9`, the version of the first image-tag trial; the fixes after the trials made the current version `6e218887d2b5`, which the A6 run uses (the versions in between are listed in the experiment log). The first change removed unused branches (outputs identical on 810 of 810 compared outputs); the second changed only the scorer. All outputs were re-scored with the final scorer and merged. Compute: Table 2.5 lists the GPU and the generation time of every experiment (`results/compute.csv`, built from the run records by `scripts/make_compute.py`).

**Table 2.5. GPU and generation time per experiment.** Wall time adds the slowest GPU of each session; GPU-hours add all GPUs. Times cover generation only (not package install or model loading), and the GPUs differ between the Kaggle runs and A6, so times are not comparable across them.

| | What ran | GPU | Wall time (h) | GPU-hours | Rows |
|---|---|---|---|---|---|
| A1 | main run: ST, TSR, VisRef (9 sessions) | 2 × T4 | 14.9 | 28.0 | 3074 |
| A2 | VisRef-fill | 2 × T4 | 11.3 | 22.3 | 994 |
| A3 | step1_branch (pilot and main) | 2 × T4 | 0.9 | 1.7 | 309 |
| A4 | VisRef-hs | 2 × T4 | 7.8 | 15.4 | 839 |
| A6 | VisRef-imgtag, TSR with answer prompt, TSR (check run and main run) | 1 × RTX A5000 (rented) | 22.8 | 22.8 | 3018 |
| B1-B6 | setting checks | 2 × T4 | 7.2 | 13.6 | 764 |
| B7 | two 10-question trials of the image-tag variant | 2 × T4 | 0.3 | 0.4 | 40 |

### 2.6 Setting checks: how the unstated settings were chosen

Each table lists the checks in time order; the name in brackets is the run folder in `repro/kaggle/runs/`. "Development questions" are questions of the 100-question development split (Section 1.1).

**B1. When to stop thinking.** The paper stops thinking when the entropy of the answer distribution falls below 0.25, but does not say how this entropy is computed. We tried four ways.

| Check | How H was computed | What happened | Decision |
|---|---|---|---|
| first small check (`smoke`, 10 development questions) | mean entropy over the answer part of the response | almost always 0 (at most 0.003), so every question stopped after step 1 and TSR became identical to ST | compute over the whole response |
| second small check (`smoke2`, 30 development questions) | the whole response | long responses give large values: summed as in the paper they were below 0.25 in only 0.8% of trial answers | let the model write one short answer after "Answer:" and sum its token entropies |
| third small check (`smoke3`, 20 development questions) | one short answer after "Answer:" | TSR stopped by entropy in 0 of 18 questions: the answer content was certain (about 0.01 per token) but the leading space (about 0.47) and the line end (about 0.84) kept H high | read the answer inside `\boxed{…}` (at most 16 tokens), with the closing-brace tokens counted as one outcome |
| check before the main run (`main_s0`, 59 questions) | the answer inside `\boxed{…}` | stopped after step 1 by entropy: TSR 50 of 59, VisRef 51 of 59; without merging the closing-brace tokens only 41 of 70 trial answers were below 0.25 (55 of 70 with it) | used in the main run |
| ten answers per step (8B model, 3 development questions, earlier code) | ten complete answers per step | a TSR run on 3 questions took 4,703.9 GPU-seconds | back to one answer (from the paper's latency table we judged that the authors did not generate several long answers per step; our judgment) |

- *Where the setting comes from.* The stopping rule (entropy of the answer distribution below 0.25) is from the paper. Ghosal et al., whom the paper cites for this criterion, sample several answers. One answer and the `\boxed{…}` format are our choice.
- *Also recorded.* In the VisRef-fill run (Section 4.2), H was measured both with and without the re-inserted image tokens: the median difference was 0.0006, and the stop-or-continue decision differed at 151 of 2,231 steps (6.8%). The two measurements used separate random streams for the sampled trial answer, so part of this difference can come from sampling rather than from the re-inserted tokens (later runs replay the same random state for both).
- *Limits.* The choice rests on 10 to 60 questions each. One sampled answer gives the right entropy on average but is noisy. In this setting TSR and VisRef rarely continued after a step the model had closed itself (7 of 359 and 35 of 398 trial answers); almost all continuations came after a 1,024-token cut.
- *Sources.* `docs/EXPERIMENT_FLOW.md`, entries of 2026-10-07 (the first, second and third small checks, `main_s0`); Sections 4.1 (items 1 and 5) and 4.2; the ten-answer check (earlier code, outputs not published): [log at commit 4776f6a, lines 642–644](https://github.com/AhnJG/visref-repro/blob/4776f6ab2607957799bef8db70241cc64313feae/docs/EXPERIMENT_FLOW.md?plain=1#L642-L644).

**B2. How to re-insert the image tokens.** InternVL reads an image wrapped in `<img>` … `</img>` tags. The paper does not say how the selected image tokens are inserted during thinking.

| Check | What happened | Decision |
|---|---|---|
| first small check (`smoke`, 10 development questions, with tags) | 7 to 8 of 9 VisRef responses broke into "draft answer + a second `</think>` + answer" (TSR 1 of 10) | also try inserting without tags |
| second small check (`smoke2`, 30 development questions) | steps followed by a second `</think>`: with tags 31 of 63, with tags and the ridge variant 25 of 67, without tags 10 of 65, TSR 4 of 70 | insert without tags, right before `</think>` |
| review of the insertion code (separate AI session) | no implementation error; the order "image tokens, then `</think>`" may cause the broken formats | kept (we do not change settings after seeing results) |

- *Where the setting comes from.* The order thinking → selected image tokens → stopping check → "Wait, think more", and the use of a subset of the original image tokens, are from the paper (Algorithm 1, Fig. 2). The `<img>` tags are InternVL's official format. Leaving them out and the position before `</think>` are our choice.
- *Limits.* Tags were compared only in the small checks, not in the main run. Separators and position ids were not tried; another insertion point was tried later in a small exploratory trial (B7). Without tags the broken format remains (A1: a second `</think>` in 352 of 713 VisRef answers).
- *Sources.* `docs/EXPERIMENT_FLOW.md`, entries of 2026-10-07 (the first and second small checks); Section 4.1 (items 3 and 6).

**B3. Thinking length limit, and how to get an answer after a cut.**

| Check | Setting | What happened | Decision |
|---|---|---|---|
| long limit (`main_s0`, `full_s1`, about 60 questions each) | 4,096 tokens per step | 48–57% of ST first steps were cut, about half of those in repetition loops; 58 questions × 3 methods took 2 hours, too slow for 1,000 questions on free GPUs | 1,024 tokens per step, 10,240 in total |
| short limit (`b_test`, 18 questions, stopped after about 8 minutes) | after a cut, `Final Answer:` asks for the answer | 12 of 26 prompted answers ran to the 1,024-token answer limit again, as often as without the prompt (14 of 30 in `main_s0`) | if the answer still does not finish, append `**Final Answer**\boxed` and allow 100 more tokens |
| first main session (`b_main_s1`, 74 questions) | as above | no question was left without an answer | used in the main run |
| answer prompt removed (`diag_st_plain`, 150 questions, rule fixed before the results) | no `Final Answer:` after a cut | −2.0 pp, 15 against 18 discordant questions | kept; 150 questions cannot detect a change of a few pp |
| official setting (`st_official`, target 100 questions, free Colab, rule fixed before the results) | InternVL's official thinking evaluation: up to 65,536 new tokens, no answer prompt; ST with practically unlimited thinking | long questions took hours each; the run stopped after 33 questions | not analysed |

- *Where the setting comes from.* The paper gives no length limits. 1,024 is the example value of the official InternVL model page. Appending `Final Answer:` after a cut is the budget forcing of s1 (Muennighoff et al. 2025), which the paper cites; applying it here is our interpretation. The last-resort `**Final Answer**\boxed` comes from the public code of DEER (Yang et al. 2025), which the paper does not cite. The official-setting run followed VLMEvalKit's InternVL3.5 configuration.
- *Limits.* The 1,024-token limit shortens ST's thinking (Section 3.1, limit 2). The main run was not repeated with a 4,096-token limit. The outputs of the official-setting run are not in this repository.
- *Sources.* `docs/EXPERIMENT_FLOW.md`, entries of 2026-10-07 (main_s0, full_s1, b_test, b_main_s1) and 2026-10-10 (st_official, the 4,096-cap check); Sections 2.1, 3.3 and 4.1 (item 6).

**B4. How many image tokens to pick.** The paper picks exactly m tokens at each step.

| Check | What happened | Decision |
|---|---|---|
| the cited selection code (Chen et al. 2018) | the greedy selection can pick only as many tokens as the kernel rank allows, which is bounded by the number of distinct tokens in the thinking step, and then stops; short steps give far fewer than m | stop where the code stops; recorded as a confirmed difference from the paper |
| first main session (`b_main_s1`, 74 questions) | ran "stop where the code stops" next to "fill up to m by relevance" | after this session, before the next one, the fill variant was dropped |
| deeper-layer vectors checked offline (`hidden_rank_check`, 6 cut steps) | with hidden-state vectors, too, only about 50–110 picks add clearly new information; none of the 6 steps reached m | the deeper-layer variant was postponed (run later as Section 4.5) |

- *Where the setting comes from.* "m tokens at every step" is from the paper. The selection follows the code the paper cites (Chen et al. 2018), which stops when no further token can be picked. The two conflict; we follow the code and state the difference. The stopping threshold (relative to the largest diagonal) is our choice.
- *Confirmed in the main run.* Median injected/m ratio 0.199; 1,895 of 2,261 steps injected fewer than m tokens.
- *Limits.* How the authors handled this is unknown (Section 5.3, question 6). Dropping the fill variant was decided after 74 questions.
- *Sources.* Section 2.1 ("DPP selection size"), Section 4.1 (item 4), Section 4.2; `docs/EXPERIMENT_FLOW.md`, entries of 2026-10-07 (selection rule, b_main_s1) and 2026-10-08 (hidden_rank_check).

**B5. How to extract and score answers.** The text after the last `</think>` is scored. Answers are taken by rules first (`\boxed{…}`, "answer is", the last line); only where the rules fail does an LLM extract the answer with the official MathVista extraction prompt.

| Check | What happened | Decision |
|---|---|---|
| LLM extraction pre-check (49 rows, gemini-3.5-flash-lite) | both rows the LLM "fixed" were wrong: it invented an answer for an empty response, and picked a different number from the text | empty responses are not sent, and an LLM answer counts only if it appears in the response; re-run: 0 wrongly fixed |
| final extraction of the main run (1,000 questions) | of 36 rule-failed rows, 30 extractions used and 6 rejected; 1 label changed; 20 control rows all agreed with the rules | used in the main scoring |
| scoring variants (five, Section 3.2) | the classification is the same under every variant; VisRef − TSR lies between −0.9 and +1.0 pp | answer handling alone does not explain the gap to the paper |

- *Where the setting comes from.* Scoring the text after the last `</think>` is VLMEvalKit's split rule. The extraction prompt is MathVista's official one. Rules first, the LLM model and the grounding check are our choice. Scorer v4, changed after the results (an empty multiple-choice extraction counts as wrong), behaves like the official MathVista option `--ignore_empty_extractions`.
- *Limits.* The official extractor is GPT; ours differs. The VisRef-fill and VisRef-hs runs used a different LLM (Claude Haiku). The grounding check stops invented answers but not a wrong number picked from the text.
- *Sources.* `docs/EXPERIMENT_FLOW.md`, entry of 2026-10-10 (LLM extraction pre-check); Sections 2.2, 3.2 and 4.4.

**B6. Do the same inputs give the same outputs?**

| Check | What happened |
|---|---|
| TSR and ST from the same context and seed (730 questions where TSR stopped after step 1) | identical text in 491; extracted answers differ in 16 (2.2%), from the order of fp16 arithmetic when questions are batched together |
| first steps regenerated in follow-up runs | first-step text identical to the main run: VisRef-fill 242 of 994, VisRef-hs 193 of 839 |
| before and after removing unused code | 810 of 810 compared outputs identical |

- *Decision.* The three methods share the first thinking step, so that they are compared on the same questions from the same start; the planned experiments start from the stored first step.
- *Limits.* The paper used A10G GPUs; we used T4. The effect of this difference was not tested.
- *Sources.* Sections 2.5, 4.1 (item 2) and 6.

**B7. Where to put a tagged image block, and how to start the answer** (exploratory, 2026-10-15; VisRef-imgtag). After the main run, the image tags were tried again at another position: the selected tokens wrapped in `<img>` … `</img>` with newlines, placed after `</think>` when the model answers and inside the thinking block when it continues; the selection uses last-layer vectors for both the thinking step and the image tokens. Both trials ran the main VisRef and this variant from the stored first step on the same 10 questions (5 whose first step the model closed itself, 5 cut at 1,024 tokens). Only the shape of the outputs was examined, not their correctness.

| Insertion | Example output (question whose first step the model closed itself) | Count |
|---|---|---|
| main run: untagged, before `</think>` (B2) | `The correct answer is A) Tilapia. … **Answer:** A.` → `</think>` → `The food web shows that tilapia directly relies on algae …` (the answer is written a second time) | second `</think>` in the answer: ST 20, VisRef 156 of 329 questions whose first step was closed and where VisRef stopped after it |
| second small check (`smoke2`): tagged, before `</think>` (B2) | `… was \boxed{7}.` → `</think>` → `To determine the rate of change between 2013 and 2014, …` | 31 of 63 steps |
| first trial: tagged, after `</think>`, free answer | `Question: Which organism with be most affected if algae was eliminated? …` (the question is written again); one answer restarted with `<think>`, one ended empty | 5 of 5 |
| second trial: as above, answer after `\nFinal Answer:` | `Final Answer: A. Tilapia`, `Final Answer: 400`, `Final Answer: 0` | 0 of 5 with any of these problems |

- *Reading.* In InternVL's prompt the question always follows `</img>\n`, so after a tagged block the model continues as if a new question started. Inside the thinking block, with or without tags, it writes an answer, closes the thinking again and repeats the answer. The answer prompt removed both patterns in this small trial. With the last-layer selection the injection reached about m tokens (median injected/m 0.97 in these trials, against 0.19 for the main VisRef on the same questions).
- *Limits.* Ten questions, two trials; no accuracy was computed. The answer prompt now precedes every answer of this variant, so a comparison needs TSR with answer prompt.
- *Sources.* `docs/EXPERIMENT_FLOW.md`, entries of 2026-10-15; `repro/diag/analyze_imgtag.py`.

## 3. Results

### 3.1 Main result (main scoring, n = 1,000)

> **Card A1.** *Question:* does VisRef beat TSR and ST? *Method:* 1,000 questions; the three methods share questions, seeds and the first thinking step; 1,024 tokens per step, 10,240 in total. *Result:* VisRef − TSR −0.1 pp (CI −2.0 to +1.7; paper +3.5). *Limits:* (1) about one fifth of the paper's m image tokens were re-inserted (median 129 of 537), so this is not "the paper's method, without effect"; (2) ST answered after a 1,024-token cut in 641 questions while TSR and VisRef could continue, so TSR − ST and VisRef − ST include extra thinking time, which the main comparison VisRef − TSR does not; (3) a second `</think>` in 352 of 713 VisRef answers that stopped after step 1 (ST 67); (4) the rules were fixed after we looked at partial results (Section 2.4). *Files:* `repro/kaggle/runs/b_main_s1`–`b_main_s9`, `repro/llm/final/merged.jsonl`, `repro/diag/final_judgment.py`.

![Paired accuracy differences with 95% CIs next to the paper's differences](figures/paired_differences.svg)

*Figure 2. Our paired differences (dot, 95% bootstrap CI) next to the paper's differences (diamond). Drawn by `scripts/make_figures.py` from `results/paired_differences.csv`.*

<!-- check:accuracy_report -->
| Method | Ours % | Paper % | Ours − paper |
|---|---|---|---|
| ST | 57.8 | 46.1 | +11.7 |
| TSR | 60.1 | 48.5 | +11.6 |
| VisRef | 60.0 | 52.0 | +8.0 |
<!-- /check -->

<!-- check:paired_report -->
| Comparison | Ours (pp) | 95% CI | Only first / only second correct | McNemar p | Holm p | Paper (pp) | Classification |
|---|---|---|---|---|---|---|---|
| VisRef − ST (main) | +2.2 | +0.3 to +4.1 | 60 / 38 | 0.033 | 0.067 | +5.9 | effect size not reproduced (same direction, different size) |
| VisRef − TSR (main) | −0.1 | −2.0 to +1.7 | 44 / 45 | 1.000 | 1.000 | +3.5 | effect size not reproduced |
| TSR − ST (suppl.) | +2.3 | +0.6 to +4.0 | 51 / 28 | 0.013 | n/a | +2.4 | consistent with the paper value |
<!-- /check -->

- VisRef−ST: the unadjusted CI excludes 0, but the difference is not significant after the Holm correction fixed in advance (p = 0.067).
- VisRef−TSR: the CI excludes +3.5 but not a small positive gain (up to +1.7 pp). We did not confirm an additional gain over TSR.
- TSR−ST is consistent with +2.4 (not proof of equivalence). Part of our TSR gain is a thinking-budget extension (Section 4.1), so this agreement says little about the mechanism.
- The paired CI half-width is about 1.7–1.9 pp at n = 1,000 (binomial SE of a single accuracy about 1.6 pp).

### 3.2 Sensitivity to answer handling

Scoring variants: *main scoring* as described in Section 2.2; *rules only*: no LLM fallback; *scored before answer forcing*: answers that did not finish and were completed by answer forcing are re-scored on the text before the forcing, with scorer v4 and v3; *unfinished answers counted wrong*: every row that needed answer forcing counts as wrong (a lower bound). Cells give difference in pp with CI.

| Scoring | ST % | TSR % | VisRef % | VisRef − ST | VisRef − TSR | TSR − ST |
|---|---|---|---|---|---|---|
| main scoring: rules + LLM (fixed in advance) | 57.8 | 60.1 | 60.0 | +2.2 (+0.3, +4.1) | −0.1 (−2.0, +1.7) | +2.3 (+0.6, +4.0) |
| rules only, v3 | 57.8 | 60.0 | 60.0 | +2.2 (+0.3, +4.1) | 0.0 (−1.9, +1.8) | +2.2 (+0.4, +3.9) |
| scored before answer forcing, v4 | 55.9 | 58.0 | 58.6 | +2.7 (+0.5, +5.0) | +0.6 (−1.7, +2.9) | +2.1 (+0.1, +4.1) |
| scored before answer forcing, v3 | 56.0 | 58.1 | 58.6 | +2.6 (+0.4, +4.9) | +0.5 (−1.8, +2.8) | +2.1 (+0.1, +4.1) |
| unfinished answers counted wrong | 44.4 | 48.3 | 47.4 | +3.0 (+0.5, +5.5) | −0.9 (−3.3, +1.5) | +3.9 (+2.0, +5.7) |

The CI-based classification is the same under every scoring variant. Significance is not: with unfinished answers counted wrong, VisRef−ST has McNemar 94 : 64 and Holm p = 0.042 (we recomputed this from the merged rows), versus 0.067 under the main scoring. LLM fallback: of 36 rule-failed rows, 30 grounded extractions were used and 6 rejected; 1 label changed from wrong to right (question 341, TSR) and none from right to wrong; 20 control rows on which the rules had succeeded agreed with the LLM extraction.

### 3.3 Run record

| | ST | TSR | VisRef |
|---|---|---|---|
| Runtime errors | 0 | 0 | 0 |
| First thinking step truncated at 1,024 tokens | 641 | 641 | 641 |
| Further step after: truncated / natural `</think>` / EOS | n/a | 1,103 / 7 / 5 | 1,214 / 32 / 15 |
| Final answer ended by EOS | 761 | 813 | 775 |
| Answer hit the cap, answer forcing used | 239 | 187 | 225 |
| Answer forcing itself ended at its 100-token cap | 164 | 118 | 143 |
| LLM extraction used | 11 | 8 | 11 |
| Empty response (counted wrong) | 7 | 4 | 2 |

## 4. Beyond the paper

Everything in this section was **chosen after seeing the results unless marked as having its rule fixed before the results**, uses the main scoring, and played no role in the conclusions of Section 3.

### 4.1 Cause analysis (after the results)

We could not identify a single cause. The items below are observations and candidates.

![How the main run ran: thinking steps per question, how each step ended, injected image tokens relative to m](figures/main_run_steps.svg)

*Figure 3. How the main run ran, counted from the recorded rows of all 1,000 questions per method: the number of thinking steps, how each step ended, and the image tokens VisRef injected relative to m (no accuracy). Drawn by `scripts/make_figures.py`.*

1. **Continuation happens almost only after cap truncation.** After a naturally closed step, the stopping statistic was ≥ δ in only 7 of 359 TSR and 35 of 398 VisRef trial answers (see the transition counts in Section 3.3). So part of TSR's gain over ST is probably a thinking-budget extension, and VisRef's injection mostly happens at a mid-sentence truncation point. Mean generated tokens per question: ST 1,114, TSR 2,205, VisRef 2,386.
2. **Questions where VisRef stops at step 1 (k = 1, 713 questions).** VisRef differs from ST only by injecting Ṽ_1 before the final answer: VisRef-only correct 15, ST-only correct 14. The extracted answer changed in 1 of 329 questions whose step closed naturally and in 33 of 366 truncated ones. Limitation: k = 1 is defined by the stopping decision after injection, and the TSR k = 1 baseline for random variation is a different item set, so this group alone cannot identify the cause. As a measure of random variation: on the 730 questions where TSR stopped at step 1 it answers from the same context and seed as ST, yet the final texts were identical in only 491 and the extracted answers in 714 (16, or 2.2%, differ), which we think is caused by fp16 batch-composition noise.
3. **Side effect of the injection.** In the same k = 1 group the final answer contains another `</think>` in 352 of 713 VisRef answers versus 67 of 713 for ST.
4. **Injection amount: confirmed procedural difference.** The median injected/m ratio is 0.199, and 1,895 of 2,261 steps inject fewer than m tokens (median m 537, median injected 129). The cause is the early stop at the kernel rank, which is bounded by the number of distinct reasoning-token embeddings. Section 4.2 tests filling up to m (the order of the added tokens turned out to be nearly unrelated to relevance); Section 4.5 tests last-layer Z, which removes this bound (no extra gain).
5. **The stopping statistic H is our estimator.** The paper defines H as the entropy of the answer distribution; the cited Ghosal et al. sample several answers. We use one answer sample, whose summed token entropy equals the answer-distribution entropy on average (chain rule) but is noisy. We started with one sample, tried 10 complete answers per step after reading Ghosal et al. (with the 8B model, a TSR run on 3 development questions took 4,703.9 GPU-seconds), and went back to one sample because we judged from the paper's latency table (Table 5: ST 7.1 s, TSR 7.7 s, VisRef 8.2 s per question, 8B model on one H100) that the authors did not generate several long, complete answers per step (`docs/EXPERIMENT_FLOW.md`, 2026-10-05 to 2026-10-07; our judgment, not stated in the paper: sampling several short answers in one batch would still fit the small latency gap); multi-sample estimation was not used in the final runs.
6. **Other checks.** Dropping the `Final Answer:` prompt after truncated thinking (150 questions, a diagnostic whose rule was fixed before the results) changed accuracy by −2.0 pp (15 vs 18 discordant, McNemar p = 0.73; 95% range about −9.5 to +5.5 pp, normal approximation), so the criterion fixed in advance (a drop of 10 pp or more) was not met; 150 questions cannot detect a change of a few pp. An independent review of the injection path by a separate AI-model session found no implementation error in the paths and samples it examined.

### 4.2 VisRef-fill: filling the injection up to m tokens (rule fixed before the results)

> **Card A2.** *Question:* does filling the selection up to m tokens help? *Result:* VisRef-fill − TSR 0.0 pp (976 questions). *Limits:* the added tokens were nearly unrelated to relevance, so this tests "more tokens of little relevance"; the first step was regenerated (242 of 994 identical); the rows the rules could not parse were extracted by a different LLM (Claude Haiku; no label changed). *Files:* `repro/kaggle/runs/b_main_fill`, `repro/diag/analyze_fill.py`.

*Design.* Identical to VisRef except that, after the greedy selection stops at the kernel rank, the same greedy continues on L + εI (ε = 1e-6 × mean diagonal) until m tokens are chosen (the leading part is checked to be identical to VisRef). *Sample.* All 1,000 questions were submitted; the session ended with 994 rows, and the analysis fixed in advance uses the complete prefix of the shuffled order (976 questions). *Rule.* A VisRef-fill − TSR CI lower bound above 0 would support "too few injected tokens reduced VisRef's gain".

| Method | ST | TSR | VisRef (main run) | VisRef-fill |
|---|---|---|---|---|
| Accuracy % (n = 976) | 57.79 | 60.04 | 59.94 | 60.04 |

| Comparison | Diff (pp) | 95% CI | Only first / only second | McNemar p |
|---|---|---|---|---|
| VisRef-fill − TSR | 0.00 | −2.15 to +2.36 | 70 / 70 | 1.000 |
| VisRef-fill − VisRef | +0.10 | −2.15 to +2.36 | 65 / 64 | 1.000 |
| VisRef-fill − ST | +2.25 | −0.31 to +4.71 | 87 / 65 | 0.088 |

*Result.* This filled variant showed no detectable gain; the CI excludes gains larger than about +2.4 pp. The reading fixed in advance ("too few injected tokens caused the missing gain" is not supported) is kept as the recorded conclusion, but its scope is narrow: filling changes both the amount and, through the ridge term, which tokens are added, so this run does not isolate the effect of the amount. Every step injected exactly m tokens (2,231 steps; the filled tail had a median of 420 tokens). *Groups* (fixed in advance, by the first-step end in the main ST run): closed 322 q, VisRef-fill − TSR −0.62 (−2.48, +1.24); truncated 629 q, +1.75 (−1.43, +5.09); EOS 25 q, −36.0 (−64, −8; 4 vs 13). *Limitations.* The filled tail was nearly unrelated to relevance (median Spearman correlation of tail order with relevance −0.10; negative means more relevant tokens first), and εI also changes the selection, so this tests "more tokens of little relevance", not "top-m by relevance". In the paper, random selection is close to ST (Table 7) while selection by diversity alone is well above it (Table 3: 77.4; Table 8 at λ = 0: 71.2; the two settings may differ); the filling is neither random nor relevance-ordered, so the paper's ablations do not tell us what to expect from it. The VisRef-fill run also generated the first thinking step again: its text matches the main run in 242 of 994 questions, which adds noise to the paired comparisons. An earlier relevance-filled variant ran on the first 74 questions only and was dropped. The reading of the small EOS group was decided after seeing the results: when step 1 ended almost at once, the kernel held few tokens and filling added hundreds. The LLM fallback for this run used a Claude Haiku-class extractor instead of gemini-3.5-flash-lite (a documented deviation); it changed no label. *Second trial answer without Ṽ_k* (recorded by design, decision unchanged; interpretation decided after the results; H recorded with and without Ṽ_k in context): median H − H_noV = 0.0006; the stop/continue decision would differ at 151 of 2,231 steps (6.8%), 110 continuing only with Ṽ_k and 41 only without, 129 of the 151 at truncated steps. The two trial answers used separate random streams, so this count mixes sampling noise with the effect of Ṽ_k.

### 4.3 step1_branch: re-injection at step 1 (rule fixed before the results)

> **Card A3.** *Question:* does re-inserting the image after step 1 help by itself? *Result:* with − without the image +1.40 pp (answer at once) and +0.97 pp (answer after up to 512 more tokens), both Holm p 0.658. *Limits:* only the 309 questions where a method continued after step 1; designed to detect effects of about 5–6 pp; says nothing about later steps, the stopping rule or the insertion format. *Files:* `repro/kaggle/runs/step1_branch_pilot`, `step1_branch_main`, `repro/diag/analyze_step1_branch.py`.

*Motivation (chosen after the results).* The stored step-1 stopping trial answers of the main run (1 sample, T = 1) gave VisRef 58.9 vs TSR 57.2 on 1,000 questions (28 vs 11 discordant, McNemar p = 0.0095, CI +0.5 to +3.0). All of the gap lies in the 309 questions where TSR or VisRef continued after step 1 (36.6 vs 31.1); the other 691 had no discordant pair. Their final answers did not differ (42.4 vs 43.0, 30 vs 32 discordant). These exploratory p-values are uncorrected.

*Design (fixed before results).* The 309 questions above (step 1 ended truncated in 297, EOS in 7, closed in 5). All conditions start from the same stored step-1 thinking z_1. **A:** no re-injection, answer immediately (`</think>` + `\n\n\boxed`). **B:** re-inject the DPP-selected Ṽ_1, answer immediately. **C:** no re-injection, up to 512 more free thinking tokens, then answer. **D:** re-injection plus up to 512 free thinking tokens, then answer. Answer = mean of 3 samples at T = 1 (plus one greedy answer), scorer v4. The two main comparisons are B−A and D−C, paired bootstrap 95% CI (2,000, seed 0), Holm over the two. A comparison counts only if its CI lower bound is above 0.

| Condition | After z_1 | Accuracy %, mean of 3 (T = 1) | Accuracy %, greedy |
|---|---|---|---|
| A | answer immediately | 33.98 | 35.28 |
| B | Ṽ_1, answer immediately | 35.38 | 36.57 |
| C | ≤512 free thinking tokens | 35.28 | 39.81 |
| D | Ṽ_1, ≤512 free thinking tokens | 36.25 | 37.86 |

| Comparison | Diff (pp) | 95% CI | Holm p | Greedy: only first / only second (McNemar p) |
|---|---|---|---|---|
| B − A (main) | +1.40 | −1.19 to +3.99 | 0.658 | 18 / 14 (0.597) |
| D − C (main) | +0.97 | −2.27 to +4.21 | 0.658 | 14 / 20 (0.392) |
| D − B (secondary) | +0.86 | −2.16 to +3.88 | n/a | 20 / 16 (0.618) |
| C − A (secondary) | +1.29 | −1.83 to +4.42 | n/a | 25 / 11 (0.029, uncorrected) |
| (D−C) − (B−A) (secondary) | −0.43 | −4.21 to +3.24 | n/a | n/a |

*Correction (2026-10-12, after publication):* the bootstrap p-values were first computed on float accuracies (thirds), so resample means that are exactly 0 fell into one tail or the other depending on the Python version (p 0.289 / 0.575 and Holm 0.578 on Python ≤ 3.11; 0.281 / 0.583 on 3.12+). The analysis now uses integer counts of correct answers; the bootstrap p is 0.329 / 0.584 (Holm 0.658 for both) on every Python version. The CIs, the other numbers and the conclusion are unchanged; the earlier output is kept in `repro/diag/step1_branch_analysis_float_py311.json`.

*Conclusion by the rule fixed before the results: the immediate-answer gain was not reproduced.* The gap seen in the stored trial answers may have been noise. The wording fixed for a null result is that effects of about 5–6 pp or more were not detected (5–6 pp was the planned 80%-power minimum detectable effect); the observed CI upper bounds are about +4 pp. In the free-thinking conditions the model closed `</think>` itself in 99 (C) vs 82 (D) of 309 cases, reached the 512 cap in 203 vs 220, and used image-related words in 22.7% vs 31.4% of the free-thinking texts, so re-injection changed what the model wrote about without a detectable change in accuracy. *Reconstruction check:* re-tokenised length matched the main run in 301 of 309 questions and the selection size in 296; restricting to the 293 matching both gives the same conclusion. *Added after the results, not fixed in advance:* D−A is +2.27 pp (95% CI −0.97 to +5.72); under greedy decoding the order was C 39.81 > D 37.86 > B 36.57 > A 35.28, so "D is best" is not robust. *Scale:* the experiment covers only re-injection at step 1 on these 309 questions. If a +3.5 pp gap on 1,000 questions came from these 309 alone it would need about +11 pp within them, while the CI upper bounds correspond to about +1.2 to +1.3 pp on the 1,000-question scale. It says nothing about stopping rules, step lengths, injection format, or steps after the first. Outside these 309 questions, where both methods stopped at step 1 and VisRef injected Ṽ_1 just before the answer, VisRef and TSR final answers also differ in correctness in 27 questions (14 vs 13).

### 4.4 Comparison with other numbers in the paper (after the results)

- **Visual tokens.** The paper reports ∼1,772 visual tokens per image (InternVL-3.5-8B, MathVista; the 1B and 8B models use the same InternViT-300M). Recomputing InternVL's dynamic tiling from the image sizes reproduces our recorded counts on 1,000/1,000 questions. Mean |V|: max_num 12 with thumbnail 1,961.5 (+10.7% vs 1,772); max_num 12 without thumbnail 1,755.6 (−0.9%); max_num 6, 8 or 10 give −24% to −41%. This is compatible with, but does not show, that the authors dropped the thumbnail tile.
- **Answer handling.** Accuracy under four scoring variants against Table 6 (ST / TSR / VisRef, gap = ours − paper): main scoring 57.8 / 60.1 / 60.0 (+11.7 / +11.6 / +8.0); unfinished answers (rows that needed answer forcing) counted wrong 44.4 / 48.3 / 47.4 (−1.7 / −0.2 / −4.6); answers forced after cut thinking counted wrong 25.2 / 26.4 / 26.9 (−20.9 / −22.1 / −25.1); both counted wrong 24.4 / 25.3 / 26.3 (−21.7 / −23.2 / −25.7). Only counting unfinished answers wrong gives numbers near the paper's ST and TSR. That is a hypothesis formed after the results and selected among four variants; under it TSR−ST is +3.9, farther from +2.4 than the main scoring's +2.3. VisRef−TSR lies between −0.9 and +1.0 under all four variants, so answer handling alone does not explain a missing +3.5.
- **Thinking volume.** Generated tokens relative to ST: TSR 1.98×, VisRef 2.14×. The paper's latency ratios (Table 5, 8B on H100) are 1.08× and 1.15×. The models differ, so this only suggests that continuation was rarer in the authors' setup. In ours, 64.1% of ST first steps hit the cap, and 73.0% (TSR) and 71.3% (VisRef) of questions stopped at step 1.

### 4.5 VisRef-hs: last-layer Z (rule fixed before the results)

> **Card A4.** *Question:* does choosing image tokens with the model's last-layer vectors help? *Result:* VisRef-hs − TSR +0.24 pp (839 questions); the injection reached m tokens. *Limits:* the first step was regenerated (193 of 839 identical), which lowered the power for +3.5 pp to about 72%; only a median of 70 picks add clearly new information; the extractor was changed after the rules-only result (rules only: same conclusion); 161 questions not run. *Files:* `repro/kaggle/runs/b_main_hs`, `repro/diag/analyze_hs.py`.

*Question.* Does VisRef gain over TSR when the thinking-step vectors Z in the DPP kernel are the model's last-layer hidden states (after the final norm, the vectors that feed the output layer) instead of the input embeddings? With input embeddings the kernel rank is bounded by the number of distinct tokens in a step, so only about one fifth of m is injected (Section 4.1, item 4). Last-layer vectors differ at every position, so the DPP can pick m tokens when a step has at least m tokens. This experiment was added after the main results.

*Design (fixed before results: [commit ccac0d9](https://github.com/AhnJG/visref-repro/commit/ccac0d9), entry of 2026-10-12 in `docs/EXPERIMENT_FLOW.md`).* Only Z changes; the DPP, the visual vectors V, the injection format, the stopping trial answer, answer handling, token caps (1,024 per step, 10,240 in total), seeds and question order are those of the main run. By design only Z changes, but the first thinking step was generated again in this session: with the same seeds, the text diverged (batch-dependent fp16 arithmetic, Section 6) and matches the main run in only 193 of 839 questions, whereas the main run's TSR and VisRef share ST's step 1 in all 1,000. All 1,000 questions are queued in the seed-0 order in one Kaggle session (2×T4, timeout 29,000 s); only the complete prefix is analysed, against the main run's ST, TSR and VisRef rows of the same questions. Scoring as in Section 2.2; the rules-only result is reported too. Main comparison VisRef-hs − TSR (paired bootstrap 95% CI, exact McNemar test): CI lower bound > 0 → extra gain with last-layer Z; CI contains 0 and upper bound ≥ +3.5 pp → undecided; upper bound < +3.5 pp → the paper-size gain (+3.5 pp) is not reproduced with last-layer Z. Fewer than 300 finished questions → reported as underpowered. Run settings: `repro/kaggle/runs/b_main_hs/run_config.json` (its code hashes equal `repro/` at commit ccac0d9). Analysis: `repro/diag/analyze_hs.py`, written and tested before the results.

*Run.* One Kaggle session finished without errors; 839 questions form the complete prefix (300 or more, so the flag for an underpowered run is off; that threshold assumed the main run's 9% disagreement rate). Because step 1 was regenerated, 134 of 839 pairs disagree (16.0%): the smallest effect detectable with 80% power is about 3.9 pp, and the power for +3.5 pp about 72% (normal approximation). The code version recorded in every row (`f1376774850d`) equals `repro/` at commit ccac0d9.

*Result (839 questions, main scoring).* ST 57.57 / TSR 59.71 / VisRef 59.36 / VisRef-hs 59.95 %. VisRef-hs − TSR +0.24 pp (95% CI −2.38 to +3.10; 68 vs 66 discordant, McNemar p = 0.93). The upper bound is below +3.5, so by the rule fixed before the results the paper-size gain over TSR (+3.5 pp) is not reproduced with last-layer Z; smaller gains are not excluded. Secondary (descriptive; step 1 also differs, see Design): VisRef-hs − VisRef +0.60 pp (−1.67 to +2.98), VisRef-hs − ST +2.38 pp (−0.24 to +5.01). Rules only: VisRef-hs 60.07 %, VisRef-hs − TSR +0.36 pp (−2.26 to +3.22), same conclusion.

*Mechanism (descriptive).* The injection reached m: median injected/m 1.0 (quartiles 0.977 / 1.0 / 1.0), against 0.205 for VisRef with input embeddings on the same questions; 1,130 of the 1,678 steps with at least m tokens injected exactly m. The steps that stayed below m were of two kinds: 288 steps shorter than m tokens (242 of them closed naturally), where the DPP can pick at most one image token per thinking token (284 picked exactly that many); and 548 steps of large images (544 with m = 998, i.e. 3,328 visual tokens, and 4 with m = 844), where m is close to the rank of the visual vectors (median 1,009) and the greedy DPP stopped at its numerical threshold at a median of 98.2% of m (of the 582 steps with m = 998 that were at least m tokens long, 38 reached m). Only a median of 70 picks (13.6% of m) added clear diversity (normalized residual ≥ 0.01), and the later picks followed relevance only weakly (median rank correlation −0.171; negative means more relevant tokens first). A second `</think>` appeared in 395 of 839 final answers (VisRef 392, ST 109, TSR 99). Mean number of steps k: 2.34 (VisRef 2.27, TSR 2.10). The first-step end type agreed with the main run in 763 of 839 questions, the step-1 text in 193.

*Deviation from the plan fixed before the results.* The 23 rows the rules could not parse were extracted with Claude Haiku, with the same official MathVista prompt and no ground truth, instead of gemini-3.5-flash-lite (as in the VisRef-fill run). 21 extractions were used and 2 were empty (rule result kept); the 20 control rows all agreed with the rules. One row changed from correct to wrong (pid 722, whose response ends with "The correct answer should be B) no"). The change of extractor was decided after the rules-only result had been computed and logged. The compared main-run rows were extracted with gemini-3.5-flash-lite, so the rules-only reading, which scores all four methods the same way, is the fair comparison; it gives the same conclusion.

*Vector spaces.* In VisRef-hs, Z comes from the last layer while the image tokens V stay in the input-embedding space, so the kernel multiplies vectors from two different spaces (the paper's Eq. 7 uses one d-dimensional space). A6 (VisRef-imgtag) takes both Z and V from the last layer.

*Reading.* Injecting m tokens chosen with last-layer vectors brought VisRef no detectable gain over TSR in this implementation (CI upper bound +3.10 pp). The other differences on our side (injection format and position, the stopping estimate, token limits; Section 4.1) remain untested.

### 4.6 VisRef-imgtag: tagged image block after `</think>` (rule fixed before the results)

> **Card A6.** *Question:* does the image-tag injection bundle (VisRef-imgtag) beat TSR with the same answer prompt? *Result:* VisRef-imgtag − TSR with answer prompt −1.0 pp (1,000 questions, 95% CI −3.1 to +1.0): no paper-size gain. *Limits:* it changes four unstated settings at once, so the result belongs to the bundle, not to any one setting; it ran on a different GPU and torch version from the main run, so it is compared only within this run; the bundle was designed after seeing the main-run outputs. *Files:* `repro/kaggle/runs/imgtag_main`, `repro/diag/analyze_imgtag_main.py`.

*Question.* In the main run, VisRef's answers often contained another `</think>` (352 of the 713 questions where VisRef stopped after one step, against 67 for ST), and only about one fifth of m image tokens were injected. VisRef-imgtag changes four settings at once: the selected tokens are wrapped in InternVL's `<img>` … `</img>` tags with line breaks; they are placed after `</think>` when the model answers and inside the thinking when it continues; and the selection uses last-layer vectors for both the thinking step (Z) and the image tokens (V), while the injected vectors stay input embeddings. Every final answer starts after `\nFinal Answer:` (B7). Its partner, TSR with answer prompt, uses the same answer prompt without the image, so their difference is the injected image block.

*Design (fixed before results: [commit 8f5c09e](https://github.com/AhnJG/visref-repro/commit/8f5c09e), entry of 2026-10-15 on the 1,000-question image-tag comparison in `docs/EXPERIMENT_FLOW.md`).* Three methods in one run, VisRef-imgtag, TSR with answer prompt and plain TSR, all starting from the main run's stored first thinking step with one shared random state per question after it. Other settings as in the main run (1,024 tokens per step, 10,240 in total, stopping threshold 0.25, at most 10 steps, m = 30% of the image tokens, temperature 0.6, seed 42, fp16). Code version `6e218887d2b5`, one RunPod RTX A5000 (torch 2.12.1; the main run used 2 × T4 and torch 2.11.0). Main comparison VisRef-imgtag − TSR with answer prompt with the three-way rule of Section 4.5 against the paper's +3.5 pp. Extra rule: an answer that is empty after the prompt is wrong. Secondary, without conclusions: VisRef-imgtag − TSR (the paper's definition of the comparison) and TSR with answer prompt − TSR (the prompt alone). Analysis: `repro/diag/analyze_imgtag_main.py`, written and tested before the results; descriptive items added during the run, before the results, are listed in the entries of 2026-10-16.

*Run.* All 3,000 rows (1,000 questions × 3 methods) finished without errors, with one code version and one stored-first-step file; generation took 22.6 h (Table 2.5 adds the 6-question check run). A structure check at 238 questions, without accuracy, found no reason to stop. TSR and TSR with answer prompt, which should think identically, did so in 969 of 1,000 questions (all 334 whose first step the model closed itself and all 25 that ended at EOS; 610 of the 641 cut at 1,024 tokens); the cause of the 31 differences was not examined. 108 of 1,000 pairs of the main comparison disagree (10.8%): the smallest effect detectable with 80% power is 2.9 pp, and the power for +3.5 pp 0.92.

*Result (1,000 questions, main scoring).* VisRef-imgtag 60.1 / TSR with answer prompt 61.1 / TSR 60.4 %. VisRef-imgtag − TSR with answer prompt −1.0 pp (95% CI −3.1 to +1.0; 49 vs 59 discordant, McNemar p = 0.39). The upper bound is below +3.5, so by the rule fixed before the results this bundle brings no paper-size (+3.5 pp) gain over TSR with the same answer prompt; gains up to +1.0 pp are not excluded. Secondary: VisRef-imgtag − TSR −0.3 pp (−2.4 to +1.8; 52 vs 55, p = 0.85); TSR with answer prompt − TSR +0.7 pp (0.0 to +1.5; 10 vs 3, p = 0.09). The LLM extracted 9 rows (3 / 1 / 5) and changed no result; the 20 control rows all agreed with the rules, so the rules-only reading gives the same accuracies and conclusion. Answers empty after the prompt, counted wrong: 16 / 3 / 2. Sensitivity (fixed before the results): resampling the 31 MathVista source datasets instead of questions gives a CI of −3.0 to +1.1; leaving out one source moves the difference between −1.6 (CLEVR-Math) and −0.5 pp (DVQA). Reference only: this run's TSR − the main run's TSR on the same questions +0.3 pp (−1.5 to +2.0).

*Mechanism (descriptive).* The injection reached m (median injected/m 1.0). VisRef-imgtag took more steps (mean k 2.47 against 2.07 for TSR with answer prompt; 88 against 58 questions reached the 10-step limit). Another `</think>` still appeared in 249 final answers (TSR with answer prompt 107, TSR 113), so the format repair seen in the 10-question trial (0 of 5) did not carry over to all questions. By how the stored first step ended (a look made after the results): after a step the model closed itself (334 questions), VisRef-imgtag lost 9 questions and gained none (69.2 against 71.9 %), and 15 of its 16 empty answers fall in this group (5 of the 9 lost questions); after a cut step (641), 55.4 against 54.9 % (49 vs 46 discordant); after EOS (25), 0 vs 4. The stop decision with and without the image differed in 287 of 2,450 steps (same random state); the mean entropy difference with − without the image was +0.08 (4 steps whose trial answer without the image did not close its answer are left out of this mean, a fix made after the results because the mean was infinite). Cost: VisRef-imgtag generated 1.13 times the tokens of TSR with answer prompt, and its median time per row was 1.44 times larger (rows run in batches, so times are approximate); the paper's Table 5 reports 8.2 against 7.7 s per question for VisRef and TSR (InternVL3.5-8B, one H100), a ratio of about 1.06.

*Reading.* Injecting m tokens chosen with last-layer vectors, in the official image-tag format after `</think>`, brought no detectable gain over TSR with the same answer prompt. The format was not fully repaired: answers still contained another `</think>` more often than without the image, and empty answers concentrated after steps the model had closed itself, where the injected block only cost questions. This is the fourth variant designed after the main run, and there is no correction across A2–A6; the planned confirmatory run (Section 5.4) is the single test fixed in advance.

## 5. Discussion

### 5.1 What was easy

- Model, dataset and benchmark are public and could be fixed to one exact revision; the official model page gives the thinking prompt and the sampling parameters.
- Algorithm 1 fixes the loop and the order z_k → Ṽ_k → H_k; the continuation cue (Section 3, Fig. 2) and δ, m and K_max are given in the text; the DPP greedy MAP has public reference code.
- Pairing (common random numbers, shared first step) made differences of 2 pp measurable with 1,000 questions on free-tier GPUs.

### 5.2 What was difficult

We could not find the following in the paper; each was a choice of ours (Section 2.1):

- token budgets per step and in total, and what happens when thinking is truncated or an answer hits the cap;
- answer extraction and scoring (rules, LLM judge, which model);
- the layer of the reasoning-token representation Z (and of V);
- what to do when the DPP kernel rank is below m;
- the stopping-entropy estimator (number of answer samples, temperature, trial-answer prompt, log base);
- the injection format: tags, separators, `</think>` handling, position ids, answer prompt after Ṽ_k;
- the image tiling (thumbnail tile).

The paper's latency table and token-usage figure were not used as reproduction targets (Section 4.4 compares the latency ratios only for reference).

### 5.3 Open questions about the paper's setup

Details that the paper and the works it cites leave open, and that we set ourselves (Section 5.2):

1. Per-step and total thinking token limits; handling of truncated thinking.
2. How answers that reach the length limit were scored.
3. The answer extraction method (rules or LLM; which model).
4. The estimator of H: samples, temperature, trial-answer prompt, log base, and whether token entropies are summed or averaged.
5. The layer used for Z and V in the DPP kernel, and whether the two are taken from the same vector space.
6. How |Ṽ_k| = m is enforced when the kernel rank is below m.
7. Injection format, separators, `</think>` handling and position ids.
8. Tiling settings (max_num, thumbnail) behind ∼1,772 visual tokens.
9. For 1B: the distribution of the number of steps k and the tokens per trace.
10. Sampling parameters, number of runs and hardware/precision for Table 6.

### 5.4 Next experiments (planned)

> **Card A5.** *Question:* does thinking more after the re-inserted image help? *Status:* decision rule fixed before the run; not run yet. *Limits:* it changes the paper's stopping behaviour, so a gain would support "a variant with thinking after the image", not a reproduction of VisRef; the extra 512 tokens are outside the 10,240-token budget; because VisRef-hs regenerated step 1, the 2 × 2 comparison with it is not directly comparable in that cell. *Files:* `repro/diag/analyze_think.py`.

One set of experiments (A5, three runs) is planned; its decision rules are fixed and published before each run. The rules for (2) and (3) were written down on 2026-10-13, before any run (entry of 2026-10-13 in `docs/EXPERIMENT_FLOW.md`; analysis `repro/diag/analyze_think.py`). (1) VisRef-hs on the 161 questions the session did not reach, with the same design, reported next to the recorded 839-question conclusion. (2) Thinking after the image (`visref_think`): when the stopping rule fires, the model continues after Ṽ_k with the paper's cue "Wait, think more" and thinks up to 512 more tokens before answering, against TSR with the same extra thinking and no image (`tsr_think`); VisRef stopped at step 1 in 71.3% of questions, so in most questions the re-injected image currently comes right before the answer. (3) Both together (`visref_hs_think`), which completes a 2 × 2 design with the main run and VisRef-hs. Following an independent review on 2026-10-13, (2) and (3) start from the main run's stored first thinking step and run together with their control, because the follow-up runs so far regenerated step 1 (only 193 of 839 VisRef-hs first steps match the main run), which roughly doubled the paired noise. The extra 512 tokens are outside the 10,240-token budget, answers are handled as in the main run, and the main comparison `visref_think` − `tsr_think` uses the three-way rule of VisRef-hs (Section 4.5) against the paper's +3.5 pp. With the measured disagreement rates of 9% (main run VisRef − TSR) and 16% (VisRef-hs − TSR), the power to detect +3.5 pp on 1,000 questions is 0.96 to 0.79. Rough cost from the measured VisRef-hs speed (839 questions in about 7.7 h on 2×T4): about 2 h for the remaining 161 questions and 9 h or more per method for 1,000 questions (an estimate, not measured). Because the existing VisRef-hs run regenerated step 1, its cell of the 2 × 2 design is not directly comparable with the other three.

A single confirmatory run with the setting closest to the paper's is planned next: Z and V from the same layer, a tagged image block, m tokens reached, a per-step limit of at least 4,096 tokens, the stored first step shared by all methods, and bf16. Its exact design is fixed before the run starts. The follow-ups above stay exploratory.

## 6. Limitations

- One model size (1B), one benchmark (MathVista testmini), one seed and one sample per question; the paper's 2B and 8B rows and other benchmarks were not tested.
- T4 fp16 results depend on batch composition: the same seed can give different outputs (16 of 730 predictions differ between TSR at k = 1 and ST; 874 of 976 first-step end types agree between the VisRef-fill run and the main run, but only 242 of 994 first-step texts are identical; for VisRef-hs, 763 and 193 of 839).
- The paper provides no per-question outputs, so the difference between our result and the paper's cannot be tested formally; the CI classification is a coarse check.
- Absolute accuracies are 8–12 pp above the paper's. Candidates (treatment of unfinished answers, thinking budget, extraction, prompt) were not isolated; the closeness of the variant that counts unfinished answers wrong to the paper's ST and TSR is a hypothesis formed after the results.
- Choices of ours (Z layer, H estimator, injection format, token limits) and the confirmed fact that too few image tokens were injected limit what the null result on VisRef−TSR means; it concerns this implementation only.
- The LLM fallback differs from the official GPT-based extractor; the VisRef-fill and VisRef-hs runs used Claude Haiku instead of the Gemini model of the main run.
- The setting checks (Section 2.6) ran on questions that are also among the 1,000 evaluated questions, so the settings may be tuned to them to some degree; most were chosen from stopping rates and broken answer formats, not from accuracy.
- The main comparisons and tests were fixed after we looked at partial results for 392 questions and the CI classification rule after one at 759 (Section 2.4); the analyses made after the results are exploratory.
- The independent reviews of the code and analyses were done by separate AI sessions, not by human reviewers. Untried ideas, such as a step cap of 4,096 or multi-sample H, were not run; last-layer Z was run as a follow-up (Section 4.5).

## 7. Where to find things

| What | Where (in this repository) |
|---|---|
| Chronological log with the decision rules fixed before results, and decisions (Korean; starts with a stage summary and a list of abbreviations) | `docs/EXPERIMENT_FLOW.md`; the original text of the work before 2026-10-07 with earlier code: [lines 5–749 of the log at commit 4776f6a](https://github.com/AhnJG/visref-repro/blob/4776f6ab2607957799bef8db70241cc64313feae/docs/EXPERIMENT_FLOW.md?plain=1#L5-L749) |
| Evidence per implementation decision, external-number list (Korean) | `docs/IMPLEMENTATION_CHOICES.md` (start with its summary table) |
| Script and numbers for the main conclusion | `repro/diag/final_judgment.py`, `repro/diag/final_judgment.json` |
| Follow-up analyses | `repro/diag/analyze_fill.json`, `repro/diag/step1_branch_analysis.json` (earlier float version: `step1_branch_analysis_float_py311.json`), `repro/diag/paper_fingerprints.json`, `repro/diag/step1_probe_analysis.json`, `repro/diag/analyze_k1_injection.json`, `repro/diag/st_plain_analysis.json`; `repro/diag/analyze_hs.json` (Section 4.5; the script was written before the results) |
| Raw generations, run configs and environments per session | `repro/kaggle/runs/<session>/` (`b_main_s1`–`b_main_s9`, `b_main_fill`, `step1_branch_pilot`, `step1_branch_main`, `diag_st_plain`, `b_main_hs`, `bundle_pilot` and `bundle_pilot2` (the two 10-question trials of the image-tag variant, B7); setting checks of Section 2.6: `smoke`, `smoke2`, `smoke3`, `main_s0`, `full_s1`, `b_test`, `hidden_rank_check`) |
| Final scored rows (rules + LLM fallback) | `repro/llm/final/merged.jsonl` (VisRef-fill run: `repro/llm/fill/`, VisRef-hs: `repro/llm/hs/`) |
| Final tables (CSV/JSON, built from the `repro/diag/*.json` files by `scripts/make_results.py`) | `results/`; the questions used by the setting checks: `results/setting_check_questions.csv` (listed from the run records, not built by `make_results.py`); GPU and generation time per experiment: `results/compute.csv` (built by `scripts/make_compute.py`) |
| Figures (the results figure and the main-run figure drawn by `scripts/make_figures.py`; method diagram drawn by hand) | `figures/` |
| Descriptive per-method report (rule-based scorer v4 only, no LLM fallback) | `repro/reports/b_main_report.html`, `repro/reports/b_main_report.json` |
| Code | `repro/run.py`, `repro/visref/` (engine, methods, DPP, scoring), `repro/diag/step1_branch.py`, `repro/diag/analyze_imgtag*.py` (image-tag analyses), `repro/runpod/` (runner for the A6 run on a rented GPU) |
| Public sources behind the logs' citations (arXiv versions, pages, tables) | `docs/SOURCES.md` |
| Reproduction commands and repository layout | `docs/REPRODUCE.md` |

## 8. Contributions and AI use

This reproduction was carried out by AhnJG with AI assistance (Claude models by Anthropic, Codex by OpenAI).

- **AhnJG:** chose the target and the evidence rules; made the design decisions recorded in `docs/EXPERIMENT_FLOW.md`, among them injecting only the DPP's rank in the main VisRef, the answer handling (s1 prompt + DEER answer forcing), rule-based scoring with an LLM fallback, fixing the main comparisons and tests before the final analysis, the follow-up experiments and their scope, and when to stop; raised the questions behind several recorded corrections.
- **Claude (AI assistant):** wrote most of the code (generation engine, methods, DPP selection, scoring, analysis scripts, tests); drafted the decision rules fixed before the results, the analyses, this report and the figures; separate AI sessions carried out independent reviews of the code and analyses.
- **Codex (OpenAI):** used throughout the project to review the content.
- **An LLM as part of the method:** gemini-3.5-flash-lite extracts the answer for the rows the rules cannot parse (Section 2.2).
