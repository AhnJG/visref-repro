# visref-repro

[한국어 (Korean)](README.ko.md)

Unofficial reproduction of **VisRef** (Ghosal et al., *VisRef: Visual Refocusing while Thinking Improves Test-Time
Scaling in Multi-Modal Large Reasoning Models*, arXiv:2603.00207, CVPR 2026), Appendix Table 6 row *InternVL-1B*:
InternVL3.5-1B on MathVista *testmini* (1,000 questions), comparing

| method | what it does | paper (Table 6) |
|---|---|---|
| **ST** – standard thinking | one reasoning trace, then the answer | 46.1 |
| **TSR** – textual self-reflection | "Wait, think more" steps until the answer entropy is below δ | 48.5 |
| **VisRef** | TSR + re-injecting a DPP-selected subset of the image's visual tokens after every step | 52.0 |

![ST, TSR and VisRef as run in this reproduction; orange notes mark settings we could not find in the paper](figures/method_overview.svg)

> **Status (2026-10-13):** the main run and two follow-up experiments are finished and analysed. A third follow-up
> (`visref_hs`, last-layer Z) was pre-registered before its results; the run has finished and its analysis is in progress ([REPORT, Section 4.5](REPORT.md#45-visref_hs-last-layer-z-pre-registered)).
> Full write-up: **[REPORT.md](REPORT.md)**.

## Results

Scoring (fixed in advance): answers are extracted by rules; for the rows the rules cannot parse (36 of
3,000), an LLM extracts the answer with the official MathVista extraction prompt, and its answer counts only if it
appears in the response (see [LLM fallback extraction](#3-llm-fallback-extraction-rows-the-rule-scorer-cannot-parse-about-1)).
n = 1,000; each method once per question, with shared seeds and a shared first thinking step.

![Paired accuracy differences with 95% CIs next to the paper's differences](figures/paired_differences.svg)

| accuracy (%) | ST | TSR | VisRef |
|---|---|---|---|
| paper (Table 6) | 46.1 | 48.5 | 52.0 |
| **ours** | **57.8** | **60.1** | **60.0** |

| comparison | ours, pp (95% CI) | only first / only second correct | McNemar p | Holm p | paper, pp | classification |
|---|---|---|---|---|---|---|
| VisRef − ST (main) | +2.2 (+0.3, +4.1) | 60 / 38 | 0.033 | 0.067 | +5.9 | effect size not reproduced (same direction) |
| VisRef − TSR (main) | −0.1 (−2.0, +1.7) | 44 / 45 | 1.000 | 1.000 | +3.5 | effect size not reproduced |
| TSR − ST (supplementary) | +2.3 (+0.6, +4.0) | 51 / 28 | 0.013 | – | +2.4 | consistent with the paper value |

- **In short:** TSR's gain over ST is consistent with the paper's value (this does not show that the effect or its
  mechanism is the same). We did not confirm an additional gain of VisRef over TSR in our InternVL3.5-1B
  implementation. VisRef − ST is smaller than in the paper (+2.2 vs +5.9 pp); its CI excludes 0, but after the Holm
  correction we fixed in advance for the two main comparisons it is not significant (p = 0.067; 0.033 uncorrected).
- **Classification column:** *consistent with the paper value* = our 95% CI contains the paper's difference and
  excludes 0 (not a proof of equivalence); *effect size not reproduced* = our 95% CI excludes the paper's difference.
  The CIs are paired bootstrap intervals (2,000 resamples, seed 0).
- **When the rules were fixed:** the main comparisons and tests after an interim look at the first 392 questions, and
  the classification rule after an interim look at 759 questions, both before the final analysis. The main run is
  therefore not a pre-registration in the strict sense and its classification is not confirmatory ([REPORT, Section 2.4](REPORT.md#24-timing-of-pre-registration)). The
  classification is the same under every scoring variant we tried ([REPORT, Section 3.2](REPORT.md#32-sensitivity-to-answer-handling)). The decision rules of the
  follow-up experiments below were fixed before their results.

### Why our accuracies are 8–12 pp above the paper's (not resolved)
Our ST (57.8) is close to the official InternVL3.5-1B MathVista (mini) score of 59.3, while the paper's ST is 46.1.
The paper's 8B ST (68.1) is also more than 10 pp below the official InternVL3.5-8B score (78.4). This suggests that the
paper evaluated its baselines in a setting where the model scores lower than in the official evaluation, but we could
not find that setting in the paper. Our estimates of the cause, none of them tested directly:

1. **Unfinished answers counted as wrong.** If every answer that did not finish within the answer limit, and so needed
   answer forcing, is counted wrong, our accuracies become ST 44.4 / TSR 48.3 / VisRef 47.4, close to the paper's ST
   and TSR (46.1 / 48.5). (We picked this comparison among several scoring variants after seeing the results;
   [REPORT, Section 4.4](REPORT.md#44-paper-fingerprints-post-hoc).)
2. **Token limits, answer extraction, prompt.** We could not find these in the paper. A lower thinking limit, for
   example, would cut more answers short.

The official scores are from the InternVL3.5 report (Table 3); where each paper number comes from: `docs/SOURCES.md`.

### Why VisRef showed no extra gain here
We look for the cause in our implementation first. The differences that could explain it, and what we checked:

| candidate on our side | why it could matter | checked? |
|---|---|---|
| We inject about one fifth of the paper's m tokens: the greedy DPP stops at the kernel rank (confirmed procedural difference) | less visual evidence per step | Not settled. A variant that topped the selection up to m (`visref_fill`, below) gave no extra gain, but the top-up chose tokens without regard to relevance (rank correlation with relevance −0.10), and in the paper random selection is also close to ST (Table 7). Injecting m relevant tokens was not tested |
| Injection format: InternVL wraps an image in `<img>` … `</img>` tags; we re-inject the selected visual embeddings without these tags, right before `</think>` (we could not find the paper's format) | the model may not treat them as part of an image | In a small early test (about 20 questions per method, other settings) the tags made the model write a second `</think>` in 31 of 63 steps, against 10 of 65 without them, so we dropped the tags; not tested in the main run. A side effect remains without tags: of the 713 questions where VisRef stopped at step 1, 352 VisRef final answers contain another `</think>`, versus 67 ST answers |
| Where the injection lands: TSR and VisRef continued almost only after a step hit the 1,024-token cap, so the injection usually comes mid-sentence | the paper may have continued mostly after completed steps | A larger step cap (e.g. 4,096) was not run |
| What the image pieces are matched against: the DPP picks image pieces that fit the thinking text, which it reads as vectors. We use the vectors right after the words are turned into numbers (input embeddings), not those of a deeper layer of the model; we could not find which layer the paper uses | a different layer would pick different image pieces | Tested as a follow-up (`visref_hs`, below; analysis in progress). Offline check on 12 questions: with deeper-layer vectors the DPP can return m tokens when a step is long enough, as the paper states, but only about 50–110 of them add clear diversity |
| Stopping entropy H from one sampled answer (see below) | noisy stop decisions | Several samples per step were not used in the final runs |

None of these is ruled out, and we could not find CIs or the number of runs behind the paper's numbers, so we cannot
say how much of the gap comes from our implementation. Details: [REPORT, Section 4](REPORT.md#4-beyond-the-paper).

### Follow-up experiments (decision rules fixed before the results)

| experiment | question | result |
|---|---|---|
| `visref_fill` (976 q) | Does VisRef change if the DPP selection is topped up to exactly m tokens? (The top-up turned out to choose tokens without regard to relevance.) | fill − TSR 0.0 pp (−2.15, +2.36); fill − VisRef +0.1 pp (−2.15, +2.36): adding tokens chosen without regard to relevance gave no extra gain. Whether m relevant tokens help was not tested. |
| `step1_branch` (309 q) | On the questions where TSR or VisRef continued after step 1: does re-injecting Ṽ₁ help an immediate answer (B − A), or an answer after ≤ 512 more thinking tokens (D − C)? | B − A +1.40 pp (−1.19, +3.99); D − C +0.97 pp (−2.27, +4.21); Holm p 0.658 for both: the immediate-answer gain seen post hoc in the main run was not reproduced. |
| `visref_hs` (839 q, analysis in progress) | Does VisRef gain over TSR when Z, the vectors of the thinking text, are the model's last-layer vectors instead of input embeddings? (With them the DPP can pick m tokens when a step has at least m tokens.) | Analysis in progress (LLM fallback extraction pending). Rule fixed before the results ([REPORT, Section 4.5](REPORT.md#45-visref_hs-last-layer-z-pre-registered)): CI lower bound of hs − TSR above 0 → extra gain; CI containing 0 with upper bound ≥ +3.5 pp → undecided; upper bound < +3.5 pp → the paper-size gain is not reproduced with last-layer Z. |

All final tables are in [`results/`](results) (built from `repro/diag/*.json`). Post-hoc analyses (cause analysis,
scoring variants, visual-token counts) are in [REPORT, Section 4](REPORT.md#4-beyond-the-paper) and are not part of the verdicts.

## What is in this repository

```
REPORT.md                  reproducibility report (MLRC style): scope, method, results, beyond the paper, discussion
results/                   final tables (CSV + summary.json), built by scripts/make_results.py
figures/                   paired_differences.svg (drawn from results/ by scripts/make_figures.py), method_overview.svg
Makefile                   make fetch / check / report / test (CPU only)
CITATION.cff               how to cite this repository and the original paper
repro/                     implementation (lean) and analysis
  run.py, visref/          generation engine, ST/TSR/VisRef, DPP selection, scoring (scorer v4)
  kaggle/build.py          builds a self-contained Kaggle notebook for one session (2×T4); kaggle_ctl.py push/status/pull
  kaggle/build_step1_branch.py   same for the step1_branch follow-up
  colab/build_colab.py     same for one Colab GPU, resumable via Google Drive
  report.py, rescore.py    offline aggregation / re-scoring (no GPU)
  llm_extract.py, gemini_extract.py   LLM fallback answer extraction for rule-failed rows
  diag/                    pre-registered analyses (final_judgment.py, analyze_fill.py, step1_branch.py,
                           analyze_step1_branch.py, analyze_hs.py) and post-hoc diagnostics, with their JSON outputs
  audit/                   independent reviews and audits
  kaggle/runs/<session>/   every session: run_config.json (source sha256, ids, args), env.json, logs, raw outputs
                           (b_main_s1–s9, b_main_fill, b_main_rescore_v2/v3, diag_st_plain, hidden_rank_check,
                            step1_branch_pilot, step1_branch_main, b_main_hs)
  llm/final, llm/fill      LLM fallback extraction: requests, responses, merged rows (read by diag/*.py)
  reports/                 descriptive per-method report (rule-based scorer v4 only)
docs/
  SPEC_LEDGER.md           every implementation decision with its evidence grade (paper / cited code / official
                           model docs / our choice) — start at "핵심 판단 요약"
  EXPERIMENT_FLOW.md       chronological log: pre-registrations, decisions, results
  SOURCES.md               (English) how the logs' citations map to public sources: arXiv versions, pages, tables
  AI_USE_REFLECTION.md     (Korean) where using AI went wrong in this project, and how to prevent it
references/SOURCES.json    third-party files (MathVista data & eval prompt, cited code, VLMEvalKit) with URL + sha256
scripts/                   fetch_sources.py (downloads and verifies them), make_results.py, make_figures.py,
                           check_readme_numbers.py
```
The documents in `docs/` are written in Korean; `REPORT.md` points to the relevant entries.

## Reproduce

### 1. Rebuild every table and the results figure from the saved outputs (CPU only)
Needs Python 3.10 or newer (standard library only; `make check` passes on 3.10, 3.11, 3.12 and 3.13), `make` and a
git checkout. No GPU and no PyTorch.
```bash
make fetch     # MathVista testmini.json, official extraction prompt and cited code, sha256-checked
make check     # re-runs the pre-registered analyses, results/ and the results figure, checks the README numbers;
               # fails on any difference
make report    # optional: descriptive per-method report (rule-based scorer v4, no LLM fallback) in build/
```
The unit tests need PyTorch, transformers and pytest (a CPU build is enough):
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt pytest
make test
```
The individual commands are listed in the `Makefile`. `repro/reports/b_main_report.json` is rule-based only, which is
why its TSR accuracy (60.0) differs from the main scoring (60.1, one row recovered by the LLM fallback).

### 2. Regenerate the model outputs (GPU)
Pinned: model `OpenGVLab/InternVL3_5-1B` @ `2f71cf52542334823e48a46ffba0e2bc9add3446`,
data `AI4Math/MathVista` @ `2b6ad69445fbb5695c9b165475e8decdbeb97747`, `transformers==4.55.0`, fp16.
Each session's exact settings are in `repro/kaggle/runs/<session>/run_config.json`. The main run took 9 Kaggle
sessions on 2×T4, about 14.9 session-hours in total ([REPORT, Section 2.5](REPORT.md#25-code-versions-and-compute)).
Use new run names: the folders under `repro/kaggle/runs/` are the records of the published sessions, and the
builders refuse to overwrite an existing run folder (`--overwrite` exists only for rebuilding a session that has not
been pushed yet).
```bash
# Kaggle (2×T4): put your kaggle.json in the repo root (never commit it)
python3 repro/kaggle/build.py rerun_b_main_s1 --timeout 7200 \
  --run-args="--methods st,tsr,visref_raw_rank --think-cap 1024 --total-think-cap 10240"
python3 repro/kaggle/kaggle_ctl.py push rerun_b_main_s1      # then: status / pull
# later sessions skip what your own sessions finished:
#   --skip-from repro/kaggle/runs/rerun_b_main_s*/output/results/results_*.jsonl

# step1_branch follow-up (pilot of 20 questions, then the rest)
cd repro && python3 kaggle/build_step1_branch.py rerun_step1_branch_pilot --limit 20 --timeout 3600
python3 kaggle/build_step1_branch.py rerun_step1_branch_main --timeout 7200 \
  --skip-from kaggle/runs/rerun_step1_branch_pilot/output/results/step1_branch_*.jsonl
```
Questions are processed in a fixed seed-0 shuffled order, so any completed prefix is a random subset; every
(question, method) uses a per-question seed (common random numbers across methods). Outputs depend slightly on batch
composition under fp16 ([REPORT, Section 6](REPORT.md#6-limitations)).

### 3. LLM fallback extraction (rows the rule scorer cannot parse, about 1%)
```bash
cd repro
python3 llm_extract.py prepare <results...> --out requests.jsonl --control 20
GEMINI_API_KEY=... python3 gemini_extract.py requests.jsonl --out responses.jsonl --model gemini-3.5-flash-lite --sleep 6
python3 llm_extract.py merge <results...> --responses responses.jsonl --out rescored.jsonl
```
Unless a response is already a bare answer, the official MathVista evaluation uses an LLM (GPT-4 in the MathVista
paper) to pull the short final answer out of it. Its prompt shows five worked examples (question, model response, extracted answer) and then the new
response (`evaluation/prompts/ext_ans.py` in the MathVista repository, pinned in `references/SOURCES.json`). We send
the same prompt to gemini-3.5-flash-lite, only for the rows the rules cannot parse. The LLM's answer counts only if it
appears in the response text, so the LLM cannot solve the question itself, and empty responses are never sent (see
`docs/EXPERIMENT_FLOW.md`).

## Settings we could not find in the paper
We could not find these in the paper: the sampling values, token limits, what to do
with truncated thinking or unfinished answers, the answer-extraction method, the layer of the vectors used for the
thinking text, what to do when the DPP kernel rank is below the token budget m, how the stopping entropy is estimated,
the injection format (tags, separators, `</think>` handling, position ids) and the image tiling. We filled each gap
from the strongest source available, in this order: the paper's figures and tables → works and code the paper cites →
official InternVL3.5 material (the [official model page](https://huggingface.co/OpenGVLab/InternVL3_5-1B) on Hugging Face, where the authors give usage and recommended settings, and VLMEvalKit, which the InternVL3.5 report used) → public code the
paper does not cite → our own choice.

| setting | our choice | basis | other possibilities (status) |
|---|---|---|---|
| Sampling | T 0.6, top-k 50, top-p 0.95 (thinking mode, which the paper uses, is turned on with the official model page's system prompt) | not found in the paper; the official model page recommends sampling with T 0.6 in thinking mode to avoid repetition | – |
| Token limits | 1,024 per thinking step, 10,240 in total, 1,024 for the answer | 1,024 is the official model page's example value; the rest is our choice | a larger step cap (e.g. 4,096) or the official evaluation's 65,536 new tokens; not run |
| Thinking cut at the limit | close the thinking and append `Final Answer:` (`</think>\nFinal Answer:`), so the model answers at once | the method of s1 (Muennighoff et al. 2025, "budget forcing"), which the paper cites; s1's official code appends exactly `\nFinal Answer:`; using it in VisRef is our interpretation | a diagnostic run without `Final Answer:` (ST, 150 questions) changed accuracy by −2.0 pp; 150 questions are too few to tell this from no change (95% range about −9.5 to +5.5 pp) |
| Answer that does not finish | force a final answer: append `\n**Final Answer**\n\boxed` and allow up to 100 more tokens | answers often restarted thinking ("Wait, wait …") and ran to the 1,024-token limit; we found no rule for this in the paper or its cited works, and the public code of DEER (Yang et al. 2025, an early-exit method for reasoning models; not cited by the paper) has a rule for exactly this case | count such answers wrong ([REPORT, Section 3.2](REPORT.md#32-sensitivity-to-answer-handling)) |
| Answer extraction | rules first (boxed answer, "answer is", last line); the official MathVista extraction prompt with an LLM only for rows the rules cannot parse | the official MathVista code handles only bare answers by rule (a choice or a number as the whole response) and sends every other response to an LLM (GPT); VLMEvalKit also tries a rule first, then GPT | the LLM for every response that is not a bare answer, with the official GPT model; not run |
| Text that is scored | the text after the last `</think>` | the same split rule as VLMEvalKit (whether the InternVL3.5 evaluation turned it on: not confirmed) | score the whole output; not tested |
| DPP selection size | the greedy DPP stops at the kernel rank, about m/5 tokens | the cited DPP code (Chen et al. 2018); conflicts with the paper's \|V_k\| = m | top up to m with tokens chosen without regard to relevance: no extra gain (`visref_fill`); m relevant tokens: not tested |
| Vectors for the thinking text (Z) | input embeddings: the vectors right after the words are turned into numbers | our choice; the paper says "text token embeddings" without naming a layer | a deeper layer (e.g. the last): the DPP could then return m tokens for long enough steps, as the paper states; the last layer: follow-up `visref_hs` (analysis in progress) |
| Stopping entropy H | one sampled answer (see below) | definition from the paper; the way to estimate it is ours | several sampled answers, entropy over distinct answer values, base-2 logarithm (the same δ = 0.25 would then be stricter); not used in the final runs |
| Injection format | selected visual embeddings appended right before `</think>`, without the `<img>` … `</img>` tags InternVL uses around an image; order z_k → Ṽ_k → H_k → "Wait, think more" | order from the paper (Algorithm 1); format our choice | with the tags: dropped after a small early test (see above); after `</think>`, other position ids: not tested |
| Image tiling | InternVL's standard input: the image is cut into up to 12 tiles of 448×448 to fit its shape, plus one shrunken copy of the whole image; each tile becomes 256 visual tokens (mean 1,961.5 per image) | the official model page's example code (`load_image(max_num=12)`, shrunken copy on) and the model's configuration file (`max_dynamic_patch` 12, `use_thumbnail` true) | without the shrunken copy the mean is 1,755.6, closer to the paper's ∼1,772 (8B); not run |
| Hardware | Kaggle T4, fp16 | free GPUs | paper: four A10G GPUs, PyTorch 2.7.0, Transformers 4.55.0; not tested (on the same GPU, fp16 noise changed 2.2% of answers) |

**The stopping entropy H.** The paper defines H as the entropy of the model's answer distribution (Algorithm 1,
line 12), and the study it cites for this criterion (Ghosal et al.) samples several answers to measure it. H has to be measured after each thinking step, before any final answer exists, because it decides whether to keep
thinking; so we ask the model to answer at that point. We let the model write one short answer after `</think>` +
`\n\n\boxed` (at most 16 tokens). This is the model's own answer, not the correct one, and it is sampled at T = 1
rather than taken greedily, because the entropy is an average over the answers the model might give. At each token of that answer the model has a probability distribution over
the next token; we take the entropy of each distribution (natural logarithm) and add them up. On average this sum
equals the entropy of the answer distribution (chain rule), so it measures the same quantity as the paper, but with one
sample it is noisy, and the noise affects when a method stops. History: we started with one sampled answer and the
average of its token entropies. After reading Ghosal et al. we also tried 10 complete answers per step; with the 8B
model, a TSR run on 3 development questions took 4,703.9 GPU-seconds. We went back to one sample because, judging from
the paper's latency table (Table 5: ST 7.1 s, TSR 7.7 s, VisRef 8.2 s per question, 8B model on one H100), the
authors cannot have generated several answers per step. Finally we replaced the average by the sum, because the
average is not the entropy of the answer (decision log: `docs/EXPERIMENT_FLOW.md`, 2026-10-05 to 2026-10-07). An
`Answer:` probe was also replaced, because its formatting tokens dominated H and TSR stopped by entropy in 0 of 18
smoke-test questions.

Every decision with its evidence and alternatives is in `docs/SPEC_LEDGER.md` (start at "핵심 판단 요약"); where each
paper claim comes from (arXiv version, page, table) is in `docs/SOURCES.md`. Questions for the authors are listed in
[REPORT, Section 5.3](REPORT.md#53-communication-with-original-authors).

## Contributions and AI use
This reproduction was carried out by AhnJG with AI assistance (Claude models by Anthropic, Codex by OpenAI).

- **AhnJG:** chose the target and the evidence rules (every external number is cited with file:line or marked
  unverified); made the design decisions recorded as "사용자 결정" in `docs/EXPERIMENT_FLOW.md`, among them injecting
  only the DPP's rank (`visref_raw_rank`), the answer handling (s1 prompt + DEER answer forcing), rule-based scoring
  with an LLM fallback, fixing the main comparisons and tests before the final analysis, the follow-up experiments and
  their scope, and when to stop; raised the questions behind several recorded corrections.
- **Claude (AI assistant):** wrote most of the code (generation engine, methods, DPP selection, scoring, analysis
  scripts, tests); drafted the pre-registrations, analyses, REPORT.md and the figures; separate model sessions carried
  out independent reviews (`repro/audit/`).
- **Codex (OpenAI):** used throughout the project to review the content.
- **An LLM as part of the method:** gemini-3.5-flash-lite extracts the answer for the rows the rules cannot parse
  (see [LLM fallback extraction](#3-llm-fallback-extraction-rows-the-rule-scorer-cannot-parse-about-1)).

## Citation
See `CITATION.cff`. Please also cite the original paper:
```bibtex
@article{ghosal2026visref,
  title   = {VisRef: Visual Refocusing while Thinking Improves Test-Time Scaling in Multi-Modal Large Reasoning Models},
  author  = {Ghosal, Soumya Suvra and Kim, Youngeun and Li, Zhuowei and Chaudhry, Ritwick and Xu, Linghan and
             Zhang, Hongjing and Zablocki, Jakub and Xing, Yifan and Zhang, Qin},
  journal = {arXiv preprint arXiv:2603.00207},
  year    = {2026}
}
```

## License
- Code: MIT (`LICENSE`).
- Data files contain material derived from MathVista, which is distributed under CC BY-SA 4.0: the run outputs
  (`repro/**/results_*.jsonl`, `repro/**/step1_branch_*.jsonl`), the LLM-extraction requests and responses in
  `repro/llm/` (questions, answers and the official MathVista extraction prompt), the analysis JSON in `repro/diag/`,
  `results/` and the reports. These data files are therefore shared under CC BY-SA 4.0. Model outputs were produced
  with InternVL3.5-1B (Apache-2.0).
- Other third-party files (MathVista data, cited code) are not included; `scripts/fetch_sources.py` downloads them from
  their original locations.
