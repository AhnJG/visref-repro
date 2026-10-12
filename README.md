# visref-repro

Unofficial reproduction of **VisRef** (Ghosal et al., *VisRef: Visual Refocusing while Thinking Improves Test-Time
Scaling in Multi-Modal Large Reasoning Models*, arXiv:2603.00207, CVPR 2026), Appendix Table 6 row *InternVL-1B*:
InternVL3.5-1B on MathVista *testmini* (1,000 questions), comparing

| method | what it does | paper (Table 6) |
|---|---|---|
| **ST** – standard thinking | one reasoning trace, then the answer | 46.1 |
| **TSR** – textual self-reflection | "Wait, think more" steps until the answer entropy is below δ | 48.5 |
| **VisRef** | TSR + re-injecting a DPP-selected subset of the image's visual tokens after every step | 52.0 |

![ST, TSR and VisRef as run in this reproduction; orange notes mark settings the paper leaves open](figures/method_overview.svg)

> **Status (2026-10-12): complete.** The main run and two pre-registered follow-ups are finished and analysed, and the
> experiments are closed. One auxiliary run (`st_official`) stopped early on free Colab and is not analysed. The
> original authors have not been contacted yet. Full write-up: **[REPORT.md](REPORT.md)**.

## Results

Reading (a), the pre-specified one: rule-based answer extraction plus a grounded LLM fallback for the rows the rules
cannot parse. n = 1,000; each method once per question, with shared seeds and a shared first thinking step.

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

- **In short:** TSR's gain over ST matches the paper. We did not confirm an additional gain of VisRef over TSR in our
  InternVL3.5-1B implementation, and VisRef − ST is not significant after the pre-specified Holm correction.
- *Consistent* = the paired bootstrap 95% CI (2,000 resamples, seed 0) excludes 0 and contains the paper's delta (not a
  proof of equivalence); *not reproduced* = the CI excludes the paper's delta. The tests were fixed before the final
  analysis, the CI classification rule after an interim look at 759 questions, so the classification is not
  confirmatory (REPORT §2.4). It is the same under every answer-handling reading we tried (REPORT §3.2).
- Our absolute accuracies are 8–12 pp above the paper's; our ST is close to the official InternVL3.5-1B MathVista
  score (59.3). The cause of the gap was not isolated.
- We do **not** conclude that VisRef has no effect: our implementation injects about one fifth of the paper's token
  budget m (a confirmed procedural difference), and several settings are not specified in the paper
  (see [Known differences](#known-differences-from-the-paper)).

### Pre-registered follow-ups

| experiment | question | result |
|---|---|---|
| `visref_fill` (976 q) | Does injecting exactly m tokens (DPP filled up to m with a ridge term) change VisRef? | fill − TSR 0.0 pp (−2.15, +2.36); fill − VisRef +0.1 pp (−2.15, +2.36): "too few injected tokens" not supported. The filled tail was not ordered by relevance, so top-m-by-relevance remains untested. |
| `step1_branch` (309 q) | On the questions where TSR or VisRef continued after step 1: does re-injecting Ṽ₁ help an immediate answer (B − A), or an answer after ≤ 512 more thinking tokens (D − C)? | B − A +1.40 pp (−1.19, +3.99); D − C +0.97 pp (−2.27, +4.21); Holm p 0.578 for both: the immediate-answer gain seen post hoc in the main run was not reproduced. |

All final tables are in [`results/`](results) (built from `repro/diag/*.json`). Post-hoc analyses (cause analysis,
answer-handling readings, visual-token counts) are in REPORT §4 and are not part of the verdicts.

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
                           analyze_step1_branch.py) and post-hoc diagnostics, with their JSON outputs
  audit/                   independent reviews and audits
  kaggle/runs/<session>/   every session: run_config.json (source sha256, ids, args), env.json, logs, raw outputs
                           (b_main_s1–s9, b_main_fill, b_main_rescore_v2/v3, diag_st_plain, hidden_rank_check,
                            step1_branch_pilot, step1_branch_main)
  llm/final, llm/fill      LLM fallback extraction: requests, responses, merged rows (read by diag/*.py)
  reports/                 descriptive per-method report (rule-based scorer v4 only)
docs/
  SPEC_LEDGER.md           every implementation decision with its evidence grade (paper / cited code / official
                           model docs / our choice) — start at "핵심 판단 요약"
  EXPERIMENT_FLOW.md       chronological log: pre-registrations, decisions, results
references/SOURCES.json    third-party files (MathVista data & eval prompt, cited code, VLMEvalKit) with URL + sha256
scripts/                   fetch_sources.py (downloads and verifies them), make_results.py, make_figures.py
```
The documents in `docs/` are written in Korean; `REPORT.md` points to the relevant entries.

## Reproduce

### 1. Rebuild every table from the saved outputs (CPU only)
```bash
pip install -r requirements.txt   # the offline steps need only the standard library + the fetched files
make fetch     # MathVista testmini.json, official extraction prompt and cited code, sha256-checked
make check     # re-runs the pre-registered analyses, results/ and the results figure; fails on any difference
make report    # optional: descriptive per-method report (rule-based scorer v4, no LLM fallback) in build/
make test      # CPU unit tests; need torch + transformers (a CPU build is enough)
```
The individual commands are listed in the `Makefile`. `repro/reports/b_main_report.json` is rule-based only, which is
why its TSR accuracy (60.0) differs from reading (a) (60.1, one row recovered by the LLM fallback).

### 2. Regenerate the model outputs (GPU)
Pinned: model `OpenGVLab/InternVL3_5-1B` @ `2f71cf52542334823e48a46ffba0e2bc9add3446`,
data `AI4Math/MathVista` @ `2b6ad69445fbb5695c9b165475e8decdbeb97747`, `transformers==4.55.0`, fp16.
Each session's exact settings are in `repro/kaggle/runs/<session>/run_config.json`.
```bash
# Kaggle (2×T4): put your kaggle.json in the repo root (never commit it)
python3 repro/kaggle/build.py b_main_s1 --timeout 7200 \
  --run-args="--methods st,tsr,visref_raw_rank --think-cap 1024 --total-think-cap 10240"
python3 repro/kaggle/kaggle_ctl.py push b_main_s1      # then: status / pull
# next sessions skip what is done:  --skip-from repro/kaggle/runs/b_main_s*/output/results/results_*.jsonl

# step1_branch follow-up (pilot of 20 questions, then the rest)
cd repro && python3 kaggle/build_step1_branch.py step1_branch_pilot --limit 20 --timeout 3600
python3 kaggle/build_step1_branch.py step1_branch_main --timeout 7200 \
  --skip-from kaggle/runs/step1_branch_pilot/output/results/step1_branch_*.jsonl
```
Questions are processed in a fixed seed-0 shuffled order, so any completed prefix is a random subset; every
(question, method) uses a per-question seed (common random numbers across methods). Outputs depend slightly on batch
composition under fp16 (REPORT §6).

### 3. LLM fallback extraction (rows the rule scorer cannot parse, about 1%)
```bash
cd repro
python3 llm_extract.py prepare <results...> --out requests.jsonl --control 20
GEMINI_API_KEY=... python3 gemini_extract.py requests.jsonl --out responses.jsonl --model gemini-3.5-flash-lite --sleep 6
python3 llm_extract.py merge <results...> --responses responses.jsonl --out rescored.jsonl
```
Uses the official MathVista extraction prompt; an LLM extraction is accepted only if it is grounded in the response
text, and empty responses are never sent (see `docs/EXPERIMENT_FLOW.md`).

## Known differences from the paper
The paper does not specify token limits, the handling of truncated thinking or capped answers, the answer-extraction
method, the layer of the reasoning-token representation, the handling of a DPP kernel rank below the token budget m,
the estimator of the stopping entropy, or the injection format (tags, separators, `</think>` handling, position ids).
Our choices and their evidence are in `docs/SPEC_LEDGER.md` ("핵심 판단 요약"). The most consequential ones:
- thinking capped at 1,024 tokens per step (10,240 total), answer 1,024; truncated thinking is answered after
  `</think>\nFinal Answer:` (s1-style budget forcing); answers hitting the cap get DEER-style answer forcing;
- VisRef injects only the tokens the greedy DPP selects (it stops at the kernel rank, as in the cited Chen et al.
  code), about one fifth of m = ⌊0.3·|V|⌋ when the reasoning tokens are input embeddings; the paper states |V_k| = m;
- the selected visual embeddings are appended without image tags right before `</think>`;
- the stopping entropy H is the summed token entropy of one sampled answer (the paper defines it on the answer
  distribution);
- images use InternVL dynamic tiling with max_num 12 plus a thumbnail tile (mean 1,961.5 visual tokens; the paper
  reports ∼1,772 for the 8B model, close to our 1,755.6 without the thumbnail);
- Kaggle T4 GPUs with fp16 (paper: four A10G GPUs, PyTorch 2.7.0, Transformers 4.55.0).

Questions for the authors are listed in REPORT §5.3.

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
- `repro/**/results_*.jsonl`, `repro/**/step1_branch_*.jsonl`, `results/` and reports contain fields derived from
  MathVista (questions' answers), which is distributed under CC BY-SA 4.0; these data files are therefore shared under
  CC BY-SA 4.0. Model outputs were produced with InternVL3.5-1B (Apache-2.0).
- Third-party files are not included; `scripts/fetch_sources.py` downloads them from their original locations.
