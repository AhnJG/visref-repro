# Reproducing visref-repro

How to rebuild the published numbers and regenerate the model outputs. Results and their reading: [REPORT.md](../REPORT.md).

## Reproduce

### 1. Rebuild every table and the results figure from the saved outputs (CPU only)
Needs Python 3.10 or newer (standard library only; `make check` passes on 3.11, 3.12 and 3.13; the README number check also ran on 3.10), `make` and a
git checkout. No GPU and no PyTorch.
```bash
make fetch     # MathVista testmini.json, official extraction prompt and cited code, sha256-checked
make check     # re-runs the analyses behind the verdicts, results/ and the results figure, and checks that the tables in
               # README.md, README.ko.md and REPORT.md match results/; fails on any difference
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
sessions on 2×T4, about 14.9 session-hours in total ([REPORT, Section 2.5](../REPORT.md#25-code-versions-and-compute)).
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
composition under fp16 ([REPORT, Section 6](../REPORT.md#6-limitations)).

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

## Repository layout

```
REPORT.md                  reproducibility report (MLRC style): scope, method, results, beyond the paper, discussion
results/                   final tables (CSV + summary.json), built by scripts/make_results.py
figures/                   paired_differences.svg (drawn from results/) and main_run_steps.svg (drawn from the main run
                           records) by scripts/make_figures.py; method_overview.svg, code_map.svg (both drawn by hand)
Makefile                   make fetch / check / report / test (CPU only)
CITATION.cff               how to cite this repository and the original paper
repro/                     implementation (lean) and analysis
  run.py, visref/          generation engine, ST/TSR/VisRef, DPP selection, scoring (scorer v4)
  kaggle/build.py          builds a self-contained Kaggle notebook for one session (2×T4); kaggle_ctl.py push/status/pull
  kaggle/build_step1_branch.py   same for the step1_branch follow-up
  colab/build_colab.py     same for one Colab GPU, resumable via Google Drive
  report.py, rescore.py    offline aggregation / re-scoring (no GPU)
  llm_extract.py, gemini_extract.py   LLM fallback answer extraction for rule-failed rows
  runpod/run_imgtag.py     runner for the image-tag run (A6) on a rented GPU
  diag/                    analyses with fixed decision rules (final_judgment.py: fixed before the final analysis;
                           analyze_fill.py, step1_branch.py, analyze_step1_branch.py, analyze_hs.py, analyze_think.py,
                           analyze_imgtag*.py: rules fixed before the results) and post-hoc diagnostics, with their
                           JSON outputs; step1_store_bmain.json is the main run's stored first step (made by
                           make_step1_store.py) for the thinking experiments
  kaggle/runs/<session>/   every session: run_config.json (source sha256, ids, args), env.json, logs, raw outputs
                           (b_main_s1–s9, b_main_fill, diag_st_plain, hidden_rank_check, step1_branch_pilot,
                            step1_branch_main, b_main_hs; bundle_pilot and bundle_pilot2 = two 10-question trials of
                            the image-tag variant, B7; imgtag_check and imgtag_main = A6; setting checks: smoke, smoke2, smoke3, main_s0,
                            full_s1, b_test)
  llm/final, llm/fill, llm/hs   LLM fallback extraction: requests, responses, merged rows (read by diag/*.py)
  reports/                 descriptive per-method report (rule-based scorer v4 only)
docs/
  REPRODUCE.md             (English) this file
  IMPLEMENTATION_CHOICES.md  every implementation decision with its evidence grade (paper / cited code / official
                           model docs / our choice) — start with its summary table
  EXPERIMENT_FLOW.md       chronological log: decision rules fixed before results, decisions, results
  SOURCES.md               (English) how the logs' citations map to public sources: arXiv versions, pages, tables
references/SOURCES.json    third-party files (MathVista data & eval prompt, cited code, VLMEvalKit) with URL + sha256
scripts/                   fetch_sources.py (downloads and verifies them), make_results.py, make_figures.py,
                           check_readme_numbers.py
```
`EXPERIMENT_FLOW.md` and `IMPLEMENTATION_CHOICES.md` are written in Korean; `REPORT.md` points to the relevant entries.
