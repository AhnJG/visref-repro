# visref-repro

Unofficial reproduction of **VisRef** (arXiv:2603.00207), Appendix Table 6 row *InternVL-1B*:
InternVL3.5-1B on MathVista *testmini* (1,000 questions), comparing

| method | what it does | paper (Table 6) |
|---|---|---|
| **ST** – standard thinking | one reasoning trace, then the answer | 46.1 |
| **TSR** – textual self-reflection | "Wait, think more" steps until the answer entropy is below δ | 48.5 |
| **VisRef** | TSR + re-injecting a DPP-selected subset of the image's visual tokens after every step | 52.0 |

> **Status:** the main run (`b_main`, 1,000 questions × ST/TSR/VisRef) is complete; the pre-registered reading is in
> `repro/diag/final_judgment.json`. Pre-registered follow-ups (`b_main_fill`, `step1_branch`) are logged in
> `docs/EXPERIMENT_FLOW.md`. The results table below is filled in at public release.

## Results

_To be filled at public release_ (pre-registered analysis: paired McNemar exact tests for VisRef−ST and VisRef−TSR with
Holm correction, paired bootstrap 95% CIs; see `docs/EXPERIMENT_FLOW.md`, entry "2026-10-10 결과 전 고정").

## What is in this repository

```
repro/                     implementation (lean, ~1.7k lines) and analysis
  run.py, visref/          generation engine, ST/TSR/VisRef, DPP selection, scoring
  kaggle/build.py          builds a self-contained Kaggle notebook for one session (2×T4)
  colab/build_colab.py     same for one Colab GPU, resumable via Google Drive
  report.py, rescore.py    offline aggregation / re-scoring (no GPU)
  llm_extract.py, gemini_extract.py   LLM fallback answer extraction for rule-failed rows
  diag/, audit/            pre-registered diagnostics (final_judgment.py, step1_branch.py, ...) and reviews/audits
  kaggle/runs/<session>/   every session: run_config.json (source sha256, ids, args), env.json, logs, results_*.jsonl
                           (b_main_s1–s9, b_main_fill, b_main_rescore_v2/v3 rows, step1_branch_pilot config)
  llm/final, llm/fill      LLM fallback extraction: requests, responses, merged rows (read by diag/*.py)
docs/
  SPEC_LEDGER.md           every implementation decision with its evidence grade (paper / cited code / official
                           model docs / our choice) — start at "핵심 판단 요약"
  EXPERIMENT_FLOW.md       chronological log: pre-registrations, decisions, results
references/SOURCES.json    third-party files (MathVista data & eval prompt, cited code, VLMEvalKit) with URL + sha256
scripts/fetch_sources.py   downloads and verifies them
```
The documents in `docs/` are written in Korean.

## Reproduce

### 1. Re-score and rebuild every table from the saved outputs (CPU only)
```bash
pip install -r requirements.txt          # the offline steps need only the standard library + the fetched files
python3 scripts/fetch_sources.py         # MathVista testmini.json + official extraction prompt, sha256-checked
cd repro
python3 report.py kaggle/runs/b_main_s*/output/results/results_{0,1}.jsonl \
  --order kaggle/runs/b_main_s1/output/results/ids.json --shards 2 \
  --rescore-data ../data/raw/mathvista_testmini.json --out reports/b_main_report.html
python3 -m pytest -q                     # CPU tests; need torch + transformers (CPU build is enough)
```

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
```
Questions are processed in a fixed seed-0 shuffled order, so any completed prefix is a random subset; every
(question, method) uses a per-question seed (common random numbers across methods).

### 3. LLM fallback extraction (rows the rule scorer cannot parse, ~2%)
```bash
cd repro
python3 llm_extract.py prepare <results...> --out requests.jsonl --control 20
GEMINI_API_KEY=... python3 gemini_extract.py requests.jsonl --out responses.jsonl --model gemini-3.5-flash-lite --sleep 6
python3 llm_extract.py merge <results...> --responses responses.jsonl --out rescored.jsonl
```
Uses the official MathVista extraction prompt; an LLM extraction is accepted only if it is grounded in the response
text, and empty responses are never sent (see `docs/EXPERIMENT_FLOW.md`).

## Known differences from the paper
The paper does not specify token limits, the handling of truncated thinking, the answer-extraction method, the layer of
the reasoning-token embeddings, or what happens when the DPP kernel rank is below the token budget m. Our choices and
their evidence are listed in `docs/SPEC_LEDGER.md` ("핵심 판단 요약" and "D 자료 대조"). The most consequential ones:
- thinking capped at 1,024 tokens per step (10,240 total), answer 1,024; truncated thinking is answered after
  `</think>\nFinal Answer:` (s1-style budget forcing), answers hitting the cap get DEER-style answer forcing;
- VisRef injects only the tokens the greedy DPP selects (stops at the kernel rank, as in the cited Chen et al. code),
  which is often fewer than m = ⌊0.3·|V|⌋ when the reasoning tokens are input embeddings;
- Kaggle T4 GPUs with fp16 (paper: A10G).

## License
- Code: MIT (`LICENSE`).
- `repro/**/results_*.jsonl` and reports contain fields derived from MathVista (questions' answers), which is
  distributed under CC BY-SA 4.0; these data files are therefore shared under CC BY-SA 4.0. Model outputs were produced
  with InternVL3.5-1B (Apache-2.0).
- Third-party files are not included; `scripts/fetch_sources.py` downloads them from their original locations.
