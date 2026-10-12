# CPU-only targets, run from the repository root. GPU generation: see README, "Regenerate the model outputs".
SHELL := /bin/bash
PY ?= python3
STEP1 := kaggle/runs/step1_branch_pilot/output/results/step1_branch_*.jsonl \
         kaggle/runs/step1_branch_main/output/results/step1_branch_*.jsonl

.PHONY: fetch analysis results figures check report test

# Download the third-party files (MathVista testmini.json, official extraction prompt, cited code); sha256-checked.
fetch:
	$(PY) scripts/fetch_sources.py

# Re-run the pre-registered analyses from the saved outputs (needs `make fetch`).
# final_judgment / analyze_fill rewrite repro/diag/*.json in place; step1_branch is written to build/ and compared.
analysis:
	mkdir -p build
	cd repro && $(PY) diag/final_judgment.py > /dev/null
	cd repro && $(PY) diag/analyze_fill.py llm/fill/merged.jsonl > /dev/null
	rm -f build/step1_branch_analysis.json
	cd repro && $(PY) diag/analyze_step1_branch.py $(STEP1) --ids diag/step1_branch_ids.json \
	  --out ../build/step1_branch_analysis.json > /dev/null
	cmp build/step1_branch_analysis.json repro/diag/step1_branch_analysis.json

# Rebuild the final tables in results/ from repro/diag/*.json.
results:
	$(PY) scripts/make_results.py

# Redraw figures/paired_differences.svg from results/ (figures/method_overview.svg is drawn by hand).
figures: results
	$(PY) scripts/make_figures.py

# Everything regenerated must equal the committed files (needs a git checkout).
check: analysis results figures
	git diff --exit-code -- repro/diag results figures
	@echo "OK: the analyses, results/ and figures/ reproduce the committed files"

# Descriptive per-method report (rule-based scorer v4 only, no LLM fallback) -> build/b_main_report.{html,json}.
# Equals repro/reports/b_main_report.json; the HTML differs only in its generation timestamp.
report:
	mkdir -p build
	cd repro && $(PY) report.py kaggle/runs/b_main_s*/output/results/results_{0,1}.jsonl \
	  --order kaggle/runs/b_main_s1/output/results/ids.json --shards 2 \
	  --rescore-data ../data/raw/mathvista_testmini.json --out ../build/b_main_report.html

# CPU unit tests (need torch + transformers; a CPU build is enough).
test:
	cd repro && $(PY) -m pytest -q
