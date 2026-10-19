# CPU-only targets, run from the repository root. GPU generation: see docs/REPRODUCE.md.
SHELL := /bin/bash
PY ?= python3
STEP1 := kaggle/runs/step1_branch_pilot/output/results/step1_branch_*.jsonl \
         kaggle/runs/step1_branch_main/output/results/step1_branch_*.jsonl

.PHONY: fetch analysis results figures check report test

# Download the third-party files (MathVista testmini.json, official extraction prompt, cited code); sha256-checked.
fetch:
	$(PY) scripts/fetch_sources.py

# Re-run the pre-registered analyses from the saved outputs (needs `make fetch`).
# final_judgment / analyze_fill / analyze_hs / analyze_imgtag_main (A6) / source_cluster_main (A1, sources resampled) / format_compare_200 (main run on the B8 questions) / analyze_imgtag (two 10-question trials) rewrite repro/diag/*.json in place; step1_branch is written to build/ and compared.
analysis:
	mkdir -p build
	cd repro && $(PY) diag/final_judgment.py > /dev/null
	cd repro && $(PY) diag/analyze_fill.py llm/fill/merged.jsonl > /dev/null
	cd repro && $(PY) diag/analyze_hs.py analyze --llm llm/hs/merged.jsonl --out diag/analyze_hs.json > /dev/null
	cd repro && $(PY) diag/analyze_imgtag_main.py analyze --llm llm/imgtag_main/merged.jsonl --out diag/analyze_imgtag_main.json > /dev/null
	cd repro && $(PY) diag/source_cluster_main.py > /dev/null
	cd repro && $(PY) diag/format_compare_200.py > /dev/null
	cd repro && $(PY) diag/tsr_wrong_pooled.py > /dev/null
	cd repro && $(PY) diag/analyze_inject_sensitivity.py kaggle/runs/inject_sens_main/output/results/inject_sensitivity_0.jsonl \
	  kaggle/runs/inject_sens_main/output/results/inject_sensitivity_1.jsonl --out diag/inject_sensitivity_analysis.json > /dev/null
	rm -f build/full_think_analysis_main.json
	cd repro && $(PY) diag/analyze_full_think.py 'kaggle/runs/full_think_main/output/results/full_think_*.jsonl' \
	  --ids diag/full_think_ids.json --set cap_continued --out ../build/full_think_analysis_main.json > /dev/null
	cmp build/full_think_analysis_main.json repro/diag/full_think_analysis_main.json
	rm -f build/step1_branch_analysis.json
	cd repro && $(PY) diag/analyze_step1_branch.py $(STEP1) --ids diag/step1_branch_ids.json \
	  --out ../build/step1_branch_analysis.json > /dev/null
	cmp build/step1_branch_analysis.json repro/diag/step1_branch_analysis.json
	cd repro && for n in 1 2; do r=bundle_pilot; [ $$n = 2 ] && r=bundle_pilot2; \
	  $(PY) diag/analyze_imgtag.py kaggle/runs/$$r/output/results/results_0.jsonl kaggle/runs/$$r/output/results/results_1.jsonl \
	  --out diag/analyze_imgtag_pilot$$n.json > /dev/null 2>&1; done

# Rebuild the final tables in results/ from repro/diag/*.json.
results:
	$(PY) scripts/make_results.py
	$(PY) scripts/make_compute.py

# Redraw figures/paired_differences.svg from results/ and figures/main_run_steps.svg from the main run records (figures/method_overview.svg is drawn by hand).
figures: results
	$(PY) scripts/make_figures.py

# Everything regenerated must equal the committed files (needs a git checkout),
# and the result tables in README.md, README.ko.md and REPORT.md must match results/ cell by cell.
check: analysis results figures
	git diff --exit-code -- repro/diag results figures
	$(PY) scripts/check_readme_numbers.py
	@echo "OK: the analyses, results/ and figures/ reproduce the committed files; the tables in the READMEs and REPORT match results/"

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
