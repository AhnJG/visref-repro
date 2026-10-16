# visref-repro

[한국어 (Korean)](README.ko.md)

Unofficial reproduction of **VisRef** (Ghosal et al., *VisRef: Visual Refocusing while Thinking Improves Test-Time
Scaling in Multi-Modal Large Reasoning Models*, arXiv:2603.00207, CVPR 2026), Appendix Table 6 row *InternVL-1B*:
InternVL3.5-1B on the 1,000 MathVista *testmini* questions. Three methods are compared:

- **ST** (standard thinking): the model thinks once, then answers.
- **TSR** (textual self-reflection): after each thinking step, "Wait, think more" makes the model think again, until
  its answer is certain enough.
- **VisRef**: like TSR, but before each new step a selected part of the image is inserted again.

![ST, TSR and VisRef as run in this reproduction; orange notes mark settings we could not find in the paper](figures/method_overview.svg)

> **Status (2026-10-14):** the main run and three follow-up experiments are finished; one set of experiments is
> pre-registered but not yet run. Full write-up: **[REPORT.md](REPORT.md)**.

## How to read this repository

| | When | Read |
|---|---|---|
| 1 | First: results and main caveats | this README |
| 2 | What was tested and what was concluded | [REPORT.md](REPORT.md): Summary and Section 1.1 (every experiment in one table), then Sections 3–4 (results, causes) and 6 (limitations) |
| 3 | How a setting the paper does not state was chosen | REPORT Sections 2.1 and 2.6; the evidence for each decision: [docs/IMPLEMENTATION_CHOICES.md](docs/IMPLEMENTATION_CHOICES.md) (Korean; start with its summary table) |
| 4 | When and why something was decided | [docs/EXPERIMENT_FLOW.md](docs/EXPERIMENT_FLOW.md) (Korean dated log; start with its stage summary) |
| 5 | Checking a number | [`results/`](results) for the tables, `repro/kaggle/runs/` for the raw outputs, `make check` (see Reproduce) |
| 6 | Following a citation such as `LAY L900` to the paper | [docs/SOURCES.md](docs/SOURCES.md) |
| 7 | Running it again or reading the code | [docs/REPRODUCE.md](docs/REPRODUCE.md) and the code map under Reproduce |

Every file location: [REPORT, Section 7](REPORT.md#7-where-to-find-things).

## Results

n = 1,000 questions; each method once per question, with shared seeds and a shared first thinking step. Answers are
extracted by rules, and by an LLM only where the rules fail (36 of 3,000 rows); the LLM's answer counts only if it
appears in the response.

![Paired accuracy differences with 95% CIs next to the paper's differences](figures/paired_differences.svg)

<!-- check:accuracy -->
| accuracy (%) | ST | TSR | VisRef |
|---|---|---|---|
| paper (Table 6) | 46.1 | 48.5 | 52.0 |
| **ours** | **57.8** | **60.1** | **60.0** |
<!-- /check -->

<!-- check:paired -->
| comparison | ours, pp (95% CI) | paper, pp | verdict |
|---|---|---|---|
| VisRef − ST (main) | +2.2 (+0.3, +4.1) | +5.9 | effect size not reproduced (same direction, different size) |
| VisRef − TSR (main) | −0.1 (−2.0, +1.7) | +3.5 | effect size not reproduced |
| TSR − ST (supplementary) | +2.3 (+0.6, +4.0) | +2.4 | consistent with the paper value |
<!-- /check -->

- **In short:** TSR's gain over ST matches the paper. We did not confirm VisRef's additional gain over TSR in our
  InternVL3.5-1B implementation. VisRef − ST is smaller than in the paper and not significant after the correction for
  two comparisons fixed in advance (Holm p = 0.067).
- **Verdict:** *consistent* = our 95% CI contains the paper's difference and excludes 0; *not reproduced* = our 95% CI
  excludes the paper's difference. The rules were fixed after interim looks at 392 and 759 questions, so the main run
  is not a strict pre-registration ([REPORT, Section 2.4](REPORT.md#24-timing-of-pre-registration)).

## Experiments at a glance

**Result experiments** test the paper's claim or look for the reason it was not reproduced (decision rules fixed
before the results). **Setting checks** chose the settings the paper does not state; how each one was chosen, and
whether it comes from the paper, a cited work, official InternVL material or our own choice, is in
[REPORT, Section 2.6](REPORT.md#26-setting-checks-how-the-unstated-settings-were-chosen).

<!-- check:glance -->
| | Question | Questions | Result | Main limitation |
|---|---|---|---|---|
| A1 | Does VisRef beat TSR? (paper: +3.5 pp) | 1,000 | VisRef − TSR: −0.1 pp | only about one fifth of the paper's m image tokens were re-inserted |
| A2 | Does topping the re-inserted image tokens up to m help? | 976 | fill − TSR: 0.0 pp | the added tokens were nearly unrelated to the thinking |
| A3 | Does re-inserting the image after step 1 help by itself? | 309 | with − without the image: +1.40 pp, +0.97 pp | only questions where a method continued after step 1 |
| A4 | Does choosing image tokens with the model's last-layer vectors help? | 839 | hs − TSR: +0.24 pp | the first step was generated anew |
| A5 | Does thinking more after the re-inserted image help? | planned | not run yet | changes the paper's stopping behaviour |
<!-- /check -->

## Why our results may differ from the paper

We look for the cause in our implementation first ([REPORT, Section 4.1](REPORT.md#41-cause-analysis-post-hoc)):

- **Fewer image tokens.** We re-insert about one fifth of the paper's m image tokens per step, because the selection
  code the paper cites stops early (a confirmed difference; A2 and A4 test two ways around it).
- **Our own settings.** The paper does not give the stopping-entropy estimate, the insertion format, the token limits or
  the answer extraction; we chose them in small checks ([REPORT, Section 2.6](REPORT.md#26-setting-checks-how-the-unstated-settings-were-chosen)).
  Those checks used questions that are also among the 1,000 evaluated ones.
- **Higher accuracies.** Our accuracies are 8–12 pp above the paper's, close to the official InternVL3.5-1B score;
  the reason is not resolved ([REPORT, Section 4.4](REPORT.md#44-comparison-with-other-numbers-in-the-paper-post-hoc)).
- **One setting.** One model size, one benchmark, one seed.

## Reproduce

Rebuild every table and figure from the saved outputs (CPU only, Python 3.10 or newer, standard library):

```bash
make fetch     # MathVista testmini.json, official extraction prompt and cited code, sha256-checked
make check     # re-runs the analyses, rebuilds results/ and the figure, checks the tables in the READMEs and REPORT
```

Regenerating the model outputs on a GPU, the LLM extraction and the full repository layout:
[docs/REPRODUCE.md](docs/REPRODUCE.md).

![How the code runs: run.py loops over questions; model.py builds the input, methods.py runs each question and method with engine.py and dpp.py, scoring.py scores the answer](figures/code_map.svg)

## Contributions and AI use

Carried out by AhnJG with AI assistance (Claude models by Anthropic, Codex by OpenAI): AhnJG chose the target, the
evidence rules and the design decisions; Claude wrote most of the code and drafted the analyses and the report.
Details: [REPORT, Section 8](REPORT.md#8-contributions-and-ai-use).

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
Code: MIT (`LICENSE`). Data files derived from MathVista (run outputs, LLM-extraction requests and responses, analysis
JSON, `results/`, reports) are shared under CC BY-SA 4.0, like MathVista; model outputs were produced with
InternVL3.5-1B (Apache-2.0). Other third-party files are not included; `scripts/fetch_sources.py` downloads them.
