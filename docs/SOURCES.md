# External sources: how to follow the citations

The Korean logs (`docs/SPEC_LEDGER.md`, `docs/EXPERIMENT_FLOW.md`) cite the paper and related reports as line numbers
in plain-text extractions, for example `LAY L900` or `REF/InternVL3.5.txt L775`. Those extractions and the PDFs are
not redistributed here. This file maps the abbreviations to public sources and lists the page and table of the claims
the results depend on.

## Abbreviations and how to regenerate the line numbers

| Abbreviation in the logs | What it is | How to regenerate |
|---|---|---|
| `LAY` = `VisRef_layout.txt` | VisRef paper, layout text | `pdftotext -layout VisRef.pdf` (poppler 22.02.0) |
| `VisRef_plain.txt` | VisRef paper, plain text | `pdftotext VisRef.pdf` |
| `REF/InternVL3.5.txt` | InternVL3.5 report, layout text | `pdftotext -layout` of the report PDF |
| `REF/Ghosal_…`, `REF/MathVista.txt`, `REF/VLMEvalKit.txt` | other cited papers | same procedure (not re-checked byte for byte) |
| `SRC/…` | third-party code and the official MathVista prompt | `make fetch` (see `references/SOURCES.json`) |
| `SRC/internvl_README.md` | model card README, local copy collected 2026-10-07 | compare with the model card at the pinned revision below |

The VisRef and InternVL3.5 text files we cited are byte-identical to the output of the commands above on the PDFs
listed below (checked 2026-10-12). `pdftotext` separates pages with a form feed, so the page of a line is one plus the
number of form feeds before it.

| Paper | Version | sha256 of the PDF we used |
|---|---|---|
| VisRef (Ghosal et al.) | [arXiv:2603.00207v1](https://arxiv.org/abs/2603.00207v1) | `f51f8f31eb4cb9388e624644910c97e3ea522930849cd838595c453ac5f34be9` |
| InternVL3.5 report | [arXiv:2508.18265v2](https://arxiv.org/abs/2508.18265v2) | `7cec38fe73c626625a00c9c7ab1d17b24f2740c6a8d7a46ed66fc32c6f8b31ee` |
| Does Thinking More Always Help? (Ghosal et al.) | [arXiv:2506.04210v3](https://arxiv.org/abs/2506.04210v3) | `d52fe513bbbec639f257ed8fa39322976851620512814f2519d30d6ed3791214` |
| MathVista | [arXiv:2310.02255v3](https://arxiv.org/abs/2310.02255v3) | `ee5a616e9481f0f0d6b8ccc9e6d261f1c35fcb0b9c64d1c5c9d2d9462d6fc5e9` |
| VLMEvalKit | [arXiv:2407.11691v5](https://arxiv.org/abs/2407.11691v5) | `bee89b8fe7c0777435e16d048bd5331d6b8e46e6ef762c661d60d6649cd5c258` |

arXiv was not reachable from our environment on 2026-10-12, so we could not confirm that a fresh download of these
versions has the same bytes. If yours differs, line numbers may shift; the page, section and table references below
are the stable ones.

## Claims the results depend on

| Claim | Public location | Cited in the logs as |
|---|---|---|
| Target accuracies, InternVL-1B: ST 46.1 / TSR 48.5 / VisRef 52.0 | VisRef v1, Appendix I, Table 6, PDF p. 14 | `LAY L900` |
| Models evaluated at 1B, 2B and 8B | VisRef v1, Appendix I, PDF p. 14 | `LAY L924` |
| δ = 0.25 for the stopping entropy, m = ⌊0.3·\|V\|⌋, K_max = 10 | VisRef v1, §5 Experiments, PDF p. 6 | `LAY L433–435` |
| The selection enforces \|V_k\| = m and stops after m picks | VisRef v1, §4, PDF p. 5 | `LAY L341`, `L360–362` |
| "∼1,772 visual tokens" per image (InternVL-3.5-8B, MathVista) | VisRef v1, §4, PDF p. 4 | `LAY L263` |
| Python 3.10.18, PyTorch 2.7.0, Transformers 4.55.0, four A10G GPUs | VisRef v1, Appendix, PDF p. 12 | `LAY L792` (`VisRef_plain.txt L1278–1280`) |
| Latency per prompt (Table 5, 8B) | VisRef v1, Table 5, PDF p. 14 | `LAY L909–913` |
| InternVL3.5-1B MathVista (mini) 59.3; InternVL3-1B 45.8 | InternVL3.5 report v2, Table 3, PDF p. 11 | `REF/InternVL3.5.txt L774–775` |

## Pinned model, data and code

- Model: [OpenGVLab/InternVL3_5-1B @ 2f71cf52542334823e48a46ffba0e2bc9add3446](https://huggingface.co/OpenGVLab/InternVL3_5-1B/tree/2f71cf52542334823e48a46ffba0e2bc9add3446)
  (license apache-2.0 on the model card).
- Data: [AI4Math/MathVista @ 2b6ad69445fbb5695c9b165475e8decdbeb97747](https://huggingface.co/datasets/AI4Math/MathVista/tree/2b6ad69445fbb5695c9b165475e8decdbeb97747).
- Third-party code, the official MathVista extraction prompt and `testmini.json`: every URL in
  `references/SOURCES.json` points to a fixed commit (pinned 2026-10-12; each file has the sha256 of the copy used).

## Dated notes in the logs

Some notes in `docs/SPEC_LEDGER.md` describe the state on the day they were written. For example, "the paper PDF is not
available locally" was true on 2026-10-07; the PDF was checked later (ledger item X-21, 2026-10-11).
