# What one question looks like in the main run (A1)

This page shows, for the main run, the exact prompt, the order in which text and image vectors enter the context, one
real three-step VisRef question, and a key to the raw output files. Every setting the paper does not state, with what
was tested later, is in [REPORT, Section 5.3](../REPORT.md#53-open-questions-about-the-papers-setup).

## 1. Prompt

InternVL3.5's `internvl2_5` template with its R1 thinking system prompt, the same text `model.chat()` builds
(`repro/visref/model.py`, `build_prompt`). `{n}` is the number of image tokens of the question (1,961.5 on average;
dynamic tiling with max_num 12 plus the thumbnail tile), and `{query}` is the `query` field of the MathVista testmini
parquet (the hint line and the question).

```text
<|im_start|>system
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.<|im_end|>
<|im_start|>user
<img><IMG_CONTEXT> × {n}</img>
{query}<|im_end|>
<|im_start|>assistant
```

Sampling: temperature 0.6, top-k 50, top-p 0.95, fp16, seed 42 derived per question. The three methods share the
first thinking step (it is generated once and copied).

## 2. Order of the context, step by step

Step k of TSR and VisRef (`repro/visref/methods.py`, module docstring):

| | Context appended | Notes |
|---|---|---|
| 1 | thinking z_k, until the model writes `</think>`, writes EOS or reaches 1,024 tokens | a self-written `</think>` is removed from the context; a cut step gets one `\n` if its text does not already end with one |
| 2 | VisRef only: Ṽ_k, the selected image vectors, as input embeddings | no `<img>` tags or separators; they take the next position ids after the thinking (not the positions they had in the image) |
| 3 | `</think>` + `\n\n\boxed`, then one sampled answer (temperature 1, up to 16 tokens, ends at the box's closing brace) | the stopping probe; H_k is the sum of its token entropies (natural log); the probe text is then removed from the context |
| 4a | if H_k < 0.25, or k = 10, or less than 256 tokens of the 10,240 budget are left: `</think>`, then the final answer | after a cut step the answer starts after `</think>` + `\nFinal Answer:` |
| 4b | otherwise `Wait, think more`, and step k + 1 starts | |

ST is step 1 followed directly by the final answer. An answer that reaches 1,024 tokens without EOS gets
`\n**Final Answer**\n\boxed` and up to 100 more tokens. The scored text is what follows the last `</think>`.

## 3. A real VisRef question with three steps (question 111)

Question (CLEVR-Math, free form, integer): "Subtract all big gray matte things. Subtract all small metallic cylinders.
How many objects are left?" Right answer 5. 3,328 image tokens, so m = ⌊0.3 · 3,328⌋ = 998.

| Step | Thinking tokens | Ended by | Image tokens injected (of m = 998) | Probe answer | H_k | Decision |
|---|---|---|---|---|---|---|
| 1 | 1,024 | cut at the limit | 202 | `{6}` | 0.700 | continue |
| 2 | 1,024 | cut at the limit | 154 | `{6}` | 0.703 | continue |
| 3 | 399 | the model's own `</think>` | 122 | `{5}` | 0.043 | stop, answer |

Final answer: 5 (correct). On the same question ST answered 6 after its cut first step, and TSR stopped after two
steps (H 0.715, then 0.005) and answered 5. Almost all of H_k sits on the digit token of the probe answer (step 1:
token entropies 0.0004, 0.6999, 0.0 for `{`, `6`, `}`).

How the stored text looks around the injections (`thinking` field; `<VISREF m=… dpp=…>` is a log marker written where
Ṽ_k entered the context, not text the model saw; " carefully." is the model's own continuation of the cue):

```text
… 6. Purple cylinder (now 0)\n7. Green cylinder (\n<VISREF m=998 dpp=202>Wait, think more carefully.\n\nWait, the list of objects after removing …
… Alternatively, maybe the user wants the answer as 5. Let me confirm\n<VISREF m=998 dpp=154>Wait, think more carefully.\n\nWait, the question is: …
… 8 - 3 = 5.\n\nYes, 5. So the answer is 5. Therefore, the integer answer is 5.\n<VISREF m=998 dpp=122>
```

Step 1 is cut in mid-sentence, as in 641 of the 1,000 questions; that is where most injections happen in this run
(REPORT, Section 4.1).

## 4. Raw outputs

Main run: `repro/kaggle/runs/b_main_s1` … `b_main_s9`, files `output/results/results_{0,1}.jsonl`, one row per
question and method. Final scored rows (rules + LLM fallback): `repro/llm/final/merged.jsonl`.

| Method id | Meaning |
|---|---|
| `st` | ST |
| `tsr` | TSR |
| `visref_raw_rank` | VisRef as run in the main run (selection stops at the kernel rank) |
| `visref_raw` | an early variant filled up to m by relevance, run only in session 1 (74 questions) and dropped |

| Field | Meaning |
|---|---|
| `pid`, `method`, `answer_gt` | question id, method id, right answer |
| `steps[]` | one record per thinking step: `think_tokens`, `think_end` (`cap` cut, `close` own `</think>`, `eos`), `m`, `n_dpp` / `n_inject` (image tokens selected / injected), `rank_V`, `n_unique_z` (distinct thinking tokens, which bounds the kernel rank with input-embedding Z), `H`, `probe_answer`, `probe_ent` (token entropies of the probe), `probe_valid` |
| `k`, `stop_reason` | number of steps; why thinking stopped (`st`, `entropy`, `kmax`, `budget`) |
| `thinking` | all thinking text with the injection markers above |
| `final_answer`, `final_finish`, `final_forced_close`, `final_forced_answer` | the answer text, how it ended, whether `\nFinal Answer:` and answer forcing were used |
| `extracted`, `prediction`, `correct` | rule extraction and score (the main scoring adds the LLM fallback in `merged.jsonl`) |
| `n_visual`, `prompt_len`, `gen_tokens`, `code_version` | image tokens, prompt length, generated tokens, hash of the generating code |
