# Audit of discordant_v3_s1s4.jsonl (50 MathVista testmini questions, 150 rows)

Scope: only the 50 supplied questions; text-only audit (no images were available, so visual claims could not be checked against the image). Source: /mnt/user-data/uploads/VisRef/repro/audit/discordant_v3_s1s4.jsonl. Row-level labels: /tmp/claude-0/audit/audit_rows.csv. "Judgment" marks a call that the data alone does not settle.

## 1. Scoring check (S) by method

| method | ok | false_pos | false_neg | unclear |
|---|---|---|---|---|
| ST | 47 | 0 | 1 | 2 |
| TSR | 50 | 0 | 0 | 0 |
| VisRef | 49 | 0 | 1 | 0 |

Non-ok rows (all four are judgment calls about what counts as the "scored text"):
- pid 441 VisRef, false_neg: the answer text reads "...= 9 cm. 9 </think>" and ends with an inner `</think>`, so the text after the last `</think>` is empty and the scorer returned None (rule=empty). The full answer clearly gives 9 = gold.
- pid 579 ST, false_neg: the whole ST output is "7.00" (EOS after 4 tokens, no `</think>`, step1_end=eos); the answer phase is empty, so scorer returned None. "7.00" equals gold 7.
- pid 173 ST, unclear (scored correct): forced `\boxed{B}` (= gold No), but the text after the box twice concludes "0<1, the answer is Yes (A)" before being cut at 100 tokens.
- pid 515 ST, unclear (scored wrong): forced `\boxed{A}` (herring), but the text after it ends mid-sentence at "Alternatively, the answer is B)" (B = gold krill).
No false_pos found. Extraction and letter-to-option mapping (letters vs option text, numerics such as 130.4, -3 vs 3, units) were consistent with the scored text in all other rows. TSR: 0 scoring problems.

## 2. Paired counts (VisRef-only correct : other-only correct)

| comparison | before S corrections | after S corrections (2 false_neg applied; unclear left as scored) |
|---|---|---|
| VisRef vs ST | 26 : 16 | 25 : 15 |
| VisRef vs TSR | 18 : 14 | 18 : 13 |

Correct counts on these 50 rows (before -> after): ST 21 -> 22, TSR 27 -> 27, VisRef 31 -> 32.
Sensitivity to the two unclear ST rows (VisRef-only : ST-only): pid 173 ST counted wrong -> 25 : 14; pid 515 ST counted correct -> 24 : 15; both -> 24 : 14. VisRef vs TSR is unaffected.
Exact two-sided binomial p on the discordant pairs, only valid if these 50 are the complete discordant set (not verified): VisRef vs ST 25:15 p=0.15; VisRef vs TSR 18:13 p=0.47.

## 3. Failure modes (F) for rows wrong after S correction

| method | n wrong | reasoning_error | visual_misread | truncated_or_loop | no_final_answer | format | other |
|---|---|---|---|---|---|---|---|
| ST | 28 | 17 | 1 | 10 | 0 | 0 | 0 |
| TSR | 23 | 16 | 3 | 4 | 0 | 0 | 0 |
| VisRef | 18 | 9 | 2 | 7 | 0 | 0 | 0 |

Judgment notes on F: (a) visual_misread vs reasoning_error could not be verified without the images; visual_misread was used only for object-count/yes-no scene questions where the stated counts or object lists were inconsistent or contradicted the outcome. (b) truncated_or_loop was used when the answer hit its cap (af=True) or looped and the box was forced while the text was unresolved or contradicted the box; ST wrong rows tagged this way: 10 of 28, mostly because ST step 1 is cut at 1024 tokens (47 of 50 ST rows end step 1 at "cap").

## 4. Bug-sign flags (B)

- empty, pid 441 VisRef: final_raw ends "...so the perimeter of the shape is 9 cm.\n\n9  \n</think>"; scored_answer is "" (rule=empty, extracted=None).
- empty, pid 579 ST: thinking_tail "7.00", think_tokens=4, step1_end=eos, scored_answer "" (rule=empty).
- garbled: none found. I read the start and end of every VisRef scored_answer and the tail of every VisRef thinking_tail (and all final_raw where present), and ran an automatic scan (U+FFFD/C1 characters, non-ASCII ratio outside CJK rows, repeated 5-gram counts); the only hits are repetition loops (pids 11, 105, 212, 273, 467, 544, 652), which are coherent English, not corrupted tokens. No text after an injection looks corrupted: each VisRef thinking_tail ends in a literal `<VISREF m=.. dpp=..>` marker and the continuation after it (e.g. pid 668 mid-text) is fluent.
- off_topic: none. prompt_echo: none.
- Not flagged but worth knowing: pid 579 VisRef final text averages (4+10+7+7+7+7)/6 over mixed countries and gets 7 by coincidence (reasoning issue, not a pipeline sign).

## 5. Winner notes per pid

Cells: ST / TSR / VisRef correctness after S correction (1/0). "Winner" = the method(s) correct where at least one other method is wrong.

| pid | gold | ST/TSR/VisRef | winner | why (judgment where marked) |
|---|---|---|---|---|
| 11 | 5 | 0/0/1 | VisRef | VisRef won by a flip-flop: its thinking alternated "answer is 5"/"answer is 6" until the 10-step cap and the last forced/continued answer was 5 (gold). Not a derived result (judgment: coin-flip). |
| 45 | -3 | 0/0/1 | VisRef | All three share identical step-1 thinking (computes -3, wonders about sign). ST/TSR answer phase chose magnitude 3; VisRef answer phase kept -3 (correct arithmetic). Difference arises only in answer phase. |
| 51 | Yes | 1/1/0 | ST+TSR | ST/TSR won: stated counts support Yes. VisRef lost by contradicting its own counts (2 vs 1) at the answer. |
| 58 | yes | 0/0/1 | VisRef | VisRef won with a cap-forced A after oscillating A/B; ST/TSR compared single values rather than area. Winner partly a guess (judgment). |
| 59 | 64° | 1/1/0 | ST+TSR | ST/TSR won; TSR derivation is correct. VisRef spent 10 steps in confused coordinate geometry and asserted 74 deg without derivation. |
| 88 | 30 | 0/1/1 | TSR+VisRef | TSR and VisRef both reached 30 via x+2x=90. ST looped at the 1024 cap ("maybe the answer is 30") and the forced boxed came out as 3 despite text saying 30 (scored as written, S=ok). |
| 95 | 80° | 0/0/1 | VisRef | VisRef won with D=80 after waffling between 80 and 50 and a cap-forced boxed D; ST/TSR settled on 50. Winner partly a guess (judgment). |
| 105 | yes | 1/1/0 | ST+TSR | ST and TSR (identical text) correct with a vacuous justification after a degenerate "86,86,..." loop; VisRef gave a plausible but wrong rationale. Winner arguably lucky (judgment). |
| 111 | 5 | 0/1/1 | TSR+VisRef | ST was cut at 1024 tokens mid-count and forced to answer 6; TSR and VisRef finished the enumeration (5). |
| 114 | 9.45 | 0/1/1 | TSR+VisRef | Both TSR and VisRef openly pick 9.45 as the "closest option" after failed derivations; winners are guesses (judgment). ST was cut at cap. |
| 141 | 4.40米 | 0/0/1 | VisRef | VisRef reached the gold option C but its own algebra does not derive 4.40 (judgment: guess/lucky); ST and TSR hand-waved to B. |
| 149 | 55° | 0/0/1 | VisRef | VisRef kept the correct triangle result 55 (=A); ST and TSR wrote the same 55 and then "corrected" it to 65. Identical step-1 thinking; difference is in answer phase. |
| 167 | 2 | 0/1/1 | TSR+VisRef | ST cut at 1024 before finishing the zero search and forced -4; TSR and VisRef completed it (2). |
| 173 | No | 1/0/0 | ST | All three counted 0 buses vs 1 SUV (which implies Yes); ST won only because its cap-forced letter came out B against its own text (judgment: luck / noise). |
| 177 | Yes | 1/0/1 | ST+VisRef | Identical step-1 thinking for ST and TSR; TSR forced letter B contradicts its own counts. Differences are answer-phase noise. |
| 198 | 1 | 0/1/1 | TSR+VisRef | TSR and VisRef both ended at cap with unresolved reasoning and forced A (gold). Winners are guesses (judgment). |
| 206 | 3 | 0/1/0 | TSR | TSR forced B (gold) from unresolved reasoning; VisRef forced A. Winner is a guess (judgment). |
| 212 | yes | 0/1/1 | TSR+VisRef | ST looped to the cap; TSR looped but forced A; VisRef answer text contradicts its own step-1 conclusion (B) and lands on A. Outcomes driven by loop/flip noise (judgment). |
| 255 | 40° | 1/0/1 | ST+VisRef | VisRef derived 40 (110-70) correctly; ST asserted 40 without derivation; TSR looped to kmax and forced A. |
| 273 | 26 | 1/1/0 | ST+TSR | Options are single values but the question asks for both diagonals, so the model loops on "B and C"; the cap-forced letter (C for ST/TSR, B for VisRef) decides the outcome (judgment: coin-flip). |
| 274 | 7 | 0/0/1 | VisRef | VisRef ends at 7 (=AD+BC, circumscribed-quadrilateral theorem) but its shown work does not derive it; ST/TSR wrong. Winner correct answer, weak support (judgment). |
| 285 | 25° | 0/1/1 | TSR+VisRef | VisRef derived 25 correctly (half of central angle 50); ST used a wrong angle sum. TSR scored text is a bare "boxed D". |
| 353 | 130.4 | 0/1/1 | TSR+VisRef | ST cut at 1024 tokens before solving; TSR and VisRef solved it (130.4). |
| 360 | 50 | 1/0/1 | ST+VisRef | ST and VisRef give 50 with only asserted justification (judgment: weak support); TSR asserts 110. |
| 373 | 120° | 0/0/1 | VisRef | VisRef picked B=120 explicitly "based on the options", not from a derivation (judgment: guess). |
| 423 | no | 0/0/1 | VisRef | VisRef kept the tie argument (equal at x=100 -> "no"); ST ignored the tie; TSR reasoned to B but the cap-forced box said A. |
| 441 | 9 | 1/1/1 | none (tie after S correction) | All three have the same 237-token step-1 and correct answer text. VisRef scored 0 only because it emitted a trailing </think> after the answer (scoring artifact, not a reasoning failure). |
| 467 | 110 | 1/1/0 | ST+TSR | ST/TSR land on 110 through loops/guess; VisRef loops on 100. Outcomes are guess-level (judgment). |
| 485 | 10° | 1/0/0 | ST | ST won with a cap-forced C before it could argue itself out of it; TSR actually derived 10 then overrode it with 20; VisRef used a false equilateral claim. |
| 495 | No | 1/1/0 | ST+TSR | ST/TSR got B although their stated counts (0 vs 1) imply Yes; VisRef oscillated and was cut. Outcome is luck-level for all (judgment). |
| 515 | kril | 0/1/1 | TSR+VisRef | TSR and VisRef answered B (krill); VisRef has a coherent food-web argument, TSR a forced guess. ST unresolved at cap. |
| 520 | yes | 1/1/0 | ST+TSR | ST/TSR forced A and VisRef forced B from near-identical step-1 reasoning; ST text itself notes a higher series, so A is weakly supported (judgment: coin-flip). |
| 544 | 44 | 1/1/0 | ST+TSR | TSR has the correct sum 27+17=44. VisRef drifted over 10 steps to 97 (wrong group). ST looped but boxed 44. |
| 556 | Yes | 0/1/0 | TSR | All three misparse the comparison (red bus count vs other objects). TSR's A is a forced guess (judgment). |
| 579 | 7 | 1/1/1 | none (tie after S correction) | After correction ST also answers 7 ("7.00"). VisRef final text averages mixed countries but lands on 7 (coincidental method). |
| 610 | 2 | 0/1/1 | TSR+VisRef | TSR and VisRef derived r=2 (4-r=r); ST answered the larger radius. |
| 637 | 120° | 0/0/1 | VisRef | VisRef states 150 and then jumps to 120 ("considering the full angle around H"); weakly supported correct option (judgment). |
| 645 | 2.4 | 1/0/0 | ST | ST won with a bare answer; TSR and VisRef both spent several steps and reached wrong values via a wrong area relation. |
| 652 | 0 | 1/0/1 | ST+VisRef | ST and VisRef give 0 (gold) without coherent justification (ST text contradicts its answer; VisRef loops "the answer must be 0"). TSR gives 3 consistently from its premise. Winners are lucky/anchored (judgment). |
| 668 | Yes | 0/0/1 | VisRef | ST and TSR are identical (same 107-token step 1 and answer). VisRef's second step flipped to Yes with a specious rationale; effectively a coin-flip win (judgment). |
| 670 | 04/02/2005 | 1/0/1 | ST+VisRef | ST and VisRef read the table correctly; TSR's cap-forced letter E contradicts its own finding of 04/02/2005. |
| 704 | 1 & 3 | 1/0/0 | ST | ST's B (1&3) is a cap-forced guess; TSR and VisRef reasoned to 1&4 (wrong). Winner ST has no supporting derivation (judgment). |
| 716 | 16 \sqrt { 5 } | 0/1/1 | TSR+VisRef | TSR and VisRef boxed D (gold 16sqrt5) though their intermediate value was 8sqrt5 (value/option mismatch); ST miscomputed sqrt(320). |
| 747 | 130 | 1/1/0 | ST+TSR | ST/TSR keep 130 (arc = central angle); VisRef halves it to 65. |
| 797 | 2 | 1/1/0 | ST+TSR | ST/TSR chose B (2); VisRef's forced box C contradicts its own text leaning B. |
| 827 | D | 0/0/1 | VisRef | VisRef landed on D after waffling (cap-forced); ST/TSR pattern claims wrong. |
| 874 | 2 | 0/0/1 | VisRef | VisRef computed 4 (same as TSR) and then overrode it with 2; correct by arbitrary jump (judgment: guess). |
| 931 | 50° | 0/1/0 | TSR | TSR derived 50 correctly; ST gave a bare C; VisRef reached the 10-step cap in unresolved circle geometry. |
| 969 | yes | 0/1/1 | TSR+VisRef | TSR and VisRef answer yes; their justification is thin but consistent with each other. |
| 975 | 6 | 1/0/0 | ST | ST enumerated 6 remaining; TSR and VisRef both ended at 5 after extra steps (removing more than brown-matte + blue-metallic). |

Judgment tally of how well the winning answer is supported by its own text (weakly supported = guess, cap-forced letter, flip-flop, or answer not derived by the shown work):
- VisRef-only wins vs ST: 15 of 25 weakly supported; ST-only wins vs VisRef: 10 of 15 weakly supported.
- VisRef-only wins vs TSR: 12 of 18 weakly supported; TSR-only wins vs VisRef: 8 of 13 weakly supported.

## 6. Pipeline facts from the 150 rows

| method | forced_answer | af (answer cap) | inner </think> | step1 ended at cap | stop=kmax | mean k | mean think_tokens |
|---|---|---|---|---|---|---|---|
| ST | 47 | 27 | 5 | 47 | 0 | 1.00 | 970 |
| TSR | 41 | 21 | 5 | 47 | 5 | 3.90 | 3882 |
| VisRef | 44 | 23 | 12 | 47 | 14 | 5.02 | 4875 |

- Step 1 is shared: for the 14 rows where TSR stopped after one step, ST and TSR step-1 text is identical, and for the 8 of those where VisRef also has k=1 the step-1 text matches too (offset only by the `<VISREF>` marker). So on those rows any outcome difference arises purely in the answer phase.
- Answer-phase non-determinism: among the 14 rows with identical ST/TSR step-1 text and TSR k=1, the extracted answers differ in 5 (pids 149, 173, 177, 515, 670), e.g. pid 173 ST=B (correct) vs TSR=A, same thinking. A fixed pipeline with greedy decoding would not do this; judgment: sampling noise or a prompt difference in the forced-answer phase. This noise is of the same order as the ST/TSR/VisRef gaps in this slice.
- ST is a truncated-thinking baseline here: 47/50 ST rows hit the 1024-token cap and 47 were forced to answer ("\nFinal Answer:"); 27 ST answers also hit the answer cap and received the appended `**Final Answer** \boxed` (VisRef 23, TSR 21). Many correct and incorrect answers in all methods are cap-forced boxes after unresolved text.
- Inner `</think>` in the answer: VisRef 12/50, ST 5/50, TSR 5/50. For every inner-`</think>` row whose pre-`</think>` answer was visible in final_raw, the pre- and post-`</think>` answers agreed, except pid 441 VisRef where the post-`</think>` text is empty. Not checkable for rows where the inner `</think>` lies outside the 1800-char window (pids 58, 88 (ST), 495).
- VisRef injection counts: n_inject_per_step equals the full marker size (m=dpp=76) on 9 rows, i.e. all visual tokens re-injected with no selection effect (pids 59, 212, 285, 441, 610, 704, 747, 874, 931).

## 7. Conclusion

I found no corrupted/garbled text, off-topic answers or prompt echoes after visual-token injection, so no evidence that injection breaks generation. Scoring is almost entirely consistent with the scored text (only 2 false negatives, 1 each against VisRef and ST, plus 2 unclear ST rows); applying them moves the comparison from 26:16 to 25:15 vs ST and from 18:14 to 18:13 vs TSR, i.e. no material bias in either direction. The scorer rule "use only text after the last `</think>`" is fragile (VisRef emits an inner `</think>` in 12/50 rows vs 5/50 for ST and TSR, and one VisRef answer was lost this way), and ST can lose its whole answer when step 1 ends by EOS without `</think>`; both are small and offsetting in this slice (judgment). The larger concern for validity is not scoring but noise: the winning answers are often cap-forced or weakly supported by their own text (60-67% of VisRef-only wins and of other-only wins alike, judgment), ST is a 1024-token-truncated baseline, and ST/TSR with identical step-1 text still give different answers in 5 of 14 rows, so the 25:15 and 18:13 splits are within plausible sampling noise (binomial p about 0.15 and 0.47 if these 50 are the full discordant set).

---
## 주 세션 검증 메모 (2026-10-10, 원자료 대조)
- 확인됨: pid 441 VisRef 답 원문 끝이 "...9 cm.\n\n9  \n</think>"라 마지막 `</think>` 뒤가 비어 rule=empty. pid 579 ST는 첫 단계가 "7.00"에서 EOS로 끝나 답 단계가 없음(final_raw None). 392문항 전체에서 rule=empty는 방법당 1행뿐.
- 정정 1: "m=dpp=76인 9행은 시각 토큰 전부 주입"은 틀림. 해당 문항은 n_visual=256(타일 1개)이고 m=⌊0.3×256⌋=76이므로 30% 주입, 선택은 정상.
- 정정 2: "같은 1단계 생각에서 답이 14개 중 5개 다름 → 잡음이 방법 차이와 같은 크기"는 불일치 문항만 골라 생긴 표본 편향. 전체 기준 TSR k=1 282문항 중 ST와 정오 일치 276, 추출 답 일치 274(report.py CRN 점검). 디코딩은 greedy가 아니라 T=0.6 샘플링이고, 같은 시드라도 fp16 배치 구성 차이로 갈라질 수 있다.
- "승자 답이 약하게 뒷받침됨(60–67%)"은 서브에이전트 판단이며 방법 간 비율이 비슷하다는 점만 참고.
