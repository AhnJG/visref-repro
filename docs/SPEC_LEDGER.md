# SPEC LEDGER: VisRef 재현 (Appendix Table 6, 행 "InternVL-1B", MathVista testmini 1,000문항)

- 목표 수치: ST 46.1 / TSR 48.5 / VisRef 52.0 (VisRef_layout.txt L900, Table 6)
- 점검 대상 실행 구성: `repro/run.py` DEFAULTS(L20-23) + `--methods st,tsr,visref_raw_rank --think-cap 1024 --total-think-cap 10240`
- 작성: 2026-10-07. 근거는 원본 소스(논문 layout 텍스트·페이지 PNG, 인용 논문, 공식 인용 코드, 모델 카드, 공식 MathVista 평가 코드)만 사용했다. `repro/README.md`의 주석은 근거로 쓰지 않고, 확인 대상으로만 다뤘다.

## 핵심 판단 요약 (2026-10-10 신설 — 재현 판단 시 먼저 읽는 곳)

논문에 없는 부분을 무엇으로 채웠는지, 그 근거가 얼마나 강한지 한 곳에 모은다. 자세한 원문 인용은 오른쪽 열의 본 표 행, 결정 경위는 `EXPERIMENT_FLOW.md`(EF) 날짜 항목에 있다.

**근거 등급** (강한 순서. 충돌하면 위가 이긴다. 충돌 기록은 Section C)
- **A 논문 명시**: VisRef 본문·부록·알고리즘에 값이나 절차가 적혀 있다.
- **B 논문 그림·표에서 유도**: 그림·표에서 읽어 낸 것.
- **C VisRef 인용 문헌과 그 공식 코드**: [7] Chen(DPP greedy), [19] Ghosal, [33] MathVista, [34] s1, [45] InternVL3.5, [55] CEIL. (인용 번호: `VisRef_plain.txt` L998-1196 참고문헌, [45]는 L735)
- **D 인용 문헌이 지정한 공식 자료**: InternVL3.5 모델 카드(`SRC/internvl_README.md`), InternVL3.5가 평가에 쓴 VLMEvalKit(`REF/InternVL3.5.txt` L644, 모델 카드 L80). VisRef가 직접 인용하지는 않는다.
- **E 비인용 참고 구현**: VisRef도 인용 문헌도 언급하지 않지만 빈 곳을 메우려고 우리가 고른 공개 구현(DEER, X-5).
- **F 운영 선택**: 근거 없이 정한 값. 결과 보고에서 한계·민감도로 다룬다.

| # | 판단 | 우리 선택 | 등급 | 핵심 근거 위치 | 본 표 행 / 결정 기록 |
|---|---|---|---|---|---|
| J-1 | 대상 체크포인트 | InternVL3.5-1B | B | Table 6 L900, 8B 행 = Table 1 InternVL3.5-8B (L396-398) | 1, C-8, B-4 |
| J-2 | thinking 모드 켜기 | R1 system prompt | D | 모델 카드 L412-423 | 3 |
| J-3 | 샘플링 | T 0.6, top-k 50, top-p 0.95 | D | 모델 카드 L412, L767 | 4, C-10 |
| J-4 | δ, m, K_max | 0.25, ⌊0.3·\|V\|⌋, 10 | A | LAY L433-435 | 13, 25, 35 |
| J-5 | 단계 경계와 이어가기 문구 | 모델이 낸 `</think>`에서 끊고 "Wait, think more" | B + C | Fig 2; Ghosal L159-161; s1 | 10, 16, C-1 |
| J-6 | V_k 주입 위치·시점 | z_k 뒤, 정지 검사 전. 마지막 단계 선택도 최종 답 직전에 들어감 | A | Alg. 1 L323-325 | 17, 26, 41; EF 2026-10-08 정정 |
| J-7 | DPP 선택 수 | Chen greedy, rank(L) < m이면 채우지 않음 | C (논문 문구 "\|Ṽk\| = m"(A)과 충돌) | Chen 코드 `dpp.py` L29-30; LAY L341, L360-362 | 36, 37, C-2, B-1; EF 2026-10-07 |
| J-8 | Z(생각 토큰) 표현 | 입력 임베딩 | F | 논문은 층 미지정 | 30, B-5; EF 2026-10-08 hidden_rank_check; 마지막 층 변형 `visref_hs`(EF 2026-10-13) |
| J-9 | 정지 통계 H_k 추정 | `</think>`+`\n\n\boxed` 뒤 답 1개(T=1) 토큰 엔트로피 합 | F (정의 자체는 A: Alg. 1 L325, C: Ghosal) | Fig 2 "Answer:" 라벨뿐 | 21-24, B-2 |
| J-10 | 토큰 상한 | 단계 1024 / 합계 10,240 / 답 1024 | F (1024는 모델 카드 예제 값, D) | 모델 카드 L527 | 11, 12, 14, B-6 |
| J-11 | 생각이 상한에서 잘렸을 때 답 | `</think>` + `\nFinal Answer:` | C (s1 [34] 공식 코드. VisRef 파이프라인에 적용한 것은 우리 해석) | `s1/vllm_causallms.py` L254, L401-411 | 44, C-11; EF 2026-10-07 |
| J-12 | 답이 EOS 없이 답 상한에 닿았을 때 | DEER af: `\n**Final Answer**\n\boxed` + 최대 100토큰 | E | `deer/vllm-deer.py` L165, L252, L655-656 | 52, X-4, X-5; EF 2026-10-07 |
| J-13 | 채점 대상 텍스트 | 마지막 `</think>` 뒤 | D (VLMEvalKit `SPLIT_THINK` 규칙과 같음. 기본값은 꺼짐, InternVL3.5 평가에서 켰는지 미확인) | `vlmevalkit…/vlmeval/inference.py` L259, L267-270 | 43; EF 2026-10-10 |
| J-14 | 답 추출 | 규칙(v4, 2026-10-11부터. b_main 판정용 기록값은 v3이고 v4로 정오가 바뀌는 행 0) 먼저, 규칙 실패 행만 공식 MathVista 추출 프롬프트로 LLM 추출 | C 일부 (MathVista는 모든 응답에 GPT-4 추출기) + D (규칙 우선·LLM 보조는 VLMEvalKit 방식) | `REF/MathVista.txt` L466-467 | 45, 46, B-8 |
| J-15 | 하드웨어·정밀도 | Kaggle T4 fp16 | 논문 A10G(A)와 불일치 | LAY L792 | 2, 48, B-7 |
| J-16 | 난수와 짝 설계 | 같은 pid 같은 시드(CRN), 첫 단계 fork 공유 | F (방법별 분포는 그대로, 짝 차이 분산만 줄임) | — | 49 |
| J-17 | 통계 판정 | 주 비교 VisRef−ST, VisRef−TSR. McNemar 정확 검정 + Holm, 부트스트랩 95% CI | F (논문은 검정·CI·시드 수 미보고) | — | EF 2026-10-10 사전등록 |
| J-18 | 답 처리 민감도 | (a) 그대로 (b) af 앞에서 잘라 재채점 (c) af 행 오답. 주 검정은 (a) | F | — | EF 2026-10-10 |
| J-19 | 문항 순서와 분석 표본 | seed 0으로 섞은 순서, 2개 shard. 앞에서부터 끊김 없이 끝난 문항(complete prefix)만 분석 | F | — | EF L807 (사용자 요청) |
| J-20 | 정지 판정용 답과 최종 답 | 최종 답은 정지 판정에 쓴 probe 답을 다시 쓰지 않고 따로 생성 | A (Alg. 1) | — | EF L776 (사용자 결정) |
| J-21 | LLM 추출 가드 | 채점 텍스트가 빈 행은 보내지 않음, LLM이 뽑은 답은 응답 텍스트에 실제로 있을 때만 인정 | F | — | EF L1237 (사용자 승인) |
| J-22 | 판정 문구 규칙 | 본 실험은 CI가 0과 논문 값을 포함하는지로 판정, 후속 실험은 각 사전등록 규칙 | F | — | EF L1296, L1277, L1534·L1543, L1666-1669 |

원칙 1 (2026-10-10 사용자 채택): A–C(VisRef 본문·그림, 인용 문헌·코드)가 정하지 않은 부분은 D(InternVL3.5 공식 자료: 모델 카드, InternVL3.5가 평가에 쓴 VLMEvalKit 설정)를 기본값으로 삼는다. 단, 이미 사전등록하고 진행 중인 b_main 설정은 바꾸지 않고, D와 다른 점은 아래 "D 자료 대조" 표로 관리하며 필요하면 보조 실험으로 확인한다(첫 보조 실험 st_official은 빌드·사전등록 후 보류: b_main 우선, EF 2026-10-10).

원칙 2: 결과를 본 뒤에는 위 선택을 바꾸지 않는다. 바꿔야 하면 EF에 "결과 후 변경"으로 이유와 함께 적고 이전 선택의 결과도 함께 보고한다.

### 결정 대장: 결정 주체·코드 위치·테스트 (2026-10-12 결정 점검)

점검 방법: Sonnet 하위 에이전트가 EF L645부터 끝까지(2026-10-06 이후)와 L497-499에서 코드 동작을 정하는 사용자 결정·지시를 뽑아 코드 줄과 맞춰 봤다(56건. 하위 에이전트 집계: 반영 43, 뒤의 사용자 결정으로 대체 9, 불일치 2, 찾지 못함 2). Claude가 아래 표에 연결한 EF 줄과 코드 줄을 원문에서 다시 확인했다. 찾지 못함 2건 중 `visref_hs` 실행 설정(단계 1,024 / 합계 10,240, 이중 probe 끔, timeout 29,000s)은 작업 폴더 run_config로 확인해 반영으로 보고(이 run_config는 2026-10-13 공개 저장소에 추가), 나머지 1건은 fill 판정 규칙이 코드 없이 사람이 적용된 것(J-22)이다. 불일치 2건은 J-8과 J-10이다. 하위 에이전트 원본 표는 작업 폴더 `repro/audit/decision_check_20261012/raw.md`(공개 안 함). EF 줄 번호는 작업 사본과 공개 사본(`docs/EXPERIMENT_FLOW.md`)이 같다. 테스트 이름은 `repro/tests/`의 함수 이름이고, `test_decisions`는 이번에 추가한 `tests/test_decisions.py`(결정된 값을 고정하는 테스트)다.

| J | 결정 주체 (EF 줄) | 코드 위치 | 테스트 | 점검 결과 |
|---|---|---|---|---|
| J-1 | 사용자: 주 목표 행(L750), "InternVL3.5-1B로 확정"(L923) | `run.py` L73, `kaggle/build.py` L19 (revision 고정) | 없음 (각 실행의 run_config에 기록) | 일치 |
| J-2 | Claude가 공식 자료에서 고름. 사용자 원칙 1(L1139)과 같은 방향 | `visref/model.py` L6 | 없음 | 일치 |
| J-3 | 사용자: 최종 답은 T 0.6 자연 응답(L820), top-p 0.95는 문서화만(L925) | `run.py` L20-23, `visref/engine.py` L69 | `test_decisions`, `test_sampler_topk_first_respects_top_p_and_k` | 일치 |
| J-4 | 논문 값 | `run.py` L20, `visref/dpp.py` L46-48 | `test_decisions` | 일치 |
| J-5 | 기록된 사용자 결정 없음 (논문·인용 근거) | `visref/methods.py` L187, L299-306 | `test_cut_step_gets_separator_before_wait_then_natural_answer_at_kmax`, `test_closed_step_continues_without_separator`, `test_decisions`(문구) | 일치 |
| J-6 | 위치는 논문. 태그 없이 주입은 사용자(L775) | `visref/methods.py` L204 | `test_kmax_and_visref_injection_of_greedy_picks`, `test_visref_cut_step_separator_comes_before_injected_tokens` | 일치 |
| J-7 | 사용자: rank만큼만(L899), 주 VisRef = `visref_raw_rank`(L976) | `visref/dpp.py` L26-34, `visref/methods.py` L203, `diag/final_judgment.py` L17 | `test_dpp_matches_naive_greedy`, `test_dpp_rank_deficient_returns_only_greedy_picks`, `test_raw_rank_with_the_same_step_is_rank_limited` | 일치 |
| J-8 | 10-07 코드 재작성 때 입력 임베딩으로 구현. 그 전 사용자 결정 "Z에 입력 임베딩이 아닌 last hidden state가 타당"(L497-499)과 다르고, 바꾼 경위 기록을 찾지 못함. 10-12 사용자 지시로 마지막 층 Z 변형 `visref_hs` 사전등록(L1659) | `visref/methods.py` L203(주 VisRef), L197-200(`visref_hs`), `visref/engine.py` `last_hidden` | `tests/test_hs.py` 6개 | **불일치**: 이번 점검에서 찾은 빠진 사용자 결정은 이 하나. `visref_hs` 결과(EF 2026-10-13): 주입 수 m 도달, hs − TSR +0.24 pp (−2.38, +3.10), 사전등록 판정 ③(논문 크기 추가 이득 재현 안 됨) |
| J-9 | 사용자: 답 하나로 측정(L750), 엔트로피 합·T 1(L820), `\boxed` probe(L827), 현행 유지(L955) | `visref/methods.py` L47, L104, L277-280, `run.py` L21 | `test_tsr_probe_sums_entropy_and_continues_until_confident`, `test_brace_probe_*` 6개, `test_engine_lumped_entropy_and_stop_check`, `test_decisions` | 일치 |
| J-10 | 사용자: 단계 1,024 / 합계 10,240(L952-956) | 실행마다 run_args로 전달(`kaggle/runs/*/run_config.json`). `run.py` L20 기본값은 옛 결정(L804)의 4,096 / 8,192 | `test_decisions`(b_main 실행 11개(s1–s9, fill, hs)의 run_config가 1,024 / 10,240) | 실행 기록은 일치. **코드 기본값은 불일치**: 인자 없이 `run.py`를 직접 돌리면 옛 값으로 동작 |
| J-11 | 사용자: s1 방식(L888-891). 줄바꿈 수는 감사 반영(L912)을 사용자가 수용(L922) | `visref/methods.py` L105-106, L240, `run.py` L22 | `test_st_cut_step_gets_s1_forced_answer`, `test_s1_*` 4개, `test_decisions` | 일치 |
| J-12 | 사용자(L952-954) | `visref/methods.py` L243-247, `run.py` L22 | `test_s1_answer_hitting_cap_gets_deer_answer_forcing`, `test_natural_answer_hitting_cap_also_gets_answer_forcing_and_eos_answer_does_not`, `test_answer_forcing_after_later_think_end_keeps_scored_tail`, `test_decisions` | 일치 |
| J-13 | 사용자: 10-06 지시(L646), 유지 결정(L1127) | `visref/methods.py` L234-236 | `test_final_text_after_last_generated_think_end`, `test_s1_later_think_end_is_respected` | 일치 |
| J-14 | 사용자: 규칙 먼저 + LLM(L928), b_main은 gemini-3.5-flash-lite(L1250), fill과 `visref_hs`는 Claude Haiku(L1449, EF 2026-10-13), 채점 v4(L1466) | `visref/scoring.py` L18-19, L130, L156, `llm_extract.py` L35, L93-94 | `test_score` 등 채점 테스트, `test_v4_*`, `test_decisions` | 일치 |
| J-15 | 운영 조건, "문서화만"(L925) | `visref/model.py` L82 | 없음 | 논문과 다름 (기록됨) |
| J-16 | 기록된 사용자 결정 없음 | `run.py`(문항별 시드), `visref/methods.py` | `test_crn_tsr_stopping_at_k1_reproduces_st_final_answer`, `test_fork_from_step1_matches_running_step1`, `test_probe_uses_its_own_generator` | 일치 |
| J-17 | 사용자: 주 비교·Holm(L1081-1083), 판정 규칙(L1296) | `diag/final_judgment.py` L17, L24, L71, `summarize.py` L39 | `make check`(저장된 판정 결과를 다시 계산해 비교) | 일치 |
| J-18 | 사용자 승인(L1098-1106: 같은 민감도 분석을 b_main 최종 보고에도 싣고, 주 검정은 주 채점 하나) | `diag/final_judgment.py` (`views`) | `make check` | 일치 |
| J-19 | 사용자 요청(L807) | `kaggle/build.py` L107, `summarize.py` L50 | `test_decisions`(shuffle_seed 0), `test_complete_prefix_stops_at_the_first_unfinished_question_of_each_shard` | 일치 |
| J-20 | 사용자(L776) | `visref/methods.py` L109, L216 | `test_tsr_probe_sums_entropy_and_continues_until_confident`(최종 답 문맥에 probe 토큰 없음) | 일치 |
| J-21 | 사용자 승인(L1237) | `llm_extract.py` L104, L131, L163 | `test_decisions`(가드 2, LLM에 보낼 행 고르기) | 일치 |
| J-22 | 사용자: 본 실험(L1296), fill(L1277), step1_branch(L1534, L1543), `visref_hs`(L1666-1669) | `diag/final_judgment.py` L24, `diag/analyze_step1_branch.py`, `diag/analyze_hs.py` L59. fill 규칙만 코드 없이 사람이 적용(결과 L1452) | `make check`, `test_analyze_step1_branch.py`, `test_analyze_hs.py` | 일치 (fill만 수동 적용) |

이번 점검에서 고치지 않은 것: `run.py` 기본값(J-10)을 1,024 / 10,240으로 바꾸면 code_version이 바뀌므로 사용자 결정 뒤로 미룸(EF 2026-10-12 "결정 점검"). 표에 없는 사용자 결정(대체된 9건 포함)은 하위 에이전트 원본 표에 있다.

### D 자료 대조: InternVL3.5 공식 평가(VLMEvalKit, 커밋 54a063c5)의 InternVL3.5-1B 설정 vs 우리 b_main (2026-10-10)

VisRef가 인용한 InternVL3.5[45]는 평가에 VLMEvalKit을 썼다(`REF/InternVL3.5.txt` L644, 모델 카드 L80). VLMEvalKit에는 1B 설정이 두 개 있고, 공식 59.3이 어느 설정·환경변수로 나왔는지는 미확인(X-1).

| 항목 | VLMEvalKit `InternVL3_5-1B-Thinking` | VLMEvalKit `InternVL3_5-1B` | 우리 b_main | 비고 |
|---|---|---|---|---|
| system prompt | R1 (`cot_prompt_version="r1"`, `SRC/vlmevalkit_postprocess_54a063c5/vlmeval/config.py` L1336-1338; `vlm/internvl/internvl_chat.py` L126-127) | 없음 (v1, `internvl_chat.py` L141-144) | R1 | 같음(Thinking 설정과) |
| 샘플링 | do_sample=True, T 0.6, top_p 0.95 (`config.py` L1338; `internvl_chat.py` L415-417). top-k 지정 없음(lmdeploy 기본값, 미확인) | greedy (`internvl_chat.py` L212, L415) | T 0.6, top-k 50, top-p 0.95 (모델 카드 L767) | top-k만 다를 수 있음 |
| 최대 생성 토큰 | 65,536 (`max_new_tokens=2**16`, `config.py` L1338) | 4,096 (`internvl_chat.py` L212) | 생각 단계 1,024(합계 10,240) + 답 1,024 | **가장 큰 차이**. 우리 ST 첫 단계 63%가 1,024에서 잘림 |
| 답 지시문 | 환경변수 `USE_COT=1`이면 질문 뒤에 "Please answer the question and put the final answer within \boxed{}." (`internvl_chat.py` L128, L276-281). 설정 여부 미확인 | 같음(지시문은 기본 cot 문구, `vlm/internvl/utils.py` L146-158) | 없음 (HF `query`만, `visref/model.py`) | 미확인 |
| 추론 엔진 | lmdeploy (`config.py` L1337) | HF `model.chat` | 자체 엔진, fp16 | |
| 생각/답 분리 | 환경변수 `SPLIT_THINK` 설정 시 마지막 `</think>` 뒤 (`inference.py` L259, L267-270), 미설정이면 전체 응답 | 같음 | 마지막 `</think>` 뒤 | 설정 여부 미확인 |
| 답 추출 | 규칙 prefetch(객관식 `can_infer`, 정수·실수 변환 일치) 실패 시 GPT-4o-mini 추출 (`dataset/image_vqa.py` L600, L626; `dataset/utils/mathvista.py` L77-113) | 같음 | 규칙 v3, 실패 행만 LLM(공식 MathVista 추출 프롬프트) | 구조 같음(규칙 우선 + LLM 보조), 규칙·판정 모델 다름 |

해석: VisRef 논문 ST 46.1은 공식 59.3보다 13pt 낮으므로(B-3), VisRef 저자가 InternVL3.5 공식 평가 설정을 그대로 썼다면 나오기 어려운 값이다. 따라서 이 설정을 따르는 것은 "모델의 공식 사용법에 충실한 baseline"에는 맞지만, VisRef 표의 절대값을 재현하는 방향은 아닐 수 있다. 적용 범위(2026-10-10 사용자 결정): 원칙 1로 채택, b_main은 유지, 공식 설정 ST 보조 실험 st_official은 보류(b_main 완주 뒤 쿼터가 남으면 재검토); 절대값 차이는 평가 설정 차이로 추정해 기술. LLM 추출은 GPT-4o-mini 대신 Gemini/Haiku 사용(공식 판정 모델과 다름, 한계로 보고).

## 0. 읽는 법과 한계

**경로 약어** (모두 `/mnt/user-data/uploads/VisRef/` 아래)
- `LAY` = `references/papers/VisRef_fulltext/VisRef_layout.txt`. 2단 조판이라 한 줄에 좌우 단이 섞여 있다. 인용은 단별로 이어 붙였고 하이픈 줄바꿈(`rea-`/`soning`)만 복원했다. 수식 줄은 깨져 있어 페이지 PNG로 교차 확인했다.
- `REF` = `references/papers/ref_fulltext/` (Ghosal, InternVL3.5, MathVista, VLMEvalKit)
- `SRC` = `references/sources/` (모델 카드 `internvl_README.md`, `internvl_conversation.py`, `internvl_modeling_internvl_chat.py`, `mathvista_evaluation.py`, `cited_code/{chen_fast-map-dpp, ceil_icl-ceil, s1, deer}`)
- `repro/` = 현재 구현. 아래 `file:line`은 `repro/` 기준이다.
- `OUT/ref_texts/` = `/mnt/user-data/outputs/ref_texts/` (alphaXiv로 받은 인용 논문 원문)
- 공개 저장소에서는 `docs/SOURCES.md`가 이 약어들을 공개 출처(arXiv 버전, PDF sha256, 페이지·절·표 번호, 고정된 모델·데이터·코드 리비전)로 연결한다(2026-10-12 추가).

**상태 4종**: 명시 / 그림·표에서 유도 / 인용 구현에서 확인 / 가정(미명시). 가정(미명시)로 표시한 행은 5항목 검색 기록을 아래 "2. 검색 기록"에 남겼다.

**5항목 체크리스트(이 문서에서의 해석)**: 작업 지시문에 항목 정의가 없어 다음 다섯 곳으로 해석했다. ① 본문(§3-5) ② 부록(A-K) ③ 그림·표(Fig 1-5, Table 1-8, 페이지 PNG) ④ 인용 문헌·공식 코드([7] Chen, [19] Ghosal, [34] s1, [55] CEIL, [46]) ⑤ 모델 카드·공식 평가 코드·InternVL3.5/MathVista 논문.

**읽지 못했거나 불완전한 소스 (정직 공개)**
1. (2026-10-07 작성 당시 상태) 논문 PDF는 로컬에 없다. `arxiv.org/pdf/2603.00207`은 프록시에서 `CONNECT tunnel failed, response 403`으로 막혔다. 논문 근거는 `LAY`와 페이지 PNG 01-16뿐이다. → 이후 PDF를 받아 확인했다(X-21, 2026-10-11). 2026-10-12 확인: `LAY`와 `VisRef_plain.txt`는 그 PDF(sha256 f51f8f31…)에 `pdftotext -layout` / `pdftotext`(poppler 22.02.0)를 실행한 결과와 바이트 단위로 같다.
2. Fig 3, Fig 4의 값은 저해상도 PNG(원본 약 2100px 폭 크롭)에서 눈금을 읽은 근사치다(정확도 ±0.2pt, 토큰 ±150). 본문·표에 적힌 수치와 다르면 본문·표를 우선했다.
3. Chen et al. 2018(1709.05135)은 alphaXiv가 추출 원문을 응답에 인라인으로만 돌려주고 파일로 저장하지 않았다. `OUT/ref_texts/1709.05135.md`에는 §3/Algorithm 1, §5, §6.1, Supp. B Theorem 1의 원문 발췌만 있다. 수식 줄바꿈은 손으로 재정렬했고 단어는 바꾸지 않았다. 나머지 5편(2501.19393 s1, 2302.05698, 1207.6083, 2503.04697, 2504.15895)은 alphaXiv가 돌려준 전문 추출 텍스트(AI 요약이 아님)를 그대로 저장했다.
4. alphaXiv 추출본에는 Ghosal 논문의 entropy 추정 표본 수·온도가 없다(`REF/Ghosal_DoesThinkingMoreAlwaysHelp.txt`에서 temperature, samples, top-p, max tokens, N= 검색 결과 entropy 추정과 관련된 서술 0건. hit은 데이터셋 표본 수 같은 무관한 문장뿐). 부록 원문이 누락됐을 가능성이 있다.
5. 업로드된 `repro/` 트리에는 결과 JSONL이 없다(`find *.jsonl` 결과 0건). README의 진단 수치(rank 부족 75%, n_dpp 중앙값 60, m≈537)는 원자료로 재검증하지 못했다.
6. Qwen3-0.6B/8B의 hidden size, 최대 position 같은 모델 config는 로컬에 없어 "미검증"으로 표시했다.

---

## 1. 본 표

열: # | 결정 사항 | 논문 근거 (verbatim ≤25단어 + 위치 + LAY 줄) | 인용 문헌·공식 코드 근거 | 상태 | 현재 구현 (file:line + 값) | 일치 여부 | 대안 | 영향도

### 1-1. 모델과 입력

| # | 결정 사항 | 논문 근거 | 인용 문헌·공식 코드 근거 | 상태 | 현재 구현 | 일치 여부 | 대안 | 영향도 |
|---|---|---|---|---|---|---|---|---|
| 1 | 체크포인트·크기: Table 6 "InternVL-1B" = OpenGVLab/InternVL3.5-1B | "we evaluate InternVL models spanning 1B, 2B, and 8B parameters" (App. I, LAY L924). Table 6 행 "InternVL-1B 46.1 48.5 52.0" (LAY L900). 8B 행 68.1/73.9/79.3은 Table 1 InternVL3.5-8B 행(LAY L396-398)과 같다. 1B가 3 계열인지 3.5 계열인지는 적혀 있지 않다. | InternVL3.5 논문 Table 1: "InternVL3.5-1B InternViT-300M Qwen3-0.6B 0.3B 0.8B 1.1B" (REF/InternVL3.5.txt L346). 공식 MathVista(mini): "InternVL3-1B [187] 43.4 45.8" (L774), "InternVL3.5-1B 44.2 59.3" (L775). | 그림·표에서 유도 | `run.py:L73-74` `OpenGVLab/InternVL3_5-1B`, rev `2f71cf52…`; `kaggle/build.py:L18` | 일치(추정) | InternVL3-1B(thinking 비지원. 공식 45.8이 논문 ST 46.1과 수치상 가깝다). 충돌 기록: 8B 행·Fig 캡션의 "InternVL-3.5"(명시적 계열 표기)와 ST 수치의 근접성이 서로 다른 쪽을 가리킨다. 8B에서도 ST 68.1은 공식 3.5-8B 78.4보다 10.3 낮고 InternVL3-8B 71.6보다 3.5 낮아 수치 근접성은 결정적이지 않으므로 명시적 표기를 더 강한 근거로 봤다(Section C-8). | 상 |
| 2 | 가중치 dtype | 없음 (App. B는 버전과 GPU만 적었다) [검색 S-A] | 모델 카드 예제는 전부 `torch_dtype=torch.bfloat16` (`SRC/internvl_README.md` L372, L386, L403, L517, L526). s1 평가 명령은 `dtype=float32` (`s1/eval/commands.sh` L11, L15. 32B이고 다른 모델이므로 참고용). | 가정(미명시) | `visref/model.py:L82-85` fp16 로드. `engine.py:L107` logits float32, `L116-117` non-finite 검사. T4 기본 (`kaggle/build.py:L79`) | 불일치 (T4가 bf16 미지원이라 하드웨어가 강제) | bf16 (A10G·Ampere 이상), T4에서 fp32 | 중 |
| 3 | Thinking mode 활성화 = R1 system prompt | "MLRMs in their reasoning (thinking) mode" (§5, LAY L428-429). 활성화 방법은 없다. | 모델 카드: "To enable thinking mode, please set the system prompt to our Thinking System Prompt." (`SRC/internvl_README.md` L412), R1_SYSTEM_PROMPT 본문 L415-423 | 인용 구현에서 확인 | `visref/model.py:L6-14` 모델 카드와 바이트 동일(대조 완료). 주입은 `model.py:L117` (`<\|im_start\|>system\n…`) | 일치 | 시스템 프롬프트 없는 비-thinking | 중 |
| 4 | 샘플링: T, top-k, top-p | 없음 (LAY에서 temperature·top-p·top-k·nucleus 검색 0건) | 모델 카드: "we recommend setting `do_sample=True` and `temperature=0.6`" (`SRC/internvl_README.md` L412). lmdeploy 예: `GenerationConfig(top_k=50, top_p=0.95, temperature=0.6, max_new_tokens=8192)` (L767) | 인용 구현에서 확인 | `run.py:L22-23` T=0.6, top_k=50, top_p=0.95. 샘플러 `engine.py:L53-67` | 일치 (README는 top-k/p를 "운영값"이라 적었으나 모델 카드 L767에 근거가 있어 README 표현을 정정할 수 있다) | greedy (s1은 `temperature=0.0`, `s1/README.md` L105 등) | 중 |
| 5 | 시드와 반복 횟수 | 없음. 모든 표가 단일 수치이고 ±가 없다 [검색 S-A] | 해당 없음 | 가정(미명시) | seed=42, 문항별 sha256 시드 (`run.py:L35-37, L116, L140-143`). (문항, method)당 1회 샘플 | 판정 불가(미명시) | 다중 시드 평균 | 중 |
| 6 | 이미지 전처리: dynamic tiling | 없음 (tile·resolution·patch·preprocess 검색 결과에는 무관한 참고문헌 1건뿐). "∼1,772 visual tokens" (LAY L263)이 간접 제약 | 모델 카드 `load_image(…, max_num=12)`, `dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)` (`SRC/internvl_README.md` L468, L506-509). InternVL3.5: "compressed into 256 tokens via a pixel shuffle module" (REF/InternVL3.5.txt L337) | 인용 구현에서 확인 | `model.py:L21-78` 공식 코드와 동일 (max_num=12, 448px, thumbnail). `encode_sample` `L121-132` | 일치 (간접 검산: 평균 \|V\|≈1,790 ← README:L24 "m≈537". 원자료 미확인. Section A-1) | max_num 축소 | 중 |
| 7 | 데이터·split·문항 수 | "MathVista [33] (testmini, 1,000 problems)" (§5, LAY L425-426). "we use the testmini split containing 1, 000 problems." (App. D, L827-828) | 해당 없음 | 명시 | `kaggle/build.py:L19-20, L89` (HF `AI4Math/MathVista` testmini parquet, rev `2b6ad694…`, 기본 1..1000). `model.py:L135-149` | 일치 | 없음 | 하 |
| 8 | 프롬프트 필드·채팅 템플릿 | 없음 (MathVista 프롬프트 형식에 대한 언급 없음) | MathVista: "baselines generate responses given the input query, which incorporates the task description, the question, the choices, and the metadata" (`REF/MathVista.txt` L463-465). `internvl2_5` 템플릿: `sep_style=MPT, sep='<\|im_end\|>\n'` (`SRC/internvl_conversation.py` L382-391). 이미지 토큰: `IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN` (`SRC/internvl_modeling_internvl_chat.py` L295) | 인용 구현에서 확인 | `model.py:L143-144` `row['query']`. `build_prompt` `L114-118`: `<img><IMG_CONTEXT>*N</img>\n{question}` | 일치(추정) (HF `query` 열이 MathVista 템플릿 산물이라는 점은 로컬 소스로 미검증) | VLMEvalKit 프롬프트 | 하 |
| 9 | 문항당 trace 수와 답 개수 | "the model generates a single reasoning trace without additional intervention" (App. C, LAY L798-800). 다수결은 Fig 3 병렬 사고에만 (L420-423) | 해당 없음 | 명시 | (pid, method)당 1 trace, 최종 답 1회 (`run.py:L113`, `methods.py:L21-23`) | 일치 | 없음 | 하 |

### 1-2. 생성

| # | 결정 사항 | 논문 근거 | 인용 문헌·공식 코드 근거 | 상태 | 현재 구현 | 일치 여부 | 대안 | 영향도 |
|---|---|---|---|---|---|---|---|---|
| 10 | 추론 단계 z_k의 정의와 종료 = 모델이 낸 `</think>`까지 (그 토큰은 z에서 제외) | "Rather than producing the final answer immediately, the model is prompted to continue reasoning using special instruction tokens" (§3, LAY L203-206). 한 step 길이: "∼615 text tokens" (L261-263)가 Fig 3(b) ST ★ x≈600과 같은 크기이므로 한 step ≈ ST 전체 trace (유도) | Ghosal: "whenever the model attempts to generate the end-of-thinking delimiter (</think>), we suppress it, append the token “Wait” to the thinking trace" (`REF/Ghosal…txt` L159-161). s1은 stop 토큰을 제거하고 이어간다 (`s1/vllm_causallms.py` L361, L367) | 인용 구현에서 확인 | `methods.py:L93-97` (stop={`</think>`, EOS}), `L191-200` (z=ids[:-1]) | 일치 | 문단(`\n\n`) 단위 step, "Wait" 토큰 기준 step | 상 |
| 11 | 단계당 최대 생성 토큰 | 없음 (max_new·token limit·max token 검색 결과 L411의 Kmax뿐) [검색 S-B] | 모델 카드 HF 예: `max_new_tokens=1024` (`SRC/internvl_README.md` L527). lmdeploy 예: `max_new_tokens=8192` (L767). s1: `max_tokens_thinking` 인자 (`s1/eval/commands.sh`) | 가정(미명시) | CLI `--think-cap 1024` (기본값은 `run.py:L20` 4096). 적용 `methods.py:L79-84` `min(think_cap, total−used, room)` | 판정 불가(운영값). Fig 3 ST ★ x≈400-1,000(근사)과 모순은 없다 (Section A-3) | 4096 / 8192 | 상 |
| 12 | thinking 총 토큰 상한 | 없음. Fig 3 가로축은 병렬 사고 예산(≈14.5K)이지 trace당 상한이 아니다 (L437 "budget of 14K thinking tokens"). "under the same decoding setup and token budget" (App. I, L925)도 값이 없다 [검색 S-B] | s1: `max_tokens_thinking=auto` 등 (`s1/eval/commands.sh`) | 가정(미명시) | CLI `--total-think-cap 10240` (= Kmax×1024). 기본 8192 (`run.py:L20`) | 판정 불가(운영값) | 8192, 무제한(context 한도만) | 중 |
| 13 | Kmax | "maximum reasoning steps Kmax =10" (§5, LAY L433-435). App. C L808-812에 같은 값 | 해당 없음 | 명시 | `run.py:L20` kmax=10. 종료 판정 `methods.py:L273-274` | 일치 | 없음 | 중 |
| 14 | 최종 답 최대 토큰 | 없음 [검색 S-B] | 모델 카드 `max_new_tokens=1024` (`SRC/internvl_README.md` L527) | 가정(미명시) | `run.py:L20` answer_cap=1024. 적용 `methods.py:L119-121` | 판정 불가(운영값) | 512 / 8192 | 하 |
| 15 | 최대 context 길이 | 없음 (context length·window 검색 결과 L259는 지연 시간 문맥) [검색 S-B] | 로컬 config 없음 → 미검증 | 가정(미명시) | `run.py:L20` max_context=40960. `methods.py:L83` room 계산 | 판정 불가(미명시) | 해당 없음 | 하 |

### 1-3. TSR와 이어가기

| # | 결정 사항 | 논문 근거 | 인용 문헌·공식 코드 근거 | 상태 | 현재 구현 | 일치 여부 | 대안 | 영향도 |
|---|---|---|---|---|---|---|---|---|
| 16 | 이어가기 문구 = "Wait, think more" | "special instruction tokens (e.g., “Wait”, “Think more”), denoted by c" (§3, LAY L205-206). Fig 2의 노란 상자 "Wait, think more" (LAY L224, 페이지 PNG로 확인) | Ghosal: "append the token “Wait”" (L160). s1: `ignore_str = "Wait"` (`s1/README.md` L128), `thinking_n_ignore_str=Wait` (`s1/eval/commands.sh` L15) | 그림·표에서 유도 | `run.py:L21` `wait_text='Wait, think more'`. 삽입 `methods.py:L272` | 일치 (충돌 기록: VisRef Fig 2의 "Wait, think more"를 인용 코드의 "Wait"보다 강한 근거로 봤다. VisRef 자체의 그림이기 때문 → Section C-1) | "Wait"만 사용 | 중 |
| 17 | 삽입 위치와 순서: …z_k → V_k → (문구) → z_{k+1}… | Eq. 3: "τ1:k = (z1 , V1 ), (z2 , V2 ), . . . , (zk , Vk )" (LAY L183). Fig 2 순서: Text Reasoning z1 → Selected Visual Tokens V1 → "Wait, think more" → … → zk → Vk (페이지 PNG) | Ghosal: "y ∼ πθ (·\|x, z1 , p, z2 , p, . . . , zk )" (L142) | 그림·표에서 유도 | `methods.py:L209-221` (V_k를 z_k 뒤에 이어 붙임), `L270-272` (`step_ctx + sep + wait`) | 일치 | 문구를 V_k 앞에 | 중 |
| 18 | 길이 상한으로 잘린 step 뒤 구분자 | 없음 [검색 S-C] | s1은 length로 끊긴 뒤에는 이어가지 않고 `"\n"`(텍스트가 `\n`으로 끝나지 않을 때만) + `thinking_end_max` (`s1/vllm_causallms.py` L401-411, L408). "Wait"는 구분자 없이 바로 붙인다 (`s1/README.md` L133, `vllm_causallms.py` L367) | 가정(미명시) | `run.py:L22` `cut_sep='\n'` (잘렸을 때만). `methods.py:L270` | 판정 불가. 인용 코드에는 이어가기 전례가 없어 외삽이다 (s1: L350-351 "Final; do not generate further") | 구분자 없음 | 하 |
| 19 | ST의 정의 | "the model generates a single reasoning trace without additional intervention" (App. C, LAY L798-800) | 해당 없음 | 명시 | `methods.py:L23, L222` (`st` = think 1단계 + 자연 답, 정지 규칙 없음) | 일치 (충돌 기록: Table 1 캡션 "all methods adopt the adaptive stopping criterion" (L387-388)은 ST에는 의미가 없다. 더 구체적인 L443-444와 L798-800을 우선 → Section C-3) | ST에도 probe를 돌려 H_k만 기록 (결과 불변) | 중 |
| 20 | TSR = 같은 정지 기준 + 시각 주입 없음 | "we employ the same adaptive stopping criterion (Section 4.1.2) for textual self-reflection as well" (§5, LAY L443-444). "extends reasoning through text-only reflection without visual refocusing" (App. C, L802-804) | 해당 없음 | 명시 | `methods.py:L202` (`visref*`만 주입), `L273` (같은 `H<δ`) | 일치 | 없음 | 중 |

### 1-4. 정지 기준

| # | 결정 사항 | 논문 근거 | 인용 문헌·공식 코드 근거 | 상태 | 현재 구현 | 일치 여부 | 대안 | 영향도 |
|---|---|---|---|---|---|---|---|---|
| 21 | H_k의 정의 | "we assess the model’s certainty by measuring the entropy of its response distribution" (§4.1.2, LAY L375-377). Alg. 1 L325: "Hk ← − Ey∼πθ [log πθ (y \| xinput , τ1:k )]" | Ghosal: "Ey∼π(·\|z1:i ,x) [− log π(y \| z1:i , x)]" (`REF/Ghosal…txt` L404-405) | 명시 | 정의 구현 `methods.py:L5-8` (docstring), `L257-258` | 일치 (추정 방식은 22-24행) | 없음 | 상 |
| 22 | H_k의 추정: 표본 수, 온도, 계산 방식 | Fig 2는 단계마다 "Answer: 600"/"Answer: 1000"과 분포 막대를 그린다(표본 수는 없음). 본문·부록에 추정법 서술 없음 (estimate·multiple answers·Monte·sampling 검색 결과 0) [검색 S-D] | Ghosal: "after appending each “Wait”, we sample multiple answers y ∼ π(·\|z1:i , x)" (L400-402). 표본 수는 추출본에 없다 | 가정(미명시) | 단일 표본 + 연쇄법칙: H_k = Σ_t H(p(·\|x,τ,y_<t)). probe는 T=1, top-k/top-p 없음 (`methods.py:L5-8, L53, L257-258`) | 판정 불가. 기댓값은 H(Y)와 같고(불편추정) 분산이 크다. δ=0.25 통과가 잡음에 민감하다 | N개 답 샘플 후 −log π(y) 평균, 답 문자열 군집 엔트로피 | 상 |
| 23 | probe 답의 형식과 엔트로피에 합산하는 토큰 | Fig 2의 "Answer:" 라벨뿐이고 프롬프트 문자열은 없다 [검색 S-D] | 해당 없음 | 가정(미명시) | `</think>` + `"\n\n\\boxed"` 뒤 답 1개. 중괄호 깊이 0에서 종료, 최대 16토큰, 유효하지 않으면 H=∞. 깊이 1에서 `}…` 토큰들은 한 결과로 합쳐 계산 (`run.py:L21`, `methods.py:L141-158, L252-259`) | 판정 불가. README의 smoke3 기록: "Answer:" probe는 형식 토큰이 H의 대부분(내용 ≈0.01 nat, 형식 ≈1.3 nat)이어서 TSR이 한 번도 δ 아래로 내려가지 않아 교체 (`methods.py:L15-16`) | `Answer:` probe (`--probe-stop newline`) | 상 |
| 24 | 엔트로피 단위 (log 밑) | 수식은 "log"만 쓴다 (L325, L377) [검색 S-D] | 해당 없음 | 가정(미명시) | nat (`torch.log_softmax`, `engine.py:L118-119`) | 판정 불가. bit이면 같은 δ=0.25가 nat 기준 ≈0.17에 해당 | log2 | 중 |
| 25 | δ_entropy | "δentropy =0.25" (§5, LAY L433). 같은 값 App. C L806-808. Fig 4(a): "Although the accuracy does not vary significantly across different thresholds" (L470-472) | 해당 없음 | 명시 | `run.py:L20` delta=0.25. 비교 `methods.py:L273` (`H < δ`) | 일치 | 0.15 / 0.30 (Fig 4a) | 중 |
| 26 | H_k 계산 시점 = V_k 주입 후 | Alg. 1: "τ1:k ← τ1:k−1 ∪ {(zk , Vek )}" (LAY L323, 10행) 다음에 "Hk ← − Ey∼πθ [log πθ (y \| xinput , τ1:k )]" (L325, 12행) | 해당 없음 | 명시 | `methods.py:L202-223` (V_k를 컨텍스트에 붙인 뒤 `answer_ctx`와 probe 구성) | 일치 (TSR은 V 없음) | V_k 주입 전에 probe | 중 |
| 27 | 루프와 종료 의미 | Alg. 1: "while k ≤ Kmax do … if Hk < δentropy then break … k ← k+1" (LAY L312-328), 이후 "Sample y ∼ πθ (· \| xinput , τ1:k )" (L330) | 해당 없음 | 명시 | `methods.py:L273-274` (이유 순서 entropy, kmax). 최소 1 step. k=Kmax에서도 probe를 한 번 더 돈다(무해) | 일치 | 없음 | 하 |
| 28 | 추가 정지: 남은 thinking 예산 < min_step_tokens | 없음 [검색 S-C] | 해당 없음 | 가정(미명시) | `run.py:L21` min_step_tokens=256. `methods.py:L273-274` `reason='budget'` | 판정 불가. 총 한도 10,240 = 10×1,024라 거의 발동하지 않는다 | 제거 | 하 |

### 1-5. VisRef 시각 토큰 선택

| # | 결정 사항 | 논문 근거 | 인용 문헌·공식 코드 근거 | 상태 | 현재 구현 | 일치 여부 | 대안 | 영향도 |
|---|---|---|---|---|---|---|---|---|
| 29 | V의 표현 공간 = `extract_feature` 출력(LLM 입력 임베딩 공간) | "denote the set of N visual token embeddings extracted from image I, where each vi ∈ Rd is a d-dimensional vector" (§3, LAY L178-180). z_k도 R^{T_k×d}(L304-306)이므로 Eq. 7이 성립하려면 같은 d가 필요하다. 층 선택은 없다 [검색 S-E] | `extract_feature`는 ViT → pixel shuffle → `mlp1` 출력을 돌려준다 (`SRC/internvl_modeling_internvl_chat.py` L195-213). 이 값이 `<IMG_CONTEXT>` 위치에 그대로 들어간다 (L295) | 가정(미명시) | `model.py:L126` `extract_feature(pv).reshape(-1, hidden)`, `L131` 입력 임베딩으로 삽입 | 판정 불가 (같은 d 조건과는 모순 없음) | LLM hidden state(층 ℓ)의 시각 위치 값 | 상 |
| 30 | z_k의 표현 = 입력 임베딩 조회값 | "denote the Tk text token embeddings in reasoning state zk" (§4.1.1, LAY L304-306) [검색 S-E] | Chen: kernel은 주어진 벡터에서 만든다 (`OUT/ref_texts/1709.05135.md` L35 "L_ij = <B_i, B_j> = <r_i f_i, r_j f_j> = r_i r_j <f_i, f_j>"). 층 선택 서술은 없다 | 가정(미명시) | `methods.py:L204` `eng.embed(z)` (= `get_input_embeddings`, `engine.py:L79-81`). 결과 rank(L) ≤ z_k의 고유 토큰 수 | 판정 불가. 영향이 크다: 입력 임베딩이면 rank가 m(≈531)보다 작은 경우가 흔하다고 README는 적었다(L26, 75% step, 미검증). hidden state면 rank ≤ min(T_k, d)이고 8B의 T_k≈615 ≥ m이므로 \|V_k\|=m이 가능하다 (유도, 미검증) | z_k의 마지막 층 hidden state | 상 |
| 31 | M_k = Σ z zᵀ (현재 step z_k만, 합) | "Mk = Σ_{i=1}^{Tk} zk(i) (zk(i))⊤" (Eq. 6, LAY L309-311. 수식 재조판). Alg. 1 6행. Markov 가정: "the utility of Vk at step k depends primarily on the current reasoning state zk" (L261-263) | 해당 없음 | 명시 | `dpp.py:L29` `A = V@Z.T`이면 L = AAᵀ = V(ZᵀZ)Vᵀ. 정규화·평균 없음 | 일치 | 누적 z | 중 |
| 32 | 커널 L | "Define kernel: Lk (vi , vj ) = vi⊤ Mk vj " (Alg. 1 7행, LAY L320). Eq. 7: J = det(L_{Vk}) | 해당 없음 | 명시 | `dpp.py:L3-4, L29-47` | 일치 | 없음 | 중 |
| 33 | 관련도 r_i (Eq. 9) | "ri² = ϕk (vi )⊤ ϕk (vi ) = vi⊤ Mk vi" (LAY L355; 수식 재조판) | Chen: "L_ij = <B_i, B_j> = <r_i f_i, r_j f_j> = r_i r_j <f_i, f_j>"와 "log det(L_{R_u}) = sum_{i in R_u} log(r_{u,i}^2) + log det(S_{R_u})" (`OUT/ref_texts/1709.05135.md` L35, L37) | 명시 | `dpp.py:L30` `d2=(A*A).sum(1)` (r_i² = L_ii) | 일치 | 없음 | 중 |
| 34 | 관련도·다양성 가중 λ | "We additionally experimented with a weighted objective of the form λ · relevance + (1−λ) · diversity" (App. K, LAY L998-999). "default (unweighted) formulation in the main paper, where the two terms contribute equally" (L1003-1004). "performance peaks at λ=0.5" (L1000) | 해당 없음 | 명시 | 가중 없음 (`dpp.py` 전체) | 일치 | λ sweep (Table 8) | 하 |
| 35 | 토큰 예산 m | "visual token budget m = ⌊0.3\|V\|⌋" (§5, LAY L434). 같은 식 App. C L808-810 | 해당 없음 | 명시 | `dpp.py:L57-59` `int(n*frac)`, `run.py:L20` vis_frac=0.3 | 일치 (실제 주입 수는 37행) | 20/40/50% (Fig 4b) | 상 |
| 36 | 선택 알고리즘 = Chen의 빠른 greedy MAP (Eq. 11) | "following prior works [7, 55], we approximate the solution via a greedy selection algorithm" (LAY L335-337). "select the token with maximum marginal gain" (L343) | Chen Algorithm 1: "e_i = (L_ji - <c_j, c_i>)/d_j", "c_i = [c_i e_i], d_i^2 = d_i^2 - e_i^2", "j = argmax_{i in Z\Y_g} log(d_i^2), Y_g = Y_g U {j}" (`OUT/ref_texts/1709.05135.md` L25-28). 코드 `chen_fast-map-dpp/dpp.py` L19-31 | 명시 | `dpp.py:L38-49` (같은 재귀. 이미 고른 항목은 −inf 마스크) | 일치 | CEIL 변형(`ceil_icl-ceil/src/utils/dpp_map.py` L9-34) | 중 |
| 37 | 종료와 rank(L) < m일 때 처리 | "enforcing \|Vk\| = m" (LAY L341). "We terminate after m iterations, setting Vek ← Vk(m)." (L360-362) | Chen Theorem 1: "when kernel L is a low rank matrix, Algorithm 1 returns at most rank(L) items" (`OUT/ref_texts/1709.05135.md` L57). Chen 코드는 `if di2s[selected_item] < epsilon: break` (`dpp.py` L29-30). CEIL 코드는 ε도 마스크도 없이 max_length번 돌며 이미 고른 항목을 반복할 수 있다 (`dpp_map.py` L24-33, `dense_retriever.py` L171-172 `sorted(map_results)`) | 인용 구현에서 확인 | `visref_raw_rank`: greedy가 고른 토큰만 주입. `fill=False` (`methods.py:L204-206`, `dpp.py:L51-54`). 정지는 상대 임계값 (`dpp.py:L34, L41-42`) | **불일치** (논문 문구 "\|V_k\|=m" 대비) / 일치 (Chen 코드의 실제 동작 대비). 수치 검증(scratchpad `dpptest/t.py`, 무작위 가우시안 N=1772, d=1024): rank=178, m=537 → Chen 198개(고유 198), CEIL 537개(고유 193). rank=100, m=531 → Chen 109, CEIL 고유 108. rank=600 ≥ m=531 → 둘 다 정확히 531 | `visref_raw`(남는 자리를 r² 순으로 채움), `visref_raw_reg`(L+εI, `methods.py:L48` 1e-6), z_k를 hidden state로 | 상 |
| 38 | 수치 임계값 ε와 정밀도 | 없음 | Chen: "we introduce a small number epsilon > 0 and add d_j^2 < epsilon to the stopping criteria for numerical stability" (`OUT/ref_texts/1709.05135.md` L18). 코드 `epsilon=1E-10` 절대값 (`dpp.py` L5). CEIL은 ε 없음 | 인용 구현에서 확인 | `dpp.py:L16, L29, L34` float64, 상대 `rel_eps=1e-9`·max d² | 부분 일치 (절대 대신 상대. 스케일 불변) | 절대 1e-10 | 하 |
| 39 | 주입 형식 (태그 유무) | Fig 2는 V_k를 원으로만 그린다. 태그 언급 없음 (`<img>`·special token 검색 0) [검색 S-F] | 모델 카드 chat은 원본 이미지에만 `<img>…</img>`를 쓴다 (`SRC/internvl_modeling_internvl_chat.py` L295) | 가정(미명시) | `visref_raw_rank`는 RAW (`methods.py:L51, L207-208`) = 태그 없이 임베딩만. README:L25: `<img>` 감싸기는 smoke1에서 두 번째 `</think>`와 질문 환각 유발 | 판정 불가(미명시) | `<img>…</img>` 감싸기 | 중 |
| 40 | 주입 순서와 position id | 없음 (position·sorted·ordering 검색 결과 무관) [검색 S-F] | CEIL만 `sorted(map_results)` (`dense_retriever.py` L172). 문맥이 다르다 | 가정(미명시) | 원래 토큰 인덱스 순 (`dpp.py:L54` `sorted(sel)`). position은 이어 붙인 연속 값 (`engine.py:L95, L152`) | 판정 불가 | 선택 순서, 관련도 순, 원 이미지 위치 재사용 | 중 |
| 41 | 이전 step의 V_i도 컨텍스트에 유지 | Eq. 3 `τ1:k = (z1 , V1 ), …, (zk , Vk )` (L183), "Vi ⊆ V represents the subset of visual tokens reinjected at that step" (L185-187), Alg. 1 10행 | 해당 없음 | 명시 | `methods.py:L209, L220, L270-272` (V_k 포함 `step_ctx`에서 이어감) → 컨텍스트 증가 ≈ k·n_inject | 일치 | 최신 V만 유지 | 중 |

### 1-6. 최종 답

| # | 결정 사항 | 논문 근거 | 인용 문헌·공식 코드 근거 | 상태 | 현재 구현 | 일치 여부 | 대안 | 영향도 |
|---|---|---|---|---|---|---|---|---|
| 42 | 최종 답 y의 생성 | Alg. 1: "Sample y ∼ πθ (· \| xinput , τ1:k )" (LAY L330). §4.1.2: "the final answer is generated as: y ∼ πθ (· \| xinput , τk )" (L417) | 해당 없음 | 명시 | probe 답은 버리고 `</think>` 뒤에서 T=0.6, top-k/p로 별도 샘플 (`methods.py:L21-23, L120-121, L237-248`) | 일치 | probe 답 재사용 | 중 |
| 43 | 최종 답의 형식과 채점 대상 텍스트 | "checking whether each model’s final prediction y matches the ground-truth answer y∗" (§5, LAY L435-437) | 모델 카드 R1 프롬프트: "provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline." (`SRC/internvl_README.md` L415-423). VLMEvalKit(InternVL3.5 공식 평가 도구: `REF/InternVL3.5.txt` L644, 모델 카드 L80): 환경변수 `SPLIT_THINK`를 켜면 `s.split('</think>')`의 마지막 조각을 prediction으로 쓴다 (`SRC/vlmevalkit_postprocess_54a063c5/vlmeval/inference.py` L259, L267-270). 기본값은 꺼짐이고 InternVL3.5 보고서 평가에서 켰는지는 미확인. VLMEvalKit은 VisRef 인용 문헌이 아니다(`VisRef_plain.txt` grep 0건) [2026-10-10 추가] | 인용 구현에서 확인 | 마지막 `</think>` 뒤 텍스트를 채점(답 원문은 `final_raw`, 시작 위치 `final_y_start`에 함께 저장) (`methods.py:L237-248`). 후보 규칙은 boxed → "answer is/:" → 마지막 줄 (`scoring.py:L52-71`). `**Answer:**` 같은 마크다운 라벨만 있는 줄이면 다음 비어 있지 않은 줄을 답으로 사용 (scorer v2_md_answer_label); "answer is:"의 콜론과 `**B**.`의 마크다운·마침표 제거 순서를 고친 v3_md_colon_emphasis가 현재 버전 (`SCORER_VERSION`) | 일치(추정). 분리 규칙은 VLMEvalKit `SPLIT_THINK`와 같다 | 전체 응답에서 추출. 첫 `</think>` 앞(첫 답) 기준은 사후 민감도로만 (EF 2026-10-10) | 중 |
| 44 | 길이 상한으로 잘린 뒤 강제 답 | 없음 (forced·Final Answer·budget forcing 서술 없음. "budget forcing"은 서론 L127-128에서 [34] 소개만) | s1: `thinking_end_max = thinking_end + "\nFinal Answer:"` (`s1/vllm_causallms.py` L254), `finish_reason == "length"`일 때만 (L401-411). README 주석: "append "Final Answer:" here … to prevent the model from just continuing to reason in its answer when early exiting" (`s1/README.md` L146) | 인용 구현에서 확인 | `run.py:L22` `forced_prompt='\nFinal Answer:'` (`</think>` 뒤; 2026-10-10 기록 정정, 이전 표기 `\n\nFinal Answer:`는 감사 전 값). `methods.py:L111-119` | 부분 일치: s1은 `<\|im_start\|>answer`+`"\nFinal Answer:"`, 여기는 `</think>`+`"\nFinal Answer:"`. s1은 VisRef 인용 [34]이지만 VisRef 파이프라인에 적용한 것은 운영 해석 | 강제 없음, `\boxed` 박스 모드 | 중 |
| 52 | 최종 답이 EOS 없이 답 상한(1024)에 닿을 때 | 없음 (X-4: 토큰 상한·잘림 처리·답 재촉 서술 없음) | DEER(Yang et al. 2025, arXiv 2504.15895) 공식 코드의 answer forcing: `--af` 옵션(`SRC/cited_code/deer/vllm-deer.py` L165), `answer_prompt_str = "\n**Final Answer**\n\\boxed"`(L252), 마지막 토큰이 EOS가 아니고 af=1일 때 적용(L655-656). DEER는 VisRef 인용 문헌이 아니다(X-5) [2026-10-10 추가] | 가정(미명시) | `run.py:L22` `af_prompt='\n**Final Answer**\n\\boxed'`, `af_cap=100`. `methods.py:L195-219` | 판정 불가(논문 외 처리). 영향은 민감도 (a)/(b)/(c)로 보고: s1–s4 af 행 84/70/87, af 구제 효과 +2.0/+2.3/+2.6pp (EF 2026-10-10) | af 없음(잘린 답 그대로 채점) | 중 |

### 1-7. 평가와 환경

| # | 결정 사항 | 논문 근거 | 인용 문헌·공식 코드 근거 | 상태 | 현재 구현 | 일치 여부 | 대안 | 영향도 |
|---|---|---|---|---|---|---|---|---|
| 45 | 정답 추출 방식 | App. D: 정확도 = "I{y = y*}" (LAY L821). 추출 단계 서술 없음 (GPT·extract·judge·rule-based·exact match 검색 결과 무관) [검색 S-G] | MathVista: 3단계(응답 생성, 답 추출, 점수) 중 "We propose an answer extractor (§F.2) based on LLMs such as GPT-4" (`REF/MathVista.txt` L466-467) | 가정(미명시) | 규칙 기반 추출 (`scoring.py:L74-107`), 규칙 실패 행(last_line/empty)만 공식 MathVista 추출 프롬프트로 LLM 추출 (`llm_extract.py`; 2026-10-10 기록 보강) | 판정 불가. 공식은 GPT-4 추출이라 규칙 기반과 차이가 날 수 있다 (모든 method에 같은 scorer) | GPT-4/gpt-4o-mini 추출기, 엄격한 문자열 일치 | 중 |
| 46 | 정규화와 비교 | 없음 | `normalize_extracted_answer`, `safe_equal` (`SRC/mathvista_evaluation.py` L27-81, L84-94) | 인용 구현에서 확인 | `scoring.py:L120-142` 정규화, `L145-151` 비교 | 일치 | 없음 | 하 |
| 47 | 소프트웨어 버전 | "We run all experiments with Python 3.10.18, PyTorch 2.7.0, and Transformers 4.55.0." (App. B, LAY L790-791) | 해당 없음 | 명시 | `kaggle/build.py:L27` `transformers==4.55.0`. torch는 Kaggle 이미지 기본값 (고정 안 함, `build.py:L34-36`이 env.json에 기록) | 부분 일치 | torch==2.7.0 고정 | 하 |
| 48 | 하드웨어 | "For all experimentation, we use four Nvidia A10G GPUs." (App. B, LAY L792). 그러나 App. H: "latency measurements (on 1 H100 GPU)" (L971) | 해당 없음 | 명시 | Kaggle T4, fp16 (`kaggle/build.py:L79`) | 불일치 (논문 내부도 불일치: 정확도 실험은 A10G가 더 강한 근거 → Section C-7) | A10G급(bf16 지원) | 중 |
| 49 | 난수 공유(CRN)와 step 1 공유 | 없음 [검색 S-H] | 해당 없음 | 가정(미명시) | 같은 pid는 모든 method가 같은 시드, probe는 전용 난수열, step 1 thinking은 fork (`run.py:L35-37, L140-143, L194-200`) | 판정 불가. method별 분포는 변하지 않고 짝지은 차이의 분산만 줄어든다 | 독립 실행 | 하 |
| 50 | 평가 대상 부분집합과 오류 처리 | 전체 1,000문항 (L425-426) [검색 S-H] | 해당 없음 | 가정(미명시) | 문항 순서를 seed 0으로 섞고 `summarize.py --order`는 "완료된 앞부분"만 집계 (README:L52-53). non-finite logits job은 오답으로 기록 (`run.py:L152-156, L213-214`) | 불일치 가능 (미완료 시 무작위 부분집합 → 표준오차 증가) | 1,000문항 전부 완료 | 중 |
| 51 | 디코딩 수치 경로 (fast-KV, SDPA) | 없음 [검색 S-H] | 해당 없음 | 가정(미명시) | 사전 할당 KV와 GQA decode attention (`fastkv.py:L13-52`, scores float32 L47). `use_flash_attn=False`, SDPA (`model.py:L84-89`). 시작 시 greedy 일치 검사 (`run.py:L40-55, L122-128`) | 판정 불가 | stock HF 경로 | 하 |

---

## 1-8. 상태별 행 수

| 상태 | 행 수 | 행 번호 |
|---|---|---|
| 명시 | 19 | 7, 9, 13, 19, 20, 21, 25, 26, 27, 31, 32, 33, 34, 35, 36, 41, 42, 47, 48 |
| 그림·표에서 유도 | 3 | 1, 16, 17 |
| 인용 구현에서 확인 | 10 | 3, 4, 6, 8, 10, 37, 38, 43, 44, 46 |
| 가정(미명시) | 20 | 2, 5, 11, 12, 14, 15, 18, 22, 23, 24, 28, 29, 30, 39, 40, 45, 49, 50, 51, 52 |
| 합계 | 52 | |

---

## 2. 검색 기록 (가정(미명시) 행용)

공통: ① `LAY`를 처음부터 끝까지 정독했고, 아래 키워드를 grep으로 다시 확인했다. 대소문자 무시, 각각 `grep -n -i`. 결과는 "N건"으로 적었다.

| 기록 | 대상 행 | ① 본문 | ② 부록 | ③ 그림·표 | ④ 인용 문헌·코드 | ⑤ 모델 카드·공식 평가 |
|---|---|---|---|---|---|---|
| S-A | 2, 5 | `bfloat\|float16\|fp16\|bf16\|dtype\|precision` 0건. `seed` 0건. `runs\|trials\|std\|standard deviation\|variance\|confidence interval` 0건 | App. B(L788-792)는 버전과 GPU뿐 | Table 1-8 모두 단일 수치, ± 없음 | `s1/eval/commands.sh`에 `dtype=float32`(32B, 별개). s1 README 예제는 `temperature=0.0` | 모델 카드 `torch_dtype=torch.bfloat16` (L372, L386, L403, L517, L526) |
| S-B | 11, 12, 14, 15 | `max_new\|max new\|token limit\|max token\|maximum number of` 1건(L411, Kmax). `context length\|context window\|context size` 1건(L259, 지연 시간 문맥) | App. C(L808-812)에 Kmax뿐 | Fig 3 가로축은 병렬 사고 예산(≈14.5K). trace당 상한이 아니다 | s1 `max_tokens_thinking` (값은 실험별) | 모델 카드 L527 `max_new_tokens=1024`, L767 `max_new_tokens=8192` |
| S-C | 18, 28 | `newline\|\n\|separator` 무관 0건. `Wait\|think more`는 Fig 1/2와 L205-206만 | 없음 | Fig 1/2 "Wait, think more" | s1 `vllm_causallms.py` L367(직접 이어붙임), L401-411(length 분기). Ghosal L160 | 없음 |
| S-D | 22, 23, 24 | `estimate\|sample.*answers\|multiple answers\|Monte\|sampling\|decoding\|greedy` 중 관련 0건 (`Monte`는 L726 참고문헌) | 없음 | Fig 2의 "Answer:"와 히스토그램(표본 수 없음) | Ghosal L400-402 "sample multiple answers" (표본 수 미기재) | 없음 |
| S-E | 29, 30 | `embedding` 관련 2건(L178-179, L304-306). `hidden state\|last layer\|projector\|connector\|final layer` 0건 | App. G(Eq. 13)도 같은 표기 | Fig 2 "Original visual tokens", "Kernel projection to text space" | Chen/CEIL은 주어진 벡터로 kernel을 만든다. 층 서술 없음 | `extract_feature` (`internvl_modeling_internvl_chat.py` L195-213) |
| S-F | 39, 40 | `<img>\|special token\|tag` 무관 0건. `position\|sorted\|ordering\|order of` 무관 0건 | 없음 | Fig 2 순서(z1 → V1 → Wait → …). 태그 없음 | CEIL `dense_retriever.py` L172 `sorted(map_results)` (다른 문맥) | chat()은 원본 이미지에만 `<img>…</img>` (L295) |
| S-G | 45 | `GPT\|extract\|judge\|rule-based\|regex\|exact match` 무관 0건 | App. D L815-822는 "I{y = y*}"만 | 없음 | `REF/MathVista.txt` L466-467 GPT-4 추출기. `REF/VLMEvalKit.txt`도 LLM 보조 추출 | `SRC/mathvista_evaluation.py`는 정규화와 비교만 포함, 추출기는 포함하지 않음 |
| S-H | 49, 50, 51 | `batch\|KV\|cache\|prefill` 무관 0건 | 없음 | 없음 | 해당 없음 | 해당 없음 |

---

## 3. Section A. 논문 보고 수치 대조표

수치는 논문에 적힌 값이다. Fig 3, Fig 4 값은 PNG에서 읽은 근사치다. "우리 값"의 출처가 README면 원자료 미확인이라고 적었다.

| # | 논문 보고 수치 (위치) | 함의 | 우리 값 (출처) | 일치? |
|---|---|---|---|---|
| A-1 | "∼1,772 visual tokens" (LAY L260-263, InternVL3.5-8B, MathVista 평균) | 타일당 256토큰(`REF/InternVL3.5.txt` L337)이면 평균 ≈6.9타일(썸네일 포함). 동적 타일링 max_num=12 + thumbnail과 정합한다. 1B와 8B는 같은 InternViT-300M을 쓰므로(L346, L349) 시각 토큰 수는 모델 크기와 무관하다. m=⌊0.3·\|V\|⌋ ≈ 531 | README:L24 smoke3 "m≈537" → \|V\|≈1,790 (원자료 미확인) | 예 (간접) |
| A-2 | "∼615 text tokens" per step (L261-263), Fig 3(b) ST ★ x≈600 | 한 step ≈ ST 전체 trace (`</think>`로 닫히는 한 구간, 10행). 8B 기준 | step 상한 1,024. 1B의 step 길이 분포는 결과 없이는 알 수 없음 (`think_tokens`, `summarize.py`) | 미검증 |
| A-3 | Fig 3 ST ★ x 위치(읽음, ±150): (a)≈900 (b)≈600 (c)≈500 (d)≈1,000 (e)≈700 (f)≈600 (g)≈850 (h)≈500 (i)≈400 | ST trace는 대체로 400-1,000토큰. 1,024 상한은 8B급 trace를 거의 자르지 않는다 | `--think-cap 1024`. 1B는 반복·장황해질 수 있어 미확인 | 8B 기준 일치 |
| A-4 | Table 5 (H100, 8B): ST 7.1s / TSR 7.7s / Look-Back 7.6s / VisRef 8.2s. "our DPP-based token selection adds only 0.5 secs of overhead compared to Textual self-reflection (TSR), and 1.1 secs compared to standard thinking (ST)" (L909-913). 비교: Fig 3(b) VisRef 첫 점 ≈4,000 thinking tokens (ST의 ≈6.5배), "2.3× increase in inference latency" (L264) | TSR가 ST보다 +8%, VisRef가 +15%뿐이라면 평균 추가 step이 매우 작아야 한다 (대부분 k=1에서 정지하거나 이어가는 step이 짧음). Fig 3에서 VisRef가 ST의 수 배 토큰을 쓰는 해석(Table 1의 79.3이 곡선 첫 점이라는 가정)과 긴장한다. 해석 불확실 | T4 1B의 `wall_s`는 하드웨어가 달라 절대 비교 불가. 비율만 비교 가능 (TSR/ST ≈ 1.08, VisRef/ST ≈ 1.15). 미측정 | 논문 내부 긴장 (해소 불가) |
| A-5 | Table 6: 1B 46.1/48.5/52.0, 2B 52.9/53.7/58.1, 8B 68.1/73.9/79.3 (L899-903) | 목표 델타 (1B): TSR−ST=+2.4, VisRef−ST=+5.9, VisRef−TSR=+3.5. (2B: +0.8/+5.2/+4.4, 8B: +5.8/+11.2/+5.4). n=1,000에서 정확도의 이항 표준오차 ≈1.6pt. 짝지은 차이의 SE는 불일치 비율 d에 따라 ≈√(d/n) (d=0.1-0.2이면 1.0-1.4pt). +2.4는 ≈2 SE라 검출력이 낮고, +5.9는 ≈4 SE | `summarize.py`가 paired bootstrap CI 출력 (실행 전) | 해당 없음 (실행 전) |
| A-6 | 공식 InternVL3.5 MathVista(mini): 1B 59.3 (parallel thinking 69.5), 2B 71.8, 8B 78.4. InternVL3: 1B 45.8, 2B 57.0, 8B 71.6 (`REF/InternVL3.5.txt` L774-775, L779-780, L793-794) | 논문 ST(46.1/52.9/68.1)는 공식 3.5 대비 −13.2/−18.9/−10.3pt, InternVL3 대비 +0.3/−4.1/−3.5pt. VisRef(52.0/58.1/79.3)는 공식 3.5 대비 −7.3/−13.7/+0.9. 논문의 baseline은 공식 설정과 다르다(토큰 예산, 추출, 프롬프트가 불명). 절대 수치는 비교할 수 없고 델타만 비교할 수 있다 | 미측정. 공식 설정에 가깝게 돌면 ST가 ≈59 쪽일 수 있다(미검증) | 기대상 불일치 |
| A-7 | Fig 4(a): δ ∈ {0.15, 0.25, 0.30} → ≈{78.8, 79.2, 79.0} (8B MathVista, ±0.1). "does not vary significantly" (L470-472) | 8B에서 δ 민감도 ≤0.4pt. 정지 추정기가 다소 달라도 영향이 작을 수 있으나 1B는 보고되지 않았다 | δ=0.25 | 8B 한정 정보 |
| A-8 | Fig 4(b): m ∈ {20, 30, 40, 50}% → ≈{77.3, 79.2, 79.2, 79.25}. 본문: "Accuracy improves from 76.1% to 79.2% as m increases from 20% to 30% but plateaus for m ≥ 30%." (L477-478; L529-530에도 "76.1% to 79.2%") | 텍스트 76.1과 막대 ≈77.3이 다르다(읽기 오차 ≲0.3pt로 설명 불가). m 20→30%에서 +1.9~3.1pt, 이후 평탄이라면, 논문 구현은 20%(≈354)와 30%(≈531)에서 실제 주입 수가 달랐다는 뜻이다(rank(L) ≳ 354). 우리 `visref_raw_rank`에서 n_dpp가 rank에 포화하면 m 민감도는 재현되지 않는다 | n_dpp 중앙값 60 (README:L24, smoke3, 1B. 원자료 미확인) | 의심. m sweep(20/30/40%)과 n_inject 로그로 검증 필요 |
| A-9 | Table 3 / 7 / 8 (8B MathVista): Table 3 rel-only 75.6, div-only 77.4, both 79.3. Table 7 Random 67.3 / Rel-only 75.6 / DPP 79.3. Table 8 λ=0.0(Div) 71.2, 0.25 76.8, 0.5 79.3, 0.75 77.4, 1.0(Rel) 75.6 | rel-only 75.6은 세 표에서 일치. div-only는 Table 3의 77.4가 Table 8 λ=0.0의 71.2와 다르다(77.4는 λ=0.75 값과 같다). diversity-only 효과는 신뢰하기 어렵다. 우리 행에는 영향 없음 | ablation 비대상 | 논문 내부 불일치 |
| A-10 | Table 1 캡션: "the numbers in parentheses indicate the performance gain over the ST baseline" (L390-393). 8B: 79.3 (+5.4), TSR 73.9 | 괄호의 +5.4는 VisRef−TSR(79.3−73.9)이고 VisRef−ST(11.2)가 아니다. 본문 L395-397 "5.4%, 11.2%, and 5.9% over standard thinking"은 ST 기준과 맞다 (44.6−39.2=5.4, 79.3−68.1=11.2, 63.1−57.2=5.9). Table 6의 델타는 단순 차이로 계산하면 모호하지 않다 | A-5의 델타 정의 | 캡션 오류 |
| A-11 | App. C 설정: δ=0.25, m=⌊0.3\|V\|⌋, Kmax=10, 단일 trace (L798-812) | 모두 구현과 동일 | `run.py:L20` | 예 |
| A-12 | 하드웨어: 정확도 실험 A10G×4 (L792), 지연 시간 H100 (L971) | 정확도 재현에는 A10G(24GB, bf16)가 관련. T4는 fp16만 지원하고 메모리도 작다 | Kaggle T4, fp16 | 불일치 |

---

## 4. Section B. 불일치·위험 목록 (영향도 순)

| 순위 | 항목 | 영향도 | 근거 | 완화·권고 |
|---|---|---|---|---|
| B-1 | 주입 토큰 수: 논문은 \|V_k\|=m, 구현은 rank(L)까지만 (37행) | 상 | 논문 L341, L360-362. Chen Theorem 1과 코드(`dpp.py` L29-30), CEIL 반복 선택(`dpp_map.py` L24-33). 수치 검증(rank=178/m=537 → Chen 198, CEIL 고유 193). README: n_dpp 중앙값 60/m≈537 (미검증). Fig 4(b)의 m 민감도(A-8) | `visref_raw`, `visref_raw_reg`와 나란히 돌려 n_inject, n_dpp를 로그로 남긴다. 어느 쪽이 목표(+5.9, +3.5)에 맞는지 결정은 결과 후에 한다 |
| B-2 | 정지 추정기와 δ=0.25 스케일 (22-24행) | 상 | 논문은 추정법과 단위가 없다. 단일 표본 \boxed probe는 분산이 크다. 형식 토큰이 H를 지배한 전례(`methods.py:L15-16`). Table 5(A-4)는 대부분 k=1에서 정지함을 시사 | `stop_reason`, k 분포, k=1 정지율, TSR의 k̄를 `summarize.py`로 확인한다. 논문의 지연 시간 비율로 보정 |
| B-3 | baseline 수준 불일치: 논문 ST 46.1 vs 공식 3.5-1B 59.3 (A-6) | 상 | 논문 ST가 공식보다 13.2pt 낮다. headroom이 델타에 영향 | 절대값 대신 델타(+2.4/+5.9/+3.5)와 paired CI로 판단한다 |
| B-4 | 체크포인트 정체: InternVL3.5-1B vs InternVL3-1B (1행) | 상 | 명시적 "InternVL-3.5" 표기와 ST 수치 근접성이 충돌 (Section C-8) | 3.5-1B를 기본으로 하되, 수치가 맞지 않으면 3-1B를 보조 실험으로 둔다 |
| B-5 | z와 V의 임베딩 공간 (29-30행) | 상 | "token embeddings" 층 미지정. 입력 임베딩이면 rank ≤ 고유 토큰 수 | B-1과 함께 hidden state 변형을 소규모로 시험 |
| B-6 | 토큰 상한과 강제 답 경로 (11, 12, 44행) | 상 | 논문에 상한·강제 답 서술이 없다. s1은 length 후 이어가지 않는다 (`vllm_causallms.py` L350-351). README: main_s0에서 ST forced 11/30 vs natural 22/30 (`methods.py:L31-32`, 분모 정의는 원문에 불명) | 잘린 step 비율(`think_end=='cap'`)과 forced 답 정확도를 method별로 집계 |
| B-7 | fp16(T4) vs bf16(모델 카드) (2, 48행) | 중 | 논문은 A10G(bf16 가능). non-finite logits는 오답 처리 (`run.py:L152-156, L213-214`) | `status=='error'` 수를 보고서에 포함. 많으면 bf16 가능한 GPU로 이동 |
| B-8 | 정답 추출: 규칙 기반 vs GPT-4 (45행) | 중 | 공식은 GPT-4 추출 (`REF/MathVista.txt` L466-467) | 모든 method에 같은 scorer라 델타 영향은 작지만 절대값에 영향. 소규모 표본으로 GPT 추출과 대조 |
| B-9 | 이어가기 문구 "Wait, think more" vs "Wait" (16행) | 중 | 논문 Fig 2는 전자, s1·Ghosal은 후자 | 소규모 ablation |
| B-10 | 주입 형식·순서·position id (39, 40행) | 중 | 논문 미명시. `<img>` 감싸기는 환각 유발 기록 (README:L25) | 현 선택(raw, sorted, 연속 position) 유지, 결과에 명시 |
| B-11 | 통계적 검출력과 부분집합 평가 (A-5, 50행) | 중 | n=1,000, SE≈1.6pt(짝지은 ≈1.0-1.4pt). +2.4는 ≈2 SE. 완료된 앞부분만 쓰면 SE 증가 | 1,000문항 전부 완료, paired bootstrap CI 보고 |
| B-12 | 논문 내부 불일치로 인한 기준 불확실 (A-4, A-8, A-9, A-10, A-12) | 중 | Table 5 vs Fig 3, Fig 4(b) 76.1 vs ≈77.3, Table 3 vs 8, 캡션, A10G vs H100 | 불일치 지점은 목표로 삼지 않고 Table 6 델타만 목표로 한다 |
| B-13 | 수치 임계값 ε (상대 vs 절대) (38행) | 하 | Chen 절대 1e-10 vs 구현 상대 1e-9 | 영향은 rank 근처 잡음 몇 개 수준 |
| B-14 | 디코딩 수치 경로 (51행) | 하 | fast-KV, SDPA. 시작 시 greedy 일치 검사로 통제 | `fast_kv_check` 로그 확인 |
| B-15 | 잘린 step 구분자 `\n` (18행), 답/컨텍스트 상한 (14, 15행) | 하 | 논문 미명시, 영향 작음 | 유지 |
| B-16 | README 표현 정정 (4행) | 하 | top-k 50/top-p 0.95는 모델 카드 L767에 근거가 있는데 README는 "운영값"이라 적음 | README의 출처 열을 정정 |

---

## 5. Section C. 소스 충돌 기록 (어느 쪽이 더 강한가와 이유)

| # | 충돌 | 더 강한 근거 | 이유 |
|---|---|---|---|
| C-1 | 이어가기 문구: VisRef Fig 2 "Wait, think more" vs s1/Ghosal "Wait" | VisRef Fig 2 | 재현 대상 논문의 자체 그림이다. 인용 코드는 다른 모델·목적이다 |
| C-2 | \|V_k\|=m (논문 L341, L360-362) vs 인용 코드가 rank(L)까지만 선택 (Chen) 또는 반복 선택 (CEIL) | 의도는 논문 문구, 실제 동작은 코드 | 논문은 rank<m 처리를 서술하지 않는다. 저자의 코드는 본 적이 없고, 인용한 두 구현은 서로 다르다. 어느 쪽도 저자 구현임을 보장하지 못한다 |
| C-3 | Table 1 캡션 "all methods adopt the adaptive stopping criterion" vs ST = 단일 trace (L798-800) | L443-444와 L798-800 | ST는 이어갈 step이 없어 정지 기준이 무의미하다. 더 구체적인 문장이 일반적 캡션보다 강하다 |
| C-4 | Table 1 캡션(ST 대비 이득) vs 괄호 값(TSR 대비) | 표의 숫자와 본문 L395-397 | 숫자를 직접 계산하면 본문과 일치하고 캡션만 어긋난다 |
| C-5 | Fig 4(b) 본문 76.1% (L477, L529) vs 막대 ≈77.3 | 본문 수치 | 읽기 오차(≲0.3pt)를 넘는 차이지만, 본문은 두 곳에서 같다. 막대는 저해상도 근사다. 둘 다 기록한다 |
| C-6 | Table 3 diversity-only 77.4 vs Table 8 λ=0.0 71.2 | 미결 | 어느 쪽이 맞는지 판정할 근거가 없다. 77.4는 λ=0.75 값과 같아 전사 오류를 의심할 수 있으나 추정에 그친다 |
| C-7 | App. B A10G×4 (L792) vs App. H 1 H100 (L971) | 정확도: App. B, 지연 시간: App. H | App. B는 "all experiments", App. H는 지연 시간 실험에 한정한다 |
| C-8 | 1B 정체: 명시적 3.5 표기(8B 행, Fig 캡션) vs ST 46.1 ≈ InternVL3-1B 45.8 | 명시적 표기 | 수치 근접성은 2B, 8B에서 일관되지 않아 증거로 약하다 (A-6) |
| C-9 | 모델 카드 bf16 vs T4 fp16 | 모델 카드 (재현 의도) | T4 제약은 하드웨어 사정이며 논문 쪽은 A10G라 bf16이 가능했다 |
| C-10 | README "top-k/p는 운영값" vs 모델 카드 L767 | 모델 카드 | 원본 소스에 값이 있다 |
| C-11 | s1: length 절단 뒤 이어가지 않음 vs VisRef: 잘린 뒤에도 이어감 | 해당 없음 (논문이 서술하지 않음) | 논문에 상한·절단 처리 서술이 없어 우리 선택은 운영 해석이다 |
| C-12 | Table 5 지연 시간(+8%, +15%) vs Fig 3 토큰(수 배)·Table 6 이득 | 미결 | 두 보고가 같은 설정인지 알 수 없다 (A-4) |


---
## 감사 반영 기록 (2026-10-07)
- 독립 감사 보고서: `repro/AUDIT_REPORT_20261007.md`. 장부 정정 대상 행: 4, 6, 8, 11, 12, 18, 19, 22, 24, 29, 30, 37, 44, A-4, A-6, C-8(내용은 감사 보고서 §2). 주요 정정:
  - 37행: |V_k|=m은 '명시', rank(L)<m일 때의 처리는 '가정(미명시)'. 인용 Chen·CEIL 코드는 rank ≥ m 상황을 전제하므로 저자 구현의 직접 근거가 아니다.
  - 22행: 정지 통계 대상은 답 수준 분포('entropy of the model's answer distribution', L252-253, L380; Fig. 2 답 히스토그램; Ghosal 'distribution over final answers').
  - 29·30행: input-embedding Z + rank 한정 선택은 Fig. 4(b)의 m 민감도와 맞지 않는다.
  - 44·19행: 강제 답은 '인용 구현에서 확인'이 아니라 '가정'(s1 방식을 운영 적용). ST 강제 답은 'without additional intervention'(L798-800)과 부분 충돌.
  - A-4: Fig. 3(b) 재판독 시 VisRef는 trace당 약 2,000토큰, ST의 약 3.5배(4.2~7.3배 판독과 충돌, 기록만).
- 감사 후 현재 구현 변경(code_version 0f8e003fbf90): 강제 답 = 상한 절단 시 [`\n` 필요 시] + `</think>` + `\nFinal Answer:`(s1/vllm_causallms.py L254, L401-412와 같음), EOS 종료는 자연 답. b_test는 `visref_raw_rank`와 `visref_raw`를 함께 실행.
- 2026-10-07 코드 정리(f2422d0e72ff): 방법은 st / tsr / visref_raw_rank만 남김. VisRef 주입 = greedy 선택만(rank<m이면 채우지 않음, 인용 Chen 코드와 같음), 강제 답 = s1 + DEER af. 이전 코드(79eef648ebd1)와 세 방법 출력 동일 810/810(`.backup_79eef648ebd1/EQUIVALENCE.md`).
- 2026-10-10 보강: 맨 앞에 "핵심 판단 요약"(근거 등급 A-F, J-1~J-18) 추가. 43행에 VLMEvalKit `SPLIT_THINK` 근거 추가, 44행 강제 답 문자열을 실제 값 `\nFinal Answer:`로 정정, 45행에 LLM 보조 추출 반영, 52행(DEER af) 신설. 이전 판은 `repro/.SPEC_LEDGER.bak_20261010`.
- 2026-10-12 결정 점검: "핵심 판단 요약"에 J-19~J-22(문항 순서와 분석 표본, 정지 판정용 답과 최종 답 분리, LLM 추출 가드, 판정 문구 규칙)를 넣고, J-1~J-22의 결정 주체·코드 위치·테스트를 "결정 대장" 표로 추가. J-14의 채점 규칙을 v4로 고침. 결정된 값을 고정하는 `repro/tests/test_decisions.py` 추가. 이전 판은 `repro/.backup_decision_check_20261012/SPEC_LEDGER.md`.

## 외부 수치 확인 대장 (2026-10-10 신설)
규칙: 외부 수치는 이번에 직접 확인한 파일:줄 또는 URL과 함께 적는다. 확인하지 못한 것은 "미확인"으로 적고, 직접 확인하기 전에는 지우지 않는다. 대화·요약·서브에이전트 보고에서 온 수치는 여기서 대조한 뒤에만 사실로 쓴다.

| # | 수치 | 의미(원문 근거) | 출처 | 확인일 | 상태 |
|---|---|---|---|---|---|
| X-1 | InternVL3.5-1B MathVista(mini) 59.3 | 단일 응답. "unless otherwise specified, the experimental results reported in Section 3 are obtained without applying TTS". Thinking 모드 사용 여부는 본문·모델 카드에 명시 없음(TTS 정의에 deep thinking 포함 → 문자 그대로는 '생각 모드 없음'으로도 읽힘) | `REF/InternVL3.5.txt` L555-561, L775; 모델 카드(huggingface.co/OpenGVLab/InternVL3_5-1B) | 2026-10-10 | 확인(thinking 여부만 미확인) |
| X-2 | 같은 모델 "w/ Parallel Thinking [143]" 69.5 | Best-of-N, VisualPRM-v1.1 critic이 여러 응답 중 하나 선택(다수결 아님). N은 본문에 없음. thinking 점수가 아님 | `REF/InternVL3.5.txt` L566-568, L776 | 2026-10-10 | 확인(N 미확인) |
| X-3 | 공식 8B 78.4 vs 논문 ST 8B 68.1 | 논문 ST가 공식 단일 응답 수치보다 모든 크기에서 10pt 이상 낮음(위 A-6) | `REF/InternVL3.5.txt` L793-794; `LAY` L899-903 | 2026-10-07(A-6), 2026-10-10 재확인 | 확인 |
| X-4 | 논문의 최종 답 처리 | "Sample y ∼ πθ(· \| xinput, τ1:k)"만 기술. 토큰 상한·잘림 처리·답 재촉 문구 없음. 우리 규칙 ①(s1 `\nFinal Answer:`)·②(DEER af)는 논문 외 처리 | `VisRef_plain.txt` Alg. 1 L16-17(파일 L528-530) | 2026-10-10 | 확인 |
| X-5 | DEER(Yang et al. 2025, arXiv 2504.15895)는 VisRef 인용 문헌이 아님 | VisRef 본문에 "DEER"·"early exit" 없음. 정지 신뢰도 측정의 빈 곳을 메우려고 우리가 고른 참고 구현 | `VisRef_plain.txt` grep 0건; `SRC/cited_code/deer` | 2026-10-10 | 확인 |
| X-6 | 생각 단계당 생성 텍스트 토큰 평균 ∼615 (시각 토큰 ∼1,772) | InternVL3.5-8B, MathVista. 평균 사용량이며 상한이 아님 | `VisRef_plain.txt` L383-385 | 2026-10-10 | 확인 |
| X-7 | 병렬 사고 예산 14K thinking tokens | MMStar, InternVL3.5-8B, Fig. 3 비교 문장. Fig. 3 가로축은 누적 thinking token 예산 B (Σ\|τ(i)\| ≤ B) | `VisRef_plain.txt` L698-702, L717 | 2026-10-10 | 확인 |
| X-8 | 토큰 상한(max new tokens) 값 | 본문·부록 어디에도 없음. App. I "same decoding setup and token budget"도 값 없음. 1B 토큰 사용량도 없음 | `VisRef_plain.txt` 검색(token·budget·length); SPEC_LEDGER 11·12행 | 2026-10-10 | 확인(없음) |
| X-9 | VLMEvalKit `InternVL3_5-1B-Thinking` 최대 생성 65,536 | `max_new_tokens=2**16`. 공식 59.3이 이 설정으로 나왔는지는 미확인 | `SRC/vlmevalkit_postprocess_54a063c5/vlmeval/config.py` L1336-1338 | 2026-10-10 | 확인(59.3 연결은 미확인) |
| X-10 | CVPR 2026 채택본·보충자료 확인: 공식 코드 링크 없음, 보충자료 A–K 구성은 arXiv 부록과 같고 1B 46.1 / 48.5 / 52.0, 2B 52.9 / 53.7 / 58.1, 8B 68.1 / 73.9 / 79.3. 토큰 상한·z_k 표현(층)·계수 부족 처리·답 추출 방식·온도는 여전히 미기재 | 새 정보 없음(미지정 항목 그대로) | https://openaccess.thecvf.com/content/CVPR2026/supplemental/Ghosal_VisRef_Visual_Refocusing_CVPR_2026_supplemental.pdf ; https://cvpr.thecvf.com/virtual/2026/poster/39354 | 2026-10-10 | 확인 |
| X-11 | Ghosal(VisRef 인용 [19]) 정지 엔트로피 추정과 기준값: "after appending each “Wait”, we sample multiple answers y ∼ π(·|z1:i, x)", "entropy of the distribution over final answers". 표본 수 N·온도는 본문 추출본에 없음. 표준 생각(Wait 없음) 평균 엔트로피 GSM-8K 0.23, MATH-500 0.02 (R1-Distill-Qwen-1.5B, 프롬프트 평균) | `REF/Ghosal_DoesThinkingMoreAlwaysHelp.txt` L400-405, L409-413 | 2026-10-11 | 확인(N·온도 미확인) |
| X-12 | RunPod Pods 시간당 요금(표 1개 값, Community/Secure 구분 없음): RTX 4090 $0.89, RTX 5090 $1.19, L4 $0.59, L40S $1.09, A40 $0.59, RTX A6000 $0.59, A100 PCIe $1.79, H100 PCIe $2.89 | GPU 임대 비용 추정용. 표 값은 4090 페이지의 Secure 값과 같음 → Secure로 추정(미확인) | https://www.runpod.io/pricing | 2026-10-11 | 확인(구분은 미확인) |
| X-13 | RunPod RTX 4090: Community $0.34/hr, Secure $0.89/hr ("access RTX 4090 instances from $0.34/hr") | 비교 섹션 값. 시점·가용성에 따라 변동("Rates vary by instance type and availability") | https://www.runpod.io/gpu-models/rtx-4090 | 2026-10-11 | 확인 |
| X-14 | RunPod 최소 충전 "as little as $10"(선불카드는 거래당 $100), 잔액 $0이면 Pod 정지·네트워크 볼륨 없는 Pod은 삭제 | 결제·데이터 보존 조건 | https://docs.runpod.io/get-started/billing-information | 2026-10-11 | 확인 |
| X-15 | 네이버클라우드 L4(24GB) gp1l4-g3 시간 1,447원(VAT 별도), L40S gp1ls16-g3 4,309원 | 국내 KRW 과금 대안 | https://www.ncloud.com/product/compute/gpuServer | 2026-10-11 | 확인 |
| X-16 | USD/KRW 1,340.63 (2026-10-09 종가) | 원화 환산용(집계 사이트, 공식 기관 값 아님) | https://www.poundsterlinglive.com/history/USD-KRW-2026 | 2026-10-11 | 확인 |
| X-17 | Modal Starter "$30 / month free credits"; GPU 초당 T4 $0.000164, L4 $0.000222, A10 $0.000306, A100 40GB $0.000583, A100 80GB $0.000694, L40S $0.000542, H100 $0.001097; CPU $0.0000131/물리코어/초, 메모리 $0.00000222/GiB/초; non-preemptible 3x | 무료 크레딧으로 T4 재실행 비용 추정. 카드 등록 요건·최대 실행 시간은 미확인 | https://modal.com/pricing | 2026-10-11 | 확인 |
| X-18 | 네이버클라우드 신규 가입 할인 크레딧 100,000원, 발급일부터 3개월, 결제 수단 등록 후 신청, 다른 할인 받았으면 불가, 제외: 무료/체험 상품·마켓플레이스·NAVER WORKS | GPU 서버 적용 여부는 명시 없음(미확인) | https://www.ncloud.com/main/creditEvent?tab=2 | 2026-10-11 | 확인(GPU 적용 미확인) |
| X-19 | RunPod RTX 3090 Community $0.22/hr, Secure $0.5/hr | 비교 섹션 값 | https://www.runpod.io/gpu-models/rtx-3090 | 2026-10-11 | 확인 |
| X-20 | Vast.ai RTX 3090 1장 온디맨드 오퍼 스냅샷 23건(잘림): dph_total 최소 0.1222(Quebec), 중앙 0.1563, reliability2 0.974–0.999 | 실시간 시장가, 스토리지 포함 여부 미확인 | https://console.vast.ai/api/v0/bundles/?q={"gpu_name":{"eq":"RTX 3090"},"num_gpus":{"eq":1},"rentable":{"eq":true}} | 2026-10-11 | 확인(시점 값) |
| X-21 | MathVista 모델 크기별 정확도 표(InternVL-1B 46.1 / 48.5 / 52.0, 2B 52.9 / 53.7 / 58.1, 8B 68.1 / 73.9 / 79.3)의 표 번호는 **Table 6** ("Table 6. Accuracy (%) across model scales on MathVista.") | 캡션이 표 아래에 붙는 형식. 바로 위 "Table 5. Latency per prompt on MathVista."는 ST 7.1s / TSR 7.7s / Look-Back 7.6s / VisRef 8.2s 지연 시간 표의 캡션. `VisRef_layout.txt` L898-900에서 Table 5 캡션이 정확도 표 헤더 옆에 찍히는 것은 2단 레이아웃 추출의 배치 효과 | `references/papers/VisRef.pdf` 14쪽(렌더링해 육안 확인); `pdftotext -layout` 14쪽 L8, L16 | 2026-10-11 | 확인 |
| X-22 | 시각 토큰 전부 재주입은 "naive solution"이며 "computationally prohibitive"로 배제(해롭다는 서술 없음). 8B MathVista에서 시각 토큰 ∼1,772 vs 단계당 텍스트 ∼615, 전부 붙이면 지연 2.3배 | `VisRef_fulltext/VisRef_plain.txt` L300, L381-385 | 2026-10-11 | 확인 |
| X-23 | 선택 수 강제 "enforcing \|Vk\| = m", "We terminate after m iterations" | `VisRef_fulltext/VisRef_plain.txt` L536, L578 | 2026-10-11 | 확인 |
| X-24 | Fig. 4(b) 본문 "76.1% to 79.2% as m increases from 20% to 30% but plateaus for m ≥ 30%"(8B) / Table 7 Random 67.3, Relevance-only 75.6, DPP 79.3(8B MathVista) — A-8·A-9와 같은 값, 이번 plain 추출본 줄 번호 | `VisRef_fulltext/VisRef_plain.txt` L794-795, L930; L1579-1590 | 2026-10-11 | 확인 |
| X-25 | MathVista 공식 채점의 빈 추출 처리: `normalize_extracted_answer(..., ignore_empty_extractions=False)` 기본값에서는 빈 객관식 추출도 최근접 선택지로 정규화, `--ignore_empty_extractions`(store_true)이면 None(오답). 구현 v4(2026-10-11)는 이 플래그를 켠 동작(본 표 46행) | `references/sources/mathvista_official_r4/evaluation/calculate_score.py` L28, L44, L121, L184 | 2026-10-11 | 확인 |
| X-26 | 공식 InternVL3-1B MathVista(mini) 45.8 (같은 표의 InternVL3.5-1B 59.3) | 논문 ST 46.1과 수치상 가까움(행 1, C-8). 2026-10-12 사용자 판단으로 README·REPORT에서 InternVL3 후보 삭제: 논문이 이름을 밝힌 InternVL 세대는 3.5뿐(LAY에 "InternVL3-" 0회), 모델을 thinking 모드로 실행(LAY L428-429), 1B·2B·8B를 "same decoding setup and token budget"으로 비교(L924-925), 8B 행 = Table 1 InternVL3.5-8B(L396-398) | `REF/InternVL3.5.txt` L774-775 (Table 3) | 2026-10-12 | 확인 |
| X-27 | MathVista 답 추출기: GPT-4 같은 LLM 기반, 예비 200개에서 GPT-4 추출 정확도 99.5% 초과 | 공식 채점은 모든 응답에서 LLM으로 최종 답을 뽑음 | `REF/MathVista.txt` L465-468 | 2026-10-12 | 확인 |
| X-28 | MathVista 공식 추출 프롬프트: 예시 5개(pids 852, 104, 824, 506, 540) 뒤에 새 응답, "Please read the following example. Then extract the answer from the model response and type it at the end of the prompt." | 우리 LLM 보조 추출이 쓰는 프롬프트 | `references/sources/mathvista_official_r4/evaluation/prompts/ext_ans.py` L3, L5-6, "Extracted answer:" 5회 (공개 URL: SOURCES.json, 커밋 c5acd37d) | 2026-10-12 | 확인 |
| X-29 | VLMEvalKit `InternVL3_5-1B-Thinking` `max_new_tokens=2**16`(=65,536) — X-9와 같은 값, 이번 줄 번호 | 공식 평가의 생성 상한 | `references/sources/vlmevalkit_postprocess_54a063c5/vlmeval/config.py` L1336-1338 | 2026-10-12 | 확인 |
| X-30 | VisRef의 H 정의: Alg. 1 12행 "Hk ← − Ey∼πθ [log πθ (y \| xinput , τ1:k )]", 본문 "entropy of its response distribution" / Ghosal: "we sample multiple answers", "entropy of the distribution over final answers" | 우리 H는 같은 양의 단일 표본 추정(J-9) | `LAY` L325, L375-377; `REF/Ghosal_DoesThinkingMoreAlwaysHelp.txt` L400-405 | 2026-10-12 | 확인 |
| X-31 | 기여·AI 사용 공개 관행: CRediT 14개 역할(ANSI/NISO Z39.104-2022); ACL 2023: 코드 도우미 사용은 README·저장소에 범위와 함께 밝힘; NeurIPS 2025: LLM이 방법의 핵심 일부일 때만 공개 의무(글쓰기·편집은 불필요) | README "Contributions and AI use" 절의 근거 | https://casrai.org/credit ; https://2023.aclweb.org/blog/ACL-2023-policy/ ; https://neurips.cc/Conferences/2025/LLM | 2026-10-12 | 확인(WebFetch 요약 기준, 원문 전체 대조 아님) |
| X-32 | 논문 Table 5 문항당 지연 시간: ST 7.1s, TSR 7.7s, Look-Back 7.6s, VisRef 8.2s (InternVL-3.5-8B, MathVista, H100 1장). 본문: DPP 선택은 TSR 대비 0.5초, ST 대비 1.1초 추가 | 단계마다 답을 여러 개 생성했다면 나올 수 없는 차이 → H를 답 1개로 재는 근거(사용자 판단, 2026-10-12) | `LAY` L892-895(표), L909-913, L971-972 | 2026-10-12 | 확인 |
| X-33 | MathVista 공식 추출 순서: 빈 응답 → 응답 전체가 선택지와 같으면 그대로 → 정수/실수로 바로 바뀌면 그대로 → (옵션 `quick_extract`) 'The answer is "…"' 정규식 → 그 밖은 LLM(GPT, 예시 5개 프롬프트) | "MathVista는 모든 답을 LLM으로 추출" 문구 정정 근거 | `references/sources/mathvista_official_r4/evaluation/extract_answer.py` L29-78 | 2026-10-12 | 확인 |
| X-34 | VLMEvalKit MathVista: `post_check(prefetch=True)`(객관식 `can_infer` 선택지 찾기, 숫자 직접 변환) 성공 시 그대로, 실패하면 GPT 추출(최대 5회 재시도) | 같은 정정의 근거 | `references/sources/vlmevalkit_postprocess_54a063c5/vlmeval/dataset/utils/mathvista.py` L77-125 | 2026-10-12 | 확인 |
| X-35 | InternVL 이미지 입력 형식: `<img>` + `<IMG_CONTEXT>`×(num_image_token×조각 수) + `</img>`; 타일링: `dynamic_preprocess(max_num=12, image_size=448)`, `load_image`는 `use_thumbnail=True`; 설정 `max_dynamic_patch` 12, `use_thumbnail` true, `force_image_size` 448, `patch_size` 14, `downsample_ratio` 0.5 → 조각당 (448/14)²×0.5² = 256토큰 | 재주입에서 태그를 뺀 것이 원래 형식과 다르다는 근거, 타일링 설명 근거 | `references/sources/internvl_modeling_internvl_chat.py` L216-217, L244, L295; `references/sources/internvl_README.md` L468, L506-509; `references/sources/internvl_config.json` L10, L13, L48, L61, L85 | 2026-10-12 | 확인 |
| X-36 | 논문 Table 7(8B): Random 67.3 / Relevance-only 75.6 / DPP 79.3(MathVista), "Random selection performs close to the ST baseline" — X-24와 같은 값, 이번 줄 번호 | visref_fill(관련성과 무관한 채움, 순위 상관 −0.10)을 "m개 관련 토큰" 시험으로 쓰지 않는 근거 | `VisRef_plain.txt` L1557, L1579-1583 | 2026-10-12 | 확인 |
| X-37 | s1 = VisRef 참고문헌 [34] "s1: Simple test-time scaling"(Muennighoff 외, arXiv:2501.19393); 논문은 샘플링 값(temperature·top-p)을 쓰지 않음(두 추출본 검색 0건), 사고 모드 실행만 명시(L428-429) | README 설정 표 근거 | `VisRef_plain.txt` L1105-1108; `LAY`·`VisRef_plain.txt` grep | 2026-10-12 | 확인 |
