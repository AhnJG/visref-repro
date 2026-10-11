# AUDIT REPORT: VisRef 재현 b_test 착수 전 독립 감사

- 대상: Appendix Table 6 "InternVL-1B" (MathVista testmini, ST 46.1 / TSR 48.5 / VisRef 52.0)
- 감사 대상: `/mnt/user-data/outputs/SPEC_LEDGER.md`(51행), `repro/` 코드(해시가 `kaggle/runs/b_test/run_config.json`의 `source_sha256`와 8개 파일 모두 일치함을 확인), b_test 구성 = run.py DEFAULTS + `--methods st,tsr,visref_raw_rank --think-cap 1024 --total-think-cap 10240`, Kaggle T4, timeout 4200 s(time_limit 2800 s, deadline 3600 s), 1,000문항 seed 0 셔플
- 작성일 2026-10-07. 근거는 원본만 썼다. 경로 약어: `LAY`=VisRef_layout.txt, `REF`=references/papers/ref_fulltext, `SRC`=references/sources, `OUT`=/mnt/user-data/outputs/ref_texts, `main_s0`=kaggle/runs/main_s0/output/results/results_{0,1}.jsonl(코드 버전 `8aaa689c94e9`, 179행 = 60 pid × st/tsr/visref_raw, visref_raw 59)

---

## 0. 결론 요약

**판정: 작은 수정 후 실행.**

- 막는 항목(실행 전 처리)
  - **B1. VisRef 쪽이 rank 한정 주입(`visref_raw_rank`) 하나뿐이다.** 논문은 |V_k| = m을 명시했다(LAY L341, L360-362). Fig. 4(b)와 Kulesza & Taskar(인용 [28])의 rank 한계를 함께 보면, 논문이 실제로 넣은 토큰은 m개에 가까웠을 가능성이 높다. b_test는 입력 임베딩 Z를 쓰고 rank까지만 고르므로 단계마다 약 100~200개(|V|의 6~11%)만 넣게 된다(main_s0에서 n_dpp 중앙값 98, m 중앙값 537, 단계의 74.6%가 m 미만). 결과가 나빠도 원인이 VisRef인지 주입량인지 가를 수 없다. `visref_raw`(m까지 채움)를 4번째 method로 추가하라. step-1을 fork하므로 추가 비용은 이어가기 단계와 답 생성 정도다.
  - **B2. 목적을 실행 전에 정해 두라(코드 수정 없음).** 4200 s 세션에서는 완료되는 문항이 수십~100여 개에 그칠 가능성이 크다(main_s0는 60 pid를 완료했고 세션 길이 기록은 없음). main_s0의 불일치 비율로 계산하면 짝지은 차이의 SE는 n=60에서 약 4.7pt다. 목표 델타(+2.4/+3.5/+5.9)를 판정할 수 없다. b_test를 "진단용 파일럿"으로 명시하고, 읽을 진단 지표와 기준을 미리 정해 두라(5절).
- 막지 않는 항목(문서화)
  - ST/TSR 예산 비대칭이 있다. ST는 1024토큰에서 잘리고, TSR/VisRef는 잘린 단계를 최대 10,240까지 이어간다.
  - main_s0에서 이어가기는 상한에 잘린(또는 EOS로 끝난) 단계 뒤에서만 일어났다. 자연스럽게 닫힌 단계의 probe는 63개 중 63개가 H < δ였다.
  - H 추정기가 단일 표본·토큰 단위다. 논문이 말하는 대상은 "answer distribution"이다.
  - s1 강제 답 문자열이 원본과 다르다.
  - top-p 근거가 약하다.
  - fp16 T4를 쓴다.
  - summarize.py 주석 "VisRef 4.2-7.3x"는 Fig. 3을 잘못 읽은 것이다.
  - 코드에서 크래시나 논리 버그는 찾지 못했다.

---

## 1. 원본 확인 범위

- 논문 16쪽 PNG를 전부 직접 봤다. 고해상도본(p.6, p.7, p.14)과, 확대 크롭한 Fig. 2, Fig. 3(b), Fig. 4도 확인했다. layout 텍스트 L1-1049는 정독했다.
- 인용 원문
  - Ghosal(REF): TTBC 정의와 엔트로피 측정(L151-170, L386-433), 부록 D·E
  - s1: 논문(OUT/2501.19393 L412-453)과 코드(`s1/vllm_causallms.py` L230-414, `s1/README.md`, `eval/commands.sh`)
  - Chen: OUT/1709.05135(§3, Alg. 1, Supp. B)와 `chen_fast-map-dpp/dpp.py`
  - CEIL: `dpp_map.py`, `dense_retriever.py`
  - Kulesza & Taskar(OUT/1207.6083 L845-852)
  - InternVL3.5(REF L330-350, L555-575, L740-830)
  - 모델 카드, 모델 코드(`chat`, `extract_feature`), 공식 MathVista 평가 코드
- 추가로 VLMEvalKit `vlmeval/vlm/internvl/internvl_chat.py`를 alphaXiv GitHub 리더로 읽었다(main 브랜치). 모델 카드 L80이 "We conduct the evaluation with VLMEvalkit"라며 이 파일을 직접 가리킨다.
- HF의 `generation_config.json`, `config.json`은 프록시가 403으로 막아 확인하지 못했다(미검증).
- 수치 재검증(scratchpad)
  - Chen과 CEIL 원본 DPP 코드를 float64, float32, bf16 모사 커널로 실행했다.
  - main_s0 JSONL로 진단 지표를 다시 계산했다.
  - Fig. 3(b)는 픽셀 좌표로 다시 읽었다.

---

## 2. 과제 1: 장부 검증. 상태나 근거가 틀렸거나 불완전한 행

장부가 단 줄 번호와 코드 file:line 인용은 표본 점검에서 모두 맞았다(행 1, 2, 4, 6, 8, 10, 11, 16-18, 21-24, 29-31, 33, 35-37, 40, 42-46, 49-51). 아래는 내용에 문제가 있는 행이다.

| 행 | 문제 | 바로잡은 상태와 근거 | 심각도 |
|---|---|---|---|
| **37** (rank<m 처리) | 상태 "인용 구현에서 확인"은 과하다. 논문은 "enforcing \|Vk\| = m"(L341), "We terminate after m iterations"(L360-362)라고 **명시**한다. rank<m인 경우는 **미명시**다. 인용 코드 두 개는 rank<m 상황을 위해 만든 것이 아니다. Chen의 합성 실험은 "f_i in R^D with D same as the total item size M"(OUT/1709.05135 L47)이라 full rank이고, CEIL은 후보 100개 중 소수를 고른다. 그래서 인용 코드는 "저자가 무엇을 했는가"의 근거가 되지 못한다. DPP 문헌 [28]은 "One immediate consequence is that \|Y\| cannot be larger than rank(L)."(OUT/1207.6083 L851-852)라고 한다. 즉 논문의 목적식 자체가 rank ≥ m을 전제한다. | \|V_k\|=m은 **명시**, rank<m 처리는 **가정(미명시)**. 장부의 수치 주장은 재현된다. 내 모사(N=1790, d=1024, 고유 z 토큰 150, m=537)에서 Chen은 float64일 때 147~150개, float32/bf16일 때 161~170개(rank + 소량의 잡음 선택)를 골랐고, CEIL은 537개를 반환하되 서로 다른 것은 161~165개였다. | 상 |
| **44** (잘린 뒤 강제 답) | "인용 구현에서 확인"은 과하다. s1은 TSR 인용([34])일 뿐이고, 논문은 강제 답·상한을 한 줄도 언급하지 않는다. ST와 VisRef에까지 적용한 것은 외삽이다. 문자열도 s1 원본과 다르다(4d 참조). s1은 `finish_reason == "length"`일 때만 적용하는데(L401) 코드는 EOS로 끝난 thinking에도 적용한다(`methods.py:L195, L223`). | **가정(미명시)**, s1은 참고 근거 | 중 |
| **22** (H 추정) | "본문·부록에 추정법 서술 없음"과 검색 기록 S-D가 핵심 문구를 놓쳤다. 논문은 Fig. 2 캡션에서 "until the entropy of the model's answer distribution falls below a confidence threshold"(L252-253), 본문에서 "converged to a confident answer distribution"(L380)라고 쓴다. Fig. 2의 "Answer:" 상자 아래에는 여러 막대로 된 히스토그램이 있다. step 1은 퍼진 분포, step k는 막대 하나가 지배한다. 인용한 Ghosal(VisRef 제1저자)은 "We quantify policy uncertainty using the entropy of the distribution over final answers"(REF/Ghosal L404-405)라고 쓴다. 장부는 L400-402만 인용했다. | "답 수준 분포의 엔트로피"라는 점은 **그림·인용 문헌에서 유도**. 표본 수와 온도만 **가정**. 코드의 단일 표본 토큰 엔트로피 합은 다른 추정기다(3절 D3). | 상 |
| **29/30** (V, Z 표현 공간) | 가정 상태는 맞다. 다만 내부 제약을 빠뜨렸다. (i) K&T의 \|Y\| ≤ rank(L). (ii) Fig. 4(b)는 m 20%(77.3 막대 / 본문 "76.1%")에서 30%(79.2)로 오른 뒤 40·50%에서 평탄하다. 입력 임베딩 Z라면 rank ≤ z_k의 고유 토큰 수다. main_s0에서 565~867토큰 단계의 고유 토큰은 128~200개였다. 그러면 m=20%(≈354)부터 이미 포화해 20%와 30%의 차이가 생기지 않아야 한다. 반대로 문맥 의존 표현(예: hidden state)이면 rank ≈ T_k ≈ 615(L263)이고, 20%→30%는 다르고 40·50%(≥615)는 포화한다. 그림의 패턴과 맞는다. | 가정 유지. "입력 임베딩 Z + rank 한정 선택" 조합은 Fig. 4(b)와 **긴장**한다고 명기할 것(근거 강도는 중간: 20%-30% 차이 1.9~3.1pt는 잡음 수준에 가깝다) | 상 |
| **1** (체크포인트) | 결론은 맞지만 근거가 불완전하고, 일부는 근거 없이 단정한다. C-8의 "명시적 3.5 표기(8B 행)"는 부정확하다. Table 6 8B 행의 이름은 "InternVL-8B"이고, 3.5라는 결론은 수치가 같다는 데서 유도한 것이다. 빠진 근거: Table 7 "(InternVL-8B)"(L981)과 Table 8 "(InternVL-8B)"(L987)은 InternVL3.5-8B와 같은 값(79.3/44.6/63.1)에 쓰였다. 즉 논문은 "InternVL-NB"를 3.5의 약칭으로 쓴다. 참고문헌의 InternVL은 [44], [45] 둘 다 InternVL3.5이고 InternVL3 인용은 없다(L655-688). 대안 열의 "InternVL3-1B(thinking 비지원)"는 제공된 소스에 근거가 없다(3.5 모델 카드만 thinking을 문서화함, SRC README L410-412). | 그림·표에서 유도(유지). 근거 보강. 4a 참조 | 중 |
| **4** (T, top-k, top-p) | top-k 50·top-p 0.95의 근거로 든 L767은 thinking 권장이 아니다. `'describe this image'` 다회 대화용 lmdeploy 예제다. thinking 권장은 "do_sample=True and temperature=0.6"뿐이다(L80, L412). 더 나은 근거가 있다. VLMEvalKit `InternVLChat.generate_v2`는 `kwargs_default['temperature'] = 0.6`, `kwargs_default['top_p'] = 0.95`를 설정한다(do_sample은 설정에 따름). 모델 카드의 HF 경로(`chat()` + `dict(max_new_tokens=1024, do_sample=True)`, L527)라면 top_k=50(HF 기본), top_p=1.0(HF 기본)이 된다. generation_config.json은 미검증이다. | T=0.6은 **인용 구현에서 확인**. top-k/top-p는 **가정(부분 근거: VLMEvalKit)** | 하 |
| **6** (tiling) | VLMEvalKit `set_max_num`은 MathVista에 `max_num = 6`을 쓴다(res_12 목록에 없음). 모델 카드 기본값 12와 충돌하는데, 장부는 이를 기록하지 않았다. 논문의 "∼1,772 visual tokens"(L263)가 이 충돌을 판정해 준다. main_s0(max_num=12)의 평균 \|V\|는 1,774.9(60문항, 중앙값 1,792)로 일치한다. max_num=6이면 같은 이미지들로 추정한 평균이 약 1,250이다. | **그림·표에서 유도 + 인용 구현**(max_num=12가 더 강하다). 장부 A-1의 "원자료 미확인"은 이제 해소됐다. | 하 |
| **11/12** (step·총 상한) | 가정 상태는 맞지만 관련 근거를 빠뜨렸다. Fig. 3의 ST ★는 InternVL MathVision ≈900, Qwen3-VL MathVision ≈1,000, SAIL MathVision ≈850 토큰이다. 경쟁 수학에서 thinking 모델의 평균이 ~1K라는 점은 ~1K 상한이 있었을 가능성을 시사한다(간접). 모델 카드 L527의 `max_new_tokens=1024`는 **thinking과 답을 합한 전체 응답** 상한이다. 반대 방향 근거도 있다. Ghosal TTBC1은 "apart from the model's inherent maximum token limit (32K)"(L157-158)로 step 상한이 없다. | 가정 유지. 위 근거를 기록 | 중 |
| **19** (ST 정의) | "명시 / 일치"라고 했지만, 코드의 ST는 잘리거나 EOS로 끝나면 `</think>\n\nFinal Answer:`를 삽입한다. 논문의 "without additional intervention"(L798-800)과 부분적으로 어긋난다. 불가피한 운영 선택이지만 "일치"는 과하다. | 명시(정의). 구현은 **부분 일치** | 중 |
| **24** (log 밑) | "수식은 log만 쓴다"는 틀렸다. 논문은 부록 G에서 "Finally, taking the natural logarithm yields"(L944)라고 log의 의미를 밝힌다. 다른 식(Eq. 18)이지만 논문의 log 관례를 보여 준다. | **본문에서 약하게 유도**(nat) | 하 |
| **18** (잘린 step 구분자) | s1은 텍스트가 `\n`으로 끝나지 않을 때만 `\n`을 넣는다(L404-409). 코드는 잘리면 무조건 `\n`을 넣는다(`methods.py:L270`). 또 s1은 길이 상한으로 끝나면 더 이어가지 않는다("Final; do not generate further", L350-351). 장부는 후자를 적었지만 전자의 불일치는 적지 않았다. | 가정 유지, 불일치 기록 | 하 |
| **2** (dtype) | 보강할 근거가 있다. VLMEvalKit InternVLChat도 `torch_dtype=torch.bfloat16`을 쓴다. | 가정 유지 | 하 |
| **8** (프롬프트) | 보강할 근거가 있다. VLMEvalKit의 R1(thinking) 경로는 질문 뒤에 `'Please answer the question and put the final answer within \\boxed{}.'`를 붙인다(`cot_prompt_version=='r1'`, USE_COT=1일 때). 코드는 HF `query`만 쓴다. "인용 구현에서 확인"보다는 가정에 가깝다(어느 프롬프트를 썼는지 논문이 말하지 않음). | **가정(미명시)** | 하 |
| 0절 5항 | "결과 JSONL 없음"은 이제 사실이 아니다. 재검증 결과, README의 "rank 부족 75%"는 74.6%(n=67 step)로 **확인**됐다. "ST forced 11/30 vs natural 22/30"도 **확인**(cap 29개 중 11 정답 + EOS 1개 중 0). "n_dpp 중앙값 60"은 smoke3의 값이고 main_s0는 **98**이다. | 갱신 | 하 |
| A-4 | "Fig 3(b) VisRef 첫 점 ≈4,000 (ST의 ≈6.5배)"는 첫 점을 C=1로 읽은 것이다. 픽셀로 다시 읽은 VisRef 점의 x는 ≈3,960, 6,040, 7,860, 9,970, 11,940, 14,310이고 간격은 일정하게 ≈2,070이다. 첫 점 ≈ 2×간격이므로 C=2..7, trace당 ≈2,000토큰(ST ★ ≈570의 **≈3.5배**)이다. 병렬 사고 점(≈1,830, 3,110, 4,400, …)도 ST 570×C와 맞는다. summarize.py L113 주석 "paper Fig. 3: VisRef 4.2-7.3x"도 같은 오독이다. | 그림 유도값 정정. Table 5와의 긴장은 여전하다(4c) | 중 |
| A-6 | 공식 InternVL3.5 수치와 비교할 때 조건을 빠뜨렸다. "unless otherwise specified, the experimental results reported in Section 3 are obtained without applying TTS"(REF/InternVL3.5 L558-559)이고, 같은 곳에서 TTS는 deep thinking을 포함한다(L557, L562). 공식 59.3이 thinking 모드 점수가 아닐 수 있다. | 비교 조건 추가 | 중 |
| A-10 | Table 4의 괄호값은 Table 1과 달리 ST 대비가 맞다(캡션과 일치). 다만 SAIL TallyQA는 73.9−69.3=4.6인데 "+5.4"로 적었다(L904, L913). 사소하다. | 기록 | 하 |
| C-8 | "명시적 3.5 표기" → "수치 동일성, 약칭 관례, 참고문헌에서 유도"로 고칠 것 | 정정 | 중 |

명시라고 표시했지만 실제로는 해석인 행: 19(구현이 개입함), 37(rank<m 처리는 미명시), 36(어느 greedy 구현인지는 미명시이나 정확한 greedy라면 고르는 결과가 같으므로 문제 없음). 그 밖의 명시 행(7, 9, 13, 20, 21, 25-27, 31-35, 41, 42, 47, 48)은 원문에서 확인했다.

---

## 3. 과제 2: 장부에 없는 구현 관련 서술

| # | 위치 | 원문 | 함의 |
|---|---|---|---|
| O1 | Fig. 2 캡션 L251-253; §4.1.2 L380 | "until the entropy of the model's answer distribution falls below a confidence threshold δentropy"; "converged to a confident answer distribution" | H는 **답 분포**의 엔트로피다(행 22) |
| O2 | Fig. 2 그림(p.4) | "Answer:" 프롬프트 상자, 그 아래 여러 막대 히스토그램(step 1은 퍼짐, step k는 한 막대 지배), "Answer: 600 ✗ / Answer: 1000 ✓" | 답을 여러 개 샘플링해 만든 경험 분포와 들어맞는다. probe 문자열의 유일한 단서는 "Answer:"다(코드는 `\boxed`를 쓴다) |
| O3 | Ghosal L404-405, L410-415 | "entropy of the distribution over final answers"; 수치 "from 0.23 to 2.79", "from 0.02 to 0.12", "0.12 to 3.0" | 답 수준 추정기. δ=0.25의 스케일 참고값 |
| O4 | OUT/1207.6083 L851-852 ([28]) | "\|Y\| cannot be larger than rank(L)" | 논문의 \|V_k\|=m은 rank ≥ m을 전제한다(행 37) |
| O5 | Fig. 4(b), L477-478, L529-531 | "Accuracy improves from 76.1% to 79.2% as m increases from 20% to 30% but plateaus for m ≥ 30%." | 실제 주입 수가 20%와 30%에서 달랐다. rank 한정 + 입력 임베딩 조합과 긴장한다(행 29/30/37) |
| O6 | Table 7 L981, Table 8 L987, L956-957 | "Token selection strategies (InternVL-8B)"; "Effect of λ weighting (InternVL-8B)" | "InternVL-NB" = InternVL3.5 약칭(4a) |
| O7 | 참고문헌 [44], [45] L655-688 | 둘 다 "Internvl3.5: Advancing open-source multimodal models…" | InternVL3 인용 없음(4a) |
| O8 | §6 L558-563 | "δentropy = 0.25 consistently yields the highest accuracy across all token budgets. A smaller threshold (e.g., 0.15) … leading to excessive reasoning steps" | 논문 추정기에서는 H가 0.15~0.30 구간에 상당히 있었다는 뜻이다. main_s0에서는 이 구간이 TSR 1/70, VisRef 3/67뿐이다 |
| O9 | Fig. 5(p.8), Fig. 7(p.16), Fig. 2 좌하단 | "After Reasoning Step #2", "#3"; 주입 시점의 attention 봉우리 2개 | 적어도 예시에서는 여러 단계 진행이 일어난다(4c) |
| O10 | Fig. 1(b)(p.1) | 텍스트 토큰 → "Visual cue"(원) → "Wait, think more" → … | 행 17의 순서를 Fig. 2 말고 그림 하나가 더 확인해 준다 |
| O11 | 부록 G L944 | "taking the natural logarithm" | log = ln 관례(행 24) |
| O12 | REF/InternVL3.5 L557-562 | "unless otherwise specified … obtained without applying TTS"; "Deep Thinking. By activating the Thinking mode" | 공식 수치는 thinking 모드가 아닐 수 있다(A-6, 4a) |
| O13 | VLMEvalKit internvl_chat.py(모델 카드 L80이 지정) | R1 경로: `cot_prompt = 'Please answer the question and put the final answer within \\boxed{}.'`; `set_max_num`: MathVista → `max_num = 6`; `generate_v2`: temperature 0.6, top_p 0.95; `kwargs_default = dict(do_sample=False, max_new_tokens=4096, top_p=None)`; bfloat16 | 공식 평가 설정. `\boxed` 형식의 근거. max_num은 논문의 1,772와 충돌하므로 논문은 12 쪽(행 6) |
| O14 | Ghosal L157-161 | "no explicit budget constraint … apart from the model's inherent maximum token limit (32K)"; "whenever the model attempts to generate the end-of-thinking delimiter (</think>), we suppress it, append the token "Wait"" | 제1저자 선행연구의 TSR은 step 상한 없이 **모델이 스스로 끝내려 할 때** Wait를 붙인다(행 11, 18) |
| O15 | s1 `vllm_causallms.py` L350-351, L381-384, L414 | length로 끝나면 "Final; do not generate further"; thinking 총량을 사후 절단; 답은 `max_tokens` 전체로 생성 | 4d |
| O16 | §5 L384-388(오른쪽 단) vs Table 1 | "gains ranging from 0.1% to 2.1% on most benchmarks" — 예외는 InternVL3.5-8B MathVista TSR +5.8 하나뿐 | 8B 행의 TSR 이득이 이례적으로 크다. 1B·2B(+2.4/+0.8)는 범위 안이다 |
| O17 | Table 4(p.14) | 괄호값은 ST 대비. SAIL TallyQA 산술 오류 | A-10 보완 |

---

## 4. 과제 3: b_test 구성에서 코드 대 장부 대 논문

### 4.1 장부대로 정확히 구현된 것(코드 확인)

- R1 system prompt: 모델 카드와 VLMEvalKit의 문자열과 같다(`model.py:L6-14`).
- 공식 dynamic tiling(max_num 12, thumbnail), `extract_feature` 출력을 V로 사용.
- T=0.6 샘플러는 temperature → top-k → top-p 순서다(HF와 같음).
- Kmax=10, δ=0.25(nat), m=⌊0.3|V|⌋.
- M_k는 현재 z_k만으로 만들고 L = (VZᵀ)(VZᵀ)ᵀ다. Chen 재귀를 float64로 수행한다.
- V_k는 z_k 뒤에 붙고, probe와 최종 답 둘 다 V_k 뒤에 온다. Alg. 1의 10→12→17행과 같다. 즉 VisRef가 k=1에서 멈춰도 최종 답은 V_1에 조건부다.
- 이전 V_i를 유지하고, 순서는 z_k → V_k → "Wait, think more" → z_{k+1}이다(Fig. 1(b)와 Fig. 2에 맞음).
- 최종 답은 probe와 별도로 샘플링한다. 같은 pid의 method들은 CRN과 step-1 fork를 공유한다(분포는 그대로).

### 4.2 가장 강한 근거에서 벗어난 곳

| ID | 내용 | 근거 | 영향 |
|---|---|---|---|
| **D1** | `visref_raw_rank`는 n_inject = n_dpp ≤ rank(L) ≤ min(rank V, z_k 고유 토큰 수)이다(`methods.py:L204-208`, `dpp.py:L38-54`). main_s0에서 n_dpp 분위(0/10/25/50/75/90%)는 4/61/76/98/196/284였고 최대 379, m 중앙값은 537이었다. n_dpp = n_unique_z가 거의 항상 성립했다. b_test는 step이 1024토큰 이하라 고유 토큰이 더 적어 약 100~200개만 넣을 것이다(\|V\|의 6~11%, 논문의 최저 ablation 20%보다 낮다). | L341, L360-362(명시), [28] L851-852, Fig. 4(b) | **상(B1)** |
| **D2** | 예산 비대칭과 잘린 뒤 이어가기. main_s0의 step-1 길이로 추정하면 1024 상한에서 ST의 39/60(65%)가 잘리거나 EOS로 끝난다. 여기에는 1.4K~2.9K 토큰에서 **자연스럽게 닫힌 9건(cap 4096에서 8/9 정답)**이 들어 있다. ST는 강제 답으로 끝나고, TSR/VisRef는 probe가 불확실하면 `\nWait, think more`로 이어간다. main_s0에서는 **자연스럽게 닫힌 단계의 probe 63/63이 H<δ**였다(TSR 32/32, VisRef 31/31, H 최댓값 0.063). 이어간 18건은 모두 step 1이 cap(17) 또는 EOS(1)로 끝난 경우였다. 결국 b_test의 TSR−ST는 주로 "잘린 trace에 예산을 더 준 효과"를 잰다. 논문의 TSR은 모델이 끝내려 할 때 이어간다(§3 L203-206, Ghosal L159-161). s1은 length 뒤에 이어가지 않는다(L350-351). | 4d, O14, O15 | 중(해석 위험). 분석을 계층화해야 한다 |
| **D3** | H 추정은 `</think>\n\n\boxed` 뒤 답 1개를 T=1로 뽑고 토큰 엔트로피를 합한다(닫는 `}`류는 병합). 논문과 Ghosal은 "answer distribution / distribution over final answers"를 말한다(O1-O3). 기댓값은 같은 H(Y)를 겨누지만 차이가 있다. 표면형("1000"/"1,000")이 다르면 다른 답으로 센다. 단일 표본이라 분산이 크다. Fig. 2의 "Answer:" 단서와도 다르다. main_s0 H 분위는 TSR [0, 0, 0.002, 0.125, 0.786]이다. | 행 22, 23 | 중 |
| D4 | 강제 답 문자열과 조건이 s1과 다르다(4d) | s1 L254, L401-412 | 하 |
| D5 | ST에 강제 답이라는 개입을 넣는다(행 19) | L798-800 | 하~중(ST 수준에 영향) |
| D6 | top-p 0.95, fp16(T4) | 행 2, 4 | 하. main_s0의 non-finite 오류는 0건 |

### 4.3 버그 점검

b_test 경로(run.py → Job 상태 머신 → Engine → select_tokens → score → summarize)를 따라가며 다음을 확인했다. **크래시나 결과를 오염시키는 논리 버그는 찾지 못했다.**

- fork 시 생성기 상태 복원
- probe_gen 분리
- `_think_cap`과 budget 판정: 총 10,240 = 10×1,024라 k<10에서 `budget`은 발동하지 않는다
- deadline, partial 처리
- 셔플 순서와 샤드 일치(`summarize.complete_prefix`)
- `lumped_entropy` 식
- left-padding position id
- fast-KV 마스크

사소한 항목:

1. `methods.py:L270`의 `cut_sep`는 무조건 들어간다. s1은 조건부다.
2. `model.py:L111` `close_ids`가 `}}`나 `},` 같은 토큰도 "닫기만 함"으로 병합한다. 영향은 무시할 만하다.
3. `scoring.normalize`는 None을 `''`로 바꾸지만 공식 코드는 `str(None)`, 즉 `'None'`이다. 무시할 만하다.
4. `summarize.py:L113` 주석의 "VisRef 4.2-7.3x"는 오독이다(A-4).
5. summarize.py에 **step-1 종료 유형(close/cap/eos)별 정확도와 쌍 비교**, **자연 종료 probe 중 H≥δ 비율**이 없다. JSONL에 `steps[0].think_end`, `H`가 기록되므로 사후에 계산할 수 있다.

---

## 5. 과제 4: 쟁점 판정

### 4a. "InternVL-1B"의 정체 → **InternVL3.5-1B(`OpenGVLab/InternVL3_5-1B`, CascadeRL 최종본), 신뢰도 약 85%**

- 3.5를 지지하는 근거
  - ① 논문은 "InternVL-8B"라는 약칭을 InternVL3.5-8B 값에 그대로 쓴다. Table 6 8B 행이 Table 1과 같고, Table 7/8 "(InternVL-8B)" = 79.3/44.6/63.1이다. 같은 표의 1B·2B 행이 다른 계열이려면 계열을 숨긴 채 섞어야 한다.
  - ② 참고문헌의 InternVL은 3.5뿐이다([44], [45]).
  - ③ 모든 모델을 "reasoning (thinking) mode"로 평가했다(L428-429, L775-777). 제공 소스에서 thinking 모드를 문서화한 것은 3.5뿐이다.
  - ④ App. I는 "under the same decoding setup"이라고 한다.
- 3.5를 반대하는 근거
  - ST 46.1이 InternVL3-1B 공식값 45.8과 거의 같다.
  - 그러나 이 근접성은 약한 증거다. 논문 ST는 3.5 공식값보다 1B −13.2, 2B −18.9, 8B −10.3으로 세 크기 모두 일관되게 낮다. 3 계열로 보면 +0.3 / −4.1 / (8B는 어차피 3.5)로 패턴이 섞인다. 게다가 공식 3.5 수치는 TTS(=deep thinking) 없이 잰 것일 수 있다(O12).
- 남는 불확실성: 3.5의 변형(-Instruct, -MPO, -HF 형식) 가운데 어느 것인지.

### 4b. rank(L) < m 처리

- **논문**: \|V_k\| = m과 m회 반복을 명시하고, rank 부족은 언급하지 않는다. [28]에 따르면 m > rank이면 det = 0이므로, 논문의 서술은 rank ≥ m을 전제한다.
- **인용 코드**
  - Chen은 d² < 1e-10에서 멈춰 ≈rank개를 반환한다. 내 모사에서는 float64일 때 rank, float32나 bf16 커널일 때 rank + 7~13%였다. 처음에는 "float32 잡음 때문에 m까지 계속 고를 것"이라는 가설을 세웠는데, 실험 결과 기각됐다.
  - CEIL은 마스크 없이 m회 반복해, m개 인덱스를 반환하되 서로 다른 것은 ≈rank개다(중복 포함).
  - 두 코드 모두 원래 rank ≥ m인 상황용이다.
- **내부 증거**: Fig. 4(b)의 m 민감도와 30% 이후 평탄은 "주입 수 ≈ min(m, ~T_k)"와 맞는다. "주입 수 ≈ 고유 토큰 수 ≪ 0.2\|V\|"와는 맞지 않는다.
- **판정**: 논문은 실제로 **약 m개(적어도 0.2\|V\|를 넘는 수)**를 넣었을 가능성이 높다(신뢰도 약 65%). 방법은 rank가 충분한 표현(문맥 의존 hidden state 등)이었거나, 어떤 식으로든 m까지 채웠거나 둘 중 하나로 보인다.
- **b_test의 `visref_raw_rank`**: greedy가 고른 토큰만 넣는다(상대 임계 1e-9·max d²). 입력 임베딩 Z이므로 주입 수 ≈ z_k의 고유 토큰 수(main_s0 중앙값 98, m의 18%)다. 논문 서술과 Fig. 4(b) 양쪽에서 가장 멀다.

### 4c. 정지 통계 H_k

- **논문이 명시한 것**
  - H_k = −E_{y∼π(·\|x,τ1:k)}[log π(y\|x,τ1:k)](L325, L377)
  - "answer distribution"(L252-253, L380)
  - V_k를 붙인 **뒤에** 계산(Alg. 1의 10→12행, Fig. 2 화살표)
  - H_k < δ=0.25이면 정지, Kmax=10
  - log는 ln 관례(L944)
- **열려 있는 것**: 표본 수, 표본 온도와 top-p, probe 프롬프트("Answer:" 단서만 있음), 답 추출과 동치 묶음(clustering), 토큰 단위인지 답 단위인지.
- **k=1 정지 빈도에 대한 내부 제약**
  - ① Table 5(H100, 8B): ST 7.1 s, TSR 7.7 s(+8.5%), VisRef 8.2 s(+15.5%). 지연이 생성 토큰에 비례하고 이어가는 단계가 ~600토큰이라고 보면, probe 비용을 빼고 이어간 비율은 TSR 약 10% 이하, VisRef 약 10~20%다. 즉 **대부분 k=1에서 멈춘다**.
  - ② Fig. 3(b)를 다시 읽으면 VisRef는 trace당 ≈2,000 thinking 토큰으로 ST의 약 3.5배다. 시각 토큰을 세지 않으면 평균 k̄≈3.3, 세면(≈615+531/단계) k̄≈1.8이다. 어느 쪽이든 Table 5와 **충돌**한다.
  - ③ §6(L560-563)은 δ=0.15가 "excessive reasoning steps"를 낳는다고 한다. H 값이 0.15~0.30 근처에 상당히 있었다는 뜻이다.
  - ④ Fig. 2, 5, 7에는 2~3단계 예시가 있다.
  - 결론: 논문의 내부 증거는 서로 부딪히며, "대부분 k=1"(Table 5)과 "평균 2~3단계"(Fig. 3) 사이에서 판정할 수 없다.
  - 참고로 main_s0는 TSR 50/60, VisRef 51/59가 k=1이라 Table 5 쪽과 맞는다. 다만 이어가기가 모두 잘린 단계 뒤에서만 일어났으므로(D2), 자연 종료 뒤 "성찰"은 한 번도 일어나지 않았다.

### 4d. thinking이 상한에 닿았을 때: s1 공식 동작과 코드 비교

- **s1 공식 동작**(`s1/vllm_causallms.py`)
  - 끝 구분자는 `thinking_end = "<|im_start|>answer"`이고, `thinking_end_max = thinking_end + "\nFinal Answer:"`다(L240, L254).
  - `finish_reason == "length"`일 때: 생성 텍스트가 `"\n"`으로 끝나면 바로 `thinking_end_max`를 붙이고, 아니면 `"\n"` + `thinking_end_max`를 붙인다(L401-409).
  - 정상 종료면 `thinking_end`만 붙인다(L410-412).
  - Wait 루프에서 length로 끝나면 "Final; do not generate further", 즉 더 이어가지 않는다(L350-351).
  - `"Wait"`는 구분자 없이 바로 붙인다(L366-368). thinking 총량은 사후에 `max_tokens_thinking`으로 자른다(L381-384). 답은 `max_tokens` 전체로 생성한다(L414).
  - README 예시는 총 예산을 차감하고 "Wait"를 바로 붙인다. "Final Answer:"는 선택이다(README의 `# You can also append "Final Answer:"` 주석).
  - 논문은 "appending the end-of-thinking token delimiter and optionally "Final Answer:""(OUT/2501.19393 L414-416)라고 쓴다.
- **코드**
  - 잘린 텍스트에 `\n` 없이 바로 `</think>`를 붙이고 이어서 `"\n\nFinal Answer:"`(줄바꿈 **2개**)를 붙인다.
  - 이 처리를 length뿐 아니라 **EOS 종료**에도 적용한다.
  - 답은 최대 1,024토큰이다.
  - TSR/VisRef는 잘린 단계 뒤에 probe를 돌리고, H≥δ이면 항상 `"\n"` + `"Wait, think more"`로 **이어간다**. s1과 Ghosal에는 이 동작의 전례가 없다.
- 차이 가운데 정확도에 의미 있는 것은 "잘린 뒤 이어가기"(D2)이고, 나머지는 사소하다.

### 4e. b_test를 무의미하거나 낭비로 만들 요인

1. **검정력.** n이 수십~100여 개면 쌍 비교 SE가 약 4~5pt라 목표 델타를 판정할 수 없다(B2).
2. **VisRef 주입량.** 단일 rank 한정 arm이라 결과가 나빠도 해석할 수 없다(B1).
3. **TSR 측정 대상의 변질.** 자연 종료 뒤 H≈0이라 사실상 "잘린 trace 예산 연장"을 재게 된다(D2). 계층별 보고 없이 TSR−ST를 해석하면 오도된다.
4. **1B 반복 루프.** main_s0에서 4,096토큰 step의 고유 토큰이 36~130개인 경우가 많았다. 1024에서 잘린 루프를 "Wait"로 이어가면 예산만 쓴다. `n_unique_z / think_tokens`로 루프를 표시해 따로 집계할 것.
5. **δ 민감도.** H가 0.15~0.30 구간에 거의 없어(1/70, 3/67) 이 추정기로는 δ ablation이 거의 무의미하다(기록용).

---

## 6. 과제 5: 실행 판정

**작은 수정 후 실행.**

**막는 항목**

- **B1**: `--methods st,tsr,visref_raw_rank,visref_raw`로 m까지 채우는 arm을 추가한다. 근거는 LAY L341, L360-362, OUT/1207.6083 L851-852, Fig. 4(b) L477-478, main_s0의 n_dpp 분포다. 대안은 `visref_raw_reg`다. 장기적으로는 z_k를 문맥 의존 hidden state로 바꾼 변형을 검토하라.
- **B2**: b_test를 파일럿으로 명시하고, 사전 기준을 적어 둔다. 예: 형식 오류율, `final_forced_valid`, step-1 종료 유형 분포, 자연 종료 probe의 H≥δ 비율, n_inject/m, k 분포, method별 처리량. 정확도 델타를 판정하려면 다회 세션으로 1,000문항을 완주해야 한다.

**막지 않는 항목(문서화)**

- D2: 예산 비대칭. 계층별 사후 분석을 하고, 장부에 "TSR−ST는 예산 연장 효과를 포함한다"고 기록한다.
- D3: H 추정기. 답 수준 N표본 대안을 기록한다.
- D4: s1 문자열과 EOS 처리.
- D5: ST 개입.
- D6: top-p와 fp16.
- 장부 정정: 행 4, 6, 8, 11, 12, 18, 19, 22, 24, 29, 30, 37, 44, A-4, A-6, C-8.
- summarize.py 주석 정정.

---

## 부록: 계산 근거

- main_s0(60 pid)
  - ST: step-1 think_end close 30 / cap 29 / eos 1. step-1 길이 분위 [5, 144, 317, 2859, 4096, 4096]. 1024 초과 63%.
  - TSR: k=1이 50, 정지 사유 entropy 55 / budget 5. 이어간 10건은 모두 step 1이 cap.
  - visref_raw: k=1이 51. 이어간 8건은 cap 7, eos 1.
  - 자연 종료 단계의 H: TSR 32/32, VisRef 31/31이 δ 미만.
  - cap 단계의 H<δ: TSR 22/37, VisRef 22/34.
  - 쌍 불일치: TSR vs ST (1,0) 5 / (0,1) 3. VisRef_raw vs ST 3/2. VisRef_raw vs TSR 3/4.
  - \|V\| 평균 1,774.9, 타일 분포 {1:15, 3:4, 4:2, 5:5, 7:13, 9:2, 13:19}.
- DPP 모사: scratchpad `sim/sim.py`, `sim/sim2.py`. Chen과 CEIL 원본 함수를 그대로 썼다.
- Fig. 3(b) 픽셀 판독: 0→x≈113, 1,000토큰당 ≈73.7px(확대 크롭 기준).
