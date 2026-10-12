> 2026-10-13 사용자 지시("현재의 계획과 논리에 오류가 없는지 상위 모델을 통해 독립적으로 검토해")로 작업 과정을 보지 않은 상위 모델(Fable) 하위 에이전트가 쓴 검토 원문이다. 하위 에이전트 보고라 그대로 사실로 쓰지 않는다. Claude가 원자료로 다시 확인한 결과: 중간 1–5와 낮음 1–10, 13은 일치(첫 단계 텍스트 일치 193/839·242/994, 불일치율 16.0%, 검출력 0.72, Table 3·7·8 값, InternVL3.5 보고서 L558-562, 548단계 중 m=998은 544, 582단계 중 38, Spearman 부호 규약, 309문항 밖 불일치 14:13 등). 낮음 11·12는 검토자도 미확인으로 표시. 기록: docs/EXPERIMENT_FLOW.md 2026-10-13 "독립 검토".

# VisRef 재현 프로젝트 독립 검토 (2026-10-12, main 881f1b6 기준)

검토 대상: `REPORT.md`, `README.md`, `README.ko.md`, `docs/EXPERIMENT_FLOW.md`(EF), `docs/SPEC_LEDGER.md`, `docs/AI_USE_REFLECTION.md`, `repro/diag/*.py·json`, `repro/kaggle/runs/{b_main_hs,b_main_fill}`, `repro/llm/{final,fill,hs}`, 논문 텍스트 `VisRef_layout.txt`(LAY)·`VisRef_plain.txt`, 인용 문헌 `InternVL3.5.txt`·`Ghosal_DoesThinkingMoreAlwaysHelp.txt`, 인용 코드 `chen_fast-map-dpp/dpp.py`.
재계산은 저장소를 건드리지 않고 `scratchpad/review/recompute.py`(읽기 전용, `repro/`에서 `python3 -I`로 실행)로 했다. 아래 "VERIFIED"는 원문 파일·줄 또는 원자료 재계산으로 직접 확인한 것, "SUSPECTED"는 근거는 있으나 끝까지 확인하지 못한 것이다.

## 전체 판정

네 개의 판정(본 실험 세 비교, visref_fill, step1_branch, visref_hs)은 각각 결과 전에 적어 둔 규칙대로 계산됐고, 스크립트와 JSON, `results/`, 두 README, REPORT의 숫자는 내가 원자료(`llm/final/merged.jsonl`, `b_main_hs` 출력, `llm/hs/merged.jsonl`)에서 다시 구한 값과 모두 일치했다(정확도, 불일치 쌍, McNemar p, Holm p, 전이 수, k=1 집단 수, 주입/m, n_meaningful 등). 판정을 뒤집어야 할 오류는 찾지 못했다.

문제는 판정이 아니라 판정을 둘러싼 설명에 있다. 가장 큰 것은 후속 실험 두 개(visref_fill, visref_hs)가 첫 생각 단계를 새로 생성하기 때문에 비교 상대인 본 실험 행과 첫 단계 텍스트가 같은 문항이 4분의 1도 안 되는데, 문서는 "Z만 바뀐다", "fp16 배치 잡음 범위에서 같다", "첫 단계 종료 유형 763/839 일치"로 써서 이 차이를 가리고, 그 결과 불일치율이 가정(9%)의 거의 두 배(16%)가 되어 +3.5 pp에 대한 실제 검출력이 약 72%로 떨어졌다는 사실이 어디에도 없다는 점이다. 그 밖에 논문 Table 5를 근거로 "저자들이 답을 여러 개 생성했을 리 없다"고 단정한 것, fill 무효 결과를 논문 Table 7(무작위 선택)로 해석하면서 같은 논문의 diversity-only 결과(Table 3·8)를 빼놓은 것, 공식 59.3의 사고 모드 여부를 "보고서에 없다"고 쓴 것이 중간 수준의 문제다. 나머지는 낡은 문장과 표현 문제다.

심각도별 건수: 높음 0, 중간 5, 낮음 13.

## 발견 사항 (심각도 순)

### 중간 1. visref_hs·visref_fill은 첫 단계가 재생성된 다른 표본이며, 문서가 이를 가린다 — VERIFIED

근거
- 본 실험에서는 TSR·VisRef의 첫 단계가 ST에서 fork되어 1,000/1,000 문항에서 ST 첫 단계 텍스트와 바이트 단위로 같다(재계산). `b_main_hs` 출력의 첫 단계 텍스트가 본 실험 ST 첫 단계와 같은 문항은 839 중 193(23%)뿐이고, 다른 646 문항은 공통 접두 길이 중앙값 약 700자(단계의 약 28%)에서 갈라진다. `b_main_fill`도 994 중 242(24%)만 같다(`recompute.py`).
- 같은 첫 단계를 가진 193 문항에서 hs−TSR 불일치율은 7.8%(6:9), 다른 646 문항에서는 18.4%(62:57). 전체 불일치율 16.0%(134/839).
- 사전등록(EF L1671)은 "본 실험 VisRef−TSR 불일치율 약 9% 기준, 600문항이면 SE 약 1.2 pp, 80% 검출력 최소 효과 약 3.4 pp"로 썼고, 300문항 미만이면 "검출력 부족"으로 표시하기로 했다. 실제 불일치율 16%에서는 n=839라도 SE 1.38 pp, 80% 검출 최소 효과 3.9 pp, +3.5 pp에 대한 검출력 약 72%다(정규 근사, `recompute.py`). 즉 300문항 기준은 9% 가정에서만 뜻이 있고, 839문항의 실제 검출력은 사전등록이 목표로 한 수준에 못 미친다.
- 문서 표현: REPORT 4.5절 "Only Z changes; … seeds and question order are those of the main run", `analyze_hs.py` 머리말 "hs − VisRef (only Z differs)", EF L1664 "같은 문항별 시드(CRN)라 첫 단계는 fp16 배치 잡음 범위에서 같음(fill에서 첫 단계 종료 유형 874/976 일치)", REPORT 4.5절·6절의 "763 of 839", "874 of 976" — 종료 유형 일치율(약 90%)만 적어 텍스트 일치율(약 23%)을 보이지 않는다. 종료 유형 불일치의 방향은 대칭(hs: cap→close 42, close→cap 34; fill: 57 vs 45)이라 편향은 아니고 분산 문제다.

왜 중요한가
- 판정 ③(상한 3.10 < 3.5)은 그대로지만, "hs − TSR +0.24 (−2.38, +3.10)"의 넓은 CI를 본 실험의 "VisRef − TSR −0.1 (−2.0, +1.7)"과 나란히 놓으면 독자는 표본 수 차이만으로 이해한다. 실제로는 재생성 잡음이 불일치를 두 배로 늘렸다.
- "Z만 다르다"는 보조 비교 hs − VisRef(+0.60)의 해석도 과장이다. 같은 이유로 fill − VisRef(+0.1)도 "주입량만 다른 비교"가 아니다.
- 검출력 부족 플래그가 켜지지 않았다는 문장(REPORT 4.5절 "300 or more, so no underpowered flag")은 그 기준의 전제가 깨진 상태에서 안심시키는 문장이 됐다.

수정 제안
- REPORT 4.2·4.5·6절에 첫 단계 텍스트 일치율(193/839, 242/994)과 그 영향(불일치율 16% vs 9%, SE 1.38, 80% 최소 검출 효과 약 3.9 pp, +3.5 검출력 약 72%)을 적고, "Only Z changes"를 "설계상 Z만 바꿨으나 첫 단계는 새 표본"으로 고친다. "검출력 부족 없음"은 삭제하거나 "사전등록 기준(300)은 9% 불일치율 가정"이라고 단서를 단다.
- 보조 분석으로 첫 단계 텍스트가 같은 193문항(hs−TSR −1.55 pp, 6:9)과 다른 646문항(+0.77, 62:57)을 나란히 보고하면 독자가 잡음 기여를 가늠할 수 있다.
- 앞으로의 후속 실험은 본 실험 ST 첫 단계를 fork해서 시작하거나(step1_branch처럼 저장된 z_1을 재토큰화), 비교 상대(TSR)를 같은 세션에서 다시 돌려야 "한 가지만 다른" 비교가 된다. 사전등록의 검출력 계산은 재생성 여부에 따라 불일치율 가정을 달리 해야 한다.

### 중간 2. "저자들이 단계마다 답을 여러 개 생성했을 리 없다"는 Table 5가 뒷받침하지 못하는 단정 — 논문 수치는 VERIFIED, 추론은 내 판단

근거
- REPORT 4.1절 5번: "judging from the paper's latency table (Table 5: ST 7.1 s, TSR 7.7 s, VisRef 8.2 s …) the authors cannot have generated several answers per step". README H 문단(L221-224)과 SPEC_LEDGER X-32도 같은 문장. X-32는 이를 "사용자 판단"이라고 적었지만 REPORT·README에는 사실처럼 쓰였다.
- 논문 수치 확인: LAY L892-895(Table 5), L971-972(8B, H100 1장), L909-913("DPP … adds only 0.5 secs … compared to TSR, and 1.1 secs compared to ST").
- Ghosal [19]는 "we sample multiple answers", "entropy of the distribution over final answers"로 다중 표본을 쓴다(`Ghosal…txt` L400-405). 우리 probe는 `\boxed` 뒤 최대 16토큰짜리 짧은 답이다(`methods.py` 머리말). 8B·H100에서 공유 접두사 위에 짧은 답 N개(예: 8~16개)를 한 배치로 디코딩하면 수백 ms 수준이라 TSR−ST 0.6초 안에 들어간다. 다중 표본이 "불가능"하려면 각 표본이 완결 응답(우리 S10 방식, EF L642: 8B 3문항 4,703.9초)이어야 하는데, Ghosal은 "answers"라고만 했지 완결 응답이라고 하지 않았다. EF L1346이 든 비용 근거(generate 호출마다 전체 재prefill, T4 KV 메모리)는 우리 엔진의 제약이지 저자 쪽 제약이 아니다.

왜 중요한가
- H 추정 방식은 SPEC_LEDGER가 등급 F(운영 선택)로 분류한 핵심 미지정 항목이고, REPORT 5.2·5.3도 저자 문의 항목으로 올렸다. 그런데 본문은 한쪽 답을 이미 확정한 듯 쓰여 있어, 독자가 "단일 표본이 논문에 더 가깝다"고 오해한다. 단일 표본으로 돌아간 결정 자체는 기록된 사실이지만, 그 근거가 논문 수치로 "입증"된 것처럼 읽히면 안 된다.

수정 제안
- "cannot have generated several answers per step"을 "Table 5의 작은 지연 차이는 단계마다 긴 완결 응답을 여러 개 생성하는 방식과는 맞지 않는다고 판단했다(저자 판단, 미확인). 짧은 답 여러 개를 배치로 뽑는 방식은 배제되지 않는다"로 고친다. README H 문단도 같이.

### 중간 3. fill 무효 결과의 해석에서 논문 Table 7(무작위)만 인용하고 Table 3·8(diversity-only)을 빼놓았다 — VERIFIED

근거
- README "Why VisRef showed no extra gain here" 표 1행과 README.ko L72: "the top-up chose tokens without regard to relevance (rank correlation −0.10), and in the paper random selection is also close to ST (Table 7)". SPEC_LEDGER X-36도 이 문장의 근거로 Table 7만 든다.
- 논문: Table 7 Random 67.3 / Relevance-only 75.6 / DPP 79.3, ST 68.1(LAY L977-979, L396). 그러나 같은 논문 Table 3은 relevance ✗ · diversity ✓ 가 77.4(LAY L538), Table 8은 λ=0.0(Div) 71.2(LAY L984). 즉 논문 안에서도 "관련성을 무시한 선택"은 무작위(67.3)와 다르게 ST보다 3~9 pp 높다. SPEC_LEDGER A-9·C-6이 두 값의 불일치를 이미 기록해 두었다.
- fill의 꼬리는 무작위가 아니다. EF L1488과 `dpp.py` `select_tokens_fill`대로 꼬리는 L+εI 위의 같은 greedy이며, 계수 소진 뒤 이득은 "이미 고른 집합이 가장 덜 덮은 방향에서 큰 토큰"을 고른다(EF L1488). 이것은 투영 공간에서의 덮임(다양성) 기준이지 무작위가 아니다. 관련성과의 순위 상관 −0.10은 "관련성 순이 아니다"를 뜻할 뿐 "무작위와 같다"를 뜻하지 않는다.

왜 중요한가
- 현재 문장은 fill의 무효 결과를 "논문도 무작위는 효과 없다고 했으니 예상된 결과"로 약화시킨다. 논문의 diversity-only 행을 함께 보면, 관련성을 무시해도 다양성 기반 선택은 논문에서 큰 이득을 냈으므로, fill의 0.0 pp는 "주입량 부족" 가설에 대해 README가 말하는 것보다 더 불리한 증거다(단, Table 3과 8이 서로 어긋나 논문 쪽 수치의 신뢰도는 낮다).

수정 제안
- 해당 문장을 "채운 부분은 관련성 순이 아니고(순위 상관 −0.10) ridge를 더한 같은 DPP의 연속 선택이다. 논문에서 무작위 선택은 ST 수준(Table 7, 67.3)이지만 관련성 없이 다양성만 쓴 선택은 71.2~77.4로 보고돼 두 표가 어긋난다(Table 3·8, 장부 A-9·C-6). 따라서 이 변형을 '관련 토큰 m개' 시험으로 볼 수는 없지만, 논문의 ablation과 견주면 관련성 무시만으로 효과가 사라진다고 보기도 어렵다"로 바꾼다. X-36에 Table 3·8 줄 번호를 추가한다.

### 중간 4. 공식 InternVL3.5-1B 59.3의 사고 모드 여부: "보고서에 없다"가 아니라 "TTS 없이 얻은 값"이라고 보고서가 말한다 — 원문 VERIFIED, 해석은 논쟁 가능

근거
- REPORT 1절: "the official InternVL3.5-1B MathVista (mini) score of 59.3 (InternVL3.5 report; the report does not say whether that number uses thinking mode)". README "Why our accuracies are 8–12 pp above the paper's"는 우리 ST 57.8이 59.3에 "close"하다는 데서 출발한다.
- `InternVL3.5.txt` L558-559: "unless otherwise specified, the experimental results reported in Section 3 are obtained without applying TTS." L557-558: TTS는 "reasoning depth (i.e., deep thinking) and breadth (i.e., parallel thinking)"를 함께 뜻하고, L562: "Deep Thinking. By activating the Thinking mode…". 59.3은 Section 3의 Table 3(L775)에 있다. 따라서 보고서는 문자 그대로 "59.3은 사고 모드 없이 얻은 값"이라고 말한다. EF L1094와 SPEC_LEDGER X-1은 이 독해를 알고도 "행 이름이 parallel thinking만 언급하므로 불명확"이라고 판단했다.

왜 중요한가
- 우리 ST는 R1 사고 프롬프트를 켠 값이다. 비교 상대가 비사고 모드 점수일 가능성이 높다면 "우리 ST가 공식 값에 가깝다"는 말은 사과와 배의 비교이고, 절대값 차이 원인 절의 출발점이 흔들린다. 적어도 "보고서에 없다"는 표현은 원문보다 약하다.

수정 제안
- REPORT 1절·README를 "InternVL3.5 보고서는 Section 3 수치가 TTS(deep thinking 포함) 없이 얻은 것이라고 밝히므로(L558-562) 59.3은 사고 모드 없는 값일 가능성이 높다. 사고 모드 설정(VLMEvalKit `InternVL3_5-1B-Thinking`, 최대 65,536토큰)으로 59.3이 나왔는지는 미확인"으로 고친다. 저자 문의 항목이 아니라 InternVL 팀 문의 또는 VLMEvalKit 재실행으로 풀 문제임을 적는다.

### 중간 5. visref_hs 추출기 이탈의 시점과 두 팔의 추출기 불일치가 REPORT에 적혀 있지 않다 — VERIFIED

근거
- EF 2026-10-13 항목 순서: "결과 수신" 항목(L1711)에서 규칙만 잠정 결과(hs−TSR +0.36, "같은 규칙을 적용하면 ③")를 본 뒤, 다음 항목(L1716)에서 사용자 지시로 Gemini 대신 Claude Haiku 하위 에이전트로 추출했다. REPORT 4.5절 "Deviation"은 "At the author's request"만 적고 잠정 결과를 본 뒤의 결정이라는 점은 적지 않았다.
- 주 채점에서 hs 행 23건은 Haiku, 비교 상대 TSR·ST·VisRef 행은 Gemini(`llm/final/responses.jsonl` model 필드 56건 전부 gemini-3.5-flash-lite; `llm/hs/responses.jsonl` 43건 전부 claude-haiku-5-5)로 추출됐다. 영향은 작다(hs 21건 사용, 정오 변화 1건: pid 722, 규칙 'yes'→LLM 'no', 응답 끝 "The correct answer should be B) no"이므로 LLM 쪽이 맞다). 규칙만 읽기는 네 방법 모두 같은 채점기라 공정한 비교이고 같은 판정이다.

왜 중요한가
- 판정은 바뀌지 않지만, "결과 전 고정" 원칙(SPEC_LEDGER 원칙 2)을 지켰는지 독자가 확인하려면 시점이 적혀 있어야 한다. 또 두 팔에 다른 추출기를 쓴 비교에서 "규칙만 읽기"가 같은 판정을 준다는 문장은 "그래서 그 읽기가 같은 조건의 비교"라는 설명과 함께 있어야 뜻이 산다.

수정 제안
- 4.5절 Deviation에 "규칙만 잠정 결과(③)를 본 뒤의 변경"임과 "비교 상대 행은 Gemini 추출이므로 규칙만 읽기가 같은 조건의 비교"임을 한 문장씩 추가한다.

### 낮음 1. REPORT 2.4절 "The two follow-ups (Sections 4.2 and 4.3)" — 낡은 문장 — VERIFIED
REPORT L3·요약은 "three follow-up experiments"인데 2.4절 L57은 후속 실험이 둘이라고 쓴다. 4.5절을 포함해 셋으로 고친다.

### 낮음 2. visref_hs 메커니즘 수치 두 곳이 조금 틀리다 — VERIFIED
- REPORT 4.5절 "548 steps of the largest images (m = 998, 3,328 visual tokens)": 재계산하면 548 중 544가 m=998이고 4는 m=844(|V|=2,816)다. EF L1723 "모두 가장 큰 이미지"도 같은 오류.
- "(only 38 of 722 such steps reached m)": m=998인 722단계에는 길이가 m보다 짧은 140단계가 들어 있어 애초에 m에 닿을 수 없다. "길이 ≥ m인 582단계 중 38"이 맞는 분모다.

### 낮음 3. "later picks were slightly anti-related to relevance (−0.171)" — 부호 뜻이 반대로 읽힌다 — VERIFIED
`dpp.py` `pick_stats`(L104-108)는 "negative = more relevant first"로 정의한다. 즉 −0.171은 뒤쪽 선택이 관련성 내림차순에 약하게 따른다는 뜻이고, 관련성에 "반하는" 선택이 아니다. "뒤쪽 선택 순서는 관련성과 약한 음의 순위 상관(−0.171; 관련 큰 것이 조금 먼저)"로 고친다. fill의 −0.10도 같은 규약이다.

### 낮음 4. SPEC_LEDGER 11·12행이 `run.py` 옛 기본값(4096/8192)을 그대로 적고 있다 — VERIFIED
`run.py` L20은 2026-10-13에 1,024/10,240으로 바뀌었고(커밋 881f1b6, J-10 정정) 11행 "기본값은 `run.py:L20` 4096", 12행 "기본 8192 (`run.py:L20`)"는 낡았다. 37행의 코드 위치 "`methods.py:L204-206`, `dpp.py:L51-54`"도 현재 줄(methods.py L192-213, dpp.py L12-43)과 맞지 않는다. AI_USE_REFLECTION "남아 있는 불일치"에는 37·44행만 있다.

### 낮음 5. REPORT 5.1절 "Algorithm 1 fixes … the order z_k → Ṽ_k → H_k → cue" — cue는 Algorithm 1에 없다 — VERIFIED
LAY L303-330의 Algorithm 1에는 이어가기 문구 c가 등장하지 않는다(4행 "Sample z_k ∼ πθ(·|x, τ_1:k−1)"). 문구와 그 위치는 §3(L203-206)과 Fig. 2(L224)에서 온다. 2.1절 표는 "Alg. 1, Fig. 2"로 바르게 적었으니 5.1절도 맞춘다.

### 낮음 6. EF L1570 "우리 구현에서 TSR과 VisRef가 갈리는 곳은 이 309문항뿐이다" — 틀림 — VERIFIED
둘 다 k=1에서 멈춘 691문항에서도 최종 답 정오가 갈린 쌍이 27(VisRef만 14, TSR만 13) 있다(`recompute.py`). k=1에서도 VisRef는 최종 답 직전에 Ṽ_1을 넣으므로 당연하다. REPORT 4.3절 "Scale" 문단은 "If … came from these 309 alone"으로 조건부로 써서 틀리지는 않았지만, 691문항 쪽의 27쌍을 한 줄 적어 주면 독자가 "309문항이 전부"로 읽지 않는다.

### 낮음 7. "no extra gain", "did not bring a gain" — 점추정 표현 — VERIFIED
README L88·L90, README.ko L85, REPORT 요약 "injecting m tokens with last-layer vectors did not bring a gain either", 4.5절 "did not bring VisRef a gain". CI 상한은 fill +2.4, hs +3.1이다. "추가 이득을 검출하지 못했다(CI 상한 +2.4 / +3.1)"로 쓰는 것이 본문의 다른 곳("We did not confirm…")과도 맞다.

### 낮음 8. README "make check passes on 3.10, 3.11, 3.12 and 3.13" — 3.10은 검사 스크립트만 확인됨 — VERIFIED
EF L1611: "`make check`가 Python 3.11·3.12·3.13에서 통과, 장치 Python 3.10에서 검사 스크립트 통과". 3.10에서 `make check` 전체를 돌린 기록은 없다.

### 낮음 9. 날짜 불일치 — VERIFIED
README·README.ko 상태 상자는 2026-10-13, REPORT 5.3절은 "None as of 2026-10-12". 같은 판에서 하루가 어긋난다(시간대 차이로 보이나 통일 필요).

### 낮음 10. README 파일 목록이 `final_judgment.py`를 "pre-registered analyses"로 부른다 — VERIFIED
README L110-111. REPORT 2.4절은 본 실험이 "not a pre-registration in the strict sense"라고 썼다. "decision-rule analyses (main run: rules fixed before the final analysis; follow-ups: before results)" 정도로 맞춘다.

### 낮음 11. "termination rule of the cited Chen code"는 절반만 맞다 — SUSPECTED
REPORT 2.1절은 greedy 정지를 "termination rule of the cited Chen code"라고 쓴다. Chen 코드(`cited_code/chen_fast-map-dpp/dpp.py` L29-30)는 절대값 ε=1e-10으로 멈추고, 우리는 상대값 1e-9·max L_ii다(`dpp.py` L7, L26). 커널 항목이 크면(정규화 없는 임베딩 내적의 제곱합) Chen의 절대 ε은 계수에서 멈추지 않고 m까지 잡음 토큰을 계속 고를 수 있어, "Chen 코드도 거기서 멈춘다"(`dpp.py` L7)는 커널 스케일에 따라 다르다. 논문 자체는 "We terminate after m iterations"(LAY L360-362)이므로 어느 쪽도 논문 절차는 아니다. 2.1절에 "상대 임계값은 우리 선택(SPEC_LEDGER 38행 B-13)"을 덧붙이면 된다. 실제 커널 스케일에서 Chen 절대 ε이 어떻게 동작하는지는 확인하지 않았다.

### 낮음 12. visref_hs에서 입력 공간 V와 마지막 층 Z의 내적이 뜻을 갖는 조건이 논의되지 않았다 — SUSPECTED
`methods.py` L197-200은 V(LLM 입력 임베딩 공간의 시각 토큰)와 Z(최종 norm 뒤 hidden state)로 L=(VZᵀ)(VZᵀ)ᵀ를 만든다. 두 벡터는 다른 공간에 있고, 내적 v·h가 "그 위치의 다음 토큰 로짓" 비슷한 뜻을 가지려면 lm_head가 입력 임베딩과 묶여 있어야 한다(Qwen3-0.6B는 tie_word_embeddings=True로 알고 있으나 로컬 config로 확인하지 못했다). REPORT 4.5절에 이 전제를 한 줄 적고, 저자 문의 5번(Z·V의 층)에 "두 벡터가 같은 공간인지"를 명시하면 좋다.

### 낮음 13. 작은 표현·기록 문제 모음 — VERIFIED
- REPORT 4.2절: "*Strata* (by the first-step end …)"는 사전 지정(EF L1371)인데 바로 뒤 "The small EOS stratum is a post-hoc observation"과 붙어 있어 층별 분석 전체가 사후처럼 읽힌다. "층 구분은 사전 지정, EOS 층의 해석은 사후"로 나눈다.
- REPORT 4.5절 "The code version recorded in every row equals `repro/` at commit ccac0d9"(f1376774850d)는 맞지만 현재 `repro/`의 code_version은 cf46b2e6222e(기본값 변경)라는 점은 EF에만 있다. 2.5절 코드 버전 목록에 한 줄 추가.
- README.ko L3 "(2026-10-12 기준)" 번역 기준일이 본문 상태 상자(10-13)와 어긋난다.

## 계획 검토

현재 계획(EF 끝, REPORT 5.3절, AI_USE_REFLECTION "남아 있는 불일치"): SPEC_LEDGER 37·44행 상태 정정, 저자 문의, 옛 보고서 초안 처리.

빠진 것
1. 위 "중간 1"의 정정이 계획에 없다. 공개 저장소의 REPORT 4.2·4.5·6절, `analyze_hs.py` 머리말, EF L1664의 "fp16 배치 잡음 범위에서 같음"을 고치고, 첫 단계 텍스트 일치율과 실제 검출력을 적어야 한다. 앞으로 후속 실험을 또 한다면 "첫 단계 fork 또는 비교 상대 동시 재실행"을 사전등록 규칙에 넣는다.
2. "중간 2~4"의 문장 정정(Table 5 단정, Table 7 인용, 59.3 사고 모드)도 계획에 없다. 모두 결과 후 표현 변경이므로 EF에 "결과 후 변경"으로 기록하면 된다.
3. SPEC_LEDGER 정정 범위가 좁다. 37·44행 외에 감사 메모 L309가 든 19행("ST 강제 답은 'without additional intervention'과 부분 충돌"), 1-8 상태별 집계(L216-219), 11·12행의 옛 기본값, 37행 코드 줄 번호도 함께 고쳐야 한다.
4. 옛 보고서 초안(Docs 아티팩트, EF L1417)은 "삭제 또는 '초안·폐기' 머리말 추가" 둘 중 하나를 정하면 되는데, 그 전에 초안의 수치가 지금 REPORT와 어디서 다른지(코드 버전 수, k=1 "주입 효과 0" 등 L1423-1424에서 이미 정정된 항목) 목록을 남겨 두면 나중에 혼동이 없다.

저자 문의(REPORT 5.3절) 검토
- 1~10번은 모두 논문에서 실제로 찾을 수 없는 항목이라 근거가 있다(토큰 상한·추출·H 추정·층·m 강제·주입 형식·타일링·k 분포·샘플링은 LAY·plain 전체 grep으로 없음을 확인; 환경은 L790-792에 있으나 정밀도는 없음).
- 추가할 질문:
  - 단계 경계와 문구 삽입 방식: z_k의 끝이 모델이 낸 `</think>`인지, 그 토큰을 억제(Ghosal 방식, `Ghosal…txt` L159-161)하는지 붙여 두고 "Wait, think more"를 그 뒤에 넣는지, 문구가 Ṽ_k 뒤에 오는지. Algorithm 1에는 문구가 없다(위 낮음 5).
  - 정지 시 최종 답 생성 방식: Ṽ_k 바로 뒤에 `</think>`를 강제하고 답하게 하는지, 모델이 스스로 `</think>`를 낼 때까지 자유 생성하는지(EF L1500-1501의 설계 쟁점). 7번의 "answer prompt"보다 구체적으로.
  - 세 방법이 첫 단계를 공유(짝 설계)했는지, 독립 표본인지. 이것이 Table 6 차이의 분산을 결정한다.
  - 문항당 프롬프트: R1 사고 시스템 프롬프트 사용 여부, `\boxed{}` 지시문(VLMEvalKit `USE_COT`) 유무.
  - 논문 ST 46.1이 공식 59.3보다 13 pp 낮은 이유를 저자가 어떻게 보는지(1~3번의 답으로 풀릴 수 있으나 직접 물을 가치가 있다).
  - 문항별 출력(또는 코드) 공유 요청. REPORT 2.3절이 말하는 "두 결과 간 정식 비교 불가"를 푸는 유일한 길이다.
- 문의 메일에는 우리 수치(k=1 정지 71%, 첫 단계 1,024 절단 64%, 주입/m 중앙값 0.199, hs에서 m 도달)를 함께 보내야 저자가 어느 설정이 다른지 짚어 줄 수 있다.

## 확인하지 않은 것

- 본 실험 세션 s1~s9의 원 출력(`results_*.jsonl`)과 `llm/final/merged.jsonl`의 행 단위 동일성, `rescore_v2/v3` 과정. 수치 확인은 merged.jsonl 기준으로만 했다.
- `summarize.paired`·`final_judgment.py`·`analyze_hs.py`의 부트스트랩을 다른 시드나 다른 구현으로 다시 돌리지 않았다(코드를 읽고 McNemar·Holm·CI 색인 규칙만 확인). step1_branch는 JSON과 스크립트의 규칙 일치만 확인하고 `step1_branch_*.jsonl`에서 재계산하지 않았다.
- `diag_st_plain`, `hidden_rank_check`, `analyze_k1_injection`, `st_plain_analysis`, `paper_fingerprints`의 타일링 재계산은 JSON 값과 문서의 일치만 보았다.
- 모델 카드(`internvl_README.md`), `internvl_config.json`, s1·DEER 공식 코드, VLMEvalKit 소스는 공개 저장소 사본에 없거나 열지 않아, 이를 근거로 한 문장(T 0.6 권장, `max_new_tokens=1024` 예제, s1 `\nFinal Answer:`, DEER af, `SPLIT_THINK`)은 확인하지 못했다.
- X-10(CVPR 보충자료에 코드 없음), 8B S10 실행 4,703.9초(EF L642), 코드 버전 간 출력 동일성 810/810(EF L983)은 기록만 보았다.
- `figures/*.svg`의 값과 `results/`의 일치, `make check`·`pytest`는 실행하지 않았다(저장소에 쓰지 않기 위해). `scripts/check_readme_numbers.py`만 읽기 전용으로 돌려 두 README가 통과함을 확인했다.
- Qwen3-0.6B의 tie_word_embeddings와 hidden size(1024)는 기억에 의존했다(위 낮음 12).
- 페이지 PNG는 보지 않았다. 논문 인용은 모두 `VisRef_layout.txt`·`VisRef_plain.txt` 줄 번호 기준이다.
