# VisRef 실험 기록

> 실험을 진행하며 날짜순으로 적은 작업 기록(한국어)입니다. 결론과 판정은 [REPORT.md](../REPORT.md)를 기준으로 합니다. 공개용으로 개인 정보와 작업 환경 메모를 정리했습니다.
> 2026-10-16 정리: 맨 앞에 읽는 순서, 단계별 요약과 핵심, 줄임말 풀이를 넣었습니다. 이전 코드로 실행한 초기 기록(2026-10-05 ~ 10-07 오전)은 이 문서에서 빼고 요약만 남겼습니다. 원문은 [정리 전 판(커밋 4776f6a)의 5–749줄](https://github.com/AhnJG/visref-repro/blob/4776f6ab2607957799bef8db70241cc64313feae/docs/EXPERIMENT_FLOW.md?plain=1#L5-L749)에 그대로 있습니다. 날짜 항목의 본문은 당시 적은 그대로이고, 이 문서를 가리키던 줄 번호("EF L952" 등)만 날짜 항목 이름으로 바꿨습니다(줄 번호가 공개본과 어긋나 있었음).

## 읽는 순서

- 결론만 보려면 [REPORT.md](../REPORT.md)를 읽습니다.
- 무엇을 언제 했는지는 아래 "단계별 요약" 표를, 단계마다 무엇을 정하고 왜 다음으로 넘어갔는지는 "단계별 핵심"을 봅니다. 더 자세한 내용은 각 단계 끝 "근거 항목"의 날짜 항목으로 갑니다.
- 날짜 항목은 오래된 것이 위에 있습니다. 뒤 항목이 앞 항목을 고치는 일이 있으므로, 같은 주제는 가장 뒤 항목을 기준으로 읽습니다.
- 제목에 "결과 전에 고정", "실행 전 계획"이 붙은 항목은 결과를 보기 전에 판정 규칙을 적어 둔 것입니다.
- 줄임말과 기호는 아래 "줄임말과 기호" 표에 풀어 두었습니다.

## 단계별 요약

| 날짜 | 단계 | 한 일 | 결과 (자세한 숫자는 REPORT) |
|---|---|---|---|
| 10-05 ~ 10-07 오전 | 초기 시도 (이전 코드) | 이미지 토큰 압축 연구로 시작했다가, 기준이 되는 논문 결과를 먼저 재현하기로 바꿈 | 설정 누락과 실행 실패가 많아 결과를 근거로 쓰지 않음 |
| 10-07 | 새 코드와 설정 시험 | 새 재현 코드 `repro/` 작성. 작은 시험 실행(smoke1–3, main_s0, full_s1, b_test)으로 논문에 적히지 않은 설정을 정함: 생각을 멈추는 기준, 이미지를 다시 넣는 형식, 생각 길이 상한, 답 처리 | REPORT 2.6절 B1–B5 |
| 10-07 ~ 10-11 | 본 실험 `b_main` | 1,000문항 × ST·TSR·VisRef, Kaggle 세션 9번 | 정확도 ST 57.8 / TSR 60.1 / VisRef 60.0 % (논문 46.1 / 48.5 / 52.0) |
| 10-11 | 후속 1 `visref_fill` | 다시 넣는 이미지 토큰을 m개까지 채움 (976문항) | fill − TSR 0.0 pp |
| 10-11 | 후속 2 `step1_branch` | 첫 생각 단계 뒤 이미지를 다시 넣는 것만의 효과 (309문항, 조건 A–D) | 바로 답할 때 +1.40 pp, 더 생각한 뒤 답할 때 +0.97 pp. 둘 다 차이를 확인하지 못함 |
| 10-12 ~ 10-13 | 공개 정리, 후속 3 `visref_hs` | 저장소 공개 준비. 모델 마지막 층 벡터로 이미지 토큰을 고름 (839문항) | hs − TSR +0.24 pp |
| 10-13 | 생각 이어가기 실험 계획 | `tsr_think`·`visref_think`·`visref_hs_think`의 실행 방법과 판정 규칙을 정함 | 아직 실행하지 않음 |
| 10-15 | 이미지 태그 주입 시험 | `visref_imgtag`(처음 이름 `visref_bundle`)를 10문항씩 두 번 시험, 1,000문항 비교 계획 | 첫 시험에서 출력 형식이 깨짐 → 답 문구를 고친 뒤 해결. REPORT B7 |
| 10-15 ~ 10-16 | 이미지 태그 주입 1,000문항 비교 `imgtag_main` | RunPod RTX A5000 한 장에서 `visref_imgtag`·`tsr_answer_prompt`·`tsr` 실행(확인 실행 6문항 통과 뒤, 생성 22.6시간) | imgtag − 같은 답 문구 TSR −1.0 pp (95% CI −3.1 ~ +1.0) → 논문 크기 이득 재현 안 됨(③) |

## 단계별 핵심

단계마다 무엇을 정했고, 무엇이 나왔고, 그래서 다음에 무엇을 했는지입니다. 숫자의 근거는 각 단계 끝 "근거 항목"의 날짜 항목과 REPORT에 있습니다.

### 초기 시도 (10-05 ~ 10-07 오전, 이전 코드)
- **정한 것**
  - 처음 목표는 재현이 아니라 후속 연구: 이미지 토큰을 더 적게 쓰고도 정확도를 지키는 압축 방법
  - 이전 코드(`src/visref`, 이 저장소에 없음), InternVL3.5-1B, MathVista testmini를 개발 100 / 선택 200 / 잠금 700으로 나눔
- **결과**: 개발 100문항 첫 비교(P01 v2)에서 ST 40% / TSR 40% / DPP30 30%
- **알게 된 것**
  - 사고 모드용 시스템 프롬프트가 빠져 있었고, 생각 단계·채점·엔트로피·재주입 조건도 확인되지 않은 상태였음 → 개선 실험을 멈추고 논문 재현부터 하기로 함
  - 번호를 붙인 Kaggle 실행(r2–r31, 이어서 R1–R31)으로 멈춤 기준·생각 길이·답 추출·DPP 선택을 시험했지만 시간 초과·실행 실패·검증 기준 미달이 많았음. R29까지 쓴 GPU 시간 50,981.329초
  - 이 구간의 결과는 이후 판정의 근거로 쓰지 않음
- **다음**: 이전 코드를 이어가지 않고 새 재현 코드 `repro/`를 씀
- 원문: [정리 전 판(커밋 4776f6a)의 5–749줄](https://github.com/AhnJG/visref-repro/blob/4776f6ab2607957799bef8db70241cc64313feae/docs/EXPERIMENT_FLOW.md?plain=1#L5-L749). 당시 글은 영어와 줄임말이 많음. 이전 코드의 마지막 실행 R29·R31은 아래 날짜순 기록에 있음

### 새 코드와 설정 시험 (10-07)
- **정한 것** (작은 시험 실행으로 하나씩)
  - 새 코드 `repro/`를 처음부터 씀. 목표는 InternVL3.5-1B × MathVista testmini 1,000문항 (논문 46.1/48.5/52.0)
  - 생각을 멈추는 기준: `\boxed{…}` 안의 짧은 답(최대 16토큰) 하나의 엔트로피. 앞 방식에서는 TSR이 한 번도 멈추지 못했음(0/18)
  - 이미지를 다시 넣는 형식: `<img>` 태그 없이 `</think>` 바로 앞. 태그를 쓰면 둘째 `</think>`가 31/63단계, 안 쓰면 10/65단계에 나옴
  - 생각 길이 상한: 단계당 1,024, 합계 10,240토큰. 앞서 쓴 단계당 4,096 상한에서는 많아야 2단계라 최대 10단계와 논문의 생각 분량에 닿을 수 없었음
  - 생각이 잘린 뒤에는 s1 방식(`Final Answer:`)으로 답을 받음. 그래도 이 답 26개 중 12개가 답 상한에 또 닿음 → 본 실험에서 마무리 방법을 더함
  - 넣는 이미지 토큰 수: 고를 수 있는 만큼만(`visref_raw_rank`). 논문은 m개, Chen 코드는 더 고를 수 없으면 멈춤 → 코드를 따름
  - 채점: 규칙으로 먼저 답을 뽑고, 못 뽑은 행만 LLM으로 뽑음(공식 MathVista 방식)
- **결과**: 설정을 고르는 시험이라 정확도 판정은 없음(정확도 차이는 어떤 결정에도 쓰지 않기로 함)
- **알게 된 것**
  - 입력 임베딩으로 고르면 고를 수 있는 수가 서로 다른 토큰 수에 막힘: 67단계 중 50단계가 m보다 적게 고름(중앙값 98, m 537)
  - 정지 기준은 작동함: 첫 단계에서 엔트로피로 멈춘 비율 TSR 50/59, VisRef 51/59
  - 시험에 쓴 문항도 평가 1,000문항에 들어 있어, 설정이 그 문항에 맞춰졌을 수 있음
- **다음**: 파일럿 없이 본 실험 `b_main` 시작
- 근거 항목: 10-07 "새 lean 재현 경로…", "smoke2 결과와 본 실험 결정…", "smoke3 결과 → probe를 `\boxed` 형식으로 교체…", "main_s0 결과와 판정…", "DPP 구현 대조…", "b_test 재빌드: VisRef = rank만큼만…", "채점: 규칙 → 실패 시 LLM 추출…"; REPORT 2.6절 (B1~B5 항목)

### 본 실험 `b_main` (10-07 ~ 10-11)
- **정한 것**
  - 생각 길이 상한: 단계당 1,024, 합계 10,240토큰
  - 생각이 잘리면 s1 방식으로 답을 받음
  - 답이 상한에 닿으면 DEER 방식으로 최대 100토큰 더 받아 마무리
  - VisRef는 고를 수 있는 만큼만 이미지 토큰을 넣음. m개까지 채우는 변형은 논문·인용 코드에 없어서 첫 세션 뒤 뺌
- **결과**: ST 57.8 / TSR 60.1 / VisRef 60.0
  - TSR − ST +2.3 pp: 논문(+2.4)과 어긋나지 않음
  - VisRef − TSR −0.1 pp: 논문(+3.5)이 재현되지 않음
- **눈에 띈 점** (759문항 시점)
  - 넣은 이미지 토큰이 m의 약 1/5뿐(중앙값 0.21)
  - VisRef는 답을 쓰다 생각을 다시 여는 경우가 많음(47%)
- **다음**: "넣은 수가 부족해서"를 의심 → 후속 1에서 m개까지 채워 봄
- 근거 항목: 10-07 "결정(작성자): s1 + DEER af 폴백…", 10-10 "관찰: VisRef 효과가 TSR을 넘지 못함", 10-11 "b_main 최종 판정"

### 후속 1 `visref_fill` (10-11)
- **정한 것** (결과 전)
  - 이미지 토큰을 m개까지 채움
  - 판정은 fill − TSR 짝 차이
  - 멈춤 판단은 이미지가 있는 문맥과 없는 문맥 둘 다 기록함. 판정에는 있는 쪽만 씀
- **결과**: 976문항에서 fill − TSR 0.0 pp (95% CI −2.15 ~ +2.36)
- **알게 된 것**: 채워 넣은 토큰이 생각과의 관련성 순서가 아니었음(순위 상관 −0.10)
  - 그래서 "넣은 수가 부족해서"라는 설명은 지지되지 않음
  - 다만 "관련 있는 토큰 m개"는 아직 시험하지 못함
- **다음**: 첫 단계 멈춤 판단용 답을 결과를 본 뒤 보니, VisRef가 바로 답할 때 309문항에서 +5.5 pp 높았음 → 후속 2(`step1_branch`)에서 결과 전에 규칙을 정하고 다시 잼
- 근거 항목: 10-10 "원인 분리 실험 실행 전 계획 수정", 10-11 "visref_fill 최종", 10-11 "계획 독립 검토와 1단계 probe 사후 탐색"

### 후속 2 `step1_branch` (10-11)
- **정한 것** (결과 전)
  - 대상: 본 실험에서 TSR이나 VisRef가 첫 단계 뒤 생각을 이어간 309문항 (첫 단계 종료: 상한 절단 297 / EOS 7 / 스스로 닫음 5)
  - 조건 A·B는 바로 답함(B만 이미지를 다시 넣음), C·D는 최대 512토큰 더 생각한 뒤 답함(D만 이미지를 다시 넣음)
  - 네 조건 모두 같은 첫 생각 글에서 출발하고, 답은 문항마다 표본 3개의 평균으로 셈
  - 주 비교는 B − A와 D − C: 95% CI 하한이 양수여야 이득으로 인정
  - 이 규모로 알아볼 수 있는 효과는 약 5–6 pp 이상(80% 검출력 기준)
- **결과**: 309문항에서 B − A +1.40 pp (95% CI −1.19 ~ +3.99), D − C +0.97 pp (95% CI −2.27 ~ +4.21)
  - 정답률(표본 3개 평균): A 33.98 / B 35.38 / C 35.28 / D 36.25 %
  - 두 비교 모두 CI가 음수에서 양수까지 걸쳐 이득을 확인하지 못함
- **알게 된 것**
  - 앞서 결과를 본 뒤에 본 +5.5 pp는 잡음이었을 가능성이 있음. 이번 CI 상한은 약 +4 pp라, 5–6 pp 이상의 효과는 검출되지 않았다고만 말할 수 있음
  - 이미지를 넣으면 더 생각한 글에서 이미지 언급이 늘었지만(22.7% → 31.4%) 정답률 차이로는 이어지지 않음
  - 309문항에서만 효과가 난다고 보고 1,000문항으로 환산한 CI 상한은 약 +1.3 pp라, 논문의 VisRef − TSR +3.5 pp에 못 미침
- **다음**: 첫 단계에서 이미지를 다시 넣는 것으로는 논문 크기의 차이를 설명하지 못함 → 이미지 토큰을 고르는 벡터 Z를 마지막 층으로 바꿔 보는 후속 3(`visref_hs`)
- 근거 항목: 10-11 "실행 전 계획: step1_branch…", "step1_branch 본 실행 결과…"; REPORT 4.3절

### 공개 정리, 후속 3 `visref_hs` (10-12 ~ 10-13)
- **정한 것**
  - 공개 정리: 새 GPU 실험 없이 영문 `REPORT.md`와 그림 2개를 갖춰 저장소를 공개함
  - 공개 후 외부 검토의 지적 5개를 원자료로 다시 확인해 모두 맞았고, 그중 1–4를 반영함(결과·판정은 그대로)
  - `visref_hs`(결과 전): 바꾸는 것은 Z 하나. 입력 임베딩 대신 마지막 층 은닉 상태로 이미지 토큰을 고름. 이유: 입력 임베딩으로는 m의 약 1/5만 들어갔음
  - 판정은 hs − TSR 짝 차이의 95% CI로: 상한이 +3.5 pp 미만이면 "논문 크기 이득 재현 안 됨"
- **결과**: 839문항에서 hs − TSR +0.24 pp (95% CI −2.38 ~ +3.10) → 논문 크기 이득 재현 안 됨
  - 정답률: ST 57.57 / TSR 59.71 / VisRef 59.36 / VisRef-hs 59.95 %
- **알게 된 것**
  - 이미지 토큰을 m까지 넣었는데도(주입 수/m 중앙값 1.0, 본 실험 VisRef는 0.205) TSR과 비교한 이득이 검출되지 않음
  - 다만 다양성을 뚜렷이 더한 선택은 중앙값 70개(m의 13.6%)뿐이었음
  - 첫 생각 단계를 새로 만들어 본 실험과 글이 같은 문항은 839문항 중 193뿐. 짝이 어긋난 문항이 134/839(16.0%)라 +3.5 pp를 알아볼 검출력은 약 72%
  - 규칙 채점만 보면 hs − TSR +0.36 pp로 판정은 같음. 추출기는 규칙만 결과를 본 뒤 Claude Haiku로 바꿈(계획과 다름)
- **다음**: 세션이 닿지 못한 161문항 마무리와 생각 이어가기 실험 세 개를 계획함
- 근거 항목: 10-12 "실험 종료 + 공개 폴더 완성…", "공개 폴더에 그림 2개 추가…", "공개 저장소 외부 검토 반영…", "후속 실험 3 `visref_hs` 실행 전 계획…"; 10-13 "`visref_hs` 결과 수신…", "`visref_hs` 최종: LLM 보조 추출…", "독립 검토와 확인"; REPORT 4.5절, 5.4절

### 생각 이어가기 실험 계획 (10-13)
- **정한 것** (결과 전)
  - 질문: 멈추기로 한 뒤 생각을 더 이어가게 하면, VisRef가 TSR보다 논문 크기(+3.5 pp)의 이득을 내는가
  - 방법 셋: `tsr_think`(이미지 없이), `visref_think`(다시 넣은 이미지 뒤에서), `visref_hs_think`(마지막 층 Z로 고른 이미지 뒤에서)
  - 멈추기로 하면 "Wait, think more"를 붙여 최대 512토큰 더 생각한 뒤 답함. 정지 사유와 상관없이 한 번만 적용
  - 추가 생각 512토큰은 합계 상한 10,240토큰과 별도로 셈
  - `tsr_think`는 대조군으로 더 생각한 효과와 이미지 효과를 나눔. 이미 TSR − ST +2.3 pp, 앞선 시험의 C − A +1.29 pp처럼 더 생각만 해도 오르는 경향이 있었음
  - 세 방법 모두 본 실험에 저장된 첫 생각 단계에서 출발. `visref_hs`는 첫 단계를 새로 만들어 839문항 중 193만 같았기 때문
  - 주 비교는 `visref_think` − `tsr_think`: 95% CI 하한이 양수면 이득, 상한이 +3.5 pp 미만이면 "논문 크기 이득 재현 안 됨"
  - 1,000문항이면 +3.5 pp를 알아볼 확률은 0.96–0.79(불일치율 8.9%~16.0% 가정). 시간은 방법당 9시간 이상으로 추정(실측 아님)
- **결과**: 아직 실행하지 않음
- **알게 된 것**
  - 이 실험은 논문의 멈춤 방식을 바꾸므로, 이득이 나와도 "이미지 뒤에 생각을 더하는 변형"의 근거이지 VisRef 재현은 아님
  - 지금은 VisRef가 첫 단계에서 멈춘 문항이 71.3%라 대부분 문항에서 다시 넣은 이미지가 답 바로 앞에 옴
- **다음**: 세션을 나눠 돌리고 끝난 앞부분으로 한 번만 판정함(세션 수는 결과가 아니라 예산으로 정함)
- 근거 항목: 10-13 "검토 반영 1차(Table 5 문장), 다음 실험 세 개 계획 공개…", "다음 실험 설계 결정: `tsr_think` 대조군 포함…", "`visref_think`·`tsr_think` 설계 결정 일부…", "실행 전 계획: 생각 이어가기 실험…"; REPORT 5.4절

### 이미지 태그 주입 시험 (10-15)
- **정한 것** (결과 전)
  - 배경: 본 실험에서 VisRef가 첫 단계에서 멈춘 문항 중 스스로 닫은 329문항은 예측이 거의 그대로(328/329)인데, 답 안 `</think>` 재출력이 ST 20 → VisRef 156으로 늘어 답의 형식만 깨짐
  - 한꺼번에 바꾸는 것(`visref_imgtag`, 처음 이름 `visref_bundle`): 공식 `<img>` 태그로 감싸기, 답할 때는 `</think>` 뒤·이어갈 때는 생각 안에 넣기, 고르는 벡터를 텍스트·이미지 모두 마지막 층으로, 앞뒤 줄바꿈
  - 시험 가동은 10문항(pid 순으로 앞 5개씩: 스스로 닫은 문항, 잘린 문항): 정답은 보지 않고 실행 문제만 봄
  - 이어서 1,000문항 비교를 계획(`imgtag_main`): 주 비교는 같은 답 문구를 쓰는 TSR(`tsr_answer_prompt`)과의 짝 차이, 상한이 +3.5 pp 미만이면 "논문 크기 이득 재현 안 됨"
- **결과**: 10문항 시험 두 번뿐(정답은 보지 않음). 1,000문항 비교 결과는 다음 단계 `imgtag_main`
  - 첫 시험: 스스로 닫은 문항 모두(5/5)가 질문을 다시 적었고 1개는 채점할 답이 빔 → 다음 단계로 가지 않고 수정
  - 모든 최종 답을 `\nFinal Answer:` 뒤에서 받게 고친 뒤 두 번째 시험: 질문 재작성·빈 답·답 안 `</think>` 재출력 0/5, 오류 없음
- **알게 된 것**
  - 원인 추정: 공식 프롬프트에서 `</img>` 줄 바로 뒤에는 질문이 오므로, 모델이 새 질문이 시작된 것으로 읽는 듯함(추정)
  - 이 변형은 m에 가깝게 넣음: 주입 수/m 중앙값 0.97 (본 실험 VisRef는 0.19)
  - 정확도는 계산하지 않음. 이득이 보여도 재현이 아니라 "구현 자유도가 결과를 좌우한다"는 근거로 보고함
- **다음**: RunPod 한 GPU에서 확인 실행(6문항)을 먼저 하고, 오류가 없으면 1,000문항 본 실행
- 근거 항목: 10-15 "실행 전 계획: 이미지 태그 주입 변형 `visref_bundle` 시험 가동…", "이미지 태그 주입 시험 가동 결과…", "이미지 태그 주입 수정: 모든 최종 답을…", "이미지 태그 주입 다시 시험 가동 결과…", "이름 바꿈: `visref_bundle` → `visref_imgtag`…", "실행 전 계획: 이미지 태그 주입 1,000문항 비교…", 10-16 "확인 실행 결과와 본 실행 시작…"; REPORT B7

### 이미지 태그 주입 1,000문항 비교 `imgtag_main` (10-15 ~ 10-16)
- **정한 것** (결과 전)
  - 세 조건을 한 실행에서: `visref_imgtag`, 같은 답 문구의 TSR(`tsr_answer_prompt`), 보통 TSR(`tsr`). 모두 본 실험에 저장된 첫 단계에서 출발하고 그 뒤 난수 상태를 함께 씀
  - 주 비교는 `visref_imgtag` − `tsr_answer_prompt`. 95% CI 상한이 +3.5 pp 미만이면 ③ "논문 크기 이득 재현 안 됨"
  - `\nFinal Answer:` 뒤 답이 비면 오답
  - 실행 중에는 정답률을 보지 않고 구조만 점검(238문항 시점). 분석 기술 항목 셋(재출력과 답 상한, 비용과 멈춤 엔트로피, 원본 단위 부트스트랩)은 결과 전에 추가
- **결과**: 1,000문항 × 3조건 3,000행, 오류 0, RunPod RTX A5000 한 장에서 생성 22.6시간
  - 정답률: imgtag 60.1 / 같은 답 문구 TSR 61.1 / TSR 60.4 %
  - 주 비교 −1.0 pp (95% CI −3.1 ~ +1.0, 한쪽만 맞힘 49 대 59, McNemar p = 0.39) → ③ 논문 크기 이득 재현 안 됨. 짝이 어긋난 문항 10.8%, +3.5 pp 검출력 0.92
  - 보조: imgtag − TSR −0.3 pp (−2.4 ~ +1.8), 같은 답 문구 TSR − TSR +0.7 pp (0.0 ~ +1.5)
  - 원본 31개 단위로 다시 뽑아도 CI −3.0 ~ +1.1. 규칙 채점만 보아도 같은 판정
- **알게 된 것** (결과를 본 뒤 나눠 본 것)
  - 이미지 토큰은 m까지 들어갔지만(주입 수/m 중앙값 1.0), 최종 답 안 `</think>` 재출력은 249건으로 남음(같은 답 문구 TSR 107건). 10문항 시험의 0/5는 전체로 이어지지 않음
  - 첫 단계를 모델이 스스로 닫은 334문항에서 imgtag가 9문항을 잃고 하나도 얻지 못함(0 대 9). 빈 답 16건 중 15건이 이 묶음에 있고, 잃은 9문항 중 5문항이 빈 답. 2026-10-17 확인: 16건 모두 실제로는 빈 답이 아니라, 답을 쓴 뒤 `</think>`를 다시 내 "마지막 `</think>` 뒤"가 빈 경우였다(재출력과 채점 규칙이 겹친 것)
  - 생성 토큰은 같은 답 문구 TSR의 1.13배. 논문 Table 5의 VisRef/TSR 시간 비 약 1.06보다 큼
- **다음**: 후속 A2–A6은 모두 이득을 보이지 못했고 서로 간 보정이 없는 탐색임. 남은 계획은 확인 실험 하나(멈춤 엔트로피 사전 점검 뒤 설계 확정)와 A5
- 근거 항목: 10-15 "실행 전 계획: 이미지 태그 주입 1,000문항 비교…", 10-16 "확인 실행 결과와 본 실행 시작…", "`imgtag_main` 실행 중 구조 점검과 분석 기술 항목 추가…", "`imgtag_main` 결과…"; REPORT 4.6절

## 줄임말과 기호

| 줄임말·기호 | 뜻 |
|---|---|
| ST / TSR / VisRef | 보통 생각 / 글로만 다시 생각하게 함("Wait, think more") / 다시 생각하기 전에 이미지 일부를 다시 넣음(논문의 방법) |
| smoke, smoke1–3 | 본 실험 전에 수십 문항으로 돌린 작은 시험 실행 |
| main_s0, full_s1, b_test | 설정을 정하려고 돌린 시험 실행의 이름 (REPORT 2.6절) |
| b_main, b_main_s1–s9 | 본 실험과 그 Kaggle 세션 1–9 |
| k, k=1 | 생각 단계 번호. "k=1"은 첫 단계에서 멈춘 경우 |
| z_k, Ṽ_k (V_k) | k번째 생각 단계의 글 / 그 단계 뒤에 다시 넣은 이미지 토큰 |
| H, δ | 지금 답하면 답이 얼마나 불확실한지(엔트로피). H가 δ = 0.25보다 작으면 생각을 멈춤 |
| probe | H를 재려고 생각 중간에 짧게 받아 보는 답 (`\boxed{…}` 안, 최대 16토큰) |
| m, K_max (kmax) | 단계마다 다시 넣을 이미지 토큰 수(질문 이미지 토큰의 30%) / 최대 단계 수 10. "kmax 정지"는 10단계에 닿아 멈춘 경우 |
| DPP, rank | 생각과 관련 있고 서로 겹치지 않는 이미지 토큰을 고르는 계산 / 그 계산으로 고를 수 있는 최대 개수. rank가 m보다 작으면 m개보다 적게 고름 |
| ridge, εI, fill | 행렬에 작은 값을 더해 m개까지 채워 고르는 변형 (후속 1) |
| Z | 이미지 토큰을 고를 때 쓰는 생각 글의 벡터. 본 실험은 입력 임베딩, 후속 3은 마지막 층 |
| V | 고를 후보인 이미지 토큰의 벡터. 다시 넣는 값은 언어 모델에 들어가는 이미지 값(비전 인코더 → 연결층)이고, A6는 고를 때만 언어 모델 마지막 층 값을 썼다 |
| M_k, 정규화 잔차 | M_k: 이번 단계 생각 벡터로 만든 행렬(Σ zzᵀ), 이미지 토큰의 관련성과 다양성을 재는 기준. 정규화 잔차: 이미 고른 토큰으로 설명되지 않고 남은 정보를 관련성으로 나눈 값(0에 가까우면 새 정보가 거의 없음) |
| 상한 절단 / 자연 종료 / EOS | 생각이 1,024토큰 상한에서 잘림 / 모델이 스스로 `</think>`로 생각을 닫음 / 출력 자체가 끝남 |
| 강제 답 (s1) | 생각이 잘린 뒤 `</think>` + `\nFinal Answer:`를 붙여 답을 받는 방식 (s1 논문의 방식) |
| af (DEER af) | 답이 1,024토큰 상한에 닿으면 `\n**Final Answer**\n\boxed`를 붙여 최대 100토큰 더 받는 마무리 (DEER 코드의 옵션 이름) |
| CRN | 비교하는 방법들이 같은 문항에서 같은 난수를 쓰게 하는 설계 |
| fork | 세 방법이 첫 생각 단계를 함께 쓰고 그 뒤에 갈라지는 방식 |
| complete prefix | 섞은 문항 순서에서 세 방법이 모두 끝낸 앞부분 (중간 집계 단위) |
| paired, 짝 비교 | 같은 문항끼리 두 방법을 비교 |
| CI | 95% 신뢰구간. 0을 포함하면 차이가 있다고 말할 수 없음 |
| McNemar, Holm | 짝 비교의 정확 검정 / 비교가 여럿일 때 p값을 보정하는 방법 |
| pp (%p) | 정확도 차이의 단위 (퍼센트포인트) |
| v3, v4 | 채점 규칙 버전. v4는 객관식에서 답을 못 뽑으면 오답 처리 (본 실험 판정은 v3, 둘이 다른 행 0) |
| 읽기 (a) / (b) / (c) | 답 처리에 따른 세 가지 채점. (a) 그대로 채점(판정용) / (b) af 문구 앞까지만 남기고 다시 채점 / (c) af로 마무리한 답을 모두 오답 처리(하한) |
| code_version | 결과를 정하는 코드 파일의 해시 앞 12자리. 결과 행마다 기록 |
| pid | MathVista 문항 번호 |
| T4, fp16, OOM | Kaggle GPU 종류 / 16비트 계산 / GPU 메모리 부족 |
| LAY, REF/, SRC/ | 논문 텍스트 파일(`VisRef_layout.txt`) / 다른 논문 텍스트 폴더 / 외부 코드 폴더. 공개 출처로 바꾸는 법은 [SOURCES.md](SOURCES.md) |
| SPEC_LEDGER, "답 상한 처리(J-12)" | [IMPLEMENTATION_CHOICES.md](IMPLEMENTATION_CHOICES.md)의 옛 이름 / 그 문서의 행 번호(J·A·B·C·X)와 이름. 번호 체계는 그 문서 맨 앞에 풀어 둠 |
| EF | 이 문서(EXPERIMENT_FLOW.md)를 줄여 부르는 이름 |
| § | 논문이나 문서의 절 |

## 날짜순 기록 (2026-10-07 새 코드부터)

## 2026-10-07 새 lean 재현 경로 (`repro/`)

작성자 결정: (1) 기존 `src/visref`를 이어가지 않고 새 간결한 코드 작성 (기존 기록 보존), (2) 주 목표는 Appendix Table 6의 InternVL3.5-1B × MathVista testmini 1,000문항 행 (46.1/48.5/52.0), (3) H는 기존 결정대로 단일 응답의 답 토큰 entropy, (4) GPU는 development 10문항 smoke test까지만 실행 후 보고. 구현 내용과 논문 값 vs 구현상 설정 표는 repro/README.md에 있다. CPU 테스트 21 passed (DPP vs brute-force greedy, batched/chunked = 단일 row, entropy = full forward, 상태 전이, run 재개). 실제 GPU 실행은 아직 없음. STOP_REQUESTED.json은 기존 pipeline용 marker로 그대로 둔다. 주의: 1,000문항 전체 실행에는 기존 따로 떼어둔 문항(700) 전부가 포함된다.

## 2026-10-07 smoke 전 독립 검토 반영
독립 검토에서 논문(Figure 포함)과 `repro/`를 대조했다. 명백한 버그는 없었다. 이어가기 문구를 `Wait, think more`로 바꾸고, DPP rank 부족 대응 변형 `visref_reg`(L+εI)를 추가했다. smoke는 st/tsr/visref/visref_reg 4개 method × development 10문항으로 다시 빌드했다. CPU 테스트 23 passed. 검토 기록.


## 2026-10-07 R29 실제 검증 회수 및 판정

R29 비공개v1 제출·회수를 완료했다. 위 미제출 상태는 당시 기록이며 현재 상태를 대체하지 않는다. 실제419 TSR의 sequence NLL3.477/6.584/2.183과 Wait3회 검산 PASS;4번째 reasoning이 thinking8192에 도달해 별도final0/1. R30 제출조건을 완화하지 않았으며 신규GPU 제출 없음. 미완성 reasoning을 final로 표시하던 작업 source 기록을 수정하고 관련29검사 PASS. 누적50981.329s, STOP·selection/locked 보존. 결과와 증거.

## 2026-10-07 smoke1 결과와 수정 (`repro/kaggle/runs/smoke`)
InternVL3.5-1B, development 10문항 × st/tsr/visref/visref_reg, T4×2. 정확도는 ST 5/10, TSR 5/10, VisRef 4/9, VisRef_reg 4/9이고 paired 차이는 0이다(표본이 작아 성능 판단은 불가). GPU0은 912번 VisRef prefill에서 OOM(15GiB 할당)이 나 2행이 누락됐다. 평균 시각 토큰은 1792개로 논문 ~1772와 일치한다. 추출 규칙은 boxed/answer phrase가 대부분이다. visref의 DPP 선택 비율은 평균 0.47, visref_reg는 1.0이다.
핵심 문제: 답 span 평균 entropy가 모든 probe에서 ≈0(최대 0.003)이라 전 문항이 k=1에서 멈췄다. 그래서 TSR≡ST가 되고 VisRef는 사고가 끝난 뒤 한 번만 주입되는 구조가 됐다. 작성자 결정으로 H를 응답 전체 토큰(EOS 제외)의 평균 entropy로 바꿨다(span 평균은 `H_span`으로 기록). 함께 수정한 것: chunked prefill(OOM 방지), 1단계 thinking을 공유해 method별로 갈라짐(같은 context·seed라 동일, 연산 약 1/4), batch 16, chunk 1024. CPU 테스트 25 passed. 다음은 smoke2(development 30문항, 4 method, 3h 상한)로 k 분포·속도·예산을 확인한다(실행 승인 대기).

## 2026-10-07 smoke2 전 독립 검토 반영
독립 검토 결과: 첫 단계 공유 후 갈라지는 방식(5 pid × 4 method, k≤4, 단독 실행과 이산 출력 동일)와 chunked prefill(left padding 포함, 단독 forward와 일치)은 정상이다. 다음을 수정했다.
(1) 투입 조건이 첫 단계를 돌리는 job(leader)만 세어 갈라진 뒤 동시 job이 최대 60개까지 늘던 문제: 갈라진 뒤 따라가는 job(follower)을 포함해 동시 job ≤ batch로 제한.
(2) deadline 이후 진행 중 job을 `.partial`로 남기고 종료(타임아웃 시 손실 방지).
(3) OOM이 나면 그룹을 반으로 나눠 재시도, expandable_segments, batch_tokens 48000.
(4) probe 토큰별 entropy, 생성 토큰 수, round별 메모리 peak 기록.
(5) smoke1 원자료에서 VisRef probe의 7~8/9가 "초안 + 두 번째 `</think>` + 답" 형태임을 확인(TSR 1/10). 이전 작성자 규칙대로 y를 응답 안의 마지막 `</think>` 뒤로 정해 H와 채점에 적용.
CPU 테스트 27 passed. smoke2는 다시 빌드했고 push 대기 중이다.

## 2026-10-07 독립 논문 충실도 검토 반영
판정: 단계 정의(자연 `</think>`), 재주입 위치, probe 폐기, Wait 이어가기, DPP greedy는 논문과 맞는다. 다만 `<img>…</img></think>` 형태의 재주입이 1B 모델에서 두 번째 `</think>`와 질문 환각을 일으켜, 포맷이 오염된 VisRef를 측정할 위험이 있다.
반영(작성자 결정): 태그 없이 주입하는 `visref_raw`를 추가하고, smoke2를 st/tsr/visref/visref_reg/visref_raw 5개 method(batch 20)로 다시 빌드했다.
반영하지 않음: probe 답을 최종 답으로 재사용하자는 제안은, 정지 결정에 쓰인 표본이라 선택 편향이 생기므로 작성자 결정으로 Algorithm 1의 별도 재표본을 유지했다. hidden-state Z(`visref_hs`)는 smoke2 결과를 본 뒤 다음 smoke에서 검토한다.
smoke2 판단 기준: 1단계 H<0.25 비율이 10~90% 범위인지, k 분포, 두 번째 `</think>` 비율, 질문 환각 수. 검토에서는 논문 Fig.3에서 VisRef의 thinking 토큰이 ST의 수 배라고 해석했으나, Table 5 지연(+1.1s)과의 차이(원인 미확인)로 기록한다. CPU 테스트 28 passed.


## 2026-10-07 마지막 단일 GPU 업로드 R31 고정

작성자가 GPU 업로드 한 번을 명시 승인했다. R31은 fixed S1 sequence entropy/.25/K10을 유지하며 실제sampled closing을 response conditioning에 commit하고, 예산 소진 시 미완성 후속step을 분리해 마지막 완성trajectory에서 별도final을 생성한다. 이 운영 budget rollback은 H/Kmax 정상종료와 분리한다. 공개419/479 TSR→계산·별도final gate→DPP 한 notebook,2T4/8B/FP16/eager, cap14400s/예상1800–14280s. 로컬및frozen bundle418PASS, 미완성probe나 runtime오류는 DPP를 막는다. 최초패키징6실패는 누락된 기존검사fixture2개를 포함해 해결했으며 실패본도 보존했다. 단일upload journal로 불확실응답 재제출도 금지한다. 명세와 사전판정.


## 2026-10-07 R31 마지막 업로드 완료

R31 arki13/visref-paper-r31-final-20261007 v1/kernel137371435를 단1회 제출했다. Source3570d85e6bebd4b89b4987a09a1142a6852765c59fa5ef864abeba6daae863ef/notebook3dc9e47f61f2d886db484a50cd63050c2c7753e169af81333a3ea6c5bf74533d,418 frozenPASS,180파일1712202bytes/credential·raw·selection·holdout 없음. cap14400s/기존누적50981.329s 보존. 작성자허용 업로드1회 소진, 자동재제출/재실행 금지. 실패·예산복원·paper정상종료를분리해 결과회수·비교자료를작성한다. launch.


## R31 마지막 실행 회수·검산·로컬 채점 정정 완료

{"runner_status": "INCOMPLETE", "methods": {"tsr_natural_sequence": {"planned": 2, "emitted": 2, "complete_finals": 2, "normal_h_or_kmax": 1, "budget_rollback_finals": 1}, "dpp30_natural_sequence": {"planned": 2, "emitted": 2, "complete_finals": 1, "normal_h_or_kmax": 1, "budget_rollback_finals": 0}}, "scores": {"tsr_natural_sequence": {"denominator": 2, "correct": 0, "extraction_failures": 0}, "dpp30_natural_sequence": {"denominator": 2, "correct": 0, "extraction_failures": 1}}, "charged_s": 3833.7392390000023, "cumulative_charged_s": 54815.068239, "gpu_uploads": 1, "new_dispatch": false, "paper_reproduction_verified": false}
R31은 부분 완료이며 Baseline 검증 성공이 아니다. TSR419 예산복원, DPP479 reasoning EOS/별도final 부재, 479 DPP fixed-m determinant0 동률을 분리 기록했다. 첫 채점 connector TypeError/외부호출0 기록은 보존하고 공식 scorer를 바꾸지 않은 동일final 재채점으로 정정했다. 추가 GPU 제출0. 저자 H estimator·Z층·재주입 semantics 및 예산복원의 저자 동일성은 미확인이고 정상 완료·성능 재현 조건은 해소되지 않았다.
원본과 비교 제한.

## 2026-10-07 smoke2 결과와 본 실험 결정 (`repro/kaggle/runs/smoke2`)
설정: development 30문항 × st/tsr/visref/visref_reg/visref_raw, T4×2, H는 응답 y(마지막 `</think>` 뒤)의 토큰 평균 entropy. GPU0은 deadline에서 `.partial`을 남기고 정상 종료했다. GPU1은 한 round가 약 670초라 Kaggle 3h 상한에 강제 종료됐다(완료 행은 보존). 완료 행: ST 24, 나머지 각 21. 공통 19문항 기준.
- 정지 규칙이 작동함: 1단계 H<0.25 비율 tsr 9/21, visref 11/21, raw 10/21. k 분포는 1~10에 퍼짐(평균 3.0~3.3). stop: entropy 대부분, kmax 2~3, budget 1~3.
- 정확도(소표본, CI 모두 0 포함): ST 62.5%(24), TSR 76.2, visref 71.4, visref_reg 66.7, visref_raw 76.2. paired: TSR−ST +10.5pp, raw−ST +10.5, visref−ST +5.3, reg−ST 0.
- 형식 오염(두 번째 `</think>`): visref 31/63 단계, reg 25/67, raw 10/65, tsr 4/70. 질문 환각 0. DPP 비율: visref 0.42, raw 0.34, reg 1.0.
- 비용: unique 생성 토큰 약 7.2k/문항(st+tsr+visref). ST thinking 중앙값 889, 4096 초과 7/24(그중 정답 2).
- 속도: GPU당 14~16 tok/s. 원인은 긴 context로 KV 예산(48000)이 그룹을 3~4개로 쪼개 순차 decode한 것, round 동기화, 전체 어휘 정렬 sampler.
조치: sampler top-k 우선(50개만 정렬), round 중에도 deadline 반영(stop_at), batch_tokens 72000, 세션 이어 달리기(`--skip-from`), Kaggle slug 하이픈화. CPU 테스트 30 passed.
작성자 결정: 본 실험은 st/tsr/visref_raw, 토큰 상한 단계 4096·합계 8192, 1,000문항 전체를 약 9h 세션으로 나눠 실행(기존 따로 떼어둔 700문항 잠금 해제, 첫 세션이 속도 점검 겸함). `full_s1` 빌드 완료(`arki13/visref-repro-full-s1`).

## 2026-10-07 본 실험 세션 단축
세션을 2h로 줄였다(투입은 1.5h까지, deadline 1h50m). 문항 순서를 seed 0으로 섞어, 어느 세션에서 멈춰도 완료된 문항이 1,000문항의 무작위 표본이 되게 했다. 세션마다 `--skip-from`으로 이어 붙인다. `full_s1` 재빌드: 1,000문항 순서 섞음, st/tsr/visref_raw.

## 2026-10-07 실행 전 독립 검토 반영
판정: "작은 수정 후 실행". 결과를 조용히 왜곡하는 버그는 없다. 반영한 것:
(1) decode 가속: 미리 할당한 KV 버퍼 + GQA decode attention. transformers 4.55의 매 스텝 cat과 repeat_kv를 없애 2~3배를 기대한다. 시작 시 GPU에서 기존 경로와 greedy 토큰을 비교해 다르면 자동으로 끈다.
(2) 투입 마감을 timeout−2700s로 당겼다(마무리 시간 확보).
(3) 셔플 seed 기본값을 0으로.
(4) summarize: 섞인 순서의 '완료된 앞부분'만 무작위 부분집합으로 집계하고, 전체 행 정확도는 참고로만 남긴다. McNemar 불일치 수, error 수도 집계한다.
(5) 그룹별 시간 로그, 갈라짐 여부, non-finite logits가 난 job만 error로 격리.
CPU 테스트 32 passed, 비정상 logits 격리 별도 확인. full_s1 재빌드(2h, 1,000문항 셔플, st/tsr/visref_raw).

## 2026-10-07 정지 통계 H를 논문 정의 쪽으로 변경 (작성자 결정)
배경: smoke2 probe 240개를 보면, 자연스러운 긴 응답에 논문식 합(sum)을 쓸 때 5~79 nats로 δ=0.25 아래가 0.8%에 불과하다. 지금 방식인 평균은 논문의 양과 단위가 다르다. 논문 Fig.2는 "Answer:" 상자와 답 분포 히스토그램을 그리지만, 프롬프트 문자열·길이·표본 수는 명시하지 않는다.
결정: probe를 `</think>` + `\n\nAnswer:` 뒤 짧은 답 1개(T=1, top-k/p 없음, 첫 줄바꿈/EOS 종료, 최대 32토큰)로 하고, H는 토큰 엔트로피의 합(종료 토큰 포함)으로 한다. 이것은 답 분포 엔트로피 H(Y)의 단일표본 추정이다. 32토큰 안에 끝나지 않으면 H=∞로 계속 진행한다. 최종 답은 자연스러운 응답(T=0.6)으로 따로 생성한다(ST와 같은 형식). 이전의 "answer prompt 금지" 결정은 작성자가 해제했다.
같이 반영: final_finish/probe_finish, code_version, extract_failed, pytest.ini, 잘린 probe 처리. CPU 테스트 35 passed.
영향: full_s1(구 H)의 TSR/VisRef 결과는 새 정의와 섞지 않는다. ST는 H와 무관하다. 다음은 smoke3(development 20문항 × st/tsr/visref_raw, 1h)로 새 H의 정지 비율과 probe 답 형식을 확인한다.

## 2026-10-07 smoke3 결과 → probe를 `\boxed` 형식으로 교체 (smoke4)
- smoke3(dev 20, `\n\nAnswer:` probe): TSR 엔트로피 정지 0/18, visref_raw 4/18. TSR probe의 유한 H 107개 중 δ 미만 0개(중앙값 1.42). 토큰 분해 시 내용 토큰 ≈0.01 nat, 앞 공백 ≈0.47, 줄 끝 ≈0.84 → 형식 엔트로피가 정지를 막음. 정확도 ST 72.2 / TSR 61.1 / visref_raw 72.2(n=18).
- full_s1(무작위 58문항, 구 H): ST 60.3%(95% CI 47.5–71.9). ST의 57%가 4096 상한(반복 루프), dev는 2/20 → dev가 더 쉬움. 2h에 58문항×3 method → 1,000문항은 이 속도로는 너무 오래 걸림.
- 결정(작성자 동의): probe = `</think>` + `\n\n\boxed`, 모델이 `{`를 직접 생성, 상자 닫힘(중괄호 깊이 0)에서 종료, 최대 16토큰. 상자 깊이 1에서는 `}`로 시작하는 토큰들을 한 결과로 합쳐 엔트로피 계산. 무효(빈 상자/EOS/상한) → H=∞. 이전 형식은 `--probe-stop newline`으로 유지.
- 진단 추가: 단계별 rank_V, n_unique_z, n_dpp; summarize에 전 단계 H 분위수, δ 미만 수, k=1 엔트로피 정지 수, probe 상자 내용.
- smoke4: smoke3와 같은 dev 20문항·순서, st/tsr/visref_raw, 3600s. 바뀐 요인은 probe 하나(진단 필드는 결과에 영향 없음, 독립 검토에서 newline 모드·ST 행 불변 확인).
- 본 실험: 무작위 셔플 앞부분 일부(200~300 예상), 크기는 smoke4 처리량을 보고 결정.

## 2026-10-07 main_s0 판정 규칙 (결과를 보기 전에 고정)
main_s0 = 1,000문항 seed-0 셔플(full_s1과 같은 순서) 앞부분, st/tsr/visref_raw, 1h, smoke4와 같은 코드.
원칙: 계속/수정 판단은 실행 상태와 probe 작동 방식만으로 한다. 방법 간 정확도 차이는 판단에 쓰지 않는다(결과를 보고 설정을 바꾸지 않기 위해). δ=0.25는 논문 값으로 고정하고 정지율에 맞춰 조정하지 않는다.

1. 실행 상태: error 행 0, fast_kv_check ok, 셔플 순서 앞부분에서 세 방법이 모두 끝난 문항 ≥12.
   실패하면 로그로 원인을 찾아 고친다. 코드가 바뀌므로 main_s0은 진단 자료로만 쓰고, 본 실험은 새 시리즈로 다시 시작한다.
2. probe 작동: probe_finish 'end' ≥85%, 첫 토큰 '{' 엔트로피(probe_ent_full[0]) 중앙값 <0.05, 상자 내용이 대부분 글자/숫자.
   실패하면 원인별로 고친다. 상자 내용이 길어 상한에 걸리면 probe_cap을 올리고, 첫 토큰이 '{'가 아니면 프롬프트를 '\boxed{' 변형으로 바꾼다. 어느 쪽이든 새 시리즈다.
3. 정지율(TSR k=1 엔트로피 정지 비율, 예상 25~60%): 범위 밖이어도 그것만으로 설정을 바꾸지 않는다.
   <15%이면 위치별 엔트로피를 확인한다. 엔트로피가 '{' 위치나 닫는 위치(형식)에 몰려 있으면 결함으로 보고 고친다(새 시리즈). 내용 토큰에 있으면(모델이 실제로 불확실, B 대 No 같은 동의 답 포함) 논문 정의대로 받아들이고 진행한다.
   >90%이면 무효 처리나 빈 상자 같은 인위적 원인이 없는지 확인하고, 없으면 진행한다(Table 5와 일치하는 방향).
4. 처리량 → 표본 크기: 시간당 완료 문항 수로 예상 n을 계산한다. 3개 방법으로 n<150이면 main_s1을 빌드하기 전에 TSR 제외 여부를 작성자에게 묻는다.
5. 진행: main_s1부터는 2h, `--skip-from main_s0/.../results_*.jsonl`(.partial 제외), 같은 run_args, run.py·visref/*.py를 바꾸지 않는다(code_version 일치). CLI 인자 변경도 새 시리즈로 본다. 세션마다 pull 후 1~2항만 빠르게 확인한다.
6. 최종: 모든 세션을 `summarize.py … --order main_s0 ids.json`으로 집계해 정확도, paired 차이와 95% CI, 불일치 쌍 수를 보고한다. 불일치 쌍은 직접 채점을 다시 확인한다. 상한·정지율·강제로 닫은 경우를 함께 보고한다.
보조 확인: main_s0 ST를 full_s1 ST와 같은 pid끼리 비교한다(정확도·상한 도달률이 크게 다르면 원인을 조사하되, 계속 여부 판단에는 쓰지 않는다).

## 2026-10-07 main_s0 결과와 판정 (code_version 8aaa689c94e9)
- 상태: error 0, fast_kv ok, 셔플 앞부분에서 세 방법이 모두 끝난 문항 59(179행/1h). → 통과
- probe: 정상 종료 TSR 68/68, VisRef 67/67. 첫 토큰 '{' 69/70, 엔트로피 중앙값 0.0. 상자 내용은 전부 글자 또는 숫자. → 통과
- 정지: k=1 엔트로피 정지 TSR 50/59(85%), VisRef 51/59(86%). 15~90% 범위라 추가 조치 없음. 병합을 빼면 0.25 미만이 TSR 41/70에 그쳐(병합 시 55/70) 닫는 토큰 병합이 효과 있음.
  이어간 문항(TSR 10, VisRef 8)은 1개를 빼고 모두 1단계가 4096 상한(반복 루프)에서 강제 종료된 경우. 2단계도 대부분 4096 상한 → 합계 예산에서 정지.
- 처리량: 1h에 59문항 → 2h 세션 약 120문항. 남은 7h로 약 450~480문항 예상 → TSR 제외 질문 불필요.
- 정확도(참고용, 판단에 쓰지 않음, n=59): ST 55.9 / TSR 59.3 / VisRef 57.6. ST는 full_s1 같은 pid 52개에서 29 vs 30(상한 도달 25 vs 28)으로 일관됨.
- 판정: 규칙상 통과 → main_s1(2h, --skip-from main_s0) 빌드 완료.
- 관찰: TSR−ST 불일치 8쌍 중 5쌍, VisRef−ST 5쌍 중 4쌍이 k=1 문항. 같은 문맥에서 최종 답만 다시 뽑아 생긴 표본 잡음이다.
  → 개선 후보(작성자 결정 대기): 공통 난수(CRN). probe는 별도 난수열을 쓰고, k=1에서 정지하면 최종 답을 1단계 직후 난수 상태로 뽑아 TSR이 ST와 같은 답을 내게 한다. 각 방법의 분포는 바뀌지 않고 paired 차이의 분산만 줄어든다.

## 2026-10-07 b_test: 단계 1024 상한 변형 + 강제 답변·CRN (판정 규칙은 결과 전에 고정)
배경
- main_s0(A: 단계 4096 / 합계 8192)에서 강제로 `</think>`를 넣으면 모델이 답 대신 추론을 이어 갔다. ST 강제 종료 30문항은 정답 11, 답변 중앙값 635토큰(14개 답변 상한)이고, 자연 종료 30문항은 정답 22, 68토큰이다.
- A는 최대 2단계라 K_max=10과 논문의 생각 토큰 비율(VisRef/ST 4.2~7.3배, Fig. 3)에 구조적으로 닿지 않는다.
- 단계 상한 600은 Fig. 3 ST 별표 최대 978과 맞지 않아 쓰지 않는다. 1024는 Fig. 3과 어긋나지 않는 가장 작은 흔한 값이고 InternVL3.5 모델 카드 예시 `max_new_tokens=1024`와 같다. 논문에 없는 구현상 가정이다.
변경 (새 시리즈, main_s0와 합치지 않음)
1. 강제 답변: 생각이 상한/EOS로 끊기면 `</think>` + `\n\n**Final Answer**\n\boxed` 뒤 상자만 생성한다(상자 닫힘에서 종료, 최대 64토큰). s1 budget forcing·DEER probe와 같은 형식이고, 모든 방법·모든 정지 사유에 똑같이 적용한다.
2. 잘린 단계 뒤 이어가기 문구 앞에 `\n`을 넣는다(자연 종료 단계는 '.\n'으로 끝남).
3. CRN: probe가 전용 난수열을 쓴다. TSR이 k=1에서 멈추면 최종 답이 ST와 같다(각 방법의 분포는 불변).
4. 상한: 단계 1024, 합계 10,240, K_max 10. 나머지(δ=0.25, `\boxed` probe, visref_raw, T=0.6)는 main_s0과 같다.
실행: seed-0 셔플 순서를 처음부터(main_s0과 같은 순서), st/tsr/visref_raw, timeout 4200s(투입 2800s, deadline 3600s).
판정
- 통과 조건(건전성, T4): error 0, fast_kv ok, 세 방법이 모두 끝난 문항 ≥45, 강제 답변 무효(빈 상자·상자 미종료) ≤10%.
- 처리량: 시간당 완료 문항으로 본 실험 n을 추정한다. 3방법 n<150이면 TSR 제외 여부를 작성자에게 묻는다.
- 보고만 하고 판단에 쓰지 않음: TSR/VisRef 1단계 정지 비율과 평균 k(T2), 생각 토큰 비율 vs ST(T3), ST(1024) vs main_s0 ST(4096) 같은 pid 정확도(T1), 방법 간 정확도. T1은 짧은 상한에서 정의상 생기는 결과라 판정에서 뺐다.
- 이상 신호: 1단계가 1024에서 잘렸는데 TSR이 한 번도 이어가지 않으면 구현 오류로 보고 조사한다.
- 통과 시 본 실험: `b_main_s1`부터 `--skip-from b_test/.../results_*.jsonl`, 코드·인자 불변. 실패 시 원인을 고쳐 새 시리즈로 한다. B 자체를 접을지는 작성자가 정한다.
- b_test 빌드 완료: `arki13/visref-repro-b-test`, code_version 02d67d405ad3, 4200s, seed-0 순서(main_s0과 동일). CPU 테스트 59 passed(강제 답변 유효/무효·공백, 줄바꿈, CRN: 작은 LM에서 TSR k=1 최종 답 = ST, CRN 없으면 다름). 독립 검토: 확정 버그 없음. 지적 반영: 상자 앞 공백 제거. 참고: GPU에서는 batch 구성에 따른 fp16 오차로 TSR k=1 = ST가 비트 단위로 보장되지 않는다(분산 감소 효과는 유지).

## 2026-10-07 DPP 구현 대조 (Chen 2018 참조 코드, CEIL)
- greedy 갱신·선택: Chen Alg. 1 / 공식 코드(fast-map-dpp `dpp()`)와 같다. 테스트에서 naive log-det greedy와 일치.
- 정지: 공식 코드는 `d_j² < 1e-10`(절대값)에서 멈추고 m개보다 적게 반환한다(채우지 않음). 재현은 상대 기준 `1e-9·max d²`(커널 크기에 무관) 후, 남은 자리를 relevance r²=L_ii 순으로 채운다(구현상 선택, 논문·Chen·CEIL 어디에도 없음).
- 커널: VisRef L = V M_k Vᵀ에는 별도 relevance 항이 없지만 L_ij = q_i q_j cos(a_i, a_j), q_i = ‖Z v_i‖로 품질×다양성 분해와 같은 꼴이다. CEIL은 L̃ = Diag(r) L Diag(r)에 λ 가중(exp(r/2λ))과 KNN 사전 축소를 두고 고정 K개를 뽑는다. VisRef에는 λ도 사전 축소도 없다. relevance 채움은 이 품질 항 방향과 일관된다.
- rank: Z가 input embedding이라 rank(L) ≤ z_k 고유 토큰 수. main_s0 visref_raw 67단계에서 n_dpp = n_unique_z가 50단계, n_dpp < m이 50단계(75%), 중앙값 n_dpp 98 / m 537. Chen 공식 코드대로라면 주입 토큰이 m의 약 1/5로 줄어든다. b_test(단계 1024)에서는 고유 토큰이 줄어 채움 비율이 더 커질 수 있다.
- 논문 쪽 단서: Fig. 4(b)에서 m 20%→30%로 76.1→79.2가 오르는데, input embedding Z + 채움 없음이면 rank를 넘는 m 증가는 선택을 바꾸지 못한다. 논문의 rank 부족 시 처리 방식은 논문에 명시되지 않음(저자 구현은 미확인). hidden-state Z 비교는 본 실험 뒤 별도 시리즈로 한다(b_test는 바꾸지 않음).
- 추가 확인(2026-10-07, PDF 원문): §4.1.1 Eq. 11 뒤 "enforcing |V_k| = m … We terminate after m iterations" → Chen 공식 코드식 조기 종료(m보다 적게 반환)는 논문과 다르다. Fig. 4(b) 원문: m 20%→30%에서 76.1→79.2, 40%는 추가 이득 없음. Table 3(8B): relevance만 MathVista 75.6, diversity만 77.4, 둘 다 79.3.
- 합성 실험(N=1790, d=1024, 단계 1024토큰 중 고유 178, m=537): Chen 공식 코드는 178~192개에서 멈춤(절대 1e-10 기준, 커널 크기에 따라 잡음 픽 0~14개). 논문 문구대로 m번 끝까지 돌리면 rank 이후 359개 선택은 반올림 잡음이 결정(relevance 채움과 겹침 float64 8개, float32 75개, 무작위 기대 약 80개) → 이 구간은 논문 수식이 정의되지 않음.
- 뜻: input embedding Z에서는 주입 토큰의 다수가 relevance 채움이라, 우리 VisRef가 Table 3의 'relevance만' 변형에 가까울 수 있다. hidden-state Z(rank ≤ min(단계 길이, d)) 비교의 우선순위를 올린다.

## 2026-10-07 b_test 재빌드: s1 강제 답변 + ridge DPP (작성자 결정, push 전)
- DEER 공식 코드 원문 확인: 주 구현(vllm-deer.py) answer_prompt_str = `\n**Final Answer**\n\boxed`, 간이 버전은 `…The final answer is \boxed`. 둘 다 신뢰도 probe용이고, 상한 도달 시 최종 답은 `\n</think>\n\n` 뒤 자유 생성(답이 안 끝날 때만 옵션 af=1로 위 문구 추가, 기본 꺼짐). → 직전 b_test의 '상자만' 방식은 어느 참고 문헌과도 그대로 같지 않았다.
- s1 README 원문: 최종 답 앞에 `Final Answer:`를 붙일 수 있고 'to prevent the model from just continuing to reason in its answer when early exiting'. VisRef가 TSR 근거로 인용([34]).
- 결정 1(강제 답변): s1 방식. 잘린 생각 + `</think>` + `\n\nFinal Answer:` 뒤 자유 답변(답 상한 1024, EOS 종료). 채점 텍스트 = `Final Answer:` + 답(답 안에서 `</think>`를 다시 내면 그 뒤 텍스트).
- 결정 2(VisRef 채움): rank 이후 relevance 채움 대신 L+εI ridge로 m개 모두 log-det greedy(`visref_raw_reg`, 태그 없는 주입). 근거: 논문 '|V_k| = m … m iterations', rank 이후 원식은 정의되지 않고 실제로는 반올림 잡음이 선택(합성 실험), relevance 채움은 Table 3 'relevance만'(75.6 < 79.3)에 가까움.
- b_test 판정 규칙 중 '강제 답변 무효'의 정의를 바꾼다: 강제 답변이 EOS 없이 답 상한(1024)에 도달한 경우. 기준(≤10%)과 나머지 규칙은 그대로다. 추출 실패(extract_failed)는 함께 보고한다.
- CPU 테스트 66 passed(s1 강제 답변 유효/상한/뒤따른 `</think>` 처리, 객관식 'Final Answer: B' 채점, visref_raw_reg가 rank 2 < m 6에서도 m개를 DPP로 선택).
- 추가 확인(PDF 부록): App. J Table 7(8B): Random 67.3 / Relevance-only 75.6 / DPP 79.3(MathVista). App. K Table 8: λ·relevance + (1−λ)·diversity, λ=0.5가 최고(=가중치 없는 Eq. 10, 우리 구현과 같음). Random ≈ ST(68.1), DPP 79.3. rank 부족 시 처리 방식은 논문에 명시되지 않음(미확인).
- main_s0 visref_raw 67단계 재집계: 단계 길이 T_k < m은 18/67뿐(중앙값 4096, 1B 반복 루프), 고유 토큰 수 < m은 50/67(중앙값 178 vs m 537). 자연 종료 400~900토큰 단계 5개의 고유 토큰 124~200. → rank를 정하는 것은 단계 길이가 아니라 고유 토큰 수(input embedding Z).
- CEIL 공식 코드(HKUNLP/icl-ceil `src/utils/dpp_map.py` `fast_map_dpp`, VisRef 인용 [55]) 확인: Chen 코드에서 ε 정지와 선택 항목 제외(-inf)를 뺀 형태로, 항상 max_length번 반복한다(VisRef 본문 'terminate after m iterations'와 일치). 합성 kernel(rank 178, m 537)에 원문 그대로 돌리면 537개를 반환하지만 서로 다른 토큰은 187~190개뿐이고 나머지는 중복. Chen 공식 코드는 178~192개에서 정지. → 인용한 두 구현 모두 input-embedding Z에서 사실상 'rank만큼만 서로 다른 토큰'을 고른다. 논문에 이 문제는 명시되지 않음(미확인). hidden-state Z 가설은 근거가 약해짐. b_test VisRef 변형(현재 ridge)을 rank-only로 바꿀지 작성자 결정 대기.

## 2026-10-07 b_test 재빌드: VisRef = rank만큼만 (`visref_raw_rank`, 작성자 결정, push 전)
- 근거: 인용한 Chen·CEIL 구현이 모두 input-embedding Z에서 사실상 rank만큼만 서로 다른 토큰을 고른다(위 CEIL 원문 확인). 논문에 이는 명시되지 않음(미확인). 직전 결정(ridge, m개 모두)은 논문 문구만 근거로 한 것이라 교체한다.
- 구현: greedy가 고른 토큰만 중복 없이 원래 순서로 주입, 채움 없음. 단계 기록에 `n_inject` 추가. 합성 kernel(rank 178, m 537)에서 178개 주입(커널 크기 무관).
- 영향: 다시 넣는 시각 토큰이 m(중앙값 537)보다 크게 적다(main_s0 기준 중앙값 98). 보고서에 명시한다. relevance 채움·ridge는 App. J Table 7 확장 실험에서 비교한다.
- 판정 규칙은 그대로(강제 답변 무효 = EOS 없이 답 상한 도달). CPU 테스트 68 passed.

## 2026-10-07 원문 명세표·독립 감사 → b_test를 파일럿으로 재정의 (결과 전에 고정)
- 명세표: `repro/SPEC_LEDGER.md`(51행: 명시 19 / 그림·표에서 유도 3 / 인용 구현에서 확인 10 / 가정 19). 원문: `references/papers/VisRef_fulltext/`(전체 텍스트, 쪽별 텍스트·이미지, 그림 쪽 고해상도), 인용 문헌 원문 `references/papers/ref_fulltext/`, 공식 코드 `references/sources/cited_code/`.
- 독립 감사: `repro/AUDIT_REPORT_20261007.md`. 판정 '작은 수정 후 실행', 크래시·결과 오염 버그 없음.
  - B1(차단): VisRef가 rank 한정 한 가지뿐이면 해석 불가. 논문은 |V_k|=m 명시(L341, L360-362), [28]은 '|Y| ≤ rank(L)'(Kulesza&Taskar L851-852)라 논문 서술은 rank ≥ m을 전제, Fig. 4(b) m 민감도. → `visref_raw`(m까지 relevance 채움)를 같은 실행에 추가.
  - B2(차단): b_test는 파일럿. n이 수십~100이면 쌍 비교 SE 4~5pt라 목표 델타(+2.4/+3.5/+5.9)를 판정할 수 없다.
  - D4: s1 문자열·조건을 공식 코드와 같게 맞춤(아래). D2(잘린 뒤 이어가기 = 예산 연장 효과), D3(H 추정기), D5(ST 개입), D6(top-p·fp16)은 문서화. 대장 정정 목록은 감사 보고서 §2, §6.
  - 4a: Table 6 'InternVL-1B'는 InternVL3.5-1B(신뢰도 약 85%). 4c: 자연 종료 단계 probe는 main_s0에서 63/63 H<δ.
- 코드 변경(code_version 0f8e003fbf90, CPU 테스트 70 passed): 잘린 단계는 텍스트가 `\n`으로 끝나지 않을 때만 바로 뒤에 `\n`(V_k·`</think>`·Wait 모두 그 뒤), 강제 답은 상한 절단에만 `</think>` + `\nFinal Answer:`(EOS 종료는 자연 답변). summarize: step-1 종료 유형별 정확도·평균 k, 자연 종료 probe의 H≥δ 수, 루프 단계 수(고유/길이 < 0.1), n_inject/m, CRN 점검(TSR k=1 최종 답 = ST, main_s0 사전-CRN 기준 1/50).
- b_test 재빌드: st/tsr/visref_raw_rank/visref_raw, 단계 1024 / 합계 10,240, 4200s, seed-0 순서 처음부터.
- 파일럿 판정(정확도 차이는 어떤 결정에도 쓰지 않음):
  1. 건전성: error 0, fast_kv ok, 네 방법 모두 끝난 문항 ≥40, 강제 답변 무효(EOS 없이 답 상한) ≤10%.
  2. CRN: TSR k=1 문항 중 최종 답이 ST와 같은 비율을 보고(기대 대부분 같음; 50% 미만이면 GPU 비결정성·구현 조사).
  3. 보고 항목: step-1 종료 유형 분포와 유형별 정확도, 자연 종료 probe의 H≥δ 비율, k 분포·정지 사유, n_inject/m(두 VisRef 변형), 루프 단계 수, 생각 토큰 비율 vs ST, 방법별 처리량.
  4. 본 실험 설계(VisRef 변형 선택, 방법 수, 세션 수)는 파일럿의 처리량·주입량 진단을 보고 작성자가 정한다.
- 작성자 검토 후 정리(2026-10-07): 다음은 '설계상 당연하거나 영향이 작아' 문서화만 하고 더 이상 수정 사유로 삼지 않는다.
  1. ST/TSR·VisRef 예산 비대칭(감사 D2): 논문 설계 그대로다. Fig. 3 캡션 'the baseline with no additional test-time compute (standard thinking) … increasing test-time token budget'(plain L714). 방법별 생각 토큰만 보고한다.
  2. ST의 상한 강제 답(D5): 유한한 상한에서 생각을 끝내는 처리일 뿐, 논문이 말하는 'intervention'(Wait 연장)과 다르다.
  3. s1 문자열의 줄바꿈 개수·EOS 처리(D4): 반영 완료, 추가 수정 없음.
  4. 체크포인트: InternVL3.5-1B로 확정(작성자). 논문 L735 'three state-of-the-art MLRMs in their reasoning (thinking) mode: InternVL3.5-8B [45] …'.
  5. ST 절대 수준(46.1 vs 공식 59.3), Fig. 3 비율 읽은 값 차이, Fig. 4(b) 본문·막대 차이: 구현과 무관, 델타와 CI로만 비교한다.
  6. top-p 0.95, fp16(T4는 bf16 미지원), GPU에서 CRN 비트 단위 불일치: 문서화만.
- 계속 중요하게 보는 것: VisRef 주입 수(rank vs m, 파일럿에서 두 변형 비교), 자연 종료 뒤 H≈0이라 이어가기가 거의 상한 절단 뒤에만 생기는 점(다단계가 실제로 일어나는지), 토큰 상한(구현상 가정), 채점 추출(마지막 줄 대체 약 8%).

## 2026-10-07 채점: 규칙 → 실패 시 LLM 추출 (MathVista 공식 방식, 작성자 결정: Claude Haiku)
- 공식: `mathvista_official_r4/evaluation/extract_answer.py` L29-80(간단한 형식만 규칙, 나머지는 GPT + `prompts/ext_ans.py` 예시 프롬프트), VLMEvalKit은 규칙 후 GPT-4o-mini. VisRef는 추출기 미명시(App. D exact match).
- 구현: `repro/llm_extract.py` — prepare(규칙이 last_line/empty이거나 추출 실패한 행만, 공식 create_test_prompt 그대로, 정답은 넣지 않음, query는 공식 hint 템플릿으로 재구성), merge(LLM 추출값을 같은 normalize로 재채점, 규칙만/규칙+LLM 정확도 둘 다 출력). 대조군(control): 규칙이 성공한 행 일부를 함께 보내 일치율 확인.
- main_s0 검증(Haiku): 1차 시도는 36행 모두 빈 값(프롬프트 끝만 보고 판단) → 질문·응답을 분리해 재시도. 대조군 일치 18/20(Haiku가 답이 있는데 빈 값 2건). 대상 16행: 정정 3(pid 7 ST·TSR 'quarter', pid 62 ST), 하락 2(pid 238 ST·TSR: 규칙은 잘린 추론의 마지막 줄에서 우연히 정답을 집었고, LLM은 '최종 답 없음'). 정확도(규칙 → 규칙+LLM): ST 55.0 → 56.7, TSR 58.3 → 58.3, VisRef 57.6 → 57.6.
- 주의: 공식 normalize는 객관식에서 추출값이 비어도 가장 비슷한(가장 짧은) 보기를 고른다. 이 때문에 빈 추출이 우연히 맞거나(62) 틀린다(238). 공식 동작이므로 그대로 둔다.
- 결론: 영향은 방법당 ±1문항 수준. 8%의 대부분은 추출 실패가 아니라 답을 쓰지 않은 경우(s1 강제 답변으로 대응). 본 실험 보고에는 두 정확도를 함께 적고, LLM 추출 때마다 대조군 일치율을 보고한다.
- Gemini 추출 경로 추가(작성자 요청): `repro/gemini_extract.py` — llm_extract.py prepare가 만든 requests.jsonl을 Gemini API로 문항당 1회 호출. 공식 GPT 추출기와 같은 조건(텍스트 1개, temperature 0, 최대 1024토큰), 표준 라이브러리만, 재개 가능, 모델 ID는 인자로(기본값 없음). 오프라인 테스트(응답 파싱·thought 파트 제외·차단 응답·재개)만 통과, 실제 API 호출은 작성자 실행으로 확인 예정. 채택 기준: main_s0 대조군 일치율로 Haiku(18/20)와 비교.
- handoff/main_s0_extract/: 외부 추출기(Gemini)용 입력만 담은 폴더(정답·채점 파일 없음). 결과 responses_gemini.jsonl → repro/llm_extract.py merge --responses
- Gemini 3.8 Flash(에이전트가 items.jsonl을 직접 읽음) main_s0 결과: 형식 정상(36줄, id·순서 일치), 빈 값 17. 대조군 일치 17/20(Haiku 18/20). 정확도(규칙 → 규칙+LLM): ST 55.0 → 55.0, TSR 58.3 → 55.0, VisRef 57.6 → 57.6.
  - Haiku와 다른 5행: 7|st·7|tsr·720|tsr(응답이 'The answer is B) quarter' / 'the answer is 2'를 반복 명시) → Gemini가 놓침. 876|tsr('Therefore, the answer is'에서 잘림)·303|st('the answer is A' 뒤 'might be B) no'로 번복하다 잘림) → Haiku는 'A', Gemini는 ''(판단이 갈리는 경우).
  - 규칙과 LLM이 엇갈린 대조군 3행(577|st, 637|visref_raw, 303|st)은 모두 답변이 1024 상한에서 잘리며 답을 번복한 경우. 규칙은 중간의 'answer is X'를 잡고, LLM은 최종 답 없음으로 봄.
  - 결론: 두 추출기 모두 이 어려운 부분집합에서 2~3행 차이. Gemini는 보수적(명시된 답도 놓침), Haiku는 번복·절단된 응답에서 보기를 고르는 경향. 쟁점 행은 모두 답변 상한 절단·번복 → b_test의 s1 강제 답변으로 줄어들 것으로 예상.
  - 제안 규칙: 두 추출기를 모두 돌려 일치하면 채택, 불일치 행만 상위 모델이 원문을 읽고 판정. 규칙만 정확도도 함께 보고.
- 2026-10-07 b_test 실제 push 버전 확인: 08:43 UTC에 감사 전 빌드(6c7daca6ebb1, st/tsr/visref_raw_rank)가 push됨(push.json). 이후 빌드는 같은 폴더를 덮어썼으므로 `b_test2`(arki13/visref-repro-b-test2)로 다시 빌드했다(0f8e003fbf90, 4방법). b_test는 '파일럿 v1'로 pull해 같은 판정 규칙으로 보되, visref_raw 관련 항목은 없음. 두 버전의 행은 섞지 않는다.

## 2026-10-07 b_test(파일럿 v1, 6c7daca6ebb1) 결과 — 작성자가 약 8분 만에 중단
- 실행 버전 확인: env.json run_args st/tsr/visref_raw_rank, log code_version 6c7daca6ebb1. error 0, fast_kv ok. 59행, 세 방법 모두 끝난 문항 18(셔플 순서 완전 접두 10). 건전성 기준 '≥40문항'은 중단으로 미충족(판정 불가).
- step 1이 1024 상한에서 잘린 비율: 8/18.
- 이어가기: TSR·VisRef 모두 18문항 중 1문항(pid 970, H1=1.08)만 2단계. 잘린 단계 뒤 probe도 H≈0(0~0.07)이라 대부분 k=1 정지. 생각 토큰 비율 vs ST 1.17배(논문 Fig. 3 약 3.5배). → 이 H 정의에서는 다단계가 거의 일어나지 않는다. Table 5(지연 +8%, +15%)와는 맞고 Fig. 3과는 다르다.
- CRN: TSR k=1 17문항 중 최종 답이 ST와 같음 13(GPU batch 구성에 따른 fp16 차이로 일부 다름).
- VisRef(rank) 주입 수: 중앙값 76(m의 약 31%, |V|의 약 9%), 범위 50~184.
- 처리량(초반 8분, 부정확): 3방법 18문항 → 4096 상한(main_s0, 시간당 59문항)보다 2배 이상 빠를 전망.
- 정확도는 판단에 쓰지 않음.

## 2026-10-07 결정(작성자): s1 + DEER af 폴백, 정지 통계 현행 유지, 파일럿 없이 본 실험 시작
- 근거: b_test v1에서 s1 강제 답 26 중 12(46%)가 답 상한 1024 도달 — 강제 답이 없던 main_s0(14/30, 47%)과 같음. 대부분 'Final Answer: ... B.' 뒤 'Wait, wait …'로 번복. DEER 공식 코드(`deer/vllm-deer.py` L485-491, L655-656, af=1)는 답이 EOS 없이 끝나면 `\n**Final Answer**\n\boxed` + 최대 100토큰을 붙인다.
- 구현(code_version 79eef648ebd1, CPU 테스트 74 passed): 최종 답(자연·강제)이 EOS 없이 답 상한에 닿으면 af 프롬프트 + 최대 100토큰(EOS 종료). 채점 텍스트 = 답 + af 프롬프트 + 이어진 생성. 기록: final_af, final_af_valid(상자 닫힘·내용 있음), final_af_finish, final_af_len. summarize: n_final_forced_unusable(s1 답 EOS 종료도, af 유효 상자도 아님), n_final_af, n_final_af_valid.
- 정지 통계 H: 현행 유지(논문 정의 '답 분포 엔트로피'에 가장 가깝다). b_test v1처럼 k≈1이면 TSR≈ST이고 VisRef 효과는 최종 답 전 1회 주입에서만 생긴다는 점을 결과로 보고한다. 목표 수치에 맞춘 H 조정은 하지 않는다.
- 본 실험 시리즈 `b_main`: 첫 세션 `b_main_s1`(arki13/visref-repro-b-main-s1), 7200s(투입 4800s, deadline 6600s), seed-0 순서 처음부터, st/tsr/visref_raw_rank/visref_raw, 단계 1024 / 합계 10,240.
- 첫 세션 판정(결과 전에 고정, 정확도 차이는 쓰지 않음):
  1. 건전성: error 0, fast_kv ok, 네 방법 모두 끝난 문항 ≥40, 강제 답 사용 불가(n_final_forced_unusable) ≤ 강제 답의 10%, af 유효율 보고.
  2. 통과 → `b_main_s2`부터 `--skip-from b_main_s1/.../results_*.jsonl`(.partial 제외), 코드·인자 불변(code_version 일치).
  3. 실패 → 중단, 원인 수정 후 새 시리즈(b_main 결과와 섞지 않음).
  4. 처리량으로 최종 n 추정. 4방법으로 n<150이면 방법 축소 여부를 작성자에게 묻는다.
  5. 최종 보고: 규칙만 / 규칙+LLM(Gemini·Haiku, 불일치 행은 상위 모델 판정) 두 정확도, paired 차이와 95% CI, step-1 종료 유형별 결과, k 분포, 주입 수/m, 생각 토큰 비율.
- b_test2(0f8e003fbf90)는 push하지 않고 폐기한다(af 폴백 반영 전 빌드). b_main_s1이 대체한다.

## 2026-10-07 b_main_s1 결과와 판정 (79eef648ebd1, 2h 세션, 실사용 약 1.65h)
- 건전성: error 0, fast_kv ok, 네 방법 모두 끝난 문항 74, 강제 답 사용 불가 0(강제 44~45 중 EOS 종료 29~31, 나머지 14~15는 af 폴백으로 전부 유효 상자). → 통과, 같은 코드·인자로 b_main_s2 빌드(--skip-from s1, 296쌍).
- step 1 상한 절단 45/74(61%). 다단계 발생: TSR k>1 18/74(절단 문항의 평균 k 3.11, k=10 7건), VisRef 두 변형도 비슷. 생각 토큰 ST와 비교해 TSR 2.76배, VisRef 2.59~2.66배(논문 Fig. 3 약 3.5배). b_test v1(18문항, 1/18)보다 다단계가 많다 — 소표본 차이.
- 자연 종료 뒤 probe H≥δ: TSR 0/29, VisRef(fill) 0/30, VisRef(rank) 6/35.
- 주입 수: rank 중앙값 104.5(m의 22%), fill = m(중앙값 345.5).
- CRN: TSR k=1 56문항 중 ST와 같은 최종 답 38.
- 추출: 규칙 실패(last_line) 방법당 1행, extract_failed 0 → LLM 추출 영향 미미.
- 처리량: 샤드 불균형(샤드 0 24문항, k≥7 행 15 / 샤드 1 50문항). 세션당 약 74문항(4방법).
- 정확도(참고, 판정에 쓰지 않음, n=74): ST 59.5 / TSR 56.8 / VisRef(fill) 59.5 / VisRef(rank) 56.8. paired 95% CI 모두 0 포함.
- 예상: s2·s3(각 약 1.65h) 후 약 1.7h로 짧은 s4 → 총 약 260문항(>150, 방법 축소 불필요). s4 길이는 s3 후 결정.

## 2026-10-07 계획 수정(작성자 결정, s2 결과 전): fill 제외, 주 결과 VisRef = visref_raw_rank
- 근거: `visref_raw`(rank 이후 relevance 채움)는 VisRef·Chen·CEIL·Kulesza&Taskar 어디에도 없는 구현상 변형이다. `visref_raw_rank`는 인용 Chen 공식 코드의 실제 동작(ε에서 정지)과 같다. fill을 빼면 생성 토큰이 약 28% 줄어 같은 시간에 처리하는 문항 수가 늘어난다(s1 기준 st 79k / tsr 176k / rank 173k / fill 169k 토큰).
- s1 결과를 본 뒤의 결정이지만, s1에서 fill이 더 높았으므로(59.5 vs 56.8) VisRef에 유리한 방향의 선택이 아니다.
- 남는 세 방법의 결과 분포는 바뀌지 않는다: 문항별 난수열은 pid로 정해지고(CRN), step-1을 대표로 돌리는 job은 계속 st다. 바뀌는 것은 batch 구성뿐(fp16 수치 잡음 수준). code_version 79eef648ebd1 그대로, 인자만 `--methods st,tsr,visref_raw_rank`. 이 변경은 '인자 불변' 규칙의 문서화된 예외다.
- b_main_s2 재빌드(--skip-from s1). 집계: `summarize.py … --methods st,tsr,visref_raw_rank`(summarize.py에 방법 필터 추가, code_version 대상 아님). `visref_raw`는 s1 74문항만 있는 보조 자료로 남긴다.
- 주 결과: ST / TSR / VisRef(rank). 최종 n 예상: 세션당 약 100문항(3방법), 약 300문항 이상.

## 2026-10-07 안 쓰는 코드 분기 제거 (작성자 결정, s2 push 전) — code_version 79eef648ebd1 → f2422d0e72ff
- 제거: VisRef 변형 4개(`visref` 태그 주입, `visref_reg`, `visref_raw` fill, `visref_raw_reg`), DPP fill·εI 분기, probe `newline` 모드(smoke3), 강제 답 상자 모드(`forced_cap`), 강제 답·af 끄기 옵션, 엔진 `soft_stop/min_new`, 특수 토큰 `img_*`·`newline_ids`, 관련 테스트·집계 항목. 남은 방법: st / tsr / visref_raw_rank. 결과 행 형식은 그대로.
- 줄 수: run.py+visref+tests 1,996 → 1,706. CPU 테스트 57 passed(통합 테스트 포함).
- 동일성 검증: CPU 작은 LM(로짓 편향으로 경로 유도)에서 이전/새 코드를 같은 입력·시드·chunk로 run.py와 같은 첫 단계 공유 후 갈라지는 방식으로 실행, 결과 레코드 전체 비교 → 810/810 동일. 실행된 경로: step 1 close/cap/eos, s1 강제 답, af(유효·무효), 답 안 </think>, k>1, 정지 entropy/kmax/budget, rank 주입. 기록 `repro/.backup_79eef648ebd1/EQUIVALENCE.md`, 스크립트 `equiv_check.py`, 이전 코드 같은 폴더.
- 따라서 b_main_s1(79eef648ebd1)과 s2 이후(f2422d0e72ff)는 같은 실험으로 합쳐 집계한다(summarize의 code_versions에 두 버전이 보이는 것은 이 기록으로 설명된다).
- b_main_s2 재빌드: 새 코드, `--methods st,tsr,visref_raw_rank`, --skip-from s1(296쌍), 7200s.

## 2026-10-07 최종 보고서 틀 `repro/report.py` (s1 데이터로 작성, 판정 규칙 변경 없음)
- 출력: `repro/reports/b_main_report.html`(+ 같은 이름 `.json`). 내용: 세 방법 정확도와 Wilson 95% CI(논문 Table 6 표시), 짝지은 차이(부트스트랩 95% CI, McNemar 정확 p), 첫 단계 종료 방식(close/cap/eos)별 정확도·평균 k, 단계 수 k 분포(그림+표), 정지 사유·생각 토큰 비, VisRef 주입 통계, 강제 답·af·추출 규칙, 실행 점검(오류, code_version, CRN).
- 집계 대상은 summarize.py와 같은 complete prefix(세 방법 모두 끝난 셔플 앞부분). 주입 통계는 다음 단계로 이어진 단계만 센다(DPP 선택은 모든 단계 끝에 기록되지만 주입은 이어갈 때만 일어난다).
- CRN 기록 정정: s1의 "TSR k=1 56문항 중 ST와 같은 최종 답 38"은 답 전문 비교였다. 추출 답 일치 56/56, 정오 일치 56/56. 전문이 다른 18건은 fp16 배치 구성 차이로 샘플링이 중간에 갈라진 것이다(같은 문맥·같은 시드 확인).
- s1 주입 통계(VisRef rank): 주입 92회, m 중앙값 307, 주입 수 중앙값 105(25–75%: 76–148), 67/92회는 Z 고유 토큰 수에서 멈춰 m보다 적게 들어갔다.
- 세션마다 다시 생성(결과 파일을 모두 넣고, LLM 재채점 파일이 생기면 `--llm` 추가):
  `cd repro && python3 report.py kaggle/runs/b_main_s1/output/results/results_{0,1}.jsonl kaggle/runs/b_main_s2/output/results/results_{0,1}.jsonl … --order kaggle/runs/b_main_s1/output/results/ids.json --shards 2 --label "b_main s1–sN" --out reports/b_main_report.html`

## 2026-10-08 b_main_s2 결과(f2422d0e72ff, 2h 세션, 실행 약 6,060s)와 s3 빌드
- 건전성: error 0, 세 방법 114문항(342행), 강제 답 사용 불가 0. code_version f2422d0e72ff(s1 79eef648ebd1과 출력 동일 확인됨) → 통과, 같은 코드·인자로 `b_main_s3` 빌드(--skip-from s1+s2, 638쌍, 7200s, source sha256·run_args·ids 모두 s2와 같음).
- 처리량: 세 방법으로 세션당 114문항(4방법 때 74).
- s1+s2 합산 complete prefix 188문항(참고, 판정은 최종 집계에서): ST 57.4 / TSR 58.5 / VisRef 60.6. paired: TSR−ST +1.1(−2.1~+4.3), VisRef−ST +3.2(−1.1~+7.4, McNemar p 0.21), VisRef−TSR +2.1(−1.6~+5.9). 첫 단계 상한 절단 123/188(65%)에서 ST 51.2 / TSR 52.8 / VisRef 55.3, 자연 종료 60문항은 70.0 / 70.0 / 71.7(k=1, fp16 잡음 1문항).
- 생각 토큰 ST와 비교해 TSR 2.62배, VisRef 2.85배. k>1: TSR 54, VisRef 58. kmax 정지 TSR 12, VisRef 19.
- CRN: TSR k=1 134문항 중 ST와 추출 답 일치 130, 정오 일치 131, 전문 일치 88.
- VisRef 주입 271회, 중앙값 123, m 중앙값 537, 225/271회가 Z 고유 토큰 수에서 멈춤.
- s3(2h) 뒤 짧은 s4, 길이는 s3 뒤 결정. 예상 최종 n ≈ 350.
- 보고서: `repro/reports/b_main_report.html` 재생성(아티팩트 v2).

## 2026-10-08 해석 메모(방향 일치 / 토큰 상한·thinking / m=30%) — 실험 설정 변경 없음
- 원문 근거: §5 "MLRMs in their reasoning (thinking) mode", App. I 1B/2B/8B "under the same decoding setup and token budget"(값 없음). Fig. 4(b) 8B: m 20%≈77(캡션 76.1) → 30·40·50% ≈79.2로 평평, "plateaus for m ≥ 30%", 30%는 정확도·효율 균형으로 선택(최고점이라고 쓰지 않음). 본문 Eq. 11 주변 "enforcing |Ṽk| = m".
- 설정 출처 정정: R1 프롬프트·T=0.6·top-k/p는 논문이 아니라 모델 카드 값(SPEC_LEDGER 3·4행과 같음). report.py 문구 수정, 아티팩트 v3.
- 데이터(s1+s2, 188문항): 첫 단계 자연 종료 뒤 H≥δ는 TSR 0/62 → 이어가기는 거의 잘린 단계에서만 일어남. 잘린 생각을 오답으로 치면 ST 23.9%(close 42 + eos 3 정답), 강제 답으로 57.4%; 논문 46.1은 그 사이. main_s0(60문항, 단계 4096)에서도 ST 생각 절반이 4,096 안에 끝나지 않음(잘림=오답이면 1024 23.3 / 2048 30.0 / 4096 36.7%).
- 주입 포화(VisRef 이어간 단계 271): rank = min(Z 고유 토큰, rank_V)이고 n_dpp가 271/271 이 값과 같다. rank/|V| 분위 10/25/50/75/90% = 0.034/0.044/0.059/0.197/0.578. m이 실제로 묶는 단계: 10% 36%, 20% 25%, 30% 17%, 40% 14%, 50% 13%. 평균 주입 98/113/119/123/127, m=100%면 132. 단계 길이별 Z 고유 토큰 중앙값: ≤256 67, 256-512 130, 512-1023 140, 1024 145.

## 2026-10-08 비용 측정 현황(논문 Table 5 같은 추가 시간) — 설정 변경 없음
- 논문 Table 5: InternVL3.5-8B, H100 1장, MathVista 문항당 지연 ST 7.1s / TSR 7.7s / Look-Back 7.6s / VisRef 8.2s(ST보다 +1.1s, TSR보다 +0.5s). 1B 지연은 없음. Fig. 3의 trace당 약 3.5배 생각 토큰과는 논문 수치 간 차이가 있음(원인 미확인; AUDIT A-4, 4c).
- 우리 기록: 문항당 지연은 측정하지 않았다. `wall_s`는 16개 동시 배치·step-1 갈라짐 대기가 섞여 지연이 아니다(예: TSR k=1은 ST보다 약 16토큰 더 생성하는데 wall_s 288s vs 197s).
- 생성 토큰(s1+s2 188문항, 생각+답+probe): ST 1,121 / TSR 2,317(2.07배) / VisRef 2,580(2.30배). k=1 문항(약 70%)은 TSR 925 vs ST 909, VisRef 975 vs ST 897. k>1 문항은 TSR 5,771 vs ST 1,648, VisRef 6,176 vs ST 1,625.
- DPP 선택 시간(장치 CPU 4스레드, float64, d=1024, Z 1,024토큰·고유 145): |V| 768/1,792/3,328 → 0.045/0.059/0.127s, 선택 145개(rank에서 멈춤). 생성 시간에 비해 무시할 수준.

## 2026-10-08 독립 검토 결과와 기록 정정
- 결론: s3는 그대로 push해도 된다(빌드 무결성 확인: notebook 내 소스 sha256·code_version f2422d0e72ff·run_args·ids·time_limit/deadline이 s2와 같고 skip 638쌍; s1·s2 로그에 deadline·error 없음 → complete prefix = 투입 집합). pull 뒤 확인: error 0, env.json gpus=2, code_version.
- 정정 1(주입 범위): Alg. 1은 V_k를 궤적에 넣은 뒤 정지 검사를 하고, 코드(`methods.py` L174-189)도 매 단계 ctx에 V_k를 붙인 뒤 probe·최종 답을 만든다. 따라서 마지막 단계의 선택도 최종 답 직전에 주입되며 k=1 VisRef ≠ ST다. 2026-10-07 보고서 항목의 "주입은 이어갈 때만 일어난다"와 2026-10-08 s2 항목의 "자연 종료 71.7은 fp16 잡음 1문항"은 틀렸다. report.py 주입 통계를 전 단계로 되돌림: s1+s2 주입 459회, 중앙값 123, m 중앙값 537, 386/459회가 Z 고유 토큰 수에서 멈춤, 주입/m 중앙값 0.20. 2026-10-08 해석 메모의 포화 분석(이어간 단계 271개)은 부분집합 기준이며 결론은 같다.
- 발견(검토자 P3): 최종 답 안에 </think>가 다시 나온 행 ST 33 / TSR 25 / VisRef 83(188문항), 자연 종료·k=1에서는 8 / 9 / 30(60문항). 답 직전 주입이 답 안에서 생각을 다시 열게 하는 것으로 보인다. 채점은 마지막 </think> 뒤 텍스트. 최종 보고에 첫 답 기준 VisRef 재채점을 함께 낸다(보고서에 "답 안 </think>" 열 추가).
- 검토자 해석(추정): 이어가기는 사실상 상한 절단에서만 발생(자연 종료 뒤 H<δ 60/60) → 최종 보고는 cap/close 층별로 제시. Fig. 4(b)의 20→30% 상승은 rank ≥ 0.2|V| ≈ 354를 뜻하고, hidden-state Z(rank ≤ T_k ≈ 615)라면 30% 뒤 평탄이 rank 포화로 설명된다 → hidden-state Z가 여전히 가장 강한 후보(저자 구현은 미확인; 기록의 "인용 코드가 rank만큼만 고르므로 가설 약화"와 다른 판단). 별도 시리즈 과제.
- 검토자 권고: s3 그대로 → 짧은 s4보다 thinking 끈 ST 진단(같은 pid, 별도 커널)이 정보량이 크다. 다음 주에 n=1,000 완주(약 12h) 우선, hidden-state Z 변형은 별도 시리즈(약 300 pid, 약 5h). 통계: n≈350이면 VisRef−ST ±3.0pp, n=1,000이면 ±1.8pp.

## 2026-10-08 진단 hidden_rank_check (CPU 커널, GPU 사용 안 함) — 결과 전에 고정
- 목적: VisRef 커널 L = (VZᵀ)(VZᵀ)ᵀ에서 Z를 입력 임베딩 대신 은닉 상태로 쓰면, 1B에서 greedy DPP가 "의미 있게" 고를 수 있는 시각 토큰 수가 m = ⌊0.3|V|⌋에 닿는지 측정. 은닉 상태 Z 변형 시리즈를 만들 가치가 있는지 판단하는 용도. 본 실험(b_main)과 무관.
- 실행 위치: Kaggle CPU 커널 `arki13/visref-hidden-rank-check`(fp32, transformers 4.55.0, 모델·데이터 리비전은 본 실험과 같음).
- 표본(결과를 보지 않고 규칙으로 선택): s1+s2 complete prefix에서 셔플 순서대로, 이미지 크기(|V| ≤768 / ≤1792 / 그 이상) × 첫 단계 종료(cap / close) 6층에서 앞의 2개씩 = 12 pid(`repro/diag/hidden_rank_samples.json`). 각 pid의 첫 단계 생각(ST trace, 세 방법 공통)을 다시 토큰화해 프롬프트 뒤에 붙이고 한 번 forward.
- Z 후보: 입력 임베딩(대조군), 층 7·14·21·28 출력, 최종 norm 출력. V 후보: 입력 공간 시각 토큰(실제 주입되는 것) / 같은 층의 시각 토큰 은닉 상태.
- 측정: greedy MAP(Chen 재귀, float64, visref/dpp.py와 같은 선택; CPU 테스트 2개로 일치 확인)을 정지 없이 |V|/2까지 돌려 각 선택의 정규화 잔차 d²/r²(Eq. 10 이득의 다양성 인자)를 기록. 1차 지표 = 정규화 잔차가 처음 1e-2 아래로 떨어지기 전 선택 수(n_meaningful). 보조: 1e-1/1e-3/1e-6 기준, 현행 정지 규칙(1e-9·max r²)의 선택 수, Z 특이값 스펙트럼.
- 타당성 조건: 입력 임베딩 Z + 입력 공간 V의 정지 규칙 선택 수가 기록된 n_dpp(= Z 고유 토큰 수)와 같아야 한다. 다르면 진단 무효.
- 해석 규칙: (a) 최종 norm(또는 마지막 층) Z에서 잘린 단계(T=1024)의 n_meaningful ≥ m이 과반이면 → 은닉 상태 Z로 1B도 m개를 채울 수 있음 → 은닉 상태 Z 변형 시리즈를 사전 규칙과 함께 설계. (b) n_meaningful이 m의 절반 미만에 머물면(은닉 상태가 한 방향으로 쏠려 있어서) → 은닉 상태 Z로는 "m개 주입"이 설명되지 않음 → 변형 우선순위를 낮추고 채움 가설을 남김. 자연 종료로 짧은 단계(T≈76–100)는 계수 ≤ T라 m에 못 미치는 게 당연하므로 따로 보고.
- 한계: 첫 단계만 본다(이후 단계는 주입된 토큰이 문맥에 섞임). 1B·fp32 CPU 수치. 층 선택은 논문에 없으므로 여러 층을 함께 보고한다.
- 산출물: `results/hidden_rank_check.json`, `log.txt`, `env.json`. push: `python3 repro/kaggle/kaggle_ctl.py push hidden_rank_check`(제한 시간 3시간).

## 2026-10-08 b_main_s3 결과(f2422d0e72ff, 실행 약 5,950s)
- 건전성: error 0, 세 방법 108문항(324행), 강제 답 사용 불가 0, code_version f2422d0e72ff, gpus T4×2, torch 2.11.0+cu128·transformers 4.55.0(s2와 같음), 로그에 deadline·error 없음. 샤드 불균형 36/72문항.
- s1+s2+s3 complete prefix 296문항(참고): ST 59.1 [53.4, 64.6] / TSR 61.1 / VisRef 61.1. paired TSR−ST +2.0(−1.0~+5.1, 14:8, p 0.29), VisRef−ST +2.0(−1.7~+6.1, 20:14, p 0.39), VisRef−TSR 0.0(−3.0~+3.0, 12:12). 첫 단계 상한 절단 187(63%): 51.9 / 54.5 / 55.1, 자연 종료 99: 71.7 / 71.7 / 71.7.
- 생각 토큰 ST와 비교해 TSR 2.58배, VisRef 2.86배. 답 안 </think>: 45 / 38 / 138. VisRef 주입 722회, 중앙값 124.5, m 중앙값 537, 618/722가 m보다 적음.
- CRN: TSR k=1 211문항 중 ST와 추출 답 206, 정오 207 일치.

## 2026-10-08 hidden_rank_check 결과와 판정 (Kaggle CPU, 25분)
- 타당성: 입력 임베딩 Z의 정지 규칙 선택 수 = 다시 토큰화한 Z의 고유 토큰 수 12/12. 기록된 n_dpp와 비교하면 m이 걸린 3건(|V|=256)은 76=m으로 맞음, 나머지 9건 중 7건 일치·2건 ±1(재토큰화로 고유 토큰 수도 1 차이). → 유효.
- 1차 지표(정규화 잔차 ≥ 1e-2인 선택 수), 잘린 단계 6건(T=1024): 최종 norm Z 94/998, 70/998, 107/537, 86/537, 58/76, 32/76; 28층 Z 88, 57, 74, 90, 52, 53; 입력 임베딩 Z 77, 80, 76, 59, 45, 65. m 이상 0/6, m/2 미만 최종 norm 5/6·28층 4/6.
- 사전 규칙 판정: (b). 은닉 상태 Z도 의미 있게 고를 수 있는 시각 토큰은 약 50–110개로 입력 임베딩과 비슷하며 m에 못 미친다 → "은닉 상태 Z라서 다양한 토큰을 m개 고른다"는 설명은 지지되지 않음, 은닉 상태 Z 변형의 우선순위를 낮춘다.
- 결과를 본 뒤의 관찰(사전 규칙 밖, 해석 참고용): 은닉 상태 Z는 수치 계수가 거의 T라서 현행 정지 규칙(1e-9)으로는 958–993개(T=1024)를 고른다 → 실제 구현하면 min(m, T)개를 주입하게 되지만, 약 100번째 이후 선택은 정규화 잔차 1e-3~1e-9(다양성 기여 거의 0)로 사실상 관련성 위주의 채움이다. Z 에너지 분포의 participation ratio 3–12(소수 방향에 집중). 같은 층 시각 토큰 은닉 상태를 V로 쓰면 의미 있는 선택 3–45개로 더 적다. 따라서 "은닉 상태 Z"와 "계수 이후 채움"은 실질적으로 같은 개입(m개 주입, 앞 약 100개만 다양성 구분)이며, 남는 질문은 "m개 주입이 rank만큼 주입보다 정확도를 바꾸는가"다. 은닉 상태 Z는 Eq. 11 이득이 m까지 유한하게 정의되는 유일한 읽기라는 점에서 본문("|Ṽk| = m", "terminate after m iterations")과 형식상 맞는다.

## 2026-10-08 b_main_s4 빌드 (이번 주 마지막 세션)
- 세션이 도중에 끝나면 출력이 사라질 수 있으므로 timeout 6,000s로 여유를 둔다(build.py 규칙: time_limit 4,000s, deadline 5,400s; s3 실측 준비 시간 약 51s).
- 코드·인자·문항 순서는 s3와 같음(source sha256·run_args·ids·모델/데이터 리비전 일치, code_version f2422d0e72ff). --skip-from s1+s2+s3(962쌍). 세션 길이만 다르며 이는 사전에 계획된 변경.
- 예상: 약 80문항 추가(s3 108문항 × 4,000/4,800), 합계 약 376문항.

## 2026-10-08 b_main_s4 결과와 이번 주 집계 (s1–s4, 392문항)
- s4 건전성: error 0, 세 방법 96문항(288행), 강제 답 사용 불가 0, code_version f2422d0e72ff, T4×2, torch·transformers s3와 같음, deadline·error 없음. 세션 약 5,466s.
- s1–s4 complete prefix 392문항(투입 전부): ST 58.9 [54.0, 63.7] / TSR 60.7 [55.8, 65.4] / VisRef 61.7 [56.8, 66.4]. paired TSR−ST +1.8(−0.8~+4.6, 18:11, p 0.27), VisRef−ST +2.8(−0.5~+6.1, 28:17, p 0.14), VisRef−TSR +1.0(−2.0~+3.8, 19:15, p 0.61). 논문 차이 +2.4 / +5.9 / +3.5.
- 첫 단계: 상한 절단 245(63%) 51.8 / 54.3 / 55.9, 자연 종료 134 71.6 / 71.6 / 71.6, EOS 13. 생각 토큰 ST와 비교해 TSR 2.57배, VisRef 2.92배. kmax 정지 TSR 24, VisRef 42. 답 안 </think> 54 / 47 / 176. VisRef 주입 963회, 중앙값 122(m 중앙값 537). CRN: TSR k=1 282문항 중 ST와 정오 276·추출 답 274 일치.
- 규칙 실패(LLM 추출 대상): 11 / 7 / 8행.
- report.py: 중간 결과 배너를 문항 수(≥250) 대신 `--final` 플래그로 끄도록 변경(재채점·남은 세션 전에는 계속 표시).
- 다음 주 계획을 위한 계산(관측 불일치율 기준): VisRef−ST가 +2.8이 유지되면 n=1,000에서 검출력 약 0.74, 논문 +5.9와의 차이(3.1pp)도 약 2.9 SE로 구분 가능. 남은 608문항은 약 9.5시간(세션당 약 65문항/시간).

## 2026-10-08 메모: 계수를 넘은 뒤 "식 그대로" 고르면 무엇이 채워지나 (설정 변경 없음)
- 정확한 수학: 계수를 넘으면 모든 후보의 이득이 log r² + log 0 = −∞로 같아 Eq. 11의 argmax가 정해지지 않는다. 식 자체는 "관련성으로 채우라"고 말하지 않는다.
- 구현에 따라 갈린다(합성 데이터 N=600, 계수 40, 계수 이후 60개 선택, 남은 토큰 중 관련성 백분위):
  - Chen·CEIL 인용 코드: ε 정지 → 채우지 않음(현행 visref_raw_rank).
  - 정지 없이 부동소수 그대로: 반올림 잡음으로 고름(백분위 중앙값 0.42, 상위 10% 비율 0.03), NaN·overflow 발생.
  - 작은 εI를 더한 커널(1e-6·평균 대각, 1e-3도 같음): 백분위 중앙값 0.98, 상위 10% 비율 0.93 → 사실상 관련성 순 채움. 계수 이전 40개 선택은 정확한 경우와 동일.
- 따라서 "m개 주입" 변형을 실험한다면, 식을 바꾸지 않고 수치 안정화만 더한 εI(= 계수 이후 관련성 채움)가 가장 자연스러운 후보다. 2026-10-07 정리 때 제거한 `visref_reg`(εI)와 `visref_raw`(관련성 채움)가 이에 해당하며 `.backup_79eef648ebd1/`에 있다.

## 2026-10-10 b_main_s5 빌드
- 코드·인자·문항 순서 s4와 같음, --skip-from s1–s4(1,176쌍), 7200s. 남은 608문항은 약 6세션(약 10시간) 예상.
- 정정(같은 날): s5 빌드 확인 결과 `visref/scoring.py`가 2026-10-08에 scorer v3(`SCORER_VERSION=v3_md_colon_emphasis`, 마크다운 라벨·콜론·강조 처리)로 바뀌어 code_version이 f2422d0e72ff → d0095f79e691이 됐다. run.py·methods·dpp·engine·model·fastkv의 sha256은 s4와 같아 생성 과정은 동일하고, 바뀐 것은 기록 시점의 규칙 채점뿐이다. 모든 세션 결과는 `repro/rescore.py`(추론 없이 저장된 최종 답을 현재 채점 규칙으로 다시 채점)로 같은 v3 규칙에 맞춰 집계한다. 이 조건에서 s5 이후를 b_main에 합친다(문서화된 예외). push 전 작성자 확인.
- v3 재채점 확인(rescore.py, 추론 없음, s1–s4 392문항): ST 231→233 (59.4%) / TSR 238→239 (61.0%) / VisRef 242→243 (62.0%), 추출 실패 1/1/1행. 방법 간 차이는 거의 그대로. 게시된 보고서(v6)는 아직 v1 채점 기준.

## 2026-10-10 작성자 결정·확인
- 잘린 생각 처리 오프라인 확인(v3, 392문항): 강제 답 행을 오답으로 치면 ST 26.8 / TSR 28.1 / VisRef 28.6%. 논문(46.1/48.5/52.0)보다 훨씬 낮아, "단계 1024 + 잘린 답 버림"으로는 논문 절대값이 설명되지 않는다. 강제 답 행 정답: ST 128/245, TSR 129/237, VisRef 131/234.

## 2026-10-10 결과 전 고정: b_main 주 비교 (작성자 결정)
- 주 비교 2개: VisRef−ST, VisRef−TSR (논문 주장 "gains over both ST and TSR", App. I·Table 1; 초록의 "test-time scaling 방법보다 높다"). 보조: TSR−ST. 중간 결과(VisRef−TSR +1.0)를 본 뒤의 결정이지만 근거는 논문 주장이다.
- 검정: 같은 문항 짝 McNemar 정확 검정, 두 주 비교에 Holm 보정(작은 p를 0.025, 큰 p를 0.05와 비교). 효과 크기는 짝 차이와 부트스트랩 95% CI. 채점은 규칙 v3 + 남은 추출 실패 행 LLM 추출. 최종 n은 b_main 완주분(목표 1,000) 전체.

## 2026-10-10 진단 diag_st_plain 빌드 (GPU 약 1h 이내) — 결과 전에 고정
- 질문: 논문 ST 46.1이 "잘린 생각 뒤 재촉 문구 없이 그냥 답하게 한 결과"로 설명되는가.
- 표본(규칙): s1–s4에서 본 실험 ST의 첫 단계가 상한에서 잘린 문항을 셔플 순서대로 앞의 150개(`repro/diag/st_plain_ids.json`). 본 실험 ST의 이 150문항 정답 74(49.3%).
- 변경은 한 가지: `--forced-prompt=`(빈 문자열) → 잘린 단계 뒤 `</think>`만 붙이고 답한다. 나머지(ST, 단계 1024, 답 1024, af 폴백, 시드, 코드 d0095f79e691)는 본 실험과 같다. 첫 단계는 다시 생성되므로 GPU 잡음으로 일부 달라질 수 있다(첫 단계 종료 방식이 다른 문항은 따로 센다).
- 판정: 같은 문항 짝 비교(본 실험 강제 답 ST vs 재촉 없음 ST). 재촉 없음 정확도가 강제 답보다 크게 낮아(예: 10%p 이상) 전체 ST 추정치가 46 근처로 내려가면 "답 처리 차이"가 유력한 설명, 차이가 작으면 이 설명은 기각.
- 전체 ST 추정치 = (자연 종료·EOS 문항 정답 + 잘린 문항 수 × 재촉 없음 정확도) / 392.

## 2026-10-10 정정: InternVL3.5-1B MathVista 59.3 / 69.5의 의미 (설정 변경 없음)
- 출처 확인(references/papers/ref_fulltext/InternVL3.5.txt L555-568, L775-776, Table 3): 69.5는 "w/ Parallel Thinking [143]" 행 = Best-of-N, VisualPRM-v1.1을 critic(보상 모델)으로 여러 응답 중 하나를 고름. 다수결이 아니며 N은 본문에 없음.
- 59.3은 "unless otherwise specified … without applying TTS" 조건의 단일 응답 점수. Thinking 모드 사용 여부는 본문·모델 카드 어디에도 명시되지 않음(TTS 정의에 deep thinking이 포함돼 있어 문자 그대로는 '생각 모드 없음'으로도 읽히나 행 이름은 parallel thinking만 언급 → 불명확).
- "59.3 = thinking 끔, 69.5 = thinking 켬" 해석은 근거 없음 → 철회. 우리 단일 응답 ST(v3 59.4)와 비교할 공식 수치는 59.3이고, 69.5는 다중 샘플+PRM 선택이라 비교 대상이 아님.
- 답 강제 규칙 정리: 논문은 최종 답을 y ∼ πθ(·|x, τ1:k)로만 쓰고(Alg. 1 L16-17) 토큰 상한·잘림 처리·답 강제 문구를 하나도 적지 않음. 우리 구현의 두 규칙(생각 단계 상한 → s1식 `\nFinal Answer:`, 답 상한 → DEER af `\n**Final Answer**\n\boxed`)은 2026-10-07 작성자 결정으로 의도적으로 넣은 논문 외 처리이며, 채점(답 추출)과는 별개.

## 2026-10-10 diag_st_plain 판정 규칙 보강 (작성자 승인, 결과 전 고정, 코드·빌드 변경 없음)
- 이유: `--forced-prompt=`는 규칙 ①(생각 상한 뒤 `\nFinal Answer:`)만 끈다. 규칙 ②(답이 EOS 없이 답 상한 1024에 닿으면 DEER af `\n**Final Answer**\n\boxed` + 최대 100토큰)는 diag에서도 작동한다. 본 실험 ST의 이 150문항 중 50행이 af 경로.
- 세 가지 읽기를 diag 행과 짝이 되는 본 실험 ST 행에 똑같이 적용(모두 채점 v3):
  (a) 그대로: 기존 사전등록 판정에 쓰는 주 읽기.
  (b) af 제거 재채점: final_answer를 마지막 af 문구 앞에서 자르고 다시 채점(추론 없음). = 규칙 ①·② 모두 없는 경우의 근사.
  (c) af=오답 하한: af 경로 행을 모두 오답으로 침.
- 보고: 각 읽기의 정확도, af 행 수와 af 행 정확도(diag/본 실험 각각), 같은 문항 짝 차이와 McNemar 정확 p. 전체 ST 추정치 = (자연 종료·EOS 문항의 같은 읽기 정답 + 245 × 해당 읽기의 재촉 없음 정확도) / 392 를 세 읽기 모두 계산.
- 판정: 기존 기준(재촉 없음이 강제 답보다 10%p 이상 낮아 전체 추정치가 46 근처)은 (a)로 판정. (b)는 "논문처럼 답 처리 규칙이 전혀 없었다면"의 근거로 함께 보고, (c)는 하한으로만 쓰고 판정에 쓰지 않는다(답 상한에 닿았어도 앞부분에 추출 가능한 답이 있는 행까지 오답 처리하므로).
- b_main 최종 보고에도 같은 민감도 분석 (a)/(b)/(c)를 세 방법 모두 싣는다. 주 검정은 사전등록대로 (a)=v3(+LLM 추출) 하나.
- 참고(결과를 본 뒤의 관찰, 판정에 쓰지 않음; s1–s4 392문항 v3): (a) 59.4 / 61.0 / 62.0, (b) 57.4 / 58.7 / 59.4, (c) 47.7 / 49.0 / 48.7 (ST/TSR/VisRef). af 행 84 / 70 / 87, 그중 v3 정답 46 / 47 / 52, af 제거 시 정답 38 / 38 / 42. af 구제 효과는 +2.0 / +2.3 / +2.6pp로 방법 간 차이 작음.

## 2026-10-10 push (작성자)
- diag_st_plain: kernel 137987959 v1. b_main_s5: kernel 137987971 v1. 둘 다 push 직전 빌드 무결성 확인(소스 8개 해시·노트북 해시 일치, s5 n_skip 1250, diag 150문항 = st_plain_ids와 같은 집합, 실행 순서는 build.py가 seed 0으로 재셔플).
- 작성자 결정: "출처 표시 / 미확인 표시 유지 / 확인된 수치 대장" 규칙을 프로젝트 지침에 넣음. 대장은 `repro/SPEC_LEDGER.md` 끝 "외부 수치 확인 대장"(X-1~X-5).

## 2026-10-10 report.py v3 통일 재채점·민감도 표, diag 분석 스크립트 (GPU 불필요, 생성 코드 변경 없음)
- 결함 수정: report.py가 행에 기록된 `correct`를 그대로 합산해 s1–s4(v1)와 s5+(v3)가 섞이던 문제. `--rescore-data ../data/raw/mathvista_testmini.json`을 주면 모든 행의 저장된 final_answer를 현재 채점 규칙(`visref/scoring.py`, rescore.py와 같은 `score()` 호출)으로 메모리에서 다시 채점한 뒤 집계한다(correct·extracted·prediction·rule·extract_failed 덮어씀, 기록값은 `correct_recorded`). 실행 점검 칸에 채점 버전과 바뀐 행 수 표시. report.py는 code_version 대상 아님.
- 같은 옵션일 때 "답 처리 규칙 민감도" 표 추가: 사전등록된 (a)/(b)/(c) + 참고로 `\boxed`만 인정(규칙 추출이 boxed에서 답을 찾은 정답만).
- 검증: 플래그 없이 돌리면 JSON이 이전 코드와 같음(새 키 2개 제외). s1–s4에서 기록값 재현: v3 233/239/243 (59.4/61.0/62.0), (b) 57.4/58.7/59.4, (c) 47.7/49.0/48.7, af 84/70/87, af 정답 (a) 46/47/52 · (b) 38/38/42.
- v3 기준 짝 차이(s1–s4, 392, 참고): TSR−ST +1.5(−1.0~+4.3, 16:10, p 0.33), VisRef−ST +2.6(−0.5~+5.9, 26:16, p 0.16), VisRef−TSR +1.0(−1.8~+3.8, 18:14, p 0.60). (v1 기준 +1.8 / +2.8 / +1.0)
- 결과를 본 뒤의 관찰(판정에 쓰지 않음): `\boxed`만 인정 시 ST 50.8 / TSR 53.1 / VisRef 49.5 — 순서가 바뀐다. boxed 밖에서 찾은 정답이 VisRef 49행(답 문구 45, 그중 답 안 </think> 25)으로 ST 34·TSR 31보다 많다. 답 직전 주입이 답 안에서 생각을 다시 열게 하는 현상(2026-10-08 P3)과 연결된다. 엄격 채점은 논문 채점 방식이 아니므로(논문은 답 처리·추출을 적지 않음) 보조 지표로만 보고한다.
- 보고서 아티팩트 v7 = s1–s4, v3 재채점, 민감도 표 포함. 이후 세션마다 같은 명령에 `--rescore-data` 추가:
  `cd repro && python3 report.py kaggle/runs/b_main_s{1..N}/output/results/results_{0,1}.jsonl --order kaggle/runs/b_main_s1/output/results/ids.json --shards 2 --label "b_main s1–sN" --rescore-data ../data/raw/mathvista_testmini.json --out reports/b_main_report.html`
- diag 분석 스크립트 `repro/diag/analyze_st_plain.py`(결과 전 작성, 사전등록 규칙 그대로 구현): 같은 pid 짝, 세 읽기 정확도·차이·McNemar p, af 행 수·정확도(diag/본 실험), diag 첫 단계 종료 방식 분포, 전체 ST 추정치(재촉 없음/강제 답 둘 다, 같은 150문항 부분집합 정확도로 계산). 판정 플래그는 (a)의 "10%p 이상 하락"만 자동 계산하고 "46 근처"는 추정치를 보고 판단한다(수치 기준 미리 정하지 않았음).
- 연습 실행(가짜 diag = 본 실험 ST 행): 차이 0, 본 실험 정답 74/150, af 50행으로 사전등록 기록과 일치. 참고: 부분집합 기준 강제 답 추정치는 (a) 57.6 / (b) 55.5 / (c) 47.5로, 실제 전체 ST(59.4)보다 낮다(150문항 정확도 49.3 < 잘린 245문항 전체 51.8). 비교는 같은 부분집합의 강제 답 추정치와 해야 한다.
- 실행: `cd repro && python3 diag/analyze_st_plain.py kaggle/runs/diag_st_plain/output/results/results_*.jsonl --main kaggle/runs/b_main_s{1,2,3,4}/output/results/results_{0,1}.jsonl --order kaggle/runs/b_main_s1/output/results/ids.json --data ../data/raw/mathvista_testmini.json --ids diag/st_plain_ids.json --out diag/st_plain_analysis.json`
- LLM 추출(규칙 실패 행, s1–s4 v3 기준 10/7/7행, v1 기준 11/7/8행)은 b_main 완주 뒤 한 번에 한다(행 수가 적고 세션마다 반복할 이유 없음).

## 2026-10-10 채점 대상(마지막 </think> 뒤) 유지 확인과 SPEC_LEDGER 핵심 판단 요약 신설 (작성자 결정, 설정 변경 없음)
- 작성자 결정: 주 채점 대상은 "마지막 `</think>` 뒤" 그대로. 근거: VLMEvalKit `SPLIT_THINK`가 `s.split('</think>')[-1]`을 prediction으로 씀(`references/sources/vlmevalkit_postprocess_54a063c5/vlmeval/inference.py` L259, L267-270). VLMEvalKit은 VisRef 인용 문헌이 아니고(VisRef_plain.txt grep 0건) VisRef가 인용한 InternVL3.5[45]의 평가 도구(InternVL3.5.txt L644, 모델 카드 L80)라 근거 등급 D. 옵션 기본값은 꺼짐이고 InternVL3.5 평가에서 켰는지는 미확인. 사전등록된 주 검정 조건이기도 하다.
- 첫 답(답 원문 `final_raw`의 첫 `</think>` 앞) 기준 재채점(결과를 본 뒤, 규칙만, s1–s4): 정답 ST 233→232 / TSR 239→238 / VisRef 243→240. VisRef−ST +2.6→+2.0(27:19, p 0.30), VisRef−TSR +1.0→+0.5(19:17, p 0.87). 답 안 </think> VisRef 176행 중 정오가 갈린 행 5(마지막만 정답 4, 첫 답만 정답 1). 첫 답 51행은 규칙 추출 실패. 보조 민감도로만 보고.
- SPEC_LEDGER 맨 앞에 "핵심 판단 요약" 새로 넣음: 근거 등급 A(논문 명시)–F(운영 선택)와 판단 J-1~J-18, 각 판단의 본 표 행·EF 날짜. 함께 고친 것: 43행 VLMEvalKit 근거 추가, 44행 강제 답 문자열 정정(실제 `\nFinal Answer:`), 45행 LLM 보조 추출 반영, 52행(DEER af) 새로 넣음. 이전 판 `repro/.SPEC_LEDGER.bak_20261010`.

## 2026-10-10 InternVL3.5 공식 평가 설정(VLMEvalKit) 확인 — 설정 변경 없음, 적용 범위는 작성자 결정 대기
- 작성자 제안: VisRef가 인용한 InternVL3.5가 쓴 방법(VLMEvalKit)을 baseline 구현에 그대로 따르는 안.
- VLMEvalKit 커밋 54a063c5의 관련 소스 5개를 받아 `references/sources/vlmevalkit_postprocess_54a063c5/`에 저장(SOURCE.json에 URL·sha256). 대조표는 SPEC_LEDGER "D 자료 대조".
- 확인된 사실: 1B 설정이 두 개(`InternVL3_5-1B` greedy·최대 4,096 / `InternVL3_5-1B-Thinking` R1·T 0.6·top_p 0.95·최대 65,536·lmdeploy). `USE_COT`(답 \boxed 지시문)·`SPLIT_THINK`(생각/답 분리)는 환경변수 옵션. MathVista 채점은 규칙 prefetch 후 실패 시 GPT-4o-mini 추출. 공식 59.3이 어느 설정으로 나왔는지는 미확인.
- 우리와 가장 큰 차이: 최대 생성 토큰(65,536 vs 생각 단계 1,024). 미정 항목 "예산을 늘린 ST"와 직결.
- 해석: VisRef ST 46.1은 공식 59.3보다 13pt 낮다(논문의 평가 설정이 이 공식 설정과 같은지는 미확인). 공식 설정을 따르면 "모델 공식 사용법에 충실한 baseline"이 되지만 VisRef 절대값 재현과는 방향이 다를 수 있다.

## 2026-10-10 작성자 결정: 원칙 채택, b_main 유지, 공식 설정 ST 보조 실험, LLM 추출 모델
- 원칙 채택: VisRef 본문·인용 문헌·인용 코드(등급 A–C)가 정하지 않은 부분은 InternVL3.5 공식 자료(등급 D: 모델 카드, VLMEvalKit 설정)를 기본값으로 삼는다(SPEC_LEDGER 원칙 1).
- b_main은 사전등록대로 유지(설정 변경 없음).
- LLM 추출: VLMEvalKit 공식 판정 모델 GPT-4o-mini 대신 기존 계획대로 Gemini/Haiku(OpenAI 키 없음). 공식과 다른 점으로 보고.
- 공식 설정 ST 보조 실험 st_official을 아래처럼 결과 전에 고정한다.

## 2026-10-10 보조 실험 st_official 실행 전 계획 (결과 전 고정)
- 질문: InternVL3.5 공식 Thinking 평가 설정(VLMEvalKit `InternVL3_5-1B-Thinking`: R1, T 0.6, top_p 0.95, 최대 생성 65,536)으로 ST를 돌리면, b_main ST(생각 1,024 상한 + 답 처리 규칙)와 정확도가 얼마나 다른가. 그리고 그 값이 공식 59.3과 VisRef 46.1 중 어디에 가까운가. 동기: 4,096 상한에서도 ST 생각의 절반 이상이 안 끝났고(main_s0 29/60, full_s1 33/58), 그중 약 절반은 반복 루프였다(마지막 4,000자 zlib 압축률 <0.2: 16/29, 18/33). 상한이 크면 루프가 정답률을 깎아 논문의 낮은 ST(46.1)를 설명할 가능성이 있다.
- 설정(run.py 인자): `--methods st --think-cap 65536 --total-think-cap 65536 --answer-cap 4096 --forced-prompt=`. 나머지(R1, T 0.6, top-k 50, top-p 0.95, seed 42, max_context 40,960, fp16, 코드 d0095f79e691)는 b_main과 같다.
  - 생각 상한은 공식 65,536을 넣되 실제로는 max_context 40,960에서 프롬프트·답 여유를 뺀 값(약 33,000)에서 잘린다. 40,960이 모델의 최대 위치 길이인지는 미확인(SPEC_LEDGER 15행).
  - 공식은 답을 따로 상한 두지 않으므로 답 상한을 4,096으로 늘림. 잘린 생각 뒤 재촉 문구 없음(규칙 ① 끔; 공식에는 강제 답이 없다). af(규칙 ②)는 코드에서 끌 수 없어 남지만 답 4,096에서만 작동.
  - 공식과 남는 차이: `USE_COT` 지시문 없음(공식 설정 여부 미확인), top-k 50(공식 lmdeploy 기본값 미확인), HF `query` 프롬프트, 자체 엔진 fp16, 추출 판정 모델.
- 표본: b_main과 같은 seed 0 셔플 순서 전체 1,000문항을 넣고, 세 방법 집계와 같은 complete prefix(셔플 앞부분 중 끝난 곳까지)만 분석. 같은 pid는 같은 시드라 생각의 처음 1,024토큰은 b_main ST와 같다(CRN, fp16 배치 잡음 제외).
- 세션: 3시간(timeout 10,800s; 투입 중단 8,100s, 덤프 10,200s). 반복 루프 1개가 약 30분 걸리므로 2시간보다 길게 잡음. 계속 여부는 끝난 문항 수와 사용 시간으로만 정한다(정확도를 보고 정하지 않음). 목표 n = 150, 상한 GPU 9시간(3세션) 중 먼저 닿는 쪽.
- 비용 추정(불확실, ±2배): 문항당 생성 약 1.2만 토큰(루프 약 28% × 3.3만 + 긴 생각 약 22% × 1만 + 나머지), 긴 컨텍스트에서 GPU당 초당 약 20–28토큰(b_main 로그, 행 1–2개) → 2 GPU 합 약 50 토큰/초 → 문항당 약 4분, 3시간 세션당 약 40–50문항. 첫 세션 실측으로 다시 추정한다.
- 분석(모두 v3 채점, 같은 pid 짝, b_main ST와 비교):
  - 주 추정: 정확도 차이(st_official − b_main ST), 부트스트랩 95% CI, McNemar 정확 p. b_main ST 첫 단계 종료 방식(close/cap/eos)별로 나눠 보고.
  - 기술: st_official 정확도와 Wilson 95% CI를 공식 59.3·VisRef 46.1과 나란히 제시(검정 없음). 해석 규칙: CI가 46.1을 포함하고 59.3을 포함하지 않으면 "큰 생각 예산이 논문 ST의 낮은 값을 설명할 수 있음", 반대면 "우리 b_main ST가 공식 설정 baseline을 대표함", 둘 다 포함하면 판정 보류.
  - 점검: b_main ST 첫 단계가 자연 종료한 문항은 두 실행의 정오가 90% 이상 같아야 한다(CRN). 아니면 비교 무효로 보고.
  - 함께 보고: 생각 길이 분포, 컨텍스트 상한에 닿은 비율, 그 행들의 정확도, 읽기 (b) af 제거와 (d) 컨텍스트 상한 행 오답 처리(하한; 공식은 상한 소진 시 답 구간이 없다).

## 2026-10-10 diag_st_plain 결과와 판정 (d0095f79e691, 약 1,400s, error 0, 150/150 완료)
- 분석: `repro/diag/analyze_st_plain.py` → `repro/diag/st_plain_analysis.json`(결과 전 작성한 스크립트 그대로).
- (a) 그대로(판정 읽기): 재촉 없음 47.3% vs 본 실험 강제 답 49.3% (같은 150문항), 차이 −2.0%p (재촉 없음만 정답 15 : 강제 답만 정답 18, McNemar p 0.73). 전체 ST 추정치 재촉 없음 56.4 vs 강제 답 57.6(같은 부분집합 기준).
- 판정: 기준(10%p 이상 하락, 추정치 46 근처) 미충족 → "잘린 생각 뒤 `Final Answer:` 재촉 문구가 논문 ST 46.1과의 차이를 만든다"는 설명은 기각.
- (b) af 제거(두 규칙 모두 없음의 근사): 40.0 vs 46.0, −6.0%p (20:29, p 0.25), 추정치 51.8 vs 55.5. (c) af 행 오답(하한, 판정에 안 씀): 26.0 vs 34.0, −8.0%p (9:21, p 0.043), 추정치 42.5 vs 47.5.
- 함께 볼 것: diag 첫 단계 종료가 본 실험과 다른 문항 15개(본 실험은 상한 절단, diag는 1,024 안에서 자연 종료; fp16 배치 잡음으로 생각이 갈라짐). 상한 절단이 유지된 135문항은 정답 67 vs 68로 사실상 같다. 재촉 없이 답하게 하면 답 길이 중앙값이 283 → 741토큰으로 늘고(답 구간에서 생각을 이어감), af 행이 50 → 64로 늘어 그중 32행이 af로 구제된다. 즉 규칙 ①을 끄면 규칙 ②가 대신 답을 끌어내는 경향 — (b)·(c)의 하락은 대부분 이 경로.
- 남는 설명 후보: 논문 ST 46.1과 우리 ~59의 차이는 답 재촉 문구로는 설명되지 않는다. 생각 예산(상한 크기와 반복 루프) 가설은 st_official로 확인한다.

## 2026-10-10 논문의 토큰 예산 관련 서술 재확인 (VisRef에 토큰 상한·누적 사용량 정보가 있나)
- 상한(max tokens) 값은 본문·부록 어디에도 없다(SPEC_LEDGER 11·12행 검색 기록과 같음).
- 있는 것은 평균 사용량과 예산 축이다:
  - "∼1,772 visual tokens vs. ∼615 text tokens" — InternVL3.5-8B, MathVista, 생각 단계당 생성 텍스트 토큰 평균 (`VisRef_plain.txt` L383-385).
  - Fig. 3 가로축 = 누적 thinking token 예산 B(병렬 사고, Σ|τ(i)| ≤ B, L698-702). ST ★ 위치는 8B에서 약 400–1,000토큰(그림에서 읽음, SPEC_LEDGER Fig 3 ST 위치(A-3)). "budget of 14K thinking tokens"는 MMStar 8B 병렬 사고 예산 (L717).
  - App. I "same decoding setup and token budget" — 값 없음 (SPEC_LEDGER 12행).
  - 1B의 토큰 사용량은 논문에 없음.
- 해석: 8B ST가 평균 약 600토큰에서 스스로 끝난다는 것은 상한이 작았다는 증거도, 컸다는 증거도 아니다(8B는 대부분 그 전에 끝남). 1B는 우리 기록상 1,024에서 63%, 4,096에서도 약 절반이 안 끝나므로 상한 값이 결과를 좌우한다. 논문이 이 값을 정하지 않았으므로 원칙 1(A–C 없음 → D)에 따라 공식 65,536이 기본값이고, st_official은 그 기본값에서의 ST를 측정하는 실험으로 타당하다.

## 2026-10-10 작성자 결정: st_official 보류, 절대값 차이는 추정으로 기술
- 결정: st_official_s1은 push하지 않고 보류(빌드·사전등록은 결과 전 상태로 유효). 이유: 본 실험(b_main)이 우선이고, b_main에 문제가 생겨 재실행하면 1,000문항에 약 20시간이 필요하다. b_main 완주 뒤에 재검토한다.
- 절대값 차이는 검증하지 않은 추정으로 기술한다. 최종 보고서용 문단(초안):
  > 우리 ST(s1–s4 v3 59.4%)는 InternVL3.5-1B 공식 단일 응답 MathVista 점수 59.3%(SPEC_LEDGER 공식 1B 점수(X-1))와 비슷하고, VisRef 논문 ST 46.1%보다 약 13pt 높다. 논문 ST는 8B에서도 공식 수치보다 10pt 이상 낮다(68.1 vs 78.4, 공식 8B 대 논문 ST(X-3)). 논문의 평가 설정이 공식과 같은지는 미확인이다. 논문은 생성 토큰 상한, 잘린 생각과 답의 처리, 답 추출 방식을 기술하지 않는다(논문의 최종 답 처리(X-4), 토큰 상한 값(X-8), 본 표 45행). 우리가 확인한 것은 하나다: 잘린 생각 뒤 답 재촉 문구(`Final Answer:`)는 차이를 설명하지 못한다(diag_st_plain, −2.0%p, p 0.73). 확인하지 않은 후보는 (1) 생각 토큰 상한의 크기 — 1B는 긴 생각에서 반복 루프가 잦아(4,096 상한에서 잘린 생각의 약 절반) 상한과 처리 방식에 따라 정답률이 크게 달라질 수 있다, (2) 답 추출·채점 방식(규칙+LLM 보조 vs GPT 추출기), (3) 프롬프트 형식, (4) 하드웨어·정밀도(논문 A10G vs 우리 T4 fp16)다. 따라서 논문과의 절대값 차이는 이러한 평가 설정 차이에서 비롯된 것으로 추정하며, 재현 여부는 같은 설정 안에서의 방법 간 상대 이득(VisRef−ST, VisRef−TSR)으로 판단한다.

## 2026-10-10 불일치 쌍 감사 (s1–s4, GPU 불필요) — 구조적 문제 없음
- 대상: VisRef가 ST 또는 TSR과 정오가 다른 50문항 × 3방법 = 150행(`repro/audit/discordant_v3_s1s4.jsonl`). 판정 기준(채점 S / 실패 유형 F / 버그 징후 B)을 먼저 고정하고 독립 검토가 전문을 읽어 판정, 핵심 주장을 원자료로 대조. 결과 `repro/audit/audit_rows_s1s4.csv`, `audit_summary_s1s4.md`(끝에 검증 메모).
- 채점: 오채점(정답 처리 오류) 0. 오답 처리 오류 2(441 VisRef: 답 뒤에 `</think>`가 붙어 채점 텍스트가 빔; 579 ST: 첫 단계가 EOS로 끝나 답 단계 없음), 애매 2(ST). 반영해도 VisRef:ST 26:16 → 25:15, VisRef:TSR 18:14 → 18:13. 한쪽으로 쏠린 채점 편향 없음.
- 버그 징후: 주입 뒤 깨진 텍스트·엉뚱한 답·프롬프트 반복 없음. 반복 루프 7문항은 정상적인 영어 반복.
- 남는 약점(기록만, 채점 규칙 변경 없음 — 원칙 2): (1) 마지막 `</think>` 뒤가 비면 답을 잃음. VLMEvalKit `SPLIT_THINK`도 같은 결과(빈 예측)라 원칙 1과 일치. 392문항 중 방법당 1행. (2) 첫 단계가 `</think>` 없이 EOS로 끝나면 ST는 답 단계가 없어 빈 답(392문항 중 1행). VLMEvalKit은 `</think>`가 없으면 전체 응답을 예측으로 씀 → 이 1행은 공식 방식과 다름. 최종 보고 민감도 메모로 남김.
- 해석 주의: 승자 답 중 상당수가 상한 강제 답·추측성(독립 검토 판단, 방법 간 비율 비슷). 차이 판정은 사전등록된 전체 검정으로만 한다.

## 2026-10-10 st_official 실행 전 계획 보정 (작성자 결정: 무료 Colab, 100문항) — 결과 전 고정
- 실행 환경: Kaggle 대신 무료 Google Colab, GPU 1장(무료 등급이면 보통 T4, 실제 GPU는 세션마다 env_*.json에 기록), fp16. 노트북은 `repro/colab/build_colab.py`로 같은 소스 묶음(code_version d0095f79e691, 소스 sha256 검증)에서 생성.
- 표본: b_main 셔플 순서의 첫 100문항(고정; 모두 b_main s1–s4에 있음 → 같은 pid 짝 비교 가능). 이 100문항의 b_main ST: 첫 단계 상한 절단 63 / 자연 종료 36 / EOS 1, 정답 57.
- 인자: 2026-10-10 사전등록과 같음(`--methods st --think-cap 65536 --total-think-cap 65536 --answer-cap 4096 --forced-prompt=`), 단일 GPU라 `--shard 0/1`.
- 이어 달리기: 결과를 Colab 로컬에 쓰고 1분마다 Drive로 복사, 다음 세션은 Drive 결과를 가져와 이미 끝난 pid를 건너뜀(run.py 기본 동작). 끊길 때 진행 중이던 문항만 다시 생성. 잘린 마지막 줄은 버림.
- 종료 규칙: 100문항을 모두 끝내면 분석. 무료 GPU를 더 받지 못해 중단하면 셔플 순서상 끝난 앞부분(complete prefix)만 분석하고 n을 그대로 보고(정확도를 보고 중단하지 않음).
- 분석·해석 규칙: 2026-10-10 사전등록 그대로(짝 차이 + 부트스트랩 CI + McNemar, 첫 단계 종료 방식별; Wilson CI를 59.3·46.1과 비교; 자연 종료 문항 정오 일치 ≥90% 점검; 생각 길이·컨텍스트 상한 도달 비율, (b)·(d) 읽기). n=100이면 Wilson CI 반폭 약 ±10%p라 46과 59의 구분은 점 추정이 한쪽 끝에 가까울 때만 가능하다.
- 하드웨어 차이 주의: Colab T4와 Kaggle T4는 같은 GPU지만 torch·CUDA 버전이 다를 수 있다. CRN(같은 시드)으로 생각 앞부분이 같기를 기대하지만 fp16 잡음과 라이브러리 차이로 갈라질 수 있다 — 자연 종료 문항 일치율 점검으로 확인.

## 2026-10-10 st_official_c1 Colab 노트북 준비 완료 (결과 전)
- 빌더 `repro/colab/build_colab.py`(kaggle/build.py의 소스 묶음·리비전 재사용, 노트북 안에서 소스 sha256 검증). 산출물 `repro/colab/runs/st_official_c1/st_official_c1.ipynb`(notebook sha256 310c7ab59c21…, 38KB) + `run_config.json`. ids = b_main 셔플 순서 첫 100(확인), 소스 해시 = b_main_s5(확인), code_version d0095f79e691.
- 노트북 동작: Drive 마운트 → transformers 4.55.0(+timm, einops: InternViT 원격 코드용, 이미 있으면 건너뜀) 설치·모델/데이터 고정 리비전 다운로드 → run.py를 로컬 `/content/out`에 쓰고 1분마다 Drive로 복사, 세션마다 env_*.json 기록. 새 런타임에서는 Drive 결과를 가져와 이어서 실행.
- 모의 실행(가짜 torch·run.py, GPU 없음): 31번째 문항 중간에 끊긴 뒤 잘린 줄 제거 → 새 런타임에서 남은 70문항만 실행 → 최종 100행·중복 0·ids 일치.
- 분석 스크립트 `repro/diag/analyze_st_official.py`(결과 전 작성): 연습 실행(가짜 공식 = b_main ST) n=100, 차이 0, CRN 점검 통과.
- 결과 회수: 작성자가 Drive 폴더의 `results_0.jsonl`(및 env_*, log_*)을 `repro/colab/runs/st_official_c1/output/`로 내려받으면 분석.
- (같은 날 추가) 노트북이 Drive에 `status.json`({state: running|stopped|finished, done, total, updated_utc, session})을 1분마다 갱신하도록 빌더 수정 후 재빌드(실험 인자·문항·소스 해시 그대로). 런타임이 끊기면 state는 running인 채 updated_utc가 멈추므로, 외부 점검은 "updated_utc가 15분 이상 지났는가"로 판단한다. 모의 실행 재확인: 끊김 → stopped 31/100, 재실행 → finished 100/100, 중복 0.

## 2026-10-10 b_main_s5 결과(d0095f79e691)와 s6 빌드
- 건전성: error 0, 111문항×3 = 333행, code_version d0095f79e691, T4×2, torch 2.11.0+cu128·transformers 4.55.0(s1–s4와 같음), .partial 없음, s1–s4와 pid 중복 0. 세션 약 5,710s.
- s5 단독(v3, 111문항): ST 46.8 / TSR 52.3 / VisRef 50.5, ST 첫 단계 상한 절단 68.5%. s1–s4(59.4 / 61.0 / 62.0)보다 크게 낮다(ST Fisher p 0.023).
- 원인 점검(설정 변경 없음): 생성 코드 sha256(run.py, methods, dpp, engine, model, fastkv)이 s2–s5에서 모두 같고(s1은 출력 동일 확인된 79eef648ebd1), 다른 것은 scoring.py뿐이며 집계는 모두 v3로 재채점. 문항 유형 구성(category·task·context·grade·question_type)은 s1–s4와 비슷한데 거의 모든 유형에서 정확도가 함께 낮다. 세 방법이 같은 문항에서 함께 떨어졌고(첫 단계 공유) 방법 간 차이 방향은 유지(TSR−ST +5.5, VisRef−ST +3.7). 세션별 ST: s1 59.5 / s2 57.0 / s3 63.0 / s4 58.3 / s5 46.8 → 실행 이상의 증거 없음, 셔플 구간의 문항 난이도 편차로 본다(5개 구간 중 하나가 이 정도 벗어날 확률은 작지만 불가능하지 않음). 이후 세션도 같은 점검을 계속한다.
- s1–s5 complete prefix 503문항(참고, v3): ST 56.7 / TSR 59.0 / VisRef 59.4. 짝 차이 TSR−ST +2.4(0.0~+5.0), VisRef−ST +2.8(0.0~+5.6), VisRef−TSR +0.4(−2.2~+3.0). 민감도 (b) 54.5 / 56.7 / 57.1, (c) 44.3 / 45.9 / 46.3. CRN: TSR k=1 362문항 중 ST와 정오 일치 354.
- s6 빌드: 코드·인자·문항 순서 s5와 같음, --skip-from s1–s5(1,583행 = 503×3 + s1 visref_raw 74), timeout 7200. 보고서 아티팩트 v8 = s1–s5.

## 2026-10-10 st_official_c1 Colab 실행 시작
- 작성자가 무료 Colab에서 실행 시작(08:55 UTC). env: torch 2.11.0+cu130, transformers 4.55.0, Python 3.13, Tesla T4 1장, code_version d0095f79e691, fast_kv_check ok. (Kaggle과 CUDA 빌드만 cu130 vs cu128)

## 2026-10-10 공개 준비 결정 (작성자)
- 공개: 실험 완료 후 GitHub 공개, 그 전까지 비공개.
- 코드 라이선스: MIT. 결과 JSONL(생성 텍스트 포함)도 올린다(MathVista·InternVL3.5 출력 재배포 조건은 공개 전 확인, 미확인).
- LLM 추출은 Gemini API 스크립트(`repro/gemini_extract.py`, 모델 ID·입출력 기록)로 한다. 요청 파일을 만들어 두고, 실행은 작성자가 터미널에서 한다.
- (같은 날) 저장소 이름 `visref-repro` 확정. 설명안: "Unofficial reproduction of VisRef (arXiv:2603.00207) — InternVL3.5-1B × MathVista testmini".
- LLM 추출 사전 점검 준비: s1–s5 결과를 v3로 재채점한 사본 `repro/llm/precheck_s1s5/rows_v3.jsonl`(원본 불변, 공개 대상 아님) → `llm_extract.py prepare --control 20` → `requests.jsonl` 49건(규칙 실패 29 + 대조군 20, 정답 미포함). 사전 점검 결과는 판정에 쓰지 않고, 최종 추출은 완주 후 전체로 다시 한다.

## 2026-10-10 b_main_s6 결과와 s7 빌드
- s6 건전성: error 0, 126문항×3 = 378행, d0095f79e691, T4×2, torch 2.11.0+cu128, 이전 세션과 pid 중복 0, .partial 없음, 세션 약 6,190s.
- s6 단독(v3): ST 69.8 / TSR 69.0 / VisRef 65.9, 첫 단계 상한 절단 61.9%. s5(46.8)와 반대로 높다 → 세션 간 편차가 크며 s5 저점은 구간 편차라는 해석과 맞음. 세션별 ST: 59.5 / 57.0 / 63.0 / 58.3 / 46.8 / 69.8.
- s1–s6 complete prefix 629문항(참고, v3): ST 59.3 / TSR 61.0 / VisRef 60.7. 짝 차이 TSR−ST +1.7(−0.5~+3.8, 30:19, p 0.15), VisRef−ST +1.4(−1.0~+4.0, 36:27, p 0.31), VisRef−TSR −0.3(−2.5~+1.9, 27:29, p 0.89). 판정은 완주 후 사전등록 검정으로만.
- s7 빌드: 코드·인자·순서 s6와 같음, --skip-from s1–s6(1,961행 = 629×3 + 74), timeout 7200. 보고서 아티팩트 v9 = s1–s6.

## 2026-10-10 LLM 추출 사전 점검 결과 (gemini-3.5-flash-lite, 무료 등급, 작성자 터미널 실행)
- 요청 49건(규칙 실패 29 + 대조군 20) → 응답 49(모두 STOP). merge: fixed 2, broke 0, 대조군 일치 18/20. 정확도 ST 56.7→56.9, TSR 59.0→59.0, VisRef 59.4→59.6.
- 원문 대조 결과 "fixed" 2건은 모두 잘못된 정답 처리: 570 ST는 채점 텍스트가 빈 문자열인데 LLM이 "Assuming … 5"로 답을 지어냄(정답 5와 우연히 일치), 106 VisRef는 텍스트의 답이 "1"인데 LLM이 "0"으로 추출(정답 0). 실제로 바로잡은 행 0.
- 대조군 불일치 2: 964 VisRef(규칙 `\boxed{1}`이 맞고 LLM "7"이 틀림), 833 TSR(규칙이 `1.4 \times 10^{19}`에서 지수 19를 추출 — 규칙 쪽 결함, LLM 1.4; 정답 −4.3이라 둘 다 오답).
- 비추출 응답 7건(빈 텍스트·긴 텍스트에서 이미지·설명 요구 등).
- 판단: 이 모델 그대로의 LLM 추출은 오답을 정답으로 만들 위험이 실재한다. 최종 추출 전에 보완이 필요(작성자 결정 대기): (1) 채점 텍스트가 비면 LLM에 보내지 않고 오답 유지, (2) LLM이 뽑은 답이 응답 원문에 실제로 나타날 때만 인정(객관식은 해당 선택지 글자 또는 선택지 텍스트), (3) 더 강한 무료 모델(gemini-3.8-flash) 비교. 사전 점검 자료는 판정에 쓰지 않는다.

## 2026-10-10 LLM 추출 보완 반영 (작성자 승인, 최종 추출 전 — 판정 결과를 보고 바꾸는 것이 아님)
- `repro/llm_extract.py`에 두 단계 추가(이전 판 `repro/llm/llm_extract.before_guard.py`):
  1) 채점 텍스트가 빈 행은 LLM에 보내지 않고 규칙 결과(오답) 유지(prepare에서 제외, merge에서도 무시).
  2) LLM 추출 답이 채점 텍스트에 실제로 있을 때만 인정: 객관식은 선택지 텍스트 또는 글자 형식("(B)", "B)", "B.", "**B**", "\boxed{B}", "answer is B"), 숫자형은 같은 수(쉼표 제거, 오차 1e-9), 그 외는 같은 문자열. 근거 없는 추출은 기록만 하고 규칙 결과 유지. 행마다 llm_grounded·llm_used 기록.
- 단위 확인 9/9 통과. 사전 점검 응답(3.5-flash-lite) 재병합: fallback 24, 빈 텍스트 건너뜀 5, 근거 없음 거부 2, fixed 0, broke 0, 대조군 일치 18/20·근거 있음 20/20 → 정확도 규칙과 동일(56.7 / 59.0 / 59.4).
- 한계: 보완 2는 지어낸 답은 막지만, 텍스트에 있는 다른 숫자를 잘못 고르는 경우(예: 964 VisRef, 대조군에서 "7 cm mark"의 7)는 막지 못한다. 최종 보고에 LLM 추출로 바뀐 행 목록을 함께 싣는다.
- 다음: 같은 요청 49건을 gemini-3.8-flash(무료 등급 표기)로 돌려 모델 비교 후 최종 추출 모델을 정한다.

## 2026-10-10 b_main_s7 결과와 s8 빌드
- s7 건전성: error 0, 130문항×3 = 390행, d0095f79e691, T4×2, torch 2.11.0+cu128, pid 중복 0, .partial 없음, 세션 약 5,710s.
- s7 단독(v3): ST 51.5 / TSR 56.2 / VisRef 53.8, 첫 단계 상한 절단 69.2%. 세션별 ST: 59.5 / 57.0 / 63.0 / 58.3 / 46.8 / 69.8 / 51.5.
- s8 빌드: 코드·인자·순서 s7과 같음, --skip-from s1–s7(2,351행 = 759×3 + 74), timeout 7200. 남은 241문항 → s8·s9 예상.

## 2026-10-10 LLM 추출 모델 결정 (작성자): gemini-3.5-flash-lite 단독 + 보완 1·2 — 최종 추출 전 고정
- 근거: MathVista 공식 파이프라인도 LLM 추출기의 오류를 감수한다("a preliminary study of 200 examples shows that GPT-4 can extract the answer text with more than 99.5% accuracy", `references/papers/ref_fulltext/MathVista.txt` L466-469; 공식은 GPT-4, VLMEvalKit 기본 판정 모델은 gpt-4o-mini — `vlmeval/dataset/image_vqa.py` L600). 3.8-flash는 최종 추출에 쓰지 않았고, 사전 점검 16건에서는 보완 적용 후 결과가 flash-lite와 같았다(근거 있는 정답 9/13 vs 9/13, 근거 없음 0 vs 0).
- 최종 절차: 완주 후 전체 결과를 v3로 재채점 → `llm_extract.py prepare --control 20`(빈 텍스트 제외) → 작성자 터미널에서 `gemini_extract.py --model gemini-3.5-flash-lite --sleep 6` → `merge`(원문 대조, 근거 없으면 규칙 결과 유지). 서로 맞춰 확인하지는 않는다. 보고서에 LLM 추출로 정오가 바뀐 행 전부와 대조군 일치율을 싣는다.

## 2026-10-10 공개용 폴더 생성 (github_export/visref-repro)
- `scripts/make_github_export.py`(재실행 가능)로 코드·실행 기록(run_config, env, log, results)·문서(EF, SPEC_LEDGER, 감사·검토)를 복사. 제외: 논문 PDF·전문, 제3자 코드 사본(URL+sha256을 `references/SOURCES.json`에 기록, `scripts/fetch_sources.py`로 내려받아 검증), 번들 노트북(재생성 가능). 약 22MB.
- README(영문), LICENSE(MIT, 저작권자 이름 자리표시), requirements.txt, .gitignore. 라이선스 근거: InternVL3.5-1B Apache-2.0(모델 카드 L2, L816-818), MathVista 저장소 CC BY-SA 4.0(LICENSE 파일) → 결과 파일은 CC BY-SA 4.0으로 표기.
- 검증: 임시 사본에서 README 1단계(오프라인 재채점·보고서)를 그대로 실행 → s1–s7 759문항 수치가 원본과 동일.

## 2026-10-10 관찰: VisRef 효과가 TSR을 넘지 못함 (설정 변경 없음 — 사후 관찰)
- s1–s7 759문항: TSR−ST +2.2(논문 +2.4와 비슷) → 다단계 이어가기(TSR)는 재현되는 쪽. VisRef−TSR −0.7(CI −2.8~+1.6)로 논문 +3.5가 CI 밖.
- 이어가기가 일어난 문항(어느 방법이든 k>1, 230문항): ST 33.5 / TSR 41.7 / VisRef 38.7. 둘 다 k=1인 529문항은 68.6 / 68.2 / 68.6(사실상 동일).
- 주입량: 주입 수/m 중앙값 0.21, 주입 수/|V| 중앙값 0.062, m에 도달한 단계 16%. 논문 Fig. 4(b) 최저 조건(m=20%)보다 훨씬 적음.
- 답 구간에서 생각을 다시 여는 행(답 안 </think>): VisRef 357/759(47%), 정확도 58.3 vs 없는 행 60.7.
- 해석(가설, 검정 전): 코드 버그 징후는 감사·CRN 점검에서 나오지 않았고, 유력 원인은 (1) 입력 임베딩 Z에서 계수 제한으로 m의 약 1/5만 주입되는 구현 선택(DPP 선택 수(J-7)/Z 표현(J-8)), (2) 최종 답 직전 주입이 답 구간을 흔드는 현상이다. b_main은 사전등록대로 완주하고, 원인 분리 실험은 완주 후 사전등록해 진행한다(작성자 결정 대기).

## 2026-10-10 주입 경로 독립 검토 (GPU 없음) — 버그 없음
- 보고서 `repro/audit/injection_review_20261010.md`(+ DPP 수치 점검 스크립트·출력). 결론: 주입 경로에 BUG 없음. 주입 텐서 = 모델이 원본 이미지에서 보는 projector 출력 그대로(같은 객체, fp16, 재스케일 없음), z_k 경계 깨끗, 순서 z_k [\n] V_k 다음 </think>+probe 또는 Wait, 위치·마스크·KV 일관(호출마다 전체 재-prefill), greedy DPP가 Chen 코드·naive log-det greedy와 선택 동일. 표본 4행에서 final_context_len이 토큰 단위로 재현됨. 확인: 표본 행 계산은 보고서 수치로만 확인(미재계산), R1 진단은 아래처럼 전체 자료로 재계산.
- RISK R1(가장 중요): 시각 토큰 묶음 바로 뒤에 </think>가 붙는 구조(논문 순서 그대로)가 답 안 </think> 재발생의 원인일 가능성. 진단(s1–s7, 759문항): 마지막 단계가 스스로 닫힌(close) 경우 재발생률 ST 6.5% / TSR 7.3% / VisRef 45.1%, 상한 절단(cap) 경우 17.0 / 13.8 / 50.0% → 상한이 아니라 V 묶음 뒤 </think> 구조가 원인으로 확인. 그러나 k=1·1단계 자연 종료 244문항에서 정답 TSR 176 vs VisRef 175 → 재발생 자체의 정확도 영향은 작다. VisRef가 TSR보다 낮은 구간은 이어가기 문항(k>1, 230문항: TSR 41.7 vs VisRef 38.7).
- R2(행에 cfg 미기록 — run_config.json에는 있음), R3(VisRef만 m+2 여유 확보 — 현재 컨텍스트에서 영향 없음), R4(상대 ε — 현재 영향 없음)는 기록만.

## 2026-10-10 원인 분리 실험 실행 전 계획 (작성자 결정, b_main 완주 후 실행) — 결과 전 고정
- 목적: VisRef−TSR이 논문(+3.5)보다 낮은 원인 분리. b_main 설정은 바꾸지 않는다. 새 방법을 추가만 하고, 기존 세 방법의 출력이 바뀌지 않음을 CPU 동일성 검사(2026-10-07 방식)로 확인한 뒤 빌드한다.
- 변형 1 `visref_fill`(m 채우기, 먼저 실행): greedy가 계수에서 멈춘 뒤에도 커널에 작은 εI(1e-6 × 평균 대각, 2026-10-08 메모의 "식은 그대로, 수치 안정화만")를 더해 정확히 m개까지 선택해 주입. 그 외는 visref_raw_rank와 같음. 논문 문구 "|Ṽk| = m"에 맞춘 해석.
- 변형 2 `visref_nolast`(마지막 단계 주입 생략): 각 단계의 V_k는 다음 단계로 이어갈 때만 주입하고, 정지 probe와 최종 답은 V_k 없이(TSR과 같은 문맥 구조) 만든다. 논문 Alg. 1과 다른 진단용.
- 표본: b_main 셔플 순서 첫 392문항(s1–s4와 같은 pid, 같은 시드 → b_main ST·TSR·VisRef와 짝). 인자 `--methods visref_fill`(이후 `visref_nolast`) `--think-cap 1024 --total-think-cap 10240`, 나머지 b_main과 같음. Kaggle 2시간 세션, 392문항 끝날 때까지.
- 분석(탐색적, 변형별): 짝 차이(변형 − b_main VisRef, 변형 − TSR), 부트스트랩 95% CI, McNemar 정확 p(두 변형에 Holm 보정 값도 함께), 이어가기 문항(k>1)과 k=1 문항 층별, 주입 수/m, 답 안 </think> 재발생률, k 분포. 채점 v3 + 최종 LLM 추출 규칙 동일.
- 해석 규칙: visref_fill − TSR의 CI 하한이 0보다 크면 "주입량 부족이 VisRef 효과를 줄였다"를 지지. visref_nolast가 재발생률을 낮추고 VisRef보다 정확도가 높으면(CI 하한 > 0) "답 직전 주입이 답을 흔든다"를 지지. 둘 다 아니면 "이 구현·모델(1B)에서는 시각 재주입 효과가 재현되지 않음"으로 보고.
- 보류(사전등록하지 않음): 검토자 R1 제안인 <img>…</img>로 감싸 주입 — 2026-10-07 smoke1에서 감싸기가 두 번째 </think>와 질문 환각을 일으킨 기록(README:L25)이 있어 우선순위 낮음.

## 2026-10-10 변형 실험 설계 독립 검토 — 결과 전
- 보고서 `repro/audit/design_review_variants_20261010.md`(입력 요약 `design_review_variants_brief.md`). 논문 근거는 `VisRef_plain.txt` 줄 번호로 인용.
- 판정 요지: 논문은 최종 답 y가 마지막 V_k를 포함한 궤적에서 생성된다고 명시(Alg. 1 L16-17, §4.1.2)하고 H_k도 V_k 추가 후 계산 → visref_raw_rank는 충실, `visref_noansinj`는 의도적으로 벗어남(진단용 ablation). |V_k| = m은 계수 부족 경우를 논문이 정하지 않은 미지정 → `visref_fill`은 "미지정 사례의 충실한 재해석"이지만 계수 이후 선택은 다양성 정보가 없고 관련성 비슷한 순서.
- 검토자 주장 원자료 확인(s1–s7 759문항): VisRef가 k=1에서 멈춘 543문항에서 VisRef 369 vs ST 369 정답(8:8) → 답 직전 주입 효과는 k=1에서 0. noansinj가 새로 주는 정보는 k>1 문항뿐(392 중 약 110), 그 구간 최소 검출 효과 약 8–10%p로 검출력 낮음.
- 새 관찰: VisRef는 TSR보다 이어가기가 잦다(k≥2 216 vs 199, k=10 64 vs 53; 1단계 probe H 중앙값 0.007 vs 0.003) → V 묶음이 정지 판정 엔트로피를 올릴 가능성.
- 검출력(검토자 계산, 불일치율 8.8% 가정): n=392 SE 1.49%p, +3.5%p 검출력 0.65, 80% 최소 검출 효과 4.2%p; n=1,000이면 2.6%p.
- 검토자 권고: fill 먼저(보완: 앞부분 선택이 rank 버전과 같음 assert, ridge 계산 fp32/64, n_inject/m·stop_reason·재발생·af 기록, 꼬리 선택과 관련성 상관 측정); noansinj는 저장 궤적 재생(replay) 방식의 k>1 진단으로만, 아니면 제외; 결정에 영향 없는 이중 probe 기록(V_k 있는/없는 H_k) 추가; 은닉 상태 Z는 오프라인 계수 점검 후 조건부.
- 메모: 은닉 상태 Z의 오프라인 점검은 2026-10-08 hidden_rank_check로 이미 했고(의미 있는 선택 약 50–110개, 그 뒤는 사실상 관련성 채움) → fill과 실질적으로 같은 개입이라 별도 실행 불필요. 사전등록 수정은 작성자 확인 후 기록.

## 2026-10-10 원인 분리 실험 실행 전 계획 수정 (작성자 결정, 결과 전) — 앞의 "원인 분리 실험 실행 전 계획"을 대체
- 변형 2(`visref_nolast`, 그 수정안 `visref_noansinj`)는 GPU 실행에서 제외. 이유: 논문 Alg. 1에서 벗어난 진단용이며, k=1 문항에서는 구조상 ST 답과 같아(실측 369 vs 369) 정보가 k>1 문항(약 30%)에만 있고 검출력이 낮다(설계 검토 보고서).
- 대신 `visref_fill` 실행에 "이중 정지 판정 기록"을 넣는다: 매 단계 정지 probe를 V_k 있는 문맥(실제 판정에 사용)과 V_k 없는 문맥(`… z_k </think>` + probe prompt, 별도 난수열)에서 각각 계산해 H_k, H_k^noV를 기록. 판정은 H_k만 사용 → 결과에 영향 없음. 분석: H_k − H_k^noV 분포, 판정이 뒤집혔을 단계 비율.
- `visref_fill` 표본: 1,000문항 전체(b_main 전 문항과 짝; 80% 최소 검출 효과 약 2.6%p, n=392면 4.2%p). 한 번의 Kaggle 세션으로 실행: timeout 41,400s(11.5h, Kaggle 세션 최대 12시간 — 2차 자료 확인, 공식 문서 미확인), 투입 중단 38,700s, 덤프 40,800s. 시간 안에 못 끝나면 남은 문항만 같은 코드로 짧은 세션을 이어 붙인다(--skip-from, 문항 순서·시드 동일).
- fill 구현 보완(설계 검토 권고): 계수 전 선택이 visref_raw_rank와 같음을 assert, ridge 계산 float64, n_inject/m·stop_reason·k·답 안 </think>·af 기록, 계수 이후 선택 순서와 관련성 r_i²의 순위 상관 기록.
- 분석·해석 규칙: 이전 등록과 같음(visref_fill − TSR, visref_fill − visref_raw_rank 짝 차이, 부트스트랩 CI, McNemar; k=1/k>1 층별). 무효 결과는 "주입량 효과 2.6%p 이상은 검출되지 않음"으로 표현.

## 2026-10-10 재현 판정 기준 (작성자 결정, 최종 결과 전 — s1–s7 759문항 중간값은 이미 관찰함)
- 대상: b_main 최종(1,000문항) 세 비교 VisRef−ST, VisRef−TSR(주 비교), TSR−ST(보조). 읽기 (a) 채점 v3 + LLM 추출(flash-lite, 보완 1·2). 효과 크기는 같은 문항 짝 차이의 부트스트랩 95% CI(2,000회, seed 0), 논문 값은 Table 6 차이(+5.9 / +3.5 / +2.4).
- 판정(비교별):
  - "재현됨": CI가 0을 포함하지 않고(논문과 같은 방향) 논문 값을 포함.
  - "효과 크기 재현 안 됨": CI가 논문 값을 포함하지 않음. CI가 0도 포함하지 않으면 "방향은 같으나 크기 다름"을 덧붙임.
  - "판정 불가": CI가 0과 논문 값을 모두 포함.
- 유의성은 기존 사전등록(McNemar 정확 + Holm)을 함께 보고하되, 판정 문구는 위 CI 규칙으로만 정한다. 범위 문구: "공식 설정에 가까운 InternVL3.5-1B 운영점(ST ≈ 58%)에서".
- 투명성 메모: 중간값(759)에 이 규칙을 적용하면 TSR−ST "재현됨", VisRef−ST·VisRef−TSR "효과 크기 재현 안 됨"이 된다. 규칙은 이 값과 무관하게 일반 형태로 정했고, 최종 판정은 1,000문항으로만 한다.

## 2026-10-10 관찰: 이어가기는 거의 전부 상한 절단 뒤에만 일어남 (사후, s1–s7)
- TSR 단계 전이: 상한 절단 → 이어감 833 / 멈춤 477, 자연 종료 → 이어감 6 / 멈춤 262. VisRef: 921 / 468, 24 / 273.
- 해석: 우리 구현의 TSR·VisRef 이어가기는 대부분 "잘린 생각의 예산 연장"이며, 자연 종료 뒤 엔트로피가 높아 반성하는 경우(논문이 그리는 메커니즘)는 드물다(TSR 2%). 정지 통계 H(정지 통계 H 추정(J-9), 등급 F)가 자연 종료 뒤에는 거의 항상 δ 미만. TSR−ST 이득의 출처(반성 vs 예산)는 st_official(상한을 푼 ST)로 일부 판별 가능.

## 2026-10-10 b_main_s8 결과와 s9 빌드
- s8 건전성: error 0, 109문항×3 = 327행, d0095f79e691, T4×2, torch 2.11.0+cu128, s1–s7과 (pid, method) 겹침 0, .partial 없음, 세션 약 5,250s(마지막 행 elapsed 5,144s, 대기열 79 남음).
- s8 단독(v3): ST 58.7 / TSR 58.7 / VisRef 62.4, ST 첫 단계 상한 절단 65.1%. 세션별 ST: 59.5 / 57.0 / 63.0 / 58.3 / 46.8 / 69.8 / 51.5 / 58.7.
- s1–s8 complete prefix 868문항(참고, v3, 재채점으로 정오 바뀐 행 4): ST 58.1 / TSR 60.0 / VisRef 59.9. 짝 차이 TSR−ST +2.0(+0.2~+3.8, 43:26, p 0.05), VisRef−ST +1.8(−0.2~+3.8, 51:35, p 0.11), VisRef−TSR −0.1(−2.1~+1.8, 38:39, p 1.00). 판정은 완주 후 사전등록 기준으로만.
- s9 빌드: 코드·인자·순서 s8과 같음(source sha256 동일), --skip-from s1–s8(2,678행 = 868×3 + 74), 남은 132문항. 세션 제한만 7,200 → 12,600s(time_limit 9,900, deadline 12,000) — 한 세션에 끝내기 위한 실행 방식 변경이며 생성 설정 변경 아님. 보고서 아티팩트 v11 = s1–s8.

## 2026-10-10 검토: "4,096 상한에서도 약 절반이 루프·절단" 주장 재확인 (원자료 재계산)
- 자료: `repro/kaggle/runs/main_s0`, `full_s1`(단계 상한 4,096). 스크립트 `repro/diag/check_4096_cap_loops.py` → `repro/diag/check_4096_cap_loops.json`.
- ST 첫 단계 4,096 상한 절단: main_s0 29/60(48%), full_s1 33/58(57%). 2026-10-07 "smoke3 결과 → probe를 `\boxed` 형식으로 교체" 항목(57%)·2026-10-08 "해석 메모(방향 일치 / 토큰 상한·thinking / m=30%)" 항목·2026-10-10 "보조 실험 st_official 실행 전 계획" 항목와 일치.
- 그중 반복 루프(생각 마지막 4,000자의 zlib 압축률 < 0.2, UTF-8 바이트 기준): 16/29, 18/33 → 2026-10-10 "보조 실험 st_official 실행 전 계획" 항목와 일치. 3,000자 창이면 14/29, 16/33. ST 전체 중 루프는 약 27–31%. 나머지 절단은 답 사이를 오가는 진동(예: pid 390 "answer A … maybe B")이나 아직 풀이 중인 텍스트(예: pid 62).
- 정정: 2026-10-07 "smoke3 결과 → probe를 `\boxed` 형식으로 교체" 항목 "ST의 57%가 4096 상한(반복 루프)"은 절단 전부를 루프로 적은 과장 — 루프는 절단의 약 절반. 정확한 표현은 "ST 첫 단계의 약 절반(48–57%)이 잘렸고, 잘린 것의 약 절반이 반복 루프"이다. 전 방법·전 단계 기준 상한 도달은 main_s0 100/197(51%), full_s1 155/427(36%).
- 한계: 두 표본은 같은 셔플 순서의 앞부분이라 pid 52개 공유(합집합 66) → 독립 반복이 아니다. zlib 기준은 결과를 본 뒤 정한 휴리스틱(full_s1에서 자연 종료한 긴 생각 2건도 < 0.2). 정답: 절단 행 11/29, 15/33 vs 비절단 22/31, 20/25.

## 2026-10-10 st_official_c1 중간 기록 (Colab 세션이 끝나 멈춤 — 판정 아님)
- 상태: done 33/100, 세션 20261010-085537, 마지막 갱신 14:43:47Z(약 5.8h). 로그·상태 사본 `repro/colab/runs/st_official_c1/interim_20261010/`(log sha256 a148b169…). results_0.jsonl은 Drive에만 있음(미회수).
- 완료 33문항은 셔플 순서의 앞부분이 아님(complete prefix 16, 첫 미완료 pid 915) → 사전등록 분석 대상 아님. 끝난 문항은 짧은 문항과 일부 극단적으로 긴 문항으로 편향.
- 사전등록 점검 항목(CRN): b_main ST 첫 단계가 자연 종료한 완료 문항 16개에서 정오 16/16 일치(기준 ≥ 90%) → Colab 환경이 Kaggle과 같은 생성을 재현한다는 증거.
- 생각 길이(완료 33): ≤1,024 14 / 1,025–4,096 9 / 4,097–16k 5 / >16k 5(33,121–35,542, max_context 근처). 정답(런타임 채점): 10/14, 6/9, 3/5, 3/5. 같은 pid b_main ST: 10, 4, 5, 2 → 합 22 vs 21(불일치 3:2). 표본이 편향돼 효과 추정에 쓰지 않는다.
- 처리 시간: 컨텍스트 상한 근처 문항은 문항당 벽시계 16,897–20,331s(4.7–5.6h). 진행 중 문항은 끊기면 처음부터 다시 생성. 최대 메모리 12.56GB(T4 15GB).
- 남은 67문항은 b_main ST 첫 단계 상한 절단 46(69%) vs 완료 33 중 17 → 남은 쪽이 긴 문항 위주. 멈춘 시점 진행 중 16개 중 9개가 컨텍스트 19k 이상. 완주까지 여러 세션(여러 날)이 필요하다는 추정(불확실).
- 진행: 사전등록 종료 규칙 유지(더 받지 못하면 complete prefix만 분석). GPU를 다시 받은 뒤 이어 실행. 투명성: 중간 정오를 이번에 봄 — 계속 여부는 정확도로 정하지 않는다.

## 2026-10-11 검토: 단계 상한 4,096 재실행의 가치 (기존 자료로 판단, 새 실행 없음)
- main_s0(단계 4,096 / 합계 8,192, 코드 8aaa689c94e9 — b_main과 다른 이전 판, 강제 답 없음, 60문항) 단계 전이: TSR 자연 종료 → 이어감 0 / 멈춤 32, 상한 절단 → 이어감 10 / 멈춤 27. visref_raw 자연 종료 → 0 / 31, 절단 → 7 / 27, EOS → 1 / 1.
- b_main(1,024)과 같은 양상: 자연 종료 뒤에는 정지 통계 H가 δ 미만이라 거의 항상 멈춘다. 상한을 4,096으로 올리면 자연 종료 비율만 늘고(ST 첫 단계 절단 63–69% → 48–57%) 이어가기는 더 줄어, TSR·VisRef가 ST에 가까워질 뿐 "완결 단계 뒤 반성"은 여전히 생기지 않을 것으로 본다(추정, 60문항·이전 코드).
- 결론(제안): 4,096 재실행은 논문 메커니즘 검증에 정보가 적고 비용이 크다(main_s0 약 60문항/시간, 1,000문항 약 17h+ 추정). 메커니즘을 가르는 변수는 단계 상한이 아니라 정지 통계 H의 추정 방식(정지 통계 H 추정(J-9), 등급 F)이다.

## 2026-10-11 검토: 정지 통계 H의 추정 방식 — 논문에 더 가까운 후보 (구현·실행 없음, 작성자 결정 대기)
- 원문: 정의는 A — H_k = −E_{y∼πθ(·|x,τ1:k)} log πθ(y|x,τ1:k)(`VisRef_plain.txt` L604-606, Alg. 1 L524). Fig. 2는 단계마다 "Answer: 600"/"Answer: 1000"과 답 분포 막대(`VisRef_plain.txt` L359, L363). 추정법은 없음(SPEC_LEDGER 22행).
- 인용 문헌 [19] Ghosal(`VisRef_plain.txt` L998; VisRef 제1저자와 같은 사람): "sample multiple answers", "entropy of the distribution over final answers"(대장 Ghosal 정지 엔트로피(X-11)). 즉 여러 답을 뽑아 최종 답 값의 분포 엔트로피를 잰다 — 등급 C(+Fig. 2 B). 우리 방식(단일 표본 `\boxed` 토큰 엔트로피 합, T=1)은 F.
- 논문 쪽 정황: Table 5(8B, H100) ST 7.1s / TSR 7.7s / VisRef 8.2s(`VisRef_layout.txt` L892-895) → TSR이 ST보다 8%만 길어 논문에서도 대부분 k=1에서 멈춘 것으로 읽힘(Fig. 3과는 논문 수치 간 차이, 원인 미확인; 지연 시간 대 이득(C-12)). Ghosal의 표준 생각 직후 평균 엔트로피 0.23 / 0.02(Ghosal 정지 엔트로피(X-11))도 δ=0.25 아래.
- 예상(추정): Ghosal식(N개 답, 답 값 군집의 엔트로피)으로 바꿔도 자연 종료 뒤에는 거의 같은 답이 나와 멈춘다(우리 자연 종료 probe H 중앙값 0.003–0.007, 2026-10-10 "변형 실험 설계 독립 검토 — 결과 전" 항목). N<20이면 δ=0.25 미만 ⇔ N개 만장일치라, 잘린 단계 뒤에는 오히려 이어가기가 늘 수 있다. 따라서 "완결 단계 뒤 반성"을 만들어 내지는 못할 가능성이 크다.
- 더 중요한 차이: 논문도 대부분 k=1에서 멈춘다면 VisRef 이득은 k=1(최종 답 직전 주입, V_k 주입 위치(J-6))에서 나와야 하는데, 우리는 k=1 정지 543문항에서 VisRef 369 vs ST 369(2026-10-10 "변형 실험 설계 독립 검토 — 결과 전" 항목). 8B와 1B 차이일 수 있음(미확인).
- 제안: b_main은 그대로(원칙 2). fill 실행에 Ghosal식 H를 그림자 기록(결정에는 쓰지 않음, 별도 난수열)으로 추가해, 정지 결정이 몇 % 바뀌는지 결과를 본 뒤 측정. N·온도 값은 원문에 없어 F로 정해야 함.

## 2026-10-11 정정: Ghosal식 다중 표본 H 그림자 기록 제안 철회
- 다중 표본 추정은 이전에 써 본 뒤 내려놓은 방식이다: 초기 단계에서 단계마다 완결 응답 10개(S10, sequence NLL)를 뽑았고, R25(8B, dev 3문항 TSR) GPU 4,703.9s([초기 기록 "R25 수집·채점 완료 — 자동 마무리 기록" 항목](https://github.com/AhnJG/visref-repro/blob/4776f6ab2607957799bef8db70241cc64313feae/docs/EXPERIMENT_FLOW.md?plain=1#L642-L644)). 이후 작성자 결정으로 자연 응답 엔트로피로 전환하고 S10은 별도 variant로 보존([초기 기록 2026-10-06 "자연 응답 entropy 전환 — 로컬 구현 진행" 항목](https://github.com/AhnJG/visref-repro/blob/4776f6ab2607957799bef8db70241cc64313feae/docs/EXPERIMENT_FLOW.md?plain=1#L647-L650)), "S10 복원 없음"([초기 기록 2026-10-06 "R27 결과 확인" 항목](https://github.com/AhnJG/visref-repro/blob/4776f6ab2607957799bef8db70241cc64313feae/docs/EXPERIMENT_FLOW.md?plain=1#L725-L727)). 전환 이유가 비용이라고 명시된 줄은 찾지 못함(미확인), 비용이 컸던 것은 기록으로 확인.
- 이번 제안(짧은 `\boxed` 답 N개)은 S10보다 생성 토큰은 적지만, 현재 엔진은 generate 호출마다 문맥 전체를 다시 prefill한다(`repro/visref/engine.py` L82-104). N개 표본 = 문맥(최대 약 1만 토큰) prefill N번, KV를 공유하면 행당 KV가 약 0.11MB/토큰(28층·KV 헤드 8·head dim 128·fp16 — 모델 config 기준 추정, 미확인)이라 1만 토큰 × 8행 ≈ 9GB로 T4에 맞지 않는다. 11.5h 안에 1,000문항을 넣어야 하는 fill에 얹기엔 과하다.
- 결정 제안: fill에 넣지 않는다. 예상 효과(자연 종료 뒤 정지 결정이 거의 안 바뀜)도 작으므로 별도 실행도 하지 않고, 최종 보고서 한계("H 추정 방식은 F, 논문·Ghosal은 다중 표본으로 읽힘")로 처리한다.

## 2026-10-11 분석: VisRef k=1 주입 효과가 0인 이유 (b_main s1–s8 868문항, v3, 사후 분석 — 판정 아님)
- 스크립트 `repro/diag/analyze_k1_injection.py` → `repro/diag/analyze_k1_injection.json`. CRN 구조상 k=1에서 VisRef와 ST의 차이는 최종 답 직전(`</think>` 앞)에 넣은 V_1뿐이다(`methods.py` _end_think).
- VisRef k=1 621문항: ST 420 vs VisRef 420, 한쪽만 정답 11:11. 예측(추출 답)이 같은 문항 591/621(95%), 다른 30 중 VisRef 정답 11 / ST 정답 11 / 둘 다 오답 8.
- 잡음 기준선: TSR k=1(이론상 ST와 같은 답, fp16 배치 잡음만) 637문항 중 예측 다름 16(2.5%).
- 첫 단계 종료 방식별(VisRef vs ST 예측 다름 / TSR vs ST 예측 다름):
  - 자연 종료: 1/281 vs 1/282 → 생각이 끝나면 답이 이미 정해져 있고, 주입은 답을 바꾸지 못한다.
  - 상한 절단: 25/324(7.7%) vs 15/337(4.5%) → 주입이 답을 바꾸는 빈도는 잡음보다 조금 많지만 방향은 무작위(VisRef 10 : ST 9).
  - EOS: 4/16 vs 0/18(소수).
- 주입의 부수 효과: 최종 답에서 `</think>` 재출력 310/621(50%) vs ST 61(10%), 자연 종료에서도 136/281 vs 18 → 모델이 주입된 이미지를 보고 답 구간에서 생각을 다시 하지만 결론은 같다. 답 길이 중앙값 123 vs 82토큰, 답 상한 도달 129 vs 114.
- 주입 양(k=1): 주입 토큰 중앙값 121(|V| 중앙값 1,792), 주입/m 중앙값 0.23.
- 해석: k=1에서 효과가 0인 것은 (1) 자연 종료 문항은 답이 생각 안에서 확정돼 있고, (2) 절단 문항에서는 주입이 답을 흔들 뿐 정답 쪽으로 밀지 않기 때문이다. 논문 Table 5가 시사하듯 8B가 대부분 k=1에서 멈췄다면 논문 이득은 k=1 주입에서 나와야 하는데, 우리 1B·우리 주입 방식(맨 임베딩, `<img>` 태그 없음, DPP가 rank에서 멈춤)에서는 그런 효과가 보이지 않는다. 어느 요인 때문인지는 이 자료로 가를 수 없다(미확인). fill(주입량을 m까지 채움)은 자연 종료 문항에서는 답이 이미 확정돼 효과가 작을 것으로 예상(추정).

## 2026-10-11 fill 실행 여부 독립 검토 (독립 검토, GPU 없음) — 결과 전, 작성자 결정 대기
- 보고서 `repro/audit/fill_decision_review_20261011.md`(입력 `fill_decision_brief.md`, 계산 `fill_decision_power.py/.json`).
- 주 세션 재계산으로 확인한 핵심 수치(s1–s8 868, v3): VisRef−TSR 정오 불일치 close 3/285, cap 71/560, eos 3/23(첫 단계 종료 방식은 세 방법 868/868 동일 = 처치 전 변수). k=1·close의 주입/m 구간별 예측 변경 0/99, 0/84, 0/25, 1/73(m의 60–100% 주입 구간 포함) → 주입량을 늘려도 자연 종료 문항 답은 안 바뀌는 쪽(단, 절대 400–900토큰 추가는 관측 범위 밖, 미확인). cap·k=1·주입≥160에서 예측 변경 17/142. close 문항은 질문의 32.8%지만 VisRef 생성 토큰의 6.3%. 모든 단계가 이미 m에 닿아 fill ≡ VisRef인 문항 133/868(그중 |V|=256이 132).
- 주 세션 주장 정정: "close 문항 때문에 1,000문항 실효 검출력이 사전등록보다 낮다"는 틀림 — 불일치는 이미 cap에 몰려 있고 사전등록 8.8%는 그 평균. 전체 %p 기준 최소 검출 효과(80%, α .05) 불일치율 8.9%면 2.64%p, 12% 3.07, 15% 3.43, 17% 3.65. 정확한 표현은 "희석"(효과가 cap/eos에만 있으면 전체 2.6%p ⇔ 그 하위집단 약 3.9%p).
- 검토 권고: (A) 사전등록대로 1,000문항 실행. 결과 전 수정 기록: MDE 문구 2.6 → 2.6–3.7%p와 희석 해석, 층별 보조 분석은 처치 전 변수(첫 단계 종료 방식)로만, 시간 초과 시 complete prefix. 이중 probe는 결정 경로 동일성 검사 통과 + 예상 시간 ≤10h일 때만 유지, 넘으면 먼저 제거. 축소안(cap만, n=392, k>1만)은 이득 없거나 편향. 실행 안 함(C)은 "1/5만 주입" 반론에 답하지 못해 신뢰도 손실이 가장 큼.

## 2026-10-11 fill 실행 전 계획 수정 (작성자 승인, 결과 전 — fill 미구현·미실행 상태) — 2026-10-10 "원인 분리 실험 실행 전 계획 수정"의 해당 항목을 대체
- 근거: 같은 날 독립 검토(`repro/audit/fill_decision_review_20261011.md`). 실행 내용(방법·1,000문항·시드·세션·이중 probe·주 비교·해석 규칙)은 그대로.
- 최소 검출 효과 문구: "2.6%p" → "2.6–3.7%p(fill−비교군 정오 불일치율 8.5–17% 범위, 80% 검출력, α 0.05)". 효과가 첫 단계 절단·EOS 문항(약 67%)에만 있다면 전체 2.6%p는 그 하위집단 약 3.9%p에 해당한다는 희석 해석을 함께 적는다. 무효 결과는 "주입량 효과 2.6–3.7%p 이상은 검출되지 않음"으로 표현.
- 층별 보조 분석: 처치 전 변수인 첫 단계 종료 방식(close / cap / eos; 세 방법·fill이 같은 첫 단계를 공유)으로만 사전 지정. k(=1 / >1)로 나눈 분석은 방법이 바꾸는 처치 뒤 변수라 탐색적으로만 보고.
- 시간 초과: 남은 문항을 같은 코드로 이어 붙이지 못하면 셔플 순서 complete prefix만 분석하고 n을 그대로 보고.
- 이중 probe: (1) 결정 경로가 바뀌지 않음을 CPU 검사로 확인하고 (2) 예상 세션 시간이 10h 이하일 때만 포함. 넘으면 문항 수보다 먼저 이중 probe를 뺀다.
- push 시점: b_main_s9를 pull해 건전성 점검을 통과한 뒤.

## 2026-10-11 b_main_s9 결과 — b_main 1,000문항 완주, 최종 LLM 추출 준비
- s9 건전성: error 0, 132문항×3 = 396행, d0095f79e691, T4×2, torch 2.11.0+cu128, s1–s8과 겹침 0, .partial 없음, 대기열 0으로 정상 종료(샤드 elapsed 7,896s / 5,301s). 누적 st/tsr/visref_raw_rank 각 1,000행(+ s1의 이전 판 visref_raw 74행은 분석 제외).
- s9 단독(v3): ST 56.1 / TSR 59.8 / VisRef 60.6, ST 첫 단계 상한 절단 61.4%.
- 1,000문항 규칙 채점(v3) 참고값 — 판정 아님(사전등록 판정은 LLM 추출 후 읽기 (a)): ST 57.8 / TSR 60.0 / VisRef 60.0. TSR−ST +2.2(0.4~3.9), VisRef−ST +2.2(0.3~4.1), VisRef−TSR 0.0(−1.9~1.8). 보고서 갱신(`repro/reports/b_main_report.html`).
- 최종 LLM 추출 준비(EF 10-10 고정 절차): v3 재채점 사본 `repro/llm/final/rows_v3.jsonl`(3,000행, 원본 불변) → `llm_extract.py prepare --control 20` → `repro/llm/final/requests.jsonl` 56건(규칙 실패 36 + 대조군 20, 빈 텍스트 13행은 보내지 않음, 정답 미포함). 실행은 작성자 터미널(gemini-3.5-flash-lite).

## 2026-10-11 visref_fill + 이중 probe 구현, CPU 점검, 세션 빌드 (작성자 승인, 결과 전)
- 코드(이전 판 백업 `repro/.backup_pre_fill_20261011/`): `visref/dpp.py` `select_tokens_fill` — `select_tokens`의 greedy 선택(head)을 고정해 먼저 넣고, 커널에 εI(ε = 1e-6 × 평균 대각 L_ii, float64)를 더해 같은 증분 Cholesky greedy로 정확히 m개까지 채움. head = visref_raw_rank 선택(함수 안 assert). 단계에 n_head·n_fill·ridge·tail_rel_spearman(꼬리 선택 순서와 관련성 r_i²의 순위 상관) 기록. `visref/methods.py` 새 방법 `visref_fill`(그 외 visref_raw_rank와 같음, _vis_room m+2 동일), 이중 probe(cfg `dual_probe`, VisRef 방법만): 주 probe 뒤 V_k 없는 문맥(`z_k [\n] </think>` + probe prompt)에서 별도 난수열(`{pid}:probe_noV`)로 두 번째 probe → H_noV·probe_noV_* 기록, 판정은 H만 사용, 생성 토큰은 gen_tokens_noV로 따로 셈. 두 번째 probe가 비유한 logit이면 job을 실패시키지 않고 기록만(`nov_failed`). `run.py` DEFAULTS `dual_probe=0`, probe2 생성기.
- CPU 점검(torch 2.14.1 CPU, transformers 4.55.0): 기존 테스트 67 + 새 테스트 9 = 76 통과. `repro/diag/fill_equiv_check.py`(작은 Qwen3 + 가짜 인코더로 run.main 전 과정) PASS — (A) st/tsr/visref_raw_rank 출력이 수정 전 코드와 행 단위로 동일(code_version·wall_s 제외), (B)(C) 이중 probe 켜기/끄기에서 H_noV 필드 외 동일, (D) fill 1단계 head 수 = raw_rank n_dpp, 주입 = m.
- 독립 코드 검토: 확정 버그 없음(L+εI 무차별 Schur 보수 greedy와 40케이스 선택 일치, 스트레스 360케이스 크래시·부족 없음). 남은 위험: 이중 probe가 GPU fp16 배치 구성(pack 그룹·패딩)을 바꿔 같은 배치 다른 job의 샘플이 드물게 갈릴 수 있음(논리적 결정 경로는 동일; b_main의 갈라짐·배치 잡음과 같은 종류). 반영: 두 번째 probe 실패 시 job 유지, 중복 DPP 계산 제거.
- 새 code_version = 39b3fb1b4121(b_main d0095f79e691과 다름 — fill 행은 별도 시리즈).
- 빌드 `repro/kaggle/runs/b_main_fill`(`arki13/visref-repro-b-main-fill`): ids = b_main 셔플 순서 1,000문항(동일 확인), skip 0, `--methods visref_fill --think-cap 1024 --total-think-cap 10240 --dual-probe 1`, timeout 41,400 / time_limit 38,700 / deadline 40,800(사전등록과 같음).
- 시간 추정(불확실): b_main 9세션 실행 시간 합 53,541s × VisRef 생성 토큰 비중 0.57 ≈ 8.5h, + 이중 probe·추가 주입 문맥 약 0.5–0.9h → 약 9–9.4h ≤ 10h → 사전등록 조건대로 이중 probe 포함. 초과하면 complete prefix 분석.
- 층별 기준 메모: fill 세션은 1단계 생각을 새로 생성하므로(같은 시드, fp16 배치 잡음 가능) 층별 변수 "첫 단계 종료 방식"은 b_main ST의 값(fill 처치 전에 확정)으로 쓰고, fill 자체의 첫 단계 종료 방식과의 일치율을 함께 보고한다.
- push: s9 점검 통과(2026-10-11) → 작성자 push 가능.

## 2026-10-11 b_main 최종 판정 (실행 전에 정한 기준, 읽기 (a) = v3 + LLM 추출) — 1,000문항
- LLM 추출(gemini-3.5-flash-lite, 작성자 터미널, 56건 모두 STOP): 규칙 실패 36 중 근거 있는 추출 사용 30, 근거 없음 거부 6(예: 106 VisRef가 텍스트 "1"인데 "0" 추출 → 거부), fixed 1(341 TSR, 텍스트 "Answering the question: C" = 정답 선택지), broke 0. 대조군 20/20 일치·근거 있음. 결과 `repro/llm/final/merged.jsonl`.
- 정확도: ST 57.8 / TSR 60.1 / VisRef 60.0 (논문 46.1 / 48.5 / 52.0).
- 판정(`repro/diag/final_judgment.py` → `.json`; 부트스트랩 2,000회 seed 0, McNemar 정확, 주 비교 2개 Holm):
  - VisRef−ST(주): +2.2%p, CI 0.3~4.1, 60:38, p 0.033(Holm 0.067), 논문 +5.9 → **효과 크기 재현 안 됨(방향은 같으나 크기 다름)**.
  - VisRef−TSR(주): −0.1%p, CI −2.0~1.7, 44:45, p 1.0(Holm 1.0), 논문 +3.5 → **효과 크기 재현 안 됨**.
  - TSR−ST(보조): +2.3%p, CI 0.6~4.0, 51:28, p 0.013, 논문 +2.4 → **재현됨**.
- 민감도(판정에 쓰지 않음): 규칙만 v3 — 같은 결론(VisRef−TSR 0.0, −1.9~1.8). (b) af 앞에서 재채점 — ST 56.0 / TSR 58.1 / VisRef 58.6, VisRef−TSR +0.5(−1.8~2.8). (c) af 행 오답 — ST 44.4 / TSR 48.3 / VisRef 47.4, VisRef−TSR −0.9(−3.3~1.5). 세 읽기 모두 VisRef−TSR CI가 논문 +3.5를 포함하지 않아 판정 동일.
- 결과를 본 뒤의 관찰(판정 아님): 읽기 (c)의 절대값(44.4 / 48.3 / 47.4)이 논문 ST·TSR(46.1 / 48.5)에 가깝다 → "답 상한에 닿은 응답을 오답 처리하는 평가"가 절대값 차이의 후보 중 하나일 수 있음(가설, 미확인).
- 보고서 `repro/reports/b_main_report.html` 최종판으로 갱신.

## 2026-10-11 외부 피드백 독립 검토 — 판정 라벨·수치 변경 없음
- 대상: 외부 도구의 피드백(표현 수정 + fill 보류 권고). 검토 보고서 `repro/audit/feedback_review_20261011.md`. 주 세션 재확인: ST 첫 단계 절단 641/1,000, (c) 읽기 VisRef−ST McNemar 94:64 p 0.0208 → Holm 0.0415, 보고서 HTML에 "대기"(LLM 열)·"문항 수가 더 필요"·"VisRef(fill 변형)" 문구 잔존 — 모두 피드백대로 맞음.
- 채택(표현): (1) "재현됨"은 사전등록 정의상 "CI가 0 제외 + 논문 값 포함 = 논문 수치와 양립"이며 효과 동등성·메커니즘 충실성 입증이 아님을 명시(라벨은 사전등록대로 유지). (2) 논문 값이 CI 밖이라는 것은 두 결과 간 차이 검정이 아님 — 논문 쪽 불일치율 3.5–20% 가정 근사 z: VisRef−TSR 2.1–3.3, VisRef−ST 2.2–3.0(가정 의존). (3) 시점: "759문항 중간 결과 확인 후, 최종 분석 전에 고정". (4) "다른 채점에서도 같다"는 "CI 기반 분류가 같다"로 한정 — 유의성은 (c)에서 Holm 0.067 → 0.042로 다름. (5) 범위: "현재 InternVL3.5-1B 운영 구현에서"; TSR 이득에는 예산 연장이 섞임, VisRef 주입/m 중앙값 0.199(2,261단계 중 1,895단계 m 미만), H 추정·Z 표현은 F.
- 기각: fill 보류 — 새 근거 없음(시간 불확실성·εI 구현상 변형은 이전 검토·사전등록 수정·승인 때 이미 반영, 보호장치 있음). εI가 충실성을 다 해결하지 못한다는 지적 자체는 맞음(fill의 목적은 DPP 선택 수(J-7) 원인 분리).
- 정정: 피드백의 "VisRef 생성 토큰 비중 41.8%"는 갈라진 TSR·VisRef 행의 gen_tokens에 1단계 토큰(각 760,920)이 중복 기록된 분모(5,704,996)를 쓴 값. fill 세션은 1단계를 자체 생성하므로 고유 토큰 분모 4,183,156 → 0.570이 맞다(2026-10-11 "visref_fill + 이중 probe 구현, CPU 점검, 세션 빌드" 항목의 57% 유지). 다만 시간 추정 9–9.4h는 여전히 검증되지 않은 추정이다.
- 반영: `repro/report.py`에 `--judgment`(사전등록 판정 표·Holm·분류 정의·범위·시점·두-결과 비교 한계·읽기별 민감도) 추가, 고정 문구 2개 수정. 보고서 재생성(입력 = `llm/final/merged.jsonl`, LLM 열 채움) → 아티팩트 v13. 앞 항목(2026-10-11 "b_main 최종 판정" 항목)의 "최종판으로 갱신"은 이 재생성으로 대체.
- 작성자 결정 필요: `AGENTS.md` L30 "All model experiments are stopped by the user…" 와 L53 "20 GPU-hours total" 조항이 현재 상태(b_main 9세션 ≈ 14.9h 실행, fill 승인)와 맞지 않음 — 갱신 여부는 작성자가 정한다(주 세션은 수정하지 않음).

## 2026-10-11 AGENTS.md 상태 조항 갱신 (작성자 지시)
- 이전 판 `.AGENTS.md.bak_20261011`. 바꾼 것은 두 조항뿐:
  - L30 "All model experiments are stopped…" → STOP(`artifacts/STOP_REQUESTED.json`, 파일 유지)은 이전 `artifacts/` 파이프라인과 압축·개선 실험에 계속 적용, `repro/` 재현 트랙(b_main 완료, 사전등록 후속 `visref_fill`·`st_official`)은 이 기록의 작성자 명시 승인으로 진행 — 에이전트는 구현·검사·빌드·분석, Kaggle/Colab push·확인·pull은 작성자. 그 밖의 GPU 작업은 여전히 명시 승인 필요.
  - L53 "20 GPU-hours total…" → 20h 한도(P0–P1 5h, P2 15h)는 이전 무료 GPU 압축 파일럿(`docs/PROTOCOL.md`) 것으로 닫힌 상태 유지·재설정 안 함. `repro/` 트랙은 b_main 우선·재실행 여유, 보조 실험은 무료 Colab. b_main 사용 약 14.9 세션시간(9세션). 실행마다 예상·실제 사용 시간을 이 기록에.
- STOP 파일과 L60(문서 수정은 STOP 해제·GPU dispatch 승인이 아님)은 그대로.

## 2026-10-11 최종 보고서 초안 (Docs 아티팩트)
- 구성: 요약 · 설정(근거 등급) · 결과와 사전등록 판정(분류 정의·시점·두 결과 비교 한계) · 민감도 · 원인 분석 · 절대값 차이와 한계 · 진행 중 실험(fill, st_official) · 재현 자료 위치.
- 원인 분석 수치는 1,000문항(merged.jsonl)으로 다시 계산: 이어가기 TSR 절단 뒤 1,103 / 자연 종료 뒤 7, VisRef 1,214 / 32. VisRef k=1 713문항 ST 479 vs VisRef 480; 자연 종료 329문항 답 변경 1(기준선 TSR 1/330), 절단 366문항 33(기준선 15/380, 맞게 14 : 틀리게 12); 답 안 </think> 재출력 352 vs 67; 주입/m 중앙값 0.199(1,895/2,261단계 m 미만).

## 2026-10-11 2차 외부 피드백 검토 (보고서 초안 대상) — 판정 수치 변경 없음, 표현·기록 정정
- 주 세션이 원자료(`repro/llm/final/merged.jsonl`)로 직접 확인: 피드백 수치 모두 맞음.
  - 코드 버전 3개: 79eef648ebd1 222행, f2422d0e72ff 954행, d0095f79e691 1,824행. 보고서 초안의 "코드 버전 하나"는 틀림 → "세 코드 버전, 첫 변경은 CPU 출력 동일성 810/810(2026-10-07 "안 쓰는 코드 분기 제거" 항목), 두 번째는 채점기만 변경(2026-10-10 "b_main_s5 빌드" 항목), 모든 출력 v3 재채점으로 통합"으로 정정.
  - VisRef k=1 713문항 정오: VisRef만 정답 15, ST만 정답 14 → 순증가 1(초안의 "주입 효과 0"은 과장). k=1은 주입 뒤 정지 판정으로 정해지는 처치 뒤 집단이고 TSR k=1 기준선과 문항 집합이 다르므로 원인 확정 불가.
  - 종료 기록: af 사용 ST 239 / TSR 187 / VisRef 225, af도 100토큰 상한 종료 164 / 118 / 143, 생각 EOS 뒤 이어감 TSR 5 / VisRef 15(초안 이어가기 설명에서 누락), 최종 답 EOS 종료 761 / 813 / 775, LLM 추출 사용 11 / 8 / 11, 빈 응답 7 / 4 / 2, 런타임 오류 0.
  - 근사 z: 논문 불일치율 30% 가정 시 VisRef−ST 1.86, VisRef−TSR 1.83 → 20% 상한 근거 없음. 본문·보고서에서 z 제거, "정식 검정 못 함"만 남김.
  - 공개 폴더는 `docs/EXPERIMENT_FLOW.md`, `docs/SPEC_LEDGER.md` 구조 → "같은 구조" 문구 정정.
  - `diag/analyze_k1_injection.json`이 868문항 기준이었음 → 기본 세션 수 9로 바꾸고 1,000문항으로 재생성.
- 채택한 표현: 고정 시점 구분(주 비교·검정 = 결과 전, CI 분류 = 759문항 확인 후), "추가 이득을 확인하지 못함", VisRef−ST "Holm 보정 후 유의한 우위 확정 못 함", diag_st_plain "af 유지 조건에서 큰 하락 관측 못 함", 주입 경로 감사 "검토한 경로·표본에서 구현 오류 발견 못 함", Z = 추론 텍스트 토큰 표현, DPP "잔차 임계값 조기 종료 = 확인된 절차 차이(논문 |V_k| = m)", 셔플 seed 0 / 생성 seed 42, probe 최대 16토큰, st_official = "공식 Thinking 설정을 참고한 큰 예산 ST"(실제 생각 예산 약 3.3만), 59.3만 포함 시 "공식 수치와 양립"으로만 서술(사전등록 규칙 문구 "대표함"보다 약하게 — 해석 표현만 낮춤), fill은 εI·추가 선택도 바뀌므로 양성이어도 단독 원인 확정 불가, fill−기존 VisRef 함께 보고.
- 반영: 보고서 문서(Docs) 수정, `repro/report.py` 판정 섹션 문구 수정 → HTML 재생성(아티팩트 v14).
- 확인: visref_fill(`b_main_fill`)은 작성자가 push해 실행 중(2026-10-11).

## 2026-10-11 공개 저장소 구성 기획
- 조사: Papers with Code ML Code Completeness Checklist, Open R1(Hugging Face), ReScience C 투고 안내, MLRC 형식 보고서 예(FairAC), NeurIPS 2026 MLRC 트랙, Cookiecutter Data Science, GitHub 대용량 파일·Zenodo 문서.
- 기획: 현재 `github_export/visref-repro` 구조 유지 + REPORT.md(영문, MLRC 형식)·CITATION.cff·results/(최종 표 CSV·JSON)·Makefile 추가, README를 최종 결과표·판정·범위·재현 명령으로 교체, 내보내기 대상에 llm/final·final_judgment·b_main_fill 추가, 공개 후 v1.0 릴리스 + Zenodo DOI. 실행은 작성자 결정 후.

## 2026-10-11 visref_fill 결과 (b_main_fill, code 39b3fb1b4121) — 규칙 채점 v3 기준 중간, LLM 추출 대기
- 건전성: 994/1,000행, 오류 0, 중복 0, T4×2, torch 2.11.0+cu128. shard0 500/500 완료(elapsed 39,557s), shard1 deadline 도달(40,772s): 진행 중 4문항 `.partial`, 미투입 2 → 미완 6(574, 617, 143, 914, 777, 865). 사전등록대로 complete prefix 976문항(shard0 500 + shard1 476)만 분석. 실제 세션 약 11.3h — 빌드 때 추정 9–9.4h는 빗나감(이중 probe 포함 조건 "예상 ≤10h"의 근거 추정이 틀렸음을 기록).
- 분석 `repro/diag/analyze_fill.py` → `.json`. b_main은 읽기 (a)(v3 + LLM), fill은 v3 규칙만(LLM 추출 전).
- 정확도(976): ST 57.8 / TSR 60.0 / VisRef 59.9 / fill 60.0.
- 주 비교: fill−TSR 0.0%p (CI −2.15~+2.36, 70:70, p 1.0), fill−VisRef +0.1%p (−2.15~+2.36, 65:64, p 1.0). fill−ST +2.25 (−0.31~+4.71, p 0.088).
- 사전등록 해석: fill−TSR CI 하한이 0보다 크지 않음 → "주입량 부족이 VisRef 효과를 줄였다"를 지지하지 않음. 주입량 효과 2.6–3.7%p 이상은 검출되지 않음.
- 층별(b_main ST 첫 단계 종료 방식, 사전 지정): close 322 fill−TSR −0.6 (−2.5~1.2), cap 629 +1.75 (−1.4~5.1), eos 25 −36.0 (−64~−8; 4:13). EOS 층(소수)에서 fill이 크게 나쁨: 첫 단계가 거의 즉시 EOS로 끝나 z가 거의 비어(raw_rank 주입 2–4개) fill이 관련성만으로 537–998개를 채워 넣고, 그 뒤 답이 이미지 속 숫자 등으로 흔들림(예: 214 "2006", 851 "1990") — 결과를 본 뒤의 관찰.
- 진단: 주입/m = 1.0(2,231단계 전부 m), 채운 수 중앙값 420(사분위 75~834), 꼬리 선택 순서와 관련성 순위상관 중앙값 −0.10. 평균 k fill 2.29 / VisRef 2.27 / TSR 2.11, 엔트로피 정지 907 vs 898, 답 안 </think> 재출력 472 vs 453, af 221 vs 224, 평균 생각 토큰 1,995 vs 2,019. fill 첫 단계 종료 방식이 b_main ST와 같음 874/976(나머지는 1단계 재생성의 fp16 배치 잡음; 층 변수는 b_main 값 사용).
- 이중 probe(2,231단계): H − H_noV 중앙값 0.0006. 판정이 V 유무로 갈렸을 단계 151(6.8%) — V 있을 때만 이어감 110, V 없을 때만 이어감 41. 단계 종료별 뒤집힘: close 8/403, cap 129/1,764, eos 14/64 → V 묶음은 주로 절단 단계에서 H를 올려 이어가기를 늘림.
- LLM 추출 준비: `repro/llm/fill/rows_v3.jsonl`(994행) → `requests.jsonl` 43건(규칙 실패 23 + 대조군 20, 빈 텍스트 4 제외). 작성자 터미널 실행 후 merge → 최종 읽기 (a)로 재분석.

## 2026-10-11 visref_fill 최종 (읽기 (a) = v3 + LLM 추출) — 판정
- LLM 추출: 작성자 지시로 gemini-3.5-flash-lite 대신 Claude Haiku 5.5가 추출기 역할(같은 공식 프롬프트, 정답 미포함, 응답 `repro/llm/fill/responses.jsonl`, model 필드에 기록). b_main 고정 절차(10-10, flash-lite)와 다른 추출기 — 벗어난 것으로 기록. 대조군 20/20 일치·근거 있음.
- merge 결함 발견·수정: Haiku가 답이 없는 응답에 빈 문자열을 10건 돌려줬는데, 기존 merge는 빈 추출을 채점해(객관식에서 기본 선택지로 매핑) 근거 검사(부분 문자열, 예 "no" ⊂ "North")를 통과시켜 정오 2건(608 0→1, 668 1→0)을 바꿨다. 보완 2의 취지(텍스트에 실제로 있는 답만 인정)대로 "빈 추출은 쓰지 않고 규칙 결과 유지"로 `llm_extract.py` merge 수정(이전 판 `repro/llm/llm_extract.before_emptyfix.py`). b_main 재-merge 결과는 행 단위로 동일(빈 추출 0건) 확인.
- 수정 후 fill merge: 규칙 실패 23 중 빈 추출 10, 근거 없음 거부 2, 사용 11(정오 변화 fixed 0 / broke 0). 결과 `repro/llm/fill/merged.jsonl`.
- 판정(complete prefix 976, `diag/analyze_fill.py llm/fill/merged.jsonl`): 규칙만 읽기와 같음. fill−TSR 0.0%p (−2.15~+2.36, 70:70, p 1.0), fill−VisRef +0.1%p (−2.15~+2.36, 65:64, p 1.0). 사전등록 해석 → "주입량 부족이 VisRef 효과를 줄였다"는 지지되지 않음; 주입량 효과 2.6–3.7%p 이상은 검출되지 않음. fill은 주입량 외에 εI·추가 선택도 바꾸므로 단독 원인 판단의 한계는 유지.

## 2026-10-11 독립 원인 검토: VisRef가 TSR보다 낫지 않은 이유 — 판정 수치 변경 없음
- 요청: 작성자 가설(이미지 토큰 선택·주입 방법이 잘못됐을 수 있다)을 포함해 원인 후보를 독립 검토하고 후속 실험 제안. 범위 = 선택 + 주입 전체. 모델 실행 없음.
- 보고서·스크립트: `repro/audit/cause_review_20261011/` (`cause_review_20261011.md`, `analyze_cause.py/json`, `extra_calcs.py/json`).
- 주 세션이 원자료(`llm/final/merged.jsonl`, sha256 앞 16자 f54371f2e502da44)로 독립 재계산해 확인한 수치:
  - 첫 단계 종료(ST 기준) cap 641 / close 334 / eos 25. 층별 VisRef−TSR 정오 불일치: close 0:2(−0.6), cap 42:42(0.0), eos 2:1(+4.0).
  - 마지막 단계가 close인 행에서 최종 답 안 `</think>` 재출력(final_y_start > 0): ST 20/334, TSR 25/352, VisRef 157/366.
  - |V|=256 문항 196개에서 VisRef−TSR −1.0%p(검토자 CI −4.6~+2.6, CI는 주 세션 미재계산).
  - k=1 문항의 ST와 비교한 예측 변경: TSR 16/730, VisRef 38/713.
- 검토 결론(검토자 판정): 구현 버그 배제 / 해석 차이 일부 열림(Z 표현층·m까지 채우는 방식) / 주원인은 구조적 한계(닫힌 생각 뒤 주입은 답을 거의 바꾸지 않고, 이어가기는 상한 절단 뒤에만 생김). 원인 순위: 구조적(높음) > Z 층·선택 머리 내용(중간) ≈ 주입 형식(중간) ≈ H 추정(중간, TSR과 공유) > 주입량(낮음, fill null).
- 제안(미승인, 작성자 결정 대기): 실험 1 `visref_all`(선택 없이 |V| 전부 재주입, ST 첫 단계 cap 문항 접두, max_context 49,152, 100문항 ≈2.7h 추정 ±30%), 해석 규칙 R1–R4(주 비교 all−TSR, 기준 5.5%p = 3.5/0.641). 실험 2 `visref_hidden`(은닉 상태 Z)은 실험 1이 R1일 때만.
- 논문 근거 재확인(`VisRef_plain.txt`): 전부 재주입은 비용 때문에 배제(L300, L381-385, 지연 2.3배), |V_k| = m 강제(L536, L578), Fig. 4(b) 76.1→79.2(L794-795), Table 7 Random 67.3 / Relevance-only 75.6 / DPP 79.3(L1579-1590).

## 2026-10-11 scorer v4 + st_official 분모 수정 (작성자 지시) — 결과 후 변경, 주 판정 수치 변경 없음
- 이유: v3는 객관식 추출이 비면 Levenshtein 최근접 선택지로 정규화해 정답이 될 수 있었다(공식 MathVista 기본값 `ignore_empty_extractions=False`와 같은 동작). 외부 피드백 검토에서 지적, 합성 예제로 재현.
- v4 (`repro/visref/scoring.py`, `SCORER_VERSION='v4_mc_empty_wrong'`): 추출 규칙은 v3와 같고, 빈 객관식 추출 = 예측 없음(오답) = 공식 `--ignore_empty_extractions` 모드(`references/sources/mathvista_official_r4/evaluation/calculate_score.py` L28, L44, L121; 대장 빈 추출 처리(X-25)). v3는 `score(..., version=SCORER_V3)`로 재현 가능. 기본값이 v4이므로 run.py·llm_extract.py·report.py 재채점·diag 스크립트가 모두 v4를 쓴다.
- 영향(원자료 재계산): 읽기 (a)(기록된 v3+LLM) 변경 행 0/3,000(`diag/final_judgment.json` `v4_changes_in_a`=0) → 주 판정 불변. 읽기 (b)(af 앞 재채점)는 ST pid 122, TSR pid 117이 v3에서 빈 추출→정답이었음 → v4: ST 56.0→55.9, TSR 58.1→58.0, VisRef 58.6 불변; VisRef−TSR +0.5→+0.6(−1.7~+2.9), VisRef−ST +2.6→+2.7(+0.5~+5.0), TSR−ST +2.1(+0.1~+4.1) 불변. v3 값은 `b_af_stripped_v3`로 함께 적음. (c)·rule_only 불변. `diag/st_plain_analysis.json`은 v4로 재실행해도 버전 문자열 외 43개 값 동일(파일은 덮어쓰지 않음).
- `repro/diag/analyze_st_official.py`: 오류 행을 분모에서 빼던 것(이전 L68) → 완료 접두의 오류 행을 분모에 남기고 오답 처리(run.py L220-221과 같음). 오류 제외 값은 `excluding_errors`로 따로 보고(판정 미사용). steps 없는 오류 행 처리 추가.
- 백업: `repro/.backup_scorer_v4_20261011/` (scoring.py, test_repro.py, final_judgment.py/json, analyze_st_official.py, EXPERIMENT_FLOW.md, SPEC_LEDGER.md 이전 판).
- 검증: CPU, torch 2.7.0 + transformers 4.55.0(고정 환경)에서 `pytest` 84 passed(신규 scorer 테스트 8건 포함). 로컬 transformers 5.14.1에서 실패하던 2건은 고정 환경에서 통과.
- st_official 중간 자료(중단): 세션 2개, 33행·14행(같은 pid 14행은 생각·답이 바이트 단위로 동일). b_main 셔플 순서 완료 접두 n=16(오류 0): 공식 설정 ST 75.0%(Wilson 50.5–89.8) vs b_main ST 68.8%, 차이 +6.3(−12.5~+25.0). 사전등록 n=100의 중간값이므로 판정에 쓰지 않는다. 4/16행이 context 한도(생각 ≥33,236토큰)까지 생성.

## 2026-10-11 논문 부수 수치 대조 (GPU 없음, `repro/diag/paper_fingerprints.py` → `diag/paper_fingerprints.json`) — 판정 수치 변경 없음
- 목적: 논문에 적힌 부수 수치와 우리 설정이 서로 맞는지를 원자료로 확인. 논문 설정을 확정하지는 못하고, 어떤 설정·읽기가 논문 수치와 어긋나지 않는지만 보인다.
- 시각 토큰 수: MathVista 메타데이터의 이미지 크기로 InternVL dynamic_preprocess를 재계산(기록된 b_main n_visual과 1,000/1,000 일치). 평균 |V|: 우리 설정(max_num=12, 썸네일 포함) 1,961.5 = 논문 "∼1,772"(8B, 단계당 토큰 평균(X-6))과 비교해 +10.7%; max_num=12·썸네일 없음 1,755.6(−0.9%); max_num 6/8/10은 −24~−41%. → 논문의 전처리·V 후보 집합은 명시되지 않음(미확인).
- 답 처리 읽기 vs Table 6 1B(46.1/48.5/52.0): (a) 57.8/60.1/60.0, (c) af 행 오답 44.4/48.3/47.4, (e) 강제 답 행 오답 25.2/26.4/26.9, (f) 둘 다 오답 24.4/25.3/26.3. 절대 수준은 (c)가 가장 가깝지만 VisRef−TSR은 모든 읽기에서 −0.9~+1.0 → 답 처리·추출만으로는 +3.5를 설명하지 못함.
- 생각 양: 생성 토큰 비 TSR/ST 1.98, VisRef/ST 2.14 vs 논문 Table 5 지연 비(8B) 1.08/1.15. ST 첫 단계 64.1%가 1,024 상한 절단. 모델 크기가 달라 참고용이지만, 논문 설정에서는 이어가기가 드물었음을 시사.
- 같은 배치 구성의 결정성: st_official 두 Colab 세션의 같은 pid 14행은 생각(최대 3,630토큰)·답이 바이트 단위로 동일. 반면 배치 구성이 다른 경로(TSR k=1 vs ST, fill 재생성)에서는 차이(16/730, 874/976) → 진단 실험은 배치 구성 고정(또는 batch=1) 필요.

## 2026-10-11 사후 관찰: 읽기 (c)와 VisRef 답 길이 (판정 아님, 판정 수치 변경 없음)
- 읽기 (c)(af 행 = 답 1,024토큰 안에 EOS 없음 → 오답): ST 44.4 / TSR 48.3 / VisRef 47.4 vs 논문 46.1 / 48.5 / 52.0. ST·TSR은 −1.7/−0.2로 근접, VisRef만 −4.6. 같은 원자료에서 af 앞 텍스트를 다시 채점한 (b)는 55.9/58.0/58.6이므로, 근접에는 "상한에 닿은 답 = 오답"이 필요하다. 엄격 추출 변형((c)+boxed·answer_phrase 규칙만) 43.7/47.7/46.7.
- VisRef가 (c)에서 논문 값에 가까워지지 않는 이유 후보: af 행 VisRef 225 / TSR 187 / ST 239. 같은 z_1(첫 단계 공유)·k=1로 V_1 주입 유무만 다른 짝(713문항): af VisRef만 70 / ST만 52(McNemar p 0.123), 답 길이 중앙값 125 vs 82토큰. 같은 문맥인 TSR k=1 vs ST(잡음 수준): 12 / 21, 84 vs 84. → 태그 없는 V 주입이 최종 답을 길게 만들어 상한 도달을 늘리는 경향(유의하지 않음, 결과를 본 뒤 확인).
- 해석 한계: 4개 이상 읽기 중 논문 두 수치에 맞는 것을 결과를 본 뒤에 고른 것. (c)의 TSR−ST +3.9는 논문 +2.4보다 (a)의 +2.3이 더 가깝다. 저자 평가 방식은 논문에 명시되지 않음(미확인).

## 2026-10-11 정정 메모: fill(ridge)의 꼬리는 관련성 채움이 아님 (판정 수치 변경 없음)
- `visref_fill`(dpp.select_tokens_fill): 머리 = rank 한정 DPP 선택(visref_raw_rank와 동일, assert), 꼬리 = 커널 L+εI(ε = 1e-6·mean L_ii)로 같은 greedy를 m까지 계속. rank 소진 뒤 이득은 ≈ ε(1 + a_jᵀ(A_SᵀA_S+εI)⁻¹a_j) → 이미 고른 집합이 가장 덜 덮은(작은 고유값) 방향에서 큰 토큰을 고름. 꼬리 선택 순서와 관련성 r_i²의 Spearman 중앙값 −0.10(`diag/analyze_fill.json`) → 관련성과 거의 무관.
- 관련성 채움(rank 뒤 r_i² 상위로 m까지)은 이전 `visref_raw`(`.backup_79eef648ebd1/visref/dpp.py` L51-52)였고 b_main_s1 74문항에서만 실행 후 제외(2026-10-07 "계획 수정(작성자 결정, s2 결과 전): fill 제외, 주 결과 VisRef = visref_raw_rank" 항목). v4 규칙 재채점: ST 59.5 / TSR 56.8 / rank 56.8 / visref_raw 59.5, visref_raw−TSR +2.7%p(4:2), visref_raw−rank +2.7%p(2:0) — 표본이 작아 판단 불가. → "fill 0.0%p"는 관련성 채움을 검증하지 않는다.

## 2026-10-11 재주입 형식·위치 엄격 재독 — 판정 수치 변경 없음
- 보고서: `repro/audit/injection_format_reread_20261011.md` (주 세션이 핵심 인용 P:203, 240-241, 244, 271-272, 295-297, 381-386과 Fig. 1·2를 직접 대조).
- 결과: 논문이 명시한 항목(재주입 대상 = 원본 V의 부분집합, z_k → Ṽ_k → H_k → 큐 순서, Ṽ_k가 H_k·최종 답 문맥에 포함, 이전 Ṽ_i 유지, 실제 문맥 추가)은 구현과 모두 일치. 태그·구분자·`</think>` 처리·position id·답 프롬프트는 논문 어디에도 없음 → 잘못 읽은 문장은 없고, 논문 미지정 이어지는 곳 `[Ṽ_k] </think>`(우리 구성)가 답 안 `</think>` 재출력·답 길이 증가의 유력 원천으로 지목됨(추론).
- 다음 후보: 고정 입력 진단(텍스트 큐 / 구분자 / `<img>` 감싸기 / 내용 대체 / 위치 재사용 / 이어지는 곳 로짓) 사전등록 후 실행.

## 2026-10-11 사후 관찰: 재출력된 답의 내용과 "주입 뒤 생각" 문제 (판정 아님)
- VisRef k=1 행 중 답 안 `</think>` 재출력 352행: 두 번째 `</think>` 앞 텍스트 중앙값 146자, 이미지 관련 단어(image/picture/figure/diagram/graph/look/shown/see) 포함 41/352. 대부분 "The correct answer is A … **Answer:** A." 같은 답 초안 → 모델이 우리 `</think>`를 생각 종료로 받아들이지 않고 초안을 쓴 뒤 스스로 다시 닫는 형식 혼동에 가깝다(이미지를 다시 따져 보는 생각은 드묾).
- 같은 첫 단계 짝 정오(ST와 비교): 재출력 352행 VisRef만 정답 9 / ST만 4, 재출력 없음 361행 6 / 10 — 소수라 판단 불가.
- 구조 관찰: VisRef의 71.3%(713/1,000)가 k=1에서 멈춰, 재주입 Ṽ가 다음 생각을 이끄는 경로("reinjected to guide subsequent reasoning", P:375)를 한 번도 거치지 않고 답 직전 주입으로만 작용한다.
- 설계 쟁점(작성자 제기): 정지 단계에서 Ṽ_k 뒤 `</think>`를 강제하는 것은 Alg. 1의 문자 그대로 읽기(10→12→17행)이지만 논문이 정한 토큰열은 아니다. 대안 = Ṽ_k 뒤 자유 생성으로 모델이 스스로 `</think>`를 내게 함(Ghosal [19] TTBC 2 "allowing the end-of-thinking delimiter (</think>) to pass"와 같은 방식). 고정 입력 진단 후보로 추가(예산 혼입 주의).
- 보충(같은 날, "주입 뒤 자유 생성" 변형의 조건): (1) 예산 — TSR도 각 단계는 모델이 `</think>`를 낼 때까지 자유 생성하므로, 자연 종료 단계에서는 VisRef 자유 생성이 불공정하지 않다. 차이는 상한 절단 단계뿐(TSR은 정지 시 `\n` + `</think>` + `\nFinal Answer:` 강제)이므로, 자유 생성은 같은 단계 상한(1,024) 안의 남은 여유로만 허용한다(절단 단계는 여유 0 → 현행과 동일). (2) 정지 통계 일관성 — 최종 답 y를 자유 생성으로 바꾸면 H_k도 같은 분포(자유 생성 후 답)에서 재야 Alg. 1 12행·17행이 같은 y 분포를 가리킨다(Ghosal식 다중 표본·답 수준 엔트로피). (3) 해석 등급 — 논문 미지정 영역의 대안 해석. Fig. 2는 Ṽ_k 바로 뒤 "Answer:" 상자를 그려 현행(곧바로 답) 쪽을 약하게 지지한다.

## 2026-10-11 Fig. 3 재판독과 단계당 토큰 비교 (판정 아님)
- Fig. 3은 VisRef vs Parallel thinking(+ST 별표) 비교이며 TSR 곡선은 없다. 가로축 = 병렬 trace 합계 thinking token 예산, 점 = trace 수 증가(단계가 아님) (VisRef_plain.txt L700-723, 캡션 L746; page-07 고해상도 확대해 읽음).
- (b) InternVL-3.5-8B MathVista 읽은 값(±약 200): ST ≈600; VisRef 점 ≈4,000 / 6,100 / 8,000 / 10,100 / 12,100 / 14,500(간격 ≈2,000 → trace당 ≈2,000토큰 ≈ ST의 3.3배; 첫 점을 C=1로 보면 6.7배 — 대장 Table 5 지연 시간(A-4)의 두 읽은 값과 같은 모호성); Parallel thinking 점 ≈1,900 / 3,150 / 4,450 / 7,550 / 9,250 / 12,750 / 14,500. 시각 토큰 포함 여부 미확인. Table 5(+15% 지연)와의 수치 간 차이 그대로(원인 미확인, 지연 시간 대 이득(C-12)).
- 우리 1B: 이어가기 단계(k≥2) 길이 TSR 1,115단계 중 97%, VisRef 1,261단계 중 94%가 1,024 상한 도달(평균 1,011 / 987); 둘 다 이어간 248문항의 2단계 길이 차 중앙값 0 → 단계당 토큰은 같고, 차이는 이어가는 빈도. 생각 토큰 평균 ST 761 / TSR 1,888 / VisRef 2,006(중앙값 모두 1,024).
- 해석(추론): 논문 VisRef는 trace당 ST의 수 배를 생각 → 주입 뒤 다시 생각하는 경우가 흔했을 가능성. 우리는 71%가 k=1 정지(자연 종료 뒤 H 중앙값 0.003–0.007) → 이미지가 다음 생각을 이끄는 경우가 드묾. 정지 통계(H 추정) 항목의 우선순위를 높일 근거.
- 보충(같은 날): Fig. 3의 Parallel thinking ≠ TSR — 병렬 thinking = 독립 텍스트 trace 여러 개 + 다수결("samples multiple text-only reasoning trajectories", P:714-716; [19],[46]), TSR = 한 trace를 "Wait, think more"로 연장("text-only reflection without visual refocusing", P:1287; [34]). 논문에 TSR의 토큰 사용량은 없다(Table 5 지연뿐).
- 상한 절단 단계에서의 이미지: VisRef cap→이어감 1,214 / cap→정지 613. 이어가면 Ṽ_k 뒤 새 1,024토큰 단계가 있고, 정지하면 Ṽ_k 직후 `</think>` + `\nFinal Answer:` 강제라 생각 여유가 없다. 다만 답 영역에서 생각을 이어 가는 경우가 흔함: 1단계 절단 후 둘 다 k=1 정지한 344문항에서 답이 1,024 상한까지 간 비율 TSR 30%(102) / VisRef 39%(134).
- 진단 설계안: "정지 시 추가 여유 A"를 TSR·VisRef에 똑같이 부여(TSR = [z_1]\n 뒤 자유 생성 ≤A, VisRef = [z_1]\n[Ṽ_1] 뒤 자유 생성 ≤A, 모델이 스스로 `</think>`를 내면 답, 못 내면 현행 강제 답). VisRef에만 주면 TSR−ST 이득의 주원인인 예산 연장 효과를 다시 재게 됨. 대상에 절단·정지 344문항 추가, 비교는 새 TSR-A vs VisRef-A(b_main TSR과 직접 비교하지 않음).

## 2026-10-11 계획 독립 검토와 1단계 probe 사후 탐색 — 판정 수치 변경 없음
- 보고서·검증: `repro/audit/plan_review_20261011.md`. 분석 스크립트 `repro/diag/analyze_step1_probe.py` → `diag/step1_probe_analysis.json`(결과를 본 뒤 탐색, 사전등록 아님).
- 관찰(주 세션 재계산): 저장된 1단계 probe 즉답(T=1, pid별 공통 probe 난수, 같은 z_1)은 VisRef 58.9 / TSR 57.2, 28:11, McNemar p 0.0095, CI +0.5~+3.0. 차이는 불확실(어느 쪽이든 이어감) 309문항에서만(36.6 / 31.1, 28:11), 둘 다 k=1인 691문항은 0:0. 같은 309문항의 최종 답은 42.4 / 43.0(30:32) → 즉답에서 보이던 V_1 이득이 이어가기 뒤 사라짐.
- 정정: (1) "자연 종료 뒤 H 중앙값 0.003–0.007"(이 파일 'Fig. 3 재판독' 항목) → 그 값은 1단계 전체 중앙값이며 자연 종료만은 TSR 2.4e-5 / VisRef 5.8e-5. (2) Fig. 3 "trace당 ≈3.3× ST"는 읽은 값이 불확실(1.7–6.8배), Table 5와의 충돌도 미확인. (3) 자유 생성은 탐색 변형으로 분류(Fig. 2 "Answer:"는 즉답 쪽 지지).
- 검토 권고(작성자 결정 대기): 즉답 확인 실험 사전등록(1,000문항, TSR_imm / VisRef_imm / Random_imm / Rel_imm, 시드 3) → 조건부 이어가기 상한 4,096 소표본. Set N 형식 변형·Set C 여유 A는 축소 또는 보류. 이후 실행은 z_k 토큰 id와 DPP 선택 인덱스를 저장.

## 2026-10-11 외부 피드백(계획 검토에 대한 2차 검토) 반영 — 판정 수치 변경 없음
- 주 세션 재계산으로 확인: 309문항 probe 차이 +5.50 vs 최종 −0.65, 문항별 차이의 차이 +6.15(부트스트랩 20,000회 95% CI −0.32~+12.3; 피드백 10만 회 −12.62~+0.32는 부호 반대 표기) → **이득 감소 자체도 유의하지 않음**. 309문항 중 한 방법만 이어간 문항 61. 최종 답 v4 규칙 재채점 131:133 유지. VisRef 첫 단계 n_inject < m 842/1,000.
- 표현 정정: (1) "이미지가 즉답에서 도움 → 이어가기 뒤 사라짐/절단이 지움"은 과함. 근거에 맞는 표현 = "추가 시각 재주입이 1단계 probe 즉답에서 이득을 보였지만(사후 탐색), 최종 출력에서는 추가 이득을 확인하지 못했다". 두 방법 모두 처음부터 이미지를 보므로 '이미지 유무'가 아니라 '재주입 유무' 비교이고, probe(T=1·boxed·16토큰)와 최종 답(T=0.6·top-k/p·1,024·af)은 생성 정책·난수가 달라 원인(절단·강제 답)을 분리할 수 없다. (2) Fig. 2의 "Answer:"는 Adaptive Stopping Criterion 패널의 엔트로피 계산 도식이므로 최종 답 생성 방식(즉답)을 지지하는 근거로 쓰지 않는다(이전 기록의 해당 문장 철회). (3) Fig. 3 배수 1.7–6.8배는 읽는 가정별 후보값이지 검증된 범위가 아니다. (4) Table 7은 8B VisRef 선택 방식 비교이므로 1B 즉답 변형은 관련 탐색이지 그 ablation 실험의 재현이 아니다. (5) hidden_rank_check "12문항 중 5 불일치"는 과함: 3문항은 절단 단계의 '\n' separator로 원 길이가 1토큰 긴 것(코드가 잘라냄, `diag/hidden_rank_check.py` L114-116), 2문항은 고유 토큰 수 차이 → "정확한 id 일치는 확인할 수 없고 복원 통계의 차이가 관찰됨". (6) 자연 종료층 "정확도 판별력이 거의 없다"는 단정 철회: 예측 327/329 동일·fill close층 4:6은 낮은 민감도를 시사할 뿐, 정답률 약 72%라 형식 변형이 오답을 바꿀 여지는 남음 → 메커니즘 지표 1차, 정확도 2차로 유지.
- 즉답 확인 실험 설계 보완(사전등록 초안 반영 예정): 같은 1,000문항 반복 확인이라는 범위 명시, 시드 반복은 답 표본만 반복하므로 문항 단위(시드 평균) 분석; 선택 방식 비교는 문항별 주입 수를 DPP 실제 수 n_i에 맞춘 Random_n·Rel_n을 주 비교로, m개 팔은 별도 계열; 답 생성 정책은 모든 팔 동일(주: 관찰과 같은 probe 정책 T=1로 재현, 보조: 최종 답 정책); 절단 가설은 별도 실험에서 TSR·VisRef × 단계 상한 1,024/4,096을 같은 실행·같은 답 정책으로 비교하고 "단계 예산 정책 효과"로 해석; 100–150문항은 메커니즘 진단용; 비용은 파일럿으로 측정 후 확정.

## 2026-10-11 실행 전 계획: step1_branch (309문항, 조건 A–D) — 결과 전 고정 (작성자 결정: 309문항)
- 질문: 1단계 뒤 재주입 Ṽ_1의 효과가 (1) 즉답에서 재현되는가, (2) 이미지를 본 뒤 더 생각하게 해도 유지되는가. 목적은 "우리 구현의 어느 구간에서 효과를 잃는가"의 위치 찾기(논문의 효과는 작업 가정으로 둠).
- 대상: b_main에서 TSR 또는 VisRef가 1단계 뒤 이어간 309문항(`repro/diag/step1_branch_ids.json`; 1단계 종료 cap 297 / eos 7 / close 5). 방법의 정지 결정으로 고른 집단이라 일반화 제한을 명시한다. 순서 = b_main 셔플 순서(seed 0). 파일럿 = 앞 20문항(시간·건전성 확인용; 규칙이 그대로면 본 분석에 포함).
- 입력: b_main ST 행의 1단계 생각 텍스트(세 방법 공유, `repro/diag/step1_branch_samples.json`)를 재토큰화. 절단 단계의 '\n' separator는 b_main 규칙대로 분리(z 토큰 = 앞 think_tokens개). Ṽ_1 = 같은 DPP(`select_tokens`, Z = z_1 입력 임베딩, m = ⌊0.3|V|⌋) 재계산. 복원 통계(재토큰 수 = b_main think_tokens 일치율, n_sel = b_main n_dpp 일치율)를 보고하고, z 토큰 id와 선택 인덱스를 결과에 저장.
- 조건(문항마다 모두 실행, 같은 z_1에서 출발):
  - A: `[z_1][sep]</think>` + "\n\n\\boxed" → 답
  - B: `[z_1][sep][Ṽ_1]</think>` + "\n\n\\boxed" → 답
  - C: `[z_1][sep]` 뒤 자유 생성 ≤512토큰(T 0.6, top-k 50, top-p 0.95; 모델 `</think>`·EOS·512에서 끝) → 모델이 `</think>`를 안 냈으면 (`\n` 필요 시) `</think>` 붙임 → "\n\n\\boxed" → 답
  - D: `[z_1][sep][Ṽ_1]` 뒤 C와 같은 자유 생성 → 같은 방식으로 답
- 답 정책(모든 조건 동일): "\\boxed" 뒤 상자가 닫히면 종료(≤16토큰). T=1(top-k/p 없음) 3개 + greedy 1개. 난수: 조건 간 같은 (pid, 표본 번호) 시드, C·D 자유 생성도 같은 시드(짝 비교).
- 채점: '\\boxed' + 답 텍스트, scorer v4. 상자 미완결·내용 없음 = 오답.
- 주 분석: 문항별 T=1 3개 평균 정답률. 공동 1차 비교 (1) B−A(즉답 재현), (2) D−C(이미지 뒤 생각). 짝 부트스트랩 95% CI(2,000회, seed 0), 두 비교 Holm. 보조: greedy 답 McNemar, D−B, C−A, (D−C)−(B−A), 1단계 종료 방식별(기술).
- 메커니즘 지표(C vs D): 자유 생성 길이, 모델이 스스로 `</think>`로 닫은 비율, 512 도달 비율, 이미지 관련 단어 언급률.
- 판정 규칙: ① B−A CI 하한 > 0 → "즉답 재주입 이득 재현". ② D−C CI 하한 > 0 → "이미지 뒤 생각에서도 이득 유지" → '주입 뒤 생각' 변형을 다음 VisRef 실험 후보로. ③ ① 성립·② CI가 0 포함 → "생각 구간에서 이득 확인 안 됨" → 이어가기 구간(1B의 생각 이어가기·절단) 조사. ④ ① 불성립 → "즉답 이득 재현 안 됨(b_main probe 관찰은 잡음 가능)".
- 검출력: n=309, 불일치율 10–15% 가정 시 짝 차이 SE ≈ 1.8–2.2%p, 80% 최소 검출 효과 ≈ 5–6%p(결과를 본 뒤 관찰한 크기 +5.5와 비슷). 무효 결과는 "5–6%p 이상의 효과는 검출되지 않음"으로만 표현.
- 비용 규칙: 파일럿 20문항의 시간으로 본 실행 시간을 추정해 3시간을 넘으면 자유 생성 상한을 256으로 줄이는 것만 허용(파일럿 정확도는 보지 않고 시간만 사용).
- 실행: Kaggle T4(1–2장), 코드 `repro/diag/step1_branch.py`. push·status·pull은 작성자.

## 2026-10-11 step1_branch 구현·파일럿 빌드 (결과 전)
- 코드: `repro/diag/step1_branch.py`(조건 A–D, b_main의 engine·DPP·scorer v4 재사용, z id·선택 인덱스·복원 통계 저장), 분석 `repro/diag/analyze_step1_branch.py`(결과 전 고정), 빌더 `repro/kaggle/build_step1_branch.py`. 입력 `diag/step1_branch_{ids,samples}.json`(1단계 종료 cap 297 / eos 7 / close 5; cap 297행은 모두 separator '\n'으로 끝남).
- 사전등록 보충(결과 전): 공동 1차 비교의 "규칙 충족" = 95% CI 하한 > 0 그리고 양측 부트스트랩 p의 Holm 보정 < 0.05.
- 검증: CPU 테스트 `tests/test_step1_branch.py` 5개(분리 규칙, 상자 종료, 생각 닫기, 결정성, V=0이면 B≡A·D≡C로 짝 난수 확인) 포함 전체 89 passed(torch 2.7.0, transformers 4.55.0). 노트북 디코딩·파일 sha 일치·`--help` 실행 확인. 분석 스크립트는 합성 행으로 코드 경로만 확인(데이터 아님).
- 파일럿: `repro/kaggle/runs/step1_branch_pilot`(앞 20문항, think_extra 512, T4, timeout 3,600s). 파일럿에서는 시간·복원 통계만 보고 정확도는 보지 않는다. 본 실행은 파일럿 pid를 `--skip-from`으로 건너뛰고 나머지 289문항.

## 2026-10-11 공개용 폴더 갱신 (github_export/visref-repro; 지금 private, 실험 완료 후 public; 저장소 생성·커밋·push는 작성자)
- 작성자 결정: 루트가 아니라 export를 최신화해 올림. 10/10 이후 코드·테스트, 추가 run(b_main_s9, b_main_fill, b_main_rescore_v2/v3), audit·diag 검토, step1_branch 사전등록·파일럿을 포함. LICENSE 저작권자 AhnJG.
- `scripts/make_github_export.py` 변경: CODE에 `kaggle/build_step1_branch.py`, `audit/cause_review_20261011/*`, `llm/final`·`llm/fill`의 requests/responses/merged(diag 스크립트가 merged를 읽음) 추가; RUNS에 `b_main_fill`, `b_main_rescore_v*`, `step1_branch_pilot` 추가; RUN_KEEP에 `summary.json`, `rows.jsonl` 추가; 숨김 파일(`.tmp_*`, 백업) 제외. 계속 제외: `audit/review_inputs`(EF 발췌 사본), `llm/*/rows_v3.jsonl`, 번들 노트북. 이전 판: `scripts/.make_github_export.bak_20261011`, `github_export/.visref-repro_snapshot_20261010`, `github_export/.README_export.bak_20261011`, `github_export/.LICENSE_export.bak_20261011`.
- export README: 상태 문구(b_main 완료, 판정 파일 `repro/diag/final_judgment.json`), 구조 목록, 테스트 요구사항(torch + transformers) 갱신. 결과표는 공개 시 채움.
- 검증: git 추적 대상 200개 파일, 72.8MB, 최대 22.9MB(`repro/llm/final/merged.jsonl`). PDF·.pt·노트북 0건. CPU `pytest` 89 passed(torch 2.11.0 + transformers 4.55.0, Python 3.13). 새 복사본에서 `scripts/fetch_sources.py` 17/17 sha256 일치, `diag/final_judgment.py` 재실행 결과가 커밋본과 바이트 단위로 동일.
- 미해결(작성자 결정 필요): README 1단계 `report.py ... --rescore-data` 명령으로 재생성한 결과(scorer v4, TSR 600/1000)가 커밋된 `repro/reports/b_main_report.json`(scorer "recorded (per-row, may mix versions)", TSR 601/1000)과 다르다. 재생성값은 위 2026-10-11 "b_main_s9 결과" 항목의 v3 규칙 채점 참고값(TSR 60.0)과 같다. 보고서를 재생성할지, README 명령을 고칠지는 아직 정하지 않았다.

## 2026-10-11 step1_branch 파일럿 결과(시간·복원만) → 본 실행 빌드
- 파일럿(20문항, T4×2, code a9e2cafef0a8, torch 2.11.0+cu128, transformers 4.55.0): 8문항 묶음 139–162초(문항당 약 18–20초/GPU), 두 GPU 병렬 기준 문항당 약 10초, 세션 약 6분. 복원: 재토큰 수 = b_main think_tokens 20/20, 선택 수 = b_main n_dpp 20/20(차이 0), 오류 0, 답 형식 유효 320/320(조건 합산; 조건별 정확도는 보지 않음).
- 결과 파일은 작성자가 `github_export/visref-repro/...`에서 pull해 그 아래에 저장됨 → `repro/kaggle/runs/step1_branch_pilot/output/`로 복사(원본은 그대로).
- 비용 규칙 판단: 본 실행 289문항 예상 약 50분(< 3시간) → 자유 생성 상한 512 유지. 본 실행 `repro/kaggle/runs/step1_branch_main`(파일럿 20 pid 건너뜀, timeout 7,200s, 코드 묶음 sha 파일럿과 동일).

- push 완료: `github.com/AhnJG/visref-repro`(private) `main` = `7ec0965`(작성자가 로컬에서 만든 커밋 그대로, 파일 200개). 원격 트리의 파일 수 200을 확인했다.

## 2026-10-11 step1_branch 본 실행 결과 (309문항 = 파일럿 20 + 본 289) — 실행 전 규칙 그대로 판정
- 데이터: 본 실행 출력은 작성자 자동 pull로 `repro/runs/kaggle/visref-step1-branch-main/20261011T160142Z/`에 저장(사전 안내한 `repro/kaggle/runs/step1_branch_main/output/`가 아님). JSONL 145+144줄, 고유 pid 289, 파일럿 skip 20과 합쳐 309/309. 같은 폴더 아래 반복 다운로드 166벌은 JSONL sha256 전부 동일(6df86ae1d93e…, 0fbb8ba07645…) → 데이터 영향 없음. 번들 src 10개 파일 sha = 로컬 repro와 동일, code_version a9e2cafef0a8(파일럿과 같음), Kaggle torch 2.11.0+cu128 / transformers 4.55.0, T4×2, GPU별 약 2,918초, 오류·OOM 0.
- 분석: `python3 diag/analyze_step1_branch.py <파일럿 2개> <본 2개> --ids diag/step1_branch_ids.json --out diag/step1_branch_analysis.json`(결과 전 고정 코드, 수정 없음).
- T=1 3개 평균 정답률(%): A 33.98 / B 35.38 / C 35.28 / D 36.25. greedy: A 35.28 / B 36.57 / C 39.81 / D 37.86. 답 형식 유효 99.1–99.6%.
- 공동 1차: B−A +1.40%p, 95% CI −1.19~+3.99, Holm p 0.578 (greedy 18:14). D−C +0.97%p, CI −2.27~+4.21, Holm p 0.578 (greedy 14:20). → ①·② 모두 불성립.
- 판정(사전 규칙 ④): 즉답 재주입 이득 재현 안 됨. b_main 1단계 즉답에서 본 +5.5%p(309문항, 표본 1개)는 잡음이었을 가능성이 있다. 무효 결과는 사전 문구대로 "5–6%p 이상의 효과는 검출되지 않음"으로만 표현한다(CI 상한은 약 +4%p).
- 보조(판정에 쓰지 않음): D−B +0.86 (CI −2.16~+3.88), C−A +1.29 (CI −1.83~+4.42; greedy만 25:11, McNemar p 0.029 — 보정 없는 보조 지표 하나), (D−C)−(B−A) −0.43 (CI −4.21~+3.24). 1단계 종료별 T=1: cap 297문항 A 33.2 / B 34.7 / C 34.6 / D 35.6, eos 7, close 5(작아서 기술만).
- 메커니즘(C vs D): 모델이 스스로 `</think>`로 닫음 99 vs 82, 512 도달 203 vs 220, 이미지 관련 단어 언급률 22.7% vs 31.4%. 이미지를 넣으면 생각 글에 이미지 언급은 늘지만 정답률로 이어지지 않았다.
- 복원: 재토큰 수 일치 301/309, 선택 수 일치 296/309(차이 ±1–3개), 둘 다 일치 293. 결과를 본 뒤 민감도 분석(판정 아님): 일치 293문항만 보면 B−A +1.59 (CI −1.14~+4.44), D−C +0.57 (CI −2.73~+3.75)로 결론 같음. 조건 A–D는 같은 z·같은 선택을 공유하므로 조건 간 비교는 이 불일치와 무관하다.
- 해석(추론, 사실 아님): 우리 구현에서 TSR과 VisRef가 갈리는 곳은 이 309문항뿐이다. 논문 1B의 VisRef−TSR +3.5%p(`references/papers/VisRef_fulltext/VisRef_layout.txt` L900: 48.5 → 52.0)를 이 309문항 안의 1단계 재주입만으로 내려면 309문항 안에서 약 +11%p가 필요하지만, 이번 CI 상한은 약 +4%p(1,000문항 환산 약 +1.3%p). 따라서 우리 1단계 지점의 재주입은 논문 크기의 차이를 설명하지 못한다. 논문의 이득이 다른 곳(정지 규칙·단계 길이·주입 방식·2단계 이후)에서 나왔을 가능성은 이 실험으로 판단할 수 없다.
- 다음: 사전 규칙 ④에는 후속 실험이 정해져 있지 않다. 남은 의문(주입 형식·위치, 단계 경계, 썸네일, 상한 도달 답 처리)은 논문에 명시되지 않음(미확인).
- (10-12 결과를 본 뒤 계산, 사전등록 아님) 같은 분석 함수(`pair`, 부트스트랩 seed 0)로 D−A를 계산: +2.27%p (95% CI −0.97~+5.72, greedy 23:15, McNemar p 0.256). greedy 정답률 순서는 C 39.81 > D 37.86 > B 36.57 > A 35.28로 T=1 평균 순서(D 최고)와 다르다 → "D가 가장 좋다"는 309문항에서 확인되지 않음.

## 2026-10-12 실험 종료 + 공개 폴더 완성 (작성자 결정)
- 실험 종료: 새 GPU 실험 없음. st_official은 미완료로 남기고 분석하지 않는다.
- step1_branch 본 실행 출력을 `repro/kaggle/runs/step1_branch_main/output/results/`로 복사(원본 `repro/runs/kaggle/visref-step1-branch-main/20261011T160142Z/`는 그대로, JSONL sha256 동일). `scripts/make_github_export.py` RUNS에 `step1_branch_main` 추가(이전 판 `scripts/.make_github_export.bak_20261012`).
- `repro/reports/b_main_report.*` 재생성: README 명령(`--rescore-data`, 규칙 채점 v4만)으로 다시 만든 결과와 커밋 파일이 달랐던 미해결 건(10-11 기록) 해소. 새 값 ST 57.8 / TSR 60.0 / VisRef 60.0(규칙만; 판정 읽기 (a)의 TSR 60.1과의 차이 1행은 LLM 추출로 바뀐 341 TSR). 이전 판(기록값, 판 혼합) 백업 `repro/.backup_report_20261012/`. 판정 수치 변경 없음.
- 공개 폴더에서 직접 관리하는 새 파일: `REPORT.md`(영문 MLRC 형식; 초안 작성 후 수치를 JSON·merged.jsonl·EF와 대조, 근거를 찾지 못한 문장 2개 삭제), README 결과표·후속 실험·재현 명령 갱신(이전 판 `github_export/.README_export.bak_20261012`), `results/`(`scripts/make_results.py`가 diag JSON에서 생성), `CITATION.cff`(cffconvert로 스키마 1.2.0 검증, 저자 표기 alias AhnJG), `Makefile`(fetch / analysis / results / check / report / test), `.gitignore`에 `build/`.
- 검증: 새 복사본에서 `make analysis results report` → final_judgment.json, analyze_fill.json, step1_branch_analysis.json, results/, b_main_report.json 모두 바이트 동일(HTML은 생성 시각만 다름). CPU pytest 89 passed(torch 2.7.0, transformers 4.55.0). 새·변경 파일에서 키 패턴·개인 경로·이메일 0건.

## 2026-10-12 공개 폴더에 그림 2개 추가 (작성자 결정: GitHub 저장소, 결과 요약 + 방법 흐름도, 영어)
- `figures/paired_differences.svg`: 세 비교(VisRef−ST, VisRef−TSR, TSR−ST)의 짝 차이와 95% 부트스트랩 CI(점·막대) 옆에 논문 Table 6 차이(마름모). `scripts/make_figures.py`(표준 라이브러리)가 `results/paired_differences.csv`에서 그림. 색은 dataviz 기본 팔레트 1·2번(파랑·주황), 검증 스크립트로 라이트·다크 모두 통과, 모양(점 vs 마름모)으로도 구분. SVG 안에 라이트/다크 색을 함께 넣음.
- `figures/method_overview.svg`(손으로 배치): 이미지+질문 → 생각 단계 → (VisRef만) Ṽ_k 재주입 → 정지 확인 → 답, "Wait, think more" 반복, ST·TSR 경로. 논문 명시 값(δ 0.25, K_max 10, m = ⌊0.3|V|⌋)은 상자 안, 우리가 정한 것(타일링·썸네일, 토큰 상한, Z = 입력 임베딩, DPP 계수 조기 종료 ≈ m/5 = 확인된 절차 차이, 주입 형식, H 추정, 답 처리)은 주황 메모.
- README(방법 표 아래 흐름도, 결과 절 첫머리 결과 그림), REPORT(Figure 1 = §2.1, Figure 2 = §3.1), 파일 위치 표, Makefile(`figures` 목표, `check`에 figures 포함) 갱신.
- 검증: 브라우저(Chromium)로 라이트·다크 렌더링을 눈으로 확인(겹침·잘림 수정 후 최종본). 장치(Python 3.10)와 컨테이너에서 결과 그림 생성물이 같음, 새 복사본에서 `make analysis figures` → results/·결과 그림 바이트 동일.

## 2026-10-12 GitHub push
- 방법: `AhnJG/visref-repro`를 받고, 공개 폴더의 변경 파일 32개(수정 5, 새 파일 27)를 옮겨 sha256 32/32 일치를 확인한 뒤 커밋·push. 삭제 파일 0. 비밀 키 패턴·개인 이메일·개인 경로 0건.
- 결과: `main` 7ec0965 → 1379f5f ("Complete reproduction: step1_branch results, report, tables and figures"), 원격 파일 227개.
- (같은 날) README 커밋 추가 후 push: `main` = 56cd36a ← b6efd0a ← 2d9ddad. README 변경: "Who did what" 절(EF의 작성자 결정·실행 기록에 근거), CPU 재현 1단계에서 torch 설치 제거(테스트에만), 본 실행 GPU 비용(9세션 약 14.9 세션시간). Kaggle 사용자명 arki13은 그대로 두기로 함(작성자 결정).
- (같은 날) 공개 전 점검에서 README 라이선스 문구가 `repro/llm/`(MathVista 질문·공식 추출 프롬프트 포함)와 `repro/diag/`를 빠뜨린 것을 고쳐 push → `main` = d3be139. 모델 라이선스 Apache-2.0은 Hugging Face 모델 카드(OpenGVLab/InternVL3_5-1B)에서 확인.
- (같은 날) 로컬 README가 옛 판으로 남아 있던 것(병렬 전송 순서 문제)을 `git show HEAD:README.md`로 맞춤. 한국어 요약 `README.ko.md`(목적·결과·후속 실험·역할·재현 방법·라이선스, 수치는 results/와 같음)를 추가하고 README 첫 줄에 링크 → push, `main` = 4ae919c. 로컬 파일은 같은 내용으로 맞춰 둠(로컬 git은 `git fetch origin && git reset origin/main` 필요).
- (같은 날) 작성자가 저장소를 공개로 전환. 확인: API `private: false`, 설명·토픽 6개 반영, 라이선스 MIT 인식. 비로그인 접근으로 저장소 페이지·README.ko.md·그림 SVG 2개 모두 200, README HTML에 그림 2개와 한국어 요약 링크·"Who did what" 절 포함. Chromium으로 라이트·다크 화면 캡처해 그림 표시 확인(캡처 환경에서 GitHub CSS는 안 불러와짐 — 그림과는 무관). `main` = 4ae919c.

## 2026-10-12 공개 저장소 외부 검토 반영 (작성자 결정: 1–4 전부) — 결과 후 변경, 판정 불변
- 검토 원문: `docs/reviews/VISREF_REPRO_PUBLIC_REVIEW_20261012_KO.md`(HEAD 4ae919c 기준). 5개 지적을 원자료로 재확인했고 모두 맞았다.
- 1) step1_branch 부트스트랩 p의 Python 버전 의존(원칙 2에 따른 결과 후 변경 기록): 문항 정확도(3분의 1 단위 float)를 더하면 정확히 0인 재표본 평균이 ±1e-15로 갈려 꼬리가 바뀐다. Python 3.11에서는 저장값 p_boot 0.289 / 0.575, Holm 0.578이 재현되지만 3.12·3.13에서는 0.281 / 0.583, Holm 0.562 / 0.583(Python 3.12의 `sum()` 변경). 수정: 정답 개수(정수) 차이로 계산하고 마지막에 나눔. 규칙(양측 p = 2·min(P(b≤0), P(b≥0)), CI와 같은 재표본)은 그대로이고 정확한 0은 두 꼬리에 모두 들어간다. 새 값 p_boot 0.329 / 0.584, Holm 0.658 / 0.658(3.11·3.12·3.13 동일, CI·나머지 필드 불변, 판정 ④ 불변). 이전 출력은 `repro/diag/step1_branch_analysis_float_py311.json`으로 보존. 테스트 `tests/test_analyze_step1_branch.py` 추가.
- 2) builder 덮어쓰기: `kaggle/build.py`에 `new_run_dir`(run_config.json·notebook.ipynb·output이 있으면 거부, `--overwrite`는 아직 push하지 않은 세션 재빌드용)를 두고 build_step1_branch·build_hidden_check가 사용, build_colab도 같은 규칙. 테스트 `tests/test_build_guard.py`. README GPU 예시는 `rerun_*` 새 이름으로 변경.
- 3) 외부 근거 경로: 공개 저장소에 `docs/SOURCES.md` 추가(arXiv 버전·PDF sha256, `pdftotext -layout`/`pdftotext`(poppler 22.02.0) 재생성 방법, 핵심 수치의 페이지·절·표: Table 6 p.14, δ·m·K_max §5 p.6, |V_k|=m §4 p.5, ∼1,772 §4 p.4, 환경 p.12, Table 5 p.14, InternVL3.5 Table 3 p.11). 확인: `LAY`·`VisRef_plain.txt`·`REF/InternVL3.5.txt`는 로컬 PDF에 같은 명령을 돌린 결과와 바이트 단위로 같다. arXiv는 이 환경(클라우드·장치)에서 접속 불가라 공개 PDF와의 바이트 일치는 미확인. SOURCES.json의 main/master URL 5개를 커밋 고정(파일 sha256 일치 확인; MathVista c5acd37d, fast-map-dpp 6ab745ca, DEER c9dd19fb), export 스크립트도 같은 값으로 수정. SPEC_LEDGER "논문 PDF는 로컬에 없다"에 2026-10-07 작성 당시 상태 표시.
- 4) 문구: README·README.ko·REPORT에 "주 비교·검정은 첫 392문항, CI 분류 기준은 759문항 중간 결과를 본 뒤 최종 분석 전에 고정 → 엄격한 의미의 사전등록 아님, 후속 실험 규칙은 각 결과 전", TSR−ST는 "논문 수치와 양립(효과·메커니즘 동일성 증명 아님)", fill은 "이 채움 변형에서 추가 이득 없음, 양만의 효과는 분리하지 못함"(사전등록 판정은 기록으로 유지). README 테스트 설치에 pytest·가상환경 추가.
- 검증: 클라우드 사본에서 `make check`가 Python 3.11·3.12·3.13 모두 통과, CPU pytest 98 passed(torch 2.7.0, transformers 4.55.0; 고정 버전 torch 2.11.0 환경은 미검증).

## 2026-10-12 README.ko.md 전체 번역 + README 숫자 자동 대조 (작성자 결정: 영문 README를 메인으로 유지, 한국어 페이지는 요약이 아닌 전체 번역)
- `README.ko.md`: 요약판을 README.md 전체 번역으로 교체(번역 후 숫자·명령·링크·코드 블록·BibTeX를 원문과 대조). 첫머리에 "두 문서가 다르면 영문이 기준" 문구. 용어는 EF·REPORT와 맞춰 "읽기 (a)".
- `scripts/check_readme_numbers.py`(표준 라이브러리): `results/`의 accuracy·paired_differences·followups 값(정확도, 짝 차이, CI, McNemar p, Holm p, 논문 값)이 README.md와 README.ko.md 모두에 들어 있는지 확인하고 하나라도 없으면 실패. `make check` 마지막 단계로 추가, README 두 곳의 `make check` 설명과 파일 목록 갱신. 한계: 값이 들어 있는지만 보므로 옛 값이 새 값과 함께 남아 있으면 잡지 못한다.
- 음성 대조: 복사본의 README.ko.md에서 0.658을 0.578로 바꾸면 실패(exit 1).
- 검증: 클라우드 사본 `make check`가 Python 3.11·3.12·3.13에서 통과, 장치 Python 3.10에서 검사 스크립트 통과, 두 README의 상대 링크·앵커 확인.
- push: `main` = 8c9be58. 공개 폴더의 README.md·README.ko.md·Makefile·scripts/check_readme_numbers.py를 같은 내용으로 맞춤(sha256 4/4 일치). Makefile은 원격 파일 전송이 막혀(보호 파일) 장치에서 같은 텍스트 치환으로 수정.

## 2026-10-12 README 문구·구성 수정 (작성자 결정: README 2개 + REPORT 문구 맞춤, H 변경 이유는 기록된 사실만, 효과 해석은 우리 구현에서 원인을 먼저 찾는다, 기여 절은 이름을 바꿔 아래로) — 결과·판정 불변
- README(영·한): 상태 상자에서 st_official 문장 삭제. "Who did what" → "Contributions and AI use"(인용 절 앞으로, "방법의 일부인 LLM" 줄 추가; 근거 관행은 SPEC_LEDGER AI 사용 공개 관행(X-31)). 채점 설명을 쉬운 말로(LLM이 뽑은 답이 응답에 있을 때만 인정, "grounded" 표현 삭제). 요약·분류 정의·시점 문단 정리(Holm 문장은 유지: 미리 정한 보정이라 빼면 유리한 값만 보고하는 셈).
- 새 절 "정확도가 논문보다 8–12 pp 높은 이유(미해결)": 공식 1B 59.3·8B 78.4와 비교하면 논문 ST가 10 pp 넘게 낮다는 점과 추정 원인 3개(끝나지 않은 답을 오답 처리 = 읽기 (c) ST 44.4 / TSR 48.3 / VisRef 47.4, 결과 후 가설; 모델 세대 InternVL3-1B 45.8; 토큰 상한·추출·프롬프트·하드웨어). 어느 것도 직접 시험하지 않음.
- 새 절 "VisRef 추가 이득이 없었던 이유": 우리 구현에서 먼저 찾는다는 문장 + 우리 쪽 후보 5개와 확인 여부(주입 수 ≈ m/5 → fill로 일부 확인; 주입 형식 → 미시험, k=1 713문항 중 `</think>` 재출력 VisRef 352 vs ST 67; 상한 절단 뒤 주입 → 4,096 미실행; Z 표현 → 은닉 상태 오프라인 확인 12문항 50–110개; 단일 표본 H → 다중 표본 미실행). "잘못했다"고 단정하지 않음(작성자 결정).
- "Pre-registered follow-ups" → "Follow-up experiments (decision rules fixed before the results)". 재현 §3에 MathVista 공식 추출 프롬프트 설명(예시 5개 + 새 응답, GPT-4 추출기; MathVista 추출기(X-27), 공식 추출 프롬프트(X-28)). "Known differences" → "Settings we could not find in the paper": 13행 표(우리 선택·근거·다른 가능성과 상태) + H 문단. "명시하지 않음/leaves open" → "찾지 못함".
- H 문단은 기록된 사실만: 이전 다중 표본(단계마다 완결 답 10개)은 8B·개발 3문항·TSR에 GPU 4,703.9초([초기 기록 "R25 수집·채점 완료 — 자동 마무리 기록" 항목](https://github.com/AhnJG/visref-repro/blob/4776f6ab2607957799bef8db70241cc64313feae/docs/EXPERIMENT_FLOW.md?plain=1#L642-L644), 2026-10-11 "정정: Ghosal식 다중 표본 H 그림자 기록 제안 철회" 항목), 이후 단일 표본으로 교체(10-06·10-07 항목). 바꾼 이유가 비용이라는 문장은 기록에 없어 쓰지 않음. REPORT §4.1-5의 "judged too costly"도 같은 사실 서술로 고침(결과 후 문구 변경, 이전 문장은 커밋 adbafc3의 REPORT.md).
- REPORT: 요약 마지막 문장("We do not conclude…" → 우리 구현에서 원인을 먼저 찾는다), 그림 1 캡션·§2.1 표·§5.2의 "not specified/leaves open" → "could not find in the paper", §6 후보에 모델 세대 추가. st_official(§4.5) 절은 재현 보고서 형식이라 유지(작성자 결정).
- 그림 `figures/method_overview.svg`: 범례·설명 문구 3곳만 "not found in the paper"로(배치 변화 없음).
- 한국어판: 번역 후 숫자·링크·용어 확인, 어색한 문장 4곳 수정(용어: LLM 보조 추출, 답 강제, 후속 실험, 판정 규칙, 논문에서 찾지 못한 설정).
- 새로 확인한 외부 수치·근거: SPEC_LEDGER X-26~X-31.
- 검증: `make check` Python 3.11·3.12·3.13 통과(README 숫자 검사 포함), GitHub 렌더링(저장소 API)에서 두 README 모두 표 6개·행 34개, 내부 링크 앵커 일치.
- push: `main` = 0a8241d. 공개 폴더의 README.md·README.ko.md·REPORT.md·method_overview.svg·docs/SPEC_LEDGER.md를 같은 내용으로 맞춤.

## 2026-10-12 § 표기 변경, InternVL3-1B 후보 삭제 (작성자 결정) — 결과·판정 불변
- README(영·한): "REPORT §4.4" → "[REPORT, Section 4.4]" / "[REPORT 4.4절]"로 바꾸고 REPORT 해당 제목으로 가는 링크를 걸었다(각 9개). README 안의 "재현 방법 §3"은 LLM 보조 추출 단계 링크로. REPORT의 § 10개 → "Section".
- README(영·한): 절대 정확도 차이의 추정 원인에서 "다른 모델 세대(InternVL3-1B)" 항목과 설정 표 모델 행의 대안 칸을 삭제(작성자 결정). REPORT §6 후보 목록에 직전에 넣었던 "the model generation"도 되돌림. REPORT 1절 범위 설명(세대 가정)과 SPEC_LEDGER 행 1·1B 정체(C-8)·공식 3-1B 점수(X-26)의 기록은 그대로 둠.
- 검증: `make check` Python 3.11·3.13 통과. GitHub 렌더링(저장소 API)에서 두 README의 REPORT 링크 9개씩 모두 REPORT 제목 id와 일치, README 내부 링크 일치, 표 6개, § 0개.
- push: `main` = 7fc9d64. 공개 폴더 README.md·README.ko.md·REPORT.md 같은 내용으로 맞춤.

## 2026-10-12 REPORT에서 InternVL3-1B 후보 삭제 (작성자 판단) — 결과·판정 불변
- 판단: 논문은 InternVL3.5-8B 위주이고 1B 행만 3 세대로 보는 것은 논리적으로 맞지 않는다. 원문 대조: 논문이 이름을 밝힌 InternVL 세대는 3.5뿐(LAY에 "InternVL3-" 0회, "InternVL-3.5"/"InternVL3.5" 다수), 모델을 "reasoning (thinking) mode"로 실행(LAY L428-429), 1B·2B·8B를 "under the same decoding setup and token budget"으로 비교(L924-925), 8B 행 68.1/73.9/79.3 = Table 1 InternVL3.5-8B(L396-398). InternVL3 후보의 근거는 1B 수치 근접(45.8 vs 46.1) 하나뿐이었다.
- REPORT: 1절 범위 설명을 위 근거로 InternVL3.5-1B를 쓴다는 문장으로 교체, 2.1절 표 "generation assumed" 삭제, 5.2절 "어느 InternVL 세대인지" 항목 삭제. README는 직전 커밋에서 이미 삭제.
- SPEC_LEDGER 공식 3-1B 점수(X-26)에 삭제 사유 메모 추가(행 1·1B 정체(C-8)의 기록은 그대로).

## 2026-10-12 H 추정 경위 정정, 기여 절 수정 (작성자 정정) — 결과·판정 불변
- 기록 대조: 10-05 v3 명세는 "1회 추정"(이 파일 2026-10-05 independent_dpp30_entropy_v3 항목), r3은 답 1개의 답 토큰 평균 엔트로피 → 10-05 작성자가 준 [19] Ghosal 근거 검토(`docs/paper_baseline_alignment/ENTROPY_REFERENCE_CLAIM_REVIEW_KO.md`: 여러 완결 응답의 sequence NLL 평균 = Monte-Carlo estimator) → 10-06 S10(단계마다 완결 응답 10개) 구현·8B 실행(R25 GPU 4,703.9초) → 10-06 작성자 지시로 자연 응답 엔트로피(1개)로 전환 → 10-07 짧은 답 1개 토큰 엔트로피 합. README·README.ko의 H 문단과 REPORT 4.1절 5번을 이 순서로 고침(이전 문장 "earlier version sampled 10 … replaced by the one-sample estimator"는 커밋 eae46a1). 바꾼 이유는 여전히 기록에 없어 쓰지 않음.
- 기여 절: "모든 GPU 세션과 Gemini 추출을 실행" 문구 삭제(작성자 판단: 중요하지 않음), "독립 검토: 프로젝트 전 과정에서 내용 검토에 사용" 추가(작성자 진술).
- push: `main` = 8b27721. 공개 폴더 README.md·README.ko.md·REPORT.md 같은 내용으로 맞춤.

## 2026-10-12 README·REPORT 설명 보강, st_official 삭제 (작성자 결정) — 결과·판정 불변
- H를 답 1개로 되돌린 이유(작성자 진술): 논문 Table 5에서 문항당 시간이 ST 7.1초, TSR 7.7초, VisRef 8.2초(8B, H100 1장)라 단계마다 답을 여러 개 생성했다면 나올 수 없는 수치(SPEC_LEDGER Table 5 문항당 지연(X-32)). README·REPORT 4.1절 5번에 반영. README H 문단은 "모델이 그 자리에서 쓴 답(정답 아님), 토큰별 엔트로피를 자연로그로 계산해 합" + 경위(평균 → S10 시도 → 답 1개 → 합)로 풀어 씀.
- visref_fill 서술 정정: 채운 부분은 관련성과 무관하게 골라졌다(꼬리 순서와 관련성 순위 상관 −0.10, 2026-10-11 fill 결과 항목). 논문 Table 7에서도 무작위 선택은 ST 수준(Table 7 선택 방식(X-36)). README·REPORT에서 "m개를 넣어도 이득 없음"이 아니라 "관련성과 무관한 토큰으로 m개까지 채운 변형은 이득 없음, 관련 토큰 m개는 시험하지 않음"으로 고침. REPORT 4.2절 제목도 "topping up … to m tokens".
- README 설정 표: 모델 행 삭제(작성자), 사고 모드·샘플링(논문은 사고 모드 실행만 명시, 켜는 법·샘플링 값 없음 s1과 샘플링 미기재(X-37)), 모델 카드 설명·링크, s1 풀어 쓰기, `Final Answer:` 진단 뜻 풀기, DEER를 쓴 이유(답이 "Wait, wait …"로 끝나지 않아 상한까지 감 — 10-07 결정 항목), 답 추출 문구 정정(MathVista도 아주 단순한 경우만 규칙, 나머지 LLM; VLMEvalKit 규칙 → GPT, 공식 추출 순서(X-33)·VLMEvalKit 추출(X-34)), Z 쉬운 설명, 주입 형식에 InternVL 원래 `<img>` … `</img>` 태그(이미지 입력 형식(X-35)), 타일링 설명·근거(이미지 입력 형식(X-35)), nat → "자연로그".
- 절대 정확도 차이 원인에서 하드웨어 삭제(같은 GPU fp16 잡음으로 바뀐 답 2.2%, REPORT 4.1절), 결과를 본 뒤 세운 가설 문장은 한 구절로 줄임(작성자 결정). README에서 "읽기 (a)/(c)"를 설명형 이름으로(REPORT는 표로 정의된 (a)~(f) 유지).
- st_official(작성자 결정: 글 + 전용 파일 삭제): REPORT 4.5절과 위치 표 언급 삭제, 공개 저장소에서 `repro/diag/analyze_st_official.py`, `repro/colab/runs/st_official_c1/run_config.json` 삭제. export 스크립트가 다시 복사하지 않도록 수정(이전 판 `scripts/.make_github_export.bak_20261012c`). 이 실험 기록·`repro/audit/`의 검토 기록 안 언급은 이력이라 유지. Colab 빌더 `repro/colab/build_colab.py`는 일반 도구라 유지.
- push: `main` = 0b508f3.

## 2026-10-12 채점 방식 이름 풀어 쓰기, README 문장 정정 3건 (작성자 결정) — 결과·판정 불변
- 결정: 기호 대신 설명형 이름을 전체 문서에. REPORT·results/ 라벨·결과 그림 부제에서 "reading (a)~(f)"를 "main scoring", "rules only", "scored before answer forcing", "unfinished answers counted wrong", "answers forced after cut thinking counted wrong" 등으로 바꿈(`scripts/make_results.py`, `scripts/make_figures.py` 수정 후 재생성). 이 실험 기록과 SPEC_LEDGER의 "읽기 (a)" 등은 당시 기록이라 그대로 둠.
- 정정 1(주입 형식): README가 "태그 붙인 주입은 시험하지 않음"이라고 썼으나, 2026-10-07 smoke2(방법당 약 21문항, 다른 설정)에서 `<img>…</img>` 태그 주입은 두 번째 `</think>`가 63단계 중 31단계, 태그 없음은 65단계 중 10단계라 태그를 뺐다(이 파일 2026-10-07 독립 검토·smoke2 항목). README에 이 사실을 넣음.
- 정정 2(`Final Answer:` 진단): "이 문구가 논문과의 차이를 만든 것은 아님"은 과한 표현. 150문항, 불일치 15:18이라 95% 범위 약 −9.5 ~ +5.5 pp(정규 근사)로 몇 pp 크기의 효과는 가려낼 수 없다. README·REPORT 4.1절 6번 고침. 이 문구는 잘린 생각에서만 쓰이고 ST가 TSR보다 더 자주 쓰므로 방법 간 차이에도 영향을 줄 수 있다.
- 정정 3(사고 모드): 논문에 사고 모드 사용이 있으므로(`LAY` L428-429) "찾지 못한 설정"에서 빼고 샘플링 값만 남김. 모델 카드는 사고 모드에서 반복을 막으려면 T 0.6 샘플링을 권장(`internvl_README.md` L412).
- 보강: Z 행에 논문 표현 "text token embeddings"(`LAY` L304-305)와 "더 깊은 층이면 DPP가 m개를 돌려줄 수 있음"(2026-10-08 hidden_rank_check) 추가. 참고: 2026-10-06 작성자는 last hidden state를 우선 후보로 정했으나(이 파일 2026-10-06 항목), 10-07 간소화 재작성에서 입력 임베딩으로 구현했고 hidden-state 변형은 오프라인 확인만 했다. H 문단에 "최종 답 전 각 단계에서 재야 하므로 그 시점에 답을 써 보게 하고, 엔트로피는 가능한 답들의 평균이라 T = 1로 샘플링"을 추가.
- push: `main` = 6e95c3e.

## 2026-10-12 후속 실험 3 `visref_hs` 실행 전 계획 (작성자 결정) — 결과 전에 고정
- 질문: Z(생각 단계 토큰 벡터)를 입력 임베딩 대신 **마지막 층 hidden state(최종 정규화 출력, lm_head 직전)**로 바꾸면 VisRef가 TSR보다 나아지는가. 근거: 논문 "text token embeddings"는 층 미지정(`LAY` L304-305), 논문은 |V_k| = m(선택 수 강제(X-23))인데 입력 임베딩으로는 서로 다른 단어 수가 상한이라 m의 약 1/5만 주입됨. 2026-10-06 작성자가 last hidden state를 우선 후보로 정했으나 10-07 간소화 재작성에서 입력 임베딩으로 구현됨(경위 기록 미확인). hidden state는 위치마다 벡터가 달라 단계가 m보다 길면 m개까지 고를 수 있음(2026-10-08 hidden_rank_check 오프라인 확인).
- 바꾸는 것은 Z 하나: `visref_hs` = `visref_raw_rank`와 같은 DPP(Chen greedy, 1e-9·max L_ii 정지), V는 그대로 입력 공간 시각 임베딩, 주입 형식·정지 probe·답 처리·토큰 상한(단계 1,024 / 합계 10,240)·시드·문항 순서(seed 0 셔플) 모두 본 실험과 같음. Z는 맥락(지금까지의 문맥 + z_k)을 한 번 더 prefill해 z_k 위치의 last_hidden_state를 취함(`Engine.last_hidden`, causal이라 뒤에 붙는 V_k·구분자 영향 없음). 이중 probe는 끔(판정과 무관한 기록용이라 시간 절약).
- 코드: `visref/engine.py` `last_hidden`, `visref/dpp.py` `select_tokens(trace=)`·`pick_stats`(선택 결과는 trace 유무와 같음, 테스트), `visref/methods.py` `visref_hs`. CPU 테스트 104 passed(신규 `tests/test_hs.py` 6개: 전체 forward와 일치(분할 prefill 포함), trace가 선택을 바꾸지 않음, 반복 토큰에서도 m 도달, job이 last_hidden으로 Z를 받음).
- 실행: Kaggle 세션 1개(2×T4), timeout 29,000s(time_limit 26,300s, deadline 28,400s, 여유 약 33분). 1,000문항을 seed 0 셔플 순서로 투입, 끝난 앞부분(complete prefix)만 분석. 예상 약 550–650문항(fill 속도 약 88문항/h 기준 추정, 미검증).
- 비교 상대: 본 실험(b_main)의 같은 문항 ST·TSR·VisRef(`visref_raw_rank`) 행. 같은 문항별 시드(CRN)라 첫 단계는 fp16 배치 잡음 범위에서 같음(fill에서 첫 단계 종료 유형 874/976 일치).
- 채점: 본 채점과 같은 절차(규칙 v4 → 규칙 실패 행만 공식 MathVista 추출 프롬프트 + gemini-3.5-flash-lite, 응답에 있는 답만 인정, 빈 응답은 오답). LLM 추출 전 규칙만 결과도 함께 보고.
- 주 비교(하나, 다중성 보정 없음): hs − TSR, 짝 bootstrap 95% CI(2,000회, seed 0), McNemar 정확 검정. 판정:
  ① CI 하한 > 0 → "마지막 층 Z에서는 VisRef가 TSR보다 추가 이득" (Z 표현이 이득 부재의 원인이었을 가능성 지지).
  ② CI가 0을 포함하고 상한 ≥ +3.5 → "판단 보류"(논문 크기 효과를 배제하지 못함).
  ③ CI 상한 < +3.5 → "마지막 층 Z로도 논문 크기(+3.5 pp)의 추가 이득은 재현 안 됨"(②와 ③이 동시에 해당하지 않음; ①이면서 상한 < 3.5면 ①과 "논문보다 작음"을 함께 적음).
- 보조(기술 통계, 판정 아님): hs − 본 실험 VisRef(Z만 다름), hs − ST의 차이·CI; 주입 수/m(단계 길이 ≥ m인 단계에서 1.0 기대), n_meaningful(정규화 잔차 ≥ 1e-2인 선택 수) 분포, 뒤쪽 선택 순서와 관련성 순위 상관, 최종 답 안 두 번째 `</think>` 비율, k 분포·정지 사유, 첫 단계 종료 유형 일치율.
- 검출력(근사): 본 실험 VisRef−TSR 불일치율 약 9%(44+45/1000) 기준, 600문항이면 차이의 표준오차 약 1.2 pp → 80% 검출력 최소 효과 약 3.4 pp. 300문항 미만으로 끝나면 "검출력 부족"으로 보고하고 판정은 같은 규칙으로 하되 해석을 제한.
- 이 실험은 결과 후 추가 실험이다. 본 실험 판정·기존 후속 실험 결과는 바꾸지 않고 별도 절로 보고한다.

## 2026-10-12 AI 활용 반성 기록 작성, "모델 카드" 표현 교체 (작성자 결정)
- 작성자 결정: AI 활용 오류로 방향을 잘못 잡거나 실험을 실수한 기록을 모아 반성 사항으로 정리. EF·SPEC_LEDGER·AUDIT·`repro/audit/`·REPORT를 훑어 후보 74건을 찾고, 문서에 쓴 사례의 근거 줄 57곳을 원문과 대조(모두 일치). 결과는 반성 기록 문서(9개 주제, 남은 불일치, 앞으로의 규칙 9개).
- 대조 중 확인한 남은 불일치: SPEC_LEDGER 37·44행 상태 열과 요약 집계(L148, L160, L183)가 감사 후 메모(L271, L274)와 달리 여전히 "인용 구현에서 확인". 아직 고치지 않음.
- README·README.ko·REPORT·docs/SOURCES.md의 "model card / 모델 카드"를 "official model page / 공식 모델 페이지"로 바꿈(`main` = 9f920d1). 이 기록과 SPEC_LEDGER의 옛 표현은 당시 기록이라 그대로 둠.

## 2026-10-12 `visref_hs` 실행 시작, AI 활용 반성 기록 공개 (작성자)
- `b_main_hs`: 작성자가 Kaggle push. 시작 20–30분 뒤 로그에 error 없음(작성자 확인). 결과·판정은 세션 종료 후 사전등록 규칙대로.
- 반성 기록에 "재발을 막는 방법"(결정 대장과 결정별 테스트, 가정 대장, 문장 출처 규칙과 다른 모델 검토, 해석 문구 사전 지정, 자율 실행 계약, 실행 위생, 위임 결과 재계산, 실측 기반 시간 추정, 쉬운 말, 작성자 질문 5개, 적용함/아직 안 함)을 추가하고, 공개 저장소 `docs/AI_USE_REFLECTION.md`에 넣음(공개용 머리말 추가). README 두 개의 파일 목록, REPORT 7절 표와 Process note(독립 내용 검토 사용 포함)에 링크.

## 2026-10-12 `visref_hs` 분석 스크립트 작성 (결과 전)
- `repro/diag/analyze_hs.py`: 위 사전등록 규칙을 그대로 코드로 옮김. 표본은 seed 0 순서의 complete prefix(2 shard), 비교 행은 b_main ST·TSR·VisRef(`llm/final/merged.jsonl`). 주 채점 = b_main 기록값 + `visref_hs` 행(현재 규칙 채점 v4로 재채점 → LLM 추출 merge). 규칙만 결과 = 네 방법의 모든 행을 현재 규칙 채점(v4)으로 다시 채점. 주 비교 hs − TSR: 짝 bootstrap 95% CI(2,000회, seed 0, `summarize.paired`와 같은 재표본)와 McNemar 정확 검정. 판정 ①②③은 CI 경계를 차이의 정수 합으로 비교한다(상한이 정확히 +3.5일 때 소수 반올림으로 판정이 바뀌지 않게). 300문항 미만이면 "검출력 부족" 표시. 보조(판정 아님): hs − VisRef, hs − ST, 주입 수/m(단계 길이 ≥ m인 단계 가운데 n_inject = m인 수 포함, 본 실험 VisRef 값과 나란히), n_meaningful, 뒤쪽 선택과 관련성의 순위 상관, 최종 답 안 두 번째 `</think>`, k 분포·정지 사유, 첫 단계 종료 유형 일치, 답 강제 수. LLM 추출 전에는 규칙만 결과를 "잠정"으로 표시.
- 결과 후 순서(repro/에서): `python3 diag/analyze_hs.py rows --out llm/hs/rows_v4.jsonl` → `python3 llm_extract.py prepare llm/hs/rows_v4.jsonl --control 20 --out llm/hs/requests.jsonl` → (작성자) Gemini 추출(`gemini_extract.py`, gemini-3.5-flash-lite) → `python3 llm_extract.py merge llm/hs/rows_v4.jsonl --responses llm/hs/responses.jsonl --out llm/hs/merged.jsonl` → `python3 diag/analyze_hs.py analyze --llm llm/hs/merged.jsonl --out diag/analyze_hs.json`.
- 시험(대역 = fill 결과, `--hs-dir kaggle/runs/b_main_fill/output/results --method visref_fill --llm llm/fill/merged.jsonl`): `diag/analyze_fill.json`과 같은 값(976문항, ST 57.79 / TSR 60.04 / VisRef 59.94 / fill 60.04, fill − TSR 0.0 (−2.15, +2.36) 70:70, fill − VisRef +0.1 (−2.15, +2.36) 65:64, fill − ST +2.25 (−0.31, +4.71)). rows → prepare 경로는 fill의 원래 요청과 같은 규칙 실패 23행을 고르고, 기존 응답으로 merge해도 같은 값. prefix 안 b_main 행 가운데 현재 채점(v4)이 정오를 바꾸는 행 0.
- CPU 테스트 `repro/tests/test_analyze_hs.py` 12개(판정 경계, +3.5 경계의 정수 비교, `summarize.paired`와 같은 재표본, complete prefix, 주 채점이 LLM 행을 쓰는지, LLM 행 누락 시 오류, 재채점이 기록값을 덮어쓰는지). 전체 116 passed(python3.13), `make check` 통과. 실행 중인 번들의 engine/dpp/methods sha256 앞 16자리가 저장소와 같음(3b2fdecbe14c1ff6 / 7ebbe7774ac757cc / 14534e592396621a).

## 2026-10-12 결정 점검: 작성자 결정 ↔ 코드
- 방법: [초기 기록 2026-10-06 "자연 응답 entropy 전환 — 로컬 구현 진행" 항목](https://github.com/AhnJG/visref-repro/blob/4776f6ab2607957799bef8db70241cc64313feae/docs/EXPERIMENT_FLOW.md?plain=1#L647-L650)부터 이 문서 끝까지(2026-10-06 이후)과 [초기 기록 "작성자 결정: Z 우선 후보 last hidden state로 동결" 항목](https://github.com/AhnJG/visref-repro/blob/4776f6ab2607957799bef8db70241cc64313feae/docs/EXPERIMENT_FLOW.md?plain=1#L501-L503)에서 코드 동작을 정하는 작성자 결정·지시를 뽑아 코드 줄과 맞춰 봄(56건; 반영 43, 뒤 작성자 결정으로 대체 9, 불일치 2, 찾지 못함 2). SPEC_LEDGER 결정 대장에 연결한 EF 줄과 코드 줄을 원문에서 다시 확인. 원본 표는 작업 폴더 `repro/audit/decision_check_20261012/raw.md`(공개 안 함).
- 불일치 1, Z: 10-06 작성자 결정(Z에 입력 임베딩이 아닌 last hidden state가 타당, [초기 기록 "작성자 결정: Z 우선 후보 last hidden state로 동결" 항목](https://github.com/AhnJG/visref-repro/blob/4776f6ab2607957799bef8db70241cc64313feae/docs/EXPERIMENT_FLOW.md?plain=1#L501-L503))이 10-07 재작성에서 빠짐(경위 기록 없음). 이번 점검에서 찾은 빠진 작성자 결정은 이 하나이고, 영향은 실행 중인 `visref_hs` 결과로 확인한다.
- 불일치 2, 토큰 상한 기본값: `run.py` DEFAULTS가 옛 결정(2026-10-07 "smoke2 결과와 본 실험 결정" 항목)의 단계 4,096 / 합계 8,192로 남아 있음. 결정(2026-10-07 "결정(작성자): s1 + DEER af 폴백, 정지 통계 현행 유지, 파일럿 없이 본 실험 시작" 항목)은 1,024 / 10,240이고 모든 생성 실행(b_main s1–s9, fill, `visref_hs`)이 run_args로 1,024 / 10,240을 넘겨 결과에는 영향 없음(run_config로 확인). 인자 없이 `run.py`를 직접 돌릴 때만 옛 값. 기본값을 바꾸면 code_version이 바뀌므로 고치지 않고 작성자 결정으로 넘김.
- 찾지 못함 2건: `visref_hs` 실행 설정은 작업 폴더 run_config(`--think-cap 1024 --total-think-cap 10240`, 이중 probe 인자 없음 = 끔, timeout 29,000 / time_limit 26,300 / deadline 28,400, shuffle_seed 0)로 확인해 반영으로 봄. fill 판정 규칙(2026-10-10 "원인 분리 실험 실행 전 계획" 항목)은 코드에 없고 사람이 적용(2026-10-11 "visref_fill 최종" 항목, CI 하한 −2.15 < 0 → 지지 안 됨, 규칙과 맞음).
- 기록: SPEC_LEDGER "핵심 판단 요약"에 J-19~J-22(문항 순서와 분석 표본, 정지 판정용 답과 최종 답 분리, LLM 추출 가드, 판정 문구 규칙) 추가, J-1~J-22의 결정 주체·코드 위치·테스트를 "결정 대장" 표로 추가, 답 추출(J-14) 채점 규칙을 v4로 고침(이전 판 `repro/.backup_decision_check_20261012/SPEC_LEDGER.md`).
- 테스트: 결정된 값이 바뀌면 결과가 달라지는데 기존 테스트(자체 작은 설정 사용)가 값을 고정하지 않아 `repro/tests/test_decisions.py` 추가(27개: run.py 기본값 15개, probe 샘플링·m, 채점 v4, b_main 실행 10개의 상한·순서, LLM 가드 2와 대상 행 고르기). 전체 143 passed(python3.13).
- 옛 표기 정정(출력 불변): `diag/analyze_fill.py` 설명과 LLM 없을 때의 라벨, `diag/analyze_st_plain.py` 설명의 "v3"를 "작성 당시 v3, 2026-10-11부터 v4"로.

## 2026-10-12 반성 기록에 "결정이 바뀌었는데 코드 기본값은 옛 값" 사례 추가 (작성자 결정)
- 실행 묶음에서 직접 확인한 경위: 각 실행 노트북에 묶인 `run.py`의 생각 상한 기본값은 smoke·smoke2 단계 8,192 / 합계 16,384, full_s1·smoke3·main_s0·main_s1·b_test 이후 모든 실행 4,096 / 8,192(2026-10-07 "smoke2 결과와 본 실험 결정" 항목 결정과 같음). 1,024 / 10,240은 b_test 변형(2026-10-07 "b_test: 단계 1024 상한 변형 + 강제 답변·CRN" 항목)부터 run_args로만 넘겼고, 본 실험 설정이 된 뒤(2026-10-07 "결정(작성자): s1 + DEER af 폴백, 정지 통계 현행 유지, 파일럿 없이 본 실험 시작" 항목)에도 기본값을 옮기지 않음. 실제 적용값은 b_main s1–s9·fill(두 shard)과 diag_st_plain 시작 로그의 cfg에서 모두 1,024 / 10,240, `visref_hs`는 run_config run_args에 같은 값.
- `docs/AI_USE_REFLECTION.md`에 반영: 1절에 사례와 반성 문장, "지금 남아 있는 불일치"에 기본값 정정(작성자 결정 대기)과 SPEC_LEDGER 줄 번호 갱신(결정 대장 삽입으로 35줄 밀림: L183, L195, L218, 메모 L306, L309), "재발을 막는 방법" 1에 결정 값은 한 곳에만, 변형이 본 설정이 되면 기본값으로 옮기기, 결정값 테스트(적용함), 묶음 빌드 때 실제 설정 대조(아직 안 함), 질문 3에 "코드(기본값 포함)" 추가, 적용함/아직 안 함 갱신.

## 2026-10-13 `visref_hs` 실행 설정 공개, README·REPORT를 "실행 중"으로 갱신
- 공개 저장소에 `repro/kaggle/runs/b_main_hs/run_config.json` 추가(작업 폴더 원본 그대로, sha256 앞 16자리 17c7ea3740ff01da; 노트북은 다른 실행처럼 제외). 내용: run_args `--methods visref_hs --think-cap 1024 --total-think-cap 10240`, timeout 29,000 / time_limit 26,300 / deadline 28,400, ids 1,000개(b_main_s1·fill과 같은 seed 0 순서, 확인), built_at 2026-10-12T12:12:04 UTC. 묶인 코드 8개 파일의 sha256이 사전등록 커밋 ccac0d9의 `repro/`와 같고 그 뒤 바뀌지 않음(확인). 사전등록 커밋 12:13:12 UTC, Kaggle push 12:16:47 UTC(`push.json`).
- 공개본 생성 스크립트(작업 폴더 `scripts/make_github_export.py`) RUNS에 `b_main_hs` 추가(이전 판 `scripts/.make_github_export.bak_20261013`). 결과가 나오면 output/results도 같은 규칙으로 들어간다.
- README·README.ko: 상태 상자("완료, 실험 종료" → 본 실험과 후속 실험 2개 완료, 세 번째 후속 실험 실행 중), 후속 실험 표에 `visref_hs` 행(질문과 사전 판정 규칙), "VisRef 추가 이득이 없던 이유" 표와 "논문에서 찾지 못한 설정" 표의 Z 행, 파일 목록(analyze_hs.py, b_main_hs는 run_config.json만). REPORT: 첫머리 상태 문장, 2.1절 Z 행, 4.1절 4번, 새 4.5절(질문·설계·상태), 6절 "시도 안 한 아이디어"에서 hidden-state Z를 빼고 실행 중으로, 7절 실행 목록과 후속 분석 목록.
- `tests/test_decisions.py`가 b_main_hs도 검사(생성 실행 11개), SPEC_LEDGER 결정 대장 토큰 상한(J-10) 테스트 칸과 점검 방법 문장 갱신.

## 2026-10-13 `visref_hs` 결과 수신, 규칙 채점, LLM 추출 요청 준비
- 작성자 pull. 실행 상태: 839행(shard 0 411, shard 1 428), 상태 오류 0, code_version f1376774850d(저장소 코드와 같음), 두 shard 시작 로그의 cfg가 상한 1,024 / 10,240, 이중 probe 0. 두 shard 모두 정상 종료(end 이벤트, 경과 27,911s / 27,520s, 대기열에 남은 문항 89 / 72). complete prefix 839문항(300 이상이라 "검출력 부족" 표시 없음), 문항 순서는 fill과 같음. prefix 안 b_main 행 가운데 현재 채점(v4)이 정오를 바꾸는 행 0.
- 규칙 재채점(v4) `repro/llm/hs/rows_v4.jsonl`(839행, 정답 504) → `llm_extract.py prepare --control 20` → `repro/llm/hs/requests.jsonl` 43건(규칙 실패 23 + 대조군 20, 빈 텍스트 4행은 보내지 않음, 정답 미포함). 작성자 Gemini 실행(gemini-3.5-flash-lite, 사전등록 절차) 대기.
- 규칙만 결과(잠정, 판정 아님. 판정은 LLM merge 뒤 주 채점으로): 839문항, ST 57.57 / TSR 59.71 / VisRef 59.36 / hs 60.07. hs − TSR +0.36 pp (−2.26, +3.22), 69:66, McNemar p 0.863 — 같은 규칙을 적용하면 ③. hs − VisRef +0.72 pp (−1.67, +3.10), 59:53; hs − ST +2.50 pp (−0.12, +5.13), 74:53.
- 동작 확인(채점과 무관): 주입 수/m 사분위 0.977 / 1.0 / 1.0(같은 문항 본 실험 VisRef 0.142 / 0.205 / 0.613). 단계 길이 ≥ m인 1,678단계 중 1,130단계가 정확히 m개. n_meaningful 사분위 54 / 70 / 89(m에 대한 비율 0.078 / 0.136 / 0.304). 뒤쪽 선택과 관련성의 순위 상관 사분위 −0.224 / −0.171 / −0.105. 최종 답 안 두 번째 `</think>`: hs 395, VisRef 392, ST 109, TSR 99(각 839문항 중). 첫 단계 종료 유형 일치 763/839. 평균 k: hs 2.34, VisRef 2.27, TSR 2.10. 정지 사유 hs entropy 775 / kmax 64.
- 공개 저장소 반영: 실행 출력(`repro/kaggle/runs/b_main_hs/output/results/`)과 `repro/llm/hs/requests.jsonl` 추가, 재채점 사본 `llm/*/rows_v*.jsonl`은 공개하지 않음(.gitignore, 공개본 생성 규칙과 같음). README·README.ko·REPORT의 "실행 중"을 "실행 마침, 분석 중(LLM 보조 추출 대기)"으로, REPORT 4.5절 제목에서 "running"을 빼고 README 링크를 맞춤. 공개본 생성 스크립트 CODE에 `llm/hs/requests.jsonl` 추가(응답·merge 파일은 생기면 추가).

## 2026-10-13 `visref_hs` 최종: LLM 보조 추출(Claude Haiku), 판정 ③
- 추출: fill 때(2026-10-11 "visref_fill 최종" 항목)처럼 같은 공식 추출 프롬프트로 Claude Haiku가 43건을 추출(정답 미포함, 대조군 표시 없이 id와 prompt만 전달). 사전등록(gemini-3.5-flash-lite)과 다른 추출기 — 벗어난 것으로 기록. 규칙 실패 23건의 추출을 응답 끝부분과 하나씩 대조해, 21건은 응답에 있는 답이고 2건(159, 943)은 응답에 최종 답이 없어 빈 값임을 확인. 응답 `repro/llm/hs/responses.jsonl`(model 필드 기록).
- merge: 규칙 실패 23 중 사용 21, 빈 추출 2(규칙 결과 유지), 근거 없음 거부 0, 대조군 20/20 일치·근거 있음, 정오 변화 fixed 0 / broke 1(pid 722: 규칙은 'yes'를 뽑았으나 응답 끝은 "The correct answer should be B) no"). 결과 `repro/llm/hs/merged.jsonl`.
- 판정(`diag/analyze_hs.py analyze --llm llm/hs/merged.jsonl`, 839문항, 주 채점): ST 57.57 / TSR 59.71 / VisRef 59.36 / hs 59.95. hs − TSR +0.24 pp (−2.38, +3.10), 68:66, McNemar p 0.931 → 사전등록 ③ "마지막 층 Z로도 논문 크기(+3.5 pp)의 추가 이득은 재현 안 됨". 보조: hs − VisRef +0.60 pp (−1.67, +2.98), 58:53; hs − ST +2.38 pp (−0.24, +5.01), 73:53. 규칙만: hs − TSR +0.36 pp (−2.26, +3.22), 같은 판정. `make analysis` 재실행 결과 `diag/analyze_hs.json`이 바이트 단위로 같음.
- 반영: `make analysis`에 hs 추가, `scripts/make_results.py`(followups.csv·summary.json에 visref_hs), `scripts/check_readme_numbers.py`(hs 주 비교 숫자 검사), README·README.ko(상태 "완료", 후속 실험 표, Z 관련 두 행, 파일 목록), REPORT(첫머리, 요약, 4.1절 4번, 4.5절 결과·동작·이탈·해석, 6절, 7절), SPEC_LEDGER 결정 대장 Z 표현(J-8)·답 추출(J-14)와 Z 표현(J-8) 본 행, 반성 기록 1절과 남은 불일치. 이 실험은 결과 후 추가 실험이므로 본 실험 판정은 바꾸지 않음.

## 2026-10-13 `run.py` 생각 상한 기본값 정정 (작성자 결정), 마지막 층 Z에서 m에 못 미친 단계 정리
- `repro/run.py` DEFAULTS를 think_cap 4,096 → 1,024, total_think_cap 8,192 → 10,240으로(결정 2026-10-07 "결정(작성자): s1 + DEER af 폴백, 정지 통계 현행 유지, 파일럿 없이 본 실험 시작" 항목). 이전 기본값은 10-07 옛 결정(2026-10-07 "smoke2 결과와 본 실험 결정" 항목) 값으로, 모든 생성 실행은 인자로 1,024 / 10,240을 넘겨 결과 영향 없음(10-12 결정 점검). code_version f1376774850d → cf46b2e6222e(`visref_hs` 행 839개는 모두 f1376774850d, 실행이 끝난 뒤 변경). `tests/test_decisions.py` DECIDED에 상한 두 값 추가(기본값 테스트), 실행 설정 테스트는 유지. CPU 테스트 145 passed. SPEC_LEDGER 결정 대장 토큰 상한(J-10)·점검 문단, 반성 기록(1절, 남은 불일치, 예방법, 적용함/아직 안 함) 갱신.
- `b_main_hs` 출력에서 확인한 m 랭크 부족 현황: 주입이 있는 1,966단계 중 m개를 정확히 넣은 단계 1,130, 못 미친 단계 836. (1) 단계 길이 < m인 288단계(자연 종료 242, EOS 46): DPP가 생각 토큰 수보다 많이 고를 수 없어 284단계가 정확히 생각 토큰 수만큼 주입, 주입/m 사분위 0.137 / 0.3 / 0.534. (2) 단계 길이 ≥ m인데 못 미친 548단계: 모두 가장 큰 이미지(시각 토큰 3,328개, m = 998). 시각 벡터 rank 사분위 1,007 / 1,009 / 1,010으로 m에 가깝고, greedy가 수치 정지 임계값에서 멈춰 주입/m 사분위 0.968 / 0.982 / 0.992(최소 0.746), m과의 차 중앙값 18. m = 998인 722단계 중 m 도달 38. (3) 다양성을 뚜렷이 더한 선택(정규화 잔차 ≥ 0.01)은 m에 도달한 단계에서도 중앙값 71개. REPORT 4.5절 동작 문단에 m에 못 미친 두 경우를 한 문장으로 추가.

## 2026-10-13 독립 검토와 확인
- 작업 과정을 보지 않은 독립 검토자가 REPORT·README·EF·SPEC_LEDGER·코드·원자료·논문 텍스트(VisRef_layout/plain, InternVL3.5, Ghosal, MathVista)를 읽기 전용으로 검토. 원문 `repro/audit/independent_review_20261013/review.md`, 재계산 `recompute.py`(repro/에서 실행, 저장소에 쓰지 않음).
- 검토 결론: 높음 0, 중간 5, 낮음 13. 본 실험 세 비교, visref_fill, step1_branch, visref_hs의 판정은 각 사전 규칙대로이고 문서·results·diag 숫자는 원자료 재계산과 일치.
- 확인(원자료·원문으로 직접): 중간 1 — `b_main_hs` 첫 단계 텍스트가 본 실험 ST 첫 단계와 같은 문항 193/839, fill 242/994(본 실험 TSR·VisRef는 ST와 1,000/1,000 같음). hs − TSR 불일치 134/839 = 16.0%(본 실험 VisRef − TSR 89/1,000), SE 1.38 pp, 80% 검출 최소 효과 3.86 pp, +3.5 pp 검출력 0.72. 사전등록 2026-10-12 "후속 실험 3 `visref_hs` 실행 전 계획" 항목의 "첫 단계는 fp16 배치 잡음 범위에서 같음"과 REPORT 4.5절 "Only Z changes"는 텍스트 기준으로 맞지 않음(종료 유형 일치 763/839만 봄). 중간 2 — REPORT 4.1절 5번·README H 문단의 Table 5 관련 문장은 작성자 판단(2026-10-12 "README·REPORT 설명 보강, st_official 삭제" 항목)을 사실처럼 쓴 문장(Table 5 값 LAY L892-895 확인). 중간 3 — README의 fill 해석이 Table 7(무작위 67.3, LAY L977-979)만 인용하고 Table 3 다양성만 77.4(LAY L538)·Table 8 λ=0 71.2(LAY L984)를 뺌. 중간 4 — InternVL3.5 보고서 L558-559 "unless otherwise specified, the experimental results reported in Section 3 are obtained without applying TTS"와 L562 Deep Thinking 정의로, REPORT 1절 "does not say whether that number uses thinking mode"는 원문보다 약함. 중간 5 — hs 추출기가 계획과 다름(Haiku)은 규칙만 잠정 결과를 본 뒤였고, 비교 상대 행은 Gemini 추출이라는 점이 REPORT에 없음. 낮음: REPORT 2.4절 "two follow-ups" 낡음; REPORT 4.5절 "548 steps … m = 998"은 544(나머지 4는 m 844), "38 of 722"의 분모는 길이 ≥ m인 582; `pick_stats`·`select_tokens_fill`의 Spearman은 "음수 = 관련성 큰 것이 먼저"(dpp.py L58-59, L107-108)라 4.5절 "anti-related to relevance"는 반대로 쓴 것; SPEC_LEDGER 11·12행이 옛 run.py 기본값; REPORT 5.1절 "Algorithm 1 fixes … cue"(Alg. 1에 문구 없음, LAY L303-330); 2026-10-11 "step1_branch 본 실행 결과" 항목 "309문항뿐"은 틀림(나머지 691문항에도 VisRef만 14 : TSR만 13); README "make check passes on 3.10"은 3.10에서 검사 스크립트만 확인(2026-10-12 "README.ko.md 전체 번역 + README 숫자 자동 대조" 항목); 날짜·"pre-registered analyses" 표기 등. 낮음 11(Chen ε 절대값 vs 우리 상대값), 12(V·Z 공간 전제)는 검토자도 미확인.
- 아직 고치지 않음: 문서 정정 범위와 표현(특히 중간 2는 작성자 판단 문장, 중간 4는 해석 문제)은 작성자 결정 후 "결과 후 변경"으로 기록하며 진행.

## 2026-10-13 검토 반영 1차(Table 5 문장), 다음 실험 세 개 계획 공개 (작성자 결정)
- 원문 확인: `InternVL3.5.txt` 2.4절 Test-Time Scaling(L553-569) — L556-558 TTS는 "reasoning depth (i.e., deep thinking) and breadth (i.e., parallel thinking)"를 함께 뜻함, L558-559 "unless otherwise specified, the experimental results reported in Section 3 are obtained without applying TTS", L562 "Deep Thinking. By activating the Thinking mode …". 59.3은 3.2절(L743) Table 3(L775, 캡션 L827)의 InternVL3.5-1B 행이고, 바로 아래 "w/ Parallel Thinking" 행(69.5)만 따로 표시됨. 문자 그대로 읽으면 59.3은 TTS(사고 모드 포함) 없이 얻은 값. 이전 판단(2026-10-10, 2026-10-10 "정정: InternVL3.5-1B MathVista 59.3 / 69.5의 의미" 항목 "행 이름은 parallel thinking만 언급 → 불명확")보다 원문이 더 분명함. 어느 VLMEvalKit 설정으로 얻었는지는 여전히 미확인(공식 1B 점수(X-1)). 문서 정정(검토 중간 4)은 작성자 결정 대기.
- 검토 중간 2 반영(결과 후 표현 변경): REPORT 4.1절 5번, README·README.ko H 문단의 Table 5 관련 문장을 "Table 5의 작은 지연 차이로 보아 긴 완결 답을 여러 개 만드는 방식은 아니라고 판단(우리 판단, 논문은 밝히지 않음, 짧은 답 여러 개를 한 번에 생성하는 방식은 배제되지 않음)"으로 바꿈.
- 다음 실험 세 개(작성자 제안: 중단된 hs 마무리, 이미지 입력 뒤 생각 이어가기, 둘을 섞은 것)를 README·README.ko "다음에 돌릴 실험(계획)" 절과 REPORT 5.4절에 추가(아직 실행·사전등록 안 함). 설계 제안(작성자 확인 전): 생각 이어가기 실험에는 이미지 없이 같은 만큼 더 생각하는 TSR 대조군(`tsr_think`)을 같은 세션에서 돌림(추가 생각만으로 생긴 이득과 구분); 새 실험 두 개는 본 실험에 저장된 첫 단계에서 출발(독립 검토 중간 1); 검출력은 실측 불일치율 16% 기준. 참고: 1단계 한정 같은 시험(step1_branch D − C, 309문항)은 +0.97 pp (−2.27, +4.21). 비용은 `visref_hs` 실측 속도(839문항 약 7.7시간) 기준 추정: 남은 161문항 약 2시간, 새 방법은 1,000문항에 방법당 9시간 이상(실측 아님).

## 2026-10-13 다음 실험 설계 결정: `tsr_think` 대조군 포함 (작성자 결정)
- 결정: `tsr_think` 대조군 유지. 이유: 기존 TSR 행은 추가 생각이 없어서, `visref_think`와 비교하면 이미지 재주입과 추가 생각(최대 512토큰) 두 가지가 함께 달라짐. 우리 자료에서도 생각을 더 하는 것만으로 점수가 오르는 경향(TSR − ST +2.3 pp (0.6, 4.0); step1_branch C − A +1.29 pp (−1.83, +4.42)). 기존 결과 중 같은 역할은 step1_branch C뿐인데 309문항·1단계·짧은 답 형식이라 1,000문항 실험의 짝이 될 수 없음. 첫 단계를 본 실험 저장본에서 시작하면 첫 단계 잡음 문제는 그것으로 풀리므로, 대조군의 목적은 "이미지 효과와 추가 생각 효과의 분리".
- 작성자 결정: `tsr_think` 포함(README·REPORT 5.4절의 계획 그대로). `visref_hs` 남은 161문항 실행 묶음은 나중에(작성자).

## 2026-10-13 `visref_think`·`tsr_think` 설계 결정 일부 (작성자 결정)
- 추가 생각 상한 512토큰(step1_branch C·D와 같음). 다시 생각을 시작할 때의 문구는 논문 문구 "Wait, think more"(논문 3절 "special instruction tokens (e.g., “Wait”, “Think more”), denoted by c", LAY L205-206; 그림 2 노란 상자 "Wait, think more", LAY L224; 현재 `run.py` wait_text와 같음, SPEC_LEDGER 16행). README·README.ko 계획 표와 REPORT 5.4절에 반영.
- 아직 정하지 않은 세부(작성자 확인 대기): 추가 생각을 모든 정지 사유(엔트로피·K_max·예산)에 적용할지, 추가 512토큰을 합계 상한 10,240과 별도로 둘지, 추가 생각 뒤 답 처리(본 실험 규칙 그대로: 스스로 닫으면 자연 답, 512에서 잘리면 s1 `\nFinal Answer:`, 답이 끝나지 않으면 DEER 답 강제), 판정 규칙과 보조 비교, 세션 분할.

## 2026-10-13 독립 검토 정정 반영, 추가 생각 적용 범위 결정
- 결과 후 표현 변경(판정·수치 변경 없음). README·README.ko·REPORT·SPEC_LEDGER·`diag/analyze_hs.py` 머리말:
  - 첫 단계 재생성 공개(검토 중간 1): REPORT 4.5절 설계("설계상 Z만 바꿨으나 첫 단계는 새로 생성, 텍스트 일치 193/839, 본 실험 TSR·VisRef는 ST와 1,000/1,000 같음"), 실행(사전등록 검출력 기준은 9% 가정; 실제 불일치 134/839 = 16.0%, 80% 검출 최소 효과 약 3.9 pp, +3.5 pp 검출력 약 72%), 보조 비교 단서, 동작 문단(텍스트 일치 193), 4.2절(fill 첫 단계 텍스트 일치 242/994), 6절, README 표 hs 행. 위 2026-10-11 "visref_fill 결과" 항목·2026-10-12 "후속 실험 3 `visref_hs` 실행 전 계획" 항목의 "fp16 배치 잡음 범위에서 같음"은 텍스트 기준으로 틀림(기록은 그대로 두고 여기서 정정).
  - Table 3·8 인용(중간 3): README "이득이 없던 이유" 표와 REPORT 4.2절 주의 문단에 Table 3 다양성만 77.4(LAY L538)·Table 8 λ = 0 71.2(LAY L984), 두 표가 서로 다름을 추가. SPEC_LEDGER Table 7 선택 방식(X-36) 보강.
  - 공식 59.3(중간 4): REPORT 1절과 README·README.ko 절대값 절에 "InternVL3.5 보고서는 Section 3 수치를 사고 모드를 포함한 TTS 없이 얻었다고 밝힘 → 사고 모드 없는 값일 가능성이 높음, 평가 설정은 미확인". SPEC_LEDGER 공식 1B 점수(X-1) 재검토 기록.
  - 추출기 변경 시점(중간 5): REPORT 4.5절에 "규칙만 결과를 계산·기록한 뒤 결정", "비교 상대 행은 Gemini 추출이라 규칙만 읽기가 같은 조건의 비교".
  - 작은 수정: REPORT 2.4절 후속 실험 셋; 4.5절 548단계 = m 998이 544 + m 844가 4, 분모 582 중 38; 순위 상관 부호(음수 = 관련성 큰 것이 먼저, `dpp.py` L107-108)대로 "약하게 관련성 순"과 fill "관련성과 거의 무관한 순서"로; "추가 이득 없음"을 "검출 안 됨"으로; README `make check` 지원 버전(3.11–3.13, 3.10은 숫자 검사만); README 파일 목록과 `make check` 설명의 "pre-registered" 표기; REPORT 5.1절(문구는 Alg. 1이 아니라 3절·그림 2); 2.1절 Chen 정지 기준(절대 1e-10, 우리 상대 기준); 2.5절 후속 실험 코드 버전과 현재 cf46b2e6222e; 4.3절 309문항 밖 불일치 27(14 : 13; 위 2026-10-11 "step1_branch 본 실행 결과" 항목 "309문항뿐"의 정정); 4.2절 층 구분은 사전 지정·EOS 층 해석은 결과를 본 뒤 붙인 것; SPEC_LEDGER 11·12행 기본값, 37행 코드 줄 번호; README.ko 번역 기준일.
  - 반성 기록(`docs/AI_USE_REFLECTION.md`): 2절 "같은 시드라 첫 단계는 같다", 3절 지표 부호 오독·59.3 판단, 예방법(직접 비교로 확인, 지표 정의 줄 인용, 후속 실험은 저장된 첫 단계에서 출발), 무엇이 오류를 잡았나에 독립 검토.
- 작성자 결정: `visref_think`·`tsr_think`의 추가 생각은 정지 사유(엔트로피·K_max·예산)에 관계없이 적용. 참고 기록: 본 실험 정지 사유 TSR 엔트로피 934 / K_max 66, VisRef 922 / 78, `visref_hs` 775 / 64, 예산 정지 0.

## 2026-10-13 실행 전 계획: 생각 이어가기 실험 `tsr_think`·`visref_think`·`visref_hs_think` (작성자 결정)
- 질문: 방법이 멈추기로 한 뒤 모델이 생각을 더 이어가게 하면(VisRef는 다시 넣은 이미지 Ṽ_k 뒤에서) VisRef가 TSR보다 논문 크기(+3.5 pp)의 추가 이득을 내는가. 같은 만큼 더 생각하되 이미지는 넣지 않는 `tsr_think`를 대조군으로 같은 실행에서 돌려, 생각을 더 한 효과와 이미지 효과를 나눈다(2026-10-13 "다음 실험 설계 결정: `tsr_think` 대조군 포함" 항목).
- 방법 세 개(`repro/visref/methods.py` THINK): `tsr_think`·`visref_think`·`visref_hs_think`는 각각 TSR·VisRef(`visref_raw_rank`)·마지막 층 Z VisRef(`visref_hs`)와 같은 반복을 돌고, 멈추기로 하면(엔트로피·K_max·예산 어느 이유든, 2026-10-13 "독립 검토 정정 반영, 추가 생각 적용 범위 결정" 항목) 마지막 단계 문맥 뒤(VisRef는 Ṽ_k 뒤)에 논문 문구 "Wait, think more"(2026-10-13 "`visref_think`·`tsr_think` 설계 결정 일부" 항목)를 붙여 최대 512토큰(2026-10-13 "`visref_think`·`tsr_think` 설계 결정 일부" 항목) 더 생각한 뒤 최종 답. 추가 생각은 한 번만이고 정지 probe를 다시 하지 않으며, 추가 생각 뒤에는 이미지를 넣지 않는다.
- 이번에 작성자가 승인한 세부: 추가 512토큰은 합계 상한 10,240과 별도(예산에서 빼지 않고 think_tokens에도 넣지 않음, extra_think_tokens로 따로 기록). 문맥 길이 40,960에 자리가 모자라면 상한을 줄이고, 자리가 없으면 건너뛰고 'skipped'로 기록. 추가 생각 뒤 답 처리는 본 실험 규칙 그대로: 스스로 `</think>`로 닫으면 자연 답, 512에서 잘리면 줄바꿈 + `</think>` + `\nFinal Answer:`(s1), EOS로 끝나면 `</think>` + 자연 답, 답이 답 상한에 닿으면 DEER 답 강제. 주 비교 `visref_think` − `tsr_think`, 판정 기준 +3.5 pp.
- 첫 단계(독립 검토 중간 1, 2026-10-13 "독립 검토와 확인" 항목): 세 방법 모두 본 실험에 저장된 첫 단계에서 출발(`run.py --step1-from diag/step1_store_bmain.json`). 저장 파일은 본 실험 ST 행 1,000개(b_main s1–s9 결과 파일; 본 실험에서는 모든 방법이 ST의 첫 단계를 공유)의 thinking·think_tokens·첫 단계 끝 유형(닫음 334 / 1,024에서 잘림 641 / EOS 25)을 `repro/diag/make_step1_store.py`로 옮긴 것(sha256 f937478f3b265f42…, 같은 입력에서 같은 바이트, step1_branch 표본 309문항과 세 값 모두 같음). 텍스트를 다시 토큰화해 저장 길이를 되찾았는지 행마다 기록(step1_info). 첫 단계 뒤 난수 상태는 문항마다 하나(job_seed(42, "pid:stored_step1"))를 세 방법이 함께 쓰므로, 새 실행 안에서 세 방법은 같은 첫 단계·같은 난수로 짝지어진다. 본 실험 TSR·VisRef 행과는 첫 단계만 같고(재토큰화 차이 제외) 그 뒤 난수 상태는 다르다.
- 나머지 설정은 본 실험과 같음: 단계 1,024 / 합계 10,240 / 답 1,024, δ 0.25, K_max 10, m = ⌊0.3|V|⌋, 정지 probe, 샘플링(T 0.6, top-k 50, top-p 0.95), 시드 42, 이중 probe 끔. 상한·추가 생각·저장 파일은 run_args로도 명시해 넘긴다(`--methods tsr_think,visref_think,visref_hs_think --think-cap 1024 --total-think-cap 10240 --think-extra 512 --step1-from diag/step1_store_bmain.json`, 빌드 `--data diag/step1_store_bmain.json`). code_version f2e54e35f667(이 사전등록 커밋의 `repro/`).
- 실행: Kaggle 2×T4, 세션당 timeout 29,000s, 1,000문항 seed 0 순서(b_main·fill·hs와 같음), 실행 이름 `b_main_think_s1`, `b_main_think_s2`, …, 두 번째 세션부터 이전 세션 결과를 --skip-from으로 건너뜀. 비용 추정(실측 아님): `visref_hs` 한 방법이 839문항에 약 7.7시간이었으므로 세 방법이면 세션당 대략 250–300문항, 1,000문항에 4세션 안팎. 세션 수는 결과가 아니라 예산으로 정한다: 세션 사이에는 상태(행 수, 오류, code_version, 시작 로그의 설정과 저장 파일 sha)만 보고, 정답률과 방법 간 비교는 계산하지 않는다. 작성자가 실행을 끝낸다고 정한 시점의 complete prefix로 판정을 한 번 한다.
- 분석 표본: 모든 세션의 행을 합친 뒤(같은 (pid, 방법)이 두 세션에 있으면 오류) 세 방법이 모두 끝난 complete prefix(2 shard). 300문항 미만이면 같은 규칙으로 판정하고 "검출력 부족" 표시.
- 채점: 새 행은 규칙 채점(v4) 뒤 규칙이 못 뽑은 행만 공식 MathVista 추출 프롬프트로 LLM 보조 추출(가드 동일, 대조군 20). 추출기는 Claude Haiku(fill·hs와 같은 방식). 바꾸려면 결과를 보기 전에 정해 기록한다. 주 비교의 두 팔은 같은 실행·같은 추출기. 규칙만 결과도 함께 보고. 본 실험 행은 `llm/final/merged.jsonl` 기록값.
- 주 비교(하나, 다중 비교 보정 없음): `visref_think` − `tsr_think`, 짝 부트스트랩 95% CI(2,000회, seed 0, summarize.paired와 같은 재표본)와 McNemar 정확 검정. 판정(CI만, 정답 수 합의 정수 경계로 비교):
  - ① 하한 > 0 → "생각을 이어가면 VisRef가 TSR보다 추가 이득"(상한 < +3.5면 "논문보다 작음"을 덧붙임)
  - ② 0을 포함하고 상한 ≥ +3.5 → 판단 보류(논문 크기 효과를 배제하지 못함)
  - ③ 상한 < +3.5(①이 아닐 때) → "생각을 이어가도 논문 크기(+3.5 pp)의 VisRef 추가 이득은 재현 안 됨"
  - +3.5는 논문 Table 6 InternVL3.5-1B의 VisRef 52.0 − TSR 48.5(SPEC_LEDGER Table 6 모델 크기별(A-5)).
- 검출력(사전 계산, 정규 근사, 불일치율 q에서 SE = √(q/n)): q는 본 실험 VisRef − TSR 89/1,000 = 8.9%(같은 첫 단계·같은 난수라 가장 비슷한 설계, `llm/final/merged.jsonl`에서 이번에 다시 셈)와 `visref_hs` − TSR 134/839 = 16.0%(첫 단계 재생성, 비관적 경우) 사이로 본다. 1,000문항이면 80% 검출 최소 효과 2.64–3.54 pp, +3.5 pp 검출력 0.96–0.79. +3.5 pp 검출력 0.8에 필요한 문항 수 570–1,023. 300문항이면 +3.5 pp 검출력 0.53–0.33. 판정과 함께 실현된 불일치율·최소 검출 효과·+3.5 pp 검출력을 보고한다.
- 보조(판정 없음, 기술): `visref_hs_think` − `tsr_think`(같은 규칙을 적용한 결과를 보조로 기록), `visref_hs_think` − `visref_think`, `visref_think` − 본 실험 VisRef, `tsr_think` − 본 실험 TSR(첫 단계만 같고 이후 난수가 다름), 2×2 대비 (`visref_think` − `tsr_think`) − (VisRef − TSR)와 부트스트랩 CI.
- 동작(기술): 추가 생각이 끝난 방식(닫음 / EOS / 512 도달 / 건너뜀), 길이 사분위, 상한이 512보다 줄어든 행 수, 추가 생각 글에 이미지 단어가 있는 비율(step1_branch와 같은 정규식), 정지 사유, k, 주입/m, 최종 답 안 두 번째 `</think>`, 답 강제 횟수, 저장 첫 단계 재토큰화 일치 수.
- 코드와 테스트(결과 전에 작성): `repro/visref/methods.py`(THINK, `_start_extra`, `_end_extra`, `stored_step1`), `repro/run.py`(`--step1-from`, think_extra 기본값 512), `repro/kaggle/build.py`(`--data`), `repro/diag/make_step1_store.py`, 분석 `repro/diag/analyze_think.py`(rows / analyze). 테스트 `repro/tests/test_think.py`(추가 생각 20개: 세 정지 사유, 512 도달·EOS·닫음 뒤 답 처리, 자리 없음, hs, 저장 첫 단계, run.py 통합), `test_analyze_think.py`(15개), `test_decisions.py`(think_extra 512, 저장 파일 = 본 실험 ST 행, `b_main_think*` 실행 인자). CPU 테스트 183 passed.
- 해석 범위: 추가 이득이 검출되지 않으면 "우리 구현에서 멈춘 뒤 생각을 이어가게 해도 검출 안 됨"까지만 말한다. 남은 차이(주입 형식·위치, 정지 추정, 토큰 상한)는 이 실험이 시험하지 않는다.

## 2026-10-13 결정 대장 코드 줄 번호 갱신 (실행 전 계획 커밋 8cd8d62 뒤, 코드 변경 없음)
- 사전등록 커밋에서 `run.py`·`visref/methods.py`·`kaggle/build.py` 머리말과 코드가 늘어 SPEC_LEDGER 결정 대장 J-1~J-20의 코드 줄 번호와 문서 머리말 "DEFAULTS(L20-23)"가 밀림. 옛 판(d708dd1)과 새 판을 diff로 줄 단위 대응시켜 같은 내용의 새 줄로 고침(예: DEFAULTS L20-23 → L25-28, 정지 판정 뒤 이어가기 L299-306 → L333-343, 주 VisRef 선택 L203 → L237). 반성 기록 "남은 불일치"의 SPEC_LEDGER 줄 번호는 행 이름으로 바꿈(표를 넣을 때마다 밀리므로).

## 2026-10-13 실행 전 계획 보완: 저장된 첫 단계의 끝 줄바꿈 처리 (코드 수정, 설계·판정 규칙 변경 없음)
- 실행 전 코드 점검: 읽기 전용 독립 검토를 했고 그 보고를 저장 파일과 테스트로 확인했다. 높음·중간 결함은 보고되지 않았다.
- 낮음 1(수정): 본 실험에서 1,024토큰에 잘린 첫 단계의 텍스트가 이미 '\n'으로 끝나면 구분자가 붙지 않는다. 그런데 `stored_step1`은 끝의 '\n'을 구분자로 보고 떼어 낼 수 있었고, 그러면 '\n\n'으로 끝난 단계는 문맥에서 줄바꿈이 하나 빠진다. 저장 파일에서 이런 문항은 잘린 641개 중 13개다(pid 57, 140, 166, 311, 519, 699, 776, 797, 819, 901, 914, 940, 941; 저장 파일에서 직접 셈). 실제 토크나이저에서도 재토큰화 길이가 같게 나와 이 후보가 골라지는지는 미확인이다(검토에서는 HF 접속이 막혀 step1_branch에 저장된 토큰으로 추정했다).
- 수정 내용: 구분자 앞 텍스트가 이미 '\n'으로 끝나면 떼지 않는다. 본 실험 규칙("텍스트가 '\n'으로 끝나지 않을 때만 구분자를 붙인다")을 거꾸로 적용한 것이다. '\n\n'을 한 토큰으로 묶는 테스트용 토크나이저로 테스트 2개를 추가했고, 옛 코드에서는 실패하고 새 코드에서는 통과함을 확인했다. 전체 테스트 186개 통과. code_version은 f2e54e35f667에서 86a5fc0f3cd9로 바뀐다. 독스트링 줄 수를 유지해 결정 대장의 줄 번호는 그대로다.
- 낮음 2(조치 없음, 확인 필요): 저장 파일 때문에 노트북 셀이 약 0.9 MB로 커졌다(base64 863,504자, 측정). Kaggle 크기 한도는 미확인이다. push 오류가 나면 파일을 줄인다.
- 낮음 3(조치 없음): `_start_extra`의 문맥 길이 계산은 최악의 경우 1토큰 모자랄 수 있다. 그러나 본 실험 final_context_len 최대가 15,666(확인)으로 40,960에 한참 못 미쳐 실제로는 일어나지 않는다.
- 검토 보고(미재현): 기존 다섯 방법의 출력이 커밋 전후 같다(무작위 시나리오). 3방법을 한 세션에서 돌린 결과와 방법별로 따로 돌린 결과가 같다(작은 모델).

## 2026-10-13 구현 근거 대장 37·44·19행 정정, 이름을 `IMPLEMENTATION_CHOICES.md`로 변경
- 정정(결과·코드 영향 없음, 문서만): 10-07 감사 결론(대장 "감사 반영 기록"의 메모)이 표에 반영되지 않았던 것을 반영. 37행(DPP가 m개 전에 멈출 때의 처리) 상태 "인용 구현에서 확인" → "가정(미명시)": \|V_k\| = m은 논문에 명시, 모자랄 때의 처리는 논문에 없고 인용한 Chen·CEIL 코드로는 저자 구현을 확인하지 못함(미확인). 44행(잘린 뒤 강제 답) 같은 변경: 논문은 강제 답을 언급하지 않고 s1 코드 방식을 우리가 가져다 씀. 19행(ST 정의)은 "명시" 유지, 우리 ST의 강제 답이 "without additional intervention"과 부분 충돌한다고 표시. 상태별 집계: 인용 구현에서 확인 10 → 8, 가정(미명시) 20 → 22(명시 19, 그림·표에서 유도 3, 합계 52 그대로; 표의 52행 상태 열을 다시 세어 확인).
- 이름 변경(작성자 결정, 후보 1): `SPEC_LEDGER.md` → `IMPLEMENTATION_CHOICES.md`(한국어 이름 "구현 근거 대장"). 공개 저장소 `docs/`, 작업 폴더 `repro/`. 현재 문서(README·README.ko·REPORT·SOURCES·반성 기록), 그림 `figures/method_overview.svg`의 문구, 분석 스크립트 주석(`diag/analyze_hs.py`, `analyze_think.py`, `analyze_st_plain.py`; 출력 불변), `tests/test_decisions.py` 머리말, `scripts/fetch_sources.py` 설명을 새 이름으로. 이 실험 기록과 감사·검토 원문, 백업 파일 이름에는 옛 이름을 그대로 둔다(당시 기록). 대장 첫머리에 옛 이름을 적음.
- 반성 기록 "지금 남아 있는 불일치"의 해당 항목을 "고침"으로 바꾸고 반성 한 줄(감사 정정은 메모가 아니라 그날 표를 고칠 것) 추가.

## 2026-10-15 실행 전 계획: 이미지 태그 주입 변형 `visref_bundle` 시험 가동 10문항 (작성자 결정)
- 배경(`repro/diag/analyze_k1_injection.json`, 본 실험에서 VisRef가 첫 단계에서 멈춘 문항들의 VisRef와 ST 비교):
  - 그중 첫 단계에서 생각을 스스로 닫은 329문항: 예측 일치 328/329, 최종 답 텍스트 일치 3/329, 최종 답 안 `</think>` 재출력 ST 20 → VisRef 156.
  - 그중 첫 단계가 상한에 잘린 366문항: 재출력 ST 47 → VisRef 196.
  - 읽기: 지금의 주입(태그 없는 벡터를 생각 블록 끝에 넣고 `</think>`를 다시 붙임)은 예측을 거의 바꾸지 않으면서 답의 형식을 깨뜨린다.
  - 이전 기록: smoke2(2026-10-07 "smoke2 결과와 본 실험 결정" 항목, REPORT B2)에서는 태그를 붙인 블록을 `</think>` 앞(생각 블록 안)에 줄바꿈 없이 넣었고(`z_k <img>Ṽ_k</img> </think>`, smoke2 실행 코드에서 확인), 두 번째 `</think>`가 태그 있음 31/63단계, 태그 없음 10/65단계로 태그를 붙인 쪽이 더 많았다. 그래서 태그만 붙이는 것은 형식 손상을 줄이지 않았다.
  - 가설: 손상의 원인은 태그 유무보다 블록이 `</think>` 앞에 있다는 위치다. 블록을 `</think>` 뒤로 옮기고(답할 때) 공식 형식으로 감싸면 형식 손상이 줄고 재주입이 답에 반영될 여지가 생긴다. 이번 시험은 태그 효과가 아니라 위치를 바꾼 묶음의 효과를 본다.
- 변형 정의(`repro/visref/methods.py` BUNDLE): 논문이 정하지 않은 자유도 네 가지를 한꺼번에 바꾼 탐색 변형. 효과가 보이면 그때 요소별로 나눈다.
  1. 형식: 고른 토큰을 공식 이미지 입력 형식 `<img>` + 토큰 + `</img>`로 감싼다(InternVL 공식 코드 `modeling_internvl_chat.py` L216-217, L295; 결정 대장 이미지 입력 형식(X-35)). 주입 벡터는 전처럼 입력 임베딩.
  2. 위치: 멈출 때(정지 probe, 강제 답, 자유 답)는 `z_k </think>\n<img>Ṽ_k</img>\n` 뒤에서 답한다. 이어갈 때는 생각 블록 안 `z_k \n<img>Ṽ_k</img>\n Wait, think more z_{k+1}`. 조건으로 쓰는 정보(x와 모든 (z_i, Ṽ_i))는 알고리즘 1의 4·12·17행과 같고, `</think>` 기준 위치는 알고리즘에 없는 템플릿 수준의 선택이다.
  3. 커널: 텍스트 Z와 이미지 V 모두 마지막 층 은닉 상태(최종 정규화 뒤)로 L = (V Zᵀ)(V Zᵀ)ᵀ를 만든다(`Engine.last_hidden`, 새 `Engine.hidden_at`; V는 프롬프트의 이미지 토큰 위치에서 문항당 한 번 계산). `visref_hs`는 Z만 마지막 층이었다. 논문은 "embeddings"라고만 하고 층을 말하지 않는다(LAY L178-180, L304-306; 결정 대장 29·30행). 참고: `kaggle/runs/hidden_rank_check` 12표본에서 Z가 마지막 층일 때 V도 마지막 층으로 바꾸면 잔차 비율 0.01 이상인 선택 수가 12/12 줄었다(예: pid 755 158 → 37). 커널 모양에 대한 진단이며 정확도 증거는 아니다.
  4. 줄바꿈: 블록 앞뒤 줄바꿈 토큰(공식 프롬프트에서 이미지 블록이 한 줄로 따로 있는 모양을 따름). 멈출 때는 `</img>\n` 뒤에 probe 문구 `\n\n\boxed`가 와서 줄바꿈 3개가 이어진다.
  - 그 밖에는 본 실험과 같다: 선택 수는 m 이하(결정 대장 37행과 같은 차이), 정지 규칙, 답 처리, 상한, 시드. 문맥 여유에 태그와 줄바꿈 4토큰을 더한다.
- 코드 변경(결과 전): `methods.py`(메서드, BUNDLE), `engine.py`(`hidden_at`), `model.py`(`<img>`·`</img>` id, `encode_sample(with_positions)`), `run.py`(이 메서드가 있으면 문항당 `vis_hs` 계산). 테스트 `tests/test_bundle.py` 6개. 기존 메서드의 코드 경로는 바뀌지 않았다(전체 테스트 200개 통과). code_version은 86a5fc0f3cd9에서 aa11cdaa52a9로 바뀐다. 아직 실행하지 않은 생각 이어가기 실험(2026-10-13 항목)을 이 코드로 돌리면 code_version만 다르고 동작은 같다.
- 이미지 없는 기록용 probe의 난수(결과 전 수정): 기록용 probe(`dual_probe`)가 결정 probe와 다른 난수열(`run.py`의 `{pid}:probe`와 `{pid}:probe_noV`)을 써서, 두 답의 차이를 이미지 유무로 읽을 수 없었다. 이제 기록용 probe는 결정 probe가 시작한 난수 상태를 복사해 쓴다(별도 생성기 객체라 결정 경로는 그대로). 단계 기록에 `probe_noV_crn=True`. 테스트 `tests/test_probe_crn.py` 3개. 이전 fill 실행의 이중 probe 기록은 이 조건을 만족하지 않아 아래 ①에 쓰지 않는다.
- 시험 가동(통계 아님): 코드가 실제 모델에서 돌고 실행 문제(빈 답, 즉시 종료, 오류)가 없는지, 형식 손상이 눈에 띄게 달라지는지만 본다.
  - 문항 10개: 저장된 첫 단계(`diag/step1_store_bmain.json`; 닫음 334, 잘림 641, EOS 25)에서 pid 순으로 앞 5개씩. 닫음 15, 18, 19, 21, 22 / 잘림 1, 2, 3, 4, 5. EOS는 제외.
  - 방법: `visref_raw_rank`(본 실험 VisRef) + `visref_bundle`. 둘 다 저장된 첫 단계에서 출발하고 첫 단계 뒤 난수를 공유한다. 이중 probe 켬.
  - 실행: `kaggle/build.py bundle_pilot --ids 15,18,19,21,22,1,2,3,4,5 --data diag/step1_store_bmain.json --run-args "--methods visref_raw_rank,visref_bundle --think-cap 1024 --total-think-cap 10240 --step1-from diag/step1_store_bmain.json --dual-probe 1"`.
- 읽기(`repro/diag/analyze_bundle.py`, 결과 전 작성, 채점 없음; 방법별 열, 닫음·잘림 묶음별 표):
  - ① 두 probe가 모두 유효한 단계에서 상자 안 답이 다른 비율(공통 난수 표시가 없는 단계가 있으면 해석 불가로 경고).
  - ② 최종 답 안 `</think>` 재출력 행 수.
  - ③ 최종 답 길이 분위수, 답 상한 도달 수, 답 강제 수.
  - 실행 문제 표시: 빈 답, 오류, 주입이 0개인 단계, 2토큰 이하에서 닫힌 단계. 주입 수/m 분위수.
- 해석 규칙(결과 전 고정):
  - 실행 문제가 하나라도 있으면 원인을 고치기 전에는 다음 단계로 가지 않는다.
  - 5문항 묶음의 비율 차이는 읽지 않는다(잘림 묶음의 재출력 기준율 약 0.5에서 0/5가 나올 확률은 약 3%이므로 0/5만 의미 있는 신호로 본다).
  - ①은 두 방법의 값을 나란히 적되 판정하지 않는다.
  - 시험 가동 뒤 정할 것(결과를 본 뒤, 다음 실행 전에 기록): 300문항 비교를 할지, 할 경우 "hs 커널 + 형식 변경만" 조건을 더해 커널 변경과 형식 변경을 나눌지.
  - 범위: 이 변형은 논문에 더 가까운 구현이 아니라 논문이 말하지 않은 자유도를 다른 쪽으로 채운 구현이다. 이득이 보여도 재현이 아니라 "구현 자유도가 결과를 좌우한다"는 근거로 보고한다.

## 2026-10-15 이미지 태그 주입 시험 가동 결과 (`bundle_pilot`, code_version aa11cdaa52a9): 실행 문제 발견, 다음 단계로 가지 않음
- 실행: Kaggle 2×T4, 세션 상한 2,400s, 커널 약 785s. 20행(10문항 × 2방법) 모두 끝남, 상태 오류 0. 저장된 첫 단계 재토큰화 20/20 일치. 정답 여부는 보지 않았다(`repro/diag/analyze_bundle.py`, 결과 `repro/diag/analyze_bundle.json`).
- 실행 문제(해석 규칙상 다음 단계로 가지 않음): 생각을 스스로 닫은 5문항에서 `visref_bundle`의 최종 답이 5/5 `Question: …`으로 시작해 질문을 다시 적었다. 1개(pid 19)는 `<think>`부터 다시 생각했고, 1개(pid 22)는 답 끝에 `</think>`를 내서 채점할 답이 비었다. 같은 문항의 `visref_raw_rank`는 5/5 평범한 답으로 시작했다.
- 원인 추정: 공식 프롬프트에서 `<img>…</img>\n` 바로 뒤에는 질문이 온다. `</think>\n<img>Ṽ_k</img>\n` 뒤에서 자유롭게 답하게 하면 모델이 새 질문 입력이 시작된다고 읽는 것으로 보인다. 첫 단계가 잘린 5문항은 강제 답 문구(`\nFinal Answer:`) 뒤에서 답해 이 현상이 없었다(예: `Final Answer: 5500`).
- 그 밖(5문항씩이라 비율은 읽지 않음): 주입이 0개인 단계 0, 2토큰 이하에서 닫힌 단계 0. 주입 수/m 중앙값 `visref_bundle` 0.97, `visref_raw_rank` 0.19. 이미지 유무로 멈춤 판단용 답이 다른 단계 `visref_bundle` 2/19(모두 잘린 묶음), `visref_raw_rank` 0/10(두 probe는 같은 난수 상태에서 시작). k: `visref_bundle`은 1문항(pid 2)이 K_max 10까지 이어감, 나머지는 모두 1단계에서 멈춤.
- 다음: 자유 답 앞의 모양을 고친 뒤 같은 10문항으로 다시 시험 가동한다. 고칠 방식은 작성자가 정한다.

## 2026-10-15 이미지 태그 주입 수정: 모든 최종 답을 `\nFinal Answer:` 뒤에서 (작성자 결정, 다시 시험 가동 전)
- 바꾼 것: `visref_bundle`은 생각을 스스로 닫은 단계나 EOS로 끝난 단계에서도 최종 답을 `</think>\n<img>Ṽ_k</img>\n` + `\nFinal Answer:` 뒤에서 생성한다(전에는 잘린 단계만). 채점 텍스트는 잘린 단계와 같이 이 문구 + 답이다. 정지 probe(`\n\n\boxed`)와 다른 방법은 그대로다. 코드 `repro/visref/methods.py` `_start_answer`(줄 수 그대로), 테스트 `tests/test_bundle.py`(닫은 단계의 답 문맥과 채점 텍스트). 전체 테스트 201개 통과. code_version aa11cdaa52a9 → d2ab88b06b24.
- 이유: 앞 항목의 실행 문제(자유 답이 `Question: …`으로 시작). 같은 시험에서 이 문구 뒤에서 답한 잘린 5문항은 이 현상이 없었다.
- 출처: 논문은 최종 답 프롬프트를 정하지 않는다(알고리즘 1의 17행 "Sample y ∼ πθ(· | xinput, τ1:k)"; 그림 2의 "Answer:" 상자는 정지 기준 도식). `\nFinal Answer:`는 논문이 인용한 s1 [34] 공식 코드의 문구로, 본 실험에서는 잘린 단계에만 썼다(결정 대장 생각 잘림 시 답(J-11)). 모든 답에 쓰는 것은 우리 선택이다.
- 비교 범위: 이제 `visref_bundle`은 모든 답이 이 문구 뒤에서 나오고, `visref_raw_rank`와 TSR은 자유 답이다. 시험 가동의 확인 항목(질문 재작성, 빈 답, 즉시 종료)에는 상관없다. 300문항 비교로 간다면 같은 문구를 붙인 TSR 비교 조건이 있어야 이미지 위치 효과와 답 문구 효과를 나눌 수 있다(그때 결정).
- 다시 시험 가동: 같은 10문항, 같은 두 방법과 실행 인자, 실행 이름 `bundle_pilot2`. 확인할 것은 앞 항목의 실행 문제가 사라지는지 하나다. 해석 규칙은 2026-10-15 실행 전 계획 항목과 같다.

## 2026-10-15 이미지 태그 주입 다시 시험 가동 결과 (`bundle_pilot2`, code_version d2ab88b06b24): 실행 문제 없음
- 실행: Kaggle 2×T4, 세션 상한 1,800s, 커널 약 437s. 20행 모두 끝남, 상태 오류 0, 저장된 첫 단계 재토큰화 20/20 일치. 정답 여부는 보지 않았다(`repro/diag/analyze_bundle.py`, 결과 `repro/diag/analyze_bundle_pilot2.json`).
- 확인 항목: 생각을 스스로 닫은 5문항에서 `visref_bundle`의 최종 답은 5/5 `Final Answer:` 뒤의 짧은 답이었다. 질문 재작성, 생각 다시 시작, 빈 답, 답 안 `</think>` 재출력은 0/5(앞 실행은 질문 재작성 5/5, 빈 답 1). 같은 문항의 `visref_raw_rank` 최종 답 텍스트는 앞 실행과 5/5 같았다(재출력 2/5; 잘린 묶음에서는 3/5만 같음).
- 그 밖(5문항씩이라 비율은 읽지 않음): 실행 문제 표시(빈 답, 오류, 주입 0개 단계, 2토큰 이하에서 닫힌 단계) 모두 0. 첫 단계가 잘린 5문항에서는 두 방법 모두 답 안 `</think>` 재출력이 3/5. `visref_bundle`에서 1문항(pid 2)이 4단계까지 이어갔다(앞 실행에서는 10단계; 코드에서 바뀐 것은 최종 답 문구뿐이므로 GPU 계산의 비결정성으로 보인다, 미확인).
- 다음: 300문항 비교를 할지, 할 경우 같은 답 문구를 붙인 TSR 비교 조건을 넣을지 작성자가 정한다.

## 2026-10-15 이름 바꿈: `visref_bundle` → `visref_imgtag`, `tsr_cue` → `tsr_answer_prompt` (작성자 결정, 코드 동작은 그대로)
- 읽기 쉬운 이름으로 바꿈. `visref_imgtag`는 이미지 태그 주입 변형, `tsr_answer_prompt`는 모든 최종 답을 `\nFinal Answer:` 뒤에서 받는 TSR(이미지 태그 주입의 비교 조건, 이번에 추가)이다. 코드 상수 `BUNDLE` → `IMGTAG`, `CUE` → `ANSWER_PROMPT`.
- 파일: `repro/diag/analyze_bundle.py` → `repro/diag/analyze_imgtag.py`, `repro/tests/test_bundle.py` → `repro/tests/test_imgtag.py`, `repro/tests/test_analyze_bundle.py` → `repro/tests/test_analyze_imgtag.py`. 시험 가동 결과 파일 `repro/diag/analyze_bundle.json` → `analyze_imgtag_pilot1.json`, `analyze_bundle_pilot2.json` → `analyze_imgtag_pilot2.json`.
- 그대로 둔 것: 앞 항목들의 옛 이름(당시 기록), 실행 이름 `bundle_pilot`·`bundle_pilot2`와 그 결과 행의 메서드 이름 `visref_bundle`. 같은 동작이다.
- 테스트 202개 통과(`tsr_answer_prompt` 테스트 1개 포함). code_version 6e218887d2b5.

## 2026-10-15 실행 전 계획: 이미지 태그 주입 1,000문항 비교 (`imgtag_main`, 작성자 결정)
- 질문: 이미지 태그 주입(`visref_imgtag`)이 같은 답 문구를 쓰는 TSR(`tsr_answer_prompt`)보다 정답률이 높은가. 같은 실행에 보통 TSR(`tsr`)을 넣어 논문과 같은 정의의 비교(`visref_imgtag` − `tsr`)와 답 문구만의 효과(`tsr_answer_prompt` − `tsr`)도 같은 조건에서 본다(독립 검토 권고, 작성자 결정).
- 비교 조건 셋, 모두 같은 실행:
  - `tsr`: 본 실험 TSR과 같은 코드.
  - `tsr_answer_prompt`: TSR과 같이 생각하고 멈추되 모든 최종 답을 `\nFinal Answer:` 뒤에서 받는다.
  - `visref_imgtag`: `tsr_answer_prompt`에 단계마다 태그로 감싼 이미지 블록 재주입을 더함(답할 때는 `</think>` 뒤, 이어갈 때는 생각 안; 고르는 기준은 텍스트·이미지 모두 마지막 층; 주입 벡터는 입력 임베딩). 정의는 2026-10-15 앞 항목들과 결정 대장 이미지 태그 주입(J-24).
- 출발점: 세 조건 모두 본 실험에 저장된 첫 단계에서 출발(`repro/diag/step1_store_bmain.json`, sha256 f937478f3b26…). 첫 단계 뒤 난수 상태는 문항마다 하나를 세 조건이 함께 쓴다(`tsr`과 `tsr_answer_prompt`는 최종 답 문구만 달라 생각 단계의 문맥과 난수가 끝까지 같다. `visref_imgtag`는 2단계부터 문맥이 달라 실제로 같은 것은 첫 단계 직후의 probe와 답 난수다).
- 나머지 설정은 본 실험과 같다: 단계 1,024 / 합계 10,240 / 답 1,024토큰, 멈춤 기준 0.25, 최대 10단계, 주입 수 상한 m = 이미지 토큰의 30%, 샘플링(온도 0.6, top-k 50, top-p 0.95), 시드 42, fp16, 배치 16 / batch_tokens 72,000 / chunk 1,024(재개할 때도 바꾸지 않음). 이미지 없는 기록용 probe는 켬(`--dual-probe 1`, 결정 probe와 같은 난수 상태).
- 실행 인자(문자 그대로): `--methods visref_imgtag,tsr_answer_prompt,tsr --think-cap 1024 --total-think-cap 10240 --step1-from diag/step1_store_bmain.json --dual-probe 1 --shard 0/1`, 문항 순서 `repro/kaggle/runs/b_main_hs/output/results/ids.json`(본 실험의 seed 0 섞은 순서, sha256 a5ad8da9…, 1,000문항). code_version 6e218887d2b5.
- 실행 장소와 방법(작성자 결정: RunPod): 한 장짜리 GPU 한 대. 실행기 `repro/runpod/run_imgtag.py`(위 설정을 한 곳에 고정; 고정 판 모델·데이터 내려받기, run_config·env 기록, 같은 명령으로 이어 달리기, 3,000행이 다 찼을 때만 summarize, GPU 이름 기록). pod는 중간에 회수되지 않는 상시형으로 빌리고, 결과 폴더는 pod를 멈춰도 남는 `/workspace` 아래에 둔다. GPU가 본 실험(T4)과 다르므로 이 실행의 행은 같은 실행 안에서만 비교한다. 결과 폴더는 `repro/kaggle/runs/imgtag_main/output/results/`로 옮겨 기존 관례에 맞춘다.
- 확인 실행(본 실행 전): 현재 코드(6e218887d2b5)와 `tsr_answer_prompt`는 GPU에서 돌려 본 적이 없으므로, 같은 pod에서 `--name imgtag_check --limit 6 --max-errors 0`으로 먼저 돌린다(작업 18개로 배치 16을 채움; 이름이 `imgtag_main`으로 시작하지 않아 분석에 섞이지 않음). 도중에 한 번 일부러 끊고 같은 명령으로 이어 달려 본다. 보는 것: 오류 행, fast_kv 점검 줄, 최대 메모리, 시간당 행 수, 이어 달리기 뒤 중복 행 없음. 정답률은 보지 않는다. 오류 행이 하나라도 나오면 본 실행을 하지 않고 원인을 고친다(코드를 고치면 code_version과 이 항목을 갱신하고 확인 실행부터 다시). 이 처리량으로 본 실행 시간을 추정하고 `--hours`를 정한다.
- 예산과 멈춤: 본 실행 시간은 미리 추정하지 않고, 확인 실행에서 잰 처리량(시간당 행 수)으로 정한다. 실행 시간은 작성자가 정하고 `--hours`로 맞춘다. 본 실행은 `--stop-pod`를 붙여 끝나면 pod를 멈춘다. 오류: 기본값 `--max-errors 10`, 오류 행이 10개를 넘으면 실행기가 run.py를 멈추고 요약을 쓰지 않는다. 원인이 코드 밖(메모리, 내려받기 등)이면 고친 뒤 `--redo-errors`로 오류 행만 다시 돌리고, 코드를 고쳐야 하면 code_version이 달라지므로 새 이름으로 처음부터 다시 한다. 1,000문항 전에 시간 한도에 걸리면 같은 GPU 종류로 같은 명령을 다시 실행해 이어 달리고, 그 시간 안에 못 채우면 complete prefix로 분석한다. 실행 중에는 상태(행 수, 오류, code_version, 시작 로그의 설정과 sha)만 보고, 정답률과 조건 간 비교는 계산하지 않는다(실행기는 화면에 정답 여부를 찍지 않고, 요약은 3,000행이 다 찼을 때만 쓴다).
- 분석 표본: 세 조건이 모두 끝난 complete prefix(shard 1개). 같은 (pid, 방법)이 두 파일에 있으면 오류. 모든 행의 code_version과 저장 파일 sha가 하나여야 한다. 300문항 미만이면 같은 규칙으로 판정하고 "검출력 부족" 표시.
- 채점: 규칙 채점(v4) 뒤 규칙이 못 뽑은 행만 공식 MathVista 추출 프롬프트로 LLM 보조 추출(가드 동일, 대조군 20, Claude Haiku). 추가 규칙(결과 전 고정): 최종 답이 `\nFinal Answer:` 뒤에서 비어 있으면(뒤에 DEER 답 강제 문구 `\n**Final Answer**\n\boxed`와 빈 상자만 있는 경우 포함; 모델이 `</think>`를 다시 냈으면 마지막 `</think>` 뒤 텍스트 기준) 오답으로 한다(지금 채점기는 빈 답에서 "Final Answer:"를 답으로 뽑고 보기 중 가장 비슷한 것을 고를 수 있다: 보기 Yes/No에서 Yes, 2026-10-15 직접 확인). 이런 행은 LLM 요청에서 빼고 조건별로 개수를 보고한다. 규칙만 결과도 함께 보고.
- 주 비교(하나, 다중 비교 보정 없음): `visref_imgtag` − `tsr_answer_prompt`, 짝 부트스트랩 95% CI(2,000회, seed 0)와 McNemar 정확 검정. 판정(CI만, 정답 수 합의 정수 경계로 비교):
  - ① 하한 > 0 → "이미지 태그 주입 묶음이 같은 답 문구의 TSR보다 이득"(상한 < +3.5면 "논문보다 작음"을 덧붙임)
  - ② 0을 포함하고 상한 ≥ +3.5 → 판단 보류
  - ③ 상한 < +3.5(①이 아닐 때) → "이 묶음으로도 논문 크기(+3.5 pp)의 이득은 재현 안 됨"
  - +3.5는 논문 Table 6 InternVL3.5-1B의 VisRef − TSR(결정 대장 Table 6 모델 크기별(A-5)).
- 보조(판정 없음): `visref_imgtag` − `tsr`(논문과 같은 정의; 같은 규칙의 결과 번호만 기록), `tsr_answer_prompt` − `tsr`(답 문구만의 효과), 이 실행의 `tsr` − 본 실험 TSR(참고만: 본 실험 TSR 1,000행은 코드 버전 3개에 걸치고 생성 중의 첫 단계와 다른 난수에서 이어졌으며 GPU·torch도 달라 차이의 원인을 나눌 수 없다). 첫 단계 끝 유형(닫음 334 / 잘림 641 / EOS 25)별 정답률과 주 차이. 규칙이 뽑은 답이 "Final Answer" 같은 문구인 행 수.
- 검출력(사전 계산, 정규 근사, 불일치율 8.9~16%): 1,000문항이면 +3.5 pp 검출력 0.96~0.79, 80% 검출 최소 효과 2.6~3.5 pp. 이번 변형은 단계 수와 주입 수를 함께 바꾸므로 불일치율이 16%보다 클 수 있다. 실현된 불일치율·최소 검출 효과·+3.5 pp 검출력을 함께 보고한다. 1,000은 testmini 전체라 늘릴 수 없다.
- 동작(기술): 조건별 단계 수, 정지 사유, 주입 수/m, 이미지 유무로 멈춤 판단 답이 갈린 비율(같은 난수 상태), 답 안 `</think>` 재출력, 빈 답 수, 답 강제(DEER) 수, 답 길이, 최대 문맥 길이, 저장 첫 단계 재토큰화 일치 수, 오류 행. 점검: `tsr`과 `tsr_answer_prompt`의 생각 단계(단계 수, 정지 사유, 단계별 생각 길이·끝 방식, 생각 텍스트)가 같은 문항 비율. 두 조건은 생각이 같아야 하므로, 다르면 배치 구성에 따른 fp16 계산 차이나 난수 공유 문제를 뜻하며 원인을 기록한다.
- 코드와 테스트(결과 전 작성): 메서드 `visref/methods.py`(`visref_imgtag`, `tsr_answer_prompt`), 실행기 `runpod/run_imgtag.py`(필요 패키지 확인·설치와 버전 기록, torch 2.5 이상 확인, 결과 폴더는 저장소 밖만 허용, 잠금 파일, 중단 신호 처리, 덜 쓰인 마지막 줄 분리, 모델 캐시 재사용, 오류 수 상한 `--max-errors`, 오류 행 다시 돌리기 `--redo-errors`, 끝나면 pod 멈춤 `--stop-pod`; 테스트 `tests/test_run_imgtag.py` 37개), 분석 `diag/analyze_imgtag_main.py`(rows / analyze; 테스트 `tests/test_analyze_imgtag_main.py` 34개). 전체 테스트 273개 통과. 실행 전 독립 검토 세 번(설계, 계획·코드, 실행 절차)의 지적을 반영했다.
- 해석 범위: 이 비교의 차이는 이미지 재주입 묶음 전체(태그 형식, 위치, 마지막 층 커널, m에 가까운 주입 수)와 그것이 정지 판단·문맥 길이·답 문구와 맺는 상호작용을 함께 담는다. ①이 나와도 "이 묶음의 효과"라고만 쓰고 요소별 기여나 본 실험 VisRef보다 나아졌다고 말하지 않는다. 이 묶음은 본 실험 VisRef 뒤 이 줄기에서 설계한 네 번째 변형(fill, hs, 생각 이어가기, 이미지 태그)이고 본 실험 자료를 보고 설계했으므로, ①이면 시드를 바꿔(저장된 첫 단계는 그대로, 2단계부터 달라짐) 한 번 더 확인한다. 이득이 검출되지 않으면 "형식 손상을 없애고 m에 가깝게 넣어도 검출 안 됨"까지만 말한다.

## 2026-10-16 확인 실행 결과와 본 실행 시작 (`imgtag_check`, `imgtag_main`)
- 장소: RunPod RTX A5000 24GB 한 장(드라이버 580.178.04), 이미지 `runpod/pytorch:1.4.0-cu1281-torch2121-ubuntu2204`(실행 기록 `env.json`의 torch는 2.14.1+cu130으로, 이미지 이름의 2.12.1과 다르다; 본 실험 Kaggle은 2.11.0), 커밋 8f5c09e, code_version 6e218887d2b5. 결과는 network volume(`/workspace/runs`)에 쓴다.
- 실행 환경 보정(코드 변경 없음, 실험 행 생성 전): 이미지가 `HF_HUB_ENABLE_HF_TRANSFER=1`을 켜 두지만 해당 패키지가 없어 모델 내려받기에서 멈췄다. pod 시작 명령에서 `HF_HUB_ENABLE_HF_TRANSFER=0`으로 끄고 다시 시작했다.
- 확인 실행(`--name imgtag_check --limit 6 --max-errors 0`): 18/18행, 오류 0, `fast_kv_check` ok. 8행에서 일부러 끊고 같은 명령으로 이어 달렸고 (pid, 방법) 중복 없이 18행이 됐다. 최대 메모리 10.8 GB / 24 GB. 정답률은 보지 않았다.
- 처리 속도: 같은 묶음 크기의 토큰 처리 속도가 T4 한 장의 약 1.5~2배(본 실험 `b_main_hs` 로그와 비교). `b_main_hs`는 T4 두 장으로 839행에 약 7.7시간이므로 본 실행 3,000행은 약 27~37시간으로 추정한다.
- 본 실행(작성자 결정): `--name imgtag_main --hours 33`. 실행 시간을 33시간으로 제한했고, 1,000문항 전에 끝나면 계획대로 complete prefix로 분석한다. 2026-10-15 18:55 UTC 시작. pod는 멈췄다 다시 켜지 않고(다시 켤 때 GPU가 배정되지 않을 수 있음) 같은 pod의 시작 명령만 바꿔 이어 간다.

## 2026-10-16 다음 실험 후보: 가장 충실한 설정으로 확인 실험 (작성자 결정, 설계는 A6 결과 뒤 확정)
- 배경: 원논문 저자 관점의 검토(2026-10-16)에서, 본 실험의 영(0) 결과는 논문의 작동 조건을 일부만 시험했다는 지적이 나왔다. 첫 단계가 1,024토큰 상한에 잘린 문항이 641/1,000이고, 다시 넣은 이미지 토큰은 m의 약 1/5(중앙값 0.199)이며, 1단계에서 멈춘 713문항 중 352문항에서 VisRef 답이 `</think>`를 다시 냈다(REPORT 3.3절, 4.1절).
- 후보 설정: 텍스트 Z와 이미지 V를 같은 층(마지막 층 은닉 상태)에서 잡고, 고른 토큰을 이미지 태그로 감싸 넣으며, m개를 채우고, 단계당 상한을 4,096토큰 이상으로 하고, 저장된 첫 단계를 함께 쓰며, bf16으로 실행한다.
- 사전 점검(작성자 결정): 멈춤 판단용 엔트로피의 온도와 추정 방식. 지금은 `\n\n\boxed` 뒤 답 하나(최대 16토큰)를 온도 1로 뽑아 토큰별 엔트로피를 더하고, 최종 답은 온도 0.6으로 만든다. 논문은 기대값 정의(알고리즘 1의 12행)만 주고 추정 방식과 온도를 적지 않았으며, 인용한 Ghosal 등은 답을 여러 개 뽑아 답 값 분포의 엔트로피를 쓴다(결정 대장 X-30). 저장된 첫 단계의 문항 일부에서 같은 문맥으로 (1) 온도 1·답 1개, (2) 온도 0.6·답 1개, (3) 답 8개의 답 값 분포 엔트로피(온도 1), (4) 같은 것(온도 0.6)을 함께 계산해, 방식마다 δ = 0.25 아래로 멈추는 비율과 멈춤 결정의 일치율을 본다. 정답률은 쓰지 않는다. 방식마다 멈춤 판단 한 번에 드는 시간도 함께 재서, 한 단계 생성 시간과 문항당 총시간에 대한 비율로 적는다(논문 Table 5: InternVL3.5-8B, H100 한 장에서 문항당 ST 7.1초, TSR 7.7초, VisRef 8.2초, 결정 대장 A-4; 답 여러 개는 같은 문맥을 공유해 한 묶음으로 뽑으므로 시간이 개수에 비례해 늘지는 않을 것으로 본다, 미확인). 결정이 크게 갈리면 확인 실험의 멈춤 방식을 이 결과로 정하고, 갈리지 않으면 지금 방식을 유지한다. 문항 수와 "크게 갈림"의 기준은 실행 전에 이 기록에 적는다.
- 성격: 후속 A2–A6은 탐색이고 서로 간 보정이 없다. 이 실행 하나를 확인 실험으로 미리 지정하고, 판정 규칙·문항 수·비교 대상은 A6 결과를 본 뒤 실행 전에 이 기록에 적는다.
- 알려진 한계(2026-10-17 추가, A6 결과 뒤): 선택 조합 후보를 "Z = 이번 단계 생각의 마지막 층, V = 언어 모델에 들어가는 이미지 값"으로 좁혔다(작성자 결정). 이 조합은 A4와 같으므로 A4에서 본 한계를 그대로 가질 수 있다: 의미 있는 선택(정규화 잔차 0.01 이상)이 이미지 크기와 거의 상관없이 60~80개에서 멈춰, 큰 이미지에서는 m의 대부분이 새 방향을 더하지 못한 채 채워진다(2026-10-17 "결과 뒤 탐색" 항목). 논문 점수는 관련성과 다양성의 곱이라(log det = Σ log rᵢ² + log det L̄, VisRef_layout.txt 365–366줄) 다양성이 바닥나면 관련성이 남은 자리를 채우지 못한다.
- 배제(2026-10-17 작성자 결정): 생각 토큰별 크기 정규화, 다양성이 바닥난 뒤 관련성 순 채우기처럼 논문 식을 바꾸는 변형은 논문 재현과 관련이 없으므로 하지 않는다.

## 2026-10-16 `imgtag_main` 실행 중 구조 점검과 분석 기술 항목 추가 (결과 전, 작성자 결정)
- 중간 점검(정답률은 계산하지 않음): 2026-10-16 01:45 UTC까지 쌓인 743행 중 세 조건이 모두 끝난 앞쪽 238문항. 오류 0, code_version과 저장 첫 단계 파일 각각 하나. 조건별(`visref_imgtag` / `tsr_answer_prompt` / `tsr`): `\nFinal Answer:` 뒤 빈 답 4 / 1 / 0, 최종 답 안 `</think>` 재출력 45 / 25 / 27, 답 상한 도달 36 / 34 / 34, 평균 단계 수 2.82 / 2.36 / 2.36. `visref_imgtag`의 주입 수/m 중앙값 1.0, 이미지 유무로 멈춤 판단 답이 갈린 단계 117/663. `tsr`과 `tsr_answer_prompt`의 생각이 같은 문항 226/238(첫 단계에서 닫힘·EOS는 모두 같고, 잘림 152문항 중 140). 멈출 이유가 없어 실행을 계속했다.
- 분석 기술 항목 추가(판정에 쓰지 않음): `diag/analyze_imgtag_main.py`의 동작 기록에 `reemit_vs_answer_cap`(최종 답에서 `</think>`를 다시 낸 행과 그렇지 않은 행 각각의 행 수, 답 상한 도달 수, DEER 답 강제 수)를 넣었다. 재출력이 답을 상한까지 끌고 가 손해를 보는지 보기 위한 것이다. 테스트 1개 추가(전체 276개 통과), code_version은 그대로 6e218887d2b5.
- 분석 기술 항목 추가 2(판정에 쓰지 않음, 같은 날 결과 전): `cost`(방법별 생성 토큰 합·분위수, 생각 토큰, 최종 문맥 길이, 행당 시간 중앙값, 그리고 `visref_imgtag`의 두 TSR 조건에 대한 비율. 행은 여러 문항을 묶어 처리하므로 시간은 참고로만 쓰고 토큰 비율을 주로 본다. 논문 Table 5의 VisRef/TSR = 8.2초/7.7초와 비율로만 대조)와 `stop_entropy`(유효한 멈춤 판단 단계의 엔트로피 H 분위수, δ = 0.25 아래 비율, δ ± 0.1 안 비율; `visref_imgtag`는 같은 난수 상태의 이미지 없는 판단과의 평균 차이 H − H_noV와 멈춤 결정이 갈린 단계 수; 합 대신 여는 중괄호 다음 첫 토큰의 엔트로피만 쓸 때 멈춤 결정이 갈리는 단계 수). 테스트 2개 추가(전체 278개 통과), code_version 그대로.
- 분석 기술 항목 추가 3(민감도 점검, 판정에 쓰지 않음, 같은 날 결과 전): `source_cluster`. testmini 1,000문항은 31개 원본 데이터셋(`metadata.source`)에서 왔고, 같은 원본의 문항끼리는 함께 맞거나 틀릴 수 있다. 그래서 주 비교와 `visref_imgtag − tsr`의 차이를 문항 대신 원본 단위로 다시 뽑아(2,000번, seed 0) 95% 구간을 내고, 원본을 하나씩 뺐을 때 차이의 최저·최고와 원본별 불일치 수를 함께 적는다. 판정 ①②③은 지금처럼 문항 단위 구간만으로 정한다. 테스트 2개 추가(전체 280개 통과), code_version 그대로.

## 2026-10-16 `imgtag_main` 결과: 이미지 태그 주입 묶음도 논문 크기 이득 없음 (판정 ③)
- 실행: 2026-10-16 17:32 UTC에 3,000/3,000행으로 끝남. 오류 0, code_version 6e218887d2b5 하나, 저장 첫 단계 sha f937478f3b26 하나. 생성 81,337초(22.6시간). 결과 16개 파일을 내려받아 줄 수(3,000쌍, 읽을 수 없는 줄 0)와 sha256을 확인한 뒤 RunPod pod와 volume을 지움. 내려받기 스크립트가 처음에 본 실행 폴더 경로를 빠뜨려 실패했고, 경로만 고쳐 다시 받음(검증 조건은 그대로).
- 채점: 규칙(v4) 뒤 규칙이 못 뽑은 9행(imgtag 3 / 같은 답 문구 TSR 1 / TSR 5)만 Claude Haiku로 공식 추출 프롬프트 사용, 정답은 넣지 않음. 바뀐 결과 0, 대조군 20행 모두 규칙과 일치. `\nFinal Answer:` 뒤 빈 답 16 / 3 / 2행은 규칙대로 오답(LLM 요청에서 뺌).
- 정답률(1,000문항): `visref_imgtag` 60.1 / `tsr_answer_prompt` 61.1 / `tsr` 60.4 %.
- 주 비교 `visref_imgtag` − `tsr_answer_prompt`: −1.0 pp (95% CI −3.1 ~ +1.0, 49 대 59, McNemar p = 0.3866). 상한이 +3.5 미만 → 판정 ③ "이 묶음으로도 논문 크기(+3.5 pp)의 이득은 재현 안 됨". 불일치 108/1,000(10.8%), 80% 검출 최소 효과 2.91 pp, +3.5 pp 검출력 0.92. 규칙 채점만: 같은 수치와 판정.
- 보조: `visref_imgtag` − `tsr` −0.3 pp (−2.4 ~ +1.8, 52 대 55, p = 0.8468; 같은 규칙 결과 번호 3). `tsr_answer_prompt` − `tsr` +0.7 pp (0.0 ~ +1.5, 10 대 3, p = 0.0923). 참고만: 이 실행 `tsr` − 본 실험 TSR +0.3 pp (−1.5 ~ +2.0).
- 원본 단위 민감도(결과 전 추가): 31개 원본 단위로 다시 뽑은 95% 구간 −3.03 ~ +1.06. 원본 하나씩 빼면 −1.6(CLEVR-Math 뺌) ~ −0.53(DVQA 뺌).
- 첫 단계 끝 유형별(서술, 결과 뒤에 들여다봄): 닫음 334문항 imgtag 69.16 / 같은 답 문구 TSR 71.86 %(0 대 9), 잘림 641문항 55.38 / 54.91 %(49 대 46), EOS 25문항 60.0 / 76.0 %(0 대 4). 닫음 묶음의 잃은 9문항 중 5문항은 빈 답, 4문항은 답이 달라짐. imgtag 빈 답 16건 중 15건이 닫음 묶음.
- 동작: imgtag 평균 단계 수 2.47(같은 답 문구 TSR 2.07, TSR 2.06), 10단계 상한 도달 88 / 58 / 58문항, 주입 수/m 사분위 1.0 / 1.0 / 1.0. 최종 답 안 `</think>` 재출력 249 / 107 / 113. 이미지 유무로 멈춤 결정이 갈린 단계 287/2,450. `tsr`과 `tsr_answer_prompt`의 생각이 같은 문항 969/1,000(닫음 334/334, EOS 25/25, 잘림 610/641; 다른 31문항의 원인은 보지 않음).
- 비용: imgtag 생성 토큰 합은 같은 답 문구 TSR의 1.132배, TSR의 1.125배. 행당 시간 중앙값 1.435배(행을 묶어 처리하므로 참고로만). 논문 Table 5 VisRef/TSR 8.2/7.7초 ≈ 1.06배(InternVL3.5-8B, H100 한 장).
- 결과 뒤 고친 것: 멈춤 엔트로피 기술 항목에서 이미지 없는 판단이 답을 닫지 못한 4단계가 엔트로피 무한대라 평균 H − H_noV가 −∞로 나옴. 평균에서만 이 단계를 빼고 수를 따로 적도록 고침(멈춤 결정 비교에는 그대로 포함). 고친 뒤 평균 +0.0817. 판정과 무관. 테스트 1개 추가(전체 281개 통과).
- 해석 범위(계획대로): "형식 손상을 줄이고 m에 가깝게 넣어도 이득이 검출되지 않음"까지만 말한다. 다만 형식 손상은 다 없어지지 않았다(재출력 249 대 107, 닫음 묶음의 빈 답). 요소별 기여는 말하지 않는다.
- 다음(작성자 결정 대기): 확인 실험 설계(멈춤 엔트로피 온도·추정 방식 사전 점검 포함), A5.

## 2026-10-17 결과 뒤 탐색: 의미 있는 선택 수는 이미지 크기와 거의 무관 (A4, A6, 정답률 사용 안 함)
- 계기: 확인 실험의 선택 방식을 논의하다가, A4의 "의미 있는 선택 중앙값 70개(m의 13.6%)"가 어디서 오는지 확인했다. 결과를 본 뒤의 서술이며 판정에 쓰지 않는다.
- 측정: 저장된 실행 행의 단계별 `n_meaningful`(정규화 잔차 = 남은 정보 / 관련성이 0.01 이상인 동안 고른 수, `visref/dpp.py` `pick_stats`)과 m. A4 `b_main_hs` 1,966단계, A6 `imgtag_main`의 `visref_imgtag` 2,466단계.
- 결과: 의미 있는 선택 / m 중앙값 A4 0.136, A6 0.048(의미 있는 선택 수 중앙값 70 / 28). 0.8 이상인 단계 A4 179(94문항), A6 0. A4에서 0.95 이상인 79단계(35문항)는 모두 m = 76(이미지 토큰 256개)이었다.
- 이미지 크기별 의미 있는 선택 수 중앙값(m = 76 / 230 / 384 / 537 / 844 / 998): A4 60 / 70 / 55 / 80 / 74.5 / 73, A6 23 / 26.5 / 23 / 28 / 27 / 31. 이미지가 13배 커져도 거의 그대로다.
- 해석(추정): 이미지 쪽 중복이 원인이면 큰 이미지에서 수가 늘어야 하므로, 생각 단계 Z가 이미지 공간에서 가리키는 쓸모 있는 방향이 수십 개뿐이라는 쪽이 더 그럴듯하다. 가능한 이유는 소수 토큰(큰 크기)의 지배나 공통 방향이며 재 보지 않았다(미확인). A6가 더 적은 것은 V도 마지막 층 값이라서일 수 있다(추정).
- 같이 정리한 것: Z는 이번 단계 생각 토큰 전부(최대 1,024개)의 마지막 층 값이고(`visref/methods.py` 247–249줄), 관련성은 rᵢ² = Σₜ (zₜ·vᵢ)²(`visref/dpp.py` 23–24줄). 이미지 쪽 값은 비전 인코더 마지막 층 → `pixel_shuffle` → `mlp1`을 거친 값으로 질문·생각 정보가 없다(대장 X-44).
- 다음: 확인 실험 사전 점검에 "Z의 쓸모 있는 방향 수(고윳값 분포)와 토큰별 크기 분포"를 넣을지 정한다.

## 2026-10-17 결과 뒤 탐색: A6 "빈 답" 16건은 답 뒤 `</think>` 재출력 (정답률은 서술로만)
- 확인: `visref_imgtag`에서 `\nFinal Answer:` 뒤가 비었다고 센 16행의 `final_raw`를 보니, 16건 모두 답을 먼저 쓰고(예: " 10\n</think>") 그 뒤에 `</think>`를 다시 냈다. 결과 전에 정한 규칙은 마지막 `</think>` 뒤 텍스트만 채점하므로 빈 답으로 처리됐다. 15건은 첫 단계를 스스로 닫은 문항, 15건은 1단계에서 멈춤(1건은 3단계).
- 서술(판정에 쓰지 않음): `</think>` 앞 텍스트를 채점하면 16건 중 6건이 맞고, 같은 문항에서 같은 답 문구 TSR이 맞힌 6건과 일치한다. 주 차이는 −1.0에서 −0.4 pp가 되지만 판정 ③은 그대로다.
- 의미: 이 16건(순손실 6문항)은 `</think>` 재출력 가운데서도 "답을 쓴 뒤 `</think>`를 내는" 형식이 채점 규칙과 겹쳐 손실로 이어진 경우다.

## 2026-10-17 추가 관찰(판정에 쓰지 않음): 답 안 `</think>` 재출력은 어디서 많이 나왔나
- 계기: 동작 점검의 표본과 기준을 정하려고, 이미 끝난 실행 행에서 답 안에 모델이 `</think>`를 다시 낸 행(`final_y_start` > 0)을 저장된 첫 단계의 끝 방식별로 셌다. 정답률은 보지 않았다.
- A6(1,000문항): imgtag 249건(첫 단계가 잘린 641문항에서 210, 스스로 닫은 334문항에서 39, EOS 25문항에서 0), 같은 답 문구 TSR 107건(92 / 15 / 0), TSR 113건(95 / 18 / 0). imgtag가 TSR보다 많은 142건 중 118건이 잘린 문항이다.
- 본 실험 VisRef 465건(306 / 159 / 0), 본 실험 TSR 121건(95 / 26 / 0). A4(VisRef-hs, 839문항, 첫 단계를 새로 생성해 끝 방식 구성이 다름) 395건.
- seed 0 순서 앞 200문항(잘림 131, 닫힘 64, EOS 5): A6 imgtag 38건, 같은 답 문구 TSR 21건, TSR 21건.
- 재출력의 모양(A4, 스스로 닫은 첫 단계 뒤 바로 답한 예): 모델이 짧게 정리하며 답을 한 번 쓰고(예: "… is 8. Final answer: 8"), `</think>`를 낸 뒤 정식 답을 다시 쓴다. 태그 없는 이미지 토큰을 생각 끝에 넣고 `</think>`를 다시 붙이는 모양이 학습 때 없던 형식이라, 모델이 그 `</think>`를 생각의 끝으로 받아들이지 않은 것으로 추정한다(확인 안 됨). A4는 답이 자유 형식이라 두 번째 답이 채점되어 손해가 작았고, A6는 `Final Answer:` 뒤에 답을 쓰고 `</think>`를 낸 뒤 아무것도 쓰지 않아 빈 답이 생겼다.

## 2026-10-17 동작 점검 계획 (정답률 사용 안 함)
- 질문: 고른 이미지 블록을 답할 때도 생각 안(`</think>` 앞)에 두면, 답 안 `</think>` 재출력이 A6보다 줄어드는가.
- 바꾸는 것 하나: 답할 때의 기록을 `…생각\n<img>고른 이미지 토큰</img>\n</think>\nFinal Answer:`로 한다(A6: `…생각</think>\n<img>고른 이미지 토큰</img>\nFinal Answer:`). 그러면 멈춤 판단, 답, 생각 이어가기가 모두 같은 기록에서 출발한다(논문 Algorithm 1의 10, 12, 17행). 생각을 이어 갈 때의 기록과 그 밖의 설정은 A6와 같다(태그와 줄바꿈, 모든 답은 `\nFinal Answer:` 뒤, 단계당 1,024토큰, 총 10,240토큰, 최대 10단계, 멈춤 기준 0.25, m = 이미지 토큰의 30%, 온도 0.6, seed 42, fp16, 저장된 첫 단계에서 출발).
- 고르는 방식(작성자 결정): Z = 이번 단계 생각 토큰 각각의 마지막 층 값, V = 언어 모델에 들어가는 이미지 값(A4와 같음). 다시 넣는 값도 언어 모델에 들어가는 이미지 값이다. 고를 수 있는 수는 생각 토큰 수를 넘지 못하므로, 생각이 m보다 짧은 단계는 m을 다 채우지 못한다(논문 식의 한계, 채우기는 하지 않음). 참고로 앞 200문항 중 저장된 첫 단계가 m보다 짧은 문항 수는 실행 뒤 기록한다.
- 메서드 둘: `visref_hs_imgtag_in`(위 형식)과 같은 답 문구의 TSR(`tsr_answer_prompt`), 같은 실행에서 짝지어 돌린다.
- 문항: seed 0 순서 앞 200문항(`diag/format_check_ids.json`, `diag/format_check_ids.py`로 다시 만들 수 있음, sha256 0c43242bfc11…). 순서만 보고 고른다.
- 세는 것: 답 안에서 모델이 `</think>`를 다시 낸 문항 수. 빈 답(`\nFinal Answer:` 뒤가 빔)과 질문 다시 적음(`Question:`)은 기록만 한다.
- 통과 기준(실행 전 고정): 새 형식의 재출력 건수 − TSR의 재출력 건수 ≤ 8. 8은 같은 200문항에서 A6가 TSR보다 많았던 17건(38 − 21)의 절반이다. 즉 "A6보다 형식 손상이 절반 이하로 줄었나"를 본다.
- 결정 규칙: 통과하면 1,000문항 확인 실험을 설계한다. 통과하지 못하면 새 실행 없이 정리한다.
- 추가 관찰 계획(판정에 쓰지 않음, 결과 전에 정함, 2026-10-17 작성자 요청): 정답률은 두 실행 모두 같은 방식으로 다시 채점한다(규칙 채점 v4 + 빈 답 규칙, LLM 추출 없음; 규칙이 답을 못 뽑은 행은 오답이며 메서드별 건수를 함께 적음). (1) 기기 차이: 이번 실행의 같은 답 문구 TSR과 같은 200문항 A6의 같은 답 문구 TSR을 비교한다. 메서드·저장된 첫 단계·seed가 같고 GPU·소프트웨어·묶음 처리만 다르므로, 기기만 바꿔도 생기는 답 차이와 정답률 차이의 크기다. (2) 새 형식과 A6 VisRef(`visref_imgtag`): 답이 다른 문항 수, 둘 중 하나만 맞은 문항 수, 정답률 차이. (1)보다 확실히 클 때만 형식·고르는 방식 묶음의 영향으로 읽는다. (3) 같은 실행 안의 새 형식 − TSR, 같은 200문항 A6의 imgtag − 같은 답 문구 TSR. (4) 재출력 행과 아닌 행의 정답 수. (5) 첫 단계 끝 방식(잘림 131, 닫힘 64, EOS 5)별 재출력과 정답 수. (6) 단계 수, 멈춘 이유, 생성 토큰, 멈춤 엔트로피 사분위, 넣은 수 / m, 의미 있는 선택 수. 200문항이라 정답률 차이의 95% 범위는 대략 ±4%p로 넓어 결론을 내지 않는다. 계산: `diag/analyze_format_check.py --observe`(결과 전 작성·테스트; 같은 행을 두 실행으로 넣으면 차이 0이 나오는지 확인).
- 어텐션 점검과 Z 방향 수 점검: 보류(작성자 결정). 나중 계산을 위해 단계별 고른 이미지 토큰 번호(`sel`), 생각 토큰 id(`z_ids`), 최종 답 토큰 id(`final_ids`, 답 강제 이어쓰기는 `final_af_ids`)를 저장한다.
- 코드: 메서드 `visref/methods.py`(기존 메서드 동작은 그대로), 실행기 `runpod/run_imgtag.py --plan format_check`(정답률 요약은 만들지 않음, 본 실행 이름 `fmtcheck_main`, 3문항 확인 실행은 `fmtcheck_try --limit 3`), 분석 `diag/analyze_format_check.py`(실행 전 작성·테스트), code_version 06435ae748c8. A6 설정(`--plan imgtag_main`)은 A6 커밋 8f5c09e에서만 돈다. 실행 장소: T4(본 실험과 같은 GPU 종류). RunPod은 A5000 재고가 없었고, RTX 3090 커뮤니티 pod 두 대가 모두 모델을 GPU에 올리는 순간 `CUDA unknown error`로 멈춰(결과 행 0, 본 실행은 시작 안 됨) 작성자 결정으로 옮겼다. 3문항 확인 실행 `fmtcheck_try_kg`을 올렸으나 1시간 넘게 Kaggle 대기열에서 시작되지 않아, 본 실행은 작성자 결정으로 Google Colab T4 한 장에서 돌린다(`repro/colab/runs/fmtcheck_main_cb`, 2026-10-17 14:44 UTC 시작). 노트북은 GitHub 커밋 3d63863을 받아 파일 해시를 확인한 뒤 같은 run.py 인수로 실행하고(code_version 06435ae748c8), 결과를 Google Drive에 1분마다 저장한다. Kaggle 본 실행 `fmtcheck_main_kg`는 올리지 않는다. A6 기준값(17건)은 A5000 실행에서 나온 값이므로, 이번 TSR 재출력 건수와 같은 200문항 A6 TSR 건수(21)를 함께 기록한다.
- 계획이 바뀐 과정: 처음에는 비교군으로 "A6 형식, 같은 고르는 방식"(`visref_hs_imgtag_out`, 코드에는 남아 있음)을 함께 두고, 첫 단계를 스스로 닫은 100문항에서 재출력·빈 답·질문 다시 적음 세 건수를 TSR의 1.5배 기준으로 보려 했다. 같은 100문항에서 A6를 이 기준에 대 보니 경계에 걸려 구분이 안 됐고, 재출력은 주로 첫 단계가 잘린 문항에서 나와(위 추가 관찰) 표본과 기준을 위처럼 바꿨다(작성자 결정, 2026-10-17). 비교군은 결정에 필요 없어 뺐다. 처음 계획에 대한 독립 검토(2026-10-17)의 지적 가운데 새 메서드가 A6 커널용 값을 요구하지 않게 분리, 분석용 기록 저장은 반영했다.

## 2026-10-18 진단 계획: 멈춤 기준이 VisRef에 기회를 주는가, 고른 이미지가 답을 바꾸는가 (일부 결과를 본 뒤 정함)
- 목표(작성자 결정): 논문 수치 맞추기가 아니라 VisRef가 TSR보다 이득을 내는 조건 찾기. 1,000문항 확인 실험 전에 값싼 진단을 먼저 한다(작성자 동의).
- 계기: 형식 점검이 통과해도 정답률 차이가 생긴다는 보장이 없다. 이미지를 다시 넣어도 예측이 거의 그대로이고(스스로 닫은 329문항 중 328), VisRef가 첫 단계에서 멈춘 문항이 71.3%라 다시 넣은 이미지가 대부분 답 바로 앞에만 온다.
- 멈춤 기준 관찰(b_main, 정답률 사용 안 함): 첫 단계 유효 probe TSR 997 / VisRef 997. 첫 단계 H가 0.15 이상 0.25 미만인 문항 29 / 32, 0.25 이상 0.30 이하 17 / 13. δ를 0.15나 0.30으로 바꾸면 첫 결정이 바뀌는 문항은 약 3% / 1.5%. 논문은 H를 기대값으로만 정의하고 추정 방식과 y의 범위를 적지 않았으며(VisRef_layout.txt 374–376줄), δ를 바꿔도 정답률이 "relatively stable"하다고 쓴다(556줄). 그래서 말할 수 있는 것은 "우리 H는 δ 근처에 거의 없어 δ 변화에 둔하다"까지다.
- 진단 A(GPU 없음, 판정 규칙 없는 서술): 첫 단계가 잘린 641문항에서 방법별(TSR, VisRef 따로)·문제 유형별(객관식, 자유형)로 "첫 단계에서 멈춤 여부 × probe 답 정답 여부" 표를 만든다. TSR은 ST 최종 답 정답 여부로도 만든다(TSR이 첫 단계에서 멈추면 ST와 같은 답을 뽑도록 설계됨). 멈췄지만 틀린 문항 수를 1,000문항 기준 pp로 바꿔 "더 생각하게 했을 때 얻을 수 있는 최대 이득"으로 적고, 다음 실험의 문항 수 근거로 쓴다. 첫 단계 H로 probe 답 오답을 가르는 AUROC는 참고로만 적는다. probe 답 정답 여부는 2026-10-11에 끝 유형별로 이미 채점해 보았으므로 이 분석은 "결과를 본 뒤의 서술"이다. δ를 이 1,000문항으로 고르지 않는다.
- 진단 B(Colab 순전파, 우선순위 낮음, 형식 점검 결과를 받은 뒤): 형식 점검 200문항 중 첫 단계에서 멈춘 문항에서, 같은 실행 TSR의 답을 강제로 넣고 이미지를 넣지 않았을 때와 비교한 답 분포 변화(KL, 답 앞부분)를 고른 조각과 무작위 조각 8개에 대해 잰다. 고른 조각의 변화가 무작위 8개보다 모두 큰 문항의 비율을 우연 기대값 1/9과 비교한다. 형식 점검 변형만 보므로 본 실험 VisRef의 결과를 설명하는 데 옮겨 쓰지 않는다.
- 뺀 것: AUROC 문턱으로 다음 행동을 고르는 규칙(어느 결과든 거의 정해져 있음), "틀린 답 중 첫 단계에서 멈춘 비율 > 1/2" 기준(H가 아무 정보도 없을 때의 멈춤 비율 59%보다 낮아 늘 통과), VisRef에 ST 정답을 대신 쓰는 것, 난수만 바꾼 잡음 바닥(강제 입력 순전파는 결정적), 주의 비율 측정.
- 검토: 독립 검토 두 번(2026-10-18). 검토자가 센 수치는 주 세션이 원자료로 다시 셌다.

## 2026-10-18 진단 A 결과: 멈춤 기준은 오답을 가르지만, 스스로 닫은 문항에서는 거의 늘 멈춘다 (서술, 판정 없음)
- 계산: `repro/diag/step1_stop_table.py` → `diag/step1_stop_table.json`(테스트 `tests/test_step1_stop_table.py`). probe 답은 규칙 채점 v4(`score('\boxed' + probe_answer)`, 무효 probe는 오답), ST 최종 답은 본 실험 판정(`llm/final/merged.jsonl`). probe 정답 수는 2026-10-11 분석(`diag/step1_probe_analysis.json`)과 같다(잘림 TSR 315, VisRef 331, 닫음 240).
- 첫 단계가 잘린 641문항: 첫 단계에서 멈춤 TSR 380 / VisRef 366. 멈췄지만 probe 답이 틀린 문항 140 / 130(1,000문항 기준 14.0 / 13.0 pp), TSR은 ST 최종 답도 틀린 문항 132(13.2 pp). 첫 단계 H로 probe 오답을 가르는 AUROC 0.74(95% CI 0.71–0.78) / 0.72(0.68–0.75). 객관식 0.74 / 0.71, 자유형 0.74 / 0.70.
- 스스로 닫은 334문항: 첫 단계에서 멈춤 330 / 329. 멈췄지만 probe 답이 틀린 문항 92 / 91(9.2 / 9.1 pp). AUROC 0.66(0.60–0.73) / 0.70(0.64–0.76).
- EOS 25문항: 멈춤 20 / 18, 멈췄지만 틀림 7 / 6. 표본이 작아 해석하지 않는다.
- 서술: H는 어느 묶음에서나 오답을 어느 정도 가른다. 그런데 스스로 닫은 문항에서는 H가 δ보다 훨씬 작아 330/334가 첫 단계에서 멈추고, 틀린 92문항도 더 생각할 기회를 받지 못한다. 첫 단계에서 멈춘 오답은 모두 합쳐 1,000문항 중 약 23%(TSR 기준 140 + 92 + 7)다. 이것은 "더 생각하게 하면 얻을 수 있는 최대치"이고, 이미지 효과의 크기가 아니다. 이미지를 넣고 더 생각하는 효과는 step1_branch(D − C)에서 309문항으로 잰 적이 있지만, 그중 스스로 닫은 문항은 5개뿐이었다.

## 2026-10-18 실행 전 계획: 첫 단계에서 멈춘 문항을 더 생각하게 하기 `step1_think` (결과 전에 고정)
- 질문: 본 실험에서 TSR과 VisRef가 모두 첫 단계에서 멈춘 문항에서, 생각을 더 이어가게 할 때 고른 이미지를 다시 넣으면 정답이 늘어나는가. 이 문항들에서는 이미지를 넣고 더 생각하는 효과를 잰 적이 없다(step1_branch는 이어간 309문항, 그중 스스로 닫은 문항 5개).
- 순서를 바꾼 이유(독립 검토, 2026-10-18; 인용은 원문과 대조함): 이 시험은 멈춤 결정과 상관없이 생각을 이어가므로 H 추정 방식에 영향을 받지 않는다. 비용이 1,000문항 전체 실행보다 훨씬 작다(step1_branch: 309문항 × 4조건 × 표본 3개, T4 두 장 GPU당 약 2,918초). H 추정 방식 점검은 뒤로 미루고 형식 점검 200문항 서술로 줄인다. 진단 B(순전파 KL)는 아래 무작위 조건이 같은 질문에 정답률로 답하므로 뺀다.
- 문항: 본 실험에서 `tsr`과 `visref_raw_rank`가 모두 첫 단계에서 멈춘 691문항(첫 단계 끝: 스스로 닫음 329, 잘림 344, EOS 18). seed 0 순서로 늘어놓아, 도중에 끊겨도 앞부분이 무작위 표본이 되게 한다. 문항은 멈춤 결정으로만 골랐다(정답 여부 사용 안 함).
- 조건(저장된 첫 단계 z_1에서 출발, 세 조건이 같은 생각 난수를 씀):
  - C: z_1 + "Wait, think more" → 최대 512토큰 더 생각 → 답
  - D: z_1 + 고른 이미지 토큰 + "Wait, think more" → 같은 방식
  - R: D와 같은 수의 무작위 이미지 토큰(문항마다 고정 난수, 번호 순 정렬)을 D와 같은 자리·같은 형식으로 넣음
  - 이미지를 넣는 방식: 형식 점검이 통과하면 그 방식(Z = 생각 토큰의 마지막 층 값, V = 언어 모델에 들어가는 이미지 임베딩, `<img>…</img>` 태그로 감싸 생각 안에 넣음), 통과하지 못하면 본 실험 방식(Z = 생각 토큰의 입력 임베딩, 태그 없음). 형식 점검은 정답률을 보지 않는 판정이다. "Wait, think more"를 넣는 자리는 본 메서드와 같다(생각 뒤, 이미지 뒤).
  - 답: step1_branch와 같이 생각을 닫은 뒤 `\n\n\boxed` 짧은 답(최대 16토큰), 온도 1 표본 3개와 greedy 1개. 문항 정답률 = 표본 3개 평균, 규칙 채점 v4(답이 무효면 오답).
- 주 비교(판정): D − C를 스스로 닫은 묶음과 잘린 묶음에서 따로 계산한다. 문항 단위 짝 부트스트랩(20,000회, 고정 seed) 95% CI와 양쪽 p값, 두 묶음에 Holm 보정.
  - Holm 보정 p < 0.05이고 CI 하한 > 0인 묶음이 있으면: "그 묶음에서 이미지를 넣고 더 생각하면 이득(탐색 결과)". 같은 1,000문항에서 탐색을 여러 번 했으므로, 다음에 1,000문항 `visref_think` 대 `tsr_think`로 확인한다.
  - 그렇지 않으면: 이득 확인 안 됨. CI 상한이 +3.5 pp 미만이면 "그 묶음에서 +3.5 pp 크기 이득 없음"까지 적고, 새 GPU 실행 없이 정리 쪽으로 간다.
  - 검출할 수 있는 크기: 묶음당 약 330문항이면 95% CI 반폭 약 3 pp(step1_branch 309문항의 D − C 반폭 3.2 pp)라, 약 5 pp 이상의 효과만 잡힌다.
- 보조(서술, 판정에 쓰지 않음): D − R(고른 조각 대 무작위 조각, 논문 Table 7의 무작위 선택 비교에 해당), R − C, 691문항 전체, EOS 18문항, greedy 답. 조건별로 더 생각한 글의 끝 방식(스스로 닫음 / EOS / 512 상한), 길이, 유효한 답 비율을 적는다(더 생각은 첫 `</think>`에서 멈추므로 글 안에 `</think>`가 다시 나오는 일은 구조상 없고, 스스로 닫은 비율로 대신 본다). 유효한 답 비율이 95% 미만인 조건이 있으면 판정과 함께 적는다.
- 코드: 새 스크립트 `diag/step1_think.py`(기존 `diag/step1_branch.py`는 끝난 실험의 기록이라 고치지 않음), 문항 목록 `diag/step1_think_ids.json`, 분석 `diag/analyze_step1_think.py`. 모두 결과 전에 작성·테스트한다. 실행 장소와 시간은 형식 점검 결과 뒤에 적는다.
- 8문항 시험 실행(2026-10-17 17:24 UTC push): `step1_think_smoke`(Kaggle T4 두 장, 문항 순서 앞 8개, `--inject hs_tag_in`, 빌더 `kaggle/build_step1_think.py`). 형식 점검이 통과한다고 보고 미리 돌리는 코드 점검이며(작성자 결정), 형식 점검이 통과하지 못하면 본 실행은 `--inject bmain`으로 한다. 보는 것: 첫 단계 재구성 일치, 고른 수와 m, C·D·R의 더 생각한 글 끝 방식, 유효한 답 비율, 오류. 정답률은 보지 않는다.
- 8문항 시험 결과(정답률은 보지 않음): 8행 오류 0, T4 두 장, 실행 140초(준비 44초 별도). 첫 단계 재구성 일치 8/8. 고른 수/m은 100/998, 992/998, 76/537, 305/384, 537/537, 76/76, 76/76, 164/537(생각이 짧으면 m을 다 채우지 못함, 계획에 적은 한계). 더 생각한 글의 끝은 C 스스로 닫음 5·512 상한 3, D와 R은 모두 스스로 닫음. 유효한 답은 모든 조건에서 문항당 4/4. 최대 메모리 10.56 GB.
- 본 실행(2026-10-17 17:29 UTC push): `step1_think_main`(Kaggle T4 두 장, 691문항, `--inject hs_tag_in`, 제한 5시간). 형식 점검 결과를 기다리지 않고 먼저 돌린다(작성자 결정). 형식 점검 판정이 나오기 전에는 이 실행의 결과를 열지 않는다. 형식 점검이 통과하지 못하면 계획대로 `--inject bmain`을 따로 돌려 그 결과로 판정하고, 이 실행은 보조로만 적는다.
- 형식 점검 중간 확인(2026-10-17 17:54 UTC, 작성자 요청, 계획에 없던 확인): 400행 중 191행, 두 메서드가 모두 끝난 92문항에서 오류·중복·깨진 줄 0. 답 안 `</think>` 재출력은 새 형식 23 / 같은 답 문구 TSR 10(차이 13). 같은 92문항에서 A6는 imgtag 16 / 같은 답 문구 TSR 9. 새 형식의 재출력은 스스로 닫은 첫 단계 9 / 잘린 첫 단계 14(A6 imgtag 2 / 14). 정답률은 보지 않았다. 판정은 200문항이 끝난 뒤 미리 정한 규칙으로만 하고, 이 확인 때문에 실행을 멈추거나 바꾸지 않았다.
- b_main 방식 실행(2026-10-17 18:08 UTC push, 작성자 결정): 중간 확인에서 형식 점검이 통과하기 어려워 보여, 계획에 적은 대체 방식 `step1_think_bmain`(Kaggle T4 두 장, 691문항, `--inject bmain`, 제한 5시간)을 미리 올렸다. 형식 점검이 통과하지 못하면 이 실행이 주 판정이고 `step1_think_main`(형식 점검 방식)은 보조다. 통과하면 반대다. 두 실행 모두 형식 점검 판정 전에는 결과를 열지 않는다.
