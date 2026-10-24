# visref-repro

[English](README.md) · 자세한 내용은 영어 보고서 **[REPORT.md](REPORT.md)**가 기준입니다.

**VisRef**(Ghosal 외, *VisRef: Visual Refocusing while Thinking Improves Test-Time Scaling in Multi-Modal Large
Reasoning Models*, arXiv:2603.00207, CVPR 2026)의 비공식 재현입니다. 논문 부록 Table 6의 *InternVL-1B* 행을
InternVL3.5-1B 모델과 MathVista *testmini* 1,000문항으로 다시 실험했습니다. 세 방법을 비교합니다.

- **ST** (standard thinking, 표준 생각): 한 번 생각하고 바로 답합니다.
- **TSR** (textual self-reflection, 텍스트 자기 성찰): 생각이 끝날 때마다 "Wait, think more"를 붙여, 답이 충분히 확실해질 때까지 다시 생각하게 합니다.
- **VisRef**: TSR과 같지만, 다시 생각하기 전에 이미지의 일부를 골라 reinject(다시 넣기)해 줍니다. 논문이 제안한 방법입니다.

> **상태 (2026-10-25):** 본 실험과 후속 실험 14개(A2–A4, A6–A16)가 끝났습니다. 원래 그림이 있는 문맥에서는 어떤 변형도 VisRef가 TSR보다 나은 것을 보이지 않았습니다(A1–A11). 프롬프트의 그림 자리가 회색 빈 그림일 때, 첫 thinking step 뒤에 이 문항 그림 전체를 다시 넣으면 정답률이 12.8%p 올랐고, 다른 문항 그림에서는 정답률 변화를 확인하지 못했습니다(A12, [REPORT 4.16절](REPORT.md#416-blank_think-does-a-re-inserted-image-raise-accuracy-when-the-image-slot-is-blank-rule-fixed-before-the-results)). 즉 그림 정보가 빠졌을 때는 다시 넣은 visual token이 생성에 쓰입니다. 같은 문맥에서 DPP 선택으로 고른 토큰만 넣어도 정답률이 6.9%p 올랐지만(좋아짐 99, 나빠짐 40), 같은 수의 무작위 토큰과의 차이는 확인하지 못했습니다(+2.0%p, 95% 신뢰구간 −0.9 ~ +4.8, 84 대 66; A16, [REPORT 4.20절](REPORT.md#420-sel_think-with-a-blank-image-slot-do-the-dpp-selected-image-tokens-raise-accuracy-rule-fixed-before-the-results)). A13–A15(생각 뒤 그림 자리, 단계 상한 4,096, 자유 답으로 잰 멈춤 엔트로피)는 이 그림을 바꾸지 않았습니다([REPORT 4.17–4.19절](REPORT.md#417-slot_cross-after-generated-thinking-does-the-image-slot-of-the-prompt-still-matter-rule-fixed-before-the-results)). 지금까지 하지 않은 실험과 그 이유는 [REPORT 5.4절](REPORT.md#54-experiments-not-run)에 있습니다.

## 읽는 순서

| | 언제 | 읽을 것 |
|---|---|---|
| 1 | 처음: 결과와 주요 한계 | 이 README |
| 2 | 무엇을 시험했고 무엇을 결론 내렸는지 | [REPORT.md](REPORT.md)(영어): 요약과 1.1절(모든 실험을 한 표에), 이어서 3–4절(결과, 원인 분석)과 6절(한계) |
| 3 | 논문에 적히지 않은 설정을 어떻게 정했는지 | REPORT 2.1절·2.6절. 결정마다의 근거: [docs/IMPLEMENTATION_CHOICES.md](docs/IMPLEMENTATION_CHOICES.md)(맨 앞 요약 표부터) |
| 4 | 언제, 왜 그렇게 정했는지 | [docs/EXPERIMENT_FLOW.md](docs/EXPERIMENT_FLOW.md)(날짜순 기록, 맨 앞 단계별 요약부터) |
| 5 | 숫자를 확인할 때 | 표는 [`results/`](results), 모델 원본 출력은 `repro/kaggle/runs/`, 다시 계산은 `make check`(아래 "재현") |
| 6 | 줄 번호로 인용된 구절(예: `LAY L900` = 논문 텍스트 사본의 900번째 줄)을 확인할 때 | [docs/SOURCES.md](docs/SOURCES.md)(영어) |
| 7 | 다시 실행하거나 코드를 읽을 때 | [docs/REPRODUCE.md](docs/REPRODUCE.md)(영어) |

파일 위치 전체: [REPORT 7절](REPORT.md#7-where-to-find-things).

## 결과

1,000문항, 문항마다 각 방법을 한 번씩 실행했습니다. 세 방법은 같은 난수와 같은 첫 thinking step(생각 단계)을 공유합니다. 답은 규칙으로
뽑고, 규칙이 실패한 행(3,000행 중 36행)만 LLM으로 뽑았습니다. LLM의 답은 응답 원문에 있을 때만 인정합니다.

<!-- check:accuracy -->
| 정확도 (%) | ST | TSR | VisRef |
|---|---|---|---|
| 논문 (Table 6) | 46.1 | 48.5 | 52.0 |
| **우리** | **57.8** | **60.1** | **60.0** |
<!-- /check -->

<!-- check:paired -->
| 비교 | 우리, %p (95% 신뢰구간) | 논문, %p | 판정 |
|---|---|---|---|
| VisRef − ST (주 비교) | +2.2 (+0.3, +4.1) | +5.9 | 방향은 같으나 효과 크기 재현 안 됨 |
| VisRef − TSR (주 비교) | −0.1 (−2.0, +1.7) | +3.5 | 논문 크기의 효과 재현 안 됨 |
| TSR − ST (보조) | +2.3 (+0.6, +4.0) | +2.4 | 논문 값과 어긋나지 않음 |
<!-- /check -->

- **요약:** TSR이 ST보다 나은 정도(+2.3%p)는 논문의 +2.4와 어긋나지 않지만, 우리 실험에서는 그 일부가 토큰 상한에서 잘린 생각을
  이어 간 데서 나온 것으로 보입니다. VisRef가 TSR보다 더 나은 효과는 우리 구현에서 확인하지 못했습니다.
  VisRef − ST는 논문보다 작고, 미리 정한 두 비교의 보정(Holm)을 하면 유의하지 않습니다(p = 0.067).

> **이번 실행이 시험한 조건**
> - 첫 thinking step이 1,000문항 중 641문항에서 토큰 상한(1,024)에 걸려 잘렸습니다. 그래서 추가 단계는 대부분 잘린 단계를 이어 가는 것입니다.
> - 모델이 스스로 닫은 단계 뒤에는 entropy H_k(stopping entropy, 멈춤 엔트로피)가 기준값(0.25) 이상이었던 경우가 TSR 359건 중 7건, VisRef 398건 중 35건뿐입니다. 즉 이어
>   생각하기는 거의 잘린 뒤에만 일어났습니다.
> - reinject한 visual token은 논문이 thinking step마다 넣는 수(m)의 중앙값 약 1/5입니다(중앙값 0.199).
> - VisRef가 한 단계 만에 멈춘 713문항에서, 최종 답에 `</think>`가 또 들어 있는 경우가 VisRef 352건, 같은 문항의 ST는 67건입니다.
>
> 이 한계 때문에 후속 실험 A2–A11이 있습니다. A6는 마지막 두 가지를 함께 다뤘습니다. 마지막 층 값으로 고른 visual token m개를
> InternVL의 이미지 태그로 감싸 `</think>` 뒤에 넣었고, 같은 답 문구의 TSR보다 −1.0%p(95% 신뢰구간 −3.1 ~ +1.0)였습니다.
> 최종 답의 `</think>` 재출력은 1,000문항 중 249건으로 여전히 남았습니다(같은 답 문구의 TSR은 107건)
> ([REPORT 4.6절](REPORT.md#46-visref-imgtag-tagged-image-block-after-think-rule-fixed-before-the-results)).
> A7에서는 두 방법이 모두 첫 thinking step에서 멈춘 691문항에서, 고른 이미지를 넣거나 넣지 않고 최대 512토큰 더 생각하게 했습니다.
> 이미지 있음 − 없음은 두 묶음에서 0.0%p, −1.74%p로 이득이 없었습니다([REPORT 4.7절](REPORT.md#47-step1_think-more-thinking-where-both-methods-stopped-rule-fixed-before-the-results)).
> A8에서는 첫 thinking step이 잘리고 한 방법이 이어 생각한 297문항에서, 이 문제의 visual token을 전부 넣거나 넣지 않고 최대 2,048토큰 더 생각하게 했습니다.
> 이미지 전체 − 이미지 없음은 −1.80%p(95% 신뢰구간 −5.95 ~ +2.36, 좋아진 문항 31개, 나빠진 문항 45개)로, 이득을 확인하지 못했습니다([REPORT 4.12절](REPORT.md#412-full_think-the-whole-image-and-longer-thinking-after-a-cut-step-1-rule-fixed-before-the-results)).
> A11에서는 모델이 첫 thinking step을 스스로 닫은 334문항에서, 단계마다 고른 visual token을 넣거나 넣지 않고 4단계까지 더 생각하게 했습니다.
> 앞 112문항에서 VisRef − TSR과 VisRef − 바로 답 모두 바뀐 문항이 0개(좋아짐 0, 나빠짐 0, 네 조건 모두 75.0%)라 여기서도 이득을 확인하지 못했고, 미리 정한 중간 멈춤 규칙으로 112문항에서 멈췄습니다([REPORT 4.15절](REPORT.md#415-closed_multistep-continuing-up-to-4-steps-after-a-first-step-the-model-closed-itself-rule-fixed-before-the-results)).
> 정답 확률만 읽은 두 시험도 그림을 바꾸지 않았습니다. 처음 그림을 회색 빈 그림으로 바꾸면 다시 넣은 이 문제의 그림이 정답 확률을 일부 되돌렸고(A9, 주로 자유형 문항), 썸네일 조각을 빼도 다시 넣은 그림의 정답 확률 이득이 커진다는 근거는 보이지 않았습니다(A10)([REPORT 4.13·4.14절](REPORT.md#413-recover_check-does-a-re-inserted-image-carry-information-when-the-image-slot-is-blank-rule-fixed-before-the-results)).
> 짝지은 실행 세 개를 미리 정한 16개 문항 묶음으로 나눠 봐도, VisRef가 TSR보다 일관되게 나은 묶음은 없었습니다([REPORT 4.9절](REPORT.md#49-subgroup-reanalysis-of-the-three-paired-runs-grouping-fixed-before-it-was-computed)).

- **판정 기준:** 95% 신뢰구간이 0을 빼고 논문 값을 포함하면 "어긋나지 않음", 논문 값을 빼면 "재현 안 됨", 0과 논문 값을 모두 포함하면 판단하지 않습니다. 판정 규칙은 중간
  결과(392문항, 759문항)를 본 뒤 정했으므로 결과 전에 정한 규칙이라고 하기는 어렵습니다.
- **1,000문항으로 가려낼 수 있는 크기:** 짝지은 차이가 약 2.6%p 이상이면 80% 확률로 잡힙니다. 그래서 논문 크기(+3.5%p)의 이득은 있었다면
  보였을 것이고, 1~2%p의 작은 이득은 있는지 없는지 가릴 수 없습니다. 같은 조건을 다시 돌리면 정답률이 약 0.3~0.5%p 달라졌습니다([REPORT 3.4절](REPORT.md#34-what-these-samples-can-detect)).

## 실험 한눈에 보기

**결과 실험**은 논문의 주장을 검증하거나, 재현되지 않은 이유를 따진 실험입니다(판정 규칙은 결과 전에 정함). 논문에 없는
설정을 정한 **설정 시험**과 각 설정의 출처(논문 / 인용 문헌 / InternVL 공식 자료 / 우리 선택)는
[REPORT 2.6절](REPORT.md#26-setting-checks-how-the-unstated-settings-were-chosen)에 있습니다.

<!-- check:glance -->
| | 질문 | 본 실험(A1)에서 바꾼 것 | 문항 수 | 결과 | 가장 큰 한계 |
|---|---|---|---|---|---|
| A1 | VisRef(랭크 한계에서 일찍 멈추는 선택)가 TSR보다 나은가? (논문 +3.5%p) | — (본 실험, 설정은 REPORT 2절) | 1,000 | VisRef − TSR: −0.1%p | reinject한 visual token이 논문의 약 1/5 |
| A2 | visual token을 m개까지 채우면? | 선택이 멈춘 뒤 reinject하는 토큰을 m개까지 채움 | 976 | VisRef-fill − TSR: 0.0%p | 채운 토큰이 생각과 거의 무관 |
| A3 | 첫 thinking step 뒤 이미지를 reinject하는 것만으로 효과가 있나? | 같은 첫 thinking step 뒤에만: 고른 visual token을 넣은 경우와 안 넣은 경우로 답함(바로, 또는 최대 512토큰 더 생각한 뒤) | 309 | 넣음 − 안 넣음: +1.40%p, +0.97%p | 2단계 이상 이어간 문항만 대상 |
| A4 | visual token을 모델 마지막 층 값으로 고르면? | 고를 때 쓰는 생각 벡터 Z를 마지막 층 값으로(A1: 입력 임베딩) | 839 | VisRef-hs − TSR: +0.24%p | 첫 thinking step이 본 실험과 달라짐 |
| A5 | 이미지를 넣은 뒤 더 생각하게 하면? | 멈춘다고 판단한 뒤, 넣은 이미지 다음에 최대 512토큰 더 생각하고 답함 | – | 하지 않음(A7 전에 정한 규칙대로) | 같은 질문을 A7에서 일부 문항으로 시험함 |
| A6 | 고른 visual token을 태그로 감싸 `</think>` 뒤에 넣으면(마지막 층 값으로 고름) 같은 답 문구의 TSR보다 나은가? | 네 가지를 한꺼번에: (1) `<img>` … `</img>` 태그와 줄바꿈, (2) 답할 때 이미지 블록을 `</think>` 뒤에(A1: 앞에), (3) 고를 때 이미지 벡터 V도 마지막 층 값(Z는 A4와 같음), (4) 모든 답을 `\nFinal Answer:` 뒤에서 시작 | 1,000 | VisRef-imgtag − 답 문구 있는 TSR: −1.0%p | 네 가지를 함께만 시험해 어느 것의 효과인지 모름 |
| A7 | 두 방법이 모두 첫 thinking step에서 멈춘 문항에서, 고른 이미지를 넣고 더 생각하게 하면? | 저장된 첫 thinking step에서 출발해 이미지 없이 / 고른 visual token을 넣고 / 같은 수의 무작위 visual token을 넣고 최대 512토큰 더 생각한 뒤 짧게 답함 | 691 | 이미지 있음 − 없음: 0.0%p(첫 thinking step을 모델이 스스로 닫음), −1.74%p(첫 thinking step이 잘림) | 두 방법이 이미 멈춘 문항만 대상 |
| A8 | 첫 thinking step이 잘리고 한 방법이 이어 생각한 문항에서, 이미지 전체를 넣고 더 길게 생각하게 하면? | 저장된 첫 thinking step에서 출발해 바로 답함 / 이미지 없이 최대 2,048토큰 더 생각 / 이 문제의 visual token을 전부 넣고 최대 2,048토큰 더 생각한 뒤 짧게 답함 | 297 | 이미지 전체 − 이미지 없음: −1.80%p | 잘린 뒤 이어 생각한 문항만 대상; 더 생각하는 것을 강제해 멈춤 판단은 시험하지 못함 |
| A9 | 처음 그림 자리가 회색 빈 그림일 때, 이 문제의 visual token을 다시 넣으면 정답 확률이 돌아오나? | 생성 없음: 프롬프트의 그림을 같은 크기의 회색 빈 그림으로 바꾸고, 짧은 고정 문장 뒤에 이 문제의 visual token을 다시 넣어 정답 확률을 읽음 | 199 | 다시 넣은 그림이 정보를 전달함: 정답 확률 +0.088(95% 신뢰구간 +0.042 ~ +0.136) | 정답률이 아니라 정답 확률; 인위적인 문맥; 회복은 자유형 문항에서 나옴 |
| A10 | 썸네일 조각을 빼면, 고른 visual token을 다시 넣었을 때 정답 확률이 더 오르나? | 입력에서 썸네일 조각을 잘라 내고 첫 thinking step을 새로 쓴 뒤 고른 visual token을 다시 넣어 정답 확률을 읽음 | 160 | 관문 통과 못 함: 두 이득의 차이 +0.006(95% 신뢰구간 −0.011 ~ +0.024) | 정답률이 아니라 정답 확률; 첫 단계 뒤 한 자리만 잼 |
| A11 | 첫 thinking step을 모델이 스스로 닫은 뒤 4단계까지 더 생각하게 하면 VisRef가 TSR보다 나은가? | 저장된 첫 thinking step에서 출발해 바로 답함 / 그림 없이 더 생각 / 단계마다 고른 visual token을 넣고 더 생각; 4단계까지, 멈춤 판단 없음; 짧게 답함 | 334 중 112 | VisRef − TSR: 좋아짐 0, 나빠짐 0(0.0%p) | 단계 수를 강제함; 미리 정한 규칙으로 112문항에서 멈춤 |
| A12 | 그림 자리가 빈 그림일 때, 첫 thinking step 뒤 이 문항 그림을 다시 넣으면 생성을 거친 정답률이 오르나? | 원래 그림 또는 회색 빈 그림으로 첫 thinking step을 새로 쓰고, 그림 없이 / 이 문항 visual token 전부 / 다른 문항 visual token 전부를 넣고 더 생각한 뒤 짧게 답함 | 396 | 빈 그림 + 이 문항 그림 − 빈 그림: +12.8%p | 인위적인 빈 그림 문맥; 그림 전체, 선택 없음 |
| A13 | 생각을 생성한 뒤에도 프롬프트 그림 자리의 그림이 정답 확률을 바꾸나? | 생성 없이 확률만: 그림 자리(원래 / 빈 그림) × 생각 뒤 블록(없음 / 이 그림 / 빈 그림) | 396 | 관문 통과 못 함: +0.022(기준 0.05) | 정답률이 아니라 확률 |
| A14 | 단계 상한을 1,024에서 4,096토큰으로 올리면 ST 정답률이 바뀌나? | 본 실험 코드를 두 상한으로 같은 문항에 실행 | 42 | ST 4,096 − 1,024: 0.0%p | 42문항 |
| A15 | 멈춤 엔트로피를 자유 답 16개로 재면 스스로 끝낸 첫 step이 이어 가나? | 저장된 첫 thinking step에서 짧은 답 16개와 자유 답 16개 | 126 | 정하지 못함: 126 중 3 ~ 94 이어 감, 상자 없이 읽으면 7 ~ 21 | 우리가 정한 추정량 |
| A16 | 그림 자리가 빈 그림일 때, DPP로 고른 visual token만 넣어도 정답률이 오르나? 같은 수의 무작위 토큰보다 더 오르나? | A12의 저장된 빈 그림 첫 thinking step에서 그림 없이 / DPP 선택 / 같은 수 무작위 / 그림 전체 / 다른 문항 그림 전체를 태그 블록으로 넣고 더 생각한 뒤 짧게 답함 | 724 | DPP − 없음: +6.9%p; DPP − 무작위: +2.0%p, 확인 못 함 | 인위적인 빈 그림 문맥 |
<!-- /check -->

## 논문과 결과가 다를 수 있는 이유

먼저 우리 구현의 차이를 의심합니다([REPORT 4.1절](REPORT.md#41-cause-analysis-after-the-results)).

- **visual token 수:** 논문이 정한 수(m)의 약 1/5만 reinject했습니다. 텍스트 토큰을 모델의 입력 임베딩으로 읽으면(우리의 해석이며 논문은 층을 밝히지 않음) 선택 커널의 랭크가 m보다 작고, 논문이 인용한 greedy 선택이 거기서 멈추기 때문입니다. A2, A4, A6에서 이를 피하는 방법을 시험했고, 어느 것도 TSR보다 나은 효과가 검출되지 않았습니다. 이후 후속 실험(A11과 진행 중인 실행)은 2026-10-17 결정대로 생각 글의 마지막 층 값으로 고릅니다.
- **우리가 정한 설정:** 멈추는 시점 계산법, 이미지를 넣는 형식, 토큰 상한, 답 추출 방법은 논문에 없어 작은 시험으로 정했습니다. 이 시험에 쓴 문항은 최종 1,000문항에도 들어 있습니다.
- **높은 정확도:** 우리 정확도가 논문보다 8~12%p 높고(우리 57.8 / 60.1 / 60.0 %, 논문 46.1 / 48.5 / 52.0 %), InternVL3.5-1B 공식 점수에 가깝습니다. 이유는 밝히지 못했습니다.
- **한 가지 조건:** 모델 크기 하나, 벤치마크 하나, 시드 하나입니다.

## 재현

저장된 출력으로 모든 표와 그림을 다시 만듭니다(CPU만, Python 3.10 이상, 표준 라이브러리).

```bash
make fetch     # MathVista testmini.json, 공식 추출 지시문, 인용 코드를 받아 sha256으로 확인
make check     # 분석을 다시 돌려 results/와 그림을 만들고, README와 REPORT의 표가 결과와 같은지 확인
```

GPU로 모델 출력을 다시 만드는 방법, LLM 추출, 저장소 구성: [docs/REPRODUCE.md](docs/REPRODUCE.md) (영어).
실험마다 쓴 GPU와 생성 시간은 [`results/compute.csv`](results/compute.csv)(REPORT 표 2.5)에 있습니다.

## 기여와 AI 사용

AhnJG가 AI의 도움(Anthropic의 Claude 모델, OpenAI의 Codex)을 받아 진행했습니다. 재현 대상, 근거 규칙, 설계 결정은
AhnJG가 정했고, Claude가 코드 대부분과 분석·보고서 초안을 썼습니다. 자세한 내용: [REPORT 8절](REPORT.md#8-contributions-and-ai-use).

## 인용과 라이선스

인용은 `CITATION.cff`를 보고, 원 논문도 함께 인용해 주세요(BibTeX는 [README.md](README.md#citation)).
코드는 MIT, MathVista에서 파생된 데이터 파일은 CC BY-SA 4.0입니다([README.md](README.md#license)).
