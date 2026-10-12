# visref-repro (한국어)

> 이 문서는 영문 [README.md](README.md)를 옮긴 것입니다(2026-10-12 기준). 두 문서의 내용이 다르면 영문이 기준입니다.

**VisRef**(Ghosal 외, *VisRef: Visual Refocusing while Thinking Improves Test-Time
Scaling in Multi-Modal Large Reasoning Models*, arXiv:2603.00207, CVPR 2026)의 비공식 재현입니다. 대상은 부록 Table 6의
*InternVL-1B* 행으로, MathVista *testmini*(1,000문항)에서 InternVL3.5-1B를 사용해 다음을 비교합니다.

| 방법 | 하는 일 | 논문 (Table 6) |
|---|---|---|
| **ST** – 표준 사고(standard thinking) | 추론을 한 번 하고 답을 낸다 | 46.1 |
| **TSR** – 텍스트 자기 성찰(textual self-reflection) | 답 엔트로피가 δ보다 낮아질 때까지 "Wait, think more" 단계를 반복한다 | 48.5 |
| **VisRef** | TSR에 더해, 매 단계 뒤에 DPP로 고른 이미지 시각 토큰의 부분집합을 다시 주입한다 | 52.0 |

![이 재현에서 실행한 ST, TSR, VisRef. 주황색 메모는 논문에서 찾지 못한 설정을 표시한 것](figures/method_overview.svg)

> **상태(2026-10-13):** 본 실험과 후속 실험 두 개는 마쳤고 분석도 끝났습니다. 세 번째 후속 실험(`visref_hs`, 마지막 층 Z)은
> 결과 전에 판정 규칙을 공개하고 실행 중입니다([REPORT 4.5절](REPORT.md#45-visref_hs-last-layer-z-pre-registered-running)).
> 전체 정리: **[REPORT.md](REPORT.md)**.

## 결과

채점(사전에 고정): 답은 규칙으로 추출하고, 규칙이 파싱하지 못하는 행(3,000개 중 36개)은 LLM이 공식 MathVista
추출 프롬프트로 답을 추출하며, 그 답은 응답에 나타나는 경우에만 인정합니다
([LLM 보조 추출](#3-llm-보조-추출-규칙-채점기가-파싱하지-못하는-행-약-1) 참조).
n = 1,000이며, 각 방법을 문항마다 한 번씩 실행하고 시드와 첫 사고 단계는 공유합니다.

![짝 비교 정확도 차이와 95% CI를 논문의 차이와 나란히 표시](figures/paired_differences.svg)

| 정확도(%) | ST | TSR | VisRef |
|---|---|---|---|
| 논문 (Table 6) | 46.1 | 48.5 | 52.0 |
| **이 재현** | **57.8** | **60.1** | **60.0** |

| 비교 | 이 재현, pp (95% CI) | 첫 번째만 / 두 번째만 정답 | McNemar p | Holm p | 논문, pp | 분류 |
|---|---|---|---|---|---|---|
| VisRef − ST (주 비교) | +2.2 (+0.3, +4.1) | 60 / 38 | 0.033 | 0.067 | +5.9 | 효과 크기 재현 안 됨 (방향은 같음) |
| VisRef − TSR (주 비교) | −0.1 (−2.0, +1.7) | 44 / 45 | 1.000 | 1.000 | +3.5 | 효과 크기 재현 안 됨 |
| TSR − ST (보조) | +2.3 (+0.6, +4.0) | 51 / 28 | 0.013 | – | +2.4 | 논문 값과 양립 |

- **요약:** TSR이 ST보다 높은 정도는 논문 값과 양립합니다(이것이 효과나 그 작동 방식이 같다는 것을 보여 주지는 않습니다).
  우리의 InternVL3.5-1B 구현에서는 VisRef가 TSR보다 추가로 더 낫다는 것을 확인하지 못했습니다. VisRef − ST는 논문보다
  작습니다(+2.2 대 +5.9 pp). CI는 0을 포함하지 않지만, 두 주 비교에 대해 사전에 정한 Holm 보정을 적용하면 유의하지
  않습니다(p = 0.067; 보정 없이는 0.033).
- **분류 열:** *논문 값과 양립* = 우리의 95% CI가 논문의 차이를 포함하고 0은 포함하지 않음(동등성의 증명은 아님).
  *효과 크기 재현 안 됨* = 우리의 95% CI가 논문의 차이를 포함하지 않음. CI는 짝 bootstrap 구간입니다(2,000회 재표집,
  seed 0).
- **규칙을 정한 시점:** 주 비교와 검정은 처음 392문항을 중간에 본 뒤에, 분류 규칙은 759문항을 중간에 본 뒤에 정했으며,
  둘 다 최종 분석 전입니다. 따라서 본 실험은 엄밀한 의미의 사전 등록이 아니며 그 분류는 확증적이지 않습니다
  ([REPORT 2.4절](REPORT.md#24-timing-of-pre-registration)). 분류는 시도한 모든 채점 방식에서 같습니다([REPORT 3.2절](REPORT.md#32-sensitivity-to-answer-handling)). 아래 후속 실험들의 판정 규칙은 결과가
  나오기 전에 정했습니다.

### 우리 정확도가 논문보다 8–12 pp 높은 이유 (해결되지 않음)
우리의 ST(57.8)는 공식 InternVL3.5-1B MathVista(mini) 점수 59.3에 가깝지만, 논문의 ST는 46.1입니다. 논문의 8B ST(68.1)도
공식 InternVL3.5-8B 점수(78.4)보다 10 pp 넘게 낮습니다. 이는 논문이 기준 방법들을 공식 평가보다 모델 점수가 낮게 나오는
설정에서 평가했다는 것을 시사하지만, 그 설정을 논문에서 찾지 못했습니다. 원인에 대한 우리의 추정은 다음과 같으며, 어느
것도 직접 시험하지는 않았습니다.

1. **끝나지 않은 답을 오답으로 처리.** 답 상한 안에 끝나지 않아 답 강제가 필요했던 답을 모두 오답으로 세면, 우리
   정확도는 ST 44.4 / TSR 48.3 / VisRef 47.4가 되어 논문의 ST, TSR(46.1 / 48.5)에 가깝습니다(결과를 본 뒤 여러 채점
   방식 중에서 고른 비교입니다. [REPORT 4.4절](REPORT.md#44-paper-fingerprints-post-hoc)).
2. **토큰 상한, 답 추출, 프롬프트.** 셋 다 논문에서 찾지 못했습니다. 예를 들어 사고 상한이 더 낮으면 더 많은 답이
   중간에 끊깁니다.

공식 점수는 InternVL3.5 보고서(Table 3)에서 가져왔습니다. 논문의 각 수치의 출처는 `docs/SOURCES.md`에 있습니다.

### 여기서 VisRef에 추가 이득이 나타나지 않은 이유
원인은 먼저 우리 구현에서 찾습니다. 이를 설명할 수 있는 차이와, 우리가 확인한 내용은 다음과 같습니다.

| 우리 쪽 후보 | 왜 영향을 줄 수 있는가 | 확인했는가? |
|---|---|---|
| 논문의 m개 토큰의 약 5분의 1만 주입합니다: greedy DPP가 커널 rank에서 멈춥니다(확인된 절차상의 차이) | 단계마다 시각적 근거가 적음 | 결론 나지 않음. 나머지를 m개까지 채운 변형(`visref_fill`, 아래)은 추가 이득이 없었지만, 채운 토큰은 관련성과 무관하게 골라졌고(관련성과의 순위 상관 −0.10), 논문에서도 무작위 선택은 ST와 비슷합니다(Table 7). 관련 있는 토큰 m개를 넣는 것은 시험하지 않았습니다 |
| 주입 형식: InternVL은 원래 이미지를 `<img>` … `</img>` 태그로 감싸서 넣는데, 우리는 다시 넣는 시각 임베딩을 이 태그 없이 `</think>` 바로 앞에 붙였습니다(논문의 형식은 찾지 못했습니다) | 모델이 이를 이미지의 일부로 받아들이지 않을 수 있음 | 초기 소규모 시험(방법당 약 20문항, 다른 설정)에서 태그를 붙이면 두 번째 `</think>`가 63단계 중 31단계에서 나왔고 태그 없이는 65단계 중 10단계였어서 태그를 뺐습니다. 본 실험에서는 시험하지 않았습니다. 태그 없이도 부작용은 남습니다: VisRef가 1단계에서 멈춘 713문항 중 VisRef 최종 답 352개에 `</think>`가 또 들어 있으며, ST 답에서는 67개입니다 |
| 주입 위치: TSR과 VisRef가 다음 단계로 이어진 것은 거의 단계가 1,024토큰 상한에 걸린 경우뿐이어서, 주입이 대개 문장 중간에 들어갑니다 | 논문에서는 대부분 완료된 단계 뒤에 이어졌을 수 있음 | 더 큰 단계 상한(예: 4,096)은 실행하지 않았습니다 |
| 이미지 조각을 무엇과 맞춰 고르는가: DPP는 생각 텍스트와 잘 맞는 이미지 조각을 고르는데, 생각 텍스트를 숫자 벡터로 읽습니다. 우리는 글자를 숫자로 바꾼 직후의 벡터(입력 임베딩)를 썼고 모델의 더 깊은 층 벡터는 쓰지 않았습니다. 논문이 어느 층을 썼는지는 찾지 못했습니다 | 다른 층을 쓰면 다른 이미지 조각을 고름 | 후속 실험으로 실행 중입니다(`visref_hs`, 아래). 12문항 오프라인 확인: 더 깊은 층 벡터를 쓰면 생각 단계가 충분히 길 때 DPP가 논문처럼 m개를 돌려줄 수 있지만, 그중 다양성을 뚜렷이 더하는 토큰은 약 50–110개뿐 |
| 샘플링한 답 하나로 구한 정지 엔트로피 H(아래 참조) | 정지 결정에 잡음이 생김 | 최종 실험에서는 단계마다 여러 번 샘플링하지 않았습니다 |

이 중 어느 것도 배제되지 않았고, 논문의 수치 뒤에 있는 CI나 실행 횟수를 찾지 못했으므로, 차이 가운데 얼마가 우리
구현에서 비롯되는지 말할 수 없습니다. 자세한 내용은 [REPORT 4절](REPORT.md#4-beyond-the-paper)에 있습니다.

### 후속 실험 (결과가 나오기 전에 고정한 판정 규칙)

| 실험 | 질문 | 결과 |
|---|---|---|
| `visref_fill` (976문항) | DPP 선택을 m개까지 채우면 VisRef가 달라지는가? (채운 부분은 관련성과 무관하게 골라진 것으로 드러났다.) | fill − TSR 0.0 pp (−2.15, +2.36); fill − VisRef +0.1 pp (−2.15, +2.36): 관련성과 무관하게 고른 토큰을 더해도 추가 이득은 없었다. 관련 있는 토큰 m개가 도움이 되는지는 시험하지 않았다. |
| `step1_branch` (309문항) | TSR 또는 VisRef가 1단계 뒤에도 계속된 문항에서, Ṽ₁을 다시 주입하면 바로 답하는 경우(B − A)나 ≤ 512개의 사고 토큰을 더 쓴 뒤 답하는 경우(D − C)에 도움이 되는가? | B − A +1.40 pp (−1.19, +3.99); D − C +0.97 pp (−2.27, +4.21); 둘 다 Holm p 0.658: 본 실험에서 사후에 관찰된 바로 답하는 경우의 이득은 재현되지 않았다. |
| `visref_hs` (실행 중) | 생각 텍스트의 벡터 Z를 입력 임베딩 대신 모델의 마지막 층 벡터로 바꾸면 VisRef가 TSR보다 나아지는가? (이 벡터를 쓰면 생각 단계가 m토큰 이상일 때 DPP가 m개를 고를 수 있다.) | 실행 중. 결과 전에 정한 판정 규칙([REPORT 4.5절](REPORT.md#45-visref_hs-last-layer-z-pre-registered-running)): hs − TSR의 CI 하한이 0보다 크면 추가 이득, CI가 0을 포함하고 상한이 +3.5 pp 이상이면 판단 보류, 상한이 +3.5 pp 미만이면 마지막 층 Z로도 논문 크기의 이득은 재현되지 않음. |

모든 최종 표는 [`results/`](results)에 있습니다(`repro/diag/*.json`으로부터 생성). 사후(post hoc) 분석(원인 분석, 답
방식, 시각 토큰 수)은 [REPORT 4절](REPORT.md#4-beyond-the-paper)에 있으며 판정에는 포함되지 않습니다.

## 저장소 구성

```
REPORT.md                  재현성 보고서(MLRC 형식): 범위, 방법, 결과, 논문을 넘어서는 부분, 논의
results/                   최종 표(CSV + summary.json), scripts/make_results.py가 생성
figures/                   paired_differences.svg(scripts/make_figures.py가 results/로부터 그림), method_overview.svg
Makefile                   make fetch / check / report / test (CPU만 사용)
CITATION.cff               이 저장소와 원 논문을 인용하는 방법
repro/                     구현(간결하게 유지)과 분석
  run.py, visref/          생성 엔진, ST/TSR/VisRef, DPP 선택, 채점(scorer v4)
  kaggle/build.py          세션 하나(2×T4)를 위한 자체 완결형 Kaggle 노트북을 만든다; kaggle_ctl.py push/status/pull
  kaggle/build_step1_branch.py   step1_branch 후속 실험에 대해 같은 일을 한다
  colab/build_colab.py     Colab GPU 하나에 대해 같은 일을 하며, Google Drive로 이어서 실행할 수 있다
  report.py, rescore.py    오프라인 집계 / 재채점(GPU 불필요)
  llm_extract.py, gemini_extract.py   규칙이 실패한 행에 대한 LLM 보조 추출
  diag/                    사전 등록된 분석(final_judgment.py, analyze_fill.py, step1_branch.py,
                           analyze_step1_branch.py, analyze_hs.py)과 사후 진단, 그리고 그 JSON 출력
  audit/                   독립 검토와 감사
  kaggle/runs/<session>/   모든 세션: run_config.json(소스 sha256, id, 인자), env.json, 로그, 원본 출력
                           (b_main_s1–s9, b_main_fill, b_main_rescore_v2/v3, diag_st_plain, hidden_rank_check,
                            step1_branch_pilot, step1_branch_main; b_main_hs는 실행 중이라 run_config.json만)
  llm/final, llm/fill      LLM 보조 추출: 요청, 응답, 병합된 행(diag/*.py가 읽음)
  reports/                 방법별 서술 보고서(규칙 기반 scorer v4만 사용)
docs/
  SPEC_LEDGER.md           모든 구현 결정과 그 근거 등급(논문 / 인용한 코드 /
                           공식 모델 문서 / 우리의 선택) — "핵심 판단 요약"부터 읽을 것
  EXPERIMENT_FLOW.md       시간순 기록: 사전 등록, 결정, 결과
  SOURCES.md               (영문) 기록의 인용이 공개 출처(arXiv 버전, 페이지, 표)에 어떻게 대응하는지
  AI_USE_REFLECTION.md     이 프로젝트에서 AI 활용이 잘못된 사례와 재발 방지 방법
references/SOURCES.json    제3자 파일(MathVista 데이터와 평가 프롬프트, 인용한 코드, VLMEvalKit)과 URL + sha256
scripts/                   fetch_sources.py(내려받고 검증), make_results.py, make_figures.py,
                           check_readme_numbers.py
```
`docs/`의 문서는 한국어로 작성되어 있으며, `REPORT.md`가 관련 항목을 가리킵니다.

## 재현 방법

### 1. 저장된 출력으로부터 모든 표와 결과 그림 다시 만들기 (CPU만 사용)
Python 3.10 이상(표준 라이브러리만 사용, `make check`는 3.10, 3.11, 3.12, 3.13에서 통과), `make`, git checkout이
필요합니다. GPU와 PyTorch는 필요하지 않습니다.
```bash
make fetch     # MathVista testmini.json, 공식 추출 프롬프트, 인용한 코드를 내려받고 sha256 확인
make check     # 사전 등록 분석, results/, 결과 그림을 다시 실행하고 README 숫자를 확인합니다.
               # 하나라도 다르면 실패
make report    # 선택 사항: build/에 방법별 서술 보고서 생성(규칙 기반 scorer v4, LLM 보조 추출 없음)
```
단위 테스트에는 PyTorch, transformers, pytest가 필요합니다(CPU 빌드면 충분합니다).
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt pytest
make test
```
개별 명령은 `Makefile`에 나와 있습니다. `repro/reports/b_main_report.json`은 규칙 기반만 사용하므로, 이 파일의 TSR
정확도(60.0)가 본 채점의 값(60.1, LLM 보조 추출로 한 행을 복구)과 다릅니다.

### 2. 모델 출력 다시 생성하기 (GPU)
고정한 버전: 모델 `OpenGVLab/InternVL3_5-1B` @ `2f71cf52542334823e48a46ffba0e2bc9add3446`,
데이터 `AI4Math/MathVista` @ `2b6ad69445fbb5695c9b165475e8decdbeb97747`, `transformers==4.55.0`, fp16.
각 세션의 정확한 설정은 `repro/kaggle/runs/<session>/run_config.json`에 있습니다. 본 실험은 2×T4의 Kaggle 세션
9개로, 총 약 14.9 세션 시간이 걸렸습니다([REPORT 2.5절](REPORT.md#25-code-versions-and-compute)).
새 실행 이름을 사용하세요. `repro/kaggle/runs/` 아래 폴더는 공개된 세션의 기록이며, 빌드 스크립트는 기존 실행 폴더를
덮어쓰기를 거부합니다(`--overwrite`는 아직 push하지 않은 세션을 다시 만들 때만 있습니다).
```bash
# Kaggle (2×T4): kaggle.json을 저장소 루트에 둡니다 (절대 커밋하지 마세요)
python3 repro/kaggle/build.py rerun_b_main_s1 --timeout 7200 \
  --run-args="--methods st,tsr,visref_raw_rank --think-cap 1024 --total-think-cap 10240"
python3 repro/kaggle/kaggle_ctl.py push rerun_b_main_s1      # 이후: status / pull
# 이후 세션은 본인의 세션이 이미 끝낸 문항을 건너뜁니다:
#   --skip-from repro/kaggle/runs/rerun_b_main_s*/output/results/results_*.jsonl

# step1_branch 후속 실험 (20문항 파일럿, 그다음 나머지)
cd repro && python3 kaggle/build_step1_branch.py rerun_step1_branch_pilot --limit 20 --timeout 3600
python3 kaggle/build_step1_branch.py rerun_step1_branch_main --timeout 7200 \
  --skip-from kaggle/runs/rerun_step1_branch_pilot/output/results/step1_branch_*.jsonl
```
문항은 seed 0으로 고정된 셔플 순서대로 처리되므로, 완료된 앞부분은 어느 것이나 무작위 부분집합입니다. 모든 (문항, 방법)
쌍은 문항별 seed를 사용합니다(방법 간 common random numbers). fp16에서는 출력이 배치 구성에 따라 약간 달라집니다
([REPORT 6절](REPORT.md#6-limitations)).

### 3. LLM 보조 추출 (규칙 채점기가 파싱하지 못하는 행, 약 1%)
```bash
cd repro
python3 llm_extract.py prepare <results...> --out requests.jsonl --control 20
GEMINI_API_KEY=... python3 gemini_extract.py requests.jsonl --out responses.jsonl --model gemini-3.5-flash-lite --sleep 6
python3 llm_extract.py merge <results...> --responses responses.jsonl --out rescored.jsonl
```
응답 전체가 곧 답인 경우를 빼면, 공식 MathVista 평가는 LLM(MathVista 논문에서는 GPT-4)을 사용해 긴 응답에서 짧은
최종 답을 뽑아냅니다. 그 프롬프트는 예시
다섯 개(문제, 모델 응답, 뽑은 답)를 먼저 보여 주고 그다음에 새 응답을 붙입니다(MathVista 저장소의
`evaluation/prompts/ext_ans.py`이며 `references/SOURCES.json`에 고정되어 있습니다). 우리는 규칙이 파싱하지 못하는 행에
대해서만 같은 프롬프트를 gemini-3.5-flash-lite에 보냅니다. LLM의 답은 응답 텍스트에 나타나는 경우에만 인정하므로 LLM이
문제를 직접 풀어 만든 답은 인정되지 않으며, 빈 응답은 보내지 않습니다(`docs/EXPERIMENT_FLOW.md` 참조).

## 논문에서 찾지 못한 설정
논문에서 다음을 찾지 못했습니다. 샘플링 값, 토큰 상한, 잘린 사고나 끝나지 않은 답의 처리, 답 추출
방법, 생각 텍스트를 나타내는 벡터의 층, DPP 커널 rank가 토큰 예산 m보다 작은 경우의 처리, 정지 엔트로피를 재는 방법, 주입
형식(태그, 구분자, `</think>` 처리, position id), 이미지 타일링입니다. 각 빈칸은 쓸 수 있는 가장 강한 출처로 채웠으며,
순서는 다음과 같습니다. 논문의 그림과 표 → 논문이 인용한 연구와 코드 → InternVL3.5 공식 자료(모델을 Hugging Face에 올리면서 사용법과 권장 설정을 함께 공개한 [공식 모델 페이지](https://huggingface.co/OpenGVLab/InternVL3_5-1B), 그리고 InternVL3.5 보고서가
사용한 VLMEvalKit) → 논문이 인용하지 않은 공개 코드 → 우리의 선택.

| 설정 | 우리의 선택 | 근거 | 다른 가능성 (상태) |
|---|---|---|---|
| 샘플링 | T 0.6, top-k 50, top-p 0.95(논문이 쓴 사고 모드는 공식 모델 페이지의 시스템 프롬프트로 켬) | 논문에서 찾지 못함. 공식 모델 페이지는 사고 모드에서 반복을 막으려면 T 0.6으로 샘플링하라고 권장 | – |
| 토큰 상한 | 사고 단계당 1,024, 총 10,240, 답 1,024 | 1,024는 공식 모델 페이지의 예시 값이고 나머지는 우리의 선택 | 더 큰 단계 상한(예: 4,096) 또는 공식 평가의 새 토큰 65,536개; 실행하지 않음 |
| 상한에서 잘린 사고 | 생각을 닫고 `Final Answer:`를 붙여(`</think>\nFinal Answer:`) 바로 답하게 함 | 논문이 인용한 s1(Muennighoff 외 2025)의 방식("budget forcing"). s1 공식 코드가 정확히 `\nFinal Answer:`를 붙임. VisRef에 적용한 것은 우리의 해석 | 이 문구 없이 돌린 진단(ST, 150문항): 정확도 −2.0 pp. 150문항으로는 차이가 없는 것과 구분할 수 없음(95% 범위 약 −9.5 ~ +5.5 pp) |
| 끝나지 않는 답 | 답을 강제로 쓰게 함: `\n**Final Answer**\n\boxed`를 붙이고 최대 100토큰 더 생성 | 답을 쓰다가 "Wait, wait …"로 다시 생각하며 1,024토큰 상한까지 가는 경우가 많았는데, 논문과 인용 문헌에서 이 경우의 처리를 찾지 못했습니다. DEER(Yang 외 2025, 추론 모델 조기 종료 방법, 논문이 인용하지 않음)의 공개 코드에 바로 이 경우를 위한 규칙이 있어 따랐습니다 | 이런 답을 오답으로 처리([REPORT 3.2절](REPORT.md#32-sensitivity-to-answer-handling)) |
| 답 추출 | 규칙 먼저(boxed 답, "answer is", 마지막 줄), 규칙이 파싱하지 못한 행에만 공식 MathVista 추출 프롬프트로 LLM 사용 | MathVista 공식 코드는 응답 전체가 곧 답일 때(선택지 문자열, 숫자 하나)만 규칙으로 처리하고 나머지는 LLM(GPT)으로 추출합니다. VLMEvalKit도 규칙을 먼저 쓰고 실패하면 GPT를 씁니다 | 응답 전체가 답이 아닌 모든 행에 공식 GPT 모델로 LLM 추출; 실행하지 않음 |
| 채점 대상 텍스트 | 마지막 `</think>` 뒤의 텍스트 | VLMEvalKit과 같은 분리 규칙(InternVL3.5 평가에서 이를 켰는지는 미확인) | 출력 전체를 채점; 시험하지 않음 |
| DPP 선택 크기 | greedy DPP가 커널 rank에서 멈춤, 약 m/5개 토큰 | 인용된 DPP 코드(Chen 외 2018). 논문의 \|V_k\| = m과 충돌함 | 관련성과 무관하게 고른 토큰으로 m개까지 채움: 추가 이득 없음(`visref_fill`); 관련 있는 토큰 m개: 시험하지 않음 |
| 생각 텍스트를 나타내는 벡터(Z) | 입력 임베딩: 글자를 숫자로 바꾼 직후의 벡터 | 우리의 선택. 논문은 "text token embeddings"라고만 쓰고 층을 밝히지 않음 | 더 깊은 층(예: 마지막 층): 그러면 생각 단계가 충분히 길 때 DPP가 논문처럼 m개를 돌려줄 수 있음; 마지막 층은 후속 실험으로 실행 중(`visref_hs`) |
| 정지 엔트로피 H | 샘플링한 답 하나(아래 참조) | 정의는 논문에서 가져옴; 재는 방법은 우리 것 | 샘플링한 답 여러 개, 서로 다른 답 값에 대한 엔트로피, 밑이 2인 로그(같은 δ = 0.25가 더 엄격한 기준이 됨); 최종 실험에서는 쓰지 않음 |
| 주입 형식 | 선택된 시각 임베딩을 `</think>` 바로 앞에 붙임. InternVL이 이미지에 쓰는 `<img>` … `</img>` 태그는 붙이지 않음; 순서는 z_k → Ṽ_k → H_k → "Wait, think more" | 순서는 논문(Algorithm 1)에서 가져옴; 형식은 우리의 선택 | 태그를 붙이는 경우: 초기 소규모 시험 뒤 뺌(위 참조); `</think>` 뒤에 붙이는 경우, 다른 position id: 시험하지 않음 |
| 이미지 타일링 | InternVL의 기본 입력 방식: 이미지를 모양에 맞춰 448×448 조각 최대 12개로 나누고, 전체를 줄인 이미지 1개를 더함. 조각 하나가 시각 토큰 256개가 됨(이미지당 평균 1,961.5개) | 공식 모델 페이지의 예제 코드(`load_image(max_num=12)`, 축소 이미지 사용)와 모델 설정 파일(`max_dynamic_patch` 12, `use_thumbnail` true) | 축소 이미지를 빼면 평균이 1,755.6으로, 논문의 ∼1,772(8B)에 더 가까움; 실행하지 않음 |
| 하드웨어 | Kaggle T4, fp16 | 무료 GPU | 논문: A10G GPU 4개, PyTorch 2.7.0, Transformers 4.55.0; 시험하지 않음(같은 GPU에서 fp16 잡음으로 바뀐 답은 2.2%) |

**정지 엔트로피 H.** 논문은 H를 모델의 답 분포의 엔트로피로 정의하며(Algorithm 1, 12행), 이 기준에 대해 논문이 인용한
연구(Ghosal 외)는 이를 재려고 답을 여러 개 샘플링합니다. H는 생각을 더 할지 정하는 값이라, 최종 답이 나오기 전 각 생각 단계가 끝날 때마다 재야 합니다. 그래서 그 시점에서
모델에게 답을 써 보게 합니다. 우리는 `</think>` + `\n\n\boxed` 뒤에 모델이 짧은 답 하나를 쓰게 합니다(최대 16토큰). 정답이
아니라 모델이 그 자리에서 쓴 답이고, 엔트로피는 모델이 낼 수 있는 답들에 대한 평균이므로 가장 그럴듯한 답만 고르지 않고
T = 1로 샘플링합니다. 이 답의 토큰을 하나씩 낼 때마다 모델은
"다음에 무엇이 나올지"에 대한 확률 분포를 갖는데, 각 분포의 엔트로피를 자연로그로 계산해 모두 더합니다. 이 합의 평균은 답
분포의 엔트로피와 같으므로(연쇄 법칙) 논문과 같은 양을 재지만, 샘플이 하나뿐이어서 흔들리고, 그 흔들림이 방법이 언제
멈추는지에 영향을 줍니다. 경위: 처음에는 답 하나의 토큰 엔트로피 평균으로 쟀습니다. Ghosal 외를 읽은 뒤 단계마다 완결된 답
10개를 샘플링하는 방식도 시도했는데, 8B 모델로 개발용 문항 3개에 대해 TSR을 실행하는 데 GPU 4,703.9초가 걸렸습니다. 논문의
지연 시간 표(Table 5: 문항당 ST 7.1초, TSR 7.7초, VisRef 8.2초, 8B 모델, H100 1장)로 보아 저자들이 단계마다 답을 여러 개
생성했을 수는 없다고 판단해 답 하나로 돌아갔습니다. 마지막으로, 평균은 답의 엔트로피가 아니므로 합으로 바꿨습니다(결정 기록:
`docs/EXPERIMENT_FLOW.md`, 2026-10-05~2026-10-07). `Answer:` 프로브도 바꿨는데, 서식 토큰이 H의 대부분을 차지했고 스모크
테스트 문항 18개 중 TSR이 엔트로피로 멈춘 것은 0개였기 때문입니다.

모든 결정과 그 근거, 대안은 `docs/SPEC_LEDGER.md`에 있습니다("핵심 판단 요약"부터 읽을 것). 논문의 각 주장이 어디에서
왔는지(arXiv 버전, 페이지, 표)는 `docs/SOURCES.md`에 있습니다. 저자에게 할 질문은
[REPORT 5.3절](REPORT.md#53-communication-with-original-authors)에 있습니다.

## 기여와 AI 사용
이 재현은 AhnJG가 AI(Anthropic의 Claude 모델, OpenAI의 Codex)의 도움을 받아 수행했습니다.

- **AhnJG:** 재현 대상과 근거 규칙(외부 수치는 모두 file:line으로 출처를 밝히거나 미확인으로 표시)을 정했습니다.
  `docs/EXPERIMENT_FLOW.md`에 "사용자 결정"으로 기록된 설계 결정을 내렸습니다. 여기에는 DPP의 rank만큼만 주입하는 것
  (`visref_raw_rank`), 답 처리 방식(s1 프롬프트 + DEER 답 강제), 규칙 기반 채점과 LLM 보조 추출,
  최종 분석 전에 주 비교와 검정을 고정한 것, 후속 실험과 그 범위, 종료 시점이 포함됩니다. 기록된 여러 정정의 계기가 된 질문을 제기했습니다.
- **Claude(AI 어시스턴트):** 코드 대부분(생성 엔진, 방법, DPP 선택, 채점, 분석 스크립트, 테스트)을 작성했습니다.
  사전 등록 문서, 분석, REPORT.md, 그림의 초안을 작성했습니다. 독립 검토는 별도의 모델 세션들이 수행했습니다
  (`repro/audit/`).
- **Codex(OpenAI):** 프로젝트 전 과정에서 내용 검토에 사용했습니다.
- **방법의 일부인 LLM:** gemini-3.5-flash-lite가 규칙이 파싱하지 못하는 행의 답을 추출합니다([LLM 보조 추출](#3-llm-보조-추출-규칙-채점기가-파싱하지-못하는-행-약-1) 참조).

## 인용
`CITATION.cff`를 참조하세요. 원 논문도 함께 인용해 주세요.
```bibtex
@article{ghosal2026visref,
  title   = {VisRef: Visual Refocusing while Thinking Improves Test-Time Scaling in Multi-Modal Large Reasoning Models},
  author  = {Ghosal, Soumya Suvra and Kim, Youngeun and Li, Zhuowei and Chaudhry, Ritwick and Xu, Linghan and
             Zhang, Hongjing and Zablocki, Jakub and Xing, Yifan and Zhang, Qin},
  journal = {arXiv preprint arXiv:2603.00207},
  year    = {2026}
}
```

## 라이선스
- 코드: MIT(`LICENSE`).
- 데이터 파일에는 CC BY-SA 4.0으로 배포되는 MathVista에서 파생된 내용이 들어 있습니다. 해당하는 파일은 실행 출력
  (`repro/**/results_*.jsonl`, `repro/**/step1_branch_*.jsonl`), `repro/llm/`의 LLM 추출 요청과 응답(문항, 답, 공식
  MathVista 추출 프롬프트), `repro/diag/`의 분석 JSON, `results/`, 보고서입니다. 따라서 이 데이터 파일은 CC BY-SA 4.0으로
  공유됩니다. 모델 출력은 InternVL3.5-1B(Apache-2.0)로 생성했습니다.
- 그 밖의 제3자 파일(MathVista 데이터, 인용한 코드)은 포함하지 않으며, `scripts/fetch_sources.py`가 원래 위치에서
  내려받습니다.
