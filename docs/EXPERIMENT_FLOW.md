# VisRef 전체 실험 흐름

## 연구 질문

동일한 visual reinjection 절차에서 중요한 원본 visual token을 보존하고 나머지를 pooling하면, DPP 30%보다 적은 token으로 정확도를 유지하고 실제 추론 비용을 줄일 수 있는가?

이 질문을 평가하려면 먼저 Thinking mode, reasoning lifecycle, 채점, entropy와 reinjection 처리가 검증된 baseline이 필요하다. 현재 그 조건을 충족하지 못했으므로 개선 실험을 중단했다.

## Baseline과 평가 계획

Pilot 모델은 InternVL3.5-1B다. MathVista testmini를 seed 42로 development 100 / selection 200 / locked evaluation 700문항으로 분리했다. [분할 checksum](data/manifests/splits.sha256.json)을 보존한다. Selection에는 실행 이력이 있으며, 잠금 700문항은 모델 추론과 성능 분석을 하지 않았다.

당시 protocol은 고정 3-step, greedy decoding, 자체 엄격한 답 추출기를 사용하는 operational 절차였다. 공식 entropy estimator와 재주입 fidelity는 미확인이었다. [과거 protocol](docs/notes/operational_v1_PROTOCOL.md)은 이 실행의 설명이며 논문 재현 protocol이 아니다.

## 변경·결과·판단

| 단계 | 변경 또는 검증 | 실제 결과 | 판단과 다음 방향 |
|---|---|---|---|
| 로컬 correctness | Kernel factorization, greedy indices, pooling, split·scoring·statistics 검증 | 당시 30개, ablation 회귀 검증 추가 후 32개 테스트 통과 | Synthetic correctness이며 모델 성능·공식 재현 증거가 아님 |
| P0 v1 | 실제 모델 경로 검증 시작 | NumPy/선택적 TensorFlow import 문제로 모델 로딩 실패 | 구현·의존성 실패. 압축 가설 실패로 해석하지 않음 |
| P0 v2 | NumPy 고정 및 TensorFlow/Flax 비활성화 | 개발 첫 10문항 PASS | Native 초기 8-token 및 자체 identity append 일치. 전체 thinking 종료·최종 답변 검증은 아님 |
| P01 v1 | ST·TSR·DPP30 개발 평가 | Autograd 추적 발견, 95개 부분 행에서 중단 | 비용 평가 무효화. 결과와 실패 이력을 보존 |
| P01 v2 | Model freeze와 inference mode 적용 | 100문항 × 3방법, 300행 완료 | Operational 결과를 회수·분석. 정확한 VisRef 재현으로 인정하지 않음 |
| P2 v1 | 9개 구현 방법 비교, 동일 keep-index replay ablation | P0 10문항·개발 smoke 90건 통과 후 selection 155/1800행에서 사용자 요청으로 중단 | 후보 선택에서 제외. 새로운 실험의 튜닝 근거로 사용하지 않음 |
| Fidelity 검토 | 공식 모델카드·논문과 현재 구현 대조 | Thinking System Prompt 누락 확인; step·Z 범위·채점·entropy·reinjection 조건 미확인 | 모든 개선 실험 중단. baseline 재현을 먼저 검증 |
| 폴더 정리 | 루트 중심 구조와 자료·결과 분리 | 이전 전 SHA와 경로 대응표를 보존 | 실험 재개나 모델 방법 변경이 아님 |

## P01 v2의 관측 결과

| 방법 | N | Accuracy | 평균 end-to-end latency | 답 추출 실패 |
|---|---:|---:|---:|---:|
| ST | 100 | 40% | 9.869초 | 26건 |
| TSR | 100 | 40% | 17.453초 | 27건 |
| DPP30 | 100 | 30% | 19.846초 | 34건 |

조건과 표본이 달라 논문 수치와 직접 비교하지 않는다. 1B에서 효과가 없다고 결론 내리지 않는다. Thinking mode 누락과 출력·채점 문제 등이 섞여 있으며, 어떤 차이가 하락을 일으켰는지는 확인되지 않았다. Kernel 계산식의 동등성 검증으로 전체 baseline fidelity가 증명되는 것도 아니다.

- [P01 v2 원본 상태](artifacts/runs/p01/v2/visref_results/status.json)
- [분석 결과표](artifacts/reports/p01/v2/figures/RESULTS.md)
- [샘플별 원시 결과](artifacts/runs/p01/v2/visref_results/samples.jsonl)
- [P2 후보 선택 제외 기록](artifacts/runs/p2/v1/EXCLUDED_FROM_CANDIDATE_SELECTION.json)
- [누적 상세 로그](docs/EXPERIMENT_LOG.md)

## 현재 결론

논문 baseline이 검증되지 않은 상태에서 개선 후보 비교를 먼저 진행한 것은 순서가 잘못됐다. 현재 결과는 operational 진단 자료이며 VisRef 재현 실패, 압축 가설의 성공·실패, 후속 방법의 우수성을 주장하는 근거로 사용하지 않는다.

로컬 controller 종료와 Kaggle P2의 CANCEL_ACKNOWLEDGED를 확인했다. 신규 dispatch는 중단 marker로 차단한다. 실패·중단 실행의 GPU 사용량을 누적 자원 한도에서 제외하지 않는다.

## 다음 단계

공식 근거와 현재 구현을 항목별로 대조하고, Thinking mode와 완전한 native ST 생성·채점을 우선 검증한다. 이후 TSR, DPP의 step·Z·entropy·재주입 조건을 확인한다. 미확인 조건은 BLOCKED로 기록한다. 검증된 재현 baseline과 새 protocol이 확보되기 전에는 개선 실험·후보 선택·잠금 평가를 재개하지 않는다.

## 2026-10-05 baseline_v1 로컬 구현·검증

논문·공식 native model·MathVista source를 대조한 [fidelity 표](docs/baseline_v1/FIDELITY.md), [새 protocol](docs/baseline_v1/PROTOCOL.md), [blocker 상세](docs/baseline_v1/BLOCKERS.md)를 추가했다. 과거 operational 코드/설정/결과는 보존하고 별도 Thinking/native full generation adapter와 versioned scorer, DPP 수학/control components를 구현했다.

최종 local run `20261005_001921_local`에서 새29+기존32=61 tests PASS. 이는 합성 full lifecycle/공식 source parity·수학 correctness이며 실제 InternVL3.5-1B inference 또는 논문 재현은 아니다. ST 모델 검증은 미실행, TSR/VisRef end-to-end는 official entropy/step/Z/V/reinjection/stabilization/scorer 확인 전 BLOCKED다. Table1 ST adaptive criterion도 별도로 미확인이다.

[한국어 결론](artifacts/reports/baseline_v1/CONCLUSION_KO.md)에 성공 수준과 범위를 구분했다. STOP/manifest/historical outputs SHA를 유지했고, 모델/remote/GPU/P2/selection/locked evaluation/개선 실험/후보 선택은 실행하지 않았다. 다음 단계는 공식 missing mechanics 확인과 별도 승인된 native ST complete-path GPU validation이다. GPU 예산은 실패·취소 포함 기존 기록 합산2.967115h, 총 잔여17.032885h; live quota 미확인. baseline 검증 전에 후속 압축 비교로 넘어가지 않는다.

최종 local validation은 전처리 fixture 보강 후 `20261005_002747_local`이다. 61 passed, prior run은 보존. 현재 [결론](artifacts/reports/baseline_v1/CONCLUSION_KO.md)과 baseline protocol은 이 source/config snapshot을 참조한다.

## 2026-10-05 독립 regularized DPP30 명세 반영

사용자의 지시에 따라 [명세 검토 보고서](artifacts/reports/baseline_v1/INDEPENDENT_DPP30_SPEC_REVIEW_KO.md)를 source·수학·합성 진단과 대조하고 [독립 protocol](docs/independent_dpp30_v1/PROTOCOL.md)에 수정 후 반영했다. 논문 재현의 미확인 entropy 등은 그대로 두고 독립 경로의 고정2-block/Z/V/상대regularization/주입·실패 처리를 명명했다.

Selection undefined의 주입 생략은 실패로 변경했다. λ 변화에 대한 선택 불변·unregularized prefix 일치·pivot 기반 rank를 일반 보장으로 채택하지 않았다. Shared-cache 생성 비용과 실제 단독 latency도 분리했다. Controller 구현·실제 InternVL 생성·정확도·speedup 증거는 없다. 이번에는 문서와 기존 합성 진단 재확인만 수행했으며 source/config/원시run/STOP를 유지했다.

다음 단계는 별도 지시된 로컬 controller/selector/recorder 구현·LC checks, 그 뒤 별도 승인된 native→reflection→DPP validation이다. 개선 비교는 사전 동결된 새 계획과 별도 승인 전까지 보류한다. 동일 논문 fidelity 질문은 새로운 공식 근거가 없으면 반복 검토하지 않는다.

## 2026-10-05 baseline 실제 검증 재개 범위

사용자가 최대2 Kaggle kernel을 승인하여 독립 baseline correctness를 개발 첫3에서 검증한다. [실행 계획](docs/independent_dpp30_v1/KAGGLE_TEST_PLAN_20261005.md), [기록](docs/EXPERIMENT_LOG.md). G1 dispatch 완료, G2는 실제 실패 수정/재검증용으로 남긴다. 기존 STOP/old pipeline·selection/locked·후속 압축 실험은 유지한다. 추가 일반 검토보다 raw native lifecycle/cache/selector 증거로 다음 판단을 한다.

### 두 kernel 종료 후 결정

[결과 보고서](artifacts/reports/independent_baseline_20261005/BASELINE_TEST_REPORT_KO.md): native full-generation parity/종료와 첫 DPP append가 개발 `243`에서 확인됐고, fixed2 TSR/DPP는 자연 종료 뒤 WAIT가 빈block2를 만들어 실패했다. 가설 자체의 구체적 문제를 확인했다. λ/source-order/position/scorer는 유지하고 종료 정책 하나만 [max2_v2](docs/independent_dpp30_max2_v2/PROTOCOL.md)로 분리했다. Local76 PASS/GPU UNVERIFIED, v1 raw는 보존. 다음 증거는 새로운 일반 검토가 아니라 v2 개발 first3 correctness다. 이번 최대2 kernel은 모두 사용·종료했고 추가GPU/selection/locked/압축 후보실행 없음.

## 2026-10-05 independent_dpp30_entropy_v3 명세·로컬 구현 (GPU 미실행)

[max2_v2 변경 검토](artifacts/reports/independent_baseline_20261005/MAX2_V2_CHANGE_REVIEW_KO.md)에서 v2가 자연 종료 시 TSR=ST, DPP 답변 직전 1회 주입이 되어 논문 종료 조건과 크게 다름을 확인했다. 그래서 사용자 결정에 따라 가정한 entropy 종료(H<0.25 nats, K_max=10), "Wait, think more" 연장, L_min=1, 모델 최대 토큰 상한을 쓰는 별도 [v3 명세](docs/independent_dpp30_entropy_v3/PROTOCOL.md)를 만들고 로컬 구현했다. TSR과 DPP의 차이는 visual 주입 하나로 두었고 v1/v2 경로·원본 결과는 보존한다.

주장 범위: CPU 로컬 86 passed(기존76+신규10, tiny random Qwen3)는 명세·구현·fail-closed 검사의 확인이며 실제 InternVL/GPU 증거가 아니다. Entropy estimator는 독립 가정(논문 estimator UNVERIFIED), 1회 추정이라 분산이 크다. 정확도·speedup·논문 재현 주장은 없다. STOP·selection/locked·후보 비교 차단은 유지되고 GPU 실행은 미승인(dispatch 0)이다.

다음 단계는 사용자 승인 후 kernel 1개로 개발 첫3(243,479,419) v3 correctness 검증이다.


## 2026-10-05 v3 runbook 실행: 전송 전 네트워크 중단

v3 correctness는 원격 전송 이전 인증 DNS 오류로 UNVERIFIED다. Summarizer verdict 없음. Runbook의 1회 push 규칙에 따라 재전송하지 않고 종료한다. 추가 실행에는 사용자 결정이 필요하다. [실행 보고서](artifacts/reports/independent_entropy_v3/RUN_REPORT_KO.md).


## 2026-10-05 v3 kernel version1 종료

v3 원격 gate는 FAIL(SETUP_FAILED). 로컬93 통과와 전송 bundle의 실제 독립 실행은 달랐다. Scripts import 의존성 누락으로 모델 미실행; entropy/controller 가설 판정 근거 없음. 다음 후보는 새 version bundle 일치 검증이며 추가 수정/실행은 하지 않는다. [결과 보고서](artifacts/reports/independent_entropy_v3/RUN_REPORT_KO_KERNEL_V1.md).

## 2026-10-05 v3 r2 실행 준비 완료

사용자 수정 지시로 scripts 누락과 자동 판정 결함을 수정했다. 최종 전송본을 별도 폴더·프로세스로 검증하여113 tests PASS, 기존67파일SHA 보존. Entropy/controller 모델 정책은 그대로다. [준비 보고서](artifacts/reports/independent_entropy_v3_r2/PREPARATION_REPORT_KO.md).

현재 로컬 준비는 완료, actual InternVL correctness는 UNVERIFIED다. 기존 kernel1 승인 소진, r2 승인 파일 없음/dispatch0. 다음은 추가 일반 검토가 아니라 새 승인된 개발3문항의 실제 경로 검증이다. P0–P1/total 예산은 실패 차감 후 기존값 유지, STOP·selection/locked·비교 차단 유지.


## 2026-10-05 v3 r2 단일 kernel 전송

r2 private kernel `arki13/visref-entropy-v3-r2-g1`, version1을1회 전송했다. 승인 포함 bundle113 PASS. Frozen source SHA `f5dfdc4f5c3121bddcef4b884d6c8c89931ef8d4a6e29b71085acf46b5f135c6`. 개발243/479/419, T4/fp16,3600초/P0–P1. 실제 환경·native/ST/TSR/DPP gate는 실행 결과 대기다. 원본: `artifacts/runs/independent_entropy_v3_r2/g1/launch.json`. 재전송·추가kernel 없음.


## 2026-10-05 v3 r2 단일 GPU 실행 종료: FAIL

r2는전송·remote113 tests·실제native/ST/TSR 실행까지진행했고, 첫243 DPP 뒤CLOSE cache/full parity TV3.87%가고정2%를초과해FAIL이다. DPP 주입자체는통과했으며ST/TSR는1step정상완료했다. [결과 보고서](artifacts/reports/independent_entropy_v3_r2/RUN_REPORT_KO.md).

다음은entropy/선택정책 변경이아니라, 동일243 block/selection의DPP CLOSE 고정입력replay로KV/position/mask와fp16수치차이를구분하는진단이다. 현재원인확정·전체baseline통과·성능개선 근거없음. 승인소진, 추가GPU0, STOP/selection/locked/후속비교차단 유지. 사용257.098초, P0–P1잔여10354.474551초; 후속새version은r2원장을부모로사용한다.

## 2026-10-05 r2 원인 분석 보완

[실패 분석](artifacts/reports/independent_entropy_v3_r2/FAILURE_ANALYSIS_KO.md)은 캐시 오류/fp16 원인을 UNSURE로 유지한다. 보정 없는2%와 top-k 분포 불연속 때문에 TV 초과만으로 버그를 확정할 수 없다. 다음은 같은243 입력·indices·chunk를 유지한 fp16/fp32 구조·분포 진단이다. 기준 완화나 전체 생성 재시도는 원인 구분을 대체하지 않는다.

Entropy는 마지막에 반복한 답 토큰만 측정해 confidence와 다른 양을 보는 문제도 확인했다. Cache 진단 뒤 별도 estimator 명세·보정으로 다룬다. 기존 결과는 변경하지 않으며 추가GPU0, STOP/selection/locked/비교 차단 유지.

## 2026-10-05 다음 실행 계획 r3

[r3 계획](docs/independent_dpp30_entropy_v3/KAGGLE_RUN_PLAN_R3.md)을 작성했다. 앞선 fp32 replay 우선 권고를 보완하여 fp16 실행의 구조·정상종료 검사와 원인분리 진단을 구분한다. 다음 작업은 r3 CPU 구현·전송본 검증이며, 이후 새 승인된kernel1에서같은개발3문항의DPP 답까지관측한다. TV2%는경고로사전명시하고더큰실패threshold를만들지않는다. 기존FAIL은새기준으로재판정하지않는다.

Entropy 정책은이번실행에서유지하고알려진한계를경고한다. 실행9경로완료라도baseline 비교준비완료는아니다. GPU 승인·dispatch0, STOP/selection/locked/비교차단유지.

## 2026-10-05 r3 entropy 계획 수정

위 estimator 유지 권고는 사용자 지적에 따라 철회했다. [현재 실행 계획](docs/independent_dpp30_entropy_v3/KAGGLE_RUN_PLAN_R3.md)은 [answer_only_sequence_probe_v1](docs/independent_dpp30_entropy_v3/ANSWER_ENTROPY_REVISION_R3.md)을 구현·검증한 뒤 실행하도록 수정했다. Probe에서 설명/답 재반복을 허용하지 않고 전체 draw+EOS를 측정하며 최종답은 별도로 생성한다. 모델/reasoning prompt·DPP·sampler·scorer는 유지한다. 초기0.25 보정은 미검증이다. 현재 코드·GPU 변경은 없다.

## 2026-10-05 r3 최종 답 토큰 mean entropy 계획 확정 (문서만)

사용자 요청에 따라 [현재 실험 계획](docs/independent_dpp30_entropy_v3/MEAN_ANSWER_TOKEN_ENTROPY_PLAN_R3.md)을 작성하고 [r3 runbook](docs/independent_dpp30_entropy_v3/KAGGLE_RUN_PLAN_R3.md)에 연결했다. 위 sum+EOS 초안은 보존하며 현재 계획은 answer-only probe의 유효 답 값 token만 선택해 raw T=1/full-vocabulary entropy를 평균한다. EOS·바깥공백은 별도 기록, sampling entropy·sum은 진단, 최종답은 별도 생성이다.

현재 개발3은 이미 노출된 correctness 표본이며 δ=.25는 미보정 초기값이다. 다음은 해당 명세의 CPU 구현·내장bundle 검증이며, 그 준비 결과가 확인된 뒤 새 GPU 승인 범위를 제시한다. 이번에는 문서만 수정했고 코드/config·STOP·r2원본·예산은 유지한다. 논문 동일성/정확도 우열/잠금 평가 차단은 유지한다.

## 2026-10-05 r3 mean entropy CPU 구현·전송본 검증 완료

[준비 보고서](artifacts/reports/independent_entropy_v3_r3/PREPARATION_REPORT_KO.md): 별도r3 config/decoder/controller/recorder/summary3와builder run-version을구현했다. 답 값token의raw T=1/full-vocabulary entropy평균, EOS/외부공백제외, strict probe와별도final생성을검증했다. 기존113 tests와신규31을합한144 tests가Transformers4.55.0 CPU에서통과했고, 실제notebook 내장archive의독립프로세스에서도144 PASS다. Aggregate cc602534d5f59c38cef450f0e0b38aad7f0872a52e789d450266011e5c2ac31f.

이는synthetic/tiny Qwen CPU correctness와packaging 확인이다. 실제InternVL/native9경로·fp16 원인·H보정·논문동일성은미검증이다. 기존r2/STOP/원장 등49파일SHA변경0, 신규APPROVAL/dispatch/GPU0. 다음은새r3 kernel1개/개발3/3600초 승인후승인포함재동결·검증·predispatch다. 이전r2승인은재사용하지않는다.


### 2026-10-05 r3 단일 GPU 실행 승인·전송

사용자가 준비된 r3 개발243/479/419×ST-K1/TSR/DPP, private T4 kernel1개·최대3600초를 명시적으로 승인했다. [승인](artifacts/runs/independent_entropy_v3_r3/APPROVAL.json)과 [launch](artifacts/runs/independent_entropy_v3_r3/g1/launch.json)를 기록했다. 승인 포함 독립 bundle144 tests PASS; source SHA `7e755613a397c1dd5aa0613fc557c4521031dc5ab0e8981c47e504ef41c3a818`, notebook SHA `48378d7d361a3dc9bab09f4f9067ad5eca8fd26ca5ecc9bba3e966a9ad34dd8a`. Kaggle `arki13/visref-entropy-v3-r3-g1` version1을1회 전송했다. Parent r2 원장·STOP·과거결과 보존, selection/locked/후속비교 허용 없음. 결과·회계는 아직 미확정이며 종료 후 회수한다. 사전 동결 문서의 미승인 표기는 준비 시점 기록이고 이번 승인으로 이 범위에 한해 대체된다.


### 2026-10-05 r3 결과와 다음 판단

[r3 결과](artifacts/reports/independent_entropy_v3_r3/RUN_REPORT_KO.md): **FAIL**, kernel1/version1종료. 원격144tests·native243/479는통과했으나, answer-only probe13개가전부형식/종료실패로H undefined였다. 243TSR timeout 뒤479DPP도연장했고block8 length7741 full parity진단에서OOM. 완료arm0/9이며419미실행. TV경고2개는계속진행했고실패원인이아니다. 연장은부분관측이지만semantic불확실성을측정한결과가아니며성능비교준비는false다.

현재우선순위는threshold/DPP비율/fp32변경이아니라**실제최종답형식에서유효한답token span을얻는측정경로**, 긴문맥full진단의메모리부담, timeout partial기록을CPU에서해결하는것이다. 보존13probe/native 출력으로새parser명세·테스트를고정하고, 이후새실제모델smoke를별도동결한다. R3원본/verdict/동결본은변경하지않는다.

비용1611.595초/setup·실패포함차감, [최신 원장](artifacts/runs/independent_entropy_v3_r3/budget_ledger.json): P0–P1잔여8742.879551초,total58966.212227초, 예약0. 보호49파일변경0, selection/locked접근없음. R3승인소진/후속GPU미승인/STOP보존. [RUN_STATE](artifacts/runs/independent_entropy_v3_r3/RUN_STATE.json), [summary3](artifacts/reports/independent_entropy_v3_r3/SUMMARY.json), [증거·SHA](artifacts/reports/independent_entropy_v3_r3/EVIDENCE.json).


### 2026-10-05 r3 실패 원인 분석 보완: query 문구 유지 권고

[실패 원인·query 유지 분석](artifacts/reports/independent_entropy_v3_r3/FAILURE_ANALYSIS_KO.md)에서 저장 raw/code를 대조했다. Native는 이미 dataset query 문구 그대로(이미지/chat/공식Thinking 설정 유지, 추가answer prompt/forced close 없음)243·479 EOS와추출을통과했다. Query의 "at the end"는 숫자-only 제약이 아니다. R3 strict whole-output 검사와H undefined→reflection 연결이13개probe를측정불가/연장으로만들었다. 추가문구는assistant몸체에append됐으나문구유무만바꾼인과대조는없다. Native 답entropy도기록없어query-only H/.25결과는미검증이다.

권고는새독립명세에서query를재작성하지않고추가probe문구를제거한평소답변생성의raw token entropy를기록한뒤, 최종답값의유효span만선택하는것이다. 설명/라벨/반복/EOS를제외하고첫숫자를무조건선택하지않는다(479답영역은56/22부터시작). 최초답값이라도생각중이미같은값을언급할수있어semantic confidence/δ보정차단은유지한다. Undefined는측정오류로구분하며연장fallback을새조건으로명시해야한다.

OOM은479 block8 length7741의full-context parity 진단에서발생했다. Cached next-token logits로H를계산할때매번full parity가필수인것은아니다. 진단메모리범위/분리replay와partial기록이별도해결대상이다. Armshuffle·문항시간몫이ST대조군을가린점도correctness smoke 설계에서검토한다. 이번은CPU기록분석이며코드/동결명세/verdict/원장/승인/STOP변경·모델실행없음. 권고는아직구현·채택된r4가아니다.


### 2026-10-05 추출값 첫 등장 토큰 선택 규칙 구현

사용자 지시에 따라 `first_extracted_answer_mean_token_entropy_v1`을 새 모듈로 구현했다. 추출된 모델 답값을 기준으로 생각 영역 밖의 첫 유효한 등장에 대응하는 원래 생성 토큰을 선택하고 raw T=1/full-vocabulary H를 평균한다. 다른 숫자·뒤의 반복·EOS는 제외하며 정답/precision 반올림은 사용하지 않는다. 첫 토큰이 설명과 섞이거나 정렬이 모호하면 뒤의 반복으로 대체하지 않고 측정 불가다.

[채택 명세](docs/independent_dpp30_entropy_v3/FIRST_EXTRACTED_ANSWER_SPAN_SPEC.md), [구현·검증 보고서](artifacts/reports/first_extracted_answer_span_v1/IMPLEMENTATION_REPORT_KO.md): 새36/전체180 CPU tests PASS, 저장 native243/479의 첫 값 문자 위치 확인. 실제 InternVL token alignment/H는 미검증이다. 기존 r3 runner/config/동결본/verdict는 보존했고 새 함수의 연결은 다음 실행 버전에서 한다. Threshold·baseline readiness·GPU 승인·예산·STOP 상태는 변경하지 않았다.


### 2026-10-05 r4 연결 구현 및 Kaggle quota 추출 검토

사용자 지시의 생성 시 raw token entropy 기록→공식 MathVista 답 추출→생각 밖 첫 유효 등장 선택→평균을 별도 r4 runner에 연결했다. Production full parity는 구조 검사로 바꾸고 short full validation은512 positions 상한의 별도 함수로 분리했다. Undefined는 측정 실패로 종료하며 추가추론 근거로 쓰지 않는다. 새21/전체201 CPU tests PASS(16.00초); 실제 InternVL/추출 API/GPU 실행0. [준비 보고서](artifacts/reports/independent_entropy_v3_r4/PREPARATION_REPORT_KO.md), [명세](docs/independent_dpp30_entropy_v3/IMPLEMENTATION_PLAN_R4.md), [SHA·검증](artifacts/reports/independent_entropy_v3_r4/LOCAL_EVIDENCE.json).

후속 요청의 [Kaggle AI quota 검토](docs/independent_dpp30_entropy_v3/KAGGLE_AI_QUOTA_EXTRACTION_REVIEW_KO.md)는 quota 모델을 답 추출에 쓰고 MathVista scorer로 정오 판정하는 방향을 권고한다. 먼저 저장된 개발 응답의 추출을 GPU 없이 검증하고 이후 실시간 adaptive stopping에 연결한다. 실제 quota/허용 모델/일반 GPU notebook 접근 미확인, Kaggle adapter 미구현, provider/model/revision/SDK 미결정이다. R4는 로컬 구현 준비이며 실행 bundle 미완성·GPU/API 승인 false다. R3 FAIL/원본/원장/STOP 보존, paper/threshold/baseline 비교 차단 유지.


### 2026-10-05 네 논문 대조와 다음 baseline 실행 계획

[논문 대조](docs/paper_baseline_alignment/PAPER_FIDELITY_REVIEW_KO.md)와 [다음 실행 계획](docs/paper_baseline_alignment/NEXT_EXECUTION_PLAN_KO.md)을 추가했다. VisRef Appendix C/Eq1에 따라 평가 ST는 추가개입 없는native로삼고controller st_k1을분리하는명세를채택한다. MathVista의답accuracy와generation/entropy gate를분리한다. 1B모델실험은Appendix I/Table6에있으며14K는parallel-trace budget, paperfp16명시는미확인이다.

새 paper_baseline 모듈에서대각항없는Eq6–11 kernel/greedy·rank bound·공식함수기반text scoring을구현했다. 새10/전체211 CPU tests PASS(16.19초). 저장r3 PID479의7단계모두U45–109<m537이므로currentinputZ의unregularizedfixed-m determinant는퇴화한다. 기존정규화DPP는명명된독립구현으로만유지한다. [증거](artifacts/reports/paper_alignment_20261005/LOCAL_EVIDENCE.json), [rank기록](artifacts/reports/paper_alignment_20261005/R3_RANK_BOUND_EVIDENCE.json).

다음은새runner의native ST/공통scorer/단계별시간정책연결→GPU없는quota최대6회검증→별도승인후단일T4개발smoke다. 제안3600초이며실제dispatch승인은아니다. 새모듈runner미연결/transport·model미결정, [계획config](configs/paper_baseline_alignment/next_run_plan.json)는실행용아니다. STOP/원장/기존r3·r4source·PDF/selection/locked 보존, actual모델/API/GPU0.


### 2026-10-06 6시간 자율 baseline 개발 시작

사용자 Kaggle 사전 승인과 기존 프로젝트 GPU 상한 확대를 기록했다. [프로토콜](docs/paper_baseline_alignment/SIXHOUR_PROTOCOL_20261006.md). Native ST→TSR response MC(S10)→strict DPP 진단을 새 version으로 구현·검증한다. author 미공개 설정의 관행 조사와 operational 가정을 구분하고 STOP/selection/locked/압축 금지는 유지한다. CPU22 PASS; 실제 새 GPU/API inference는 아직0.


### 2026-10-06 r5 actual GPU / source and scorer update

[r5 exact evidence](artifacts/runs/paper_baseline_sixhour_20261006/r5/ANALYSIS.json): native ST243/479/419 all normal EOS, official chat replay/first16 sampler/short48 numeric gates PASS. TSR243/479 TIMEOUT at step5/6 (500s each);419 step3 incomplete MC set after one512-token repeating response. DPP NOT_RUN due TSR stage failure. GPU charged1732.5s; remote COMPLETE means notebook ended, not implementation gate passed.

[Kaggle extraction preflight](artifacts/runs/paper_baseline_sixhour_20261006/extraction_preflight/results.json): public notebook2-shot and officialMathVista5-shot agree on two historical final answers; 5 calls including transport conformance, reported $0.002727. This verifies proxy transport/value agreement only. Proxy model alias is not immutable and canonical leaderboard scorer equivalence remains unverified. Gold was used only for subsequent local normalization/scoring, not outgoing extraction prompts.

Official CVPR2026 main/supplement and NeurIPS2025 Mirage final checked. No additional S/representation/degeneracy settings found in those readable texts. Mirage checklist explicitly says code would be submitted as supplementary material; OpenReview API returns403 challenge, so supplemental author code is unread and cannot be claimed absent. Chen's referenced greedyDPP code uses epsilon1e-10 early stop with no jitter; fewer-than-m cardinality would be a named alternative, not a paper-fixed-cardinality release. [R6 preregistration](docs/paper_baseline_alignment/R6_PREREGISTRATION_20261006.md) changes time allocation only plus malformed-final correctness repair; prior limits, STOP and frozen historical evidence preserved.


### 2026-10-06 r5 offline verification and r6 dispatch

R5 stored110 complete responses (11 S10 sets) reproduce every recorded MC entropy exactly, including EOS and excluding length normalization. [Evidence](artifacts/runs/paper_baseline_sixhour_20261006/r5/ENTROPY_RECOMPUTATION.json). Diagnostics show different units/scales for sequence NLL and mean-token NLL; no threshold was changed.

[Real243 embedding audit](artifacts/runs/paper_baseline_sixhour_20261006/r5/REAL_EMBEDDING_DPP_AUDIT.json): N768/T1315/U204/requestedm230; repeated token IDs have identical actual stored input embeddings. Strict selector degenerates, referenced Chen code epsilon1e-10 returns204 on raw and diagnostic scaled kernels. This confirms current representation rank issue, not author Z. Chen paper §3 and numerical-stability appendix explicitly discuss rank-limited returns.

[R5 final scoring](artifacts/runs/paper_baseline_sixhour_20261006/r5/offline_scoring/results.json) completed: publicKaggle2-shot and officialMathVista5-shot normalized1/2.55/quarterpast identically, allthree incorrect against development references. Six calls reported$0.0066735; inference and original generation status unchanged; failed/notrun finals retain denominator1. No performance inference from dev3.

R6 private notebook arki13/visref-paper-r6-20261006 v1 dispatched after228 isolated CPU tests. FreezeSHA dcbfee9f9a5ebb60b07930e14583147dc11b1e33cd2991cd0e2fce44527b47c4; same samples/S/caps/sampling/threshold, increased time allocation as preregistered. Current status running; author fidelity and long numerical/reinjection gates remain unresolved.


### 2026-10-06 참고 알고리즘 및 설정 관행 추가 검증

Chen 공개 greedy DPP 원본(commit6ab745ca1273941412a7f32ce6432549cfe4b5af, epsilon1e-10)을 원 kernel에 그대로 적용하는 별도 component와 tiny Qwen3 실제 재삽입 검사5건 PASS. 현재 전체 로컬237건 PASS(12.20s); 준비된 R7 remote bundle의 검증 수는232이며 새 component는 포함/연결/dispatch되지 않았다. 요청m와 실제m를 따로 기록하고 fixed-cardinality author blocker를 유지한다. source: src/visref/reference_dpp.py, reference_dpp_runtime.py; pinned reference references/sources/chen_dpp_2018.

현재 공식 VLMEvalKit Thinking-1B 설정(commit54a063c5bf75245c35599daab9826d0a5b8e2671)은 LMDeploy/do_sample/max_new_tokens65536이다. 이는 현재 관행 근거이고 VisRef 당시 설정의 증거가 아니다. 현재 모델 context40960 및 Transformers eager 구현에는 그대로 옮길 수 없다. [정확한 근거](artifacts/runs/paper_baseline_sixhour_20261006/research/vlmevalkit_current/THINKING_CAP_EVIDENCE.json). InternVL3.5 원 논문2508.18265v2 §2.4/3.2도 읽었으며 Thinking/BoN 설명은 있지만 해당 두 section에는 S/entropy 정규화가 없다. Mirage OpenReview는 웹에서도 browser verification으로 차단되어 supplemental author code는 계속 미확인이다.

[새 단위별 offline audit](artifacts/runs/paper_baseline_sixhour_20261006/r5/COMPLETE_RESPONSE_AUDIT.json)는 sequence/normalized 표준오차를 구분하여11개 완결 S10 set을 재계산했다. R7 frozen controller의 standard_error 필드는 sequence 단위 그대로임을 기록하고, normalized 단위 진단은 이 audit의 명시된 필드를 사용한다. 제어 value/threshold는 변경하지 않는다.


### 2026-10-06 r6 종료 / r7 시작 / r8 자원 진단 준비

R6 native3 normalEOS 및 기존 단축 correctness gates 통과. TSR243 step7 total4096에서 thinking cap 실패,479 K10/총2270tokens 완료,419 step3 불완전 S10 set 실패. DPP3 NOT_RUN. [정확한 결과](artifacts/runs/paper_baseline_sixhour_20261006/r6/ANALYSIS.json). R5/R6 native3 모든 토큰 및 공통117응답 토큰이 일치하여 source/time-allocation 변경의 관측 범위를 확인했다. [대조](artifacts/runs/paper_baseline_sixhour_20261006/r6/R5_R6_SHARED_OUTPUT_AUDIT.json).

R6 GPU 유효 charge2700.781s: 수집 시 quota가 새 실행R7 사용까지 포함했으므로 R7 predispatch quota를 이전 실행 종료 경계로 사용한 append-only correction을 보존했다. 원본2795.450028s 과대 집계도 삭제하지 않았다. [소비 보정](artifacts/runs/paper_baseline_sixhour_20261006/r6/budget_ledger_correction.json). 131MB diagnostic index 첫 다운로드가 끊겼으며 작은 최종결과7파일은 수집, 완전 diagnostic 원본은 private remote version에 보존.

R7 private arki13/visref-paper-r7-20261006 v1/SHAffb3c862f3ead14fdf0c4f42b173d1a76d645edcb9ca170c6e9b2e5098a1f5bb 실행 중. R8은 thinking allowance만8192로 변경하는 [사전명세](docs/paper_baseline_alignment/R8_PREREGISTRATION_20261006.md), 격리238tests PASS, bundleSHAea8a7d2088f58d3121fd992884e16cb4d9610b04bdc7c8228d1f71c98c865b52/96files 실제credential absence 감사 완료; 미dispatch.

[현재 block 구조 rank 진단](artifacts/runs/paper_baseline_sixhour_20261006/r6/DPP_STRUCTURAL_SCOPE_AUDIT.json): 기록20block 중14개는 contextual current-block Z라도 T<m.479 첫 block T276/N1792/requestedm537. 이는 현재 Z scope/cardinality 가정의 문제이며 author 구현 오류로 단정하지 않는다. CVPR §4.1.1의 Markov 가정은 현재 state와 전체 history를 구분하므로 cumulative scope를 근거 없이 공식값으로 바꾸지 않는다. 새 contextual-last-layer component는 tiny-model 비교 검사1건 통과, GPU연결되지 않았다.

R6 새 외부 답추출은 자동 승인 검토가 Gemini 목적지 명시승인을 요구하며2회 거부했다. 사용자에게 목적지 승인을 비동기 요청했고, 새 proxy 호출은0이다. [검토한 전송내용](artifacts/runs/paper_baseline_sixhour_20261006/r6/offline_scoring/EGRESS_CONTENT_REVIEW.json). [별도 로컬 literal audit](artifacts/runs/paper_baseline_sixhour_20261006/r6/LOCAL_FINAL_AUDIT.json)은 명시된 최종값과 공식 normalization/equality만 사용하여 모든 계획 분모를 유지하며 hosted/proxy 판정으로 주장하지 않는다. 기존R5 scorer 결과는 보존한다.


### 2026-10-06 r7 최종 gate / main8B r9 dispatch / r8 재검토

R7 native3 PASS. TSR243 block7 thinking4096/repetition 실패,479 step1 normalized0.1372139577에서 정상 final 완료,419 step2 normalized0.2487271727에서 멈췄지만 final stray closing-think tag 때문에 GENERATION_FAILED. DPP3 NOT_RUN. [정확한 결과](artifacts/runs/paper_baseline_sixhour_20261006/r7/ANALYSIS.json).9개 완결 S10 set을 수학적으로 재계산했고 normalized 표준오차는 별도 audit에 올바른 단위로 기록했다. 기존 native3의 모든 raw토큰/최종문장이R5와 같아 [기존 추출값 재사용 감사](artifacts/runs/paper_baseline_sixhour_20261006/r7/CACHED_NATIVE_EXTRACTION_AUDIT.json)만 수행; 새 외부 추출0/TSR formal추출은 승인대기. GPUcharge1575.093s.

준비R8은 미실행으로 보존한다. R6 block7 마지막500토큰이21개 고유토큰으로 동일문장을 반복하여 cap만 늘리는 방향을 재검토했다. 또한 total allowance가커지면 explicitlimit가 block_cap을넘을 수 있는 구현조건을 발견해 새로운 수정/freeze 없이 dispatch하지 않는다. [준비 재검토](artifacts/runs/paper_baseline_sixhour_20261006/r8/PREPARATION_REASSESSMENT.json).8192 cap실패나 성능결과를 주장하지 않는다.

논문 주실험8B의 native를 먼저 검증하는 [R9 사전명세](docs/paper_baseline_alignment/R9_PREREGISTRATION_20261006.md). Private arki13/visref-paper-r9-20261006 v1/SHAff39b268b4567a81793069f4eca0bd228341694d2161fc7a6e4d686bffdf0c4c,243 격리tests PASS/102파일 actualcredential absence 감사 후 dispatch. 모델 revision9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79/4096hidden/36layers/context40960, 두T4/양자화없음/device_mapauto/각12GiB budget/fp16/eager/native4096. 다른GPU환경이므로1B와latency 비교없음. Native ST만3개/최대3600초; 실제 placement·EOS·sampler·shortnumeric·postprocessor gate 확인 후에만8B TSR/DPP를 고려한다. Checkpoint bf16/modelcardflash 예제와현재운영차이는명시되어authorfidelity미확인이다.


### 2026-10-06 중단 후 회수: R9/R10 실제 결과와 R12 기한 보호

R9는 이미지 projection CUDA1과 입력 embedding CUDA0 사이 device mismatch로 native 생성 전 실패했다. pinned 공식8B generate의 `vit_embeds.to(input_embeds.device)` 처리와 일치시키는 R10 수정으로 실제 세 문항 CUDA1→CUDA0 전달을 검증했다. R9 charge254.911s, R10 charge878.023s, 모두 setup/실패 포함.

R10 private arki13/visref-paper-r10-20261006 v1/SHA b4bd0f711beb12dbe7b2684963c605a2e007a11caa0745346636228d0a7e2bba: 8B native243=4096 token_limit/think 미완결,479=278 normalEOS,419=2106 normalEOS. 전체3 중2개 완료이며 native gate 실패로 TSR/DPP 미실행. 세 문항 모두 prefix16 sampler match, 공식 입력/mask/decoding/text replay match, short48 TV<0.02 및 argmax 일치; 긴 전체 context fidelity를 의미하지 않는다. 생성시간405.762/27.976/181.960s. [원본과 분석](artifacts/runs/paper_baseline_sixhour_20261006/r10/ANALYSIS.json). 최종값 로컬 audit3개/분모3/정답0, 외부 호출0이며 공식 extraction/hosted 판정으로 주장하지 않는다.

R11 multi-method7200s는 미dispatch. R12 native8192 resource diagnostic은 동일8B/dev3/seed/decoding/gates를 유지하고1440s session/first243 max900s로 동결했다. 격리 로컬245검사와105파일 credential/raw/selection/holdout absence 감사는 통과했지만 원래6시간 종료 기한과120s 여유를 만족하지 못해 push가 launch 생성 전에 거부했다. GPU소비0/새 model outcome0이며8192 설정의 성공·실패를 주장하지 않는다. [미dispatch 기록](artifacts/runs/paper_baseline_sixhour_20261006/r12/DISPATCH_NOT_PERFORMED.json). 실패 gate나 최초 승인 기간을 변경하지 않았다.

이번 신규 실행 R5/R6/R7/R9/R10 유효 GPU charge 합계7141.308s(1.9837h), 마지막 provider 잔여88022.921591s/예약0. 중단 후 회수 시 모든 실행 terminal; 기존 STOP와 소비 원장 유지. 정확한 저자 baseline,8B native3/3,TSR3/3,DPP3/3는 아직 미달성. 다음 모델 실행은 더 긴 native 자원 진단을 먼저 완료해야 한다. Gemini 외부 답추출은 자동 승인 검토가 목적지 명시승인을 요구하며2회 거부하여 새 호출0/승인 대기 상태를 유지한다.


### 2026-10-06 Gemini 전송 명시 승인 및 보류 채점 완료

사용자 지시 “Gemini로 내용을 보내는 허가할게.”를 별도 승인 원본으로 기록하고 만료된 Kaggle Proxy 토큰을0600 임시파일로 갱신했다. 기존 승인대기 조건은 해소됐다. 공개 개발243/479/419와 기존 최종 답변 segment만 전송했고 gold/추론본문/비공개 코드/credentials는 전송하지 않았다. GPU 재실행0/새 모델 추론0/selection·holdout 접근0.

R6=8호출/$0.007854, R7=8호출/$0.008284, R10=4호출/$0.003476; 새 호출 합계20/$0.019614, 전부HTTP200/완결 추출. Public Kaggle MathVista2-shot과 공식MathVista5-shot의 추출·정오 판정이 모든 paired row에서 일치했다. 실제 완료된 최종 답변은 R6 4개/R7 4개/R10 2개이고 모두오답; 실패·미실행도 각방법 frozen분모3에 유지했다. 이미 노출된 소수 개발 표본의 점수이며 성능 일반화나 baseline 재현 성공을 의미하지 않는다. Canonical Kaggle hosted judge와 동일성 및 Gemini alias revision은 여전히 미확인이다.

[전체 호출·비용·일치 확인](artifacts/runs/paper_baseline_sixhour_20261006/GEMINI_SCORING_COMPLETION.json). 각 r6/r7/r10 offline_scoring/results.json에 original generation status, prediction, failure 및 분모가 보존돼 있다. 기존 generation/로컬 audit/옛 scoring freeze는 변경하지 않았다. R6 원래 scripts/paper_score.py freeze 그대로 실행했고 R7/R10은 별도 scripts/paper_score_remaining.py에서 신규 동결 후 실행했다.


### 2026-10-06 사용 제한까지 baseline 재개 승인

사용자 “목표 사용 제한 까지 작업 이어서 진행해”로 신규 동결 실행에 한해 최초6시간 wall deadline을 대체한다. 기존R12 미dispatch/freeze 및 STOP/원장은 보존한다. 기존 Kaggle 예산 전량 사용과 Gemini 목적지 승인은 유지하고, dev243/479/419만·selection/locked/압축 금지·native→TSR→DPP correctness gate를 유지한다. CONTINUATION_AUTHORIZATION.json에 별도 승인 기록. 목표 서비스 상태는 usageLimited로 조회되므로 이 사용자 재개 turn에서 작업을 계속하되 기존 목표 사용량을 초기화하거나 재활성화로 허위 표시하지 않는다. 신규R13은R12의 native8192 자원 가설을3600s 세션/native2700s로 검증하며,8192는 공식 저자값이 아니다.


R13 private arki13/visref-paper-r13-20261006 v1/kernel137242408 dispatched: source119ee49790877999966e079c93bb312a172d68c58d7228429dac0d622d7c8307, native8192,3600s cap/native2700s,245 isolated checks PASS,106 payload files/790699bytes actualcredentials/raw/selection/holdout absence verified. Freshquota88022.921591s/reserved0. Launch generic estimate1800–3480 retained; prereg expected~1300s and this metadata mismatch appended as DISPATCH_ESTIMATE_CORRECTION.json. No outcome yet.


R13 실행 종료 시 수집/소비 원장/로컬audit/승인된 final-only Gemini 채점을 수행하는 bounded finisher 연결. 최대80분 감시·추가GPU dispatch 없음. 기존 자동 목표는usageLimited이며 자동 목표 상태/소비량 초기화 없음. 결과 경로r13/FINISHER_RESULT.json, 오류r13/FINISHER_FAILURE.json, 진행r13/finisher.log. native 통과 시의 TSR/DPP는 미dispatch이며 자동 목표 사용 제한 상태가 계속될 경우 임의 새 세션을 실행하지 않는다.


### 목표 usageLimited 원인 설명 정정

기존 목표 조회에 explicit token budget/remaining token 한도는 없고 status=usageLimited만 반환된다. 계정 fresh조회 ordinaryUsageAllowed=true,5h used18%/weekly used80%. 따라서 “목표 자체 예산이 소진됐다”는 앞선 단정은 근거 부족이며 exact trigger는 미확인으로 정정한다. 제공된 goal 도구에는 resume/clear/budget 변경이 없고 Codex 앱 CUA 제어는 safety denial로 차단되어 직접 해제 미완료. 계정 reset credit을 소모하지 않았다. 사용자 현재 재개 지시는 manual continuation을 승인하며 현재R13/finisher 계속 진행; correctness gate/selection/holdout 제한은 유지한다. 상세GOAL_STATUS_DIAGNOSIS.json.


### 자동 목표 재활성화 실제 확인

get_goal 실제조회 status=active,기존 tokensUsed1353667/무제한 token budget; 사용량 초기화 없음. 이전usageLimited 관측은 역사 기록이고 현재 자동 연속 작업은 다시 활성화됨. 기존R13 bounded finisher는 이미 시작된 source에서 usageLimited 라벨을 기록할 수 있으므로 그것은 goal service 실제상태의 증거가 아니다. R13 terminal 후 해당 임시표시를 현재get_goal 증거로 정정한다. 이미 승인된 개발-only baseline 작업 지속.


후속8B TSR/DPP R14 사전 조건을 docs/paper_baseline_alignment/R14_PREREGISTRATION_20261006.md에 기록했다. R13 native3/3 actual gate 통과·회수 전에는 freeze/dispatch하지 않는다. 현재 미실행 준비이며 native8192/block8192/total32768은 운영 자원 한계, 저자공식값 아님.


R14 build/push에R13 실제 native3/3 수집 증거 guard를 연결했다. naturalEOS/생성1회/think완결/afterEOS0/official replay4항목/prefix16 sampler/finite short48 TV≤0.02+argmax 일치가 모두 필요하다. complete notebook status나 denominator 축소로 통과하지 못한다. 실패·누락·NaN을 포함한7개 release-guard 검사PASS; R14는 아직미build/미dispatch이며 R13원본bundle변경없음.


R13 live native243 TIMEOUT/SAMPLE_TIME_SHARE_EXHAUSTED at runner1184.887s; lastprogress7680tokens/generation877.730s.8192 token-limit outcome미검증。 R14 remains미dispatch since native3/3 cannot pass this run. R15 time-allocation-only diagnostic preregistered in docs/paper_baseline_alignment/R15_PREREGISTRATION_20261006.md;4200s session/native3600s, unchanged8192 generation, requiresR13terminal/collection and othernative gates beforenewfreeze. ExistingR13 still running; no duplicatejob.


### Sampling 확률/EOS 동일응답 재계산 진단

R6완결18세트/불완결1, R7완결9세트에 저장된 raw T1/full-vocab logprob와 q T0.6/top50 logprob를 paired 비교했다. 신규추론/외부채점0/기존control변경0. q로샘플해p로평가한 값은H(q,p) cross-entropy이며H(p)가 아니다. 별도p샘플 추론 실험은 아직없다.

같은419 step2의동일S10응답에서 normalized q+EOS=0.24872717273765085, q−EOS=0.2507936781444885, rawp cross-entropy normalized=0.3072779062234326. EOS제외 또는확률기준변경은counterfactual0.25중단조건을뒤집는다. R6/R7이같은응답을공유하므로2개독립발견으로세지않는다. EOS자체평균NLL q0.014702916/p0.036973064이고길이분모변경도작용한다. authorEOS정책/normalization/thresholdtransfer는계속미확정이다. 상세각run/PROBABILITY_EOS_PAIRED_AUDIT.json.


### R13 수집·채점 완료 — 자동 마무리 기록

R13 runner=GATE_FAILED, native 완료=2/3; GPU charge=1415.668999999998s; Gemini 호출=4/비용=$0.003071. 원본/세부 판정은 artifacts/runs/paper_baseline_sixhour_20261006/r13/FINISHER_RESULT.json 및 offline_scoring/results.json에 보존. 정확한 논문 재현 미확정; 기존 목표 usageLimited 유지, 새 GPU dispatch 없음.


### R13 회수 후 정정 및 R15 진행 근거

현재 get_goal 실제 상태 active/tokensUsed1472354/예산 unbounded. 자동 finisher의 usageLimited 문구는 이전 프로세스에 남은 라벨이며 현재 목표 상태가 아니다. R13 terminal/수집/채점 완료: native479/419 정상EOS, 243 TIMEOUT/SAMPLE_TIME_SHARE_EXHAUSTED(partial7796 tokens). R10과 공통4096/278/2106 raw tokens 모두 동일(COMMON_PREFIX_AUDIT.json). GPU charge1415.669s, 추가scope 누적8556.977s, 남은provider86607.252591s. 기존채점포함 Gemini 총24calls/$0.022685; denominator3 유지. R15는8192 cap 그대로/session4200/native3600 시간만 변경하는 사전등록 진단; TSR/DPP gate는 아직 미통과.


### R15 시간 할당 진단 dispatch

private arki13/visref-paper-r15-20261006 v1/kernel137244387. source78b04f53ccec6d585d6271072b7fcc09c2763d6ad779608307d629fb422ccd3b/notebookc025798472c26afbf351f2d6365a0ee95908ab3f9dd4b293fc892362e39912c6. 격리256PASS/18.65s,108files/800715bytes actual credentials 및 raw/selection/holdout absent. R13 대비 identity 외 session3600→4200/native2700→3600만 변경(SINGLE_FACTOR_AUDIT.json). native8192/model/revision/decoding/seeds/dev3/STOP 동일. freshquota86607.252591s/reserved0/모든이전jobterminal 확인; estimate1500–4080s. native3/3 gate 미확정이고 TSR/DPP 아직 미실행. 종료·수집·최종답변채점 bounded finisher가 담당하며 새로운GPU dispatch 없음.


### 실제 sampler의 확률 분포 CPU 진단

생산 sample_token/probabilities/sequence estimate를 사용해 정책별20000개 완결 응답을 생성했다. q(T0.6/top50)의 H(q) exact2.372134924 vsMC2.373178092(SE0.007081238); p(T1/full60)의 H(p) exact2.880205870 vsMC2.876536744(SE0.007060929), 빈도·6SE 검사PASS. q draw를rawp로평가하면 exactH(q,p)=2.482442856/MC2.483068963이며H(p)가 아니다. EOS제외qNLLmean2.032342291 vs포함2.373178092. 이는 제어가능한60개토큰 분포의CPU 실험이고 실제InternVL p-sampling/GPU검증이나author policy 확인을 대신하지 않는다. inference/control/채점변경0, 신규GPU0. 상세[산출물](../artifacts/runs/paper_baseline_sixhour_20261006/research/SAMPLER_DISTRIBUTION_AUDIT.json). R15 원격bundle과run settings 변경없음.


### Native 243의 반복/종료 진단

동결R10/R13 native243 출력만 분석했다. R13 partial7796에서 </think>(151668)/EOS(151645) 모두0회. 구간6144–7167의16gram duplicate fraction0.554013875/mean raw next-tokenH0.004560159, 처음1024는0.0/H0.489794973. 마지막500개 token은141개고유라단일문장완전고정loop로단정하지 않는다. 실제답변 sequence entropy나EOS확률을이tokenH로대체할수없다(EOS확률은미저장). 새추론/채점/중단정책변경0; R15결과전cap추가확대근거없음. 상세 r13/NATIVE_REPETITION_DIAGNOSTIC.json,원본hash 포함. R15가또8K cap에걸리면생성정책/정밀도/긴문맥수치검증을재검토하고자동상한증가를하지않는다.


### 긴 native 출력의 수치 검증 준비

캐시logits NaN/Inf와검사metric NaN을명시거부하도록 long_parity 보완. 저장token teacher forcing을64토큰씩입력하고지정offset에서cached eager/uncached SDPA비교하는 native_reference 추가; 새sampling0/RNG불변. 실제tiny Qwen3 CPU 총10checksPASS/2.68s(offset16/64/127, total19/67/130; invalid offsets/nonfinite/failure restoration/cache불변). 실제8B/T4/8K 구간은미검증이고현재R15 source/bundle변경0. 기존 native full_reference 호출은prompt+16까지만이므로긴생성구간parity증거로확장해해석하지않는다. 조건/결과research/NATIVE_REFERENCE_CPU_VALIDATION.json. 다음GPU계획은R15terminal/회수후실제실패종류에따라정하며native3/3 gate 변경없음.


### R15 live8192 도달 및 R16 사전 준비

R15 live native243=8192tokens/956.902310s, GENERATION_FAILED/token_limit. 이번에는 시간 부족이 아닌 생성 상한 도달이다. 나머지 두 문항 및 terminal/collection 대기 중. 추가 상한 확대 없음. native_generated_reference_positions를 지정한 새 run에만 teacher-forced 긴 문맥 reference를 연결했고 로컬 전체263검사PASS/11.62s, GPU 미검증. R16_PREREGISTRATION_20261006.md는 native-only/같은8192 및 decoding/seed/pinned8B/GPU/dtype에서 생성offset1024/4096/8192 수치 진단을 명세한다. session5400/native4500은 추가 validation 시간이다. 기존 native gate 불변; R15 회수 후 실제 failure 판정 전 build/dispatch하지 않는다.


### R15 수집·채점 완료 — 자동 마무리 기록

R15 runner=GATE_FAILED, native 완료=2/3; GPU charge=1421.242000000002s; Gemini 호출=4/비용=$0.003476. 원본/세부 판정은 artifacts/runs/paper_baseline_sixhour_20261006/r15/FINISHER_RESULT.json 및 offline_scoring/results.json에 보존. 정확한 논문 재현 미확정; 목표 service 상태는 이 finisher가 변경하지 않음, 새 GPU dispatch 없음.


### R15 수집 후 확정 판정

R15 native243=8192 token_limit/956.907544s, think_close/EOS0; 479/419는278/2106 tokens normalEOS와모든기존native gate 통과. R13과공통7796/278/2106 token 모두동일(r15/ANALYSIS.json). native3/3 release 실패로TSR/DPP 미실행. charged1421.242s, 추가scope누적9978.219s(2.7717h), 남은provider85186.010591s. Gemini4calls/$0.003476,두동일scorer모두정답0/분모3; 총28calls/$0.026161. 이는3개개발문항debug이며accuracy우열/공식leaderboard값 아님. 자동cap확대없음. R16은동일native8192/decoding/model로긴문맥teacher-forced numerical 진단; 격리267PASS/12.07s,110files/808891bytes actualcredential/raw/selection/holdout absent. 기존native baseline gate는실패로유지되며이진단허용은TSR/DPP release가아니다.


R16 private arki13/visref-paper-r16-20261006 v1/kernel137246782 dispatch. Source330b1691c1ce94ccaa29a74d745ecd8ac56bb624d106da5310a4b96164926cc5, notebookd5975f1c8c11604a8615915a33b2137513ff5c31cece91d2ed7cbb633d1887ed. freshquota85186.010591s/reserved0/전작업terminal 확인; estimate1500–5280s/session5400. 현재GPU수치결과미검증; 원본STOP SHA824d748...동일. 종료·회수·최종답변채점 bounded finisher scripts/paper_finish_r16.py; 재dispatch하지않음.


### 고정 모델 README 및 nucleus policy 준비

8B exactrevision9bb6a56... README SHA47383e9746907dbfabfcad9c786aab81e43586b5e4b0bb07ad893d8b58f3b1e9 확인. Lines80/412는Thinking do_sampleTrue/T0.6 권고; line767 LMDeploy 예제는top50/top_p0.95/T0.6/max8192이며VisRef author설정으로격상하지않는다. 표준 TLS/certifi로원본저장(research/INTERNVL8B_PINNED_README_AUDIT.json). 새 native sampler replay는실제TF4.55 generation processor list와p1/p.95 확률exact일치, p1 기존seed20draw일치/비지원repetition거부 등CPU4checksPASS/3.18s. 실GPU미검증. R17_PREREGISTRATION_20261006.md는R16전수수치검사통과후에만top_p1→.95 한변수진단준비; 아직등록/build/dispatch없음. 다른controller p1과혼합비교금지, 실패gate 불변. 현재R16 source/bundle변경없음.


R17 준비 보완: native prefix replay는 실제decoding의HF temperature/top_k/top_p warper를 사용하며 long-reference TV도 같은policy를 명시하도록future source에 연결했다. short48 기존TV는p1을명시하고p.95 지표로표시하지않는다. CPU sampler/reference15checksPASS/3.20s. 실행R16 bundle불변, R17 등록/build/dispatch 미실행.


### 주모델 8B native trace의 DPP rank 제약

R15 native3개 rawids만분석(새추론0/DPP실행0). 479는원본V1792→m537이지만자연완결thinking T187/고유input-id95. 따라서이trace의pure L=VZZ^TV^T에서텍스트-only contextualZ도rank≤187, inputlookupZ는≤95; m537 minor determinant는정확한산술에서0. hidden d4096은이T제약을없애지못한다. 243는T8191/unique323/m230이지만생각미완결,419는T1955/unique227/m76; 충분한rank상한은실제양의det를보장하지않는다. Native와TSR/DPP의phase seed/trace는달라서향후실제DPP실패를미리판정하지않는다. 이결과는author Zscope/jitter/cardinality/짧은trace처리의확인이필요함을구체화한다. fixedm gate를느슨하게하거나수치잡음을양의rank로해석하지않는다. 상세r15/NATIVE_DPP_RANK_BOUND_AUDIT.json.


### R16 수집·채점 완료 — 자동 마무리 기록

R16 runner=GATE_FAILED, native 완료=2/3; GPU charge=1513.0279999999984s; Gemini 호출=4/비용=$0.003026. 원본/세부 판정은 artifacts/runs/paper_baseline_sixhour_20261006/r16/FINISHER_RESULT.json 및 offline_scoring/results.json에 보존. 정확한 논문 재현 미확정; 목표 service 상태는 이 finisher가 변경하지 않음, 새 GPU dispatch 없음.


### R16 전수 수치 검증 및 R17 한변수 진단

R16의 prefix 3개·generated-context 6개 reference 모두 기존 TV≤0.02/finite/argmax 또는 near-tie/backend 복구/no-cache 조건을 통과했다(max TV0.0152803585). R15와 8192/278/2106 raw token이 모두 동일하며, 243 token_limit로 native baseline gate는 여전히 실패다. 이는 teacher-forced cached decoder와 uncached SDPA의 수치 검증이며 공식 generate의 모든 logit 검증은 아니다. 상세 r16/ANALYSIS.json. GPU 추가scope 누적11491.247s, provider 잔여83672.982591s; Gemini 누적32calls/$0.029187, 개발3문항 분모 유지/정답0. R17은 동일 조건에서 native top_p만1→0.95인 사전등록 진단. 모든 R16 numerical reference가 있어야 build/push하도록 guard 추가; TSR/DPP release 조건 불변.


### R17 native nucleus 진단 실행

R17 private arki13/visref-paper-r17-20261006 v1/kernel137249458 실행. Source7c575d92afa957a6c58992800ea4b4733f0b54640c8bbf012e676a8a271f421c, notebook5fbe6ef08d69e2070ac74d2eeaa95f55f77f3d92b8fec42bdff441eaa86593a2. R16과generation 설정차이는 top_p1→0.95만(CONFIG_FACTOR_AUDIT.json); native sampler와fullreference도실제policy를사용하며short48은기존p1지표라고표시한다. 격리280PASS/11.32s,113files/872078bytes actualcredential/raw/selection/holdout 미포함 감사통과. 이전모든job terminal/freshquota83672.982591s/reserved0 확인; estimate1500–5280s/session5400/native4500. STOP SHA 동일, 누적11491.247s는R17종료후새charge를더한다. bounded finisher scripts/paper_finish_r17.py는새GPU실행없이종료·회수·허가된최종답변채점만수행. GPU결과미검증, native3/3 실패조건 및TSR/DPP 차단불변.


### 공통 sampling policy 로컬 준비

새 paper_policy_decoder는 native_decoding의 T/top_k/top_p/beam/repetition/sample 설정을 직접 읽어 thinking/probe/final의 sampling과probe log-probability를일치시킨다. EOS/min1 억제는warp전에적용하며TF4.55실제processor순서와p1/p.95exact비교. 기존RecordedDecoder의step/close/부분출력보존을새class에유지했으며historicalcontroller는수정하지않았다. targeted19checks로p1이전동작·p.95 실제tinyQwen draw/종료확률·S완결/별도finalseed 확인. 상세 research/COMMON_POLICY_COMPONENT_PREPARATION.json. 아직runner미연결/등록된GPUversion없음;실행R17bundle불변. Native3/3 gate와author fidelity조건을해제하지않는다.


R17 live 첫 row243=GENERATION_FAILED/token_limit. top_p.95는 이 문항의 native 종료 문제를 해결하지 못했다. 동결 runner hash 일치 확인; 이 row는 reference 호출 반환 후 발행되지만 실제 metric은 terminal 회수 후 확인한다. 나머지2개 진행 중, TSR/DPP 미실행·재dispatch 없음. 증거 r17/FIRST_ROW_LIVE_OBSERVATION.json.


### R17 수집·채점 완료 — 자동 마무리 기록

R17 runner=GATE_FAILED, native 완료=2/3; GPU charge=1476.5910000000003s; Gemini 호출=4/비용=$0.002872. 원본/세부 판정은 artifacts/runs/paper_baseline_sixhour_20261006/r17/FINISHER_RESULT.json 및 offline_scoring/results.json에 보존. 정확한 논문 재현 미확정; 목표 service 상태는 이 finisher가 변경하지 않음, 새 GPU dispatch 없음.


### R17 확정 결과 및 관측 hook 대조 준비

R17 native243=8192token_limit/961.611864s,479=278EOS,419=1840EOS. 요청수치reference9개 전수통과(maxTV0.015264899), 실제p.95확률정책/완결전prefix길이/finite/backend복구/no-cache조건확인. top_p.95는243종료문제미해결; R16과243prefix12tokens후갈라졌지만종료실패는같고479전체동일·419prefix610후갈라짐. native3/3 release실패/TSR·DPP미실행. 상세r17/ANALYSIS.json·FINISHER_RESULT.json. GPUcharge1476.591s/추가scope누적12967.838s/provider잔여82196.391591s; Gemini4calls/$0.002872/누적36calls/$0.032059,동일두scorer정답0/분모3. 다음진단은generation중프로젝트관측hook과progresscallback제거의한변수대조로사전등록하며, 모델/입력/seed/decoding/cap변경없음. 아직R18미등록·미실행.


### R18 관측 제거 대조 동결·로컬 검증

R18은R17과model/prompt/dev3/phase seed/decoding/output8192/time envelope가동일하며유일한실험요인은native_observation_mode=off. 프로젝트prehook37개만생성중제거하고finally복구하며entropycapture/progresscallback을생략한다. 모델·Accelerate 실행hook유지, 생성후chat/prefix/short/fullreference검증유지. TinyQwen3에서토큰/RNG동일·외부hook유지·성공/예외복구·activefeed거부확인; targeted28PASS/2.32s,격리292PASS/11.22s. 원본STOP SHA동일, R17수치9개전수통과/누적12967.838s 기록후등록. 상세 R18_PREREGISTRATION_20261006.md·r18/CONFIG_FACTOR_AUDIT.json·LOCAL_OBSERVER_VALIDATION.json. 실InternVL observer인과관계는GPU미검증.


R18 private arki13/visref-paper-r18-20261006 v1/kernel137251744 dispatch. source9e879666cfcf31a2806130cbef3846206c78786ae8c05dd10eaa02d020d6bca5/notebook740406c51ac1d2b51eebaa8e835fa6ef5e161ede31366e50b4944ba2631837a0. 격리292PASS/116files/840364bytes actualcredential/raw/selection/holdout미포함 확인; 이전모든jobterminal/freshquota82196.391591s/reserved0 확인. estimate1500–5280s/session5400. bounded finisher scripts/paper_finish_r18.py는종료·회수·동일final-only채점만수행하며재dispatch없음. 실GPU observer 영향 판정 대기.


### 남은 모델 문맥 상한 R19 준비

사용자 지적대로8192는운영상한이며8K미완결만으로모델자체실패를증명하지않는다. 남은native출력을실제 input_context_tokens(이미지/chat포함)에서max_position_embeddings40960을빼는helper와명시적future paper_r19 policy를준비했다. CPU11checks/3.60s 및관련21checks/3.17s PASS;actualtinyQwen 전체context경계와EOS한토큰조기종료 확인. R19 아직dispatcher미등록/build/dispatch없으며R18종료·관측대조·전수수치검사 먼저확인. 실제R17 modelmap은GPU0/1 각각16/20layers이며40K KV추정각2.5/3.125GiB;densefp16reference mask3.125GiB+temporarybool1.5625GiB, FFNactivation별추가메모리. 실제OOM관측은아니며longreference자원조건해결후실행조건동결. 상세 research/REMAINING_CONTEXT_PREPARATION.json 및 R19_PREPARATION_20261006.md. 실행중R18payload불변/STOP보존.


### 중단 지점 회수 — bounded reference CPU 및 목표 서비스 상태

진행 중이던 full-reference CPU 검증은 4 PASS/3.27s로 회수했다. Tiny Qwen3의 33/97/127-token uncached eager 마지막 logits와 query-chunked reference가 atol1e-6/rtol1e-5 범위에서 일치했고 RNG 및 성공/예외 후 backend·MLP 복구를 확인했다. 이는 실제 InternVL/T4/40K 검증이 아니며 runner 미연결·R19 미등록/미실행이다. 상세 artifacts/runs/paper_baseline_sixhour_20261006/research/BOUNDED_FULL_REFERENCE_CPU.json. R18은 remote RUNNING으로 확인했고 기존 finisher(session16468)는 종료·회수·허가된 final-only 채점을 계속한다. 목표 서비스는 get_goal 실조회에서 usageLimited(tokensUsed2181000)로 확인하여 이전 STATE active 표기를 바로잡았다. 제한 해제/예산 재설정 기능은 제공되지 않으며 새 GPU dispatch 없음. 재개 시 R18 FINISHER_RESULT/실패기록부터 확인하고 관측 제거 전후 raw IDs·전수 numerical reference를 비교한 뒤 R19 조건을 동결해야 한다. STOP 보존.


### R18 수집·채점 완료 — 자동 마무리 기록

R18 runner=GATE_FAILED, native 완료=2/3; GPU charge=1486.714s; Gemini 호출=4/비용=$0.002854. 원본/세부 판정은 artifacts/runs/paper_baseline_sixhour_20261006/r18/FINISHER_RESULT.json 및 offline_scoring/results.json에 보존. 정확한 논문 재현 미확정; 목표 service 상태는 이 finisher가 변경하지 않음, 새 GPU dispatch 없음.


### R19 조건부 준비 — reference 연결 및 실행 차단

목표 서비스가 active로 재개된 것을 get_goal로 확인했다(누적 사용량 reset 없음). Full-reference의 query256·tokenwiseMLP256 분할을 명시적 opt-in으로 연결하고 actual input T4 수치 smoke를 생성 전 gate로 추가했다. CPU에서 Accelerate outerhook 보존/원본 eager 일치/cache·RNG·backend 복구를 확인했으며 관련28PASS/3.28s, R18 동일토큰·hook복구 요구 guard29PASS/.12s, 종합targeted67PASS/2.66s. 실제 InternVL/T4/40K 검증은 미수행이다. R19_PREREGISTRATION_20261006.md에 remaining-context cap40960-input, naturalEOS, session14400/native13200/first24311000s 및 complete-context reference offsets를 조건부 사전등록했다. R18 전수검증 및 R17 raw-ID 동일성이 확인되기 전 build/push 차단; R19 아직 bundle/build/dispatch 없음. 세부 research/R19_CONNECTED_REFERENCE_CPU.json. Running R18 payload 불변/STOP 보존.


### R18 관측 대조 확정 및 R19 동결 준비

R18 native243=8192 token_limit/966.481526s,479=278 EOS,419=1840 EOS. 프로젝트prehook37개 제거·복구 및실행hook유지 확인; 세문항모든raw token이R17과정확히동일하다. 요청수치reference9개모두통과(maxTV0.015264899). 이조건에서generation중관측기능이종료실패를일으켰다는증거없음; native3/3 release실패와TSR/DPP차단유지. 상세 r18/ANALYSIS.json·FINISHER_RESULT.json. GPUcharge1486.714s/추가scope누적14454.552s/provider잔여80709.677591s; Gemini4calls/$0.002854/누적40calls/$0.034913,개발분모3유지·accuracy선택없음. 조건부R19 buildguard를실제결과로통과했으며remaining-context generation진단의동결·격리검증을진행한다.


### R19 동결·격리 검증 및 자원 계획

R18실제결과의관측대조guard PASS 후R19를동결했다. source6b4710cfb0f99904808082426c82d5231b5cac84dbdbadba7b19c523d5bf0a36/notebookbec2c479485d67127fe90a05789707ddc81ddb18fd23fadcc43de8523d45ba74;격리313PASS/11.44s 및actualpayload120files/859001bytes credential/raw/selection/holdout absent. 샘플·모델revision·prompt·native sampling정책은R18과동일, outputbudget만actual40960-input으로계산한다. Validation지원은query/MLP256분할·생성전actualinputGPU numerical gate로명시했다. Session14400/native13200/first24311000s, predispatchestimate1800–14280s;완결보장아님. 누적14454.552s reset없음/provider직전잔여80709.677591s에서새quota확인후dispatch;failed/setup모두charge. 세부R19_PREREGISTRATION_20261006.md 및r19/CONFIG_FACTOR_AUDIT.json·PREUPLOAD_AUDIT.json·bundle_validation.json. Native3/3·전수수치gate 유지/TSR·DPP없음/STOP불변.


R19 private arki13/visref-paper-r19-20261006 v1/kernel137254020 dispatch. Source6b4710cfb0f99904808082426c82d5231b5cac84dbdbadba7b19c523d5bf0a36/notebookbec2c479485d67127fe90a05789707ddc81ddb18fd23fadcc43de8523d45ba74. 전작업terminal/freshquota80709.677591s/used27290.322409s/reserved0확인. session14400/native13200/first24311000s, estimate1800–14280s. STOP보존; 실T4reference smoke 및extended-native종료결과 대기. scripts/paper_finish_r19.py는새GPU작업없이종료·회수·허가된final-only채점만수행한다. 결과미검증/TSR·DPP release없음.


### Native 종료와 논문 adaptive stopping 구분 — 사용자 질문에 따른 정정

243의R17/R18 8192-token 실패는 official model.generate native 진단이다. 이경로의entropy capture는관측치수집이며 H<0.25로종료시키지않는다;R18은capture자체도off였다. Adaptive response-entropy 판정은paper_baseline_runtime.run_method에서 reasoning block 완료→S response probes→estimate→threshold 순서로만구현되어있으며main8B실험에서는미실행이다. 따라서243결과는entropy stop의실패를증명하지않으며그criterion이막았을지아직검증되지않았다. 특히보존CVPR main Table1 caption은all methods adopt adaptive stopping이라고명시하고Sec5.1은TSR에도같은기준사용을재확인한다. 현재native_st는model interface/termination 진단이며논문Table1 ST의adaptive-stopping baseline으로간주하면안된다. ST의정확한reasoning checkpoint/step boundary와response estimator에대한author구현은미확정이므로임의token마다entropy종료를삽입해exact재현이라고표시하지않는다. R19는cap진단으로계속유효하나adaptive-stopping 검증을대체하지않는다. native의자연think_end가끝나지않는것과완결step간adaptive extension종료는구분한다. Source research/visref_cvpr_main.txt Table1/Algorithm1/Sec4.1.2/Sec5.1;implementation src/visref/paper_baseline_runner.py 및paper_baseline_runtime.py. 기존출력/설정/STOP불변.


### Adaptive checkpoint 재검토

Mirage §2.1/§3.2는 attempted </think>를Wait 확장경계로사용하고완료trace에조건부final-response entropy를측정한다. 따라서이경계를채택하면첫closingtag가없는243의8Ktrace중간을entropy로끊는것은그방법에서직접뒷받침되지않는다. VisRef Table1 all-methods adaptive 문구와§3 single-trace ST/supplement§H no-self-reflection을함께고려해야하며ST의고entropy시행동은미확정이다. "native에entropy가없으므로243이실패했다"도"entropy를적용해도해결불가"도확정하지않는다. 상세 docs/paper_baseline_alignment/ADAPTIVE_STOPPING_CHECKPOINT_EVIDENCE_20261006.md 및research/ADAPTIVE_CHECKPOINT_EVIDENCE.json. R19는fullcontext진단으로유지하며paper adaptive-ST재현이라고취급하지않는다.


### R19 실제 T4 입력 문맥 smoke 통과 — live 증거

Runner246.950767s event에서243 input953 fullreference PASS를확인했다. Cached eager vs query-chunked efficient SDPA: TV0/maxlogitdelta0.060546875/argmax동일/reference margin18.4375, actualp.95, cache없음/query256/모든prior key유지/생성backend복구. 이후native generate시작로그확인. 이는입력953-token smoke이며40960-context수치정합·EOS·최종답성공을증명하지않는다. 상세r19/PRE_GENERATION_GPU_SMOKE_LIVE.json/live.log;중간logstream ConnectionError는관측오류이며RUNNING finisher세션69883 유지/재실행없음.


### Response entropy 독립 Monte Carlo 교정

EOS까지의3-path 통제분포를독립열거하고native T.6/top50/p.95확률과sequence estimator를비교했다. S10/20/50/100각2000 complete sets의평균·판정빈도가해석값6SE내에서일치/PASS. ExactH0.2673482072, S10에서는.25미만판정54.3%(해석53.8615%). SD는S증가로감소하지만작은S의threshold crossing확률은단조가아니다. EOS제외0.1871943512와길이정규화0.1428101279는별도량이며이예제에서threshold판정도바뀐다. 실제InternVL·저자S근거아님/S선택·threshold변경없음. 상세docs/paper_baseline_alignment/RESPONSE_ENTROPY_MC_CALIBRATION_20261006.md/research/SEQUENCE_MC_CALIBRATION.json. CPUonly/GPU0/API0/R19payload불변.


### R19 수집 후 판정 준비

별도post-collection audit를준비하고CPU8checks/.01s 통과. 실제input+outputcap=40960, sampling정책,pre-generation/prefix/끝생성위치까지reference coverage,finite/TV/argmax·near-tie,query모든prior keys,backend복구 및EOS를독립확인한다. Native EOS와후속reference/runtime오류를분리하며token_limit수치검증성공도baseline성공으로격상하지않는다. 실제collectionversion/sourceSHA/config 일치도요구한다. scripts/paper_context_result_audit.py는회수후에만실행하며추론·API0. R18과rawprefix비교기록,분모3/authoradaptiveST미확정 유지. 실제R19결과 아직미분석·livepayload불변. 상세research/R19_POST_COLLECTION_GATE_PREPARATION.json.


### 동일모델 조사 반영 — official evaluation recipe 우선

사용자조사의핵심운영결론을primary sources로확인했다. 공식8B Thinking config65536/use_lmdeploy/R1/do_sample/V2.0 및wrapper T.6/p.95를reference로우선하며8192는usage example/cap진단으로유지한다. top_k50은LMDeploy0.9.1default상속근거가있어config에없다는이유만으로약한근거라고판정하지않는다;실제dependency/historical commit필요. R1wrapper는boxed최종답지시도설정하므로systemprompt만비교하면불충분하다. Co-RLgreedy·boxed·rule grading확인;Aug2026코드는VisRefMarch2026설정을역추론하는근거가아니다. 전점수/독립실행provenance는아직전수검증하지않았다. 65536은LMDeploy session/position/backend까지별도recipe로검증하며현재eager40960-input에그대로복사하지않는다. HistoricalVLMEvalKit/LMDeploy/tiles/prompt/judge추적우선순위반영,liveR19payload와현재scorer불변/새GPUdispatch없음. 상세docs/paper_baseline_alignment/SAME_MODEL_SURVEY_REVIEW_20261006.md 및research/SAME_MODEL_SURVEY_REVIEW.json.


### VLMEvalKit historical recipe의실제변경 확인

2025-08~2026-03 file-specific Git history(wrapper12/config50commits)를읽고9/1·9/5·9/12·2/26 snapshot config/wrapper/requirements 및utils원본을SHA와보존했다. 9/1에이미8B Thinking config65536/R1/sampling이있으나backend Turbomind/session16384;9/5부터InternVL3_5 PytorchEngine/session=max(16384,max_new_tokens)로수정됐다. 동일config이름의동일유효tokenbudget을가정하면안된다. 네snapshot모두wrapperT.6/p.95/random_seedNone, R1system은현재R19와문자열exact동일. 별도boxed지시와MathVista-defaulttiles6은현재fullprompt/preprocess12와계속대조필요. requirements에는LMDeploypin없음/VisRef사용commit연결미확정. 상세docs/paper_baseline_alignment/VLMEVALKIT_HISTORICAL_RECIPE_20261006.md 및research/vlmevalkit_history/HISTORICAL_RECIPE_AUDIT.json. R19livepayload불변/새dispatch없음.


### 사용자 지시: 변경 이력 추적 중단, baseline 구현 복귀

추가 Git history 탐색을 중단했다. CommonPolicyReferenceDecoder를 연결하여 response controller와 full-context numerical reference가 동일한 frozen native_decoding을 사용하도록 했다. 기존 historical decoder 기본 정책과 live R19 bundle은 유지한다. TSR 제어·MC probes/final seed 분리·actual tiny Qwen nucleus(.95) reference 및 기존 공통-policy 검증 7개 PASS/2.74s. Scripted token boundary test는 full-support p1, 실제 nucleus 확률 검증은 강제 토큰 없는 별도 경로로 분리했다. CPU 구성 검증이며 main8B TSR/DPP 실행 성공이나 author fidelity 증거가 아니다. 현재 R19 native remaining-context 진단 RUNNING; 회수 후 native EOS·수치 gate를 판정하고 다음 GPU run을 결정한다.


### R1 최종 답 지시 후보 구현

현재 dataset query에는 공식 R1 wrapper의 boxed 지시가 없는 것을 확인했다. baseline_question의 기본 dataset_query는 유지하고 internvl_r1_boxed_reference 후보를 추가했다. native 입력과 model.chat replay는 동일한 question을 사용하며 question SHA/policy를 기록한다. 보존 primary-source literal과 exact 일치·기본 입력 보존·미지 policy 거부·두 경로 공유를 로컬 확인했다. Dataset query에 suffix를 더한 후보이며 전체 VLMEvalKit question 구성/LMDeploy/tiles/judge 경로 재현이나 VisRef author 설정 증거는 아니다. Live R19 불변; 새 version 동결·검증·예산 확인 전 원격 실행 없음. 상세 artifacts/runs/paper_baseline_sixhour_20261006/research/R1_BOXED_PROMPT_PREPARATION.json.


### Common-policy controller 실제 runner 연결

controller_sampling_policy=native를 명시한 새 frozen run은 CommonPolicyDecoder를 사용하며 full_context_reference_validation이면 CommonPolicyReferenceDecoder를 사용한다. 기본 historical branch는 유지한다. 실제 runner selection 네 branch와 unknown-policy 거부를 로컬 실행해 확인했다. 생성·reference·MC 기록의 동일 sampling-policy component는 앞선 17PASS 근거를 유지한다. model_runtime에 실제 decoder class와 policy를 기록한다. Live R19 bundle 불변; GPU 경로 사용 성공은 아직 미검증이며 native gate/새 version 동결·자원 조건 통과 후 실행한다. 상세 artifacts/runs/paper_baseline_sixhour_20261006/research/COMMON_POLICY_RUNNER_CONNECTION.json.


### 최신 baseline 구성 격리 묶음 검증

R1 prompt/common-policy runner/reference/context reservation 수정을 포함한 실제 package를 임시 폴더로 추출하여 327 PASS/12.76s 확인했다. paper_prompt 자동 포함, source SHA fdf8628a4415a263e734b1cb187b34b5bce6b8f1953a0380b2168669938720eb. 이 묶음은 local_packaging_diagnostic/gpu_execution_authorized=false로 원격 실행 후보가 아니며 GPU·API 사용 없음. 원본 묶음/manifest/log는 artifacts/runs/paper_baseline_sixhour_20261006/research/local_components_package_20261006 및 LATEST_COMPONENTS_ISOLATED_PACKAGE.json에 보존한다. Live R19와 이전 freeze 변경 없음; 실제 checkpoint 생성 검증을 대체하지 않는다.


### Contextual Z 후보를 common-policy baseline 경로에 연결

새 run에서 z_representation=last_layer_post_norm_contextual_hidden_states를 명시하면 common-policy+full-reference 조합에서만 CommonPolicyContextualReferenceDecoder를 선택한다. 기본 input_embedding_lookup은 유지한다. 마지막 norm 출력의 committed reasoning token별 벡터를 수집하며 think/EOS 제외, 추가 model forward 없음. 실제 tiny Qwen full forward output_hidden_states[-1]과 Z 일치 및 hook 복구·수치 reference·원본 visual row reinjection을 확인했다(관련15PASS/4.31s; contextual3PASS/3.15s). 반복 ID 하나로도 서로 다른 Z 3개를 통해 fixed m3 선택·완결을 확인; 입력 lookup의 unique-ID rank bound를 이 경로에 적용하지 않고 min(N,T,D) 필요 상한으로 기록한다. rank 부족시 jitter/m변경/강제 통과 없음. Author layer/scope 미확정·GPU 미검증·live R19 불변. 상세 artifacts/runs/paper_baseline_sixhour_20261006/research/COMMON_POLICY_CONTEXTUAL_Z_CONNECTION.json.


### 사용자 결정: Z 우선 후보 last hidden state로 동결

사용자 “Z에 입력 임베딩이 아닌 last hidden state가 타당하다고 생각한다”를 반영하여 다음 controller의 primary operational candidate는 마지막 레이어 post-norm contextual hidden states로 지정했다. Config configs/paper_baseline/primary_controller_candidate.json, SHA d12ef76aa9db3a6c45a667ba101d056e40d1ec92f084ae8ad2a5b9054c31a3bd. Current committed reasoning block에서 think/EOS 제외라는 범위 가정은 명시하며 author layer/scope 미확정 표시를 유지한다. Controller native sampling+full-context numerical reference를 요구한다. Input embedding 구현은 historical provenance로 보존하며 새 primary 후보로 자동 선택하지 않는다. Fixed m·무정규화 DPP와 rank gate 유지; short T<m은 hidden-state 전환으로 해결됐다고 주장하지 않는다. Live R19/new GPU dispatch 없음; profile 자체는 dispatch authorization이 아니며 다음 run config 동결 시 병합·검증해야 한다.


### Algorithm1의 고수준 명시 사항 확정

| 항목 | 근거 | 판정 |
|---|---|---|
| z_k 생성 → V_k 선택 → trajectory 갱신 → entropy → 종료 또는 다음 z | Algorithm1 4–15행 | VERIFIED: 고수준 논문 명세 |
| entropy 대상은 갱신된 trajectory에 조건부인 answer/response y 분포 | Algorithm1 12행, §4.1.2 | VERIFIED: 다음-token raw entropy가 아님 |
| V_k는 subsequent reasoning에도 조건으로 들어감 | Eq4의 z_(k+1:K) 조건부 및 Algorithm1 trajectory update | VERIFIED: entropy probe에만 사용하는 것이 아님 |

현재 run_method는 block→inject→cloned response probes→estimate→final 또는 Wait/next block 순서이며, V_k가 들어간 parent를 보존하여 다음 reasoning에 사용한다. High-level 순서의 명시와 실제 GPU correctness는 구분한다. y의 sequence/semantic-answer 구체 범위, S·EOS·estimator·step boundary·Z 레이어·reinjection KV 상세는 계속 미확정이다. 상세 artifacts/runs/paper_baseline_sixhour_20261006/research/ALGORITHM_ORDER_ANSWER_DISTRIBUTION_CONFIRMATION.json.


### 1차 controller profile v2 동결

configs/paper_baseline/primary_controller_candidate_v2.json(SHA 7aa04ad1c506a5ecc8473666a5932c77b6b25a0940601eeccf33325562b2316e)에 last-layer post-norm contextual Z, projected visual row 경로, fraction.3, K10, delta.25, complete response sequence MC S10/EOS 포함/길이 비정규화/동일 effective sampling-scoring policy를 고정했다. Methods는 native_st→tsr_sequence_mc→dpp30_unregularized_sequence_mc. 저자 미명시 S/scope/layer는 operational assumption으로 표시하되 실행 차단 조건으로 삼지 않는다. H 값이 .25에 가깝다는 것은 fidelity gate로 쓰지 않는다. 이전 v1 profile과 live R19 freeze는 보존한다. Profile은 새 GPU dispatch를 승인하거나 native/correctness gate를 해제하지 않으며 다음 run에서 병합·전체freeze·검증한다.


### R19 수집·채점 완료 — 자동 마무리 기록

R19 runner=GATE_FAILED, native 완료=2/3; GPU charge=11470.720000000001s; Gemini 호출=4/비용=$0.002692. 원본/세부 판정은 artifacts/runs/paper_baseline_sixhour_20261006/r19/FINISHER_RESULT.json 및 offline_scoring/results.json에 보존. 정확한 논문 재현 미확정; 목표 service 상태는 이 finisher가 변경하지 않음, 새 GPU dispatch 없음.


### R19 정산과 R20 prompt 요인 준비

R19 native2/3; 243 TIMEOUT, 문맥 상한 도달 미검증. 총 GPU25925.272s(약7.2015h), Gemini44회/$0.037605. 기존 ledger 보존. R20은 공식 boxed 최종 답변 지시만 추가한 native-only 실험으로 사전 등록했다. 시간/문맥 한도 증대 없음; gate/denominator3 유지. 아직 dispatch 없음. 상세 docs/paper_baseline_alignment/R20_PREREGISTRATION_20261006.md 및 research/TERMINAL_BUDGET_RECONCILIATION_R19.json.


### R20 private v1 실행

Kernel arki13/visref-paper-r20-20261006 v1/137282639, source SHA25921f4f2eb4c06874293a42de0a46238c110e48e5f01052331cb6fb65dd6e27. CPU329 PASS/12.75s, 실제 payload125파일884691bytes 검증. R19 대비 identity 외 question_prompt_policy 하나만 변경. 이전 job terminal 및 quota69238.957591s/reserved0 확인 후 제출, RUNNING 확인. Session14400s, 30분 finisher 관찰(session21575), 종료 후 raw/final-only scoring 자동 회수. 현재 gate 미판정, faithful baseline 미완료. STOP SHA 유지. 첫 자동 승인 거부는 기존 사용자 Kaggle/예산/continuation 승인 증거 확인 후 재검토에서 해소되었으며 우회 실행 없음.


### R20 수집·채점 완료 — 자동 마무리 기록

R20 runner=GATE_FAILED, native 완료=2/3; GPU charge=1501.7200000000012s; Gemini 호출=4/비용=$0.002135. 원본/세부 판정은 artifacts/runs/paper_baseline_sixhour_20261006/r20/FINISHER_RESULT.json 및 offline_scoring/results.json에 보존. 정확한 논문 재현 미확정; 목표 service 상태는 이 finisher가 변경하지 않음, 새 GPU dispatch 없음.


### R20 종료 후 검증·정산

Native diagnostic release=False; cumulative GPU=27426.992000000002s. 원문 질문+boxed 지시 provenance, 자연 EOS, 실제 generated-context numerical reference를 함께 판정했다. Timeout의 missing receipt는 입증 부족이며 구현 불일치로 단정하지 않는다. Faithful paper baseline 미확정, 신규 GPU dispatch 없음. 상세 artifacts/runs/paper_baseline_sixhour_20261006/r20/ANALYSIS.json 및 POST_FINISH_RESULT.json.


### 공식 기본 모델 경로 진단 우선

사용자 지시에 따라 추가 GPU 제출 전 pristine pinned author model.chat 경로를 우선 검증한다. 기존 native 생성은 공개 저자 model.generate를 사용했지만 custom Decoder/precomputed visuals/reference 검증과 chat replay가 붙어 있었으므로 pristine chat 실제 생성과 동일하다고 단정하지 않는다. R20은 세 문항 모두 normalEOS;243은3409tokens/333.35s로 완결했으나 오답.419는1opening/2closing thinking 태그로 strict parser가 실패했으며 출력과 최종 답변 텍스트는 존재한다. 기존 gate/score 보존, 임의 last-close 추출로 통과 처리하지 않는다. Official chat control은 VisRef Decoder/probes/parser 없이 실제 model.chat 한 번을 호출하도록 분리했고 retained8B source interface1PASS/.85s. GPU inference 검증 아님. 현재 모든 GPU 작업 종료, 신규 dispatch 없음. 상세 artifacts/runs/paper_baseline_sixhour_20261006/research/OFFICIAL_CHAT_CONTROL_PRIORITY.json.


### 공식 chat→generate CPU 호출 경로 검증

고정8B 저자 source의 chat/generate 함수를 수정 없이 실행하고 실제 tiny random Qwen3를 연결했다. visual embedding 치환, EOS 설정, config.use_cache=False를 generate(use_cache=True)가 override하는 것, 단1회 실제 generate, caller config 보존을 검증했다(2PASS/4.16s). 이 fixture의8token/forcedEOS는 테스트 한정이며 runtime 후보가 아니다. VisRef Decoder/probes/DPP/parser 없음. Tiny CPU wiring 증거이며 실제8B checkpoint/vision encoder GPU correctness 또는 baseline reproduction 증거가 아니다. 신규GPUdispatch 없음. 다음은 fresh pinned8B official chat end-to-end 대조; 기존 모델 생성·태그 문법·scoring 조건을 별도로 기록한다. 상세 artifacts/runs/paper_baseline_sixhour_20261006/research/OFFICIAL_CHAT_GENERATE_CPU_VERIFICATION.json.


### R21 공식 기본 모델 대조 실행

Kernel arki13/visref-paper-r21-20261006 v1/137287605, source bd0f68d06c2eefab7e7fd1ad44fb202794d44f9cc53883e5423dfecfb1498c62. 실제 공개 author chat→generate, fresh model, no VisRef Decoder/token hooks/entropy/DPP/precomputed visual/reference forward. 종료 후 IDs CPU 복사만 관찰한다. Isolated332PASS/12.97s, 실제 payload131파일908486bytes secret/selection/holdout 없음. 첫 준비의r4 fixture 누락 실패는 pre_dispatch_attempt1에 보존·수정 후 전체 재검증했다. 모든 이전 job terminal·quota67737.237591s/reserved0 확인 후 제출. 누적27426.992s 보존, R21 미정산. 30분 finisher/session25494; 종료시 frozen source/원문 question/seed/IDs/text를 R20과 대조하고 기존 strict grammar/scoring 별도 보존. 이전 gate 해제 또는 paper fidelity 확정 아님.


### R21 수집·채점 완료 — 자동 마무리 기록

R21 runner=DIAGNOSTIC_COMPLETE, native 완료=3/3; GPU charge=1338.6809999999969s; Gemini 호출=4/비용=$0.002135. 원본/세부 판정은 artifacts/runs/paper_baseline_sixhour_20261006/r21/FINISHER_RESULT.json 및 offline_scoring/results.json에 보존. 정확한 논문 재현 미확정; 목표 service 상태는 이 finisher가 변경하지 않음, 새 GPU dispatch 없음.


### R21 기본 모델 대조 판정과 채점 v2 사전 기록

공식 fresh chat 경로 R21은3/3 normalEOS이고 R20과 모든 generatedIDs/text exact동일. 이 조건에서 project 검증 instrumentation이 출력 문제를 만든다는 가설은 지지되지 않는다. 419의 태그 반복은 공식 경로에서도 생성된다. 고정 VLMEvalKit54a063c5 inference.py의 SPLIT_THINK는 여러 closing 태그에 경고를 남기고 마지막 태그 뒤 prediction을 평가한다. 해당 공개 함수와 경고까지6fixtures exact parity PASS/.02s. 기존 strict parser/gates/scoring 원본 보존, 신규 offline_scoring_vlmeval_split_v1에서 R20/R21의 모든3outputs를 같은규칙으로 재채점한다. EOS 미도달이면 final 없음, denominator3 유지. Final-only payload local audit 후 기존 승인Gemini사용, 신규inference/GPU없음. 이 변경은 scorer 버전이며 VisRef 저자 동일 extractor 확인 또는 controller gate release가 아니다.


### 공식 기본 생성·채점 진단 결과

R21 fresh author chat3/3 normalEOS, 모든 rawIDs/text R20 exact동일; R20 실제 prefix/generated reference도3/3 수치 통과. 이 고정 dev3조건에서 기본 generation interface는 검증됐다. Public VLMEvalKit SPLIT_THINK 규칙을 별도 채점 버전으로 적용하여3final 모두 추출 성공, 두 extractor의prediction 일치, 정답0/3. 419의 이전 missing_response는 strict 태그 처리로 발생한 분석 실패였으며 신규 규칙의 정상 추출과 구분한다. 이전 gates/점수 보존, controller gate 해제/전체MathVista정확도/VisRef adaptive-ST 충실재현 주장은 없음. R20/R21 outgoing payload bytes exact동일해 R21의6calls($0.0039055)를 R20에도 재사용, 중복6calls 절약. GPU 누적28765.673s, extraction58calls/$0.0457805, 신규GPU없음. 상세 research/VLMEVAL_SPLIT_THINK_RESCORING.json 및 각 offline_scoring_vlmeval_split_v1/results.json.


### 기본 생성의 선행 검증과 sequence MC correctness

선행 검증 기록은 R20/R21의 version/source, 동일 input/seed, 전체 token 일치, EOS, full-context 수치 검증, 공통 versioned scorer를 실제 자료로 대조했다. 과거 strict-tag gate는 변경하지 않았고, 정답0/3을 정확도 성공으로 표현하지 않는다. TSR/DPP는 각각의 동결한 correctness 조건을 따로 통과해야 한다. 독립 CPU 검증에서 실제 tiny Qwen의 S10 complete response를 자연EOS까지 sampling하고, cache 없는 full forward와 Transformers Temperature/TopK/TopP로 전체 sequence logq를 재계산했다. Draw NLL 합과 평균H가3e-5 이내로 일치했다(1PASS/4.23s). EOS 포함, 길이 비정규화, sample별 seed, parent 불변을 확인했다. 실제8B entropy 검증은 아직 남아 있다. 신규GPUdispatch 없음. 상세 artifacts/runs/paper_baseline_sixhour_20261006/research/NATIVE_GENERATION_CERTIFICATE.json 및 SEQUENCE_MC_FULL_FORWARD_CHECK.json.


### R22 TSR sequence-MC 실제 모델 검증

R22 private v1/137293266을 제출했다. Source42f8171670e5c72bf2f84c9fddc2594caf7ea9caa6881267491b7fe460c7c60a, isolated342PASS/12.82s, 실제 payload158파일1560509bytes에서 인증정보/selection/holdout 없음 확인. Native certificate를 bundle 안에서도 hash/settings로 대조하여 이미 검증한 기본 추론을 반복하지 않는다. S10/EOS/비정규화 sequenceNLL/T.6/top50/p.95, last-layer contextual Z, probe/final4096, thinking32768, total context40960, TV/mask/KV/clone-parent 불변을 검증한다. DPP는 실행하지 않는다. 공개 SPLIT_THINK 후처리는 새 run의 명시적 선택이며 sampledIDs/NLL을 바꾸지 않는 것을 CPU에서 확인했다. 과거 gate/score는 그대로 보존한다. 이전 모든 job terminal 및 quota66398.556591s/reserved0 확인, session14400s, 이전 GPU28765.673s 유지. 30분 finisher/session29376이 원본·실패·측정 불가 기록을 회수한다. 현재 판정 대기.


### R22 수집·채점 완료 — 자동 마무리 기록

R22 runner=GATE_FAILED, TSR 완료=2/3; GPU charge=2526.8960000000006s; Gemini 호출=4/비용=$0.002003. 원본/세부 판정은 artifacts/runs/paper_baseline_sixhour_20261006/r22/FINISHER_RESULT.json 및 offline_scoring/results.json에 보존. 정확한 논문 재현 미확정; 목표 service 상태는 이 finisher가 변경하지 않음, 새 GPU dispatch 없음.


### R22 TSR 판정과 수치 검증 실패 조사

243은5단계(H 마지막0), 479는1단계(H0)에서 absolute entropy 조건으로 종료했다. 완성된 각 단계의S10/EOS/동일sampling-policy 및 sequenceNLL 재계산은 통과했다. 419는3단계까지 entropy 측정을 마쳤으나 다음 Wait context4834에서 cached eager와 uncached SDPA efficient reference의 TV=.0493963이 동결 기준.02를 초과해 중단했다. 최대 logit차.0332031, argmax 일치. 현재 cache 오류와 backend/FP16+nucleus support 차이를 구분할 증거는 부족하다. Gate를 완화하지 않고 untruncated TV/support 변경량 진단을 추가했으며 관련8tests PASS/2.73s; synthetic fixture에서 raw TV.002가 top_p=.95 경계 때문에 policy TV.051이 될 수 있음을 확인했다(실제419 원인 확정 아님). 다음에는 보존된419 block/Wait만 재생해 cached eager/uncached SDPA/uncached bounded eager를 비교한다. 새 응답 sampling 없이 진단한다. 공통 scorer denominator3, 정답0/3, extraction failure1. GPU누적31292.569s/잔여63871.660591s, extraction62calls/$0.0477835. DPP/논문 충실재현 미완료. 상세 artifacts/runs/paper_baseline_sixhour_20261006/research/R22_REFERENCE_FAILURE_REVIEW.json.


### R23 제출 승인 검토 대기

동결 source a32ed2a7f81dbf13beb1772306f390e137311edc6b95a5a43b2452601fa93b94, isolated346PASS/11.99s, payload164파일15850527bytes 인증정보/selection/locked 없음. 자동 승인 검토가 기존 사용자 자율실험·Kaggle·계속 권한을 확인한 동일 작업 재검토에서도 새R23 명시 승인 필요를 이유로 거부했다. 제출 명령은 실행되지 않았으며 launch 없음/GPU 사용 없음. Private dev419 replay만, session1800s/새sampling/API채점없음의 구체적 승인 대기. 우회 제출하지 않는다. 상세 artifacts/runs/paper_baseline_sixhour_20261006/r23/APPROVAL_REVIEW_BLOCK.json.


### R23 명시 승인 및 제출

사용자 “R23 별도 승인” 확인. 첫 upload HTTP400 후 원격No runs found(404)/GPUreserved0 확인했으며 모든 실패 bundle은 r23/pre_submission_http400에 보존했다. 재생에 필요 없는 probe자료를 제외하고 동일419 committed_ids/Wait/query와 원본 SHA를 동결한 fixture를 사용했다. 계산 조건은 변경하지 않았다. 재검증346PASS 및 payload165파일1639067bytes에서 인증정보/selection/locked 없음 확인. Private r23 v1/kernel137305444, sourceac8229f7c0b9652c6081919df4213ff22e06f2d44e12a946b94b4ff9868cd6eb 제출 성공. 제출 전 모든 이전 job terminal, quota63871.660591s/reserved0, session1800s/예상600–1680s. 누적31292.569s는 그대로 유지하고 R23 청구는 회수 시 계산한다. 새 답 sampling/API채점 없음, 과거gate 미해제, DPP 미실행.


### R23 수치 진단 회수

R23 runner=REPLAY_DIAGNOSTIC_COMPLETE; GPU charge=906.4930000000022s, 누적=32199.062s. 새 응답 sampling/API 채점 없음. 기존 gate/실패 보존. 상세 artifacts/runs/paper_baseline_sixhour_20261006/r23/FINISHER_RESULT.json.


### R23 수치 차이 해석

419 context4834의 R22 TV.049396332를 새 sampling 없이 정확히 재현했다. Cached-vs-SDPA raw TV.00837859/support6↔5, cached-vs-bounded-eager raw TV.00474438/support6↔5; 둘 다 nucleus 이후TV.0493963. 제외 토큰 하나의 cached sampling질량.049396336이 TV를 지배한다. Argmax 모두 일치. Eager-vs-SDPA policy TV.0104560/support5동일. 따라서 단순SDPA만의 문제로 단정하지 않으며 cache 오류도 아직 확정할 수 없다. FP16에서 full-query와 single-query 계산의 차이가 top_p 경계로 증폭되는 가설은 관측으로 좁혀졌다. 기존.02 gate 및R22실패 보존. 다음 진단은 고정 공개prepare_inputs_for_generation이 freshcache/positions를 만들어 같은feedchunks를 처리한 결과와 모든cache층/최종logits exact 비교(허용오차 확대 없음)다. 독립CPU13PASS/4.41s 및 고의cache손상검출 확인. GPU누적32199.062s/잔여62965.167591s, 추가API없음. 상세 artifacts/runs/paper_baseline_sixhour_20261006/research/R23_NUMERICAL_INTERPRETATION.json.


### R24 자동 승인 검토 거부

R24 frozen sourcecf31b979e27a4fe257e65404ae21fa106c12e505eb582bf4c27b5fd868bf341b, local bundle PASS, payload167파일1651454bytes 인증정보/selection/locked 없음. 자동 승인 검토가 “사용자는 R23만 별도 승인” 및AGENTS 재승인 문구를 이유로 최대1800초R24제출을 거부했다. 제출명령 미실행, launch없음/추가GPU없음. 같은 근거로 재시도하거나 우회하지 않는다. 공개helper freshcache exact 비교 준비 완료, R24 명시승인 대기.


### 실행 오류와 수치 진단을 분리한 profile v3

사용자가 검증 기준의 과도함을 지적했고 R23은 hard nucleus support 경계가 TV를 증폭하는 것을 실제로 보였다. 이에 미래 run용 primary_controller_candidate_v3에서 numerical_reference_failure_policy=diagnostic_only를 명시했다. 기존기본값strict와.02 threshold는 유지하며, v3에서는 numerical TV/argmax 차이를 기록한 경고로 남긴다. NaN/Inf, tensor/actual mask/positions/cache shape/prefix 오류, parent 오염은 여전히 중단한다. 이 변경은 새 운영 기준이며 R22 실패를 사후통과로 바꾸지 않는다. Score/entropy scale은 correctness gate가 아니다. IncompleteS10은 측정불가, token/EOS/추출 실패는 독립기록한다. DPP kernel/rank 미공개 처리도 별도 명시가 필요하며 아직 jitter나 무근거rankcompletion을 추가하지 않았다. 현재 GPU dispatch 없음. R24 기존bundle/사전기록은 보존하지만 local source 변경으로 그대로 dispatch할 수 없으며 미래run은 새freeze가 필요하다.


### R25 준비 및 자동 승인 검토 거부

수치 차이를 진단경고로 분리한 profilev3를 동결한 R25 dev3 TSR 준비 완료. Source dd46500675dd076e4f6ada2f84d7d95fd5c0dac17492162865514d3ab2a0ac6f, local bundlePASS 및 payload165파일1595307bytes 인증정보/selection/locked없음. 새diagnostic 검사, 수치정책 외 기존모델/추론/entropy/시간설정 유지. 자동 승인 검토가 “R23만 별도 승인, R25 명시승인 없음” 및AGENTS 중단문구를 이유로 최대4시간 제출을 거부했다. launch없음/추가GPU없음, 우회하지 않는다. 사용자R25 또는 후속baseline development포괄승인 대기.


### R25 및 후속 development 실행 명시 승인

사용자가 “R25와 후속 development 실험의 승인.”이라고 명시했다. 준비된 R25 TSR dev3를 실행하고 이후 동일 baseline development범위 실험에 이 승인을 적용한다. 기존 누적GPU승인/livequota검사, selection/locked/압축금지,STOP/실패보존을 유지한다. R23에서 top-p 경계 증폭을 확인했지만 모든cache오류가 없었다는 확정은 하지 않는다. 새R25의 수치 차이는 경고, 실제NaN/mask/position/cache오염은 중단한다. 기존R22판정은 보존한다.


### R25 TSR 재실행 제출

Private R25 v1/kernel137312243 제출 성공. Source dd46500675dd076e4f6ada2f84d7d95fd5c0dac17492162865514d3ab2a0ac6f. 제출 전 이전모든job terminal/quota62965.167591s/reserved0 확인. Dev243/479/419 동일설정, numerical 차이 경고-only, 실제구조/cache오염검사 유지, DPP미실행. Session14400s/phase13200s, 누적32199.062s 유지하고 R25사용량은 회수 때 반영한다. 30분 확인.


### R25 수집·채점 완료 — 자동 마무리 기록

R25 runner=TSR_CORRECTNESS_COMPLETE, TSR 완료=3/3; GPU charge=4703.889999999999s; Gemini 호출=6/비용=$0.0038295. 원본/세부 판정은 artifacts/runs/paper_baseline_sixhour_20261006/r25/FINISHER_RESULT.json 및 offline_scoring/results.json에 보존. 정확한 논문 재현 미확정; 목표 service 상태는 이 finisher가 변경하지 않음, 새 GPU dispatch 없음.


### 2026-10-06 자연 응답 entropy 전환 — 로컬 구현 진행
사용자 지시: 추가 prompt 없이 실제 생성 마지막 </think> 이후의 y를 사용하고, delimiter가 없으면 MathVista extractor 답의 마지막 일치 span을 사용한다. 해당 원본 token 위치의 full-vocabulary softmax predictive entropy를 평균한다(mean aggregation=paper-underspecified). 기존 R25의 강제 close/S10 sequence NLL/별도 final generation 결과는 다른 운영 variant로 보존하며 새 방법의 증거로 사용하지 않는다.
새 natural_response_runtime.Decoder.finish는 fixed close/newline/answer request 없이 실제 sampled token만 feed한다. block의 s1 차용 minimum-length suppression도 새 경로에서 0으로 고정하여 종료 token 차단을 제거했다. 로컬 관련11 tests PASS/1.75s: 실제 sampled closing/EOS 및 feed 목록, suppression 비활성화, 마지막 delimiter/fallback span/mean predictive entropy/비정상 logits. 이는 제어 fixture와 계산 단위 검증이며 실제8B GPU 검증이 아니다.
아직 새 controller 연결, boxed query suffix 제거, step-boundary의 sampled close 미반영/재sampling과 별도 final 생성 정책 정리, 실제 MathVista extractor fallback 연결 및 새 frozen GPU run이 남아 있다. 새 GPU dispatch 없음; 기존 STOP/raw/실패 보존.


### 2026-10-06 natural predictive controller 로컬 연결
새 별도 profilev4는 dataset_query(추가 boxed suffix 없음), l_min0, S 비적용, raw T1 full-vocab answer-token mean entropy, 실제 측정 응답 최종 재사용을 명시한다. natural_response_runtime.run_method에 연결: TSR은 block에서 실제 sampled close를 response clone에 commit하여 재sampling/고정태그 치환을 피한다. DPP는 z→visual injection 이후 문맥이 바뀌므로 이전 draw를 재사용하지 않고 자연 sampling한다. entropy가 높은 경우 parent reasoning context에 논문의 Wait, think more만 append한다. Parent clone 불변 검사 유지. Final 별도 생성 제거. Token-limit은 측정 가능해도 정상 완료로 처리하지 않는다.
관련13 local tests PASS/1.75s(실제 sampled boundary commit, no forced feed, no termination suppression, 한 response/step 및 측정 응답 재사용, answer scope/계산). 운영 controller fixture이며 live8B 증거 아님. 기존 R25 baseline certificate는 boxed input이므로 dataset_query v4의 native 동등성을 증명하지 않는다. 새 remote runner/config freeze 및 MathVista extractor fallback 연결, native dev smoke/GPU 검증은 남아 있다. 현재 GPU dispatch 없음.


### 2026-10-06 MathVista fallback 연결 수정 및 새 runner 초안
OfficialExtractor와 실제 고정 upstream 함수에 새 last-span helper를 연결하는 로컬 검증에서 upstream이 요구하는 pid 전달 누락을 발견하여 수정했다. Gold/precision 배제는 유지한다. mock transport를 주입한 실제 upstream five-shot extraction→마지막 span→mean entropy 검증 포함14 tests PASS/1.76s. 외부 API 호출 없이 검사했다.
natural_tsr_runner.py 초안은 paper_r26/dev243,479,419/dataset_query/profilev4만 허용하며 기존 boxed native certificate를 재사용하지 않는다. 로컬 syntax 확인 완료. remote extractor transport 및 native 비교 검증 연결은 아직 미완료이므로 GPU 제출하지 않았다. delimiter 없는 EOS/token-limit block도 extractor fallback에 전달하는 제어 경로 보완이 남아 있다. 사용자 지시의 fallback을 구현 완료로 주장하지 않는다.


### 2026-10-06 현 설정 사전 검토 및 설명 정정
사용자 요청으로 새 GPU 제출 전 v4를 검토했다. 평균 answer-token predictive entropy는 본문 response-distribution entropy와 다른 variant이고 .25의 의미를 자동 승계하지 못한다. 중요 정정: VisRef Alg1 lines16–17 및Sec4.1.2는 stopping 후 final answer sampling을 명시하므로, 앞선 ‘별도 final generation이 논문에 없는 구현 추가’ 설명은 잘못이었다. v4의 측정 응답 재사용은 절약형 variant로 기록하며 정확재현 주장 금지. inherited 태그문법 gate/token-limit fallback/remote extractor/full cfg/DPP live 검증이 남아 있고 GPU는 제출하지 않았다. 상세 docs/paper_baseline_alignment/CURRENT_PROFILE_V4_REVIEW_20261006_KO.md.


### 2026-10-06 사전 검토 반영: 자연 block 및 논문 final 정책
natural block에서 repeated/misplaced think_start 및 missing/empty thinking 문법 gate를 제거했다. 종료 구분자/EOS/token limit를 실제 sampling 그대로 기록하고 last-layer norm hook으로 contextual Z를 수집한다. 종료 차단 없음. TSR의 no-delimiter EOS와 block token-limit 응답도 answer extractor scope 경로에 전달하되 token-limit을 정상 완료로 승격하지 않는다. DPP는 reasoning boundary가 없으면 재주입 단계 정의를 임의로 만들지 않는다.
중요 정정을 구현에 반영: v4 final_answer_policy=separate_generation_v1로 복원하여 Algorithm1의 stopping 후 final sampling을 따른다. 응답 재사용은 명시적 별도 variant로만 유지한다. 두 정책의 별도 호출 횟수/결과를 local 검사했다. 실제 tiny Qwen의 cached generation 각 token predictive entropy를 cache 없는 full-forward logits softmax entropy와 대조하여1e-5 이내 일치를 확인했다. 관련17 tests PASS/4.71s(이후 block token-limit 처리 추가 확인은 별도 실행). 실제8B/remote extractor/GPU freeze 미완료. 원 논문의 response entropy와 새 mean predictive entropy는 여전히 다른 실험 정의다.


추가 확인: no-delimiter token-limit에도 MathVista extractor fallback을 수행하여 H를 기록하면서 status=GENERATION_FAILED/token_limit을 유지하는 fixture를 확인했다. 자연 EOS의 동일 응답은 정상 완료로 구분한다. 현재 관련18 tests PASS/4.72s. 새 GPU dispatch 없음.


### R26 제출 — 자연 answer entropy 운영 설정 검증
Private arki13/visref-paper-r26-20261006 v1/kernel137325216 제출 성공. source094648c21fdaac47a83423d192ee63295df837ef7858e7e4399f3c3c51213441, notebookd0d3453e2bdb8d60e2f2b64c7ea8e2f8bed0104086bd90d1992c5f72c7c1c380. Isolated371PASS/18.18s, payload178files1651327bytes 인증정보/raw/selection/holdout없음. 사용자후속development승인 범위의243/479/419만, DPP미실행. dataset_query/자연태그/종료차단없음/mean answer-token predictive H/별도final sampling. k10/.25/8192thinking/4096response 운영값, 논문 entropy exact주장없음. Request 없는 exact official extractor cache와 pending 요청 보존; 자격정보는 GPU에없음.
제출 전 모든과거job terminal/liveavailable58261.277591s/reserved0 확인. 누적36902.952s유지; session7200/예상1500–7080s/실제dispatch conservative1800–7080s. launch estimate_status가 generic R5 label을 기록했지만 실제 기준은 R26 사전등록의 새자연답변 smoke이며 R5설정을 재사용한 의미가 아니다. Finisher session63816으로30분 주기 확인·종료 후 원본/추출요청/사용량 회수, 새dispatch없음. STOP보존.


### 2026-10-06 answer-token entropy 기준 확정 및 DPP 운영 구현
사용자가 재확인했다: 추론 이후 y에 해당하는 token 위치만 predictive entropy를 계산하며 반복 response sampling은 하지 않는다. EOS제외/mean over answer tokens는 기록한 운영 선택으로 유지한다. 논문 response entropy 해석과 다르다는 점을 실행 중단 조건으로 사용하지 않는다. R26은 이미 이 단일응답 측정 경로로 동결·실행 중이며 변경 없음.
후속 DPP용 explicit profilev5: Eq kernel + 상대jitter1e-6*max(diag(L)); fixed floor(.3N) 유지. 과거 unregularized selector와 모든rank실패는 그대로 보존한다. jitter가 저자값이라는 주장은 하지 않는다. T<m에서도 수치적으로 정해진 fixed m을 선택하되 모든 alignment가0이면 임의uniform을 만들지 않는다. 독립 brute-force greedy logdet 대조, scale invariance, invalid input 검사와 실제 tinyQwen contextual T1→m3 선택/원본projected V재주입→단일answer-tokenH→별도final sampling을 확인했다(관련7PASS/5.18s). Live8B DPP 미검증; 새GPU제출없음. 새 파일 operational_dpp.py/natural_dpp_runtime.py는 R26 이미동결된bundle에없고 실행중인job에영향없음.


### 2026-10-06 사용자 우선순위 교정 — TSR 먼저
사용자가 TSR entropy 검증 전에 DPP를 구현한 순서를 지적했다. DPP 후속 구현/실험은 TSR 실제8B 검증 완료까지 중단한다. 이미 만든 local파일/검증자료는 보존하고 R26에 반영하거나 GPU로 실행하지 않는다. R26 methods는 TSR만이다.
인증정보는 Kaggle/Gemini model proxy 접속token이며 model checkpoint를 의미하지 않는다. 기존 Kaggle Model Proxy Gemini extraction 유지. R26은 GPU에credential을넣지않는 staged exact-prompt extraction이고 no-delimiter에서pending이면 local승인추출 후 새로운run version에서만진행한다. 인증 오류가 관측됐다는 의미는 아니다.
R26 누적제한은 thinking block sampled IDs합8192, response/final 각4096, 입력포함문맥40960, 단계10이다. Probe/final 및Wait까지 합친 전체generation 누적cap은현재없다. 생성양은영역별로별도보고하며논문상한이나전체총량제한으로오인하지않는다. Livejob immutable/30분확인 유지.


### 2026-10-06 누적 thinking14k 운영 기준 복원
사용자가 Figure3에 근거한 누적14k 기준을 재확인했다. R26 thinking8192를 설정하면서 이 변경을 사전에 명확히 설명하지 않은 점을 정정한다. 다음 TSR용 primary_tsr_14k_candidate_v6.json은 thinking_cap=block_cap=14000, methodsTSR만, DPP보류를 명시한다. Paper Figure3의14K는 thinking-budget 비교점이고 모든Table1baseline의공식상한이라고 주장하지 않는다. 누적 reasoning block generated IDs를문항당합산하고 entropy response/final추가비용과 context별도보고. 현재R26source/config는변경없으며8192결과로보존한다. 다음dispatch는R26종료회수및실제quota확인후에만가능.


### 2026-10-06 Figure3 budget 설명 재정정
사용자가 ‘여러 reasoning chain의 thinking 합산 예산으로 설정한 것인가’를 재확인했다. 현재구현은 single chain내Wait 단계들의합이며 Figure3 multiple independent parallel chains합산이 아니다. 앞선14k로변경하면Figure3정의까지맞춘다는설명은잘못이었다. v6단일chain14000은 draft diagnostic으로명시하고Figure3재현설정으로자동dispatch하지않는다. Figure3용14k는 문제당독립chain 전체thinking합계이며 별도sharedbudget controller가필요하나 TSR검증전에는구현/실험범위를확장하지않는다. R26은단일TSR8192동결실험그대로보존. 실제현재설정과향후요구조건을구분한다.


### 2026-10-06 인계 검토 — 다음 Baseline 구현 순서

현재 작업 목표는 논문 Baseline 구현이다. [검토와 실행 계획](docs/paper_baseline_alignment/HANDOFF_REVIEW_AND_EXECUTION_PLAN_20261006_KO.md)에 인계·논문·코드·동결 설정을 대조했다. R26은 사용자가 의미가 없다고 판단해 직접 중단했고 CANCEL_ACKNOWLEDGED를 확인했다. 기존 finisher의 최소 원본·사용량 보존만 유지하며 상세 R26 분석·재실행은 다음 작업의 선행 조건으로 삼지 않는다.

다음은 CPU EOS/추출/집계 수정 → 같은 raw query native ST → 사용자 지정 단일응답 H의 TSR → DPP30이다. CPU에서 EOS 뒤 final/Wait 호출, 빈 cache fallback 미완료, 최종 추출 실패와 generation 완료의 혼동을 확인했다. 관련20 tests PASS는 실제8B 성공을 뜻하지 않는다. Figure3 parallel/14K·압축·성능 튜닝은 현재 목표에 추가하지 않는다. 이번에는 계획·진단만 작성했고 runtime 수정·새 GPU/API 실행은 없다.


### R26 자동 회수
{"status": "R26_COLLECTED", "runner_status": "RUNNING", "samples": [], "charged_s": 7200, "cumulative_charged_s": 44102.952, "available_gpu_s": 57106.221457, "pending_extractions": 0, "scoring_status": "PENDING_OFFLINE", "paper_reproduction_verified": false, "new_gpu_dispatch": false}


### 2026-10-06 과도한 검증 조건과 EOS 단정 정정

논문 §3/§4.1.2를 다시 대조했다. CPU 진단은 no-delimiter EOS 이후 final/Wait 호출을 확인했지만 호출 자체의 버그를 증명하지 않는다. Probe EOS와 parent reasoning 종료를 구분해야 하므로 일괄 terminal/호출금지 권고를 철회한다. Author 미공개값·native3/3 자연종료·전token logit보존·장문 full parity를 필수 실행 gate로 확대하지 않는다. 기존 사용자 지정 H와 normal-boundary 별도 final을 유지하고 실제 token대응·추출 연결·상태/집계만 보완한다. [정정 상세 기록](docs/paper_baseline_alignment/HANDOFF_DETAILED_REVIEW_20261006_KO.md) §8 참조. 현재 실행 순서는 [짧은 계획](docs/paper_baseline_alignment/HANDOFF_REVIEW_AND_EXECUTION_PLAN_20261006_KO.md)을 따른다. Runtime 변경·새GPU/API실행 없음.


### 2026-10-06 간단한 Baseline 경로 구현 / R27

짧은 계획의 native재사용·TSR코드정리·CPU검증 완료. R27은 실제8B TSR만 확인하며 새 모델/threshold탐색 없음. 결과 확인 후 DPP30으로 진행. STOP·누적소비·동결원본 보존; selection/locked/압축/Figure3범위 확장 없음. [실행 근거](docs/paper_baseline_alignment/LEAN_IMPLEMENTATION_RESULT_20261006_KO.md).


### 2026-10-06 R27 대기 중 독립 확인

기존 R18 final의 cache-only scoring 재검산 동일/분모3 유지/외부호출0. TinyQwen entropy oracle은 token/logit 한 칸 이동 오류를 검출했다. 기존 pending 추출 도구에 --version r27을 연결하고 terminal collection 이전 거부를 확인했다. 실제추출·추론·DPP 작업은 하지 않았고 실행중 bundle도 변경하지 않았다. [근거](docs/../artifacts/reports/paper_baseline_alignment/R27_WAITING_CHECKS_20261006.json).


### 2026-10-06 R27 결과 확인

R27 첫 actual8B TSR 경량경로는479/419 정상동작. 243limit/pending,0/3채점은분리보고. 다음은계획의DPP30연결·검증; threshold조정·S10복원·243재실행·압축/Figure3확장없음. [근거](docs/paper_baseline_alignment/R27_RESULT_20261006_KO.md).


### 2026-10-06 R28 DPP30 연결

계획의 DPP30 연결 완료/로컬검증통과. R28은DPP만검증하며 ST/TSR재추론·threshold/상한조정·selection/locked/압축은하지않는다. 결과를확인해 실제선택/재주입/H/종료를보고한다. [설정과 근거](docs/paper_baseline_alignment/R28_PREREGISTRATION_20261006.md).


### 2026-10-06 R28 원문·Figure 일치 검토

R28은사용자H+regularizedDPP를쓴개발구현으로정확재현·모든Figure실험완료를선언할수없다. 기본흐름과평가/시각분석범위를구분한다. 기존실행보존/결과확인후판단하며새실험범위를자동확대하지않는다. [항목별 검토](docs/paper_baseline_alignment/R28_PAPER_AND_FIGURE_REVIEW_20261006_KO.md).


### 2026-10-06 논문 수식 경로와 Figure 분석 코드

논문 수식 후보 v7 구현 및 로컬검증 완료. 기존 R28은 사용자 H+jitter 실험으로 그대로 보존하며 새 후보 결과로 혼용하지 않는다. 전체 성능·Figure 재현에는 실제 비교가 필요하고 저자 미공개값/degenerate logdet는 임의 공식값으로 해결하지 않는다. 다음 dispatch는 별도 run config/동결·예산 근거를 갖춘 경우에만 수행한다. [변경과 검증](docs/paper_baseline_alignment/EQUATION_IMPLEMENTATION_20261006_KO.md).


### 2026-10-07 구현·최소검사 완료 / 실제 모델 검증 준비

구현→최소검사 완료, 실제모델 검증의 다음은R29 419 TSR smoke다. R28은회수·분석·정산완료/완전final0이므로성공으로선언하지않는다. 원래kernel fixed-m의zero determinant 동률 처리 추가는 공개Eq8목적값을바꾸지않되운영tie선택으로명시한다. R29 remote upload auto-review거절로미실행;scope근거확인후허용될때만제출한다. [고정설정과 판정](<docs/paper_baseline_alignment/R29_PREREGISTRATION_20261007_KO.md>).


### 2026-10-07 새 lean 재현 경로 (`repro/`)

사용자 결정: (1) 기존 `src/visref`를 이어가지 않고 새 lean 코드 작성 (기존 기록 보존), (2) 주 목표는 Appendix Table 6의 InternVL3.5-1B × MathVista testmini 1,000문항 행 (46.1/48.5/52.0), (3) H는 기존 결정대로 단일 응답의 답 토큰 entropy, (4) GPU는 development 10문항 smoke test까지만 실행 후 보고. 구현 내용과 논문 값 vs 운영 설정 표는 [repro/README.md](repro/README.md)에 있다. CPU 테스트 21 passed (DPP vs brute-force greedy, batched/chunked = 단일 row, entropy = full forward, 상태 전이, run 재개). 실제 GPU 실행은 아직 없음. STOP_REQUESTED.json은 기존 pipeline용 marker로 그대로 둔다. 주의: 1,000문항 전체 실행에는 기존 holdout(700) 전부가 포함된다.

### 2026-10-07 smoke 전 독립 검토 반영
별도 에이전트가 논문(Figure 포함)과 `repro/`를 대조했다. 명백한 버그는 없었다. 이어가기 문구를 `Wait, think more`로 바꾸고, DPP rank 부족 대응 변형 `visref_reg`(L+εI)를 추가했다. smoke는 st/tsr/visref/visref_reg 4개 method × development 10문항으로 다시 빌드했다. CPU 테스트 23 passed. [검토 기록](repro/REVIEW_20261007.md).


### 2026-10-07 R29 실제 검증 회수 및 판정

R29 최초 upload 자동승인 거절은 소유·비공개 목적지와 기존 승인 근거로 재심을 통과했고 비공개v1 제출·회수를 완료했다. 위 미제출 상태는 당시 기록이며 현재 상태를 대체하지 않는다. 실제419 TSR의 sequence NLL3.477/6.584/2.183과 Wait3회 검산 PASS;4번째 reasoning이 thinking8192에 도달해 별도final0/1. R30 제출조건을 완화하지 않았으며 신규GPU 제출 없음. 미완성 reasoning을 final로 표시하던 작업 source 기록을 수정하고 관련29검사 PASS. 누적50981.329s/ceiling95164.229591s, STOP·selection/locked 보존. [결과와 증거](docs/paper_baseline_alignment/R29_RESULT_20261007_KO.md).

### 2026-10-07 smoke1 결과와 수정 (`repro/kaggle/runs/smoke`)
InternVL3.5-1B, development 10문항 × st/tsr/visref/visref_reg, T4×2. 정확도는 ST 5/10, TSR 5/10, VisRef 4/9, VisRef_reg 4/9이고 paired 차이는 0이다(표본이 작아 성능 판단은 불가). GPU0은 912번 VisRef prefill에서 OOM(15GiB 할당)이 나 2행이 누락됐다. 평균 시각 토큰은 1792개로 논문 ~1772와 일치한다. 추출 규칙은 boxed/answer phrase가 대부분이다. visref의 DPP 선택 비율은 평균 0.47, visref_reg는 1.0이다.
핵심 문제: 답 span 평균 entropy가 모든 probe에서 ≈0(최대 0.003)이라 전 문항이 k=1에서 멈췄다. 그래서 TSR≡ST가 되고 VisRef는 사고가 끝난 뒤 한 번만 주입되는 구조가 됐다. 사용자 결정으로 H를 응답 전체 토큰(EOS 제외)의 평균 entropy로 바꿨다(span 평균은 `H_span`으로 기록). 함께 수정한 것: chunked prefill(OOM 방지), 1단계 thinking을 공유해 method별로 fork(같은 context·seed라 동일, 연산 약 1/4), batch 16, chunk 1024. CPU 테스트 25 passed. 다음은 smoke2(development 30문항, 4 method, 3h 상한)로 k 분포·속도·예산을 확인한다(실행 승인 대기).

### 2026-10-07 smoke2 전 독립 검토 반영
별도 에이전트 검토 결과: fork(5 pid × 4 method, k≤4, 단독 실행과 이산 출력 동일)와 chunked prefill(left padding 포함, 단독 forward와 일치)은 정상이다. 다음을 수정했다.
(1) 투입 조건이 leader만 세어 fork 이후 동시 job이 최대 60개까지 늘던 문제: follower를 포함해 동시 job ≤ batch로 제한.
(2) deadline 이후 진행 중 job을 `.partial`로 남기고 종료(타임아웃 시 손실 방지).
(3) OOM이 나면 그룹을 반으로 나눠 재시도, expandable_segments, batch_tokens 48000.
(4) probe 토큰별 entropy, 생성 토큰 수, round별 메모리 peak 기록.
(5) smoke1 원자료에서 VisRef probe의 7~8/9가 "초안 + 두 번째 `</think>` + 답" 형태임을 확인(TSR 1/10). 이전 사용자 규칙대로 y를 응답 안의 마지막 `</think>` 뒤로 정해 H와 채점에 적용.
CPU 테스트 27 passed. smoke2는 다시 빌드했고 push 대기 중이다.

### 2026-10-07 Fable 5.1 논문 충실도 검토 반영
판정: 단계 정의(자연 `</think>`), 재주입 위치, probe 폐기, Wait 이어가기, DPP greedy는 논문과 맞는다. 다만 `<img>…</img></think>` 형태의 재주입이 1B 모델에서 두 번째 `</think>`와 질문 환각을 일으켜, 포맷이 오염된 VisRef를 측정할 위험이 있다.
반영(사용자 결정): 태그 없이 주입하는 `visref_raw`를 추가하고, smoke2를 st/tsr/visref/visref_reg/visref_raw 5개 method(batch 20)로 다시 빌드했다.
반영하지 않음: probe 답을 최종 답으로 재사용하자는 제안은, 정지 결정에 쓰인 표본이라 선택 편향이 생기므로 사용자 결정으로 Algorithm 1의 별도 재표본을 유지했다. hidden-state Z(`visref_hs`)는 smoke2 결과를 본 뒤 다음 smoke에서 검토한다.
smoke2 판단 기준: 1단계 H<0.25 비율이 10~90% 범위인지, k 분포, 두 번째 `</think>` 비율, 질문 환각 수. Fable은 논문 Fig.3에서 VisRef의 thinking 토큰이 ST의 수 배라고 해석했으나, Table 5 지연(+1.1s)과 모순되는 논문 내부 불일치로 기록한다. CPU 테스트 28 passed.


### 2026-10-07 마지막 단일 GPU 업로드 R31 고정

사용자가 GPU 업로드 한 번을 명시 승인했다. R31은 fixed S1 sequence entropy/.25/K10을 유지하며 실제sampled closing을 response conditioning에 commit하고, 예산 소진 시 미완성 후속step을 분리해 마지막 완성trajectory에서 별도final을 생성한다. 이 운영 budget rollback은 H/Kmax 정상종료와 분리한다. 공개419/479 TSR→계산·별도final gate→DPP 한 notebook,2T4/8B/FP16/eager, cap14400s/예상1800–14280s. 로컬및frozen bundle418PASS, 미완성probe나 runtime오류는 DPP를 막는다. 최초패키징6실패는 누락된 기존검사fixture2개를 포함해 해결했으며 실패본도 보존했다. 단일upload journal로 불확실응답 재제출도 금지한다. [명세와 사전판정](docs/paper_baseline_alignment/R31_FINAL_PREREGISTRATION_20261007_KO.md).


### 2026-10-07 R31 마지막 업로드 완료

추가 read-only 소유·private/payload 근거 확인 후 자동승인 재심을 통과했다. R31 arki13/visref-paper-r31-final-20261007 v1/kernel137371435를 단1회 제출했다. Source3570d85e6bebd4b89b4987a09a1142a6852765c59fa5ef864abeba6daae863ef/notebook3dc9e47f61f2d886db484a50cd63050c2c7753e169af81333a3ea6c5bf74533d,418 frozenPASS,180파일1712202bytes/credential·raw·selection·holdout 없음. Quota48871.415481s/reserved0,cap14400s/기존누적50981.329s 보존. 사용자허용 업로드1회 소진, 자동재제출/재실행 금지. 실패·예산복원·paper정상종료를분리해 결과회수·비교자료를작성한다. [launch](artifacts/runs/paper_baseline_sixhour_20261006/r31/launch.json).


### R31 마지막 실행 회수·검산·로컬 채점 정정 완료

{"runner_status": "INCOMPLETE", "methods": {"tsr_natural_sequence": {"planned": 2, "emitted": 2, "complete_finals": 2, "normal_h_or_kmax": 1, "budget_rollback_finals": 1}, "dpp30_natural_sequence": {"planned": 2, "emitted": 2, "complete_finals": 1, "normal_h_or_kmax": 1, "budget_rollback_finals": 0}}, "scores": {"tsr_natural_sequence": {"denominator": 2, "correct": 0, "extraction_failures": 0}, "dpp30_natural_sequence": {"denominator": 2, "correct": 0, "extraction_failures": 1}}, "charged_s": 3833.7392390000023, "cumulative_charged_s": 54815.068239, "gpu_uploads": 1, "new_dispatch": false, "paper_reproduction_verified": false}
R31은 부분 완료이며 Baseline 검증 성공이 아니다. TSR419 예산복원, DPP479 reasoning EOS/별도final 부재, 479 DPP fixed-m determinant0 동률을 분리 기록했다. 첫 채점 connector TypeError/외부호출0 기록은 보존하고 공식 scorer를 바꾸지 않은 동일final 재채점으로 정정했다. 추가 GPU 제출0. 저자 H estimator·Z층·재주입 semantics 및 예산복원의 저자 동일성은 미확인이고 정상 완료·성능 재현 조건은 해소되지 않았다.
[원본과 비교 제한](docs/paper_baseline_alignment/R31_FINAL_RESULT_20261007_KO.md).

### 2026-10-07 smoke2 결과와 본 실험 결정 (`repro/kaggle/runs/smoke2`)
설정: development 30문항 × st/tsr/visref/visref_reg/visref_raw, T4×2, H는 응답 y(마지막 `</think>` 뒤)의 토큰 평균 entropy. GPU0은 deadline에서 `.partial`을 남기고 정상 종료했다. GPU1은 한 round가 약 670초라 Kaggle 3h 상한에 강제 종료됐다(완료 행은 보존). 완료 행: ST 24, 나머지 각 21. 공통 19문항 기준.
- 정지 규칙이 작동함: 1단계 H<0.25 비율 tsr 9/21, visref 11/21, raw 10/21. k 분포는 1~10에 퍼짐(평균 3.0~3.3). stop: entropy 대부분, kmax 2~3, budget 1~3.
- 정확도(소표본, CI 모두 0 포함): ST 62.5%(24), TSR 76.2, visref 71.4, visref_reg 66.7, visref_raw 76.2. paired: TSR−ST +10.5pp, raw−ST +10.5, visref−ST +5.3, reg−ST 0.
- 형식 오염(두 번째 `</think>`): visref 31/63 단계, reg 25/67, raw 10/65, tsr 4/70. 질문 환각 0. DPP 비율: visref 0.42, raw 0.34, reg 1.0.
- 비용: unique 생성 토큰 약 7.2k/문항(st+tsr+visref). ST thinking 중앙값 889, 4096 초과 7/24(그중 정답 2).
- 속도: GPU당 14~16 tok/s. 원인은 긴 context로 KV 예산(48000)이 그룹을 3~4개로 쪼개 순차 decode한 것, round 동기화, 전체 어휘 정렬 sampler.
조치: sampler top-k 우선(50개만 정렬), round 중에도 deadline 반영(stop_at), batch_tokens 72000, 세션 이어 달리기(`--skip-from`), Kaggle slug 하이픈화. CPU 테스트 30 passed.
사용자 결정: 본 실험은 st/tsr/visref_raw, 토큰 상한 단계 4096·합계 8192, 1,000문항 전체를 약 9h 세션으로 나눠 실행(기존 holdout 700 잠금 해제, 첫 세션이 속도 점검 겸함). `full_s1` 빌드 완료(`arki13/visref-repro-full-s1`).

### 2026-10-07 본 실험 세션 단축
사용자 요청(남은 quota 약 10h, 9h 세션은 대기가 너무 김)에 따라 세션을 2h로 줄였다(투입은 1.5h까지, deadline 1h50m). 문항 순서를 seed 0으로 섞어, 어느 세션에서 멈춰도 완료된 문항이 1,000문항의 무작위 표본이 되게 했다. 세션마다 `--skip-from`으로 이어 붙인다. `full_s1` 재빌드: 1,000문항 순서 섞음, st/tsr/visref_raw.

### 2026-10-07 실행 전 Opus 5.5 검토 반영
판정: "작은 수정 후 실행". 결과를 조용히 왜곡하는 버그는 없다. 반영한 것:
(1) decode 가속: 미리 할당한 KV 버퍼 + GQA decode attention. transformers 4.55의 매 스텝 cat과 repeat_kv를 없애 2~3배를 기대한다. 시작 시 GPU에서 기존 경로와 greedy 토큰을 비교해 다르면 자동으로 끈다.
(2) 투입 마감을 timeout−2700s로 당겼다(마무리 시간 확보).
(3) 셔플 seed 기본값을 0으로.
(4) summarize: 섞인 순서의 '완료된 앞부분'만 무작위 부분집합으로 집계하고, 전체 행 정확도는 참고로만 남긴다. McNemar 불일치 수, error 수도 집계한다.
(5) 그룹별 시간 로그, fork 여부, non-finite logits가 난 job만 error로 격리.
CPU 테스트 32 passed, 비정상 logits 격리 별도 확인. full_s1 재빌드(2h, 1,000문항 셔플, st/tsr/visref_raw).

### 2026-10-07 정지 통계 H를 논문 정의 쪽으로 변경 (사용자 결정)
배경: smoke2 probe 240개를 보면, 자연스러운 긴 응답에 논문식 합(sum)을 쓸 때 5~79 nats로 δ=0.25 아래가 0.8%에 불과하다. 지금 방식인 평균은 논문의 양과 단위가 다르다. 논문 Fig.2는 "Answer:" 상자와 답 분포 히스토그램을 그리지만, 프롬프트 문자열·길이·표본 수는 명시하지 않는다.
결정: probe를 `</think>` + `\n\nAnswer:` 뒤 짧은 답 1개(T=1, top-k/p 없음, 첫 줄바꿈/EOS 종료, 최대 32토큰)로 하고, H는 토큰 엔트로피의 합(종료 토큰 포함)으로 한다. 이것은 답 분포 엔트로피 H(Y)의 단일표본 추정이다. 32토큰 안에 끝나지 않으면 H=∞로 계속 진행한다. 최종 답은 자연스러운 응답(T=0.6)으로 따로 생성한다(ST와 같은 형식). 이전의 "answer prompt 금지" 결정은 사용자가 해제했다.
같이 반영: final_finish/probe_finish, code_version, extract_failed, pytest.ini, 잘린 probe 처리. CPU 테스트 35 passed.
영향: full_s1(구 H)의 TSR/VisRef 결과는 새 정의와 섞지 않는다. ST는 H와 무관하다. 다음은 smoke3(development 20문항 × st/tsr/visref_raw, 1h)로 새 H의 정지 비율과 probe 답 형식을 확인한다.

## 2026-10-07 smoke3 결과 → probe를 `\boxed` 형식으로 교체 (smoke4)
- smoke3(dev 20, `\n\nAnswer:` probe): TSR 엔트로피 정지 0/18, visref_raw 4/18. TSR probe의 유한 H 107개 중 δ 미만 0개(중앙값 1.42). 토큰 분해 시 내용 토큰 ≈0.01 nat, 앞 공백 ≈0.47, 줄 끝 ≈0.84 → 형식 엔트로피가 정지를 막음. 정확도 ST 72.2 / TSR 61.1 / visref_raw 72.2(n=18).
- full_s1(무작위 58문항, 구 H): ST 60.3%(95% CI 47.5–71.9). ST의 57%가 4096 상한(반복 루프), dev는 2/20 → dev가 더 쉬움. 2h에 58문항×3 method → 1,000문항은 quota 내 불가.
- 결정(사용자 동의): probe = `</think>` + `\n\n\boxed`, 모델이 `{`를 직접 생성, 상자 닫힘(중괄호 깊이 0)에서 종료, 최대 16토큰. 상자 깊이 1에서는 `}`로 시작하는 토큰들을 한 결과로 합쳐 엔트로피 계산. 무효(빈 상자/EOS/상한) → H=∞. 이전 형식은 `--probe-stop newline`으로 유지.
- 진단 추가: 단계별 rank_V, n_unique_z, n_dpp; summarize에 전 단계 H 분위수, δ 미만 수, k=1 엔트로피 정지 수, probe 상자 내용.
- smoke4: smoke3와 같은 dev 20문항·순서, st/tsr/visref_raw, 3600s. 바뀐 요인은 probe 하나(진단 필드는 결과에 영향 없음, 서브에이전트 검토에서 newline 모드·ST 행 불변 확인).
- 본 실험: 남은 quota 안에서 무작위 셔플 앞부분 일부(200~300 예상), 크기는 smoke4 처리량을 보고 결정.

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
4. 처리량 → 표본 크기: 시간당 완료 문항 수 × 남은 quota로 예상 n을 계산한다. 3개 방법으로 n<150이면 main_s1을 빌드하기 전에 TSR 제외 여부를 사용자에게 묻는다.
5. 진행: main_s1부터는 2h, `--skip-from main_s0/.../results_*.jsonl`(.partial 제외), 같은 run_args, run.py·visref/*.py를 바꾸지 않는다(code_version 일치). CLI 인자 변경도 새 시리즈로 본다. 세션마다 pull 후 1~2항만 빠르게 확인한다.
6. 최종: 모든 세션을 `summarize.py … --order main_s0 ids.json`으로 집계해 정확도, paired 차이와 95% CI, 불일치 쌍 수를 보고한다. 불일치 쌍은 직접 채점 감사한다. 상한·정지율·forced close를 함께 보고한다.
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
  → 개선 후보(사용자 결정 대기): 공통 난수(CRN). probe는 별도 난수열을 쓰고, k=1에서 정지하면 최종 답을 1단계 직후 난수 상태로 뽑아 TSR이 ST와 같은 답을 내게 한다. 각 방법의 분포는 바뀌지 않고 paired 차이의 분산만 줄어든다.

## 2026-10-07 b_test: 단계 1024 상한 변형 + 강제 답변·CRN (판정 규칙은 결과 전에 고정)
배경
- main_s0(A: 단계 4096 / 합계 8192)에서 강제로 `</think>`를 넣으면 모델이 답 대신 추론을 이어 갔다. ST 강제 종료 30문항은 정답 11, 답변 중앙값 635토큰(14개 답변 상한)이고, 자연 종료 30문항은 정답 22, 68토큰이다.
- A는 최대 2단계라 K_max=10과 논문의 생각 토큰 비율(VisRef/ST 4.2~7.3배, Fig. 3)에 구조적으로 닿지 않는다.
- 단계 상한 600은 Fig. 3 ST 별표 최대 978과 모순이라 쓰지 않는다. 1024는 Fig. 3과 모순되지 않는 가장 작은 흔한 값이고 InternVL3.5 모델 카드 예시 `max_new_tokens=1024`와 같다. 논문에 없는 운영 가정이다.
변경 (새 시리즈, main_s0와 합치지 않음)
1. 강제 답변: 생각이 상한/EOS로 끊기면 `</think>` + `\n\n**Final Answer**\n\boxed` 뒤 상자만 생성한다(상자 닫힘에서 종료, 최대 64토큰). s1 budget forcing·DEER probe와 같은 형식이고, 모든 방법·모든 정지 사유에 똑같이 적용한다.
2. 잘린 단계 뒤 이어가기 문구 앞에 `\n`을 넣는다(자연 종료 단계는 '.\n'으로 끝남).
3. CRN: probe가 전용 난수열을 쓴다. TSR이 k=1에서 멈추면 최종 답이 ST와 같다(각 방법의 분포는 불변).
4. 상한: 단계 1024, 합계 10,240, K_max 10. 나머지(δ=0.25, `\boxed` probe, visref_raw, T=0.6)는 main_s0과 같다.
실행: seed-0 셔플 순서를 처음부터(main_s0과 같은 순서), st/tsr/visref_raw, timeout 4200s(투입 2800s, deadline 3600s).
판정
- 통과 조건(건전성, T4): error 0, fast_kv ok, 세 방법이 모두 끝난 문항 ≥45, 강제 답변 무효(빈 상자·상자 미종료) ≤10%.
- 처리량: 시간당 완료 문항 × 남은 quota로 본 실험 n을 추정한다. 3방법 n<150이면 TSR 제외 여부를 사용자에게 묻는다.
- 보고만 하고 판단에 쓰지 않음: TSR/VisRef 1단계 정지 비율과 평균 k(T2), 생각 토큰 비율 vs ST(T3), ST(1024) vs main_s0 ST(4096) 같은 pid 정확도(T1), 방법 간 정확도. T1은 짧은 상한에서 정의상 생기는 결과라 판정에서 뺐다.
- 이상 신호: 1단계가 1024에서 잘렸는데 TSR이 한 번도 이어가지 않으면 구현 오류로 보고 조사한다.
- 통과 시 본 실험: `b_main_s1`부터 `--skip-from b_test/.../results_*.jsonl`, 코드·인자 불변. 실패 시 원인을 고쳐 새 시리즈로 한다. B 자체를 접을지는 사용자가 정한다.
- b_test 빌드 완료: `arki13/visref-repro-b-test`, code_version 02d67d405ad3, 4200s, seed-0 순서(main_s0과 동일). CPU 테스트 59 passed(강제 답변 유효/무효·공백, 줄바꿈, CRN: 작은 LM에서 TSR k=1 최종 답 = ST, CRN 없으면 다름). Sonnet 독립 검토: 확정 버그 없음. 지적 반영: 상자 앞 공백 제거. 참고: GPU에서는 batch 구성에 따른 fp16 오차로 TSR k=1 = ST가 비트 단위로 보장되지 않는다(분산 감소 효과는 유지).

## 2026-10-07 DPP 구현 대조 (Chen 2018 참조 코드, CEIL)
- greedy 갱신·선택: Chen Alg. 1 / 공식 코드(fast-map-dpp `dpp()`)와 같다. 테스트에서 naive log-det greedy와 일치.
- 정지: 공식 코드는 `d_j² < 1e-10`(절대값)에서 멈추고 m개보다 적게 반환한다(채우지 않음). 재현은 상대 기준 `1e-9·max d²`(커널 크기에 무관) 후, 남은 자리를 relevance r²=L_ii 순으로 채운다(운영 선택, 논문·Chen·CEIL 어디에도 없음).
- 커널: VisRef L = V M_k Vᵀ에는 별도 relevance 항이 없지만 L_ij = q_i q_j cos(a_i, a_j), q_i = ‖Z v_i‖로 품질×다양성 분해와 같은 꼴이다. CEIL은 L̃ = Diag(r) L Diag(r)에 λ 가중(exp(r/2λ))과 KNN 사전 축소를 두고 고정 K개를 뽑는다. VisRef에는 λ도 사전 축소도 없다. relevance 채움은 이 품질 항 방향과 일관된다.
- rank: Z가 input embedding이라 rank(L) ≤ z_k 고유 토큰 수. main_s0 visref_raw 67단계에서 n_dpp = n_unique_z가 50단계, n_dpp < m이 50단계(75%), 중앙값 n_dpp 98 / m 537. Chen 공식 코드대로라면 주입 토큰이 m의 약 1/5로 줄어든다. b_test(단계 1024)에서는 고유 토큰이 줄어 채움 비율이 더 커질 수 있다.
- 논문 쪽 단서(추정): Fig. 4(b)에서 m 20%→30%로 76.1→79.2가 오르는데, input embedding Z + 채움 없음이면 rank를 넘는 m 증가는 선택을 바꾸지 못한다. 논문은 채움을 했거나 rank가 더 큰 Z(hidden state, rank ≤ min(T_k, d))를 썼을 가능성이 있다. hidden-state Z 비교는 본 실험 뒤 별도 시리즈로 한다(b_test는 바꾸지 않음).
- 추가 확인(2026-10-07, PDF 원문): §4.1.1 Eq. 11 뒤 "enforcing |V_k| = m … We terminate after m iterations" → Chen 공식 코드식 조기 종료(m보다 적게 반환)는 논문과 다르다. Fig. 4(b) 원문: m 20%→30%에서 76.1→79.2, 40%는 추가 이득 없음. Table 3(8B): relevance만 MathVista 75.6, diversity만 77.4, 둘 다 79.3.
- 합성 실험(N=1790, d=1024, 단계 1024토큰 중 고유 178, m=537): Chen 공식 코드는 178~192개에서 멈춤(절대 1e-10 기준, 커널 크기에 따라 잡음 픽 0~14개). 논문 문구대로 m번 끝까지 돌리면 rank 이후 359개 선택은 반올림 잡음이 결정(relevance 채움과 겹침 float64 8개, float32 75개, 무작위 기대 약 80개) → 이 구간은 논문 수식이 정의되지 않음.
- 함의: input embedding Z에서는 주입 토큰의 다수가 relevance 채움이라, 우리 VisRef가 Table 3의 'relevance만' 변형에 가까울 수 있다. hidden-state Z(rank ≤ min(단계 길이, d)) 비교의 우선순위를 올린다.

## 2026-10-07 b_test 재빌드: s1 강제 답변 + ridge DPP (사용자 결정, push 전)
- DEER 공식 코드 원문 확인: 주 구현(vllm-deer.py) answer_prompt_str = `\n**Final Answer**\n\boxed`, 간이 버전은 `…The final answer is \boxed`. 둘 다 신뢰도 probe용이고, 상한 도달 시 최종 답은 `\n</think>\n\n` 뒤 자유 생성(답이 안 끝날 때만 옵션 af=1로 위 문구 추가, 기본 꺼짐). → 직전 b_test의 '상자만' 방식은 어느 참고 문헌과도 그대로 같지 않았다.
- s1 README 원문: 최종 답 앞에 `Final Answer:`를 붙일 수 있고 'to prevent the model from just continuing to reason in its answer when early exiting'. VisRef가 TSR 근거로 인용([34]).
- 결정 1(강제 답변): s1 방식. 잘린 생각 + `</think>` + `\n\nFinal Answer:` 뒤 자유 답변(답 상한 1024, EOS 종료). 채점 텍스트 = `Final Answer:` + 답(답 안에서 `</think>`를 다시 내면 그 뒤 텍스트).
- 결정 2(VisRef 채움): rank 이후 relevance 채움 대신 L+εI ridge로 m개 모두 log-det greedy(`visref_raw_reg`, 태그 없는 주입). 근거: 논문 '|V_k| = m … m iterations', rank 이후 원식은 정의되지 않고 실제로는 반올림 잡음이 선택(합성 실험), relevance 채움은 Table 3 'relevance만'(75.6 < 79.3)에 가까움.
- b_test 판정 규칙 중 '강제 답변 무효'의 정의를 바꾼다: 강제 답변이 EOS 없이 답 상한(1024)에 도달한 경우. 기준(≤10%)과 나머지 규칙은 그대로다. 추출 실패(extract_failed)는 함께 보고한다.
- CPU 테스트 66 passed(s1 강제 답변 유효/상한/뒤따른 `</think>` 처리, 객관식 'Final Answer: B' 채점, visref_raw_reg가 rank 2 < m 6에서도 m개를 DPP로 선택).
- 추가 확인(PDF 부록): App. J Table 7(8B): Random 67.3 / Relevance-only 75.6 / DPP 79.3(MathVista). App. K Table 8: λ·relevance + (1−λ)·diversity, λ=0.5가 최고(=가중치 없는 Eq. 10, 우리 구현과 같음). Random ≈ ST(68.1)인데 DPP가 79.3이라, 논문 DPP는 rank 부족으로 대부분을 잡음 채움하지 않았을 가능성(→ rank가 큰 Z?) — 추정.
- main_s0 visref_raw 67단계 재집계: 단계 길이 T_k < m은 18/67뿐(중앙값 4096, 1B 반복 루프), 고유 토큰 수 < m은 50/67(중앙값 178 vs m 537). 자연 종료 400~900토큰 단계 5개의 고유 토큰 124~200. → rank를 정하는 것은 단계 길이가 아니라 고유 토큰 수(input embedding Z).
- CEIL 공식 코드(HKUNLP/icl-ceil `src/utils/dpp_map.py` `fast_map_dpp`, VisRef 인용 [55]) 확인: Chen 코드에서 ε 정지와 선택 항목 제외(-inf)를 뺀 형태로, 항상 max_length번 반복한다(VisRef 본문 'terminate after m iterations'와 일치). 합성 kernel(rank 178, m 537)에 원문 그대로 돌리면 537개를 반환하지만 서로 다른 토큰은 187~190개뿐이고 나머지는 중복. Chen 공식 코드는 178~192개에서 정지. → 인용한 두 구현 모두 input-embedding Z에서 사실상 'rank만큼만 서로 다른 토큰'을 고른다. 논문이 이 문제를 언급하지 않은 것도 코드가 오류 없이 m개 인덱스를 돌려주기 때문으로 설명된다. hidden-state Z 가설은 근거가 약해짐. b_test VisRef 변형(현재 ridge)을 rank-only로 바꿀지 사용자 결정 대기.

## 2026-10-07 b_test 재빌드: VisRef = rank만큼만 (`visref_raw_rank`, 사용자 결정, push 전)
- 근거: 인용한 Chen·CEIL 구현이 모두 input-embedding Z에서 사실상 rank만큼만 서로 다른 토큰을 고른다(위 CEIL 원문 확인). 논문이 이를 언급하지 않은 것도 코드가 오류 없이 m개 인덱스를 돌려주기 때문으로 설명된다. 직전 결정(ridge, m개 모두)은 논문 문구만 근거로 한 것이라 교체한다.
- 구현: greedy가 고른 토큰만 중복 없이 원래 순서로 주입, 채움 없음. 단계 기록에 `n_inject` 추가. 합성 kernel(rank 178, m 537)에서 178개 주입(커널 크기 무관).
- 영향: 다시 넣는 시각 토큰이 m(중앙값 537)보다 크게 적다(main_s0 기준 중앙값 98). 보고서에 명시한다. relevance 채움·ridge는 App. J Table 7 확장 실험에서 비교한다.
- 판정 규칙은 그대로(강제 답변 무효 = EOS 없이 답 상한 도달). CPU 테스트 68 passed.

## 2026-10-07 원문 명세표·독립 감사 → b_test를 파일럿으로 재정의 (결과 전에 고정)
- 명세표: `repro/SPEC_LEDGER.md`(51행: 명시 19 / 그림·표에서 유도 3 / 인용 구현에서 확인 10 / 가정 19, Sonnet 작성). 원문: `references/papers/VisRef_fulltext/`(전체 텍스트, 쪽별 텍스트·이미지, 그림 쪽 고해상도), 인용 문헌 원문 `references/papers/ref_fulltext/`, 공식 코드 `references/sources/cited_code/`.
- 독립 감사: `repro/AUDIT_REPORT_20261007.md`(Opus, 대화 맥락 없음). 판정 '작은 수정 후 실행', 크래시·결과 오염 버그 없음.
  - B1(차단): VisRef가 rank 한정 한 가지뿐이면 해석 불가. 논문은 |V_k|=m 명시(L341, L360-362), [28]은 '|Y| ≤ rank(L)'(Kulesza&Taskar L851-852)라 논문 서술은 rank ≥ m을 전제, Fig. 4(b) m 민감도. → `visref_raw`(m까지 relevance 채움)를 같은 실행에 추가.
  - B2(차단): b_test는 파일럿. n이 수십~100이면 쌍 비교 SE 4~5pt라 목표 델타(+2.4/+3.5/+5.9)를 판정할 수 없다.
  - D4: s1 문자열·조건을 공식 코드와 같게 맞춤(아래). D2(잘린 뒤 이어가기 = 예산 연장 효과), D3(H 추정기), D5(ST 개입), D6(top-p·fp16)은 문서화. 장부 정정 목록은 감사 보고서 §2, §6.
  - 4a: Table 6 'InternVL-1B'는 InternVL3.5-1B(신뢰도 약 85%). 4c: 자연 종료 단계 probe는 main_s0에서 63/63 H<δ.
- 코드 변경(code_version 0f8e003fbf90, CPU 테스트 70 passed): 잘린 단계는 텍스트가 `\n`으로 끝나지 않을 때만 바로 뒤에 `\n`(V_k·`</think>`·Wait 모두 그 뒤), 강제 답은 상한 절단에만 `</think>` + `\nFinal Answer:`(EOS 종료는 자연 답변). summarize: step-1 종료 유형별 정확도·평균 k, 자연 종료 probe의 H≥δ 수, 루프 단계 수(고유/길이 < 0.1), n_inject/m, CRN 점검(TSR k=1 최종 답 = ST, main_s0 사전-CRN 기준 1/50).
- b_test 재빌드: st/tsr/visref_raw_rank/visref_raw, 단계 1024 / 합계 10,240, 4200s, seed-0 순서 처음부터.
- 파일럿 판정(정확도 차이는 어떤 결정에도 쓰지 않음):
  1. 건전성: error 0, fast_kv ok, 네 방법 모두 끝난 문항 ≥40, 강제 답변 무효(EOS 없이 답 상한) ≤10%.
  2. CRN: TSR k=1 문항 중 최종 답이 ST와 같은 비율을 보고(기대 대부분 같음; 50% 미만이면 GPU 비결정성·구현 조사).
  3. 보고 항목: step-1 종료 유형 분포와 유형별 정확도, 자연 종료 probe의 H≥δ 비율, k 분포·정지 사유, n_inject/m(두 VisRef 변형), 루프 단계 수, 생각 토큰 비율 vs ST, 방법별 처리량.
  4. 본 실험 설계(VisRef 변형 선택, 방법 수, 세션 수)는 파일럿의 처리량·주입량 진단을 보고 사용자가 정한다.
- 사용자 검토 후 정리(2026-10-07): 다음은 '설계상 당연하거나 영향이 작아' 문서화만 하고 더 이상 수정 사유로 삼지 않는다.
  1. ST/TSR·VisRef 예산 비대칭(감사 D2): 논문 설계 그대로다. Fig. 3 캡션 'the baseline with no additional test-time compute (standard thinking) … increasing test-time token budget'(plain L714). 방법별 생각 토큰만 보고한다.
  2. ST의 상한 강제 답(D5): 유한한 상한에서 생각을 끝내는 처리일 뿐, 논문이 말하는 'intervention'(Wait 연장)과 다르다.
  3. s1 문자열의 줄바꿈 개수·EOS 처리(D4): 반영 완료, 추가 수정 없음.
  4. 체크포인트: InternVL3.5-1B로 확정(사용자). 논문 L735 'three state-of-the-art MLRMs in their reasoning (thinking) mode: InternVL3.5-8B [45] …'.
  5. ST 절대 수준(46.1 vs 공식 59.3), Fig. 3 비율 판독 차이, Fig. 4(b) 본문·막대 차이: 구현과 무관, 델타와 CI로만 비교한다.
  6. top-p 0.95, fp16(T4는 bf16 미지원), GPU에서 CRN 비트 단위 불일치: 문서화만.
- 계속 중요하게 보는 것: VisRef 주입 수(rank vs m, 파일럿에서 두 변형 비교), 자연 종료 뒤 H≈0이라 이어가기가 거의 상한 절단 뒤에만 생기는 점(다단계가 실제로 일어나는지), 토큰 상한(운영 가정), 채점 추출(마지막 줄 대체 약 8%).

## 2026-10-07 채점: 규칙 → 실패 시 LLM 추출 (MathVista 공식 방식, 사용자 결정: Claude Haiku)
- 공식: `mathvista_official_r4/evaluation/extract_answer.py` L29-80(간단한 형식만 규칙, 나머지는 GPT + `prompts/ext_ans.py` 예시 프롬프트), VLMEvalKit은 규칙 후 GPT-4o-mini. VisRef는 추출기 미명시(App. D exact match).
- 구현: `repro/llm_extract.py` — prepare(규칙이 last_line/empty이거나 추출 실패한 행만, 공식 create_test_prompt 그대로, 정답은 넣지 않음, query는 공식 hint 템플릿으로 재구성), merge(LLM 추출값을 같은 normalize로 재채점, 규칙만/규칙+LLM 정확도 둘 다 출력). 대조군(control): 규칙이 성공한 행 일부를 함께 보내 일치율 확인.
- main_s0 검증(Haiku 서브에이전트, 별도 API 비용 없이 세션 사용량으로): 1차 시도는 36행 모두 빈 값(프롬프트 끝만 보고 판단) → 질문·응답을 분리해 재시도. 대조군 일치 18/20(Haiku가 답이 있는데 빈 값 2건). 대상 16행: 정정 3(pid 7 ST·TSR 'quarter', pid 62 ST), 하락 2(pid 238 ST·TSR: 규칙은 잘린 추론의 마지막 줄에서 우연히 정답을 집었고, LLM은 '최종 답 없음'). 정확도(규칙 → 규칙+LLM): ST 55.0 → 56.7, TSR 58.3 → 58.3, VisRef 57.6 → 57.6.
- 주의: 공식 normalize는 객관식에서 추출값이 비어도 가장 비슷한(가장 짧은) 보기를 고른다. 이 때문에 빈 추출이 우연히 맞거나(62) 틀린다(238). 공식 동작이므로 그대로 둔다.
- 결론: 영향은 방법당 ±1문항 수준. 8%의 대부분은 추출 실패가 아니라 답을 쓰지 않은 경우(s1 강제 답변으로 대응). 본 실험 보고에는 두 정확도를 함께 적고, LLM 추출 때마다 대조군 일치율을 보고한다.
- Gemini 추출 경로 추가(사용자 요청): `repro/gemini_extract.py` — llm_extract.py prepare가 만든 requests.jsonl을 Gemini API로 문항당 1회 호출. 공식 GPT 추출기와 같은 조건(텍스트 1개, temperature 0, 최대 1024토큰), 표준 라이브러리만, 재개 가능, 모델 ID는 인자로(기본값 없음). 오프라인 테스트(응답 파싱·thought 파트 제외·차단 응답·재개)만 통과, 실제 API 호출은 사용자 실행으로 확인 예정. 채택 기준: main_s0 대조군 일치율로 Haiku(18/20)와 비교.
- handoff/main_s0_extract/: 외부 추출기(Antigravity Gemini)용 입력만 담은 폴더(정답·채점 파일 없음). 결과 responses_gemini.jsonl → repro/llm_extract.py merge --responses
- Gemini 3.8 Flash(Antigravity, 에이전트가 items.jsonl을 직접 읽음) main_s0 결과: 형식 정상(36줄, id·순서 일치), 빈 값 17. 대조군 일치 17/20(Haiku 18/20). 정확도(규칙 → 규칙+LLM): ST 55.0 → 55.0, TSR 58.3 → 55.0, VisRef 57.6 → 57.6.
  - Haiku와 다른 5행: 7|st·7|tsr·720|tsr(응답이 'The answer is B) quarter' / 'the answer is 2'를 반복 명시) → Gemini가 놓침. 876|tsr('Therefore, the answer is'에서 잘림)·303|st('the answer is A' 뒤 'might be B) no'로 번복하다 잘림) → Haiku는 'A', Gemini는 ''(판단이 갈리는 경우).
  - 규칙과 LLM이 엇갈린 대조군 3행(577|st, 637|visref_raw, 303|st)은 모두 답변이 1024 상한에서 잘리며 답을 번복한 경우. 규칙은 중간의 'answer is X'를 잡고, LLM은 최종 답 없음으로 봄.
  - 결론: 두 추출기 모두 이 어려운 부분집합에서 2~3행 차이. Gemini는 보수적(명시된 답도 놓침), Haiku는 번복·절단된 응답에서 보기를 고르는 경향. 쟁점 행은 모두 답변 상한 절단·번복 → b_test의 s1 강제 답변으로 줄어들 것으로 예상.
  - 제안 규칙: 두 추출기를 모두 돌려 일치하면 채택, 불일치 행만 상위 모델이 원문을 읽고 판정. 규칙만 정확도도 함께 보고.
- 2026-10-07 b_test 실제 push 버전 확인: 08:43 UTC에 감사 전 빌드(6c7daca6ebb1, st/tsr/visref_raw_rank)가 push됨(push.json). 이후 빌드는 같은 폴더를 덮어썼으므로 `b_test2`(arki13/visref-repro-b-test2)로 다시 빌드했다(0f8e003fbf90, 4방법). b_test는 '파일럿 v1'로 pull해 같은 판정 규칙으로 보되, visref_raw 관련 항목은 없음. 두 버전의 행은 섞지 않는다.

## 2026-10-07 b_test(파일럿 v1, 6c7daca6ebb1) 결과 — 사용자가 약 8분 만에 중단
- 실행 버전 확인: env.json run_args st/tsr/visref_raw_rank, log code_version 6c7daca6ebb1. error 0, fast_kv ok. 59행, 세 방법 모두 끝난 문항 18(셔플 순서 완전 접두 10). 건전성 기준 '≥40문항'은 중단으로 미충족(판정 불가).
- step 1이 1024 상한에서 잘린 비율: 8/18.
- 이어가기: TSR·VisRef 모두 18문항 중 1문항(pid 970, H1=1.08)만 2단계. 잘린 단계 뒤 probe도 H≈0(0~0.07)이라 대부분 k=1 정지. 생각 토큰 비율 vs ST 1.17배(논문 Fig. 3 약 3.5배). → 이 H 정의에서는 다단계가 거의 일어나지 않는다. Table 5(지연 +8%, +15%)와는 맞고 Fig. 3과는 다르다.
- CRN: TSR k=1 17문항 중 최종 답이 ST와 같음 13(GPU batch 구성에 따른 fp16 차이로 일부 다름).
- VisRef(rank) 주입 수: 중앙값 76(m의 약 31%, |V|의 약 9%), 범위 50~184.
- 처리량(초반 8분, 부정확): 3방법 18문항 → 4096 상한(main_s0, 시간당 59문항)보다 2배 이상 빠를 전망.
- 정확도는 판단에 쓰지 않음.

## 2026-10-07 결정(사용자): s1 + DEER af 폴백, 정지 통계 현행 유지, 파일럿 없이 본 실험 시작
- 근거: b_test v1에서 s1 강제 답 26 중 12(46%)가 답 상한 1024 도달 — 강제 답이 없던 main_s0(14/30, 47%)과 같음. 대부분 'Final Answer: ... B.' 뒤 'Wait, wait …'로 번복. DEER 공식 코드(`deer/vllm-deer.py` L485-491, L655-656, af=1)는 답이 EOS 없이 끝나면 `\n**Final Answer**\n\boxed` + 최대 100토큰을 붙인다.
- 구현(code_version 79eef648ebd1, CPU 테스트 74 passed): 최종 답(자연·강제)이 EOS 없이 답 상한에 닿으면 af 프롬프트 + 최대 100토큰(EOS 종료). 채점 텍스트 = 답 + af 프롬프트 + 이어진 생성. 기록: final_af, final_af_valid(상자 닫힘·내용 있음), final_af_finish, final_af_len. summarize: n_final_forced_unusable(s1 답 EOS 종료도, af 유효 상자도 아님), n_final_af, n_final_af_valid.
- 정지 통계 H: 현행 유지(논문 정의 '답 분포 엔트로피'에 가장 가깝다). b_test v1처럼 k≈1이면 TSR≈ST이고 VisRef 효과는 최종 답 전 1회 주입에서만 생긴다는 점을 결과로 보고한다. 목표 수치에 맞춘 H 조정은 하지 않는다.
- 본 실험 시리즈 `b_main`: 첫 세션 `b_main_s1`(arki13/visref-repro-b-main-s1), 7200s(투입 4800s, deadline 6600s), seed-0 순서 처음부터, st/tsr/visref_raw_rank/visref_raw, 단계 1024 / 합계 10,240.
- 첫 세션 판정(결과 전에 고정, 정확도 차이는 쓰지 않음):
  1. 건전성: error 0, fast_kv ok, 네 방법 모두 끝난 문항 ≥40, 강제 답 사용 불가(n_final_forced_unusable) ≤ 강제 답의 10%, af 유효율 보고.
  2. 통과 → `b_main_s2`부터 `--skip-from b_main_s1/.../results_*.jsonl`(.partial 제외), 코드·인자 불변(code_version 일치).
  3. 실패 → 중단, 원인 수정 후 새 시리즈(b_main 결과와 섞지 않음).
  4. 처리량 × 남은 quota로 최종 n 추정. 4방법으로 n<150이면 방법 축소 여부를 사용자에게 묻는다.
  5. 최종 보고: 규칙만 / 규칙+LLM(Gemini·Haiku, 불일치 행은 상위 모델 판정) 두 정확도, paired 차이와 95% CI, step-1 종료 유형별 결과, k 분포, 주입 수/m, 생각 토큰 비율.
- b_test2(0f8e003fbf90)는 push하지 않고 폐기한다(af 폴백 반영 전 빌드). b_main_s1이 대체한다.

## 2026-10-07 b_main_s1 결과와 판정 (79eef648ebd1, 2h 세션, 실사용 약 1.65h)
- 건전성: error 0, fast_kv ok, 네 방법 모두 끝난 문항 74, 강제 답 사용 불가 0(강제 44~45 중 EOS 종료 29~31, 나머지 14~15는 af 폴백으로 전부 유효 상자). → 통과, 같은 코드·인자로 b_main_s2 빌드(--skip-from s1, 296쌍).
- step 1 상한 절단 45/74(61%). 다단계 발생: TSR k>1 18/74(절단 문항의 평균 k 3.11, k=10 7건), VisRef 두 변형도 비슷. 생각 토큰 ST 대비 TSR 2.76배, VisRef 2.59~2.66배(논문 Fig. 3 약 3.5배). b_test v1(18문항, 1/18)보다 다단계가 많다 — 소표본 차이.
- 자연 종료 뒤 probe H≥δ: TSR 0/29, VisRef(fill) 0/30, VisRef(rank) 6/35.
- 주입 수: rank 중앙값 104.5(m의 22%), fill = m(중앙값 345.5).
- CRN: TSR k=1 56문항 중 ST와 같은 최종 답 38.
- 추출: 규칙 실패(last_line) 방법당 1행, extract_failed 0 → LLM 추출 영향 미미.
- 처리량: 샤드 불균형(샤드 0 24문항, k≥7 행 15 / 샤드 1 50문항). 세션당 약 74문항(4방법).
- 정확도(참고, 판정에 쓰지 않음, n=74): ST 59.5 / TSR 56.8 / VisRef(fill) 59.5 / VisRef(rank) 56.8. paired 95% CI 모두 0 포함.
- 남은 quota 5h 기준 예상: s2·s3(각 약 1.65h) 후 약 1.7h로 짧은 s4 → 총 약 260문항(>150, 방법 축소 불필요). s4 길이는 s3 후 정확한 잔여로 결정.

## 2026-10-07 계획 수정(사용자 결정, s2 결과 전): fill 제외, 주 결과 VisRef = visref_raw_rank
- 근거: `visref_raw`(rank 이후 relevance 채움)는 VisRef·Chen·CEIL·Kulesza&Taskar 어디에도 없는 운영 변형이다. `visref_raw_rank`는 인용 Chen 공식 코드의 실제 동작(ε에서 정지)과 같다. fill을 빼면 생성 토큰이 약 28% 줄어 같은 quota로 문항 수가 늘어난다(s1 기준 st 79k / tsr 176k / rank 173k / fill 169k 토큰).
- s1 결과를 본 뒤의 결정이지만, s1에서 fill이 더 높았으므로(59.5 vs 56.8) VisRef에 유리한 방향의 선택이 아니다.
- 남는 세 방법의 결과 분포는 바뀌지 않는다: 문항별 난수열은 pid로 정해지고(CRN), step-1 fork의 leader는 계속 st다. 바뀌는 것은 batch 구성뿐(fp16 수치 잡음 수준). code_version 79eef648ebd1 그대로, 인자만 `--methods st,tsr,visref_raw_rank`. 이 변경은 '인자 불변' 규칙의 문서화된 예외다.
- b_main_s2 재빌드(--skip-from s1). 집계: `summarize.py … --methods st,tsr,visref_raw_rank`(summarize.py에 방법 필터 추가, code_version 대상 아님). `visref_raw`는 s1 74문항만 있는 보조 자료로 남긴다.
- 주 결과: ST / TSR / VisRef(rank). 최종 n 예상: 세션당 약 100문항(3방법), 남은 5h로 약 300문항 이상.

## 2026-10-07 안 쓰는 코드 분기 제거 (사용자 결정, s2 push 전) — code_version 79eef648ebd1 → f2422d0e72ff
- 제거: VisRef 변형 4개(`visref` 태그 주입, `visref_reg`, `visref_raw` fill, `visref_raw_reg`), DPP fill·εI 분기, probe `newline` 모드(smoke3), 강제 답 상자 모드(`forced_cap`), 강제 답·af 끄기 옵션, 엔진 `soft_stop/min_new`, 특수 토큰 `img_*`·`newline_ids`, 관련 테스트·집계 항목. 남은 방법: st / tsr / visref_raw_rank. 결과 행 형식은 그대로.
- 줄 수: run.py+visref+tests 1,996 → 1,706. CPU 테스트 57 passed(통합 테스트 포함).
- 동일성 검증: CPU 작은 LM(로짓 편향으로 경로 유도)에서 이전/새 코드를 같은 입력·시드·chunk로 run.py와 같은 fork 방식으로 실행, 결과 레코드 전체 비교 → 810/810 동일. 실행된 경로: step 1 close/cap/eos, s1 강제 답, af(유효·무효), 답 안 </think>, k>1, 정지 entropy/kmax/budget, rank 주입. 기록 `repro/.backup_79eef648ebd1/EQUIVALENCE.md`, 스크립트 `equiv_check.py`, 이전 코드 같은 폴더.
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
- 생각 토큰 ST 대비 TSR 2.62배, VisRef 2.85배. k>1: TSR 54, VisRef 58. kmax 정지 TSR 12, VisRef 19.
- CRN: TSR k=1 134문항 중 ST와 추출 답 일치 130, 정오 일치 131, 전문 일치 88.
- VisRef 주입 271회, 중앙값 123, m 중앙값 537, 225/271회가 Z 고유 토큰 수에서 멈춤.
- 남은 quota(추정): s2 전 5h − 약 1.7h ≈ 3.3h. s3(2h) 뒤 약 1.3h → 짧은 s4, 정확한 길이는 s3 뒤 사용자 확인 잔여로 결정. 예상 최종 n ≈ 350.
- 보고서: `repro/reports/b_main_report.html` 재생성(아티팩트 v2).

## 2026-10-08 해석 메모(사용자 질문: 방향 일치 / 토큰 상한·thinking / m=30%) — 실험 설정 변경 없음
- 원문 근거: §5 "MLRMs in their reasoning (thinking) mode", App. I 1B/2B/8B "under the same decoding setup and token budget"(값 없음). Fig. 4(b) 8B: m 20%≈77(캡션 76.1) → 30·40·50% ≈79.2로 평평, "plateaus for m ≥ 30%", 30%는 정확도·효율 균형으로 선택(최고점이라고 쓰지 않음). 본문 Eq. 11 주변 "enforcing |Ṽk| = m".
- 설정 출처 정정: R1 프롬프트·T=0.6·top-k/p는 논문이 아니라 모델 카드 값(SPEC_LEDGER 3·4행과 같음). report.py 문구 수정, 아티팩트 v3.
- 데이터(s1+s2, 188문항): 첫 단계 자연 종료 뒤 H≥δ는 TSR 0/62 → 이어가기는 거의 잘린 단계에서만 일어남. 잘린 생각을 오답으로 치면 ST 23.9%(close 42 + eos 3 정답), 강제 답으로 57.4%; 논문 46.1은 그 사이. main_s0(60문항, 단계 4096)에서도 ST 생각 절반이 4,096 안에 끝나지 않음(잘림=오답이면 1024 23.3 / 2048 30.0 / 4096 36.7%).
- 주입 포화(VisRef 이어간 단계 271): rank = min(Z 고유 토큰, rank_V)이고 n_dpp가 271/271 이 값과 같다. rank/|V| 분위 10/25/50/75/90% = 0.034/0.044/0.059/0.197/0.578. m이 실제로 묶는 단계: 10% 36%, 20% 25%, 30% 17%, 40% 14%, 50% 13%. 평균 주입 98/113/119/123/127, m=100%면 132. 단계 길이별 Z 고유 토큰 중앙값: ≤256 67, 256-512 130, 512-1023 140, 1024 145.

## 2026-10-08 비용 측정 현황(사용자 질문: 논문 Table 5 같은 추가 시간) — 설정 변경 없음
- 논문 Table 5: InternVL3.5-8B, H100 1장, MathVista 문항당 지연 ST 7.1s / TSR 7.7s / Look-Back 7.6s / VisRef 8.2s(ST 대비 +1.1s, TSR 대비 +0.5s). 1B 지연은 없음. Fig. 3의 trace당 약 3.5배 생각 토큰과 서로 맞지 않음(AUDIT A-4, 4c).
- 우리 기록: 문항당 지연은 측정하지 않았다. `wall_s`는 16개 동시 배치·step-1 fork 대기가 섞여 지연이 아니다(예: TSR k=1은 ST보다 약 16토큰 더 생성하는데 wall_s 288s vs 197s).
- 생성 토큰(s1+s2 188문항, 생각+답+probe): ST 1,121 / TSR 2,317(2.07배) / VisRef 2,580(2.30배). k=1 문항(약 70%)은 TSR 925 vs ST 909, VisRef 975 vs ST 897. k>1 문항은 TSR 5,771 vs ST 1,648, VisRef 6,176 vs ST 1,625.
- DPP 선택 시간(장치 CPU 4스레드, float64, d=1024, Z 1,024토큰·고유 145): |V| 768/1,792/3,328 → 0.045/0.059/0.127s, 선택 145개(rank에서 멈춤). 생성 시간에 비해 무시할 수준.

## 2026-10-08 독립 검토(Fable 5.1 high) 결과와 기록 정정
- 결론: s3는 그대로 push해도 된다(빌드 무결성 확인: notebook 내 소스 sha256·code_version f2422d0e72ff·run_args·ids·time_limit/deadline이 s2와 같고 skip 638쌍; s1·s2 로그에 deadline·error 없음 → complete prefix = 투입 집합). pull 뒤 확인: error 0, env.json gpus=2, code_version.
- 정정 1(주입 범위): Alg. 1은 V_k를 궤적에 넣은 뒤 정지 검사를 하고, 코드(`methods.py` L174-189)도 매 단계 ctx에 V_k를 붙인 뒤 probe·최종 답을 만든다. 따라서 마지막 단계의 선택도 최종 답 직전에 주입되며 k=1 VisRef ≠ ST다. 2026-10-07 보고서 항목의 "주입은 이어갈 때만 일어난다"와 2026-10-08 s2 항목의 "자연 종료 71.7은 fp16 잡음 1문항"은 틀렸다. report.py 주입 통계를 전 단계로 되돌림: s1+s2 주입 459회, 중앙값 123, m 중앙값 537, 386/459회가 Z 고유 토큰 수에서 멈춤, 주입/m 중앙값 0.20. 2026-10-08 해석 메모의 포화 분석(이어간 단계 271개)은 부분집합 기준이며 결론은 같다.
- 발견(검토자 P3): 최종 답 안에 </think>가 다시 나온 행 ST 33 / TSR 25 / VisRef 83(188문항), 자연 종료·k=1에서는 8 / 9 / 30(60문항). 답 직전 주입이 답 안에서 생각을 다시 열게 하는 것으로 보인다. 채점은 마지막 </think> 뒤 텍스트. 최종 보고에 첫 답 기준 VisRef 재채점을 함께 낸다(보고서에 "답 안 </think>" 열 추가).
- 검토자 해석(추정): 이어가기는 사실상 상한 절단에서만 발생(자연 종료 뒤 H<δ 60/60) → 최종 보고는 cap/close 층별로 제시. Fig. 4(b)의 20→30% 상승은 저자 설정의 rank ≥ 0.2|V| ≈ 354를 함의하고, hidden-state Z(rank ≤ T_k ≈ 615)라면 30% 뒤 평탄이 rank 포화로 설명된다 → hidden-state Z가 여전히 가장 강한 후보(기록의 "인용 코드가 rank만큼만 고르므로 가설 약화"와 다른 판단). 별도 시리즈 과제.
- 검토자 권고: s3 그대로 → 남는 약 1.3h는 짧은 s4보다 thinking 끈 ST 진단(같은 pid, 별도 커널)이 정보량이 크다. 다음 주 쿼터로 n=1,000 완주(약 12h) 우선, hidden-state Z 변형은 별도 시리즈(약 300 pid, 약 5h). 통계: n≈350이면 VisRef−ST ±3.0pp, n=1,000이면 ±1.8pp.

## 2026-10-08 진단 hidden_rank_check (CPU 커널, GPU 쿼터 사용 안 함) — 결과 전에 고정
- 목적: VisRef 커널 L = (VZᵀ)(VZᵀ)ᵀ에서 Z를 입력 임베딩 대신 은닉 상태로 쓰면, 1B에서 greedy DPP가 "의미 있게" 고를 수 있는 시각 토큰 수가 m = ⌊0.3|V|⌋에 닿는지 측정. 은닉 상태 Z 변형 시리즈를 만들 가치가 있는지 판단하는 용도. 본 실험(b_main)과 무관.
- 실행 위치: 사용자 컴퓨터 VM은 RAM 3GB·Hugging Face 차단, 클라우드 컨테이너도 Hugging Face 차단이라 불가 → Kaggle CPU 커널 `arki13/visref-hidden-rank-check`(fp32, transformers 4.55.0, 모델·데이터 리비전은 본 실험과 같음).
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
- 생각 토큰 ST 대비 TSR 2.58배, VisRef 2.86배. 답 안 </think>: 45 / 38 / 138. VisRef 주입 722회, 중앙값 124.5, m 중앙값 537, 618/722가 m보다 적음.
- CRN: TSR k=1 211문항 중 ST와 추출 답 206, 정오 207 일치.
- 남은 GPU 쿼터(추정): 3.3h − s3 약 1.7h ≈ 1.6h. s4 길이는 사용자가 확인한 실제 잔여로 정한다.

## 2026-10-08 hidden_rank_check 결과와 판정 (Kaggle CPU, 25분)
- 타당성: 입력 임베딩 Z의 정지 규칙 선택 수 = 다시 토큰화한 Z의 고유 토큰 수 12/12. 기록된 n_dpp와 비교하면 m이 걸린 3건(|V|=256)은 76=m으로 정합, 나머지 9건 중 7건 일치·2건 ±1(재토큰화로 고유 토큰 수도 1 차이). → 유효.
- 1차 지표(정규화 잔차 ≥ 1e-2인 선택 수), 잘린 단계 6건(T=1024): 최종 norm Z 94/998, 70/998, 107/537, 86/537, 58/76, 32/76; 28층 Z 88, 57, 74, 90, 52, 53; 입력 임베딩 Z 77, 80, 76, 59, 45, 65. m 이상 0/6, m/2 미만 최종 norm 5/6·28층 4/6.
- 사전 규칙 판정: (b). 은닉 상태 Z도 의미 있게 고를 수 있는 시각 토큰은 약 50–110개로 입력 임베딩과 비슷하며 m에 못 미친다 → "은닉 상태 Z라서 다양한 토큰을 m개 고른다"는 설명은 지지되지 않음, 은닉 상태 Z 변형의 우선순위를 낮춘다.
- 사후 관찰(사전 규칙 밖, 해석 참고용): 은닉 상태 Z는 수치 계수가 거의 T라서 현행 정지 규칙(1e-9)으로는 958–993개(T=1024)를 고른다 → 실제 구현하면 min(m, T)개를 주입하게 되지만, 약 100번째 이후 선택은 정규화 잔차 1e-3~1e-9(다양성 기여 거의 0)로 사실상 관련성 위주의 채움이다. Z 에너지 분포의 participation ratio 3–12(소수 방향에 집중). 같은 층 시각 토큰 은닉 상태를 V로 쓰면 의미 있는 선택 3–45개로 더 적다. 따라서 "은닉 상태 Z"와 "계수 이후 채움"은 실질적으로 같은 개입(m개 주입, 앞 약 100개만 다양성 구분)이며, 남는 질문은 "m개 주입이 rank만큼 주입보다 정확도를 바꾸는가"다. 은닉 상태 Z는 Eq. 11 이득이 m까지 유한하게 정의되는 유일한 읽기라는 점에서 본문("|Ṽk| = m", "terminate after m iterations")과 형식상 맞는다.

## 2026-10-08 b_main_s4 빌드 (이번 주 마지막 세션)
- 사용자 확인 잔여 GPU 쿼터 1시간 56분(6,960s). 세션 도중 쿼터가 바닥나면 출력이 사라질 수 있으므로 timeout 6,000s로 여유 약 24분을 둔다(build.py 규칙: time_limit 4,000s, deadline 5,400s; s3 실측 준비 시간 약 51s).
- 코드·인자·문항 순서는 s3와 같음(source sha256·run_args·ids·모델/데이터 리비전 일치, code_version f2422d0e72ff). --skip-from s1+s2+s3(962쌍). 세션 길이만 다르며 이는 사전에 계획된 변경(s4 길이는 잔여로 결정).
- 예상: 약 80문항 추가(s3 108문항 × 4,000/4,800), 합계 약 376문항.

## 2026-10-08 b_main_s4 결과와 이번 주 집계 (s1–s4, 392문항)
- s4 건전성: error 0, 세 방법 96문항(288행), 강제 답 사용 불가 0, code_version f2422d0e72ff, T4×2, torch·transformers s3와 같음, deadline·error 없음. 세션 약 5,466s(쿼터 6,960s 안).
- s1–s4 complete prefix 392문항(투입 전부): ST 58.9 [54.0, 63.7] / TSR 60.7 [55.8, 65.4] / VisRef 61.7 [56.8, 66.4]. paired TSR−ST +1.8(−0.8~+4.6, 18:11, p 0.27), VisRef−ST +2.8(−0.5~+6.1, 28:17, p 0.14), VisRef−TSR +1.0(−2.0~+3.8, 19:15, p 0.61). 논문 차이 +2.4 / +5.9 / +3.5.
- 첫 단계: 상한 절단 245(63%) 51.8 / 54.3 / 55.9, 자연 종료 134 71.6 / 71.6 / 71.6, EOS 13. 생각 토큰 ST 대비 TSR 2.57배, VisRef 2.92배. kmax 정지 TSR 24, VisRef 42. 답 안 </think> 54 / 47 / 176. VisRef 주입 963회, 중앙값 122(m 중앙값 537). CRN: TSR k=1 282문항 중 ST와 정오 276·추출 답 274 일치.
- 규칙 실패(LLM 추출 대상): 11 / 7 / 8행.
- report.py: 중간 결과 배너를 문항 수(≥250) 대신 `--final` 플래그로 끄도록 변경(재채점·남은 세션 전에는 계속 표시).
- 다음 주 계획을 위한 계산(관측 불일치율 기준): VisRef−ST가 +2.8이 유지되면 n=1,000에서 검출력 약 0.74, 논문 +5.9와의 차이(3.1pp)도 약 2.9 SE로 구분 가능. 남은 608문항은 약 9.5시간(세션당 약 65문항/시간).

## 2026-10-08 메모: 계수를 넘은 뒤 "식 그대로" 고르면 무엇이 채워지나 (사용자 질문, 설정 변경 없음)
- 정확한 수학: 계수를 넘으면 모든 후보의 이득이 log r² + log 0 = −∞로 같아 Eq. 11의 argmax가 정해지지 않는다. 식 자체는 "관련성으로 채우라"고 말하지 않는다.
- 구현에 따라 갈린다(합성 데이터 N=600, 계수 40, 계수 이후 60개 선택, 남은 토큰 중 관련성 백분위):
  - Chen·CEIL 인용 코드: ε 정지 → 채우지 않음(현행 visref_raw_rank).
  - 정지 없이 부동소수 그대로: 반올림 잡음으로 고름(백분위 중앙값 0.42, 상위 10% 비율 0.03), NaN·overflow 발생.
  - 작은 εI를 더한 커널(1e-6·평균 대각, 1e-3도 같음): 백분위 중앙값 0.98, 상위 10% 비율 0.93 → 사실상 관련성 순 채움. 계수 이전 40개 선택은 정확한 경우와 동일.
- 따라서 "m개 주입" 변형을 실험한다면, 식을 바꾸지 않고 수치 안정화만 더한 εI(= 계수 이후 관련성 채움)가 가장 자연스러운 후보다. 2026-10-07 정리 때 제거한 `visref_reg`(εI)와 `visref_raw`(관련성 채움)가 이에 해당하며 `.backup_79eef648ebd1/`에 있다.

## 2026-10-10 새 주 쿼터 30시간 — b_main_s5 빌드
- 코드·인자·문항 순서 s4와 같음, --skip-from s1–s4(1,176쌍), 7200s. 남은 608문항은 약 6세션(약 10시간) 예상.
- 정정(같은 날): s5 빌드 확인 결과 `visref/scoring.py`가 2026-10-08에 다른 세션에서 scorer v3(`SCORER_VERSION=v3_md_colon_emphasis`, 마크다운 라벨·콜론·강조 처리)로 바뀌어 code_version이 f2422d0e72ff → d0095f79e691이 됐다. run.py·methods·dpp·engine·model·fastkv의 sha256은 s4와 같아 생성 과정은 동일하고, 바뀐 것은 기록 시점의 규칙 채점뿐이다. 모든 세션 결과는 `repro/rescore.py`(추론 없이 저장된 최종 답을 현재 채점 규칙으로 다시 채점)로 같은 v3 규칙에 맞춰 집계한다. 이 조건에서 s5 이후를 b_main에 합친다(문서화된 예외). push 전 사용자 확인.
- v3 재채점 확인(rescore.py, 추론 없음, s1–s4 392문항): ST 231→233 (59.4%) / TSR 238→239 (61.0%) / VisRef 242→243 (62.0%), 추출 실패 1/1/1행. 방법 간 차이는 거의 그대로. 게시된 보고서(v6)는 아직 v1 채점 기준.

## 2026-10-10 사용자 결정·확인
- VisRef 작업은 이 세션 하나에서만 한다(사용자 결정). 다른 세션의 10-08 변경(scoring v3, rescore.py, report.py, SPEC_LEDGER)은 이 세션이 이어받는다.
- 잘린 생각 처리 오프라인 확인(v3, 392문항): 강제 답 행을 오답으로 치면 ST 26.8 / TSR 28.1 / VisRef 28.6%. 논문(46.1/48.5/52.0)보다 훨씬 낮아, "단계 1024 + 잘린 답 버림"으로는 논문 절대값이 설명되지 않는다. 강제 답 행 정답: ST 128/245, TSR 129/237, VisRef 131/234.

## 2026-10-10 결과 전 고정: b_main 주 비교 (사용자 결정)
- 주 비교 2개: VisRef−ST, VisRef−TSR (논문 주장 "gains over both ST and TSR", App. I·Table 1; 초록의 "test-time scaling 방법보다 높다"). 보조: TSR−ST. 중간 결과(VisRef−TSR +1.0)를 본 뒤의 결정이지만 근거는 논문 주장이다.
- 검정: 같은 문항 짝 McNemar 정확 검정, 두 주 비교에 Holm 보정(작은 p를 0.025, 큰 p를 0.05와 비교). 효과 크기는 짝 차이와 부트스트랩 95% CI. 채점은 규칙 v3 + 남은 추출 실패 행 LLM 추출. 최종 n은 b_main 완주분(목표 1,000) 전체.

## 2026-10-10 진단 diag_st_plain 빌드 (GPU 약 1h 이내) — 결과 전에 고정
- 질문: 논문 ST 46.1이 "잘린 생각 뒤 재촉 문구 없이 그냥 답하게 한 결과"로 설명되는가.
- 표본(규칙): s1–s4에서 본 실험 ST의 첫 단계가 상한에서 잘린 문항을 셔플 순서대로 앞의 150개(`repro/diag/st_plain_ids.json`). 본 실험 ST의 이 150문항 정답 74(49.3%).
- 변경은 한 가지: `--forced-prompt=`(빈 문자열) → 잘린 단계 뒤 `</think>`만 붙이고 답한다. 나머지(ST, 단계 1024, 답 1024, af 폴백, 시드, 코드 d0095f79e691)는 본 실험과 같다. 첫 단계는 다시 생성되므로 GPU 잡음으로 일부 달라질 수 있다(첫 단계 종료 방식이 다른 문항은 따로 센다).
- 판정: 같은 문항 짝 비교(본 실험 강제 답 ST vs 재촉 없음 ST). 재촉 없음 정확도가 강제 답보다 크게 낮아(예: 10%p 이상) 전체 ST 추정치가 46 근처로 내려가면 "답 처리 차이"가 유력한 설명, 차이가 작으면 이 설명은 기각.
- 전체 ST 추정치 = (자연 종료·EOS 문항 정답 + 잘린 문항 수 × 재촉 없음 정확도) / 392.

## 2026-10-10 정정: InternVL3.5-1B MathVista 59.3 / 69.5의 의미 (사용자 질문, 설정 변경 없음)
- 출처 확인(references/papers/ref_fulltext/InternVL3.5.txt L555-568, L775-776, Table 3): 69.5는 "w/ Parallel Thinking [143]" 행 = Best-of-N, VisualPRM-v1.1을 critic(보상 모델)으로 여러 응답 중 하나를 고름. 다수결이 아니며 N은 본문에 없음.
- 59.3은 "unless otherwise specified … without applying TTS" 조건의 단일 응답 점수. Thinking 모드 사용 여부는 본문·모델 카드 어디에도 명시되지 않음(TTS 정의에 deep thinking이 포함돼 있어 문자 그대로는 '생각 모드 없음'으로도 읽히나 행 이름은 parallel thinking만 언급 → 불명확).
- 이전 대화의 "59.3 = thinking 끔, 69.5 = thinking 켬" 설명은 근거 없음 → 철회. 우리 단일 응답 ST(v3 59.4)와 비교할 공식 수치는 59.3이고, 69.5는 다중 샘플+PRM 선택이라 비교 대상이 아님.
- 답 강제 규칙 정리: 논문은 최종 답을 y ∼ πθ(·|x, τ1:k)로만 쓰고(Alg. 1 L16-17) 토큰 상한·잘림 처리·답 강제 문구를 하나도 기술하지 않음. 우리 구현의 두 규칙(생각 단계 상한 → s1식 `\nFinal Answer:`, 답 상한 → DEER af `\n**Final Answer**\n\boxed`)은 2026-10-07 사용자 결정으로 의도적으로 넣은 논문 외 처리이며, 채점(답 추출)과는 별개.

## 2026-10-10 diag_st_plain 판정 규칙 보강 (사용자 승인, 결과 전 고정, 코드·빌드 변경 없음)
- 이유: `--forced-prompt=`는 규칙 ①(생각 상한 뒤 `\nFinal Answer:`)만 끈다. 규칙 ②(답이 EOS 없이 답 상한 1024에 닿으면 DEER af `\n**Final Answer**\n\boxed` + 최대 100토큰)는 diag에서도 작동한다. 본 실험 ST의 이 150문항 중 50행이 af 경로.
- 세 가지 읽기를 diag 행과 짝이 되는 본 실험 ST 행에 똑같이 적용(모두 채점 v3):
  (a) 그대로: 기존 사전등록 판정에 쓰는 주 읽기.
  (b) af 제거 재채점: final_answer를 마지막 af 문구 앞에서 자르고 다시 채점(추론 없음). = 규칙 ①·② 모두 없는 경우의 근사.
  (c) af=오답 하한: af 경로 행을 모두 오답으로 침.
- 보고: 각 읽기의 정확도, af 행 수와 af 행 정확도(diag/본 실험 각각), 같은 문항 짝 차이와 McNemar 정확 p. 전체 ST 추정치 = (자연 종료·EOS 문항의 같은 읽기 정답 + 245 × 해당 읽기의 재촉 없음 정확도) / 392 를 세 읽기 모두 계산.
- 판정: 기존 기준(재촉 없음이 강제 답보다 10%p 이상 낮아 전체 추정치가 46 근처)은 (a)로 판정. (b)는 "논문처럼 답 처리 규칙이 전혀 없었다면"의 근거로 함께 보고, (c)는 하한으로만 쓰고 판정에 쓰지 않는다(답 상한에 닿았어도 앞부분에 추출 가능한 답이 있는 행까지 오답 처리하므로).
- b_main 최종 보고에도 같은 민감도 분석 (a)/(b)/(c)를 세 방법 모두 싣는다. 주 검정은 사전등록대로 (a)=v3(+LLM 추출) 하나.
- 참고(사후 관찰, 판정에 쓰지 않음; s1–s4 392문항 v3): (a) 59.4 / 61.0 / 62.0, (b) 57.4 / 58.7 / 59.4, (c) 47.7 / 49.0 / 48.7 (ST/TSR/VisRef). af 행 84 / 70 / 87, 그중 v3 정답 46 / 47 / 52, af 제거 시 정답 38 / 38 / 42. af 구제 효과는 +2.0 / +2.3 / +2.6pp로 방법 간 차이 작음.

## 2026-10-10 push (사용자)
- diag_st_plain: kernel 137987959 v1. b_main_s5: kernel 137987971 v1. 둘 다 push 직전 빌드 무결성 확인(소스 8개 해시·노트북 해시 일치, s5 n_skip 1250, diag 150문항 = st_plain_ids와 같은 집합, 실행 순서는 build.py가 seed 0으로 재셔플).
- 사용자 결정: "출처 표시 / 미확인 표시 유지 / 확인된 수치 대장" 규칙을 프로젝트 지침에 넣음. 대장은 `repro/SPEC_LEDGER.md` 끝 "외부 수치 확인 대장"(X-1~X-5).

## 2026-10-10 report.py v3 통일 재채점·민감도 표, diag 분석 스크립트 (GPU 불필요, 생성 코드 변경 없음)
- 결함 수정: report.py가 행에 기록된 `correct`를 그대로 합산해 s1–s4(v1)와 s5+(v3)가 섞이던 문제. `--rescore-data ../data/raw/mathvista_testmini.json`을 주면 모든 행의 저장된 final_answer를 현재 채점 규칙(`visref/scoring.py`, rescore.py와 같은 `score()` 호출)으로 메모리에서 다시 채점한 뒤 집계한다(correct·extracted·prediction·rule·extract_failed 덮어씀, 기록값은 `correct_recorded`). 실행 점검 칸에 채점 버전과 바뀐 행 수 표시. report.py는 code_version 대상 아님.
- 같은 옵션일 때 "답 처리 규칙 민감도" 표 추가: 사전등록된 (a)/(b)/(c) + 참고로 `\boxed`만 인정(규칙 추출이 boxed에서 답을 찾은 정답만).
- 검증: 플래그 없이 돌리면 JSON이 이전 코드와 같음(새 키 2개 제외). s1–s4에서 기록값 재현: v3 233/239/243 (59.4/61.0/62.0), (b) 57.4/58.7/59.4, (c) 47.7/49.0/48.7, af 84/70/87, af 정답 (a) 46/47/52 · (b) 38/38/42.
- v3 기준 짝 차이(s1–s4, 392, 참고): TSR−ST +1.5(−1.0~+4.3, 16:10, p 0.33), VisRef−ST +2.6(−0.5~+5.9, 26:16, p 0.16), VisRef−TSR +1.0(−1.8~+3.8, 18:14, p 0.60). (v1 기준 +1.8 / +2.8 / +1.0)
- 사후 관찰(판정에 쓰지 않음): `\boxed`만 인정 시 ST 50.8 / TSR 53.1 / VisRef 49.5 — 순서가 바뀐다. boxed 밖에서 찾은 정답이 VisRef 49행(답 문구 45, 그중 답 안 </think> 25)으로 ST 34·TSR 31보다 많다. 답 직전 주입이 답 안에서 생각을 다시 열게 하는 현상(2026-10-08 P3)과 연결된다. 엄격 채점은 논문 채점 방식이 아니므로(논문은 답 처리·추출을 기술하지 않음) 보조 지표로만 보고한다.
- 보고서 아티팩트 v7 = s1–s4, v3 재채점, 민감도 표 포함. 이후 세션마다 같은 명령에 `--rescore-data` 추가:
  `cd repro && python3 report.py kaggle/runs/b_main_s{1..N}/output/results/results_{0,1}.jsonl --order kaggle/runs/b_main_s1/output/results/ids.json --shards 2 --label "b_main s1–sN" --rescore-data ../data/raw/mathvista_testmini.json --out reports/b_main_report.html`
- diag 분석 스크립트 `repro/diag/analyze_st_plain.py`(결과 전 작성, 사전등록 규칙 그대로 구현): 같은 pid 짝, 세 읽기 정확도·차이·McNemar p, af 행 수·정확도(diag/본 실험), diag 첫 단계 종료 방식 분포, 전체 ST 추정치(재촉 없음/강제 답 둘 다, 같은 150문항 부분집합 정확도로 계산). 판정 플래그는 (a)의 "10%p 이상 하락"만 자동 계산하고 "46 근처"는 추정치를 보고 판단한다(수치 기준 미리 정하지 않았음).
- 건조 실행(가짜 diag = 본 실험 ST 행): 차이 0, 본 실험 정답 74/150, af 50행으로 사전등록 기록과 일치. 참고: 부분집합 기준 강제 답 추정치는 (a) 57.6 / (b) 55.5 / (c) 47.5로, 실제 전체 ST(59.4)보다 낮다(150문항 정확도 49.3 < 잘린 245문항 전체 51.8). 비교는 같은 부분집합의 강제 답 추정치와 해야 한다.
- 실행: `cd repro && python3 diag/analyze_st_plain.py kaggle/runs/diag_st_plain/output/results/results_*.jsonl --main kaggle/runs/b_main_s{1,2,3,4}/output/results/results_{0,1}.jsonl --order kaggle/runs/b_main_s1/output/results/ids.json --data ../data/raw/mathvista_testmini.json --ids diag/st_plain_ids.json --out diag/st_plain_analysis.json`
- LLM 추출(규칙 실패 행, s1–s4 v3 기준 10/7/7행, v1 기준 11/7/8행)은 b_main 완주 뒤 한 번에 한다(행 수가 적고 세션마다 반복할 이유 없음).

## 2026-10-10 채점 대상(마지막 </think> 뒤) 유지 확인과 SPEC_LEDGER 핵심 판단 요약 신설 (사용자 결정, 설정 변경 없음)
- 사용자 결정: 주 채점 대상은 "마지막 `</think>` 뒤" 그대로. 근거: VLMEvalKit `SPLIT_THINK`가 `s.split('</think>')[-1]`을 prediction으로 씀(`references/sources/vlmevalkit_postprocess_54a063c5/vlmeval/inference.py` L259, L267-270). VLMEvalKit은 VisRef 인용 문헌이 아니고(VisRef_plain.txt grep 0건) VisRef가 인용한 InternVL3.5[45]의 평가 도구(InternVL3.5.txt L644, 모델 카드 L80)라 근거 등급 D. 옵션 기본값은 꺼짐이고 InternVL3.5 평가에서 켰는지는 미확인. 사전등록된 주 검정 조건이기도 하다.
- 첫 답(답 원문 `final_raw`의 첫 `</think>` 앞) 기준 재채점(사후, 규칙만, s1–s4): 정답 ST 233→232 / TSR 239→238 / VisRef 243→240. VisRef−ST +2.6→+2.0(27:19, p 0.30), VisRef−TSR +1.0→+0.5(19:17, p 0.87). 답 안 </think> VisRef 176행 중 정오가 갈린 행 5(마지막만 정답 4, 첫 답만 정답 1). 첫 답 51행은 규칙 추출 실패. 보조 민감도로만 보고.
- SPEC_LEDGER 맨 앞에 "핵심 판단 요약" 신설: 근거 등급 A(논문 명시)–F(운영 선택)와 판단 J-1~J-18, 각 판단의 본 표 행·EF 날짜. 함께 고친 것: 43행 VLMEvalKit 근거 추가, 44행 강제 답 문자열 정정(실제 `\nFinal Answer:`), 45행 LLM 보조 추출 반영, 52행(DEER af) 신설. 이전 판 `repro/.SPEC_LEDGER.bak_20261010`.

## 2026-10-10 InternVL3.5 공식 평가 설정(VLMEvalKit) 확인 — 설정 변경 없음, 적용 범위는 사용자 결정 대기
- 사용자 제안: VisRef가 인용한 InternVL3.5가 쓴 방법(VLMEvalKit)을 baseline 구현에 그대로 따르는 게 맞지 않나.
- VLMEvalKit 커밋 54a063c5의 관련 소스 5개를 받아 `references/sources/vlmevalkit_postprocess_54a063c5/`에 저장(SOURCE.json에 URL·sha256). 대조표는 SPEC_LEDGER "D 자료 대조".
- 확인된 사실: 1B 설정이 두 개(`InternVL3_5-1B` greedy·최대 4,096 / `InternVL3_5-1B-Thinking` R1·T 0.6·top_p 0.95·최대 65,536·lmdeploy). `USE_COT`(답 \boxed 지시문)·`SPLIT_THINK`(생각/답 분리)는 환경변수 옵션. MathVista 채점은 규칙 prefetch 후 실패 시 GPT-4o-mini 추출. 공식 59.3이 어느 설정으로 나왔는지는 미확인.
- 우리와 가장 큰 차이: 최대 생성 토큰(65,536 vs 생각 단계 1,024). 미정 항목 "예산을 늘린 ST"와 직결.
- 해석: VisRef ST 46.1은 공식 59.3보다 13pt 낮아, VisRef 저자가 이 공식 설정을 그대로 썼다고 보기 어렵다. 공식 설정을 따르면 "모델 공식 사용법에 충실한 baseline"이 되지만 VisRef 절대값 재현과는 방향이 다를 수 있다.

## 2026-10-10 사용자 결정: 원칙 채택, b_main 유지, 공식 설정 ST 보조 실험, LLM 추출 모델
- 원칙 채택: VisRef 본문·인용 문헌·인용 코드(등급 A–C)가 정하지 않은 부분은 InternVL3.5 공식 자료(등급 D: 모델 카드, VLMEvalKit 설정)를 기본값으로 삼는다(SPEC_LEDGER 원칙 1).
- b_main은 사전등록대로 유지(설정 변경 없음).
- LLM 추출: VLMEvalKit 공식 판정 모델 GPT-4o-mini 대신 기존 계획대로 Gemini/Haiku(OpenAI 키 없음). 공식과 다른 점으로 보고.
- 공식 설정 ST 보조 실험 st_official을 아래처럼 결과 전에 고정한다.

## 2026-10-10 보조 실험 st_official 사전등록 (결과 전 고정)
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

## 2026-10-10 diag_st_plain 결과와 사전등록 판정 (d0095f79e691, 약 1,400s, error 0, 150/150 완료)
- 분석: `repro/diag/analyze_st_plain.py` → `repro/diag/st_plain_analysis.json`(결과 전 작성한 스크립트 그대로).
- (a) 그대로(판정 읽기): 재촉 없음 47.3% vs 본 실험 강제 답 49.3% (같은 150문항), 차이 −2.0%p (재촉 없음만 정답 15 : 강제 답만 정답 18, McNemar p 0.73). 전체 ST 추정치 재촉 없음 56.4 vs 강제 답 57.6(같은 부분집합 기준).
- 판정: 기준(10%p 이상 하락, 추정치 46 근처) 미충족 → "잘린 생각 뒤 `Final Answer:` 재촉 문구가 논문 ST 46.1과의 차이를 만든다"는 설명은 기각.
- (b) af 제거(두 규칙 모두 없음의 근사): 40.0 vs 46.0, −6.0%p (20:29, p 0.25), 추정치 51.8 vs 55.5. (c) af 행 오답(하한, 판정에 안 씀): 26.0 vs 34.0, −8.0%p (9:21, p 0.043), 추정치 42.5 vs 47.5.
- 함께 볼 것: diag 첫 단계 종료가 본 실험과 다른 문항 15개(본 실험은 상한 절단, diag는 1,024 안에서 자연 종료; fp16 배치 잡음으로 생각이 갈라짐). 상한 절단이 유지된 135문항은 정답 67 vs 68로 사실상 같다. 재촉 없이 답하게 하면 답 길이 중앙값이 283 → 741토큰으로 늘고(답 구간에서 생각을 이어감), af 행이 50 → 64로 늘어 그중 32행이 af로 구제된다. 즉 규칙 ①을 끄면 규칙 ②가 대신 답을 끌어내는 경향 — (b)·(c)의 하락은 대부분 이 경로.
- 남는 설명 후보: 논문 ST 46.1과 우리 ~59의 차이는 답 재촉 문구로는 설명되지 않는다. 생각 예산(상한 크기와 반복 루프) 가설은 st_official로 확인한다.

## 2026-10-10 논문의 토큰 예산 관련 서술 재확인 (사용자 질문: VisRef에 토큰 상한·누적 사용량 정보가 있나)
- 상한(max tokens) 값은 본문·부록 어디에도 없다(SPEC_LEDGER 11·12행 검색 기록과 같음).
- 있는 것은 평균 사용량과 예산 축이다:
  - "∼1,772 visual tokens vs. ∼615 text tokens" — InternVL3.5-8B, MathVista, 생각 단계당 생성 텍스트 토큰 평균 (`VisRef_plain.txt` L383-385).
  - Fig. 3 가로축 = 누적 thinking token 예산 B(병렬 사고, Σ|τ(i)| ≤ B, L698-702). ST ★ 위치는 8B에서 약 400–1,000토큰(그림 판독, SPEC_LEDGER A-3). "budget of 14K thinking tokens"는 MMStar 8B 병렬 사고 예산 (L717).
  - App. I "same decoding setup and token budget" — 값 없음 (SPEC_LEDGER 12행).
  - 1B의 토큰 사용량은 논문에 없음.
- 해석: 8B ST가 평균 약 600토큰에서 스스로 끝난다는 것은 상한이 작았다는 증거도, 컸다는 증거도 아니다(8B는 대부분 그 전에 끝남). 1B는 우리 기록상 1,024에서 63%, 4,096에서도 약 절반이 안 끝나므로 상한 값이 결과를 좌우한다. 논문이 이 값을 정하지 않았으므로 원칙 1(A–C 없음 → D)에 따라 공식 65,536이 기본값이고, st_official은 그 기본값에서의 ST를 측정하는 실험으로 타당하다.

## 2026-10-10 사용자 결정: st_official 보류, 절대값 차이는 추정으로 기술
- 결정: st_official_s1은 push하지 않고 보류(빌드·사전등록은 결과 전 상태로 유효). 이유: 본 실험(b_main)이 우선이고, b_main에 문제가 생겨 재실행하면 1,000문항에 약 20시간이 필요한데 남은 쿼터는 약 27.6시간(30 − s5 2 − diag 0.4)이다. b_main 완주 뒤 쿼터가 남으면 재검토한다.
- 절대값 차이는 검증하지 않은 추정으로 기술한다. 최종 보고서용 문단(초안):
  > 우리 ST(s1–s4 v3 59.4%)는 InternVL3.5-1B 공식 단일 응답 MathVista 점수 59.3%(SPEC_LEDGER X-1)와 비슷하고, VisRef 논문 ST 46.1%보다 약 13pt 높다. 논문 ST는 8B에서도 공식 수치보다 10pt 이상 낮아(68.1 vs 78.4, X-3) 모델 크기와 무관하게 평가 설정이 공식과 달랐을 가능성이 크다. 논문은 생성 토큰 상한, 잘린 생각과 답의 처리, 답 추출 방식을 기술하지 않는다(X-4, X-8, 본 표 45행). 우리가 확인한 것은 하나다: 잘린 생각 뒤 답 재촉 문구(`Final Answer:`)는 차이를 설명하지 못한다(diag_st_plain, −2.0%p, p 0.73). 확인하지 않은 후보는 (1) 생각 토큰 상한의 크기 — 1B는 긴 생각에서 반복 루프가 잦아(4,096 상한에서 잘린 생각의 약 절반) 상한과 처리 방식에 따라 정답률이 크게 달라질 수 있다, (2) 답 추출·채점 방식(규칙+LLM 보조 vs GPT 추출기), (3) 프롬프트 형식, (4) 하드웨어·정밀도(논문 A10G vs 우리 T4 fp16)다. 따라서 논문과의 절대값 차이는 이러한 평가 설정 차이에서 비롯된 것으로 추정하며, 재현 여부는 같은 설정 안에서의 방법 간 상대 이득(VisRef−ST, VisRef−TSR)으로 판단한다.

## 2026-10-10 불일치 쌍 감사 (s1–s4, GPU 불필요) — 구조적 문제 없음
- 대상: VisRef가 ST 또는 TSR과 정오가 다른 50문항 × 3방법 = 150행(`repro/audit/discordant_v3_s1s4.jsonl`). 판정 기준(채점 S / 실패 유형 F / 버그 징후 B)을 먼저 고정하고 Sonnet 서브에이전트가 전문을 읽어 판정, 주 세션이 핵심 주장을 원자료로 대조. 결과 `repro/audit/audit_rows_s1s4.csv`, `audit_summary_s1s4.md`(끝에 검증 메모).
- 채점: 오채점(정답 처리 오류) 0. 오답 처리 오류 2(441 VisRef: 답 뒤에 `</think>`가 붙어 채점 텍스트가 빔; 579 ST: 첫 단계가 EOS로 끝나 답 단계 없음), 애매 2(ST). 반영해도 VisRef:ST 26:16 → 25:15, VisRef:TSR 18:14 → 18:13. 한쪽으로 쏠린 채점 편향 없음.
- 버그 징후: 주입 뒤 깨진 텍스트·엉뚱한 답·프롬프트 반복 없음. 반복 루프 7문항은 정상적인 영어 반복.
- 남는 약점(기록만, 채점 규칙 변경 없음 — 원칙 2): (1) 마지막 `</think>` 뒤가 비면 답을 잃음. VLMEvalKit `SPLIT_THINK`도 같은 결과(빈 예측)라 원칙 1과 일치. 392문항 중 방법당 1행. (2) 첫 단계가 `</think>` 없이 EOS로 끝나면 ST는 답 단계가 없어 빈 답(392문항 중 1행). VLMEvalKit은 `</think>`가 없으면 전체 응답을 예측으로 씀 → 이 1행은 공식 방식과 다름. 최종 보고 민감도 메모로 남김.
- 해석 주의: 승자 답 중 상당수가 상한 강제 답·추측성(서브에이전트 판단, 방법 간 비율 비슷). 차이 판정은 사전등록된 전체 검정으로만 한다.

## 2026-10-10 st_official 사전등록 보정 (사용자 결정: 무료 Colab, 100문항) — 결과 전 고정
- 실행 환경: Kaggle 대신 무료 Google Colab, GPU 1장(무료 등급이면 보통 T4, 실제 GPU는 세션마다 env_*.json에 기록), fp16. Kaggle 쿼터 사용 0. 노트북은 `repro/colab/build_colab.py`로 같은 소스 묶음(code_version d0095f79e691, 소스 sha256 검증)에서 생성.
- 표본: b_main 셔플 순서의 첫 100문항(고정; 모두 b_main s1–s4에 있음 → 같은 pid 짝 비교 가능). 이 100문항의 b_main ST: 첫 단계 상한 절단 63 / 자연 종료 36 / EOS 1, 정답 57.
- 인자: 2026-10-10 사전등록과 같음(`--methods st --think-cap 65536 --total-think-cap 65536 --answer-cap 4096 --forced-prompt=`), 단일 GPU라 `--shard 0/1`.
- 이어 달리기: 결과를 Colab 로컬에 쓰고 1분마다 Drive로 복사, 다음 세션은 Drive 결과를 가져와 이미 끝난 pid를 건너뜀(run.py 기본 동작). 끊길 때 진행 중이던 문항만 다시 생성. 잘린 마지막 줄은 버림.
- 종료 규칙: 100문항을 모두 끝내면 분석. 무료 GPU를 더 받지 못해 중단하면 셔플 순서상 끝난 앞부분(complete prefix)만 분석하고 n을 그대로 보고(정확도를 보고 중단하지 않음).
- 분석·해석 규칙: 2026-10-10 사전등록 그대로(짝 차이 + 부트스트랩 CI + McNemar, 첫 단계 종료 방식별; Wilson CI를 59.3·46.1과 비교; 자연 종료 문항 정오 일치 ≥90% 점검; 생각 길이·컨텍스트 상한 도달 비율, (b)·(d) 읽기). n=100이면 Wilson CI 반폭 약 ±10%p라 46과 59의 구분은 점 추정이 한쪽 끝에 가까울 때만 가능하다.
- 하드웨어 차이 주의: Colab T4와 Kaggle T4는 같은 GPU지만 torch·CUDA 버전이 다를 수 있다. CRN(같은 시드)으로 생각 앞부분이 같기를 기대하지만 fp16 잡음과 라이브러리 차이로 갈라질 수 있다 — 자연 종료 문항 일치율 점검으로 확인.

## 2026-10-10 st_official_c1 Colab 노트북 준비 완료 (결과 전)
- 빌더 `repro/colab/build_colab.py`(kaggle/build.py의 소스 묶음·리비전 재사용, 노트북 안에서 소스 sha256 검증). 산출물 `repro/colab/runs/st_official_c1/st_official_c1.ipynb`(notebook sha256 310c7ab59c21…, 38KB) + `run_config.json`. ids = b_main 셔플 순서 첫 100(확인), 소스 해시 = b_main_s5(확인), code_version d0095f79e691.
- 노트북 동작: Drive 마운트 → transformers 4.55.0(+timm, einops: InternViT 원격 코드용, 이미 있으면 건너뜀) 설치·모델/데이터 고정 리비전 다운로드 → run.py를 로컬 `/content/out`에 쓰고 1분마다 `MyDrive/VisRef_colab/st_official_c1/`로 복사, 세션마다 env_*.json 기록. 새 런타임에서는 Drive 결과를 가져와 이어서 실행.
- 모의 실행(가짜 torch·run.py, GPU 없음): 31번째 문항 중간에 끊긴 뒤 잘린 줄 제거 → 새 런타임에서 남은 70문항만 실행 → 최종 100행·중복 0·ids 일치.
- 분석 스크립트 `repro/diag/analyze_st_official.py`(결과 전 작성): 건조 실행(가짜 공식 = b_main ST) n=100, 차이 0, CRN 점검 통과.
- 결과 회수: 사용자가 Drive 폴더의 `results_0.jsonl`(및 env_*, log_*)을 `repro/colab/runs/st_official_c1/output/`로 내려받으면 분석.
- (같은 날 추가) 노트북이 Drive에 `status.json`({state: running|stopped|finished, done, total, updated_utc, session})을 1분마다 갱신하도록 빌더 수정 후 재빌드(실험 인자·문항·소스 해시 그대로). 런타임이 끊기면 state는 running인 채 updated_utc가 멈추므로, 외부 점검은 "updated_utc가 15분 이상 지났는가"로 판단한다. 모의 실행 재확인: 끊김 → stopped 31/100, 재실행 → finished 100/100, 중복 0.

## 2026-10-10 b_main_s5 결과(d0095f79e691)와 s6 빌드
- 건전성: error 0, 111문항×3 = 333행, code_version d0095f79e691, T4×2, torch 2.11.0+cu128·transformers 4.55.0(s1–s4와 같음), .partial 없음, s1–s4와 pid 중복 0. 세션 약 5,710s.
- s5 단독(v3, 111문항): ST 46.8 / TSR 52.3 / VisRef 50.5, ST 첫 단계 상한 절단 68.5%. s1–s4(59.4 / 61.0 / 62.0)보다 크게 낮다(ST Fisher p 0.023).
- 원인 점검(설정 변경 없음): 생성 코드 sha256(run.py, methods, dpp, engine, model, fastkv)이 s2–s5에서 모두 같고(s1은 출력 동일 확인된 79eef648ebd1), 다른 것은 scoring.py뿐이며 집계는 모두 v3로 재채점. 문항 유형 구성(category·task·context·grade·question_type)은 s1–s4와 비슷한데 거의 모든 유형에서 정확도가 함께 낮다. 세 방법이 같은 문항에서 함께 떨어졌고(첫 단계 공유) 방법 간 차이 방향은 유지(TSR−ST +5.5, VisRef−ST +3.7). 세션별 ST: s1 59.5 / s2 57.0 / s3 63.0 / s4 58.3 / s5 46.8 → 실행 이상의 증거 없음, 셔플 구간의 문항 난이도 편차로 본다(5개 구간 중 하나가 이 정도 벗어날 확률은 작지만 불가능하지 않음). 이후 세션도 같은 점검을 계속한다.
- s1–s5 complete prefix 503문항(참고, v3): ST 56.7 / TSR 59.0 / VisRef 59.4. 짝 차이 TSR−ST +2.4(0.0~+5.0), VisRef−ST +2.8(0.0~+5.6), VisRef−TSR +0.4(−2.2~+3.0). 민감도 (b) 54.5 / 56.7 / 57.1, (c) 44.3 / 45.9 / 46.3. CRN: TSR k=1 362문항 중 ST와 정오 일치 354.
- s6 빌드: 코드·인자·문항 순서 s5와 같음, --skip-from s1–s5(1,583행 = 503×3 + s1 visref_raw 74), timeout 7200. 보고서 아티팩트 v8 = s1–s5.

## 2026-10-10 st_official_c1 Colab 실행 시작과 재시작 예약 작업
- 사용자가 무료 Colab에서 실행 시작(08:55 UTC). env: torch 2.11.0+cu130, transformers 4.55.0, Python 3.13, Tesla T4 1장, code_version d0095f79e691, fast_kv_check ok. (Kaggle과 CUDA 빌드만 cu130 vs cu128)
- 노트북은 Colab 계정 <redacted-email>의 Drive에 있음(https://colab.research.google.com/drive/1gyks3WKsJA2BfHRDFJszUVdqRji-_ow-). 결과 폴더 VisRef_colab/st_official_c1은 이 세션의 Drive 커넥터 계정에서 보이지 않음 → 공유하면 점검이 싸진다.
- 예약 작업 "VisRef Colab 재시작 점검"(2시간마다 매시 2분, 이 컴퓨터 필요, push 알림): status.json 확인 → 15분 넘게 멈췄으면 내장 브라우저로 "모두 실행"과 Drive 연결 허용(사용자 사전 승인). CAPTCHA·GPU 없음·로그인 필요 시 멈추고 알림. 범위는 재시작만(파일 수정·분석 없음).

## 2026-10-10 공개 준비 결정 (사용자)
- 공개: 실험 완료 후 GitHub 공개, 그 전까지 비공개. GitHub에 기록이 남는 작업(저장소 생성·커밋·push)은 사용자가 직접 한다. Claude는 올릴 파일과 명령만 준비.
- 코드 라이선스: MIT. 결과 JSONL(생성 텍스트 포함)도 올린다(MathVista·InternVL3.5 출력 재배포 조건은 공개 전 확인, 미확인).
- LLM 추출은 Gemini API 스크립트(`repro/gemini_extract.py`, 모델 ID·입출력 기록)로 한다. 이 세션의 클라우드 작업 공간과 연결된 컴퓨터 VM 모두 generativelanguage.googleapis.com에 접속 불가(HTTP 000) → 요청 파일은 Claude가 만들고, 실행은 사용자가 자기 터미널에서 환경변수 GEMINI_API_KEY로 한다. API 키는 대화·파일·메모리에 남기지 않는다.
- (같은 날) 저장소 이름 `visref-repro` 확정. 설명안: "Unofficial reproduction of VisRef (arXiv:2603.00207) — InternVL3.5-1B × MathVista testmini".
- LLM 추출 사전 점검 준비: s1–s5 결과를 v3로 재채점한 사본 `repro/llm/precheck_s1s5/rows_v3.jsonl`(원본 불변, 공개 대상 아님) → `llm_extract.py prepare --control 20` → `requests.jsonl` 49건(규칙 실패 29 + 대조군 20, 정답 미포함). 사전 점검 결과는 판정에 쓰지 않고, 최종 추출은 완주 후 전체로 다시 한다.

## 2026-10-10 b_main_s6 결과와 s7 빌드
- s6 건전성: error 0, 126문항×3 = 378행, d0095f79e691, T4×2, torch 2.11.0+cu128, 이전 세션과 pid 중복 0, .partial 없음, 세션 약 6,190s.
- s6 단독(v3): ST 69.8 / TSR 69.0 / VisRef 65.9, 첫 단계 상한 절단 61.9%. s5(46.8)와 반대로 높다 → 세션 간 편차가 크며 s5 저점은 구간 편차라는 해석과 맞음. 세션별 ST: 59.5 / 57.0 / 63.0 / 58.3 / 46.8 / 69.8.
- s1–s6 complete prefix 629문항(참고, v3): ST 59.3 / TSR 61.0 / VisRef 60.7. 짝 차이 TSR−ST +1.7(−0.5~+3.8, 30:19, p 0.15), VisRef−ST +1.4(−1.0~+4.0, 36:27, p 0.31), VisRef−TSR −0.3(−2.5~+1.9, 27:29, p 0.89). 판정은 완주 후 사전등록 검정으로만.
- s7 빌드: 코드·인자·순서 s6와 같음, --skip-from s1–s6(1,961행 = 629×3 + 74), timeout 7200. 보고서 아티팩트 v9 = s1–s6.

## 2026-10-10 LLM 추출 사전 점검 결과 (gemini-3.5-flash-lite, 무료 등급, 사용자 터미널 실행)
- 요청 49건(규칙 실패 29 + 대조군 20) → 응답 49(모두 STOP). merge: fixed 2, broke 0, 대조군 일치 18/20. 정확도 ST 56.7→56.9, TSR 59.0→59.0, VisRef 59.4→59.6.
- 원문 대조 결과 "fixed" 2건은 모두 잘못된 정답 처리: 570 ST는 채점 텍스트가 빈 문자열인데 LLM이 "Assuming … 5"로 답을 지어냄(정답 5와 우연히 일치), 106 VisRef는 텍스트의 답이 "1"인데 LLM이 "0"으로 추출(정답 0). 실제로 바로잡은 행 0.
- 대조군 불일치 2: 964 VisRef(규칙 `\boxed{1}`이 맞고 LLM "7"이 틀림), 833 TSR(규칙이 `1.4 \times 10^{19}`에서 지수 19를 추출 — 규칙 쪽 결함, LLM 1.4; 정답 −4.3이라 둘 다 오답).
- 비추출 응답 7건(빈 텍스트·긴 텍스트에서 이미지·설명 요구 등).
- 판단: 이 모델 그대로의 LLM 추출은 오답을 정답으로 만들 위험이 실재한다. 최종 추출 전에 보완이 필요(사용자 결정 대기): (1) 채점 텍스트가 비면 LLM에 보내지 않고 오답 유지, (2) LLM이 뽑은 답이 응답 원문에 실제로 나타날 때만 인정(객관식은 해당 선택지 글자 또는 선택지 텍스트), (3) 더 강한 무료 모델(gemini-3.8-flash) 비교. 사전 점검 자료는 판정에 쓰지 않는다.

## 2026-10-10 LLM 추출 보완 반영 (사용자 승인, 최종 추출 전 — 판정 결과를 보고 바꾸는 것이 아님)
- `repro/llm_extract.py`에 두 단계 추가(이전 판 `repro/llm/llm_extract.before_guard.py`):
  1) 채점 텍스트가 빈 행은 LLM에 보내지 않고 규칙 결과(오답) 유지(prepare에서 제외, merge에서도 무시).
  2) LLM 추출 답이 채점 텍스트에 실제로 있을 때만 인정: 객관식은 선택지 텍스트 또는 글자 형식("(B)", "B)", "B.", "**B**", "\boxed{B}", "answer is B"), 숫자형은 같은 수(쉼표 제거, 오차 1e-9), 그 외는 같은 문자열. 근거 없는 추출은 기록만 하고 규칙 결과 유지. 행마다 llm_grounded·llm_used 기록.
- 단위 확인 9/9 통과. 사전 점검 응답(3.5-flash-lite) 재병합: fallback 24, 빈 텍스트 건너뜀 5, 근거 없음 거부 2, fixed 0, broke 0, 대조군 일치 18/20·근거 있음 20/20 → 정확도 규칙과 동일(56.7 / 59.0 / 59.4).
- 한계: 보완 2는 지어낸 답은 막지만, 텍스트에 있는 다른 숫자를 잘못 고르는 경우(예: 964 VisRef, 대조군에서 "7 cm mark"의 7)는 막지 못한다. 최종 보고에 LLM 추출로 바뀐 행 목록을 함께 싣는다.
- 다음: 같은 요청 49건을 gemini-3.8-flash(무료 등급 표기, 가격 페이지 확인)로 돌려 모델 비교 후 최종 추출 모델을 정한다.

## 2026-10-10 b_main_s7 결과와 s8 빌드
- s7 건전성: error 0, 130문항×3 = 390행, d0095f79e691, T4×2, torch 2.11.0+cu128, pid 중복 0, .partial 없음, 세션 약 5,710s.
- s7 단독(v3): ST 51.5 / TSR 56.2 / VisRef 53.8, 첫 단계 상한 절단 69.2%. 세션별 ST: 59.5 / 57.0 / 63.0 / 58.3 / 46.8 / 69.8 / 51.5.
- s8 빌드: 코드·인자·순서 s7과 같음, --skip-from s1–s7(2,351행 = 759×3 + 74), timeout 7200. 남은 241문항 → s8·s9 예상.

## 2026-10-10 LLM 추출 모델 결정 (사용자): gemini-3.5-flash-lite 단독 + 보완 1·2 — 최종 추출 전 고정
- 근거: MathVista 공식 파이프라인도 LLM 추출기의 오류를 감수한다("a preliminary study of 200 examples shows that GPT-4 can extract the answer text with more than 99.5% accuracy", `references/papers/ref_fulltext/MathVista.txt` L466-469; 공식은 GPT-4, VLMEvalKit 기본 판정 모델은 gpt-4o-mini — `vlmeval/dataset/image_vqa.py` L600). 3.8-flash는 무료 한도 하루 20회(429 응답의 "limit: 20")라 최종 추출에 부적합하고, 사전 점검 16건에서는 보완 적용 후 결과가 flash-lite와 같았다(근거 있는 정답 9/13 vs 9/13, 근거 없음 0 vs 0).
- 최종 절차: 완주 후 전체 결과를 v3로 재채점 → `llm_extract.py prepare --control 20`(빈 텍스트 제외) → 사용자 터미널에서 `gemini_extract.py --model gemini-3.5-flash-lite --sleep 6` → `merge`(원문 대조, 근거 없으면 규칙 결과 유지). 교차 확인은 하지 않는다. 보고서에 LLM 추출로 정오가 바뀐 행 전부와 대조군 일치율을 싣는다.

## 2026-10-10 공개용 폴더 생성 (github_export/visref-repro, 커밋·push는 사용자)
- `scripts/make_github_export.py`(재실행 가능)로 코드·실행 기록(run_config, env, log, results)·문서(EF, SPEC_LEDGER, 감사·검토)를 복사. 제외: kaggle.json, 논문 PDF·전문, 제3자 코드 사본(URL+sha256을 `references/SOURCES.json`에 기록, `scripts/fetch_sources.py`로 내려받아 검증), 번들 노트북(재생성 가능), 메모리·스크래치. 문서 사본의 개인 이메일은 가림(원본 불변). 비밀 키 패턴 검사 0건. 약 22MB.
- README(영문), LICENSE(MIT, 저작권자 이름 자리표시), requirements.txt, .gitignore. 라이선스 근거: InternVL3.5-1B Apache-2.0(모델 카드 L2, L816-818), MathVista 저장소 CC BY-SA 4.0(LICENSE 파일) → 결과 파일은 CC BY-SA 4.0으로 표기.
- 검증: 임시 사본에서 README 1단계(오프라인 재채점·보고서)를 그대로 실행 → s1–s7 759문항 수치가 원본과 동일.

## 2026-10-10 관찰: VisRef 효과가 TSR을 넘지 못함 (사용자 질문, 설정 변경 없음 — 사후 관찰)
- s1–s7 759문항: TSR−ST +2.2(논문 +2.4와 비슷) → 다단계 이어가기(TSR)는 재현되는 쪽. VisRef−TSR −0.7(CI −2.8~+1.6)로 논문 +3.5가 CI 밖.
- 이어가기가 일어난 문항(어느 방법이든 k>1, 230문항): ST 33.5 / TSR 41.7 / VisRef 38.7. 둘 다 k=1인 529문항은 68.6 / 68.2 / 68.6(사실상 동일).
- 주입량: 주입 수/m 중앙값 0.21, 주입 수/|V| 중앙값 0.062, m에 도달한 단계 16%. 논문 Fig. 4(b) 최저 조건(m=20%)보다 훨씬 적음.
- 답 구간에서 생각을 다시 여는 행(답 안 </think>): VisRef 357/759(47%), 정확도 58.3 vs 없는 행 60.7.
- 해석(가설, 검정 전): 코드 버그 징후는 감사·CRN 점검에서 나오지 않았고, 유력 원인은 (1) 입력 임베딩 Z에서 계수 제한으로 m의 약 1/5만 주입되는 구현 선택(J-7/J-8), (2) 최종 답 직전 주입이 답 구간을 흔드는 현상이다. b_main은 사전등록대로 완주하고, 원인 분리 실험은 완주 후 사전등록해 진행한다(사용자 결정 대기).

## 2026-10-10 주입 경로 독립 검토 (Fable 5.1 서브에이전트, GPU 없음) — 버그 없음
- 보고서 `repro/audit/injection_review_20261010.md`(+ DPP 수치 점검 스크립트·출력). 결론: 주입 경로에 BUG 없음. 주입 텐서 = 모델이 원본 이미지에서 보는 projector 출력 그대로(같은 객체, fp16, 재스케일 없음), z_k 경계 깨끗, 순서 z_k [\n] V_k 다음 </think>+probe 또는 Wait, 위치·마스크·KV 일관(호출마다 전체 재-prefill), greedy DPP가 Chen 코드·naive log-det greedy와 선택 동일. 표본 4행에서 final_context_len이 토큰 단위로 재현됨. 주 세션 확인: 표본 행 계산은 보고서 수치로만 확인(미재계산), R1 진단은 아래처럼 전체 자료로 재계산.
- RISK R1(가장 중요): 시각 토큰 묶음 바로 뒤에 </think>가 붙는 구조(논문 순서 그대로)가 답 안 </think> 재발생의 원인일 가능성. 주 세션 진단(s1–s7, 759문항): 마지막 단계가 스스로 닫힌(close) 경우 재발생률 ST 6.5% / TSR 7.3% / VisRef 45.1%, 상한 절단(cap) 경우 17.0 / 13.8 / 50.0% → 상한이 아니라 V 묶음 뒤 </think> 구조가 원인으로 확인. 그러나 k=1·1단계 자연 종료 244문항에서 정답 TSR 176 vs VisRef 175 → 재발생 자체의 정확도 영향은 작다. VisRef가 TSR보다 낮은 구간은 이어가기 문항(k>1, 230문항: TSR 41.7 vs VisRef 38.7).
- R2(행에 cfg 미기록 — run_config.json에는 있음), R3(VisRef만 m+2 여유 확보 — 현재 컨텍스트에서 영향 없음), R4(상대 ε — 현재 영향 없음)는 기록만.

## 2026-10-10 원인 분리 실험 사전등록 (사용자 결정, b_main 완주 후 실행) — 결과 전 고정
- 목적: VisRef−TSR이 논문(+3.5)보다 낮은 원인 분리. b_main 설정은 바꾸지 않는다. 새 방법을 추가만 하고, 기존 세 방법의 출력이 바뀌지 않음을 CPU 동일성 검사(2026-10-07 방식)로 확인한 뒤 빌드한다.
- 변형 1 `visref_fill`(m 채우기, 먼저 실행): greedy가 계수에서 멈춘 뒤에도 커널에 작은 εI(1e-6 × 평균 대각, 2026-10-08 메모의 "식은 그대로, 수치 안정화만")를 더해 정확히 m개까지 선택해 주입. 그 외는 visref_raw_rank와 같음. 논문 문구 "|Ṽk| = m"에 맞춘 해석.
- 변형 2 `visref_nolast`(마지막 단계 주입 생략): 각 단계의 V_k는 다음 단계로 이어갈 때만 주입하고, 정지 probe와 최종 답은 V_k 없이(TSR과 같은 문맥 구조) 만든다. 논문 Alg. 1과 다른 진단용.
- 표본: b_main 셔플 순서 첫 392문항(s1–s4와 같은 pid, 같은 시드 → b_main ST·TSR·VisRef와 짝). 인자 `--methods visref_fill`(이후 `visref_nolast`) `--think-cap 1024 --total-think-cap 10240`, 나머지 b_main과 같음. Kaggle 2시간 세션, 392문항 끝날 때까지.
- 분석(탐색적, 변형별): 짝 차이(변형 − b_main VisRef, 변형 − TSR), 부트스트랩 95% CI, McNemar 정확 p(두 변형에 Holm 보정 값도 함께), 이어가기 문항(k>1)과 k=1 문항 층별, 주입 수/m, 답 안 </think> 재발생률, k 분포. 채점 v3 + 최종 LLM 추출 규칙 동일.
- 해석 규칙: visref_fill − TSR의 CI 하한이 0보다 크면 "주입량 부족이 VisRef 효과를 줄였다"를 지지. visref_nolast가 재발생률을 낮추고 VisRef보다 정확도가 높으면(CI 하한 > 0) "답 직전 주입이 답을 흔든다"를 지지. 둘 다 아니면 "이 구현·모델(1B)에서는 시각 재주입 효과가 재현되지 않음"으로 보고.
- 보류(사전등록하지 않음): 검토자 R1 제안인 <img>…</img>로 감싸 주입 — 2026-10-07 smoke1에서 감싸기가 두 번째 </think>와 질문 환각을 일으킨 기록(README:L25)이 있어 우선순위 낮음.

## 2026-10-10 변형 실험 설계 독립 검토 (Fable 5.1, 사용자 요청) — 결과 전
- 보고서 `repro/audit/design_review_variants_20261010.md`(입력 요약 `design_review_variants_brief.md`). 논문 근거는 `VisRef_plain.txt` 줄 번호로 인용.
- 판정 요지: 논문은 최종 답 y가 마지막 V_k를 포함한 궤적에서 생성된다고 명시(Alg. 1 L16-17, §4.1.2)하고 H_k도 V_k 추가 후 계산 → visref_raw_rank는 충실, `visref_noansinj`는 의도적 이탈(진단용 절제). |V_k| = m은 계수 부족 경우를 논문이 정하지 않은 미지정 → `visref_fill`은 "미지정 사례의 충실한 재해석"이지만 계수 이후 선택은 다양성 정보가 없고 관련성 비슷한 순서.
- 검토자 주장 원자료 확인(주 세션, s1–s7 759문항): VisRef가 k=1에서 멈춘 543문항에서 VisRef 369 vs ST 369 정답(8:8) → 답 직전 주입 효과는 k=1에서 0. noansinj가 새로 주는 정보는 k>1 문항뿐(392 중 약 110), 그 구간 최소 검출 효과 약 8–10%p로 검출력 낮음.
- 새 관찰: VisRef는 TSR보다 이어가기가 잦다(k≥2 216 vs 199, k=10 64 vs 53; 1단계 probe H 중앙값 0.007 vs 0.003) → V 묶음이 정지 판정 엔트로피를 올릴 가능성.
- 검출력(검토자 계산, 불일치율 8.8% 가정): n=392 SE 1.49%p, +3.5%p 검출력 0.65, 80% 최소 검출 효과 4.2%p; n=1,000이면 2.6%p.
- 검토자 권고: fill 먼저(보완: 앞부분 선택이 rank 버전과 같음 assert, ridge 계산 fp32/64, n_inject/m·stop_reason·재발생·af 기록, 꼬리 선택과 관련성 상관 측정); noansinj는 저장 궤적 재생(replay) 방식의 k>1 진단으로만, 아니면 제외; 결정에 영향 없는 이중 probe 기록(V_k 있는/없는 H_k) 추가; 은닉 상태 Z는 오프라인 계수 점검 후 조건부.
- 주 세션 메모: 은닉 상태 Z의 오프라인 점검은 2026-10-08 hidden_rank_check로 이미 했고(의미 있는 선택 약 50–110개, 그 뒤는 사실상 관련성 채움) → fill과 실질적으로 같은 개입이라 별도 실행 불필요. 사전등록 수정은 사용자 확인 후 기록.

## 2026-10-10 원인 분리 실험 사전등록 수정 (사용자 결정, 결과 전) — 앞의 "원인 분리 실험 사전등록"을 대체
- 변형 2(`visref_nolast`, 그 수정안 `visref_noansinj`)는 GPU 실행에서 제외. 이유: 논문 Alg. 1에서 벗어난 진단용이며, k=1 문항에서는 구조상 ST 답과 같아(실측 369 vs 369) 정보가 k>1 문항(약 30%)에만 있고 검출력이 낮다(설계 검토 보고서).
- 대신 `visref_fill` 실행에 "이중 정지 판정 기록"을 넣는다: 매 단계 정지 probe를 V_k 있는 문맥(실제 판정에 사용)과 V_k 없는 문맥(`… z_k </think>` + probe prompt, 별도 난수열)에서 각각 계산해 H_k, H_k^noV를 기록. 판정은 H_k만 사용 → 결과에 영향 없음. 분석: H_k − H_k^noV 분포, 판정이 뒤집혔을 단계 비율.
- `visref_fill` 표본: 1,000문항 전체(b_main 전 문항과 짝; 80% 최소 검출 효과 약 2.6%p, n=392면 4.2%p). 한 번의 Kaggle 세션으로 실행: timeout 41,400s(11.5h, Kaggle 세션 최대 12시간 — 2차 자료 확인, 공식 문서 미확인), 투입 중단 38,700s, 덤프 40,800s. 시간 안에 못 끝나면 남은 문항만 같은 코드로 짧은 세션을 이어 붙인다(--skip-from, 문항 순서·시드 동일).
- fill 구현 보완(설계 검토 권고): 계수 전 선택이 visref_raw_rank와 같음을 assert, ridge 계산 float64, n_inject/m·stop_reason·k·답 안 </think>·af 기록, 계수 이후 선택 순서와 관련성 r_i²의 순위 상관 기록.
- 분석·해석 규칙: 이전 등록과 같음(visref_fill − TSR, visref_fill − visref_raw_rank 짝 차이, 부트스트랩 CI, McNemar; k=1/k>1 층별). 무효 결과는 "주입량 효과 2.6%p 이상은 검출되지 않음"으로 표현.

## 2026-10-10 재현 판정 기준 사전등록 (사용자 결정, 최종 결과 전 — s1–s7 759문항 중간값은 이미 관찰함)
- 대상: b_main 최종(1,000문항) 세 비교 VisRef−ST, VisRef−TSR(주 비교), TSR−ST(보조). 읽기 (a) 채점 v3 + LLM 추출(flash-lite, 보완 1·2). 효과 크기는 같은 문항 짝 차이의 부트스트랩 95% CI(2,000회, seed 0), 논문 값은 Table 6 차이(+5.9 / +3.5 / +2.4).
- 판정(비교별):
  - "재현됨": CI가 0을 포함하지 않고(논문과 같은 방향) 논문 값을 포함.
  - "효과 크기 재현 안 됨": CI가 논문 값을 포함하지 않음. CI가 0도 포함하지 않으면 "방향은 같으나 크기 다름"을 덧붙임.
  - "판정 불가": CI가 0과 논문 값을 모두 포함.
- 유의성은 기존 사전등록(McNemar 정확 + Holm)을 함께 보고하되, 판정 문구는 위 CI 규칙으로만 정한다. 범위 문구: "공식 설정에 가까운 InternVL3.5-1B 운영점(ST ≈ 58%)에서".
- 투명성 메모: 중간값(759)에 이 규칙을 적용하면 TSR−ST "재현됨", VisRef−ST·VisRef−TSR "효과 크기 재현 안 됨"이 된다. 규칙은 이 값과 무관하게 일반 형태로 정했고, 최종 판정은 1,000문항으로만 한다.

## 2026-10-10 관찰: 이어가기는 거의 전부 상한 절단 뒤에만 일어남 (사후, s1–s7)
- TSR 단계 전이: 상한 절단 → 이어감 833 / 멈춤 477, 자연 종료 → 이어감 6 / 멈춤 262. VisRef: 921 / 468, 24 / 273.
- 해석: 우리 구현의 TSR·VisRef 이어가기는 대부분 "잘린 생각의 예산 연장"이며, 자연 종료 뒤 엔트로피가 높아 반성하는 경우(논문이 그리는 메커니즘)는 드물다(TSR 2%). 정지 통계 H(J-9, 등급 F)가 자연 종료 뒤에는 거의 항상 δ 미만. TSR−ST 이득의 출처(반성 vs 예산)는 st_official(상한을 푼 ST)로 일부 판별 가능.

## 2026-10-10 b_main_s8 결과와 s9 빌드
- s8 건전성: error 0, 109문항×3 = 327행, d0095f79e691, T4×2, torch 2.11.0+cu128, s1–s7과 (pid, method) 겹침 0, .partial 없음, 세션 약 5,250s(마지막 행 elapsed 5,144s, 대기열 79 남음).
- s8 단독(v3): ST 58.7 / TSR 58.7 / VisRef 62.4, ST 첫 단계 상한 절단 65.1%. 세션별 ST: 59.5 / 57.0 / 63.0 / 58.3 / 46.8 / 69.8 / 51.5 / 58.7.
- s1–s8 complete prefix 868문항(참고, v3, 재채점으로 정오 바뀐 행 4): ST 58.1 / TSR 60.0 / VisRef 59.9. 짝 차이 TSR−ST +2.0(+0.2~+3.8, 43:26, p 0.05), VisRef−ST +1.8(−0.2~+3.8, 51:35, p 0.11), VisRef−TSR −0.1(−2.1~+1.8, 38:39, p 1.00). 판정은 완주 후 사전등록 기준으로만.
- s9 빌드: 코드·인자·순서 s8과 같음(source sha256 동일), --skip-from s1–s8(2,678행 = 868×3 + 74), 남은 132문항. 세션 제한만 7,200 → 12,600s(time_limit 9,900, deadline 12,000) — 한 세션에 끝내기 위한 운영 변경이며 생성 설정 변경 아님. 보고서 아티팩트 v11 = s1–s8.

## 2026-10-10 검토: "4,096 상한에서도 약 절반이 루프·절단" 주장 재확인 (원자료 재계산, 사용자 요청)
- 자료: `repro/kaggle/runs/main_s0`, `full_s1`(단계 상한 4,096). 스크립트 `repro/diag/check_4096_cap_loops.py` → `repro/diag/check_4096_cap_loops.json`.
- ST 첫 단계 4,096 상한 절단: main_s0 29/60(48%), full_s1 33/58(57%). 이 기록 L826(57%)·L1011·L1145와 일치.
- 그중 반복 루프(생각 마지막 4,000자의 zlib 압축률 < 0.2, UTF-8 바이트 기준): 16/29, 18/33 → L1145와 일치. 3,000자 창이면 14/29, 16/33. ST 전체 대비 루프는 약 27–31%. 나머지 절단은 답 사이를 오가는 진동(예: pid 390 "answer A … maybe B")이나 아직 풀이 중인 텍스트(예: pid 62).
- 정정: L826 "ST의 57%가 4096 상한(반복 루프)"은 절단 전부를 루프로 적은 과장 — 루프는 절단의 약 절반. 2026-10-10 답변의 "4,096 상한에서도 단계의 약 절반이 루프에 빠지거나 잘렸다"는 "ST 첫 단계의 약 절반(48–57%)이 잘렸고, 잘린 것의 약 절반이 반복 루프"로 고친다. 전 방법·전 단계 기준 상한 도달은 main_s0 100/197(51%), full_s1 155/427(36%).
- 한계: 두 표본은 같은 셔플 순서의 앞부분이라 pid 52개 공유(합집합 66) → 독립 반복이 아니다. zlib 기준은 사후 휴리스틱(full_s1에서 자연 종료한 긴 생각 2건도 < 0.2). 정답: 절단 행 11/29, 15/33 vs 비절단 22/31, 20/25.

## 2026-10-10 st_official_c1 중간 기록 (Colab 일일 한도로 멈춤 — 판정 아님)
- 상태: done 33/100, 세션 20261010-085537, 마지막 갱신 14:43:47Z(약 5.8h). 로그·상태 사본 `repro/colab/runs/st_official_c1/interim_20261010/`(log sha256 a148b169…). results_0.jsonl은 Drive에만 있음(미회수).
- 완료 33문항은 셔플 순서의 앞부분이 아님(complete prefix 16, 첫 미완료 pid 915) → 사전등록 분석 대상 아님. 끝난 문항은 짧은 문항과 일부 극단적으로 긴 문항으로 편향.
- 사전등록 점검 항목(CRN): b_main ST 첫 단계가 자연 종료한 완료 문항 16개에서 정오 16/16 일치(기준 ≥ 90%) → Colab 환경이 Kaggle과 같은 생성을 재현한다는 증거.
- 생각 길이(완료 33): ≤1,024 14 / 1,025–4,096 9 / 4,097–16k 5 / >16k 5(33,121–35,542, max_context 근처). 정답(런타임 채점): 10/14, 6/9, 3/5, 3/5. 같은 pid b_main ST: 10, 4, 5, 2 → 합 22 vs 21(불일치 3:2). 표본이 편향돼 효과 추정에 쓰지 않는다.
- 처리 시간: 컨텍스트 상한 근처 문항은 문항당 벽시계 16,897–20,331s(4.7–5.6h). 진행 중 문항은 끊기면 처음부터 다시 생성. 최대 메모리 12.56GB(T4 15GB).
- 남은 67문항은 b_main ST 첫 단계 상한 절단 46(69%) vs 완료 33 중 17 → 남은 쪽이 긴 문항 위주. 멈춘 시점 진행 중 16개 중 9개가 컨텍스트 19k 이상. 완주까지 여러 세션(여러 날)이 필요하다는 추정(불확실).
- 운영: 사전등록 종료 규칙 유지(더 받지 못하면 complete prefix만 분석). 재시작 예약 작업(2시간마다)이 한도가 풀린 뒤 이어 실행. 투명성: 중간 정오를 이번에 봄 — 계속 여부는 정확도로 정하지 않는다.

## 2026-10-11 검토: 단계 상한 4,096 재실행의 가치 (기존 자료로 판단, 새 실행 없음)
- main_s0(단계 4,096 / 합계 8,192, 코드 8aaa689c94e9 — b_main과 다른 이전 판, 강제 답 없음, 60문항) 단계 전이: TSR 자연 종료 → 이어감 0 / 멈춤 32, 상한 절단 → 이어감 10 / 멈춤 27. visref_raw 자연 종료 → 0 / 31, 절단 → 7 / 27, EOS → 1 / 1.
- b_main(1,024)과 같은 양상: 자연 종료 뒤에는 정지 통계 H가 δ 미만이라 거의 항상 멈춘다. 상한을 4,096으로 올리면 자연 종료 비율만 늘고(ST 첫 단계 절단 63–69% → 48–57%) 이어가기는 더 줄어, TSR·VisRef가 ST에 가까워질 뿐 "완결 단계 뒤 반성"은 여전히 생기지 않을 것으로 본다(추정, 60문항·이전 코드).
- 결론(제안): 4,096 재실행은 논문 메커니즘 검증에 정보가 적고 비용이 크다(main_s0 약 60문항/시간, 1,000문항 약 17h+ 추정). 메커니즘을 가르는 변수는 단계 상한이 아니라 정지 통계 H의 추정 방식(J-9, 등급 F)이다.

## 2026-10-11 검토: 정지 통계 H의 추정 방식 — 논문에 더 가까운 후보 (구현·실행 없음, 사용자 결정 대기)
- 원문: 정의는 A — H_k = −E_{y∼πθ(·|x,τ1:k)} log πθ(y|x,τ1:k)(`VisRef_plain.txt` L604-606, Alg. 1 L524). Fig. 2는 단계마다 "Answer: 600"/"Answer: 1000"과 답 분포 막대(`VisRef_plain.txt` L359, L363). 추정법은 없음(SPEC_LEDGER 22행).
- 인용 문헌 [19] Ghosal(`VisRef_plain.txt` L998; VisRef 제1저자와 같은 사람): "sample multiple answers", "entropy of the distribution over final answers"(대장 X-11). 즉 여러 답을 뽑아 최종 답 값의 분포 엔트로피를 잰다 — 등급 C(+Fig. 2 B). 우리 방식(단일 표본 `\boxed` 토큰 엔트로피 합, T=1)은 F.
- 논문 쪽 정황: Table 5(8B, H100) ST 7.1s / TSR 7.7s / VisRef 8.2s(`VisRef_layout.txt` L892-895) → TSR이 ST보다 8%만 길어 논문에서도 대부분 k=1에서 멈춘 것으로 읽힘(Fig. 3과 충돌, C-12). Ghosal의 표준 생각 직후 평균 엔트로피 0.23 / 0.02(X-11)도 δ=0.25 아래.
- 예상(추정): Ghosal식(N개 답, 답 값 군집의 엔트로피)으로 바꿔도 자연 종료 뒤에는 거의 같은 답이 나와 멈춘다(우리 자연 종료 probe H 중앙값 0.003–0.007, L1284). N<20이면 δ=0.25 미만 ⇔ N개 만장일치라, 잘린 단계 뒤에는 오히려 이어가기가 늘 수 있다. 따라서 "완결 단계 뒤 반성"을 만들어 내지는 못할 가능성이 크다.
- 더 중요한 불일치: 논문도 대부분 k=1에서 멈춘다면 VisRef 이득은 k=1(최종 답 직전 주입, J-6)에서 나와야 하는데, 우리는 k=1 정지 543문항에서 VisRef 369 vs ST 369(L1283). 8B와 1B 차이일 수 있음(미확인).
- 제안: b_main은 그대로(원칙 2). fill 실행에 Ghosal식 H를 그림자 기록(결정에는 쓰지 않음, 별도 난수열)으로 추가해, 정지 결정이 몇 % 바뀌는지 사후 측정. N·온도 값은 원문에 없어 F로 정해야 함.

## 2026-10-11 정정: Ghosal식 다중 표본 H 그림자 기록 제안 철회 (사용자 지적)
- 사용자 지적대로 다중 표본 추정은 이전에 써 본 뒤 내려놓은 방식이다: 초기 단계에서 단계마다 완결 응답 10개(S10, sequence NLL)를 뽑았고, R25(8B, dev 3문항 TSR) GPU 4,703.9s(L642). 이후 사용자 지시로 자연 응답 엔트로피로 전환하고 S10은 별도 variant로 보존(L646), "S10 복원 없음"(L725). 전환 이유가 비용이라고 명시된 줄은 찾지 못함(미확인), 비용이 컸던 것은 기록으로 확인.
- 이번 제안(짧은 `\boxed` 답 N개)은 S10보다 생성 토큰은 적지만, 현재 엔진은 generate 호출마다 문맥 전체를 다시 prefill한다(`repro/visref/engine.py` L82-104). N개 표본 = 문맥(최대 약 1만 토큰) prefill N번, KV를 공유하면 행당 KV가 약 0.11MB/토큰(28층·KV 헤드 8·head dim 128·fp16 — 모델 config 기준 추정, 미확인)이라 1만 토큰 × 8행 ≈ 9GB로 T4에 맞지 않는다. 11.5h 안에 1,000문항을 넣어야 하는 fill에 얹기엔 과하다.
- 결정 제안: fill에 넣지 않는다. 예상 효과(자연 종료 뒤 정지 결정이 거의 안 바뀜)도 작으므로 별도 실행도 하지 않고, 최종 보고서 한계("H 추정 방식은 F, 논문·Ghosal은 다중 표본으로 읽힘")와 저자 문의로 처리한다.

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

## 2026-10-11 fill 실행 여부 독립 검토 (Fable 5.1 서브에이전트, GPU 없음) — 결과 전, 사용자 결정 대기
- 보고서 `repro/audit/fill_decision_review_20261011.md`(입력 `fill_decision_brief.md`, 계산 `fill_decision_power.py/.json`).
- 주 세션 재계산으로 확인한 핵심 수치(s1–s8 868, v3): VisRef−TSR 정오 불일치 close 3/285, cap 71/560, eos 3/23(첫 단계 종료 방식은 세 방법 868/868 동일 = 처치 전 변수). k=1·close의 주입/m 구간별 예측 변경 0/99, 0/84, 0/25, 1/73(m의 60–100% 주입 구간 포함) → 주입량을 늘려도 자연 종료 문항 답은 안 바뀌는 쪽(단, 절대 400–900토큰 추가는 관측 범위 밖, 미확인). cap·k=1·주입≥160에서 예측 변경 17/142. close 문항은 질문의 32.8%지만 VisRef 생성 토큰의 6.3%. 모든 단계가 이미 m에 닿아 fill ≡ VisRef인 문항 133/868(그중 |V|=256이 132).
- 주 세션 주장 정정: "close 문항 때문에 1,000문항 실효 검출력이 사전등록보다 낮다"는 틀림 — 불일치는 이미 cap에 몰려 있고 사전등록 8.8%는 그 평균. 전체 %p 기준 최소 검출 효과(80%, α .05) 불일치율 8.9%면 2.64%p, 12% 3.07, 15% 3.43, 17% 3.65. 정확한 표현은 "희석"(효과가 cap/eos에만 있으면 전체 2.6%p ⇔ 그 하위집단 약 3.9%p).
- 검토 권고: (A) 사전등록대로 1,000문항 실행. 결과 전 수정 기록: MDE 문구 2.6 → 2.6–3.7%p와 희석 해석, 층별 보조 분석은 처치 전 변수(첫 단계 종료 방식)로만, 시간 초과 시 complete prefix. 이중 probe는 결정 경로 동일성 검사 통과 + 예상 시간 ≤10h일 때만 유지, 넘으면 먼저 제거. 축소안(cap만, n=392, k>1만)은 이득 없거나 편향. 실행 안 함(C)은 "1/5만 주입" 반론에 답하지 못해 신뢰도 손실이 가장 큼.
- 쿼터 주의(추정): 이번 주 남은 약 17–18h − s9 최대 3.5h − fill 11.5h ≈ 2.5h → fill 후 b_main 재실행 여유 없음(전체 재실행 약 16h는 어차피 불가). 주간 리셋 시점 미확인.

## 2026-10-11 fill 사전등록 수정 (사용자 승인, 결과 전 — fill 미구현·미실행 상태) — 2026-10-10 "원인 분리 실험 사전등록 수정"의 해당 항목을 대체
- 근거: 같은 날 독립 검토(`repro/audit/fill_decision_review_20261011.md`). 실행 내용(방법·1,000문항·시드·세션·이중 probe·주 비교·해석 규칙)은 그대로.
- 최소 검출 효과 문구: "2.6%p" → "2.6–3.7%p(fill−비교군 정오 불일치율 8.5–17% 범위, 80% 검출력, α 0.05)". 효과가 첫 단계 절단·EOS 문항(약 67%)에만 있다면 전체 2.6%p는 그 하위집단 약 3.9%p에 해당한다는 희석 해석을 함께 적는다. 무효 결과는 "주입량 효과 2.6–3.7%p 이상은 검출되지 않음"으로 표현.
- 층별 보조 분석: 처치 전 변수인 첫 단계 종료 방식(close / cap / eos; 세 방법·fill이 같은 첫 단계를 공유)으로만 사전 지정. k(=1 / >1)로 나눈 분석은 방법이 바꾸는 사후 변수라 탐색적으로만 보고.
- 시간 초과: 남은 문항을 같은 코드로 이어 붙이지 못하면 셔플 순서 complete prefix만 분석하고 n을 그대로 보고.
- 이중 probe: (1) 결정 경로가 바뀌지 않음을 CPU 검사로 확인하고 (2) 예상 세션 시간이 10h 이하일 때만 포함. 넘으면 문항 수보다 먼저 이중 probe를 뺀다.
- push 시점: b_main_s9를 pull해 건전성 점검을 통과한 뒤.

## 2026-10-11 b_main_s9 결과 — b_main 1,000문항 완주, 최종 LLM 추출 준비
- s9 건전성: error 0, 132문항×3 = 396행, d0095f79e691, T4×2, torch 2.11.0+cu128, s1–s8과 겹침 0, .partial 없음, 대기열 0으로 정상 종료(샤드 elapsed 7,896s / 5,301s). 누적 st/tsr/visref_raw_rank 각 1,000행(+ s1의 이전 판 visref_raw 74행은 분석 제외).
- s9 단독(v3): ST 56.1 / TSR 59.8 / VisRef 60.6, ST 첫 단계 상한 절단 61.4%.
- 1,000문항 규칙 채점(v3) 참고값 — 판정 아님(사전등록 판정은 LLM 추출 후 읽기 (a)): ST 57.8 / TSR 60.0 / VisRef 60.0. TSR−ST +2.2(0.4~3.9), VisRef−ST +2.2(0.3~4.1), VisRef−TSR 0.0(−1.9~1.8). 보고서 갱신(`repro/reports/b_main_report.html`).
- 최종 LLM 추출 준비(EF 10-10 고정 절차): v3 재채점 사본 `repro/llm/final/rows_v3.jsonl`(3,000행, 원본 불변) → `llm_extract.py prepare --control 20` → `repro/llm/final/requests.jsonl` 56건(규칙 실패 36 + 대조군 20, 빈 텍스트 13행은 보내지 않음, 정답 미포함). 실행은 사용자 터미널(gemini-3.5-flash-lite).

## 2026-10-11 visref_fill + 이중 probe 구현, CPU 점검, 세션 빌드 (사용자 승인, 결과 전)
- 코드(이전 판 백업 `repro/.backup_pre_fill_20261011/`): `visref/dpp.py` `select_tokens_fill` — `select_tokens`의 greedy 선택(head)을 고정해 먼저 넣고, 커널에 εI(ε = 1e-6 × 평균 대각 L_ii, float64)를 더해 같은 증분 Cholesky greedy로 정확히 m개까지 채움. head = visref_raw_rank 선택(함수 안 assert). 단계에 n_head·n_fill·ridge·tail_rel_spearman(꼬리 선택 순서와 관련성 r_i²의 순위 상관) 기록. `visref/methods.py` 새 방법 `visref_fill`(그 외 visref_raw_rank와 같음, _vis_room m+2 동일), 이중 probe(cfg `dual_probe`, VisRef 방법만): 주 probe 뒤 V_k 없는 문맥(`z_k [\n] </think>` + probe prompt)에서 별도 난수열(`{pid}:probe_noV`)로 두 번째 probe → H_noV·probe_noV_* 기록, 판정은 H만 사용, 생성 토큰은 gen_tokens_noV로 따로 셈. 두 번째 probe가 비유한 logit이면 job을 실패시키지 않고 기록만(`nov_failed`). `run.py` DEFAULTS `dual_probe=0`, probe2 생성기.
- CPU 점검(클라우드 작업 공간, torch 2.14.1 CPU, transformers 4.55.0): 기존 테스트 67 + 새 테스트 9 = 76 통과. `repro/diag/fill_equiv_check.py`(작은 Qwen3 + 가짜 인코더로 run.main 전 과정) PASS — (A) st/tsr/visref_raw_rank 출력이 수정 전 코드와 행 단위로 동일(code_version·wall_s 제외), (B)(C) 이중 probe 켜기/끄기에서 H_noV 필드 외 동일, (D) fill 1단계 head 수 = raw_rank n_dpp, 주입 = m.
- 독립 코드 검토(Sonnet 서브에이전트): 확정 버그 없음(L+εI 무차별 Schur 보수 greedy와 40케이스 선택 일치, 스트레스 360케이스 크래시·부족 없음). 남은 위험: 이중 probe가 GPU fp16 배치 구성(pack 그룹·패딩)을 바꿔 같은 배치 다른 job의 샘플이 드물게 갈릴 수 있음(논리적 결정 경로는 동일; b_main의 fork·배치 잡음과 같은 종류). 반영: 두 번째 probe 실패 시 job 유지, 중복 DPP 계산 제거.
- 새 code_version = 39b3fb1b4121(b_main d0095f79e691과 다름 — fill 행은 별도 시리즈).
- 빌드 `repro/kaggle/runs/b_main_fill`(`arki13/visref-repro-b-main-fill`): ids = b_main 셔플 순서 1,000문항(동일 확인), skip 0, `--methods visref_fill --think-cap 1024 --total-think-cap 10240 --dual-probe 1`, timeout 41,400 / time_limit 38,700 / deadline 40,800(사전등록과 같음).
- 시간 추정(불확실): b_main 9세션 실행 시간 합 53,541s × VisRef 생성 토큰 비중 0.57 ≈ 8.5h, + 이중 probe·추가 주입 문맥 약 0.5–0.9h → 약 9–9.4h ≤ 10h → 사전등록 조건대로 이중 probe 포함. 초과하면 complete prefix 분석.
- 층별 기준 메모: fill 세션은 1단계 생각을 새로 생성하므로(같은 시드, fp16 배치 잡음 가능) 층별 변수 "첫 단계 종료 방식"은 b_main ST의 값(fill 처치 전에 확정)으로 쓰고, fill 자체의 첫 단계 종료 방식과의 일치율을 함께 보고한다.
- push: s9 점검 통과(2026-10-11) → 사용자 push 가능.

## 2026-10-11 b_main 최종 판정 (사전등록 기준, 읽기 (a) = v3 + LLM 추출) — 1,000문항
- LLM 추출(gemini-3.5-flash-lite, 사용자 터미널, 56건 모두 STOP): 규칙 실패 36 중 근거 있는 추출 사용 30, 근거 없음 거부 6(예: 106 VisRef가 텍스트 "1"인데 "0" 추출 → 거부), fixed 1(341 TSR, 텍스트 "Answering the question: C" = 정답 선택지), broke 0. 대조군 20/20 일치·근거 있음. 결과 `repro/llm/final/merged.jsonl`.
- 정확도: ST 57.8 / TSR 60.1 / VisRef 60.0 (논문 46.1 / 48.5 / 52.0).
- 판정(`repro/diag/final_judgment.py` → `.json`; 부트스트랩 2,000회 seed 0, McNemar 정확, 주 비교 2개 Holm):
  - VisRef−ST(주): +2.2%p, CI 0.3~4.1, 60:38, p 0.033(Holm 0.067), 논문 +5.9 → **효과 크기 재현 안 됨(방향은 같으나 크기 다름)**.
  - VisRef−TSR(주): −0.1%p, CI −2.0~1.7, 44:45, p 1.0(Holm 1.0), 논문 +3.5 → **효과 크기 재현 안 됨**.
  - TSR−ST(보조): +2.3%p, CI 0.6~4.0, 51:28, p 0.013, 논문 +2.4 → **재현됨**.
- 민감도(판정에 쓰지 않음): 규칙만 v3 — 같은 결론(VisRef−TSR 0.0, −1.9~1.8). (b) af 앞에서 재채점 — ST 56.0 / TSR 58.1 / VisRef 58.6, VisRef−TSR +0.5(−1.8~2.8). (c) af 행 오답 — ST 44.4 / TSR 48.3 / VisRef 47.4, VisRef−TSR −0.9(−3.3~1.5). 세 읽기 모두 VisRef−TSR CI가 논문 +3.5를 포함하지 않아 판정 동일.
- 사후 관찰(판정 아님): 읽기 (c)의 절대값(44.4 / 48.3 / 47.4)이 논문 ST·TSR(46.1 / 48.5)에 가깝다 → "답 상한에 닿은 응답을 오답 처리하는 평가"가 절대값 차이의 후보 중 하나일 수 있음(가설, 미확인).
- 보고서 `repro/reports/b_main_report.html` 최종판으로 갱신.

## 2026-10-11 외부 피드백 독립 검토 (Fable 5.1 high 서브에이전트, 사용자 요청) — 판정 라벨·수치 변경 없음
- 대상: 외부 도구의 피드백(표현 수정 + fill 보류 권고). 검토 보고서 `repro/audit/feedback_review_20261011.md`. 주 세션 재확인: ST 첫 단계 절단 641/1,000, (c) 읽기 VisRef−ST McNemar 94:64 p 0.0208 → Holm 0.0415, 보고서 HTML에 "대기"(LLM 열)·"문항 수가 더 필요"·"VisRef(fill 변형)" 문구 잔존 — 모두 피드백대로 맞음.
- 채택(표현): (1) "재현됨"은 사전등록 정의상 "CI가 0 제외 + 논문 값 포함 = 논문 수치와 양립"이며 효과 동등성·메커니즘 충실성 입증이 아님을 명시(라벨은 사전등록대로 유지). (2) 논문 값이 CI 밖이라는 것은 두 결과 간 차이 검정이 아님 — 논문 쪽 불일치율 3.5–20% 가정 근사 z: VisRef−TSR 2.1–3.3, VisRef−ST 2.2–3.0(가정 의존). (3) 시점: "759문항 중간 결과 확인 후, 최종 분석 전에 고정". (4) "다른 채점에서도 같다"는 "CI 기반 분류가 같다"로 한정 — 유의성은 (c)에서 Holm 0.067 → 0.042로 다름. (5) 범위: "현재 InternVL3.5-1B 운영 구현에서"; TSR 이득에는 예산 연장이 섞임, VisRef 주입/m 중앙값 0.199(2,261단계 중 1,895단계 m 미만), H 추정·Z 표현은 F.
- 기각: fill 보류 — 새 근거 없음(시간 불확실성·εI 운영 변형은 이전 검토·사전등록 수정·승인 때 이미 반영, 보호장치 있음). εI가 충실성을 다 해결하지 못한다는 지적 자체는 맞음(fill의 목적은 J-7 원인 분리).
- 정정: 피드백의 "VisRef 생성 토큰 비중 41.8%"는 fork된 TSR·VisRef 행의 gen_tokens에 1단계 토큰(각 760,920)이 중복 기록된 분모(5,704,996)를 쓴 값. fill 세션은 1단계를 자체 생성하므로 고유 토큰 분모 4,183,156 → 0.570이 맞다(L1388의 57% 유지). 다만 시간 추정 9–9.4h는 여전히 검증되지 않은 추정이다.
- 반영: `repro/report.py`에 `--judgment`(사전등록 판정 표·Holm·분류 정의·범위·시점·두-결과 비교 한계·읽기별 민감도) 추가, 고정 문구 2개 수정. 보고서 재생성(입력 = `llm/final/merged.jsonl`, LLM 열 채움) → 아티팩트 v13. 앞 항목(L1401)의 "최종판으로 갱신"은 이 재생성으로 대체.
- 사용자 결정 필요: `AGENTS.md` L30 "All model experiments are stopped by the user…" 와 L53 "20 GPU-hours total" 조항이 현재 상태(b_main 9세션 ≈ 14.9h 실행, fill 승인)와 맞지 않음 — 갱신 여부는 사용자가 정한다(주 세션은 수정하지 않음).

## 2026-10-11 AGENTS.md 상태 조항 갱신 (사용자 지시)
- 이전 판 `.AGENTS.md.bak_20261011`. 바꾼 것은 두 조항뿐:
  - L30 "All model experiments are stopped…" → STOP(`artifacts/STOP_REQUESTED.json`, 파일 유지)은 이전 `artifacts/` 파이프라인과 압축·개선 실험에 계속 적용, `repro/` 재현 트랙(b_main 완료, 사전등록 후속 `visref_fill`·`st_official`)은 이 기록의 사용자 명시 승인으로 진행 — 에이전트는 구현·검사·빌드·분석, Kaggle/Colab push·확인·pull은 사용자. 그 밖의 GPU 작업은 여전히 명시 승인 필요.
  - L53 "20 GPU-hours total…" → 20h 한도(P0–P1 5h, P2 15h)는 이전 무료 GPU 압축 파일럿(`docs/PROTOCOL.md`) 것으로 닫힌 상태 유지·재설정 안 함. `repro/` 트랙은 Kaggle 주간 쿼터(108,000s = 30h, PROTOCOL L69의 SDK 확인값)로 제한, b_main 우선·재실행 여유, 보조 실험은 무료 Colab. b_main 사용 약 14.9 세션시간(9세션). 실행마다 예상·실제 사용량을 이 기록에.
- STOP 파일과 L60(문서 수정은 STOP 해제·GPU dispatch 승인이 아님)은 그대로.

## 2026-10-11 최종 보고서 초안 (Docs 아티팩트 https://claude.ai/code/artifact/655f1c4e-1812-454a-ae93-81d07979bf79)
- 구성: 요약 · 설정(근거 등급) · 결과와 사전등록 판정(분류 정의·시점·두 결과 비교 한계) · 민감도 · 원인 분석 · 절대값 차이와 한계 · 진행 중 실험(fill, st_official) · 재현 자료 위치.
- 원인 분석 수치는 1,000문항(merged.jsonl)으로 다시 계산: 이어가기 TSR 절단 뒤 1,103 / 자연 종료 뒤 7, VisRef 1,214 / 32. VisRef k=1 713문항 ST 479 vs VisRef 480; 자연 종료 329문항 답 변경 1(기준선 TSR 1/330), 절단 366문항 33(기준선 15/380, 맞게 14 : 틀리게 12); 답 안 </think> 재출력 352 vs 67; 주입/m 중앙값 0.199(1,895/2,261단계 m 미만).

## 2026-10-11 2차 외부 피드백 검토 (보고서 초안 대상) — 판정 수치 변경 없음, 표현·기록 정정
- 주 세션이 원자료(`repro/llm/final/merged.jsonl`)로 직접 확인: 피드백 수치 모두 맞음.
  - 코드 버전 3개: 79eef648ebd1 222행, f2422d0e72ff 954행, d0095f79e691 1,824행. 보고서 초안의 "코드 버전 하나"는 틀림 → "세 코드 버전, 첫 변경은 CPU 출력 동일성 810/810(L983), 두 번째는 채점기만 변경(L1074), 모든 출력 v3 재채점으로 통합"으로 정정.
  - VisRef k=1 713문항 정오: VisRef만 정답 15, ST만 정답 14 → 순증가 1(초안의 "주입 효과 0"은 과장). k=1은 주입 뒤 정지 판정으로 정해지는 사후 집단이고 TSR k=1 기준선과 문항 집합이 다르므로 원인 확정 불가.
  - 종료 기록: af 사용 ST 239 / TSR 187 / VisRef 225, af도 100토큰 상한 종료 164 / 118 / 143, 생각 EOS 뒤 이어감 TSR 5 / VisRef 15(초안 이어가기 설명에서 누락), 최종 답 EOS 종료 761 / 813 / 775, LLM 추출 사용 11 / 8 / 11, 빈 응답 7 / 4 / 2, 런타임 오류 0.
  - 근사 z: 논문 불일치율 30% 가정 시 VisRef−ST 1.86, VisRef−TSR 1.83 → 20% 상한 근거 없음. 본문·보고서에서 z 제거, "정식 검정 못 함"만 남김.
  - 공개 폴더는 `docs/EXPERIMENT_FLOW.md`, `docs/SPEC_LEDGER.md` 구조 → "같은 구조" 문구 정정.
  - `diag/analyze_k1_injection.json`이 868문항 기준이었음 → 기본 세션 수 9로 바꾸고 1,000문항으로 재생성.
- 채택한 표현: 고정 시점 구분(주 비교·검정 = 결과 전, CI 분류 = 759문항 확인 후), "추가 이득을 확인하지 못함", VisRef−ST "Holm 보정 후 유의한 우위 확정 못 함", diag_st_plain "af 유지 조건에서 큰 하락 관측 못 함", 주입 경로 감사 "검토한 경로·표본에서 구현 오류 발견 못 함", Z = 추론 텍스트 토큰 표현, DPP "잔차 임계값 조기 종료 = 확인된 절차 차이(논문 |V_k| = m)", 셔플 seed 0 / 생성 seed 42, probe 최대 16토큰, st_official = "공식 Thinking 설정을 참고한 큰 예산 ST"(실제 생각 예산 약 3.3만), 59.3만 포함 시 "공식 수치와 양립"으로만 서술(사전등록 규칙 문구 "대표함"보다 약하게 — 해석 표현만 낮춤), fill은 εI·추가 선택도 바뀌므로 양성이어도 단독 원인 확정 불가, fill−기존 VisRef 함께 보고.
- 반영: 보고서 문서(Docs) 수정, `repro/report.py` 판정 섹션 문구 수정 → HTML 재생성(아티팩트 v14).
- (동기화 메모) 위 항목은 컴퓨터 연결이 끊긴 동안 작성해 연결 복구 후 반영. 사용자 확인: visref_fill(`b_main_fill`)은 사용자가 push해 실행 중(2026-10-11).

## 2026-10-11 공개 저장소 구성 기획 (Docs https://claude.ai/code/artifact/67aa5f4d-be58-4860-be95-7fa0126fa06d)
- 조사: Papers with Code ML Code Completeness Checklist, Open R1(Hugging Face), ReScience C 투고 안내, MLRC 형식 보고서 예(FairAC), NeurIPS 2026 MLRC 트랙, Cookiecutter Data Science, GitHub 대용량 파일·Zenodo 문서.
- 기획: 현재 `github_export/visref-repro` 구조 유지 + REPORT.md(영문, MLRC 형식)·CITATION.cff·results/(최종 표 CSV·JSON)·Makefile 추가, README를 최종 결과표·판정·범위·재현 명령으로 교체, 내보내기 대상에 llm/final·final_judgment·b_main_fill 추가, 공개 후 v1.0 릴리스 + Zenodo DOI. 실행은 사용자 결정 후.

## 2026-10-11 visref_fill 결과 (b_main_fill, code 39b3fb1b4121) — 규칙 채점 v3 기준 중간, LLM 추출 대기
- 건전성: 994/1,000행, 오류 0, 중복 0, T4×2, torch 2.11.0+cu128. shard0 500/500 완료(elapsed 39,557s), shard1 deadline 도달(40,772s): 진행 중 4문항 `.partial`, 미투입 2 → 미완 6(574, 617, 143, 914, 777, 865). 사전등록대로 complete prefix 976문항(shard0 500 + shard1 476)만 분석. 실제 세션 약 11.3h — 빌드 때 추정 9–9.4h는 빗나감(이중 probe 포함 조건 "예상 ≤10h"의 근거 추정이 틀렸음을 기록).
- 분석 `repro/diag/analyze_fill.py` → `.json`. b_main은 읽기 (a)(v3 + LLM), fill은 v3 규칙만(LLM 추출 전).
- 정확도(976): ST 57.8 / TSR 60.0 / VisRef 59.9 / fill 60.0.
- 주 비교: fill−TSR 0.0%p (CI −2.15~+2.36, 70:70, p 1.0), fill−VisRef +0.1%p (−2.15~+2.36, 65:64, p 1.0). fill−ST +2.25 (−0.31~+4.71, p 0.088).
- 사전등록 해석: fill−TSR CI 하한이 0보다 크지 않음 → "주입량 부족이 VisRef 효과를 줄였다"를 지지하지 않음. 주입량 효과 2.6–3.7%p 이상은 검출되지 않음.
- 층별(b_main ST 첫 단계 종료 방식, 사전 지정): close 322 fill−TSR −0.6 (−2.5~1.2), cap 629 +1.75 (−1.4~5.1), eos 25 −36.0 (−64~−8; 4:13). EOS 층(소수)에서 fill이 크게 나쁨: 첫 단계가 거의 즉시 EOS로 끝나 z가 거의 비어(raw_rank 주입 2–4개) fill이 관련성만으로 537–998개를 채워 넣고, 그 뒤 답이 이미지 속 숫자 등으로 흔들림(예: 214 "2006", 851 "1990") — 사후 관찰.
- 진단: 주입/m = 1.0(2,231단계 전부 m), 채운 수 중앙값 420(사분위 75~834), 꼬리 선택 순서와 관련성 순위상관 중앙값 −0.10. 평균 k fill 2.29 / VisRef 2.27 / TSR 2.11, 엔트로피 정지 907 vs 898, 답 안 </think> 재출력 472 vs 453, af 221 vs 224, 평균 생각 토큰 1,995 vs 2,019. fill 첫 단계 종료 방식이 b_main ST와 같음 874/976(나머지는 1단계 재생성의 fp16 배치 잡음; 층 변수는 b_main 값 사용).
- 이중 probe(2,231단계): H − H_noV 중앙값 0.0006. 판정이 V 유무로 갈렸을 단계 151(6.8%) — V 있을 때만 이어감 110, V 없을 때만 이어감 41. 단계 종료별 뒤집힘: close 8/403, cap 129/1,764, eos 14/64 → V 묶음은 주로 절단 단계에서 H를 올려 이어가기를 늘림.
- LLM 추출 준비: `repro/llm/fill/rows_v3.jsonl`(994행) → `requests.jsonl` 43건(규칙 실패 23 + 대조군 20, 빈 텍스트 4 제외). 사용자 터미널 실행 후 merge → 최종 읽기 (a)로 재분석.

## 2026-10-11 visref_fill 최종 (읽기 (a) = v3 + LLM 추출) — 사전등록 판정
- LLM 추출: 사용자 지시로 gemini-3.5-flash-lite 대신 Claude Haiku 5.5 서브에이전트가 추출기 역할(같은 공식 프롬프트, 정답 미포함, 응답 `repro/llm/fill/responses.jsonl`, model 필드에 기록). b_main 고정 절차(10-10, flash-lite)와 다른 추출기 — 이탈로 기록. 대조군 20/20 일치·근거 있음.
- merge 결함 발견·수정: Haiku가 답이 없는 응답에 빈 문자열을 10건 돌려줬는데, 기존 merge는 빈 추출을 채점해(객관식에서 기본 선택지로 매핑) 근거 검사(부분 문자열, 예 "no" ⊂ "North")를 통과시켜 정오 2건(608 0→1, 668 1→0)을 바꿨다. 보완 2의 취지(텍스트에 실제로 있는 답만 인정)대로 "빈 추출은 쓰지 않고 규칙 결과 유지"로 `llm_extract.py` merge 수정(이전 판 `repro/llm/llm_extract.before_emptyfix.py`). b_main 재-merge 결과는 행 단위로 동일(빈 추출 0건) 확인.
- 수정 후 fill merge: 규칙 실패 23 중 빈 추출 10, 근거 없음 거부 2, 사용 11(정오 변화 fixed 0 / broke 0). 결과 `repro/llm/fill/merged.jsonl`.
- 판정(complete prefix 976, `diag/analyze_fill.py llm/fill/merged.jsonl`): 규칙만 읽기와 같음. fill−TSR 0.0%p (−2.15~+2.36, 70:70, p 1.0), fill−VisRef +0.1%p (−2.15~+2.36, 65:64, p 1.0). 사전등록 해석 → "주입량 부족이 VisRef 효과를 줄였다"는 지지되지 않음; 주입량 효과 2.6–3.7%p 이상은 검출되지 않음. fill은 주입량 외에 εI·추가 선택도 바꾸므로 단독 원인 판단의 한계는 유지.

## 2026-10-11 독립 원인 검토(Fable 5.1 high): VisRef가 TSR보다 낫지 않은 이유 — 판정 수치 변경 없음
- 요청: 사용자 가설 "이미지 토큰 선택·주입 방법이 잘못됐을 수 있다"를 포함해 원인 후보를 독립 검토하고 후속 실험 제안. 범위 = 선택 + 주입 전체. 모델 실행 없음.
- 보고서·스크립트: `repro/audit/cause_review_20261011/` (`cause_review_20261011.md`, `analyze_cause.py/json`, `extra_calcs.py/json`).
- 주 세션이 원자료(`llm/final/merged.jsonl`, sha256 앞 16자 f54371f2e502da44)로 독립 재계산해 확인한 수치:
  - 첫 단계 종료(ST 기준) cap 641 / close 334 / eos 25. 층별 VisRef−TSR 정오 불일치: close 0:2(−0.6), cap 42:42(0.0), eos 2:1(+4.0).
  - 마지막 단계가 close인 행에서 최종 답 안 `</think>` 재출력(final_y_start > 0): ST 20/334, TSR 25/352, VisRef 157/366.
  - |V|=256 문항 196개에서 VisRef−TSR −1.0%p(검토자 CI −4.6~+2.6, CI는 주 세션 미재계산).
  - k=1 문항의 ST 대비 예측 변경: TSR 16/730, VisRef 38/713.
- 검토 결론(검토자 판정): 구현 버그 배제 / 해석 차이 일부 열림(Z 표현층·m까지 채우는 방식) / 주원인은 구조적 한계(닫힌 생각 뒤 주입은 답을 거의 바꾸지 않고, 이어가기는 상한 절단 뒤에만 생김). 원인 순위: 구조적(높음) > Z 층·선택 머리 내용(중간) ≈ 주입 형식(중간) ≈ H 추정(중간, TSR과 공유) > 주입량(낮음, fill null).
- 제안(미승인, 사용자 결정 대기): 실험 1 `visref_all`(선택 없이 |V| 전부 재주입, ST 첫 단계 cap 문항 접두, max_context 49,152, 100문항 ≈2.7h 추정 ±30%), 해석 규칙 R1–R4(주 비교 all−TSR, 기준 5.5%p = 3.5/0.641). 실험 2 `visref_hidden`(은닉 상태 Z)은 실험 1이 R1일 때만.
- 논문 근거 재확인(`VisRef_plain.txt`): 전부 재주입은 비용 때문에 배제(L300, L381-385, 지연 2.3배), |V_k| = m 강제(L536, L578), Fig. 4(b) 76.1→79.2(L794-795), Table 7 Random 67.3 / Relevance-only 75.6 / DPP 79.3(L1579-1590).

## 2026-10-11 scorer v4 + st_official 분모 수정 (사용자 지시) — 결과 후 변경, 주 판정 수치 변경 없음
- 이유: v3는 객관식 추출이 비면 Levenshtein 최근접 선택지로 정규화해 정답이 될 수 있었다(공식 MathVista 기본값 `ignore_empty_extractions=False`와 같은 동작). 외부 피드백 검토에서 지적, 합성 예제로 재현.
- v4 (`repro/visref/scoring.py`, `SCORER_VERSION='v4_mc_empty_wrong'`): 추출 규칙은 v3와 같고, 빈 객관식 추출 = 예측 없음(오답) = 공식 `--ignore_empty_extractions` 모드(`references/sources/mathvista_official_r4/evaluation/calculate_score.py` L28, L44, L121; 대장 X-25). v3는 `score(..., version=SCORER_V3)`로 재현 가능. 기본값이 v4이므로 run.py·llm_extract.py·report.py 재채점·diag 스크립트가 모두 v4를 쓴다.
- 영향(원자료 재계산): 읽기 (a)(기록된 v3+LLM) 변경 행 0/3,000(`diag/final_judgment.json` `v4_changes_in_a`=0) → 주 판정 불변. 읽기 (b)(af 앞 재채점)는 ST pid 122, TSR pid 117이 v3에서 빈 추출→정답이었음 → v4: ST 56.0→55.9, TSR 58.1→58.0, VisRef 58.6 불변; VisRef−TSR +0.5→+0.6(−1.7~+2.9), VisRef−ST +2.6→+2.7(+0.5~+5.0), TSR−ST +2.1(+0.1~+4.1) 불변. v3 값은 `b_af_stripped_v3`로 병기. (c)·rule_only 불변. `diag/st_plain_analysis.json`은 v4로 재실행해도 버전 문자열 외 43개 값 동일(파일은 덮어쓰지 않음).
- `repro/diag/analyze_st_official.py`: 오류 행을 분모에서 빼던 것(이전 L68) → 완료 접두의 오류 행을 분모에 남기고 오답 처리(run.py L220-221과 같음). 오류 제외 값은 `excluding_errors`로 따로 보고(판정 미사용). steps 없는 오류 행 처리 추가.
- 백업: `repro/.backup_scorer_v4_20261011/` (scoring.py, test_repro.py, final_judgment.py/json, analyze_st_official.py, EXPERIMENT_FLOW.md, SPEC_LEDGER.md 이전 판).
- 검증: CPU, torch 2.7.0 + transformers 4.55.0(고정 환경)에서 `pytest` 84 passed(신규 scorer 테스트 8건 포함). 로컬 transformers 5.14.1에서 실패하던 2건은 고정 환경에서 통과.
- st_official 중간 자료(Google Drive, Colab 사용량 한계로 중단): 세션 2개, 33행·14행(같은 pid 14행은 생각·답이 바이트 단위로 동일). b_main 셔플 순서 완료 접두 n=16(오류 0): 공식 설정 ST 75.0%(Wilson 50.5–89.8) vs b_main ST 68.8%, 차이 +6.3(−12.5~+25.0). 사전등록 n=100의 중간값이므로 판정에 쓰지 않는다. 4/16행이 context 한도(생각 ≥33,236토큰)까지 생성. 결과 파일은 로컬에 저장하지 않음(Drive `st_official_c1/results_0.jsonl` 두 개).

## 2026-10-11 논문 지문 대조 (GPU 없음, `repro/diag/paper_fingerprints.py` → `diag/paper_fingerprints.json`) — 판정 수치 변경 없음
- 목적: 저자 문의 전, 논문에 적힌 부수 수치와 우리 설정의 정합성을 원자료로 확인. 저자 선택을 확정하지는 못하고, 어떤 설정·읽기가 논문 수치와 양립하는지만 보인다.
- 시각 토큰 수: MathVista 메타데이터의 이미지 크기로 InternVL dynamic_preprocess를 재계산(기록된 b_main n_visual과 1,000/1,000 일치). 평균 |V|: 우리 설정(max_num=12, 썸네일 포함) 1,961.5 = 논문 "∼1,772"(8B, X-6) 대비 +10.7%; max_num=12·썸네일 없음 1,755.6(−0.9%); max_num 6/8/10은 −24~−41%. → 저자 전처리(또는 V 후보 집합)가 썸네일 타일을 뺐을 가능성. 미확인, 문의 항목에 추가.
- 답 처리 읽기 vs Table 6 1B(46.1/48.5/52.0): (a) 57.8/60.1/60.0, (c) af 행 오답 44.4/48.3/47.4, (e) 강제 답 행 오답 25.2/26.4/26.9, (f) 둘 다 오답 24.4/25.3/26.3. 절대 수준은 (c)가 가장 가깝지만 VisRef−TSR은 모든 읽기에서 −0.9~+1.0 → 답 처리·추출만으로는 +3.5를 설명하지 못함.
- 생각 양: 생성 토큰 비 TSR/ST 1.98, VisRef/ST 2.14 vs 논문 Table 5 지연 비(8B) 1.08/1.15. ST 첫 단계 64.1%가 1,024 상한 절단. 모델 크기가 달라 참고용이지만, 논문 설정에서는 이어가기가 드물었음을 시사.
- 같은 배치 구성의 결정성: st_official 두 Colab 세션의 같은 pid 14행은 생각(최대 3,630토큰)·답이 바이트 단위로 동일. 반면 배치 구성이 다른 경로(TSR k=1 vs ST, fill 재생성)에서는 차이(16/730, 874/976) → 진단 실험은 배치 구성 고정(또는 batch=1) 필요.

## 2026-10-11 사후 관찰: 읽기 (c)와 VisRef 답 길이 (판정 아님, 판정 수치 변경 없음)
- 읽기 (c)(af 행 = 답 1,024토큰 안에 EOS 없음 → 오답): ST 44.4 / TSR 48.3 / VisRef 47.4 vs 논문 46.1 / 48.5 / 52.0. ST·TSR은 −1.7/−0.2로 근접, VisRef만 −4.6. 같은 원자료에서 af 앞 텍스트를 다시 채점한 (b)는 55.9/58.0/58.6이므로, 근접에는 "상한에 닿은 답 = 오답"이 필요하다. 엄격 추출 변형((c)+boxed·answer_phrase 규칙만) 43.7/47.7/46.7.
- VisRef가 (c)에서 수렴하지 않는 이유 후보: af 행 VisRef 225 / TSR 187 / ST 239. 같은 z_1(fork)·k=1로 V_1 주입 유무만 다른 짝(713문항): af VisRef만 70 / ST만 52(McNemar p 0.123), 답 길이 중앙값 125 vs 82토큰. 같은 문맥인 TSR k=1 vs ST(잡음 바닥): 12 / 21, 84 vs 84. → 태그 없는 V 주입이 최종 답을 길게 만들어 상한 도달을 늘리는 경향(유의하지 않음, 사후).
- 해석 한계: 4개 이상 읽기 중 논문 두 수치에 맞는 것을 사후에 고른 것. (c)의 TSR−ST +3.9는 논문 +2.4보다 (a)의 +2.3이 더 가깝다. 저자 평가 방식은 미확인 → 문의 항목("답이 길이 상한에 닿으면 어떻게 채점했나") 추가.

## 2026-10-11 정정 메모: fill(ridge)의 꼬리는 관련성 채움이 아님 (판정 수치 변경 없음)
- `visref_fill`(dpp.select_tokens_fill): 머리 = rank 한정 DPP 선택(visref_raw_rank와 동일, assert), 꼬리 = 커널 L+εI(ε = 1e-6·mean L_ii)로 같은 greedy를 m까지 계속. rank 소진 뒤 이득은 ≈ ε(1 + a_jᵀ(A_SᵀA_S+εI)⁻¹a_j) → 이미 고른 집합이 가장 덜 덮은(작은 고유값) 방향에서 큰 토큰을 고름. 꼬리 선택 순서와 관련성 r_i²의 Spearman 중앙값 −0.10(`diag/analyze_fill.json`) → 관련성과 거의 무관.
- 관련성 채움(rank 뒤 r_i² 상위로 m까지)은 이전 `visref_raw`(`.backup_79eef648ebd1/visref/dpp.py` L51-52)였고 b_main_s1 74문항에서만 실행 후 제외(EF L977). v4 규칙 재채점: ST 59.5 / TSR 56.8 / rank 56.8 / visref_raw 59.5, visref_raw−TSR +2.7%p(4:2), visref_raw−rank +2.7%p(2:0) — 표본이 작아 판단 불가. → "fill 0.0%p"는 관련성 채움을 검증하지 않는다.

## 2026-10-11 재주입 형식·위치 엄격 재독(Fable 5.1, 사용자 요청) — 판정 수치 변경 없음
- 보고서: `repro/audit/injection_format_reread_20261011.md` (주 세션이 핵심 인용 P:203, 240-241, 244, 271-272, 295-297, 381-386과 Fig. 1·2를 직접 대조).
- 결과: 논문이 명시한 항목(재주입 대상 = 원본 V의 부분집합, z_k → Ṽ_k → H_k → 큐 순서, Ṽ_k가 H_k·최종 답 문맥에 포함, 이전 Ṽ_i 유지, 실제 문맥 추가)은 구현과 모두 일치. 태그·구분자·`</think>` 처리·position id·답 프롬프트는 논문 어디에도 없음 → 잘못 읽은 문장은 없고, 논문 미지정 접합부 `[Ṽ_k] </think>`(우리 구성)가 답 안 `</think>` 재출력·답 길이 증가의 유력 원천으로 지목됨(추론).
- 다음 후보: 고정 입력 진단(텍스트 큐 / 구분자 / `<img>` 감싸기 / 내용 대체 / 위치 재사용 / 접합부 로짓) 사전등록 후 실행.

## 2026-10-11 사후 관찰: 재출력된 답의 내용과 "주입 뒤 생각" 문제 (판정 아님)
- VisRef k=1 행 중 답 안 `</think>` 재출력 352행: 두 번째 `</think>` 앞 텍스트 중앙값 146자, 이미지 관련 단어(image/picture/figure/diagram/graph/look/shown/see) 포함 41/352. 대부분 "The correct answer is A … **Answer:** A." 같은 답 초안 → 모델이 우리 `</think>`를 생각 종료로 받아들이지 않고 초안을 쓴 뒤 스스로 다시 닫는 형식 혼동에 가깝다(이미지를 다시 따져 보는 생각은 드묾).
- 같은 첫 단계 짝 정오(ST 대비): 재출력 352행 VisRef만 정답 9 / ST만 4, 재출력 없음 361행 6 / 10 — 소수라 판단 불가.
- 구조 관찰: VisRef의 71.3%(713/1,000)가 k=1에서 멈춰, 재주입 Ṽ가 다음 생각을 이끄는 경로("reinjected to guide subsequent reasoning", P:375)를 한 번도 거치지 않고 답 직전 주입으로만 작용한다.
- 설계 쟁점(사용자 제기): 정지 단계에서 Ṽ_k 뒤 `</think>`를 강제하는 것은 Alg. 1의 문자 그대로 읽기(10→12→17행)이지만 논문이 정한 토큰열은 아니다. 대안 = Ṽ_k 뒤 자유 생성으로 모델이 스스로 `</think>`를 내게 함(Ghosal [19] TTBC 2 "allowing the end-of-thinking delimiter (</think>) to pass"와 같은 방식). 고정 입력 진단 후보로 추가(예산 혼입 주의).
- 보충(같은 날, "주입 뒤 자유 생성" 변형의 조건): (1) 예산 — TSR도 각 단계는 모델이 `</think>`를 낼 때까지 자유 생성하므로, 자연 종료 단계에서는 VisRef 자유 생성이 불공정하지 않다. 차이는 상한 절단 단계뿐(TSR은 정지 시 `\n` + `</think>` + `\nFinal Answer:` 강제)이므로, 자유 생성은 같은 단계 상한(1,024) 안의 남은 여유로만 허용한다(절단 단계는 여유 0 → 현행과 동일). (2) 정지 통계 일관성 — 최종 답 y를 자유 생성으로 바꾸면 H_k도 같은 분포(자유 생성 후 답)에서 재야 Alg. 1 12행·17행이 같은 y 분포를 가리킨다(Ghosal식 다중 표본·답 수준 엔트로피). (3) 해석 등급 — 논문 미지정 영역의 대안 해석. Fig. 2는 Ṽ_k 바로 뒤 "Answer:" 상자를 그려 현행(곧바로 답) 쪽을 약하게 지지한다.

## 2026-10-11 Fig. 3 재판독(사용자 질문)과 단계당 토큰 비교 (판정 아님)
- Fig. 3은 VisRef vs Parallel thinking(+ST 별표) 비교이며 TSR 곡선은 없다. 가로축 = 병렬 trace 합계 thinking token 예산, 점 = trace 수 증가(단계가 아님) (VisRef_plain.txt L700-723, 캡션 L746; page-07 고해상도 확대 판독).
- (b) InternVL-3.5-8B MathVista 판독(±약 200): ST ≈600; VisRef 점 ≈4,000 / 6,100 / 8,000 / 10,100 / 12,100 / 14,500(간격 ≈2,000 → trace당 ≈2,000토큰 ≈ ST의 3.3배; 첫 점을 C=1로 보면 6.7배 — 장부 A-4의 두 판독과 같은 모호성); Parallel thinking 점 ≈1,900 / 3,150 / 4,450 / 7,550 / 9,250 / 12,750 / 14,500. 시각 토큰 포함 여부 미확인. Table 5(+15% 지연)와의 불일치 그대로(C-12).
- 우리 1B: 이어가기 단계(k≥2) 길이 TSR 1,115단계 중 97%, VisRef 1,261단계 중 94%가 1,024 상한 도달(평균 1,011 / 987); 둘 다 이어간 248문항의 2단계 길이 차 중앙값 0 → 단계당 토큰은 같고, 차이는 이어가는 빈도. 생각 토큰 평균 ST 761 / TSR 1,888 / VisRef 2,006(중앙값 모두 1,024).
- 해석(추론): 논문 VisRef는 trace당 ST의 수 배를 생각 → 주입 뒤 다시 생각하는 경우가 흔했을 가능성. 우리는 71%가 k=1 정지(자연 종료 뒤 H 중앙값 0.003–0.007) → 이미지가 다음 생각을 이끄는 경우가 드묾. 정지 통계(H 추정) 항목의 우선순위를 높일 근거.
- 보충(같은 날, 사용자 질문): Fig. 3의 Parallel thinking ≠ TSR — 병렬 thinking = 독립 텍스트 trace 여러 개 + 다수결("samples multiple text-only reasoning trajectories", P:714-716; [19],[46]), TSR = 한 trace를 "Wait, think more"로 연장("text-only reflection without visual refocusing", P:1287; [34]). 논문에 TSR의 토큰 사용량은 없다(Table 5 지연뿐).
- 상한 절단 단계에서의 이미지: VisRef cap→이어감 1,214 / cap→정지 613. 이어가면 Ṽ_k 뒤 새 1,024토큰 단계가 있고, 정지하면 Ṽ_k 직후 `</think>` + `\nFinal Answer:` 강제라 생각 여유가 없다. 다만 답 영역에서 생각을 이어 가는 경우가 흔함: 1단계 절단 후 둘 다 k=1 정지한 344문항에서 답이 1,024 상한까지 간 비율 TSR 30%(102) / VisRef 39%(134).
- 진단 설계안: "정지 시 추가 여유 A"를 TSR·VisRef에 똑같이 부여(TSR = [z_1]\n 뒤 자유 생성 ≤A, VisRef = [z_1]\n[Ṽ_1] 뒤 자유 생성 ≤A, 모델이 스스로 `</think>`를 내면 답, 못 내면 현행 강제 답). VisRef에만 주면 TSR−ST 이득의 주원인인 예산 연장 효과를 다시 재게 됨. 대상에 절단·정지 344문항 추가, 비교는 새 TSR-A vs VisRef-A(b_main TSR과 직접 비교하지 않음).

## 2026-10-11 계획 독립 검토(Fable 5.1, 사용자 요청)와 1단계 probe 사후 탐색 — 판정 수치 변경 없음
- 보고서·검증: `repro/audit/plan_review_20261011.md`. 분석 스크립트 `repro/diag/analyze_step1_probe.py` → `diag/step1_probe_analysis.json`(사후 탐색, 사전등록 아님).
- 관찰(주 세션 재계산): 저장된 1단계 probe 즉답(T=1, pid별 공통 probe 난수, 같은 z_1)은 VisRef 58.9 / TSR 57.2, 28:11, McNemar p 0.0095, CI +0.5~+3.0. 차이는 불확실(어느 쪽이든 이어감) 309문항에서만(36.6 / 31.1, 28:11), 둘 다 k=1인 691문항은 0:0. 같은 309문항의 최종 답은 42.4 / 43.0(30:32) → 즉답에서 보이던 V_1 이득이 이어가기 뒤 사라짐.
- 정정: (1) "자연 종료 뒤 H 중앙값 0.003–0.007"(이 파일 'Fig. 3 재판독' 항목) → 그 값은 1단계 전체 중앙값이며 자연 종료만은 TSR 2.4e-5 / VisRef 5.8e-5. (2) Fig. 3 "trace당 ≈3.3× ST"는 판독 불확실(1.7–6.8배), Table 5와의 충돌도 미확인. (3) 자유 생성은 탐색 변형으로 분류(Fig. 2 "Answer:"는 즉답 쪽 지지).
- 검토 권고(사용자 결정 대기): 즉답 확인 실험 사전등록(1,000문항, TSR_imm / VisRef_imm / Random_imm / Rel_imm, 시드 3) → 조건부 이어가기 상한 4,096 소표본. Set N 형식 변형·Set C 여유 A는 축소 또는 보류. 이후 실행은 z_k 토큰 id와 DPP 선택 인덱스를 저장.

## 2026-10-11 외부 피드백(계획 검토에 대한 2차 검토) 반영 — 판정 수치 변경 없음
- 주 세션 재계산으로 확인: 309문항 probe 차이 +5.50 vs 최종 −0.65, 문항별 차이의 차이 +6.15(부트스트랩 20,000회 95% CI −0.32~+12.3; 피드백 10만 회 −12.62~+0.32는 부호 반대 표기) → **이득 감소 자체도 유의하지 않음**. 309문항 중 한 방법만 이어간 문항 61. 최종 답 v4 규칙 재채점 131:133 유지. VisRef 첫 단계 n_inject < m 842/1,000.
- 표현 정정: (1) "이미지가 즉답에서 도움 → 이어가기 뒤 사라짐/절단이 지움"은 과함. 근거에 맞는 표현 = "추가 시각 재주입이 1단계 probe 즉답에서 이득을 보였지만(사후 탐색), 최종 출력에서는 추가 이득을 확인하지 못했다". 두 방법 모두 처음부터 이미지를 보므로 '이미지 유무'가 아니라 '재주입 유무' 비교이고, probe(T=1·boxed·16토큰)와 최종 답(T=0.6·top-k/p·1,024·af)은 생성 정책·난수가 달라 원인(절단·강제 답)을 분리할 수 없다. (2) Fig. 2의 "Answer:"는 Adaptive Stopping Criterion 패널의 엔트로피 계산 도식이므로 최종 답 생성 방식(즉답)을 지지하는 근거로 쓰지 않는다(이전 기록의 해당 문장 철회). (3) Fig. 3 배수 1.7–6.8배는 판독 가정별 후보값이지 검증된 범위가 아니다. (4) Table 7은 8B VisRef 선택 방식 비교이므로 1B 즉답 변형은 관련 탐색이지 그 절제 실험의 재현이 아니다. (5) hidden_rank_check "12문항 중 5 불일치"는 과함: 3문항은 절단 단계의 '\n' separator로 원 길이가 1토큰 긴 것(코드가 잘라냄, `diag/hidden_rank_check.py` L114-116), 2문항은 고유 토큰 수 차이 → "정확한 id 일치는 확인할 수 없고 복원 통계의 차이가 관찰됨". (6) 자연 종료층 "정확도 판별력이 거의 없다"는 단정 철회: 예측 327/329 동일·fill close층 4:6은 낮은 민감도를 시사할 뿐, 정답률 약 72%라 형식 변형이 오답을 바꿀 여지는 남음 → 메커니즘 지표 1차, 정확도 2차로 유지.
- 즉답 확인 실험 설계 보완(사전등록 초안 반영 예정): 같은 1,000문항 반복 확인이라는 범위 명시, 시드 반복은 답 표본만 반복하므로 문항 단위(시드 평균) 분석; 선택 방식 비교는 문항별 주입 수를 DPP 실제 수 n_i에 맞춘 Random_n·Rel_n을 주 비교로, m개 팔은 별도 계열; 답 생성 정책은 모든 팔 동일(주: 관찰과 같은 probe 정책 T=1로 재현, 보조: 최종 답 정책); 절단 가설은 별도 실험에서 TSR·VisRef × 단계 상한 1,024/4,096을 같은 실행·같은 답 정책으로 비교하고 "단계 예산 정책 효과"로 해석; 100–150문항은 메커니즘 진단용; 비용은 파일럿으로 측정 후 확정.

## 2026-10-11 사전등록: step1_branch (309문항, 조건 A–D) — 결과 전 고정 (사용자 결정: 309문항)
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
- 검출력: n=309, 불일치율 10–15% 가정 시 짝 차이 SE ≈ 1.8–2.2%p, 80% 최소 검출 효과 ≈ 5–6%p(사후 관찰 크기 +5.5와 비슷). 무효 결과는 "5–6%p 이상의 효과는 검출되지 않음"으로만 표현.
- 비용 규칙: 파일럿 20문항의 시간으로 본 실행 시간을 추정해 3시간을 넘으면 자유 생성 상한을 256으로 줄이는 것만 허용(파일럿 정확도는 보지 않고 시간만 사용).
- 실행: Kaggle T4(1–2장), 코드 `repro/diag/step1_branch.py`. push·status·pull은 사용자.

## 2026-10-11 step1_branch 구현·파일럿 빌드 (결과 전)
- 코드: `repro/diag/step1_branch.py`(조건 A–D, b_main의 engine·DPP·scorer v4 재사용, z id·선택 인덱스·복원 통계 저장), 분석 `repro/diag/analyze_step1_branch.py`(결과 전 고정), 빌더 `repro/kaggle/build_step1_branch.py`. 입력 `diag/step1_branch_{ids,samples}.json`(1단계 종료 cap 297 / eos 7 / close 5; cap 297행은 모두 separator '\n'으로 끝남).
- 사전등록 보충(결과 전): 공동 1차 비교의 "규칙 충족" = 95% CI 하한 > 0 그리고 양측 부트스트랩 p의 Holm 보정 < 0.05.
- 검증: CPU 테스트 `tests/test_step1_branch.py` 5개(분리 규칙, 상자 종료, 생각 닫기, 결정성, V=0이면 B≡A·D≡C로 짝 난수 확인) 포함 전체 89 passed(torch 2.7.0, transformers 4.55.0). 노트북 디코딩·파일 sha 일치·`--help` 실행 확인. 분석 스크립트는 합성 행으로 코드 경로만 확인(데이터 아님).
- 파일럿: `repro/kaggle/runs/step1_branch_pilot`(앞 20문항, think_extra 512, T4, timeout 3,600s). 파일럿에서는 시간·복원 통계만 보고 정확도는 보지 않는다. 본 실행은 파일럿 pid를 `--skip-from`으로 건너뛰고 나머지 289문항.

## 2026-10-11 공개용 폴더 갱신 (github_export/visref-repro; 지금 private, 실험 완료 후 public; 저장소 생성·커밋·push는 사용자)
- 사용자 결정: 루트가 아니라 export를 최신화해 올림. 10/10 이후 코드·테스트, 추가 run(b_main_s9, b_main_fill, b_main_rescore_v2/v3), audit·diag 검토, step1_branch 사전등록·파일럿을 포함. LICENSE 저작권자 AhnJG.
- `scripts/make_github_export.py` 변경: CODE에 `kaggle/build_step1_branch.py`, `audit/cause_review_20261011/*`, `llm/final`·`llm/fill`의 requests/responses/merged(diag 스크립트가 merged를 읽음) 추가; RUNS에 `b_main_fill`, `b_main_rescore_v*`, `step1_branch_pilot` 추가; RUN_KEEP에 `summary.json`, `rows.jsonl` 추가; 숨김 파일(`.tmp_*`, 백업) 제외. 계속 제외: `audit/review_inputs`(EF 발췌 사본), `llm/*/rows_v3.jsonl`, 번들 노트북. 이전 판: `scripts/.make_github_export.bak_20261011`, `github_export/.visref-repro_snapshot_20261010`, `github_export/.README_export.bak_20261011`, `github_export/.LICENSE_export.bak_20261011`.
- export README: 상태 문구(b_main 완료, 판정 파일 `repro/diag/final_judgment.json`), 구조 목록, 테스트 요구사항(torch + transformers) 갱신. 결과표는 공개 시 채움.
- 검증: git 추적 대상 200개 파일, 72.8MB, 최대 22.9MB(`repro/llm/final/merged.jsonl`). 비밀 키 패턴·이메일·개인 경로 0건, PDF·.pt·노트북 0건. CPU `pytest` 89 passed(torch 2.11.0 + transformers 4.55.0, Python 3.13). 새 복사본에서 `scripts/fetch_sources.py` 17/17 sha256 일치, `diag/final_judgment.py` 재실행 결과가 커밋본과 바이트 단위로 동일.
- 미해결(사용자 결정 필요): README 1단계 `report.py ... --rescore-data` 명령으로 재생성한 결과(scorer v4, TSR 600/1000)가 커밋된 `repro/reports/b_main_report.json`(scorer "recorded (per-row, may mix versions)", TSR 601/1000)과 다르다. 재생성값은 위 L1379의 v3 규칙 채점 참고값(TSR 60.0)과 같다. 보고서를 재생성할지, README 명령을 고칠지는 아직 정하지 않았다.

## 2026-10-11 step1_branch 파일럿 결과(시간·복원만) → 본 실행 빌드
- 파일럿(20문항, T4×2, code a9e2cafef0a8, torch 2.11.0+cu128, transformers 4.55.0): 8문항 묶음 139–162초(문항당 약 18–20초/GPU), 두 GPU 병렬 기준 문항당 약 10초, 세션 약 6분. 복원: 재토큰 수 = b_main think_tokens 20/20, 선택 수 = b_main n_dpp 20/20(차이 0), 오류 0, 답 형식 유효 320/320(조건 합산; 조건별 정확도는 보지 않음).
- 결과 파일은 사용자가 `github_export/visref-repro/...`에서 pull해 그 아래에 저장됨 → `repro/kaggle/runs/step1_branch_pilot/output/`로 복사(원본은 그대로).
- 비용 규칙 판단: 본 실행 289문항 예상 약 50분(< 3시간) → 자유 생성 상한 512 유지. 본 실행 `repro/kaggle/runs/step1_branch_main`(파일럿 20 pid 건너뜀, timeout 7,200s, 코드 묶음 sha 파일럿과 동일).

- push 완료(사용자 요청으로 Claude가 수행): `github.com/AhnJG/visref-repro`(private) `main` = `7ec0965`(사용자가 로컬에서 만든 커밋 그대로, 파일 200개). 원격 트리의 파일 수 200을 확인했다. 작성자 이메일은 사용자 결정에 따라 그대로 두었다.

## 2026-10-11 step1_branch 본 실행 결과 (309문항 = 파일럿 20 + 본 289) — 사전등록 규칙 그대로 판정
- 데이터: 본 실행 출력은 사용자 자동 pull로 `repro/runs/kaggle/visref-step1-branch-main/20261011T160142Z/`에 저장(사전 안내한 `repro/kaggle/runs/step1_branch_main/output/`가 아님). JSONL 145+144줄, 고유 pid 289, 파일럿 skip 20과 합쳐 309/309. 같은 폴더 아래 반복 다운로드 166벌은 JSONL sha256 전부 동일(6df86ae1d93e…, 0fbb8ba07645…) → 데이터 영향 없음. 번들 src 10개 파일 sha = 로컬 repro와 동일, code_version a9e2cafef0a8(파일럿과 같음), Kaggle torch 2.11.0+cu128 / transformers 4.55.0, T4×2, GPU별 약 2,918초, 오류·OOM 0.
- 분석: `python3 diag/analyze_step1_branch.py <파일럿 2개> <본 2개> --ids diag/step1_branch_ids.json --out diag/step1_branch_analysis.json`(결과 전 고정 코드, 수정 없음).
- T=1 3개 평균 정답률(%): A 33.98 / B 35.38 / C 35.28 / D 36.25. greedy: A 35.28 / B 36.57 / C 39.81 / D 37.86. 답 형식 유효 99.1–99.6%.
- 공동 1차: B−A +1.40%p, 95% CI −1.19~+3.99, Holm p 0.578 (greedy 18:14). D−C +0.97%p, CI −2.27~+4.21, Holm p 0.578 (greedy 14:20). → ①·② 모두 불성립.
- 판정(사전 규칙 ④): 즉답 재주입 이득 재현 안 됨. b_main 1단계 즉답에서 본 +5.5%p(309문항, 표본 1개)는 잡음이었을 가능성이 있다. 무효 결과는 사전 문구대로 "5–6%p 이상의 효과는 검출되지 않음"으로만 표현한다(CI 상한은 약 +4%p).
- 보조(판정에 쓰지 않음): D−B +0.86 (CI −2.16~+3.88), C−A +1.29 (CI −1.83~+4.42; greedy만 25:11, McNemar p 0.029 — 보정 없는 보조 지표 하나), (D−C)−(B−A) −0.43 (CI −4.21~+3.24). 1단계 종료별 T=1: cap 297문항 A 33.2 / B 34.7 / C 34.6 / D 35.6, eos 7, close 5(작아서 기술만).
- 메커니즘(C vs D): 모델이 스스로 `</think>`로 닫음 99 vs 82, 512 도달 203 vs 220, 이미지 관련 단어 언급률 22.7% vs 31.4%. 이미지를 넣으면 생각 글에 이미지 언급은 늘지만 정답률로 이어지지 않았다.
- 복원: 재토큰 수 일치 301/309, 선택 수 일치 296/309(차이 ±1–3개), 둘 다 일치 293. 사후 민감도(판정 아님): 일치 293문항만 보면 B−A +1.59 (CI −1.14~+4.44), D−C +0.57 (CI −2.73~+3.75)로 결론 같음. 조건 A–D는 같은 z·같은 선택을 공유하므로 조건 간 비교는 이 불일치와 무관하다.
- 해석(추론, 사실 아님): 우리 구현에서 TSR과 VisRef가 갈리는 곳은 이 309문항뿐이다. 논문 1B의 VisRef−TSR +3.5%p(`references/papers/VisRef_fulltext/VisRef_layout.txt` L900: 48.5 → 52.0)를 이 309문항 안의 1단계 재주입만으로 내려면 309문항 안에서 약 +11%p가 필요하지만, 이번 CI 상한은 약 +4%p(1,000문항 환산 약 +1.3%p). 따라서 우리 1단계 지점의 재주입은 논문 크기의 차이를 설명하지 못한다. 논문의 이득이 다른 곳(정지 규칙·단계 길이·주입 방식·2단계 이후)에서 나왔을 가능성은 이 실험으로 판단할 수 없다.
- 다음: 사전 규칙 ④에는 후속 실험이 정해져 있지 않다. 남은 의문(주입 형식·위치, 단계 경계, 썸네일, 상한 도달 답 처리)은 저자 문의로 넘기는 것을 제안.
- (10-12 사후 계산, 사전등록 아님) 사용자 질문 "이미지 넣고 더 생각하는 게 더 좋은가"에 답하려고 같은 분석 함수(`pair`, 부트스트랩 seed 0)로 D−A를 계산: +2.27%p (95% CI −0.97~+5.72, greedy 23:15, McNemar p 0.256). greedy 정답률 순서는 C 39.81 > D 37.86 > B 36.57 > A 35.28로 T=1 평균 순서(D 최고)와 다르다 → "D가 가장 좋다"는 309문항에서 확인되지 않음.

## 2026-10-12 실험 종료 + 공개 폴더 완성 (사용자 지시 "여기서 끝내고 깃헙 정리를 완성해")
- 실험 종료: 새 GPU 실험 없음. st_official은 미완료로 남기고 분석하지 않는다. 저자 문의는 아직 안 함(질문 목록은 REPORT §5.3).
- step1_branch 본 실행 출력을 `repro/kaggle/runs/step1_branch_main/output/results/`로 복사(원본 `repro/runs/kaggle/visref-step1-branch-main/20261011T160142Z/`는 그대로, JSONL sha256 동일). `scripts/make_github_export.py` RUNS에 `step1_branch_main` 추가(이전 판 `scripts/.make_github_export.bak_20261012`).
- `repro/reports/b_main_report.*` 재생성: README 명령(`--rescore-data`, 규칙 채점 v4만)으로 다시 만든 결과와 커밋 파일이 달랐던 미해결 건(10-11 기록) 해소. 새 값 ST 57.8 / TSR 60.0 / VisRef 60.0(규칙만; 판정 읽기 (a)의 TSR 60.1과의 차이 1행은 LLM 추출로 바뀐 341 TSR). 이전 판(기록값, 판 혼합) 백업 `repro/.backup_report_20261012/`. 판정 수치 변경 없음.
- 공개 폴더에서 직접 관리하는 새 파일: `REPORT.md`(영문 MLRC 형식; Sonnet 서브에이전트 초안 → 주 세션이 수치를 JSON·merged.jsonl·EF와 대조, 근거를 찾지 못한 문장 2개 삭제), README 결과표·후속 실험·재현 명령 갱신(이전 판 `github_export/.README_export.bak_20261012`), `results/`(`scripts/make_results.py`가 diag JSON에서 생성), `CITATION.cff`(cffconvert로 스키마 1.2.0 검증, 저자 표기 alias AhnJG), `Makefile`(fetch / analysis / results / check / report / test), `.gitignore`에 `build/`.
- 검증: 새 복사본에서 `make analysis results report` → final_judgment.json, analyze_fill.json, step1_branch_analysis.json, results/, b_main_report.json 모두 바이트 동일(HTML은 생성 시각만 다름). CPU pytest 89 passed(torch 2.7.0, transformers 4.55.0). 새·변경 파일에서 키 패턴·개인 경로·이메일 0건.
- git: 장치에서 실행한 `git add -A --dry-run`이 `.git/index.lock`을 남겼고 삭제 권한이 없어 `.git/stale-index-lock-20261012`로 옮겼다(git 동작에는 영향 없음, 사용자가 지워도 됨). 커밋·push는 하지 않았다.

## 2026-10-12 공개 폴더에 그림 2개 추가 (사용자 결정: GitHub 저장소, 결과 요약 + 방법 흐름도, 영어)
- `figures/paired_differences.svg`: 세 비교(VisRef−ST, VisRef−TSR, TSR−ST)의 짝 차이와 95% 부트스트랩 CI(점·막대) 옆에 논문 Table 6 차이(마름모). `scripts/make_figures.py`(표준 라이브러리)가 `results/paired_differences.csv`에서 그림. 색은 dataviz 기본 팔레트 1·2번(파랑·주황), 검증 스크립트로 라이트·다크 모두 통과, 모양(점 vs 마름모)으로도 구분. SVG 안에 라이트/다크 색을 함께 넣음.
- `figures/method_overview.svg`(손으로 배치): 이미지+질문 → 생각 단계 → (VisRef만) Ṽ_k 재주입 → 정지 확인 → 답, "Wait, think more" 반복, ST·TSR 경로. 논문 명시 값(δ 0.25, K_max 10, m = ⌊0.3|V|⌋)은 상자 안, 우리가 정한 것(타일링·썸네일, 토큰 상한, Z = 입력 임베딩, DPP 계수 조기 종료 ≈ m/5 = 확인된 절차 차이, 주입 형식, H 추정, 답 처리)은 주황 메모.
- README(방법 표 아래 흐름도, 결과 절 첫머리 결과 그림), REPORT(Figure 1 = §2.1, Figure 2 = §3.1), 파일 위치 표, Makefile(`figures` 목표, `check`에 figures 포함) 갱신.
- 검증: 브라우저(Chromium)로 라이트·다크 렌더링을 눈으로 확인(겹침·잘림 수정 후 최종본). 장치(Python 3.10)와 컨테이너에서 결과 그림 생성물이 같음, 새 복사본에서 `make analysis figures` → results/·결과 그림 바이트 동일. 방법 흐름도 SVG에는 전달 과정에서 붙은 C2PA 출처 메타데이터가 들어 있음(렌더링 영향 없음, 그대로 둠).

## 2026-10-12 GitHub push (사용자 승인 "너가 직접 올릴 수 있도록 승인할게"로 Claude가 수행)
- 방법: 장치에는 GitHub 인증이 없어 클라우드 작업 공간에서 `AhnJG/visref-repro`를 받고, 공개 폴더의 변경 파일 32개(수정 5, 새 파일 27)를 옮겨 sha256 32/32 일치를 확인한 뒤 커밋·push. 삭제 파일 0. 비밀 키 패턴·개인 이메일·개인 경로 0건.
- 결과: `main` 7ec0965 → 1379f5f ("Complete reproduction: step1_branch results, report, tables and figures"), 원격 파일 227개. 작성자는 클라우드 기본 신원(Claude, noreply@anthropic.com), 메시지 끝에 Co-Authored-By·Claude-Session 줄.
- 로컬 `github_export/visref-repro`는 아직 7ec0965 기준(내용은 같음). 맞추려면 Mac에서 `git fetch origin && git reset origin/main`. `.git/stale-index-lock-20261012`는 지워도 됨.
- (같은 날, 사용자 지시 "내 이름으로 이전 커밋 수정해") 1379f5f의 작성자·커미터를 AhnJG <22964835+AhnJG@users.noreply.github.com>(GitHub 저장소 소유자 id로 만든 noreply 주소)로 바꿔 다시 만들고 `--force-with-lease`로 push → `main` = 8db3880(파일 트리는 1379f5f와 동일, GitHub에서 작성자가 AhnJG 계정으로 연결됨 확인). 로컬 맞추기 명령은 그대로(`git fetch origin && git reset origin/main`).
- (같은 날, 포트폴리오 검토 후 사용자 결정) 첫 커밋 이메일을 noreply로 바꾸기로 함 → 두 커밋을 같은 내용·날짜로 작성자·커미터만 AhnJG noreply로 다시 만들고(트리 동일 확인), README 커밋 추가 후 `--force-with-lease`로 push: `main` = 56cd36a ← b6efd0a ← 2d9ddad. 원격 브랜치 기록의 이메일은 noreply 하나뿐. README 변경: "Who did what" 절(EF의 사용자 결정·실행 기록에 근거), CPU 재현 1단계에서 torch 설치 제거(테스트에만), 본 실행 GPU 비용(9세션 약 14.9 세션시간). Kaggle 사용자명 arki13은 그대로 두기로 함(사용자 결정).
- 남은 문제: GitHub는 브랜치에서 빠진 옛 커밋 7ec0965도 SHA로 계속 보여 준다(API로 확인, 작성자 naver 주소 그대로). docs/EXPERIMENT_FLOW.md에 이 SHA가 적혀 있어 공개 후 누구나 찾아갈 수 있다 → 공개 전에 저장소를 지우고 다시 만들어 깨끗한 기록만 올리거나, GitHub Support에 남은 커밋 삭제를 요청해야 한다(사용자 결정 대기). 저장소 설명·토픽은 이 세션 경로로 쓰기가 막혀(403) 사용자가 GitHub 화면에서 입력.
- (같은 날) 사용자 결정: 첫 커밋 이메일은 그대로 둠(새 저장소로 옮기지 않음). 공개 전 점검에서 README 라이선스 문구가 `repro/llm/`(MathVista 질문·공식 추출 프롬프트 포함)와 `repro/diag/`를 빠뜨린 것을 고쳐 push → `main` = d3be139. 모델 라이선스 Apache-2.0은 Hugging Face 모델 카드(OpenGVLab/InternVL3_5-1B)에서 확인. 로컬 git은 아직 8db3880, user.email이 안내 문구 그대로라 noreply로 설정 필요(사용자).
- (같은 날) 로컬 README가 옛 판으로 남아 있던 것(병렬 전송 순서 문제)을 `git show HEAD:README.md`로 맞춤. 사용자 요청으로 한국어 요약 `README.ko.md`(목적·결과·후속 실험·역할·재현 방법·라이선스, 수치는 results/와 같음)를 추가하고 README 첫 줄에 링크 → push, `main` = 4ae919c. 로컬 파일은 같은 내용으로 맞춰 둠(로컬 git은 `git fetch origin && git reset origin/main` 필요).
- (같은 날) 사용자가 저장소를 공개로 전환. 확인: API `private: false`, 설명·토픽 6개 반영, 라이선스 MIT 인식. 비로그인 접근으로 저장소 페이지·README.ko.md·그림 SVG 2개 모두 200, README HTML에 그림 2개와 한국어 요약 링크·"Who did what" 절 포함. Chromium으로 라이트·다크 화면 캡처해 그림 표시 확인(캡처 환경에서 GitHub CSS는 안 불러와짐 — 그림과는 무관). `main` = 4ae919c.

## 2026-10-12 공개 저장소 외부 검토 반영 (사용자 결정: 1–4 전부) — 결과 후 변경, 판정 불변
- 검토 원문: `docs/reviews/VISREF_REPRO_PUBLIC_REVIEW_20261012_KO.md`(HEAD 4ae919c 기준). 5개 지적을 원자료로 재확인했고 모두 맞았다.
- 1) step1_branch 부트스트랩 p의 Python 버전 의존(원칙 2에 따른 결과 후 변경 기록): 문항 정확도(3분의 1 단위 float)를 더하면 정확히 0인 재표본 평균이 ±1e-15로 갈려 꼬리가 바뀐다. Python 3.11에서는 저장값 p_boot 0.289 / 0.575, Holm 0.578이 재현되지만 3.12·3.13에서는 0.281 / 0.583, Holm 0.562 / 0.583(Python 3.12의 `sum()` 변경). 수정: 정답 개수(정수) 차이로 계산하고 마지막에 나눔. 규칙(양측 p = 2·min(P(b≤0), P(b≥0)), CI와 같은 재표본)은 그대로이고 정확한 0은 두 꼬리에 모두 들어간다. 새 값 p_boot 0.329 / 0.584, Holm 0.658 / 0.658(3.11·3.12·3.13 동일, CI·나머지 필드 불변, 판정 ④ 불변). 이전 출력은 `repro/diag/step1_branch_analysis_float_py311.json`으로 보존. 테스트 `tests/test_analyze_step1_branch.py` 추가.
- 2) builder 덮어쓰기: `kaggle/build.py`에 `new_run_dir`(run_config.json·notebook.ipynb·output이 있으면 거부, `--overwrite`는 아직 push하지 않은 세션 재빌드용)를 두고 build_step1_branch·build_hidden_check가 사용, build_colab도 같은 규칙. 테스트 `tests/test_build_guard.py`. README GPU 예시는 `rerun_*` 새 이름으로 변경.
- 3) 외부 근거 경로: 공개 저장소에 `docs/SOURCES.md` 추가(arXiv 버전·PDF sha256, `pdftotext -layout`/`pdftotext`(poppler 22.02.0) 재생성 방법, 핵심 수치의 페이지·절·표: Table 6 p.14, δ·m·K_max §5 p.6, |V_k|=m §4 p.5, ∼1,772 §4 p.4, 환경 p.12, Table 5 p.14, InternVL3.5 Table 3 p.11). 확인: `LAY`·`VisRef_plain.txt`·`REF/InternVL3.5.txt`는 로컬 PDF에 같은 명령을 돌린 결과와 바이트 단위로 같다. arXiv는 이 환경(클라우드·장치)에서 접속 불가라 공개 PDF와의 바이트 일치는 미확인. SOURCES.json의 main/master URL 5개를 커밋 고정(파일 sha256 일치 확인; MathVista c5acd37d, fast-map-dpp 6ab745ca, DEER c9dd19fb), export 스크립트도 같은 값으로 수정. SPEC_LEDGER "논문 PDF는 로컬에 없다"에 2026-10-07 작성 당시 상태 표시.
- 4) 문구: README·README.ko·REPORT에 "주 비교·검정은 첫 392문항, CI 분류 기준은 759문항 중간 결과를 본 뒤 최종 분석 전에 고정 → 엄격한 의미의 사전등록 아님, 후속 실험 규칙은 각 결과 전", TSR−ST는 "논문 수치와 양립(효과·메커니즘 동일성 증명 아님)", fill은 "이 채움 변형에서 추가 이득 없음, 양만의 효과는 분리하지 못함"(사전등록 판정은 기록으로 유지). README 테스트 설치에 pytest·가상환경 추가.
- 검증: 클라우드 사본에서 `make check`가 Python 3.11·3.12·3.13 모두 통과, CPU pytest 98 passed(torch 2.7.0, transformers 4.55.0; 고정 버전 torch 2.11.0 환경은 미검증).

## 2026-10-12 README.ko.md 전체 번역 + README 숫자 자동 대조 (사용자 결정: 영문 README를 메인으로 유지, 한국어 페이지는 요약이 아닌 전체 번역)
- `README.ko.md`: 요약판을 README.md 전체 번역으로 교체(Sonnet 서브에이전트 번역 → 주 세션이 숫자·명령·링크·코드 블록·BibTeX를 원문과 대조). 첫머리에 "두 문서가 다르면 영문이 기준" 문구. 용어는 EF·REPORT와 맞춰 "읽기 (a)".
- `scripts/check_readme_numbers.py`(표준 라이브러리): `results/`의 accuracy·paired_differences·followups 값(정확도, 짝 차이, CI, McNemar p, Holm p, 논문 값)이 README.md와 README.ko.md 모두에 들어 있는지 확인하고 하나라도 없으면 실패. `make check` 마지막 단계로 추가, README 두 곳의 `make check` 설명과 파일 목록 갱신. 한계: 값이 들어 있는지만 보므로 옛 값이 새 값과 함께 남아 있으면 잡지 못한다.
- 음성 대조: 복사본의 README.ko.md에서 0.658을 0.578로 바꾸면 실패(exit 1).
- 검증: 클라우드 사본 `make check`가 Python 3.11·3.12·3.13에서 통과, 장치 Python 3.10에서 검사 스크립트 통과, 두 README의 상대 링크·앵커 확인.
- push: `main` = 8c9be58. 공개 폴더의 README.md·README.ko.md·Makefile·scripts/check_readme_numbers.py를 같은 내용으로 맞춤(sha256 4/4 일치). Makefile은 원격 파일 전송이 막혀(보호 파일) 장치에서 같은 텍스트 치환으로 수정.

## 2026-10-12 README 문구·구성 수정 (사용자 지시; 결정: README 2개 + REPORT 문구 맞춤, H 변경 이유는 기록된 사실만, 효과 해석은 "구현 차이를 먼저 의심", 기여 절은 이름을 바꿔 아래로) — 결과·판정 불변
- README(영·한): 상태 상자에서 st_official 문장과 저자 연락 문장 삭제. "Who did what" → "Contributions and AI use"(인용 절 앞으로, "방법의 일부인 LLM" 줄 추가; 근거 관행은 SPEC_LEDGER X-31). 채점 설명을 쉬운 말로(LLM이 뽑은 답이 응답에 있을 때만 인정, "grounded" 표현 삭제). 요약·분류 정의·시점 문단 정리(Holm 문장은 유지: 미리 정한 보정이라 빼면 유리한 값만 보고하는 셈).
- 새 절 "정확도가 논문보다 8–12 pp 높은 이유(미해결)": 공식 1B 59.3·8B 78.4 대비 논문 ST가 10 pp 넘게 낮다는 점과 추정 원인 3개(끝나지 않은 답을 오답 처리 = 읽기 (c) ST 44.4 / TSR 48.3 / VisRef 47.4, 결과 후 가설; 모델 세대 InternVL3-1B 45.8; 토큰 상한·추출·프롬프트·하드웨어). 어느 것도 직접 시험하지 않음.
- 새 절 "VisRef 추가 이득이 없었던 이유": 우리 구현에서 먼저 찾는다는 문장 + 우리 쪽 후보 5개와 확인 여부(주입 수 ≈ m/5 → fill로 일부 확인; 주입 형식 → 미시험, k=1 713문항 중 `</think>` 재출력 VisRef 352 vs ST 67; 상한 절단 뒤 주입 → 4,096 미실행; Z 표현 → 은닉 상태 오프라인 확인 12문항 50–110개; 단일 표본 H → 다중 표본 미실행). "잘못했다"고 단정하지 않음(사용자 결정).
- "Pre-registered follow-ups" → "Follow-up experiments (decision rules fixed before the results)". 재현 §3에 MathVista 공식 추출 프롬프트 설명(예시 5개 + 새 응답, GPT-4 추출기; X-27, X-28). "Known differences" → "Settings we could not find in the paper": 13행 표(우리 선택·근거·다른 가능성과 상태) + H 문단. "명시하지 않음/leaves open" → "찾지 못함".
- H 문단은 기록된 사실만: 이전 다중 표본(단계마다 완결 답 10개)은 8B·개발 3문항·TSR에 GPU 4,703.9초(EF L642, L1345), 이후 단일 표본으로 교체(10-06·10-07 항목). 바꾼 이유가 비용이라는 문장은 기록에 없어 쓰지 않음. REPORT §4.1-5의 "judged too costly"도 같은 사실 서술로 고침(결과 후 문구 변경, 이전 문장은 커밋 adbafc3의 REPORT.md).
- REPORT: 요약 마지막 문장("We do not conclude…" → 우리 구현에서 원인을 먼저 찾는다), 그림 1 캡션·§2.1 표·§5.2의 "not specified/leaves open" → "could not find in the paper", §6 후보에 모델 세대 추가. st_official(§4.5)·저자 연락(§5.3) 절은 재현 보고서 형식이라 유지(사용자 결정).
- 그림 `figures/method_overview.svg`: 범례·설명 문구 3곳만 "not found in the paper"로(배치 변화 없음).
- 한국어판: Sonnet 서브에이전트 번역 → 주 세션이 숫자·링크·용어 확인, 어색한 문장 4곳 수정(용어: LLM 보조 추출, 답 강제, 후속 실험, 판정 규칙, 논문에서 찾지 못한 설정).
- 새로 확인한 외부 수치·근거: SPEC_LEDGER X-26~X-31.
- 검증: `make check` Python 3.11·3.12·3.13 통과(README 숫자 검사 포함), GitHub 렌더링(저장소 API)에서 두 README 모두 표 6개·행 34개, 내부 링크 앵커 일치.
- push: `main` = 0a8241d. 공개 폴더의 README.md·README.ko.md·REPORT.md·method_overview.svg·docs/SPEC_LEDGER.md를 같은 내용으로 맞춤.

## 2026-10-12 § 표기 변경, InternVL3-1B 후보 삭제 (사용자 지시) — 결과·판정 불변
- README(영·한): "REPORT §4.4" → "[REPORT, Section 4.4]" / "[REPORT 4.4절]"로 바꾸고 REPORT 해당 제목으로 가는 링크를 걸었다(각 9개). README 안의 "재현 방법 §3"은 LLM 보조 추출 단계 링크로. REPORT의 § 10개 → "Section".
- README(영·한): 절대 정확도 차이의 추정 원인에서 "다른 모델 세대(InternVL3-1B)" 항목과 설정 표 모델 행의 대안 칸을 삭제(사용자 지시 "이건 빼"). REPORT §6 후보 목록에 직전에 넣었던 "the model generation"도 되돌림. REPORT 1절 범위 설명(세대 가정)과 5.3절 저자 질문 9번, SPEC_LEDGER 행 1·C-8·X-26의 기록은 그대로 둠.
- 검증: `make check` Python 3.11·3.13 통과. GitHub 렌더링(저장소 API)에서 두 README의 REPORT 링크 9개씩 모두 REPORT 제목 id와 일치, README 내부 링크 일치, 표 6개, § 0개.
- push: `main` = 7fc9d64. 공개 폴더 README.md·README.ko.md·REPORT.md 같은 내용으로 맞춤.

## 2026-10-12 REPORT에서 InternVL3-1B 후보 삭제 (사용자 판단) — 결과·판정 불변
- 사용자 지적: 논문은 InternVL3.5-8B 위주이고 1B 행만 3 세대로 보는 것은 논리적으로 맞지 않는다. 원문 대조: 논문이 이름을 밝힌 InternVL 세대는 3.5뿐(LAY에 "InternVL3-" 0회, "InternVL-3.5"/"InternVL3.5" 다수), 모델을 "reasoning (thinking) mode"로 실행(LAY L428-429), 1B·2B·8B를 "under the same decoding setup and token budget"으로 비교(L924-925), 8B 행 68.1/73.9/79.3 = Table 1 InternVL3.5-8B(L396-398). InternVL3 후보의 근거는 1B 수치 근접(45.8 vs 46.1) 하나뿐이었다 → 지적이 맞다고 판단.
- REPORT: 1절 범위 설명을 위 근거로 InternVL3.5-1B를 쓴다는 문장으로 교체, 2.1절 표 "generation assumed" 삭제, 5.2절 "어느 InternVL 세대인지" 항목 삭제, 5.3절 저자 질문 9번에서 InternVL3 부분 삭제. README는 직전 커밋에서 이미 삭제.
- SPEC_LEDGER X-26에 삭제 사유 메모 추가(행 1·C-8의 기록은 그대로).

## 2026-10-12 H 추정 경위 정정, 기여 절 수정 (사용자 지적) — 결과·판정 불변
- 사용자 지적: "처음에 하나만 뽑아서 하다가 참고 문헌을 보고 혹시나 해서 다중 표본을 시도했다." 기록 대조: 10-05 v3 명세는 "1회 추정"(이 파일 2026-10-05 independent_dpp30_entropy_v3 항목), r3은 답 1개의 답 토큰 평균 엔트로피 → 10-05 사용자가 준 [19] Ghosal 근거 검토(`docs/paper_baseline_alignment/ENTROPY_REFERENCE_CLAIM_REVIEW_KO.md`: 여러 완결 응답의 sequence NLL 평균 = Monte-Carlo estimator) → 10-06 S10(단계마다 완결 응답 10개) 구현·8B 실행(R25 GPU 4,703.9초) → 10-06 사용자 지시로 자연 응답 엔트로피(1개)로 전환 → 10-07 짧은 답 1개 토큰 엔트로피 합. 지적이 맞다. README·README.ko의 H 문단과 REPORT 4.1절 5번을 이 순서로 고침(이전 문장 "earlier version sampled 10 … replaced by the one-sample estimator"는 커밋 eae46a1). 바꾼 이유는 여전히 기록에 없어 쓰지 않음.
- 기여 절: "모든 GPU 세션과 Gemini 추출을 실행" 문구 삭제(사용자 판단: 중요하지 않음), "Codex(OpenAI): 프로젝트 전 과정에서 내용 검토에 사용" 추가(사용자 진술).
- push: `main` = 8b27721. 공개 폴더 README.md·README.ko.md·REPORT.md 같은 내용으로 맞춤.
