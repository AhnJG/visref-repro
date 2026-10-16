# VisRef 실험 기록: 초기 단계 원문 (2026-10-05 ~ 2026-10-07 오전)

> 이전 코드(`src/visref`, 이 저장소에 없음)로 실행한 초기 단계 기록입니다. 2026-10-16에 [EXPERIMENT_FLOW.md](EXPERIMENT_FLOW.md)에서 옮겨 왔고, 당시 적은 글을 그대로 두었습니다(영어와 줄임말이 많습니다). 이 구간의 결과는 이후 판정의 근거로 쓰지 않습니다. 한국어 요약과 줄임말 풀이는 [EXPERIMENT_FLOW.md](EXPERIMENT_FLOW.md) 앞부분에 있습니다.
> 줄 번호: 이 파일의 N번째 줄은 정리 전 `EXPERIMENT_FLOW.md`의 (N − 1)번째 줄입니다.

## 연구 질문

동일한 visual reinjection 절차에서 중요한 원본 visual token을 보존하고 나머지를 pooling하면, DPP 30%보다 적은 token으로 정확도를 유지하고 실제 추론 비용을 줄일 수 있는가?

이 질문을 평가하려면 먼저 Thinking mode, reasoning lifecycle, 채점, entropy와 reinjection 처리가 검증된 baseline이 필요하다. 현재 그 조건을 충족하지 못했으므로 개선 실험을 중단했다.

## Baseline과 평가 계획

Pilot 모델은 InternVL3.5-1B다. MathVista testmini를 seed 42로 development 100 / selection 200 / locked evaluation 700문항으로 분리했다. 분할 checksum을 보존한다. Selection에는 실행 이력이 있으며, 잠금 700문항은 모델 추론과 성능 분석을 하지 않았다.

당시 protocol은 고정 3-step, greedy decoding, 자체 엄격한 답 추출기를 사용하는 operational 절차였다. 공식 entropy estimator와 재주입 fidelity는 미확인이었다. 과거 protocol은 이 실행의 설명이며 논문 재현 protocol이 아니다.

## 변경·결과·판단

| 단계 | 변경 또는 검증 | 실제 결과 | 판단과 다음 방향 |
|---|---|---|---|
| 로컬 correctness | Kernel factorization, greedy indices, pooling, split·scoring·statistics 검증 | 당시 30개, ablation 회귀 검증 추가 후 32개 테스트 통과 | Synthetic correctness이며 모델 성능·공식 재현 증거가 아님 |
| P0 v1 | 실제 모델 경로 검증 시작 | NumPy/선택적 TensorFlow import 문제로 모델 로딩 실패 | 구현·의존성 실패. 압축 가설 실패로 해석하지 않음 |
| P0 v2 | NumPy 고정 및 TensorFlow/Flax 비활성화 | 개발 첫 10문항 PASS | Native 초기 8-token 및 자체 identity append 일치. 전체 thinking 종료·최종 답변 검증은 아님 |
| P01 v1 | ST·TSR·DPP30 개발 평가 | Autograd 추적 발견, 95개 부분 행에서 중단 | 비용 평가 무효화. 결과와 실패 이력을 보존 |
| P01 v2 | Model freeze와 inference mode 적용 | 100문항 × 3방법, 300행 완료 | Operational 결과를 회수·분석. 정확한 VisRef 재현으로 인정하지 않음 |
| P2 v1 | 9개 구현 방법 비교, 동일 keep-index replay ablation | P0 10문항·개발 smoke 90건 통과 후 selection 155/1800행에서 작성자 요청으로 중단 | 후보 선택에서 제외. 새로운 실험의 튜닝 근거로 사용하지 않음 |
| Fidelity 검토 | 공식 모델카드·논문과 현재 구현 대조 | Thinking System Prompt 누락 확인; step·Z 범위·채점·entropy·reinjection 조건 미확인 | 모든 개선 실험 중단. baseline 재현을 먼저 검증 |
| 폴더 정리 | 루트 중심 구조와 자료·결과 분리 | 이전 전 SHA와 경로 대응표를 보존 | 실험 재개나 모델 방법 변경이 아님 |

## P01 v2의 관측 결과

| 방법 | N | Accuracy | 평균 end-to-end latency | 답 추출 실패 |
|---|---:|---:|---:|---:|
| ST | 100 | 40% | 9.869초 | 26건 |
| TSR | 100 | 40% | 17.453초 | 27건 |
| DPP30 | 100 | 30% | 19.846초 | 34건 |

조건과 표본이 달라 논문 수치와 직접 비교하지 않는다. 1B에서 효과가 없다고 결론 내리지 않는다. Thinking mode 누락과 출력·채점 문제 등이 섞여 있으며, 어떤 차이가 하락을 일으켰는지는 확인되지 않았다. Kernel 계산식의 동등성 검증으로 전체 baseline fidelity가 증명되는 것도 아니다.

- P01 v2 원본 상태
- 분석 결과표
- 샘플별 원시 결과
- P2 후보 선택 제외 기록
- 누적 상세 로그

## 현재 결론

논문 baseline이 검증되지 않은 상태에서 개선 후보 비교를 먼저 진행한 것은 순서가 잘못됐다. 현재 결과는 operational 진단 자료이며 VisRef 재현 실패, 압축 가설의 성공·실패, 후속 방법의 우수성을 주장하는 근거로 사용하지 않는다.

로컬 controller 종료와 Kaggle P2의 CANCEL_ACKNOWLEDGED를 확인했다. 신규 dispatch는 중단 marker로 차단한다. 실패·중단 실행의 GPU 사용량을 누적 자원 한도에서 제외하지 않는다.

## 다음 단계

공식 근거와 현재 구현을 항목별로 대조하고, Thinking mode와 완전한 native ST 생성·채점을 우선 검증한다. 이후 TSR, DPP의 step·Z·entropy·재주입 조건을 확인한다. 미확인 조건은 BLOCKED로 기록한다. 검증된 재현 baseline과 새 protocol이 확보되기 전에는 개선 실험·후보 선택·잠금 평가를 재개하지 않는다.

## 2026-10-05 baseline_v1 로컬 구현·검증

논문·공식 native model·MathVista source를 대조한 fidelity 표, 새 protocol, blocker 상세를 추가했다. 과거 operational 코드/설정/결과는 보존하고 별도 Thinking/native full generation adapter와 versioned scorer, DPP 수학/control components를 구현했다.

최종 local run `20261005_001921_local`에서 새29+기존32=61 tests PASS. 이는 합성 full lifecycle/공식 source parity·수학 correctness이며 실제 InternVL3.5-1B inference 또는 논문 재현은 아니다. ST 모델 검증은 미실행, TSR/VisRef end-to-end는 official entropy/step/Z/V/reinjection/stabilization/scorer 확인 전 BLOCKED다. Table1 ST adaptive criterion도 별도로 미확인이다.

한국어 결론에 성공 수준과 범위를 구분했다. STOP/manifest/historical outputs SHA를 유지했고, 모델/remote/GPU/P2/selection/locked evaluation/개선 실험/후보 선택은 실행하지 않았다. 다음 단계는 공식 missing mechanics 확인과 별도 승인된 native ST complete-path GPU validation이다. GPU 예산은 실패·취소 포함 기존 기록 합산2.967115h. baseline 검증 전에 후속 압축 비교로 넘어가지 않는다.

최종 local validation은 전처리 fixture 보강 후 `20261005_002747_local`이다. 61 passed, prior run은 보존. 현재 결론과 baseline protocol은 이 source/config snapshot을 참조한다.

## 2026-10-05 독립 regularized DPP30 명세 반영

명세 검토 보고서를 source·수학·합성 진단과 대조하고 독립 protocol에 수정 후 반영했다. 논문 재현의 미확인 entropy 등은 그대로 두고 독립 경로의 고정2-block/Z/V/상대regularization/주입·실패 처리를 명명했다.

Selection undefined의 주입 생략은 실패로 변경했다. λ 변화에 대한 선택 불변·unregularized prefix 일치·pivot 기반 rank를 일반 보장으로 채택하지 않았다. Shared-cache 생성 비용과 실제 단독 latency도 분리했다. Controller 구현·실제 InternVL 생성·정확도·speedup 증거는 없다. 이번에는 문서와 기존 합성 진단 재확인만 수행했으며 source/config/원시run/STOP를 유지했다.

다음 단계는 별도 지시된 로컬 controller/selector/recorder 구현·LC checks, 그 뒤 별도 승인된 native→reflection→DPP validation이다. 개선 비교는 사전 동결된 새 계획과 별도 승인 전까지 보류한다. 동일 논문 fidelity 질문은 새로운 공식 근거가 없으면 반복 검토하지 않는다.

## 2026-10-05 baseline 실제 검증 재개 범위

작성자가 최대2 Kaggle kernel을 승인하여 독립 baseline correctness를 개발 첫3에서 검증한다. 실행 계획, 기록. G1 dispatch 완료, G2는 실제 실패 수정/재검증용으로 남긴다. 기존 STOP/old pipeline·selection/locked·후속 압축 실험은 유지한다. 추가 일반 검토보다 raw native lifecycle/cache/selector 증거로 다음 판단을 한다.

### 두 kernel 종료 후 결정

결과 보고서: native full-generation parity/종료와 첫 DPP append가 개발 `243`에서 확인됐고, fixed2 TSR/DPP는 자연 종료 뒤 WAIT가 빈block2를 만들어 실패했다. 가설 자체의 구체적 문제를 확인했다. λ/source-order/position/scorer는 유지하고 종료 정책 하나만 max2_v2로 분리했다. Local76 PASS/GPU UNVERIFIED, v1 raw는 보존. 다음 증거는 새로운 일반 검토가 아니라 v2 개발 first3 correctness다. 이번 최대2 kernel은 모두 사용·종료했고 추가GPU/selection/locked/압축 후보실행 없음.

## 2026-10-05 independent_dpp30_entropy_v3 명세·로컬 구현 (GPU 미실행)

max2_v2 변경 검토에서 v2가 자연 종료 시 TSR=ST, DPP 답변 직전 1회 주입이 되어 논문 종료 조건과 크게 다름을 확인했다. 그래서 작성자 결정에 따라 가정한 entropy 종료(H<0.25 nats, K_max=10), "Wait, think more" 연장, L_min=1, 모델 최대 토큰 상한을 쓰는 별도 v3 명세를 만들고 로컬 구현했다. TSR과 DPP의 차이는 visual 주입 하나로 두었고 v1/v2 경로·원본 결과는 보존한다.

주장 범위: CPU 로컬 86 passed(기존76+신규10, tiny random Qwen3)는 명세·구현·fail-closed 검사의 확인이며 실제 InternVL/GPU 증거가 아니다. Entropy estimator는 독립 가정(논문 estimator UNVERIFIED), 1회 추정이라 분산이 크다. 정확도·speedup·논문 재현 주장은 없다. STOP·selection/locked·후보 비교 차단은 유지되고 GPU 실행은 미승인(dispatch 0)이다.

다음 단계는 작성자 승인 후 kernel 1개로 개발 첫3(243,479,419) v3 correctness 검증이다.


## 2026-10-05 v3 runbook 실행: 전송 전 네트워크 중단

v3 correctness는 원격 전송 이전 인증 DNS 오류로 UNVERIFIED다. Summarizer verdict 없음. Runbook의 1회 push 규칙에 따라 재전송하지 않고 종료한다. 추가 실행에는 작성자 결정이 필요하다. 실행 보고서.


## 2026-10-05 v3 kernel version1 종료

v3 원격 gate는 FAIL(SETUP_FAILED). 로컬93 통과와 전송 bundle의 실제 독립 실행은 달랐다. Scripts import 의존성 누락으로 모델 미실행; entropy/controller 가설 판정 근거 없음. 다음 후보는 새 version bundle 일치 검증이며 추가 수정/실행은 하지 않는다. 결과 보고서.

## 2026-10-05 v3 r2 실행 준비 완료

scripts 누락과 자동 판정 결함을 수정했다. 최종 전송본을 별도 폴더·프로세스로 검증하여113 tests PASS, 기존67파일SHA 보존. Entropy/controller 모델 정책은 그대로다. 준비 보고서.

현재 로컬 준비는 완료, actual InternVL correctness는 UNVERIFIED다. 기존 kernel1 승인 소진, r2 승인 파일 없음/dispatch0. 다음은 추가 일반 검토가 아니라 새 승인된 개발3문항의 실제 경로 검증이다. P0–P1/total 예산은 실패 차감 후 기존값 유지, STOP·selection/locked·비교 차단 유지.


## 2026-10-05 v3 r2 단일 kernel 전송

r2 private kernel `arki13/visref-entropy-v3-r2-g1`, version1을1회 전송했다. 승인 포함 bundle113 PASS. Frozen source SHA `f5dfdc4f5c3121bddcef4b884d6c8c89931ef8d4a6e29b71085acf46b5f135c6`. 개발243/479/419, T4/fp16,3600초/P0–P1. 실제 환경·native/ST/TSR/DPP gate는 실행 결과 대기다. 원본: `artifacts/runs/independent_entropy_v3_r2/g1/launch.json`. 재전송·추가kernel 없음.


## 2026-10-05 v3 r2 단일 GPU 실행 종료: FAIL

r2는전송·remote113 tests·실제native/ST/TSR 실행까지진행했고, 첫243 DPP 뒤CLOSE cache/full parity TV3.87%가고정2%를초과해FAIL이다. DPP 주입자체는통과했으며ST/TSR는1step정상완료했다. 결과 보고서.

다음은entropy/선택정책 변경이아니라, 동일243 block/selection의DPP CLOSE 고정입력replay로KV/position/mask와fp16수치차이를구분하는진단이다. 현재원인확정·전체baseline통과·성능개선 근거없음. 승인소진, 추가GPU0, STOP/selection/locked/후속비교차단 유지. 사용257.098초; 후속새version은r2원장을부모로사용한다.

## 2026-10-05 r2 원인 분석 보완

실패 분석은 캐시 오류/fp16 원인을 UNSURE로 유지한다. 보정 없는2%와 top-k 분포 불연속 때문에 TV 초과만으로 버그를 확정할 수 없다. 다음은 같은243 입력·indices·chunk를 유지한 fp16/fp32 구조·분포 진단이다. 기준 완화나 전체 생성 재시도는 원인 구분을 대체하지 않는다.

Entropy는 마지막에 반복한 답 토큰만 측정해 confidence와 다른 양을 보는 문제도 확인했다. Cache 진단 뒤 별도 estimator 명세·보정으로 다룬다. 기존 결과는 변경하지 않으며 추가GPU0, STOP/selection/locked/비교 차단 유지.

## 2026-10-05 다음 실행 계획 r3

r3 계획을 작성했다. 앞선 fp32 replay 우선 권고를 보완하여 fp16 실행의 구조·정상종료 검사와 원인분리 진단을 구분한다. 다음 작업은 r3 CPU 구현·전송본 검증이며, 이후 새 승인된kernel1에서같은개발3문항의DPP 답까지관측한다. TV2%는경고로사전명시하고더큰실패threshold를만들지않는다. 기존FAIL은새기준으로재판정하지않는다.

Entropy 정책은이번실행에서유지하고알려진한계를경고한다. 실행9경로완료라도baseline 비교준비완료는아니다. GPU 승인·dispatch0, STOP/selection/locked/비교차단유지.

## 2026-10-05 r3 entropy 계획 수정

위 estimator 유지 권고는 철회했다. 현재 실행 계획은 answer_only_sequence_probe_v1을 구현·검증한 뒤 실행하도록 수정했다. Probe에서 설명/답 재반복을 허용하지 않고 전체 draw+EOS를 측정하며 최종답은 별도로 생성한다. 모델/reasoning prompt·DPP·sampler·scorer는 유지한다. 초기0.25 보정은 미검증이다. 현재 코드·GPU 변경은 없다.

## 2026-10-05 r3 최종 답 토큰 mean entropy 계획 확정 (문서만)

현재 실험 계획을 작성하고 r3 runbook에 연결했다. 위 sum+EOS 초안은 보존하며 현재 계획은 answer-only probe의 유효 답 값 token만 선택해 raw T=1/full-vocabulary entropy를 평균한다. EOS·바깥공백은 별도 기록, sampling entropy·sum은 진단, 최종답은 별도 생성이다.

현재 개발3은 이미 노출된 correctness 표본이며 δ=.25는 미보정 초기값이다. 다음은 해당 명세의 CPU 구현·내장bundle 검증이며, 그 준비 결과가 확인된 뒤 새 GPU 승인 범위를 제시한다. 이번에는 문서만 수정했고 코드/config·STOP·r2원본·예산은 유지한다. 논문 동일성/정확도 우열/잠금 평가 차단은 유지한다.

## 2026-10-05 r3 mean entropy CPU 구현·전송본 검증 완료

준비 보고서: 별도r3 config/decoder/controller/recorder/summary3와builder run-version을구현했다. 답 값token의raw T=1/full-vocabulary entropy평균, EOS/외부공백제외, strict probe와별도final생성을검증했다. 기존113 tests와신규31을합한144 tests가Transformers4.55.0 CPU에서통과했고, 실제notebook 내장archive의독립프로세스에서도144 PASS다. Aggregate cc602534d5f59c38cef450f0e0b38aad7f0872a52e789d450266011e5c2ac31f.

이는synthetic/tiny Qwen CPU correctness와packaging 확인이다. 실제InternVL/native9경로·fp16 원인·H보정·논문동일성은미검증이다. 기존r2/STOP/원장 등49파일SHA변경0, 신규APPROVAL/dispatch/GPU0. 다음은새r3 kernel1개/개발3/3600초 승인후승인포함재동결·검증·predispatch다. 이전r2승인은재사용하지않는다.


### 2026-10-05 r3 단일 GPU 실행 승인·전송

작성자가 준비된 r3 개발243/479/419×ST-K1/TSR/DPP, private T4 kernel1개·최대3600초를 명시적으로 승인했다. 승인과 launch를 기록했다. 승인 포함 독립 bundle144 tests PASS; source SHA `7e755613a397c1dd5aa0613fc557c4521031dc5ab0e8981c47e504ef41c3a818`, notebook SHA `48378d7d361a3dc9bab09f4f9067ad5eca8fd26ca5ecc9bba3e966a9ad34dd8a`. Kaggle `arki13/visref-entropy-v3-r3-g1` version1을1회 전송했다. Parent r2 원장·STOP·과거결과 보존, selection/locked/후속비교 허용 없음. 결과·회계는 아직 미확정이며 종료 후 회수한다. 사전 동결 문서의 미승인 표기는 준비 시점 기록이고 이번 승인으로 이 범위에 한해 대체된다.


### 2026-10-05 r3 결과와 다음 판단

r3 결과: **FAIL**, kernel1/version1종료. 원격144tests·native243/479는통과했으나, answer-only probe13개가전부형식/종료실패로H undefined였다. 243TSR timeout 뒤479DPP도연장했고block8 length7741 full parity진단에서OOM. 완료arm0/9이며419미실행. TV경고2개는계속진행했고실패원인이아니다. 연장은부분관측이지만semantic불확실성을측정한결과가아니며성능비교준비는false다.

현재우선순위는threshold/DPP비율/fp32변경이아니라**실제최종답형식에서유효한답token span을얻는측정경로**, 긴문맥full진단의메모리부담, timeout partial기록을CPU에서해결하는것이다. 보존13probe/native 출력으로새parser명세·테스트를고정하고, 이후새실제모델smoke를별도동결한다. R3원본/verdict/동결본은변경하지않는다.

비용1611.595초/setup·실패포함차감, 최신 원장. 보호49파일변경0, selection/locked접근없음. R3승인소진/후속GPU미승인/STOP보존. RUN_STATE, summary3, 증거·SHA.


### 2026-10-05 r3 실패 원인 분석 보완: query 문구 유지 권고

실패 원인·query 유지 분석에서 저장 raw/code를 대조했다. Native는 이미 dataset query 문구 그대로(이미지/chat/공식Thinking 설정 유지, 추가answer prompt/forced close 없음)243·479 EOS와추출을통과했다. Query의 "at the end"는 숫자-only 제약이 아니다. R3 strict whole-output 검사와H undefined→reflection 연결이13개probe를측정불가/연장으로만들었다. 추가문구는assistant몸체에append됐으나문구유무만바꾼인과대조는없다. Native 답entropy도기록없어query-only H/.25결과는미검증이다.

권고는새독립명세에서query를재작성하지않고추가probe문구를제거한평소답변생성의raw token entropy를기록한뒤, 최종답값의유효span만선택하는것이다. 설명/라벨/반복/EOS를제외하고첫숫자를무조건선택하지않는다(479답영역은56/22부터시작). 최초답값이라도생각중이미같은값을언급할수있어semantic confidence/δ보정차단은유지한다. Undefined는측정오류로구분하며연장fallback을새조건으로명시해야한다.

OOM은479 block8 length7741의full-context parity 진단에서발생했다. Cached next-token logits로H를계산할때매번full parity가필수인것은아니다. 진단메모리범위/분리replay와partial기록이별도해결대상이다. Armshuffle·문항시간몫이ST대조군을가린점도correctness smoke 설계에서검토한다. 이번은CPU기록분석이며코드/동결명세/verdict/원장/승인/STOP변경·모델실행없음. 권고는아직구현·채택된r4가아니다.


### 2026-10-05 추출값 첫 등장 토큰 선택 규칙 구현

`first_extracted_answer_mean_token_entropy_v1`을 새 모듈로 구현했다. 추출된 모델 답값을 기준으로 생각 영역 밖의 첫 유효한 등장에 대응하는 원래 생성 토큰을 선택하고 raw T=1/full-vocabulary H를 평균한다. 다른 숫자·뒤의 반복·EOS는 제외하며 정답/precision 반올림은 사용하지 않는다. 첫 토큰이 설명과 섞이거나 정렬이 모호하면 뒤의 반복으로 대체하지 않고 측정 불가다.

채택 명세, 구현·검증 보고서: 새36/전체180 CPU tests PASS, 저장 native243/479의 첫 값 문자 위치 확인. 실제 InternVL token alignment/H는 미검증이다. 기존 r3 runner/config/동결본/verdict는 보존했고 새 함수의 연결은 다음 실행 버전에서 한다. Threshold·baseline readiness·GPU 승인·예산·STOP 상태는 변경하지 않았다.


### 2026-10-05 r4 연결 구현 및 Kaggle quota 추출 검토

생성 시 raw token entropy 기록→공식 MathVista 답 추출→생각 밖 첫 유효 등장 선택→평균을 별도 r4 runner에 연결했다. Production full parity는 구조 검사로 바꾸고 short full validation은512 positions 상한의 별도 함수로 분리했다. Undefined는 측정 실패로 종료하며 추가추론 근거로 쓰지 않는다. 새21/전체201 CPU tests PASS(16.00초); 실제 InternVL/추출 API/GPU 실행0. 준비 보고서, 명세, SHA·검증.

Kaggle AI quota 검토는 quota 모델을 답 추출에 쓰고 MathVista scorer로 정오 판정하는 방향을 권고한다. 먼저 저장된 개발 응답의 추출을 GPU 없이 검증하고 이후 실시간 adaptive stopping에 연결한다. 실제 quota/허용 모델/일반 GPU notebook 접근 미확인, Kaggle adapter 미구현, provider/model/revision/SDK 미결정이다. R4는 로컬 구현 준비이며 실행 bundle 미완성·GPU/API 승인 false다. R3 FAIL/원본/원장/STOP 보존, paper/threshold/baseline 비교 차단 유지.


### 2026-10-05 네 논문 대조와 다음 baseline 실행 계획

논문 대조와 다음 실행 계획을 추가했다. VisRef Appendix C/Eq1에 따라 평가 ST는 추가개입 없는native로삼고controller st_k1을분리하는명세를채택한다. MathVista의답accuracy와generation/entropy gate를분리한다. 1B모델실험은Appendix I/Table6에있으며14K는parallel-trace budget, paperfp16명시는미확인이다.

새 paper_baseline 모듈에서대각항없는Eq6–11 kernel/greedy·rank bound·공식함수기반text scoring을구현했다. 새10/전체211 CPU tests PASS(16.19초). 저장r3 PID479의7단계모두U45–109<m537이므로currentinputZ의unregularizedfixed-m determinant는퇴화한다. 기존정규화DPP는명명된독립구현으로만유지한다. 증거, rank기록.

다음은새runner의native ST/공통scorer/단계별시간정책연결→GPU없는quota최대6회검증→별도승인후단일T4개발smoke다. 제안3600초이며실제dispatch승인은아니다. 새모듈runner미연결/transport·model미결정, 계획config는실행용아니다. STOP/원장/기존r3·r4source·PDF/selection/locked 보존, actual모델/API/GPU0.


### 2026-10-06 6시간 자율 baseline 개발 시작

작성자 Kaggle 사전 승인과 기존 프로젝트 GPU 상한 확대를 기록했다. 프로토콜. Native ST→TSR response MC(S10)→strict DPP 진단을 새 version으로 구현·검증한다. author 미공개 설정의 관행 조사와 operational 가정을 구분하고 STOP/selection/locked/압축 금지는 유지한다. CPU22 PASS; 실제 새 GPU/API inference는 아직0.


### 2026-10-06 r5 actual GPU / source and scorer update

r5 exact evidence: native ST243/479/419 all normal EOS, official chat replay/first16 sampler/short48 numeric gates PASS. TSR243/479 TIMEOUT at step5/6 (500s each);419 step3 incomplete MC set after one512-token repeating response. DPP NOT_RUN due TSR stage failure. GPU charged1732.5s; remote COMPLETE means notebook ended, not implementation gate passed.

Kaggle extraction preflight: public notebook2-shot and officialMathVista5-shot agree on two historical final answers; 5 calls including transport conformance. This verifies proxy transport/value agreement only. Proxy model alias is not immutable and canonical leaderboard scorer equivalence remains unverified. Gold was used only for subsequent local normalization/scoring, not outgoing extraction prompts.

Official CVPR2026 main/supplement and NeurIPS2025 Mirage final checked. No additional S/representation/degeneracy settings found in those readable texts. Mirage checklist explicitly says code would be submitted as supplementary material; OpenReview API returns403 challenge, so supplemental author code is unread and cannot be claimed absent. Chen's referenced greedyDPP code uses epsilon1e-10 early stop with no jitter; fewer-than-m cardinality would be a named alternative, not a paper-fixed-cardinality release. R6 preregistration changes time allocation only plus malformed-final correctness repair; prior limits, STOP and frozen historical evidence preserved.


### 2026-10-06 r5 offline verification and r6 dispatch

R5 stored110 complete responses (11 S10 sets) reproduce every recorded MC entropy exactly, including EOS and excluding length normalization. Evidence. Diagnostics show different units/scales for sequence NLL and mean-token NLL; no threshold was changed.

Real243 embedding audit: N768/T1315/U204/requestedm230; repeated token IDs have identical actual stored input embeddings. Strict selector degenerates, referenced Chen code epsilon1e-10 returns204 on raw and diagnostic scaled kernels. This confirms current representation rank issue, not author Z. Chen paper §3 and numerical-stability appendix explicitly discuss rank-limited returns.

R5 final scoring completed: publicKaggle2-shot and officialMathVista5-shot normalized1/2.55/quarterpast identically, allthree incorrect against development references. Six calls ; inference and original generation status unchanged; failed/notrun finals retain denominator1. No performance inference from dev3.

R6 private notebook arki13/visref-paper-r6-20261006 v1 dispatched after228 isolated CPU tests. FreezeSHA dcbfee9f9a5ebb60b07930e14583147dc11b1e33cd2991cd0e2fce44527b47c4; same samples/S/caps/sampling/threshold, increased time allocation as preregistered. Current status running; author fidelity and long numerical/reinjection gates remain unresolved.


### 2026-10-06 참고 알고리즘 및 설정 관행 추가 검증

Chen 공개 greedy DPP 원본(commit6ab745ca1273941412a7f32ce6432549cfe4b5af, epsilon1e-10)을 원 kernel에 그대로 적용하는 별도 component와 tiny Qwen3 실제 재삽입 검사5건 PASS. 현재 전체 로컬237건 PASS(12.20s); 준비된 R7 remote bundle의 검증 수는232이며 새 component는 포함/연결/dispatch되지 않았다. 요청m와 실제m를 따로 기록하고 fixed-cardinality author blocker를 유지한다. source: src/visref/reference_dpp.py, reference_dpp_runtime.py; pinned reference references/sources/chen_dpp_2018.

현재 공식 VLMEvalKit Thinking-1B 설정(commit54a063c5bf75245c35599daab9826d0a5b8e2671)은 LMDeploy/do_sample/max_new_tokens65536이다. 이는 현재 관행 근거이고 VisRef 당시 설정의 증거가 아니다. 현재 모델 context40960 및 Transformers eager 구현에는 그대로 옮길 수 없다. 정확한 근거. InternVL3.5 원 논문2508.18265v2 §2.4/3.2도 읽었으며 Thinking/BoN 설명은 있지만 해당 두 section에는 S/entropy 정규화가 없다. Mirage OpenReview는 웹에서도 browser verification으로 차단되어 supplemental author code는 계속 미확인이다.

새 단위별 offline audit는 sequence/normalized 표준오차를 구분하여11개 완결 S10 set을 재계산했다. R7 frozen controller의 standard_error 필드는 sequence 단위 그대로임을 기록하고, normalized 단위 진단은 이 audit의 명시된 필드를 사용한다. 제어 value/threshold는 변경하지 않는다.


### 2026-10-06 r6 종료 / r7 시작 / r8 자원 진단 준비

R6 native3 normalEOS 및 기존 단축 correctness gates 통과. TSR243 step7 total4096에서 thinking cap 실패,479 K10/총2270tokens 완료,419 step3 불완전 S10 set 실패. DPP3 NOT_RUN. 정확한 결과. R5/R6 native3 모든 토큰 및 공통117응답 토큰이 일치하여 source/time-allocation 변경의 관측 범위를 확인했다. 대조.

R6 GPU 유효 charge2700.781s: 수집 시 quota가 새 실행R7 사용까지 포함했으므로 R7 predispatch quota를 이전 실행 종료 경계로 사용한 append-only correction을 보존했다. 원본2795.450028s 과대 집계도 삭제하지 않았다. 소비 보정. 131MB diagnostic index 첫 다운로드가 끊겼으며 작은 최종결과7파일은 수집, 완전 diagnostic 원본은 private remote version에 보존.

R7 private arki13/visref-paper-r7-20261006 v1/SHAffb3c862f3ead14fdf0c4f42b173d1a76d645edcb9ca170c6e9b2e5098a1f5bb 실행 중. R8은 thinking allowance만8192로 변경하는 사전명세, 격리238tests PASS, bundleSHAea8a7d2088f58d3121fd992884e16cb4d9610b04bdc7c8228d1f71c98c865b52/96files 실제credential absence 감사 완료; 미dispatch.

현재 block 구조 rank 진단: 기록20block 중14개는 contextual current-block Z라도 T<m.479 첫 block T276/N1792/requestedm537. 이는 현재 Z scope/cardinality 가정의 문제이며 author 구현은 미확인이다. CVPR §4.1.1의 Markov 가정은 현재 state와 전체 history를 구분하므로 cumulative scope를 근거 없이 공식값으로 바꾸지 않는다. 새 contextual-last-layer component는 tiny-model 비교 검사1건 통과, GPU연결되지 않았다.

R6 새 외부 답추출은 Gemini 목적지 명시승인이 필요하며, 새 proxy 호출은0이다. 검토한 전송내용. 별도 로컬 literal audit은 명시된 최종값과 공식 normalization/equality만 사용하여 모든 계획 분모를 유지하며 hosted/proxy 판정으로 주장하지 않는다. 기존R5 scorer 결과는 보존한다.


### 2026-10-06 r7 최종 gate / main8B r9 dispatch / r8 재검토

R7 native3 PASS. TSR243 block7 thinking4096/repetition 실패,479 step1 normalized0.1372139577에서 정상 final 완료,419 step2 normalized0.2487271727에서 멈췄지만 final stray closing-think tag 때문에 GENERATION_FAILED. DPP3 NOT_RUN. 정확한 결과.9개 완결 S10 set을 수학적으로 재계산했고 normalized 표준오차는 별도 audit에 올바른 단위로 기록했다. 기존 native3의 모든 raw토큰/최종문장이R5와 같아 기존 추출값 재사용 감사만 수행; 새 외부 추출0/TSR formal추출은 승인대기. GPUcharge1575.093s.

준비R8은 미실행으로 보존한다. R6 block7 마지막500토큰이21개 고유토큰으로 동일문장을 반복하여 cap만 늘리는 방향을 재검토했다. 또한 total allowance가커지면 explicitlimit가 block_cap을넘을 수 있는 구현조건을 발견해 새로운 수정/freeze 없이 dispatch하지 않는다. 준비 재검토.8192 cap실패나 성능결과를 주장하지 않는다.

논문 주실험8B의 native를 먼저 검증하는 R9 사전명세. Private arki13/visref-paper-r9-20261006 v1/SHAff39b268b4567a81793069f4eca0bd228341694d2161fc7a6e4d686bffdf0c4c,243 격리tests PASS/102파일 actualcredential absence 감사 후 dispatch. 모델 revision9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79/4096hidden/36layers/context40960, 두T4/양자화없음/device_mapauto/각12GiB budget/fp16/eager/native4096. 다른GPU환경이므로1B와latency 비교없음. Native ST만3개/최대3600초; 실제 placement·EOS·sampler·shortnumeric·postprocessor gate 확인 후에만8B TSR/DPP를 고려한다. Checkpoint bf16/modelcardflash 예제와현재운영차이는명시되어authorfidelity미확인이다.


### 2026-10-06 중단 후 회수: R9/R10 실제 결과와 R12 기한 보호

R9는 이미지 projection CUDA1과 입력 embedding CUDA0 사이 device mismatch로 native 생성 전 실패했다. pinned 공식8B generate의 `vit_embeds.to(input_embeds.device)` 처리와 일치시키는 R10 수정으로 실제 세 문항 CUDA1→CUDA0 전달을 검증했다. R9 charge254.911s, R10 charge878.023s, 모두 setup/실패 포함.

R10 private arki13/visref-paper-r10-20261006 v1/SHA b4bd0f711beb12dbe7b2684963c605a2e007a11caa0745346636228d0a7e2bba: 8B native243=4096 token_limit/think 미완결,479=278 normalEOS,419=2106 normalEOS. 전체3 중2개 완료이며 native gate 실패로 TSR/DPP 미실행. 세 문항 모두 prefix16 sampler match, 공식 입력/mask/decoding/text replay match, short48 TV<0.02 및 argmax 일치; 긴 전체 context fidelity를 의미하지 않는다. 생성시간405.762/27.976/181.960s. 원본과 분석. 최종값 로컬 audit3개/분모3/정답0, 외부 호출0이며 공식 extraction/hosted 판정으로 주장하지 않는다.

R11 multi-method7200s는 미dispatch. R12 native8192 resource diagnostic은 동일8B/dev3/seed/decoding/gates를 유지하고1440s session/first243 max900s로 동결했다. 격리 로컬245검사와105파일 credential/raw/selection/holdout absence 감사는 통과했지만 원래6시간 종료 기한과120s 여유를 만족하지 못해 push가 launch 생성 전에 거부했다. GPU소비0/새 model outcome0이며8192 설정의 성공·실패를 주장하지 않는다. 미dispatch 기록. 실패 gate나 최초 승인 기간을 변경하지 않았다.

이번 신규 실행 R5/R6/R7/R9/R10 유효 GPU charge 합계7141.308s(1.9837h). 중단 후 회수 시 모든 실행 terminal; 기존 STOP와 소비 원장 유지. 정확한 저자 baseline,8B native3/3,TSR3/3,DPP3/3는 아직 미달성. 다음 모델 실행은 더 긴 native 자원 진단을 먼저 완료해야 한다. Gemini 외부 답추출은 목적지 명시승인이 필요하여 새 호출0/승인 대기 상태를 유지한다.


### 2026-10-06 Gemini 전송 명시 승인 및 보류 채점 완료

결정: Gemini 전송 허가를 별도 승인 원본으로 기록했다. 기존 승인대기 조건은 해소됐다. 공개 개발243/479/419와 기존 최종 답변 segment만 전송했고 gold/추론본문/비공개 코드/credentials는 전송하지 않았다. GPU 재실행0/새 모델 추론0/selection·holdout 접근0.

R6=8호출, R7=8호출, R10=4호출; 새 호출 합계20, 전부HTTP200/완결 추출. Public Kaggle MathVista2-shot과 공식MathVista5-shot의 추출·정오 판정이 모든 paired row에서 일치했다. 실제 완료된 최종 답변은 R6 4개/R7 4개/R10 2개이고 모두오답; 실패·미실행도 각방법 frozen분모3에 유지했다. 이미 노출된 소수 개발 표본의 점수이며 성능 일반화나 baseline 재현 성공을 의미하지 않는다. Canonical Kaggle hosted judge와 동일성 및 Gemini alias revision은 여전히 미확인이다.

전체 호출·일치 확인. 각 r6/r7/r10 offline_scoring/results.json에 original generation status, prediction, failure 및 분모가 보존돼 있다. 기존 generation/로컬 audit/옛 scoring freeze는 변경하지 않았다. R6 원래 scripts/paper_score.py freeze 그대로 실행했고 R7/R10은 별도 scripts/paper_score_remaining.py에서 신규 동결 후 실행했다.


### 2026-10-06 baseline 재개 승인

결정: 신규 동결 실행에 한해 최초6시간 wall deadline을 대체한다. 기존R12 미dispatch/freeze 및 STOP/원장은 보존한다. 기존 Kaggle 예산 전량 사용과 Gemini 목적지 승인은 유지하고, dev243/479/419만·selection/locked/압축 금지·native→TSR→DPP correctness gate를 유지한다. CONTINUATION_AUTHORIZATION.json에 별도 승인 기록. 신규R13은R12의 native8192 자원 가설을3600s 세션/native2700s로 검증하며,8192는 공식 저자값이 아니다.


R13 private arki13/visref-paper-r13-20261006 v1/kernel137242408 dispatched: source119ee49790877999966e079c93bb312a172d68c58d7228429dac0d622d7c8307, native8192,3600s cap/native2700s,245 isolated checks PASS,106 payload files/790699bytes actualcredentials/raw/selection/holdout absence verified. Launch generic estimate1800–3480 retained; prereg expected~1300s and this metadata mismatch appended as DISPATCH_ESTIMATE_CORRECTION.json. No outcome yet.


R13 실행 종료 시 수집/소비 원장/로컬audit/승인된 final-only Gemini 채점을 수행하는 bounded finisher 연결. 최대80분 감시·추가GPU dispatch 없음. 결과 경로r13/FINISHER_RESULT.json, 오류r13/FINISHER_FAILURE.json, 진행r13/finisher.log. native 통과 시의 TSR/DPP는 미dispatch이다.


### 원인 설명 정정

현재R13/finisher 계속 진행; correctness gate/selection/holdout 제한은 유지한다.


### 개발-only baseline 작업 지속

이미 승인된 개발-only baseline 작업 지속.


후속8B TSR/DPP R14 사전 조건을 docs/paper_baseline_alignment/R14_PREREGISTRATION_20261006.md에 기록했다. R13 native3/3 actual gate 통과·회수 전에는 freeze/dispatch하지 않는다. 현재 미실행 준비이며 native8192/block8192/total32768은 운영 자원 한계, 저자공식값 아님.


R14 build/push에R13 실제 native3/3 수집 증거 guard를 연결했다. naturalEOS/생성1회/think완결/afterEOS0/official replay4항목/prefix16 sampler/finite short48 TV≤0.02+argmax 일치가 모두 필요하다. complete notebook status나 denominator 축소로 통과하지 못한다. 실패·누락·NaN을 포함한7개 release-guard 검사PASS; R14는 아직미build/미dispatch이며 R13원본bundle변경없음.


R13 live native243 TIMEOUT/SAMPLE_TIME_SHARE_EXHAUSTED at runner1184.887s; lastprogress7680tokens/generation877.730s.8192 token-limit outcome미검증。 R14 remains미dispatch since native3/3 cannot pass this run. R15 time-allocation-only diagnostic preregistered in docs/paper_baseline_alignment/R15_PREREGISTRATION_20261006.md;4200s session/native3600s, unchanged8192 generation, requiresR13terminal/collection and othernative gates beforenewfreeze. ExistingR13 still running; no duplicatejob.


### Sampling 확률/EOS 동일응답 재계산 진단

R6완결18세트/불완결1, R7완결9세트에 저장된 raw T1/full-vocab logprob와 q T0.6/top50 logprob를 paired 비교했다. 신규추론/외부채점0/기존control변경0. q로샘플해p로평가한 값은H(q,p) cross-entropy이며H(p)가 아니다. 별도p샘플 추론 실험은 아직없다.

같은419 step2의동일S10응답에서 normalized q+EOS=0.24872717273765085, q−EOS=0.2507936781444885, rawp cross-entropy normalized=0.3072779062234326. EOS제외 또는확률기준변경은counterfactual0.25중단조건을뒤집는다. R6/R7이같은응답을공유하므로2개독립발견으로세지않는다. EOS자체평균NLL q0.014702916/p0.036973064이고길이분모변경도작용한다. authorEOS정책/normalization/thresholdtransfer는계속미확정이다. 상세각run/PROBABILITY_EOS_PAIRED_AUDIT.json.


### R13 수집·채점 완료 — 자동 마무리 기록

R13 runner=GATE_FAILED, native 완료=2/3; GPU charge=1415.668999999998s; Gemini 호출=4. 원본/세부 판정은 artifacts/runs/paper_baseline_sixhour_20261006/r13/FINISHER_RESULT.json 및 offline_scoring/results.json에 보존. 정확한 논문 재현 미확정; 새 GPU dispatch 없음.


### R13 회수 후 정정 및 R15 진행 근거

R13 terminal/수집/채점 완료: native479/419 정상EOS, 243 TIMEOUT/SAMPLE_TIME_SHARE_EXHAUSTED(partial7796 tokens). R10과 공통4096/278/2106 raw tokens 모두 동일(COMMON_PREFIX_AUDIT.json). GPU charge1415.669s, 추가scope 누적8556.977s. 기존채점포함 Gemini 총24calls; denominator3 유지. R15는8192 cap 그대로/session4200/native3600 시간만 변경하는 사전등록 진단; TSR/DPP gate는 아직 미통과.


### R15 시간 할당 진단 dispatch

private arki13/visref-paper-r15-20261006 v1/kernel137244387. source78b04f53ccec6d585d6271072b7fcc09c2763d6ad779608307d629fb422ccd3b/notebookc025798472c26afbf351f2d6365a0ee95908ab3f9dd4b293fc892362e39912c6. 격리256PASS/18.65s,108files/800715bytes actual credentials 및 raw/selection/holdout absent. R13 대비 identity 외 session3600→4200/native2700→3600만 변경(SINGLE_FACTOR_AUDIT.json). native8192/model/revision/decoding/seeds/dev3/STOP 동일. 모든이전jobterminal 확인; estimate1500–4080s. native3/3 gate 미확정이고 TSR/DPP 아직 미실행. 종료·수집·최종답변채점 bounded finisher가 담당하며 새로운GPU dispatch 없음.


### 실제 sampler의 확률 분포 CPU 진단

생산 sample_token/probabilities/sequence estimate를 사용해 정책별20000개 완결 응답을 생성했다. q(T0.6/top50)의 H(q) exact2.372134924 vsMC2.373178092(SE0.007081238); p(T1/full60)의 H(p) exact2.880205870 vsMC2.876536744(SE0.007060929), 빈도·6SE 검사PASS. q draw를rawp로평가하면 exactH(q,p)=2.482442856/MC2.483068963이며H(p)가 아니다. EOS제외qNLLmean2.032342291 vs포함2.373178092. 이는 제어가능한60개토큰 분포의CPU 실험이고 실제InternVL p-sampling/GPU검증이나author policy 확인을 대신하지 않는다. inference/control/채점변경0, 신규GPU0. 상세산출물. R15 원격bundle과run settings 변경없음.


### Native 243의 반복/종료 진단

동결R10/R13 native243 출력만 분석했다. R13 partial7796에서 </think>(151668)/EOS(151645) 모두0회. 구간6144–7167의16gram duplicate fraction0.554013875/mean raw next-tokenH0.004560159, 처음1024는0.0/H0.489794973. 마지막500개 token은141개고유라단일문장완전고정loop로단정하지 않는다. 실제답변 sequence entropy나EOS확률을이tokenH로대체할수없다(EOS확률은미저장). 새추론/채점/중단정책변경0; R15결과전cap추가확대근거없음. 상세 r13/NATIVE_REPETITION_DIAGNOSTIC.json,원본hash 포함. R15가또8K cap에걸리면생성정책/정밀도/긴문맥수치검증을재검토하고자동상한증가를하지않는다.


### 긴 native 출력의 수치 검증 준비

캐시logits NaN/Inf와검사metric NaN을명시거부하도록 long_parity 보완. 저장token teacher forcing을64토큰씩입력하고지정offset에서cached eager/uncached SDPA비교하는 native_reference 추가; 새sampling0/RNG불변. 실제tiny Qwen3 CPU 총10checksPASS/2.68s(offset16/64/127, total19/67/130; invalid offsets/nonfinite/failure restoration/cache불변). 실제8B/T4/8K 구간은미검증이고현재R15 source/bundle변경0. 기존 native full_reference 호출은prompt+16까지만이므로긴생성구간parity증거로확장해해석하지않는다. 조건/결과research/NATIVE_REFERENCE_CPU_VALIDATION.json. 다음GPU계획은R15terminal/회수후실제실패종류에따라정하며native3/3 gate 변경없음.


### R15 live8192 도달 및 R16 사전 준비

R15 live native243=8192tokens/956.902310s, GENERATION_FAILED/token_limit. 이번에는 시간 부족이 아닌 생성 상한 도달이다. 나머지 두 문항 및 terminal/collection 대기 중. 추가 상한 확대 없음. native_generated_reference_positions를 지정한 새 run에만 teacher-forced 긴 문맥 reference를 연결했고 로컬 전체263검사PASS/11.62s, GPU 미검증. R16_PREREGISTRATION_20261006.md는 native-only/같은8192 및 decoding/seed/pinned8B/GPU/dtype에서 생성offset1024/4096/8192 수치 진단을 명세한다. session5400/native4500은 추가 validation 시간이다. 기존 native gate 불변; R15 회수 후 실제 failure 판정 전 build/dispatch하지 않는다.


### R15 수집·채점 완료 — 자동 마무리 기록

R15 runner=GATE_FAILED, native 완료=2/3; GPU charge=1421.242000000002s; Gemini 호출=4. 원본/세부 판정은 artifacts/runs/paper_baseline_sixhour_20261006/r15/FINISHER_RESULT.json 및 offline_scoring/results.json에 보존. 정확한 논문 재현 미확정; 새 GPU dispatch 없음.


### R15 수집 후 확정 판정

R15 native243=8192 token_limit/956.907544s, think_close/EOS0; 479/419는278/2106 tokens normalEOS와모든기존native gate 통과. R13과공통7796/278/2106 token 모두동일(r15/ANALYSIS.json). native3/3 release 실패로TSR/DPP 미실행. charged1421.242s, 추가scope누적9978.219s(2.7717h). Gemini4calls,두동일scorer모두정답0/분모3; 총28calls. 이는3개개발문항debug이며accuracy우열/공식leaderboard값 아님. 자동cap확대없음. R16은동일native8192/decoding/model로긴문맥teacher-forced numerical 진단; 격리267PASS/12.07s,110files/808891bytes actualcredential/raw/selection/holdout absent. 기존native baseline gate는실패로유지되며이진단허용은TSR/DPP release가아니다.


R16 private arki13/visref-paper-r16-20261006 v1/kernel137246782 dispatch. Source330b1691c1ce94ccaa29a74d745ecd8ac56bb624d106da5310a4b96164926cc5, notebookd5975f1c8c11604a8615915a33b2137513ff5c31cece91d2ed7cbb633d1887ed. 전작업terminal 확인; estimate1500–5280s/session5400. 현재GPU수치결과미검증; 원본STOP SHA824d748...동일. 종료·회수·최종답변채점 bounded finisher scripts/paper_finish_r16.py; 재dispatch하지않음.


### 고정 모델 README 및 nucleus policy 준비

8B exactrevision9bb6a56... README SHA47383e9746907dbfabfcad9c786aab81e43586b5e4b0bb07ad893d8b58f3b1e9 확인. Lines80/412는Thinking do_sampleTrue/T0.6 권고; line767 LMDeploy 예제는top50/top_p0.95/T0.6/max8192이며VisRef author설정으로격상하지않는다. 표준 TLS/certifi로원본저장(research/INTERNVL8B_PINNED_README_AUDIT.json). 새 native sampler replay는실제TF4.55 generation processor list와p1/p.95 확률exact일치, p1 기존seed20draw일치/비지원repetition거부 등CPU4checksPASS/3.18s. 실GPU미검증. R17_PREREGISTRATION_20261006.md는R16전수수치검사통과후에만top_p1→.95 한변수진단준비; 아직등록/build/dispatch없음. 다른controller p1과혼합비교금지, 실패gate 불변. 현재R16 source/bundle변경없음.


R17 준비 보완: native prefix replay는 실제decoding의HF temperature/top_k/top_p warper를 사용하며 long-reference TV도 같은policy를 명시하도록future source에 연결했다. short48 기존TV는p1을명시하고p.95 지표로표시하지않는다. CPU sampler/reference15checksPASS/3.20s. 실행R16 bundle불변, R17 등록/build/dispatch 미실행.


### 주모델 8B native trace의 DPP rank 제약

R15 native3개 rawids만분석(새추론0/DPP실행0). 479는원본V1792→m537이지만자연완결thinking T187/고유input-id95. 따라서이trace의pure L=VZZ^TV^T에서텍스트-only contextualZ도rank≤187, inputlookupZ는≤95; m537 minor determinant는정확한산술에서0. hidden d4096은이T제약을없애지못한다. 243는T8191/unique323/m230이지만생각미완결,419는T1955/unique227/m76; 충분한rank상한은실제양의det를보장하지않는다. Native와TSR/DPP의phase seed/trace는달라서향후실제DPP실패를미리판정하지않는다. 이결과는author Zscope/jitter/cardinality/짧은trace처리의확인이필요함을구체화한다. fixedm gate를느슨하게하거나수치잡음을양의rank로해석하지않는다. 상세r15/NATIVE_DPP_RANK_BOUND_AUDIT.json.


### R16 수집·채점 완료 — 자동 마무리 기록

R16 runner=GATE_FAILED, native 완료=2/3; GPU charge=1513.0279999999984s; Gemini 호출=4. 원본/세부 판정은 artifacts/runs/paper_baseline_sixhour_20261006/r16/FINISHER_RESULT.json 및 offline_scoring/results.json에 보존. 정확한 논문 재현 미확정; 새 GPU dispatch 없음.


### R16 전수 수치 검증 및 R17 한변수 진단

R16의 prefix 3개·generated-context 6개 reference 모두 기존 TV≤0.02/finite/argmax 또는 near-tie/backend 복구/no-cache 조건을 통과했다(max TV0.0152803585). R15와 8192/278/2106 raw token이 모두 동일하며, 243 token_limit로 native baseline gate는 여전히 실패다. 이는 teacher-forced cached decoder와 uncached SDPA의 수치 검증이며 공식 generate의 모든 logit 검증은 아니다. 상세 r16/ANALYSIS.json. GPU 추가scope 누적11491.247s; Gemini 누적32calls, 개발3문항 분모 유지/정답0. R17은 동일 조건에서 native top_p만1→0.95인 사전등록 진단. 모든 R16 numerical reference가 있어야 build/push하도록 guard 추가; TSR/DPP release 조건 불변.


### R17 native nucleus 진단 실행

R17 private arki13/visref-paper-r17-20261006 v1/kernel137249458 실행. Source7c575d92afa957a6c58992800ea4b4733f0b54640c8bbf012e676a8a271f421c, notebook5fbe6ef08d69e2070ac74d2eeaa95f55f77f3d92b8fec42bdff441eaa86593a2. R16과generation 설정차이는 top_p1→0.95만(CONFIG_FACTOR_AUDIT.json); native sampler와fullreference도실제policy를사용하며short48은기존p1지표라고표시한다. 격리280PASS/11.32s,113files/872078bytes actualcredential/raw/selection/holdout 미포함 감사통과. 이전모든job terminal/확인; estimate1500–5280s/session5400/native4500. STOP SHA 동일, 누적11491.247s는R17종료후새charge를더한다. bounded finisher scripts/paper_finish_r17.py는새GPU실행없이종료·회수·허가된최종답변채점만수행. GPU결과미검증, native3/3 실패조건 및TSR/DPP 차단불변.


### 공통 sampling policy 로컬 준비

새 paper_policy_decoder는 native_decoding의 T/top_k/top_p/beam/repetition/sample 설정을 직접 읽어 thinking/probe/final의 sampling과probe log-probability를일치시킨다. EOS/min1 억제는warp전에적용하며TF4.55실제processor순서와p1/p.95exact비교. 기존RecordedDecoder의step/close/부분출력보존을새class에유지했으며historicalcontroller는수정하지않았다. targeted19checks로p1이전동작·p.95 실제tinyQwen draw/종료확률·S완결/별도finalseed 확인. 상세 research/COMMON_POLICY_COMPONENT_PREPARATION.json. 아직runner미연결/등록된GPUversion없음;실행R17bundle불변. Native3/3 gate와author fidelity조건을해제하지않는다.


R17 live 첫 row243=GENERATION_FAILED/token_limit. top_p.95는 이 문항의 native 종료 문제를 해결하지 못했다. 동결 runner hash 일치 확인; 이 row는 reference 호출 반환 후 발행되지만 실제 metric은 terminal 회수 후 확인한다. 나머지2개 진행 중, TSR/DPP 미실행·재dispatch 없음. 증거 r17/FIRST_ROW_LIVE_OBSERVATION.json.


### R17 수집·채점 완료 — 자동 마무리 기록

R17 runner=GATE_FAILED, native 완료=2/3; GPU charge=1476.5910000000003s; Gemini 호출=4. 원본/세부 판정은 artifacts/runs/paper_baseline_sixhour_20261006/r17/FINISHER_RESULT.json 및 offline_scoring/results.json에 보존. 정확한 논문 재현 미확정; 새 GPU dispatch 없음.


### R17 확정 결과 및 관측 hook 대조 준비

R17 native243=8192token_limit/961.611864s,479=278EOS,419=1840EOS. 요청수치reference9개 전수통과(maxTV0.015264899), 실제p.95확률정책/완결전prefix길이/finite/backend복구/no-cache조건확인. top_p.95는243종료문제미해결; R16과243prefix12tokens후갈라졌지만종료실패는같고479전체동일·419prefix610후갈라짐. native3/3 release실패/TSR·DPP미실행. 상세r17/ANALYSIS.json·FINISHER_RESULT.json. GPUcharge1476.591s/추가scope누적12967.838s; Gemini4calls/누적36calls,동일두scorer정답0/분모3. 다음진단은generation중프로젝트관측hook과progresscallback제거의한변수대조로사전등록하며, 모델/입력/seed/decoding/cap변경없음. 아직R18미등록·미실행.


### R18 관측 제거 대조 동결·로컬 검증

R18은R17과model/prompt/dev3/phase seed/decoding/output8192/time envelope가동일하며유일한실험요인은native_observation_mode=off. 프로젝트prehook37개만생성중제거하고finally복구하며entropycapture/progresscallback을생략한다. 모델·Accelerate 실행hook유지, 생성후chat/prefix/short/fullreference검증유지. TinyQwen3에서토큰/RNG동일·외부hook유지·성공/예외복구·activefeed거부확인; targeted28PASS/2.32s,격리292PASS/11.22s. 원본STOP SHA동일, R17수치9개전수통과/누적12967.838s 기록후등록. 상세 R18_PREREGISTRATION_20261006.md·r18/CONFIG_FACTOR_AUDIT.json·LOCAL_OBSERVER_VALIDATION.json. 실InternVL observer인과관계는GPU미검증.


R18 private arki13/visref-paper-r18-20261006 v1/kernel137251744 dispatch. source9e879666cfcf31a2806130cbef3846206c78786ae8c05dd10eaa02d020d6bca5/notebook740406c51ac1d2b51eebaa8e835fa6ef5e161ede31366e50b4944ba2631837a0. 격리292PASS/116files/840364bytes actualcredential/raw/selection/holdout미포함 확인; 이전모든jobterminal확인. estimate1500–5280s/session5400. bounded finisher scripts/paper_finish_r18.py는종료·회수·동일final-only채점만수행하며재dispatch없음. 실GPU observer 영향 판정 대기.


### 남은 모델 문맥 상한 R19 준비

8192는운영상한이며8K미완결만으로모델자체실패를증명하지않는다. 남은native출력을실제 input_context_tokens(이미지/chat포함)에서max_position_embeddings40960을빼는helper와명시적future paper_r19 policy를준비했다. CPU11checks/3.60s 및관련21checks/3.17s PASS;actualtinyQwen 전체context경계와EOS한토큰조기종료 확인. R19 아직dispatcher미등록/build/dispatch없으며R18종료·관측대조·전수수치검사 먼저확인. 실제R17 modelmap은GPU0/1 각각16/20layers이며40K KV추정각2.5/3.125GiB;densefp16reference mask3.125GiB+temporarybool1.5625GiB, FFNactivation별추가메모리. 실제OOM관측은아니며longreference자원조건해결후실행조건동결. 상세 research/REMAINING_CONTEXT_PREPARATION.json 및 R19_PREPARATION_20261006.md. 실행중R18payload불변/STOP보존.


### 중단 지점 회수 — bounded reference CPU

진행 중이던 full-reference CPU 검증은 4 PASS/3.27s로 회수했다. Tiny Qwen3의 33/97/127-token uncached eager 마지막 logits와 query-chunked reference가 atol1e-6/rtol1e-5 범위에서 일치했고 RNG 및 성공/예외 후 backend·MLP 복구를 확인했다. 이는 실제 InternVL/T4/40K 검증이 아니며 runner 미연결·R19 미등록/미실행이다. 상세 artifacts/runs/paper_baseline_sixhour_20261006/research/BOUNDED_FULL_REFERENCE_CPU.json. R18은 remote RUNNING으로 확인했고 기존 finisher는 종료·회수·허가된 final-only 채점을 계속한다. 새 GPU dispatch 없음. 재개 시 R18 FINISHER_RESULT/실패기록부터 확인하고 관측 제거 전후 raw IDs·전수 numerical reference를 비교한 뒤 R19 조건을 동결해야 한다. STOP 보존.


### R18 수집·채점 완료 — 자동 마무리 기록

R18 runner=GATE_FAILED, native 완료=2/3; GPU charge=1486.714s; Gemini 호출=4. 원본/세부 판정은 artifacts/runs/paper_baseline_sixhour_20261006/r18/FINISHER_RESULT.json 및 offline_scoring/results.json에 보존. 정확한 논문 재현 미확정; 새 GPU dispatch 없음.


### R19 조건부 준비 — reference 연결 및 실행 차단

Full-reference의 query256·tokenwiseMLP256 분할을 명시적 opt-in으로 연결하고 actual input T4 수치 smoke를 생성 전 gate로 추가했다. CPU에서 Accelerate outerhook 보존/원본 eager 일치/cache·RNG·backend 복구를 확인했으며 관련28PASS/3.28s, R18 동일토큰·hook복구 요구 guard29PASS/.12s, 종합targeted67PASS/2.66s. 실제 InternVL/T4/40K 검증은 미수행이다. R19_PREREGISTRATION_20261006.md에 remaining-context cap40960-input, naturalEOS, session14400/native13200/first24311000s 및 complete-context reference offsets를 조건부 사전등록했다. R18 전수검증 및 R17 raw-ID 동일성이 확인되기 전 build/push 차단; R19 아직 bundle/build/dispatch 없음. 세부 research/R19_CONNECTED_REFERENCE_CPU.json. Running R18 payload 불변/STOP 보존.


### R18 관측 대조 확정 및 R19 동결 준비

R18 native243=8192 token_limit/966.481526s,479=278 EOS,419=1840 EOS. 프로젝트prehook37개 제거·복구 및실행hook유지 확인; 세문항모든raw token이R17과정확히동일하다. 요청수치reference9개모두통과(maxTV0.015264899). 이조건에서generation중관측기능이종료실패를일으켰다는증거없음; native3/3 release실패와TSR/DPP차단유지. 상세 r18/ANALYSIS.json·FINISHER_RESULT.json. GPUcharge1486.714s/추가scope누적14454.552s; Gemini4calls/누적40calls,개발분모3유지·accuracy선택없음. 조건부R19 buildguard를실제결과로통과했으며remaining-context generation진단의동결·격리검증을진행한다.


### R19 동결·격리 검증 및 자원 계획

R18실제결과의관측대조guard PASS 후R19를동결했다. source6b4710cfb0f99904808082426c82d5231b5cac84dbdbadba7b19c523d5bf0a36/notebookbec2c479485d67127fe90a05789707ddc81ddb18fd23fadcc43de8523d45ba74;격리313PASS/11.44s 및actualpayload120files/859001bytes credential/raw/selection/holdout absent. 샘플·모델revision·prompt·native sampling정책은R18과동일, outputbudget만actual40960-input으로계산한다. Validation지원은query/MLP256분할·생성전actualinputGPU numerical gate로명시했다. Session14400/native13200/first24311000s, predispatchestimate1800–14280s;완결보장아님. 누적14454.552s;failed/setup모두charge. 세부R19_PREREGISTRATION_20261006.md 및r19/CONFIG_FACTOR_AUDIT.json·PREUPLOAD_AUDIT.json·bundle_validation.json. Native3/3·전수수치gate 유지/TSR·DPP없음/STOP불변.


R19 private arki13/visref-paper-r19-20261006 v1/kernel137254020 dispatch. Source6b4710cfb0f99904808082426c82d5231b5cac84dbdbadba7b19c523d5bf0a36/notebookbec2c479485d67127fe90a05789707ddc81ddb18fd23fadcc43de8523d45ba74. 전작업terminal확인. session14400/native13200/first24311000s, estimate1800–14280s. STOP보존; 실T4reference smoke 및extended-native종료결과 대기. scripts/paper_finish_r19.py는새GPU작업없이종료·회수·허가된final-only채점만수행한다. 결과미검증/TSR·DPP release없음.


### Native 종료와 논문 adaptive stopping 구분

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

작성자조사의핵심운영결론을primary sources로확인했다. 공식8B Thinking config65536/use_lmdeploy/R1/do_sample/V2.0 및wrapper T.6/p.95를reference로우선하며8192는usage example/cap진단으로유지한다. top_k50은LMDeploy0.9.1default상속근거가있어config에없다는이유만으로약한근거라고판정하지않는다;실제dependency/historical commit필요. R1wrapper는boxed최종답지시도설정하므로systemprompt만비교하면불충분하다. Co-RLgreedy·boxed·rule grading확인;Aug2026코드는VisRefMarch2026설정을역추론하는근거가아니다. 전점수/독립실행provenance는아직전수검증하지않았다. 65536은LMDeploy session/position/backend까지별도recipe로검증하며현재eager40960-input에그대로복사하지않는다. HistoricalVLMEvalKit/LMDeploy/tiles/prompt/judge추적우선순위반영,liveR19payload와현재scorer불변/새GPUdispatch없음. 상세docs/paper_baseline_alignment/SAME_MODEL_SURVEY_REVIEW_20261006.md 및research/SAME_MODEL_SURVEY_REVIEW.json.


### VLMEvalKit historical recipe의실제변경 확인

2025-08~2026-03 file-specific Git history(wrapper12/config50commits)를읽고9/1·9/5·9/12·2/26 snapshot config/wrapper/requirements 및utils원본을SHA와보존했다. 9/1에이미8B Thinking config65536/R1/sampling이있으나backend Turbomind/session16384;9/5부터InternVL3_5 PytorchEngine/session=max(16384,max_new_tokens)로수정됐다. 동일config이름의동일유효tokenbudget을가정하면안된다. 네snapshot모두wrapperT.6/p.95/random_seedNone, R1system은현재R19와문자열exact동일. 별도boxed지시와MathVista-defaulttiles6은현재fullprompt/preprocess12와계속대조필요. requirements에는LMDeploypin없음/VisRef사용commit연결미확정. 상세docs/paper_baseline_alignment/VLMEVALKIT_HISTORICAL_RECIPE_20261006.md 및research/vlmevalkit_history/HISTORICAL_RECIPE_AUDIT.json. R19livepayload불변/새dispatch없음.


### 작성자 결정: 변경 이력 추적 중단, baseline 구현 복귀

추가 Git history 탐색을 중단했다. CommonPolicyReferenceDecoder를 연결하여 response controller와 full-context numerical reference가 동일한 frozen native_decoding을 사용하도록 했다. 기존 historical decoder 기본 정책과 live R19 bundle은 유지한다. TSR 제어·MC probes/final seed 분리·actual tiny Qwen nucleus(.95) reference 및 기존 공통-policy 검증 7개 PASS/2.74s. Scripted token boundary test는 full-support p1, 실제 nucleus 확률 검증은 강제 토큰 없는 별도 경로로 분리했다. CPU 구성 검증이며 main8B TSR/DPP 실행 성공이나 author fidelity 증거가 아니다. 현재 R19 native remaining-context 진단 RUNNING; 회수 후 native EOS·수치 gate를 판정하고 다음 GPU run을 결정한다.


### R1 최종 답 지시 후보 구현

현재 dataset query에는 공식 R1 wrapper의 boxed 지시가 없는 것을 확인했다. baseline_question의 기본 dataset_query는 유지하고 internvl_r1_boxed_reference 후보를 추가했다. native 입력과 model.chat replay는 동일한 question을 사용하며 question SHA/policy를 기록한다. 보존 primary-source literal과 exact 일치·기본 입력 보존·미지 policy 거부·두 경로 공유를 로컬 확인했다. Dataset query에 suffix를 더한 후보이며 전체 VLMEvalKit question 구성/LMDeploy/tiles/judge 경로 재현이나 VisRef author 설정 증거는 아니다. Live R19 불변; 새 version 동결·검증·예산 확인 전 원격 실행 없음. 상세 artifacts/runs/paper_baseline_sixhour_20261006/research/R1_BOXED_PROMPT_PREPARATION.json.


### Common-policy controller 실제 runner 연결

controller_sampling_policy=native를 명시한 새 frozen run은 CommonPolicyDecoder를 사용하며 full_context_reference_validation이면 CommonPolicyReferenceDecoder를 사용한다. 기본 historical branch는 유지한다. 실제 runner selection 네 branch와 unknown-policy 거부를 로컬 실행해 확인했다. 생성·reference·MC 기록의 동일 sampling-policy component는 앞선 17PASS 근거를 유지한다. model_runtime에 실제 decoder class와 policy를 기록한다. Live R19 bundle 불변; GPU 경로 사용 성공은 아직 미검증이며 native gate/새 version 동결·자원 조건 통과 후 실행한다. 상세 artifacts/runs/paper_baseline_sixhour_20261006/research/COMMON_POLICY_RUNNER_CONNECTION.json.


### 최신 baseline 구성 격리 묶음 검증

R1 prompt/common-policy runner/reference/context reservation 수정을 포함한 실제 package를 임시 폴더로 추출하여 327 PASS/12.76s 확인했다. paper_prompt 자동 포함, source SHA fdf8628a4415a263e734b1cb187b34b5bce6b8f1953a0380b2168669938720eb. 이 묶음은 local_packaging_diagnostic/gpu_execution_authorized=false로 원격 실행 후보가 아니며 GPU·API 사용 없음. 원본 묶음/manifest/log는 artifacts/runs/paper_baseline_sixhour_20261006/research/local_components_package_20261006 및 LATEST_COMPONENTS_ISOLATED_PACKAGE.json에 보존한다. Live R19와 이전 freeze 변경 없음; 실제 checkpoint 생성 검증을 대체하지 않는다.


### Contextual Z 후보를 common-policy baseline 경로에 연결

새 run에서 z_representation=last_layer_post_norm_contextual_hidden_states를 명시하면 common-policy+full-reference 조합에서만 CommonPolicyContextualReferenceDecoder를 선택한다. 기본 input_embedding_lookup은 유지한다. 마지막 norm 출력의 committed reasoning token별 벡터를 수집하며 think/EOS 제외, 추가 model forward 없음. 실제 tiny Qwen full forward output_hidden_states[-1]과 Z 일치 및 hook 복구·수치 reference·원본 visual row reinjection을 확인했다(관련15PASS/4.31s; contextual3PASS/3.15s). 반복 ID 하나로도 서로 다른 Z 3개를 통해 fixed m3 선택·완결을 확인; 입력 lookup의 unique-ID rank bound를 이 경로에 적용하지 않고 min(N,T,D) 필요 상한으로 기록한다. rank 부족시 jitter/m변경/강제 통과 없음. Author layer/scope 미확정·GPU 미검증·live R19 불변. 상세 artifacts/runs/paper_baseline_sixhour_20261006/research/COMMON_POLICY_CONTEXTUAL_Z_CONNECTION.json.


### 작성자 결정: Z 우선 후보 last hidden state로 동결

작성자 결정(Z에 입력 임베딩이 아닌 last hidden state 사용)을 반영하여 다음 controller의 primary operational candidate는 마지막 레이어 post-norm contextual hidden states로 지정했다. Config configs/paper_baseline/primary_controller_candidate.json, SHA d12ef76aa9db3a6c45a667ba101d056e40d1ec92f084ae8ad2a5b9054c31a3bd. Current committed reasoning block에서 think/EOS 제외라는 범위 가정은 명시하며 author layer/scope 미확정 표시를 유지한다. Controller native sampling+full-context numerical reference를 요구한다. Input embedding 구현은 historical provenance로 보존하며 새 primary 후보로 자동 선택하지 않는다. Fixed m·무정규화 DPP와 rank gate 유지; short T<m은 hidden-state 전환으로 해결됐다고 주장하지 않는다. Live R19/new GPU dispatch 없음; profile 자체는 dispatch authorization이 아니며 다음 run config 동결 시 병합·검증해야 한다.


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

R19 runner=GATE_FAILED, native 완료=2/3; GPU charge=11470.720000000001s; Gemini 호출=4. 원본/세부 판정은 artifacts/runs/paper_baseline_sixhour_20261006/r19/FINISHER_RESULT.json 및 offline_scoring/results.json에 보존. 정확한 논문 재현 미확정; 목표 service 상태는 이 finisher가 변경하지 않음, 새 GPU dispatch 없음.


### R19 정산과 R20 prompt 요인 준비

R19 native2/3; 243 TIMEOUT, 문맥 상한 도달 미검증. 총 GPU25925.272s(약7.2015h), Gemini44회. 기존 ledger 보존. R20은 공식 boxed 최종 답변 지시만 추가한 native-only 실험으로 사전 등록했다. 시간/문맥 한도 증대 없음; gate/denominator3 유지. 아직 dispatch 없음. 상세 docs/paper_baseline_alignment/R20_PREREGISTRATION_20261006.md 및 research/TERMINAL_BUDGET_RECONCILIATION_R19.json.


### R20 private v1 실행

Kernel arki13/visref-paper-r20-20261006 v1/137282639, source SHA25921f4f2eb4c06874293a42de0a46238c110e48e5f01052331cb6fb65dd6e27. CPU329 PASS/12.75s, 실제 payload125파일884691bytes 검증. R19 대비 identity 외 question_prompt_policy 하나만 변경. 이전 job terminal 확인 후 제출, RUNNING 확인. Session14400s, 30분 finisher 관찰(session21575), 종료 후 raw/final-only scoring 자동 회수. 현재 gate 미판정, faithful baseline 미완료. STOP SHA 유지.


### R20 수집·채점 완료 — 자동 마무리 기록

R20 runner=GATE_FAILED, native 완료=2/3; GPU charge=1501.7200000000012s; Gemini 호출=4. 원본/세부 판정은 artifacts/runs/paper_baseline_sixhour_20261006/r20/FINISHER_RESULT.json 및 offline_scoring/results.json에 보존. 정확한 논문 재현 미확정; 목표 service 상태는 이 finisher가 변경하지 않음, 새 GPU dispatch 없음.


### R20 종료 후 검증·정산

Native diagnostic release=False; cumulative GPU=27426.992000000002s. 원문 질문+boxed 지시 provenance, 자연 EOS, 실제 generated-context numerical reference를 함께 판정했다. Timeout의 missing receipt는 입증 부족이며 구현 불일치로 단정하지 않는다. Faithful paper baseline 미확정, 신규 GPU dispatch 없음. 상세 artifacts/runs/paper_baseline_sixhour_20261006/r20/ANALYSIS.json 및 POST_FINISH_RESULT.json.


### 공식 기본 모델 경로 진단 우선

추가 GPU 제출 전 pristine pinned author model.chat 경로를 우선 검증한다. 기존 native 생성은 공개 저자 model.generate를 사용했지만 custom Decoder/precomputed visuals/reference 검증과 chat replay가 붙어 있었으므로 pristine chat 실제 생성과 동일하다고 단정하지 않는다. R20은 세 문항 모두 normalEOS;243은3409tokens/333.35s로 완결했으나 오답.419는1opening/2closing thinking 태그로 strict parser가 실패했으며 출력과 최종 답변 텍스트는 존재한다. 기존 gate/score 보존, 임의 last-close 추출로 통과 처리하지 않는다. Official chat control은 VisRef Decoder/probes/parser 없이 실제 model.chat 한 번을 호출하도록 분리했고 retained8B source interface1PASS/.85s. GPU inference 검증 아님. 현재 모든 GPU 작업 종료, 신규 dispatch 없음. 상세 artifacts/runs/paper_baseline_sixhour_20261006/research/OFFICIAL_CHAT_CONTROL_PRIORITY.json.


### 공식 chat→generate CPU 호출 경로 검증

고정8B 저자 source의 chat/generate 함수를 수정 없이 실행하고 실제 tiny random Qwen3를 연결했다. visual embedding 치환, EOS 설정, config.use_cache=False를 generate(use_cache=True)가 override하는 것, 단1회 실제 generate, caller config 보존을 검증했다(2PASS/4.16s). 이 fixture의8token/forcedEOS는 테스트 한정이며 runtime 후보가 아니다. VisRef Decoder/probes/DPP/parser 없음. Tiny CPU wiring 증거이며 실제8B checkpoint/vision encoder GPU correctness 또는 baseline reproduction 증거가 아니다. 신규GPUdispatch 없음. 다음은 fresh pinned8B official chat end-to-end 대조; 기존 모델 생성·태그 문법·scoring 조건을 별도로 기록한다. 상세 artifacts/runs/paper_baseline_sixhour_20261006/research/OFFICIAL_CHAT_GENERATE_CPU_VERIFICATION.json.


### R21 공식 기본 모델 대조 실행

Kernel arki13/visref-paper-r21-20261006 v1/137287605, source bd0f68d06c2eefab7e7fd1ad44fb202794d44f9cc53883e5423dfecfb1498c62. 실제 공개 author chat→generate, fresh model, no VisRef Decoder/token hooks/entropy/DPP/precomputed visual/reference forward. 종료 후 IDs CPU 복사만 관찰한다. Isolated332PASS/12.97s, 실제 payload131파일908486bytes secret/selection/holdout 없음. 첫 준비의r4 fixture 누락 실패는 pre_dispatch_attempt1에 보존·수정 후 전체 재검증했다. 모든 이전 job terminal 확인 후 제출. 누적27426.992s 보존, R21 미정산. 30분 finisher/session25494; 종료시 frozen source/원문 question/seed/IDs/text를 R20과 대조하고 기존 strict grammar/scoring 별도 보존. 이전 gate 해제 또는 paper fidelity 확정 아님.


### R21 수집·채점 완료 — 자동 마무리 기록

R21 runner=DIAGNOSTIC_COMPLETE, native 완료=3/3; GPU charge=1338.6809999999969s; Gemini 호출=4. 원본/세부 판정은 artifacts/runs/paper_baseline_sixhour_20261006/r21/FINISHER_RESULT.json 및 offline_scoring/results.json에 보존. 정확한 논문 재현 미확정; 목표 service 상태는 이 finisher가 변경하지 않음, 새 GPU dispatch 없음.


### R21 기본 모델 대조 판정과 채점 v2 사전 기록

공식 fresh chat 경로 R21은3/3 normalEOS이고 R20과 모든 generatedIDs/text exact동일. 이 조건에서 project 검증 instrumentation이 출력 문제를 만든다는 가설은 지지되지 않는다. 419의 태그 반복은 공식 경로에서도 생성된다. 고정 VLMEvalKit54a063c5 inference.py의 SPLIT_THINK는 여러 closing 태그에 경고를 남기고 마지막 태그 뒤 prediction을 평가한다. 해당 공개 함수와 경고까지6fixtures exact parity PASS/.02s. 기존 strict parser/gates/scoring 원본 보존, 신규 offline_scoring_vlmeval_split_v1에서 R20/R21의 모든3outputs를 같은규칙으로 재채점한다. EOS 미도달이면 final 없음, denominator3 유지. Final-only payload local audit 후 기존 승인Gemini사용, 신규inference/GPU없음. 이 변경은 scorer 버전이며 VisRef 저자 동일 extractor 확인 또는 controller gate release가 아니다.


### 공식 기본 생성·채점 진단 결과

R21 fresh author chat3/3 normalEOS, 모든 rawIDs/text R20 exact동일; R20 실제 prefix/generated reference도3/3 수치 통과. 이 고정 dev3조건에서 기본 generation interface는 검증됐다. Public VLMEvalKit SPLIT_THINK 규칙을 별도 채점 버전으로 적용하여3final 모두 추출 성공, 두 extractor의prediction 일치, 정답0/3. 419의 이전 missing_response는 strict 태그 처리로 발생한 분석 실패였으며 신규 규칙의 정상 추출과 구분한다. 이전 gates/점수 보존, controller gate 해제/전체MathVista정확도/VisRef adaptive-ST 충실재현 주장은 없음. R20/R21 outgoing payload bytes exact동일해 R21의6calls를 R20에도 재사용, 중복6calls 절약. GPU 누적28765.673s, extraction58calls, 신규GPU없음. 상세 research/VLMEVAL_SPLIT_THINK_RESCORING.json 및 각 offline_scoring_vlmeval_split_v1/results.json.


### 기본 생성의 선행 검증과 sequence MC correctness

선행 검증 기록은 R20/R21의 version/source, 동일 input/seed, 전체 token 일치, EOS, full-context 수치 검증, 공통 versioned scorer를 실제 자료로 대조했다. 과거 strict-tag gate는 변경하지 않았고, 정답0/3을 정확도 성공으로 표현하지 않는다. TSR/DPP는 각각의 동결한 correctness 조건을 따로 통과해야 한다. 독립 CPU 검증에서 실제 tiny Qwen의 S10 complete response를 자연EOS까지 sampling하고, cache 없는 full forward와 Transformers Temperature/TopK/TopP로 전체 sequence logq를 재계산했다. Draw NLL 합과 평균H가3e-5 이내로 일치했다(1PASS/4.23s). EOS 포함, 길이 비정규화, sample별 seed, parent 불변을 확인했다. 실제8B entropy 검증은 아직 남아 있다. 신규GPUdispatch 없음. 상세 artifacts/runs/paper_baseline_sixhour_20261006/research/NATIVE_GENERATION_CERTIFICATE.json 및 SEQUENCE_MC_FULL_FORWARD_CHECK.json.


### R22 TSR sequence-MC 실제 모델 검증

R22 private v1/137293266을 제출했다. Source42f8171670e5c72bf2f84c9fddc2594caf7ea9caa6881267491b7fe460c7c60a, isolated342PASS/12.82s, 실제 payload158파일1560509bytes에서 인증정보/selection/holdout 없음 확인. Native certificate를 bundle 안에서도 hash/settings로 대조하여 이미 검증한 기본 추론을 반복하지 않는다. S10/EOS/비정규화 sequenceNLL/T.6/top50/p.95, last-layer contextual Z, probe/final4096, thinking32768, total context40960, TV/mask/KV/clone-parent 불변을 검증한다. DPP는 실행하지 않는다. 공개 SPLIT_THINK 후처리는 새 run의 명시적 선택이며 sampledIDs/NLL을 바꾸지 않는 것을 CPU에서 확인했다. 과거 gate/score는 그대로 보존한다. 이전 모든 job terminal 확인, session14400s, 이전 GPU28765.673s 유지. 30분 finisher/session29376이 원본·실패·측정 불가 기록을 회수한다. 현재 판정 대기.


### R22 수집·채점 완료 — 자동 마무리 기록

R22 runner=GATE_FAILED, TSR 완료=2/3; GPU charge=2526.8960000000006s; Gemini 호출=4. 원본/세부 판정은 artifacts/runs/paper_baseline_sixhour_20261006/r22/FINISHER_RESULT.json 및 offline_scoring/results.json에 보존. 정확한 논문 재현 미확정; 목표 service 상태는 이 finisher가 변경하지 않음, 새 GPU dispatch 없음.


### R22 TSR 판정과 수치 검증 실패 조사

243은5단계(H 마지막0), 479는1단계(H0)에서 absolute entropy 조건으로 종료했다. 완성된 각 단계의S10/EOS/동일sampling-policy 및 sequenceNLL 재계산은 통과했다. 419는3단계까지 entropy 측정을 마쳤으나 다음 Wait context4834에서 cached eager와 uncached SDPA efficient reference의 TV=.0493963이 동결 기준.02를 초과해 중단했다. 최대 logit차.0332031, argmax 일치. 현재 cache 오류와 backend/FP16+nucleus support 차이를 구분할 증거는 부족하다. Gate를 완화하지 않고 untruncated TV/support 변경량 진단을 추가했으며 관련8tests PASS/2.73s; synthetic fixture에서 raw TV.002가 top_p=.95 경계 때문에 policy TV.051이 될 수 있음을 확인했다(실제419 원인 확정 아님). 다음에는 보존된419 block/Wait만 재생해 cached eager/uncached SDPA/uncached bounded eager를 비교한다. 새 응답 sampling 없이 진단한다. 공통 scorer denominator3, 정답0/3, extraction failure1. GPU누적31292.569s, extraction62calls. DPP/논문 충실재현 미완료. 상세 artifacts/runs/paper_baseline_sixhour_20261006/research/R22_REFERENCE_FAILURE_REVIEW.json.


### R23 제출 승인 검토 대기

동결 source a32ed2a7f81dbf13beb1772306f390e137311edc6b95a5a43b2452601fa93b94, isolated346PASS/11.99s, payload164파일15850527bytes 인증정보/selection/locked 없음. 제출 명령은 실행되지 않았으며 launch 없음/GPU 사용 없음. Private dev419 replay만, session1800s/새sampling/API채점없음의 구체적 승인 대기. 우회 제출하지 않는다. 상세 artifacts/runs/paper_baseline_sixhour_20261006/r23/APPROVAL_REVIEW_BLOCK.json.


### R23 명시 승인 및 제출

작성자 결정: R23 별도 승인. 첫 upload HTTP400 후 원격No runs found(404)/GPUreserved0 확인했으며 모든 실패 bundle은 r23/pre_submission_http400에 보존했다. 재생에 필요 없는 probe자료를 제외하고 동일419 committed_ids/Wait/query와 원본 SHA를 동결한 fixture를 사용했다. 계산 조건은 변경하지 않았다. 재검증346PASS 및 payload165파일1639067bytes에서 인증정보/selection/locked 없음 확인. Private r23 v1/kernel137305444, sourceac8229f7c0b9652c6081919df4213ff22e06f2d44e12a946b94b4ff9868cd6eb 제출 성공. 제출 전 모든 이전 job terminal, session1800s/예상600–1680s. 누적31292.569s는 그대로 유지하고 R23 청구는 회수 시 계산한다. 새 답 sampling/API채점 없음, 과거gate 미해제, DPP 미실행.


### R23 수치 진단 회수

R23 runner=REPLAY_DIAGNOSTIC_COMPLETE; GPU charge=906.4930000000022s, 누적=32199.062s. 새 응답 sampling/API 채점 없음. 기존 gate/실패 보존. 상세 artifacts/runs/paper_baseline_sixhour_20261006/r23/FINISHER_RESULT.json.


### R23 수치 차이 해석

419 context4834의 R22 TV.049396332를 새 sampling 없이 정확히 재현했다. Cached-vs-SDPA raw TV.00837859/support6↔5, cached-vs-bounded-eager raw TV.00474438/support6↔5; 둘 다 nucleus 이후TV.0493963. 제외 토큰 하나의 cached sampling질량.049396336이 TV를 지배한다. Argmax 모두 일치. Eager-vs-SDPA policy TV.0104560/support5동일. 따라서 단순SDPA만의 문제로 단정하지 않으며 cache 오류도 아직 확정할 수 없다. FP16에서 full-query와 single-query 계산의 차이가 top_p 경계로 증폭되는 가설은 관측으로 좁혀졌다. 기존.02 gate 및R22실패 보존. 다음 진단은 고정 공개prepare_inputs_for_generation이 freshcache/positions를 만들어 같은feedchunks를 처리한 결과와 모든cache층/최종logits exact 비교(허용오차 확대 없음)다. 독립CPU13PASS/4.41s 및 고의cache손상검출 확인. GPU누적32199.062s, 추가API없음. 상세 artifacts/runs/paper_baseline_sixhour_20261006/research/R23_NUMERICAL_INTERPRETATION.json.


### R24 제출 보류

R24 frozen sourcecf31b979e27a4fe257e65404ae21fa106c12e505eb582bf4c27b5fd868bf341b, local bundle PASS, payload167파일1651454bytes 인증정보/selection/locked 없음. 제출명령 미실행, launch없음/추가GPU없음. 공개helper freshcache exact 비교 준비 완료, R24 명시승인 대기.


### 실행 오류와 수치 진단을 분리한 profile v3

R23은 hard nucleus support 경계가 TV를 증폭하는 것을 실제로 보였다. 이에 미래 run용 primary_controller_candidate_v3에서 numerical_reference_failure_policy=diagnostic_only를 명시했다. 기존기본값strict와.02 threshold는 유지하며, v3에서는 numerical TV/argmax 차이를 기록한 경고로 남긴다. NaN/Inf, tensor/actual mask/positions/cache shape/prefix 오류, parent 오염은 여전히 중단한다. 이 변경은 새 운영 기준이며 R22 실패를 사후통과로 바꾸지 않는다. Score/entropy scale은 correctness gate가 아니다. IncompleteS10은 측정불가, token/EOS/추출 실패는 독립기록한다. DPP kernel/rank 미공개 처리도 별도 명시가 필요하며 아직 jitter나 무근거rankcompletion을 추가하지 않았다. 현재 GPU dispatch 없음. R24 기존bundle/사전기록은 보존하지만 local source 변경으로 그대로 dispatch할 수 없으며 미래run은 새freeze가 필요하다.


### R25 준비 및 제출 보류

수치 차이를 진단경고로 분리한 profilev3를 동결한 R25 dev3 TSR 준비 완료. Source dd46500675dd076e4f6ada2f84d7d95fd5c0dac17492162865514d3ab2a0ac6f, local bundlePASS 및 payload165파일1595307bytes 인증정보/selection/locked없음. 새diagnostic 검사, 수치정책 외 기존모델/추론/entropy/시간설정 유지. launch없음/추가GPU없음. 작성자R25 또는 후속baseline development포괄승인 대기.


### R25 및 후속 development 실행 명시 승인

작성자가 R25와 후속 development 실험을 승인했다. 준비된 R25 TSR dev3를 실행하고 이후 동일 baseline development범위 실험에 이 승인을 적용한다. 기존 누적GPU승인/livequota검사, selection/locked/압축금지,STOP/실패보존을 유지한다. R23에서 top-p 경계 증폭을 확인했지만 모든cache오류가 없었다는 확정은 하지 않는다. 새R25의 수치 차이는 경고, 실제NaN/mask/position/cache오염은 중단한다. 기존R22판정은 보존한다.


### R25 TSR 재실행 제출

Private R25 v1/kernel137312243 제출 성공. Source dd46500675dd076e4f6ada2f84d7d95fd5c0dac17492162865514d3ab2a0ac6f. 제출 전 이전모든job terminal 확인. Dev243/479/419 동일설정, numerical 차이 경고-only, 실제구조/cache오염검사 유지, DPP미실행. Session14400s/phase13200s, 누적32199.062s 유지하고 R25사용량은 회수 때 반영한다. 30분 확인.


### R25 수집·채점 완료 — 자동 마무리 기록

R25 runner=TSR_CORRECTNESS_COMPLETE, TSR 완료=3/3; GPU charge=4703.889999999999s; Gemini 호출=6. 원본/세부 판정은 artifacts/runs/paper_baseline_sixhour_20261006/r25/FINISHER_RESULT.json 및 offline_scoring/results.json에 보존. 정확한 논문 재현 미확정; 목표 service 상태는 이 finisher가 변경하지 않음, 새 GPU dispatch 없음.


### 2026-10-06 자연 응답 entropy 전환 — 로컬 구현 진행
작성자 결정: 추가 prompt 없이 실제 생성 마지막 </think> 이후의 y를 사용하고, delimiter가 없으면 MathVista extractor 답의 마지막 일치 span을 사용한다. 해당 원본 token 위치의 full-vocabulary softmax predictive entropy를 평균한다(mean aggregation=paper-underspecified). 기존 R25의 강제 close/S10 sequence NLL/별도 final generation 결과는 다른 운영 variant로 보존하며 새 방법의 증거로 사용하지 않는다.
새 natural_response_runtime.Decoder.finish는 fixed close/newline/answer request 없이 실제 sampled token만 feed한다. block의 s1 차용 minimum-length suppression도 새 경로에서 0으로 고정하여 종료 token 차단을 제거했다. 로컬 관련11 tests PASS/1.75s: 실제 sampled closing/EOS 및 feed 목록, suppression 비활성화, 마지막 delimiter/fallback span/mean predictive entropy/비정상 logits. 이는 제어 fixture와 계산 단위 검증이며 실제8B GPU 검증이 아니다.
아직 새 controller 연결, boxed query suffix 제거, step-boundary의 sampled close 미반영/재sampling과 별도 final 생성 정책 정리, 실제 MathVista extractor fallback 연결 및 새 frozen GPU run이 남아 있다. 새 GPU dispatch 없음; 기존 STOP/raw/실패 보존.


### 2026-10-06 natural predictive controller 로컬 연결
새 별도 profilev4는 dataset_query(추가 boxed suffix 없음), l_min0, S 비적용, raw T1 full-vocab answer-token mean entropy, 실제 측정 응답 최종 재사용을 명시한다. natural_response_runtime.run_method에 연결: TSR은 block에서 실제 sampled close를 response clone에 commit하여 재sampling/고정태그 치환을 피한다. DPP는 z→visual injection 이후 문맥이 바뀌므로 이전 draw를 재사용하지 않고 자연 sampling한다. entropy가 높은 경우 parent reasoning context에 논문의 Wait, think more만 append한다. Parent clone 불변 검사 유지. Final 별도 생성 제거. Token-limit은 측정 가능해도 정상 완료로 처리하지 않는다.
관련13 local tests PASS/1.75s(실제 sampled boundary commit, no forced feed, no termination suppression, 한 response/step 및 측정 응답 재사용, answer scope/계산). 운영 controller fixture이며 live8B 증거 아님. 기존 R25 baseline certificate는 boxed input이므로 dataset_query v4의 native 동등성을 증명하지 않는다. 새 remote runner/config freeze 및 MathVista extractor fallback 연결, native dev smoke/GPU 검증은 남아 있다. 현재 GPU dispatch 없음.


### 2026-10-06 MathVista fallback 연결 수정 및 새 runner 초안
OfficialExtractor와 실제 고정 upstream 함수에 새 last-span helper를 연결하는 로컬 검증에서 upstream이 요구하는 pid 전달 누락을 발견하여 수정했다. Gold/precision 배제는 유지한다. mock transport를 주입한 실제 upstream five-shot extraction→마지막 span→mean entropy 검증 포함14 tests PASS/1.76s. 외부 API 호출 없이 검사했다.
natural_tsr_runner.py 초안은 paper_r26/dev243,479,419/dataset_query/profilev4만 허용하며 기존 boxed native certificate를 재사용하지 않는다. 로컬 syntax 확인 완료. remote extractor transport 및 native 비교 검증 연결은 아직 미완료이므로 GPU 제출하지 않았다. delimiter 없는 EOS/token-limit block도 extractor fallback에 전달하는 제어 경로 보완이 남아 있다. 작성자 결정의 fallback을 구현 완료로 주장하지 않는다.


### 2026-10-06 현 설정 사전 검토 및 설명 정정
새 GPU 제출 전 v4를 검토했다. 평균 answer-token predictive entropy는 본문 response-distribution entropy와 다른 variant이고 .25의 의미를 자동 승계하지 못한다. 중요 정정: VisRef Alg1 lines16–17 및Sec4.1.2는 stopping 후 final answer sampling을 명시하므로, 앞선 ‘별도 final generation이 논문에 없는 구현 추가’ 설명은 잘못이었다. v4의 측정 응답 재사용은 절약형 variant로 기록하며 정확재현 주장 금지. inherited 태그문법 gate/token-limit fallback/remote extractor/full cfg/DPP live 검증이 남아 있고 GPU는 제출하지 않았다. 상세 docs/paper_baseline_alignment/CURRENT_PROFILE_V4_REVIEW_20261006_KO.md.


### 2026-10-06 사전 검토 반영: 자연 block 및 논문 final 정책
natural block에서 repeated/misplaced think_start 및 missing/empty thinking 문법 gate를 제거했다. 종료 구분자/EOS/token limit를 실제 sampling 그대로 기록하고 last-layer norm hook으로 contextual Z를 수집한다. 종료 차단 없음. TSR의 no-delimiter EOS와 block token-limit 응답도 answer extractor scope 경로에 전달하되 token-limit을 정상 완료로 승격하지 않는다. DPP는 reasoning boundary가 없으면 재주입 단계 정의를 임의로 만들지 않는다.
중요 정정을 구현에 반영: v4 final_answer_policy=separate_generation_v1로 복원하여 Algorithm1의 stopping 후 final sampling을 따른다. 응답 재사용은 명시적 별도 variant로만 유지한다. 두 정책의 별도 호출 횟수/결과를 local 검사했다. 실제 tiny Qwen의 cached generation 각 token predictive entropy를 cache 없는 full-forward logits softmax entropy와 대조하여1e-5 이내 일치를 확인했다. 관련17 tests PASS/4.71s(이후 block token-limit 처리 추가 확인은 별도 실행). 실제8B/remote extractor/GPU freeze 미완료. 원 논문의 response entropy와 새 mean predictive entropy는 여전히 다른 실험 정의다.


추가 확인: no-delimiter token-limit에도 MathVista extractor fallback을 수행하여 H를 기록하면서 status=GENERATION_FAILED/token_limit을 유지하는 fixture를 확인했다. 자연 EOS의 동일 응답은 정상 완료로 구분한다. 현재 관련18 tests PASS/4.72s. 새 GPU dispatch 없음.


### R26 제출 — 자연 answer entropy 운영 설정 검증
Private arki13/visref-paper-r26-20261006 v1/kernel137325216 제출 성공. source094648c21fdaac47a83423d192ee63295df837ef7858e7e4399f3c3c51213441, notebookd0d3453e2bdb8d60e2f2b64c7ea8e2f8bed0104086bd90d1992c5f72c7c1c380. Isolated371PASS/18.18s, payload178files1651327bytes 인증정보/raw/selection/holdout없음. 작성자후속development승인 범위의243/479/419만, DPP미실행. dataset_query/자연태그/종료차단없음/mean answer-token predictive H/별도final sampling. k10/.25/8192thinking/4096response 운영값, 논문 entropy exact주장없음. Request 없는 exact official extractor cache와 pending 요청 보존.
제출 전 모든과거job terminal 확인. 누적36902.952s유지; session7200/예상1500–7080s/실제dispatch conservative1800–7080s. launch estimate_status가 generic R5 label을 기록했지만 실제 기준은 R26 사전등록의 새자연답변 smoke이며 R5설정을 재사용한 의미가 아니다. Finisher session63816으로30분 주기 확인·종료 후 원본/추출요청/사용량 회수, 새dispatch없음. STOP보존.


### 2026-10-06 answer-token entropy 기준 확정 및 DPP 운영 구현
작성자 결정: 추론 이후 y에 해당하는 token 위치만 predictive entropy를 계산하며 반복 response sampling은 하지 않는다. EOS제외/mean over answer tokens는 기록한 운영 선택으로 유지한다. 논문 response entropy 해석과 다르다는 점을 실행 중단 조건으로 사용하지 않는다. R26은 이미 이 단일응답 측정 경로로 동결·실행 중이며 변경 없음.
후속 DPP용 explicit profilev5: Eq kernel + 상대jitter1e-6*max(diag(L)); fixed floor(.3N) 유지. 과거 unregularized selector와 모든rank실패는 그대로 보존한다. jitter가 저자값이라는 주장은 하지 않는다. T<m에서도 수치적으로 정해진 fixed m을 선택하되 모든 alignment가0이면 임의uniform을 만들지 않는다. 독립 brute-force greedy logdet 대조, scale invariance, invalid input 검사와 실제 tinyQwen contextual T1→m3 선택/원본projected V재주입→단일answer-tokenH→별도final sampling을 확인했다(관련7PASS/5.18s). Live8B DPP 미검증; 새GPU제출없음. 새 파일 operational_dpp.py/natural_dpp_runtime.py는 R26 이미동결된bundle에없고 실행중인job에영향없음.


### 2026-10-06 작성자 우선순위 교정 — TSR 먼저
DPP 후속 구현/실험은 TSR 실제8B 검증 완료까지 중단한다. 이미 만든 local파일/검증자료는 보존하고 R26에 반영하거나 GPU로 실행하지 않는다. R26 methods는 TSR만이다.
기존 Kaggle Model Proxy Gemini extraction 유지. R26은 staged exact-prompt extraction이고 no-delimiter에서pending이면 local승인추출 후 새로운run version에서만진행한다.
R26 누적제한은 thinking block sampled IDs합8192, response/final 각4096, 입력포함문맥40960, 단계10이다. Probe/final 및Wait까지 합친 전체generation 누적cap은현재없다. 생성양은영역별로별도보고하며논문상한이나전체총량제한으로오인하지않는다. Livejob immutable/30분확인 유지.


### 2026-10-06 누적 thinking14k 운영 기준 복원
Figure3에 근거한 누적14k 운영 기준을 복원한다. 다음 TSR용 primary_tsr_14k_candidate_v6.json은 thinking_cap=block_cap=14000, methodsTSR만, DPP보류를 명시한다. Paper Figure3의14K는 thinking-budget 비교점이고 모든Table1baseline의공식상한이라고 주장하지 않는다. 누적 reasoning block generated IDs를문항당합산하고 entropy response/final추가비용과 context별도보고. 현재R26source/config는변경없으며8192결과로보존한다. 다음dispatch는R26종료회수및실제quota확인후에만가능.


### 2026-10-06 Figure3 budget 설명 재정정
현재구현은 single chain내Wait 단계들의합이며 Figure3 multiple independent parallel chains합산이 아니다. 앞선14k로변경하면Figure3정의까지맞춘다는설명은잘못이었다. v6단일chain14000은 draft diagnostic으로명시하고Figure3재현설정으로자동dispatch하지않는다. Figure3용14k는 문제당독립chain 전체thinking합계이며 별도sharedbudget controller가필요하나 TSR검증전에는구현/실험범위를확장하지않는다. R26은단일TSR8192동결실험그대로보존. 실제현재설정과향후요구조건을구분한다.


### 2026-10-06 인계 검토 — 다음 Baseline 구현 순서

현재 작업 목표는 논문 Baseline 구현이다. 검토와 실행 계획에 인계·논문·코드·동결 설정을 대조했다. R26은 작성자가 의미가 없다고 판단해 직접 중단했고 CANCEL_ACKNOWLEDGED를 확인했다. 기존 finisher의 최소 원본·사용량 보존만 유지하며 상세 R26 분석·재실행은 다음 작업의 선행 조건으로 삼지 않는다.

다음은 CPU EOS/추출/집계 수정 → 같은 raw query native ST → 작성자 지정 단일응답 H의 TSR → DPP30이다. CPU에서 EOS 뒤 final/Wait 호출, 빈 cache fallback 미완료, 최종 추출 실패와 generation 완료의 혼동을 확인했다. 관련20 tests PASS는 실제8B 성공을 뜻하지 않는다. Figure3 parallel/14K·압축·성능 튜닝은 현재 목표에 추가하지 않는다. 이번에는 계획·진단만 작성했고 runtime 수정·새 GPU/API 실행은 없다.


### R26 자동 회수
{"status": "R26_COLLECTED", "runner_status": "RUNNING", "samples": [], "charged_s": 7200, "cumulative_charged_s": 44102.952, "pending_extractions": 0, "scoring_status": "PENDING_OFFLINE", "paper_reproduction_verified": false, "new_gpu_dispatch": false}


### 2026-10-06 과도한 검증 조건과 EOS 단정 정정

논문 §3/§4.1.2를 다시 대조했다. CPU 진단은 no-delimiter EOS 이후 final/Wait 호출을 확인했지만 호출 자체의 버그를 증명하지 않는다. Probe EOS와 parent reasoning 종료를 구분해야 하므로 일괄 terminal/호출금지 권고를 철회한다. Author 미공개값·native3/3 자연종료·전token logit보존·장문 full parity를 필수 실행 gate로 확대하지 않는다. 기존 작성자 지정 H와 normal-boundary 별도 final을 유지하고 실제 token대응·추출 연결·상태/집계만 보완한다. 정정 상세 기록 §8 참조. 현재 실행 순서는 짧은 계획을 따른다. Runtime 변경·새GPU/API실행 없음.


### 2026-10-06 간단한 Baseline 경로 구현 / R27

짧은 계획의 native재사용·TSR코드정리·CPU검증 완료. R27은 실제8B TSR만 확인하며 새 모델/threshold탐색 없음. 결과 확인 후 DPP30으로 진행. STOP·누적소비·동결원본 보존; selection/locked/압축/Figure3범위 확장 없음. 실행 근거.


### 2026-10-06 R27 대기 중 독립 확인

기존 R18 final의 cache-only scoring 재검산 동일/분모3 유지/외부호출0. TinyQwen entropy oracle은 token/logit 한 칸 이동 오류를 검출했다. 기존 pending 추출 도구에 --version r27을 연결하고 terminal collection 이전 거부를 확인했다. 실제추출·추론·DPP 작업은 하지 않았고 실행중 bundle도 변경하지 않았다. 근거.


### 2026-10-06 R27 결과 확인

R27 첫 actual8B TSR 경량경로는479/419 정상동작. 243limit/pending,0/3채점은분리보고. 다음은계획의DPP30연결·검증; threshold조정·S10복원·243재실행·압축/Figure3확장없음. 근거.


### 2026-10-06 R28 DPP30 연결

계획의 DPP30 연결 완료/로컬검증통과. R28은DPP만검증하며 ST/TSR재추론·threshold/상한조정·selection/locked/압축은하지않는다. 결과를확인해 실제선택/재주입/H/종료를보고한다. 설정과 근거.


### 2026-10-06 R28 원문·Figure 일치 검토

R28은작성자H+regularizedDPP를쓴개발구현으로정확재현·모든Figure실험완료를선언할수없다. 기본흐름과평가/시각분석범위를구분한다. 기존실행보존/결과확인후판단하며새실험범위를자동확대하지않는다. 항목별 검토.


### 2026-10-06 논문 수식 경로와 Figure 분석 코드

논문 수식 후보 v7 구현 및 로컬검증 완료. 기존 R28은 작성자 H+jitter 실험으로 그대로 보존하며 새 후보 결과로 혼용하지 않는다. 전체 성능·Figure 재현에는 실제 비교가 필요하고 저자 미공개값/degenerate logdet는 임의 공식값으로 해결하지 않는다. 다음 dispatch는 별도 run config/동결·예산 근거를 갖춘 경우에만 수행한다. 변경과 검증.


### 2026-10-07 구현·최소검사 완료 / 실제 모델 검증 준비

구현→최소검사 완료, 실제모델 검증의 다음은R29 419 TSR smoke다. R28은회수·분석·정산완료/완전final0이므로성공으로선언하지않는다. 원래kernel fixed-m의zero determinant 동률 처리 추가는 공개Eq8목적값을바꾸지않되운영tie선택으로명시한다. R29 remote upload 미실행;scope근거확인후허용될때만제출한다. 고정설정과 판정.
