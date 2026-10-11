# 독립 검토 요청: `visref_fill` 실험을 그대로 실행할지 (2026-10-11)

## 배경 (요약 — 원자료로 직접 확인할 것)
- 목표: VisRef(arXiv 2603.00207) Appendix Table 6, InternVL3.5-1B × MathVista testmini 1,000문항, ST 46.1 / TSR 48.5 / VisRef 52.0 재현.
- 본 실험 b_main(ST/TSR/VisRef=`visref_raw_rank`)은 868/1,000 완료, 마지막 세션 s9(132문항) 실행 대기/중. b_main 결과 파일: `VisRef/repro/kaggle/runs/b_main_s{1..8}/output/results/results_{0,1}.jsonl` (한 행 = (pid, method)). 채점은 `report.py`의 `rescore_rows(rows, data)`(scorer v3, data = `VisRef/data/raw/mathvista_testmini.json`)로 재채점 후 사용(바뀌는 행 4개뿐이라 기록된 `correct`를 써도 결론엔 영향 거의 없음).
- s1–s8(868): ST 58.1 / TSR 60.0 / VisRef 59.9. 짝 차이 TSR−ST +2.0 (CI 0.2~3.8), VisRef−ST +1.8 (−0.2~3.8), VisRef−TSR −0.1 (−2.1~1.8). 논문 +2.4 / +5.9 / +3.5.
- 사전등록된 다음 실험(EF 발췌 "원인 분리 실험 사전등록 수정"): `visref_fill` = DPP greedy가 rank에서 멈춘 뒤 εI(1e-6×평균 대각)를 더해 정확히 m=⌊0.3|V|⌋개까지 시각 토큰을 채워 주입. 그 외는 visref_raw_rank와 동일. 1,000문항 전부, Kaggle 한 세션 11.5h. 이중 probe 기록(V_k 있는/없는 H_k, 결정엔 H_k만) 포함. 해석 규칙: fill − TSR CI 하한 > 0이면 "주입량 부족이 원인" 지지. 80% 최소 검출 효과 약 2.6%p(이전 검토자 계산, 불일치율 8.8% 가정).
- 아직 구현·실행 전이다(결과 없음). 사전등록 수정은 결과 전이므로 가능하나, 수정 이유는 기록해야 한다.

## 새로 나온 사실 (2026-10-11 주 세션 분석, `diag/analyze_k1_injection.py/.json`)
- CRN 설계상 k=1에서 VisRef와 ST의 차이는 최종 답 직전(`</think>` 앞)에 붙인 V_1뿐(`methods.py` `_end_think`).
- VisRef k=1 621문항: ST 420 vs VisRef 420(한쪽만 정답 11:11), 예측 같음 591/621.
- 첫 단계 자연 종료(close) 281문항: VisRef가 ST와 예측 다름 1/281 — 잡음 기준선(TSR vs ST, 이론상 동일 답) 1/282와 같음. 상한 절단(cap): 25/324 vs 기준선 15/337, 방향 10:9.
- 주입 부수효과: 답 안 `</think>` 재출력 VisRef 310/621 vs ST 61.
- 주 세션 해석(검증 대상): 자연 종료 문항은 답이 생각 안에서 확정돼 주입량을 늘려도 바뀌지 않을 것 → fill이 정보를 줄 수 있는 곳은 절단 문항·k>1 문항뿐 → 1,000문항 11.5h의 실효 검출력이 사전등록 계산보다 낮을 수 있다.
- 다른 관찰: 이어가기는 거의 전부 단계 상한(1024) 절단 뒤에만 일어남(TSR close→continue 6/268). fill은 V 내용이 바뀌므로 probe H와 정지 결정, k 분포도 바뀐다.

## 자원 제약
- Kaggle GPU 쿼터: 주당 30h(2차 자료, 공식 미확인). 남은 양 추정 약 17–18h(이번 주, 리셋 시점 미확인). b_main 재실행 여유를 남기는 것이 사용자 원칙(GPU 우선순위: b_main + 재실행 여유).
- 무료 Colab은 st_official(공식 설정 ST, 100문항, 긴 생성) 중이며 일일 한도로 끊김.

## 검토해 줄 것 (독립적으로, 주 세션 해석에 동의할 의무 없음)
1. 주 세션의 "자연 종료 문항은 fill로도 안 바뀐다" 추론이 데이터로 정당한가? 반례 가능성(예: fill은 주입량이 3–5배라 효과가 비선형일 수 있음, 자연 종료 문항에서도 V가 정지 판정/k를 바꿀 수 있음)을 평가.
2. 원자료로 fill이 VisRef/TSR과 달라질 수 있는 문항 비율과, 그 하위집단에서의 기대 검출력(최소 검출 효과)을 다시 계산. 전체 1,000문항 기준 최소 검출 효과가 어떻게 바뀌는지. 계산 근거와 가정을 명시.
3. 선택지 비교와 권고:
   (A) 사전등록대로 1,000문항 11.5h 실행
   (B) 축소/재설계(예: 표본 수 축소, 특정 하위집단만, 정지 규칙 고정 등) — 사후 표본 선택이 편향을 만들지 않는 방식인지 검토
   (C) fill 실행 안 함(현재 결과 + 한계 기술 + 저자 문의로 마무리)
   (D) 같은 GPU 시간을 다른 질문에 쓰기(제안이 있으면)
   각 선택지가 재현 보고서에 실제로 어떤 결론을 더하는지(결과가 양/음일 때 각각), 비용, 위험을 정리.
4. 이중 probe 기록(V_k 유무 H) 등 fill에 묶인 부가 측정의 가치도 평가.

## 규칙
- 수치는 직접 계산하거나 파일:줄로 인용하고, 확인 못 한 것은 "미확인"으로 표시.
- 코드·데이터 파일을 수정하지 말 것. 계산 스크립트는 `/tmp/claude-0/-home-claude/d6b6d0f0-3759-50dd-854f-c6669aba76e9/scratchpad/fill_review/` 아래에만 만들 것.
- 보고서는 한국어로 `/tmp/claude-0/-home-claude/d6b6d0f0-3759-50dd-854f-c6669aba76e9/scratchpad/fill_review/fill_decision_review_20261011.md`에 저장(결론 → 근거 → 계산 → 권고 순, 3–5쪽 이내). 최종 메시지에는 핵심 결론과 권고, 주요 수치를 요약.

## 입력 파일 (모두 `/mnt/user-data/uploads/VisRef/` 아래)
- `repro/audit/review_inputs/fill_decision/EF_excerpt_L1259_end.md` (실험 흐름 기록 발췌: 사전등록·관찰·분석)
- `repro/audit/design_review_variants_20261010.md` (이전 설계 검토 보고서)
- `repro/diag/analyze_k1_injection.{py,json}`
- `repro/visref/methods.py`, `repro/visref/dpp.py`, `repro/run.py`, `repro/visref/scoring.py`, `repro/report.py`
- `repro/SPEC_LEDGER.md` (구현 근거 장부, 맨 앞 "핵심 판단 요약")
- `references/papers/VisRef_fulltext/VisRef_plain.txt` (논문 본문)
- `data/raw/mathvista_testmini.json`
- `repro/kaggle/runs/b_main_s{1..8}/output/results/results_{0,1}.jsonl`
