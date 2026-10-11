# VisRef 재주입 형식·위치 엄격 재독 (2026-10-11, Fable 5.1 서브에이전트, 읽기 전용)

주 세션 대조: P:203, P:240-241, P:244, P:271-272, P:295-297, P:381-386 원문과 Fig. 1(page-01)·Fig. 2(page-04) 그림을 직접 확인함. Qwen-LA([9]) "copy … before <REFLECTION>" 인용은 서브에이전트의 WebFetch 요약이라 **미확인**.

## 결론 요약
- 논문이 명시한 것과 현재 구현은 모두 일치: 재주입 대상 = 원본 시각 토큰 V의 부분집합("Vi ⊆ V", P:271-272; "visual token embeddings extracted from image I", P:240-241; Fig. 2 "Original visual tokens"), 순서 z_k → Ṽ_k → H_k → "Wait, think more" → z_{k+1}(Eq. 3, Alg. 1 4/10/12/17행, Fig. 1(b)·Fig. 2), H_k와 최종 답 문맥에 Ṽ_k 포함, 이전 Ṽ_i 유지, 토큰을 실제로 문맥에 추가("appending all visual tokens", P:385; 재가중 해석은 P:381-382와 모순).
- 논문 어디에도 없는 것(본문·그림·부록·arXiv HTML·CVPR 보충자료): 태그·구분자, `</think>` 처리, position id, 토큰 순서, probe·최종 답 프롬프트(Fig. 2의 "Answer:" 라벨뿐).
- 따라서 "잘못 읽은 문장"은 찾지 못함. 대신 **논문이 정하지 않아 우리가 만든 접합부 `[Ṽ_k] </think>`**가 증상(답 안 `</think>` 재출력 VisRef 157/366 vs TSR 25/352, 답 길이 증가)의 가장 유력한 원천으로 지목됨. 모델 훈련 분포에서 시각 임베딩은 항상 user 턴의 `<img>…</img>` 안에 있음(HF modeling L244, L295).

## 놓쳤을 수 있는 것 (증상 영향 순)
1. Ṽ_k 바로 뒤 `</think>`(우리 구성, 논문 무언급; Fig. 2는 Ṽ_k 뒤에 "Answer:"를 그림).
2. 감싸기·구분자 부재(논문 침묵; smoke1 소표본의 `<img>` 감싸기 악화 기록만 있음).
3. 운영점: 저자 8B는 단계 ≈615토큰 자연 종료·다단계(Fig. 2 그래프), 우리 1B는 첫 단계 641/1,000이 1,024 상한 절단 → Ṽ_k가 문장 중간 뒤에 놓임.
4. 논문 내부 불일치: Table 5(TSR−ST 0.6 s)는 대부분 k=1 정지를, Fig. 2/3은 다단계 궤적을 시사(장부 A-4, C-12).
5. position id·토큰 순서: 논문 무언급, 연속 위치는 분포 내 → 영향 낮음(추론).

## 저비용 진단 후보 (k=1 자연 종료 행, 저장된 z_1·V_1 재사용, 답 생성만)
- A 텍스트 큐: `[z_1][V_1]</think>\nAnswer:` 등 vs 현행
- B 구분자: `[z_1][V_1]\n</think>`, `[z_1]\n[V_1]</think>`
- C 감싸기: `[z_1]<img>[V_1]</img></think>`
- D 내용 vs 형식: V_1을 무작위 m개 시각 토큰 / 관련성 상위 m개 / 같은 수 텍스트 임베딩으로 대체
- E 위치: 선택 토큰에 원 이미지 position id 재사용
- F probe 답 재사용: 저장된 probe_answer 채점
- G 접합부 로짓(forward 1회): `[z_1][V_1]` 뒤와 `</think>` 뒤 다음 토큰 분포(`</think>` 재출력 확률)를 TSR과 비교
- 지표: `</think>` 재출력률, 답 길이·상한 도달률, ST 대비 답 변경률, 정답률. 배치 구성 고정.

## 저자 문의 문항(형식·위치)
1. 단계 경계의 정확한 토큰열(`</think>` 억제 후 Ṽ_k 삽입 → 바로 "Wait, think more"인가, 개행·구분 텍스트가 있는가).
2. Ṽ_k를 맨 임베딩으로 넣었는가, `<img>`/`</img>` 같은 이미지 구분자나 "Image-2:" 같은 텍스트로 감쌌는가.
3. 재주입 토큰의 position id(연속값 vs 원 이미지 위치)와 토큰 순서.
4. 재주입 벡터 = projector 출력인가 LLM 층 은닉 상태인가; z_k의 "text token embeddings"는 어느 층인가.
5. 정지 후 최종 답 앞 토큰열(`</think>` 직후 자유 생성 / "Answer:" 프롬프트 / probe 답 재사용), 엔트로피 probe의 프롬프트·표본 수·온도.
6. 1B 실험의 단계당·총 생성 상한과 절단 처리.
7. MathVista 8B에서 TSR·VisRef 평균 단계 수(Table 5와 Fig. 2/3의 조화).
8. Fig. 2 attention ratio의 "visual"에 재주입 토큰이 포함되는가.
