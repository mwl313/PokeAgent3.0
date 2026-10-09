# PokeAgent 3.0 — 프로젝트 현황판

> 갱신 규칙: 체크포인트(머지/게이트/측정)마다 갱신하고, 작성 시점 기준 SHA를 항상 기록한다.
> 최초 작성: 2026-10-09 (아리아). 본 문서는 미니DC 코덱스 세션이 커밋하고, 같은 작업에서 main으로 fast-forward 승격됐다.

## 0. 작업 구분 (중요)

- **맥미니 = 엔진 제작 라인**: Rust 네이티브 엔진 포팅(기술/특성/아이템/동적 클로저)과 fixture 검증까지.
- **미니DC = 최적화/측정 라인**: 그 이후의 처리량 최적화, 정확성 게이트, PPO 스모크, 벤치마크 기록 전부.
- 두 라인은 같은 리포(mwl313/PokeAgent3.0)의 브랜치로 존재하며, 현재는 아래 §1의 통합 정본 하나로 수렴돼 있다.

## 1. 브랜치 정본

| 항목 | 값 |
|---|---|
| **통합 정본** | `optimization/pa3-realpolicy-throughput` @ `2540b82` (10-09 12:15) + 본 문서 커밋 |
| 포함 관계 | 엔진 줄 전체 포함(engine-only 커밋 0), 최적화 27커밋 추가 |
| 엔진 줄 | `mac/long-horizon-engine-tail` @ `54d0ff1` (10-08 18:44) — 정본에 흡수 완료, 이후 신규 커밋 0(휴면) |
| GitHub main | `fa723ed` (10-07)에서 **2026-10-09 fast-forward 승격 (본 문서 포함)** |
| 잔존 ref(미통합) 11개(로컬+원격 중복 포함): move-batch-2 계열 2 + delayed-status 2 + mac 포크 3(hazards, hazard-setters, revival-blessing-v2) + 스캐폴딩 4(agent/*) — 소검증 후 흡수/아카이브 (삭제 금지) |

## 2. 여정 요약 (여태 무엇을 했나)

### 2.1 엔진 라인 (맥) — 엔진까지

- 네이티브 Rust 배틀 엔진(PyO3 `pa3-engine`) 포팅: 기술/특성/아이템을 feature family 단위로 커밋, 각각 fixture 증인과 코퍼스 재생성 동반.
- `fa723ed`(10-07 머신 핸드오버): 374 moves / 141 abilities / 166 items / 703 fixtures.
- `54d0ff1`(10-08 롱호라이즌 테일): 486/515 moves(29 blocked), 200/223 abilities(23 blocked), 아이템 166/166, 동적 클로저 33 callers 중 11 blocked, 코퍼스 1,148 fixtures / 25,134 decision boundaries.
- 훈련 풀 1,137팀 전량 자연완결(100%), 운영 오류 0, `readiness_check` 10/16.
- 이후 휴면(신규 커밋 0). 재개 시 잔여: blocked 이동 29, 특성 23, 동적 클로저 11, 코퍼스 전수화, readiness 6기준.

### 2.2 최적화 라인 (미니DC) — 그 이후

- **v1.1** (10-08): 실정책 롤아웃 최적화 Phase 0-6 — 수집(actor-only) 15.07 → 103.44 games/s(6.86x). 고정 관측 배치, packed 후보 wire, packed prefix walk, FP16 autocast, inference_mode, packed 롤아웃 행. 듀얼 액터-온리 162.73 games/s. all-in 20.87 → 26.43 games/s(단일 카드, 당시 계약).
- **v2.0** (10-09 새벽): P0 정확성 교정(지표 집계, exact row-weighted accumulation), M0 학습자 단계 분해, P1.3 스트리밍 물질화(peak RSS 18.0 → 7.34 GiB), P2.3 micro배치 스윕(256 → 1024: all-in 24.66 → 31.57, 3회 확인 32.80).
- **v3** (10-09 02:15): 양좌석(현재 정책 2석) 수집 계약으로 전환 — 19.37 games/s, 26.87 rows/경기, learner rows/s 460 → 520. PA3-8M 실그래디언트 파리티 게이트, DDP 전역 목표 수학·런처(DDP 집합연산 불일치로 end-to-end 블록 확인).
- **v4** (10-09 오전): C0 엔트로피/KL 정규화기 그래디언트 복원(학습 수학 교정), C1 단일 GPU oracle 확정, D0/D1/M1 DDP 고정-step 집합 프로토콜 + 수동 all-reduce(파리티 통과), D2 실행기 A/B(DDP 기본 유지, 통계 동급), F0 전스택 병목 아틀라스(업데이트 구간 GPU-busy 95.4% 등), F1 최종 패널 — **단일 20.53 / 듀얼 24.67(DDP), 25.60(manual) games/s**.
- 모든 수치는 `docs/perf/` 보고서와 `runs/perf/` 원본 JSON에 기록 (아래 §6).

## 3. 엔진 준비도 (readiness)

- `readiness_check`: 10/16 PASS — 실패 2, 3, 5, 8, 15, 16 (full-scope 커버리지/코퍼스 계열).
- 커버리지(10-08 기준, 최신은 `engine/ENGINE_SCOPE.md`): 이동 486/515, 특성 200/223, 아이템 166/166, 동적 클로저 33 callers 중 11 blocked.
- 팀풀 1,137팀: 자연완결 100%, 운영 오류 0.
- 100M 본학습: 미승인 상태가 정상 (조건 = readiness 16/16 + 사용자 승인).

## 4. 처리량 정본 (v4 F1 패널, 2,048경기, 양좌석 계약)

| 항목 | 단일 GPU | 듀얼 GPU |
|---|---:|---:|
| all-in games/s | 20.53 (19.44-20.96) | 24.67 (DDP) / 25.60 (manual) |
| 수집 | 22.1s (약 93 games/s) | 12.7s/랭크 |
| PPO 업데이트 | 77.1s | 58.5-61.5s/랭크 |
| 미세배치 | 1024 | 256 (기본값 잔존 — P0 교정 대상) |

- 측정 SHA: `fbf878e` (이후 커밋은 문서/프로비넌스만). 상세: `docs/perf/V4_OPERATIONAL_FRONTIER_REPORT.md`.
- 물리 상한(계약 유지 시, 추정): 두 V100 합산 이론 천장 약 110-115 games/s, 실용 목표 40-55.
- 100M 소요 예상(추정): 현 속도 약 46일 → P0 반영 시 약 34-37일.

## 5. 다음 최적화 작업 (권장 순서)

| # | 작업 | 근거(실측) | 예상 효과(추정) | 검증 |
|---|---|---|---|---|
| 1 | ~~듀얼 micro 1024 A/B~~ **완료·승격 (v5 W1)** | 실측: 듀얼 256 median 26.25 → 1024 median 32.50 games/s(+23.8%, 3회 반복, 게이트 전부 PASS), update 57.5→42.2s/랭크 | 런처 기본값 1024로 승격 + 확인 런 32.50 | `docs/perf/V5_MICROBATCH_AB.md` |
| 2 | 수집 record columnar화 | record 7.9s/22.1s(36%) | record 약 2-3s, 수집 -25-30% | recompute gate + 후보 wire oracle 유지 |
| 3 | 학습자 물질화 columnar + 후보 u8 | 물질화+선택 12.5s/77.1s(16%) | update -8-11% | streaming equivalence 테스트 재사용 |
| 4 | pinned + async H2D (수집/학습) | H2D 합 약 4.5s | -2-3s 전체 | profiler 타임라인 overlap 증거 |
| 5 | optimizer 블록 정리(flat isfinite 등) | optimizer 8.5s/77.1s(11.0%) | -3-5s | global finite/skip parity |
| 6 | digest/체크포인트/로그 정리 | 런처 오버헤드 약 9s/런 | 런당 수 초 | 증거 요건 유지 |
| 7 | P2: rolling slots, round당 encode 1회화, CUDA Graph PoC | tail 미측정, 그룹별 encode 2-4회/round | 약 5-10% | rolling equivalence 테스트 |
| 제외 | torch.compile/Triton | Triton이 CC 8.0+만 지원, V100 fp16 tl.dot open bug | 해당 없음 | — |

- 계약 변경(별도 승인): 4 epochs→2, 관측 f16 wire, 모델 축소 — 천장을 움직이지만 학습 수학 변경이라 별도 A/B 필요.
- 미착수 트랙: 샘플효율 E0-E4 (`PokeAgent3_RL_Sample_Efficiency_Research_2026-10-08.md`).
- 로드맵 및 옵션 분석: `docs/PokeAgent3_Optimization_Roadmap_2026-10-09.md`

## 6. 기록 인덱스 (잃지 않도록)

| 기록 | 위치 | 상태 |
|---|---|---|
| 최적화 보고서 20종 | `docs/perf/` (BASELINE → V2_* → V3_* → V4_* → FINAL_REPORT) | git 추적 중 |
| 계획 문서 | `docs/PokeAgent3_Ultimate_Throughput_Optimization_Plan_v1.1.md`, `..._Optimization_Master_Plan_2026-10-08.md`(v1), `..._Master_Plan_v2.0_...`, `..._Extreme_Optimization_Master_Plan_v3.0_...`, `..._Correctness_First_Extreme_Optimization_Plan_v4.0_...`, `..._RL_Sample_Efficiency_Research_2026-10-08.md` | git 추적 중 |
| 측정 원본 JSON 98개 (0.64 MiB) | `runs/perf/**/*.json` (traces 제외) | **본 커밋에서 git에 보존** |
| 대형 산출물 | `runs/` 내 .pt 체크포인트(100-220MB), 대형 trace | 미니DC 디스크에만 있음 (git 제외 — 정책) |
| 테스트 | `tests/agent/`, `tests/integration/` | git 추적 중 |
| 엔진 코퍼스/픽스처 | `engine/tests/`, `engine/data/` | git 추적 중 |
| 학습 준비 상태 | `engine/TRAINING_READINESS.md`, `docs/PA3_LEARNER_STATUS.md` | git 추적 중 |

## 7. 규율 요약

- 성능 수치는 clean HEAD + 3회 반복 + raw JSON 기준으로만 인용. 계약이 다른 수치끼리 배수 비교 금지.
- 코드 변경/머지 후에는 재측정 또는 "미재측정" 주석. 운영 오류 0, 자연완결 카운트 규율 유지.
- 시스템 변경(드라이버/파워/서비스) 금지 목록 유지. 브랜치 삭제는 별도 승인 패스로만.

## 8. 변경 이력

- 2026-10-09: 최초 작성 + main 승격 (아리아 작성, 미니DC 코덱스 세션 실행).
