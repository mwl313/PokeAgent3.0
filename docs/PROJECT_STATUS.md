# PokeAgent 3.0 — 프로젝트 현황판

> 갱신 규칙: 체크포인트(머지/게이트/측정)마다 갱신하고, 작성 시점 기준 SHA를 항상 기록한다.
> 최초 작성: 2026-10-09 (아리아). 본 문서는 미니DC 코덱스 세션이 커밋하고, 같은 작업에서 main으로 fast-forward 승격됐다.

## 0. 작업 구분 (중요)

- **맥미니 = 엔진 제작 라인**: Rust 네이티브 엔진 포팅(기술/특성/아이템/동적 클로저)과 fixture 검증까지.
- **미니DC = 최적화/측정 라인**: 그 이후의 처리량 최적화, 정확성 게이트, PPO 스모크, 벤치마크 기록 전부.
- 두 라인은 같은 리포(mwl313/PokeAgent3.0)의 브랜치로 존재하며, 현재는 아래 §1의 통합 정본 하나로 수렴돼 있다.

## 1. 진행 지도 — 브랜치와 문서 (2026-10-09 밤 갱신, 새 에이전트 필독)

### 1.1 브랜치 지도

| 항목 | 값 |
|---|---|
| **원격 정본 라인** | `origin/main` = `origin/optimization/pa3-realpolicy-throughput` @ `50c597b` (이번 작업 시작 시 원격 확인) |
| **미니DC v6 작업** | 로컬 `optimization/ppo-v6-throughput`, 코드 `3fbc749` — PPO 최적화·정확성 수정 완료, main 미병합/미푸시 |
| 포함 관계 | 엔진 줄 전체 흡수(engine-only 커밋 0) + 최적화 전체(P0부터 s5a W2까지) + 현황판/리포트/측정원본 |
| 엔진 라인(의도적 파킹) | `mac/long-horizon-engine-tail` @ `91c0a21` (10-09 19:26 싱크 머지) — GitHub에 있으며 재개는 이 브랜치에서 |
| 미통합 아카이브(보존, 삭제 금지) | `mac/*` 포크(hazards, hazard-setters, revival-blessing-v2 등) + `family/*`, `agent/*`, `integrate/*`, `training-pool/*`, `wip/*` |

### 1.2 문서 지도 (읽는 순서)

1. `docs/PROJECT_STATUS.md` — 이 문서 (살아있는 현황판)
2. `docs/research/TRAINING_APPROACH_v1_FINAL_20261009.md` — 훈련 방식 리서치 정본 (+ `research_ecosystem.md`, `research_sample_efficiency.md`, `draft_TRAINING_APPROACH_v0.md`)
3. `docs/spec/fullspec-1.1-minidc-20261006/` — 설계 스펙 (Full Spec 1.1)
4. `docs/perf/` — 최신: `V6_PPO_THROUGHPUT_REPORT.md` (동일 세션 3회 비교 및 검증), 이전: `V5_ROLLING_SLOT_ENGINE_FIX.md`, `V5_F16_WIRE_POC.md`
5. `docs/PokeAgent3_Optimization_Roadmap_2026-10-09.md` — 로드맵 (옵션 A-D, STEP 1-6)

## 2. 여정 요약 (여태 무엇을 했나)

### 2.1 엔진 라인 (맥) — 엔진까지

- 네이티브 Rust 배틀 엔진(PyO3 `pa3-engine`) 포팅: 기술/특성/아이템을 feature family 단위로 커밋, 각각 fixture 증인과 코퍼스 재생성 동반.
- `fa723ed`(10-07 머신 핸드오버): 374 moves / 141 abilities / 166 items / 703 fixtures.
- `54d0ff1`(10-08 롱호라이즌 테일): 486/515 moves(29 blocked), 200/223 abilities(23 blocked), 아이템 166/166, 동적 클로저 33 callers 중 11 blocked, 코퍼스 1,148 fixtures / 25,134 decision boundaries.
- 훈련 풀 1,137팀 전량 자연완결(100%), 운영 오류 0, `readiness_check` 10/16.
- 이후 휴면 후 10-09 저녁 파킹 커밋(`91c0a21`: Klutz/억제 패밀리 + 기록 + 싱크 머지)으로 의도적 정지 (§1.1 참조). 재개 시 잔여: blocked 이동 29, 특성 23, 동적 클로저 11, 코퍼스 전수화, readiness 6기준.

### 2.2 최적화 라인 (미니DC) — 그 이후

- **v1.1** (10-08): 실정책 롤아웃 최적화 Phase 0-6 — 수집(actor-only) 15.07 → 103.44 games/s(6.86x). 고정 관측 배치, packed 후보 wire, packed prefix walk, FP16 autocast, inference_mode, packed 롤아웃 행. 듀얼 액터-온리 162.73 games/s. all-in 20.87 → 26.43 games/s(단일 카드, 당시 계약).
- **v2.0** (10-09 새벽): P0 정확성 교정(지표 집계, exact row-weighted accumulation), M0 학습자 단계 분해, P1.3 스트리밍 물질화(peak RSS 18.0 → 7.34 GiB), P2.3 micro배치 스윕(256 → 1024: all-in 24.66 → 31.57, 3회 확인 32.80).
- **v3** (10-09 02:15): 양좌석(현재 정책 2석) 수집 계약으로 전환 — 19.37 games/s, 26.87 rows/경기, learner rows/s 460 → 520. PA3-8M 실그래디언트 파리티 게이트, DDP 전역 목표 수학·런처(DDP 집합연산 불일치로 end-to-end 블록 확인).
- **v4** (10-09 오전): C0 엔트로피/KL 정규화기 그래디언트 복원(학습 수학 교정), C1 단일 GPU oracle 확정, D0/D1/M1 DDP 고정-step 집합 프로토콜 + 수동 all-reduce(파리티 통과), D2 실행기 A/B(DDP 기본 유지, 통계 동급), F0 전스택 병목 아틀라스(업데이트 구간 GPU-busy 95.4% 등), F1 최종 패널 — **단일 20.53 / 듀얼 24.67(DDP), 25.60(manual) games/s**.
- **v5** (10-09 저녁): P0 듀얼 micro 1024 승격 — all-in **32.50 games/s**(+23.8%, update 42.2s/랭크). P0b columnar record+물질화=게이트 PASS 후 미승격(±2% 노이즈 내, 플래그 보존). P0c rolling slots(collector측)+H2D 트림=판정 기록. **s5a**: rolling-slot 엔진 수정(per-slot generation, opt-in 유지, A/B 동률) + f16 관측 wire no-go(실효 1.0-1.4%, 노이즈 이하) → **최종 32.50 games/s, s5a 종결**.
- 모든 수치는 `docs/perf/` 보고서와 `runs/perf/` 원본 JSON에 기록 (아래 §6).
- **v6 (미니DC 로컬)**: bounded CUDA rollout cache, 빈 후보 패딩 제거, 중복 모델 연산/동기화 감소, manual 초기화·공유 overflow·체크포인트 게이트 교정. 동일 세션 재시작 3회 중앙값 **28.84 → 37.23 games/s (+29.09%)**, PPO update **46.07 → 29.99초**. 4 epochs/PA3-8M/global minibatch4096 유지. 수정 중 코어의 지속 실행 보조 측정은 35.21 → 41.74 games/s이며 최종 HEAD 지속 실행 확인과 구분한다. 상세: `docs/perf/V6_PPO_THROUGHPUT_REPORT.md`.

## 3. 엔진 준비도 (readiness)

- `readiness_check`: 10/16 PASS — 실패 2, 3, 5, 8, 15, 16 (full-scope 커버리지/코퍼스 계열).
- 커버리지(10-08 기준, 최신은 `engine/ENGINE_SCOPE.md`): 이동 486/515, 특성 200/223, 아이템 166/166, 동적 클로저 33 callers 중 11 blocked.
- 팀풀 1,137팀: 자연완결 100%, 운영 오류 0.
- 100M 본학습: 미승인 상태가 정상 (조건 = readiness 16/16 + 사용자 승인).

## 4. 처리량 정본 (v4 F1 패널, 2,048경기, 양좌석 계약)

> ★ 최신 로컬 v6: 듀얼 **37.23 games/s** (clean `3fbc749`, 재시작 3회 중앙값, update 29.99s/느린 rank). 동일 세션 원본 50c597b는 28.84 games/s. 과거 v5 32.50과 아래 v4는 당시 스냅샷이며 이번 동일 세션 개선율의 분모가 아니다.

| 항목 | 단일 GPU | 듀얼 GPU |
|---|---:|---:|
| all-in games/s | 20.53 (19.44-20.96) | 24.67 (DDP) / 25.60 (manual) |
| 수집 | 22.1s (약 93 games/s) | 12.7s/랭크 |
| PPO 업데이트 | 77.1s | 58.5-61.5s/랭크 |
| 미세배치 | 1024 | 256 (기본값 잔존 — P0 교정 대상) |

- 측정 SHA: `fbf878e` (이후 커밋은 문서/프로비넌스만). 상세: `docs/perf/V4_OPERATIONAL_FRONTIER_REPORT.md`.
- 물리 상한은 미확정. 기존 110–115 games/s 추정과 GPU-busy만으로는 천장을 입증할 수 없다. kernel busy는 SM/Tensor Core 활용률과 다르다.
- 100M 일정은 단기 패널을 그대로 외삽하지 않는다. readiness 10/16과 장기 안정성·resume 검증이 남아 있다.

## 5. 다음 최적화 작업 (권장 순서)

| # | 작업 | 근거(실측) | 예상 효과(추정) | 검증 |
|---|---|---|---|---|
| 1 | ~~듀얼 micro 1024 A/B~~ **완료·승격 (v5 W1)** | 실측: 듀얼 256 median 26.25 → 1024 median 32.50 games/s(+23.8%, 3회 반복, 게이트 전부 PASS), update 57.5→42.2s/랭크 | 런처 기본값 1024로 승격 + 확인 런 32.50 | `docs/perf/V5_MICROBATCH_AB.md` |
| 2 | 수집 record columnar화 — **측정 완료·미승격 (v5b T1)** | row-SHA 동일 게이트 PASS, 실측 collect +2.0%(역행)·all-in +1.6%(노이즈 내) | 기본 경로 유지, `--columnar-store` 플래그로 보존 | `docs/perf/V5_COLUMNAR_RECORD.md` |
| 3 | 학습자 물질화 columnar + 후보 u8 — **부분 적용·미승격 (v5b T2)** | 물질화 −8.2%·update −2.0%, all-in +1.6%(노이즈 내); u8은 미착수(별도 결정) | 기본 경로 유지 | `docs/perf/V5_COLUMNAR_MATERIALIZATION.md` |
| 4 | pinned + async H2D (수집/학습) — **미실행 (v5c T1)** | H2D 약 1.7s/랭크(63s 중 2.7%) → 기대 이득이 ±2% 노이즈 내 | 다음 작업으로 이월 | profiler 타임라인 overlap 증거 |
| 5 | optimizer 블록 정리 — **v6 적용** | clip norm 유한성 재사용, rank 공유 overflow/scaler 교정 | update·all-in 개선에 포함 | global finite/skip 및 Adam 상태 parity PASS |
| 6 | digest/체크포인트/로그 정리 — **v6 적용** | 실패 게이트 실제 중단, 검증 후 원자적 체크포인트, persistent 실행·단계별 계측 | 안전성 및 startup 분리 | rank digest·worker 옵션·실패 정리 테스트 PASS |
| 7 | rolling slots — **엔진 수정·게이트 PASS·A/B 동률 (v5d s5a)** | per-slot generation + `reset_slots_batch` 구현, 등가성 테스트 2종 PASS, idle slot 21.1%→10.4%; 듀얼 A/B all-in −0.00%(collect +0.67%, update −0.31%) | 옵트인 `--rolling-slots` 유지, 기본값 불변 | `docs/perf/V5_ROLLING_SLOT_ENGINE_FIX.md` |
| 제외 | torch.compile/Triton | Triton이 CC 8.0+만 지원, V100 fp16 tl.dot open bug | 해당 없음 | — |

- 계약 변경(별도 승인): 4 epochs→2, 관측 f16 wire(B1 — s5a에서 **비용 상한 분석 후 보류**: 실효 1.0–1.4%·낙관 상한 2.7%로 ±2% 노이즈 내, `docs/perf/V5_F16_WIRE_POC.md`), 모델 축소 — 천장을 움직이지만 학습 수학 변경이라 별도 A/B 필요.
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
- 2026-10-09 (밤): v5 체인 반영(s5a 종결, 최종 32.50 games/s), 진행 지도(브랜치/문서) 갱신, 훈련 리서치 v1 정본 `docs/research/` 커밋.
- 2026-10-09 (미니DC v6): 로컬 최적화 코드 3fbc749, clean 3회 패널 37.23 games/s, 현황/상한 추정 교정. main 변경 없음.
