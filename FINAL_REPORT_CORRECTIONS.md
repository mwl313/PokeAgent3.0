# PA3 V2 AHL — 라운드로빈 교정 및 반복 진단 최종 보고서

작성일: 2026-10-17 (KST)  
브랜치: `fix/ahl-v2-corrections` (기준 `1d90d82`)  
구현 커밋: `473092b` (`fix(ahl-v2): correct Bradley-Terry fit and add recurring diagnostics`)  
문서/보고서 커밋: `c53d39f` (`docs(ahl-v2): add Korean correction and operational readiness report`)  
기준 커밋: `1d90d82`  
워크트리: `/home/minidc/Desktop/PokeAgent3.0-AHL-fix`

## 0. 한 줄 결론

Bradley–Terry MM 업데이트의 수학적 오류를 고쳐 435-pair 전체 결과를
재계산했고, 2.8M/3.8M 기준의 **내구성 있는 반복 진단 스케줄러**와
**토폴로지 기반 자원 게이트**를 추가했으며, smoke-era `opponent_plan.json`을
**OBSOLETE/비활성**으로 은퇴시켰다. 실제 PPO·보상·모델·하이퍼파라미터·
체크포인트는 전혀 변경하지 않았고, 라이브 PPO와의 공존 성능은
**UNVERIFIED**다.

## 1. 확인된 결함과 수정 내용

| # | 결함 | 확인 | 수정 |
|---|---|---|---|
| 1 | BT MM 업데이트가 `numerator/denominator`로 previous strength를 누락 | 재현: 기존 상위 1·2위가 2,600,960 / 2,764,800, corrected는 2,764,800 1위·2,600,960 4위 | `strength * numerator / denominator`, geometric-mean 정규화, log-파라미터·quasi-likelihood 수렴 검사, pair-cluster bootstrap |
| 2 | milestone이 overshoot를 만족하지 못함 | `next_full_boundary(2,802,944)`가 2,800,000을 반환하지 못하는 구조 | `latest_full_boundary()` floor semantics + checkpoint를 실제로 도달한 verified snapshot에 매핑. 2,802,944 → 2,800,000 만족 |
| 3 | 발행 metadata가 45-pair smoke 상태 | `gate_report.json`의 `games_run=5,760`, `status.json`의 `pairs_completed=45` | `FULL_COMPLETE.md`/`run_report.json` 생성, gate/status/README를 435 pairs / 77,184 games로 재조정. smoke는 역사 증거로 보존 |
| 4 | shadow plan이 45 summaries·2,600,960 기준 | `evidence.pair_summaries=45`, reference clock 2,600,960 | 원본 bytes를 `opponent_plan.smoke-20261017.json`으로 보존, `opponent_plan.json`은 `OBSOLETE`/`activatable=false`/`FORBIDDEN`, training loader는 항상 거부 |
| 5 | CPU affinity 가정이 검증되지 않음 | `run_ddp_ppo.py`의 `RANKS[].cpus`는 미사용, 실제는 `numactl --cpunodebind/--membind`; 기존 `rr_pool.sh`는 0–31 사용으로 formal 4–7과 충돌 | sysfs 토폴로지 기반 마스크 도출, formal 4–7 + SMT 44–47 예약, 충돌 시 fail-closed. PPO launcher는 변경하지 않음 |

## 2. IMPLEMENTED (구현 완료)

- **BT 교정**: `agent/evaluation/historical_league.py`
  - 올바른 MM update, 반복 누적(중복 row), draws=0.5 win, unequal game counts.
  - 수렴: 62 iterations, log-likelihood `-46,376.91`, max log-delta `7.3e-11`.
  - pair-cluster bootstrap 500 draws, rating/rank CI, adjacent-rank probability,
    `difference_supported`.
  - `ratings.csv`, `ratings.json`, `ranking_confidence.csv` 재생성.
- **내구성 스케줄러**: `agent/evaluation/historical_schedule.py`
  - Tier A: 2.8M부터 200K마다, 최대 8 anchors × 256 games.
  - Tier B: 3.8M부터 1M마다, 최대 24 distinct model digests.
  - overshoot-aware, `tier:clock` idempotent, receipt 기반 resume,
    corrupt receipt fail-closed + operator 제거 후 resume,
    다중 missed milestone, Tier B가 동일 clock Tier A를 supersede.
  - formal pending/running이면 `deferred_formal_priority`로 신규 작업 중단.
  - 참가자: digest dedup, latest+champion 항상 포함, stable anchors,
    오래된 snapshot일수록 넓은 temporal spacing.
  - 안전한 pair cache identity (digest/fixture/environment/panel 일치 시만 재사용).
- **자원/토폴로지**: `agent/evaluation/historical_resources.py`,
  `scripts/ahl_v2_resource_gate.py`
  - Mode A(PPO 실행): 2 workers × 2 threads, 3–5% 악화 시 1×1, >5% pause.
  - Mode B(PPO 정지): 최대 16 workers × 2 threads.
  - formal 4–7 + SMT 44–47 예약, mask 충돌 시 0 workers.
  - disk 12 GiB, available memory 8 GiB, swap 2 GiB 가드.
  - `coexistence_verified=false` 고정.
- **rr_pool 교정**: `scripts/rr_pool.sh`가 하드코딩 0–31 대신 resource gate의
  마스크를 사용. PPO 실행 감지 시 Mode A로 전환.
- **보고/은퇴**:
  - `FULL_COMPLETE.md`, `run_report.json` (validation ok=true, complete=true).
  - `opponent_plan.json` OBSOLETE wrapper + 원본 archive + `load_opponent_plan` guard.
  - `incremental_plan.json`을 SUPERSEDED 포인터로 교체(원본 archive 보존).
  - `gate_report.json` Stage A=`FULL_COMPLETE`, Stage B/C 재조정, Stage D=`NO_GO`.
  - `status.json` `completed`, 435/435, 77,184 games.
- **문서**: `docs/operations/AHL_V2_ROUND_ROBIN.md`,
  `AHL_V2_DIAGNOSTIC_SCHEDULE.md`, `AHL_V2_CPU_TOPOLOGY.md`,
  `docs/operations/ahl-v2/results/README.md`, `FINAL_REPORT.md` 상단 superseded 배너,
  `docs/TRAINING_SOURCE_OF_TRUTH.md` 최상단 owner decision.

## 3. TESTED (테스트 완료)

- 신규/수정 테스트: `tests/agent/test_v2_historical_corrections.py` (19 tests),
  `tests/agent/test_v2_historical_league.py` (14 tests).
- 최종 targeted suite 결과 (`pytest.log`):
  **77 passed, 1 skipped in 14.66s**.
- 포함: BT fixed-point/독립 score equation 검증, deprecated 구현과의 차이,
  bootstrap 결정성, exact/overshoot/missed milestone, idempotency, restart,
  duplicate discovery, corrupt receipt + recovery, formal priority,
  Tier B supersede/cap/dedup, stable anchors, pair-cache identity,
  topology mask/formal CPU/SMT 회피, mask conflict fail-closed,
  throughput 3%/5% 임계값, Mode A/B, shadow retirement/provenance,
  run validation, 게시된 435-pair 완전성, active config history=0.0,
  protected verification.
- 환경 제약: `tests/agent/test_evaluation_league.py` 6건은 이 워크트리에
  native `pa3_engine.so`가 없어 실패한다(코드 변경과 무관한 기존 환경 문제).
  로그: `pytest_pre_existing_native.log` (5 passed, 6 failed, ImportError).

## 4. DEPLOYED (배포 상태)

- 코드/보고서는 **로컬 브랜치 `fix/ahl-v2-corrections`에 커밋**됨.
  Aria 검증 후 push 예정이며, 본 작업에서 force-push/merge/main 변경은 없음.
- 라이브 진단 루트 `/home/minidc/eval-league/roundrobin/v1`의 **파생 보고서만**
  재생성됨: ratings/matrix/ranking confidence/run_report/FULL_COMPLETE/status/
  gate/schedule/resource_gate/shadow retirement. **pair receipts는 재작성하지 않음.**
- systemd 유닛 설치/시작/중지/재시작 없음. formal eval/training 서비스 무변경.
- 스케줄러는 기본 **plan-only**이며, 배틀을 자동 실행하지 않는다.

## 5. VERIFIED UNDER LIVE PPO

**없음 — UNVERIFIED UNDER LIVE PPO.**

- 현재 production PPO는 2,764,800 / iter 675에서 STOP 상태다.
- 따라서 2 workers × 2 threads의 실제 PPO throughput 영향, 3%/5% 임계값의
  타당성, `numactl --cpunodebind`와의 실제 경합은 **측정되지 않았다**.
- 단위 테스트와 토폴로지 검사는 통과했지만, 이는 실측 공존 증거가 아니다.
- 기존 baseline `381.9 learner rows/sec`는 덮어쓰지 않았다.

## 6. 교정된 상위 5개 rating (descriptive only)

| 순위 | policy | clock | rating | pair-cluster 95% CI | P(above next) |
|---:|---|---:|---:|---|---:|
| 1 | `v2-000002764800-da35700f` | 2,764,800 | +0.8801 | [+0.7489, +0.9798] | 0.670 |
| 2 | `v2-000002502656-da35700f` | 2,502,656 | +0.8428 | [+0.7409, +0.9383] | 0.694 |
| 3 | `v2-000002400256-da35700f` | 2,400,256 | +0.8110 | [+0.7390, +0.9112] | 0.608 |
| 4 | `v2-000002600960-da35700f` (champion) | 2,600,960 | +0.8041 | [+0.7075, +0.8911] | 0.716 |
| 5 | `v2-000002703360-da35700f` | 2,703,360 | +0.7662 | [+0.6937, +0.8460] | 0.520 |

- 상위 5개 인접 순위 차이는 모두 `difference_supported=false`다. 점추정 순위는
  설명용이며 통계적으로 확정된 순서가 아니다.
- champion vs latest: latest 0.5254, paired 95% CI [0.4863, 0.5664] (512 games)
  → 통계적으로 구분 불가.
- cycles: 0. formal promotion authority/alpha는 무변경.

## 7. 보호 파일 무변경 증명

`docs/operations/ahl-v2/results/protected_snapshot/corrections_verification.json`:

- `unchanged=true`, `critical_mismatches=[]`
- training HEAD `da35700f75c95833b0e8250d7b4b04e11320f8b7`,
  worktree porcelain empty, `STOP` present.
- evaluation HEAD `16cc0bf1db712cf23effe2614dc0d58ab463c19e`,
  worktree porcelain empty.
- `alpha_budget.json` SHA256 `6fdf62a3…ef14f` (unchanged)
- `best_policy.json` `bc7c61fb…8779` (unchanged)
- `registry.json` `3a5c6b93…1437` (unchanged)
- 2,764,800 generation manifest `3cc75293…88e7` (unchanged)
- production profile `c816aa28…7941` (unchanged)
- formal schedule jobs 29, all completed; `schedule.json` 해시 변화는
  formal `v2-eval-discover` timer 소유이며 historical 코드는 쓰지 않음.

## 8. PENDING OWNER ACTION

1. **브랜치 push/리뷰**: Aria가 `fix/ahl-v2-corrections`를 검증 후 push.
   (`473092b` 및 후속 문서 커밋)
2. **2.8M milestone 도달 시 실행 승인**:
   training이 2,800,000 이상으로 재개된 뒤 아래 순서로 plan-only 실행.
   ```bash
   cd /home/minidc/Desktop/PokeAgent3.0-AHL-fix
   PY=/home/minidc/Desktop/PokeAgent3.0/.venv/bin/python
   RR=/home/minidc/eval-league/roundrobin/v1
   "$PY" scripts/v2_history_roundrobin.py --root "$RR" freeze
   "$PY" scripts/v2_history_roundrobin.py --root "$RR" schedule
   "$PY" scripts/ahl_v2_resource_gate.py --root "$RR" --ppo-running
   ```
   formal queue가 비어 있고 resource gate가 허용할 때만 실제 worker 실행.
3. **라이브 PPO 공존 실측**: PPO 재개 후 최소 3개 완료 iteration의
   learner rows/sec로 3%/5% 게이트를 검증하고 baseline 381.9는 보존.
4. **선택적 launcher migration 검토**: 완전한 physical-core isolation이 필요하면
   `run_ddp_ppo.py`의 `RANKS[].cpus`를 실제 cpuset으로 소비하는 별도 reviewed
   변경 필요. 본 패치는 production launcher를 건드리지 않았다.
5. **Historical PPO 재검토**: 검증된 역사적 regression + owner 승인 전에는
   절대 재활성화하지 않는다. Stage D는 `NO_GO`.

## 9. 다음 단계 명령 (검증/운영)

```bash
# 커밋/테스트 확인
cd /home/minidc/Desktop/PokeAgent3.0-AHL-fix
git log --oneline -3
git show --stat 473092b
/home/minidc/Desktop/PokeAgent3.0/.venv/bin/python -m pytest \
  tests/agent/test_v2_historical_league.py \
  tests/agent/test_v2_historical_corrections.py -q

# 파생 결과 재생성 (배틀 재실행 없음)
/home/minidc/Desktop/PokeAgent3.0/.venv/bin/python \
  scripts/v2_history_roundrobin.py --root /home/minidc/eval-league/roundrobin/v1 report

# 현재 상태/게이트 확인
/home/minidc/Desktop/PokeAgent3.0/.venv/bin/python \
  scripts/v2_history_roundrobin.py --root /home/minidc/eval-league/roundrobin/v1 gate
/home/minidc/Desktop/PokeAgent3.0/.venv/bin/python \
  scripts/ahl_v2_resource_gate.py --root /home/minidc/eval-league/roundrobin/v1 --ppo-stopped
```

## 10. 남은 blocker

- 라이브 PPO 공존 성능/임계값: **UNVERIFIED** (training stopped).
- PPO launcher의 full physical-core isolation: 현재 구조로는 불가능하며 별도
  reviewed migration 필요.
- 2.8M/3.8M 실제 진단 실행: owner 승인 및 formal queue 상태에 의존.
- native `pa3_engine.so`가 없는 워크트리에서는 레거시 eval 테스트 6건이
  환경 문제로 실패(코드 회귀 아님).
- 브랜치 push/merge는 Aria/owner 검증 대기.
