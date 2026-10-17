# PA3 V2 AHL v2 — Stage A–C 최종 보고서

작성일: 2026-10-17 (KST)  
브랜치: `feature/ahl-v2-diagnostics`  
기준 커밋: `16cc0bf1db712cf23effe2614dc0d58ab463c19e`  
구현 커밋: `80c9237` (`feat(ahl-v2): add isolated historical diagnostics and shadow planning`)  
워크트리: `/home/minidc/Desktop/PokeAgent3.0-AHL-v2`

## 1. 한 줄 결론

Stage A는 구현 후 **5,760게임(45/435 pair) bounded smoke**를 완료했고,
Stage B는 코드·테스트만 구현했으며 milestone이 없어 실행하지 않았다.
Stage C는 **SHADOW_ONLY** opponent plan을 생성했고 활성화하지 않았다.
Stage D는 **NO_GO**다. 훈련 코드, PPO, STOP, alpha, champion, checkpoint는
건드리지 않았다.

## 2. 구현 vs 설계

| 구분 | 내용 |
|---|---|
| 구현 | Stage A RR 코드, bounded smoke, harness equivalence, resumable status/cache |
| 구현 | Stage B milestone hook + 8-anchor planner + 테스트 (실행은 안 함) |
| 구현 | Stage C reference selector 이식, digest-pinned shadow plan, gate report |
| 구현 | SOT dated addendum 적용, pytest log |
| 설계만 | Stage D migration contract, collector weighting, A/B protocol, rollback plan |
| 미수행 | training code 수정, PPO resume, migration receipt 생성, SHADOW activation |

## 3. Stage A — Round Robin diagnostics

### 3.1 Harness equivalence (Gate 1)

- 목표 formal receipt: `235eda3ee09c66db6a6801e6f305fc3e40146460beffb08881a36f8f7f35e256`
- 비교 조건: `v2-000000503808-da35700f` vs `v2-000000200704-da35700f`,
  64 fixture pairs (128 games), seed `20261014`
- 결과: **PASS**
  - identity 일치
  - environment 일치
  - fixture manifest 일치 (`7914d698...e5106`)
  - W/D/L 일치 (69/0/59)
  - score 일치 (0.5390625)
- calibration 게임: 128

### 3.2 Registry snapshot

- 원본 formal registry SHA256: `3a5c6b938d4d0539fc8292fa1e55a8a39fb6be9d82cc9f67751707f513991437`
- 진단 snapshot SHA256: `24da9771e2f78e176cca8e04fe81b98cfebd4085b00cbdbc0af98f370177aa66`
- 정책 수: 30 (dedup 전 30, dedup 후 30, 중복 digest 0)
- Champion: `v2-000002600960-da35700f`, digest `4fafd07d...b434f`
- Latest: `v2-000002764800-da35700f`, digest `2540efce...c40a5`
- Pair arithmetic: `C(30,2)=435`, fine 56, broad 379, full grid `77,184` games

### 3.3 Bounded smoke 실행

- 선택: 45 pair / 5,760 games (전체 77,184의 약 7.46%)
- 구성:
  - champion↔latest: 1 pair
  - champion row: 29 opponents 전체 (29 pair)
  - adjacent-clock neighbours: 16 pair
- 각 pair: 64 fixture pairs = 128 games (bounded smoke를 위해 fine pair도 128게임으로 제한)
- 결과: 45/45 `success`, operational errors 0, illegal actions 0
- elapsed: **1,649.817초 (27분 29.8초)**
- equivalence 포함 총 실행 게임: **5,888**
- 전체 435-pair grid는 `pair_plan.json`, `status.json`,
  content-addressed receipts로 **resumable** 상태로 남김

### 3.4 Diagnostic 결과

- Champion row: 29 opponents, 3,712 games, game-weighted champion score
  **0.667565**
- Champion 기준 latest `2,764,800`의 점수: **0.429688**
  (latest 기준으로는 0.5703125, 128 games, RR seed)
  - formal 170-game light diagnostic의 0.4824와 fixture/seed가 다르므로
    직접 비교하거나 promotion 근거로 사용하면 안 됨
- Cycles: **0**
- BT rating 1위 champion, 2위 latest (descriptive only)
- Alpha 사용: **0** (formal alpha ledger 무변경)

## 4. Stage B — Incremental diagnostics

- `milestone_hook` + 8-anchor planner 구현 및 테스트 완료
- 현재 latest clock `2,764,800`, next nominal full boundary `2,800,000`
- `available=false`, `reason=no_new_full_milestone`, `executed=false`
- Trigger 시 계획: 최대 8 anchors × 256 games = 2,048 games
- 역할: former champions 3, verified weakness 2, informative nonchampions 2,
  distant anchor 1 (중복 digest 제거)
- Training이 paused이므로 실제 실행하지 않음

## 5. Stage C — Shadow selector

- Reference: champion `v2-000002600960-da35700f`
- Reference digest: `4fafd07d6bba71adad5460f6ec8583f76d776882d45a6ea7e70ba7af080b434f`
- Plan: `/home/minidc/eval-league/roundrobin/v1/opponent_plan.json`
- `shadow_only=true`
- `history_rows_target=0`
- `history_probability=0.0`
- `status=SHADOW_ONLY_NOT_ACTIVE`
- `activation=FORBIDDEN_WITHOUT_OWNER_APPROVAL_AND_REVIEWED_MIGRATION`
- 8 opponents, weights sum **1.0**, cap **0.40** 준수 (최대 0.185065)
- Stale/unknown opponent는 exploration mass만 받음
  - `v2-000000057344-bootstrap`: stale, weight 0.0125
  - `v2-000002002944-da35700f`: stale, weight 0.0125
- Plan hash: `dfe1f3a727d12c9cb08d1e9b1af88dd1f17b01e61e3b814e51dac6637112af60`
- Pool hash: `ce9c4c0e271051c3175ab96a6e883b366a0fdadc297ca957f2a4caf8c09f3349`
- Selector SHA256: `d61bea05a25bd789b584b8e4a072c387a41d5d1a29f9166f6530e386969e2e28`
- Active training은 전혀 변경하지 않음

## 6. Stage D — 설계만, NO_GO

작성 문서:

- `docs/operations/AHL_V2_STAGE_D_MIGRATION_CONTRACT.md`
- `docs/operations/AHL_V2_STAGE_D_COLLECTOR_WEIGHTING.md`
- `docs/operations/AHL_V2_STAGE_D_AB_PROTOCOL.md`
- `docs/operations/AHL_V2_STAGE_D_ROLLBACK.md`

NO_GO 이유:

1. Owner approval 없음
2. Reviewed V2 source/profile/opponent migration contract 미승인
3. Old opponent-manifest serialization/compare 불일치 검증·수정 필요
4. 동일 checkpoint에서의 isolated A/B pilot 미실시
5. Rollback/restore 검증 및 mode-specific rows/s baseline 승인 필요

금지 사항 준수: training code 수정 없음, PPO resume 없음, migration receipt
생성 없음, systemctl start/stop/restart 없음, GPU 사용 없음.

## 7. 보호 파일 무변경 증명

보호 snapshot:

- `protected_snapshot/key_hashes_before.txt`
- `protected_snapshot/key_hashes_after.txt`
- `protected_snapshot/training_status_after.txt` (빈 파일 = clean)
- `protected_snapshot/eval_status_after.txt` (빈 파일 = clean)

| 대상 | before | after | 판정 |
|---|---|---|---|
| training HEAD | `da35700f...f8b7` | 동일 | 무변경 |
| training worktree status | clean | clean | 무변경 |
| eval worktree HEAD | `16cc0bf1...c19e` | 동일 | 무변경 |
| eval worktree status | clean | clean | 무변경 |
| STOP | `e3b0c442...b855` (empty) | 동일 | 무변경 |
| production profile | `c816aa28...7941` | 동일 | 무변경 |
| 2,764,800 generation manifest | `3cc75293...88e7` | 동일 | 무변경 |
| `alpha_budget.json` | `6fdf62a3...ef14f` | 동일 | 무변경 |
| `best_policy.json` | `bc7c61fb...8779` | 동일 | 무변경 |
| `registry.json` | `3a5c6b93...1437` | 동일 | 무변경 |

`schedule.json`은 **formal discovery timer**(`v2-eval-discover.service`)가
주기적으로 다시 쓰기 때문에 해시가 바뀌었다. 이는 RR 작업 이전인 15:12:24에도
발생했고, RR 코드는 이 파일을 읽기만 한다. 현재 job은 29개 모두 `completed`,
pending/running은 0이다. Gate report는 이 파일을 `formal_discovery_owner`로
분리 기록하며, critical protected file mismatch는 **0**이다.

Alpha ledger 참고:

- Aria 브리핑: `0.002857/0.05`
- 실제 `alpha_budget.json`의 `consumed`: `0.002368351574500769/0.05`
- 전후 SHA256 동일. RR은 alpha를 전혀 쓰지 않았다.

Disk:

- before: 77,534,470,144 bytes (약 72.2 GiB free)
- after: 77,353,472,000 bytes (약 72.0 GiB free)
- `df`: 73 GiB free

## 8. 테스트

- `pytest.log`: **70 passed in 21.54s**
- 대상:
  - `tests/agent/test_v2_historical_league.py` (신규 14 tests)
  - `tests/agent/test_v2_source_of_truth.py`
  - `tests/agent/test_v2_evaluation_league.py`
  - `tests/agent/test_v2_evaluation_league_v2.py`
  - `tests/agent/test_v2_eval_resource_gate.py`
  - `tests/agent/test_evaluation_league.py`
- 신규 테스트 항목: N30=435/fine=56/full=77,184, champion endpoint fine=57,
  plan priority/bounded selection, registry dedup, pair-cluster bootstrap,
  partial-panel invalidation, BT/cycle, shadow-only/digest pinning/weight cap,
  stale evidence, corrupted pair cache, milestone hook, selector formula,
  SOT addendum

## 9. 산출물

Code:

- `agent/evaluation/historical_league.py`
- `agent/evaluation/historical_selector.py`
- `scripts/v2_history_roundrobin.py`
- `tests/agent/test_v2_historical_league.py`

Diagnostic root `/home/minidc/eval-league/roundrobin/v1/`:

- `status.json`
- `receipts/` (45 pair summaries + battle receipts + equivalence receipts)
- `matrix.csv`
- `ratings.csv`
- `cycles.md`
- `champion_validation.md`
- `opponent_plan.json`
- `incremental_plan.json`
- `gate_report.json`
- `equivalence.json`
- `pair_plan.json`
- `registry.snapshot.json` + `registry.meta.json`
- `protected_snapshot/`

Docs:

- `docs/operations/AHL_V2_ROUND_ROBIN.md`
- Stage D 4 documents
- `docs/TRAINING_SOURCE_OF_TRUTH.md` dated addendum (blob prefix match)

## 10. 최종 판정

- Stage A: **SMOKE COMPLETE / resumable**
- Stage B: **IMPLEMENTED, NOT TRIGGERED**
- Stage C: **SHADOW PLAN READY, NOT ACTIVE**
- Stage D: **NO_GO — owner approval과 reviewed migration/A-B/rollback 증거 필요**

역사적 PPO rollout은 실제로 실행되지 않았고, SHADOW는 활성화되지 않았으며,
훈련 코드 경로는 수정되지 않았다.
