# FINAL REPORT — Random / Deterministic Baseline Evaluation v1

- 작성일: 2026-10-17 (KST)
- 평가 대상: `v2-000002764800-da35700f` (2,764,800 matches, iteration 675)
- 후보 소스 SHA: `da35700f75c95833b0e8250d7b4b04e11320f8b7`
- 고정 세대: `iter-000002764800-1792212738467610757`
- 평가 작업 트리: `/home/minidc/Desktop/PokeAgent3.0-RandomBaseline` (branch `eval/random-baseline-v1`)
- 산출물 런 루트: `/home/minidc/eval-league/random-baseline/v1`
- 공개 산출물: `docs/operations/random-baseline-v1/`

---

## 0. 한 줄 결론

**READY_FOR_SHOWDOWN_SMOKE_TEST** — 오프라인 정합성 하드 게이트(불법 행동 0, 운영 오류 0, 누락/중복 0)를 모두 통과했고, branch-wise uniform legal random 상대로 **8,099승 / 93패 / 0무 (score 0.9886, paired 95% CI 0.9863–0.9910)** 를 기록했다. 다만 이는 Showdown ladder 강함의 증거가 아니며, Showdown adapter 자체는 아직 검증되지 않았다. Smoke test를 진행할 근거는 충분하다.

---

## 1. IMPLEMENTED (구현)

### 1.1 재사용 원칙

- 새 배틀 엔진을 만들지 않았다.
- 기존 `agent/evaluation/batched_policy.py::BatchedPolicyEvaluator.run(...)` 를 그대로 사용했다.
- 기존 `agent/evaluation/fixed_policy.py::load_frozen_model` 로 고정 checkpoint를 CPU에 로드했다.
- 외부 read-only 엔진 빌드를 사용했다:
  `PYTHONPATH=/home/minidc/Desktop/PokeAgent3.0-V2-evaluation-league/engine/python`
  - `pa3_engine.so` SHA-256: `8f3643a9f7b2febfa567f039b911d7af2f88bbf7bfc912849ada9e6a690d1c62`
  - 재빌드/복사/수정하지 않았다.

### 1.2 추가한 thin runner

- `scripts/random_baseline_eval.py`
  - `prepare`: 1,220-team pool에서 fixture를 **한 번만** 생성하고 `fixtures.json` + `fixtures.manifest.json` 으로 고정.
  - `worker`: shard 하나를 처리하고 atomic/resumable receipt를 기록.
  - `calibrate`: 8/12/16/18/20 worker 및 thread/batch 변형을 독립 calibration fixture로 측정.
  - `suite`: 특정 test를 N개 shard로 실행 (`--fresh` 로 해당 suite receipt만 재생성 가능).
  - `traces`: Test A 결과에서 10승 / 최대 10패 / 10개의 unusually long game을 keyed replay로 재현하고 decision trace를 기록.
  - `aggregate`: exact coverage 검증, W/D/L, paired bootstrap CI, seat/team/duration 집계.
  - `all`: calibrate → Test A → Test B → controls → traces → aggregate → docs publish.

### 1.3 Evaluator 계측 변경 (수학적 동작 불변)

`agent/evaluation/batched_policy.py` 에 outcome별 `rounds`/`decisions` 필드만 추가했다.

- sampling, masking, model forward, action 선택 로직은 변경하지 않았다.
- `rounds`는 배틀 round당 1회만 증가하도록 구현했다.
- 재실행 후에도 W/D/L과 fixture coverage가 동일함을 확인했다.

### 1.4 Fixture 설계

- Seed domain: `pa3-random-baseline-v1`
- Test A: 4,096 seat-swapped pairs = 8,192 games
- Test B: 1,024 pairs = 2,048 games
- Control RR: 256 pairs = 512 games
- Control LR: 256 pairs = 512 games
- Calibration: 256 pairs = 512 games (독립 seed domain)
- Stable global fixture ID, pair ID, seat(`policy_side`)를 부여.
- Suite 내 `(team_a, team_b, seed)` identity 중복 0.
- Shard는 **pair 단위**로 분할해 seat-swapped pair가 항상 같은 shard에 남도록 했다.
- Fixture manifest hash: `5fe97fe15be9874f2461ac89516957ece515761d9cf770732fb70333eaacd84e`
  - Test A fixture SHA-256: `39a63f647fddfc01fdccea3ce5b78c2f36fbf07eb6f2eb822f1f2f6f9e9fca9c`
  - Test B fixture SHA-256: `bf0f1552fbd1cb5020709aca130c1437af17c638f393127de599cb25d6e1365c`

### 1.5 병렬 실행

- OS process 1개 = shard 1개.
- `taskset` 로 서로 다른 physical core에 고정.
- CPU 0–19 = NUMA node 0, CPU 20–39 = NUMA node 1의 첫 SMT thread.
- 20 worker 설정은 10/10으로 양 NUMA node에 균형 배치.
- `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.
- `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`.
- `torch.inference_mode()` 사용.
- `BatchedPolicyEvaluator(device="cpu", batch_size=32)`.
- Atomic receipt: temp file + `fsync` + `os.replace`.
- Resumable: receipt에 이미 완료된 fixture ID는 재실행하지 않음.
- Formal evaluation job이 active가 되면 graceful stop file을 만들고 새 chunk를 발행하지 않도록 했다. 이번 실행에서는 formal 우선순위 개입이 발생하지 않았다.

---

## 2. TESTED (검증)

### 2.1 사전 smoke test

- 2 pairs(4 games), CPU, `checkpoint vs random`:
  - 4 games, 0 operational errors, 0 illegal actions.
  - Engine team pool 1,220 확인.
  - Model precision mode `E` 확인.
  - `pa3_engine.so` 로드 및 packed candidate walk 정상.

### 2.2 Calibration

독립 512-game calibration set으로 7개 설정을 측정했다.

| config | workers | engine workers | batch | steady games/s | wall games/s | CPU avg % | RAM avg GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| w08 | 8 | 1 | 32 | 29.12 | 25.41 | 10.3 | 9.14 |
| w12 | 12 | 1 | 32 | 40.80 | 25.41 | 11.4 | 9.34 |
| w16 | 16 | 1 | 32 | 49.15 | 33.83 | 16.8 | 10.77 |
| w16-b64 | 16 | 1 | 64 | 49.59 | 33.83 | 16.8 | 10.81 |
| w16-ew2 | 16 | 2 | 32 | 49.52 | 33.82 | 16.8 | 10.81 |
| w18 | 18 | 1 | 32 | 56.17 | 33.81 | 17.9 | 10.91 |
| **w20** | **20** | **1** | **32** | **57.51** | **33.80** | **18.8** | **11.05** |

선정: **w20** (zero errors / zero illegal, highest steady-state valid games/s).  
Full Test A에서도 w20이 **65.35 games/s steady, 62.47 games/s wall-inclusive** 를 기록해 calibration 선택이 유효함을 확인했다.

### 2.3 정확성 하드 게이트

| 항목 | 결과 |
|---|---|
| Illegal actions | **0** |
| Operational errors | **0** |
| Silently dropped games | **0** |
| Fixture coverage 누락 | **0** |
| Fixture coverage 중복 | **0** |
| Broken seat-swapped pairs | **0** |
| Trace replay 일치 | **30 / 30** |
| Checkpoint mutation | 없음 (read-only load) |
| Formal state write | 없음 |

### 2.4 Test별 coverage

- Test A: expected 8,192 / observed 8,192 / valid 8,192.
- Test B: expected 2,048 / observed 2,048 / valid 2,048.
- Control RR: 512 / 512.
- Control LR: 512 / 512.

---

## 3. MEASURED (측정 결과)

### 3.1 메인 결과

| Test | 상대 | Games | W | L | D | Score | Paired 95% CI | Errors | Illegal | Hard gate |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| **Test A** | branch-wise uniform legal random | 8,192 | **8,099** | **93** | **0** | **0.988647** | **0.986328 – 0.990967** | 0 | 0 | PASS |
| **Test B** | lexicographic legal | 2,048 | **2,046** | **2** | **0** | **0.999023** | **0.997559 – 1.000000** | 0 | 0 | PASS |
| Control RR | random vs random | 512 | 243 | 269 | 0 | 0.474609 | 0.435547 – 0.513672 | 0 | 0 | PASS |
| Control LR | lexicographic vs random | 512 | 49 | 463 | 0 | 0.095703 | 0.072266 – 0.121094 | 0 | 0 | PASS |

- Score = `(wins + 0.5 × draws) / valid games`.
- CI는 **fixture pair 단위 paired bootstrap 10,000회** (seat game 단위 아님).
- Test A/B 모두 draw 0.

### 3.2 Seat별 결과

| Test | Seat | Games | W | L | D | Score | Wilson 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| Test A | 0 | 4,096 | 4,048 | 48 | 0 | 0.988281 | 0.984498 – 0.991150 |
| Test A | 1 | 4,096 | 4,051 | 45 | 0 | 0.989014 | 0.985332 – 0.991779 |
| Test B | 0 | 1,024 | 1,023 | 1 | 0 | 0.999023 | 0.994489 – 0.999828 |
| Test B | 1 | 1,024 | 1,023 | 1 | 0 | 0.999023 | 0.994489 – 0.999828 |

Seat 간 차이는 0.0007 (Test A)로 사실상 없음. Seat bias 증거 없음.

### 3.3 Team 일반화

- Test A에서 **1,219 / 1,220** 팀이 policy-controlled team으로 최소 1회 등장.
- Test A policy team 중 10게임 이상인 팀의 최저 score:
  - team 1163: 8승 2패 (0.800)
  - team 187: 9승 2패 (0.818)
  - 그 외 10게임 이상 팀은 모두 ≥ 0.900
- Test A 패배 93건은 특정 팀에 집중되지 않았다 (최다 2패).
- 상대 팀 기준으로도 패배를 유발한 팀이 1–2패 수준으로 분산됨.
- Test B 패배 2건: team 932, team 44 (각 1패).
- 결론: 1,220-team pool 내에서 특정 팀만 과도하게 무너지는 패턴은 관찰되지 않았다. 단, 이 pool 자체가 training-team pool이므로 외부 team-composition 일반화 주장은 할 수 없다.

### 3.4 Battle duration

| Suite | Median rounds | P90 | P99 | Max | 21–40 rounds | 41–80 rounds |
|---|---:|---:|---:|---:|---:|---:|
| Test A | 9 | 13 | 19 | 60 | 44 | 4 |
| Test B | 8 | 12 | 18 | 28 | 11 | 0 |
| Control RR | 14 | 21 | 29 | 60 | 52 | 2 |
| Control LR | 11 | 20 | 35 | 59 | 45 | 3 |

Test A의 99%는 19 round 이내. 다만 **최대 60 round**의 long-tail game이 존재하며, 이는 아래 게임플레이 문제와 연결된다.

### 3.5 Runtime / throughput

| Suite | Games | Wall s | Steady games/s | Wall-inclusive games/s | CPU avg % (machine) | CPU equiv. threads | RAM avg GiB | RAM max GiB | load1 max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Test A | 8,192 | 131.1 | **65.35** | **62.47** | 24.7% | 19.77 | 13.73 | 14.58 | 20.20 |
| Test B | 2,048 | 35.3 | **74.66** | **58.02** | 21.2% | 16.93 | 12.45 | 14.44 | 20.05 |
| Control RR | 512 | 5.1 | 1,195.22 | 101.33 | 16.7% | 13.35 | 9.50 | 12.34 | 18.58 |
| Control LR | 512 | 5.1 | 1,037.54 | 101.34 | 16.8% | 13.41 | 9.46 | 12.34 | 20.13 |

- Best worker config: **20 workers, engine_workers=1, batch_size=32, torch_threads=1**.
- Test A 기준 8,192 valid games를 131.1초에 완료.
- Machine-wide CPU는 80 logical thread 기준 24.7% (약 19.8 thread 상당). 20 physical core를 사용했으므로 worker당 1 core가 거의 포화된 상태와 일치한다.
- RAM은 최대 14.58 GiB로 62 GiB host에서 안전. Swap thrash 없음.
- Disk free: 평가 중/후 약 72 GB 유지 (12 GiB reserve 이상).

### 3.6 Controls가 말해주는 것

- Random vs Random: 0.4746 (CI 0.4355–0.5137) → fixture/seat/team sampling이 한쪽으로 치우치지 않음.
- Lexicographic vs Random: **0.0957** (CI 0.0723–0.1211) → **lexicographic legal은 random보다 훨씬 약한 baseline**이다.
- 따라서 Test B의 0.9990은 “lexicographic을 압도했다”는 sanity check일 뿐, random 대비 성능 이상의 학습된 전략성을 입증하지 않는다. 이 해석을 README/readiness 문서에도 명시했다.

### 3.7 Trace sample

- 30 traces: 10 random wins, 10 random losses, 10 longest remaining games.
- 30 / 30 replay가 원래 outcome과 round count를 재현.
- Trace sample 기준 (population 추정 아님):
  - `move` 854 (72.0%), `pass` 127 (10.7%), `pick` 120 (10.1%), `switch` 51 (4.3%), `move+mega` 23 (1.9%), `move-no-slot` 11 (0.9%).
  - 가장 많이 선택된 move: Protect 119, Flare Blitz 46, Close Combat 30, Stockpile 30, Minimize 27, Yawn 25, Soak 25.
  - Mega-bearing candidate level 176회 중 Mega 선택 23회.
  - Protect-family 선택 123회.
  - 연속 동일 policy decision tuple 64 / 582 (11.0%).
- Trace sample은 long game을 의도적으로 포함하므로 Protect/Stockpile/Minimize 비율을 전체 population 비율로 해석하면 안 된다.

---

## 4. 가장 중요한 게임플레이 문제

1. **Long-tail stall/loop 경향**
   - Test A max 60 round, 41–80 round game 4건.
   - 60-round loss trace(1285)와 54/51-round win trace에서 Protect, Stockpile, Minimize, Life Dew, `move-no-slot`/pass가 반복되는 구간이 관찰됐다.
   - 즉, 대부분의 게임은 9 round 내외로 끝나지만, 일부 게임에서 마무리하지 못하고 장기전으로 늘어지는 경향이 있다.

2. **반복 action pattern**
   - Trace sample에서 연속 동일 policy decision tuple이 64/582 (11.0%).
   - 일부는 PP 소진/강제 no-op 때문이지만, Stockpile/Minimize/Protect 반복은 모델의 closing ability 약점으로 볼 여지가 있다.

3. **Mega 사용률**
   - Trace sample에서 Mega candidate가 있던 level 176회 중 23회만 Mega를 선택했다.
   - 이것이 항상 잘못된 것은 아니지만, Mega opportunity 활용이 보수적일 수 있다는 관찰이다. Population 전체의 Mega opportunity/usage 비율은 별도 계측이 필요하다.

4. **Test B baseline의 약함**
   - Lexicographic legal은 random보다 훨씬 약하다 (control 0.0957).
   - 따라서 “deterministic legal baseline을 이겼다”는 사실만으로 전략적 학습을 주장하면 안 된다.

5. **없는 문제 (중요)**
   - Illegal action 0.
   - Operational error 0.
   - Draw 0.
   - Missing/duplicate fixture 0.
   - Trace replay mismatch 0.
   - Seat bias 징후 없음.
   - 특정 team에 패배가 집중되는 현상 없음.

---

## 5. 해석상 한계

- 상대는 **branch-wise uniform legal random**이다. “uniform over all joint action combinations”가 아니다.
- Test B의 lexicographic legal은 강한 strategic opponent가 아니다.
- Fixture는 frozen 1,220 training-team pool에서 나왔다. 외부 team pool 일반화는 검증하지 않았다.
- 이 평가는 offline native engine + frozen policy path만 검증한다. **Showdown adapter, network protocol, timer, deployment wiring은 검증하지 않았다.**
- 98.9% random 승률은 ladder competitive strength의 증거가 아니다.

---

## 6. Showdown readiness 판정

### 판정: **READY_FOR_SHOWDOWN_SMOKE_TEST**

근거:

- A. Basic game competence: random 상대 8,099/93/0, score 0.9886, CI 0.9863–0.9910. Pass.
- B. Learned behaviour: lexicographic legal 상대 2,046/2/0, score 0.9990. 단, control이 보여주듯 baseline이 약하므로 “random 이상의 전략성” 증거로 과대해석하지 않음.
- C. Gameplay quality: trace 30/30 재현. Move/target/switch/Protect/Mega action이 기록됨. Long-game stall/loop 경향은 관찰됐지만 즉시 smoke test를 막을 correctness issue는 아님.
- D. Technical correctness: illegal 0, operational error 0, coverage 누락/중복 0, checkpoint mutation 0, formal state write 0. Pass.
- E. Showdown readiness: offline path만 검증됨. Showdown adapter smoke test는 별도로 필요.

이 판정은 **smoke test 승인**이지 ladder 성능 보장이 아니다. Smoke test에서 adapter/protocol/timer/deployment를 확인한 뒤 별도 판단이 필요하다.

---

## 7. 보호 상태 검증 (평가 후)

| 항목 | 상태 |
|---|---|
| Production PPO | stopped 유지 |
| STOP marker | 존재, empty, SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`, mtime 2026-10-17 13:51:24 KST — 평가 중 변경 없음 |
| Production source | `da35700f75c95833b0e8250d7b4b04e11320f8b7` 유지 |
| Frozen checkpoint rank0 | `fd3b5655a2b7d602043531cf31addcbca53c4c41523dce4abc430a8ea52fe02b` 유지 |
| Frozen checkpoint rank1 | `6cd1ae7b84a12bbc8e12bc0e30f538ee59d03e4dd602f68eb47ed0f079b39607` 유지 |
| Formal worker | idle (child job 없음), read-only 관찰만 함 |
| Formal alpha/registry | 평가 중 write 없음 (mtime 13:22 / 13:57 유지) |
| Champion / promotion receipts | 변경 없음 |
| GPU | 사용하지 않음 (`CUDA_VISIBLE_DEVICES=""`) |
| SIGKILL | 사용하지 않음 |
| Disk reserve | 12 GiB 이상 유지 (약 72 GB free) |

---

## 8. 재현 명령

```bash
cd /home/minidc/Desktop/PokeAgent3.0-RandomBaseline

PY=/home/minidc/Desktop/PokeAgent3.0/PokeAgent3.0-agent1-pa3/.venv/bin/python
RUN=/home/minidc/eval-league/random-baseline/v1
CKPT=/home/minidc/eval-league/v2/snapshots/43/434be431de0e25a9c7730a7a0f38eda66d5db30397835983ebde9f4b8f1d9b36

# 1) frozen fixtures
$PY scripts/random_baseline_eval.py --run-dir "$RUN" prepare

# 2) calibration (8/12/16/18/20 workers + thread/batch variants)
$PY scripts/random_baseline_eval.py --run-dir "$RUN" --checkpoint "$CKPT" calibrate

# 3) suites (best config = 20 workers)
$PY scripts/random_baseline_eval.py --run-dir "$RUN" --checkpoint "$CKPT" --workers 20 suite --test test_a
$PY scripts/random_baseline_eval.py --run-dir "$RUN" --checkpoint "$CKPT" --workers 20 suite --test test_b
$PY scripts/random_baseline_eval.py --run-dir "$RUN" --checkpoint "$CKPT" --workers 20 suite --test control_random_random
$PY scripts/random_baseline_eval.py --run-dir "$RUN" --checkpoint "$CKPT" --workers 20 suite --test control_lexicographic_random

# 4) traces + aggregate + publish
$PY scripts/random_baseline_eval.py --run-dir "$RUN" --checkpoint "$CKPT" traces
$PY scripts/random_baseline_eval.py --run-dir "$RUN" --checkpoint "$CKPT" aggregate
```

또는 전체를 한 번에:

```bash
$PY scripts/random_baseline_eval.py --run-dir "$RUN" --checkpoint "$CKPT" all
```

---

## 9. Hash / identity appendix

- Checkpoint file SHA-256: `434be431de0e25a9c7730a7a0f38eda66d5db30397835983ebde9f4b8f1d9b36`
- Candidate source SHA: `da35700f75c95833b0e8250d7b4b04e11320f8b7`
- Engine SHA-256: `8f3643a9f7b2febfa567f039b911d7af2f88bbf7bfc912849ada9e6a690d1c62`
- Team pool SHA-256: `ecabd47a8d000ba8dd9f3010c09992505fe21e95efe244e2c8a3d75ac3f70b1f`
- Dataset manifest SHA-256: `1aa64507e17f4a1f819d59834dbd314e96a1a1b6d13f94a791bd3f51fca309bb`
- Fixture manifest hash: `5fe97fe15be9874f2461ac89516957ece515761d9cf770732fb70333eaacd84e`
- Evaluator: `pa3-native-batched-evaluator-v3`
- Sampling: `fixture-seat-decision-branch-keyed-valid-cdf-v3`
- Seed domain: `pa3-random-baseline-v1`
- Python: 3.12.3, torch 2.14.0+cu126 (CPU only)

---

## 10. 산출물 위치

- `docs/operations/random-baseline-v1/README.md`
- `docs/operations/random-baseline-v1/summary.md`
- `docs/operations/random-baseline-v1/performance.csv`
- `docs/operations/random-baseline-v1/team_results.csv`
- `docs/operations/random-baseline-v1/seat_results.csv`
- `docs/operations/random-baseline-v1/behavior_report.md`
- `docs/operations/random-baseline-v1/performance_benchmark.md`
- `docs/operations/random-baseline-v1/readiness_assessment.md`
- `docs/operations/random-baseline-v1/manifest.json`
- `docs/operations/random-baseline-v1/traces/` (30 JSON + 30 Markdown + index)

대용량 raw receipt는 Git 밖의 run root에 보관한다:
`/home/minidc/eval-league/random-baseline/v1/`
