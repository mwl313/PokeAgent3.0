# PokeAgent 3.0 — Extreme Throughput Optimization Master Plan v3.0

**문서 성격:** Codex / miniDC 에이전트 실행 가능한 후속 엔지니어링 사양, 벤치마크 및 승격 게이트  
**작성 기준:** 2026-10-09 (Asia/Seoul)  
**프로젝트:** [`mwl313/PokeAgent3.0`](https://github.com/mwl313/PokeAgent3.0)  
**확인한 기준 Git SHA:** `cba9dfac2b57e32574d612194c478affb2867cb2`  
**작업 브랜치:** `optimization/pa3-realpolicy-throughput`  
**기본 실행 구조:** Rust CPU battle engine + PyO3 + PA3-8M PyTorch GPU policy + on-policy PPO + Tesla V100 PCIe 32GiB ×2  
**상위 계약:** `docs/spec/fullspec-1.1-minidc-20261006/POKEAGENT_3_0_FULL_SPEC.md`, `configs/train.yaml`  
**계승:** `docs/PokeAgent3_Optimization_Master_Plan_v2.0_2026-10-09.md` 및 `docs/perf/` 측정/감사 결과  
**목표:** 현 장비의 처리량 프런티어를 공격적으로 추적한다. **학습이 제대로 되는 상태에서** 자연완결 games/s, learner decisions/s, 장기간 PPO all-in games/s를 최대화한다. 임의로 1,000 games/s를 성능 상한선으로 두지 않는다.

> **집행 규칙:** 이것은 최적화 작업 *지시서*이며 본학습 개시 승인 문서가 아니다. 모든 변경은 측정 → 최소 PoC → 수학 및 의미 보존 검사 → 동일 조건 A/B → 반복 장시간 벤치마크 → 채택/폐기 → 커밋 순서로 진행한다. 높은 속도 수치를 위해 규칙을 생략하거나, 행동 후보를 자르거나, 정책을 stale하게 만들거나, PPO 목적함수를 바꾸지 않는다. 불변 계약을 바꿔야만 가능한 아이디어는 별도 제안서로 격리한다.

---

## 0. 작업자용 1페이지 결정 요약

1. **기존 진행을 보존:** `git fetch` 후 origin HEAD와 작업 트리 상태를 확인한다. 기준 시점 HEAD는 `cba9dfa`지만 그 이후 새 커밋이 있으면 그것이 정본이다. v1.1/v2.0 성과, 기존 raw JSON, checkpoints를 덮어쓰지 않는다. 새 브랜치나 후속 커밋으로 진행한다.
2. **현재 가장 큰 미활용 자원:** GPU1은 단일 GPU learner에서는 유휴. 먼저 **정확한 실제 2-GPU DDP 전체 PPO**를 완성한다. 동시에 단일 GPU `microbatch=2048`을 별도 A/B로 시험한다.
3. **VRAM을 실제로 활용:** 32GiB 장당 **28GiB soft budget**. 지금까지 `microbatch=1024` 학습은 9.25GiB 예약. 공간이 남았다는 것과 GPU compute가 남았다는 것은 다르다. 큰 batch를 성능 실험으로 시험하되 메모리만 많이 쓰는 설정은 폐기한다.
4. **Actor의 구조적 제약:** 512, 1024, 2048 환경 수를 바꿔도 98~103 games/s였다. 따라서 환경 수만 늘리는 것보다 **Rolling Slots, 독립 코호트 파이프라인, NUMA-로컬 버퍼, 추론 전송 중첩**을 우선한다.
5. **준비된 최적화는 다시 하지 않는다:** fixed observation batch, packed legal candidate/prefix, FP16 actor, `inference_mode`, fast rollout, 정확한 PPO 집계, streaming minibatch, 1024 microbatch 실험은 완료. 기존 코드를 반드시 재활용한다.
6. **기록의 정직성:** `33.37 games/s`는 **단일 GPU의 제한된 PPO+실제 checkpoint 포함 실측**, 주기 평가 및 DDP 없는 수치다. `162.73 games/s`는 듀얼 GPU **actor-only**이다. 둘을 더하거나 같다고 쓰지 않는다.
7. **규칙 readiness는 별개:** 현재 10/16 PASS, 미완료 2/3/5/8/15/16. 16/16과 사용자 승인 이전에는 100M 본학습을 실행하지 않는다.
8. **실험과 운영을 구별:** 벤치마크는 제한적이고 복구 가능해야 하며, 모든 최적화는 기본값 변경 전에 gate를 통과해야 한다. 사용자가 지정한 드라이버, CUDA, PyTorch, 전력, BIOS, 서비스 및 다른 GPU 프로세스는 임의 변경 금지.

---

## 1. 증거 등급 및 확인된 성능 정본

### 1.1 원본 기준점

| 관측 항목 | 검증된 결과 | 범위와 한계 |
|---|---:|---|
| 초기 actor | **15.07 games/s** | GPU0, 2,048게임 ×3회, fp32/per-view 초기 상태 |
| 현 단일 GPU actor | **103.44 games/s** | GPU0, 2,048게임 ×3회, 최적화된 real-policy 수집 |
| 현 장기 단일 GPU actor | **약 102 games/s** | 10,240게임 단일 실행, 버전 간 약간의 편차 |
| 듀얼 GPU actor | **162.73 games/s** | 4,096게임, wall-aligned. PPO learner 미포함 |
| 기존 단일 GPU bounded PPO | **26.43 games/s** | 10,240게임, microbatch 256, packed from_rows learner, 체크포인트 방식 상이 |
| 최종 단일 GPU PPO + checkpoint | **33.3743 games/s** | **10,240 자연완결게임**, collect 100.206초, learner 205.983초, checkpoint 0.635초. GPU0, streaming + microbatch 1024 |
| 1024 microbatch 반복 검증 | **중앙값 32.80 games/s** | 2,048게임 ×3회, 실제 체크포인트 포함 |
| 1024 microbatch GPU 예약 | **약 9.25GiB** | 단일 V100, 32GiB 중 28GiB 운영 예산 |
| 10k streaming peak RSS | **7.34GiB** | 과거 18.0GiB, 약 59% 감소 |
| 현재 학습 병목 | **약 206초 learner / 약 100초 actor** | 10k 제한 PPO, 동일 시스템에서 측정 |
| full engine readiness | **10/16 PASS** | FAIL 2,3,5,8,15,16. 최적화 성공과 별개 |

**기준 파일:** `docs/perf/V2_LEARNER_BREAKDOWN.md`, `V2_FRONTIER_REPORT.md`, `V2_CORRECTNESS_AUDIT.md`, `ULTIMATE_FRONTIER_REPORT.md`, `FINAL_REPORT.md`, `runs/perf/v2_stream_10k.json`, `runs/perf/v2_micro_1024_x3.json`, `runs/perf/dual-real.json`.

**증거 등급:** `MEASURED` = raw JSON과 일치하는 실측, `CODE-OBSERVED` = 현재 소스 구조에서 확인된 지연 후보, `HYPOTHESIS` = 아직 증명되지 않은 성능 가능성, `PROPOSED` = 실험/새 구현. 모든 문서에서 이 구분을 유지한다.

### 1.2 GPU를 두 장 쓴다고 성능이 자동으로 2배가 되는 것은 아니다

- GPU0과 GPU1은 각각 독립적인 32GiB VRAM이며 **합산 64GiB 공유 주소공간이 아니다**. GPU 간 P2P/NVLink가 없고 host-staged 통신이다.
- GPU당 독립 정책 모델과 배틀 코호트를 운영하고, learner는 **global minibatch 4,096을 두 rank로 분할**한다. 모델 텐서 병렬화를 요구하지 않는다.
- GPU1의 전력 제한 150W는 최근 actor 측정에서 더 느린 성능으로 나타나지 않았다. 사용자가 승인하지 않은 power limit 변경은 하지 않는다.

### 1.3 실제 현재 병목의 분해와 Amdahl 기준

10,240게임에서 `100.206 + 205.983 + 0.635 ≈ 306.824초`, `33.374 games/s`.

- **Learner만 시간 1/2** → 약 **50 games/s** (actor는 그대로라는 계산상 예시).
- **Actor만 시간 1/2** → 약 **40 games/s**.
- **양쪽 시간 1/2** → 약 **67 games/s**.
- **Learner가 시간 0에 수렴**해도 actor가 현재 상태면 약 **102 games/s** 부근이 계산상 상한.

위 숫자는 **달성 가능성 전망이 아니라 병목 설명용 산술**이다. 개선 범위가 달라지면 다른 병목이 생기고, DDP 통신/overlap/운영비로 결과도 달라진다. 어느 하나만 개선하는 계획으로는 지속적으로 높은 all-in 성능을 얻을 수 없다.

---

## 2. 손대면 안 되는 계약: 0타협 원칙

### 2.1 모델 및 학습 수학

- 모델: PA3-8M, 6-layer d=320 Transformer, 현재 scorer와 value head, **96 padded tokens**, 원래 관측/마스크/정보 의미 유지.
- 팀: 동결된 **1,137개** `mb-mc-v3-userteam-all-train`, manifest digest 변경 금지, 균등한 지정 분포 유지.
- 규칙: `gen9championsvgc2026regmc`, Closed Team Sheet, Bo1, Mega 관련 pinned regulation, player-safe 정보, **모든 합법 prefix 및 타겟 후보**. 임의 truncation/skip/fallback/조기 승패 추정 금지.
- 학습: scratch self-play, 단위 자연완결 경기, 1억 경기 목표(승인 전 실행 금지). `gamma=1`, `GAE λ=.95`, terminal-only 승리 +1/패배 -1/무승부 0.
- PPO: 4 epochs, **global minibatch 4,096 유지**, Adam 설정 및 scheduler, 클리핑, entropy, uniform reverse-KL, target KL, masked statistics, actor와 value의 구분된 표본 분모 유지. FP16 계산 허용 범위와 FP32 확률 및 loss 유지.
- `PPOConfig`의 microbatch 256은 정본 기본값이다. 1024는 실험으로 유효성을 보인 launcher 옵션이며 **기본값 승격이나 DDP 값 변경은 별도 안전성 gate 후** 한다.
- 정책은 rollout iteration 동안 freeze하고 각 episode의 상대 정책도 freeze한다. PPO epoch 중 다음 데이터 수집을 위해 learner의 갱신된 weight와 구형 weight를 섞어 쓰지 않는다.
- `target_min_learner_decisions=131072`, `target_min_completed_matches=2048`, 두 기준 충족 후 자연 종료 drain, 초과 경기 보존, epoch 간 전체 관측 입력 동일성을 지킨다.

### 2.2 시스템/자원 보호

- Xeon E5-2673 v4 ×2, NUMA 2노드, RAM 약 62GiB, V100 PCIe 32GiB ×2.
- per-GPU 28GiB soft limit; host total 48GiB, rollout host 24GiB, pinned host 전체 1GiB. 실험 peak 기록, 이 한도를 잠재적으로 넘어가는 실험은 dry-run estimate와 사용자 승인 필요.
- driver 580.178.04, CUDA Toolkit 12.8.2, PyTorch 2.14.0+cu126, Python 3.12, Volta sm70 핀 유지. CUDA 13, BF16/FP8 전제, FlashAttention-2 필수화, 무단 `torch.compile`, 무단 pip/apt 변경 금지.
- GPU0 175W, GPU1 150W, service `dsh-web`/`llama-swap` 등 그대로 둔다. 관련 없는 PID 종료, GPU 전력 설정 변경, BIOS, governor, 팬, OS 설정 변경 금지.
- 필요한 검증 동안 프로파일링 trace는 로컬에 저장하고 Git에는 compact summary만 올린다. 디스크 여유와 실수로 생성한 대용량 trace를 관리한다.

---

## 3. 완료된 작업과 보존 대상

| 영역 | 상태 | 보존 및 후속 조치 |
|---|---|---|
| Fixed batched observation | **DONE** | per-view oracle 대비 동일성 유지 |
| Packed candidate + packed prefix walk | **DONE** | 모든 prefix 오라클 및 legal masks 무손실 유지 |
| Actor FP16 + `inference_mode` | **DONE** | FP32 logprob/확률 정확성 보호 |
| Fast packed rollout rows | **DONE** | row pairing 오류 회귀 테스트 유지 |
| PPO 마지막 micro 통계/KL 수정 | **DONE** | weighted sum/denom, early-stop 기준 보호 |
| Padded/uneven actor gradient correction | **DONE (single rank)** | 실제 PA3-8M gradient와 DDP rank별 재검증 필요 |
| GradScaler skip-step accounting | **DONE** | skipped step/report/scheduler accounting 보호 |
| 실제 checkpoint write와 stage share 수정 | **DONE** | metrics+eval 포함 장시간 operational benchmark 아직 필요 |
| Learner profiler (M0) | **DONE, 재프로파일 필요** | 256 마이크로 기준 58초 breakdown은 1024 최신 GPU kernel profile이 아님 |
| Worker 8–20, env 512–2048 sweep | **DONE** | 98–111 games/s의 평탄 구간, 단순 env 확대만으로 얻는 이득 작음 |
| Real dual-GPU actor | **DONE** | 162.73 games/s wall-aligned, learner DDP 없음 |
| Microbatch A/B 128–1024 | **DONE** | 1024 ×3 및 10k 확인, 2048/4096 미시험 |
| Streaming minibatch | **DONE, opt-in** | 10k peak RSS 7.34GiB, 속도 중립, full mathematical equivalence gate 추가 |
| SATA/swap/NUMA 조사 | **DONE** | 학습 read_bytes=0, swap=0, major faults=0, iowait=0, NVMe 미관측 병목 |
| CUDA stream overlap, Rolling Slots, DDP learner | **OPEN** | v3 핵심 작업 |
| Full-scope mechanics readiness | **OPEN** | 10/16, 최적화 브랜치에서 임의 rule 축소 금지 |

**보고서 정정 과제:** 일부 기존 문서의 `2,112 per 2k microbatch syncs`는 같은 2k 조건의 28 optimizer steps와 256 microbatch를 사용한 `28×16=448` micro calls와 일치하지 않는다. 실행 경로별 실제 호출 횟수를 이벤트/카운터로 측정해 해당 설명을 정정한다. 상충 숫자를 새로운 성능 증거로 재사용하지 않는다.

---

## 4. 전체 실행 로드맵: 우선순위, 파급효과, 위험

| ID | 우선순위 | 실행 패키지 | 대상 병목/목적 | 예상 증거 상태 | 위험 |
|---|---|---|---|---|---|
| **G0** | **BLOCKER** | 실제 PA3 수치 parity, streaming 일치, 정책 수집 감사 | DDP 착수 전 정확성 | tests 확대 필요 | 높음 |
| **G1** | **매우 높음** | 듀얼 V100 DDP **all-in** | 유휴 GPU1, 206초 learner | 실제 미측정 | 높음 |
| **G2** | **매우 높음** | VRAM 기반 `microbatch 2048/조건부 4096` | 많은 GPU 작업 호출과 accumulation | 1024까지 개선 실측 | 중간 |
| **G3** | **매우 높음** | Rolling Slots와 late-tail 계측 | 종료 게임 이후 유휴 env | 구조상 후보, 속도 미측정 | 높음 |
| **G4** | **매우 높음** | 독립 코호트 CPU/GPU 중첩 | 직렬 round-bound pipeline | 환경 sweep상 유력 | 높음 |
| **G5** | 높음 | GPU 추론 batch/버킷 및 buffer residency | 모델 호출 24.5%, H2D 10.2% actor | 비중 실측, 구현 가설 | 중간 |
| **G6** | 높음 | Native observation/packed row/encoder cache | actor 관측 생성 약 17% | 비중 실측 | 중간 |
| **G7** | 중간~높음 | Compressed candidate, columnar/SoA, CPU meta 분리 | int64 dense, Python allocation, PCIe | 코드상 확인 | 중간 |
| **G8** | 중간 | 최신 GPU kernel, SDPA, embedding, optimizer, CUDA Graph PoC | learner compute 및 kernel dispatch | 256 시점 kernel trace | 높음 |
| **G9** | 중간 | Pinned NUMA staging + double buffer | pageable H2D 측정 | 전송 일부 실측 | 중간 |
| **G10** | 낮음 | GAE 및 작은 CPU 루프 튜닝 | 예전 learner 약 2% GAE | 낮은 우선순위 | 낮음 |
| **G11** | 필수 | 장시간 운영, 리소스, 체크포인트, 평가 포함 | 벤치마크 정직성/재현성 | 미완료 | 중간 |
| **R** | 별도 | full engine readiness + sample efficiency | 실력 및 규칙의 품질 | 별도 작업 트랙 | 별도 |

**작업 순서:** G0 → G1의 기본 DDP correctness/throughput → G2의 독립 단일 GPU A/B → G1+G2 복합 확인 → G3과 G4 PoC → G5/G6/G7 계측 기반 최적화 → G8/G9 선별 적용 → G11. 동시에 개발 가능한 코드라도 **성능 승격과 baseline 측정은 격리**한다. G3/G4가 G1을 무기한 지연시키지 않도록 한다.

---

## 5. G0 — 시작 전 진짜 학습 정확성 게이트

### 5.1 단일 GPU PA3-8M 전체 gradient 검증

대상: `agent/ppo/learner.py`, `tests/agent/test_update_statistics.py`, 새 `tests/agent/test_real_gradient_parity.py`.

- **실제 PA3-8M**의 identical weights/obs/candidate/mask/advantage/minibatch 4096에서 한 번에 처리한 참조 gradient와 microbatch `256/512/1024/2048`를 통한 합산 gradient를 비교한다. 여유가 없으면 참조용 1,024 크기 단계부터 시작한다.
- FP32에서 parameter별 max abs/relative error, global norm, actor-specific, critic-specific gradient를 별도 기록. FP16은 미리 선언한 수치 오차 및 loss-scaler step 상태로 비교한다.
- singleton actor 행 0, 값만 업데이트되는 행, masked candidate, 마지막 padded minibatch, 분기 1/2/3/4, rare request 종류를 포함한다.
- 한 microbatch가 actor 0개 또는 valid value 0개일 때 NaN/0 division/optimizer step 불일치가 발생하지 않아야 한다.
- loss value 0.5 MSE, entropy, reverse KL의 계수가 바뀌지 않아야 한다. FP16 확률 계산은 FP32 유지.

### 5.2 Streaming vs 기존 경로의 동등성

- `prepare_batch/update()`와 `prepare_streaming/update_streaming()`에 **동일 shuffle order, model seed, weights, rows, padding/permutation, flags**를 주어 loss, gradient, optimizer state, report, checkpoint/state_dict를 비교한다.
- `update_streaming`이 마지막 부분 배치의 padding을 한쪽에 몰아두고, `RolloutBatch.iter_minibatches`는 padding을 흩는 차이가 있다. gradient weighting이 같다면 정상일 수 있으나 **forward dynamics와 GPU microbatch 경계 효과**를 포함해 A/B 검증해야 한다.
- 현 `tests/agent/test_update_statistics.py`는 작은 StubModel 중심이다. 실제 모델 parity는 별도 시험 결과가 필요하다.

### 5.3 게임 수집 스펙과 실제 경로 대조

- `NativeCollector` 주석/코드에서는 opponent 정책이 current여도 learner seat만 학습 행으로 적재하는 방식이 보인다. 한편 `configs/train.yaml`은 `collect_both_sides_when_current_self_play: true`이다. **한 iteration에서 양측 current row가 실제로 수집되는지** 정확히 감사한다. 이는 구현/주석/설정 불일치 가능성이며, 버그라고 미리 단정하지 않는다.
- current/historical mix, opponent-only 행 exclusion, side trajectory와 terminal reward, natural match counting, prefix-dependent masks, GAE all-pass.
- 정책 freeze, 배틀 별 RNG seed, full-group uniform team sampling 및 재현성 보호.

**G0 수락:** 새 real gradient 및 streaming parity tests PASS. 실제 양측 수집 계약 감사 문서 작성. 현재 수치와 상충하는 동기화 횟수 문서 정정. 이 Gate가 실패하면 DDP 속도 실험을 정상 학습 성능으로 승격하지 않는다.

---

## 6. G1 — 두 V100의 DDP all-in 완성 (최우선 구조 개선)

**계측 근거:** 두 GPU의 real-policy actor는 이미 동작하나 PPO backward/update는 GPU0에서만 실행. 현재 all-in의 약 2/3는 learner. 정확한 DDP가 되면 두 GPU 연산 능력을 활용할 여지가 크다. **기대 향상 배수는 미측정**.

### 6.1 아키텍처

- 각 rank: 독립 PyO3 NativeEngine, 로컬 배틀 풀, frozen model replica, 로컬 rollout compact rows, NUMA-local host allocation, GPU별 FP16 policy forward.
- iteration 정책과 opponent snapshot은 동일 digest, learner를 업데이트하기 전 local rollout 완료 및 global match/decision counter로 동기화.
- 두 rank는 local minibatch **2048 + 2048 = global 4096**. 랭크 2개에서의 gradient 동기화는 NCCL. `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=0`을 **해당 프로세스 환경에서만** 사용.
- `DistributedDataParallel(model)`의 **DDP forward 경로를 우회하지 않는다**. 현재 `self.model.encode()`와 `self.model.evaluate_encoded()` 같은 메서드 직접 호출 대신 `DDP(model)(batch, mode="learner")`처럼 공통 graph를 소유하는 `forward()` 엔트리를 설계한다. 모델 math는 동일하게 유지하고 `model.module` 직접 호출로 DDP reducer를 건너뛰지 않는다.
- Gradient accumulation 중 최초 N-1 micro의 **forward와 backward 모두** `with ddp.no_sync():` 안에서 실행하고, 마지막 유효 micro의 양쪽은 DDP sync를 수행. PyTorch 공식 경고에 따라 no_sync를 backward에만 적용하면 안 된다.

### 6.2 **중요 수학: actor/value의 글로벌 분모는 다르다**

rank r의 policy/entropy/uniform-KL 항에 참여하는 유효 actor 행 수를 `A_r`, value 항에 참여하는 유효 행 수를 `V_r`라 하자. rank 간 all_reduce로 `A = Σ_r A_r`, `V = Σ_r V_r`를 **해당 global optimizer minibatch마다** 계산한다. `R=2`인 기본 PyTorch DDP 평균화(reduce 이후 world-size로 나눔)가 켜져 있다면 rank별 loss는:

```
L_rank_r = R * [
    (sum(actor PPO + entropy + uniform KL terms on rank r) / max(A, 1))
    + value_coef * (sum(value terms on rank r) / max(V, 1))
]
```

- DDP가 gradient를 평균내면 총 gradient는 참조 single-process global loss `Σactor/A + value_coef*Σvalue/V`와 일치한다. **이미 rank local mean을 계산한 뒤 각 rank를 같은 가중치로 DDP 평균내는 방법은 A_r/V_r가 불균등할 때 틀리다.**
- 기존 `sample_weight`를 추가로 중첩해 **이중 가중**하지 않는다. 현재 단일 rank의 `exact_row_weighted_accumulation`과 위 세계 공통 분모를 조화시킨 새 DDP loss adapter가 필요하다.
- DDP grad_clip은 동기화 뒤 전체 global gradient에 수행. GradScaler overflow/skip 여부는 rank 간 합의되어야 하고 optimizer state, scheduler 및 committed match clock이 일치해야 한다.
- 마지막 불균등 minibatch와 rank에 actor 0인 극단 케이스도 모든 rank가 동일한 collective 및 optimizer-step 경계를 지나도록 제어한다. Dummy padding은 mask=0으로 계산에서 제외하며 실제 matches/advantages를 늘리지 않는다.

### 6.3 DDP 실험 구성

1. NCCL 최소 collective smoke, 2 GPU link/NUMA/SHM 시간 계측. 설정 변경 없이 실행.
2. Synthetic uneven actor/value count gradient test: 1 GPU global 4096 참조 vs 2 rank 2048, 마지막 부분 배치와 actor=0 rank. FP32 우선.
3. 실제 PA3-8M 동일 batch, 동일 global sample order 단일 GPU vs DDP gradient/one-step optimizer parity. FP16 후속.
4. Per-rank microbatch `256`, `512`, `1024`, `2048(조건부)` A/B. 동일 global minibatch 4096 유지. `no_sync` 횟수, GPU memory, NCCL time, rank별 idle/straggler, host RAM 기록.
5. Dual actual-policy 2k×3, 10k 이상, 4 PPO epochs와 실제 checkpoint/resume까지 포함. `single learner + dual actor` 대비도 비교해 **DDP가 느리면 fallback** 수용.
6. `GPU0` 주 학습 / `GPU1` frozen actor 또는 critic eval 전담 등 대체 스케줄도 계측만 해보고 전체 all-in이 빠를 때만 채택. 명시적 정책 freeze 규칙 유지.

**G1 수락:** 글로벌 gradient/maxnorm/optimizer-step/score parity, 0 NCCL hang, 0 illegal/abort, 0 unseen overflow, synchronized model digests, `two_gpu_all_in_games/s`가 실제 단일 GPU 기준보다 측정 오차 이상 개선되거나 명확한 실패 원인 기록. DDP만 느리면 GPU1 actor 전담 병렬 운영도 대체 비교한다.

---

## 7. G2 — VRAM 32GiB 극대화: microbatch/추론 batch 탐색

### 7.1 현재 학습 microbatch 실측

| microbatch (단일 GPU) | 2k bounded all-in | learner 시간 | GPU reserved |
|---:|---:|---:|---:|
| 128 | 20.46 games/s | 79.3 s | 2.84GiB |
| 256 | 24.66 games/s | 60.0 s | 2.86GiB |
| 512 | 29.14 games/s | 48.2 s | 4.88GiB |
| **1024** | **31.57 games/s** (3회 median 32.80) | **42.9 s** | **9.25GiB** |
| **2048** | **UNMEASURED** | **UNMEASURED** | **UNMEASURED** |
| **4096** | **UNMEASURED, 조건부** | **UNMEASURED** | **UNMEASURED** |

- 1024 설정은 `28GiB soft budget` 대비 여유가 있다. **2048은 최우선 A/B 후보**. 4096은 2048의 실제 최고 할당/예약 메모리와 GPU utilization을 본 후 OOM 위험이 충분히 낮은 경우에만 짧은 trial. VRAM 요구량을 선형 외삽해서 예약하지 않는다.
- 정확한 `microbatch`는 **PPO update 중 GPU에 동시에 들어가는 학습 row 수**이다. Rust concurrent environments 1024/2048와는 다른 변수다. `global_minibatch=4096`은 유지.
- 2048 이후 compute/launch 및 memory bandwidth 포화가 나타나면 더 큰 batch를 무작정 늘리지 말고 가장 빠른 설정으로 복귀한다. GPU reserved <28GiB뿐 아니라 allocated peak, fragmentation, activation, DDP gradient buckets, 다른 정책 replica까지 감시한다.

### 7.2 학습 데이터 작업 공간의 GPU 상주와 재사용

- Per-rank double-buffered static HBM staging: ObservationBatch 고정 features, masks, compact candidate slab, score/value outputs. 필요한 minibatch를 GPU로 전송하되 allocation 재사용으로 GPU allocator/Python overhead 감소 가능성 측정.
- Observation 관련 category index가 embedding lookup에서 `long`을 요구하더라도 **압축된 전송** 후 GPU에서 안전하게 cast할 수 있는지 비교한다. CPU에서 이미 확장한 `int64` 전체를 복사하는 현재 비용과 비교.
- `index_select`를 반복하는 현재 `RolloutBatch.select` 경로 대신 contiguous narrow `slice` 또는 preallocated gather + CUDA-safe permutation을 비교한다. **shuffle 전체 distribution, sample weighting 및 결과 동일성** 보존.
- 동일한 GPU RAM에 전체 rollout을 resident로 두는 아이디어는 136k개의 96-token input을 모두 고밀도 float32로 펼치면 매우 클 수 있으므로 기본이 아니다. **작은 순환형 minibatch working set**을 우선한다.

### 7.3 Actor의 추론 batching: 게임 수 증설과 분리

- 로그에 **실제 GPU encoder batch 크기 분포**를 기록: 각 라운드에서 1,2,3,4 branch grouping별 request 수, 가득 채워진 정도, P50/P90/P99와 유휴 시간. `envs=512,1024,2048`에서 batch distribution 비교.
- `inference_max_batch_per_gpu=512`, 현재 실측 시 group 구현, `sample_level_step` 실제 입력 배치 용량 사이 차이를 감사. 기존 계약상 max512를 무단 변경하지 않는다. 256/512/조건부 1024+를 *실험 오버라이드*로 사용하려면 입력 분할/merge를 의미 보존하여 별도 명시/승인한다.
- Request/branch shape 기반 **shape bucketing**으로 같은 토큰 길이(96) 및 candidate capacity의 실제 사용량별 배치를 만든다. Bucket 8/16/32/64의 **합법 후보 전원 포함** 검증; 후보 수가 64보다 크면 silent cutoff 금지, bounded dynamic fallback.
- 동일 정책에 대해 여러 cohort의 준비 완료 requests를 합쳐 **큰 GPU 추론 batch**를 만들되, 같은 게임의 prefix 의존성은 절대 건너뛰지 않는다. 2~4 cohort에서 dispatcher 대기 시간과 batch size 효과를 측정한다.

**G2 수락:** 2k×3 및 10k 확인, GPU allocated/reserved, CUDA kernel actual time, optimizer steps/GAE/score parity. 메모리를 더 썼을 뿐 games/s가 증가하지 않으면 채택하지 않는다.

---

## 8. G3 — Rolling Slots: 먼저 끝난 게임 즉시 대체

**코드상 근거:** `agent/train/native_collector.py::collect()`는 `reset_cohort()` 뒤 `while not all(finished)`로 해당 코호트가 전부 자연 종료할 때까지 기다린다. 끝난 슬롯에 새 게임을 즉시 넣지 못한다. 512/1024/2048 env sweep에서 처리량이 평탄했고, 긴 경기의 꼬리(tail)가 배치를 줄이는 구조적 위험이 존재한다. **실제 long-tail 손실량은 아직 직접 측정하지 않았다.**

### 8.1 먼저 진단해야 할 지표

- 라운드별 `open_envs`, active requests, GPU encoder batch size, game length CDF/P50/P95/P99/max, cohort drain tail seconds, finished-but-idle slot seconds, live-env efficiency (`active_slots / allocated_slots`).
- 2k/10k fixed cohort에서 처음 50% 게임 완료 시점부터 마지막 1개 게임까지 걸린 시간을 따로 측정.
- 짧은 게임과 긴 게임 분포가 다른 팀 아키타입에서 나타나는지, 기존 sample distribution 왜곡 여부 기록.

### 8.2 구현 사양

1. Rust Engine에 **개별 완료된 handle/slot만 reset**할 수 있는 안전한 API가 있는지 확인. 없다면 batch 내부 특정 slot의 안전한 재초기화 API를 개발하되 기존 `reset_batch`는 그대로 유지하고 참조 oracle 검증.
2. `slot_id`와 `match_id`를 별도 관리. `generation` 번호를 붙여 이전 게임의 비동기 요청/응답이 재사용된 슬롯으로 들어가지 못하게 한다.
3. 한 게임의 reward, done, trajectory, RNG, team pair, learner seat가 완결되면 기록하고, **신규 reset이 허용되는 수집 기간**에만 그 슬롯을 재사용한다.
4. Iteration 종료 조건: learner decisions 최소 131072, 자연완결 게임 최소 2048 모두 충족 후 **새 reset 중단**. 이미 시작된 게임은 전부 자연 종료하여 overshoot 보존. 장기 미완료를 draw 처리하거나 삭제하지 않는다.
5. 정책 freeze는 iteration 동안 보존. 재시작 게임도 현재 iteration의 동결 정책과 상대풀을 사용한다.
6. GAE grouping `(match_id, side)`, request order, per-game seed와 replay 동일성, slot-generation crossing 0건을 테스트한다.

### 8.3 평가

- 동일 policy digest/팀셋/seed 방식에서 old cohorts vs rolling slots, 512/1024/2048 slots, 각 2k×3 + 10k. RNG 실행 순서가 바뀌면 bitwise 동일 battle sequence가 아닐 수 있으므로 **정책/팀 표집과 전체 game semantics parity**, 통계 분포와 결과 replay를 별도 검증.
- 게임이 끝날 때마다 Python/Rust reset 호출이 너무 잦아지면 통신비가 증가할 수 있으므로 native vectorized reset queue로 묶는다.
- **성과 지표:** tail loss 감소, 평균 active batch 증가, actor games/s 증가, 오류 0. 증가가 없으면 original cohort path 유지.

---

## 9. G4 — 독립 코호트 비동기 파이프라인 (핵심 구조 전환)

### 9.1 목표와 dependency

현재 라운드는 대략 `Rust observe → H2D → model encode → prefix candidate/pick → D2H → Rust step`의 직렬 파이프라인이다. **하나의 게임 안에서는 전 단계 행동이 다음 legality를 결정**하지만, **다른 코호트**는 독립이므로 CPU 준비, GPU 계산, 결과 적용을 동시에 실행할 수 있다.

### 9.2 설계

- 먼저 **2코호트** ping-pong, 이후 3/4코호트 비교. 각 cohort는 고유 `match_ids`, CPU native handles, player-safe observation, packed candidate buffer, GPU staging slot, CUDA event, generation token을 가진다.
- Stage 1 Rust/PyO3 worker가 다음 cohort 관측 생성 및 후보 준비. Stage 2 H2D/encoder/scorer GPU 추론. Stage 3 D2H 결과를 기다리는 코호트만 처리하고 Rust step. 세 작업은 *서로 다른 코호트*에서만 중첩한다.
- 고정 batch보다 작은 코호트는 GPU 연산 처리량을 낮출 수 있으므로 `1×1024`, `2×512`, `4×256`을 **같은 총 live environments**로 비교한다. 동시에 2×1024, 4×512 등 추가 실험을 하되 RAM 예산을 지킨다.
- 데이터 공유는 queue + bounded staging으로 설계하고 Python GIL 경합을 측정한다. Native Rust worker가 GIL을 풀어도 Python bookkeeping이 직렬일 수 있으므로 필요 시 별도 producer/consumer process (모델 복제 및 interprocess copy 비용 포함)를 A/B.
- Pinned host slabs **전체 1GiB 이하**, each GPU NUMA-local; 2~3개 H2D/D2H staging buffer를 미리 만들고 재사용. CUDA compute stream과 copy stream 사이 이벤트 종속성 정확히 설정. CPU가 pinned buffer를 GPU copy 완료 전에 덮어쓰는 행위 금지.
- GPU를 코호트별로 여러 번 context 전환하거나 multi-stream을 남발해 overhead가 증가하지 않게 한다. 실제 GPU timeline에서 **동시 실행 구간**이 나오는지 확인한다.
- `torch.cuda.synchronize()`를 every round에 넣어 correctness를 만들지 않는다. 필요한 이벤트에서만 기다리며 매 결과의 buffer lifetime을 보증한다.

### 9.3 On-policy 및 correctness

- 각 rollout iteration의 weights/opponent policy는 고정. **업데이트되는 learner와 이전 정책으로 다음 iteration 수집을 겹치는 것은 기본 계획에서 금지**한다. 그 방식은 policy staleness와 on-policy PPO 계약 변경 가능성이 있으므로 별도 실험 승인 필요.
- CUDA stream과 Python thread 실행 순서는 바뀔 수 있어도 `(cohort,slot,generation,side,request_index,branch_level)` identity와 선택한 합법 행동이 서로 틀리지 않도록 trace를 생성.
- async path vs serial path의 prefix candidate bit parity, reward/GAE/terminal/seed replay, no leaks, no skipped matches 검증.

**G4 수락:** 실제 overlap 시간(fraction), CPU/GPU idle/queue, latency P50/P99, cache/memory, actor games/s와 dual total all-in. 단일 GPU actor가 개선돼도 실제 dual DDP learner와 경쟁해 전체 학습이 느려지면 해당 조합 폐기.

---

## 10. G5 — AI 추론 자체의 GPU 활용 및 후보 처리 최적화

### 10.1 Encoder와 scorer 별도 프로파일

- `PA3Model.encode()`의 6-layer Transformer, SDPA, embeddings, FFN와 `sample_level_step()`의 GRUCell, action embeddings, candidate logit/scatter, entropy, final sampling 각각 CUDA kernel/device time 및 Python launch time을 측정.
- Request별 `branch_count`에 따른 encoder reuse 1회가 실제로 지켜지는지, 같은 observation을 한 request에서 여러 번 encode하지 않는지 audit.
- Weighted model time에는 CPU wait가 포함될 수 있으므로 `torch.profiler`와 장시간 nvidia-smi dmon을 병행. 학습 구간 SM 69~80%를 **actor inference**의 SM utilization으로 대체하지 않는다.

### 10.2 개선 후보

1. 인코더 내 layout/contiguous 및 multi-op Tensor 변환 감소, `torch.stack`/`torch.arange`/mask fill 반복 작은 tensor 할당 제거. CUDA allocator와 CPU dispatch 절감.
2. Padding candidate를 한 번에 항상 64개 연산하는 dense path와 실제 candidate count에 맞춘 dynamic max/bucket 비교. **64 후보 capacity를 줄이거나 합법 후보를 잘라서는 안 된다.**
3. GPU 상주 action vocabulary lookup, entity/move token gather LUT, reusable device indices/table. In-place 변경은 autograd 및 checkpoint에 영향을 주지 않는 actor 경로부터 적용.
4. Explicit GPU `torch.multinomial`/categorical sampling vs 기존 method A/B, **probability temperature 1.0, generator seed, exact legal mask** 유지. NumPy/CPU sampling으로 다운그레이드 금지.
5. **GPU round-trip 횟수 줄이기**: prefix마다 CPU legality 호출이 필요한 현재 경로를 유지하되, GPU에서 batch output을 모아서 가져오기. Bulk D2H 이미 도입됨을 존중하고 실제 남은 왕복 수만 줄인다.
6. 한 player view에서 가능한 **조건부 후보 트리를 미리 생성**하는 고위험 PoC는 별도 분기. Branch 수/target branching의 조합 폭발과 숨겨진 정보 문제를 먼저 산정. 트리 생성 비용, memory, exact legal prefix oracle, GPU HBM 비교. 비용 대비 이득이 없으면 즉시 폐기.

### 10.3 CUDA Graph 및 정적 버킷

- 현재 시스템의 `torch.compile`/Inductor는 기본 비활성. **기본 운영 설정을 건드리지 않는 실험 분기**에서 eager static encoder 또는 shape bucket만 graph capture.
- Captured graph의 입력 shape/메모리 주소 고정, mask/seed/결과 buffer 재사용, prefix dynamic fallback 조건, output lifetime 검증. Capture마다 큰 고정 메모리를 요구하면 VRAM budget에 포함.
- 수치 평가 대상은 **real actor games/s**, GPU kernel launch overhead, warmup/capture cost를 포함한 steady throughput. Graph replay가 빨라도 전체 actor가 느리면 채택하지 않는다.

---

## 11. G6 — Rust 관측 생성과 복사/패킹 최적화

**실측 근거:** actor wall의 약 17%가 `native_observation`, 약 5%가 `parse_convert`, 10%가 `H2D`; `rust_step`은 1.6%밖에 안 된다.

- `engine/src/python.rs::observe_fixed_batch()`의 `Encoder::from_validated_dex(&batch.dex)` 반복 생성 비용을 따로 측정. Immutable Dex 기반 공용 encoder를 재사용할 경우 ownership / `Send + Sync` / thread safety 및 state independence 검증.
- 관측 결과를 매번 큰 `PyBytes`로 만드는 비용, Rust buffer 생성 및 복사, Python parse/NumPy view 생성, category+flags+floats dtype 확대, CPU→GPU 전송을 **Byte accounting**으로 분리.
- Rust source-of-truth 상태는 f32 및 정수 정확도 보존. **무손실 packed/SoA staging** 우선: category/flag/mask가 표현 가능한 가장 작은 wire dtype으로 이동하고 GPU에서 필요시 확장. 음수 sentinel과 known mask, source provenance가 같아야 한다.
- `float32→float16` 관측 전송은 **손실 압축**이다. 이것은 bit-identical input 변경이므로 v3 기본 최적화로 자동 채택 금지. 별도 schema bump, numeric tolerance, policy logprob gradient and rollout parity, 실제 점수 영향 시험 후 사용자 사양 승인.
- 다섯 ragged sidecar 정보가 wire에 보존되지만 모델이 직접 소비하지 않는 현 설계는 **별도 observation completeness** 과제. 최적화 명목으로 정보를 누락하거나 모델 feature 의미를 바꾸지 않는다.
- Rust `step_batch`에 많은 개발 시간을 우선 배정하지 않는다. 새 프로파일에서 1.6%보다 훨씬 커졌을 때만 재평가.

**G6 수락:** per-view vs fixed and ragged exact parity, hidden-information leakage test, native vs Python byte identity, thread-safety, 2k×3 +10k actor 및 all-in 개선.

---

## 12. G7 — CPU 메모리, rollout 및 후보 패킹: 작은 낭비까지 줄이기

### 12.1 Candidate wire / dtype

- 현재 `BranchCandidatesBatch.from_rows()`에서 padded `[B,4,64,6] int64` tensor를 생성한다. u8 packed wire를 유지하여 **CPU/H2D 바이트 수**를 줄이고 embedding gather 직전에 `int64`로 복원. Signed target index, `-1` entity/move token sentinel, selected/prefix masks에 주의.
- 동일 후보의 action_id/entity_token/move_token을 Python에서 중복 변환하거나 반복 정렬하는 작업을 제거할 수 있는지 profiling.
- 실제 padding occupancy와 row/K를 기록하고, 정확한 variable cap bucketing을 비교. **Legal 64 capacity를 26으로 줄이는 무단 변경 금지**.

### 12.2 Columnar buffer / overhead

- Streaming minibatch는 전체 iteration 물질화를 피했지만, 현재 `InlineObservationStore._rows`는 **row별 작은 NumPy record 목록**이다. 이를 NUMA-로컬 chunked SoA 또는 preallocated slab + offsets로 변경하는 PoC 수행.
- 96 token의 category, floats, known masks, flags, candidate records, metadata 각각 별도 column; append와 gather가 큰 연속 메모리에서 작동하도록 한다.
- Action source IDs, team/match provenance, policy_id, reward, done, old logprob의 원래 의미를 모두 보존한다. `GAE`와 `normalize_advantages`는 **전체 iteration**에 대해 계산하고 minibatch only materialization 유지.
- Memory allocation / peak RSS, Python GC pauses, memcpy, data creation per learner row, throughput을 측정. G7는 코드 복잡도가 크므로 수치 개선이 없다면 streaming의 현재 단순 구현 유지.

### 12.3 GPU에 불필요한 메타데이터를 보내지 않기

- `RolloutBatch.to()`는 match_ids, side, request_index, turn 등 metadata도 GPU로 이동한다. 학습의 실제 `_forward_terms()`가 요구하는 필드만 GPU 전송하는 `LearnerDeviceBatch` 경로 실험.
- Debug/replay/checkpoint 용 metadata는 CPU에 보존하며 timing과 same stats/gradient parity를 확인. Transfer 횟수/bytes 및 CUDA call count 기록.

---

## 13. G8 — Learner GPU 계산과 커널 극한 최적화

### 13.1 최신 1024 마이크로 기준 재측정부터

이전 M0 kernel trace는 microbatch 256 시점이다. 최신 1024에서는 단순히 동일한 kernel fraction을 적용하지 않는다. CPU timing과 CUDA GPU kernel timeline을 별도 계층으로 기록한다.

1. **Embedding-backward scatter**: `sum_and_scatter` dominant kernel의 인덱스 중복/패딩, sparse gradient 가능성 및 profiler evidence. Embedding semantics/optimizer state 변경 없이 kernel/gather 최적화 검증.
2. **SDPA**: V100에서 동작하는 실제 PyTorch backend와 `math` pin 사이 성능 비교. FlashAttention-2/BF16을 강제하지 않는다. Mask semantics, dropout 0, gradient parity 동일해야 한다.
3. **FFN GEMM / LayerNorm / elementwise**: Tensor Core 활용, contiguous layout, launch count, pointwise operations; 참조 그래프 유지. 새 라이브러리 설치 없는 built-in 가능성 우선.
4. **Optimizer Adam**: torch 내장 fused/foreach 옵션을 실제 버전/V100에서 제한 A/B. Adam betas/eps, zero weight decay, FP32 states, GradScaler skip, checkpoint resume 동등성 우선. GPU VRAM/throughput 개선이 없으면 롤백.
5. **CUDA Graph learner static microbatch**: VRAM과 고정 static IO를 많이 사용한다. Optimizer/GradScaler, partial micro, variance masking/optimizer state의 capture 안전성을 입증하는 전용 PoC가 선행돼야 한다. 실패하거나 graph setup cost가 크면 eager 유지.
6. Select, allocator, `index_select` vs slices, dynamic shape 및 Python dispatch를 profiler에 맞춰 최적화한다. `torch.compile`을 자동 켜거나 torch version을 올려서 문제를 해결하려 하지 않는다.

### 13.2 측정 규칙

- Pure forward/backward kernel time과 CPU wall이 겹칠 수 있으므로 합산하지 않는다. `torch.profiler`/CUDA event scope를 특정 stage에만 두고 overhead 있는 trace와 production run 구별.
- 실제 GPU SM util 69~80%는 단일 학습 GPU의 **이전 설정** 측정이다. 최신 1024/2048, DDP 2GPU에서 다시 측정한다.
- GPU 시간/row, optimizer steps/s, 4-epoch update seconds, all-in games/s, VRAM/RSS, overflow, KL, gradient relative error를 함께 기록.

---

## 14. G9 — Pinned staging, PCIe, NUMA와 멀티 GPU 조율

- MiniDC `GPU0 PCI 05:00.0 ↔ NUMA0`, `GPU1 PCI 84:00.0 ↔ NUMA1`. 각 rank가 **큰 Rust buffer와 pinned staging을 할당하기 전에** 해당 노드 CPU 및 메모리 affinity를 적용한다. `numactl`이 이미 있을 경우 프로젝트 프로세스에만 사용.
- Host pinned memory는 합계 1GiB 이하, 독립 buffers 재사용, `non_blocking=True` H2D를 실제 copy/compute overlap하도록 지정. `pin_memory()`를 매 iteration 반복 생성하면 오히려 느릴 수 있으므로 preallocate vs ephemeral 비교.
- `torch.cuda.Stream`의 current/copy stream과 `Event` 순서를 명시하고 buffer lifetime 및 D2H 결과 사용 이전 sync 보장. Record stream/ownership 필요 시 확인.
- PCIe 전송량, pageable→pinned 비율, GPU host memory controller, NCCL SHM/host-staged 통신과 transfer timeline 계측. `GPU1` DDP 통신 동안 CPU memory bandwidth contention 가능성 점검.
- 데이터가 RAM에서 이동하고 disk reads/swap=0이라는 현 실측을 존중한다. SSD/NVMe 교체는 학습 가속 실험의 우선 작업에서 제외. 장기 checkpoint/fsync log/spill이 병목인 경우 별도 케이스로만 승격.
- Host NUMA 메모리 이동이나 affinity 설정으로 service interference가 발생하지 않도록 rank별 CPU 제한, RSS, GPU telemetry와 system load 확인.

---

## 15. G10 — GAE와 CPU 내부 소규모 튜닝: 후순위, 측정 시에만

- 기존 M0에서 GAE는 2k 학습 58.3초 중 약 1.26초, learner의 약 2%. GPU/actor의 큰 작업을 뒤로 미루고 GAE부터 재작성하지 않는다.
- G0/G1/G4 후 GAE가 실제 병목으로 승격될 경우 `(match_id,side)` side trajectories를 array-based 순차 계산. 정확한 `gamma`, lambda, terminal/bootstrapped semantics 및 sorted request order 동일성 테스트.
- Rust memory pool, CPU loops SIMD, vec packing, branchless kernel 등을 무턱대고 구현하지 않는다. 실측된 hotspot에서만 채택한다.

---

## 16. G11 — 측정/실행 및 성능 승격 게이트

### 16.1 세 층의 성능 수치

1. `actor_only_real_policy_games_per_s`: 실제 PA3-8M sampling, frozen team, 자연완결 게임, 0 운영 에러. PPO 전파/학습 제외.
2. `bounded_ppo_all_in_committed_games_per_s`: 위 게임 수집 + GAE + 준비 + 실제 4 PPO epochs + optimizer/GradScaler + 지정 checkpoint fsync/rename. `recompute_check`가 측정 외이면 반드시 별도 기록.
3. `steady_state_operational_games_per_s`: 전체 실행 과정(DDP sync, 정책 snapshot, metrics, checkpoint/resume, 지정된 evaluation)을 일정 운영 창에서 측정. **100M 본학습을 시작하지 않고도** 제한된 반복 운영 smoke로 시험한다.

### 16.2 실험 매트릭스

| 실험 | 기본 자료량 | 반복 | 통제 변수 |
|---|---|---:|---|
| Unit kernel/GAE/DDP micro | 고정 같은 텐서 | 10+ warm repeats | data/weight/shape 동일 |
| Actor-only single rank | 2,048 naturally complete | **3** | seed panel, model SHA, team manifest, env/workers |
| Actor-only long | 10,240 naturally complete | 1+, 후보 승격 시 3 | tail fraction, ops, RSS |
| PPO single candidate | 2,048 naturally complete | **3** | global batch 4096, PPO4, same seed panel |
| PPO single long | 10,240 naturally complete | 1+, 최종 승격 시 3 | checkpoint 포함 |
| Dual DDP all-in | 합산 2k 이상 시작 후 10k+ | 3 | 동일 global samples, optimizer steps, NCCL |
| Async overlap/rolling | 동일 완료 games | 3 + long | in-flight counts, seed distribution, replay |
| Final frontier smoke | 2+ bounded iterations | 필요시 3 | checkpoint+resume, metrics, eval defined window |

- 실제 measurement 보고서에 exact command, `git SHA`, `git_dirty + diff hash`, frozen dataset manifest SHA, model SHA, per-rank GPU PCI/NUMA, driver/torch stack, power, seed, total matches/decisions, game length tail, errors, compiler env, peak RSS/VRAM 및 timing 포함.
- **비교 가능한 반복**에서 median/min/max와 각 repeat wall을 사용. 다른 반복의 model/seed를 임의로 혼합해 개선 비율을 계산하지 않는다.
- 개선 정도가 run-to-run noise와 비슷하면 `inconclusive` 또는 `neutral`로 기록. 코드 복잡도, VRAM 사용량, 재현성 위험까지 고려해 채택 여부 결정.

### 16.3 Full correctness gate (실험마다)

- 0 operational errors, 0 illegal action, exact packed legal candidate parity for every prefix, no hidden info leak.
- Complete natural game accounting: wins + losses + draws = completed matches, no early truncation/timeout masquerading as draw, all launched games drained at iteration boundary.
- Source RNG, requested team pair distribution/seat balance, request indices, actor vs opponent trajectory policy filtering exactly correct.
- Stratified frozen sampled vs learner recomputed joint logprob with FP16 <= 1e-3 gate and FP32 <=1e-4 (existing declared gates), exact GAE/advantage normalization, FP32 critical loss.
- Padding and uneven actor/value rows, 1 GPU global vs 2 GPU DDP gradients/optimizer step parity, GradScaler skip consensus, KL early stop, preserved 4 epochs.
- Checkpoint atomic write/reload optimizer/scaler/RNG/cursor/policy digest. No missing observations or ragged drop, no real-match duplication.
- Budget: GPU <=28GiB soft, total host <=48GiB, rollout <=24GiB, pinned <=1GiB, swap=0 preferred; if overshoot or CPU/GPU conflict, abort that experiment safely.

### 16.4 Performance stop/branch conditions

- **GPU OOM** 또는 28GiB soft budget 초과: 즉시 해당 microbatch/graph 실험 폐기하고 이전 안정값으로 복원. OOM aftereffects/caching reset 및 active process 보호.
- **DDP deadlock/gradient mismatch/NaN:** 아직 결과를 속도 성과로 발표하지 말고 최소 재현 테스트로 돌아간다.
- **Concurrency hazard:** negative counts, overwritten staging buffer, seed mismatch, late result to reused slot, illegal action 또는 lost terminal reward 발생 시 병렬 path 롤백.
- **Improvement absent:** 새로운 backend가 기존 성능보다 느리거나 noise 이내이며 구조 비용이 크면 폐기. 명확한 원인/trace만 보존.
- **지속 최적화:** 달성 숫자(예: 100/500/1000)를 종료 조건으로 삼지 않는다. 반복 평가한 모든 고우선순위 실험이 noise 이내이고 더 이상의 개선 예상 이익이 구현 위험보다 낮을 때 plateau 보고.

---

## 17. 단계별 파일별 작업 지시와 산출물

| Package | 핵심 대상 코드 | 반드시 추가할 문서/검사 |
|---|---|---|
| G0 | `agent/ppo/learner.py`, `agent/train/native_collector.py`, `tests/agent/` | `V3_NUMERIC_PARITY.md`, `test_real_gradient_parity.py`, `test_streaming_equivalence.py`, actor collection contract audit |
| G1 | `agent/ppo/learner.py`, `agent/model/pa3_model.py`, 신규 dual trainer/launcher | `V3_DDP_GLOBAL_GRADIENT.md`, NCCL logs, 1GPU/2GPU parity JSON, 2GPU all-in JSON |
| G2 | `scripts/bench_pa3_end_to_end.py`, PPO config/launcher | `V3_VRAM_BATCH_SWEEP.md`, micro 1024/2048/4096 controlled raw JSON |
| G3 | `agent/train/native_collector.py`, engine PyO3 Rust reset bridge | `V3_ROLLING_SLOTS.md`, long-tail histogram, invariants/test |
| G4 | collector/dispatcher, reusable pinned slabs / CUDA event wrappers | `V3_ASYNC_COHORTS.md`, CPU/GPU timeline, 1/2/4 cohorts A/B |
| G5 | `agent/model/pa3_model.py`, scorer/candidate adapter | `V3_INFERENCE_FRONTIER.md`, batch histogram, CUDA kernel analysis, static bucket A/B |
| G6 | `engine/src/python.rs`, `engine/src/observation.rs`, `engine/python/pa3_engine/observation.py` | `V3_OBSERVATION_PERF.md`, schema golden parity, hidden-info audit |
| G7 | `agent/types/requests.py`, `agent/buffer/rollout_buffer.py`, `agent/types/observation.py` | `V3_CANDIDATE_MEMORY.md`, dtype proof, columnar rows, memory accounting |
| G8 | learner GPU kernels, optimizer, controlled CUDA Graph PoC | `V3_KERNEL_FRONTIER.md`, trace summary, gradient proof |
| G9 | NUMA launch wrappers, memory staging | `V3_PCIe_NUMA.md`, transfer timeline, swap/disk counters |
| G11 | benchmark scripts, `docs/perf/FINAL_REPORT.md`, `runs/perf/` | `V3_FINAL_REPORT.md`, final run matrix and manifests, clean SHA, next frontier |

**전달 원칙:** 작업마다 하나의 기능/수학 패치와 검증을 별도 커밋한다. raw JSON 이름은 변경사항/seed/SHA가 추적되게 만든다. 가능한 경우 기존 도구와 파일 포맷을 재사용하고, 새 테스트는 기존 `tests/agent`로 추가한다. 구현 중 사용자 승인 없이 게임 규칙이나 고정 팀풀을 변경하지 않는다.

---

## 18. 즉시 실행 가능한 명령 초안

> 실행 전 `python scripts/bench_pa3_end_to_end.py --help`, 현재 venv, checkout, GPU availability를 확인한다. 이 명령은 **짧은 제한 벤치마크**이며 100M 본학습을 시작하지 않는다. 아래 옵션은 `cba9dfa` 시점 CLI에 존재하는 것으로 GitHub 코드에서 확인했다.

```bash
# Verify state and protect existing work
git fetch origin
git status --short
git rev-parse HEAD
git branch --show-current
nvidia-smi --query-gpu=index,pci.bus_id,memory.used,memory.total,power.draw --format=csv

# Existing regression suite
PYTHONPATH=engine/python:. .venv/bin/python -m pytest tests/agent -q

# Single-GPU current winning baseline: 2k x 3, 1024 microbatch, streaming
PYTHONPATH=engine/python:. .venv/bin/python scripts/bench_pa3_end_to_end.py \
  --games 2048 --envs 1024 --workers 16 --repeats 3 \
  --mode full --observations fixed --precision fp16 --inference-mode \
  --candidate-wire packed --device cuda:0 --microbatch 1024 \
  --streaming-minibatch --checkpoint runs/perf/v3_baseline_ckpt.pt \
  --report runs/perf/v3_baseline_1024.json --tag v3-baseline

# Bounded microbatch 2048 A/B; must observe VRAM/RSS and validate gradient parity first
PYTHONPATH=engine/python:. .venv/bin/python scripts/bench_pa3_end_to_end.py \
  --games 2048 --envs 1024 --workers 16 --repeats 3 \
  --mode full --observations fixed --precision fp16 --inference-mode \
  --candidate-wire packed --device cuda:0 --microbatch 2048 \
  --streaming-minibatch --checkpoint runs/perf/v3_micro2048_ckpt.pt \
  --report runs/perf/v3_micro2048.json --tag v3-micro2048

# 4096 is NOT a default: only run if 2048 profiling makes 28 GiB safe.
# Dual DDP, rolling-slot, async commands are to be added and recorded only
# after their implementation and --help are verified. Do not invent flags.

# Independent full-engine readiness (currently nonzero expected)
bash scripts/cargo.sh run --locked --release --example readiness_check
```

**주의:** 현재 `--checkpoint` 파일은 각 repeat마다 덮어쓸 수 있다. 벤치마크 전용 경로만 사용하고 기존 사용자 checkpoint 또는 다른 실험의 파일을 지정하지 않는다. 새로운 기능 플래그는 `--help`/실제 코드에 추가된 뒤 사용한다.

---

## 19. 운영 수준으로 나가기 전 별도 준비 사항

### 19.1 규칙 준비

`engine/TRAINING_READINESS.md`에서 **FAIL 2, 3, 5, 8, 15, 16**을 전부 PASS로 수정하는 작업은 별도 workstream이다. 규정 전체 legal moves 515, abilities 223, dynamic closure, corpus 오류 0, full coverage throughput, Showdown reference와 interaction/differential 검증을 절대 생략하지 않는다.

### 19.2 AI 입력 정보

`docs/perf/OBSERVATION_CONTRACT.md`의 다섯 ragged 섹션은 전송 parity와 모델 consumption을 구분한다. 현재 어떤 정보가 모델에게 실질적으로 전달되지 않는지 feature inventory를 감사하되, **성능 최적화 브랜치에서 모델 구조를 몰래 수정하지 않는다.** 필요시 사용자 사양 승인 후 별도 train-policy 브랜치.

### 19.3 Sample efficiency

별도 연구 문서 `docs/PokeAgent3_RL_Sample_Efficiency_Research_2026-10-08.md`를 유지한다. 강함을 확인할 고정 상대, seed, seat-balanced 평가, payoff matrix, sample efficiency curve는 속도 최적화와 병행 준비하지만 PFSP/PPG/reward shaping 및 학습 팀 가중치를 본 v3 작업에 혼입하지 않는다.

---

## 20. 최종 수락 조건과 종료 보고서 템플릿

**필수 완료:**

- [ ] G0 actual PA3 model gradient 및 streaming, actor collection contract parity 통과.
- [ ] DDP 2GPU에 대한 forward 경로/no_sync/global actor-value denominator correctness 입증.
- [ ] Per-GPU VRAM, NUMA, NCCL 동기화, 2 GPU 실제 4-epoch PPO all-in 성능 측정.
- [ ] Microbatch 1024 vs 2048, 조건부 4096 GPU/host memory, compute time A/B와 3-repeat 승격 기준.
- [ ] 현재 actor의 in-flight/request-batch/tail histogram과 Rolling Slots PoC, 실제 gains 또는 반증.
- [ ] 1/2/4 코호트 overlap timeline 및 GPU/CPU idle 분리, all-in A/B.
- [ ] 최신 learner/actor의 CUDA kernel, observer wire, byte copies, candidate table dtype/allocations 재프로파일.
- [ ] Rust Encoder reuse, packed/SoA, GPU resident working set, pinned staging의 채택/폐기 보고.
- [ ] `steady_state_operational` 범위 명시, checkpoint/resume + metrics + eval overhead 기록.
- [ ] 각 실험별 raw JSON, manifest, repeat median, min/max, CPU/GPU time, RSS/VRAM, 0 operational errors, parity 결과, git SHA.
- [ ] 기존 서비스, GPU power, driver, torch, CUDA, 팀풀과 모델 구조, PPO 및 관측 의미 보호.
- [ ] Full engine 16/16 readiness는 별개이며 미완료면 **100M 본학습 금지**를 유지.

**최종 보고서 제안 형식:**

```
HEAD / branch / clean: ...
Dataset SHA / model SHA / run seeds / software & GPU inventory: ...
Accepted / rejected / unmeasured experiments: ...
Single GPU actor games/s (3 repeats; long): ...
Dual GPU actor games/s (wall aligned): ...
Single GPU 4epoch PPO + checkpoint games/s: ...
Dual GPU 4epoch PPO + DDP sync + checkpoint games/s: ...
Steady-state all-in including metrics/eval (if measured): ...
Actual GPU kernel time / CPU orchestration / H2D-D2H / NCCL overhead: ...
Peak RSS per rank / HBM allocated and reserved / pinned memory / swap: ...
Natural completed games / learner decisions / PPO optimizer steps: ...
PPO parity / masks / GAE / KL / checkpoints / engine readiness: ...
Highest-priority unexploited bottleneck and next exact experiment: ...
```

---

## 21. 반드시 확인할 출처

### 21.1 PokeAgent 소스 및 실측

- [v2 learner breakdown](https://github.com/mwl313/PokeAgent3.0/blob/cba9dfa/docs/perf/V2_LEARNER_BREAKDOWN.md)
- [v2 frontier](https://github.com/mwl313/PokeAgent3.0/blob/cba9dfa/docs/perf/V2_FRONTIER_REPORT.md)
- [v1.1 real actor frontier](https://github.com/mwl313/PokeAgent3.0/blob/cba9dfa/docs/perf/ULTIMATE_FRONTIER_REPORT.md)
- [v2 correctness audit](https://github.com/mwl313/PokeAgent3.0/blob/cba9dfa/docs/perf/V2_CORRECTNESS_AUDIT.md)
- [v2 IO NUMA](https://github.com/mwl313/PokeAgent3.0/blob/cba9dfa/docs/perf/V2_IO_NUMA_REPORT.md)
- [10k current benchmark JSON](https://github.com/mwl313/PokeAgent3.0/blob/cba9dfa/runs/perf/v2_stream_10k.json)
- [microbatch 1024 x3 JSON](https://github.com/mwl313/PokeAgent3.0/blob/cba9dfa/runs/perf/v2_micro_1024_x3.json)
- [native collector](https://github.com/mwl313/PokeAgent3.0/blob/cba9dfa/agent/train/native_collector.py)
- [PPO learner](https://github.com/mwl313/PokeAgent3.0/blob/cba9dfa/agent/ppo/learner.py)
- [rollout buffer](https://github.com/mwl313/PokeAgent3.0/blob/cba9dfa/agent/buffer/rollout_buffer.py)
- [full spec 1.1](https://github.com/mwl313/PokeAgent3.0/blob/cba9dfa/docs/spec/fullspec-1.1-minidc-20261006/POKEAGENT_3_0_FULL_SPEC.md)

### 21.2 공식 GPU / PyTorch 동작 근거

- [PyTorch DDP `no_sync` 공식 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html): forward까지 context 안에 있어야 동기화가 생략됨.
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda): pinned buffers와 `non_blocking=True` 관련 제약.
- [PyTorch pin-memory and non-blocking guide](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html): 매번 `pin_memory()`하면 오히려 느려질 수 있고 async D2H는 충분한 sync가 필요함.
- [NVIDIA CUDA Best Practices Guide (CUDA 12.4)](https://docs.nvidia.com/cuda/archive/12.4.1/pdf/CUDA_C_Best_Practices_Guide.pdf): compute/transfer overlap은 서로 독립적인 작업, pinned host 메모리, 적절한 streams가 필요.

---

## 22. 에이전트에게 바로 전달할 실행 프롬프트

```
You are the PokeAgent3.0 optimization implementation agent. Read this entire
"Extreme Throughput Optimization Master Plan v3.0" including its safety,
numerical correctness, source, benchmark and acceptance sections.

1. Fetch latest origin/optimization/pa3-realpolicy-throughput. The last
   verified baseline was cba9dfa; reconcile any newer user or agent commits.
   Preserve existing work and never overwrite raw measurements or checkpoints.
2. Continue from the completed v1.1/v2.0 work. Do not redo proven fast paths.
3. First close G0: real PA3-8M gradient parity, streaming-vs-non-streaming
   parity and the current-vs-opponent seat collection contract. Reconcile
   the contradictory reported 2,112 micro-sync count for the 2k run.
4. Prioritize G1 dual V100 actual DDP all-in. Ensure the model's true
   DDP.forward() path is used, the no_sync context encloses forward/backward,
   and policy actor/value terms are normalized by DIFFERENT global valid-row
   denominators across ranks. Compare gradients and one-step optimizer updates
   against single-GPU reference before any speed claim.
5. Run G2 controlled VRAM sweeps (1024 vs 2048; 4096 only if safely feasible),
   maintaining global minibatch 4096 and 28 GiB per-card soft ceiling. Compare
   actual throughput, not just GPU memory utilization.
6. Implement and test G3 Rolling Slots and G4 independent-cohort async CPU/GPU
   overlap with slot generations, frozen policy, RNG correctness, natural drain,
   fixed workload and profiler-verified concurrency. Do not stale the PPO policy.
7. Attack remaining measured CPU/GPU bottlenecks using G5-G10: encoder and
   scorer GPU batch/kernel profiling, packed candidate dtype and bucket,
   Rust observation encoder reuse, chunked columnar slabs, NUMA-local pinned
   reuse, H2D overlap, static CUDA Graph PoCs and safe optimizer alternatives.
   Isolate one hypothesis at a time and reject neutral or unsafe changes.
8. For each accepted change: run correctness gates, comparable x3 2k trials
   and 10k+ confirmation, checkpoint/RNG/GAE/masks/logprob parity, all-in
   (not just actor-only) measurement, save raw JSON and provenance, commit
   and push an atomic validated patch.
9. Preserve team dataset/model/game mechanics/PPO contract and the miniDC
   torch/CUDA/driver/power/services. Do not install packages, change host
   settings, reduce legality, hide failures or train 100M matches. Rule
   readiness 16/16 and explicit user authorization remain mandatory.
10. Deliver a measured final frontier report (single/dual actor, single/dual
    full PPO, steady-state if measured), every accepted/rejected experiment,
    source SHA, memory/NUMA/NCCL timings, known limits, exact next blockers.

Do not stop at any arbitrary games/s number. Continue measured improvements
until the main remaining opportunities are disproved, unsafe or not economical.
```

**문서 집행 상태:** v3.0은 새 최적화 계획이며, 이 문서 작성 자체가 miniDC 코드 변경, DDP 구축 또는 추가 벤치마크 실행을 의미하지 않는다.
