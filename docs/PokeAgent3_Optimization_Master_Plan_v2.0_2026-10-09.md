# PokeAgent 3.0 — Extreme Throughput Optimization Master Plan v2.0

**문서 유형:** miniDC 에이전트 실행 지시서 및 기술 검증 계획  
**작성 기준:** 2026-10-09, Asia/Seoul  
**대상 프로젝트:** `mwl313/PokeAgent3.0`  
**검증한 기준점:** `optimization/pa3-realpolicy-throughput` @ `b72f5f4c73dc8cbdf06584a5e0ed9c0ea7f6b8d8`  
**상위 계약:** `docs/spec/fullspec-1.1-minidc-20261006/POKEAGENT_3_0_FULL_SPEC.md`  
**상속 문서:** `docs/PokeAgent3_Ultimate_Throughput_Optimization_Plan_v1.1.md`  
**보완 연구:** `docs/PokeAgent3_RL_Sample_Efficiency_Research_2026-10-08.md`  
**상태:** 새 계획, 이 문서의 작업은 아직 수행되지 않음. Phase 0–6의 완료 표시는 이전 에이전트 보고서에 한정된다.

> **최상위 임무:** 현재의 **Rust CPU 배틀 엔진 + PyTorch GPU PA3-8M + PPO** 구조를 유지한다. CPU, RAM, PCIe, 듀얼 V100의 모든 불필요한 오버헤드를 실측 기반으로 제거한다. *자연완결 경기 생성 속도*와 *실제 PPO 업데이트를 완료하는 속도*를 별도 계측하고, 둘 다 측정 가능한 성능 한계까지 높인다. 1,000 games/s는 종료 목표가 아닌 도전 이정표다. 모델 구조, 전투 규칙, 관측 정보, 합법 행동, on-policy PPO 의미, 팀 전체 범위를 속도와 교환하지 않는다.

---

## 0. 에이전트가 가장 먼저 읽어야 할 결정 사항

1. **v1.1을 중단하거나 초기화하지 않는다.** `b72f5f4` 이후 새 작업 브랜치를 생성하거나 기존 브랜치의 새 커밋으로 진행하되, 기존 모든 성과 및 JSON, manifest, checkpoint를 보존한다. 최신 origin 상태를 우선 확인한다.
2. **우선순위 최상위는 학습 통계 정확성과 learner 290.7초의 분해다.** 보고된 PPO KL은 현재 `PPOLearner.update()`에서 microbatch 전체 집계가 아닌 마지막 microbatch 통계를 사용하는 구현을 반드시 감사해야 한다. 지표 수학과 loss/gradient 수학은 구분해 교정한다.
3. **학습 속도의 진짜 기준은 all-in committed games/s다.** 현 `26.43 games/s`는 수집 + GAE + PPO update의 bounded 측정으로, 모델 평가, 실 checkpoint write, 완전한 서비스 운영비, 사전 recompute 검증 시간이 전부 포함된 숫자는 아니다. 지표명을 정확하게 사용한다.
4. **스토리지는 우선 병목이 아니다.** 현재 rollout과 모델 계산은 RAM, GPU VRAM에서 이루어진다. SATA↔NVMe 업그레이드는 `swap`, page fault, actual I/O wait, checkpoint 및 spill 지연이 실측으로 의미 있게 확인될 때만 학습 가속 과제로 승격한다.
5. **Full Regulation readiness 10/16 FAIL을 우회하지 않는다.** 빠르다는 이유로 본학습 1억 경기를 시작하지 않는다. 성능 트랙과 규칙 완성 트랙은 별도 진척을 기록한다.
6. **작은 개선이라도 재현성과 정확성을 입증한다.** 비교 가능한 3회 반복, 동일 게임 조건, 각 단계 버전 및 실제 실행 로그를 남긴다. 최종 속도 숫자의 출처를 꾸미지 않는다.

### 우선순위 및 코드 위험 분류

| 등급 | 목적 | 예시 | 채택 조건 |
|---|---|---|---|
| **C0: Correctness** | 먼저 잘못된 측정 및 PPO 계약 수정 | microbatch 통계 집계, padded gradient/weighting 검증, LR clock/종료 보상 | 참조 구현 대비 정의가 맞고 새 regression 통과 |
| **M0: Measurement** | 290.7초의 실제 원인 특정 | GAE/pack/forward/backward/optimizer/복사/동기화 stage | CPU wall/CUDA kernel 시간 분리, 측정 오버헤드 표시 |
| **P0: Performance** | 가장 큰 데이터 경로 개선 | GAE 연속 배열, columnar rollout, streaming minibatch | all-in 성능 증가, full gradient 및 logprob parity |
| **P1: Parallelism** | 두 V100과 40 CPU cores 활용 | DDP `no_sync`, NUMA-local staging, 비동기 코호트 | dual all-in 증가, 정책 버전 동결, memory cap 준수 |
| **P2: Frontier** | 남은 계산/전송 극한 튜닝 | Rust encoder cache, CUDA Graph PoC, static buckets | 실제 관련 hot path일 때만, A/B 실측 및 rollback |
| **R: Separate Research** | 적은 RL로 더 강하게 | 평가, PFSP/PPG 비교 | 본래 PPO 학습 설정 무단 변경 불가 |

---

## 1. 변경 불가 계약과 하드웨어 핀

### 1.1 시스템

- CPU: Xeon E5-2673 v4 ×2, **40 physical cores / 80 threads**, **2 NUMA nodes**.
- GPU0: Tesla V100 PCIe 32GiB, PCI **05:00.0**, NUMA0, **175W**.
- GPU1: Tesla V100 PCIe 32GiB, PCI **84:00.0**, NUMA1, **150W**.
- GPU P2P/NVLink 직결 없음. `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=0`를 프로젝트 프로세스에만 지정한다.
- RAM 약 62GiB 실측 보고. 전체 프로세스 RAM soft budget **48GiB**, rollout **24GiB**, pinned staging 전체 **1GiB**. GPU별 VRAM soft budget **28GiB**.
- Python 3.12.3, PyTorch 2.14.0+cu126, runtime CUDA 12.6, system toolkit CUDA 12.8.2, driver 580.178.04, Volta `sm_70`. 패키지/드라이버/CUDA/전력/BIOS/서비스 자동 변경 금지.
- `dsh-web`, `llama-swap` 및 다른 서비스의 변경/중지 금지. 사용자가 승인한 GPU 사용 시간과 장비 상태만 활용.

### 1.2 불변 ML 계약

- PA3-8M 구조와 96 padded entity token(기존 88 active)을 유지한다. Transformer depth/width, 후보 공간, model vocab 의미를 임의 변경하지 않는다.
- 1,137개 frozen team, `mb-mc-v3-userteam-all-train`. 매치업 전량과 분포를 유지한다. 사용자가 확인한 기본 Charizard Ability는 **Blaze**, Mega Y는 **Drought**. Frozen raw/manifest 임의 변경 금지.
- 정확한 M-C 배틀 규칙, prefix-dependent legal candidates, 모든 합법 target, player-safe 관측, no hidden state leak.
- `gamma=1`, `GAE lambda=.95`, terminal-only reward, PPO **4 epochs**, global minibatch **4096**, microbatch **256**, `clip=.2`, entropy `.01`, uniform reverse-KL `.001`, target KL `.03`, FP32 probability and loss, FP16 autocast plus GradScaler.
- Learner 적어도 131,072 completed current-policy decisions 및 2,048 natural matches/iteration; 실제 overshoot는 유지하고 중복 집계하지 않는다. Epoch 도중 정책/상대 checkpoint를 교체하지 않는다.
- 기존 optimizer, scheduler, old-policy logprob semantics를 유지한다. 불가피한 수정은 동일 수학임을 수치 검증하고 기록한다.
- 100M training 및 새로운 sample-efficiency 알고리즘 변경은 별도 승인과 full readiness 조건이 필요하다. 본 문서는 성능 최적화와 제한된 smoke 및 profiler만 승인한다.

---

## 2. GitHub 실측 정본과 검증 수준

### 2.1 실제 확인된 성능

| 측정 | 실제 수치 | 비교/해석 |
|---|---:|---|
| Old single GPU real-policy actor median | **15.07 games/s** | 2,048 games, 3 repeats, GPU0, fp32 |
| New single GPU real-policy actor median | **103.44 games/s** | **6.86x**; 3 repeats의 중앙값 |
| Optimized GPU0 actor 10,240 games | **105.78 games/s**, collect 96.803s | FP16 + packed+fastrows |
| Old measured full PPO bounded iteration | **20.87 games/s**, 490.634s | typed learner build |
| New measured full PPO bounded iteration | **26.43 games/s**, 387.490s | packed `from_rows`, 10,240 matches |
| New learner phase | **290.686s** | 387.490s 중 **75.0%**, GAE/prepare/update |
| New end-to-end peak RSS | **18.05GiB** | 구형 25.21GiB; 24GiB rollout-budget과 혼동 금지 |
| Dual real-policy actors | **162.733 games/s**, 4,096 games, 25.170s launcher wall | GPU0 94.15 + GPU1 100.26 individual rates. DDP learner 아님 |
| Last `readiness_check` | **10/16 PASS** | FAIL: 2,3,5,8,15,16 |

**중앙값 표시 원칙:** `103.44 games/s`에 대응하는 반복의 wall은 **19.798s**이다. 보고서의 21.4s는 다른 반복(95.59 games/s)의 wall이다. 앞으로 하나의 표에 중앙값과 특정 반복 wall을 혼합해도 각 통계의 출처를 명시한다.

**원본 파일:** `docs/perf/BASELINE.md`, `docs/perf/OBSERVATION_CONTRACT.md`, `docs/perf/CANDIDATE_WIRE.md`, `docs/perf/ULTIMATE_FRONTIER_REPORT.md`, `docs/perf/FINAL_REPORT.md`, `runs/perf/*.json`, `runs/perf/run_manifest.json`.

### 2.2 측정상 제한 사항 및 결론 유보

- 단일 GPU all-in 결과는 **1회** bounded 10,240-game 실행. 3회 반복된 것은 actor-only 단계 일부다.
- 기존 `gpu_event_ms`는 전체 구간을 감싸는 이벤트이며 GPU kernel-only busy time 또는 GPU occupancy를 나타내지 않는다. `nvidia-smi` inventory의 GPU utilization 0은 실행 전 상태다.
- `scripts/bench_pa3_end_to_end.py`는 `all_in_wall_s = collect_wall + ppo_wall`로 계산한다. 사전 `recompute_check`, 실제 checkpoint write, evaluation, 장기 프로세스 운영비를 제외한다. 이 값을 무조건 production `steady_state_all_in`으로 명명하면 안 된다.
- 10k optimized JSON의 `git_dirty=true`: 새 HEAD에서 clean-tree 재측정하여 provenance를 강화한다.
- sampled-vs-recomputed logprob diff 0.0은 **최대 1,024개 초기 행**에 대한 `recompute_check` 근거다. 분기/행 전체를 전수 검증한 수치로 확대 해석하지 않는다.
- `_forward`가 반환한 손실/상태와 optimizer report의 최종 통계값은 집계 방식 때문에 별개로 검증해야 한다.
- `OBSERVATION_CONTRACT.md`에서 다섯 종류 ragged 데이터가 wire에서 보존되지만 현재 신경망에는 직접 소비되지 않는다. 고정 필드에 정보가 중복됐는지를 확인해야 한다. 본 최적화 중 몰래 ragged 데이터를 버리거나 모델 구조를 바꾸지 않는다.

### 2.3 병목: 확인된 것과 가설인 것

| 항목 | 근거 | 판정 | 조치 |
|---|---|---|---|
| Learner 290.7s | 10k JSON | **측정 확정** | CPU prepare, GPU backward/optimizer 등 내부 항목 분해 필요 |
| Full iteration 배치 생성 | `PPOLearner.prepare_batch -> RolloutBuffer.to_batch` | **코드 확정** | streaming 설계, 실제 비용 계측 |
| GAE per-trajectory small CPU tensors | `agent/ppo/gae.py` | **코드 확정, 시간 미계측** | isolated profiler 후 연속 배열로 리팩터 |
| `update()` report 통계가 마지막 micro만 반영 | `agent/ppo/learner.py`, `stats` loop | **코드상 결함 강하게 확인** | weighted epoch statistics + KL criterion 수학 확인 |
| CPU→GPU 전체 metadata 이동 | `RolloutBatch.to()` | **코드 확정, 시간 미계측** | forward-only columns staging |
| Packed action 후보를 저장했다가 `int64` dense table 재구성 | `BranchCandidatesBatch.from_rows` | **코드 확정** | minimal dtype/column buffer 유지, 연산 직전 안전 cast |
| Fixed observation 생성 17% actor | 2k stage JSON | **측정 확정** | pack/encode/byte-copy 내부 분해 |
| 1 rank model ~24.5% actor | 2k stage JSON | **측정 확정된 호출 측정, 순수 kernel time 아님** | profiler CPU/CUDA correlation |
| SSD/NVMe I/O 병목 | 보고서에 디스크/스왑 profile 없음 | **미확인, 낮은 사전 가능성** | `/proc/PID/io`, `vmstat`, optional `iostat` 실측 |
| GPU1 느림 / NUMA 포화 | 듀얼 ranks 94.15/100.26 | **보편적 결론 불가** | 장시간 rank별 utilization/launch overhead 측정 |

---

## 3. 실행 규칙: baseline, 테스트, 롤백, 증거

### 3.1 변경 작업의 최소 단위

각 phase에서 **프로파일링 기록 → 독립 최적화/버그 패치 → 참조와 parity → 같은 조건 A/B → 3회 반복 → 대표 JSON/manifest → commit → 다음 병목** 순서로 진행한다. 원인 분리가 필요한 서로 다른 최적화를 한 benchmark에 한꺼번에 섞지 않는다. 서로 충돌하지 않는 파일은 개발을 병렬로 수행할 수 있지만 통합 성능은 순서대로 검사한다.

### 3.2 검증 명명

- `actor_real_policy_games_per_s`: 정상 종료 수집만. PPO 업데이트 제외.
- `bounded_ppo_games_per_s`: 자연완결 match / (collect + GAE + learner update 등 명시적으로 포함한 wall).
- `steady_state_all_in_committed_games_per_s`: 전처리, 수집, 학습, 모델 sync, checkpoint, metrics, 필요한 평가를 포함하는 명확한 운영 창의 완료 경기 수 / elapsed wall.
- `GPU_model_kernel_time_ms`: NVIDIA/PyTorch profiler가 실제 GPU kernel에서 측정한 합리적 집계값. CUDA 이벤트로 전체 벽시계를 잰 값과 혼동 금지.
- `learner_rows_per_s`, `actor_requests_per_s`, `policy_decisions_per_s`, `turns_per_game`: 속도가 샘플 특성 왜곡으로 얻어진 것인지 검증.
- `sample_efficiency`: 동일 훈련 경기 수에서 공통 고정 평가 패널의 실력. 이 문서의 속도 지표와 분리.

### 3.3 공통 검증 게이트

1. Same hardware pin, dataset manifest SHA, branch SHA, seed panel, GPU/NUMA topology, team sampling; 0 operational errors, 0 illegal actions, match counts exact.
2. Observation fixed/per-view parity, five ragged sections transport, legality oracle exact every sampled prefix; no capacity overflow/truncation.
3. FP16 actor sample vs learner recompute tolerance <=1e-3, FP32 <=1e-4; frozen weights and identical old-policy precision. 단순 aggregate diff 0 보고가 아닌 representative row/branch stratification.
4. GAE, returns, actor/value masks, singleton handling, last padded batch, gradient and optimizer state parity. Floating-point bit-exact이 불가능한 GPU 순서는 사전 선언 오차 조건으로 비교.
5. `torch.isfinite` all relevant loss/gradients, KL and ratio stable, checkpoint resume output equality, actual committed natural matches clock. DDP step count/optimizer sync 기록.
6. Peak RSS/HBM budget 및 swap checks. Short test로 성공 판정 금지, bounded longer smoke 반드시 실행.
7. Any precision/packing changes: known/masked/hidden-observation contract, no silent semantic narrowing.
8. 실패 시 feature flag rollback, 원인 기록, 영구 커밋 전에 재시험.

---

## 4. P0 — 성능보다 먼저: PPO 지표와 수치 계약 정확성 수정

### P0.1 `update()` microbatch 통계 집계 결함

**문제 위치:** `agent/ppo/learner.py::PPOLearner.update`.

현재 `for micro in microbatches:` 내부에서 매 micro마다 `stats`를 갱신하지만, 바깥의 `for key in totals`, `epoch_kl`은 **마지막 micro의 stats만** 사용한다. 이것이 최종 `approx_kl`, clip_fraction, entropy, ratio 및 early stop 모니터의 체계적 편향을 유발할 수 있다. 단일 microbatch 테스트만으로는 이 결함을 잡기 어렵다.

**작업:**

- `actor_rows`, `valid_rows`에 대한 **분자/분모 원시 합계**를 microbatch마다 `detach`해 수집한다. `KL`, ratio, entropy, uniform-KL, clip fraction은 **유효 actor rows**를 가중치로; value loss는 **유효 value rows**를 가중치로 계산한다. 한 지표를 전체 행과 actor 행 중 어느 분모로 평균낼지 Full Spec을 따른다.
- GPU에서 작은 통계 Tensor를 micro마다 `.item()`으로 꺼내지 않도록 가능한 한 minibatch 수준에서 GPU scalar sums를 누적한 다음 1회 일괄 D2H로 가져온다.
- Epoch KL 조기 종료는 epoch의 전체 유효 actor sample 평균에 대해 계산하고, 변동이 있으면 old vs new 집계 및 첫 iteration 안전성 영향 설명. PPO gradient 수식 자체를 `report` 패치의 부수 효과로 수정하지 않는다.
- `GradScaler.step()`이 NaN/Inf로 실제 optimizer step을 skip한 경우 `optimizer_steps`와 진행 장부가 무조건 +1 되는지 검증하고, 실제 step 성공 기준으로 기록한다.
- `report.grad_norm`의 마지막 step vs epoch/global representative 지표를 명확히 구분하고 분포/최댓값을 추가할 수 있다.

**Tests to add:** 마지막 micro의 KL은 작지만 앞선 micro의 KL이 큰 synthetic batch, 반대 경우, 마지막 padded batch, singleton-only actor mask, loss equality before/after, skipped optimizer step. 이 시나리오가 실제 새 test에서 기존 코드 FAIL, 수정 후 PASS를 보여야 한다.

### P0.2 Gradient accumulation과 padding의 실제 수학 검증

- `micro.sample_weight / len(microbatches)`가 global 4,096-row minibatch 전체의 유효 sample 및 actor/value 분모에 대해 같은 gradient를 제공하는지 테스트한다. 특별히 last partial minibatch의 `row_valid`, micro별 actor row 수가 불균일한 adversarial case를 검증한다.
- 실제 `policy_loss`는 actor mask, `value_loss`는 value mask에 따라 다른 분모를 가질 수 있다. mini/micro mean을 단순 균등 평균하면 두 loss의 full-batch objective와 다를 수 있다. **전체 목적함수의 동등성**을 수학적으로 계산해서 parity를 입증하거나, 계약에 맞게 weighting을 고쳐야 한다.
- 만약 기존 수학에 문제를 발견하면 correction으로 별도 커밋하고, 성능 개선 커밋과 분리한다. 성능 벤치마크의 모델 결과 분포가 달라질 수 있음을 별도 기록.

### P0.3 보상, GAE, 샘플 예산, 계측 통제

- `match_id`, `side`, `request_index` 기준의 trajectory chain이 interleaved/permuted rows에서도 동일한지 검사한다. incomplete/missing terminal은 무승부로 바꾸지 않는다.
- GAE gamma=1/lambda=.95, per-side reward terminal once, bootstrap, overshoot/commit clock 검증. GAE NumPy/CPU 개선 시 동일 입력에 FP32 tolerance gate를 적용한다.
- `recompute_check`는 기본 최대 1,024 sample만 확인한다. 대표 계층화 strata(대진, preview/normal/replacement, 1–4 branch, singleton/multi, packed, f16/f32)로 검사 범위를 넓힌다. 검증 전수/표본 분모를 보고한다.
- `bench_pa3_end_to_end.py`의 주석에 명시된 checkpoint 경로가 실제 실행됐는지 확인한다. `--checkpoint` 인자만 있고 실제 write가 없는 경우 명칭 수정, 실 저장 포함 별도 operational benchmark를 추가한다.
- 원본 report의 `git_dirty`, model SHA, seed, GPU PCI, `all_in_wall_s` 구간을 확인한다. `median games/s`에 다른 repeat의 wall 값을 대응시키지 않는다. `stage_medians.share_of_wall` 분모는 동일 repeat 기준 또는 명확한 통계 구조로 수정.

**P0 수락:** 구현 정확성 통과, regression 확대, 각 지표 sample-weighted reference oracle PASS, 최신 clean SHA에서 신규 수치 baseline 보존. P0가 끝나기 전 throughput 승격 주장 금지.

---

## 5. M0 — PPO learner 290.7초를 실제 stage로 분해

### M0.1 계측 위치

`agent/ppo/learner.py`, `agent/buffer/rollout_buffer.py`, `agent/ppo/gae.py`, `scripts/bench_pa3_end_to_end.py`에 scoped opt-in profiler를 추가한다.

1. `compute_gae`: groups build, sort, per-sequence tensor construction, recurrence, row write-back.
2. `prepare_batch`: observation store `stacked()`, `from_compact_numpy()`, category/floats cast, `BranchCandidatesBatch.from_rows`, scalar columns, advantage normalize.
3. `update()`: epoch shuffle, global minibatch index/select, microbatch select, H2D, `_forward` encoder/scorer, backward, grad clip, optimizer/scaler, metrics GPU→CPU sync, checkpoint write.
4. GPU breakdown: kernel CPU dispatch, SDPA, embedding, GRU scorer, policy/value head, backward kernels, optimizer kernels, PCIe H2D/D2H, GPU idle/wait, allocated vs reserved.
5. CPU breakdown: per-thread wall/cpu time, allocation, NUMA local/remote pages, major page faults, swap-in/out, context switches, memory bandwidth if profiler available.

### M0.2 계측 신뢰성

- `time.perf_counter()`를 CPU stage로 사용한다. CUDA 비동기 작업은 **sampling profiler / CUDA events / profiler trace**로 검증하고 GPU 작업의 벽시간을 무조건 CPU stage와 합산하지 않는다.
- GPU timing을 위해 **모든 micro마다 `torch.cuda.synchronize()`하지 않는다**. 계측 자체가 최적화를 방해하기 때문이다. 짧은 profiling session만 계측 추가, production off.
- 같은 weight SHA, match seed, 1,137 team manifest로 CPU/NUMA0, GPU0 및 듀얼 비교. 계측 추가 전후 overhead를 측정한다.
- 각 stage `count`, bytes, nanoseconds, memory peak, H2D throughput, CUDA kernel occupancy/launch counts를 기록할 수 있는 json schema를 만든다.

**산출물:** `docs/perf/V2_LEARNER_BREAKDOWN.md`, `runs/perf/v2_learner_breakdown_*.json`, 필요한 경우 `runs/perf/traces/`에 안전한 분량만 보관. 가장 큰 3개 stage의 실측 시간과 1차 개선 가설이 나와야 P1로 간다.

---

## 6. P1 — GAE/롤아웃 데이터 레이아웃 혁신

### P1.1 GAE hot path 최적화

**위치:** `agent/ppo/gae.py`, `agent/buffer/rollout_buffer.py`.

- `rows`를 Python 객체별 작은 `torch.tensor`로 만들지 않고 episode/side offset과 연속 NumPy `float32`/`bool` 배열로 표현한다.
- 역방향 GAE recurrence는 실측상 필요한 경우 순수 ndarray 계산, 효율적 단일 CPU loop 또는 프로젝트 도구로 최적화한다. Numba/새 패키지 설치는 기본 금지. native Rust hot-path 이전은 그 자체가 실제로 필요하다고 증명될 때 별도 실험.
- Reward/GAE/return/value FP32 precision, side별 terminal/bootstrapping, empty/long/irregular trajectory, rollouts permutation에 대한 reference fixture를 만든다.
- Small batches와 130k+ rows에서 **GAE-only**, 전체 PPO all-in을 모두 재측정. 속도 개선이 없으면 복잡한 구현 채택 금지.

### P1.2 Columnar on-host storage

**위치:** `agent/buffer/rollout_buffer.py`, `agent/types/requests.py`, `agent/types/observation.py`.

- 현재 `list[RolloutRow]`, `list[dict[str,np.ndarray]]`, tuple of candidate actions 및 per-row `.copy()`를 유지하는 경로를 자세히 프로파일링한다.
- `ObservationStore`를 열 단위/슬랩 단위 typed contiguous buffer로 개편한다. 예: `[N,96,32]` category IDs `uint16`, masks `uint8`/`bool`, float `float32` 또는 계약 허용의 lossless storage, branch actions small integer, candidate offsets/prefix counters 등. **압축/정밀도 변경을 무조건 lossless로 주장하지 않는다.** 기존 `float16` compact 저장이 근사치임을 드러내고 sample/recompute parity 및 semantic critical feature 검증을 유지한다.
- Ragged effects, repertoire, types, base moves, move effects는 **원본 wire 보존**, 모델 현재 미소비 상태와 구분한다. 모델 입력의 의미 정보 완전성은 별도 readiness gate.
- 한 iteration을 담을 RAM budget 24GiB와 전체 host 48GiB, dual ranks 합계 및 spill 시 filesystem free-space를 감시한다.
- 늘어나는 ndarray 재할당 대신 bounded chunk/slab와 reuse. 끝난 iteration은 참조를 해제하고 memory growth 3 iterations에서 확인한다.

### P1.3 Streaming minibatch materialization

- **핵심:** `prepare_batch()`에서 전체 130k+ observations, candidates를 `RolloutBatch`로 한 번에 펼치지 말고 compact host store의 row IDs + 스칼라 FP32 advantage/return columns를 보존한다.
- 4096 global minibatch의 shuffled indices만 생성하고, **선택된 행에 대해서만** 필요한 candidate/observation columns를 생성한 뒤 256 microbatch 단위 GPU로 전송한다.
- Epoch마다 full observation를 재변환하는 비용과 full pre-materialization overhead 간 tradeoff를 profiler로 비교한다. 소규모 CPU cache/pinned slab 또는 prefetch queue를 A/B 시험한다.
- `from_rows()`에서 매 호출 dense `[B,4,64,6] int64`를 만드는 비용 제거 후보. Pack bytes + offsets → bounded contiguous slots → GPU에서 필요 시 `long` cast. Candidate order와 mask가 보존되어야 하며 legal capacity 초과는 에러, 절삭 금지.
- `match_ids`, `sides`, `turn`, `policy_id` 등 forward에 불필요한 metadata는 CPU에 둔다. 데이터를 GPU로 넘길 때 **모델 실제 소비 필드만** 옮긴다.
- 사전 pad/selection과 scatter 인덱스는 동일 `torch.Generator` RNG 의미를 유지한다. Last partial valid-mask를 유실하거나 중복 학습하지 않는다.
- Valid micro별 **최종 gradient**, optimizer state, PPO metric, action/logprob parity를 FP32/FP16에 대해 reference와 비교.

**P1 수락:** all-in 개선과 RAM 감소, 2k/10k/long bounded cases, no OOM, no swap, exact row count. 속도 개선 없이 메모리만 좋아졌다면 이를 분리 보고하고 채택 여부 판단.

---

## 7. P2 — V100 learner CUDA 연산 최적화

**선행:** M0 breakdown에서 모델 forward/backward/optimizer가 실제 주요 병목인 경우. GPU가 이미 바쁘지 않은데 커널부터 바꾸지 않는다.

1. FP16 autocast + dynamic scaler, FP32 critical math를 유지한다. GradScaler overflow 감지, missing actual optimizer steps, logs/metrics sync 비용을 계측한다.
2. GPU memory transfer: repeated `.to(cuda)`, `index_select`, metadata duplicate copies, batch padding waste. Reusable NUMA-local pinned staging slabs를 capped(전체 1GiB)로 실험하고 CUDA stream overlap을 profiler timeline으로 입증한다.
3. Global minibatch 4096 유지. GPU당 microbatch 128/256/512 등 **gradient accumulation을 재설계하되** effective sample weighting과 optimizer step timing 동일을 증명하고 메모리/속도를 비교한다. Microbatch 튜닝이 policy/hyperparameter 변경으로 둔갑하면 안 된다.
4. FP32 probability, logits mask, no illegal candidate, entropy/KL and value loss 유지. Unsafe fused kernel이나 TF32/BF16/FP8/FlashAttention2를 V100에 무단 도입하지 않는다.
5. `torch.optim.Adam`의 foreach/fused/CUDA path는 **현재 Torch 2.14 + sm_70 지원 확인 후** 옵트인 벤치마크만 한다. 새 패키지나 CUDA runtime을 무단 교체하지 않는다.
6. `torch.compile`은 현 Full Spec의 eager baseline에서 비활성. CUDA Graph PoC는 **동일 weights, shape buckets, RNG, buffer lifetime, masks 및 gradients parity를 검증하는 별도 feature branch에서만** 시험하고 승인된 v2 기본 경로와 섞지 않는다.
7. `_forward` 내 GPU scalar sync, `bool(tensor)`, `.item()`, `.tolist()`, Python instrumentation을 Nsight/PyTorch Profiler로 찾아 줄인다. Concurrency는 buffer ownership 및 order 정확성을 위반하면 중단.
8. 순수 `GPU forward/backward decisions/s`, `effective optimizer steps/s`, `H2D bytes/s`, `all-in games/s` 모두 보고한다.

**P2 수락:** 안정성 및 gradient/logprob parity 통과, profiler에서 실제 GPU/launch 개선 근거, 3회 반복 all-in 의미 있는 개선. 복잡도가 크고 개선이 noise 아래면 거절.

---

## 8. P3 — Dual GPU actor + 실 DDP PPO 완성

**현 상황:** `scripts/run_actor_pair_real.py`는 두 GPU real-policy collection까지만 실행하며 `ddp_learner`는 `not part of this measurement`. Phase 6 전체 완료 아님.

1. GPU0/NUMA0, GPU1/NUMA1 각각 같은 frozen policy를 사용한다. 각 rank 1024 env, 16 native workers baseline, physical GPU bus ID 재확인.
2. CPU/NUMA memory pinning이 **환경 엔진 생성 및 메모리 슬랩 할당 전** 수행되는지 보장한다. Rank 성능 차이는 장시간 시험으로 확인한다.
3. Two-process NCCL DDP learner, `no_sync()`의 context 안에 **첫 7개 microbatch의 forward와 backward 모두** 넣고 마지막 microbatch에서 gradient reduction. `256 × 8 × 2 = 4096` global contract.
4. DDP all-reduce weighted loss and gradients: 각 rank rows 수가 다르거나 final padded minibatch에 유효 row 수가 다를 때 single-process reference global objective/gradient와 비교한다. `no_sync`/allreduce는 FP rounding 차이가 있을 수 있으므로 지정 tolerance로 검증.
5. Model version, optimizer/scaler/scheduler state, match clock, current vs historical opponent row ownership이 두 rank에서 동일하게 동기화되는지 검증. 같은 자연완결 경기를 두 번 count하지 않는다.
6. NCCL P2P 비활성, SHM 활성, user service 영향 0. Transport/log warnings와 staggered launcher, DDP barrier time, rank idle time 기록.
7. GPU1 150W의 실제 지속 처리량과 GPU0 175W를 분리 측정한다. 동일 workload A/B(`single gpu0`, `single gpu1`, `dual`)을 2k 단기 및 10k 이상 장시간으로 반복한다.
8. DDP가 all-in에서 실제로 느리다면 spec의 **single GPU learner global 4096 = micro 256 × 16** fallback과 actor 2-GPU collect 가능성을 함께 측정한다. 이는 모델 압축/규칙 축소가 아닌 장비 운영 모드 비교다.

**P3 수락:** dual all-in **실측**, 4 epochs, DDP gradient parity, no_sync 확인, zero stale rollout rows, 0 operational errors, checkpoint save/resume rank consistency, RAM cap, 3회 반복 및 long test.

---

## 9. P4 — Actor serial round, rolling slots, CPU/GPU 동시 실행

### P4.1 현재 확정된 actor 비용

최적화된 2k single rank stage 중앙값: model ~5.25s (24.5%), buffer record ~4.31s (20.1%), Rust observe ~3.65s (17%), H2D ~2.18s (10.2%), parse ~1.09s (5.1%), Rust step ~0.34s (1.6%). 입력 단계와 GPU inference가 앞뒤로 직렬화되어 있다.

### P4.2 개선 실험 순서

1. **독립 코호트 2개**로 나눠 한쪽 CPU 관측을 준비하는 동안 다른 쪽 GPU encode/sample을 수행하는 bounded pipeline. `A/B` queue size=1/2/3, per-rank engine state ownership, GPU stream completion barrier 명시.
2. **Asynchronous prefetch**: pinned memory와 nonblocking H2D, double buffer. overlap은 profiler GPU/CPU timeline의 실제 겹침으로 입증한다.
3. **Rolling environment slots**: 종료된 env에 새 episode를 할당해 긴 tail 배틀 때문에 다른 환경이 유휴 상태로 남지 않도록 한다. 동일 팀/seed distribution과 natural completions 유지. 정해진 match count에서 신규 reset 중단 후 나머지 open games 자연 종료, total overshoot 집계.
4. **Branch prefix walk**는 각 level에서 샘플링된 행동 이후 새 합법 후보를 구해야 한다. prefix-dependent mask를 미리 고정하거나 예측해 legality를 생략하는 shortcut 금지.
5. 긴 경기 starvation 없도록 fairness, bounded queue/backpressure, seed and side attribution, order-independent trajectory collector, collision-free match ID, independent RNG stream 보존.
6. Serial actor baseline / overlapped actor / two-rank overlapped actor / end-to-end PPO를 같은 10k+ match panel에서 비교.

**P4 수락:** 단일 actor 3회 수집 및 학습 all-in 증가, old vs new outcome/trajectory invariants, 0 operational failures, no deadlock after bounded stress, no dropped long games.

---

## 10. P5 — Rust 관측 및 후보 경로의 추가 최적화

**위치:** `engine/src/python.rs`, `engine/src/batch.rs`, `engine/src/observation.rs`, `engine/python/pa3_engine/observation.py`, `agent/types/observation.py`.

1. `observe_fixed_batch()` 호출 때마다 `Encoder::from_validated_dex(&batch.dex)`를 생성한다. Dex가 immutable일 때 `Encoder` 캐시 가능성을 검토하되 **실제 생성 시간부터 계측**한다. Python-bound Rust ownership, lifetime, concurrent worker 안전성을 테스트한다.
2. Rust observe loop: `observe_encoded_batch_into`, `pack_fixed_into`, `pack_ragged_into`, `PyBytes::new`의 개별 비용과 allocation/memcpy 총량 계측. 41,954 bytes per fixed view, 96 tokens, 32 categorical, 50 float, 40 flag features를 기준으로 bytes accounting.
3. 현재 fixed->numpy adapter가 bool/int64/FP32 변환과 copy를 하는 구간은 필요한 자료형만 GPU에 올리도록 최적화. 단, player-safe known masks, token roles, candidate dependencies 불변.
4. Lossy `f16` observation transport는 **독립 PoC**: Rust simulation은 원래 f32/정수로 진행, GPU 입력 wire만 바꾸더라도 정책 입력 수치/확률은 변할 수 있으므로 포맷 schema bump 및 golden parity, precision-aware rollout collection/recompute gate, ablation이 선행되어야 한다. f32에서 f16으로 좁히면서 '동일 관측' 주장 금지.
5. Columnar direct scatter 혹은 pinned slab에 쓸 때 버퍼 수명, reuse, zero-copy ownership 보장. Rust `step_batch` 자체는 현재 actor wall의 1.6%이므로 먼저 최적화하지 않는다.
6. Parser의 ragged 5종 필드가 wire에 보존되나 모델이 직접 미소비하는 현 상태를 별도의 **observation completeness** 게이트로 유지한다. Semantics feature gap이 있으면 학습 승인 전 별도 설계/승인 필요.
7. 풀 규정 coverage가 확장되면 현재 P=64 padded candidate bound의 충분성 재평가. 현재 observed max 26은 full scope의 보증이 아니다. Candidate truncation 불가.

**P5 수락:** Rust reference tuple oracle 0 mismatches, exact wire parity, hidden info test, all schema feature inventory, stage wall and all-in speed gain, no RAM budget break.

---

## 11. P6 — 디스크, 메모리, NUMA, PCIe, 전력 검증

### 11.1 SSD 질문의 정확한 답을 실측으로 확정

현재 코드상 팀/Dex는 시작 때 `std::fs::read`로 RAM에 읽고, `InlineObservationStore`와 `RolloutBuffer`는 메모리다. 일반 PPO 학습의 관측 이동은 **RAM→GPU VRAM(PCIe)** 이며 SATA SSD→GPU 전송이 아니다. **NVMe 업그레이드를 주요 throughput 병목이라고 가정하지 않는다.**

저장장치가 학습을 제한하는 경우: swap-in/out, heavy major page faults, NVMe/HDD spill, checkpoint/fsync, 평가/로그 대량 쓰기가 실행 중 관측되는 경우. 따라서 아래 읽기 전용 진단부터 수행한다.

```bash
# 장비와 현재 메모리/스왑 상태를 읽기 전용으로 확인
lsblk -o NAME,TYPE,TRAN,ROTA,SIZE,MOUNTPOINTS
free -h
cat /proc/swaps
vmstat 1 15
# iostat, numastat, pidstat, perf는 이미 설치된 경우에만 사용
command -v iostat && iostat -xz 1 15
command -v numastat && numastat -p <PA3_PID>
command -v pidstat && pidstat -dru -p <PA3_PID> 1 15
# PA3 학습 프로세스만 관측: i/o counters, major faults (프로세스 종료 전)
cat /proc/<PA3_PID>/io
cat /proc/<PA3_PID>/status
```

`<PA3_PID>`는 실제 프로젝트 프로세스 PID로 바꿔 실행한다. 시스템 전체 package/driver 변경, 서비스 stop, CPU governor/BIOS/팬 설정 변경 금지.

### 11.2 Memory / NUMA / PCIe

- RSS와 per-rank RSS, PSS/anonymous/locked, rollout actual bytes, Python object overhead, pinned bytes, GPU allocated/reserved를 **같은 실행 창**에 기록. `resource.ru_maxrss`는 프로세스 누적 peak이므로 다른 반복과 정확히 대응하도록 subprocess isolation 적용.
- `vmstat si/so`, major faults, swap size, disk queue await, user/system CPU, iowait. Swap 0, 실제 disk I/O 낮으면 SSD 원인 기각.
- NUMA-local Rust buffers, pinned staging allocation 시점, memory locality. `numactl`이 있으면 baseline 확인 및 읽기 전용 정책 준수.
- `nvidia-smi dmon` 혹은 쿼리 기반 장기 GPU util, memory, power/clock/temp 샘플링. 시작 시 idle util=0을 training util로 보고하지 않는다.
- H2D와 D2H의 bytes, pinned staging overlap, DDP host-staged communications, PCIe throughput. SSD의 **NVMe PCIe lane 소비와 GPU 슬롯 경합**까지 고려하지만, 실측 없이 PCIe 링크나 BIOS 설정 변경 금지.
- Disk free, checkpoint path, log growth, compact spill, crash recovery footprint. 사용자용 NVMe 선택은 저장 용량/신뢰성과 long-run 운영성으로 별도 결정.

**P6 수락:** Storage bottleneck 여부를 `observed / not observed / unmeasured`로 선언하고 근거 숫자 첨부. 메모리 안정성, swap=0 권장, GPU telemetry와 NIC/서비스 영향 기록.

---

## 12. P7 — 실제 운영 속도, 재현성, 수락 기준

### 12.1 벤치마크 3층

**Layer A: 단위 마이크로벤치** GAE-only, column store, candidate build, pure GPU forward/backward/optimizer, Rust encode/pack, H2D/D2H, NCCL barrier. 실행 비용 및 semantics를 단위로 분리.

**Layer B: Real-policy actor only** 1 rank GPU0, 1 rank GPU1, 2 ranks wall-aligned. 동일 seed, 팀풀, 관측/정책, 자연완결 경기 수, 0 운영 오류. 2k×3 + 10k 이상 warm longer.

**Layer C: Bounded PPO / true operational** collect + GAE + 4 epoch update + scaler/scheduler/metrics + 실제 checkpoint write + DDP sync + optional fixed evaluation window. 상호 다른 포함 범위는 별도 column, `report_includes` manifest에 명시.

### 12.2 필수 run metadata

- git SHA, clean/dirty 상태, diff hash if dirty, script SHA, frozen dataset digest, model SHA, policy ID, opponent ID, seed, real natural games, decisions, rows/actor rows, full team coverage distribution, median and min/max or variation; 각 repeat의 wall 값.
- Hardware PCI/NUMA, per-GPU TDP limit, throttling/clocks, RAM/RSS/PSS/pinned, disk free, CPU core map, stack pins, services touched (should be none).
- CPU stage wall, GPU kernel time, actual H2D and D2H, optimizer/no_sync time, comm wait, memory allocation overhead.
- `readiness_check` exact FAIL list and independent rule lane status. GPU0/GPU1 DDP checksum and checkpoint resume parity.
- `bounded_ppo_games/s` vs `steady_state_all_in_games/s` 명칭 명확히 구분.

### 12.3 금지할 벤치마크 착시

- 1 repeat만으로 강한 개선 결론 내기, 서로 다른 모델 SHA/팀셋 비교를 '통제 실험'이라 명명하기.
- Random-policy / Rust-only games/s를 real-policy 수치에 혼합.
- 후보 행동 일부를 자르거나, 긴 배틀을 버리거나, 종료 이전을 승리로 count.
- GPU-event elapsed wall을 GPU pure kernel time으로 주장.
- all-in에서 GAE, model sync, checkpoint/metrics/eval을 제외하고 정상 운영 속도라고 보고.
- 디스크 I/O 근거 없이 SATA가 학습을 저해한다고 단정.
- 성능 수치가 좋아졌다는 이유로 준비되지 않은 engine에서 100M training 시작.

### 12.4 Success tiers

- **Correctness tier:** P0 repaired, ready smoke remains green, PPO parity tests no regression.
- **Training throughput tier:** measured 10k bounded all-in improvement, repeated; two GPU DDP validated, memory budget respected.
- **Production tier:** 16/16 readiness independently passed, 2+ iteration checkpoint/recovery and eval end-to-end verified, user explicitly authorizes long training.
- **Extreme frontier tier:** actor and learner each profiled and optimized until repeated proposed modifications no longer improve all-in beyond noise; hardware and complexity tradeoffs documented. 1,000/2,000/5,000+ are aspirations, not promises.

---

## 13. P8 — Full-scope engine readiness는 별도 필수 게이트

`engine/TRAINING_READINESS.md` 기준 16개 항목 중 현재 10개 PASS, **2,3,5,8,15,16 FAIL**. 이는 성능 트랙의 버그나 `P0`보다 더 넓은 Full Regulation 지원 범위 문제다. 별도 engine workstream으로 진행하되 속도 향상과 혼동하지 않는다.

- 2: 515 legal moves 전체 지원.
- 3: 223 legal abilities 전체 지원.
- 5: dynamic/reachable effect closure.
- 8: full-scope validation corpus에서 unsupported-mechanic errors 0.
- 15: full-coverage throughput 재측정.
- 16: reference와 full-battle/differential mechanics validation.

관측 completeness 및 model consumption gate는 `readiness_check` PASS의 의미와 별도로 계약상 보존한다. Frozen train 1,137 teams에서 0 operational error가 났다는 사실은 모든 규정에 대해 정확하다는 증거가 아니다.

**본학습 시작 금지:** full-scope 16/16, training team/checkpoint model readiness 및 사용자 승인 전까지 대규모 100M run이나 자동 ladder 학습/평가는 시작하지 않는다.

---

## 14. P9 — RL Sample Efficiency 연구와의 인터페이스

v2.0은 속도와 PPO 구현 정확성을 개선하는 트랙이다. 동일한 계산 예산에서 성능이 향상되는지 확인하는 연구는 **`docs/PokeAgent3_RL_Sample_Efficiency_Research_2026-10-08.md`**를 유지한다.

- Before advanced RL tuning: 고정 대진, 양좌석, 공통 seed, frozen anchors, overfitting/weakness metrics, 여러 학습 시드와 confidence interval를 포함하는 평가 파이프라인 설계.
- PPO metrics correctness, critic explained variance/calibration, policy entropy/KL, opponent snapshots 구현은 sample-efficiency 실험의 선결 사항.
- Uniform historical vs PFSP, critic update changes, PPG, reward shaping은 **독립 실험**, 동등 경기 수와 동일 실제 학습 시간 평가 후에만 승격 검토.
- PPO 4 epochs, reward, scheduler, historical opponent ratio 등을 조용히 바꾸지 않는다. Sample efficiency를 개선하기 위해 더 좋은 행동을 가르치는 teacher BC는 현재 scope가 아니다.
- 본 프로젝트의 실제 목표는 단순 최대 games/s가 아닌 *예산당/시간당 실력 향상*이며, 학습 품질과 속도 지표를 함께 기록하되 서로 혼동하지 않는다.

---

## 15. 권장 원자 커밋 및 진행 순서

| 실행 순서 | 브랜치/커밋 주제 예시 | 반드시 제공할 산출물 | Gate |
|---|---|---|---|
| 0 | `v2-baseline-manifest` | Clean `b72f5f4` baseline, source/test inventory, disk/RAM snapshot | HEAD/seed verified |
| 1 | `v2-correct-ppo-metrics` | Last-micro regression tests, correct epoch KL, scaler accounting | C0 numeric parity |
| 2 | `v2-profile-learner` | 290.7s stage breakdown, CUDA kernel trace, CPU wait report | biggest 3 stages isolated |
| 3 | `v2-columnar-gae` | GAE reference parity and CPU cost difference | same trajectory returns |
| 4 | `v2-stream-rollout` | Streaming batch/columns/padding/gradient parity | all-in improvement |
| 5 | `v2-optimize-learner` | Microbatch/staging/optimizer A/B, CUDA trace | safe FP32-critical math |
| 6 | `v2-dual-ddp` | Real two-GPU all-in and `no_sync` parity | DDP stable, correct weighted gradients |
| 7 | `v2-async-cohorts` | Actor overlap, rolling slots, long-tail correct results | real overlap and gain |
| 8 | `v2-observation-native` | Encoder/cache/pack/buffer optimization, schema audit | exact observation parity |
| 9 | `v2-system-telemetry` | Disk/NUMA/PCIe/swap conclusions, service check | no unapproved config change |
| 10 | `v2-frontier-final` | Final report plus raw 3-repeat JSON, long smoke, readiness separately | all gates green |

**원칙:** 1~2는 필요하면 병렬 계측 가능, 단 correctness patch와 speed measurements 버전 혼동 금지. DDP는 loader와 accuracy가 안정화될 때 시행. 뒤쪽 고급 단계는 항상 직전 profiler가 가리킨 가장 큰 병목을 우선한다. 숫자 목표 때문에 순서를 맹목적으로 고수하지 않는다.

---

## 16. 실행 및 검증 명령 초안

다음 명령은 **현재 브랜치에서 확인한 CLI**에 기반한 기본 smoke 예시다. v2 구현으로 새 플래그가 추가되면 새 명령을 run manifest에 기록한다. `PYTHONPATH`는 repo root에서 사용한다. Benchmark는 100M 학습을 시작하지 않는다.

```bash
# 0. 현재 작업 상황 기록
pwd
git fetch origin
git status --short
git rev-parse HEAD
git branch --show-current
# b72f5f4 이상 최신 HEAD를 기록하고, 작업 브랜치 보호.

# 1. 기존 model/GAE/PPO 테스트
PYTHONPATH=engine/python:. .venv/bin/python -m pytest tests/agent -q

# 2. Rust full readiness는 아직 nonzero expected. exit code를 보존.
bash scripts/cargo.sh run --locked --release --example readiness_check

# 3. Single GPU bounded actor (3 repeats), optimized fixed/packed FP16
PYTHONPATH=engine/python:. .venv/bin/python scripts/bench_pa3_end_to_end.py \
  --games 2048 --envs 1024 --workers 16 --repeats 3 \
  --mode collect --observations fixed --candidate-wire packed \
  --precision fp16 --inference-mode --device cuda:0 \
  --report runs/perf/v2_actor_single.json --tag v2-actor

# 4. Bounded PPO, includes collect + prepare + 4 epochs.
PYTHONPATH=engine/python:. .venv/bin/python scripts/bench_pa3_end_to_end.py \
  --games 10000 --envs 1024 --workers 16 --repeats 1 \
  --mode full --observations fixed --candidate-wire packed \
  --precision fp16 --inference-mode --device cuda:0 \
  --report runs/perf/v2_full_10k.json --tag v2-bounded-ppo

# 5. Dual real-policy actor only. 실제 CLI help를 우선 확인.
PYTHONPATH=engine/python:. .venv/bin/python scripts/run_actor_pair_real.py --help

# 6. Read-only system I/O/memory snapshot
lsblk -o NAME,TYPE,TRAN,ROTA,SIZE,MOUNTPOINTS
free -h
cat /proc/swaps
vmstat 1 15
```

**주의:** `scripts/cargo.sh`가 miniDC에서 프로젝트 전용 툴체인 경로를 올바르게 구성하는지 실행 전 확인한다. 기존 benchmark script의 `--checkpoint` 인자는 실제 저장 동작을 보증하지 않으므로 operational checkpoint benchmark는 **실제 `torch.save`/atomic rename/fsync 경로가 구현되고 검증된 후** 별도 계측한다. 긴 실행과 DDP는 GPU 시간/운영 안전 조건을 확인한 뒤 작은 smoke부터 시행한다.

---

## 17. 수락 체크리스트: 작업 종료 조건

- [ ] New baseline clean SHA+dataset manifest, repeated 2k and longer 10k run data saved.
- [ ] PPO last-micro stats bug reproduction and resolution, correct KL/actor/value denominators; padded/microbatch gradient equivalence.
- [ ] Correct optimizer step counting when scaler skips; actual match-count LR/commit invariants.
- [ ] GAE and advantage normalization reference parity; terminal and interleaved trajectory edge cases.
- [ ] 290.7s PPO stage breakdown with true GPU kernel vs CPU wall/copy/sync.
- [ ] Columnar rollout, bounded chunk storage, no loss of provenance or ragged transport, memory leak checks.
- [ ] Streaming 4096 global/256 microbatches; no full-iteration expansion, equal update math.
- [ ] Model H2D improvements and nonessential metadata on CPU; GPU-only and learner-side speed measured.
- [ ] Dual GPU DDP actual all-in 4 epochs, `no_sync`, NCCL, gradient parity, checkpoint/resume.
- [ ] Async cohort/rolling-slot experiments with real overlap measured, slow games not dropped.
- [ ] Rust observation pack/encoder cache performance and parity, candidate capacity full-scope safety.
- [ ] Disk/Swap/NUMA/PCIe and GPU utilization sampled during run. NVMe priority evidence-based.
- [ ] GPU0/GPU1/dual, actor vs bounded PPO vs operational all-in clearly separated.
- [ ] Every optimized step remeasured 3 times where feasible, no illegal action, 0 operational aborts, no policy semantics change.
- [ ] Observation info completeness separate gate, 16/16 readiness remains independent. No 100M run started.
- [ ] `docs/perf/V2_LEARNER_BREAKDOWN.md`, `docs/perf/V2_CORRECTNESS_AUDIT.md`, `docs/perf/V2_IO_NUMA_REPORT.md`, `docs/perf/V2_FRONTIER_REPORT.md`, raw manifests/traces, `docs/perf/FINAL_REPORT.md` updated.
- [ ] Commit/push SHA and clean-tree report, reproducible commands, accepted/rejected experiments + exact remaining limit and what evidence is missing.

**종료 조건:** 1,000 games/s를 넘거나 넘지 못한다는 이유로 종료하지 않는다. 높은 우선순위 개선의 유효성이 판별되고, 더 가능한 개선의 비용/정확성 위험 대비 실측 수익이 명시된 시점에 기술적으로 plateau를 보고한다. 완료 가능한 항목을 미완료로 남겼다면 정확한 blockers와 다음 실행 명령을 제출한다.

---

## 18. GitHub 근거 및 원본 링크

기준 ref는 아래 SHA를 사용한다. 이후 커밋으로 변해도 최초 근거가 남도록 고정 SHA를 포함한다.

- [Final report](https://github.com/mwl313/PokeAgent3.0/blob/b72f5f4c73dc8cbdf06584a5e0ed9c0ea7f6b8d8/docs/perf/FINAL_REPORT.md)
- [Frontier report](https://github.com/mwl313/PokeAgent3.0/blob/b72f5f4c73dc8cbdf06584a5e0ed9c0ea7f6b8d8/docs/perf/ULTIMATE_FRONTIER_REPORT.md)
- [Baseline and phase timings](https://github.com/mwl313/PokeAgent3.0/blob/b72f5f4c73dc8cbdf06584a5e0ed9c0ea7f6b8d8/docs/perf/BASELINE.md)
- [10,240-game optimized all-in raw JSON](https://github.com/mwl313/PokeAgent3.0/blob/b72f5f4c73dc8cbdf06584a5e0ed9c0ea7f6b8d8/runs/perf/allin-fp16-10k-fastrows.json)
- [2k repeats raw JSON](https://github.com/mwl313/PokeAgent3.0/blob/b72f5f4c73dc8cbdf06584a5e0ed9c0ea7f6b8d8/runs/perf/fastrow-fp16-im.json)
- [Dual rank raw JSON](https://github.com/mwl313/PokeAgent3.0/blob/b72f5f4c73dc8cbdf06584a5e0ed9c0ea7f6b8d8/runs/perf/dual-real.json)
- [PPO learner](https://github.com/mwl313/PokeAgent3.0/blob/b72f5f4c73dc8cbdf06584a5e0ed9c0ea7f6b8d8/agent/ppo/learner.py)
- [GAE implementation](https://github.com/mwl313/PokeAgent3.0/blob/b72f5f4c73dc8cbdf06584a5e0ed9c0ea7f6b8d8/agent/ppo/gae.py)
- [Rollout buffer](https://github.com/mwl313/PokeAgent3.0/blob/b72f5f4c73dc8cbdf06584a5e0ed9c0ea7f6b8d8/agent/buffer/rollout_buffer.py)
- [Candidate batch data path](https://github.com/mwl313/PokeAgent3.0/blob/b72f5f4c73dc8cbdf06584a5e0ed9c0ea7f6b8d8/agent/types/requests.py)
- [Native collector](https://github.com/mwl313/PokeAgent3.0/blob/b72f5f4c73dc8cbdf06584a5e0ed9c0ea7f6b8d8/agent/train/native_collector.py)
- [Rust/PyO3 observation path](https://github.com/mwl313/PokeAgent3.0/blob/b72f5f4c73dc8cbdf06584a5e0ed9c0ea7f6b8d8/engine/src/python.rs)
- [Benchmark driver](https://github.com/mwl313/PokeAgent3.0/blob/b72f5f4c73dc8cbdf06584a5e0ed9c0ea7f6b8d8/scripts/bench_pa3_end_to_end.py)
- [Full Spec 1.1](https://github.com/mwl313/PokeAgent3.0/blob/b72f5f4c73dc8cbdf06584a5e0ed9c0ea7f6b8d8/docs/spec/fullspec-1.1-minidc-20261006/POKEAGENT_3_0_FULL_SPEC.md)
- [v1.1 Master Plan](https://github.com/mwl313/PokeAgent3.0/blob/b72f5f4c73dc8cbdf06584a5e0ed9c0ea7f6b8d8/docs/PokeAgent3_Ultimate_Throughput_Optimization_Plan_v1.1.md)
- [Sample Efficiency Research](https://github.com/mwl313/PokeAgent3.0/blob/b72f5f4c73dc8cbdf06584a5e0ed9c0ea7f6b8d8/docs/PokeAgent3_RL_Sample_Efficiency_Research_2026-10-08.md)

### 확인 수준의 명시

- **GitHub 코드 확인:** P0 통계 집계 구조, per-row small-tensor GAE, full-batch learner materialization, unused metadata GPU transfers, incomplete DDP, cohort actor structure.
- **GitHub JSON 실측:** 15.07/103.44/105.78 actor, 20.87/26.43 bounded PPO, 290.7s learner, 162.73 dual actor, 18.05GiB peak RSS, 10/16 readiness in reports.
- **미확인, 직접 miniDC에서 조사:** SSD swap/IO, 학습 시 GPU utilization, kernel/backward split, actual 2-GPU DDP, long-run steady state, performance gain of all proposed optimizations.
- **권고 설계:** phased profiling, columnar+streaming, DDP correctness, overlap, Rust encoder reuse, optional CUDA Graph, hardware telemetry. 미래 성능 개선 수치는 이 문서에서 보장하지 않는다.

---

## 19. 에이전트에게 그대로 전달할 실행 프롬프트

```text
PokeAgent 3.0 optimization task: execute the attached "Extreme Throughput Optimization Master Plan v2.0" against current GitHub branch optimization/pa3-realpolicy-throughput, known base HEAD b72f5f4. First fetch latest HEAD and reconcile any later user/agent changes; do not overwrite new work. Preserve the v1.1 changes and frozen 1,137-team data/model/PPO contracts.

Prioritize: (1) prove and fix microbatch-last-only metric/KL reporting and examine exact padding-weighting gradient equivalence, (2) profile 290.7 s learner phase into GAE, preparation, streaming, GPU forward/backward/optimizer, sync and staging, (3) implement and prove columnar/streaming PPO loader plus efficient GAE and metadata selection, (4) real two-GPU DDP all-in parity and benchmarks, (5) independent cohort CPU/GPU overlap, rolling slots, (6) Rust observation/packing, kernel and NUMA/PCIe/I/O frontier based on new profiler traces.

Run each experiment as a reproducible baseline versus isolated change, preserve exact sampled/recomputed joint logprob, legal masks, no hidden info leakage, full training-pool distribution, terminal reward attribution, true natural games and 4-epoch PPO. Do not alter the Rust game rules, neural architecture, frozen teams, global PPO semantics, installed CUDA/PyTorch/driver/power settings or other running services. Do not launch 100M training until 16/16 full engine readiness and explicit authorization. Avoid installed packages or unnecessary system configuration changes.

For every stage, commit and push verified code, regression tests, original raw JSON, manifest with model/data/Git SHA, stage-level CPU/CUDA time, actual matched games/decisions, memory/swap, error counts, actor-only and bounded PPO or operational all-in numbers. Re-run final clean-HEAD 3-repeat and 10k+ bounded benchmarks. Mark unfinished DDP and readiness explicitly, provide next blocker and the reason. Continue searching bottlenecks beyond 1,000 games/s, but never promise or fabricate a ceiling or outcome. Review the accompanying document for all exact gates, tests and allowed commands.
```

**전달 완료의 정의:** 에이전트가 이 파일을 읽고 현재 HEAD 확인, P0 정확성 테스트 작성, M0 learner instrumentation 착수까지 들어갈 수 있어야 한다. 이 문서를 만들었다는 것 자체가 코드를 수정하거나 miniDC 테스트를 실행했다는 의미는 아니다.
