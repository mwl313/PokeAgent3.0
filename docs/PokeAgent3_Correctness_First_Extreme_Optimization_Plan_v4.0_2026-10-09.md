# PokeAgent 3.0 — Correctness-First Extreme Optimization & Full-Stack Bottleneck Audit v4.0

**문서 목적:** miniDC 에이전트에게 전달하는 실제 구현, 계측, 검증, 롤백 지시서  
**작성일:** 2026-10-09 (Asia/Seoul)  
**검토한 GitHub HEAD:** `mwl313/PokeAgent3.0`, `optimization/pa3-realpolicy-throughput` @ `816707ecf8036ae493140774dc2d3b43e6e1a72e`  
**상위 계약:** `docs/spec/fullspec-1.1-minidc-20261006/POKEAGENT_3_0_FULL_SPEC.md`, `configs/train.yaml`  
**상속 계획:** `docs/PokeAgent3_Extreme_Optimization_Master_Plan_v3.0_2026-10-09.md`, v2.0 및 v1.1  
**임무:** 정확한 PPO 학습을 먼저 보증하고, 두 V100과 CPU 전체의 실제 활용 병목을 끝까지 찾아 동등한 계산을 최대한 빠르게 수행한다. **근거 없는 하드웨어 성능 부족 주장이나 속도 배수 약속 금지.**

> **절대 순서:** 학습 수학 및 single-GPU oracle 확정 → multi-GPU 수학, 통신, 동기화 교정 → DDP와 수동 All-Reduce 경쟁 → 진짜 all-in 성능 측정 → 전체 스택 프로파일링 및 공격적 최적화 → 깨끗한 재현 및 readiness 분리. 정확성 게이트를 통과하지 않은 속도 변경은 승격하지 않는다.

---

## 0. 이 문서를 읽은 에이전트가 즉시 할 일

1. `git fetch origin`, 현재 브랜치, HEAD, working tree, 현재 진행 중인 별도 변경 확인. 기준 SHA는 작성 당시의 고정 참조다. **더 새 커밋이 있으면 덮어쓰지 말고 변경 사항과 본 계획을 대조**한다.
2. 기존 v1.1, v2.0, v3.0의 성공적인 최적화, 원본 JSON, 팀풀, 테스트를 보존한다. 이 문서는 **대체 구현 또는 포괄적 리셋 지시가 아니다.**
3. **G0 correctness emergency:** `PA3Model._branch_stats`의 Entropy 및 Uniform-KL gradient 분리 버그를 독립 테스트로 재현하고 수정한다. 성능 실험에 앞서 수치 계약을 잠근다.
4. **G1 distributed correctness:** 실제 두 GPU에서 실패한 DDP collective 불일치를 최소 재현으로 수리하되, global advantage, LR clock, KL, GradScaler, 결과 집계를 함께 점검한다.
5. **G2 competing executors:** 자동 DDP와 수동 gradient All-Reduce를 **동일한 global batch, 동일 학습 수학, 동일 전체 수집 계약**에서 비교한다. 더 단순하고 빠른 쪽을 채택한다.
6. **G3 full-stack forensic profiling:** 각 CPU, Rust, Python, CUDA, NCCL, 메모리 전송의 임계경로를 계측한다. 모든 최적화 가설은 추정이 아니라 계측 가능한 병목과 A/B 결과로 승격한다.
7. 성능 개선의 목적은 raw games/s뿐 아니라 **현재 정책 학습 행/s 및 신뢰할 수 있는 PPO 업데이트 처리량**을 높이는 것이다. 100M 경기 본학습은 readiness 16/16과 별도 사용자 승인 전 절대 실행하지 않는다.

### 0.1 완료와 미완료를 혼동하지 말 것

| 영역 | `816707e` 기준 상태 | 소스 또는 측정 근거 |
|---|---|---|
| Fixed observation, packed candidate/prefix, packed rollout, FP16 actor | 이전 최적화 완료 | `docs/perf/ULTIMATE_FRONTIER_REPORT.md` |
| PPO 지표와 uneven micro gradient 계산 수정 | 수정 및 회귀 테스트 완료 | `V2_CORRECTNESS_AUDIT.md` |
| 실제 PA3-8M gradient parity 및 streaming equivalence | 제한된 fixture로 검증 | `V3_NUMERIC_PARITY.md`, `test_real_gradient_parity.py`, `test_streaming_equivalence.py` |
| Both-current-side 수집 계약 | 수정 완료, 추가 전용 regression 필요 | `NativeCollector._records_side()`, `V3_NUMERIC_PARITY.md` |
| Microbatch 1024 | 양쪽 좌석에서 속도 유리 | `V3_VRAM_BATCH_SWEEP.md` |
| Microbatch 2048 | 1024 대비 이득 없음, **채택 금지** | `V3_VRAM_BATCH_SWEEP.md` |
| Dual-actor only | 162.73 games/s, 과거 단일 좌석 수집 측정 | `ULTIMATE_FRONTIER_REPORT.md` |
| NCCL two-process probe | 성공, `NCCL_SOCKET_IFNAME=lo` 필요 | `V3_DDP_GLOBAL_GRADIENT.md` |
| Actual dual PPO | collective mismatch로 중단 | `V3_DDP_GLOBAL_GRADIENT.md` |
| GPU kernel/actor overlap, rolling slots, full-columnar, CUDA Graph | 미검증/미완료 | v3.0 open tasks |
| Full engine readiness | **10/16**, 실패 2,3,5,8,15,16 | `engine/TRAINING_READINESS.md` |

### 0.2 현재 속도의 올바른 기준

**새로운 양쪽 좌석 계약**, 단일 V100, 2,048 자연완결 경기, 1,024 env, 16 workers, `microbatch=1024`, streaming, checkpoint 포함, 3회 실행:

- **all-in median 19.365 games/s**, 각 경기당 약 26.87개의 현재 정책 learner row.
- **약 520 learner rows/s**. 동일 작업의 과거 단일 좌석 수집은 34.245 games/s, 약 460 learner rows/s였다.
- 양쪽 좌석 계약에선 2,048 경기당 learner rows 약 54,000~55,000, global 4,096 minibatch, 4 epochs, optimizer steps 56.
- Micro 1024는 9.25 GiB, micro 2048은 17.98 GiB reserved. 2048 median 19.318 games/s로 개선되지 않았다.
- 10,240경기 33.37 games/s 등 **v1/v2 값은 단일 좌석 계약**이다. 새 계약의 동일한 all-in 학습 성능과 직접 비교하거나 이어 붙이지 않는다.
- 2k 양쪽 좌석 수집의 GPU0 Actor-only 측정치를 PPO의 19 games/s와 구분한다.

**기준 A/B 명명:** `single_seat_legacy` vs `both_current_sides_correct`; 실측 결과의 모델, 팀, 규칙, 행 수, 양쪽 보상, optimizer step, checkpoint 포함 여부를 항상 명시한다.

---

## 1. 절대 변경 금지 계약 및 하드웨어

### 1.1 머신과 예산

- CPU: Xeon E5-2673 v4 × 2, 물리 40코어/논리 80스레드, NUMA 2개.
- GPU0: V100 PCIe 32GiB, NUMA0, 175W. GPU1: V100 PCIe 32GiB, NUMA1, 150W. 직접 P2P/NVLink 없음.
- `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=0`; NCCL local probe에서 `NCCL_SOCKET_IFNAME=lo` 작동 확인. 환경 변수는 프로젝트 자식 프로세스 범위에만 지정.
- GPU별 VRAM soft budget 28GiB, 전체 host RAM soft budget 48GiB, pinned staging 전체 1GiB.
- 기존 driver/CUDA/PyTorch pin, power cap, BIOS, kernel, 팬 제어, dsh-web, llama-swap 서비스는 승인 없이 변경/중지 금지.
- 현재 SSD는 학습 병목으로 관측되지 않았다: 디스크 read 0, swap 0, major faults 0, iowait 0. NVMe 구매로 속도가 빨라진다고 간주 금지.

### 1.2 모델, 게임, PPO

- PA3-8M 아키텍처, 96 padded entity tokens, full candidate semantics, Rust 배틀 결과, player-safe 관측, 숨겨진 정보 비노출, 1,137개 frozen team과 dataset SHA 유지.
- **같은 current policy 두 좌석은 모두 수집**, 한 게임은 한 번만 count, 각 좌석 마지막 요청에 상대 부호의 terminal reward. 실제 historical opponent 행은 학습에 저장 금지.
- Global minibatch 4096, PPO 4 epochs, Adam, FP16 autocast, FP32 probability/loss, GradScaler, GAE `gamma=1`, `lambda=.95`, reward shaping 없음, KL stop 0.03. Microbatch 크기만 정해진 A/B 계약하에서 바꿀 수 있음.
- 동일 iteration의 frozen current policy, frozen opponent snapshot, 전체 GAE 및 advantage normalization, true natural completion, 무손실 데이터 계약 유지.
- optimizer/scaler/scheduler/global committed matches, 통합 팀 샘플링과 정책 버전 검증 없이 관측/엔진/보상/학습 하이퍼파라미터를 바꾸지 않는다.

### 1.3 성능을 위하여 허용하지 않는 지름길

- 게임 조기 종료, 긴 게임 삭제, 합법 후보 상한 임의 축소, 불법 행동 무시, 약한 임의 정책 치환, 일부 메커니즘 생략, hidden info leak.
- 학습 행의 임의 drop, 두 좌석 대신 한 좌석으로 돌아가기, 기준 목표 4096을 임의 변경, PPO epoch 줄이기, gradient skip을 학습 속도 향상으로 주장하기.
- P2P 없는 두 VRAM을 가상 64GiB 단일 GPU라고 표현하기.
- 동일 조건이 아닌 수치를 속도 향상 배수로 나눠 발표하거나, CPU wall stage의 합을 CUDA kernel time이라고 부르기.

---

## 2. 실행 순서 및 게이트: 먼저 학습이 맞아야 한다

| 게이트 | 이름 | 핵심 통과 기준 | 다음 단계 |
|---|---|---|---|
| **C0** | Policy loss gradient audit | Entropy/KL 실제 gradient 복구, 단일 수학 oracle 통과 | C1 |
| **C1** | Single-GPU reference | real PA3 loss, gradient, optimizer, replay, 양쪽 보상 검증 | D0 |
| **D0** | Distributed protocol design | 공통 4096 배치 및 count, LR, advantage, KL, AMP 계약 정의 | D1 / M1 |
| **D1** | DDP correctness | 불균등/빈 rank, 모든 collective, full one-step parity | D2 |
| **M1** | Manual All-Reduce correctness | 동일 조건 full one-step parity | D2 |
| **D2** | Two-GPU A/B | 2k×3 및 긴 동일 총 경기 수 benchmark, 동일 수학 | F0 |
| **F0** | Full-stack forensic profile | CPU/GPU/PCIe/NCCL stage와 critical path 지도 | P1 이후 |
| **P1~P9** | Bottleneck attacks | 단계별 고립 A/B, 0 의미 변화, 반복 검증 | F1 |
| **F1** | Operational frontier | clean HEAD, resume, long soak, real all-in 측정 | 완료 |
| **R** | Full rules readiness | 16/16, 별도 트랙 | 본학습 승인 전 |

각 게이트 실패 시 그 **게이트의 오류만** 수정하고 동일한 테스트로 재확인한다. 정확성 게이트 실패 상태에서 throughput 순위 선정 금지.

---

## 3. C0 — PPO 손실 수식 전체 감사: 즉시 수정

### 3.1 확인된 위험: 정규화항의 gradient가 끊어짐

`agent/model/pa3_model.py::PA3Model._branch_stats()` 현재 형태:

```python
entropy = -(probs.detach() * log_prob.detach()).sum(dim=-1)
mean_log_prob = (log_prob.detach() * mask).sum(dim=-1) / safe_k
```

`agent/ppo/learner.py::_forward_terms()`의 PPO loss에는 `-entropy_coefficient * entropy + uniform_kl_coefficient * uniform_kl`가 더해진다. 그런데 이 두 계산은 모두 `detach()`로 끊겨 있어 **현재 정규화 항이 scorer에 gradient를 전달하지 못한다.** 보고용 값은 계산되더라도 실제 학습 목적함수에서 기능하지 않는다.

**구현 지시:**

1. 학습 경로의 entropy에 `probs`와 `log_prob`가 gradient를 가진 상태로 들어가게 한다. `probs.detach() * log_prob`처럼 일부만 detach하면 entropy 유도식이 바뀌거나 gradient가 소실될 수 있으므로 analytic derivative 및 autograd reference로 검증.
2. `KL(U || pi)`는 합법 후보에 한해 `-log K - mean_{legal}(log pi)`이며 differentiable `log_prob`에서 계산. 분기별 K<2이면 0.
3. 별도의 monitoring/output/recording 경로에서만 `.detach()`. Actor 추론은 `inference_mode` 그대로.
4. FP32 policy probability/loss 유지, invalid mask 0 probability 유지. Padded branch가 NaN이나 gradient를 만들지 않는지 검사.
5. 정책 손실항, entropy 정규화항, KL 정규화항, value 손실항을 **독립적으로** parameter-wise gradient 비교. 합산 시 계수와 부호 확인.
6. 정책 logits 직접 지정한 작은 2/3/5 후보 수에서 finite-difference 검증. 균등 정책 및 편향 정책, singleton, 모든 padding, action mask edge case 포함.
7. 기존 weight와 fixture 기준 **수정 전/후 report scalar parity**와 **예상되는 gradient 변경**을 별개로 기록. 이는 속도 최적화가 아닌 실제 학습 correctness correction이다.

**수락 테스트 이름 제안:**

- `test_entropy_loss_produces_nonzero_logit_gradient`
- `test_uniform_kl_loss_produces_nonzero_logit_gradient`
- `test_entropy_kl_gradient_matches_finite_difference`
- `test_singleton_and_padded_branch_zero_gradient`
- `test_all_illegal_mask_is_never_sampled_or_used`
- `test_policy_value_loss_and_logprob_parity_after_fix`

### 3.2 학습 수식 전 항목 전수 감사

- PPO clipped surrogate `min(r*A, clamp(r)*A)`의 부호, clipped policy gradient, actor mask와 `ratio=exp(new-old)`.
- `approx_kl=(r-1)-log(r)`, numerator/denominator, epoch stop 0.03.
- Value half-MSE, critic-only 행 포함 여부, actor-free 행 처리, terminal returns 및 GAE 부호.
- Entropy와 Uniform-KL 정의, branch 정규화, `K=1`, invalid candidates, FP32 loss.
- Advantage 계산 전체 iteration actor-only 분모 및 `std(unbiased=False)`, zero variance 기준.
- FP16 GradScaler, NaN/Inf 처리, clipping 이후 optimizer update, skipped step 통계, LR clock.
- 동일한 초기 파라미터, action index, old logprob, PPO minibatch permutation, seed와 policy version 사용.
- Dual rank와 single rank의 loss scalar 값이 일치해도 gradient가 같을지 별도 확인.

### 3.3 데이터 및 배틀 경험 계약 전수 감사

- 동일 자연완결 배틀의 2개 `side` trajectory에서 terminal reward가 반대 부호이며 `match_id`는 하나다.
- preview/normal/replacement, forced singleton, target-dependent choices, 두 sides의 관측 접근권한, team/player seating.
- `request_index` 기준 GAE 연쇄가 `match_id,side` 경계 밖으로 넘어가지 않는다. Incomplete/operational-aborted trajectory가 정상 완료로 계산되지 않는다.
- `collect_both_sides_when_current_self_play`일 때만 2개 current policy 행 저장. 실제 historical policy actor row 제외와 상대 메타데이터 확인.
- 저장 관측과 정책이 본 candidate, selected prefix, old logprob가 동일 요청에서 왔는지 stratified sample/recompute로 재검증.
- 새 회귀 테스트가 **수정 전 코드에서 실패, 수정 후 통과**한 결과를 보고서에 남긴다.

**C0 Gate:** `pytest tests/agent`, 독립 objective tests, fixed vs packed contract, end-to-end recompute 모두 통과. 아직 miniDC에서 수행되지 않았으므로 본 계획에서 PASS를 가정하지 않는다.

---

## 4. C1 — 단일 GPU를 분산 계산의 정답 oracle로 확정

같은 일련의 실제 PA3-8M 학습 행을 준비해 다음을 검증한다.

1. **FP32 reference** single-device, same initial cloned `state_dict`, same optimizer state, 같은 정렬과 shuffled indices.
2. Full minibatch 4096 vs micro splits 64/128/256/512/1024 및 uneven/actor-free/padding, clipping 전 loss 및 원시 gradient.
3. Reconstructed gradient의 encoder, scorer, value head 별 absolute/relative norm 및 최대 좌표 오차; 0 기준값에는 절대 오차.
4. Real FP16 autocast + GradScaler에서도 동일한 의미를 재현. 비결정적 커널로 bitwise exact를 강요하지 말되 사전 정의한 허용 오차 및 반복 변동 기준.
5. Adam 첫 업데이트와 2~4 연속 업데이트, optimizer `exp_avg`, `exp_avg_sq`, `step`, 모델 가중치, gradient clip 결과와 scaler 확인.
6. 체크포인트 저장과 재개 후 같은 입력과 seed에서 동일한 다음 업데이트. 테스트 fixture에서 정책/상대 모델의 동일 snapshot을 사용.
7. 수정 후 `PA3Model` 정규화 loss 값 자체와 gradient가 각각 올바른지 체크.

**권장 수치 기준:** FP32 raw gradient 최대 오차 `atol=1e-6`에서 시작하되 연산 규모에 맞춰 max relative norm `<=1e-4`의 실제 허용 기준을 문서화한다. FP16은 같은 reference에 대한 사전 등록 tolerance를 별도 보고. 우연한 aggregated KL 일치는 gradient parity 증거가 아니다.

**C1 Gate:** single-GPU full-loss/gradient/update reference 확정. 그 다음 분산 실행을 통과 판정한다.

---

## 5. D0 — 분산 학습의 전역 수학과 동기화 계약

### 5.1 정확한 목적함수

전체 minibatch 유효 actor 수를 `A`, 유효 value 수를 `V`로 정의. rank `r`의 actor 항 합을 `S_actor,r`, value 항 합을 `S_value,r`로 정의:

```
GLOBAL_LOSS = sum_r(S_actor,r)/max(A,1)
              + value_coef * sum_r(S_value,r)/max(V,1)
```

`S_actor`는 policy clipping, differentiable entropy, differentiable uniform-KL를 포함한 actor 합이다. `A`, `V`는 rank 전체에서 한 번만 집계한다.

- **표준 DDP 평균 gradient:** rank별 backward loss에 `world_size`를 곱해 `world_size*(S_actor,r/A + value_coef*S_value,r/V)`를 사용한다.
- **수동 `all_reduce(SUM)` gradient:** rank별 backward loss에 `world_size`를 곱하지 않는다. Local gradients를 SUM하여 글로벌 gradient를 얻는다.
- 각 구현은 동일한 **4096 real-valid global batch를 가능한 한 유지**한다. 마지막 불완전 배치에서만 실제 부족분을 masking. 두 rank 중 하나가 끝났다고 real row를 폐기하지 않는다.
- Current iteration actor 전체의 advantage normalization은 **rank별이 아니라 글로벌**이다. Rank별 GAE 계산은 허용하지만 actor의 `sum`, `sum_sq`, `count` 전역 집계로 공통 mean/std를 적용. 안정성이 필요하면 2-pass variance/Welford.
- `global_committed_matches=sum(all_ranks_completed_natural_matches)`를 rank마다 동일한 절대 LR clock으로 전달하며 두 좌석 row 수를 경기 수로 오인하지 않는다.
- Epoch마다 글로벌 `KL_sum`과 `actor_count`로 early stop 결정을 하나만 내린다. 모든 rank가 같은 epoch 수와 동일 step 여부를 본다.
- GradScaler scale, overflow detection, optimizer applied/skipped, global norm clipping, scheduler와 Adam state는 rank 간 동일해야 한다.
- 모든 rank는 동일 policy weights와 frozen opponent version을 사용해야 하며 DDP rank 간 replay, value/bootstrap 타임라인이 섞이면 안 된다.

### 5.2 단일 GPU oracle과 동일한 global batch 구성

**현재 코드 위험:** rank마다 개별 shuffle + `per_rank_minibatch=2048`는 두 rank의 행 수가 다를 때 마지막 global batch가 단일 프로세스의 4096 연속 샘플과 달라질 수 있다.

권장 기본 구현:

1. 모든 rank의 현재 iteration row count를 공유한다.
2. 전체 row의 `rank_id, local_row_id`로 구성된 **global permutation**을 seed 하나로 생성한다.
3. 이 순서를 global 4096의 논리적 minibatch로 자른 뒤, 각 minibatch의 2048-row shard를 rank별로 할당한다. 필요하면 compact CPU rows를 NUMA-local stage에서 교환하되 정확한 결과를 위해 데이터 복제/누락 금지.
4. 마지막 global minibatch는 valid row만 계산하고 고정된 masked 행으로 패딩한다.
5. Global row identity와 policy version, 수집된 매치 수는 별도 보존. 실제 수집 비용과 rank 간 데이터 재분배 비용을 전부 측정한다.

**빠른 1차 구현:** rank-local 순서를 유지하되 고정 step protocol을 사용해 deadlock부터 해결할 수 있다. 다만 이 상태의 단일 GPU one-step parity는 **실제로 동일하게 편성한 rows**를 기준으로 검사한다. 순열이 다르면 gradient까지 같은 것을 기대하면 안 된다.

---

## 6. D1 — PyTorch DDP 오류를 완전히 수리

### 6.1 현재 구현에서 실제 확인된 결함

| 파일 / 지점 | 위험 및 증거 | 수정 |
|---|---|---|
| `PPOLearner.update_ddp` | rank-local padding-only minibatch는 backward 생략하여 collective mismatch | 글로벌 step plan과 모든 rank 참여 |
| `DDPCommunication.context` | sync counter가 `synchronise_every` 기준인데 유효 micro 수가 더 적으면 마지막 sync가 생략 | **고정 micro-step 수**, 마지막 step 명시 sync |
| `prepare_streaming` | rank별 advantage normalization | 글로벌 sum, sum_sq, count |
| `run_ddp_ppo.py` | rank별 `committed_matches`로 LR clock 차이 발생 | global natural matches 및 broadcast |
| `update_ddp` | epoch KL이 rank-local이고 early stop도 rank-local | 전역 KL 및 공동 epoch stop |
| Rank optimizer/AMP | 각 rank의 overflow/step skip이 달라질 수 있음 | global finite/overflow 합의, 동일 scaler update |
| `PA3Model.forward` / DDP | `find_unused_parameters=True`에서 반환 dataclass graph 탐지 위험 | Tensor-only learner forward PoC, parameter-used 검사 |
| `run_ddp_ppo.py` worker/parent | rank0만 결과 JSON 출력, parent는 모든 rank의 JSON 기대 | 양쪽 JSON 출력 또는 rank0 gather 통합 |
| `run_ddp_ppo.py` stdout pipe | parent가 자식 stdout을 순차적으로 `communicate` | rank별 로그 파일 또는 동시 소진 및 timeout |
| `run_ddp_ppo.py` param check | parameter 합계로 동등성 검증 | 전체 named tensor SHA256 및 optimizer state 검증 |
| Rank checkpoint | rank0 learner state만 저장, rank별 RNG/rollout cursor/metadata 추가 검증 필요 | resume contract 및 rank-local state 보존 |

### 6.2 DDP fixed-step protocol

1. 모든 rank가 같은 `global_minibatch_id`를 처리한다.
2. Global `(A,V)`와 valid row 수를 all-reduce한다. 두 값 모두 0이면 **모두 동일하게 optimizer step skip**.
3. 각 rank가 정확히 같은 micro-step 수를 실행한다. 자기 데이터가 없으면 **DDP forward를 포함한 graph-connected masked dummy micro**를 구성하며 loss는 0이지만 모든 필요한 파라미터 경로는 그래프에 연결돼야 한다.
4. DDP `no_sync`는 non-final micro의 **forward와 backward 전체**를 감싼다. 마지막 micro는 반드시 동기화 경로에서 실행한다. 마지막 micro가 dummy라도 참여시킨다.
5. 작은 micro에 가변 사용 파라미터가 생기는 경우 `find_unused_parameters` 유효성 확인. 반환은 가능한 Tensor-only tuple로 구조 단순화하고 실제 gradient parity 확인.
6. GradScaler scale 일정하게 유지하면서 accumulation 완료 → unscale → global finite check → global grad clip → optimizer step. 하나라도 overflow이면 모든 rank가 같은 step을 skip한다.
7. Global stats epoch KL all-reduce 및 공통 stop. Rank별 optimizer steps, scaler scale, LR, model digest가 일치하는지 assert.
8. 한 rank 오류 발생 시 collective watchdog으로 **해당 벤치마크 자식 그룹만** 종료하고 JSON에 실패 stage와 communicator trace 기록. 절대 서비스/다른 GPU 작업 종료 금지.

**주의:** masking으로 loss가 0이라도 `0 * NaN`은 NaN이다. Dummy inputs는 실제 모델에서 안전한 legal placeholder로 구성하고 zero gradient가 finite인지 검증한다. 명시적으로 사용되지 않은 파라미터의 gradient 0 경로도 보증한다.

### 6.3 DDP 초기화 및 프로세스

- `NCCL_SOCKET_IFNAME=lo`를 **자식 프로세스에만** 유지. 이 설정으로 minimal NCCL all-reduce 작동이 기록됨.
- `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=0` 유지. 시스템 NCCL이나 드라이버 업그레이드 금지.
- Process-group timeout을 예컨대 진단용 90초로 **명시적으로 설정**하고, 운영 테스트에서는 예상 가장 느린 단계와 맞춰 합리적 상향. 실효 timeout을 제어하기 전 강제 장기 hang 금지.
- Trace 및 stuck collective에는 `TORCH_NCCL_DESYNC_DEBUG=1`, `TORCH_NCCL_TRACE_BUFFER_SIZE=2048`, `TORCH_NCCL_DUMP_ON_TIMEOUT=1`, `TORCH_NCCL_ENABLE_MONITORING=1`, `TORCH_DISTRIBUTED_DEBUG=DETAIL`을 진단 실행에서만 사용. 지원 여부는 현재 설치 PyTorch 기준 확인.
- rank0와 rank1의 로그를 동시에 drain하거나 독립 파일에 쓰고, parent가 실패를 감지했을 때 자체 자식만 정리. Rank0에 모델/optimizer 진단이 도달하지 못해도 원인 stack을 보존.
- `numactl --cpunodebind/--membind`로 각 rank를 로컬 GPU NUMA에 배치. NUMA가 실제 bandwidth 병목인지 별도로 실측.

### 6.4 DDP-specific 검증 fixture

- `{2048,2048}`, `{2048,100}`, `{2048,0}`, `{0,2048}`, actor masks `{all, none, asymmetric}`, 마지막 batch padding, optimizer skip, KL early-stop.
- 실제 PA3-8M, 동일 4096행 full reference, 한 번 및 연속 3번 optimizer update. 두 rank의 gradients, global clip norm, full model weights, Adam step/exp_avg/exp_avg_sq, scale, LR/epoch KL 체크.
- DDP forward `find_unused_parameters` 및 반환 그래프, static_graph 가능 여부는 사용된/미사용 파라미터 측정 후 결정. 절대 안정성 검증 없이 `find_unused_parameters=False` 승격 금지.
- 결과 집계 rank별 source 행 수, actor valid/value valid, step index, sync count, collective sequence digest, gradient SHA. Row drop 0, no illegal action 0, zero operational error.
- 정량 정확성 oracle이 PASS되기 전에 speed measurement 채택 금지.

---

## 7. M1 — 수동 Gradient All-Reduce 대안 실험

**목표:** DDP의 reducer hook과 가변 `no_sync` 순서를 회피하면서 같은 글로벌 PPO 목적함수를 학습. 두 카드가 논리적으로 한 번의 optimizer update를 수행하게 한다. 두 VRAM은 통합되지 않는다.

### 7.1 수동 구현 프로토콜

1. 동일 초기 모델, Adam/scaler/scheduler 상태와 전역 4096 logical minibatch.
2. 각 rank에서 DDP wrapper 없이 `model.forward()` + gradient accumulation을 수행한다. 각 local loss는 `S_actor_local/A + value_coef*S_value_local/V`이며 **world_size 배수 없음**.
3. 동일 GradScaler scale로 backward를 끝내고 `unscale_` 한 뒤 모든 파라미터의 gradient를 **고정 name/order**로 FP32 contiguous flat buffer에 넣는다. `grad is None`이면 0으로 채운다.
4. 글로벌 finite check 후, `dist.all_reduce(flat_grad, SUM)`을 정확히 한 번 또는 정당한 bucket 수만큼 호출한다. DDP와 달리 기본 평균이 아니므로 다시 `/world_size` 하지 않는다.
5. 전역 gradient를 각 rank의 parameter `.grad`로 연결하고 동일한 global clip norm을 적용한다.
6. 글로벌 overflow/skip 결정을 공유하고 같은 Adam step + scaler update + LR apply. 모델/optimizer digest를 모든 rank에서 확인.
7. rank-local padding-only minibatch도 통신은 참여하며 zero gradients. global valid 수 0이면 두 rank 모두 step skip.
8. 수동 bucket allreduce와 1회 flat allreduce 각각의 latency, VRAM peak, CPU staging bandwidth, synchronization overhead 비교.

### 7.2 DDP vs 수동 방식 채택 기준

- 동일 **총 자연완결 경기 수**, 동일 both-current-side 수집, 동일 global 4096 batch, 동일 4 epochs, 같은 초기 모델과 학습 데이터. 자연완결 한 경기당 rows 분포도 함께 기록.
- 2k×3 반복은 **총 합계 2k**임을 명확히 하는 비교 패널: 예를 들어 single 2048 vs dual 각 1024. 별도로 긴 **총 합계 10,240** 패널. Rank별 2k를 합쳐 4k로 만든 실행을 single 2k와 직접 비교 금지.
- 벤치마크의 분모에는 process startup, collect, GAE/normalize, PPO, gradient sync, checkpoint, logging/metrics 포함 범위를 분명히 한다.
- DDP의 통신 계산 중첩 능력과 수동 방식의 통신 단계 후행 오버헤드를 profiler로 구분.
- 두 방법 모두 오류 없고 gradient/update parity를 통과해야 하며, 반복 중앙값과 분산/최악 성능을 보고. 승자가 불명확하면 더 단순하고 안정적인 구현 우선.
- 모델 비교 시 `sum(params)` digest 단독 금지. Canonical tensor name + shape + dtype + raw bytes로 SHA256. optimizer moments와 scaler/scheduler도 검사.

### 7.3 남은 대안: Actor/Learner GPU 역할 분리

DDP와 수동 allreduce가 통신 병목으로 기대보다 느리면, GPU0 learner, GPU1 actor 역할 분리도 제어 실험. 단, **현재 PPO의 frozen-policy per iteration과 on-policy 사용을 유지**해야 한다. 무작정 수집과 업데이트를 시간상 중첩해 오래된 행동 정책의 경험을 다음 업데이트에 섞지 않는다. 동시 오버랩 실험은 frozen snapshot과 버전 경계를 지키거나 PPO 학습과 직접 독립된 evaluation으로 제한한다.

---

## 8. F0 — 전 스택 병목 계측: 빠짐없이 추적

**목표:** '느리다'를 코드에서 정확히 어느 함수, 어떤 호출 빈도, 어느 버스, 어느 메모리, 어떤 GPU kernel이 처리량을 제한하는지로 치환한다. 가장 비싼 세 원인을 1회 profiler로 단정하지 말고 독립 계측과 재현으로 확정한다.

### 8.1 계측 단계와 필수 숫자

| 층 | 샅샅이 측정할 항목 | 최소 산출 |
|---|---|---|
| **시작 및 정책 버전** | 프로젝트 import, Rust engine init, Dex/team load, CUDA context, DDP process group init, model copy/broadcast, snapshot freeze | startup wall, per-rank dependencies, model SHA |
| **게임 생성** | reset, team RNG, request info, native observe, Python parser, candidate prefix Rust/Rust-Python crossing, GPU inference, D2H, assemble, Rust step, terminal/drain | 각 stage cumulative wall, calls, bytes, requests, active-env occupancy |
| **Python main thread** | GIL 보유 구간, object allocation, list/dict per-row loop, Python/C++ transition, garbage collection, reference count and allocator stall | wall vs thread CPU, py-spy/perf if already present |
| **Rust CPU** | threadpool scheduling/Rayon worker wait, per-env engine progress, observation encoder cache, object copies, memcpy, heap alloc, lock contention | per thread CPU, request/turn, p50/p95 latency |
| **CUDA actor** | each encoder invocation, branch scorer, GRU, masked logits, multinomial sampling, H2D/D2H and sync | GPU active duration, launch count, idle gaps, SM and memory-bandwidth occupancy |
| **Rollout memory** | record packed rows, observations/metadata/candidate table, malloc/realloc, trajectory indexing, Python heap | bytes/row, RSS/PSS, allocate/copy call counts |
| **GAE / prepare** | group/sort trajectories, GAE loop, actor stats, normalization allreduce, batch materialization, collate, row shuffle | wall, CPU cycles, copies and page faults |
| **GPU learner** | forward encoder/scorer/value, backward embeddings/scatter/SDPA/GRU/FFN, optimizer, global grad clipping, scalar `.item()`, kernel launch stalls | CUDA kernel table and timeline, CPU launch vs GPU work, GPU busy/idle |
| **Inter-GPU** | NCCL init, host-mediated transfer, count reductions, grad bucket sync, all-gather, wait for slow rank | collective name/size/calls/bandwidth/wall, per rank idle time |
| **Memory paths** | host RAM bandwidth/NUMA remote pages, CPU staging, pinned/pageable H2D, PCIe link speed/width, VRAM reserved/allocated/highwater | per direction bytes/s, RTT, overlapping timeline, swapping |
| **운영 비용** | metric writer, checkpoint serialize, fsync, verification/recompute, eval, cleanup/restart | bounded vs operational all-in with includes manifest |

### 8.2 CPU/GPU 시간 해석 규칙

- `perf_counter()`는 CPU thread의 호출 구간 시간이다. 비동기 CUDA dispatch 때문에 내부 GPU kernel wall과 동일하지 않다.
- CUDA active execution time은 `torch.profiler` timeline과 CUDA event scoped measurements로 분리한다. CUDA event로 전체 wall을 감싼 값은 GPU pure compute utilization이 아니다.
- CPU `cuda_tensor.item()`, `.cpu()`, `.tolist()`, `bool(cuda_tensor.any())`, `torch.cuda.synchronize`, implicit sync를 **파일별 grep 및 profiling stack**으로 확인.
- GPU stage별 CPU launch 지연과 GPU 실행의 간극을 timeline으로 확인. `nvidia-smi` GPU 활용률 순간 스냅샷만으로 원인 판정 금지.
- `torch.profiler`는 **warmup 이후의 제한된 active step**만 추적. 무차별 1.87GB trace 재생성 금지, compact per-kernel summary와 선택 window trace만 보존, 대용량 trace는 `.gitignore` 처리.
- profiler on/off 두 조건을 별도 수행해 계측 오버헤드를 명시한다. 필요한 경우 `record_shapes`, `profile_memory`, `with_stack`은 각 별도 짧은 실험에만 켠다.

### 8.3 동시에 수집해야 할 시스템 telemetry

- `nvidia-smi --query-gpu=... --format=csv`와 `nvidia-smi dmon`으로 **실행 중** per GPU SM, memory util, VRAM, power cap/current, clocks, perf state, throttling reason, temperature, PCIe RX/TX가 지원된다면 수집.
- `vmstat`, `/proc/$PID/io`, `/proc/$PID/status`, `numastat -p`, `pidstat`/`iostat`가 설치돼 있으면 사용. 권한/도구가 없으면 absent로 표기, 임의 설치 금지.
- CPU cores, process/thread affinity, NUMA0 vs NUMA1 페이지, page faults, per worker CPU utilization, IPC/scheduling stall을 구분.
- `torch.cuda.memory_summary()`/`max_memory_reserved/allocated`와 RSS/PSS/USS를 rank마다 기록. Python `ru_maxrss`는 누적 peak라 반복 비교는 isolated subprocess로 측정.
- Torch allocator fragmentation: allocated vs reserved, active vs inactive split, cache/reused buffers; OOM이 없고 VRAM이 남는다고 kernel 효율이 좋다는 뜻이 아님.
- Rust Rayon worker count와 env count를 고정한 뒤, 각 stage가 `CPU compute`, `GPU busy`, `GPU sync` 중 무엇을 기다리는지 표시.

### 8.4 병목 원인 트리 및 가설 기각

- CPU-bound이면 CPU 모든 코어가 바쁜지 vs 메인 스레드 한 개가 바쁜지, GIL인지 Rust worker idle인지 검증.
- GPU-bound이면 math unit, memory bandwidth, kernel count/launch, embedding backward scatter/SDPA/FFN 각각 확인.
- GPU idle이면 actor prefix D2H sync 및 Python parse/pack/round barrier, minibatch preparation으로 대기 구간 추적.
- NCCL-bound이면 GPU별 계산 속도 차이와 gradient bucket wait, host-staged PCIe 대역폭을 분리.
- RAM-bound이면 Rust/Python allocation bandwidth와 NUMA remote overhead, page cache 여부 검사. Swap과 disk IO가 0이면 SSD 원인 배제.
- 데이터가 부족해 작은 GPU batch가 생성되는 경우 active decision batch 크기 분포와 cohort drain tail p50/p95, 요청 종류에 따른 branch-group fragmentation을 확인.

### 8.5 반드시 별도 계측할 게임 기반 지표

- `active_envs_per_round`, `decision_requests_per_round`, `branch_count_group_size`, `batch_size_distribution`, `minibatch_rows`, `rows_per_match`, `turns_per_match`, `natural_completion`.
- cohort 마지막 25%에서 요청 배치가 얼마나 줄어드는지, barrier 때문에 기다리는 시간, idle GPU window.
- Actor-only games/s, decisions/s, current-policy learner rows/s, actual GPU policy forward invocations/s.
- PPO-only rows/s, optimizer steps/s, kernel time / row, all-in games/s, complete eval included operating rate.

**F0 Gate:** 코드의 주요 함수 또는 cross-device 단계를 모두 category에 배정하고, wall/critical path를 설명하지 못하는 unaccounted 구간을 좁혀 결과 보고. **임의 100% 합계로 맞추거나 GPU 비동기 시간을 이중 합산하지 않는다.**

---

## 9. P1 — GPU Learner compute 공격

1. **1024 vs 2048은 결론이 이미 나왔다:** 양쪽 좌석 계약에서는 micro 2048의 games/s 개선이 없다. 기본 채택은 1024 검증값으로; DDP의 rank-local micro는 글로벌 4096 계약 아래 별도 A/B.
2. GPU0과 GPU1 각각 pure forward/backward microbench 256/512/1024/2048, 제어된 batch distribution, `torch.cuda.Event`와 profiler. GPU1의 150W cap 및 thermal clock을 실측, cap 변경 금지.
3. `embedding backward scatter`, `SDPA math`, `FFN GEMM`, `GRUCell`, candidate projection, `masked_log_softmax` 및 `gather`를 개별 kernel time+launch count로 분석.
4. CUDA Graph / `torch.compile`은 **실행 중인 기존 pin 스택에서 가능성이 검증될 때** 별도 branch PoC. 정적 shape bucketing, allocated input/output and RNG semantics, gradients/scaler, memory residency 확인. Kernel 캐시 생성/재컴파일 비용 포함한 전체 PPO 속도로만 승격.
5. `torch.optim.Adam`의 foreach/fused 사용 가능 여부는 **현재 PyTorch+sm70에서 지원되는 조합부터 탐색**. Optimizer result/state parity가 없으면 보류. 다른 optimizer 교체 금지.
6. Gradient norm 계산과 skipped-step 판단에 포함된 GPU→CPU sync, per-minibatch tensor indexing, candidate conversion, repeated allocate/copy를 probe. 다양한 AMP scale 조건에서 overflow 체크.
7. Double/triple-buffered fixed-shape forward workspace와 `torch.empty` 재사용은 현재 28GiB GPU 예산 내 실험. 메모리 점유만 늘고 total wall이 개선되지 않으면 폐기.
8. `torch.backends.cuda.sdp_kernel` 또는 kernel backend 후보는 **sm70과 실제 설치 패키지**에서 가능한지 확인, FP32 critical math는 보존. FA2 등 지원되지 않는 최신 커널을 임의 설치 금지.

**P1 수락:** pre-fix canonical baseline과 혼합하지 않는 same-correctness version, GPU kernel count/active time 및 PPO full wall 개선을 함께 입증.

---

## 10. P2 — Rust 배틀 엔진, 파이썬 및 관측 변환 공격

1. `engine/src/python.rs::observe_fixed_batch`에서 batch마다 `Encoder::from_validated_dex` 재구성 여부와 비용을 profiler로 확인하고, immutable Dex 기반 캐시를 안전하게 재사용할 수 있으면 변경.
2. Observation 41,954 bytes/view 실측: fixed wire와 ragged sections 데이터 복사, alignment, allocation, parser, tokenized converter 각각의 단계 분해.
3. `[B, tokens, features]` packed data를 numpy view 및 최소 copy 경로로 변환. Python typed object 및 `dict/list` per-row 생성 감소.
4. 무손실 관측 패킹만 기본 허용. f32→f16 transport는 실제 정책 입력 수치가 달라지므로 별도 experiment 및 schema, parity gate가 필요한 변경으로 취급.
5. candidate prefix table, selected actions, packed prefix slab 처리에서 Rust→Python 왕복 호출과 GPU→CPU sync 병목 제거 가능성. Request-dependency legality 유지.
6. 모든 5종 ragged sections가 wire에 포함되지만 네트워크가 직접 소비하지 않는 문제는 **관측 완전성 연구**로 별도 감사, 임의 제거하거나 hidden 정보를 추가하지 않는다.
7. Legacy vs optimized observation and candidates의 bit/field parity, all legal prefix oracle, sampled vs recomputed joint logprob, team/match identity, hidden info 검사.
8. 각 Rust worker의 lock contention, thread dispatch, allocator, Rayon idle를 확인. `Rust step_batch` 자체가 과거 actor wall 1.6%이므로 관련 profiler 없이 game core 최적화부터 시작하지 않는다.

**P2 수락:** 참조 engine state/observation/action 정확히 일치하면서 Actor-only 및 실제 PPO 전체 벽시계에서 측정 가능한 향상.

---

## 11. P3 — Rolling Slots: 코호트 drain-tail 제거

**현 코드:** `NativeCollector.collect()`는 한 코호트 전체가 종료할 때까지 `while not all(finished)`로 기다린 후 다음 cohort를 reset한다. 환경 512/1024/2048만 늘렸을 때 속도는 대체로 98~104 games/s로 변화가 없었다. **단순 env 증설은 검증된 해결책이 아니다.**

1. 먼저 per round `active_envs`, GPU actor request batch, tail wall, longest games, request kind distribution을 측정해 실제 idle gap을 확인.
2. `terminated`된 game slot을 **다음 round 시작 전에 즉시 안전하게 새 팀/seed/역할로 reset**. `slot_id`와 `match_id`를 구별하고 각 매치의 trajectory와 side, terminal reward, request_index와 RNG stream을 별도 유지.
3. PPO iteration policy snapshot은 reset 후에도 동일 frozen version을 계속 사용. 종료 조건은 min completed natural matches **및** min current-policy decisions를 만족한 뒤 새 reset만 멈추고 열린 게임을 종료할 때까지 drain. Overshoot는 보존.
4. 비교 실험에서 매치 길이를 강제로 잘라내거나 truncation을 정상 game count로 기록 금지.
5. 팀 샘플링 분포가 장기 게임/빠른 게임에 따라 편향되지 않도록 frozen random stream과 동일 match assignment panel로 비교. 동시 환경 수는 1024로 고정한 1차 A/B.
6. 실제 종료 경기, 승패, 각 side의 GAE terminal 수, reward sum, rows/match, team coverage, match IDs, policy IDs가 참조와 일치하는지 확인.

**P3 수락:** tail time과 GPU batch occupancy 개선이 실제 Actor-only + all-in 속도에 연결되어야 한다. Game semantics/보상, 게임 길이 분포 보존.

---

## 12. P4 — 독립 코호트 CPU/GPU 파이프라인 중첩

1. 현재 `observe → H2D → encode → sample → prefix candidates → step` 임계경로를 분해. 같은 게임 안의 prefix dependency는 병렬화하면 안 된다.
2. 두 개 이상의 **독립 코호트**를 producer/consumer 구조로 분리: CPU Rust prepare/candidates, GPU inference, D2H/step, recording 단계가 서로 다른 코호트에 대해 겹치도록 한다.
3. `2×512`, `4×256`, `1×1024`, 조건부 다른 분할을 비교. **총 동시 슬롯 수는 같게** 유지하여 단순 env 증가 착시를 제거.
4. bounded pinned staging 메모리(총 1GiB 이하), NUMA locality, CUDA streams, events, ownership, output buffers/IDs를 명시. Copy가 진짜 GPU compute와 overlap하는지 profiler로 확인.
5. per-cohort frozen model version, global RNG order, self-play both seats, prefix legal masks, actor logprobs, trajectory/terminal record 보존.
6. queue depth를 무조건 늘려 VRAM만 채우지 않는다. 각 단계 idle-gap 감소와 total wall 증가가 있는지 정량화.
7. Python GIL과 Rust `release_gil`, per-thread CPU utilization 및 lock contention 검사. 두 GPU를 같이 쓰면 NUMA/메모리 대역폭 경합을 함께 측정.

**P4 수락:** GPU actor busy time 중첩 증거 및 same-total-games actor + PPO all-in 개선. 원인을 확인하지 못한 단순 queue 확장은 반려.

---

## 13. P5 — Rollout Buffer, candidate dtype, 메모리 레이아웃

1. 현재 `prepare_streaming/update_streaming`은 **메모리 최적화 성공, 속도 중립**: single-seat 10k peak RSS 18.0→7.34GiB. 단, 완전한 columnar memory store가 아니다.
2. Per-row observation Python objects/list/dict 복제, `InlineObservationStore.stacked_indices`, Numpy concatenate, Torch Tensor creation, GC/object lifetime 측정.
3. `[B,4,64,6] int64` candidate staging을 wire의 compact `u8` 기반으로 유지해 필요한 연산 직전에만 안전히 int64로 확장. Signed target field를 포함한 값 범위, sentinel, no-truncation 보장.
4. `RolloutBatch.to()`에서 learner forward에 필요 없는 metadata `match_id`, `side`, `turn` 등을 GPU 전송하지 않고 CPU에서 보존. 실제 H2D 바이트와 full wall 비교.
5. Preallocated per-minibatch CPU/GPU slabs, slab lifetime/ref reuse, pinned buffer staging, alignment. 반복 메모리 할당과 zeroing이 병목일 때만 도입.
6. GAE reference parity를 유지하며 `compute_gae_for_rows`의 per-side small Torch tensors와 Python loop를 vectorized/NumPy contiguous array로 교체하는 PoC. 기존 측정 1.26초/58초 learner이므로 우선순위 낮음.
7. CPU RSS, page faults, pinned bytes, GPU VRAM, allocator fragmentation, throughput, logprob recompute, gradient/update parity 보고. 메모리만 줄고 속도 중립이면 장기 안정성 항목으로 보존하되 속도 상승으로 발표하지 않는다.

---

## 14. P6 — NCCL, PCIe, NUMA, dual-GPU 구조 최적화

- NUMA0/GPU0과 NUMA1/GPU1 맞춰 team/Rust buffers, pinned staging, rollout buffers를 초기화할 때 locality 확보.
- 원격 NUMA 메모리 비중, cross-socket traffic, aggregate memory bandwidth, PCIe 실제 link speed/width, H2D/D2H per rank, gradients 통신을 측정.
- DDP gradient bucket 크기, `gradient_as_bucket_view`, `find_unused_parameters`, static graph 가능성 등은 실제 모델 graph가 안정적임을 검증한 뒤 1개씩 A/B. CPU PCIe link를 GPU0↔GPU1 P2P처럼 오인 금지.
- Manual flat allreduce SUM의 latency 및 버킷별 NCCL 효율과 DDP reduction overlap을 측정. 통신이 훨씬 느리면 두 GPU Actor-only + single learner 같은 대안 비교.
- GPU1 150W/GPU0 175W 설정 불변. Thermal/clock 차이가 성능에 미치는 영향 기록. 권한 없이 파워캡 상향 금지.
- 단일 GPU의 bottleneck이 learner라면 Actor-only 162.73 games/s 숫자만으로 two-GPU PPO speedup을 약속하지 않는다.

---

## 15. P7 — 높은 리스크의 고급 GPU 가속 PoC

**우선순위는 profiler가 확인해줄 때만 높인다.** CUDA Graph, `torch.compile`, Triton, custom CUDA kernel은 기본 pin stack에서 지원되는 최소 영역부터 실험한다. 설치/업그레이드 금지.

- 96-token encoder forward의 정적 shape, scorer action candidates branch count buckets, policy/value loss의 small kernel fusion, kernel launch count.
- `torch.compile`을 전체 모델에 갑자기 적용하지 말고 개별 계산블록부터 profile; graph break, compile overhead, cache memory, steady-state wall 측정.
- CUDA Graph는 dynamic candidates와 random sampling, CPU prefix dependency 때문에 actor 전체를 무리하게 capture하지 않는다. Learner 고정 microbatch, 정적 buffer에서 PoC.
- SDPA math implementation vs 실제 sm70 지원 backend, embedding backward scatter custom alternative. 확장 설치/새 CUDA extension은 사용자 별도 승인 후.
- Graph/capture 검증에는 RNG, same logits/logprob, gradient parity, saved tensors, GradScaler/optimizer, graph replay의 메모리 안정성을 포함.
- 파라미터 수/모델 층/토큰 수를 임의 줄여 학습을 빠르게 만드는 것은 본 최적화 트랙이 아니다.

---

## 16. P8 — 체크포인트, 긴 실행 및 테스트 품질

- 체크포인트에는 model, Adam, GradScaler, LR clock, global match counter, all rank RNG/seed, policy snapshot identifiers, pending rollout disposition, team manifest SHA 및 experiment config 보존.
- 재개 직전/후 동일 정책 입력에서 model logprob/value, 다음 optimizer update와 checkpoint hashes를 비교. rank0 only 저장 시 rank1 고유 RNG 재생 복원 방법 명확히.
- 중간 강제 종료 이후 incomplete match를 completed로 세지 않고, 잘못된 rows를 다음 on-policy iteration에 유출하지 않는다.
- `all-in`은 collect, GAE, normalize, PPO 4 epochs, NCCL, gradient, scheduler, real checkpoint, metric writer, required eval을 **각 별도 column과 전체로** 기록. 순수 collector 또는 bounded PPO와 operational 속도 명칭 구분.
- 2k×3 warm small, 10k 이상 long, 2 iteration replay/resume, 지속 실행 시 메모리 증가/열/서비스 간섭 등을 기록.
- Dirty-tree benchmark는 탐색용만, 최종 clean commit 재측정 필요. 각 반복 개별 model SHA, dataset SHA, RNG seeds, code/version, wall, rows, error/skip 명시.
- 두 rank의 global natural completed match 합계는 게임 중복 없이 계산. 동일 target games 기반으로 single과 dual의 등가성 판단.

---

## 17. 성능 실험용 run matrix 및 단계별 강등 조건

| 패널 | 조건 | 필수 산출물 | 성능 수락 기준 |
|---|---|---|---|
| A0 | single GPU corrected PPO, both seats, 2k total ×3 | baseline, current-policy rows/s, RSS/VRAM | loss/gradient parity |
| A1 | single GPU corrected PPO, 10,240 total | steady bounded reference | checkpoint 포함 |
| D0 | single vs distributed **같은 4096 실제 rows** | loss/gradient/Adam/scaler/LR/epoch KL | stated tolerance PASS |
| D1 | single GPU 2,048 total vs 2 GPU 1,024/rank total 2,048 | startup-separated and launcher wall | median, worst, distribution |
| D2 | single GPU 10,240 total vs dual 5,120/rank, total 10,240 | all-in wall, sync stage, RSS/VRAM | same contracts |
| M0 | actor batch+worker sweep under corrected contract | p50/p95 batch sizes, tail, CPU/GPU timeline | not speculative |
| P1 | rolling slots off/on, 1,024 total slots | real-policy actor and full PPO | exact rewards and games |
| P2 | 1×1,024, 2×512, 4×256 cohort pipeline | GPU compute overlap and full wall | same total concurrency |
| P3 | 1024 vs 2048 and kernel alternatives | GPU kernels, forward/backward, all-in | 2048 previous failure preserved |
| F1 | clean HEAD long soak + checkpoint resume | final report and raw JSON | readiness separate |

**Speed experiments default minimum:** 3 independent repeats at same config; compare median, min/max, uncertainty. Numerical/benchmark acceptance threshold is **measurable effect beyond repeat noise** with no regression, not arbitrary >X% guarantee. Small effect goes to `neutral/rejected` record.

### 17.1 정밀 비교에서 반드시 분리할 것

- Corrected Entropy/KL gradient vs buggy baseline: 값 자체가 달라지는 correctness 변경. 향상률 판단용은 수정 후 baseline끼리 비교.
- Single-seat vs both-side: rows/match와 optimizer steps 달라 직접 속도 비교 금지.
- One rank 2048 games vs two ranks 각각 2048 games: 총 2k vs 4k workload라 직접 speedup 비교 금지.
- CUDA/NCCL debug mode vs release mode: 통신 타이밍과 profiler overhead 다름.
- Runtime `git_dirty` 및 init SHA 달라지는 반복: paired or seeded conditions 명시.
- GPU event ms vs sum kernel active ms, total summed rank update wall vs actual launcher makespan.

---

## 18. 성능 분석용 안전한 명령 초안

> **현재 HEAD에서 확인된 기존 인터페이스만 사용한다.** DDP runner는 현 시점 broken이므로 아래 짧은 명령도 C0/D0를 통과하고 실행기를 수정한 뒤에만 실행한다. 설치돼 있지 않은 도구를 무단 설치하지 않는다. 명령은 miniDC repo root 기준이다.

```bash
# A. Git 및 현재 프로세스 인벤토리
pwd
git fetch origin
git status --short
git rev-parse HEAD
git branch --show-current
nvidia-smi --query-gpu=index,name,memory.used,memory.total,power.draw,temperature.gpu --format=csv

# B. 정밀 유닛 테스트, 현재 기존 모듈 범위
PYTHONPATH=engine/python:. .venv/bin/python -m pytest tests/agent -q

# C. Single-GPU benchmark, 2k both-sides, corrected G0 patch 이후
PYTHONPATH=engine/python:. .venv/bin/python scripts/bench_pa3_end_to_end.py \
  --games 2048 --envs 1024 --workers 16 --repeats 3 \
  --mode full --observations fixed --candidate-wire packed \
  --precision fp16 --inference-mode --device cuda:0 \
  --microbatch 1024 --streaming-minibatch \
  --report runs/perf/v4_corrected_single_2k.json \
  --checkpoint runs/perf/v4_corrected_single_checkpoint.pt \
  --tag v4-corrected-single

# D. DDP runner CLI sanity (수정 전 실행 불가)
PYTHONPATH=engine/python:. .venv/bin/python scripts/run_ddp_ppo.py --help

# E. Read-only disk/swap/memory checks
free -h
cat /proc/swaps
vmstat 1 15
lsblk -o NAME,TYPE,TRAN,ROTA,SIZE,MOUNTPOINTS

# F. 독립 엔진 readiness, PASS를 강제하지 않는다
bash scripts/cargo.sh run --locked --release --example readiness_check
```

**DDP 수리 후 smoke 순서:** forced uneven synthetic → true PA3 one-step → 32/rank collect/update → 256/rank → total 2k×3 → total 10,240+ → two iterations and resume. 진단 trace `TORCH_DISTRIBUTED_DEBUG=DETAIL`은 profiler off 성능 측정과 분리.

### 18.1 자동화 가능한 계측 분류 제안

각 stage에 `torch.profiler.record_function("actor/observe_fixed")`, `"actor/candidates"`, `"actor/h2d"`, `"actor/inference_encoder"`, `"actor/branch_step"`, `"actor/d2h"`, `"actor/record"`, `"learner/gae"`, `"learner/minibatch_materialize"`, `"learner/h2d"`, `"learner/forward"`, `"learner/backward"`, `"learner/grad_allreduce"`, `"learner/clip_optimizer"` 같은 레이블 삽입. 핵심 Rust/Python 타이머는 다른 시간축의 중복 없이 기록. 이름은 필요에 따라 조정 가능.

---

## 19. 정확성/성능 자동 테스트 계획

**새 테스트 파일 제안** (`tests/agent/`, `tests/integration/`, `scripts/`):

1. `test_policy_regularizer_gradients.py`: entropy/Uniform-KL 미분, finite difference, zero/singleton/hidden mask.
2. `test_full_ppo_objective_reference.py`: policy/value/actor_mask/value_mask, padding, all gradients, old/new logprob, mixed precision.
3. `test_collect_both_current_sides.py`: both seats terminal rewards, count once, historical policies 제외, request provenance.
4. `test_global_advantage_norm.py`: uneven ranks, actor-free rank, masked rows, single global reference.
5. `test_ddp_collective_protocol.py`: rank exhaustion, zero-rank, final partial, KL stop, skipped step, exact fixed micro-step count.
6. `test_ddp_global_optimizer_parity.py`: real model 4096 rows, full model/optimizer state/scaler/LR after 1/3 steps.
7. `test_manual_allreduce_parity.py`: variable actor/value counts, zero gradients, global clipping, SUM vs DDP average coefficient.
8. `test_ddp_checkpoint_resume.py`: two rank checkpoints, history/policy version, full RNG/optimizer continuation.
9. `test_rolling_slot_equivalence.py`: natural match identity, terminal/drain, no double count, no dropped long match.
10. `test_cohort_pipeline_equivalence.py`: stable policy version, action prefix, per-cohort RNG, order-independent records.
11. `test_observation_pack_and_candidate_parity.py`: Rust oracle vs candidate/known masks, signed target sentinel, padding bound.
12. `test_benchmark_accounting_v4.py`: collection contract, global game count, actor rows, mixed stage denominator, true checkpoint included.

각 test는 가능하다면 **old failing case / new passing case**를 명시. 테스트가 통과하더라도 장기 성능, full-scope rule readiness 및 최종 모델 강함은 따로 검증한다.

---

## 20. 실험 우선순위 결정 알고리즘

매 단계에서 다음 우선순위를 재계산한다.

1. `correctness severity`: 학습 수학, illegal move, hidden info, rank divergence가 있으면 모든 speed 실험 정지.
2. `measured critical-path share`: 실제 전체 wall에 기여한 시간. API 호출 빈도, GPU idle, synchronizations, memory allocation 포함.
3. `theoretical removable share`: 해당 stage를 완전히 없애도 전체 속도가 얼마나 개선되는지 Amdahl 한계 계산. 예를 들어 GAE가 2%라면 전체 속도 개선 2% 수준이 상한인 조건이다.
4. `complexity / regression risk`: 구현 난도, 안전성, 코드 소유권, 유지 보수, 검증 범위.
5. `A/B evidence`: 동일 조건 반복 중앙값, worst, GPU/rank memory budget, actual parity.
6. 다음 병목: 하나 제거되면 전 스택 profiler를 다시 실행. **한 번 만든 오래된 병목 순위를 끝까지 따르지 않는다.**

**중단/승격 규칙:** 개선이 노이즈 이내거나 정확성 비용이 크면 안전한 baseline으로 rollback. 명백히 빨라졌어도 값이 바뀌는 최적화는 학습 계약 검증 전 승격 불가. 추가 성능 잠재력을 수치 근거 없이 확정하지 않는다.

---

## 21. Full-engine readiness와 sample efficiency는 별도 트랙

현재 `readiness_check`는 10/16 PASS이며 **2,3,5,8,15,16 FAIL**. 515 legal moves, 223 legal abilities, dynamic effect closure, full-scope unsupported-mechanic 0, full-coverage throughput, showdown reference full-battle diff 등은 별도 규칙 완성 과제다.

- 빨라진 현재 frozen 1,137-team self-play가 모든 Champions M-C 규칙을 충족하는 것은 아니다.
- Full Spec이 정의한 hidden-info safety, legal move oracle, complete rule coverage를 축소해 throughput 승격 금지.
- 본 4.0 작업에서 수행하는 짧은 PPO 및 분산 테스트는 **본학습 승인과 다르다.** 100M-run, 공개 ladder RL, 새로운 상대 리그, 대규모 학습률 변경은 별도 승인.
- 장기 목적은 `games/s`뿐 아니라 실제 model strength per wall-hour. 고정 opponent/seed panel과 Elo 불확실성, 샘플 효율 비교는 별도 연구 계획을 따른다.

---

## 22. 필수 보고서 및 파일 목록

새 `docs/perf/` 산출물 제안:

- `V4_CORRECTNESS_FORMULA_AUDIT.md`: entropy/KL gradient 전후, PPO 수식 oracle, test evidence.
- `V4_COLLECTION_CONTRACT_AUDIT.md`: both sides, terminal reward, historical exclusion, row provenance.
- `V4_DISTRIBUTED_MATH_AND_PROTOCOL.md`: DDP/global advantage/LR/KL/GradScaler, no_sync schedule, zero-rank tests.
- `V4_MANUAL_ALLREDUCE_COMPARISON.md`: 구현과 실제 동일 총 경기 수 DDP 경쟁 결과.
- `V4_FULLSTACK_BOTTLENECK_ATLAS.md`: 전체 CPU/GPU/PCIe/NCCL/Rust/Python runtime trace, critical path, unaccounted.
- `V4_VRAM_ACTOR_PIPELINE.md`: rolling slots, 1/2/4 cohort, pinned resident buffers, tail tracking.
- `V4_KERNEL_AND_MEMORY_FRONTIER.md`: microbatch, SDPA, embedding, optimizer, row materialization, allocator.
- `V4_OPERATIONAL_FRONTIER_REPORT.md`: 3-repeat, longer run, true operational stage, successes/rejected, next plateau.

`runs/perf/v4/`: immutable raw JSON, compact kernel/collective summaries, report includes, run manifest per step, error snapshots, SHA, normalized totals, reproducible commands. 1GB급 Chrome trace 등은 git에 푸시하지 않는다.

### 22.1 각 커밋에 적어야 할 것

- 측정 이전 코드 SHA와 이후 코드 SHA, 변경 파일, 변경 전/후 수치와 비교 가능한 정도.
- 소유한 test 추가 및 old-fail/new-pass 여부, 통과한 테스트 명령의 stdout 요약.
- 과정에서 실제 실행된 명령과 장비에서 측정한 환경 (추정으로 결괏값 생성 금지).
- 모든 성능 패널의 source/provenance raw JSON, 실패한 실험 원인, rollback 여부, 다음 실제 blocker.
- branch HEAD와 remote push 동기화, working tree clean 여부. 남아 있는 별도 engine readiness 상태.

---

## 23. 사전 완료 체크리스트

### 학습 정확성

- [ ] Entropy/Uniform-KL gradient 결함 재현 및 수정, 실제 scorer nonzero gradient와 finite diff.
- [ ] Reference PPO policy/value/KL/entropy 수학, mask, GAE, single model full gradients.
- [ ] Both-current-side collector 별도 회귀 테스트, historical opponent 제외, reward 반대 부호.
- [ ] Streaming vs standard real model corrected-objective update parity.
- [ ] Identical checkpoint/resume single GPU 및 global match clock.

### 듀얼 GPU

- [ ] Rank-global advantage normalization.
- [ ] Rank-global LR clock, optimizer step, scaler skip, gradient clipping.
- [ ] Rank-global KL early stop.
- [ ] Padding-only rank fixed-step DDP collective sequence 및 zero-gradient participation.
- [ ] Rank0/1 output/launcher child process safe logging, watchdog, digest.
- [ ] Real model DDP 1-step/3-step parity on actual matched dataset with uneven shards.
- [ ] Manual All-Reduce independent oracle parity, 2 implementations true all-in A/B.
- [ ] Per-rank NUMA and NCCL telemetry, no disallowed system changes.

### 전 스택 성능

- [ ] CPU/Rust/Python main thread, GPU model, PCIe/NCCL, H2D, memory allocator, queue idle complete bottleneck map.
- [ ] GPU kernel active and wall/dispatch/timeline separated.
- [ ] Games/learner rows/optimizer steps/actual active work per match in corrected both-seat contract.
- [ ] Microbatch 2048 rejected conclusion preserved, alternative kernel/VRAM pipeline experiments justified.
- [ ] Rolling slots and 1/2/4 independent cohorts measured at equal concurrency.
- [ ] Candidate slab, Rust encoder cache, NumPy/metadata copies measured before refactor.
- [ ] 3 repeats each adopted candidate, 10k+ confirmation, long-run checkpoint/resume.
- [ ] Same TOTAL matched natural games for single vs dual full PPO.
- [ ] 28GiB each GPU, 48GiB host, pinned 1GiB budget, no swap, no illegal/operational errors.
- [ ] Full engine readiness remains independently tracked; no 100M training.

---

## 24. 근거: 프로젝트 코드와 공식 문서

**프로젝트 기준 SHA:** `816707ecf8036ae493140774dc2d3b43e6e1a72e`

- [v3 numeric parity](https://github.com/mwl313/PokeAgent3.0/blob/816707e/docs/perf/V3_NUMERIC_PARITY.md)
- [v3 VRAM batch sweep](https://github.com/mwl313/PokeAgent3.0/blob/816707e/docs/perf/V3_VRAM_BATCH_SWEEP.md)
- [v3 DDP error and implementation](https://github.com/mwl313/PokeAgent3.0/blob/816707e/docs/perf/V3_DDP_GLOBAL_GRADIENT.md)
- [PPO learner](https://github.com/mwl313/PokeAgent3.0/blob/816707e/agent/ppo/learner.py)
- [PA3 model: gradient graph](https://github.com/mwl313/PokeAgent3.0/blob/816707e/agent/model/pa3_model.py)
- [DDP adapter](https://github.com/mwl313/PokeAgent3.0/blob/816707e/agent/ppo/ddp.py)
- [Dual-run launcher](https://github.com/mwl313/PokeAgent3.0/blob/816707e/scripts/run_ddp_ppo.py)
- [Native collector](https://github.com/mwl313/PokeAgent3.0/blob/816707e/agent/train/native_collector.py)
- [PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) — `no_sync`, `join`, unused-parameter semantics.
- [PyTorch ProcessGroupNCCL variables](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html) — watchdog and collective desync trace variables.
- [PyTorch profiler guide](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) — scheduled trace, compact profiling.
- [PyTorch performance tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) — unnecessary sync, data pipeline, CUDA Graph trade-offs.
- [PyTorch AMP accumulation/clipping](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) — accumulation scale and unscale ordering.

**근거 구분:** 기존 수치/코드 문제는 상기 GitHub 소스의 기록에 근거한다. DDP/Manual All-Reduce 구조, 수락 기준, 성능 실험은 이 문서가 새로 제안하는 설계이며 아직 miniDC에서 새로 실행한 것이 아니다. 속도 목표는 모두 미측정 가설이다.

---

## 25. 에이전트에 그대로 전달할 실행 프롬프트

```text
Execute the attached “PokeAgent 3.0 Correctness-First Extreme Optimization & Full-Stack Bottleneck Audit v4.0” on the existing optimization/pa3-realpolicy-throughput branch. The document was authored against verified commit 816707ecf8036ae493140774dc2d3b43e6e1a72e. Fetch current HEAD first; reconcile and preserve any subsequent code and results. Do not overwrite work or reset the branch.

The two top-level tracks are (A) prove and correct all PPO math, particularly the presently detached entropy and uniform-KL regularizer gradients, plus both-current-seat trajectory/reward provenance; and (B) repair multi-GPU training correctness, then compare two executors: standard PyTorch DDP vs explicit manually synchronized gradient All-Reduce. Track B must use Track A's verified single-GPU real-PA3 oracle.

Mandatory distributed correctness audits: global advantage normalization; shared cumulative natural match LR clock; globally aggregated epoch KL and identical early stopping; synchronized AMP/optimizer step skip and global gradient clipping; padded/empty rank collective ordering; DDP no_sync coverage of forward and backward; well-defined global 4096 logical minibatches; consistent 4 PPO epochs; rank output/launcher logging; checkpoint/state and SHA parity. Test unequal and actor-free ranks and a truly empty rank. Do not accept a DDP result merely because it no longer hangs.

For the manual All-Reduce alternative, unscale identical local AMP grads, use rank-local gradient SUM formulas without DDP's world-size correction, all-reduce FP32 gradient SUM on every parameter including zeros for missing grads, globally check finiteness, clip globally, apply exactly one common Adam update, and verify model, optimizer and scaler parity.

Once correctness gates pass, aggressively profile every real execution layer: Rust game engine, Python interpreter/GIL and object allocation, observation packing, candidate prefix operations, rollout storage, GAE, minibatch materialization, CUDA kernel launches and actual GPU active time, embedding/SDPA/GRU/backward/optimizer kernels, H2D/D2H, pinned/pageable copies, host RAM/NUMA, PCIe, NCCL calls and waits, rank synchronization, process startup, checkpoint, evaluation and all-in wall. Build a per-stage measured critical-path and contention report. Re-profile after every accepted bottleneck removal. Never guess a limit from VRAM capacity or CPU/GPU utilization alone.

Then attack the measured bottlenecks using rolling game-slot refill, true independent-cohort CPU/GPU pipelining with equal total concurrency, fixed-shape GPU batching/buffer reuse, packed candidates/columnar slabs, Rust encoder caching and observation zero/low-copy, kernel-optimization PoCs and safe pinned staging. The prior 2048-microbatch sweep was neutral and should not be adopted without new evidence. Avoid changing the model, game rules, full legal action space, data/team pool, PPO hyperparameters, reward and observation semantics.

For each isolated change run tests, single-GPU numerical reference checks, 3-repeat paired performance A/B, per-rank CPU/GPU/NCCL telemetry and a longer 10,240-game confirmation when justified. Compare equal TOTAL natural games across single and dual, and always report both games/s and valid current-policy learner rows/s under the now-correct both-seat collection contract. Report training-step counts, all-in scope, seed, model and dataset SHA, memory, zero operational errors, and confidence/noise. Commit and push successful changes and docs with raw compact JSON; keep huge profiler traces out of git. Reject or rollback neutral, slower or incorrect experiments.

DO NOT start the 100-million-game run. Full engine readiness is separately 10/16 and must reach 16/16, plus explicit user authorization, before full training. Do not change host drivers, installed PyTorch/CUDA, BIOS, power caps, network/services, or terminate unrelated processes. Diagnose first and use process-local switches only.

Start with the C0 entropy/KL gradient tests and corrections, proceed to C1 single-GPU reference, then D0/D1/M1 distributed paths and F0 full-stack forensic profiling. Provide each step's measured results, blockers and concrete next experiments, not speculative speedup promises.
```

**종료 기준:** DDP 여부와 무관하게 정확한 학습의 정본이 있고, 두 GPU의 최적 실행 방식이 실측으로 결정되며, 전체 시스템의 남은 병목과 예상 개선 한계가 측정되고, 높은 가치의 후보 실험이 재현 가능하게 검증되는 것. 목표 games/s를 임의로 정해 도달했다고 최적화를 끝내지 않는다.
