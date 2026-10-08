# PokeAgent 3.0 — Extreme Optimization Master Plan

**Version:** 1.1 — uncapped throughput revision  
**Date:** 2026-10-08 (Asia/Seoul)  
**Target:** `mwl313/PokeAgent3.0`  
**Reviewed ref:** `9cbaa1fbe42264eb124f9ea3ee74dc7af13132af` on `training-pool/minidc-userteam-benchmark`  
**Approach:** **A. 현재 구조 개선 (무상한 처리량 목표)**. Rust CPU 배틀 엔진 + PyO3 + PyTorch V100 학습을 유지하면서 데이터 변환, CPU/GPU 교환, 모델 실행, 버퍼, NUMA, DDP의 전체 비용을 줄이고, 첫 1,000 games/s 이후에도 측정된 시스템 병목을 끝까지 추적한다.  
**Status:** 실행 및 구현 **계획서**. 아래의 신규 성능 수치, 개선율, 테스트 결과는 아직 실측하지 않았다.

> **핵심 결정:** 모델 구조, 게임 규칙, 관측 정보, 학습 의미를 축소해 속도를 얻지 않는다. 대규모 Python 객체 생성과 불필요한 복사, GPU 동기화, 반복 인코딩을 제거하고 하드웨어를 실제로 병렬 활용한다. **1,000 자연완결 games/s는 최종 목표가 아니라 중간 이정표다. 이후에도 2,000, 5,000, 10,000+ games/s 구간에 도전하되 어느 수치도 성능 보장 또는 필수 통과 하한으로 주장하지 않는다. 실제 하드웨어와 정확성 조건을 만족하는 최댓값을 탐색한다.**

---

## 0. 실행자가 먼저 알아야 할 사항

### 0.1 변경하지 않는 것

- **게임:** pinned Pokémon Showdown `gen9championsvgc2026regmc`, CTS/BO1, Mega 허용, Tera 비활성.
- **학습팀:** 활성 `mb-mc-v3-userteam-all-train`, 1,137 unique teams, 904 groups, 전량 train 및 uniform team sampling. 원본 1,136팀의 불변 이관 유지. 사용자의 리자몽은 Mega 전 `Charizard / Blaze`, Mega Y 후 `Drought`로 **2026-10-08 확인 완료**. 원문 Poképaste는 변경하지 않는다.
- **AI:** PA3-8M, 약 8.76M params, Transformer 6 layers, d_model 320, heads 5, hidden 320 conditional prefix GRU, 88 active entity tokens in 96 padded rows, 행동 후보 prefix-dependent masking.
- **학습:** scratch self-play PPO, GAE, 원래 loss 및 entropy/KL 계약, iteration당 신규 learner decisions 최소 131,072개 및 요구된 자연완결 cohort, global minibatch 4,096, PPO 4 epochs. 일관된 frozen policy version 및 올바른 sampled/recomputed logprob 필수.
- **장비:** Xeon E5-2673 v4 ×2, 총 물리 40코어/80스레드, RAM 약 62GiB, V100 PCIe 32GiB ×2, GPU0 175W, GPU1 150W. P2P 미지원, PCIe/SYS 경유. GPU0↔NUMA0, GPU1↔NUMA1. **Full Spec 예산:** 전체 host RAM soft budget 48GiB, rollout buffer 24GiB, pinned staging 총 1GiB 이하, GPU당 28GiB soft budget.
- **런타임:** driver 580.178.04, CUDA Toolkit 12.8.2, nvcc V12.8.93, Python 3.12.3, torch 2.14.0+cu126, sm_70 포함. eager PyTorch + SDPA math가 승인된 기준선. FP16/FP32 허용, BF16/FP8/TF32/FlashAttention-2/Inductor/필수 Triton 교체 금지.
- **운영:** `dsh-web`, `llama-swap` 등 외부 서비스, 전력 제한, 드라이버, 시스템 CUDA, 기존 venv 및 패키지를 임의 변경하지 않는다. 새 학습 전에 GPU 실제 점유자와 가용 메모리를 확인한다.
- **정확성:** 불법 행동 허용, 관측 정보 유실, hidden-information leakage, 승패/완주 조건 변경, truncation을 통한 속도 부풀리기 금지.
- **범위:** Full Spec 1.1의 전체 M-C readiness는 아직 미완료. `readiness_check` **10/16 PASS, 6 FAIL (2, 3, 5, 8, 15, 16)**. 성능 개선은 이 사실을 변경하지 않는다. 1억 경기 본학습 자동 시작 금지.

### 0.2 이미 달성한 실측 기준선

| 측정 | 환경 | 기존 수치 | 해석 |
|---|---|---:|---|
| Rust engine-only | single-thread, 22,740 games | **2,579–2,689 games/s** | 단일 CPU 코어 기준, AI 없는 자연 배틀 |
| PyO3 random two-rank | 2 × 1,024 env, rank별 16 workers | **합산 592.6 games/s** | GPU 실정책 사용하지 않음 |
| Random actor | 1,024 env, 16 workers | **522.7 games/s** | Python/Rust 왕복 포함, neural inference 제외 |
| PA3-8M actual rollout | GPU0 1개, 1,024 env, 16 workers, 10,240 games | **16.4 games/s** | **실정책 수집만**, PPO 업데이트 비용 별도 |
| PA3-8M collector wall | 10,240 games | **625.9 s** | model forward 약 185.0s, native observation 13.5s, legal masks 2.2s, Rust step 2.0s. 나머지는 Python 처리 및 미계측 오버헤드 |
| observation standalone | 256-env fixture | **36.97ms/round individual NumPy vs 2.45ms/round fixed-batch parse** | 이 약 15배는 디코딩 단계에만 해당 |
| observation standalone | same fixture | **212.53ms/round naive torch conversion** | 특정 test 구현 비용이며 실제 collector 값과 혼동 금지 |
| rollout memory | 10,240 games | **호스트 약 25GiB, GPU peak 3.0GiB** | GPU peak 메모리와 utilization은 다른 지표 |
| rollout candidate activity | 10,240 games | **1,499 batched candidate crossings, 약 409만 candidates** | 객체 생성 최적화의 근거 |
| smoke correctness | 10,240 games | **0 abort, ~1.9e-5 logprob recompute diff** | 테스트 경로만 성공. 승률 개선 검증 아님 |

**측정 주의:** 서로 다른 random-policy, engine-only 및 actual-policy benchmark의 games/s는 직접적인 speedup 대조군이 아니다. 모든 최적화의 before/after는 **같은 스크립트, 같은 모델 가중치, 동일 환경 수, CPU affinity, 팀 집합, 시드 패널, 유사 온도/전력 상태**에서 측정한다. 자연 경기 길이 변화가 throughput 착시를 만드는지 `decisions/s`, `turns/game`도 함께 보고한다.

---

## 1. 가장 먼저 해결할 기술적 발견: 현재 코드의 실병목

| 코드 위치 | 현재 동작 | 추정 문제와 개선 방향 |
|---|---|---|
| `agent/train/native_collector.py::_collect_round` | `observe_encoded_batch`를 호출해 **view별 bytes**를 수령, `parse_view` → `ObservationBatch.from_native_payload` 반복, 이후 `cat`과 `.to(device)` | 수천 개 개별 Python 객체와 텐서 및 반복 concatenate. 이미 존재하는 `observe_fixed_batch` 기반 **batch-first** 경로로 교체 |
| `engine/src/python.rs::observe_fixed_batch` | 고정 stride와 ragged sidecar를 함께 `PyBytes`로 내보내는 고속 API가 이미 있음 | 현재 실제 PPO collector가 사용하지 않음. 연속 버퍼의 lifetime, memoryview, dtype, contiguous packing 및 ragged 의미 보존을 검증 후 연결 |
| `engine/src/python.rs::observe_encoded_batch` | 매 호출마다 `Encoder::from_validated_dex(&batch.dex)` 생성 | 매 배치마다 동일한 도감/feature 매핑을 재구축하는지 **프로파일링 후**, immutable encoder를 `NativeEngine`에 캐싱 검토 |
| `agent/train/native_collector.py::_level_table` | 후보마다 `AtomicAction`과 `ActionRef`, `torch.tensor(ref.tuple)` 생성 | 보고된 약 409만 후보의 per-candidate Python allocation. Rust 내부에서 **packed batch candidates** 반환 |
| `engine/src/python.rs::candidates_batch` | `Vec<Vec<ActionTuple>>` 반환 | 개별 Python tuple/list 객체 다량 생성. flat packed ID/mask/count/offset 및 resolved entity/move-token 인덱스 생성 |
| `agent/train/native_collector.py` | `step.pick[offset]`, `.item()`, 텐서로부터 `int`/`float` 변환 반복 | 비동기 GPU 연산을 **반복 CPU 동기화**할 위험. level별 한 번에 D2H 반환 |
| `agent/model/pa3_model.py::encode` | `bool(token_mask[:,global_index].any())` 호출 | GPU boolean을 Python으로 평가해 불필요한 sync 가능. schema가 보장하는 global token 조건을 CPU에서 검증하거나 GPU 순수 수식으로 처리 |
| `agent/train/native_collector.py` | 추론에 `torch.no_grad()` 사용, 수집 경로에 명시적 FP16 autocast 없음 | `inference_mode` 및 **fp16 autocast**를 별도 A/B 실측. PPO 확률/ratio/log-softmax는 FP32 |
| `agent/buffer/rollout_buffer.py`, `agent/types/observation.py` | 요청별 관측, 후보, mask, prefix를 Python 객체 및 compact NumPy로 저장 | 1만 경기에 호스트 약 25GiB 사용. 하나의 고정 packed row slab과 가변 길이 sidecar, offset index로 전환 |
| `scripts/run_ppo_smoke.py` | 10,240경기 후 `committed_matches=args.games` (10,000)로 update 호출 | **실제 자연완결 10,240**을 policy LR/counter에 반영하도록 누적 장부 수정. 재시작 및 overshoot test 필요 |
| `scripts/run_actor_pair.py` | 2개 rank/NUMA를 측정하지만 **random policy** 사용 | 실제 2-GPU PA3 inference와 DDP 학습 결과가 아님. 별도 동일 조건 실정책 benchmark 필요 |

### 1.1 정확성 선결 사항: 고정 토큰과 ragged 정보의 계약

`engine/python/pa3_engine/observation.py::parse_view/parse_batch`는 effects, repertoire, types, base moves, move effects라는 가변 길이 데이터도 파싱한다. **반면 현재 `agent/types/observation.py::ObservationBatch.from_native_payload`는 고정 9종 필드만 받아들이고 나머지 payload key는 무시한다.** 이것은 단순 복사 최적화 이슈를 넘어 최종 모델 입력의 **정보 완전성 미검증 영역**이다. 일부 정보가 다른 고정 특징으로 이미 표현되어 있을 수 있으므로 단정적인 결론을 내리기 전에 의미별 inventory를 대조해야 한다.

**P0 필수 작업:** Rust `PlayerView` → native `EncodedObservation` → packed fixed/ragged → typed `ObservationBatch` → `TokenEmbedding`까지 각 feature의 존재, known-mask, 실제 모델 소비 여부를 표로 감사한다. Full Spec의 **동적 effect와 공개 repertoire를 임의로 버리는 최적화 금지**. 이미 미구현인 private effect/ability/item metadata는 별도 engine completeness blocker로 기록하고 정상 데이터처럼 채워 넣지 않는다.

---

## 2. 최적화 원칙과 점진적 설계

### 2.1 목표 데이터 흐름

```text
NUMA0 CPU: Rust Batch Engine 1,024 env / Rayon workers
  └─ encode + legal candidates + request metadata [Rust native contiguous buffers]
       └─ PyO3 low-overhead bounded batch boundary
            └─ reused/pinned CPU staging, dtype conversion at most once
                 └─ GPU0: PA3-8M encode once per request
                      └─ GPU0: prefix-conditional action scoring, masked sampling
                           └─ one batched selected action transfer per prefix level
                                └─ Rust native step_batch, no per-action Python conversion
                                     └─ packed rollout buffer → PPO learner

NUMA1 CPU and GPU1 mirror the rank for separate 1,024 env;
DDP sync only at optimizer boundaries, never exchange token activations per action.
```

- **CPU 게임 엔진 유지:** 정확한 배틀 및 full effect closure를 GPU에 이식하는 대형 재작성 금지.
- **Python은 coordinator:** 개별 기술, 후보, 관측 field 변환을 Python hot loop에서 제거. 학습 자동 미분 및 optimizer는 PyTorch로 유지.
- **Batch before parallelism:** 각 rank 안에서 dense batch 처리 후 두 rank로 확장.
- **Same semantics:** 기존 policy/model/action/observation schema, 2 side information boundaries, prefix-dependent masks, GAE trajectory grouping 보존.
- **Measure not assume:** 개선 예상 효과를 사전 숫자로 단정하지 않는다. 모든 최적화에 rollback 가능한 feature flag와 동일 조건 baseline을 확보한다.
- **No GPU memory abuse:** two V100 32GiB are two distinct devices; max GPU memory soft budget 28GiB/card, buffer primarily host RAM with measured staging.

### 2.2 성공 지표 우선순위

1. **Correctness:** 모든 게임 자연완결, zero operational errors, illegal action zero, observation feature parity, no information leakage, PPO recompute 및 finite updates.
2. **Rollout throughput:** real PA3 policy 기준 자연완결 games/s, decisions/s, full turn distribution.
3. **True end-to-end throughput:** 수집 + GAE + 4 PPO epochs + checkpoint + 정상 평가를 포함한 committed games/s.
4. **자원:** peak CPU RSS, GPU allocated/reserved, H2D/D2H transferred bytes, Python allocations, CPU core utilization, NUMA remote memory %, GPU SM utilization, power 및 온도/스로틀.
5. **Scaling:** matched workload의 GPU0 한 rank 대비 NUMA+GPU 두 rank speedup, synchronization overhead, 느린 GPU 1의 대기 영향.

---

## 3. 실행 순서: 단일 프로젝트 안에서 8개 최적화 단계

> 각 단계는 **새 코드 구현 → 고정 시드 정확성 테스트 → 동일 조건 A/B 성능 시험 → 결과 기록 → commit** 순서로 완료한다. 위험도가 낮고 서로 독립인 부분은 병렬 개발할 수 있으나 성능 비교는 단계별로 수행한다.

### Phase 0 — Benchmarks, 관측 계약, 안전 기준선 [MANDATORY]

**목적:** 16.4 games/s가 어디서 소비되는지 반박 가능한 데이터로 확정한다.

**구현 항목**

1. branch/ref, exact dataset+manifest SHA, pinned oracle, GPU PCI IDs, 온도, 전력 제한, 명령, random seed, 모델/optimizer snapshot을 실험 `run_manifest.json`에 기록한다.
2. `native_collector._collect_round`의 native observation packing, `parse_view`, `ObservationBatch` 변환, `cat`, H2D, encoder, per-branch candidate pack, sampler, D2H action extraction, buffer record, Rust step 각각을 분리 계측한다.
3. CUDA 비동기를 구분한다. GPU 구간은 `torch.cuda.Event` 및 제한된 `torch.profiler` CPU+CUDA trace로 측정한다. 전체 wall 측정 시작/끝에는 명시적 GPU synchronization을 적용한다. 프로파일러 사용으로 발생한 slowdown 수치는 별도 표기한다.
4. 배치 크기, 게임 길이, 팀, 후보 개수, branch count, 순수 GPU 커널 시간과 CPU가 대기하는 시간을 함께 로깅한다.
5. `torch` memory allocated/reserved와 GPU utilization, 장치 온도/스로틀, rank별 CPU%/NUMA/page-fault/RSS를 기록한다. 필요한 경우 `perf`, `numastat`, `pidstat`, `nvidia-smi`의 **이미 설치된 도구**만 사용한다.
6. **기능 완전성 감사:** fixed + ragged 관측 각각에 대해 `Rust field → packed wire → model input` matrix와 시드별 parity fixtures를 만든다. `category_vocab=8192` placeholder의 최종 Dex ID 의미 및 범위를 감사한다.
7. 기존 smoke의 `committed_matches` target/actual 차이를 별도 테스트로 재현하고 카운터를 수정한다. 고유 checkpoint/optimizer step 재개 의미도 확인한다.
8. `readiness_check`를 저장하되 실패 6개를 성능 문제로 오인하지 않는다. 본학습은 여전히 금지한다.

**산출물:** `docs/perf/BASELINE.md`, `runs/perf/<run_id>/baseline.json`, Chrome trace(optional), fixed/ragged feature coverage matrix, `scripts/bench_pa3_end_to_end.py` 또는 기존 runner 확장.

**통과 조건:** 동일 설정에서 반복 3회 이상의 real-policy throughput, CPU/GPU stage wall accounting, seed/branch/legality parity, benchmark와 실제 training pipeline을 분리한 지표가 있어야 한다.

### Phase 1 — Native observation batch-first data path [HIGHEST ROI CANDIDATE]

**목적:** 개별 관측 bytes 및 수천 개 Python tensor 생성과 `cat` 제거.

**변경 지점:** `engine/src/python.rs`, `engine/python/pa3_engine/observation.py`, `agent/types/observation.py`, `agent/train/native_collector.py`, `engine/python/bench_observation.py`.

**구현 순서**

1. 기존 `observe_fixed_batch`의 fixed buffer와 ragged sidecar를 사용하는 **opt-in** 경로를 구축. 기본 schema는 유지한다.
2. Python은 `parse_batch` 기반 하나의 NumPy structured view로 필요한 fixed fields를 얻는다. 필드별 최소한의 contiguous 변환을 **batch 단위**로 수행한다. `u16` category → embedding index dtype, `u8` flags → bool, `f32` float 변환의 복사 횟수를 계측한다.
3. `ObservationBatch.from_native_payload`에 진짜 batch path를 제공하고, `cat` per-view, repeated default role/side vector clone, per-row `.to()` 제거. 역할/진영 벡터는 사전 계산된 불변 template에서 의미를 보존하면서 vectorized broadcast.
4. ragged buffers는 feature 감사 결과에 따라 batch offsets/metadata로 넘기거나 기존 정보와 같은 형태로 결합한다. **silently ignore 금지.** 모델이 모든 필요 정보를 소비하도록 별도 readiness 계약을 유지한다.
5. `NativeEngine`에서 immutable `Encoder::from_validated_dex`를 매 호출 재생성하는 비용을 측정하고 안전하면 객체 생성 시 캐싱한다. borrowed Dex lifetime, immutable shared state, thread safety를 확인한다.
6. 버퍼 lifetime을 API 계약으로 강제한다. Python zero-copy NumPy view가 접근 중인 Rust 버퍼를 다음 batch에서 덮어쓰지 않도록 **owned immutable Python bytes 또는 ref-counted storage**를 제공하거나 명시적으로 안전한 snapshot copy를 쓴다. GPU에 보내기 전 overwrite를 금지한다.
7. CPU→GPU는 pinned contiguous staging이 이득인 경우에만 `non_blocking=True`와 CUDA stream으로 이관한다. CPU view zero-copy가 **GPU H2D zero-copy**를 뜻하지 않는다는 점을 명시한다.

**검증:** 동일 frozen snapshots에서 field-by-field bitwise parity 및 known-mask parity, 별도 ragged sidecar fingerprint parity, PvP/opponent hidden leak 검사, 모든 team/seed probe 및 PPO smoke.

**성능 지표:** one request당 Python objects, decoded bytes, bytes copied, `observation_cpu_ms`, H2D ms, sustained games/s. 36.97→2.45ms standalone 결과를 전체 games/s speedup으로 쓰지 않는다.

### Phase 2 — Rust packed legal action + prefix path [HIGHEST ROI CANDIDATE]

**목적:** 약 409만 후보의 Python object/tensor 반복 생성 제거.

**변경 지점:** `engine/src/actions.rs`, `engine/src/python.rs`, `agent/types/actions.py`, `agent/types/requests.py`, `agent/train/native_collector.py`.

**구현 순서**

1. 기존 `candidates_batch`를 oracle API로 유지하고 `candidates_packed_batch`를 새로 도입. flat `u8/u16` action records, `u32` per-request offsets 또는 `[B, P, 6]` 고정 배치, `u8` mask, int entity/move-token refs, per-request candidate_count를 Rust에서 한 번에 생성한다.
2. `[B, max_candidates, 6]`의 `max_candidates`는 **실제 full M-C legal bound**를 측정하여 정한다. 현재 64 padding을 넘는 경우 silent clipping 금지, 명시적 capacity error 또는 bounded dynamic overflow path. `Struggle/NO_SLOT`, switch/replacement, Mega resource, target-location 등 특수값 보존.
3. Rust에서 이미 알고 있는 token layout/action ref mapping을 이용하여 Python의 `AtomicAction.from_tuple`, `ActionRef.resolve` 반복을 배치화한다. wire schema에 version, count, offset, dtype, little-endian 계약을 명기.
4. prefix `j`의 후보는 선택된 `0..j-1` 행동에 의존한다. **반드시 선택 직후 Rust가 정확한 다음 level mask를 생성**한다. 시작할 때 모든 후보를 미리 고정하는 최적화 금지.
5. 후보 묶음은 NumPy/torch의 단일 텐서 생성으로 전달. `nn` 입력에 복제한 `u8`/`u16` raw values를 그대로 입력하지 않고 명시적 dtype 변환 및 ID 범위 검사.
6. bench에서는 대규모 후보 수뿐 아니라 빈 후보, singleton, branch 수 1–4, 최대 64 초과 조건과 invalid-prefix를 검사한다.

**검증:** legacy candidate tuple path 대비 모든 prefix마다 집합/순서/mask가 동일하거나 canonical mapping이 증명된다. 동일 policy 및 seed로 sampled action legality 100%, selected-prefix learner recompute 정확히 일치.

### Phase 3 — GPU↔CPU synchronization and model inference optimization

**목적:** 한 level 내 수백 개 선택을 개별 `.item()`으로 가져오지 않고 GPU 연산의 실효 처리량을 높인다.

**변경 지점:** `agent/train/native_collector.py`, `agent/model/pa3_model.py`, `agent/model/encoder.py`, `agent/model/scorer.py`.

1. 각 branch level의 `step.pick`/value/logprob 등을 **한 번의 D2H bulk copy**로 수령한다. CPU 쪽 Python list 반복 자체가 필요해도 GPU에서 개별 scalar를 꺼내지 않는다.
2. `bool(mask.any())` 등 GPU scalar → Python 제어 흐름은 host-side shape/schema assertion 또는 비동기 GPU 순수 계산으로 치환한다. **완료를 기다려야 하는 선택 prefix 의존성 자체는 제거할 수 없다.** 불필요한 sync만 없앤다.
3. `torch.no_grad()`와 `torch.inference_mode()`를 A/B한다. 모델 `.eval()` 유지, actor state와 PPO learner autograd 영역 분리. inference tensor가 learner autograd에 직접 섞이지 않도록 buffer에 저장되는 데이터 소유권을 점검한다.
4. **FP32 control**, **FP16 autocast candidate**를 같은 GPU, 같은 모델, 같은 shapes에서 비교. V100 sm_70에서 Tensor Core alignment, layernorm/softmax dtype, SDPA math 실제 backend를 profiler로 확인한다. FP16 전후 샘플 행동이 bitwise 같을 필요는 없지만 같은 실행 precision과 checkpoint로 recomputed logprob 수치가 허용 오차 안에 있어야 한다. probabilities, logprob, ratios, gradients, reduction은 FP32.
5. encoder는 요청당 단 한 번 실행하고 cached token representation을 1–4 branch에서 재활용한다. no re-encode per branch invariant를 test한다.
6. batch sizes per GPU `64, 128, 256, 384, 512` 등 sweep을 실측. GPU kernel occupancy, latency p50/p95, CPU queue/short cohorts와 동기화 빈도까지 보고 **real completed games/s**로 선택한다.
7. dynamic allocations 최소화: 모델 입력과 후보/GRU hidden scratch preallocation 가능 범위를 실험. masked logits, entropy, KL 의미를 바꾸는 fusing 금지. 작은 kernel issue가 남을 때만 CUDA Graph를 별도 feature flag로 시험, pinned runtime과 그래프 캡처 제한 확인. 기본 eager path는 유지.

**검증:** Torch CPU/CUDA profiler 및 CUDA Events; illegal 0, NaN/Inf 0, PPO sampled/recomputed ratio≈1, deterministic FP32 reference, stochastic distribution/seed drift 별도 평가. temperature 1.0, categorical exploration 정책 유지.

### Phase 4 — Compact, bounded rollout memory + streaming PPO

**목적:** 10,240경기 ≈25GiB 호스트 RSS 폭증 방지, 큰 PPO iteration과 OOM 회피.

**변경 지점:** `agent/buffer/rollout_buffer.py`, `agent/types/observation.py`, `agent/ppo/learner.py`, `agent/train/native_collector.py`.

1. Python `RolloutRow` per-action 객체 중심 구조에서 **typed columnar packed arrays**로 전환. observation feature는 native dtype (`u16`, `f32` 또는 **검증된 lossless/acceptable float precision**, masks packed)으로 보존. action mask 및 prefixes는 fixed slab + variable offsets.
2. 재사용 가능한 row/slab chunk로 append-only storage; 만료된 중간 Python 객체는 버리고 rollouts는 CPU 메모리에서 minibatch indexing을 지원한다. growth bound와 backpressure 설정.
3. gradient/Transformer activation은 rollout에 영구 보존하지 않는다. 단, **동일 관측의 learner recomputation에 필요한 모든 features와 exact per-level candidate tables, old logprob, value, rewards, match/side keys, done/prefix, policy ID**는 보존.
4. target natural game/learner decision 조건을 채운 뒤 **모든 진행 중 경기의 자연 종료**를 기다린다. overshoot는 잘라 버리지 않고 실제 수로 counter/learning-rate 스케줄에 반영한다. 더 짧은 게임을 우선 계산해 경험 분포를 왜곡하지 않는다.
5. 64GiB 시스템 RAM과 기존 서비스의 실제 사용량을 고려해 Full Spec host 전체 48GiB 및 rollout 24GiB soft budget을 기본 한계로 **measured RSS**에 안전 상한 설정. memory-mapped spill은 시스템 디스크 여유/쓰기 성능 확인 후 사용. 캐시/파일 수명 및 checkpoint crash recovery 문서화.
6. PPO minibatch 4,096를 유지하면서 GPU로 필요한 microbatch만 비동기 전달. CPU pinned staging 소규모 재사용, page-lock 과다 할당 금지.
7. `GAE` match-side grouping, actor rows, singleton masking, bootstrapping, optimizer step count, loss scaling과 padding/sample weights 동작 유지.

**검증:** 1,024→10,240→확장된 bounded rollout에서 RSS vs row count 추세, 완주/보상 합, minibatch 데이터 parity, finite 4-epoch PPO, checkpoint resume, overshoot counter invariants. RAM 사용량이 선형 폭증하는 설계를 통과시키지 않는다.

### Phase 5 — Native engine CPU scaling, NUMA, pipeline concurrency

**목적:** 16 workers가 3–4.8 CPU 코어 상당만 활용하는 문제와 관측 병렬화 한계를 분석.

**변경 지점:** `engine/src/batch.rs`, `engine/src/python.rs`, `engine/src/observation.rs`, rank launcher.

1. `rayon` worker pool의 실제 parallel regions를 `perf`/flamegraph/스레드 CPU times로 파악한다. `reset_batch`, `observe_encoded_batch_into`, `step_batch` 별 병렬 실행 비중 검증.
2. PyO3 경계에서 Rust-only 장시간 작업에 `Python::detach`가 필요한지 해당 pinned PyO3 버전에서 확인하고, **소유권과 worker GIL 안전성이 입증된 경우만** 적용한다. Python 객체 생성은 detach closure 바깥에서 처리.
3. `1, 2, 4, 8, 12, 16, 20` workers × `256, 512, 1,024` env/rank 조합의 scaling sweep. 총 env=2,048 목표를 유지하되 tune 과정에서 세부 rank allocation은 변경 가능. 각 실험의 team/seed natural duration을 통제.
4. CPU affinity 및 NUMA memory bind를 각 actor/rank 시작 전에 적용. GPU0 PCI `05:00.0`→NUMA0, GPU1 PCI `84:00.0`→NUMA1 실제 시스템 topology 대조. context switching, remote node memory, CPU caches 및 DRAM bandwidth 확인.
5. batching 크기와 memory locality를 개선하고 **가능하면 CPU에서 다음 독립 environment cohort의 관측을 준비하는 동안 GPU가 이전 cohort 추론**하도록 bounded double-buffer/in-flight pipeline 시험. 동일 env의 prefix 의존 순서를 지키고 policy snapshot version 혼합 금지.
6. native work 7%라는 관측은 현재 smoke 경로 기준일 뿐. Python 병목 제거 후 Rust engine/encoding 비중이 증가하면 그때 SIMD/SoA, feature encode vectorization, cached tables, compact handles 등의 Rust 최적화를 우선한다.

**검증:** zero data race, consistent observations, deterministic single/rank regression, CPU time per stage, worker scaling, 1-rank throughput 및 memory locality 개선.

### Phase 6 — Dual V100 actor and DDP PPO: exact same policy, real dual-GPU benchmark

**목적:** single-GPU에서 최적화한 경로를 2×Xeon + 2×V100로 수평 확대.

1. rank0=GPU0/NUMA0/1,024 env/16 workers, rank1=GPU1/NUMA1/1,024 env/16 workers로 시작. **모델 가중치의 동일 frozen policy ID**를 두 카드에 각자 로드. per-step activation/hidden state를 GPU 사이로 전송하지 않는다.
2. 실제 PA3 actor의 two-rank runner 신규 작성/확장. `scripts/run_actor_pair.py`는 random policy 벤치마크이므로 **실정책 테스트로 오기재 금지**. 두 rank의 local games/s, CPU/GPU metrics 및 wall-aligned 합산 처리량 측정.
3. NCCL DDP는 기존 설정 `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=0`, SYS/host-staged 경로 사용. global PPO minibatch 4,096 = rank별 2,048, microbatch 256 × accumulation 8. 첫 7 microbatch의 forward/backward를 `no_sync()`로 감싸고 마지막 microbatch에서 gradient sync. frozen policy collection과 learner update phase 경계 분명히 유지.
4. GPU1 150W 및 냉각 성능 차이로 느린 rank가 전체 wall을 제한할 수 있다. rank env 수와 microbatch workload balance 조정 후 **실제 two-rank end-to-end throughput**이 one-rank보다 이득인지 판정.
5. GPU0 learner fallback은 **Full Spec 조건**대로 global batch 4,096, 256 × 누적 16, history/optimizer/counters 유지. GPU1은 승인된 역할의 수집 또는 evaluation으로 활용 가능. 자동 드라이버 reset 또는 전력 상향 금지.
6. NCCL 시간/버킷, GPU utilization, wall idle %를 별도로 기록하고, GPU 두 장이 64GiB unified VRAM이 아니라는 점을 운영 문서에 명시한다.

**검증:** 동일 weights로 각 rank 단독과 dual-run action/logprob correctness, full DDP gradient accumulation equivalence (수치 허용오차), checkpoint parity, zero hangs, zero operational errors, rank scale ratio. 상대 history와 evaluate는 장기 학습 전에 integration 계약 완료 필요.

### Phase 7 — integrated optimization, adaptive tuning, long smoke

1. 채택된 feature flags만 모아 optimised-by-default 경로를 생성하되, legacy path는 regression reference로 남긴다.
2. 최적화된 real-policy benchmark를 **1 rank vs 2 ranks**, FP32 vs approved FP16, several batch/workers permutations로 재실행한다.
3. 1k, 10k, 더 큰 **명시적으로 승인된 bounded smoke**에서 속도 유지, 메모리 상한, DDP, 4 PPO epochs, checkpoint/restart 검증. GPU 시간이 빨라져도 `readiness_check 16/16` 이전에는 본학습 금지.
4. actor/sample, learner/GAE, optimizer, checkpoint/metrics, 평가까지 포함한 **honest all-in games/s**를 보고한다. smoke의 16.4 collector-only와 혼동하지 않는다.
5. 각 변경은 게임 결과나 policy 정확성에 영향을 주면 revert 후 수정. 확실한 성능 이득이 없으면 새 native code 복잡성을 유지하지 않는다.
6. 마지막은 `BASELINE → stage-by-stage speedups → combined speedup → remaining bottlenecks → recommendation` 형태로 요약하고, validated high-performance mode를 config로 보존한다.

---

## 4. 우선순위 트리와 병렬 개발 지침

```text
P0 baseline + schema/probability audit
 ├── 1A: Rust fixed observation + immutable encoder cache
 ├── 1B: Rust packed legal candidates + action IDs
 ├── 1C: per-level GPU bulk D2H, sync removal
 └── 1D: compact buffer scaffolding
          ↓ integration gates and benchmark
P2 FP16 / inference_mode / batch tuning
          ↓
P3 NUMA workers / bounded actor pipelining
          ↓
P4 dual-rank actual-policy actors + DDP learner
          ↓
P5 long bounded smoke / end-to-end report
```

- **병렬 가능한 개발:** 1A, 1B, 1C, 1D는 별도 파일 중심으로 동시에 작업 가능. 하지만 wire schema와 Rust/Python API는 통합 담당자가 먼저 versioned contract를 확정한다.
- **직렬 통합:** 같은 seed의 기준선을 보존하고 한 변경씩 merge하여 throughput을 측정. 성능 개선 원인을 추적 가능하도록 separate PR/commit을 유지한다.
- **한 세션에서 모두 시행 가능:** 세션 목표가 위 Phase 0–7이라도 결과를 보기 전에 모든 옵션을 무조건 ON 하지 않는다. 전체 run은 gated pipeline이므로 실패 단계는 수정하고 이후 단계로 무리하게 진행하지 않는다.

---

## 5. 실측 실험 계획과 재현성

### 5.1 고정 기준선 명령 예시

명령은 repository root에서, **miniDC에 해당 브랜치가 checkout되고 프로젝트 venv가 동작하는 조건**을 전제로 한다. 실제 스크립트 옵션은 `--help`로 확인하고 조건 및 실행 SHA를 기록한다.

```bash
# 환경/서비스 점유 확인. 읽기 전용.
git rev-parse HEAD
nvidia-smi --query-gpu=index,pci.bus_id,name,memory.used,memory.free,power.limit,temperature.gpu,utilization.gpu --format=csv

# Rust-only engine. AI 성능으로 보고하지 말 것.
POOL_REPEAT=20 ./target/release/examples/pool_run_report

# 관측 경로 비교.
PYTHONPATH=engine/python .venv/bin/python engine/python/bench_observation.py --envs 256 --rounds 10

# 기존 single GPU real-policy bounded smoke.
PYTHONPATH=engine/python:. .venv/bin/python scripts/run_ppo_smoke.py \
  --games 10000 --envs 1024 --workers 16 --device cuda:0 \
  --report runs/perf/single-baseline.json \
  --checkpoint runs/perf/single-baseline.pt

# 필요한 경우 별도 report path/seed를 지정해 A/B 재실행.
```

**유의:** 실제 smoke 실행은 시간이 소요되고 GPU 서비스를 점유할 수 있으므로 가용 리소스 확인 후 승인된 범위의 실험만 진행한다. random-policy speedtest와 real-policy smoke를 같은 성능 결과로 합치지 않는다.

### 5.2 실험 행렬

| 축 | 비교 설정 | 기록할 결정 기준 |
|---|---|---|
| Observation | legacy per-view vs fixed batch | parity, parse+convert ms/view, H2D time, games/s |
| Candidates | Python per-candidate vs Rust packed | # Python allocs, candidate throughput, legal parity |
| GPU sync | per-scalar vs per-level bulk | D2H stalls, host idle, games/s |
| Precision | FP32 vs FP16 autocast | kernels/ms, sampled/recompute policy integrity, NaN/Inf |
| Batch | 64/128/256/384/512 infer groups | occupancy, p95 latency, games/s |
| Engine | workers 1/2/4/8/12/16/20 | cores used, NUMA memory, complete games/s |
| Host buffer | row objects vs columnar slab/spill | RSS peak, GB/1000 games, sample replay correctness |
| Rank | GPU0, GPU1, GPU0+1 | **matched real-policy** local and total games/s, DDP sync |
| PPO | 1k/10k/approved longer bounded run | GAE/step/counter, wall all-in speed, memory slope |

### 5.3 매 실행 report schema 예시

```json
{
  "run_id": "unique-run-id",
  "git_sha": "...",
  "dataset_id": "mb-mc-v3-userteam-all-train",
  "dataset_sha256": "...",
  "policy_sha256": "...",
  "mode": "real_policy_collect | full_ppo_iteration",
  "seed_panel_id": "...",
  "rank_count": 1,
  "gpu_pci_ids": ["05:00.0"],
  "envs_per_rank": 1024,
  "workers_per_rank": 16,
  "games_natural_complete": 0,
  "decisions": 0,
  "turns": 0,
  "operational_errors": 0,
  "illegal_actions": 0,
  "actor_collect_wall_s": 0.0,
  "ppo_update_wall_s": 0.0,
  "checkpoint_eval_wall_s": 0.0,
  "all_in_committed_games_per_s": 0.0,
  "observation_cpu_wall_s": 0.0,
  "gpu_forward_event_ms": 0.0,
  "candidate_pack_cpu_wall_s": 0.0,
  "h2d_bytes": 0,
  "d2h_bytes": 0,
  "rss_peak_gib": 0.0,
  "gpu_peak_gib_by_rank": [],
  "logprob_recompute_max_abs_diff": 0.0,
  "ppo_nan_inf_count": 0,
  "readiness_pass": 10,
  "readiness_total": 16
}
```

### 5.4 하드 가드레일

- **완결 정확성:** `natural_matches = wins + losses + draws`, zero abort/truncate, zero unsupported effect silently ignored, zero invalid action.
- **행동 일치:** oracle 및 legacy path 기준 각 stage의 legal candidate sets, prefix mask, resource flag, target location, private/public 정보 동일.
- **PPO 일치:** sampled/recomputed logprob 최대 오차는 기존 `1e-4` smoke gate 유지 또는 엄격하게 개선. 혼합정밀도 parity는 tolerance를 사전에 명시하고 같은 forward precision으로 recompute. Ratio, KL, grad norm, reward, GAE 안정성 유지.
- **입력 보존:** fixed+ragged feature inventory 100% 확인, 마스크/범주/의미 이벤트 훼손 금지. 아직 미구현인 정보를 임의로 0으로 채우고 지원한다고 주장하지 않는다.
- **성능 판단:** 최적화 변경은 동일 조건 반복 최소 3회에서 안정적인 향상 여부를 확인. 실무 채택 기준은 통계 변동과 코드 복잡성을 고려하며, 우연히 한 번 빨랐던 경로를 채택하지 않는다.
- **메모리:** 서비스 공존 기준으로 64GiB 시스템 중 사용 가능한 메모리를 먼저 기록. System RAM과 GPU VRAM budget을 구분. 전체 host 48GiB, rollout 24GiB, pinned staging 1GiB 및 GPU/card 28GiB soft budget을 확인. Pinned host memory를 무제한 할당 금지.
- **Readiness:** 전체 규정 legality 및 differential readiness의 16/16 통과는 독립된 최종 요구사항. 학습 풀만 통과했다고 본학습 허가 아님.

---

## 6. 하드웨어 사용 극대화를 위한 세부 점검

### CPU

- 코어당 성능이 최신 CPU보다 낮은 듀얼 Broadwell 기반이므로 Python 객체 분해, branchy loops, 작은 allocations 절감이 특히 중요하다.
- `numactl --cpunodebind=N --membind=N`으로 rank별 메모리 locality 설정. OS가 보고하는 actual cpulist 및 PCI-NUMA 노드를 먼저 검사한다.
- 16 Rust worker를 기본값으로 고수하지 말고 8, 12, 16, 20 workers에서 task 분해와 관측 패킹 병렬 효율 테스트. Hyper-thread 80개를 물리 80코어로 간주하지 않는다.
- Rayon pool과 PyTorch intra-op CPU thread 경쟁 확인. `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `RAYON_NUM_THREADS=16`은 기존 출발점이며 변경은 벤치마크 승인 하에 진행.
- Rust encoder 캐싱과 데이터 SoA 개선을 우선하고, 근거 없는 CPU governor, BIOS, BIOS microcode, hugepage, kernel tunables 변경 금지.

### GPU

- GPU0 175W, GPU1 150W 유지. `nvidia-smi`로 SM utilization, memory-controller utilization, temperature, throttle reason을 기록. VRAM 3GiB 사용만으로 GPU compute idle이라고 단정하지 않는다.
- 두 GPU는 같은 모델 두 복제본으로 사용. 서로 다른 카드의 냉각과 전력 차이로 인한 batch imbalance를 NUMA rank 단위에서 조정.
- V100 Volta sm_70에 맞는 FP16/FP32 eager 수치 경로를 먼저 최적화. BF16/FP8/TF32 등을 강제로 쓰지 않는다. 외부 `sm70-attn` llama.cpp 라이브러리를 PyTorch attention에 무검증 이식하지 않는다.
- `CUDA_VISIBLE_DEVICES`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, existing NCCL config를 기록하되 새 프로세스의 local rank numbering과 실제 PCI bus ID를 분리해 사용.
- pinned host buffer 및 overlap은 CUDA event/profiler로 **실제 H2D + compute overlap 여부**를 증명. stream 동기화/버퍼 lifetime 레이스를 추가하지 않는다.

### 서비스 및 데이터

- `dsh-web`, `llama-swap` 등 실행 중인 서비스 강제 종료 금지. GPU resident 모델이 있으면 현행 문서의 승인된 unload 절차만 수동 사용 가능.
- 새 NVMe/SSD가 설치됐는지 추측하지 말 것. spill을 도입한다면 실제 디스크 free space, mount, IOPS, 디스크 쓰기량을 측정하고 RAM fallback을 제공한다.
- 기존 데이터셋, Showdown pinned oracle, Full Spec 문서, vendor를 성능 편의로 바꾸지 않는다.

---

## 7. 책임 분리 및 예상 파일 수정 범위

| 작업 | 주요 변경 파일 | 최소 테스트 |
|---|---|---|
| Benchmark/profiler | `scripts/run_ppo_smoke.py`, `docs/perf/*`, 신규 `scripts/bench_*.py` | baseline reproducibility, CUDA Event correctness |
| Observations | `engine/src/python.rs`, `engine/src/observation.rs`, `engine/python/pa3_engine/observation.py`, `agent/types/observation.py`, `agent/train/native_collector.py` | field parity, ragged parity, leakage, Rust test, PyO3 test |
| Packed candidates | `engine/src/actions.rs`, `engine/src/python.rs`, `agent/types/actions.py`, `agent/types/requests.py`, `agent/train/native_collector.py` | every-prefix legal parity, no truncation, move/target/mega tests |
| GPU inference | `agent/model/pa3_model.py`, `agent/model/scorer.py`, `agent/model/encoder.py`, collector | policy distribution, sample/recompute, FP32/FP16 benchmarks |
| PPO buffer | `agent/buffer/rollout_buffer.py`, `agent/ppo/learner.py`, collector | GAE, replay, trajectory grouping, checkpoint, memory scaling |
| CPU/NUMA | `engine/src/batch.rs`, `engine/src/python.rs`, rank launcher | NUMA mapping, worker scaling, correct completion |
| Dual GPU | new actual-policy 2-rank launcher, learner DDP coordinator, metrics | gradient parity, no_sync, NCCL, actual throughput |

권장 Git workflow: `optimization/pa3-realpolicy-throughput` 신규 브랜치 생성, Phase별 소규모 커밋, 주요 단계마다 checkpoint 태그 또는 커밋 SHA 기록, clean tree 및 push. 사용자 동의 없이 main 강제 push 금지.

---

## 8. 위험 및 실패 대응

| 위험 | 조기 신호 | 대응 |
|---|---|---|
| zero-copy buffer overwrite/use-after-free | 동일 시드에서 관측 값 변화, 비결정적 오류 | buffer lifetime 강제, immutable owner 또는 safe copy로 fallback |
| ragged 정보 손실 | model input feature parity mismatch, full spec feature 미표현 | feature inventory 우선, sidecar 유지, 별도 semantic encoder 구현 없이는 승인 금지 |
| prefix mask stale | invalid action, learner ratio drift | 매 선택 후 NativeEngine candidate 갱신, deterministic fixture |
| FP16 확률 변형 | recompute diff, KL/entropy drift, NaN | FP32 critical math, forward precision parity, feature-flag rollback |
| CPU↔GPU deadlock | profiler idle/교착, worker timeout | scoped GIL detachment, request ordering, bounded queues, watchdog |
| Pinned RAM/host OOM | RSS 상승, swap, OS kill | bounded staging, backpressure, measured disk spill |
| GPU1 slow rank | dual-rank aggregate speedup 낮음, DDP barrier 증가 | rank balancing, gradient sync profiling, validated single-learner fallback |
| 성능 수치 착시 | 짧은 게임 위주, `games/s`만 증가 | 동일 팀/seed panel, turns/decisions, all-in wall accounting |
| Smoke vs full readiness 혼동 | criterion FAIL인데 100M launch | 16/16 readiness independent blocker 계속 유지 |

---

## 9. 최종 수락 조건과 완료 보고서

### 9.1 Required deliverables

- [ ] Baseline (`real policy`, `GPU0`, 1,024 envs, 16 workers) 3회 이상 및 profiler trace
- [ ] Optimized fixed observation path, ragged/features parity
- [ ] Packed legal actions, exact prefix-dependent masks, no overflow/truncation
- [ ] Bulk GPU selection D2H, synchronization hotspots removed
- [ ] Verified FP16 vs FP32 inference result and selected safe default
- [ ] Bounded columnar rollout memory, streaming PPO and actual committed-match clock
- [ ] CPU/NUMA/worker scaling matrix
- [ ] Dual GPU **actual policy** actor + 2-rank DDP smoke, without claiming random actor numbers
- [ ] End-to-end natural games/s benchmark with collection + PPO 4 epochs + metrics/checkpoint
- [ ] Full regression green, all feature contracts preserved, no new operational errors
- [ ] `readiness_check` current PASS/FAIL reported honestly, 100M training not run
- [ ] Git commits, exact SHA, performance JSON and `docs/perf/FINAL_REPORT.md`

### 9.2 단계별 결과 테이블 양식

| Stage | Frozen baseline games/s | New games/s | Relative speedup | CPU RSS GiB | GPU0/GPU1 peak GiB | Correctness | Decision |
|---|---:|---:|---:|---:|---|---|---|
| Baseline | 16.4 (historical) | re-measure | n/a | ~25 historical | 3.0 / n/a historical | record | baseline |
| Fixed obs | measured | measured | measured | measured | measured | pass/fail | keep/revert |
| Packed candidates | measured | measured | measured | measured | measured | pass/fail | keep/revert |
| Sync/inference | measured | measured | measured | measured | measured | pass/fail | keep/revert |
| Buffer/CPU | measured | measured | measured | measured | measured | pass/fail | keep/revert |
| Dual GPU | matched single | matched dual | measured | measured | measured | pass/fail | keep/revert |
| Full PPO all-in | new all-in baseline | optimised all-in | measured | measured | measured | pass/fail | keep/revert |

**Throughput target:** **NO ARTIFICIAL CEILING.** 1,000 games/s는 첫 이정표, 2,000/5,000/10,000+는 상향 탐색용 **가설적 측정 레벨**이다. 특정 수치에서 프로젝트를 자동 종료하지 않는다. **성능 향상의 수익, 구현 복잡도, 안전성과 정확성, 시스템 한계가 확인될 때까지** 실측과 개선을 반복한다. 어떤 레벨도 달성 예측치가 아니며, `actor-only`와 `full PPO all-in`을 구분해야 한다.

### 9.3 Final agent report template

1. `git SHA`, files, source/manifest version, hardware mapping and service safety check.
2. Optimization list and stage-by-stage **measured** deltas and confidence/variation.
3. Explicit actor-only and full PPO end-to-end throughput, single and dual GPU, CPU/GPU hardware telemetry.
4. Correctness tests including fixed/ragged data, hidden information, prefix masks, FP16 logprobs, PPO update, checkpoint/recovery, full engine smoke.
5. Host/GPU memory growth and actual 10k/long bounded smoke results.
6. `readiness_check` gates PASS/FAIL. Remaining full-scope mechanics blockers and independent correctness blockers.
7. Remaining top bottlenecks and next high-value, low-risk optimizations.
8. Successful commits pushed, clean tree confirmation, reproducible run commands and report paths.

---

## 10. 1,000 games/s를 넘어서는 무상한 성능 전략 (v1.1 신규)

### 10.1 목표의 재정의: threshold가 아니라 frontier

- **최적화 목적함수:** `(실제 완결된 학습 게임 수) / (rollout 수집 + GAE + PPO 전체 4 epochs + 모델 동기화 + 평가/체크포인트 등 정상 운영 시간)`인 **steady-state all-in committed games/s**를 최대화한다. 별도 `real-policy rollout games/s`도 필수로 기록한다.
- **1,000 games/s:** 기존 목표치를 상한에서 **첫 검문점**으로 낮춘다. 1,000 도달을 근거로 진행을 중단하지 않는다.
- **2,000 / 5,000 / 10,000+ games/s:** 엔지니어가 측정 중 상태를 구분하기 위한 **도전적 시나리오 구간**이다. V100 두 장에서 도달한다고 주장하지 않는다.
- **성능 정의를 바꾸는 지름길 금지:** random/heuristic 상대를 써서 `real-policy`라고 보고하기, 경기 조기 종료, 미완료 게임 완결로 계산하기, 짧은 경기만 선택하기, 전체 팀풀 축소, 불법 후보 생략, 정보 마스킹 완화, PPO epoch/학습량 축소 금지.
- **증분 우선:** 동일 종단점 비교에서 gain이 있는 변경을 채택하고, gain이 미약한 개조는 지우거나 실험 기능으로 격리한다. `games/s`가 증가해도 `decisions/s`, `turns/game`, 행동의 분포, 학습 신호의 수치 안정성, 사용 RAM/전력 등이 악화하면 채택하지 않는다.

| 표지 | 의미 | 다음 행동 |
|---|---|---|
| `16.4` | 2026-10-08의 **single-GPU, rollout-only** 기준선 | 동일 조건으로 재측정 |
| `100` | 중간 품질/속도 측정 구간 예시 | profiler 병목 갱신 |
| `500` | 기존 설계상 처리량 목표 근접 구간 | 두 rank end-to-end 및 buffer 안정성 검증 |
| `1,000` | **중간 성능 이정표, 종료점 아님** | 과열/NUMA/PCIe/후속 병목 분석, Phase 8–10 진행 |
| `2,000`, `5,000`, `10,000+` | stretch checkpoint, **예측치 아님** | 실제 도달 때만 결과 인정, 계속 병목 추적 |

**주의:** Rust-only 약 2,585 games/s는 AI 추론/학습을 포함하지 않으며, 전체 시스템의 직접적인 물리적 상한 또는 보장되는 하한이 아니다. 다중 코어 환경이라는 이유로 이 수치를 곱해 실제 full PPO 속도라고 주장해서도 안 된다.

### 10.2 최적화 회전문: 가장 큰 병목을 계속 다시 찾아라

Phase 0–7을 통과한 뒤에도 다음 계측 → 개선 → 검증 루프를 반복한다.

1. **Roofline-style stage accounting:** stage별 CPU wall, CUDA event kernel time, GPU stall, queue wait, PCIe H2D/D2H, allocations, CPU cycles/branch misses/cache misses, socket-local bandwidth, DDP wait를 수집한다. 한 단계 최적화가 완료될 때마다 새 hot path를 다시 측정한다.
2. **Theoretical lower bound와 실제 floor 구분:** 실측 `min stage time`, 전력/온도 유지시간, CPU 코어 활용, GPU occupancy, 시스템 대역폭으로 포화 원인을 구분한다. 직접 측정 없이 `더 빠를 수 없다`고 주장하지 않는다.
3. **Co-optimization:** GPU 배치만 늘리면 GPU 효율은 좋아져도 CPU 대기와 동기화가 늘 수 있다. socket별 worker, env count, inference chunk, pinned staging, number of in-flight batches를 **조합 sweep**으로 최적화한다. 아무 매개변수도 고정된 신성한 값으로 보지 않되 전체 팀/학습 유효 배치/정확성 계약은 유지한다.
4. **Pareto frontier:** `all-in committed games/s`, `peak host RSS`, `latency`, `reproducibility`, `failure rate`를 함께 기록한다. 동점 성능이면 더 단순하고 안전한 경로를 선택한다.
5. **No automatic stop at 1,000:** 새로운 최적화가 재현성 있는 개선을 내는 동안 계속한다. plateau를 선언하려면 병목 기여율, 최소 3회 동일 실험, 시험한 대안과 회귀 결과가 근거로 제시되어야 한다.

### 10.3 Phase 8 — 이벤트 기반 CPU/GPU 중첩 실행 및 rolling game slots

- **Independent environment cohorts:** 같은 게임의 다음 branch는 이전 선택이 필요하지만, 서로 다른 게임/코호트 사이에는 독립성이 있다. `A cohort` GPU가 추론하는 동안 `B cohort` CPU가 관측을 준비하도록 **double/triple buffering**과 event-driven queue를 실험한다.
- **Backpressure와 fairness:** 큐 길이를 제한하고 느린/긴 게임도 끝까지 처리한다. 오래 걸리는 게임을 건너뛰거나 먼저 완료된 게임만 통계에 포함하지 않는다. 각 trajectory의 seed, policy version, action prefix, reward attribution을 유지한다.
- **Async host staging:** NUMA-local pinned slabs, reusable CPU staging buffer, 명시적 CUDA event, nonblocking H2D/D2H를 **실제 overlapping trace**로 검증한다. `non_blocking` 표기 자체만으로 overlap이라고 주장하지 않는다.
- **Minimize round-trip count:** 합법성 조건을 바꾸지 않는 범위에서 per-branch prefix-dependent candidate 재요청은 유지하되, `candidate packed output + sampled action bulk D2H + next level candidate request`를 하나의 고정 비용 단계로 최적화한다. 행동 샘플러를 무리하게 CPU로 옮기거나 prefix mask를 미리 고정하지 않는다.
- **Cohort/worker scheduling:** 1,024환경을 고정된 한 집단으로 계속 종료까지 기다려 tail을 발생시키는지 검증하고, 종료한 환경을 안전하게 즉시 재설정하는 **rolling/asynchronous slot refill**을 실험한다. 이전의 게임 배치가 전부 끝나야 새 게임을 시작한다는 불필요한 장벽은 제거하되 정상 완주, 동일 분포, 결정론적 seed/accounting을 보존한다.

### 10.4 Phase 9A — CPU 엔진의 극한 경로 (GPU 문제가 해결된 뒤)

1. `perf`/native flamegraph에서 Rust `encode_into`, candidate generation, reset, step, PyO3 buffer pack와 `Encoder::from_validated_dex`가 새 병목이 됐는지 확인한다.
2. 반복적 heap allocation과 hash/map/string resolution을 사전 계산된 immutable ID lookup으로 치환할 수 있는지 확인한다. 전투 종료 및 effect lifetime에 따른 cache invalidation 검증.
3. 가능할 경우 SIMD/vectorization, SoA 배열, cache-friendly grouping, cache-line contention과 false sharing 감소를 **profiling evidence** 후 도입한다. 기계별 CPU ISA(예: AVX2)는 실제 CPUID/compiler flags 확인 후 사용하고, 일반 빌드와 결과 동치성을 유지한다.
4. PyO3 GIL detachment 또는 `Python::detach`를 pinned PyO3 버전에서 검증한다. Python 소유 자료구조 접근 없이 순수 Rust 계산을 진행하는 동안만 허용한다.
5. `NUMA-local allocation`과 정적 팀 데이터의 복제/공유 정책을 비교한다. socket 사이 왕복 대역폭이 클 때만 변경한다. Huge pages/governor/BIOS나 전력 설정을 근거 없이 건드리지 않는다.

### 10.5 Phase 9B — V100 추론의 극한 경로 (기본 PA3 정책 불변)

1. **Profiler-first:** `encoder`, `embedding projection`, `SDPA math`, `GRU scorer`, masked softmax, categorical sampling, D2H conversion이 실제 얼마나 비싼지 별개 CUDA event로 측정한다.
2. **Static shapes and reusable buffers:** `[B,96,32/50/40]` 관측과 `[B,branch,P,6]` 후보를 bucketize하고 같은 메모리 slab를 재사용한다. 데이터 의미와 `P` 최대 정합성이 우선이다.
3. **FP16 inference:** FP32와 FP16 autocast를 실제 `sm_70`에서 비교한다. FP16을 value/logprob/softmax/GAE까지 무분별하게 확장하지 않는다. current-policy sampling 및 PPO learner recompute의 probability 의미를 지킨다.
4. **CUDA Graph 실험:** 지원되는 eager-PyTorch CUDA Graph 경로가 `torch==2.14.0+cu126` + V100에서 실제 가능한지 독립 branch에서 검증한다. 입력 shape bucket, stateless RNG/categorical sampling reproducibility, captured allocation lifetime, dynamic candidate prefixes, replay correctness를 확인한다. 기존 pinned stack을 설치/업그레이드하지 않는다.
5. **Backend alternatives (옵션, 강제 전환 금지):** PyTorch eager로 최적화 가능한 범위가 포화된 뒤에만 **동일 weight와 모델 구조의 별도 inference backend**(예: 지원되는 ONNX Runtime 또는 LibTorch 기반 Rust/C++ 호출)를 isolated prototype으로 측정한다. V100 CUDA 지원, 동적 행동 prefix, masks, end-to-end sampled logprob parity, per-rank throughput이 검증되지 않으면 채택 금지. 프로덕션 learner는 PyTorch로 유지한다.
6. **Model redesign는 승인 범위 밖:** 토큰을 88에서 줄이거나 transformer layer/head를 줄이는 방법은 이번 A-track의 속도 최적화로 처리하지 않는다. 추후 별도 ablation과 사용자 승인 대상이다.

### 10.6 Phase 10 — 두 GPU의 실제 최대 활용과 전 구간 동시 최적화

- **권장 기본:** GPU0/NUMA0와 GPU1/NUMA1에 동일 정책의 독립 actor를 놓고 두 GPU 모두 추론에 활용. 학습 업데이트는 pinned NCCL DDP 또는 실측 우위인 승인된 single-learner fallback.
- **기울기 동기화 주기:** per-microbatch가 아닌 `no_sync`에 따른 global-minibatch 경계. DDP latency/bandwidth와 GPU1 150W 지속 성능 측정. GPU1이 느린 경우 rank별 env count, microbatch assignment, overlap을 조정하되 sample-weighting과 학습 수학이 같아야 한다.
- **Use independent tasks to hide communication:** DDP 동기화와 engine worker 계산이 겹칠 수 있는지를 측정한다. 정책 버전 동결/iteration 경계를 침범하는 asynchronous stale-policy 학습은 금지한다.
- **NUMA locality:** 사용 가능한 RAM 62GiB 내에서 두 rank의 page placement, pinned staging, buffer exhaustion 및 swap 가능성을 확인한다. 상한 48GiB host / 24GiB rollout / 28GiB VRAM per card 원칙을 유지.
- **Fast rank alone vs both ranks:** `GPU0 alone`, `GPU1 alone`, `both`를 동일 workload, 완료된 정상 게임과 전체 PPO 업데이트 기준으로 모두 보고한다. 두 번째 GPU가 오히려 병목이라면 idle 시간과 수정 근거를 명시한다.

### 10.7 고급 최적화 채택/중단 기준

- **필수 채택:** 동등한 기능과 오류 없는 정확성이 입증되고, 3회 이상 동조건 벤치마크에서 변동을 넘어서는 의미 있는 end-to-end 증가가 확인된 개선.
- **조건부 채택:** throughput는 향상되나 CPU RAM, GPU 메모리, 전력 throttling, checkpoint reproducibility, 샘플 분포 등 위험이 증가하는 개선은 근거와 rollback 계획을 먼저 제시.
- **채택 금지:** 입력/행동 정보 축소, 숨겨진 상대 정보 사용, oracle 정확성 저하, policy mismatch, PPO math 변형, simulated-vs-real speed 혼동 또는 완주 집계 왜곡.
- **실측 plateau:** 동일 조건에서 주요 병목이 해소됐고, 다음 몇 개의 기술적으로 가능한 개선이 정확성/안정성 대비 유의미한 all-in 성능 증가를 내지 못했다는 기록으로만 선언.
- **Stop rule:** 1,000 games/s, 2,000 games/s 또는 어떤 숫자만으로 작업 종료하지 않는다. **물리적 및 소프트웨어적 유효 성능 frontier**를 보고하고, 대안과 trade-off를 사용자에게 제공한다. `full engine readiness 16/16`와 100M 본학습 승인 조건은 계속 독립적이다.

---

## 11. v1.1 에이전트 수행 계약: 속도 상한을 먼저 정하지 말 것

### 반드시 수행할 측정 세트

| Metric | 표기 | 목적 |
|---|---|---|
| Single GPU, PA3-8M rollout only | `single_real_policy_games_per_s` | 최초 기준선 및 1-rank 포화점 |
| Dual GPU, PA3-8M rollout only | `dual_real_policy_games_per_s` | CPU+GPU 수평 확장 효율 |
| Single GPU, full PPO 4 epochs | `single_all_in_committed_games_per_s` | 진짜 학습 비용 |
| Dual GPU, full PPO 4 epochs | `dual_all_in_committed_games_per_s` | 진짜 두 카드 학습 비용 |
| PPO learner update only | `learner_rows_per_s` | forward/backward/optimizer 비용 |
| CPU native game engine only | `engine_only_games_per_s` | 별도 참조, 실제 학습 수치가 아님 |
| GPU-only batched model inference | `model_inference_decisions_per_s` | 정책 모델 순수 throughput, 실제 학습 수치가 아님 |
| Resource limits | GPU utilization, HBM, CPU%, DRAM, PCIe, power | 병목의 하드웨어 근거 |
| Correctness | sampled/recomputed logprob, legality, readiness | 허위 성능 개선 차단 |

### 추가 우선순위 실행 순서

1. 기존 Phase 0–3에서 **관측, 후보 행동, GPU sync, inference FP16**을 개선하고 개별 성능을 측정한다.
2. Phase 4–7에서 **packed rollout, Rust CPU/NUMA, 진짜 dual-GPU PPO, all-in 측정**을 완료한다.
3. 그 시점 가장 큰 병목이 **관측/메모리**라면 fixed+ragged batch 및 pinned staging을 더 단순화한다.
4. 가장 큰 병목이 **GPU 추론**이라면 더 큰 효율적 배치, static-shape bucket, CUDA Graph, 별도 inference backend를 순서대로 시험한다.
5. 가장 큰 병목이 **Rust 게임 계산**이라면 SIMD, allocation/cache, batch scheduling과 worker NUMA를 최적화한다.
6. 가장 큰 병목이 **GPU-CPU round trip / rank barrier**라면 bulk D2H, 비동기 독립 코호트, rolling slot refill, staged pipeline을 시험한다.
7. 각 유효 개선 후 **다시 프로파일링한다**. 1,000을 넘어도 계속한다. 2,000/5,000/10,000+는 도달하면 기록할 뿐, 달성을 가정하지 않는다.

### 최종 제출 형태

`docs/perf/ULTIMATE_FRONTIER_REPORT.md`에 단계별 성능, 최댓값뿐 아니라 3회 반복 분포, 전체 학습 all-in 속도, CPU/GPU 병목 기여, full-spec 관측 계약, 16/16 readiness 별도 결과, 채택/기각한 실험, 더 빨라지기 위해 필요한 다음 설계상의 전환점을 함께 제출한다. `docs/perf/FINAL_REPORT.md`도 최신 결과를 반영한다.

---

## 12. 참고 자료 및 권위

### Project source of truth

- [Full Spec 1.1](https://github.com/mwl313/PokeAgent3.0/blob/9cbaa1fbe42264eb124f9ea3ee74dc7af13132af/docs/spec/fullspec-1.1-minidc-20261006/POKEAGENT_3_0_FULL_SPEC.md)
- [miniDC training pool speedtest](https://github.com/mwl313/PokeAgent3.0/blob/9cbaa1fbe42264eb124f9ea3ee74dc7af13132af/docs/MINIDC_TRAINING_POOL_SPEEDTEST.md)
- [PA3 learner status and PPO smoke](https://github.com/mwl313/PokeAgent3.0/blob/9cbaa1fbe42264eb124f9ea3ee74dc7af13132af/docs/PA3_LEARNER_STATUS.md)
- [Native collector](https://github.com/mwl313/PokeAgent3.0/blob/9cbaa1fbe42264eb124f9ea3ee74dc7af13132af/agent/train/native_collector.py)
- [Rust PyO3 API](https://github.com/mwl313/PokeAgent3.0/blob/9cbaa1fbe42264eb124f9ea3ee74dc7af13132af/engine/src/python.rs)
- [Native observation parser](https://github.com/mwl313/PokeAgent3.0/blob/9cbaa1fbe42264eb124f9ea3ee74dc7af13132af/engine/python/pa3_engine/observation.py)
- [Model observation adapter](https://github.com/mwl313/PokeAgent3.0/blob/9cbaa1fbe42264eb124f9ea3ee74dc7af13132af/agent/types/observation.py)
- [Active team pool and confirmed Blaze](https://github.com/mwl313/PokeAgent3.0/blob/9cbaa1fbe42264eb124f9ea3ee74dc7af13132af/docs/TEAM_POOL.md)

### External performance and correctness engineering references

- [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler)
- [PyTorch CUDA asynchronous timing and CUDA Events](https://docs.pytorch.org/docs/stable/notes/cuda)
- [PyTorch inference_mode vs no_grad](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)
- [PyO3 parallelism and Python::detach](https://pyo3.rs/main/parallelism)
- [NVIDIA CUDA Best Practices, pinned memory and stream overlap](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [PyTorch DistributedDataParallel no_sync](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)

### 자료 유형 구분

- **실측/확인된 사실:** miniDC benchmark 수치, 현재 GitHub 파일과 구현 경로, 팀 구성 및 confirmed Blaze, 현재 미완료 readiness.
- **코드 근거의 기술적 추론:** Python per-view/per-candidate allocations, GPU scalar sync 및 encoder recreation이 성능 병목 후보라는 판단. 정량적 기여율은 Phase 0 profiling으로 확정해야 한다.
- **새 설계 제안:** Rust packed candidates, streaming buffer, pinned overlap, CPU pipeline, 2-rank DDP, FP16 A/B. 구현 및 성능 검증 전에는 완료 또는 효과 달성으로 표시하지 않는다.

---

## 13. 새 에이전트에게 전달할 실행 요약

> 이 문서의 Phase 0부터 Phase 10까지를 **동일 프로젝트 내 하나의 성능 최적화 트랙**으로 수행하라. 가장 먼저 current HEAD 및 팀 Blaze 확인, native observation fixed/ragged의 모델 소비 여부와 baseline profiler를 확인한다. 이후 Rust fixed-batch observation, Rust packed candidate arrays, GPU selection bulk readback, FP16/inference-mode A/B, compact streaming PPO buffer, CPU/NUMA scaling, dual GPU 실제 PA3 inference/DDP를 순서대로 통합한다. **이후 Phase 10의 고급 최적화 루프를 반복하고 1,000 games/s 도달을 이유로 종료하지 않는다.** 매 단계에서 correctness와 동일 조건 throughput을 측정하고 실패 시 rollback한다. 원래 Rust 엔진과 PA3-8M architecture, PPO 의미, 팀 전체 범위, 하드웨어 전력 및 pinned stack을 유지한다. 모든 benchmark는 real policy인지 random인지 명확히 표시한다. **1,000 games/s를 목표 상한 또는 작업 종료 조건으로 취급하지 않고, 최적화 가능한 실제 상한을 찾는 일을 계속한다.** 단일 GPU와 듀얼 GPU 모두에서 최대 real-policy rollout 처리량과 PPO all-in 처리량을 측정하고, 1,000을 넘었다면 추가 병목에 따라 Phase 8–10 고급 최적화를 진행한다. 성능 숫자를 만들거나 16/16 readiness를 건너뛰거나 1억 본학습을 시작하지 않는다. 단계별 measured deltas, 최대 도달 성능, 실제 한계, next bottleneck을 `docs/perf/FINAL_REPORT.md`와 `docs/perf/ULTIMATE_FRONTIER_REPORT.md`로 제출하고 각 기능을 독립 commit으로 push하라.
