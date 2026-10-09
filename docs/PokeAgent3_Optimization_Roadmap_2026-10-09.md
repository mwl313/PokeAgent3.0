# PokeAgent 3.0 최적화 로드맵 및 옵션 분석 (2026-10-09)

> 작성: 아리아 (2026-10-09). 기준: 통합 정본 `main` @ `b6da590`, v4 F0/F1 실측 + 플랜 v3/v4.
> 목적: 미니DC(2×V100)의 처리량 한계를 최대한 끌어내어, 최단 시간에 최대한 강한 모델 학습에 도달한다.
> 구성: 1) 목표와 상한, 2) 개선 지점, 3) 옵션 메뉴, 4) 실행 순서, 5) 계약 변경 가이드, 6) 외부 검증, 7) 용어.

## 1. 목표와 물리 상한

- 최종 목표: 1억 자연완결 경기 본학습(승인 후)과 강한 체크포인트. 현재 처리량 기준점: **듀얼 24.67 games/s** (v4 F1 최종 패널).
- 물리 상한(계약 기준 산술): 경기당 약 0.5 TFLOP(액터 추론 + PPO 4 epochs 포함) ÷ 두 V100 fp16 합산 56.5 TFLOPS ≈ **이론 천장 110-115 games/s**. learner 시간이 0에 수렴해도 actor 현행이면 약 102 games/s (v3 플랜 산술).
- 스펙 문서의 "500-1,000 games/s" 목표는 이 하드웨어에서 물리적으로 도달 불가 → **실용 목표를 40-55 games/s로 재설정**.
- 전략: 계약(frozen-policy, 정확성)을 지키며 계약 내 최적화로 약 42-46 games/s까지 도달 → 본학습 기간을 약 46일에서 약 22-25일로 단축(추정, ±10%).

## 2. 개선 지점 (실측 시간 분해)

단일 GPU, 2,048경기, all-in 약 101초 기준 (F0 아틀라스):

| 구간 | 시간 | 비중 |
|---|---:|---:|
| 수집 (경기 생성) | 23.2s | 23.0% |
| PPO 업데이트 | 78.2s | 77.4% |
| 체크포인트 | 0.5s | 0.5% |

### 2.1 학습(PPO) 효율 = 전체의 77%

- 업데이트 구간 GPU-busy **95.4%** (커널 트레이스 실측) → CPU 측 최적화 여지는 약 5%.
- 그러나 실효 연산 효율은 피크의 약 1/3: 20.53 games/s x 0.5 TFLOP ≈ 10.3 TFLOPS vs V100 28.3 TFLOPS.
- 주 원인: 작은 micro 배치(듀얼 256 잔존), elementwise/copy 소형 커널 다발(1,761회, 합 406ms), 8M 소형 모델의 낮은 텐서 유틸.
- 최대 레버: **micro 256→1024 (듀얼)** → update 59 → 약 43-45s/랭크 기대 (단일 스윕 -28% 근거).

### 2.2 데이터 관문 (Python) = 전체의 약 13-15% + 겹침 가능분

| 항목 | 시간 | 위치 |
|---|---:|---|
| record (행당 Python 객체 조립) | 8.2s | 수집의 35% |
| 관측 Rust→Python crossing | 3.1s | 수집의 13% |
| parse + 미계측 | 3.7s | 수집 (일부 Python 추정) |
| 물질화 (to_batch) | 13.3s | 업데이트의 17% (겹침 가능) |

- 원칙: "더 빠른 Python"도 "Rust 재작성"도 아님 → **행당 객체 생성을 멈추는 columnar/SoA 전환**. Rust는 잔여 열에만 (엔진 라인 비용·픽스처 규율).

### 2.3 직렬 구조 = 수집 23% 동안 learner GPU 유휴

- frozen-policy 계약: 수집(현 정책 스냅샷) → 업데이트 직렬. 이 23% 회수가 구조적 최대 지렛대(상한 23%).
- 계약 내 접근: rolling slots(tail 제거), record/물질화 처리 겹침, (조건부) GPU 역할 분리.
- 계약 외 접근: async actor-learner(정책 지연 수용) = 학습 알고리즘 변경 → §5 계약 트랙.

### 2.4 엔진 (참고)

- Rust `step_batch` = 수집의 1.6%, 전체의 약 0.35%. **엔진은 병목이 아님** — 핫패스는 이미 네이티브.

## 3. 옵션 메뉴

### A그룹. 지금 가능 (계약 유지, 등가성 테스트 부착)

| 옵션 | 내용 | 예상 효과 | 근거/비고 |
|---|---|---|---|
| **A1. 듀얼 micro 1024** | 학습 micro 256→1024 | +25-30% (31-34 games/s) | 단일 스윕 -28% 실측; 듀얼 미측정(최우선 실험) |
| A2. 수집 record columnar화 | 행당 객체 조립 제거 | 수집 -25-30% | record 8.2s/23.2s |
| A3. 학습자 물질화 columnar화 | to_batch + 후보 u8 | update -8-11% | 13.3s/78.2s, 겹침 준비 |
| A4. pinned + async H2D | 전송과 계산 겹침 | -2-3s 전체 | 표준 기법 |
| A5. optimizer/digest 정리 | 자잘한 오버헤드 제거 | 런당 수초 | optimizer는 drain-time 실체 19ms/스텝 (한계 작음) |
| A6. rolling slots | tail 대기 제거 | 미측정 (측정 선행) | v3 플랜 §8.1 지표 |
| A7. CUDA Graph PoC | 커널 런치 묶음 | 소폭 (미측정) | learner는 고정 shape 가능, PyTorch API 제약 확인됨 |

### B그룹. 값싼 검증 (표현 변경)

- **B1. 관측 f16 wire**: -1~3% 예상. 등가성 범위 명시 + 짧은 회귀 + f32 폴백 유지 조건.

### C그룹. 계약 변경 (별도 승인 + A/B + 문서 버전)

- **C1. async 오버랩**: 이득 상한 **+25-30%**. 대가 = off-policy(정책 지연) → V-trace 등 학습 보정 필요, 재현성 기준 리셋, 복잡도 증가. 표준(IMPALA/Ape-X/Sample Factory)이지만 "학습 알고리즘 교체"에 해당. **평가 벤치 + 본학습 베이스라인 이후에 실험.**
- **C2. epochs 4→2**: update wall -50%, 단 같은 데이터로 덜 학습 → 순효과 불확실 (품질-시간 곡선 필요).
- C3. 레시피(minibatch/LR) 미세 조정: 기대 작음.

### D그룹. 불가/금지 (측정 또는 하드웨어로 확정)

- torch.compile/Triton: Triton 공식 지원 CC 8.0+, V100(sm70) 불가 (2026-08-28 Triton 3.8 README) + 프로젝트 기준선(Inductor 금지).
- bf16/TF32/FlashAttention-2: V100 하드웨어 미지원.
- micro 2048+: 단일 both-seat 실측 기각 (VRAM 2배: 9.25→17.98GiB, 속도 동일).
- fused Adam, 더 많은 워커: 측정 후 무의미.
- NVMe/NCCL: 측정상 병목 아님 (디스크 read 0, all-reduce 10ms/35MB).
- 트렁케이션, 후보 컷, 정보 누출, 운영오류 draw화, 모델 축소: 정확성·스펙 위반 금지.

## 4. 실행 순서 (STEP)

| 단계 | 내용 | 상태 |
|---|---|---|
| STEP 1 | A1 듀얼 micro 1024 A/B (D1 프로토콜 + 3회 + digest parity) | P0 배치로 실행 |
| STEP 2 | A2 + A3 columnar (recompute/streaming 등가성 게이트 재사용) | 대기 |
| STEP 3 | A4 + A5 + 전체 재측정 | 대기 |
| STEP 4 | A6 (측정→적용) + A7 PoC | 대기 |
| STEP 5 | 본학습 준비 (맥 병렬): readiness 잔여 (blocked 이동 29 / 특성 23 / 동적 클로저 11 / 코퍼스 전수화, 6기준) | 병렬 가능 |
| STEP 6 | 계약 트랙 (미래): 평가 벤치 + 본학습 베이스라인 이후 C1 실험 | 조건부 |

궤적 (듀얼, 추정): 현재 24.7 (약 46일) → S1-2 후 31-34 (34-37일) → S3 후 37-40 (26-28일) → S4 후 42-46 (22-25일). 이론 천장 110-115.

## 5. 계약 변경 가이드

- 계약 인벤토리 (PA3_TRAINING_CONFIG.yaml 발췌): frozen-policy 직렬 / 양좌석 수집 / 자연완결 1회 카운트 / 오버슈트 유지 / 운영오류를 draw로 바꾸지 않음 / PPO 4 epochs, global minibatch 4096, per-rank 2048, micro 256, γ=1, GAE λ=0.95, terminal reward / 관측 스키마 v1 (96토큰, no truncation) / PA3-8M 고정 / 스택 핀 (torch 2.14.0+cu126, eager+SDPA 기준선, Inductor 금지).
- 3계층 분류: ①성능 트랙(지금 가능) ②경계(표현 변경, 조건부) ③학습 의미 변경(별도 승인) ④금지.
- 변경 기준 5: 실측 이득 / 의미 변화 명시 / 등가성·회귀 검증 / 플래그+폴백 / 승인+문서 버전.
- 타이밍: readiness 10/16 + 평가 벤치 부재 + P0-P2 잔여 가치 → 계약 트랙은 본학습 베이스라인 이후.

## 6. 외부 검증 메모 (2026-10-09 확인)

- Triton 3.8.0 README: "NVIDIA GPUs (Compute Capability 8.0+)" → torch.compile 배제 확정.
- PyTorch CUDA graph API: beta, 고정 shape/순서, AMP cache 비활성 조건 (torch/cuda/graphs.py).
- Sample Factory(arXiv 2006.11751): 단일 머신 처리량 극대화 — 비동기·배치 추론·네이티브 핫패스.
- IMPALA(arXiv 1802.01561): actor-learner 분리 + importance weighting = 우리 계약이 금지하는 구조의 표준 원형.
- PufferLib(arXiv 2406.12905) 벡터화·네이티브 / EnvPool(arXiv 2206.10558) 환경 실행이 시스템 병목 (우리는 엔진으로 이미 해결).

## 7. 용어

- micro(미세배치): GPU에 한 번에 넣는 학습 덩어리. 클수록 효율 좋음.
- columnar(SoA): 행당 객체 대신 열 단위 표로 데이터를 다루는 방식.
- pinned + async: CPU 메모리 고정 + 비동기 복사로 전송과 계산을 겹침.
- rolling slot: 경기 종료 슬롯에 다음 경기를 즉시 투입 (tail 제거).
- off-policy: 낡은 정책 데이터로 학습하는 것 (속도 상승, 알고리즘 변경 필요).

## 부록: 근거 파일

- `docs/perf/V4_FULLSTACK_BOTTLENECK_ATLAS.md`, `V4_OPERATIONAL_FRONTIER_REPORT.md`, `V4_KERNEL_AND_MEMORY_FRONTIER.md`, `V3_VRAM_BATCH_SWEEP.md`, `V3_DDP_GLOBAL_GRADIENT.md`
- `docs/PokeAgent3_Correctness_First_Extreme_Optimization_Plan_v4.0_2026-10-09.md`, `_Extreme_Optimization_Master_Plan_v3.0_...`, `_Optimization_Master_Plan_v2.0_...`, `_Ultimate_Throughput_Optimization_Plan_v1.1.md`
- `docs/spec/fullspec-1.1-minidc-20261006/PA3_TRAINING_CONFIG.yaml`
