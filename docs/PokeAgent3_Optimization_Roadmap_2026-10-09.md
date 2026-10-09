# PokeAgent 3.0 최적화 로드맵 및 옵션 분석 (2026-10-09)

> 작성: 아리아 (2026-10-09). 기준: 통합 정본 `main` @ `b6da590`, v4 F0/F1 실측 + 플랜 v3/v4.
> 목적: 미니DC(2×V100)의 처리량 한계를 최대한 끌어내어, 최단 시간에 최대한 강한 모델 학습에 도달한다.
> 구성: 1) 목표와 상한, 2) 개선 지점, 3) 옵션 메뉴, 4) 실행 순서, 5) 계약 변경 가이드, 6) 외부 검증, 7) 용어.

> **2026-10-09 정정:** 원래 v4 계획의 근거는 역사적 기록으로 남기되, 완료된 v5 실험과 잘못된 상한 해석을 아래에 반영했다. v4 옵션의 예상 효과를 현재 미실행 작업으로 합산하지 않는다. 새 구현과 최종 A/B·채택 결과는 [V6 PPO 처리량 보고서](perf/V6_PPO_THROUGHPUT_REPORT.md)를 정본으로 사용한다. 이 문서는 v6 최종 성능을 예측 또는 확정하지 않는다.

## 1. 목표와 측정 범위

- 목표: 강한 체크포인트에 도달하는 시간과 경기 예산을 줄인다. 1억 자연완결 경기는 원래 본학습 계약이며, 훈련 방식 연구의 타깃 변경 제안과 구분한다.
- **과거 성능 기준선**: v4 F1 듀얼 24.67 games/s → v5 듀얼 micro 1024 **32.50 games/s**. v5 같은 세션 micro 256→1024는 26.25→32.50(+23.8%)다. v4→v5의 +31.7%에는 시점/측정 조건 차이가 포함된다.
- **v5 전체 분모 정정**: `runs/perf/v5/confirm_default1024.json`의 2,048경기/런처 **63.0127s**가 32.5014 games/s다. rank 0 수집 12.3668s, 업데이트 42.3294s, 별도 GAE·정규화 준비 1.2788s, 재계산 검증 0.1332s, 저장 0.4404s이며, 나머지 약 6.5s는 런처·초기화·digest·종료 등을 추가 계측해야 한다. 수집+업데이트 약 54.7s와 전체 속도를 혼용하지 않는다.
- **기존 110-115 games/s 물리 천장 주장은 철회**한다. 두 V100의 일반 FP16 합계 56.5 TFLOPS를 Tensor Core 경로까지 포함한 상한으로 쓰면 안 된다. V100 PCIe의 명목 Tensor 성능은 카드당 112 TFLOPS이며, 실제 커널/메모리 이용률과 현재 175W/150W 전력 제한에서의 동작은 별도로 측정해야 한다. 이 명목값을 경기당 추정 FLOPs로 나눈 값도 실용 상한이 아니다. [기존 검토에 사용한 NVIDIA 자료](https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf)
- "500-1,000 games/s"를 뒷받침하는 측정은 없다. 다만 잘못된 FP16 산술로 엄밀한 불가능성을 증명할 수도 없다. 원래 40-55 및 42-46 games/s 예상과 완료일 전망은 **v4 당시 실험 가설**이며 현재 결과나 보장치로 사용하지 않는다.

## 2. 개선 지점 (실측 시간 분해)

아래는 **v4 단일 GPU**, 2,048경기, all-in 약 101초 기준 F0 아틀라스다. 최신 듀얼 시간 분해와 합산하거나 그대로 현재 비중으로 적용하지 않는다.

| 구간 | 시간 | 비중 |
|---|---:|---:|
| 수집 (경기 생성) | 23.2s | 23.0% |
| PPO 업데이트 | 78.2s | 77.4% |
| 체크포인트 | 0.5s | 0.5% |

### 2.1 학습(PPO) 효율: 여전히 최우선, 과거 포화 해석은 정정

- GPU-busy **95.4%는 v4 단일 GPU의 짧은 트레이스**다. 커널 실행 중인 시간 비율이며, 최신 듀얼의 Tensor 연산·메모리 대역폭 포화율이 아니다. 따라서 CPU 개선 여지를 5%로 제한하거나 하드웨어 천장 도달의 근거로 사용하지 않는다.
- 경기당 추정 FLOPs와 일반 FP16 명목값을 나눈 기존 "피크의 약 1/3" 주장도 병목 판정 근거에서 제외한다. 커널별 실행 시간, 메모리 대역폭, 실제 클럭과 동기화를 새 프로파일로 확인한다.
- 과거 elementwise/copy 소형 커널 다발과 반복 물질화는 조사 후보를 제공한다. 현재는 micro 1024가 이미 기본이며, immutable rollout 재사용·padding 계산 제거·불필요한 동기화 감소 등 의미 보존 변경을 별도 A/B로 검증한다.
- v5 확인 런에서는 업데이트 약 **42.33s가 전체 63.01s의 약 67%**다. 학습자가 가장 큰 절감 대상이라는 결론은 이 최신 측정으로 지지된다.

### 2.2 데이터 관문: 아래 비중은 v4의 과거 측정

| 항목 | 시간 | 위치 |
|---|---:|---|
| record (행당 Python 객체 조립) | 8.2s | 수집의 35% |
| 관측 Rust→Python crossing | 3.1s | 수집의 13% |
| parse + 미계측 | 3.7s | 수집 (일부 Python 추정) |
| 물질화 (to_batch) | 13.3s | 업데이트의 17% (겹침 가능) |

- **v5 실험으로 갱신**: columnar 관측은 물질화 7.60→6.98s(-8.2%), 업데이트 42.22→41.38s(-2.0%)였지만 수집이 12.44→12.69s로 퇴행했다. 전체 +1.6%는 선언한 ±2% 노이즈 내여서 기본값 미승격이다. [기록 경로](perf/V5_COLUMNAR_RECORD.md), [물질화](perf/V5_COLUMNAR_MATERIALIZATION.md)
- 따라서 표의 비용 비중을 그대로 예상 개선율로 바꾸지 않는다. 매 epoch 재물질화 대신 반복 재사용하는 설계는 columnar append 실험과 다른 후보이며, 준비 비용·최대 메모리까지 포함해 측정해야 한다. 후보 u8은 당시 미시도이며, 빈 candidate padding 제거와 legal candidate 축소도 구별한다.

### 2.3 직렬 구조: 오버랩 이득과 자원 경합을 구별

- frozen-policy 계약: 수집(현 정책 스냅샷) → 업데이트 직렬. 수집 중에도 같은 GPU가 **이미 그룹별 배치 정책 추론**을 수행하므로 GPU 자체가 전부 유휴라는 설명은 잘못이다.
- v5 수집+업데이트 구간을 완전히 겹친다는 약 1.29× 계산은 경합 없는 이상치다. 업데이트 42.33s를 그대로 두면 모든 수집·기타 비용을 숨겨도 **2,048/42.33 ≈ 48.4 games/s 이하**다. 60에 도달하려면 업데이트도 단축해야 한다.
- actor와 learner가 같은 두 GPU에서 동시에 실행하면 연산·메모리 대역폭을 놓고 경합한다. GPU를 한 장씩 나누면 듀얼 learner 능력을 잃는다. 실용 이득은 별도 파일럿에서 확인한다.
- rolling slots는 **목표 경기 수가 슬롯 수보다 큰 조건**으로 시험한다. v5 듀얼 games=envs=1024는 정상 진행 중 실제 refill을 실행하지 않아 동률 결과가 일반적 기각 근거가 아니다. record/물질화 겹침도 전체 반복에서 측정한다.
- 계약 외 접근: async actor-learner(정책 지연 수용) = 학습 알고리즘 변경 → §5 계약 트랙.

### 2.4 엔진 (참고)

- v4 Rust `step_batch` = 수집의 1.6%, 전체의 약 0.35%. 당시 엔진 step은 우선 병목이 아니었다. 이 과거 비중과 정책 없는 엔진 단독 속도를 최신 학습 전체의 상한으로 쓰지 않는다.

## 3. 옵션 메뉴

### A그룹. 의미 보존 옵션과 v5 이후 상태

| 옵션 | 내용 | 상태 | 근거/비고 |
|---|---|---|---|
| **A1. 듀얼 micro 1024** | 학습 micro 256→1024 | **v5 완료·기본 승격** | 같은 세션 +23.8%, 약 32.50 games/s; [A/B](perf/V5_MICROBATCH_AB.md) |
| A2. 수집 record columnar화 | 행당 객체 조립 제거 | **v5 미승격**, opt-in | 수집 +2.0% 퇴행, 전체 +1.6%는 노이즈 내 |
| A3. 학습자 물질화/재사용 | columnar 및 immutable rollout cache | columnar는 미승격, 재사용은 v6 별도 검증 | 후보 u8 당시 미시도. 캐시 준비 비용·메모리 포함 |
| A4. H2D 정리 | 반복 전송·복사/계산 겹침 | 측정 후 판단 | v5 H2D 약 1.7s; 과거 예상 -2~3s를 보장하지 않음 |
| A5. optimizer/digest/동기화 | 불필요한 호스트 왕복·런처 반복 비용 정리 | 최신 프로파일 필요 | CPU optimizer 타이머에 앞선 GPU queue drain이 포함될 수 있음 |
| A6. rolling slots | tail 대기 제거 | per-slot API 완료, **유효한 듀얼 refill A/B 필요** | games>envs, overshoot·rows/game·update 비용 함께 기록 |
| A7. CUDA Graph PoC | 커널 런치 묶음 | 미측정 | 고정 shape 외에도 CPU/GPU 동기화·제어 흐름 제약 해결 필요 |

### B그룹. 값싼 검증 (표현 변경)

- **B1. 관측 f16 wire**: **v5 no-go**. 당시 낙관 모델 약 2.7%, 현실 1.0-1.4%로 선언한 ±2% 노이즈와 스키마 변경 비용을 넘지 못했다. 모든 전송/캐시 개선을 기각한 결과는 아니다. [실험 범위와 판단](perf/V5_F16_WIRE_POC.md)

### C그룹. 계약 변경 (별도 승인 + A/B + 문서 버전)

- **C1. async 오버랩**: 미측정. 구간 산술 약 +29%는 자원 경합 없는 이상치이며 실용 상한이나 보장치가 아니다. 정책 지연을 허용하면 off-policy 보정·재현성·평가 기준도 함께 바뀐다. **평가 벤치와 지속 학습 베이스라인 이후 별도 파일럿.**
- **C2. epochs 4→2**: update wall -50%, 단 같은 데이터로 덜 학습 → 순효과 불확실 (품질-시간 곡선 필요).
- C3. 레시피(minibatch/LR) 미세 조정: 기대 작음.

### D그룹. 기존 제외 경로와 정확성 제약

- torch.compile/Triton: Triton 공식 지원 CC 8.0+, V100(sm70) 불가 (2026-08-28 Triton 3.8 README) + 프로젝트 기준선(Inductor 금지).
- bf16/TF32/FlashAttention-2: V100 하드웨어 미지원.
- micro 2048+: **기존 단일 GPU 구현**은 VRAM 9.25→17.98GiB, 속도 동일로 미승격. 다른 계산·캐시 경로의 듀얼 결과까지 일반화하지 않으며 재측정한다면 별도 실험이다.
- fused Adam, 더 많은 워커: 기존 실험에서는 승격 근거 부족. 최신 프로파일 없이 절대적인 무효 경로로 단정하지 않는다.
- NVMe/NCCL: 기존 측정에서 우선 병목 아님 (디스크 read 0, all-reduce 약 10ms/35MB).
- 경기 truncation, **legal/stored 후보 삭제**, 정보 누출, 운영오류 draw화, 모델 축소: 정확성·스펙 위반 금지. 모든 실제 후보를 유지하는 **빈 padding 제거**는 후보 컷과 다르며 등가성/A-B 검증 대상이다.

## 4. 실행 순서 (STEP)

| 단계 | 내용 | 상태 |
|---|---|---|
| STEP 1 | A1 듀얼 micro 1024 A/B (D1 프로토콜 + 3회 + digest parity) | **v5 완료·승격** |
| STEP 2 | A2 + A3 columnar | **v5 완료·미승격**, 새 재사용 설계와 구별 |
| STEP 3 | 지속 반복 계측 + learner 재사용·padding/동기화 개선 A/B | **v6 보고서로 갱신**, 준비·전체 런처 비용 포함 |
| STEP 4 | games>envs rolling refill A/B + 조건 충족 시 CUDA Graph PoC | 유효한 refill 실험 필요 / Graph 미검증 |
| STEP 5 | readiness와 평가 스파인 | 현재 상태는 `docs/PROJECT_STATUS.md`를 따름; 과거 잔여 개수를 현재로 재사용하지 않음 |
| STEP 6 | 평가 곡선 기반 epochs 4→2→1 및 오버랩 연구 | 학습 의미 변경 별도 결정 |

원래 v4의 24.7→31-34→37-40→42-46 궤적은 사전 추정이다. 완료된 실험의 효과를 중복 합산하거나 천장 110-115와 함께 현재 계획으로 재사용하지 않는다. 실제 채택은 동일 workload의 전체 실행 시간, 반복 가능한 개선, 정확성 게이트와 메모리 한도로 판단한다.

## 5. 계약 변경 가이드

- 원래 계약 인벤토리 (PA3_TRAINING_CONFIG.yaml): frozen-policy 직렬 / 양좌석 수집 / 자연완결 1회 카운트 / 오버슈트 유지 / 운영오류를 draw로 바꾸지 않음 / PPO 4 epochs, global minibatch 4096, per-rank 2048, 원본 micro 256, γ=1, GAE λ=0.95, terminal reward / 관측 스키마 v1 (96토큰, no truncation) / PA3-8M 고정 / 스택 핀 (torch 2.14.0+cu126, eager+SDPA 기준선, Inductor 금지). **런처 micro는 v5에서 1024로 승격됐으며 effective batch·목적함수는 유지됐다.** 원본 config와 운영 기본값을 구별한다.
- 3계층 분류: ①성능 트랙(지금 가능) ②경계(표현 변경, 조건부) ③학습 의미 변경(별도 승인) ④금지.
- 변경 기준 5: 실측 이득 / 의미 변화 명시 / 등가성·회귀 검증 / 플래그+폴백 / 승인+문서 버전.
- 타이밍: v4/v5 당시 readiness 10/16과 평가 벤치 부재가 기록됐다. 최신 상태는 현황판으로 확인하고, 학습 의미 변경의 판단은 평가 곡선과 의미 보존 최적화 결과를 바탕으로 한다.

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
- [V5 micro A/B](perf/V5_MICROBATCH_AB.md), [V5 columnar](perf/V5_COLUMNAR_MATERIALIZATION.md), [V5 f16 wire](perf/V5_F16_WIRE_POC.md), [V5 rolling API](perf/V5_ROLLING_SLOT_ENGINE_FIX.md)
- [V6 PPO 처리량 보고서](perf/V6_PPO_THROUGHPUT_REPORT.md) — 후속 코드·최종 측정/채택 결과의 정본.
