# 안정 냉각·175W/175W 재측정 및 전 구간 병목 보고서

측정일: **2026-10-11, Asia/Seoul**. 실제 코드: `6eb48c02f51404ded0ba7b89fa7edd5b17d1d4a6`.

## 1. 핵심 결론

현재 검증된 지속 실행 처리량은 **43.477 games/sec**다. 이는 3개의 독립 프로세스 세트에서 첫 반복을 제외한 총 12,288경기 / 282.633초다. 반복당 평균 **47.105초**, 재시작 비용을 포함한 별도 3회 중앙값은 **36.563 games/sec**다. 둘은 분모가 다르며 섞어 비교하지 않는다.

두 V100 전력 제한은 모두 **175W**이며 사용자가 냉각 개선을 확인했다. 기본 측정 부하 중 최고 온도는 **57°C / 53°C**, 1초 샘플에서 thermal slowdown은 0회였다. 반면 PPO 중 power-cap 사유는 관찰된다. 현재 관측에서 냉각은 주된 병목이 아니다.

PPO 약 **29초**, 수집의 느린 rank 약 **12.53초**, 준비 약 **4.47초**가 큰 비용이다. 특히 수집의 행 저장 **4.46초**가 native battle step **0.19초**보다 크다. GPU kernel은 PPO에서 바쁘지만 이는 Tensor Core/SM을 최대로 사용한다는 뜻이 아니다.

**후속 엔진 수정 (2026-10-11):** 관측 검증기가 `EventKind::Item`의 item ID 0만 허용하도록 수정했다. 선택 대상의 아이템 gate가 통과한 뒤 아이템 없는 대상으로 redirect되면, pinned Showdown의 `onTryHit`도 빈 item name을 공개한다. 기존 emitter와 `Known::new(0)` 지식 처리는 이 의미에 맞으므로 유지했다. EndItem(0)과 범위 밖 ID는 계속 거부한다. `poltergeist`, `observation`, `batch_observation` 테스트 14개 및 재빌드 Python 확장의 두 snapshot × 양측 관측 4개 검증 통과. Showdown callback 직접 실행으로 gate/빈 activation도 확인했다. 전체 엔진 테스트는 corpus 검사 도중 중단했으며 전체 통과 또는 장기 학습 검증을 주장하지 않는다. 아래 오류 기록은 수정 전 측정 이력이다.

**안정성 결함 발견:** 계획된 기본 성능 패널 12회는 모두 성공했으나, 별도 진단의 두 번째 수집에서 `observation catalogue ID`가 발생했다. 동일 checkpoint 수집과 GPU 없는 snapshot 복원에서 재현했다. Poltergeist 이후 ID 0인 Item 이벤트를 관측 검증이 거부한다. 따라서 아래 속도는 성공한 벤치마크 계약의 속도이며, 무중단 장기 학습 가능성을 입증하지 않는다. §10 참조.

## 2. 실험 설계와 측정 경계

- PA3-8M **8,758,963 parameters**, 양좌석 current-policy self-play, 실제 Rust 엔진·고정 1,137팀 풀, 자연완결만 경기로 센다.
- rank당 1,024경기 / 1,024 env / Rust workers 16, 총 2,048경기/반복. PPO 4 epochs, global minibatch 4,096, local minibatch 2,048, microbatch 1,024.
- CUDA rollout cache·compact candidate padding 활성, observation token trim 비활성. FP16 autocast + GradScaler, 동일 기본 seed 20261009, persistent iteration별 독립 rollout seed와 누적 LR clock.
- `cold1 → persistent1 → cold2 → persistent2 → cold3 → persistent3`로 교차 실행했다. 각 persistent는 3반복이며 2·3반복만 steady로 집계한다. 세트마다 모델/optimizer를 새로 시작한다. 정책이 업데이트되므로 steady의 경기당 행 수가 완전히 같지는 않다.
- 기본 패널: **12반복 / 24,576경기**, source HEAD 한 개, 시작 시 모두 clean tree. 별도 profiling과 NCCL microbenchmark는 속도 통계에 포함하지 않았다. 엔진/모델/학습 로직 변경이나 드라이버·전력·서비스 조작은 없다.
- 단계 wall은 시작·끝의 CUDA synchronize를 포함한다. rank별 단계는 직렬 합산 가능하지만 **서로 다른 rank의 단계 최댓값 합은 반복 wall이 아니다**. 부모는 느린 rank의 실제 반복 wall을 사용한다.
- launcher all-in은 부모가 workers를 spawn하기 전부터 결과 JSON 읽기까지다. 부모 자신의 최초 torch import는 그 타이머 바깥이다. 부모의 종료 확인이 1초 polling이므로 짧은 실행에 최대 약 1초의 관측 지연이 있다. 외부 모니터 case wall은 snapshot·polling까지 포함해 별도 기록했다.
- GPU/host telemetry는 1초 간격이다. rank report 수정 시각과 worker/startup/iteration durations로 단계 창을 재구성했다. 파일 쓰기·경계 샘플 오차가 있으므로 단계별 telemetry는 근사치다. 전력은 NVML 순간 보고값이고, memory utilization은 HBM GB/s가 아니다.
- 각 반복에서 recompute gate, 두 rank 모델·Adam digest, checkpoint 검증을 유지했다. profiler를 뺀 측정에도 telemetry subprocess의 작은 부하가 있으며 그 비용의 별도 A/B는 하지 않았다.

## 3. 현재 속도와 과거 비교

| 실행 | launcher all-in games/s | steady games/s | 전체 경기 |
|---|---|---|---|
| cold1 | 35.921 | — | 2048 |
| cold2 | 36.563 | — | 2048 |
| cold3 | 37.227 | — | 2048 |
| persistent1 | 40.952 | 43.656 | 6144 |
| persistent2 | 41.226 | 43.471 | 6144 |
| persistent3 | 41.227 | 43.305 | 6144 |

재시작 3회 중앙값 **36.563**(35.921–37.227, 표준편차 0.653). 독립 persistent 세트의 steady 중앙값 **43.471**(43.305–43.656, 표준편차 0.175). steady 합산 **43.477**은 평균 rate를 단순 평균한 것이 아니라 총경기/총시간이다. 총 **323,373행**, 평균 **26.316행/경기**이며 steady rollout 행 생성률은 약 **1,144행/s**다.

2026-10-09 clean v6 재시작 중앙값 37.228 대비 이번 36.563은 **−1.79%**, 당시 반복 범위와 이번 polling 분해능 안의 차이다. 당시 업데이트 29.988초와 이번 steady 느린 rank 28.997초도 같은 warm/cold 조건이 아니므로 냉각 개선율로 표현하지 않는다. 과거 수정 중 v6 persistent 41.739 대비 이번 43.477은 +4.17%지만 dirty source·다른 정책 경로·175/150→175/175 변경이 있어 인과 실험이 아니다.

v4 dual DDP 24.67 / manual 25.60, v5 32.50은 이전 코드와 microbatch 조건의 역사값이다. 현재 하드웨어에서 그 코드를 다시 A/B하지 않았다. **냉각만으로 얼마 빨라졌는지는 이번 결과에서 분리할 수 없다.**

## 4. 반복 전체 시간: 어느 단계에서 무엇을 기다리는가

| 단계 | rank0 평균 초 | rank1 평균 초 | rank0 반복 내 비중 |
|---|---|---|---|
| 반복 설정/시작 barrier | 0.003 | 0.003 | 0.01% |
| 자연경기 수집 | 12.533 | 11.346 | 26.61% |
| 행 logprob 재검증 | 0.104 | 0.094 | 0.22% |
| global match clock all-reduce | 0.000 | 1.197 | 0.00% |
| GAE·정규화·rollout cache | 4.468 | 4.234 | 9.49% |
| PPO update | 28.764 | 28.997 | 61.06% |
| 모델·Adam digest/교환 | 0.542 | 0.543 | 1.15% |
| 검증 checkpoint 저장 | 0.472 | 0.000 | 1.00% |
| rollout/cache 정리 | 0.218 | 0.194 | 0.46% |
| 반복 종료 barrier | 0.000 | 0.497 | 0.00% |
| 미계측 잔여 | 0.001 | 0.000 | 0.00% |

모든 값은 steady 6반복의 rank별 평균이다. rank0에서 약 **61.1% PPO / 26.6% 수집 / 9.5% 준비**이며 나머지는 검증·저장·정리다. rank1은 수집이 약 1.19초 빠른 대신 global clock all-reduce에서 약 1.20초 기다린다. 이것은 scalar를 전송하는 데 1.2초가 걸린다는 뜻이 아니다. rank0 checkpoint 약 0.47초 동안 rank1은 종료 barrier 약 0.50초로 기다린다. 두 비용을 더하면 중복이다.

cold worker startup은 rank별 약 **3.00–4.02초**로 엔진·모델·CUDA·DDP 초기화를 포함한다. 부모 launcher wall에서 느린 worker wall을 뺀 나머지는 약 **3.24–3.92초**이며 worker의 초기 import, 보고/종료 및 polling 지연이 섞인다. 이 잔여를 모두 NCCL이나 체크포인트 비용으로 부르면 안 된다.

## 5. 수집 내부: 배틀 엔진보다 행 저장 비용이 크다

| 수집 항목 | rank0 초 | rank1 초 | rank0 수집 내 비중 |
|---|---|---|---|
| rollout 행·관측 복사/저장 | 4.457 | 3.939 | 35.56% |
| Rust 관측 인코딩/배치 반환 | 1.867 | 1.739 | 14.89% |
| 후보 tensor table 구성 | 1.353 | 1.359 | 10.79% |
| 관측 이동/선택/전송의 host wall | 1.178 | 1.092 | 9.40% |
| 정책 연산 제출의 host wall | 1.118 | 0.944 | 8.92% |
| 관측 파싱 | 0.695 | 0.630 | 5.55% |
| native battle step | 0.188 | 0.179 | 1.50% |
| 후보 조회 | 0.141 | 0.116 | 1.12% |
| request 종류 조회 | 0.046 | 0.015 | 0.36% |
| 샘플·value CPU readback | 0.046 | 0.039 | 0.37% |
| 배틀 초기화 | 0.021 | 0.020 | 0.17% |
| 행동 묶음 조립 | 0.005 | 0.004 | 0.04% |
| legacy spec 구간 | 0.000 | 0.000 | 0.00% |
| 반복·메타데이터 등 잔여 | 1.418 | 1.271 | 11.32% |

`NativeCollector._collect_round`는 각 요청의 branch 후보를 prefix에 따라 순차 조회하고, 샘플링한 뒤 Python 루프로 `record_packed`를 호출한다. 현재 기본 store는 행마다 관측 배열의 `.copy()`와 후보 tuple/metadata를 만든다. 이 경로는 저장만 rank0 수집의 약 **35.6%**다. Rust workers를 늘려도 이 Python 저장 루프가 자동으로 병렬화되지는 않는다.

Rust 관측 생성은 약 **14.9%**, 후보 table 구성은 약 **10.8%**다. battle step 자체는 약 **1.5%**라서 step을 두 배 빠르게 해도 전체 반복 이득은 약 0.2% 수준이다. step 시간만 가지고 전체 엔진의 관측·후보 비용까지 작다고 해석해서도 안 된다.

수집 중 process CPU time/wall은 rank0 **1.291 cores**, rank1 **1.321 cores** 수준이다(프로세스 전체 thread CPU 합). 전체 머신 40 physical cores가 바쁘다는 뜻이 아니다. 기본 샘플에서 수집 GPU utilization 평균은 약 **14.1% / 21.9%**이며 CPU와 GPU를 번갈아 쓰는 구조가 보인다.

기존 collector 타이머는 host에서 관찰한 시간이다. 모델 시간은 주로 CUDA 연산 제출이며, 실제 GPU 작업이 후속 동기화/H2D/readback 구간에 걸쳐 끝날 수 있다. 따라서 위 `model_seconds≈1초`를 정책의 전체 GPU 연산 시간이 1초라는 뜻으로 해석하지 않는다.

별도 두 early collection-round trace에서는 kernel union **218.7/222.2ms**, kernel span **1,040.1/951.9ms**, busy **21.0%/23.3%**였다. 프로파일링 오버헤드가 섞인 작은 표본이지만 기본 telemetry의 낮은 수집 utilization과 방향이 일치한다. 이 표본 GPU kernel 시간 중 encoder가 약 94%로 가장 크다.

tail telemetry는 별도 profiling 실행에서만 켰다. 50%→100% 경기 완결 tail은 **2.49/2.95초**, idle slot-seconds 비율은 **17.1%/20.9%**였다. profiling으로 전체 수집 wall이 15.70/15.02초까지 늘었으므로 이를 기본 실행의 tail 시간으로 대입하면 안 된다. 현재 games=envs인 한 cohort는 중간 refill로 추가 경기를 넣을 여지가 없으며 rolling-slot 효과는 games>envs 조건에서 따로 검증해야 한다.

## 6. 학습 데이터 준비: 매 반복 약 4.5초

| 준비 항목 | rank0 초 | rank1 초 |
|---|---|---|
| GAE | 1.481 | 1.227 |
| global advantage 정규화 | 0.016 | 0.270 |
| 한 번의 확장/후보 trim/GPU cache | 2.969 | 2.735 |

캐시는 모든 steady 반복·두 rank에서 활성화됐다. 추정 cache 크기는 약 **1.94–1.96GiB/rank**, admission budget은 8GiB, 저장된 후보 padding은 64에서 32로 줄었다. GPU memory soft budget 28GiB와 activation 여유를 유지한다. 준비 내부 cache_build는 host 확장·검증·전송을 함께 포함하며, 이번 기본 계측으로 이 셋의 순수 시간을 각각 분리할 수는 없다.

GAE는 trajectory별 Python 행 처리와 역방향 계산 경로를 지나고, 정규화는 두 rank 통계 collective를 포함한다. rank1 normalize가 약 0.27초로 rank0 0.016초보다 긴 것은 GAE가 먼저 끝나 peer를 기다리는 영향이 포함된다. CPU cache를 GPU cache로 다시 바꾸는 최적화는 이미 적용돼 있다. PPO 중 반복 H2D 비용은 작다.

## 7. PPO: host 타이머와 GPU 연산 시간을 구분

| 기존 host 타이머 | rank0 초/반복 | rank1 초/반복 |
|---|---|---|
| cached minibatch 선택 | 0.087 | 0.069 |
| device 이동 + global counts | 0.032 | 0.103 |
| forward host wall | 3.792 | 3.717 |
| backward host wall | 15.404 | 14.904 |
| unscale·norm·shared finite·Adam·scaler host wall | 9.335 | 9.877 |
| 통계 제출 | 0.032 | 0.030 |

host optimizer 9.34–9.88초를 **Adam 자체 비용으로 보면 틀린다**. 이 구간은 앞서 queue에 넣은 backward를 기다리고, unscale/clip norm, rank 공통 finite 검사, scalar readback, optimizer/scaler를 처리한다. 실제 Adam.step만 annotation한 별도 CUDA event 값은 훨씬 작다.

두 GPU에서 동일한 **2개 local minibatch / 4개 microbatch(각 1,024행)**를 profiler로 캡처했다. 각 단계 사이 CUDA event를 기록했으며 표본 내 current-stream elapsed time은 아래와 같다. nested encoder는 forward의 일부라 두 행을 더하면 안 된다.

| CUDA event 구간(표본 합계) | GPU0 ms | GPU1 ms |
|---|---|---|
| 전체 forward | 287.60 | 279.23 |
| 그중 encoder (nested) | 246.31 | 239.60 |
| backward + 관련 dependency | 747.69 | 754.82 |
| Adam.step만 | 11.28 | 10.64 |
| cached select | 5.14 | 6.21 |
| 명시적 counts all-reduce | 1.68 | 0.53 |
| 명시적 finite flag | 0.04 | 0.72 |

backward는 약 **748–755ms**, forward **279–288ms**, Adam.step은 **10.6–11.3ms**다. event 구간은 GPU dependency와 해당 stream의 대기도 포함하며 순수 kernel duration과 동일하지 않다. 그래도 계산을 바꾸지 않은 Adam 튜닝만으로 전체 9초를 없앨 수 있다는 해석은 명확히 배제한다.

기본 steady 각 rank는 **4 epochs / 56 optimizer steps / 112 micro-forwards**, global minibatch 경계마다 동기화한다. 전체 valid 학습 행은 323,373×4이며, padded 실행 행은 6반복×2rank×56step×2,048 = **1,376,256행**이다. 약 **6.01%**가 padding이고 loss는 유효 행만 반영한다. empty/tail 제어를 줄이는 변경은 분산 objective와 collective 순서를 보존해야 한다.

## 8. GPU kernel 구성과 NCCL 겹침

| kernel 종류(2 minibatch 표본) | GPU0 합 ms | GPU1 합 ms | GPU0 호출 수 |
|---|---|---|---|
| GEMM | 309.90 | 304.70 | 588 |
| attention | 110.47 | 108.16 | 48 |
| gather_scatter_index | 137.35 | 133.38 | 396 |
| copy_cast | 136.64 | 135.20 | 2348 |
| elementwise | 149.15 | 148.93 | 2754 |
| reduce_norm_softmax | 100.16 | 97.26 | 854 |
| NCCL | 37.18 | 98.33 | 12 |
| other | 56.63 | 55.08 | 1512 |

| 동시 실행을 고려한 실제 구간 | GPU0 ms | GPU1 ms |
|---|---|---|
| 첫 kernel→마지막 kernel | 1059.16 | 1058.20 |
| 전체 kernel interval union | 1004.38 | 1007.19 |
| NCCL union | 37.18 | 98.33 |
| NCCL·compute 겹침 | 33.10 | 73.83 |
| compute와 겹치지 않은 NCCL | 4.08 | 24.50 |

GPU busy는 표본 kernel span 내 **94.8% / 95.2%**다. SM occupancy·Tensor Core 사용률·HBM bandwidth를 측정한 값이 아니다. 커널 종류별 합계는 stream overlap이 있어 union과 같지 않으며, 큰 GEMM/attention 외에도 gather/scatter·copy/cast·elementwise·reduce가 상당하다. 따라서 CUDA core/Tensor FLOPS 명목값으로 games/sec 천장을 계산할 수 없다.

NCCL과 겹치지 않은 kernel 실행은 표본에서 **4.1/24.5ms**, span의 약 **0.4%/2.3%**다. NCCL kernel에는 peer 대기와 transport 진행이 함께 들어 있으므로 순수 네트워크 비용도 아니다. 이 표본에서 GPU 간 전송만 제거해 큰 배수의 이득을 낼 근거는 없다.

별도 SUM microbenchmark는 동일 host-staged NCCL 조건으로 warmup 5회 + 측정 20회였다.

| payload | GPU0 평균 ms | GPU1 평균 ms | GPU0 payload GB/s |
|---|---|---|---|
| one_float (4B) | 0.185 | 0.159 | 0.000 |
| two_floats (8B) | 0.185 | 0.159 | 0.000 |
| model_gradient (35035852B) | 16.937 | 16.912 | 2.069 |

35,035,852B는 모델 전체 FP32 gradient 크기다. 약 **16.9ms / 2.07 payload GB/s**는 반복된 큰 all-reduce의 지표이며 물리 링크 단방향 대역폭이나 실제 DDP step의 노출 지연과 같지 않다. scalar는 약 **0.16–0.19ms**로, §4의 rank1 match-clock 1.20초가 주로 peer wait임을 뒷받침한다.

프로파일러 실행 update는 약 **39.4초**, 전체 all-in은 **28.84 games/s**로 느려졌다. trace·CUDA event·shape 기록·export·NCCL microbenchmark를 포함한 **진단용 오버헤드 결과이며 현재 실속도로 사용하지 않는다**. GPU launch-owner attribution은 autograd thread를 완전히 연결하지 못해 많은 kernel이 unattributed다. forward/backward 시간은 이 불완전한 분류 대신 명시한 CUDA event를 사용했다.

## 9. 냉각·전력·메모리·CPU 배치

| 기본 측정 항목 | GPU0/rank0 | GPU1/rank1 |
|---|---|---|
| power limit | 175W | 175W |
| 전체 기본 패널 최고 온도 | 57°C | 53°C |
| steady PPO 평균 온도 | 53.88°C | 49.64°C |
| steady PPO 평균 SM clock | 1243.4MHz | 1281.1MHz |
| steady PPO 평균 GPU utilization | 98.55% | 98.11% |
| steady PPO 평균 memory-controller utilization | 58.88% | 56.01% |
| steady PPO power-cap Active 샘플 비율 | 34.1% | 21.8% |
| thermal slowdown Active 샘플 | 0 | 0 |
| steady peak reserved VRAM | 10.100GiB | 10.744GiB |
| worker lifetime peak RSS | 5.401GiB | 5.428GiB |

SM clock은 온도만으로 결정되지 않는다. 이번에는 열 제한은 없지만 power-cap 사유가 존재하고 카드별 clock도 다르다. 순간 `power.draw`는 175W를 넘는 샘플도 있었으나 모든 설정 limit은 175W로 유지됐다. 순간 전력 값만으로 cap이 바뀌었다고 추정하지 않는다. 전력 제한 상향은 수행하지 않았다.

host available RAM 최저 **47.95GiB**, swap-in/out 증가 **0페이지**였다. 시스템 전체 major fault 증가 5,347은 background/import/file I/O도 포함하므로 이 workload의 RAM 부족으로 귀속할 수 없다. 두 worker RSS를 단순 합산하면 shared library page를 중복 계산한다. 현재 표본에서 RAM/VRAM 부족이 속도 병목이라는 근거는 없다. 디스크 여유는 약 9.7GiB였고 checkpoint는 약 100MiB를 같은 파일에 교체했다. 저장 시간에는 torch serialization·GPU→CPU·file fsync·atomic replace가 포함되며 directory fsync는 별도로 하지 않는다.

CPU는 Xeon E5-2673 v4 ×2, 40 physical cores/80 threads, NUMA 2개다. 실제 worker affinity는 **rank0 0–19,40–59 / rank1 20–39,60–79**, numactl node binding과 node별 memory policy다. launcher의 `RANKS.cpus=0-15/20-35` 문자열은 실제 core pin에 사용되지 않는다. 따라서 16개 특정 코어에 고정됐다고 기록하지 않는다. Rust workers 16과 OMP/MKL 1을 사용하며 NCCL 등 추가 thread가 있어 process thread 수는 26–27개였다.

GPU 간 topology는 SYS이며 이번 NCCL은 P2P 비활성·shared memory 활성·loopback interface 설정이다. 장치 배치·현재 소프트웨어·전력 기록 정본은 [HARDWARE_MINIDC.md](../HARDWARE_MINIDC.md)다.

## 10. 이번에 발견한 재현 가능한 엔진 오류

첫 진단 실행의 iteration 2, rank1 수집에서 `ValueError: observation catalogue ID`가 발생했다. 실패한 실행은 중단됐으며 자연경기 성공으로 세거나 속도 평균에 섞지 않았다. 프로파일링된 첫 update checkpoint로 **profiler 없이** 동일 rank1 수집(seed 20261012)을 실행해 재현했다.

문제 slot 942 / generation 2 / turn 5의 snapshot을 저장했다. 별도 CPU-only 엔진에 복원한 뒤 **양쪽 side 모두** `observe_fixed_batch`가 같은 오류를 낸다. 이 최소 재현에는 학습·GPU·profiler·100MB checkpoint가 필요 없다.

확인한 원인은 public event의 `kind=Item, effect_kind=Item, effect=0`이다. 직전 이력에 Sitrus Berry 소비와 Rage Powder, 이후 Poltergeist가 있다. `engine/src/battle.rs`의 Poltergeist onTry는 선택 target의 item을 검사하고, 이후 hit 단계는 실제 target item을 이벤트로 내보낸다. snapshot은 item 없는 실제 target의 Item(0) 이벤트를 보여준다. `observation.rs`는 Item event의 catalogue ID에 0을 허용하지 않아 중단한다. **redirection/소비 후 no-item 이벤트의 의미를 reference와 대조하고 emitter·knowledge·observation 계약을 함께 수정해야 한다.** 단순히 예외를 무시하거나 빈 관측으로 바꾸면 안 된다.

이번 요청은 현재 속도 계측이므로 엔진 의미는 수정하지 않았다. 오류 snapshot, state, checkpoint 재현 결과와 CPU-only 재현 도구를 남겼다. 전체 catalog 검증을 약화시키는 우회도 넣지 않았다. 기본 패널의 0오류와 이 결함은 동시에 참이며, 장시간 학습 준비 완료라고 결론내릴 수 없다.

## 11. 우선순위와 기대 효과의 한계

| 우선순위 | 작업 | 근거·기대의 범위 |
|---|---|---|
| P0 | Poltergeist/no-item observation 계약 수정 + snapshot 회귀 및 reference 검증 | FPS 이전에 중단 없는 학습을 위해 필요 |
| P1 | backward의 embedding/gather/copy/elementwise 및 kernel launch 병합 | PPO가 약 61%; backward가 sampled update GPU 시간의 대부분. V100 지원과 전 gradient/Adam parity 필수 |
| P2 | rollout row 저장의 columnar/native append 재설계 | 저장 4.46초. 이를 절반으로 줄여도 전체 약 +5%이며 과거 columnar 중립 결과를 뒤집는 실측이 필요 |
| P3 | GAE와 한 번의 rollout 물질화/전송 분리·최적화 | 준비 4.47초. 준비 전체 50% 단축 가정 시 전체 약 +5% |
| P4 | 빈 관측 token trim 96→88 dual A/B | 현재 기본 OFF, 정확성 검증은 있으나 최신 dual 속도 효과 미검증 |
| P5 | actor/learner overlap 및 games>envs refill | 수집·준비 동안 GPU idle이 크지만 policy lag와 on-policy 계약 보존이 필요; 단순 thread 추가로 해결되지 않음 |
| 낮음 | NCCL-only, native battle step-only, Adam.step-only 최적화 | 현재 측정에서 순수 노출 비용이 작아 큰 배수의 근거가 없음 |

47.105초를 기준으로 PPO 20% 단축은 대략 **+14% games/s**, 수집 25% 단축은 **+7%**, 준비 50% 단축은 **+5%**다. 각 항목은 나머지가 같다는 가정이며 독립적으로 더하면 안 된다. 추가 5–15%를 검증할 여지는 있으나 하드웨어 천장이나 추가 2배를 주장할 근거는 없다. 1.3 core 수준의 수집 CPU 사용과 낮은 수집 GPU 이용률은 전체 pipeline을 개선할 여지가 남았다는 증거다.

## 12. 정확성 결과와 남은 불확실성

기본 12반복: **24,576경기, 운영 오류 0, optimizer skip 0, 4 epochs 전부 완료, rank 모델·optimizer digest 전부 일치, recompute gate 전부 통과**. 최대 sampled logprob 차이는 **2.2531e-4**, gate 1e-3 이하다. 이 검사는 표본이며 전체 trajectory의 수학적 증명이 아니다. 지속 세트 steady 표본은 6반복이어서 수시간·수일의 thermal soak나 학습 품질을 보장하지 않는다.

성공한 first-iteration 진단도 digest/recompute를 통과했다. 반면 실패 진단과 unprofiled/CPU snapshot 재현은 §10에 그대로 기록했다. 엔진 readiness는 기존 10/16이고 이번 발견으로 long-run 안전성을 더 엄격히 확인해야 한다. historical opponent 혼합, 정확한 dual resume, full-scope 규칙 검증은 이번 벤치마크의 보증 범위 밖이다. 본학습은 실행하지 않았다.

각 세트는 같은 초기 seed의 새 모델로 시작해 3반복만 학습했다. 후반 학습 정책의 경기 길이·행 수 분포는 재현하지 않았으므로 실제 장기 훈련 games/s가 같다고 보장하지 않는다.

측정하지 않은 항목: SM occupancy, Tensor Core instruction utilization, 실제 HBM GB/s, Rust 함수별 native call stack, CPU memory bandwidth. `nvidia-smi` utilization과 profiler kernel-time만으로 이 값들을 대신하지 않았다. 모든 함수의 순수 시간을 잰다고 주장하지 않으며, 주요 end-to-end 단계는 CUDA 동기화로 측정하고 더 깊은 단계는 범위·오버헤드·미귀속을 명시했다.

## 13. 원본과 재현 방법

원본 루트: `runs/perf/cooling_20261011/`. [summary.json](../../runs/perf/cooling_20261011/summary.json), [manifest.json](../../runs/perf/cooling_20261011/manifest.json), [worker_affinity.json](../../runs/perf/cooling_20261011/worker_affinity.json).

각 `coldN.json`, `persistentN.json`은 rank별 iteration/stage/collector/prepare/PPO 결과를 모두 포함한다. `*_telemetry.jsonl`은 1초 GPU·process CPU/RSS/I/O·host memory/swap 샘플, `*_before_gpu.xml`/`*_after_gpu.xml`은 NVIDIA 전체 상태, `*_case.json`은 정확한 명령·외부 시작/종료 시각이다. `diagnostic_retry.json.rank*.{collect,update}.json`과 `.events.json`은 bounded dual trace 요약, `.nccl_micro.json`은 독립 SUM 측정이다. 압축 trace·checkpoint는 로컬 산출물로 보존한다.

실행한 도구를 `scripts/perf_cooling_measure.py`, `scripts/perf_cooling_diagnostic.py`, `scripts/perf_cooling_analyze.py`로 보존했다. 이 도구의 ROOT 기본값은 이 minidc checkout이며 자동으로 전력/드라이버/서비스를 바꾸지 않는다. 새 측정을 재현하면 같은 이름의 출력 파일을 덮어쓰므로 기존 결과를 먼저 별도 보관해야 한다.

```bash
# 기본 6개 패널 (각 persistent 3반복), 1초 telemetry 포함
.venv/bin/python scripts/perf_cooling_measure.py

# 기본 GPU 계측과 분리한 bounded diagnostic
.venv/bin/python scripts/perf_cooling_measure.py diagnostic_retry 1 scripts/perf_cooling_diagnostic.py

# 기본 패널 통계 재생성 (새 GPU 실행 없음)
.venv/bin/python scripts/perf_cooling_analyze.py

# 캡처된 엔진 오류: GPU/학습/profiler 없이 CPU-only snapshot replay
.venv/bin/python scripts/perf_catalogue_snapshot_repro.py
```

historical report: [V6_PPO_THROUGHPUT_REPORT.md](V6_PPO_THROUGHPUT_REPORT.md). 이전 전력 175W/150W는 당시 조건으로 남기고 현재 hardware 문서에서 175W/175W를 정본으로 삼는다.
