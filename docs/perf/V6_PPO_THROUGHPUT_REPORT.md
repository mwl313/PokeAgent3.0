# v6 PPO 처리량 최적화 — minidc, 2026-10-09

## 판정과 측정 범위

PPO 업데이트는 가장 큰 병목이었고, 학습 목적을 바꾸지 않고 줄일 수 있었다. 이번 구현은 **PA3-8M, 양좌석 현재 정책, 자연완결 경기, 4 epochs, global minibatch 4096**을 유지한다. 최종 기본값은 microbatch 1024, CUDA rollout cache, 빈 후보 패딩 제거다. 빈 관측 토큰 제거는 정확성 검증만 통과한 선택 기능이며 기본값은 꺼 두었다.

코드 기준은 main `50c597ba56944ed62949b4e936f452cf57243e42` → 로컬 `optimization/ppo-v6-throughput`의 `3fbc749`이다. 최종 성능 패널은 수정 사항이 없는 이 HEAD에서 측정했다. 이후 보고서 커밋은 코드 변경이 없다. main/remote에는 병합하거나 푸시하지 않았다.

장비는 기존 dual V100-PCIE 32GB, 기존 전력 제한 175W/150W, PyTorch 2.14.0+cu126이다. 엔진 바이너리와 데이터, NUMA 배치, 경기 수를 유지했다. 원본 코드 복사본과 바이너리/런처 식별자는 [baseline_manifest.json](../../runs/perf/v6/baseline_manifest.json)에 있다.

## 최종 결과

| 동일 세션 재시작 비교(3회 중앙값) | 원본 50c597b | 최종 3fbc749 | 변화 |
|---|---:|---:|---:|
| 런처 all-in games/s | 28.84 | **37.23** | **+29.09%** |
| PPO update, 느린 rank | 46.07초 | **29.99초** | **−34.91%** |
| 최고 reserved GPU memory | 9.06GiB | 10.12GiB | +1.06GiB |

최종 각 실행: **34.70, 37.92, 37.23 games/s**. 원본 각 실행: 28.44, 29.67, 28.84 games/s. 초기화와 수집 시간의 변동이 남으므로 중앙값 이상으로 정밀한 예측은 피한다.

각 실행은 rank당 1,024경기, 총 2,048경기이고, 재시작 패널은 각 3회 반복의 중앙값이다. 두 버전 모두 첫 수집은 총 54,609행이다. all-in은 부모 런처 시작부터 두 rank 보고서 읽기까지이며 초기화·준비·체크포인트·종료 비용을 포함한다. 업데이트 시간은 두 rank 중 느린 쪽이다. 서로 다른 단계의 rank 최댓값을 합쳐 전체 시간을 만들지 않는다.

과거 v4 24.67(DDP)/25.60(manual), v5 32.50 games/s는 해당 시점의 참고값이다. 동일 세션 원본은 28.84 games/s로 재현됐으므로 개선율의 분모는 28.84를 사용한다. v5 32.50을 이번 재현값처럼 취급하면 안 된다.

지속 실행 보조 측정은 기존 코드 35.21 → 코어 최적화 41.74 games/s(+18.54%, iteration 2–3 가중 평균)였다. 초기화까지 포함하면 33.39 → 39.63(+18.71%)다. 다만 최적화 측정은 `65e6143`의 수정 중인 트리에서 실행한 탐색값이다. 최종 HEAD의 지속 실행 재검증으로 주장하지 않는다. 두 버전의 이후 정책 경로가 달라 steady 행 수 106,723/107,898, step 수 108/112로 달라진 점도 있다. 원본은 [baseline_persistent.json](../../runs/perf/v6/baseline_persistent.json), [core_persistent.json](../../runs/perf/v6/core_persistent.json)이다.

## 구현한 변경

- **매 epoch의 재물질화/H2D 제거:** immutable rollout을 한 번 확장하고 GPU에서 동일 셔플 인덱스로 선택한다. 캐시는 8GiB 한도, 카드별 28GiB soft budget 및 microbatch별 activation 여유를 확인한다. 큰 rollout은 기존 스트리밍으로 돌아간다. 준비 비용이 약 1.5초에서 4–5초로 이동했으므로 update 감소분 전체를 end-to-end 이득으로 해석하면 안 된다.
- **실제 후보만 계산:** 저장된 후보 길이를 기준으로 빈 trailing candidate padding만 줄인다. 후보 순서·불법 후보·선택 prefix까지 보존하며 검증 후 8의 배수로 자른다. 이 워크로드는 64 → 32폭이었다.
- **중복 모델 계산과 동기화 제거:** 후보 key를 prefix GRU에서 재사용하고 사용하지 않는 마지막 GRU를 생략했다. CUDA 텐서를 Python bool로 읽던 경로와 policy-id 선택의 GPU→CPU 왕복을 제거했다.
- **PPO 손실/optimizer 경로 정리:** value residual을 유효 행에서만 고정 shape로 계산하며, 무효 행의 NaN/Inf가 gradient에 섞이지 않게 square 이전에 마스킹한다. clip norm의 유한성 검사를 재사용해 매 파라미터 중복 검사를 줄였다.
- **정확성 수정:** manual executor의 초기 모델을 rank 0에서 방송하고, 한 rank의 overflow일 때 모든 rank scaler가 함께 감소하도록 고쳤다. digest/recompute 실패는 실제 실행 실패로 처리하고 검증 후에만 체크포인트를 원자적으로 교체한다. 실패한 자식 프로세스는 함께 정리한다.
- **운영/측정 정리:** 모델·optimizer·scaler·엔진을 유지하는 `--iterations` 모드, 누적 자연경기 LR clock, iteration별 fresh collector 및 seed를 추가했다. worker 옵션 echo를 검증한다. microbatch 1024 설정을 PPOConfig와 두 YAML에도 일치시켰다. profiler는 중복 집계 시간이 아닌 kernel interval union을 사용하도록 수정했다.

## 검증과 보류한 선택

CPU/통합 전체 테스트에서 155개가 통과했고, 새 기본값으로 바뀐 기대치 한 항목을 수정한 후 관련 14개가 모두 통과했다(총 156개 항목 검증). 실제 2-rank Gloo의 불균등/빈 rank·꼬리 마스크, cached/streaming 결과와 Adam 상태, 독립 PPO 식, 모델 전체 gradient, peer overflow, worker 실패/체크포인트 거부를 포함한다. 기존 경고 두 개(테스트 scalar 변환, 읽기 전용 NumPy 입력)는 남는다.

독립된 원본 `50c597b`와 실제 rollout 128행에서 FP16 forward/gradient를 비교했다. 최종 기본 경로는 output 차이 0, gradient 상대 L2 **7.113e-5 (0.00711%)**, 최대 절대 차이 **3.052e-5**로 선언한 0.2% 게이트를 통과했다. 부동소수점 누적 순서 때문에 bitwise 동일 학습 궤적을 약속하지 않는다. [compact_fp16_parity.json](../../runs/perf/v6/compact_fp16_parity.json)

최종 3회 모두 모델/optimizer digest 일치, recompute 게이트 통과, 운영 오류 0, 4 epochs 완료, rank별 optimizer 56 steps·skip 0을 확인했다. worker가 CUDA cache=true 경로와 compact=true, observation trim=false를 실제 적용했는지도 부모에서 확인했다.

CPU cache는 GPU cache보다 느렸고, microbatch 2048은 메모리를 크게 늘린 반면 이득이 일관되지 않아 승격하지 않았다. `--trim-observation-padding`은 96중 CPU에서 빈 것으로 증명한 tail만 생략하며 mutation/alias 검증과 원본 gradient 비교를 통과했다. 그러나 처음 dual probe에서 부모가 이 옵션을 worker에 전달하지 않는 문제를 발견했다. 전달/echo 검증을 수정했지만 해당 `trim_probe.json`은 padding 성능 근거에서 제외하고 기본값을 false로 유지했다.

## 남은 병목과 다음 실험

PPO update가 여전히 iteration의 약 60%다. [짧은 단일 GPU 프로파일](../../runs/perf/v6/optimized_kernel_profile.json)은 FP16 GEMM, attention backward, embedding/gather와 다수의 작은 elementwise kernel을 보여준다. 이 프로파일은 128경기·단일 GPU·관측 trim 활성 조건으로, 최종 dual 처리량 패널과 다르다. kernel span 내 busy 95.1%는 **SM/Tensor Core 활용률이나 하드웨어 한계**를 뜻하지 않는다. profiler wall까지 포함하면 65.8%이며, 집계 CUDA op 시간을 합산한 수치는 중복이다.

| 우선순위 | 다음 실험 | 승격 조건 |
|---|---|---|
| 1 | 최종 HEAD의 장시간 persistent 실행과 GPU cache admission 확인 | 최소 3개 독립 실행, natural-games와 rows/s, 모든 digest·overflow 게이트, 메모리 안정성 |
| 2 | 빈 관측 토큰 trim 88/96의 실제 dual A/B | worker 옵션 확인, 원본 모든 gradient 게이트, 3회 반복에서 총시간 개선 |
| 3 | embedding/scorer 및 작은 optimizer kernel 묶기 | V100 지원 경로 확인, tail/overflow/Adam-state parity, update 및 all-in 동시 개선 |
| 4 | engine collect와 learner overlap/rolling refill | games > envs 조건, policy lag 명시, on-policy 계약과 정확성 보존 |
| 5 | 대규모 rollout 물질화/전송 | 10k경기급에서 메모리 예산·fallback·동기화 측정 후 캐시/스트리밍 선택 |

추가 **5–15%**는 실험 계획의 보수적 목표, **20–35%**는 kernel/파이프라인 작업이 함께 성공하는 stretch 가정이다. 실측 보장은 아니다. 현재 근거로 추가 2배를 약속할 수 없고, 물리적 천장에 도달했다고도 판단할 수 없다. epochs 축소나 모델 축소의 이득을 동일 학습 계약의 속도 개선으로 세지 않았다.

엔진 readiness는 여전히 10/16이며, 장시간 학습 품질·정확한 dual checkpoint resume·historical opponent 경로는 이번 속도 검증 범위 밖이다. 100M 본학습은 실행하지 않았다.

## 재현

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/run_ddp_ppo.py \
  --games 1024 --envs 1024 --workers 16 --microbatch 1024 \
  --minibatch 4096 --timeout 1800 \
  --checkpoint runs/perf/v6/recheck.pt --report runs/perf/v6/recheck.json
```

지속 실행은 `--iterations 3` 이상을 추가한다. 캐시 제외 A/B는 `--batch-cache none`, 후보 trim 제외는 `--no-compact-candidates`다. 기본값에서도 메모리 admission 실패 시 안전하게 스트리밍한다. 3회 최종 원본: [promoted1](../../runs/perf/v6/promoted1.json), [promoted2](../../runs/perf/v6/promoted2.json), [promoted3](../../runs/perf/v6/promoted3.json).
