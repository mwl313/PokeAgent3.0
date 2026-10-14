# V2 PPO 병목 분석 및 최적화 실험 제안 — 2026-10-14

분석 기준은 **2026-10-14 15:29:06 KST에 확보한 완료 로그**다. 학습은 계속 진행되므로 이 문서의 경기 수는 실시간 현재값이 아니다.

## 1. 결론

현재 이어서 실행한 5 iteration의 처리량은 **31.35 games/s**다. 첫 실행에서 보고한 **27.55 games/s**는 프로세스 시작 비용을 포함한 값이다. 최초 iteration을 포함한 6 iteration은 24,576경기 / 792.846초 = **31.00 games/s**다. 이 분모는 iteration 시간의 합이며, 두 실행 사이의 운영 대기 시간을 포함하지 않는다.

가장 큰 병목은 **수집 및 rank 대기 50.91%**, 다음은 **PPO 업데이트 37.91%**, **배치 준비 9.19%**다. 특히 수집 내부에서 Python의 행 저장, 후보 테이블 생성, 관측 해석이 Rust 전투 step보다 훨씬 오래 걸린다. 따라서 현재 속도를 V100 두 장의 하드웨어 한계라고 판단할 근거는 없다. 동시에 수정 전의 90 games/s를 현재의 정확성 계약에서도 곧바로 달성할 수 있다고 보장할 근거도 없다.

권장 순서는 후보 매핑 중복 제거 → rollout 저장·배치 생성 비용 축소 → forward의 반복 검증과 동기화 비용 측정·축소다. PPO objective, shuffle, 합법 행동, 관측 정보 경계, DDP 전역 가중치와 체크포인트 검증은 유지해야 한다. **이번 작업은 분석·문서화이며 운영 최적화를 배포하지 않았다.**

## 2. 실행 환경과 측정 방법

- 실행 소스: `5ccc51b0cd4a49efb93018258b9750d5e804ae32`.
- 운영 소스 고정 worktree: `/home/minidc/.codex/worktrees/pa3-v2-runtime/PokeAgent3.0`.
- 실행 디렉터리: `/home/minidc/Desktop/PokeAgent3.0-V2/runs/pa3_v2_scratch_20261014T060525Z`.
- GPU: Tesla V100-PCIE-32GB × 2, 설정 전력 제한 각 **175W**, 개선된 냉각 사용.
- PPO 2 epochs, global minibatch 4096, microbatch 1024, rank당 2048 natural games/iteration, global 4096 games/iteration.
- ModeD, FP16 reduction off, deterministic algorithms 활성화, LR scale 0.5, rollout/shuffle seed 20261014. 신규 모델 초기화 seed는 모델 생성기의 기본값 **0**이다.
- 실제 모델 파라미터 9,889,755개. frozen 팀 풀 1,220개, historical opponent 비율 0.
- native 모듈 SHA256: `8f3643a9f7b2febfa567f039b911d7af2f88bbf7bfc912849ada9e6a690d1c62`.
- 팀 파일 SHA256: `ecabd47a8d000ba8dd9f3010c09992505fe21e95efe244e2c8a3d75ac3f70b1f`.

기존 JSONL에서 newline까지 완성된 행만 읽고 두 rank를 absolute iteration으로 맞췄다. source, match clock, 모델·optimizer digest 일치와 중복·누락을 검사했다. 처리량은 `global games / max(rank iteration wall)`로 산출했다. 두 GPU의 처리량을 다시 더하지 않는다.

시간 비중은 각 iteration에서 **전체 wall이 긴 하나의 rank**의 stage를 합산했다. 수집 직후 LR clock collective에서 생기는 대기를 수집과 묶었다. stage별로 rank 최대값을 각각 고른 뒤 합하면 다른 rank에서 이미 수집 중이던 시간을 다시 세므로 잘못된 합계가 된다. 원본에는 비교용 stage별 최대값도 따로 보존했다.

로그를 읽고 짧은 `nvidia-smi` 관측만 수행했다. 학습에 profiler를 붙이거나 추가 GPU benchmark를 병행하지 않았고 프로세스를 재시작하지 않았다. 모델이 계속 바뀌는 짧은 운영 표본이므로 동일 고정 정책을 반복 측정한 benchmark와 구별해야 한다.

## 3. 실제 처리량과 정상성

| Iteration | 누적 natural games | Wall 초 | Games/s | Global learner rows | 누적 optimizer steps | 최대 recompute 오차 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 4,096 | 139.578 | 29.346 | 108,401 | 54 | 9.54e-07 |
| 2 | 8,192 | 137.045 | 29.888 | 107,682 | 108 | 9.54e-07 |
| 3 | 12,288 | 131.259 | 31.206 | 107,228 | 162 | 0.000215 |
| 4 | 16,384 | 129.004 | 31.751 | 105,101 | 214 | 9.54e-07 |
| 5 | 20,480 | 127.735 | 32.066 | 103,449 | 266 | 9.54e-07 |
| 6 | 24,576 | 128.225 | 31.944 | 104,281 | 318 | 9.54e-07 |

6개 모두 두 rank에서 실제 2 epochs를 수행했고, optimizer skip 및 operational error는 0이었다. 모든 완료 경계에서 모델·Adam digest가 rank 간 일치했다. 최대 관측 recompute 오차는 `2.145767e-4`로 기존 `1e-3` 기준 이내다. 이 표본의 통과는 향후 모든 상태에서 오류가 없다는 증명은 아니다.

Continuation 5개는 20,480경기 / 653.268초이며 iteration별 29.89–32.07 games/s다. 행 수가 감소하고 캐시·시스템 상태도 달라지므로 속도 상승을 단순한 GPU 워밍업 효과로 단정할 수 없다. 긴 경기를 많이 돌리면 자동으로 90 games/s로 오른다는 증거는 없다.

**체크포인트와 진행 수를 구분해야 한다.** 캡처 시점 메모리 내 검증 완료 clock은 24,576이지만 마지막 게시 완료 durable checkpoint는 **4,096**이다. generation은 `iter-000000004096-1791958101732995099`다. 통상 약 100k 간격 저장 설정 때문에 둘은 다르며, 갑작스러운 전원 차단 시 메모리 내 진행분이 모두 복원되는 것은 아니다.

## 4. 전체 시간 예산

Continuation 기준 iteration 평균 130.654초:

| 구간 | 초/iteration | 비중 | 해석 |
|---|---:|---:|---|
| 수집 + clock 대기 | 66.514 | 50.91% | 환경·관측·후보·기록 및 수집 rank 불균형 |
| PPO update | 49.537 | 37.91% | 선택, forward/backward, optimizer, 통신·동기화 포함 |
| 준비 | 12.003 | 9.19% | 관측 결합, 후보 재구성, GPU 캐시, GAE 등 |
| 나머지 및 잔차 | 2.599 | 1.99% | 검증, 통계, 정리, 저장, barrier 등 |
| 합계 | 130.654 | 100% | 하나의 critical rank 기준 |

첫 iteration의 stage별 rank 최대값은 collect 73.299초, update 50.889초, prepare 12.338초다. LR clock 3.314초는 scheduler 산술이 느린 것이 아니라 먼저 끝난 rank가 다른 rank를 기다리는 collective 시간이 포함된 값이다. 이 최대값들을 위의 additive 예산에 다시 더하지 않는다.

첫 iteration recompute gate 0.248초, digest 0.777초, 팀 통계 0.166초, checkpoint 0.833초, cleanup 0.589초였다. 이들은 각각 rank 최대값이므로 합산 attribution으로 사용하지 않는다. 현재 gate를 없애서 수십 games/s를 회복할 수 있는 구조는 아니다. 드문 대형 checkpoint 게시 비용은 이 짧은 표본만으로 장기 평균을 확정할 수 없다.

## 5. 수집 내부: 가장 큰 개선 후보

아래 세부 profile은 **최초 production iteration만**의 값이다. 후속 JSONL에 같은 세부 계측이 없으므로 이를 최신 5개 평균으로 표현하지 않는다.

| 수집 세부 구간 | Rank 0 초 | Rank 1 초 |
|---|---:|---:|
| Rollout 행 기록 | 30.494 | 32.560 |
| Python 후보 테이블 구성 | 13.877 | 15.301 |
| 관측 parsing | 9.226 | 9.282 |
| 정책 forward | 5.874 | 5.902 |
| Host → GPU 관련 | 3.141 | 3.179 |
| Native 관측 생성 | 3.785 | 3.611 |
| Rust battle step | 0.351 | 0.347 |
| Native 후보 생성 | 0.265 | 0.244 |
| 전체 수집 | 69.855 | 73.298 |

각 rank는 54,439 / 53,962 learner rows, 약 842k / 839k 후보 레코드를 처리했다. 수집 process CPU time은 77.315 / 80.711초, wall 대비 약 1.11 / 1.10 CPU core다. 이는 수집 주 경로가 많은 CPU 코어를 충분히 쓰지 못한다는 단서지만, GIL이나 개별 thread의 점유율을 직접 profile한 결과는 아니다. workers=16이라고 Python 기록 코드까지 16배 병렬화되는 것은 아니다.

실제 소스에서 확인되는 비용 경로:

1. `agent/train/native_collector.py::_level_table_packed`는 request별 Python loop, 배열 생성, `token_refs` 매핑을 수행한다. 문서상 vectorized 표현만 보고 전부 벡터화됐다고 판단하면 안 된다.
2. `_packed_row_inputs`에서도 row/level별 매핑을 다시 만든다. `agent/types/action_mapping.py::canonical_records`는 int64 복사와 signed target 검증을 수행하며 `token_refs`도 이 canonicalization을 거친다.
3. `agent/buffer/rollout_buffer.py::record_packed`에서 다시 canonicalization하고 NumPy 값을 Python list/tuple과 `RolloutRow`로 바꾼다. 동일 요청을 actor용·저장용으로 반복 가공하는 구조다.
4. 기본 Inline store 경로는 고정 배열을 slice-copy한 뒤 `InlineObservationStore.add_compact`에서 다시 복사한다. ragged 배열도 소유권 확보를 위해 복사한다. object array의 shallow copy와 내부 ragged 데이터 deep copy를 혼동하면 안 된다.
5. `stacked`/`to_batch`에서 필드별 concatenate, 후보 padding 재구성, scalar tensor 생성이 이어진다. 이미 생성한 정보를 다시 Python 객체에서 배열로 되돌리는 비용이 발생한다.

Rust step을 두 배 빠르게 해도 이 표본에서 rank당 약 0.17초만 절약한다. 반면 행 기록과 테이블 생성에는 rank당 약 44–48초가 들어간다. 엔진 mechanics를 바꾸는 최적화보다 **동일한 합법 행동·관측 데이터를 덜 복사하고 덜 재구성하는 것**이 우선이다.

## 6. 준비 및 GPU 업데이트

준비 단계에서 GPU cache build는 rank당 9.572 / 8.941초, GAE는 2.735 / 2.503초였다. GPU cache는 켜져 있으며 예상 용량은 약 3.94 / 3.90 GiB, 예산 8 GiB, candidate padding 32다. 이 용량은 admission용 추정값이지 정확한 전체 GPU 점유량은 아니다. GPU peak reserved는 13.395 / 13.318 GiB였다.

| PPO 세부 구간 | Rank 0 초 | Rank 1 초 |
|---|---:|---:|
| Forward | 21.617 | 21.101 |
| Backward | 18.445 | 18.096 |
| Optimizer 블록 | 6.463 | 7.332 |
| Minibatch 선택 | 2.940 | 2.725 |
| H2D 표시 구간 | 0.141 | 0.380 |
| Metrics | 0.382 | 0.410 |
| 전체 내부 update | 50.256 | 50.887 |

**이 표는 CUDA kernel 시간표가 아니다.** `perf_counter`로 측정한 host wall이며 비동기 CUDA launch, 뒤늦은 동기화, collective 대기가 인접 구간으로 귀속될 수 있다. Optimizer 블록에는 gradient clipping, finite 검증, rank consensus, AMP step 등이 포함된다. Backward에는 DDP communication이 포함될 수 있고 H2D 표시 구간에도 count collective가 있다. 현재 자료로 attention·ragged encoder·NCCL 각각의 정확한 GPU 비중을 말할 수 없다.

코드상 추가 조사 가치가 높은 지점:

- `agent/types/observation.py::validate_semantics`와 `active_to_roster`: CUDA tensor의 `bool(any())`, CPU 변환 및 row loop가 존재한다.
- `agent/model/embeddings.py`: forward마다 semantics 검증을 호출한다. V2 numeric·flag·known 입력은 이전보다 풍부하며 이를 삭제해서 속도를 비교하면 다른 모델이다.
- `agent/model/ragged_embeddings.py`: 5개 ragged family의 embedding/projection/segment reduction 및 검증이 있다. 빈 family에서도 DDP graph 연결을 유지해야 한다.
- `agent/model/scorer.py`: 범위·entity·move 검증의 CUDA scalar 판정이 동기화를 유발할 수 있다. 잘못된 입력의 fail-closed 동작은 유지해야 한다.
- `agent/types/ragged.py::select/narrow`: offset/cumsum 및 index 구성의 반복을 줄일 여지가 있다. writable alias가 남아 있는 상태에서 cache를 신뢰하면 안 된다.
- `agent/ppo/learner.py`: shuffled cached selection이 54회 호출됐지만 **contiguous hit 0회**였다. 연속 선택 fast path는 현재 실전 경로를 개선하지 못했다. 이를 위해 `torch.randperm`을 없애면 PPO sampling 계약을 바꾸게 된다.
- `scripts/run_ddp_ppo.py`: `find_unused_parameters=True` 경고만으로 끄는 것은 위험하다. empty family·padding·zero actor 사례까지 검증해야 한다. NCCL 환경값을 바꿀 근거가 되는 communication trace도 아직 없다.

## 7. GPU가 쉬는 이유와 하드웨어 한계

수집·준비·업데이트를 순차 수행하고 CPU의 row 처리 중에는 GPU에 큰 작업이 공급되지 않는다. 따라서 nvtop의 순간 0%는 학습 중단이나 GPU 성능 한계의 증거가 아니다.

약 2초 간격 15회 관측에서 GPU 0/1의 평균 사용률은 11.87% / 15.07%, 최대 64% / 61%, 최고 온도 48°C / 45°C였다. **약 30초짜리 특정 phase 표본**이므로 전체 iteration 평균 GPU 사용률로 일반화할 수 없다. GPU 0 reported power 최대값은 190.19W, GPU 1은 172.34W였고 설정 limit은 모두 175W였다. 순간 보고값·평균 창 차이가 있을 수 있으며 이 자료만으로 power throttling을 배제하거나 확정할 수 없다. 온도는 과열을 시사하지 않지만 throttle-reason trace를 확보한 것은 아니다.

순간 PCIe link는 Gen3 x16, memory clock 877MHz였다. 지속 PCIe bandwidth와 메모리 bandwidth 포화는 측정하지 않았다. 메모리 capacity에 여유가 보여도 microbatch 두 배가 반드시 들어가거나 빨라지는 것은 아니다. CPU 처리 병목과 불필요한 동기화를 먼저 줄인 뒤 GPU kernel trace로 남은 하한을 확인해야 한다.

## 8. 과거 25 / 90 games/s와의 공정한 비교

| 기록 | 보고 처리량 | 작업량·차이 |
|---|---:|---|
| V4 operational frontier | mean 25.09, median 24.67 | 2048 global games, 약 54,609 rows, micro256, 56 optimizer steps; collection 약13초, update 약59–62초 |
| A5 control | 92.57 all-in, 93.94 steady | 기존 훈련 정책, recompute gate 비활성, 당시 알려진 오차 1.0507107e-3 |
| A9.3 D1 | 88.41 all-in, 89.45 steady | 51,200 games / 25 iterations, 2048 games·2 epochs·18 steps/iteration, 당시 numerical gate 25/25 통과 |
| V2 현재 | 31.00 iteration 전체, continuation 31.35 | 4096 games/iteration, 약103k–108k rows, 2 epochs·52–54 steps, V2 관측·행동 correctness 및 deterministic 실행 |

원문: `docs/perf/V4_OPERATIONAL_FRONTIER_REPORT.md`, `docs/experiments/A5_HISTORICAL_SELF_PLAY_REPORT.md`, `docs/experiments/A9_2_THROUGHPUT_ROOT_CAUSE_REPORT.md`, `docs/experiments/A9_3_FINAL_PERFORMANCE_REPORT.md`, `docs/experiments/A9_3_PRECISION_STABILITY_RESULTS.json`.

A5 보고서에는 **rows/game 산술 불일치**가 있다. control의 원시 합계 875,177 rows / 51,200 games는 **17.09 rows/game**인데 본문은 34.19로 표기한다. historical도 748,530 / 51,200 = 14.62인데 29.24로 표기한다. 새 비교에서는 합계에서 직접 나눈 값을 사용하며 과거 기록 자체를 고쳐 쓰지 않는다.

V2 최초 iteration은 108,401 / 4096 = **26.47 rows/game**, A5 control 대비 약 1.55배다. 신생 정책과 이미 훈련된 정책의 경기 길이, 팀 풀, 올바른 row 처리 차이가 섞여 있다. 이 증가를 전부 V2 모델 변경 탓으로 돌릴 수 없다.

A9.3은 4096 games로 정규화하면 36 optimizer steps, V2 최초는 54로 **경기당 작업량 약 1.5배**다. A9.3 rank1 update 약10.407초 / 18 = 0.578초/step, V2 약50.889 / 54 = 0.942초/step으로 약1.63배다. 따라서 update 비용/game 차이는 대략 **2.45배**다. 이는 서로 다른 실행의 정규화 비교이지 특정 기능 하나의 원인 분리 실험이 아니다.

A9.3 numerical gate 통과를 부정할 필요는 없다. 다만 당시 통과한 계약과 이후 V2에서 수정한 의미·관측 계약은 같지 않다. A5의 gate-off 속도를 현재 production 목표로 그대로 사용할 수도 없다. V4와 현재 games/s가 비슷해도 collection/update 비율은 크게 바뀌었으므로 같은 병목·같은 하드웨어 ceiling을 뜻하지 않는다.

P4에서 bulk decoder는 동일 1024-view payload를 0.8911→0.0944초로 줄였고 입력 hash가 일치했다(`docs/audit_v2/p4/DECODE_BENCHMARK.json`). 이것은 국소 개선 9.44배이며 전체 PPO가 9배 빨라졌다는 뜻은 아니다. P4 non-deterministic 약29.76과 deterministic 약27.01 all-in 차이도 소수 실행의 참고값이다. 운영 deterministic 설정을 제거할 근거로 삼지 않는다.

## 9. 우선순위 실험

| 순위 | 실험 | 확인할 지표와 승인 기준 |
|---|---|---|
| 1 | 후보 canonical record/token reference를 request batch에서 한 번 만들고 sampling·저장에서 재사용 | table+record 시간 감소, 후보 순서·signed target·roster·logprob 완전 일치 |
| 2 | 소유권이 명확한 block/columnar rollout 저장, 고정 필드 이중 복사 및 list/tuple 왕복 제거 | record+prepare 감소, ragged·known mask·row hash·mutation isolation 검증 |
| 3 | batch materialization 및 ragged offset/index 생성 벡터화 | prepare+shuffled select 감소, shuffle/sample 순서와 gradient 유지 |
| 4 | 고정 benchmark에서 CUDA events/NVTX trace 후 반복 validation 동기화 축소 | 실제 kernel/host sync 분리, boundary 검증·immutable proof·invalidation 유지 |
| 5 | micro1024→2048 제한 실험 | 같은 global4096 objective·DDP gradient, 메모리·cache admission·steps/sec 확인 |
| 6 | DDP unused graph 및 통신 옵션의 좁은 실험 | empty/padding/zero actor correctness와 trace에서 입증된 통신 비용 |

기존 Columnar store가 있다는 이유만으로 운영 flag를 즉시 바꾸지 않는다. 관측 소유권·복원 config hash·V2 ragged 호환성을 먼저 확인해야 한다. validation을 무조건 삭제하는 것 역시 최적화가 아니다. 입력이 생성된 경계에서 검증하고 이후 불변성을 보증하는 방식이어야 한다.

Micro2048은 cache admission의 activation reserve 추정을 키워 **GPU rollout cache를 꺼 버릴 수도 있다**. 현재 peak13.4GiB만 보고 무조건 이득이라고 판단하지 않는다. 연속 select 최적화, Rust battle step, 작은 gate 제거는 우선순위가 낮다.

동일 iteration 안에서 frozen policy의 CPU 준비와 GPU inference를 겹치는 설계는 후속 후보가 될 수 있다. 그러나 업데이트된 정책 대신 이전 정책으로 다음 iteration을 미리 수집하면 on-policy PPO 계약이 달라진다. 이를 무료 pipeline 최적화로 취급하지 않는다.

## 10. 기대치와 90 games/s 조건

현재 continuation 시간 모델은 `T = 66.514(C) + 49.537(U) + 12.003(P) + 2.599(O) = 130.654초`다. 이는 측정 구간에 대한 분해이며 물리적 하드웨어 하한은 아니다.

- Update만 절반: T≈105.885초 → **38.68 games/s**.
- 수집·대기를 전부 없앤 가상 상황: T≈64.139초 → **63.86 games/s**.
- Update만 남긴 가상 상황: 4096/49.537≈**82.69 games/s**.
- 90 games/s에는 iteration 전체가 **45.51초** 이하여야 한다. 현재 update만으로도 이를 넘으므로 CPU와 GPU 양쪽 개선이 필요하다.

첫 iteration의 상세 시간으로 단순 가산 모델을 만들면 rank0 record+table 약44.37초를 절반으로 줄여 29.35→약34.9 games/s가 된다. 여기에 prepare 25%, update 15% 절감까지 가정하면 약38.4다. record+table 70%, prepare50%, update40% 절감을 가정하면 약49.8이다. 이것은 **아직 구현·측정되지 않은 절감률을 넣은 예시**이며 stage 상호작용과 rank 병목 이동을 반영하지 못한다.

보수적인 다음 목표는 **35–40 games/s**, 여러 경로가 실제로 개선되면 **40–45**, CPU 객체 경로 및 GPU sync가 크게 줄어드는 stretch 목표는 **45–55**로 제안한다. 90+는 더 큰 변경과 측정이 필요한 연구 목표로 남긴다. 이 범위는 신뢰구간이나 보장값이 아니다. 정책이 강해지면서 rows/game이 줄면 games/s가 구현 개선 없이 오를 수도 있어 rows/s와 optimizer steps/s를 반드시 같이 봐야 한다.

## 11. 재현 가능한 후속 benchmark 계약

운영 GPU에 무거운 실험을 병행하지 않는다. 별도의 정해진 실험 시간 또는 사용자가 승인한 안전 경계에서 replica를 사용한다. 현재 checkpoint/source/data/초기 RNG를 고정하고 각 arm 최소3회, 실행당5–10 iterations의 제한 benchmark를 권장한다. cold start와 steady, 실제 natural games와 learner rows, 실제 epochs/KL early-stop, optimizer steps/skips, phase wall, peak memory를 기록한다.

기존 objective와 전역 DDP weighting, action legality, hidden-information 경계, `torch.randperm` 순서, optimizer/scheduler/scaler/RNG 복원, immutable checkpoint 계약을 유지한다. 값·mask·sample ordering·gradient·recompute `1e-3` 기준·rank digest·실제 자연 종료 accounting을 확인한다. cache mutation 및 빈 ragged/zero actor도 포함한다. 반복 실험에서 대략10% 이상 일관된 end-to-end 이득을 보인 뒤 승격을 검토한다. 작은 이득도 의미는 있지만 측정 잡음과 구분해야 한다.

새 코드 SHA로 V2 checkpoint를 이어받으려면 명시적인 호환성 증거 및 지원되는 migration이 필요하다. 현재 source check를 무시하거나 기존 legacy A10 migration을 임의로 넓혀서는 안 된다. 이번 문서 commit은 운영 고정 worktree의 SHA를 바꾸지 않는다.

## 12. 운영 상태와 증거 위치

캡처 당시 PPO `pa3-v2-ppo-20261014-060525.service`는 active/running, 평가 timer `pa3-v2-evaluation-20261014-060525.timer`는 active/waiting이었다. 평가 service는 timer 실행 사이 inactive/success일 수 있다. 아직 첫 4096 policy만 등록된 시점이므로 신뢰할 head-to-head 학습 향상 결과는 없다. 평가와 학습 경기 카운터는 구분한다.

Persistent training telemetry는 run root 아래 `reports/continuation.json.rank0.ppo.jsonl` 및 `reports/continuation.json.rank1.ppo.jsonl`에 기록된다. 최초 상세 결과는 `reports/first.json`, rank 보고서 및 로그에 있다. 운영 manifest는 `release/manifest.json`, 체크포인트는 `rolling/generations/`, 평가는 `evaluation/v2/`에 있다. 모델 checkpoint는 로컬에 보존하며 Git에는 넣지 않았다.

본 보고서와 함께 저장한 `docs/perf/v2_bottleneck_20261014/SUMMARY.json`, 4개 JSONL snapshot, `first.json`, `launch_manifest.json`, `gpu_samples.json`은 캡처 시점 증거다. `scripts/analyze_v2_live_bottlenecks.py`는 read-only 재계산 도구다. GPU 샘플은 짧은 보조 자료이며 kernel trace를 대신하지 않는다.

남은 불확실성은 전체 iteration GPU occupancy·kernel별 비용·NCCL overlap·실제 메모리 bandwidth·장기간 checkpoint overhead·정책 강도에 따른 경기 길이 변화다. 이 보고서는 이들을 측정한 것처럼 수치를 만들지 않았다. 현재 근거로는 하드웨어 ceiling보다 **Python 데이터 경로와 반복 동기화·학습 작업량**이 더 설득력 있는 설명이다.
