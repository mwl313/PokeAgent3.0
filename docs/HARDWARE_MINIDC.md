# MiniDC 현재 하드웨어 정본

갱신: **2026-10-11 (Asia/Seoul)**. 사용자 확인과 현재 장치 조회를 구분해 기록한다.

| 항목 | 현재 상태 |
|---|---|
| CPU | Intel Xeon E5-2673 v4 ×2, 총 40 physical cores / 80 threads |
| NUMA | 2 nodes; node0 CPUs 0–19,40–59 / node1 CPUs 20–39,60–79 |
| RAM | 62.77GiB (현재 /proc/meminfo) |
| GPU0 | Tesla V100-PCIE-32GB, PCI 00000000:05:00.0, NUMA0, **175W limit** |
| GPU1 | Tesla V100-PCIE-32GB, PCI 00000000:84:00.0, NUMA1, **175W limit** |
| 냉각 | 사용자가 두 카드 모두 냉각 개선·안정 상태임을 2026-10-11 확인 |
| 부하 중 냉각 검증 | 기본 12반복·24,576경기 동안 최고 GPU0 **57°C**, GPU1 **53°C**; 1초 샘플 thermal slowdown **0회** |
| GPU 연결 | PCIe Gen3 x16, GPU 간 SYS topology. 현재 PPO NCCL은 host staging 설정(P2P OFF, SHM ON) |
| 드라이버 | 580.178.04 (현재 nvidia-smi XML) |
| 시스템 | Linux 6.8.0-146-generic |
| PyTorch | 2.14.0+cu126, wheel CUDA runtime 12.6 |
| CUDA Toolkit | nvcc 12.8.93 (toolkit 12.8), driver 표시 CUDA 최대 버전과 별개 |
| Rust engine | Rust 1.90.0, locked release + python feature; 엔진 바이너리 SHA는 측정 manifest 참조 |

## 전력·냉각 기록 해석

**현재 GPU0/GPU1 제한은 175W/175W다.** 과거 175W/150W 또는 GPU1 냉각 비대칭 표기는 2026-10-06~09 당시의 역사값이며 현재 운영값이 아니다. 이번 작업은 현재 설정을 조회·기록했으며 power cap·fan·driver·서비스를 변경하지 않았다. 이 기록은 향후 agent에 전력 변경 권한을 주는 지시가 아니다.

기본 PPO 부하에서 power-cap 사유는 GPU0 약 34.1%, GPU1 약 21.8%의 샘플에 나타났고 thermal slowdown은 나타나지 않았다. 따라서 안정 냉각과 전력 제한에 따른 clock 변동은 구분한다. 1초 샘플 및 약 11분 기본 패널의 범위이며, 수시간/수일 동안의 상태를 보장하지 않는다.

## 현재 벤치마크 배치와 성능

rank0는 NUMA0 CPU 0–19,40–59, rank1은 NUMA1 CPU 20–39,60–79에 실제 affinity가 잡혔다. numactl은 node별 memory policy도 설정한다. Rust workers는 rank당 16, OMP/MKL threads는 1이다. 특정 16개 코어만 고정하는 설정은 현재 launcher에서 사용하지 않는다.

동일 코드 `6eb48c0`의 재시작 3회 중앙값 **36.563 games/s**, 독립 persistent 3세트 steady 가중 합산 **43.477 games/s**. 자세한 단계·온도·clock·전력·메모리·통신 결과와 발견된 엔진 오류는 [2026-10-11 상세 보고서](perf/COOLED_DUAL_V100_BOTTLENECK_REPORT_2026-10-11.md)에 있다. 현재 속도를 장기 학습 안정성 완료로 해석하지 않는다.

## 근거와 역사 문서

- [현재 manifest](../runs/perf/cooling_20261011/manifest.json), [요약](../runs/perf/cooling_20261011/summary.json), [실제 CPU affinity](../runs/perf/cooling_20261011/worker_affinity.json).
- [2026-10-06 장비·소프트웨어 제약 원본](spec/fullspec-1.1-minidc-20261006/reference/73-v100-driver-cuda-constraints.md)은 당시 snapshot으로 보존한다. 현재 전력/냉각/커널/디스크 상태는 이 문서와 새 측정을 우선한다. 원본의 모든 외부 호환성 주장을 이번에 다시 검증한 것은 아니다.
- [2026-10-09 v6 보고서](perf/V6_PPO_THROUGHPUT_REPORT.md)의 175W/150W는 그 측정 당시 조건이다.
