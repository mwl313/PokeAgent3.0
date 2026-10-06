# PokeAgent 3.0 Full Spec 1.1 — MiniDC 적용 변경표

작성: 2026-10-06. 이 문서는 환경 개정의 설명이며 실제 미니DC 실행 영수증이 아니다.

## 판단

**연구 방향과 본학습 범위는 유지하고 실행 스펙만 수정한다.** Rust CPU 엔진 → 전체 적격 팀의 PA3-8M scratch PPO → 평가. 1억 자연완결 경기 목표, 2,048 환경, 32 worker, global minibatch 4,096과 PPO 설정을 그대로 둔다. 별도 호환성 실험 단계, 소형팀 커리큘럼, model sweep은 추가하지 않는다.

| 영역 | Full Spec 1.0 | Full Spec 1.1 |
|---|---|---|
| Python | 3.11 | 장비의 3.12.3 |
| PyTorch | 2.6.0/cu124 초기 기준 | 로컬 검증 2.14.0+cu126 정확 pin |
| 시스템 driver | 구체값 없음 | 580.178.04 유지 |
| 시스템 CUDA Toolkit | 구체값 없음 | 12.8.2, nvcc V12.8.93 유지 |
| Torch runtime | cu124 | 12.6, 시스템 12.8과 역할 구분 |
| 설치 | 구현 시 lock | 기존 venv 우선, exact constraints, 무단 pip/apt upgrade 금지 |
| attention / compiler | 포괄적 fallback/추후 compile 가능 | eager SDPA math, compile/Inductor/Triton 필수 경로 없음 |
| GPU 연결 | NVLink 없음, NUMA 탐지 | SYS/P2P 불가, host staging, NUMA0/1에 rank별 고정 |
| DDP | 256×8×2 | 그대로. no_sync forward/backward, 마지막 microbatch에서 동기 |
| single learner fallback | 같은 global batch | GPU0 우선, 256×16=4,096 명시 |
| 전력 | 구체값 없음 | GPU0 175W, GPU1 150W 유지 |
| 서비스 | 타 프로세스 무단 종료 금지 | dsh-web/llama-swap 보호, 필요한 승인 실행에서 unload API만 |
| 본학습 | 전체 팀, 1억 경기, 동일 PPO | 변경 없음 |

## 자료의 의미

장비 수치와 `torch 2.14.0+cu126` 실동작 근거는 첨부 `73-v100-driver-cuda-constraints.md`를 계승했다. 공식 문서는 CUDA runtime/Toolkit 공존과 NCCL host transport의 일반 원리를 확인하는 데 사용했다. 모든 다른 버전이 언제나 silent failure를 낸다고 일반화하지 않는다. 현재 공개 다운로드 목록의 표시만으로 장비 설치를 다시 검증했다고 하지 않는다. 정확한 pin의 신규 다운로드가 실패하면 검증된 기존 venv/wheel을 재사용하고 다른 버전으로 자동 교체하지 않는다.

## 에이전트가 읽어야 할 점

이번 자료 때문에 이미 합의한 full-scope를 작은 점검 작업으로 대체하지 않는다. 필요한 시작 metadata와 첫 실제 batch/collective 확인은 원래 작업에 포함하고, 정상인 경우 전체 구현 및 본학습으로 이어간다. 코드에 아직 없는 실행 설정은 실제 launcher/model/installer에 구현해 적용해야 하며, YAML 키만 추가했다고 DDP/SDPA 제약이 자동으로 강제되지는 않는다.

원 패키지와 장비 보고서는 삭제하거나 덮어쓰지 않았다. `PA3_TEAM_RECORD_SCHEMA.json`은 원본과 바이트 동일하다.
