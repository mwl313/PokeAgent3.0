# PokeAgent 3.0 Full Scope Spec 1.1 — MiniDC 고정 환경

**전체 팀풀의 고속 배틀 엔진 → PA3-8M 집중 자기대전 RL → 모델 평가**를 유지한다. 이번 개정은 하드웨어와 실행 스택 반영이며 연구 축소가 아니다.

## 바로 읽을 파일

`POKEAGENT_3_0_FULL_SPEC.md`가 통합 정본, `PA3_TRAINING_CONFIG.yaml`이 대응 설정이다. `START_AGENT.md`는 전달문, `RUNTIME_CHANGES.md`는 변경 내역이다. `PA3_TEAM_RECORD_SCHEMA.json`은 이전 버전 그대로다.

## 고정 실행 조합

| 항목 | 적용 |
|---|---|
| driver | 580.178.04 |
| system CUDA Toolkit | 12.8.2, nvcc V12.8.93 |
| Python | 3.12.3 |
| PyTorch | 2.14.0+cu126, runtime 12.6 |
| GPU | V100 32GB ×2, sm_70, SYS/P2P 불가 |
| power | GPU0 175W / GPU1 150W |
| update | host-staged NCCL DDP, global minibatch 4,096 |
| execution | eager, FP16/FP32, SDPA math |

`runtime/minidc.env`는 프로젝트 shell에 source할 환경변수만 담으며, 설치/서비스 unload/전력 변경/학습을 실행하지 않는다. 실제 실행 코드가 YAML의 backend 및 NUMA 설정을 적용해야 한다.

`runtime/constraints-minidc.txt`는 torch pin 보호용이다. 전체 의존성 lock은 실제 설치 환경에서 확정하며 이 파일만으로 모든 전이 의존성이 고정됐다고 하지 않는다.

`reference/73-v100-driver-cuda-constraints.md`는 장비 정본의 원문 사본이다. 원문은 참고 자료이며 내부 설치 예시를 자동 실행하지 않는다. 이 패키지에는 driver/Toolkit 설치나 시스템 서비스 변경 스크립트가 없다.

## 바뀌지 않는 본학습

PA3-8M, VGCPastes 적격 전량, M-C CTS, 전체 관측/행동, PPO 수치, 1억 자연완결 경기, 2,048환경/32worker, 저장/평가 주기는 유지한다. 원래의 full-scope 패키지를 대체하는 통합본이므로 과거 ZIP을 중첩 첨부할 필요가 없다.

## 확인 범위

장비 실측은 사용자 보고를 계승했고, 이 작업에서는 문서/YAML/파일 구조만 확인했다. 미니DC 접속, GPU 또는 DDP 실행, 설치, 서비스 제어, 팀 수집이나 학습은 하지 않았다. 기존 Full Spec 1.0 원본은 그대로 남겼다.
