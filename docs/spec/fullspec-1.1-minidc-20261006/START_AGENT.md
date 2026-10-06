# 에이전트 전달문 — PokeAgent 3.0 MiniDC 스택 반영

이번 `Full Spec 1.1`을 기존 Full Spec 1.0 대신 사용해줘. 별도 단계나 소형 실험으로 바꾸지 말고, 원래의 전체 범위 구현 → 집중 RL → 평가 기조를 유지해.

장비 운영 정본은 `reference/73-v100-driver-cuda-constraints.md`야. 실제 확인된 다음 조합에 맞춰 코드, requirements와 launcher를 작성해.

- NVIDIA driver 580.178.04, system CUDA Toolkit 12.8.2 / nvcc V12.8.93는 현재 상태를 유지.
- Python 3.12.3, torch 2.14.0+cu126, torch runtime 12.6. cu128로 올리거나 Toolkit을 12.6으로 내리지 마.
- 기존에 맞는 project venv가 있으면 재사용. 없을 때만 현재 python3로 project venv를 만들고 exact pin을 적용해.
- `runtime/constraints-minidc.txt`를 모든 runtime pip 의존성 설치에 적용하고, 충돌하면 의존성 요구를 수정해. pin을 풀거나 자동 upgrade하지 마.
- PA3는 eager PyTorch, FP16/FP32 및 SDPA math로 구현. FA2, BF16/FP8/TF32, 새 Triton/Inductor 경로를 필수로 넣지 마. 현재 llama.cpp의 sm70-attn을 자동으로 가져오지 마.
- GPU0는 NUMA0, GPU1은 NUMA1에 배정. P2P 없는 host-staged DDP로 기존 global batch 4,096을 유지하고, microbatch 256×누적8×rank2로 업데이트해. no_sync는 forward/backward 모두에 적용해.
- GPU0 175W / GPU1 150W 그대로. DDP가 실제로 불안정하거나 더 느리면 안전한 학습 경계에서 GPU0 learner 256×누적16으로 전환 가능하지만 optimizer와 counters, full-scope는 유지해.
- dsh-web/llama-swap, apt hold와 systemd를 중지/재시작/변경하지 마. pkill -f/pgrep -f도 사용하지 마. GPU 점유 때문에 필요하면 승인된 GPU 실행에서 문서화된 model unload API만 사용하고 서비스는 유지해.

전체 적격 VGCPastes 팀, 단일 PA3-8M, 전체 행동, PPO 설정, 2,048환경, 32worker, 1억 자연완결 경기 목표는 그대로야. 기본 환경 확인은 시작 절차 안에 넣고, 그것만 마친 보고서로 구현/본학습을 대체하지 마.

이 팩은 설계 업데이트이며 이 채팅에서 실제 설치, unload 또는 학습은 수행되지 않았어. 실제 자원 사용은 진행 중인 사용자 승인 범위를 따르고, 시스템 변경이 필요하면 본학습 권한과 별도로 다뤄줘.
