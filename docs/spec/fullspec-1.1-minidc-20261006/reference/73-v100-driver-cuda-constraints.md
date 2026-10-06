# 73. 미니DC V100 하드웨어·드라이버·CUDA 제약 정본 (2026-10-06)

**Audience:** 미니DC에서 GPU를 쓰는 작업(추론·학습·시뮬레이터·커스텀 커널)을 맡는 에이전트/작업자
**Author:** Aria (Haven orchestrator, Mac mini)
**Created:** 2026-10-06
**성격:** 제약 시트(handoff). 여기 적힌 버전은 **실측 검증된 값**이고, 하나라도 올리면 **조용히(silent) 깨진다** — 에러 없이 결과만 틀리거나 커널 이미지가 무효가 된다.
**근거:** miniDC 직접 조회(2026-10-06) + 미니데이터센터 docs 01, 02, 26, 27, 38, 62, 64, 65.

---

## 0. HARD RULES (다른 무엇보다 우선)

| # | 규칙 |
|---|---|
| **R1** | **드라이버를 절대 올리지 말 것.** `580.178.04` 고정. R580 = Maxwell/Pascal/Volta 마지막 지원 브랜치이고 585+ 는 V100 미지원이다. `apt upgrade`도 금지(드라이버·툴킷이 딸려 올라간다). |
| **R2** | **CUDA 툴킷은 12.8 고정. 13.x 금지.** CUDA 13은 sm_70(Volta)을 제거했다. 12.9.x도 llama.cpp 계열에서 sm_70 빌드 버그가 있다. |
| **R3** | **PyTorch는 `2.14.0+cu126` 핀.** `pip install -r requirements.txt`를 그대로 돌리면 torch가 최신으로 올라가 V100이 조용히 죽는다. torch를 먼저 핀으로 설치하고, requirements에서 `torch*`를 제외한 뒤 나머지를 설치한다. |
| **R4** | **bf16, fp8, int8 텐서코어, FlashAttention2를 기대하지 말 것.** V100(Volta)에 해당 하드웨어가 없다(FA2 = sm_80+). fp16/fp32 + SDPA/xformers로 간다. |
| **R5** | **환경 변경(드라이버·툴킷·apt·systemd unit)은 사람 승인 후에만.** GPU 사용 전 `nvidia-smi`로 점유자 확인. llama-swap이 모델을 물고 있으면 `curl -X POST http://127.0.0.1:8080/api/models/unload`. |
| **R6** | **`dsh-web` / `llama-swap`을 재시작·중지·kill 하지 말 것** (박스 서비스이자 원격 작업의 I/O 채널). `pkill -f` 금지 — 자기 SSH 세션을 죽인다. 정리는 `pgrep -x` / `pkill -x`로만. |

---

## 1. 실측 환경 (2026-10-06 miniDC 직접 조회)

| 항목 | 값 |
|---|---|
| OS / 커널 | Ubuntu 24.04.5 LTS / 6.8.0-142-generic |
| GPU | Tesla V100-PCIE-32GB ×2, **compute capability 7.0**, VRAM 32,768 MiB ×2 |
| 드라이버 | **580.178.04** (`nvidia-driver-580-server 580.178.04-0ubuntu0.24.04.1`) |
| CUDA 툴킷 | **12.8.2** (`cuda-toolkit-12-8`, nvcc V12.8.93), `/usr/local/cuda` → `cuda-12.8` |
| 상한/하한 | 드라이버 585+ 불가, CUDA 13.x 불가, CUDA 12 요구 드라이버 하한 525 (현재 580으로 충족) |
| 툴체인 | gcc 13.3.0, cmake 3.28.3, ninja·ccache 설치됨, Python 3.12.3, pip 24.0, **uv 없음** |
| GPU 토폴로지 | **SYS** — GPU0(05:00.0)=NUMA0(CPU 0-19,40-59), GPU1(84:00.0)=NUMA1(CPU 20-39,60-79). **P2P 불가** (GPU 간 전송은 호스트 경유, 실측 ~4GB/s) |
| 전력 상한 | GPU0 **175W**, GPU1 **150W** (`nvidia-powerlimit.service` override, 부팅 자동 적용) |
| 패키지 hold | **55개 hold (2026-10-06 적용)** = nvidia 580-server 계열 15종 + CUDA 12.8 계열 39종 + `libnvidia-egl-wayland1` |
| 자동 업데이트 | `unattended-upgrades` 활성 — 단 **보안 업데이트만**(기본 정책상 `noble-updates` 제외), 자동 재부팅 없음. hold 패키지는 자동 업데이트에서 제외됨 |
| 검증된 PyTorch 조합 | **torch 2.14.0+cu126** — 이 박스에서 실동작 확인(arch list에 `sm_70` 포함, `cuda.is_available()=True`) |
| 설치된 엔진 빌드 산출물 | `/opt/llm/llama.cpp`(mainline 4fea119, build 완료), `/opt/llm/sm70-attn`(v0.7.0 a57720f, build 완료), `/opt/strata`(v0.1.36) |
| 디스크 | / 218G 중 여유 89G |
| modprobe | `options nvidia-drm modeset=0` (컴퓨트 전용) |
| 기타 | python3-venv 사용 가능, `/opt/minidc-tools/vadd`(sm_70 연산 검증 바이너리, 양 카드 PASS 이력) |

---

## 2. 왜 이 조합 하나뿐인가 (근거)

Volta(sm_70)는 2026년에 신규 스택에서 순차 제외됐다.

- CUDA **13** = sm_70 제거 → 툴킷 13 불가
- PyTorch **2.11+ (cu128)** = sm70 커널 제거, **2.15+** = cu126 휠 발행 중단 → torch 상한 2.14
- vLLM **0.20+** 프리빌드 = sm_70 제외
- Triton 최신 README = "Compute Capability 8.0+" 요구
- JAX **cuda13 휠** = SM 7.5+ 요구 (cuda12 휠은 SM 5.2+ = V100 OK)
- 드라이버는 **R580이 마지막 브랜치** → 615.x 계열 완화책(int8_convrot 리셋 완화 등)을 이 박스에서는 **못 쓴다**

따라서 "돌아가는 조합"은 하나로 좁혀져 있고, 그 조합은 이미 이 박스에서 검증됐다:
**드라이버 580.178.04 + CUDA 12.8 + (torch 2.14.0+cu126 또는 jax[cuda12]) + sm_70 컴파일.**

---

## 3. 스택별 호환 매트릭스

| 스택 | V100 판정 | 비고 / 핀 |
|---|---|---|
| PyTorch 2.14.0 + cu126 | ✅ 검증 | 이 박스 실동작. 2.15+ cu126 휠 없음, 2.11+ cu128은 sm_70 없음 |
| **JAX `jax[cuda12]`** | ✅ | 공식 문서: CUDA 12 = SM 5.2+ 지원, 드라이버 ≥525. **`jax[cuda13]`은 SM 7.5+ = V100 불가** |
| llama.cpp 직접 빌드 | ✅ | `-DCMAKE_CUDA_ARCHITECTURES=70-real -DGGML_CUDA_FORCE_MMQ=ON`, CUDA 12.6~12.8 |
| sm70-attn 포크(fishlikeX) | ✅ | V100 FlashAttention 복원 + DFlash2, prefill 이득. shape 게이트로 자동 발동/폴백 |
| FlashAttention 2 | ❌ | sm_80+ 전용 → `torch.nn.functional.scaled_dot_product_attention` / xformers 사용 |
| vLLM | ⚠️ | 프리빌드 **≤0.18.x**만. 0.20+ sm_70 제외. 소스빌드(CUDA 12.6)로 sm_70 재활성 가능 |
| bitsandbytes (QLoRA nf4) | ✅ | 단 설치 시 torch를 올리지 않게 핀 준수 |
| torch.distributed DDP (2장) | ✅ | 2장 데이터 병렬 정상 (실사용 확인) |
| TP / model parallel / `-sm tensor` | ❌ | P2P 없음(SYS) → 카드별 독립 또는 `-sm layer`만 |
| Triton (최신) | ⚠️ | README 요구 CC 8.0+ → Triton 커널 의존 경로는 **버전 실측 전 도입 금지** |
| torch.compile (inductor) | ⚠️ | sm_70 inductor 컴파일 실패 가능 → 실패 시 `TORCHDYNAMO_DISABLE=1` 폴백 |
| cupy / numba CUDA | 미확인 | 금지는 아니지만 **쓰기 전 실측** (설치만 하고 가정하지 말 것) |

---

## 4. 설치 레시피 (검증된 형태)

### 4.1 PyTorch venv

```bash
python3 -m venv ~/<작업디렉터리>/.venv
source ~/<작업디렉터리>/.venv/bin/activate
pip install --upgrade pip
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
# requirements가 있으면 torch 계열을 제외하고 설치
grep -viE '^(torch|torchvision|torchaudio)' requirements.txt > /tmp/req.notorch.txt
pip install -r /tmp/req.notorch.txt
```

검증(4개 전부 기대값이어야 통과):

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); \
print(torch.cuda.is_available(), torch.cuda.get_device_capability(), 'sm_70' in torch.cuda.get_arch_list())"
# 기대: 2.14.0+cu126 12.6 / True (7, 0) True
```

### 4.2 JAX (GPU 벡터화 경로)

```bash
pip install "jax[cuda12]"
python -c "import jax; print(jax.devices())"
# 기대: [CudaDevice(id=0), CudaDevice(id=1)]
```

- `jax[cuda13]` 금지. 휠 설치 후 `jax.devices()`가 실제로 GPU를 잡는지까지 확인(휠만 깔고 가정 금지).

### 4.3 CUDA C++ / 커스텀 커널

```bash
nvcc -arch=sm_70 ...        # 헤더: /usr/local/cuda-12.8
```

- 사전 검증 바이너리: `/opt/minidc-tools/vadd` (양 카드 PASS 이력 — 커널/드라이버 건강 체크로 재사용)

### 4.4 GPU를 안 쓰는 코드도 제약은 있다

순수 CPU(파이썬·C++)로 도는 코드는 이 문서 §4.1~4.3의 GPU 핀과 무관하다. 다만 학습·검증 단계에서 GPU를 다시 물리면 같은 제약이 즉시 적용되고, GPU 간 통신(§5)과 전력/열(§7) 규칙은 그대로 걸린다.

---

## 5. 멀티GPU 규칙

- `nvidia-smi topo -m` = **SYS**. GPU0=NUMA0, GPU1=NUMA1.
- **P2P 불가** → GPU 간 텐서 이동은 호스트 스테이징(실측 ~4GB/s, Gen3 x16 실효의 약 1/3).
- **허용:** DDP(데이터 병렬), 카드별 독립 프로세스/서빙, 배치 분할.
- **금지:** 텐서 병렬, 토큰마다 all-reduce 하는 구조, 잦은 GPU 간 텐서 교환.
- DDP 시 `numactl --cpunodebind=<n> --membind=<n>`로 카드↔NUMA 고정 권장.
- 열 특성이 카드마다 다르다: GPU1이 열위(장시간 부하 시 소프트 스로틀) → 장시간 작업은 GPU0 우선.

---

## 6. GPU 공유 함정 (OOM 방지)

llama-swap(:8080)은 유휴에도 **마지막 모델을 VRAM에 계속 물고 있다.**

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader   # 점유자 확인
curl -X POST http://127.0.0.1:8080/api/models/unload                                 # 필요 시 언로드
```

- 현재 모델 로스터: `strata-iq3-xxs`, `strata-uncensored` (둘 다 대형, GPU0/GPU1 VRAM 대부분 점유).
- 언로드는 다음 요청 때 자동 재로드되므로 되돌리기 부담이 없다. **GPU 작업 중에는 LLM에 요청을 보내지 않는다**(재로드로 OOM 재발).
- "Free VRAM이 0인데 내 프로세스는 20GB만 쓴다"는 서명이 나오면 십중팔구 이 모델이다.

---

## 7. 변경 금지 / 주의 목록 (체크리스트)

- ❌ `apt upgrade`, `apt install cuda-toolkit-*`(12.8 외) — 툴킷·드라이버 상승 위험. **2026-10-06 기준 nvidia 580-server 15종 + CUDA 12.8 39종 + libnvidia-egl-wayland1 = 55개에 `apt-mark hold`가 걸려 있다.** 해제는 `sudo apt-mark unhold <패키지>`, 전체 해제는 `sudo apt-mark unhold $(apt-mark showhold)`. hold를 풀고 업그레이드하는 판단은 사람 승인 사항(검증: `apt-mark showhold | wc -l` = 55).
- ⚠️ 커널: unattended-upgrades가 **보안 커널 업데이트를 자동 설치**하면 DKMS가 nvidia-srv 모듈을 새 커널용으로 재빌드한다(같은 6.8.0 계열이면 실패 확률 낮음). 커널 변경 후 **재부팅 전** `dkms status`(nvidia-srv/580.178.04 ... installed)와 `nvidia-smi`로 확인할 것 — 실패 시에도 이전 커널로 부팅하면 복구된다.
- ❌ `pip install torch` (핀 없이), `pip install -r requirements.txt` 그대로, `pip install --upgrade`
- ❌ `pip install jax[cuda13]`
- ❌ `to(torch.bfloat16)` / bf16 하드코딩 경로, FlashAttention2 import
- ❌ `systemctl stop|restart llama-swap dsh-web`, `pkill -f`, `pgrep -f`
- ❌ 패키지/노드 설치 후 torch 핀 재확인 없이 진행(설치가 torch를 끌어올리는 사례 있음)
- ⚠️ 대형 배치·대형 컨텍스트: GPU0 VRAM 여유를 계산하고 시작
- ⚠️ 200W 승격 금지 — 실측상 GPU1은 150W에서도 장시간 부하 시 스로틀. 현행 175/150W 유지.

---

## 8. 한눈 요약

**드라이버 580.178.04 + CUDA 12.8 + (torch 2.14.0+cu126 | jax[cuda12]) + sm_70 컴파일. bf16·FA2 없음, P2P 없음, GPU0 175W / GPU1 150W.**

이 조합을 벗어나는 순간 V100은 조용히 죽는다. 변경이 필요하면 먼저 사람 승인.

---

## 9. 출처

- miniDC 직접 조회 2026-10-06: `nvidia-smi`(드라이버·cc·전력), `nvcc --version`, `dpkg -l`, `nvidia-smi topo -m`, venv 조회(torch 2.14.0+cu126 / sm_70), `/opt` 빌드 산출물
- 미니데이터센터 docs: 01(하드웨어), 02(소프트웨어), 26(소프트웨어 준비 체크리스트), 27(엔진 V100 판정), 38·64(전력·쿨링), 62(Strata 평가), 65(설치 런북)
- V100 실측 기록: 드라이버 윈도우(R570/R580 록인), 엔진 호환 매트릭스, torch 핀, 멀티GPU 토폴로지, JAX·Triton 제약
- JAX 공식 문서(설치): CUDA 12 = SM 5.2 이상, CUDA 13 = SM 7.5 이상 (2026-10-06 확인)
- Triton README: "NVIDIA GPUs (Compute Capability 8.0+)" (2026-10-06 확인)
