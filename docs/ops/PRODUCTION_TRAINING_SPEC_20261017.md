# PA3 V2 — 프로덕션 트레이닝 사양 (2026-10-17 기준)

이 문서는 2026-10-17 13:53에 유지보수(라운드로빈 구현)를 위해 **협조적 정지된** PA3 V2 PPO 캠페인의 실제 사양을 그대로 기록한 것이다.
기계가독 원본은 같은 디렉터리의 `production_training_spec_20261017.json` 이다.

- 소스: `mwl313/PokeAgent3.0` · 브랜치 `codex/v2-p00b-profile-propagation` · 커밋 **`da35700`** (tree clean)
- 런타임: Python 3.12.3 · torch 2.14.0+cu126 · CUDA 12.6 · **V100 2장, world_size 2**
- 프로필: `configs/pa3_v2_production.yaml` (sha256 `c816aa28…27941`)

## 1. 실행 형태

훈련은 유닛 파일이 아니라 **systemd transient unit** 으로 띄운다. 즉 재부팅·유저 매니저 재시작 후에는 **자동으로 살아나지 않는다**(`Restart=no`).

```
unit      pa3-v2-mode-e-da35700.service
cwd       /home/minidc/Desktop/PokeAgent3.0-V2-p00b-hotfix
exec      <venv>/bin/python scripts/run_ddp_ppo.py --profile configs/pa3_v2_production.yaml
          --games 2048 --iterations 24401 --seed 20261014
          --stop-file <run>/STOP --experimental-checkpoint <run>/boundary
          --rolling-checkpoint-root <run>/rolling --rolling-keep-generations 3
          --team-stats-root <run>/v2_final_release/.../smoke/team_stats
          --experimental-resume <generation> --experimental-source-migration <run>/.../migration.json
          --report <run>/reports/continuation.json --timeout 1800
env       PYTHONPATH=<root>/engine/python:<root>  OMP_NUM_THREADS=2  CUBLAS_WORKSPACE_CONFIG=:4096:8
props     Restart=no  KillMode=control-group  TimeoutStopSec=3min
log       <run>/reports/service.log (append)
```

`<run>` = `runs/production/mode_e_da35700_20261016`.
원래 기동 시 `--experimental-resume` 는 부트스트랩 세대(`iter-000000057344`)를 가리켰다. **정지 후 재개할 때는 최신 완전 세대를 지정해야 한다.**

## 2. 정지 시점 상태 (재개 기준)

| 필드 | 값 |
|---|---|
| 정지 시각 | 2026-10-17 13:53:25 KST (`Result=success`, `ExecMainStatus=0`) |
| 커밋 매치 | **2,764,800** |
| 절대 iteration | 675 |
| optimizer steps | 24,573 |
| learning rate | 1.497883942e-04 |
| AMP scaler | 262,144 |
| model digest | `7697797df97cf3c6d5c11466c294c8a72be1bf06d437fb1dba0c8db5109f2e6e` |
| optimizer digest | `bafeee072e87845b4dc7837d485efcf1ac79de70877fe051684dcab440d0e4c6` |
| digests equal / skipped / errors | true / 0 / 0 |
| 세대 | `iter-000002764800-1792212738467610757` (complete, iteration_complete, world 2) |
| rank payload sha256 | rank0 `fd3b5655…fe02b` · rank1 `6cd1ae7b…39607` |
| boundary | rank0 119,087,509 B · rank1 119,087,445 B |

## 3. 재개

```bash
rm -f /home/minidc/Desktop/PokeAgent3.0-V2-p00b-hotfix/runs/production/mode_e_da35700_20261016/STOP
RESUME_GEN=/home/minidc/Desktop/PokeAgent3.0-V2-p00b-hotfix/runs/production/mode_e_da35700_20261016/rolling/generations/iter-000002764800-1792212738467610757 \
  /home/minidc/bin/pa3-resume-training.sh
```

- **STOP 파일 삭제 필수** — 남아 있으면 재개 즉시 다시 정지한다.
- 헬퍼가 같은 인자·속성으로 transient unit을 다시 만든다.
- 재개 수용 기준: rank0 로그의 `resume state verified: matches=2764800 optimizer_steps=24573` + digest 일치, 2 iteration 연속(digests equal, skipped 0), **LR 재웜업 없음**, AMP scaler 262,144에서 재개, 팀 통계 중복 계상 없음.

## 4. 협조적 정지 절차

`--stop-file` 경로에 파일을 만들면, 두 rank가 **현재 iteration을 끝내고** resumable envelope을 발행한 뒤 exit 0 한다.
**SIGKILL·업데이트 중간 종료 금지.** 실측 소요는 약 2분(요청 13:51:24 → 종료 13:53:25)이었다.

## 5. 평가 서비스 (같은 시점 가동 중)

| 항목 | 값 |
|---|---|
| 브랜치 / 커밋 | `evaluation/v2-league-adaptive-v2` / `810ee3a` |
| eval root | `/home/minidc/eval-league/v2` |
| discover | `v2-eval-discover.timer` — `OnBootSec=3min`, `OnUnitActiveSec=5min`, `Persistent=true` |
| worker | `v2-eval-worker.service` — nice 15, I/O idle, **CPUAffinity 4-7**, `CPUQuota=250%`, `MemoryMax=4G`, `CUDA_VISIBLE_DEVICES=` |
| 룰셋 | `pa3-v2-promotion-rules-v2` |
| 자원 게이트 | `learner-rows-per-second-v1` (commit `810ee3a`) |

유닛 원문은 `docs/ops/unit-definitions/` 에 있다(훈련 transient 정의 + 평가 3종).

## 6. 재현 시 주의

1. **훈련 워크트리에 커밋하지 말 것** — 체크포인트가 `source_sha=da35700` 을 기록하므로, 재개 시 동일한 소스 동일성이 보여야 한다.
2. transient unit은 재부팅 후 사라진다 → 이 문서/헬퍼로 재생성.
3. 평가 워커는 CPU 전용이며 PPO 상태를 절대 건드리지 않는다.
4. 스로틀 게이트는 이제 **결정(rows)/초** 기준이다 — 매치/초는 경기 길이 증가로 자연 감소하므로 판단 근거로 쓰지 말 것.

## 7. 재개 시 필수: 체크포인트의 team-stats 절대경로 (2026-10-17 실측)

체크포인트 full-state manifest의 `team_stats_receipts` 는 **절대경로**를 담고 있고, 재개 시
`scripts/run_ddp_ppo.py` 가 `load_team_stats(ref)` 로 그 파일을 직접 읽는다
(`agent/analysis/team_stats.py:124`). 현재 세대
(`iter-000002764800-1792212738467610757`)가 참조하는 루트는 **정리 때 제거된 옛 워크트리**다.

```
/home/minidc/Desktop/PokeAgent3.0-V2/runs/pa3_v2_scratch_20261014T060525Z/team_stats/receipts/<sha256>.json.gz
```

이 경로가 없으면 재개가 즉시 죽는다(2026-10-17 19:14 실측):

```
FileNotFoundError: .../team_stats/receipts/0b0f80e7...json.gz
  File "scripts/run_ddp_ppo.py", line 947, in worker
    value = load_team_stats(ref)
```

**복원 절차** (원본 바이트, 84개 영수증 6.9MB):

```bash
mkdir -p /home/minidc/Desktop/PokeAgent3.0-V2
tar xzf /home/minidc/pa3-archive/PokeAgent3.0-V2__runs_pa3_v2_scratch_team_stats.tgz \
  -C /home/minidc/Desktop/PokeAgent3.0-V2
```

**재개 전 점검** (0이면 경로가 깨진 것):

```bash
python3 - <<'PY'
import json, os
run='/home/minidc/Desktop/PokeAgent3.0-V2-p00b-hotfix/runs/production/mode_e_da35700_20261016'
gen='iter-000002764800-1792212738467610757'
m=json.load(open(f'{run}/rolling/generations/{gen}/manifest.json'))
refs=m.get('team_stats_receipts') or []
path = refs[0]['path'] if refs and isinstance(refs[0], dict) else refs[0]
missing=[r for r in refs if not os.path.exists(r['path'] if isinstance(r,dict) else r)]
print('refs', len(refs), 'missing', len(missing), 'sample', path)
PY
```

이 디렉토리는 워크트리가 아니라 **읽기 전용 데이터**다. `git worktree` 로 되살리지 말고,
디스크 정리 대상에서 제외한다(삭제하면 다음 재개가 같은 지점에서 실패한다).
