# PokeAgent 3.0 최종 감사 및 Clean PPO 재학습 패키지

> 작성 기준: 2026-10-13. GitHub에서 확인 가능한 `mwl313/PokeAgent3.0`의 `codex/a10-ppo-training-telemetry` @ `d10ac692009566686fd32285c4456309a5f383a7`를 주된 정적 검토 대상으로 사용했다. **실제 MiniDC 복구 커밋 `266bb65`, `3df5abd`는 이 감사 환경에서 직접 조회하거나 실행 검증하지 못했다.** 실행 에이전트는 로컬 HEAD와 동작 중인 프로세스의 source SHA부터 확인해야 한다.

## 목표

1. 구 A10 PPO 약 133만 경기 모델을 **학습 초기화에 재사용하지 않고** 새로운 Observation/Action V2 정책을 scratch로 학습한다.
2. 1~5차 감사의 중대한 관측 및 행동 결함을 재현한 뒤 수정한다.
3. Rust 전투 상태, 플레이어에게 공개된 Knowledge, 신경망 관측, 합법 행동, PPO 업데이트의 **의미적 일치**를 테스트로 증명한다.
4. 새 PPO의 첫 번째 실제 경기부터 1,220팀의 팀별 W/L/D 및 대전 상대, seat, 시간 구간별 통계를 수집한다.
5. 기존 체크포인트, Eval League, 텔레메트리 및 문제 보고서는 보존한다.

## 패키지 구성과 읽는 순서

| 순서 | 문서 | 목적 |
|---|---|---|
| 1 | `AGENT_MASTER_PROMPT.md` | 에이전트에 복사하여 실행하는 최상위 영어 지시 |
| 2 | `01_AUDIT_VERDICT_EVIDENCE.md` | 확정, 의심, 반증의 근거와 실제 코드 위치 |
| 3 | `02_DEFECT_REGISTER.csv` | 결함별 ID, 우선순위, 상태, 담당 코드, 통과 기준 |
| 4 | `03_PHASED_EXECUTION_PLAN.md` | 단계별 작업과 선후관계, 중단·승격 조건 |
| 5 | `04_OBSERVATION_V2_CONTRACT.md` | 96토큰 관측, 보조 데이터, 공개 정보, 수치 정밀도 계약 |
| 6 | `05_ACTION_V2_CONTRACT.md` | Signed Target, 행동 슬롯, Conditional Candidate, Typed/Packed 동일성 |
| 7 | `06_PPO_DDP_ENGINE_VALIDATION.md` | PPO, GAE, DDP, 엔진 대조, 체크포인트 검증 |
| 8 | `07_TEAM_STATISTICS_SPEC.md` | 팀별 결과와 crash-safe 집계의 구현 규격 |
| 9 | `08_CLEAN_RETRAINING_RUNBOOK.md` | 구 모델 보존부터 신규 학습 활성화까지 운영 절차 |
| 10 | `09_TEST_MATRIX.csv` | 실험 및 회귀 테스트에 바로 쓸 수 있는 체크리스트 |
| 11 | `10_REPORT_TEMPLATES.md` | 에이전트가 채워야 할 검증 보고서 양식 |
| 12 | `prompts/` | 단계별로 실행할 경우 쓰는 짧은 영어 프롬프트 |

## 중요한 해석 원칙

- **CODE-CONFIRMED**: 확인한 소스의 생산자와 소비자 사이에 명백한 계약 불일치가 있음. 실제 로컬 빌드의 재현과 영향 범위는 여전히 테스트가 필요함.
- **RISK-REPRO**: 위험 경로가 있지만 실제 결함인지 확인하기 위한 재현 테스트가 필요함. 재현 전에 단정적으로 코드를 고치지 말 것.
- **DESIGN**: 최적화/표현력 개선 제안. 실제 수학적 버그가 아니며 최초 재학습의 차단 조건이 아님.
- **NOT-A-BUG**: 기능 구현이 확인되어 이전 우려를 정정한 항목. 방어적 회귀 테스트는 할 수 있으나 무조건 재작성하지 말 것.
- **CONFIG-BLOCKER**: 새 학습에서 반드시 일치시켜야 할 운영/데이터 정체성 조건. 과거에 잘못 학습했다는 증거와 구분할 것.

## 엄격한 출시 기준

**새 scratch PPO 프로덕션을 시작하지 말 것**, 다음이 끝나기 전까지:

1. 실제 동작하던 운영 소스와 데이터의 SHA 확보 및 안전한 구 PPO 중단/아카이빙.
2. P0 CODE-CONFIRMED 수정과 필수 부정 테스트 통과.
3. P0 RISK-REPRO를 실제 재현하고 결함이면 수정, 아니면 반증 증거 보존.
4. 관측과 후보의 정보 공개 경계, typed/packed 동등성, FP16 정밀도 검증.
5. PPO 수학, 2-rank DDP, 안전 재개, 경기 결과와 팀별 성적 원장 검증.
6. 학습 팀 풀을 정확히 결정하고 각 rank가 동일 해시의 **실제 1,220팀**을 로드했다는 증거.
7. 새 정책의 schema/model/action IDs 및 old checkpoint 불호환 차단.

## 프로젝트 운영 제약

- MiniDC Ubuntu, Tesla V100 32GB PCIe 2장, DDP, PPO-2.
- 후보 기존 설정: LR scale 0.5, effective peak LR 1.5e-4, global minibatch 4096, microbatch 1024, Mode D와 `fp16-reduction-off`, 1억 경기, historical self-play probability 0. **기존 설정은 보존하되 V2 구조가 수정되어 실제 성능 및 수치 안정성 게이트를 다시 통과해야 한다.**
- 기존 평가 리그는 **절대 임의 종료하거나 구 버전과 새 모델을 같은 리그의 비교 가능한 체크포인트로 혼합하지 말 것**. 새 V2 리그를 별도 생성한다.
- 96개 고정 토큰은 1 Global, 1 Field, 2 Side, 12 Pokémon, 48 Move, 24 Event, 8 Padding. **96개 데이터 필드가 아니며**, 토큰 외의 가변 길이 관측 부가 데이터가 있음.
- 팀별 성적은 정책의 관측에 넣지 않는다. 훈련 중 사용한 팀에 과적합하지 않도록 평가용 미사용 팀 패널은 추후 별도 추가한다.

## 사용 방법

1. `AGENT_MASTER_PROMPT.md` 전문을 로컬 코딩 에이전트에 전달하고 함께 이 압축파일의 경로를 알려준다.
2. 에이전트는 패키지를 작업 공간의 독립된 문서 디렉터리에 복사하여 체크리스트와 테스트 증거를 추적한다.
3. 복잡한 일괄 실행이 불편하면 `prompts/PHASE_0...`부터 순서대로 단계별 실행한다. **마스터 프롬프트와 단계별 프롬프트를 동시에 중복 실행하지 말 것.**
4. Phase 종료 보고서를 제출받을 때마다 `10_REPORT_TEMPLATES.md`와 `09_TEST_MATRIX.csv`를 기준으로 실제 테스트 증거와 로컬 SHA를 확인한다.

## 범위 주의

이는 **정적 감사와 실행 지시 패키지**다. 실제 엔진 빌드, RNG 고정 Showdown 대조, 두 V100 PPO 스모크 및 운영 전환은 아직 실행되지 않았다. 최종 작업의 승인 여부는 로컬 에이전트의 실제 테스트 결과에 따라 결정한다.
