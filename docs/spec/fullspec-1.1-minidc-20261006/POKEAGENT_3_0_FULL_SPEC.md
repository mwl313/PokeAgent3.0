# PokeAgent 3.0 상세 구현 및 본학습 명세

**문서 버전:** 3.0 Full Spec 1.1 — MiniDC 실행환경 고정  
**작성일:** 2026-10-06  
**주 경로:** 고속 배틀 엔진 제작 → 전체 팀풀의 집중 자기대전 RL → 모델 평가  
**주력 모델:** PA3-8M, 무작위 초기화  
**장치:** 미니DC, V100 PCIe 32GB 두 장  
**환경 정본:** 사용자 제공 `73-v100-driver-cuda-constraints.md` (2026-10-06). 원문을 `reference/`에 보존한다.  
**개정 범위:** 실행환경, NUMA/DDP 전송, 전력 및 서비스 보호. 팀 수집, 모델, 입력/행동, PPO 수치와 1억 경기 목표는 변경하지 않는다.  
**문서의 효력:** 구현 및 본학습을 위한 설계 정본이다. 이 문서 작성으로 실제 수집, 구현, 학습이나 계정 변경을 수행한 것은 아니다.

---

## 0. 한눈에 보는 결정

| 항목 | 결정 |
|---|---|
| 납품 목표 | 전체 대상 팀을 처리하는 배치 엔진, 1억 자연완결 경기 목표의 본학습 실행, 평가된 체크포인트 |
| 연구 단계 | 엔진 제작, 집중 RL, 모델 평가의 세 가지 |
| 모델 | 6층, 차원 320, attention head 5개, FFN 1,280인 약 8M급 Entity Transformer 하나 |
| 초기화 | 모든 학습 가중치를 무작위 초기화. Laya, P1, AUX, 외부 pretrained 모델 미사용 |
| 학습법 | PPO + GAE, 동결 정책으로 수집한 뒤 업데이트 |
| 보상 | 자연 승리 +1, 자연 패배 -1, 규칙상 무승부 0 |
| 팀 데이터 | VGCPastes가 연결하는 공개 원문에서 6마리 모두 상세 배분이 있는 팀을 전량 수집 |
| 규정 | `[Gen 9 Champions] VGC 2026 Reg M-C`, CTS, 단판 |
| 팀 수 | 실제 수집과 정제 후 확정. 적격 팀 수에 인위적 상한 없음. 1,000은 희망 규모이지 확보 사실이나 필수 문턱 아님 |
| 데이터 분리 | 같은 종 구성의 세트 변형을 묶은 팀 그룹 기준 train/dev/final 약 90/5/5 |
| 시작 범위 | train 전체를 첫 본학습부터 사용. 4개 → 16개 → 64개 같은 팀 커리큘럼 없음 |
| 행동 | 4마리 선출, 리드, 두 슬롯의 기술/타겟/자원/자발적 교체, 강제 및 피벗 보충 |
| 정책 입력 | 도감 특징, 자기 세트, 해당 선수에게 공개된 정보. 자연어와 외부 LLM 호출 없음 |
| 엔진 | Rust 배치 라이브러리. Showdown은 규칙 기준 및 전이 평가에만 사용 |
| 기본 규모 | 환경 2,048개, CPU worker 32개, 두 GPU 배치 추론 및 동기 데이터 병렬 업데이트 |
| 고정 스택 | 드라이버 580.178.04, 시스템 Toolkit 12.8.2, Python 3.12.3, torch 2.14.0+cu126 (런타임 12.6) |
| GPU 연결과 전력 | P2P 없음, SYS/호스트 경유 DDP, GPU0 175W / GPU1 150W 유지 |
| 실행 커널 | FP16/FP32, eager PyTorch SDPA math. FA2, BF16/FP8/TF32, Inductor/Triton 필수 경로 없음 |
| 학습 배치 | 새 learner 의사결정 최소 131,072개를 수집, PPO 4 epochs, global minibatch 4,096 |
| 탐험 | categorical sampling, entropy 정규화와 약한 uniform reverse-KL |
| 상대 | 현재 정책 50%, 같은 구조의 과거 정책풀 50%, 최대 8개 |
| 자동 운영 | 같은 목표 안에서 구현, 오류 수정, 재개, 저장 및 정기 평가까지 진행 |
| 검색 | 본학습에는 없음. 검색 없는 정책 평가 후 별도 후속 기능 |

**중요:** 세부 수치는 이번 프로젝트의 설계 선택이다. mikumiku37의 비공개 레시피를 복원한 값이나 V100 실측 최적값이 아니다. “전체 범위”는 실제 확보한 적격 전체 팀과 그 효과 의존성을 뜻하며, 존재하지 않는 팀 수를 문서로 만들어 내지 않는다.

## 1. 이 문서가 이전 계획에서 바꾸는 것

기존 `POKEAGENT_3_0_PLAN.md`의 모델 하나, scratch RL, Rust 엔진, CTS 원칙은 계승한다. 다만 “소수 팀의 최소 정상 시스템 → 첫 업데이트 → 나중에 팀과 처리량 확대”를 납품 단계로 삼는 부분은 이 문서로 대체한다.

첫 목표는 작은 데모가 아니다. 팀 수집과 전체 효과 범위 산출, 배치 엔진, PA3-8M, PPO, 재개와 평가를 하나의 작업으로 완성한다. 개발 중 작은 단위검사나 짧은 실행이 필요하면 내부적으로 수행하고 즉시 다음 구현과 본학습으로 이어간다. 그것을 별도의 사용자 승인 단계나 신규 연구 과제로 분리하지 않는다.

소형 검사와 대량 학습은 대립하지 않는다. 검사는 잘못된 보상이나 숨은 정보 누수로 대량 계산을 낭비하지 않기 위한 코드의 일부다. 매 경기 clone 검증, 매 행동 전체 자산 해시, 반복 보고서, 같은 자료의 전수 재검사는 학습 경로에 넣지 않는다.

이번 실행 경로는 1.0/2.0의 가중치, 텍스트 정책, AUX 데이터, validator와 복잡한 재개 팩에 의존하지 않는다. 기존 자료를 물리적으로 지우지는 않는다.

## 2. 고정 규정과 지원 범위

### 2.1 규칙 기준

- 포맷 ID: `gen9championsvgc2026regmc`.
- 정보: CTS. 공개 팀시트 수락 옵션은 끈다.
- 경기: 단판, 6마리 팀에서 4마리 선출, 두 마리 리드.
- reference: 공식 `smogon/pokemon-showdown`의 고정 commit.
- 이번 조사에서 확인한 reference 후보: `14546894d86f9589ac11130c510bbe73b6968665` [W3, W4]. 실행자는 이 commit의 포맷과 컴파일 결과를 한 번 연결한다. 이후 자동으로 master를 갱신하지 않는다.
- 지원 자원은 포맷 규칙에서 추출한다. 현재 읽은 Champions 구현의 Mega 및 Tera 처리와 일반 Gen9를 혼동하지 않는다. 일반 Gen9라는 이유로 Terastallization을 임의 활성화하지 않는다.
- 팀 적법성, stat formula, 기술 효과, PP, 동률 처리, 시간 및 무승부 규칙은 reference 포맷을 따른다.

### 2.2 전체 팀풀이 엔진 범위를 정한다

수집된 적격 train/dev/final 전체 inventory에서 종, 기술, 아이템, 특성, 변신, 필드와 연쇄 호출 효과를 추출한다. 엔진은 그 전체 범위와 서로의 상호작용을 처리해야 한다. 유효한 팀을 “엔진 구현이 어렵다”는 이유로 데이터에서 조용히 제거하지 않는다.

Metronome, Copycat, Transform 등 다른 효과나 기술을 호출하는 규칙이 포함되면 호출 가능한 의존성까지 범위에 넣는다. 포켓몬 이름 목록만 지원하고 실제 효과를 생략하는 방식은 안 된다.

미지원 효과는 구현 작업으로 남기고 해결한다. 본학습 중 새로운 결함이 확인되면 영향을 받는 경험만 격리하거나 공통 오류면 수집을 멈추고 수정한다. 팀풀 변경이나 규칙 변경이 있었다면 기록한다. 이를 숨기고 “전량 지원”이라고 하지 않는다.

## 3. VGCPastes 팀 수집: 수량은 전수 수집 결과로 확정

### 3.1 수집 원천

VGCPastes의 공개 팀 목록은 공개 스프레드시트의 규정별 탭과 연결된 paste 원문으로 구성된다. MunchStats의 공개 구현은 이 목록과 paste를 읽어 캐시 및 공개 snapshot으로 제공하며, `has_evs` 필터도 제공한다 [W1, W2]. 목록의 표시만으로 원문 6마리의 배분 완전성을 확정하지 않는다.

수집 범위는 **Champions M-A, M-B, M-C 탭의 모든 공개 팀**이다. 경기 규정은 전부 M-C로 통일한다. M-A/M-B 출처라도 원 세트를 바꾸지 않고 M-C에서 합법이면 사용한다. 출처별로 개수를 같게 맞추지 않는다.

기본 경로:

1. VGCPastes 공개 목록을 페이지 또는 탭 단위로 전부 읽는다.
2. 목록의 paste 및 명시적인 full-team/report 링크를 해석한다.
3. 접근 가능한 공개 paste의 실제 텍스트를 받는다.
4. 6마리 각각의 배분과 세트를 파싱한다.
5. 배분 단위와 원 세트의 M-C 적법성을 확인한다.
6. 중복을 제거하고 팀 그룹을 나눈 뒤 전체 inventory를 고정한다.

수집 속도를 위해 현재 공개 export/CSV나 문서화된 JSON snapshot을 우선할 수 있다. 미러를 쓰면 원 VGCPastes 팀 ID와 원 paste URL을 보존하고 미러를 독립적인 추가 팀으로 세지 않는다. 원 목록과 snapshot이 다르면 버전과 누락을 기록한다.

이번에 확인한 공개 연결 정보 [W2]:
- sheet ID: `1axlwmzPA49rYkqXh7zHvAtSP-TKbM0ijGYBPRflLSWw`
- 탭: `Champions M-A`, `Champions M-B`, `Champions M-C`
- MunchStats repository slug: `champions-ma`, `champions-mb`, `champions-mc`
- 공개 조회 코드 경로: `PizzaTimeJoshua/munchstats/vgcpastes.py`
- published snapshot 기준 경로: `PizzaTimeJoshua/munchstats` 저장소의 `teams-data` branch, `site/`

이 경로들은 이번에 소스로 확인한 접근 단서다. 실제 수집 시 response schema와 출처를 확인하며 고정 열 번호나 오늘의 팀 총수를 영구 API 계약으로 가정하지 않는다.

### 3.2 “상세 EV가 있는 팀”의 정확한 정의

**여섯 포켓몬 모두 배분 줄이 명시되어 있어야 한다.** 다섯 마리에만 배분이 있거나 팀 링크가 OTS만 제공하면 주 학습풀에서 제외한다.

각 포켓몬의 필수 자료:

| 필드 | 채택 기준 |
|---|---|
| species/form | 정확한 종과 형태가 식별됨 |
| moves | 네 기술이 명시됨. 엔진에서 실제 네 기술을 유지 |
| ability | 원문에 명시되거나 포맷의 공식적인 단일값이 명확히 결정함. 추정으로 인기 특성을 채우지 않음 |
| item | 아이템 또는 명시적인 no-item. no-item은 포맷 적법성을 확인 |
| nature | 원문에 명시. beta export의 +/− 표기는 공식 파서의 의미대로 해석 |
| allocation | EV/SP 등 명시적인 전체 배분 줄. 이름이 빠진 능력치는 그 줄의 포맷 규칙에 따라 0 |
| IV/level 등 | 공식 포맷의 생략 기본값만 사용하고 `defaulted_by_format`으로 기록 |
| source | 원 paste와 원 목록 ID, 가져온 시각, 원문 hash |

배분 줄 자체가 없는 것과, `EVs: 252 Atk / 4 SpD / 252 Spe`에서 나머지 세 능력치가 0인 것은 다르다. 후자는 완전한 배분 표현일 수 있다. 전체 생략을 임의로 0이나 252/252로 메우지 않는다. 명시적인 all-zero 배분은 오류로 단정하지 않으며 해당 규칙에 따라 검사한다.

**Champions 주의:** Showdown은 `evs`라는 필드에 Champions의 Stat Points를 저장할 수 있다. 필드 이름이 EV라고 일반 세대의 0~252 EV 단위라고 가정하면 안 된다 [W3, W4].

다음 메타데이터를 보존한다.
- `allocation.raw_line`
- `allocation.source_unit`: `champions_stat_points`, `legacy_evs`, `unknown`
- `allocation.unit_evidence`: 출처 규정, 명시적 SP label, 문서화된 source adapter 등
- `allocation.parsed_values`: HP/Atk/Def/SpA/SpD/Spe 정수
- `allocation.engine_values`와 `stats_at_battle_start`
- `conversion`: 변환이 없으면 `none`; 정확한 출처 매핑이 있으면 그 ID와 원값

일반 EV를 Champions로 바꾸는 공식을 인터넷 예시만 보고 넣지 않는다. 원 자료가 legacy 단위라는 근거와 reference와 동일한 문서화된 mapping이 있을 때만 변환하고 provenance를 남긴다. 단위가 애매하거나 손실 있는 변환만 가능하면 그 팀은 `ambiguous_allocation`으로 분리한다. 누락 배분을 생성하거나 다른 팀의 배분을 복사하지 않는다.

### 3.3 수집 운영

- 공개 원천만 읽는다. 비밀번호, 인증 제한 또는 CAPTCHA를 우회하지 않는다.
- host당 최대 동시 요청 2개, 기본 초당 요청 1개, Retry-After와 robots/서비스 조건을 따른다.
- 정상 수집 캐시를 재사용한다. 429/5xx/네트워크 오류는 최대 5회 지수 backoff로 재시도한다.
- 수집 대상 URL allowlist, redirect 재검증, 응답 크기 상한을 두어 로컬/사설 주소를 따라가지 않는다.
- 막힌 링크 하나로 전체 수집을 원점 재시작하지 않는다. 원 목록 대비 완료/누락을 남긴다.
- 팀에 EV가 있다고 표시되지 않아도 접근 가능한 paste를 확인할 수 있다. 목록 플래그의 거짓 음성 때문에 완전 팀을 놓치지 않는다.
- 수집은 설계 스냅샷에서 한 번 완료하고, 본학습 중 자동으로 새 팀을 추가하지 않는다.
- 코드와 데이터의 라이선스 및 출처를 구분한다. 원문 공개와 재배포 허용을 같은 것으로 취급하지 않는다.

### 3.4 중복과 분할

**정확 중복:** 종, 세트, 배분, 실제 의미 있는 옵션을 정규화해 팀 순서와 닉네임 같은 비전략 표기 차이를 무시한 fingerprint. 원 출처는 모두 합쳐 보존한다. 서로 다른 세트는 별도 팀이다.

**팀 그룹:** Mega를 단순히 무시하지 않는 포맷별 종/형태 정규화 후, 여섯 종 구성의 unordered fingerprint를 그룹으로 삼는다. 같은 여섯 종의 기술/아이템/배분 변형은 같은 그룹에 두어 train과 평가로 흩어지지 않게 한다. 이것은 비슷한 다섯 종 팀까지 전부 미노출이라는 보장은 아니다.

그룹 ID의 안정적인 hash 순서로 train/dev/final 그룹을 약 90/5/5에 배정한다. 그룹 크기가 달라 팀 수 비율은 정확히 90/5/5가 아닐 수 있다. 모든 분할에 있는 그룹 및 팀 수를 실제로 기록한다.

`N_eligible = N_train + N_dev + N_final`. 모든 `N_train` 팀을 첫 본학습 reset부터 균등 표집한다. 희망 규모는 1,000개 이상의 고유 train 팀이지만, 엄격한 배분 조건으로 확보 수가 적더라도 임의 세트로 숫자를 채우거나 작은 임시 팀 단계로 바꾸지 않는다. 적격 전량을 사용하고 실제 수량을 보고한다.

### 3.5 팀셋의 효과 범위와 평가 누수

dev/final 팀의 효과도 엔진이 구현할 수 있어야 한다. 도감과 효과 지원 목록을 확인하는 것은 정책에 해당 팀의 세트를 학습시키는 것과 다르다. final 팀의 대전 결과로 모델과 하이퍼파라미터를 고치지 않는다.

팀 ID, 대회 순위, 작성자, source regulation, 상대의 실제 팀 인덱스를 actor에 넣지 않는다. 기본 표집은 승리자나 유명 선수에 가중하지 않는 고유 팀 균등이다. 대회팀으로 초기화해도 사람 행동을 정답으로 쓰지 않으므로 BC가 아니다.

## 4. 고속 엔진 구현

### 4.1 구조

Rust + PyO3의 로컬 배치 라이브러리를 만든다. 외부 서버/WebSocket, 후보별 clone 검증, Python에서 개별 효과를 처리하는 hot loop를 없앤다. 기존 공개 Rust doubles 구현에서 코드 재사용이 가능하면 라이선스와 실제 지원을 확인해 가져온다. 언어 이름만으로 성능과 정확성을 승계하지 않는다.

두 actor 프로세스에 Rust 환경 묶음을 하나씩 붙인다. 각 묶음은 기본 1,024환경, 16 CPU worker thread다. 전체 2,048환경/32 worker이며, 첨부 실측 프로필의 GPU0/NUMA0와 GPU1/NUMA1에 각각 메모리 및 CPU를 묶는다. 실제 PCI 주소와 CPU map을 시작 때 대조하고, 환경마다 별도 OS 프로세스를 만들지 않는다. GPU1의 열적 제약은 worker/배치 배치의 운영 문제이며 전체 팀풀이나 PPO 목표 축소의 이유로 사용하지 않는다.

상태는 정수 ID, 고정 또는 사전 할당 배열, 정확한 정적 테이블, bitset을 사용한다. 모든 상태의 동적 길이를 억지로 작게 고정하지 않는다. 효과 범위에서 필요한 최대 필드와 overflow-safe storage를 설계한다. 엔진의 정확한 HP/피해/난수는 정수 또는 reference와 동일한 계산을 유지한다.

### 4.2 공개 인터페이스

| API | 계약 |
|---|---|
| `reset_batch(team_a, team_b, seeds, role_map)` | 전체 세트로 게임 초기화, world state와 선수별 knowledge 초기화 |
| `observe_batch(handles)` | 지금 행동해야 할 side와 typed observation, request kind, 후보와 mask 반환 |
| `step_batch(handles, actions)` | 실제 선택을 적용하고 다음 decision boundary까지 진행 |
| `snapshot/restore` | 완전 world/knowledge/RNG를 테스트 및 운영 복구용으로 저장/복원 |
| `export_trace` | 요청된 실패/평가 사례만 의미 사건과 행동을 출력 |

반환에는 `terminated`, `truncated`, `operational_error`를 구분한다. 한쪽의 보충 요청을 양쪽 동시 normal 턴으로 만들지 않는다. 상대의 미제출 행동은 actor 관측으로 보내지 않는다.

### 4.3 규칙과 난수

동률, 명중, 피해 범위, 급소, 추가효과와 forced-switch 등은 고정한 reference 의미를 따른다. 기대 피해량으로 난수를 대체하지 않고 예측 승률로 조기 종료하지 않는다.

검증용 reference 경기에서는 전체 팀, seed, 실제 양측 행동과 필요 시 RNG draw stream을 직접 기록한다. 공개 리플레이로 원 RNG나 숨은 세트를 완벽하게 복원한다고 가정하지 않는다.

관측용 의미 사건은 엔진 내부의 숫자 이벤트로 유지하고, 보기 위한 긴 문자열은 필요할 때만 만든다. 로그를 끈다는 이유로 reveal 처리를 생략하지 않는다.

### 4.4 합법성과 정보 경계

합법 mask는 플레이어에게 허용된 요청 의미를 따른다. 상대의 비공개 특성을 읽어서 플레이어가 아직 알 수 없는 사실을 mask로 누설하지 않는다. reference의 `maybe` 상태나 숨은 요인으로 인한 재요청이 있다면 같은 프로토콜을 표현한다. 이런 규칙상 재요청과 잘못된 내부 command encoding을 구분한다.

각 슬롯의 후보를 엔진 규칙으로 직접 만든다. joint 후보의 완성 가능성, 같은 벤치 중복, side 자원 중복은 prefix mask로 제한한다. 후보마다 Showdown Battle을 생성하지 않는다.

### 4.5 처리량 목표

목표는 실제 정책을 연결한 상태에서 초당 수백 자연완결 경기, 최적화 목표 500~1,000경기/s다. 아직 V100 실측이 아니며 납품 보고에서는 다음을 구분한다.

- 환경 전이/s
- 양측 합계 policy decisions/s
- learner로 실제 저장한 decisions/s
- 자연완결 경기/s
- PPO epochs를 포함한 학습 row 처리량/s
- 수집, 업데이트, 저장과 평가를 모두 포함한 end-to-end 처리량

정확한 전체 엔진의 속도가 목표보다 낮으면 같은 실행 안에서 worker, 버퍼와 batching을 개선한다. 목표를 맞추려고 긴 경기를 잘라 승리로 세거나, 짧은 팀만 남기거나, GPU 이론 FLOPS를 실제 경기 수로 환산하지 않는다. 낮은 초기 속도를 이유로 무기한 데모에서 멈추지도 않는다.

## 5. 관측: 구조화된 개체 토큰

### 5.1 토큰 레이아웃

큰 상태 인코딩은 선수 의사결정마다 한 번만 한다. 자연어 tokenizer나 1,024/2,048 문맥 제한을 사용하지 않는다.

| 위치 | 개수 | 내용 |
|---|---:|---|
| global | 1 | 요청 종류, 현재 턴, 자신 side, 규정, 남은 자원, 선택해야 하는 슬롯 |
| field | 1 | 날씨, 지형, Trick Room 등 공개 전역 상태 |
| side | 2 | 양측 스크린, hazards, Tailwind 등 공개 진영 상태 |
| Pokémon | 12 | 자기 6, 상대 6의 선수 관점 상태 |
| moves | 48 | 각 포켓몬당 현재 기술 4개. 상대 미공개 슬롯은 unknown |
| recent events | 24 | 관측한 최근 의미 사건. 사용하지 않는 슬롯은 padding |
| 합계 | **88** | 구현 기본 padding 길이 **96** |

이 레이아웃은 128 이하 개체 토큰이라는 이전 설계를 구체화한 것이다. 각종 volatile를 “40개까지만” 같은 토큰 상한으로 버리지 않는다. full dex/effect 목록으로 만든 고정 effect ID와 flag/value/known 배열을 해당 개체의 특징 투영에 포함한다.

Transform, Mimic처럼 현재 기술 네 슬롯 밖의 공개 이력이 필요할 때는 지속 knowledge의 공개 기술 repertoire와 효과 메타데이터에 보존한다. 현재 네 기술과 영구 공개 지식을 같은 것으로 덮어쓰지 않는다.

### 5.2 필수 특징

| 종류 | 포함할 정보 |
|---|---|
| 자기 포켓몬 | 실제 현재/최대 HP, 실제 능력치와 boosts, 상태, 아이템/소비/교환, 특성/변화, 배분, 현재 기술/PP, 벤치/활성/기절/선출 |
| 상대 포켓몬 | 공개 종/형태/타입, 공개 HP 표시 또는 범위, 공개 boosts/status, 확인된 기술/아이템/특성, 공개 효과, 아직 확인되지 않은 필드의 known mask |
| 정적 도감 | 종족값, 타입, 기술 위력/명중/PP/우선도/대상/범주/효과 특징, 아이템과 특성 ID 및 효과 특징 |
| 효과 | 상태 flag, 정당하게 알려진 지속시간/스택/출처 개체, 공개되지 않은 값의 mask |
| 사건 | 사건 종류, 관측된 주체/대상, 기술/효과, 공개 변화량과 순서 |

정적 도감이 단일 특성을 유일하게 결정하는 등 공개 자료로 알 수 있는 값은 명시적 추론 규칙으로 표시할 수 있다. 팀 ID나 상대 실제 stat point를 lookup해 채우지는 않는다. 상대 슬롯은 preview/공개 등장 순서에 맞춰 추적하며, 내부 team array의 비공개 순서를 노출하지 않는다.

종류형 값은 integer embedding, 연속값은 정규화된 특징, boolean 및 known mask는 별도다. 원 엔진의 정확한 integer 상태와 dtype을 FP16으로 변환해 게임을 진행하지 않는다.

### 5.3 입력에 넣지 않는 것

인간의 정답 행동, 외부 usage/EV 빈도표, 별도의 예상 피해표, exact hidden opponent stats, 아직 제출되지 않은 상대 행동, 승패 및 미래 사건, 팀 ID와 출처 ID, 검색 결과를 넣지 않는다. 엔진 자체의 정확한 피해 계산은 별개다.

관측 레이아웃과 도감 vocab은 전체 inventory로 첫 학습 전에 생성한다. 이후 “새 팀을 위해 입력을 다시 디자인”하는 커리큘럼은 없다. 실기 조작을 붙일 때는 동일한 선수 관점 상태를 만드는 별도 어댑터를 사용한다.

## 6. PA3-8M 모델

### 6.1 네트워크

| 항목 | 값 |
|---|---|
| Encoder | 6-layer pre-LayerNorm, non-causal Transformer |
| d_model | 320 |
| Attention | 5 heads, head dimension 64 |
| FFN | 1,280, GELU |
| Dropout | 0.0, attention dropout 0.0 |
| 위치/역할 | 고정 token role 및 side/entity/slot 임베딩 |
| 정책 | 인코딩한 상태와 앞선 자기 선택을 받는 작은 conditional pointer scorer |
| prefix 처리 | GRUCell hidden 320, 최대 4단계의 요청 내부 prefix만 처리 |
| value | global representation → 256-unit MLP → scalar, action prefix 미입력 |
| 파라미터 | encoder 주요 행렬 약 7.37M, 임베딩/출력부 포함 목표 8~10M |
| 학습 대상 | 전체 actor 및 value 파라미터 |
| 초기화 | seed 20261006, 무작위. 입력 embedding normal std 0.02, 일반 linear Xavier, 정책 마지막 투영 small gain 0.01 |
| 구조 비교 | 하지 않음 |

88개 상태 토큰을 한 번 인코딩하고 각 행동 branch는 캐시된 representation을 사용한다. branch마다 encoder를 다시 돌리지 않는다. value는 같은 선수 관점 representation을 사용하지만 선택된 행동을 보지 않는다.

GRUCell은 경기 전체의 recurrent memory가 아니라 **한 요청 내부 선택 연결**이다. 경기 기억은 지속 knowledge와 recent events가 제공한다. 학습 시 임의의 과거 hidden state를 복구해야 하는 recurrent PPO는 첫 범위에 넣지 않는다.

### 6.2 행동 표현

normal의 슬롯별 atomic action에는 `kind, own_slot, move_slot, target_location, switch_destination, resource_flag`가 들어간다. 기술과 타겟을 묶은 하나의 슬롯 후보를 고르고, 두 번째 슬롯을 첫 번째 슬롯에 조건화한다.

| 요청 | 선택 순서 |
|---|---|
| preview | lead A → lead B → reserve 1 → reserve 2, 중복 없는 4개 |
| normal | A의 완전한 슬롯 행동 → A를 고려한 B의 완전한 슬롯 행동 |
| replacement | 필요한 첫 슬롯의 벤치 → 필요한 두 번째 슬롯의 벤치 |
| singleton/pass | 유일 합법 선택은 확률 1. 행동 없음은 PPO actor row로 위조하지 않음 |

normal atomic 후보 padding은 슬롯당 기본 64다. 포맷의 실제 move/target/resource 조합으로 최대치를 compile-time 계산한다. 64를 넘는 합법 후보가 확인되면 **본학습 이전** 배열 크기를 맞추며 후보를 자르지 않는다. encoder나 모델 개체 입력은 이 크기와 독립이다.

모든 candidate는 정수 action tuple로 보존하고, 점수는 해당 포켓몬/기술의 encoding과 target/entity/resource 특징으로 만든다. 임의 텍스트 option을 생성하지 않는다. preview와 replacement도 같은 scorer를 사용한다.

요청의 행동 확률은 조건부 확률의 곱이다. 최대 4개 branch log-prob를 더하되 request를 여러 보상 시점으로 늘리지 않는다. rollout과 learner가 같은 실제 선택 prefix와 후보 순서 및 mask를 사용한다.

### 6.3 전략을 섞는 방식

학습과 기본 정책 평가 모두 temperature 1.0 categorical sampling을 사용한다. opponent도 각자의 정해진 같은 방식으로 선택한다. greedy는 기본 평가에 섞지 않는다. 확률표에서 샘플링한다고 내시해가 인증되는 것은 아니지만 argmax만 실행해 학습한 혼합을 버리지 않는다.

## 7. 두 V100과 배치 실행 — MiniDC 정본 반영

### 7.1 유지하는 본학습 규모

| 설정 | 기본값 |
|---|---:|
| GPU | V100 PCIe 32GB ×2, Compute Capability 7.0 |
| CPU | E5-2673 v4 ×2, 40 physical cores/80 threads 기준 |
| System RAM | 약 64GB 기준, 실제 여유는 시작 시 확인 |
| Rust environment groups | 2 |
| Workers/group | 16 threads |
| Environments/group | 1,024 |
| 전체 동시 환경 | 2,048 |
| Inference max batch/GPU | 512 |
| Inference microbatch | 256 |
| GPU 메모리 운영 soft budget | GPU당 28 GiB |
| 전체 프로세스 host RAM soft budget | 48 GiB |
| Rollout buffer host budget | 24 GiB |
| Pinned staging | 전체 최대 1 GiB, 각 NUMA에 로컬 할당 |
| DDP global minibatch | 4,096 |
| Rank당 microbatch × accumulation | 256 × 8, rank 2개 |

이 규모, 모델 구조, 전체 학습팀과 1억 경기 목표는 기존 Full Spec 1.0과 같다. 이번 환경 반영을 새 소형 파일럿이나 모델 크기 실험으로 바꾸지 않는다. HBM 사용량을 CPU RAM 제한과 합치거나, 32GB 두 장을 하나의 64GB 장치처럼 취급하지 않는다.

### 7.2 실측 하드웨어 및 소프트웨어 pin

아래 값은 **사용자가 제공한 장비 조회 보고 [H1]의 값**이다. 이번 문서 작성자가 미니DC에 접속해 새로 측정한 것이 아니다. 별도 열이 있는 `Torch CUDA runtime`과 `System CUDA Toolkit`을 혼동하지 않는다.

| 구성 | 고정 또는 계승 값 | 적용 |
|---|---|---|
| OS / kernel | Ubuntu 24.04.5 LTS / 6.8.0-142-generic | 현재 관측값. 이를 설치/재부팅/교체하라는 명령 아님 |
| NVIDIA driver | **580.178.04**, 580-server | 기존 드라이버 유지, 자동 업그레이드/다운그레이드 금지 |
| System CUDA Toolkit | **12.8.2**, nvcc **V12.8.93** | `/usr/local/cuda-12.8` 유지, symlink 변경 금지 |
| Python | **3.12.3** | 기존 `python3`와 프로젝트 venv 사용, 3.11 설치 지시 삭제 |
| PyTorch | **2.14.0+cu126** | 로컬 검증 조합을 그대로 pin |
| Torch CUDA runtime | **12.6** | `torch.version.cuda` 기대값. 툴킷 12.8.2를 바꾸는 요구 아님 |
| pip / uv | 24.0 / uv 없음 | uv 불필요, pip 자동 업그레이드 없음 |
| gcc / cmake | 13.3.0 / 3.28.3 | 기존 도구 사용. 다른 도구의 시스템 설치는 별도 권한 필요 |
| GPU0 | PCI 05:00.0, NUMA0, **175W** | 주 rank 및 single-learner fallback 우선 |
| GPU1 | PCI 84:00.0, NUMA1, **150W** | 데이터 병렬 및 수집에 계속 사용. 열위/스로틀 가능성을 기록 |
| GPU interconnect | **SYS**, P2P 불가, 호스트 경유 약 4GB/s 보고 | DDP는 유지하되 직접 P2P/텐서 병렬을 요구하지 않음 |
| 패키지 hold | 55개 보고 | hold 유지. 개수가 향후 달라졌다는 이유로 자동 unhold/복구하지 않음 |
| Disk | 여유 89GB 보고 | 시점값일 뿐 예약 용량 아님. 데이터/버퍼/저장 여유를 실제 확인 |

**버전 선택의 의미:** 위 스택을 이 장비의 운영 정본으로 채택한다. 다른 모든 조합이 반드시 조용히 오동작한다는 일반 명제를 새로 증명한 것은 아니다. 원문이 경고한 CUDA 13/12.9 및 최신 패키지 경로를 피하고, 이미 검증된 조합을 바꿀 이유를 만들지 않는다.

### 7.3 CUDA Toolkit과 PyTorch runtime의 공존

`nvcc --version`의 12.8과 `torch.version.cuda`의 12.6은 서로 다른 계층이다. 일반 PyTorch wheel 실행은 배포된 CUDA 사용자 공간 의존성을 사용하고, 로컬 Toolkit은 소스/커스텀 CUDA extension을 컴파일할 때 주로 사용된다 [W8]. 따라서 **Torch를 cu128로 바꾸거나 시스템 Toolkit을 12.6으로 내려 숫자를 맞추지 않는다.**

드라이버 호환성과 GPU 아키텍처 지원도 다르다. `nvidia-smi`의 CUDA 표시가 13.x여도 그것은 해당 드라이버의 API 지원 표시일 수 있으며, V100용 Toolkit 13이나 sm_70 없는 wheel을 쓰라는 뜻이 아니다 [W9]. 실제 사용하는 wheel의 arch list에 `sm_70`이 있고 장치가 `(7, 0)`이어야 한다.

새 코드의 기본 GPU 경로는 표준 PyTorch 연산이다. Rust/PyO3 CPU 엔진 자체에 CUDA Toolkit 컴파일을 요구하지 않는다. 커스텀 CUDA 코드가 정말 필요해지는 경우에만 기존 12.8 compiler와 `sm_70`으로 빌드하고, wheel runtime과 extension ABI/링크 라이브러리의 호환을 그 구현 안에서 확인한다. 이를 위해 시스템 Toolkit을 바꾸지 않는다. `LD_LIBRARY_PATH`에 다른 CUDA 라이브러리를 앞세워 wheel을 강제 교체하지 않는다.

### 7.4 두 GPU의 실제 실행 배치

수집 동안 두 GPU는 같은 current policy와 필요한 동결 상대를 각자 완전하게 적재한다. 같은 policy ID별 요청을 묶고, GPU 사이에 매 행동의 activation/KV/텐서를 교환하지 않는다. PPO 수집 버전은 기존처럼 iteration 단위로 동결한다.

| Rank | GPU / NUMA | 보고된 CPU 범위 | 기본 Rust worker 배치 예 |
|---|---|---|---|
| 0 | GPU0 / NUMA0 | 0-19, 40-59 | 0-15의 16개, 실제 CPU map을 확인 |
| 1 | GPU1 / NUMA1 | 20-39, 60-79 | 20-35의 16개, 실제 CPU map을 확인 |

각 rank는 GPU 할당 및 큰 CPU 버퍼 생성 전에 해당 NUMA에 CPU와 메모리를 묶는다. 기존 `numactl`을 사용할 수 있으면 `--cpunodebind`와 `--membind`를 사용하고, 도구 설치를 위해 무단 `apt` 명령을 실행하지 않는다. 초기의 1,024환경씩 배정은 유지하며, 서로 다른 카드 속도 때문에 필요한 환경 배치 조정은 같은 frozen policy와 총 환경/팀 분포를 유지하는 범위에서 기록한다.

업데이트는 **NCCL DDP**를 유지한다. 장비 보고가 DDP 실사용을 확인했고 [H1], P2P가 없어도 NCCL에는 host/shared-memory 경로가 있다 [W10]. 프로젝트 프로세스에만 `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=0`을 설정한다. SHM을 꺼서 불필요한 socket 우회를 강제하거나 `/etc/nccl.conf`를 수정하지 않는다. 실제 transport 선택은 최초 통신 로그로 확인하며, 인터넷 문서만으로 이 장비의 NCCL 속도를 재인증했다고 하지 않는다.

Global minibatch 4,096은 `256 × 8 × 2`로 유지한다. 첫 7개 microbatch의 **forward와 backward 모두** `DDP.no_sync()` 범위에 넣고 마지막 microbatch에서만 gradient를 동기화한다 [W11]. 내부 bucket 수에 따라 collective 호출은 여러 번일 수 있으므로 'all-reduce API를 반드시 한 번만'이라는 조건으로 해석하지 않는다. Adam step, gradient clipping 및 scaler 처리 횟수는 global minibatch 기준으로 유지한다. 샘플 padding과 rank별 유효 row 가중치는 기존 계약을 지킨다.

GPU1 열위나 NCCL 문제로 fallback이 필요한 경우 **GPU0에서 같은 global batch 4,096을 microbatch 256 × 누적 16회**로 학습한다. 원래의 모델, optimizer/scaler state, 학습률 진행, 팀풀과 경험량을 보존하며 weight reset을 하지 않는다. GPU1은 가능한 동일 버전 경험 수집/동결 평가에 계속 사용한다. DDP와 single-learner를 정상적인 같은 실험의 실행 조정으로 기록하되, 부동소수점까지 bitwise 동일하다고 주장하지 않는다. 살아 있는 DDP group을 불완전하게 남기지 말고 자기 소유 rank만 안전한 경계에서 정리해 재개한다.

### 7.5 정밀도와 attention

- FP32 master weights, Adam state 및 주요 누적, FP16 autocast와 dynamic loss scaling을 유지한다.
- log-softmax, 확률/ratio, advantage/return과 loss 누적은 FP32다.
- 기본 attention은 **PyTorch SDPA math backend**, 또는 같은 수식의 명시적 `matmul → softmax → matmul`이다. dropout 0과 관측/마스크 의미를 보존한다.
- **eager 실행을 기본으로 하고 `torch.compile`/Inductor는 끈다.** `TORCHDYNAMO_DISABLE=1`을 프로젝트 프로세스에 적용한다.
- FlashAttention-2, BF16, FP8, TF32, INT8 Tensor Core, 새 Triton 커널을 요구하지 않는다. `sm70-attn`은 기존 llama.cpp 계열의 설치 자산이지 PA3에서 검증한 PyTorch attention backend가 아니므로 자동 재사용하지 않는다.
- xformers는 필수 의존성이 아니다. 새 wheel 설치로 torch가 교체되는 경로를 만들지 않는다. torch 패키지 의존성에 Triton이 존재하더라도 무단 삭제하지 않으며, 새 PA3 연산 경로에서 그것을 요구하지 않는다.
- CUDA Graph나 다른 가속 backend는 이번 pin 반영의 선행조건이 아니다. 현재 eager/SDPA로 전체 본학습을 진행하고, 추후 실제 병목과 동일 의미의 지원이 확인됐을 때만 한정 적용한다.

위는 실행 backend 선택이지 모델 크기, 팀 수, 전체 행동 범위나 학습량을 줄이는 변경이 아니다. V100이 해당 모델 연산을 지원한다는 것과 기존 500~1,000 경기/s 목표가 달성됐다는 것은 다르다. 실제 속도는 175/150W, host-staged 통신 및 적용 backend 상태를 함께 기록한다.

### 7.6 설치와 의존성 pin

이미 정확한 스택을 사용하는 프로젝트 venv가 있으면 재사용한다. 없으면 현재 `python3`로 프로젝트 안에 `.venv`를 만든다. 다른 서비스의 venv에 설치하지 않고, 예전 Full Spec의 2.6/cu124 조합으로 되돌리지 않는다.

필요한 경우의 프로젝트 내부 설치 예:

```bash
# 이 부분은 구현자가 승인된 로컬 개발 작업에서만 실행한다.
# 기존에 일치하는 venv가 있으면 생성/재설치하지 않는다.
python3 -m venv .venv
source .venv/bin/activate
source runtime/minidc.env
export PIP_CONSTRAINT="$PWD/runtime/constraints-minidc.txt"
python -m pip install -c "$PIP_CONSTRAINT" \
  'torch==2.14.0+cu126' --index-url https://download.pytorch.org/whl/cu126
# requirements에는 torch/torchvision/torchaudio 업그레이드 요구를 넣지 않는다.
# 나머지 실제 프로젝트 의존성에도 같은 constraints를 적용한다.
python -m pip install -c "$PIP_CONSTRAINT" -r requirements.txt
python -m pip check
```

`requirements.txt`는 구현에서 만드는 실제 의존성 파일이며 이 문서팩에 엔진 구현이 있다는 뜻은 아니다. Exact pin과 의존성이 충돌하면 충돌하는 요청을 고치고, `latest` 설치나 pin 해제로 우회하지 않는다. torchvision/torchaudio는 이 tensor 정책에 필요하지 않으므로 추가하지 않는다. 빌드 의존성은 runtime resolver와 별개일 수 있으므로 PyO3 빌드에 torch 재설치를 요구하는 격리 빌드를 만들지 않는다 [W12]. 필요한 build dependency를 같은 환경에서 고정한 후 통제된 빌드를 사용한다.

정확한 wheel이 다운로드되지 않으면 **기존 검증 venv, 로컬 wheel/cache 또는 그 정확한 아티팩트의 출처**부터 재사용한다. 존재하지 않는 다운로드 성공을 보고하거나 다른 버전을 자동 선택하지 않는다. 원문 보고가 설치 및 동작의 근거이며, 여기서 공개 index 확인은 새 장비 실행 검증을 대신하지 않는다.

설치가 필요했던 경우 최종 `pip freeze`, wheel/runtime, arch list와 Python을 `run.json`/해당 환경 lock에 한 번 기록한다. 미보고 cuDNN/NCCL/다른 패키지 버전을 문서에서 추측해 채우지 않는다. **pip 24.0도 기본 유지하며, 원문 설치 예시의 `pip install --upgrade pip`는 이 프로젝트의 필수 실행 명령으로 가져오지 않는다.**

### 7.7 서비스, 전력과 장기 실행 보호

드라이버, Toolkit, OS, kernel, apt hold, systemd와 전력 상한은 변경하지 않는다. `apt upgrade`, driver/toolkit 설치, 패키지 unhold, 재부팅 및 전력 승격은 본 문서의 권한이 아니다. 기존 보안 업데이트 정책을 끄지도 않는다. 향후 승인된 kernel 변경이 있었다면 장비 운영 문서에 따라 DKMS/driver 상태를 확인하며, 현재 계획에서 새 kernel 작업을 생성하지 않는다.

**`dsh-web`과 `llama-swap`은 원격 작업 채널이므로 stop/restart/kill하지 않는다. `pkill -f`와 `pgrep -f`도 사용하지 않는다.** 자기 작업 자원은 생성 시 보존한 process handle/PID로만 관리한다.

외부 API LLM을 쓰더라도 llama-swap은 마지막 로컬 모델을 VRAM에 남길 수 있다 [H1]. 시작 때 `nvidia-smi`의 점유자와 실제 RAM/VRAM/디스크 여유를 확인한다. 승인된 독점 GPU 실행에 필요하면 **문서화된 unload API만** 사용해 resident model을 내리고 서비스를 유지한다.

```bash
# GPU 실행 승인과 실제 점유 필요가 있는 경우에만 수동 적용. 자동 시작 스크립트는 호출하지 않음.
curl --fail --max-time 30 -X POST http://127.0.0.1:8080/api/models/unload
```

학습 동안 해당 로컬 LLM provider에 요청을 보내지 않는다. 다음 요청이 모델을 다시 적재할 수 있기 때문이다. unload 실패나 다른 사용자의 점유가 남으면 무단 kill 대신 resource 충돌을 보고한다. 이것은 모델/팀 규모를 축소하라는 의미가 아니다.

GPU0 **175W**, GPU1 **150W**를 그대로 유지한다. GPU1이 느리다고 200W로 올리거나 서비스 override를 수정하지 않는다. 일반 throttle이면 기록하고 같은 일의 배치를 조정한다. Xid/장치 소실 등 실제 하드웨어 오류는 자기 작업만 안전하게 중지/회수하고 자동 driver reset을 하지 않는다.

### 7.8 확인은 시작 절차 안에 통합한다

독립적인 장기 호환성 연구나 새 승인 단계는 만들지 않는다. 아래 값 확인과 최초 실제 batch/collective를 원래의 시작 절차에 넣는다.

```bash
nvidia-smi --query-gpu=index,pci.bus_id,name,driver_version,power.limit,memory.free --format=csv
/usr/local/cuda-12.8/bin/nvcc --version
python3 --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_arch_list()); print([torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())])"
```

기대 핵심: driver `580.178.04`, Toolkit release `12.8`/nvcc `V12.8.93`, torch `2.14.0+cu126`, runtime `12.6`, `(7,0)` 두 장과 `sm_70` 포함이다. `CUDA_VISIBLE_DEVICES`로 물리 번호를 바꿨다면 PCI 주소 기준 mapping을 기록한다. 단순 `import torch` 성공을 full RL 성능 인증으로 확대하지 않으며, 값이 일치하면 원래 전체 범위 구현/본학습을 계속한다. 이미 [H1]에서 수행한 모든 검사를 다시 보고서 프로젝트로 재생산하지 않는다.

## 8. 집중 RL의 전체 설정

### 8.1 경험 예산과 카운터

**1차 본학습 목표는 100,000,000 자연완결 경기**다. 한 경기의 양측 trajectory를 저장했다고 두 경기로 세지 않는다. 평가 경기, 실패, debug와 clone은 학습 경기 수에 포함하지 않는다. 이 숫자는 연구 목표이지 성능 보장이나 완료시간 약속이 아니다.

주 counter:
- `natural_training_matches`
- `learner_decisions`
- `all_policy_decisions`
- `environment_transitions`
- `optimizer_steps`
- `evaluation_matches`
- `operational_failures`

학습이 끝나는 조건은 1억 자연완결 경기 및 그 경험의 최종 PPO 처리가 완료되는 것이다. 마지막 cohort는 남은 자연완결 목표를 초과하지 않도록 새 reset 할당을 줄인다. 오류로 무효가 된 경기만 대체한다.

### 8.2 경험 수집과 old-policy 일관성

매 PPO iteration에서 current weights와 상대풀을 동결한다. 기본 2,048환경을 순환 실행하며 **완결된 learner 의사결정 131,072개 이상, 완결 경기 2,048개 이상**이 모일 때까지 새 경기를 생성한다. 두 조건을 만족하면 새 reset을 멈추고 실행 중인 경기들을 자연 종료시킨다. 그 후 업데이트한다.

마지막 iteration에서는 1억 목표까지 남은 경기 수가 이 최소량보다 적을 수 있다. 이때는 잔여 목표가 최소 cohort 조건보다 우선하며, 남은 경기 전량을 자연 종료하고 작은 마지막 minibatch를 padding/mask로 학습한 뒤 정상 종료한다. 최소량을 채우려고 1억 목표를 넘기거나 영원히 기다리지 않는다.

따라서 한 iteration의 row 수는 고정 131,072가 아니라 그 이상인 실제 전량이다. 완결을 위해 발생한 overshoot를 버리지 않는다. 긴 경기의 앞부분만 채택하고 뒤는 버리지 않는다. row 저장이 많으면 CPU의 packed buffer와 파일 spill을 사용하며 full neural embedding은 저장하지 않는다.

한 경기를 끝내기 전에 actor를 새 버전으로 교체하지 않는다. episode 동안 상대 checkpoint도 고정한다. 양측이 current일 때는 두 side를 각각 current learner 데이터로 저장할 수 있다. 과거 상대가 고른 행동은 current policy의 학습 row에 넣지 않는다.

사건이 없거나 wait인 것을 추가 배우기 표본으로 세지 않는다. 한 side의 결정 뒤, 그 side가 다시 결정하거나 경기 종료가 될 때까지의 결과를 연결한다. 자기 결정 간의 opponent decision 횟수로 보상을 중복하지 않는다.

자연 종료 보상을 요청 trajectory의 끝에 한 번 배치한다. reference가 정의한 무승부는 자연 종료다. 행정상 wall timeout이나 운영 오류는 무승부 0으로 바꾸지 않는다. rollout 수집만 중단한 경우 정책을 보존해 같은 버전으로 재개하거나 해당 미완료 episode를 제외하고 이유를 남긴다.

### 8.3 PPO 수치

| 항목 | 값 |
|---|---:|
| Optimizer | Adam |
| Learning rate peak | 0.0003 |
| Adam betas | (0.9, 0.999) |
| Adam epsilon | 0.00001 |
| Weight decay | 0 |
| PPO epochs | 4 |
| Global minibatch | 4,096 learner requests |
| Microbatch/rank | 256 |
| Gradient accumulation/rank | 8, 두 GPU 기준 |
| Clip epsilon | 0.2 |
| Gamma | 1.0 |
| GAE lambda | 0.95 |
| Value coefficient | 0.5 |
| Value loss | 0.5 × MSE(value, GAE return), actor objective와 별도 계수 적용 |
| Value clipping | 사용하지 않음 |
| Max gradient norm | 0.5 |
| Advantage normalization | iteration의 유효 actor rows 전체, std floor 1e-8 |
| Entropy coefficient | 0.01 |
| Uniform reverse-KL coefficient | 0.001 |
| Target approximate KL | 0.03 |
| Temperature | 1.0 |
| Model dropout | 0 |
| Seed | 20261006 |

PPO의 수집 후 여러 minibatch epochs라는 기본 알고리즘은 [W7]을 따른다. 이 표의 포켓몬용 수치는 설계자가 정한 시작 설정이다. 비공개 봇의 검증된 최적값이라고 하지 않는다.

### 8.4 Loss 정의

각 request의 실제 선택 prefix를 사용해:

`log_pi = sum_j log pi(a_j | obs, selected_prefix_j)`

`ratio = exp(log_pi_current - log_pi_behavior)`

`L_actor = -mean(min(ratio*A, clip(ratio, 0.8, 1.2)*A))`

`L_value = 0.5 * mean((V - return)^2)`

branch 후보 수 K가 2 이상인 경우의 entropy를 `log(K)`로 나누고 request 내부의 유효 branch 평균을 `H_norm`으로 정의한다. 같은 prefix에서 uniform 분포 U에 대한 `KL(U || pi)/log(K)`의 branch 평균을 `KL_uniform`으로 정의한다.

`L_total = L_actor + 0.5*L_value - 0.01*H_norm + 0.001*KL_uniform`

singleton은 actor의 결정 자유도가 없으므로 entropy/KL contribution 0이다. 모든 branch가 singleton인 요청은 actor loss에서 제외하고 필요한 value 학습에는 유지한다. uniform 정규화는 특정 기술이나 교체를 정답으로 지정하는 것이 아니며, 최종 정책을 균등하게 만들기 위한 quota도 아니다.

GAE는 같은 side의 결정 시퀀스를 따라 계산한다. gamma=1, lambda=0.95다. 유일 보상을 여러 슬롯과 target에 따로 지급하지 않는다. PPO ratio도 전체 경기 확률곱이 아니라 request 단위다.

epoch 후 `mean((ratio - 1) - log(ratio))`가 0.03을 넘으면 **현재 iteration의 남은 epochs만 생략하고 새 경험 수집으로 이동**한다. 본학습을 끝내거나 사용자에게 매번 승인을 요구하지 않는다. 이 조기 종료는 PPO 업데이트 폭 제어이지 경기의 승리 조기 판정이 아니다.

NaN/Inf gradient는 loss scaler가 업데이트를 skip하고 장부에 실제 step 성공 여부를 기록한다. 반복적 수치 오류는 해당 batch를 분리해 같은 작업 안에서 수정한다. 강제로 NaN을 0으로 바꿔 정상처럼 진행하지 않는다.

### 8.5 학습률 일정

진행률은 성공적으로 학습에 반영된 자연완결 경기 수로 계산한다. first 250,000 matches 동안 lr 1e-5에서 3e-4로 선형 warmup한다. 이후 100,000,000 matches까지 cosine으로 3e-5까지 내린다.

중간 checkpoint나 평가 milestone을 새 scheduler horizon으로 사용하지 않는다. 재개 시 optimizer, scaler와 scheduler progress를 그대로 읽는다. 1억 이후 추가 학습이 필요하면 그때 별도 연장 계획을 세우며, 이번에 1억 horizon을 몰래 늘리지는 않는다.

## 9. 팀과 상대 표집

매 reset에서 train 고유 팀을 양측 독립 균등 표집하고 learner seat는 50:50으로 배정한다. mirror match를 의도적으로 제외하지 않는다. source tab MA/MB/MC는 표집 strata로 쓰지 않는다. 모든 경기는 M-C 규칙이다.

팀 내 여섯 포켓몬과 기술 슬롯의 표시는 episode 시작 때 seed 기반으로 permutation해 순서 암기를 줄일 수 있다. 실제 canonical entity/명령 mapping을 같이 바꾼다. 도중에 slot identity를 임의로 바꾸지 않는다. 이 permutation은 처음부터 켜고 학습과 평가 모두 의미를 보존한다.

과거 상대풀은 최대 8개의 **동일 PA3-8M 구조**다. 현재 대 과거 비율은 50:50. 과거 풀이 비어 있으면 100% 현재다. 과거 상대 하나는 episode 시작에 선택하고 끝까지 유지한다.

기본 보관은 첫 100만 경기 snapshot, 현재 best_dev, 가장 최근 정기 snapshot 6개다. 같은 digest가 겹치면 중복을 세지 않고 그다음 최근 snapshot으로 채운다. pool 안의 모델은 균등 표집한다. 매 100만 자연완결 경기 milestone 이후 checkpoint를 등록하며, 첫 snapshot 전에도 전체 팀풀로 본학습을 계속한다.

과거 정책은 보존할 정답 교사가 아니다. 해당 행동을 따라 하는 KL teacher loss나 distillation을 넣지 않는다. 전략을 상대하고 학습하는 용도다.

## 10. 평가: 본학습 안의 주기적 관측과 마지막 결과

### 10.1 중간 평가

매 1,000,000 학습 경기 milestone 후 가까운 iteration 경계에서 dev 평가 **512경기**를 수행한다. milestone 하나마다 작업을 끝내거나 재승인을 받지 않는다.

- 256경기: latest 대 incumbent best_dev.
- 256경기: 고정 anchor 상대. random-initial과 first-1M을 균등 사용한다. first-1M 전에는 존재하는 anchor만 사용하고 이를 명시한다.
- dev 팀 그룹을 양측에 사용하고 좌석/팀 배정을 균형화한다.
- sampled policy, temperature 1.0, seed panel을 고정한다.
- 승리 1, 무승부 0.5, 패배 0의 score와 팀/상대별 분모를 보고한다.

latest가 incumbent과의 256경기에서 mean score 0.55 이상이고, 같은 anchor panel의 incumbent score보다 0.05 넘게 나쁘지 않으면 best_dev를 교체한다. 이것은 개발용 선택 규칙이지 통계적 우월성 증명이나 학습 계속 조건이 아니다. best_dev와 latest는 별도로 보존한다.

초기 낮은 승률, 교체율 또는 높은 value error 자체는 중단 조건이 아니다. 반복 milestone에서 개선이 전혀 없으면 summary에 한 가지 원인 후보를 기록하되, 자동으로 모델을 폐기하거나 BC로 전환하지 않는다.

### 10.2 Showdown 전이 평가

매 10,000,000 경기 milestone에서 256개의 동일한 dev matchup을 Rust와 reference Showdown 양쪽에서 실행한다. actor 입력과 행동 어댑터는 같은 논리 schema다. 규칙과 관측 차이, 불법 명령, 성향과 성적 차이를 확인한다.

승률이 비슷하다는 사실만으로 모든 엔진 메커니즘이 정확하다고 인증하지 않는다. 일치가 필요한 deterministic mechanics와 확률분포 검사는 개발 자동 회귀에서 따로 수행한다. Showdown 평가가 느리면 별도 CPU 작업으로 돌릴 수 있지만 같은 GPU를 무단 공유해 학습 속도를 왜곡하지 않는다.

### 10.3 최종 평가

최종 candidate는 latest와 best_dev다. 같은 digest면 하나만 평가한다.

- candidate당 Rust **2,048경기**, 두 후보가 다르면 총 4,096.
- 상대는 random-initial, 1M, 10M, 50M의 동결 PA3-8M, 상대당 512경기.
- final 팀 그룹을 사용하며 팀쌍/좌석/seed block을 공통으로 적용한다.
- 후보당 그중 512개 조건을 reference Showdown에서도 평가한다. 두 후보면 총 1,024 Showdown 경기.
- raw score, 규정 고정, 팀별 결과, paired block 차이와 group 단위 불확실성을 보고한다. 같은 경기의 두 side를 독립 경기로 부풀리지 않는다.
- final 결과를 보고 학습을 다시 조절하면 그 팀은 더 이상 미사용 final이 아니다.

이 상대들은 외부 최고 수준 benchmark를 대신하지 않는다. 평가가 완료돼도 “mikumiku보다 강함”이나 특정 ladder Elo를 자동 주장하지 않는다. 실제 외부 대전은 서비스 조건과 사용자 승인에 따른 별도 작업이다. 본 문서에는 공개 ladder 자동 실행을 넣지 않는다.

## 11. 본학습 진행과 회복

### 11.1 정상 운영

실행자는 한 번 합의한 범위 안에서 구현, 자동 검사, 디버깅, 본학습과 평가까지 진행한다. 소형 batch 성공이나 문서 제출은 작업 종료점이 아니다. 처음부터 전체 팀풀, 전체 action과 전체 입력을 사용한다.

하드웨어의 batch/worker 튜닝은 같은 모델의 처리량 조정이다. 모델 종류 sweep, 별도 소형팀 커리큘럼, phase 이름 증식은 하지 않는다. 처리량 튜닝으로 microbatch를 바꿔도 global learning minibatch와 effective loss는 유지한다.

### 11.2 저장

최신 full resume checkpoint는 **10분마다 또는 milestone 중 먼저 오는 시점의 안전한 iteration 경계**에 저장한다. latest 두 세대와 best_dev를 유지한다. 매 100만 경기의 actor-only snapshot은 과거 상대 및 평가 참조용으로 저장한다. full optimizer checkpoint를 100개 모두 유지할 필요는 없다.

resume 상태는 model/optimizer/scaler, committed counters, scheduler 진행, policy pool, 팀/표집 RNG, 엔진 version과 다음 seed cursor를 포함한다. 기본 iteration 종료는 자연 drain 상태이므로 체크포인트에 수천 live Battle을 중복 직렬화할 필요가 없다. 운영 중단 시 미완료 cohort를 다시 수집할 수 있으며 마지막 유효 학습점은 보존한다.

atomic temporary-write → rename을 사용한다. 저장 hash는 파일 연결/재개/회수 때 확인하고 hot loop에서 반복 계산하지 않는다.

### 11.3 오류 대응

| 오류 | 대응 |
|---|---|
| worker 종료, 일시 파일 오류 | 해당 작업을 동일 run에서 복구. 유효 checkpoint부터 계속 |
| 일부 effect 오류 | 영향되는 episode 및 경험을 격리하고 구현 수정. 원 목표 팀풀 지원을 메움 |
| 공통 hidden leak, reward 반전, PPO 행동확률 연결 오류 | 영향 학습을 멈추고 마지막 유효 상태 및 영향 범위를 확인한 뒤 수정 |
| 지원 밖 팀/효과 | 시작 전에 탐지해 구현 항목으로 해결. 조용히 제거 후 전량이라고 보고하지 않음 |
| 낮은 초기 승률 | 진행 기록. 즉시 중단/재설계/재승인 요청하지 않음 |
| 목표 속도 미달 | 원인별 프로파일과 같은 의미의 최적화. 성능을 부풀리지 않음 |

모델이 잘못된 데이터를 이미 학습한 사실이 입증됐으면 가중치까지 정상인 척 보존하지 않는다. last-valid가 있으면 그것부터 이어가고, 불확실한 영향은 명시한다. 반대로 파일명/저장장치 변경만으로 전체 scratch 재학습을 요구하지 않는다.

## 12. 검사는 구현의 일부로 자동화

별도 인증 단계를 계속 추가하지 않는다. 기본 검사는 세 묶음이다.

1. **규칙 및 CTS 관측:** 전체 효과 closure의 단위/상호작용, reference 생성 상태 및 행동과의 대조.
2. **학습 연결:** 실제 sampled request 확률 재계산, 동일 가중치 ratio≈1, 유효 gradient와 checkpoint 재개.
3. **실전 평가:** dev 추세와 최종 reference 전이.

검사에서 필요한 fixture와 probe를 실제 학습 경기 수에 섞지 않는다. 검사를 시간/메모리로 관리하고 clone 하나를 자연완결 한 경기라고 세지 않는다. 원 자료 수십만 건을 매 변경마다 감사하지 않는다.

다만 “정합성 검사는 하지 않는다”를 보상 반전이나 상대 hidden 입력을 허용한다는 뜻으로 해석하지 않는다. 이 오류들은 대량 학습의 성공과 직접 관련된다. 관련 자동검사와 제한된 실패 trace는 유지한다.

## 13. 파일 구조와 반환

```text
PokeAgent3.0/
  engine/          Rust 환경 및 reference adapter
  agent/           PA3-8M, PPO, collector와 evaluator
  data/            팀 원문, normalized teams, split, compiled dex
  configs/         train.yaml 및 실행 lock
  tests/           핵심 자동 회귀
  runs/<run_id>/
    run.json
    metrics.jsonl
    checkpoints/
    summary.md
```

`run.json`은 실제 N_train/N_dev/N_final, 코드와 규정 pin, 모델 수, 학습 목표/실제량, parent와 설정을 담는다. `metrics.jsonl`은 PPO, action phase, score, games/s와 오류를 시간 순서로 기록한다. `summary.md` 첫 화면에는 **자연완결 학습 경기 수, 사용 팀 수, 실제 처리량, latest/best 성적과 다음 상태**를 적는다. 검사 개수가 첫 성과가 아니다.

전체 raw rollout을 1억 경기 내내 영구 저장하지 않는다. 기본 trace는 실패 전부와 학습 경기 1/10,000 표본, 평가에서 선택한 재현 표본이다. 모델 가중치나 GPU가 빠진 문서만 반환하고 본학습을 했다고 하지 않는다.

## 14. 세 단계와 각 납품물

| 단계 | 포함 작업 | 사용자가 받는 것 |
|---|---|---|
| 1. 전체 범위의 고속 엔진 제작 | 전량 팀 수집, full effect closure, 구조화 관측, PA3-8M 연결, CPU/GPU 배치 실행과 내부 자동 검사 | 실제 전체 팀 수와 지원 범위, 실행 가능한 학습 시스템, end-to-end 처리량 |
| 2. 집중 RL | 같은 모델을 전체 train 팀과 current/history 상대로 1억 자연완결 경기 목표까지 학습, 자동 저장과 중간 평가 | 재개 가능한 latest/best, 경험량과 성장 곡선 |
| 3. 모델 평가 | final 팀 및 reference Showdown에서 고정 후보 평가 | 정책만의 강도, 팀 일반화, engine transfer 한계 |

단계 1~2 사이에서 “한 번 업데이트 성공”을 최종 반환으로 삼지 않는다. 내부 점검이 정상이면 같은 config로 본학습을 수행한다. 목표와 실행 자원이 별도 요청에서 승인되어 있는 동안 사소한 수리마다 사용자에게 재승인을 요청하지 않는다.

원하는 수준의 정책을 얻었다면 후속으로 동시수 검색과 적응형 계산 예산, 복기 및 실기 하네스를 검토한다. 이번 본학습에 그것들을 선행조건으로 붙이지 않는다.

## 15. 구현자가 임의로 바꾸지 않을 것

- 단일 PA3-8M을 여러 구조 sweep으로 바꾸지 않는다.
- 상세 배분 없는 팀을 만들어 숫자를 맞추지 않는다.
- 첫 train 범위를 소수 팀으로 축소해 끝내지 않는다.
- 사람 행동 BC, Laya/P1/AUX, 외부 LLM의 행동 추천을 넣지 않는다.
- 손실을 줄이려고 타겟/교체를 별도 고정 규칙으로 바꾸지 않는다.
- 엔진의 난수를 기대값으로 바꾸거나 predicted win으로 종료하지 않는다.
- 검증용 Showdown clone을 학습의 매 후보 경로에 넣지 않는다.
- 기존 계정의 충전, 클라우드 임대, 공개 ladder 또는 원격 업로드를 문서 작성만으로 수행하지 않는다.

팀 수, vocab 수, 정확한 total parameters, 실제 처리량은 실행이 정하는 값이다. 학습률 등 본 문서가 정한 수치와 혼동하지 않는다. 수집 및 compile 결과로 결정되는 값은 첫 학습 전에 `run.json`에 한 번 넣고, 이후 학습 중 몰래 바꾸지 않는다.

## 16. 출처와 검증 범위

### 사용자와 합의한 설계

V100 두 장, PA3-8M 한 종류, scratch self-play, 전체 팀 범위를 첫 시작에 포함, Rust 고속 엔진 → 집중 RL → 평가라는 세 단계, 상세 EV 팀의 VGCPastes 수집은 본 대화의 요구다. 과거 작은 단계 위주의 `POKEAGENT_3_0_PLAN.md`는 위 §1에 따라 대체한다.

### 외부 확인 자료

- **[W1]** MunchStats 팀 검색: https://www.munchstats.com/teams/ . VGCPastes 출처와 EV 필터 확인. 표시 팀 수는 변할 수 있으며 상세 배분 적격 수가 아니다.
- **[W2]** MunchStats 공개 수집 구현, 확인 commit `f481dfd4607af5c93518e9d8dd7236104899bd6a`: https://github.com/PizzaTimeJoshua/munchstats/blob/f481dfd4607af5c93518e9d8dd7236104899bd6a/vgcpastes.py . 공개 spreadsheet, 탭, paste 링크, `has_evs`, snapshot 경로 확인.
- **[W3]** 공식 Showdown 팀 문서: https://github.com/smogon/pokemon-showdown/blob/14546894d86f9589ac11130c510bbe73b6968665/sim/TEAMS.md . 팀 import/export, 기본값, Champions `EVS` 필드의 Stat Points 의미 확인.
- **[W4]** 공식 Champions 구현: https://github.com/smogon/pokemon-showdown/blob/14546894d86f9589ac11130c510bbe73b6968665/data/mods/champions/scripts.ts 및 같은 commit의 `config/formats.ts`, `data/mods/champions/rulesets.ts` . 포맷/정적 코드 조사. 이번에 실제 엔진을 실행한 것은 아님.
- **[W5]** 최초 스펙의 일반 wheel 선택 근거는 이번 미니DC 환경 선택에서 사용하지 않는다. 실제 장비의 Python/torch/Toolkit pin은 아래 [H1]을 우선한다.
- **[W6]** PyTorch 개발 공지: https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432 . CUDA 13 및 구형 GPU binary 지원 변경 주의.
- **[W7]** Schulman et al., PPO: https://arxiv.org/abs/1707.06347 . 경험 수집과 minibatch 기반 surrogate optimization의 근거. 위의 모든 하이퍼파라미터나 Pokemon 성능을 보증하는 논문이 아님.
- **[R1]** 사용자 제공 `64-mikumiku37-analysis-2026-10-04(1).md` . 소형 모델, BC 없는 self-play와 과거 정책 상대, 정책 학습 후 검색이라는 참고점. 비공개 구현과 기록을 이번 문서에서 독립 재현하지 않음.

- **[H1]** 사용자 제공 정본: `reference/73-v100-driver-cuda-constraints.md` (2026-10-06), SHA-256 `9860b5cca4de76a6c6487c5ec4ab0175fd790ca4c26a6589e0bb4de3bbe680a8`. 이 문서의 장비 수치는 해당 보고에서 계승한다. 원문은 편집 없이 보존했다.
- **[W8]** PyTorch maintainer 설명: https://discuss.pytorch.org/t/assertionerror-torch-not-compiled-with-cuda-enabled/124287/6 . wheel CUDA runtime과 local Toolkit 역할 구분.
- **[W9]** NVIDIA CUDA 호환성 문서: https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html . driver/runtime 호환과 아키텍처 지원을 분리해서 해석한다.
- **[W10]** NVIDIA NCCL 환경변수: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html . `NCCL_P2P_DISABLE`, SHM transport 설명. 실제 wheel의 NCCL 버전은 구현 환경에서 기록한다.
- **[W11]** PyTorch DDP `no_sync`: https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html . forward와 backward를 함께 포함하는 누적 동기 제어의 근거. main의 모든 새 옵션을 2.14 실행 계약으로 가져오지 않는다.
- **[W12]** pip constraints: https://pip.pypa.io/en/latest/user_guide/#constraints-files . pin은 의존성 충돌을 제한하는 것이지 해결된 의존성 전체나 빌드 격리 환경을 자동으로 보증하지 않는다. pip 버전은 [H1]의 기존 24.0 유지.

**이번 환경 개정에서 실제로 한 일:** 이전 스펙과 YAML 및 제약 시트 대조, 공식 호환/통신 자료 확인, 환경과 보호 규칙 갱신, 본학습 설정 불변 비교 및 패키지 파일 검증.  
**이번 환경 개정에서 하지 않은 일:** 미니DC 접속, driver/Toolkit/Python/pip 설치 또는 변경, 서비스 unload/중지, GPU 연산 및 DDP 실측, 팀 scraping, Rust/PyTorch 모델 구현, 학습, 원격 저장소 변경.
