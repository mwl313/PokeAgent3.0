# PokeAgent 3.0 — Sample-Efficient Self-Play RL: Evidence Review and Experimental Roadmap

**날짜:** 2026-10-08, KST  
**상태:** 연구 및 실험 설계, 학습 실행 전  
**프로젝트 참조:** [`mwl313/PokeAgent3.0`](https://github.com/mwl313/PokeAgent3.0), `training-pool/minidc-userteam-benchmark`의 확인 가능한 HEAD `9cbaa1fbe42264eb124f9ea3ee74dc7af13132af`  
**관계:** v1.1 CPU+GPU 극한 처리량 최적화 계획의 **보완 연구**. 모델/규칙/데이터셋/본학습 설정 자동 변경 금지.

## 핵심 판정

**P0 — 반드시 먼저:** 실제 모델이 강해지는지 평가하는 **고정 상대 정책 및 매치업 평가 패널, 여러 학습 시드, 불확실성 측정, 비용 장부**를 구축한다. 이는 새로운 RL 알고리즘보다 먼저 수행해야 하는 측정 인프라다.

**P1 — 유망, 상대적으로 저위험:** 현행 50% current/50% historical 정책을 보존한 실험에서 **early history snapshots와 균등 historical selection 대비 혼합 PFSP**를 연구한다. AlphaStar의 실험 근거가 있으나 Pokémon M-C에서 유효하다는 보장은 없다. 대립되는 역사 상대들 간 상대전적 표본이 충분하지 않으면 PFSP를 강제로 켜지 않는다.

**P2 — PPO 재현성과 작은 ablation:** 기존 4 PPO epochs, critic loss, GAE, entropy, KL, warmup scheduler, advantage 및 bootstrap에 대한 정확성 감사 후 **작은 범위의 분리 실험**. 구현 세부가 결과에 중요하다는 ICLR 2021 대규모 연구가 근거다. 무작정 epoch를 늘리지는 않는다.

**P3 — 더 비싼 연구 후보:** **PPG 및 PPG Reloaded식 정책 정규화/가치학습 분리**. ICML 2021은 Procgen에서 PPO보다 샘플 효율 향상을 보고했지만, ICML 2023은 높은 value reuse 자체가 본질적 원인이 아니라고 보고했다. PPG는 단순한 추가 critic epochs가 아니다.

**P4 — 별도 사양 변경이 필요한 탐색:** potential-based reward shaping, 반복 샘플 replay, PSRO 및 추가 탐색 기법. 이론적인 policy invariance와 실험적인 실력 향상을 구분하고, 기존 terminal-only win/loss reward와 on-policy PPO 계약을 임의 변경하지 않는다.

**확정되지 않은 것:** PA3-8M에 대한 어떤 방법의 승률 향상도 아직 검증되지 않았다. 모든 학술 결과는 서로 다른 규칙, 관측, 모델, 보상, 상대 정책, 계산 예산에서 나온 것이다. 본 문서는 성공 확률이나 절약 경기 수를 임의 추정하지 않는다.

---

## 1. PokeAgent의 현재 계약: 실제 파일 확인

| 위치 | 확인된 항목 | 실험에 미치는 영향 |
|---|---|---|
| `configs/train.yaml`, `scope` | Scratch self-play, 인간 BC 금지, 모델 sweep 금지, small team curriculum 금지 | 교사 정책이나 pre-trained weights를 기본 경로에 추가하지 않는다 |
| `configs/train.yaml`, `teams` | `mb-mc-v3-userteam-all-train`, 1,137 unique, dev 0, final 0, 균등 독립 팀 표집 | training team weighting은 기존 사용자 승인 계약 변경에 해당 |
| `configs/train.yaml`, `rollout` | 현재 PPO 반복마다 learner decisions >=131,072, 자연완결 matches >=2,048, 누적 후 자연종료, current 정책 및 episode 상대 동결 | PPO replay/특수 초기화/간편 경기 조기종료 금지 |
| `configs/train.yaml`, `training` | Adam 3e-4 peak, warmup 250k matches, 4 epochs, 4,096 global minibatch, clip=0.2, gamma=1, GAE lambda=.95, advantage norm, entropy 0.01, uniform-KL 0.001 | 변경은 독립된 실험 분기로만 수행 |
| `configs/train.yaml`, `opponents` | current 50%, historical 50%, 최대 8개, 과거 정책 균등 표집, 매 100만 경기부터 기록 | **1백만 경기 이전에는 historical 정책이 없어 100% current self-play** |
| `configs/train.yaml`, `evaluation` | 매 100만 경기 512게임, incumbent/anchor 256씩, train team groups 사용 | 100만 미만 sample efficiency 실험에는 조기 평가 프로토콜 별도로 필요 |
| `docs/PA3_LEARNER_STATUS.md` 및 smoke 보고 | 실제 PPO smoke는 기능 테스트이며 **강해졌다는 실험 증거가 아님** | PPO loss/KL/logprob ≠ policy strength |
| `engine/TRAINING_READINESS.md` | 10/16 PASS, 6 FAIL | 본학습/강한 정책 검증으로 혼동 금지; 제한된 연구만 별도 승인 시 진행 |

**중요:** 원래 Full Spec 1.1은 PPO와 현재 팀/상대 표집의 *설계값*을 정할 뿐, VGC M-C 환경에서 최적의 샘플 효율이라고 실측 검증하지 않는다. 최적화 이후 커밋되지 않은 miniDC 개선사항은 이 문서의 코드 검증 범위에 포함되지 않는다.

---

## 2. 핵심 1차 연구와 PokeAgent에 대한 해석

### 2.1 상대풀, PFSP, 적응형 self-play

- **AlphaStar, Nature 2019:** PFSP 기반 league는 FSP 대비 population performance, exploitability 및 최종 리그 내 성능에서 우수한 ablation 결과를 발표했다. [원논문](https://www.nature.com/articles/s41586-019-1724-z).
- **중요한 전이 제한:** AlphaStar는 StarCraft II에 인간 리플레이 기반 supervised pretraining과 풍부한 exploiter population을 포함했다. **scratch 1개 PA3 모델 + 8개 과거 정책**에 동일한 효율이 나타난다는 연구는 아니다. [DeepMind 설명](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/).
- **OpenAI Five 기술 보고:** 학습 경기의 80%는 최신 정책 상대, 20%는 과거 상대 정책이며, 과거 상대 정책의 표집 확률을 적응적으로 바꿨다. 이는 **실제 사용 사례**이나 균등 vs 적응형 표집만을 통제한 Pokémon 실험이 아니다. [원보고서](https://cdn.openai.com/dota-2.pdf).
- **비추이적 게임:** 특정 AI가 A를 이긴다고 B를 이긴다는 보장이 없다. 상대 payoff matrix와 정책 집합이 단일 Elo보다 중요한 이유다. [Balduzzi et al., ICML 2019](https://proceedings.mlr.press/v97/balduzzi19a.html); [AlphaStar Nature 2019](https://www.nature.com/articles/s41586-019-1724-z).

**권고 실험:** historical 비중 50%와 full team uniform은 고정. 과거 정책 선택만 `uniform`, `PFSP-match`, `PFSP-hard`, `mixed-uniform+PFSP` 간 독립 비교. `w_i`는 현재 정책이 상대 snapshot i에 대해 내는 실측 좌석/매치업 균형 승률이다. 예시 weight: `f_match(w)=w(1-w)`; `f_hard(w)=(1-w)^p`. 이는 AlphaStar 계열에서 사용한 family이며, **PokeAgent에 최적이라는 증거는 없다**. 추정치가 매우 불확실하거나 historical 후보가 없으면 uniform fallback. 노이즈가 있는 상태에서 hard-only는 특정 상대에 과적합될 위험이 있다.

**실무 선결:** 현 설정은 첫 snapshot이 100만 경기이므로, 10만/25만/50만 경기 샘플 효율 연구에서는 PFSP 자체가 거의 작동하지 않는다. 별도 실험에서 **초기 history snapshot 간격**을 짧게 하는 정책을 비교해야 한다. checkpoint마다 충분한 상대전적을 기록하기 전에는 PFSP 점수의 의미가 약하다.

### 2.2 PPO의 올바른 baseline 및 작은 ablation

- **PPO 원논문 (2017)**은 같은 on-policy batch를 여러 minibatch epochs 업데이트하도록 설계했다. 이미 4 epochs를 사용 중이다. [Schulman et al.](https://arxiv.org/abs/1707.06347).
- **ICLR 2021의 대규모 empirical study:** 50개 넘는 구현 결정과 25만 개가 넘는 에이전트 실험에서 on-policy RL 세부 구현 선택이 결과를 크게 바꿀 수 있음을 확인했다. 단, MuJoCo류 연속 제어 환경을 근거로 포켓몬에 특정 하이퍼파라미터의 우월함을 확정하면 안 된다. [Andrychowicz et al.](https://mlanthology.org/iclr/2021/andrychowicz2021iclr-matters/).

**권고:** 현 4 epochs를 강제로 8/16으로 올리지 않는다. 먼저 (1) 모든 actor sample이 최종 learner 업데이트까지 사용됐는지, (2) clipped ratio와 KL early stop으로 버려지는 업데이트량, (3) critic fit과 explained variance 및 terminal value calibration, (4) policy entropy/실제 sampling diversity, (5) outcome attribution, (6) warmup learning-rate clock이 real committed matches를 따르는지를 측정한다. 그 후 epochs/value weight/GAE lambda/scheduler 선택을 **변수 하나씩** 비교한다. 비슷한 결과가 나오면 계산량이 낮은 설정을 선호한다.

### 2.3 PPG의 검증된 장점과 후속 연구의 수정

- **PPG, ICML 2021:** policy와 value 학습을 분리하고 추가적인 auxiliary phase를 사용해 Procgen에서 PPO 대비 높은 sample efficiency 보고. [Cobbe et al.](https://proceedings.mlr.press/v139/cobbe21a.html).
- **PPG Reloaded, ICML 2023:** 16개 Procgen 게임의 실험에서 높은 가치함수 경험 재사용 그 자체보다 **policy regularization과 auxiliary phase 데이터 다양성**이 주요 요인임을 제시. 유사 성능을 PPO에 가까운 계산 비용으로 달성할 수 있음을 보고. [Wang et al.](https://proceedings.mlr.press/v202/wang23aw.html).

**중요한 수정:** `value_epochs`만 추가하는 변경을 'PPG 구현'이라고 하지 않는다. 정식 PPG에는 policy/distillation 구간, auxiliary data, policy-preserving KL/clone loss, distinct head 및 수치 검증이 필요하다. 해당 KL은 인간 행동 모방 BC와 구분되지만, 현 프로젝트의 `no_teacher_loss`/model shape 변경 여부를 먼저 검토해야 한다. 이 연구는 강력한 후보이지만 **PFSP 기초 실험보다 구현/학습 비용과 계약 변경 위험이 크다**.

**단계적 구현안:** ① frozen network에서 critic fit 진단, ② policy frozen 또는 carefully isolated critic-only update 실험 (진정한 PPG라고 부르지 않음), ③ 필요 시 PPG-inspired policy auxiliary/distillation branch 구현, ④ PPO와 동일 자연완결 matches 그리고 동일 wall time으로 비교.

### 2.4 Potential-based reward shaping

- **Ng, Harada, Russell, ICML 1999:** `F(s,s') = gamma * Phi(s') - Phi(s)` 형태의 보상 가산이 일정 조건에서 최적 정책을 보존함을 증명. [원논문](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf).
- 본 프로젝트는 `gamma=1`, terminal-only win/loss reward다. **episodic terminal의 Phi를 일정값(보통 0)으로 설정하고** trajectory telescope 성질을 지키면 승리 목표를 변경하지 않으면서 중간 credit signal을 제공할 수 있는 방향을 이론상 검토할 수 있다.
- **주의:** 이론상의 policy invariance는 *실제 finite-network PPO에서 학습 속도가 개선된다는 증명은 아니다*. 관측별 정보 제약을 위반하는 true hidden state 기반 shaping 금지, 체력 피해량당 임의의 보상 지급 금지, full-scope rule correctness 필수.
- **우선순위 낮음:** 완전한 evaluation baseline과 value diagnostics 없이 shaping부터 구현하면 자기기만적인 reward hacking을 감지하기 어렵다.

### 2.5 Prioritized Experience Replay, AlphaStar league 복제, PSRO

- **PER, ICLR 2016**은 DQN의 Atari benchmark에서 개선을 보였다. [Schaul et al.](https://mlanthology.org/iclr/2016/schaul2016iclr-prioritized/).
- PPO는 on-policy old-policy ratio를 사용한다. 긴 기간 지난 actor transitions를 **그대로 재사용**하면 정책 변화, value targets, KL, 중요도 가중치 문제를 만든다. PER의 Atari 결과만으로 PA3 PPO replay 개선을 주장할 수 없다.
- **PSRO 및 full league**는 비추이적 정책 집단에 대한 강력한 방법이지만, 여러 best response와 population 평가에 추가 환경 및 GPU 비용이 든다. 제한된 V100 두 장, 작은 경험 예산에는 먼저 8개 history pool과 PFSP를 평가하는 편이 타당한 engineering choice다. [Balduzzi et al., ICML 2019](https://proceedings.mlr.press/v97/balduzzi19a.html), [PSRO Survey](https://arxiv.org/abs/2403.02227).

---

## 3. 가장 중요한 선결: 평가가 정직해야 한다

### 3.1 self-play 승률 50%로는 학습 진전 측정 불가

현재 정책과 동일 정책이 양측에서 대결하면 대칭 조건에서는 대략 50%로 수렴한다. 이 값은 강함의 지표가 아니다. PPO loss 하락, clip fraction, entropy, small KL, sampled/recomputed logprob 일치는 **엔지니어링/수치 정확성의 증거**이지 정책이 강해졌다는 증거가 아니다.

기본 평가구성은 다음과 같다.

1. 같은 고정된 팀 매치업과 seed, **양측 seat 교환**의 paired games.
2. 같은 기준 opponent anchors: random-initial, 초기 checkpoint, fixed 1M 이후 anchor (존재하면), 이전 best 및 최근 policy.
3. policy population cross-play payoff matrix. 어떤 상대에게는 강하고 다른 상대에게는 약한 주기/취약성 확인.
4. **동일 학습 경기 수**에서 성능 비교: 예시 100k, 250k, 500k, 1M. **동일 소요 시간**에서도 별도 비교한다. 예시 숫자는 실험 설계 후보일 뿐 통계적으로 최적화된 budget은 아니다.
5. 가능한 한 여러 독립 training seeds, 독립 eval stochastic seeds. 소수 run에서 불확실성을 무시하면 잘못된 결론이 나오는 문제가 NeurIPS 2021에서 강조되었다. [Agarwal et al.](https://papers.nips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html).
6. **선택/보고 편향 분리:** tuning 도중 확인하는 matchups/seed panel과 최종 보고 panel을 서로 구별. 이는 팀셋을 holdout으로 빼는 것과 다르며, 모두 학습 팀 데이터셋 출처 안에서 고를 수 있다.
7. 상대별 wins/losses/draws와 team group, seat, mega/resource composition 별 성적을 기록. 단일 Elo 점수만으로 개선 판정 금지. [Balduzzi et al.](https://proceedings.mlr.press/v97/balduzzi19a.html).

**스코프:** 현 사용자의 명시적 설정은 모든 1,137개 고유 팀을 train에 활용하는 것이다. 별도 승인 없이 학습팀을 빼거나 신규 팀을 훈련 데이터로 편입하지 않는다. 평가에서 기존 훈련팀의 **독립 시드/상대 정책**을 사용하는 것은 가능하지만 **처음 보는 팀으로의 일반화**를 입증하지는 못한다. 외부 독립 팀 평가를 하려면 별도 승인된 evaluation-only 출처와 프로토콜이 필요하다.

### 3.2 지표 정의

- `sample_efficiency`: 고정 학습 자연완결 match count에서 동일 frozen evaluation panel의 상대별 win score, population payoff matrix, 성과 곡선 면적.
- `wall_clock_efficiency`: 수집 + 4 PPO epochs + DDP + checkpoint/metrics + 평가의 end-to-end 시간 대비 성능. 상대 표집, PPG auxiliary 계산비도 포함.
- `risk`: 상대별 최악 성적, policy forgetting, training collapse, hidden-information leaks, illegal action rate, loss/approx-KL/entropy instability.
- **이중 계수 금지:** 같은 경기를 양측 current trajectories에 기록해도 natural training match는 여전히 1이다. snapshots 대전에서 과거 상대 decision은 learner data가 아니다.

---

## 4. 비용을 낭비하지 않는 단계형 검증 프로토콜 (제안, 실행 전 승인 필요)

### Track E0 — 평가/진단만, 학습 알고리즘 변화 없음

- current vs random-initial, fixed snapshot, incumbent, cross-play을 실행할 **same seed/seat/team paired evaluator** 생성.
- baseline seed 개수, CI, team-group payoff, seed panel hash, RNG/version 상태 기록.
- 모델 forward/value prediction, terminal returns, actor KL, critic explained variance, entropy, advantages, singleton rate, historical/current proportion 기록.
- 기존 512-game/1M evaluation은 그대로 두고, sample-efficiency 비교 실험만 별도의 조기 평가 cadence를 도입 (학습 경기 카운터와 섞지 않음).

**승격:** 성능을 측정하지 못한다면 그 이후 기법의 효과를 주장하지 않는다.

### Track E1 — Early checkpoint 및 PFSP (가장 먼저 비교)

- **E1-A control:** Full Spec baseline = current 50%, history 50%(존재할 때), history uniform, snapshot every 1M.
- **E1-B:** same ratio, **더 이른 history snapshot**만 변경. history sampling은 uniform.
- **E1-C:** E1-B snapshot과 역사 풀을 동일하게 유지하고 historical **uniform vs PFSP-match**만 비교.
- **E1-D (optional):** smoothed PFSP-hard 혹은 uniform/PFSP 혼합을 추가 비교.
- **매치업 고정:** 두 side team selection independent uniform from all 1,137, roster and move permutation intact, same actor/learner resource budget, original 8-slot history cap, no new trainer policy shape.
- `f_match(w)=w(1−w)` 및 `f_hard(w)=(1−w)^p`는 연구에서 언급된 함수 family. 샘플 수와 CI가 부족하면 smoothed uniform/neutral fallback을 쓴다. `p`, smoothing constant, 최소 평가표본, snapshot cadence는 사전 실험 스펙으로 지정하고 비교 없이 임의 확정하지 않는다.
- PFSP의 효과는 **기존 1M 이전 history 없음 문제**와 교란되지 않도록 early snapshots control(E1-B)을 둔다.

### Track E2 — PPO/critic 진단 기반 ablation

- E1에서 채택된 opponent protocol을 고정한다.
- 기존 4 PPO epochs 대비 `epochs`, `critic-only extra training`, `GAE lambda`, `entropy/regularization`, learning-rate warmup 변경은 **독립된 한 변수 실험**으로 분리.
- PPO repeated epochs가 old-policy ratio drift 또는 KL exceed, value loss overfit로 이어지는지 기록.
- 독립 eval payoff를 사용해 성능 차이를 비교한다. 손실 감소만으로 개선 채택 금지.

### Track E3 — PPG-inspired / PPG Reloaded

- PPG의 정식 objective/policy-preserving distillation을 구현하는 선택지는 **기존 `PA3-8M` architecture 및 `no_teacher_loss`와 충돌하는지 사전 검토** 후 별도 승인된 branch에서 진행.
- baseline PPO, lightweight critic-only, full PPG-inspired 3군을 동일 게임 수, 동일 wall-time 양쪽에서 비교.
- 데이터 다양성과 regularization을 별도 측정하지 않으면 2023 후속 논문 요인을 재현했다고 주장할 수 없다.

### Track E4 — Shaping/PSRO/replay 고위험 연구

- 본격적인 정책 강도 상승이 E0–E3에서 확인되고 feature correctness/readiness를 충족한 뒤에만 정식 고려.
- Reward shaping: terminal potential, gamma=1, player-safe potential, 모든 episode의 telescoping test 필수.
- Replay: behavior policy ratio, stale-policy bounds, importance correction의 수학적/실측 검증 없는 past rollout PPO 재사용 금지.
- PSRO/League: 추가 모델 학습과 crossplay 계산을 모두 wall time budget에 포함.

### 단계형 자원 집행 (통계 확신 중심)

- 먼저 **작은 파일럿**으로 구현 정상, 상대가 실제로 다양해지는지, 유효한 학습 신호가 생기는지 확인. 파일럿 한 시드의 이득만으로 정책 승격 금지.
- 충분히 가능해 보이는 소수 후보만 여러 training seeds로 중간 budget 비교. 고정 match count 비교와 wall time 비교를 모두 실시.
- 채택 후보만 충분히 큰 budget에서 재검증. 비교 대상 checkpoint/seed 선정 기준 및 실패 기준을 **실험 전** 문서화.
- 특정 승률 차이를 보여주기 위한 임의의 seed 교체, team subset 조정, 평가 조건 변경 금지.

---

## 5. 기존 v1.1 성능 최적화와 충돌하지 않도록 하는 법

| Throughput track | Sample-efficiency track |
|---|---|
| Phase 0–10 CPU/GPU 데이터 처리 최적화, FP16 실측, DDP 스케일링 | 평가 인프라, opponent history/PFSP, PPO/PPG ablations |
| 처리량 목표: `real-policy actor games/s`, `full PPO all-in games/s` | 학습 효율 목표: fixed matchup/opponent-panel에서 equal matches 및 equal wall time의 성능 |
| 정확성: 관측, mask, logprob, 자연완결, PPO consistency | 상대 선택이 바뀐 이유와 정책 실력 향상을 독립 평가 |
| Full Spec 기본 설정 유지 | 설정 변경은 **명시적으로 승인된 experimental config/branch**, baseline은 보존 |

**권장 병렬화:** v1.1 에이전트는 코드 성능 최적화를 이어가고, 다른 에이전트는 E0 evaluator, payoff recorder, smoothed PFSP estimator의 **dry run + tests**만 준비한다. 최적화 코드와 RL algorithm 변경을 동시에 합쳐서 속도와 성능 원인을 흐리게 하지 않는다. 실제 학습 실험은 현재 full engine readiness 조건과 사용자의 실행 승인 범위가 충족된 뒤에 별도 수행한다.

---

## 6. 실험 리포트의 최소 요건

| 기록 | 이유 |
|---|---|
| `git SHA`, dataset manifest SHA, seed, source checkpoint | 재현성 |
| Training matches, learner decisions, optimizer steps | 학습 예산의 정확한 분모 |
| Eval matches 별도 집계, 상대 IDs 및 team group, side | 평가 분모/경향 검증 |
| Opponent mix: current/historical, snapshot ages, PFSP weights, empirical win estimates+CI | 어려운 상대를 실제로 더 만나도록 했는지 |
| Pairwise payoff matrix, fixed anchors, per-group score | Elo 하나가 가리는 전략적 약점 감지 |
| Training seed 간 분산과 interval estimates | 과도한 확신 방지 |
| Rollout cost, learner cost, per-update GPU time 및 메모리 | 샘플 효율 개선이 wall-time 악화를 동반하는지 |
| PPO safety: logits/masks/logprob, stale-policy %, KL/entropy/value calibration | 학습 유효성 |
| Clear recommendation: adopt, reject, needs more evidence | 사전 정한 승격 기준 준수 |

**최종 의사결정:** 후보가 fixed match budget에서 이기지만 full wall-time에서는 뒤지면 ‘sample-efficient but not compute-efficient’로 표시한다. 충분한 CI와 repeat run 없이 승격시키지 않는다.

---

## 7. 필수 1차 출처

1. Vinyals et al. (2019), **Grandmaster level in StarCraft II using multi-agent reinforcement learning**, Nature — PFSP league와 FSP 비교, 비추이적 전략: https://www.nature.com/articles/s41586-019-1724-z
2. OpenAI (2019), **Dota 2 / OpenAI Five technical report** — latest/history mix와 adaptive historical opponent selection: https://cdn.openai.com/dota-2.pdf
3. Cobbe et al. (2021), **Phasic Policy Gradient**, ICML — Procgen PPO 대비 sample efficiency: https://proceedings.mlr.press/v139/cobbe21a.html
4. Wang et al. (2023), **PPG Reloaded**, ICML — policy regularization, data diversity, low-compute variants: https://proceedings.mlr.press/v202/wang23aw.html
5. Andrychowicz et al. (2021), **What Matters for On-Policy Deep Actor-Critic Methods?**, ICLR — 50+ implementation decisions, 250k+ agents in continuous control: https://mlanthology.org/iclr/2021/andrychowicz2021iclr-matters/
6. Ng, Harada, Russell (1999), **Policy Invariance Under Reward Transformations**, ICML — potential-based shaping theorem: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
7. Schaul et al. (2016), **Prioritized Experience Replay**, ICLR — DQN Atari evidence, *not PPO*: https://mlanthology.org/iclr/2016/schaul2016iclr-prioritized/
8. Balduzzi et al. (2019), **Open-ended learning in symmetric zero-sum games**, ICML — non-transitivity and population objectives: https://proceedings.mlr.press/v97/balduzzi19a.html
9. Agarwal et al. (2021), **Deep RL at the Edge of the Statistical Precipice**, NeurIPS — training/evaluation seed uncertainty: https://papers.nips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html
10. Schulman et al. (2017), **Proximal Policy Optimization Algorithms** — on-policy old policy, multiple minibatch epochs: https://arxiv.org/abs/1707.06347

**정리:** 지금 당장 성공을 보장하는 알고리즘 변경은 없다. **측정 가능한 강함의 정의 → early opponent history → uniform PFSP 비교 → PPO/critic 진단 기반 ablation → 필요하면 PPG** 순으로 진행하는 것이 가장 근거 중심적이고 사용자의 경험 예산을 보호하는 접근이다.
