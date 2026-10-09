# 게임 자기대전 RL의 표본효율 / 벽시계당 학습속도 최신 근거
기준일: 2026-10-09 (KST). 대상 맥락: 커스텀 Rust VGC(포켓몬 더블) 시뮬레이터 + 2×V100 32GB, 약 8M 파라미터 정책, 현재 온폴리시 PPO를 직렬(수집 12.4s + 업데이트 42.2s) 루프로 학습 중.
근거 등급 표기: [실증]=논문 실험 수치, [주장]=저자의 주장/추정 견해, [2차]=백과/보도 등 2차 문헌 확인, [추정]=본 보고서의 계산/유추.

## TL;DR
- PPO 다중 에폭: 단일 최적값은 없고 실무 밴드는 대략 3~10 에폭. 원논문 설정은 Atari 3 / MuJoCo 10 / Roboschool 15였고, 최신(2026) 연구는 에폭 수 증식보다 직전 2~4회 반복 데이터를 윈도우로 재사용하는 쪽이 낫다고 보고(ω=4에서 이득 대부분 확보, ω=8 체감, naive 에폭 증식은 ω=4에서 붕괴). 다중에이전트에선 고재사용이 비정상성 때문에 해로울 수 있음(MAPPO 15/10/5 권장).
- 액터-러너 분리 + V-trace류 보정이 벽시계에서 가장 큰 레버. IMPALA: 단일머신 A3C 대비 30배 처리량(250k fps), A3C가 7.5일 걸린 DMLab-30 성능을 1 러너로 약 10시간. OpenAI Five는 staleness를 파라미터 버전 M−N으로 정의해 목표 0~1을 유지했고, 약 8 버전부터 유의한 감속을 보고.
- 리그 자기대전: AlphaStar 이후 메인+익스플로이터+PFSP가 사실상 표준. 어블레이션에서 리그 익스플로이터 제거 시 메인 Elo 1824→1540, 순정 self-play는 과거 상대 최소 승률 46%(망각) vs PFSP+SP 71%.
- 서치 증강: 소데이터에서 압도적. EfficientZero는 100k 프레임(약 2시간 경험)에서 194.3% mean/109% median, 4 GPU 7시간(비교: MuZero 64 TPU 12시간). KataGo는 기법 묶음으로 50배 컴퓨트 절감 주장, 19일·30 V100 미만으로 ELF OpenGo 최종 모델 능가.
- 동시수/불완전정보 게임(VGC 포함): 2024~2026 실증 중심은 (a) equilibrium 근사 트리서치(NN-CCE, Google Football/StarCraft 우위)와 (b) 정규화 다이나믹스(R-NaD) 계열 + 테스트시 서치. Ataraxos가 16 H100 1주, 8천 달러 미만으로 인간 최강 Stratego 선수 15-1-4 격파.
- 워밍스타트: VGC 도메인 직접 증거(VGC-Bench)에서 BC로 초기화한 뒤 population 방법(BCSP/BCFP/BCDO)이 순정 self-play보다 교차전 승률 우세. AlphaStar도 인간데이터 초기화가 결정적(제거 시 Elo 1540→1020).
- 소규모 성공 사례: Ataraxos(16 GPU·1주), EfficientZero(4 GPU), VGC-Bench(8×A40), MiniZero(4×1080Ti), Jaxpot(T4에서 90초 데모). KataGo README는 소비자용 GPU 1장으로 수개월 내 슈퍼휴먼 가능성을 언급[주장].
- 즉효 레버(본 보고서 추정): 수집-업데이트 파이프라이닝으로 직렬 54.6s/iter → max(42.2, 12.4)=42.2s/iter(약 1.29×), 여기에 업데이트 자체 축소(어드밴티지 필터링 등, 2× 사례)를 겹치면 추가 배수.

## 1. PPO 데이터 재사용(다중 에폭) — 몇 에폭이 최적인가, 온폴리시성 손실 대비 이득

핵심 수치
- PPO 원논문(2017) 하이퍼파라미터: Atari는 3 epochs(horizon 128, minibatch 32×8), MuJoCo 1M 벤치마크는 10 epochs(horizon 2048), Roboschool은 15 epochs. 즉 저자들도 도메인별로 3~15를 썼다. [실증] https://arxiv.org/abs/1707.06347
- MAPPO(다중에이전트, NeurIPS 2021): 쉬운 작업 15 에폭, 어려운 작업 10 또는 5 에폭 권장. 미니배치 분할 회피가 최선. 고데이터 재사용은 에이전트 동시 정책 변화(비정상성)를 키워 해로움. [실증] https://bair.berkeley.edu/blog/2021/07/14/mappo/ , https://arxiv.org/abs/2103.01955
- What Matters in On-Policy RL(2020, 250,000 에이전트 대규모 스터디): 경험을 여러 번 통과(multiple passes)하는 것이 표본복잡도에 결정적이라고 결론. 권장은 전이 단위 셔플 + pass마다 advantage 재계산. 테스트한 num_epochs는 {1,3,10}, 기본값 10. 부수 관찰: 병렬 환경 수를 늘리면 벽시계는 빨라지나 표본복잡도는 나빠질 수 있음. [실증] https://arxiv.org/abs/2006.05990
- 2026 최신(ωPPO, 2610.01399): 최근 반복들의 데이터를 importance weighting으로 재사용하는 두 변형(ωPPO-U, ωPPO-BH) 연구. 결과: ω=2부터 이득, ω=4에서 이득 대부분 확보, ω=8은 체감. 이득은 주로 최종 성능 쪽이고 수렴 속도 개선은 환경 의존. naive하게 에폭만 늘린 ωK-epoch 변형은 ω=2에서 소폭 개선 후 ω=4~8에서 완전 붕괴. 참고로 업데이트 횟수: PPO 기준 320회/iter vs ω=8 변형 2560회/iter(스테일 advantage 편향 증폭). [실증] https://arxiv.org/abs/2610.01399 , https://arxiv.org/html/2610.01399v1
- GIPO(2026): PPO 하드 클리핑이 스테일 데이터의 importance ratio를 잘라 그래디언트 기여를 0으로 만드는 문제를 정량화하고, 가우시안 중요도 샘플링으로 회수. 데이터 신선도가 제한될 때 클리핑 계열 베이스라인 대비 표본효율/리플레이 활용 우위. 10,000+ H200 GPU-hours 규모 스터디. [실증] https://arxiv.org/html/2603.03955
- 게임 스케일 실측 설정값: VGC-Bench(포켓몬 VGC, 2025)는 PPO에 epochs=10, batch=64, steps/update 24×128, 총 5,013,504 timestep을 사용. Ataraxos(2025)는 반복당 1 epoch + advantage filtering으로 배치를 1/4 이하로 줄이고 반복 wall-clock을 약 2배 단축. Jaxpot(2026 도구)의 기본값은 num_epochs=4. [실증]

종합(사실+해석)
- 사실: 3~10 에폭 밴드가 게임/연속제어 실무의 표준이며, 원리적으로 다중 통과는 표본효율을 올리지만 지나치면 (i) 스테일 어드밴티지 편향(2026 ωPPO), (ii) 다중에이전트 비정상성(MAPPO), (iii) 클리핑에 의한 그래디언트 손실(GIPO)로 상쇄된다.
- 해석(추정): 에폭 수를 크게 늘리는 것보다 (a) 에폭 4~10 밴드에서 클립비율/KL을 보며 튜닝, (b) 반복 간 재사용 윈도우(ω=2~4)를 도입하는 쪽이 2025~2026 문헌의 방향과 부합.

## 2. 오프폴리시 actor-critic(IMPALA/V-trace, APEX 등)의 이득과 정책 지연(staleness) 허용 범위

대규모 실증 수치
- IMPALA/V-trace(2018): 액터-러너 분리로 250,000 frames/s, 단일머신 A3C 대비 30배 처리량. DMLab-30 평균 capped human-normalized: IMPALA(deep, PBT) 49.4% vs A3C 23.8%. 벽시계: IMPALA 1 러너가 A3C의 7.5일 분량을 약 10시간에 달성, 8 러너는 추가 7배(30K→210K fps). 액터가 여러 업데이트만큼 뒤처질 수 있음을 명시하고 V-trace로 보정. 중요도 가중치는 ρ̄=c̄=1 클리핑이 최적(ρ̄∈{1,10,100} 비교). 리플레이를 넣어 지연이 커지면 V-trace가 1-step IS/ε-correction/no-correction보다 5개 태스크 중 4개에서 우수(예: seekavoid 최종 리턴 43.8 vs 41.6 vs 37.6 vs 11.2). [실증] https://arxiv.org/abs/1802.01561
- APEX(Ape-X DQN, 2018): 5일 학습, 376 cores + GPU 1장(P100), Atari 57개 median no-op 434% / human-starts 358%. 비교: Rainbow는 10일, 200M 프레임, 223%/153%. 분산 경험 리플레이 + 우선순위화의 벽시계 이득이 크다. [실증] https://arxiv.org/abs/1803.00933
- SEED RL(2019): IMPALA/V-trace + R2D2, 중앙집중 추론으로 초당 수백만 프레임, Atari-57 SOTA를 wall-time 3배 빠르게 달성, 실험비 40~80% 절감. Google Research Football SOTA 개선. [실증] https://arxiv.org/abs/1910.06591
- R2D2(RL): 리플레이+번인+zero-start로 Atari 강자. 정확 수치는 원논문 접근 실패로 위키백과 요약 사용: 2M 학습스텝/5일, MuZero가 500k 스텝에 평균/중앙값 매칭, 1M에 초과. [2차] https://en.wikipedia.org/wiki/MuZero

자기대전 특정 증거
- AlphaStar(Nature 2019): 학습자-행위자 비동기 + 리플레이 버퍼 + V-trace(정책)와 TD(λ)(가치) 하이브리드. 논문은 V-trace가 큰 구조화 행동공간에서 비효율적일 수 있어 가치 추정에는 off-policy 보정을 생략했다고 기술하며, V-trace 정책 보정이 학습 안정성을 개선했다고 보고. 리그 학습(44일, 약 900 플레이어)의 전제 조건. 정책지연 어블레이션(Fig 3I)은 그림만 제공되어 수치 미확보. [실증+주장 혼합] https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf , https://www.nature.com/articles/s41586-019-1724-z
- OpenAI Five(2019): Dota2(45분 장기전)에서 완전 온폴리시는 불가능하다고 명시. staleness를 파라미터 버전 차 M−N으로 정의, 최종 시스템은 staleness 0~1을 목표로 32 그래디언트 스텝마다 파라미터 발행. 약 8 버전의 staleness에서 유의한 학습 감속. 배치 최대 2,949,120 timestep(최대 1536 GPU), self-play 80% 최신 정책 + 20% 과거 정책. 총 학습 180일(10개월 실시간). [실증] https://cdn.openai.com/dota-2.pdf , https://arxiv.org/abs/1912.06680
- 정책 지연 허용범위 요약: 자기대전 대규모 사례 기준 (i) OpenAI Five 목표 0~1 버전, (ii) IMPALA는 여러 업데이트 지연 허용(보정 전제), (iii) 8 버전 근처에서 자체 감속 보고(OpenAI Five 실측). 보편 임계값은 없으며 보정 기법과 클리핑 설정에 종속. [실증+추정]
- 2025~2026 비동기 RL(LLM 도메인이지만 방법 전이 가능): A-3PO(staleness-aware PPO 근사), AReaL(NeurIPS 2025, 처리량↑+staleness 제어), COPC(2026), Staleness-Adaptive Trust Regions(2026). verl 완전비동기 문서는 스테일 샘플 한도를 배치 1개 미만으로 권장. 요지: 오버랩은 처리량을 크게 올리고, staleness는 명시적 상한+보정으로 관리. [실증(타도메인)] https://arxiv.org/abs/2512.06547 , https://neurips.cc/virtual/2025/poster/117538 , https://verl.readthedocs.io/en/latest/advance/fully_async.html

## 3. 리그 자기대전(AlphaStar league, PSRO, NFSP, PFSP) — 무엇이 지배적이 되었나

- AlphaStar league(실증, 상세 수치): 메인 3 + 메인 익스플로이터 3 + 리그 익스플로이터 6, 에이전트당 32 TPU, 44일, 약 900 플레이어 생성. PFSP 기본 가중 f_hard(x)=(1−x)^p(어려운 상대 집중, 이미 이기는 상대와 0게임), 수준 맞춤 커리큘럼 f_var=x(1−x)도 사용. 단순화 셋업 어블레이션: 풀 리그 Elo 1824 vs 메인 익스플로이터만 1693 vs 메인만 1540, 상대집단 성능 62%/35%/6%. 학습 알고리즘 비교 Elo: pFSP+SP 1540 > SP 1519 > pFSP 1273 > FSP 1143, 과거 상대 최소 승률은 pFSP+SP 71% vs 순정 SP 46%(순정 self-play가 가장 망각적). 최종 6,275(Protoss)/6,048(Terran)/5,835(Zerg) MMR, 상위 99.8%. [실증] https://www.nature.com/articles/s41586-019-1724-z
- NFSP(2016): 딥러닝+FSP로 Leduc에서 Nash 근접, Limit Texas Holdem에서 초인급 기존 알고리즘 성능에 근접. FSP의 최초 확장 사례. [실증] https://arxiv.org/abs/1603.01121
- PSRO(2017): 독립 RL(InRL)이 훈련 상대에 과적합됨을 지적(관절정책 상관 지표 제안), best-response+메타전략 프레임워크로 InRL·반복 BR·double oracle·fictitious play를 일반화. decoupled meta-solver로 확장성 확보. [실증] https://arxiv.org/abs/1711.00832
- 최근 변형(소규모 게임 실증): SP-PSRO(결정론적 BR 대신 근사 최적 확률정책 추가, 많은 게임에서 단 몇 반복으로 수렴), Self-adaptive PSRO(IJCAI 2024, 하이퍼파라미터 자동화), A-PSRO(ICML 2025, exploitability 감소). [실증] https://arxiv.org/abs/2207.06541 , https://arxiv.org/abs/2404.11144
- 자기대전 서베이(2024, Tsinghua 등): 알고리즘 통합 프레임워크 제공, 리그(main/exploiter+PFSP) 패턴이 전략/팀 게임(마작, 축구류, Diplomacy 응용 등)에서 반복 채택됨을 정리. [2차/서베이] https://arxiv.org/abs/2408.01072
- 소규모 적용(VGC 직접): VGC-Bench(2025)는 self-play, fictitious play, double oracle, PSRO 계열을 OpenSpiel 기반 병렬 학습으로 구현(8×A40 클러스터, RL 총 5M timestep). 교차대전 승률 결과: BC 워밍스타트 + population 계열(BCSP/BCFP/BCDO)이 최강, 순정 self-play는 중위권. 예(64팀 세팅): BCSP가 MaxBasePower 상대 0.878 vs 순정 SP 0.614. 인간 초인 수준 AI는 아직 없음(저자들 명시). [실증] https://arxiv.org/abs/2506.10326 , https://arxiv.org/html/2506.10326v2
- 무엇이 지배적으로 되었나(종합): 대규모 실전(AlphaStar, 마작/축구류 계열, GT Sophy 계열 파이프라인)은 리그+우선순위 상대샘플링(PFSP류)이 지배적. 학술 소규모에서는 PSRO 프레임워크(및 SP-PSRO, SPSRO 등 변형)가 표준 실험 도구. 순정 self-play 단독은 사이클/망각 문제로 실패 사례가 축적. [실증 기반 종합]

## 4. 서치 증강 학습(AlphaZero/MuZero/KataGo)의 표본효율과 동시수 게임 적용

- AlphaZero(2017): 5,000 1세대 TPU(셀프플레이 생성) + 64 2세대 TPU(학습). 체스는 4시간(300k 스텝)에 Stockfish 능가, 쇼기는 2시간 미만(110k), 바둑은 8시간(165k)에 AlphaGo Lee 능가. 총 학습 게임 수 44M(체스)/24M(쇼기)/21M(바둑). 24시간 내 3게임 초인. [실증] https://arxiv.org/abs/1712.01815
- MuZero(2019-2020): 규칙 없이 Go/chess/shogi에서 AlphaZero와 동급, Atari-57 신 SOTA. 컴퓨트: 보드게임 16 TPU 학습+1000 TPU 셀프플레이, Atari 8 TPU 학습+32 TPU 셀프플레이. 스텝 효율(위키 요약): 체스/쇼기는 약 1M 학습스텝에 AlphaZero 매칭, 바둑은 500k/1M. Atari는 R2D2 평균/중앙값을 500k 스텝에 매칭, 1M에 초과. [실증+2차] https://arxiv.org/abs/1911.08265 , https://en.wikipedia.org/wiki/MuZero
- KataGo(2019): AlphaZero 계열 개선(playout cap randomization, 정책 타깃 프루닝, global pooling, 보조 타깃 등)으로 50배 컴퓨트 절감 주장. 19일, 최대 28 V100(평균 26~27; 초기 16 → 24 self-play)으로 ELF OpenGo(수천 GPU 2주) 최종 모델 능가. 241M 샘플/4.2M 게임. 어블레이션 배수(2.5G 쿼리 도달 시간 기준): playout cap randomization 1.37×, 고정 플레이아웃+정책 프루닝 1.25×, global pooling 1.60×, 보조 정책 타깃 1.30×, 보조 owner/score 타깃 1.65×, 게임특화 피처 1.55×. [실증/주장] https://arxiv.org/abs/1902.10565
- EfficientZero(2021): Atari 100k(약 2시간 경험)에서 194.3% mean/109% median human-normalized, DQN@200M프레임과 유사 성능을 500배 적은 데이터로. 학습비용 4 GPU×7시간(비교: MuZero 64 TPU×12시간). [실증] https://arxiv.org/abs/2111.00210 , https://arxiv.org/html/2111.00210v2
- MiniZero(2023-2024, IEEE ToG): 4×GTX 1080Ti 단일 머신으로 9x9 Go/8x8 Othello/57 Atari에서 AlphaZero/MuZero/Gumbel 변형 통제 비교. 핵심: 보드게임은 시뮬레이션 수가 많을수록 강해지지만, 동일 시간 예산이면 소수 시뮬레이션 Gumbel Zero가 대형 시뮬레이션 AZ/MZ와 필적. progressive simulation 제안. [실증] https://arxiv.org/abs/2310.11305
- 동시수 게임(핵심, VGC와 구조 유사): NN-CCE 트리서치(2406.10411) — 동시수 게임에서 트리서치 내 equilibrium(CCE) 근사를 도입해 Google Research Football 3개 태스크에서 다른 equilibrium 근사 기법보다 높은 승률, MPE/GFR/SMAC 모두에서 MADDPG·SM-MCTS보다 우세. OpenSpiel 소형 시나리오(행렬게임 포함)에서도 우위. [실증] https://arxiv.org/abs/2406.10411
- 불완전정보(턴제) 대표 결과: Deep CFR(추상화 없이 대형 포커 성공), ReBeL(2인 제로섬 Nash 수렴 보장+HU NLHE 초인, AlphaZero의 불완전정보 일반화), Pluribus(6인 NLHE 초인; 블루프린트를 64코어 서버 8일/12,400 코어아워, 라이브 플레이는 1노드; 프로 상대 평균 +48 mbb/game), DeepNash(Stratego, 서치 없는 모델프리 R-NaD, 1024 TPU v3로 2~3개월, 비용 추정 $3~4.5M(보도/추정)). [실증+2차] https://arxiv.org/abs/1811.00164 , https://arxiv.org/abs/2007.13544 , https://www.science.org/doi/10.1126/science.aay2400 , https://arxiv.org/abs/2206.15378

해석: 완전정보 턴제에선 AZ/MZ식 서치가 표본효율 최강. 동시수+불완전정보(VGC)에선 2024~2026 실증이 (a) equilibrium 근사 트리서치와 (b) 정규화 다이나믹스(R-NaD 계열)+테스트시 서치로 수렴하는 중이며, 순수 모델프리 PPO는 이 클래스에서 약하다는 것이 최신 문헌의 그림. [종합]

## 5. 오프라인 RL(인간 데이터) + 워밍스타트

- AlphaStar: 슈퍼바이즈드 초기화가 결정적. 인간 리플레이(SC2 4.8 기반 971k개, 상위 22% 플레이어)로 SL → 평균 3,699 MMR(인간 상위 84%) → RL 44일 후 5,835~6,275 MMR(상위 99.8%). 어블레이션 Elo: 인간 init 1020 → +Supervised KL 1400 → +Statistics 1540. 논문: 인간 데이터 사용이 RL 성능에 결정적이었다고 명시. [실증]
- VGC-Bench(도메인 직접): 700,000+ 오픈팀시트 배틀로그 데이터셋 제공. BC 초기화 후 population 방법(BCSP/BCFP/BCDO)이 모든 팀 규모(1/4/16/64팀) 교차전에서 최상위권. 즉 VGC에서도 지도학습 워밍스타트+population 조합이 순정 RL보다 강함. [실증]
- RLPD(ICML 2023): 기존 오프폴리시 방법에 최소 변경만으로 오프라인 데이터 혼합 → 경쟁 벤치마크에서 2.5× 개선, 추가 컴퓨트 없음. [실증] https://arxiv.org/abs/2302.02948
- Cal-QL(NeurIPS 2023): 오프라인 초기화 후 빠른 온라인 파인튜닝을 위해 보정된 보수적 Q 학습. 9/11 파인튜닝 벤치마크에서 SOTA. [실증] https://arxiv.org/abs/2303.05479
- WSRL(2024): 파인튜닝 시 오프라인 데이터 리텐션이 불필요함을 보임(초기 발산 방지용 워밍업으로 대체). 대규모 오프라인 데이터 동시학습 비용 제거. [실증] https://arxiv.org/abs/2412.07762
- AlphaStar Unplugged(2023): 오프라인 데이터만으로 학습해 직전 BC 에이전트에 90% 승률(SOTA 갱신), StarCraft II 오프라인 RL 벤치마크. [실증] https://arxiv.org/abs/2308.03526
- 대비: OpenAI Five는 인간 게임을 아이템 빌드 조건화에만 극히 제한적으로 사용(논문: 시스템에서 인간 게임에 의존하는 유일한 지점). 자기대전만으로 초인 달성이 가능함을 보인 사례. [실증]

## 6. 소규모(1~8 GPU급) 자기대전 성공 사례 2024~2026 구체 수치

- Ataraxos(2025-11 arXiv, 2026 Nature; Stratego, 불완전정보): 최종 런 16×H100 1주(설치망/무브망) + 신념망 4×H100 4일, 총비용 8천 달러 미만(저자 추정). 핵심 기법: CUDA C++ GPU 시뮬레이터(10M state-updates/s/H100), GPU당 1,536 병렬 환경, 반복당 약 5M 전이, 반복당 1 epoch + advantage filtering(업데이트 wall-clock 약 2배 단축), 정규화 강도와 업데이트 크기의 자기대전 어닐링(정규화 다이나믹스 계열), 테스트시 belief 네트워크+서치. 결과: 세계 최강 인간 선수 상대 15-1-4, Barrage Stratego 초인, Hanabi/dou dizhu SOTA. 약 1억 6,300만 게임 셀프플레이(보도). [실증+2차] https://arxiv.org/abs/2511.07312 , https://arstechnica.com/science/2026/10/ai-finally-beat-the-best-stratego-player-in-history-and-did-it-on-a-budget/
- EfficientZero: 4 GPU×7시간으로 Atari 100k 초인. [실증]
- VGC-Bench: 8×A40 클러스터, RL 5M timestep 규모(포켓몬 VGC 표준 벤치마크). [실증]
- MiniZero: 4×1080Ti로 Go/Othello/Atari 학습-평가. [실증]
- Jaxpot(2026, 오픈소스, bards.ai): GPU 병렬 환경에서 자기대전(PPO+AlphaZero 스타일+리그+불완전정보 게임), 1,024 병렬 게임, Colab T4에서 90초 데모 학습, 1M+ 게임 규모 실험 동기. [주장/도구] https://www.bards.ai/blog/jaxpot-fast-self-play-rl-on-gpu
- KataGo README: 잘 튜닝하면 소비자용 최상급 GPU 1장으로 수개월 내 슈퍼휴먼 가능성 언급(실제 대규모 실적은 28 V100). [주장] https://raw.githubusercontent.com/lightvector/KataGo/master/README.md
- LightZero(NeurIPS 2023 spotlight): MCTS 계열 통합 벤치마크(소예산 RL 포함). [실증 프레임워크] https://arxiv.org/abs/2310.08348

## 7. 사실/추정 구분
- 실증(논문 실험): IMPALA/APEX/SEED 수치, AlphaStar 리그/인간데이터 수치, OpenAI Five staleness 실험, AlphaZero/MuZero/KataGo/EfficientZero/MiniZero 수치, ωPPO/GIPO 결과, RLPD 2.5×/Cal-QL/WSRL, VGC-Bench 교차전 표, Ataraxos 학습 구성.
- 주장(저자/보도): KataGo 50배·단일 GPU 가능성, Ataraxos 8천 달러 비용 추정, DeepNash 3~4.5M 달러 추정(보도), Jaxpot 속도 주장, AlphaStar off-policy 어블레이션의 세부 수치 부재 상태의 질적 서술.
- 추정(본 보고서): 사용자 시스템에서 오버랩만으로 약 1.29× (54.6s→42.2s/iter), 업데이트 축소 병행 시 그 이상; 도메인 차이로 타 게임 수치의 직접 환산 불가.

## 8. 실패/제한 검색 경로 기록
1) export.arxiv.org API가 빈 응답 → arxiv.org/abs HTML 파싱 + PDF 다운로드 후 pdftotext로 전환.
2) Semantic Scholar Graph API: 18건 조회 중 약 10건 HTTP 429(레이트리밋) → arXiv abs 페이지로 대체.
3) APEX OpenReview PDF 직접 페치 403 → arXiv PDF(1803.00933) 성공.
4) R2D2(RL) OpenReview 첨부 403(‘original_pdf’ 경로도 실패). 주의: arXiv 1906.06195는 동명의 컴퓨터비전 논문(R2D2 detectors)이라 혼동 위험 → R2D2(RL) 정확 수치는 Wikipedia(MuZero 항목)와 MuZero 논문 비교로 대체(원논문 수치 미확보).
5) What Matters(2006.05990) 에폭 최적: Figure 6/54가 그래프라 정확 수치 판독 불가 → 본문 권고문으로 대체.
6) AlphaStar 정책지연/off-policy 어블레이션(Fig 3I) 수치: 그림 형태 → 질적 서술만 확보.
7) DeepNash 학습비용: 원논문 초록에 없음 → Ars Technica(2026) 및 Ataraxos related work 각주(1024 TPU v3, 2~3개월)로 보완.
8) GT Sophy 리그 구성 상세, NFSP의 정확한 mbb 수치: 미확보(범위 밖 판단).
9) ‘PPO wall-clock 최적화’ 일반 검색은 블로그성 노이즈가 많아 논문/1차 소스 중심으로 재검색.
10) Synapse 검색(127.0.0.1:8888)은 정상(HTTP 200)이었고, 차단/레이트리밋은 위 API/PDF 경로에서만 발생.

## 9. 2×V100 적용 메모(요약)
- 최대 레버 우선순위(추정): (a) 수집-업데이트 오버랩/비동기화+staleness 보정, (b) 업데이트 비용 절감(어드밴티지 필터링·에폭 튜닝·배치 조정), (c) 상대 풀(PFSP 소형 리그), (d) BC 워밍스타트, (e) 시뮬레이터 병렬화(프로세스/GPU 배칭).
- V100 주의: bf16 미지원(fp16/AMP 사용), 8M 파라미터 규모에서는 2×32GB로 배치 확대 여유가 큼.
- 도메인 정합: VGC는 동시수+불완전정보이므로, 포커/Stratego 계열(CFR, equilibrium 근사, R-NaD) 문헌이 AZ/MZ식 완전정보 서치보다 참조 가치가 높다. VGC-Bench(2025)가 가장 가까운 벤치마크이며, 순정 RL보다 BC+population 조합이 우세했다는 결과가 핵심 시사점이다.