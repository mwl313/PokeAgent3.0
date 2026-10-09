# 포켓몬 배틀 AI / 자기대전 RL 생태계 리서치 (기준일 2026-10-09)

> 조사 방법: 로컬 Synapse(SearXNG) 검색 + 1차 소스 직접 페치(arXiv/GitLab API/HF/Smogon 스레드/X). 사실은 출처와 함께, 추정은 `(추정)`으로 명시. 실패 경로는 §9에 기록. GitHub 스타 수는 2026-10-09 GitHub API 실측.

---

## 📌 TL;DR

- **(1) IEEE CoG "VGC AI Competition"는 Pokémon Showdown 기반이 아니다.** 주최측 자체 파이썬 엔진(`GitLab DracoStriker/pokemon-vgc-engine`, "Pokémon-like" 배틀)으로 열리는 학술 대회이며 2023(1st)·2024(2nd)·2025·2026(4th) 4회 개최. 2026 배틀 트랙 우승 **SelfPlayFlat2026 (Hiroshi Kiyota)**, 챔피언십 트랙 우승 **Loblolly (David Kinlaw, ELO 1616)** — 상위권 대부분 한국 참가자. [1][2][3][4][5]
- **(2) Metamon** (UT Austin, RLC 2025): 인간 리플레이 1인칭 재구성 + **IL→오프라인 RL→셀프플레이 파인튜닝**, 200M 트랜스포머, **8×A5000 1대·≥1M 스텝/모델**. 데이터는 논문 시점 475k 데모 → 현재 README 기준 **인간 5M+ 트래젝토리 + 셀프플레이 20M+**, 인간 상대 **90~99퍼센타일**. [9][10][11][13]
- **(3) PokéLLMon** (Georgia Tech, ICML 2024): ICRL+도감검색(KAG)+일관행동 생성. **래더 48.57%(105판) / 초청 56.0%(50판)** 승률. 한계 = hallucination·panic switching·소모전/기만 취약. [15][16]
- **(4) 2026-08~10 쇼다운 커뮤니티 봇 웨이브**: **Jaxcalibur**(randbats, 8.5M, H100 1주·약 1억 경기, peak 2557 Elo/95.3 GXE, **10-04 인간 최강자 MDB와 Bo9에서 5-3 승**) / **mikumiku37**(VGC M-C, 8.7M, RTX 5090 48.3h·약 3.3억 경기, 래더 #1 1857 Elo/81.2% GXE) / **Nessie123**(OTS Bo3 #1, 1.5M, 맥미니 20h+$120). **셋 다 코드·가중치 비공개.** [25][26][28][30][31][35]
- **Jaxcalibur는 9-12 비공개 결정 후 10-04부터 공개 챌린지 재개**(래더는 계속 오프), Smogon(SSUS)은 봇 대응 **정책 성명·액션 플랜 준비 중**. mikumiku37은 10-08 기준 "더 강해졌고 OTS 대개선 예정"·writeup 미출시. Nessie는 학습/분석 툴을 **Reg M-C 종료(2026-12-02) 이후** 공개 계획. [26][28][30][33][34][35][47]
- **핵심 레버 = 시뮬레이터 속도**: miku 자체 Rust 엔진 ≈**7,000 배틀/s/코어**(쇼다운 대비 ~650x, 8192 병렬·수집 10만 step/s, PPO 업데이트가 wall 70%) vs 쇼다운 ~700 배틀/분. 우리 현재 32.5 games/s는 최대 격차. [30][W64]
- **서치·상대예측이 명시적 성능 축**: Jaxcalibur 서치(테스트타임 pUCT depth4·20,480 롤아웃·32월드) **+100~150 Elo 추정**, 상대 다음행동 예측 보조헤드 "subsequently stronger" 주 원인. miku는 1턴 동시수 내시(16월드×top8×8, 트리 확장 없음). [25][30]
- **미개척 차별화 축 (증거 있음)**: ①인간 리플레이 활용(miku/Jax/Nessie **0**, 우리 786K+) ②학습된 팀빌딩·티프리뷰(miku 팀편집 약함 실측 — "Rillaboom/Gambit만 넣음"; Jax는 randbats라 팀빌딩 자체가 없음) ③OTS/CTS 혼합·상대모델링(Nessie는 OTS 편향) ④검색 강도 확장. §7~§8. [10][11][20][30][32]
- **2×V100 전제 각도**: 오프라인 RL 워밍스타트로 경기 수 요구를 줄이고(Metamon 실증), V100은 8M급 추론 여력 충분 — 승부처는 **시뮬 처리량 + 팀/정보 축 + 서치 add-on**. §8. (추정 포함)

---

## 1. IEEE CoG VGC AI Competition — 연도별 데이터

**정체 (사실)**: IEEE Conference on Games(CoG) 공식 컴페티션. 주최 Simão Reis(VORTEX-CoLab/LIACC, 포르투 대). 2021년 논문으로 프레임워크·모델 제안 후 `pokemon-vgc-engine`(Python, GitLab) 위에서 진행. **쇼다운 기반이 아니라 자체 '포켓몬 유사' 엔진** — 2024 배틀 트랙 기준 몬스터 3마리 팀 랜덤 생성, PkmBattleEnv는 Gymnasium API(v3.0.4) 준수. 트랙 = Battle / Championship (+2025부터 Rules Balance 실험). [1][2][3][4-A][6][8]

| 에디션 | 개최/발표 | 참가 | 결과 (상위) | 공개된 접근법·예산 | 링크 |
|---|---|---|---|---|---|
| **2023 (1st)** | CoG 2023 | — | Battle: **1위 NiBot (Nizar Haimoud, 7pt)**, 2위 동률 다수(NeatPlayer 등), KidMeStrategy 실격 | 코드 공개(`agent/vgc2023`). 알고리즘 비상세 | [4-B] |
| **2024 (2nd)** | 제출 7/15, 결과 7/22 | 18팀(Battle 18·Champ 11) | Battle: **1위 Daniel Ladwig·Marlon Hörner**, 2위 Jo ha young, **3위 AurelianTactics** / Champ: 1위 Eita Aoki. 상금 $500/300/200 | **3위 writeup 상세**: 상태공간 11차원으로 축약 + first-visit MC 계열 **tabular RL, 3천만 trial**(+버그로 10억 trial 폐기), 실패한 Deep RL 대체. 컴퓨트=개인 PC급(추정). 에이전트 코드 공개(`competition/vgc2024`) | [5][6][8] |
| **2025** | 시작 2/9, 마감 6/30–7/13, 발표 8/1 | 13팀(Battle 13·Champ 11) | Battle: **1위 Masatoshi Hidaka**, 2위 Jun Sung(=Champ 1위), 3위 KIM DONGMIN / Champ: 2위 Aytur Farhadi, 3위 Leon Brunke | 신규 **Rules Balance 트랙** 논문(IEEE CoG 2025) 공개. 우승팀 상세 writeup은 공개 확인 안 됨 (미확인) | [2][7] |
| **2026 (4th)** | 시작 2/9, 마감 7/13(CoG페이지)·7/15(위키), 발표 8/1 | 11팀(Battle 11·Champ 10·Balance 1→취소) | Battle(100 토너먼트): **1위 SelfPlayFlat2026 (Hiroshi Kiyota; 23 wins/187, avg #4.05)**, 2위 DaehoV3(Daehan Choi), 3위 Example 3(HYEJEONG JE) / Champ(Wave100): **1위 Loblolly (David Kinlaw, ELO 1616)**, 2위 SYbot, 3위 DaehoV3 | 우승 이름 "SelfPlayFlat" = 셀프플레이 플랫 정책(추정). 상세 예산 비공개 (사실: 미기재). 한국 참가자 다수 | [1][7-B][45] |

- 참고: 2026 페이지·2025 페이지 모두 제출 코드 공개 디렉토리 관행이 있으나 2026 상위 제출물의 실제 공개 여부는 미확인. `Balance 트랙`은 신청 1팀으로 취소. [7-B]
- 연도별 우승 "알고리즘/컴퓨트"의 공식 공개는 2024 3위 외에는 확인 실패(검색 미인덱스, §9 실패 경로).

---

## 2. Metamon (Pokémon Showdown 오프라인 RL 벤치마크)

| 항목 | 내용 | 출처 |
|---|---|---|
| 논문 | "Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers", **RLC 2025**, arXiv 2504.04395 (2025-04-06) | [9][10] |
| 저자/기관 | **Jake Grigsby, Yuqi Xie, Justin Sasek, Steven Zheng, Yuke Zhu** — UT Austin RPL | [10] |
| 데이터(논문) | 쇼다운 리플레이(2014~)에서 1인칭 관점 재구성 **>475k 인간 데모**(shaped reward); HF v0 = **약 1M 리플레이**(numpy) | [10][13] |
| 데이터(현재) | README: **인간 배틀 유래 5M+ 트래젝토리 + 셀프플레이 20M+**; 리플레이는 "v1(2025-04-25)·v2(2025-05-29)" 이후 지속 갱신, CC BY-NC 4.0, Gen1–4 OU(+Gen9 OU 확장) | [11][13] |
| 방법 | IL(BC) → **오프라인 RL** → 셀프플레이 데이터 **오프라인 파인튜닝**; 비인과 시퀀스 트랜스포머(15M/50M/57M/62M/142M/**200M**); **탐색(서치) 없음** — 상대 적응을 메모리로 추론; V2는 value classification | [10][11] |
| 성능 | 논문: 인간 상대 승률 **64–80%**, **top 10% 랭킹**; LLM 에이전트·강력 휴리스틱 엔진 격파. 현재: **90–99퍼센타일**(룰셋별). GXE 실측 — Kakuna(142M): G1 82/G2 70/G3 63/G4 64/G9 71; TaurosV0(62M): G1 83; SyntheticRLV2(200M): 77/68/64/66 | [10][11] |
| 컴퓨트 | **단일 8×NVIDIA A5000 머신, 모델당 ≥1M gradient steps**; 훈련 데이터 = 인간 1M(+셀프플레이 0~4M) 배틀 | [10][11] |
| 인프라 | `metamon.tech`, GitHub(143★), HF 체크포인트 40+종(`jakegrigsby/metamon`), RL 코드베이스 `amago`(172★), PokéAgent Leaderboard(battling.pokeagentchallenge.com) | [11][12][14] |
| 비고 | NeurIPS 2025 PokéAgent Challenge 스타터킷+베이스라인(자체가 상위 성적): Kadabra3 #1 Gen1OU·#2 Gen9OU; **TaurosEnsemble이 인간 쇼다운 래더 #1 유지**(KakunaEnsemble에 이어 2번째) | [11][22] |

---

## 3. PokéLLMon (LLM 배틀 에이전트)

| 항목 | 내용 | 출처 |
|---|---|---|
| 논문 | "PokéLLMon: A Human-Parity Agent for Pokémon Battles with LLMs", arXiv 2402.01118, **ICML 2024**; Georgia Tech (Sihao Hu, Tiansheng Huang, Ling Liu) | [15][16] |
| 방법 | ①**In-Context RL**: 배틀 텍스트 피드백을 즉시 '보상'처럼 소비해 무학습 정책 개선 ②**KAG**: 포켓덱스 검색으로 hallucination(가짜 상성) 억제 ③**일관 행동 생성**: 다수결 투표로 panic switching 완화 | [15][16] |
| 성능 | 인간 상대: **래더 48.57% (105판)**, **초청 플레이어 56.0% (50판)** [Table 7]. 예비 실험: GPT-4 원시 26% vs 휴리스틱 봇(인간 59.84%). 타입상성 예측 훈련으로 GPT-4 26%→KAG 58% | [16] |
| 한계 | hallucination(상성 역전 등)·**panic switching**(체인오브소트 시 악화) · 인간의 **attrition(소모전)·deception(기만) 취약** — 논문이 미해결로 명시. 후속: ICLR 2025 워크숍 "Grounding and Reasoning Benchmark" 버전 | [15][16][49] |
| 컴퓨트 | 학습 없음(프롬프트+검색), LLM API 추론 비용 — **수치 비공개**(추정: 판당 소액). 4분/판 수준 인간 상대 지연은 배틀 문제였음 | [16] |

---

## 4. 기타 Showdown/포켓몬 배틀 RL·검색 봇 + 후속 동향 (2026-10-09)

### 4.1 GitHub 주요 리포 (★ = 2026-10-09 실측)

| 리포 | ★ | 접근법 | 공개 성능/비고 | 출처 |
|---|---:|---|---|---|
| smogon/pokemon-showdown | 5,942 | 배틀 시뮬레이터(본체) | 생태계 기반 | [41] |
| hsahovic/poke-env | 525 | Python 봇 인터페이스 | 사실상 표준(2026-10-09 푸시) | [40] |
| pmariglia/foul-play | 384 | **DUCT MCTS**(루트 병렬화)+Rust `poke-engine`(52★); 커스텀 평가함수, **포지션별 set prediction**(Smogon 사용률+포럼+리플레이 스크래핑) | PokéAgent Gen9 OU **챔피언**(결승 Q에 50–14), depth≈5턴(유망 라인 10+) | [22][36] |
| git-disl/PokeLLMon | 206 | LLM(§3) | 래더 48.6% | [17] |
| sethkarten/pokechamp | 185 | **미니맥스 언어 에이전트**(ICML 2025 spotlight): LLM이 플레이어 샘플링·상대 모델링·가치추정 | PokéAgent 상위 | [39][22] |
| UT-Austin-RPL/amago | 172 | 장시퀀스 off-policy RL 코드베이스 | Metamon 학습 기반 | [11][42] |
| UT-Austin-RPL/metamon | 143 | §2 | top10~99% | [11] |
| cameronangliss/vgc-bench | 54 | VGC 더블 벤치: 700K 로그, BC+SP/FP/DO(PSRO), LLM·휴리스틱 | **8×A40 클러스터**; 미러 단일팀에서 프로 격파(§4.3) | [19][20][21] |
| heatz123/pokeagent-solution | 10 | PokéAgent 스피드런 우승팀 공개 솔루션 | SPD(스크립트 정책 증류) | [42] |
| TurboRx/Chronoquazicalar | 1 (신규) | 9.65M 비인과 TF + JAX 텐서 시뮬 + **동시수 CFR+pUCT 내시** + Bayesian set inference(509종) + 상대행동 예측 헤드 | **TPU v3-8 >4,400 turns/s, 140M+ turns; HF에 체크포인트 공개** — randbats판 오픈 복제(Jaxcalibur 파급) | [37] |
| Xyqc/japlace | 0 | Laplace 변형: poke-engine + **학습된 policy prior MCTS** + 아이템/특성/테라 예측 + 히든정보 posterior sampling | 2026-09-21 푸시 | [38] |
| Nolelle/pokemon-vgc-ai | 1 | 규칙 기반+1턴 엔진, **Reg M-C 챔피언스 더블 래더** | 2026-10-08 푸시 | [43] |
| philmantatsky/VGC-Pokemon-Showdown-AI | 1 | PPO 정책 + exact-simulation search, Reg M-B | 2026-10-09 푸시 | [44] |
| gary5876/vgc-ai | 1 | 휴리스틱 탐색 + LP-minimax 팀빌딩, n=2000 벤치 | **한국인, IEEE CoG 2026 VGC AI 대회 준비** | [45] |
| 기타 신생(0–1★) | — | EmoMeza/poke-emo-bot(VGC self-play), FYYHU/PokBot(Champions Reg M-B, BC+RL), DylanB03/pokemonBattlerML(Qwen2.5-0.5B+Metamon self-play), PriyadarshanPatil, alexmunzon, HalluNasty/Elytrai | 2026-08~10 | [46] |

### 4.2 PokéAgent Challenge (NeurIPS 2025, 쇼다운 기반 — 사실상 최대 공개 대회)

- Battle 트랙: **Gen1 OU 챔피언 PA-Agent**(결승 4thLesson에 50–28, GXE 80.35%), **Gen9 OU 챔피언 FoulPlay**(결승 Q에 50–14). 16개 토너먼트 슬롯 중 13개가 공개 RL 베이스라인 확장 팀; 비-베이스라인 = Porygon2AI(AlphaStar식 리그 학습)와 FoulPlay(검색). [22]
- 스피드런: 우승 **Heatz**(LLM이 서브골→스크립트 정책→증류+RL, 40:13), 2위 Hamburg PokéRunners(순환 PPO), Judge's Choice: Porygon2AI·August(순수 LLM CoT). [22]
- 규모: 100+ 팀, 650+ Discord, 대회 서버 10만+ 배틀, 데이터셋 3.5M+ 배틀, 랭킹 분석에 1.6M+ 에이전트 매치. **"모든 상위 팀은 순수 LLM이 아니라 RL/서치 계열"**. [22][24]

### 4.3 VGC-Bench 실전 수치 (사실)

- 미러 단일팀(BC+SP): 중급자 5-0, 상급자 2-5, **전문가에게도 승리(≥1승)** — 전문가 = **Aaron Traylor**(WCS 다회 참가·2020 Dallas Regional 챔피언). 단 논문 자체가 "일화적 평가"로 한정, 강한 인간은 반복 대국으로 적응 가능. [20]
- 팀 수 1→4→16→64 확대 시: 단일팀 최고 알고리즘의 성능↓·착취 취약↑, 미학습 팀 일반화↑ (팀 공간 10^139). [19][20]

### 4.4 mikumiku37 / Nessie123 / Jaxcalibur — 10-09 기준 후속 업데이트

| 프로젝트 | 최신 동향 (2026-10-01~09) | 코드/모델 공개 |
|---|---|---|
| **mikumiku37** (Shao, @shaokthx, Discord shao) | 10-01 M-C CTS 래더 #1 (**1857 Elo / 81.2% GXE / Glicko 1775±25**); 10-03 공개 트윗: 챌린지 775판 **557-218 (72% WR)**, 20팀 로테이션; 10-06 **OTS 챌린지 지원**(20초 대기, 선택 매칭)·Bo3 예고; 10-07 스레드 기술 답변(서치=샘플링 월드+value net, regret matching 혼합; 리그=PFSP식, "절제된 실험 없음"; 엔드게임 심화 시도→51.4%로 미미); **10-08 "봇 더 강해짐 + 대형 OTS 개선 예정", 운영비 ~$150/월, 자기대전 Twitch 스트리밍 아이디어**; writeup "곧"(미출시) | **엔진·가중치 비공개 유지**("래더/챔피언스 대회 영향 우려"), writeup 예고 |
| **Nessie123** (nessie_dev) | 10-03 스레드: OTS Bo3 래더 #1 검증; "상세 writeup은 추후"; **포지션 분석 툴(수 선택·RNG 결과 지정·포지션 분석) 공개 계획 — "Reg M-C 종료 후 며칠 내, 커뮤니티 준비되면 그보다 이르게"** (M-C는 12-02 종료 예정 → 10-09 미공개). Shao와 무관 확인 | 코드·가중치 비공개; 툴만 예정 |
| **Jaxcalibur** (jaxcalibur_author) | 8/6–27 래더 (**peak 2557 Elo/95.3 GXE**, top-1 약 20% 시간, top 상대 승률 68%; 인간 MDB는 GXE 95.5로 능가); 8/28 스레드+상세 웹 writeup; **9-12 코드·가중치 비공개 결정**+공개 챌린지 잠정 중단; **10-04 Smogon 주최 MDB(인간 최강 랜배) vs Jaxcalibur Bo9 전시 경기 → Jaxcalibur 5-3 승** (freezai "2026년의 카스파로프 vs 딥블루"; 리플레이 9판 공개, MDB 영상 예정); 10-02-03: 결전용 신버전이 구버전 상대 117-83; **이후 공개 챌린지 재개(래더는 오프 유지)**; SSUS 정책 성명 준비 중 | **코드·가중치 비공개(결정)** — 단 방법론은 웹 writeup에 상세 공개, 셀프플레이 승률 통계도 공개 |

---

## 5. 프로젝트별 필요 컴퓨트 / 경기 수 요약

| 프로젝트 | 모델 | 시뮬레이터 | 학습량 | 하드웨어/시간 | 공개 성능 | 성격 |
|---|---|---|---|---|---|---|
| **mikumiku37** | 8.7M 트랜스포머(스크래치) | 자체 Rust, **~7,000 배틀/s/코어**(작성자 주장 쇼다운 대비 ~650x); 훈련 시 8,192 배틀 병렬·32스텝/업데이트(~262k 샘플), 수집 10만 step/s, **PPO 업데이트 = wall 70%** | **~330M 게임(48.3h 시점) → 최신 ~500M**(스레드 답변) | **RTX 5090 1장, 48.3h** | M-C #1: 1857 Elo/81.2 GXE; 서치 없던 동일망 1747 → 서치판 358판 만에 #1 | 휴먼 데이터 0, PPO+과거버전 리그(PFSP), 서치=1턴 동시수 내시(16월드×top8×8, RNG 시드 16) |
| **Jaxcalibur** | 8.5M 비인과 TF(+위치별 MoE) | **순수 JAX 재구현**(vmapped/jitted; 모든 스텝 실행해 FLOPs 낭비·구조 균일화) | **≈1억 게임 self-play** | **H100 1장 ≈1주** | randbats peak **2557 Elo / 95.3 GXE**; FoulPlay 비교 2341/88(top-50) | PPO+GAE, 무할인 1/0 보상, 엔트로피+균일정책 KL; 보조헤드=상대 아이템·특성·기술·테라+**다음 행동 예측**; 서치=테스트타임 pUCT(depth4/20,480롤아웃/32월드, state-hash 노드, HP 10버킷, 상대 서치 절단), **+100~150 Elo 추정** |
| **Nessie123** | 1.5M | (미공개) | **~375k self-play 게임**(내부 스냅샷 기록) | 맥미니 20h + **$120 클라우드**; 운용=맥미니 CPU 1코어 | OTS Bo3 #1 | OTS 근사 완전정보 학습, **더블오라클+선택적 심화**, 2v2 이하 엔드게임 사실상 완전풀이(엄밀 솔버와 0.5% 이내) |
| **Metamon** | 15M~**200M** TF | 쇼다운(로컬) | 1M 인간(+셀프플레이 0~4M 배틀); ≥1M 그래디언트 스텝 | **8×A5000 1대** | 인간 64-80% 승률, top10%→현 90-99pct | 오프라인 RL 파이프라인(서치 없음) |
| **VGC-Bench** | 소형(비공개) | 쇼다운 기반 | BC+SP/FP/DO(팀 세트 1~64) | **8×A40 클러스터** | 미러 단일팀: 전문가에게 승리(§4.3) | PSRO 계열 MARL 벤치 |
| **PokéLLMon** | GPT-3.5/4 API | 쇼다운(온라인) | 0(프롬프트) | API 추론 | 래더 48.57%(105판) | LLM 에이전트 |
| **Chronoquazicalar** | 9.65M TF | JAX(TPU) | 140M+ turns | **TPU v3-8 >4,400 turns/s** | (신규, HF 공개) | CFR+pUCT 내시+set inference |
| **(우리, 참고)** | 8M 정책 | 자체 Rust | 32.5 games/s(2048판=수집 12.4s+업데이트 42.2s) | **2×V100 32GB** | 목표 1억 경기=35일+ → 전략 재설계 중 | BC 자산(786K+ 리플레이 코퍼스) + 자기대전 PPO |

> 주의: miku의 게임 수(330M/500M)·650x, Jax의 1억 게임·+100~150 Elo, Nessie의 375k는 **작성자 본인 진술**(검증 불가). Metamon 수치는 논문+README 실측. 우리 수치는 사용자 제공 실측.

---

## 6. Jaxcalibur vs mikumiku37 정밀 비교 (강점·약점·미공개)

**공통점(사실)**: ① 소형 트랜스포머(8.5M/8.7M) 스크래치 ② 순수 자기대전 RL(PPO 계열, 휴먼 데이터 0) ③ 고속 자체 엔진(JAX GPU 병렬 vs Rust 멀티코어) ④ 테스트타임 서치 결합 ⑤ 과거 버전 상대 리그/스냅샷 학습 ⑥ 코드·가중치 비공개 ⑦ 커뮤니티 자기식별/비공개 관행 주도.

| 축 | Jaxcalibur | mikumiku37 |
|---|---|---|
| 포맷 | Gen9 randbats 싱글 | **VGC Reg M-C 더블**(챔피언스 룰) |
| 학습량/장비 | ~1억 게임 / H100 1주 | ~3.3억→5억+ 게임 / 5090 48h+ |
| 서치 | pUCT: 32월드·20,480롤아웃·depth 4(≈2턴)+**상대 서치 절단**(<65% 예측 행동 시 prior로) | **1턴 동시수 내시**(16월드 — "64~128이 이상적" 자인, top8×8 페이오프, regret matching, **트리 확장 없음**) |
| 상대 모델링 | **전용 보조헤드 5종**(승률/상대 아이템·특성·기술·테라/다음 행동) — 다음행동 예측이 최대 기여 | 관측=자기 진영+정적 도감만; 댐계·사용률·스피드 리졸버 없음; 별도 상대모델 헤드 미공개(추정: 없음) |
| 정보 가정 | 히든정보=월드 샘플링으로 평균, 정보 누출 방지 설계 | CTS(OTS 거부) 중심, 관측만으로 추론 |
| 검증 | Bo9에서 인간 최강자 5-3 격파(10-04) | 래더 #1(일시)+775판 72% WR(챌린지 상대 혼합) |
| 약점 | randbats 전용(팀빌딩 없음), depth 4라 장기계획 제한, 싱글 중심; 검증된 약점 커뮤니티 보고는 미확인 | 팀 편집 약함(10-07 트윗: "Rillaboom/Gambit만 넣음"), 테레인 효과 의혹·Kingambit 예측 실패(커뮤니티 제보, **미검증**), 4시드 서치 이상수 이력, 승률 포화 시 BM 버그 이력 |
| 미공개 | 엔진·가중치(결정적 비공개) — **단 방법론 writeup·셀프플레이 승률·리플레이 공개** | 엔진·가중치·writeup 전부 미공개(스레드 답변 수준) |

- 시사점: **Jaxcalibur는 "방법론은 공개, 자산은 비공개"** 전략으로 학술적 재현성을 열어둠. miku는 자산·방법 모두 비공개 — 우리가 참조할 수 있는 문서는 스레드 답변 2건(10-07)이 전부. [25][26][30]

---

## 7. 차별화 축 증거 (OOB ③ — 서치/상대모델링/팀/인간데이터)

**① 추론 서치 강도**
- miku: 서치 없던 동일 네트워크 100게임 최고 1747 → 서치판이 358게임 만에 #1; 현재 16월드·1턴·무롤아웃. 엔드게임 심화는 +1.4%p 수준으로 기각, "64~128 시드가 이상적"이라고 본인 언급. [30][W64]
- Jax: 서치 +100~150 Elo(본인 추정), 32월드·depth4. 학습 중 서치는 미사용("포켓몬에선 서치 가치가 Go/체스보다 낮다"). [25]
- Nessie: double oracle + "도달확률²×엔트로피" 우선순위로 선택적 심화; 엔드게임 풀이 정확도 0.5%. [W64]
- FoulPlay: DUCT MCTS depth≈5, 유망 라인 10+턴. [22]
→ 결론: **월드 수·깊이·롤아웃 예산이 공개적으로 검증된 성능 축**이며, 상위 봇들의 서치 강도는 아직 (16~32월드, ≤4뎁스) 수준 = 확장 여지 존재.

**② 상대모델링/불완전정보**
- Jaxcalibur: 월드 샘플링 + 상대 예측 보조헤드(다음행동 예측이 "substantially stronger"), 놀라운 수에서 상대 서치 절단으로 정보 누출 차단. [25]
- Nessie: 학습을 **OTS 근사 완전정보**(스탯포인트·선출 공개)로 진행 — CTS(비공개) 일반화는 약점 가능성(추정, 본인도 "일반화 위해 diverse teams 강조"). [35][W64]
- miku: 관측만+도감, 스피드 리졸버 없음 → 킬각/스피드 판단이 가치망 의존. [W64]
- Metamon: "모델이 트래젝토리만 보고 상대 팀을 추론"(서치 없이 상대 적응) 설계 철학. [10][12]

**③ 팀 선택·팀빌딩**
- miku: 20팀 로테이션·1,260개 대회팀 학습(스프레드 일부 추정); **팀 편집 능력은 약함**(10-07 트윗). [30][32]
- Jax: randbats 전용(팀빌딩 없음), "self-play로 팀 강도 평가 가능" 언급만. [25]
- Nessie: 수작업 팀풀(볼티모어 리저널 6-2+ 팀들), 팀프리뷰 자평 48-52. [W64]
- VGC-Bench: 팀 다양성↑ → 단일팀 성능 저하·착취 취약 증가, 일반화 증가. **학습된 팀빌딩/선출은 공개 연구·공개 봇 모두에서 미개척.** [19][20]

**④ 인간 리플레이 데이터 활용 사례**
- Metamon: 인간 데모(475k→1M+)로 IL + **오프라인 RL** → 200M 모델이 top10~99% (인간 데이터가 코어 자산). [10][11]
- VGC-Bench: 700K 인간 로그로 BC 베이스라인 + RL 파인튜닝. [19][20]
- PokéLLMon: 데이터 학습 없음(ICRL). [16]
- **mikumiku37·Jaxcalibur·Nessie123: 휴먼 리플레이 0** (miku 명시, Jax "purely self-play", Nessie 자기대전). [25][30][W64]
→ 결론: 인간 데이터 활용은 학술 라인에서 검증됐고, 2026 커뮤니티 최상위 봇들은 **의도적으로 인간 데이터를 안 쓰는 중**(순수 자기대전) = 우리가 가진 786K+ 코퍼스는 "이미 검증된 대안 경로"의 재료.

---

## 8. 그들 대비 우리(2×V100)가 이길 수 있는 각도 — 전략 함의

> 전제: 우리 하드웨어(2×V100 32GB)는 5090/H100보다 약함(추정: fp16 FP16 텐서 기준 2×V100 ≈ 250 TFLOPS 이론 vs 5090 ≈ 400+, H100 ≈ 990). 따라서 "같은 방법으로 더 많은 게임"은 불리 — **샘플 효율·서치·팀·정보 축에서 이겨야 함.**

1. **샘플 효율 축 (우선순위 1)**: 1억 경기 목표(35일+) 폐기 대신, ①인간 리플레이 오프라인 RL 워밍스타트 ②소형 모델+고속 시뮬 조합으로 **1천만~3천만 게임 내 유효 성능** 목표. 근거: Metamon이 인간 1M 데모만으로 200M 모델 top10% 달성(서치 0), Abra/Kadabra3가 57M으로 Gen9 GXE 50→64. [10][11]
2. **검색 강도 축**: 8M급 추론은 V100에서 저렴(추정) → miku(16월드/1턴/무롤아웃)와 Jax(32월드/depth4)보다 **월드 수·깊이·롤아웃 예산을 한 단계 상향**한 테스트타임 서치가 곧바로 Elo 축(Jax +100~150, miku 서치 전환 효과 실증). 단 필요조건 = 인-프로세스 롤아웃이 가능한 시뮬 처리량 확보(현 32.5 games/s → 병목; miku는 7k b/s/코어·8,192 병렬). [25][30]
3. **팀 축**: 티프리뷰 선출(6→4)+팀풀 자기대전(메타게임)을 정책에 포함 — 공개 봇 웨이브 전원 미개척(§7-③), VGC-Bench가 팀 다양성 트레이드오프를 정식 연구 주제로 남겨둠. miku의 팀편집 약점·Jax randbats 한계와 정면 대비되는 차별점. [20][30]
4. **정보 축**: Jaxcalibur식 **상대 예측 보조헤드(다음 행동·아이템·기술)** 이식 + **OTS/CTS 혼합 학습**(Nessie의 OTS 편향 회피). 우리 OTS 시트 데이터(단판 OTS 2,885건·Bo3 ≈100% 시트 보존)와 observed-aux 파이프라인이 자산. [25][35][W-skill]
5. **하드웨어 상쇄 공식**: (모델 8M + 서치 소량 + 시뮬 고속) 조합이 miku의 실증 경로 — 5090 48h급 결과를 V100 2장에서 재현하려면 **시뮬 처리량을 수천 b/s급으로 올리는 것**이 최우선 공학 과제(우리 sim은 이미 Rust → 병렬화·관측 인코딩 병목 제거 여지). [30]
6. **운영·평판 축**: 세 봇 모두 ①비공개 ②래더는 검증용/제한 ③자기식별+챌린지 중심 운영 — Smogon/PS가 정책 성명을 준비 중이므로, 우리도 공개 범위(비공개/제한 공개)·자기식별·래더 사용 규율을 **선제적으로 설계**하는 것이 리스크 관리이자 신뢰 자산.
7. **벤치마크 기준선(제안)**: 단기 = 공개 봇 상대 승률(Metamon Kakuna/Abra·FoulPlay 등 미러 조건) / 중기 = M-C(또는 챔피언스) 래더 Elo 기준선 1857(miku)·OTS Bo3 #1(Nessie) / 장기 = "동일 성능 도달에 필요한 경기 수" 곡선(샘플 효율을 1급 지표로). [25][30][35]

---

## 9. 사실/추정 구분 · 실패한 검색 경로 · 출처

### 9.1 추정으로 표기한 항목
- miku/Jax/Nessie의 컴퓨트·경기 수 = **작성자 진술**(제3자 검증 불가, 650x·+100~150 Elo 포함).
- 2026 VGC AI 배틀 우승팀 접근법 = "SelfPlayFlat" 이름 기반 추정(셀프플레이 플랫 정책).
- V100 대비 성능비 = 스펙 기반 추정(워크로드 미실측).
- Nessie CTS 일반화 약점 = 설계 맥락상 추정(본인 명시 아님).
- miku 테레인/Kingambit 약점 = 커뮤니티 제보(미검증).
- "우리 미개척" 류 판단 = 공개 자료 스캔 기반(공개되지 않은 유사 연구가 있을 가능성 배제 못 함).

### 9.2 실패한 검색 경로 (기록)
- Medium 원문: curl 403 → **scrapling stealthy_fetch로 성공**(최종 확보).
- GitLab 위키 HTML: curl로 본문 빈 응답(JS 렌더) → **GitLab API(/wikis)로 성공**.
- Synapse 검색: "VGC AI Competition 2024/2025 winner" 등 — 대회 결과는 검색엔진 미인덱스, 위키 직접 조회로 해결. 2025 우승팀 상세 writeup은 끝내 미발견.
- "nessie_dev" X 계정: 검색 실패(동명 타 계정만) — Nessie X 활동 미확인.
- GitHub 검색 "mikumiku"·"nessie+pokemon": 무관 결과(MMD 플러그인 등)만 반환 — **해당 개발자 공개 리포 없음(그 자체가 조사 결과)**.
- Reddit `.rss` 직접 호출: 429/0바이트(1차) → pullpush·검색 스니펫으로 대체.
- IEEE Xplore(11114412 등): 페이월 추정으로 미시도 → SemanticScholar/ResearchGate 요약 대체.
- mikumiku38(2nd 계정, 10-03): 이번 라운드에서 독립 검증 안 됨(내부 스냅샷 2026-10-04 기록만).
- Jaxcalibur Bo9 스코어: 스레드 page-5 검색 스니펫 + freezai 트윗으로 확정(5-3) — 스레드 page-5 본문 직접 캡처는 미완(스니펫 의존).

### 9.3 출처
- [1] CoG 2026 Competitions — https://cog2026.org/competitions
- [2] CoG 2025 VGC AI Competition 페이지 — https://cog2025.inesc-id.pt/vgc-ai-competition/
- [3] VGC AI Framework 위키 홈 — https://gitlab.com/DracoStriker/pokemon-vgc-engine/-/wikis/home
- [4-A] 2021 논문(ACM DL) — https://dl.acm.org/doi/10.1109/cog52621.2021.9618985
- [4-B] 2023 에디션 위키 — https://gitlab.com/DracoStriker/pokemon-vgc-engine/-/wikis/VGC-AI-Competition/2023
- [5] 2024 에디션 위키 — https://gitlab.com/DracoStriker/pokemon-vgc-engine/-/wikis/VGC-AI-Competition/2024
- [6] 2025 에디션 위키 — https://gitlab.com/DracoStriker/pokemon-vgc-engine/-/wikis/VGC-AI-Competition/2025
- [7] 2026 에디션 위키(결과표) — https://gitlab.com/DracoStriker/pokemon-vgc-engine/-/wikis/VGC-AI-Competition/2026-Edition
- [7-B] Rules Balance 트랙 논문(IEEE CoG 2025) — https://ieeexplore.ieee.org/document/11114412
- [8] AurelianTactics 2024 3위 writeup — https://medium.com/@aureliantactics/vgc-ai-competition-2024-edition-3rd-place-submission-5420d2f6aafe
- [9] Metamon arXiv abs — https://arxiv.org/abs/2504.04395
- [10] Metamon 논문 HTML — https://arxiv.org/html/2504.04395v1
- [11] Metamon GitHub README — https://github.com/UT-Austin-RPL/metamon
- [12] Metamon 웹사이트 — https://metamon.tech/
- [13] HF 데이터셋(파싱 리플레이) — https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays
- [14] HF 모델(체크포인트) — https://huggingface.co/jakegrigsby/metamon
- [15] PokéLLMon arXiv abs — https://arxiv.org/abs/2402.01118
- [16] PokéLLMon 논문 HTML(v3, Table 1/4/7) — https://arxiv.org/html/2402.01118v3
- [17] PokéLLMon GitHub — https://github.com/git-disl/PokeLLMon
- [18] PokéLLMon 프로젝트 페이지 — https://poke-llm-on.github.io/
- [19] VGC-Bench arXiv abs — https://arxiv.org/abs/2506.10326
- [20] VGC-Bench 논문 HTML(v3, 인간평가 §5.4) — https://arxiv.org/html/2506.10326v3
- [21] VGC-Bench GitHub — https://github.com/cameronangliss/vgc-bench
- [22] PokéAgent Challenge 논문 HTML(2603.15563) — https://arxiv.org/html/2603.15563v1
- [23] PokéAgent Challenge arXiv abs — https://arxiv.org/abs/2603.15563
- [24] PokéAgent Challenge 사이트 — https://pokeagent.github.io/
- [25] Jaxcalibur 기술 writeup — https://jaxcalibur.github.io/
- [26] Jaxcalibur Smogon 스레드(1P, 비공개 결정 포함) — https://www.smogon.com/forums/threads/jaxcalibur-a-gen-9-randbats-bot-that-reached-1-on-the-ladder.3787537/
- [27] 같은 스레드 page-5(신버전 117-83, 5-3 결과 언급) — https://www.smogon.com/forums/threads/jaxcalibur-a-gen-9-randbats-bot-that-reached-1-on-the-ladder.3787537/page-5
- [28] 같은 스레드 page-6(전시전 회고·정책 준비) — https://www.smogon.com/forums/threads/jaxcalibur-a-gen-9-randbats-bot-that-reached-1-on-the-ladder.3787537/page-6
- [29] freezai X(5-3 결과) — https://x.com/freezaiYT/status/2106851666205004202
- [30] mikumiku37 Smogon 스레드 — https://www.smogon.com/forums/threads/mikumiku37-a-self-play-vgc-bot-reached-1-on-the-vgc-reg-m-c-ladder.3789199/
- [31] shaokthx X(10-03 발표) — https://x.com/shaokthx/status/2106149313466564815
- [32] shaokthx X(10-06 OTS 지원) — https://x.com/shaokthx/status/2107157493533266049
- [33] shaokthx X(10-08 더 강해짐+OTS 개선 예고) — https://x.com/shaokthx/status/2107894569853698544
- [34] shaokthx X(10-08 운영비 $150/월) — https://x.com/shaokthx/status/2107900349210313136
- [35] Nessie123 Smogon 스레드 — https://www.smogon.com/forums/threads/nessie123-an-ots-vgc-bot-that-topped-the-reg-m-c-bo3-ladder.3789213/
- [36] Foul Play GitHub — https://github.com/pmariglia/foul-play / poke-engine — https://github.com/pmariglia/poke-engine
- [37] Chronoquazicalar GitHub — https://github.com/TurboRx/Chronoquazicalar (HF: https://huggingface.co/TurboRx/Chronoquazicalar)
- [38] japlace GitHub — https://github.com/Xyqc/japlace
- [39] PokeChamp GitHub — https://github.com/sethkarten/pokechamp
- [40] poke-env GitHub — https://github.com/hsahovic/poke-env
- [41] pokemon-showdown GitHub — https://github.com/smogon/pokemon-showdown
- [42] pokeagent-solution GitHub — https://github.com/heatz123/pokeagent-solution / amago — https://github.com/UT-Austin-RPL/amago
- [43] Nolelle/pokemon-vgc-ai — https://github.com/Nolelle/pokemon-vgc-ai
- [44] philmantatsky/VGC-Pokemon-Showdown-AI — https://github.com/philmantatsky/VGC-Pokemon-Showdown-AI
- [45] gary5876/vgc-ai(한국, CoG2026 준비) — https://github.com/gary5876/vgc-ai
- [46] 기타 신생 리포(EmoMeza/poke-emo-bot, FYYHU/PokBot, DylanB03/pokemonBattlerML, HalluNasty/Elytrai, PriyadarshanPatil/pokemon-showdown-rl, alexmunzon/pokemon-showdown-ai) — GitHub 검색 API 2026-10-09
- [47] Reg M-C 기간(2026-09-09~12-02) — https://www.pokemon.com/us/news/get-ready-for-regulation-set-m-c-in-pokemon-champions
- [48] Smogon Technical Projects 포럼 — https://www.smogon.com/forums/forums/technical-projects.107/
- [49] LLM Pokémon League 논문(VGC AI Competition 인용) — https://arxiv.org/html/2508.01623v1
- [W64] 내부 스냅샷(2026-10-04, `~/Projects/MWL-PokeAgent/research/64-mikumiku37-analysis-2026-10-04.md` + 스킬 참조 vgc-bot-wave-2026-10.md) — Nessie·Jaxcalibur 세부 스펙 일부
- [W-skill] 내부 자산(스킬 pokemon-competitive-data: OTS/리플레이 코퍼스 786K+, 라이브 수집기 현황)

*보고서 끝. 핵심 사실 약 90개(표·불릿 기준), 재사용 가능한 신규 사실 ~60개.*
