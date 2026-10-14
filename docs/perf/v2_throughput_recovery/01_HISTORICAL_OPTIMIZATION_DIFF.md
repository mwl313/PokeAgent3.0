# V2 historical optimization comparison

Implementation base: `5ccc51b0cd4a49efb93018258b9750d5e804ae32`;
documentation parent: `c96c35f12d3f3aff32b072b48e32ac243a7b0425`.
Frozen production source remains untouched. This work uses the isolated
`optimization/pa3-v2-ppo-throughput` branch/worktree.

Historical evidence was read at these documented execution revisions:

| Evidence | Execution revision | Qualification |
|---|---|---|
| V2 frontier / learner breakdown | report lineage `b72f5f4` → realpolicy branch | single-seat historical figures; corrected in V3, not V2 semantic version 2 |
| V4 atlas | `83a7383695ccfee17f647250c5379312b6afe949` plus instrumentation; frontier `fbf878e124f35e9168aa0bf60b2662fa6f6b992e` | corrected both-seat accounting, old observation semantics |
| V5 micro promotion | `50c597ba56944ed62949b4e936f452cf57243e42` lineage | micro1024/default promoted; V6 report distinguishes V5 from same-session baseline |
| V6 final panel | `3fbc749c3d6c3298df52b62f4a43a9373c9c9bb0` | 3 repeats, 4 epochs, 2048 games; 28.84→37.23 all-in |
| A9.3 D1 | `6665024401660a307dfd760be48f7d6082e134d5` | PPO2, 18 steps/2048 games, 88.41 all-in, V1 semantics |
| A10 telemetry | `d10ac692009566686fd32285c4456309a5f383a7` | initial standalone utility; inspect active launcher integration separately |

Inspected actual local branch tips: `optimization/ppo-v6-throughput` at
`8997cb40c18481f96ba64b72d7480896510a9840`, realpolicy at `50c597b…`,
`research/a-ppo-efficiency` at `4ace3cf24426f141ff9cc1c4816d55c5fdc44f83`,
telemetry at `d10ac69…`. A branch name does not identify its old benchmark tip.
Read V6, V4 atlas, V2 frontier/breakdown, A9.2/A9.3/A9.4/A9.4.1/A9.4.2,
A10-P5 and current P2/P3/P4/frozen-contract documents.

| Technique and source location | V2 status before this change | Active setting / diagnosis | Action |
|---|---|---|---|
| Fixed native observation wire; `NativeCollector._collect_round` | PRESENT_AND_ACTIVE | fixed stride + uncapped five-family sidecar; P4 bulk decode removes quadratic scan | retain |
| Packed branch-conditioned candidate walk | PRESENT_AND_ACTIVE | `candidates_packed_walk`; no per-candidate ActionRef on hot path | retain native prefix legality |
| Flat signed decode/token mapping; `_level_table_packed`, `_packed_row_inputs`, `record_packed` | REGRESSED | P3 required correct signed targets/actual roster, implemented per row and repeated at actor/store boundaries | one immutable validated branch block, reused |
| Batched GPU inference and native candidate generation | PRESENT_AND_ACTIVE | group by policy/branch count; envs1024, workers16 | preserve grouping, order, RNG |
| Reduced candidate Python objects | PRESENT_BUT_INEFFECTIVE in part | typed ActionRef removed, but list/tuple round trip and repeated validation remain | remove duplicate canonicalization; retain stable tuple row API for now |
| Columnar observation store | PRESENT_BUT_DISABLED | `columnar_store=false`; V2 tests already prove five-family compatibility | no evidence why disabled beyond conservative release profile; do not assert incompatibility |
| Compact observation ownership / recording | REGRESSED in cost | Inline per-row dictionary + fixed double copy; more fields/masks/ragged allocations | own producer blocks once, gather by references |
| Streaming materialization | PRESENT_AND_ACTIVE | `prepare_streaming_ddp`; cached batch or per-minibatch fallback | retain memory fallback |
| Rolling slots | PRESENT_BUT_DISABLED | false; drained static restart contract and P3 partial-refill fix | do not enable |
| CUDA rollout caching | PRESENT_AND_ACTIVE | `learner._cache_plan`, 8GiB cap, micro activation reserve, approximately4GiB estimate first iteration | retain admission/fallback |
| Candidate padding compaction | PRESENT_AND_ACTIVE | compact=true; initial rollout 64→32 | retain masks and order |
| Microbatch sizing | PRESENT_AND_ACTIVE | global4096, local2048, micro1024, two accumulation micros | retain; micro2048 NEEDS_BENCHMARK |
| Exact row-weighted objective/DDP | PRESENT_AND_ACTIVE | actor/value global counts, no_sync around non-final forward/backward | retain unchanged |
| Candidate keys and unused final GRU removal | PRESENT_AND_ACTIVE | `PA3Model._walk_branches`, `return_final_hidden=False`, reusable keys | no duplicate implementation |
| Per-micro valid-row `.item()` removal | PRESENT_AND_ACTIVE | existing CPU selection / device slicing | retain |
| Cached contiguous selection | PRESENT_BUT_INEFFECTIVE | 0/54 initial hits; shuffled torch.randperm unchanged | no further work |
| Attention backend | PRESENT_AND_ACTIVE | V100-compatible SDPA with math fallback; no FlashAttention assumption | profile current kernels |
| Finite checks and clipping | PRESENT_AND_ACTIVE | V6 reused gradient norm; global overflow consensus remains | retain |
| Manual allreduce option | PRESENT_BUT_DISABLED | executor=ddp; historical flat FP32 SUM not faster | no NCCL/default change |
| Strict observation/action validation | new required V2 cost + NEEDS_BENCHMARK for synchronization | masks/category/ragged/scorer guards repeatedly inspect CUDA scalars | measure first; no unsafe permanent validated flag |
| Five ragged family embeddings | genuinely new required V2 work | family projection/reduction/count features, empty graph connectivity | preserve all families |
| Expanded known masks / public knowledge | genuinely new required V2 work | numeric/boolean knowns explicit; own/public repertoire distinct | preserve |
| Active→roster mapping | genuinely new correctness requirement | old slot=roster shortcut was wrong | batch map correctly; no clamping |
| Team statistics / generation accounting | PRESENT_AND_ACTIVE in campaign | path-bound original ledger, durable manifest reconciliation | experiment does not write campaign ledger |
| Deterministic CUDA | PRESENT_AND_ACTIVE | true, reduction-off; costs throughput but bounded restore proof | preserve |
| Continuous PPO telemetry | PRESENT_AND_ACTIVE | V2 launcher writes persistent JSONL, unlike initial d10 utility alone | retain |

V4 GPU busy95% was a particular old profiled window, not proof of today's
hardware ceiling. A9 ~90 games/s had fewer rows and optimizer steps per game.
Current policy progress itself changes game length. All future comparisons
report rows/game, rows/s and actual optimizer work beside games/s.
