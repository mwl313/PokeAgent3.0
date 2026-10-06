# PokeAgent 3.0

PokeAgent 3.0 aims to train a strong Pokémon VGC doubles policy through a full-scope Rust battle engine, scratch self-play reinforcement learning, and evaluation. The supplied Full Spec 1.1 defines the intended design.

**Current status: the combined M-B/M-C pool is normalized, validated, deduplicated and frozen for the project.** It contains **1,136 eligible unique teams, all assigned to training with no dev/final holdouts**. The 1,207 original source records are preserved; 56 records are quarantined and 15 accepted duplicates merged. See [the fixed team pool](docs/TEAM_POOL.md) for rules, artifacts and verification. M-A and further collection are excluded by the user's source selection. Battle engine and model implementation have not started.

## Project reference

The original ZIP remains at the project root. Its ten files are preserved byte-for-byte in [the versioned package](docs/spec/fullspec-1.1-minidc-20261006/README.md).

- [Full specification](docs/spec/fullspec-1.1-minidc-20261006/POKEAGENT_3_0_FULL_SPEC.md): complete intended scope and design contracts.
- [Training configuration](configs/train.yaml): active config with the user-selected frozen M-B/M-C pool, actual counts, split paths and manifest hash.
- [Accepted team record schema](data/schemas/team-record.schema.json): relative symlink to the supplied JSON Schema, unchanged.
- [Runtime changes](docs/spec/fullspec-1.1-minidc-20261006/RUNTIME_CHANGES.md): version 1.1 changes and their limits.
- [Hardware reference](docs/spec/fullspec-1.1-minidc-20261006/reference/73-v100-driver-cuda-constraints.md): supplied equipment report, not a fresh measurement.
- [Preparation receipt](docs/workspace-preparation.json): archive hash, file hashes, checks, and preserved pre-existing deletions.

The package's internal paths retain their original meaning relative to the package directory. The active config is now a separate file; the original YAML remains unchanged. Schema and runtime convenience links remain, and runtime settings have not been sourced or executed.

## Intended system

| Area | Supplied design |
|---|---|
| Battle format | `gen9championsvgc2026regmc`, closed team sheets, best of one, bring four of six with two active slots |
| Rules reference | Pokémon Showdown commit `14546894d86f9589ac11130c510bbe73b6968665`; built and used for team validation and initial stats |
| Teams | Frozen eligible teams from the two collected M-B/M-C batches, legal in M-C; no invented sets, M-A additions or automatic refresh |
| Split | All 1,136 eligible teams train by user request; roster groups retained for analysis; zero held-out teams |
| Engine | Rust with a PyO3 batch interface; implement the effect dependencies of the entire eligible inventory |
| Policy | One randomly initialized PA3-8M Entity Transformer: six layers, width 320, five attention heads; target total 8–10 million parameters |
| Observation | Player-visible structured state, 88 tokens padded to 96, persistent revealed knowledge and explicit unknown masks |
| Actions | Preview selection and leads, complete move/target/resource choices, voluntary switches, forced and pivot replacements |
| Learning | PPO with GAE, terminal win/loss/draw rewards of +1/−1/0, four epochs, global minibatch 4,096 |
| Collection | 2,048 environments and 32 CPU workers; freeze policy versions during each collection iteration and drain games naturally before updating |
| Opponents | 50% current policy and 50% historical policies, up to eight; use current policy alone until history exists |
| Experience target | 100 million naturally completed training matches, counted once per match, followed by processing the final experience |
| Evaluation | Fixed diagnostics every 1 million matches and Showdown transfer every 10 million; final evaluation uses training teams, without unseen-team claims |

The full train pool is used from the first training reset. Human-action imitation, pretrained weights, model architecture sweeps, small-team curricula, search, and external LLM battle decisions are outside the specified training approach.

## Important implementation contracts

Closed team sheets require information boundaries in both observations and legal-action masks. The actor must not receive hidden opponent sets, true hidden stats, internal team identities, or unsubmitted opponent actions. The value head uses the same player-visible observation and does not receive the selected action prefix.

Champions Stat Points must retain their source units and provenance. An `evs` field name does not establish ordinary EV units. The accepted-record schema excludes unknown units; ambiguous or incomplete teams belong outside the accepted pool. JSON Schema validation alone does not establish battle legality.

Training rows represent complete player requests, with joint log-probabilities across conditional action branches. Historical opponent actions are excluded from current-policy learning. Operational errors and timeouts are not natural draws, and evaluation games do not count toward the training target. Collection overshoot is retained; the final cohort respects the remaining match budget.

## Runtime design

The package pins Python 3.12.3, PyTorch `2.14.0+cu126` with CUDA runtime 12.6, NVIDIA driver 580.178.04, and system CUDA Toolkit 12.8.2. These are supplied requirements and reported values; this preparation did not inspect the live hardware or verify wheel availability.

The intended hardware is two V100 PCIe 32GB GPUs at 175W and 150W, assigned to NUMA nodes 0 and 1. Host-staged NCCL DDP uses `256 × 8 × 2 = 4,096` effective rows. The documented GPU0 learner fallback preserves that batch with `256 × 16`. Execution uses eager PyTorch, FP16/FP32, and SDPA math.

[Runtime constraints](runtime/constraints-minidc.txt) pin torch only; they are not a complete dependency lock. [Process environment settings](runtime/minidc.env) do not enforce the future trainer's attention, precision, or NUMA behavior by themselves. Driver, Toolkit, power limits, protected services, and package holds remain unchanged.

## Directory layout

| Path | Purpose and present contents |
|---|---|
| `engine/` | Reserved for the Rust engine and reference adapter; empty placeholder |
| `agent/` | Reserved for model, PPO, collector, and evaluator; empty placeholder |
| `configs/` | Active `train.yaml` pins the prepared team pool |
| `data/schemas/` | Accepted team schema link; collected raw records remain separate from this legality-validated schema |
| `data/raw/vgcpastes/champions-mc/20261006/` | M-C sheet snapshot, 424 source pastes and extracted teams, provenance and extraction checks |
| `data/raw/vgcpastes/champions-mb/20261006/` | M-B sheet snapshot, 782 distinct source pastes and 783 extracted team records, provenance and extraction checks |
| `data/teams/mb-mc-v2-all-train/` | Frozen accepted teams all in train, empty dev/final files, quarantine, provenance, inventory and hashes |
| `scripts/` | Pinned reference setup and reproducible team preparation |
| `tests/` | Eight data-preparation regression tests |
| `vendor/pokemon-showdown/` | Ignored, pinned reference checkout and local build |
| `runtime/` | Links to the two supplied runtime reference files |
| `docs/spec/fullspec-1.1-minidc-20261006/` | Complete unchanged source package |
| `runs/` | Empty local output directory; already ignored by Git |

Git history was reset at the user's request to establish this prepared project as the initial commit. The two raw team batches, both prepared snapshots, configuration, source documents and preparation code are tracked locally. Earlier deleted project documents remain absent. Dependencies, the reproducible reference checkout, temporary files and future training outputs remain ignored.

## M-C team collection

The requested M-C sheet contained 463 team rows: 424 marked EVs Yes and 39 marked No. All 424 selected Poképastes were fetched and extracted successfully. All contain six Pokémon with four moves and nature lines. Of these, 423 contain explicit allocations for all six members; MC98 contains no allocation lines despite its sheet label. Two pairs have byte-identical extracted team text: MC245/MC189 and MC35/MC26. All source records are retained.

- [Poképaste links](data/raw/vgcpastes/champions-mc/20261006/pokepaste_links.txt)
- [Combined team text](data/raw/vgcpastes/champions-mc/20261006/all_teams.txt)
- [All extracted records](data/raw/vgcpastes/champions-mc/20261006/teams.jsonl)
- [Records with six explicit allocations](data/raw/vgcpastes/champions-mc/20261006/teams_with_allocations.jsonl)
- [Collection receipt](docs/20261006-mc-team-collection.json)

Original HTML, individual team files, sheet rows, URLs, timestamps and hashes are retained. Independent reconstruction verified all 424 extracted texts against their saved HTML. These unchanged raw records feed the separate validated pool described above. M-C rows marked No were outside the collection request. The approved dated raw snapshots are tracked in local Git; other raw-data directories remain ignored by default.

## M-B team collection

The subsequent M-B collection covered all 783 rows marked EVs Yes, out of 861 team rows. These point to 782 distinct Poképastes: MB843 and MB791 share one link. All 783 source records were saved with no remaining fetch failures. The HTTP source link for MB539 was retrieved over HTTPS while retaining its original URL in the source record. M-A teams were not collected.

All records contain six Pokémon. There are 782 records with explicit allocations for all six; MB382 omits the allocation for Ditto. Sixteen records have a member with fewer than four listed moves, and three have a missing nature line. These categories overlap, giving 17 records with completeness flags. Short movesets include Ditto and Last Resort sets and are preserved as written; these flags are not a battle-legality judgment. Ten pairs have identical extracted team text, leaving 773 distinct exact texts. No source records were removed.

- [M-B Poképaste links](data/raw/vgcpastes/champions-mb/20261006/pokepaste_links.txt)
- [M-B combined team text](data/raw/vgcpastes/champions-mb/20261006/all_teams.txt)
- [All M-B extracted records](data/raw/vgcpastes/champions-mb/20261006/teams.jsonl)
- [M-B records with six explicit allocations](data/raw/vgcpastes/champions-mb/20261006/teams_with_allocations.jsonl)
- [M-B collection receipt and flagged IDs](docs/20261006-mb-team-collection.json)

Independent reconstruction and hashes verified every saved M-B team against its source HTML. Both original batches remain unchanged. Together they contain 1,207 source team records, including 1,205 with allocations for all six members. The combined prepared pool applies the additional documented eligibility checks, deduplication and the all-training assignment.

## Values to establish during implementation

Team counts, source hashes, roster groups and split assignments are now frozen. Full effect closure, tensor vocabularies, maximum action capacity and exact model parameter count remain engine/model implementation work. The Rust toolchain and complete training dependency lock are also unresolved. GPU runtime availability, memory headroom, DDP transport and end-to-end throughput require actual checks when that work is requested.

The package's 1,000-team aspiration and 500–1,000 completed games per second goal are not measured results. With the all-training override, final evaluation measures policy progress on training teams and engine transfer. Unseen-team generalization and external competitive strength require separate evidence.
