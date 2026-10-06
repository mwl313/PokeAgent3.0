# Fixed M-B and M-C team pool

The project now uses **mb-mc-v2-all-train**, prepared from the two collected batches selected by the user on 2026-10-06. This selection replaces the original specification's broader M-A/M-B/M-C collection scope. No M-A teams, new sources, additional crawls, or automatic inventory refreshes are part of the active pool.

The 1,207 original source records remain intact. Applying Full Spec 1.1's completeness requirements and the pinned M-C validator yields 1,151 accepted source records, which collapse to **1,136 unique teams**. The other **56 source records are quarantined**, with their original sets and exact reasons preserved. No missing set details were invented.

| Split | Unique teams | Roster groups |
|---|---:|---:|
| Train | 1,136 | 903 |
| Dev | 0 | 0 |
| Final | 0 | 0 |
| Total | 1,136 | 903 |

The user's subsequent instruction assigns every eligible team to training and supersedes the specification's 90/5/5 split. All 1,136 teams are sampled uniformly from the first training reset. Dev and final dataset files are empty. The earlier `mb-mc-v1` grouped-holdout snapshot is preserved for provenance and is no longer active.

## Dataset artifacts

The active configuration is [configs/train.yaml](../configs/train.yaml). It pins the manifest's SHA-256, the three split paths, the actual counts, and the M-B/M-C-only source scope. The original specification ZIP and extracted YAML are unchanged.

| Artifact | Purpose |
|---|---|
| [Manifest](../data/teams/mb-mc-v2-all-train/manifest.json) | Frozen source hashes, counts, split rules, output hashes and dataset identity |
| [All accepted teams](../data/teams/mb-mc-v2-all-train/all.jsonl) | Normalized records conforming to the original `pa3-team-v1` schema |
| [Train](../data/teams/mb-mc-v2-all-train/train.jsonl), [dev](../data/teams/mb-mc-v2-all-train/dev.jsonl), [final](../data/teams/mb-mc-v2-all-train/final.jsonl) | All accepted records in train; empty dev/final files retained for an explicit, verifiable zero-holdout contract |
| [Train simulator sets](../data/teams/mb-mc-v2-all-train/train.showdown.jsonl) | Canonical full team sets for simulator reset; analogous dev/final files are alongside it |
| [Team text](../data/teams/mb-mc-v2-all-train/all_teams.txt) | Readable normalized exports |
| [Source index](../data/teams/mb-mc-v2-all-train/source-index.jsonl) | Disposition of every one of the 1,207 source records |
| [Duplicates](../data/teams/mb-mc-v2-all-train/duplicates.json) | Fifteen redundant source records merged into their matching canonical teams |
| [Quarantine](../data/teams/mb-mc-v2-all-train/quarantine.jsonl) | Full original records and reasons for exclusion from the active pool |
| [Quarantine review](team-pool-review.json) | Compact source IDs, URLs and specific issues |
| [Groups](../data/teams/mb-mc-v2-all-train/groups.json) | Roster group membership; every group assigned to train |
| [Inventory](../data/teams/mb-mc-v2-all-train/inventory.json) | Direct species, resource-form, move, item, ability and nature IDs for the whole training pool |

The approved raw batches and both prepared snapshots are tracked in the local initial Git commit, along with the [manifest copy](team-pool-manifest.json), preparation code, configuration and documentation. The reference checkout, dependencies and temporary files remain ignored. No remote publication is part of this save. Public accessibility is retained as provenance, not treated as permission to redistribute the pastes.

## Normalization and reference validation

All accepted teams pass Pokémon Showdown's `gen9championsvgc2026regmc` validator at commit `14546894d86f9589ac11130c510bbe73b6968665`. The [reference build record](pokemon-showdown-reference.lock.json) identifies the checkout, lockfile, source hashes and local Node/esbuild versions. The reference is used locally; no Showdown server, public ladder, GPU work or model training was started.

Champions allocation values are interpreted as Stat Points using their Champions source context and the pinned validator: 32 maximum per stat and 66 total. Original allocation lines are retained. Omitted stats within an explicit line become zero according to the importer; an absent allocation line is never filled in. No legacy EV conversion was performed.

Canonical records retain base species, items, abilities, natures, four moves, allocations, effective level, IVs, gender semantics and official form mappings. Level 50 adjustment and omitted IV defaults are recorded. An unspecified gender remains the reference's per-reset random choice when the species does not fix it; the seed used to inspect starting stats does not freeze that gender into the dataset.

Starting stats come from real reference Pokémon objects at team preview, before lead switch-in effects, boosts or Mega Evolution. The reference permits Mega Evolution and disables Terastallization in this format. The base format offers optional open team sheets; no acceptance is sent, and the active project setting remains CTS/BO1. Full private team records belong to the simulator. Future actor observations must still enforce the player's information boundary and exclude opponent hidden sets and source identifiers.

Each member retains its source text and each deduplicated team retains every source ID, URL, timestamp, hash and normalization log. Cosmetic names, roster order and move order do not affect team identity. Source shiny/ball details and disabled Tera annotations do not create strategically different teams.

## Quarantine decisions

| Issue | Affected source records |
|---|---:|
| Mega-form ability leaves multiple possible pre-Mega abilities | 32 |
| Fewer than four explicitly listed moves | 16 |
| Missing explicit item or explicit no-item declaration | 3 |
| Missing explicit nature | 3 |
| Missing allocation lines | 2 |
| Pinned reference validator rejects the set | 5 |

These categories overlap; their sum is not the 56-record quarantine total. Three missing-nature records also trigger the guard against the validator silently inserting a neutral nature.

Showdown can replace a listed Mega ability with the first base-form ability even when other base abilities exist. The documented no-imputation rule does not permit guessing which one the author intended, so those records remain quarantined. Where the reference determines a unique base ability, that official normalization is recorded and accepted.

Ditto and Last Resort sets may intentionally have fewer than four moves. They are preserved without additions, but the active Full Spec 1.1 and team schema explicitly require four moves per member. These are completeness exclusions, not blanket claims that the sets are illegal.

The legality failures include MB495 (Grimmsnarl with Scratch), MB135 (Scrafty with Infiltrator), MC214 and MC213 (Archaludon with Precipice Blades). MB382 additionally receives the reference's zero-investment warning after its absent Ditto allocation is parsed. MC98 lacks all six allocation lines. The original records are available for review; no replacements were generated.

## Grouping and reproducibility

Exact team fingerprints hash canonical strategic sets while ignoring roster/move order and nicknames. Group fingerprints hash the unordered six reference species/form identities, including item-enabled Mega forms. Ordinary moveset, spread and non-transforming item changes remain in the same group. Distinct Mega forms remain distinguishable. Groups remain useful for analysis and provenance; every group now participates in training.

Unique groups retain stable ordering by `SHA256("20261006:" + group_id)`, but all 903 groups are assigned to train. Periodic and final policy evaluations use fixed matchup/seed panels drawn from the training pool. They can measure policy progress and reference-engine transfer, but cannot establish generalization to unseen teams. The existing `best_dev` checkpoint label is retained for compatibility and now means the best checkpoint on these training-pool diagnostics; it does not indicate a held-out dev set.

From the repository root, using the existing Node.js installation:

```bash
bash scripts/setup_reference.sh
npm run prepare:teams
npm test
```

The setup script installs only local reference dependencies from its pinned lockfile and builds the reference. It does not install Python/torch or change the host stack. The preparation step reads only the two saved batches, verifies their source hashes, and refuses to replace an existing frozen pool with different bytes. Any future approved corrections require an explicit new dataset version and manifest update.

Verification passed: all 1,136 accepted records satisfy the supplied JSON Schema; all remain reference-legal after normalization and pack/unpack roundtrips; every source record is accounted for; all eligible records and groups are assigned to train with zero dev/final records; starting-stat and information-mode checks pass; and rebuilding reproduces the same files byte-for-byte. Eight automated preparation tests passed.

## Engine handoff

The direct inventory covers 194 base species/forms, 72 resource forms, 344 moves, 134 items, 139 base abilities, 52 resource-form abilities and 18 natures. Every eligible team contributes to both engine support requirements and the training pool. Evaluation episodes still remain separate from training match counters.

The next engine work must resolve and implement the complete dependencies and interactions of these effects, including called moves such as Sleep Talk. The inventory is a frozen input to that work, not a claim that the Rust engine or full effect closure has been implemented. Engine difficulty must not be used to remove accepted teams.
