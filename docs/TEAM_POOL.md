# Fixed M-B and M-C team pool

The project now uses **mb-mc-v3-userteam-all-train**. It contains every team of the previous frozen pool plus exactly one manual source: the user's own Poképaste, explicitly approved for training on 2026-10-08. The user selected the two 2026-10-06 M-B/M-C batches as the lifetime collection scope; no M-A teams, crawls, discovery or automatic inventory refreshes are part of the active pool, and no other manual team has been approved.

The predecessor **mb-mc-v2-all-train** (1,136 teams) is preserved unchanged. Rebuilding v3 verifies every predecessor record byte-for-byte before writing, so the 1,136 original teams are carried over without alteration. The earlier **mb-mc-v1** grouped-holdout snapshot is also preserved for provenance.

| Dataset | Teams | Groups | Sources | Status |
|---|---:|---:|---:|---|
| `mb-mc-v1` | 1,136 | 903 | 1,207 source rows | preserved provenance only |
| `mb-mc-v2-all-train` | 1,136 | 903 | 1,207 source rows | frozen predecessor, immutable |
| `mb-mc-v3-userteam-all-train` | 1,137 | 904 | 1,208 source rows | **active** |

## The 2026-10-08 manual source

The single approved submission is `https://pokepast.es/c2cfbd453aa9172e` (paste title `리자몽`). The raw HTML and raw `/raw` text were fetched on 2026-10-08 and stored under `data/raw/user-pokepaste/20261008/` with their hashes and fetch headers. `scripts/import_user_pokepaste.mjs` re-derives the extraction record from those saved artifacts, cross-checks the HTML against the raw text, and records the pinned reference's verdict; `scripts/prepare_teams.mjs` refuses a stale or hand-edited record. Source id: `UT20261008`.

The team is Charizard-Mega-Y / Whimsicott / Gengar / Sneasler / Indeedee / Garchomp with 66 Champions Stat Points on every member (each stat ≤ 32), four explicit moves each, explicit items and natures. The pinned `gen9championsvgc2026regmc` validator accepts it as written. Its canonical fingerprint and roster group match no team in the predecessor pool, so it is a genuinely new training team and the pool grows from 1,136 to **1,137 unique teams (904 roster groups)**.

**Documented ability normalization.** The source declares `Charizard-Mega-Y` with `Ability: Drought` (the Mega forme's own ability). The pinned reference rewrites the set to `Charizard` and fills the base-form ability with the first legal one, `Blaze`; Charizard's base abilities are `Blaze` and `Solar Power`, so the author's intended pre-Mega ability is not determined by the source. Nothing else about the set changes: species, item, nature, four moves, all 66 Stat Points, level adjustment and IVs are preserved exactly, and the Mega forme itself is preserved through the Charizardite Y mapping. The dataset records both values (`policy: reference_validator_base_form_default`, `source_value: Drought`, `reference_value: Blaze`) in the team's `eligibility.mega_ability_resolutions`, in its `source.records[].ability_resolutions` and in the manifest's `documented_normalizations`, with `user_confirmation_required: true`. Mega-form ability defaults stay quarantined for every crawled batch; only this one approved submission may use the resolution.

## Split policy

All 1,137 unique teams are training teams. Dev and final files stay empty. The user's instruction from 2026-10-06 assigns every eligible team to training and supersedes the specification's 90/5/5 split, and the 2026-10-08 submission is added as a normal uniformly-sampled team: it receives no extra weight, no special schedule and no evaluation-only status.

The 1,207 original source records remain intact. Applying Full Spec 1.1's completeness requirements and the pinned M-C validator yields 1,151 accepted source records, which collapse to **1,136 unique teams**. The other **56 source records are quarantined**, with their original sets and exact reasons preserved. No missing set details were invented.

| Split | Unique teams | Roster groups |
|---|---:|---:|
| Train | 1,137 | 904 |
| Dev | 0 | 0 |
| Final | 0 | 0 |
| Total | 1,137 | 904 |

The v3 manifest's `predecessor_verification` block records that all 1,136 predecessor records were verified unchanged and that exactly one team was added. The 1,208 raw source rows are accounted for as 1,152 accepted and 56 quarantined; the manual submission is the only accepted record that is not from the two frozen batches.

## Dataset artifacts

The active configuration is [configs/train.yaml](../configs/train.yaml). It pins the v3 manifest's SHA-256, the three split paths, the actual counts, the predecessor manifest hash, and the single manual-source exception. The original specification ZIP and extracted YAML are unchanged.

| Artifact | Purpose |
|---|---|
| [Manifest](../data/teams/mb-mc-v3-userteam-all-train/manifest.json) | Frozen source hashes, counts, split rules, output hashes, dataset identity, predecessor verification and documented normalizations |
| [All accepted teams](../data/teams/mb-mc-v3-userteam-all-train/all.jsonl) | Normalized records conforming to `pa3-team-v1` (VGCPastes) or `pa3-team-v2` (manual submission); see [the v2 schema](../data/schemas/team-record-v2.schema.json) |
| [Train](../data/teams/mb-mc-v3-userteam-all-train/train.jsonl), [dev](../data/teams/mb-mc-v3-userteam-all-train/dev.jsonl), [final](../data/teams/mb-mc-v3-userteam-all-train/final.jsonl) | All accepted records in train; empty dev/final files retained for an explicit, verifiable zero-holdout contract |
| [Train simulator sets](../data/teams/mb-mc-v3-userteam-all-train/train.showdown.jsonl) | Canonical full team sets for simulator reset; analogous dev/final files are alongside it |
| [Team text](../data/teams/mb-mc-v3-userteam-all-train/all_teams.txt) | Readable normalized exports |
| [Source index](../data/teams/mb-mc-v3-userteam-all-train/source-index.jsonl) | Disposition of all 1,208 source records including `UT20261008` |
| [Duplicates](../data/teams/mb-mc-v3-userteam-all-train/duplicates.json) | Fifteen redundant source records merged into their matching canonical teams |
| [Quarantine](../data/teams/mb-mc-v3-userteam-all-train/quarantine.jsonl) | Full original records and reasons for exclusion from the active pool |
| [Quarantine review](team-pool-review.json) | Compact source IDs, URLs and specific issues |
| [Groups](../data/teams/mb-mc-v3-userteam-all-train/groups.json) | Roster group membership; every group assigned to train |
| [Inventory](../data/teams/mb-mc-v3-userteam-all-train/inventory.json) | Direct species, resource-form, move, item, ability and nature IDs for the whole training pool |
| [Manual source](../data/raw/user-pokepaste/20261008/summary.json) | The approved Poképaste's saved artifacts, hashes and pinned-reference verdict |
| [v2 schema](../data/schemas/team-record-v2.schema.json) | Derived `pa3-team-v2` contract: v1 plus the one manual provider; the supplied v1 schema is untouched |

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
npm run prepare:teams        # rebuilds the frozen v2 snapshot and verifies it byte-for-byte
node scripts/import_user_pokepaste.mjs
node scripts/prepare_teams.mjs --dataset=mb-mc-v3-userteam-all-train
npm test
```

The setup script installs only local reference dependencies from its pinned lockfile and builds the reference. It does not install Python/torch or change the host stack. The preparation step reads only the two saved batches plus the one saved manual submission, verifies their source hashes, and refuses to replace an existing frozen pool with different bytes. Any future approved corrections require an explicit new dataset version and manifest update.

Verification passed: all v3 accepted records satisfy their declared contract (`pa3-team-v1` for VGCPastes, `pa3-team-v2` for the manual source); all remain reference-legal after normalization and pack/unpack roundtrips; every source record is accounted for; all eligible records and groups are assigned to train with zero dev/final records; the 1,136 predecessor records are byte-identical; starting-stat and information-mode checks pass; and rebuilding v2 reproduces the same files byte-for-byte. Eleven automated preparation tests pass.

## Engine handoff

The v3 direct inventory is unchanged from v2: the manual team introduces no new species, resource form, move, item or ability. It covers 194 base species/forms, 72 resource forms, 344 moves, 134 items, 139 base abilities, 52 resource-form abilities and 18 natures. Every eligible team contributes to both engine support requirements and the training pool. Evaluation episodes still remain separate from training match counters.

The next engine work must resolve and implement the complete dependencies and interactions of these effects, including called moves such as Sleep Talk. The inventory is a frozen input to that work, not a claim that the Rust engine or full effect closure has been implemented. Engine difficulty must not be used to remove accepted teams.
