# Observation V2 — required semantic contract

## A. Core shape and role table

Preserve a fixed 96-token interface if correct and feasible:

| Token indices (zero-based) | Count | Role | Source |
|---|---:|---|---|
| 0 | 1 | Global | public request/format/own resource |
| 1 | 1 | Field | Knowledge.field plus dynamic effects |
| 2–3 | 2 | Side | own/opponent public side knowledge |
| 4–15 | 12 | Pokémon | own 6 and known opponent preview 6 |
| 16–63 | 48 | Move | 4 slots for each of 12 Pokémon |
| 64–87 | 24 | Event | most recent publicly visible semantic events |
| 88–95 | 8 | masked padding | never model-visible |

Existing fixed token feature slots: 32 category, 50 continuous, 40 boolean plus known masks. Existing parser delivers separate ragged groups: `effects`, `repertoire`, `types`, `base_moves`, `move_effects`. **The 96-token limit is not a 96-feature ceiling.** Preserve per-token ragged offsets, lengths and known visibility.

## B. Semantic source of truth

- `World`: the physical battle engine state, allowed ONLY for mechanics and own-team information the player already knows.
- `OwnSideView`: own Pokémon true known stats, moves, types, selection order, active roster mapping and own resources.
- `PlayerView.request`: fields the player is legitimately served, using public semantics rather than internal flags.
- `Knowledge`: opponent preview/public reveals, effects, publicly visible event ledger, exposed HP display and hidden/known provenance.
- `Dex`: deterministic mechanics derived from *already public* IDs and rule set; never infer hidden move IDs from a private opponent set.

Player-side observations must not expose private opponent moves, set allocations, IV/Stat Points, abilities/items not revealed, pending choices, RNG or true identity masked by Illusion.

## C. Ragged families: producer, decoder and neural consumer must agree

| Family | Data examples | Requirements |
|---|---|---|
| `effects` | kind/id, duration, stacks, effect source, known flags | current presence and remaining duration, never silently truncate; both temporal and public semantics |
| `repertoire` | opponent revealed move IDs beyond current 4 | preserve reveal history without assuming actual private slot order |
| `types` | 0–N dynamic types, placeholder type semantics | current player-visible types; distinguish unknown from empty/typeless; no silent clipping |
| `base_moves` | original/current move data where legitimately known | own full known set, opponent public-only; transform effects treated separately |
| `move_effects` | hit/secondary effects and associated metadata | deterministic Dex metadata based on known move ID; correct effect kind/namespace; dynamic effects handled |

One option: `(TokenEmbedding + typed ragged embeddings + masked sum/mean/attention pooling) → 320d tokens`. A different design is permitted if it demonstrably preserves all legal rows and V100 memory/performance is acceptable. **Do not declare success when raw ragged bytes exist only in a Python dictionary or rollout buffer.** Prove changing a permitted dynamic effect changes the encoder output in a controlled fixture.

Avoid fixed arbitrary small caps; if caps exist, derive proven contract bounds with explicit overflow rejection, never silent truncation. Use per-row count/offset validation, including empty groups.

## D. Known masks

Categorical values: distinguish unknown, known special-zero and valid IDs without embedding collision. Numeric and boolean values: explicitly append or otherwise encode known masks (not merely multiply by them). Include known flag state in serialized observations and neural input. Differentiate known=false vs known=true,value=0. Dynamic masks must also be model-visible.

## E. Numeric scales and FP16 readiness

Before modifying normalization, auto-generate a complete CSV of the 50 continuous features with: name, semantic units, legal min/max, effective range in frozen teams, normalization mapping and inverse, Mode D `half().float()` minimum useful step, underflow collisions, default/missing representation.

Release blockers:

- `EventChange` ±1 and ±6 distinct, including negative vs positive; +/- very small HP changes preserved when allowed.
- Weight values associated with distinct legal Pokémon remain meaningful (small weights cannot all underflow to zero).
- Priority, accuracy, PP, boost, duration and turn distinguish required states.
- No NaN, +/-Inf, silent clip or uncontrolled saturation.
- Do not accidentally change simulation stat or damage values while changing model normalization.
- Unknown values never become fabricated zero observations. All real zero values remain distinguishable.

Quantization should be deterministic for Actor and Learner. Mode D must quantize at the **same versioned stage** in both. If precision strategy changes, it is a new execution contract and requires evidence; do not disguise as same V1 mode.

## F. Stable identity and event ordering

The owner has six roster identities and two active positions; they differ after selection/switch. Opponent team identities are public preview entities but privacy rules govern current moves, item/ability, identity and transformations. Preserve stable per-match entity IDs across switches; do not associate events with field position alone. Semantically ordered latest 24 events, with clear overflow/drop-oldest and no future leakage; distinguish filtered private integer from known zero.

Before every sample, assert the observation corresponds to the latest **committed public decision state**; no events from the action about to be chosen or from a future queued action. Fail closed when a required input field is missing.

## G. Serialization and version hash

One frozen schema ID, e.g. `pa3-observation-v2` plus a cryptographic semantic contract hash, must be included in:

- Rust fixed wire header and ragged counts/metadata;
- Python `ObservationBatch` (including slices/concat and in-flight microbatch materialization);
- `to_compact_numpy` / `from_compact_numpy`, SoA, rollout checkpoints;
- Actor/Learner/Evaluator model construction and recompute snapshots;
- Model weights and Checkpoint Generation manifests;
- DDP rank receipt/activation report.

Mismatch, missing required keys or an unsupported old schema must produce a hard failure with diagnostic paths. Non-actor metadata such as team provenance must remain explicitly excluded from observation.

## H. Acceptance

Run `T-OBS-01..06` and `T-OPS-01`. Include exact tensor hashes and feature-level comparisons for each tested state. At least one controlled positive effect injection per ragged family and a paired hidden-only adversarial test are mandatory. Do not overinterpret network output equality caused by random weights: the **model-visible representation** must be checked directly.
