# Audit verdict and source-evidence ledger

## Immutable source baseline

- Primary remote branch head: `d10ac692009566686fd32285c4456309a5f383a7` (`codex/a10-ppo-training-telemetry`).
- Remote `main` inspected: `50c597ba56944ed62949b4e936f452cf57243e42`.
- Actual MiniDC recovery commits reported: `266bb65`, `3df5abd` **not independently inspected**. Phase 0 MUST reconcile their ancestry, diff and executable code with the audit baseline.
- Code links below use the primary branch head and are for navigational context, not proof of executed behavior.
- `engine/data/manifest.json` at this baseline still records the legacy dataset `mb-mc-v3-userteam-all-train` (1,137). New A10 frozen manifest `data/teams/a10-p0-proposed-20261013/manifest.json` records 1,220 training teams, 970 roster groups, no dev/final heldout. `configs/validated_a94_runtime.yaml` also records 1,137. This is a **configuration drift risk**, not proof of which dataset actually ran.

## Severity definitions

P0 = must reproduce/resolve and pass acceptance before scratch PPO. P1 = necessary correction or justified sign-off, can be release-blocking when it affects the chosen production path. P2 = operational improvement, nonblocking absent a demonstrated critical defect.

## Source-confirmed discrepancies

### OBS-01 Ragged features silently dropped (P0, CODE-CONFIRMED)

`engine/src/observation.rs` generates fixed tokens plus effect, repertoire, types, base_moves and move_effects append-only sections. `engine/python/pa3_engine/observation.py::parse_batch()` decodes them, but `agent/types/observation.py::from_native_payload()` constructs `ObservationBatch` from fixed `token_mask, categories, category_known, floats, float_known, flags, flag_known` with roles/sides only. `agent/model/embeddings.py::TokenEmbedding` consumes only those fixed tensors. `agent/train/native_collector.py` and its rollout storage use that adapter. Consequence: the five sections are not model-visible in the reviewed production path. **Qualification:** some related facts may also appear via fixed features/events; the full strategic impact must be measured, not guessed.

### ACT-01 Action actor-index aliasing (P0, CODE-CONFIRMED)

`engine/src/actions.rs` creates MOVE/SWITCH actions with `own_slot = active battlefield position`. `engine/src/state.rs` stores `own.active` roster index mapping. `agent/types/requests.py::resolve_action_tokens()` uses `layout.self_pokemon_token(own_slot)` and `layout.move_token(own_slot, move_slot)`, while `agent/train/native_collector.py::_level_table_packed` and `_packed_row_inputs` resolve the packed references the same way. This aliases active position to six-member roster index whenever they differ. Requires selected leads, substitutions and double-replacement fixtures.

### ACT-02 Signed target aliasing (P0, CODE-CONFIRMED)

`engine/src/actions.rs::Request::candidates` enumerates targets `[-2,-1,0,1,2]`. `agent/types/actions.py` declares `TARGET_LOCATION_MIN=-1` and clamps lower values. `engine/src/python.rs::candidates_packed_batch` and `candidates_packed_walk` encode `target as u8`, `agent/train/native_collector.py::_level_table_packed` copies all six `u8` fields to action IDs without re-signing, although `_packed_action()` and prefix helpers correctly cast `np.int8` later for engine submissions. Scorer target ID calculation clamps bad representations. Hence policy scoring and submitted engine action can have different target semantics.

### OBS-02 Unknown numeric/boolean semantics lost (P0, CODE-CONFIRMED)

`agent/model/embeddings.py::TokenEmbedding.forward`: float values are multiplied by `float_known`, boolean values by `flag_known`; the masks themselves are not supplied to the numeric projection. The model therefore cannot distinguish explicitly confirmed zero/false from unknown. Category known-state behavior is a separate design and must be checked in V2.

### OBS-03 FP16-destructive normalization (P0, CODE-CONFIRMED)

`engine/src/observation.rs` encodes event change `event.value as f32 / 2147483648.0`; the A10 Mode D `agent/model/precision.py::PrecisionContract.prepare_observation()` applies `floats.half().float()`. Signed changes of ±1 and ±6 underflow to zero. Weight is similarly normalized by `u32::MAX` and small valid values underflow. Audit all 50 numeric slots and effects, not only these examples. Retain sign and meaningful precision through the *real* production precision contract.

### KNOW-01 Dynamic type changes unsynchronized (P0, CODE-CONFIRMED call-path mismatch)

`engine/src/battle.rs::set_type()` modifies `self.mon_mut(e).types` but does not change public `Knowledge.pokemon[].types`. Callers include Soak, Magic Powder, Double Shock, Burn Up, Protean/Libero. Other separate paths (`run_mega`, some forme changes, Mimicry) explicitly update public knowledge. Need public-observability tests before changes: hidden Illusion behavior may prohibit copying exact World state.

### ENG-01 Benched stale disabled state (P1, CODE-CONFIRMED stale state path)

`engine/src/battle.rs::clear_volatile` retains prior disabled flags on base moves, and the comment explicitly says these are recomputed for active Pokémon later. `engine/src/observation.rs` exports `MoveDisabled` for all six owned Pokémon without checking active/request context. Therefore a benched Pokémon can carry a historical disabled flag that is not its current actionable state. Determine rules of visible restriction before altering it.

### OPS-01 Rolling-slot assignment corruption (P1, CODE-CONFIRMED optional path)

`agent/train/native_collector.py::_reset_slots(count, ..., slots=...)` unconditionally replaces `self._assignments` with assignments only for `count` refilled slots, then caller `_collect_rolling` updates a subset of handles/teams/roles. It fails to preserve live slots' assignment mapping. **Not active** in reviewed static A10 profile (`rolling_slots=false`). Fix if supported, or explicitly quarantine path with failing-safe guard.

### OBS-04 Current-vs-public historical move semantics (P1, CODE-CONFIRMED representational approximation)

`engine/src/knowledge.rs::reveal_move` adds revealed moves into the first unknown `current_moves` slot, not necessarily the Pokémon's actual private slot. This is defensible **as a public-information approximation**, but V2 must not interpret these indices as true opponent move slot positions. Keep revealed repertoire separate from actual known current slot when the latter is known.

### OPS-02 Schema validation (P0 new-contract blocker)

`agent/types/observation.py::from_native_payload` reads schema_version but does not reject unsupported version; `from_compact_numpy` defaults to version 1. Checkpoint, Actor, Learner, Eval must fail closed on any V2 schema/wire/precision mismatch. This is a **future migration blocker**, not proof current v1 training was incorrect.

### OPS-03 Training dataset version drift (P0 CONFIG-BLOCKER)

`configs/validated_a94_runtime.yaml`, default `scripts/run_ddp_ppo.py --teams` and `engine/data/manifest.json` reference legacy 1,137 teams. `data/teams/a10-p0-proposed-20261013/manifest.json` has 1,220. Local production may have supplied overrides. Require actual dataset bytes, team count, manifest fingerprint, loader order, side-seed determinism and rank parity receipts.

### OPS-04 Eval CDF numerical boundary (P2 potential runtime rare failure)

`agent/evaluation/batched_policy.py::_sample_model` samples using float32 `cdf = probs.cpu().cumsum(-1)` and `np.searchsorted`. Last cumulative sum may slightly undershoot 1; `min(index, cdf.shape[1]-1)` may select padding and throw. Preserve fixture-keyed deterministic sampling while ensuring any out-of-bounds result maps to the **last legal** candidate without introducing bias beyond roundoff.

## Plausible risks: reproduce first, never assert as fixed before testing

- `RISK-01`: Imprison-related hidden move data could leak via raw `MoveDisabled`: `engine/src/actions.rs` uses `(!m.disabled || m.hidden && slot.last_active)`, while `engine/src/observation.rs` takes `m.disabled` directly. Need paired native states with identical **player-served request, legal candidates, public history**, different hidden move metadata; assert identical model-visible observations and distribution. Confirm distinction between hidden internal restriction and actually observable disabled status.
- `RISK-02`: temporary typing and forme reversions on switch may leave Knowledge stale; `clear_volatile` changes World, `Knowledge.apply(EventKind::Switch)` doesn't universally repair types. Verify exactly when a change/reversion is visible to the viewer; do not leak hidden identity.
- `RISK-03`: transformations, Illusion and roster identity histories may have subtle inconsistencies. Test specific fixtures.
- `RISK-04`: ID out-of-vocabulary silent clamping and category namespace semantics. `agent/model/embeddings.py` clamps IDs to `category_vocab-1`; audit actual Dex max and cross-catalogue collisions *in context* rather than claiming an unproven overflow.
- `RISK-05`: actor/learner agree on the same bad observation and pass recompute. Compare against an independent semantic reference, not only model parity.
- `RISK-06`: some effects and callbacks remain unsupported in full Reg M-C scope; native terminal completion alone does not certify reference battle fidelity.

## Explicitly not confirmed as bugs

- Main Trick Room, weather, terrain, Tailwind and screen duration-decrement routines exist; do not rewrite without a failing targeted test.
- Mega and Mimicry public typing updates exist; preserve and regression-test.
- PPO loss, GAE trajectories and DDP row weighting have unit/reference tests. Their correctness isn't formally proven for every case, but prior review found no new obvious defect; prefer reference comparison over blanket rewrite.
- Direct target-Pokémon and switch-destination token lookups are promising scorer architecture refinements, **not automatically accuracy bugs**. They can be implemented during scratch rebuild only if design/test evidence supports it.
- League vs historical checkpoint win rate is not an independent holdout team generalization measure. The 1,220-team manifest specifies 0 dev/final holdout.

## Navigable reference paths

GitHub root: `https://github.com/mwl313/PokeAgent3.0/tree/d10ac692009566686fd32285c4456309a5f383a7`

Critical files: `engine/src/{actions.rs,observation.rs,knowledge.rs,state.rs,battle.rs,python.rs}`, `agent/types/{actions.py,requests.py,observation.py}`, `agent/model/{embeddings.py,scorer.py,precision.py,pa3_model.py}`, `agent/train/native_collector.py`, `agent/buffer/rollout_buffer.py`, `agent/ppo/{gae.py,learner.py,ddp.py,losses.py}`, `scripts/run_ddp_ppo.py`, `agent/evaluation/batched_policy.py`.
