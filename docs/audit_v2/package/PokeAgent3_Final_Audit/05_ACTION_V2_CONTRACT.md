# Action V2 — canonical identity and sampling semantics

## A. Atomic action contract

`(kind, own_slot, move_slot, target_location, switch_destination, resource)` is a six-component tuple.

- `kind`: PICK=0, MOVE=1, SWITCH=2, PASS=3.
- `own_slot`: for normal requests, **battlefield active position** (0,1), not the six-member roster index. Preview semantics differ: use explicit branch context.
- `move_slot`: actual acting Pokémon's move slot (0..3), or `NO_SLOT=255` for an engine sentinel such as Struggle/recharge.
- `target_location`: signed **i8**, valid native target locations among `[-2,-1,0,1,2]` for current engine scope. Do not conflate with unsigned `255` sentinel of other fields. Validate actual valid target contract for each move and request type.
- `switch_destination`: stable roster member choice (not active position); NO_SLOT sentinel where unused.
- `resource`: NONE=0, MEGA=1, subject to shared resource constraints across branches.

The engine's `Target::valid_location(slot_id, target)` defines legality; the model can only select candidates it receives. Do not invent target IDs.

## B. Canonical decode

The packed wire stores six `u8` bytes and emits `target as u8` for the fourth field. Explicitly cast the fourth byte as signed (`np.int8` or an equivalent unambiguous decoder) **before** constructing model IDs. Ensure every branch and every code path uses the same decode, not just the command sent back to Rust. Enforce a single canonical mapping for neural target embedding (e.g. offset `target - (-2)`) and fail on unknown known locations. Never silently clamp multiple valid targets onto one ID.

Create exhaustive proof:

```
for target in [-2, -1, 0, 1, 2]:
    assert decode(encode(target)) == target
    assert typed_embedding_id(target) == packed_embedding_id(target)
assert len(set(target_embedding_ids)) == 5
```

Actual target legality is move-dependent, so use real native candidate fixtures rather than fabricating illegal complete actions for policy evaluation.

## C. Actor slot identity

Acquire `active_to_roster[0..1]` explicitly from `PlayerView.own.active` or from a verified public observation category field. Resolve MOVE's actor token with its roster index and move slot. PASS/SWITCH actor token also resolves actual departing/fainted actor when applicable; PICK and SWITCH destination must be separately resolved with their correct semantics. No assumptions about roster slot 0/1.

Require tests with arbitrary lead indices `(5,3)`, switch replacement `(2,4)`, faint then forced replacement, two simultaneous empty positions, Mega changes and a duplicate species team.

## D. Conditional action sequence

The policy scores at most four dependent branches per request. The legal candidate set for branch j depends on prior selected actions. For each request:

1. Obtain native legal candidates with the exact prefix.
2. Map each candidate to canonical action ID and **correct actor/move/destination/target token references**.
3. Mask all illegal or padded actions before softmax.
4. Sample/select according to the configured policy.
5. Append chosen action to prefix, recompute next legal candidates.
6. Sum conditional log probabilities to get request logprob. PPO recomputation must replay the **original same candidate sets and chosen indices** and compare hashes.

Invariant: every chosen prefix extends to a complete legal joint action. No duplicate switch into the same destination, no double Mega, correct replacement order and Struggle/recharge behavior.

## E. Typed/Packed/Batched/Eval parity

For each fixture, require exact match of:

- canonical ordered candidate tuples (order is critical for selected indices),
- candidate masks and branch validity,
- actor entity token, current move token, destination token and target token when implemented,
- selected actions/prefixes and resulting legal completion,
- old logprob vs recompute logprob under identical precision contract,
- action score logits up to declared numeric tolerance (no aliasing).

If the typed path and packed path disagree in candidate order, fix mapping/order at source or implement a bijective reordering before evaluating; do not declare parity from set equality alone.

## F. Architecture enhancement vs correctness

Mandatory: correct actor and move token, signed target, canonical action tuple/legality. Optional but recommended during V2 rebuild: direct target Pokémon representation, switch destination Pokémon representation, globally contextual fields. These are architecture changes and should have controlled ablation/performance rationale. The transformer may already attend to target state indirectly; lack of direct target token is **not in itself a proven mathematical bug**.

## G. Tests

- `T-ACT-01`: active-to-roster remapping before and after switches.
- `T-ACT-02`: all signed target locations, typed and packed scoring parity.
- `T-ACT-03`: target context architecture sanity if chosen.
- `T-ACT-04`: legal candidate set vs native oracle across preview, normal, replacement, Mega, trapped, locked, Struggle, forced-switch scenarios.

Block training if any applicable P0 test fails. Keep training and eval candidate normalization aligned.
