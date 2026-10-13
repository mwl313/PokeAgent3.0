# P1 — historical investigation; see final exit report

Latest status: [P1 final exit](P1_FINAL_EXIT_REPORT.md) passes the frozen-pool
engine/public-information gate at `548902547a5d5235f532193a47a77a89abfb1cdd`.
The earlier blocked states below are retained as historical evidence.

## Latest Imprison follow-up

The original fixture discrepancy and native Imprison retry blocker are now
repaired. Source `1f93685a224ebe21933c7a390c3e5b86b7e16435` implements request
revision, model resampling, peer-pending preservation and Struggle substitution.
41 relevant Rust tests, 8 Python tests and 5 Node adapter tests pass; the
independent verifier passes 1,152 fixtures / 25,231 boundaries, plus two new
reference-derived retry witnesses. In Gen 9 doubles the first rejection
discloses all disabled moves for that Pokemon, not one move per retry.
P1 remains **BLOCKED** by broader exact-pool dynamic reachability and public
observation certification, not by the now-fixed Imprison retry transaction.
See [the current implementation report](p1_imprison/RETRY_IMPLEMENTATION_REPORT.md)
and `p1_imprison/RETRY_RECEIPT.json` for source, tests and scope limits.
The remaining sections preserve the earlier investigation and failure history.
During the follow-up the user explicitly authorized stopping legacy evaluation;
its worker/timer are now stopped and disabled, with old evidence preserved.

This is not a claim that the master task or P1 gate is complete. No V2 model,
optimizer, checkpoint, training campaign or mixed V1/V2 league was launched.

## Reproduced and repaired paths

* KNOW-01: `set_type()` changed World types but left player Knowledge stale.
  `KNOW01_BEFORE.log` records the failure; `KNOW01_AFTER.log` records the
  primitive fix. Both viewers now receive publicly changed types without
  copying the concealed species, ability or item. RNG and no-op behavior stay
  unchanged. This is a shared primitive witness, not a separate end-to-end
  certification of every Soak/Libero/etc. caller.
* KNOW-02: a type-changed Pokémon switched out and re-entered with original
  World types but stale public types. `KNOW02_BEFORE.log` reproduces this.
  Entry now synchronizes publicly apparent types; Illusion uses its apparent
  partner for the opponent and true known types for the owner.
* KNOW-03: a paired active state with the same served legal actions but a
  different internal Imprison hidden-disable flag produced different encoded
  observations (`KNOW03_BEFORE.log`). Player views now normalize request and
  own-move flags using the reference last-active/restrictData semantics.
  Public disabled flags remain visible. Raw World restrictions remain intact.
  This controlled witness does not replace a naturally executed Imprison
  battle and independent public-protocol oracle.
* KNOW-04: stale benched restrictions no longer become current restrictions
  in OwnSideView, and original/base move metadata no longer carries a current
  request restriction. Explicit V2 unknown/applicable masks remain pending.

Five new primitive tests pass (`P1_PRIMITIVES_AFTER.log`). All Rust library
unit tests pass: 10 tests (`P1_ALL_UNIT.log`). The existing contracts,
observation completeness and hidden-state tests passed: 16 + 2 + 1 tests
(`P1_REGRESSION.log`), executed before the final entry/served-view patch.
They must not be represented as full post-patch certification. Final release
regression also passed 19/19 after the patches (`P1_FINAL_REGRESSION.log`).
Thus 29 distinct unit/regression tests pass, while the full-corpus oracle
acceptance test fails at the counterexample documented below.

Runtime code commit: `0492577d2ae611b3720f52f57939422b0148e103`.
`d48ef2e` subsequently changes only inventory/status scripts; Rust runtime
matches the tested commit. Evidence is committed separately.

## Exact commands

All cargo commands use:

```
CARGO_HOME=/home/minidc/Desktop/PokeAgent3.0-A/.tools/cargo
RUSTUP_HOME=/home/minidc/Desktop/PokeAgent3.0-A/.tools/rustup
CARGO_TARGET_DIR=/home/minidc/Desktop/PokeAgent3.0-V2/.audit-target
```

Cargo binary: `/home/minidc/Desktop/PokeAgent3.0-A/.tools/cargo/bin/cargo`.
Manifest: `/home/minidc/Desktop/PokeAgent3.0-V2/engine/Cargo.toml`.

Commands (append to the binary with `--manifest-path <manifest>`):

```
test --locked --lib public_type_tests
test --locked --lib
test --locked --test observation_leakage --test observation_completeness --test contracts
test --locked --release --test turns native_battles_match_reference_at_every_decision_boundary
run --locked --example coverage_report -- /home/minidc/Desktop/PokeAgent3.0-V2/engine/data /home/minidc/Desktop/PokeAgent3.0-A/engine/data/a10-training-teams.json
```

The original debug full-corpus test was cancelled specifically by its owned
test PID to rerun optimized; it is not a PASS. See `P1_ORACLE_RELEASE.log` for
the completed release test result and any counterexample.

## Confirmed oracle blocker (do not bypass)

`P1_ORACLE_RELEASE.log` FAILS at `imprison_shared_pool`, decision 3, P2,
slot 1. The new served view offers move IDs 193 and 635; the stored fixture
offers only 193. The original fixture collector
`engine/tests/fixture_scaffold.mjs:73` records
`disabled: Boolean(m.disabled)` from privileged `p.moveSlots`. A reference
`'hidden'` flag therefore becomes true even though pinned
`sim/pokemon.ts:getMoves` rewrites it to `!restrictData` in the actual served
request. The collector already uses the served request for targets but not
disabled status. Multiple copied generators share this pattern.

This is not a passing full-corpus validation or evidence that the new
public-view normalization is fully certified. Repair/re-generate public
request fixtures from the pinned reference before accepting the P1 gate;
retain privileged World checks separately. Do not edit stored expected masks
by hand or suppress the failing assertion. The production extension remains
the unchanged legacy binary, so this incomplete migration cannot affect the
active legacy League.

## Further release blockers

The exact 1,220-team static classifier reports 0 direct move, ability or item
blockers. The broader dynamic closure reports 11 unimplemented entries:
Harvest, Imposter, Klutz, Pickup, Receiver, Copycat, Gastro Acid,
Instruct, Magic Room, Transform and Worry Seed (see exact canonical spellings
in `EXACT_POOL_COVERAGE.log`). That global list does not prove these are all
reachable from the actual pool. Conversely, 0 direct blockers does not prove
the exact pool's dynamic closure is safe. Exact-pool closure and independent
public-observation oracle certification are unresolved P1 release gates.

The pinned Showdown checkout exists locally at
`14546894d86f9589ac11130c510bbe73b6968665`. Stored corpus World/RNG/request
parity is useful but does not certify all player Knowledge after type changes,
Transform/Illusion or hidden metadata. No unsupported handler was invented or
silently treated as successful.

P2-P5 are not implemented: five ragged neural consumers, explicit known masks,
numeric normalization, signed action identities, active-roster remapping,
strict V2 checkpoints, generation-bound per-team statistics, V2 DDP smoke and
scratch initialization remain release blockers. There is no V2 schema hash,
new-run throughput or per-team committed counter to report yet.

Next milestone: close P1 with actual public-protocol differential fixtures
and exact-pool dynamic reachability, then freeze the shared V2 observation and
action interface before implementing P2/P3. The safe legacy restore state is
the P0 report's 1,400,832-match generation; it must not seed V2 weights.
