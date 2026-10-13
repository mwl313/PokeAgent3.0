# P1 Imprison selection retry — implemented and verified

Source commit: `1f93685a224ebe21933c7a390c3e5b86b7e16435`.
Branch/worktree: `codex/pa3-v2-correctness`, `/home/minidc/Desktop/PokeAgent3.0-V2`.
This supersedes the unsupported-retry blocker in `RESOLUTION_REPORT.md`.
**Imprison retry gate: PASS. Overall P1 release: still BLOCKED** by the broader
exact-pool dynamic reachability and public-observation certification gates.
No P2 work, PPO updates, production restart, or historical model evaluation.

## Reference behavior and correction to the earlier explanation

Pinned, clean Showdown: `14546894d86f9589ac11130c510bbe73b6968665`;
format `gen9championsvgc2026regmc`.
`Side.chooseMove` checks unrestricted `Pokemon.getMoves()`; if there are no
enabled moves, it substitutes Struggle before the disabled-choice rejection.
Otherwise a hidden-disabled choice invokes `updateDisabledRequest` and emits
the updated player request. In this Gen 9 doubles format, `maybeDisabled`
causes **all disabled moves for that Pokemon** to be disclosed on the first
rejection, not just the selected move. Thus another shared move is immediately
masked out. The earlier conversational description of successive disclosures
of individual shared moves was inaccurate for this format.

The engine now returns `accepted=false`, no operational error, and updated
request kinds for this normal retry. It does not advance the turn or consume
battle RNG. Only the affected slot's cached request disclosure changes;
physical state, other slots, and the peer's pending submission are preserved.
The whole rejected joint submission remains uncommitted and is resampled.
Submitting a stale disabled action is invalid input and remains non-mutating.
Restrictions reset with the next freshly generated request, not permanently.
Execution-time failure when Imprison activates after choices were accepted
continues to use the existing move-execution handler.

The batch result retains a retry signal even when the rejecting side was
submitted before the accepted peer. Both ordering cases are tested. Snapshots
preserve cached disclosure and already substituted pending Struggle. Restore
still verifies IDs, PP, disabled flags, targets and other request/world fields;
it permits the all-at-once monotone clearing of hidden metadata for the
last-active slot. Trace replay now includes rejected selection attempts.

## Model and evaluation integration

The native collector and development actor return to their normal request
sampling loop after a non-operational retry. Only actionable sides are sampled,
so an already committed peer waits. The collector retains the rejected public
decision as a trajectory row, followed by the revised-mask observation; it
advances request index but not its turn counter on the retry. This represents
a real information-revealing environment transition, not an optimizer update.
Fixed and batched evaluators also resample; they do not count this legitimate
request revision as an illegal action, loss, or operational failure. Round
limits still bound every loop. Existing tuple layout and PPO defaults are
unchanged. External consumers of the native API must follow this documented
non-error retry contract too.

## Verification

- **41 relevant Rust tests PASS:** 14 library, 16 contracts, 2 observation
  completeness, 1 hidden-state leakage, 8 turns tests. Includes full stored
  oracle parity and the existing 2048-complete-battle batch determinism test.
  `retry_native.log` records the command output. This is the listed relevant
  suite, not a claim that every engine integration binary was run.
- **3 targeted Imprison tests PASS** after adding HP/PP/status comparisons to
  the differential witness: `retry_witness.log`. The full suite preceded only
  these extra test assertions, extra fixture evidence and comment corrections;
  production behavior was unchanged.
- **Independent oracle PASS:** 1152/1152 existing fixtures and 25231 decision
  boundaries against actual emitted Showdown requests; `retry_independent.log`.
- **Two new reference-derived differential fixtures PASS:** multiple shared
  moves plus one usable move; all-hidden Struggle with the substituting side
  submitted first. Compare acceptance, public disabled masks, turn, RNG, HP,
  PP and status through the next turn. Snapshot/restore at every boundary and
  trace replay also pass. Generator uses actual emitted request payloads,
  independently of the existing public-request adapter. Regenerating twice
  produces byte-identical JSON, SHA256
  `81da1f4db41eebc5fa2f1c20d919e852b508b571e4bf803c406e34f9e6150f63`.
- **8 Python tests PASS:** 2 actual native-binding fixture replays; 2 evaluator
  retry control-flow smoke cases; 1 tiny-model collector test completing one
  natural match without any optimizer; 3 existing fixed-policy tests.
  The consumer control-flow tests inject a retry signal independently of mask
  semantics, which are tested against the actual native/reference fixtures.
  `retry_python.log` records results. One pre-existing read-only NumPy-to-Torch
  warning occurs; no tensor mutation was added.
- **5 Node adapter tests PASS:** `retry_adapter.log`.

No existing corpus expectations or frozen training teams were changed in this
follow-up. The new two-case file is additional test data, not merged into the
existing 1152-fixture count. These bounded checks do not claim exhaustive
certification of every move interaction or other battle formats.

Exact commands, environment, source paths and artifact hashes are in
`RETRY_RECEIPT.json`. Code paths changed: `engine/src/battle.rs`, `batch.rs`,
`state.rs`, `actions.rs`, `python.rs`, `battle/public_type_tests.rs`;
`engine/python/pa3_actor.py`; `agent/train/native_collector.py`;
`agent/evaluation/{fixed_policy,batched_policy}.py`;
`engine/tests/p1_generate_retry_fixtures.mjs`;
`engine/data/imprison-retry-fixtures.json`;
`tests/integration/test_imprison_retry.py`.

## Operations and remaining scope

The legacy evaluation worker and timer remain inactive and disabled. Old
checkpoints and completed results are preserved. Correction to the previous
report's stop note: the discovery oneshot raced with the user-authorized stop
at 23:22:57 KST on 2026-10-13 and received SIGTERM; its retained `failed/signal`
status is from that stop, with MainPID 0, not a newly diagnosed model failure.
PPO remains inactive. Only the isolated V2 native extension was rebuilt.

Next P1 work is exact-pool dynamic closure and the outstanding broader public
typing/identity oracle certification. Do not infer P1 PASS or start V2 training
from this narrow Imprison repair. P2 remains outside this task.
