# P1 Imprison oracle repair — fixture discrepancy resolved, P1 still BLOCKED

Runtime/pipeline commit: `86ffebc09f9c91e765314c79f04d2b8004fc7ac9`.
Starting revision: `ec5912736440d5f5456f678c4d76f8fd26fd20aa`.
No Observation V2, Action V2, statistics or PPO training was started.

## Findings and independent reference evidence

The original unchanged-corpus failure was reproduced in
`original_failure.log`: `imprison_shared_pool`, decision 3, P2, slot 1.
Native advertised move IDs 193 (Dragon Pulse) and 635 (Protect); the old
fixture retained only 193. Teams, seed `[2026,10,8,4201]`, all previous
commands, both sides' raw move states, native snapshots, public requests and
native candidate tuples are saved in `reference_before.json` and
`native_before.json`.

The reference was independently cloned at
`14546894d86f9589ac11130c510bbe73b6968665`, built using `node build`, and
verified clean. Format: `gen9championsvgc2026regmc`. Requests were captured
from the actual `sideupdate` / `|request|` payload after `sendUpdates`, not
assumed from `activeRequest`. The payload and activeRequest were compared.
`reference_fresh_build.json` repeats the direct evidence with the fresh build.
The two probes agree after excluding only the protocol wall-clock `|t:|`
line; stored fixture expectation changes are not battle/RNG differences.

Three separate meanings must remain distinct:

| Layer | Protect on last-active Goodra |
|---|---|
| Privileged `moveSlots` | `disabled: "hidden"` |
| Outbound player request | `disabled: false` |
| Submitting `move 1, move 2` | rejected; revised request marks Protect disabled |

Thus the initial hypothesis was only part of the cause:

1. **Fixture error confirmed:** `Boolean('hidden')` recorded internal state
   as a public disabled boolean. Public moves must be mapped by stable move
   ID to the served list, including rewritten targets and substituted moves.
2. **Native executable discrepancy confirmed:** the native engine accepted
   the advertised hidden Protect choice; reference `Side.chooseMove` checks
   unrestricted `getMoves()` and rejects before committing it. The reference
   revises the request without advancing turn or RNG. Native previously
   deferred this to an execution-time failure, which is not equivalent.
3. **Non-actionable request gap found:** replacement/wait snapshots retained
   raw old move lists despite no outbound active move request. The pipeline
   and native public view now omit those lists. Privileged World state remains
   separate. An Emergency Exit replacement witness and snapshot immutability
   test cover this case.

Imprison that becomes active *after* choices were accepted can legitimately
cause a move to fail during execution. That case is distinct from rejection
of a choice that was already hidden-disabled when submitted; the activation
logs in the reference evidence retain this distinction.

## Changes and regeneration scope

`public_request_moves.mjs` is shared by 40 request-detail generators, including
the main generator, interaction/disable generators, the fixture scaffold and
the inline Aura Guard generator. It uses only served boolean restrictions and
matches by move ID. It rejects a non-boolean public disabled value, handles
Struggle/locked list substitution, preserves own slot/PP identity, and reads
rewritten target types. Non-active requests expose no current move list.
Privileged debugging `pokemon.disabled` arrays in the ability/move-hook
development schemas were intentionally retained and explicitly commented.
They are not public request or policy inputs. Their alternate request schemas
were not expanded as part of this repair.

`verify_turn_fixtures.mjs` independently reads the serialized outbound request;
it does not call the generator normalization helper. Full reference World/RNG,
accepted command and public-request comparison remains enabled. No fixture was
skipped and no assertion was removed to force a pass.

The replay migration touches **103 derived fixture/corpus files**, because the
same non-actionable raw-mask pattern occurred across cohorts, not only the
Imprison fixture. It replays 2,141 fixture instances (including source-cohort
duplicates of merged corpus cases), 44,099 boundaries, with unchanged teams,
seeds and commands. Every changed expected field and its reason is in
`regeneration.json`. Only existing `request_detail` fields changed. An
independent structural comparison proves all other expected World/RNG fields,
actions, teams, fixtures and ordering are identical to the starting revision.
The two merged corpus files contain 1,152 distinct cases.

| Corpus | Before SHA256 | After SHA256 |
|---|---|---|
| turn-fixtures.json | `5a5a7431bab23425d3b6475cbd4083e50173852f260866963c97cd20b76017ad` | `167b30e09b6a2b5e6f59a0ebc40b64957e15611a047ec47a9c2993df149beffb` |
| turn-fixtures-2.json | `185c5c602c251e5ca5f86ed297f82fbae8c89626117cbd41aaf14c74de0c6d0d` | `bd10e253a763e539fc55dd8252e548bf3ad0c0624c072b34b6c571a3dcd326e6` |

All other before/after hashes, counts, paths and manifest changes are in
`regeneration.json`; the exact 157 changed runtime/pipeline/fixture paths and
artifact hashes are recorded in `RECEIPT.json`. Frozen training teams and Dex
were not regenerated or changed.

## Validation

* Original full oracle: FAIL at the original Imprison assertion, preserved.
* Final Rust library tests: 12 PASS, including two new regression witnesses.
* Existing contracts/completeness/leakage: 16 + 2 + 1 PASS.
* Turns integration suite: 8 PASS, including the complete oracle and two
  16-worker batches of 2,048 complete battles.
* Total relevant Rust tests: **39 PASS, 0 FAIL, 0 filtered**.
* Node public request adapter: **5 PASS**.
* Independent fresh-built outbound verifier: **1,152/1,152 fixtures,
  25,231 boundaries PASS**.
* Normal and last-active Imprison masks, nonshared moves, true non-Imprison
  disables, locked moves, Struggle, recharge and switch/reentry are witnessed
  by the independent corpus verifier. Coverage counts and first witnesses are
  emitted in `independent_final.log`.
* Additional naturally played switch-out/deactivation/reentry/reactivation
  commands pass (`additional_reference_scenarios.json`). A separate
  **controlled reference state** tests only-survivor behavior; it is explicitly
  not claimed as a naturally played native/reference differential fixture.
* Existing hidden-state noninterference tests still pass.

Commands, run from the V2 root:

```
node engine/tests/p1_reference_probe.mjs docs/audit_v2/p1_imprison/reference_before.json
node engine/tests/p1_reference_scenarios.mjs docs/audit_v2/p1_imprison/additional_reference_scenarios.json
node --test engine/tests/public_request_moves.test.mjs
node --max-old-space-size=8192 engine/tests/p1_regenerate_public_requests.mjs docs/audit_v2/p1_imprison/regeneration.json
node --max-old-space-size=8192 engine/tests/verify_turn_fixtures.mjs
python3 scripts/p1_imprison_receipt.py
```

Rust command:

```
CARGO_HOME=/home/minidc/Desktop/PokeAgent3.0-A/.tools/cargo RUSTUP_HOME=/home/minidc/Desktop/PokeAgent3.0-A/.tools/rustup CARGO_TARGET_DIR=/home/minidc/Desktop/PokeAgent3.0-V2/.audit-target /home/minidc/Desktop/PokeAgent3.0-A/.tools/cargo/bin/cargo test --manifest-path engine/Cargo.toml --locked --release --lib --test observation_leakage --test observation_completeness --test contracts --test turns
```

## Remaining release blocker — do not call this a full engine fix

The native request retry transaction is **not implemented**. The new guard
returns a precise `Unsupported` error before trace/pending/queue mutation for
a hidden-disabled last-active submitted move. Its regression fails before the
guard and passes after it, proving snapshot/RNG immutability. This is safe
containment, not reference-equivalent retry support. All-hidden moves can also
need reference Struggle substitution rather than ordinary acceptance/rejection.

This is a confirmed reachable concern: **48 of the frozen 1,220 teams have an
Imprison holder**. See canonical team IDs in `RECEIPT.json`. Therefore a passing
scripted corpus does not permit production release or P1 PASS.

Required next P1 work: implement the cached revised public request and its
selection retry transaction, preserve pending opponent choices and exact RNG,
handle Struggle substitution, and propagate a distinct non-operational retry
result through native batching and its consumers. Test accepted choices,
rejected choices, retry/resume and final execution independently. Do not expose
the hidden restriction before the reference would reveal it. Broader exact-pool
dynamic closure and outstanding public type/identity certification from the
master plan also remain unresolved.

P2 entry remains the contract-freeze step in the master plan, **after** P1
release gates pass; this task did not begin it.

## Legacy evaluation retirement authorized during this task

The user explicitly authorized stopping old-model evaluation. The worker and
discovery timer were stopped and disabled, so they will not restart at boot.
The discovery oneshot has no process and retains its failed service status;
that prior status was not erased. Old checkpoints, completed receipts and
queue evidence were not deleted. This supersedes the earlier instruction to
leave the legacy League running, while preserving its historical evidence.
