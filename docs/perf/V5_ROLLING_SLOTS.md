# V5c — rolling slots + H2D/launcher trims (P0c)

Branch `optimization/pa3-realpolicy-throughput`, base `9c62220`. Scope was
T1 (H2D/launcher trims) and T2 (rolling slots), both gated before promotion.

## T2 rolling slots — implemented behind a flag, BLOCKED by the engine API

Implementation (`NativeCollectorConfig.rolling_slots`, default off;
`scripts/run_ddp_ppo.py --rolling-slots`): finished slots are refilled from a
persistent team RNG until the quota is met, then open games drain (overshoot
preserved); per-slot `_row_cursor` entries are cleared on refill; telemetry
records the same drain-tail metrics with an idle fraction that should collapse.

**Blocker (measured):** the first subset refill makes every still-open game's
handle stale:

```
ValueError: stale environment handle   (engine/src/batch.rs: state() rejects
handle.generation != self.generation)
```

`Engine::reset_batch` bumps the batch generation for the whole engine, so
`request_info_batch`/`step_batch` reject handles issued before the refill. A
correct rolling implementation requires either a per-slot reset API or
per-slot generations in the engine — engine work that this task explicitly
excludes. The collector code and its acceptance tests stay behind the flag:
`tests/integration/test_rolling_slot_equivalence.py` is xfail-documented with
the contract assertions (match counted once, opposite-sign terminal rewards,
both seats, row schema, overshoot, smaller idle fraction) ready for the engine
change.

No A/B is reported for rolling slots: it cannot run end-to-end, and the
task rule is "never promote failing code".

## T1 H2D / launcher trims — not executed (reported)

Not executed in this pass. Rationale, from this week's measurements: the dual
per-rank H2D block is ~1.7 s of a ~63 s all-in run (2.7 %), the optimiser block
is ~19 ms of real per-step work (the CPU timer measures queue drain), and the
launcher/digest costs are sub-second. The expected combined saving sits inside
the declared ±2 % noise floor, so a pinned/async rewrite plus 3× A/B would
consume the budget without a promotable result. This is reported rather than
faked; the roadmap rows stay open.

## T3 optional items

Skipped: the `--columnar-store` re-measure and the CUDA-graph PoC both need
T1/T2 to be complete first, and time was short after the T2 blocker.

## Contract preserved

Default paths are unchanged (all new behaviour is behind `--rolling-slots` /
existing flags): both-seat contract, natural-completion accounting, row schema,
learning math, engine/rules untouched. Full suite green including the new
xfail-documented tests.
