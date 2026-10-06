# Force-switch (phazing) family: implemented and differentially verified

2026-10-07 (updated). `roar`, `whirlwind`, `dragontail` and `circlethrow` are
**enabled**: `assets.rs::classify_move` no longer rejects `forceSwitch`,
`generate_more_force_switch.mjs` records one complete legal reference battle per
move (seeds 3900-3903), and
`native_battles_match_reference_at_every_decision_boundary` compares every
boundary of those battles. Pinned reference:
`14546894d86f9589ac11130c510bbe73b6968665`.

## Verified root causes and fixes (2026-10-07, primary agent)

The earlier version of this note blamed a two-draw divergence on the gen>=8
queue re-sort. That was not the cause. Two independent bugs were reproduced
with a temporary in-memory ungate (`engine/examples/tmp_force_switch_probe.rs`),
per-draw reference traces (`ReferenceSession({record_rng: true})` plus a wrapped
`battle.prng.rng.next` that records `new Error().stack` and the seed after each
draw), and gdb backtraces on the native `each_update` call sites:

1. **Empty force-switch payload counted as "did anything".**
   `hit_effect` returns true for an empty `HitEffect` payload, so a phazing
   status move against a side with no reserve (`canSwitch(target.side) ==
   false`, reference `runMoveEffects`) ran the success path: the two move-loop
   `Update` events plus the post-action one. The reference fails the move and
   runs only the post-action `Update`. Fix: when `m.force_switch`, the
   per-target contribution to `did_anything` is `can_switch(target.side)`,
   exactly like the reference's
   `hitResult = !!this.battle.canSwitch(target.side)`. Verified: the roar
   fixture's divergent turn changed from the native's +26 draws to the
   reference's +24, and every subsequent boundary matches.
2. **Drag-in sampled the request-order bench instead of the live party.**
   `bench()` intentionally lists switch destinations in preview-pick order for
   player requests. The reference's `getRandomSwitchable` ->
   `possibleSwitches` instead iterates `side.pokemon` from `active.length`,
   i.e. the live party arrangement that swaps on every switch-in. Sampling the
   pick-order list picks a different reserve whenever an earlier switch has
   reordered the party. Fix: new `BattleState::party_reserves(side)` reads
   `positions[2..]` (non-fainted entries), and `resolve_forced_switches`
   samples that. Verified: whirlwind previously failed the first replacement
   request with `InvalidInput("illegal joint action")` because the wrong
   Pokémon had been dragged in; it now matches all 20 boundaries.

Both fixes are inert while the family was gated and were verified with the
temporary probe before the gate opened.

## What is implemented

* `Move.force_switch` is loaded from the pinned declaration.
* `PokemonState.force_switch_flag` mirrors the reference `forceSwitchFlag`.
* `resolve_forced_switches` runs the reference phazing block directly after the
  action and before the faint check: it samples `Battle.getRandomSwitchable`
  (one uniform draw over the live party reserves), then calls
  `switch_in_inner(..., drag = true)`, which skips `BeforeSwitchOut`/Update
  exactly like `isDrag` does and runs the `runSwitch` SwitchIn tie sort
  synchronously.
* `SideState.positions` mirrors the reference `side.pokemon` party array so the
  random reserve sample is taken in reference order; `switch_in` swaps the
  incoming reserve's array slot with the active slot it takes.
* `use_move` marks every surviving in-range target exactly as the reference's
  `forceSwitch()` step does, after self drops and secondaries, and the
  `did_anything` rule above covers the no-reserve failure.

## Remaining obligations for this family

* **`DragOut` refusal handlers** (Suction Cups, Guard Dog, Ingrain) are still
  unported abilities/conditions, so those holders remain explicit operational
  errors. When ported, the reference rule is
  `hitResult = runEvent('DragOut', ...)`: a `false` result makes a **status**
  force-switch move fail (`-fail` + `[still]`, no drag) while a damaging
  force-switch move continues; Guard Dog / Suction Cups return `null`, which
  blocks the flag without the fail message. The native must mirror that
  distinction before those abilities are enabled.
* **Hazards on the dragged-in Pokémon** (Stealth Rock etc.) require the hazard
  family first; no fixture currently covers them.
* **Interaction witnesses** for Substitute, Ingrain and the
  `forceSwitch` + Suction Cups legality case are owed when those effects land.
