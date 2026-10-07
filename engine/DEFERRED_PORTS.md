# Deferred ports (development notes)

Notes for mechanics that were prototyped but deliberately not landed, so the
next attempt starts from evidence instead of from scratch. Nothing here is a
readiness claim.

## Raging Bull (Tauros / Paldea forms, pool weight 1)

- Implemented and verified through the mechanic itself: the `onTryHit` screen
  shatter (reflect / light screen / aurora veil, through substitutes) plus the
  form-driven `onModifyType` taken from the user's primary type (Paldea Tauros
  forms are Fighting / Fire / Water; plain Tauros stays Normal). `onModifyType`
  was added to the handled move fields and the hook bit assigned.
- Why it was reverted: the witness scene's auto-play tail diverged on
  *unrelated* state - decision 11 turn counter 5 vs 6 and decision 16 P2
  roster-2 HP 44 vs 85. That pattern matches the still-open fainted-slot /
  turn-boundary accounting family, so a stable witness scene is not reachable
  yet. Landing the move without a witness would break the coverage invariant
  ("every executable move has a differential fixture"), so the port was
  reverted in full rather than held out.
- Retry after the `move_magicpowder_29910` ledger entry is fixed; the
  implementation sketch above is the whole port (roughly 100 lines).
