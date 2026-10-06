# Ability interaction mismatches against the pinned reference

Recorded 2026-10-07 by the ability-fixture workstream. Pinned reference commit
`14546894d86f9589ac11130c510bbe73b6968665`. Every entry below is a *ported*
ability whose interaction fixture in `engine/data/ability-interactions.json`
does not yet reproduce the reference at every decision boundary. The fixtures
are generated (never hand-written) and each carries a verified trigger probe.

Reproduce any entry with:

```bash
bash scripts/cargo.sh test --locked --release --test ability_interactions
node engine/tests/generate_ability_interactions.mjs   # regenerate the corpus
```

`tests/ability_interactions.rs::KNOWN_MISMATCHES` lists these fixtures and the
test **panics if one starts passing**, so an entry cannot outlive its bug.

## Magic Bounce — reflected status move draws

Fixture `magicbounce_reflects_status_7216` (Espeon with Magic Bounce vs Arbok,
seed `[2026,10,7,7216]`), divergence at decision 5, side `P2`:

```
left  (native):    11186,4752,53833,5560
right (reference): 64258,9598,18367,9809
```

The boundary where the streams diverge is the turn on which the original
attacker's Toxic is reflected back at it:

```
|move|p1a: s0m0|Toxic|p2a: s1m0|[from] ability: Magic Bounce
```

The reference re-executes the reflected move as a normal hit step (accuracy
roll, `TryHit` gates, status application, post-hit phases). The native port
reflects the move without the same number/order of PRNG draws. Earlier
boundaries agree; the divergence is a draw-count mismatch, not a state-shape
mismatch. Until the reflected hit step is exact, Magic Bounce should stay
`is_ported() == false` (explicit operational error) or be fixed.

## Damp — blocked selfdestruct draws (resolved 2026-10-07)

Fixture `damp_blocks_explosion_6128` (Damp holder vs an Explosion user, seed
`[2026,10,7,6128]`), divergence at decision 3, side `P2`:

```
left  (native):    14013,48170,43272,64495
right (reference): 17798,42269,14093,60174
```

Fixed by running the single in-move `Update` that precedes the action's queue
re-sort on the TryMove-abort path (`battle.rs`, Damp gate). The fixture now
matches the reference seed and draw count, so Damp is `is_ported() == true`
with no exemption.

## Flower Veil — ally status guard (resolved 2026-10-07)

Fixture `flowerveil_ally_status_guard_7536` (Flower Veil holder beside a
Grass-type ally, seed `[2026,10,7,7536]`), divergence at decision 3, side `P2`:

```
side 0 mon 1 hp  left(native): 190   right(reference): 202
```

Fixed by porting `onAllySetStatus` into the native status path
(`battle.rs::hit_effect_with_ability`): a Grass-type ally of a Flower Veil
holder refuses statuses from another Pokémon and emits the public block only
for non-secondary move sources. The fixture now matches every boundary.

`onAllyTryAddVolatile` (yawn) remains a latent part of the same ability: Yawn
itself is still an explicit operational error, so no reachable legal battle can
exercise it yet.

## Guts (resolved 2026-10-07)

Fixture `guts_burn_attack_8368` (Guts holder burned by Will-O-Wisp, then using
Close Combat), divergence at decision 6, side `P2` (the foe's decision),
command `move 1, move 1`:

```
side 1 mon 4 hp: left(native)=142   right(reference)=102
```

Fixed by porting both halves of the reference rule: the priority-5
`onModifyAtk` 1.5x modifier while statused, and the `modifyDamage` burn-drop
exception (`pokemon.status === 'brn' && physical && !hasAbility('guts')`).
The fixture now matches every boundary with no exemption.
