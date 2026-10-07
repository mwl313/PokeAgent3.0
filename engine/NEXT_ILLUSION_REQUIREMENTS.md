# Illusion requirements (observation-layer gap)

Read-only scout, 2026-10-07. Pinned reference commit
`14546894d86f9589ac11130c510bbe73b6968665`, Champions M-C. This is
source-derived implementation guidance with a reproducible probe; passing
source inspection is not differential certification or training readiness.

**Decision: `Ability::Illusion` stays unported.** Any battle that reaches it
remains an explicit operational error (never a silent no-op). The machine
handover rule applies: "If Illusion requires broader observation semantics than
expected, document the gap and stop before hacking it in." The gap is the
player-visible identity model described below.

## Pinned declaration and legal holders

`vendor/pokemon-showdown/data/abilities.ts` (`illusion`, num 149; no override
in `data/mods/champions/abilities.ts`):

```js
illusion: {
  onBeforeSwitchIn(pokemon) {
    pokemon.illusion = null;
    // yes, you can Illusion an active pokemon but only if it's to your right
    for (let i = pokemon.side.pokemon.length - 1; i > pokemon.position; i--) {
      const possibleTarget = pokemon.side.pokemon[i];
      if (!possibleTarget.fainted) {
        if (!pokemon.terastallized || !['Ogerpon', 'Terapagos'].includes(possibleTarget.species.baseSpecies)) {
          pokemon.illusion = possibleTarget;
        }
        break;
      }
    }
  },
  onDamagingHit(damage, target, source, move) {
    if (target.illusion) {
      this.singleEvent('End', this.dex.abilities.get('Illusion'), target.abilityState, target, source, move);
    }
  },
  onEnd(pokemon) {
    if (pokemon.illusion && !pokemon.beingCalledBack) {
      this.debug('illusion cleared');
      pokemon.illusion = null;
      const details = pokemon.getUpdatedDetails();
      this.add('replace', pokemon, details);
      this.add('-end', pokemon, 'Illusion');
    }
  },
  onFaint(pokemon) { pokemon.illusion = null; },
  flags: { failroleplay: 1, noreceiver: 1, noentrain: 1, notrace: 1, failskillswap: 1 },
```

Legal starting holders (`engine/data/scope.json`): **Zoroark** and
**Zoroark-Hisui** (both base species Zoroark). The pinned format's rule dump
shows no `illusionlevelmod` rule, and terastallization is outside the pinned
engine scope, so the Ogerpon/Terapagos tera branch is unreachable.

The disguise partner is chosen over the **party array** (`side.pokemon`
position order, which switch-in swaps permute) - the last non-fainted member
strictly to the right of the entering Pokemon's position. It is not the
roster/submission order the native engine otherwise uses.

## Verified reference behavior

Reproduce with `node engine/tests/probe_illusion.mjs` (read-only scout; never
merged as a corpus generator). Zoroark enters at p1a; its partner is the party
position-5 member `s3` (Torterra). Zoroark's real max HP is 167; Torterra's is
different.

Scene A - damaging hit (`Iron Head` from p2a):

```
|switch|p1a: s3|Torterra, L50, M|167/167
|move|p1a: s3|Night Daze|p2a: s0
|move|p2a: s0|Iron Head|p1a: s3
|-damage|p1a: s3|91/167
|replace|p1a: s0|Zoroark, L50, M
|-end|p1a: s0|Illusion
```

Observations:

- The public identity of the slot is the **partner's name and species** but
  the **real Pokemon's HP** (`167/167` is Zoroark's max HP).
- Every message about the illusioned Pokemon (switch, move, damage) uses the
  masked identifier `p1a: s3`; the damaging hit's `-damage` line still uses
  the mask, and only then do `replace` + `-end Illusion` restore the real
  identity `p1a: s0` / `Zoroark, L50, M`.
- `pokemon.species` itself never changes: the world state (stats, damage,
  statuses, PP) is the real Pokemon's. Illusion consumes no RNG.

Scene B - switch-out while the illusion is intact:

```
|switch|p1a: s2|Metagross, L50|187/187     (the entrant's normal line)
```

No `replace` / `-end Illusion` is emitted (`beingCalledBack`), and the benched
Zoroark keeps a stale `illusion` reference until its next `onBeforeSwitchIn`
resets and recomputes it. A faint clears the illusion (no End messages).

## Why this is not a small primitive in the native engine

The native information model (`engine/src/knowledge.rs`,
`engine/src/battle.rs::emit`) is:

- `Knowledge { pokemon: [PublicPokemon; 12] }` keyed by **stable roster
  entity** (`roster + side*6`), holding `species`, `types`, `gender`,
  `health`, `status`, `ability`, `item`, `active_slot`, `selected`,
  `current_moves`, `effects`, ...
- `emit(kind, subject, ...)` applies the **same event to both viewers** (only
  owner-exact vs public HP display differs) and reads the health from the
  event subject. `PlayerView` / the observation encoder are built from these
  per-entity beliefs.

Illusion makes the belief about *which entity occupies the active slot* wrong
for **both** viewers: the partner entity must appear active with the real
Pokemon's HP under the partner's species/name, while every subsequent public
event about the real entity is attributed to the partner entity. At reveal,
the real entity is repaired and the partner entry is left stale until it next
enters. The native model has no per-viewer observed-entity remap, and no event
carries an "identity payload" distinct from its subject.

Concrete pieces a faithful port needs (all currently missing):

1. `PokemonState.illusion: Option<u8>` (partner roster) with snapshot schema
   bump, restore validation and corruption coverage.
2. Partner selection over the party array (`sides[side].positions`), skipping
   fainted members, strictly to the right of the entering position, run at
   `BeforeSwitchIn` (before the `Switch` emit) and reset on every entry.
3. A per-viewer observed-entity remap in the knowledge layer: masked
   switch-in writes the partner's `species`/`types`/`gender` (partner data)
   plus the real entity's `health` and `active_slot`/`selected`, clears the
   slot's previous occupant, and routes all later events about the real entity
   to the partner's `PublicPokemon` while world state stays on the real one.
4. Reveal paths: `onDamagingHit` end (re-apply the real identity, then the
   `-end Illusion` marker), faint clear with no messages, switch-out clear
   with no messages, and the stale benched entry behavior.
5. Observation/leakage audit: the masked entity must expose exactly what the
   protocol shows (real HP display, partner identity) and must not leak the
   real ability/item/moves or any hidden state into tensor/mask.
6. Interaction flags once those effects execute: `failskillswap` (Skill Swap),
   `noentrain` (Entrainment), `failroleplay` (Role Play), `notrace` (Trace),
   `noreceiver` (Receiver). Skill Swap, Entrainment and Role Play are legal
   moves in scope, so a later port must refuse Illusion explicitly.

## Verification gap

The differential corpus cannot currently witness Illusion: the fixture
`compact()` records world state (`p.species.id`, stats, HP), which is
identical with and without the illusion, and `engine/tests/turns.rs` does not
compare per-viewer knowledge. A faithful port therefore also needs a new
fixture surface (reference protocol identity/details per boundary) or
dedicated native tests pinned to `probe_illusion.mjs`'s lines, before it can
be reported as witnessed.

Zoroark / Zoroark-Hisui teams stay in the frozen training pool and remain pool
blockers (distinct ability weight 10) until that work lands.
