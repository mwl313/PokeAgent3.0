# Next family: Disguise (Mimikyu)

Status: prepared, **not implemented**. Reference semantics are probed and four
complete legal fixtures are generated and verified against the pinned
Showdown checkout; the native port and the merge into the corpus are still to
do. This is the single largest remaining training-pool blocker after the
batches already in flight (11 teams in the last trajectory probe).

## Reference semantics (pinned commit `14546894d86f9589ac11130c510bbe73b6968665`)

`abilities:disguise` is one ability whose observable parts are:

1. **`onDamage` (priority 1, i.e. before Magic Guard/Sturdy/Focus Sash).**
   While the holder's *species* is `mimikyu`/`mimikyutotem` and the damage
   effect is a **move**, the handler emits `-activate <target> ability:
   Disguise`, records `effectState.busted = true` and returns `0`, so the hit
   deals no HP damage but still counts as a hit. The protected-crit and
   effectiveness hooks are cosmetic on this hit: the crit roll still happens
   (RNG parity) and the damage is zero either way.
2. **`onUpdate`.** When `effectState.busted` is set, the holder changes forme
   to `Mimikyu-Busted` (Totem → `Mimikyu-Busted-Totem`) and then pays
   `floor(baseMaxHP / 8)` as damage whose effect is **the new species**
   (`-damage ... [from] pokemon: Mimikyu-Busted`), so it is not move damage.
   The forme change is permanent for the battle; the bust flag itself is not
   needed after the species changes.

Probed consequences that the fixtures pin:

- Single hit: `|-activate|..|ability: Disguise`, a `-damage` line showing the
  full HP, then `detailschange` + the 1/8 damage in the same turn's Update.
- Multi-hit move (Excadrill Rock Blast): the first hit is absorbed, the Update
  changes forme and pays 1/8, and the remaining hits of the same move land on
  Mimikyu-Busted (`-hitcount` still reports the full count).
- A super-effective first hit is still absorbed; only the 1/8 busting damage
  applies on that turn.
- The busted forme takes normal damage on later turns.

## Fixtures already prepared

`tmp/generate_more_disguise.mjs` (move it to `engine/tests/` when porting)
writes `engine/data/more_disguise.json` with four complete legal battles:

| Fixture | Covers |
|---|---|
| `disguise_absorbs_first_hit` | Snorlax Crunch: activate, forme change, 1/8 busting damage |
| `disguise_multi_hit_later_hits_land` | Excadrill Rock Blast: later hits land on the busted forme |
| `disguise_super_effective_hit_absorbed` | Gholdengo Shadow Ball: 2x hit still deals zero |
| `disguise_busted_forme_takes_damage_next_turn` | Turn 2 damage lands normally |

Each fixture records the reference request at every decision boundary, so the
served legal-action mask is compared as well. The trials verify their own
mechanic from the reference log (activation, `detailschange`, the
`[from] pokemon: Mimikyu-Busted` damage line) and, for the state checks, follow
the roster slot rather than party order, because the party order changes when
Mimikyu faints and is replaced.

## Native implementation sketch

1. `Ability::is_ported`: drop `Ability::Disguise` from the exclusion list.
2. `PokemonState`: add `disguise_busted: bool`; bump `SNAPSHOT_SCHEMA` and
   validate that the flag implies `ability == Disguise` (a malformed snapshot
   must be rejected, not silently repaired).
3. Damage application (`battle/hooks.rs`, the move-damage path): before Magic
   Guard/Sturdy/Focus Sash, if the target's ability is Disguise and its species
   is `mimikyu`/`mimikyutotem` and the damage source is this action's move,
   reveal the ability (`EventKind::Ability`), set `disguise_busted`, and apply
   no HP change (the reference's zero-damage `-damage` line is a protocol
   detail; the native state delta is "no damage" plus the reveal).
4. `each_update` (ability Update phase): when the flag is set, clear it, change
   the species to `mimikyubusted` (`EventKind::Forme`), and apply
   `stats[0] / 8` (floor) through `indirect_damage` with
   `EffectRef::Species(mimikyubusted)` so observations and the faint source
   match the reference.
5. Merge the four fixtures (`tmp/merge_*.mjs` pattern or
   `npm run export:engine`) and run the full gate set.

Open questions to re-probe if the corpus disagrees: the exact ordering between
Disguise's Update and item Update handlers (Leftovers) when the bust damage and
a heal land in the same Update, and whether a Substitute intercepts the hit
before Disguise (the substitute work landed after this probe).
