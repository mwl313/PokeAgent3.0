// Development-only corpus for Mimicry (Galarian Stunfisk): the holder adopts
// the active terrain's type, reverts to its base typing when the terrain ends,
// and re-evaluates through both reference call sites - the `onStart`
// `singleEvent('TerrainChange')` on entry and the global
// `eachEvent('TerrainChange')` that every terrain change runs.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, logHas, everHas, runTrials} = createScaffold();

const POOL = [
  ['Metagross', 'Clear Body', ['Iron Head', 'Protect']],
  ['Milotic', 'Competitive', ['Ice Beam', 'Protect']],
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']],
  ['Falinks', 'Battle Armor', ['Smart Strike', 'Protect']],
  ['Chimecho', 'Levitate', ['Psychic', 'Protect']],
  ['Ariados', 'Swarm', ['Leech Life', 'Protect']],
];
const team = (...heads) => [
  ...heads,
  ...POOL.filter(([species]) => !heads.some(head => head.species === species))
    .map(([species, ability, moves]) => offensive(species, ability, moves)),
].slice(0, 6);

const stunfisk = setOf('Stunfisk-Galar', 'Mimicry', ['Protect', 'Earthquake']);
const terrainUp = (fixture, roster, type) => everHas(fixture, 0, roster,
  p => p.types.length === 1 && p.types[0] === ids.types[type]);
const baseTyping = (fixture, roster) => everHas(fixture, 0, roster,
  p => p.types.length === 2 && p.types.includes(ids.types.ground) && p.types.includes(ids.types.steel));

const TRIALS = [
  {
    name: 'mimicry_enters_under_a_live_terrain',
    // Pincurchin leads and its Electric Surge is already up when Stunfisk is
    // switched in on turn 2: `onStart` runs its own TerrainChange.
    p1: () => team(
      setOf('Pincurchin', 'Electric Surge', ['Protect', 'Thunder Wave']),
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
      // Bench slot: entry 0-1 stay the leads, so the Mimicry holder is listed
      // third and switched in on turn 2.
      stunfisk,
    ),
    p2: () => team(setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: [{switch: 's2'}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {ability: 'mimicry'},
    verify(fixture, session) {
      if (!logHas(session, /\|switch\|p1a: s2\|Stunfisk-Galar/)) return 'Stunfisk never entered';
      if (!logHas(session, /\|-start\|p1a: s2\|typechange\|Electric\|\[from\] ability: Mimicry/)) {
        return 'the entry type change never announced Mimicry';
      }
      if (!terrainUp(fixture, 2, 'electric')) return 'Stunfisk never became Electric';
      return null;
    },
  },
  {
    name: 'mimicry_follows_a_later_terrain_change',
    // Stunfisk is active first; the terrain starts Electric (Pincurchin, its
    // partner) and is replaced by Grassy Terrain on turn 2, so the type follows
    // the global TerrainChange event.
    p1: () => team(
      stunfisk,
      setOf('Pincurchin', 'Electric Surge', ['Protect', 'Thunder Wave']),
    ),
    p2: () => team(setOf('Torterra', 'Shell Armor', ['Grassy Terrain', 'Seed Bomb'])),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['grassyterrain', 'protect']},
    ],
    coverage: {ability: 'mimicry'},
    verify(fixture, session) {
      if (!logHas(session, /\|-start\|p1a: s0\|typechange\|Electric\|\[from\] ability: Mimicry/)) {
        return 'the Electric start typing was never recorded';
      }
      if (!logHas(session, /\|-start\|p1a: s0\|typechange\|Grass\|\[from\] ability: Mimicry/)) {
        return 'the Grassy Terrain change never retyped Mimicry';
      }
      if (!terrainUp(fixture, 0, 'grass')) return 'Stunfisk never became Grass';
      return null;
    },
  },
  {
    name: 'mimicry_reverts_when_steel_roller_clears_the_terrain',
    p1: () => team(
      stunfisk,
      setOf('Pincurchin', 'Electric Surge', ['Protect', 'Thunder Wave']),
    ),
    p2: () => team(setOf('Metagross', 'Clear Body', ['Steel Roller', 'Iron Head', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: [{move: 'steelroller', target: 1}, 'protect']},
    ],
    coverage: {move: 'steelroller'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Steel Roller\|p1a: s0/)) return 'Steel Roller never resolved';
      if (!logHas(session, /\|-activate\|p1a: s0\|ability: Mimicry/)) {
        return 'the revert never announced Mimicry';
      }
      if (!baseTyping(fixture, 0)) return 'Stunfisk never reverted to Ground/Steel';
      return null;
    },
  },
  {
    name: 'mimicry_reverts_when_the_terrain_expires',
    // Five Protect turns let Electric Terrain run out in the residual phase;
    // the field handler's `end` (Field#clearTerrain) runs the global
    // TerrainChange event and Mimicry goes back to Ground/Steel.
    p1: () => team(
      stunfisk,
      setOf('Pincurchin', 'Electric Surge', ['Protect', 'Thunder Wave']),
    ),
    p2: () => team(setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {ability: 'mimicry'},
    verify(fixture, session) {
      if (!terrainUp(fixture, 0, 'electric')) return 'Stunfisk never became Electric';
      const reverted = fixture.steps.some(step => step.expected.turn > 5
        && step.expected.sides[0].pokemon.some(p => p.roster === 0
          && p.types.length === 2 && p.types.includes(ids.types.ground) && p.types.includes(ids.types.steel)));
      if (!reverted) return 'the expiry never reverted the typing';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 38000, artifact: 'more_mimicry.json', debugEnv: 'DEBUG_MIMICRY'});
