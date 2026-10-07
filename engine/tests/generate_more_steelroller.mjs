// Development-only corpus for Steel Roller:
// - a landed hit clears the active terrain (Electric Terrain from a faster
//   partner lands first in the same turn),
// - the move fails outright without an active terrain,
// - a Protect block leaves the terrain untouched, and
// - a substitute that absorbs the hit still clears the terrain through the
//   substitute condition's `AfterSubDamage` dispatch.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, monAt, runTrials} = createScaffold();

const BODY_TEAM = [
  ['Metagross', 'Clear Body', ['Steel Roller', 'Iron Head', 'Protect']],
  ['Jolteon', 'Volt Absorb', ['Electric Terrain', 'Thunderbolt', 'Protect']],
  ['Milotic', 'Competitive', ['Surf', 'Protect']],
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']],
  ['Falinks', 'Battle Armor', ['Smart Strike', 'Protect']],
  ['Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']],
];
const p1Team = () => BODY_TEAM.map(([species, ability, moves]) => setOf(species, ability, moves));
const p2Team = () => [
  setOf('Snorlax', 'Thick Fat', ['Body Slam', 'Substitute', 'Protect']),
  offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
  offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
  offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
  offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
  offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
];

const terrainId = ids.conditions.electricterrain;
const indexOf = (session, pattern, before = Infinity) =>
  session.battle.log.findIndex((line, index) => index < before && pattern.test(line));
const turnIndex = (session, turn) => session.battle.log.indexOf(`|turn|${turn}`);
const terrainAt = (fixture, turn) => {
  const step = fixture.steps.find(entry => entry.expected.turn >= turn);
  return step ? step.expected.field.some(([id]) => id === terrainId) : null;
};

const TRIALS = [
  {
    name: 'steelroller_clears_terrain',
    p1: p1Team,
    p2: p2Team,
    script: [{
      p1: [{move: 'steelroller', target: 1}, 'electricterrain'],
      p2: [{move: 'bodyslam', target: 1}, 'protect'],
    }],
    coverage: {move: 'steelroller'},
    verify(fixture, session) {
      const next = turnIndex(session, 2);
      const start = indexOf(session, /\|-fieldstart\|move: Electric Terrain/, next);
      const clear = indexOf(session, /\|-fieldend\|move: Electric Terrain/, next);
      const hit = indexOf(session, /\|move\|p1a: s0\|Steel Roller\|/, next);
      if (start < 0) return 'the terrain never started';
      if (hit < 0) return 'Steel Roller never resolved';
      if (clear < 0) return 'Steel Roller never cleared the terrain';
      if (!(start < hit && hit < clear)) return 'the terrain/end ordering is wrong';
      if (terrainAt(fixture, 2) !== false) return 'the terrain was still up at the next boundary';
      return null;
    },
  },
  {
    name: 'steelroller_fails_without_terrain',
    p1: p1Team,
    p2: p2Team,
    script: [{
      p1: [{move: 'steelroller', target: 1}, 'protect'],
      p2: ['protect', 'protect'],
    }],
    coverage: {move: 'steelroller'},
    verify(fixture, session) {
      const next = turnIndex(session, 2);
      if (indexOf(session, /\|-fail\|p1a: s0/, next) < 0) return 'the move never reported the Try failure';
      if (indexOf(session, /\|-damage\|p2a: s0/, next) >= 0) return 'the failed move still dealt damage';
      const step = fixture.steps.find(entry => entry.expected.turn >= 2);
      if (!step) return 'the battle ended before the next boundary';
      const snorlax = step.expected.sides[1].pokemon.find(p => p.roster === 0);
      if (snorlax.hp !== snorlax.max_hp) return 'the failed move still changed the target HP';
      return null;
    },
  },
  {
    name: 'steelroller_protect_leaves_terrain',
    p1: p1Team,
    p2: p2Team,
    script: [
      {p1: ['protect', 'electricterrain'], p2: [{move: 'bodyslam', target: 1}, 'protect']},
      {p1: [{move: 'steelroller', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'steelroller'},
    verify(fixture, session) {
      const until = turnIndex(session, 3);
      if (indexOf(session, /\|-fieldstart\|move: Electric Terrain/, until) < 0) return 'the terrain never started';
      if (indexOf(session, /\|-activate\|p2a: s0\|move: Protect/, until) < 0) return 'the target never Protected';
      if (indexOf(session, /\|-fieldend\|move: Electric Terrain/, until) >= 0) return 'the blocked move still cleared the terrain';
      if (terrainAt(fixture, 3) !== true) return 'the terrain was gone at the next boundary';
      return null;
    },
  },
  {
    name: 'steelroller_substitute_clears_terrain',
    p1: p1Team,
    p2: p2Team,
    script: [
      {p1: ['protect', 'electricterrain'], p2: ['substitute', 'protect']},
      {p1: [{move: 'steelroller', target: 1}, 'protect'], p2: [{move: 'bodyslam', target: 1}, 'protect']},
    ],
    coverage: {move: 'steelroller'},
    verify(fixture, session) {
      const until = turnIndex(session, 3);
      if (indexOf(session, /\|-fieldstart\|move: Electric Terrain/, until) < 0) return 'the terrain never started';
      // A surviving decoy logs `-activate|...|move: Substitute`; a popped one
      // logs only `-end|...|Substitute`.
      if (indexOf(session, /\|(-end\|p2a: s0\|Substitute|-activate\|p2a: s0\|move: Substitute)/, until) < 0) {
        return 'the substitute never absorbed the hit';
      }
      if (indexOf(session, /\|-fieldend\|move: Electric Terrain/, until) < 0) return 'the decoy hit never cleared the terrain';
      if (terrainAt(fixture, 3) !== false) return 'the terrain was still up at the next boundary';
      // The real target must not lose HP to the absorbed hit.
      const step = fixture.steps.find(entry => entry.expected.turn >= 3);
      const snorlax = step.expected.sides[1].pokemon.find(p => p.roster === 0);
      if (snorlax.hp !== 201) return `the absorbed hit still changed the target HP (${snorlax.hp})`;
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 7400, artifact: 'more_steelroller.json', debugEnv: 'DEBUG_STEELROLLER'});
