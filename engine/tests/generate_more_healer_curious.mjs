// Development-only corpus for two legal-ability-tail entries:
// - Healer rolls 3/10 at residual order 5/sub-order 3 to cure a statused
//   adjacent ally and reveals itself with each successful cure.
// - Curious Medicine clears every adjacent ally's stat boosts on entry.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, fillerAfter, foeTeam, foeWith, logHas, everHas, runTrials} = createScaffold();

const TRIALS = [
  {
    name: 'healer_cures_statused_ally',
    p1: () => [
      setOf('Audino', 'Healer', ['Dazzling Gleam', 'Protect']),
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
      offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
      offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
      offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
      offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
    ],
    p2: () => foeWith(offensive('Absol', 'Super Luck', ['Will-O-Wisp', 'Protect'])),
    script: [
      {p1: ['protect', {move: 'ironhead', target: 1}], p2: [{move: 'willowisp', target: 2}, 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {ability: 'healer'},
    verify(fixture, session) {
      if (!everHas(fixture, 0, 1, p => p.status === ids.conditions.brn)) {
        return 'the adjacent ally was never burned';
      }
      if (!logHas(session, /\|-activate\|p1a: s0\|ability: Healer/)) {
        return 'Healer never activated across the attempted seeds';
      }
      if (!everHas(fixture, 0, 1, p => p.status === 0)) {
        return 'the burn was never cured';
      }
      return null;
    },
  },
  {
    name: 'healer_control_without_holder',
    p1: () => [
      offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
      offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
      offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
      offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
      offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
    ],
    p2: () => foeWith(offensive('Absol', 'Super Luck', ['Will-O-Wisp', 'Protect'])),
    script: [
      {p1: ['protect', {move: 'ironhead', target: 1}], p2: [{move: 'willowisp', target: 2}, 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {ability: 'healer'},
    verify(fixture, session) {
      if (logHas(session, /ability: Healer/)) return 'Healer activated without a holder';
      if (!everHas(fixture, 0, 1, p => p.status === ids.conditions.brn)) {
        return 'the control ally was never burned';
      }
      return null;
    },
  },
  {
    name: 'curiousmedicine_clears_ally_boosts',
    p1: () => [
      offensive('Metagross', 'Clear Body', ['Iron Defense', 'Iron Head', 'Protect']),
      offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
      setOf('Slowking-Galar', 'Curious Medicine', ['Psychic', 'Protect']),
      offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
      offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
      offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
    ],
    p2: () => foeTeam(),
    script: [
      {p1: ['irondefense', 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'ironhead', target: 1}, {switch: 's2'}], p2: ['protect', 'protect']},
    ],
    coverage: {ability: 'curiousmedicine'},
    verify(fixture, session) {
      if (!everHas(fixture, 0, 0, p => p.boosts[1] === 2)) {
        return 'the ally never accumulated the Defense boost';
      }
      if (!logHas(session, /\|-clearboost\|p1a: s0\|\[from\] ability: Curious Medicine/)) {
        return 'Curious Medicine never announced the clear';
      }
      // Once the holder is active, the ally's boosts must be gone.
      const switched = fixture.steps.find(step => step.expected.sides[0].pokemon
        .some(p => p.roster === 2 && p.active_slot !== null));
      if (!switched) return 'Slowking-Galar never entered the field';
      const metagross = switched.expected.sides[0].pokemon.find(p => p.roster === 0);
      if (!metagross || metagross.boosts.some(b => b !== 0)) {
        return 'the ally boosts were not cleared on entry';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 9000, artifact: 'more_healer_curious.json', debugEnv: 'DEBUG_HEALER_CURIOUS'});
