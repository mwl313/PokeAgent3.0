// Development-only corpus for Aroma Veil:
// - the holder refuses Taunt on itself and on an adjacent ally (the public
//   block message names the holder), and
// - a Mold Breaker user's Taunt ignores the breakable gate.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, fillerAfter, foeTeam, foeWith, logHas, everHas, runTrials} = createScaffold();

const veilTeam = tail => [
  setOf('Aromatisse', 'Aroma Veil', ['Dazzling Gleam', 'Protect']),
  ...foeTeam().filter(p => !['Aromatisse', 'Metagross'].includes(p.species)).slice(0, 1),
  ...tail,
].slice(0, 6);
const fillerX = [
  offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
  offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
  offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
  offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
  offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
];

const TRIALS = [
  {
    name: 'aromaveil_blocks_taunt_on_holder',
    p1: () => [
      setOf('Aromatisse', 'Aroma Veil', ['Dazzling Gleam', 'Protect']),
      ...fillerX.slice(0, 5),
    ],
    p2: () => foeWith(offensive('Absol', 'Super Luck', ['Taunt', 'Protect'])),
    script: [{p1: ['dazzlinggleam', 'protect'], p2: [{move: 'taunt', target: 1}, 'protect']}],
    coverage: {ability: 'aromaveil'},
    verify(fixture, session) {
      if (!logHas(session, /\|-block\|p1a: s0\|ability: Aroma Veil/)) {
        return 'Aroma Veil never announced the block';
      }
      if (everHas(fixture, 0, 0, p => p.volatiles.includes('taunt'))) {
        return 'the holder was taunted through Aroma Veil';
      }
      return null;
    },
  },
  {
    name: 'aromaveil_blocks_taunt_on_ally',
    p1: () => [
      setOf('Aromatisse', 'Aroma Veil', ['Dazzling Gleam', 'Protect']),
      ...fillerX.slice(0, 5),
    ],
    p2: () => foeWith(offensive('Absol', 'Super Luck', ['Taunt', 'Protect'])),
    script: [{p1: ['dazzlinggleam', {move: 'ironhead', target: 1}], p2: [{move: 'taunt', target: 2}, 'protect']}],
    coverage: {ability: 'aromaveil'},
    verify(fixture, session) {
      if (!logHas(session, /\|-block\|p1b: s1\|ability: Aroma Veil\|\[of\] p1a: s0/)) {
        return 'the ally block never named the Aroma Veil holder';
      }
      if (everHas(fixture, 0, 1, p => p.volatiles.includes('taunt'))) {
        return 'the ally was taunted through Aroma Veil';
      }
      return null;
    },
  },
  {
    name: 'aromaveil_control_taunt_applies',
    p1: () => [
      offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
      ...fillerX.slice(0, 5),
    ],
    p2: () => foeWith(offensive('Absol', 'Super Luck', ['Taunt', 'Protect'])),
    script: [{p1: ['bodyslam', 'protect'], p2: [{move: 'taunt', target: 1}, 'protect']}],
    coverage: {ability: 'aromaveil'},
    verify(fixture, session) {
      if (logHas(session, /ability: Aroma Veil/)) return 'Aroma Veil activated without a holder';
      if (!everHas(fixture, 0, 0, p => p.volatiles.includes('taunt'))) {
        return 'Taunt did not apply without Aroma Veil';
      }
      return null;
    },
  },
  {
    name: 'aromaveil_suppressed_by_mold_breaker',
    p1: () => [
      setOf('Aromatisse', 'Aroma Veil', ['Dazzling Gleam', 'Protect']),
      ...fillerX.slice(0, 5),
    ],
    p2: () => foeWith(offensive('Pangoro', 'Mold Breaker', ['Taunt', 'Protect'])),
    script: [{p1: ['dazzlinggleam', 'protect'], p2: [{move: 'taunt', target: 1}, 'protect']}],
    coverage: {ability: 'aromaveil'},
    verify(fixture, session) {
      if (logHas(session, /ability: Aroma Veil/)) {
        return 'the suppressed gate still announced itself';
      }
      if (!everHas(fixture, 0, 0, p => p.volatiles.includes('taunt'))) {
        return 'Mold Breaker failed to suppress Aroma Veil';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 10000, artifact: 'more_aromaveil.json', debugEnv: 'DEBUG_AROMAVEIL'});
