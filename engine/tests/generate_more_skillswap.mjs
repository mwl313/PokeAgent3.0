// Development-only corpus for Skill Swap:
// - a foe-targeted swap exchanges both abilities,
// - an ally-targeted swap exchanges them without naming the abilities,
// - `flags.failskillswap` (Stance Change) refuses the exchange, and
// - the incoming abilities' Start callbacks run (Intimidate drops Attack).
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, fillerAfter, foeTeam, foeWith, logHas, everHas, runTrials} = createScaffold();

const TRIALS = [
  {
    name: 'skillswap_swaps_foe_abilities',
    p1: () => [
      setOf('Chimecho', 'Levitate', ['Skill Swap', 'Protect']),
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
      offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
      offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
      offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
      offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
    ],
    p2: () => [
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
      offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
      offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
      offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
      offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
      offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
    ],
    script: [{p1: [{move: 'skillswap', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']}],
    coverage: {move: 'skillswap'},
    verify(fixture, session) {
      if (!logHas(session, /\|-activate\|p1a: s0\|Skill Swap\|Clear Body\|Levitate\|\[of\] p2a: s0/)) {
        return 'the reference never announced the exchange';
      }
      if (!everHas(fixture, 0, 0, p => p.ability === ids.abilities.clearbody)) {
        return 'the user never received the target ability';
      }
      if (!everHas(fixture, 1, 0, p => p.ability === ids.abilities.levitate)) {
        return 'the target never received the user ability';
      }
      return null;
    },
  },
  {
    name: 'skillswap_allies_exchange_abilities',
    p1: () => [
      setOf('Chimecho', 'Levitate', ['Skill Swap', 'Protect']),
      offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
      offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
      offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
      offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
    ],
    p2: () => [
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
      offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
      offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
      offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
      offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
      offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
    ],
    script: [{p1: [{move: 'skillswap', target: -2}, {move: 'bodyslam', target: 1}], p2: ['protect', 'protect']}],
    coverage: {move: 'skillswap'},
    verify(fixture, session) {
      if (!everHas(fixture, 0, 0, p => p.ability === ids.abilities.thickfat)) {
        return 'the ally user never received the partner ability';
      }
      if (!everHas(fixture, 0, 1, p => p.ability === ids.abilities.levitate)) {
        return 'the ally never received the user ability';
      }
      return null;
    },
  },
  {
    name: 'skillswap_refuses_failskillswap',
    p1: () => [
      setOf('Chimecho', 'Levitate', ['Skill Swap', 'Protect']),
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
      offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
      offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
      offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
      offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
    ],
    p2: () => [
      offensive('Aegislash', 'Stance Change', ['Iron Head', 'Protect']),
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
      offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
      offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
      offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
      offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
    ],
    script: [{p1: [{move: 'skillswap', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']}],
    coverage: {move: 'skillswap'},
    verify(fixture, session) {
      if (logHas(session, /\|-activate\|p1a: s0\|Skill Swap\|/)) return 'the refused exchange was announced';
      if (!everHas(fixture, 0, 0, p => p.ability === ids.abilities.levitate)) {
        return 'the user ability changed despite the refusal';
      }
      if (!everHas(fixture, 1, 0, p => p.ability === ids.abilities.stancechange)) {
        return 'the failskillswap ability was exchanged anyway';
      }
      return null;
    },
  },
  {
    name: 'skillswap_incoming_start_callbacks_run',
    p1: () => [
      setOf('Chimecho', 'Levitate', ['Skill Swap', 'Protect']),
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
      offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
      offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
      offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
      offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
    ],
    p2: () => [
      offensive('Incineroar', 'Intimidate', ['Throat Chop', 'Protect']),
      offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
      offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
      offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
      offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
    ],
    script: [{p1: [{move: 'skillswap', target: 1}, 'protect'], p2: [{move: 'throatchop', target: 1}, 'protect']}],
    coverage: {move: 'skillswap'},
    verify(fixture, session) {
      if (!everHas(fixture, 0, 0, p => p.ability === ids.abilities.intimidate)) {
        return 'the incoming Intimidate never reached the user';
      }
      if (!everHas(fixture, 1, 0, p => p.boosts[0] === -1)) {
        return 'the incoming Intimidate Start callback never ran';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 11000, artifact: 'more_skillswap.json', debugEnv: 'DEBUG_SKILLSWAP'});
