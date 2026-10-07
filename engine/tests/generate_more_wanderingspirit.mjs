// Development-only corpus for Wandering Spirit:
// - a contact hit swaps both abilities through the shared Skill Swap
//   primitive (fail gates and End/Start ordering included),
// - a non-contact hit leaves both abilities alone, and
// - an attacker with a `failskillswap` ability refuses the exchange.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, fillerAfter, logHas, everHas, runTrials} = createScaffold();

const holderTeam = head => [
  head,
  offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
  offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
  offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
  offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
  offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
];
const foeTeam2 = () => [
  offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
  offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
  offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
  offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
  offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
  offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
];

const TRIALS = [
  {
    name: 'wanderingspirit_swaps_on_contact',
    p1: () => holderTeam(setOf('Runerigus', 'Wandering Spirit', ['Earthquake', 'Protect'])),
    p2: () => foeTeam2(),
    script: [{p1: ['earthquake', 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']}],
    coverage: {ability: 'wanderingspirit'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Iron Head\|p1a: s0/)) return 'the contact hit never landed';
      if (!everHas(fixture, 0, 0, p => p.ability === ids.abilities.clearbody)) {
        return 'the holder never received the attacker ability';
      }
      if (!everHas(fixture, 1, 0, p => p.ability === ids.abilities.wanderingspirit)) {
        return 'the attacker never received the holder ability';
      }
      return null;
    },
  },
  {
    name: 'wanderingspirit_ignores_non_contact_hits',
    p1: () => holderTeam(setOf('Runerigus', 'Wandering Spirit', ['Earthquake', 'Protect'])),
    p2: () => [
      offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
      offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
      offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
      offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
      offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
    ],
    script: [{p1: ['earthquake', 'protect'], p2: ['surf', 'protect']}],
    coverage: {ability: 'wanderingspirit'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Surf\|/)) return 'the non-contact hit never landed';
      // Only the first turn's boundary matters: later turns can bring contact
      // hits from other attackers that legitimately swap the ability.
      const afterTurn = fixture.steps[3]?.expected.sides[0].pokemon.find(p => p.roster === 0);
      if (!afterTurn || afterTurn.ability !== ids.abilities.wanderingspirit) {
        return 'a non-contact hit swapped the holder ability';
      }
      return null;
    },
  },
  {
    name: 'wanderingspirit_refused_by_failskillswap',
    p1: () => holderTeam(setOf('Runerigus', 'Wandering Spirit', ['Earthquake', 'Protect'])),
    p2: () => [
      offensive('Aegislash', 'Stance Change', ['Iron Head', 'Protect']),
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
      offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
      offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
      offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
      offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
    ],
    script: [{p1: ['earthquake', 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']}],
    coverage: {ability: 'wanderingspirit'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Iron Head\|p1a: s0/)) return 'the contact hit never landed';
      // Only the first turn's boundary matters: later turns can bring contact
      // hits from other attackers that legitimately swap the ability.
      const holder = fixture.steps[3]?.expected.sides[0].pokemon.find(p => p.roster === 0);
      const attacker = fixture.steps[3]?.expected.sides[1].pokemon.find(p => p.roster === 0);
      if (!holder || holder.ability !== ids.abilities.wanderingspirit) {
        return 'the holder ability changed despite failskillswap';
      }
      if (!attacker || attacker.ability !== ids.abilities.stancechange) {
        return 'the failskillswap ability was exchanged anyway';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 12000, artifact: 'more_wanderingspirit.json', debugEnv: 'DEBUG_WANDERINGSPIRIT'});
