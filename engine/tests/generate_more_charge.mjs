// Development-only corpus for the Charge volatile and Electromorphosis:
// - Charge raises Special Defense and grants the `charge` volatile, which
//   doubles the holder's Electric base power and is consumed by that move,
// - Electromorphosis grants the same volatile whenever the holder takes a
//   damaging hit (no boost).
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, logHas, monAt, runTrials} = createScaffold();

const POOL = [
  ['Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic']],
  ['Milotic', 'Competitive', ['Surf', 'Protect', 'Ice Beam']],
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect', 'Body Slam']],
  ['Falinks', 'Battle Armor', ['Smart Strike', 'Protect', 'Close Combat']],
  ['Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect', 'Psychic']],
  ['Aggron', 'Sturdy', ['Iron Head', 'Protect', 'Body Slam']],
  ['Ariados', 'Swarm', ['Leech Life', 'Protect', 'Sucker Punch']],
];
const team = (...heads) => [
  ...heads,
  ...POOL.filter(([species]) => !heads.some(head => head.species === species))
    .map(([species, ability, moves]) => offensive(species, ability, moves)),
].slice(0, 6);

// Damage deltas straight out of the protocol stream: the state snapshots only
// record turn boundaries.
const damageDeltas = (log, ident, maxHp) => {
  const hpOf = line => {
    const parts = line.split('|');
    if (parts[2] !== ident) return null;
    const value = parts[3] ? parts[3].split(' ')[0] : '';
    if (value === '0') return 0;
    if (!/^\d+\/\d+$/.test(value)) return null;
    const [hp, max] = value.split('/').map(Number);
    return max === maxHp ? hp : null;
  };
  const deltas = [];
  let previous = null;
  for (const line of log) {
    const hp = hpOf(line);
    if (hp == null) continue;
    if (previous != null && hp < previous) deltas.push(previous - hp);
    previous = hp;
  }
  return deltas;
};

const TRIALS = [
  {
    name: 'chargefamily_charge_doubles_then_is_consumed',
    p1: () => team(setOf('Bellibolt', 'Electromorphosis', ['Charge', 'Charge Beam', 'Protect'])),
    p2: () => team(setOf('Torkoal', 'Shell Armor', ['Eruption', 'Protect', 'Body Press'])),
    script: [
      {p1: [{move: 'charge', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['chargebeam', 'protect'], p2: ['protect', 'protect']},
      {p1: ['chargebeam', 'protect'],
        p2: ['eruption', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'charge'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Charge\|/)) return 'Charge never executed';
      if (!monAt(fixture, 0, 0).some(p => p.volatiles.includes('charge'))) {
        return 'the charge volatile was never recorded';
      }
      if (!monAt(fixture, 0, 0).some(p => p.boosts[3] === 1)) return 'Charge did not raise Special Defense';
      const max = fixture.steps[0].expected.sides[1].pokemon.find(p => p.roster === 0).max_hp;
      const deltas = damageDeltas(session.battle.log, 'p2a: s0', max);
      if (deltas.length < 2) return 'the two Charge Beams were not both recorded';
      if (deltas[0] * 2 < deltas[1] * 3) {
        return `the charged Charge Beam did not double (${deltas[0]} vs ${deltas[1]})`;
      }
      const carried = fixture.steps
        .map(step => step.expected.sides[0].pokemon.find(p => p.roster === 0))
        .filter(Boolean);
      if (!carried.some(p => !p.volatiles.includes('charge')) || carried[0].volatiles.includes('charge')) {
        return 'the charge volatile was not consumed by the Electric move';
      }
      return null;
    },
  },
  {
    name: 'chargefamily_electromorphosis_grants_charge_when_hit',
    p1: () => team(setOf('Bellibolt', 'Electromorphosis', ['Mud Shot', 'Thunderbolt', 'Protect'])),
    p2: () => team(setOf('Torkoal', 'Shell Armor', ['Eruption', 'Protect', 'Body Press'])),
    script: [
      {p1: [{move: 'mudshot', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['eruption', 'protect']},
      {p1: [{move: 'thunderbolt', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'thunderbolt', target: 1}, 'protect'],
        p2: ['eruption', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'charge'},
    verify(fixture, session) {
      if (!logHas(session, /ability: Electromorphosis/)) return 'Electromorphosis never activated';
      const carried = fixture.steps
        .map(step => step.expected.sides[0].pokemon.find(p => p.roster === 0))
        .filter(Boolean);
      if (!carried.some(p => p.volatiles.includes('charge'))) {
        return 'the granted charge volatile was never recorded';
      }
      const max = fixture.steps[0].expected.sides[1].pokemon.find(p => p.roster === 0).max_hp;
      const deltas = damageDeltas(session.battle.log, 'p2a: s0', max);
      if (deltas.length < 2) return 'the two Thunderbolts were not both recorded';
      if (deltas[0] * 2 < deltas[1] * 3) {
        return `the Electromorphosis-charged Thunderbolt did not double (${deltas[0]} vs ${deltas[1]})`;
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 22000, artifact: 'more_charge.json', debugEnv: 'DEBUG_CHARGE'});
