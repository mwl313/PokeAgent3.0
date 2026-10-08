// Development-only corpus for the "refusal" ability cluster: Boost refusals
// (White Smoke, Keen Eye, Illuminate), sleep refusals (Vital Spirit) and the
// phazing refusal (Suction Cups). Each scene pairs a legal holder with the
// effect the reference refuses and asserts the refusal actually happened in
// the recorded reference battle; the boundary-by-boundary corpus comparison is
// the real witness.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, logHas, everHas, runTrials} = createScaffold();

const POOL = [
  ['Milotic', 'Competitive', ['Ice Beam', 'Protect']],
  ['Metagross', 'Clear Body', ['Iron Head', 'Protect']],
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

// Boost slots: [atk, def, spa, spd, spe, accuracy, evasion].
const accuracy = id => id === 5;

const TRIALS = [
  {
    name: 'whitesmoke_refuses_charm',
    p1: () => team(setOf('Torkoal', 'White Smoke', ['Protect', 'Body Press'])),
    p2: () => team(setOf('Alcremie', 'Aroma Veil', ['Charm', 'Dazzling Gleam', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: [{move: 'charm', target: 1}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{move: 'charm', target: 1}, 'protect']},
    ],
    coverage: {ability: 'whitesmoke'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Charm\|p1a: s0/)) return 'Charm never resolved';
      if (!logHas(session, /\|-fail\|p1a: s0\|unboost\|\[from\] ability: White Smoke/)) {
        return 'White Smoke never refused the drop';
      }
      const dropped = everHas(fixture, 0, 0, p => p.boosts[1] < 0);
      if (dropped) return 'Torkoal lost Attack anyway';
      return null;
    },
  },
  {
    name: 'keeneye_refuses_mudslap_accuracy',
    p1: () => team(setOf('Pidgeot', 'Keen Eye', ['Protect', 'Aerial Ace'])),
    p2: () => team(setOf('Aggron', 'Sturdy', ['Mud-Slap', 'Iron Head', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: [{move: 'mudslap', target: 1}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{move: 'mudslap', target: 1}, 'protect']},
    ],
    coverage: {ability: 'keeneye'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Mud-Slap\|p1a: s0/)) return 'Mud-Slap never resolved';
      const dropped = everHas(fixture, 0, 0, p => p.boosts.some((b, i) => accuracy(i) && b < 0));
      if (dropped) return 'Keen Eye let the accuracy drop through';
      return null;
    },
  },
  {
    name: 'illuminate_refuses_mudslap_accuracy',
    p1: () => team(setOf('Starmie', 'Illuminate', ['Protect', 'Ice Beam'])),
    p2: () => team(setOf('Aggron', 'Sturdy', ['Mud-Slap', 'Iron Head', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: [{move: 'mudslap', target: 1}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{move: 'mudslap', target: 1}, 'protect']},
    ],
    coverage: {ability: 'illuminate'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Mud-Slap\|p1a: s0/)) return 'Mud-Slap never resolved';
      const dropped = everHas(fixture, 0, 0, p => p.boosts.some((b, i) => accuracy(i) && b < 0));
      if (dropped) return 'Illuminate let the accuracy drop through';
      return null;
    },
  },
  {
    name: 'illuminate_ignores_evasion',
    p1: () => team(setOf('Starmie', 'Illuminate', ['Protect', 'Ice Beam'])),
    p2: () => team(setOf('Absol', 'Super Luck', ['Double Team', 'Protect', 'Night Slash'])),
    script: [
      {p1: ['protect', 'protect'], p2: [{move: 'doubleteam', target: 0}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{move: 'doubleteam', target: 0}, 'protect']},
      {p1: [{move: 'icebeam', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'icebeam', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {ability: 'illuminate'},
    verify(fixture, session) {
      if (!everHas(fixture, 1, 0, p => p.boosts[6] >= 2)) return 'Absol never built up evasion';
      if (!logHas(session, /\|move\|p1a: s0\|Ice Beam\|p2a: s0/)) return 'Ice Beam never resolved';
      const hit = session.battle.log
        .filter(line => line.startsWith('|-damage|p2a: s0|')).length;
      if (!hit) return 'the Illuminate holder never landed through the evasion';
      return null;
    },
  },
  {
    name: 'vitalspirit_refuses_sleep_powder',
    p1: () => team(setOf('Annihilape', 'Vital Spirit', ['Protect', 'Drain Punch'])),
    p2: () => team(setOf('Venusaur', 'Overgrow', ['Sleep Powder', 'Sludge Bomb', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: [{move: 'sleeppowder', target: 1}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{move: 'sleeppowder', target: 1}, 'protect']},
    ],
    coverage: {ability: 'vitalspirit'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Sleep Powder\|p1a: s0/)) return 'Sleep Powder never resolved';
      if (!logHas(session, /\|-immune\|p1a: s0\|\[from\] ability: Vital Spirit/)) {
        return 'Vital Spirit never announced the sleep immunity';
      }
      const slept = everHas(fixture, 0, 0, p => p.status === ids.conditions.slp);
      if (slept) return 'Annihilape fell asleep anyway';
      return null;
    },
  },
  {
    name: 'vitalspirit_refuses_yawn',
    p1: () => team(setOf('Annihilape', 'Vital Spirit', ['Protect', 'Drain Punch'])),
    p2: () => team(setOf('Chimecho', 'Levitate', ['Yawn', 'Psychic', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: [{move: 'yawn', target: 1}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{move: 'yawn', target: 1}, 'protect']},
    ],
    coverage: {ability: 'vitalspirit'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Yawn\|p1a: s0/)) return 'Yawn never resolved';
      const yawned = everHas(fixture, 0, 0, p => p.volatiles.some(v => v.includes('yawn')));
      if (yawned) return 'the Yawn marker landed on a Vital Spirit holder';
      const slept = everHas(fixture, 0, 0, p => p.status === ids.conditions.slp);
      if (slept) return 'Annihilape fell asleep anyway';
      return null;
    },
  },
  {
    name: 'suctioncups_refuses_roar',
    p1: () => team(setOf('Malamar', 'Suction Cups', ['Protect', 'Night Slash'])),
    p2: () => team(setOf('Aggron', 'Sturdy', ['Roar', 'Iron Head', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: [{move: 'roar', target: 1}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{move: 'roar', target: 1}, 'protect']},
    ],
    coverage: {ability: 'suctioncups'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Roar\|p1a: s0/)) return 'Roar never resolved';
      if (!logHas(session, /\|-activate\|p1a: s0\|ability: Suction Cups/)) {
        return 'Suction Cups never announced the refusal';
      }
      // Turn-0 boundaries are team preview, where nobody occupies a slot yet.
      const scripted = fixture.steps.filter(step => step.expected.turn >= 1
        && step.expected.turn <= 2);
      if (scripted.some(step => step.expected.sides[0].pokemon
        .some(p => p.roster === 0 && !p.fainted && p.active_slot !== 0))) {
        return 'Malamar left its slot while the Roar resolved';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 39000, artifact: 'more_refusals.json', debugEnv: 'DEBUG_REFUSALS'});
