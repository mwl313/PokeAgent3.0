// Development-only corpus for the ability-transfer moves:
// - Entrainment gives the target the user's ability (refused for the same
//   ability, a `cantsuppress`/Truant target ability or a `noentrain` user),
// - Role Play gives the user the target's ability,
// - Simple Beam replaces the target's ability with Simple.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, logHas, monAt, runTrials} = createScaffold();

const POOL = [
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

const TRIALS = [
  {
    name: 'abilitytransfer_entrainment_copies_user_ability',
    p1: () => team(setOf('Alcremie', 'Aroma Veil', ['Entrainment', 'Protect', 'Dazzling Gleam'])),
    p2: () => team(setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic'])),
    script: [
      {p1: [{move: 'entrainment', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'entrainment', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'entrainment'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Entrainment\|/)) return 'Entrainment never executed';
      if (!monAt(fixture, 1, 0).some(p => p.ability === ids.abilities.aromaveil)) {
        return 'the target never adopted the user ability';
      }
      const fails = session.battle.log.filter(line => line.startsWith('|-fail|p1a: s0')).length;
      if (!fails) return 'the second Entrainment did not fail on an identical ability';
      return null;
    },
  },
  {
    name: 'abilitytransfer_roleplay_copies_target_ability',
    p1: () => team(setOf('Alakazam', 'Synchronize', ['Role Play', 'Protect', 'Psychic'])),
    p2: () => team(setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic'])),
    script: [
      {p1: [{move: 'roleplay', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'roleplay', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'roleplay'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Role Play\|/)) return 'Role Play never executed';
      if (!monAt(fixture, 0, 0).some(p => p.ability === ids.abilities.clearbody)) {
        return 'the user never adopted the target ability';
      }
      const fails = session.battle.log.filter(line => line.startsWith('|-fail|p1a: s0')).length;
      if (!fails) return 'the second Role Play did not fail on an identical ability';
      return null;
    },
  },
  {
    name: 'abilitytransfer_simplebeam_replaces_target_ability',
    p1: () => team(setOf('Audino', 'Healer', ['Simple Beam', 'Protect', 'Dazzling Gleam'])),
    p2: () => team(setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic'])),
    script: [
      {p1: [{move: 'simplebeam', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'simplebeam', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'simplebeam'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Simple Beam\|/)) return 'Simple Beam never executed';
      if (!monAt(fixture, 1, 0).some(p => p.ability === ids.abilities.simple)) {
        return 'the target ability never became Simple';
      }
      const fails = session.battle.log.filter(line => line.startsWith('|-fail|p1a: s0')).length;
      if (!fails) return 'the second Simple Beam did not fail on an already-Simple target';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 25000, artifact: 'more_ability_transfer.json', debugEnv: 'DEBUG_ABILITYTRANSFER'});
