// Development-only corpus for small control moves:
// - Venoshock doubles its BasePower against a poisoned target,
// - Topsy-Turvy inverts every nonzero boost stage of the target (and fails when
//   they are all zero),
// - Clear Smog damages first and then resets the target's stages.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, foeTeam, foeWith, logHas, everHas, runTrials} = createScaffold();

const team = (...heads) => [
  ...heads,
  ...foeTeam().filter(p => !heads.some(head => head.species === p.species)),
].slice(0, 6);
const boostsAt = (fixture, side, roster) => fixture.steps.flatMap(step => step.expected.sides[side].pokemon
  .filter(p => p.roster === roster).map(p => p.boosts));

const TRIALS = [
  {
    name: 'smallcontrol_venoshock_doubles_against_a_poisoned_target',
    p1: () => team(setOf('Arbok', 'Intimidate', ['Venoshock', 'Toxic', 'Protect'])),
    p2: () => foeTeam(),
    script: [
      {p1: [{move: 'toxic', target: 2}, 'protect'], p2: [{move: 'ironhead', target: 1}, {move: 'bodyslam', target: 1}]},
      {p1: [{move: 'venoshock', target: 2}, 'protect'], p2: ['protect', {move: 'bodyslam', target: 1}]},
    ],
    coverage: {move: 'venoshock'},
    verify(fixture, session) {
      if (!logHas(session, /\|-status\|p2b: s1\|tox/)) return 'the target was never poisoned';
      if (!logHas(session, /\|move\|p1a: s0\|Venoshock\|p2b: s1/)) return 'Venoshock never executed';
      if (!logHas(session, /\|-damage\|p2b: s1\|/)) return 'Venoshock never connected';
      return null;
    },
  },
  {
    name: 'smallcontrol_venoshock_control_unpoisoned_target',
    p1: () => team(setOf('Arbok', 'Intimidate', ['Venoshock', 'Protect'])),
    p2: () => foeTeam(),
    script: [{p1: [{move: 'venoshock', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']}],
    coverage: {move: 'venoshock'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Venoshock\|p2a: s0/)) return 'Venoshock never executed';
      if (logHas(session, /\|-status\|p2a: s0\|/)) return 'the control target was statused';
      return null;
    },
  },
  {
    name: 'smallcontrol_topsyturvy_inverts_the_targets_stages',
    p1: () => team(setOf('Malamar', 'Contrary', ['Topsy-Turvy', 'Protect'])),
    p2: () => foeWith(setOf('Alakazam', 'Magic Guard', ['Nasty Plot', 'Shadow Ball', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: [{move: 'nastyplot'}, 'protect']},
      {p1: [{move: 'topsyturvy', target: 1}, 'protect'], p2: [{move: 'nastyplot'}, 'protect']},
    ],
    coverage: {move: 'topsyturvy'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Topsy-Turvy\|/)) return 'Topsy-Turvy never executed';
      const spa = boostsAt(fixture, 1, 0).map(b => b[2]);
      if (!spa.some((value, i) => i > 0 && value === -spa[i - 1] && value !== 0)) {
        return 'the target stages were never inverted';
      }
      return null;
    },
  },
  {
    name: 'smallcontrol_clearsmog_resets_the_targets_stages',
    p1: () => team(setOf('Gengar', 'Cursed Body', ['Clear Smog', 'Protect'])),
    p2: () => foeWith(setOf('Alakazam', 'Magic Guard', ['Nasty Plot', 'Shadow Ball', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: [{move: 'nastyplot'}, 'protect']},
      {p1: [{move: 'clearsmog', target: 1}, 'protect'], p2: [{move: 'nastyplot'}, 'protect']},
    ],
    coverage: {move: 'clearsmog'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Clear Smog\|p2a: s0/)) return 'Clear Smog never executed';
      if (!logHas(session, /\|-damage\|p2a: s0\|/)) return 'Clear Smog never connected';
      const spa = boostsAt(fixture, 1, 0).map(b => b[2]);
      if (!spa.some((value, i) => i > 0 && value === 0 && spa[i - 1] > 0)) {
        return 'the target stages were never reset';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 8700, artifact: 'more_smallcontrol.json', debugEnv: 'DEBUG_SMALLCONTROL'});
