// Development-only corpus for the trapping moves:
// - Block and Mean Look add the `trapped` volatile sourced by the user, so the
//   target's request advertises `trapped`,
// - Jaw Lock pins both the target and the user,
// - Spirit Shackle pins through its 100% secondary,
// - a Ghost-type target still receives the marker (and its activation message)
//   but stays immune to the actual trap.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {setOf, offensive, foeTeam, foeWith, logHas, runTrials} = createScaffold();

const team = (...heads) => [
  ...heads,
  ...foeTeam().filter(p => !heads.some(head => head.species === p.species)),
].slice(0, 6);
const trappedAt = (fixture, side, slot) => fixture.steps.flatMap(step => {
  const detail = step.expected.sides[side].request_detail;
  const entry = detail?.slots?.[slot];
  return entry ? [entry.trapped] : [];
});

const TRIALS = [
  {
    name: 'trap_block_pins_the_target',
    p1: () => team(setOf('Froslass', 'Snow Cloak', ['Block', 'Protect'])),
    p2: () => foeTeam(),
    script: [{p1: [{move: 'block', target: 1}, 'protect'], p2: ['protect', 'protect']}],
    coverage: {move: 'block'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Block\|p2a: s0/)) return 'Block never executed';
      if (!logHas(session, /\|-activate\|p2a: s0\|trapped/)) return 'the trap was never announced';
      if (!trappedAt(fixture, 1, 0).includes(true)) return 'the target was never trapped in its request';
      return null;
    },
  },
  {
    name: 'trap_meanlook_pins_the_target',
    p1: () => team(setOf('Gardevoir', 'Synchronize', ['Mean Look', 'Protect'])),
    p2: () => foeTeam(),
    script: [{p1: [{move: 'meanlook', target: 1}, 'protect'], p2: ['protect', 'protect']}],
    coverage: {move: 'meanlook'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Mean Look\|p2a: s0/)) return 'Mean Look never executed';
      if (!trappedAt(fixture, 1, 0).includes(true)) return 'the target was never trapped in its request';
      return null;
    },
  },
  {
    name: 'trap_jawlock_pins_both_sides',
    p1: () => team(setOf('Mabosstiff', 'Intimidate', ['Jaw Lock', 'Protect'])),
    p2: () => foeTeam(),
    script: [{p1: [{move: 'jawlock', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']}],
    coverage: {move: 'jawlock'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Jaw Lock\|p2a: s0/)) return 'Jaw Lock never executed';
      if (!trappedAt(fixture, 1, 0).includes(true)) return 'the target was never trapped';
      if (!trappedAt(fixture, 0, 0).includes(true)) return 'the user was never trapped';
      return null;
    },
  },
  {
    name: 'trap_spiritshackle_pins_through_the_secondary',
    p1: () => team(setOf('Decidueye', 'Overgrow', ['Spirit Shackle', 'Protect'])),
    p2: () => foeTeam(),
    script: [{p1: [{move: 'spiritshackle', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']}],
    coverage: {move: 'spiritshackle'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Spirit Shackle\|p2a: s0/)) return 'Spirit Shackle never executed';
      if (!logHas(session, /\|-damage\|p2a: s0\|/)) return 'Spirit Shackle never connected';
      if (!trappedAt(fixture, 1, 0).includes(true)) return 'the target was never trapped';
      return null;
    },
  },
  {
    name: 'trap_control_ghost_target_is_not_pinned',
    p1: () => team(setOf('Froslass', 'Snow Cloak', ['Block', 'Protect'])),
    p2: () => foeWith(setOf('Gengar', 'Cursed Body', ['Shadow Ball', 'Protect'])),
    script: [{p1: [{move: 'block', target: 1}, 'protect'], p2: ['protect', 'protect']}],
    coverage: {move: 'block'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Block\|/)) return 'Block never executed';
      if (!logHas(session, /\|-fail\|p1a: s0/)) return 'Block never reported the Ghost immunity';
      if (logHas(session, /\|-activate\|p2a: s0\|trapped/)) return 'the marker was announced despite the immunity';
      if (trappedAt(fixture, 1, 0).includes(true)) return 'the Ghost was pinned despite the pseudo-type immunity';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 8800, artifact: 'more_trap.json', debugEnv: 'DEBUG_TRAP'});
