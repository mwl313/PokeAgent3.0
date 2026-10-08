// Development-only interaction corpus for the two trap markers:
//
// - Ingrain grounds its holder, pins it in place (a refused drag) and heals a
//   sixteenth of its maximum HP at residual order 7.
// - Octolock pins the target and lowers its Defense and Special Defense every
//   residual while the octolocking Pokémon is still active; the marker ends
//   silently when that source leaves the field.
//
// Every fixture is a complete legal reference battle recorded at every decision
// boundary, including the served request mask (whose `trapped` flag witnesses
// the pin).
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, foeTeam, foeWith, logHas, everHas, runTrials} = createScaffold();

const team = (...heads) => [
  ...heads,
  ...foeTeam().filter(p => !heads.some(head => head.species === p.species)),
].slice(0, 6);
const hpAt = (fixture, side, roster) => fixture.steps.flatMap(step => step.expected.sides[side].pokemon
  .filter(p => p.roster === roster).map(p => p.hp));
const boostsAt = (fixture, side, roster) => fixture.steps.flatMap(step => step.expected.sides[side].pokemon
  .filter(p => p.roster === roster).map(p => p.boosts));
const volatilesAt = (fixture, side, roster) => fixture.steps.flatMap(step => step.expected.sides[side].pokemon
  .filter(p => p.roster === roster).map(p => p.volatiles));
const trappedAt = (fixture, side, slot) => fixture.steps.flatMap(step =>
  (step.expected.sides[side].request_detail?.slots ?? [])[slot]?.trapped ?? []);

const TRIALS = [
  {
    // Ingrain heals its holder every residual and pins it in place.
    name: 'ingrain_heals_and_pins',
    p1: () => team(setOf('Meganium', 'Overgrow', ['Ingrain', 'Protect'])),
    p2: () => foeTeam(),
    script: [
      {p1: ['protect', 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: ['ingrain', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'ingrain'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Ingrain\|/)) return 'Ingrain never executed';
      const heals = session.battle.log.filter(line => line.startsWith('|-heal|p1a: s0|')
        && !/\|\d+\/100/.test(line)).length;
      if (heals < 2) return `expected the residual heal every turn, saw ${heals}`;
      const pinned = trappedAt(fixture, 0, 0).some(Boolean);
      const marked = volatilesAt(fixture, 0, 0).some(list => list.includes('ingrain'));
      return pinned && marked ? null : `ingrain pin not observed (trapped=${pinned} marked=${marked})`;
    },
  },
  {
    // A phazing move cannot drag an Ingrained holder out.
    name: 'ingrain_refuses_drag',
    p1: () => team(setOf('Meganium', 'Overgrow', ['Ingrain', 'Protect'])),
    p2: () => foeWith(setOf('Skarmory', 'Sturdy', ['Whirlwind', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['ingrain', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: [{move: 'whirlwind', target: 1}, 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'ingrain'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Ingrain\|/)) return 'Ingrain never executed';
      if (!logHas(session, /\|move\|p2a: s0\|Whirlwind\|/)) return 'Whirlwind never executed';
      // The holder never leaves: the first boundary after the Whirlwind turn
      // still shows it in slot 0 and no replacement request.
      const after = fixture.steps.find(step => step.expected.turn === 4);
      if (!after) return 'no boundary after the Whirlwind turn';
      const stillActive = after.expected.sides[0].pokemon
        .some(p => p.roster === 0 && p.active_slot === 0);
      return stillActive ? null : 'the Ingrained holder was dragged out';
    },
  },
  {
    // Octolock pins the target and drops its defenses each residual until the
    // octolocking Pokémon leaves the field.
    name: 'octolock_lowers_defenses',
    p1: () => team(setOf('Grapploct', 'Limber', ['Octolock', 'Protect'])),
    p2: () => foeWith(setOf('Rhyperior', 'Solid Rock', ['Earthquake', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'octolock', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: [{switch: 's2'}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'octolock'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Octolock\|/)) return 'Octolock never executed';
      const drops = boostsAt(fixture, 1, 0).filter(b => b[1] <= -1 && b[3] <= -1);
      if (drops.length < 2) return `expected repeated defense drops (${drops.length})`;
      const pinned = trappedAt(fixture, 1, 0).some(Boolean);
      const ended = volatilesAt(fixture, 1, 0).at(-1)?.includes('octolock') === false;
      return pinned && ended ? null : `octolock pin/end not observed (trapped=${pinned} ended=${ended})`;
    },
  },
];

runTrials(TRIALS, {seedBase: 10900, artifact: 'more_traps.json', debugEnv: 'DEBUG_TRAPS'});
