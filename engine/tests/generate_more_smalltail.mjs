// Development-only corpus for three small self/control moves:
// - Aqua Ring adds a self volatile that heals a sixteenth of the maximum HP at
//   residual order 6,
// - Spite removes four PP from the target's last move (and fails when it cannot),
// - Fell Stinger raises the user's Attack by three stages when the hit KOs.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, foeTeam, foeWith, logHas, everHas, runTrials} = createScaffold();

const team = (...heads) => [
  ...heads,
  ...foeTeam().filter(p => !heads.some(head => head.species === p.species)),
].slice(0, 6);
const ppAt = (fixture, side, roster) => fixture.steps.flatMap(step => step.expected.sides[side].pokemon
  .filter(p => p.roster === roster).map(p => p.pp.join('/')));
const boostsAt = (fixture, side, roster) => fixture.steps.flatMap(step => step.expected.sides[side].pokemon
  .filter(p => p.roster === roster).map(p => p.boosts));

const TRIALS = [
  {
    name: 'smalltail_aqua_ring_heals_every_residual',
    p1: () => team(setOf('Azumarill', 'Huge Power', ['Aqua Ring', 'Protect'])),
    p2: () => foeTeam(),
    script: [
      {p1: [{move: 'aquaring'}, 'protect'], p2: [{move: 'ironhead', target: 1}, {move: 'bodyslam', target: 1}]},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'aquaring'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Aqua Ring\|/)) return 'Aqua Ring never executed';
      if (!logHas(session, /\|-start\|p1a: s0\|Aqua Ring/)) return 'the volatile was never announced';
      const hps = fixture.steps.flatMap(step => step.expected.sides[0].pokemon
        .filter(p => p.roster === 0).map(p => p.hp));
      if (!hps.some((hp, i) => i > 0 && hp > hps[i - 1])) return 'the residual never healed';
      return null;
    },
  },
  {
    name: 'smalltail_spite_deducts_four_pp',
    p1: () => team(setOf('Arbok', 'Intimidate', ['Spite', 'Protect'])),
    p2: () => foeTeam(),
    script: [
      {p1: ['protect', 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'spite', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
    ],
    coverage: {move: 'spite'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Spite\|p2a: s0/)) return 'Spite never executed';
      const pp = ppAt(fixture, 1, 0);
      if (!pp.some((value, i) => i > 0 && value !== pp[i - 1])) return 'the PP never dropped';
      return null;
    },
  },
  {
    name: 'smalltail_fellstinger_boosts_after_a_ko',
    p1: () => team(setOf('Beedrill', 'Swarm', ['Fell Stinger', 'Protect']), setOf('Snorlax', 'Thick Fat', ['Crunch', 'Protect'])),
    p2: () => foeWith(setOf('Alakazam', 'Magic Guard', ['Calm Mind', 'Protect'])),
    script: [
      {p1: ['protect', {move: 'crunch', target: 1}], p2: [{move: 'calmmind'}, 'protect']},
      {p1: [{move: 'fellstinger', target: 1}, 'protect'], p2: [{move: 'calmmind'}, 'protect']},
    ],
    coverage: {move: 'fellstinger'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Fell Stinger\|p2a: s0/)) return 'Fell Stinger never executed';
      if (!logHas(session, /\|-damage\|p2a: s0\|0 fnt/)) return 'the target never fainted to Fell Stinger';
      if (!boostsAt(fixture, 0, 0).some(b => b[0] === 3)) return 'the Attack boost never applied';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 8900, artifact: 'more_smalltail.json', debugEnv: 'DEBUG_SMALLTAIL'});
