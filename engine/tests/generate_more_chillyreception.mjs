// Development-only corpus for Chilly Reception, the snow-setting pivot:
// - the queued move inserts a `priorityChargeMove` action (queue order 107)
//   that runs before every move and adds the move's one-turn
//   `chillyreception` volatile, whose `onBeforeMovePriority: 100` handler only
//   prints `-prepare` for its own move;
// - the move itself sets Snowscape through the generic weather field and then
//   leaves the field (`selfSwitch`).
import {createScaffold} from './fixture_scaffold.mjs';
const {setOf, foeTeam, foeWith, logHas, runTrials} = createScaffold();

const team = (...heads) => [
  ...heads,
  ...foeTeam().filter(p => !heads.some(head => head.species === p.species)),
].slice(0, 6);

const TRIALS = [
  {
    name: 'chillyreception_sets_snow_and_pivots',
    p1: () => team(setOf('Slowking', 'Regenerator', ['Chilly Reception', 'Protect', 'Surf'])),
    p2: () => foeWith(setOf('Milotic', 'Competitive', ['Surf', 'Protect', 'Ice Beam'])),
    script: [
      {p1: [{move: 'chillyreception'}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'chillyreception'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Chilly Reception\|/)) return 'Chilly Reception never executed';
      if (!logHas(session, /\|-weather\|Snowscape/)) return 'the move never set snow';
      const pivot = fixture.steps.findIndex(step => step.actions.some(action => action.kind === 'Switch'));
      if (pivot < 0) return 'the pivot replacement was never recorded';
      const before = fixture.steps[pivot - 1].expected;
      const outgoing = before.sides[0].pokemon.find(p => p.active_slot === 0);
      if (!outgoing.volatiles.includes('chillyreception')) {
        return `the one-turn volatile is missing at the pivot boundary (${outgoing.volatiles})`;
      }
      const after = fixture.steps[pivot].expected;
      const benched = after.sides[0].pokemon.find(p => p.roster === outgoing.roster);
      if (benched.volatiles.includes('chillyreception')) {
        return 'the volatile survived the switch-out';
      }
      if (after.climate.effective !== 'snowscape') return 'snow did not persist past the pivot';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 30000, artifact: 'more_chillyreception.json', debugEnv: 'DEBUG_CHILLY'});
