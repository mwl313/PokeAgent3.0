// Development-only corpus for Dragon Darts, the `smartTarget` double hit:
// `getSmartTargets` resolves the action against the chosen target and that
// target's first live adjacent ally, and `hitStepMoveHitLoop` gives each hit
// one entry of that list (`targets[hit - 1]`). A target that any hit step
// refuses (protection, immunity, accuracy) clears the smart-target flag, and
// the remaining hits then resolve like a normal multi-hit move against the
// surviving target.
import {createScaffold} from './fixture_scaffold.mjs';
const {setOf, foeTeam, foeWith, logHas, runTrials} = createScaffold();

const team = (...heads) => [
  ...heads,
  ...foeTeam().filter(p => !heads.some(head => head.species === p.species)),
].slice(0, 6);
const hpAt = (fixture, index, roster) => fixture.steps[index].expected.sides[1].pokemon
  .find(p => p.roster === roster).hp;
/// The boundary recorded after the Dragon Darts turn resolves.
const afterAttack = fixture => {
  const attack = fixture.steps.findIndex(step => step.actions.some(action =>
    action.kind === 'Move' && action.own_slot === 0 && action.target_location === 1));
  if (attack < 0) return null;
  return {before: attack - 1, after: attack + 1};
};

const TRIALS = [
  {
    name: 'dragondarts_splits_across_both_foes',
    p1: () => team(setOf('Dragapult', 'Clear Body', ['Dragon Darts', 'Protect', 'Dragon Pulse'])),
    p2: () => foeWith(setOf('Milotic', 'Competitive', ['Surf', 'Protect', 'Ice Beam'])),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'dragondarts', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'dragondarts'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Dragon Darts\|/)) return 'Dragon Darts never executed';
      const at = afterAttack(fixture);
      if (!at) return 'the attack never ran';
      const first = hpAt(fixture, at.before, 0) > hpAt(fixture, at.after, 0);
      const second = hpAt(fixture, at.before, 1) > hpAt(fixture, at.after, 1);
      if (!first || !second) {
        return `the hits did not split (lead ${hpAt(fixture, at.before, 0)}->${hpAt(fixture, at.after, 0)}, partner ${hpAt(fixture, at.before, 1)}->${hpAt(fixture, at.after, 1)})`;
      }
      return null;
    },
  },
  {
    // A Fairy partner is immune to the Dragon hit: the reference clears the
    // smart-target flag and spends both hits on the chosen target.
    name: 'dragondarts_immune_ally_keeps_both_hits_on_target',
    p1: () => team(setOf('Dragapult', 'Clear Body', ['Dragon Darts', 'Protect', 'Dragon Pulse'])),
    p2: () => team(
      setOf('Milotic', 'Competitive', ['Surf', 'Protect', 'Ice Beam']),
      setOf('Clefable', 'Unaware', ['Protect', 'Moonblast', 'Calm Mind']),
    ),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'dragondarts', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'dragondarts'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Dragon Darts\|/)) return 'Dragon Darts never executed';
      const at = afterAttack(fixture);
      if (!at) return 'the attack never ran';
      if (hpAt(fixture, at.before, 1) !== hpAt(fixture, at.after, 1)) {
        return 'the Fairy partner took Dragon damage';
      }
      if (!(hpAt(fixture, at.before, 0) > hpAt(fixture, at.after, 0))) {
        return 'the chosen target did not take the hits';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 31000, artifact: 'more_dragondarts.json', debugEnv: 'DEBUG_DRAGONDARTS'});
