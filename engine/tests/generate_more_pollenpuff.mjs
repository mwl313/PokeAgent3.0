// Development-only corpus for Pollen Puff, whose target decides the payload:
// `moves:pollenpuff.onTryHit` drops the move to zero power and sets
// `move.infiltrates` when the target is an ally (so the heal passes through the
// ally's decoy), `onTryMove` refuses that use while the *user* is under Heal
// Block, and `onHit` heals the ally for half of its maximum HP - a refused heal
// is `NOT_FAIL`, which fails the action without the move-loop Update pair.
import {createScaffold} from './fixture_scaffold.mjs';
const {setOf, foeTeam, foeWith, logHas, runTrials} = createScaffold();

const team = (...heads) => [
  ...heads,
  ...foeTeam().filter(p => !heads.some(head => head.species === p.species)),
].slice(0, 6);
const mon = (fixture, index, side, roster) => fixture.steps[index].expected.sides[side].pokemon
  .find(p => p.roster === roster);
/// The boundary recorded just before and just after the Pollen Puff turn.
const attackBounds = fixture => {
  const attack = fixture.steps.findIndex(step => step.actions.some(action =>
    action.kind === 'Move' && action.own_slot === 0 && action.target_location !== 0));
  if (attack < 0) return null;
  return {before: attack - 1, after: attack + 1};
};

const TRIALS = [
  {
    name: 'pollenpuff_heals_ally_through_its_decoy',
    p1: () => team(
      setOf('Meganium', 'Overgrow', ['Pollen Puff', 'Protect', 'Giga Drain']),
      setOf('Metagross', 'Clear Body', ['Substitute', 'Protect', 'Iron Head']),
    ),
    p2: () => foeWith(setOf('Milotic', 'Competitive', ['Surf', 'Protect', 'Ice Beam'])),
    script: [
      {p1: ['protect', {move: 'substitute'}], p2: ['protect', 'protect']},
      // The healed ally attacks a protecting foe, so nothing blocks the heal
      // and the decoy stays untouched.
      {p1: [{move: 'pollenpuff', target: -2}, {move: 'ironhead', target: 1}], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'pollenpuff'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Pollen Puff\|/)) return 'Pollen Puff never executed';
      if (!logHas(session, /\|-heal\|p1b: s1\|/)) return 'the ally was never healed';
      const at = attackBounds(fixture);
      if (!at) return 'the heal turn never ran';
      const before = mon(fixture, at.before, 0, 1);
      const after = mon(fixture, at.after, 0, 1);
      if (!(after.hp > before.hp)) {
        return `the ally did not gain HP (${before.hp} -> ${after.hp})`;
      }
      if (!after.volatiles.includes('substitute')) return 'the heal consumed the ally decoy';
      return null;
    },
  },
  {
    name: 'pollenpuff_damages_foe',
    p1: () => team(setOf('Meganium', 'Overgrow', ['Pollen Puff', 'Protect', 'Giga Drain'])),
    p2: () => foeWith(setOf('Meowscarada', 'Overgrow', ['Protect', 'Flower Trick'])),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'pollenpuff', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'pollenpuff'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Pollen Puff\|/)) return 'Pollen Puff never executed';
      if (logHas(session, /\|-heal\|p2a/)) return 'the foe-targeted use healed instead of damaging';
      const at = attackBounds(fixture);
      if (!at) return 'the attack turn never ran';
      const before = mon(fixture, at.before, 1, 0);
      const after = mon(fixture, at.after, 1, 0);
      if (!(after.hp < before.hp)) return `the foe did not take damage (${before.hp} -> ${after.hp})`;
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 32000, artifact: 'more_pollenpuff.json', debugEnv: 'DEBUG_POLLENPUFF'});
