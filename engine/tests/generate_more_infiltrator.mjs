// Development-only corpus for `abilities:infiltrator.onModifyMove`, which sets
// `move.infiltrates` for every move the holder uses: the target's decoy
// (`moves:substitute.condition.onTryPrimaryHit`) returns early, so the hit
// lands on the Pokémon while the decoy keeps every point of its HP. The paired
// scene uses the same species and move with Frisk instead of Infiltrator, where
// the decoy absorbs the hit and the Pokémon never loses HP.
import {createScaffold} from './fixture_scaffold.mjs';
const {setOf, foeTeam, foeWith, logHas, runTrials} = createScaffold();

const team = (...heads) => [
  ...heads,
  ...foeTeam().filter(p => !heads.some(head => head.species === p.species)),
].slice(0, 6);

/// Script that puts a decoy up on P2's lead and then attacks it. The decoy
/// holder uses Body Slam into the protecting partner on the attack turn so
/// nothing blocks or disturbs the hit under test.
const script = [
  {p1: ['protect', 'protect'], p2: [{move: 'substitute'}, 'protect']},
  {p1: [{move: 'airslash', target: 1}, 'protect'], p2: [{move: 'bodyslam', target: 1}, 'protect']},
];
const lead = (fixture, index) => fixture.steps[index].expected.sides[1].pokemon.find(p => p.roster === 0);
/// The lead's boundary recorded just before the attack turn and every boundary
/// recorded after it, in order.
const afterAttack = fixture => {
  const attack = fixture.steps.findIndex(step => step.actions.some(action =>
    action.kind === 'Move' && action.own_slot === 0 && action.target_location === 1));
  if (attack < 0) return null;
  return {
    before: lead(fixture, attack - 1),
    after: fixture.steps.slice(attack + 1).map((_, i) => lead(fixture, attack + 1 + i)),
  };
};

const TRIALS = [
  {
    name: 'infiltrator_ignores_the_decoy',
    p1: () => team(setOf('Noivern', 'Infiltrator', ['Air Slash', 'Protect', 'Boomburst'])),
    p2: () => foeWith(setOf('Torterra', 'Shell Armor', ['Substitute', 'Protect', 'Body Slam'])),
    script,
    coverage: {move: 'airslash'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Air Slash\|/)) return 'Air Slash never executed';
      const after = afterAttack(fixture);
      if (!after) return 'the attack never ran';
      const hit = after.after.find(p => p.hp < after.before.hp);
      if (!hit) return 'the infiltrated hit never damaged the Pokémon';
      if (!hit.volatiles.includes('substitute')) return 'the decoy did not survive an Infiltrator hit';
      return null;
    },
  },
  {
    name: 'frisk_user_decoy_absorbs',
    p1: () => team(setOf('Noivern', 'Frisk', ['Air Slash', 'Protect', 'Boomburst'])),
    p2: () => foeWith(setOf('Torterra', 'Shell Armor', ['Substitute', 'Protect', 'Body Slam'])),
    script,
    coverage: {move: 'airslash'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Air Slash\|/)) return 'Air Slash never executed';
      const after = afterAttack(fixture);
      if (!after) return 'the attack never ran';
      const broke = after.after.find(p => !p.volatiles.includes('substitute'));
      if (!broke) return 'the decoy never broke';
      if (broke.hp !== after.before.hp) {
        return `the decoy let damage through (${broke.hp}/${after.before.hp})`;
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 29000, artifact: 'more_infiltrator.json', debugEnv: 'DEBUG_INFILTRATOR'});
