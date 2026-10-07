// Development-only corpus for Gravity:
// - `moves:gravity.condition` is a five-turn pseudo-weather whose `onFieldStart`
//   clears the Fly/Bounce markers (and cancels those queued actions),
// - `isGrounded` treats every active Pokémon as grounded, so a Flying target
//   takes Earthquake while the pseudo-weather is up,
// - `onDisableMove` disables every `flags.gravity` move in the request and
//   `onBeforeMove` refuses them outright,
// - `onModifyAccuracy` raises every numbered accuracy by 6840/4096, which turns
//   Inferno (50) into 83 and Fire Blast (85) into 142.
import {createScaffold} from './fixture_scaffold.mjs';
const {setOf, foeTeam, foeWith, logHas, runTrials} = createScaffold();

const team = (...heads) => [
  ...heads,
  ...foeTeam().filter(p => !heads.some(head => head.species === p.species)),
].slice(0, 6);
const mon = (fixture, index, side, roster) => fixture.steps[index].expected.sides[side].pokemon
  .find(p => p.roster === roster);
/// The boundary recorded after P1's submission in the given turn.
const afterTurn = (fixture, turn) => {
  const index = fixture.steps.findIndex(step =>
    step.side === 'P1' && step.expected.turn === turn
    && step.actions.some(action => action.kind === 'Move' && action.own_slot === 1));
  return index < 0 ? null : index + 1;
};
// Espeon (speed 110) outruns Blaziken's High Jump Kick (speed 80), so the turn
// that starts Gravity is also the turn the already-chosen `flags.gravity` move
// is refused by the condition's BeforeMove handler.
const gravityScript = [
  {p1: [{move: 'gravity'}, 'protect'], p2: [{move: 'protect'}, {move: 'highjumpkick', target: 1}]},
  {p1: ['protect', {move: 'earthquake'}], p2: [{move: 'inferno', target: 1}, 'protect']},
  {p1: ['protect', 'protect'], p2: [{move: 'inferno', target: 1}, 'protect']},
  {p1: ['protect', 'protect'], p2: [{move: 'inferno', target: 1}, 'protect']},
  {p1: ['protect', 'protect'], p2: [{move: 'fireblast', target: 1}, 'protect']},
];

const TRIALS = [
  {
    name: 'gravity_grounds_flying_target_and_boosts_accuracy',
    p1: () => team(
      setOf('Espeon', 'Synchronize', ['Gravity', 'Protect', 'Psychic']),
      setOf('Metagross', 'Clear Body', ['Earthquake', 'Protect', 'Iron Head']),
    ),
    p2: () => team(
      setOf('Charizard', 'Blaze', ['Inferno', 'Protect', 'Fire Blast']),
      setOf('Blaziken', 'Blaze', ['High Jump Kick', 'Protect', 'Close Combat']),
    ),
    script: gravityScript,
    coverage: {move: 'gravity'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Gravity\|/)) return 'Gravity never executed';
      if (!logHas(session, /\|-fieldstart\|move: Gravity/)) return 'the pseudo-weather never started';
      // The already-chosen `flags.gravity` move is refused once Gravity lands.
      if (!logHas(session, /\|cant\|p2b: s1\|move: Gravity\|High Jump Kick/)) {
        return 'the chosen High Jump Kick was not refused';
      }
      // Grounding: Earthquake hits the Flying foe only while Gravity is up.
      const quake = afterTurn(fixture, 2);
      if (quake == null) return 'the Earthquake turn never ran';
      if (!(mon(fixture, quake, 1, 0).hp < mon(fixture, quake - 1, 1, 0).hp)) {
        return 'the Flying target was not grounded';
      }
      // The `flags.gravity` move is disabled in the reference request.
      const disabled = fixture.steps.some(step => step.expected.sides[1].request_detail
        ?.slots?.some(slot => (slot.moves ?? []).some(move => move.id === 402 && move.disabled)));
      if (!disabled) return 'High Jump Kick was not disabled in the request';
      // Accuracy: Fire Blast (85 -> 142 under the 6840/4096 modifier) cannot
      // miss; the two Inferno turns (50 -> 83) exercise the same modifier with
      // a roll that can still fail.
      const lines = session.battle.log;
      const lowAccuracy = lines.filter(line =>
        /\|move\|p2a: s0\|(Inferno|Fire Blast)\|/.test(line)).length;
      if (lowAccuracy < 3) return 'the low-accuracy turns did not all run';
      const blast = lines.findIndex(line => line.startsWith('|move|p2a: s0|Fire Blast|'));
      if (blast < 0) return 'Fire Blast never ran';
      if (lines.slice(blast + 1).some(line => line.startsWith('|-miss|p2a: s0'))) {
        return 'Fire Blast missed under Gravity';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 34000, artifact: 'more_gravity.json', debugEnv: 'DEBUG_GRAVITY'});
