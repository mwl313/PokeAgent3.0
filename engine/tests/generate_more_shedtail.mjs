// Development-only corpus for Shed Tail, the second `selfSwitch` payload:
// - `moves:shedtail.onTryHit` refuses the move when the user cannot switch, is
//   commanded, already carries a decoy, or holds `hp <= ceil(maxHP/2)`;
// - the shared `substitute` volatileStatus creates the decoy with
//   floor(maxHP/4) HP (ending any `partiallytrapped` volatile) and
//   `moves:shedtail.onHit` then pays `ceil(maxHP/2)` as direct damage;
// - the pivot replacement runs `copyVolatileFrom(..., 'shedtail')`, which
//   transfers only the decoy and no boost stages.
import {createScaffold} from './fixture_scaffold.mjs';
const {setOf, foeTeam, foeWith, logHas, runTrials} = createScaffold();

const team = (...heads) => [
  ...heads,
  ...foeTeam().filter(p => !heads.some(head => head.species === p.species)),
].slice(0, 6);

const TRIALS = [
  {
    name: 'shedtail_carries_only_the_decoy_and_pays_half_max_hp',
    p1: () => team(setOf('Sceptile', 'Overgrow', ['Swords Dance', 'Shed Tail', 'Protect', 'Leaf Blade'])),
    p2: () => foeWith(setOf('Milotic', 'Competitive', ['Surf', 'Protect', 'Ice Beam'])),
    script: [
      {p1: [{move: 'swordsdance'}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'shedtail'}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'shedtail'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Shed Tail\|/)) return 'Shed Tail never executed';
      if (!logHas(session, /\|-start\|p1a: s0\|Substitute\|\[from\] move: Shed Tail/)) {
        return 'the decoy never started from Shed Tail';
      }
      const pivot = fixture.steps.findIndex((step, i) => i > 0 && step.side === 'P1'
        && step.actions.some(action => action.kind === 'Switch'));
      if (pivot < 1) return 'the pivot replacement was never recorded';
      const before = fixture.steps[pivot - 1].expected.sides[0];
      const after = fixture.steps[pivot].expected.sides[0];
      const outgoing = before.pokemon.find(p => p.active_slot === 0);
      const incoming = after.pokemon.find(p => p.active_slot === 0);
      if (!outgoing || !incoming || incoming.roster === outgoing.roster) {
        return 'the acting slot did not change occupant';
      }
      // The scene opens with every slot protecting, so the user is at full HP
      // when it pays the cost: the recorded post-cost HP is `maxHP -
      // ceil(maxHP/2)`.
      const full = fixture.steps[0].expected.sides[0].pokemon
        .find(p => p.roster === outgoing.roster);
      if (full.hp !== outgoing.max_hp) return 'the user was not at full HP before Shed Tail';
      const expected = outgoing.max_hp - Math.ceil(outgoing.max_hp / 2);
      const benched = after.pokemon.find(p => p.roster === outgoing.roster);
      if (benched.hp !== expected) {
        return `the half-maximum cost is wrong (${outgoing.hp} -> ${benched.hp}, want ${expected})`;
      }
      if (!incoming.volatiles.includes('substitute')) return 'the decoy did not carry over';
      if (incoming.volatiles.length !== 1) {
        return `Shed Tail carried more than the decoy (${incoming.volatiles})`;
      }
      if (incoming.boosts.some(stage => stage !== 0)) {
        return `Shed Tail copied boost stages it should not (${incoming.boosts})`;
      }
      if (benched.volatiles.length) return `the outgoing volatiles were not cleared (${benched.volatiles})`;
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 28000, artifact: 'more_shedtail.json', debugEnv: 'DEBUG_SHEDTAIL'});
