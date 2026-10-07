// Development-only corpus for Baton Pass, the `selfSwitch: 'copyvolatile'`
// pivot:
// - `moves:batonpass.onHit` fails the move when the user's side has no switch
//   target (or the user is commanded);
// - on success the user leaves the field and the incoming Pokémon runs the
//   reference `copyVolatileFrom`: it adopts the outgoing boost stages and a
//   shallow copy of every volatile whose condition does not declare `noCopy`
//   (so Substitute and Focus Energy carry over while Yawn and the
//   selection-lock family do not), and the outgoing Pokémon's volatile set is
//   cleared.
import {createScaffold} from './fixture_scaffold.mjs';
const {setOf, foeTeam, foeWith, logHas, runTrials} = createScaffold();

const team = (...heads) => [
  ...heads,
  ...foeTeam().filter(p => !heads.some(head => head.species === p.species)),
].slice(0, 6);

const TRIALS = [
  {
    name: 'batonpass_copies_boosts_substitute_and_focus_energy',
    p1: () => team(setOf('Scizor', 'Swarm', ['Swords Dance', 'Substitute', 'Focus Energy', 'Baton Pass'])),
    p2: () => foeWith(setOf('Slowbro', 'Regenerator', ['Yawn', 'Protect', 'Surf'])),
    script: [
      {p1: [{move: 'swordsdance'}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'substitute'}, 'protect'], p2: [{move: 'yawn', target: 1}, 'protect']},
      {p1: [{move: 'focusenergy'}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'batonpass'}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'batonpass'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Baton Pass\|/)) return 'Baton Pass never executed';
      if (!logHas(session, /\|switch\|p1a: s\d\|/)) return 'the pivot replacement never switched in';
      const pivot = fixture.steps.findIndex(step => step.side === 'P1'
        && step.actions.some(action => action.kind === 'Switch'));
      if (pivot < 0) return 'the pivot replacement was never recorded';
      const before = fixture.steps[pivot - 1].expected.sides[0];
      const after = fixture.steps[pivot].expected.sides[0];
      const outgoing = before.pokemon.find(p => p.active_slot === 0);
      const incoming = after.pokemon.find(p => p.active_slot === 0);
      if (!outgoing || !incoming || incoming.roster === outgoing.roster) {
        return 'the acting slot did not change occupant';
      }
      // Reference `p.boosts` order is [atk, def, spa, spd, spe, accuracy, evasion].
      if (incoming.boosts[0] !== 2) {
        return `the Swords Dance stages did not carry over (${incoming.boosts})`;
      }
      if (!incoming.volatiles.includes('substitute')) return 'the substitute did not carry over';
      if (!incoming.volatiles.includes('focusenergy')) return 'Focus Energy did not carry over';
      if (incoming.volatiles.includes('yawn')) return 'the noCopy Yawn volatile was transferred';
      const cleared = after.pokemon.find(p => p.roster === outgoing.roster);
      if (cleared.boosts.some(stage => stage !== 0)) return 'the outgoing boosts were not cleared';
      if (cleared.volatiles.length) return `the outgoing volatiles were not cleared (${cleared.volatiles})`;
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 27000, artifact: 'more_batonpass.json', debugEnv: 'DEBUG_BATONPASS'});
