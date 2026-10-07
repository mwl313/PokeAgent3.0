// Development-only corpus for Curse, whose payload depends on the user's
// Ghost type:
// - a Ghost user (`moves:curse.onTryHit`) refuses a target that already
//   carries the curse volatile, applies it otherwise and pays half of its own
//   maximum HP through `onHit`; `condition.onResidual` (order 12) drains a
//   quarter of the cursed holder's maximum HP each turn;
// - any other user (`onModifyMove` redirects the target to itself and
//   `onTryHit` replaces the payload) gains Attack +1, Defense +1 and Speed -1
//   through the self-drop roll instead.
import {createScaffold} from './fixture_scaffold.mjs';
const {setOf, foeTeam, foeWith, logHas, runTrials} = createScaffold();

const team = (...heads) => [
  ...heads,
  ...foeTeam().filter(p => !heads.some(head => head.species === p.species)),
].slice(0, 6);
const mon = (fixture, index, side, roster) => fixture.steps[index].expected.sides[side].pokemon
  .find(p => p.roster === roster);

const TRIALS = [
  {
    name: 'curse_ghost_drains_the_target',
    p1: () => team(setOf('Gengar', 'Cursed Body', ['Curse', 'Protect', 'Shadow Ball'])),
    p2: () => foeWith(setOf('Milotic', 'Competitive', ['Surf', 'Protect', 'Ice Beam'])),
    script: [
      {p1: [{move: 'curse'}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'curse'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Curse\|/)) return 'Curse never executed';
      if (!logHas(session, /\|-start\|p2a: s0\|Curse\|\[of\]/)) return 'the target was never cursed';
      const curseTurn = fixture.steps.findIndex(step => step.actions.some(action =>
        action.kind === 'Move' && action.own_slot === 0 && action.target_location === 1));
      if (curseTurn < 0) return 'the curse turn was never recorded';
      const before = mon(fixture, curseTurn - 1, 0, 0);
      const after = mon(fixture, curseTurn + 1, 0, 0);
      const paid = before.hp - after.hp;
      if (Math.abs(paid - Math.floor(before.max_hp / 2)) > 1) {
        return `the user did not pay half its maximum HP (${before.hp} -> ${after.hp})`;
      }
      const cursed = mon(fixture, curseTurn + 1, 1, 0);
      if (!cursed.volatiles.includes('curse')) return 'the target does not carry the curse volatile';
      // The drain ticks once per residual: compare two later boundaries.
      const later = Math.min(curseTurn + 3, fixture.steps.length - 1);
      const drained = cursed.hp - mon(fixture, later, 1, 0).hp;
      if (drained < Math.floor(cursed.max_hp / 4) - 1) {
        return `the curse never drained a quarter of the maximum HP (${drained})`;
      }
      return null;
    },
  },
  {
    name: 'curse_non_ghost_boosts_the_user',
    p1: () => team(setOf('Aggron', 'Sturdy', ['Curse', 'Protect', 'Iron Head'])),
    p2: () => foeWith(setOf('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect'])),
    script: [
      {p1: [{move: 'curse'}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'curse'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Curse\|/)) return 'Curse never executed';
      const turn = fixture.steps.findIndex(step => step.actions.some(action =>
        action.kind === 'Move' && action.own_slot === 0));
      if (turn < 0) return 'the curse turn was never recorded';
      const boosted = mon(fixture, turn + 1, 0, 0);
      // Reference `p.boosts` order is [atk, def, spa, spd, spe, accuracy, evasion].
      if (boosted.boosts[0] !== 1 || boosted.boosts[1] !== 1 || boosted.boosts[4] !== -1) {
        return `the self boost is wrong (${boosted.boosts})`;
      }
      if (!logHas(session, /\|-boost\|p1a: s0\|atk\|1/) || !logHas(session, /\|-boost\|p1a: s0\|def\|1/)) {
        return 'the self boost never announced';
      }
      const cursed = fixture.steps.some(step => step.expected.sides.flatMap(side => side.pokemon)
        .some(p => p.volatiles.includes('curse')));
      if (cursed) return 'a non-Ghost Curse applied the curse volatile';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 35000, artifact: 'more_curse.json', debugEnv: 'DEBUG_CURSE'});
