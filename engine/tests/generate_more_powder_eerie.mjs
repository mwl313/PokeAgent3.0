// Development-only corpus for two small tail moves:
// - Magic Powder overwrites the target's types with pure Psychic and refuses a
//   target that already is pure Psychic,
// - Eerie Spell's 100% secondary drains three PP from the target's last move.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, logHas, monAt, runTrials} = createScaffold();

const POOL = [
  ['Milotic', 'Competitive', ['Surf', 'Protect', 'Ice Beam']],
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect', 'Body Slam']],
  ['Falinks', 'Battle Armor', ['Smart Strike', 'Protect', 'Close Combat']],
  ['Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect', 'Psychic']],
  ['Aggron', 'Sturdy', ['Iron Head', 'Protect', 'Body Slam']],
  ['Ariados', 'Swarm', ['Leech Life', 'Protect', 'Sucker Punch']],
];
const team = (...heads) => [
  ...heads,
  ...POOL.filter(([species]) => !heads.some(head => head.species === species))
    .map(([species, ability, moves]) => offensive(species, ability, moves)),
].slice(0, 6);

const TRIALS = [
  {
    name: 'tailmagicpowder_overwrites_types_then_fails',
    p1: () => team(setOf('Hatterene', 'Magic Bounce', ['Magic Powder', 'Protect', 'Psychic'])),
    p2: () => team(setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic'])),
    script: [
      {p1: [{move: 'magicpowder', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'magicpowder', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'magicpowder'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Magic Powder\|/)) return 'Magic Powder never executed';
      if (!logHas(session, /\|-start\|p2a: s0\|typechange\|Psychic/)) return 'the type change never appeared';
      if (!monAt(fixture, 1, 0).some(p => p.types.length === 1 && p.types[0] === ids.types.psychic)) {
        return 'the target never became pure Psychic';
      }
      const fails = session.battle.log.filter(line => line.startsWith('|-fail|p1a: s0')).length;
      if (!fails) return 'the second Magic Powder did not fail on a pure-Psychic target';
      return null;
    },
  },
  {
    name: 'taileeriespell_drains_last_move_pp',
    p1: () => team(setOf('Slowking-Galar', 'Regenerator', ['Eerie Spell', 'Protect', 'Sludge Bomb'])),
    p2: () => team(setOf('Milotic', 'Competitive', ['Surf', 'Protect', 'Ice Beam'])),
    script: [
      {p1: ['protect', 'protect'], p2: ['surf', 'protect']},
      {p1: [{move: 'eeriespell', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'eeriespell'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Eerie Spell\|/)) return 'Eerie Spell never executed';
      if (!logHas(session, /move: Eerie Spell/)) return 'the PP drain never activated';
      const steps = fixture.steps.map(step => step.expected.sides[1].pokemon.find(p => p.roster === 0));
      const ppOf = p => p && p.pp[0];
      const before = ppOf(steps.find(p => p && p.pp[0] < 24));
      const after = ppOf(steps.at(-1));
      if (before == null || after == null) return 'the Surf PP was not recorded';
      if (before - after < 3) return `the drain was less than three PP (${before} -> ${after})`;
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 24000, artifact: 'more_powder_eerie.json', debugEnv: 'DEBUG_POWDER'});
