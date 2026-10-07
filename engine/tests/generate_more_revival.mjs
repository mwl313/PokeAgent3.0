// Development-only corpus for Revival Blessing (Pawmot's revive protocol):
// - the move fails outright with no fainted party member,
// - a fainted *reserve* returns at half HP with its status cleared (no
//   instaswitch, the user stays on the field),
// - a party member that faints during the same turn is revived into its still
//   occupied active slot, which queues the reference's `instaswitch` re-entry.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, logHas, monAt, runTrials} = createScaffold();

const POOL = [
  ['Milotic', 'Competitive', ['Surf', 'Protect', 'Ice Beam']],
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect', 'Body Slam']],
  ['Falinks', 'Battle Armor', ['Smart Strike', 'Protect', 'Close Combat']],
  ['Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect', 'Psychic']],
  ['Aggron', 'Sturdy', ['Iron Head', 'Protect', 'Body Slam']],
  ['Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic']],
];
const team = (...heads) => [
  ...heads,
  ...POOL.filter(([species]) => !heads.some(head => head.species === species))
    .map(([species, ability, moves]) => offensive(species, ability, moves)),
].slice(0, 6);

const pawmot = () => setOf('Pawmot', 'Iron Fist',
  ['Revival Blessing', 'Close Combat', 'Protect'], '', {hp: 32, atk: 0, def: 17, spa: 0, spd: 17, spe: 0});
const ariados = () => setOf('Ariados', 'Swarm', ['Leech Life', 'Protect', 'Sucker Punch']);

const TRIALS = [
  {
    name: 'revivalblessing_fails_without_fainted',
    p1: () => team(pawmot(), ariados()),
    p2: () => team(setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic'])),
    script: [
      {p1: [{move: 'revivalblessing', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'revivalblessing'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Revival Blessing\|/)) return 'Revival Blessing was never used';
      if (!logHas(session, /\|-fail\|p1a: s0/)) return 'the move did not fail without a fainted ally';
      if (logHas(session, /Revival Blessing/)) {
        const heals = session.battle.log.filter(line => line.includes('[from] move: Revival Blessing'));
        if (heals.length) return 'a revival happened without a fainted ally';
      }
      return null;
    },
  },
  {
    name: 'revivalblessing_revives_fainted_reserve',
    p1: () => team(pawmot(), ariados()),
    p2: () => team(
      offensive('Arcanine', 'Intimidate', ['Will-O-Wisp', 'Protect', 'Extreme Speed']),
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic']),
    ),
    script: [
      {p1: [{move: 'protect', target: 0}, 'leechlife'],
        p2: [{move: 'willowisp', target: 2}, {move: 'ironhead', target: 2}]},
      {p1: [{move: 'protect', target: 0}, 'leechlife'],
        p2: [{move: 'extremespeed', target: 2}, {move: 'ironhead', target: 2}]},
      {p1: [{move: 'revivalblessing', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'revivalblessing'},
    verify(fixture, session) {
      if (!monAt(fixture, 0, 1).some(p => p.fainted)) return 'the ally never fainted';
      if (!logHas(session, /\|-heal\|p1: s1\|[^|]*\|\[from\] move: Revival Blessing/)) {
        return 'the revive heal never appeared';
      }
      // The heal resolves during turn 3, so the turn-3 P2 boundary is the
      // first recorded state that contains the revived member.
      // The revival is the only event that puts the ally exactly at half HP.
      const revived = fixture.steps
        .map(step => step.expected.sides[0].pokemon.find(p => p.roster === 1))
        .find(p => p && !p.fainted && p.hp === Math.max(1, Math.trunc(p.max_hp / 2)));
      if (!revived) return 'the ally was never revived';
      if (revived.status !== 0) return 'the revived ally kept a status';
      if (revived.active_slot !== null) return 'the reserve revive switched the ally in';
      return null;
    },
  },
  {
    name: 'revivalblessing_instaswitch_same_turn_faint',
    p1: () => team(pawmot(), ariados()),
    p2: () => team(
      offensive('Dragonite', 'Inner Focus', ['Extreme Speed', 'Dragon Claw', 'Protect']),
      offensive('Metagross', 'Clear Body', ['Bullet Punch', 'Protect', 'Iron Head']),
    ),
    script: [
      {p1: [{move: 'protect', target: 0}, 'leechlife'],
        p2: [{move: 'dragonclaw', target: 2}, {move: 'bulletpunch', target: 2}]},
      {p1: [{move: 'revivalblessing', target: 0}, 'leechlife'],
        p2: [{move: 'extremespeed', target: 2}, {move: 'bulletpunch', target: 2}]},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'revivalblessing'},
    verify(fixture, session) {
      if (!logHas(session, /\|-heal\|p1: s1\|[^|]*\|\[from\] move: Revival Blessing/)) {
        return 'the revive heal never appeared';
      }
      if (!logHas(session, /\|switch\|p1b: s1\|/)) return 'the revived ally never re-entered the field';
      const revived = fixture.steps
        .map(step => step.expected.sides[0].pokemon.find(p => p.roster === 1))
        .find(p => p && !p.fainted && p.hp === Math.max(1, Math.trunc(p.max_hp / 2)));
      if (!revived) return 'the ally was not revived';
      if (revived.active_slot === null) return 'the ally did not return to its slot';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 18000, artifact: 'more_revival.json', debugEnv: 'DEBUG_REVIVAL'});
