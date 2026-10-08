// Development-only corpus for the third passive cluster:
//   Shield Dust - target-side secondaries are dropped before their roll
//   Stall       - the holder's moves resolve last inside their bracket
//   Shed Skin   - a 33% residual roll cures the holder's status
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, logHas, everHas, runTrials} = createScaffold();

const POOL = [
  ['Milotic', 'Competitive', ['Ice Beam', 'Protect']],
  ['Metagross', 'Clear Body', ['Iron Head', 'Protect']],
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']],
  ['Falinks', 'Battle Armor', ['Smart Strike', 'Protect']],
  ['Chimecho', 'Levitate', ['Psychic', 'Protect']],
  ['Aggron', 'Sturdy', ['Iron Head', 'Protect']],
];
const team = (...heads) => [
  ...heads,
  ...POOL.filter(([species]) => !heads.some(head => head.species === species))
    .map(([species, ability, moves]) => offensive(species, ability, moves)),
].slice(0, 6);

const TRIALS = [
  {
    name: 'shielddust_blocks_a_target_secondary',
    p1: () => team(setOf('Vivillon', 'Shield Dust', ['Bug Buzz', 'Protect'])),
    p2: () => team(setOf('Ariados', 'Swarm', ['Acid Spray', 'Protect'])),
    script: [
      // Acid Spray's 100% Special Defense drop is a target-side secondary.
      {p1: [{move: 'bugbuzz', target: 1}, 'protect'], p2: [{move: 'acidspray', target: 1}, 'protect']},
      {p1: [{move: 'bugbuzz', target: 1}, 'protect'], p2: [{move: 'acidspray', target: 1}, 'protect']},
    ],
    coverage: {ability: 'shielddust'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Acid Spray\|p1a: s0/)) return 'Acid Spray never resolved';
      const dropped = everHas(fixture, 0, 0, p => p.boosts[3] < 0);
      if (dropped) return 'Shield Dust let the Special Defense drop through';
      return null;
    },
  },
  {
    name: 'stall_moves_last_within_the_bracket',
    p1: () => team(setOf('Sableye', 'Stall', ['Night Slash', 'Protect'])),
    p2: () => team(setOf('Torkoal', 'Shell Armor', ['Flamethrower', 'Protect'])),
    script: [
      {p1: [{move: 'nightslash', target: 1}, 'protect'], p2: [{move: 'flamethrower', target: 1}, 'protect']},
      {p1: [{move: 'nightslash', target: 1}, 'protect'], p2: [{move: 'flamethrower', target: 1}, 'protect']},
    ],
    coverage: {ability: 'stall'},
    verify(fixture, session) {
      const slower = everHas(fixture, 1, 0, p => p.cached_speed !== null);
      if (!slower) return 'no speed snapshot was recorded';
      const fastest = fixture.steps[0].expected.sides[0].pokemon.find(p => p.roster === 0);
      const foe = fixture.steps[0].expected.sides[1].pokemon.find(p => p.roster === 0);
      if (!(fastest.cached_speed > foe.cached_speed)) return 'the Stall holder was not the faster mon';
      const log = session.battle.log;
      const foeMove = log.findIndex(line => line.startsWith('|move|p2a: s0|Flamethrower|'));
      const stallMove = log.findIndex(line => line.startsWith('|move|p1a: s0|Night Slash|'));
      if (foeMove < 0 || stallMove < 0) return 'a scripted move never resolved';
      if (stallMove < foeMove) return 'Stall did not push the holder behind the slower foe';
      return null;
    },
  },
  {
    name: 'shedskin_cures_its_status',
    // A Poison-type holder could never be poisoned in the first place, so the
    // scene uses the Ground-type Shed Skin holder.
    p1: () => team(setOf('Sandaconda', 'Shed Skin', ['Iron Head', 'Protect'])),
    p2: () => team(setOf('Ariados', 'Swarm', ['Toxic', 'Protect'])),
    script: [
      // No Protect on the Toxic turn; the residual rolls cure at 33%/turn.
      {p1: [{move: 'ironhead', target: 1}, 'protect'], p2: [{move: 'toxic', target: 1}, 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {ability: 'shedskin'},
    verify(fixture, session) {
      const poisoned = everHas(fixture, 0, 0,
        p => p.status === ids.conditions.tox || p.status === ids.conditions.psn);
      if (!poisoned) return 'the holder never got poisoned';
      if (!logHas(session, /\|-activate\|p1a: s0\|ability: Shed Skin/)) {
        return 'Shed Skin never announced a cure';
      }
      const cured = everHas(fixture, 0, 0, p => p.status === 0);
      if (!cured) return 'the status was never cured';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 41000, artifact: 'more_passives3.json', debugEnv: 'DEBUG_PASSIVES3'});
