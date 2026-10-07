// Development-only corpus for Water Shuriken: a priority 2-5 hit Water move
// whose base-power callback degenerates to the declared power in the pinned
// regulation (the Greninja-Ash / Battle Bond branch is not a legal state).
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, logHas, runTrials} = createScaffold();

const POOL = [
  ['Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic']],
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
    name: 'watershuriken_priority_multihit',
    p1: () => team(setOf('Greninja', 'Torrent', ['Water Shuriken', 'Protect', 'Dark Pulse'])),
    p2: () => team(setOf('Torkoal', 'Shell Armor', ['Eruption', 'Protect', 'Body Press'])),
    script: [
      {p1: [{move: 'watershuriken', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'watershuriken', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'watershuriken'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Water Shuriken\|p2a: s0/)) return 'Water Shuriken never hit';
      const counts = session.battle.log
        .filter(line => line.startsWith('|-hitcount|p2a: s0|'))
        .map(line => Number(line.split('|')[3]));
      if (!counts.length) return 'no hit count was reported';
      if (counts.some(n => n < 2 || n > 5)) return `hit count out of range: ${counts.join(',')}`;
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 23000, artifact: 'more_watershuriken.json', debugEnv: 'DEBUG_WATERSHURIKEN'});
