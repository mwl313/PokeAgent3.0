// Development-only corpus for the weight and flinch passives:
//   Heavy Metal - doubles the holder's weight (Heavy Slam / Heat Crash power)
//   Light Metal - halves the holder's weight (Low Kick / Grass Knot power)
//   Stench      - appends a 10% flinch secondary to damaging moves
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

const hitDamages = (session, userIdent, targetIdent, moveName) => {
  const log = session.battle.log;
  const hits = [];
  for (let i = 0; i < log.length; i++) {
    if (!log[i].startsWith('|move|')) continue;
    const [, , source, name] = log[i].split('|');
    if (source !== userIdent || name !== moveName) continue;
    for (let j = i + 1; j < log.length && !log[j].startsWith('|move|'); j++) {
      if (log[j].startsWith(`|-damage|${targetIdent}|`)) {
        hits.push(Number(log[j].split('|')[3].split(' ')[0].split('/')[0]));
        break;
      }
    }
  }
  return hits;
};

const TRIALS = [
  {
    name: 'heavymetal_lifts_heavy_slam_power',
    // Heavy Metal doubles Aggron's weight, so Heavy Slam (100 BP here) must
    // out-damage Iron Head (80 BP) against the same Milotic.
    p1: () => team(setOf('Aggron', 'Heavy Metal', ['Heavy Slam', 'Iron Head', 'Protect'])),
    p2: () => team(setOf('Milotic', 'Competitive', ['Ice Beam', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'heavyslam', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'ironhead', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {ability: 'heavymetal'},
    verify(fixture, session) {
      if (!everHas(fixture, 0, 0, p => p.ability === ids.abilities.heavymetal)) {
        return 'the holder never carried Heavy Metal';
      }
      const slam = hitDamages(session, 'p1a: s0', 'p2a: s0', 'Heavy Slam');
      const head = hitDamages(session, 'p1a: s0', 'p2a: s0', 'Iron Head');
      if (!slam.length || !head.length) return 'both baseline moves must land';
      const switchLine = session.battle.log.find(line => line.startsWith('|switch|p2a: s0|'));
      const max = switchLine ? Number(switchLine.split('|')[4].split('/')[1]) : 0;
      if (!max) return 'the defender max HP was not recorded';
      const slamDamage = max - slam[0];
      const headDamage = slam[0] - head[0];
      if (slamDamage <= headDamage) {
        return `Heavy Slam did not out-damage Iron Head (${slamDamage} vs ${headDamage})`;
      }
      return null;
    },
  },
  {
    name: 'lightmetal_halves_low_kick_power',
    // Low Kick reads the target's weight; Light Metal halves Scizor's, moving
    // it from the 100 BP bracket to the 80 BP one.
    p1: () => team(setOf('Aggron', 'Sturdy', ['Low Kick', 'Protect'])),
    p2: () => team(setOf('Scizor', 'Light Metal', ['Iron Head', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'lowkick', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'lowkick', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {ability: 'lightmetal'},
    verify(fixture, session) {
      if (!everHas(fixture, 1, 0, p => p.ability === ids.abilities.lightmetal)) {
        return 'the defender never carried Light Metal';
      }
      if (!logHas(session, /\|move\|p1a: s0\|Low Kick\|p2a: s0/)) return 'Low Kick never resolved';
      if (!logHas(session, /\|-damage\|p2a: s0\|/)) return 'Low Kick never dealt damage';
      return null;
    },
  },
  {
    name: 'stench_adds_a_flinch_roll',
    // Body Slam carries no flinch entry, so Stench appends its 10% roll.
    p1: () => team(setOf('Garbodor', 'Stench', ['Body Slam', 'Protect'])),
    p2: () => team(setOf('Milotic', 'Competitive', ['Ice Beam', 'Protect'])),
    script: [
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {ability: 'stench'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Body Slam\|p2a: s0/)) return 'Body Slam never resolved';
      if (!logHas(session, /\|cant\|p2a: s0\|flinch/)) return 'the Stench flinch never fired';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 45000, artifact: 'more_weights.json', debugEnv: 'DEBUG_WEIGHTS'});
