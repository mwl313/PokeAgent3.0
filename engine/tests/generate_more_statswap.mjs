// Development-only corpus for the stat-swap moves:
// - Power Swap exchanges the Attack / Sp. Atk boost stages with the target,
// - Guard Swap exchanges the Defense / Sp. Def stages,
// - Speed Swap exchanges the two stored Speed stats (which also reorders every
//   later action).
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, foeTeam, logHas, runTrials} = createScaffold();

const team = (...heads) => [
  ...heads,
  ...foeTeam().filter(p => !heads.some(head => head.species === p.species)),
].slice(0, 6);
const boostsAt = (fixture, side, roster) => fixture.steps.flatMap(step => step.expected.sides[side].pokemon
  .filter(p => p.roster === roster).map(p => p.boosts));
const statsAt = (fixture, side, roster) => fixture.steps.flatMap(step => step.expected.sides[side].pokemon
  .filter(p => p.roster === roster).map(p => p.stats));

const TRIALS = [
  {
    name: 'statswap_power_swap_moves_the_attack_and_spa_stages',
    p1: () => team(setOf('Alakazam', 'Magic Guard', ['Nasty Plot', 'Power Swap', 'Protect'])),
    p2: () => foeTeam(),
    script: [
      {p1: [{move: 'nastyplot'}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'powerswap', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'powerswap'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Power Swap\|/)) return 'Power Swap never executed';
      const mine = boostsAt(fixture, 0, 0).map(b => b[2]);
      const theirs = boostsAt(fixture, 1, 0).map(b => b[2]);
      if (!mine.some((spa, i) => i > 0 && spa === 0 && mine[i - 1] > 0)) {
        return 'the holder never gave its Sp. Atk stages away';
      }
      if (!theirs.some((spa, i) => i > 0 && spa > 0 && theirs[i - 1] === 0)) {
        return 'the target never received the Sp. Atk stages';
      }
      return null;
    },
  },
  {
    name: 'statswap_guard_swap_moves_the_defense_and_spd_stages',
    p1: () => team(setOf('Alakazam', 'Magic Guard', ['Calm Mind', 'Guard Swap', 'Protect'])),
    p2: () => foeTeam(),
    script: [
      {p1: [{move: 'calmmind'}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'guardswap', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'guardswap'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Guard Swap\|/)) return 'Guard Swap never executed';
      const mine = boostsAt(fixture, 0, 0).map(b => b[3]);
      const theirs = boostsAt(fixture, 1, 0).map(b => b[3]);
      if (!mine.some((spd, i) => i > 0 && spd === 0 && mine[i - 1] > 0)) {
        return 'the holder never gave its Sp. Def stages away';
      }
      if (!theirs.some((spd, i) => i > 0 && spd > 0 && theirs[i - 1] === 0)) {
        return 'the target never received the Sp. Def stages';
      }
      return null;
    },
  },
  {
    name: 'statswap_speed_swap_exchanges_the_stored_speeds',
    p1: () => team(setOf('Alakazam', 'Magic Guard', ['Speed Swap', 'Protect'])),
    p2: () => foeTeam(),
    script: [{p1: [{move: 'speedswap', target: 1}, 'protect'], p2: ['protect', 'protect']}],
    coverage: {move: 'speedswap'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Speed Swap\|/)) return 'Speed Swap never executed';
      const mine = statsAt(fixture, 0, 0);
      const theirs = statsAt(fixture, 1, 0);
      const swapped = mine.some((stats, i) => i > 0
        && stats[6] === theirs[0][6] && theirs[i][6] === mine[0][6]);
      if (!swapped) return 'the stored Speed stats never exchanged';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 8600, artifact: 'more_statswap.json', debugEnv: 'DEBUG_STATSWAP'});
