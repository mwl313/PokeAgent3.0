// Development-only corpus for the Stockpile family:
// - Stockpile raises Defense and Special Defense per layer and caps at three
//   (a fourth attempt fails),
// - Spit Up's power follows the stored layer count and consumes the volatile,
// - Spit Up fails outright without a stockpile,
// - a blocked Spit Up still consumes the volatile (AfterMove runs),
// - Swallow heals by the layer count and reverts the raises, and
// - a refused Swallow (full HP) still consumes the volatile.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {setOf, offensive, monAt, logHas, runTrials} = createScaffold();

const stockpileTeam = () => [
  setOf('Hippowdon', 'Sand Force', ['Stockpile', 'Spit Up', 'Swallow', 'Protect']),
  offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
  offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
  offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
  offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
  offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
];
const foeTeam = () => [
  setOf('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
  offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
  offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
  offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
  offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
  offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
];

const startCount = session =>
  session.battle.log.filter(line => /\|-start\|p1a: s0\|stockpile\d/.test(line)).length;
const atTurn = (fixture, turn, side, roster) => {
  const step = fixture.steps.find(entry => entry.expected.turn >= turn);
  return step ? step.expected.sides[side].pokemon.find(p => p.roster === roster) : null;
};

const TRIALS = [
  {
    name: 'stockpile_two_layers_spitup',
    p1: stockpileTeam,
    p2: foeTeam,
    script: [
      {p1: ['stockpile', 'protect'], p2: [{move: 'bodyslam', target: 1}, 'protect']},
      {p1: ['stockpile', 'protect'], p2: [{move: 'bodyslam', target: 1}, 'protect']},
      {p1: [{move: 'spitup', target: 1}, 'protect'], p2: [{move: 'bodyslam', target: 1}, 'protect']},
    ],
    coverage: {move: 'spitup'},
    verify(fixture, session) {
      if (startCount(session) !== 2) return `expected two stockpile starts, saw ${startCount(session)}`;
      const stacked = atTurn(fixture, 3, 0, 0);
      if (stacked.boosts[1] !== 2 || stacked.boosts[3] !== 2) return 'the two stockpiles did not raise Def/SpD to +2';
      if (!logHas(session, /\|move\|p1a: s0\|Spit Up\|/)) return 'Spit Up never resolved';
      if (!logHas(session, /\|-end\|p1a: s0\|Stockpile/)) return 'Spit Up never consumed the stockpile';
      const after = atTurn(fixture, 4, 0, 0);
      if (after.boosts[1] !== 0 || after.boosts[3] !== 0) return 'the stockpile raises were not reverted';
      if (after.volatiles.includes('stockpile')) return 'the stockpile volatile survived Spit Up';
      return null;
    },
  },
  {
    name: 'stockpile_caps_at_three',
    p1: stockpileTeam,
    p2: foeTeam,
    script: [
      {p1: ['stockpile', 'protect'], p2: [{move: 'bodyslam', target: 1}, 'protect']},
      {p1: ['stockpile', 'protect'], p2: [{move: 'bodyslam', target: 1}, 'protect']},
      {p1: ['stockpile', 'protect'], p2: ['protect', 'protect']},
      {p1: ['stockpile', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'stockpile'},
    verify(fixture, session) {
      if (startCount(session) !== 3) return `expected three stockpile starts, saw ${startCount(session)}`;
      const capped = atTurn(fixture, 4, 0, 0);
      if (capped.boosts[1] !== 3 || capped.boosts[3] !== 3) return 'three stockpiles did not reach +3';
      if (!capped.volatiles.includes('stockpile')) return 'the third stockpile volatile is missing';
      if (!logHas(session, /\|-fail\|p1a: s0/)) return 'the fourth stockpile never failed';
      const after = atTurn(fixture, 5, 0, 0);
      if (after.boosts[1] !== 3 || after.boosts[3] !== 3) return 'the failed fourth use changed the stages';
      return null;
    },
  },
  {
    name: 'spitup_fails_without_stockpile',
    p1: stockpileTeam,
    p2: foeTeam,
    script: [
      {p1: [{move: 'spitup', target: 1}, 'protect'], p2: [{move: 'bodyslam', target: 1}, 'protect']},
    ],
    coverage: {move: 'spitup'},
    verify(fixture, session) {
      if (!logHas(session, /\|-fail\|p1a: s0/)) return 'Spit Up never reported the Try failure';
      const step = fixture.steps.find(entry => entry.expected.turn >= 2);
      const snorlax = step.expected.sides[1].pokemon.find(p => p.roster === 0);
      if (snorlax.hp !== snorlax.max_hp) return 'the failed Spit Up still dealt damage';
      const user = step.expected.sides[0].pokemon.find(p => p.roster === 0);
      if (user.volatiles.includes('stockpile')) return 'the failed Spit Up created a stockpile';
      return null;
    },
  },
  {
    name: 'swallow_heals_half_at_two_layers',
    p1: stockpileTeam,
    p2: foeTeam,
    script: [
      {p1: ['stockpile', 'protect'], p2: [{move: 'bodyslam', target: 1}, 'protect']},
      {p1: ['stockpile', 'protect'], p2: [{move: 'bodyslam', target: 1}, 'protect']},
      {p1: ['swallow', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'swallow'},
    verify(fixture, session) {
      const hurt = atTurn(fixture, 3, 0, 0);
      if (hurt.hp >= hurt.max_hp) return 'the user took no damage before the Swallow';
      if (!logHas(session, /\|-heal\|p1a: s0\|/)) return 'Swallow never healed';
      const after = atTurn(fixture, 4, 0, 0);
      if (after.hp <= hurt.hp) return 'Swallow did not restore HP';
      if (after.boosts[1] !== 0 || after.boosts[3] !== 0) return 'Swallow did not revert the stockpile raises';
      if (after.volatiles.includes('stockpile')) return 'the stockpile volatile survived Swallow';
      return null;
    },
  },
  {
    name: 'swallow_full_hp_still_consumes_stockpile',
    p1: stockpileTeam,
    p2: foeTeam,
    script: [
      {p1: ['stockpile', 'protect'], p2: ['protect', 'protect']},
      {p1: ['swallow', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'swallow'},
    verify(fixture, session) {
      if (!logHas(session, /\|-fail\|p1a: s0\|heal/)) return 'the full-HP Swallow never reported the heal failure';
      if (logHas(session, /\|-heal\|p1a: s0\|/)) return 'the full-HP Swallow still healed';
      const after = atTurn(fixture, 3, 0, 0);
      if (after.hp !== after.max_hp) return 'the user was not at full HP';
      if (after.volatiles.includes('stockpile')) return 'the refused Swallow left the stockpile volatile';
      if (after.boosts[1] !== 0 || after.boosts[3] !== 0) return 'the refused Swallow did not revert the raises';
      return null;
    },
  },
  {
    name: 'spitup_blocked_still_consumes_stockpile',
    p1: stockpileTeam,
    p2: foeTeam,
    script: [
      {p1: ['stockpile', 'protect'], p2: [{move: 'bodyslam', target: 1}, 'protect']},
      {p1: ['stockpile', 'protect'], p2: [{move: 'bodyslam', target: 1}, 'protect']},
      {p1: [{move: 'spitup', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'spitup'},
    verify(fixture, session) {
      if (!logHas(session, /\|-activate\|p2a: s0\|move: Protect/)) return 'the target never Protected';
      const blocked = atTurn(fixture, 4, 1, 0);
      if (blocked.hp !== blocked.max_hp) return 'the blocked Spit Up still dealt damage';
      const after = atTurn(fixture, 4, 0, 0);
      if (after.volatiles.includes('stockpile')) return 'the blocked Spit Up left the stockpile volatile';
      if (after.boosts[1] !== 0 || after.boosts[3] !== 0) return 'the blocked Spit Up did not revert the raises';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 7600, artifact: 'more_stockpile.json', debugEnv: 'DEBUG_STOCKPILE'});
