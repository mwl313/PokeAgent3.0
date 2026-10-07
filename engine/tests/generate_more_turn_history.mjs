// Development-only corpus for the turn-history power/secondary callbacks.
// Assurance doubles against a target already damaged this turn, Temper Flare
// doubles after the user's previous move failed, Lash Out doubles after the
// user's stats were lowered this turn, Barb Barrage doubles against a poisoned
// target, and Alluring Voice confuses a target whose stats were raised this
// turn. Each effect gets a control scene without the trigger. Every fixture is
// a complete legal synthetic reference battle with the pinned Showdown state
// recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {setOf, offensive, logHas, runTrials} = createScaffold();

const fillers = () => [
  offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
  offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
  offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
  offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
  offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
  offensive('Skarmory', 'Sturdy', ['Iron Head', 'Protect']),
];
const p2Team = () => [
  setOf('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
  ...fillers().slice(0, 5),
];

const indexOf = (session, pattern) => session.battle.log.findIndex(line => pattern.test(line));
const turnIndex = (session, turn) => session.battle.log.indexOf(`|turn|${turn}`);
const logHasBefore = (session, pattern, before) =>
  session.battle.log.some((line, index) => index < before && pattern.test(line));
const logBefore = (session, a, b) => {
  const ia = indexOf(session, a);
  const ib = indexOf(session, b);
  return ia >= 0 && ib >= 0 && ia < ib;
};

const assuranceTeam = () => [
  setOf('Jolteon', 'Volt Absorb', ['Thunderbolt', 'Protect']),
  setOf('Corviknight', 'Pressure', ['Assurance', 'Protect']),
  ...fillers().slice(0, 4),
];
const temperTeam = () => [
  setOf('Arcanine', 'Intimidate', ['Temper Flare', 'Protect']),
  ...fillers().slice(0, 5),
];
const lashTeam = () => [
  setOf('Krookodile', 'Intimidate', ['Lash Out', 'Protect']),
  ...fillers().slice(0, 5),
];
const lashFoe = () => [
  offensive('Houndoom', 'Flash Fire', ['Snarl', 'Protect']),
  ...p2Team().slice(1, 6),
];
const barbTeam = () => [
  setOf('Gengar', 'Cursed Body', ['Toxic', 'Protect']),
  setOf('Overqwil', 'Intimidate', ['Barb Barrage', 'Protect']),
  ...fillers().slice(0, 4),
];
const voiceTeam = () => [
  setOf('Espeon', 'Synchronize', ['Alluring Voice', 'Protect']),
  ...fillers().slice(0, 5),
];
const voiceFoe = () => [
  offensive('Cinderace', 'Libero', ['Swords Dance', 'Protect']),
  ...p2Team().slice(1, 6),
];

const TRIALS = [
  {
    name: 'assurance_doubles_on_damaged_target',
    p1: assuranceTeam,
    p2: p2Team,
    script: [
      {p1: [{move: 'thunderbolt', target: 1}, {move: 'assurance', target: 1}], p2: [{move: 'bodyslam', target: 1}, 'protect']},
    ],
    coverage: {move: 'assurance'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Thunderbolt\|p2a: s0/)) return 'the ally never damaged the target';
      if (!logBefore(session, /\|move\|p1a: s0\|Thunderbolt/, /\|move\|p1b: s1\|Assurance/)) {
        return 'Assurance resolved before the target was damaged';
      }
      if (!logHas(session, /\|-damage\|p2a: s0/)) return 'the target never took damage';
      return null;
    },
  },
  {
    name: 'assurance_control_undamaged_target',
    p1: assuranceTeam,
    p2: p2Team,
    script: [
      {p1: ['protect', {move: 'assurance', target: 1}], p2: [{move: 'bodyslam', target: 1}, 'protect']},
    ],
    coverage: {move: 'assurance'},
    verify(fixture, session) {
      const assault = indexOf(session, /\|move\|p1b: s1\|Assurance/);
      if (assault < 0) return 'Assurance never resolved';
      const damageBefore = indexOf(session, /\|-damage\|p2a: s0/);
      if (damageBefore >= 0 && damageBefore < assault) return 'the target was damaged before Assurance';
      return null;
    },
  },
  {
    name: 'temperflare_doubles_after_failed_move',
    p1: temperTeam,
    p2: p2Team,
    script: [
      {p1: [{move: 'temperflare', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'temperflare', target: 1}, 'protect'], p2: [{move: 'bodyslam', target: 1}, 'protect']},
    ],
    coverage: {move: 'temperflare'},
    verify(fixture, session) {
      if (indexOf(session, /\|-activate\|p2a: s0\|move: Protect/) < 0) return 'the first Temper Flare was never blocked';
      const uses = session.battle.log.filter(line => /\|move\|p1a: s0\|Temper Flare/.test(line));
      if (uses.length < 2) return 'Temper Flare was not used twice';
      return null;
    },
  },
  {
    name: 'temperflare_control_after_success',
    p1: temperTeam,
    p2: p2Team,
    script: [
      {p1: [{move: 'temperflare', target: 1}, 'protect'], p2: [{move: 'bodyslam', target: 1}, 'protect']},
      {p1: [{move: 'temperflare', target: 1}, 'protect'], p2: [{move: 'bodyslam', target: 1}, 'protect']},
    ],
    coverage: {move: 'temperflare'},
    verify(fixture, session) {
      const uses = session.battle.log.filter(line => /\|move\|p1a: s0\|Temper Flare/.test(line));
      if (uses.length < 2) return 'Temper Flare was not used twice';
      if (logHas(session, /\|-activate\|p2a: s0\|move: Protect/)) return 'the control battle still blocked the first use';
      return null;
    },
  },
  {
    name: 'lashout_doubles_after_stat_drop',
    p1: lashTeam,
    p2: lashFoe,
    script: [
      {p1: [{move: 'lashout', target: 1}, 'protect'], p2: ['snarl', 'protect']},
    ],
    coverage: {move: 'lashout'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Snarl\|/)) return 'the foe never used Snarl';
      if (!logBefore(session, /\|move\|p2a: s0\|Snarl/, /\|move\|p1a: s0\|Lash Out/)) {
        return 'Lash Out resolved before the stat drop';
      }
      if (!logHasBefore(session, /\|-unboost\|p1a: s0\|spa/, turnIndex(session, 2))) {
        return 'the user never lost Special Attack before Lash Out';
      }
      return null;
    },
  },
  {
    name: 'lashout_control_without_stat_drop',
    p1: lashTeam,
    p2: lashFoe,
    script: [
      {p1: [{move: 'lashout', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'lashout'},
    verify(fixture, session) {
      if (indexOf(session, /\|move\|p1a: s0\|Lash Out/) < 0) return 'Lash Out never resolved';
      if (logHasBefore(session, /\|-unboost\|p1a: s0/, turnIndex(session, 2))) {
        return 'the control battle still dropped a stat before Lash Out';
      }
      return null;
    },
  },
  {
    name: 'barbbarrage_doubles_on_poisoned_target',
    p1: barbTeam,
    p2: p2Team,
    script: [
      {p1: [{move: 'toxic', target: 1}, {move: 'barbbarrage', target: 1}], p2: [{move: 'bodyslam', target: 1}, 'protect']},
    ],
    coverage: {move: 'barbbarrage'},
    verify(fixture, session) {
      if (!logHas(session, /\|-status\|p2a: s0\|tox/)) return 'the target was never poisoned';
      if (!logBefore(session, /\|-status\|p2a: s0\|tox/, /\|move\|p1b: s1\|Barb Barrage/)) {
        return 'Barb Barrage resolved before the poison';
      }
      return null;
    },
  },
  {
    name: 'barbbarrage_control_unpoisoned_target',
    p1: barbTeam,
    p2: p2Team,
    script: [
      {p1: ['protect', {move: 'barbbarrage', target: 1}], p2: [{move: 'bodyslam', target: 1}, 'protect']},
    ],
    coverage: {move: 'barbbarrage'},
    verify(fixture, session) {
      const use = indexOf(session, /\|move\|p1b: s1\|Barb Barrage/);
      if (use < 0) return 'Barb Barrage never resolved';
      const status = indexOf(session, /\|-status\|p2a: s0\|(tox|psn)/);
      if (status >= 0 && status < use) return 'the control target was poisoned before Barb Barrage';
      return null;
    },
  },
  {
    name: 'alluringvoice_confuses_raised_target',
    p1: voiceTeam,
    p2: voiceFoe,
    script: [
      {p1: [{move: 'alluringvoice', target: 1}, 'protect'], p2: ['swordsdance', 'protect']},
    ],
    coverage: {move: 'alluringvoice'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Swords Dance\|/)) return 'the target never raised its stats';
      if (!logBefore(session, /\|move\|p2a: s0\|Swords Dance/, /\|move\|p1a: s0\|Alluring Voice/)) {
        return 'Alluring Voice resolved before the stat raise';
      }
      if (!logHas(session, /\|-start\|p2a: s0\|confusion/)) return 'the raised target was not confused';
      return null;
    },
  },
  {
    name: 'alluringvoice_control_unboosted_target',
    p1: voiceTeam,
    p2: voiceFoe,
    script: [
      {p1: [{move: 'alluringvoice', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'alluringvoice'},
    verify(fixture, session) {
      if (indexOf(session, /\|move\|p1a: s0\|Alluring Voice/) < 0) return 'Alluring Voice never resolved';
      if (logHasBefore(session, /\|-start\|p2a: s0\|confusion/, turnIndex(session, 2))) {
        return 'the unboosted target was still confused';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 7800, artifact: 'more_turn_history.json', debugEnv: 'DEBUG_TURNHISTORY'});
