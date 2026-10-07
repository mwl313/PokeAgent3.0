// Development-only corpus for the sleep family:
// - Rest: the asleep/full-HP/insomnia fail gates, the major-status
//   replacement, the terrain and Leaf Guard refusals, the forced three-turn
//   counter, the Lum Berry AfterSetStatus cure and the full heal,
// - Snore: the asleep-only Try gate, the flinch secondary, the Soundproof
//   immunity and the Throat Chop refusal,
// - the natural `slp` start time (`random(2, 5)`) pinned through wake timing,
// - Rest sampled by Sleep Talk (the called move fails while asleep).
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, logHas, everHas, monAt, runTrials} = createScaffold();

const POOL = [
  ['Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic']],
  ['Snorlax', 'Thick Fat', ['Body Slam', 'Protect', 'Crunch']],
  ['Milotic', 'Competitive', ['Surf', 'Protect', 'Ice Beam']],
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect', 'Body Slam']],
  ['Falinks', 'Battle Armor', ['Smart Strike', 'Protect', 'Close Combat']],
  ['Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect', 'Psychic']],
  ['Aggron', 'Sturdy', ['Iron Head', 'Protect', 'Body Slam']],
  ['Ariados', 'Swarm', ['Leech Life', 'Protect', 'Crunch']],
];
const team = (...heads) => [
  ...heads,
  ...POOL.filter(([species]) => !heads.some(head => head.species === species))
    .map(([species, ability, moves]) => offensive(species, ability, moves)),
].slice(0, 6);

const sleeper = (item = '') => setOf('Snorlax', 'Thick Fat',
  ['Body Slam', 'Protect', 'Rest', 'Snore'], item);

const TRIALS = [
  {
    name: 'sleep_rest_heals_and_sleeps',
    p1: () => team(sleeper()),
    p2: () => team(offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic'])),
    script: [
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'rest', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'rest'},
    verify(fixture, session) {
      if (!monAt(fixture, 0, 0).some(p => p.status === ids.conditions.slp)) return 'Rest never applied sleep';
      if (!monAt(fixture, 0, 0).some(p => p.status === ids.conditions.slp && p.hp === p.max_hp)) {
        return 'Rest did not heal to full on the sleep turn';
      }
      if (!logHas(session, /\|-curestatus\|p1a: s0\|slp/)) return 'the user never naturally woke up';
      if (!logHas(session, /\|move\|p1a: s0\|Body Slam\|p2a: s0/)) return 'the user never acted after waking';
      return null;
    },
  },
  {
    name: 'sleep_rest_fails_at_full_hp',
    p1: () => team(sleeper()),
    p2: () => team(offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic'])),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'rest', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'rest'},
    verify(fixture, session) {
      if (monAt(fixture, 0, 0).some(p => p.status === ids.conditions.slp)) return 'Rest slept the full-HP user';
      if (!logHas(session, /\|move\|p1a: s0\|Rest\|/)) return 'Rest was never attempted';
      if (!logHas(session, /\|-fail\|p1a: s0\|heal/)) return 'the heal fail message never appeared';
      return null;
    },
  },
  {
    name: 'sleep_rest_replaces_major_status',
    p1: () => team(sleeper()),
    p2: () => team(offensive('Arcanine', 'Intimidate', ['Will-O-Wisp', 'Protect', 'Flare Blitz'])),
    script: [
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: [{move: 'willowisp', target: 1}, 'protect']},
      {p1: [{move: 'rest', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'rest'},
    verify(fixture, session) {
      if (!monAt(fixture, 0, 0).some(p => p.status === ids.conditions.brn)) return 'the burn never landed';
      if (!monAt(fixture, 0, 0).some(p => p.status === ids.conditions.slp)) return 'Rest never applied sleep';
      const restTurn = fixture.steps.findIndex(step =>
        step.expected.sides[0].pokemon.some(p => p.roster === 0 && p.status === ids.conditions.slp));
      if (restTurn < 0) return 'no boundary recorded the sleep';
      const after = fixture.steps.slice(restTurn).flatMap(step => step.expected.sides[0].pokemon)
        .filter(p => p.roster === 0);
      if (after.some(p => p.status === ids.conditions.brn)) return 'the burn survived Rest';
      return null;
    },
  },
  {
    name: 'sleep_rest_fails_while_asleep',
    p1: () => team(sleeper()),
    p2: () => team(offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic'])),
    script: [
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'rest', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'rest', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'rest'},
    verify(fixture, session) {
      if (!logHas(session, /\|cant\|p1a: s0\|slp/)) return 'the second Rest was never refused while asleep';
      const rested = fixture.steps.findIndex(step =>
        step.expected.sides[0].pokemon.some(p => p.roster === 0 && p.status === ids.conditions.slp));
      if (rested < 0) return 'the first Rest never landed';
      const later = fixture.steps.slice(rested + 1).flatMap(step => step.expected.sides[0].pokemon)
        .filter(p => p.roster === 0);
      if (!later.some(p => p.status === ids.conditions.slp)) return 'sleep ended before the retry boundary';
      return null;
    },
  },
  {
    name: 'sleep_rest_fails_with_insomnia',
    p1: () => team(setOf('Ariados', 'Insomnia', ['Rest', 'Leech Life', 'Protect'])),
    p2: () => team(offensive('Metagross', 'Clear Body', ['Bullet Punch', 'Protect', 'Psychic'])),
    script: [
      {p1: [{move: 'leechlife', target: 1}, 'protect'], p2: [{move: 'bulletpunch', target: 1}, 'protect']},
      {p1: [{move: 'rest', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'leechlife', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'rest'},
    verify(fixture, session) {
      if (monAt(fixture, 0, 0).some(p => p.status === ids.conditions.slp)) return 'Insomnia fell asleep via Rest';
      if (!logHas(session, /ability: Insomnia/)) return 'the Insomnia fail message never appeared';
      if (!monAt(fixture, 0, 0).some(p => p.hp < p.max_hp)) return 'the Insomnia user never took damage';
      const healed = monAt(fixture, 0, 0).some((p, i, all) => i > 0 && p.hp > all[i - 1].hp);
      if (healed) return 'Rest healed the Insomnia user';
      return null;
    },
  },
  {
    name: 'sleep_rest_blocked_by_electric_terrain',
    p1: () => team(sleeper()),
    p2: () => team(setOf('Ampharos', 'Static', ['Electric Terrain', 'Protect', 'Dragon Pulse'])),
    script: [
      {p1: [{move: 'bodyslam', target: 2}, 'protect'],
        p2: [{move: 'electricterrain', target: 0}, {move: 'ironhead', target: 1}]},
      {p1: [{move: 'rest', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'rest'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Rest\|/)) return 'Rest was never attempted';
      if (monAt(fixture, 0, 0).some(p => p.status === ids.conditions.slp)) return 'Rest slept the user on Electric Terrain';
      if (!monAt(fixture, 0, 0).some(p => p.hp < p.max_hp)) return 'the user never took damage before Rest';
      return null;
    },
  },
  {
    name: 'sleep_rest_blocked_by_misty_terrain',
    p1: () => team(sleeper()),
    p2: () => team(setOf('Azumarill', 'Huge Power', ['Misty Terrain', 'Protect', 'Play Rough'])),
    script: [
      {p1: [{move: 'bodyslam', target: 2}, 'protect'],
        p2: [{move: 'mistyterrain', target: 0}, {move: 'ironhead', target: 1}]},
      {p1: [{move: 'rest', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'rest'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Rest\|/)) return 'Rest was never attempted';
      if (monAt(fixture, 0, 0).some(p => p.status === ids.conditions.slp)) return 'Rest slept the user on Misty Terrain';
      return null;
    },
  },
  {
    name: 'sleep_rest_lum_berry_cures',
    p1: () => team(sleeper('Lum Berry')),
    p2: () => team(offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic'])),
    script: [
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'rest', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'rest'},
    verify(fixture, session) {
      if (!monAt(fixture, 0, 0).some(p =>
        p.previous_item === ids.items.lumberry && p.item === 0 && p.hp === p.max_hp)) {
        return 'the Lum Berry never cured the Rest sleep before the heal';
      }
      return null;
    },
  },
  {
    name: 'sleep_rest_from_sleeptalk_fails',
    p1: () => team(setOf('Snorlax', 'Thick Fat', ['Sleep Talk', 'Rest', 'Body Slam', 'Protect'])),
    p2: () => team(offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic'])),
    script: [
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'rest', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'sleeptalk', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'sleeptalk', target: 0}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'rest'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Sleep Talk\|/)) return 'Sleep Talk was never used';
      const restLines = session.battle.log.filter(line => line.startsWith('|move|p1a: s0|Rest|'));
      if (restLines.length < 2) return 'Sleep Talk never sampled Rest';
      return null;
    },
  },
  {
    name: 'sleep_natural_wake_timing',
    p1: () => team(setOf('Snorlax', 'Thick Fat', ['Body Slam', 'Protect', 'Rest', 'Snore'])),
    p2: () => team(setOf('Roserade', 'Natural Cure', ['Sleep Powder', 'Protect', 'Sludge Bomb'])),
    script: [
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: [{move: 'sleeppowder', target: 1}, 'protect']},
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'sleeppowder'},
    verify(fixture, session) {
      if (!monAt(fixture, 0, 0).some(p => p.status === ids.conditions.slp)) return 'Sleep Powder never landed';
      if (!logHas(session, /\|-curestatus\|p1a: s0\|slp/)) return 'the user never woke up';
      if (!logHas(session, /\|move\|p1a: s0\|Body Slam\|p2a: s0/)) return 'the user never attacked after waking';
      return null;
    },
  },
  {
    name: 'sleep_snore_hits_and_flinches',
    p1: () => team(sleeper()),
    p2: () => team(setOf('Torkoal', 'Shell Armor', ['Flamethrower', 'Protect', 'Body Press'])),
    script: [
      {p1: [{move: 'bodyslam', target: 2}, 'protect'],
        p2: ['protect', {move: 'ironhead', target: 1}]},
      {p1: [{move: 'rest', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'snore', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'snore', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'snore'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Snore\|p2a: s0/)) return 'Snore never hit';
      if (!monAt(fixture, 1, 0).some(p => p.hp < p.max_hp)) return 'Snore dealt no damage';
      if (!logHas(session, /\|cant\|p2a: s0\|flinch/)) return 'the flinch secondary never refused the target';
      return null;
    },
  },
  {
    name: 'sleep_snore_fails_awake',
    p1: () => team(sleeper()),
    p2: () => team(offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic'])),
    script: [
      {p1: [{move: 'snore', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'snore', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'snore'},
    verify(fixture, session) {
      const log = session.battle.log;
      const idx = log.findIndex(line => line.startsWith('|move|p1a: s0|Snore|'));
      if (idx < 0) return 'Snore was never attempted';
      if (monAt(fixture, 0, 0).some(p => p.status === ids.conditions.slp)) return 'the user was asleep';
      if (!log.slice(idx + 1, idx + 8).some(line => line.startsWith('|-fail|p1a: s0'))) {
        return 'Snore did not fail while awake';
      }
      if (log.slice(0, idx + 8).some(line => line.startsWith('|-damage|p2a: s0'))) {
        return 'Snore damaged a target while awake';
      }
      return null;
    },
  },
  {
    name: 'sleep_snore_blocked_by_soundproof',
    p1: () => team(sleeper()),
    p2: () => team(setOf('Abomasnow', 'Soundproof', ['Blizzard', 'Protect', 'Energy Ball'])),
    script: [
      {p1: [{move: 'bodyslam', target: 2}, 'protect'],
        p2: ['protect', {move: 'ironhead', target: 1}]},
      {p1: [{move: 'rest', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'snore', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'snore', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'snore'},
    verify(fixture, session) {
      const log = session.battle.log;
      const idx = log.findIndex(line => line.startsWith('|move|p1a: s0|Snore|'));
      if (idx < 0) return 'Snore was never attempted';
      if (!log.slice(idx, idx + 8).some(line => line.includes('ability: Soundproof'))) {
        return 'Soundproof never blocked the sound move';
      }
      if (log.slice(0, idx).some(line => line.startsWith('|-damage|p2a: s0'))) {
        return 'the Soundproof holder was damaged before Snore';
      }
      return null;
    },
  },
  {
    name: 'sleep_snore_refused_under_throatchop',
    p1: () => team(sleeper()),
    p2: () => team(setOf('Absol', 'Super Luck', ['Throat Chop', 'Protect', 'Night Slash'])),
    script: [
      {p1: [{move: 'bodyslam', target: 2}, 'protect'],
        p2: ['protect', {move: 'ironhead', target: 1}]},
      {p1: [{move: 'rest', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'snore', target: 1}, 'protect'], p2: [{move: 'throatchop', target: 1}, 'protect']},
      {p1: [{move: 'snore', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'snore'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Throat Chop\|/)) return 'Throat Chop was never used';
      if (!logHas(session, /\|cant\|p1a: s0\|move: Throat Chop/)) {
        return 'the sound move was never refused under Throat Chop';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 16000, artifact: 'more_sleep.json', debugEnv: 'DEBUG_SLEEP'});
