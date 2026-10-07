// Development-only corpus for four small self/crit moves:
// - Belly Drum pays half the user's maximum HP to reach +6 Attack (and fails
//   at half HP or less),
// - Focus Energy / Dragon Cheer add the mutually exclusive crit-ratio
//   volatiles (Dragon Cheer scales with the target's captured typing),
// - Acupressure raises one sampled stat below +6 by two stages.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, logHas, monAt, runTrials} = createScaffold();

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

const boosted = p => p.boosts.reduce((a, b) => a + Math.abs(b), 0);

const TRIALS = [
  {
    name: 'smallcrit_bellydrum_reaches_max_attack',
    p1: () => team(setOf('Snorlax', 'Thick Fat', ['Belly Drum', 'Body Slam', 'Protect'])),
    p2: () => team(setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic'])),
    script: [
      {p1: [{move: 'bellydrum', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'bellydrum'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Belly Drum\|/)) return 'Belly Drum never executed';
      const steps = fixture.steps.map(step => step.expected.sides[0].pokemon.find(p => p.roster === 0));
      const drummed = steps.find(p => p && p.boosts[0] === 6);
      if (!drummed) return 'Attack never reached +6';
      if (drummed.hp !== drummed.max_hp - Math.trunc(drummed.max_hp / 2)) {
        return 'Belly Drum did not pay half the maximum HP';
      }
      if (steps.some(p => p && boosted(p) > 6)) return 'a stat outside Attack changed';
      return null;
    },
  },
  {
    name: 'smallcrit_bellydrum_fails_at_half_hp',
    p1: () => team(setOf('Snorlax', 'Thick Fat', ['Belly Drum', 'Body Slam', 'Protect'])),
    p2: () => team(offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic'])),
    script: [
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'bellydrum', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'bellydrum'},
    verify(fixture, session) {
      const before = fixture.steps
        .filter(step => step.expected.turn <= 2)
        .map(step => step.expected.sides[0].pokemon.find(p => p.roster === 0))
        .filter(Boolean)
        .at(-1);
      if (!before || before.hp * 2 > before.max_hp) return 'the user was not at half HP before Belly Drum';
      if (!logHas(session, /\|move\|p1a: s0\|Belly Drum\|/)) return 'Belly Drum was never used';
      if (!logHas(session, /\|-fail\|p1a: s0/)) return 'Belly Drum did not fail at half HP';
      const after = monAt(fixture, 0, 0).at(-1);
      if (!after || after.boosts[0] > 0) return 'Attack was boosted despite the failure';
      return null;
    },
  },
  {
    name: 'smallcrit_focusenergy_crits_through_high_ratio_move',
    p1: () => team(setOf('Absol', 'Super Luck', ['Focus Energy', 'Night Slash', 'Protect'])),
    p2: () => team(setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic'])),
    script: [
      {p1: [{move: 'focusenergy', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'nightslash', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'nightslash', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'focusenergy'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Focus Energy\|/)) return 'Focus Energy never executed';
      if (!monAt(fixture, 0, 0).some(p => p.volatiles.includes('focusenergy'))) {
        return 'the focusenergy volatile was never recorded';
      }
      if (!logHas(session, /\|-crit\|p2a: s0/)) return 'the boosted crit ratio never produced a crit';
      return null;
    },
  },
  {
    name: 'smallcrit_dragoncheer_grants_ally_crit_ratio',
    p1: () => team(
      setOf('Dragonite', 'Inner Focus', ['Dragon Claw', 'Protect', 'Extreme Speed']),
      setOf('Ampharos', 'Static', ['Dragon Cheer', 'Protect', 'Dragon Pulse']),
    ),
    p2: () => team(setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic'])),
    script: [
      {p1: ['protect', {move: 'dragoncheer', target: -1}], p2: ['protect', 'protect']},
      {p1: [{move: 'dragonclaw', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'dragonclaw', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'dragoncheer'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1b: s1\|Dragon Cheer\|/)) return 'Dragon Cheer never executed';
      if (!monAt(fixture, 0, 0).some(p => p.volatiles.includes('dragoncheer'))) {
        return 'the dragoncheer volatile was never recorded on the Dragon-type ally';
      }
      if (!logHas(session, /\|-crit\|p2a: s0/)) return 'the boosted crit ratio never produced a crit';
      return null;
    },
  },
  {
    name: 'smallcrit_acupressure_raises_one_sampled_stat',
    p1: () => team(setOf('Medicham', 'Pure Power', ['Acupressure', 'Protect', 'Close Combat'])),
    p2: () => team(setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic'])),
    script: [
      {p1: [{move: 'acupressure', target: -1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'acupressure', target: -1}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'acupressure'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Acupressure\|/)) return 'Acupressure never executed';
      const steps = fixture.steps.map(step => step.expected.sides[0].pokemon.find(p => p.roster === 0));
      const first = steps.find(p => p && boosted(p) > 0);
      if (!first) return 'no stat ever rose';
      const nonzero = first.boosts.filter(b => b !== 0);
      if (nonzero.length !== 1 || nonzero[0] !== 2) {
        return 'the first Acupressure did not raise exactly one stat by two stages';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 19000, artifact: 'more_smallcrit.json', debugEnv: 'DEBUG_SMALLCRIT'});
