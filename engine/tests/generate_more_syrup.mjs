// Development-only corpus for Early Bird (sleep ticks twice) and Supersweet
// Syrup (a once-per-battle evasion drop on entry).
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
    name: 'earlybird_halves_the_sleep_timer',
    p1: () => team(setOf('Houndoom', 'Early Bird', ['Dark Pulse', 'Protect'])),
    p2: () => team(setOf('Venusaur', 'Overgrow', ['Sleep Powder', 'Sludge Bomb', 'Protect'])),
    script: [
      // No Protect on the sleep turn, so Sleep Powder can land.
      {p1: [{move: 'darkpulse', target: 1}, 'protect'], p2: [{move: 'sleeppowder', target: 1}, 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {ability: 'earlybird'},
    verify(fixture, session) {
      if (!logHas(session, /\|-status\|p1a: s0\|slp/)) return 'the holder never fell asleep';
      if (!logHas(session, /\|-curestatus\|p1a: s0\|slp/)) return 'the sleep was never cured';
      const statuses = fixture.steps
        .map(step => step.expected.sides[0].pokemon.find(p => p.roster === 0)?.status)
        .filter(status => status !== undefined);
      const asleep = statuses.indexOf(ids.conditions.slp);
      if (asleep < 0) return 'no boundary recorded the sleep';
      if (!statuses.slice(asleep + 1).includes(0)) return 'the holder never woke up';
      return null;
    },
  },
  {
    name: 'supersweetsyrup_drops_evasion_once',
    // Hydrapple leaves and re-enters; the second entry must not drop the foes
    // a second time.
    p1: () => team(
      setOf('Hydrapple', 'Supersweet Syrup', ['Protect', 'Body Press']),
      offensive('Milotic', 'Competitive', ['Ice Beam', 'Protect']),
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
    ),
    p2: () => team(setOf('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      // Slot 0 switches out and returns on the following turn.
      {p1: [{switch: 's2'}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{switch: 's0'}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {ability: 'supersweetsyrup'},
    verify(fixture, session) {
      if (!logHas(session, /\|switch\|p1a: s0\|Hydrapple/)) return 'the holder never entered';
      const dropped = everHas(fixture, 1, 0, p => p.boosts[6] === -1)
        && everHas(fixture, 1, 1, p => p.boosts[6] === -1);
      if (!dropped) return 'the entry evasion drop never landed on both foes';
      const doubled = everHas(fixture, 1, 0, p => p.boosts[6] <= -2)
        || everHas(fixture, 1, 1, p => p.boosts[6] <= -2);
      if (doubled) return 'the drop fired a second time after re-entering';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 46000, artifact: 'more_syrup.json', debugEnv: 'DEBUG_SYRUP'});
