// Development-only corpus for the suppression family:
//   Klutz      - the holder's held item is inert (Leftovers never heals)
//   Gastro Acid - the target's ability is suppressed until it leaves the field
//   Worry Seed  - the target's ability is replaced with Insomnia
//   Magic Room  - every held item is inert while the pseudo-weather lasts
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
    name: 'klutz_makes_the_held_item_inert',
    // P1a is a Klutz holder with Leftovers; P2b (no Klutz) also holds Leftovers.
    // Both take Dazzling Gleam damage, so only the non-Klutz holder heals.
    p1: () => team(
      setOf('Audino', 'Klutz', ['Dazzling Gleam', 'Protect'], 'Leftovers'),
      offensive('Alcremie', 'Aroma Veil', ['Dazzling Gleam', 'Protect']),
    ),
    p2: () => team(
      setOf('Golurk', 'Iron Fist', ['Earthquake', 'Protect']),
      offensive('Milotic', 'Competitive', ['Ice Beam', 'Protect'], 'Leftovers'),
    ),
    script: [
      {p1: ['protect', {move: 'dazzlinggleam', target: 0}], p2: [{move: 'earthquake', target: 0}, 'protect']},
      {p1: ['protect', {move: 'dazzlinggleam', target: 0}], p2: [{move: 'earthquake', target: 0}, 'protect']},
      {p1: ['protect', {move: 'dazzlinggleam', target: 0}], p2: [{move: 'earthquake', target: 0}, 'protect']},
    ],
    coverage: {ability: 'klutz'},
    verify(fixture, session) {
      if (!logHas(session, /\|-damage\|p1a: s0\|/)) return 'the Klutz holder was never damaged';
      if (logHas(session, /\|-heal\|p1a: s0\|.*\[from\] item: Leftovers/)) {
        return 'Klutz did not stop the Leftovers heal';
      }
      if (!logHas(session, /\|-heal\|p2b: s1\|.*\[from\] item: Leftovers/)) {
        return 'the control Leftovers never healed';
      }
      // The holder's HP must not rise on any recorded boundary where it was
      // damaged and nothing else could heal it.
      const trail = fixture.steps
        .map(step => step.expected.sides[0].pokemon.find(p => p.roster === 0)?.hp);
      for (let i = 1; i < trail.length; i++) {
        if (trail[i] > trail[i - 1]) return 'the Klutz holder healed at a residual';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 51000, artifact: 'more_suppression.json', debugEnv: 'DEBUG_SUPPRESSION'});
