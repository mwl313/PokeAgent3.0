// Development-only corpus for Innards Out, which is only reachable through the
// permitted Mega Victreebel battle form: once the holder is knocked out, the
// attacker takes the felling hit's damage back.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, logHas, runTrials} = createScaffold();

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
    name: 'innardsout_punishes_the_knockout',
    // Chlorophyll keeps the base form inside the ported set; the mega's Innards
    // Out is what the scene exercises.
    p1: () => team(setOf('Victreebel', 'Chlorophyll', ['Protect', 'Sludge Bomb'], 'Victreebelite')),
    p2: () => team(setOf('Aggron', 'Sturdy', ['Iron Head', 'Protect'])),
    script: [
      // Mega-evolve on the first turn, then let the Iron Heads knock it out.
      {p1: [{move: 'protect', mega: true}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'sludgebomb', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'sludgebomb', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'sludgebomb', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
    ],
    coverage: {ability: 'innardsout'},
    verify(fixture, session) {
      if (!logHas(session, /\|-mega\|p1a: s0\|Victreebel/)) return 'the holder never mega-evolved';
      if (!logHas(session, /\|faint\|p1a: s0/)) return 'the holder was never knocked out';
      if (!logHas(session, /\|-damage\|p2[ab]: s\d+\|.*\[from\] ability: Innards Out/)) {
        return 'the attacker was not damaged by Innards Out';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 49000, artifact: 'more_innardsout.json', debugEnv: 'DEBUG_INNARDSOUT'});
