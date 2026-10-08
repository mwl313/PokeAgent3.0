// Development-only corpus for Sticky Hold: another Pokémon's removal move is
// refused (the holder keeps its item), while an identical Knock Off against a
// partner without the ability still removes its item.
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
    name: 'stickyhold_refuses_knock_off',
    p1: () => team(
      setOf('Swalot', 'Sticky Hold', ['Body Slam', 'Protect'], 'Leftovers'),
      offensive('Aggron', 'Sturdy', ['Iron Head', 'Protect'], 'Sitrus Berry'),
    ),
    p2: () => team(setOf('Absol', 'Super Luck', ['Knock Off', 'Protect'])),
    script: [
      // Turn 1 hits the Sticky Hold holder; turns 2-3 hit the control partner.
      // Neither target may Protect on the turn its item is removed.
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: [{move: 'knockoff', target: 1}, 'protect']},
      {p1: ['protect', {move: 'ironhead', target: 1}], p2: [{move: 'knockoff', target: 2}, 'protect']},
      {p1: ['protect', {move: 'ironhead', target: 1}], p2: [{move: 'knockoff', target: 2}, 'protect']},
    ],
    coverage: {ability: 'stickyhold'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Knock Off\|p1a: s0/)) return 'Knock Off never hit the holder';
      if (!logHas(session, /\|-activate\|p1a: s0\|ability: Sticky Hold/)) {
        return 'Sticky Hold never announced the refusal';
      }
      if (!everHas(fixture, 0, 0, p => p.item === ids.items.leftovers)) {
        return 'the holder lost its Leftovers anyway';
      }
      // A taken item never lands in `lastItem`, so the control is asserted on
      // the removal message plus the empty slot.
      if (!logHas(session, /\|-enditem\|p1b: s1\|Sitrus Berry\|\[from\] move: Knock Off/)) {
        return 'the control partner kept its berry';
      }
      if (!everHas(fixture, 0, 1, p => p.item === 0)) return 'the control slot never emptied';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 47000, artifact: 'more_stickyhold.json', debugEnv: 'DEBUG_STICKYHOLD'});
