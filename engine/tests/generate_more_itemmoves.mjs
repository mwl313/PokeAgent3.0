// Development-only corpus for the item-move family:
// - Thief and Covet take a damaged target's item while the user is empty,
// - Thief does nothing when the user already holds an item,
// - Corrosive Gas destroys every adjacent target's item, and
// - Recycle restores the user's last consumed item.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, logHas, everHas, runTrials} = createScaffold();

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

const TRIALS = [
  {
    name: 'itemmoves_thief_steals_target_item',
    p1: () => team(offensive('Absol', 'Super Luck', ['Thief', 'Protect', 'Night Slash'])),
    p2: () => team(setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect'], 'Leftovers')),
    script: [{p1: [{move: 'thief', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']}],
    coverage: {move: 'thief'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Thief\|p2a: s0/)) return 'Thief never landed';
      if (!everHas(fixture, 0, 0, p => p.item === ids.items.leftovers)) {
        return 'the user never received the target item';
      }
      if (!everHas(fixture, 1, 0, p => p.item === 0)) {
        return 'the target kept the stolen item';
      }
      return null;
    },
  },
  {
    name: 'itemmoves_thief_skips_when_holder_has_item',
    p1: () => team(setOf('Absol', 'Super Luck', ['Thief', 'Protect', 'Night Slash'], 'Sitrus Berry')),
    p2: () => team(setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect'], 'Leftovers')),
    script: [{p1: [{move: 'thief', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']}],
    coverage: {move: 'thief'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Thief\|p2a: s0/)) return 'Thief never landed';
      if (!everHas(fixture, 1, 0, p => p.item === ids.items.leftovers)) {
        return 'the target item was taken despite the held item';
      }
      return null;
    },
  },
  {
    name: 'itemmoves_covet_steals_target_item',
    p1: () => team(offensive('Emolga', 'Static', ['Covet', 'Protect', 'Nuzzle'])),
    p2: () => team(setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect'], 'Leftovers')),
    script: [{p1: [{move: 'covet', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']}],
    coverage: {move: 'covet'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Covet\|p2a: s0/)) return 'Covet never landed';
      if (!everHas(fixture, 0, 0, p => p.item === ids.items.leftovers)) {
        return 'the user never received the target item';
      }
      if (!everHas(fixture, 1, 0, p => p.item === 0)) {
        return 'the target kept the stolen item';
      }
      return null;
    },
  },
  {
    name: 'itemmoves_corrosivegas_destroys_adjacent_items',
    p1: () => team(offensive('Garbodor', 'Weak Armor', ['Corrosive Gas', 'Protect', 'Body Slam'])),
    p2: () => team(
      setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect'], 'Leftovers'),
      setOf('Snorlax', 'Thick Fat', ['Body Slam', 'Protect'], 'Sitrus Berry'),
    ),
    script: [{p1: ['corrosivegas', 'protect'], p2: [{move: 'ironhead', target: 1}, {move: 'bodyslam', target: 1}]}],
    coverage: {move: 'corrosivegas'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Corrosive Gas\|/)) return 'Corrosive Gas never executed';
      if (!everHas(fixture, 1, 0, p => p.item === 0)) return 'the first target kept its item';
      if (!everHas(fixture, 1, 1, p => p.item === 0)) return 'the second target kept its item';
      return null;
    },
  },
  {
    name: 'itemmoves_recycle_restores_consumed_item',
    p1: () => team(
      offensive('Chimecho', 'Levitate', ['Recycle', 'Protect', 'Dazzling Gleam'], 'Sitrus Berry'),
    ),
    p2: () => team(offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic'])),
    script: [
      {p1: ['protect', 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'recycle', target: 0}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
    ],
    coverage: {move: 'recycle'},
    verify(fixture, session) {
      if (!everHas(fixture, 0, 0, p => p.item === 0 && p.previous_item === ids.items.sitrusberry)) {
        return 'the Sitrus Berry was never consumed and recorded';
      }
      if (!logHas(session, /\|move\|p1a: s0\|Recycle\|/)) return 'Recycle never executed';
      if (!everHas(fixture, 0, 0, p => p.item === ids.items.sitrusberry)) {
        return 'Recycle never restored the consumed berry';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 15000, artifact: 'more_itemmoves.json', debugEnv: 'DEBUG_ITEMMOVES'});
