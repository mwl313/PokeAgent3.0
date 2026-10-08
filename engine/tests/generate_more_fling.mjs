// Development-only corpus for Fling on the pinned plain-item path:
// - a thrown item with a `fling` base power deals that damage and is consumed
//   by the marker volatile's `onUpdate` (the loss also feeds Unburden),
// - empty hands fail outright,
// - an item without `fling` data (Normal Gem) is not thrown and stays held.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, foeTeam, logHas, everHas, runTrials} = createScaffold();

const team = (...heads) => [
  ...heads,
  ...foeTeam().filter(p => !heads.some(head => head.species === p.species)),
].slice(0, 6);
const sneasler = (item) => setOf('Sneasler', 'Unburden', ['Fling', 'Protect'], item);
// A partner that cannot tie the foe list's own Snorlax: a mirrored speed tie
// would make the scene sensitive to queue tie-breaks instead of Fling.
const partner = () => setOf('Torterra', 'Shell Armor', ['Protect', 'Seed Bomb']);

const throwTurn = [
  {p1: [{move: 'fling', target: 1}, 'protect'], p2: ['protect', 'protect']},
];
// Log lines between the Fling announcement and the next action boundary: a
// refused throw must not deal damage inside its own move window.
const flingWindow = (session) => {
  const log = session.battle.log;
  const at = log.findIndex(line => line.startsWith('|move|p1a: s0|Fling'));
  if (at < 0) return null;
  const rest = log.slice(at + 1);
  const end = rest.findIndex(line => line.startsWith('|move|') || line.startsWith('|upkeep|') || line.startsWith('|turn|'));
  return rest.slice(0, end < 0 ? undefined : end);
};

const TRIALS = [
  {
    name: 'fling_iron_ball_deals_and_consumes_the_item',
    p1: () => team(sneasler('Iron Ball'), partner()),
    p2: () => foeTeam(),
    script: throwTurn,
    coverage: {move: 'fling'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Fling\|p2a: s0/)) return 'Fling never executed';
      if (!logHas(session, /\|-damage\|p2a: s0\|/)) return 'the thrown item never dealt damage';
      if (!logHas(session, /\|-enditem\|p1a: s0\|Iron Ball\|\[from\] move: Fling/)) {
        return 'the thrown item was never consumed with the Fling attribution';
      }
      if (!everHas(fixture, 0, 0, p => p.item === 0)) return 'the user kept the thrown item';
      if (!everHas(fixture, 0, 0, p => p.previous_item === ids.items.ironball)) {
        return 'the consumed item never became the last item';
      }
      if (!everHas(fixture, 0, 0, p => p.volatiles.includes('unburden'))) {
        return 'the item loss never granted Unburden';
      }
      return null;
    },
  },
  {
    name: 'fling_without_an_item_fails',
    p1: () => team(sneasler(''), partner()),
    p2: () => foeTeam(),
    script: throwTurn,
    coverage: {move: 'fling'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Fling/)) return 'Fling never executed';
      if (flingWindow(session).some(line => line.startsWith('|-damage|'))) {
        return 'an itemless Fling dealt damage';
      }
      if (logHas(session, /\[from\] move: Fling/)) return 'an itemless Fling consumed something';
      if (!everHas(fixture, 0, 0, p => p.item === 0)) return 'the itemless user gained an item';
      return null;
    },
  },
  {
    name: 'fling_item_without_data_is_refused',
    p1: () => team(sneasler('Normal Gem'), partner()),
    p2: () => foeTeam(),
    script: throwTurn,
    coverage: {move: 'fling'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Fling/)) return 'Fling never executed';
      if (flingWindow(session).some(line => line.startsWith('|-damage|'))) {
        return 'a fling-less item dealt damage';
      }
      if (!everHas(fixture, 0, 0, p => p.item === ids.items.normalgem)) {
        return 'the fling-less item was consumed anyway';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 8300, artifact: 'more_fling.json', debugEnv: 'DEBUG_FLING'});
