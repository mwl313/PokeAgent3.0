// Development-only corpus for Magician:
// - a damaging hit steals the first hit target's item while the user is empty,
// - a holder that already carries an item steals nothing, and
// - an itemless target leaves the Magician holder empty-handed.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, fillerAfter, foeTeam, logHas, everHas, runTrials} = createScaffold();

const TRIALS = [
  {
    name: 'magician_steals_target_item',
    p1: () => fillerAfter(setOf('Klefki', 'Magician', ['Play Rough', 'Protect'])),
    p2: () => [
      setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect'], 'Leftovers'),
      ...foeTeam().filter(p => p.species !== 'Metagross').slice(0, 5),
    ],
    script: [{p1: [{move: 'playrough', target: 1}, 'protect'], p2: ['protect', 'protect']}],
    coverage: {ability: 'magician'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Play Rough\|p2a: s0/)) return 'Play Rough never landed';
      if (!logHas(session, /\|-item\|p1a: s0\|Leftovers\|\[from\] ability: Magician/)) {
        return 'the item was never stolen with the Magician attribution';
      }
      if (!everHas(fixture, 0, 0, p => p.item === ids.items.leftovers)) {
        return 'the Magician holder never received the target item';
      }
      if (!everHas(fixture, 1, 0, p => p.item === 0)) {
        return 'the target kept the stolen item';
      }
      return null;
    },
  },
  {
    name: 'magician_control_holder_already_has_item',
    p1: () => fillerAfter(setOf('Klefki', 'Magician', ['Play Rough', 'Protect'], 'Leftovers')),
    p2: () => [
      setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect'], 'Sitrus Berry'),
      ...foeTeam().filter(p => p.species !== 'Metagross').slice(0, 5),
    ],
    script: [{p1: [{move: 'playrough', target: 1}, 'protect'], p2: ['protect', 'protect']}],
    coverage: {ability: 'magician'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Play Rough\|p2a: s0/)) return 'Play Rough never landed';
      if (logHas(session, /\[from\] ability: Magician/)) return 'an item was stolen despite the held item';
      if (!everHas(fixture, 0, 0, p => p.item === ids.items.leftovers)) {
        return 'the Magician holder lost its own item';
      }
      if (!everHas(fixture, 1, 0, p => p.item === ids.items.sitrusberry)) {
        return 'the target item was removed without a successful steal';
      }
      return null;
    },
  },
  {
    name: 'magician_control_itemless_target',
    p1: () => fillerAfter(setOf('Klefki', 'Magician', ['Play Rough', 'Protect'])),
    p2: () => foeTeam(),
    script: [{p1: [{move: 'playrough', target: 1}, 'protect'], p2: ['protect', 'protect']}],
    coverage: {ability: 'magician'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Play Rough\|p2a: s0/)) return 'Play Rough never landed';
      if (logHas(session, /\[from\] ability: Magician/)) return 'an item was stolen from an itemless target';
      if (everHas(fixture, 0, 0, p => p.item !== 0)) return 'the empty-handed holder gained an item';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 8000, artifact: 'more_magician.json', debugEnv: 'DEBUG_MAGICIAN'});
