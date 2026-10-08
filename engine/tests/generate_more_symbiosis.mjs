// Development-only corpus for `abilities:symbiosis` (Oranguru):
// - `onAllyAfterUseItem` hands the holder's item to the ally that just ate its
//   own Berry, so the eater ends up holding the passed item and the holder
//   goes empty,
// - the same scene without Symbiosis leaves both items where they started
//   (the control), and a holder with empty hands transfers nothing,
// - an ally whose ability also answers `AfterUseItem` (Unburden) resolves in
//   the same speed-sorted handler set.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, foeTeam, logHas, everHas, monAt, runTrials} = createScaffold();

const bench = (...species) => foeTeam()
  .filter(p => !species.includes(p.species));
const team = (...heads) => [
  ...heads,
  ...bench(...heads.map(p => p.species)),
].slice(0, 6);

const oranguru = (ability, item) => setOf('Oranguru', ability, ['Protect', 'Foul Play'], item);
const snorlax = (ability, item) => setOf('Snorlax', ability, ['Protect', 'Body Slam'], item);
const hawlucha = (item) => setOf('Hawlucha', 'Unburden', ['Protect', 'Acrobatics'], item);
// Both foes attack the berry holder so the Sitrus Berry fires in turn 2.
const pressure = [
  {p1: ['protect', 'protect'], p2: [{move: 'ironhead', target: 2}, {move: 'bodyslam', target: 2}]},
  {p1: ['protect', {move: 'bodyslam', target: 1}], p2: [{move: 'ironhead', target: 2}, {move: 'bodyslam', target: 2}]},
];
const last = (fixture, side, roster) => monAt(fixture, side, roster).at(-1);

const TRIALS = [
  {
    name: 'symbiosis_hands_its_item_to_the_ally_that_ate',
    p1: () => team(oranguru('Symbiosis', 'Leftovers'), snorlax('Thick Fat', 'Sitrus Berry')),
    p2: () => foeTeam(),
    script: pressure,
    coverage: {ability: 'symbiosis'},
    verify(fixture, session) {
      if (!logHas(session, /\|-heal\|p1b: s1\|.*Sitrus Berry/)) return 'the ally never ate its Sitrus Berry';
      if (!logHas(session, /\|-activate\|p1a: s0\|ability: Symbiosis\|Leftovers\|\[of\] p1b: s1/)) {
        return 'Symbiosis never announced the hand-off';
      }
      if (last(fixture, 0, 1).item !== ids.items.leftovers) {
        return 'the eater never received the passed item';
      }
      if (last(fixture, 0, 1).previous_item !== ids.items.sitrusberry) {
        return 'the eater did not record the eaten Berry as its last item';
      }
      if (last(fixture, 0, 0).item !== 0) {
        return 'the Symbiosis holder kept its item after the hand-off';
      }
      return null;
    },
  },
  {
    name: 'symbiosis_control_without_the_ability',
    p1: () => team(oranguru('Inner Focus', 'Leftovers'), snorlax('Thick Fat', 'Sitrus Berry')),
    p2: () => foeTeam(),
    script: pressure,
    coverage: {ability: 'symbiosis'},
    verify(fixture, session) {
      if (!logHas(session, /\|-heal\|p1b: s1\|.*Sitrus Berry/)) return 'the ally never ate its Sitrus Berry';
      if (logHas(session, /ability: Symbiosis/)) return 'an item moved without the ability';
      if (last(fixture, 0, 0).item !== ids.items.leftovers) {
        return 'the holder lost its item without Symbiosis';
      }
      if (last(fixture, 0, 1).item !== 0) {
        return 'the eater gained an item without Symbiosis';
      }
      return null;
    },
  },
  {
    name: 'symbiosis_control_empty_holder',
    p1: () => team(oranguru('Symbiosis', ''), snorlax('Thick Fat', 'Sitrus Berry')),
    p2: () => foeTeam(),
    script: pressure,
    coverage: {ability: 'symbiosis'},
    verify(fixture, session) {
      if (!logHas(session, /\|-heal\|p1b: s1\|.*Sitrus Berry/)) return 'the ally never ate its Sitrus Berry';
      if (logHas(session, /ability: Symbiosis/)) return 'Symbiosis announced a hand-off without an item';
      if (last(fixture, 0, 1).item !== 0) return 'the eater gained an item from an empty holder';
      return null;
    },
  },
  {
    name: 'symbiosis_and_unburden_share_the_handler_set',
    p1: () => team(oranguru('Symbiosis', 'Leftovers'), hawlucha('Sitrus Berry')),
    p2: () => foeTeam(),
    script: [
      {p1: ['protect', 'protect'], p2: [{move: 'ironhead', target: 2}, {move: 'bodyslam', target: 2}]},
      {p1: ['protect', {move: 'acrobatics', target: 1}], p2: [{move: 'ironhead', target: 2}, {move: 'bodyslam', target: 2}]},
    ],
    coverage: {ability: 'symbiosis'},
    verify(fixture, session) {
      if (!logHas(session, /\|-heal\|p1b: s1\|.*Sitrus Berry/)) return 'the ally never ate its Sitrus Berry';
      if (!logHas(session, /\|-activate\|p1a: s0\|ability: Symbiosis\|Leftovers\|\[of\] p1b: s1/)) {
        return 'Symbiosis never announced the hand-off';
      }
      if (last(fixture, 0, 1).item !== ids.items.leftovers) {
        return 'the Unburden holder never received the passed item';
      }
      // The volatile is removed again by `unburden.onEnd` when the holder
      // faints, so any boundary may carry it.
      if (!everHas(fixture, 0, 1, p => p.volatiles.includes('unburden'))) {
        return 'the ally never gained the Unburden volatile';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 8100, artifact: 'more_symbiosis.json', debugEnv: 'DEBUG_SYMBIOSIS'});
