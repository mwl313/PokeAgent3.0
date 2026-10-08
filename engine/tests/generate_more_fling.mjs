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
// The same throw against a foe that does not Protect, so the item's on-hit
// payload actually lands.
const strikeTurn = [
  {p1: [{move: 'fling', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
];
// Same, but aimed at the foe's second slot (a Snorlax: a status payload needs a
// target that can actually take the status).
const strikeTurnB = [
  {p1: [{move: 'fling', target: 2}, 'protect'], p2: ['protect', {move: 'bodyslam', target: 1}]},
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
  {
    name: 'fling_sitrus_berry_heals_the_target',
    p1: () => team(sneasler('Sitrus Berry'), partner()),
    p2: () => foeTeam(),
    script: [
      {p1: ['protect', {move: 'seedbomb', target: 1}], p2: ['protect', 'protect']},
      {p1: [{move: 'fling', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'fling'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Fling\|p2a: s0/)) return 'Fling never executed';
      if (!logHas(session, /\|-heal\|p2a: s0\|.*Sitrus Berry/)) {
        return 'the target never ate the flung Berry';
      }
      if (!everHas(fixture, 0, 0, p => p.item === 0)) return 'the user kept the thrown Berry';
      return null;
    },
  },
  {
    name: 'fling_poison_barb_poisons_the_target',
    p1: () => team(sneasler('Poison Barb'), partner()),
    p2: () => foeTeam(),
    script: strikeTurnB,
    coverage: {move: 'fling'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Fling\|p2b: s1/)) return 'Fling never executed';
      if (!logHas(session, /\|-status\|p2b: s1\|psn/)) return 'the flung Poison Barb never poisoned';
      if (!everHas(fixture, 1, 1, p => p.status === ids.conditions.psn)) {
        return 'the poisoned status was never recorded';
      }
      return null;
    },
  },
  {
    name: 'fling_white_herb_clears_negative_boosts',
    p1: () => team(sneasler('White Herb'), setOf('Milotic', 'Competitive', ['Icy Wind', 'Protect'])),
    p2: () => foeTeam(),
    script: [
      {p1: ['protect', {move: 'icywind'}], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'fling', target: 2}, 'protect'], p2: ['protect', {move: 'bodyslam', target: 1}]},
    ],
    coverage: {move: 'fling'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Fling\|p2b: s1/)) return 'Fling never executed';
      if (!logHas(session, /\|-unboost\|p2b: s1\|spe\|1/)) return 'the target never carried a negative boost';
      const boosts = fixture.steps.map(step => step.expected.sides[1].pokemon
        .find(p => p.roster === 1).boosts[4]);
      const cleared = boosts.some((speed, i) => i > 0 && speed === 0 && boosts[i - 1] < 0);
      if (!cleared) return 'the flung White Herb never cleared the drop';
      return null;
    },
  },
  {
    name: 'fling_mental_herb_ends_the_taunt',
    p1: () => team(setOf('Sneasler', 'Unburden', ['Taunt', 'Fling', 'Protect'], 'Mental Herb'), partner()),
    p2: () => foeTeam(),
    script: [
      {p1: [{move: 'taunt', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'fling', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
    ],
    coverage: {move: 'fling'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Fling\|p2a: s0/)) return 'Fling never executed';
      if (!logHas(session, /\|-start\|p2a: s0\|move: Taunt/)) return 'the target was never taunted';
      const taunted = fixture.steps.map(step => step.expected.sides[1].pokemon
        .find(p => p.roster === 0).volatiles.includes('taunt'));
      const ended = taunted.some((value, i) => i > 0 && !value && taunted[i - 1]);
      if (!ended) return 'the flung Mental Herb never ended the taunt';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 8300, artifact: 'more_fling.json', debugEnv: 'DEBUG_FLING'});
