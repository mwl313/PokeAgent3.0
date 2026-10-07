// Development-only corpus for the on-hit hazard setters:
// - Ceaseless Edge scatters one Spikes layer onto the foe side when it lands,
// - Stone Axe sets Stealth Rock on the foe side when it lands,
// - Stone Axe under Sheer Force never sets the hazard (the ability consumed
//   the action's secondary), and
// - a decoy that absorbs Ceaseless Edge still scatters Spikes through
//   `onAfterSubDamage`.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, logHas, runTrials} = createScaffold();

const SPIKES = ids.conditions.spikes;
const STEALTH_ROCK = ids.conditions.stealthrock;

const fill = () => [
  offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
  offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
  offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
  offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
  offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
];
const team = head => [head, ...fill()];
const withBench = (actives, bench) => [...actives, ...bench];
const conditionsAt = (fixture, turn, side) => {
  const step = fixture.steps.find(entry => entry.expected.turn >= turn);
  return step ? step.expected.sides[side].conditions : [];
};
const layers = (conditions, id) => {
  const hit = conditions.find(([cond]) => cond === id);
  return hit ? hit[1] : null;
};

const foePair = () => withBench(
  [offensive('Milotic', 'Competitive', ['Surf', 'Ice Beam', 'Protect']),
    offensive('Talonflame', 'Flame Body', ['Brave Bird', 'Protect'])],
  [offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Substitute', 'Protect']),
    offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
    offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
    offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect'])]);

const TRIALS = [
  {
    name: 'ceaselessedge_sets_spikes',
    p1: () => team(setOf('Samurott-Hisui', 'Torrent', ['Ceaseless Edge', 'Protect', 'Aqua Cutter'])),
    p2: foePair,
    script: [
      {p1: [{move: 'ceaselessedge', target: 1}, 'protect'], p2: [{move: 'icebeam', target: 1}, 'protect']},
    ],
    coverage: {move: 'ceaselessedge'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: [^|]*\|Ceaseless Edge\|/)) return 'Ceaseless Edge never resolved';
      if (layers(conditionsAt(fixture, 2, 1), SPIKES) !== 1) return 'Ceaseless Edge did not set one Spikes layer';
      return null;
    },
  },
  {
    name: 'stoneaxe_sets_stealthrock',
    p1: () => team(setOf('Kleavor', 'Swarm', ['Stone Axe', 'Protect', 'X-Scissor'])),
    p2: foePair,
    script: [
      {p1: [{move: 'stoneaxe', target: 1}, 'protect'], p2: [{move: 'icebeam', target: 1}, 'protect']},
    ],
    coverage: {move: 'stoneaxe'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: [^|]*\|Stone Axe\|/)) return 'Stone Axe never resolved';
      if (layers(conditionsAt(fixture, 2, 1), STEALTH_ROCK) !== 0) return 'Stone Axe did not set Stealth Rock';
      return null;
    },
  },
  {
    name: 'stoneaxe_sheerforce_suppresses',
    p1: () => team(setOf('Kleavor', 'Sheer Force', ['Stone Axe', 'Protect', 'X-Scissor'])),
    p2: foePair,
    script: [
      {p1: [{move: 'stoneaxe', target: 1}, 'protect'], p2: [{move: 'icebeam', target: 1}, 'protect']},
    ],
    coverage: {move: 'stoneaxe'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: [^|]*\|Stone Axe\|/)) return 'Stone Axe never resolved';
      if (fixture.steps.some(step => step.expected.sides[1].conditions.some(([id]) => id === STEALTH_ROCK))) {
        return 'Sheer Force Stone Axe still set Stealth Rock';
      }
      return null;
    },
  },
  {
    name: 'ceaselessedge_substitute_scatters',
    p1: () => team(setOf('Samurott-Hisui', 'Torrent', ['Ceaseless Edge', 'Protect', 'Aqua Cutter'])),
    p2: () => withBench(
      [offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Substitute', 'Protect']),
        offensive('Talonflame', 'Flame Body', ['Brave Bird', 'Protect'])],
      [offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
        offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
        offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
        offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect'])]),
    script: [
      {p1: ['protect', 'protect'], p2: ['substitute', 'protect']},
      {p1: [{move: 'ceaselessedge', target: 1}, 'protect'], p2: [{move: 'bodyslam', target: 1}, 'protect']},
    ],
    coverage: {move: 'ceaselessedge'},
    verify(fixture, session) {
      if (!logHas(session, /\|-start\|p2a: [^|]*\|Substitute/)) return 'the decoy was never raised';
      if (!logHas(session, /\|move\|p1a: [^|]*\|Ceaseless Edge\|/)) return 'Ceaseless Edge never resolved';
      if (layers(conditionsAt(fixture, 3, 1), SPIKES) !== 1) return 'the decoy-absorbed Ceaseless Edge did not scatter Spikes';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 9500, artifact: 'more_hazardsetters.json', debugEnv: 'DEBUG_HAZARDSETTERS'});
