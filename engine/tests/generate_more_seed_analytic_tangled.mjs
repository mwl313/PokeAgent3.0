// Development-only corpus for three legal-ability-tail entries:
// - Seed Sower scatters Grassy Terrain on any damaging hit, sourced to the
//   holder (Terrain Extender included).
// - Analytic boosts 1.3x while no other active Pokémon still has a queued move.
// - Tangled Feet halves the accuracy of moves aimed at a confused holder.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, fillerAfter, foeTeam, foeWith, logHas, everHas, runTrials} = createScaffold();

const fast = (species, ability, moves, item = '') =>
  setOf(species, ability, moves, item, {hp: 2, atk: 32, def: 0, spa: 0, spd: 0, spe: 32});
const slow = (species, ability, moves, item = '') =>
  setOf(species, ability, moves, item, {hp: 32, atk: 32, def: 0, spa: 0, spd: 0, spe: 0});

const TRIALS = [
  {
    name: 'seedsower_scatters_grassy_terrain',
    p1: () => fillerAfter(setOf('Arboliva', 'Seed Sower', ['Leaf Storm', 'Protect'])),
    p2: () => foeWith(offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect'])),
    script: [{p1: ['protect', 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']}],
    coverage: {ability: 'seedsower'},
    verify(fixture, session) {
      if (!logHas(session, /\|-fieldstart\|move: Grassy Terrain\|\[from\] ability: Seed Sower/)) {
        return 'Seed Sower never scattered Grassy Terrain';
      }
      const terrain = ids.conditions.grassyterrain;
      if (!fixture.steps.some(step => step.expected.field.some(([id]) => id === terrain))) {
        return 'the recorded boundaries never show the terrain';
      }
      return null;
    },
  },
  {
    name: 'seedsower_does_not_refresh_existing_terrain',
    p1: () => [
      setOf('Arboliva', 'Seed Sower', ['Leaf Storm', 'Protect']),
      setOf('Pincurchin', 'Electric Surge', ['Thunderbolt', 'Protect']),
      ...foeTeam().slice(0, 4),
    ],
    p2: () => foeWith(offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect'])),
    script: [{p1: ['protect', 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']}],
    coverage: {ability: 'seedsower'},
    verify(fixture, session) {
      // Electric Surge starts Electric Terrain; Seed Sower's Grassy Terrain
      // still replaces it because the ids differ.
      if (!logHas(session, /\|-fieldstart\|move: Grassy Terrain/)) {
        return 'Seed Sower did not replace the existing terrain';
      }
      return null;
    },
  },
  {
    name: 'analytic_boosts_when_moving_last',
    p1: () => [
      slow('Watchog', 'Analytic', ['Crunch', 'Protect']),
      fast('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
      offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
      offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
      offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
      offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
    ],
    p2: () => [
      fast('Milotic', 'Competitive', ['Surf', 'Protect']),
      fast('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
      offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
      offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
      offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
    ],
    script: [{p1: [{move: 'crunch', target: 1}, 'protect'], p2: ['surf', 'protect']}],
    coverage: {ability: 'analytic'},
    verify(fixture, session) {
      const first = session.battle.log.find(line => /\|move\|p1a: s0\|Crunch\|/.test(line));
      if (!first || first.includes('[still]')) return 'the Analytic Crunch never executed';
      return null;
    },
  },
  {
    name: 'analytic_control_moving_first',
    p1: () => [
      fast('Watchog', 'Analytic', ['Crunch', 'Protect']),
      slow('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
      offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
      offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
      offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
      offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
    ],
    p2: () => [
      slow('Milotic', 'Competitive', ['Surf', 'Protect']),
      slow('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
      offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
      offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
      offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
    ],
    script: [{p1: [{move: 'crunch', target: 1}, 'protect'], p2: ['surf', 'protect']}],
    coverage: {ability: 'analytic'},
    verify(fixture, session) {
      const first = session.battle.log.find(line => /\|move\|p1a: s0\|Crunch\|/.test(line));
      if (!first || first.includes('[still]')) return 'the control Crunch never executed';
      return null;
    },
  },
  {
    name: 'tangledfeet_halves_accuracy_when_confused',
    p1: () => fillerAfter(setOf('Pidgeot', 'Tangled Feet', ['Hurricane', 'Protect'])),
    p2: () => [
      offensive('Ampharos', 'Static', ['Confuse Ray', 'Protect']),
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
      ...foeTeam().filter(p => !['Ampharos', 'Metagross'].includes(p.species)).slice(0, 4),
    ],
    script: [
      {p1: ['hurricane', 'protect'], p2: [{move: 'confuseray', target: 1}, 'protect']},
      {p1: ['hurricane', 'protect'], p2: ['protect', {move: 'ironhead', target: 1}]},
      {p1: ['hurricane', 'protect'], p2: ['protect', {move: 'ironhead', target: 1}]},
      {p1: ['hurricane', 'protect'], p2: ['protect', {move: 'ironhead', target: 1}]},
    ],
    coverage: {ability: 'tangledfeet'},
    verify(fixture, session) {
      if (!everHas(fixture, 0, 0, p => p.volatiles.includes('confusion'))) {
        return 'the holder was never confused';
      }
      if (!logHas(session, /\|-miss\|p2b: s1\|p1a: s0/)) {
        return 'the halved-accuracy Iron Head never missed across the attempted seeds';
      }
      return null;
    },
  },
  {
    name: 'tangledfeet_control_full_accuracy',
    p1: () => fillerAfter(setOf('Pidgeot', 'Big Pecks', ['Hurricane', 'Protect'])),
    p2: () => [
      offensive('Ampharos', 'Static', ['Confuse Ray', 'Protect']),
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
      ...foeTeam().filter(p => !['Ampharos', 'Metagross'].includes(p.species)).slice(0, 4),
    ],
    script: [
      {p1: ['hurricane', 'protect'], p2: [{move: 'confuseray', target: 1}, 'protect']},
      {p1: ['hurricane', 'protect'], p2: ['protect', {move: 'ironhead', target: 1}]},
      {p1: ['hurricane', 'protect'], p2: ['protect', {move: 'ironhead', target: 1}]},
      {p1: ['hurricane', 'protect'], p2: ['protect', {move: 'ironhead', target: 1}]},
    ],
    coverage: {ability: 'tangledfeet'},
    verify(fixture, session) {
      if (!everHas(fixture, 0, 0, p => p.volatiles.includes('confusion'))) {
        return 'the control holder was never confused';
      }
      if (logHas(session, /\|-miss\|p2b: s1\|p1a: s0/)) {
        return 'a 100-accuracy move missed without Tangled Feet';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 6000, artifact: 'more_seed_analytic_tangled.json', debugEnv: 'DEBUG_SEED_ANALYTIC_TANGLED'});
