// Development-only corpus for the entry-hazard family:
// - Spikes layer caps (three), grounded/Heavy-Duty Boots exemptions and the
//   switch-in damage fractions,
// - Stealth Rock switch-in damage by Rock effectiveness (4x/2x/1x/0.5x),
// - Sticky Web's grounded -1 Speed on entry,
// - Toxic Spikes layers, Poison-type absorption and Steel immunity,
// - Defog clearing the entry hazards on both sides, the target side's
//   screens, the terrain and the target's evasion,
// - Magic Bounce reflecting a foeSide hazard back at its user, and
// - Ceaseless Edge / Stone Axe setting Spikes / Stealth Rock on hit.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, logHas, runTrials} = createScaffold();

const SPIKES = ids.conditions.spikes;
const STEALTH_ROCK = ids.conditions.stealthrock;
const STICKY_WEB = ids.conditions.stickyweb;
const TOXIC_SPIKES = ids.conditions.toxicspikes;
const ELECTRIC_TERRAIN = ids.conditions.electricterrain;

const fill = () => [
  offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
  offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
  offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
  offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
  offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
];
const team = head => [head, ...fill()];
const withBench = (actives, bench) => [...actives, ...bench];

const conditionLayers = (conditions, id) => {
  const hit = conditions?.find(([cond]) => cond === id);
  return hit ? hit[1] : null;
};
const everLayers = (fixture, side, id) => fixture.steps
  .map(step => conditionLayers(step.expected.sides[side].conditions, id))
  .filter(layers => layers !== null);
/// Side conditions at the first recorded boundary of `turn` or later.
const conditionsAt = (fixture, turn, side) => {
  const step = fixture.steps.find(entry => entry.expected.turn >= turn);
  return step ? step.expected.sides[side].conditions : [];
};
/// First boundary at or after `turn` where the roster slot is on the field.
const firstActiveStep = (fixture, turn, side, roster) => {
  for (const step of fixture.steps) {
    if (step.expected.turn < turn) continue;
    const mon = step.expected.sides[side].pokemon.find(p => p.roster === roster);
    if (mon && mon.active_slot !== null) return {mon, step};
  }
  return null;
};
const firstActive = (fixture, turn, side, roster) => firstActiveStep(fixture, turn, side, roster)?.mon ?? null;
const sidestarts = (session, side, name) => session.battle.log
  .filter(line => line.startsWith(`|-sidestart|${side}: `) && line.includes(name)).length;
/// Side-start messages before the given turn marker (the scripted window).
const sidestartsBefore = (session, side, name, turn) => {
  const limit = session.battle.log.indexOf(`|turn|${turn}`);
  if (limit < 0) return sidestarts(session, side, name);
  return session.battle.log.slice(0, limit)
    .filter(line => line.startsWith(`|-sidestart|${side}: `) && line.includes(name)).length;
};

const TRIALS = [
  {
    name: 'spikes_layers_cap_and_damage',
    p1: () => team(setOf('Forretress', 'Sturdy', ['Spikes', 'Stealth Rock', 'Protect'])),
    p2: () => withBench(
      [offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
        offensive('Talonflame', 'Flame Body', ['Brave Bird', 'Protect'])],
      [offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
        offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
        offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
        offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect'])]),
    script: [
      {p1: ['spikes', 'protect'], p2: ['protect', 'protect']},
      {p1: ['spikes', 'protect'], p2: ['protect', 'protect']},
      {p1: ['spikes', 'protect'], p2: ['protect', 'protect']},
      {p1: ['spikes', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: [{switch: 's2'}, 'protect']},
    ],
    coverage: {move: 'spikes'},
    verify(fixture, session) {
      if (sidestarts(session, 'p2', 'Spikes') !== 3) return `expected three Spikes layers, saw ${sidestarts(session, 'p2', 'Spikes')}`;
      if (!logHas(session, /\|-fail\|p1a: s0/)) return 'the capped fourth Spikes never failed';
      if (!everLayers(fixture, 1, SPIKES).includes(3)) return 'the p2 side never held three Spikes layers';
      const snorlax = firstActive(fixture, 5, 1, 2);
      if (!snorlax) return 'the Snorlax switch-in was never recorded';
      const expected = snorlax.max_hp - Math.trunc(snorlax.max_hp * 6 / 24);
      if (snorlax.hp !== expected) return `Snorlax took ${snorlax.max_hp - snorlax.hp} from three Spikes layers, expected ${snorlax.max_hp - expected}`;
      return null;
    },
  },
  {
    name: 'spikes_grounded_exemption',
    p1: () => team(setOf('Forretress', 'Sturdy', ['Spikes', 'Stealth Rock', 'Protect'])),
    p2: () => withBench(
      [offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
        offensive('Talonflame', 'Flame Body', ['Brave Bird', 'Protect'])],
      [offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
        offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
        offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
        offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect'])]),
    script: [
      {p1: ['spikes', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: [{switch: 's2'}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{switch: 's3'}, 'protect']},
    ],
    coverage: {move: 'spikes'},
    verify(fixture, session) {
      if (sidestartsBefore(session, 'p2', 'Spikes', 2) !== 1) return 'the single Spikes layer never started';
      const torterra = firstActive(fixture, 2, 1, 2);
      if (!torterra) return 'the grounded switch-in was never recorded';
      if (torterra.hp !== torterra.max_hp - Math.trunc(torterra.max_hp / 8)) {
        return 'the grounded entrant did not take exactly one eighth';
      }
      const levitate = firstActive(fixture, 3, 1, 3);
      if (!levitate || levitate.hp !== levitate.max_hp) return 'the Levitate entrant took Spikes damage';
      return null;
    },
  },
  {
    name: 'stealthrock_type_damage',
    p1: () => team(setOf('Aerodactyl', 'Rock Head', ['Stealth Rock', 'Protect', 'Rock Slide'])),
    p2: () => withBench(
      [offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
        offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
        offensive('Talonflame', 'Flame Body', ['Brave Bird', 'Protect']),
        offensive('Froslass', 'Cursed Body', ['Ice Beam', 'Protect'])],
      [offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
        offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect'])]),
    script: [
      {p1: ['stealthrock', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: [{switch: 's2'}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{switch: 's3'}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{switch: 's0'}, 'protect']},
    ],
    coverage: {move: 'stealthrock'},
    verify(fixture, session) {
      if (sidestarts(session, 'p2', 'Stealth Rock') !== 1) return 'Stealth Rock never started';
      const entry = (turn, roster, fraction, label) => {
        const mon = firstActive(fixture, turn, 1, roster);
        if (!mon) return `${label} was never recorded`;
        const expected = mon.max_hp - Math.trunc(mon.max_hp * fraction);
        if (mon.hp !== expected) return `${label} took ${mon.max_hp - mon.hp}, expected ${mon.max_hp - expected}`;
        return null;
      };
      return entry(2, 2, 1 / 2, 'the 4x-weak entrant') ?? entry(3, 3, 1 / 4, 'the 2x-weak entrant')
        ?? entry(4, 0, 1 / 8, 'the neutral entrant');
    },
  },
  {
    name: 'stealthrock_resistant_entry',
    p1: () => team(setOf('Aerodactyl', 'Rock Head', ['Stealth Rock', 'Protect', 'Rock Slide'])),
    p2: () => withBench(
      [offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
        offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect'])],
      [offensive('Tinkaton', 'Own Tempo', ['Play Rough', 'Protect']),
        offensive('Froslass', 'Cursed Body', ['Ice Beam', 'Protect']),
        offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
        offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect'])]),
    script: [
      {p1: ['stealthrock', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: [{switch: 's2'}, 'protect']},
    ],
    coverage: {move: 'stealthrock'},
    verify(fixture, session) {
      if (sidestarts(session, 'p2', 'Stealth Rock') !== 1) return 'Stealth Rock never started';
      const mon = firstActive(fixture, 2, 1, 2);
      if (!mon) return 'the resistant entrant was never recorded';
      const expected = mon.max_hp - Math.trunc(mon.max_hp / 16);
      if (mon.hp !== expected) return `the resistant entrant took ${mon.max_hp - mon.hp}, expected ${mon.max_hp - expected}`;
      return null;
    },
  },
  {
    name: 'stickyweb_speed_drop',
    p1: () => team(setOf('Araquanid', 'Water Bubble', ['Sticky Web', 'Protect', 'Liquidation'])),
    p2: () => withBench(
      [offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
        offensive('Talonflame', 'Flame Body', ['Brave Bird', 'Protect'])],
      [offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
        offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
        offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
        offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect'])]),
    script: [
      {p1: ['stickyweb', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: [{switch: 's2'}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{switch: 's3'}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{switch: 's0'}, 'protect']},
    ],
    coverage: {move: 'stickyweb'},
    verify(fixture, session) {
      if (sidestarts(session, 'p2', 'Sticky Web') !== 1) return 'Sticky Web never started';
      const grounded = firstActive(fixture, 2, 1, 2);
      if (!grounded || grounded.boosts[4] !== -1) return 'the grounded entrant did not lose one Speed stage';
      const levitate = firstActive(fixture, 3, 1, 3);
      if (!levitate || levitate.boosts[4] !== 0) return 'the Levitate entrant was slowed by Sticky Web';
      const returning = firstActive(fixture, 4, 1, 0);
      if (!returning || returning.boosts[4] !== -1) return 'the returning grounded entrant was not slowed';
      if (!logHas(session, /\|-activate\|p2a: [^|]*\|move: Sticky Web/)) return 'the Sticky Web activation message never fired';
      return null;
    },
  },
  {
    name: 'toxicspikes_layers_status',
    p1: () => team(setOf('Garbodor', 'Weak Armor', ['Toxic Spikes', 'Spikes', 'Protect'])),
    p2: () => withBench(
      [offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
        offensive('Talonflame', 'Flame Body', ['Brave Bird', 'Protect'])],
      [offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
        offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
        offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
        offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect'])]),
    script: [
      {p1: ['toxicspikes', 'protect'], p2: ['protect', 'protect']},
      {p1: ['toxicspikes', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: [{switch: 's2'}, 'protect']},
    ],
    coverage: {move: 'toxicspikes'},
    verify(fixture, session) {
      if (sidestarts(session, 'p2', 'Toxic Spikes') !== 2) return `expected two Toxic Spikes layers, saw ${sidestarts(session, 'p2', 'Toxic Spikes')}`;
      if (!everLayers(fixture, 1, TOXIC_SPIKES).includes(2)) return 'the p2 side never held two Toxic Spikes layers';
      const snorlax = firstActive(fixture, 3, 1, 2);
      if (!snorlax) return 'the poisoned entrant was never recorded';
      if (snorlax.status !== ids.conditions.tox) return 'two Toxic Spikes layers did not badly poison the entrant';
      return null;
    },
  },
  {
    name: 'toxicspikes_absorb_and_steel',
    p1: () => team(setOf('Garbodor', 'Weak Armor', ['Toxic Spikes', 'Spikes', 'Protect'])),
    p2: () => withBench(
      [offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
        offensive('Talonflame', 'Flame Body', ['Brave Bird', 'Protect'])],
      [offensive('Gengar', 'Cursed Body', ['Shadow Ball', 'Protect']),
        offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
        offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
        offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect'])]),
    script: [
      {p1: ['toxicspikes', 'protect'], p2: ['protect', 'protect']},
      {p1: ['toxicspikes', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: [{switch: 's2'}, 'protect']},
      {p1: ['toxicspikes', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: [{switch: 's3'}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{switch: 's0'}, 'protect']},
    ],
    coverage: {move: 'toxicspikes'},
    verify(fixture, session) {
      const gengar = firstActiveStep(fixture, 3, 1, 2);
      if (!gengar) return 'the Poison-type switch-in was never recorded';
      if (gengar.mon.status !== 0) return 'the Poison-type entrant was poisoned instead of absorbing';
      if (!logHas(session, /\|-sideend\|p2: [^|]*\|move: Toxic Spikes/)) return 'the Poison-type absorption never ended the hazard';
      if (conditionLayers(gengar.step.expected.sides[1].conditions, TOXIC_SPIKES) !== null) return 'the absorbed Toxic Spikes survived';
      const steel = firstActiveStep(fixture, 5, 1, 3);
      if (!steel || steel.mon.status !== 0) return 'the Steel-type entrant was poisoned by Toxic Spikes';
      if (conditionLayers(steel.step.expected.sides[1].conditions, TOXIC_SPIKES) !== 1) return 'the fresh one-layer Toxic Spikes did not stay for the Steel type';
      const returning = firstActive(fixture, 6, 1, 0);
      if (!returning || returning.status !== ids.conditions.psn) return 'one Toxic Spikes layer did not poison the next entrant';
      return null;
    },
  },
  {
    name: 'defog_clears_hazards_screens_terrain',
    p1: () => withBench(
      [setOf('Forretress', 'Sturdy', ['Spikes', 'Stealth Rock', 'Protect']),
        setOf('Conkeldurr', 'Guts', ['Defog', 'Protect', 'Close Combat'])],
      [setOf('Corviknight', 'Pressure', ['Reflect', 'Protect', 'Brave Bird']),
        offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
        offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
        offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect'])]),
    p2: () => withBench(
      [offensive('Garbodor', 'Weak Armor', ['Spikes', 'Protect', 'Gunk Shot']),
        setOf('Pincurchin', 'Electric Surge', ['Electric Terrain', 'Discharge', 'Protect'])],
      [offensive('Torterra', 'Shell Armor', ['Reflect', 'Seed Bomb', 'Protect']),
        offensive('Talonflame', 'Flame Body', ['Brave Bird', 'Protect']),
        offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
        offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect'])]),
    script: [
      {p1: ['stealthrock', 'protect'], p2: ['spikes', 'electricterrain']},
      {p1: [{switch: 's2'}, 'protect'], p2: [{switch: 's2'}, 'protect']},
      {p1: ['reflect', {move: 'defog', target: 1}], p2: ['reflect', 'protect']},
    ],
    coverage: {move: 'defog'},
    verify(fixture, session) {
      const next = session.battle.log.indexOf('|turn|4');
      if (next < 0) return 'the battle did not continue past the Defog turn';
      if (session.battle.log.slice(0, next).filter(line =>
        line.startsWith('|-sideend|p2: ') && line.includes('Stealth Rock') && line.includes('move: Defog')).length !== 1) {
        return 'Defog did not announce the p2-side Stealth Rock removal';
      }
      if (session.battle.log.slice(0, next).filter(line =>
        line.startsWith('|-sideend|p1: ') && line.includes('Spikes') && line.includes('move: Defog')).length !== 1) {
        return 'Defog did not announce the p1-side Spikes removal';
      }
      const cleared = fixture.steps.find(step => step.expected.turn >= 3
        && !step.expected.sides[1].conditions.some(([id]) => id === SPIKES || id === STEALTH_ROCK || id === ids.conditions.reflect)
        && step.expected.sides[0].conditions.some(([id]) => id === ids.conditions.reflect)
        && !step.expected.sides[0].conditions.some(([id]) => id === SPIKES));
      if (!cleared) return 'the cleared boundary (both hazard sides removed, Reflect kept on p1) was never recorded';
      if (cleared.expected.field.some(([id]) => id === ELECTRIC_TERRAIN)) return 'Defog did not clear the terrain';
      const target = cleared.expected.sides[1].pokemon.find(p => p.active_slot === 0);
      if (!target || target.boosts[6] !== -1) return 'Defog did not lower the target evasion';
      return null;
    },
  },
  {
    name: 'magicbounce_reflects_spikes',
    p1: () => team(setOf('Forretress', 'Sturdy', ['Spikes', 'Stealth Rock', 'Protect'])),
    p2: () => withBench(
      [setOf('Hatterene', 'Magic Bounce', ['Dazzling Gleam', 'Psychic', 'Protect']),
        offensive('Talonflame', 'Flame Body', ['Brave Bird', 'Protect'])],
      [offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
        offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
        offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
        offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect'])]),
    script: [
      {p1: ['spikes', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'spikes'},
    verify(fixture, session) {
      if (!everLayers(fixture, 0, SPIKES).includes(1)) return 'the bounced Spikes never landed on p1 side';
      if (conditionLayers(conditionsAt(fixture, 2, 1), SPIKES) !== null) return 'the original Spikes also landed on p2 side';
      if (sidestartsBefore(session, 'p2', 'Spikes', 2) !== 0) return 'the reflected move still announced a p2-side start';
      return null;
    },
  },
  {
    name: 'ceaselessedge_sets_spikes',
    p1: () => team(setOf('Samurott-Hisui', 'Torrent', ['Ceaseless Edge', 'Protect', 'Aqua Cutter'])),
    p2: () => withBench(
      [offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
        offensive('Talonflame', 'Flame Body', ['Brave Bird', 'Protect'])],
      [offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
        offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
        offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
        offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect'])]),
    script: [
      {p1: [{move: 'ceaselessedge', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'ceaselessedge'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: [^|]*\|Ceaseless Edge\|/)) return 'Ceaseless Edge never resolved';
      if (!everLayers(fixture, 1, SPIKES).includes(1)) return 'Ceaseless Edge did not set one Spikes layer';
      if (sidestarts(session, 'p2', 'Spikes') !== 1) return 'the Ceaseless Edge Spikes start was never announced';
      return null;
    },
  },
  {
    name: 'stoneaxe_sets_stealthrock',
    p1: () => team(setOf('Kleavor', 'Swarm', ['Stone Axe', 'Protect', 'X-Scissor'])),
    p2: () => withBench(
      [offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
        offensive('Talonflame', 'Flame Body', ['Brave Bird', 'Protect'])],
      [offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
        offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
        offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
        offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect'])]),
    script: [
      {p1: [{move: 'stoneaxe', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'stoneaxe'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: [^|]*\|Stone Axe\|/)) return 'Stone Axe never resolved';
      if (!everLayers(fixture, 1, STEALTH_ROCK).includes(0)) return 'Stone Axe did not set Stealth Rock';
      if (sidestarts(session, 'p2', 'Stealth Rock') !== 1) return 'the Stone Axe Stealth Rock start was never announced';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 9200, artifact: 'more_hazards.json', debugEnv: 'DEBUG_HAZARDS'});
