// Development-only corpus for the entry-hazard family:
// - Spikes layers damage grounded entrants (and skip airborne ones),
// - Stealth Rock scales with Rock effectiveness,
// - Sticky Web drops a grounded entrant's Speed,
// - Toxic Spikes poison by layers, are absorbed by Poison types and ignored
//   by Steel types, and
// - Magic Bounce reflects a hazard back at its user.
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
// Pad a team to six distinct species from the shared pool.
const team = (...heads) => [
  ...heads,
  ...POOL.filter(([species]) => !heads.some(head => head.species === species))
    .map(([species, ability, moves]) => offensive(species, ability, moves)),
].slice(0, 6);

const TRIALS = [
  {
    name: 'hazards_spikes_layers_damage_grounded',
    p1: () => team(setOf('Forretress', 'Sturdy', ['Spikes', 'Protect', 'Body Press'])),
    p2: () => team(offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']), offensive('Milotic', 'Competitive', ['Surf', 'Protect']), offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect'])),
    script: [
      {p1: ['spikes', 'protect'], p2: ['protect', 'protect']},
      {p1: ['spikes', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: [{switch: 's2'}, 'protect']},
    ],
    coverage: {move: 'spikes'},
    verify(fixture, session) {
      const spikes = ids.conditions.spikes;
      const layerTwo = fixture.steps.some(step => step.expected.sides[1].conditions
        .some(([id, layers]) => id === spikes && layers === 2));
      if (!layerTwo) return 'two Spikes layers were never recorded';
      // The switched-in Snorlax takes maxhp / 6 (two layers).
      const snorlax = fixture.steps.flatMap(step => step.expected.sides[1].pokemon)
        .find(p => p.roster === 2);
      if (!snorlax) return 'the switch-in target is missing';
      const expected = Math.floor(snorlax.max_hp / 6);
      if (!everHas(fixture, 1, 2, p => p.max_hp - p.hp === expected)) {
        return `the two-layer Spikes damage was not ${expected}`;
      }
      return null;
    },
  },
  {
    name: 'hazards_stealthrock_scales_with_effectiveness',
    p1: () => team(setOf('Forretress', 'Sturdy', ['Stealth Rock', 'Protect', 'Body Press'])),
    p2: () => team(offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']), offensive('Milotic', 'Competitive', ['Surf', 'Protect']), offensive('Vivillon', 'Compound Eyes', ['Hurricane', 'Protect'])),
    script: [
      {p1: ['stealthrock', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: [{switch: 's2'}, 'protect']},
    ],
    coverage: {move: 'stealthrock'},
    verify(fixture, session) {
      const rock = ids.conditions.stealthrock;
      if (!fixture.steps.some(step => step.expected.sides[1].conditions.some(([id]) => id === rock))) {
        return 'Stealth Rock was never recorded on the foe side';
      }
      // Vivillon is Bug/Flying: a 4x Rock weakness halves its maximum HP.
      const vivillon = fixture.steps.flatMap(step => step.expected.sides[1].pokemon)
        .find(p => p.roster === 2);
      if (!vivillon) return 'the switch-in target is missing';
      const expected = Math.floor(vivillon.max_hp / 2);
      if (!everHas(fixture, 1, 2, p => p.max_hp - p.hp === expected)) {
        return `the 4x Stealth Rock damage was not ${expected}`;
      }
      return null;
    },
  },
  {
    name: 'hazards_stickyweb_grounded_only',
    p1: () => team(setOf('Araquanid', 'Water Absorb', ['Sticky Web', 'Protect', 'Leech Life'])),
    p2: () => team(offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']), offensive('Milotic', 'Competitive', ['Surf', 'Protect']), offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']), offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect'])),
    script: [
      {p1: ['stickyweb', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: [{switch: 's2'}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{switch: 's3'}, 'protect']},
    ],
    coverage: {move: 'stickyweb'},
    verify(fixture, session) {
      const web = ids.conditions.stickyweb;
      if (!fixture.steps.some(step => step.expected.sides[1].conditions.some(([id]) => id === web))) {
        return 'Sticky Web was never recorded on the foe side';
      }
      if (!everHas(fixture, 1, 2, p => p.boosts[4] === -1)) {
        return 'the grounded entrant never lost a Speed stage';
      }
      // Chimecho's Levitate keeps it airborne, so the web must not touch it.
      if (everHas(fixture, 1, 3, p => p.boosts[4] !== 0)) {
        return 'the airborne entrant was caught by Sticky Web';
      }
      return null;
    },
  },
  {
    name: 'hazards_toxicspikes_poison_and_air_immunity',
    p1: () => team(setOf('Forretress', 'Sturdy', ['Toxic Spikes', 'Protect', 'Body Press'])),
    p2: () => team(offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']), offensive('Milotic', 'Competitive', ['Surf', 'Protect']), offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']), offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']), offensive('Arbok', 'Intimidate', ['Crunch', 'Protect'])),
    script: [
      {p1: ['toxicspikes', 'protect'], p2: ['protect', 'protect']},
      {p1: ['toxicspikes', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: [{switch: 's2'}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{switch: 's3'}, 'protect']},
    ],
    coverage: {move: 'toxicspikes'},
    verify(fixture, session) {
      if (!everHas(fixture, 1, 2, p => p.status === ids.conditions.tox)) {
        return 'the two-layer Toxic Spikes never badly poisoned the entrant';
      }
      if (everHas(fixture, 1, 3, p => p.status !== 0)) {
        return 'the airborne entrant was poisoned by Toxic Spikes';
      }
      return null;
    },
  },
  {
    name: 'hazards_toxicspikes_poison_absorbs',
    p1: () => team(setOf('Forretress', 'Sturdy', ['Toxic Spikes', 'Protect', 'Body Press'])),
    p2: () => team(offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']), offensive('Milotic', 'Competitive', ['Surf', 'Protect']), offensive('Arbok', 'Intimidate', ['Crunch', 'Protect']), offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect'])),
    script: [
      {p1: ['toxicspikes', 'protect'], p2: ['protect', 'protect']},
      {p1: ['toxicspikes', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: [{switch: 's2'}, 'protect']},
    ],
    coverage: {move: 'toxicspikes'},
    verify(fixture, session) {
      if (!everHas(fixture, 1, 2, p => p.active_slot !== null)) {
        return 'the Poison absorber never entered';
      }
      if (everHas(fixture, 1, 2, p => p.status !== 0)) {
        return 'the Poison absorber was poisoned by its own hazard';
      }
      const spikes = ids.conditions.toxicspikes;
      if (fixture.steps.at(-1).expected.sides[1].conditions.some(([id]) => id === spikes)) {
        return 'the Poison entrant never removed the Toxic Spikes';
      }
      return null;
    },
  },
  {
    name: 'hazards_magicbounce_reflects_spikes',
    p1: () => team(setOf('Forretress', 'Sturdy', ['Spikes', 'Protect', 'Body Press'])),
    p2: () => team(setOf('Hatterene', 'Magic Bounce', ['Dazzling Gleam', 'Protect', 'Psychic'])),
    script: [{p1: ['spikes', 'protect'], p2: ['dazzlinggleam', 'protect']}],
    coverage: {move: 'spikes'},
    verify(fixture, session) {
      if (!logHas(session, /\[from\] ability: Magic Bounce/)) {
        return 'the hazard was never reflected';
      }
      const spikes = ids.conditions.spikes;
      if (!fixture.steps.some(step => step.expected.sides[0].conditions.some(([id]) => id === spikes))) {
        return 'the reflected Spikes never landed on the user side';
      }
      if (fixture.steps.some(step => step.expected.sides[1].conditions.some(([id]) => id === spikes))) {
        return 'the Spikes still landed on the Magic Bounce side';
      }
      return null;
    },
  },
  {
    name: 'hazards_defog_clears_both_sides',
    p1: () => team(setOf('Corviknight', 'Pressure', ['Defog', 'Protect', 'Brave Bird']), setOf('Forretress', 'Sturdy', ['Spikes', 'Protect', 'Body Press'])),
    p2: () => team(setOf('Aggron', 'Sturdy', ['Stealth Rock', 'Protect', 'Body Slam']), setOf('Chimecho', 'Levitate', ['Reflect', 'Dazzling Gleam', 'Psychic'])),
    script: [
      {p1: ['protect', 'protect'], p2: ['stealthrock', 'reflect']},
      {p1: ['protect', 'spikes'], p2: ['protect', 'protect']},
      {p1: [{move: 'defog', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'defog'},
    verify(fixture, session) {
      const spikes = ids.conditions.spikes;
      const rock = ids.conditions.stealthrock;
      const reflect = ids.conditions.reflect;
      const hadAll = fixture.steps.some(step =>
        step.expected.sides[0].conditions.some(([id]) => id === rock)
        && step.expected.sides[1].conditions.some(([id]) => id === spikes)
        && step.expected.sides[1].conditions.some(([id]) => id === reflect));
      if (!hadAll) return 'the setup turn never recorded all three effects';
      const last = fixture.steps.at(-1).expected;
      if (last.sides[0].conditions.some(([id]) => id === rock)) return 'Stealth Rock survived Defog';
      if (last.sides[1].conditions.some(([id]) => id === spikes)) return 'Spikes survived Defog';
      if (last.sides[1].conditions.some(([id]) => id === reflect)) return 'Reflect survived Defog';
      if (!everHas(fixture, 1, 0, p => p.boosts[6] === -1)) {
        return 'the Defog target never lost an evasion stage';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 14000, artifact: 'more_hazards.json', debugEnv: 'DEBUG_HAZARDS'});
