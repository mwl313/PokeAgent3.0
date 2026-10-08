// Development-only witness corpus for ported abilities that no other corpus
// exercises. Each scene drives the ability's condition at least once; the
// boundary-by-boundary differential comparison against the pinned reference is
// what actually verifies the port, so the `verify` hooks only assert that the
// precondition really happened.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, foeTeam, logHas, everHas, runTrials} = createScaffold();

const team = (...heads) => [
  ...heads,
  ...foeTeam().filter(p => !heads.some(head => head.species === p.species)),
].slice(0, 6);
const foeWith = (head) => [head, ...foeTeam().filter(p => p.species !== head.species)].slice(0, 6);
const protectTurn = (p1, p2 = ['protect', 'protect']) => ({p1, p2});
const hpAt = (fixture, side, roster) => fixture.steps.flatMap(step => step.expected.sides[side].pokemon
  .filter(p => p.roster === roster).map(p => p.hp));
const boostsAt = (fixture, side, roster) => fixture.steps.flatMap(step => step.expected.sides[side].pokemon
  .filter(p => p.roster === roster).map(p => p.boosts[4]));

const TRIALS = [
  {
    name: 'witness_pixilate_turns_a_normal_move_fairy',
    p1: () => team(setOf('Sylveon', 'Pixilate', ['Hyper Voice', 'Protect'])),
    p2: () => foeWith(offensive('Gengar', 'Cursed Body', ['Shadow Ball', 'Protect'])),
    script: [{p1: [{move: 'hypervoice'}, 'protect'], p2: [{move: 'shadowball', target: 1}, 'protect']}],
    coverage: {ability: 'pixilate'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Hyper Voice\|/)) return 'Hyper Voice never executed';
      if (logHas(session, /\|-immune\|p2a: s0/)) return 'the Ghost stayed immune to the Normal move';
      if (!logHas(session, /\|-damage\|p2a: s0\|/)) return 'the converted move never damaged the Ghost';
      return null;
    },
  },
  {
    name: 'witness_emergencyexit_switches_at_half_hp',
    p1: () => team(setOf('Golisopod', 'Emergency Exit', ['Protect', 'First Impression'])),
    p2: () => foeTeam(),
    script: [
      protectTurn([{move: 'firstimpression', target: 1}, 'protect'], [{move: 'ironhead', target: 1}, {move: 'bodyslam', target: 1}]),
      protectTurn(['protect', 'protect'], [{move: 'ironhead', target: 1}, {move: 'bodyslam', target: 1}]),
      protectTurn(['protect', 'protect'], [{move: 'ironhead', target: 1}, {move: 'bodyslam', target: 1}]),
    ],
    coverage: {ability: 'emergencyexit'},
    verify(fixture, session) {
      const hps = hpAt(fixture, 0, 0);
      const max = fixture.steps[0].expected.sides[0].pokemon[0].max_hp;
      if (!hps.some(hp => hp > 0 && hp * 2 <= max)) return 'the holder never dropped to half HP';
      if (!logHas(session, /\|switch\|p1a: /)) return 'Emergency Exit never switched the holder out';
      return null;
    },
  },
  {
    name: 'witness_torrent_boosts_a_water_move_at_low_hp',
    p1: () => team(setOf('Primarina', 'Torrent', ['Surf', 'Protect'])),
    p2: () => foeTeam(),
    script: [
      protectTurn([{move: 'surf'}, 'protect'], [{move: 'ironhead', target: 1}, {move: 'bodyslam', target: 1}]),
      protectTurn([{move: 'surf'}, 'protect'], [{move: 'ironhead', target: 1}, {move: 'bodyslam', target: 1}]),
      protectTurn([{move: 'surf'}, 'protect'], [{move: 'ironhead', target: 1}, {move: 'bodyslam', target: 1}]),
    ],
    coverage: {ability: 'torrent'},
    verify(fixture, session) {
      const max = fixture.steps[0].expected.sides[0].pokemon[0].max_hp;
      const pairs = fixture.steps.map(step => step.expected.sides[0].pokemon
        .find(p => p.roster === 0).hp);
      if (!pairs.some(hp => hp > 0 && hp * 3 <= max)) return 'the holder never reached a third of its HP';
      if (!logHas(session, /\|move\|p1a: s0\|Surf\|/)) return 'the Water move never executed';
      return null;
    },
  },
  {
    name: 'witness_speedboost_raises_speed_each_turn',
    p1: () => team(setOf('Blaziken', 'Speed Boost', ['Protect', 'Flare Blitz'])),
    p2: () => foeTeam(),
    script: [
      protectTurn(['protect', 'protect']),
      protectTurn(['protect', 'protect']),
      protectTurn(['protect', 'protect']),
    ],
    coverage: {ability: 'speedboost'},
    verify(fixture) {
      const boosts = boostsAt(fixture, 0, 0);
      if (!boosts.some((value, i) => i > 0 && value >= boosts[0] + 2)) {
        return 'the Speed Boost stages never accumulated';
      }
      return null;
    },
  },
  {
    name: 'witness_regenerator_heals_on_switch_out',
    p1: () => team(setOf('Slowbro', 'Regenerator', ['Scald', 'Protect']), setOf('Snorlax', 'Thick Fat', ['Body Slam', 'Protect'])),
    p2: () => foeTeam(),
    script: [
      protectTurn([{move: 'scald', target: 1}, 'protect'], [{move: 'bodyslam', target: 1}, {move: 'ironhead', target: 1}]),
      {p1: [{switch: 's2'}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {ability: 'regenerator'},
    verify(fixture, session) {
      const hps = hpAt(fixture, 0, 0);
      if (!hps.some((hp, i) => i > 0 && hp > hps[i - 1])) return 'the switch-out never healed the holder';
      if (!logHas(session, /\|switch\|p1a: s2/)) return 'the holder never switched out';
      return null;
    },
  },
  {
    name: 'witness_toughclaws_boosts_a_contact_move',
    p1: () => team(setOf('Barbaracle', 'Tough Claws', ['Razor Shell', 'Protect'])),
    p2: () => foeTeam(),
    script: [{p1: [{move: 'razorshell', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']}],
    coverage: {ability: 'toughclaws'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Razor Shell\|/)) return 'the contact move never executed';
      if (!logHas(session, /\|-damage\|p2a: s0\|/)) return 'the contact move never connected';
      return null;
    },
  },
  {
    name: 'witness_moody_reshuffles_each_turn',
    p1: () => team(setOf('Glalie', 'Moody', ['Protect', 'Ice Beam'])),
    p2: () => foeTeam(),
    script: [
      protectTurn(['protect', 'protect']),
      protectTurn(['protect', 'protect']),
      protectTurn(['protect', 'protect']),
    ],
    coverage: {ability: 'moody'},
    verify(fixture) {
      const boosts = fixture.steps.map(step => step.expected.sides[0].pokemon
        .find(p => p.roster === 0).boosts);
      if (!boosts.some((value, i) => i > 0 && JSON.stringify(value) !== JSON.stringify(boosts[i - 1]))) {
        return 'the Moody stages never changed';
      }
      return null;
    },
  },
  {
    name: 'witness_solarpower_drains_in_sun',
    p1: () => team(setOf('Charizard', 'Solar Power', ['Protect', 'Flamethrower'])),
    p2: () => foeWith(offensive('Torterra', 'Shell Armor', ['Sunny Day', 'Protect'])),
    script: [
      protectTurn(['protect', 'protect'], [{move: 'sunnyday'}, 'protect']),
      protectTurn(['protect', 'protect']),
      protectTurn(['protect', 'protect']),
    ],
    coverage: {ability: 'solarpower'},
    verify(fixture, session) {
      const hps = hpAt(fixture, 0, 0);
      if (!hps.some((hp, i) => i > 0 && hp < hps[i - 1])) return 'the holder never took the sun drain';
      if (!session.battle.log.some(line => line.startsWith('|-weather|SunnyDay'))) return 'the sun never rose';
      return null;
    },
  },
  {
    name: 'witness_swiftswim_moves_first_in_rain',
    p1: () => team(setOf('Basculegion', 'Swift Swim', ['Wave Crash', 'Protect'])),
    p2: () => foeWith(setOf('Politoed', 'Drizzle', ['Protect', 'Surf', 'Ice Beam'])),
    script: [protectTurn([{move: 'wavecrash', target: 1}, 'protect'], [{move: 'icebeam', target: 1}, 'protect'])],
    coverage: {ability: 'swiftswim'},
    verify(fixture, session) {
      if (!logHas(session, /\|-weather\|RainDance/)) return 'the rain never started';
      const log = session.battle.log;
      const mine = log.findIndex(line => line.startsWith('|move|p1a: s0|Wave Crash'));
      const theirs = log.findIndex(line => line.startsWith('|move|p2a: s0|Ice Beam'));
      if (mine < 0 || theirs < 0) return 'the turn never resolved both moves';
      if (mine > theirs) return 'the Swift Swim holder did not move first';
      return null;
    },
  },
  {
    name: 'witness_liquidvoice_turns_a_sound_move_water',
    p1: () => team(setOf('Primarina', 'Liquid Voice', ['Hyper Voice', 'Protect'])),
    p2: () => foeWith(offensive('Chandelure', 'Flash Fire', ['Shadow Ball', 'Protect'])),
    script: [{p1: [{move: 'hypervoice'}, 'protect'], p2: [{move: 'shadowball', target: 1}, 'protect']}],
    coverage: {ability: 'liquidvoice'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Hyper Voice\|/)) return 'the sound move never executed';
      if (!logHas(session, /\|-supereffective\|p2a: s0/)) return 'the sound move was not Water-typed';
      return null;
    },
  },
  {
    name: 'witness_solidrock_softens_a_super_effective_hit',
    p1: () => team(setOf('Rhyperior', 'Solid Rock', ['Protect', 'Rock Slide'])),
    p2: () => foeTeam(),
    script: [
      protectTurn(['protect', 'protect'], [{move: 'surf', target: 1}, {move: 'ironhead', target: 1}]),
      protectTurn(['protect', 'protect'], [{move: 'surf', target: 1}, {move: 'ironhead', target: 1}]),
    ],
    coverage: {ability: 'solidrock'},
    verify(fixture, session) {
      if (!logHas(session, /\|-supereffective\|p1a: s0/)) return 'the holder never took a super-effective hit';
      if (!logHas(session, /\|-damage\|p1a: s0\|/)) return 'the hit never landed';
      return null;
    },
  },
  {
    name: 'witness_snowcloak_hides_in_snow',
    p1: () => team(setOf('Glaceon', 'Snow Cloak', ['Protect', 'Ice Beam'])),
    p2: () => foeWith(offensive('Abomasnow', 'Snow Warning', ['Blizzard', 'Protect'])),
    script: [
      protectTurn(['protect', 'protect'], ['protect', 'protect']),
      protectTurn([{move: 'icebeam', target: 1}, 'protect'], [{move: 'blizzard'}, 'protect']),
    ],
    coverage: {ability: 'snowcloak'},
    verify(fixture, session) {
      if (!session.battle.log.some(line => line.startsWith('|-weather|Snowscape'))) return 'the snow never started';
      if (!logHas(session, /\|move\|p1a: s0\|Ice Beam\|/)) return 'the holder never attacked under snow';
      return null;
    },
  },
  {
    name: 'witness_infiltrator_attacks_through_a_decoy',
    p1: () => team(setOf('Dragapult', 'Infiltrator', ['Shadow Ball', 'Protect'])),
    p2: () => foeWith(setOf('Aggron', 'Sturdy', ['Substitute', 'Protect'])),
    script: [
      protectTurn(['protect', 'protect'], [{move: 'substitute'}, 'protect']),
      protectTurn([{move: 'bulletseed', target: 1}, 'protect'], ['protect', 'protect']),
    ],
    coverage: {ability: 'infiltrator'},
    verify(fixture, session) {
      if (!everHas(fixture, 1, 0, p => p.volatiles.includes('substitute'))) return 'the decoy never went up';
      if (!logHas(session, /\|move\|p1a: s0\|Shadow Ball\|/)) return 'the holder never attacked the decoy';
      return null;
    },
  },
  {
    name: 'witness_reckless_boosts_a_recoil_move',
    p1: () => team(setOf('Staraptor', 'Reckless', ['Brave Bird', 'Protect'])),
    p2: () => foeTeam(),
    script: [{p1: [{move: 'bravebird', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']}],
    coverage: {ability: 'reckless'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Brave Bird\|/)) return 'the recoil move never executed';
      if (!logHas(session, /\|-damage\|p1a: s0\|.*Recoil/)) return 'the recoil never applied';
      return null;
    },
  },
  {
    name: 'witness_synchronize_mirrors_a_status',
    p1: () => team(setOf('Gardevoir', 'Synchronize', ['Protect', 'Dazzling Gleam'])),
    p2: () => foeWith(offensive('Bellibolt', 'Electromorphosis', ['Toxic', 'Protect'])),
    script: [protectTurn(['protect', 'protect'], [{move: 'toxic', target: 1}, 'protect'])],
    coverage: {ability: 'synchronize'},
    verify(fixture, session) {
      if (!logHas(session, /\|-status\|p1a: s0\|tox/)) return 'the holder was never statused';
      if (!logHas(session, /\|-status\|p2a: s0\|tox/)) return 'Synchronize never mirrored the status';
      return null;
    },
  },
  {
    name: 'witness_voltabsorb_heals_from_an_electric_move',
    p1: () => team(setOf('Jolteon', 'Volt Absorb', ['Protect', 'Thunderbolt'])),
    p2: () => foeWith(offensive('Ampharos', 'Static', ['Thunderbolt', 'Protect'])),
    script: [
      protectTurn(['protect', 'protect'], [{move: 'thunderbolt', target: 1}, 'protect']),
      protectTurn(['protect', 'protect'], [{move: 'thunderbolt', target: 1}, 'protect']),
    ],
    coverage: {ability: 'voltabsorb'},
    verify(fixture, session) {
      if (!logHas(session, /\|-heal\|p1a: s0\|/)) return 'the Electric move never healed the holder';
      return null;
    },
  },
  {
    name: 'witness_libero_changes_the_holders_type',
    p1: () => team(setOf('Cinderace', 'Libero', ['Pyro Ball', 'Protect'])),
    p2: () => foeTeam(),
    script: [{p1: [{move: 'pyroball', target: 1}, 'protect'], p2: ['protect', 'protect']}],
    coverage: {ability: 'libero'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Pyro Ball\|/)) return 'the move never executed';
      if (!everHas(fixture, 0, 0, p => p.types.includes(ids.types.fire))) {
        return 'the holder never changed to the move type';
      }
      return null;
    },
  },
  {
    name: 'witness_noguard_lands_a_low_accuracy_move',
    p1: () => team(setOf('Machamp', 'No Guard', ['Stone Edge', 'Protect'])),
    p2: () => foeTeam(),
    script: [{p1: [{move: 'stoneedge', target: 1}, 'protect'], p2: ['protect', 'protect']}],
    coverage: {ability: 'noguard'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Stone Edge\|/)) return 'the low-accuracy move never executed';
      if (logHas(session, /\|-miss\|p1a: s0/)) return 'the low-accuracy move missed under No Guard';
      return null;
    },
  },
  {
    name: 'witness_shellarmor_blocks_critical_hits',
    p1: () => team(setOf('Torterra', 'Shell Armor', ['Protect', 'Seed Bomb'])),
    p2: () => foeTeam(),
    script: [
      protectTurn(['protect', 'protect'], [{move: 'ironhead', target: 1}, {move: 'bodyslam', target: 1}]),
      protectTurn(['protect', 'protect'], [{move: 'ironhead', target: 1}, {move: 'bodyslam', target: 1}]),
    ],
    coverage: {ability: 'shellarmor'},
    verify(fixture, session) {
      if (!logHas(session, /\|-damage\|p1a: s0\|/)) return 'the holder never took a hit';
      if (logHas(session, /\|-crit\|p1a: s0/)) return 'a critical hit landed on the Shell Armor holder';
      return null;
    },
  },
  {
    name: 'witness_marvelscale_halves_physical_damage_when_statused',
    p1: () => team(setOf('Milotic', 'Marvel Scale', ['Protect', 'Scald'])),
    p2: () => foeWith(offensive('Bellibolt', 'Electromorphosis', ['Toxic', 'Protect'])),
    script: [
      protectTurn(['protect', 'protect'], [{move: 'toxic', target: 1}, 'protect']),
      protectTurn(['protect', 'protect'], [{move: 'seedbomb', target: 1}, 'protect']),
      protectTurn(['protect', 'protect'], [{move: 'seedbomb', target: 1}, 'protect']),
    ],
    coverage: {ability: 'marvelscale'},
    verify(fixture, session) {
      if (!logHas(session, /\|-status\|p1a: s0\|tox/)) return 'the holder was never statused';
      if (!logHas(session, /\|-damage\|p1a: s0\|/)) return 'the physical hit never landed';
      return null;
    },
  },
  {
    name: 'witness_purepower_doubles_attack',
    p1: () => team(setOf('Medicham', 'Pure Power', ['Zen Headbutt', 'Protect'])),
    p2: () => foeTeam(),
    script: [{p1: [{move: 'zenheadbutt', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']}],
    coverage: {ability: 'purepower'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Zen Headbutt\|/)) return 'the physical move never executed';
      if (!logHas(session, /\|-damage\|p2a: s0\|/)) return 'the physical move never connected';
      return null;
    },
  },
  {
    name: 'witness_sandforce_boosts_a_rock_move_in_sand',
    p1: () => team(setOf('Excadrill', 'Sand Force', ['Rock Slide', 'Protect'])),
    p2: () => foeWith(setOf('Hippowdon', 'Sand Stream', ['Earthquake', 'Protect'])),
    script: [
      protectTurn(['protect', 'protect'], ['protect', 'protect']),
      protectTurn([{move: 'rockslide'}, 'protect'], ['protect', 'protect']),
    ],
    coverage: {ability: 'sandforce'},
    verify(fixture, session) {
      if (!logHas(session, /\|-weather\|Sandstorm/)) return 'the sandstorm never started';
      if (!logHas(session, /\|move\|p1a: s0\|Rock Slide\|/)) return 'the Rock move never executed';
      return null;
    },
  },
  {
    name: 'witness_sandveil_hides_in_sand',
    p1: () => team(setOf('Garchomp', 'Sand Veil', ['Protect', 'Earthquake'])),
    p2: () => foeWith(setOf('Hippowdon', 'Sand Stream', ['Earthquake', 'Protect'])),
    script: [
      protectTurn(['protect', 'protect'], ['protect', 'protect']),
      protectTurn([{move: 'earthquake'}, 'protect'], ['protect', 'protect']),
    ],
    coverage: {ability: 'sandveil'},
    verify(fixture, session) {
      if (!logHas(session, /\|-weather\|Sandstorm/)) return 'the sandstorm never started';
      if (!logHas(session, /\|move\|p1a: s0\|Earthquake\|/)) return 'the holder never attacked under sand';
      return null;
    },
  },
  {
    name: 'witness_sapsipper_raises_attack_from_a_grass_move',
    p1: () => team(setOf('Azumarill', 'Sap Sipper', ['Protect', 'Play Rough'])),
    p2: () => foeWith(offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect'])),
    script: [
      protectTurn([{move: 'playrough', target: 1}, 'protect'], [{move: 'seedbomb', target: 1}, 'protect']),
      protectTurn([{move: 'playrough', target: 1}, 'protect'], [{move: 'seedbomb', target: 1}, 'protect']),
    ],
    coverage: {ability: 'sapsipper'},
    verify(fixture, session) {
      if (!logHas(session, /\|-ability\|p1a: s0\|Sap Sipper\|boost/)) return 'the Grass move was not absorbed';
      if (!everHas(fixture, 0, 0, p => p.boosts[0] > 0)) return 'the Attack stage never rose';
      return null;
    },
  },
  {
    name: 'witness_strongjaw_boosts_a_biting_move',
    p1: () => team(setOf('Tyrantrum', 'Strong Jaw', ['Crunch', 'Protect'])),
    p2: () => foeTeam(),
    script: [{p1: [{move: 'crunch', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']}],
    coverage: {ability: 'strongjaw'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Crunch\|/)) return 'the biting move never executed';
      if (!logHas(session, /\|-damage\|p2a: s0\|/)) return 'the biting move never connected';
      return null;
    },
  },
  {
    name: 'witness_superluck_raises_the_critical_ratio',
    p1: () => team(setOf('Absol', 'Super Luck', ['Night Slash', 'Protect'])),
    p2: () => foeTeam(),
    script: [
      {p1: [{move: 'nightslash', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'nightslash', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'nightslash', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'nightslash', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
    ],
    coverage: {ability: 'superluck'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Night Slash\|/)) return 'the high-crit move never executed';
      if (!logHas(session, /\|-crit\|p2a: s0/)) return 'the boosted ratio never produced a critical hit';
      return null;
    },
  },
  {
    name: 'witness_swarm_boosts_a_bug_move_at_low_hp',
    p1: () => team(setOf('Scizor', 'Swarm', ['X-Scissor', 'Protect'])),
    p2: () => foeTeam(),
    script: [
      protectTurn([{move: 'xscissor', target: 1}, 'protect'], [{move: 'ironhead', target: 1}, {move: 'bodyslam', target: 1}]),
      protectTurn([{move: 'xscissor', target: 1}, 'protect'], [{move: 'ironhead', target: 1}, {move: 'bodyslam', target: 1}]),
      protectTurn([{move: 'xscissor', target: 1}, 'protect'], [{move: 'ironhead', target: 1}, {move: 'bodyslam', target: 1}]),
    ],
    coverage: {ability: 'swarm'},
    verify(fixture, session) {
      const max = fixture.steps[0].expected.sides[0].pokemon[0].max_hp;
      if (!hpAt(fixture, 0, 0).some(hp => hp > 0 && hp * 3 <= max)) return 'the holder never reached a third of its HP';
      if (!logHas(session, /\|move\|p1a: s0\|X-Scissor\|/)) return 'the Bug move never executed';
      return null;
    },
  },
  {
    name: 'witness_filter_softens_a_super_effective_hit',
    p1: () => team(setOf('Mr. Mime', 'Filter', ['Protect', 'Dazzling Gleam'])),
    p2: () => foeTeam(),
    script: [
      protectTurn(['protect', 'protect'], [{move: 'ironhead', target: 1}, {move: 'surf', target: 1}]),
      protectTurn(['protect', 'protect'], [{move: 'ironhead', target: 1}, {move: 'surf', target: 1}]),
    ],
    coverage: {ability: 'filter'},
    verify(fixture, session) {
      if (!logHas(session, /\|-supereffective\|p1a: s0/)) return 'the holder never took a super-effective hit';
      return null;
    },
  },
  {
    name: 'witness_hydration_cures_status_in_rain',
    p1: () => team(setOf('Goodra', 'Hydration', ['Protect', 'Ice Beam'])),
    p2: () => [setOf('Politoed', 'Drizzle', ['Surf', 'Protect', 'Ice Beam']), setOf('Arbok', 'Intimidate', ['Toxic', 'Protect']),
      ...foeTeam().filter(p => !['Politoed', 'Arbok'].includes(p.species))].slice(0, 6),
    script: [
      protectTurn(['protect', 'protect'], ['protect', {move: 'toxic', target: 1}]),
      protectTurn(['protect', 'protect'], ['protect', 'protect']),
    ],
    coverage: {ability: 'hydration'},
    verify(fixture, session) {
      if (!logHas(session, /\|-status\|p1a: s0\|tox/)) return 'the holder was never statused';
      if (!logHas(session, /\|-curestatus\|p1a: s0\|tox/)) return 'Hydration never cured the status';
      return null;
    },
  },
  {
    name: 'witness_liquidooze_damages_a_draining_attacker',
    p1: () => team(setOf('Swalot', 'Liquid Ooze', ['Protect', 'Sludge Bomb'])),
    p2: () => foeWith(offensive('Abomasnow', 'Snow Warning', ['Giga Drain', 'Protect'])),
    script: [
      protectTurn(['protect', 'protect'], [{move: 'gigadrain', target: 1}, 'protect']),
      protectTurn(['protect', 'protect'], [{move: 'gigadrain', target: 1}, 'protect']),
    ],
    coverage: {ability: 'liquidooze'},
    verify(fixture, session) {
      if (!logHas(session, /\|-damage\|p1a: s0\|/)) return 'the drain move never hit the holder';
      if (!logHas(session, /\|-damage\|p2a: s0\|/)) return 'the draining attacker was never hurt';
      return null;
    },
  },
  {
    name: 'witness_plus_boosts_special_attack_with_an_ally',
    p1: () => team(setOf('Dedenne', 'Plus', ['Thunderbolt', 'Protect']), setOf('Toxtricity-Low-Key', 'Minus', ['Thunderbolt', 'Protect'])),
    p2: () => foeTeam(),
    script: [{p1: [{move: 'thunderbolt', target: 1}, {move: 'thunderbolt', target: 1}], p2: ['protect', 'protect']}],
    coverage: {ability: 'plus'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Thunderbolt\|/)) return 'the holder never attacked';
      if (!logHas(session, /\|move\|p1b: s1\|Thunderbolt\|/)) return 'the Minus ally never attacked';
      return null;
    },
  },
  {
    name: 'witness_minus_boosts_special_attack_with_an_ally',
    p1: () => team(setOf('Toxtricity-Low-Key', 'Minus', ['Thunderbolt', 'Protect']), setOf('Dedenne', 'Plus', ['Thunderbolt', 'Protect'])),
    p2: () => foeTeam(),
    script: [{p1: [{move: 'thunderbolt', target: 1}, {move: 'thunderbolt', target: 1}], p2: ['protect', 'protect']}],
    coverage: {ability: 'minus'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Thunderbolt\|/)) return 'the holder never attacked';
      if (!logHas(session, /\|move\|p1b: s1\|Thunderbolt\|/)) return 'the Plus ally never attacked';
      return null;
    },
  },
  {
    name: 'witness_motordrive_raises_speed_from_an_electric_move',
    p1: () => team(setOf('Emolga', 'Motor Drive', ['Protect', 'Thunderbolt'])),
    p2: () => foeWith(offensive('Ampharos', 'Static', ['Thunderbolt', 'Protect'])),
    script: [
      protectTurn([{move: 'thunderbolt', target: 1}, 'protect'], [{move: 'thunderbolt', target: 1}, 'protect']),
      protectTurn([{move: 'thunderbolt', target: 1}, 'protect'], [{move: 'thunderbolt', target: 1}, 'protect']),
    ],
    coverage: {ability: 'motordrive'},
    verify(fixture, session) {
      if (!logHas(session, /\|-ability\|p1a: s0\|Motor Drive\|/)) return 'the Electric move was not absorbed';
      if (!everHas(fixture, 0, 0, p => p.boosts[4] > 0)) return 'the Speed stage never rose';
      return null;
    },
  },
  {
    name: 'witness_owntempo_refuses_confusion',
    p1: () => team(setOf('Mudsdale', 'Own Tempo', ['Protect', 'Earthquake'])),
    p2: () => foeWith(offensive('Ampharos', 'Static', ['Confuse Ray', 'Protect'])),
    script: [
      protectTurn(['protect', 'protect'], [{move: 'confuseray', target: 1}, 'protect']),
      protectTurn(['protect', 'protect'], ['protect', 'protect']),
    ],
    coverage: {ability: 'owntempo'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Confuse Ray\|/)) return 'the confusion move never executed';
      if (everHas(fixture, 0, 0, p => p.volatiles.includes('confusion'))) {
        return 'the Own Tempo holder got confused';
      }
      return null;
    },
  },
  {
    name: 'witness_quickfeet_boosts_speed_when_statused',
    p1: () => team(setOf('Jolteon', 'Quick Feet', ['Facade', 'Protect'])),
    p2: () => foeWith(offensive('Bellibolt', 'Electromorphosis', ['Toxic', 'Protect'])),
    script: [
      protectTurn(['protect', 'protect'], [{move: 'toxic', target: 1}, 'protect']),
      protectTurn([{move: 'facade', target: 1}, 'protect'], ['protect', 'protect']),
    ],
    coverage: {ability: 'quickfeet'},
    verify(fixture, session) {
      if (!logHas(session, /\|-status\|p1a: s0\|tox/)) return 'the holder was never statused';
      if (!logHas(session, /\|move\|p1a: s0\|Facade\|/)) return 'the holder never attacked while statused';
      return null;
    },
  },
  {
    name: 'witness_slushrush_moves_first_in_snow',
    p1: () => team(setOf('Beartic', 'Slush Rush', ['Icicle Crash', 'Protect'])),
    p2: () => foeWith(offensive('Abomasnow', 'Snow Warning', ['Blizzard', 'Protect'])),
    script: [protectTurn([{move: 'iciclecrash', target: 1}, 'protect'], [{move: 'blizzard'}, 'protect'])],
    coverage: {ability: 'slushrush'},
    verify(fixture, session) {
      if (!session.battle.log.some(line => line.startsWith('|-weather|Snowscape'))) return 'the snow never started';
      const log = session.battle.log;
      const mine = log.findIndex(line => line.startsWith('|move|p1a: s0|Icicle Crash'));
      const theirs = log.findIndex(line => line.startsWith('|move|p2a: s0|Blizzard'));
      if (mine < 0 || theirs < 0) return 'the turn never resolved both moves';
      if (mine > theirs) return 'the Slush Rush holder did not move first';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 8500, artifact: 'more_witness_tail.json', debugEnv: 'DEBUG_WITNESS'});
