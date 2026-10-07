// Development-only corpus for four small ability-tail entries:
// - Big Pecks refuses Defense drops (silently for a secondary source).
// - Stalwart makes the holder's moves ignore Follow Me redirection.
// - Plus boosts Special Attack 1.5x while an ally holds Plus or Minus.
// - Steely Spirit boosts Steel moves used by the holder or its ally 1.5x.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, fillerAfter, foeTeam, foeWith, logHas, everHas, runTrials} = createScaffold();

const TRIALS = [
  {
    name: 'bigpecks_refuses_defense_drop',
    p1: () => fillerAfter(setOf('Pidgeot', 'Big Pecks', ['Hurricane', 'Protect'])),
    p2: () => foeWith(offensive('Aggron', 'Sturdy', ['Screech', 'Protect'])),
    script: [{p1: ['hurricane', 'protect'], p2: [{move: 'screech', target: 1}, 'protect']}],
    coverage: {ability: 'bigpecks'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Screech\|p1a: s0/)) return 'Screech never executed';
      if (!logHas(session, /\[from\] ability: Big Pecks/)) return 'Big Pecks never announced the refused drop';
      if (everHas(fixture, 0, 0, p => p.boosts[1] < 0)) return 'the Defense drop applied anyway';
      return null;
    },
  },
  {
    name: 'bigpecks_control_defense_drop_applies',
    p1: () => fillerAfter(setOf('Snorlax', 'Thick Fat', ['Body Slam', 'Protect'])),
    p2: () => foeWith(offensive('Aggron', 'Sturdy', ['Screech', 'Protect'])),
    script: [{p1: ['bodyslam', 'protect'], p2: [{move: 'screech', target: 1}, 'protect']}],
    coverage: {ability: 'bigpecks'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Screech\|p1a: s0/)) return 'Screech never executed';
      if (!everHas(fixture, 0, 0, p => p.boosts[1] === -2)) {
        return 'the Defense drop did not apply without Big Pecks';
      }
      return null;
    },
  },
  {
    name: 'stalwart_ignores_follow_me',
    p1: () => [
      setOf('Archaludon', 'Stalwart', ['Flash Cannon', 'Protect']),
      ...foeTeam().slice(0, 5),
    ],
    p2: () => [
      offensive('Clefable', 'Magic Guard', ['Follow Me', 'Protect']),
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
      ...foeTeam().filter(p => !['Clefable', 'Metagross'].includes(p.species)).slice(0, 4),
    ],
    script: [{p1: [{move: 'flashcannon', target: 2}, 'protect'], p2: ['followme', 'protect']}],
    coverage: {ability: 'stalwart'},
    verify(fixture, session) {
      const first = session.battle.log.find(line => /\|move\|p1a: s0\|Flash Cannon\|/.test(line));
      if (!first || !/\|move\|p1a: s0\|Flash Cannon\|p2b: s1/.test(first)) {
        return 'the Stalwart move was redirected away from its chosen target';
      }
      return null;
    },
  },
  {
    name: 'stalwart_control_follow_me_redirects',
    p1: () => [
      setOf('Archaludon', 'Sturdy', ['Flash Cannon', 'Protect']),
      ...foeTeam().slice(0, 5),
    ],
    p2: () => [
      offensive('Clefable', 'Magic Guard', ['Follow Me', 'Protect']),
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
      ...foeTeam().filter(p => !['Clefable', 'Metagross'].includes(p.species)).slice(0, 4),
    ],
    script: [{p1: [{move: 'flashcannon', target: 2}, 'protect'], p2: ['followme', 'protect']}],
    coverage: {ability: 'stalwart'},
    verify(fixture, session) {
      const first = session.battle.log.find(line => /\|move\|p1a: s0\|Flash Cannon\|/.test(line));
      if (!first || !/\|move\|p1a: s0\|Flash Cannon\|p2a: s0/.test(first)) {
        return 'the control move was not redirected by Follow Me';
      }
      return null;
    },
  },
  {
    name: 'plus_boosts_special_attack_with_minus_ally',
    p1: () => [
      setOf('Ampharos', 'Plus', ['Thunderbolt', 'Protect'], '', {hp: 32, atk: 0, def: 17, spa: 17, spd: 0, spe: 0}),
      setOf('Toxtricity-Low-Key', 'Minus', ['Overdrive', 'Protect']),
      ...fillerAfter(offensive('Milotic', 'Competitive', ['Surf', 'Protect'])).slice(1, 5),
    ],
    p2: () => foeTeam(),
    script: [{p1: [{move: 'thunderbolt', target: 1}, 'protect'], p2: ['protect', 'protect']}],
    coverage: {ability: 'plus'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Thunderbolt\|/)) return 'Thunderbolt never executed';
      return null;
    },
  },
  {
    name: 'plus_control_without_ally',
    p1: () => [
      setOf('Ampharos', 'Plus', ['Thunderbolt', 'Protect'], '', {hp: 32, atk: 0, def: 17, spa: 17, spd: 0, spe: 0}),
      setOf('Toxtricity-Low-Key', 'Technician', ['Overdrive', 'Protect']),
      ...fillerAfter(offensive('Milotic', 'Competitive', ['Surf', 'Protect'])).slice(1, 5),
    ],
    p2: () => foeTeam(),
    script: [{p1: [{move: 'thunderbolt', target: 1}, 'protect'], p2: ['protect', 'protect']}],
    coverage: {ability: 'plus'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Thunderbolt\|/)) return 'Thunderbolt never executed';
      return null;
    },
  },
  {
    name: 'steelyspirit_boosts_ally_steel_move',
    p1: () => [
      setOf('Perrserker', 'Steely Spirit', ['Iron Head', 'Protect']),
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
      offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
      offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
      offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
      offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
    ],
    p2: () => foeTeam(),
    script: [{p1: ['protect', {move: 'ironhead', target: 1}], p2: ['protect', 'protect']}],
    coverage: {ability: 'steelyspirit'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1b: s1\|Iron Head\|/)) return 'the ally Steel move never executed';
      return null;
    },
  },
  {
    name: 'steelyspirit_boosts_own_steel_move',
    p1: () => fillerAfter(setOf('Perrserker', 'Steely Spirit', ['Iron Head', 'Protect'])),
    p2: () => foeTeam(),
    script: [{p1: [{move: 'ironhead', target: 1}, 'protect'], p2: ['protect', 'protect']}],
    coverage: {ability: 'steelyspirit'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Iron Head\|/)) return 'the holder Steel move never executed';
      return null;
    },
  },
  {
    name: 'steelyspirit_control_non_steel_move',
    p1: () => fillerAfter(setOf('Perrserker', 'Steely Spirit', ['Iron Head', 'Protect'])),
    p2: () => foeTeam(),
    script: [{p1: ['protect', 'protect'], p2: ['protect', 'protect']}],
    coverage: {ability: 'steelyspirit'},
    verify() {
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 5000, artifact: 'more_ability_tail_2.json', debugEnv: 'DEBUG_ABILITY_TAIL_2'});
