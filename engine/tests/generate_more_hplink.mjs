// Development-only corpus for the HP-manipulation moves:
// - Heal Pulse heals an ally or a foe for ceil(baseMaxhp / 2) and is refused
//   at full HP with the `heal` fail message,
// - Pain Split sets both HP values to the floor of their average (which can
//   raise the target),
// - Endeavor deals the target's HP minus the user's and is refused outright
//   unless the user is strictly lower.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, logHas, runTrials} = createScaffold();

const turnLines = (session, turn) => {
  const log = session.battle.log;
  const marker = `|turn|${turn}`;
  const start = log.indexOf(marker);
  if (start < 0) return [];
  const end = log.findIndex((line, index) => index > start && line === `|turn|${turn + 1}`);
  return log.slice(start, end < 0 ? undefined : end);
};
const turnHas = (session, turn, pattern) => turnLines(session, turn).some(line => pattern.test(line));
const stateAt = (fixture, turn) => fixture.steps.find(step => step.expected.turn === turn)?.expected;

const fill = () => [
  offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
  offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
  offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
  offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
  offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
  offensive('Aggron', 'Sturdy', ['Iron Head', 'Protect']),
];
const benchOf = (...species) => fill().filter(p => !species.includes(p.species));
const team = (head, ...rest) => [head, ...rest, ...benchOf(head.species, ...rest.map(p => p.species))].slice(0, 6);

const audino = () => setOf('Audino', 'Healer', ['Heal Pulse', 'Protect', 'Simple Beam']);
const sableye = () => setOf('Sableye', 'Prankster', ['Pain Split', 'Protect', 'Rain Dance']);
const raichu = () => offensive('Raichu', 'Lightning Rod', ['Endeavor', 'Protect', 'Fake Out']);
// A neutral foe: no recoil, no contact punishment and no status, so the HP
// arithmetic in the checks is not perturbed by secondary damage.
const foe = () => offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']);

const TRIALS = [
  {
    name: 'hplink_healpulse_heals_ally',
    p1: () => team(audino(), offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect'])),
    p2: () => [foe(), ...benchOf('Metagross')].slice(0, 6),
    script: [
      {p1: ['protect', {move: 'ironhead', target: 1}], p2: [{move: 'ironhead', target: 2}, 'protect']},
      {p1: [{move: 'healpulse', target: -2}, {move: 'ironhead', target: 1}], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'healpulse'},
    verify(fixture, session) {
      if (!turnHas(session, 2, /\|-heal\|p1b: /)) return 'Heal Pulse never healed the ally';
      const before = stateAt(fixture, 2)?.sides[0].pokemon.find(p => p.roster === 1);
      const after = stateAt(fixture, 3)?.sides[0].pokemon.find(p => p.roster === 1);
      if (!before || !after) return 'the ally state was never recorded';
      const expected = Math.min(after.max_hp, before.hp + Math.ceil(after.max_hp / 2));
      if (after.hp !== expected) return `the ally healed to ${after.hp} instead of ${expected}`;
      return null;
    },
  },
  {
    name: 'hplink_healpulse_refused_at_full_hp',
    p1: () => team(audino(), offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect'])),
    p2: () => [foe(), ...benchOf('Metagross')].slice(0, 6),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'healpulse', target: -2}, {move: 'ironhead', target: 1}], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'healpulse'},
    verify(fixture, session) {
      if (!turnHas(session, 2, /\|-fail\|p1b: [^|]*\|heal/)) return 'the full-HP heal was never refused';
      const before = stateAt(fixture, 2)?.sides[0].pokemon.find(p => p.roster === 1);
      const after = stateAt(fixture, 3)?.sides[0].pokemon.find(p => p.roster === 1);
      if (!before || !after || before.hp !== before.max_hp) return 'the ally was not at full HP';
      if (after.hp !== after.max_hp) return 'the refused heal still changed HP';
      return null;
    },
  },
  {
    name: 'hplink_healpulse_heals_foe',
    p1: () => team(audino(), offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect'])),
    p2: () => [foe(), ...benchOf('Metagross')].slice(0, 6),
    script: [
      {p1: ['protect', {move: 'ironhead', target: 1}], p2: [{move: 'ironhead', target: 2}, 'protect']},
      {p1: [{move: 'healpulse', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 2}, 'protect']},
    ],
    coverage: {move: 'healpulse'},
    verify(fixture, session) {
      if (!turnHas(session, 2, /\|-heal\|p2a: /)) return 'Heal Pulse never healed the foe';
      const before = stateAt(fixture, 2)?.sides[1].pokemon.find(p => p.roster === 0);
      const after = stateAt(fixture, 3)?.sides[1].pokemon.find(p => p.roster === 0);
      if (!before || !after) return 'the foe state was never recorded';
      if (before.hp === before.max_hp) return 'the foe was not damaged before the heal';
      const expected = Math.min(after.max_hp, before.hp + Math.ceil(after.max_hp / 2));
      if (after.hp !== expected) return `the foe healed to ${after.hp} instead of ${expected}`;
      return null;
    },
  },
  {
    name: 'hplink_painsplit_equalises_hp',
    p1: () => team(sableye(), offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect'])),
    p2: () => [foe(), ...benchOf('Metagross')].slice(0, 6),
    script: [
      {p1: ['raindance', 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'painsplit', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 2}, 'protect']},
    ],
    coverage: {move: 'painsplit'},
    verify(fixture, session) {
      if (!turnHas(session, 2, /\|-sethp\|p1a: /)) return 'Pain Split never set the user HP';
      const state = stateAt(fixture, 3);
      const user = state?.sides[0].pokemon.find(p => p.roster === 0);
      const target = state?.sides[1].pokemon.find(p => p.roster === 0);
      if (!user || !target) return 'the turn-3 state was never recorded';
      if (user.hp !== target.hp) return `HP values stayed unequal (${user.hp} vs ${target.hp})`;
      if (user.hp === user.max_hp) return 'Pain Split did not actually transfer HP';
      return null;
    },
  },
  {
    name: 'hplink_painsplit_raises_damaged_target',
    p1: () => team(sableye(), offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect'])),
    p2: () => [foe(), ...benchOf('Metagross')].slice(0, 6),
    script: [
      {p1: ['protect', {move: 'ironhead', target: 1}], p2: [{move: 'ironhead', target: 2}, 'protect']},
      {p1: [{move: 'painsplit', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 2}, 'protect']},
    ],
    coverage: {move: 'painsplit'},
    verify(fixture, session) {
      const beforeState = stateAt(fixture, 2);
      const before = beforeState?.sides[1].pokemon.find(p => p.roster === 0);
      const after = stateAt(fixture, 3)?.sides[1].pokemon.find(p => p.roster === 0);
      if (!before || !after) return 'the foe state was never recorded';
      if (before.hp === before.max_hp) return 'the foe was not damaged first';
      if (after.hp <= before.hp) return `Pain Split did not raise the damaged target (${before.hp} -> ${after.hp})`;
      return null;
    },
  },
  {
    name: 'hplink_endeavor_equalises_hp',
    p1: () => team(raichu(), offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect'])),
    p2: () => [foe(), ...benchOf('Metagross')].slice(0, 6),
    script: [
      {p1: [{move: 'endeavor', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'endeavor', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 2}, 'protect']},
    ],
    coverage: {move: 'endeavor'},
    verify(fixture, session) {
      if (!turnHas(session, 2, /^\|move\|p1a: [^|]*\|Endeavor\|/)) return 'Endeavor was never used';
      const state = stateAt(fixture, 3);
      const user = state?.sides[0].pokemon.find(p => p.roster === 0);
      const target = state?.sides[1].pokemon.find(p => p.roster === 0);
      if (!user || !target) return 'the turn-3 state was never recorded';
      if (target.hp !== user.hp) return `Endeavor left ${target.hp} HP against the user's ${user.hp}`;
      return null;
    },
  },
  {
    name: 'hplink_endeavor_refused_when_not_lower',
    p1: () => team(raichu(), offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect'])),
    p2: () => [foe(), ...benchOf('Metagross')].slice(0, 6),
    script: [
      {p1: ['protect', {move: 'ironhead', target: 1}], p2: [{move: 'ironhead', target: 2}, 'protect']},
      {p1: [{move: 'endeavor', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 2}, 'protect']},
    ],
    coverage: {move: 'endeavor'},
    verify(fixture, session) {
      if (!turnHas(session, 2, /^\|move\|p1a: [^|]*\|Endeavor\|/)) return 'Endeavor was never used';
      if (!turnHas(session, 2, /\|-immune\|p2a: /)) return 'the refused Endeavor never failed';
      const before = stateAt(fixture, 2)?.sides[1].pokemon.find(p => p.roster === 0);
      const after = stateAt(fixture, 3)?.sides[1].pokemon.find(p => p.roster === 0);
      if (!before || !after) return 'the foe state was never recorded';
      const user = stateAt(fixture, 2)?.sides[0].pokemon.find(p => p.roster === 0);
      if (!user || user.hp !== user.max_hp) return 'the user was not at full HP';
      if (after.hp !== before.hp) return 'the refused Endeavor still dealt damage';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 18500, artifact: 'more_hplink.json', debugEnv: 'DEBUG_HPLINK'});
