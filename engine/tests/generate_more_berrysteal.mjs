// Development-only corpus for the berry-stealing moves:
// - Bug Bite and Pluck take the target's held Berry through the reference
//   `takeItem` pipeline and immediately eat it (the user gains the `onEat`
//   effect: Sitrus/Oran healing, Lum/status cures, resistance berries with an
//   empty `onEat`),
// - a non-Berry held item and empty hands leave both moves without a steal.
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
  offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
  offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
  offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
  offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
  offensive('Aggron', 'Sturdy', ['Iron Head', 'Protect']),
];
const benchOf = (...species) => fill().filter(p => !species.includes(p.species));
const team = (head, ...rest) => [head, ...rest, ...benchOf(head.species, ...rest.map(p => p.species))].slice(0, 6);

const scizor = item => setOf('Scizor', 'Technician', ['Bug Bite', 'Protect'], item);
const corviknight = item => setOf('Corviknight', 'Pressure', ['Pluck', 'Protect'], item);
const foeWith = item => offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect'], item);

const TRIALS = [
  {
    name: 'berry_bugbite_steals_and_eats_sitrus',
    p1: () => team(scizor('')),
    p2: () => [foeWith('Sitrus Berry'), ...benchOf('Metagross')].slice(0, 6),
    script: [
      {p1: [{move: 'bugbite', target: 2}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'bugbite', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 2}, 'protect']},
    ],
    coverage: {move: 'bugbite'},
    verify(fixture, session) {
      if (!turnHas(session, 2, /\|-enditem\|p2a: [^|]*\|Sitrus Berry/)) return 'Bug Bite never took the Sitrus Berry';
      const before = stateAt(fixture, 2)?.sides[0].pokemon.find(p => p.roster === 0);
      const after = stateAt(fixture, 3)?.sides[0].pokemon.find(p => p.roster === 0);
      const foe = stateAt(fixture, 3)?.sides[1].pokemon.find(p => p.roster === 0);
      if (!before || !after || !foe) return 'the states were never recorded';
      if (before.hp === before.max_hp) return 'the user was not damaged first';
      const expected = Math.min(after.max_hp, before.hp + Math.floor(after.max_hp / 4));
      if (after.hp !== expected) return `the eater healed to ${after.hp} instead of ${expected}`;
      if (foe.item !== 0) return 'the target kept the Berry';
      return null;
    },
  },
  {
    name: 'berry_bugbite_steals_lum_and_cures',
    p1: () => team(scizor('')),
    p2: () => [
      setOf('Sableye', 'Prankster', ['Will-O-Wisp', 'Protect'], 'Lum Berry'),
      ...benchOf('Sableye'),
    ].slice(0, 6),
    script: [
      {p1: [{move: 'bugbite', target: 2}, 'protect'], p2: [{move: 'willowisp', target: 1}, 'protect']},
      {p1: [{move: 'bugbite', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 2}, 'protect']},
    ],
    coverage: {move: 'bugbite'},
    verify(fixture, session) {
      const before = stateAt(fixture, 2)?.sides[0].pokemon.find(p => p.roster === 0);
      if (before?.status !== ids.conditions.brn) return 'the user was not burned first';
      if (!turnHas(session, 2, /\|-enditem\|p2a: [^|]*\|Lum Berry/)) return 'Bug Bite never took the Lum Berry';
      if (!turnHas(session, 2, /\|-curestatus\|p1a: [^|]*\|brn/)) return 'the eaten Lum Berry never cured the burn';
      const after = stateAt(fixture, 3)?.sides[0].pokemon.find(p => p.roster === 0);
      if (after?.status !== 0) return 'the burn survived the Lum Berry';
      return null;
    },
  },
  {
    name: 'berry_bugbite_resist_berry_has_no_effect',
    p1: () => team(scizor('')),
    p2: () => [foeWith('Occa Berry'), ...benchOf('Metagross')].slice(0, 6),
    script: [
      {p1: [{move: 'bugbite', target: 2}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'bugbite', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 2}, 'protect']},
    ],
    coverage: {move: 'bugbite'},
    verify(fixture, session) {
      if (!turnHas(session, 2, /\|-enditem\|p2a: [^|]*\|Occa Berry/)) return 'Bug Bite never took the Occa Berry';
      const before = stateAt(fixture, 2)?.sides[0].pokemon.find(p => p.roster === 0);
      const after = stateAt(fixture, 3)?.sides[0].pokemon.find(p => p.roster === 0);
      if (!before || !after) return 'the states were never recorded';
      if (after.hp !== before.hp) return 'the resistance Berry changed the eater HP';
      if (after.status !== before.status) return 'the resistance Berry changed the eater status';
      return null;
    },
  },
  {
    name: 'berry_bugbite_ignores_non_berry_item',
    p1: () => team(scizor('')),
    p2: () => [foeWith('Focus Sash'), ...benchOf('Metagross')].slice(0, 6),
    script: [
      {p1: [{move: 'bugbite', target: 2}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'bugbite', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 2}, 'protect']},
    ],
    coverage: {move: 'bugbite'},
    verify(fixture, session) {
      if (turnHas(session, 2, /\|-enditem\|p2a: /)) return 'Bug Bite took a non-Berry item';
      const foe = stateAt(fixture, 3)?.sides[1].pokemon.find(p => p.roster === 0);
      if (foe && foe.item === 0) return 'the Focus Sash disappeared';
      return null;
    },
  },
  {
    name: 'berry_pluck_steals_and_eats_sitrus',
    p1: () => team(corviknight('')),
    p2: () => [foeWith('Sitrus Berry'), ...benchOf('Metagross')].slice(0, 6),
    script: [
      {p1: [{move: 'pluck', target: 2}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'pluck', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 2}, 'protect']},
    ],
    coverage: {move: 'pluck'},
    verify(fixture, session) {
      if (!turnHas(session, 2, /\|-enditem\|p2a: [^|]*\|Sitrus Berry/)) return 'Pluck never took the Sitrus Berry';
      const before = stateAt(fixture, 2)?.sides[0].pokemon.find(p => p.roster === 0);
      const after = stateAt(fixture, 3)?.sides[0].pokemon.find(p => p.roster === 0);
      if (!before || !after) return 'the states were never recorded';
      const expected = Math.min(after.max_hp, before.hp + Math.floor(after.max_hp / 4));
      if (after.hp !== expected) return `the eater healed to ${after.hp} instead of ${expected}`;
      return null;
    },
  },
  {
    name: 'berry_bugbite_empty_hands_nothing',
    p1: () => team(scizor('')),
    p2: () => [foeWith(''), ...benchOf('Metagross')].slice(0, 6),
    script: [
      {p1: [{move: 'bugbite', target: 2}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'bugbite', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 2}, 'protect']},
    ],
    coverage: {move: 'bugbite'},
    verify(fixture, session) {
      if (turnHas(session, 2, /\|-enditem\|/)) return 'Bug Bite ended an item on empty hands';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 19000, artifact: 'more_berrysteal.json', debugEnv: 'DEBUG_BERRYSTEAL'});
