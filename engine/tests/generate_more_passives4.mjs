// Development-only corpus for the fourth passive cluster:
//   Guard Dog   - Intimidate raises its own Attack; phazing is refused
//   Sweet Veil  - the ally aura refuses sleep and Yawn
//   Sand Spit   - a damaging hit starts a sandstorm
//   Rattled     - Dark/Bug/Ghost hits raise Speed
//   Steadfast   - a flinch raises Speed
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, logHas, everHas, runTrials} = createScaffold();

const POOL = [
  ['Milotic', 'Competitive', ['Ice Beam', 'Protect']],
  ['Metagross', 'Clear Body', ['Iron Head', 'Protect']],
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']],
  ['Falinks', 'Battle Armor', ['Smart Strike', 'Protect']],
  ['Chimecho', 'Levitate', ['Psychic', 'Protect']],
  ['Aggron', 'Sturdy', ['Iron Head', 'Protect']],
];
const team = (...heads) => [
  ...heads,
  ...POOL.filter(([species]) => !heads.some(head => head.species === species))
    .map(([species, ability, moves]) => offensive(species, ability, moves)),
].slice(0, 6);

const sheltered = (fixture, roster) => {
  const steps = fixture.steps.filter(step => step.expected.turn >= 1 && step.expected.turn <= 2);
  return !steps.some(step => step.expected.sides[0].pokemon
    .some(p => p.roster === roster && !p.fainted && p.active_slot !== 0));
};

const TRIALS = [
  {
    name: 'guarddog_intimidate_raises_attack',
    p1: () => team(setOf('Mabosstiff', 'Guard Dog', ['Crunch', 'Protect'])),
    p2: () => team(setOf('Incineroar', 'Intimidate', ['Flare Blitz', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {ability: 'guarddog'},
    verify(fixture, session) {
      if (!logHas(session, /\|switch\|p2a: s0\|Incineroar/)) return 'the Intimidate lead never entered';
      const raised = everHas(fixture, 0, 0, p => p.boosts[0] > 0);
      if (!raised) return 'Guard Dog never raised its own Attack';
      const dropped = everHas(fixture, 0, 0, p => p.boosts[0] < 0);
      if (dropped) return 'the Intimidate drop still landed';
      return null;
    },
  },
  {
    name: 'guarddog_refuses_phazing',
    p1: () => team(setOf('Mabosstiff', 'Guard Dog', ['Crunch', 'Protect'])),
    p2: () => team(setOf('Aggron', 'Sturdy', ['Roar', 'Iron Head', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: [{move: 'roar', target: 1}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{move: 'roar', target: 1}, 'protect']},
    ],
    coverage: {ability: 'guarddog'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Roar\|p1a: s0/)) return 'Roar never resolved';
      if (!logHas(session, /\|-activate\|p1a: s0\|ability: Guard Dog/)) {
        return 'Guard Dog never announced the refusal';
      }
      if (!sheltered(fixture, 0)) return 'the holder was dragged out anyway';
      return null;
    },
  },
  {
    name: 'sweetveil_blocks_sleep_for_the_team',
    p1: () => team(
      setOf('Alcremie', 'Sweet Veil', ['Dazzling Gleam', 'Protect']),
      offensive('Milotic', 'Competitive', ['Ice Beam', 'Protect']),
    ),
    p2: () => team(setOf('Venusaur', 'Overgrow', ['Sleep Powder', 'Sludge Bomb', 'Protect'])),
    script: [
      // First the ally, then the holder itself: the aura covers the whole side.
      {p1: ['protect', 'protect'], p2: [{move: 'sleeppowder', target: 2}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{move: 'sleeppowder', target: 1}, 'protect']},
    ],
    coverage: {ability: 'sweetveil'},
    verify(fixture, session) {
      if (!logHas(session, /\|-block\|p1[ab]: s[01]\|ability: Sweet Veil/)) {
        return 'Sweet Veil never blocked a sleep attempt';
      }
      const slept = everHas(fixture, 0, 0, p => p.status === ids.conditions.slp)
        || everHas(fixture, 0, 1, p => p.status === ids.conditions.slp);
      if (slept) return 'a side member fell asleep despite the aura';
      return null;
    },
  },
  {
    name: 'sandspit_starts_a_sandstorm',
    p1: () => team(setOf('Sandaconda', 'Sand Spit', ['Iron Head', 'Protect'])),
    p2: () => team(setOf('Aggron', 'Sturdy', ['Iron Head', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
    ],
    coverage: {ability: 'sandspit'},
    verify(fixture, session) {
      if (!logHas(session, /\|-damage\|p1a: s0\|/)) return 'the holder was never damaged';
      if (!logHas(session, /\|-weather\|Sandstorm/)) return 'no sandstorm was announced';
      const sand = fixture.steps.some(step => step.expected.climate.raw === 'sandstorm');
      if (!sand) return 'the field weather never became sandstorm';
      return null;
    },
  },
  {
    name: 'rattled_speed_boost_on_dark_hit',
    p1: () => team(setOf('Persian-Alola', 'Rattled', ['Night Slash', 'Protect'])),
    p2: () => team(setOf('Aggron', 'Sturdy', ['Dark Pulse', 'Iron Head', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: [{move: 'darkpulse', target: 1}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{move: 'darkpulse', target: 1}, 'protect']},
    ],
    coverage: {ability: 'rattled'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Dark Pulse\|p1a: s0/)) return 'Dark Pulse never resolved';
      if (!everHas(fixture, 0, 0, p => p.boosts[4] > 0)) return 'Rattled never raised Speed';
      return null;
    },
  },
  {
    name: 'steadfast_speed_boost_on_flinch',
    p1: () => team(setOf('Lucario', 'Steadfast', ['Close Combat', 'Protect'])),
    p2: () => team(setOf('Grimmsnarl', 'Frisk', ['Fake Out', 'Protect'])),
    script: [
      {p1: [{move: 'closecombat', target: 1}, 'protect'], p2: [{move: 'fakeout', target: 1}, 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {ability: 'steadfast'},
    verify(fixture, session) {
      if (!logHas(session, /\|cant\|p1a: s0\|flinch/)) return 'the holder was never flinched';
      if (!everHas(fixture, 0, 0, p => p.boosts[4] > 0)) return 'Steadfast never raised Speed';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 42000, artifact: 'more_passives4.json', debugEnv: 'DEBUG_PASSIVES4'});
