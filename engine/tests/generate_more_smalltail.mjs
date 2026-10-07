// Development-only corpus for three tail moves:
// - Synthesis / Moonlight / Morning Sun heal a weather-scaled fraction of the
//   user's maximum HP (half, two thirds in sun, a quarter in other weather),
// - Burn Up fails once the user is no longer Fire-type and strips that type
//   from the user on a landed hit (the `'???'` placeholder),
// - Tri Attack's 20% secondary samples burn, paralysis or freeze.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, logHas, monAt, runTrials} = createScaffold();

const POOL = [
  ['Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic']],
  ['Milotic', 'Competitive', ['Surf', 'Protect', 'Ice Beam']],
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect', 'Body Slam']],
  ['Falinks', 'Battle Armor', ['Smart Strike', 'Protect', 'Close Combat']],
  ['Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect', 'Psychic']],
  ['Aggron', 'Sturdy', ['Iron Head', 'Protect', 'Body Slam']],
  ['Ariados', 'Swarm', ['Leech Life', 'Protect', 'Sucker Punch']],
];
const team = (...heads) => [
  ...heads,
  ...POOL.filter(([species]) => !heads.some(head => head.species === species))
    .map(([species, ability, moves]) => offensive(species, ability, moves)),
].slice(0, 6);

// Extract the HP restored by a heal line for one ident (`|-heal|p1a: s0|H/MAX|`),
// using the last HP value logged for that ident before the heal. Lines whose
// maximum does not equal the Pokémon's maximum are the percentage duplicates
// the protocol emits for the opponent view and are skipped.
const healDelta = (log, ident, maxHp) => {
  const hpOf = line => {
    const parts = line.split('|');
    if (parts[2] !== ident) return null;
    const value = parts[3] ? parts[3].split(' ')[0] : '';
    if (!/^\d+\/\d+$/.test(value)) return null;
    const [hp, max] = value.split('/').map(Number);
    return max === maxHp ? hp : null;
  };
  for (let i = 0; i < log.length; i++) {
    const line = log[i];
    if (!line.startsWith('|-heal|' + ident + '|')) continue;
    const after = hpOf(line);
    if (after == null) continue;
    let before = null;
    for (let j = i - 1; j >= 0; j--) {
      const value = hpOf(log[j]);
      if (value != null) { before = value; break; }
    }
    if (before == null) return null;
    return {before, after};
  }
  return null;
};

const TRIALS = [
  {
    name: 'smalltail_synthesis_heals_half_in_clear_weather',
    p1: () => team(setOf('Gogoat', 'Sap Sipper', ['Synthesis', 'Protect', 'Leaf Blade'])),
    p2: () => team(offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic'])),
    script: [
      {p1: [{move: 'leafblade', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'synthesis', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'synthesis'},
    verify(fixture, session) {
      // The heal amount is read from the protocol stream: the state snapshots
      // only record turn boundaries, so a heal followed by same-turn damage
      // would be invisible there.
      const max = fixture.steps[0].expected.sides[0].pokemon.find(p => p.roster === 0).max_hp;
      const heal = healDelta(session.battle.log, 'p1a: s0', max);
      if (!heal) return 'no Synthesis heal appeared';
      const expected = Math.min(Math.floor(max * 0.5), max - heal.before);
      if (heal.after - heal.before !== expected) {
        return `the clear-weather heal was not half (${heal.after - heal.before} vs ${expected})`;
      }
      return null;
    },
  },
  {
    name: 'smalltail_synthesis_heals_two_thirds_in_sun',
    p1: () => team(setOf('Gogoat', 'Sap Sipper', ['Synthesis', 'Protect', 'Leaf Blade'])),
    p2: () => team(setOf('Torkoal', 'Shell Armor', ['Sunny Day', 'Protect', 'Body Press'])),
    script: [
      {p1: [{move: 'leafblade', target: 1}, 'protect'], p2: [{move: 'sunnyday', target: 0}, {move: 'ironhead', target: 1}]},
      {p1: [{move: 'leafblade', target: 1}, 'protect'], p2: ['protect', {move: 'ironhead', target: 1}]},
      {p1: [{move: 'synthesis', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'synthesis'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Synthesis\|/)) return 'Synthesis never executed';
      const max = fixture.steps[0].expected.sides[0].pokemon.find(p => p.roster === 0).max_hp;
      const heal = healDelta(session.battle.log, 'p1a: s0', max);
      if (!heal) return 'no Synthesis heal appeared';
      const expected = Math.min(Math.floor(max * 0.667), max - heal.before);
      if (Math.floor(max * 0.667) >= max - heal.before) return 'the sun heal was capped';
      if (heal.after - heal.before !== expected) {
        return `the sun heal was not two thirds (${heal.after - heal.before} vs ${expected})`;
      }
      if (!logHas(session, /\|move\|p2a: s0\|Sunny Day\|/)) return 'Sunny Day never set the weather';
      return null;
    },
  },
  {
    name: 'smalltail_burnup_strips_fire_type',
    p1: () => team(setOf('Arcanine', 'Intimidate', ['Burn Up', 'Protect', 'Flare Blitz'])),
    p2: () => team(setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic'])),
    script: [
      {p1: [{move: 'burnup', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'burnup', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'burnup'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Burn Up\|/)) return 'Burn Up never executed';
      if (!logHas(session, /\|-start\|p1a: s0\|typechange\|/)) return 'the type change never appeared';
      const stripped = fixture.steps
        .map(step => step.expected.sides[0].pokemon.find(p => p.roster === 0))
        .find(p => p && p.types.includes(0));
      if (!stripped) return 'the Fire type was never replaced by the placeholder';
      const fails = session.battle.log.filter(line => line.startsWith('|-fail|p1a: s0')).length;
      if (!fails) return 'the second Burn Up did not fail without the Fire type';
      return null;
    },
  },
  {
    name: 'smalltail_triattack_secondary_samples_status',
    p1: () => team(setOf('Alakazam', 'Magic Guard', ['Tri Attack', 'Protect', 'Psychic'])),
    p2: () => team(setOf('Milotic', 'Competitive', ['Surf', 'Protect', 'Ice Beam'])),
    script: [
      {p1: [{move: 'triattack', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'triattack', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'triattack', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'triattack', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'triattack'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Tri Attack\|/)) return 'Tri Attack never executed';
      const statuses = [ids.conditions.brn, ids.conditions.par, ids.conditions.frz];
      const afflicted = monAt(fixture, 1, 0).find(p => statuses.includes(p.status));
      if (!afflicted) return 'no sampled status ever landed';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 20000, artifact: 'more_smalltail.json', debugEnv: 'DEBUG_SMALLTAIL'});
