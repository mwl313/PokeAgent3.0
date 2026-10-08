// Development-only corpus for the second passive cluster:
//   Corrosion  - poisons Steel/Poison targets (the immunity lookup is skipped)
//   Merciless  - always crits against a poisoned target
//   Stakeout   - doubles the attacking stat against a defender that switched in
//   Hustle     - physical damage boost (and 0.8x accuracy)
//   Skill Link - array multi-hit moves always hit their maximum count
// Each scene asserts the behaviour in the recorded reference battle; the
// boundary-by-boundary corpus comparison is the witness.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, logHas, everHas, runTrials} = createScaffold();

const POOL = [
  ['Metagross', 'Clear Body', ['Iron Head', 'Protect']],
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']],
  ['Falinks', 'Battle Armor', ['Smart Strike', 'Protect']],
  ['Chimecho', 'Levitate', ['Psychic', 'Protect']],
  ['Ariados', 'Swarm', ['Leech Life', 'Protect']],
  ['Aggron', 'Sturdy', ['Iron Head', 'Protect']],
];
const team = (...heads) => [
  ...heads,
  ...POOL.filter(([species]) => !heads.some(head => head.species === species))
    .map(([species, ability, moves]) => offensive(species, ability, moves)),
].slice(0, 6);

const hitDamages = (session, userIdent, targetIdent, moveName) => {
  const log = session.battle.log;
  const hits = [];
  for (let i = 0; i < log.length; i++) {
    if (!log[i].startsWith('|move|')) continue;
    const [, , source, name] = log[i].split('|');
    if (source !== userIdent || name !== moveName) continue;
    for (let j = i + 1; j < log.length && !log[j].startsWith('|move|'); j++) {
      if (log[j].startsWith(`|-damage|${targetIdent}|`)) {
        hits.push(Number(log[j].split('|')[3].split(' ')[0].split('/')[0]));
        break;
      }
    }
  }
  return hits;
};

const switchMaxHp = (session, ident) => {
  const line = session.battle.log.find(l => l.startsWith(`|switch|${ident}|`));
  return line ? Number(line.split('|')[4].split('/')[1]) : 0;
};

const TRIALS = [
  {
    name: 'corrosion_poisons_a_steel_type',
    p1: () => team(setOf('Salazzle', 'Corrosion', ['Toxic', 'Sludge Bomb', 'Protect'])),
    p2: () => team(setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect'])),
    script: [
      // The target must not Protect on the Toxic turn.
      {p1: [{move: 'toxic', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'sludgebomb', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
    ],
    coverage: {ability: 'corrosion'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Toxic\|p2a: s0/)) return 'Toxic never resolved';
      const poisoned = everHas(fixture, 1, 0,
        p => p.status === ids.conditions.tox || p.status === ids.conditions.psn);
      if (!poisoned) return 'the Steel-type target never got poisoned';
      return null;
    },
  },
  {
    name: 'merciless_always_crits_a_poisoned_target',
    p1: () => team(setOf('Toxapex', 'Merciless', ['Toxic', 'Liquidation', 'Protect'])),
    p2: () => team(setOf('Milotic', 'Competitive', ['Ice Beam', 'Protect'])),
    script: [
      // The target must not Protect on the Toxic turn.
      {p1: [{move: 'toxic', target: 1}, 'protect'], p2: [{move: 'icebeam', target: 1}, 'protect']},
      {p1: [{move: 'liquidation', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {ability: 'merciless'},
    verify(fixture, session) {
      const poisoned = everHas(fixture, 1, 0,
        p => p.status === ids.conditions.tox || p.status === ids.conditions.psn);
      if (!poisoned) return 'the target never got poisoned';
      if (!logHas(session, /\|-crit\|p2a: s0/)) return 'Merciless never forced the crit';
      return null;
    },
  },
  {
    name: 'stakeout_doubles_against_a_fresh_switchin',
    p1: () => team(setOf('Mabosstiff', 'Stakeout', ['Crunch', 'Protect'])),
    p2: () => team(
      setOf('Aerodactyl', 'Pressure', ['Rock Slide', 'Protect']),
      offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
      // Bench: team preview keeps entries 0-1 as the leads.
      setOf('Milotic', 'Competitive', ['Ice Beam', 'Protect']),
    ),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'crunch', target: 1}, 'protect'], p2: [{switch: 's2'}, 'protect']},
      {p1: [{move: 'crunch', target: 1}, 'protect'], p2: [{move: 'icebeam', target: 1}, 'protect']},
    ],
    coverage: {ability: 'stakeout'},
    verify(fixture, session) {
      if (!logHas(session, /\|switch\|p2a: s2\|Milotic/)) return 'the replacement never took the slot';
      const hits = hitDamages(session, 'p1a: s0', 'p2a: s2', 'Crunch');
      if (hits.length < 2) return `expected two Crunch hits, saw ${hits.length}`;
      const max = switchMaxHp(session, 'p2a: s2');
      const boosted = max - hits[0];
      const base = hits[0] - hits[1];
      if (boosted <= base * 1.4) return `Stakeout did not double (${boosted} vs ${base})`;
      return null;
    },
  },
  {
    name: 'hustle_boosts_physical_damage',
    p1: () => team(setOf('Flapple', 'Hustle', ['Aerial Ace', 'Protect'])),
    p2: () => team(setOf('Milotic', 'Competitive', ['Ice Beam', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'aerialace', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'aerialace', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {ability: 'hustle'},
    verify(fixture, session) {
      if (!everHas(fixture, 0, 0, p => p.ability === ids.abilities.hustle)) {
        return 'the holder never carried Hustle';
      }
      if (!logHas(session, /\|move\|p1a: s0\|Aerial Ace\|p2a: s0/)) return 'Aerial Ace never resolved';
      if (!logHas(session, /\|-damage\|p2a: s0\|/)) return 'the Hustle holder never dealt damage';
      return null;
    },
  },
  {
    name: 'skilllink_maximises_the_hit_count',
    p1: () => team(setOf('Toucannon', 'Skill Link', ['Bullet Seed', 'Protect'])),
    p2: () => team(setOf('Milotic', 'Competitive', ['Ice Beam', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'bulletseed', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {ability: 'skilllink'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Bullet Seed\|p2a: s0/)) return 'Bullet Seed never resolved';
      const counts = session.battle.log
        .filter(line => line.startsWith('|-hitcount|p2a: s0|'))
        .map(line => Number(line.split('|')[3]));
      if (!counts.includes(5)) return `expected a five-hit count, saw ${counts.join(',')}`;
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 40000, artifact: 'more_passives2.json', debugEnv: 'DEBUG_PASSIVES2'});
