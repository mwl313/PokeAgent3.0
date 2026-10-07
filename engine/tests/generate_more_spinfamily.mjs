// Development-only corpus for the Rapid-Spin family:
// - Ice Spinner clears the active terrain (including through a substitute),
// - Mortal Spin sheds the user's Leech Seed, partial trapping and its own
//   entry hazards while poisoning the target.
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

const TRIALS = [
  {
    name: 'spinfamily_icespinner_clears_terrain',
    p1: () => team(setOf('Abomasnow', 'Snow Warning', ['Ice Spinner', 'Protect', 'Energy Ball'])),
    p2: () => team(setOf('Ampharos', 'Static', ['Electric Terrain', 'Protect', 'Dragon Pulse'])),
    script: [
      {p1: [{move: 'protect', target: 0}, 'protect'], p2: [{move: 'electricterrain', target: 0}, 'protect']},
      {p1: [{move: 'icespinner', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'icespinner'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Ice Spinner\|/)) return 'Ice Spinner never executed';
      const terrains = fixture.steps.map(step => step.expected.field.map(f => f[0]));
      if (!terrains.some(list => list.includes(ids.conditions.electricterrain))) {
        return 'the terrain was never set';
      }
      const last = terrains.at(-1) ?? [];
      if (last.includes(ids.conditions.electricterrain)) return 'the terrain survived the hit';
      return null;
    },
  },
  {
    name: 'spinfamily_mortalspin_sheds_hazards_and_poisons',
    p1: () => team(setOf('Glimmora', 'Toxic Debris', ['Mortal Spin', 'Protect', 'Sludge Bomb'])),
    p2: () => team(setOf('Chesnaught', 'Bulletproof', ['Spikes', 'Protect', 'Seed Bomb'])),
    script: [
      {p1: [{move: 'protect', target: 0}, 'protect'], p2: [{move: 'spikes', target: 0}, 'protect']},
      {p1: ['mortalspin', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'mortalspin'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Mortal Spin\|/)) return 'Mortal Spin never executed';
      const spikes = ids.conditions.spikes;
      const own = fixture.steps.map(step => step.expected.sides[0].conditions.map(c => c[0]));
      if (!own.some(list => list.includes(spikes))) return 'the user never had its own Spikes';
      const last = own.at(-1) ?? [];
      if (last.includes(spikes)) return 'the Spikes survived Mortal Spin';
      if (!monAt(fixture, 1, 0).some(p => p.status === ids.conditions.psn)) {
        return 'the target was never poisoned';
      }
      return null;
    },
  },
  {
    name: 'spinfamily_noretreat_boosts_once',
    p1: () => team(setOf('Falinks', 'Battle Armor', ['No Retreat', 'Protect', 'Close Combat'])),
    p2: () => team(setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic'])),
    script: [
      {p1: [{move: 'noretreat', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'noretreat', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'closecombat', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'noretreat'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|No Retreat\|/)) return 'No Retreat never executed';
      const steps = fixture.steps
        .map(step => step.expected.sides[0].pokemon.find(p => p.roster === 0))
        .filter(Boolean);
      const boosted = steps.find(p => p.boosts[0] === 1);
      if (!boosted) return 'no Attack stage was raised';
      if (!(boosted.boosts[1] === 1 && boosted.boosts[2] === 1 && boosted.boosts[3] === 1 && boosted.boosts[4] === 1)) {
        return 'the five-stat boost was incomplete';
      }
      if (!steps.some(p => p.volatiles.includes('noretreat'))) return 'the marker volatile was never recorded';
      const fails = session.battle.log.filter(line => line.startsWith('|-fail|p1a: s0')).length;
      if (!fails) return 'the second No Retreat did not fail';
      return null;
    },
  },
  {
    name: 'crashdamage_pays_half_max_on_block',
    p1: () => team(setOf('Medicham', 'Pure Power', ['High Jump Kick', 'Protect', 'Close Combat'])),
    p2: () => team(setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic'])),
    script: [
      {p1: [{move: 'highjumpkick', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'highjumpkick'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|High Jump Kick\|/)) return 'High Jump Kick never executed';
      const max = fixture.steps[0].expected.sides[0].pokemon.find(p => p.roster === 0).max_hp;
      const half = Math.trunc(max / 2);
      const hp = fixture.steps.map(step => step.expected.sides[0].pokemon.find(p => p.roster === 0)).filter(Boolean).map(p => p.hp);
      if (!hp.some((value, i) => i > 0 && hp[i - 1] - value === half)) {
        return 'the user never paid half its maximum HP';
      }
      return null;
    },
  },
  {
    name: 'crashdamage_supercellslam_and_axekick_pay_half_max',
    p1: () => team(
      setOf('Ampharos', 'Static', ['Supercell Slam', 'Protect', 'Dragon Pulse']),
      setOf('Medicham', 'Pure Power', ['Axe Kick', 'Protect', 'Close Combat']),
    ),
    p2: () => team(setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic'])),
    script: [
      {p1: [{move: 'supercellslam', target: 1}, {move: 'axekick', target: 1}], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'supercellslam'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Supercell Slam\|/)) return 'Supercell Slam never executed';
      if (!logHas(session, /\|move\|p1b: s1\|Axe Kick\|/)) return 'Axe Kick never executed';
      const paid = (side, roster) => {
        const steps = fixture.steps
          .map(step => step.expected.sides[side].pokemon.find(p => p.roster === roster))
          .filter(Boolean)
          .map(p => p.hp);
        return steps.some((value, i) => i > 0 && steps[i - 1] - value > 0 && value <= steps[0] / 2);
      };
      if (!paid(0, 0)) return 'Supercell Slam never paid crash damage';
      if (!paid(0, 1)) return 'Axe Kick never paid crash damage';
      return null;
    },
  },
  {
    name: 'crashdamage_axekick_pays_half_max',
    p1: () => team(setOf('Medicham', 'Pure Power', ['Axe Kick', 'Protect', 'Close Combat'])),
    p2: () => team(setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic'])),
    script: [
      {p1: [{move: 'axekick', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'axekick'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Axe Kick\|/)) return 'Axe Kick never executed';
      const max = fixture.steps[0].expected.sides[0].pokemon.find(p => p.roster === 0).max_hp;
      const hp = fixture.steps.map(step => step.expected.sides[0].pokemon.find(p => p.roster === 0)).filter(Boolean).map(p => p.hp);
      if (!hp.some((value, i) => i > 0 && hp[i - 1] - value === Math.trunc(max / 2))) {
        return 'the user never paid half its maximum HP';
      }
      return null;
    },
  },
  {
    name: 'safeguard_refuses_foe_status_and_yawn',
    p1: () => team(setOf('Milotic', 'Competitive', ['Safeguard', 'Protect', 'Ice Beam'])),
    p2: () => team(
      setOf('Ampharos', 'Static', ['Thunder Wave', 'Protect', 'Dragon Pulse']),
      setOf('Chimecho', 'Levitate', ['Yawn', 'Protect', 'Psychic']),
    ),
    script: [
      {p1: ['safeguard', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: [{move: 'thunderwave', target: 1}, {move: 'yawn', target: 1}]},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'safeguard'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Safeguard\|/)) return 'Safeguard never executed';
      if (!monAt(fixture, 0, 0).every(p => p.status === 0)) return 'a foe status landed under Safeguard';
      if (monAt(fixture, 0, 0).some(p => p.volatiles.includes('yawn'))) {
        return 'the Yawn countdown was added under Safeguard';
      }
      const conditions = fixture.steps.map(step => step.expected.sides[0].conditions.map(c => c[0]));
      if (!conditions.some(list => list.includes(ids.conditions.safeguard))) {
        return 'the Safeguard side condition was never recorded';
      }
      return null;
    },
  },
  {
    name: 'smackdown_grounds_a_flying_target',
    p1: () => team(setOf('Chesnaught', 'Bulletproof', ['Smack Down', 'Protect', 'Earthquake'])),
    p2: () => team(
      setOf('Corviknight', 'Mirror Armor', ['Brave Bird', 'Protect', 'Iron Head']),
      setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic']),
    ),
    script: [
      {p1: [{move: 'protect', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'smackdown', target: 1}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['earthquake', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'smackdown'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Smack Down\|/)) return 'Smack Down never executed';
      if (!monAt(fixture, 1, 0).some(p => p.volatiles.includes('smackdown'))) {
        return 'the smackdown marker was never recorded';
      }
      if (!logHas(session, /\|move\|p1a: s0\|Earthquake\|/)) return 'Earthquake never executed';
      if (!logHas(session, /\|-damage\|p2a: s0\|/)) return 'the grounded target never took Earthquake damage';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 26000, artifact: 'more_spinfamily.json', debugEnv: 'DEBUG_SPIN'});
