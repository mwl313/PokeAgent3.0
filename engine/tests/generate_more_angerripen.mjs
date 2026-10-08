// Development-only corpus for Anger Point (a landed crit maximises Attack) and
// Ripen (berry healing doubles and weaken-berry damage halves again).
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

const hpAfter = (session, ident, kind) => {
  const lines = session.battle.log.filter(line => line.startsWith(`|${kind}|${ident}|`));
  return lines.map(line => Number(line.split('|')[3].split(' ')[0].split('/')[0]));
};

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

const TRIALS = [
  {
    name: 'angerpoint_maximises_attack_after_a_crit',
    // Flower Trick always crits, so the Attack cap must be reached.
    p1: () => team(setOf('Krookodile', 'Anger Point', ['Crunch', 'Protect'])),
    p2: () => team(setOf('Meowscarada', 'Overgrow', ['Flower Trick', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: [{move: 'flowertrick', target: 1}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{move: 'flowertrick', target: 1}, 'protect']},
    ],
    coverage: {ability: 'angerpoint'},
    verify(fixture, session) {
      if (!logHas(session, /\|-crit\|p1a: s0/)) return 'no critical hit landed on the holder';
      if (!everHas(fixture, 0, 0, p => p.boosts[0] === 6)) {
        return 'Anger Point never maximised Attack';
      }
      return null;
    },
  },
  {
    name: 'ripen_doubles_the_sitrus_heal',
    p1: () => team(setOf('Appletun', 'Ripen', ['Body Press', 'Protect'], 'Sitrus Berry')),
    p2: () => team(setOf('Aggron', 'Sturdy', ['Iron Head', 'Protect'])),
    script: [
      {p1: [{move: 'bodypress', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'bodypress', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'bodypress', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'bodypress', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
    ],
    coverage: {ability: 'ripen'},
    verify(fixture, session) {
      if (!logHas(session, /\|-enditem\|p1a: s0\|Sitrus Berry/)) return 'the berry was never eaten';
      // The Sitrus heal's own line carries the HP after the heal; the damage
      // line just before it gives the HP the heal started from.
      const log = session.battle.log;
      const heal = log.findIndex(line => /^\|-heal\|p1a: s0\|.*\[from\] item: Sitrus Berry/.test(line));
      if (heal < 0) return 'the Sitrus heal was never logged';
      let damage = -1;
      for (let i = heal - 1; i >= 0; i--) {
        if (log[i].startsWith('|-damage|p1a: s0|')) {
          damage = i;
          break;
        }
        if (log[i].startsWith('|move|')) break;
      }
      if (damage < 0) return 'no damage line preceded the berry heal';
      const before = Number(log[damage].split('|')[3].split(' ')[0].split('/')[0]);
      const after = Number(log[heal].split('|')[3].split(' ')[0].split('/')[0]);
      const max = Number(log[heal].split('|')[3].split(' ')[0].split('/')[1]) || 100;
      // A plain Sitrus heals a quarter; Ripen doubles it to a half.
      if (after - before < max * 0.4) {
        return `the Ripen heal was too small (${after - before} of ${max})`;
      }
      return null;
    },
  },
  {
    name: 'ripen_quarters_the_resist_berry_damage',
    // Yache Berry halves a super-effective Ice hit (Appletun is 4x weak); Ripen
    // adds its own halving, so the second Ice Beam (berry gone) must be at
    // least twice the first.
    p1: () => team(setOf('Appletun', 'Ripen', ['Body Press', 'Protect'], 'Yache Berry')),
    p2: () => team(setOf('Milotic', 'Competitive', ['Ice Beam', 'Protect'])),
    script: [
      {p1: [{move: 'bodypress', target: 1}, 'protect'], p2: [{move: 'icebeam', target: 1}, 'protect']},
      {p1: [{move: 'bodypress', target: 1}, 'protect'], p2: [{move: 'icebeam', target: 1}, 'protect']},
    ],
    coverage: {ability: 'ripen'},
    verify(fixture, session) {
      if (!logHas(session, /\|-enditem\|p1a: s0\|Yache Berry/)) return 'the Yache Berry was never eaten';
      const hits = hitDamages(session, 'p2a: s0', 'p1a: s0', 'Ice Beam');
      if (hits.length < 2) return `expected two Ice Beam hits, saw ${hits.length}`;
      const max = 100;
      const first = max - hits[0];
      const second = hits[0] - hits[1];
      if (second < first * 1.8) {
        return `the weakened hit was not far smaller (${first} vs ${second})`;
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 48000, artifact: 'more_angerripen.json', debugEnv: 'DEBUG_ANGERRIPEN'});
