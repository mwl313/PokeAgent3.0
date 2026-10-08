// Development-only corpus for the two retaliation-power callbacks, which read
// *different* state:
//   moves:payback.basePowerCallback   - doubles unless the current target just
//                                       switched in or still has a move action
//                                       queued (`queue.willMove`).
//   moves:avalanche.basePowerCallback - doubles when the current target already
//                                       dealt damage to the user this turn.
// Every scene is one battle in which the same user hits the same defender
// twice - once with the boosted power, once with the declared power - so the
// verify can compare the two damage numbers directly and prove the reference
// really took both branches (otherwise the seed is retried).
import {createScaffold} from './fixture_scaffold.mjs';
const {setOf, offensive, logHas, runTrials} = createScaffold();

const POOL = [
  ['Aggron', 'Sturdy', ['Iron Head', 'Protect']],
  ['Metagross', 'Clear Body', ['Bullet Punch', 'Protect']],
  ['Milotic', 'Competitive', ['Ice Beam', 'Protect']],
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']],
  ['Chimecho', 'Levitate', ['Psychic', 'Protect']],
  ['Falinks', 'Battle Armor', ['Smart Strike', 'Protect']],
  ['Ariados', 'Swarm', ['Leech Life', 'Protect']],
];
const team = (...heads) => [
  ...heads,
  ...POOL.filter(([species]) => !heads.some(head => head.species === species))
    .map(([species, ability, moves]) => offensive(species, ability, moves)),
].slice(0, 6);

// HP after every `|-damage|target|hp/max` line that belongs to a `moveName`
// action of `userIdent`. The scan stops at the next move line so ability,
// effectiveness and critical-hit lines between the move and its damage are
// ignored.
function retaliationHits(session, userIdent, targetIdent, moveName) {
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
}

// No healing happens in these scenes and Protects do not change HP, so the
// defender's HP trail gives both damage numbers directly.
function damagePair(hits, maxHp) {
  if (hits.length < 2) return null;
  return {first: maxHp - hits[0], second: hits[0] - hits[1]};
}

const switchMaxHp = (session, ident) => {
  const line = session.battle.log.find(l => l.startsWith(`|switch|${ident}|`));
  return line ? Number(line.split('|')[4].split('/')[1]) : 0;
};

const TRIALS = [
  {
    name: 'avalanche_doubles_after_being_hit',
    p1: () => team(setOf('Abomasnow', 'Snow Warning', ['Avalanche', 'Protect'])),
    p2: () => team(setOf('Milotic', 'Competitive', ['Ice Beam', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      // The foe damages the Avalanche user first: doubled.
      {p1: [{move: 'avalanche', target: 1}, 'protect'], p2: [{move: 'icebeam', target: 1}, 'protect']},
      // The foe attacks the partner instead: the user is undamaged, base power.
      {p1: [{move: 'avalanche', target: 1}, 'protect'], p2: [{move: 'icebeam', target: 2}, 'protect']},
    ],
    coverage: {move: 'avalanche'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Avalanche\|p2a: s0/)) return 'Avalanche never hit p2a';
      if (!logHas(session, /\|move\|p2a: s0\|Ice Beam\|p1a: s0/)) return 'the foe never damaged the user on turn 2';
      if (!logHas(session, /\|move\|p2a: s0\|Ice Beam\|p1b: s1/)) return 'the foe never attacked the partner on turn 3';
      const hits = retaliationHits(session, 'p1a: s0', 'p2a: s0', 'Avalanche');
      if (hits.length < 2) return `expected two Avalanche hits, saw ${hits.length}`;
      const pair = damagePair(hits, switchMaxHp(session, 'p2a: s0'));
      if (!pair) return 'not enough damage samples';
      if (pair.first <= pair.second * 1.4) {
        return `Avalanche did not double (hit ${pair.first} vs ${pair.second})`;
      }
      return null;
    },
  },
  {
    name: 'payback_not_boosted_for_a_fresh_switchin',
    p1: () => team(setOf('Aggron', 'Sturdy', ['Payback', 'Protect'])),
    p2: () => team(
      setOf('Aerodactyl', 'Pressure', ['Rock Slide', 'Protect']),
      offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
      // Bench slot: team preview keeps entries 0-1 as the leads, so the
      // replacement must be listed third.
      setOf('Milotic', 'Competitive', ['Ice Beam', 'Protect']),
    ),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      // The target switches out; Payback retargets to the fresh Pokémon, which
      // is still `newlySwitched` and therefore not doubled.
      {p1: [{move: 'payback', target: 1}, 'protect'], p2: [{switch: 's2'}, 'protect']},
      // The replacement is now an ordinary target that already moved: doubled.
      {p1: [{move: 'payback', target: 1}, 'protect'], p2: [{move: 'icebeam', target: 1}, 'protect']},
    ],
    coverage: {move: 'payback'},
    verify(fixture, session) {
      if (!logHas(session, /\|switch\|p2a: s2\|Milotic/)) return 'the replacement never took the slot';
      const hits = retaliationHits(session, 'p1a: s0', 'p2a: s2', 'Payback');
      if (hits.length < 2) return `expected two Payback hits, saw ${hits.length}`;
      const pair = damagePair(hits, switchMaxHp(session, 'p2a: s2'));
      if (!pair) return 'not enough damage samples';
      if (pair.second <= pair.first * 1.4) {
        return `the switch-in hit was not the lower one (${pair.first} vs ${pair.second})`;
      }
      return null;
    },
  },
  {
    name: 'payback_boosted_only_after_the_target_acted',
    p1: () => team(setOf('Aerodactyl', 'Pressure', ['Payback', 'Protect'])),
    p2: () => team(setOf('Metagross', 'Clear Body', ['Bullet Punch', 'Iron Head', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      // Bullet Punch moves first even though Metagross is slower: the target's
      // action is already off the queue when Payback lands.
      {p1: [{move: 'payback', target: 1}, 'protect'], p2: [{move: 'bulletpunch', target: 1}, 'protect']},
      // Now the user moves first, so the target's Iron Head is still queued.
      {p1: [{move: 'payback', target: 1}, 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
    ],
    coverage: {move: 'payback'},
    verify(fixture, session) {
      const log = session.battle.log;
      const punch = log.findIndex(line => line.startsWith('|move|p2a: s0|Bullet Punch|'));
      const firstPayback = log.findIndex(line => line.startsWith('|move|p1a: s0|Payback|'));
      const secondPayback = log.findIndex((line, i) => i > firstPayback && line.startsWith('|move|p1a: s0|Payback|'));
      const head = log.findIndex((line, i) => i > secondPayback && line.startsWith('|move|p2a: s0|Iron Head|'));
      if (punch < 0 || firstPayback < 0 || secondPayback < 0 || head < 0) return 'a scripted move never executed';
      if (punch > firstPayback) return 'Bullet Punch did not precede the first Payback';
      if (secondPayback > head) return 'the second Payback did not precede Iron Head';
      const hits = retaliationHits(session, 'p1a: s0', 'p2a: s0', 'Payback');
      if (hits.length < 2) return `expected two Payback hits, saw ${hits.length}`;
      const pair = damagePair(hits, switchMaxHp(session, 'p2a: s0'));
      if (!pair) return 'not enough damage samples';
      if (pair.first <= pair.second * 1.4) {
        return `Payback did not double after the target acted (${pair.first} vs ${pair.second})`;
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 37000, artifact: 'more_payback_avalanche.json', debugEnv: 'DEBUG_PAYBACK_AVALANCHE'});
