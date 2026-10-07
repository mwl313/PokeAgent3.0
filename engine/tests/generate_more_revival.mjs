// Development-only corpus for Revival Blessing:
// - the `onTryHit` gate refusing the move while no party member is fainted,
// - a successful revive of a fainted bench member at `floor(maxhp/2)` with
//   the user staying on the field (no instaswitch),
// - the doubles instaswitch branch: all three sacrificial partners self-KO
//   with Explosion until no live reserve remains, so the last corpse still
//   occupies the active side slot; reviving it must queue the immediate
//   switch-in back onto the field.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, logHas, monAt, runTrials} = createScaffold();

const P2_FILL = [
  ['Metagross', 'Clear Body', ['Iron Head', 'Protect']],
  ['Snorlax', 'Thick Fat', ['Body Slam', 'Protect']],
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']],
  ['Falinks', 'Battle Armor', ['Smart Strike', 'Protect']],
  ['Aggron', 'Sturdy', ['Iron Head', 'Protect']],
  ['Milotic', 'Competitive', ['Surf', 'Protect']],
];
const withFill = heads => [
  ...heads,
  ...P2_FILL.filter(([species]) => !heads.some(h => h.species === species))
    .map(([species, ability, moves]) => offensive(species, ability, moves)),
].slice(0, 6);

// Pawmot is the only legal Revival Blessing holder in the pinned regulation
// (Rabsca is `Past`). Protect first: a scripted entry whose move the active
// holder does not know (or has run out of) falls back to the first usable
// move, which must stay a harmless Protect.
const pawmot = () => setOf('Pawmot', 'Volt Absorb', ['Protect', 'Revival Blessing']);
const hydreigon = () => offensive('Hydreigon', 'Levitate', ['Crunch', 'Protect']);
const chimecho = () => offensive('Chimecho', 'Levitate', ['Rain Dance', 'Protect']);
const exploder = (species, ability) => setOf(species, ability, ['Explosion']);
const protector = (species, ability) => setOf(species, ability, ['Protect']);

const TRIALS = [
  {
    name: 'revival_fails_without_fainted',
    // Nobody has fainted yet: `onTryHit` refuses the move outright.
    p1: () => withFill([pawmot()]),
    p2: () => withFill([hydreigon()]),
    script: [
      {p1: ['revivalblessing', 'protect'], p2: [{move: 'crunch', target: 1}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{move: 'crunch', target: 1}, 'protect']},
    ],
    coverage: {move: 'revivalblessing'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Revival Blessing\|/)) return 'Revival Blessing was never used';
      const log = session.battle.log;
      const fail = log.findIndex(line => /^\|-fail\|p1a: s0/.test(line));
      if (fail < 0) return 'the failed revival never reported a fail';
      const faint = log.findIndex(line => line.startsWith('|faint|'));
      if (faint >= 0 && faint < fail) return 'a faint preceded the failed revival';
      if (log.some(line => /\[from\] move: Revival Blessing/.test(line))) {
        return 'a revive healed without any fainted party member';
      }
      return null;
    },
  },
  {
    name: 'revival_revives_bench_member',
    // Chimecho (s0) faints against Hydreigon; the auto replacement brings a
    // reserve in, which parks the corpse in the reserve half of the party.
    // Pawmot (s1) then revives it: no instaswitch, the user stays.
    p1: () => withFill([chimecho(), pawmot()]),
    p2: () => withFill([hydreigon()]),
    script: [
      {p1: ['raindance', 'protect'], p2: [{move: 'crunch', target: 1}, 'protect']},
      {p1: ['raindance', 'protect'], p2: [{move: 'crunch', target: 1}, 'protect']},
      {p1: ['raindance', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'revivalblessing'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'revivalblessing'},
    verify(fixture, session) {
      if (!logHas(session, /\|-heal\|p1: s0\|[^|]*\|\[from\] move: Revival Blessing/)) {
        return 'the revived bench member was never healed by Revival Blessing';
      }
      // The first boundary AFTER the faint where the member is alive again
      // must show it in reserve (no instaswitch) at exactly floor(maxhp/2);
      // the reviving user must not have swapped with it.
      const timeline = monAt(fixture, 0, 0);
      const firstDown = timeline.findIndex(p => p.hp === 0);
      if (firstDown < 0) return 'the lead never fainted to make the revive legal';
      const back = timeline.slice(firstDown + 1).find(p => p.hp > 0);
      if (!back) return 'the fainted member was never revived';
      if (back.active_slot !== null) return 'the bench revive put the member on the field';
      if (back.hp !== Math.floor(back.max_hp / 2)) return 'the revive did not land on floor(maxhp/2)';
      return null;
    },
  },
  {
    name: 'revival_instaswitch_doubles',
    // Doubles: Metagross and Golem Explode, Electrode Explodes last with no
    // live reserve left, so its corpse keeps the active slot. Reviving it
    // must switch it straight back onto the field at half HP.
    p1: () => withFill([
      pawmot(),
      exploder('Metagross', 'Clear Body'),
      exploder('Garganacl', 'Sturdy'),
      exploder('Forretress', 'Sturdy'),
    ]),
    p2: () => withFill([protector('Snorlax', 'Thick Fat'), protector('Aggron', 'Sturdy'), protector('Metagross', 'Clear Body')]),
    script: [
      {p1: ['protect', 'explosion'], p2: ['protect', 'protect']},
      {p1: ['protect', 'explosion'], p2: ['protect', 'protect']},
      {p1: ['protect', 'explosion'], p2: ['protect', 'protect']},
      {p1: ['revivalblessing', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'revivalblessing'},
    verify(fixture, session) {
      if (!logHas(session, /\|-heal\|p1: s3\|[^|]*\|\[from\] move: Revival Blessing/)) {
        return 'the active-slot partner was never revived';
      }
      // The instaswitch must put the revived partner back on the field at
      // slot 1 (b) at exactly the revived HP before any later damage.
      const backOnField = fixture.steps.some(step =>
        step.expected.sides[0].pokemon.some(p =>
          p.roster === 3 && p.active_slot === 1 && p.hp === Math.floor(p.max_hp / 2)));
      if (!backOnField) return 'the revived partner never returned to the field at half HP';
      // The reviving user never leaves the field while alive (it may still
      // faint later in the auto-played tail).
      const leftAlive = fixture.steps.some(step =>
        step.expected.turn > 0 && step.expected.sides[0].pokemon.some(p =>
          p.roster === 0 && p.active_slot === null && p.hp > 0));
      if (leftAlive) return 'the reviving user left the field';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 20000, artifact: 'more_revival.json', debugEnv: 'DEBUG_REVIVAL'});
