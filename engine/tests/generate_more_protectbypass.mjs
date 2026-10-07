// Development-only corpus for the two contact-protection bypass abilities:
// - `abilities:piercingdrill` (Mega Excadrill) and `abilities:unseenfist`
//   (Mega Golurk) register an `onHitProtect` handler that cancels the target's
//   protection for contact moves and marks the hit with `bypassProtect`;
// - the reference's `modifyDamage` turns that marker into a quarter-damage
//   modifier and adds the `-zbroken` message.
// Each scene attacks twice: once against an unprotected target (full damage)
// and once while the target protects (bypassed, quartered, and the target
// still loses HP).
import {createScaffold} from './fixture_scaffold.mjs';
const {setOf, foeTeam, foeWith, logHas, runTrials} = createScaffold();

const team = (...heads) => [
  ...heads,
  ...foeTeam().filter(p => !heads.some(head => head.species === p.species)),
].slice(0, 6);
const mon = (fixture, index, side, roster) => fixture.steps[index].expected.sides[side].pokemon
  .find(p => p.roster === roster);

/// The two attack turns: a free hit, then a hit into a fresh Protect.
const script = (mega) => [
  {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
  {p1: [{move: 'ironhead', target: 1, mega}, 'protect'], p2: ['protect', 'protect']},
  {p1: [{move: 'ironhead', target: 1}, 'protect'], p2: [{move: 'protect'}, 'protect']},
];
const verifyBypass = (fixture, session) => {
  if (!logHas(session, /\|move\|p1a: s0\|Iron Head\|/)) return 'Iron Head never executed';
  const hits = fixture.steps
    .map((step, i) => ({step, i}))
    .filter(({step}) => step.actions.some(action =>
      action.kind === 'Move' && action.own_slot === 0 && action.target_location === 1))
    .map(({i}) => i);
  if (hits.length < 2) return 'fewer than two attack turns were recorded';
  const free = hits[0];
  const protectedTurn = hits[1];
  const freeDamage = mon(fixture, free - 1, 1, 0).hp - mon(fixture, free + 1, 1, 0).hp;
  const bypassDamage = mon(fixture, protectedTurn - 1, 1, 0).hp
    - mon(fixture, protectedTurn + 1, 1, 0).hp;
  if (freeDamage <= 0) return 'the free hit did not damage the target';
  if (bypassDamage <= 0) return 'the protected hit was not bypassed';
  if (bypassDamage * 2 > freeDamage) {
    return `the bypassed hit was not quartered (${bypassDamage} vs ${freeDamage})`;
  }
  if (!logHas(session, /\|-zbroken\|/)) return 'the bypass marker never announced';
  return null;
};

const TRIALS = [
  {
    name: 'piercingdrill_bypasses_protect',
    p1: () => team(setOf('Excadrill', 'Mold Breaker', ['Iron Head', 'Protect', 'Earthquake'], 'Excadrite')),
    p2: () => foeWith(setOf('Milotic', 'Competitive', ['Surf', 'Protect', 'Ice Beam'])),
    script: script(true),
    coverage: {move: 'ironhead'},
    verify: verifyBypass,
  },
  {
    name: 'unseenfist_bypasses_protect',
    p1: () => team(setOf('Golurk', 'Iron Fist', ['Iron Head', 'Protect', 'Earthquake'], 'Golurkite')),
    p2: () => foeWith(setOf('Milotic', 'Competitive', ['Surf', 'Protect', 'Ice Beam'])),
    script: script(true),
    coverage: {move: 'ironhead'},
    verify: verifyBypass,
  },
];

runTrials(TRIALS, {seedBase: 33000, artifact: 'more_protectbypass.json', debugEnv: 'DEBUG_PROTECTBYPASS'});
