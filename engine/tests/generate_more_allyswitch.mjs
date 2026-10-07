// Development-only corpus for Ally Switch:
// - the doubles position swap between the user and its partner,
// - the consecutive-use restart gate (`condition.onRestart` runs
//   `randomChance(1, counter)` and deletes the volatile on failure),
// - both restart branches, and
// - the `onHit` failure when the partner slot holds a fainted Pokemon
//   (First Impression is also +2; the faster user faints the partner first).
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {setOf, offensive, fillerAfter, monAt, logHas, runTrials} = createScaffold();

// First Impression (+2) from Falinks (Speed 75) outspeeds Chimecho's Ally
// Switch (+2, Speed 65), so Meowscarada faints before the swap step runs.
const partnerTeam = () => [
  setOf('Chimecho', 'Levitate', ['Ally Switch', 'Protect', 'Psychic', 'Dazzling Gleam']),
  setOf('Meowscarada', 'Overgrow', ['Protect', 'Night Slash'], '',
    {hp: 0, atk: 32, def: 0, spa: 0, spd: 0, spe: 0}),
  offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
  offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
  offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
  offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
];
const foeTeam = () => [
  offensive('Falinks', 'Battle Armor', ['First Impression', 'Protect']),
  offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
  offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
  offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
  offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
  offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
];

const swapCount = session =>
  session.battle.log.filter(line => /\|swap\|p1[ab]: s0\|/.test(line)).length;

const slotSequence = (fixture, roster) => {
  const slots = monAt(fixture, 0, roster).map(p => p.active_slot);
  return slots.filter((value, index) => index === 0 || value !== slots[index - 1]);
};

const TRIALS = [
  {
    name: 'allyswitch_swaps_partner_slots',
    p1: partnerTeam,
    p2: foeTeam,
    script: [{p1: ['allyswitch', 'protect'], p2: ['protect', 'protect']}],
    coverage: {move: 'allyswitch'},
    verify(fixture, session) {
      if (swapCount(session) !== 1) return `expected one swap, saw ${swapCount(session)}`;
      const user = slotSequence(fixture, 0);
      const partner = slotSequence(fixture, 1);
      if (!user.includes(1)) return 'the user never appeared in slot 1';
      if (!partner.includes(0)) return 'the partner never appeared in slot 0';
      if (!logHas(session, /\|swap\|p1a: s0\|1\|/)) return 'the swap was not attributed to p1a';
      return null;
    },
  },
  {
    name: 'allyswitch_restart_gate_succeeds',
    p1: partnerTeam,
    p2: foeTeam,
    script: [
      {p1: ['allyswitch', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'allyswitch'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'allyswitch'},
    verify(fixture, session) {
      // The restart roll succeeds only on a zero draw (chance 1/3), so the
      // second use swaps back and the recorded slot sequence returns to 0.
      if (swapCount(session) !== 2) return `expected two swaps, saw ${swapCount(session)}`;
      const user = slotSequence(fixture, 0);
      const first = user.indexOf(1);
      if (first < 0) return 'the user never appeared in slot 1';
      if (!user.slice(first + 1).includes(0)) return 'the user never returned to slot 0';
      return null;
    },
  },
  {
    name: 'allyswitch_restart_gate_fails',
    p1: partnerTeam,
    p2: foeTeam,
    script: [
      {p1: ['allyswitch', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'allyswitch'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'allyswitch'},
    verify(fixture, session) {
      if (swapCount(session) !== 1) return `expected one swap, saw ${swapCount(session)}`;
      // The preparation failure prints the bare fail line for the slot-1 user.
      if (!logHas(session, /\|-fail\|p1b: s0/)) return 'the restart never failed for p1b';
      const user = slotSequence(fixture, 0);
      const first = user.indexOf(1);
      if (first < 0) return 'the user never appeared in slot 1';
      if (user.slice(first + 1).includes(0)) return 'the failed restart still swapped back';
      return null;
    },
  },
  {
    name: 'allyswitch_fails_with_fainted_partner',
    p1: partnerTeam,
    p2: foeTeam,
    script: [{
      // The partner must not Protect: First Impression has to land and KO it
      // before the slower +2 Ally Switch resolves.
      p1: ['allyswitch', {move: 'nightslash', target: 1}],
      // Relative locations: for a P2 slot, `+2` is P1's slot 1.
      p2: [{move: 'firstimpression', target: 2}, 'protect'],
    }],
    coverage: {move: 'allyswitch'},
    verify(fixture, session) {
      const log = session.battle.log;
      const faint = log.findIndex(line => /\|faint\|p1b: s1/.test(line));
      const fail = log.findIndex(line => /\|-fail\|p1a: s0\|move: Ally Switch/.test(line));
      if (faint < 0) return 'the partner never fainted';
      if (fail < 0) return 'Ally Switch never reported the onHit failure';
      if (faint > fail) return 'the partner fainted after the Ally Switch step';
      if (swapCount(session) !== 0) return `a swap happened despite the fainted partner (${swapCount(session)})`;
      const user = slotSequence(fixture, 0);
      if (user.includes(1)) return 'the user still changed slots';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 7200, artifact: 'more_allyswitch.json', debugEnv: 'DEBUG_ALLYSWITCH'});
