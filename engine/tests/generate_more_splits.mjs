// Development-only interaction corpus for the swap/split family:
//
// - Power Trick adds a self volatile that swaps the user's stored Attack and
//   Defense; using it again removes the volatile and swaps them back.
// - Power Split / Guard Split set both sides' stored Attack and Special Attack
//   (Defense and Special Defense) to their floored average.
// - Magnetic Flux raises Defense and Special Defense of every Plus/Minus holder
//   on the user's side.
//
// Every fixture is a complete legal reference battle recorded at every decision
// boundary; the fixtures carry each Pokémon's stored stats, which is the direct
// witness for the two stat moves and the swap volatile.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, foeTeam, foeWith, logHas, everHas, runTrials} = createScaffold();

const team = (...heads) => [
  ...heads,
  ...foeTeam().filter(p => !heads.some(head => head.species === p.species)),
].slice(0, 6);
const statsAt = (fixture, side, roster) => fixture.steps.flatMap(step => step.expected.sides[side].pokemon
  .filter(p => p.roster === roster).map(p => p.stats));
const boostsAt = (fixture, side, roster) => fixture.steps.flatMap(step => step.expected.sides[side].pokemon
  .filter(p => p.roster === roster).map(p => p.boosts));
const volatilesAt = (fixture, side, roster) => fixture.steps.flatMap(step => step.expected.sides[side].pokemon
  .filter(p => p.roster === roster).map(p => p.volatiles));

const TRIALS = [
  {
    // Power Trick swaps the user's stored Attack and Defense, and using it
    // again cancels the marker (swapping them back).
    name: 'powertrick_swaps_and_cancels',
    p1: () => team(setOf('Forretress', 'Sturdy', ['Power Trick', 'Protect'])),
    p2: () => foeTeam(),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['powertrick', 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'powertrick', target: 0}, 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'powertrick'},
    verify(fixture, session) {
      // Boundary states repeat within a turn, so index by the recorded marker
      // rather than by step order: the swap is visible while the volatile is
      // up and gone again after the cancelling re-use.
      const boundaries = fixture.steps.map(step => step.expected.sides[0].pokemon[0]);
      const base = boundaries[0].stats;
      const marked = boundaries.filter(mon => mon.volatiles.includes('powertrick'));
      const swapped = marked.find(mon => mon.stats[1] === base[2] && mon.stats[2] === base[1]);
      const cancelled = boundaries.find(mon => !mon.volatiles.includes('powertrick')
        && mon.stats[1] === base[1] && mon.stats[2] === base[2]
        && boundaries.indexOf(mon) > boundaries.indexOf(marked[0] ?? boundaries[0]));
      return marked.length > 0 && swapped && cancelled
        ? null
        : `power trick not observed (base=${base} marked=${marked.length} swapped=${JSON.stringify(swapped?.stats)} cancelled=${JSON.stringify(cancelled?.stats)})`;
    },
  },
  {
    // Guard Split sets both sides' stored Defense and Special Defense to the
    // floored average of the two.
    name: 'guardsplit_averages_bulk',
    p1: () => team(setOf('Bastiodon', 'Sturdy', ['Guard Split', 'Protect'])),
    p2: () => foeWith(setOf('Chimecho', 'Levitate', ['Psychic', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'guardsplit', target: 1}, 'protect'], p2: [{move: 'psychic', target: 1}, 'protect']},
    ],
    coverage: {move: 'guardsplit'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Guard Split\|/)) return 'Guard Split never executed';
      const mine = statsAt(fixture, 0, 0);
      const theirs = statsAt(fixture, 1, 0);
      if (mine.length < 2 || theirs.length < 2) return 'not enough boundaries';
      const expectedDef = Math.floor((mine[0][2] + theirs[0][2]) / 2);
      const expectedSpd = Math.floor((mine[0][4] + theirs[0][4]) / 2);
      // A later switch-out recomputes the base stats, so look for any boundary
      // that shows the averaged values rather than the final one.
      const after = mine.find(stats => stats[2] === expectedDef && stats[4] === expectedSpd);
      const foeAfter = theirs.find(stats => stats[2] === expectedDef && stats[4] === expectedSpd);
      return after && foeAfter
        ? null
        : `guard split not observed (def ${JSON.stringify(mine)} vs ${expectedDef}, spd ${expectedSpd})`;
    },
  },
  {
    // Power Split is Guard Split's offensive mirror (Attack and Sp. Atk).
    name: 'powersplit_averages_offense',
    p1: () => team(setOf('Cofagrigus', 'Mummy', ['Power Split', 'Protect'])),
    p2: () => foeWith(setOf('Chimecho', 'Levitate', ['Psychic', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'powersplit', target: 1}, 'protect'], p2: [{move: 'psychic', target: 1}, 'protect']},
    ],
    coverage: {move: 'powersplit'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Power Split\|/)) return 'Power Split never executed';
      const mine = statsAt(fixture, 0, 0);
      const theirs = statsAt(fixture, 1, 0);
      if (mine.length < 2 || theirs.length < 2) return 'not enough boundaries';
      const expectedAtk = Math.floor((mine[0][1] + theirs[0][1]) / 2);
      const expectedSpa = Math.floor((mine[0][3] + theirs[0][3]) / 2);
      const after = mine.find(stats => stats[1] === expectedAtk && stats[3] === expectedSpa);
      const foeAfter = theirs.find(stats => stats[1] === expectedAtk && stats[3] === expectedSpa);
      return after && foeAfter
        ? null
        : `power split not observed (atk ${JSON.stringify(mine)} vs ${expectedAtk}, spa ${expectedSpa})`;
    },
  },
  {
    // Magnetic Flux raises Defense and Special Defense of every Plus/Minus
    // holder on the user's side.
    name: 'magneticflux_boosts_holders',
    p1: () => team(
      setOf('Ampharos', 'Plus', ['Magnetic Flux', 'Protect']),
      setOf('Manectric', 'Minus', ['Thunderbolt', 'Protect']),
    ),
    p2: () => foeTeam(),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['magneticflux', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'magneticflux'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Magnetic Flux\|/)) return 'Magnetic Flux never executed';
      const user = boostsAt(fixture, 0, 0);
      const ally = boostsAt(fixture, 0, 1);
      const raised = boosts => boosts.some(b => b[1] >= 1 && b[3] >= 1);
      return raised(user) && raised(ally)
        ? null
        : `magnetic flux boosts not observed (user=${JSON.stringify(user.slice(0, 3))} ally=${JSON.stringify(ally.slice(0, 3))})`;
    },
  },
];

runTrials(TRIALS, {seedBase: 9900, artifact: 'more_splits.json', debugEnv: 'DEBUG_SPLITS'});
