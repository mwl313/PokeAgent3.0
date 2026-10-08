// Development-only interaction corpus for the wish family:
//
// - Wish leaves a slot condition that heals the *slot's* occupant for half of
//   the wisher's maximum HP at the following turn's residual; a second Wish
//   while the marker is up fails.
// - Healing Wish faints the user and fully heals (and cures) the replacement
//   that enters its slot; with no reserve the move is refused and the user
//   stays alive.
// - Heal Bell cures the whole party of the user's side, skipping Soundproof and
//   Good as Gold allies, and fails when nobody was cured.
//
// Every fixture is a complete legal reference battle recorded at every decision
// boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, foeTeam, foeWith, logHas, everHas, runTrials} = createScaffold();

const team = (...heads) => [
  ...heads,
  ...foeTeam().filter(p => !heads.some(head => head.species === p.species)),
].slice(0, 6);
const hpAt = (fixture, side, roster) => fixture.steps.flatMap(step => step.expected.sides[side].pokemon
  .filter(p => p.roster === roster).map(p => p.hp));
const statusAt = (fixture, side, roster) => fixture.steps.flatMap(step => step.expected.sides[side].pokemon
  .filter(p => p.roster === roster).map(p => p.status));
const volatilesAt = (fixture, side, roster) => fixture.steps.flatMap(step => step.expected.sides[side].pokemon
  .filter(p => p.roster === roster).map(p => p.volatiles));
/// Protocol lines inside one turn (between its `|turn|N` marker and the next).
const turnLog = (log, turn) => {
  const start = log.findIndex(line => line === `|turn|${turn}`);
  if (start < 0) return [];
  const end = log.findIndex((line, i) => i > start && line.startsWith('|turn|'));
  return log.slice(start, end < 0 ? undefined : end);
};
// Only the raw-scale lines count: the Champions log mirrors every HP change
// on a 0-100 scale as well.
const wishHealsInTurn = (log, turn) => turnLog(log, turn)
  .filter(line => line.startsWith('|-heal|') && line.includes('[from] move: Wish')
    && !/\|\d+\/100/.test(line)).length;

const TRIALS = [
  {
    // The marker heals the slot's occupant at the next turn's residual.
    name: 'wish_heals_slot_at_next_residual',
    p1: () => team(setOf('Chimecho', 'Levitate', ['Wish', 'Protect'])),
    p2: () => foeTeam(),
    script: [
      // The wisher takes a hit the same turn it wishes, so the next residual
      // has something to heal.
      {p1: ['wish', 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'wish'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Wish\|/)) return 'Wish never executed';
      if (!logHas(session, /\|-heal\|p1a: s0\|.*\[from\] move: Wish/)) return 'the Wish never healed its slot';
      const hps = hpAt(fixture, 0, 0);
      if (!hps.some((hp, i) => i > 0 && hp > hps[i - 1])) return 'the heal never appeared in the boundary states';
      return null;
    },
  },
  {
    // A second Wish while the marker is up is refused: the refused duplicate
    // neither heals at once nor refreshes the timer.
    name: 'wish_duplicate_is_refused',
    p1: () => team(setOf('Chimecho', 'Levitate', ['Wish', 'Protect'])),
    p2: () => foeTeam(),
    script: [
      {p1: ['wish', 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: ['wish', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'wish'},
    verify(fixture, session) {
      const uses = session.battle.log.filter(line => line.startsWith('|move|p1a: s0|Wish|')).length;
      if (uses < 2) return 'the second Wish was never attempted';
      if (!logHas(session, /\|move\|p1a: s0\|Wish\|\|\[still\]/)) return 'the duplicate Wish was not refused';
      // The marker from turn 1 resolves at the residual of turn 2; the refused
      // duplicate must not add a heal on its own turn.
      if (wishHealsInTurn(session.battle.log, 1) !== 0) return 'the marker resolved on its own turn';
      if (wishHealsInTurn(session.battle.log, 2) !== 1) return `expected one heal at the end of turn 2, saw ${wishHealsInTurn(session.battle.log, 2)}`;
      return null;
    },
  },
  {
    // The marker follows the slot: the wisher leaves and the replacement is
    // healed at the following residual instead.
    name: 'wish_heals_the_replacement',
    p1: () => [
      setOf('Chimecho', 'Levitate', ['Wish', 'Protect']),
      setOf('Goodra-Hisui', 'Shell Armor', ['Dragon Pulse', 'Protect']),
      setOf('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
      ...foeTeam().filter(p => !['Goodra-Hisui', 'Torterra', 'Chimecho'].includes(p.species)),
    ].slice(0, 6),
    p2: () => foeTeam(),
    script: [
      // Damage the reserve-to-be, bench it, wish, then bring it back into the
      // wished slot.
      {p1: ['protect', {move: 'dragonpulse', target: 1}], p2: [{move: 'ironhead', target: 2}, 'protect']},
      {p1: ['wish', {switch: 's2'}], p2: ['protect', 'protect']},
      {p1: [{switch: 's1'}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'wish'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Wish\|/)) return 'Wish never executed';
      if (!logHas(session, /\|-heal\|p1a: s1\|.*\[from\] move: Wish/)) return 'the replacement was never healed';
      const s1 = hpAt(fixture, 0, 1);
      const max = Math.max(...s1);
      return s1.some(hp => hp === max) && s1.some(hp => hp < max) ? null : `replacement never healed (${s1})`;
    },
  },
  {
    // Healing Wish faints the user; its replacement enters fully healed.
    name: 'healingwish_heals_the_replacement',
    p1: () => [
      setOf('Chimecho', 'Levitate', ['Healing Wish', 'Protect']),
      setOf('Goodra-Hisui', 'Shell Armor', ['Dragon Pulse', 'Protect']),
      setOf('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
      ...foeTeam().filter(p => !['Goodra-Hisui', 'Torterra', 'Chimecho'].includes(p.species)),
    ].slice(0, 6),
    p2: () => foeTeam(),
    script: [
      // Damage the reserve-to-be, then bench it.
      {p1: ['protect', {move: 'dragonpulse', target: 1}], p2: [{move: 'ironhead', target: 2}, 'protect']},
      {p1: ['protect', {switch: 's2'}], p2: ['protect', 'protect']},
      {p1: ['healingwish', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'healingwish'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Healing Wish\|/)) return 'Healing Wish never executed';
      if (!logHas(session, /\|faint\|p1a: s0/)) return 'the user never fainted';
      const s1 = hpAt(fixture, 0, 1);
      const max = Math.max(...s1);
      const damaged = s1.some(hp => hp < max);
      const restored = s1.some((hp, i) => i > 0 && hp > s1[i - 1]);
      return damaged && restored ? null : `replacement heal not observed (${s1})`;
    },
  },
  {
    // Heal Bell cures the whole party, including a benched member.
    name: 'healbell_cures_bench_and_active',
    p1: () => team(setOf('Chimecho', 'Levitate', ['Heal Bell', 'Protect'])),
    p2: () => foeWith(setOf('Beedrill', 'Swarm', ['Toxic', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: [{move: 'toxic', target: 1}, 'protect']},
      {p1: ['protect', {switch: 's2'}], p2: ['protect', 'protect']},
      {p1: ['healbell', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'healbell'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Heal Bell\|/)) return 'Heal Bell never executed';
      const cured = session.battle.log.filter(line => line.startsWith('|-curestatus|p1')).length;
      if (cured < 2) return `expected the active and the benched member to be cured (${cured})`;
      const statuses = fixture.steps.at(-1).expected.sides[0].pokemon
        .filter(p => p.roster < 4).map(p => p.status);
      return statuses.every(status => status === 0) ? null : `statuses remain: ${statuses}`;
    },
  },
  {
    // Heal Bell fails when nobody on the user's side has a status.
    name: 'healbell_fails_without_status',
    p1: () => team(setOf('Chimecho', 'Levitate', ['Heal Bell', 'Protect'])),
    p2: () => foeTeam(),
    script: [
      {p1: ['healbell', 'protect'], p2: ['protect', 'protect']},
      {p1: ['healbell', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'healbell'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Heal Bell\|/)) return 'Heal Bell never executed';
      if (logHas(session, /\|-curestatus\|/)) return 'a cure appeared without any status';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 8900, artifact: 'more_wishes.json', debugEnv: 'DEBUG_WISHES'});
