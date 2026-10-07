// Development-only corpus for Raging Bull, the one move that combines a
// TryHit stage screen shatter with a form-driven type override:
// - `moves:ragingbull.onTryHit` removes Reflect / Light Screen / Aurora Veil
//   from the target side inside the TryHit step, i.e. before type immunity,
//   accuracy and the decoy intercept, but after every higher-priority TryHit
//   handler (protection guards, ability absorptions);
// - `moves:ragingbull.onModifyType` gives the three Paldea Tauros forms their
//   primary type (Combat/Fighting, Blaze/Fire, Aqua/Water); plain Tauros and
//   any other caller keep Normal.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, foeTeam, logHas, runTrials} = createScaffold();
const idsReflect = ids.conditions.reflect;

const team = (...heads) => [
  ...heads,
  ...foeTeam().filter(p => !heads.some(head => head.species === p.species)),
].slice(0, 6);
// The reference `-sideend` shapes: Reflect keeps its plain name while the
// Light Screen condition displays as `move: Light Screen`.
const sideEnd = (log, from, kind) => log.findIndex((line, i) =>
  i > from && line.startsWith('|-sideend|p2') && line.includes(kind));

const TRIALS = [
  {
    // The screens shatter even though the substitute eats the hit, and the
    // target behind the decoy keeps every point of HP.
    name: 'ragingbull_blaze_shatters_screens_through_substitute',
    p1: () => team(setOf('Tauros-Paldea-Blaze', 'Intimidate', ['Raging Bull', 'Protect', 'Zen Headbutt'])),
    p2: () => team(
      setOf('Torterra', 'Shell Armor', ['Reflect', 'Substitute', 'Body Slam']),
      setOf('Metagross', 'Clear Body', ['Light Screen', 'Protect', 'Iron Head']),
    ),
    script: [
      {p1: ['protect', 'protect'], p2: [{move: 'reflect'}, {move: 'lightscreen'}]},
      {p1: ['protect', 'protect'], p2: [{move: 'substitute'}, 'protect']},
      // The decoy holder attacks the protecting partner, so nothing blocks
      // the shatter this turn.
      {p1: [{move: 'ragingbull', target: 1}, 'protect'], p2: [{move: 'bodyslam', target: 1}, 'protect']},
    ],
    coverage: {move: 'ragingbull'},
    verify(fixture, session) {
      const log = session.battle.log;
      if (!logHas(session, /\|move\|p1a: s0\|Raging Bull\|/)) return 'Raging Bull never executed';
      const move = log.findIndex(line => line.startsWith('|move|p1a: s0|Raging Bull|'));
      const reflect = sideEnd(log, move, 'Reflect');
      const light = sideEnd(log, move, 'Light Screen');
      if (reflect < 0 || light < 0) return 'the screens were not shattered through the substitute';
      // The decoy costs HP when it goes up, so compare the target across the
      // shatter boundary instead of against its maximum.
      const hpAt = step => step.expected.sides[1].pokemon.find(p => p.roster === 0).hp;
      const hasReflect = step => step.expected.sides[1].conditions.some(([id]) => id === idsReflect);
      const cast = fixture.steps.findIndex(hasReflect);
      const shatter = fixture.steps.findIndex((step, i) => i > cast && !hasReflect(step));
      if (cast < 0 || shatter < 1) return 'the shatter never showed in the recorded boundaries';
      if (hpAt(fixture.steps[shatter]) !== hpAt(fixture.steps[shatter - 1])) {
        return `the decoy let damage through (${hpAt(fixture.steps[shatter])} after ${hpAt(fixture.steps[shatter - 1])})`;
      }
      return null;
    },
  },
  {
    // Without a decoy the Fire-type hit lands: the two `-sideend` messages
    // must precede the damage line of the same move, i.e. the shatter ran in
    // the TryHit step rather than after the damage calculation.
    name: 'ragingbull_blaze_shatters_screens_before_fire_damage',
    p1: () => team(setOf('Tauros-Paldea-Blaze', 'Intimidate', ['Raging Bull', 'Protect', 'Zen Headbutt'])),
    p2: () => team(
      setOf('Torterra', 'Shell Armor', ['Reflect', 'Body Slam', 'Protect']),
      setOf('Metagross', 'Clear Body', ['Light Screen', 'Protect', 'Iron Head']),
    ),
    script: [
      {p1: ['protect', 'protect'], p2: [{move: 'reflect'}, {move: 'lightscreen'}]},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'ragingbull', target: 1}, 'protect'], p2: [{move: 'bodyslam', target: 1}, 'protect']},
    ],
    coverage: {move: 'ragingbull'},
    verify(fixture, session) {
      const log = session.battle.log;
      if (!logHas(session, /\|move\|p1a: s0\|Raging Bull\|/)) return 'Raging Bull never executed';
      const move = log.findIndex(line => line.startsWith('|move|p1a: s0|Raging Bull|'));
      const reflect = sideEnd(log, move, 'Reflect');
      const light = sideEnd(log, move, 'Light Screen');
      const damage = log.findIndex((line, i) => i > move && line.startsWith('|-damage|p2a: s0|'));
      if (damage < 0) return 'the Blaze-form hit never damaged the Grass target';
      if (reflect < 0 || light < 0) return 'the screens were not shattered';
      if (!(reflect < damage && light < damage)) return 'the shatter ran after the damage roll';
      if (!logHas(session, /\|-supereffective\|p2a: s0\|/)) return 'the Fire type never showed as super effective';
      return null;
    },
  },
  {
    // Combat form: Fighting is super effective against the pure-Normal foe.
    name: 'ragingbull_combat_form_is_fighting',
    p1: () => team(setOf('Tauros-Paldea-Combat', 'Intimidate', ['Raging Bull', 'Protect', 'Body Press'])),
    p2: () => team(setOf('Snorlax', 'Thick Fat', ['Body Slam', 'Protect', 'Crunch'])),
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'ragingbull', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'ragingbull'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Raging Bull\|/)) return 'Raging Bull never executed';
      if (!logHas(session, /\|-supereffective\|p2a: s0\|/)) return 'the Fighting type never showed as super effective';
      const first = fixture.steps[0].expected.sides[1].pokemon.find(p => p.roster === 0);
      const last = fixture.steps.at(-1).expected.sides[1].pokemon.find(p => p.roster === 0);
      if (last.hp >= first.hp) return 'the target never lost HP';
      return null;
    },
  },
  {
    // Aqua form against Water Absorb: the priority-0 ability handler absorbs
    // and breaks the TryHit event before the move's own handler runs, so the
    // screens survive - the ordering counter-scene for the shatter.
    name: 'ragingbull_aqua_absorbed_keeps_screens',
    p1: () => team(setOf('Tauros-Paldea-Aqua', 'Intimidate', ['Raging Bull', 'Protect', 'Aqua Jet'])),
    p2: () => team(
      setOf('Vaporeon', 'Water Absorb', ['Surf', 'Protect', 'Substitute']),
      setOf('Metagross', 'Clear Body', ['Light Screen', 'Protect', 'Iron Head']),
    ),
    script: [
      {p1: ['protect', 'protect'], p2: [{move: 'surf'}, {move: 'reflect'}]},
      {p1: ['protect', 'protect'], p2: [{move: 'surf'}, {move: 'lightscreen'}]},
      {p1: [{move: 'ragingbull', target: 1}, 'protect'], p2: [{move: 'surf'}, 'protect']},
    ],
    coverage: {move: 'ragingbull'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Raging Bull\|/)) return 'Raging Bull never executed';
      if (!logHas(session, /ability: Water Absorb/)) return 'Water Absorb never absorbed the hit';
      if (logHas(session, /\|-sideend\|p2\|(Reflect|Light Screen)/)) {
        return 'the absorbed move shattered screens the reference keeps';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 26000, artifact: 'more_ragingbull.json', debugEnv: 'DEBUG_RAGINGBULL'});
