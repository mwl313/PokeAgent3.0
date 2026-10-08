// Development-only corpus for Illusion. The world-state `compact()` cannot see
// the disguise (the reference keeps the real species internally), so each
// boundary additionally records the *observed* identity the protocol displays
// for every active slot: the ident line of the most recent `switch`/`drag`/
// `replace` message, i.e. the masked species until a damaging hit ends it.
//
// engine/tests/illusion.rs replays these fixtures and compares the native
// per-viewer knowledge against the recorded surface.
import {createScaffold} from './fixture_scaffold.mjs';
const {setOf, offensive, logHas, runTrials} = createScaffold();

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

// `|switch|p1a: s3|Torterra, L50, M|167/167` -> p1a displays Torterra.
// `|replace|p1a: s0|Zoroark, L50, M`      -> p1a displays Zoroark again.
const observedIdentity = session => {
  const slots = {};
  for (const line of session.battle.log) {
    const match = line.match(/^\|(?:switch|drag|replace)\|(p[12][ab]): [^|]+\|([^,|]+)/);
    if (match) slots[match[1]] = match[2].trim();
  }
  return slots.p1a || slots.p2a || slots.p1b || slots.p2b
    ? {p1: [slots.p1a ?? null, slots.p1b ?? null], p2: [slots.p2a ?? null, slots.p2b ?? null]}
    : null;
};

const TRIALS = [
  {
    name: 'illusion_masks_until_a_damaging_hit',
    // Zoroark-Hisui leads at p1a; the disguise is the last party member to its
    // right, and Iron Head's hit reveals the real identity.
    p1: () => team(
      setOf('Zoroark-Hisui', 'Illusion', ['Shadow Ball', 'Protect']),
      offensive('Milotic', 'Competitive', ['Ice Beam', 'Protect']),
    ),
    p2: () => team(setOf('Aggron', 'Sturdy', ['Iron Head', 'Protect'])),
    observe: observedIdentity,
    script: [
      {p1: [{move: 'shadowball', target: 1}, {move: 'icebeam', target: 1}], p2: ['protect', 'protect']},
      {p1: [{move: 'shadowball', target: 1}, {move: 'icebeam', target: 1}], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: [{move: 'shadowball', target: 1}, {move: 'icebeam', target: 1}], p2: [{move: 'ironhead', target: 1}, 'protect']},
    ],
    coverage: {ability: 'illusion'},
    verify(fixture, session) {
      const first = fixture.steps[fixture.steps.length - 1]?.observed?.p1?.[0];
      if (!first) return 'no masked identity was recorded';
      if (!logHas(session, /\|replace\|p1a: /)) return 'the disguise was never replaced';
      // The disguise is the last selected party member to the holder's right
      // (team preview picks entries 0-3, so that is Torterra, s3).
      const masked = fixture.steps.some(step => step.observed?.p1?.[0] === 'Torterra');
      if (!masked) return 'the disguise never displayed the partner species';
      const revealed = fixture.steps.some(step => step.observed?.p1?.[0] === 'Zoroark-Hisui');
      if (!revealed) return 'the real identity never became public';
      return null;
    },
  },
  {
    name: 'illusion_survives_a_switch_out',
    // Scene B from the scout note: leaving the field with the disguise intact
    // emits no `replace`, and re-entering recomputes it.
    p1: () => team(
      setOf('Zoroark', 'Illusion', ['Dark Pulse', 'Protect']),
      offensive('Milotic', 'Competitive', ['Ice Beam', 'Protect']),
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
    ),
    p2: () => team(setOf('Aggron', 'Sturdy', ['Iron Head', 'Protect'])),
    observe: observedIdentity,
    script: [
      // Leave the field immediately, before any damaging hit can reveal it.
      {p1: [{switch: 's2'}, {move: 'icebeam', target: 1}], p2: [{move: 'ironhead', target: 2}, 'protect']},
      {p1: [{move: 'ironhead', target: 1}, {move: 'icebeam', target: 1}], p2: [{move: 'ironhead', target: 1}, 'protect']},
    ],
    coverage: {ability: 'illusion'},
    verify(fixture, session) {
      // The lead's own switch-in is masked: the displayed partner is the last
      // selected party member to its right, so the ident is not `s0`.
      if (!logHas(session, /\|switch\|p1a: s[1-5]\|Torterra/)) {
        return 'the lead was never masked as the partner';
      }
      // Leaving with the disguise intact announces nothing: no replacement
      // line anywhere before the holder re-enters (it does not in this scene).
      const replaces = session.battle.log.filter(line => line.startsWith('|replace|')).length;
      if (replaces !== 0) return `a switch-out announced a replacement ${replaces} time(s)`;
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 50000, artifact: 'more_illusion.json', debugEnv: 'DEBUG_ILLUSION'});
