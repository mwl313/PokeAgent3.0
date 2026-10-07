// Development-only corpus for Destiny Bond:
// - the foe's KO drags the move's source down with it (`onFaint`),
// - the bond is dropped before the holder's next non-Destiny-Bond action,
// - attempting Destiny Bond while the bond is already up removes it and
//   fails (a second consecutive use only clears the bond).
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, logHas, monAt, runTrials} = createScaffold();

const P2_FILL = [
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']],
  ['Falinks', 'Battle Armor', ['Smart Strike', 'Protect']],
  ['Aggron', 'Sturdy', ['Iron Head', 'Protect']],
  ['Milotic', 'Competitive', ['Surf', 'Protect']],
  ['Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']],
];
const withFill = heads => [
  ...heads,
  ...P2_FILL.filter(([species]) => !heads.some(h => h.species === species))
    .map(([species, ability, moves]) => offensive(species, ability, moves)),
].slice(0, 6);

const gengar = () => offensive('Gengar', 'Cursed Body', ['Protect', 'Destiny Bond', 'Shadow Ball']);
const psychics = () => offensive('Metagross', 'Clear Body', ['Psychic', 'Protect']);
const cruncher = () => offensive('Snorlax', 'Thick Fat', ['Crunch', 'Protect']);

const TRIALS = [
  {
    name: 'destinybond_takes_foe_down',
    // Both foes jump the frail holder in the same turn; it is faster, so the
    // bond is up before either hit, and the killer goes down with it.
    p1: () => withFill([gengar()]),
    p2: () => withFill([psychics(), cruncher()]),
    script: [
      {p1: ['destinybond', 'protect'], p2: [{move: 'psychic', target: 1}, {move: 'crunch', target: 1}]},
      {p1: ['shadowball', 'protect'], p2: [{move: 'psychic', target: 1}, {move: 'crunch', target: 1}]},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'destinybond'},
    verify(fixture, session) {
      if (!logHas(session, /\|-singlemove\|p1a: s0\|Destiny Bond/)) return 'the bond was never shown';
      if (!logHas(session, /\|-activate\|p1a: s0\|move: Destiny Bond/)) return 'the bond never activated';
      const p1Down = monAt(fixture, 0, 0).some(p => p.fainted);
      if (!p1Down) return 'the holder never fainted';
      const killerDown = monAt(fixture, 1, 1).some(p => p.fainted);
      if (!killerDown) return 'the move source did not go down with the holder';
      if (!monAt(fixture, 1, 0).some(p => p.hp > 0)) return 'the non-source foe fainted unexpectedly';
      return null;
    },
  },
  {
    name: 'destinybond_drops_before_next_move',
    // The holder survives the first hit, opens the next turn with an attack:
    // that attempt drops the bond before the foe's KO lands.
    p1: () => withFill([gengar()]),
    p2: () => withFill([psychics(), cruncher()]),
    script: [
      {p1: ['destinybond', 'protect'], p2: ['protect', 'protect']},
      {p1: ['shadowball', 'protect'], p2: [{move: 'psychic', target: 1}, {move: 'crunch', target: 1}]},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'destinybond'},
    verify(fixture, session) {
      if (!logHas(session, /\|-singlemove\|p1a: s0\|Destiny Bond/)) return 'the bond was never shown';
      if (logHas(session, /\|-activate\|p1a: s0\|move: Destiny Bond/)) {
        return 'the dropped bond still activated';
      }
      if (!monAt(fixture, 0, 0).some(p => p.fainted)) return 'the holder never fainted';
      if (!monAt(fixture, 1, 1).some(p => p.hp > 0)) return 'the source fainted without the bond';
      return null;
    },
  },
  {
    name: 'destinybond_replace_fails',
    // A second consecutive Destiny Bond only removes the existing bond and
    // fails; the later KO must not drag the source down.
    p1: () => withFill([gengar()]),
    p2: () => withFill([psychics(), cruncher()]),
    script: [
      {p1: ['destinybond', 'protect'], p2: ['protect', 'protect']},
      {p1: ['destinybond', 'protect'], p2: [{move: 'psychic', target: 1}, {move: 'crunch', target: 1}]},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'destinybond'},
    verify(fixture, session) {
      if (!logHas(session, /\|-singlemove\|p1a: s0\|Destiny Bond/)) return 'the bond was never shown';
      if (!logHas(session, /\|-fail\|p1a: s0/)) return 'the replacement attempt never failed';
      if (logHas(session, /\|-activate\|p1a: s0\|move: Destiny Bond/)) {
        return 'the replaced bond still activated';
      }
      if (!monAt(fixture, 0, 0).some(p => p.fainted)) return 'the holder never fainted';
      if (!monAt(fixture, 1, 0).some(p => p.hp > 0) || !monAt(fixture, 1, 1).some(p => p.hp > 0)) {
        return 'a foe fainted without an active bond';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 21000, artifact: 'more_destinybond.json', debugEnv: 'DEBUG_DESTINYBOND'});
