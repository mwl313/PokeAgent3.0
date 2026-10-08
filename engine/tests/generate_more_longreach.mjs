// Development-only corpus for Long Reach: the holder's moves delete their
// contact flag, so contact-gated abilities (Rough Skin here) never fire against
// it. The scene keeps a partner without Long Reach using the same contact move
// so the reference log proves both branches in one battle.
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

const TRIALS = [
  {
    name: 'longreach_suppresses_contact_recoil',
    p1: () => team(
      setOf('Decidueye', 'Long Reach', ['Leaf Blade', 'Protect']),
      offensive('Aggron', 'Sturdy', ['Iron Head', 'Protect']),
    ),
    p2: () => team(setOf('Garchomp', 'Rough Skin', ['Protect', 'Earthquake'])),
    script: [
      // Both P1 slots use a contact move against the Rough Skin holder: only
      // the partner without Long Reach takes the recoil.
      {
        p1: [{move: 'leafblade', target: 1}, {move: 'ironhead', target: 1}],
        p2: ['protect', 'protect'],
      },
      {
        p1: [{move: 'leafblade', target: 1}, {move: 'ironhead', target: 1}],
        p2: ['protect', 'protect'],
      },
    ],
    coverage: {ability: 'longreach'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Leaf Blade\|p2a: s0/)) return 'the Long Reach holder never attacked';
      if (!logHas(session, /\|move\|p1b: s1\|Iron Head\|p2a: s0/)) return 'the control attacker never attacked';
      if (!logHas(session, /\|-damage\|p1b: s1\|.*\[from\] ability: Rough Skin/)) {
        return 'the control attacker took no Rough Skin recoil';
      }
      if (logHas(session, /\|-damage\|p1a: s0\|.*\[from\] ability: Rough Skin/)) {
        return 'the Long Reach holder still took contact recoil';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 44000, artifact: 'more_longreach.json', debugEnv: 'DEBUG_LONGREACH'});
