// Development-only corpus for `abilities:megasol` (Mega Meganium), whose
// `Pokemon#effectiveWeather` override resolves the holder's own moves under
// sun no matter the field weather:
// - Weather Ball becomes Fire (super effective against the Grass foe) instead
//   of Water, and its power is the sun value;
// - Solar Beam's charge is skipped, so the move fires on the turn it is used.
// The paired scene never Mega evolves: the same Weather Ball is Water and is
// resisted by the Grass foe.
import {createScaffold} from './fixture_scaffold.mjs';
const {setOf, foeTeam, logHas, runTrials} = createScaffold();

const team = (...heads) => [
  ...heads,
  ...foeTeam().filter(p => !heads.some(head => head.species === p.species)),
].slice(0, 6);
const foes = () => team(
  setOf('Politoed', 'Drizzle', ['Protect', 'Surf', 'Ice Beam']),
  setOf('Leafeon', 'Chlorophyll', ['Protect', 'Leaf Blade']),
);
const healerTeam = () => team(setOf('Meganium', 'Overgrow', ['Weather Ball', 'Solar Beam', 'Protect'], 'Meganiumite'));
const weatherBallTurn = [
  {p1: [{move: 'protect', mega: true}, 'protect'], p2: ['protect', 'protect']},
  // The Grass foe must be hittable: it attacks the partner instead.
  {p1: [{move: 'weatherball', target: 2}, 'protect'], p2: [{move: 'surf'}, {move: 'leafblade', target: 2}]},
];

const TRIALS = [
  {
    name: 'megasol_weatherball_is_fire_and_solarbeam_skips_charge',
    p1: healerTeam,
    p2: foes,
    script: [
      ...weatherBallTurn,
      // Solar Beam fires on this turn only if the charge is skipped.
      {p1: [{move: 'solarbeam', target: 2}, 'protect'], p2: [{move: 'surf'}, {move: 'leafblade', target: 2}]},
    ],
    coverage: {move: 'weatherball'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Weather Ball\|/)) return 'Weather Ball never executed';
      if (!logHas(session, /\|-supereffective\|p2b: s1\|/)) return 'Weather Ball was not Fire-type';
      const lines = session.battle.log;
      const beam = lines.findIndex(line => line.startsWith('|move|p1a: s0|Solar Beam|'));
      if (beam < 0) return 'Solar Beam never executed';
      const next = lines.slice(beam + 1).findIndex(line =>
        line.startsWith('|move|') || line.startsWith('|upkeep|'));
      const window = lines.slice(beam + 1, next < 0 ? undefined : beam + 1 + next);
      if (!window.some(line => line.startsWith('|-damage|p2b:'))) {
        return 'Solar Beam spent a charge turn under Mega Sol';
      }
      return null;
    },
  },
  {
    name: 'without_megasol_weatherball_is_water',
    p1: healerTeam,
    p2: foes,
    // The control never Mega evolves, so Mega Sol is not active.
    script: [
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'weatherball', target: 2}, 'protect'], p2: [{move: 'surf'}, {move: 'leafblade', target: 2}]},
    ],
    coverage: {move: 'weatherball'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Weather Ball\|/)) return 'Weather Ball never executed';
      if (!logHas(session, /\|-resisted\|p2b: s1\|/)) return 'Weather Ball was not Water-type';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 36000, artifact: 'more_megasol.json', debugEnv: 'DEBUG_MEGASOL'});
