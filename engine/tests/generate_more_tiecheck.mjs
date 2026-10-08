// Development-only probe corpus for queue speed ties: mirrored, identical sets
// on both sides make every lead (and partner) act at exactly the same speed, so
// the queue's Fischer-Yates tie-break decides the action order.
import {createScaffold} from './fixture_scaffold.mjs';
const {setOf, foeTeam, runTrials} = createScaffold();

const mirror = () => [
  setOf('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
  ...foeTeam().filter(p => p.species !== 'Snorlax'),
].slice(0, 6);
const mirrorTwo = () => [
  setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
  setOf('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
  ...foeTeam().filter(p => p.species !== 'Metagross' && p.species !== 'Snorlax'),
].slice(0, 6);

const TRIALS = [
  {
    name: 'tiecheck_mirrored_leads',
    p1: mirror,
    p2: mirror,
    script: [
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: [{move: 'bodyslam', target: 1}, 'protect']},
    ],
    coverage: {ability: 'tiecheck'},
    verify: () => null,
  },
  {
    name: 'tiecheck_mirrored_partners',
    p1: mirrorTwo,
    p2: mirrorTwo,
    script: [
      {p1: [{move: 'ironhead', target: 1}, {move: 'bodyslam', target: 1}], p2: [{move: 'ironhead', target: 1}, {move: 'bodyslam', target: 1}]},
    ],
    coverage: {ability: 'tiecheck'},
    verify: () => null,
  },
];

runTrials(TRIALS, {seedBase: 8400, artifact: 'more_tiecheck.json', debugEnv: 'DEBUG_TIECHECK'});
