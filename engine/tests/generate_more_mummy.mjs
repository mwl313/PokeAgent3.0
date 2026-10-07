// Development-only corpus for Mummy:
// - a contact hit overwrites the attacker's ability with Mummy,
// - a non-contact hit leaves the attacker alone, and
// - an attacker whose ability carries `flags.cantsuppress` (Stance Change)
//   keeps its ability.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, fillerAfter, foeTeam, foeWith, logHas, everHas, runTrials} = createScaffold();

const TRIALS = [
  {
    name: 'mummy_replaces_attacker_ability_on_contact',
    p1: () => fillerAfter(setOf('Cofagrigus', 'Mummy', ['Shadow Ball', 'Protect'])),
    p2: () => foeWith(offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect'])),
    script: [{p1: ['shadowball', 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']}],
    coverage: {ability: 'mummy'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Iron Head\|p1a: s0/)) return 'the contact hit never landed';
      if (!everHas(fixture, 1, 0, p => p.ability === ids.abilities.mummy)) {
        return 'the attacker ability was never overwritten with Mummy';
      }
      if (everHas(fixture, 0, 0, p => p.ability !== ids.abilities.mummy)) {
        return 'the Mummy holder lost its own ability';
      }
      return null;
    },
  },
  {
    name: 'mummy_ignores_non_contact_hits',
    p1: () => fillerAfter(setOf('Cofagrigus', 'Mummy', ['Shadow Ball', 'Protect'])),
    p2: () => foeWith(offensive('Milotic', 'Competitive', ['Surf', 'Protect'])),
    script: [{p1: ['shadowball', 'protect'], p2: ['surf', 'protect']}],
    coverage: {ability: 'mummy'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Surf\|/)) return 'the non-contact hit never landed';
      if (everHas(fixture, 1, 0, p => p.ability !== ids.abilities.competitive)) {
        return 'a non-contact hit overwrote the attacker ability';
      }
      return null;
    },
  },
  {
    name: 'mummy_skips_cantsuppress_abilities',
    p1: () => fillerAfter(setOf('Cofagrigus', 'Mummy', ['Shadow Ball', 'Protect'])),
    p2: () => foeWith(offensive('Aegislash', 'Stance Change', ['Iron Head', 'Protect'])),
    script: [{p1: ['shadowball', 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']}],
    coverage: {ability: 'mummy'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Iron Head\|p1a: s0/)) return 'the contact hit never landed';
      if (everHas(fixture, 1, 0, p => p.ability !== ids.abilities.stancechange)) {
        return 'a cantsuppress ability was overwritten with Mummy';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 7000, artifact: 'more_mummy.json', debugEnv: 'DEBUG_MUMMY'});
