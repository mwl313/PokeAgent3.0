// Development-only differential witnesses for two ported pool abilities that no
// other corpus exercises: Sand Rush (x2 Speed while sand is up) and Thick Fat
// (halves incoming Fire/Ice damage). Each pair of scenes contains the ability
// scene plus a control that removes only the ability (no sand / another
// Snorlax ability), so the boundary-by-boundary comparison against the pinned
// reference sees the effect itself rather than a coincidence.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, foeTeam, logHas, runTrials} = createScaffold();

const foeWith = head => [head, ...foeTeam().filter(p => p.species !== head.species)].slice(0, 6);
// Fill the rest of the team without repeating a species (Species Clause).
const withFiller = heads => [...heads, ...foeTeam().filter(p => !heads.some(head => head.species === p.species))].slice(0, 6);
const fast = (species, ability, moves) =>
  setOf(species, ability, moves, '', {hp: 2, atk: 0, def: 0, spa: 0, spd: 0, spe: 32});
const orderOf = (session, move, who) => session.battle.log.findIndex(line => line.startsWith(`|move|${who}|${move}`));

const TRIALS = [
  {
    // Sand Rush: Excadrill (base 88) must outrun Gengar (base 110) while the
    // partner's Sand Stream is up.
    name: 'witness_sandrush_moves_first_in_sand',
    p1: () => withFiller([
      setOf('Excadrill', 'Sand Rush', ['Iron Head', 'Protect'], '', {hp: 2, atk: 0, def: 0, spa: 0, spd: 0, spe: 32}),
      setOf('Tyranitar', 'Sand Stream', ['Rock Slide', 'Protect']),
    ]),
    p2: () => foeWith(fast('Gengar', 'Cursed Body', ['Shadow Ball', 'Protect'])),
    script: [{p1: [{move: 'ironhead', target: 1}, 'protect'], p2: [{move: 'shadowball', target: 1}, 'protect']}],
    coverage: {ability: 'sandrush'},
    verify(fixture, session) {
      if (!session.battle.log.some(line => line.startsWith('|-weather|Sandstorm'))) return 'the sand never started';
      const mine = orderOf(session, 'Iron Head', 'p1a: s0');
      const theirs = orderOf(session, 'Shadow Ball', 'p2a: s0');
      if (mine < 0 || theirs < 0) return 'the turn never resolved both moves';
      if (mine > theirs) return 'the Sand Rush holder did not move first';
      return null;
    },
  },
  {
    // Control: identical scene, no sand, so Gengar's higher base Speed wins.
    name: 'witness_sandrush_control_without_sand',
    p1: () => withFiller([
      setOf('Excadrill', 'Sand Rush', ['Iron Head', 'Protect'], '', {hp: 2, atk: 0, def: 0, spa: 0, spd: 0, spe: 32}),
      setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
    ]),
    p2: () => foeWith(fast('Gengar', 'Cursed Body', ['Shadow Ball', 'Protect'])),
    script: [{p1: [{move: 'ironhead', target: 1}, 'protect'], p2: [{move: 'shadowball', target: 1}, 'protect']}],
    coverage: {ability: 'sandrush'},
    verify(fixture, session) {
      if (session.battle.log.some(line => line.startsWith('|-weather|Sandstorm'))) return 'sand started in the control';
      const mine = orderOf(session, 'Iron Head', 'p1a: s0');
      const theirs = orderOf(session, 'Shadow Ball', 'p2a: s0');
      if (mine < 0 || theirs < 0) return 'the turn never resolved both moves';
      if (mine < theirs) return 'the control already moved first without sand';
      return null;
    },
  },
  {
    // Thick Fat: Snorlax takes a Flamethrower while the ability halves it.
    name: 'witness_thickfat_halves_flamethrower',
    p1: () => withFiller([
      setOf('Snorlax', 'Thick Fat', ['Body Slam', 'Protect'], '', {hp: 32, atk: 0, def: 17, spa: 0, spd: 17, spe: 0}),
      setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
    ]),
    p2: () => foeWith(offensive('Incineroar', 'Blaze', ['Flamethrower', 'Protect'])),
    script: [{p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: [{move: 'flamethrower', target: 1}, 'protect']}],
    coverage: {ability: 'thickfat'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Flamethrower\|/)) return 'the Fire move never executed';
      if (!logHas(session, /\|-damage\|p1a: s0\|/)) return 'the Thick Fat holder took no damage';
      return null;
    },
  },
  {
    // Control: the same scene with the same Snorlax set and a non-halving ability.
    name: 'witness_thickfat_control_immunity',
    p1: () => withFiller([
      setOf('Snorlax', 'Immunity', ['Body Slam', 'Protect'], '', {hp: 32, atk: 0, def: 17, spa: 0, spd: 17, spe: 0}),
      setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
    ]),
    p2: () => foeWith(offensive('Incineroar', 'Blaze', ['Flamethrower', 'Protect'])),
    script: [{p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: [{move: 'flamethrower', target: 1}, 'protect']}],
    coverage: {ability: 'thickfat'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Flamethrower\|/)) return 'the Fire move never executed';
      if (!logHas(session, /\|-damage\|p1a: s0\|/)) return 'the control holder took no damage';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 9400, artifact: 'more_sandrush_thickfat.json', debugEnv: 'DEBUG_SANDRUSH_THICKFAT'});
