// Development-only corpus for two small base-power/on-hit moves:
// - Facade doubles its base power while the user carries a major status
//   (paralysis here, so the burn Attack drop cannot confound the comparison),
// - Burning Jealousy burns each target whose stats were raised this turn.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, logHas, monAt, runTrials} = createScaffold();

const POOL = [
  ['Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic']],
  ['Milotic', 'Competitive', ['Surf', 'Protect', 'Ice Beam']],
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect', 'Body Slam']],
  ['Falinks', 'Battle Armor', ['Smart Strike', 'Protect', 'Close Combat']],
  ['Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect', 'Psychic']],
  ['Aggron', 'Sturdy', ['Iron Head', 'Protect', 'Body Slam']],
  ['Ariados', 'Swarm', ['Leech Life', 'Protect', 'Crunch']],
];
const team = (...heads) => [
  ...heads,
  ...POOL.filter(([species]) => !heads.some(head => head.species === species))
    .map(([species, ability, moves]) => offensive(species, ability, moves)),
].slice(0, 6);

const TRIALS = [
  {
    name: 'smallmoves_facade_doubles_while_paralysed',
    p1: () => team(setOf('Snorlax', 'Thick Fat', ['Facade', 'Protect', 'Body Slam'],
      '', {hp: 26, atk: 20, def: 10, spa: 0, spd: 10, spe: 0})),
    p2: () => team(
      setOf('Ampharos', 'Static', ['Thunder Wave', 'Protect', 'Dragon Pulse']),
      offensive('Milotic', 'Competitive', ['Surf', 'Protect', 'Ice Beam']),
    ),
    script: [
      {p1: [{move: 'facade', target: 2}, 'protect'], p2: ['protect', 'surf']},
      {p1: [{move: 'bodyslam', target: 1}, 'protect'], p2: [{move: 'thunderwave', target: 1}, 'protect']},
      {p1: [{move: 'facade', target: 2}, 'protect'], p2: ['protect', 'surf']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'facade'},
    verify(fixture, session) {
      // The reserve only ever takes the two scripted Facade hits, so the first
      // two HP drops are the healthy hit and the paralysed one.
      const seq = fixture.steps
        .map(step => step.expected.sides[1].pokemon.find(p => p.roster === 1))
        .filter(Boolean)
        .map(p => p.hp);
      const drops = [];
      for (let i = 1; i < seq.length; i++) {
        if (seq[i] < seq[i - 1]) drops.push(seq[i - 1] - seq[i]);
      }
      if (drops.length < 2) return 'the two Facade hits were not both recorded';
      const [plain, doubled] = drops;
      if (doubled * 2 < plain * 3) {
        return `the paralysed Facade did not double (${plain} -> ${doubled})`;
      }
      if (!monAt(fixture, 0, 0).some(p => p.status === ids.conditions.par)) {
        return 'the user never got paralysed';
      }
      if (!logHas(session, /\|move\|p1a: s0\|Facade\|p2b: s1/)) return 'Facade never targeted the reserve';
      return null;
    },
  },
  {
    name: 'smallmoves_burningjealousy_burns_boosted_target',
    p1: () => team(setOf('Torkoal', 'Shell Armor', ['Burning Jealousy', 'Protect', 'Body Press'])),
    p2: () => team(
      setOf('Alakazam', 'Magic Guard', ['Calm Mind', 'Protect', 'Psychic']),
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic']),
    ),
    script: [
      {p1: [{move: 'burningjealousy', target: 0}, 'protect'], p2: [{move: 'calmmind', target: 0}, 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: [{move: 'burningjealousy', target: 0}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'burningjealousy'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Burning Jealousy\|/)) return 'Burning Jealousy never executed';
      if (!monAt(fixture, 1, 0).some(p => p.hp < p.max_hp)) return 'Burning Jealousy dealt no damage';
      if (!monAt(fixture, 1, 0).some(p => p.status === ids.conditions.brn)) {
        return 'the boosted target was never burned';
      }
      if (monAt(fixture, 1, 1).some(p => p.status === ids.conditions.brn)) {
        return 'an unboosted target was burned';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 17000, artifact: 'more_smallmoves.json', debugEnv: 'DEBUG_SMALLMOVES'});
