// Development-only corpus for the fifth passive cluster:
//   Aftermath   - a contact move that KOs the holder damages the attacker
//   Cheek Pouch - any eaten berry heals the holder for a third of max HP
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, logHas, everHas, runTrials} = createScaffold();

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
    name: 'aftermath_punishes_the_contact_ko',
    // Garbodor starts at full HP; the first hit weakens it and the second
    // contact move knocks it out, which is when Aftermath fires.
    p1: () => team(setOf('Garbodor', 'Aftermath', ['Body Slam', 'Protect'])),
    p2: () => team(setOf('Aggron', 'Sturdy', ['Iron Head', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
    ],
    coverage: {ability: 'aftermath'},
    verify(fixture, session) {
      if (!logHas(session, /\|faint\|p1a: s0/)) return 'the holder never fainted';
      if (!logHas(session, /\|-damage\|p2a: s0\|/)) return 'the attacker never took Aftermath damage';
      // The attacker's HP must drop across the boundary where the holder faints.
      const hp = side => fixture.steps.map(step => step.expected.sides[side].pokemon
        .find(p => p.roster === 0));
      const attacker = hp(1);
      const holderFaint = fixture.steps.findIndex(step => step.expected.sides[0].pokemon
        .some(p => p.roster === 0 && p.fainted));
      if (holderFaint <= 0) return 'no faint boundary was recorded';
      if (!(attacker[holderFaint].hp < attacker[holderFaint - 1].hp)) {
        return 'the attacker was not damaged by the knockout';
      }
      return null;
    },
  },
  {
    name: 'cheekpouch_heals_when_the_berry_is_eaten',
    p1: () => team(setOf('Diggersby', 'Cheek Pouch', ['Body Slam', 'Protect'], 'Sitrus Berry')),
    p2: () => team(setOf('Aggron', 'Sturdy', ['Iron Head', 'Protect'])),
    script: [
      {p1: ['protect', 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
      {p1: ['protect', 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']},
    ],
    coverage: {ability: 'cheekpouch'},
    verify(fixture, session) {
      if (!logHas(session, /\|-enditem\|p1a: s0\|Sitrus Berry/)) return 'the Sitrus Berry was never eaten';
      // Sitrus alone logs `[from] item: Sitrus Berry`; the extra heal is
      // attributed to the ability, which is exactly what the scene needs.
      if (!logHas(session, /\|-heal\|p1a: s0\|.*\[from\] ability: Cheek Pouch/)) {
        return 'Cheek Pouch never healed the holder';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 43000, artifact: 'more_passives5.json', debugEnv: 'DEBUG_PASSIVES5'});
