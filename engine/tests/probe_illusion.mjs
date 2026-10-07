// Read-only scout (development only, never merged as a corpus generator):
// prints the pinned reference's Illusion protocol lines and state for two
// scenes. Companion evidence for engine/NEXT_ILLUSION_REQUIREMENTS.md.
//
//   node engine/tests/probe_illusion.mjs
//
// Scene A shows the masked switch-in identity and the reveal sequence when a
// damaging hit lands. Scene B shows that a switch-out ends the illusion with
// no `replace` / `-end Illusion` messages (the outgoing Pokemon is being
// called back), while the benched entity keeps its stale illusion reference.
import {createRequire} from 'node:module';
import {ReferenceSession, verifyReference, FORMAT} from '../reference.mjs';
const require = createRequire(import.meta.url);
const {TeamValidator} = require('../../vendor/pokemon-showdown/dist/sim');
const validator = new TeamValidator(FORMAT);
verifyReference();

const set = (species, ability, moves) =>
  ({name: species, species, ability, item: '', nature: 'Serious', level: 50, gender: 'M', moves,
    evs: {hp: 32, atk: 2, def: 16, spa: 0, spd: 16, spe: 0},
    ivs: {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31}});
const team = list => list.map((s, i) => ({...s, name: `s${i}`}));
const zTeam = () => team([
  set('Zoroark', 'Illusion', ['Night Daze', 'Protect']),
  set('Milotic', 'Competitive', ['Surf', 'Protect']),
  set('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
  set('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
  set('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
  set('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
]);
const foeTeam = () => team([
  set('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
  set('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
  set('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
  set('Milotic', 'Competitive', ['Surf', 'Protect']),
  set('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
  set('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
]);

const run = (label, script) => {
  console.log(`\n================ ${label}`);
  const a = zTeam(), b = foeTeam();
  for (const t of [a, b]) {
    const problems = validator.validateTeam(t);
    if (problems) { console.log('INVALID', problems.join('; ')); return; }
  }
  const session = new ReferenceSession({teams: [a, b], seed: [3, 5, 7, 4242]});
  let n = 0;
  const dump = () => {
    for (const line of session.battle.log.slice(n)) {
      if (/^\|-?(switch|replace|move|damage|end|formechange)\|/.test(line)) console.log('   ', line);
    }
    n = session.battle.log.length;
  };
  session.choose('p1', 'team 1234');
  session.choose('p2', 'team 1234');
  dump();
  for (let turn = 1; turn <= 3 && !session.battle.ended; turn++) {
    const [c1s, c2s] = script[turn - 1] ?? [];
    if (c1s) console.log(`-- turn ${turn} p1 ${c1s}: ${session.choose('p1', c1s).accepted}`);
    if (c2s) console.log(`-- turn ${turn} p2 ${c2s}: ${session.choose('p2', c2s).accepted}`);
    dump();
    for (const side of session.battle.sides) {
      const p = side.pokemon.find(x => x.species.id === 'zoroark');
      if (p) {
        console.log(`   STATE ${side.n} ${p.name} species=${p.species.id} ` +
          `illusion=${p.illusion ? p.illusion.name : 'null'} pos=${p.position} hp=${p.hp}`);
      }
    }
  }
  session.destroy();
};

// A: a damaging hit (Iron Head from p2a) breaks the illusion on Zoroark.
run('damaging_hit_breaks_illusion', [
  ['move 1 1, move 2', 'move 1 1, move 2'],
  ['move 2, move 2', 'move 2, move 2'],
]);

// B: switching the illusioned Pokemon out must not emit the End messages.
run('switch_out_does_not_end_illusion', [
  ['move 2, move 2', 'move 2, move 2'],
  ['switch 3, move 2', 'move 2, move 2'],
]);
