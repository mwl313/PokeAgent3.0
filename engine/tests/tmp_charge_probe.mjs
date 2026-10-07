import {ReferenceSession, verifyReference} from '../reference.mjs';
verifyReference();
const setOf = (species, ability, moves, item = '', nature = 'Serious', points = {hp: 32, atk: 0, def: 17, spa: 0, spd: 17, spe: 0}) =>
  ({name: species, species, ability, item, nature, level: 50, gender: 'M', moves,
    evs: {hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0, ...points}});
const team1 = [
  setOf('Venusaur', 'Overgrow', ['Solar Beam', 'Protect', 'Seed Bomb']),
  setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
  setOf('Milotic', 'Competitive', ['Surf', 'Protect']),
  setOf('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
  setOf('Goodra-Hisui', 'Shell Armor', ['Dragon Pulse', 'Protect']),
  setOf('Samurott', 'Shell Armor', ['Aqua Jet', 'Protect']),
];
const team2 = [
  setOf('Milotic', 'Competitive', ['Surf', 'Protect']),
  setOf('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
  setOf('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
  setOf('Perrserker', 'Battle Armor', ['Iron Head', 'Protect']),
  setOf('Hydreigon', 'Levitate', ['Dragon Pulse', 'Protect']),
  setOf('Gallade', 'Sharpness', ['Psycho Cut', 'Protect']),
];
const seed = [1,2,3,4];
const s = new ReferenceSession({teams: [team1, team2], seed});
const show = (label) => {
  const b = s.battle;
  console.log('---', label, 'turn', b.turn, 'ended', b.ended);
  for (const [i, side] of b.sides.entries()) {
    for (const [slot, p] of side.active.entries()) {
      console.log(` p${i+1} slot${slot} ${p.species.name} hp=${p.hp}/${p.maxhp} vol=${Object.keys(p.volatiles).join(',')} locked=${JSON.stringify(p.getLockedMove()||null)} slots=${p.moveSlots.map(m=>`${m.id}:${m.pp}:${m.disabled?1:0}`).join(' ')}`);
    }
  }
};
s.choose('p1', 'team 1234'); s.choose('p2', 'team 1234');
console.log(JSON.stringify(s.messages.slice(-2), null, 1).slice(0, 1500));
show('after preview submit');
let r = s.choose('p1', 'move 1 1, move 2'); console.log('p1 msg', JSON.stringify(r.messages));
show('after p1 choose t1');
r = s.choose('p2', 'move 2, move 1 1'); console.log('p2 msg', JSON.stringify(r.messages).slice(0,2000));
show('end t1');
console.log('request p1', JSON.stringify(s.battle.sides[0].activeRequest));
r = s.choose('p1', 'move 2, move 2');
console.log('p1 turn2 msgs', JSON.stringify(r.messages).slice(0,3000));
show('end t2');


console.log('=== turn 2 choose');
let r2 = s.choose('p1', 'move 1 1, move 2');
console.log('p1 t2 msgs', JSON.stringify(r2.messages).slice(0,1200));
let r3 = s.choose('p2', 'move 2, move 2');
console.log('p2 t2 msgs', JSON.stringify(r3.messages).slice(0,3000));
show('end t2');
console.log('req p1', JSON.stringify(s.battle.sides[0].activeRequest.active).slice(0,400));
