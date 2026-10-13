// Independent differential retry witnesses, using actual emitted requests.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {ReferenceSession,verifyReference,ORACLE_COMMIT,FORMAT} from '../reference.mjs';
import {referenceTeams,boundaryEvidence} from './p1_reference_probe.mjs';
verifyReference();
const base=JSON.parse(fs.readFileSync(new URL('../data/more_interactions.json',import.meta.url))).fixtures.find(f=>f.name==='imprison_shared_pool');
const tables=JSON.parse(fs.readFileSync(new URL('../data/dex.json',import.meta.url))).tables;
const table=tables.moves;
const id=name=>table.find(m=>m.id===name).numeric_id;
const fixtures=[];
for (const allHidden of [false,true]) {
  const f=structuredClone(base); f.name=allHidden?'imprison_all_hidden_struggle':'imprison_retry_multiple_shared';
  f.teams[0].members[0].moves=[id('imprison'),id('protect'),id('rest')];
  f.teams[1].members[1].moves=allHidden?[id('protect'),id('rest')]:[id('dragonpulse'),id('protect'),id('rest')];
  const session=new ReferenceSession({teams:referenceTeams(f),seed:f.seed});
  const steps=[];
  const apply=(side,command,actions)=>{
    const before=boundaryEvidence(session), result=session.choose(side==='P1'?'p1':'p2',command), after=boundaryEvidence(session);
    steps.push({side,command,actions,accepted:result.accepted,turn:after.turn,rng:typeof after.rng==='string'?after.rng.split(',').map(Number):after.rng,
      masks:session.battle.sides.map(s=>s.activeRequest?.active?.map(r=>r.moves.map(m=>({id:id(m.id),disabled:!!m.disabled})))??null),
      world:session.battle.sides.map(s=>s.pokemon.map(p=>({roster:Number(p.name.match(/m(\d+)$/)[1]),hp:p.hp,
        status:p.status?tables.conditions.find(c=>c.id===p.status).numeric_id:0,
        moves:p.moveSlots.map(m=>({id:id(m.id),pp:m.pp}))})))});
    if (!result.accepted) {assert.deepEqual(after.rng,before.rng);assert.equal(after.turn,before.turn);}
    return {before,after,result};
  };
  for (const [i,step] of f.steps.slice(0,4).entries()) {
    let command=step.command,actions=structuredClone(step.actions);
    if (allHidden && i===3) {command='move 1, move 1';actions[1].target_location=0;}
    assert.equal(apply(step.side,command,actions).result.accepted,true);
  }
  const action=(own_slot,move_slot,target_location=0)=>({kind:'Move',own_slot,move_slot,target_location,switch_destination:255,resource:'None'});
  // Commit the peer first: a rejection must preserve its pending choice.
  if (!allHidden) assert.equal(apply('P1','move 3, move 1 2',[action(0,2),action(1,0,2)]).result.accepted,true);
  const probe=apply('P2',allHidden?'move 1, move 1':'move 1, move 2',[action(0,255),action(1,allHidden?0:1)]);
  assert.equal(probe.result.accepted,allHidden);
  if (!allHidden) {
    const moves=probe.after.outbound.p2.active[1].moves;
    assert.equal(moves.find(m=>m.id==='protect').disabled,true);
    assert.equal(moves.find(m=>m.id==='rest').disabled,true);
    assert.equal(moves.find(m=>m.id==='dragonpulse').disabled,false);
    assert.equal(apply('P2','move 1, move 1 2',[action(0,255),action(1,0,2)]).result.accepted,true);
  } else {
    assert.equal(apply('P1','move 3, move 1 2',[action(0,2),action(1,0,2)]).result.accepted,true);
    assert(session.battle.log.some(line=>line.includes('|move|p2b:')&&line.includes('|Struggle|')));
  }
  fixtures.push({name:f.name,teams:f.teams,seed:f.seed,steps}); session.destroy();
}
fs.writeFileSync(new URL('../data/imprison-retry-fixtures.json',import.meta.url),JSON.stringify({oracle_commit:ORACLE_COMMIT,format:FORMAT,fixtures},null,2)+'\n');
console.log(JSON.stringify({status:'PASS',fixtures:fixtures.length}));
