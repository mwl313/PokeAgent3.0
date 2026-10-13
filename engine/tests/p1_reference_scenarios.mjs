// Additional reference-only witnesses. The one-survivor case is explicitly a
// controlled state probe, not claimed as a naturally played differential game.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {ReferenceSession,verifyReference,ORACLE_COMMIT,FORMAT} from '../reference.mjs';
import {referenceTeams,boundaryEvidence} from './p1_reference_probe.mjs';
verifyReference();
const fixture=JSON.parse(fs.readFileSync(new URL('../data/more_interactions.json',import.meta.url))).fixtures.find(f=>f.name==='imprison_shared_pool');
function setup() {
  const s=new ReferenceSession({teams:referenceTeams(fixture),seed:fixture.seed});
  for (const step of fixture.steps.slice(0,4)) {
    assert.equal(s.choose(step.side==='P1'?'p1':'p2',step.command).accepted,true);
    boundaryEvidence(s);
  }
  return s;
}
const results={oracle_commit:ORACLE_COMMIT,format:FORMAT,seed:fixture.seed,scenarios:[]};
{
  const s=setup(), events=[];
  for (const [side,command] of [['p1','switch 3, move 2'],['p2','move 1, move 1 2'],
    ['p1','switch 3, move 2'],['p2','move 2, move 2'],['p1','move 1, move 2'],['p2','move 2, move 1 2']]) {
    const result=s.choose(side,command);
    assert.equal(result.accepted,true,command);
    events.push({side,command,accepted:result.accepted,boundary:boundaryEvidence(s)});
  }
  const switched=events[1].boundary;
  assert.equal(switched.internal[0].active.some(p=>p.volatiles.includes('imprison')),false);
  assert.equal(switched.internal[1].active.some(p=>p.moves.some(m=>m.disabled==='hidden')),false);
  assert.equal(events[3].boundary.internal[0].active[0].volatiles.includes('imprison'),false);
  assert.equal(events[5].boundary.internal[0].active[0].volatiles.includes('imprison'),true);
  results.scenarios.push({name:'natural_switch_deactivation_reentry_reactivation',events});
  s.destroy();
}
{
  const s=setup(), side=s.battle.sides[1];
  // Controlled public-interface fixture: only Goodra remains alive.
  for (const pokemon of side.pokemon) if (pokemon!==side.active[1]) {pokemon.hp=0;pokemon.fainted=true;}
  side.pokemonLeft=1;
  s.battle.makeRequest('move');
  for (const player of s.battle.sides) player.emitRequest();
  const before=boundaryEvidence(s);
  assert.equal(before.internal[1].active.filter(p=>p && !p.name.endsWith('0')).length,1);
  assert.equal(side.active[1].isLastActive(),true);
  const protect=before.outbound.p2.active[1].moves.find(m=>m.id==='protect');
  assert.equal(protect.disabled,false);
  const choice=s.choose('p2','pass, move 2');
  assert.equal(choice.accepted,false);
  results.scenarios.push({name:'controlled_only_survivor_hidden_shared_move',natural_battle:false,
    before,accepted:choice.accepted,after:boundaryEvidence(s)});
  s.destroy();
}
fs.writeFileSync(process.argv[2],JSON.stringify(results,null,2)+'\n');
console.log(JSON.stringify({scenarios:results.scenarios.map(s=>s.name),status:'PASS'}));
