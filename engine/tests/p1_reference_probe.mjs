// Independent outbound-request evidence; deliberately does not import the
// generator's public-request adapter.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {ReferenceSession, verifyReference, ORACLE_COMMIT, FORMAT} from '../reference.mjs';
const tables = JSON.parse(fs.readFileSync(new URL('../data/dex.json', import.meta.url))).tables;
const numeric = Object.fromEntries(Object.entries(tables).map(([k, rows]) => [k, Object.fromEntries(rows.map(r => [r.numeric_id, r]))]));
const name = (kind, id) => numeric[kind][id]?.data?.name ?? '';
const stats = ['hp','atk','def','spa','spd','spe'];
export function referenceTeams(fixture) {
  return fixture.teams.map((team, side) => team.members.map((m, i) => ({
    name: `s${side}m${i}`, species: name('species',m.species), ability: name('abilities',m.ability),
    item: name('items',m.item), nature: name('natures',m.nature), level: 50, gender: m.gender || '',
    moves: m.moves.map(x => name('moves',x)), evs: Object.fromEntries(stats.map((k,j)=>[k,m.points[j]])),
    ivs: Object.fromEntries(stats.map((k,j)=>[k,m.ivs[j]])),
  })));
}
export function outboundRequests(session) {
  session.battle.sendUpdates();
  const result = {};
  for (const message of session.messages) {
    if (message.type !== 'sideupdate' || typeof message.payload !== 'string') continue;
    const match = message.payload.match(/^(p[12])\n\|request\|(.*)$/s);
    if (match) result[match[1]] = JSON.parse(match[2]);
  }
  return result;
}
export function boundaryEvidence(session) {
  const outbound = outboundRequests(session);
  for (const side of session.battle.sides) {
    if (outbound[side.id]) assert.deepEqual(outbound[side.id], side.activeRequest);
  }
  return {turn:session.battle.turn, rng:session.battle.prng.getSeed(), outbound,
    internal:session.battle.sides.map(s=>({id:s.id, choiceDone:s.isChoiceDone(),
      active:s.active.map(p=>p && ({name:p.name, lastActive:p.isLastActive(),
        moves:p.moveSlots.map(m=>({id:m.id,pp:m.pp,disabled:m.disabled,disabledSource:m.disabledSource,target:m.target})),
        volatiles:Object.keys(p.volatiles)}))})), log:[...session.battle.log]};
}
if (process.argv[1]?.endsWith('p1_reference_probe.mjs')) {
  verifyReference();
  const fixture = JSON.parse(fs.readFileSync(new URL('../data/more_interactions.json', import.meta.url))).fixtures.find(f=>f.name==='imprison_shared_pool');
  const session=new ReferenceSession({teams:referenceTeams(fixture),seed:fixture.seed});
  const boundaries=[boundaryEvidence(session)];
  for (const step of fixture.steps.slice(0,4)) {
    assert.equal(session.choose(step.side==='P1'?'p1':'p2',step.command).accepted,true);
    boundaries.push(boundaryEvidence(session));
  }
  // Both moves are advertised for last-active Goodra, but raw hidden Protect
  // is rechecked by Side.chooseMove using getMoves(restrictData=false).
  const beforeChoice=boundaryEvidence(session);
  const probe=session.choose('p2','move 1, move 2');
  const afterChoice=boundaryEvidence(session);
  const evidence={oracle_commit:ORACLE_COMMIT,format:FORMAT,fixture:fixture.name,seed:fixture.seed,
    teams:referenceTeams(fixture),previous_actions:fixture.steps.slice(0,4),boundaries,
    hidden_choice:{command:'move 1, move 2',accepted:probe.accepted,before:beforeChoice,after:afterChoice,messages:probe.messages}};
  fs.writeFileSync(process.argv[2],JSON.stringify(evidence,null,2)+'\n');
  console.log(JSON.stringify({fixture:fixture.name,choiceAccepted:probe.accepted,requests:beforeChoice.outbound.p2.active.map(x=>x.moves)}));
  session.destroy();
}
