// Cold oracle: opponent type expectations come ONLY from public protocol lines.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {ReferenceSession,verifyReference,ORACLE_COMMIT,FORMAT} from '../reference.mjs';
import {referenceTeams} from './p1_reference_probe.mjs';
const require=createRequire(import.meta.url),{Dex}=require('../../vendor/pokemon-showdown/dist/sim');
const dex=Dex.forFormat(FORMAT);
const types=JSON.parse(fs.readFileSync(new URL('../data/dex.json',import.meta.url))).tables.types;
// Native ID 0 represents the reference's explicitly typeless `???`, not an
// unknown type. V2's known masks must preserve that distinction.
const typeID=name=>{if(name==='???')return 0; const row=types.find(t=>t.data.name===name); assert(row,`unknown type ${name}`);return row.numeric_id;};
const speciesTypes=name=>dex.species.get(name).types.map(typeID).filter(x=>x!==null);
verifyReference();
const corpus=[...JSON.parse(fs.readFileSync(new URL('../data/turn-fixtures.json',import.meta.url))).fixtures,
  ...JSON.parse(fs.readFileSync(new URL('../data/turn-fixtures-2.json',import.meta.url))).fixtures];
const selected=corpus.filter(f=>/soak|magicpowder|doubleshock|burnup|protean|libero|mimicry|illusion|stancechange|forecast/i.test(f.name));
selected.push(...corpus.filter(f=>/mega/i.test(f.name)).slice(0,8));
const fixtures=[];const events={};
for(const f of selected) {
  const session=new ReferenceSession({teams:referenceTeams(f),seed:f.seed});
  const slots={};let cursor=0;
  function publicTypes() {
    for(const line of session.battle.log.slice(cursor)) {
      let m=line.match(/^\|(?:switch|drag|replace|detailschange|-formechange)\|(p[12][ab]): [^|]+\|([^,|]+)/);
      if(m){slots[m[1]]={species:m[2],types:speciesTypes(m[2])};continue;}
      m=line.match(/^\|-start\|(p[12][ab]): [^|]+\|typechange\|([^|]+)/);
      if(m){assert(slots[m[1]],line); slots[m[1]].types=m[2].split('/').map(typeID).filter(x=>x!==null);events.typechange=(events.typechange??0)+1;continue;}
      m=line.match(/^\|-start\|(p[12][ab]): [^|]+\|typeadd\|([^|]+)/);
      if(m){slots[m[1]].types.push(typeID(m[2]));continue;}
      m=line.match(/^\|-end\|(p[12][ab]): [^|]+\|typechange/);
      if(m){slots[m[1]].types=speciesTypes(slots[m[1]].species);}
    }
    cursor=session.battle.log.length;
    return structuredClone([0,1].map(s=>[0,1].map(slot=>slots[`p${s+1}${slot?'b':'a'}`]??null)));
  }
  const steps=[];
  for(const step of f.steps) {
    assert.equal(session.choose(step.side==='P1'?'p1':'p2',step.command).accepted,true,`${f.name} ${step.command}`);
    const observed=publicTypes();
    const own=session.battle.sides.map(s=>s.active.map(p=>p?{roster:Number(p.name.match(/m(\d+)$/)[1]),types:p.getTypes().map(typeID).filter(x=>x!==null)}:null));
    steps.push({side:step.side,actions:step.actions,observed,own});
  }
  fixtures.push({name:f.name,teams:f.teams,seed:f.seed,steps});session.destroy();
}
const output={oracle_commit:ORACLE_COMMIT,format:FORMAT,source:'public switch/drag/replace/detailschange/formechange/typechange/typeadd lines; own getTypes only for owner',events,fixtures};
fs.writeFileSync(new URL('../data/public-type-oracle.json',import.meta.url),JSON.stringify(output,null,2)+'\n');
console.log(JSON.stringify({fixtures:fixtures.length,boundaries:fixtures.reduce((n,f)=>n+f.steps.length,0),events}));
