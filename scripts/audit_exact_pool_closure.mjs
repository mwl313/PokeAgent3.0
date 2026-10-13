// Cold, dataset-specific closure. Never changes the frozen pool or global scope.
import fs from 'node:fs';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';
import {computeDynamicClosure} from './dynamic_closure.mjs';
import {verifyReference,FORMAT,ORACLE_COMMIT} from '../engine/reference.mjs';
const require=createRequire(import.meta.url),{Dex,toID}=require('../vendor/pokemon-showdown/dist/sim');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const hash=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const sorted=set=>[...set].sort();
const formRules={
  disguise:['Mimikyu-Busted'],forecast:['Castform','Castform-Sunny','Castform-Rainy','Castform-Snowy'],
  stancechange:['Aegislash','Aegislash-Blade'],zerotohero:['Palafin-Hero'],
};
// These rules constrain the SOURCE of an effect, not just its allowed universe.
const reviewed={
  'ability:magicbounce':'reflects the received move; no new move identity',
  'move:sleeptalk':'chooses among the users current moves; no new move identity',
  'ability:magician':'takes a held item', 'ability:pickpocket':'takes a held item',
  'ability:symbiosis':'transfers an allies held item',
  'move:bugbite':'consumes targets berry','move:fling':'uses held item',
  'move:knockoff':'removes held item','move:switcheroo':'swaps held items','move:trick':'swaps held items',
  'ability:trace':'copies a currently present ability','ability:wanderingspirit':'swaps present abilities',
  'move:entrainment':'copies users current ability','move:roleplay':'copies targets current ability',
  'move:skillswap':'swaps present abilities','ability:mummy':'sets constant Mummy',
  'move:simplebeam':'sets constant Simple',
};
function handlers(entity) {
  const found=[];
  function walk(value,key='') {
    if(typeof value==='function')found.push([key,value.toString()]);
    else if(value&&typeof value==='object')for(const [k,v]of Object.entries(value))walk(v,`${key}.${k}`);
  }
  walk(entity);return found;
}
export function auditPool(pool, tables, dex, scope) {
  const names=Object.fromEntries(Object.entries(tables).map(([k,v])=>[k,Object.fromEntries(v.map(r=>[r.numeric_id,r.id]))]));
  const reachable={moves:new Set(['struggle']),abilities:new Set(),items:new Set(),species:new Set()};
  for(const team of pool)for(const m of team.members) {
    for(const id of m.moves){assert(names.moves[id],`unknown move ${id}`);reachable.moves.add(names.moves[id]);}
    for(const kind of ['abilities','items','species']) {
      const value=m[{abilities:'ability',items:'item',species:'species'}[kind]];
      if(kind==='items'&&value===0)continue;
      assert(names[kind][value],`unknown ${kind} ${value}`);reachable[kind].add(names[kind][value]);
    }
  }
  const roots=Object.fromEntries(Object.entries(reachable).map(([k,v])=>[k,sorted(v)]));
  const closure=computeDynamicClosure(dex,scope),edges=new Map(),unknown=new Set();
  let previous='';
  while(previous!==JSON.stringify(Object.fromEntries(Object.entries(reachable).map(([k,v])=>[k,sorted(v)])))) {
    previous=JSON.stringify(Object.fromEntries(Object.entries(reachable).map(([k,v])=>[k,sorted(v)])));
    // Any pool item may conservatively reach any compatible pool species, even
    // if an item-transfer restriction would prevent the particular combination.
    for(const item of reachable.items)for(const [base,form] of Object.entries(dex.items.get(item).megaStone??{})) {
      if(!reachable.species.has(toID(base)))continue;
      const f=dex.species.get(form);assert(f.exists,form);
      reachable.species.add(f.id);reachable.abilities.add(toID(f.abilities[0]));
      edges.set(`mega:${item}:${f.id}`,{source:`item:${item}`,rule:'compatible Mega evolution',target:f.id,ability:toID(f.abilities[0])});
    }
    for(const entry of closure.entries) {
      const kind=entry.kind==='move'?'moves':entry.kind==='ability'?'abilities':'items';
      if(!reachable[kind].has(entry.id))continue;
      const key=`${entry.kind}:${entry.id}`;
      if(!reviewed[key]){unknown.add(key);continue;}
      edges.set(key,{source:key,rule:reviewed[key],reference_rules:entry.rules});
      if(key==='move:simplebeam')reachable.abilities.add('simple');
      if(key==='ability:mummy')reachable.abilities.add('mummy');
    }
    // Audit every reachable move/ability hook for identity-generating calls,
    // including forme changes that the older global scanner did not cover.
    for(const kind of ['moves','abilities','items'])for(const id of reachable[kind]) {
      const singular={moves:'move',abilities:'ability',items:'item'}[kind];
      const entity=dex[kind].get(id), source=handlers(entity).map(([k,s])=>s).join('\n');
      if(/(?:useMove|resolveAction|setAbility|skillSwap|transformInto)\(/.test(source)&&!reviewed[`${singular}:${id}`])unknown.add(`${singular}:${id}:unreviewed-call`);
      if(/formeChange\(/.test(source)) {
        if(kind!=='abilities'||!formRules[id]) {unknown.add(`${singular}:${id}:unreviewed-form`);continue;}
        for(const form of formRules[id]){const f=dex.species.get(form);assert(f.exists,form);reachable.species.add(f.id);}
        edges.set(`form:${id}`,{source:`ability:${id}`,rule:'reference species-guarded form transition',targets:formRules[id].map(toID)});
      }
    }
  }
  const hookHashes={};
  for(const kind of ['moves','abilities','items'])for(const id of reachable[kind])hookHashes[`${kind}:${id}`]=hash(JSON.stringify(handlers(dex[kind].get(id))));
  return {roots,reachable:Object.fromEntries(Object.entries(reachable).map(([k,v])=>[k,sorted(v)])),
    edges:[...edges.values()],unreviewed:sorted(unknown),hook_sha256:hookHashes,
    limitations:['Conservative identity closure for this exact dataset and pinned reference, not all Regulation M-C.',
      'Does not establish every executable handler is mechanically correct; differential fixtures and native guards remain required.',
      'Does not include effects introduced by external opponents or future dataset changes.']};
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  verifyReference();
  const poolPath=path.join(root,'engine/data/a10-training-teams.json'),bytes=fs.readFileSync(poolPath);
  const pool=JSON.parse(bytes),tables=JSON.parse(fs.readFileSync(path.join(root,'engine/data/dex.json'))).tables;
  assert.equal(pool.length,1220);
  const report={schema:'pa3-exact-pool-closure-v1',oracle_commit:ORACLE_COMMIT,format:FORMAT,
    team_count:pool.length,dataset_sha256:hash(bytes),...auditPool(pool,tables,Dex.forFormat(FORMAT),JSON.parse(fs.readFileSync(path.join(root,'engine/data/scope.json'))))};
  fs.writeFileSync(path.join(root,'engine/data/exact-pool-closure-audit.json'),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({team_count:pool.length,reachable:Object.fromEntries(Object.entries(report.reachable).map(([k,v])=>[k,v.length])),unreviewed:report.unreviewed}));
  if(report.unreviewed.length)process.exitCode=1;
}
