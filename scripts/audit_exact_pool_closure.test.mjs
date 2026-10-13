import fs from 'node:fs';
import {createRequire} from 'node:module';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {auditPool} from './audit_exact_pool_closure.mjs';
const require=createRequire(import.meta.url),{Dex}=require('../vendor/pokemon-showdown/dist/sim');
const tables=JSON.parse(fs.readFileSync(new URL('../engine/data/dex.json',import.meta.url))).tables;
const pool=JSON.parse(fs.readFileSync(new URL('../engine/data/a10-training-teams.json',import.meta.url)));
const scope=JSON.parse(fs.readFileSync(new URL('../engine/data/scope.json',import.meta.url)));
const run=p=>auditPool(p,tables,Dex.forFormat('gen9championsvgc2026regmc'),scope);
test('pool closure includes Mega and natural form transitions without expanding to entire format',()=>{
 const r=run(pool);assert.deepEqual(r.unreviewed,[]);
 for(const f of ['palafinhero','aegislashblade','mimikyubusted','castformrainy','metagrossmega'])assert(r.reachable.species.includes(f));
 assert(r.reachable.abilities.includes('simple'));
 for(const id of ['transform','copycat','instruct'])assert(!r.reachable.moves.includes(id));
 for(const id of ['imposter','receiver','harvest'])assert(!r.reachable.abilities.includes(id));
});
test('new Transform source fails closed rather than reusing old certification',()=>{
 const changed=structuredClone(pool);changed[0].members[0].moves=[tables.moves.find(m=>m.id==='transform').numeric_id];
 assert(run(changed).unreviewed.some(x=>x.startsWith('move:transform')));
});
test('new Imposter source fails closed',()=>{
 const changed=structuredClone(pool);changed[0].members[0].ability=tables.abilities.find(m=>m.id==='imposter').numeric_id;
 assert(run(changed).unreviewed.some(x=>x.startsWith('ability:imposter')));
});
