// Replay unchanged teams/seeds/commands; regenerate only public request
// disabled/target fields from the pinned clean reference. World/RNG/events
// and accepted command histories are never edited by this migration.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {ReferenceSession,verifyReference,ORACLE_COMMIT,FORMAT} from '../reference.mjs';
import {referenceTeams} from './p1_reference_probe.mjs';
import {publicMoveDisabled,publicMoveTarget} from './public_request_moves.mjs';
verifyReference();
const dataDir=new URL('../data/',import.meta.url);
const tables=JSON.parse(fs.readFileSync(new URL('dex.json',dataDir))).tables;
const moveIDs=Object.fromEntries(tables.moves.map(row=>[row.numeric_id,row.id]));
const sha=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const report={oracle_commit:ORACLE_COMMIT,format:FORMAT,files:[],changed_fields:[],fixtures_replayed:0,boundaries_replayed:0};
for (const file of fs.readdirSync(dataDir).filter(f=>f.endsWith('.json'))) {
  // Always compare with the immutable starting revision, including if an
  // interrupted attempt wrote some derived files before the receipt.
  const url=new URL(file,dataDir),before=execFileSync('git',['show',`ec5912736440d5f5456f678c4d76f8fd26fd20aa:engine/data/${file}`],{maxBuffer:256*1024*1024});
  const body=JSON.parse(before);
  if (!Array.isArray(body.fixtures)) continue;
  let changes=0, fixtures=0;
  for (const fixture of body.fixtures) {
    if (!fixture.teams?.[0]?.members || !fixture.steps?.every(s=>typeof s.command==='string' && s.expected)) continue;
    const session=new ReferenceSession({teams:referenceTeams(fixture),seed:fixture.seed});
    function regenerate(expected,boundary) {
      for (let side=0;side<2;side++) {
        const details=expected?.sides?.[side]?.request_detail;
        if (!details?.slots) continue;
        for (let slot=0;slot<details.slots.length;slot++) {
          const info=session.battle.sides[side].activeRequest?.active?.[slot];
          if (!info) {
            const row=details.slots[slot];
            for (const [key,value] of Object.entries({moves:[],trapped:false,maybe_trapped:false,can_mega:false})) {
              if (key!=='moves' && !(key in row)) continue;
              if (JSON.stringify(row[key])===JSON.stringify(value)) continue;
              report.changed_fields.push({file,fixture:fixture.name,boundary,side,slot,field:key,
                before:row[key],after:value,reason:'outbound request has no active move request; remove stale privileged move mask'});
              row[key]=value;changes++;
            }
            for (const key of ['locked','locked_recharge']) if(key in row) {
              report.changed_fields.push({file,fixture:fixture.name,boundary,side,slot,field:key,
                before:row[key],reason:'no active move request; no public locked move'});
              delete row[key];changes++;
            }
            continue;
          }
          for (const [index,old] of (details.slots[slot].moves??[]).entries()) {
            const move={id:moveIDs[old.id],target:old.target};
            const values={disabled:publicMoveDisabled(info,move),target:publicMoveTarget(info,move)};
            for (const [key,value] of Object.entries(values)) {
              if (old[key]===value) continue;
              report.changed_fields.push({file,fixture:fixture.name,boundary,side,slot,index,move:move.id,
                field:key,before:old[key],after:value,reason:'serialized player request semantics; matched by move ID'});
              old[key]=value;changes++;
            }
          }
        }
      }
    }
    try {
      regenerate(fixture.initial,-1);
      for (const [index,step] of fixture.steps.entries()) {
        assert.equal(session.choose(step.side==='P1'?'p1':'p2',step.command).accepted,true,`${file} ${fixture.name} ${index}`);
        regenerate(step.expected,index);report.boundaries_replayed++;
      }
      fixtures++;report.fixtures_replayed++;
    } finally {session.destroy();}
  }
  if (!changes) {
    if (!fs.readFileSync(url).equals(before)) fs.writeFileSync(url,before);
    continue;
  }
  const after=Buffer.from(JSON.stringify(body)+'\n');
  fs.writeFileSync(url,after);
  report.files.push({file,fixture_count:body.fixtures.length,replayed:fixtures,changed_fields:changes,
    before_sha256:sha(before),after_sha256:sha(after),bytes:after.length});
  console.log(JSON.stringify(report.files.at(-1)));
}
const manifestPath=new URL('manifest.json',dataDir);
const manifestBefore=execFileSync('git',['show','ec5912736440d5f5456f678c4d76f8fd26fd20aa:engine/data/manifest.json']);
const manifest=JSON.parse(manifestBefore);
report.manifest_before_sha256=sha(manifestBefore);
for (const entry of report.files) {
  if (manifest.files[entry.file]) manifest.files[entry.file]={...manifest.files[entry.file],sha256:entry.after_sha256,bytes:entry.bytes};
}
fs.writeFileSync(manifestPath,JSON.stringify(manifest,null,2)+'\n');
report.manifest_after_sha256=sha(fs.readFileSync(manifestPath));
fs.writeFileSync(process.argv[2],JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({files:report.files.length,fields:report.changed_fields.length,fixtures:report.fixtures_replayed,boundaries:report.boundaries_replayed}));
