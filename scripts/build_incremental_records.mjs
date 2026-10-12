import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
const root=process.cwd(), dir=path.join(root,'data/raw/vgcpastes/champions-mc/20261012');
const require=createRequire(import.meta.url);
const showdownRoot=process.env.PA3_SHOWDOWN_ROOT || path.join(root,'vendor/pokemon-showdown');
const {Teams}=require(path.join(showdownRoot,'dist/sim/index.js'));
const rows=JSON.parse(fs.readFileSync(path.join(dir,'delta.json'))); const out=[];
for(const row of rows){ if(row.status!=='fetched') continue; const text=fs.readFileSync(path.join(dir,'teams',row.id+'.txt'),'utf8'); let sets;
  try { sets=Teams.import(text); } catch(e) { out.push({source:{team_id:row.id},error:String(e)}); continue; }
  const blocks=text.split(/\n\s*\n/).filter(Boolean); const members=sets.map((set,i)=>({header:[set.species,set.item?' @ '+set.item:''].join(''),fields:{Ability:set.ability,Level:String(set.level||50),EVs:Object.entries(set.evs||{}).filter(([,v])=>v).map(([k,v])=>`${k.toUpperCase()}: ${v}`).join(' / ')},nature_line:(set.nature||'Serious')+' Nature',moves:set.moves||[],allocations:[{raw_line:'EVs: '+Object.entries(set.evs||{}).filter(([,v])=>v).map(([k,v])=>`${k.toUpperCase()}: ${v}`).join(' / '),explicit_values:set.evs||{},unparsed_parts:[]}],raw_export_block:blocks[i] || ''}));
  out.push({collection_schema:'pa3-pokepaste-extraction-v1',source:{sheet_row:null,team_id:row.id,description:row.desc,full_name:'',pokepaste_url:row.url,evs:'Yes',paste_origin:'Current incremental capture',date_shared:row.date,event:'',rank:'',source_url:row.source,report_url:'',other_url:'',owner:''},fetch:{requested_url:row.url,response_url:row.url,http_status:200,content_type:'text/html',fetched_at:new Date().toISOString(),sha256:JSON.parse(fs.readFileSync(path.join(dir,'fetch_metadata',row.key+'.json'))).sha256,bytes:fs.statSync(path.join(dir,'html',row.key+'.html')).size},paste_title:row.desc,paste_notes:[],raw_html_path:`html/${row.key}.html`,team_text_path:`teams/${row.id}.txt`,team_text_sha256:crypto.createHash('sha256').update(text).digest('hex'),members});
}
fs.writeFileSync(path.join(dir,'teams.jsonl'),out.map(x=>JSON.stringify(x)).join('\n')+'\n'); console.log(JSON.stringify({records:out.length,errors:out.filter(x=>x.error).length}));
