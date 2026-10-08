// Cold differential fixtures; never part of the engine or training runtime.
import fs from 'node:fs';
import {ReferenceSession, ORACLE_COMMIT, FORMAT, verifyReference} from '../reference.mjs';
verifyReference();
const tables = JSON.parse(fs.readFileSync(new URL('../data/dex.json',import.meta.url))).tables;
// The corpus is committed as size-bounded parts; merge them back.
const corpusParts = fs.readdirSync(new URL('../data',import.meta.url))
  .filter(name=>/^turn-fixtures(?:-\d+)?\.json$/.test(name))
  .sort((a,b)=>(Number(a.match(/-(\d+)\.json$/)?.[1]??1)-Number(b.match(/-(\d+)\.json$/)?.[1]??1)));
const corpus = corpusParts.map(name=>JSON.parse(fs.readFileSync(new URL(`../data/${name}`,import.meta.url))))
  .reduce((merged,part)=>({oracle_commit:merged.oracle_commit??part.oracle_commit,format:merged.format??part.format,
    fixtures:[...merged.fixtures,...part.fixtures]}),{fixtures:[]});
const name = (kind,id) => id ? tables[kind].find(row=>row.numeric_id===id).data.name : '';
const keys=['hp','atk','def','spa','spd','spe'];
const ids = Object.fromEntries(Object.entries(tables).map(([kind,rows])=>[kind,Object.fromEntries(rows.map(row=>[row.id,row.numeric_id]))]));
const patterns=['protect_stall_','side_effects_basic_','burn_paralysis_boost_heal_','freeze_flinch_','items_sash_helmet_orb_','tailwind_scarf_','trickroom_expiration_priority_','terrain_weather_room_coexistence_','cloudnine_rain_switch_','hydration_rain_order_','mega_both_sides_','mega_intimidate_hugepower_'];
const selected=[...patterns.flatMap(prefix=>corpus.fixtures.filter(f=>f.name.startsWith(prefix)).slice(0,1)), ...corpus.fixtures.filter(f=>['flashfire_','lightningrod_','pixilate_','aerilate_','refrigerate_','dragonize_','liquidvoice_','convert_','synchronize_','hypercutter_'].some(prefix=>f.name.startsWith(prefix)))];
const snapshot = (battle, rosters) => ({next: battle.effectOrder, sides:battle.sides.map((side,index)=>({pokemon:rosters[index].map(p=>({roster:Number(p.name.slice(-1)),ability:p.abilityState.effectOrder,item:p.itemState.effectOrder,status:p.statusState.effectOrder,volatiles:Object.entries(p.volatiles).map(([id,s])=>[ids.conditions[id],s.effectOrder]).sort((a,b)=>a[0]-b[0])})).sort((a,b)=>a.roster-b.roster),conditions:Object.entries(side.sideConditions).map(([id,s])=>[ids.conditions[id],s.effectOrder]).sort((a,b)=>a[0]-b[0])})),field:[...(battle.field.weather?[[ids.conditions[battle.field.weather],battle.field.weatherState.effectOrder]]:[]),...(battle.field.terrain?[[ids.conditions[battle.field.terrain],battle.field.terrainState.effectOrder]]:[]),...Object.entries(battle.field.pseudoWeather).map(([id,s])=>[ids.conditions[id],s.effectOrder])].sort((a,b)=>a[0]-b[0])});
const fixtures=[];
for(const f of selected){
 const teams=f.teams.map((team,side)=>team.members.map((p,i)=>({name:`s${side}m${i}`,species:name('species',p.species),ability:name('abilities',p.ability),item:name('items',p.item),nature:name('natures',p.nature),moves:p.moves.map(x=>name('moves',x)),evs:Object.fromEntries(keys.map((k,i)=>[k,p.points[i]])),ivs:Object.fromEntries(keys.map((k,i)=>[k,p.ivs[i]])),level:p.level,gender:p.gender})));
 const session=new ReferenceSession({teams,seed:f.seed});
 const rosters=session.battle.sides.map(side=>[...side.pokemon]);
 const fixture={name:f.name,seed:f.seed,teams:f.teams,initial:snapshot(session.battle,rosters),steps:[]};
 for(const step of f.steps){const r=session.choose(step.side==='P1'?'p1':'p2',step.command);if(!r.accepted)throw new Error(f.name);fixture.steps.push({side:step.side,actions:step.actions,expected:snapshot(session.battle,rosters)});}
 fixtures.push(fixture);session.destroy();
}
fs.writeFileSync(new URL('./effect-order-reference.json',import.meta.url),JSON.stringify({oracle_commit:ORACLE_COMMIT,format:FORMAT,fixtures}));
console.log(`Effect order fixtures: ${fixtures.length}`);
