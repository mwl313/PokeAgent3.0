// Cold primary-callback probes, deliberately not legal full battle fixtures.
import fs from 'node:fs';
import {createRequire} from 'node:module';
import {FORMAT, ORACLE_COMMIT, verifyReference} from '../reference.mjs';
const require=createRequire(import.meta.url);
const {Dex,Battle}=require('../../vendor/pokemon-showdown/dist/sim');
verifyReference();
const dex=Dex.forFormat(FORMAT), battle=new Battle({formatid:FORMAT,seed:[1,2,3,4]});
const cases=[];
const abilities=['pixilate','aerilate','refrigerate','galvanize','normalize','dragonize','liquidvoice'];
const common=['judgment','multiattack','naturalgift','revelationdance','technoblast','terrainpulse','weatherball'];
function probe(ability,id,type='Normal',category='Special',sound=false,isZ=false,isMax=false,markerMode='actual',power=90,weather='') {
 const a=dex.abilities.get(ability), m={...dex.moves.get(id),type,category,flags:{sound},isZ,isMax,basePower:power};
 const ctx={effect:a,activeMove:m,chainModify:r=>r,debug(){},add(){}};
 const pokemon={terastallized:false,volatiles:{},effectiveWeather:()=>weather};
 // Actual move-owned callbacks for the two supported preparations.
 if(id==='struggle') dex.moves.get(id).onModifyMove.call(ctx,m,pokemon);
 if(id==='weatherball') {dex.moves.get(id).onModifyType.call(ctx,m,pokemon);dex.moves.get(id).onModifyMove.call(ctx,m,pokemon);}
 a.onModifyType.call(ctx,m,pokemon);
 const actualMarker=!!m.typeChangerBoosted;
 if(markerMode==='mismatch') m.typeChangerBoosted=dex.abilities.get(ability==='pixilate'?'aerilate':'pixilate');
 if(markerMode==='laterElectric') m.type='Electric';
 const ratio=a.onBasePower?.call(ctx,power,pokemon,{},m)??[1,1];
 cases.push({ability,id,type,category,sound,is_z:isZ,is_max:isMax,marker_mode:markerMode,power,weather,expected_type:m.type,expected_marker:actualMarker,expected_action_power:m.basePower,expected_power:battle.modify(power,ratio)});
}
for(const a of abilities) {
 for(const [id,type,sound] of [['hypervoice','Normal',true],['aerialace','Flying',false],['snarl','Dark',true],['flamethrower','Fire',false]]) probe(a,id,type,'Special',sound);
 for(const id of [...common,'hiddenpower']) probe(a,id);
 probe(a,'struggle','Normal','Physical',false,false,false,'actual',50);
 probe(a,'hypervoice','Normal','Special',true,true);probe(a,'growl','Normal','Status',true,true,false,'actual',0);
 probe(a,'judgment','Normal','Special',false,false,true);
 for(const w of ['','sunnyday','raindance','sandstorm','snowscape']) probe(a,'weatherball','Normal','Special',false,false,false,'actual',50,w);
 for(const p of [1,23,40,85,90,101]) probe(a,'hypervoice','Normal','Special',true,false,false,'actual',p);
 probe(a,'hypervoice','Normal','Special',true,false,false,'mismatch');
 probe(a,'hypervoice','Normal','Special',true,false,false,'laterElectric');
}
battle.destroy();
fs.writeFileSync(new URL('./type_conversion_fixtures.json',import.meta.url),JSON.stringify({oracle_commit:ORACLE_COMMIT,format:FORMAT,scope:'isolated primary ability callbacks; illegal starting abilities and Z/Max are synthetic primitives, not legal battle certification',cases},null,2)+'\n');
console.log(`Generated ${cases.length} primary callback probes`);
