import path from 'node:path';
import {prepare} from './prepare_teams.mjs';
const root=process.cwd();
const output=path.join(root,'data/teams/a10-p0-proposed-20261013');
const custom={kind:'vgcpastes',tab:'mc',provider:'VGCPastes',source_tab:'Champions M-C',relative:'data/raw/vgcpastes/champions-mc/20261012',collection_file:path.join(root,'data/raw/vgcpastes/champions-mc/20261012/none.json')};
prepare(output,{datasetId:'a10-p0-proposed-20261013',predecessor_dataset:'mb-mc-v3-userteam-all-train',sources:['mb','mc','user',custom],scope_exception_note:'A10-P0 imports only the user-authorized EVs=Yes rows added after the 2026-10-06 M-C/M-B boundary; no training, auto-refresh, or unrelated sources are included.',datasetConfig:{sources:['mb','mc','user'],dataset_schema:'pa3-team-record-v2',supersedes_dataset:'mb-mc-v3-userteam-all-train',source_selection:'A10-P0 user-authorized incremental M-B/M-C EVs Yes delta; v3 predecessor immutable'}});
