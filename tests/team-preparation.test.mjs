import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {ROOT, FORMAT, DATASET_ID, DATASET_ID_V3, normalizeRecord, megaFormAbilityResolution, fingerprint, groupID, splitGroups, stable, hash, prepare} from '../scripts/prepare_teams.mjs';
import {SUBMISSION as USER_SUBMISSION} from '../scripts/import_user_pokepaste.mjs';
import {validateDataset} from '../scripts/validate_team_records.mjs';

const require = createRequire(import.meta.url);
const {TeamValidator, Battle, Teams} = require('../vendor/pokemon-showdown/dist/sim/index.js');
const records = fs.readFileSync(path.join(ROOT, 'data/raw/vgcpastes/champions-mb/20261006/teams.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
const original = records.find(r => r.source.team_id === 'MB861');
const prepared = normalizeRecord(original, 'mb');
const output = path.join(ROOT, 'data/teams', DATASET_ID);
const readRows = name => fs.readFileSync(path.join(output, name), 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
const outputV3 = path.join(ROOT, 'data/teams', DATASET_ID_V3);
const readRowsV3 = name => fs.readFileSync(path.join(outputV3, name), 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);

test('team identity ignores presentation order and nicknames but preserves strategic changes', () => {
  assert.deepEqual(prepared.problems, []);
  const reordered = structuredClone(prepared.sets).reverse();
  for (const set of reordered) { set.moves.reverse(); set.name = 'arbitrary nickname'; set.shiny = !set.shiny; }
  assert.equal(fingerprint(reordered), prepared.fingerprint);
  const changed = structuredClone(prepared.sets);
  changed[0].evs.hp += 1;
  assert.notEqual(fingerprint(changed), prepared.fingerprint);
  assert.equal(groupID(changed), prepared.group_id);
});

test('roster grouping keeps set variants together and distinguishes resource forms', () => {
  const changed = structuredClone(prepared.sets);
  changed[1].moves = ['Protect', 'Substitute', 'Rest', 'Sleep Talk'];
  changed[1].item = 'Leftovers';
  assert.equal(groupID(changed), prepared.group_id);
  changed[0].item = '';
  assert.notEqual(groupID(changed), prepared.group_id);
});

test('no allocation, nature, moves or ambiguous base ability is silently invented', () => {
  const expected = {MB382: 'missing_or_multiple_allocation_lines', MB391: 'missing_explicit_nature', MB576: 'requires_four_explicit_moves', MB852: 'ambiguous_base_ability', MB661: 'item_not_explicit'};
  for (const [id, code] of Object.entries(expected)) {
    const record = records.find(r => r.source.team_id === id);
    const before = stable(record);
    const result = normalizeRecord(record, 'mb');
    assert(result.problems.some(p => p.code === code), id);
    assert.equal(stable(record), before, 'Raw source must not be mutated');
  }
  const explicitNoItem = structuredClone(original);
  explicitNoItem.members[0].header = explicitNoItem.members[0].header.replace(/ @ .+$/, ' @ No Item');
  explicitNoItem.members[0].raw_export_block = explicitNoItem.members[0].raw_export_block.replace(/ @ [^\r\n]+/, ' @ No Item');
  assert.deepEqual(normalizeRecord(explicitNoItem, 'mb').problems, []);
});

test('the pinned validator rejects known illegal source sets', () => {
  for (const id of ['MB495', 'MB135']) {
    const result = normalizeRecord(records.find(r => r.source.team_id === id), 'mb');
    assert(result.problems.some(p => p.code === 'reference_illegal'), id);
  }
});

test('all source records are accounted for and all eligible teams belong to training', () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(output, 'manifest.json')));
  const teams = readRows('all.jsonl');
  const index = readRows('source-index.jsonl');
  assert.equal(index.length, 1207);
  assert.equal(new Set(index.map(r => r.source_team_id)).size, 1207);
  assert(index.every(r => /^(MB|MC)\d+$/.test(r.source_team_id)));
  assert.equal(manifest.accepted_source_rows + manifest.quarantined_source_rows, 1207);
  const splitByGroup = new Map();
  const seen = new Set();
  for (const split of ['train', 'dev', 'final']) {
    const rows = readRows(`${split}.jsonl`);
    assert.equal(rows.length, manifest.counts[split].teams);
    for (const row of rows) {
      assert.equal(row.split, split);
      assert(!seen.has(row.team_id)); seen.add(row.team_id);
      if (splitByGroup.has(row.group_id)) assert.equal(splitByGroup.get(row.group_id), split);
      splitByGroup.set(row.group_id, split);
    }
  }
  assert.equal(seen.size, teams.length);
  assert.equal(manifest.counts.train.teams, teams.length);
  assert.equal(manifest.counts.dev.teams, 0);
  assert.equal(manifest.counts.final.teams, 0);
  assert(teams.every(t => t.split === 'train'));
  assert(index.filter(r => r.status === 'accepted').every(r => r.split === 'train'));
  assert.equal(manifest.heldout_team_generalization_available, false);
  assert.deepEqual(splitGroups(teams.map(r => r.group_id)), splitGroups(teams.map(r => r.group_id).reverse()));
  for (const [name, digest] of Object.entries(manifest.artifact_sha256)) assert.equal(hash(fs.readFileSync(path.join(output, name))), digest);
});

test('every normalized team remains reference-legal and its packed representation roundtrips', () => {
  const validator = new TeamValidator(FORMAT);
  for (const team of readRows('all.jsonl')) {
    const sets = team.members.map(m => m.showdown_set);
    assert.equal(validator.validateTeam(structuredClone(sets)), null, team.team_id);
    assert.equal(fingerprint(Teams.unpack(Teams.pack(sets))), team.team_fingerprint, team.team_id);
  }
});

test('stored starting stats match reference preview objects; Tera and OTS stay inactive', () => {
  const teams = readRows('all.jsonl');
  // Cover every distinct species, Mega option and nature combination rather than one arbitrary team.
  const covered = new Set();
  for (const team of teams) {
    const signatures = team.members.map(m => `${m.species_id}/${m.nature}/${m.resource_forms.join(',')}`);
    if (signatures.every(s => covered.has(s))) continue;
    signatures.forEach(s => covered.add(s));
    const b = new Battle({formatid: FORMAT, seed: [2026, 10, 6, 2]});
    try {
      b.setPlayer('p1', {name: 'p1', team: structuredClone(team.members.map(m => m.showdown_set))});
      b.setPlayer('p2', {name: 'p2', team: structuredClone(team.members.map(m => m.showdown_set))});
      for (let i = 0; i < 6; i++) {
        const p = b.p1.pokemon[i];
        assert.deepEqual({...p.baseStoredStats, hp: p.maxhp}, team.members[i].stats_at_battle_start);
        assert.equal(p.canTerastallize, null);
      }
      assert(!b.log.some(line => line.startsWith('|showteam|')));
    } finally { b.destroy(); }
  }
});

test('rebuilding the frozen snapshot is byte-for-byte deterministic', () => {
  const digest = hash(fs.readFileSync(path.join(output, 'manifest.json')));
  prepare(output);
  assert.equal(hash(fs.readFileSync(path.join(output, 'manifest.json'))), digest);
});

test('the approved manual source is imported without touching the frozen v2 records', () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(outputV3, 'manifest.json')));
  const v2 = readRows('all.jsonl');
  const v3 = readRowsV3('all.jsonl');
  assert.equal(v3.length, 1137);
  assert.equal(manifest.unique_eligible_teams, 1137);
  assert.equal(manifest.counts.train.teams, 1137);
  assert.equal(manifest.counts.dev.teams, 0);
  assert.equal(manifest.counts.final.teams, 0);
  assert.equal(manifest.roster_groups, 904);
  assert.equal(manifest.source_rows, 1208);
  assert.equal(manifest.accepted_source_rows, 1152);
  assert.equal(manifest.quarantined_source_rows, 56);
  assert.deepEqual(manifest.predecessor_verification, {dataset_id: DATASET_ID, teams_verified_unchanged: 1136, new_teams: 1});
  // Every predecessor record is carried over byte-for-byte.
  const current = new Map(v3.map(team => [team.team_id, team]));
  for (const team of v2) assert.equal(stable(current.get(team.team_id)), stable(team), team.team_id);
  // The new team is the only one from the manual provider.
  const manual = v3.filter(team => team.source.provider === 'user_submitted_pokepaste');
  assert.equal(manual.length, 1);
  assert.equal(manual[0].schema_version, 'pa3-team-v2');
  assert.deepEqual(manual[0].source.urls, [USER_SUBMISSION.pokepaste_url]);
  assert.deepEqual(manual[0].source.repository_team_ids, [USER_SUBMISSION.submission_id]);
  assert.deepEqual(manual[0].source.source_tabs, ['User Submission']);
  assert.equal(manual[0].source.submission.approved_at, '2026-10-08');
  assert.equal(manual[0].eligibility.original_set_not_imputed, true);
  assert.equal(manual[0].split, 'train');
  assert.deepEqual(manual[0].members.map(m => m.species_id).sort(), ['charizard', 'garchomp', 'gengar', 'indeedee', 'sneasler', 'whimsicott']);
  // The manual team is genuinely new: no roster-group neighbour and no exact duplicate.
  assert(v2.every(team => team.team_fingerprint !== manual[0].team_fingerprint));
  assert(v2.every(team => team.group_id !== manual[0].group_id));
  // The declared Mega-form ability and the reference-resolved base ability are both recorded.
  const resolution = manual[0].eligibility.mega_ability_resolutions[0];
  assert.equal(resolution.source_value, 'Drought');
  assert.equal(resolution.reference_value, 'Blaze');
  assert.equal(resolution.policy, 'reference_validator_base_form_default');
  assert.deepEqual(resolution.base_ability_choices, ['Blaze', 'Solar Power']);
  assert.equal(resolution.user_confirmation_required, true);
  assert.deepEqual(manual[0].members.find(m => m.species_id === 'charizard').format_defaults.ability,
    {defaulted_by_format: true, policy: 'reference_validator_base_form_default', source_ability: 'Drought',
      value: 'Blaze', base_ability_choices: ['Blaze', 'Solar Power'], user_confirmation_required: true});
  // Source index accounts for every row including the manual submission.
  const indexV3 = readRowsV3('source-index.jsonl');
  assert.equal(indexV3.length, 1208);
  assert(indexV3.some(row => row.source_team_id === USER_SUBMISSION.submission_id && row.status === 'accepted' && row.split === 'train'));
  // The v2 dataset never gains the manual source.
  assert(readRows('source-index.jsonl').every(row => !row.source_team_id.startsWith('UT')));
});

test('Mega-form ability defaults stay quarantined unless the dataset approves them', () => {
  const record = fs.readFileSync(path.join(ROOT, 'data/raw/user-pokepaste/20261008/teams.jsonl'), 'utf8').trim().split('\n').map(JSON.parse)[0];
  const strict = normalizeRecord(record, 'user');
  assert(strict.problems.some(problem => problem.code === 'ambiguous_base_ability'));
  const approved = normalizeRecord(record, 'user', {approvedMegaAbilityDefault: true});
  assert.deepEqual(approved.problems, []);
  assert.equal(approved.fingerprint, 'e06f6fdfc7d1657067bc846c248e8b657197dffb31e1dc23351106e88abacede');
  assert.equal(approved.group_id, 'a79b53e935bcf9dfd20bce10839b8f1fede995d31125f5d4675ca342646e4d6c');
  assert.equal(approved.mega_ability_resolutions.length, 1);
  assert.equal(megaFormAbilityResolution({species: 'Gengar', ability: 'Cursed Body'}, {species: 'Gengar', ability: 'Cursed Body'}), null);
});

test('both frozen datasets satisfy their declared accepted-record schema', () => {
  assert.deepEqual(validateDataset(output, 'pa3-team-v1'), []);
  assert.deepEqual(validateDataset(outputV3, 'pa3-team-v2'), []);
});
