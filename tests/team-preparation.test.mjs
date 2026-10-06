import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {ROOT, FORMAT, DATASET_ID, normalizeRecord, fingerprint, groupID, splitGroups, stable, hash, prepare} from '../scripts/prepare_teams.mjs';

const require = createRequire(import.meta.url);
const {TeamValidator, Battle, Teams} = require('../vendor/pokemon-showdown/dist/sim/index.js');
const records = fs.readFileSync(path.join(ROOT, 'data/raw/vgcpastes/champions-mb/20261006/teams.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
const original = records.find(r => r.source.team_id === 'MB861');
const prepared = normalizeRecord(original, 'mb');
const output = path.join(ROOT, 'data/teams', DATASET_ID);
const readRows = name => fs.readFileSync(path.join(output, name), 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);

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
