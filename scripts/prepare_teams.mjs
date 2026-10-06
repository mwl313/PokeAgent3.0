import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const ORACLE_COMMIT = '14546894d86f9589ac11130c510bbe73b6968665';
export const FORMAT = 'gen9championsvgc2026regmc';
export const DATASET_ID = 'mb-mc-v2-all-train';
export const SEED = 20261006;
const require = createRequire(import.meta.url);
const oraclePath = path.join(ROOT, 'vendor/pokemon-showdown');
const {Teams, TeamValidator, Battle, toID} = require(path.join(oraclePath, 'dist/sim/index.js'));
const validator = new TeamValidator(FORMAT);
const dex = validator.dex;
const statIDs = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const SOURCE_TABS = {mb: 'Champions M-B', mc: 'Champions M-C'};

export function stable(value) {
  if (Array.isArray(value)) return '[' + value.map(stable).join(',') + ']';
  if (value && typeof value === 'object') return '{' + Object.keys(value).sort().filter(k => value[k] !== undefined)
    .map(k => JSON.stringify(k) + ':' + stable(value[k])).join(',') + '}';
  return JSON.stringify(value);
}
export const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const readJSON = name => JSON.parse(fs.readFileSync(name, 'utf8'));
const readJSONL = name => fs.readFileSync(name, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
const writeJSON = (dir, name, value) => fs.writeFileSync(path.join(dir, name), JSON.stringify(value, null, 2) + '\n');
const writeJSONL = (dir, name, rows) => fs.writeFileSync(path.join(dir, name), rows.map(stable).join('\n') + (rows.length ? '\n' : ''));
const unique = values => [...new Set(values)].sort();

export function canonicalSet(set) {
  // Move/roster order and nicknames are randomized presentation, not team identity.
  // Tera is disabled by this Champions mod; shiny/ball are cosmetic.
  const species = dex.species.get(set.species);
  const result = {
    species: species.id, item: toID(set.item), ability: toID(set.ability), nature: toID(set.nature),
    moves: set.moves.map(toID).sort(),
    evs: Object.fromEntries(statIDs.map(id => [id, set.evs?.[id] ?? 0])),
    ivs: Object.fromEntries(statIDs.map(id => [id, set.ivs?.[id] ?? 31])),
    level: set.level,
    gender: set.gender || species.gender || 'reference_random_at_reset',
  };
  // Preserve any potentially meaningful optional set parameters conservatively.
  for (const field of ['happiness', 'hpType', 'gigantamax', 'dynamaxLevel']) {
    if (set[field] !== undefined) result[field] = set[field];
  }
  return result;
}

export const fingerprint = sets => hash(stable(sets.map(canonicalSet).sort((a, b) => stable(a).localeCompare(stable(b), 'en'))));

export function rosterKey(sets) {
  // Resolve explicit Mega notation and item-enabled Megas with the same reference rules.
  return sets.map(set => {
    const {outOfBattleSpecies, tierSpecies} = validator.getValidationSpecies(set);
    return {species: outOfBattleSpecies.id, resource_form: tierSpecies.id};
  }).sort((a, b) => stable(a).localeCompare(stable(b), 'en'));
}
export const groupID = sets => hash(stable(rosterKey(sets)));

export function splitGroups(ids) {
  const ordered = unique(ids).sort((a, b) => {
    const ah = hash(`${SEED}:${a}`), bh = hash(`${SEED}:${b}`);
    return ah < bh ? -1 : ah > bh ? 1 : a.localeCompare(b, 'en');
  });
  // User override: every eligible team is used for training; no held-out groups.
  return new Map(ordered.map(id => [id, 'train']));
}

function issue(code, member, details) { return {code, member_index: member, details}; }

export function normalizeRecord(record, tab) {
  assert(SOURCE_TABS[tab], 'Only the two approved batches are allowed');
  const problems = [];
  if (record.source.evs.trim().toLowerCase() !== 'yes') problems.push(issue('source_not_evs_yes', null, 'Outside selected pool'));
  const originalText = record.members.map(m => m.raw_export_block).join('\n\n');
  const imported = Teams.import(originalText);
  if (imported.length !== 6 || record.members.length !== 6) {
    return {problems: [issue('member_count', null, imported.length)]};
  }
  const before = structuredClone(imported);
  for (let i = 0; i < 6; i++) {
    const raw = record.members[i];
    const lines = raw.raw_export_block.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
    const allocationLines = lines.filter(s => /^EVs: /i.test(s));
    if (allocationLines.length !== 1) problems.push(issue('missing_or_multiple_allocation_lines', i, allocationLines));
    else {
      const parsed = Object.fromEntries(statIDs.map(k => [k, 0]));
      const seen = new Set();
      for (const part of allocationLines[0].slice(5).split('/')) {
        const match = part.trim().match(/^(\d+)\s+(HP|Atk|Def|SpA|SpD|Spe)$/i);
        if (!match || seen.has(match[2].toLowerCase())) {
          problems.push(issue('ambiguous_allocation', i, part)); continue;
        }
        const key = match[2].toLowerCase(); seen.add(key); parsed[key] = Number(match[1]);
      }
      if (stable(parsed) !== stable(before[i].evs)) problems.push(issue('allocation_parser_disagreement', i, {parsed, oracle: before[i].evs}));
      if (Object.values(parsed).some(n => n > 32)) problems.push(issue('ambiguous_allocation_unit', i, 'Exceeds the pinned Champions per-stat bound; no EV conversion is authorized'));
    }
    if (before[i].moves.length !== 4) problems.push(issue('requires_four_explicit_moves', i, before[i].moves));
    if (!lines.some(s => / Nature$/i.test(s)) || !before[i].nature) problems.push(issue('missing_explicit_nature', i, raw.header));
    if (!raw.header.includes(' @ ') || !raw.header.split(' @ ')[1]?.trim()) problems.push(issue('item_not_explicit', i, raw.header));
    if (!before[i].ability) {
      const options = unique(Object.values(dex.species.get(before[i].species).abilities));
      if (options.length === 1) imported[i].ability = options[0];
      else problems.push(issue('missing_ambiguous_ability', i, options));
    }
    // Nicknames must not cause false bans or survive into policy-facing set data.
    imported[i].name = '';
  }
  const referenceProblems = validator.validateTeam(imported) || [];
  for (const message of referenceProblems) problems.push(issue('reference_illegal', null, message));
  const normalizations = [];
  for (let i = 0; i < 6; i++) {
    const old = before[i], current = imported[i];
    for (const key of unique([...Object.keys(old), ...Object.keys(current)])) {
      if (stable(old[key]) === stable(current[key])) continue;
      normalizations.push({member_index: i, field: key, before: old[key] ?? null, after: current[key] ?? null});
      if (['name', 'gender', 'level', 'teraType'].includes(key)) continue;
      if (key === 'species') {
        const previous = dex.species.get(old.species);
        const item = dex.items.get(current.item);
        if (toID(previous.battleOnly) === toID(current.species) || toID(item.forcedForme) === toID(current.species)) continue;
      }
      if (key === 'ability') {
        const choices = unique(Object.values(dex.species.get(current.species).abilities));
        if (choices.length === 1 && choices[0] === current.ability) continue;
        problems.push(issue('ambiguous_base_ability', i, {source: old.ability, oracle_default: current.ability, base_species: current.species, alternatives: choices}));
        continue;
      }
      if (['species', 'nature', 'item'].includes(key) && toID(old[key]) === toID(current[key])) continue;
      problems.push(issue('unapproved_set_change', i, {field: key, before: old[key], after: current[key]}));
    }
    if (current.moves.length !== 4 && !problems.some(p => p.code === 'requires_four_explicit_moves' && p.member_index === i)) problems.push(issue('requires_four_explicit_moves', i, current.moves));
  }
  if (problems.length) return {problems, reference_problems: referenceProblems, normalizations};
  const sets = imported.map(s => ({...s, name: ''}));
  return {problems, sets, before, normalizations, fingerprint: fingerprint(sets), group_id: groupID(sets), roster: rosterKey(sets)};
}

function sourceRecord(record, tab, normalizations) {
  return {
    ...record.source, source_tab: SOURCE_TABS[tab], fetched_at: record.fetch.fetched_at,
    raw_sha256: record.fetch.sha256, team_text_sha256: record.team_text_sha256,
    response_url: record.fetch.response_url,
    collection_record_file: `data/raw/vgcpastes/champions-${tab}/20261006/teams.jsonl`,
    raw_html_file: `data/raw/vgcpastes/champions-${tab}/20261006/${record.raw_html_path}`,
    team_text_file: `data/raw/vgcpastes/champions-${tab}/20261006/${record.team_text_path}`,
    normalization_changes: normalizations,
  };
}

function battleMembers(result, record, tab) {
  const battle = new Battle({formatid: FORMAT, seed: [2026, 10, 6, 1]});
  try {
    battle.setPlayer('p1', {name: 'reference-a', team: structuredClone(result.sets)});
    battle.setPlayer('p2', {name: 'reference-b', team: structuredClone(result.sets)});
    assert.equal(battle.requestState, 'teampreview');
    return result.sets.map((set, i) => {
      const pokemon = battle.p1.pokemon[i];
      assert.equal(pokemon.canTerastallize, null, 'Unexpected resource rule');
      const raw = record.members[i];
      const species = dex.species.get(set.species);
      const defaults = {};
      if (!raw.fields.Level) defaults.level = {defaulted_by_format: true, parser_default: result.before[i].level, effective: set.level};
      defaults.level_adjustment = {source_level: result.before[i].level, effective: set.level, rule: 'Adjust Level = 50'};
      if (!raw.fields.IVs) defaults.ivs = {defaulted_by_format: true, values: set.ivs};
      if (!result.before[i].gender) defaults.gender = {defaulted_by_format: true, policy: species.gender ? 'fixed_species_gender' : 'reference_rng_each_reset', value: species.gender || null};
      if (!result.before[i].ability || toID(result.before[i].ability) !== toID(set.ability)) defaults.ability = {defaulted_by_format: true, policy: 'unique_reference_base_ability', value: set.ability};
      return {
        species: species.name, species_id: species.id, ability: set.ability, item: set.item || null,
        nature: set.nature, moves: set.moves.map(m => dex.moves.get(m).name),
        level: set.level, ivs: set.ivs, gender: set.gender || species.gender || null,
        allocation: {raw_line: raw.allocations[0].raw_line, source_unit: 'champions_stat_points',
          unit_evidence: `${SOURCE_TABS[tab]} source context; pinned Champions Teams.import EVs field, per-stat bound 32 and total bound ${validator.ruleTable.evLimit}; no numerical conversion`,
          parsed_values: result.before[i].evs, engine_values: set.evs, conversion: {kind: 'none'}},
        stats_at_battle_start: {...pokemon.baseStoredStats, hp: pokemon.maxhp},
        stats_context: 'Reference team-preview state before lead switch-in effects, items, boosts, weather or Mega Evolution',
        format_defaults: defaults, raw_export_block: raw.raw_export_block,
        source_member_index: i, showdown_set: set,
        resource_forms: pokemon.canMegaEvo ? [pokemon.canMegaEvo] : [],
      };
    }).sort((a, b) => stable(canonicalSet(a.showdown_set)).localeCompare(stable(canonicalSet(b.showdown_set)), 'en'));
  } finally { battle.destroy(); }
}

function assertReference() {
  assert.equal(execFileSync('git', ['-C', oraclePath, 'rev-parse', 'HEAD'], {encoding: 'utf8'}).trim(), ORACLE_COMMIT);
  execFileSync('git', ['-C', oraclePath, 'diff', '--quiet', 'HEAD']);
  assert(!fs.existsSync(path.join(oraclePath, 'config/custom-formats.ts')), 'Unexpected custom formats');
  assert(validator.format.exists && validator.format.mod === 'champions' && validator.format.gameType === 'doubles');
  assert.equal(validator.ruleTable.adjustLevel, 50);
  assert.equal(validator.ruleTable.evLimit, 66);
  assert(!validator.ruleTable.has('forceopenteamsheets'));
}

export function prepare(outputDir = path.join(ROOT, 'data/teams', DATASET_ID)) {
  assertReference();
  const accepted = new Map(), quarantined = [], index = [], inputs = [];
  let sourceCount = 0;
  for (const tab of ['mb', 'mc']) {
    const relative = `data/raw/vgcpastes/champions-${tab}/20261006`;
    const root = path.join(ROOT, relative);
    const collection = readJSON(path.join(ROOT, `docs/20261006-${tab}-team-collection.json`));
    for (const [name, digest] of Object.entries(collection.artifact_sha256)) {
      assert.equal(hash(fs.readFileSync(path.join(root, name))), digest, `Raw source changed: ${tab}/${name}`);
    }
    const records = readJSONL(path.join(root, 'teams.jsonl'));
    const selected = readJSON(path.join(root, 'sheet_snapshot.json')).rows.filter(r => r.evs.trim().toLowerCase() === 'yes');
    assert.deepEqual(records.map(r => r.source.team_id).sort(), selected.map(r => r.team_id).sort());
    inputs.push({tab: SOURCE_TABS[tab], path: relative, records: records.length,
      teams_jsonl_sha256: hash(fs.readFileSync(path.join(root, 'teams.jsonl'))),
      sheet_snapshot_sha256: hash(fs.readFileSync(path.join(root, 'sheet_snapshot.json')))});
    for (const record of records) {
      sourceCount++;
      assert.equal(hash(fs.readFileSync(path.join(root, record.raw_html_path))), record.fetch.sha256);
      assert.equal(hash(fs.readFileSync(path.join(root, record.team_text_path))), record.team_text_sha256);
      const result = normalizeRecord(record, tab);
      const source = sourceRecord(record, tab, result.normalizations || []);
      if (result.problems.length) {
        quarantined.push({source, reasons: result.problems, reference_problems: result.reference_problems || [], raw_members: record.members});
        index.push({source_team_id: record.source.team_id, status: 'quarantined', reason_codes: unique(result.problems.map(p => p.code))});
        continue;
      }
      let team = accepted.get(result.fingerprint);
      if (!team) {
        const members = battleMembers(result, record, tab);
        team = {schema_version: 'pa3-team-v1', team_id: 'pa3-' + result.fingerprint,
          team_fingerprint: result.fingerprint, group_id: result.group_id, roster_group_key: result.roster,
          source: {provider: 'VGCPastes', repository_team_ids: [], urls: [], raw_sha256: source.raw_sha256,
            fetched_at: source.fetched_at, source_tabs: [], records: [], redistribution_permission: 'not_established_by_public_access'},
          regulation: {battle_format: FORMAT, oracle_commit: ORACLE_COMMIT, team_sheet: 'closed', series: 'bo1'},
          members, eligibility: {all_six_explicit_allocations: true, reference_format_legal: true, original_set_not_imputed: true},
          split: null};
        accepted.set(result.fingerprint, team);
      }
      team.source.records.push(source);
      team.source.repository_team_ids.push(record.source.team_id);
      team.source.urls.push(record.source.pokepaste_url);
      team.source.source_tabs.push(SOURCE_TABS[tab]);
      index.push({source_team_id: record.source.team_id, status: 'accepted', team_id: team.team_id, group_id: team.group_id});
    }
  }
  const teams = [...accepted.values()].sort((a, b) => a.team_fingerprint.localeCompare(b.team_fingerprint, 'en'));
  const splits = splitGroups(teams.map(t => t.group_id));
  for (const t of teams) {
    t.split = splits.get(t.group_id);
    for (const field of ['repository_team_ids', 'urls', 'source_tabs']) t.source[field] = unique(t.source[field]);
    t.source.records.sort((a, b) => a.team_id.localeCompare(b.team_id, 'en'));
  }
  for (const row of index) if (row.status === 'accepted') row.split = splits.get(row.group_id);
  const splitRows = Object.fromEntries(['train', 'dev', 'final'].map(name => [name, teams.filter(t => t.split === name)]));
  const direct = {
    species: unique(teams.flatMap(t => t.members.map(m => m.species_id))),
    resource_forms: unique(teams.flatMap(t => t.members.flatMap(m => m.resource_forms.map(toID)))),
    moves: unique(teams.flatMap(t => t.members.flatMap(m => m.moves.map(toID)))),
    items: unique(teams.flatMap(t => t.members.map(m => toID(m.item))).filter(Boolean)),
    abilities: unique(teams.flatMap(t => t.members.map(m => toID(m.ability)))),
    natures: unique(teams.flatMap(t => t.members.map(m => toID(m.nature)))),
  };
  direct.resource_abilities = unique(direct.resource_forms.flatMap(id => Object.values(dex.species.get(id).abilities).map(toID)));
  const inventory = {oracle_commit: ORACLE_COMMIT, format: FORMAT, covers_all_splits: true, direct,
    dependency_status: 'Direct inventory frozen. Engine implementation must resolve callback, called-move, form, field and interaction dependencies; this is not a claim of implemented effect closure.',
    broad_dynamic_effects_present: direct.moves.filter(id => ['metronome','copycat','transform','mimic','sketch','sleeptalk','assist','naturepower','instruct','mefirst','mirrormove'].includes(id)),
    engine_scope_may_not_exclude_valid_teams: true};
  const groups = [...splits].map(([group_id, split]) => ({group_id, split, team_ids: teams.filter(t => t.group_id === group_id).map(t => t.team_id)}));
  const acceptedSourceRows = index.filter(r => r.status === 'accepted').length;
  const sourceInputSha = hash(stable(inputs));
  const manifest = {
    dataset_id: DATASET_ID, schema_version: 'pa3-frozen-team-pool-v1', format: FORMAT, oracle_commit: ORACLE_COMMIT,
    source_selection: 'User-approved frozen M-B and M-C EVs Yes batches only; M-A excluded; no future automatic additions',
    source_inputs: inputs, source_input_sha256: sourceInputSha, source_rows: sourceCount,
    accepted_source_rows: acceptedSourceRows, quarantined_source_rows: quarantined.length,
    unique_eligible_teams: teams.length, duplicate_source_rows_collapsed: acceptedSourceRows - teams.length,
    roster_groups: splits.size, split_seed: SEED, split_algorithm: 'All eligible groups assigned to train by user request; seeded group ordering only; no dev/final holdouts',
    heldout_team_generalization_available: false,
    supersedes_dataset: 'mb-mc-v1',
    fingerprint_algorithm: 'SHA256 canonical JSON; canonical reference base forms/sets; ignore nickname, roster order, move order, shiny/ball and disabled Tera; preserve gender/default policy and all battle-relevant values',
    group_algorithm: 'Unordered reference out-of-battle species plus item-enabled resource form for all six members',
    counts: Object.fromEntries(Object.entries(splitRows).map(([name, rows]) => [name, {teams: rows.length, groups: new Set(rows.map(t => t.group_id)).size}])),
    quarantine_reason_counts: Object.fromEntries(unique(quarantined.flatMap(r => r.reasons.map(p => p.code))).map(code => [code, quarantined.filter(r => r.reasons.some(p => p.code === code)).length])),
    training_sampling: 'uniform_unique_team_from_train_only', final_results_for_tuning: false,
    information_boundary: 'Canonical records contain full private sets for the simulator. Actor inputs must come from player-specific observations, never source IDs or hidden opponent set records.',
    source_permission: 'Public collection provenance preserved; redistribution permission not inferred. Keep dataset local.',
    state: 'frozen_ready_for_engine_integration', engine_effect_closure_implemented: false,
    counts_are_not_training_results: true,
  };
  // Write to a temporary sibling and refuse to replace a different frozen dataset.
  const staging = outputDir + `.staging-${process.pid}`;
  fs.mkdirSync(staging, {recursive: false});
  try {
    writeJSONL(staging, 'all.jsonl', teams);
    for (const [name, rows] of Object.entries(splitRows)) {
      writeJSONL(staging, `${name}.jsonl`, rows);
      writeJSONL(staging, `${name}.showdown.jsonl`, rows.map(t => ({team_id: t.team_id, team: t.members.map(m => m.showdown_set)})));
    }
    writeJSONL(staging, 'quarantine.jsonl', quarantined);
    writeJSONL(staging, 'source-index.jsonl', index.sort((a, b) => a.source_team_id.localeCompare(b.source_team_id, 'en')));
    writeJSON(staging, 'groups.json', groups);
    writeJSON(staging, 'inventory.json', inventory);
    writeJSON(staging, 'duplicates.json', teams.filter(t => t.source.records.length > 1).map(t => ({team_id: t.team_id, source_team_ids: t.source.repository_team_ids})));
    fs.writeFileSync(path.join(staging, 'all_teams.txt'), teams.map(t => `=== ${t.team_id} (${t.split}) ===\n\n` + Teams.export(t.members.map(m => m.showdown_set))).join('\n'));
    manifest.artifact_sha256 = Object.fromEntries(fs.readdirSync(staging).sort().map(name => [name, hash(fs.readFileSync(path.join(staging, name)))]));
    writeJSON(staging, 'manifest.json', manifest);
    if (fs.existsSync(outputDir)) {
      for (const name of fs.readdirSync(staging)) assert.equal(hash(fs.readFileSync(path.join(staging, name))), hash(fs.readFileSync(path.join(outputDir, name))), `Frozen dataset differs: ${name}; choose a new explicitly versioned pool`);
      fs.rmSync(staging, {recursive: true});
    } else fs.renameSync(staging, outputDir);
  } catch (error) {
    fs.rmSync(staging, {recursive: true, force: true}); throw error;
  }
  return manifest;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const output = process.argv[2] ? path.resolve(process.argv[2]) : path.join(ROOT, 'data/teams', DATASET_ID);
  fs.mkdirSync(path.dirname(output), {recursive: true});
  console.log(JSON.stringify(prepare(output), null, 2));
}
