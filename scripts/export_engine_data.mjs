// Development-only compiler. Never imported by the Rust/Python battle loop.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {computeDynamicClosure} from './dynamic_closure.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ref = path.join(root, 'vendor/pokemon-showdown');
const require = createRequire(import.meta.url);
const {TeamValidator, Battle, PRNG, toID} = require(path.join(ref, 'dist/sim'));
const pin = '14546894d86f9589ac11130c510bbe73b6968665';
const format = 'gen9championsvgc2026regmc';
assert.equal(execFileSync('git', ['rev-parse', 'HEAD'], {cwd: ref, encoding: 'utf8'}).trim(), pin);
assert.equal(execFileSync('git', ['diff', '--name-only', 'HEAD'], {cwd: ref, encoding: 'utf8'}).trim(), '');
const validator = new TeamValidator(format);
const dex = validator.dex;
const output = path.join(root, 'engine/data');
fs.mkdirSync(output, {recursive: true});
const hash = x => crypto.createHash('sha256').update(x).digest('hex');
const statNames = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const sorted = xs => xs.sort((a, b) => a.id < b.id ? -1 : a.id > b.id ? 1 : 0);
const unique = xs => [...new Set(xs)].sort();
const callbacks = [];
function encode(value, owner) {
  if (typeof value === 'function') {
    const source = value.toString();
    callbacks.push({key: owner, sha256: hash(source), source, status: 'pending_native_port'});
    return {callback: owner};
  }
  if (Array.isArray(value)) return value.map((x, i) => encode(x, `${owner}.${i}`));
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.keys(value).sort().filter(k => value[k] !== undefined)
      .map(k => [k, encode(value[k], `${owner}.${k}`)]));
  }
  return value;
}
const raw = {
  species: sorted(dex.species.all().filter(x => x.exists)),
  moves: sorted(dex.moves.all().filter(x => x.exists)),
  items: sorted(dex.items.all().filter(x => x.exists)),
  abilities: sorted(dex.abilities.all().filter(x => x.exists)),
  natures: sorted(dex.natures.all().filter(x => x.exists)),
  types: sorted(dex.types.all().filter(x => x.exists)),
  conditions: sorted(Object.keys(dex.data.Conditions).map(id => dex.conditions.get(id)).filter(x => x.exists)),
};
// Keep reference metadata for nonstandard catalogue entries (e.g. Bird on
// MissingNo.) without treating these types/species as legal regulation entries.
const typeNames = new Set(raw.types.map(x => x.name));
for (const name of unique([...raw.species.flatMap(s => s.types), ...raw.moves.map(m => m.type)])) {
  if (!typeNames.has(name)) { raw.types.push(dex.types.get(name)); typeNames.add(name); }
}
sorted(raw.types);
// Include move/item/ability conditions as separate effects. Some lack an exists property.
const conditions = new Map(raw.conditions.map(x => [x.id, x]));
// Simulator-created effect identities are not keys in data.Conditions.
for (const id of ['drain', 'recoil']) conditions.set(id, dex.conditions.get(id));
for (const kind of ['moves', 'items', 'abilities']) for (const effect of raw[kind]) {
  if (effect.condition) conditions.set(effect.id, {id: effect.id, ...effect.condition});
}
raw.conditions = sorted([...conditions.values()]);
const tables = Object.fromEntries(Object.entries(raw).map(([kind, values]) => [kind, values.map((v, i) => ({
  numeric_id: i + 1, id: v.id, data: encode(v, `${kind}:${v.id}`),
}))]));
const ids = Object.fromEntries(Object.entries(tables).map(([kind, xs]) => [kind, new Map(xs.map(x => [x.id, x.numeric_id]))]));
const startingCandidates = raw.species.filter(s => !s.battleOnly && !validator.checkSpecies(
  {name: s.name, species: s.name, ability: s.abilities['0']}, s, s, {}));
const allowedMoves = raw.moves.filter(m => !validator.checkMove({name: 'Scope'}, m, {}));
const starting = [], unresolved = [];
for (const species of startingCandidates) {
  const abilities = unique(Object.values(species.abilities)).filter(a => !validator.checkAbility(
    {name: species.name, species: species.name}, dex.abilities.get(a), {}));
  const legalMoves = allowedMoves.filter(m => !validator.checkCanLearn(m, species));
  let witness;
  for (const ability of abilities) {
    for (const move of legalMoves) {
      const set = {name: '', species: species.name, ability, item: species.requiredItem || species.requiredItems?.[0] || '',
        moves: [move.name], nature: 'Serious', level: 50,
        evs: {hp: 1, atk: 0, def: 0, spa: 0, spd: 0, spe: 0}};
      if (!validator.validateSet(set)) { witness = set; break; }
    }
    if (witness) break;
  }
  if (!witness) { unresolved.push(species.id); continue; }
  starting.push({species: species.id, witness, learnable_moves: legalMoves.map(m => m.id), abilities: abilities.map(toID)});
}
// Battle-only forms are not legal initial sets. Retain format-permitted forms
// plus the entire dex as a conservative dependency superset for effect porting.
const battleForms = raw.species.filter(s => s.battleOnly && !validator.checkSpecies(
  {name: s.name, species: s.name, ability: s.abilities['0']}, dex.species.get(Array.isArray(s.battleOnly) ? s.battleOnly[0] : s.battleOnly), s, {}));
const legalItems = raw.items.filter(i => !validator.checkItem({name: 'Scope'}, i, {})).map(x => x.id);
const scope = {
  schema: 'pa3-regulation-scope-v1', oracle_commit: pin, format,
  source_scope: 'all_legal_regulation_species_and_sets_not_training_inventory',
  starting_species: starting, format_permitted_battle_forms: battleForms.map(x => x.id),
  mega_forms: battleForms.filter(x => x.isMega).map(x => x.id),
  allowed_items: legalItems, allowed_moves: allowedMoves.map(x => x.id),
  unresolved_starting_candidates: unresolved,
  dependency_policy: 'Full pinned Champions dex retained as a conservative superset. Runtime reachability and callback interactions must be resolved and tested; catalogue inclusion does not certify implementation.',
  callback_count: callbacks.length, closure_certified: false, training_ready: false,
};
const rules = {
  format, oracle_commit: pin, game_type: 'doubles', roster_size: 6, picked_size: validator.ruleTable.pickedTeamSize,
  level: validator.ruleTable.adjustLevel, active_per_side: 2, open_team_sheets: false, tera: false, mega: true,
  level_clause_mod: validator.ruleTable.has('levelclausemod'),
  rule_table: [...validator.ruleTable], value_rules: Object.fromEntries(validator.ruleTable.valueRules),
};
assert.equal(rules.picked_size, 4);
assert.equal(rules.level_clause_mod, false);
const teams = fs.readFileSync(path.join(root, 'data/teams/mb-mc-v2-all-train/train.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
const trainingTeams = teams.map(t => ({id: t.team_id, members: t.members.map(m => ({
  species: ids.species.get(m.species_id), ability: ids.abilities.get(toID(m.ability)), item: ids.items.get(toID(m.item)) || 0,
  nature: ids.natures.get(toID(m.nature)), moves: m.moves.map(x => ids.moves.get(toID(x))),
  points: statNames.map(x => m.allocation.engine_values[x]), ivs: statNames.map(x => m.ivs[x]),
  gender: m.showdown_set.gender || '', level: m.level,
}))}));
const statFixtures = [];
const witnesses = starting.map(x => x.witness);
// Actual reference preview states exercise stat defaults and initialization.
for (let i = 0; i < witnesses.length; i += 6) {
  const team = witnesses.slice(i, i + 6);
  while (team.length < 6) team.push(witnesses[0]);
  const battle = new Battle({formatid: format, seed: [1, 2, 3, 4], p1: {name: 'a', team}, p2: {name: 'b', team}});
  for (const mon of battle.p1.pokemon) statFixtures.push({species: ids.species.get(mon.species.id),
    points: statNames.map(x => mon.set.evs[x]), nature: ids.natures.get(toID(mon.set.nature)),
    stats: [mon.maxhp, ...statNames.slice(1).map(x => mon.storedStats[x])]});
  battle.destroy();
}
for (const t of teams) for (const m of t.members) statFixtures.push({species: ids.species.get(m.species_id),
  points: statNames.map(x => m.allocation.engine_values[x]), nature: ids.natures.get(toID(m.nature)),
  stats: statNames.map(x => m.stats_at_battle_start[x])});
const rng = [];
for (const seed of [[1, 2, 3, 4], [0, 0, 0, 0], [65535, 65535, 65535, 65535], [2026, 10, 6, 1136]]) {
  const prng = new PRNG(seed); const draws = Array.from({length: 1024}, () => prng.rng.next());
  rng.push({seed, draws, final_seed: prng.getSeed()});
}
const probeTeam = teams[0].members.map(m => m.showdown_set);
const initialization = [];
for (let index = 0; index < 32; index++) {
  const team_indices = [index * 31 % teams.length, (index * 37 + 11) % teams.length];
  const seed = [2026, 10, 6, index];
  const battle = new Battle({formatid: format, seed,
    p1: {name: 'a', team: teams[team_indices[0]].members.map(m => m.showdown_set)},
    p2: {name: 'b', team: teams[team_indices[1]].members.map(m => m.showdown_set)}});
  initialization.push({team_indices, seed, final_seed: battle.prng.getSeed(), sides: battle.sides.map(s => s.pokemon.map(p => ({
    species: ids.species.get(p.species.id), gender: p.gender === 'M' ? 1 : p.gender === 'F' ? 2 : 0,
    stats: [p.maxhp, ...statNames.slice(1).map(x => p.storedStats[x])], pp: p.moveSlots.map(m => m.pp),
  })))});
  battle.destroy();
}
const probe = new Battle({formatid: format, seed: [1, 2, 3, 4], p1: {name: 'a', team: probeTeam}, p2: {name: 'b', team: probeTeam}});
const pp = raw.moves.map(m => ({move_id: ids.moves.get(m.id), pp: probe.calculatePP(m, 3)}));
const targeting = [];
for (const target of unique(raw.moves.map(m => m.target))) for (let slot = 0; slot < 2; slot++) {
  const mon = probe.p1.pokemon[slot]; mon.position = slot;
  const chooses = probe.actions.targetTypeChoices(target);
  targeting.push({target, slot, locations: [-2, -1, 0, 1, 2].filter(loc => chooses ? loc !== 0 && probe.validTargetLoc(loc, mon, target) : loc === 0)});
}
const modifiers = [];
for (const value of [0, 1, 2, 3, 47, 100, 101, 65535, 1048576, 0xffffffff]) for (const mod of [1024, 2048, 2732, 3072, 4096, 4915, 5325, 6144, 8192]) {
  modifiers.push({value, modifier: mod, result: probe.modify(value, [mod, 4096])});
}
const actionSpeed = [];
const speedProbe = probe.p1.pokemon[0];
const originalGetStat = speedProbe.getStat;
for (const trick_room of [false, true]) {
  if (trick_room) probe.field.pseudoWeather.trickroom = {id: 'trickroom'};
  else delete probe.field.pseudoWeather.trickroom;
  for (const modified_speed of [0, 1, 1807, 1808, 1809, 8191, 8192, 8193, 9999, 10000]) {
    speedProbe.getStat = () => modified_speed;
    actionSpeed.push({modified_speed, trick_room, result: speedProbe.getActionSpeed()});
  }
}
speedProbe.getStat = originalGetStat;
delete probe.field.pseudoWeather.trickroom;
const recoilRounding = [];
const originalDamage = probe.damage;
// Invoke the real reference recoil calculation and intercept only HP mutation.
probe.damage = amount => amount;
for (const moveId of ['wildcharge', 'doubleedge', 'headsmash']) {
  const move = dex.moves.get(moveId);
  for (let damage = 1; damage <= 1200; damage++) {
    const amount = probe.actions.applyRecoilDamage(damage, move, probe.p1.pokemon[0]);
    recoilRounding.push({damage, fraction: move.recoil, amount});
  }
}
probe.damage = originalDamage;
const ordering = [];
const leftToRightOrdering = [];
const caseRng = new PRNG([20, 26, 10, 6]);
for (let trial = 0; trial < 80; trial++) {
  const seed = [1, 2, 3, trial]; probe.prng = new PRNG(seed);
  const entries = Array.from({length: caseRng.random(1, 40)}, (_, id) => ({id,
    order: caseRng.sample([0, 103, 104, 200, 300]), priority: caseRng.sample([-1, 0, 0.1, 1, 4]),
    speed: caseRng.sample([-200, 0, 100, 100, 200]), subOrder: caseRng.sample([0, 0, 1]), effectOrder: caseRng.sample([0, 0, 1]),
  }));
  const sortedEntries = structuredClone(entries); probe.speedSort(sortedEntries);
  ordering.push({seed, entries, order: sortedEntries.map(x => x.id), final_seed: probe.prng.getSeed()});
  const indexed = entries.map(x => ({...x, index: caseRng.random(0, 4)}));
  leftToRightOrdering.push({entries: indexed, order: [...indexed].sort(Battle.compareLeftToRightOrder).map(x => x.id)});
}
// A large tied handler set exercises inline-buffer spill without truncation.
{
  const seed = [1, 2, 3, 444]; probe.prng = new PRNG(seed);
  const entries = Array.from({length: 70}, (_, id) => ({id, order: 0, priority: 0, speed: 100, subOrder: 0, effectOrder: 0}));
  const sorted = structuredClone(entries); probe.speedSort(sorted);
  ordering.push({seed, entries, order: sorted.map(x => x.id), final_seed: probe.prng.getSeed()});
}
const health = [];
const hpMon = probe.p1.pokemon[0];
for (const maxhp of [1, 101, 150, 201, 999]) for (let hp = 0; hp <= maxhp; hp++) {
  hpMon.maxhp = maxhp; hpMon.hp = hp;
  health.push({hp, max_hp: maxhp, shared: hpMon.getHealth().shared});
}
// Isolate the reference's damage kernel by explicitly supplying resolved event
// modifiers. This is NOT an interaction test or a complete battle fixture.
const damageKernel = [];
const originalRunEvent = probe.runEvent, originalPriorityEvent = probe.priorityEvent;
for (let trial = 0; trial < 512; trial++) {
  const seed = [1, 2, 3, trial]; probe.prng = new PRNG(seed);
  const input = {level: 50, power: caseRng.random(1, 501), attack: caseRng.random(1, 1501), defense: caseRng.random(1, 1501),
    spread: caseRng.randomChance(1, 2), parental_bond_second_hit: caseRng.randomChance(1, 2),
    weather_modifier: caseRng.sample([2048, 4096, 6144]), critical: caseRng.randomChance(1, 2),
    stab_modifier: caseRng.sample([4096, 6144, 8192]), effectiveness: caseRng.random(0, 13) - 6,
    burn: caseRng.randomChance(1, 2), final_modifier: caseRng.sample([2048, 2732, 4096, 5325, 6144]), bypass_protect: caseRng.randomChance(1, 2)};
  probe.runEvent = (event, a, b, c, relay) => event === 'ModifySTAB' ? input.stab_modifier / 4096 :
    event === 'ModifyDamage' ? probe.modify(relay, [input.final_modifier, 4096]) : relay;
  probe.priorityEvent = (event, a, b, c, relay) => probe.modify(relay, [input.weather_modifier, 4096]);
  const source = {status: input.burn ? 'brn' : '', hasAbility: () => false, hasType: () => true, getTypes: () => ['Normal']};
  const target = {getMoveHitData: () => ({crit: input.critical, bypassProtect: input.bypass_protect}), runEffectiveness: () => input.effectiveness};
  const move = {id: 'tackle', type: 'Normal', category: 'Physical', spreadHit: input.spread,
    multihitType: input.parental_bond_second_hit ? 'parentalbond' : '', hit: 2};
  const tr = probe.trunc;
  const base = tr(tr(tr(tr(2 * input.level / 5 + 2) * input.power * input.attack) / input.defense) / 50);
  const damage = probe.actions.modifyDamage(base, source, target, move, true);
  damageKernel.push({seed, input, damage, final_seed: probe.prng.getSeed()});
}
probe.runEvent = originalRunEvent; probe.priorityEvent = originalPriorityEvent;
probe.destroy();
const files = {
  'dex.json': {schema: 'pa3-dex-v1', rules, tables}, 'scope.json': scope,
  'dynamic-closure.json': computeDynamicClosure(dex, scope),
  'callbacks.json': callbacks, 'training-teams.json': trainingTeams,
  'reference-fixtures.json': {oracle_commit: pin, rng, stats: statFixtures, pp, targeting, modifiers, recoil_rounding: recoilRounding, action_speed: actionSpeed, ordering, left_to_right_ordering: leftToRightOrdering, health, damage_kernel: damageKernel, initialization},
};
const manifest = {schema: 'pa3-engine-assets-v1', oracle_commit: pin, format, files: {}};
for (const [name, value] of Object.entries(files)) {
  const bytes = JSON.stringify(value) + '\n'; fs.writeFileSync(path.join(output, name), bytes);
  manifest.files[name] = {sha256: hash(bytes), bytes: Buffer.byteLength(bytes)};
}
// Complete-battle fixtures use legal synthetic teams, independently of the training pool.
execFileSync(process.execPath, [path.join(root, 'engine/tests/generate_turn_fixtures.mjs')], {cwd: root});
// Additional corpus generators (generate_more_*.mjs) each write one JSON file
// next to turn-fixtures.json; their fixtures are merged into the same corpus so
// every differential test sees one authoritative boundary list.
const extraGenerators = fs.readdirSync(path.join(root, 'engine/tests'))
  .filter(name => /^generate_more_.+\.mjs$/.test(name)).sort();
if (extraGenerators.length) {
  const merged = JSON.parse(fs.readFileSync(path.join(output, 'turn-fixtures.json'), 'utf8'));
  const names = new Set(merged.fixtures.map(f => f.name));
  // Reference-generated fixtures with a tracked native divergence stay out of
  // the merged corpus (see engine/data/known-mismatches.json). This is an
  // explicit, checked ledger rather than a silent skip: every open entry must
  // still be produced by a generator, and removing the entry merges the
  // fixture back on the next export.
  const ledger = JSON.parse(fs.readFileSync(path.join(output, 'known-mismatches.json'), 'utf8'));
  const open = new Map(ledger.mismatches.filter(m => m.status === 'open').map(m => [m.name, m]));
  const generated = new Set();
  for (const name of extraGenerators) {
    const artifact = path.join(output, name.replace(/^generate_/, '').replace(/\.mjs$/, '.json'));
    execFileSync(process.execPath, [path.join(root, 'engine/tests', name)], {cwd: root});
    const extra = JSON.parse(fs.readFileSync(artifact, 'utf8'));
    assert.equal(extra.oracle_commit, pin, `${name} oracle pin`);
    assert.equal(extra.format, format, `${name} format`);
    for (const fixture of extra.fixtures) {
      generated.add(fixture.name);
      if (open.has(fixture.name)) continue;
      assert(!names.has(fixture.name), `duplicate fixture name ${fixture.name}`);
      names.add(fixture.name);
      merged.fixtures.push(fixture);
    }
    console.log(`${name}: merged ${extra.fixtures.length - extra.fixtures.filter(f => open.has(f.name)).length} fixtures`);
  }
  for (const name of open.keys()) {
    assert(generated.has(name), `known-mismatches entry ${name} has no generated fixture (stale ledger)`);
    assert(!names.has(name), `known-mismatches entry ${name} is still merged into the corpus`);
  }
  fs.writeFileSync(path.join(output, 'turn-fixtures.json'), JSON.stringify(merged) + '\n');
}
const turnBytes = fs.readFileSync(path.join(output, 'turn-fixtures.json'));
manifest.files['turn-fixtures.json'] = {sha256: hash(turnBytes), bytes: turnBytes.length};
fs.writeFileSync(path.join(output, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
fs.copyFileSync(path.join(ref, 'LICENSE'), path.join(output, 'SHOWDOWN-LICENSE'));
console.log(JSON.stringify({starting_species: starting.length, battle_forms: battleForms.length, mega_forms: scope.mega_forms.length,
  allowed_moves: allowedMoves.length, items: legalItems.length, unresolved, callbacks: callbacks.length, training_teams: trainingTeams.length}));
