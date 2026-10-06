// Development-only corpus for the native team validator.
//
// Every verdict in this corpus comes from the pinned Showdown TeamValidator
// itself; the Rust legality test asserts the native verdict and category agree.
// Never imported by the battle loop.
import fs from 'node:fs';
import {createRequire} from 'node:module';
import {verifyReference, FORMAT, ORACLE_COMMIT} from '../reference.mjs';
const require = createRequire(import.meta.url);
const {TeamValidator, toID} = require('../../vendor/pokemon-showdown/dist/sim');
const validator = new TeamValidator(FORMAT);
const data = JSON.parse(fs.readFileSync(new URL('../data/dex.json', import.meta.url), 'utf8'));
const numeric = Object.fromEntries(Object.entries(data.tables)
  .map(([kind, rows]) => [kind, Object.fromEntries(rows.map(r => [r.id, r.numeric_id]))]));
const stats = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
verifyReference();

// Categories are derived from the reference's own problem strings, so the
// corpus records the reference verdict rather than our expectation of it.
function category(problems) {
  const text = (problems || []).join(' ');
  if (!text) return 'legal';
  if (/Species Clause/.test(text)) return 'species_clause';
  if (/Item Clause/.test(text)) return 'item_clause';
  if (/banned by Flat Rules/.test(text)) return 'banned';
  if (/multiple copies of/.test(text)) return 'duplicate_move';
  if (/more than the limit of 4|does not exist in Gen 9.*move|can't learn/.test(text)) return 'illegal_move';
  if (/can't have [A-Z]/.test(text)) return 'illegal_ability';
  if (/item .* does not exist|does not exist in Gen 9\.$/.test(text)) return 'illegal_item';
  if (/Stat Points/.test(text)) return 'stat_points';
  if (/IVs/.test(text)) return 'ivs';
  if (/nicknames/.test(text)) return 'nicknames';
  if (/level/i.test(text)) return 'level';
  if (/does not exist|not obtainable|was not found/i.test(text)) return 'species';
  if (/Mega|forme|form/i.test(text)) return 'form_or_mega';
  if (/gender/i.test(text)) return 'gender';
  return 'other';
}

// A legal synthetic team with unambiguous members, used as the mutation base so
// crafted cases isolate exactly one rule.
const SYNTH = [
  ['Goodra-Hisui', 'Shell Armor', '', ['Dragon Pulse', 'Protect']],
  ['Torterra', 'Shell Armor', '', ['Seed Bomb', 'Protect']],
  ['Perrserker', 'Battle Armor', '', ['Iron Head', 'Protect']],
  ['Samurott', 'Shell Armor', '', ['Aqua Jet', 'Protect']],
  ['Hydreigon', 'Levitate', '', ['Dragon Pulse', 'Protect']],
  ['Falinks', 'Battle Armor', '', ['Smart Strike', 'Protect']],
];
const idOf = (kind, name) => numeric[kind][toID(name)] ?? 0;
const synthMember = ([species, ability, item, moves], extra = {}) => ({
  species: idOf('species', species),
  ability: idOf('abilities', ability),
  item: idOf('items', item || ''),
  nature: idOf('natures', extra.nature || 'Serious'),
  moves: moves.map(m => idOf('moves', m)),
  points: extra.points ?? [24, 8, 8, 8, 8, 4],
  ivs: extra.ivs ?? [31, 31, 31, 31, 31, 31],
  gender: extra.gender ?? '',
  level: extra.level ?? 50,
});
const synthTeam = (extra = {}) => SYNTH.map((row, i) => synthMember(row, extra[i] || {}));

const showdownMember = (member, index) => ({
  name: `n${index}`,
  species: data.tables.species.find(r => r.numeric_id === member.species)?.data.name ?? '',
  ability: data.tables.abilities.find(r => r.numeric_id === member.ability)?.data.name ?? '',
  item: member.item ? data.tables.items.find(r => r.numeric_id === member.item)?.data.name ?? '' : '',
  nature: data.tables.natures.find(r => r.numeric_id === member.nature)?.data.name ?? 'Serious',
  level: member.level,
  gender: member.gender,
  moves: member.moves.map(id => data.tables.moves.find(r => r.numeric_id === id)?.data.name ?? ''),
  evs: Object.fromEntries(stats.map((k, i) => [k, member.points[i]])),
  ivs: Object.fromEntries(stats.map((k, i) => [k, member.ivs[i]])),
});

const cases = [];
const record = (name, source, members) => {
  const showdown = members.map((m, i) => showdownMember(m, i));
  const problems = validator.validateTeam(showdown);
  cases.push({
    name,
    source,
    teams: [{id: name, members}],
    reference_legal: !problems,
    reference_problems: problems || [],
    reference_category: category(problems),
  });
};

// Every crafted case starts from the same legal synthetic team.
const mutate = (name, fn) => {
  const team = synthTeam();
  fn(team);
  record(name, 'crafted', team);
};

record('synthetic_legal', 'crafted', synthTeam());
mutate('duplicate_species', team => {
  team[1] = {...team[0]};
});
mutate('duplicate_item', team => {
  team[0].item = idOf('items', 'Leftovers');
  team[1].item = idOf('items', 'Leftovers');
});
mutate('over_budget_stat_points', team => {
  team[0].points = [32, 32, 32, 32, 32, 32];
});
mutate('single_stat_over_limit', team => {
  team[0].points = [33, 0, 0, 0, 0, 0];
});
mutate('zero_points_serious', team => {
  team[0].points = [0, 0, 0, 0, 0, 0];
  team[0].nature = idOf('natures', 'Serious');
});
mutate('zero_points_neutral_other', team => {
  team[0].points = [0, 0, 0, 0, 0, 0];
  team[0].nature = idOf('natures', 'Hardy');
});
mutate('level_100_adjusted', team => {
  team[0].level = 100;
});
mutate('level_1', team => {
  team[0].level = 1;
});
mutate('iv_not_maxed', team => {
  team[0].ivs = [31, 30, 31, 31, 31, 31];
});
mutate('unlearnable_move', team => {
  team[0].moves = [idOf('moves', 'Blue Flare'), idOf('moves', 'Protect')];
});
mutate('duplicate_move', team => {
  team[0].moves = [idOf('moves', 'Protect'), idOf('moves', 'Protect')];
});
mutate('five_moves', team => {
  team[0].moves = ['Dragon Pulse', 'Surf', 'Fire Blast', 'Thunderbolt', 'Protect'].map(m => idOf('moves', m));
});
mutate('illegal_ability', team => {
  team[0].ability = idOf('abilities', 'Wonder Guard');
});
mutate('mythical_species', team => {
  team[0] = synthMember(['Mew', 'Synchronize', '', ['Psychic', 'Ice Beam', 'Earthquake', 'Protect']]);
});
mutate('restricted_legendary', team => {
  team[0] = synthMember(['Mewtwo', 'Pressure', '', ['Psychic', 'Ice Beam', 'Earthquake', 'Protect']]);
});
mutate('illegal_item', team => {
  team[0].item = idOf('items', 'Rusted Sword');
});
mutate('mega_stone_on_wrong_species', team => {
  team[0] = synthMember(['Pikachu', 'Static', 'Charizardite Y', ['Thunderbolt', 'Quick Attack', 'Iron Tail', 'Protect']]);
});
mutate('mega_stone_on_base_form', team => {
  team[0] = synthMember(['Charizard', 'Blaze', 'Charizardite Y', ['Flamethrower', 'Air Slash', 'Dragon Pulse', 'Protect']]);
});
mutate('mega_form_submitted', team => {
  team[0] = synthMember(['Charizard-Mega-Y', 'Drought', 'Charizardite Y', ['Flamethrower', 'Air Slash', 'Dragon Pulse', 'Protect']]);
});
mutate('stat_points_sum_67', team => {
  team[0].points = [32, 25, 5, 5, 0, 0];
});

// A slice of the frozen pool as positive controls (all must be legal).
const pool = fs.readFileSync(new URL('../../data/teams/mb-mc-v2-all-train/train.jsonl', import.meta.url), 'utf8')
  .trim().split('\n').map(line => JSON.parse(line));
const poolMember = member => ({
  species: idOf('species', member.species),
  ability: idOf('abilities', member.ability),
  item: idOf('items', member.item || ''),
  nature: idOf('natures', member.nature),
  moves: member.moves.map(m => idOf('moves', m)),
  points: stats.map(k => member.allocation.engine_values[k]),
  ivs: stats.map(k => member.ivs[k]),
  gender: member.showdown_set.gender || '',
  level: member.level,
});
for (const [index, team] of pool.entries()) {
  if (index >= 61) break;
  record(`pool_${team.team_id}`, 'frozen_training_pool', team.members.map(poolMember));
}

fs.writeFileSync(new URL('../data/legality-cases.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, cases}) + '\n');
console.log(JSON.stringify({
  cases: cases.length,
  rejected: cases.filter(c => !c.reference_legal).length,
  categories: cases.reduce((acc, c) => ((acc[c.reference_category] = (acc[c.reference_category] || 0) + 1), acc), {}),
}));
