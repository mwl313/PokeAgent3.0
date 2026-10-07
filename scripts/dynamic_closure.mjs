// Machine-generated dynamic-reachability closure for the pinned M-C reference.
//
// Some in-scope mechanics do not carry their gameplay on the team sheet: they
// call, copy, transform into, inherit or transfer another effect (Copycat,
// Sleep Talk, Instruct, Transform, Trace, Imposter, Skill Swap, Thief, ...).
// Coverage accounting for those callers is only honest if it also covers the
// universe of effects they can reach. This module derives that closure from
// the pinned Showdown sources instead of a hand-maintained list, so it cannot
// drift as the port proceeds.
//
// Cold tool: never imported by the Rust/Python battle loop. `export_engine_data.mjs`
// embeds the result into `dynamic-closure.json`; running this file directly
// regenerates just that artifact plus its manifest entry.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const requireModule = createRequire(import.meta.url);
const referenceRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'vendor/pokemon-showdown');
const {toID} = requireModule(path.join(referenceRoot, 'dist/sim'));

export const ORACLE_PIN = '14546894d86f9589ac11130c510bbe73b6968665';
export const ORACLE_FORMAT = 'gen9championsvgc2026regmc';

/// Handler-source token rules. A handler that matches any token inherits the
/// listed reachable universes. Rules stay deliberately conservative (an
/// over-inclusive closure can only make readiness stricter, never looser).
export const CLOSURE_RULES = [
  {
    id: 'call-move',
    tokens: ['useMove(', 'resolveAction('],
    reach: ['moves'],
    note: 'executes another move (Copycat, Sleep Talk, Instruct, Dancer-style abilities)',
  },
  {
    id: 'transform',
    tokens: ['transformInto('],
    reach: ['species_forms', 'moves', 'abilities'],
    note: 'copies species, types, stats, moves and ability from another Pokémon',
  },
  {
    id: 'copy-ability',
    // Copying requires a read of the source ability plus a write, or the
    // reference `skillSwap` helper (Skill Swap, Trace, Receiver, Wandering
    // Spirit, Mummy's suppression guard).
    anyTokens: [['setAbility(', 'getAbility('], ['skillSwap(', 'getAbility('], ['skillSwap(']],
    reach: ['abilities'],
    note: 'copies or swaps an ability, so every legal ability must be executable',
  },
  {
    id: 'transfer-item',
    tokens: ['takeItem(', 'setItem('],
    reach: ['items'],
    note: 'moves another Pokémon\'s item (Thief, Covet, Fling, Recycle, Magician, Pickpocket, Symbiosis)',
  },
  {
    id: 'suppress-ability',
    anyTokens: [['singleEvent("End"', 'getAbility(']],
    reach: ['abilities'],
    note: 'suppresses an arbitrary ability and must run its End event (Gastroacid)',
  },
  {
    id: 'end-held-item',
    anyTokens: [['singleEvent("End"', 'getItem(']],
    reach: ['items'],
    note: 'ends the held item\'s effects and so needs every legal item\'s End semantics (Klutz)',
  },
];

function scanEntity(raw, kind, id, out) {
  if (!raw || typeof raw !== 'object') return;
  const handlers = [];
  const walk = (value, prefix) => {
    if (typeof value === 'function') {
      handlers.push({name: prefix, source: value.toString()});
      return;
    }
    if (value && typeof value === 'object') {
      for (const key of Object.keys(value)) walk(value[key], `${prefix}.${key}`);
    }
  };
  walk(raw, kind);
  if (!handlers.length) return;
  // Rule evaluation is entity-level: a copy usually reads the source value in
  // one handler (`onTryHit`) and writes the copy in another (`onHit`).
  const all = handlers.map(handler => handler.source).join('\n');
  const reach = new Map();
  for (const rule of CLOSURE_RULES) {
    const hit = rule.anyTokens
      ? rule.anyTokens.some(group => group.every(token => all.includes(token)))
      : rule.tokens.some(token => all.includes(token));
    if (!hit) continue;
    const matched = handlers
      .filter(handler => (rule.anyTokens
        ? rule.anyTokens.some(group => group.every(token => handler.source.includes(token)))
        : rule.tokens.some(token => handler.source.includes(token))))
      .map(handler => handler.name);
    reach.set(rule.id, {
      rule: rule.id,
      note: rule.note,
      handlers: matched.length ? matched.sort() : handlers.map(handler => handler.name).sort(),
      reach: rule.reach,
    });
  }
  if (!reach.size) return;
  out.push({
    kind,
    id,
    rules: [...reach.values()].map(entry => ({
      rule: entry.rule,
      handlers: [...new Set(entry.handlers)].sort(),
      reach: entry.reach,
      note: entry.note,
    })),
    reach: [...new Set([...reach.values()].flatMap(entry => entry.reach))].sort(),
  });
}

/// `scope` is `engine/data/scope.json`: the same legal sets the readiness gate
/// uses, so the closure and the coverage counters can never disagree about
/// what "in scope" means.
export function computeDynamicClosure(dex, scope) {
  const legalAbilities = new Set();
  for (const row of scope.starting_species ?? []) {
    for (const ability of row.abilities ?? []) legalAbilities.add(ability);
  }
  for (const form of scope.format_permitted_battle_forms ?? []) {
    const species = dex.species.get(form);
    for (const ability of Object.values(species.abilities ?? {})) legalAbilities.add(toID(ability));
  }
  const entities = [];
  for (const id of scope.allowed_moves ?? []) {
    scanEntity(dex.data.Moves?.[id], 'move', id, entities);
  }
  for (const id of [...legalAbilities].sort()) {
    scanEntity(dex.data.Abilities?.[id], 'ability', id, entities);
  }
  for (const id of scope.allowed_items ?? []) {
    scanEntity(dex.data.Items?.[id], 'item', id, entities);
  }
  entities.sort((a, b) => (a.kind < b.kind ? -1 : a.kind > b.kind ? 1 : a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  return {
    schema: 'pa3-dynamic-closure-v1',
    oracle_commit: ORACLE_PIN,
    format: ORACLE_FORMAT,
    generated_by: 'scripts/dynamic_closure.mjs',
    rules: CLOSURE_RULES.map(({id, reach, note}) => ({id, reach, note})),
    universes: {
      moves: (scope.allowed_moves ?? []).length,
      abilities: legalAbilities.size,
      items: (scope.allowed_items ?? []).length,
      species_forms: (scope.starting_species ?? []).length + (scope.format_permitted_battle_forms ?? []).length,
    },
    entries: entities,
  };
}

function loadReference() {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  const ref = path.join(root, 'vendor/pokemon-showdown');
  const require = createRequire(import.meta.url);
  const {TeamValidator} = require(path.join(ref, 'dist/sim'));
  const head = execFileSync('git', ['rev-parse', 'HEAD'], {cwd: ref, encoding: 'utf8'}).trim();
  if (head !== ORACLE_PIN) throw new Error(`reference pin changed: ${head}`);
  const dirty = execFileSync('git', ['diff', '--name-only', 'HEAD'], {cwd: ref, encoding: 'utf8'}).trim();
  if (dirty) throw new Error(`reference worktree dirty: ${dirty}`);
  return {root, dex: new TeamValidator(ORACLE_FORMAT).dex};
}

function main() {
  const {root, dex} = loadReference();
  const output = path.join(root, 'engine/data');
  const scope = JSON.parse(fs.readFileSync(path.join(output, 'scope.json'), 'utf8'));
  const closure = computeDynamicClosure(dex, scope);
  const bytes = JSON.stringify(closure) + '\n';
  fs.writeFileSync(path.join(output, 'dynamic-closure.json'), bytes);
  const manifestPath = path.join(output, 'manifest.json');
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  manifest.files['dynamic-closure.json'] = {
    sha256: crypto.createHash('sha256').update(bytes).digest('hex'),
    bytes: Buffer.byteLength(bytes),
  };
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
  const byKind = {};
  for (const entry of closure.entries) byKind[entry.kind] = (byKind[entry.kind] ?? 0) + 1;
  console.log(JSON.stringify({entries: closure.entries.length, by_kind: byKind, universes: closure.universes}));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main();
}
