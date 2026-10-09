// Deterministic JSON-Schema-subset checker for the accepted team records.
// It implements the keywords used by the supplied pa3-team-v1 schema and the
// derived pa3-team-v2 schema (type/const/enum/required/properties/
// additionalProperties/items/minItems/maxItems/minLength/pattern/minimum/
// allOf/if-then-else/$ref/format). It is a cold development instrument, never
// part of the battle path.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {ROOT} from './pa3-common.mjs';

export const SCHEMAS = {
  'pa3-team-v1': 'docs/spec/fullspec-1.1-minidc-20261006/PA3_TEAM_RECORD_SCHEMA.json',
  'pa3-team-v2': 'data/schemas/team-record-v2.schema.json',
};

const URI = /^[a-z][a-z0-9+.-]*:/i;
const DATE_TIME = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$/;

export function validate(schema, data, root = schema, path = '$') {
  const errors = [];
  const push = message => errors.push(`${path}: ${message}`);
  if (!schema || typeof schema !== 'object') return errors;
  if (schema.$ref) {
    const target = schema.$ref.split('/').slice(1).reduce((node, key) => node?.[key], root);
    if (!target) throw new Error(`Unresolvable $ref ${schema.$ref}`);
    return validate(target, data, root, path);
  }
  if (schema.const !== undefined && JSON.stringify(data) !== JSON.stringify(schema.const)) {
    push(`expected const ${JSON.stringify(schema.const)}, got ${JSON.stringify(data)}`);
  }
  if (schema.enum && !schema.enum.some(value => JSON.stringify(value) === JSON.stringify(data))) {
    push(`not in enum ${JSON.stringify(schema.enum)}`);
  }
  if (schema.type) {
    const types = Array.isArray(schema.type) ? schema.type : [schema.type];
    const actual = data === null ? 'null' : Array.isArray(data) ? 'array' : typeof data;
    const integer = typeof data === 'number' && Number.isInteger(data);
    const accepted = types.includes(actual) || (integer && types.includes('integer'));
    if (!accepted) push(`expected type ${types.join('|')}, got ${actual}`);
  }
  if (typeof data === 'string') {
    if (schema.minLength !== undefined && data.length < schema.minLength) push(`shorter than minLength ${schema.minLength}`);
    if (schema.pattern && !new RegExp(schema.pattern).test(data)) push(`does not match ${schema.pattern}`);
    if (schema.format === 'uri' && !URI.test(data)) push('not a URI');
    if (schema.format === 'date-time' && !DATE_TIME.test(data)) push('not an RFC3339 date-time');
  }
  if (typeof data === 'number' && schema.minimum !== undefined && data < schema.minimum) push(`below minimum ${schema.minimum}`);
  if (Array.isArray(data)) {
    if (schema.minItems !== undefined && data.length < schema.minItems) push(`fewer than ${schema.minItems} items`);
    if (schema.maxItems !== undefined && data.length > schema.maxItems) push(`more than ${schema.maxItems} items`);
    if (schema.items) data.forEach((item, index) => errors.push(...validate(schema.items, item, root, `${path}[${index}]`)));
  }
  if (data && typeof data === 'object' && !Array.isArray(data)) {
    for (const key of schema.required || []) if (!(key in data)) push(`missing required ${key}`);
    if (schema.properties) {
      for (const [key, value] of Object.entries(data)) {
        if (schema.properties[key]) errors.push(...validate(schema.properties[key], value, root, `${path}.${key}`));
        else if (schema.additionalProperties === false) push(`unexpected property ${key}`);
      }
    }
  }
  for (const branch of schema.allOf || []) errors.push(...validate(branch, data, root, path));
  if (schema.if) {
    const matched = validate(schema.if, data, root, path).length === 0;
    if (matched && schema.then) errors.push(...validate(schema.then, data, root, path));
    if (!matched && schema.else) errors.push(...validate(schema.else, data, root, path));
  }
  return errors;
}

export function validateDataset(dir, schemaName, {schemaPath} = {}) {
  const schema = JSON.parse(fs.readFileSync(schemaPath || path.join(ROOT, SCHEMAS[schemaName]), 'utf8'));
  const errors = [];
  for (const name of ['all.jsonl', 'train.jsonl', 'dev.jsonl', 'final.jsonl', 'quarantine.jsonl', 'source-index.jsonl']) {
    const file = path.join(dir, name);
    if (!fs.existsSync(file)) continue;
    const lines = fs.readFileSync(file, 'utf8').split('\n').filter(Boolean);
    // The accepted-record schema applies to the accepted splits only.
    if (!['all.jsonl', 'train.jsonl', 'dev.jsonl', 'final.jsonl'].includes(name)) continue;
    lines.forEach((line, index) => {
      const row = JSON.parse(line);
      errors.push(...validate(schema, row, schema, `${name}:${index + 1}`));
    });
  }
  return errors;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const datasetId = process.argv[2] || 'mb-mc-v3-userteam-all-train';
  const schemaName = process.argv[3] || (datasetId.includes('v2') ? 'pa3-team-v1' : 'pa3-team-v2');
  const dir = path.join(ROOT, 'data/teams', datasetId);
  const errors = validateDataset(dir, schemaName);
  if (errors.length) {
    console.error(errors.slice(0, 20).join('\n'));
    console.error(`${errors.length} schema violations in ${datasetId}`);
    process.exit(1);
  }
  console.log(`${datasetId}: all accepted records satisfy ${SCHEMAS[schemaName]}`);
}
