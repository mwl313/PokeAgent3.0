// Shared cold-path constants and canonical hashing helpers.
// These values are imported by the development-only preparation scripts; no
// battle-path code depends on this module.
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const ORACLE_COMMIT = '14546894d86f9589ac11130c510bbe73b6968665';
export const FORMAT = 'gen9championsvgc2026regmc';
export const SEED = 20261006;

export function stable(value) {
  if (Array.isArray(value)) return '[' + value.map(stable).join(',') + ']';
  if (value && typeof value === 'object') return '{' + Object.keys(value).sort().filter(k => value[k] !== undefined)
    .map(k => JSON.stringify(k) + ':' + stable(value[k])).join(',') + '}';
  return JSON.stringify(value);
}

export const hash = value => crypto.createHash('sha256').update(value).digest('hex');
