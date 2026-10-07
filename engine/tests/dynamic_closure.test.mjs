// Drift gate for engine/data/dynamic-closure.json.
//
// The closure is generated from the pinned Showdown sources, but it is also a
// committed artifact that readiness and fixture-coverage gates read. This test
// recomputes it and requires byte-level agreement, so hand edits, a stale
// oracle pin, or a forgotten regeneration fail loudly.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import {computeDynamicClosure, ORACLE_FORMAT, ORACLE_PIN} from '../../scripts/dynamic_closure.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const require = createRequire(import.meta.url);
const {TeamValidator} = require(path.join(root, 'vendor/pokemon-showdown/dist/sim'));
const dex = new TeamValidator(ORACLE_FORMAT).dex;
const scope = JSON.parse(fs.readFileSync(path.join(root, 'engine/data/scope.json'), 'utf8'));
const stored = JSON.parse(
  fs.readFileSync(path.join(root, 'engine/data/dynamic-closure.json'), 'utf8'),
);

assert.equal(stored.oracle_commit, ORACLE_PIN, 'closure oracle pin');
assert.equal(stored.format, ORACLE_FORMAT, 'closure format pin');
assert.deepStrictEqual(
  computeDynamicClosure(dex, scope),
  stored,
  'dynamic-closure.json is stale; run node scripts/dynamic_closure.mjs',
);
console.log(`dynamic closure ok: ${stored.entries.length} callers`);
