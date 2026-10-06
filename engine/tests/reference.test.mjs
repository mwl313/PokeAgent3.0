import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {ReferenceSession, verifyReference} from '../reference.mjs';

const pool = fs.readFileSync(new URL('../../data/teams/mb-mc-v2-all-train/train.jsonl', import.meta.url), 'utf8').trim().split('\n').map(JSON.parse);
const teams = [0, 1].map(i => pool[i].members.map(m => m.showdown_set));

test('reference adapter preserves CTS, pending choices, seed, snapshot and trace', () => {
  verifyReference();
  const session = new ReferenceSession({teams, seed: [1, 2, 3, 4], record_rng: true});
  try {
    let s = session.debugState(); assert.equal(s.request_state, 'teampreview');
    assert.equal(s.sides[0].request.side.pokemon.length, 6);
    assert.ok(s.sides.flatMap(s => s.pokemon).every(p => p.can_tera === null));
    assert.ok(session.choose('p1', 'team 1234').accepted);
    const pending = session.snapshot();
    session.choose('p2', 'team 1234');
    const first = session.debugState(); assert.equal(first.request_state, 'move');
    const firstTrace = session.exportTrace();
    assert.ok(first.sides.every(s => s.pokemon.length === 4));
    session.restore(pending); session.choose('p2', 'team 1234');
    assert.deepEqual(session.debugState(), first);
    assert.deepEqual(session.exportTrace().actions, firstTrace.actions);
    assert.deepEqual(session.exportTrace().rng_draws, firstTrace.rng_draws);
    assert.ok(session.exportTrace().rng_draws.length > 0);
    assert.ok(!session.exportTrace().log.some(s => s.includes('|showteam|')));
    assert.throws(() => session.choose('p1', 'showteam'));
  } finally { session.destroy(); }
});
