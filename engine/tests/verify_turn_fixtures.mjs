// Independent provenance check for engine/data/turn-fixtures.json.
//
// Regenerating the corpus requires the full fixture generator. This script
// instead replays every committed fixture step against a freshly booted pinned
// Showdown reference and deep-compares the recomputed decision-boundary state
// with the stored `expected` values. It proves the Rust differential corpus is
// genuine reference output rather than hand-written or stale expectations.
//
// Development-only: never call this from Rust or the training loop.
// Usage: node engine/tests/verify_turn_fixtures.mjs
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {ReferenceSession, verifyReference} from '../reference.mjs';

const root = new URL('../../', import.meta.url);
const data = JSON.parse(fs.readFileSync(new URL('engine/data/dex.json', root), 'utf8'));
const byNumeric = Object.fromEntries(
  Object.entries(data.tables).map(([k, rows]) => [k, Object.fromEntries(rows.map(r => [r.numeric_id, r]))]),
);
const ids = Object.fromEntries(
  Object.entries(data.tables).map(([k, rows]) => [k, Object.fromEntries(rows.map(r => [r.id, r.numeric_id]))]),
);
const name = (kind, id) => ((byNumeric[kind][id] || {}).data || {}).name || null;
const stats = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const toID = s => (s || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
const roster = p => Number(p.name.slice(-1));

// Mirrors compact() in generate_turn_fixtures.mjs; keep the two in sync.
const requestDetail = (session, side) => {
  if (session.battle.ended) return null;
  if (side.requestState === 'teampreview') {
    return {kind: side.isChoiceDone() ? 'Wait' : 'Preview', slots: [], bench: [], preview: [0, 1, 2, 3, 4, 5]};
  }
  const req = side.activeRequest ?? {};
  const kind = req.wait || side.isChoiceDone() ? 'Wait' : side.requestState === 'switch' ? 'Replacement' : 'Normal';
  const slots = [0, 1].map(slot => {
    const p = side.active[slot];
    const info = req.active?.[slot];
    const forced = Boolean(req.forceSwitch?.[slot]);
    if (!p) return {present: false, requires_replacement: forced, can_mega: false, moves: []};
    // World move list and raw disable flag: the served choice legality.
    return {present: !p.fainted, requires_replacement: forced, can_mega: Boolean(info?.canMegaEvo),
      moves: p.moveSlots.map(m => ({id: ids.moves[m.id], pp: m.pp,
        disabled: Boolean(m.disabled), target: m.target}))};
  });
  const bench = side.pokemon.map((p, i) => [p, i])
    .filter(([p]) => !p.fainted && !side.active.includes(p)).map(([p]) => roster(p));
  return {kind, slots, bench, preview: []};
};
const compact = session => ({turn: session.battle.turn, rng_seed: session.battle.prng.getSeed(),
  climate: {raw: session.battle.field.weather, effective: session.battle.field.effectiveWeather(), suppressed: session.battle.field.suppressingWeather()},
  field: [...(session.battle.field.weather ? [[ids.conditions[session.battle.field.weather], session.battle.field.weatherState.duration, session.battle.field.weatherState.source.side.n]] : []), ...(session.battle.field.terrain ? [[ids.conditions[session.battle.field.terrain], session.battle.field.terrainState.duration, session.battle.field.terrainState.source.side.n]] : []), ...Object.entries(session.battle.field.pseudoWeather).map(([id, effect]) => [ids.conditions[id], effect.duration, effect.source.side.n])].sort((a, b) => a[0] - b[0]),
  terminated: session.battle.ended, winner: session.battle.ended ? session.battle.winner || null : null,
  sides: session.battle.sides.map(s => ({request: session.battle.ended ? 'Finished' : s.activeRequest?.wait || s.isChoiceDone() ? 'Wait' : s.requestState === 'teampreview' ? 'Preview' : s.requestState === 'switch' ? 'Replacement' : 'Normal',
    conditions: Object.entries(s.sideConditions).map(([id, state]) => [ids.conditions[id], state.duration]).sort((a,b) => a[0]-b[0]),
    pokemon: s.pokemon.map(p => ({roster: roster(p), species: ids.species[p.species.id], hp: p.hp,
      max_hp: p.maxhp, fainted: p.fainted, active_slot: s.active.indexOf(p) >= 0 ? s.active.indexOf(p) : null,
      ability_ending: Boolean(p.abilityState.ending), cached_speed: p.speed ?? null, status: ids.conditions[p.status] ?? 0, boosts: Object.values(p.boosts), stats: [p.maxhp, ...Object.values(p.storedStats)], ability: ids.abilities[p.ability], item: ids.items[p.item] ?? 0, types: p.types.map(t => ids.types[toID(t)]),
      previous_item: ids.items[p.lastItem] ?? 0, can_mega: Boolean(p.canMegaEvo), pp: p.moveSlots.map(m => m.pp), volatiles: Object.keys(p.volatiles).sort()})),
    request_detail: requestDetail(session, s)}))});

function sets(team, side) {
  return team.members.map((m, i) => ({name: `s${side}m${i}`, species: name('species', m.species),
    ability: name('abilities', m.ability), item: name('items', m.item) || '', nature: name('natures', m.nature),
    level: 50, gender: m.gender || '', moves: m.moves.map(x => name('moves', x)),
    evs: Object.fromEntries(stats.map((k, j) => [k, m.points[j]])),
    ivs: Object.fromEntries(stats.map((k, j) => [k, m.ivs[j]]))}));
}

verifyReference();
const corpus = JSON.parse(fs.readFileSync(new URL('engine/data/turn-fixtures.json', root), 'utf8'));

// Report the first divergent JSON path instead of a 500-line deep diff, so a
// one-value regression names the side/roster/move it came from.
function firstDiff(actual, expected, path = '$') {
  if (Object.is(actual, expected)) return null;
  if (Array.isArray(actual) && Array.isArray(expected)) {
    if (actual.length !== expected.length) {
      return `${path}.length: actual ${actual.length} vs expected ${expected.length}`;
    }
    for (let i = 0; i < actual.length; i++) {
      const found = firstDiff(actual[i], expected[i], `${path}[${i}]`);
      if (found) return found;
    }
    return null;
  }
  if (actual && expected && typeof actual === 'object' && typeof expected === 'object') {
    const actualOnly = Object.keys(actual).filter(key => !(key in expected));
    const expectedOnly = Object.keys(expected).filter(key => !(key in actual));
    if (actualOnly.length || expectedOnly.length) {
      return `${path}: key mismatch actual-only=${JSON.stringify(actualOnly)} expected-only=${JSON.stringify(expectedOnly)}`;
    }
    const keys = new Set([...Object.keys(actual), ...Object.keys(expected)]);
    for (const key of keys) {
      const found = firstDiff(actual[key], expected[key], `${path}.${key}`);
      if (found) return found;
    }
    return null;
  }
  return `${path}: actual ${JSON.stringify(actual)} vs expected ${JSON.stringify(expected)}`;
}

function compare(label, actual, expected) {
  const diff = firstDiff(actual, expected);
  if (diff) throw new Error(`${label}: first divergence at ${diff}`);
}

let checkedSteps = 0;
let checkedFixtures = 0;
for (const fixture of corpus.fixtures) {
  const session = new ReferenceSession({
    teams: [sets(fixture.teams[0], 0), sets(fixture.teams[1], 1)], seed: fixture.seed,
  });
  try {
    compare(`${fixture.name}: initial`, compact(session), fixture.initial);
    for (const [stepIndex, step] of fixture.steps.entries()) {
      const result = session.choose(step.side === 'P1' ? 'p1' : 'p2', step.command);
      assert.ok(result.accepted, `${fixture.name}: rejected ${step.command}`);
      compare(
        `${fixture.name}: step ${stepIndex} (${step.command})`,
        compact(session),
        step.expected,
      );
      checkedSteps++;
    }
    checkedFixtures++;
  } finally {
    session.destroy();
  }
}
console.log(`verified ${checkedFixtures}/${corpus.fixtures.length} fixtures / ${checkedSteps} decision boundaries against the pinned reference`);
