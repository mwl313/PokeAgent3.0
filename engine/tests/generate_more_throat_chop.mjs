// Development-only interaction corpus for Throat Chop's sound lock.
//
// The generic move corpus records the volatile but its filler teams carry no
// sound moves, so the request-level `onDisableMove` behaviour is never
// exercised. This generator scripts a battle where the target holds a sound
// move for the whole lock, requiring the reference request to mark it disabled
// while the two-turn volatile is active and to clear the flag afterwards.
import fs from 'node:fs';
import {createRequire} from 'node:module';
import {ReferenceSession, verifyReference, FORMAT, ORACLE_COMMIT} from '../reference.mjs';
const require = createRequire(import.meta.url);
const {TeamValidator, toID} = require('../../vendor/pokemon-showdown/dist/sim');
const validator = new TeamValidator(FORMAT);
const dex = validator.dex;
const data = JSON.parse(fs.readFileSync(new URL('../data/dex.json', import.meta.url), 'utf8'));
const ids = Object.fromEntries(Object.entries(data.tables).map(([k, rows]) => [k,
  Object.fromEntries(rows.map(r => [r.id, r.numeric_id]))]));
const stats = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
verifyReference();

const nameSets = team => team.map((set, i) => ({...set, name: `s${i}`}));
const setOf = (species, ability, moves, item) =>
  ({name: species, species, ability, item, nature: 'Serious', level: 50, gender: 'M', moves,
    evs: {hp: 24, atk: 8, def: 8, spa: 8, spd: 8, spe: 4}});
const roster = p => Number(p.name.slice(-1));

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
    return {present: !p.fainted, requires_replacement: forced, can_mega: Boolean(info?.canMegaEvo),
      trapped: Boolean(info?.trapped), maybe_trapped: Boolean(info?.maybeTrapped),
      moves: p.moveSlots.map(m => ({id: ids.moves[m.id], pp: m.pp, disabled: Boolean(m.disabled), target: m.target}))};
  });
  const bench = side.pokemon.map((p, i) => [p, i])
    .filter(([p]) => !p.fainted && !side.active.includes(p)).map(([p]) => roster(p));
  return {kind, slots, bench, preview: []};
};

const compact = session => ({turn: session.battle.turn, rng_seed: session.battle.prng.getSeed(),
  climate: {raw: session.battle.field.weather, effective: session.battle.field.effectiveWeather(), suppressed: session.battle.field.suppressingWeather()},
  field: [...(session.battle.field.weather ? [[ids.conditions[session.battle.field.weather], session.battle.field.weatherState.duration, session.battle.field.weatherState.source.side.n]] : []),
    ...(session.battle.field.terrain ? [[ids.conditions[session.battle.field.terrain], session.battle.field.terrainState.duration, session.battle.field.terrainState.source.side.n]] : []),
    ...Object.entries(session.battle.field.pseudoWeather).map(([id, effect]) => [ids.conditions[id], effect.duration, effect.source.side.n])].sort((a, b) => a[0] - b[0]),
  terminated: session.battle.ended, winner: session.battle.ended ? session.battle.winner || null : null,
  sides: session.battle.sides.map(s => ({request: session.battle.ended ? 'Finished' : s.activeRequest?.wait || s.isChoiceDone() ? 'Wait' : s.requestState === 'teampreview' ? 'Preview' : s.requestState === 'switch' ? 'Replacement' : 'Normal',
    conditions: Object.entries(s.sideConditions).map(([id, state]) => [ids.conditions[id], state.duration ?? state.layers ?? 0]).sort((a, b) => a[0] - b[0]),
    pokemon: s.pokemon.map(p => ({roster: roster(p), species: ids.species[p.species.id], hp: p.hp, max_hp: p.maxhp, fainted: p.fainted,
      active_slot: s.active.indexOf(p) >= 0 ? s.active.indexOf(p) : null, ability_ending: Boolean(p.abilityState.ending),
      cached_speed: p.speed ?? null, status: ids.conditions[p.status] ?? 0, boosts: Object.values(p.boosts),
      stats: [p.maxhp, ...Object.values(p.storedStats)], ability: ids.abilities[p.ability], item: ids.items[p.item] ?? 0,
      types: p.types.map(t => ids.types[toID(t)] ?? 0), previous_item: ids.items[p.lastItem] ?? 0, can_mega: Boolean(p.canMegaEvo),
      pp: p.moveSlots.map(m => m.pp), volatiles: Object.keys(p.volatiles).sort()})),
    request_detail: requestDetail(session, s)}))});

const select = (kind, own_slot, destination = 255) =>
  ({kind, own_slot, move_slot: 255, target_location: 0, switch_destination: destination, resource: 'None'});
const move = (slot, moveSlot, target = 0) =>
  ({kind: 'Move', own_slot: slot, move_slot: moveSlot, target_location: target, switch_destination: 255, resource: 'None'});

function choose(session, sideIndex, plan) {
  const b = session.battle, side = b.sides[sideIndex];
  if (side.requestState === 'teampreview') {
    return {actions: [0, 1, 4, 5].map((r, i) => select('Pick', i, r)), command: 'team 1256'};
  }
  const bench = side.pokemon.filter(p => !p.fainted && !side.active.includes(p));
  const actions = [], commands = [];
  for (let slot = 0; slot < 2; slot++) {
    const p = side.active[slot];
    if (side.requestState === 'switch') {
      if (!side.activeRequest?.forceSwitch?.[slot]) { commands.push('pass'); continue; }
      const reserve = bench.shift();
      if (reserve) { actions.push(select('Switch', slot, roster(reserve))); commands.push(`switch ${side.pokemon.indexOf(reserve) + 1}`); }
      else { commands.push('pass'); }
      continue;
    }
    if (p.fainted) { actions.push(select('Pass', slot)); commands.push('pass'); continue; }
    const wanted = sideIndex === 0 && slot === 0 ? plan.moveSlot : plan.attackSlot;
    const slotIndex = p.moveSlots.findIndex(m => m.id === wanted && !m.disabled && m.pp > 0);
    const choice = slotIndex >= 0 ? slotIndex : p.moveSlots.findIndex(m => !m.disabled && m.pp > 0);
    const chosen = p.moveSlots[choice];
    const candidates = sideIndex === 0 ? [2, 1, -1, -2, 0] : [-2, -1, 1, 2, 0];
    const location = chosen && b.actions.targetTypeChoices(chosen.target)
      ? candidates.find(loc => b.validTargetLoc(loc, p, chosen.target)) ?? 0 : 0;
    actions.push(move(slot, choice, location));
    commands.push(`move ${choice + 1}${location ? ` ${location}` : ''}`);
  }
  return {actions, command: commands.join(', ')};
}

const holder = setOf('Incineroar', 'Intimidate', ['Throat Chop', 'Protect'], '');
const target = setOf('Sylveon', 'Pixilate', ['Hyper Voice', 'Protect'], '');
const fillers = [
  ['Goodra-Hisui', 'Shell Armor', ['Dragon Pulse', 'Protect']],
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']],
  ['Metagross', 'Clear Body', ['Iron Head', 'Protect']],
  ['Milotic', 'Competitive', ['Surf', 'Protect']],
  ['Scolipede', 'Swarm', ['X-Scissor', 'Protect']],
  ['Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']],
  ['Starmie', 'Natural Cure', ['Ice Beam', 'Protect']],
].map(([species, ability, moves]) => setOf(species, ability, moves, ''));
const teamA = nameSets([holder, ...fillers.slice(0, 5)]);
const teamB = nameSets([target, ...fillers.slice(0, 5)]);
const problems = validator.validateTeam(teamA) || validator.validateTeam(teamB);
if (problems) throw new Error(`invalid fixture teams: ${problems.join('; ')}`);

const fixtures = [];
const skipped = [];
let recorded = null;
let lastReason = null;
for (let trial = 0; trial < 64; trial++) {
  const seed = [2026, 10, 7, 3950 + trial];
  const session = new ReferenceSession({teams: [teamA, teamB], seed});
  const plan = {moveSlot: ids.moves.throatchop, attackSlot: ids.moves.hypervoice};
  const fixture = {name: `move_throatchop_lock_${seed[3]}`, seed,
    teams: [teamA, teamB].map((team, side) => ({id: `throat-${side}`, members: team.map(s => ({
      species: ids.species[toID(s.species)], ability: ids.abilities[toID(s.ability)], item: ids.items[toID(s.item)] ?? 0,
      nature: ids.natures[toID(s.nature)], gender: s.gender || '', level: 50, moves: s.moves.map(x => ids.moves[toID(x)]),
      points: stats.map(k => s.evs[k]), ivs: stats.map(k => s.ivs[k])}))})),
    initial: compact(session), steps: []};
  let failure = null;
  while (!session.battle.ended && fixture.steps.length < 200) {
    for (let side = 0; side < 2; side++) {
      if (session.battle.ended) break;
      const s = session.battle.sides[side];
      if (s.activeRequest?.wait || s.isChoiceDone()) continue;
      const choice = choose(session, side, plan);
      const result = session.choose(side ? 'p2' : 'p1', choice.command);
      if (!result.accepted) { failure = JSON.stringify({side, choice, messages: result.messages}); break; }
      fixture.steps.push({side: side ? 'P2' : 'P1', ...choice, expected: compact(session)});
    }
    if (failure) break;
  }
  if (failure) { session.destroy(); lastReason = failure; continue; }
  if (!session.battle.ended) { session.destroy(); lastReason = 'did not complete'; continue; }
  // Require the reference lock to appear and to have disabled the sound move.
  const volatileSeen = fixture.steps.some(step => step.expected.sides.some(side =>
    side.pokemon.some(p => p.volatiles.includes('throatchop'))));
  const disabledSeen = fixture.steps.some(step => step.expected.sides.some(side =>
    (side.request_detail?.slots ?? []).some(slot =>
      slot.moves.some(m => m.id === ids.moves.hypervoice && m.disabled))));
  const cleared = fixture.steps.some((step, i) => i > 2 && step.expected.sides.every(side =>
    (side.request_detail?.slots ?? []).every(slot =>
      slot.moves.every(m => m.id !== ids.moves.hypervoice || !m.disabled))));
  if (!volatileSeen || !disabledSeen) {
    session.destroy();
    lastReason = `lock not observed (volatile=${volatileSeen} disabled=${disabledSeen})`;
    continue;
  }
  fixture.coverage = {move: 'throatchop', lock: true, disabled_seen: disabledSeen, cleared_seen: cleared};
  recorded = fixture;
  session.destroy();
  break;
}
if (recorded) fixtures.push(recorded);
else skipped.push({name: 'throatchop_lock', reason: lastReason ?? 'no seed produced the required behavior'});

fs.writeFileSync(new URL('../data/more_throat_chop.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify({fixtures: fixtures.length, skipped: skipped.length}));
if (skipped.length) console.log(JSON.stringify(skipped));
