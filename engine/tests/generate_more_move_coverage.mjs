// Development-only generic move corpus. One complete legal battle per
// regulation move the native engine claims to execute, so every enabled move
// has a differential witness. Fixtures are merged into turn-fixtures.json by
// scripts/export_engine_data.mjs; this file is never used by training.
import fs from 'node:fs';
import {createRequire} from 'node:module';
import {ReferenceSession, verifyReference, FORMAT, ORACLE_COMMIT} from '../reference.mjs';
const require = createRequire(import.meta.url);
const {TeamValidator, toID} = require('../../vendor/pokemon-showdown/dist/sim');
const validator = new TeamValidator(FORMAT);
const dex = validator.dex;
const data = JSON.parse(fs.readFileSync(new URL('../data/dex.json', import.meta.url), 'utf8'));
const scope = JSON.parse(fs.readFileSync(new URL('../data/scope.json', import.meta.url), 'utf8'));
const ids = Object.fromEntries(Object.entries(data.tables).map(([k, rows]) => [k, Object.fromEntries(rows.map(r => [r.id, r.numeric_id]))]));
// Numeric id -> reference string id, for building server commands.
const moveNames = Object.fromEntries(data.tables.moves.map(r => [r.numeric_id, r.id]));
const rows = Object.fromEntries(Object.entries(data.tables).map(([k, list]) => [k, new Map(list.map(r => [r.id, r]))]));
const stats = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const limit = Number(process.env.MOVE_FIXTURE_LIMIT || 0);
verifyReference();

// Native classification mirror of engine/src/assets.rs::classify_move. A move
// whose data declaration is not fully implemented stays an operational error in
// the engine and is therefore not part of this corpus. Drift shows up loudly as
// a differential failure rather than silence.
// Mirror of engine/src/assets.rs HANDLED_MOVE_FIELDS / PORTED_MOVE_CALLBACK_KEYS.
// Rust remains authoritative; drift shows up as a loud differential failure.
// Rust is authoritative. Every classification input below is parsed directly
// out of the engine source so this mirror cannot drift: a move is only put in
// the corpus when the native classifier would also accept it.
const RUST_ASSETS = fs.readFileSync(new URL('../src/assets.rs', import.meta.url), 'utf8');
const RUST_EFFECTS = fs.readFileSync(new URL('../src/effects.rs', import.meta.url), 'utf8');
const rustList = name => {
  const m = RUST_ASSETS.match(new RegExp(`const ${name}: &\\[&str\\] = &\\[([\\s\\S]*?)\\];`));
  if (!m) throw new Error(`could not read ${name} from engine/src/assets.rs`);
  return [...m[1].matchAll(/"([^"]+)"/g)].map(x => x[1]);
};
const HANDLED_FIELDS = new Set(rustList('HANDLED_MOVE_FIELDS'));
const PORTED_CALLBACK_KEYS = new Set(rustList('PORTED_MOVE_CALLBACK_KEYS'));
const HANDLED_STATUSES = new Set(rustList('HANDLED_STATUSES'));
const HANDLED_VOLATILES = new Set(rustList('HANDLED_VOLATILES'));
const HANDLED_FLAGS = new Set(rustList('HANDLED_MOVE_FLAGS'));
const EXPLICIT_MOVES = (() => {
  const block = RUST_EFFECTS.match(/impl MoveBehavior \{\s*pub fn compile\(id: &str\) -> Self \{\s*match id \{([\s\S]*?)\n            _ => Self::Unimplemented,/);
  if (!block) throw new Error('could not read MoveBehavior::compile from engine/src/effects.rs');
  return new Set([...block[1].matchAll(/"([^"]+)"/g)].map(m => m[1]));
})();
const collectCallbacks = (value, out) => {
  if (!value || typeof value !== 'object') return out;
  if (typeof value.callback === 'string') out.push(value.callback);
  for (const nested of Object.values(value)) collectCallbacks(nested, out);
  return out;
};
const payloadHandled = effect => !effect || typeof effect !== 'object' || Object.entries(effect).every(([key, value]) =>
  key === 'self' ? payloadHandled(value) : key === 'chance' || key === 'boosts' ||
  key === 'onHit' ||
  (key === 'status' && HANDLED_STATUSES.has(value)) || (key === 'volatileStatus' && HANDLED_VOLATILES.has(value)));
function executable(id) {
  const m = dex.moves.get(id);
  const encoded = rows.moves.get(id)?.data;
  if (!m.exists || !encoded) return false;
  if (EXPLICIT_MOVES.has(id)) return true;
  if (collectCallbacks(encoded, []).some(key => !PORTED_CALLBACK_KEYS.has(key))) return false;
  if (Object.keys(encoded).some(key => !HANDLED_FIELDS.has(key))) return false;
  if (Object.keys(encoded.flags || {}).some(flag => encoded.flags[flag] && !HANDLED_FLAGS.has(flag))) return false;
  for (const key of ['secondary', 'self']) if (!payloadHandled(encoded[key])) return false;
  if ((encoded.secondaries || []).some(entry => !payloadHandled(entry))) return false;
  if (encoded.status && !HANDLED_STATUSES.has(encoded.status)) return false;
  if (encoded.volatileStatus && !HANDLED_VOLATILES.has(encoded.volatileStatus)) return false;
  // `selfSwitch: 'copyvolatile' | 'shedtail'` still transfers a volatile payload
  // the engine does not port; plain `selfSwitch` and `forceSwitch` are native.
  // Plain `selfSwitch` pivots are native; `copyvolatile`/`shedtail` payloads
  // and `forceSwitch` phazing are not ported yet.
  if (typeof encoded.selfSwitch === 'string' || encoded.forceSwitch ||
      encoded.pseudoWeather || encoded.slotCondition ||
      encoded.stallingMove || encoded.sleepUsable || encoded.multiaccuracy || encoded.mindBlownRecoil ||
      encoded.hasCrashDamage) return false;
  return true;
}

// Implemented support effects, mirrored from the native coverage report. A
// fixture never pairs the probed move with an unsupported ability or item.
// Ported abilities are derived from `Ability::is_ported` in the Rust source so
// this mirror cannot drift while the ability port advances. The block is a
// negated `matches!` over *unported* variants; everything else is executable.
const RUST_HOOKS = fs.readFileSync(new URL('../src/battle/hooks.rs', import.meta.url), 'utf8');
const UNPORTED_ABILITIES = (() => {
  const block = RUST_HOOKS.match(/pub fn is_ported\(self\) -> bool \{\s*!matches!\(\s*self,([\s\S]*?)\n\s*\)\n\s*\}/);
  if (!block) throw new Error('could not read Ability::is_ported from engine/src/battle/hooks.rs');
  return new Set([...block[1].matchAll(/Ability::(\w+)/g)].map(m => m[1].toLowerCase()));
})();
const abilityPorted = id => !UNPORTED_ABILITIES.has(id);

const FILLERS = [
  ['Goodra-Hisui', 'Shell Armor', ['Dragon Pulse', 'Protect'], ''],
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect'], ''],
  ['Perrserker', 'Battle Armor', ['Iron Head', 'Protect'], ''],
  ['Samurott', 'Shell Armor', ['Aqua Jet', 'Protect'], ''],
  ['Hydreigon', 'Levitate', ['Dragon Pulse', 'Protect'], ''],
  ['Falinks', 'Battle Armor', ['Smart Strike', 'Protect'], ''],
  ['Metagross', 'Clear Body', ['Iron Head', 'Protect'], ''],
  ['Milotic', 'Competitive', ['Surf', 'Protect'], ''],
  ['Scolipede', 'Swarm', ['X-Scissor', 'Protect'], ''],
  ['Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect'], ''],
  ['Blaziken', 'Speed Boost', ['Close Combat', 'Protect'], ''],
  ['Starmie', 'Natural Cure', ['Ice Beam', 'Protect'], ''],
];
const nameSets = team => team.map((set, i) => ({...set, name: `s${i}`}));
const setOf = (species, ability, moves, item, points) => ({name: species, species, ability, item, nature: 'Serious', level: 50, gender: 'M',
  moves, evs: {hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0, ...points}});
const fillerTeam = () => FILLERS.map(([species, ability, moves, item]) => setOf(species, ability, moves, item, {hp: 24, atk: 8, def: 8, spa: 8, spd: 8, spe: 4}));
const roster = p => Number(p.name.slice(-1));
// Privileged request detail so the native legal-action mask can be compared
// with the reference request at every boundary (development fixtures only).
const requestDetail = (session, side) => {
  if (session.battle.ended) return null;
  if (side.requestState === 'teampreview') {
    // A side that has already submitted its picks waits for the opponent.
    return {kind: side.isChoiceDone() ? 'Wait' : 'Preview', slots: [], bench: [], preview: [0, 1, 2, 3, 4, 5]};
  }
  const req = side.activeRequest ?? {};
  const kind = req.wait || side.isChoiceDone() ? 'Wait' : side.requestState === 'switch' ? 'Replacement' : 'Normal';
  const slots = [0, 1].map(slot => {
    const p = side.active[slot];
    const info = req.active?.[slot];
    const forced = Boolean(req.forceSwitch?.[slot]);
    if (!p) return {present: false, requires_replacement: forced, can_mega: false, moves: []};
    // Reference `getLockedMove()`: a charging or recharging Pokémon offers
    // exactly one entry (or none, for the Recharge pseudo-move) and refuses
    // switches. `p.moveSlots` alone would over-report the legal mask.
    const locked = p.getLockedMove();
    if (locked === 'recharge') {
      return {present: !p.fainted, requires_replacement: forced, can_mega: false,
        trapped: Boolean(info?.trapped), maybe_trapped: Boolean(info?.maybeTrapped),
        locked_recharge: true, trapped: true, maybe_trapped: false, moves: []};
    }
    if (locked) {
      const slotData = p.moveSlots.find(m => m.id === locked);
      return {present: !p.fainted, requires_replacement: forced, can_mega: false,
        trapped: Boolean(info?.trapped), maybe_trapped: Boolean(info?.maybeTrapped),
        locked: ids.moves[locked], trapped: true,
        moves: [{id: ids.moves[locked], pp: slotData?.pp ?? 0, disabled: false, target: slotData?.target ?? 'normal'}]};
    }
    return {present: !p.fainted, requires_replacement: forced, can_mega: Boolean(info?.canMegaEvo),
      trapped: Boolean(info?.trapped), maybe_trapped: Boolean(info?.maybeTrapped),
      moves: p.moveSlots.map(m => ({id: ids.moves[m.id], pp: m.pp, disabled: Boolean(m.disabled), target: m.target}))};
  });
  // Reference switch destinations are positions in the request team order,
  // which after preview is the pick order. The native request reports stable
  // roster indices, so map positions back through the fixture name suffix.
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
      active_slot: s.active.indexOf(p) >= 0 ? s.active.indexOf(p) : null, ability_ending: Boolean(p.abilityState.ending), cached_speed: p.speed ?? null,
      status: ids.conditions[p.status] ?? 0, boosts: Object.values(p.boosts), stats: [p.maxhp, ...Object.values(p.storedStats)], ability: ids.abilities[p.ability],
      item: ids.items[p.item] ?? 0, types: p.types.map(t => ids.types[toID(t)]), previous_item: ids.items[p.lastItem] ?? 0, can_mega: Boolean(p.canMegaEvo),
      pp: p.moveSlots.map(m => m.pp), volatiles: Object.keys(p.volatiles).sort()})),
    request_detail: requestDetail(session, s)}))});
const select = (kind, own_slot, destination = 255) => ({kind, own_slot, move_slot: 255, target_location: 0, switch_destination: destination, resource: 'None'});
const move = (slot, moveSlot, target = 0) => ({kind: 'Move', own_slot: slot, move_slot: moveSlot, target_location: target, switch_destination: 255, resource: 'None'});

function choose(session, sideIndex, plan) {
  const b = session.battle, side = b.sides[sideIndex], req = side.activeRequest;
  if (side.requestState === 'teampreview') return {actions: [0, 1, 4, 5].map((r, i) => select('Pick', i, r)), command: 'team 1256'};
  const bench = side.pokemon.filter(p => !p.fainted && !side.active.includes(p));
  const actions = [], commands = [];
  for (let slot = 0; slot < 2; slot++) {
    const p = side.active[slot];
    if (side.requestState === 'switch') {
      if (!req.forceSwitch[slot]) { commands.push('pass'); continue; }
      const reserve = bench.shift();
      if (reserve) { actions.push(select('Switch', slot, roster(reserve))); commands.push(`switch ${side.pokemon.indexOf(reserve) + 1}`); }
      else { actions.push(select('Pass', slot)); commands.push('pass'); }
      continue;
    }
    if (p.fainted) { actions.push(select('Pass', slot)); commands.push('pass'); continue; }
    // A charging or recharging Pokémon is served one entry (or the no-op
    // Recharge pseudo-move); anything else is rejected by the reference.
    const locked = p.getLockedMove();
    if (locked) {
      if (locked === 'recharge') {
        actions.push(move(slot, 255, 0));
      } else {
        const recorded = p.volatiles[locked]?.targetLoc ?? p.lastMoveTargetLoc ?? 0;
        const lockedSlot = Math.max(0, p.moveSlots.findIndex(m => m.id === locked));
        actions.push(move(slot, lockedSlot, recorded));
      }
      commands.push('move 1');
      continue;
    }
    const wanted = sideIndex === 0 && slot === 0 ? plan.moveSlot : plan.attackSlot;
    const wantedName = moveNames[wanted];
    const slotIndex = wantedName ? p.moveSlots.findIndex(m => m.id === wantedName && !m.disabled && m.pp > 0) : -1;
    const choice = slotIndex >= 0 ? slotIndex : p.moveSlots.findIndex(m => !m.disabled && m.pp > 0);
    const chosen = p.moveSlots[choice];
    if (!chosen) { actions.push(move(slot, 255, 0)); commands.push('move 1'); continue; }
    // Foe-side locations are relative: positive to the right, negative to the
    // left. Ask the reference which of the two foe slots is a legal target.
    const candidates = sideIndex === 0 ? [2, 1, -1, -2, 0] : [-2, -1, 1, 2, 0];
    const location = chosen && b.actions.targetTypeChoices(chosen.target)
      ? candidates.find(loc => b.validTargetLoc(loc, p, chosen.target)) ?? 0 : 0;
    actions.push(move(slot, choice, location));
    commands.push(`move ${choice + 1}${location ? ` ${location}` : ''}`);
  }
  return {actions, command: commands.join(', ')};
}

function holderFor(moveId) {
  for (const entry of scope.starting_species) {
    if (!entry.learnable_moves.includes(moveId)) continue;
    for (const ability of entry.abilities) {
      if (!abilityPorted(ability)) continue;
      const species = dex.species.get(entry.species);
      const abilityName = dex.abilities.get(ability).name;
      // A single-move holder can exhaust its PP before the battle ends, which
      // would strand the fixture with no legal choice. Every legal set may
      // carry four moves, so pair the probe with one more implemented move the
      // holder can actually learn.
      const spare = entry.learnable_moves.find(id => id !== moveId && executable(id));
      const moves = [dex.moves.get(moveId).name];
      if (spare) moves.push(dex.moves.get(spare).name);
      const set = setOf(species.name, abilityName, moves, '', {hp: 24, def: 8, spd: 8});
      if (!validator.validateSet(set)) return {set, species: species.name, ability};
    }
  }
  return null;
}

const fixtures = [];
const skipped = [];
const moves = scope.allowed_moves.filter(executable).sort();
for (const moveId of moves) {
  if (limit && fixtures.length >= limit) break;
  const holder = holderFor(moveId);
  if (!holder) { skipped.push({move: moveId, reason: 'no legal implemented holder'}); continue; }
  const holderBase = dex.species.get(holder.species).baseSpecies;
  const fillers = fillerTeam().filter(set => dex.species.get(set.species).baseSpecies !== holderBase);
  const teamA = nameSets([holder.set, ...fillers.slice(0, 5)]);
  const teamB = nameSets(fillerTeam().filter(set => dex.species.get(set.species).baseSpecies !== holderBase).slice(0, 6));
  const problems = validator.validateTeam(teamA) || validator.validateTeam(teamB);
  if (problems) { skipped.push({move: moveId, reason: problems.join('; ')}); continue; }
  const index = fixtures.length;
  const seed = [2026, 10, 7, 3000 + index];
  const session = new ReferenceSession({teams: [teamA, teamB], seed});
  const attackSlot = ids.moves.ironhead;
  const plan = {moveSlot: ids.moves[moveId], attackSlot};
  const fixture = {name: `move_${moveId}_${seed[3]}`, seed,
    teams: [teamA, teamB].map((team, side) => ({id: `move-${moveId}-${side}`, members: team.map(s => ({
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
      if (!result.accepted) { failure = JSON.stringify({move: moveId, side, choice, messages: result.messages}); break; }
      fixture.steps.push({side: side ? 'P2' : 'P1', ...choice, expected: compact(session)});
    }
    if (failure) break;
  }
  if (failure) { session.destroy(); skipped.push({move: moveId, reason: failure}); continue; }
  if (!session.battle.ended) { session.destroy(); skipped.push({move: moveId, reason: 'did not complete'}); continue; }
  const used = session.battle.log.some(line => line.startsWith('|move|p1a:') && line.split('|')[3] === dex.moves.get(moveId).name);
  if (!used) { session.destroy(); skipped.push({move: moveId, reason: 'move never fired'}); continue; }
  fixture.coverage = {move: moveId, holder: holder.species, ability: holder.ability,
    turns: fixture.steps.at(-1).expected.turn, statuses: [...new Set(session.battle.log.filter(x => x.startsWith('|-status|')).map(x => x.split('|')[3]))].sort()};
  fixtures.push(fixture);
  session.destroy();
}
fs.writeFileSync(new URL('../data/more_move_coverage.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify({moves: moves.length, fixtures: fixtures.length, skipped: skipped.length}));
if (skipped.length) console.log(JSON.stringify(skipped));
