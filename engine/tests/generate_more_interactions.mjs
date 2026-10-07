// Development-only interaction corpus: battles that combine landed mechanic
// families so their ordering, request-mask and RNG interactions are compared
// against the pinned reference at every decision boundary.
import fs from 'node:fs';
import {createRequire} from 'node:module';
import {ReferenceSession, verifyReference, FORMAT, ORACLE_COMMIT} from '../reference.mjs';
const require = createRequire(import.meta.url);
const {TeamValidator, toID} = require('../../vendor/pokemon-showdown/dist/sim');
const validator = new TeamValidator(FORMAT);
const dex = validator.dex;
const data = JSON.parse(fs.readFileSync(new URL('../data/dex.json', import.meta.url), 'utf8'));
const scope = JSON.parse(fs.readFileSync(new URL('../data/scope.json', import.meta.url), 'utf8'));
const ids = Object.fromEntries(Object.entries(data.tables)
  .map(([kind, rows]) => [kind, Object.fromEntries(rows.map(r => [r.id, r.numeric_id]))]));
const stats = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
verifyReference();

const IMPLEMENTED = new Set(['battlearmor', 'shellarmor', 'levitate', 'blaze', 'torrent', 'overgrow', 'swarm', 'intimidate', 'defiant', 'competitive',
  'clearbody', 'innerfocus', 'owntempo', 'oblivious', 'speedboost', 'toughclaws', 'technician', 'adaptability', 'megalauncher', 'ironfist', 'sharpness',
  'strongjaw', 'hugepower', 'purepower', 'filter', 'solidrock', 'multiscale', 'thickfat', 'regenerator', 'naturalcure', 'rockhead', 'reckless', 'liquidooze',
  'infiltrator', 'drizzle', 'drought', 'sandstream', 'snowwarning', 'swiftswim', 'chlorophyll', 'sandrush', 'slushrush', 'electricsurge', 'grassysurge',
  'mistysurge', 'psychicsurge', 'raindish', 'icebody', 'solarpower', 'cloudnine', 'airlock', 'sandforce', 'sandveil', 'snowcloak', 'overcoat', 'hydration',
  'dryskin', 'waterabsorb', 'voltabsorb', 'eartheater', 'sapsipper', 'motordrive', 'static', 'flashfire', 'lightningrod', 'stormdrain', 'pixilate', 'aerilate',
  'refrigerate', 'galvanize', 'normalize', 'dragonize', 'liquidvoice', 'hypercutter', 'synchronize', 'flowerveil']);

const FILLERS = [
  ['Goodra-Hisui', 'Shell Armor', ['Dragon Pulse', 'Protect']],
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']],
  ['Perrserker', 'Battle Armor', ['Iron Head', 'Protect']],
  ['Samurott', 'Shell Armor', ['Aqua Jet', 'Protect']],
  ['Hydreigon', 'Levitate', ['Dragon Pulse', 'Protect']],
  ['Falinks', 'Battle Armor', ['Smart Strike', 'Protect']],
];
const setOf = (name, species, ability, moves, points = {hp: 24, atk: 8, def: 8, spa: 8, spd: 8, spe: 4}) =>
  ({name, species, ability, item: '', nature: 'Serious', level: 50, gender: 'M', moves,
    evs: points, ivs: {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31}});

/// Finds one legal set that knows every requested move and can run natively.
function holderFor(required) {
  // Try the full requirement first, then degrade to the leading moves so a
  // missing secondary move does not block the fixture.
  for (let keep = required.length; keep >= 1; keep--) {
    const needed = required.slice(0, keep);
    const found = holderForExact(needed);
    if (found) return found;
  }
  throw new Error(`no legal native holder for ${required.join(', ')}`);
}
function holderForExact(required) {
  for (const entry of scope.starting_species) {
    if (!required.every(m => entry.learnable_moves.includes(m))) continue;
    for (const ability of entry.abilities) {
      if (!IMPLEMENTED.has(ability)) continue;
      const species = dex.species.get(entry.species);
      const moves = required.map(m => dex.moves.get(m).name);
      if (moves.length < 4) moves.push('Protect');
      // The fixture name suffix is the roster index map, so the lead set must
      // end in its roster digit just like the filler sets do.
      const set = setOf('h0', species.name, dex.abilities.get(ability).name, moves);
      if (!validator.validateSet(set)) return set;
    }
  }
  return null;
}

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
    const locked = p.getLockedMove();
    if (locked === 'recharge') {
      return {present: !p.fainted, requires_replacement: forced, can_mega: false, locked_recharge: true, trapped: true, moves: []};
    }
    if (locked) {
      const slotData = p.moveSlots.find(m => m.id === locked);
      return {present: !p.fainted, requires_replacement: forced, can_mega: false, locked: ids.moves[locked], trapped: true,
        moves: [{id: ids.moves[locked], pp: slotData?.pp ?? 0, disabled: false, target: slotData?.target ?? 'normal'}]};
    }
    return {present: !p.fainted, requires_replacement: forced, can_mega: Boolean(info?.canMegaEvo),
      moves: p.moveSlots.map(m => ({id: ids.moves[m.id], pp: m.pp, disabled: Boolean(m.disabled), target: m.target}))};
  });
  const bench = side.pokemon.map((p, i) => [p, i]).filter(([p]) => !p.fainted && !side.active.includes(p)).map(([p]) => roster(p));
  return {kind, slots, bench, preview: []};
};
const compact = session => ({turn: session.battle.turn, rng_seed: session.battle.prng.getSeed(),
  climate: {raw: session.battle.field.weather, effective: session.battle.field.effectiveWeather(), suppressed: session.battle.field.suppressingWeather()},
  field: [...(session.battle.field.weather ? [[ids.conditions[session.battle.field.weather], session.battle.field.weatherState.duration, session.battle.field.weatherState.source.side.n]] : []),
    ...(session.battle.field.terrain ? [[ids.conditions[session.battle.field.terrain], session.battle.field.terrainState.duration, session.battle.field.terrainState.source.side.n]] : []),
    ...Object.entries(session.battle.field.pseudoWeather).map(([id, effect]) => [ids.conditions[id], effect.duration, effect.source.side.n])].sort((a, b) => a[0] - b[0]),
  terminated: session.battle.ended, winner: session.battle.ended ? session.battle.winner || null : null,
  sides: session.battle.sides.map(s => ({request: session.battle.ended ? 'Finished' : s.activeRequest?.wait || s.isChoiceDone() ? 'Wait' : s.requestState === 'teampreview' ? 'Preview' : s.requestState === 'switch' ? 'Replacement' : 'Normal',
    conditions: Object.entries(s.sideConditions).map(([id, state]) => [ids.conditions[id], state.duration]).sort((a, b) => a[0] - b[0]),
    pokemon: s.pokemon.map(p => ({roster: roster(p), species: ids.species[p.species.id], hp: p.hp, max_hp: p.maxhp, fainted: p.fainted,
      active_slot: s.active.indexOf(p) >= 0 ? s.active.indexOf(p) : null, ability_ending: Boolean(p.abilityState.ending), cached_speed: p.speed ?? null,
      status: ids.conditions[p.status] ?? 0, boosts: Object.values(p.boosts), stats: [p.maxhp, ...Object.values(p.storedStats)], ability: ids.abilities[p.ability],
      item: ids.items[p.item] ?? 0, types: p.types.map(t => ids.types[toID(t)]), previous_item: ids.items[p.lastItem] ?? 0, can_mega: Boolean(p.canMegaEvo),
      pp: p.moveSlots.map(m => m.pp), volatiles: Object.keys(p.volatiles).sort()})),
    request_detail: requestDetail(session, s)}))});
const select = (kind, own_slot, destination = 255) => ({kind, own_slot, move_slot: 255, target_location: 0, switch_destination: destination, resource: 'None'});
const move = (slot, moveSlot, target = 0) => ({kind: 'Move', own_slot: slot, move_slot: moveSlot, target_location: target, switch_destination: 255, resource: 'None'});

/// Scripted opener: each side uses the given move index for its first slot on
/// each turn, then falls back to the first usable move. Slot 1 always protects
/// or attacks so the interactions land in a normal doubles turn.
function choose(session, sideIndex, plan) {
  const b = session.battle, side = b.sides[sideIndex], req = side.activeRequest;
  if (side.requestState === 'teampreview') return {actions: [0, 1, 4, 5].map((r, i) => select('Pick', i, r)), command: 'team 1256'};
  const bench = side.pokemon.filter(p => !p.fainted && !side.active.includes(p));
  const turn = Math.min(b.turn, plan.length) - 1;
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
    if (!p || p.fainted) { actions.push(select('Pass', slot)); commands.push('pass'); continue; }
    const lockedMove = p.getLockedMove();
    if (lockedMove) {
      // A charging/recharging Pokemon is served exactly one entry: the command
      // string is always `move 1`, while the native action encodes either the
      // recorded charge target or the Recharge pseudo-move (no slot).
      if (lockedMove === 'recharge') {
        actions.push(move(slot, 255, 0));
      } else {
        const recorded = p.volatiles[lockedMove]?.targetLoc ?? p.lastMoveTargetLoc ?? 0;
        const lockedSlot = Math.max(0, p.moveSlots.findIndex(m => m.id === lockedMove));
        actions.push(move(slot, lockedSlot, recorded));
      }
      commands.push('move 1');
      continue;
    }
    const requested = slot === 0 ? (plan[Math.max(turn, 0)]?.[sideIndex] ?? 0) : 0;
    // The plan is a hint: a holder may know fewer moves than the script names.
    const wanted = requested < p.moveSlots.length ? requested : 0;
    const usable = p.moveSlots.findIndex((m, i) => i === wanted && !m.disabled && m.pp > 0);
    const fallback = p.moveSlots.findIndex(m => !m.disabled && m.pp > 0);
    if (usable < 0 && fallback < 0) {
      // Every move is disabled: the reference resolves this as Struggle, which
      // takes no explicit target.
      actions.push(move(slot, 0, 0));
      commands.push('move 1');
      continue;
    }
    const choice = usable >= 0 ? usable : fallback;
    const chosen = p.moveSlots[choice];
    const candidates = sideIndex === 0 ? [2, 1, -1, -2, 0] : [-2, -1, 1, 2, 0];
    const location = chosen && b.actions.targetTypeChoices(chosen.target)
      ? candidates.find(loc => b.validTargetLoc(loc, p, chosen.target)) ?? 0 : 0;
    actions.push(move(slot, choice, location));
    commands.push(`move ${choice + 1}${location ? ` ${location}` : ''}`);
  }
  return {actions, command: commands.join(', ')};
}

const compile = (team, side, name) => ({id: `${name}_${side}`, members: team.map(m => ({
  species: ids.species[toID(m.species)], ability: ids.abilities[toID(m.ability)], item: ids.items[toID(m.item)] ?? 0,
  nature: ids.natures[toID(m.nature)], gender: m.gender || '', level: 50, moves: m.moves.map(x => ids.moves[toID(x)]),
  points: stats.map(k => m.evs[k]), ivs: stats.map(k => m.ivs[k])}))});

const cases = [];
const addCase = (name, p1Set, p2Set, plan, seed) => {
  // The fixture name suffix is the roster index, so the lead (suffix 0) and the
  // five reserves (suffixes 1..5) must not collide.
  const p1 = [p1Set, ...FILLERS.map(([species, ability, moves], i) => setOf(`f${i + 1}`, species, ability, moves)).slice(0, 5)];
  const p2 = [p2Set, ...FILLERS.map(([species, ability, moves], i) => setOf(`g${i + 1}`, species, ability, moves)).slice(0, 5)];
  const problems = validator.validateTeam(p1) || validator.validateTeam(p2);
  if (problems) throw new Error(`${name}: ${problems.join('; ')}`);
  const session = new ReferenceSession({teams: [p1, p2], seed});
  const fixture = {name, seed, teams: [compile(p1, 0, name), compile(p2, 1, name)], initial: compact(session), steps: []};
  while (!session.battle.ended && fixture.steps.length < 260) {
    for (let side = 0; side < 2; side++) {
      if (session.battle.ended) break;
      const s = session.battle.sides[side];
      if (s.activeRequest?.wait || s.isChoiceDone()) continue;
      const choice = choose(session, side, plan);
      const result = session.choose(side ? 'p2' : 'p1', choice.command);
      if (!result.accepted) throw new Error(JSON.stringify({name, side, choice, messages: result.messages}));
      fixture.steps.push({side: side ? 'P2' : 'P1', ...choice, expected: compact(session)});
    }
  }
  if (!session.battle.ended) throw new Error(`${name} did not complete`);
  fixture.coverage = {family: 'interactions', name};
  cases.push(fixture);
  session.destroy();
};

// 1. Encore + Disable + Taunt across both sides: mask shape and ordering.
const encorer = holderFor(['encore', 'disable', 'taunt']);
const taunter = holderFor(['taunt', 'encore', 'disable']);
addCase('lock_crossfire', encorer, taunter, [[0, 0], [1, 1], [2, 2], [3, 0], [0, 1], [1, 2], [2, 0], [3, 1]], [2026, 10, 8, 4200]);
// 2. Imprison + shared move pool: the imprisoning side must not restrict
//    its own selections while the foe's shared moves become illegal.
const imprisoner = holderFor(['imprison']);
const shared = holderFor(['imprison', 'protect']);
addCase('imprison_shared_pool', imprisoner, shared, [[0, 0], [1, 0], [0, 0], [1, 0], [0, 0], [1, 0]], [2026, 10, 8, 4201]);
// 3. Charge + weather change while charging, then recharge + switch.
const solar = holderFor(['solarbeam', 'sunnyday']);
const rain = holderFor(['raindance', 'hyperbeam']);
addCase('charge_weather_recharge_switch', solar, rain, [[0, 1], [1, 0], [0, 0], [1, 1], [0, 1], [1, 0], [0, 0], [1, 0]], [2026, 10, 8, 4202]);
// 4. Helping Hand + redirection + priority tie on the same turn.
const helper = holderFor(['helpinghand', 'followme']);
const powder = holderFor(['ragepowder', 'spore']);
addCase('help_redirect_tie', helper, powder, [[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1]], [2026, 10, 8, 4203]);

fs.writeFileSync(new URL('../data/more_interactions.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures: cases}) + '\n');
console.log(JSON.stringify({fixtures: cases.length, steps: cases.reduce((n, f) => n + f.steps.length, 0)}));
