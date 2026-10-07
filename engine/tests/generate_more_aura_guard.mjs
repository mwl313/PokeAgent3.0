// Development-only corpus for the global aura (Fairy Aura) and No Guard
// abilities, generated from the pinned reference. Merged into
// engine/data/turn-fixtures.json by scripts/export_engine_data.mjs.
import fs from 'node:fs';
import {createRequire} from 'node:module';
import {ReferenceSession, verifyReference, FORMAT, ORACLE_COMMIT} from '../reference.mjs';
const require = createRequire(import.meta.url);
const {TeamValidator, toID} = require('../../vendor/pokemon-showdown/dist/sim');
const validator = new TeamValidator(FORMAT);
const dex = validator.dex;
const data = JSON.parse(fs.readFileSync(new URL('../data/dex.json', import.meta.url), 'utf8'));
const ids = Object.fromEntries(Object.entries(data.tables)
  .map(([kind, rows]) => [kind, Object.fromEntries(rows.map(r => [r.id, r.numeric_id]))]));
const stats = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
verifyReference();

const set = (name, species, ability, item, moves, evs = {hp: 24, atk: 8, def: 8, spa: 8, spd: 8, spe: 4}) =>
  ({name, species, ability, item, nature: 'Serious', level: 50, gender: 'M', moves,
    evs, ivs: {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31}});

const auraTeam = () => [
  set('s0', 'Floette-Eternal', 'Flower Veil', 'Floettite', ['Moonblast', 'Dazzling Gleam', 'Protect']),
  set('s1', 'Goodra-Hisui', 'Shell Armor', '', ['Dragon Pulse', 'Protect']),
  set('s2', 'Torterra', 'Shell Armor', '', ['Seed Bomb', 'Protect']),
  set('s3', 'Perrserker', 'Battle Armor', '', ['Iron Head', 'Protect']),
  set('s4', 'Samurott', 'Shell Armor', '', ['Aqua Jet', 'Protect']),
  set('s5', 'Hydreigon', 'Levitate', '', ['Dragon Pulse', 'Protect']),
];
// No Guard has a legal base-form holder in Machamp (its other legality
// requirements are met), so the fixture does not depend on an unported
// base ability such as Pidgeot's Keen Eye.
const guardTeam = () => [
  set('s0', 'Machamp', 'No Guard', '', ['Dynamic Punch', 'Close Combat', 'Stone Edge', 'Protect']),
  set('s1', 'Goodra-Hisui', 'Shell Armor', '', ['Dragon Pulse', 'Protect']),
  set('s2', 'Torterra', 'Shell Armor', '', ['Seed Bomb', 'Protect']),
  set('s3', 'Perrserker', 'Battle Armor', '', ['Iron Head', 'Protect']),
  set('s4', 'Samurott', 'Shell Armor', '', ['Aqua Jet', 'Protect']),
  set('s5', 'Hydreigon', 'Levitate', '', ['Dragon Pulse', 'Protect']),
];

const roster = p => Number(p.name.slice(-1));
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
    request_detail: (() => {
      if (session.battle.ended) return null;
      if (s.requestState === 'teampreview') return {kind: s.isChoiceDone() ? 'Wait' : 'Preview', slots: [], bench: [], preview: [0, 1, 2, 3, 4, 5]};
      const req = s.activeRequest ?? {};
      const kind = req.wait || s.isChoiceDone() ? 'Wait' : s.requestState === 'switch' ? 'Replacement' : 'Normal';
      const slots = [0, 1].map(slot => {
        const p = s.active[slot];
        const info = req.active?.[slot];
        const forced = Boolean(req.forceSwitch?.[slot]);
        if (!p) return {present: false, requires_replacement: forced, can_mega: false, moves: []};
        return {present: !p.fainted, requires_replacement: forced, can_mega: Boolean(info?.canMegaEvo),
          trapped: Boolean(info?.trapped), maybe_trapped: Boolean(info?.maybeTrapped),
          moves: p.moveSlots.map(m => ({id: ids.moves[m.id], pp: m.pp, disabled: Boolean(m.disabled), target: m.target}))};
      });
      const bench = s.pokemon.map((p, i) => [p, i]).filter(([p]) => !p.fainted && !s.active.includes(p)).map(([p]) => roster(p));
      return {kind, slots, bench, preview: []};
    })(),
  }))});
const select = (kind, own_slot, destination = 255) => ({kind, own_slot, move_slot: 255, target_location: 0, switch_destination: destination, resource: 'None'});
const move = (slot, moveSlot, target = 0, mega = false) => ({kind: 'Move', own_slot: slot, move_slot: moveSlot, target_location: target, switch_destination: 255, resource: mega ? 'Mega' : 'None'});

function choose(session, sideIndex) {
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
    if (!p || p.fainted) { actions.push(select('Pass', slot)); commands.push('pass'); continue; }
    const mega = slot === 0 && b.turn === 1 && p.canMegaEvo;
    const usable = p.moveSlots.findIndex(m => !m.disabled && m.pp > 0);
    const choice = usable >= 0 ? usable : 0;
    const chosen = p.moveSlots[choice];
    const candidates = sideIndex === 0 ? [2, 1, -1, -2] : [-2, -1, 1, 2];
    const location = chosen && b.actions.targetTypeChoices(chosen.target)
      ? candidates.find(loc => b.validTargetLoc(loc, p, chosen.target)) ?? 0 : 0;
    actions.push(move(slot, choice, location, mega));
    commands.push(`move ${choice + 1}${location ? ` ${location}` : ''}${mega ? ' mega' : ''}`);
  }
  return {actions, command: commands.join(', ')};
}

const compiled = (team, side) => ({id: `aura_guard_${side}`, members: team.map(m => ({
  species: ids.species[toID(m.species)], ability: ids.abilities[toID(m.ability)], item: ids.items[toID(m.item)] ?? 0,
  nature: ids.natures[toID(m.nature)], gender: m.gender || '', level: 50, moves: m.moves.map(x => ids.moves[toID(x)]),
  points: stats.map(k => m.evs[k]), ivs: stats.map(k => m.ivs[k])}))});

const fixtures = [];
const cases = [
  ['aura_and_noguard', auraTeam(), guardTeam(), [2026, 10, 8, 4100]],
  ['aura_mirror', auraTeam(), auraTeam(), [2026, 10, 8, 4101]],
  ['noguard_hurricane', guardTeam(), auraTeam(), [2026, 10, 8, 4102]],
];
for (const [name, p1, p2, seed] of cases) {
  const problems = validator.validateTeam(p1) || validator.validateTeam(p2);
  if (problems) throw new Error(`${name}: ${problems.join('; ')}`);
  const session = new ReferenceSession({teams: [p1, p2], seed});
  const fixture = {name, seed, teams: [compiled(p1, 0), compiled(p2, 1)], initial: compact(session), steps: []};
  while (!session.battle.ended && fixture.steps.length < 200) {
    for (let side = 0; side < 2; side++) {
      if (session.battle.ended) break;
      const s = session.battle.sides[side];
      if (s.activeRequest?.wait || s.isChoiceDone()) continue;
      const choice = choose(session, side);
      const result = session.choose(side ? 'p2' : 'p1', choice.command);
      if (!result.accepted) throw new Error(JSON.stringify({name, side, choice, messages: result.messages}));
      fixture.steps.push({side: side ? 'P2' : 'P1', ...choice, expected: compact(session)});
    }
  }
  if (!session.battle.ended) throw new Error(`${name} did not complete`);
  const megas = session.battle.log.filter(x => x.startsWith('|-mega|')).length;
  if (megas === 0) throw new Error(`${name}: no Mega evolution occurred`);
  fixture.coverage = {family: 'aura_guard', megas};
  fixtures.push(fixture);
  session.destroy();
}
fs.writeFileSync(new URL('../data/more_aura_guard.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify({fixtures: fixtures.length, steps: fixtures.reduce((n, f) => n + f.steps.length, 0)}));
