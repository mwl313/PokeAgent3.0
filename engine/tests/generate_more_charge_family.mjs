// Development-only interaction corpus for the two-turn charge family and the
// forced Recharge turn.
//
// The generic move corpus records one battle per move, which exercises the
// basic charge/release and Hyper Beam recharge paths. This generator scripts
// the interactions those battles cannot reach: weather-skipped charges (sun /
// rain), the prepare-step boosts, semi-invulnerability with its exception
// lists, doubled damage against a charging target, and the Protect-cancelled
// recharge. Every fixture is a complete legal reference battle recorded at
// every decision boundary, including the served request mask.
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
// Numeric id -> reference string id, for building server commands.
const moveNames = Object.fromEntries(data.tables.moves.map(r => [r.numeric_id, r.id]));
const stats = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
verifyReference();

const nameSets = team => team.map((set, i) => ({...set, name: `s${i}`}));
const setOf = (species, ability, moves, points = {hp: 24, atk: 8, def: 8, spa: 8, spd: 8, spe: 4}) =>
  ({name: species, species, ability, item: '', nature: 'Serious', level: 50, gender: 'M', moves,
    evs: {hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0, ...points}});
const roster = p => Number(p.name.slice(-1));

// Fillers keep every fixture battle decisive without adding unported moves.
const FILLERS = [
  ['Goodra-Hisui', 'Shell Armor', ['Dragon Pulse', 'Protect']],
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']],
  ['Metagross', 'Clear Body', ['Iron Head', 'Protect']],
  ['Milotic', 'Competitive', ['Surf', 'Protect']],
  ['Scolipede', 'Swarm', ['X-Scissor', 'Protect']],
  ['Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']],
  ['Starmie', 'Natural Cure', ['Ice Beam', 'Protect']],
].map(([species, ability, moves]) => setOf(species, ability, moves));

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
    // Reference `getLockedMove()`: a charging or recharging Pokémon is served
    // exactly one entry and refuses switches; `moveSlots` alone would
    // over-report the legal mask.
    const locked = p.getLockedMove();
    if (locked === 'recharge') {
      return {present: !p.fainted, requires_replacement: forced, can_mega: false,
        locked_recharge: true, trapped: true, moves: []};
    }
    if (locked) {
      const slotData = p.moveSlots.find(m => m.id === locked);
      return {present: !p.fainted, requires_replacement: forced, can_mega: false,
        locked: ids.moves[locked], trapped: true,
        moves: [{id: ids.moves[locked], pp: slotData?.pp ?? 0, disabled: false, target: slotData?.target ?? 'normal'}]};
    }
    return {present: !p.fainted, requires_replacement: forced, can_mega: Boolean(info?.canMegaEvo),
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
      active_slot: s.active.indexOf(p) >= 0 ? s.active.indexOf(p) : null, ability_ending: Boolean(p.abilityState.ending), cached_speed: p.speed ?? null,
      status: ids.conditions[p.status] ?? 0, boosts: Object.values(p.boosts), stats: [p.maxhp, ...Object.values(p.storedStats)], ability: ids.abilities[p.ability],
      item: ids.items[p.item] ?? 0, types: p.types.map(t => ids.types[toID(t)]), previous_item: ids.items[p.lastItem] ?? 0, can_mega: Boolean(p.canMegaEvo),
      pp: p.moveSlots.map(m => m.pp), volatiles: Object.keys(p.volatiles).sort()})),
    request_detail: requestDetail(session, s)}))});

const select = (kind, own_slot, destination = 255) =>
  ({kind, own_slot, move_slot: 255, target_location: 0, switch_destination: destination, resource: 'None'});
const moveAction = (slot, moveSlot, target = 0) =>
  ({kind: 'Move', own_slot: slot, move_slot: moveSlot, target_location: target, switch_destination: 255, resource: 'None'});

// `plan(side, slot, turn, p)` returns a move id to use, or null for the
// scripted default (Protect for side 1's holder slot, filler attack otherwise).
function choose(session, sideIndex, plan) {
  const b = session.battle, side = b.sides[sideIndex], req = side.activeRequest;
  if (side.requestState === 'teampreview') {
    return {actions: [0, 1, 4, 5].map((r, i) => select('Pick', i, r)), command: 'team 1256'};
  }
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
    // A locked Pokémon is served exactly one entry, so the only legal command
    // is `move 1` with no target (the location comes from the volatile).
    const locked = p.getLockedMove();
    if (locked) {
      if (locked === 'recharge') {
        // Forced Recharge: the native action is the no-op pseudo-move.
        actions.push(moveAction(slot, 255, 0));
      } else {
        // The location is the one recorded by `twoturnmove.onStart`; the
        // native action uses the locked move's real slot index.
        const recorded = p.volatiles[locked]?.targetLoc ?? p.lastMoveTargetLoc ?? 0;
        const lockedSlot = Math.max(0, p.moveSlots.findIndex(m => m.id === locked));
        actions.push(moveAction(slot, lockedSlot, recorded));
      }
      commands.push('move 1');
      continue;
    }
    const wanted = plan(sideIndex, slot, b.turn, p);
    // `moveSlots[].id` is the reference's string id, while the fixture carries
    // compact numeric ids; resolve before matching.
    const wantedName = wanted ? moveNames[wanted] : null;
    const slotIndex = wantedName ? p.moveSlots.findIndex(m => m.id === wantedName && !m.disabled && m.pp > 0) : -1;
    const choice = slotIndex >= 0 ? slotIndex : p.moveSlots.findIndex(m => !m.disabled && m.pp > 0);
    const chosen = p.moveSlots[choice];
    if (!chosen) {
      // Every move is out of PP: the reference serves one Struggle entry.
      actions.push(moveAction(slot, 255, 0));
      commands.push('move 1');
      continue;
    }
    const candidates = sideIndex === 0 ? [2, 1, -1, -2, 0] : [-2, -1, 1, 2, 0];
    const location = chosen && b.actions.targetTypeChoices(chosen.target)
      ? candidates.find(loc => b.validTargetLoc(loc, p, chosen.target)) ?? 0 : 0;
    actions.push(moveAction(slot, choice, location));
    commands.push(`move ${choice + 1}${location ? ` ${location}` : ''}`);
  }
  return {actions, command: commands.join(', ')};
}

function teamsFor(holder, opponentHolder, usedSpecies) {
  const pick = (extra, exclude) => {
    const out = [];
    for (const set of [...extra, ...FILLERS]) {
      const base = dex.species.get(set.species).baseSpecies;
      if (exclude.has(base)) continue;
      exclude.add(base);
      out.push(set);
      if (out.length === 6) break;
    }
    return out;
  };
  const teamA = nameSets(pick(holder, new Set()));
  const teamB = nameSets(pick(opponentHolder, new Set()));
  const problems = validator.validateTeam(teamA) || validator.validateTeam(teamB);
  if (problems) throw new Error(`invalid fixture teams (${usedSpecies}): ${problems.join('; ')}`);
  return [teamA, teamB];
}

function play({name, holder, opponent = [], plan, require: requireEvidence, seed}) {
  const [teamA, teamB] = teamsFor(holder, opponent, name);
  const session = new ReferenceSession({teams: [teamA, teamB], seed});
  const fixture = {name: `${name}_${seed[3]}`, seed,
    teams: [teamA, teamB].map((team, side) => ({id: `${name}-${side}`, members: team.map(s => ({
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
      if (!result.accepted) { failure = JSON.stringify({name, side, choice, messages: result.messages}); break; }
      fixture.steps.push({side: side ? 'P2' : 'P1', ...choice, expected: compact(session)});
    }
    if (failure) break;
  }
  const log = session.battle.log.slice();
  if (failure) { session.destroy(); return {name, reason: failure}; }
  if (!session.battle.ended) { session.destroy(); return {name, reason: 'did not complete'}; }
  const evidence = requireEvidence(fixture, log);
  session.destroy();
  if (!evidence.ok) return {name, reason: evidence.reason};
  fixture.coverage = evidence.coverage;
  return {fixture};
}

const slotVolatiles = (fixture, side, slot) => fixture.steps.flatMap(step =>
  step.expected.sides[side].pokemon
    .filter(p => p.active_slot === slot)
    .map(p => ({volatiles: p.volatiles, boosts: p.boosts})));

/// The protocol lines of one battle turn, for evidence that a specific
/// interaction happened inside that turn.
const turnLog = (log, turn) => {
  const start = log.findIndex(line => line === `|turn|${turn}`);
  if (start < 0) return [];
  const end = log.findIndex((line, i) => i > start && line.startsWith('|turn|'));
  return log.slice(start, end < 0 ? undefined : end);
};

/// Whether an active slot holds a volatile at every recorded boundary of the
/// given turn.
const volatileDuringTurn = (fixture, turn, side, slot, volatile) => fixture.steps
  .filter(step => step.expected.turn === turn)
  .flatMap(step => step.expected.sides[side].pokemon.filter(p => p.active_slot === slot))
  .some(p => p.volatiles.includes(volatile));

const scenarios = [
  {
    // Drought sets sun before turn 1, so Solar Beam skips the charge turn.
    name: 'charge_sun_skip',
    holder: [setOf('Ninetales', 'Drought', ['Solar Beam', 'Protect']),
      setOf('Venusaur', 'Overgrow', ['Solar Beam', 'Protect'])],
    plan: (side, slot) => (side === 0 ? ids.moves.solarbeam : null),
    require: (fixture, log) => {
      const turn1 = turnLog(log, 1);
      const fired = turn1.some(line => line.startsWith('|move|p1a:') && line.includes('Solar Beam'));
      const charged = volatileDuringTurn(fixture, 1, 0, 0, 'twoturnmove');
      const damage = turn1.some(line => line.startsWith('|-damage|p2'));
      return fired && !charged && damage
        ? {ok: true, coverage: {weather: 'sun', charge_skipped: true}}
        : {ok: false, reason: `sun skip not observed (fired=${fired} charged=${charged})`};
    },
  },
  {
    // Rain forces the charge turn and halves the release.
    name: 'charge_rain_slow',
    holder: [setOf('Venusaur', 'Overgrow', ['Solar Beam', 'Protect'])],
    opponent: [setOf('Pelipper', 'Drizzle', ['Surf', 'Protect'])],
    plan: (side, slot) => (side === 0 && slot === 0 ? ids.moves.solarbeam : null),
    require: (fixture, log) => {
      const charged = slotVolatiles(fixture, 0, 0).some(v => v.volatiles.includes('twoturnmove'));
      const locked = fixture.steps.some(step => step.expected.sides[0].request_detail?.slots?.[0]?.locked === ids.moves.solarbeam);
      const released = log.filter(line => line.startsWith('|move|p1a:') && line.includes('Solar Beam')).length >= 2;
      return charged && locked && released
        ? {ok: true, coverage: {weather: 'rain', charge_turns: 2, locked_request: true}}
        : {ok: false, reason: `rain charge not observed (charged=${charged} locked=${locked} released=${released})`};
    },
  },
  {
    // Rain completes Electro Shot immediately and still applies the +1 SpA.
    name: 'charge_electroshot_rain',
    holder: [setOf('Archaludon', 'Sturdy', ['Electro Shot', 'Protect'])],
    opponent: [setOf('Pelipper', 'Drizzle', ['Surf', 'Protect'])],
    plan: (side, slot) => (side === 0 && slot === 0 ? ids.moves.electroshot : null),
    require: (fixture) => {
      const boosted = slotVolatiles(fixture, 0, 0).some(v => v.boosts[2] === 1);
      const charged = slotVolatiles(fixture, 0, 0).some(v => v.volatiles.includes('twoturnmove'));
      return boosted && !charged
        ? {ok: true, coverage: {weather: 'rain', charge_skipped: true, prepare_boost: 'spa'}}
        : {ok: false, reason: `electro shot rain path not observed (boosted=${boosted} charged=${charged})`};
    },
  },
  {
    // Meteor Beam charges, boosting SpA on the prepare step.
    name: 'charge_meteorbeam_boost',
    holder: [setOf('Aerodactyl', 'Rock Head', ['Meteor Beam', 'Protect'])],
    plan: (side, slot) => (side === 0 && slot === 0 ? ids.moves.meteorbeam : null),
    require: (fixture, log) => {
      const charged = slotVolatiles(fixture, 0, 0).some(v => v.volatiles.includes('twoturnmove'));
      const boosted = slotVolatiles(fixture, 0, 0).some(v => v.boosts[2] === 1);
      const released = log.filter(line => line.startsWith('|move|p1a:') && line.includes('Meteor Beam')).length >= 2;
      return charged && boosted && released
        ? {ok: true, coverage: {charge_turns: 2, prepare_boost: 'spa'}}
        : {ok: false, reason: `meteor beam charge not observed (charged=${charged} boosted=${boosted} released=${released})`};
    },
  },
  {
    // Fly's own condition makes the user semi-invulnerable for the charge turn.
    name: 'charge_fly_invulnerable',
    holder: [setOf('Aerodactyl', 'Rock Head', ['Fly', 'Protect'])],
    plan: (side, slot) => (side === 0 && slot === 0 ? ids.moves.fly : null),
    require: (fixture, log) => {
      const flew = slotVolatiles(fixture, 0, 0).some(v => v.volatiles.includes('fly'));
      const missed = log.some(line => line.startsWith('|-miss|'));
      const released = log.filter(line => line.startsWith('|move|p1a:') && line.includes('Fly')).length >= 2;
      return flew && missed && released
        ? {ok: true, coverage: {semi_invulnerable: true, miss_observed: true}}
        : {ok: false, reason: `fly invulnerability not observed (flew=${flew} missed=${missed} released=${released})`};
    },
  },
  {
    // Dig doubles the damage of the listed Earthquake from a charging target.
    name: 'charge_dig_earthquake',
    holder: [setOf('Aggron', 'Sturdy', ['Dig', 'Protect'])],
    opponent: [setOf('Torterra', 'Shell Armor', ['Earthquake', 'Protect'])],
    plan: (side, slot) => (side === 0 && slot === 0 ? ids.moves.dig
      : side === 1 && slot === 0 ? ids.moves.earthquake : null),
    require: (fixture, log) => {
      const dug = slotVolatiles(fixture, 0, 0).some(v => v.volatiles.includes('dig'));
      const quake = log.some(line => line.startsWith('|move|p2a:') && line.includes('Earthquake'));
      return dug && quake
        ? {ok: true, coverage: {semi_invulnerable: true, exception_used: 'earthquake'}}
        : {ok: false, reason: `dig/earthquake interaction not observed (dug=${dug} quake=${quake})`};
    },
  },
  {
    // A successful Hyper Beam forces the Recharge pseudo-move next turn.
    name: 'recharge_hyperbeam',
    holder: [setOf('Hydreigon', 'Levitate', ['Hyper Beam', 'Protect'])],
    plan: (side, slot, turn) => (side === 0 && slot === 0 && turn === 1 ? ids.moves.hyperbeam
      : side === 0 && slot === 0 ? ids.moves.dragonpulse : null),
    require: (fixture, log) => {
      const recharging = fixture.steps.some(step => step.expected.sides[0].request_detail?.slots?.[0]?.locked_recharge === true);
      const volatileSeen = slotVolatiles(fixture, 0, 0).some(v => v.volatiles.includes('mustrecharge'));
      const cant = log.some(line => line.startsWith('|cant|p1a:') && line.includes('recharge'));
      return recharging && volatileSeen && cant
        ? {ok: true, coverage: {recharge_turn: true, locked_recharge_request: true}}
        : {ok: false, reason: `recharge not observed (request=${recharging} volatile=${volatileSeen} cant=${cant})`};
    },
  },
  {
    // A blocked Hyper Beam never starts the recharge.
    name: 'recharge_protect_cancel',
    holder: [setOf('Hydreigon', 'Levitate', ['Hyper Beam', 'Dragon Pulse', 'Protect'])],
    plan: (side, slot, turn) => (side === 0 && slot === 0 && turn === 1 ? ids.moves.hyperbeam : null),
    require: (fixture, log) => {
      const fired = log.some(line => line.startsWith('|move|p1a:') && line.includes('Hyper Beam'));
      const blocked = log.some(line => line.startsWith('|-activate|p2') && line.includes('Protect'));
      // With Hyper Beam used only into Protect and Dragon Pulse afterwards,
      // the forced Recharge turn must never appear.
      const recharged = fixture.steps.some(step =>
        step.expected.sides[0].pokemon.some(p => p.active_slot === 0 && p.volatiles.includes('mustrecharge')))
        || fixture.steps.some(step => step.expected.sides[0].request_detail?.slots?.[0]?.locked_recharge === true);
      return fired && blocked && !recharged
        ? {ok: true, coverage: {blocked_hyper_beam: true, recharge_cancelled: true}}
        : {ok: false, reason: `protect cancel not observed (fired=${fired} blocked=${blocked} recharged=${recharged})`};
    },
  },
];

const fixtures = [];
const skipped = [];
for (const [index, scenario] of scenarios.entries()) {
  // Protect must be the opponent's first action for the cancel scenario.
  const plan = scenario.name === 'recharge_protect_cancel'
    ? (side, slot, turn) => (side === 1
        ? (turn === 1 ? ids.moves.protect : null)
        : (slot === 0 ? (turn === 1 ? ids.moves.hyperbeam : ids.moves.dragonpulse) : ids.moves.dragonpulse))
    : scenario.plan;
  let result = null;
  for (const seedWord of [4000 + index, 4100 + index, 4200 + index]) {
    result = play({...scenario, plan, seed: [2026, 10, 7, seedWord]});
    if (result.fixture) break;
  }
  if (result?.fixture) fixtures.push(result.fixture);
  else skipped.push({name: scenario.name, reason: result?.reason ?? 'no seed produced the required behaviour'});
}

fs.writeFileSync(new URL('../data/more_charge_family.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify({fixtures: fixtures.length, skipped: skipped.length,
  steps: fixtures.reduce((n, f) => n + f.steps.length, 0)}));
if (skipped.length) console.log(JSON.stringify(skipped, null, 1));
if (fixtures.length !== scenarios.length) process.exitCode = 1;
