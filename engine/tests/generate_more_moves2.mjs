// Development-only corpus for the damage-condition and control move batch:
// - Assurance doubles when the target already lost HP this turn.
// - Lash Out doubles when the user's stats were lowered this turn.
// - Temper Flare doubles when the user's previous move failed.
// - Gigaton Hammer is unselectable while the holder's last move was itself.
// - Steel Roller fails without a terrain and clears the terrain on a hit or
//   after a Substitute takes the damage.
// - Upper Hand only runs against a queued positive-priority damaging move and
//   flinches the target on a landed hit.
// Every fixture is a complete legal synthetic reference battle recorded at
// every decision boundary, with the reference's per-turn flags included so the
// native state is compared directly.
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
const setOf = (species, ability, moves, item = '', points = {hp: 32, atk: 0, def: 17, spa: 0, spd: 17, spe: 0}, gender = 'M') =>
  ({name: species, species, ability, item, nature: 'Serious', level: 50, gender, moves,
    evs: {hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0, ...points}});
const offensive = (species, ability, moves, item = '') =>
  setOf(species, ability, moves, item, {hp: 2, atk: 32, def: 0, spa: 32, spd: 0, spe: 0});
const FILLERS = [
  ['Metagross', 'Clear Body', ['Iron Head', 'Protect']],
  ['Milotic', 'Competitive', ['Surf', 'Protect']],
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']],
  ['Falinks', 'Battle Armor', ['Smart Strike', 'Protect']],
  ['Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']],
  ['Incineroar', 'Intimidate', ['Flare Blitz', 'Protect']],
  ['Azumarill', 'Huge Power', ['Play Rough', 'Protect']],
  ['Sneasler', 'Poison Touch', ['Close Combat', 'Protect']],
];
const fillerTeam = (used = []) => FILLERS
  .filter(([species]) => !used.includes(species))
  .slice(0, Math.max(0, 6 - used.length))
  .map(([species, ability, moves]) => offensive(species, ability, moves));
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
      moves: p.moveSlots.map(m => ({id: ids.moves[m.id], pp: m.pp,
        disabled: Boolean(m.disabled), target: m.target}))};
  });
  const bench = side.pokemon.map((p, i) => [p, i])
    .filter(([p]) => !p.fainted && !side.active.includes(p)).map(([p]) => roster(p));
  return {kind, slots, bench, preview: []};
};

const compact = session => {
  const b = session.battle;
  return {turn: b.turn, rng_seed: b.prng.getSeed(),
    climate: {raw: b.field.weather, effective: b.field.effectiveWeather(), suppressed: b.field.suppressingWeather()},
    field: [...(b.field.weather ? [[ids.conditions[b.field.weather], b.field.weatherState.duration, b.field.weatherState.source.side.n]] : []),
      ...(b.field.terrain ? [[ids.conditions[b.field.terrain], b.field.terrainState.duration, b.field.terrainState.source.side.n]] : []),
      ...Object.entries(b.field.pseudoWeather).map(([id, effect]) => [ids.conditions[id], effect.duration, effect.source.side.n])].sort((a, c) => a[0] - c[0]),
    terminated: b.ended, winner: b.ended ? b.winner || null : null,
    sides: b.sides.map(s => ({
      request: b.ended ? 'Finished' : s.activeRequest?.wait || s.isChoiceDone() ? 'Wait' : s.requestState === 'teampreview' ? 'Preview' : s.requestState === 'switch' ? 'Replacement' : 'Normal',
      conditions: Object.entries(s.sideConditions).map(([id, state]) => [ids.conditions[id], state.duration ?? state.layers ?? 0]).sort((a, c) => a[0] - c[0]),
      pokemon: s.pokemon.map(p => ({roster: roster(p), species: ids.species[p.species.id], hp: p.hp, max_hp: p.maxhp, fainted: p.fainted,
        active_slot: s.active.indexOf(p) >= 0 ? s.active.indexOf(p) : null, ability_ending: Boolean(p.abilityState.ending), cached_speed: p.speed ?? null,
        status: ids.conditions[p.status] ?? 0, boosts: Object.values(p.boosts), stats: [p.maxhp, ...Object.values(p.storedStats)], ability: ids.abilities[p.ability],
        item: ids.items[p.item] ?? 0, types: p.types.map(t => ids.types[toID(t)] ?? 0), previous_item: ids.items[p.lastItem] ?? 0, can_mega: Boolean(p.canMegaEvo),
        pp: p.moveSlots.map(m => m.pp), volatiles: Object.keys(p.volatiles).sort(),
        hurt_this_turn: Boolean(p.hurtThisTurn), stats_raised_this_turn: Boolean(p.statsRaisedThisTurn),
        stats_lowered_this_turn: Boolean(p.statsLoweredThisTurn)})),
      request_detail: requestDetail(session, s)}))};
};

const select = (kind, own_slot, destination = 255) => ({
  kind, own_slot, move_slot: 255, target_location: 0, switch_destination: destination, resource: 'None',
});
const moveAction = (slot, moveSlot, target = 0) => ({
  kind: 'Move', own_slot: slot, move_slot: moveSlot, target_location: target, switch_destination: 255, resource: 'None',
});

function choose(session, sideIndex, wanted) {
  const b = session.battle, side = b.sides[sideIndex], req = side.activeRequest;
  if (side.requestState === 'teampreview') {
    return {actions: [0, 1, 4, 5].map((r, i) => select('Pick', i, r)), command: 'team 1256'};
  }
  const bench = side.pokemon.filter(p => !p.fainted && !side.active.includes(p));
  const actions = [], commands = [];
  const chosen = new Set();
  for (let slot = 0; slot < 2; slot++) {
    const p = side.active[slot];
    const info = req?.active?.[slot];
    if (side.requestState === 'switch') {
      if (!req.forceSwitch[slot]) { commands.push('pass'); continue; }
      const reserve = bench.find(x => !chosen.has(x));
      if (!reserve) { actions.push(select('Pass', slot)); commands.push('pass'); continue; }
      chosen.add(reserve);
      actions.push(select('Switch', slot, roster(reserve)));
      commands.push(`switch ${side.pokemon.indexOf(reserve) + 1}`);
      continue;
    }
    if (!p || p.fainted) { actions.push(select('Pass', slot)); commands.push('pass'); continue; }
    const view = p.moveSlots;
    const requestMoves = info?.moves && info.moves.length === view.length ? info.moves : null;
    const targetClass = index => requestMoves?.[index]?.target ?? dex.moves.get(view[index].id).target;
    const needsTarget = index => {
      const target = targetClass(index);
      return b.actions.targetTypeChoices(target) &&
        (target === 'normal' || target === 'any' || target === 'adjacentAllyOrSelf');
    };
    const targetFor = index => {
      if (!needsTarget(index)) return 0;
      const foes = sideIndex === 0 ? [1, 2, -1, -2] : [-1, -2, 1, 2];
      const target = targetClass(index);
      return foes.find(l => b.validTargetLoc(l, p, target) &&
        (() => { const other = p.getAtLoc(l); return other && other.side !== p.side; })()) ?? null;
    };
    const lastActive = p.isLastActive();
    const servedUsable = m => (!m.disabled || (m.disabled === 'hidden' && lastActive)) && m.pp > 0;
    const noMovesLeft = view.every(m => !servedUsable(m));
    if (noMovesLeft) {
      actions.push(moveAction(slot, 255, 0));
      // The reference refuses a target location for the Struggle pseudo-move.
      commands.push('move 1');
      continue;
    }
    const want = (wanted ?? [])[slot];
    const order = view
      .map((m, index) => ({m, index}))
      .filter(({m}) => servedUsable(m));
    const preferred = want
      ? order.filter(({m}) => m.id === want).concat(order.filter(({m}) => m.id !== want))
      : order.filter(({m}) => dex.moves.get(m.id).category !== 'Status')
        .concat(order.filter(({m}) => dex.moves.get(m.id).category === 'Status'));
    const chosenMove = preferred.find(({index}) => targetFor(index) !== null) ?? preferred[0];
    const index = chosenMove ? chosenMove.index : 0;
    const target = chosenMove ? targetFor(chosenMove.index) : 0;
    actions.push(moveAction(slot, index, target ?? 0));
    commands.push(`move ${index + 1}${target ? ` ${target}` : ''}`);
  }
  return {actions, command: commands.join(', ')};
}

const logHas = (session, pattern) => session.battle.log.some(line => pattern.test(line));

/// Damage dealt by the first landed `moveName` hit on `targetToken`, read from
/// the reference log (the previous absolute HP minus the post-hit HP). A move
/// that never connects returns null.
const logDamage = (session, moveName, targetToken) => {
  const hp = new Map();
  let currentMove = null;
  for (const line of session.battle.log) {
    const parts = line.split('|');
    const kind = parts[1];
    if (kind === 'switch' || kind === 'drag' || kind === 'replace') {
      const m = /^(\d+)\/(\d+)$/.exec(parts[4] ?? '');
      if (m) hp.set(parts[2], Number(m[1]));
      continue;
    }
    if (kind === '-damage' || kind === '-heal' || kind === '-sethp') {
      const m = /^(\d+)\/(\d+)/.exec(parts[3] ?? '');
      const value = m ? Number(m[1]) : null;
      if (currentMove === moveName && kind === '-damage' && parts[2] === targetToken
          && value !== null && hp.has(targetToken)) {
        const dealt = hp.get(targetToken) - value;
        hp.set(targetToken, value);
        return dealt;
      }
      if (value !== null) hp.set(parts[2], value);
      continue;
    }
    if (kind === 'move') {
      // Only damage inside this action's own window counts; a later spread
      // hit from another move must not be mistaken for this one.
      currentMove = parts[4] === '' || parts[4] === targetToken ? parts[3] : null;
      continue;
    }
    if (kind === 'faint') { hp.set(parts[2], 0); continue; }
    if (kind === 'turn' || kind === 'upkeep') currentMove = null;
  }
  return null;
};

const movesOf = (session, side) => session.battle.sides[side].activeRequest?.active?.[0]?.moves ?? [];
const moveDisabled = (session, side, moveId) =>
  movesOf(session, side).find(m => m.id === moveId)?.disabled ?? null;

const TRIALS = [
  {
    name: 'moves2_temperflare_doubles_after_failed_move',
    p1: [setOf('Gyarados', 'Intimidate', ['Temper Flare', 'Protect']), ...fillerTeam(['Gyarados'])],
    p2: [setOf('Metagross', 'Clear Body', ['Protect', 'Iron Head']), ...fillerTeam(['Metagross'])],
    seeds: [[3, 5, 7, 201], [11, 13, 17, 202], [23, 29, 31, 203]],
    script: [{p1: ['temperflare', 'protect'], p2: ['protect', 'protect']},
      {p1: ['temperflare', 'protect'], p2: ['ironhead', 'protect']}],
    control: [{p1: ['protect', 'protect'], p2: ['ironhead', 'protect']},
      {p1: ['temperflare', 'protect'], p2: ['ironhead', 'protect']}],
    coverage: {move: 'temperflare'},
    verify(fixture, session, extra) {
      const test = logDamage(session, 'Temper Flare', 'p2a: s0');
      const control = logDamage(extra.control.session, 'Temper Flare', 'p2a: s0');
      if (test === null || control === null) return 'Temper Flare never connected in test/control';
      if (test < control * 1.5) return `Temper Flare not doubled (test ${test}, control ${control})`;
      return null;
    },
  },
  {
    name: 'moves2_assurance_doubles_after_ally_hit',
    p1: [setOf('Raichu', 'Static', ['Quick Attack', 'Protect']),
      setOf('Kingambit', 'Defiant', ['Assurance', 'Protect']), ...fillerTeam(['Raichu', 'Kingambit'])],
    p2: [setOf('Metagross', 'Clear Body', ['Protect', 'Iron Head']), ...fillerTeam(['Metagross'])],
    seeds: [[3, 5, 7, 211], [11, 13, 17, 212], [23, 29, 31, 213]],
    script: [{p1: ['quickattack', 'assurance'], p2: ['ironhead', 'protect']},
      {p1: ['protect', 'protect'], p2: ['ironhead', 'protect']}],
    control: [{p1: ['protect', 'assurance'], p2: ['ironhead', 'protect']},
      {p1: ['protect', 'protect'], p2: ['ironhead', 'protect']}],
    coverage: {move: 'assurance'},
    verify(fixture, session, extra) {
      const test = logDamage(session, 'Assurance', 'p2a: s0');
      const control = logDamage(extra.control.session, 'Assurance', 'p2a: s0');
      if (test === null || control === null) return 'Assurance never connected in test/control';
      if (test < control * 1.5) return `Assurance not doubled (test ${test}, control ${control})`;
      return null;
    },
  },
  {
    name: 'moves2_lashout_doubles_after_speed_drop',
    p1: [setOf('Gyarados', 'Intimidate', ['Lash Out', 'Protect']), ...fillerTeam(['Gyarados'])],
    p2: [setOf('Ninetales-Alola', 'Snow Warning', ['Icy Wind', 'Freeze-Dry', 'Protect'],
      '', {hp: 2, atk: 0, def: 0, spa: 0, spd: 0, spe: 32}), ...fillerTeam(['Ninetales-Alola'])],
    seeds: [[3, 5, 7, 221], [11, 13, 17, 222], [23, 29, 31, 223]],
    script: [{p1: ['lashout', 'protect'], p2: ['icywind', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']}],
    // The control must not lower the user's stats but must still let Lash Out
    // connect: Freeze-Dry is a damaging move without stat changes.
    control: [{p1: ['lashout', 'protect'], p2: ['freezedry', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']}],
    coverage: {move: 'lashout'},
    verify(fixture, session, extra) {
      const test = logDamage(session, 'Lash Out', 'p2a: s0');
      const control = logDamage(extra.control.session, 'Lash Out', 'p2a: s0');
      if (test === null || control === null) return 'Lash Out never connected in test/control';
      if (test < control * 1.5) return `Lash Out not doubled (test ${test}, control ${control})`;
      if (!logHas(session, /\|-unboost\|p1a: s0\|spe/)) return 'Icy Wind never lowered the user Speed';
      return null;
    },
  },
  {
    name: 'moves2_gigatonhammer_request_disable',
    p1: [setOf('Tinkaton', 'Mold Breaker', ['Gigaton Hammer', 'Protect']), ...fillerTeam(['Tinkaton'])],
    p2: [setOf('Metagross', 'Clear Body', ['Protect', 'Iron Head']), ...fillerTeam(['Metagross'])],
    seeds: [[3, 5, 7, 231], [11, 13, 17, 232]],
    script: [{p1: ['gigatonhammer', null], p2: ['ironhead', null]},
      {p1: ['protect', null], p2: ['protect', null]},
      {p1: ['gigatonhammer', null], p2: ['protect', null]}],
    coverage: {move: 'gigatonhammer'},
    observe(session, side, turn, out) {
      if (side !== 0 || turn < 2) return;
      if (out.turn2 === undefined && turn === 2) out.turn2 = moveDisabled(session, 0, 'gigatonhammer');
      if (out.turn3 === undefined && turn === 3) out.turn3 = moveDisabled(session, 0, 'gigatonhammer');
    },
    verify(fixture, session, extra) {
      const {turn2, turn3} = extra.observations;
      if (turn2 !== true) return `Gigaton Hammer not disabled after use (got ${turn2})`;
      if (turn3 !== false) return `Gigaton Hammer still disabled after another move (got ${turn3})`;
      if (!logHas(session, /\|move\|p1a: s0\|Gigaton Hammer/)) return 'Gigaton Hammer never used';
      return null;
    },
  },
  {
    name: 'moves2_steelroller_clears_terrain_then_fails',
    p1: [setOf('Rillaboom', 'Grassy Surge', ['Protect', 'Grassy Glide']),
      setOf('Metagross', 'Clear Body', ['Steel Roller', 'Protect']), ...fillerTeam(['Rillaboom', 'Metagross'])],
    p2: [setOf('Milotic', 'Competitive', ['Protect', 'Surf']), ...fillerTeam(['Milotic'])],
    seeds: [[3, 5, 7, 241], [11, 13, 17, 242], [23, 29, 31, 243]],
    script: [{p1: ['protect', 'steelroller'], p2: ['surf', null]},
      {p1: ['protect', 'steelroller'], p2: ['protect', null]}],
    coverage: {move: 'steelroller'},
    observe(session, side, turn, out) {
      if (side === 0 && turn === 1) out.terrain_start = session.battle.field.terrain;
      if (side === 0 && turn === 2) out.terrain_after_hit = session.battle.field.terrain;
    },
    verify(fixture, session, extra) {
      const {terrain_start, terrain_after_hit} = extra.observations;
      if (terrain_start !== 'grassyterrain') return `terrain was not active at turn 1 (got ${terrain_start})`;
      if (terrain_after_hit !== '') return `terrain not cleared by Steel Roller (got ${terrain_after_hit})`;
      if (!logHas(session, /\|-fieldend\|.*Grassy Terrain/)) return 'no terrain end message';
      if (!logHas(session, /\|move\|p1b: s1\|Steel Roller\|\|\[still\]/)) {
        return 'Steel Roller never failed without a terrain';
      }
      return null;
    },
  },
  {
    name: 'moves2_steelroller_after_sub_damage_clears_terrain',
    p1: [setOf('Rillaboom', 'Grassy Surge', ['Protect', 'Grassy Glide']),
      setOf('Metagross', 'Clear Body', ['Steel Roller', 'Protect']), ...fillerTeam(['Rillaboom', 'Metagross'])],
    p2: [setOf('Gengar', 'Cursed Body', ['Substitute', 'Protect'],
      '', {hp: 2, atk: 0, def: 0, spa: 0, spd: 0, spe: 32}), ...fillerTeam(['Gengar'])],
    seeds: [[3, 5, 7, 251], [11, 13, 17, 252], [23, 29, 31, 253]],
    script: [{p1: ['protect', 'steelroller'], p2: ['substitute', null]},
      {p1: ['protect', 'protect'], p2: ['protect', null]}],
    coverage: {move: 'steelroller'},
    observe(session, side, turn, out) {
      if (side === 0 && turn === 2) out.terrain_after = session.battle.field.terrain;
    },
    verify(fixture, session, extra) {
      const {terrain_after} = extra.observations;
      if (!logHas(session, /\|move\|p2a: s0\|Substitute/)) return 'Substitute was never used';
      if (!logHas(session, /move: Substitute\|\[damage\]/)) {
        return 'the decoy never took the Steel Roller damage';
      }
      if (terrain_after !== '') return `terrain not cleared by onAfterSubDamage (got ${terrain_after})`;
      return null;
    },
  },
  {
    name: 'moves2_upperhand_flinch_and_non_priority_fail',
    p1: [setOf('Blaziken', 'Blaze', ['Upper Hand', 'Protect']), ...fillerTeam(['Blaziken'])],
    // Basculegion is Water/Ghost, which is immune to Upper Hand's Fighting
    // type; Feraligatr keeps the priority-move test on a hittable target.
    p2: [setOf('Feraligatr', 'Torrent', ['Aqua Jet', 'Waterfall', 'Protect']), ...fillerTeam(['Feraligatr'])],
    seeds: [[3, 5, 7, 261], [11, 13, 17, 262], [23, 29, 31, 263]],
    script: [{p1: ['upperhand', 'protect'], p2: ['aquajet', 'protect']},
      {p1: ['upperhand', 'protect'], p2: ['waterfall', 'protect']}],
    coverage: {move: 'upperhand'},
    verify(fixture, session, extra) {
      if (!logHas(session, /\|move\|p1a: s0\|Upper Hand\|p2a: s0/)) {
        return 'Upper Hand never connected against the priority move';
      }
      if (!logHas(session, /\|cant\|p2a: s0\|flinch/)) return 'the priority move was never flinched';
      if (logHas(session, /\|move\|p2a: s0\|Aqua Jet\|/)) return 'Aqua Jet still resolved after the flinch';
      if (!logHas(session, /\|move\|p1a: s0\|Upper Hand\|\|\[still\]/)) {
        return 'Upper Hand never failed against the non-priority move';
      }
      if (!logHas(session, /\|move\|p2a: s0\|Waterfall\|/)) return 'Waterfall never resolved on the fail turn';
      return null;
    },
  },
];

const runTrial = (trial, seed, script, teams) => {
  const [teamA, teamB] = teams;
  const session = new ReferenceSession({teams: [teamA, teamB], seed});
  const fixture = {name: `${trial.name}_${seed[3]}`, seed,
    teams: [teamA, teamB].map((team, side) => ({id: `${trial.name}-${side}`, members: team.map(s => ({
      species: ids.species[toID(s.species)], ability: ids.abilities[toID(s.ability)], item: ids.items[toID(s.item)] ?? 0,
      nature: ids.natures[toID(s.nature)], gender: s.gender || '', level: 50, moves: s.moves.map(x => ids.moves[toID(x)]),
      points: stats.map(k => s.evs[k]), ivs: stats.map(k => s.ivs[k])}))})),
    initial: compact(session), steps: []};
  const observations = {};
  const control = {session, observations};
  let failure = null;
  while (!session.battle.ended && fixture.steps.length < 300) {
    for (let side = 0; side < 2; side++) {
      if (session.battle.ended) break;
      const s = session.battle.sides[side];
      if (s.activeRequest?.wait || s.isChoiceDone()) continue;
      const turn = session.battle.turn;
      trial.observe?.(session, side, turn, observations);
      const scripted = script[turn - 1];
      const wanted = scripted ? (side === 0 ? scripted.p1 : scripted.p2) : null;
      const choice = choose(session, side, wanted);
      const result = session.choose(side ? 'p2' : 'p1', choice.command);
      if (!result.accepted) { failure = JSON.stringify({side, turn, wanted, choice, err: s.choice.error}); break; }
      fixture.steps.push({side: side ? 'P2' : 'P1', ...choice, expected: compact(session)});
    }
    if (failure) break;
  }
  if (failure) { session.destroy(); return {failure}; }
  if (!session.battle.ended) { session.destroy(); return {failure: 'did not complete'}; }
  if (process.env.PA3_DBG) {
    console.log('LOG', fixture.name, JSON.stringify(session.battle.log.filter(l =>
      /\|(-damage|-heal|move|cant|activate|fail|still|unboost|boost|fieldstart|fieldend)\|/.test(l))));
  }
  return {session, fixture, observations};
};

const fixtures = [];
const skipped = [];
for (const trial of TRIALS) {
  let recorded = null;
  let lastReason = null;
  for (const seed of trial.seeds) {
    const teamA = nameSets(trial.p1);
    const teamB = nameSets(trial.p2);
    const problems = validator.validateTeam(teamA) || validator.validateTeam(teamB);
    if (problems) { lastReason = problems.join('; '); break; }
    const primary = runTrial(trial, seed, trial.script, [teamA, teamB]);
    if (primary.failure) { lastReason = primary.failure; continue; }
    let control = null;
    if (trial.control) {
      // Reuse the validator-normalized sets (they carry the default IVs).
      control = runTrial(trial, seed, trial.control, [teamA, teamB]);
      if (control.failure) { primary.session.destroy(); lastReason = `control: ${control.failure}`; continue; }
    }
    const reason = trial.verify(primary.fixture, primary.session, {
      control, observations: primary.observations,
    });
    if (reason) {
      primary.session.destroy();
      control?.session.destroy();
      lastReason = reason;
      continue;
    }
    primary.fixture.coverage = trial.coverage ?? {};
    if (control) {
      control.fixture.name = `${control.fixture.name}_control`;
      control.fixture.coverage = trial.coverage ?? {};
    }
    recorded = {primary: primary.fixture, control: control?.fixture, primarySession: primary.session, controlSession: control?.session};
    break;
  }
  if (recorded) {
    fixtures.push(recorded.primary);
    if (recorded.control) fixtures.push(recorded.control);
    recorded.primarySession.destroy();
    recorded.controlSession?.destroy();
  } else {
    skipped.push({name: trial.name, reason: lastReason ?? 'no seed produced the required behavior'});
  }
}
fs.writeFileSync(new URL('../data/more_moves2.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify({trials: TRIALS.length, fixtures: fixtures.length, skipped: skipped.length}));
for (const s of skipped) console.log('SKIP', JSON.stringify(s));
