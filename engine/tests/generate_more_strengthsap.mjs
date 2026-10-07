// Development-only interaction corpus for Strength Sap.
//
// Four witnesses, all scripted against the pinned reference:
//   1. heal by the target's stage-boosted Attack and the -1 Attack drop;
//   2. user at full HP still drops the target (the move succeeds);
//   3. Clear Body refuses the drop with no heal available -> the move fails;
//   4. a target already at -6 Attack makes the move fail outright.
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
const moveAction = (slot, moveSlot, target = 0) =>
  ({kind: 'Move', own_slot: slot, move_slot: moveSlot, target_location: target, switch_destination: 255, resource: 'None'});

const FILLERS = [
  ['Goodra-Hisui', 'Shell Armor', ['Dragon Pulse', 'Protect']],
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']],
  ['Metagross', 'Clear Body', ['Iron Head', 'Protect']],
  ['Milotic', 'Competitive', ['Surf', 'Protect']],
  ['Scolipede', 'Swarm', ['X-Scissor', 'Protect']],
  ['Starmie', 'Natural Cure', ['Ice Beam', 'Protect']],
  ['Blaziken', 'Speed Boost', ['Close Combat', 'Protect']],
].map(([species, ability, moves]) => setOf(species, ability, moves, ''));

const SINISTCHA = setOf('Sinistcha', 'Heatproof', ['Strength Sap', 'Shadow Ball', 'Protect'], '');
const TRIALS = [
  {
    name: 'strengthsap_heal_and_drop',
    foe: setOf('Torterra', 'Shell Armor', ['Bulldoze', 'Protect'], ''),
    // Turn 1 trades hits (user damaged); turn 2 saps; later turns attack.
    script: (side, slot, turn) => {
      if (side === 0 && slot === 0) return turn === 2 ? 'strengthsap' : 'shadowball';
      if (side === 1 && slot === 1) return 'protect';
      return null;
    },
    check: (steps, logs) => {
      const used = logs.some(l => l.startsWith('|move|p1a:') && l.split('|')[3] === 'Strength Sap');
      if (!used) return 'Strength Sap never fired';
      const healed = steps.some((s, i) => i > 0
        && mon(s, 0, 0).hp > mon(steps[i - 1], 0, 0).hp && !mon(s, 0, 0).fainted);
      if (!healed) return 'no heal boundary observed';
      const dropped = steps.some(s => mon(s, 1, 0).boosts[0] === -1);
      if (!dropped) return 'target never dropped to -1 Attack';
      return null;
    },
  },
  {
    name: 'strengthsap_full_hp_drops',
    foe: setOf('Torterra', 'Shell Armor', ['Iron Defense', 'Protect'], ''),
    // The foe never damages: the faster user saps at full HP on turn 1 and
    // the drop still marks the move as successful.
    script: (side, slot, turn) => {
      if (side === 0 && slot === 0) return turn === 1 ? 'strengthsap' : 'shadowball';
      if (side === 1 && slot === 0) return 'irondefense';
      if (side === 1 && slot === 1) return 'protect';
      return null;
    },
    check: (steps, logs) => {
      const used = logs.some(l => l.startsWith('|move|p1a:') && l.split('|')[3] === 'Strength Sap');
      if (!used) return 'Strength Sap never fired';
      const fullAtUse = steps.some(s => mon(s, 0, 0).hp === mon(s, 0, 0).max_hp && mon(s, 1, 0).boosts[0] === -1);
      if (!fullAtUse) return 'never saw a full-HP user alongside the -1 drop';
      return null;
    },
  },
  {
    name: 'strengthsap_clearbody_full_hp_fails',
    foe: setOf('Metagross', 'Clear Body', ['Iron Defense', 'Protect'], ''),
    // The foe never damages: the user stays at full HP, Clear Body refuses
    // the drop and the heal restores nothing, so the move must fail.
    script: (side, slot, turn) => {
      if (side === 0 && slot === 0) return turn === 1 ? 'strengthsap' : 'shadowball';
      if (side === 1 && slot === 0) return 'irondefense';
      if (side === 1 && slot === 1) return 'protect';
      return null;
    },
    check: (steps, logs) => {
      // A failed hit logs the move line with `[still]` and then `|-fail|p1a: s0`
      // without the move name.
      const failed = logs.some(l => l.startsWith('|move|p1a:')
        && l.split('|')[3] === 'Strength Sap' && l.includes('[still]'));
      if (!failed) return 'no failing Strength Sap boundary observed';
      const untouched = steps.every(s => mon(s, 1, 0).boosts[0] === 0);
      if (!untouched) return 'Clear Body let a drop through';
      return null;
    },
  },
  {
    name: 'strengthsap_at_minus_six_fails',
    foe: setOf('Torterra', 'Shell Armor', ['Iron Defense', 'Protect'], ''),
    // Sap every turn: six drops land, the seventh use must fail outright.
    script: (side, slot, turn) => {
      if (side === 0 && slot === 0) return turn <= 7 ? 'strengthsap' : 'shadowball';
      // Keep the ally from KOing the target before the seventh attempt.
      if (side === 0 && slot === 1) return 'protect';
      if (side === 1 && slot === 0) return 'irondefense';
      if (side === 1 && slot === 1) return 'protect';
      return null;
    },
    check: (steps, logs) => {
      const uses = logs.filter(l => l.startsWith('|move|p1a:') && l.split('|')[3] === 'Strength Sap').length;
      const failed = logs.some(l => l.startsWith('|move|p1a:')
        && l.split('|')[3] === 'Strength Sap' && l.includes('[still]'));
      if (uses < 7 || !failed) return `uses=${uses} failed=${failed}`;
      const floored = steps.some(s => mon(s, 1, 0).boosts[0] === -6);
      if (!floored) return 'target never reached -6 Attack';
      return null;
    },
  },
];

const mon = (step, side, rosterId) => step.expected.sides[side].pokemon.find(p => p.roster === rosterId);

function choose(session, sideIndex, trial) {
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
      else { actions.push(select('Pass', slot)); commands.push('pass'); }
      continue;
    }
    if (p.fainted) { actions.push(select('Pass', slot)); commands.push('pass'); continue; }
    const wanted = trial.script(sideIndex, slot, b.turn);
    let slotIndex = wanted === null ? -1 : p.moveSlots.findIndex(m => m.id === wanted && !m.disabled && m.pp > 0);
    if (slotIndex < 0) {
      slotIndex = p.moveSlots.findIndex(m => !m.disabled && m.pp > 0 && dex.moves.get(m.id).basePower > 0);
    }
    if (slotIndex < 0) slotIndex = p.moveSlots.findIndex(m => !m.disabled && m.pp > 0);
    // All PP exhausted: the request serves the Struggle pseudo-move, which
    // the native encodes as move_slot NO_SLOT.
    const struggle = slotIndex < 0;
    if (struggle) slotIndex = 0;
    const chosen = struggle ? {target: 'randomNormal'} : p.moveSlots[slotIndex];
    const candidates = sideIndex === 0 ? [1, 2, -1, -2, 0] : [-1, -2, 1, 2, 0];
    const location = chosen && b.actions.targetTypeChoices(chosen.target)
      ? candidates.find(loc => b.validTargetLoc(loc, p, chosen.target)) ?? 0 : 0;
    actions.push(moveAction(slot, struggle ? 255 : slotIndex, location));
    commands.push(`move ${slotIndex + 1}${location ? ` ${location}` : ''}`);
  }
  return {actions, command: commands.join(', ')};
}

const fixtures = [];
const skipped = [];
let seedIndex = 0;
for (const trial of TRIALS) {
  const base = dex.species.get(SINISTCHA.species).baseSpecies;
  const foeBase = dex.species.get(trial.foe.species).baseSpecies;
  const extras = FILLERS.filter(s => ![base, foeBase].includes(dex.species.get(s.species).baseSpecies));
  const teamA = nameSets([SINISTCHA, ...extras].slice(0, 6));
  const teamB = nameSets([trial.foe, ...FILLERS.filter(s => ![base, foeBase].includes(dex.species.get(s.species).baseSpecies))].slice(0, 6));
  const problems = validator.validateTeam(teamA) || validator.validateTeam(teamB);
  if (problems) { skipped.push({name: trial.name, reason: problems.join('; ')}); continue; }
  let recorded = null, lastReason = null;
  for (let attempt = 0; attempt < 64 && !recorded; attempt++) {
    const seed = [2026, 10, 7, 4300 + seedIndex * 64 + attempt];
    const session = new ReferenceSession({teams: [teamA, teamB], seed});
    const fixture = {name: `move_${trial.name}_${seed[3]}`, seed,
      teams: [teamA, teamB].map((team, side) => ({id: `sap-${trial.name}-${side}`, members: team.map(s => ({
        species: ids.species[toID(s.species)], ability: ids.abilities[toID(s.ability)], item: ids.items[toID(s.item)] ?? 0,
        nature: ids.natures[toID(s.nature)], gender: s.gender || '', level: 50, moves: s.moves.map(x => ids.moves[toID(x)]),
        points: stats.map(k => s.evs[k]), ivs: stats.map(k => s.ivs[k])}))})),
      initial: compact(session), steps: []};
    const logs = [];
    let failure = null;
    while (!session.battle.ended && fixture.steps.length < 200) {
      for (let side = 0; side < 2; side++) {
        if (session.battle.ended) break;
        const s = session.battle.sides[side];
        if (s.activeRequest?.wait || s.isChoiceDone()) continue;
        const choice = choose(session, side, trial);
        const before = session.battle.log.length;
        const result = session.choose(side ? 'p2' : 'p1', choice.command);
        if (!result.accepted) { failure = JSON.stringify({side, choice, messages: result.messages}); break; }
        logs.push(...session.battle.log.slice(before));
        fixture.steps.push({side: side ? 'P2' : 'P1', ...choice, expected: compact(session)});
      }
      if (failure) break;
    }
    if (failure) { session.destroy(); lastReason = failure; continue; }
    if (!session.battle.ended) { session.destroy(); lastReason = 'did not complete'; continue; }
    const reason = trial.check(fixture.steps, logs);
    if (reason) {
      if (process.env.SAP_DEBUG) {
        console.log(`--- ${trial.name} attempt seed ${seed[3]}: ${reason}`);
        console.log(logs.filter(l => l.includes('Strength Sap') || l.includes('-fail') || l.includes('|cant|')).join('\n'));
      }
      session.destroy();
      lastReason = reason;
      continue;
    }
    fixture.coverage = {move: 'strengthsap'};
    recorded = fixture;
    session.destroy();
  }
  if (recorded) fixtures.push(recorded);
  else skipped.push({name: trial.name, reason: lastReason ?? 'no seed produced the required behavior'});
  seedIndex++;
}

fs.writeFileSync(new URL('../data/more_strengthsap.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify({fixtures: fixtures.length, skipped: skipped.length}));
if (skipped.length) console.log(JSON.stringify(skipped));
