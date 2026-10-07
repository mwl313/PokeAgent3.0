// Development-only corpus for the held item Metronome.
//
// Three scripted witnesses against the pinned reference:
//   1. consecutive Iron Heads ramp 4096 -> 4915 -> 5734 -> 6553 (20% a use);
//   2. a different move in between resets the counter;
//   3. Knock Off removes the item and the counter volatile on the next try.
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

const HOLDER = setOf('Excadrill', 'Sand Rush', ['Iron Head', 'Rock Slide', 'Protect'], 'Metronome');
const RAIN_FOE = setOf('Milotic', 'Competitive', ['Recover', 'Rain Dance', 'Protect'], '');
const KNOCK_FOE = setOf('Chesnaught', 'Overgrow', ['Knock Off', 'Rain Dance', 'Protect'], '');
// High-PP self-move walls: the scripted battles never exhaust PP, so no
// Protect-stall fallback or Struggle can perturb the scene.
const AGILITY_ALLY = setOf('Metagross', 'Clear Body', ['Agility', 'Protect'], '');
const AGILITY_WALLS = [
  setOf('Dragonite', 'Inner Focus', ['Agility', 'Protect'], ''),
  setOf('Jolteon', 'Volt Absorb', ['Agility', 'Protect'], ''),
  setOf('Talonflame', 'Flame Body', ['Agility', 'Protect'], ''),
  setOf('Scizor', 'Swarm', ['Agility', 'Protect'], ''),
  setOf('Lucario', 'Inner Focus', ['Agility', 'Protect'], ''),
];
// The ramp target survives four unboosted Iron Heads (Steel is resisted).
const RAMP_FOE = setOf('Metagross', 'Clear Body', ['Agility', 'Protect'], '');
const mon = (step, side, rosterId) => step.expected.sides[side].pokemon.find(p => p.roster === rosterId);

const TRIALS = [
  {name: 'control_no_item', foe: RAIN_FOE,
    holder: setOf('Excadrill', 'Sand Rush', ['Iron Head', 'Rock Slide', 'Protect'], ''),
    ally: null, wall_mode: 'fillers',
    script: (side, slot, turn, battle) => {
      if (side === 0 && slot === 0) return turn <= 4 ? 'ironhead' : 'rockslide';
      if (side === 0 && slot === 1) return 'protect';
      if (side === 1 && slot === 0) return 'raindance';
      if (side === 1 && slot === 1) return 'protect';
      return null;
    },
    check: (steps, logs) => {
      const hits = ironHeadDamages(logs);
      return hits.length >= 4 ? null : `control only ${hits.length} hits`;
    }},
  {name: 'metronome_ramp', foe: RAMP_FOE, holder: HOLDER, ally: AGILITY_ALLY, wall_mode: 'agility',
    script: (side, slot, turn, battle) => {
      if (side === 0 && slot === 0) return turn <= 4 ? 'ironhead' : 'rockslide';
      if (side === 0 && slot === 1) return 'agility';
      if (side === 1) return 'agility';
      return null;
    },
    check: (steps, logs) => {
      const hits = ironHeadDamages(logs);
      if (hits.length < 4) return `only ${hits.length} Iron Head hits`;
      if (hits.slice(0, 4).some(h => h.crit)) return 'crit broke the ramp scene';
      const [d1, d2, d3, d4] = hits.map(h => h.damage);
      if (!(d2 > d1 && d3 > d1 && d4 > d2)) return `damages not ramping: ${d1},${d2},${d3},${d4}`;
      return null;
    }},
  {name: 'metronome_reset_on_different_move', foe: RAMP_FOE, holder: HOLDER, ally: AGILITY_ALLY, wall_mode: 'agility',
    script: (side, slot, turn, battle) => {
      if (side === 0 && slot === 0) {
        // IH, IH, Rock Slide, IH: the interleaved move resets the counter, so
        // the third Iron Head lands unboosted again.
        return [1, 2, 4].includes(turn) ? 'ironhead' : 'rockslide';
      }
      if (side === 0 && slot === 1) return 'agility';
      if (side === 1) return 'agility';
      return null;
    },
    check: (steps, logs) => {
      const hits = ironHeadDamages(logs);
      if (hits.length < 3) return `only ${hits.length} Iron Head hits`;
      if (hits.slice(0, 3).some(h => h.crit)) return 'crit broke the reset scene';
      const [d1, d2, d3] = hits.map(h => h.damage);
      if (!(d2 > d1 && d3 < d2)) return `reset not visible: ${d1},${d2},${d3}`;
      return null;
    }},
  {name: 'metronome_item_lost', foe: KNOCK_FOE, holder: HOLDER, ally: AGILITY_ALLY, wall_mode: 'agility',
    script: (side, slot, turn) => {
      if (side === 0 && slot === 0) return turn <= 6 ? 'ironhead' : 'rockslide';
      if (side === 0 && slot === 1) return 'agility';
      if (side === 1 && slot === 0) return turn === 1 ? 'knockoff' : 'agility';
      if (side === 1 && slot === 1) return 'agility';
      return null;
    },
    check: (steps, logs) => {
      const withVol = steps.some(s => mon(s, 0, 0).volatiles.includes('metronome'));
      const without = steps.some(s => !mon(s, 0, 0).volatiles.includes('metronome')
        && mon(s, 0, 0).item === 0 && !mon(s, 0, 0).fainted);
      if (!withVol) return 'metronome volatile never observed';
      if (!without) return 'volatile/item did not clear after Knock Off';
      return null;
    }},
];

// Iron Head damage extraction: the log's `|-damage|p2a: s0|hp/max` lines give
// the remaining HP after each hit, and the denominator gives the starting HP.
function ironHeadDamages(logs) {
  const hits = [];
  let pending = null;
  for (const line of logs) {
    const parts = line.split('|');
    if (line.startsWith('|move|p1a:') && parts[3] === 'Iron Head') { pending = {crit: false}; continue; }
    if (pending && !pending.crit && line === '|crit|p2a: s0') { pending.crit = true; continue; }
    if (pending && line.startsWith('|-damage|p2a: s0|')) {
      const [hp, max] = parts[3].split('/').map(Number);
      pending.max = max;
      pending.remaining = hp;
      hits.push(pending);
      pending = null;
    }
  }
  return hits.map((hit, i) => ({
    crit: hit.crit,
    damage: (i === 0 ? hit.max : hits[i - 1].remaining) - hit.remaining,
  }));
}

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
    const wanted = trial.script(sideIndex, slot, b.turn, b);
    let slotIndex = wanted === null ? -1 : p.moveSlots.findIndex(m => m.id === wanted && !m.disabled && m.pp > 0);
    if (slotIndex < 0) {
      slotIndex = p.moveSlots.findIndex(m => !m.disabled && m.pp > 0 && dex.moves.get(m.id).basePower > 0);
    }
    if (slotIndex < 0) slotIndex = p.moveSlots.findIndex(m => !m.disabled && m.pp > 0);
    const struggle = slotIndex < 0;
    if (struggle) slotIndex = 0;
    const chosen = struggle ? {target: 'randomNormal'} : p.moveSlots[slotIndex];
    // Resolve target locations against the reference's own resolver: the
    // sign of a location is relative to the user's field position, so a fixed
    // candidate order silently aims at the ally for one of the sides.
    const location = targetLocation(b, p, chosen);
    actions.push(moveAction(slot, struggle ? 255 : slotIndex, location));
    commands.push(`move ${slotIndex + 1}${location ? ` ${location}` : ''}`);
  }
  return {actions, command: commands.join(', ')};
}

function targetLocation(b, p, chosen) {
  if (!chosen || !b.actions.targetTypeChoices(chosen.target)) return 0;
  const candidates = [1, 2, -1, -2, 0];
  for (const loc of candidates) {
    if (!b.validTargetLoc(loc, p, chosen.target)) continue;
    if (chosen.id === undefined) return loc;
    // Pure position lookup. `Battle.getTarget` falls back to
    // `Battle.getRandomTarget` when a location is empty, which consumes PRNG
    // draws during *planning* and would bake a seed the replay can never
    // reproduce. The sim still resolves an empty location with a random
    // target inside its own action handling, where the fixture captures it.
    const target = p.getAtLoc(loc);
    if (target && target.side !== p.side) return loc;
  }
  return candidates.find(loc => b.validTargetLoc(loc, p, chosen.target)) ?? 0;
}

const fixtures = [];
const skipped = [];
let seedIndex = 0;
for (const trial of TRIALS) {
  let teamA, teamB;
  if (trial.wall_mode === 'agility') {
    teamA = nameSets([trial.holder, trial.ally, ...AGILITY_WALLS.slice(0, 4)].slice(0, 6));
    teamB = nameSets([trial.foe, ...AGILITY_WALLS].slice(0, 6));
  } else {
    const base = dex.species.get(trial.holder.species).baseSpecies;
    const foeBase = dex.species.get(trial.foe.species).baseSpecies;
    const extras = FILLERS.filter(s => ![base, foeBase].includes(dex.species.get(s.species).baseSpecies));
    teamA = nameSets([trial.holder, ...extras].slice(0, 6));
    teamB = nameSets([trial.foe, ...FILLERS.filter(s => ![base, foeBase].includes(dex.species.get(s.species).baseSpecies))].slice(0, 6));
  }
  const problems = validator.validateTeam(teamA) || validator.validateTeam(teamB);
  if (problems) { skipped.push({name: trial.name, reason: problems.join('; ')}); continue; }
  let recorded = null, lastReason = null;
  for (let attempt = 0; attempt < 64 && !recorded; attempt++) {
    const seed = [2026, 10, 7, 4400 + seedIndex * 64 + attempt];
    const session = new ReferenceSession({teams: [teamA, teamB], seed});
    const prngProbe = process.env.METRO_LOGS ? (() => {
      const prng = session.battle.prng;
      let draws = 0;
      const original = prng.next.bind(prng);
      prng.next = (...args) => { draws++; return original(...args); };
      return () => draws;
    })() : null;
    const fixture = {name: `item_${trial.name}_${seed[3]}`, seed,
      teams: [teamA, teamB].map((team, side) => ({id: `metro-${trial.name}-${side}`, members: team.map(s => ({
        species: ids.species[toID(s.species)], ability: ids.abilities[toID(s.ability)], item: ids.items[toID(s.item)] ?? 0,
        nature: ids.natures[toID(s.nature)], gender: s.gender || '', level: 50, moves: s.moves.map(x => ids.moves[toID(x)]),
        points: stats.map(k => s.evs[k]), ivs: stats.map(k => s.ivs[k])}))})),
      initial: compact(session), steps: []};
    const logs = [];
    let failure = null;
    while (!session.battle.ended && fixture.steps.length < 300) {
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
        logs.push(`===STEP ${fixture.steps.length - 1} ${side ? 'P2' : 'P1'} ${choice.command} draws=${prngProbe ? prngProbe() : '?'}`);
      }
      if (failure) break;
    }
    if (failure) { session.destroy(); lastReason = failure; continue; }
    if (!session.battle.ended) { session.destroy(); lastReason = 'did not complete'; continue; }
    const reason = trial.check(fixture.steps, logs);
    if (reason) {
      if (process.env.METRO_DEBUG) {
        console.log(`--- ${trial.name} seed ${seed[3]}: ${reason}`);
        if (trial.name === 'metronome_item_lost') {
          console.log(logs.filter(l => l.includes('|move|') || l.includes('|item|') || l.includes('-enditem')).slice(0, 30).join('\n'));
        }
        for (const s of fixture.steps) {
          const p = mon(s, 0, 0);
          console.log('turn', s.expected.turn, s.side, 'item', p.item, 'vol', JSON.stringify(p.volatiles), 'hp', p.hp);
        }
      }
      session.destroy();
      lastReason = reason;
      continue;
    }
    fixture.coverage = {item: 'metronome'};
    if (process.env.METRO_LOGS) {
      fs.writeFileSync(`/tmp/ref_${fixture.name}.log`, logs.join('\n') + '\n');
    }
    recorded = fixture;
    session.destroy();
  }
  if (recorded) fixtures.push(recorded);
  else skipped.push({name: trial.name, reason: lastReason ?? 'no seed produced the required behavior'});
  seedIndex++;
}

fs.writeFileSync(new URL('../data/more_metronome_item.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify({fixtures: fixtures.length, skipped: skipped.length}));
if (skipped.length) console.log(JSON.stringify(skipped));
