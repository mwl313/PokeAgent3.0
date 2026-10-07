// Development-only item-interaction corpus. One complete legal reference
// battle per item family the native engine claims to execute, so item
// removal / consumption semantics have differential witnesses. Fixtures are
// merged into turn-fixtures.json by scripts/export_engine_data.mjs and are
// never used by training.
import fs from 'node:fs';
import {createRequire} from 'node:module';
import {ReferenceSession, verifyReference, FORMAT, ORACLE_COMMIT} from '../reference.mjs';
const require = createRequire(import.meta.url);
const {TeamValidator, toID} = require('../../vendor/pokemon-showdown/dist/sim');
const validator = new TeamValidator(FORMAT);
const dex = validator.dex;
const data = JSON.parse(fs.readFileSync(new URL('../data/dex.json', import.meta.url), 'utf8'));
const ids = Object.fromEntries(Object.entries(data.tables).map(([k, rows]) => [k, Object.fromEntries(rows.map(r => [r.id, r.numeric_id]))]));
const stats = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
verifyReference();

const nameSets = team => team.map((set, i) => ({...set, name: `s${i}`}));
const setOf = (species, ability, moves, item = '', points = {hp: 24, atk: 8, def: 8, spa: 8, spd: 8, spe: 4}) =>
  ({name: species, species, ability, item, nature: 'Serious', level: 50, gender: 'M', moves,
    evs: {hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0, ...points}});

// Fillers never share a base species with the probed holders below.
const FILLERS = [
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect'], ''],
  ['Samurott', 'Shell Armor', ['Aqua Jet', 'Protect'], ''],
  ['Hydreigon', 'Levitate', ['Dragon Pulse', 'Protect'], ''],
  ['Falinks', 'Battle Armor', ['Smart Strike', 'Protect'], ''],
  ['Metagross', 'Clear Body', ['Iron Head', 'Protect'], ''],
  ['Milotic', 'Competitive', ['Surf', 'Protect'], ''],
  ['Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect'], ''],
  ['Starmie', 'Natural Cure', ['Ice Beam', 'Protect'], ''],
];
const fillerTeam = () => FILLERS.map(([species, ability, moves, item]) => setOf(species, ability, moves, item));
const roster = p => Number(p.name.slice(-1));

// Privileged request detail so the native legal-action mask can be compared
// with the reference request at every boundary (development fixtures only).
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
    const wanted = sideIndex === 0 && slot === 0 ? plan.moveSlot : plan.attackSlot;
    const slotIndex = p.moveSlots.findIndex(m => m.id === wanted && !m.disabled && m.pp > 0);
    const fallback = p.moveSlots.findIndex(m => !m.disabled && m.pp > 0);
    const choice = slotIndex >= 0 ? slotIndex : fallback >= 0 ? fallback : 0;
    const chosen = p.moveSlots[choice];
    // Prefer the foe directly across so the probed holder is the target.
    const candidates = sideIndex === 0 ? [1, 2, -1, -2] : [-1, -2, 1, 2];
    const foeLocation = chosen && chosen.pp > 0 && b.actions.targetTypeChoices(chosen.target)
      ? candidates.find(loc => {
        if (!b.validTargetLoc(loc, p, chosen.target)) return false;
        // `battle.getTarget` falls back to a random target and consumes RNG;
        // the position lookup is the pure mapping and keeps fixture RNG exact.
        const other = p.getAtLoc(loc);
        return other && other.side !== p.side;
      })
      : undefined;
    // A zero-PP choice becomes Struggle in the reference and takes no target.
    const location = foeLocation ?? 0;
    actions.push(move(slot, choice, location));
    commands.push(`move ${choice + 1}${location ? ` ${location}` : ''}`);
  }
  return {actions, command: commands.join(', ')};
}

// Each trial: an item-removal attacker against a specific held item.
const TRIALS = [
  {name: 'item_knockoff_leftovers',
    move: 'knockoff',
    attacker: setOf('Beedrill', 'Swarm', ['Knock Off', 'Protect', 'X-Scissor', 'Poison Jab']),
    target: setOf('Goodra-Hisui', 'Shell Armor', ['Dragon Pulse', 'Protect'], 'Leftovers')},
  {name: 'item_knockoff_sitrus',
    move: 'knockoff',
    attacker: setOf('Beedrill', 'Swarm', ['Knock Off', 'Protect', 'X-Scissor', 'Poison Jab']),
    target: setOf('Goodra-Hisui', 'Shell Armor', ['Dragon Pulse', 'Protect'], 'Sitrus Berry')},
  {name: 'item_knockoff_megastone',
    move: 'knockoff',
    attacker: setOf('Beedrill', 'Swarm', ['Knock Off', 'Protect', 'X-Scissor', 'Poison Jab']),
    target: setOf('Charizard', 'Drought', ['Flamethrower', 'Protect'], 'Charizardite Y')},
  {name: 'item_knockoff_choicescarf',
    move: 'knockoff',
    attacker: setOf('Beedrill', 'Swarm', ['Knock Off', 'Protect', 'X-Scissor', 'Poison Jab']),
    target: setOf('Perrserker', 'Battle Armor', ['Iron Head', 'Protect'], 'Choice Scarf')},
  {name: 'item_knockoff_noitem',
    move: 'knockoff',
    attacker: setOf('Beedrill', 'Swarm', ['Knock Off', 'Protect', 'X-Scissor', 'Poison Jab']),
    target: setOf('Goodra-Hisui', 'Shell Armor', ['Dragon Pulse', 'Protect'])},
  {name: 'item_trick_swap',
    move: 'trick',
    attacker: setOf('Chimecho', 'Levitate', ['Trick', 'Protect', 'Psychic', 'Dazzling Gleam'], 'Leftovers'),
    target: setOf('Goodra-Hisui', 'Shell Armor', ['Dragon Pulse', 'Protect'], 'Sitrus Berry')},
  {name: 'item_trick_refused_stone',
    move: 'trick',
    attacker: setOf('Chimecho', 'Levitate', ['Trick', 'Protect', 'Psychic', 'Dazzling Gleam'], 'Leftovers'),
    target: setOf('Charizard', 'Drought', ['Flamethrower', 'Protect'], 'Charizardite Y')},
  {name: 'item_trick_give_when_empty',
    move: 'trick',
    attacker: setOf('Chimecho', 'Levitate', ['Trick', 'Protect', 'Psychic', 'Dazzling Gleam'], 'Leftovers'),
    target: setOf('Goodra-Hisui', 'Shell Armor', ['Dragon Pulse', 'Protect'])},
  {name: 'item_trick_both_empty',
    move: 'trick',
    attacker: setOf('Chimecho', 'Levitate', ['Trick', 'Protect', 'Psychic', 'Dazzling Gleam']),
    target: setOf('Goodra-Hisui', 'Shell Armor', ['Dragon Pulse', 'Protect'])},
  {name: 'item_trick_stone_to_other',
    move: 'trick',
    attacker: setOf('Chimecho', 'Levitate', ['Trick', 'Protect', 'Psychic', 'Dazzling Gleam'], 'Charizardite Y'),
    target: setOf('Goodra-Hisui', 'Shell Armor', ['Dragon Pulse', 'Protect'], 'Leftovers')},
];

const fixtures = [];
const skipped = [];
for (const trial of TRIALS) {
  const teamA = nameSets([trial.attacker, ...fillerTeam().slice(0, 5)]);
  const teamB = nameSets([trial.target, ...fillerTeam().slice(0, 5)]);
  const problems = validator.validateTeam(teamA) || validator.validateTeam(teamB);
  if (problems) { skipped.push({name: trial.name, reason: problems.join('; ')}); continue; }
  const seed = [2026, 10, 7, 4000 + fixtures.length];
  const session = new ReferenceSession({teams: [teamA, teamB], seed});
  const plan = {moveSlot: ids.moves[trial.move], attackSlot: ids.moves.ironhead};
  const fixture = {name: `${trial.name}_${seed[3]}`, seed,
    teams: [teamA, teamB].map((team, side) => ({id: `item-${trial.name}-${side}`, members: team.map(s => ({
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
  if (failure) { session.destroy(); skipped.push({name: trial.name, reason: failure}); continue; }
  if (!session.battle.ended) { session.destroy(); skipped.push({name: trial.name, reason: 'did not complete'}); continue; }
  const log = session.battle.log;
  const moveName = dex.moves.get(trial.move).name;
  const used = log.some(line => line.startsWith('|move|p1a:') && line.split('|')[3] === moveName);
  const removed = log.some(line => line.includes('[from] move: Knock Off'));
  if (!used) {
    session.destroy();
    skipped.push({name: trial.name, reason: 'probe failed: move never fired'});
    continue;
  }
  if (trial.move === 'knockoff') {
    const expectsRemoval = !trial.name.includes('megastone') && !trial.name.includes('noitem');
    if (removed !== expectsRemoval) {
      session.destroy();
      skipped.push({name: trial.name, reason: `probe failed: removed=${removed} expected=${expectsRemoval}`});
      continue;
    }
    fixture.coverage = {item: toID(trial.target.item) || 'none', move: 'knockoff', removed};
  } else {
    // Trick leaves a public item hand-off: `-item` for the recipient or a
    // silent `-enditem` when the other side had nothing to give.
    const swapped = log.some(line => line.startsWith('|-item|'));
    const expectsSwap = !trial.name.includes('refused') && !trial.name.includes('both_empty');
    if (swapped !== expectsSwap) {
      session.destroy();
      skipped.push({name: trial.name, reason: `probe failed: swapped=${swapped} expected=${expectsSwap}`});
      continue;
    }
    fixture.coverage = {item: toID(trial.target.item) || 'none', move: trial.move, swapped};
  }
  fixtures.push(fixture);
  session.destroy();
}
fs.writeFileSync(new URL('../data/more_item_interactions.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify({trials: TRIALS.length, fixtures: fixtures.length, skipped: skipped.length}));
if (skipped.length) console.log(JSON.stringify(skipped));
