// Development-only interaction corpus for Expanding Force.
//
// The generic move corpus never sets Psychic Terrain, so the two conditional
// callbacks (`onModifyMove` spread conversion and `onBasePower` 1.5x boost for
// a grounded user) need dedicated witnesses:
//   1. grounded Indeedee (Psychic Surge) hits both foes in the same action;
//   2. airborne Chimecho (Levitate) with a Psychic Surge partner stays
//      single-target.
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
    conditions: Object.entries(s.sideConditions).map(([id, state]) => [ids.conditions[id], state.duration]).sort((a, b) => a[0] - b[0]),
    pokemon: s.pokemon.map(p => ({roster: roster(p), species: ids.species[p.species.id], hp: p.hp, max_hp: p.maxhp, fainted: p.fainted,
      active_slot: s.active.indexOf(p) >= 0 ? s.active.indexOf(p) : null, ability_ending: Boolean(p.abilityState.ending),
      cached_speed: p.speed ?? null, status: ids.conditions[p.status] ?? 0, boosts: Object.values(p.boosts),
      stats: [p.maxhp, ...Object.values(p.storedStats)], ability: ids.abilities[p.ability], item: ids.items[p.item] ?? 0,
      types: p.types.map(t => ids.types[toID(t)]), previous_item: ids.items[p.lastItem] ?? 0, can_mega: Boolean(p.canMegaEvo),
      pp: p.moveSlots.map(m => m.pp), volatiles: Object.keys(p.volatiles).sort()})),
    request_detail: requestDetail(session, s)}))});

const select = (kind, own_slot, destination = 255) =>
  ({kind, own_slot, move_slot: 255, target_location: 0, switch_destination: destination, resource: 'None'});
const moveAction = (slot, moveSlot, target = 0) =>
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
    const wanted = sideIndex === 0 && slot === plan.slot ? plan.moveSlot : plan.attackSlot;
    const slotIndex = p.moveSlots.findIndex(m => m.id === wanted && !m.disabled && m.pp > 0);
    const choice = slotIndex >= 0 ? slotIndex : p.moveSlots.findIndex(m => !m.disabled && m.pp > 0);
    const chosen = p.moveSlots[choice];
    const candidates = sideIndex === 0 ? [2, 1, -1, -2, 0] : [-2, -1, 1, 2, 0];
    const location = chosen && b.actions.targetTypeChoices(chosen.target)
      ? candidates.find(loc => b.validTargetLoc(loc, p, chosen.target)) ?? 0 : 0;
    actions.push(moveAction(slot, choice, location));
    commands.push(`move ${choice + 1}${location ? ` ${location}` : ''}`);
  }
  return {actions, command: commands.join(', ')};
}

const FILLERS = [
  ['Goodra-Hisui', 'Shell Armor', ['Dragon Pulse', 'Protect']],
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']],
  ['Metagross', 'Clear Body', ['Iron Head', 'Protect']],
  ['Milotic', 'Competitive', ['Surf', 'Protect']],
  ['Scolipede', 'Swarm', ['X-Scissor', 'Protect']],
  ['Starmie', 'Natural Cure', ['Ice Beam', 'Protect']],
  ['Blaziken', 'Speed Boost', ['Close Combat', 'Protect']],
].map(([species, ability, moves]) => setOf(species, ability, moves, ''));

const TRIALS = [
  {
    name: 'expandingforce_grounded_spread',
    holder: setOf('Indeedee', 'Psychic Surge', ['Expanding Force', 'Protect'], ''),
    partner: FILLERS[2],
    slot: 0,
    expected_targets: 2,
  },
  {
    name: 'expandingforce_airborne_single',
    holder: setOf('Chimecho', 'Levitate', ['Expanding Force', 'Protect'], ''),
    partner: setOf('Indeedee', 'Psychic Surge', ['Protect', 'Helping Hand'], ''),
    slot: 0,
    expected_targets: 1,
  },
];

const fixtures = [];
const skipped = [];
let seedIndex = 0;
for (const trial of TRIALS) {
  const base = dex.species.get(trial.holder.species).baseSpecies;
  const extras = FILLERS.filter(s => dex.species.get(s.species).baseSpecies !== base)
    .filter(s => trial.partner && dex.species.get(s.species).baseSpecies !== dex.species.get(trial.partner.species).baseSpecies);
  const teamA = nameSets([trial.holder, trial.partner, ...extras].slice(0, 6));
  const teamB = nameSets(FILLERS.filter(s => dex.species.get(s.species).baseSpecies !== base).slice(0, 6));
  const problems = validator.validateTeam(teamA) || validator.validateTeam(teamB);
  if (problems) { skipped.push({name: trial.name, reason: problems.join('; ')}); continue; }
  let recorded = null, lastReason = null;
  for (let attempt = 0; attempt < 64 && !recorded; attempt++) {
    const seed = [2026, 10, 7, 3980 + seedIndex * 64 + attempt];
    const session = new ReferenceSession({teams: [teamA, teamB], seed});
    if (process.env.PA3_DMG_DBG) {
      const actions = session.battle.actions;
      const original = actions.getDamage.bind(actions);
      const battle = session.battle;
      const originalRunEvent = battle.runEvent.bind(battle);
      battle.runEvent = (eventid, ...rest) => {
        if (eventid === 'BasePower') {
          const before = rest[3];
          const move = rest[2];
          const after = originalRunEvent(eventid, ...rest);
          if (move?.id === 'expandingforce') {
            console.log('BASEPOWER', before, '->', typeof after === 'number' ? after : JSON.stringify(after),
              'modifier', battle.event?.modifier);
          }
          return after;
        }
        return originalRunEvent(eventid, ...rest);
      };
      actions.getDamage = (source, target, move, suppressMessages) => {
        const result = original(source, target, move, suppressMessages);
        if (move?.id === 'expandingforce') {
          console.log('DMG', target.name, 'power', JSON.stringify(move.basePower),
            'atk', source.getStat('spa'), 'def', target.getStat('spd'),
            'multiplier', session.battle.event?.modifier, 'result', result);
        }
        return result;
      };
    }
    const plan = {slot: trial.slot, moveSlot: ids.moves.expandingforce, attackSlot: ids.moves.ironhead};
    const fixture = {name: `move_${trial.name}_${seed[3]}`, seed,
      teams: [teamA, teamB].map((team, side) => ({id: `expanding-${trial.name}-${side}`, members: team.map(s => ({
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
    // Require an Expanding Force use that hit exactly the expected number of foes.
    let observed = null;
    const log = session.battle.log;
    for (let i = 0; i < log.length; i++) {
      if (!log[i].startsWith('|move|p1a:') || log[i].split('|')[3] !== 'Expanding Force') continue;
      // Champions logs each hit twice (raw HP and percentage), so count unique
      // damaged foe slots rather than damage lines.
      const hitTargets = new Set();
      for (let j = i + 1; j < log.length && !log[j].startsWith('|move|'); j++) {
        if (log[j].startsWith('|-damage|p2')) hitTargets.add(log[j].split('|')[2]);
      }
      const hits = hitTargets.size;
      if (hits === trial.expected_targets) { observed = hits; break; }
    }
    if (observed === null) { session.destroy(); lastReason = `no Expanding Force use hit ${trial.expected_targets} foes`; continue; }
    fixture.coverage = {move: 'expandingforce', grounded_targets: observed,
      terrain: 'psychicterrain', slot: trial.slot};
    recorded = fixture;
    session.destroy();
  }
  if (recorded) fixtures.push(recorded);
  else skipped.push({name: trial.name, reason: lastReason ?? 'no seed produced the required behavior'});
  seedIndex++;
}

fs.writeFileSync(new URL('../data/more_expanding_force.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify({fixtures: fixtures.length, skipped: skipped.length}));
if (skipped.length) console.log(JSON.stringify(skipped));
