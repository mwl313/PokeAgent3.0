// Development-only interaction corpus for the rampage lock
// (`conditions:lockedmove`: Outrage / Thrash / Petal Dance / Raging Fury).
//
// The generic move corpus records one battle per move and exercises the basic
// lock, its replay through `getLockedMove` requests and the end-of-rampage
// fatigue confusion. This generator scripts the paths that battle cannot reach:
// the two- and three-turn rolls, an immune target that never starts the lock, a
// mid-rampage sleep that ends the lock without confusion, and a fully
// paralysed continuation turn whose expiry residual ends the lock instead.
// Every fixture is a complete legal reference battle recorded at every
// decision boundary, including the served request mask.
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
  ['Milotic', 'Marvel Scale', ['Surf', 'Protect']],
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
    // Reference `getLockedMove()`: a charging, recharging or rampaging Pokémon
    // is served exactly one entry and refuses switches; `moveSlots` alone would
    // over-report the legal mask.
    const locked = p.getLockedMove();
    if (locked === 'recharge') {
      return {present: !p.fainted, requires_replacement: forced, can_mega: false,
        locked_recharge: true, trapped: true, maybe_trapped: false, moves: []};
    }
    if (locked) {
      const slotData = p.moveSlots.find(m => m.id === locked);
      return {present: !p.fainted, requires_replacement: forced, can_mega: false,
        locked: ids.moves[locked], trapped: true, maybe_trapped: false,
        moves: [{id: ids.moves[locked], pp: slotData?.pp ?? 0, disabled: false, target: slotData?.target ?? 'normal'}]};
    }
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
      active_slot: s.active.indexOf(p) >= 0 ? s.active.indexOf(p) : null, ability_ending: Boolean(p.abilityState.ending), cached_speed: p.speed ?? null,
      status: ids.conditions[p.status] ?? 0, boosts: Object.values(p.boosts), stats: [p.maxhp, ...Object.values(p.storedStats)], ability: ids.abilities[p.ability],
      item: ids.items[p.item] ?? 0, types: p.types.map(t => ids.types[toID(t)] ?? 0), previous_item: ids.items[p.lastItem] ?? 0, can_mega: Boolean(p.canMegaEvo),
      pp: p.moveSlots.map(m => m.pp), volatiles: Object.keys(p.volatiles).sort()})),
    request_detail: requestDetail(session, s)}))});

const select = (kind, own_slot, destination = 255) =>
  ({kind, own_slot, move_slot: 255, target_location: 0, switch_destination: destination, resource: 'None'});
const moveAction = (slot, moveSlot, target = 0) =>
  ({kind: 'Move', own_slot: slot, move_slot: moveSlot, target_location: target, switch_destination: 255, resource: 'None'});

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
    // is `move 1` with no target: the rampage re-samples its `randomNormal`
    // target from `lastMoveTargetLoc` (0) exactly like the reference.
    const locked = p.getLockedMove();
    if (locked) {
      if (locked === 'recharge') {
        actions.push(moveAction(slot, 255, 0));
      } else {
        const recorded = p.volatiles[locked]?.targetLoc ?? p.lastMoveTargetLoc ?? 0;
        const lockedSlot = Math.max(0, p.moveSlots.findIndex(m => m.id === locked));
        actions.push(moveAction(slot, lockedSlot, recorded));
      }
      commands.push('move 1');
      continue;
    }
    const wanted = plan(sideIndex, slot, b.turn, p);
    // A plan may name an explicit target slot on the opposing side so the
    // scripted move lands on the intended Pokémon (`getLocOf` resolves the
    // relative location exactly as the server would validate it).
    const wantedMove = typeof wanted === 'object' && wanted !== null ? wanted.move : wanted;
    const wantedName = wantedMove ? moveNames[wantedMove] : null;
    const slotIndex = wantedName ? p.moveSlots.findIndex(m => m.id === wantedName && !m.disabled && m.pp > 0) : -1;
    const choice = slotIndex >= 0 ? slotIndex : p.moveSlots.findIndex(m => !m.disabled && m.pp > 0);
    const chosen = p.moveSlots[choice];
    if (!chosen) { actions.push(moveAction(slot, 255, 0)); commands.push('move 1'); continue; }
    // Foe-side locations are relative: positive to the right, negative to the
    // left. `randomNormal` takes no chosen location (the server samples one),
    // so those commands carry no target.
    const foeSide = b.sides[sideIndex === 0 ? 1 : 0];
    const wantedTarget = typeof wanted === 'object' && wanted !== null && wanted.targetSlot !== undefined
      ? p.getLocOf(foeSide.active[wanted.targetSlot]) : null;
    const candidates = sideIndex === 0 ? [2, 1, -1, -2, 0] : [-2, -1, 1, 2, 0];
    const location = wantedTarget ??
      (chosen && b.actions.targetTypeChoices(chosen.target)
        ? candidates.find(loc => b.validTargetLoc(loc, p, chosen.target)) ?? 0 : 0);
    actions.push(moveAction(slot, choice, location));
    commands.push(`move ${choice + 1}${location ? ` ${location}` : ''}`);
  }
  return {actions, command: commands.join(', ')};
}

function teamsFor(holder, opponent) {
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
  const teamB = nameSets(pick(opponent, new Set()));
  const problems = validator.validateTeam(teamA) || validator.validateTeam(teamB);
  if (problems) throw new Error(`invalid fixture teams: ${problems.join('; ')}`);
  return [teamA, teamB];
}

function play({name, holder, opponent = [], plan, require: requireEvidence, seed}) {
  const [teamA, teamB] = teamsFor(holder, opponent);
  const session = new ReferenceSession({teams: [teamA, teamB], seed});
  const fixture = {name: `${name}_${seed[3]}`, seed,
    teams: [teamA, teamB].map((team, side) => ({id: `${name}-${side}`, members: team.map(s => ({
      species: ids.species[toID(s.species)], ability: ids.abilities[toID(s.ability)], item: ids.items[toID(s.item)] ?? 0,
      nature: ids.natures[toID(s.nature)], gender: s.gender || '', level: 50, moves: s.moves.map(x => ids.moves[toID(x)]),
      points: stats.map(k => s.evs[k]), ivs: stats.map(k => s.ivs[k])}))})),
    initial: compact(session), steps: []};
  let failure = null;
  const debugStates = [];
  while (!session.battle.ended && fixture.steps.length < 200) {
    for (let side = 0; side < 2; side++) {
      if (session.battle.ended) break;
      const s = session.battle.sides[side];
      if (s.activeRequest?.wait || s.isChoiceDone()) continue;
      const choice = choose(session, side, plan);
      const result = session.choose(side ? 'p2' : 'p1', choice.command);
      if (!result.accepted) { failure = JSON.stringify({name, side, choice, messages: result.messages}); break; }
      fixture.steps.push({side: side ? 'P2' : 'P1', ...choice, expected: compact(session)});
      const dbg = session.battle.sides[0].active[0]?.volatiles?.lockedmove;
      debugStates.push(`t${session.battle.turn} locked=${dbg ? `${dbg.duration}/${dbg.trueDuration}` : '-'}`);
    }
    if (failure) break;
  }
  const log = session.battle.log.slice();
  if (failure) { session.destroy(); return {name, reason: failure}; }
  if (!session.battle.ended) { session.destroy(); return {name, reason: 'did not complete'}; }
  // Development aid: `DEBUG_RAMPAGE=<name substring>` prints the reference
  // protocol lines, the per-boundary lock state (declared duration / rolled
  // true duration, from the reference's own volatile) and a per-seed summary of
  // every candidate seed, so a scenario's expectations can be checked against
  // what the pinned reference actually did.
  if (process.env.DEBUG_RAMPAGE && name.includes(process.env.DEBUG_RAMPAGE)) {
    console.log(`--- ${name} ${seed.join(',')}\n${log.filter(line => line.startsWith('|')).join('\n')}`);
    console.log(debugStates.join(' | '));
    console.log(JSON.stringify({seed: seed[3], cant: cantTurns(log, 'par'),
      uses: usedTurns(log, 'Outrage'), fatigue: fatigueConfusion(log),
      locked: lockedAt(fixture, 0, 0, 'outrage')}));
  }
  const evidence = requireEvidence(fixture, log);
  session.destroy();
  if (!evidence.ok) return {name, reason: evidence.reason};
  fixture.coverage = evidence.coverage;
  return {fixture};
}

/// The turns p1a used the named move, in order.
const usedTurns = (log, move) => {
  const out = [];
  let turn = 0;
  for (const line of log) {
    const match = line.match(/^\|turn\|(\d+)$/);
    if (match) { turn = Number(match[1]); continue; }
    if (line.startsWith(`|move|p1a: s0|${move}|`)) out.push(turn);
  }
  return out;
};
/// The turns p1a could not move for the given reason prefix (e.g. `par`).
const cantTurns = (log, reason) => {
  const out = [];
  let turn = 0;
  for (const line of log) {
    const match = line.match(/^\|turn\|(\d+)$/);
    if (match) { turn = Number(match[1]); continue; }
    if (line.startsWith(`|cant|p1a: s0|${reason}`)) out.push(turn);
  }
  return out;
};
const lockedAt = (fixture, side, slot, move) => [...new Set(fixture.steps
  .filter(step => step.expected.sides[side].request_detail?.slots?.[slot]?.locked === ids.moves[move])
  .map(step => step.expected.turn))];
const volatileAt = (fixture, side, slot, volatile) => fixture.steps
  .flatMap(step => step.expected.sides[side].pokemon.filter(p => p.active_slot === slot))
  .some(p => p.volatiles.includes(volatile));
const fatigueConfusion = log => log.some(line =>
  line.startsWith('|-start|p1a: s0|confusion') && line.includes('[fatigue]'));

const scenarios = [
  {
    // The short roll (random(2, 4) === 2) locks for two turns and confuses.
    name: 'rampage_short_two_turn_lock',
    holder: [setOf('Arcanine', 'Intimidate', ['Protect', 'Outrage'])],
    plan: (side, slot, turn) => (side === 0 && slot === 0 && turn === 1 ? ids.moves.outrage : null),
    require: (fixture, log) => {
      const uses = usedTurns(log, 'Outrage');
      const lockedTurns = lockedAt(fixture, 0, 0, 'outrage');
      const locked = volatileAt(fixture, 0, 0, 'lockedmove');
      const fatigue = fatigueConfusion(log);
      const stillLocked = lockedTurns.includes(3);
      return uses.length === 2 && lockedTurns.length === 1 && locked && fatigue && !stillLocked
        ? {ok: true, coverage: {move: 'outrage', rampage_turns: 2, fatigue_confusion: true}}
        : {ok: false, reason: `short lock not observed (uses=${uses.length} lockedTurns=${lockedTurns} volatile=${locked} fatigue=${fatigue} late=${stillLocked})`};
    },
  },
  {
    // The long roll (random(2, 4) === 3) locks for three turns and confuses.
    name: 'rampage_long_three_turn_lock',
    holder: [setOf('Arcanine', 'Intimidate', ['Protect', 'Outrage'])],
    plan: (side, slot, turn) => (side === 0 && slot === 0 && turn === 1 ? ids.moves.outrage : null),
    require: (fixture, log) => {
      const uses = usedTurns(log, 'Outrage');
      const lockedTurns = lockedAt(fixture, 0, 0, 'outrage');
      const fatigue = fatigueConfusion(log);
      return uses.length === 3 && lockedTurns.includes(3) && fatigue
        ? {ok: true, coverage: {move: 'outrage', rampage_turns: 3, fatigue_confusion: true}}
        : {ok: false, reason: `long lock not observed (uses=${uses.length} lockedTurns=${lockedTurns} fatigue=${fatigue})`};
    },
  },
  {
    // A Dragon move into two Fairy-types never lands: the self payload is
    // dropped with the false target, so the rampage never starts.
    name: 'rampage_immune_target_skips_lock',
    holder: [setOf('Arcanine', 'Intimidate', ['Protect', 'Outrage'])],
    opponent: [
      setOf('Florges', 'Flower Veil', ['Dazzling Gleam', 'Protect']),
      setOf('Whimsicott', 'Infiltrator', ['Dazzling Gleam', 'Protect']),
    ],
    plan: (side, slot, turn) => (side === 0 && slot === 0 && turn === 1 ? ids.moves.outrage : null),
    require: (fixture, log) => {
      const uses = usedTurns(log, 'Outrage');
      const immune = log.some(line => line.startsWith('|-immune|'));
      const locked = volatileAt(fixture, 0, 0, 'lockedmove') || lockedAt(fixture, 0, 0, 'outrage').length > 0;
      return uses.length >= 1 && immune && !locked
        ? {ok: true, coverage: {move: 'outrage', immune_target: true, lock_skipped: true}}
        : {ok: false, reason: `immunity path not observed (uses=${uses.length} immune=${immune} locked=${locked})`};
    },
  },
  {
    // Yawn resolves on the second residual of a three-turn roll: the sleeping
    // holder drops the lock on the non-expiry residual, so no fatigue confusion.
    name: 'rampage_sleep_cancels_lock',
    holder: [setOf('Arcanine', 'Intimidate', ['Protect', 'Outrage'])],
    opponent: [setOf('Snorlax', 'Thick Fat', ['Yawn', 'Protect'])],
    plan: (side, slot, turn) => (side === 0 && slot === 0 && turn === 1 ? ids.moves.outrage
      : side === 1 && slot === 0 && turn === 1 ? {move: ids.moves.yawn, targetSlot: 0} : null),
    require: (fixture, log) => {
      const uses = usedTurns(log, 'Outrage');
      const slept = log.some(line => line.startsWith('|-status|p1a: s0|slp'));
      const locked = volatileAt(fixture, 0, 0, 'lockedmove');
      const afterSleep = fixture.steps.some(step => step.expected.sides[0]
        .pokemon.some(p => p.active_slot === 0 && p.status === ids.conditions.slp));
      const stillLocked = lockedAt(fixture, 0, 0, 'outrage').includes(3);
      return uses.length === 2 && slept && locked && afterSleep && !fatigueConfusion(log) && !stillLocked
        ? {ok: true, coverage: {move: 'outrage', sleep_cancel: true, no_fatigue: true}}
        : {ok: false, reason: `sleep cancel not observed (uses=${uses.length} slept=${slept} volatile=${locked} after=${afterSleep} fatigue=${fatigueConfusion(log)} late=${stillLocked})`};
    },
  },
  {
    // A fully paralysed continuation turn skips the move, so the declared
    // duration expires in the residual and `onEnd` confuses the holder.
    name: 'rampage_paralysis_expiry_confuses',
    holder: [setOf('Arcanine', 'Intimidate', ['Protect', 'Outrage'])],
    opponent: [setOf('Gyarados', 'Intimidate', ['Thunder Wave', 'Protect'])],
    plan: (side, slot, turn) => (side === 0 && slot === 0 && turn === 1 ? ids.moves.outrage
      : side === 1 && slot === 0 && turn === 1 ? {move: ids.moves.thunderwave, targetSlot: 0} : null),
    require: (fixture, log) => {
      // The continuation turn itself must be the one lost: a later `par` cant
      // (the holder trying Protect after the lock) is not this path.
      const cant = cantTurns(log, 'par').includes(2);
      const uses = usedTurns(log, 'Outrage');
      const fatigue = fatigueConfusion(log);
      const lockedTurns = lockedAt(fixture, 0, 0, 'outrage');
      const stillLocked = lockedTurns.includes(3);
      return cant && uses.length === 1 && lockedTurns.includes(2) && fatigue && !stillLocked
        ? {ok: true, coverage: {move: 'outrage', aborted_turn: 'paralysis', expiry_confusion: true}}
        : {ok: false, reason: `paralysis expiry not observed (cant=${cant} cantTurns=${cantTurns(log, 'par')} uses=${uses.length} lockedTurns=${lockedTurns} fatigue=${fatigue} late=${stillLocked})`};
    },
  },
  {
    // Thrash is the same condition on a Normal move: a second witness for the
    // shared lock with a different declaring move.
    name: 'rampage_thrash_two_turn_lock',
    holder: [setOf('Arcanine', 'Intimidate', ['Protect', 'Thrash'])],
    plan: (side, slot, turn) => (side === 0 && slot === 0 && turn === 1 ? ids.moves.thrash : null),
    require: (fixture, log) => {
      const uses = usedTurns(log, 'Thrash');
      const lockedTurns = lockedAt(fixture, 0, 0, 'thrash');
      const fatigue = fatigueConfusion(log);
      const stillLocked = lockedTurns.includes(3);
      return uses.length === 2 && lockedTurns.length === 1 && fatigue && !stillLocked
        ? {ok: true, coverage: {move: 'thrash', rampage_turns: 2, fatigue_confusion: true}}
        : {ok: false, reason: `thrash lock not observed (uses=${uses.length} lockedTurns=${lockedTurns} fatigue=${fatigue} late=${stillLocked})`};
    },
  },
];

const fixtures = [];
const skipped = [];
for (const [index, scenario] of scenarios.entries()) {
  let result = null;
  for (let i = 0; i < 220; i++) {
    result = play({...scenario, seed: [2026, 10, 8, 5000 + index * 220 + i]});
    if (result.fixture) break;
  }
  if (result?.fixture) fixtures.push(result.fixture);
  else skipped.push({name: scenario.name, reason: result?.reason ?? 'no seed produced the required behaviour'});
}

fs.writeFileSync(new URL('../data/more_rampage.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify({fixtures: fixtures.length, skipped: skipped.length,
  steps: fixtures.reduce((n, f) => n + f.steps.length, 0)}));
if (skipped.length) console.log(JSON.stringify(skipped, null, 1));
if (fixtures.length !== scenarios.length) process.exitCode = 1;
