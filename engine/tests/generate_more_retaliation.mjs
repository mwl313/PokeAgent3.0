// Development-only interaction corpus for the retaliation family:
//
// - Counter / Mirror Coat add a one-turn volatile before the turn runs, record
//   the last qualifying (Physical / Special, non-ally) hit in it, and strike
//   that attacker for twice the recorded damage.
// - Metal Burst / Comeuppance read the turn's `attackedBy` record, retarget the
//   last non-ally attacker and deal 1.5x that move's damage; they fail while no
//   such hit happened this turn.
//
// The generic move corpus records one battle per move; this generator scripts
// the paths that battle cannot reach: the recorded-hit requirement, the ally
// filter, the multi-hit accumulation and the fail-then-hit sequence.
// Every fixture is a complete legal reference battle recorded at every decision
// boundary, including the served request mask.
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
  ['Milotic', 'Marvel Scale', ['Dragon Pulse', 'Protect']],
  ['Scolipede', 'Swarm', ['X-Scissor', 'Protect']],
  ['Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']],
  ['Starmie', 'Natural Cure', ['Dragon Pulse', 'Protect']],
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

// `plan(side, slot, turn, pokemon)` returns a numeric move id, `{move, targetSlot}`
// for a move aimed at the opposing side's active slot, `{move, targetAlly}` for
// one aimed at the acting side's own slot, or null for the default choice.
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
    // A locked Pokémon is served exactly one entry: `move 1` with no target.
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
    const wantedMove = typeof wanted === 'object' && wanted !== null ? wanted.move : wanted;
    const wantedName = wantedMove ? moveNames[wantedMove] : null;
    const slotIndex = wantedName ? p.moveSlots.findIndex(m => m.id === wantedName && !m.disabled && m.pp > 0) : -1;
    const choice = slotIndex >= 0 ? slotIndex : p.moveSlots.findIndex(m => !m.disabled && m.pp > 0);
    const chosen = p.moveSlots[choice];
    if (!chosen) { actions.push(moveAction(slot, 255, 0)); commands.push('move 1'); continue; }
    // `scripted` targets are not choosable, so their commands carry no location
    // (the server samples one); explicit plans resolve their own location.
    const foeSide = b.sides[sideIndex === 0 ? 1 : 0];
    const wantedTarget = slotIndex >= 0 && typeof wanted === 'object' && wanted !== null
      ? (wanted.targetSlot !== undefined
          ? p.getLocOf(foeSide.active[wanted.targetSlot])
          : wanted.targetAlly !== undefined
            ? p.getLocOf(side.active[wanted.targetAlly])
            : null)
      : null;
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
  // Development aid: `DEBUG_RETALIATION=<name substring>` prints the reference
  // protocol lines of every candidate seed.
  if (process.env.DEBUG_RETALIATION && name.includes(process.env.DEBUG_RETALIATION)) {
    console.log(`--- ${name} ${seed.join(',')}\n${log.filter(line => line.startsWith('|')).join('\n')}`);
  }
  const evidence = requireEvidence(fixture, log);
  session.destroy();
  if (!evidence.ok) return {name, reason: evidence.reason};
  fixture.coverage = evidence.coverage;
  return {fixture};
}

/// Every `|-damage|` line, attributed to the move that preceded it. A lethal
/// hit reports `0 fnt`, so its delta comes from the previous HP.
const damageEvents = log => {
  const events = [];
  let turn = 0, attacker = null, move = null;
  // The Champions log mirrors every HP change on a 0-100 scale, so each ident
  // records the raw maximum it first appeared with and ignores the mirror.
  const raw = new Map();
  for (const line of log) {
    const turnMatch = line.match(/^\|turn\|(\d+)$/);
    if (turnMatch) { turn = Number(turnMatch[1]); continue; }
    const moveMatch = line.match(/^\|move\|(p[12][ab]: s\d+)\|([^|]+)\|/);
    if (moveMatch) { attacker = moveMatch[1]; move = moveMatch[2]; continue; }
    const parts = line.split('|');
    const ident = parts[2] ?? '';
    // Switch lines report the entrant's raw HP, so a replacement resets the
    // baseline instead of creating a negative delta.
    if (['switch', 'drag', 'replace'].includes(parts[1]) && /^p[12][ab]: s\d+$/.test(ident)) {
      const value = (parts[4] ?? parts[3] ?? '').split(' ')[0];
      if (/^\d+\/\d+$/.test(value)) {
        const [hpValue, max] = value.split('/').map(Number);
        const known = raw.get(ident);
        // The 0-100 mirror line never replaces the raw baseline.
        if (!known || known.max === null || known.max === max) raw.set(ident, {hp: hpValue, max});
      }
      continue;
    }
    if (parts[1] !== '-damage' || !/^p[12][ab]: s\d+$/.test(ident)) continue;
    const value = (parts[3] ?? '').split(' ')[0];
    let current = null, max = null;
    if (/^\d+\/\d+$/.test(value)) [current, max] = value.split('/').map(Number);
    else if (value === '0' || value === '0 fnt') current = 0;
    if (current === null) continue;
    const known = raw.get(ident);
    if (known && max !== null && known.max !== null && max !== known.max) continue;
    const previous = known ? known.hp : (max ?? current);
    events.push({turn, ident, attacker, move, hp: current, delta: previous - current});
    raw.set(ident, {hp: current, max: known?.max ?? max ?? null});
  }
  return events;
};
/// Damage one ident took in one turn from one move (all attackers when
/// `attacker` is omitted).
const damageFrom = (log, ident, turn, move, attacker = null) => damageEvents(log)
  .filter(event => event.turn === turn && event.ident === ident && event.move === move
    && (attacker === null || event.attacker === attacker))
  .reduce((total, event) => total + event.delta, 0);
const damageInTurn = (log, ident, turn) => damageEvents(log)
  .filter(event => event.turn === turn && event.ident === ident)
  .reduce((total, event) => total + event.delta, 0);
const usedMove = (log, ident, name) => log.some(line => line.startsWith(`|move|${ident}|${name}|`));
const ppOf = (fixture, side, slot, move) => fixture.steps
  .flatMap(step => step.expected.sides[side].pokemon.filter(p => p.roster === slot))
  .map(p => p.pp[fixture.teams[side].members[slot].moves.indexOf(ids.moves[move])])
  .filter(value => value !== undefined);

const scenarios = [
  {
    // Counter records the Physical hit and answers for twice that damage.
    name: 'retaliation_counter_doubles_physical',
    holder: [setOf('Bastiodon', 'Sturdy', ['Protect', 'Counter'])],
    opponent: [setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect'])],
    plan: (side, slot, turn) => (side === 0 && slot === 0 && turn === 1 ? ids.moves.counter
      : side === 1 && slot === 0 ? {move: ids.moves.ironhead, targetSlot: 0} : null),
    require: (fixture, log) => {
      const incoming = damageFrom(log, 'p1a: s0', 1, 'Iron Head');
      const back = damageFrom(log, 'p2a: s0', 1, 'Counter');
      const fired = usedMove(log, 'p1a: s0', 'Counter');
      const pp = ppOf(fixture, 0, 0, 'counter');
      return fired && incoming > 0 && back === incoming * 2 && pp.at(-1) === pp[0] - 1
        ? {ok: true, coverage: {move: 'counter', kind: 'physical', doubled: true}}
        : {ok: false, reason: `counter doubling not observed (fired=${fired} incoming=${incoming} back=${back} pp=${pp.at(-1)})`};
    },
  },
  {
    // Counter's volatile only records Physical hits: a Special-only turn leaves
    // the move failing even though it was selected.
    name: 'retaliation_counter_refuses_special_hit',
    holder: [setOf('Bastiodon', 'Sturdy', ['Protect', 'Counter'])],
    opponent: [setOf('Milotic', 'Marvel Scale', ['Dragon Pulse', 'Protect'])],
    plan: (side, slot, turn) => (side === 0 && slot === 0 && turn === 1 ? ids.moves.counter
      : side === 1 && slot === 0 && turn === 1 ? {move: ids.moves.dragonpulse, targetSlot: 0} : null),
    require: (fixture, log) => {
      const incoming = damageFrom(log, 'p1a: s0', 1, 'Dragon Pulse');
      const back = damageFrom(log, 'p2a: s0', 1, 'Counter');
      const fired = usedMove(log, 'p1a: s0', 'Counter');
      return fired && incoming > 0 && back === 0
        ? {ok: true, coverage: {move: 'counter', refused: 'special-only turn'}}
        : {ok: false, reason: `counter special refusal not observed (fired=${fired} incoming=${incoming} back=${back})`};
    },
  },
  {
    // Mirror Coat is the Special mirror image of Counter.
    name: 'retaliation_mirrorcoat_doubles_special',
    holder: [setOf('Milotic', 'Marvel Scale', ['Protect', 'Mirror Coat'])],
    opponent: [setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect', 'Psychic'])],
    plan: (side, slot, turn) => (side === 0 && slot === 0 && turn === 1 ? ids.moves.mirrorcoat
      : side === 1 && slot === 0 ? {move: ids.moves.psychic, targetSlot: 0} : null),
    require: (fixture, log) => {
      const incoming = damageFrom(log, 'p1a: s0', 1, 'Psychic');
      const back = damageFrom(log, 'p2a: s0', 1, 'Mirror Coat');
      const fired = usedMove(log, 'p1a: s0', 'Mirror Coat');
      return fired && incoming > 0 && back === incoming * 2
        ? {ok: true, coverage: {move: 'mirrorcoat', kind: 'special', doubled: true}}
        : {ok: false, reason: `mirror coat doubling not observed (fired=${fired} incoming=${incoming} back=${back})`};
    },
  },
  {
    // The reference records one `attackedBy` entry per move with its *last*
    // hit's damage (its hit loop replaces `moveDamage` each hit), so a two-hit
    // Double Hit is answered for 1.5x the second hit only.
    name: 'retaliation_metalburst_last_hit',
    holder: [setOf('Aggron', 'Sturdy', ['Protect', 'Metal Burst'])],
    opponent: [setOf('Falinks', 'Battle Armor', ['Double Hit', 'Protect'])],
    plan: (side, slot, turn) => (side === 0 && slot === 0 && turn === 1 ? ids.moves.metalburst
      : side === 1 && slot === 0 ? {move: ids.moves.doublehit, targetSlot: 0} : null),
    require: (fixture, log) => {
      const hits = damageEvents(log).filter(event => event.turn === 1 && event.ident === 'p1a: s0'
        && event.move === 'Double Hit' && event.delta > 0);
      const back = damageFrom(log, 'p2a: s0', 1, 'Metal Burst');
      const fired = usedMove(log, 'p1a: s0', 'Metal Burst');
      const last = hits.at(-1)?.delta ?? 0;
      return fired && hits.length === 2 && back === Math.floor(last * 1.5)
        ? {ok: true, coverage: {move: 'metalburst', multi_hit: true, last_hit: true}}
        : {ok: false, reason: `metal burst last-hit rule not observed (fired=${fired} hits=${hits.map(h => h.delta)} back=${back})`};
    },
  },
  {
    // Comeuppance fails on a turn without damage and strikes the recorded
    // attacker for 1.5x on the next one.
    name: 'retaliation_comeuppance_fails_then_hits',
    holder: [setOf('Pangoro', 'Iron Fist', ['Protect', 'Comeuppance'])],
    opponent: [setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect'])],
    plan: (side, slot, turn) => (side === 0 && slot === 0 ? ids.moves.comeuppance
      : side === 1 && slot === 0
        ? (turn === 1 ? ids.moves.protect : {move: ids.moves.ironhead, targetSlot: 0})
        : null),
    require: (fixture, log) => {
      const firstTurnBack = damageInTurn(log, 'p2a: s0', 1) + damageInTurn(log, 'p2a: s1', 1);
      const secondTurnIncoming = damageFrom(log, 'p1a: s0', 2, 'Iron Head');
      const secondTurnBack = damageFrom(log, 'p2a: s0', 2, 'Comeuppance');
      const fired = log.filter(line => line.startsWith('|move|p1a: s0|Comeuppance|')).length;
      return fired >= 2 && firstTurnBack === 0 && secondTurnIncoming > 0
        && secondTurnBack === Math.floor(secondTurnIncoming * 1.5)
        ? {ok: true, coverage: {move: 'comeuppance', failed_without_damage: true, retargeted: true}}
        : {ok: false, reason: `comeuppance sequence not observed (fired=${fired} first=${firstTurnBack} incoming=${secondTurnIncoming} back=${secondTurnBack})`};
    },
  },
  {
    // An ally's hit never becomes the recorded attacker: the foe strikes first,
    // the ally second, and Counter still aims at (and doubles) the foe.
    name: 'retaliation_counter_ignores_ally_hit',
    holder: [
      setOf('Bastiodon', 'Sturdy', ['Protect', 'Counter']),
      setOf('Goodra-Hisui', 'Shell Armor', ['Dragon Pulse', 'Protect']),
    ],
    opponent: [setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect'])],
    plan: (side, slot, turn) => {
      if (side === 0 && slot === 0 && turn === 1) return ids.moves.counter;
      if (side === 0 && slot === 1 && turn === 1) return {move: ids.moves.dragonpulse, targetAlly: 0};
      if (side === 1 && slot === 0) return {move: ids.moves.ironhead, targetSlot: 0};
      return null;
    },
    require: (fixture, log) => {
      const foeBack = damageFrom(log, 'p2a: s0', 1, 'Counter');
      const events = damageEvents(log).filter(event => event.turn === 1 && event.ident === 'p1a: s0'
        && event.delta > 0);
      const foeHit = events.find(event => event.move === 'Iron Head');
      const allyHit = events.find(event => event.move === 'Dragon Pulse' && event.attacker === 'p1b: s1');
      const allyLast = foeHit && allyHit && events.indexOf(allyHit) > events.indexOf(foeHit);
      const fired = usedMove(log, 'p1a: s0', 'Counter');
      return fired && foeHit && allyLast && foeBack === foeHit.delta * 2
        ? {ok: true, coverage: {move: 'counter', ally_filtered: true}}
        : {ok: false, reason: `ally filter not observed (fired=${fired} events=${events.map(e => `${e.attacker}:${e.move}:${e.delta}`)} back=${foeBack})`};
    },
  },
];

const fixtures = [];
const skipped = [];
for (const [index, scenario] of scenarios.entries()) {
  let result = null;
  for (let i = 0; i < 40; i++) {
    result = play({...scenario, seed: [2026, 10, 8, 7000 + index * 40 + i]});
    if (result.fixture) break;
  }
  if (result?.fixture) fixtures.push(result.fixture);
  else skipped.push({name: scenario.name, reason: result?.reason ?? 'no seed produced the required behaviour'});
}

fs.writeFileSync(new URL('../data/more_retaliation.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify({fixtures: fixtures.length, skipped: skipped.length,
  steps: fixtures.reduce((n, f) => n + f.steps.length, 0)}));
if (skipped.length) console.log(JSON.stringify(skipped, null, 1));
if (fixtures.length !== scenarios.length) process.exitCode = 1;
