// Development-only corpus for the volatile selection family: Encore, Taunt,
// Disable, Imprison, Torment and the Cursed Body ability.
//
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary (including the reference
// request, so the native legal-action mask is compared as well). The exporter
// merges the artifact (`more_disable_family.json`) into turn-fixtures.json.
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
const setOf = (species, ability, moves, item = '', points = {hp: 32, atk: 0, def: 17, spa: 0, spd: 17, spe: 0}) =>
  ({name: species, species, ability, item, nature: 'Serious', level: 50, gender: 'M', moves,
    evs: {hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0, ...points}});
const offensive = (species, ability, moves, item = '') =>
  setOf(species, ability, moves, item, {hp: 2, atk: 32, def: 0, spa: 32, spd: 0, spe: 0});

// Fillers keep every fixture battle decisive without adding unported moves.
const FILLERS = [
  ['Metagross', 'Clear Body', ['Iron Head', 'Protect']],
  ['Milotic', 'Competitive', ['Surf', 'Protect']],
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']],
  ['Falinks', 'Battle Armor', ['Smart Strike', 'Protect']],
  ['Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']],
];
const fillerTeam = () => FILLERS.map(([species, ability, moves]) => offensive(species, ability, moves));
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
    // World move list and raw disable flag: the served choice legality. A
    // `'hidden'` Imprison disable keeps `disabled: true` here even though the
    // client display can show `false` for the last active slot.
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
        pp: p.moveSlots.map(m => m.pp), volatiles: Object.keys(p.volatiles).sort()})),
      request_detail: requestDetail(session, s)}))};
};

const select = (kind, own_slot, destination = 255) => ({
  kind, own_slot, move_slot: 255, target_location: 0, switch_destination: destination, resource: 'None',
});
const moveAction = (slot, moveSlot, target = 0) => ({
  kind: 'Move', own_slot: slot, move_slot: moveSlot, target_location: target, switch_destination: 255, resource: 'None',
});

// Resolve one side's actions: scripted move ids per slot, falling back to a
// damaging move so every fixture battle terminates naturally.
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
    // Choose with the served legality (`moveSlots`): the client display can
    // show a `'hidden'` Imprison disable as enabled, but the server rejects
    // that choice with a refreshed request, so a legal script never sends it.
    const view = p.moveSlots;
    // The request's computed target class is authoritative; a move slot's own
    // `target` field can be unset.
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
    // Reference `getMoves(lockedMove, restrictData = isLastActive())`: an
    // Imprison `'hidden'` disable is served as *enabled* for the side's last
    // active Pokemon (its execution-time `onFoeBeforeMove` gate then refuses
    // the move), and as disabled for every other slot. The Struggle override
    // applies only when the served request has no usable entry at all.
    const lastActive = p.isLastActive();
    const servedUsable = m => (!m.disabled || (m.disabled === 'hidden' && lastActive)) && m.pp > 0;
    const noMovesLeft = view.every(m => !servedUsable(m));
    if (noMovesLeft) {
      // The server overrides any submitted moveid with Struggle once the world
      // mask has no usable move. The native action space models that override
      // as the Struggle sentinel (move_slot 255), while the reference command
      // still names the first request entry and honours its target class.
      const target = targetFor(0);
      actions.push(moveAction(slot, 255, 0));
      commands.push(`move 1${target ? ` ${target}` : ''}`);
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

// Structural verification over the recorded boundaries: the fixture must show
// the mechanic actually applied, regardless of how the battle ends later.
const sawVolatile = (fixture, side, rosterIndex, volatile) =>
  fixture.steps.some(step => step.expected.sides[side].pokemon[rosterIndex].volatiles.includes(volatile));
const sawDisabled = (fixture, side, rosterIndex, moveId) =>
  fixture.steps.some(step => {
    const detail = step.expected.sides[side].request_detail;
    const slot = step.expected.sides[side].pokemon[rosterIndex].active_slot;
    if (!detail || slot === null) return false;
    const request = detail.slots[slot];
    return request && request.moves.some(m => m.id === ids.moves[moveId] && m.disabled);
  });

const TRIALS = [
  {
    name: 'disable_family_encore_after_move',
    p1: [setOf('Samurott', 'Shell Armor', ['Aqua Jet', 'Protect', 'Ice Beam']), ...fillerTeam().slice(0, 5)],
    p2: [setOf('Alakazam', 'Synchronize', ['Encore', 'Psychic', 'Shadow Ball', 'Protect']), ...fillerTeam().slice(0, 5)],
    seeds: [[7, 8, 9, 10], [11, 22, 33, 44], [101, 102, 103, 104]],
    script: [
      {p1: ['aquajet', null], p2: ['encore', null]},
    ],
    coverage: {move: 'encore'},
    verify(fixture) {
      if (!sawVolatile(fixture, 0, 0, 'encore')) return 'Encore volatile never recorded';
      if (!sawDisabled(fixture, 0, 0, 'protect')) return 'Encore did not disable the other moves';
      return null;
    },
  },
  {
    name: 'disable_family_encore_before_move',
    p1: [setOf('Samurott', 'Shell Armor', ['Aqua Jet', 'Protect', 'Ice Beam']), ...fillerTeam().slice(0, 5)],
    // Alakazam is faster than Samurott, so Encore lands before the queued
    // Protect and the Champions queue change replaces it with Aqua Jet.
    p2: [setOf('Alakazam', 'Synchronize', ['Encore', 'Psychic', 'Shadow Ball', 'Protect']), ...fillerTeam().slice(0, 5)],
    seeds: [[3, 14, 15, 92], [55, 66, 77, 88]],
    script: [
      {p1: ['aquajet', null], p2: ['protect', null]},
      // Ice Beam rather than Protect: Protect carries `failencore`, so it can
      // never become the encored move.
      {p1: ['icebeam', null], p2: ['encore', null]},
    ],
    coverage: {move: 'encore'},
    verify(fixture) {
      if (!sawVolatile(fixture, 0, 0, 'encore')) return 'Encore volatile never recorded';
      return null;
    },
  },
  {
    name: 'disable_family_taunt_status_lock',
    p1: [setOf('Samurott', 'Shell Armor', ['Protect', 'Aqua Jet', 'Ice Beam']), ...fillerTeam().slice(0, 5)],
    p2: [setOf('Alakazam', 'Synchronize', ['Taunt', 'Psychic', 'Shadow Ball', 'Protect']), ...fillerTeam().slice(0, 5)],
    seeds: [[21, 22, 23, 24], [31, 32, 33, 34]],
    script: [
      {p1: ['aquajet', null], p2: ['taunt', null]},
    ],
    coverage: {move: 'taunt'},
    verify(fixture) {
      if (!sawVolatile(fixture, 0, 0, 'taunt')) return 'Taunt volatile never recorded';
      if (!sawDisabled(fixture, 0, 0, 'protect')) return 'Taunt did not disable the Status move';
      return null;
    },
  },
  {
    name: 'disable_family_disable_last_move',
    p1: [setOf('Samurott', 'Shell Armor', ['Aqua Jet', 'Protect', 'Ice Beam']), ...fillerTeam().slice(0, 5)],
    p2: [setOf('Alakazam', 'Synchronize', ['Disable', 'Psychic', 'Shadow Ball', 'Protect']), ...fillerTeam().slice(0, 5)],
    seeds: [[41, 42, 43, 44], [51, 52, 53, 54]],
    script: [
      {p1: ['aquajet', null], p2: ['protect', null]},
      {p1: ['icebeam', null], p2: ['disable', null]},
    ],
    coverage: {move: 'disable'},
    verify(fixture) {
      if (!sawVolatile(fixture, 0, 0, 'disable')) return 'Disable volatile never recorded';
      if (!sawDisabled(fixture, 0, 0, 'aquajet')) return 'Disable did not disable the last move';
      return null;
    },
  },
  {
    name: 'disable_family_imprison_shared_moves',
    // Both leads know Ice Beam, which the imprisoning Pokémon also knows: the
    // choice is legal when committed and is refused by the priority-4
    // `onFoeBeforeMove` gate once Imprison lands mid-turn.
    p1: [setOf('Samurott', 'Shell Armor', ['Ice Beam', 'Protect', 'Aqua Jet']),
      setOf('Milotic', 'Competitive', ['Ice Beam', 'Protect', 'Surf']),
      ...fillerTeam().slice(2, 5), fillerTeam()[0]],
    // The imprisoning Pokémon sits in the second slot so the opponent's
    // fillers target the bulky first slot instead of fainting it.
    p2: [offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
      setOf('Froslass', 'Cursed Body', ['Imprison', 'Ice Beam', 'Protect', 'Shadow Ball']),
      ...fillerTeam().slice(1, 5)],
    seeds: [[3, 4, 5, 6], [61, 62, 63, 64]],
    script: [
      {p1: ['icebeam', 'icebeam'], p2: [null, 'imprison']},
    ],
    coverage: {move: 'imprison'},
    verify(fixture, session) {
      if (!sawVolatile(fixture, 1, 1, 'imprison')) return 'Imprison volatile never recorded';
      if (!sawDisabled(fixture, 0, 0, 'icebeam')) return 'Imprison did not hide the shared move';
      if (!sawDisabled(fixture, 0, 1, 'icebeam')) return 'Imprison did not hide the ally shared move';
      // Both queued Ice Beams are refused by the priority-4 gate in the turn
      // Imprison lands.
      if (!session.battle.log.some(line => line.includes('|cant|') && line.includes('Imprison'))) {
        return 'Imprison never refused the hidden move';
      }
      return null;
    },
  },
  {
    name: 'disable_family_torment_last_move',
    p1: [setOf('Samurott', 'Shell Armor', ['Aqua Jet', 'Protect', 'Ice Beam']), ...fillerTeam().slice(0, 5)],
    p2: [setOf('Kingambit', 'Defiant', ['Torment', 'Iron Head', 'Protect', 'Sucker Punch']), ...fillerTeam().slice(0, 5)],
    seeds: [[71, 72, 73, 74], [81, 82, 83, 84]],
    script: [
      {p1: ['aquajet', null], p2: ['protect', null]},
      {p1: ['icebeam', null], p2: ['torment', null]},
    ],
    coverage: {move: 'torment'},
    verify(fixture) {
      if (!sawVolatile(fixture, 0, 0, 'torment')) return 'Torment volatile never recorded';
      if (!sawDisabled(fixture, 0, 0, 'icebeam')) return 'Torment did not disable the last move';
      return null;
    },
  },
  {
    name: 'disable_family_cursed_body_disable',
    p1: [setOf('Samurott', 'Shell Armor', ['Aqua Jet', 'Protect', 'Ice Beam']), ...fillerTeam().slice(0, 5)],
    p2: [setOf('Froslass', 'Cursed Body', ['Imprison', 'Ice Beam', 'Protect', 'Shadow Ball']), ...fillerTeam().slice(0, 5)],
    seeds: Array.from({length: 256}, (_, k) =>
      [(k * 7 + 1) & 0xffff, (k * 13 + 5) & 0xffff, (k * 3 + 2) & 0xffff, (k * 29 + 3) & 0xffff]),
    script: [
      {p1: ['aquajet', null], p2: ['imprison', null]},
    ],
    coverage: {},
    verify(fixture) {
      if (!sawVolatile(fixture, 0, 0, 'disable')) return 'Cursed Body never disabled the attacker';
      return null;
    },
  },
];

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
    const session = new ReferenceSession({teams: [teamA, teamB], seed});
    const fixture = {name: `${trial.name}_${seed[3]}`, seed,
      teams: [teamA, teamB].map((team, side) => ({id: `${trial.name}-${side}`, members: team.map(s => ({
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
        const turn = session.battle.turn;
        const scripted = trial.script[turn - 1];
        const wanted = scripted ? (side === 0 ? scripted.p1 : scripted.p2) : null;
        const choice = choose(session, side, wanted);
        if (process.env.PA3_DBG) {
          console.log('DBG turn', turn, 'side', side, 'wanted', JSON.stringify(wanted), 'cmd', choice.command,
            'moves', JSON.stringify(s.active.map(p => p && p.moveSlots.map(m => m.id + (m.disabled ? '(X)' : '')))));
        }
        const result = session.choose(side ? 'p2' : 'p1', choice.command);
        if (!result.accepted) { failure = JSON.stringify({side, turn, wanted, choice, err: s.choice.error, req: s.activeRequest?.active?.map(a => a?.moves?.map(m => `${m.id}:${m.pp}:${m.disabled}`)), world: s.active.map(p => p && p.moveSlots.map(m => `${m.id}:${m.pp}:${m.disabled}`))}); break; }
        fixture.steps.push({side: side ? 'P2' : 'P1', ...choice, expected: compact(session)});
      }
      if (failure) break;
    }
    if (failure) { session.destroy(); lastReason = failure; continue; }
    if (!session.battle.ended) { session.destroy(); lastReason = 'did not complete'; continue; }
    const reason = trial.verify(fixture, session);
    if (reason) { session.destroy(); lastReason = reason; continue; }
    fixture.coverage = trial.coverage ?? {};
    recorded = fixture;
    session.destroy();
    break;
  }
  if (recorded) fixtures.push(recorded);
  else skipped.push({name: trial.name, reason: lastReason ?? 'no seed produced the required behavior'});
}
fs.writeFileSync(new URL('../data/more_disable_family.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify({trials: TRIALS.length, fixtures: fixtures.length, skipped: skipped.length}));
for (const s of skipped) console.log('SKIP', JSON.stringify(s));
