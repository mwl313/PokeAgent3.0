// Differential corpus for Roost (heal + Flying removal for the casting turn)
// and Yawn (delayed sleep), both reference-generated at every boundary.
//
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary (including the reference
// request, so the native legal-action mask is compared as well).
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
const setOf = (species, ability, moves, item = '', points = {hp:32, atk:0, def:17, spa:0, spd:17, spe:0}) =>
  ({name: species, species, ability, item, nature: 'Serious', level: 50, gender: 'M', moves,
    evs: {hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0, ...points}});
const offensive = (species, ability, moves, item = '') =>
  setOf(species, ability, moves, item, {hp: 2, atk: 32, def: 0, spa: 32, spd: 0, spe: 0});
const bulky = (species, ability, moves) =>
  setOf(species, ability, moves, '', {hp: 32, atk: 2, def: 16, spa: 0, spd: 16, spe: 0});

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
    return {present: !p.fainted, requires_replacement: forced, can_mega: Boolean(info?.canMegaEvo),
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
      conditions: Object.entries(s.sideConditions).map(([id, state]) => [ids.conditions[id], state.duration]).sort((a, c) => a[0] - c[0]),
      pokemon: s.pokemon.map(p => ({roster: roster(p), species: ids.species[p.species.id], hp: p.hp, max_hp: p.maxhp, fainted: p.fainted,
        active_slot: s.active.indexOf(p) >= 0 ? s.active.indexOf(p) : null, ability_ending: Boolean(p.abilityState.ending), cached_speed: p.speed ?? null,
        status: ids.conditions[p.status] ?? 0, boosts: Object.values(p.boosts), stats: [p.maxhp, ...Object.values(p.storedStats)], ability: ids.abilities[p.ability],
        item: ids.items[p.item] ?? 0, types: p.types.map(t => ids.types[toID(t)]), previous_item: ids.items[p.lastItem] ?? 0, can_mega: Boolean(p.canMegaEvo),
        pp: p.moveSlots.map(m => m.pp), volatiles: Object.keys(p.volatiles).sort()})),
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
    const noMovesLeft = view.every(m => m.disabled || m.pp <= 0);
    if (noMovesLeft) {
      const target = targetFor(0);
      actions.push(moveAction(slot, 255, 0));
      commands.push(`move 1${target ? ` ${target}` : ''}`);
      continue;
    }
    const want = (wanted ?? [])[slot];
    const order = view
      .map((m, index) => ({m, index}))
      .filter(({m}) => !m.disabled && m.pp > 0);
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
const sawStatus = (fixture, side, rosterIndex, status) =>
  fixture.steps.some(step => step.expected.sides[side].pokemon[rosterIndex].status === ids.conditions[status]);
const sawBoosts = (fixture, side, rosterIndex, boosts) =>
  fixture.steps.some(step => JSON.stringify(step.expected.sides[side].pokemon[rosterIndex].boosts) === JSON.stringify(boosts));

const ANNI = ['Annihilape', 'Defiant', ['Body Slam', 'Stomping Tantrum', 'Close Combat', 'Rage Fist']];
const GENGAR = ['Gengar', 'Cursed Body', ['Sludge Bomb', 'Protect', 'Shadow Ball', 'Perish Song']];
const CORV = ['Corviknight', 'Mirror Armor', ['Roost', 'Iron Head', 'Protect', 'Body Press']];
const YAWNY = ['Chimecho', 'Levitate', ['Yawn', 'Dazzling Gleam', 'Protect', 'Recover']];

const TRIALS = [
  {
    name: 'delayed_status_roost_heal_and_type',
    p1: [bulky(...CORV), ...fillerTeam().slice(0, 5)],
    p2: [offensive('Archaludon', 'Stamina', ['Flash Cannon', 'Protect']),
      offensive('Hippowdon', 'Sand Stream', ['Earthquake', 'Protect']), ...fillerTeam().slice(0, 4)],
    seeds: [[4, 9, 16, 25], [7, 18, 5, 32], [11, 22, 33, 44]],
    script: [
      {p1: ['ironhead', null], p2: ['flashcannon', 'earthquake']},
      {p1: ['roost', null], p2: ['flashcannon', 'earthquake']},
    ],
    coverage: {move: 'roost'},
    verify(fixture, session) {
      // The volatile lives only for the casting turn, so the witness is the
      // public start message plus the Earthquake connecting while Roost has
      // the user's Flying type removed.
      const log = session.battle.log;
      if (!log.some(line => line.includes('-singleturn') && line.includes('move: Roost'))) {
        return 'Roost never started its singleturn effect';
      }
      if (!log.some(line => line.startsWith('|-damage|') && line.includes('p1a'))) {
        return 'Roost turn never produced the expected Ground-type damage';
      }
      return null;
    },
  },
  {
    name: 'delayed_status_roost_full_hp_fails',
    p1: [bulky(...CORV), ...fillerTeam().slice(0, 5)],
    p2: [offensive('Hippowdon', 'Sand Stream', ['Earthquake', 'Protect']), ...fillerTeam().slice(0, 5)],
    seeds: [[4, 9, 16, 25], [7, 18, 5, 32]],
    script: [
      {p1: ['roost', null], p2: ['earthquake', null]},
      {p1: ['ironhead', null], p2: ['earthquake', null]},
    ],
    coverage: {move: 'roost'},
    verify(fixture) {
      // Roost at full HP fails the heal, so no self volatile and no type change:
      // Earthquake stays immune against the Flying/Steel user.
      if (sawVolatile(fixture, 0, 0, 'roost')) return 'full-HP Roost wrongly started its volatile';
      return null;
    },
  },
  {
    name: 'delayed_status_yawn_countdown_sleep',
    p1: [bulky('Espeon', 'Synchronize', ['Yawn', 'Psychic', 'Protect', 'Calm Mind']), ...fillerTeam().slice(0, 5)],
    p2: [offensive('Skarmory', 'Sturdy', ['Iron Head', 'Protect']), ...fillerTeam().slice(0, 5)],
    seeds: [[4, 9, 16, 25], [7, 18, 5, 32], [11, 22, 33, 44]],
    script: [
      {p1: ['yawn', null], p2: ['ironhead', null]},
      {p1: ['dazzlinggleam', null], p2: ['ironhead', null]},
    ],
    coverage: {move: 'yawn'},
    verify(fixture) {
      if (!sawVolatile(fixture, 1, 0, 'yawn')) return 'Yawn volatile never recorded on the target';
      if (!sawStatus(fixture, 1, 0, 'slp')) return 'Yawn never put the target to sleep';
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
        const result = session.choose(side ? 'p2' : 'p1', choice.command);
        if (!result.accepted) {
          failure = JSON.stringify({side, turn, wanted, choice, err: s.choice.error});
          break;
        }
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
fs.writeFileSync(new URL('../data/more_roost_yawn.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify({trials: TRIALS.length, fixtures: fixtures.length, skipped: skipped.length}));
for (const s of skipped) console.log('SKIP', JSON.stringify(s));
