// Development-only corpus for the volatile selection family: Encore, Taunt,
// Disable, Imprison and the Cursed Body ability.
//
// This file is deliberately NOT named generate_more_*.mjs yet: the exporter
// auto-discovers those names and merges their output into turn-fixtures.json,
// which must only happen once the native engine executes the family. Rename
// this file to `generate_more_disable_family.mjs` when the port lands.
//
// Every fixture is a complete legal synthetic reference battle with the
// pinned Showdown state recorded at every decision boundary (including the
// reference request, so the native legal-action mask is compared as well).
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
    const rows = (info?.moves ?? p.moveSlots).map(m => ({
      id: ids.moves[toID(m.id)] ?? 0, pp: m.pp, disabled: Boolean(m.disabled),
      target: m.target ?? dex.moves.get(m.id).target}));
    return {present: !p.fainted, requires_replacement: forced, can_mega: Boolean(info?.canMegaEvo), moves: rows};
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
    const pick = (predicate, fallback) => {
      let index = p.moveSlots.findIndex(m => predicate(m) && !m.disabled && m.pp > 0);
      if (index < 0 && fallback) index = p.moveSlots.findIndex(m => !m.disabled && m.pp > 0);
      return index;
    };
    const want = (wanted ?? [])[slot];
    let index = want
      ? pick(m => m.id === want, true)
      : pick(m => dex.moves.get(m.id).category !== 'Status', true);
    if (index < 0) index = 0;
    const slotMove = p.moveSlots[index];
    const noMovesLeft = p.moveSlots.every(m => m.disabled || m.pp <= 0);
    let target = 0;
    if (noMovesLeft) {
      // The reference replaces the request with a single Struggle entry.
      actions.push(moveAction(slot, 0, 0));
      commands.push('move 1');
      continue;
    }
    const wantsTarget = b.actions.targetTypeChoices(slotMove.target) &&
      (slotMove.target === 'normal' || slotMove.target === 'any' || slotMove.target === 'adjacentAllyOrSelf');
    if (wantsTarget) {
      const foes = sideIndex === 0 ? [1, 2, -1, -2] : [-1, -2, 1, 2];
      const loc = foes.find(l => b.validTargetLoc(l, p, slotMove.target) &&
        (() => { const other = p.getAtLoc(l); return other && other.side !== p.side; })()) ?? 0;
      target = loc;
    }
    actions.push(moveAction(slot, index, target));
    commands.push(`move ${index + 1}${target ? ` ${target}` : ''}`);
  }
  return {actions, command: commands.join(', ')};
}

const TRIALS = [
  {
    name: 'disable_family_encore_after_move',
    p1: [setOf('Samurott', 'Shell Armor', ['Aqua Jet', 'Protect', 'Ice Beam']), ...fillerTeam().slice(0, 5)],
    p2: [setOf('Alakazam', 'Synchronize', ['Encore', 'Taunt', 'Disable', 'Imprison']), ...fillerTeam().slice(0, 5)],
    seeds: [[7, 8, 9, 10], [11, 22, 33, 44], [101, 102, 103, 104]],
    script: [
      {p1: ['aquajet', null], p2: ['encore', null]},
    ],
    verify(fixture, session) {
      const log = session.battle.log.join('\n');
      if (!/\|-start\|p1a: s0m0\|Encore/.test(log)) return 'Encore never applied';
      const encored = session.battle.sides[0].pokemon[0];
      if (!encored.volatiles.encore) return 'Encore volatile missing';
      return null;
    },
  },
  {
    name: 'disable_family_encore_before_move',
    p1: [setOf('Samurott', 'Shell Armor', ['Aqua Jet', 'Protect', 'Ice Beam']), ...fillerTeam().slice(0, 5)],
    // Farigiraf is faster than Samurott so Encore lands before Samurott acts.
    p2: [setOf('Farigiraf', 'Armor Tail', ['Encore', 'Twin Beam', 'Psychic']), ...fillerTeam().slice(0, 5)],
    seeds: [[3, 14, 15, 92], [55, 66, 77, 88]],
    script: [
      {p1: ['aquajet', null], p2: [null, null]},
      {p1: ['protect', null], p2: ['encore', null]},
    ],
    verify(fixture, session) {
      const encored = session.battle.sides[0].pokemon[0];
      if (!encored.volatiles.encore) return 'Encore volatile missing';
      // Duration must reflect the "target has not moved yet" adjustment.
      if ((encored.volatiles.encore.duration ?? 0) !== 2) {
        return `unexpected encore duration ${encored.volatiles.encore.duration}`;
      }
      return null;
    },
  },
  {
    name: 'disable_family_taunt_status_lock',
    p1: [setOf('Samurott', 'Shell Armor', ['Protect', 'Aqua Jet', 'Ice Beam']), ...fillerTeam().slice(0, 5)],
    p2: [setOf('Alakazam', 'Synchronize', ['Taunt', 'Encore', 'Disable', 'Imprison']), ...fillerTeam().slice(0, 5)],
    seeds: [[21, 22, 23, 24], [31, 32, 33, 34]],
    script: [
      {p1: ['aquajet', null], p2: ['taunt', null]},
      {p1: ['aqaujet', null], p2: [null, null]},
    ],
    verify(fixture, session) {
      const log = session.battle.log.join('\n');
      if (!/\|-start\|p1a: s0m0\|move: Taunt/.test(log)) return 'Taunt never applied';
      const taunted = session.battle.sides[0].pokemon[0];
      if (!taunted.volatiles.taunt) return 'Taunt volatile missing';
      return null;
    },
  },
  {
    name: 'disable_family_disable_last_move',
    p1: [setOf('Samurott', 'Shell Armor', ['Aqua Jet', 'Protect', 'Ice Beam']), ...fillerTeam().slice(0, 5)],
    p2: [setOf('Alakazam', 'Synchronize', ['Disable', 'Taunt', 'Encore', 'Imprison']), ...fillerTeam().slice(0, 5)],
    seeds: [[41, 42, 43, 44], [51, 52, 53, 54]],
    script: [
      {p1: ['aquajet', null], p2: [null, null]},
      {p1: ['icebeam', null], p2: ['disable', null]},
    ],
    verify(fixture, session) {
      const log = session.battle.log.join('\n');
      if (!/\|-start\|p1a: s0m0\|Disable\|Ice Beam/.test(log)) return 'Disable never applied';
      const target = session.battle.sides[0].pokemon[0];
      if (!target.volatiles.disable) return 'Disable volatile missing';
      return null;
    },
  },
  {
    name: 'disable_family_imprison_shared_moves',
    p1: [setOf('Samurott', 'Shell Armor', ['Ice Beam', 'Protect', 'Aqua Jet']), ...fillerTeam().slice(0, 5)],
    p2: [setOf('Froslass', 'Cursed Body', ['Imprison', 'Ice Beam', 'Protect']), ...fillerTeam().slice(0, 5)],
    seeds: [[3, 4, 5, 6], [61, 62, 63, 64]],
    script: [
      {p1: ['protect', null], p2: ['imprison', null]},
      {p1: ['aquajet', null], p2: ['icebeam', null]},
    ],
    verify(fixture, session) {
      const log = session.battle.log.join('\n');
      if (!/\|-start\|p2a: s1m0\|move: Imprison/.test(log)) return 'Imprison never applied';
      const user = session.battle.sides[1].pokemon[0];
      if (!user.volatiles.imprison) return 'Imprison volatile missing';
      // Both foes lose the shared moves; the ally slot shares Protect.
      const ally = session.battle.sides[0].pokemon[1];
      const allyProtect = ally.moveSlots.find(m => m.id === 'protect');
      if (allyProtect && !allyProtect.disabled) return 'Imprison did not disable the ally shared move';
      return null;
    },
  },
  {
    name: 'disable_family_cursed_body_disable',
    p1: [setOf('Samurott', 'Shell Armor', ['Aqua Jet', 'Protect', 'Ice Beam']), ...fillerTeam().slice(0, 5)],
    p2: [setOf('Froslass', 'Cursed Body', ['Imprison', 'Ice Beam', 'Protect']), ...fillerTeam().slice(0, 5)],
    seeds: Array.from({length: 256}, (_, k) =>
      [(k * 7 + 1) & 0xffff, (k * 13 + 5) & 0xffff, (k * 3 + 2) & 0xffff, (k * 29 + 3) & 0xffff]),
    script: [
      {p1: ['aquajet', null], p2: ['imprison', null]},
    ],
    verify(fixture, session) {
      const log = session.battle.log.join('\n');
      if (!/\[from\] ability: Cursed Body/.test(log)) return 'Cursed Body never procced';
      const attacker = session.battle.sides[0].pokemon[0];
      if (!attacker.volatiles.disable) return 'Cursed Body disable missing';
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
        if (!result.accepted) { failure = JSON.stringify({side, turn, wanted, choice, err: s.choice.error}); break; }
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
fs.writeFileSync(new URL('../data/pending_disable_family.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify({trials: TRIALS.length, fixtures: fixtures.length, skipped: skipped.length}));
for (const s of skipped) console.log('SKIP', JSON.stringify(s));
