// Development-only corpus for Poltergeist: the move-owned `onTry` gate (no
// item -> the move fails without an accuracy draw) and the `onTryHit` item
// reveal, which runs only when the move actually connects (not on Protect,
// type immunity or a miss). Every fixture is a complete legal synthetic
// reference battle recorded at every decision boundary.
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

const logHas = (session, pattern) => session.battle.log.some(line => pattern.test(line));
const REVEAL = /\|-activate\|p2a: s0\|move: Poltergeist\|Expert Belt/;
const HIT = /\|move\|p1a: s0\|Poltergeist\|p2a: s0$/;

const TRIALS = [
  {
    name: 'poltergeist_reveals_item_on_hit',
    // The Froslass lead must actually connect for the reveal to happen; the
    // seed sweep lets the generator find a hit (90% accuracy).
    p1: [setOf('Froslass', 'Cursed Body', ['Poltergeist', 'Ice Beam', 'Protect']), ...fillerTeam()],
    p2: [setOf('Trevenant', 'Natural Cure', ['Poltergeist', 'Protect'], 'Expert Belt'), ...fillerTeam()],
    seeds: [[3, 5, 7, 101], [11, 13, 17, 102], [23, 29, 31, 103], [41, 43, 47, 104]],
    script: [{p1: ['poltergeist', null], p2: ['poltergeist', null]}],
    coverage: {move: 'poltergeist'},
    verify(fixture, session) {
      if (!logHas(session, HIT)) return 'Poltergeist never connected with the item holder';
      if (!logHas(session, REVEAL)) return 'the hit never revealed the target item';
      return null;
    },
  },
  {
    name: 'poltergeist_fails_without_item',
    // The same move against an itemless target must fail at the Try gate and
    // consume no accuracy draw; the corpus RNG compare pins that.
    p1: [setOf('Froslass', 'Cursed Body', ['Poltergeist', 'Ice Beam', 'Protect']), ...fillerTeam()],
    p2: [setOf('Trevenant', 'Natural Cure', ['Poltergeist', 'Protect']), ...fillerTeam()],
    seeds: [[3, 5, 7, 111], [11, 13, 17, 112]],
    script: [{p1: ['poltergeist', null], p2: ['poltergeist', null]}],
    coverage: {move: 'poltergeist'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Poltergeist\|\|\[still\]/)) {
        return 'Poltergeist never failed against the itemless target';
      }
      if (logHas(session, REVEAL)) return 'the failed move still revealed an item';
      return null;
    },
  },
  {
    name: 'poltergeist_blocked_by_protect',
    // A protected item holder keeps its item hidden: the move-owned TryHit
    // never runs, which the native knowledge assertion in tests/poltergeist.rs
    // checks on top of the corpus boundary comparison.
    p1: [setOf('Froslass', 'Cursed Body', ['Poltergeist', 'Ice Beam', 'Protect']), ...fillerTeam()],
    p2: [setOf('Trevenant', 'Natural Cure', ['Poltergeist', 'Protect'], 'Expert Belt'), ...fillerTeam()],
    seeds: [[3, 5, 7, 121], [11, 13, 17, 122]],
    script: [{p1: ['poltergeist', null], p2: ['protect', null]}],
    coverage: {move: 'poltergeist'},
    verify(fixture, session) {
      if (!logHas(session, /\|-activate\|p2a: s0\|move: Protect/)) {
        return 'Protect never blocked Poltergeist';
      }
      if (logHas(session, REVEAL)) return 'the protected move still revealed the item';
      return null;
    },
  },
  {
    name: 'poltergeist_type_immunity_keeps_item_hidden',
    // Ghost vs Normal: the type immunity precedes the damage step, so the
    // reveal must not happen even though the target holds an item.
    p1: [setOf('Froslass', 'Cursed Body', ['Poltergeist', 'Ice Beam', 'Protect']), ...fillerTeam()],
    p2: [setOf('Snorlax', 'Thick Fat', ['Body Slam', 'Protect'], 'Expert Belt'), ...fillerTeam()],
    seeds: [[3, 5, 7, 131], [11, 13, 17, 132]],
    script: [{p1: ['poltergeist', null], p2: [null, null]}],
    coverage: {move: 'poltergeist'},
    verify(fixture, session) {
      if (!logHas(session, /\|-immune\|p2a: s0/)) return 'the Normal-type target was not immune';
      if (logHas(session, REVEAL)) return 'the immune move still revealed the item';
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
        if (!result.accepted) { failure = JSON.stringify({side, turn, wanted, choice, err: s.choice.error}); break; }
        fixture.steps.push({side: side ? 'P2' : 'P1', ...choice, expected: compact(session)});
      }
      if (failure) break;
    }
    if (failure) { session.destroy(); lastReason = failure; continue; }
    if (process.env.PA3_DBG) {
      console.log('LOG', fixture.name, JSON.stringify(session.battle.log.filter(l =>
        l.includes('Poltergeist') || l.includes('-fail') || l.includes('-activate') || l.includes('-immune') || l.includes('[still]'))));
    }
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
fs.writeFileSync(new URL('../data/more_poltergeist.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify({trials: TRIALS.length, fixtures: fixtures.length, skipped: skipped.length}));
for (const s of skipped) console.log('SKIP', JSON.stringify(s));
