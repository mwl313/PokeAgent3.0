// Development-only corpus for Beat Up's `onModifyMove` party capture and its
// per-hit `basePowerCallback`:
// - the action's hit count is the number of party members that are the user or
//   are alive and free of a major status,
// - each hit's power is `5 + floor(setSpecies.baseStats.atk / 10)` of the next
//   captured member, consumed in reference `side.pokemon` order.
// Every fixture is a complete legal synthetic reference battle recorded at
// every decision boundary.
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
const fragile = (species, ability, moves, item = '') =>
  ({...setOf(species, ability, moves, item, {hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0}),
    nature: 'Bashful'});
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

// `wanted` entries are a move id, `null` for the auto choice, or
// `{move, target}` for an explicitly targeted move.
function choose(session, sideIndex, wanted) {
  const b = session.battle, side = b.sides[sideIndex], req = side.activeRequest;
  if (side.requestState === 'teampreview') {
    const preview = wanted?.preview ?? [0, 1, 2, 3];
    return {actions: preview.map((r, i) => select('Pick', i, r)),
      command: `team ${preview.map(r => r + 1).join('')}`};
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
    const wantedId = typeof want === 'string' ? want : want?.move;
    const order = view
      .map((m, index) => ({m, index}))
      .filter(({m}) => servedUsable(m));
    const preferred = wantedId
      ? order.filter(({m}) => m.id === wantedId).concat(order.filter(({m}) => m.id !== wantedId))
      : order.filter(({m}) => dex.moves.get(m.id).category !== 'Status')
        .concat(order.filter(({m}) => dex.moves.get(m.id).category === 'Status'));
    const chosenMove = preferred.find(({index}) => (want?.target ?? targetFor(index)) !== null) ?? preferred[0];
    const index = chosenMove ? chosenMove.index : 0;
    const target = chosenMove ? (want?.target ?? targetFor(chosenMove.index)) : 0;
    actions.push(moveAction(slot, index, target ?? 0));
    commands.push(`move ${index + 1}${target ? ` ${target}` : ''}`);
  }
  return {actions, command: commands.join(', ')};
}

const logHas = (session, pattern) => session.battle.log.some(line => pattern.test(line));
const hitCounts = session => session.battle.log
  .filter(line => line.startsWith('|-hitcount|'))
  .map(line => line.split('|').filter(Boolean).slice(1));

// The Beat Up user is Whimsicott (the only frozen-pool Beat Up user whose
// abilities are all ported). Its teammates double as the per-hit base-power
// witnesses: 135 / 110 / 109 / 60 / 50 base Attack give five distinct powers.
const beatUpTeam = () => [
  setOf('Whimsicott', 'Prankster', ['Beat Up', 'Protect', 'Tailwind']),
  offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
  offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
  offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
  offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
  fragile('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
];
const foeTeam = () => [
  setOf('Hydreigon', 'Levitate', ['Dark Pulse', 'Thunder Wave', 'Protect'], 'Life Orb'),
  offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
  offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
  offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
  offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
  fragile('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
];

const TRIALS = [
  {
    name: 'beatup_four_hits_all_healthy',
    // Four healthy, status-free members are brought in the 1234 order, so the
    // per-hit powers follow Whimsicott (11), Metagross (18), Snorlax (16) and
    // Torterra (15) - a mis-ordered shift cannot produce the same damage.
    preview: [0, 1, 2, 3],
    script: [{p1: ['beatup', 'protect'], p2: [null, null]}],
    coverage: {move: 'beatup'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Beat Up\|/)) return 'Beat Up never executed';
      const counts = hitCounts(session);
      if (!counts.some(parts => parts[0] === 'p2a: s0' && parts[1] === '4')) {
        return `expected 4 hits, saw ${JSON.stringify(counts)}`;
      }
      const damage = session.battle.log.filter(line => line.startsWith('|-damage|p2a: s0|'));
      if (damage.length < 4) return `expected four damage lines, saw ${damage.length}`;
      return null;
    },
  },
  {
    name: 'beatup_skips_statused_ally',
    // p2 paralyzes the Metagross in the second active slot on turn one; the
    // statused member drops out of the party filter, so Beat Up lands three
    // hits on turn two.
    preview: [0, 1, 2, 3],
    script: [
      {p1: ['protect', null], p2: [{move: 'thunderwave', target: 2}, 'protect']},
      {p1: ['beatup', 'protect'], p2: [null, null]},
    ],
    coverage: {move: 'beatup'},
    verify(fixture, session) {
      const paralyzed = fixture.steps.some(step => step.expected.sides[0].pokemon
        .some(p => p.status === ids.conditions.par));
      if (!paralyzed) return 'no party member ended up paralyzed';
      const counts = hitCounts(session);
      if (!counts.some(parts => parts[0] === 'p2a: s0' && parts[1] === '3')) {
        return `expected 3 hits after the status skip, saw ${JSON.stringify(counts)}`;
      }
      return null;
    },
  },
  {
    name: 'beatup_skips_fainted_ally',
    // The fragile Chimecho is knocked out on turn one; Beat Up on turn two
    // consumes only the three remaining members.
    preview: [0, 5, 1, 2],
    script: [
      {p1: ['protect', null], p2: [{move: 'darkpulse', target: 2}, 'protect']},
      {p1: ['beatup', 'protect'], p2: [null, null]},
    ],
    coverage: {move: 'beatup'},
    verify(fixture, session) {
      const fainted = fixture.steps.some(step => step.expected.sides[0].pokemon
        .some(p => p.fainted && p.roster !== 0));
      if (!fainted) return 'no ally fainted before Beat Up';
      const counts = hitCounts(session);
      if (!counts.some(parts => parts[0] === 'p2a: s0' && parts[1] === '3')) {
        return `expected 3 hits after the faint skip, saw ${JSON.stringify(counts)}`;
      }
      return null;
    },
  },
];

const fixtures = [];
const skipped = [];
for (const trial of TRIALS) {
  let recorded = null;
  let lastReason = null;
  const seeds = [];
  for (let i = 0; i < 12; i++) seeds.push([3, 5, 7, 900 + i]);
  for (const seed of seeds) {
    const teamA = nameSets(beatUpTeam());
    const teamB = nameSets(foeTeam());
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
    while (!session.battle.ended && fixture.steps.length < 220) {
      for (let side = 0; side < 2; side++) {
        if (session.battle.ended) break;
        const s = session.battle.sides[side];
        if (s.activeRequest?.wait || s.isChoiceDone()) continue;
        const turn = session.battle.turn;
        const scripted = trial.script[turn - 1];
        const wanted = scripted ? (side === 0 ? scripted.p1 : scripted.p2) : null;
        const choice = s.requestState === 'teampreview'
          ? choose(session, side, {preview: trial.preview})
          : choose(session, side, wanted);
        const result = session.choose(side ? 'p2' : 'p1', choice.command);
        if (!result.accepted) { failure = JSON.stringify({side, turn, wanted, choice, err: s.choice.error}); break; }
        fixture.steps.push({side: side ? 'P2' : 'P1', ...choice, expected: compact(session)});
      }
      if (failure) break;
    }
    if (failure) { session.destroy(); lastReason = failure; continue; }
    if (!session.battle.ended) { session.destroy(); lastReason = 'did not complete'; continue; }
    const reason = trial.verify(fixture, session);
    if (reason) {
      if (process.env.DEBUG_BEATUP) {
        fs.appendFileSync(new URL('../../tmp/beatup_debug.log', import.meta.url),
          `=== ${trial.name} seed ${seed[3]}: ${reason}\n${session.battle.log.join('\n')}\n\n`);
      }
      session.destroy();
      lastReason = reason;
      continue;
    }
    fixture.coverage = trial.coverage ?? {};
    recorded = fixture;
    session.destroy();
    break;
  }
  if (recorded) fixtures.push(recorded);
  else skipped.push({name: trial.name, reason: lastReason ?? 'no seed produced the required behavior'});
}
fs.writeFileSync(new URL('../data/more_beatup.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify({trials: TRIALS.length, fixtures: fixtures.length, skipped: skipped.length}));
for (const s of skipped) console.log('SKIP', JSON.stringify(s));
