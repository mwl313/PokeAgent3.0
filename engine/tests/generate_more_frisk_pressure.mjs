// Development-only corpus for two entries of the legal ability tail:
// - Frisk announces every active foe's held item on entry (public knowledge).
// - Pressure charges one extra PP for each opposing Pressure holder the move
//   actually targets, and never for an ally it protects.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
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
const offensive = (species, ability, moves, item = '', gender = 'M') =>
  ({...setOf(species, ability, moves, item, {hp: 2, atk: 32, def: 0, spa: 32, spd: 0, spe: 0}), gender});
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
    const wish = (wanted ?? [])[slot];
    const named = typeof wish === 'object' && wish?.switch
      ? side.pokemon.find(x => x.name === wish.switch)
      : null;
    if (side.requestState === 'switch') {
      if (!req.forceSwitch[slot]) { commands.push('pass'); continue; }
      const reserve = named && !named.fainted && !side.active.includes(named)
        ? named
        : bench.find(x => !chosen.has(x));
      if (!reserve) { actions.push(select('Pass', slot)); commands.push('pass'); continue; }
      chosen.add(reserve);
      actions.push(select('Switch', slot, roster(reserve)));
      commands.push(`switch ${side.pokemon.indexOf(reserve) + 1}`);
      continue;
    }
    if (!p || p.fainted) { actions.push(select('Pass', slot)); commands.push('pass'); continue; }
    if (named) {
      const reserve = bench.find(x => x === named);
      if (reserve && !chosen.has(reserve)) {
        chosen.add(reserve);
        actions.push(select('Switch', slot, roster(reserve)));
        commands.push(`switch ${side.pokemon.indexOf(reserve) + 1}`);
        continue;
      }
    }
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
    const wantedId = typeof wish === 'string' ? wish : wish?.move;
    const order = view
      .map((m, index) => ({m, index}))
      .filter(({m}) => servedUsable(m));
    const preferred = wantedId
      ? order.filter(({m}) => m.id === wantedId).concat(order.filter(({m}) => m.id !== wantedId))
      : order.filter(({m}) => dex.moves.get(m.id).category !== 'Status')
        .concat(order.filter(({m}) => dex.moves.get(m.id).category === 'Status'));
    const chosenMove = preferred.find(({index}) => (wish?.target ?? targetFor(index)) !== null) ?? preferred[0];
    const index = chosenMove ? chosenMove.index : 0;
    const target = chosenMove ? (wish?.target ?? targetFor(chosenMove.index)) : 0;
    actions.push(moveAction(slot, index, target ?? 0));
    commands.push(`move ${index + 1}${target ? ` ${target}` : ''}`);
  }
  return {actions, command: commands.join(', ')};
}

const logHas = (session, pattern) => session.battle.log.some(line => pattern.test(line));
const countLog = (session, pattern) => session.battle.log.filter(line => pattern.test(line)).length;
const FILL = [
  ['Metagross', 'Clear Body', ['Iron Head', 'Protect']],
  ['Milotic', 'Competitive', ['Surf', 'Protect']],
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']],
  ['Falinks', 'Battle Armor', ['Smart Strike', 'Protect']],
  ['Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']],
];
const fillerAfter = (head, count = 5) => [
  head,
  ...[...FILL, ...FILL_EXTRA].filter(([s]) => s !== head.species).slice(0, count)
    .map(([s, a, m]) => offensive(s, a, m)),
];
const FILL_EXTRA = [
  ['Snorlax', 'Thick Fat', ['Body Slam', 'Protect']],
  ['Incineroar', 'Intimidate', ['Knock Off', 'Protect']],
];
const foeTeam = () => [
  offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
  offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
  offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
  offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
  offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
  offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
];
const foeWith = head => [head, ...foeTeam().filter(p => p.species !== head.species)].slice(0, 6);

const monAt = (fixture, side, slot) => fixture.steps.flatMap(step => step.expected.sides[side].pokemon
  .filter(p => p.roster === slot));
const everPp = (fixture, side, slot, index, pp) => monAt(fixture, side, slot)
  .some(p => p.pp[index] === pp);
// The Champions mod overrides some base PP values, so the surcharge assertions
// read the reference's own starting PP from the first recorded boundary
// instead of the mainline dex value.
const basePp = (fixture, side, slot, index) => monAt(fixture, side, slot)[0].pp[index];

const TRIALS = [
  {
    name: 'pressure_charges_extra_pp_single_target',
    // Iron Head has 15 PP; a single-target use against a Pressure holder pays 2.
    p1: () => fillerAfter(offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect'])),
    p2: () => foeWith(offensive('Absol', 'Pressure', ['Night Slash', 'Protect'])),
    script: [{p1: ['ironhead', 'protect'], p2: ['nightslash', 'protect']}],
    coverage: {ability: 'pressure'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Iron Head\|/)) return 'Iron Head never executed';
      if (!logHas(session, /\|-ability\|p2a: s0\|Pressure/)) return 'Pressure was never announced';
      if (!everPp(fixture, 0, 0, 0, basePp(fixture, 0, 0, 0) - 2)) {
        return 'the single-target use did not pay the Pressure surcharge';
      }
      return null;
    },
  },
  {
    name: 'pressure_charges_per_spread_holder',
    // Dazzling Gleam has 10 PP. One opposing holder among the two targets is +1.
    p1: () => fillerAfter(setOf('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect'])),
    p2: () => foeWith(offensive('Absol', 'Pressure', ['Night Slash', 'Protect'])),
    script: [{p1: ['dazzlinggleam', 'protect'], p2: ['protect', 'protect']}],
    coverage: {ability: 'pressure'},
    verify(fixture, session) {
      if (!everPp(fixture, 0, 0, 0, basePp(fixture, 0, 0, 0) - 2)) {
        return 'the spread use did not pay exactly one Pressure surcharge';
      }
      return null;
    },
  },
  {
    name: 'pressure_charges_per_distinct_holder',
    // Both opposing actives hold Pressure: a spread use pays 3 PP.
    p1: () => fillerAfter(setOf('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect'])),
    p2: () => [
      offensive('Absol', 'Pressure', ['Night Slash', 'Protect']),
      offensive('Corviknight', 'Pressure', ['Iron Head', 'Protect']),
      ...foeTeam().filter(p => !['Absol', 'Corviknight'].includes(p.species)).slice(0, 4),
    ],
    script: [{p1: ['dazzlinggleam', 'protect'], p2: ['protect', 'protect']}],
    coverage: {ability: 'pressure'},
    verify(fixture, session) {
      if (countLog(session, /\|-ability\|p2[ab]: s\d\|Pressure/) < 2) {
        return 'both Pressure holders were not announced';
      }
      if (!everPp(fixture, 0, 0, 0, basePp(fixture, 0, 0, 0) - 3)) {
        return 'the spread use did not pay one surcharge per distinct holder';
      }
      return null;
    },
  },
  {
    name: 'pressure_spares_ally_target',
    // Helping Hand targets the user's own Pressure ally: exactly 1 PP.
    p1: () => [
      offensive('Absol', 'Pressure', ['Night Slash', 'Protect']),
      offensive('Snorlax', 'Thick Fat', ['Helping Hand', 'Protect', 'Body Slam']),
      ...FILL.slice(0, 4).map(([s, a, m]) => offensive(s, a, m)),
    ],
    p2: () => foeTeam(),
    script: [{p1: [{move: 'nightslash', target: 1}, {move: 'helpinghand', target: -1}], p2: ['protect', 'protect']}],
    coverage: {ability: 'pressure'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1b: s1\|Helping Hand\|p1a: s0/)) {
        return 'the ally-targeted Helping Hand never executed';
      }
      if (!everPp(fixture, 0, 1, 0, basePp(fixture, 0, 1, 0) - 1)) {
        return 'the ally-targeted move paid a Pressure surcharge';
      }
      return null;
    },
  },
  {
    name: 'frisk_announces_foe_items',
    p1: () => fillerAfter(setOf('Noivern', 'Frisk', ['Hurricane', 'Protect'])),
    p2: () => [
      setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect'], 'Leftovers'),
      setOf('Milotic', 'Competitive', ['Surf', 'Protect'], 'Sitrus Berry'),
      ...foeTeam().filter(p => !['Metagross', 'Milotic'].includes(p.species)).slice(0, 4),
    ],
    script: [{p1: ['hurricane', 'protect'], p2: ['ironhead', 'surf']}],
    coverage: {ability: 'frisk'},
    verify(fixture, session) {
      const announcements = countLog(session, /\|-item\|.*\[from\] ability: Frisk/);
      if (announcements !== 2) return `Frisk announced ${announcements} items instead of 2`;
      if (!logHas(session, /\[of\] p1a: s0/)) return 'the announcement attributed the wrong holder';
      return null;
    },
  },
  {
    name: 'frisk_skips_itemless_foes',
    p1: () => fillerAfter(setOf('Noivern', 'Frisk', ['Hurricane', 'Protect'])),
    p2: () => [
      setOf('Metagross', 'Clear Body', ['Iron Head', 'Protect'], 'Leftovers'),
      offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
      ...foeTeam().filter(p => !['Metagross', 'Milotic'].includes(p.species)).slice(0, 4),
    ],
    script: [{p1: ['hurricane', 'protect'], p2: ['ironhead', 'surf']}],
    coverage: {ability: 'frisk'},
    verify(fixture, session) {
      const announcements = countLog(session, /\|-item\|.*\[from\] ability: Frisk/);
      if (announcements !== 1) return `Frisk announced ${announcements} items instead of 1`;
      if (!logHas(session, /\|-item\|p2a: s0\|Leftovers/)) return 'the wrong foe item was announced';
      return null;
    },
  },
  {
    name: 'frisk_silent_without_items',
    p1: () => fillerAfter(setOf('Noivern', 'Frisk', ['Hurricane', 'Protect'])),
    p2: () => foeTeam(),
    script: [{p1: ['hurricane', 'protect'], p2: ['protect', 'protect']}],
    coverage: {ability: 'frisk'},
    verify(fixture, session) {
      if (logHas(session, /Frisk/)) return 'Frisk announced itself without any foe item';
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
  for (let i = 0; i < 12; i++) seeds.push([3, 5, 7, 3000 + i]);
  for (const seed of seeds) {
    const teamA = nameSets(trial.p1());
    const teamB = nameSets(trial.p2());
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
          ? choose(session, side, {preview: [0, 1, 2, 3]})
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
      if (process.env.DEBUG_FRISK_PRESSURE) {
        fs.appendFileSync(new URL('../../tmp/frisk_pressure_debug.log', import.meta.url),
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
fs.writeFileSync(new URL('../data/more_frisk_pressure.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify({trials: TRIALS.length, fixtures: fixtures.length, skipped: skipped.length}));
for (const s of skipped) console.log('SKIP', JSON.stringify(s));
