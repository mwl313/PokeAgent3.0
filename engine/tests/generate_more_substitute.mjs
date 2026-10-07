// Differential corpus for Substitute (move 612): the HP cost and fail gates,
// the decoy's primary-hit interception of damage and status, `bypasssub`
// pass-through, spread secondaries, multi-hit depletion, fixed-damage clamps,
// and decoy-driven recoil/drain.
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
const bulky = (species, ability, moves, item = '') =>
  setOf(species, ability, moves, item, {hp: 32, atk: 2, def: 16, spa: 0, spd: 16, spe: 0});

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
    // Reference `getMoves(lockedMove, restrictData = isLastActive())`: an
    // Imprison `'hidden'` disable is served as *enabled* for the side's last
    // active Pokemon (its execution-time `onFoeBeforeMove` gate then refuses
    // the move), and as disabled for every other slot. The Struggle override
    // applies only when the served request has no usable entry at all.
    const lastActive = p.isLastActive();
    const servedUsable = m => (!m.disabled || (m.disabled === 'hidden' && lastActive)) && m.pp > 0;
    const noMovesLeft = view.every(m => !servedUsable(m));
    if (noMovesLeft) {
      actions.push(moveAction(slot, 255, 0));
      commands.push('move 1');
      continue;
    }
    const want = (wanted ?? [])[slot];
    if (want === 'switch') {
      const reserve = bench.find(x => !chosen.has(x));
      if (reserve) {
        chosen.add(reserve);
        actions.push(select('Switch', slot, roster(reserve)));
        commands.push(`switch ${side.pokemon.indexOf(reserve) + 1}`);
        continue;
      }
    }
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
const mon = (state, side, index) => state.sides[side].pokemon[index];
const hasSub = p => p.volatiles.includes('substitute');
// A resolved turn pairs the P1 choice boundary (pre-turn) with the following
// P2 choice boundary (post-turn): `expected.turn` advances by exactly one.
const resolvedTurns = fixture => {
  const turns = [];
  for (let i = 0; i + 1 < fixture.steps.length; i++) {
    const before = fixture.steps[i], after = fixture.steps[i + 1];
    if (before.side === 'P1' && after.side === 'P2' &&
        after.expected.turn === before.expected.turn + 1) {
      turns.push({before: before.expected, after: after.expected});
    }
  }
  return turns;
};
const sawVolatile = (fixture, side, index, volatile) =>
  fixture.steps.some(step => mon(step.expected, side, index).volatiles.includes(volatile));

const SUB = moves => bulky('Snorlax', 'Thick Fat', ['Substitute', 'Body Slam', 'Rain Dance', 'Protect', ...moves]);

const TRIALS = [
  {
    name: 'substitute_absorbs_hit_blocks_toxic_and_refuses_reuse',
    // Goodra's Toxic would poison this Thick Fat Snorlax if the decoy did not
    // intercept it; the second Substitute attempt must fail without paying.
    p1: [SUB([]), ...fillerTeam()],
    p2: [offensive('Metagross', 'Clear Body', ['Meteor Mash', 'Protect']),
      bulky('Goodra', 'Sap Sipper', ['Toxic', 'Protect']),
      ...FILLERS.filter(([s]) => !['Metagross', 'Goodra'].includes(s)).map(([s, a, m]) => offensive(s, a, m))],
    seeds: [[5, 10, 20, 40], [7, 14, 28, 56], [11, 22, 44, 88]],
    script: [
      {p1: ['substitute', 'protect'], p2: ['protect', 'protect']},
      {p1: ['substitute', 'protect'], p2: ['protect', 'toxic']},
      {p1: ['raindance', 'protect'], p2: ['meteormash', 'protect']},
      {p1: ['raindance', 'protect'], p2: ['meteormash', 'protect']},
      {p1: ['raindance', 'protect'], p2: ['meteormash', 'protect']},
      {p1: ['raindance', 'protect'], p2: ['meteormash', 'protect']},
    ],
    coverage: {move: 'substitute'},
    verify(fixture) {
      if (!sawVolatile(fixture, 0, 0, 'substitute')) return 'the decoy was never created';
      for (const step of fixture.steps) {
        const p = mon(step.expected, 0, 0);
        if (p.status !== 0) return 'Toxic poisoned the decoy owner through the Substitute';
      }
      const turns = resolvedTurns(fixture);
      const createdIndex = turns.findIndex(t => !hasSub(mon(t.before, 0, 0)) && hasSub(mon(t.after, 0, 0)));
      if (createdIndex < 0) return 'no resolved boundary created the decoy';
      const created = turns[createdIndex];
      const subHp = mon(created.after, 0, 0).hp;
      if (mon(created.before, 0, 0).hp - subHp !== Math.floor(mon(created.after, 0, 0).stats[0] / 4)) {
        return 'the decoy cost was not floor(maxHP/4)';
      }
      // Turn 2: the repeated Substitute must fail without paying and Toxic
      // must be absorbed with no status.
      const reuse = turns[createdIndex + 1];
      if (!reuse || !hasSub(mon(reuse.before, 0, 0)) || !hasSub(mon(reuse.after, 0, 0))) {
        return 'the decoy did not survive the reuse turn';
      }
      if (mon(reuse.after, 0, 0).hp !== mon(reuse.before, 0, 0).hp) {
        return 'the failed reuse attempt paid a second decoy cost';
      }
      // Turn 3: Meteor Mash must not overflow onto the owner even if the hit
      // breaks the decoy.
      const absorbed = turns[createdIndex + 2];
      if (!absorbed || !hasSub(mon(absorbed.before, 0, 0))) {
        return 'the decoy did not reach the absorbed hit';
      }
      if (mon(absorbed.after, 0, 0).hp !== mon(absorbed.before, 0, 0).hp) {
        return 'Meteor Mash overflowed onto the decoy owner';
      }
      const broke = turns.slice(createdIndex + 2).some(t => hasSub(mon(t.before, 0, 0)) && !hasSub(mon(t.after, 0, 0)));
      if (!broke) return 'the decoy never broke';
      return null;
    },
  },
  {
    name: 'substitute_blocks_spread_secondary',
    // Icy Wind's speed drop is a secondary: the decoy owner keeps +0, the
    // unsubstituted ally takes -1.
    p1: [SUB([]), bulky('Goodra', 'Sap Sipper', ['Surf', 'Protect']),
      ...FILLERS.slice(0, 4).map(([s, a, m]) => offensive(s, a, m))],
    p2: [bulky('Milotic', 'Competitive', ['Icy Wind', 'Protect']),
      ...FILLERS.filter(([s]) => s !== 'Milotic').map(([s, a, m]) => offensive(s, a, m)),
      offensive('Reuniclus', 'Overcoat', ['Iron Defense', 'Protect'])],
    seeds: [[3, 6, 12, 24], [13, 26, 52, 104], [17, 34, 68, 136]],
    script: [
      {p1: ['substitute', 'protect'], p2: ['protect', 'protect']},
      {p1: ['raindance', 'protect'], p2: ['icywind', 'protect']},
      {p1: ['raindance', 'protect'], p2: ['icywind', 'protect']},
    ],
    coverage: {move: 'substitute'},
    verify(fixture) {
      if (!sawVolatile(fixture, 0, 0, 'substitute')) return 'the decoy was never created';
      const blocked = resolvedTurns(fixture).some(t =>
        hasSub(mon(t.before, 0, 0)) && mon(t.after, 0, 0).boosts[4] === 0 &&
        mon(t.after, 0, 1).boosts[4] === -1);
      if (!blocked) return 'the decoy owner took Icy Wind speed drops or the ally did not';
      return null;
    },
  },
  {
    name: 'substitute_absorbs_multi_hit_move',
    // Population Bomb's hits are all consumed by the decoy; the owner's HP
    // only moves (if the decoy breaks) in the boundary where it disappears.
    p1: [SUB([]), ...fillerTeam()],
    p2: [offensive('Maushold', 'Friend Guard', ['Population Bomb', 'Protect']), ...fillerTeam().slice(0, 5)],
    seeds: [[2, 4, 8, 16], [9, 18, 36, 72], [23, 46, 92, 184]],
    script: [
      {p1: ['substitute', 'protect'], p2: ['protect', 'protect']},
      {p1: ['raindance', 'protect'], p2: ['populationbomb', 'protect']},
      {p1: ['raindance', 'protect'], p2: ['populationbomb', 'protect']},
    ],
    coverage: {move: 'substitute'},
    verify(fixture) {
      if (!sawVolatile(fixture, 0, 0, 'substitute')) return 'the decoy was never created';
      const turns = resolvedTurns(fixture);
      const created = turns.find(t => !hasSub(mon(t.before, 0, 0)) && hasSub(mon(t.after, 0, 0)));
      if (!created) return 'no resolved boundary created the decoy';
      const subHp = mon(created.after, 0, 0).hp;
      for (const t of turns) {
        if (hasSub(mon(t.after, 0, 0)) && mon(t.after, 0, 0).hp !== subHp) {
          return 'the decoy owner lost HP while the decoy survived a multi-hit move';
        }
      }
      return null;
    },
  },
  {
    name: 'substitute_fixed_damage_clamp_and_low_hp_fail',
    // Night Shade deals a fixed 50: the first hit is fully clamped to the
    // decoy HP with no overflow onto the owner, and once the owner falls to
    // maxHP/4 the next Substitute fails without paying.
    p1: [bulky('Goodra', 'Sap Sipper', ['Substitute', 'Rain Dance', 'Protect']),
      ...FILLERS.map(([s, a, m]) => offensive(s, a, m))],
    p2: [bulky('Gengar', 'Cursed Body', ['Night Shade', 'Protect']),
      ...FILLERS.filter(([s]) => !['Gengar', 'Metagross'].includes(s)).map(([s, a, m]) => offensive(s, a, m)),
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect'])],
    seeds: [[31, 62, 124, 248], [19, 38, 76, 152], [29, 58, 116, 232]],
    script: [
      {p1: ['substitute', 'protect'], p2: ['protect', 'protect']},
      {p1: ['raindance', 'protect'], p2: ['nightshade', 'protect']},
      {p1: ['raindance', 'protect'], p2: ['nightshade', 'protect']},
      {p1: ['raindance', 'protect'], p2: ['nightshade', 'protect']},
      {p1: ['raindance', 'protect'], p2: ['nightshade', 'protect']},
      {p1: ['substitute', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'substitute'},
    verify(fixture) {
      if (!sawVolatile(fixture, 0, 0, 'substitute')) return 'the decoy was never created';
      const turns = resolvedTurns(fixture);
      const clamped = turns.find(t => hasSub(mon(t.before, 0, 0)) && !hasSub(mon(t.after, 0, 0)));
      if (!clamped) return 'the fixed-damage hit never broke the decoy';
      if (mon(clamped.after, 0, 0).hp !== mon(clamped.before, 0, 0).hp) {
        return 'the breaking fixed-damage hit overflowed onto the decoy owner';
      }
      const failed = turns.some(t => !hasSub(mon(t.before, 0, 0)) && !hasSub(mon(t.after, 0, 0)) &&
        mon(t.before, 0, 0).hp <= Math.floor(mon(t.before, 0, 0).stats[0] / 4) &&
        mon(t.before, 0, 0).hp > 0 &&
        mon(t.after, 0, 0).hp === mon(t.before, 0, 0).hp);
      if (!failed) return 'no low-HP Substitute attempt failed without paying';
      return null;
    },
  },
  {
    name: 'substitute_absorbs_recoil',
    // Double-Edge recoil is driven by the damage the decoy eats; the owner's
    // HP never moves, even when the hit breaks the decoy (no overflow).
    p1: [SUB([]), ...fillerTeam()],
    p2: [bulky('Salamence', 'Intimidate', ['Double-Edge', 'Protect']), ...fillerTeam().slice(0, 5)],
    seeds: [[41, 82, 164, 328], [43, 86, 172, 344], [47, 94, 188, 376]],
    script: [
      {p1: ['substitute', 'protect'], p2: ['protect', 'protect']},
      {p1: ['raindance', 'protect'], p2: ['doubleedge', 'protect']},
      {p1: ['raindance', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'substitute'},
    verify(fixture) {
      if (!sawVolatile(fixture, 0, 0, 'substitute')) return 'the decoy was never created';
      const turns = resolvedTurns(fixture);
      const recoiled = turns.some(t => hasSub(mon(t.before, 0, 0)) &&
        mon(t.after, 0, 0).hp === mon(t.before, 0, 0).hp &&
        mon(t.after, 1, 0).hp < mon(t.before, 1, 0).hp);
      if (!recoiled) return 'Double-Edge recoil was not driven by decoy damage';
      return null;
    },
  },
  {
    name: 'substitute_absorbs_drain_heal',
    // Draining Kiss heals from the damage the decoy eats; the owner's HP never
    // moves and Surf chips the same decoy without touching the owner.
    p1: [SUB([]), bulky('Goodra', 'Sap Sipper', ['Surf', 'Protect']),
      ...FILLERS.filter(([s]) => s !== 'Goodra').slice(0, 4).map(([s, a, m]) => offensive(s, a, m))],
    p2: [bulky('Milotic', 'Competitive', ['Draining Kiss', 'Protect']),
      ...FILLERS.filter(([s]) => s !== 'Milotic').map(([s, a, m]) => offensive(s, a, m)),
      offensive('Reuniclus', 'Overcoat', ['Iron Defense', 'Protect'])],
    seeds: [[67, 134, 268, 536], [71, 142, 284, 568], [73, 146, 292, 584]],
    script: [
      {p1: ['substitute', 'protect'], p2: ['protect', 'protect']},
      {p1: ['raindance', 'surf'], p2: ['drainingkiss', 'protect']},
      {p1: ['raindance', 'protect'], p2: ['drainingkiss', 'protect']},
      {p1: ['raindance', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'substitute'},
    verify(fixture) {
      if (!sawVolatile(fixture, 0, 0, 'substitute')) return 'the decoy was never created';
      const turns = resolvedTurns(fixture);
      const drained = turns.some(t => hasSub(mon(t.before, 0, 0)) &&
        mon(t.after, 0, 0).hp === mon(t.before, 0, 0).hp &&
        mon(t.after, 1, 0).hp > mon(t.before, 1, 0).hp);
      if (!drained) return 'Draining Kiss did not heal from decoy damage';
      return null;
    },
  },
  {
    name: 'bypasssub_taunt_and_snarl_pass_the_decoy',
    // Taunt and Snarl both carry `flags.bypasssub`: the decoy stays, the taunt
    // volatile lands and Snarl removes the owner's HP.
    p1: [SUB([]), ...fillerTeam()],
    p2: [bulky('Incineroar', 'Intimidate', ['Taunt', 'Snarl', 'Protect']),
      offensive('Metagross', 'Clear Body', ['Meteor Mash', 'Protect']),
      ...FILLERS.filter(([s]) => !['Incineroar', 'Metagross'].includes(s)).map(([s, a, m]) => offensive(s, a, m)).slice(0, 4)],
    seeds: [[53, 106, 212, 424], [59, 118, 236, 472], [61, 122, 244, 488]],
    script: [
      {p1: ['substitute', 'protect'], p2: ['protect', 'protect']},
      {p1: ['raindance', 'protect'], p2: ['taunt', 'protect']},
      {p1: ['raindance', 'protect'], p2: ['snarl', 'protect']},
      {p1: ['raindance', 'protect'], p2: ['snarl', 'protect']},
    ],
    coverage: {move: 'substitute'},
    verify(fixture) {
      if (!sawVolatile(fixture, 0, 0, 'substitute')) return 'the decoy was never created';
      const taunted = fixture.steps.some(step => hasSub(mon(step.expected, 0, 0)) &&
        mon(step.expected, 0, 0).volatiles.includes('taunt'));
      if (!taunted) return 'Taunt was blocked by the decoy';
      const bypassed = resolvedTurns(fixture).some(t => hasSub(mon(t.before, 0, 0)) &&
        mon(t.after, 0, 0).hp < mon(t.before, 0, 0).hp);
      if (!bypassed) return 'Snarl did not bypass the decoy';
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
fs.writeFileSync(new URL('../data/more_substitute.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify({trials: TRIALS.length, fixtures: fixtures.length, skipped: skipped.length}));
for (const s of skipped) console.log('SKIP', JSON.stringify(s));
