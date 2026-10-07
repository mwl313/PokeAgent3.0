// Development-only corpus for four small ability ports:
// - Poison Point: the reference 3/10 contact roll poisons the attacker.
// - Surge Surfer: doubles Speed while the field terrain is Electric Terrain.
// - Screen Cleaner: the switch-in removes Reflect / Light Screen / Aurora Veil
//   from every side, announcing the ability once before the first removal.
// - Rivalry: same-gender 1.25x, opposite-gender 0.75x, genderless neutral.
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
const bulky = (species, ability, moves, gender = 'M') =>
  ({...setOf(species, ability, moves, '', {hp: 32, atk: 2, def: 16, spa: 0, spd: 16, spe: 0}), gender});
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

// `wanted` entries are a move id, `null` for the auto choice, `{move, target}`
// for an explicitly targeted move, or `{switch: 'sN'}` for a named switch-in.
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
      // A voluntary switch inside a normal request: the reference command is
      // `switch N` for this slot while the other slot still moves.
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
const FILL = [
  ['Metagross', 'Clear Body', ['Iron Head', 'Protect']],
  ['Milotic', 'Competitive', ['Surf', 'Protect']],
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']],
  ['Falinks', 'Battle Armor', ['Smart Strike', 'Protect']],
  ['Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']],
];
const fillerAfter = (head, count = 5) => [head, ...FILL.slice(0, count).map(([s, a, m]) => offensive(s, a, m))];
const foeTeam = () => [
  offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
  offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
  offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
  offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
  offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
  bulky('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
];
const foeWith = head => [head, ...foeTeam().filter(p => p.species !== head.species)];

const TRIALS = [
  {
    name: 'poisonpoint_contact_poisons_attacker',
    p1: () => fillerAfter(setOf('Overqwil', 'Poison Point', ['Poison Jab', 'Protect'])),
    p2: () => foeTeam(),
    script: [{p1: ['poisonjab', 'protect'], p2: [{move: 'ironhead', target: 1}, 'protect']}],
    coverage: {ability: 'poisonpoint'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Iron Head\|p1a: s0/)) return 'the contact hit never landed';
      const poisoned = fixture.steps.some(step => step.expected.sides[1].pokemon
        .some(p => p.status === ids.conditions.psn));
      if (!poisoned) return 'the 3/10 roll never poisoned the attacker across the attempted seeds';
      return null;
    },
  },
  {
    name: 'poisonpoint_noncontact_leaves_attacker_clean',
    p1: () => fillerAfter(setOf('Overqwil', 'Poison Point', ['Poison Jab', 'Protect'])),
    p2: () => foeWith(offensive('Milotic', 'Competitive', ['Surf', 'Protect'])),
    script: [{p1: ['poisonjab', 'protect'], p2: ['surf', 'protect']}],
    coverage: {ability: 'poisonpoint'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p2a: s0\|Surf\|/)) return 'Surf never executed';
      if (!logHas(session, /\|-damage\|p1a: s0/)) return 'the non-contact hit never damaged Overqwil';
      const poisoned = fixture.steps.some(step => step.expected.sides[1].pokemon
        .some(p => p.status === ids.conditions.psn));
      if (poisoned) return 'a non-contact hit wrongly triggered Poison Point';
      return null;
    },
  },
  {
    name: 'surgesurfer_doubles_speed_in_electric_terrain',
    p1: () => [
      setOf('Pincurchin', 'Electric Surge', ['Thunderbolt', 'Protect']),
      setOf('Raichu-Alola', 'Surge Surfer', ['Thunderbolt', 'Protect']),
      ...FILL.slice(0, 4).map(([s, a, m]) => offensive(s, a, m)),
    ],
    p2: () => foeTeam(),
    script: [{p1: ['thunderbolt', 'protect'], p2: ['protect', 'protect']}],
    coverage: {ability: 'surgesurfer'},
    verify(fixture) {
      const active = fixture.steps.some(step => {
        const terrain = step.expected.field.some(([id]) => id === ids.conditions.electricterrain);
        const raichu = step.expected.sides[0].pokemon.find(p => p.roster === 1);
        return terrain && raichu && raichu.cached_speed === raichu.stats[5] * 2;
      });
      if (!active) return 'Raichu-Alola never had the doubled Speed under Electric Terrain';
      return null;
    },
  },
  {
    name: 'screencleaner_clears_both_sides',
    p1: () => [
      bulky('Chimecho', 'Levitate', ['Reflect', 'Protect']),
      offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
      setOf('Mr. Rime', 'Screen Cleaner', ['Psychic', 'Protect']),
      offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
      offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
      offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
    ],
    p2: () => foeWith(bulky('Chimecho', 'Levitate', ['Light Screen', 'Protect'])),
    script: [
      {p1: ['reflect', 'protect'], p2: ['lightscreen', 'protect']},
      {p1: [{switch: 's2'}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {ability: 'screencleaner'},
    verify(fixture, session) {
      const reflect = ids.conditions.reflect;
      const light = ids.conditions.lightscreen;
      const had = fixture.steps.some(step =>
        step.expected.sides[0].conditions.some(([id]) => id === reflect)
        || step.expected.sides[1].conditions.some(([id]) => id === light));
      // The switch-in step is the first boundary where Mr. Rime is active;
      // later turns may legitimately re-cast a screen, so only that boundary
      // (and the one right after it) must show both screens removed.
      const switchIndex = fixture.steps.findIndex(step => step.expected.sides[0].pokemon
        .some(p => p.roster === 2 && p.active_slot !== null));
      const atSwitch = switchIndex >= 0 ? fixture.steps[switchIndex] : null;
      const cleared = atSwitch
        && atSwitch.expected.sides[0].conditions.every(([id]) => id !== reflect)
        && atSwitch.expected.sides[1].conditions.every(([id]) => id !== light);
      if (!had) return 'no screen was ever set';
      if (switchIndex < 0) return 'Mr. Rime never entered the field';
      if (!cleared) {
        return `Screen Cleaner left a screen standing (switchIndex ${switchIndex}, `
          + `p1 ${JSON.stringify(fixture.steps[switchIndex].expected.sides[0].conditions)}, `
          + `p2 ${JSON.stringify(fixture.steps[switchIndex].expected.sides[1].conditions)})`;
      }
      return null;
    },
  },
  {
    name: 'rivalry_same_gender_boosts',
    p1: () => fillerAfter(setOf('Luxray', 'Rivalry', ['Wild Charge', 'Protect'])),
    p2: () => foeTeam(),
    script: [{p1: ['wildcharge', 'protect'], p2: ['protect', 'protect']}],
    coverage: {ability: 'rivalry'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Wild Charge\|p2a: s0/)) return 'Wild Charge never landed';
      const damaged = fixture.steps.some(step => step.expected.sides[1].pokemon
        .some(p => p.roster === 0 && p.hp < p.max_hp));
      if (!damaged) return 'the same-gender target never lost HP';
      return null;
    },
  },
  {
    name: 'rivalry_opposite_gender_weakens',
    p1: () => fillerAfter(setOf('Luxray', 'Rivalry', ['Wild Charge', 'Protect'])),
    p2: () => foeWith(offensive('Milotic', 'Competitive', ['Surf', 'Protect'], '', 'F')),
    script: [{p1: ['wildcharge', 'protect'], p2: ['surf', 'protect']}],
    coverage: {ability: 'rivalry'},
    verify(fixture, session) {
      if (!logHas(session, /\|move\|p1a: s0\|Wild Charge\|p2a: s0/)) return 'Wild Charge never landed';
      const damaged = fixture.steps.some(step => step.expected.sides[1].pokemon
        .some(p => p.roster === 0 && p.hp < p.max_hp));
      if (!damaged) return 'the opposite-gender target never lost HP';
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
  for (let i = 0; i < 12; i++) seeds.push([3, 5, 7, 1000 + i]);
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
      if (process.env.DEBUG_SMALL_ABILITIES) {
        fs.appendFileSync(new URL('../../tmp/small_abilities_debug.log', import.meta.url),
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
fs.writeFileSync(new URL('../data/more_small_abilities.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify({trials: TRIALS.length, fixtures: fixtures.length, skipped: skipped.length}));
for (const s of skipped) console.log('SKIP', JSON.stringify(s));
