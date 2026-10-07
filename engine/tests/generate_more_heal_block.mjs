// Development-only interaction corpus for Psychic Noise's Heal Block.
//
// The generic move corpus records the volatile but its filler teams carry no
// recovery, so the request-level `onDisableMove`, the priority-6 BeforeMove
// refusal and the `TryHeal` refusal are never exercised. This generator
// scripts complete legal battles that force each ported recovery path:
//
//   * Leftovers and a `heal`-flag move (Recover) refused for two turns, with
//     the request marking Recover disabled and the same move legal again after
//     the volatile expires.
//   * A drain move whose healing is refused while its damage still lands.
//   * Liquid Ooze versus Heal Block on a drain heal, both handler orders
//     (the priority-0 handlers sort by cached speed, then by effect order).
//   * Regenerator's switch-out heal refused while the volatile is still up
//     (`SwitchOut` runs before `clearVolatile`).
import fs from 'node:fs';
import {createRequire} from 'node:module';
import {ReferenceSession, verifyReference, FORMAT, ORACLE_COMMIT} from '../reference.mjs';
const require = createRequire(import.meta.url);
const {TeamValidator, toID} = require('../../vendor/pokemon-showdown/dist/sim');
const validator = new TeamValidator(FORMAT);
const data = JSON.parse(fs.readFileSync(new URL('../data/dex.json', import.meta.url), 'utf8'));
const ids = Object.fromEntries(Object.entries(data.tables).map(([k, rows]) => [k,
  Object.fromEntries(rows.map(r => [r.id, r.numeric_id]))]));
const stats = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
verifyReference();

const nameSets = team => team.map((set, i) => ({...set, name: `s${i}`}));
const setOf = (species, ability, moves, item = '', points = {hp: 24, atk: 8, def: 8, spa: 8, spd: 8, spe: 4},
  nature = 'Serious', gender = 'M') =>
  ({name: species, species, ability, item, nature, level: 50, gender, moves,
    evs: {hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0, ...points}});
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
      active_slot: s.active.indexOf(p) >= 0 ? s.active.indexOf(p) : null, ability_ending: Boolean(p.abilityState.ending),
      cached_speed: p.speed ?? null, status: ids.conditions[p.status] ?? 0, boosts: Object.values(p.boosts),
      stats: [p.maxhp, ...Object.values(p.storedStats)], ability: ids.abilities[p.ability], item: ids.items[p.item] ?? 0,
      types: p.types.map(t => ids.types[toID(t)] ?? 0), previous_item: ids.items[p.lastItem] ?? 0, can_mega: Boolean(p.canMegaEvo),
      pp: p.moveSlots.map(m => m.pp), volatiles: Object.keys(p.volatiles).sort()})),
    request_detail: requestDetail(session, s)}))});

const select = (kind, own_slot, destination = 255) =>
  ({kind, own_slot, move_slot: 255, target_location: 0, switch_destination: destination, resource: 'None'});
const move = (slot, moveSlot, target = 0) =>
  ({kind: 'Move', own_slot: slot, move_slot: moveSlot, target_location: target, switch_destination: 255, resource: 'None'});

// A script entry is either a move name (auto target) or {move, foe} where
// `foe` is the opposing active slot to target (0 or 1). `null` falls back to
// the first usable move, which is what the reference replays when a scripted
// move is disabled by the Heal Block request.
function choose(session, sideIndex, script) {
  const b = session.battle, side = b.sides[sideIndex];
  if (side.requestState === 'teampreview') {
    return {actions: [0, 1, 4, 5].map((r, i) => select('Pick', i, r)), command: 'team 1256'};
  }
  const bench = side.pokemon.filter(p => !p.fainted && !side.active.includes(p));
  const actions = [], commands = [];
  for (let slot = 0; slot < 2; slot++) {
    const p = side.active[slot];
    if (side.requestState === 'switch') {
      if (!side.activeRequest?.forceSwitch?.[slot]) { commands.push('pass'); continue; }
      const reserve = bench.shift();
      if (reserve) { actions.push(select('Switch', slot, roster(reserve))); commands.push(`switch ${side.pokemon.indexOf(reserve) + 1}`); }
      else { commands.push('pass'); }
      continue;
    }
    if (p.fainted) { actions.push(select('Pass', slot)); commands.push('pass'); continue; }
    const entry = script(sideIndex, b.turn, slot);
    if (entry === 'SWITCH') {
      const reserve = bench.shift();
      if (reserve) {
        actions.push(select('Switch', slot, roster(reserve)));
        commands.push(`switch ${side.pokemon.indexOf(reserve) + 1}`);
        continue;
      }
    }
    const wantedName = typeof entry === 'string' ? entry : entry?.move;
    // `moveSlots[i].id` is the reference string id, not the numeric export id.
    const wanted = wantedName ? toID(wantedName) : null;
    let slotIndex = wanted != null
      ? p.moveSlots.findIndex(m => m.id === wanted && !m.disabled && m.pp > 0) : -1;
    if (slotIndex < 0) slotIndex = p.moveSlots.findIndex(m => !m.disabled && m.pp > 0);
    const chosen = p.moveSlots[slotIndex];
    let location = 0;
    if (chosen && b.actions.targetTypeChoices(chosen.target)) {
      const foe = typeof entry === 'object' && entry?.foe != null ? entry.foe : 0;
      location = b.validTargetLoc(foe + 1, p, chosen.target) ? foe + 1 : 0;
    }
    actions.push(move(slot, slotIndex, location));
    commands.push(`move ${slotIndex + 1}${location ? ` ${location}` : ''}`);
  }
  return {actions, command: commands.join(', ')};
}

const fillers = [
  ['Goodra-Hisui', 'Shell Armor', ['Dragon Pulse', 'Protect']],
  ['Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']],
  ['Metagross', 'Clear Body', ['Iron Head', 'Protect']],
  ['Scolipede', 'Swarm', ['X-Scissor', 'Protect']],
  ['Starmie', 'Natural Cure', ['Ice Beam', 'Protect']],
  ['Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']],
].map(([species, ability, moves]) => setOf(species, ability, moves));

const trial = (scene) => {
  const asList = value => Array.isArray(value) ? value : [value];
  const build = value => {
    const core = asList(value);
    return nameSets([...core, ...fillers.slice(0, 6 - core.length)]);
  };
  const teamA = build(scene.a);
  const teamB = build(scene.b);
  const problems = validator.validateTeam(teamA) || validator.validateTeam(teamB);
  if (problems) throw new Error(`${scene.name}: invalid teams: ${problems.join('; ')}`);
  let lastReason = null;
  for (let trialIndex = 0; trialIndex < scene.trials; trialIndex++) {
    const seed = [2026, 10, 7, scene.seed + trialIndex];
    const session = new ReferenceSession({teams: [teamA, teamB], seed});
    const fixture = {name: `${scene.name}_${seed[3]}`, seed,
      teams: [teamA, teamB].map((team, side) => ({id: `${scene.name}-${side}`, members: team.map(s => ({
        species: ids.species[toID(s.species)], ability: ids.abilities[toID(s.ability)], item: ids.items[toID(s.item)] ?? 0,
        nature: ids.natures[toID(s.nature)], gender: s.gender || '', level: 50, moves: s.moves.map(x => ids.moves[toID(x)]),
        points: stats.map(k => s.evs[k]), ivs: stats.map(k => s.ivs[k])}))})),
      initial: compact(session), steps: []};
    let failure = null;
    while (!session.battle.ended && fixture.steps.length < 240) {
      for (let side = 0; side < 2; side++) {
        if (session.battle.ended) break;
        const s = session.battle.sides[side];
        if (s.activeRequest?.wait || s.isChoiceDone()) continue;
        const choice = choose(session, side, scene.script);
        const result = session.choose(side ? 'p2' : 'p1', choice.command);
        if (!result.accepted) { failure = JSON.stringify({side, choice, messages: result.messages}); break; }
        fixture.steps.push({side: side ? 'P2' : 'P1', ...choice, expected: compact(session)});
      }
      if (failure) break;
    }
    if (failure) { session.destroy(); lastReason = failure; continue; }
    if (!session.battle.ended) { session.destroy(); lastReason = 'did not complete'; continue; }
    const reason = scene.require(fixture, session.battle.log);
    if (reason) {
      if (process.env.PA3_HB_DEBUG) {
        console.error(`${scene.name}#${seed[3]}: ${reason}`);
        for (const step of fixture.steps) {
          const sides = step.expected.sides.map(s => s.pokemon.slice(0, 2)
            .map(p => `${p.species}:${p.hp}/${p.max_hp}@${p.active_slot}` + (p.volatiles.length ? `[${p.volatiles}]` : '')).join(' '));
          console.error(`  ${step.side} t${step.expected.turn} ${step.command} | ${sides.join(' || ')}`);
        }
      }
      session.destroy(); lastReason = reason; continue;
    }
    session.destroy();
    return fixture;
  }
  return {skipped: {name: scene.name, reason: lastReason ?? 'no seed produced the required behavior'}};
};

const monByRoster = (step, side, rosterIndex) =>
  step.expected.sides[side].pokemon.find(p => p.roster === rosterIndex);
const healBlockSteps = (fixture, side, rosterIndex) => fixture.steps
  .filter(step => monByRoster(step, side, rosterIndex)?.volatiles.includes('healblock'));
const requestDisabled = (fixture, side, rosterIndex, moveName) => fixture.steps.some(step =>
  step.expected.sides[side].request_detail?.slots
    ?.filter(slot => slot.present && slot.moves.length)
    .some(slot => slot.moves.some(m => m.id === ids.moves[toID(moveName)] && m.disabled)));
const requestEnabled = (fixture, side, rosterIndex, moveName) => fixture.steps.some(step =>
  step.expected.sides[side].request_detail?.slots
    ?.some(slot => slot.moves.some(m => m.id === ids.moves[toID(moveName)] && !m.disabled)));
const hpSeries = (fixture, side, rosterIndex) => fixture.steps.map(step => monByRoster(step, side, rosterIndex).hp);

const scenes = [];

// 1. Leftovers + Recover vs Psychic Noise: the request disables Recover, the
// committed Recover is refused, Leftovers recovers nothing while the volatile
// is up, a second Psychic Noise does not refresh it, and Recover works again
// after the two-turn window closes.
const gardevoir = setOf('Gardevoir', 'Synchronize', ['Psychic Noise', 'Moonblast', 'Protect'],
  '', {hp: 32, atk: 0, def: 0, spa: 0, spd: 2, spe: 32}, 'Timid');
const milotic = setOf('Milotic', 'Competitive', ['Recover', 'Scald', 'Protect'],
  'Leftovers', {hp: 32, atk: 0, def: 32, spa: 0, spd: 2, spe: 0}, 'Bold', 'F');
scenes.push({
  name: 'healblock_leftovers_recover', seed: 5200, trials: 64, a: gardevoir, b: milotic,
  script: (side, turn, slot) => {
    if (side === 0) return slot === 0 ? (turn <= 2 ? {move: 'Psychic Noise', foe: 0} : {move: 'Moonblast', foe: 1})
      : (turn === 1 ? 'Protect' : {move: 'Dragon Pulse', foe: 1});
    return slot === 0 ? (turn <= 3 ? 'Recover' : {move: 'Scald', foe: 0})
      : (turn === 1 ? 'Protect' : {move: 'Iron Head', foe: 0});
  },
  require: (fixture) => {
    const blocked = healBlockSteps(fixture, 1, 0);
    if (!blocked.length) return 'healblock never applied to Milotic';
    if (!requestDisabled(fixture, 1, 0, 'Recover')) return 'Recover never marked disabled in the request';
    const lastBlocked = fixture.steps.indexOf(blocked[blocked.length - 1]);
    const after = fixture.steps.slice(lastBlocked);
    const resumed = after.some((step, i) => {
      if (i === 0) return false;
      const now = monByRoster(step, 1, 0), previous = monByRoster(after[i - 1], 1, 0);
      return now.hp - previous.hp > now.max_hp / 8;
    });
    if (!resumed) return 'no Recover-sized heal after the volatile expired';
    return null;
  },
});

// 2. Drain moves carry the `heal` flag, so Heal Block refuses the move itself
// (priority-6 BeforeMove and the request's disabled flag), not only its
// healing. Giga Drain therefore never damages its target while the user is
// blocked, and the request marks it disabled.
const appletunDrain = setOf('Appletun', 'Thick Fat', ['Giga Drain', 'Protect'],
  '', {hp: 32, atk: 0, def: 0, spa: 0, spd: 32, spe: 2}, 'Calm');
const gardevoirFast = setOf('Gardevoir', 'Synchronize', ['Psychic Noise', 'Moonblast', 'Protect'],
  '', {hp: 2, atk: 0, def: 0, spa: 0, spd: 32, spe: 32}, 'Timid');
scenes.push({
  name: 'healblock_refuses_drain_move', seed: 5260, trials: 64, a: appletunDrain, b: gardevoirFast,
  script: (side, turn, slot) => {
    if (side === 1) return slot === 0 ? (turn === 1 ? {move: 'Psychic Noise', foe: 0} : {move: 'Moonblast', foe: 1})
      : (turn === 1 ? 'Protect' : {move: 'Iron Head', foe: 1});
    return slot === 0 ? (turn === 1 ? {move: 'Giga Drain', foe: 0} : {move: 'Protect', foe: 0})
      : (turn === 1 ? 'Protect' : {move: 'Dragon Pulse', foe: 0});
  },
  require: (fixture, log) => {
    if (!fixture.steps.some(step => monByRoster(step, 0, 0).volatiles.includes('healblock')))
      return 'healblock never applied to the Giga Drain user';
    if (!log.some(line => line.includes('|cant|p1a:') && line.includes('Giga Drain')))
      return 'Giga Drain was never refused by the mid-turn Heal Block check';
    if (!requestDisabled(fixture, 0, 0, 'Giga Drain'))
      return 'Giga Drain was never marked disabled in the request';
    if (log.some(line => line.includes('|move|p1a:') && line.includes('|Giga Drain|')))
      return 'Giga Drain still executed while Heal Block was up';
    return null;
  },
});

// 3. Liquid Ooze ordering. Both handlers are priority 0, so the faster cached
// speed runs first: a faster drainer's Heal Block refuses the heal before Ooze
// can damage it, while a faster Ooze holder damages the drainer and stops the
// heal itself. Swalot is the only legal Liquid Ooze holder in the scope.
const swalotSlow = setOf('Swalot', 'Liquid Ooze', ['Amnesia', 'Sludge Bomb', 'Protect'],
  '', {hp: 32, atk: 0, def: 32, spa: 0, spd: 2, spe: 0}, 'Bold');
const swalotFast = setOf('Swalot', 'Liquid Ooze', ['Amnesia', 'Sludge Bomb', 'Protect'],
  '', {hp: 2, atk: 0, def: 0, spa: 0, spd: 32, spe: 32}, 'Timid');
const appletunSlow = setOf('Appletun', 'Thick Fat', ['Giga Drain', 'Protect'],
  '', {hp: 32, atk: 0, def: 0, spa: 32, spd: 2, spe: 0}, 'Quiet');
const appletunFast = setOf('Appletun', 'Thick Fat', ['Giga Drain', 'Protect'],
  '', {hp: 2, atk: 0, def: 0, spa: 32, spd: 0, spe: 32}, 'Mild');

// 3. Leech Seed's residual heal goes through `this.heal`, so it is refused
// while the seeding slot is Heal Blocked even though its damage still lands.
// With a Liquid Ooze holder as the seeded target the two priority-0 TryHeal
// handlers sort by cached speed, which is what the two ordering scenes pin.
const miloticSeed = setOf('Milotic', 'Competitive', ['Scald', 'Protect'],
  '', {hp: 32, atk: 0, def: 32, spa: 0, spd: 2, spe: 0}, 'Bold', 'F');
const seederSlow = setOf('Appletun', 'Thick Fat', ['Leech Seed', 'Protect'],
  '', {hp: 32, atk: 0, def: 0, spa: 0, spd: 32, spe: 0}, 'Calm');
const seederFast = setOf('Appletun', 'Thick Fat', ['Leech Seed', 'Protect'],
  '', {hp: 2, atk: 0, def: 0, spa: 0, spd: 0, spe: 32}, 'Hasty');

const seedScene = (name, seed, seeder, seeded, expectOoze) => ({
  name, seed, trials: 64, a: seeder, b: [seeded, gardevoirFast], expectOoze,
  script: (side, turn, slot) => {
    if (side === 1) return slot === 0 ? (turn === 1 ? {move: 'Scald', foe: 1} : 'Protect')
      : (turn === 1 ? {move: 'Psychic Noise', foe: 0} : {move: 'Moonblast', foe: 1});
    return slot === 0 ? (turn === 1 ? {move: 'Leech Seed', foe: 0} : {move: 'Protect', foe: 0})
      : (turn === 1 ? 'Protect' : {move: 'Dragon Pulse', foe: 1});
  },
  require: (fixture, log) => {
    const seeded = fixture.steps.some((step, i) => i > 0
      && monByRoster(step, 1, 0).hp < monByRoster(fixture.steps[i - 1], 1, 0).hp);
    if (!seeded) return 'Leech Seed never damaged the seeded target';
    if (!fixture.steps.some(step => monByRoster(step, 0, 0).volatiles.includes('healblock')))
      return 'healblock never applied to the seeding slot';
    // Restrict the checks to the exact window between the Heal Block start and
    // end lines: the seed keeps healing once the volatile expires.
    const start = log.findIndex(line => line.startsWith('|-start|p1a:') && line.includes('move: Heal Block'));
    if (start < 0) return 'the reference never started Heal Block';
    const end = log.findIndex((line, i) => i > start && line.startsWith('|-end|p1a:') && line.includes('Heal Block'));
    const window = log.slice(start, end < 0 ? log.length : end + 1);
    const oozeHit = window.some(line => line.startsWith('|-damage|p1a:') && line.includes('ability: Liquid Ooze'));
    const healed = window.some(line => line.startsWith('|-heal|p1a:'));
    if (expectOoze) {
      if (!oozeHit) return 'Liquid Ooze never damaged the faster seeding slot';
      return null;
    }
    if (oozeHit) return 'Liquid Ooze damaged a seeding slot Heal Block should have refused first';
    if (healed) return 'Leech Seed healed the blocked seeding slot';
    return null;
  },
});

const oozeScene = (name, seed, drainer, ooze, psychicNoise, expectDrainerDamaged) => ({
  name, seed, trials: 64, a: drainer, b: [ooze, psychicNoise], expectDrainerDamaged,
  script: (side, turn, slot) => {
    if (side === 1) return slot === 0 ? (turn === 1 ? 'Amnesia' : {move: 'Sludge Bomb', foe: 1})
      : (turn === 1 ? {move: 'Psychic Noise', foe: 0} : {move: 'Moonblast', foe: 1});
    return slot === 0 ? (turn === 1 ? {move: 'Giga Drain', foe: 0} : {move: 'Protect', foe: 0})
      : (turn === 1 ? 'Protect' : {move: 'Dragon Pulse', foe: 1});
  },
  require: (fixture, log) => {
    const healer = hpSeries(fixture, 0, 0);
    const oozeHp = hpSeries(fixture, 1, 0);
    if (!fixture.steps.some(step => monByRoster(step, 0, 0).volatiles.includes('healblock')))
      return 'healblock never applied to the drainer';
    const healedDrainer = healer.some((hp, i) => i > 0 && hp > healer[i - 1]);
    const damagedOoze = oozeHp.some((hp, i) => i > 0 && hp < oozeHp[i - 1]);
    if (!damagedOoze) return 'Giga Drain never damaged the Ooze holder';
    if (healedDrainer) return 'the drain healed despite the TryHeal refusal';
    // The reference log names the Ooze damage explicitly, so the scene can
    // tell it apart from Psychic Noise's own damage on the drainer.
    const drainerDamaged = log.some(line => line.startsWith('|-damage|p1a:') && line.includes('ability: Liquid Ooze'));
    if (expectDrainerDamaged && !drainerDamaged) return 'Liquid Ooze never damaged the faster drainer';
    if (!expectDrainerDamaged && drainerDamaged) return 'Liquid Ooze damaged a drainer that Heal Block should have refused first';
    return null;
  },
});
scenes.push(seedScene('healblock_blocks_leech_seed', 5500, seederSlow, miloticSeed, false));
scenes.push(seedScene('healblock_before_liquid_ooze_seed', 5560, seederFast, swalotSlow, false));
scenes.push(seedScene('liquid_ooze_before_healblock_seed', 5620, seederSlow, swalotFast, true));

// 4. Regenerator's switch-out heal uses the raw `pokemon.heal`, so it happens
// even while Heal Block is up. This fixture pins that bypass against the
// reference (a `this.heal` implementation would refuse it).
const reuniclus = setOf('Reuniclus', 'Regenerator', ['Recover', 'Psychic', 'Protect'],
  '', {hp: 32, atk: 0, def: 32, spa: 0, spd: 2, spe: 0}, 'Relaxed');
scenes.push({
  name: 'healblock_regenerator_raw_heal', seed: 5440, trials: 64, a: reuniclus, b: gardevoirFast,
  script: (side, turn, slot) => {
    if (side === 1) return slot === 0 ? {move: 'Psychic Noise', foe: 0}
      : (turn === 1 ? 'Protect' : {move: 'Ice Beam', foe: 0});
    if (side === 0 && turn === 2) return slot === 0 ? 'SWITCH' : {move: 'Dragon Pulse', foe: 0};
    return slot === 0 ? (turn === 1 ? 'Recover' : 'Psychic')
      : (turn === 1 ? 'Protect' : {move: 'Seed Bomb', foe: 0});
  },
  require: (fixture) => {
    const steps = fixture.steps;
    const before = steps.findIndex(step => {
      const p = monByRoster(step, 0, 0);
      return p.volatiles.includes('healblock') && p.hp < p.max_hp;
    });
    if (before < 0) return 'healblock never applied to a damaged Regenerator holder';
    const damagedHp = monByRoster(steps[before], 0, 0).hp;
    const switched = steps.findIndex(step => monByRoster(step, 0, 0).active_slot === null);
    if (switched < 0) return 'the Regenerator holder never switched out';
    const carried = monByRoster(steps[switched], 0, 0).hp;
    if (carried <= damagedHp) return 'Regenerator did not heal on switch-out';
    return null;
  },
});

const fixtures = [];
const skipped = [];
for (const scene of scenes) {
  const result = trial(scene);
  if (result.skipped) { skipped.push(result.skipped); continue; }
  result.coverage = scene.expectDrainerDamaged === undefined
    ? {move: 'psychicnoise', heal_block: true}
    : {move: 'psychicnoise', heal_block: true, ooze_ordering: true,
        drainer_damaged: scene.expectDrainerDamaged};
  fixtures.push(result);
}

fs.writeFileSync(new URL('../data/more_heal_block.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify({fixtures: fixtures.length, skipped: skipped.length}));
if (skipped.length) console.log(JSON.stringify(skipped, null, 1));
