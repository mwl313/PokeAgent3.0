// Development-only corpus for the Stance Change port (Aegislash):
// - a damaging move switches Aegislash to the Blade forme before the action
//   resolves, so the stored stats and the damage use the Blade values;
// - King's Shield switches it back to the Shield forme;
// - any other status move leaves the current forme alone;
// - the change is non-permanent and reverts when Aegislash switches out.
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
const foeTeam = () => [
  offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
  offensive('Snorlax', 'Thick Fat', ['Body Slam', 'Protect']),
  offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
  offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
  offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
  bulky('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
];
const aegislash = moves => [setOf('Aegislash', 'Stance Change', moves), ...FILL.slice(0, 5).map(([s, a, m]) => offensive(s, a, m))];
const shield = () => ids.species.aegislash;
const blade = () => ids.species.aegislashblade;
const aegisAt = step => step.expected.sides[0].pokemon.find(p => p.roster === 0);

const TRIALS = [
  {
    // `onModifyMove`: Iron Head switches Aegislash to Blade *before* the hit,
    // so the recorded stored stats are the Blade stats and the damage on the
    // unprotected target is computed from the Blade Attack.
    name: 'stancechange_blades_on_damaging_move',
    p1: () => aegislash(['Iron Head', "King's Shield"]),
    p2: () => foeTeam(),
    script: [{p1: ['ironhead', 'protect'], p2: ['ironhead', 'protect']}],
    coverage: {ability: 'stancechange'},
    verify(fixture, session) {
      if (!logHas(session, /\|-formechange\|p1a: s0\|Aegislash-Blade\|/)) {
        return 'Aegislash never announced the Blade forme';
      }
      const initial = fixture.initial.sides[0].pokemon.find(p => p.roster === 0);
      const bladeAtk = initial.stats[1] + 90; // Aegislash 140 vs 50 base Attack
      const bladeDef = initial.stats[2] - 90; // ... and 50 vs 140 base Defense
      const statsSeen = fixture.steps.some(step => {
        const a = aegisAt(step);
        return a && a.species === blade() && a.stats[1] === bladeAtk && a.stats[0] === a.max_hp;
      });
      if (!statsSeen) return 'the Blade forme never showed its recomputed Attack stats';
      const defenceSeen = fixture.steps.some(step => aegisAt(step)?.stats[2] === bladeDef);
      if (!defenceSeen) return 'the Blade forme never showed its recomputed Defence stats';
      if (!logHas(session, /\|move\|p1a: s0\|Iron Head\|p2a: s0/)) return 'Iron Head never landed';
      const damaged = fixture.steps.some(step => {
        const target = step.expected.sides[1].pokemon.find(p => p.roster === 0);
        return target && target.hp < target.max_hp;
      });
      if (!damaged) return 'the Blade-forme Iron Head never damaged the target';
      return null;
    },
  },
  {
    // King's Shield is the one status move that switches back to Shield.
    name: 'stancechange_kings_shield_returns_shield',
    p1: () => aegislash(['Iron Head', "King's Shield"]),
    p2: () => foeTeam(),
    script: [
      {p1: ['ironhead', 'protect'], p2: ['protect', 'protect']},
      {p1: ['kingsshield', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {ability: 'stancechange'},
    verify(fixture, session) {
      if (!logHas(session, /\|-formechange\|p1a: s0\|Aegislash-Blade\|/)) return 'no Blade forme';
      if (!logHas(session, /\|-formechange\|p1a: s0\|Aegislash\|/)) return "King's Shield never switched back to Shield";
      const last = fixture.steps[fixture.steps.length - 1];
      const a = aegisAt(last);
      if (!a || a.species !== shield()) return `the final boundary is not the Shield forme (${a && a.species})`;
      return null;
    },
  },
  {
    // Any other status move leaves the forme untouched: Iron Defense after
    // Iron Head must stay in Blade.
    name: 'stancechange_other_status_keeps_forme',
    p1: () => aegislash(['Iron Head', 'Iron Defense']),
    p2: () => foeTeam(),
    script: [
      {p1: ['ironhead', 'protect'], p2: ['protect', 'protect']},
      {p1: ['irondefense', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {ability: 'stancechange'},
    verify(fixture, session) {
      const changes = session.battle.log.filter(line => line.includes('-formechange')).length;
      if (changes !== 1) return `expected exactly one forme change, saw ${changes}`;
      // Auto-play keeps the battle running past the scripted turns, so the
      // forme check reads the last boundary where Aegislash is on the field.
      const activeBlade = fixture.steps.some(step => {
        const a = aegisAt(step);
        return a && a.active_slot !== null && a.species === blade();
      });
      if (!activeBlade) return 'Iron Defense changed the active forme away from Blade';
      return null;
    },
  },
  {
    // The change is non-permanent: switching out restores the submitted
    // Shield forme even while Aegislash sits on the bench.
    name: 'stancechange_switch_out_reverts_forme',
    p1: () => aegislash(['Iron Head', "King's Shield"]),
    p2: () => foeTeam(),
    script: [
      {p1: ['ironhead', 'protect'], p2: ['protect', 'protect']},
      {p1: [{switch: 's1'}, 'protect'], p2: ['protect', 'protect']},
      {p1: [{switch: 's0'}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {ability: 'stancechange'},
    verify(fixture, session) {
      if (!logHas(session, /\|-formechange\|p1a: s0\|Aegislash-Blade\|/)) return 'Aegislash never became Blade';
      const reverted = fixture.steps.some(step => {
        const a = aegisAt(step);
        return a && a.active_slot === null && a.species === shield();
      });
      if (!reverted) return 'the benched Aegislash did not revert to the Shield forme';
      const last = fixture.steps[fixture.steps.length - 1];
      const a = aegisAt(last);
      if (!a || a.species !== shield()) return `Aegislash did not re-enter in the Shield forme (${a && a.species})`;
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
      if (process.env.DEBUG_STANCE_CHANGE) {
        fs.appendFileSync(new URL('../../tmp/stance_change_debug.log', import.meta.url),
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
fs.writeFileSync(new URL('../data/more_stance_change.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify({trials: TRIALS.length, fixtures: fixtures.length, skipped: skipped.length}));
for (const s of skipped) console.log('SKIP', JSON.stringify(s));
