// Development-only corpus for Mold Breaker: `suppressingAbility` makes the
// user's move ignore every `flags.breakable` ability of the target, so the
// immunity, absorb, damage-clamp, damage-modifier and onSetStatus handlers all
// stop firing for that move. Each suppressed scene is paired with a control
// battle where the same move is used by a holder without Mold Breaker. Every
// fixture is a
// complete legal synthetic reference battle recorded at every boundary.
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
  ['Skarmory', 'Sturdy', ['Iron Head', 'Protect']],
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
// Lines emitted inside one turn, so a "did the move land" probe cannot be
// satisfied by a later turn's damage.
const turnLines = (session, turn) => {
  const log = session.battle.log;
  const start = log.indexOf(`|turn|${turn}`);
  if (start < 0) return [];
  const rest = log.slice(start + 1);
  const end = rest.findIndex(line => /^\|turn\|/.test(line));
  return end < 0 ? rest : rest.slice(0, end);
};
const hit = (session, turn, name = 'p2a: s0') =>
  turnLines(session, turn).some(line => line.startsWith(`|-damage|${name}|`) && !/\|0 fnt$/.test(line));
const gotStatus = (session, turn, name = 'p2a: s0', status = 'par') =>
  turnLines(session, turn).some(line => line.startsWith(`|-status|${name}|${status}`));
const disguised = (session, turn) =>
  turnLines(session, turn).some(line => /^\|-activate\|p2a: s0\|ability: Disguise/.test(line));
const fainted = (session, turn, name = 'p2a: s0') =>
  turnLines(session, turn).some(line => line.startsWith(`|faint|${name}`));

// Two identical turns; the target attacks instead of Protecting so the probed
// move is never blocked by a one-turn shield.
const script = (p1Move, p2Move) => [
  {p1: [p1Move, 'protect'], p2: [p2Move, 'protect']},
  {p1: [p1Move, 'protect'], p2: [p2Move, 'protect']},
];
const MB_EXCADRILL = setOf('Excadrill', 'Mold Breaker', ['Earthquake', 'Iron Head', 'Protect']);
const SR_EXCADRILL = setOf('Excadrill', 'Sand Rush', ['Earthquake', 'Iron Head', 'Protect']);
const HYDREIGON = setOf('Hydreigon', 'Levitate', ['Dragon Pulse', 'Protect']);
const ORTHWORM = setOf('Orthworm', 'Earth Eater', ['Iron Head', 'Protect']);
const AVALUGG = setOf('Avalugg-Hisui', 'Sturdy', ['Crunch', 'Protect']);
const DRAGONITE = setOf('Dragonite', 'Multiscale', ['Dragon Claw', 'Protect']);

const MB_TINKATON = setOf('Tinkaton', 'Mold Breaker', ['Thunder Wave', 'Play Rough', 'Protect']);
// Same species and moves as the Mold Breaker lead, with a ported ability that
// has no suppression: the control isolates Mold Breaker from every other
// difference between the two battles.
const PK_TINKATON = setOf('Tinkaton', 'Pickpocket', ['Play Rough', 'Thunder Wave', 'Protect']);
const LEV_ROTOM = setOf('Rotom', 'Levitate', ['Thunder Wave', 'Volt Switch', 'Protect']);


const team = (lead, others) => [lead, ...others];
const FILL = () => fillerTeam();

const TRIALS = [
  {name: 'moldbreaker_earthquake_levitate',
    p1: team(MB_EXCADRILL, FILL().slice(1, 6)), p2: team(HYDREIGON, FILL().slice(1, 6)),
    seeds: [[3, 5, 7, 501], [11, 13, 17, 502]], script: script('earthquake', 'dragonpulse'), coverage: {ability: 'levitate', move: 'earthquake'},
    verify(fixture, session) {
      if (!hit(session, 1)) return 'Mold Breaker Earthquake never damaged the Levitate holder';
      return null;
    }},
  {name: 'control_earthquake_levitate',
    p1: team(SR_EXCADRILL, FILL().slice(1, 6)), p2: team(HYDREIGON, FILL().slice(1, 6)),
    seeds: [[3, 5, 7, 511], [11, 13, 17, 512]], script: script('earthquake', 'dragonpulse'), coverage: {ability: 'levitate', move: 'earthquake', control: true},
    verify(fixture, session) {
      if (hit(session, 1)) return 'the Levitate holder took Earthquake damage without Mold Breaker';
      return null;
    }},
  {name: 'moldbreaker_earthquake_eartheater',
    p1: team(MB_EXCADRILL, FILL().slice(1, 6)), p2: team(ORTHWORM, FILL().slice(1, 6)),
    seeds: [[3, 5, 7, 521], [11, 13, 17, 522]], script: script('earthquake', 'ironhead'), coverage: {ability: 'eartheater', move: 'earthquake'},
    verify(fixture, session) {
      if (!hit(session, 1)) return 'Mold Breaker Earthquake was still absorbed by Earth Eater';
      return null;
    }},
  {name: 'control_earthquake_eartheater',
    p1: team(SR_EXCADRILL, FILL().slice(1, 6)), p2: team(ORTHWORM, FILL().slice(1, 6)),
    seeds: [[3, 5, 7, 531], [11, 13, 17, 532]], script: script('earthquake', 'ironhead'), coverage: {ability: 'eartheater', move: 'earthquake', control: true},
    verify(fixture, session) {
      if (hit(session, 1)) return 'Earth Eater did not absorb the control Earthquake';
      return null;
    }},
  {name: 'moldbreaker_ironhead_sturdy_ko',
    p1: team(MB_EXCADRILL, FILL().slice(1, 6)), p2: team(AVALUGG, FILL().slice(1, 6)),
    seeds: [[3, 5, 7, 541], [11, 13, 17, 542]], script: script('ironhead', 'crunch'), coverage: {ability: 'sturdy', move: 'ironhead'},
    verify(fixture, session) {
      if (!hit(session, 1)) return 'Mold Breaker Iron Head never damaged the Sturdy holder';
      return null;
    }},
  {name: 'control_ironhead_sturdy_survives',
    p1: team(SR_EXCADRILL, FILL().slice(1, 6)), p2: team(AVALUGG, FILL().slice(1, 6)),
    seeds: [[3, 5, 7, 551], [11, 13, 17, 552]], script: script('ironhead', 'crunch'), coverage: {ability: 'sturdy', move: 'ironhead', control: true},
    verify(fixture, session) {
      if (!hit(session, 1)) return 'the control Iron Head never damaged the Sturdy holder';
      return null;
    }},
  {name: 'moldbreaker_rockslide_multiscale',
    p1: team(setOf('Excadrill', 'Mold Breaker', ['Rock Slide', 'Protect']), FILL().slice(1, 6)),
    p2: team(DRAGONITE, FILL().slice(1, 6)),
    seeds: [[3, 5, 7, 561], [11, 13, 17, 562]],
    script: script('rockslide', 'dragonclaw'),
    coverage: {ability: 'multiscale', move: 'rockslide'},
    verify(fixture, session) {
      if (!hit(session, 1)) return 'Mold Breaker Rock Slide never damaged the Multiscale holder';
      return null;
    }},
  {name: 'moldbreaker_thunderwave_limber',
    p1: team(MB_TINKATON, FILL().slice(1, 6)), p2: team(setOf('Hawlucha', 'Limber', ['Close Combat', 'Protect']), FILL().slice(1, 6)),
    seeds: [[3, 5, 7, 571], [11, 13, 17, 572]], script: script('thunderwave', 'closecombat'), coverage: {ability: 'limber', move: 'thunderwave'},
    verify(fixture, session) {
      if (!gotStatus(session, 1)) return 'Mold Breaker Thunder Wave did not paralyse the Limber holder';
      return null;
    }},
  {name: 'control_thunderwave_limber',
    p1: team(LEV_ROTOM, FILL().slice(1, 6)), p2: team(setOf('Hawlucha', 'Limber', ['Close Combat', 'Protect']), FILL().slice(1, 6)),
    seeds: [[3, 5, 7, 581], [11, 13, 17, 582]], script: script('thunderwave', 'closecombat'), coverage: {ability: 'limber', move: 'thunderwave', control: true},
    verify(fixture, session) {
      if (gotStatus(session, 1)) return 'Limber did not refuse the control Thunder Wave';
      return null;
    }},
  {name: 'moldbreaker_thunderwave_goodasgold',
    p1: team(MB_TINKATON, FILL().slice(1, 6)), p2: team(setOf('Gholdengo', 'Good as Gold', ['Shadow Ball', 'Protect']), FILL().slice(1, 6)),
    seeds: [[3, 5, 7, 591], [11, 13, 17, 592]], script: script('thunderwave', 'shadowball'), coverage: {ability: 'goodasgold', move: 'thunderwave'},
    verify(fixture, session) {
      if (!gotStatus(session, 1)) return 'Mold Breaker Thunder Wave was still refused by Good as Gold';
      return null;
    }},
  {name: 'control_thunderwave_goodasgold',
    p1: team(LEV_ROTOM, FILL().slice(1, 6)), p2: team(setOf('Gholdengo', 'Good as Gold', ['Shadow Ball', 'Protect']), FILL().slice(1, 6)),
    seeds: [[3, 5, 7, 601], [11, 13, 17, 602]], script: script('thunderwave', 'shadowball'), coverage: {ability: 'goodasgold', move: 'thunderwave', control: true},
    verify(fixture, session) {
      if (gotStatus(session, 1)) return 'Good as Gold did not refuse the control Thunder Wave';
      return null;
    }},
  {name: 'moldbreaker_playrough_disguise',
    p1: team(MB_TINKATON, FILL().slice(1, 6)), p2: team(setOf('Mimikyu', 'Disguise', ['Play Rough', 'Shadow Claw', 'Protect']), FILL().slice(1, 6)),
    seeds: [[3, 5, 7, 611], [11, 13, 17, 612]], script: script('playrough', 'shadowclaw'), coverage: {ability: 'disguise', move: 'playrough'},
    verify(fixture, session) {
      if (disguised(session, 1)) return 'Mold Breaker Play Rough was still absorbed by Disguise';
      if (!hit(session, 1)) return 'Mold Breaker Play Rough never damaged the Mimikyu';
      return null;
    }},
  {name: 'control_playrough_disguise',
    p1: team(PK_TINKATON, FILL().slice(1, 6)), p2: team(setOf('Mimikyu', 'Disguise', ['Play Rough', 'Shadow Claw', 'Protect']), FILL().slice(1, 6)),
    seeds: [[3, 5, 7, 621], [11, 13, 17, 622]], script: script('playrough', 'shadowclaw'), coverage: {ability: 'disguise', move: 'playrough', control: true},
    verify(fixture, session) {
      if (!disguised(session, 1)) return 'Disguise never absorbed the control Play Rough';
      return null;
    }},
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
    if (!session.battle.ended) { session.destroy(); lastReason = 'did not complete'; continue; }
    const reason = trial.verify(fixture, session);
    if (reason) { if (process.env.PA3_MB_DEBUG) console.log('DEBUG', trial.name, reason, '\n' + session.battle.log.filter(l => /^\|(turn|move|damage|status|immune|activate|faint|-)/.test(l)).slice(0, 18).join('\n')); session.destroy(); lastReason = reason; continue; }
    fixture.coverage = trial.coverage ?? {};
    recorded = fixture;
    session.destroy();
    break;
  }
  if (recorded) fixtures.push(recorded);
  else skipped.push({name: trial.name, reason: lastReason ?? 'no seed produced the required behavior'});
}
fs.writeFileSync(new URL('../data/more_moldbreaker.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify({trials: TRIALS.length, fixtures: fixtures.length, skipped: skipped.length}));
for (const s of skipped) console.log('SKIP', JSON.stringify(s));
