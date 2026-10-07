// Development-only generator for ability interaction fixtures.
//
// One complete legal battle per ability interaction, recorded at every decision
// boundary from the pinned reference (RNG seed, HP, status, boosts, types,
// items, volatiles, side/field effects and request legality). The Rust test
// `ability_interactions.rs` replays them natively: fixtures whose featured
// ability is not yet ported (`Ability::is_ported()` false) are smoke-checked
// with the feature swapped out and become full boundary comparisons the moment
// the port lands. Synthetic teams are mechanics fixtures only, never
// training-pool additions; Showdown is never on the training path.
import fs from 'node:fs';
import crypto from 'node:crypto';
import {ReferenceSession, verifyReference, FORMAT, ORACLE_COMMIT} from '../reference.mjs';
import {createRequire} from 'node:module';
const require = createRequire(import.meta.url);
const {TeamValidator, toID} = require('../../vendor/pokemon-showdown/dist/sim');
const validator = new TeamValidator(FORMAT);
const dex = validator.dex;
const data = JSON.parse(fs.readFileSync(new URL('../data/dex.json', import.meta.url), 'utf8'));
const scope = JSON.parse(fs.readFileSync(new URL('../data/scope.json', import.meta.url), 'utf8'));
const ids = Object.fromEntries(Object.entries(data.tables).map(([k, rows]) => [k,
  Object.fromEntries(rows.map(r => [r.id, r.numeric_id]))]));
const stats = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
verifyReference();

// Ported-ability mirror parsed from the native gate in hooks.rs. A fixture's
// featured ability may be pending, but every other ability in a fixture must be
// ported so the pending smoke replay stays inside the native subset.
const RUST_HOOKS = fs.readFileSync(new URL('../src/battle/hooks.rs', import.meta.url), 'utf8');
const gateBlock = RUST_HOOKS.match(/pub fn is_ported\(self\) -> bool \{[\s\S]*?!matches!\(\s*self,\s*Ability::Unimplemented([\s\S]*?)\)\s*\}/);
if (!gateBlock) throw new Error('could not read is_ported() gate from hooks.rs');
const unported = new Set([...gateBlock[1].matchAll(/Ability::([A-Za-z0-9]+)/g)].map(m => m[1]));
const variant = id => id[0].toUpperCase() + id.slice(1);
const abilityIdOf = name => toID(dex.abilities.get(name).name);
const portedAbility = name => dex.abilities.get(name).exists && !unported.has(variant(abilityIdOf(name)));

const setOf = (name, species, ability, moves, item, evs) => ({
  name, species, ability, item: item || '', nature: 'Serious', level: 50, gender: 'M', moves,
  evs: evs ?? {hp: 24, atk: 8, def: 8, spa: 8, spd: 8, spe: 4},
  ivs: {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31},
});

const FILLERS = [
  ['Alakazam', 'Synchronize', 'Calm Mind'],
  ['Appletun', 'Thick Fat', 'Iron Defense'],
  ['Avalugg-Hisui', 'Strong Jaw', 'Iron Defense'],
  ['Chimecho', 'Levitate', 'Calm Mind'],
  ['Medicham', 'Pure Power', 'Agility'],
  ['Reuniclus', 'Overcoat', 'Iron Defense'],
  ['Sableye', 'Prankster', 'Calm Mind'],
  ['Starmie', 'Natural Cure', 'Agility'],
  ['Toxapex', 'Limber', 'Iron Defense'],
  ['Pincurchin', 'Lightning Rod', 'Calm Mind'],
  ['Hydrapple', 'Regenerator', 'Iron Defense'],
];
const fillerTeam = () => FILLERS.map(([species, ability, setup]) =>
  setOf('s0m0', species, ability, [setup, 'Protect', 'Recover'], '', {hp: 32, def: 16, spd: 16, spe: 2}));
const learnable = entry => new Set(entry.learnable_moves || []);
const abilityIds = species => Object.values(dex.species.get(species).abilities || {}).map(toID);

// First legal starting species holding `ability` that can learn every pool move.
function findHolder(ability, pool, primary, trialControl) {
  const generic = ['Psychic', 'Dazzling Gleam', 'Body Slam', 'Surf', 'Flamethrower', 'Ice Beam', 'Tackle', 'Pound'];
  for (const entry of scope.starting_species) {
    if (!abilityIds(entry.species).includes(abilityIdOf(ability))) continue;
    const learn = learnable(entry);
    const picks = pool.filter(move => learn.has(toID(move)));
    if (!picks.some(move => toID(move) === 'protect')) continue;
    if (primary && !picks.some(move => toID(move) === toID(primary))) continue;
    // Control-probe trials need a legal neutral ability to compare against.
    if (trialControl && !neutralAbility(dex.species.get(entry.species).name, ability)) continue;
    for (const move of generic) if (picks.length < 3 && learn.has(toID(move)) && !picks.includes(move)) picks.push(move);
    if (picks.length < 2) continue;
    return {species: dex.species.get(entry.species).name, moves: picks.slice(0, 4)};
  }
  return null;
}

// First legal starting species that learns `move` and has a ported ability.
function findFoe(move, {preferAbility, item, species, moves: moveOverride} = {}) {
  const entries = species
    ? scope.starting_species.filter(entry => toID(entry.species) === toID(species))
    : scope.starting_species;
  for (const entry of entries) {
    const learn = learnable(entry);
    if (!learn.has(toID(move))) continue;
    const abilities = Object.values(dex.species.get(entry.species).abilities || {});
    const ability = preferAbility ?? abilities.find(portedAbility);
    if (!ability || !abilities.map(toID).includes(abilityIdOf(ability))) continue;
    if (preferAbility && !portedAbility(ability)) continue;
    const filler = ['recover', 'slackoff', 'tackle', 'pound'].find(m => learn.has(m));
    const pool = moveOverride ?? [move, 'Protect', ...(filler ? [filler] : [])];
    if (!pool.every(name => learn.has(toID(name)))) continue;
    const moves = [...new Set(pool.map(name => dex.moves.get(name).name))].slice(0, 4);
    return {species: dex.species.get(entry.species).name, ability, moves, item};
  }
  return null;
}

const roster = p => Number(p.name.slice(-1));
const volatileId = id => ids.conditions[id] ?? ids.moves[id] ?? 0;

// Privileged reference state at a decision boundary.
const compact = session => {
  const b = session.battle;
  const field = [
    ...(b.field.weather ? [[ids.conditions[b.field.weather], b.field.weatherState.duration, b.field.weatherState.source.side.n]] : []),
    ...(b.field.terrain ? [[ids.conditions[b.field.terrain], b.field.terrainState.duration, b.field.terrainState.source.side.n]] : []),
    ...Object.entries(b.field.pseudoWeather).map(([id, effect]) => [ids.conditions[id], effect.duration ?? 0, effect.source.side.n]),
  ].sort((a, c) => a[0] - c[0]);
  return {
    turn: b.turn,
    rng_seed: b.prng.getSeed(),
    ended: b.ended,
    winner: b.ended ? b.winner || null : null,
    field,
    sides: b.sides.map(s => ({
      request: b.ended ? 'Finished' : s.activeRequest?.wait || s.isChoiceDone() ? 'Wait' : s.requestState === 'teampreview' ? 'Preview' : s.requestState === 'switch' ? 'Replacement' : 'Normal',
      conditions: Object.entries(s.sideConditions).map(([id, state]) => [ids.conditions[id], state.duration ?? state.layers ?? 0]).sort((a, c) => a[0] - c[0]),
      pokemon: s.pokemon.map(p => ({
        roster: roster(p),
        species: ids.species[p.species.id] ?? 0,
        hp: p.hp,
        max_hp: p.maxhp,
        fainted: p.fainted,
        active_slot: s.active.indexOf(p) >= 0 ? s.active.indexOf(p) : null,
        status: ids.conditions[p.status] ?? 0,
        boosts: Object.values(p.boosts),
        stats: [p.maxhp, ...Object.values(p.storedStats)],
        ability: ids.abilities[p.ability] ?? 0,
        ability_ending: Boolean(p.abilityState.ending),
        item: ids.items[p.item] ?? 0,
        previous_item: ids.items[p.lastItem] ?? 0,
        types: p.types.map(t => ids.types[toID(t)] ?? 0),
        pp: p.moveSlots.map(m => m.pp),
        disabled: p.moveSlots.map(m => Boolean(m.disabled)),
        volatiles: Object.keys(p.volatiles).map(volatileId).sort((a, c) => a - c),
      })),
    })),
  };
};

const select = (kind, own_slot, destination = 255) => ({kind, own_slot, move_slot: 255, target_location: 0, switch_destination: destination, resource: 'None'});
const moveAction = (slot, moveSlot, target = 0) => ({kind: 'Move', own_slot: slot, move_slot: moveSlot, target_location: target, switch_destination: 255, resource: 'None'});

// Build one side's actions for a round. `want` = {move, partnerMove, switch, target}.
function choose(session, sideIndex, want) {
  const b = session.battle, side = b.sides[sideIndex], req = side.activeRequest;
  if (side.requestState === 'teampreview') {
    return {actions: [0, 1, 4, 5].map((r, i) => select('Pick', i, r)), command: 'team 1256'};
  }
  const bench = side.pokemon.map((p, i) => [p, i]).filter(([p]) => !p.fainted && !side.active.includes(p));
  const actions = [], commands = [];
  for (let slot = 0; slot < 2; slot++) {
    const p = side.active[slot];
    if (side.requestState === 'switch') {
      const forced = Boolean(req?.forceSwitch?.[slot]);
      // Non-forced slots are pass-only in the reference command and produce no
      // native branch action; only forced slots carry an action.
      if (!forced) { commands.push('pass'); continue; }
      const wanted = want.switch !== undefined && slot === 0 ? bench.find(([, i]) => i === want.switch) : undefined;
      const reserve = wanted ?? bench.shift();
      if (reserve) {
        actions.push(select('Switch', slot, roster(reserve[0])));
        commands.push(`switch ${side.pokemon.indexOf(reserve[0]) + 1}`);
      }
      else { actions.push(select('Pass', slot)); commands.push('pass'); }
      continue;
    }
    if (!p || p.fainted) { actions.push(select('Pass', slot)); commands.push('pass'); continue; }
    if (slot === 0 && want.switch !== undefined) {
      const target = side.pokemon.find(pp => roster(pp) === want.switch);
      actions.push(select('Switch', slot, want.switch));
      commands.push(`switch ${side.pokemon.indexOf(target) + 1}`);
      continue;
    }
    const wantedName = slot === 0 ? want.move : want.partnerMove;
    const wantedKey = wantedName ? toID(dex.moves.get(wantedName).name) : undefined;
    const slotIndex = wantedKey ? p.moveSlots.findIndex(m => m.id === wantedKey && !m.disabled && m.pp > 0) : -1;
    const fallback = p.moveSlots.findIndex(m => !m.disabled && m.pp > 0);
    const choice = slotIndex >= 0 ? slotIndex : fallback >= 0 ? fallback : 0;
    const chosen = p.moveSlots[choice];
    let location = 0;
    if (chosen && chosen.pp > 0 && b.actions.targetTypeChoices(chosen.target)) {
      const candidates = [1, 2, -1, -2];
      const match = want.target
        ? candidates.find(loc => b.validTargetLoc(loc, p, chosen.target) && p.getAtLoc(loc) === want.target)
        : candidates.find(loc => b.validTargetLoc(loc, p, chosen.target) && p.getAtLoc(loc)?.side !== p.side);
      location = match ?? 0;
    }
    actions.push(moveAction(slot, choice, location));
    commands.push(`move ${choice + 1}${location ? ` ${location}` : ''}`);
  }
  return {actions, command: commands.join(', ')};
}

const logHas = (session, needle) => session.battle.log.some(line => line.includes(needle));
const neutralAbility = (species, featured) => {
  const abilities = Object.values(dex.species.get(species).abilities || {});
  return abilities.find(name => portedAbility(name) && toID(name) !== toID(featured));
};

// Replays the recorded commands in a control battle where the featured ability
// is replaced by a legal neutral ability. A pure damage/accuracy/priority
// modifier has no public message, so its trigger proof is a state difference
// against this control at the same seed and commands.
function controlDiffers(session, commands, fixture) {
  for (const [index, command] of commands.entries()) {
    const result = session.choose(command.side, command.command);
    if (!result.accepted) return false;
    const expected = fixture.steps[index]?.expected;
    const actual = compact(session);
    for (let side = 0; side < 2; side++) {
      for (const mon of actual.sides[side].pokemon) {
        const base = expected?.sides?.[side]?.pokemon?.[mon.roster];
        if (!base) continue;
        if (mon.hp !== base.hp || mon.status !== base.status ||
            JSON.stringify(mon.boosts) !== JSON.stringify(base.boosts) ||
            JSON.stringify(mon.volatiles) !== JSON.stringify(base.volatiles)) {
          return true;
        }
      }
    }
  }
  return false;
}

const TRIALS = [
  {
    name: 'goodasgold_status_immune',
    ability: 'Good as Gold',
    holderPool: ['Protect', 'Make It Rain', 'Shadow Ball'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Toxic'},
    rounds: 3,
    probe: session => ({ok: logHas(session, 'Good as Gold'), detail: {log: session.battle.log.filter(l => l.includes('Good as Gold'))}}),
  },
  {
    name: 'armortail_priority_block',
    ability: 'Armor Tail',
    holderPool: ['Protect', 'Body Slam', 'Foul Play'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Aqua Jet'},
    rounds: 3,
    probe: session => ({ok: logHas(session, 'Armor Tail'), detail: {log: session.battle.log.filter(l => l.includes('Armor Tail'))}}),
  },
  {
    name: 'damp_blocks_explosion',
    ability: 'Damp',
    holderPool: ['Protect', 'Surf', 'Ice Beam'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Explosion'},
    rounds: 3,
    probe: session => ({ok: logHas(session, 'Damp'), detail: {log: session.battle.log.filter(l => l.includes('Damp'))}}),
  },
  {
    name: 'disguise_absorbs_first_hit',
    primary: 'Shadow Claw',
    ability: 'Disguise',
    holderPool: ['Protect', 'Shadow Claw', 'Play Rough'],
    holderPlan: () => ({move: 'shadowclaw'}),
    foe: {move: 'Iron Head'},
    rounds: 4,
    probe: session => ({ok: logHas(session, 'Disguise'), detail: {species: session.battle.sides[0].pokemon[0].species.name}}),
  },
  {
    name: 'mirrorarmor_reflects_intimidate',
    ability: 'Mirror Armor',
    holderPool: ['Protect', 'Iron Head', 'Body Press'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Protect', ability: 'Intimidate'},
    rounds: 3,
    probe: session => ({ok: logHas(session, 'Mirror Armor'),
      detail: {foeBoosts: {...session.battle.sides[1].pokemon[0].boosts}, holderBoosts: {...session.battle.sides[0].pokemon[0].boosts}}}),
  },
  {
    name: 'toxicdebris_contact_spikes',
    primary: 'Power Gem',
    ability: 'Toxic Debris',
    holderPool: ['Protect', 'Power Gem', 'Sludge Wave'],
    holderPlan: () => ({move: 'powergem'}),
    foe: {move: 'Iron Head'},
    rounds: 4,
    probe: session => ({ok: Boolean(session.battle.sides[1].sideConditions['toxicspikes']),
      detail: {conditions: Object.keys(session.battle.sides[1].sideConditions)}}),
  },
  {
    name: 'cursedbody_contact_disable',
    primary: 'Icy Wind',
    ability: 'Cursed Body',
    holderPool: ['Protect', 'Shadow Ball', 'Icy Wind'],
    holderPlan: () => ({move: 'icywind'}),
    foe: {move: 'Iron Head'},
    rounds: 6,
    probe: session => {
      const foe = session.battle.sides[1].pokemon[0];
      return {ok: Boolean(foe.volatiles['disable']) || foe.moveSlots.some(m => m.disabled),
        detail: {volatiles: Object.keys(foe.volatiles), disabled: foe.moveSlots.map(m => m.disabled)}};
    },
  },
  {
    name: 'thermalexchange_fire_attack',
    primary: 'Icicle Crash',
    ability: 'Thermal Exchange',
    holderPool: ['Protect', 'Icicle Crash', 'Earthquake'],
    holderPlan: () => ({move: 'iciclecrash'}),
    foe: {move: 'Flamethrower'},
    rounds: 4,
    probe: session => ({ok: session.battle.sides[0].pokemon[0].boosts.atk >= 1,
      detail: {boosts: {...session.battle.sides[0].pokemon[0].boosts}}}),
  },
  {
    name: 'trace_copies_foe_ability',
    ability: 'Trace',
    holderPool: ['Protect', 'Ice Beam', 'Recover'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Protect', ability: 'Levitate'},
    rounds: 3,
    probe: session => ({ok: toID(session.battle.sides[0].pokemon[0].ability) === 'levitate',
      detail: {ability: session.battle.sides[0].pokemon[0].ability}}),
  },
  {
    name: 'unburden_item_loss_speed',
    primary: 'Swords Dance',
    ability: 'Unburden',
    item: 'Sitrus Berry',
    holderPool: ['Swords Dance', 'Protect', 'Close Combat'],
    holderPlan: () => ({move: 'swordsdance'}),
    foe: {move: 'Knock Off'},
    rounds: 4,
    probe: session => ({ok: Boolean(session.battle.sides[0].pokemon[0].volatiles['unburden']),
      detail: {volatiles: Object.keys(session.battle.sides[0].pokemon[0].volatiles)}}),
  },
  {
    name: 'hospitality_switch_heal',
    ability: 'Hospitality',
    holderPool: ['Protect', 'Matcha Gotcha', 'Shadow Ball'],
    holderBench: true,
    holderPlan: () => ({move: 'protect'}),
    partner: {species: 'Perrserker', ability: 'Battle Armor', moves: ['Iron Head', 'Protect']},
    foe: {move: 'Dazzling Gleam'},
    rounds: 5,
    probe: session => ({ok: logHas(session, '|-heal|p1'),
      detail: {log: session.battle.log.filter(l => l.startsWith('|-heal|'))}}),
  },
  {
    name: 'limber_paralysis_immune',
    ability: 'Limber',
    holderPool: ['Protect', 'Body Slam', 'Psychic'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Thunder Wave'},
    rounds: 3,
    probe: session => ({ok: logHas(session, 'Limber'), detail: {status: session.battle.sides[0].pokemon[0].status}}),
  },
  {
    name: 'immunity_poison_immune',
    ability: 'Immunity',
    holderPool: ['Protect', 'Body Slam', 'Crunch'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Toxic'},
    rounds: 3,
    probe: session => ({ok: logHas(session, 'Immunity'), detail: {status: session.battle.sides[0].pokemon[0].status}}),
  },
  {
    name: 'insomnia_sleep_immune',
    ability: 'Insomnia',
    holderPool: ['Protect', 'Body Slam', 'Crunch'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Hypnosis'},
    rounds: 4,
    probe: session => ({ok: logHas(session, 'Insomnia'), detail: {status: session.battle.sides[0].pokemon[0].status}}),
  },
  {
    name: 'waterbubble_burn_immune',
    ability: 'Water Bubble',
    holderPool: ['Protect', 'Surf', 'Ice Beam'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Will-O-Wisp'},
    rounds: 3,
    probe: session => ({ok: logHas(session, 'Water Bubble'), detail: {status: session.battle.sides[0].pokemon[0].status}}),
  },
  {
    name: 'telepathy_ally_spread',
    ability: 'Telepathy',
    holderPool: ['Protect', 'Psychic', 'Dazzling Gleam'],
    holderPlan: () => ({move: 'protect'}),
    partner: {species: 'Milotic', ability: 'Competitive', moves: ['Surf', 'Protect']},
    partnerMove: 'Surf',
    foe: {move: 'Protect'},
    rounds: 3,
    probe: session => ({ok: logHas(session, 'Telepathy'), detail: {holderHp: session.battle.sides[0].pokemon[0].hp}}),
  },
  {
    name: 'moldbreaker_ignores_levitate',
    primary: 'Earthquake',
    ability: 'Mold Breaker',
    holderPool: ['Protect', 'Earthquake', 'Iron Head'],
    holderPlan: () => ({move: 'earthquake'}),
    foe: {move: 'Protect', ability: 'Levitate'},
    rounds: 3,
    probe: session => ({ok: logHas(session, 'Mold Breaker'), detail: {foeHp: session.battle.sides[1].pokemon[0].hp}}),
  },
  {
    name: 'scrappy_hits_ghost',
    primary: 'Body Slam',
    ability: 'Scrappy',
    holderPool: ['Protect', 'Body Slam', 'Close Combat'],
    holderPlan: () => ({move: 'bodyslam'}),
    foe: {move: 'Protect', species: 'Trevenant', ability: 'Natural Cure'},
    rounds: 4,
    probe: session => ({ok: session.battle.sides[1].pokemon[0].hp < session.battle.sides[1].pokemon[0].maxhp,
      detail: {foeHp: session.battle.sides[1].pokemon[0].hp}}),
  },
  {
    name: 'contrary_selfdrop_inverted',
    primary: 'Leaf Storm',
    ability: 'Contrary',
    holderPool: ['Protect', 'Leaf Storm', 'Giga Drain'],
    holderPlan: () => ({move: 'leafstorm'}),
    foe: {move: 'Protect'},
    rounds: 3,
    probe: session => ({ok: session.battle.sides[0].pokemon[0].boosts.spa >= 2,
      detail: {boosts: {...session.battle.sides[0].pokemon[0].boosts}}}),
  },
  {
    name: 'magicbounce_reflects_status',
    ability: 'Magic Bounce',
    holderPool: ['Protect', 'Psychic', 'Dazzling Gleam'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Toxic'},
    rounds: 3,
    probe: session => ({ok: Boolean(session.battle.sides[1].pokemon[0].status) || logHas(session, 'Magic Bounce'),
      detail: {foeStatus: session.battle.sides[1].pokemon[0].status, log: session.battle.log.filter(l => l.includes('Magic Bounce'))}}),
  },
  {
    name: 'queenlymajesty_priority_block',
    ability: 'Queenly Majesty',
    holderPool: ['Protect', 'Body Slam', 'Foul Play'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Aqua Jet'},
    rounds: 3,
    probe: session => ({ok: logHas(session, 'Queenly Majesty'),
      detail: {log: session.battle.log.filter(l => l.includes('Queenly Majesty'))}}),
  },
  {
    name: 'sturdy_survives_ohko',
    ability: 'Sturdy',
    holderPool: ['Protect', 'Iron Head', 'Body Press'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Focus Blast'},
    rounds: 3,
    probe: session => ({ok: logHas(session, 'Sturdy'), detail: {hp: session.battle.sides[0].pokemon[0].hp}}),
  },
  {
    name: 'moxie_after_ko',
    primary: 'Crunch',
    ability: 'Moxie',
    holderPool: ['Protect', 'Crunch', 'Close Combat'],
    holderPlan: () => ({move: 'crunch'}),
    foe: {move: 'Protect'},
    rounds: 14,
    probe: session => ({ok: session.battle.sides[0].pokemon[0].boosts.atk >= 1,
      detail: {boosts: {...session.battle.sides[0].pokemon[0].boosts}, foeHp: session.battle.sides[1].pokemon[0].hp}}),
  },
  {
    name: 'unnerve_blocks_berry',
    ability: 'Unnerve',
    holderPool: ['Protect', 'Body Slam', 'Crunch'],
    holderPlan: () => ({move: 'protect'}),
    partner: {species: 'Falinks', ability: 'Battle Armor', moves: ['Close Combat', 'Protect']},
    partnerMove: 'closecombat',
    foe: {move: 'Protect', item: 'Sitrus Berry'},
    rounds: 8,
    probe: session => {
      const foe = session.battle.sides[1].pokemon[0];
      return {ok: Boolean(foe.item) && foe.hp <= foe.maxhp / 2 && logHas(session, 'Unnerve'),
        detail: {item: foe.item, hp: foe.hp, maxhp: foe.maxhp, log: session.battle.log.filter(l => l.includes('Unnerve'))}};
    },
  },
  {
    name: 'flowerveil_ally_status_guard',
    ability: 'Flower Veil',
    holderPool: ['Protect', 'Dazzling Gleam', 'Psychic'],
    holderPlan: () => ({move: 'protect'}),
    partner: {species: 'Torterra', ability: 'Shell Armor', moves: ['Seed Bomb', 'Protect']},
    foe: {move: 'Toxic', targetAlly: true},
    rounds: 3,
    probe: session => ({ok: logHas(session, 'Flower Veil'),
      detail: {allyStatus: session.battle.sides[0].pokemon[1].status, log: session.battle.log.filter(l => l.includes('Flower Veil'))}}),
  },
  {
    name: 'soundproof_blocks_sound',
    ability: 'Soundproof',
    holderPool: ['Protect', 'Iron Head', 'Dragon Claw'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Snarl'},
    rounds: 3,
    probe: session => ({ok: logHas(session, 'Soundproof'),
      detail: {log: session.battle.log.filter(l => l.includes('Soundproof'))}}),
  },
  {
    name: 'bulletproof_blocks_ball',
    ability: 'Bulletproof',
    holderPool: ['Protect', 'Iron Head', 'Body Press'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Shadow Ball'},
    rounds: 3,
    probe: session => ({ok: logHas(session, 'Bulletproof'),
      detail: {log: session.battle.log.filter(l => l.includes('Bulletproof'))}}),
  },
  {
    name: 'weakarmor_contact_boost',
    ability: 'Weak Armor',
    primary: 'Iron Head',
    holderPool: ['Iron Head', 'Protect', 'Body Press'],
    holderPlan: () => ({move: 'ironhead'}),
    foe: {move: 'Iron Head'},
    rounds: 4,
    probe: session => ({ok: session.battle.sides[0].pokemon[0].boosts.spe >= 2,
      detail: {boosts: {...session.battle.sides[0].pokemon[0].boosts}}}),
  },
  {
    name: 'justified_dark_boost',
    ability: 'Justified',
    primary: 'Close Combat',
    holderPool: ['Close Combat', 'Protect', 'Iron Head'],
    holderPlan: () => ({move: 'closecombat'}),
    foe: {move: 'Crunch'},
    rounds: 4,
    probe: session => ({ok: session.battle.sides[0].pokemon[0].boosts.atk >= 1,
      detail: {boosts: {...session.battle.sides[0].pokemon[0].boosts}}}),
  },
  {
    name: 'electromorphosis_electric_charge',
    ability: 'Electromorphosis',
    holderPool: ['Protect', 'Thunderbolt', 'Discharge'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Thunderbolt'},
    rounds: 4,
    probe: session => ({ok: Boolean(session.battle.sides[0].pokemon[0].volatiles['charge']),
      detail: {volatiles: Object.keys(session.battle.sides[0].pokemon[0].volatiles)}}),
  },
  {
    name: 'friendguard_ally_damage',
    ability: 'Friend Guard',
    holderPool: ['Protect', 'Dazzling Gleam', 'Psychic'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Iron Head', targetAlly: true},
    control: true,
    rounds: 4,
  },
  {
    name: 'unaware_ignores_boosts',
    ability: 'Unaware',
    holderPool: ['Protect', 'Body Slam', 'Crunch'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Iron Head', moves: ['Swords Dance', 'Iron Head', 'Protect']},
    foePlan: round => ({move: round === 0 ? 'Swords Dance' : 'Iron Head'}),
    control: true,
    rounds: 5,
  },
  {
    name: 'punkrock_sound_boost',
    ability: 'Punk Rock',
    primary: 'Snarl',
    holderPool: ['Snarl', 'Protect', 'Dark Pulse'],
    holderPlan: () => ({move: 'snarl'}),
    foe: {move: 'Protect'},
    control: true,
    rounds: 4,
  },
  {
    name: 'sheerforce_attack_boost',
    ability: 'Sheer Force',
    primary: 'Iron Head',
    holderPool: ['Iron Head', 'Protect', 'Body Press'],
    holderPlan: () => ({move: 'ironhead'}),
    foe: {move: 'Calm Mind'},
    control: true,
    rounds: 5,
  },
  {
    name: 'furcoat_physical_halved',
    ability: 'Fur Coat',
    primary: 'Body Slam',
    holderPool: ['Protect', 'Body Slam', 'Dark Pulse'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Iron Head'},
    control: true,
    rounds: 4,
  },
  {
    name: 'heatproof_fire_halved',
    ability: 'Heatproof',
    holderPool: ['Protect', 'Calm Mind'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Flamethrower'},
    control: true,
    rounds: 4,
  },
  {
    name: 'fluffy_contact_halved',
    ability: 'Fluffy',
    holderPool: ['Protect', 'Body Slam', 'Crunch'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Iron Head'},
    control: true,
    rounds: 4,
  },
  {
    name: 'guts_burn_attack',
    ability: 'Guts',
    primary: 'Close Combat',
    holderPool: ['Close Combat', 'Protect', 'Iron Head'],
    holderPlan: round => ({move: round === 0 ? 'protect' : 'closecombat'}),
    foe: {move: 'Will-O-Wisp', plan: 'willowisp'},
    foePlan: round => ({move: round === 0 ? 'Will-O-Wisp' : 'Calm Mind'}),
    control: true,
    rounds: 5,
  },
  {
    name: 'compoundeyes_accuracy',
    ability: 'Compound Eyes',
    primary: 'Hurricane',
    holderPool: ['Hurricane', 'Protect', 'Psychic'],
    holderPlan: () => ({move: 'hurricane'}),
    foe: {move: 'Calm Mind'},
    control: true,
    rounds: 4,
  },
  {
    name: 'poisonheal_no_poison_damage',
    ability: 'Poison Heal',
    holderPool: ['Protect', 'Body Slam', 'Crunch'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Toxic'},
    control: true,
    rounds: 5,
  },
  {
    name: 'magicguard_sand_immunity',
    ability: 'Magic Guard',
    holderPool: ['Protect', 'Psychic', 'Dazzling Gleam'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Protect', ability: 'Sand Stream'},
    rounds: 4,
    probe: session => {
      const holder = session.battle.sides[0].pokemon[0];
      const ally = session.battle.sides[0].pokemon[1];
      return {ok: holder.hp === holder.maxhp && ally.hp < ally.maxhp,
        detail: {holder: `${holder.hp}/${holder.maxhp}`, ally: `${ally.hp}/${ally.maxhp}`,
          weather: session.battle.field.weather}};
    },
  },
  {
    name: 'purifyingsalt_status_immune',
    ability: 'Purifying Salt',
    holderPool: ['Protect', 'Body Slam', 'Iron Head'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Toxic'},
    rounds: 3,
    probe: session => ({ok: logHas(session, 'Purifying Salt'),
      detail: {status: session.battle.sides[0].pokemon[0].status, log: session.battle.log.filter(l => l.includes('Purifying Salt'))}}),
  },
  {
    name: 'galewings_priority',
    ability: 'Gale Wings',
    primary: 'Brave Bird',
    holderPool: ['Brave Bird', 'Protect', 'Flare Blitz'],
    holderPlan: () => ({move: 'bravebird'}),
    foe: {move: 'Psychic'},
    rounds: 3,
    probe: session => {
      const moves = session.battle.log.filter(line => line.startsWith('|move|'));
      return {ok: moves[0]?.startsWith('|move|p1a:'), detail: {moves: moves.slice(0, 4)}};
    },
  },
  {
    name: 'gooey_contact_speed_drop',
    ability: 'Gooey',
    holderPool: ['Protect', 'Body Slam', 'Crunch'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Iron Head'},
    rounds: 4,
    probe: session => ({ok: session.battle.sides[1].pokemon[0].boosts.spe <= -1,
      detail: {foeBoosts: {...session.battle.sides[1].pokemon[0].boosts}}}),
  },
  {
    name: 'effectspore_contact_status',
    ability: 'Effect Spore',
    holderPool: ['Protect', 'Body Slam', 'Crunch'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Iron Head'},
    rounds: 5,
    probe: session => ({ok: Boolean(session.battle.sides[1].pokemon[0].status) || logHas(session, 'Effect Spore'),
      detail: {foeStatus: session.battle.sides[1].pokemon[0].status}}),
  },
  {
    // `imprison` declares `mustpressure`, so the Pressure event names every
    // live foe even though the move targets the user.
    name: 'pressure_mustpressure_imprison',
    ability: 'Pressure',
    holderPool: ['Iron Head', 'Protect', 'Sucker Punch'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Imprison', ability: 'Sap Sipper'},
    rounds: 3,
    probe: session => {
      const foe = session.battle.sides[1].pokemon[0];
      const slot = foe.moveSlots.find(m => m.id === 'imprison');
      // Two uses must cost two PP each: one normal plus one from Pressure.
      return {ok: Boolean(slot) && slot.pp === slot.maxpp - 4,
        detail: {pp: slot?.pp, maxpp: slot?.maxpp, moves: foe.moveSlots.map(m => [m.id, m.pp, m.maxpp])}};
    },
  },
  {
    name: 'frisk_reveals_foe_items',
    ability: 'Frisk',
    holderPool: ['Psychic', 'Protect', 'Calm Mind'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Protect', item: 'Leftovers', ability: 'Clear Body'},
    rounds: 2,
    probe: session => ({ok: logHas(session, 'ability: Frisk'),
      detail: {log: session.battle.log.filter(l => l.includes('Frisk'))}}),
  },
  {
    name: 'pressure_extra_pp',
    ability: 'Pressure',
    holderPool: ['Iron Head', 'Protect', 'Sucker Punch'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Taunt', ability: 'Levitate'},
    // The harness counts the teampreview round, so three rounds = two Taunts.
    rounds: 3,
    probe: session => {
      const foe = session.battle.sides[1].pokemon[0];
      const slot = foe.moveSlots.find(m => m.id === 'taunt');
      // Two uses must cost two PP each: one normal plus one from Pressure.
      return {ok: Boolean(slot) && slot.pp === slot.maxpp - 4,
        detail: {pp: slot?.pp, maxpp: slot?.maxpp, moves: foe.moveSlots.map(m => [m.id, m.pp, m.maxpp])}};
    },
  },
  {
    name: 'zerotohero_switch_form',
    ability: 'Zero to Hero',
    primary: 'Protect',
    holderPool: ['Protect', 'Surf', 'Bulk Up'],
    holderPlan: round => (round === 1 ? {switch: 4} : round === 2 ? {switch: 0} : {move: 'protect'}),
    foe: {move: 'Protect', ability: 'Clear Body'},
    rounds: 5,
    probe: session => {
      const palafin = session.battle.sides[0].pokemon.find(p => p.baseSpecies.baseSpecies === 'Palafin');
      return {ok: palafin?.species.forme === 'Hero' && logHas(session, 'ability: Zero to Hero'),
        detail: {species: palafin?.species.id, forme: palafin?.species.forme,
          log: session.battle.log.filter(l => l.includes('Zero to Hero') || l.startsWith('|detailschange'))}};
    },
  },
  {
    name: 'intimidate_lowers_foe_attack',
    ability: 'Intimidate',
    holderPool: ['Protect', 'Body Slam', 'Crunch'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Body Slam', species: 'Milotic'},
    rounds: 2,
    probe: session => {
      const foe = session.battle.sides[1].pokemon[0];
      return {ok: foe.boosts.atk === -1, detail: {foeBoosts: {...foe.boosts}}};
    },
  },
  {
    name: 'defiant_boosts_attack_on_drop',
    ability: 'Defiant',
    holderPool: ['Protect', 'Iron Head', 'Crunch'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Screech', species: 'Aggron'},
    rounds: 3,
    probe: session => {
      const holder = session.battle.sides[0].pokemon[0];
      return {ok: holder.boosts.atk === 2, detail: {holderBoosts: {...holder.boosts}}};
    },
  },
  {
    name: 'prankster_status_priority_taunt',
    ability: 'Prankster',
    holderPool: ['Protect', 'Taunt', 'Foul Play'],
    holderPlan: () => ({move: 'taunt'}),
    foe: {move: 'Body Slam', species: 'Milotic'},
    rounds: 3,
    probe: session => {
      const foe = session.battle.sides[1].pokemon[0];
      return {ok: Boolean(foe.volatiles['taunt']), detail: {volatiles: Object.keys(foe.volatiles)}};
    },
  },
  {
    name: 'stamina_boosts_defense_on_hit',
    ability: 'Stamina',
    holderPool: ['Protect', 'Body Slam', 'Iron Head'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Body Slam', species: 'Milotic'},
    rounds: 3,
    probe: session => {
      const holder = session.battle.sides[0].pokemon[0];
      return {ok: holder.boosts.def >= 1, detail: {holderBoosts: {...holder.boosts}}};
    },
  },
  {
    name: 'roughskin_contact_damage',
    ability: 'Rough Skin',
    holderPool: ['Protect', 'Body Slam', 'Surf'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Body Slam', species: 'Milotic'},
    rounds: 3,
    probe: session => {
      const foe = session.battle.sides[1].pokemon[0];
      return {ok: foe.hp < foe.maxhp, detail: {foeHp: foe.hp, foeMax: foe.maxhp}};
    },
  },
  {
    name: 'drought_sun_on_entry',
    ability: 'Drought',
    holderPool: ['Protect', 'Flamethrower', 'Solar Beam'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Body Slam', species: 'Milotic'},
    rounds: 2,
    probe: session => ({ok: session.battle.field.weather === 'sunnyday', detail: {weather: session.battle.field.weather}}),
  },
  {
    name: 'drizzle_rain_on_entry',
    ability: 'Drizzle',
    holderPool: ['Protect', 'Surf', 'Ice Beam'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Body Slam', species: 'Milotic'},
    rounds: 2,
    probe: session => ({ok: session.battle.field.weather === 'raindance', detail: {weather: session.battle.field.weather}}),
  },
  {
    name: 'sandstream_sand_on_entry',
    ability: 'Sand Stream',
    holderPool: ['Protect', 'Earthquake', 'Rock Slide'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Body Slam', species: 'Milotic'},
    rounds: 2,
    probe: session => ({ok: session.battle.field.weather === 'sandstorm', detail: {weather: session.battle.field.weather}}),
  },
  {
    name: 'snowwarning_snow_on_entry',
    ability: 'Snow Warning',
    holderPool: ['Protect', 'Ice Beam', 'Blizzard'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Body Slam', species: 'Milotic'},
    rounds: 2,
    probe: session => ({ok: session.battle.field.weather === 'snowscape', detail: {weather: session.battle.field.weather}}),
  },
  {
    name: 'grassysurge_terrain_on_entry',
    ability: 'Grassy Surge',
    holderPool: ['Protect', 'Grassy Glide', 'Seed Bomb'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Body Slam', species: 'Milotic'},
    rounds: 2,
    probe: session => ({ok: session.battle.field.terrain === 'grassyterrain', detail: {terrain: session.battle.field.terrain}}),
  },
  {
    name: 'psychicsurge_terrain_on_entry',
    ability: 'Psychic Surge',
    holderPool: ['Protect', 'Psychic', 'Psyshock'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Body Slam', species: 'Milotic'},
    rounds: 2,
    probe: session => ({ok: session.battle.field.terrain === 'psychicterrain', detail: {terrain: session.battle.field.terrain}}),
  },
  {
    name: 'electricsurge_terrain_on_entry',
    ability: 'Electric Surge',
    holderPool: ['Protect', 'Thunderbolt', 'Volt Switch'],
    holderPlan: () => ({move: 'protect'}),
    foe: {move: 'Body Slam', species: 'Milotic'},
    rounds: 2,
    probe: session => ({ok: session.battle.field.terrain === 'electricterrain', detail: {terrain: session.battle.field.terrain}}),
  },
];

const fixtures = [];
const skipped = [];
for (const [index, trial] of TRIALS.entries()) {
  const holder = findHolder(trial.ability, trial.holderPool, trial.primary, trial.control);
  if (!holder) { skipped.push({name: trial.name, reason: 'no legal holder for pool'}); continue; }
  const foe = findFoe(trial.foe.move, {preferAbility: trial.foe.ability, item: trial.foe.item,
    species: trial.foe.species, moves: trial.foe.moves});
  if (!foe) { skipped.push({name: trial.name, reason: 'no legal foe'}); continue; }
  const holderSet = setOf('s0m0', holder.species, dex.abilities.get(trial.ability).name, holder.moves, trial.item);
  const used = new Set([toID(holderSet.species)]);
  const spare = () => {
    const pick = fillerTeam().find(s => !used.has(toID(s.species)));
    used.add(toID(pick.species));
    return pick;
  };
  const partnerSet = trial.partner
    ? setOf('s0m1', trial.partner.species, trial.partner.ability, trial.partner.moves, '', {hp: 32, atk: 16, def: 16, spa: 2, spd: 0, spe: 0})
    : spare();
  used.add(toID(partnerSet.species));
  const teamA = trial.holderBench
    ? [partnerSet, spare(), spare(), spare(), holderSet, spare()]
    : [holderSet, partnerSet, spare(), spare(), spare(), spare()];
  teamA.forEach((s, i) => { s.name = `s0m${i}`; });
  const foeSet = setOf('s1m0', foe.species, foe.ability, foe.moves, foe.item);
  const teamB = [foeSet, ...fillerTeam().filter(s => toID(s.species) !== toID(foeSet.species)).slice(0, 5)];
  teamB.forEach((s, i) => { s.name = `s1m${i}`; });
  const problems = validator.validateTeam(teamA) || validator.validateTeam(teamB);
  if (problems) { skipped.push({name: trial.name, reason: problems.join('; ')}); continue; }
  const seeds = Array.from({length: 64}, (_, k) => [2026, 10, 7, 6000 + index * 64 + k]);
  let recorded = null, lastProbe = null, failure = null;
  for (const seed of seeds) {
    const session = new ReferenceSession({teams: [teamA, teamB], seed});
    const fixture = {name: `${trial.name}_${seed[3]}`, seed, ability: toID(trial.ability),
      teams: [teamA, teamB].map((team, side) => ({id: `ability-${trial.name}-${side}`, members: team.map(s => ({
        species: ids.species[toID(s.species)], ability: ids.abilities[toID(s.ability)], item: ids.items[toID(s.item)] ?? 0,
        nature: ids.natures[toID(s.nature)], gender: s.gender || '', level: 50, moves: s.moves.map(x => ids.moves[toID(x)]),
        points: stats.map(k => s.evs[k] ?? 0), ivs: stats.map(k => s.ivs[k] ?? 31)}))})),
      initial: compact(session), steps: []};
    let round = 0;
    while (!session.battle.ended && round < trial.rounds && fixture.steps.length < 120) {
      for (let side = 0; side < 2; side++) {
        if (session.battle.ended) break;
        const s = session.battle.sides[side];
        if (s.activeRequest?.wait || s.isChoiceDone()) continue;
        const battleTurn = session.battle.turn;
        let want;
        if (side === 0) {
          const switchIn = trial.holderBench && battleTurn === 2 ? {switch: 4} : {};
          want = {...trial.holderPlan(round), ...switchIn, ...(trial.partnerMove ? {partnerMove: trial.partnerMove} : {})};
        } else {
          const foeTarget = trial.foe.targetAlly || (trial.holderBench && battleTurn === 1);
          const foeMove = trial.foePlan ? trial.foePlan(round).move : trial.foe.move;
          want = {move: foeMove, target: foeTarget
            ? session.battle.sides[0].active[1]
            : session.battle.sides[0].active[0]};
        }
        const choice = choose(session, side, want);
        const result = session.choose(side ? 'p2' : 'p1', choice.command);
        if (!result.accepted) { failure = JSON.stringify({side, round, choice, messages: result.messages}); break; }
        fixture.steps.push({side: side ? 'P2' : 'P1', ...choice, expected: compact(session)});
      }
      if (failure) break;
      round++;
    }
    let probe = failure ? null : trial.probe ? trial.probe(session) : {ok: true, detail: {}};
    if (!failure && probe?.ok && trial.control) {
      // Control battle: same seed and commands with a neutral legal ability.
      const swapped = [teamA, teamB].map(team => team.map(set =>
        toID(set.ability) === toID(trial.ability)
          ? {...set, ability: neutralAbility(set.species, trial.ability)}
          : set));
      if (swapped.some(team => team.some(set => !set.ability))) {
        probe = {ok: false, detail: {...probe.detail, control: 'no neutral ability'}};
      } else {
        const control = new ReferenceSession({teams: swapped, seed});
        const commands = fixture.steps.map(step => ({side: step.side === 'P1' ? 'p1' : 'p2', command: step.command}));
        const fired = controlDiffers(control, commands, fixture);
        control.destroy();
        if (!fired) probe = {ok: false, detail: {...probe.detail, control: 'identical to control'}};
      }
    }
    if (!failure && probe?.ok) {
      fixture.coverage = {ability: toID(trial.ability), holder: holder.species, foe: foe.species, detail: probe.detail};
      recorded = fixture;
      session.destroy();
      break;
    }
    lastProbe = probe?.detail ?? lastProbe;
    session.destroy();
  }
  if (recorded) fixtures.push(recorded);
  else skipped.push({name: trial.name, reason: failure ?? `probe never fired (${JSON.stringify(lastProbe)})`});
}
// Staleness guard: the Rust corpus test rehashes this generator and fails if
// the corpus was not regenerated after a change.
const generatorSha256 = crypto.createHash('sha256').update(fs.readFileSync(new URL(import.meta.url))).digest('hex');
fs.writeFileSync(new URL('../data/ability-interactions.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, generator_sha256: generatorSha256, fixtures}) + '\n');
console.log(JSON.stringify({trials: TRIALS.length, fixtures: fixtures.length, skipped: skipped.length}));
for (const s of skipped) console.log(JSON.stringify(s));
