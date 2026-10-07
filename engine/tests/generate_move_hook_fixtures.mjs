// Development-only generator for ported action-local move callbacks
// (Fake Out, Sucker Punch, Grassy Glide priority, weather accuracy,
// Freeze-Dry, Feint, stat overrides and selfBoost).
//
// Synthetic legal teams are mechanics fixtures only, never training-pool
// additions. Every fixture completes naturally and records the reference
// decision-boundary state (including request move legality) for the Rust
// differential test in `move_hooks.rs`.
import fs from 'node:fs';
import {createRequire} from 'node:module';
import {ReferenceSession, verifyReference, FORMAT, ORACLE_COMMIT} from '../reference.mjs';
const require = createRequire(import.meta.url);
const {TeamValidator, toID} = require('../../vendor/pokemon-showdown/dist/sim');
const dex = new TeamValidator(FORMAT).dex;
const data = JSON.parse(fs.readFileSync(new URL('../data/dex.json', import.meta.url), 'utf8'));
const ids = Object.fromEntries(Object.entries(data.tables).map(([k, rows]) => [k,
  Object.fromEntries(rows.map(r => [r.id, r.numeric_id]))]));
const stats = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const roster = p => Number(p.name.slice(-1));

const mk = (name, species, ability, item, moves, evs) => ({
  name, species, ability, item: item || '', nature: 'Serious', level: 50, gender: 'M', moves,
  evs: evs ?? {hp: 10, atk: 15, def: 5, spa: 15, spd: 5, spe: 0},
  ivs: {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31},
});

const filler = () => [
  mk('s0m2', 'Perrserker', 'Battle Armor', '', ['Iron Head', 'Protect', 'Seed Bomb']),
  mk('s0m3', 'Chimecho', 'Levitate', '', ['Dazzling Gleam', 'Protect', 'Recover']),
  mk('s0m4', 'Venusaur', 'Overgrow', '', ['Seed Bomb', 'Protect', 'Sludge Bomb']),
  mk('s0m5', 'Toxtricity-Low-Key', 'Technician', '', ['Thunderbolt', 'Protect', 'Drain Punch']),
];
const filler2 = [
  mk('s1m2', 'Delphox', 'Blaze', '', ['Flamethrower', 'Protect', 'Psychic']),
  mk('s1m3', 'Starmie', 'Natural Cure', '', ['Surf', 'Protect', 'Ice Beam']),
  mk('s1m4', 'Trevenant', 'Natural Cure', '', ['Shadow Claw', 'Protect', 'Horn Leech']),
  mk('s1m5', 'Gallade', 'Sharpness', '', ['Psycho Cut', 'Protect', 'Drain Punch']),
];

// Protocol damage follows the move line without naming the move, so scan from
// each move message until the next action boundary.
const moveHit = (log, moveName) => log.some((line, i) => {
  if (!line.startsWith('|move|') || !line.includes(`|${moveName}|`)) return false;
  for (let j = i + 1; j < log.length && !log[j].startsWith('|move|') && !log[j].startsWith('|turn|'); j++) {
    if (log[j].startsWith('|-damage|')) return true;
  }
  return false;
});

const cases = [
  {
    // Turn 1 flinches with Fake Out; Grassy Glide's terrain priority decides
    // whether Decidueye's Sucker Punch still finds a queued action.
    name: 'fakeout_suckerpunch_grassy',
    seed: [11, 22, 33, 44],
    p1: [
      mk('s0m0', 'Incineroar', 'Intimidate', '', ['Fake Out', 'Darkest Lariat', 'Protect', 'Flare Blitz']),
      mk('s0m1', 'Rillaboom', 'Grassy Surge', '', ['Grassy Glide', 'Drain Punch', 'Protect'],
        {hp: 10, atk: 15, def: 5, spa: 5, spd: 5, spe: 20}),
      ...filler(),
    ],
    p2: [
      mk('s1m0', 'Chesnaught', 'Overgrow', '', ['Bulk Up', 'Seed Bomb', 'Protect']),
      mk('s1m1', 'Decidueye', 'Overgrow', '', ['Sucker Punch', 'Shadow Ball', 'Protect'],
        {hp: 10, atk: 15, def: 5, spa: 5, spd: 5, spe: 0}),
      ...filler2,
    ],
    script: [
      ['p1', 'team 1234'], ['p2', 'team 1234'],
      ['p1', 'move Fake Out 1, move Grassy Glide 1'], ['p2:Chesnaught', 'move Seed Bomb 1, move Sucker Punch 2'],
      ['p1', 'move Darkest Lariat 1, move Grassy Glide 1'], ['p2:Chesnaught', 'move Bulk Up, move Shadow Ball 1'],
      ['p1', 'move Darkest Lariat 1, move Grassy Glide 1'], ['p2:Chesnaught', 'move Seed Bomb 1, move Sucker Punch 2'],
      ['p1', 'switch Perrserker, move Grassy Glide 1'], ['p2:Chesnaught', 'move Seed Bomb 1, move Shadow Ball 1'],
      ['p1', 'switch Incineroar, move Grassy Glide 2'], ['p2:Chesnaught', 'move Seed Bomb 1, move Sucker Punch 1'],
      ['p1', 'move Fake Out 1, move Grassy Glide 1'], ['p2:Chesnaught', 'move Seed Bomb 1, move Sucker Punch 2'],
    ],
    coverage(fixture, session) {
      const log = session.battle.log;
      const flinches = log.filter(x => x.startsWith('|cant|') && x.endsWith('|flinch')).length;
      if (flinches < 1) throw new Error('Fake Out never produced a flinch');
      const suckerFails = log.some((line, i) => {
        if (!line.startsWith('|move|') || !line.includes('|Sucker Punch|')) return false;
        const actor = line.split('|')[2];
        for (let j = i + 1; j < log.length && !log[j].startsWith('|move|') && !log[j].startsWith('|turn|'); j++) {
          if (log[j].startsWith('|-fail|') && log[j].split('|')[2] === actor) return true;
        }
        return false;
      });
      if (!suckerFails) throw new Error('Sucker Punch never failed against an already-acted target');
      const lead = step => step.expected.sides[0].pokemon.find(p => p.roster === 0);
      const disabled = fixture.steps.some(step => lead(step).disabled[0] === true);
      if (!disabled) throw new Error('Fake Out was never disabled after its first use');
      const reenabled = fixture.steps.some((step, i) => {
        const mon = lead(step);
        return mon.active_slot === 0 && mon.disabled[0] === false
          && fixture.steps.slice(0, i).some(p => lead(p).disabled[0] === true);
      });
      if (!reenabled) throw new Error('Fake Out was never re-enabled after re-entry');
      if (!moveHit(log, 'Darkest Lariat')) throw new Error('Darkest Lariat never dealt damage');
    },
  },
  {
    // Snow Warning makes Blizzard always hit; Freeze-Dry is super effective on
    // the Water/Fairy target where a plain Ice move is resisted.
    name: 'snow_blizzard_freezedry',
    seed: [5, 6, 7, 8],
    p1: [
      mk('s0m0', 'Aurorus', 'Snow Warning', '', ['Blizzard', 'Freeze-Dry', 'Protect']),
      mk('s0m1', 'Blastoise', 'Torrent', '', ['Body Press', 'Ice Beam', 'Protect']),
      ...filler(),
    ],
    p2: [
      mk('s1m0', 'Primarina', 'Torrent', '', ['Psychic', 'Draining Kiss', 'Protect']),
      mk('s1m1', 'Aggron', 'Rock Head', '', ['Iron Head', 'Protect', 'Earthquake']),
      ...filler2,
    ],
    script: [
      ['p1', 'team 1234'], ['p2', 'team 1234'],
      ['p1:Aurorus', 'move Freeze-Dry 1, move Protect'], ['p2', 'move Psychic 1, move Protect'],
      ['p1:Aurorus', 'move Blizzard, move Body Press 1'], ['p2:Primarina', 'move Psychic 1, move Iron Head 1'],
    ],
    coverage(fixture, session) {
      const log = session.battle.log;
      if (!log.some(x => x.startsWith('|move|') && x.includes('|Blizzard|'))) throw new Error('Blizzard never used');
      if (log.some(x => x.startsWith('|-miss|') && x.includes('Blizzard'))) throw new Error('Blizzard missed in snow');
      if (!log.some(x => x.startsWith('|-supereffective|'))) throw new Error('Freeze-Dry was never super effective');
      if (!moveHit(log, 'Blizzard')) throw new Error('Blizzard never dealt damage');
      if (!moveHit(log, 'Freeze-Dry')) throw new Error('Freeze-Dry never dealt damage');
    },
  },
  {
    // Rain from Drizzle makes Thunder always hit while keeping its exact
    // secondary paralysis roll.
    name: 'rain_thunder_accuracy',
    seed: [9, 10, 11, 12],
    p1: [
      mk('s0m0', 'Pelipper', 'Drizzle', '', ['Weather Ball', 'Protect', 'Air Slash']),
      mk('s0m1', 'Ampharos', 'Static', '', ['Thunder', 'Dragon Pulse', 'Protect']),
      ...filler(),
    ],
    p2: [
      mk('s1m0', 'Primarina', 'Torrent', '', ['Psychic', 'Draining Kiss', 'Protect']),
      mk('s1m1', 'Decidueye', 'Overgrow', '', ['Sucker Punch', 'Shadow Ball', 'Protect']),
      ...filler2,
    ],
    script: [
      ['p1', 'team 1234'], ['p2', 'team 1234'],
      ['p1:Ampharos', 'move Weather Ball 1, move Thunder 1'],
      ['p1:Ampharos', 'move Protect, move Thunder 1'],
    ],
    coverage(fixture, session) {
      const log = session.battle.log;
      if (!log.some(x => x.startsWith('|move|') && x.includes('|Thunder|'))) throw new Error('Thunder never used');
      if (log.some(x => x.startsWith('|-miss|') && x.includes('Thunder'))) throw new Error('Thunder missed in rain');
      if (!moveHit(log, 'Thunder')) throw new Error('Thunder never dealt damage');
    },
  },
  {
    // Body Press uses the user's Defense stat and its stages as its Attack.
    name: 'bodypress_uses_defense',
    seed: [21, 22, 23, 24],
    p1: [
      mk('s0m0', 'Blastoise', 'Torrent', '', ['Body Press', 'Iron Defense', 'Protect', 'Ice Beam']),
      mk('s0m1', 'Snorlax', 'Thick Fat', '', ['Body Slam', 'Protect', 'Earthquake']),
      ...filler(),
    ],
    p2: [
      mk('s1m0', 'Aggron', 'Rock Head', '', ['Iron Head', 'Protect', 'Earthquake']),
      mk('s1m1', 'Primarina', 'Torrent', '', ['Psychic', 'Draining Kiss', 'Protect']),
      ...filler2,
    ],
    script: [
      ['p1', 'team 1234'], ['p2', 'team 1234'],
      ['p1:Blastoise', 'move Body Press 1, move Body Slam 1'], ['p2', 'move Iron Head 1, move Psychic 1'],
      ['p1:Blastoise', 'move Iron Defense, move Body Slam 1'], ['p2', 'move Iron Head 1, move Psychic 1'],
      ['p1:Blastoise', 'move Body Press 1, move Body Slam 1'], ['p2', 'move Iron Head 1, move Psychic 1'],
    ],
    coverage(fixture, session) {
      const log = session.battle.log;
      if (!moveHit(log, 'Body Press')) throw new Error('Body Press never hit');
      const boosted = fixture.steps.some(step =>
        step.expected.sides[0].pokemon.find(p => p.roster === 0).boosts[1] > 0);
      if (!boosted) throw new Error('Iron Defense never raised Defense');
    },
  },
  {
    // Psyshock uses the target's Defense; Foul Play uses the target's Attack.
    name: 'psyshock_foulplay_overrides',
    seed: [31, 32, 33, 34],
    p1: [
      mk('s0m0', 'Alakazam', 'Synchronize', '', ['Psyshock', 'Psychic', 'Protect']),
      mk('s0m1', 'Farigiraf', 'Sap Sipper', '', ['Foul Play', 'Psychic', 'Protect']),
      ...filler(),
    ],
    p2: [
      mk('s1m0', 'Kingambit', 'Defiant', '', ['Kowtow Cleave', 'Sucker Punch', 'Protect']),
      mk('s1m1', 'Goodra-Hisui', 'Shell Armor', '', ['Dragon Pulse', 'Ice Beam', 'Protect']),
      ...filler2,
    ],
    script: [
      ['p1', 'team 1234'], ['p2', 'team 1234'],
      ['p1:Alakazam', 'move Psyshock 2, move Foul Play 1'], ['p2:Kingambit', 'move Protect, move Dragon Pulse 2'],
      ['p1:Alakazam', 'move Psyshock 2, move Foul Play 1'], ['p2:Kingambit', 'move Kowtow Cleave 1, move Dragon Pulse 2'],
    ],
    coverage(fixture, session) {
      const log = session.battle.log;
      if (!moveHit(log, 'Psyshock')) throw new Error('Psyshock never hit');
      if (!moveHit(log, 'Foul Play')) throw new Error('Foul Play never hit');
    },
  },
  {
    // Clanging Scales lowers the user's Defense after it connects.
    name: 'clangingscales_selfboost',
    seed: [41, 42, 43, 44],
    p1: [
      mk('s0m0', 'Kommo-o', 'Overcoat', '', ['Clanging Scales', 'Drain Punch', 'Protect']),
      mk('s0m1', 'Snorlax', 'Thick Fat', '', ['Body Slam', 'Protect', 'Earthquake']),
      ...filler(),
    ],
    p2: [
      mk('s1m0', 'Primarina', 'Torrent', '', ['Psychic', 'Draining Kiss', 'Protect']),
      mk('s1m1', 'Aggron', 'Rock Head', '', ['Iron Head', 'Protect', 'Earthquake']),
      ...filler2,
    ],
    script: [
      ['p1', 'team 1234'], ['p2', 'team 1234'],
      ['p1:Kommo-o', 'move Clanging Scales, move Body Slam 1'], ['p2:Primarina', 'move Protect, move Protect'],
      ['p1:Kommo-o', 'move Clanging Scales, move Body Slam 1'], ['p2:Primarina', 'move Protect, move Earthquake'],
    ],
    coverage(fixture, session) {
      const log = session.battle.log;
      if (!moveHit(log, 'Clanging Scales')) throw new Error('Clanging Scales never hit');
      const lowered = fixture.steps.some(step => {
        const mon = step.expected.sides[0].pokemon.find(p => p.roster === 0);
        return mon && mon.boosts[1] < 0 && mon.active_slot !== null && mon.hp > 0;
      });
      if (!lowered) throw new Error('Clanging Scales never lowered the user Defense');
    },
  },
  {
    // Feint removes protection after the accuracy step and then damages.
    name: 'feint_breaks_protect',
    seed: [51, 52, 53, 54],
    p1: [
      mk('s0m0', 'Goodra-Hisui', 'Shell Armor', '', ['Feint', 'Dragon Pulse', 'Protect']),
      mk('s0m1', 'Snorlax', 'Thick Fat', '', ['Body Slam', 'Protect', 'Earthquake']),
      ...filler(),
    ],
    p2: [
      mk('s1m0', 'Primarina', 'Torrent', '', ['Psychic', 'Draining Kiss', 'Protect']),
      mk('s1m1', 'Torterra', 'Shell Armor', '', ['Seed Bomb', 'Protect', 'Earthquake']),
      ...filler2,
    ],
    script: [
      ['p1', 'team 1234'], ['p2', 'team 1234'],
      ['p1:Goodra-Hisui', 'move Feint 1, move Body Slam 1'], ['p2:Primarina', 'move Protect, move Protect'],
      ['p1:Goodra-Hisui', 'move Feint 1, move Body Slam 1'], ['p2:Primarina', 'move Psychic 1, move Seed Bomb 1'],
    ],
    coverage(fixture, session) {
      const log = session.battle.log;
      if (!log.some(x => x.startsWith('|-activate|') && x.includes('move: Feint'))) {
        throw new Error('Feint never broke protection');
      }
      if (!moveHit(log, 'Feint')) throw new Error('Feint never damaged');
    },
  },
];

verifyReference();
const fixtures = [];

const select = (kind, own_slot, destination = 255) => ({
  kind, own_slot, move_slot: 255, target_location: 0, switch_destination: destination, resource: 'None',
});

const parseChoice = command => command.split(',').map((part, slot) => {
  const [kind, a, b] = part.trim().split(/\s+/);
  if (kind === 'pass') return select('Pass', slot);
  if (kind === 'switch') return select('Switch', slot, Number(a) - 1);
  const action = select('Move', slot);
  action.move_slot = Number(a) - 1;
  action.target_location = b ? Number(b) : 0;
  return action;
});

// Fallback so every fixture completes a natural battle.
const autoChoice = (battle, sideIndex) => {
  const side = battle.sides[sideIndex];
  const actions = [], commands = [];
  const chosen = new Set();
  for (let slot = 0; slot < 2; slot++) {
    if (side.requestState === 'switch') {
      // Only slots that must replace contribute an action; the other slot's
      // `pass` is protocol-only, matching the native joint-action contract.
      if (!side.activeRequest.forceSwitch[slot]) { commands.push('pass'); continue; }
      const reserve = side.pokemon.find(p => !p.fainted && !side.active.includes(p) && !chosen.has(p));
      if (!reserve) { actions.push(select('Pass', slot)); commands.push('pass'); continue; }
      chosen.add(reserve);
      actions.push(select('Switch', slot, roster(reserve)));
      commands.push(`switch ${side.pokemon.indexOf(reserve) + 1}`);
      continue;
    }
    const p = side.active[slot];
    if (!p || p.fainted) { actions.push(select('Pass', slot)); commands.push('pass'); continue; }
    let use = p.moveSlots.findIndex(m => m.pp > 0 && !m.disabled && m.target === 'normal');
    if (use < 0) use = p.moveSlots.findIndex(m => m.pp > 0 && !m.disabled);
    if (use < 0) use = 0;
    const move = p.moveSlots[use];
    let target = 0;
    if (move.target === 'normal' || move.target === 'any') {
      const foe = battle.sides[1 - sideIndex];
      target = foe.active[0] && !foe.active[0].fainted ? 1 : 2;
    }
    const action = select('Move', slot);
    action.move_slot = use;
    action.target_location = target;
    actions.push(action);
    commands.push(`move ${use + 1}${target ? ` ${target}` : ''}`);
  }
  return {actions, command: commands.join(', ')};
};

const capture = session => {
  const b = session.battle;
  const request = side => {
    if (b.ended) return {kind: 'Finished', moves: []};
    if (!side.activeRequest || side.activeRequest.wait || side.isChoiceDone()) return {kind: 'Wait', moves: []};
    if (side.requestState === 'teampreview') return {kind: 'Preview', moves: []};
    if (side.requestState === 'switch') return {kind: 'Replacement', moves: []};
    return {
      kind: 'Normal',
      moves: (side.activeRequest.pokemon ?? []).map(p =>
        (p?.moves ?? []).map(m => [ids.moves[toID(m.id)] ?? 0, m.pp ?? 0, Boolean(m.disabled)])),
    };
  };
  return {
    turn: b.turn,
    rng: b.prng.getSeed(),
    ended: b.ended,
    winner: b.ended ? b.winner || null : null,
    sides: b.sides.map(side => ({
      request: request(side),
      pokemon: side.pokemon.map(p => ({
        roster: roster(p),
        species: ids.species[p.species.id],
        hp: p.hp,
        max_hp: p.maxhp,
        fainted: p.fainted,
        active_slot: side.active.indexOf(p) >= 0 ? side.active.indexOf(p) : null,
        status: p.status && p.status !== 'fnt' ? (ids.conditions[p.status] ?? 0) : 0,
        boosts: [p.boosts.atk, p.boosts.def, p.boosts.spa, p.boosts.spd, p.boosts.spe,
          p.boosts.accuracy, p.boosts.evasion],
        stats: [p.maxhp, p.storedStats.atk, p.storedStats.def, p.storedStats.spa, p.storedStats.spd, p.storedStats.spe],
        ability: ids.abilities[p.ability] ?? 0,
        ability_ending: Boolean(p.abilityState?.ending),
        item: p.item ? ids.items[p.item] : 0,
        types: p.types.map(t => ids.types[toID(t)] ?? 0),
        pp: p.moveSlots.map(m => m.pp),
        disabled: p.moveSlots.map(m => Boolean(m.disabled)),
        volatiles: Object.keys(p.volatiles).map(id => ids.conditions[id]).sort((a, b) => a - b),
      })),
    })),
  };
};

for (const testCase of cases) {
  const session = new ReferenceSession({teams: [testCase.p1, testCase.p2], seed: testCase.seed});
  const fixture = {
    name: testCase.name,
    seed: testCase.seed,
    teams: session.teams.map((team, side) => ({
      id: `move-hook-${testCase.name}-${side}`,
      members: team.map(s => ({
        species: ids.species[toID(s.species)],
        ability: ids.abilities[toID(s.ability)],
        item: ids.items[toID(s.item)] ?? 0,
        nature: ids.natures[toID(s.nature)],
        gender: s.gender || '',
        level: 50,
        moves: s.moves.map(m => ids.moves[toID(m)]),
        points: stats.map(k => s.evs[k]),
        ivs: stats.map(k => s.ivs[k]),
      })),
    })),
    initial: capture(session),
    steps: [],
  };
  let cursor = 0;
  while (!session.battle.ended && fixture.steps.length < 400) {
    for (let side = 0; side < 2; side++) {
      if (session.battle.ended) break;
      const s = session.battle.sides[side];
      if (s.activeRequest?.wait || s.isChoiceDone()) continue;
      const scripted = testCase.script[cursor];
      let choice;
      const expectsSwitch = /^switch /.test(scripted?.[1] ?? '');
      // Script entries are `p1`/`p2`, optionally qualified with the species that
      // must occupy slot 0 (`p1:Aurorus`) so scripts survive replacements.
      const [scriptSide, scriptMon] = (scripted?.[0] ?? '').split(':');
      const activeMons = session.battle.sides[side].active.filter(Boolean);
      const monMatches = !scriptMon || activeMons.some(m => m.species.name === scriptMon
        || m.baseSpecies?.name === scriptMon);
      const parts = (scripted?.[1] ?? '').split(',').map(p => p.trim().split(/\s+/)[0]);
      const allSwitchy = parts.length > 0 && parts.every(kind => kind === 'switch' || kind === 'pass');
      const sideMatches = scriptSide === (side ? 'p2' : 'p1');
      const kindMatches = s.requestState === 'switch' ? allSwitchy : true;
      const applicable = Boolean(scripted) && sideMatches && monMatches && kindMatches;
      if (applicable) {
        cursor++;
        // Switch entries accept a reference position (`switch 3`) or a name
        // (`switch Goodra-Hisui`); the native destination is always the stable
        // roster index of the incoming Pokémon.
        const reference = session.battle.sides[side];
        const actions = [], tokens = [];
        if (scripted[1].startsWith('team ')) {
          const order = scripted[1].slice(5).trim();
          order.split('').forEach((digit, slot) => {
            actions.push(select('Pick', slot, roster(reference.pokemon[Number(digit) - 1])));
          });
          choice = {actions, command: `team ${order}`};
        } else {
        scripted[1].split(',').forEach((part, slot) => {
          const words = part.trim().split(/\s+/);
          const kind = words[0];
          if (kind === 'pass') {
            if (s.requestState !== 'switch') actions.push(select('Pass', slot));
            tokens.push('pass');
            return;
          }
          if (kind === 'switch') {
            const target = words.slice(1).join(' ');
            const numeric = Number(target);
            const position = Number.isInteger(numeric) && /^\d+$/.test(target) ? numeric - 1
              : reference.pokemon.findIndex(p => p.species.id === toID(target) || p.name.toLowerCase() === target.toLowerCase());
            if (position < 0) throw new Error(`${testCase.name}: no switch target ${target}`);
            actions.push(select('Switch', slot, roster(reference.pokemon[position])));
            tokens.push(`switch ${position + 1}`);
            return;
          }
          // `move <Name|Slot> [target]` resolves the move against the
          // currently active Pokemon so scripts survive replacements.
          const rest = words.slice(1);
          let target = 0;
          let name = rest.join(' ');
          const last = Number(rest.at(-1));
          if (rest.length > 1 && Number.isInteger(last)) {
            target = last;
            name = rest.slice(0, -1).join(' ');
          }
          const mon = reference.active[slot];
          let resolved = mon ? mon.moveSlots.findIndex(m => m.id === toID(name)) : -1;
          if (resolved < 0 && /^\d+$/.test(name)) resolved = Number(name) - 1;
          if (resolved < 0 || !mon?.moveSlots?.[resolved]) {
            throw new Error(`${testCase.name}: ${mon?.name} has no move ${name}`);
          }
          const action = select('Move', slot);
          action.move_slot = resolved;
          action.target_location = target;
          actions.push(action);
          tokens.push(`move ${resolved + 1}${target ? ` ${target}` : ''}`);
        });
        choice = {actions, command: tokens.join(', ')};
        }
      } else {
        choice = autoChoice(session.battle, side);
      }
      const result = session.choose(side ? 'p2' : 'p1', choice.command);
      if (!result.accepted) {
        throw new Error(JSON.stringify({name: testCase.name, side, choice, messages: result.messages}));
      }
      fixture.steps.push({side: side ? 'P2' : 'P1', ...choice, expected: capture(session)});
    }
  }
  if (cursor !== testCase.script.length) {
    // A scripted Pokémon can faint before its entry comes up; the coverage
    // checks below still require the intended mechanic to have executed.
    console.error(`${testCase.name}: ${testCase.script.length - cursor} scripted choices were not reached`);
  }
  if (!session.battle.ended) throw new Error(`${testCase.name} did not naturally complete`);
  testCase.coverage(fixture, session);
  fixtures.push(fixture);
  session.destroy();
}

fs.writeFileSync(new URL('move-hook-fixtures.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify(fixtures.map(f => ({name: f.name, decisions: f.steps.length}))));
