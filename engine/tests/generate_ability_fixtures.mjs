// Development-only generator for the contact-triggered ability family
// (Rough Skin, Stamina, Flame Body, Poison Touch).
//
// Synthetic legal teams are mechanics fixtures only, never training-pool
// additions. Every fixture completes naturally and records the pinned
// reference's decision-boundary state (including RNG and request legality) for
// the Rust differential test in `ability_contact.rs`.
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

const cases = [
  {
    // Turn 1 exercises all four contact abilities at once: Sharpedo's Aqua Jet
    // into Flame Body, Sneasler's Close Combat into Stamina plus Poison Touch,
    // and Archaludon's Iron Head into Rough Skin.
    name: 'contact_ability_family_turn_one',
    // Deterministic candidate seeds; the first one whose RNG stream makes all
    // four 3/10 rolls and the contact hits actually happen is recorded.
    seeds: Array.from({length: 256}, (_, k) =>
      [(k * 37 + 1) & 0xffff, (k * 91 + 7) & 0xffff, (k * 13 + 3) & 0xffff, (k * 101 + 11) & 0xffff]),
    p1: [
      mk('s0m0', 'Volcarona', 'Flame Body', '', ['Heat Wave', 'Protect', 'Flamethrower'],
        {hp: 32, atk: 0, def: 17, spa: 0, spd: 17, spe: 0}),
      mk('s0m1', 'Mudsdale', 'Stamina', '', ['Protect', 'Body Press', 'Earthquake'],
        {hp: 32, atk: 0, def: 17, spa: 0, spd: 17, spe: 0}),
      mk('s0m2', 'Sharpedo', 'Rough Skin', '', ['Aqua Jet', 'Protect', 'Crunch'],
        {hp: 32, atk: 0, def: 17, spa: 0, spd: 17, spe: 0}),
      mk('s0m3', 'Chimecho', 'Levitate', '', ['Dazzling Gleam', 'Protect', 'Recover']),
      mk('s0m4', 'Venusaur', 'Overgrow', '', ['Seed Bomb', 'Protect', 'Sludge Bomb']),
      mk('s0m5', 'Toxtricity-Low-Key', 'Technician', '', ['Thunderbolt', 'Protect', 'Drain Punch']),
    ],
    p2: [
      mk('s1m0', 'Snorlax', 'Thick Fat', '', ['Body Slam', 'Protect', 'Earthquake'],
        {hp: 32, atk: 0, def: 17, spa: 0, spd: 17, spe: 0}),
      mk('s1m1', 'Sneasler', 'Poison Touch', '', ['Close Combat', 'Protect', 'Shadow Claw'],
        {hp: 32, atk: 0, def: 17, spa: 0, spd: 17, spe: 0}),
      mk('s1m2', 'Delphox', 'Blaze', '', ['Flamethrower', 'Protect', 'Psychic']),
      mk('s1m3', 'Starmie', 'Natural Cure', '', ['Surf', 'Protect', 'Ice Beam']),
      mk('s1m4', 'Trevenant', 'Natural Cure', '', ['Shadow Claw', 'Protect', 'Horn Leech']),
      mk('s1m5', 'Gallade', 'Sharpness', '', ['Psycho Cut', 'Protect', 'Drain Punch']),
    ],
    script: [
      ['p1', 'team 1234'], ['p2', 'team 1234'],
      ['p1', 'move Protect, move Protect'],
      ['p2', 'move Body Slam 1, move Close Combat 2'],
      ['p1', 'move Protect, switch 3'],
      ['p2', 'move Body Slam 1, move Close Combat 2'],
    ],
    coverage(fixture, session) {
      const log = session.battle.log;
      if (!log.some(line => line.startsWith('|-damage|') && line.includes('[from] ability: Rough Skin'))) {
        throw new Error('Rough Skin never damaged the attacker');
      }
      if (!log.some(line => line.startsWith('|-status|') && line.includes('ability: Flame Body'))) {
        throw new Error('Flame Body never burned the attacker');
      }
      if (!log.some(line => line.startsWith('|-status|') && line.includes('ability: Poison Touch'))) {
        throw new Error('Poison Touch never poisoned the target');
      }
      if (!log.some(line => line.startsWith('|-ability|') && line.includes('Stamina|boost'))) {
        throw new Error('Stamina never raised Defense');
      }
      const burnt = fixture.steps.some(step => {
        const mon = step.expected.sides[1].pokemon.find(p => p.roster === 0);
        return mon && mon.status === ids.conditions.brn;
      });
      if (!burnt) throw new Error('Flame Body burn never reached a decision boundary');
      const poisoned = fixture.steps.some(step => {
        const mon = step.expected.sides[0].pokemon.find(p => p.roster === 1);
        return mon && mon.status === ids.conditions.psn;
      });
      if (!poisoned) throw new Error('Poison Touch poison never reached a decision boundary');
      const raised = fixture.steps.some(step => {
        const mon = step.expected.sides[0].pokemon.find(p => p.roster === 1);
        return mon && mon.boosts[1] > 0;
      });
      if (!raised) throw new Error('Stamina boost never reached a decision boundary');
    },
  },
  {
    // Prankster: a slower Whimsicott still applies Charm before Excadrill's
    // Iron Head, so the recorded damage is taken at -2 Attack, and the same
    // status move is naturally immune against the Dark-type Kingambit.
    name: 'prankster_priority_and_dark_immunity',
    seeds: Array.from({length: 64}, (_, k) =>
      [(k * 53 + 5) & 0xffff, (k * 29 + 13) & 0xffff, (k * 71 + 17) & 0xffff, (k * 97 + 19) & 0xffff]),
    p1: [
      mk('s0m0', 'Whimsicott', 'Prankster', '', ['Charm', 'Sunny Day', 'Protect', 'Dazzling Gleam'],
        {hp: 32, atk: 0, def: 17, spa: 0, spd: 17, spe: 0}),
      mk('s0m1', 'Chimecho', 'Levitate', '', ['Dazzling Gleam', 'Protect', 'Recover'],
        {hp: 32, atk: 0, def: 17, spa: 0, spd: 17, spe: 0}),
      mk('s0m2', 'Venusaur', 'Overgrow', '', ['Seed Bomb', 'Protect', 'Sludge Bomb']),
      mk('s0m3', 'Toxtricity-Low-Key', 'Technician', '', ['Thunderbolt', 'Protect', 'Drain Punch']),
      mk('s0m4', 'Mudsdale', 'Stamina', '', ['Protect', 'Body Press', 'Earthquake']),
      mk('s0m5', 'Vaporeon', 'Water Absorb', '', ['Surf', 'Protect', 'Ice Beam']),
    ],
    p2: [
      mk('s1m0', 'Excadrill', 'Sand Rush', '', ['Drill Run', 'Iron Head', 'Protect'],
        {hp: 32, atk: 0, def: 0, spa: 0, spd: 2, spe: 32}),
      mk('s1m1', 'Kingambit', 'Defiant', '', ['Kowtow Cleave', 'Protect', 'Iron Head'],
        {hp: 32, atk: 0, def: 17, spa: 0, spd: 17, spe: 0}),
      mk('s1m2', 'Delphox', 'Blaze', '', ['Flamethrower', 'Protect', 'Psychic']),
      mk('s1m3', 'Starmie', 'Natural Cure', '', ['Surf', 'Protect', 'Ice Beam']),
      mk('s1m4', 'Trevenant', 'Natural Cure', '', ['Shadow Claw', 'Protect', 'Horn Leech']),
      mk('s1m5', 'Gallade', 'Sharpness', '', ['Psycho Cut', 'Protect', 'Drain Punch']),
    ],
    script: [
      ['p1', 'team 1234'], ['p2', 'team 1234'],
      ['p1', 'move Charm 1, move Protect'],
      ['p2', 'move Drill Run 1, move Protect'],
      ['p1', 'move Charm 2, move Protect'],
      ['p2', 'move Drill Run 1, move Protect'],
    ],
    coverage(fixture, session) {
      const log = session.battle.log;
      if (!log.some(line => line.startsWith('|-unboost|') && line.includes('s1m0')
        && line.includes('|atk|'))) {
        throw new Error('Charm never lowered Excadrill Attack');
      }
      const lowered = fixture.steps.some(step => {
        const mon = step.expected.sides[1].pokemon.find(p => p.roster === 0);
        return mon && mon.boosts[0] === -2;
      });
      if (!lowered) throw new Error('Charm Attack drop never reached a decision boundary');
      // The pinned reference reports the natural immunity and the gen-7 hint;
      // the `-immune` line itself carries no `[from]` tag for this gate.
      if (!log.some(line => line.startsWith('|-immune|') && line.includes('s1m1'))) {
        throw new Error('Prankster never failed against the Dark-type Kingambit');
      }
      const kingambitUntouched = fixture.steps.every(step => {
        const mon = step.expected.sides[1].pokemon.find(p => p.roster === 1);
        return mon && mon.boosts[0] === 0;
      });
      if (!kingambitUntouched) throw new Error('Dark-type Kingambit was affected by the Prankster move');
    },
  },
];

verifyReference();
const fixtures = [];

const select = (kind, own_slot, destination = 255) => ({
  kind, own_slot, move_slot: 255, target_location: 0, switch_destination: destination, resource: 'None',
});

const autoChoice = (battle, sideIndex) => {
  const side = battle.sides[sideIndex];
  const actions = [], commands = [];
  const chosen = new Set();
  for (let slot = 0; slot < 2; slot++) {
    if (side.requestState === 'switch') {
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
        types: p.types.map(t => ids.types[toID(t)]),
        pp: p.moveSlots.map(m => m.pp),
        disabled: p.moveSlots.map(m => Boolean(m.disabled)),
        volatiles: Object.keys(p.volatiles).map(id => ids.conditions[id]).sort((a, b) => a - b),
      })),
    })),
  };
};

// The 3/10 contact rolls depend on the exact draw order, so search the
// candidate seeds until the reference actually exercises every ability.
const buildCase = (testCase, seed) => {
  const session = new ReferenceSession({teams: [testCase.p1, testCase.p2], seed});
  const fixture = {
    name: testCase.name,
    seed,
    teams: session.teams.map((team, side) => ({
      id: `ability-${testCase.name}-${side}`,
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
      const [scriptSide, scriptMon] = (scripted?.[0] ?? '').split(':');
      const activeMons = session.battle.sides[side].active.filter(Boolean);
      const monMatches = !scriptMon || activeMons.some(m => m.species.name === scriptMon
        || m.baseSpecies?.name === scriptMon);
      const parts = (scripted?.[1] ?? '').split(',').map(p => p.trim().split(/\s+/)[0]);
      const allSwitchy = parts.length > 0 && parts.every(kind => kind === 'switch' || kind === 'pass');
      const sideMatches = scriptSide === (side ? 'p2' : 'p1');
      const kindMatches = s.requestState === 'switch' ? allSwitchy : true;
      const applicable = Boolean(scripted) && sideMatches && monMatches && kindMatches;
      let choice;
      if (applicable) {
        cursor++;
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
                : reference.pokemon.findIndex(p => p.species.id === toID(target)
                  || p.name.toLowerCase() === target.toLowerCase());
              if (position < 0) throw new Error(`${testCase.name}: no switch target ${target}`);
              actions.push(select('Switch', slot, roster(reference.pokemon[position])));
              tokens.push(`switch ${position + 1}`);
              return;
            }
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
  if (!session.battle.ended) throw new Error(`${testCase.name} did not naturally complete`);
  return {session, fixture};
};

for (const testCase of cases) {
  let built = null;
  let lastError = null;
  for (const seed of testCase.seeds) {
    const attempt = buildCase(testCase, seed);
    try {
      testCase.coverage(attempt.fixture, attempt.session);
      built = attempt;
      break;
    } catch (error) {
      lastError = error;
      attempt.session.destroy();
    }
  }
  if (!built) throw lastError ?? new Error(`${testCase.name}: no seed exercised the mechanics`);
  fixtures.push(built.fixture);
  built.session.destroy();
}

fs.writeFileSync(new URL('ability-fixtures.json', import.meta.url),
  JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify(fixtures.map(f => ({name: f.name, decisions: f.steps.length}))));
