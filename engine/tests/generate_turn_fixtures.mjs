// Synthetic legal teams are mechanics fixtures only, never training-pool additions.
import fs from 'node:fs';
import {createRequire} from 'node:module';
import {ReferenceSession, verifyReference, FORMAT, ORACLE_COMMIT} from '../reference.mjs';
const require = createRequire(import.meta.url);
const {TeamValidator, toID} = require('../../vendor/pokemon-showdown/dist/sim');
const dex = new TeamValidator(FORMAT).dex;
const data = JSON.parse(fs.readFileSync(new URL('../data/dex.json', import.meta.url), 'utf8'));
const ids = Object.fromEntries(Object.entries(data.tables).map(([k, rows]) => [k, Object.fromEntries(rows.map(r => [r.id, r.numeric_id]))]));
const stats = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const species = ['Goodra-Hisui', 'Torterra', 'Falinks', 'Perrserker', 'Samurott', 'Hydreigon'];
const abilities = ['Shell Armor', 'Shell Armor', 'Battle Armor', 'Battle Armor', 'Shell Armor', 'Levitate'];
const moves = [['Dragon Pulse', 'Dragon Claw', 'Surf', 'Protect'], ['Seed Bomb', 'Earthquake', 'High Horsepower', 'Protect'],
  ['Smart Strike', 'Seed Bomb', 'High Horsepower', 'Protect'], ['X-Scissor', 'Aerial Ace', 'Night Slash', 'Protect'],
  ['Aqua Jet', 'Hydro Pump', 'Surf', 'Protect'], ['Dragon Pulse', 'Hyper Voice', 'Hydro Pump', 'Protect']];
const teamsFor = (trial, items) => [0, 1].map(side => species.map((name, i) => {
  const set = {name: `s${side}m${i}`, species: name,
    ability: abilities[i], item: '', nature: 'Serious', level: 50, gender: 'M', moves: trial === 6 ? ['Protect'] : moves[i],
    evs: {hp: 10, atk: 15, def: 5, spa: 15, spd: 5, spe: trial === 4 ? 0 : i * 2 + side}};
  if (trial >= 7 && trial <= 12) {
    if (i === 0) Object.assign(set, {species: 'Chimecho', ability: 'Levitate', item: 'Chimechite', moves: ['Dazzling Gleam', 'Hyper Voice', 'Protect']});
    if (i === 2) Object.assign(set, {species: 'Delphox', ability: 'Blaze', item: 'Delphoxite', moves: ['Dazzling Gleam', 'Hyper Voice', 'Protect']});
    if (i === 4) Object.assign(set, {species: 'Scolipede', ability: 'Swarm', item: 'Scolipite', moves: ['X-Scissor', 'Earthquake', 'Protect']});
  }
  if (trial === 10 || trial === 11) {
    set.moves = [
      [trial === 10 ? 'Thunder Wave' : 'Hypnosis', 'Dazzling Gleam', 'Recover', 'Protect'],
      ['Swords Dance', 'Seed Bomb', 'Earthquake', 'Protect'],
      [trial === 10 ? 'Will-O-Wisp' : 'Hypnosis', 'Dazzling Gleam', 'Calm Mind', 'Protect'],
      ['Iron Defense', 'X-Scissor', 'Night Slash', 'Protect'],
      ['Agility', 'X-Scissor', 'Earthquake', 'Protect'],
      ['Nasty Plot', 'Dragon Pulse', 'Heat Wave', 'Protect'],
    ][i];
  }
  if (trial === 12) set.moves = [
    ['Icy Wind', 'Dazzling Gleam', 'Recover', 'Protect'], moves[1],
    ['Heat Wave', 'Dazzling Gleam', 'Protect'], moves[3],
    ['Rock Slide', 'X-Scissor', 'Protect'], ['Snarl', 'Dragon Pulse', 'Protect'],
  ][i];
  if (trial === 13) set.moves = [
    ['Ice Beam', 'Flamethrower', 'Dragon Pulse', 'Protect'], ['Rock Slide', 'Seed Bomb', 'Protect'],
    ['Rock Slide', 'Close Combat', 'Protect'], ['Close Combat', 'X-Scissor', 'Protect'],
    ['Ice Beam', 'Aqua Jet', 'Protect'], ['Heat Wave', 'Dragon Pulse', 'Protect'],
  ][i];
  if (trial === 14) set.moves = [
    ['Draco Meteor', 'Flamethrower', 'Protect'], ['Leaf Storm', 'Seed Bomb', 'Protect'],
    ['Close Combat', 'Rock Slide', 'Protect'], ['Close Combat', 'X-Scissor', 'Protect'],
    ['Icy Wind', 'Aqua Jet', 'Protect'], ['Draco Meteor', 'Heat Wave', 'Protect'],
  ][i];
  if ((trial === 15 || trial === 16) && i === 0) Object.assign(set, {species: 'Venusaur', ability: 'Overgrow',
    moves: [trial === 15 ? 'Poison Powder' : 'Toxic', 'Seed Bomb', 'Leaf Storm', 'Protect']});
  const customize = (species, ability, moves, item = '') => Object.assign(set, {species, ability, moves, item});
  if ([17, 18, 25, 26].includes(trial)) {
    if (i === 0) customize('Salamence', 'Intimidate', ['Dragon Claw', 'Rock Slide', 'Protect']);
    if (i === 1) customize('Gyarados', 'Intimidate', ['Aqua Tail', 'Surf', 'Protect']);
    if (i === 2) customize(trial === 17 ? 'Falinks' : 'Metagross', trial === 17 ? 'Defiant' : 'Clear Body', trial === 17 ? ['Close Combat', 'Rock Slide', 'Protect'] : ['Iron Head', 'Bullet Punch', 'Protect']);
    if (i === 3) {
      if (trial === 17) customize('Milotic', 'Competitive', ['Surf', 'Ice Beam', 'Recover', 'Protect']);
      else if (trial === 18) customize('Dragonite', 'Inner Focus', ['Dragon Claw', 'Ice Punch', 'Protect']);
      else customize('Slowbro', trial === 25 ? 'Own Tempo' : 'Oblivious', ['Surf', 'Ice Beam', 'Protect']);
    }
  }
  if (trial === 19) {
    if (i === 0) customize('Mawile', 'Intimidate', ['Play Rough', 'Iron Head', 'Protect'], 'Mawilite');
    if (i === 2) customize('Scrafty', 'Intimidate', ['Close Combat', 'Crunch', 'Protect'], dex.species.get('Scrafty-Mega').requiredItem);
  }
  if (trial === 20) {
    if (i === 0) customize('Blaziken', 'Speed Boost', ['Close Combat', 'Fire Punch', 'Protect'], 'Blazikenite');
    if (i === 2) customize('Scolipede', 'Speed Boost', ['X-Scissor', 'Rock Slide', 'Protect']);
  }
  if (trial === 21) {
    if (i === 0) customize('Starmie', 'Natural Cure', ['Ice Beam', 'Surf', 'Recover', 'Protect']);
    if (i === 1) customize('Slowbro', 'Regenerator', ['Ice Beam', 'Surf', 'Slack Off', 'Protect']);
    if (i === 2) customize('Chimecho', 'Levitate', ['Thunder Wave', 'Dazzling Gleam', 'Protect']);
    if (i === 3) customize('Delphox', 'Blaze', ['Will-O-Wisp', 'Dazzling Gleam', 'Protect']);
  }
  if (trial === 22) {
    if (i === 0) customize('Scizor', 'Technician', ['Bullet Punch', 'X-Scissor', 'Protect'], 'Scizorite');
    if (i === 1) customize('Gallade', 'Sharpness', ['Psycho Cut', 'Night Slash', 'Protect']);
    if (i === 2) customize('Blastoise', 'Torrent', ['Dragon Pulse', 'Surf', 'Protect'], 'Blastoisinite');
  }
  if (trial === 23) {
    if (i === 0) customize('Medicham', 'Pure Power', ['Fire Punch', 'Close Combat', 'Protect'], 'Medichamite');
    if (i === 1) customize('Infernape', 'Iron Fist', ['Fire Punch', 'Thunder Punch', 'Protect']);
    if (i === 2) customize('Sharpedo', 'Speed Boost', ['Crunch', 'Ice Beam', 'Protect'], 'Sharpedonite');
  }
  if (trial === 24) {
    if (i === 0) customize('Venusaur', 'Overgrow', ['Leaf Storm', 'Seed Bomb', 'Protect'], 'Venusaurite');
    if (i === 1) customize('Rhyperior', 'Solid Rock', ['Rock Slide', 'Earthquake', 'Protect']);
    if (i === 2) customize('Metagross', 'Clear Body', ['Ice Punch', 'Iron Head', 'Protect'], 'Metagrossite');
    if (i === 4) customize('Dragonite', 'Multiscale', ['Fire Punch', 'Dragon Claw', 'Protect']);
  }
  if (trial === 27 && i === 0) customize('Medicham', 'Pure Power', ['Close Combat', 'Fire Punch', 'Protect']);
  if (trial === 28) {
    if (i === 0) { customize('Delphox', 'Blaze', ['Flamethrower', 'Heat Wave', 'Protect']); set.evs = {hp: 10, atk: 0, def: 5, spa: 25, spd: 0, spe: 0}; }
    if (i === 2) { customize('Venusaur', 'Overgrow', ['Leaf Storm', 'Seed Bomb', 'Protect'], 'Venusaurite'); set.evs = {hp: 10, atk: 0, def: 5, spa: 25, spd: 0, spe: 24}; }
  }
  if (trial === 29) {
    if (i === 0) customize('Arcanine-Hisui', 'Rock Head', ['Flare Blitz', 'Wild Charge', 'Protect'], 'Life Orb');
    if (i === 1) customize('Staraptor', 'Reckless', ['Brave Bird', 'Double-Edge', 'Protect']);
    if (i === 2) customize('Tyrantrum', 'Strong Jaw', ['Head Smash', 'Dragon Claw', 'Protect'], 'Rocky Helmet');
    if (i === 3) customize('Torterra', 'Shell Armor', ['Wood Hammer', 'Earthquake', 'Protect'], 'Sitrus Berry');
    if (i === 4) customize('Eelektross', 'Levitate', ['Wild Charge', 'Drain Punch', 'Protect']);
  }
  if (trial === 30) {
    if (i === 0) customize('Venusaur', 'Overgrow', ['Giga Drain', 'Seed Bomb', 'Protect'], 'Big Root');
    if (i === 1) customize('Trevenant', 'Natural Cure', ['Horn Leech', 'Shadow Claw', 'Protect']);
    if (i === 2) customize('Swalot', 'Liquid Ooze', ['Ice Beam', 'Protect']);
    if (i === 3) customize('Gallade', 'Sharpness', ['Drain Punch', 'Psycho Cut', 'Protect'], 'Rocky Helmet');
    if (i === 4) customize('Scolipede', 'Swarm', ['Leech Life', 'X-Scissor', 'Protect']);
    if (i === 5) customize('Chimecho', 'Levitate', ['Draining Kiss', 'Dazzling Gleam', 'Protect']);
  }
  if (trial === 31) {
    if (i === 0) customize('Toxtricity-Low-Key', 'Technician', ['Parabolic Charge', 'Drain Punch', 'Protect'], 'Big Root');
    if (i === 1) customize('Swalot', 'Liquid Ooze', ['Ice Beam', 'Protect']);
    if (i === 2) customize('Goodra-Hisui', 'Shell Armor', ['Ice Beam', 'Dragon Pulse', 'Protect']);
  }
  if (trial === 32) {
    if (i === 0) customize('Chimecho', 'Levitate', ['Draining Kiss', 'Dazzling Gleam', 'Protect'], 'Big Root');
    if (i === 1) customize('Infernape', 'Iron Fist', ['Drain Punch', 'Fire Punch', 'Protect'], 'Life Orb');
    if (i === 2) customize('Swalot', 'Liquid Ooze', ['Ice Beam', 'Protect'], 'Rocky Helmet');
  }
  if (trial === 33) {
    set.moves = ['Protect'];
    if (i === 0) customize('Tyrantrum', 'Rock Head', ['Protect']);
  }
  if (trial >= 34 && trial <= 42) {
    if (i === 0) customize('Chimecho', 'Levitate', ['Light Screen', 'Reflect', 'Dazzling Gleam', 'Protect'], trial === 36 && ((i === 0 && side === 0) || (i === 2 && side === 1)) ? 'Light Clay' : '');
    if (i === 1) customize('Dragonite', 'Inner Focus', ['Tailwind', 'Dragon Claw', 'Protect'], trial === 38 ? 'Choice Scarf' : '');
    if (i === 2) customize('Delphox', 'Blaze', ['Reflect', 'Light Screen', 'Dazzling Gleam', 'Protect'], trial === 36 && ((i === 0 && side === 0) || (i === 2 && side === 1)) ? 'Light Clay' : '');
    if (i === 3) customize(trial === 37 ? 'Gallade' : 'Salamence', trial === 37 ? 'Sharpness' : 'Intimidate', trial === 37 ? ['Agility', 'Psycho Cut', 'Protect'] : ['Tailwind', 'Dragon Claw', 'Protect']);
    if (i < 4) set.evs = {hp: 32, atk: 0, def: 17, spa: 0, spd: 17, spe: 0};
  }
  if (trial === 40 && i === 1) customize('Gallade', 'Sharpness', ['Agility', 'Brick Break', 'Protect']);
  if (trial === 41 && i === 3) customize('Sharpedo', 'Speed Boost', ['Agility', 'Psychic Fangs', 'Protect']);
  if (trial === 42 && i === 1) customize('Noivern', 'Infiltrator', ['Tailwind', 'Dragon Pulse', 'Protect']);
  if (trial >= 43 && trial <= 48) {
    const weather = ['Rain Dance', 'Sunny Day', 'Sandstorm', 'Snowscape'][Math.min(trial - 43, 3)];
    const rock = ['Damp Rock', 'Heat Rock', 'Smooth Rock', 'Icy Rock'][Math.min(trial - 43, 3)];
    if (i < 4) set.evs = {hp: 32, atk: 0, def: 17, spa: 0, spd: 17, spe: 0};
    if (trial <= 46) {
      if (i === 0) customize(trial === 46 ? 'Beartic' : 'Goodra-Hisui', trial === 46 ? 'Slush Rush' : 'Shell Armor', [weather, trial === 46 ? 'Ice Punch' : 'Weather Ball', 'Protect']);
      if (i === 1) customize(['Qwilfish', 'Venusaur', 'Excadrill', 'Torterra'][trial - 43], ['Swift Swim', 'Chlorophyll', 'Sand Rush', 'Shell Armor'][trial - 43], [trial === 43 ? 'Aqua Jet' : trial === 44 ? 'Seed Bomb' : 'Earthquake', 'Protect'], trial === 43 ? 'Choice Scarf' : '');
      if (i === 2) customize(['Pelipper', 'Ninetales', 'Tyranitar', 'Abomasnow'][trial - 43], ['Drizzle', 'Drought', 'Sand Stream', 'Snow Warning'][trial - 43], [trial === 43 ? 'Tailwind' : weather, trial === 45 ? 'Ice Beam' : 'Weather Ball', 'Protect'], rock);
      if (i === 3) customize('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']);
    } else {
      if (i === 0) customize('Pelipper', 'Drizzle', ['Tailwind', 'Weather Ball', 'Protect'], 'Damp Rock');
      if (i === 1) customize('Venusaur', 'Chlorophyll', ['Sunny Day', 'Weather Ball', 'Protect'], 'Choice Scarf');
      if (i === 2) customize('Tyranitar', 'Sand Stream', ['Sandstorm', 'Ice Beam', 'Protect'], 'Smooth Rock');
      if (i === 3) customize('Beartic', 'Slush Rush', ['Snowscape', 'Ice Punch', 'Protect'], 'Icy Rock');
      if (i === 4) customize('Ninetales', 'Drought', ['Flamethrower', 'Weather Ball', 'Protect'], 'Heat Rock');
      if (i === 5) customize('Qwilfish', 'Swift Swim', ['Aqua Jet', 'Protect']);
    }
  }
  if (trial >= 49 && trial <= 53) {
    if (i < 4) set.evs = {hp: 32, atk: 0, def: 17, spa: 0, spd: 17, spe: 0};
    if (i === 0) customize('Chimecho', 'Levitate', ['Trick Room', 'Dazzling Gleam', 'Protect']);
    if (i === 1) customize(trial === 52 ? 'Dragonite' : 'Samurott', trial === 52 ? 'Inner Focus' : 'Shell Armor', trial === 52 ? ['Tailwind', 'Dragon Claw', 'Protect'] : ['Aqua Jet', 'Protect']);
    if (i === 2) customize('Slowbro', 'Oblivious', ['Trick Room', 'Thunder Wave', 'Surf', 'Protect']);
    if (i === 3) customize('Scizor', 'Technician', ['Bullet Punch', 'X-Scissor', 'Protect'], trial === 52 ? 'Choice Scarf' : '');
    if (i === 4) customize('Delphox', 'Blaze', ['Flamethrower', 'Protect']);
  }
  if (trial >= 54 && trial <= 60) {
    if (i < 4) set.evs = {hp: 32, atk: 0, def: 17, spa: 0, spd: 17, spe: 0};
    const terrain = ['Electric Terrain', 'Grassy Terrain', 'Psychic Terrain', 'Misty Terrain'][Math.min(trial - 54, 3)];
    if (trial <= 57 || trial === 60) {
      const t = trial === 60 ? 56 : trial;
      if (i === 0) customize(['Pincurchin', 'Rillaboom', 'Indeedee', 'Primarina'][t - 54], ['Electric Surge', 'Grassy Surge', 'Psychic Surge', 'Torrent'][t - 54], [terrain, ['Thunderbolt', 'Seed Bomb', 'Psychic', 'Dazzling Gleam'][t - 54], 'Protect'], 'Terrain Extender');
      if (i === 1) customize(['Eelektross', 'Dragonite', 'Chimecho', 'Hydreigon'][t - 54], ['Levitate', 'Inner Focus', 'Levitate', 'Levitate'][t - 54], [t === 54 ? 'Thunderbolt' : t === 55 ? 'Earthquake' : t === 56 ? 'Psychic' : 'Dragon Pulse', 'Protect']);
      if (i === 2) customize(['Chimecho', 'Torterra', 'Slowbro', 'Delphox'][t - 54], ['Levitate', 'Shell Armor', 'Oblivious', 'Blaze'][t - 54], t === 54 ? ['Hypnosis', 'Psychic', 'Protect'] : t === 55 ? ['Earthquake', 'Seed Bomb', 'Protect'] : t === 56 ? ['Psychic Terrain', 'Psychic', 'Protect'] : ['Will-O-Wisp', 'Flamethrower', 'Protect'], t === 55 ? 'Leftovers' : '');
      if (i === 3) customize(['Dragonite', 'Chimecho', 'Dragonite', 'Chimecho'][t - 54], ['Inner Focus', 'Levitate', 'Inner Focus', 'Levitate'][t - 54], t === 54 ? ['Thunder Punch', 'Protect'] : t === 55 ? ['Energy Ball', 'Protect'] : t === 56 ? ['Aqua Jet', 'Dragon Claw', 'Protect'] : ['Hypnosis', 'Thunder Wave', 'Dazzling Gleam', 'Protect']);
      if (trial === 60 && i === 0) set.moves[0] = 'Psychic Terrain';
      if (t === 57 && i === 5) customize('Goodra-Hisui', 'Shell Armor', ['Dragon Pulse', 'Protect']);
    } else {
      if (i === 0) customize('Pincurchin', 'Electric Surge', ['Electric Terrain', 'Thunderbolt', 'Protect'], 'Terrain Extender');
      if (i === 1) customize('Chimecho', 'Levitate', ['Trick Room', 'Psychic', 'Protect']);
      if (i === 2) customize('Rillaboom', 'Grassy Surge', ['Grassy Terrain', 'Seed Bomb', 'Protect']);
      if (i === 3) customize('Slowbro', 'Oblivious', ['Psychic Terrain', 'Surf', 'Protect']);
      if (i === 4) customize('Primarina', 'Torrent', ['Misty Terrain', 'Dazzling Gleam', 'Protect']);
      if (i === 5) customize('Pelipper', 'Drizzle', ['Weather Ball', 'Surf', 'Protect']);
    }
  }
  if (trial >= 61 && trial <= 65) {
    if (i < 4) set.evs = {hp: 32, atk: 0, def: 17, spa: 0, spd: 17, spe: 0};
    const t = Math.min(trial, 63);
    if (i === 0) customize(['Blastoise', 'Glaceon', 'Charizard'][t - 61], ['Rain Dish', 'Ice Body', 'Solar Power'][t - 61], [["Rain Dance", "Surf", "Ice Beam", "Protect"], ['Snowscape', 'Ice Beam', 'Protect'], ['Sunny Day', 'Flamethrower', 'Protect']][t - 61], ['Damp Rock', 'Icy Rock', 'Heat Rock'][t - 61]);
    if (i === 1 && t === 63) customize('Goodra-Hisui', 'Shell Armor', ['Dragon Pulse', 'Protect']);
    if (i === 2) customize(['Pelipper', 'Vanilluxe', 'Heliolisk'][t - 61], ['Rain Dish', 'Ice Body', 'Solar Power'][t - 61], [["Rain Dance", "Weather Ball", "Protect"], ['Snowscape', 'Ice Beam', 'Protect'], ['Sunny Day', 'Thunderbolt', 'Protect']][t - 61], t === 63 ? 'Sitrus Berry' : 'Leftovers');
    if (i === 3) customize(t === 63 ? 'Slowbro' : 'Goodra-Hisui', t === 63 ? 'Oblivious' : 'Shell Armor', t === 63 ? ['Surf', 'Ice Beam', 'Protect'] : ['Flamethrower', 'Dragon Pulse', 'Protect']);
    if (trial === 65 && i === 1) set.moves = ['Rain Dance', 'Dragon Pulse', 'Protect'];
    if (trial >= 75 && trial <= 87 && b.actions.targetTypeChoices(move.target)) target = 1;
    if ([76, 82].includes(trial) && roster(p) === 3 && move.id === 'seedbomb') target = 2;
    if ((trial === 67 || trial === 71) && roster(p) === 1 && move.id === 'flamethrower') target = 1;
    if (trial >= 66 && trial <= 73 && sideIndex === 1 && b.turn <= 9 && b.actions.targetTypeChoices(move.target)) target = 2;
    if (trial === 74 && sideIndex === 1 && b.actions.targetTypeChoices(move.target)) target = 1;
    if ((trial === 64 || trial === 65) && i === 2) {
      customize('Venusaur', 'Overgrow', ['Leaf Storm', 'Seed Bomb', 'Protect'], 'Venusaurite');
      set.evs = {hp: 20, atk: 0, def: 13, spa: 0, spd: 13, spe: 20};
    }
  }
  if (trial >= 66 && trial <= 74) {
    const t = trial === 74 ? 3 : trial <= 69 ? trial - 66 : trial - 70;
    if (i < 4) set.evs = {hp: 32, atk: 0, def: 17, spa: 0, spd: 17, spe: 0};
    if (i === 0) customize('Altaria', 'Cloud Nine', ['Dragon Pulse', 'Ice Beam', 'Protect']);
    if (i === 1) customize(['Blastoise', 'Charizard', 'Goodra-Hisui', 'Vanilluxe'][t], ['Rain Dish', 'Solar Power', 'Shell Armor', 'Ice Body'][t], [['Surf', 'Dragon Pulse', 'Protect'], ['Flamethrower', 'Dragon Pulse', 'Protect'], ['Weather Ball', 'Dragon Pulse', 'Protect'], ['Ice Beam', 'Weather Ball', 'Protect']][t]);
    if (i === 2) customize(['Pelipper', 'Ninetales', 'Tyranitar', 'Abomasnow'][t], ['Drizzle', 'Drought', 'Sand Stream', 'Snow Warning'][t], [['Rain Dance', 'Weather Ball', 'Protect'], ['Sunny Day', 'Weather Ball', 'Protect'], ['Sandstorm', 'Ice Beam', 'Protect'], ['Snowscape', 'Weather Ball', 'Protect']][t], ['Damp Rock', 'Heat Rock', 'Smooth Rock', 'Icy Rock'][t]);
    if (i === 3) customize(['Qwilfish', 'Venusaur', 'Lycanroc', 'Beartic'][t], ['Swift Swim', 'Chlorophyll', 'Sand Rush', 'Slush Rush'][t], [["Aqua Jet", "Protect"], ['Seed Bomb', 'Protect'], ['Rock Slide', 'Protect'], ['Ice Punch', 'Protect']][t]);
    if (i === 4) customize(t === 2 ? 'Blastoise' : 'Goodra-Hisui', t === 2 ? 'Torrent' : 'Shell Armor', t === 2 ? ['Surf', 'Dragon Pulse', 'Protect'] : ['Weather Ball', 'Dragon Pulse', 'Protect']);
    if (trial === 74 && i === 2) set.moves[1] = 'Ice Beam';
    if (trial === 74 && i === 0) set.evs = {hp: 1, atk: 0, def: 0, spa: 0, spd: 0, spe: 0};
  }
  if (trial >= 75 && trial <= 87) {
    if (i < 4) set.evs = {hp: 32, atk: 0, def: 17, spa: 0, spd: 17, spe: 0};
    if ([75, 80, 81, 85].includes(trial)) {
      if (i === 0) customize('Excadrill', 'Sand Force', ['High Horsepower', 'Rock Slide', 'Smart Strike', 'X-Scissor']);
      if (i === 1) customize(trial === 80 ? 'Altaria' : 'Goodra-Hisui', trial === 80 ? 'Cloud Nine' : 'Shell Armor', ['Dragon Pulse', 'Protect']);
      if (i === 2) customize(trial === 81 ? 'Chimecho' : 'Tyranitar', trial === 81 ? 'Levitate' : 'Sand Stream', trial === 81 ? ['Psychic', 'Protect'] : ['Rock Slide', 'Ice Beam', 'Protect'], trial === 81 ? '' : 'Smooth Rock');
      if (i === 3) customize('Reuniclus', 'Overcoat', ['Psychic', 'Protect']);
    } else if ([76, 77, 82, 86].includes(trial)) {
      const snow = trial === 77 || trial === 86;
      if (i === 0) { customize(snow ? 'Froslass' : 'Heliolisk', snow ? 'Snow Cloak' : 'Sand Veil', snow ? ['Double Team', 'Ice Beam', 'Protect'] : ['Mud-Slap', 'Thunderbolt', 'Protect']); if (snow) set.gender = 'F'; }
      if (i === 1 && (trial === 82 || trial === 86)) customize('Altaria', 'Cloud Nine', ['Dragon Pulse', 'Protect']);
      if (i === 2) customize(snow ? 'Abomasnow' : 'Tyranitar', snow ? 'Snow Warning' : 'Sand Stream', snow ? ['Mud-Slap', 'Ice Beam', 'Protect'] : ['Rock Slide', 'Ice Beam', 'Protect'], snow ? 'Icy Rock' : 'Smooth Rock');
      if (i === 3) customize('Venusaur', 'Overgrow', ['Sweet Scent', 'Seed Bomb', 'Protect']);
    } else if (trial === 78 || trial === 87) {
      if (i === 0) customize('Reuniclus', 'Overcoat', ['Psychic', 'Protect'], 'Leftovers');
      if (i === 1) customize(trial === 87 ? 'Altaria' : 'Kommo-o', trial === 87 ? 'Cloud Nine' : 'Overcoat', ['Dragon Claw', 'Protect']);
      if (i === 2) customize('Tyranitar', 'Sand Stream', ['Rock Slide', 'Ice Beam', 'Protect'], 'Smooth Rock');
      if (i === 3) customize('Venusaur', 'Overgrow', ['Sleep Powder', 'Poison Powder', 'Seed Bomb', 'Protect']);
    } else {
      if (i === 0) customize('Goodra', 'Hydration', ['Rain Dance', 'Dragon Pulse', 'Protect'], 'Leftovers');
      if (i === 1) customize(trial === 83 ? 'Altaria' : 'Vaporeon', trial === 83 ? 'Cloud Nine' : 'Hydration', trial === 83 ? ['Dragon Pulse', 'Protect'] : ['Ice Beam', 'Protect']);
      if (i === 2) customize('Delphox', 'Blaze', ['Will-O-Wisp', 'Psychic', 'Protect']);
      if (i === 3) customize('Chimecho', 'Levitate', ['Psychic', 'Thunder Wave', 'Hypnosis', 'Protect']);
    }
  }
  if (trial === 85 && i === 0) customize('Garchomp', 'Sand Veil', ['Earthquake', 'Rock Slide', 'Iron Head', 'Dragon Claw'], 'Garchompite');
  if (trial >= 88 && trial <= 94) {
    if (i < 4) set.evs = {hp: 32, atk: 0, def: 17, spa: 0, spd: 17, spe: 0};
    if (i === 0) customize('Heliolisk', 'Dry Skin', ['Surf', 'Thunderbolt', 'Protect']);
    if (i === 1) customize(trial >= 92 ? 'Altaria' : 'Toxicroak', trial >= 92 ? 'Cloud Nine' : 'Dry Skin', trial >= 92 ? ['Dragon Pulse', 'Protect'] : ['Drain Punch', 'Rock Slide', 'Protect']);
    if (i === 2) customize(trial === 90 || trial === 92 ? 'Pelipper' : trial === 91 || trial === 93 ? 'Ninetales' : 'Samurott', trial === 90 || trial === 92 ? 'Drizzle' : trial === 91 || trial === 93 ? 'Drought' : 'Shell Armor', trial === 90 || trial === 92 ? ['Surf', 'Weather Ball', 'Protect'] : trial === 91 || trial === 93 ? ['Flamethrower', 'Weather Ball', 'Protect'] : ['Aqua Jet', 'Ice Beam', 'Protect'], trial === 90 || trial === 92 ? 'Damp Rock' : trial === 91 || trial === 93 ? 'Heat Rock' : '');
    if (i === 3) customize('Delphox', 'Blaze', ['Flamethrower', 'Dazzling Gleam', 'Protect']);
    if (i === 4) customize('Toxicroak', 'Dry Skin', ['Drain Punch', 'Protect']);
    if (i === 1 && trial < 92) set.item = 'Leftovers';
    if (i === 4 && trial < 92) customize('Goodra-Hisui', 'Shell Armor', ['Dragon Pulse', 'Protect']);
    if (trial === 94) {
      if (i === 1) customize('Toxicroak', 'Dry Skin', ['Drain Punch', 'Rock Slide', 'Protect']);
      if (i === 2) customize('Samurott', 'Shell Armor', ['Hydro Pump', 'Aqua Jet', 'Protect']);
      if (i === 4) customize('Goodra-Hisui', 'Shell Armor', ['Dragon Pulse', 'Protect']);
    }
  }
  if (items) set.item = items[i];
  if (trial >= 95 && trial <= 104 && trial !== 103) {
    const t = (trial - 95) % 5;
    if (i < 4) set.evs = {hp: 32, atk: 0, def: 17, spa: 0, spd: 17, spe: 0};
    if (i === 0) customize(['Vaporeon','Jolteon','Orthworm','Gogoat','Emolga'][t], ['Water Absorb','Volt Absorb','Earth Eater','Sap Sipper','Motor Drive'][t], [['Ice Beam','Protect'],['Thunderbolt','Protect'],['Iron Head','Protect'],['Seed Bomb','Protect'],['Thunderbolt','Protect']][t]);
    if (i === 1) customize(t === 3 ? 'Torterra' : 'Goodra-Hisui','Shell Armor',t === 3 ? ['Seed Bomb','Earthquake','Protect'] : t === 0 ? ['Surf','Dragon Pulse','Protect'] : t === 2 ? ['Earthquake','Dragon Pulse','Protect'] : ['Thunderbolt','Dragon Pulse','Protect']);
    if (i === 2) customize(t === 0 ? 'Samurott' : t === 2 ? 'Torterra' : t === 3 ? 'Venusaur' : 'Ampharos', t === 0 || t === 2 ? 'Shell Armor' : t === 3 ? 'Overgrow' : 'Static', t === 0 ? ['Hydro Pump','Ice Beam','Protect'] : t === 2 ? ['Earthquake','Mud-Slap','Protect'] : t === 3 ? ['Seed Bomb','Sleep Powder','Protect'] : ['Discharge','Thunder Wave','Zap Cannon','Protect']);
    if (i === 3) customize('Falinks','Battle Armor',['Rock Slide','Protect']);
    if (i === 4) customize('Perrserker','Battle Armor',['Iron Head','Protect']);
  }
  if (trial === 103) {
    if (i === 0) {customize('Farigiraf','Sap Sipper',['Reflect','Psychic','Protect']); set.evs={hp:32,def:14,spd:15,spe:5};}
    if (i === 1) {customize('Wyrdeer','Sap Sipper',['Light Screen','Psychic','Protect']); set.evs={hp:32,def:17,spd:17,spe:0};}
  }
  if (trial >= 105 && trial <= 109) {
    if (i < 4) set.evs={hp:32,def:17,spd:17,spe:0};
    if (i === 0) customize(trial === 107 ? 'Emolga' : 'Samurott',trial === 107 ? 'Motor Drive' : 'Shell Armor',trial === 107 ? ['Quick Attack','Protect'] : ['Aqua Jet','Protect'],trial === 105 ? 'Lum Berry' : '');
    if (i === 1) customize(trial === 108 ? 'Primarina' : 'Goodra-Hisui',trial === 108 ? 'Torrent' : 'Shell Armor',trial === 108 ? ['Misty Terrain','Dazzling Gleam','Protect'] : ['Dragon Pulse','Protect']);
    if (i === 2) customize('Ampharos','Static',['Thunderbolt','Protect'],trial === 106 || trial === 109 ? 'Rocky Helmet' : '');
    if (i === 3) customize('Falinks','Battle Armor',['Rock Slide','Protect']);
    if (i === 4) customize('Perrserker','Battle Armor',['Iron Head','Protect']);
    if (trial === 109 && i === 0) set.evs={hp:0,atk:1,def:0,spd:0,spe:0};
    if (trial === 109 && i === 2) set.evs={hp:0,atk:1,def:0,spd:0,spe:0};
  }
  if (trial >= 110 && trial <= 120) {
    if(i<4)set.evs={hp:32,def:17,spd:17,spe:0};
    if(i===0)customize('Arcanine','Flash Fire',['Flamethrower','Flare Blitz','Dragon Pulse','Protect']);
    if(i===1)customize('Goodra-Hisui','Shell Armor',['Dragon Pulse','Protect']);
    if(i===2)customize('Delphox','Blaze',['Heat Wave','Will-O-Wisp','Fire Punch','Protect']);
    if(i===3)customize('Torterra','Shell Armor',['Mud-Slap','Seed Bomb','Protect']);
    if(trial===112){if(i===0)customize('Goodra-Hisui','Shell Armor',['Dragon Pulse','Protect']);if(i===1)customize('Flareon','Flash Fire',['Flamethrower','Flare Blitz','Shadow Ball','Protect']);}
    if(trial===114&&i===0)customize('Houndoom','Flash Fire',['Flamethrower','Dark Pulse','Protect'],'Houndoominite');
    if(trial===115){if(i===0)customize('Rhyperior','Lightning Rod',['Rock Slide','Protect']);if(i===1)customize('Ampharos','Static',['Thunderbolt','Thunder Wave','Discharge','Protect']);if(i===2)customize('Goodra-Hisui','Shell Armor',['Dragon Pulse','Protect']);}
    if(trial===116){if(i===0){customize('Manectric','Lightning Rod',['Flamethrower','Protect']);set.evs={hp:32,def:14,spd:15,spe:5};}if(i===1)customize('Raichu','Lightning Rod',['Quick Attack','Protect']);if(i===2)customize('Ampharos','Static',['Thunderbolt','Thunder Wave','Discharge','Protect']);}
    if(trial===117){if(i===0)customize('Manectric','Lightning Rod',['Flamethrower','Protect']);if(i===1)customize('Altaria','Cloud Nine',['Tailwind','Dragon Pulse','Protect']);if(i===2)customize('Raichu','Lightning Rod',['Quick Attack','Protect']);if(i===3)customize('Ampharos','Static',['Thunderbolt','Discharge','Protect']);}
    if(trial===118||trial===119){if(i===0)customize(trial===118?'Sceptile':'Manectric',trial===118?'Overgrow':'Lightning Rod',[trial===118?'Dragon Pulse':'Flamethrower','Protect'],trial===118?'Sceptilite':'Manectite');if(i===1)customize('Ampharos','Static',['Thunderbolt','Thunder Wave','Discharge','Protect']);if(i===2)customize('Goodra-Hisui','Shell Armor',['Dragon Pulse','Protect']);}
    if(trial===120){if(i===1)customize('Altaria','Cloud Nine',['Dragon Pulse','Protect']);if(i===3)customize('Venusaur','Overgrow',['Seed Bomb','Protect'],'Venusaurite');}
  }
  if(trial>=121&&trial<=123){
    if(i<4)set.evs={hp:32,def:17,spd:17,spe:0};
    if(i===0)customize('Arcanine','Flash Fire',['Flamethrower','Flare Blitz','Dragon Pulse','Protect']);
    if(i===1)customize(trial===123?'Altaria':'Goodra-Hisui',trial===123?'Cloud Nine':'Shell Armor',['Dragon Pulse','Protect']);
    if(i===2)customize('Delphox','Blaze',['Heat Wave','Will-O-Wisp','Fire Punch','Protect']);
    if(i===3)customize(trial===122?'Pelipper':'Ninetales',trial===122?'Drizzle':'Drought',['Weather Ball','Protect'],trial===122?'Damp Rock':'Heat Rock');
  }
  if(trial===120&&i===3)set.evs={hp:32,def:10,spd:9,spe:15};
  if(trial>=124&&trial<=141){
    if(i<4)set.evs={hp:32,def:17,spd:17,spe:0};
    if(i===0){
      const t=trial<=132?trial-124:trial===137?2:trial===138?8:trial===140?5:trial===141?1:0;
      const names=['Sylveon','Aurorus','Altaria','Gardevoir','Pinsir','Salamence','Glalie','Feraligatr','Primarina'];
      const abs=['Pixilate','Refrigerate','Cloud Nine','Synchronize','Hyper Cutter','Intimidate','Ice Body','Torrent','Liquid Voice'];
      const ms=[['Hyper Voice','Quick Attack','Shadow Ball','Protect'],['Hyper Voice','Body Slam','Rock Slide','Protect'],['Hyper Voice','Double-Edge','Weather Ball','Protect'],['Hyper Voice','Body Slam','Psychic','Double Team'],['Quick Attack','X-Scissor','Protect'],['Hyper Voice','Double-Edge','Dragon Claw','Protect'],['Body Slam','Ice Beam','Weather Ball','Double Team'],['Double-Edge','Body Slam','Aqua Jet','Protect'],['Hyper Voice','Energy Ball','Surf','Protect']];
      const stones=['','','Altarianite','Gardevoirite','Pinsirite','Salamencite','Glalitite','Feraligite',''];
      customize(names[t],abs[t],ms[t],stones[t]);
      if(trial>=133&&trial<=136)set.moves=['Weather Ball','Hyper Voice','Quick Attack','Protect'];
      if(trial===137)set.moves=['Weather Ball','Hyper Voice','Double-Edge','Protect'];
      if(trial===139)set.moves=['Quick Attack','Hyper Voice','Shadow Ball','Protect'];
    }
    if(i===1)customize(trial===136?'Altaria':'Goodra-Hisui',trial===136?'Cloud Nine':'Shell Armor',['Dragon Pulse','Protect']);
    if(i===2)customize('Torterra','Shell Armor',['Seed Bomb','Protect']);
    if(i===3)customize('Perrserker','Battle Armor',['Iron Head','Protect']);
    if([134,136,137,138].includes(trial)&&i===2)customize('Heliolisk','Dry Skin',['Thunderbolt','Protect']);
    if([134,136,137].includes(trial)&&i===3)customize('Pelipper','Drizzle',['Weather Ball','Protect'],'Damp Rock');
    if(trial===135){if(i===2)customize('Arcanine','Flash Fire',['Dragon Pulse','Protect']);if(i===3)customize('Ninetales','Drought',['Weather Ball','Protect'],'Heat Rock');}
    if(trial===138&&i===3)customize('Vaporeon','Water Absorb',['Ice Beam','Protect']);
    if(trial===139&&i===2)customize('Slowbro','Oblivious',['Psychic Terrain','Psychic','Protect']);
    if(trial===141){if(i===2)customize('Torterra','Shell Armor',['Mud-Slap','Seed Bomb','Protect']);if(i===3)customize('Venusaur','Overgrow',['Seed Bomb','Protect'],'Venusaurite');}
  }
  if(trial>=142&&trial<=147){
    if(i<4)set.evs={hp:32,def:17,spd:17,spe:0};
    if(i===0)customize(trial===147?'Pinsir':'Gardevoir',trial===147?'Hyper Cutter':'Synchronize',trial===147?['Quick Attack','X-Scissor','Protect']:['Psychic','Calm Mind','Protect'],trial===147?'Pinsirite':[144,146].includes(trial)?'Lum Berry':'');
    if(i===1)customize('Goodra-Hisui','Shell Armor',['Dragon Pulse','Protect']);
    if(i===2)customize(trial===143?'Delphox':trial===145?'Gardevoir':trial===147?'Salamence':'Chimecho',trial===143?'Blaze':trial===145?'Synchronize':trial===147?'Intimidate':'Levitate',trial===143?['Will-O-Wisp','Psychic','Protect']:trial===145?['Thunder Wave','Psychic','Protect']:trial===147?['Dragon Claw','Protect']:trial===146?['Hypnosis','Psychic','Protect']:['Thunder Wave','Psychic','Protect']);
    if(i===3)customize(trial===146?'Samurott':trial===147?'Rhyperior':'Torterra',trial===146?'Shell Armor':trial===147?'Lightning Rod':'Shell Armor',trial===146?['Ice Beam','Protect']:trial===147?['Crunch','Rock Slide','Protect']:['Seed Bomb','Protect']);
    if(trial===145&&((side===0&&i===2)||(side===1&&i===0)))customize('Chimecho','Levitate',['Thunder Wave','Psychic','Protect']);
    if(trial===146&&i===4)customize('Perrserker','Battle Armor',['Iron Head','Protect']);
  }
  return set;
}));
const roster = p => Number(p.name.slice(-1));
// Privileged request detail so the native legal-action mask can be compared
// with the reference request at every boundary (development fixtures only).
const requestDetail = (session, side) => {
  if (session.battle.ended) return null;
  if (side.requestState === 'teampreview') {
    // A side that has already submitted its picks waits for the opponent.
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
      moves: p.moveSlots.map(m => ({id: ids.moves[m.id], pp: m.pp, disabled: Boolean(m.disabled), target: m.target}))};
  });
  // Reference switch destinations are positions in the request team order,
  // which after preview is the pick order. The native request reports stable
  // roster indices, so map positions back through the fixture name suffix.
  const bench = side.pokemon.map((p, i) => [p, i])
    .filter(([p]) => !p.fainted && !side.active.includes(p)).map(([p]) => roster(p));
  return {kind, slots, bench, preview: []};
};
const compact = session => ({turn: session.battle.turn, rng_seed: session.battle.prng.getSeed(),
  climate: {raw: session.battle.field.weather, effective: session.battle.field.effectiveWeather(), suppressed: session.battle.field.suppressingWeather()},
  field: [...(session.battle.field.weather ? [[ids.conditions[session.battle.field.weather], session.battle.field.weatherState.duration, session.battle.field.weatherState.source.side.n]] : []), ...(session.battle.field.terrain ? [[ids.conditions[session.battle.field.terrain], session.battle.field.terrainState.duration, session.battle.field.terrainState.source.side.n]] : []), ...Object.entries(session.battle.field.pseudoWeather).map(([id, effect]) => [ids.conditions[id], effect.duration, effect.source.side.n])].sort((a, b) => a[0] - b[0]),
  terminated: session.battle.ended, winner: session.battle.ended ? session.battle.winner || null : null,
  sides: session.battle.sides.map(s => ({request: session.battle.ended ? 'Finished' : s.activeRequest?.wait || s.isChoiceDone() ? 'Wait' : s.requestState === 'teampreview' ? 'Preview' : s.requestState === 'switch' ? 'Replacement' : 'Normal',
    conditions: Object.entries(s.sideConditions).map(([id, state]) => [ids.conditions[id], state.duration]).sort((a,b) => a[0]-b[0]),
    pokemon: s.pokemon.map(p => ({roster: roster(p), species: ids.species[p.species.id], hp: p.hp,
      max_hp: p.maxhp, fainted: p.fainted, active_slot: s.active.indexOf(p) >= 0 ? s.active.indexOf(p) : null,
      ability_ending: Boolean(p.abilityState.ending), cached_speed: p.speed ?? null, status: ids.conditions[p.status] ?? 0, boosts: Object.values(p.boosts), stats: [p.maxhp, ...Object.values(p.storedStats)], ability: ids.abilities[p.ability], item: ids.items[p.item] ?? 0, types: p.types.map(t => ids.types[toID(t)]),
      previous_item: ids.items[p.lastItem] ?? 0, can_mega: Boolean(p.canMegaEvo), pp: p.moveSlots.map(m => m.pp), volatiles: Object.keys(p.volatiles).sort()})),
  request_detail: requestDetail(session, s)}))});
const select = (kind, own_slot, destination = 255) => ({kind, own_slot, move_slot: 255, target_location: 0, switch_destination: destination, resource: 'None'});

function choices(session, sideIndex, trial) {
  const b = session.battle, side = b.sides[sideIndex], req = side.activeRequest;
  if (side.requestState === 'teampreview') {
    const order = sideIndex ? [2, 3, 4, 5] : [0, 1, 4, 5];
    return {actions: order.map((r, i) => select('Pick', i, r)), command: `team ${order.map(x => x + 1).join('')}`};
  }
  const actions = [], commands = [];
  const bench = side.pokemon.filter(p => !p.fainted && !side.active.includes(p));
  for (let slot = 0; slot < 2; slot++) {
    const p = side.active[slot];
    if (side.requestState === 'switch') {
      if (!req.forceSwitch[slot]) { commands.push('pass'); continue; }
      const reserve = bench.shift();
      if (reserve) { actions.push(select('Switch', slot, roster(reserve))); commands.push(`switch ${side.pokemon.indexOf(reserve) + 1}`); }
      else { actions.push(select('Pass', slot)); commands.push('pass'); }
      continue;
    }
    if (p.fainted) { actions.push(select('Pass', slot)); commands.push('pass'); continue; }
    if (([3, 8, 20, 39].includes(trial)) && ((b.turn === 2 && slot === 0) || (b.turn === 5 && slot === 1)) && bench.length) {
      const reserve = bench.shift(); actions.push(select('Switch', slot, roster(reserve)));
      commands.push(`switch ${side.pokemon.indexOf(reserve) + 1}`); continue;
    }
    if ([113,116,140].includes(trial) && sideIndex===0 && slot===0 && (b.turn===4 || b.turn===6) && bench.length) {
      const index=b.turn===6?bench.findIndex(p=>roster(p)===0):0;
      if(index>=0){const reserve=bench.splice(index,1)[0];actions.push(select('Switch',slot,roster(reserve)));commands.push(`switch ${side.pokemon.indexOf(reserve)+1}`);continue;}
    }
    if (trial===136 && sideIndex===0 && slot===1 && b.turn===4 && bench.length){const reserve=bench.shift();actions.push(select('Switch',slot,roster(reserve)));commands.push(`switch ${side.pokemon.indexOf(reserve)+1}`);continue;}
    if ([92, 93].includes(trial) && sideIndex === 0 && b.turn === 4 && slot === 1 && bench.length) {
      const reserve = bench.shift(); actions.push(select('Switch', slot, roster(reserve))); commands.push(`switch ${side.pokemon.indexOf(reserve) + 1}`); continue;
    }
    if ([80, 82, 83, 86, 87].includes(trial) && sideIndex === 0 && b.turn === (trial === 83 ? 3 : 4) && slot === 1 && bench.length) {
      const reserve = bench.shift(); actions.push(select('Switch', slot, roster(reserve))); commands.push(`switch ${side.pokemon.indexOf(reserve) + 1}`); continue;
    }
    if (trial === 70 && sideIndex === 0 && b.turn === 6 && slot === 0 && bench.some(p => roster(p) === 0)) {
      const reserve = bench.splice(bench.findIndex(p => roster(p) === 0), 1)[0];
      actions.push(select('Switch', slot, roster(reserve))); commands.push(`switch ${side.pokemon.indexOf(reserve) + 1}`); continue;
    }
    if (trial >= 70 && trial <= 73 && sideIndex === 0 && b.turn === 4 && slot === 0 && bench.length) {
      const reserve = bench.shift(); actions.push(select('Switch', slot, roster(reserve)));
      commands.push(`switch ${side.pokemon.indexOf(reserve) + 1}`); continue;
    }
    if (trial === 59 && b.turn === 2 && slot === 0 && bench.length) {
      const reserve = sideIndex === 1 ? bench.splice(bench.findIndex(p => roster(p) === 5), 1)[0] : bench.shift(); actions.push(select('Switch', slot, roster(reserve)));
      commands.push(`switch ${side.pokemon.indexOf(reserve) + 1}`); continue;
    }
    if (trial === 53 && sideIndex === 0 && b.turn === 2 && slot === 0 && bench.length) {
      const reserve = bench.shift(); actions.push(select('Switch', slot, roster(reserve)));
      commands.push(`switch ${side.pokemon.indexOf(reserve) + 1}`); continue;
    }
    if (trial === 47 && b.turn === 3 && slot === 0 && bench.length) {
      const reserve = bench.shift(); actions.push(select('Switch', slot, roster(reserve)));
      commands.push(`switch ${side.pokemon.indexOf(reserve) + 1}`); continue;
    }
    if (trial === 21 && ((b.turn === 3 && slot === 0) || (b.turn === 4 && slot === 1)) && bench.length) {
      const reserve = bench.shift(); actions.push(select('Switch', slot, roster(reserve)));
      commands.push(`switch ${side.pokemon.indexOf(reserve) + 1}`); continue;
    }
    let order = trial === 1 ? [2, 0, 1, 3] : trial === 2 && b.turn <= 3 ? [3, 0, 1, 2] : [0, 1, 2, 3];
    if ((trial === 10 || trial === 11) && b.turn > 2) order = b.turn % 3 === 0 && p.hp < p.maxhp / 2 && p.species.baseSpecies === 'Chimecho' ? [2, 1, 0, 3] : [1, 2, 0, 3];
    if ((trial === 15 || trial === 16) && b.turn > 2) order = [1, 2, 0, 3];
    if (trial === 21 && sideIndex === 1 && b.turn > 2) order = [1, 0, 2, 3];
    if (trial >= 34 && trial <= 42) {
      const isScreen = ['Chimecho', 'Delphox'].includes(p.species.baseSpecies);
      if (isScreen) order = b.turn <= 2 ? [b.turn - 1] : b.turn === 3 ? [0] : [2, 3, 0, 1];
      else if (['Dragonite', 'Salamence', 'Gallade', 'Sharpedo', 'Noivern'].includes(p.species.baseSpecies)) order = [1, 2, 0].includes(b.turn) || b.turn === 6 ? [0] : [1, 2, 0];
    }
    if (trial === 37 && p.species.baseSpecies === 'Gallade') order = b.turn === 1 ? [0] : [1, 2, 0];
    if (trial >= 43 && trial <= 46 && roster(p) < 4) {
      const setter = roster(p) === 0 || roster(p) === 2;
      order = b.turn <= 2 && setter ? [0] : b.turn === 3 && setter ? [1] : b.turn <= 9 ? [setter ? 2 : 1] : [setter ? 1 : 0];
    }
    if (trial === 47 && roster(p) < 4) order = b.turn <= 4 ? [0] : [1, 2, 0];
    if (trial === 48 && roster(p) < 4) {
      if (p.species.baseSpecies === 'Venusaur') order = b.turn <= 2 ? [0] : [1, 2, 0];
      else order = b.turn === 1 ? [0] : [1, 2, 0];
    }
    if (trial >= 49 && trial <= 53 && roster(p) < 4) {
      const r = roster(p);
      if (r === 0) order = b.turn === 1 || (trial === 51 && b.turn === 2) ? [0] : b.turn <= 6 ? [2] : [1, 2, 0];
      else if (r === 2) order = trial === 50 && b.turn === 1 ? [0] : trial === 52 && b.turn === 1 ? [1] : b.turn <= 6 ? [3] : [2, 1, 3];
      else if (r === 1 && trial === 52) order = b.turn === 1 ? [0] : b.turn <= 6 ? [2] : [1, 2, 0];
      else order = trial === 52 && r === 3 ? [0, 1, 2] : b.turn <= 6 ? [r === 3 ? 2 : 1] : [0];
    }
    if ((trial >= 54 && trial <= 57) || trial === 60) {
      const r = roster(p);
      if (r === 0) order = b.turn <= 2 ? [0] : b.turn === 3 ? [1] : b.turn <= 9 ? [2] : [1, 2, 0];
      else if (r === 2) order = b.turn <= 3 ? [0] : b.turn <= 9 ? [2] : [1, 0, 2];
      else if (r === 3 && trial === 57) order = b.turn <= 3 ? [b.turn === 3 ? 1 : 0] : b.turn <= 9 ? [3] : [2, 1, 0, 3];
      else if (r < 4) order = b.turn === 1 || b.turn === 3 ? [0] : b.turn <= 9 ? [r === 3 && [56, 60].includes(trial) ? 2 : 1] : [0, 1, 2];
    }
    if (trial === 57 && roster(p) === 1 && b.turn <= 3) order = b.turn === 1 ? [1] : [0];
    if (trial === 57 && roster(p) === 2 && b.turn <= 3) order = b.turn === 1 ? [2] : [0];
    if (trial === 57 && roster(p) === 3 && b.turn <= 3) order = b.turn === 1 ? [3] : [b.turn === 3 ? 1 : 0];
    if (trial === 58 || trial === 59) {
      const r = roster(p);
      if (r < 4) order = b.turn <= 3 ? [0] : b.turn <= 6 ? [2] : [1, 2, 0];
      else if (r === 4) order = b.turn === 2 ? [0] : [1, 2, 0];
    }
    if (trial >= 61 && trial <= 65 && roster(p) < 4) {
      const r = roster(p);
      const caster = r === 0 || (r === 2 && trial !== 64);
      if (caster) order = b.turn <= 2 ? [0] : b.turn <= 3 || b.turn > 9 ? [1, 2, 3, 0] : [trial === 61 && r === 0 ? 3 : 2];
      else if (r === 2) order = b.turn <= 3 || b.turn > 9 ? [0, 1, 2] : [2];
      else order = b.turn <= 3 || b.turn > 9 ? [0, 1, 2, 3] : [r === 1 ? (trial === 61 || trial === 62 ? 3 : 1) : 2];
    }
    if (trial === 65 && roster(p) === 0 && b.turn <= 2) order = [2];
    if (trial === 65 && roster(p) === 1) order = b.turn <= 2 ? [0] : [1, 2, 0];
    if (trial === 65 && roster(p) === 3 && b.turn <= 3) order = [2];
    if (trial >= 66 && trial <= 74 && roster(p) < 4) {
      const r = roster(p);
      const stall = trial <= 69 ? 9 : 3;
      if (r === 0) order = trial === 74 ? [0, 1, 2] : b.turn <= stall ? [2] : [0, 1, 2];
      else if (r === 2) order = b.turn <= 2 ? [0] : b.turn === 3 || b.turn > stall ? [1, 2, 0] : [2];
      else order = b.turn === 1 || b.turn === 3 || b.turn > stall ? [0, 1, 2] : [r === 1 ? 2 : 1];
    }
    if (trial >= 75 && trial <= 87 && roster(p) < 4) {
      const r = roster(p);
      if ([75, 80, 81, 85].includes(trial) && r === 0) order = [(b.turn - 1) % 4, 0, 1, 2, 3];
      else if ([76, 77, 82, 86].includes(trial)) {
        if (r === 0) order = b.turn <= 2 ? [0] : [1, 2, 0];
        else if (r === 3) order = b.turn === 2 ? [0] : [1, 2, 0];
        else if (r === 2) order = (trial === 77 || trial === 86) && b.turn <= 2 ? [0] : [1, 0, 2];
      } else if ((trial === 78 || trial === 87) && r === 3) order = b.turn <= 3 ? [b.turn % 2 ? 0 : 1] : [2, 3, 0, 1];
      else if ([79, 83, 84].includes(trial)) {
        if (r === 0) order = b.turn === 1 ? [0] : b.turn <= 7 ? [2] : [1, 2, 0];
        else if (r === 1) order = trial === 83 && b.turn <= 7 ? [1] : [0, 1];
        else if (r === 2) order = b.turn <= 7 ? [0] : [1, 2, 0];
        else if (r === 3) order = b.turn <= 7 ? [b.turn <= 3 ? 0 : b.turn === 4 ? 1 : 2] : [0, 1, 2, 3];
      }
      if ([80, 82, 86, 87].includes(trial) && r === 1 && b.turn <= 3) order = [1];
    }
    if (trial >= 88 && trial <= 94 && roster(p) < 4) {
      const r = roster(p);
      if (r === 0) order = trial === 88 && b.turn <= 3 ? [0] : [1, 0, 2];
      else if (r === 1) order = trial >= 92 && b.turn <= 3 ? [1] : [0, 1, 2];
      else if (r === 2) order = trial === 88 && b.turn === 2 ? [1] : trial === 89 ? [1, 0, 2] : [0, 1, 2];
      else if (r === 3) order = [0, 1, 2];
    }
    if (trial >= 92 && trial <= 94 && roster(p) === 0 && [1, 3].includes(b.turn)) order = [2];
    if ([92, 93].includes(trial) && roster(p) === 2 && b.turn <= 3) order = [1];
    if (trial === 93 && [2, 3].includes(roster(p)) && b.turn === 4) order = [2];
    if (trial >= 95 && trial <= 104 && trial !== 103 && roster(p) < 4) {
      const r = roster(p), t = (trial - 95) % 5;
      if (r === 0) order = b.turn <= 9 ? [1] : [0,1];
      if (r === 1) order = b.turn <= 9 ? [b.turn === 2 ? 1 : 0] : [0,1,2];
      if (r === 2) order = b.turn <= 9 ? [t === 1 || t === 4 ? (b.turn === 4 ? 2 : trial >= 100 ? 1 : 0) : b.turn === 2 || trial >= 100 ? 1 : 0] : [0,1,2];
      if (r === 3) order = b.turn <= 9 ? [1] : [0,1];
      if (r === 0 && b.turn <= 9 && b.turn !== 3) order = [0];
    }
    if (trial === 103 && roster(p) < 2) order = b.turn <= 2 ? [0] : [1,0,2];
    if (trial >= 105 && trial <= 109 && roster(p) < 4) {
      const r=roster(p);
      order=r===0 ? [0,1] : r===1 ? trial===108&&b.turn===1 ? [0] : b.turn<=8 ? [trial===108?2:1] : [trial===108?1:0] : r===2 ? [0,1] : b.turn<=8 ? [1] : [0,1];
    }
    if (trial>=110&&trial<=120&&roster(p)<4) {
      const r=roster(p);
      if(trial<=114||trial===120){
        if(r===0||r===1){const holder=trial===112?r===1:r===0;order=holder ? [b.turn===1?2:b.turn%2?1:0,0,1,2,3] : b.turn<=9?[trial===120?1:1]:[0,1];if(trial===114&&holder)order=b.turn===1?[1]:[0,1,2];if(trial===112&&holder&&b.turn===1)order=[2];}
        if(r===2)order=b.turn===1&&![111,112].includes(trial)?[1]:b.turn===3?[2]:[0,1,2,3];
        if(r===3)order=b.turn<=9?[trial===120?1:2]:[trial===120?0:1,0];
        if(trial===111&&r===0&&b.turn===3)order=[3];
        if(trial===112&&r===1&&b.turn===3)order=[3];
      }else{
        if(r===0)order=b.turn===3?[1]:[0,1];
        if(r===1)order=trial===117?b.turn===2?[0]:b.turn<=9?[2]:[1,0,2]:trial===116?b.turn<=9?[1]:[0,1]:b.turn<=9?[b.turn===2?1:b.turn===5?2:0]:[0,1,2,3];
        if(r===2)order=trial===116?[b.turn===2?1:b.turn===5?2:0,0,1,2,3]:b.turn<=9?[1]:[0,1];
        if(r===3)order=trial===117?[b.turn===5?1:0,0,1,2]:b.turn<=9?[2]:[1,0,2];
      }
    }
    if(trial===110&&roster(p)===0&&b.turn===4)order=[2];
    if([111,112].includes(trial)&&roster(p)===(trial===111?1:0))order=[0,1];
    if([111,112].includes(trial)&&roster(p)===2)order=[0,1,2,3];
    if([111,112].includes(trial)&&roster(p)===3&&b.turn<=2)order=[0];
    if(trial===116&&roster(p)>=4&&b.turn<=6)order=[3,1,0];
    if(trial>=121&&trial<=123&&roster(p)<4){const r=roster(p);order=r===0?[b.turn%2?1:0,0,1,2,3]:r===1?b.turn<=9?[1]:[0,1]:r===2?b.turn===1?[1]:[0,1,2,3]:b.turn<=9?[1]:[0,1];}
    if(trial>=124&&trial<=141&&roster(p)<4){
      const r=roster(p);
      if(r===0){
        order=trial<=132||trial===140||trial===141?[((b.turn-1)%3),0,1,2,3]:trial===138?b.turn<=2?[0]:[1,0,2,3]:trial===139?b.turn<=3?[0]:[1,2,0,3]:trial===137?b.turn<=2?[0]:[1,2,0,3]:b.turn<=2||b.turn===4?[0]:[1,2,0,3];
        if([127,130].includes(trial)&&b.turn===4)order=[3];
      }else if(r===1)order=b.turn<=9?[1]:[0,1];
      else if(r===2)order=trial===139?b.turn===1?[0]:[1,0,2]:trial===141?b.turn<=3?[0]:[1,0,2]:b.turn<=9?[1]:[0,1];
      else order=b.turn<=9?[1]:[0,1];
    }
    if([136,140].includes(trial)&&roster(p)>=4&&b.turn<=6)order=[3,1,0];
    if(trial===141&&roster(p)===3&&b.turn<=9)order=[0];
    if([126,127,128,129,130,131,140].includes(trial)&&roster(p)===0&&b.turn===2)order=[0];
    if(trial===138&&[2,3].includes(roster(p))&&b.turn<=2)order=[0];
    if(trial>=142&&trial<=147&&roster(p)<4){const r=roster(p);order=r===0?trial===147?[0,1,2]:b.turn<=3?[1]:[0,1,2]:r===1?b.turn<=9?[1]:[0,1]:r===2?trial===147?b.turn<=9?[1]:[0,1]:b.turn===1||trial===143&&b.turn<=2?[0]:[1,0,2]:trial===146||trial===147?[0,1,2]:b.turn<=9?[1]:[0,1];}
    if(trial===145&&roster(p)===2&&b.turn<=3)order=[0];
    const moveSlot = order.find(i => req.active[slot].moves[i] && !req.active[slot].moves[i].disabled && req.active[slot].moves[i].pp !== 0) ?? 0;
    const move = req.active[slot].moves[moveSlot];
    let target = b.actions.targetTypeChoices(move.target) ? ([30, 32].includes(trial) ? slot + 1 : trial === 21 ? slot + 1 : trial === 27 ? 2 - slot : trial === 28 ? 1 : (b.turn % 2) + 1) : 0;
    if(trial>=142&&trial<=147&&b.actions.targetTypeChoices(move.target))target=roster(p)===0?2:1;
    if(trial===146&&roster(p)===2&&b.turn>1&&b.actions.targetTypeChoices(move.target))target=2;
    if(trial>=124&&trial<=141&&b.actions.targetTypeChoices(move.target))target=roster(p)===0&&trial===141?2:1;
    if(trial===141&&roster(p)===2&&move.id==='mudslap')target=2;
    if(trial===141&&roster(p)===3&&b.turn<=9)target=-1;
    if(trial>=121&&trial<=123&&b.actions.targetTypeChoices(move.target))target=roster(p)===0?2:1;
    if (trial>=110&&trial<=120&&b.actions.targetTypeChoices(move.target)) target=roster(p)===0||roster(p)===1?2:1;
    if (trial===112&&roster(p)===2&&move.id==='firepunch')target=2;
    if([111,112].includes(trial)&&roster(p)===3&&move.id==='mudslap'&&b.turn<=2)target=-1;
    if (trial===120&&roster(p)===0&&b.turn>1&&b.actions.targetTypeChoices(move.target))target=2;
    if (trial >= 105 && trial <= 109 && b.actions.targetTypeChoices(move.target)) target=roster(p)===2 ? 2 : 1;
    if (trial >= 95 && trial <= 104 && trial !== 103 && b.actions.targetTypeChoices(move.target)) target = roster(p) === 0 && b.turn <= 9 ? -2 : roster(p) === 0 ? 2 : roster(p) === 1 && (b.turn === 2 || [1,3,4].includes((trial-95)%5)) ? -1 : 1;
    if (trial === 94 && roster(p) === 2 && move.id === 'hydropump') target = b.turn === 1 ? 2 : 1;
    if (trial === 88 && roster(p) === 2 && move.id === 'icebeam') target = 2;
    if (trial >= 88 && trial <= 94 && b.actions.targetTypeChoices(move.target) && !(trial === 88 && roster(p) === 2 && move.id === 'icebeam') && !(trial === 94 && roster(p) === 2 && move.id === 'hydropump')) target = 1;
    if (trial >= 75 && trial <= 87 && b.actions.targetTypeChoices(move.target)) target = 1;
    if ([76, 82].includes(trial) && roster(p) === 3 && move.id === 'seedbomb') target = 2;
    if ((trial === 67 || trial === 71) && roster(p) === 1 && move.id === 'flamethrower') target = 1;
    if (trial >= 66 && trial <= 73 && sideIndex === 1 && b.turn <= 9 && b.actions.targetTypeChoices(move.target)) target = 2;
    if (trial === 74 && sideIndex === 1 && b.actions.targetTypeChoices(move.target)) target = 1;
    if ((trial === 64 || trial === 65) && roster(p) === 0 && move.id === 'flamethrower') target = 1;
    if (trial === 55 && roster(p) === 0 && move.id === 'seedbomb') target = 1;
    if (trial === 60 && sideIndex === 1 && move.id === 'aquajet') target = -1;
    if ([54, 56, 57].includes(trial) && roster(p) >= 2 && b.turn <= 3 && b.actions.targetTypeChoices(move.target)) target = trial === 57 ? (b.turn === 2 ? 1 : 2) : b.turn === 1 ? 1 : 2;
    if (trial === 57 && roster(p) === 1 && b.turn <= 3 && b.actions.targetTypeChoices(move.target)) target = b.turn === 2 ? 1 : 2;
    actions.push({kind: 'Move', own_slot: slot, move_slot: move.id === 'struggle' ? 255 : moveSlot,
      target_location: target, switch_destination: 255, resource: req.active[slot].canMegaEvo && b.turn === (trial===147?4:[9,114,119,126,127,128,129,130,131,137,140].includes(trial) ? 2 : 1) && (slot === 0 || [120,141].includes(trial)&&roster(p)===3) ? 'Mega' : 'None'});
    commands.push(`move ${moveSlot + 1}${target ? ` ${target}` : ''}${actions.at(-1).resource === 'Mega' ? ' mega' : ''}`);
  }
  return {actions, command: commands.join(', ')};
}

verifyReference();
const fixtures = [];
const cases = [
  ...Array.from({length:6},(_,i)=>({trial:142+i,seedWord:1640+i})),
  ...Array.from({length:8},(_,i)=>({trial:i%2?147:146,seedWord:1660+i})),
  ...Array.from({length:18},(_,i)=>({trial:124+i,seedWord:1600+i})),
  ...Array.from({length:3},(_,i)=>({trial:121+i,seedWord:1530+i})),
  ...Array.from({length:11},(_,i)=>({trial:110+i,seedWord:1500+i})),
  ...Array.from({length:6},(_,i)=>({trial:111+i%2,seedWord:1520+i})),
  ...Array.from({length:15},(_,i)=>({trial:105+i%5,seedWord:1400+i})),
  {trial:103,seedWord:1310},
  ...Array.from({length:8},(_,i)=>({trial:95+i,seedWord:1300+i})),
  {trial:104,seedWord:1311},
  ...Array.from({length: 27}, (_, trial) => ({trial, seedWord: trial + 4})),
  ...Array.from({length: 8}, (_, i) => ({trial: 13, seedWord: i + 100})),
  {trial: 28, seedWord: 32},
  ...Array.from({length: 5}, (_, i) => ({trial: 29 + i, seedWord: 400 + i})),
  ...Array.from({length: 9}, (_, i) => ({trial: 34 + i, seedWord: 500 + i})),
  ...Array.from({length: 8}, (_, i) => ({trial: 37, seedWord: 510 + i})),
  ...Array.from({length: 7}, (_, i) => ({trial: 88 + i, seedWord: 1200 + i})),
  ...Array.from({length: 4}, (_, i) => ({trial: i % 2 ? 90 : 88, seedWord: 1220 + i})),
  ...Array.from({length: 13}, (_, i) => ({trial: 75 + i, seedWord: 1100 + i})),
  ...Array.from({length: 8}, (_, i) => ({trial: i % 2 ? 76 : 77, seedWord: 1120 + i})),
  ...Array.from({length: 9}, (_, i) => ({trial: 66 + i, seedWord: 1000 + i})),
  ...Array.from({length: 5}, (_, i) => ({trial: 61 + i, seedWord: 900 + i})),
  ...Array.from({length: 6}, (_, i) => ({trial: 61 + i % 3, seedWord: 910 + i})),
  ...Array.from({length: 7}, (_, i) => ({trial: 54 + i, seedWord: 800 + i})),
  ...Array.from({length: 4}, (_, i) => ({trial: 54 + i, seedWord: 810 + i, items: ['', '', '', '', '', '']})),
  ...Array.from({length: 3}, (_, i) => ({trial: 57, seedWord: 820 + i})),
  ...Array.from({length: 5}, (_, i) => ({trial: 49 + i, seedWord: 700 + i})),
  ...Array.from({length: 3}, (_, i) => ({trial: 52, seedWord: 710 + i})),
  ...Array.from({length: 6}, (_, i) => ({trial: 43 + i, seedWord: 600 + i})),
  ...Array.from({length: 4}, (_, i) => ({trial: 43 + i, seedWord: 610 + i, items: ['', '', '', '', '', '']})),
  {trial: 27, seedWord: 31, items: ['Life Orb', 'Choice Scarf', 'Rocky Helmet', 'Focus Sash', 'Leftovers', 'Sitrus Berry']},
  ...[10, 11, 13].map(trial => ({trial, seedWord: trial + 200, items: ['Sitrus Berry', 'Lum Berry', 'Life Orb', 'Expert Belt', 'Choice Scarf', 'Leftovers']})),
  {trial: 6, seedWord: 301, items: ['Life Orb', 'Sitrus Berry', 'Rocky Helmet', 'Focus Sash', 'Choice Scarf', 'Leftovers']},
];
for (const {trial, seedWord, items} of cases) {
  const sourceTeams = teamsFor(trial, items), seed = [1, 2, 3, seedWord];
  const session = new ReferenceSession({teams: sourceTeams, seed, record_rng: true});
  const dependency={sync:[],hypercutter:[],body_slam_paralyses:0};
  const conversions={type:[],power:[],hit:[],stats:[]};
  const fireRod={hit:[],redirect:[],power:[],accuracy:[],ends:[],spread_damage:[]};
  const originalAccuracy=session.battle.actions.hitStepAccuracy;
  session.battle.actions.hitStepAccuracy=function(targets,source,move) {
    const before=this.battle.prng.getSeed(),always=move.accuracy===true;
    const result=originalAccuracy.call(this,targets,source,move);
    if(move.id==='heatwave')fireRod.accuracy.push({always,targets:targets.map(p=>({side:p.side.n,roster:roster(p),ability:p.ability})),seed_unchanged:before===this.battle.prng.getSeed(),results:result});
    return result;
  };
  const originalSingleEvent=session.battle.singleEvent;
  session.battle.singleEvent=function(event,effect,state,target,...rest){
    if(event==='End'&&effect.id==='flashfire')fireRod.ends.push({hp:target.hp,volatile_before:Boolean(target.volatiles.flashfire),condition:effect.effectType==='Condition'});
    return originalSingleEvent.call(this,event,effect,state,target,...rest);
  };
  const weatherBallCoverage = new Set();
  const eventStack = [];
  const eventContexts = [];
  const staticProbes = [];
  const originalChance=session.battle.randomChance;
  session.battle.randomChance=function(numerator,denominator) {
    const result=originalChance.call(this,numerator,denominator);
    if (this.effect?.id==='static' && numerator===3 && denominator===10) {
      const ctx=eventContexts.findLast(x=>x.event==='DamagingHit');
      if (!ctx) throw new Error('Static chance outside DamagingHit');
      staticProbes.push({chance:result,actor_hp:ctx.source.hp,holder_hp:this.effectState.target.hp,holder_item:this.effectState.target.item,actor_hp_before_handlers:ctx.source_hp,preceding_protocol:this.log.slice(ctx.log_start),status_before:ctx.source.status,item_before:ctx.source.item,electric:ctx.source.hasType('Electric'),misty:this.field.terrain==='mistyterrain'&&ctx.source.isGrounded(),log_start:this.log.length,actor:ctx.source});
    }
    return result;
  };
  let sapSideTies = 0;
  const absorption = [];
  const drySkinInteractions = {water: [], power: [], healing: [], damage: []};
  const weatherAbilityInteractions = {force: [], accuracy: [], powder: [], sand_damage: [], sand_immunity: [], hydration_damage: []};
  const suppressionInteractions = {power: [], speed: [], weather_ball: [], healing: [], sand: [], solar: []};
  const solarPowerCoverage = {boosted: 0, inactive: 0, thickfat_ties: 0, inactive_thickfat_ties: 0, active_thickfat_ties: 0};
  const terrainInteractions = {power: [], status: [], priority: []};
  const screenCoverage = {physical: 0, special: 0, critical: 0};
  const originalRunEvent = session.battle.runEvent;
  session.battle.runEvent = function(event, target, source, move, ...rest) {
    eventStack.push(event);
    eventContexts.push({event,target,source,source_hp:source?.hp,log_start:this.log.length});
    if (event === 'TryHitSide' && ['reflect','lightscreen','tailwind'].includes(move?.id) && source?.side.active.filter(p=>p?.ability==='sapsipper'&&!p.fainted).length===2 && source.side.active[0].speed===source.side.active[1].speed) sapSideTies++;
    if(event==='DamagingHit'&&move?.id==='discharge')fireRod.spread_damage.push(...(Array.isArray(target)?target:[target]).filter(p=>p&&p.ability!=='lightningrod').map(p=>({side:p.side.n,roster:roster(p),hp:p.hp,source_ability:source.ability})));
    if(event==='AfterSetStatus'&&move?.id==='bodyslam'&&rest[0]?.id==='par')dependency.body_slam_paralyses++;
    const syncBefore=event==='AfterSetStatus'&&target?.ability==='synchronize'?{status:rest[0]?.id??null,target:target,source:source,effect:move?.id??null,target_hp:target.hp,source_hp:source?.hp,source_status:source?.status,source_types:source?.getTypes(),source_grounded:source?.isGrounded(),terrain:this.field.terrain,log_start:this.log.length}:null;
    const cutterBefore=event==='TryBoost'&&target?.ability==='hypercutter'?{before:{...rest[0]},target:target,source:source}:null;
    const converterAbilities=['pixilate','aerilate','refrigerate','galvanize','normalize','dragonize','liquidvoice'];
    const converter=converterAbilities.includes(target?.ability)?target:null;
    const typeBefore=event==='ModifyType'&&converter?move.type:null;
    const powerBefore=event==='BasePower'&&converter?rest[0]:null;
    const convertedHits=event==='TryHit'&&converterAbilities.includes(source?.ability)?(Array.isArray(target)?target:[target]).map((p,index)=>({p,index,hp:p.hp,protected:Boolean(p.volatiles.protect)})):[];
    const converterStats=['ModifyAtk','ModifySpA'].includes(event)&&converter?{event,ability:converter.ability,type:move.type,thickfat:source?.ability==='thickfat',before:rest[0]}:null;
    const fireTargets=event==='TryHit'?(Array.isArray(target)?target:[target]).map((mon,index)=>({mon,index})).filter(x=>['flashfire','lightningrod'].includes(x.mon?.ability)).map(x=>({...x,hp:x.mon.hp,protected:Boolean(x.mon.volatiles.protect),spa:x.mon.boosts.spa,charged:Boolean(x.mon.volatiles.flashfire)})):[];
    const rodCandidates=event==='RedirectTarget'?this.getAllActive().filter(p=>p.ability==='lightningrod'&&!p.fainted).map(p=>({side:p.side.n,roster:roster(p),speed:p.speed,order:p.abilityState.effectOrder,valid:this.validTarget(p,source,['randomNormal','adjacentFoe'].includes(move.target)?'normal':move.target)})):[];
    const redirectSeed=event==='RedirectTarget'?this.prng.getSeed():null;
    const firePower=['ModifyAtk','ModifySpA'].includes(event)&&target?.volatiles?.flashfire?{event,type:move.type,before:rest[0],ability:target.ability,thickfat:source?.ability==='thickfat'}:null;
    const absorberTargets = event === 'TryHit' ? (Array.isArray(target) ? target : [target]).map((mon,index)=>({mon,index})).filter(({mon})=>['waterabsorb','voltabsorb','eartheater','sapsipper','motordrive'].includes(mon?.ability)).map(x=>({...x,hp:x.mon.hp,maxhp:x.mon.maxhp,boosts:{...x.mon.boosts},protected:Boolean(x.mon.volatiles.protect),ally:x.mon.isAlly(source)})) : [];
    const dryTargets = event === 'TryHit' && move?.type === 'Water' ? (Array.isArray(target) ? target : [target]).map((mon, index) => ({mon, index})).filter(({mon}) => mon?.ability === 'dryskin').map(({mon, index}) => ({mon, index, hp: mon.hp, maxhp: mon.maxhp, protected: Boolean(mon.volatiles.protect), ally: mon.isAlly(source), self: mon === source})) : [];
    if (event === 'WeatherModifyDamage' && move?.id === 'weatherball' && this.field.weather) weatherBallCoverage.add(this.field.weather);
    if (event === 'ModifyDamage' && source !== target && move && source?.side) {
      const screen = move.category === 'Physical' ? 'reflect' : move.category === 'Special' ? 'lightscreen' : null;
      if (screen && source.side.sideConditions[screen]) {
        screenCoverage[move.category.toLowerCase()]++;
        if (source.getMoveHitData(move).crit) screenCoverage.critical++;
      }
    }
    const rawWeather = this.field.weather;
    const suppressed = this.field.suppressingWeather();
    const solarHolder = event === 'ModifySpA' && target?.ability === 'solarpower';
    if (solarHolder) {
      if (this.field.isWeather('sunnyday')) solarPowerCoverage.boosted++; else solarPowerCoverage.inactive++;
      if (source?.ability === 'thickfat' && source.speed === target.speed) {
        solarPowerCoverage.thickfat_ties++;
        if (!this.field.isWeather('sunnyday')) solarPowerCoverage.inactive_thickfat_ties++;
        else solarPowerCoverage.active_thickfat_ties++;
      }
    }
    const terrain = this.field.terrain;
    const result = originalRunEvent.call(this, event, target, source, move, ...rest);
    if(syncBefore){const x=syncBefore;dependency.sync.push({status:x.status,effect:x.effect,target_hp:x.target_hp,source_hp:x.source_hp,source_status_before:x.source_status,source_status_after:x.source?.status,source_types:x.source_types,source_grounded:x.source_grounded,terrain:x.terrain,protocol:this.log.slice(x.log_start)});}
    if(cutterBefore)dependency.hypercutter.push({before:cutterBefore.before,after:{...result},enemy:cutterBefore.source?.side!==cutterBefore.target.side,turn:this.turn});
    if(typeBefore!==null)conversions.type.push({ability:converter.ability,move:move.id,category:move.category,before:typeBefore,after:move.type,marker:move.typeChangerBoosted?.id??null,weather:rawWeather,suppressed,species:converter.species.id,turn:this.turn});
    if(powerBefore!==null)conversions.power.push({ability:converter.ability,move:move.id,type:move.type,marker:move.typeChangerBoosted?.id??null,before:powerBefore,after:result,weather:rawWeather,suppressed});
    for(const x of convertedHits){const value=Array.isArray(result)?result[x.index]:result;conversions.hit.push({ability:source.ability,move:move.id,type:move.type,target_ability:x.p.ability,protected:x.protected,absorbed:value===null,blocked:value===null||value===false||value===this.NOT_FAIL,weather:rawWeather,suppressed});}
    if(converterStats)conversions.stats.push({...converterStats,after:result});
    for(const x of fireTargets){const value=Array.isArray(result)?result[x.index]:result;fireRod.hit.push({ability:x.mon.ability,type:move.type,category:move.category,move:move.id,source_side:source.side.n,target_side:x.mon.side.n,target_roster:roster(x.mon),protected:x.protected,blocked:value===null||value===false||value===this.NOT_FAIL,spa_before:x.spa,spa_after:x.mon.boosts.spa,charged_before:x.charged,charged_after:Boolean(x.mon.volatiles.flashfire),accuracy_always:move.accuracy===true});}
    if(event==='RedirectTarget')fireRod.redirect.push({type:move.type,move:move.id,initial:{side:rest[0].side.n,roster:roster(rest[0])},selected:{side:result.side.n,roster:roster(result)},candidates:rodCandidates,seed_unchanged:redirectSeed===this.prng.getSeed(),source_side:source.side.n});
    if(firePower)fireRod.power.push({...firePower,after:result,weather:rawWeather,suppressed});
    for (const x of absorberTargets) {
      const value = Array.isArray(result) ? result[x.index] : result;
      absorption.push({ability:x.mon.ability,type:move.type,category:move.category,move:move.id,accuracy:move.accuracy,protected:x.protected,ally:x.ally,full:x.hp===x.maxhp,maxhp:x.maxhp,missing:x.maxhp-x.hp,healed:x.mon.hp-x.hp,before:x.boosts,after:{...x.mon.boosts},blocked:value===null||value===false||value===this.NOT_FAIL});
    }
    for (const probe of dryTargets) {
      const value = Array.isArray(result) ? result[probe.index] : result;
      drySkinInteractions.water.push({move: move.id, accuracy: move.accuracy, full_hp: probe.hp === probe.maxhp, protected: probe.protected, ally: probe.ally, self: probe.self, blocked: value === null || value === false || value === this.NOT_FAIL, healed: probe.mon.hp - probe.hp});
    }
    if (event === 'BasePower' && source?.ability === 'dryskin') drySkinInteractions.power.push({type: move.type, before: rest[0], after: result});
    if (event === 'Heal' && move?.id === 'dryskin') drySkinInteractions.healing.push({weather: rawWeather, suppressed, weather_callback: eventStack.includes('Weather'), amount: rest[0]});
    if (event === 'Damage' && move?.id === 'dryskin') drySkinInteractions.damage.push({weather: rawWeather, suppressed, weather_callback: eventStack.includes('Weather'), amount: rest[0]});
    if (event === 'BasePower' && target?.ability === 'sandforce') weatherAbilityInteractions.force.push({weather: rawWeather, suppressed, type: move.type, before: rest[0], after: result});
    if (event === 'ModifyAccuracy' && ['sandveil', 'snowcloak'].includes(target?.ability)) weatherAbilityInteractions.accuracy.push({weather: rawWeather, suppressed, ability: target.ability, before: rest[0], after: result, attacker_stage: source.boosts.accuracy, defender_stage: target.boosts.evasion});
    if (event === 'Immunity' && rest[0] === 'sandstorm' && ['sandveil', 'overcoat'].includes(target?.ability)) weatherAbilityInteractions.sand_immunity.push({ability: target.ability, blocked: result === false});
    if (event === 'Damage' && target?.ability === 'hydration' && ['brn', 'psn', 'tox'].includes(move?.id)) weatherAbilityInteractions.hydration_damage.push({weather: rawWeather, suppressed, effective: this.field.effectiveWeather(), status: move.id, amount: rest[0]});
    if (event === 'Damage' && move?.id === 'sandstorm') weatherAbilityInteractions.sand_damage.push({ability: target?.ability, amount: rest[0]});
    if (event === 'TryHit' && move?.flags?.powder) {
      for (const [index, mon] of (Array.isArray(target) ? target : [target]).entries()) {
        if (mon?.ability !== 'overcoat') continue;
        const value = Array.isArray(result) ? result[index] : result;
        weatherAbilityInteractions.powder.push({move: move.id, suppressed, blocked: value === null || value === false});
      }
    }
    if (rawWeather && event === 'WeatherModifyDamage' && move) suppressionInteractions.power.push({weather: rawWeather, suppressed, type: move.type, before: rest[0], after: result});
    if (rawWeather && event === 'ModifySpe' && target?.ability) suppressionInteractions.speed.push({weather: rawWeather, suppressed, ability: target.ability, before: rest[0], after: result});
    if (rawWeather && event === 'BasePower' && move?.id === 'weatherball') suppressionInteractions.weather_ball.push({weather: rawWeather, suppressed, type: move.type, power: rest[0]});
    if (rawWeather && event === 'Heal' && move?.effectType === 'Ability') suppressionInteractions.healing.push({weather: rawWeather, suppressed, ability: move.id, amount: rest[0]});
    if (rawWeather && event === 'Damage' && move?.id === 'sandstorm') suppressionInteractions.sand.push({weather: rawWeather, suppressed, amount: rest[0]});
    if (rawWeather && event === 'Damage' && move?.id === 'solarpower') suppressionInteractions.solar.push({weather: rawWeather, suppressed, amount: rest[0]});
    if (terrain && event === 'BasePower' && move && target?.isGrounded && source?.isGrounded) {
      terrainInteractions.power.push({terrain, move: move.id, type: move.type, attacker_grounded: !!target.isGrounded(), defender_grounded: !!source.isGrounded(), before: rest[0], after: result});
    }
    if (terrain && event === 'SetStatus' && target?.isGrounded) terrainInteractions.status.push({terrain, grounded: !!target.isGrounded(), status: move?.id, blocked: result === false});
    if (terrain === 'psychicterrain' && event === 'TryHit' && move?.priority > 0) {
      for (const [index, mon] of (Array.isArray(target) ? target : [target]).entries()) {
        if (!mon?.isGrounded) continue;
        const value = Array.isArray(result) ? result[index] : result;
        terrainInteractions.priority.push({grounded: !!mon.isGrounded(), ally: mon.isAlly(source), blocked: value === null || value === false});
      }
    }
    if (event==='DamagingHit') for (const x of staticProbes.filter(x=>x.actor && x.actor===source)) {
      x.status_after=x.actor.status; x.item_after=x.actor.item;
      x.protocol=this.log.slice(x.log_start); delete x.actor;
    }
    eventContexts.pop();
    eventStack.pop();
    return result;
  };
  const fixture = {name: ['direct_damage', 'spread_damage', 'protect_stall', 'voluntary_switch', 'speed_ties', 'retarget', 'struggle_recoil', 'mega_both_sides', 'mega_switch_persistence', 'mega_after_damage', 'burn_paralysis_boost_heal', 'sleep_boost_heal', 'spread_secondary_speed', 'freeze_flinch', 'self_stat_drops', 'poison_powder', 'toxic_poison_user', 'intimidate_defiant_competitive', 'intimidate_immunity', 'mega_intimidate_hugepower', 'speedboost_switch', 'switchout_recovery', 'technician_sharpness_megalauncher', 'purepower_ironfist_strongjaw', 'thickfat_filter_toughclaws_multiscale', 'owntempo_intimidate', 'oblivious_intimidate', 'items_sash_helmet_orb', 'modifier_handler_speed_tie', 'recoil_rockhead_reckless', 'drain_ooze_root', 'spread_drain_ooze', 'drain_contact_ooze_root', 'rockhead_struggle', 'side_effects_basic', 'screens_tailwind', 'lightclay_duration', 'screen_critical', 'tailwind_scarf', 'side_effects_switch', 'brickbreak_screens', 'psychicfangs_screens', 'infiltrator_screens', 'rain_duration_speed_ball', 'sun_duration_speed_ball', 'sand_duration_speed_defense', 'snow_duration_speed_defense', 'weather_replacement_switch', 'weather_speed_scarf_tailwind', 'trickroom_expiration_priority', 'trickroom_simultaneous', 'trickroom_recast', 'trickroom_tailwind_scarf_paralysis', 'trickroom_switch_persistence', 'electric_terrain_grounded_sleep', 'grassy_terrain_damage_healing', 'psychic_terrain_grounded_priority', 'misty_terrain_status_dragon', 'terrain_replacement', 'terrain_weather_room_coexistence', 'psychic_terrain_ally_priority', 'rain_dish_weather_healing', 'ice_body_weather_healing', 'solar_power_weather_damage', 'solar_power_thickfat_speed_tie', 'solar_power_inactive_rain', 'cloudnine_rain_expiration', 'cloudnine_sun_expiration', 'cloudnine_sand_expiration', 'cloudnine_snow_expiration', 'cloudnine_rain_switch', 'cloudnine_sun_switch', 'cloudnine_sand_switch', 'cloudnine_snow_switch', 'cloudnine_faint_snow', 'sand_force_types', 'sand_veil_stages', 'snow_cloak_stages', 'overcoat_sand_powder', 'hydration_rain_order', 'sand_force_suppression', 'sand_force_no_weather', 'sand_veil_suppression', 'hydration_suppression', 'hydration_expiry_status', 'sand_force_mega_garchomp', 'snow_cloak_suppression', 'overcoat_powder_suppression', 'dryskin_water_spread_ally', 'dryskin_fire_power', 'dryskin_rain_healing', 'dryskin_sun_damage', 'dryskin_rain_suppression', 'dryskin_sun_suppression', 'dryskin_protect_fullhp_water_accuracy','waterabsorb','voltabsorb','eartheater','sapsipper','motordrive','waterabsorb_status','voltabsorb_status','eartheater_stages','sapsipper_side_ties','motordrive_status','static_lum','static_helmet','static_electric','static_misty','static_faint_helmet','flashfire_categories','flashfire_spread_before','flashfire_spread_after','flashfire_switch','flashfire_mega_loss','lightningrod_ally_ground_cap','lightningrod_tie_reentry','lightningrod_tailwind','lightningrod_mega_gain','lightningrod_mega_loss','flashfire_thickfat','flashfire_sun','flashfire_rain','flashfire_cloudnine_sun','pixilate_sylveon','refrigerate_aurorus','pixilate_mega_altaria','pixilate_mega_gardevoir','aerilate_mega_pinsir','aerilate_mega_salamence','refrigerate_mega_glalie','dragonize_mega_feraligatr','liquidvoice_primarina','convert_weatherball_clear','convert_weatherball_rain','convert_weatherball_sun','convert_weatherball_cloudnine','convert_altaria_weather_release','liquidvoice_absorption','pixilate_psychic_priority','aerilate_switch_persistence','refrigerate_thickfat','synchronize_success','synchronize_fire_immunity','synchronize_lum','synchronize_pair','synchronize_sleep_freeze','hypercutter_before_mega'][trial] + `_${seedWord}`, seed,
    teams: session.teams.map((team, side) => ({id: `fixture-${trial}-${side}`, members: team.map(s => ({species: ids.species[toID(s.species)],
      ability: ids.abilities[toID(s.ability)], item: ids.items[toID(s.item)] ?? 0, nature: ids.natures[toID(s.nature)], gender: s.gender || '', level: 50,
      moves: s.moves.map(m => ids.moves[toID(m)]), points: stats.map(k => s.evs[k]), ivs: stats.map(k => s.ivs[k])}))})),
    initial: compact(session), steps: []};
  while (!session.battle.ended && fixture.steps.length < 300) {
    for (let side = 0; side < 2; side++) {
      if (session.battle.ended) break;
      const s = session.battle.sides[side];
      if (s.activeRequest?.wait || s.isChoiceDone()) continue;
      const choice = choices(session, side, trial);
      const result = session.choose(side ? 'p2' : 'p1', choice.command);
      if (!result.accepted) throw new Error(JSON.stringify({trial, side, choice, messages: result.messages}));
      fixture.steps.push({side: side ? 'P2' : 'P1', ...choice, expected: compact(session)});
    }
  }
  if (!session.battle.ended) throw new Error(`Fixture ${trial} did not naturally complete`);
  let logTurn = 0;
  const terrainResidualLog = [];
  for (const line of session.battle.log) {
    if (line.startsWith('|turn|')) logTurn = Number(line.split('|')[2]);
    if (line.startsWith('|-heal|') && (line.includes('[from] Grassy Terrain') || line.includes('[from] item: Leftovers'))) terrainResidualLog.push({turn: logTurn, mon: line.split('|')[2], kind: line.includes('[from] Grassy Terrain') ? 'grass' : 'leftovers'});
    if (line.startsWith('|-fieldend|') && line.includes('Grassy Terrain')) terrainResidualLog.push({turn: logTurn, kind: 'end'});
  }
  fixture.coverage = {
    dependency,
    conversion:conversions,
    fire_rod:fireRod,
    static:staticProbes,
    absorption, sap_side_ties:sapSideTies,
    dryskin: drySkinInteractions,
    weather_abilities: weatherAbilityInteractions,
    weather_evasion_misses: session.battle.log.filter(x => x.startsWith('|-miss|') && x.includes('p1a: s0m0')).length,
    hydration_activations: session.battle.log.filter(x => x.startsWith('|-activate|') && x.includes('ability: Hydration')).length,
    hydration_order: (() => {
      const lines = session.battle.log;
      return lines.some((line, i) => line.startsWith('|-activate|') && line.includes('ability: Hydration') && (lines.slice(i + 1).find(x => x.startsWith('|-heal|') || x.startsWith('|upkeep')) || '').includes('[from] item: Leftovers'));
    })(),
    hydration_cured_in_rain: fixture.steps.some((step, i) => i && step.expected.climate.effective === 'raindance' && step.expected.sides[0].pokemon.some(p => p.ability === ids.abilities.hydration && p.status === 0) && session.battle.log.some(x => x.startsWith('|-curestatus|'))),
    hydration_expiry_retained: fixture.steps.some(step => !step.expected.climate.raw && step.expected.sides[0].pokemon.some(p => p.ability === ids.abilities.hydration && p.status !== 0 && !p.fainted)),
    hydration_suppressed_retained: fixture.steps.some(step => step.expected.climate.raw === 'raindance' && step.expected.climate.suppressed && step.expected.sides[0].pokemon.some(p => p.ability === ids.abilities.hydration && p.status !== 0 && !p.fainted)),
    suppression: suppressionInteractions,
    solar_power: solarPowerCoverage,
    weather_ability_heals: [...new Set(session.battle.log.filter(x => x.startsWith('|-heal|')).flatMap(x => [...x.matchAll(/\[from\] ability: ([^|]+)/g)].map(m => toID(m[1]))))].sort(),
    weather_ability_damage: [...new Set(session.battle.log.filter(x => x.startsWith('|-damage|')).flatMap(x => [...x.matchAll(/\[from\] ability: ([^|]+)/g)].map(m => toID(m[1]))))].sort(),
    grassy_final_heal: terrainResidualLog.some(x => x.kind === 'end' && terrainResidualLog.some(y => y.kind === 'grass' && y.turn === x.turn)),
    grassy_leftovers_order: terrainResidualLog.some((x, i) => x.kind === 'leftovers' && terrainResidualLog.slice(0, i).some(y => y.kind === 'grass' && y.turn === x.turn && y.mon === x.mon)),
    terrain_interactions: terrainInteractions,
    terrain_starts: [...new Set(session.battle.log.filter(x => x.startsWith('|-fieldstart|') && x.includes('Terrain')).map(x => toID(x.split('|')[2].replace('move: ', ''))))].sort(),
    terrain_ends: [...new Set(session.battle.log.filter(x => x.startsWith('|-fieldend|') && x.includes('Terrain')).map(x => toID(x.split('|')[2].replace('move: ', ''))))].sort(),
    terrain_expired: fixture.steps.flatMap((step, i) => {
      const prior = i ? fixture.steps[i - 1].expected.field : [];
      return prior.filter(x => x[1] === 1 && ['electricterrain', 'grassyterrain', 'psychicterrain', 'mistyterrain'].some(t => ids.conditions[t] === x[0]) && !step.expected.field.some(y => y[0] === x[0])).map(x => data.tables.conditions.find(row => row.numeric_id === x[0]).id);
    }),
    grassy_heals: session.battle.log.filter(x => x.startsWith('|-heal|') && x.includes('[from] Grassy Terrain')).length,
    room_expired: fixture.steps.some((step, i) => {
      const prior = (i ? fixture.steps[i - 1].expected.field : []).find(x => x[0] === ids.conditions.trickroom);
      return prior?.[1] === 1 && !step.expected.field.some(x => x[0] === ids.conditions.trickroom);
    }),
    room_recast: fixture.steps.some((step, i) => {
      const prior = (i ? fixture.steps[i - 1].expected.field : []).find(x => x[0] === ids.conditions.trickroom);
      return prior?.[1] > 1 && !step.expected.field.some(x => x[0] === ids.conditions.trickroom);
    }),
    room_switch_persistence: fixture.steps.some(step => step.command.includes('switch') && step.expected.field.some(x => x[0] === ids.conditions.trickroom)),
    room_starts: session.battle.log.filter(x => x.startsWith('|-fieldstart|') && x.includes('Trick Room')).length,
    room_ends: session.battle.log.filter(x => x.startsWith('|-fieldend|') && x.includes('Trick Room')).length,
    weather_starts: [...new Set(session.battle.log.filter(x => x.startsWith('|-weather|') && !x.includes('[upkeep]')).map(x => toID(x.split('|')[2])))].sort(),
    weather_expired: fixture.steps.flatMap((step, i) => { const previous = i ? fixture.steps[i - 1].expected.field : fixture.initial.field; return previous.length && !step.expected.field.length ? [data.tables.conditions.find(row => row.numeric_id === previous[0][0]).id] : []; }),
    weather_ball: [...weatherBallCoverage].sort(),
    weather_ends: session.battle.log.filter(x => x === '|-weather|none').length,
    sand_hits: session.battle.log.filter(x => x.startsWith('|-damage|') && x.includes('[from] Sandstorm')).length,
    statuses: [...new Set(session.battle.log.filter(x => x.startsWith('|-status|')).map(x => x.split('|')[3]))].sort(),
    flinches: session.battle.log.filter(x => x.startsWith('|cant|') && x.endsWith('|flinch')).length,
    visible_item_effects: [...new Set(session.battle.log.flatMap(x => [...x.matchAll(/\[from\] item: ([^|]+)/g)].map(m => toID(m[1]))))].sort(),
    consumed_items: [...new Set(session.battle.log.filter(x => x.startsWith('|-enditem|')).map(x => toID(x.split('|')[3])))].sort(),
    screens: screenCoverage,
    side_starts: [...new Set(session.battle.log.filter(x => x.startsWith('|-sidestart|')).map(x => toID(x.split('|')[3].replace('move: ', ''))))].sort(),
    side_ends: [...new Set(session.battle.log.filter(x => x.startsWith('|-sideend|')).map(x => toID(x.split('|')[3].replace('move: ', ''))))].sort(),
    drain_heals: session.battle.log.filter(x => x.startsWith('|-heal|') && x.includes('[from] drain')).length,
    recoil_hits: session.battle.log.filter(x => x.startsWith('|-damage|') && x.includes('[from] Recoil')).length,
    ooze_hits: session.battle.log.filter(x => x.startsWith('|-damage|') && x.includes('[from] ability: Liquid Ooze')).length,
    megas: session.battle.log.filter(x => x.startsWith('|-mega|')).length,
  };
  fixtures.push(fixture); session.destroy();
}
const syncAll=fixtures.flatMap(f=>f.coverage.dependency.sync),cutterAll=fixtures.flatMap(f=>f.coverage.dependency.hypercutter);
if(!syncAll.some(x=>x.source_status_before==='par'&&x.source_status_after==='par'&&x.protocol.some(l=>l.startsWith('|-fail|'))))throw new Error('Missing Synchronize already-status reflection rejection');
if(!fixtures.some(f=>f.coverage.dependency.body_slam_paralyses>0))throw new Error('Missing Body Slam actual paralysis secondary');
if(!syncAll.some(x=>x.source_status_before===''&&x.source_status_after==='par'))throw new Error('Missing Synchronize successful reflection');
if(!syncAll.some(x=>x.status==='brn'&&x.source_types?.includes('Fire')&&x.source_status_before===x.source_status_after&&x.protocol.some(l=>l.includes('ability: Synchronize'))))throw new Error('Missing Synchronize failed Fire reflection');
if(!fixtures.some(f=>f.name.startsWith('synchronize_lum')&&f.coverage.dependency.sync.some(x=>x.source_status_after==='par')&&f.steps.some(s=>s.expected.sides[0].pokemon.some(p=>p.roster===0&&p.previous_item===ids.items.lumberry&&p.status===0))))throw new Error('Missing Synchronize reflection and Lum cure');
if(!fixtures.some(f=>f.name.startsWith('synchronize_pair')&&f.coverage.dependency.sync.some(x=>x.effect==='synchronize')&&f.coverage.dependency.sync.length<12))throw new Error('Missing bounded two-Synchronize reflection');
for(const status of ['slp','frz'])if(!syncAll.some(x=>x.status===status&&x.source_status_before===x.source_status_after&&!x.protocol.some(l=>l.includes('ability: Synchronize'))))throw new Error(`Missing Synchronize ${status} noop`);
if(!cutterAll.some(x=>x.enemy&&x.before.atk<0&&(x.after.atk??0)===0))throw new Error('Missing Hyper Cutter Attack block');
if(!cutterAll.some(x=>x.enemy&&x.before.def<0&&x.after.def===x.before.def&&x.turn<4))throw new Error('Missing Hyper Cutter Defense allowed before Mega');
const conversionTypes=fixtures.flatMap(f=>f.coverage.conversion.type),conversionPower=fixtures.flatMap(f=>f.coverage.conversion.power),conversionHits=fixtures.flatMap(f=>f.coverage.conversion.hit);
for(const [ability,type]of [['pixilate','Fairy'],['aerilate','Flying'],['refrigerate','Ice'],['dragonize','Dragon']]){
  if(!conversionTypes.some(x=>x.ability===ability&&x.before==='Normal'&&x.after===type&&x.marker===ability))throw new Error(`Missing actual conversion ${ability}`);
  if(!conversionPower.some(x=>x.ability===ability&&x.marker===ability&&x.after===Math.floor((x.before*4915+2047)/4096)))throw new Error(`Missing conversion boost ${ability}`);
  if(!conversionPower.some(x=>x.ability===ability&&x.marker===null&&x.before===x.after))throw new Error(`Missing conversion noop ${ability}`);
}
if(!conversionTypes.some(x=>x.ability==='liquidvoice'&&x.move==='hypervoice'&&x.after==='Water'&&x.marker===null)||!conversionPower.some(x=>x.ability==='liquidvoice'&&x.move==='hypervoice'&&x.after===x.before))throw new Error('Missing Liquid Voice no bonus conversion');
if(!conversionTypes.some(x=>x.category==='Status'&&x.before==='Normal'&&x.after!=='Normal'&&x.marker===x.ability))throw new Error('Missing status conversion');
for(const prefix of ['pixilate_mega_altaria','pixilate_mega_gardevoir','aerilate_mega_pinsir','aerilate_mega_salamence','refrigerate_mega_glalie','dragonize_mega_feraligatr'])if(!fixtures.some(f=>f.name.startsWith(prefix)&&f.coverage.megas>0&&f.coverage.conversion.type.some(x=>x.turn===2&&x.marker===x.ability)))throw new Error(`Missing Mega acquisition ${prefix}`);
for(const [prefix,type,weather,suppressed]of [['convert_weatherball_clear','Normal','',false],['convert_weatherball_rain','Water','raindance',false],['convert_weatherball_sun','Fire','sunnyday',false],['convert_weatherball_cloudnine','Normal','raindance',true]])if(!fixtures.some(f=>f.name.startsWith(prefix)&&f.coverage.conversion.type.some(x=>x.move==='weatherball'&&x.after===type&&x.marker===null&&x.weather===weather&&x.suppressed===suppressed)))throw new Error(`Missing Weather Ball exclusion ${prefix}`);
if(!fixtures.some(f=>f.name.startsWith('convert_altaria_weather_release')&&f.coverage.conversion.type.some(x=>x.move==='weatherball'&&x.after==='Water'&&x.marker===null&&!x.suppressed)&&f.steps.some(s=>s.expected.climate.suppressed)))throw new Error('Missing Altaria Mega weather release');
for(const ability of ['dryskin','waterabsorb'])if(!conversionHits.some(x=>x.ability==='liquidvoice'&&x.move==='hypervoice'&&x.type==='Water'&&x.target_ability===ability&&!x.protected&&x.absorbed))throw new Error(`Missing Liquid Voice absorption ${ability}`);
if(!fixtures.some(f=>f.name.startsWith('pixilate_psychic_priority')&&f.coverage.conversion.hit.some(x=>x.move==='quickattack'&&x.type==='Fairy'&&!x.protected&&x.blocked)&&f.steps.some(s=>s.expected.field.some(x=>x[0]===ids.conditions.psychicterrain))))throw new Error('Missing converted priority terrain block');
if(!fixtures.some(f=>f.name.startsWith('aerilate_switch_persistence')&&f.coverage.conversion.type.some(x=>x.turn>=6&&x.marker==='aerilate')))throw new Error('Missing conversion Mega switch persistence');
if(!fixtures.some(f=>f.name.startsWith('refrigerate_thickfat')&&f.coverage.conversion.stats.some(x=>x.thickfat&&x.type==='Ice'&&x.after===Math.floor((x.before*2048+2047)/4096))))throw new Error('Missing converted Ice Thick Fat');
const fireRodAll=fixtures.flatMap(f=>f.coverage.fire_rod.hit),redirectAll=fixtures.flatMap(f=>f.coverage.fire_rod.redirect),firePowerAll=fixtures.flatMap(f=>f.coverage.fire_rod.power);
for(const category of ['Physical','Special','Status'])if(!fireRodAll.some(x=>x.ability==='flashfire'&&x.type==='Fire'&&x.category===category&&!x.protected&&x.blocked&&x.charged_after&&x.accuracy_always))throw new Error(`Missing Flash Fire ${category}`);
if(!fireRodAll.some(x=>x.ability==='flashfire'&&x.charged_before&&x.type==='Fire'&&!x.protected&&x.blocked))throw new Error('Missing Flash Fire repeated activation');
for(const event of ['ModifyAtk','ModifySpA'])if(!firePowerAll.some(x=>x.event===event&&x.type==='Fire'&&!x.thickfat&&x.after===Math.floor((x.before*6144+2047)/4096)))throw new Error(`Missing Flash Fire modifier ${event}`);
if(!firePowerAll.some(x=>x.type!=='Fire'&&x.before===x.after))throw new Error('Missing Flash Fire nonFire noop');
if(!firePowerAll.some(x=>x.type==='Fire'&&x.thickfat&&x.after===Math.floor((x.before*3072+2047)/4096)))throw new Error('Missing Flash Fire Thick Fat chain');
for(const prefix of ['flashfire_spread_before','flashfire_spread_after'])if(!fixtures.some(f=>f.name.startsWith(prefix)&&f.coverage.fire_rod.accuracy.some(x=>x.always&&x.seed_unchanged&&x.targets.some(p=>p.ability!=='flashfire'))&&f.coverage.fire_rod.accuracy.some(x=>!x.always&&!x.seed_unchanged&&x.results.some(result=>result===false))))throw new Error(`Missing Flash Fire spread accuracy ${prefix}`);
for(const category of ['Physical','Special','Status'])if(category!=='Physical'&&!fireRodAll.some(x=>x.ability==='lightningrod'&&x.type==='Electric'&&x.category===category&&!x.protected&&x.blocked&&x.spa_after===Math.min(6,x.spa_before+1)))throw new Error(`Missing Lightning Rod ${category}`);
if(!fireRodAll.some(x=>x.ability==='lightningrod'&&x.spa_before===6&&x.blocked))throw new Error('Missing Lightning Rod cap');
if(!fireRodAll.some(x=>x.ability==='lightningrod'&&x.protected&&x.blocked&&x.spa_before===x.spa_after))throw new Error('Missing Lightning Rod Protect');
if(!fireRodAll.some(x=>x.ability==='lightningrod'&&x.source_side===x.target_side&&x.blocked))throw new Error('Missing Lightning Rod ally');
for(const x of redirectAll.filter(x=>x.type==='Electric')){if(!x.seed_unchanged)throw new Error('Redirect consumed RNG');const first=x.candidates.filter(x=>x.valid).sort((a,b)=>b.speed-a.speed||a.order-b.order)[0];if(first&&(first.side!==x.selected.side||first.roster!==x.selected.roster))throw new Error('Redirect selection differs from speed/activation order');}
if(!fixtures.some(f=>f.name.startsWith('lightningrod_tie_reentry')&&f.coverage.fire_rod.redirect.some(x=>x.type==='Electric'&&x.candidates.filter(p=>p.valid).length===2&&x.candidates[0].speed===x.candidates[1].speed&&x.selected.roster===0)&&f.coverage.fire_rod.redirect.some(x=>x.type==='Electric'&&x.candidates.filter(p=>p.valid).length===2&&x.candidates[0].speed===x.candidates[1].speed&&x.selected.roster===1)))throw new Error('Missing Lightning Rod tie/reentry reversal');
if(!fixtures.some(f=>f.name.startsWith('lightningrod_tailwind')&&f.coverage.fire_rod.redirect.some(x=>x.type==='Electric'&&x.selected.side===1)&&f.coverage.fire_rod.redirect.some(x=>x.type==='Electric'&&x.selected.side===0)))throw new Error('Missing Lightning Rod Tailwind reversal');
if(!fixtures.some(f=>f.name.startsWith('flashfire_switch')&&f.coverage.fire_rod.ends.some(x=>x.condition&&x.hp>0)&&f.coverage.fire_rod.hit.filter(x=>x.ability==='flashfire'&&!x.charged_before&&x.charged_after).length>=2))throw new Error('Missing Flash Fire switch/reentry reactivation');
if(!fixtures.some(f=>f.coverage.fire_rod.ends.some(x=>!x.condition&&x.hp===0&&x.volatile_before)&&!f.coverage.fire_rod.ends.some(x=>x.condition&&x.hp===0)))throw new Error('Missing Flash Fire faint end rejection');
if(!fixtures.some(f=>f.name.startsWith('flashfire_mega_loss')&&f.coverage.megas>0&&f.coverage.fire_rod.ends.some(x=>x.condition&&x.hp>0)&&f.steps.some(s=>s.expected.sides[0].pokemon.some(p=>p.roster===0&&p.ability===ids.abilities.solarpower&&!p.volatiles.includes('flashfire')))))throw new Error('Missing Flash Fire Mega loss');
if(!fixtures.some(f=>f.name.startsWith('lightningrod_mega_gain')&&f.coverage.megas>0&&f.coverage.fire_rod.hit.some(x=>x.ability==='lightningrod'&&x.blocked&&x.spa_after>x.spa_before)))throw new Error('Missing Mega Lightning Rod gain');
if(!fixtures.some(f=>f.name.startsWith('lightningrod_mega_loss')&&f.coverage.megas>0&&f.coverage.fire_rod.redirect.some(x=>x.type==='Electric'&&x.selected.side===0)&&f.coverage.fire_rod.redirect.some(x=>x.type==='Electric'&&x.selected.side===1&&!x.candidates.some(p=>p.valid))))throw new Error('Missing Mega Lightning Rod loss');
if(!fixtures.some(f=>f.coverage.fire_rod.spread_damage.length>0&&f.coverage.fire_rod.hit.some(x=>x.move==='discharge'&&x.ability==='lightningrod'&&x.blocked)&&!f.coverage.fire_rod.redirect.some(x=>x.move==='discharge')))throw new Error('Missing Lightning Rod spread independent hits');
if(!redirectAll.some(x=>x.type!=='Electric'&&x.candidates.some(p=>p.valid)&&JSON.stringify(x.initial)===JSON.stringify(x.selected)))throw new Error('Missing nonElectric redirect noop');
for(const [prefix,weather,suppressed] of [['flashfire_sun','sunnyday',false],['flashfire_rain','raindance',false],['flashfire_cloudnine_sun','sunnyday',true]])if(!fixtures.some(f=>f.name.startsWith(prefix)&&f.coverage.fire_rod.power.some(x=>x.type==='Fire'&&x.weather===weather&&x.suppressed===suppressed&&x.after>x.before)))throw new Error(`Missing Flash Fire weather independence ${prefix}`);
const staticAll=fixtures.flatMap(f=>f.coverage.static);
if (!staticAll.some(x=>x.holder_hp===0)) throw new Error('Missing Static fainting-holder callback');
if (!staticAll.some(x=>x.holder_item==='rockyhelmet'&&x.actor_hp_before_handlers>x.actor_hp&&x.preceding_protocol.some(l=>l.startsWith('|-damage|')&&l.includes('Rocky Helmet')))) throw new Error('Missing Rocky Helmet before Static chance');
if (!staticAll.some(x=>x.chance) || !staticAll.some(x=>!x.chance)) throw new Error('Missing Static chance success/failure');
if (!staticAll.some(x=>x.chance&&x.status_before===''&&x.status_after==='par')) throw new Error('Missing Static status success');
for (const kind of ['electric','misty','status','zero_hp']) {
  if (!staticAll.some(x=>x.chance&&(kind==='electric'?x.electric:kind==='misty'?x.misty:kind==='status'?x.status_before==='par':x.actor_hp===0)&&x.status_before===x.status_after)) throw new Error(`Missing Static noop ${kind}`);
}
if (!fixtures.some(f=>f.name.startsWith('static_lum')&&f.coverage.static.some(x=>x.chance&&x.item_before==='lumberry'&&x.item_after===''&&x.status_after===''&&x.protocol.some(l=>l.startsWith('|-status|')&&l.includes('|par'))&&x.protocol.some(l=>l.startsWith('|-enditem|')&&l.includes('Lum Berry'))&&x.protocol.some(l=>l.startsWith('|-curestatus|'))))) throw new Error('Missing Static Lum consumption/cure');
if (!fixtures.some(f=>f.coverage.sap_side_ties>0)) throw new Error('Missing Sap Sipper side event ties');
for (const [ability,type] of [['waterabsorb','Water'],['voltabsorb','Electric'],['eartheater','Ground'],['sapsipper','Grass'],['motordrive','Electric']]) {
  const probes = fixtures.flatMap(f=>f.coverage.absorption).filter(x=>x.ability===ability&&x.type===type&&!x.protected);
  if (!probes.some(x=>x.blocked&&x.category==='Physical'||x.blocked&&x.category==='Special')) throw new Error(`Missing absorption damage ${ability}`);
  if (['waterabsorb','voltabsorb','eartheater'].includes(ability)) {
    if (!probes.some(x=>x.full&&x.blocked&&x.healed===0)) throw new Error(`Missing full HP ${ability}`);
    if (!probes.some(x=>x.blocked&&x.healed===Math.min(x.missing,Math.max(1,Math.floor(x.maxhp/4)))&&x.healed>0)) throw new Error(`Missing exact heal ${ability}`);
  } else {
    const stat=ability==='sapsipper'?'atk':'spe';
    if (!probes.some(x=>x.blocked&&x.after[stat]===x.before[stat]+1)) throw new Error(`Missing boost ${ability}`);
    if (!probes.some(x=>x.blocked&&x.before[stat]===6&&x.after[stat]===6)) throw new Error(`Missing boost cap ${ability}`);
  }
}
for (const [ability,type] of [['waterabsorb','Water'],['voltabsorb','Electric'],['eartheater','Ground'],['sapsipper','Grass'],['motordrive','Electric']]) {
  const probes=fixtures.flatMap(f=>f.coverage.absorption).filter(x=>x.ability===ability&&x.type===type);
  if (!probes.some(x=>x.protected&&x.blocked&&x.healed===0&&JSON.stringify(x.before)===JSON.stringify(x.after))) throw new Error(`Missing protected absorption ${ability}`);
  if (['voltabsorb','sapsipper','motordrive'].includes(ability)&&!probes.some(x=>x.category==='Status'&&!x.protected&&x.blocked)) throw new Error(`Missing status absorption ${ability}`);
  if (['waterabsorb','voltabsorb','eartheater','sapsipper','motordrive'].includes(ability)&&!probes.some(x=>x.ally&&!x.protected&&x.blocked)) throw new Error(`Missing ally absorption ${ability}`);
  if (ability!=='eartheater'&&!probes.some(x=>typeof x.accuracy==='number'&&x.accuracy<100&&!x.protected&&x.blocked)) throw new Error(`Missing low accuracy absorption ${ability}`);
}
for (const kind of ['full', 'damaged', 'ally', 'spread', 'low_accuracy', 'protected']) {
  if (!fixtures.some(f => f.coverage.dryskin.water.some(x => kind === 'full' ? x.full_hp && !x.protected && x.blocked && x.healed === 0 : kind === 'damaged' ? !x.full_hp && x.blocked && x.healed > 0 : kind === 'ally' ? x.ally && !x.self && x.blocked : kind === 'spread' ? x.move === 'surf' && x.blocked : kind === 'low_accuracy' ? x.accuracy < 100 && x.full_hp && !x.protected && x.blocked : x.protected && x.blocked && x.healed === 0))) throw new Error(`Missing Dry Skin Water interaction ${kind}`);
}
if (!fixtures.some(f => f.coverage.dryskin.power.some(x => x.type === 'Fire' && x.after === Math.floor((x.before * 5120 + 2047) / 4096)))) throw new Error('Missing Dry Skin exact Fire power');
if (!fixtures.some(f => f.coverage.dryskin.power.some(x => x.type !== 'Fire' && x.after === x.before))) throw new Error('Missing Dry Skin nonFire noop');
if (!fixtures.some(f => f.coverage.dryskin.healing.some(x => x.weather === 'raindance' && x.weather_callback && !x.suppressed && x.amount > 0))) throw new Error('Missing Dry Skin rain healing');
if (!fixtures.some(f => f.coverage.dryskin.damage.some(x => x.weather === 'sunnyday' && x.weather_callback && !x.suppressed && x.amount > 0))) throw new Error('Missing Dry Skin sun damage');
for (const [name, channel] of [['dryskin_rain_suppression', 'healing'], ['dryskin_sun_suppression', 'damage']]) {
  if (!fixtures.some(f => f.name.startsWith(name) && f.steps.some(s => s.expected.climate.suppressed && s.expected.climate.raw) && f.coverage.dryskin[channel].some(x => x.weather_callback && !x.suppressed && x.amount > 0))) throw new Error(`Missing Dry Skin weather resume ${name}`);
}
if (fixtures.some(f => [...f.coverage.dryskin.healing, ...f.coverage.dryskin.damage].some(x => x.weather_callback && x.suppressed && x.amount > 0))) throw new Error('Suppressed Dry Skin weather callback leaked');
if (!fixtures.some(f => f.name.startsWith('sand_force_mega') && f.coverage.megas > 0 && f.coverage.weather_abilities.force.some(x => x.after > x.before))) throw new Error('Missing Mega Sand Force activation');
for (const type of ['Rock', 'Ground', 'Steel']) {
  if (!fixtures.some(f => f.coverage.weather_abilities.force.some(x => x.weather === 'sandstorm' && !x.suppressed && x.type === type && x.after > x.before))) throw new Error(`Missing Sand Force ${type}`);
}
for (const kind of ['other_type', 'no_weather', 'suppressed']) {
  if (!fixtures.some(f => f.coverage.weather_abilities.force.some(x => x.after === x.before && (kind === 'other_type' ? x.type === 'Bug' : kind === 'no_weather' ? !x.weather : x.weather === 'sandstorm' && x.suppressed)))) throw new Error(`Missing Sand Force noop ${kind}`);
}
for (const ability of ['sandveil', 'snowcloak']) {
  if (!fixtures.some(f => f.coverage.weather_abilities.accuracy.some(x => x.ability === ability && x.after < x.before))) throw new Error(`Missing weather accuracy ${ability}`);
  if (!fixtures.some(f => f.name.startsWith(ability === 'sandveil' ? 'sand_veil' : 'snow_cloak') && f.coverage.weather_evasion_misses > 0)) throw new Error(`Missing weather miss ${ability}`);
}
for (const ability of ['sandveil', 'snowcloak']) {
  if (!fixtures.some(f => f.coverage.weather_abilities.accuracy.some(x => x.ability === ability && x.suppressed && x.after === x.before))) throw new Error(`Missing suppressed evasion noop ${ability}`);
}
if (!fixtures.some(f => f.coverage.weather_abilities.accuracy.some(x => x.attacker_stage !== 0 && x.defender_stage !== 0))) throw new Error('Missing accuracy/evasion stages');
for (const move of ['sleeppowder', 'poisonpowder']) {
  if (!fixtures.some(f => f.coverage.weather_abilities.powder.some(x => x.move === move && x.blocked))) throw new Error(`Missing Overcoat powder ${move}`);
}
if (!fixtures.some(f => f.coverage.weather_abilities.powder.some(x => x.suppressed && x.blocked))) throw new Error('Missing Overcoat powder under weather suppression');
for (const ability of ['sandveil', 'overcoat']) {
  if (!fixtures.some(f => f.coverage.weather_abilities.sand_immunity.some(x => x.ability === ability && x.blocked))) throw new Error(`Missing actual sand immunity ${ability}`);
}
for (const ability of ['sandforce', 'sandveil', 'overcoat']) {
  if (fixtures.some(f => f.coverage.weather_abilities.sand_damage.some(x => x.ability === ability && x.amount > 0))) throw new Error(`Sand immunity leaked ${ability}`);
}
for (const metric of ['hydration_order', 'hydration_expiry_retained', 'hydration_suppressed_retained']) {
  if (!fixtures.some(f => f.coverage[metric])) throw new Error(`Missing Hydration interaction ${metric}`);
}
if (fixtures.some(f => f.coverage.weather_abilities.hydration_damage.some(x => x.effective === 'raindance' && x.amount > 0))) throw new Error('Hydration cured status still dealt damage');
for (const suppressed of [true, false]) {
  if (!fixtures.some(f => f.coverage.weather_abilities.hydration_damage.some(x => x.suppressed === suppressed && !x.effective && x.amount > 0))) throw new Error(`Missing Hydration inactive status damage ${suppressed}`);
}
if (!fixtures.some(f => f.coverage.hydration_activations > 0)) throw new Error('Missing Hydration cure');
for (const weather of ['raindance', 'sunnyday', 'sandstorm', 'snowscape']) {
  if (!fixtures.some(f => f.name.startsWith('cloudnine_') && f.steps.some(s => s.expected.climate.raw === weather && s.expected.climate.suppressed && !s.expected.climate.effective))) throw new Error(`Missing weather suppression ${weather}`);
  if (!fixtures.some(f => f.name.startsWith('cloudnine_') && f.steps.some((s, i) => i && f.steps[i - 1].expected.climate.raw === weather && f.steps[i - 1].expected.climate.suppressed && !s.expected.climate.raw))) throw new Error(`Missing suppressed expiration ${weather}`);
  if (!fixtures.some(f => f.name.includes('_switch_') && f.name.startsWith('cloudnine_') && f.steps.some(s => s.expected.sides[0].pokemon.some(p => p.roster === 0 && !p.fainted && p.active_slot === null) && s.expected.climate.raw === weather && !s.expected.climate.suppressed && s.expected.climate.effective === weather))) throw new Error(`Missing switch suppression end ${weather}`);
}
if (!fixtures.some(f => f.name.startsWith('cloudnine_rain_switch') && f.steps.some((s, i) => i && f.steps[i - 1].expected.sides[0].pokemon.find(p => p.roster === 0).ability_ending && s.expected.sides[0].pokemon.some(p => p.roster === 0 && p.active_slot !== null && !p.ability_ending) && s.expected.climate.suppressed))) throw new Error('Missing Cloud Nine re-entry ending reset');
if (!fixtures.some(f => f.name.startsWith('cloudnine_faint') && f.steps.some(s => s.expected.sides[0].pokemon.find(p => p.roster === 0).fainted && s.expected.climate.raw && !s.expected.climate.suppressed))) throw new Error('Missing faint suppression end');
for (const [weather, type, modifier] of [['raindance', 'Water', 1.5], ['sunnyday', 'Fire', 1.5]]) {
  for (const suppressed of [true, false]) {
    if (!fixtures.some(f => f.name.startsWith('cloudnine_') && f.coverage.suppression.power.some(x => x.weather === weather && x.type === type && x.suppressed === suppressed && (suppressed ? x.after === x.before : x.after > x.before)))) throw new Error(`Missing suppression damage ${weather}/${suppressed}`);
  }
}
for (const [weather, ability] of [['raindance', 'swiftswim'], ['sunnyday', 'chlorophyll'], ['sandstorm', 'sandrush'], ['snowscape', 'slushrush']]) {
  for (const suppressed of [true, false]) {
    if (!fixtures.some(f => f.name.startsWith('cloudnine_') && f.coverage.suppression.speed.some(x => x.weather === weather && x.ability === ability && x.suppressed === suppressed && (suppressed ? x.after === x.before : x.after > x.before)))) throw new Error(`Missing suppression speed ${weather}/${suppressed}`);
  }
}
for (const [weather, type] of [['raindance', 'Water'], ['sunnyday', 'Fire'], ['sandstorm', 'Rock'], ['snowscape', 'Ice']]) {
  for (const suppressed of [true, false]) {
    if (!fixtures.some(f => f.name.startsWith('cloudnine_') && f.coverage.suppression.weather_ball.some(x => x.weather === weather && x.suppressed === suppressed && x.type === (suppressed ? 'Normal' : type) && x.power === (suppressed ? 50 : 100)))) throw new Error(`Missing suppression Weather Ball ${weather}/${suppressed}`);
  }
}
for (const [weather, ability] of [['raindance', 'raindish'], ['snowscape', 'icebody']]) {
  if (!fixtures.some(f => f.name.startsWith('cloudnine_') && f.coverage.suppression.healing.some(x => x.weather === weather && x.ability === ability && !x.suppressed && x.amount > 0))) throw new Error(`Missing resumed weather healing ${ability}`);
}
if (!fixtures.some(f => f.name.startsWith('cloudnine_sand_switch') && f.coverage.suppression.sand.some(x => !x.suppressed && x.amount > 0))) throw new Error('Missing resumed sand damage');
if (!fixtures.some(f => f.name.startsWith('cloudnine_sun_switch') && f.coverage.suppression.solar.some(x => !x.suppressed && x.amount > 0))) throw new Error('Missing resumed Solar Power damage');
for (const f of fixtures.filter(f => f.name.startsWith('cloudnine_'))) {
  if ([...f.coverage.suppression.healing, ...f.coverage.suppression.sand, ...f.coverage.suppression.solar].some(x => x.suppressed && x.amount > 0)) throw new Error(`Suppressed weather effect leaked ${f.name}`);
}
for (const ability of ['raindish', 'icebody']) {
  if (!fixtures.some(f => f.coverage.weather_ability_heals.includes(ability))) throw new Error(`Missing weather healing ${ability}`);
}
if (!fixtures.some(f => f.coverage.weather_ability_damage.includes('solarpower'))) throw new Error('Missing Solar Power residual damage');
for (const metric of ['boosted', 'inactive', 'thickfat_ties', 'inactive_thickfat_ties', 'active_thickfat_ties']) {
  if (!fixtures.some(f => f.coverage.solar_power[metric] > 0)) throw new Error(`Missing Solar Power modifier ${metric}`);
}
for (const terrain of ['electricterrain', 'grassyterrain', 'psychicterrain', 'mistyterrain']) {
  if (!fixtures.some(f => f.coverage.terrain_starts.includes(terrain))) throw new Error(`Missing terrain start ${terrain}`);
  if (!fixtures.some(f => f.coverage.terrain_expired.includes(terrain))) throw new Error(`Missing terrain expiration ${terrain}`);
  for (const duration of [5, 8]) {
    if (!fixtures.some(f => f.teams.some(t => t.members.some(p => p.item === ids.items.terrainextender)) === (duration === 8) && f.steps.some(s => s.expected.field.some(x => x[0] === ids.conditions[terrain] && x[1] === duration - 1)))) throw new Error(`Missing terrain duration ${terrain}/${duration}`);
  }
}
for (const [terrain, type] of [['electricterrain', 'Electric'], ['grassyterrain', 'Grass'], ['psychicterrain', 'Psychic']]) {
  for (const grounded of [true, false]) {
    if (!fixtures.some(f => f.coverage.terrain_interactions.power.some(x => x.terrain === terrain && x.type === type && x.attacker_grounded === grounded && (grounded ? x.after > x.before : x.after === x.before)))) throw new Error(`Missing terrain grounded power ${terrain}/${grounded}`);
  }
}
for (const grounded of [true, false]) {
  if (!fixtures.some(f => f.coverage.terrain_interactions.power.some(x => x.terrain === 'mistyterrain' && x.type === 'Dragon' && x.defender_grounded === grounded && (grounded ? x.after < x.before : x.after === x.before)))) throw new Error(`Missing Misty dragon power ${grounded}`);
}
for (const terrain of ['electricterrain', 'mistyterrain']) {
  for (const grounded of [true, false]) {
    if (!fixtures.some(f => f.coverage.terrain_interactions.status.some(x => x.terrain === terrain && x.grounded === grounded && x.blocked === grounded))) throw new Error(`Missing terrain status prevention ${terrain}/${grounded}`);
  }
}
for (const [grounded, ally, blocked] of [[true, false, true], [false, false, false], [true, true, false]]) {
  if (!fixtures.some(f => f.coverage.terrain_interactions.priority.some(x => x.grounded === grounded && x.ally === ally && x.blocked === blocked))) throw new Error(`Missing Psychic priority ${grounded}/${ally}/${blocked}`);
}
if (!fixtures.some(f => f.coverage.grassy_final_heal)) throw new Error('Missing final-turn Grassy healing');
if (!fixtures.some(f => f.coverage.grassy_leftovers_order)) throw new Error('Missing Grassy healing before Leftovers');
if (!fixtures.some(f => f.coverage.grassy_heals > 0 && f.coverage.terrain_interactions.power.some(x => x.terrain === 'grassyterrain' && x.move === 'earthquake' && x.defender_grounded && x.after < x.before))) throw new Error('Missing Grassy heal/Earthquake interaction');
if (!fixtures.some(f => f.name.startsWith('terrain_replacement') && f.coverage.terrain_starts.length >= 3)) throw new Error('Missing terrain replacement');
if (!fixtures.some(f => f.name.startsWith('terrain_weather_room') && f.steps.some(s => s.expected.field.length === 3))) throw new Error('Missing terrain/weather/room coexistence');
for (const name of ['trickroom_expiration_priority', 'trickroom_simultaneous', 'trickroom_recast', 'trickroom_tailwind_scarf_paralysis', 'trickroom_switch_persistence']) {
  if (!fixtures.some(f => f.name.startsWith(name) && f.coverage.room_starts > 0 && f.coverage.room_ends > 0)) throw new Error(`Missing room start/end ${name}`);
}
if (!fixtures.some(f => f.name.startsWith('trickroom_expiration') && f.coverage.room_expired)) throw new Error('Missing room natural expiration');
if (!fixtures.some(f => f.name.startsWith('trickroom_recast') && f.coverage.room_recast)) throw new Error('Missing room recast toggle');
if (!fixtures.some(f => f.name.startsWith('trickroom_simultaneous') && f.steps.filter(s => s.expected.turn === 2).every(s => !s.expected.field.some(x => x[0] === ids.conditions.trickroom)))) throw new Error('Missing simultaneous room cancellation');
if (!fixtures.some(f => f.name.startsWith('trickroom_switch') && f.coverage.room_switch_persistence)) throw new Error('Missing room switch persistence');
if (!fixtures.some(f => f.name.startsWith('trickroom_tailwind') && f.coverage.statuses.includes('par') && f.coverage.side_starts.includes('tailwind'))) throw new Error('Missing room paralysis/Tailwind interaction');
for (const weather of ['raindance', 'sunnyday', 'sandstorm', 'snowscape']) {
  if (!fixtures.some(f => f.coverage.weather_starts.includes(weather))) throw new Error(`Missing weather start ${weather}`);
  if (!fixtures.some(f => f.coverage.weather_expired.includes(weather))) throw new Error(`Missing weather expiration ${weather}`);
}
for (const weather of ['raindance', 'sunnyday', 'sandstorm', 'snowscape']) {
  if (!fixtures.some(f => f.coverage.weather_ball.includes(weather))) throw new Error(`Missing Weather Ball damage in ${weather}`);
}
if (!fixtures.some(f => f.name.startsWith('weather_replacement') && f.coverage.weather_starts.length >= 3)) throw new Error('Missing weather replacement');
if (!fixtures.some(f => f.coverage.weather_ends > 0)) throw new Error('Missing weather expiration');
if (!fixtures.some(f => f.coverage.sand_hits > 0)) throw new Error('Missing sand damage');
for (const status of ['brn', 'par', 'slp', 'frz', 'psn', 'tox']) {
  if (!fixtures.some(f => f.coverage.statuses.includes(status))) throw new Error(`Missing exercised status ${status}`);
}
for (const item of ['focussash', 'sitrusberry', 'lumberry']) {
  if (!fixtures.some(f => f.coverage.consumed_items.includes(item))) throw new Error(`Missing exercised item ${item}`);
}
for (const item of ['leftovers', 'lifeorb', 'rockyhelmet']) {
  if (!fixtures.some(f => f.coverage.visible_item_effects.includes(item))) throw new Error(`Missing visible item effect ${item}`);
}
for (const effect of ['tailwind', 'reflect', 'lightscreen']) {
  if (!fixtures.some(f => f.coverage.side_starts.includes(effect)) || !fixtures.some(f => f.coverage.side_ends.includes(effect))) throw new Error(`Missing side effect start/end ${effect}`);
}
for (const type of ['physical', 'special', 'critical']) {
  if (!fixtures.some(f => f.coverage.screens[type] > 0)) throw new Error(`Missing exercised screen interaction ${type}`);
}
for (const metric of ['drain_heals', 'recoil_hits', 'ooze_hits']) {
  if (!fixtures.some(f => f.coverage[metric] > 0)) throw new Error(`Missing exercised effect ${metric}`);
}
if (!fixtures.some(f => f.coverage.flinches > 0)) throw new Error('Missing exercised flinch');
fs.writeFileSync(new URL('../data/turn-fixtures.json', import.meta.url), JSON.stringify({oracle_commit: ORACLE_COMMIT, format: FORMAT, fixtures}) + '\n');
console.log(JSON.stringify(fixtures.map(f => ({name: f.name, decisions: f.steps.length, turns: f.steps.at(-1).expected.turn}))));
