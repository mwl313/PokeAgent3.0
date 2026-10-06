// Development oracle only. Never launch this from Rust or the training loop.
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import {createRequire} from 'node:module';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(import.meta.url);
const reference = path.join(root, 'vendor/pokemon-showdown');
const {Battle, TeamValidator, PRNG} = require(path.join(reference, 'dist/sim'));
export const ORACLE_COMMIT = '14546894d86f9589ac11130c510bbe73b6968665';
export const FORMAT = 'gen9championsvgc2026regmc';

export function verifyReference() {
  if (execFileSync('git', ['rev-parse', 'HEAD'], {cwd: reference, encoding: 'utf8'}).trim() !== ORACLE_COMMIT ||
      execFileSync('git', ['diff', '--name-only', 'HEAD'], {cwd: reference, encoding: 'utf8'}).trim()) {
    throw new Error('Reference checkout does not match the pinned clean source');
  }
}

export class ReferenceSession {
  constructor({teams, seed, record_rng = false}) {
    if (!Array.isArray(teams) || teams.length !== 2 || !Array.isArray(seed) || seed.length !== 4 ||
      !seed.every(x => Number.isInteger(x) && x >= 0 && x <= 65535)) throw new Error('Expected two teams and four uint16 seed words');
    const validator = new TeamValidator(FORMAT);
    this.teams = structuredClone(teams);
    for (const team of this.teams) {
      const problems = validator.validateTeam(team);
      if (problems) throw new Error(`Invalid reference team: ${problems.join('; ')}`);
    }
    this.seed = [...seed]; this.actions = []; this.rng_draws = []; this.messages = []; this.record_rng = record_rng;
    const prng = new PRNG(seed);
    if (record_rng) {
      const original = prng.rng.next.bind(prng.rng);
      prng.rng.next = () => { const value = original(); this.rng_draws.push(value); return value; };
    }
    this.battle = new Battle({formatid: FORMAT, prng,
      p1: {name: 'p1', team: this.teams[0]}, p2: {name: 'p2', team: this.teams[1]},
      send: (type, payload) => this.messages.push({type, payload})});
  }

  // This explicitly privileged view is ONLY for differential testing, not actors.
  debugState() {
    const b = this.battle;
    return {oracle_commit: ORACLE_COMMIT, format: FORMAT, turn: b.turn,
      terminated: b.ended, truncated: false, operational_error: null, winner: b.ended ? b.winner || null : null,
      rng_seed: b.prng.getSeed(), request_state: b.requestState,
      sides: b.sides.map(s => ({id: s.id, request: structuredClone(s.activeRequest),
        pokemon: s.pokemon.map(p => ({species: p.species.id, base_species: p.baseSpecies.id,
          hp: p.hp, maxhp: p.maxhp, stats: {...p.storedStats}, boosts: {...p.boosts}, status: p.status,
          item: p.item, ability: p.ability, active: p.isActive, fainted: p.fainted,
          gender: p.gender, position: p.position, can_mega: p.canMegaEvo, can_tera: p.canTerastallize,
          moves: p.moveSlots.map(m => ({id: m.id, pp: m.pp, maxpp: m.maxpp, disabled: m.disabled, target: m.target})),
          volatiles: Object.keys(p.volatiles).sort()}))})),
    };
  }

  choose(side, command) {
    if (!['p1', 'p2'].includes(side) || typeof command !== 'string') throw new Error('Invalid side/command');
    if (/showteam|openteamsheets/i.test(command)) throw new Error('Open team sheets are disabled');
    const start = this.messages.length;
    const accepted = this.battle.choose(side, command);
    this.actions.push({side, command, accepted, rng_after: this.battle.prng.getSeed()});
    return {accepted, messages: this.messages.slice(start), state: this.debugState()};
  }

  snapshot() { return {oracle_commit: ORACLE_COMMIT, state: this.battle.toJSON(),
    teams: structuredClone(this.teams), seed: [...this.seed], record_rng: this.record_rng,
    actions: structuredClone(this.actions), rng_draws: [...this.rng_draws]}; }

  restore(snapshot) {
    if (snapshot.oracle_commit !== ORACLE_COMMIT) throw new Error('Wrong reference snapshot pin');
    const restored = Battle.fromJSON(snapshot.state);
    if (restored.format.id !== FORMAT) { restored.destroy(); throw new Error('Wrong snapshot format'); }
    restored.send = (type, payload) => this.messages.push({type, payload});
    this.battle.destroy(); this.battle = restored;
    this.teams = structuredClone(snapshot.teams); this.seed = [...snapshot.seed]; this.record_rng = snapshot.record_rng;
    this.actions = structuredClone(snapshot.actions); this.rng_draws = [...snapshot.rng_draws]; this.messages = [];
    // Record restored RNG draws with the same wrapper if tracing was requested.
    if (this.record_rng) {
      const original = this.battle.prng.rng.next.bind(this.battle.prng.rng);
      this.battle.prng.rng.next = () => { const value = original(); this.rng_draws.push(value); return value; };
    }
  }

  exportTrace() {
    return {schema: 'pa3-reference-trace-v1', oracle_commit: ORACLE_COMMIT, format: FORMAT,
      teams: this.teams, seed: this.seed, actions: this.actions, rng_draws: this.rng_draws, log: [...this.battle.log]};
  }

  destroy() { this.battle.destroy(); }
}

async function main() {
  verifyReference();
  const pool = fs.readFileSync(path.join(root, 'data/teams/mb-mc-v2-all-train/train.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  let session;
  const lines = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
  for await (const line of lines) {
    let response;
    try {
      if (line.length > 16 * 1024 * 1024) throw new Error('Oracle request exceeds 16 MiB');
      const q = JSON.parse(line);
      if (q.op === 'reset') {
        const teams = q.teams || [q.team_a, q.team_b].map(i => {
          if (!Number.isInteger(i) || !pool[i]) throw new Error('Invalid training team index');
          return pool[i].members.map(m => m.showdown_set);
        });
        const next = new ReferenceSession({teams, seed: q.seed, record_rng: q.record_rng});
        session?.destroy(); session = next; response = session.debugState();
      } else {
        if (!session) throw new Error('Reset the oracle first');
        if (q.op === 'observe_debug') response = session.debugState();
        else if (q.op === 'choose') response = session.choose(q.side, q.command);
        else if (q.op === 'snapshot') response = session.snapshot();
        else if (q.op === 'restore') { session.restore(q.snapshot); response = session.debugState(); }
        else if (q.op === 'export_trace') response = session.exportTrace();
        else throw new Error('Unknown oracle operation');
      }
      process.stdout.write(JSON.stringify({ok: true, result: response}) + '\n');
    } catch (error) {
      process.stdout.write(JSON.stringify({ok: false, operational_error: error.message}) + '\n');
    }
  }
  session?.destroy();
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
