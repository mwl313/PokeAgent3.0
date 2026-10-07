// Development-only corpus for the priority-queue move family:
// - Quash rewrites the target's queued move action order to 201 (doubles
//   only), and fails when the target has no queued move left,
// - Upper Hand gates itself on the target's queued move (positive data
//   priority, non-status) and flinches it before it can act,
// - Gigaton Hammer's `cantusetwice` flag disables it in the next request
//   while it is still the holder's last used move.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, logHas, monAt, runTrials} = createScaffold();

const GH = ids.moves.gigatonhammer;
const logIndex = (session, pattern) => session.battle.log.findIndex(line => pattern.test(line));
const turnLines = (session, turn) => {
  const log = session.battle.log;
  const marker = `|turn|${turn}`;
  const start = log.indexOf(marker);
  if (start < 0) return [];
  const end = log.findIndex((line, index) => index > start && line === `|turn|${turn + 1}`);
  return log.slice(start, end < 0 ? undefined : end);
};
const turnHas = (session, turn, pattern) => turnLines(session, turn).some(line => pattern.test(line));
const turnIndex = (session, turn, pattern) => turnLines(session, turn).findIndex(line => pattern.test(line));

const fill = () => [
  offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
  offensive('Milotic', 'Competitive', ['Surf', 'Protect']),
  offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
  offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
  offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
];
const team = head => [head, ...fill().filter(p => p.species !== head.species).slice(0, 5)];
const milotic = () => offensive('Milotic', 'Competitive', ['Surf', 'Protect']);
const benchOf = (...species) => [
  milotic(),
  offensive('Torterra', 'Shell Armor', ['Seed Bomb', 'Protect']),
  offensive('Falinks', 'Battle Armor', ['Smart Strike', 'Protect']),
  offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
  offensive('Metagross', 'Clear Body', ['Iron Head', 'Protect']),
  offensive('Aggron', 'Sturdy', ['Iron Head', 'Protect']),
].filter(p => !species.includes(p.species));
const withPartner = (partner, bench) => [partner, ...bench].slice(0, 6);
const sneasler = () => offensive('Sneasler', 'Poison Touch', ['Dire Claw', 'Close Combat', 'Protect']);

const sableye = moves => setOf('Sableye', 'Prankster', moves);
const tinkaton = moves => setOf('Tinkaton', 'Mold Breaker', moves);

const TRIALS = [
  {
    name: 'queue_quash_delays_faster_foe',
    // P1a Quashes the faster foe; the foe's move must resolve after every
    // other move of the turn but before the residual phase.
    p1: () => withPartner(sableye(['Quash', 'Protect', 'Rain Dance', 'Reflect']),
      benchOf('Sableye')),
    p2: () => [sneasler(), ...benchOf('Sneasler')].slice(0, 6),
    script: [
      {p1: [{move: 'quash', target: 1}, 'surf'], p2: [{move: 'direclaw', target: 2}, 'protect']},
    ],
    coverage: {move: 'quash'},
    verify(fixture, session) {
      if (!turnHas(session, 1, /\|-activate\|p2a: [^|]*\|move: Quash/)) return 'Quash never activated on the faster foe';
      const quashed = turnIndex(session, 1, /^\|move\|p2a: /);
      const partner = turnIndex(session, 1, /^\|move\|p1b: /);
      const foePartner = turnIndex(session, 1, /^\|move\|p2b: /);
      if (quashed < 0 || partner < 0) return 'the turn never resolved both moves';
      if (quashed < partner) return 'the quashed foe still moved before the partner';
      if (foePartner >= 0 && quashed < foePartner) return 'the quashed foe still moved before its own partner';
      return null;
    },
  },
  {
    name: 'queue_quash_fails_after_target_moved',
    // The foe is just as fast and wins the +1 priority tie, so its action is
    // already off the queue when Quash resolves: the move fails.
    p1: () => withPartner(sableye(['Quash', 'Protect', 'Rain Dance', 'Reflect']),
      benchOf('Sableye')),
    p2: () => [
      offensive('Blaziken', 'Speed Boost', ['Feint', 'Protect']),
      ...benchOf('Blaziken'),
    ].slice(0, 6),
    script: [
      {p1: [{move: 'quash', target: 1}, 'surf'], p2: [{move: 'feint', target: 2}, 'protect']},
    ],
    coverage: {move: 'quash'},
    verify(fixture, session) {
      if (!turnHas(session, 1, /^\|move\|p1a: [^|]*\|Quash\|/)) return 'Quash was never used';
      if (turnHas(session, 1, /\|-activate\|p2a: [^|]*\|move: Quash/)) return 'Quash activated after the target had already moved';
      const foeMove = turnIndex(session, 1, /^\|move\|p2a: /);
      const quash = turnIndex(session, 1, /^\|move\|p1a: [^|]*\|Quash\|/);
      if (foeMove < 0 || quash < 0) return 'the turn never resolved both moves';
      if (foeMove > quash) return 'the faster foe did not move before Quash';
      return null;
    },
  },
  {
    name: 'queue_quash_fails_target_switches',
    // The foe switches out before any move action runs, so the original
    // target has no queued move when Quash resolves.
    p1: () => withPartner(sableye(['Quash', 'Protect', 'Rain Dance', 'Reflect']),
      benchOf('Sableye')),
    p2: () => [sneasler(), ...benchOf('Sneasler')].slice(0, 6),
    script: [
      {p1: [{move: 'quash', target: 1}, 'protect'], p2: [{switch: 's2'}, 'protect']},
    ],
    coverage: {move: 'quash'},
    verify(fixture, session) {
      if (!turnHas(session, 1, /^\|move\|p1a: [^|]*\|Quash\|/)) return 'Quash was never used';
      if (turnHas(session, 1, /\|-activate\|p2a: [^|]*\|move: Quash/)) return 'Quash activated against a switching target';
      if (!turnHas(session, 1, /^\|switch\|p2a: /)) return 'the foe never switched';
      return null;
    },
  },
  {
    name: 'queue_upperhand_flinches_priority',
    // Upper Hand (priority +3) acts before the foe's Quick Attack (priority
    // +1) and its 100% flinch secondary aborts that move.
    p1: () => withPartner(setOf('Sneasler', 'Unburden', ['Upper Hand', 'Close Combat', 'Protect']),
      benchOf('Sneasler')),
    p2: () => [
      offensive('Blaziken', 'Speed Boost', ['Feint', 'Protect']),
      ...benchOf('Blaziken'),
    ].slice(0, 6),
    script: [
      {p1: [{move: 'upperhand', target: 1}, 'protect'], p2: [{move: 'feint', target: 2}, 'protect']},
    ],
    coverage: {move: 'upperhand'},
    verify(fixture, session) {
      if (!logHas(session, /^\|move\|p1a: [^|]*\|Upper Hand\|/)) return 'Upper Hand was never used';
      if (!logHas(session, /\|cant\|p2a: [^|]*\|flinch/)) return 'the priority foe still moved despite the flinch';
      return null;
    },
  },
  {
    name: 'queue_upperhand_fails_vs_status_move',
    p1: () => withPartner(setOf('Sneasler', 'Unburden', ['Upper Hand', 'Close Combat', 'Protect']),
      benchOf('Sneasler')),
    p2: () => [
      offensive('Blaziken', 'Speed Boost', ['Protect', 'Close Combat']),
      ...benchOf('Blaziken'),
    ].slice(0, 6),
    script: [
      {p1: [{move: 'upperhand', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'upperhand'},
    verify(fixture, session) {
      if (!logHas(session, /^\|move\|p1a: [^|]*\|Upper Hand\|/)) return 'Upper Hand was never used';
      if (logHas(session, /^\|move\|p1a: [^|]*\|Upper Hand\|p2a: [^|]*\|\[miss\]/)) return 'Upper Hand resolved against a status move';
      if (!logHas(session, /\|-fail\|p1a: /)) return 'the refused Upper Hand never failed';
      const turnTwo = fixture.steps.find(step => step.expected.turn === 2);
      const blaziken = turnTwo?.expected.sides[1].pokemon[0];
      if (!blaziken || blaziken.hp < blaziken.max_hp) return 'Upper Hand damaged the foe despite the status gate';
      return null;
    },
  },
  {
    name: 'queue_upperhand_fails_vs_non_priority_move',
    p1: () => withPartner(setOf('Sneasler', 'Unburden', ['Upper Hand', 'Close Combat', 'Protect']),
      benchOf('Sneasler')),
    p2: () => [
      offensive('Blaziken', 'Speed Boost', ['Flare Blitz', 'Protect']),
      ...benchOf('Blaziken'),
    ].slice(0, 6),
    script: [
      {p1: [{move: 'upperhand', target: 1}, 'protect'], p2: [{move: 'flareblitz', target: 1}, 'protect']},
    ],
    coverage: {move: 'upperhand'},
    verify(fixture, session) {
      if (!logHas(session, /^\|move\|p1a: [^|]*\|Upper Hand\|/)) return 'Upper Hand was never used';
      if (!logHas(session, /\|-fail\|p1a: /)) return 'the refused Upper Hand never failed';
      if (!logHas(session, /^\|move\|p2a: [^|]*\|Flare Blitz\|/)) return 'the foe never used its normal move';
      return null;
    },
  },
  {
    name: 'queue_gigaton_hammer_request_disable',
    p1: () => team(tinkaton(['Gigaton Hammer', 'Protect', 'Encore', 'Fake Out'])),
    p2: () => [sneasler(), ...benchOf('Sneasler')].slice(0, 6),
    script: [
      {p1: [{move: 'gigatonhammer', target: 1}, 'protect']},
      {p1: ['protect', 'protect']},
      {p1: [{move: 'gigatonhammer', target: 1}, 'protect']},
    ],
    coverage: {move: 'gigatonhammer'},
    verify(fixture, session) {
      if (!logHas(session, /^\|move\|p1a: [^|]*\|Gigaton Hammer\|/)) return 'Gigaton Hammer was never used';
      const turnTwo = fixture.steps.find(step => step.expected.turn === 2 && step.side === 'P1');
      if (!turnTwo) return 'no turn-2 request was recorded';
      const slot = turnTwo.expected.sides[0].request_detail?.slots?.[0];
      if (!slot) return 'the turn-2 request has no active slot';
      const gh = slot.moves.find(m => m.id === GH);
      if (!gh) return 'the turn-2 request did not advertise Gigaton Hammer';
      if (!gh.disabled) return 'Gigaton Hammer was not disabled on the turn after its use';
      const turnThree = fixture.steps.find(step => step.expected.turn === 3 && step.side === 'P1');
      const later = turnThree?.expected.sides[0].request_detail?.slots?.[0]?.moves.find(m => m.id === GH);
      if (!later) return 'the turn-3 request did not advertise Gigaton Hammer';
      if (later.disabled) return 'Gigaton Hammer stayed disabled after an intervening move';
      const uses = session.battle.log.filter(line => /^\|move\|p1a: [^|]*\|Gigaton Hammer\|/.test(line));
      if (uses.length < 2) {
        return 'Gigaton Hammer was never used twice across the scene';
      }
      return null;
    },
  },
  {
    name: 'queue_round_chain_doubles_partner',
    // The faster Round finds its partner's queued Round action and pulls it
    // to the front of the queue: it resolves immediately afterwards with its
    // base power doubled.
    p1: () => [
      offensive('Milotic', 'Competitive', ['Round', 'Protect']),
      offensive('Chimecho', 'Levitate', ['Round', 'Protect']),
      ...benchOf('Milotic', 'Chimecho'),
    ].slice(0, 6),
    p2: () => [sneasler(), ...benchOf('Sneasler')].slice(0, 6),
    script: [
      {p1: [{move: 'round', target: 1}, {move: 'round', target: 1}], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'round'},
    verify(fixture, session) {
      const moves = turnLines(session, 1).filter(line => line.startsWith('|move|'));
      const rounds = moves.filter(line => line.includes('|Round|'));
      if (rounds.length < 2) return 'both Rounds never resolved';
      const first = moves.findIndex(line => line.includes('|Round|'));
      if (!moves[first + 1]?.includes('|Round|')) return 'the partner Round did not resolve immediately after the first';
      if (!moves[first + 1].startsWith('|move|p1b: ') || !moves[first].startsWith('|move|p1a: ')) {
        return 'the Round chain resolved in the wrong order';
      }
      return null;
    },
  },
  {
    name: 'queue_round_alone_no_chain',
    p1: () => [
      offensive('Milotic', 'Competitive', ['Round', 'Protect']),
      offensive('Chimecho', 'Levitate', ['Dazzling Gleam', 'Protect']),
      ...benchOf('Milotic', 'Chimecho'),
    ].slice(0, 6),
    p2: () => [sneasler(), ...benchOf('Sneasler')].slice(0, 6),
    script: [
      {p1: [{move: 'round', target: 1}, 'dazzlinggleam'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'round'},
    verify(fixture, session) {
      const moves = turnLines(session, 1).filter(line => line.startsWith('|move|'));
      const first = moves.findIndex(line => line.includes('|Round|'));
      if (first < 0) return 'Round was never used';
      if (moves[first + 1]?.includes('|Round|')) return 'a lone Round still pulled a chained Round';
      if (!moves[first + 1]?.includes('|Dazzling Gleam|')) return 'the partner never used its own move';
      return null;
    },
  },
  {
    // The reference runs a move's own `onTry` before the `PrepareHit` event,
    // so a failed action gate must suppress Protean/Libero's type change.
    name: 'queue_gate_precedes_prepare_hit',
    p1: () => [offensive('Meowscarada', 'Protean', ['Sucker Punch', 'Protect', 'Knock Off']), ...benchOf()].slice(0, 6),
    p2: () => [offensive('Blaziken', 'Speed Boost', ['Protect', 'Flare Blitz']), ...benchOf('Blaziken')].slice(0, 6),
    script: [
      {p1: [{move: 'suckerpunch', target: 1}, 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {move: 'suckerpunch'},
    verify(fixture, session) {
      if (!logHas(session, /^\|move\|p1a: [^|]*\|Sucker Punch\|/)) return 'Sucker Punch was never used';
      const turnTwo = fixture.steps.find(step => step.expected.turn === 2);
      const cat = turnTwo?.expected.sides[0].pokemon[0];
      if (!cat) return 'no turn-2 state was recorded';
      if (cat.types.length === 1 && cat.types[0] === ids.types.dark) {
        return 'Protean changed the type despite the failed gate';
      }
      return null;
    },
  },
  {
    // The same ordering with a passing gate: Protean changes the type before
    // the hit resolves.
    name: 'queue_gate_allows_prepare_hit',
    p1: () => [offensive('Meowscarada', 'Protean', ['Sucker Punch', 'Protect', 'Knock Off']), ...benchOf()].slice(0, 6),
    p2: () => [offensive('Blaziken', 'Speed Boost', ['Flare Blitz', 'Protect']), ...benchOf('Blaziken')].slice(0, 6),
    script: [
      {p1: [{move: 'suckerpunch', target: 1}, 'protect'], p2: [{move: 'flareblitz', target: 1}, 'protect']},
    ],
    coverage: {move: 'suckerpunch'},
    verify(fixture, session) {
      if (!logHas(session, /^\|move\|p1a: [^|]*\|Sucker Punch\|p2a: /)) return 'Sucker Punch never resolved against the foe';
      const changed = fixture.steps.some(step => {
        const cat = step.expected.sides[0].pokemon[0];
        return cat.types.length === 1 && cat.types[0] === ids.types.dark;
      });
      if (!changed) return 'Protean never changed the type on the passing gate';
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 17000, artifact: 'more_queue.json', debugEnv: 'DEBUG_QUEUE'});
