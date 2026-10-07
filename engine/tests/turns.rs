use pa3_engine::batch::{BattleBatch, ResetSpec, SideChoice, StepSpec};
use pa3_engine::{
    actions::{AtomicAction, RequestKind},
    assets::Dex,
    state::{BattleState, SideId, Team},
};
use serde::Deserialize;
use std::path::Path;
use std::sync::Arc;

#[derive(Deserialize)]
struct Corpus {
    oracle_commit: String,
    format: String,
    fixtures: Vec<Fixture>,
}
#[derive(Deserialize)]
struct Fixture {
    name: String,
    seed: [u16; 4],
    teams: [Team; 2],
    steps: Vec<Step>,
}
#[derive(Deserialize)]
struct Step {
    side: SideId,
    actions: Vec<AtomicAction>,
    command: String,
    expected: Expected,
}
#[derive(Deserialize)]
struct Expected {
    turn: u16,
    rng_seed: String,
    terminated: bool,
    winner: Option<String>,
    sides: [ExpectedSide; 2],
    field: Vec<(u16, u16, usize)>,
}
#[derive(Deserialize)]
struct ExpectedSide {
    request: RequestKind,
    #[serde(default)]
    request_detail: Option<RequestDetail>,
    pokemon: Vec<ExpectedMon>,
    conditions: Vec<(u16, u16)>,
}

/// Privileged reference request data used to compare the native legal action
/// mask (which moves are selectable, at what PP, whether Mega is available and
/// which bench destinations are offered).
#[derive(Deserialize)]
struct RequestDetail {
    kind: RequestKind,
    slots: Vec<ExpectedSlotRequest>,
    bench: Vec<u8>,
    preview: Vec<u8>,
}

#[derive(Deserialize)]
struct ExpectedSlotRequest {
    present: bool,
    requires_replacement: bool,
    can_mega: bool,
    moves: Vec<ExpectedMoveChoice>,
    /// Reference `getLockedMove()`: the single legal move of a charging
    /// Pokémon. Absent for ordinary slots.
    #[serde(default)]
    locked: Option<u16>,
    /// `onLockMove: 'recharge'` — the forced Recharge pseudo-move.
    #[serde(default)]
    locked_recharge: Option<bool>,
    #[serde(default)]
    trapped: Option<bool>,
    /// Reference `maybeTrapped`: only the side's last active slot exposes it.
    #[serde(default)]
    maybe_trapped: Option<bool>,
}

#[derive(Deserialize)]
struct ExpectedMoveChoice {
    id: u16,
    pp: u8,
    disabled: bool,
    target: String,
}

/// Reference spelling of a target class, used to compare request move masks.
fn target_name(target: pa3_engine::assets::Target) -> &'static str {
    use pa3_engine::assets::Target::*;
    match target {
        Normal => "normal",
        AdjacentFoe => "adjacentFoe",
        AdjacentAlly => "adjacentAlly",
        AdjacentAllyOrSelf => "adjacentAllyOrSelf",
        Any => "any",
        RandomNormal => "randomNormal",
        SelfOnly => "self",
        AllAdjacent => "allAdjacent",
        AllAdjacentFoes => "allAdjacentFoes",
        All => "all",
        AllySide => "allySide",
        FoeSide => "foeSide",
        AllyTeam => "allyTeam",
        Allies => "allies",
        Scripted => "scripted",
    }
}
#[derive(Deserialize)]
struct ExpectedMon {
    roster: u8,
    cached_speed: Option<i32>,
    species: u16,
    hp: u16,
    max_hp: u16,
    fainted: bool,
    active_slot: Option<u8>,
    status: u16,
    boosts: [i8; 7],
    stats: [u16; 6],
    ability: u16,
    #[serde(default)]
    ability_ending: bool,
    item: u16,
    previous_item: u16,
    types: Vec<u16>,
    can_mega: bool,
    pp: Vec<u8>,
    volatiles: Vec<String>,
}

#[test]
fn static_is_revealed_before_status_and_lum_cure() {
    use pa3_engine::knowledge::EventKind;
    let dex = Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap();
    let corpus: Corpus = serde_json::from_str(include_str!("../data/turn-fixtures.json")).unwrap();
    let static_id = dex.id("abilities", "static").unwrap();
    let lum = dex.id("items", "lumberry").unwrap();
    let paralysis = dex.id("conditions", "par").unwrap();
    let mut checked = 0;
    for fixture in corpus
        .fixtures
        .into_iter()
        .filter(|fixture| fixture.name.starts_with("static_lum_"))
    {
        let mut state = BattleState::reset(
            &dex,
            [&fixture.teams[0], &fixture.teams[1]],
            fixture.seed,
            [0, 1],
        )
        .unwrap();
        state.enable_trace().unwrap();
        for step in fixture.steps {
            let result = state.step(&dex, step.side, &step.actions).unwrap();
            assert_eq!(result.outcome.operational_error, None, "{}", fixture.name);
        }
        let trace = state.export_trace().unwrap();
        assert!(
            trace.events[0].windows(4).any(|events| {
                let [ability, status, item, cure] = events else {
                    unreachable!()
                };
                ability.event.kind == EventKind::Ability
                    && ability.event.effect == static_id
                    && status.event.kind == EventKind::Status
                    && status.event.effect == paralysis
                    && status.event.target == Some(ability.event.subject)
                    && item.event.kind == EventKind::EndItem
                    && item.event.effect == lum
                    && item.event.subject == status.event.subject
                    && cure.event.kind == EventKind::CureStatus
                    && cure.event.subject == status.event.subject
                    && cure.event.effect == paralysis
                    && events.iter().all(|event| event.turn == ability.turn)
            }),
            "{} must actually exercise ability reveal before Lum's immediate cure",
            fixture.name
        );
        checked += 1;
    }
    assert_eq!(checked, 3, "all guarded Static/Lum reference seeds");
}

#[test]
fn native_battles_match_reference_at_every_decision_boundary() {
    let dex = Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap();
    let corpus: Corpus = serde_json::from_str(include_str!("../data/turn-fixtures.json")).unwrap();
    assert_eq!(corpus.oracle_commit, pa3_engine::ORACLE_COMMIT);
    assert_eq!(corpus.format, pa3_engine::FORMAT);
    use sha2::{Digest, Sha256};
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("../data/manifest.json")).unwrap();
    assert_eq!(
        manifest["files"]["turn-fixtures.json"]["sha256"],
        format!(
            "{:x}",
            Sha256::digest(include_bytes!("../data/turn-fixtures.json"))
        )
    );
    // Development aid: `PA3_SKIP_FIXTURES=a,b` skips named corpus entries so a
    // single known-divergent battle does not hide the rest of the report. The
    // skipped set is empty in normal runs, so no coverage is silently dropped.
    let skip: Vec<String> = std::env::var("PA3_SKIP_FIXTURES")
        .map(|list| {
            list.split(',')
                .filter(|name| !name.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    for f in corpus.fixtures {
        if skip.contains(&f.name) {
            continue;
        }
        let mut state =
            BattleState::reset(&dex, [&f.teams[0], &f.teams[1]], f.seed, [0, 1]).unwrap();
        state.enable_trace().unwrap();
        for (i, step) in f.steps.into_iter().enumerate() {
            let context = format!("{} decision {i} {:?} {}", f.name, step.side, step.command);
            let result = state
                .step(&dex, step.side, &step.actions)
                .unwrap_or_else(|e| panic!("{context}: {e}"));
            assert_eq!(result.outcome.operational_error, None, "{context}");
            // Privileged snapshot inspection is differential test code only.
            let envelope: serde_json::Value =
                serde_json::from_slice(&state.snapshot().unwrap_or_else(|e| {
                    panic!("{context}: snapshot: {e}")
                }))
                .unwrap();
            let world: serde_json::Value =
                serde_json::from_str(envelope["payload"].as_str().unwrap()).unwrap();
            for side in [SideId::P1, SideId::P2] {
                let view = state.observe(side);
                let expected = &step.expected.sides[side.index()];
                let mut actual_field: Vec<(u16, u16, usize)> = world["field"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(id, effect)| {
                        let owner = if effect["source"][0] == "P1" { 0 } else { 1 };
                        (
                            id.parse().unwrap(),
                            effect["duration"].as_u64().unwrap() as u16,
                            owner,
                        )
                    })
                    .collect();
                actual_field.sort_by_key(|effect| effect.0);
                assert_eq!(actual_field, step.expected.field, "{context} battle field");
                assert_eq!(
                    view.knowledge.field.keys().copied().collect::<Vec<_>>(),
                    step.expected.field.iter().map(|x| x.0).collect::<Vec<_>>(),
                    "{context} public battle field"
                );
                for &(id, duration, owner) in &step.expected.field {
                    let effect = &view.knowledge.field[&id];
                    assert!(effect.present, "{context} public field");
                    assert_eq!(
                        effect.duration.known,
                        owner == side.index() || id == dex.id("conditions", "trickroom").unwrap(),
                        "{context} field duration visibility"
                    );
                    if effect.duration.known {
                        assert_eq!(
                            effect.duration.value, duration as i16,
                            "{context} known field duration"
                        );
                    }
                }

                assert_eq!(view.turn, step.expected.turn, "{context}");
                assert_eq!(view.request.kind, expected.request, "{context}");
                if let Some(detail) = &expected.request_detail {
                    assert_eq!(view.request.kind, detail.kind, "{context} request kind");
                    if matches!(detail.kind, RequestKind::Preview) {
                        assert_eq!(
                            view.request.preview_roster, detail.preview,
                            "{context} preview roster mask"
                        );
                    } else if !matches!(detail.kind, RequestKind::Finished) {
                        // The reference lists switch destinations in its own
                        // request team order; the engine reports stable roster
                        // indices. Compare the destination sets, and compare
                        // the preview roster in order because that order is
                        // itself the player's pick ordering.
                        let mut reference_bench = detail.bench.clone();
                        reference_bench.sort_unstable();
                        let mut actual_bench = view.request.bench.clone();
                        actual_bench.sort_unstable();
                        assert_eq!(
                            actual_bench, reference_bench,
                            "{context} switch destination mask"
                        );
                        let actionable = matches!(detail.kind, RequestKind::Normal);
                        for (slot, want) in detail.slots.iter().enumerate() {
                            if !actionable && !want.requires_replacement {
                                continue;
                            }
                            let got = &view.request.slots[slot];
                            let place = format!("{context} slot {slot}");
                            // A replacement request populates only the slots
                            // that must be replaced, and a waiting side keeps
                            // placeholder slot data; only actionable slots are
                            // compared.
                            if matches!(detail.kind, RequestKind::Replacement)
                                && !want.requires_replacement
                            {
                                continue;
                            }
                            assert_eq!(got.present, want.present, "{place} presence");
                            assert_eq!(
                                got.requires_replacement, want.requires_replacement,
                                "{place} replacement flag"
                            );
                            assert_eq!(got.can_mega, want.can_mega, "{place} mega availability");
                            if !want.present {
                                // A fainted or replaced slot cannot act; the
                                // engine intentionally leaves its move list
                                // empty while the reference shows the last
                                // request's stale entries.
                                continue;
                            }
                            let actual: Vec<(u16, u8, &str)> = got
                                .moves
                                .iter()
                                .filter(|m| !m.disabled && m.pp > 0)
                                .map(|m| (m.id, m.pp, target_name(m.target)))
                                .collect();
                            let reference: Vec<(u16, u8, &str)> = want
                                .moves
                                .iter()
                                .filter(|m| !m.disabled && m.pp > 0)
                                .map(|m| (m.id, m.pp, m.target.as_str()))
                                .collect();
                            assert_eq!(actual, reference, "{place} legal move mask");
                            // Locked slots: the reference request offers the
                            // charging move (or the Recharge pseudo-move) and
                            // refuses switches, so the native mask must agree.
                            assert_eq!(
                                got.locked_move, want.locked,
                                "{place} locked move"
                            );
                            if let Some(recharge) = want.locked_recharge {
                                assert_eq!(
                                    got.locked_recharge, recharge,
                                    "{place} recharge lock"
                                );
                            }
                            if let Some(trapped) = want.trapped {
                                assert_eq!(got.trapped, trapped, "{place} trapped flag");
                            }
                            if let Some(maybe) = want.maybe_trapped {
                                assert_eq!(
                                    got.maybe_trapped, maybe,
                                    "{place} maybe-trapped flag"
                                );
                            }
                        }
                    }
                }
                let actual_conditions: Vec<(u16, u16)> = world["sides"][side.index()]["conditions"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(id, state)| {
                        (
                            id.parse().unwrap(),
                            // Duration-less side conditions (entry hazards)
                            // serialize as null and compare as 0.
                            state["duration"].as_u64().unwrap_or(0) as u16,
                        )
                    })
                    .collect::<std::collections::BTreeMap<_, _>>()
                    .into_iter()
                    .collect();
                assert_eq!(
                    actual_conditions, expected.conditions,
                    "{context} side conditions"
                );
                for owner in 0..2 {
                    let relative = usize::from(owner != side.index());
                    let public = &view.knowledge.sides[relative];
                    let expected_ids: Vec<_> = step.expected.sides[owner]
                        .conditions
                        .iter()
                        .map(|x| x.0)
                        .collect();
                    assert_eq!(
                        public.keys().copied().collect::<Vec<_>>(),
                        expected_ids,
                        "{context} public side conditions"
                    );
                    for &(id, duration) in &step.expected.sides[owner].conditions {
                        let effect = &public[&id];
                        assert!(effect.present, "{context} public side effect");
                        assert_eq!(
                            effect.duration.known,
                            owner == side.index(),
                            "{context} duration visibility"
                        );
                        if effect.duration.known {
                            assert_eq!(
                                effect.duration.value, duration as i16,
                                "{context} own side-effect duration"
                            );
                        }
                    }
                }
                assert_eq!(
                    view.outcome.terminated, step.expected.terminated,
                    "{context}"
                );
                if view.outcome.terminated {
                    assert_eq!(
                        view.outcome
                            .winner
                            .map(|s| if s == SideId::P1 { "p1" } else { "p2" }),
                        step.expected.winner.as_deref(),
                        "{context}"
                    );
                }
                for p in &expected.pokemon {
                    let actual = &view.own.pokemon[p.roster as usize];
                    assert_eq!(
                        actual.status, p.status,
                        "{context} {:?} mon {} status",
                        side, p.roster
                    );
                    assert_eq!(
                        actual.boosts, p.boosts,
                        "{context} {:?} mon {} boosts",
                        side, p.roster
                    );
                    if let Some(speed) = p.cached_speed {
                        assert_eq!(
                            world["sides"][side.index()]["pokemon"][p.roster as usize]["cached_speed"],
                            speed,
                            "{context} cached action speed mon {}",
                            p.roster
                        );
                    }
                    assert_eq!(actual.stats, p.stats, "{context} stats");
                    assert_eq!(actual.ability, p.ability, "{context} ability");
                    assert_eq!(
                        world["sides"][side.index()]["pokemon"][p.roster as usize]["ability_ending"],
                        p.ability_ending,
                        "{context} ability ending"
                    );
                    assert_eq!(actual.item, p.item, "{context} item");
                    assert_eq!(
                        actual.previous_item, p.previous_item,
                        "{context} consumed item"
                    );
                    assert_eq!(actual.types, p.types, "{context} types");
                    if expected.request == RequestKind::Normal
                        && let Some(slot) = actual.active_slot
                        && !p.fainted
                    {
                        assert_eq!(
                            view.request.slots[slot as usize].can_mega, p.can_mega,
                            "{context} Mega availability"
                        );
                    }
                    assert_eq!(actual.species, p.species, "{context} mon {}", p.roster);
                    assert_eq!(actual.hp, p.hp, "{context} {:?} mon {}", side, p.roster);
                    assert_eq!(actual.stats[0], p.max_hp, "{context}");
                    assert_eq!(actual.fainted, p.fainted, "{context}");
                    assert_eq!(actual.active_slot, p.active_slot, "{context}");
                    assert_eq!(
                        actual.moves.iter().map(|m| m.pp).collect::<Vec<_>>(),
                        p.pp,
                        "{context} mon {}",
                        p.roster
                    );
                    // Remaining durations are world data, not assumed actor knowledge.
                    let mon = &world["sides"][side.index()]["pokemon"][p.roster as usize];
                    let mut volatiles: Vec<_> = mon["volatiles"]
                        .as_object()
                        .unwrap()
                        .keys()
                        .map(|id| {
                            let key = id.parse::<usize>().unwrap();
                            // A two-turn charge marker is keyed by its move id
                            // in the reference, not by a condition id.
                            if key < dex.moves.len() && dex.moves[key].charge.is_some() {
                                dex.names["moves"][key].clone()
                            } else {
                                dex.names["conditions"][key].clone()
                            }
                        })
                        .collect();
                    volatiles.sort();
                    assert_eq!(
                        volatiles, p.volatiles,
                        "{context} {:?} mon {} volatiles",
                        side, p.roster
                    );
                }
            }
            assert_eq!(
                state.rng_seed().map(|x| x.to_string()).join(","),
                step.expected.rng_seed,
                "{context} RNG"
            );
            if i % 7 == 0 {
                state = BattleState::restore(
                    &dex,
                    &state
                        .snapshot()
                        .unwrap_or_else(|e| panic!("{context}: snapshot: {e}")),
                )
                .unwrap_or_else(|e| panic!("{context}: restore: {e}"));
            }
        }
        let trace = state.export_trace().unwrap();
        assert!(trace.events.iter().all(|events| !events.is_empty()));
        let replayed = BattleState::replay_trace(&dex, trace).unwrap();
        assert_eq!(replayed, state, "{} complete trace replay", f.name);
    }
}

#[test]
fn two_16_worker_batches_match_serial_for_2048_complete_battles() {
    let dex = Arc::new(Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap());
    let corpus: Corpus = serde_json::from_str(include_str!("../data/turn-fixtures.json")).unwrap();
    let fixtures = Arc::new(corpus.fixtures);
    let teams = Arc::new(
        fixtures
            .iter()
            .flat_map(|f| f.teams.clone())
            .collect::<Vec<_>>(),
    );
    let expected = Arc::new(
        fixtures
            .iter()
            .map(|f| {
                let mut state =
                    BattleState::reset(&dex, [&f.teams[0], &f.teams[1]], f.seed, [0, 1]).unwrap();
                for (decision, step) in f.steps.iter().enumerate() {
                    state
                        .step(&dex, step.side, &step.actions)
                        .unwrap_or_else(|e| {
                            panic!(
                                "{} decision {decision} {:?} {}: {e}",
                                f.name, step.side, step.command
                            )
                        });
                }
                assert!(state.observe(SideId::P1).outcome.terminated);
                state.snapshot().unwrap()
            })
            .collect::<Vec<_>>(),
    );
    let mut groups = vec![];
    for group in 0..2 {
        let dex = dex.clone();
        let teams = teams.clone();
        let fixtures = fixtures.clone();
        let expected = expected.clone();
        groups.push(std::thread::spawn(move || {
            let mut batch = BattleBatch::new(dex, teams, 16).unwrap();
            let fixture_index = |slot: usize| (group * 1024 + slot) % fixtures.len();
            let specs: Vec<_> = (0..1024)
                .map(|slot| {
                    let f = fixture_index(slot);
                    ResetSpec {
                        team_a: 2 * f,
                        team_b: 2 * f + 1,
                        seed: fixtures[f].seed,
                        role_map: [0, 1],
                    }
                })
                .collect();
            let handles = batch.reset_batch(&specs).unwrap();
            for decision in 0..fixtures.iter().map(|f| f.steps.len()).max().unwrap() {
                // Reverse/sparse submissions verify result routing independently
                // of storage order, while some environments have already ended.
                let inputs: Vec<_> = handles
                    .iter()
                    .enumerate()
                    .rev()
                    .filter_map(|(slot, handle)| {
                        fixtures[fixture_index(slot)]
                            .steps
                            .get(decision)
                            .map(|step| StepSpec {
                                handle: *handle,
                                choices: vec![SideChoice {
                                    side: step.side,
                                    actions: step.actions.clone(),
                                }],
                            })
                    })
                    .collect();
                let results = batch.step_batch(&inputs).unwrap();
                assert_eq!(results.len(), inputs.len());
                for (input, result) in inputs.iter().zip(results) {
                    let f = &fixtures[fixture_index(input.handle.slot as usize)];
                    assert_eq!(
                        result.outcome.operational_error, None,
                        "{} decision {decision}",
                        f.name
                    );
                    assert_eq!(
                        result.outcome.terminated, f.steps[decision].expected.terminated,
                        "{} decision {decision}",
                        f.name
                    );
                }
            }
            for (slot, handle) in handles.into_iter().enumerate() {
                assert_eq!(
                    batch.snapshot(handle).unwrap(),
                    expected[fixture_index(slot)],
                    "group {group} environment {slot}"
                );
            }
        }));
    }
    for group in groups {
        group.join().unwrap();
    }
}

#[test]
fn batch_preflight_is_atomic_and_pending_choices_are_private() {
    let dex = Arc::new(Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap());
    let corpus: Corpus = serde_json::from_str(include_str!("../data/turn-fixtures.json")).unwrap();
    let f = &corpus.fixtures[0];
    let mut batch = BattleBatch::new(dex.clone(), Arc::new(f.teams.to_vec()), 2).unwrap();
    let handles = batch
        .reset_batch(
            &[ResetSpec {
                team_a: 0,
                team_b: 1,
                seed: f.seed,
                role_map: [0, 1],
            }; 2],
        )
        .unwrap();
    let before: Vec<_> = handles
        .iter()
        .map(|h| batch.snapshot(*h).unwrap())
        .collect();
    let choice = SideChoice {
        side: f.steps[0].side,
        actions: f.steps[0].actions.clone(),
    };
    let valid = StepSpec {
        handle: handles[0],
        choices: vec![choice.clone()],
    };
    let invalid = StepSpec {
        handle: handles[1],
        choices: vec![choice.clone(), choice.clone()],
    };
    assert!(batch.step_batch(&[valid.clone(), invalid]).is_err());
    for (i, handle) in handles.iter().enumerate() {
        assert_eq!(batch.snapshot(*handle).unwrap(), before[i]);
    }
    assert!(batch.step_batch(&[valid.clone(), valid.clone()]).is_err());
    let opponent_before = batch.observe_batch(&[(handles[0], SideId::P2)]).unwrap();
    batch.step_batch(&[valid]).unwrap();
    assert_eq!(
        batch.observe_batch(&[(handles[0], SideId::P2)]).unwrap(),
        opponent_before
    );
    let pending = batch.snapshot(handles[0]).unwrap();
    assert!(
        batch
            .step_batch(&[StepSpec {
                handle: handles[0],
                choices: vec![choice]
            }])
            .is_err()
    );
    assert_eq!(batch.snapshot(handles[0]).unwrap(), pending);
    batch.restore(handles[0], &before[0]).unwrap();
    let choices = f.steps[..2]
        .iter()
        .rev()
        .map(|s| SideChoice {
            side: s.side,
            actions: s.actions.clone(),
        })
        .collect();
    batch
        .step_batch(&[StepSpec {
            handle: handles[0],
            choices,
        }])
        .unwrap();
    let mut serial = BattleState::reset(&dex, [&f.teams[0], &f.teams[1]], f.seed, [0, 1]).unwrap();
    for step in &f.steps[..2] {
        serial.step(&dex, step.side, &step.actions).unwrap();
    }
    assert_eq!(
        batch.snapshot(handles[0]).unwrap(),
        serial.snapshot().unwrap()
    );
}

#[test]
fn unsupported_effects_are_replayable_operational_failures() {
    let dex = Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap();
    let corpus: Corpus = serde_json::from_str(include_str!("../data/turn-fixtures.json")).unwrap();
    let f = &corpus.fixtures[0];
    // The probe needs an unimplemented move the first fixture's lead can
    // legally hold: the pinned learnsets only overlap some of the open
    // families, so take the first candidate that resets and fails cleanly.
    // Each name below is a deliberate operational error until its family is
    // ported, which is exactly the contract this test pins.
    let mut probe = None;
    for name in ["Transform", "Wish", "Round", "Imprison", "Fling"] {
        let mut teams = f.teams.clone();
        teams[0].members[0].moves[0] = dex.id("moves", name).unwrap();
        let Ok(mut state) = BattleState::reset(&dex, [&teams[0], &teams[1]], f.seed, [0, 1])
        else {
            continue;
        };
        state.enable_trace().unwrap();
        for step in &f.steps[..2] {
            state.step(&dex, step.side, &step.actions).unwrap();
        }
        let mut matched = true;
        for side in [SideId::P1, SideId::P2] {
            let request = state.observe(side).request;
            let mut actions = vec![];
            for _ in request.branch_slots() {
                actions.push(request.candidates(&actions).unwrap()[0]);
            }
            if state.step(&dex, side, &actions).is_err() {
                matched = false;
                break;
            }
        }
        if !matched {
            continue;
        }
        let expected = name.to_lowercase();
        for side in [SideId::P1, SideId::P2] {
            let view = state.observe(side);
            if view.request.kind != RequestKind::Finished
                || view.outcome.terminated
                || view.outcome.truncated
                || view.outcome.reward(side).is_some()
                || !view
                    .outcome
                    .operational_error
                    .as_deref()
                    .is_some_and(|error| error.contains(&expected))
            {
                matched = false;
            }
        }
        if matched {
            probe = Some(state);
            break;
        }
    }
    let state = probe.expect("no legal unimplemented probe move for fixture 0");
    let trace = state.export_trace().unwrap();
    assert_eq!(BattleState::replay_trace(&dex, trace).unwrap(), state);
    let mut corrupted = trace.clone();
    corrupted.actions[1].rng_before[0] ^= 1;
    assert!(BattleState::replay_trace(&dex, &corrupted).is_err());
    let mut corrupted = trace.clone();
    corrupted.events[0][0].event.value += 1;
    assert!(BattleState::replay_trace(&dex, &corrupted).is_err());
}

#[test]
fn active_views_and_masks_exclude_hidden_opponent_world_fields() {
    use sha2::{Digest, Sha256};
    let dex = Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap();
    let corpus: Corpus = serde_json::from_str(include_str!("../data/turn-fixtures.json")).unwrap();
    let f = &corpus.fixtures[0];
    let mut state = BattleState::reset(&dex, [&f.teams[0], &f.teams[1]], f.seed, [0, 1]).unwrap();
    for step in &f.steps[..4] {
        state.step(&dex, step.side, &step.actions).unwrap();
    }
    let before = state.observe(SideId::P1);
    assert!(!before.knowledge.events.is_empty());
    let envelope: serde_json::Value = serde_json::from_slice(&state.snapshot().unwrap()).unwrap();
    let mut world: serde_json::Value =
        serde_json::from_str(envelope["payload"].as_str().unwrap()).unwrap();
    world["role_map"] = serde_json::json!([1, 0]);
    for mon in world["sides"][1]["pokemon"].as_array_mut().unwrap() {
        mon["nature"] = serde_json::json!(dex.id("natures", "Modest").unwrap());
        mon["stats"][1] = serde_json::json!(250);
        mon["stats"][3] = serde_json::json!(50);
    }
    // Substitute an unseen reserve while preserving the same public history.
    world["sides"][1]["selected_order"] = serde_json::json!([2, 3, 0, 5]);
    world["sides"][1]["pokemon"][0]["selected"] = true.into();
    world["sides"][1]["pokemon"][4]["selected"] = false.into();
    // A player's own request is derived from the selection/active/faint state,
    // so keep P2's private request consistent with the substituted roster.
    // P1's view and mask must not change because of any of these edits.
    let selected: Vec<u8> =
        serde_json::from_value(world["sides"][1]["selected_order"].clone()).unwrap();
    let bench: Vec<u8> = {
        let pokes = world["sides"][1]["pokemon"].as_array().unwrap();
        selected
            .into_iter()
            .filter(|r| {
                let p = &pokes[*r as usize];
                p["fainted"] != true && p["active_slot"].is_null()
            })
            .collect()
    };
    world["requests"][1]["bench"] = serde_json::json!(bench);
    // The substituted selection is still a consistent world, so side 1's stored
    // request must list exactly the new non-active reserves. Regenerate its
    // bench; the opponent view below must not depend on it.
    if world["requests"][1]["kind"] == "Normal" {
        let order: Vec<u8> = world["sides"][1]["selected_order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        let fainted: Vec<u8> = world["sides"][1]["pokemon"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .filter(|(_, p)| p["fainted"].as_bool().unwrap_or(false))
            .map(|(i, _)| i as u8)
            .collect();
        let active = world["sides"][1]["active"].clone();
        let bench: Vec<u8> = order
            .into_iter()
            .filter(|r| {
                !fainted.contains(r)
                    && !active
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|a| a.as_u64() == Some(u64::from(*r)))
            })
            .collect();
        world["requests"][1]["bench"] = serde_json::json!(bench);
    }
    let payload = serde_json::to_string(&world).unwrap();
    let snapshot = serde_json::to_vec(&serde_json::json!({"sha256": format!("{:x}", Sha256::digest(payload.as_bytes())), "payload": payload})).unwrap();
    let alternate = BattleState::restore(&dex, &snapshot).unwrap();
    assert_eq!(before, alternate.observe(SideId::P1));
    assert_eq!(
        before.request.candidates(&[]).unwrap(),
        alternate
            .observe(SideId::P1)
            .request
            .candidates(&[])
            .unwrap()
    );
    for event in before.knowledge.events {
        if event.subject >= 6
            && matches!(
                event.kind,
                pa3_engine::knowledge::EventKind::Damage | pa3_engine::knowledge::EventKind::Heal
            )
        {
            assert_eq!(event.value, 0);
            assert_eq!(event.health.unwrap().denominator, 100);
        }
    }
}

#[test]
fn reused_buffers_preserve_sparse_order_and_preflight_atomicity() {
    let dex = Arc::new(Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap());
    let corpus: Corpus = serde_json::from_str(include_str!("../data/turn-fixtures.json")).unwrap();
    let f = &corpus.fixtures[0];
    let mut batch = BattleBatch::new(dex.clone(), Arc::new(f.teams.to_vec()), 2).unwrap();
    let handles = batch
        .reset_batch(
            &[ResetSpec {
                team_a: 0,
                team_b: 1,
                seed: f.seed,
                role_map: [0, 1],
            }; 3],
        )
        .unwrap();
    let untouched = batch.snapshot(handles[1]).unwrap();
    let mut serial: Vec<_> = (0..3)
        .map(|_| BattleState::reset(&dex, [&f.teams[0], &f.teams[1]], f.seed, [0, 1]).unwrap())
        .collect();
    let mut results = Vec::with_capacity(4);
    let mut candidates = Vec::with_capacity(64);
    for (round, step) in f.steps.iter().take(6).enumerate() {
        let order = if round % 2 == 0 { [2, 0] } else { [0, 2] };
        let specs: Vec<_> = order
            .iter()
            .map(|&slot| StepSpec {
                handle: handles[slot],
                choices: vec![SideChoice {
                    side: step.side,
                    actions: step.actions.clone(),
                }],
            })
            .collect();
        for &slot in &order {
            let request = serial[slot].observe(step.side).request;
            let mut prefix = vec![];
            for &action in &step.actions {
                request.candidates_into(&prefix, &mut candidates).unwrap();
                assert_eq!(candidates, request.candidates(&prefix).unwrap());
                assert!(candidates.contains(&action));
                prefix.push(action);
            }
        }
        batch.step_batch_into(&specs, &mut results).unwrap();
        assert_eq!(results.len(), 2);
        for (index, &slot) in order.iter().enumerate() {
            assert_eq!(
                results[index],
                serial[slot].step(&dex, step.side, &step.actions).unwrap()
            );
            assert_eq!(
                batch.snapshot(handles[slot]).unwrap(),
                serial[slot].snapshot().unwrap()
            );
        }
        assert_eq!(batch.snapshot(handles[1]).unwrap(), untouched);
        assert_eq!(results.capacity(), 4);
        assert_eq!(candidates.capacity(), 64);
        let before_results = results.clone();
        let invalid = vec![specs[0].clone(), specs[0].clone()];
        assert!(batch.step_batch_into(&invalid, &mut results).is_err());
        assert_eq!(results, before_results);
        for &slot in &order {
            assert_eq!(
                batch.snapshot(handles[slot]).unwrap(),
                serial[slot].snapshot().unwrap()
            );
        }
    }
    // Different request phases make result ordering observable, rather than
    // checking only identical results from synchronized copies of one game.
    batch.restore(handles[2], &untouched).unwrap();
    serial[2] = BattleState::reset(&dex, [&f.teams[0], &f.teams[1]], f.seed, [0, 1]).unwrap();
    let steps = [&f.steps[0], &f.steps[6]];
    let order = [2, 0];
    let specs: Vec<_> = order
        .iter()
        .zip(steps)
        .map(|(&slot, step)| StepSpec {
            handle: handles[slot],
            choices: vec![SideChoice {
                side: step.side,
                actions: step.actions.clone(),
            }],
        })
        .collect();
    batch.step_batch_into(&specs, &mut results).unwrap();
    assert_ne!(results[0].request_kinds, results[1].request_kinds);
    for (index, (&slot, step)) in order.iter().zip(steps).enumerate() {
        assert_eq!(
            results[index],
            serial[slot].step(&dex, step.side, &step.actions).unwrap()
        );
        assert_eq!(
            batch.snapshot(handles[slot]).unwrap(),
            serial[slot].snapshot().unwrap()
        );
    }
    batch.step_batch_into(&[], &mut results).unwrap();
    assert!(results.is_empty());
    assert_eq!(results.capacity(), 4);
}

#[test]
fn snapshot_suppression_lifecycle_is_private_and_rejects_impossible_end_state() {
    use sha2::{Digest, Sha256};
    let dex = Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap();
    let corpus: Corpus = serde_json::from_str(include_str!("../data/turn-fixtures.json")).unwrap();
    let cloud_nine = dex.id("abilities", "Cloud Nine").unwrap();
    let f = corpus
        .fixtures
        .iter()
        .find(|f| {
            f.teams
                .iter()
                .any(|t| t.members.iter().any(|m| m.ability == cloud_nine))
                && f.steps.iter().any(|step| {
                    step.expected
                        .sides
                        .iter()
                        .any(|side| side.pokemon.iter().any(|mon| mon.ability_ending))
                })
        })
        .expect("Cloud Nine fixture with persisted End");
    let mut state = BattleState::reset(&dex, [&f.teams[0], &f.teams[1]], f.seed, [0, 1]).unwrap();
    for step in &f.steps[..2] {
        state.step(&dex, step.side, &step.actions).unwrap();
    }
    let envelope: serde_json::Value = serde_json::from_slice(&state.snapshot().unwrap()).unwrap();
    let initial: serde_json::Value =
        serde_json::from_str(envelope["payload"].as_str().unwrap()).unwrap();
    assert_eq!(
        initial["schema"],
        u64::from(pa3_engine::state::SNAPSHOT_SCHEMA)
    );
    let resign = |world: &serde_json::Value| {
        let payload = serde_json::to_string(world).unwrap();
        serde_json::to_vec(&serde_json::json!({"sha256": format!("{:x}", Sha256::digest(payload.as_bytes())), "payload": payload})).unwrap()
    };
    let mut legacy = initial.clone();
    legacy["schema"] = 2.into();
    assert!(BattleState::restore(&dex, &resign(&legacy)).is_err());
    let mut tested_active = false;
    for (side, owner) in initial["sides"].as_array().unwrap().iter().enumerate() {
        for (roster, mon) in owner["pokemon"].as_array().unwrap().iter().enumerate() {
            if mon["ability"] == cloud_nine
                && !mon["active_slot"].is_null()
                && !mon["fainted"].as_bool().unwrap()
            {
                tested_active = true;
                let mut invalid = initial.clone();
                invalid["sides"][side]["pokemon"][roster]["ability_ending"] = true.into();
                assert!(BattleState::restore(&dex, &resign(&invalid)).is_err());
            }
        }
    }
    assert!(
        tested_active,
        "fixture must start with an active suppressor"
    );
    let mut found_ended = false;
    for step in &f.steps[2..] {
        state.step(&dex, step.side, &step.actions).unwrap();
        let snapshot = state.snapshot().unwrap();
        let restored = BattleState::restore(&dex, &snapshot).unwrap();
        assert_eq!(state, restored);
        let views = [state.observe(SideId::P1), state.observe(SideId::P2)];
        let world_envelope: serde_json::Value = serde_json::from_slice(&snapshot).unwrap();
        let world: serde_json::Value =
            serde_json::from_str(world_envelope["payload"].as_str().unwrap()).unwrap();
        let mut alternate = world.clone();
        for owner in alternate["sides"].as_array_mut().unwrap() {
            for mon in owner["pokemon"].as_array_mut().unwrap() {
                if mon["ability_ending"] == true {
                    found_ended = true;
                    mon["ability_ending"] = false.into();
                }
            }
        }
        let alternate = BattleState::restore(&dex, &resign(&alternate)).unwrap();
        assert_eq!(
            views,
            [alternate.observe(SideId::P1), alternate.observe(SideId::P2)]
        );
    }
    assert!(
        found_ended,
        "fixture must reach a persisted ability End state"
    );
}
