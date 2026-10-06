use pa3_engine::{
    assets::Dex,
    batch::{BattleBatch, ResetSpec},
    state::{BattleState, EndReason, Outcome, SideId, Team},
};
use serde_json::Value;
use std::{
    path::Path,
    sync::{Arc, OnceLock},
};

fn dex() -> Arc<Dex> {
    static DEX: OnceLock<Arc<Dex>> = OnceLock::new();
    DEX.get_or_init(|| {
        Arc::new(Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap())
    })
    .clone()
}
fn teams() -> Arc<Vec<Team>> {
    static TEAMS: OnceLock<Arc<Vec<Team>>> = OnceLock::new();
    TEAMS
        .get_or_init(|| {
            Arc::new(serde_json::from_str(include_str!("../data/training-teams.json")).unwrap())
        })
        .clone()
}

#[test]
fn initialization_matches_reference_including_gender_rng_consumption() {
    let fixtures: Value =
        serde_json::from_str(include_str!("../data/reference-fixtures.json")).unwrap();
    let teams = teams();
    let dex = dex();
    for c in fixtures["initialization"].as_array().unwrap() {
        let indices: [usize; 2] = serde_json::from_value(c["team_indices"].clone()).unwrap();
        let seed = serde_json::from_value(c["seed"].clone()).unwrap();
        let state =
            BattleState::reset(&dex, [&teams[indices[0]], &teams[indices[1]]], seed, [0, 1])
                .unwrap();
        assert_eq!(
            state.rng_seed().map(|s| s.to_string()).join(","),
            c["final_seed"]
        );
        for (i, side) in [SideId::P1, SideId::P2].into_iter().enumerate() {
            let view = state.observe(side);
            for (j, mon) in view.own.pokemon.iter().enumerate() {
                let expected = &c["sides"][i][j];
                assert_eq!(
                    u64::from(mon.species),
                    expected["species"].as_u64().unwrap()
                );
                assert_eq!(u64::from(mon.gender), expected["gender"].as_u64().unwrap());
                assert_eq!(serde_json::to_value(mon.stats).unwrap(), expected["stats"]);
                assert_eq!(
                    serde_json::to_value(mon.moves.iter().map(|m| m.pp).collect::<Vec<_>>())
                        .unwrap(),
                    expected["pp"]
                );
            }
        }
        let restored = BattleState::restore(&dex, &state.snapshot().unwrap()).unwrap();
        assert_eq!(restored, state);
    }
}

#[test]
fn opponent_hidden_allocations_nature_and_team_id_do_not_change_view() {
    let teams = teams();
    let dex = dex();
    let a = BattleState::reset(&dex, [&teams[0], &teams[1]], [1, 2, 3, 4], [0, 1]).unwrap();
    let mut hidden = teams[1].clone();
    hidden.id = "private provenance must not enter observations".into();
    for set in &mut hidden.members {
        set.points = [1, 0, 0, 0, 0, 0];
        set.nature = dex.id("natures", "serious").unwrap();
    }
    let b = BattleState::reset(&dex, [&teams[0], &hidden], [1, 2, 3, 4], [1, 0]).unwrap();
    assert_eq!(a.observe(SideId::P1), b.observe(SideId::P1));
    assert_ne!(a.observe(SideId::P2).own, b.observe(SideId::P2).own);
    for opponent in &a.observe(SideId::P1).knowledge.pokemon[6..] {
        assert!(
            !opponent.item.known
                && !opponent.ability.known
                && !opponent.health.known
                && !opponent.selected.known
        );
        assert!(opponent.current_moves.iter().all(|m| !m.known));
    }
}

#[test]
fn all_1136_teams_initialize_in_a_2048_environment_group() {
    let teams = teams();
    let mut batch = BattleBatch::new(dex(), teams.clone(), 16).unwrap();
    let specs = (0..2048)
        .map(|i| ResetSpec {
            team_a: i % teams.len(),
            team_b: (i * 17) % teams.len(),
            seed: [2026, 10, 6, i as u16],
            role_map: [0, 1],
        })
        .collect::<Vec<_>>();
    let handles = batch.reset_batch(&specs).unwrap();
    assert_eq!(handles.len(), 2048);
    let views = batch
        .observe_batch(&handles.iter().map(|h| (*h, SideId::P1)).collect::<Vec<_>>())
        .unwrap();
    assert_eq!(views.len(), 2048);
    assert!(
        views
            .iter()
            .all(|v| v.request.candidates(&[]).unwrap().len() == 6)
    );
    let snapshot = batch.snapshot(handles[0]).unwrap();
    batch.restore(handles[0], &snapshot).unwrap();
    let mut invalid = specs[0];
    invalid.team_a = teams.len();
    assert!(batch.reset_batch(&[invalid]).is_err());
    assert_eq!(snapshot, batch.snapshot(handles[0]).unwrap()); // failed reset is atomic
    let new = batch.reset_batch(&specs[..1]).unwrap();
    assert!(batch.snapshot(handles[0]).is_err());
    assert!(batch.snapshot(new[0]).is_ok());
}

#[test]
fn invalid_format_teams_are_rejected_before_initialization() {
    let dex = dex();
    let teams = teams();
    let mut bad = teams[0].clone();
    bad.members[0].species = dex.id("species", "mewtwo").unwrap();
    assert!(BattleState::reset(&dex, [&bad, &teams[1]], [1, 2, 3, 4], [0, 1]).is_err());
    bad = teams[0].clone();
    bad.members[0].points[0] = 33;
    assert!(BattleState::reset(&dex, [&bad, &teams[1]], [1, 2, 3, 4], [0, 1]).is_err());
    bad = teams[0].clone();
    bad.members[1] = bad.members[0].clone();
    assert!(BattleState::reset(&dex, [&bad, &teams[1]], [1, 2, 3, 4], [0, 1]).is_err());
}

#[test]
fn corruption_and_reference_mismatch_do_not_restore() {
    let dex = dex();
    let teams = teams();
    let state = BattleState::reset(&dex, [&teams[0], &teams[1]], [1, 2, 3, 4], [0, 1]).unwrap();
    let snapshot = state.snapshot().unwrap();
    let mut corrupted: Value = serde_json::from_slice(&snapshot).unwrap();
    corrupted["sha256"] = Value::from("wrong");
    assert!(BattleState::restore(&dex, &serde_json::to_vec(&corrupted).unwrap()).is_err());
    let mut other_dex = (*dex).clone();
    other_dex.asset_digest = "different rules".into();
    assert!(BattleState::restore(&other_dex, &snapshot).is_err());
}

#[test]
fn valid_checksums_do_not_bypass_snapshot_structure_checks() {
    use sha2::{Digest, Sha256};
    let dex = dex();
    let teams = teams();
    let state = BattleState::reset(&dex, [&teams[0], &teams[1]], [1, 2, 3, 4], [0, 1]).unwrap();
    let envelope: Value = serde_json::from_slice(&state.snapshot().unwrap()).unwrap();
    let initial: Value = serde_json::from_str(envelope["payload"].as_str().unwrap()).unwrap();
    for (pointer, value) in [
        ("/sides/0/pokemon/0/ability", serde_json::json!(65535)),
        ("/sides/0/pokemon/0/item", serde_json::json!(65535)),
        ("/sides/0/pokemon/0/base_species", serde_json::json!(65535)),
        ("/sides/0/pokemon/0/stats/0", serde_json::json!(0)),
        ("/sides/0/active/0", serde_json::json!(250)),
        ("/sides/0/pokemon/0/active_slot", serde_json::json!(0)),
        ("/sides/0/selected_order", serde_json::json!([0, 0, 1, 2])),
        (
            "/faint_queue",
            serde_json::json!([{"side": 9, "roster": 0}]),
        ),
    ] {
        let mut world = initial.clone();
        *world.pointer_mut(pointer).unwrap() = value;
        let payload = serde_json::to_string(&world).unwrap();
        let snapshot = serde_json::to_vec(&serde_json::json!({"sha256": format!("{:x}", Sha256::digest(payload.as_bytes())), "payload": payload})).unwrap();
        assert!(BattleState::restore(&dex, &snapshot).is_err(), "{pointer}");
    }
    for (id, duration, source) in [
        (dex.effects.tailwind, Some(0), Some((SideId::P1, 0))),
        (dex.effects.tailwind, Some(5), Some((SideId::P1, 0))),
        (dex.effects.reflect, None, Some((SideId::P1, 0))),
        (dex.effects.light_screen, Some(9), Some((SideId::P1, 0))),
        (dex.effects.reflect, Some(5), Some((SideId::P2, 0))),
        (dex.effects.tailwind, Some(4), None),
    ] {
        let mut world = initial.clone();
        world["sides"][0]["conditions"][id.to_string()] = serde_json::json!({
            "id": id, "duration": duration, "source": source, "effect_order": 0, "effect_order_assigned": false, "values": []
        });
        let payload = serde_json::to_string(&world).unwrap();
        let snapshot = serde_json::to_vec(&serde_json::json!({"sha256": format!("{:x}", Sha256::digest(payload.as_bytes())), "payload": payload})).unwrap();
        assert!(
            BattleState::restore(&dex, &snapshot).is_err(),
            "invalid side condition {id}"
        );
    }
}

#[test]
fn snapshot_weather_requires_one_valid_timed_source() {
    use sha2::{Digest, Sha256};
    let dex = dex();
    let teams = teams();
    let state = BattleState::reset(&dex, [&teams[0], &teams[1]], [1, 2, 3, 4], [0, 1]).unwrap();
    let envelope: Value = serde_json::from_slice(&state.snapshot().unwrap()).unwrap();
    let initial: Value = serde_json::from_str(envelope["payload"].as_str().unwrap()).unwrap();
    let snapshot_with_field = |field: Value| {
        let mut world = initial.clone();
        world["field"] = field;
        let payload = serde_json::to_string(&world).unwrap();
        serde_json::to_vec(&serde_json::json!({
            "sha256": format!("{:x}", Sha256::digest(payload.as_bytes())),
            "payload": payload
        }))
        .unwrap()
    };
    for id in [
        dex.effects.rain,
        dex.effects.sun,
        dex.effects.sand,
        dex.effects.snow,
    ] {
        for duration in [1, 5, 8] {
            for side in [SideId::P1, SideId::P2] {
                let field = serde_json::json!({id.to_string(): {
                    "id": id, "duration": duration, "source": [side, 5],
                    "effect_order": 0, "effect_order_assigned": false, "values": []
                }});
                let bytes = snapshot_with_field(field);
                let restored = BattleState::restore(&dex, &bytes).unwrap();
                let restored_envelope: Value =
                    serde_json::from_slice(&restored.snapshot().unwrap()).unwrap();
                let restored_world: Value =
                    serde_json::from_str(restored_envelope["payload"].as_str().unwrap()).unwrap();
                assert_eq!(
                    restored_world["field"][id.to_string()]["duration"],
                    duration
                );
                assert_eq!(
                    restored_world["field"][id.to_string()]["source"],
                    serde_json::json!([side, 5])
                );
            }
        }
        for (duration, source) in [
            (serde_json::json!(null), serde_json::json!([SideId::P1, 0])),
            (serde_json::json!(0), serde_json::json!([SideId::P1, 0])),
            (serde_json::json!(9), serde_json::json!([SideId::P1, 0])),
            (serde_json::json!(5), serde_json::json!(null)),
            (serde_json::json!(5), serde_json::json!([SideId::P2, 6])),
            (serde_json::json!(5), serde_json::json!(["P3", 0])),
        ] {
            let field = serde_json::json!({id.to_string(): {
                "id": id, "duration": duration, "source": source,
                "effect_order": 0, "effect_order_assigned": false, "values": []
            }});
            assert!(BattleState::restore(&dex, &snapshot_with_field(field)).is_err());
        }
    }
    let field = serde_json::json!({
        dex.effects.rain.to_string(): {
            "id": dex.effects.rain, "duration": 5, "source": [SideId::P1, 0],
            "effect_order": 0, "effect_order_assigned": false, "values": []
        },
        dex.effects.sun.to_string(): {
            "id": dex.effects.sun, "duration": 8, "source": [SideId::P2, 0],
            "effect_order": 0, "effect_order_assigned": false, "values": []
        }
    });
    assert!(BattleState::restore(&dex, &snapshot_with_field(field)).is_err());
}

#[test]
fn snapshot_trick_room_requires_timed_source_and_can_coexist_with_weather() {
    use sha2::{Digest, Sha256};
    let dex = dex();
    let teams = teams();
    let state = BattleState::reset(&dex, [&teams[0], &teams[1]], [1, 2, 3, 4], [0, 1]).unwrap();
    let envelope: Value = serde_json::from_slice(&state.snapshot().unwrap()).unwrap();
    let initial: Value = serde_json::from_str(envelope["payload"].as_str().unwrap()).unwrap();
    let snapshot_with_room = |duration: Value, source: Value, weather: bool| {
        let mut world = initial.clone();
        world["field"][dex.effects.trick_room.to_string()] = serde_json::json!({
            "id": dex.effects.trick_room, "duration": duration, "source": source,
            "effect_order": 0, "effect_order_assigned": false, "values": []
        });
        if weather {
            world["field"][dex.effects.rain.to_string()] = serde_json::json!({
                "id": dex.effects.rain, "duration": 8, "source": [SideId::P2, 5],
                "effect_order": 0, "effect_order_assigned": false, "values": []
            });
        }
        let payload = serde_json::to_string(&world).unwrap();
        serde_json::to_vec(&serde_json::json!({
            "sha256": format!("{:x}", Sha256::digest(payload.as_bytes())),
            "payload": payload
        }))
        .unwrap()
    };
    for duration in [1, 5] {
        for source in [SideId::P1, SideId::P2] {
            for weather in [false, true] {
                let bytes = snapshot_with_room(
                    serde_json::json!(duration),
                    serde_json::json!([source, 5]),
                    weather,
                );
                assert!(BattleState::restore(&dex, &bytes).is_ok());
            }
        }
    }
    for (duration, source) in [
        (serde_json::json!(null), serde_json::json!([SideId::P1, 0])),
        (serde_json::json!(0), serde_json::json!([SideId::P1, 0])),
        (serde_json::json!(6), serde_json::json!([SideId::P1, 0])),
        (serde_json::json!(5), serde_json::json!(null)),
        (serde_json::json!(5), serde_json::json!([SideId::P2, 6])),
        (serde_json::json!(5), serde_json::json!(["P3", 0])),
    ] {
        let bytes = snapshot_with_room(duration, source, true);
        assert!(BattleState::restore(&dex, &bytes).is_err());
    }
}

#[test]
fn snapshot_terrain_requires_one_timed_source_and_coexists_with_weather_and_room() {
    use sha2::{Digest, Sha256};
    let dex = dex();
    let teams = teams();
    let state = BattleState::reset(&dex, [&teams[0], &teams[1]], [1, 2, 3, 4], [0, 1]).unwrap();
    let envelope: Value = serde_json::from_slice(&state.snapshot().unwrap()).unwrap();
    let initial: Value = serde_json::from_str(envelope["payload"].as_str().unwrap()).unwrap();
    let condition = |id, duration: Value, source: Value| {
        serde_json::json!({
            "id": id, "duration": duration, "source": source, "effect_order": 0, "effect_order_assigned": false, "values": []
        })
    };
    let snapshot_with_terrain = |id: u16, duration: Value, source: Value, second: bool| {
        let mut world = initial.clone();
        world["field"][dex.effects.rain.to_string()] = condition(
            dex.effects.rain,
            serde_json::json!(8),
            serde_json::json!([SideId::P2, 5]),
        );
        world["field"][dex.effects.trick_room.to_string()] = condition(
            dex.effects.trick_room,
            serde_json::json!(5),
            serde_json::json!([SideId::P1, 0]),
        );
        world["field"][id.to_string()] = condition(id, duration, source);
        if second {
            let other = if id == dex.effects.electric_terrain {
                dex.effects.grassy_terrain
            } else {
                dex.effects.electric_terrain
            };
            world["field"][other.to_string()] = condition(
                other,
                serde_json::json!(5),
                serde_json::json!([SideId::P1, 0]),
            );
        }
        let payload = serde_json::to_string(&world).unwrap();
        serde_json::to_vec(&serde_json::json!({
            "sha256": format!("{:x}", Sha256::digest(payload.as_bytes())), "payload": payload
        }))
        .unwrap()
    };
    for id in [
        dex.effects.electric_terrain,
        dex.effects.grassy_terrain,
        dex.effects.misty_terrain,
        dex.effects.psychic_terrain,
    ] {
        for duration in [1, 5, 8] {
            for side in [SideId::P1, SideId::P2] {
                let bytes = snapshot_with_terrain(
                    id,
                    serde_json::json!(duration),
                    serde_json::json!([side, 5]),
                    false,
                );
                assert!(BattleState::restore(&dex, &bytes).is_ok());
            }
        }
        for (duration, source) in [
            (serde_json::json!(null), serde_json::json!([SideId::P1, 0])),
            (serde_json::json!(0), serde_json::json!([SideId::P1, 0])),
            (serde_json::json!(9), serde_json::json!([SideId::P1, 0])),
            (serde_json::json!(5), serde_json::json!(null)),
            (serde_json::json!(5), serde_json::json!([SideId::P2, 6])),
            (serde_json::json!(5), serde_json::json!(["P3", 0])),
        ] {
            let bytes = snapshot_with_terrain(id, duration, source, false);
            assert!(BattleState::restore(&dex, &bytes).is_err());
        }
        let bytes = snapshot_with_terrain(
            id,
            serde_json::json!(5),
            serde_json::json!([SideId::P1, 0]),
            true,
        );
        assert!(BattleState::restore(&dex, &bytes).is_err());
    }
}

#[test]
fn operational_outcomes_never_become_draw_rewards() {
    let mut outcome = Outcome::default();
    assert_eq!(outcome.reward(SideId::P1), None);
    outcome.operational_error = Some("unimplemented effect".into());
    assert_eq!(outcome.reward(SideId::P1), None);
    outcome = Outcome {
        terminated: true,
        reason: Some(EndReason::LastPokemon),
        winner: Some(SideId::P1),
        ..Default::default()
    };
    assert_eq!(outcome.reward(SideId::P1), Some(1));
    assert_eq!(outcome.reward(SideId::P2), Some(-1));
    outcome.winner = None;
    outcome.reason = Some(EndReason::RuleTurnLimit);
    assert_eq!(outcome.reward(SideId::P1), Some(0));
    outcome.truncated = true;
    assert_eq!(outcome.reward(SideId::P1), None);
}
