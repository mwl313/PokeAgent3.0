use pa3_engine::{
    assets::Dex,
    knowledge::{
        EffectKind, EffectKnowledge, EventKind, HealthDisplay, Known, SemanticEvent, TokenRole,
        token_layout,
    },
    observation::{
        CategoryFeature as C, Encoder, FlagFeature as B, FloatFeature as F, ObservationBuffers,
        SCHEMA_VERSION, TokenFeatures,
    },
    state::{BattleState, SideId, Team},
};
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
fn view() -> pa3_engine::state::PlayerView {
    let dex = dex();
    let teams = teams();
    BattleState::reset(&dex, [&teams[0], &teams[1]], [1, 2, 3, 4], [0, 1])
        .unwrap()
        .observe(SideId::P1)
}
fn encoded(view: &pa3_engine::state::PlayerView) -> ObservationBuffers {
    let dex = dex();
    let encoder = Encoder::new(&dex).unwrap();
    let mut out = ObservationBuffers::default();
    encoder.encode_into(view, &mut out).unwrap();
    out
}

#[test]
fn fixed_roles_and_unknown_fields_are_masked_zero() {
    let view = view();
    let out = encoded(&view);
    assert_eq!(out.schema_version, SCHEMA_VERSION);
    for (i, role) in token_layout().into_iter().enumerate() {
        let expected = match role {
            TokenRole::Global => 1,
            TokenRole::Field => 2,
            TokenRole::Side(_) => 3,
            TokenRole::Pokemon(_) => 4,
            TokenRole::Move(_, _) => 5,
            TokenRole::Event(_) | TokenRole::Padding => 0,
        };
        assert_eq!(out.tokens[i].categories[C::Role as usize], expected);
        assert_eq!(out.token_mask[i], expected != 0);
        if expected == 0 {
            assert_eq!(out.tokens[i], TokenFeatures::default());
        }
    }
    for entity in 6..12 {
        let row = &out.tokens[4 + entity];
        assert_eq!(row.categories[C::Entity as usize], entity as u16 + 1);
        assert!(row.category_known[C::Species as usize]);
        for field in [C::Ability, C::Item, C::PreviousItem, C::Status, C::Nature] {
            assert!(!row.category_known[field as usize]);
            assert_eq!(row.categories[field as usize], 0);
        }
        for field in [F::Hp, F::MaxHp, F::Stats, F::Points, F::Ivs] {
            assert!(!row.float_known[field as usize]);
            assert_eq!(row.floats[field as usize], 0.0);
        }
        for slot in 0..4 {
            let row = &out.tokens[16 + 4 * entity + slot];
            assert!(!row.category_known[C::Move as usize]);
            assert!(!row.float_known[F::Pp as usize]);
            assert!(!row.float_known[F::Power as usize]);
        }
    }
    for row in &out.tokens {
        for (value, known) in row.categories.iter().zip(row.category_known) {
            if !known {
                assert_eq!(*value, 0);
            }
        }
        for (value, known) in row.floats.iter().zip(row.float_known) {
            if !known {
                assert_eq!(*value, 0.0);
            }
        }
        for (value, known) in row.flags.iter().zip(row.flag_known) {
            if !known {
                assert!(!value);
            }
        }
    }
}

#[test]
fn hidden_world_changes_and_unknown_stale_payloads_do_not_change_encoding() {
    let dex = dex();
    let teams = teams();
    let original = BattleState::reset(&dex, [&teams[0], &teams[1]], [1, 2, 3, 4], [0, 1]).unwrap();
    let mut hidden = teams[1].clone();
    hidden.id = "private source changed".into();
    for set in &mut hidden.members {
        set.points = [1, 0, 0, 0, 0, 0];
        set.ivs = [0; 6];
        set.nature = dex.id("natures", "serious").unwrap();
        set.item = 0;
        set.ability = *dex.legal_abilities_by_species[set.species as usize]
            .last()
            .unwrap();
        set.moves = vec![dex.legal_moves_by_species[set.species as usize][0]];
    }
    let changed = BattleState::reset(&dex, [&teams[0], &hidden], [1, 2, 3, 4], [1, 0]).unwrap();
    assert_ne!(
        original.observe(SideId::P2).own,
        changed.observe(SideId::P2).own
    );
    assert_eq!(
        encoded(&original.observe(SideId::P1)),
        encoded(&changed.observe(SideId::P1))
    );
    let mut stale = original.observe(SideId::P1);
    let expected = encoded(&stale);
    for mon in &mut stale.knowledge.pokemon[6..] {
        for value in [
            &mut mon.ability,
            &mut mon.item,
            &mut mon.previous_item,
            &mut mon.status,
        ] {
            assert!(!value.known);
            value.value = u16::MAX;
        }
        for value in &mut mon.current_moves {
            assert!(!value.known);
            value.value = u16::MAX;
        }
        if !mon.gender.known {
            mon.gender.value = u8::MAX;
        }
        mon.selected.value = true;
        mon.health.value = HealthDisplay {
            numerator: u16::MAX,
            denominator: 0,
            boundary_color: u8::MAX,
        };
    }
    stale.outcome.terminated = true;
    stale.outcome.winner = Some(SideId::P2);
    stale.outcome.operational_error = Some("never policy input".into());
    assert_eq!(expected, encoded(&stale));
}

#[test]
fn own_actual_state_pp_and_original_moves_survive_current_slot_changes() {
    let mut view = view();
    view.own.pokemon[0].hp -= 7;
    view.own.pokemon[0].boosts = [1, -2, 3, -4, 5, -6, 6];
    view.own.pokemon[0].moves[0].pp = 1;
    view.own.pokemon[0].moves[0].disabled = true;
    let original = view.own.pokemon[0].base_moves.clone();
    let other_move = dex().id("moves", "thunderbolt").unwrap();
    view.own.pokemon[0].moves[0].id = other_move;
    let out = encoded(&view);
    let own = &view.own.pokemon[0];
    let row = &out.tokens[4];
    assert_eq!(row.floats[F::Hp as usize], f32::from(own.hp) / 65535.0);
    assert_eq!(
        row.floats[F::MaxHp as usize],
        f32::from(own.stats[0]) / 65535.0
    );
    assert_eq!(
        row.floats[F::HpFraction as usize],
        f32::from(own.hp) / f32::from(own.stats[0])
    );
    for i in 0..6 {
        assert_eq!(
            row.floats[F::Stats as usize + i],
            f32::from(own.stats[i]) / 65535.0
        );
        assert_eq!(
            row.floats[F::Points as usize + i],
            f32::from(own.points[i]) / 32.0
        );
        assert_eq!(
            row.floats[F::Ivs as usize + i],
            f32::from(own.ivs[i]) / 31.0
        );
    }
    for i in 0..7 {
        assert_eq!(
            row.floats[F::Boosts as usize + i],
            f32::from(own.boosts[i]) / 6.0
        );
    }
    let m = &out.tokens[16];
    assert_eq!(m.categories[C::Move as usize], other_move);
    assert_eq!(m.floats[F::Pp as usize], 1.0 / 255.0);
    assert!(m.flags[B::MoveDisabled as usize]);
    let range = out.base_move_ranges[4];
    let history = &out.base_moves[range.start..range.start + range.len];
    assert_eq!(
        history.iter().map(|m| m.id).collect::<Vec<_>>(),
        original.iter().map(|m| m.id).collect::<Vec<_>>()
    );
    assert_eq!(history[0].pp, f32::from(original[0].pp) / 255.0);
    assert!(!row.flags[B::PrivateEffectsAvailable as usize]);
    assert!(row.flag_known[B::PrivateEffectsAvailable as usize]);
}

#[test]
fn effects_repertoire_and_dynamic_types_have_no_token_cap_and_unknown_values_are_zero() {
    let dex = dex();
    let mut view = view();
    let mon = &mut view.knowledge.pokemon[6];
    mon.revealed_move_repertoire = (1..=17).collect();
    mon.types = (1..=8).collect();
    for id in 1..=60 {
        mon.effects.insert(
            id,
            EffectKnowledge {
                present: id % 2 == 0,
                duration: Known {
                    value: if id % 2 == 0 { 7 } else { 12345 },
                    known: id % 2 == 0,
                },
                stacks: Known {
                    value: -3,
                    known: true,
                },
                source: Known {
                    value: if id % 2 == 0 { 4 } else { 255 },
                    known: id % 2 == 0,
                },
            },
        );
    }
    view.knowledge.field.insert(
        dex.effects.rain,
        EffectKnowledge {
            present: true,
            duration: Known::new(3),
            ..Default::default()
        },
    );
    view.knowledge.sides[1].insert(
        dex.effects.tailwind,
        EffectKnowledge {
            present: true,
            stacks: Known::new(1),
            ..Default::default()
        },
    );
    let out = encoded(&view);
    let range = out.effect_ranges[10];
    assert_eq!(range.len, 60);
    let effects = &out.effects[range.start..range.start + range.len];
    assert_eq!(effects.last().unwrap().id, 60);
    for effect in effects {
        assert_eq!(effect.stacks, -3.0 / 32768.0);
        if effect.id % 2 == 0 {
            assert_eq!(effect.duration, 7.0 / 32768.0);
            assert_eq!(effect.source, 4);
        } else {
            assert_eq!(effect.duration, 0.0);
            assert_eq!(effect.source, 0);
            assert!(!effect.duration_known && !effect.source_known);
        }
    }
    let range = out.repertoire_ranges[10];
    assert_eq!(
        &out.repertoire[range.start..range.start + range.len],
        &(1..=17).collect::<Vec<_>>()
    );
    let range = out.type_ranges[10];
    assert_eq!(range.len, 8);
    assert_eq!(out.effect_ranges[1].len, 1);
    assert_eq!(out.effect_ranges[3].len, 1);
}

#[test]
fn public_health_boundary_and_event_order_are_preserved_without_exact_hp() {
    let mut view = view();
    let health = HealthDisplay {
        numerator: 20,
        denominator: 100,
        boundary_color: 2,
    };
    view.knowledge.pokemon[8].health = Known::new(health);
    view.knowledge.pokemon[8].ability = Known::new(dex().id("abilities", "levitate").unwrap());
    view.knowledge.pokemon[8].current_moves[2] = Known::new(dex().id("moves", "surf").unwrap());
    view.knowledge.events = vec![
        SemanticEvent {
            kind: EventKind::Move,
            subject: 8,
            target: Some(2),
            effect: dex().id("moves", "surf").unwrap(),
            effect_kind: EffectKind::Move,
            value: 0,
            health: None,
        },
        SemanticEvent {
            kind: EventKind::Damage,
            subject: 2,
            target: None,
            effect: 0,
            effect_kind: EffectKind::None,
            value: -7,
            health: Some(health),
        },
    ];
    let out = encoded(&view);
    let row = &out.tokens[12];
    assert_eq!(row.categories[C::Entity as usize], 9);
    assert_eq!(row.floats[F::HpFraction as usize], 0.2);
    assert_eq!(row.categories[C::BoundaryColor as usize], 2);
    assert!(!row.float_known[F::Hp as usize] && !row.float_known[F::MaxHp as usize]);
    assert!(out.tokens[16 + 8 * 4 + 2].category_known[C::Move as usize]);
    assert!(!out.tokens[16 + 8 * 4 + 2].float_known[F::Pp as usize]);
    assert_eq!(out.tokens[64].categories[C::Entity as usize], 9);
    assert_eq!(out.tokens[64].categories[C::EventTarget as usize], 3);
    assert_eq!(out.tokens[65].categories[C::EventKind as usize], 3);
    assert_eq!(
        out.tokens[65].floats[F::EventChange as usize],
        -7.0 / 2147483648.0
    );
    assert!(!out.token_mask[66]);
}

#[test]
fn output_reuse_clears_stale_padding_and_retains_ragged_capacity() {
    let dex = dex();
    let encoder = Encoder::new(&dex).unwrap();
    let mut view = view();
    let mut out = ObservationBuffers::default();
    for id in 1..=80 {
        view.knowledge.pokemon[6].effects.insert(
            id,
            EffectKnowledge {
                present: true,
                ..Default::default()
            },
        );
    }
    view.knowledge.pokemon[6].revealed_move_repertoire = (1..=40).collect();
    view.knowledge.events.push(SemanticEvent {
        kind: EventKind::Faint,
        subject: 6,
        target: None,
        effect: 0,
        effect_kind: EffectKind::None,
        value: 0,
        health: None,
    });
    encoder.encode_into(&view, &mut out).unwrap();
    let capacities = (
        out.effects.capacity(),
        out.repertoire.capacity(),
        out.types.capacity(),
        out.move_effects.capacity(),
        out.base_moves.capacity(),
    );
    let pointers = (
        out.effects.as_ptr(),
        out.repertoire.as_ptr(),
        out.types.as_ptr(),
        out.move_effects.as_ptr(),
        out.base_moves.as_ptr(),
    );
    view.knowledge.pokemon[6].effects.clear();
    view.knowledge.pokemon[6].revealed_move_repertoire.clear();
    view.knowledge.events.clear();
    encoder.encode_into(&view, &mut out).unwrap();
    assert_eq!(
        capacities,
        (
            out.effects.capacity(),
            out.repertoire.capacity(),
            out.types.capacity(),
            out.move_effects.capacity(),
            out.base_moves.capacity()
        )
    );
    assert_eq!(
        pointers,
        (
            out.effects.as_ptr(),
            out.repertoire.as_ptr(),
            out.types.as_ptr(),
            out.move_effects.as_ptr(),
            out.base_moves.as_ptr()
        )
    );
    assert!(out.effects.is_empty() && out.repertoire.is_empty());
    assert_eq!(out.effect_ranges[10].len, 0);
    assert_eq!(out.repertoire_ranges[10].len, 0);
    assert!(!out.token_mask[64]);
    assert_eq!(out.tokens[64], TokenFeatures::default());
}

#[test]
fn malformed_known_ids_health_sources_and_event_windows_fail_without_replacing_output() {
    let dex = dex();
    let encoder = Encoder::new(&dex).unwrap();
    let original = view();
    let mut out = encoded(&original);
    let saved = out.clone();
    let mut invalid = original.clone();
    invalid.knowledge.pokemon[6].ability = Known::new(u16::MAX);
    assert!(encoder.encode_into(&invalid, &mut out).is_err());
    assert_eq!(out, saved);
    let mut invalid = original.clone();
    invalid.knowledge.pokemon[6].health = Known::new(HealthDisplay {
        numerator: 2,
        denominator: 1,
        boundary_color: 0,
    });
    assert!(encoder.encode_into(&invalid, &mut out).is_err());
    assert_eq!(out, saved);
    let mut invalid = original.clone();
    invalid.knowledge.field.insert(
        dex.effects.rain,
        EffectKnowledge {
            source: Known::new(12),
            ..Default::default()
        },
    );
    assert!(encoder.encode_into(&invalid, &mut out).is_err());
    assert_eq!(out, saved);
    let mut invalid = original;
    invalid.knowledge.events = vec![
        SemanticEvent {
            kind: EventKind::Faint,
            subject: 6,
            target: None,
            effect: 0,
            effect_kind: EffectKind::None,
            value: 0,
            health: None
        };
        25
    ];
    assert!(encoder.encode_into(&invalid, &mut out).is_err());
    assert_eq!(out, saved);
}

#[test]
fn cold_metadata_validation_rejects_bad_catalogue_and_ratio_and_move_effects_are_complete() {
    let shared = dex();
    let encoder = Encoder::new(&shared).unwrap();
    assert!(encoder.matches_dex(&shared));
    assert!(!encoder.matches_dex(&(*shared).clone()));
    let mut broken = (*dex()).clone();
    broken.names.get_mut("moves").unwrap().pop();
    assert!(Encoder::new(&broken).is_err());
    let mut broken = (*dex()).clone();
    broken.moves[1].hit.heal = Some([1, 0]);
    assert!(Encoder::new(&broken).is_err());
    let dex = dex();
    let mut view = view();
    let flame = dex.id("moves", "flamethrower").unwrap();
    let recoil = dex.id("moves", "flareblitz").unwrap();
    view.own.pokemon[0].moves[0].id = flame;
    view.own.pokemon[0].moves[1].id = recoil;
    let out = encoded(&view);
    let range = out.move_effect_ranges[16];
    let effects = &out.move_effects[range.start..range.start + range.len];
    assert!(
        effects
            .iter()
            .any(|e| e.kind == 3 && e.chance == 0.1 && e.status == dex.effects.burn)
    );
    assert_eq!(dex.moves[recoil as usize].recoil, Some([33, 100]));
    assert_eq!(out.tokens[17].floats[F::Recoil as usize], 0.33);
    assert!(out.tokens[17].float_known[F::Recoil as usize]);
}

#[test]
fn audience_filtered_event_integers_remain_unknown_while_hp_and_public_boosts_stay_known() {
    let dex = dex();
    let mut view = view();
    let kinds = [
        EventKind::Damage,
        EventKind::Heal,
        EventKind::SideEffectStart,
        EventKind::FieldEffectStart,
    ];
    let health = HealthDisplay {
        numerator: 50,
        denominator: 100,
        boundary_color: 3,
    };
    for kind in kinds {
        let (effect, effect_kind) = match kind {
            EventKind::SideEffectStart => (dex.effects.tailwind, EffectKind::Condition),
            EventKind::FieldEffectStart => (dex.effects.rain, EffectKind::Condition),
            _ => (0, EffectKind::None),
        };
        view.knowledge.events.push(SemanticEvent {
            kind,
            subject: 1,
            target: None,
            effect,
            effect_kind,
            value: 7,
            health: Some(health),
        });
        view.knowledge.events.push(SemanticEvent {
            kind,
            subject: 7,
            target: None,
            effect,
            effect_kind,
            value: 0,
            health: Some(health),
        });
    }
    view.knowledge.events.push(SemanticEvent {
        kind: EventKind::Boost,
        subject: 7,
        target: None,
        effect: 2,
        effect_kind: EffectKind::Stat,
        value: 1,
        health: None,
    });
    let out = encoded(&view);
    for i in 0..4 {
        let own = &out.tokens[64 + i * 2];
        let opponent = &out.tokens[65 + i * 2];
        assert!(own.float_known[F::EventChange as usize]);
        assert_eq!(own.floats[F::EventChange as usize], 7.0 / 2147483648.0);
        assert!(!opponent.float_known[F::EventChange as usize]);
        assert_eq!(opponent.floats[F::EventChange as usize], 0.0);
        assert!(opponent.float_known[F::HpFraction as usize]);
        assert_eq!(opponent.floats[F::HpFraction as usize], 0.5);
        assert_eq!(opponent.categories[C::BoundaryColor as usize], 3);
    }
    assert!(out.tokens[72].float_known[F::EventChange as usize]);
    assert_eq!(
        out.tokens[72].floats[F::EventChange as usize],
        1.0 / 2147483648.0
    );
}
