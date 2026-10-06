use pa3_engine::{
    actions::*,
    assets::{Dex, Target},
    knowledge::*,
    rng::BattleRng,
    stats,
};
use serde_json::Value;
use std::{path::Path, sync::OnceLock};

fn data() -> &'static Dex {
    static DATA: OnceLock<Dex> = OnceLock::new();
    DATA.get_or_init(|| Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap())
}
fn fixtures() -> Value {
    serde_json::from_str(include_str!("../data/reference-fixtures.json")).unwrap()
}

#[test]
fn rng_matches_reference_4096_draws_and_restores() {
    for case in fixtures()["rng"].as_array().unwrap() {
        let seed: [u16; 4] = serde_json::from_value(case["seed"].clone()).unwrap();
        let mut rng = BattleRng::new(seed);
        for draw in case["draws"].as_array().unwrap() {
            assert_eq!(u64::from(rng.next_u32()), draw.as_u64().unwrap());
        }
        assert_eq!(
            rng.seed().map(|x| x.to_string()).join(","),
            case["final_seed"].as_str().unwrap()
        );
        let mut restored: BattleRng =
            serde_json::from_str(&serde_json::to_string(&rng).unwrap()).unwrap();
        assert_eq!(rng.next_u32(), restored.next_u32());
        assert_eq!(rng, restored);
    }
}

#[test]
fn all_training_sets_and_legal_species_stats_match_reference() {
    let dex = data();
    for case in fixtures()["stats"].as_array().unwrap() {
        let s = &dex.species[case["species"].as_u64().unwrap() as usize];
        let points = serde_json::from_value(case["points"].clone()).unwrap();
        let expected: [u16; 6] = serde_json::from_value(case["stats"].clone()).unwrap();
        let nature = dex.natures[case["nature"].as_u64().unwrap() as usize];
        assert_eq!(
            stats::champions_stats(s.base_stats, points, nature, s.max_hp),
            expected,
            "species {}",
            s.id
        );
    }
}

#[test]
fn pp_and_fixed_point_rounding_match_reference() {
    let dex = data();
    for case in fixtures()["pp"].as_array().unwrap() {
        let m = &dex.moves[case["move_id"].as_u64().unwrap() as usize];
        assert_eq!(
            u64::from(stats::champions_pp(m.pp, m.no_pp_boosts)),
            case["pp"].as_u64().unwrap(),
            "move {}",
            m.id
        );
    }
    for c in fixtures()["modifiers"].as_array().unwrap() {
        assert_eq!(
            u64::from(stats::modify(
                c["value"].as_u64().unwrap() as u32,
                c["modifier"].as_u64().unwrap() as u32
            )),
            c["result"].as_u64().unwrap()
        );
    }
}

#[test]
fn recoil_fraction_rounding_matches_reference_for_all_hp_damage_values() {
    for c in fixtures()["recoil_rounding"].as_array().unwrap() {
        let fraction = serde_json::from_value(c["fraction"].clone()).unwrap();
        assert_eq!(
            u64::from(stats::round_fraction(c["damage"].as_u64().unwrap() as u32, fraction).max(1)),
            c["amount"].as_u64().unwrap(),
            "{c}"
        );
    }
}

#[test]
fn every_target_class_matches_reference_in_both_doubles_slots() {
    for c in fixtures()["targeting"].as_array().unwrap() {
        let target = Target::parse(c["target"].as_str().unwrap()).unwrap();
        let slot = c["slot"].as_u64().unwrap() as u8;
        let expected: Vec<i8> = serde_json::from_value(c["locations"].clone()).unwrap();
        assert_eq!(
            [-2, -1, 0, 1, 2]
                .into_iter()
                .filter(|loc| target.valid_location(slot, *loc))
                .collect::<Vec<_>>(),
            expected
        );
    }
}

fn normal() -> Request {
    let slot = SlotRequest {
        present: true,
        can_mega: true,
        moves: (0..4)
            .map(|slot| MoveChoice {
                id: slot as u16 + 1,
                slot,
                target: Target::Normal,
                disabled: false,
                pp: 10,
            })
            .collect(),
        ..Default::default()
    };
    Request {
        kind: RequestKind::Normal,
        slots: [slot.clone(), slot],
        bench: vec![2, 3],
        preview_roster: vec![],
    }
}

#[test]
fn preview_has_all_360_distinct_ordered_selections() {
    fn visit(r: &Request, p: &mut Vec<AtomicAction>, n: &mut usize) {
        if p.len() == 4 {
            r.validate_joint(p).unwrap();
            *n += 1;
            return;
        }
        for a in r.candidates(p).unwrap() {
            p.push(a);
            visit(r, p, n);
            p.pop();
        }
    }
    let mut count = 0;
    visit(&Request::preview(), &mut vec![], &mut count);
    assert_eq!(count, 360);
}

#[test]
fn joint_masks_prevent_duplicate_bench_and_mega_but_allow_ally_targets() {
    let r = normal();
    let first = r.candidates(&[]).unwrap();
    assert_eq!(first.len(), 26); // 4 moves * 3 targets * 2 resource states + 2 switches.
    assert!(first.iter().any(|a| a.target_location == -2));
    for a in first {
        for b in r.candidates(&[a]).unwrap() {
            assert!(!(a.resource == Resource::Mega && b.resource == Resource::Mega));
            assert!(
                !(a.kind == ActionKind::Switch
                    && b.kind == ActionKind::Switch
                    && a.switch_destination == b.switch_destination)
            );
            r.validate_joint(&[a, b]).unwrap();
        }
    }
    let mut trapped = r.clone();
    trapped.slots[0].trapped = true;
    assert!(
        !trapped
            .candidates(&[])
            .unwrap()
            .iter()
            .any(|a| a.kind == ActionKind::Switch)
    );
    let mut maybe = r.clone();
    maybe.slots[0].maybe_trapped = true;
    assert_eq!(r.candidates(&[]).unwrap(), maybe.candidates(&[]).unwrap());
}

#[test]
fn forced_replacement_can_fill_either_slot_with_last_reserve() {
    let mut r = normal();
    r.kind = RequestKind::Replacement;
    r.bench = vec![3];
    for s in &mut r.slots {
        s.requires_replacement = true;
    }
    let first = r.candidates(&[]).unwrap();
    assert_eq!(first.len(), 2);
    for a in first {
        let second = r.candidates(&[a]).unwrap();
        assert_eq!(second.len(), 1);
        assert_ne!(a.kind, second[0].kind);
        r.validate_joint(&[a, second[0]]).unwrap();
    }
    r.slots[0].requires_replacement = false;
    let first = r.candidates(&[]).unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].own_slot, 1);
}

#[test]
fn struggle_pass_and_invalid_prefixes_are_distinct() {
    let mut r = normal();
    for m in &mut r.slots[0].moves {
        m.pp = 0;
    }
    r.slots[0].can_mega = false;
    r.slots[0].trapped = true;
    r.slots[1].present = false;
    let first = r.candidates(&[]).unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].kind, ActionKind::Move);
    assert_eq!(first[0].move_slot, NO_SLOT);
    assert_eq!(r.candidates(&first).unwrap(), vec![AtomicAction::pass(1)]);
    assert!(r.candidates(&[AtomicAction::pass(0)]).is_err());
}

#[test]
fn knowledge_is_persistent_beyond_recent_events_and_four_move_slots() {
    let mut k = Knowledge::default();
    k.preview(6, 10, vec![2, 4], Known::default()).unwrap();
    for id in 1..=30 {
        k.apply(SemanticEvent {
            kind: EventKind::Move,
            subject: 6,
            target: Some(0),
            effect: id,
            effect_kind: pa3_engine::knowledge::EffectKind::Move,
            value: 0,
            health: None,
        })
        .unwrap();
    }
    assert_eq!(k.events.len(), 24);
    assert_eq!(k.pokemon[6].revealed_move_repertoire.len(), 30);
    k.replace_current_moves(6, [Known::new(100); 4]).unwrap();
    assert_eq!(k.pokemon[6].revealed_move_repertoire.len(), 31);
    assert!(!k.pokemon[6].ability.known);
    assert!(!k.pokemon[6].item.known);
    assert!(!k.pokemon[6].selected.known);
    let restored: Knowledge = serde_json::from_str(&serde_json::to_string(&k).unwrap()).unwrap();
    assert_eq!(k, restored);
    assert_eq!(
        token_layout()
            .iter()
            .filter(|r| **r != TokenRole::Padding)
            .count(),
        ACTIVE_TOKENS
    );
}

#[test]
fn regulation_scope_includes_uncollected_species_and_mega_forms() {
    let scope: Value = serde_json::from_str(include_str!("../data/scope.json")).unwrap();
    assert!(
        scope["unresolved_starting_candidates"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert_eq!(scope["starting_species"].as_array().unwrap().len(), 293);
    assert_eq!(scope["mega_forms"].as_array().unwrap().len(), 82);
    assert!(
        scope["starting_species"]
            .as_array()
            .unwrap()
            .iter()
            .any(|s| s["species"] == "ditto")
    );
    assert!(
        !scope["starting_species"]
            .as_array()
            .unwrap()
            .iter()
            .any(|s| s["species"] == "mewtwo")
    );
    assert!(
        scope["mega_forms"]
            .as_array()
            .unwrap()
            .contains(&Value::from("charizardmegax"))
    );
    assert_eq!(scope["training_ready"], false);
}

#[test]
fn event_ordering_and_rng_consumption_match_reference() {
    use pa3_engine::queue::{Priority, speed_sort};
    for case in fixtures()["ordering"].as_array().unwrap() {
        let mut entries = case["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| {
                (
                    x["id"].as_u64().unwrap(),
                    Priority {
                        order: x["order"].as_u64().unwrap() as u32,
                        priority: (x["priority"].as_f64().unwrap() * 10000.0).round() as i32,
                        speed: x["speed"].as_i64().unwrap() as i32,
                        sub_order: x["subOrder"].as_i64().unwrap() as i32,
                        effect_order: x["effectOrder"].as_u64().unwrap() as u32,
                    },
                )
            })
            .collect::<Vec<_>>();
        let mut rng = BattleRng::new(serde_json::from_value(case["seed"].clone()).unwrap());
        speed_sort(&mut entries, &mut rng, |x| x.1);
        assert_eq!(
            serde_json::to_value(entries.iter().map(|x| x.0).collect::<Vec<_>>()).unwrap(),
            case["order"]
        );
        assert_eq!(
            rng.seed().map(|x| x.to_string()).join(","),
            case["final_seed"]
        );
    }
}

#[test]
fn target_ordered_handlers_match_reference_without_speed_tie_randomization() {
    use pa3_engine::queue::Priority;
    for c in fixtures()["left_to_right_ordering"].as_array().unwrap() {
        let mut entries: Vec<_> = c["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| {
                (
                    x["id"].as_u64().unwrap(),
                    x["index"].as_u64().unwrap() as usize,
                    Priority {
                        order: x["order"].as_u64().unwrap() as u32,
                        priority: (x["priority"].as_f64().unwrap() * 10000.0).round() as i32,
                        speed: x["speed"].as_i64().unwrap() as i32,
                        sub_order: x["subOrder"].as_i64().unwrap() as i32,
                        effect_order: x["effectOrder"].as_u64().unwrap() as u32,
                    },
                )
            })
            .collect();
        entries.sort_by(|a, b| a.2.compare_left_to_right(a.1, &b.2, b.1));
        assert_eq!(
            serde_json::to_value(entries.iter().map(|x| x.0).collect::<Vec<_>>()).unwrap(),
            c["order"]
        );
    }
}

#[test]
fn champions_public_hp_includes_boundary_colours_and_flooring() {
    for case in fixtures()["health"].as_array().unwrap() {
        let hp = case["hp"].as_u64().unwrap() as u16;
        let display = public_health(hp, case["max_hp"].as_u64().unwrap() as u16);
        let suffix = ["", "r", "y", "g"][display.boundary_color as usize];
        let rendered = if hp == 0 {
            "0 fnt".into()
        } else {
            format!("{}/100{suffix}", display.numerator)
        };
        assert_eq!(rendered, case["shared"]);
    }
}

#[test]
fn isolated_damage_kernel_matches_reference_modifiers_and_rng() {
    for case in fixtures()["damage_kernel"].as_array().unwrap() {
        let mut rng = BattleRng::new(serde_json::from_value(case["seed"].clone()).unwrap());
        let input = serde_json::from_value(case["input"].clone()).unwrap();
        let damage = pa3_engine::damage::calculate(input, &mut rng).unwrap();
        assert_eq!(
            u64::from(damage),
            case["damage"].as_u64().unwrap(),
            "{input:?}"
        );
        assert_eq!(
            rng.seed().map(|x| x.to_string()).join(","),
            case["final_seed"]
        );
    }
}

#[test]
fn champions_action_speed_and_trick_room_match_reference_boundaries() {
    for case in fixtures()["action_speed"].as_array().unwrap() {
        assert_eq!(
            stats::action_speed(
                case["modified_speed"].as_u64().unwrap() as u32,
                case["trick_room"].as_bool().unwrap()
            ),
            case["result"].as_i64().unwrap() as i32
        );
    }
}
