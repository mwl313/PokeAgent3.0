//! Primary callback primitives, not legal full battle/acquisition certification.
//! In particular Galvanize/Normalize, Z/Max flags and excluded move preparation
//! are synthetic contexts; the independent legal turn corpus covers holders.
use super::*;
use crate::state::Team;
use std::path::Path;

#[derive(Deserialize)]
struct Corpus {
    oracle_commit: String,
    cases: Vec<Probe>,
}
#[derive(Deserialize)]
struct Probe {
    ability: String,
    id: String,
    #[serde(rename = "type")]
    incoming_type: String,
    category: Category,
    sound: bool,
    is_z: bool,
    is_max: bool,
    marker_mode: String,
    power: u16,
    weather: String,
    expected_type: String,
    expected_marker: bool,
    expected_action_power: u16,
    expected_power: u32,
}
#[derive(Deserialize)]
struct Turns {
    fixtures: Vec<Fixture>,
}
#[derive(Deserialize)]
struct Fixture {
    teams: [Team; 2],
    seed: [u16; 4],
    steps: Vec<Step>,
}
#[derive(Deserialize)]
struct Step {
    side: SideId,
    actions: Vec<AtomicAction>,
}

#[test]
fn all_seven_type_callbacks_match_pinned_primary_primitives() {
    let dex = Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap();
    let corpus: Corpus =
        serde_json::from_str(include_str!("../../tests/type_conversion_fixtures.json")).unwrap();
    assert_eq!(corpus.oracle_commit, crate::ORACLE_COMMIT);
    assert_eq!(corpus.cases.len(), 203);
    let turns: Turns = serde_json::from_str(&corpus::corpus_json(&corpus::data_dir())).unwrap();
    let fixture = &turns.fixtures[0];
    let mut base = BattleState::reset(
        &dex,
        [&fixture.teams[0], &fixture.teams[1]],
        fixture.seed,
        [0, 1],
    )
    .unwrap();
    for step in fixture.steps.iter().take(2) {
        base.step(&dex, step.side, &step.actions).unwrap();
    }
    let actor = Entity {
        side: 0,
        roster: base.sides[0].active[0].unwrap(),
    };
    let target = Entity {
        side: 1,
        roster: base.sides[1].active[0].unwrap(),
    };
    for c in corpus.cases {
        let mut state = base.clone();
        state.field.clear();
        for side in &mut state.sides {
            for mon in &mut side.pokemon {
                mon.ability = dex.id("abilities", "shellarmor").unwrap();
                mon.item = 0;
                mon.volatiles.clear();
            }
        }
        state.mon_mut(actor).ability = dex.id("abilities", &c.ability).unwrap();
        state.mon_mut(actor).item = 0;
        state.mon_mut(target).ability = dex.id("abilities", "shellarmor").unwrap();
        state.mon_mut(target).item = 0;
        if !c.weather.is_empty() {
            state
                .start_weather(
                    &dex,
                    actor,
                    dex.id("conditions", &c.weather).unwrap(),
                    false,
                )
                .unwrap();
        }
        // Clone only in tests to construct isolated incoming callback contexts.
        let mut data = dex.moves[dex.id("moves", "hypervoice").unwrap() as usize].clone();
        data.move_type = dex.id("types", &c.incoming_type).unwrap();
        data.category = c.category;
        data.sound = c.sound;
        data.power = c.power;
        data.is_z = c.is_z;
        data.is_max = c.is_max;
        data.conversion_excluded = matches!(
            c.id.as_str(),
            "judgment"
                | "multiattack"
                | "naturalgift"
                | "revelationdance"
                | "technoblast"
                | "terrainpulse"
                | "weatherball"
        );
        data.normalize_excluded =
            data.conversion_excluded || matches!(c.id.as_str(), "hiddenpower" | "struggle");
        let behavior = match c.id.as_str() {
            "weatherball" => MoveBehavior::WeatherBall,
            "struggle" => MoveBehavior::Struggle,
            _ => MoveBehavior::Damage,
        };
        let seed = state.rng_seed();
        let views = [state.observe(SideId::P1), state.observe(SideId::P2)];
        let mut action = state.active_move(&dex, actor, &data, behavior);
        let marker = dex.effects.abilities[state.mon(actor).ability as usize];
        assert_eq!(
            action.type_changer_boosted,
            c.expected_marker.then_some(marker),
            "{} {}",
            c.ability,
            c.id
        );
        assert_eq!(
            action.power, c.expected_action_power,
            "{} {}",
            c.ability, c.id
        );
        assert_eq!(state.rng_seed(), seed);
        assert_eq!(
            [state.observe(SideId::P1), state.observe(SideId::P2)],
            views,
            "conversion must not disclose abilities"
        );
        if c.marker_mode == "mismatch" {
            action.type_changer_boosted = Some(if marker == Ability::Pixilate {
                Ability::Aerilate
            } else {
                Ability::Pixilate
            });
        }
        if c.marker_mode == "laterElectric" {
            action.move_type = dex.effects.electric;
        }
        let expected_type = if c.expected_type == "???" {
            0
        } else {
            dex.id("types", &c.expected_type).unwrap()
        };
        assert_eq!(action.move_type, expected_type, "{} {}", c.ability, c.id);
        let value = state
            .modify_value(
                &dex,
                ModifierEvent::BasePower,
                MoveContext {
                    actor,
                    target,
                    move_data: &action,
                    effectiveness: 0,
                    critical: false,
                },
                c.power as u32,
            )
            .unwrap();
        assert_eq!(
            value, c.expected_power,
            "{} {} {}",
            c.ability, c.id, c.marker_mode
        );
        assert_eq!(
            state.rng_seed(),
            seed,
            "single callback and distinct suborders do not draw tie RNG"
        );
    }
}

#[test]
fn cold_move_metadata_preserves_canonical_exclusions() {
    let dex = Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap();
    for name in [
        "judgment",
        "multiattack",
        "naturalgift",
        "revelationdance",
        "technoblast",
        "terrainpulse",
        "weatherball",
    ] {
        let m = &dex.moves[dex.id("moves", name).unwrap() as usize];
        assert!(m.conversion_excluded && m.normalize_excluded, "{name}");
    }
    for name in ["hiddenpower", "struggle"] {
        let m = &dex.moves[dex.id("moves", name).unwrap() as usize];
        assert!(!m.conversion_excluded && m.normalize_excluded, "{name}");
    }
    let m = &dex.moves[dex.id("moves", "hypervoice").unwrap() as usize];
    assert!(!m.conversion_excluded && !m.normalize_excluded);
    assert!(!m.is_z && !m.is_max);
}

#[test]
fn synchronize_failed_reflection_keeps_activation_but_allocates_no_source_status() {
    // Isolated native callback contexts: artificial airborne Synchronize holder
    // and a pending-faint source do not certify a complete legal battle path.
    // Pinned abilities.ts:onAfterSetStatus activates first, then calls
    // source.trySetStatus; HP0 and grounded Misty are that attempt's rejections.
    let dex = Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap();
    let turns: Turns = serde_json::from_str(&corpus::corpus_json(&corpus::data_dir())).unwrap();
    let f = &turns.fixtures[0];
    let mut base = BattleState::reset(&dex, [&f.teams[0], &f.teams[1]], f.seed, [0, 1]).unwrap();
    for s in f.steps.iter().take(2) {
        base.step(&dex, s.side, &s.actions).unwrap();
    }
    let holder = Entity {
        side: 0,
        roster: base.sides[0].active[0].unwrap(),
    };
    let source = Entity {
        side: 1,
        roster: base.sides[1].active[0].unwrap(),
    };
    for status in [dex.effects.burn, dex.effects.paralysis] {
        for misty in [false, true] {
            let mut state = base.clone();
            state.field.clear();
            for side in &mut state.sides {
                for mon in &mut side.pokemon {
                    mon.ability = dex.id("abilities", "shellarmor").unwrap();
                    mon.item = 0;
                    mon.status = 0;
                    mon.status_state = Default::default();
                    mon.types = vec![dex.effects.normal];
                    mon.volatiles.clear();
                }
            }
            state.mon_mut(holder).ability = dex.id("abilities", "synchronize").unwrap();
            if misty {
                state.mon_mut(holder).types = vec![dex.effects.flying];
                state
                    .start_terrain(&dex, holder, dex.effects.misty_terrain, false)
                    .unwrap();
                assert!(!state.grounded(&dex, holder));
                assert!(state.grounded(&dex, source));
            } else {
                state.mon_mut(source).hp = 0;
            }
            state.enable_trace().unwrap();
            let source_before = state.mon(source).clone();
            let counter = state.next_effect_order;
            let rng = state.rng_seed();
            let effect = crate::effects::HitEffect {
                status,
                ..Default::default()
            };
            assert!(
                state
                    .hit_effect(&dex, holder, source, &effect, false)
                    .unwrap()
            );
            assert_eq!(state.mon(holder).status, status);
            assert_eq!(state.mon(source), &source_before);
            assert_eq!(
                state.next_effect_order,
                counter + 1,
                "only accepted holder status gets order"
            );
            assert_eq!(state.rng_seed(), rng);
            let events = &state.trace.as_ref().unwrap().events[0];
            assert_eq!(
                events
                    .iter()
                    .filter(|e| e.event.kind == EventKind::Status)
                    .count(),
                1
            );
            assert_eq!(
                events
                    .iter()
                    .filter(
                        |e| e.event.kind == EventKind::Ability && e.event.subject == holder.roster
                    )
                    .count(),
                1
            );
            let status_index = events
                .iter()
                .position(|e| e.event.kind == EventKind::Status)
                .unwrap();
            let activation_index = events
                .iter()
                .position(|e| e.event.kind == EventKind::Ability)
                .unwrap();
            assert!(status_index < activation_index);
        }
    }
}

#[path = "../../test_support/corpus.rs"]
mod corpus;
