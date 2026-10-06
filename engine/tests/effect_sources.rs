use pa3_engine::knowledge::{EffectKind, EventKind, Knowledge, Known, SemanticEvent};

fn event(kind: EventKind, subject: u8, target: Option<u8>) -> SemanticEvent {
    SemanticEvent {
        kind,
        subject,
        target,
        effect: 42,
        effect_kind: EffectKind::Condition,
        value: 0,
        health: None,
    }
}

#[test]
fn only_explicit_public_effect_start_sources_become_known_and_end_clears_them() {
    let mut knowledge = Knowledge::default();
    knowledge
        .apply(event(EventKind::EffectStart, 6, None))
        .unwrap();
    let effect = &knowledge.pokemon[6].effects[&42];
    assert!(effect.present);
    assert!(!effect.source.known);
    assert!(!effect.duration.known && !effect.stacks.known);
    knowledge
        .apply(event(EventKind::EffectStart, 7, Some(0)))
        .unwrap();
    assert_eq!(knowledge.pokemon[7].effects[&42].source, Known::new(0));
    assert!(!knowledge.pokemon[7].effects[&42].duration.known);
    // An event with no source carries no new inference or hidden lookup.
    knowledge
        .apply(event(EventKind::EffectStart, 7, None))
        .unwrap();
    assert_eq!(knowledge.pokemon[7].effects[&42].source, Known::new(0));
    knowledge
        .apply(event(EventKind::EffectEnd, 7, None))
        .unwrap();
    assert!(!knowledge.pokemon[7].effects.contains_key(&42));
    knowledge
        .apply(event(EventKind::EffectStart, 7, None))
        .unwrap();
    assert!(!knowledge.pokemon[7].effects[&42].source.known);
}

#[test]
fn ordinary_targets_do_not_set_effect_sources_and_invalid_source_indices_are_atomic() {
    let mut knowledge = Knowledge::default();
    knowledge
        .apply(event(EventKind::EffectStart, 6, None))
        .unwrap();
    knowledge
        .apply(event(EventKind::Status, 6, Some(3)))
        .unwrap();
    assert!(!knowledge.pokemon[6].effects[&42].source.known);
    let before = knowledge.clone();
    assert!(
        knowledge
            .apply(event(EventKind::EffectStart, 6, Some(12)))
            .is_err()
    );
    assert_eq!(knowledge, before);
}
