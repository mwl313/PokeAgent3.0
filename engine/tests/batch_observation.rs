use pa3_engine::{
    assets::Dex,
    batch::{BattleBatch, ResetSpec},
    observation::{Encoder, ObservationBatchBuffers, ObservationBuffers},
    state::{SideId, Team},
};
use std::{path::Path, sync::Arc};

fn assets() -> (Arc<Dex>, Arc<Vec<Team>>) {
    (
        Arc::new(Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap()),
        Arc::new(serde_json::from_str(include_str!("../data/training-teams.json")).unwrap()),
    )
}

#[test]
fn encoded_batches_cover_all_teams_and_match_serial_views_with_reused_buffers() {
    let (dex, teams) = assets();
    assert_eq!(teams.len(), 1136);
    let encoder = Encoder::new(&dex).unwrap();
    let mut batch = BattleBatch::new(dex.clone(), teams.clone(), 16).unwrap();
    let handles = batch
        .reset_batch(
            &(0..2048)
                .map(|i| ResetSpec {
                    team_a: i % teams.len(),
                    team_b: (17 * i + 1) % teams.len(),
                    seed: [1, 2, 3, i as u16],
                    role_map: if i % 2 == 0 { [0, 1] } else { [1, 0] },
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let requests: Vec<_> = handles
        .iter()
        .rev()
        .enumerate()
        .map(|(i, &handle)| (handle, if i % 2 == 0 { SideId::P1 } else { SideId::P2 }))
        .collect();
    let mut out = ObservationBatchBuffers::default();
    batch
        .observe_encoded_batch_into(&requests, &encoder, &mut out)
        .unwrap();
    assert_eq!(out.len(), 2048);
    let row_pointer = out.as_ptr();
    let type_pointer = out[0].types.as_ptr();
    let base_move_pointer = out[0].base_moves.as_ptr();
    let views = batch.observe_batch(&requests).unwrap();
    let mut serial = ObservationBuffers::default();
    for (view, encoded) in views.iter().zip(out.iter()) {
        encoder.encode_into(view, &mut serial).unwrap();
        assert_eq!(encoded, &serial);
    }
    batch
        .observe_encoded_batch_into(&requests, &encoder, &mut out)
        .unwrap();
    assert_eq!(out.as_ptr(), row_pointer);
    assert_eq!(out[0].types.as_ptr(), type_pointer);
    assert_eq!(out[0].base_moves.as_ptr(), base_move_pointer);
    let retained_type_pointer = out[17].types.as_ptr();
    let retained_base_move_pointer = out[17].base_moves.as_ptr();
    // Reordered sparse requests and two viewers of one environment are valid.
    let sparse = [
        (handles[17], SideId::P2),
        (handles[1], SideId::P1),
        (handles[17], SideId::P1),
    ];
    batch
        .observe_encoded_batch_into(&sparse, &encoder, &mut out)
        .unwrap();
    for (view, encoded) in batch.observe_batch(&sparse).unwrap().iter().zip(out.iter()) {
        encoder.encode_into(view, &mut serial).unwrap();
        assert_eq!(encoded, &serial);
    }
    batch
        .observe_encoded_batch_into(&[], &encoder, &mut out)
        .unwrap();
    assert!(out.is_empty());
    batch
        .observe_encoded_batch_into(&requests, &encoder, &mut out)
        .unwrap();
    assert_eq!(out.as_ptr(), row_pointer);
    assert_eq!(out[17].types.as_ptr(), retained_type_pointer);
    assert_eq!(out[17].base_moves.as_ptr(), retained_base_move_pointer);
}

#[test]
fn encoded_batch_preflight_preserves_output_on_stale_handles_or_wrong_dex() {
    let (dex, teams) = assets();
    let encoder = Encoder::new(&dex).unwrap();
    let mut batch = BattleBatch::new(dex.clone(), teams, 2).unwrap();
    let specs = [ResetSpec {
        team_a: 0,
        team_b: 1,
        seed: [1, 2, 3, 4],
        role_map: [0, 1],
    }; 2];
    let stale = batch.reset_batch(&specs).unwrap();
    let handles = batch.reset_batch(&specs).unwrap();
    let mut out = ObservationBatchBuffers::default();
    batch
        .observe_encoded_batch_into(&[(handles[0], SideId::P1)], &encoder, &mut out)
        .unwrap();
    let before = out.clone();
    assert!(
        batch
            .observe_encoded_batch_into(
                &[(handles[0], SideId::P1), (stale[1], SideId::P2)],
                &encoder,
                &mut out,
            )
            .is_err()
    );
    assert_eq!(out, before);
    let cloned_dex = (*dex).clone();
    let wrong_encoder = Encoder::new(&cloned_dex).unwrap();
    assert!(
        batch
            .observe_encoded_batch_into(&[(handles[0], SideId::P1)], &wrong_encoder, &mut out)
            .is_err()
    );
    assert_eq!(out, before);
}
