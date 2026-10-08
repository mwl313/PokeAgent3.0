//! Native preview-only encoder probe; no model, Python bridge or battle steps.
use pa3_engine::{
    assets::Dex,
    batch::{BattleBatch, Handle, ResetSpec},
    observation::{Encoder, ObservationBatchBuffers, ObservationBuffers},
    state::{SideId, Team},
};
use std::{hint::black_box, path::Path, sync::Arc, time::Instant};

const ENVIRONMENTS: usize = 2048;
const ITERATIONS: usize = 64;
const REPETITIONS: usize = 3;

fn group(
    dex: Arc<Dex>,
    teams: Arc<Vec<Team>>,
    workers: usize,
    count: usize,
    offset: usize,
) -> (BattleBatch, Vec<(Handle, SideId)>) {
    let mut batch = BattleBatch::new(dex, teams.clone(), workers).unwrap();
    let specs: Vec<_> = (offset..offset + count)
        .map(|i| ResetSpec {
            team_a: i % teams.len(),
            team_b: (17 * i + 1) % teams.len(),
            seed: [1, 2, 3, i as u16],
            role_map: if i % 2 == 0 { [0, 1] } else { [1, 0] },
        })
        .collect();
    let requests = batch
        .reset_batch(&specs)
        .unwrap()
        .into_iter()
        .enumerate()
        .map(|(i, handle)| (handle, if i % 2 == 0 { SideId::P1 } else { SideId::P2 }))
        .collect();
    (batch, requests)
}

fn report(mode: &str, repetition: usize, elapsed: f64) {
    println!(
        "mode={mode} repetition={repetition} wall_s={elapsed:.6} views_per_s={:.1} batch_2048_ms={:.3}",
        (ENVIRONMENTS * ITERATIONS) as f64 / elapsed,
        elapsed * 1000.0 / ITERATIONS as f64,
    );
}

fn main() {
    let dex = Arc::new(Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap());
    let teams: Arc<Vec<Team>> =
        Arc::new(serde_json::from_str(include_str!("../data/training-teams.json")).unwrap());
    assert_eq!(teams.len(), 1137);
    let encoder = Encoder::new(&dex).unwrap();
    println!(
        "native_observation_probe environments={ENVIRONMENTS} teams=1137 phase=preview iterations={ITERATIONS} repetitions={REPETITIONS} bridge=unmeasured model=unmeasured numa=unconfigured cold_load_reset_validation_and_warmup=excluded full_battle_observation_cost=unmeasured"
    );
    {
        let (batch, requests) = group(dex.clone(), teams.clone(), 1, ENVIRONMENTS, 0);
        let views = batch.observe_batch(&requests).unwrap();
        let mut direct: Vec<_> = (0..ENVIRONMENTS)
            .map(|_| ObservationBuffers::default())
            .collect();
        for (view, out) in views.iter().zip(&mut direct) {
            encoder.encode_into(view, out).unwrap();
        }
        let mut batched = ObservationBatchBuffers::default();
        batch
            .observe_encoded_batch_into(&requests, &encoder, &mut batched)
            .unwrap();
        assert_eq!(batched.as_slice(), direct.as_slice());
        for repetition in 1..=REPETITIONS {
            let start = Instant::now();
            for _ in 0..ITERATIONS {
                for (view, out) in views.iter().zip(&mut direct) {
                    encoder.encode_into(view, out).unwrap();
                }
                black_box(direct.as_slice());
            }
            report(
                "cached_views_serial_encoding",
                repetition,
                start.elapsed().as_secs_f64(),
            );
            let start = Instant::now();
            for _ in 0..ITERATIONS {
                batch
                    .observe_encoded_batch_into(&requests, &encoder, &mut batched)
                    .unwrap();
                black_box(batched.as_slice());
            }
            report(
                "view_generation_and_encoding_1_worker",
                repetition,
                start.elapsed().as_secs_f64(),
            );
        }
    }
    {
        let groups: Vec<_> = (0..2)
            .map(|i| group(dex.clone(), teams.clone(), 16, 1024, 1024 * i))
            .collect();
        let mut outputs = [
            ObservationBatchBuffers::default(),
            ObservationBatchBuffers::default(),
        ];
        let mut serial = ObservationBuffers::default();
        for ((batch, requests), out) in groups.iter().zip(&mut outputs) {
            batch
                .observe_encoded_batch_into(requests, &encoder, out)
                .unwrap();
            for (view, encoded) in batch
                .observe_batch(requests)
                .unwrap()
                .iter()
                .zip(out.iter())
            {
                encoder.encode_into(view, &mut serial).unwrap();
                assert_eq!(encoded, &serial);
            }
        }
        for repetition in 1..=REPETITIONS {
            let start = Instant::now();
            std::thread::scope(|scope| {
                for ((batch, requests), out) in groups.iter().zip(&mut outputs) {
                    let encoder = &encoder;
                    scope.spawn(move || {
                        for _ in 0..ITERATIONS {
                            batch
                                .observe_encoded_batch_into(requests, encoder, out)
                                .unwrap();
                            black_box(out.as_slice());
                        }
                    });
                }
            });
            report(
                "view_generation_and_encoding_2x16_workers",
                repetition,
                start.elapsed().as_secs_f64(),
            );
        }
    }
}
