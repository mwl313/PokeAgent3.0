//! Component probes. Direct steps still include all native event handling;
//! idle batch scanning and sorting are microbenchmarks, not a full decomposition.
use pa3_engine::{
    actions::AtomicAction,
    assets::Dex,
    batch::{BattleBatch, ResetSpec},
    queue::{Priority, speed_sort},
    rng::BattleRng,
    state::{BattleState, SideId, Team},
};
use serde::Deserialize;
use std::{hint::black_box, path::Path, sync::Arc, time::Instant};
#[derive(Deserialize)]
struct Corpus {
    oracle_commit: String,
    format: String,
    fixtures: Vec<Fixture>,
}
#[derive(Deserialize)]
struct Fixture {
    seed: [u16; 4],
    teams: [Team; 2],
    steps: Vec<Step>,
}
#[derive(Deserialize)]
struct Step {
    side: SideId,
    actions: Vec<AtomicAction>,
}
fn direct(dex: &Dex, corpus: &Corpus, trace: bool) {
    let indices: Vec<_> = (0..2048).map(|i| i % corpus.fixtures.len()).collect();
    let mut states: Vec<_> = indices
        .iter()
        .map(|&i| {
            let f = &corpus.fixtures[i];
            let mut state =
                BattleState::reset(dex, [&f.teams[0], &f.teams[1]], f.seed, [0, 1]).unwrap();
            if trace {
                state.enable_trace().unwrap();
            }
            state
        })
        .collect();
    let mut submissions = 0;
    let mut completed = 0;
    let before = Instant::now();
    for (state, &index) in states.iter_mut().zip(&indices) {
        for step in &corpus.fixtures[index].steps {
            let result = state.step(dex, step.side, &step.actions).unwrap();
            assert!(result.outcome.operational_error.is_none());
            submissions += 1;
            completed += usize::from(result.outcome.terminated);
            black_box(result);
        }
    }
    let seconds = before.elapsed().as_secs_f64();
    assert_eq!(completed, states.len());
    println!(
        "direct_state_step trace={} wall_s={:.6} side_submissions={} side_submissions_per_s={:.1} completed_battles={} completed_battles_per_s={:.1} mean_side_submissions={:.2}",
        trace,
        seconds,
        submissions,
        submissions as f64 / seconds,
        completed,
        completed as f64 / seconds,
        submissions as f64 / completed as f64
    );
    black_box(states);
}
fn idle_scan(dex: Arc<Dex>, teams: Arc<Vec<Team>>, workers: usize) {
    let mut batch = BattleBatch::new(dex, teams, workers).unwrap();
    let specs: Vec<_> = (0..2048)
        .map(|i| ResetSpec {
            team_a: 0,
            team_b: 1,
            seed: [2026, 10, 7, i as u16],
            role_map: [0, 1],
        })
        .collect();
    batch.reset_batch(&specs).unwrap();
    let mut out = Vec::new();
    batch.step_batch_into(&[], &mut out).unwrap();
    let iterations = 1000;
    let before = Instant::now();
    for _ in 0..iterations {
        batch
            .step_batch_into(black_box(&[]), black_box(&mut out))
            .unwrap();
    }
    let seconds = before.elapsed().as_secs_f64();
    assert!(out.is_empty());
    println!(
        "idle_batch_scan_only environments=2048 workers={} calls={} wall_s={:.6} us_per_idle_call={:.3} active_submission_validation=unmeasured",
        workers,
        iterations,
        seconds,
        seconds * 1e6 / f64::from(iterations)
    );
}
fn sort_primitive() {
    for count in [4, 16, 32] {
        for ties in [false, true] {
            let original: Vec<_> = (0..count)
                .map(|i| Priority {
                    speed: if ties { 100 } else { i * 13 },
                    ..Default::default()
                })
                .collect();
            let mut values = original.clone();
            let mut rng = BattleRng::new([2026, 10, 7, 1]);
            let iterations = 100_000;
            let before = Instant::now();
            for _ in 0..iterations {
                values.copy_from_slice(&original);
                speed_sort(black_box(&mut values), black_box(&mut rng), |p| *p);
                black_box(&values);
            }
            let seconds = before.elapsed().as_secs_f64();
            println!(
                "speed_sort_primitive handlers={} ties={} sorts={} wall_s={:.6} ns_per_sort={:.1} includes_input_copy=true full_event_dispatch=unmeasured",
                count,
                ties,
                iterations,
                seconds,
                seconds * 1e9 / f64::from(iterations)
            );
        }
    }
}
fn main() {
    let dex = Arc::new(Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap());
    let corpus: Corpus = serde_json::from_slice(
        &std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/turn-fixtures.json"
        ))
        .unwrap(),
    )
    .unwrap();
    assert_eq!(corpus.oracle_commit, pa3_engine::ORACLE_COMMIT);
    assert_eq!(corpus.format, pa3_engine::FORMAT);
    let teams: Arc<Vec<Team>> = Arc::new(
        corpus
            .fixtures
            .iter()
            .flat_map(|f| f.teams.clone())
            .collect(),
    );
    println!(
        "component_probes fixtures={} coldsetup=excluded bridge=unmeasured policy=unmeasured full_event_dispatch_decomposition=unmeasured",
        corpus.fixtures.len()
    );
    direct(&dex, &corpus, false);
    direct(&dex, &corpus, true);
    idle_scan(dex.clone(), teams.clone(), 1);
    idle_scan(dex, teams, 16);
    sort_primitive();
}
