//! Native-only replay benchmark; JSON, traces and snapshots are outside timed paths.
use pa3_engine::{
    actions::AtomicAction,
    assets::Dex,
    batch::{BattleBatch, ResetSpec, SideChoice, StepSpec},
    state::{SideId, Team},
};
use serde::Deserialize;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

struct CountingAllocator;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(size as u64, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

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
}
struct Prepared {
    batch: BattleBatch,
    rounds: Vec<Vec<StepSpec>>,
    observations: Vec<Vec<(pa3_engine::batch::Handle, SideId)>>,
    games: usize,
}
#[derive(Default)]
struct Metrics {
    step: Duration,
    observe: Duration,
    candidates: Duration,
    submissions: usize,
    completed: usize,
    candidate_count: usize,
}
fn prepare(
    dex: Arc<Dex>,
    teams: Arc<Vec<Team>>,
    corpus: &Corpus,
    workers: usize,
    count: usize,
    offset: usize,
) -> Prepared {
    let mut batch = BattleBatch::new(dex, teams, workers).unwrap();
    let indices: Vec<_> = (0..count)
        .map(|i| (i + offset) % corpus.fixtures.len())
        .collect();
    let resets: Vec<_> = indices
        .iter()
        .map(|&i| ResetSpec {
            team_a: i * 2,
            team_b: i * 2 + 1,
            seed: corpus.fixtures[i].seed,
            role_map: [0, 1],
        })
        .collect();
    let handles = batch.reset_batch(&resets).unwrap();
    let max_steps = indices
        .iter()
        .map(|&i| corpus.fixtures[i].steps.len())
        .max()
        .unwrap();
    let rounds: Vec<Vec<StepSpec>> = (0..max_steps)
        .map(|step| {
            indices
                .iter()
                .enumerate()
                .filter_map(|(slot, &i)| {
                    corpus.fixtures[i].steps.get(step).map(|s| StepSpec {
                        handle: handles[slot],
                        choices: vec![SideChoice {
                            side: s.side,
                            actions: s.actions.clone(),
                        }],
                    })
                })
                .collect()
        })
        .collect();
    let observations = rounds
        .iter()
        .map(|round| {
            round
                .iter()
                .map(|s| (s.handle, s.choices[0].side))
                .collect()
        })
        .collect();
    Prepared {
        batch,
        rounds,
        observations,
        games: count,
    }
}
fn run(mut prepared: Prepared, observations: bool) -> (Prepared, Metrics) {
    let mut metrics = Metrics::default();
    for (round, requests) in prepared.rounds.iter().zip(&prepared.observations) {
        if observations {
            let before = Instant::now();
            let views = prepared.batch.observe_batch(requests).unwrap();
            metrics.observe += before.elapsed();
            let before = Instant::now();
            for view in &views {
                metrics.candidate_count += black_box(view.request.candidates(&[]).unwrap()).len();
            }
            metrics.candidates += before.elapsed();
            black_box(views);
        }
        let before = Instant::now();
        let results = prepared.batch.step_batch(round).unwrap();
        metrics.step += before.elapsed();
        metrics.submissions += round.len();
        for result in &results {
            assert!(result.outcome.operational_error.is_none());
            if result.outcome.terminated {
                metrics.completed += 1;
            }
        }
        black_box(results);
    }
    assert_eq!(
        metrics.completed, prepared.games,
        "all replays must naturally finish exactly once"
    );
    (prepared, metrics)
}
fn rss_kib() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find(|l| l.starts_with("VmRSS:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}
fn scenario(
    dex: Arc<Dex>,
    teams: Arc<Vec<Team>>,
    corpus: &Corpus,
    parallel: bool,
    observations: bool,
    allocations: bool,
) {
    let prepared: Vec<_> = if parallel {
        vec![
            prepare(dex.clone(), teams.clone(), corpus, 16, 1024, 0),
            prepare(dex, teams, corpus, 16, 1024, 1024),
        ]
    } else {
        vec![prepare(dex, teams, corpus, 1, 2048, 0)]
    };
    let rss = rss_kib();
    ALLOCATIONS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    COUNTING.store(allocations, Ordering::Relaxed);
    let before = Instant::now();
    let completed_groups: Vec<_> = std::thread::scope(|scope| {
        let joins: Vec<_> = prepared
            .into_iter()
            .map(|group| scope.spawn(move || run(group, observations)))
            .collect();
        joins.into_iter().map(|j| j.join().unwrap()).collect()
    });
    let wall = before.elapsed();
    COUNTING.store(false, Ordering::Relaxed);
    let metrics: Vec<_> = completed_groups
        .iter()
        .map(|(_, metrics)| metrics)
        .collect();
    let submissions: usize = metrics.iter().map(|m| m.submissions).sum();
    let completed: usize = metrics.iter().map(|m| m.completed).sum();
    let seconds = wall.as_secs_f64();
    println!(
        "{} observe={} allocations={} wall_s={:.6} submissions_per_s={:.1} complete_battles_per_s={:.1} mean_side_submissions={:.2} step_group_seconds={:.6} observe_group_seconds={:.6} candidates_group_seconds={:.6} alloc_calls={} allocated_bytes={} rss_2048_kib={:?}",
        if parallel {
            "two_groups_16_workers"
        } else {
            "one_group_1_worker"
        },
        observations,
        allocations,
        seconds,
        submissions as f64 / seconds,
        completed as f64 / seconds,
        submissions as f64 / completed as f64,
        metrics.iter().map(|m| m.step.as_secs_f64()).sum::<f64>(),
        metrics.iter().map(|m| m.observe.as_secs_f64()).sum::<f64>(),
        metrics
            .iter()
            .map(|m| m.candidates.as_secs_f64())
            .sum::<f64>(),
        ALLOCATIONS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
        rss
    );
}
fn main() {
    let allocations = std::env::args().any(|s| s == "--count-allocations");
    let dex = Arc::new(Dex::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"))).unwrap());
    let mut corpus: Corpus = serde_json::from_slice(
        &std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/turn-fixtures.json"
        ))
        .unwrap(),
    )
    .unwrap();
    // Compare identical historical inputs after subsequent mechanic additions;
    // the default benchmark and differential suite retain the entire corpus.
    if std::env::args().any(|s| s == "--audit-baseline") {
        let names: Vec<String> =
            serde_json::from_str(include_str!("../benchmarks/hot_path_baseline_names.json"))
                .unwrap();
        corpus.fixtures.retain(|f| names.contains(&f.name));
        corpus
            .fixtures
            .sort_by_key(|f| names.iter().position(|name| name == &f.name).unwrap());
        assert_eq!(corpus.fixtures.len(), 95, "historical audit workload");
    }
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
        "native_fixture_replay fixtures={} environments=2048 bridge=unmeasured numa=unconfigured policy=unmeasured coverage=ported_subset allocation_counter_overhead={}",
        corpus.fixtures.len(),
        allocations
    );
    for observations in [false, true] {
        for parallel in [false, true] {
            scenario(
                dex.clone(),
                teams.clone(),
                &corpus,
                parallel,
                observations,
                allocations,
            );
        }
    }
}
