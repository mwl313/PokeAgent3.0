//! Development instrument (never used by training): attempts one complete
//! natural battle for every frozen training team under a deterministic,
//! seeded native policy, and reports the first operational blocker each team
//! actually reaches. Static gap analysis says which mechanics are missing;
//! this says which teams can genuinely finish a battle today, and which
//! unsupported mechanic stops them first.
//!
//! Usage: cargo run --release --example pool_run_report [data-dir] [--verbose]
use pa3_engine::actions::{AtomicAction, RequestKind};
use pa3_engine::assets::Dex;
use pa3_engine::state::{BattleState, SideId, Team};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Deterministic per-team policy stream (SplitMix64). Independent of the
/// battle RNG so the probe is reproducible without touching engine state.
struct Probe(u64);

impl Probe {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn pick(&mut self, upper: usize) -> usize {
        (self.next() % upper.max(1) as u64) as usize
    }
}

enum ProbeResult {
    Complete { turns: u16, winner: Option<SideId> },
    Unsupported(String),
    Truncated { turns: u16 },
    EngineError(String),
}

fn normalize(dex: &Dex, message: &str) -> String {
    // `unsupported mechanic: move ID 123` -> `move <name>`.
    if let Some(rest) = message.strip_prefix("unsupported mechanic: move ID ")
        && let Ok(id) = rest.trim().parse::<usize>()
        && let Some(name) = dex.names["moves"].get(id)
    {
        return format!("move {name}");
    }
    message.to_string()
}

fn run_team(dex: &Dex, team: &Team, index: usize, decision_cap: u32) -> ProbeResult {
    let seed = [
        1 + (index as u16 % 30_000),
        (index as u16).wrapping_mul(7).wrapping_add(11),
        (index as u16).wrapping_mul(29).wrapping_add(101),
        (index as u16).wrapping_mul(131).wrapping_add(1009),
    ];
    let mut state = match BattleState::reset(dex, [team, team], seed, [0, 1]) {
        Ok(state) => state,
        Err(error) => return ProbeResult::EngineError(error.to_string()),
    };
    let mut probe = Probe(0x51ED_2701_0000_0000 ^ (index as u64).wrapping_mul(0x2545_F491_4F6C_DD1D));
    let mut decisions = 0u32;
    loop {
        let outcome = state.observe(SideId::P1).outcome.clone();
        if let Some(error) = outcome.operational_error {
            return ProbeResult::Unsupported(normalize(dex, &error));
        }
        if outcome.terminated {
            return ProbeResult::Complete {
                turns: state.observe(SideId::P1).turn,
                winner: outcome.winner,
            };
        }
        if outcome.truncated || decisions >= decision_cap {
            return ProbeResult::Truncated {
                turns: state.observe(SideId::P1).turn,
            };
        }
        let mut submitted = false;
        for side in [SideId::P1, SideId::P2] {
            let view = state.observe(side);
            if matches!(view.request.kind, RequestKind::Wait | RequestKind::Finished) {
                continue;
            }
            let mut actions: Vec<AtomicAction> = Vec::with_capacity(4);
            let mut complete = true;
            for _ in 0..view.request.branch_slots().len() {
                let candidates = match view.request.candidates(&actions) {
                    Ok(candidates) => candidates,
                    Err(error) => return ProbeResult::EngineError(error.to_string()),
                };
                if candidates.is_empty() {
                    complete = false;
                    break;
                }
                let pick = if actions.is_empty() {
                    // Prefer acting over switching when the engine offers both,
                    // so battles actually progress instead of shuffling.
                    let attacking: Vec<usize> = candidates
                        .iter()
                        .enumerate()
                        .filter(|(_, a)| {
                            !matches!(a.kind, pa3_engine::actions::ActionKind::Switch)
                        })
                        .map(|(i, _)| i)
                        .collect();
                    if attacking.is_empty() {
                        probe.pick(candidates.len())
                    } else {
                        attacking[probe.pick(attacking.len())]
                    }
                } else {
                    probe.pick(candidates.len())
                };
                actions.push(candidates[pick]);
            }
            if !complete {
                return ProbeResult::EngineError("no legal candidates".into());
            }
            if let Err(error) = state.step(dex, side, &actions) {
                return ProbeResult::EngineError(error.to_string());
            }
            submitted = true;
        }
        if !submitted {
            return ProbeResult::EngineError("no actionable side at a boundary".into());
        }
        decisions += 1;
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data");
    let mut verbose = false;
    for arg in args.by_ref() {
        match arg.as_str() {
            "--verbose" => verbose = true,
            _ => dir = Path::new(&arg).to_path_buf(),
        }
    }
    let dex = Dex::load(&dir).expect("dex");
    let teams: Vec<Team> =
        serde_json::from_slice(&std::fs::read(dir.join("training-teams.json")).expect("teams"))
            .expect("training teams");
    let decision_cap: u32 = std::env::var("POOL_DECISION_CAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    let mut complete = 0usize;
    let mut blockers: BTreeMap<String, usize> = BTreeMap::new();
    let mut details: Vec<(usize, String)> = Vec::new();
    let mut turns_total = 0u64;
    let mut decisive = 0usize;
    for (index, team) in teams.iter().enumerate() {
        match run_team(&dex, team, index, decision_cap) {
            ProbeResult::Complete { turns, winner } => {
                complete += 1;
                turns_total += u64::from(turns);
                if winner.is_some() {
                    decisive += 1;
                }
            }
            ProbeResult::Unsupported(message) => {
                *blockers.entry(message.clone()).or_default() += 1;
                details.push((index, message));
            }
            ProbeResult::Truncated { turns } => {
                *blockers.entry("decision cap / turn limit".into()).or_default() += 1;
                details.push((index, format!("truncated after {turns} turns")));
            }
            ProbeResult::EngineError(message) => {
                *blockers.entry(format!("ENGINE ERROR {message}")).or_default() += 1;
                details.push((index, format!("engine error: {message}")));
            }
        }
    }
    println!(
        "training pool: {} teams, {} complete natural battles ({:.1}%), {} blocked",
        teams.len(),
        complete,
        complete as f64 * 100.0 / teams.len().max(1) as f64,
        teams.len() - complete
    );
    if complete > 0 {
        println!(
            "mean completed battle length: {:.1} turns, {decisive} decisive (decision cap {decision_cap})",
            turns_total as f64 / complete as f64,
        );
    }
    let mut ranked: Vec<(String, usize)> = blockers.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    println!("first blockers (each team counted once, at its earliest unsupported mechanic):");
    for (message, count) in ranked.iter().take(40) {
        println!("{count:6}  {message}");
    }
    if verbose {
        println!("per-team blockers:");
        for (index, message) in &details {
            println!("{index:5}  {message}");
        }
    }
}
