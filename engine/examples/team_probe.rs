//! Development instrument (never used in training): focused compatibility probe
//! for one training-pool team. It statically scans every member's moves,
//! ability and item, then plays natural battles against several fixed
//! opponents and seeds, reporting operational errors, truncations and the
//! outcome distribution.
//!
//! Usage:
//!   cargo run --release --locked --example team_probe -- <team-id-prefix> [--seeds N] [--opponents K] [--data DIR]
use pa3_engine::actions::{AtomicAction, RequestKind};
use pa3_engine::assets::Dex;
use pa3_engine::state::{BattleState, SideId, Team};
use std::path::PathBuf;

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
    if let Some(rest) = message.strip_prefix("unsupported mechanic: move ID ")
        && let Ok(id) = rest.trim().parse::<usize>()
        && let Some(name) = dex.names["moves"].get(id)
    {
        return format!("move {name}");
    }
    message.to_string()
}

fn run_pair(
    dex: &Dex,
    target: &Team,
    opponent: &Team,
    swap: bool,
    seed: [u16; 4],
    decision_cap: u32,
    tag: u64,
) -> ProbeResult {
    let (p1, p2) = if swap { (opponent, target) } else { (target, opponent) };
    let mut state = match BattleState::reset(dex, [p1, p2], seed, [0, 1]) {
        Ok(state) => state,
        Err(error) => return ProbeResult::EngineError(error.to_string()),
    };
    let mut probe = Probe(0x51ED_2701_0000_0000 ^ tag.wrapping_mul(0x2545_F491_4F6C_DD1D));
    let mut decisions = 0u32;
    loop {
        let outcome = state.observe(SideId::P1).outcome.clone();
        if let Some(error) = outcome.operational_error {
            return ProbeResult::Unsupported(normalize(dex, &error));
        }
        if outcome.terminated {
            return ProbeResult::Complete { turns: state.observe(SideId::P1).turn, winner: outcome.winner };
        }
        if outcome.truncated || decisions >= decision_cap {
            return ProbeResult::Truncated { turns: state.observe(SideId::P1).turn };
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
                    let attacking: Vec<usize> = candidates
                        .iter()
                        .enumerate()
                        .filter(|(_, a)| !matches!(a.kind, pa3_engine::actions::ActionKind::Switch))
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

fn static_gaps(dex: &Dex, team: &Team) -> Vec<String> {
    let mut gaps = Vec::new();
    for (index, member) in team.members.iter().enumerate() {
        let species = dex.names["species"]
            .get(member.species as usize)
            .cloned()
            .unwrap_or_else(|| format!("species {}", member.species));
        if !dex.effects.abilities[member.ability as usize].is_ported() {
            gaps.push(format!("member {index} {species}: ability {}", dex.names["abilities"].get(member.ability as usize).cloned().unwrap_or_default()));
        }
        let mega_ok = dex.effects.mega_stones[member.item as usize]
            .iter()
            .find_map(|(base, mega)| (*base == dex.species[member.species as usize].base_species).then_some(*mega))
            .map(|mega| {
                let ability = dex.species[mega as usize].abilities[0];
                dex.effects.abilities[ability as usize].is_ported()
            })
            .unwrap_or(true);
        if !mega_ok {
            gaps.push(format!("member {index} {species}: Mega-form ability"));
        }
        if member.item != 0
            && dex.effects.items[member.item as usize] == pa3_engine::effects::Item::Unimplemented
            && dex.effects.mega_stones[member.item as usize].is_empty()
        {
            gaps.push(format!("member {index} {species}: item {}", dex.names["items"].get(member.item as usize).cloned().unwrap_or_default()));
        }
        for mv in &member.moves {
            if dex.effects.moves[*mv as usize] == pa3_engine::effects::MoveBehavior::Unimplemented {
                gaps.push(format!("member {index} {species}: move {}", dex.names["moves"].get(*mv as usize).cloned().unwrap_or_default()));
            }
        }
    }
    gaps
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data");
    let mut needle: Option<String> = None;
    let mut index_filter: Option<usize> = None;
    let mut seeds = 8usize;
    let mut opponents = 6usize;
    let mut decision_cap: u32 = 400;
    let mut arg_iter = args.iter();
    while let Some(arg) = arg_iter.next() {
        match arg.as_str() {
            "--seeds" => seeds = arg_iter.next().and_then(|v| v.parse().ok()).unwrap_or(seeds),
            "--opponents" => opponents = arg_iter.next().and_then(|v| v.parse().ok()).unwrap_or(opponents),
            "--decision-cap" => decision_cap = arg_iter.next().and_then(|v| v.parse().ok()).unwrap_or(decision_cap),
            "--index" => index_filter = arg_iter.next().and_then(|v| v.parse().ok()),
            "--data" => dir = PathBuf::from(arg_iter.next().expect("--data path")),
            other => needle = Some(other.to_string()),
        }
    }
    let dex = Dex::load(&dir).expect("dex");
    let teams: Vec<Team> =
        serde_json::from_slice(&std::fs::read(dir.join("training-teams.json")).expect("teams"))
            .expect("training teams");
    let target_index = match (index_filter, needle.as_deref()) {
        (Some(index), _) => index,
        (None, Some(needle)) => teams
            .iter()
            .position(|team| team.id.starts_with(needle))
            .unwrap_or_else(|| panic!("no training team matches {needle}")),
        (None, None) => panic!("usage: team_probe <team-id-prefix> [--index N]"),
    };
    let target = &teams[target_index];
    println!("team {} index {target_index} of {}", target.id, teams.len());
    println!("  members: {}", target
        .members
        .iter()
        .map(|m| format!("{} ({})", dex.names["species"][m.species as usize], dex.names["abilities"][m.ability as usize]))
        .collect::<Vec<_>>()
        .join(", "));
    let gaps = static_gaps(&dex, target);
    println!("  static mechanic gaps: {}", if gaps.is_empty() { "none".to_string() } else { gaps.join("; ") });
    let mut completed = 0usize;
    let mut unsupported: Vec<String> = Vec::new();
    let mut truncated = 0usize;
    let mut engine_errors: Vec<String> = Vec::new();
    let mut turns_total = 0u64;
    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut draws = 0usize;
    let mut games = 0usize;
    for seed_index in 0..seeds {
        for opponent_offset in 0..opponents {
            let opponent_index = (target_index + 1 + opponent_offset * 53) % teams.len();
            for swap in [false, true] {
                let seed = [
                    (2026u16).wrapping_add(seed_index as u16),
                    (10u16).wrapping_add((opponent_index % 97) as u16),
                    (6u16).wrapping_add((opponent_offset * 17) as u16),
                    (seed_index as u16).wrapping_mul(131).wrapping_add(7),
                ];
                let tag = (seed_index as u64) << 24 | (opponent_index as u64) << 4 | u64::from(swap);
                games += 1;
                match run_pair(&dex, target, &teams[opponent_index], swap, seed, decision_cap, tag) {
                    ProbeResult::Complete { turns, winner } => {
                        completed += 1;
                        turns_total += u64::from(turns);
                        match winner {
                            Some(SideId::P1) if !swap => wins += 1,
                            Some(SideId::P2) if swap => wins += 1,
                            Some(_) => losses += 1,
                            None => draws += 1,
                        }
                    }
                    ProbeResult::Unsupported(message) => unsupported.push(message),
                    ProbeResult::Truncated { turns } => {
                        truncated += 1;
                        println!("  truncated after {turns} turns (seed {seed:?}, opponent {opponent_index})");
                    }
                    ProbeResult::EngineError(message) => engine_errors.push(message),
                }
            }
        }
    }
    println!("  probe games: {games}, natural completions: {completed} ({:.1}%)", completed as f64 * 100.0 / games.max(1) as f64);
    if completed > 0 {
        println!("  mean turns: {:.1}, target side W/L/D: {wins}/{losses}/{draws}", turns_total as f64 / completed as f64);
    }
    println!("  operational aborts: {} (unsupported: {}, truncated: {}, engine error: {})", unsupported.len() + truncated + engine_errors.len(), unsupported.len(), truncated, engine_errors.len());
    for message in unsupported.iter().take(5).chain(engine_errors.iter().take(5)) {
        println!("    {message}");
    }
    let clean = gaps.is_empty() && unsupported.is_empty() && truncation_clean(truncated) && engine_errors.is_empty() && completed == games;
    println!("  focused_probe_clean: {clean}");
    if !clean {
        std::process::exit(1);
    }
}

fn truncation_clean(truncated: usize) -> bool {
    truncated == 0
}
