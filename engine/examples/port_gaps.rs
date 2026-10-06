//! Development gap instrument. Groups the remaining unimplemented regulation
//! moves, abilities and items by their exact blocking cause so porting work can
//! be ordered by cause instead of by symptom. Cold tool: never called by battles.
use pa3_engine::assets::{ability_block_reasons, item_block_reasons, move_block_reasons};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

struct Entry {
    name: String,
    reasons: Vec<String>,
    pool: usize,
}

fn load(dir: &Path, file: &str) -> Value {
    serde_json::from_slice(&std::fs::read(dir.join(file)).unwrap_or_else(|e| {
        panic!("read {file}: {e}");
    }))
    .unwrap_or_else(|e| panic!("parse {file}: {e}"))
}

fn report(kind: &str, entries: Vec<Entry>, show: usize) {
    let blocked: Vec<_> = entries.into_iter().filter(|e| !e.reasons.is_empty()).collect();
    let mut by_reason: BTreeMap<String, (usize, usize, Vec<String>)> = BTreeMap::new();
    for entry in &blocked {
        for reason in &entry.reasons {
            let slot = by_reason.entry(reason.clone()).or_default();
            slot.0 += 1;
            slot.1 += entry.pool;
            slot.2.push(format!("{}:{}", entry.name, entry.pool));
        }
    }
    let mut ranked: Vec<_> = by_reason.into_iter().collect();
    ranked.sort_by(|a, b| {
        b.1
            .1
            .cmp(&a.1.1)
            .then(b.1.0.cmp(&a.1.0))
            .then(a.0.cmp(&b.0))
    });
    println!("==== {kind}: {} blocked entries", blocked.len());
    for (reason, (count, pool, mut members)) in ranked.into_iter().take(show) {
        members.sort_by_key(|m| {
            std::cmp::Reverse(m.rsplit(':').next().unwrap().parse::<usize>().unwrap_or(0))
        });
        let head: Vec<&str> = members.iter().take(14).map(|s| s.as_str()).collect();
        println!("  pool={pool:>4} n={count:>3}  {reason}   [{}]", head.join(" "));
    }
}

fn main() {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| format!("{}/data", env!("CARGO_MANIFEST_DIR")))
        .into();
    let dex = load(&dir, "dex.json");
    let scope = load(&dir, "scope.json");
    let pool = load(&dir, "training-teams.json");

    let pool_index = |kind: &str| -> BTreeMap<String, usize> {
        let mut weights: BTreeMap<String, usize> = BTreeMap::new();
        let allowed: Vec<String> = scope[match kind {
            "moves" => "allowed_moves",
            "items" => "allowed_items",
            _ => "",
        }]
        .as_array()
        .map(|v| v.iter().map(|s| s.as_str().unwrap().to_string()).collect())
        .unwrap_or_default();
        for team in pool.as_array().unwrap() {
            for member in team["members"].as_array().unwrap() {
                if kind == "moves" {
                    for mv in member["moves"].as_array().unwrap() {
                        *weights.entry(mv.as_u64().unwrap().to_string()).or_default() += 1;
                    }
                } else {
                    let id = member[match kind {
                        "items" => "item",
                        _ => "ability",
                    }]
                    .as_u64()
                    .unwrap();
                    *weights.entry(id.to_string()).or_default() += 1;
                }
            }
        }
        let _ = allowed;
        weights
    };
    let _ = pool_index("moves");
    // The pool stores resolved numeric ids, so weight by the engine's name for
    // that id when the name exists in the scope-legal list.
    let mut pool_moves: BTreeMap<String, usize> = BTreeMap::new();
    let mut pool_abilities: BTreeMap<String, usize> = BTreeMap::new();
    let mut pool_items: BTreeMap<String, usize> = BTreeMap::new();
    let scoped: BTreeMap<String, std::collections::BTreeSet<String>> = [
        ("moves", "allowed_moves"),
        ("items", "allowed_items"),
    ]
    .into_iter()
    .map(|(kind, key)| {
        (
            kind.to_string(),
            scope[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect(),
        )
    })
    .collect();
    for team in pool.as_array().unwrap() {
        for member in team["members"].as_array().unwrap() {
            for mv in member["moves"].as_array().unwrap() {
                let id = mv.as_u64().unwrap();
                let name = &dex["tables"]["moves"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|r| r["numeric_id"].as_u64() == Some(id))
                    .map(|r| r["id"].as_str().unwrap().to_string())
                    .unwrap_or_default();
                if scoped["moves"].contains(name) {
                    *pool_moves.entry(name.clone()).or_default() += 1;
                }
            }
            let ability_id = member["ability"].as_u64().unwrap();
            let ability_name = &dex["tables"]["abilities"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["numeric_id"].as_u64() == Some(ability_id))
                .map(|r| r["id"].as_str().unwrap().to_string())
                .unwrap_or_default();
            *pool_abilities.entry(ability_name.clone()).or_default() += 1;
            let item_id = member["item"].as_u64().unwrap();
            if item_id != 0 {
                let item_name = &dex["tables"]["items"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|r| r["numeric_id"].as_u64() == Some(item_id))
                    .map(|r| r["id"].as_str().unwrap().to_string())
                    .unwrap_or_default();
                if scoped["items"].contains(item_name) {
                    *pool_items.entry(item_name.clone()).or_default() += 1;
                }
            }
        }
    }

    let mut moves = Vec::new();
    for row in dex["tables"]["moves"].as_array().unwrap() {
        let id = row["id"].as_str().unwrap();
        let name = id.to_string();
        if !scoped["moves"].contains(&name) {
            continue;
        }
        moves.push(Entry {
            pool: pool_moves.get(&name).copied().unwrap_or(0),
            reasons: move_block_reasons(id, &row["data"]),
            name,
        });
    }
    report("moves", moves, 40);

    let mut abilities = Vec::new();
    for row in dex["tables"]["abilities"].as_array().unwrap() {
        let id = row["id"].as_str().unwrap();
        let name = id.to_string();
        abilities.push(Entry {
            pool: pool_abilities.get(&name).copied().unwrap_or(0),
            reasons: ability_block_reasons(id, &row["data"]),
            name,
        });
    }
    report("abilities (all catalogue; pool weights are training-pool slots)", abilities, 40);

    let mut items = Vec::new();
    for row in dex["tables"]["items"].as_array().unwrap() {
        let id = row["id"].as_str().unwrap();
        let name = id.to_string();
        if !scoped["items"].contains(&name) {
            continue;
        }
        items.push(Entry {
            pool: pool_items.get(&name).copied().unwrap_or(0),
            reasons: item_block_reasons(id, &row["data"]),
            name,
        });
    }
    report("items", items, 40);
}
