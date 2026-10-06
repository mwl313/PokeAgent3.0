//! Development coverage instrument. Prints which regulation-legal moves,
//! abilities and items the native engine can currently execute, and which
//! remain explicit operational errors. Cold tool: never called by battles.
use pa3_engine::assets::{Dex, Id};
use pa3_engine::effects::{Item, MoveBehavior};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| format!("{}/data", env!("CARGO_MANIFEST_DIR")));
    let dex = Dex::load(Path::new(&dir)).expect("dex");
    let scope: serde_json::Value =
        serde_json::from_slice(&std::fs::read(Path::new(&dir).join("scope.json")).expect("scope"))
            .expect("scope json");
    let names = |kind: &str, id: Id| dex.names[kind][id as usize].clone();

    let allowed_moves: BTreeSet<String> = scope["allowed_moves"]
        .as_array()
        .expect("allowed_moves")
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let mut executable = 0usize;
    let mut blocked = Vec::new();
    let mut explicit = 0usize;
    for id in &allowed_moves {
        let Ok(numeric) = dex.id("moves", id) else {
            continue;
        };
        let behavior = dex.effects.moves[numeric as usize];
        if behavior != MoveBehavior::Unimplemented {
            executable += 1;
            if behavior != MoveBehavior::Damage && behavior != MoveBehavior::Effect {
                explicit += 1;
            }
        } else {
            blocked.push(names("moves", numeric));
        }
    }
    println!(
        "moves: allowed={} executable={} (special={} generic={}) blocked={}",
        allowed_moves.len(),
        executable,
        explicit,
        executable - explicit,
        blocked.len()
    );
    println!("blocked_moves={}", blocked.join(","));

    let mut legal_abilities: BTreeSet<Id> = BTreeSet::new();
    let mut legal_items: BTreeSet<Id> = BTreeSet::new();
    for (index, legal) in dex.legal_starting_species.iter().enumerate() {
        if !*legal || index == 0 {
            continue;
        }
        for &ability in &dex.species[index].abilities {
            legal_abilities.insert(ability);
        }
    }
    for id in scope["allowed_items"].as_array().expect("allowed_items") {
        if let Ok(numeric) = dex.id("items", id.as_str().unwrap()) {
            legal_items.insert(numeric);
        }
    }
    for id in scope["format_permitted_battle_forms"]
        .as_array()
        .expect("battle_forms")
    {
        if let Ok(numeric) = dex.id("species", id.as_str().unwrap()) {
            for &ability in &dex.species[numeric as usize].abilities {
                legal_abilities.insert(ability);
            }
        }
    }
    let mut ability_blocked = Vec::new();
    for &id in &legal_abilities {
        if !dex.effects.abilities[id as usize].is_ported() {
            ability_blocked.push(names("abilities", id));
        }
    }
    println!(
        "abilities: legal={} executable={} blocked={}",
        legal_abilities.len(),
        legal_abilities.len() - ability_blocked.len(),
        ability_blocked.len()
    );
    println!("blocked_abilities={}", ability_blocked.join(","));
    let mut item_blocked = Vec::new();
    for &id in &legal_items {
        if dex.effects.items[id as usize] == Item::Unimplemented {
            item_blocked.push(names("items", id));
        }
    }
    println!(
        "items: legal={} executable={} blocked={}",
        legal_items.len(),
        legal_items.len() - item_blocked.len(),
        item_blocked.len()
    );
    println!("blocked_items={}", item_blocked.join(","));

    // Training-pool blockers: what stops the frozen 1,136 teams from running.
    let pool: serde_json::Value = serde_json::from_slice(
        &std::fs::read(Path::new(&dir).join("training-teams.json")).expect("training teams"),
    )
    .expect("training json");
    let mut pool_moves: BTreeMap<String, usize> = BTreeMap::new();
    let mut pool_abilities: BTreeMap<String, usize> = BTreeMap::new();
    let mut pool_items: BTreeMap<String, usize> = BTreeMap::new();
    for team in pool.as_array().expect("teams") {
        for member in team["members"].as_array().expect("members") {
            for mv in member["moves"].as_array().expect("moves") {
                let id = mv.as_u64().unwrap() as Id;
                if dex.effects.moves[id as usize] == MoveBehavior::Unimplemented {
                    *pool_moves.entry(names("moves", id)).or_default() += 1;
                }
            }
            let ability = member["ability"].as_u64().unwrap() as Id;
            if !dex.effects.abilities[ability as usize].is_ported() {
                *pool_abilities
                    .entry(names("abilities", ability))
                    .or_default() += 1;
            }
            let item = member["item"].as_u64().unwrap() as Id;
            if item != 0 && dex.effects.items[item as usize] == Item::Unimplemented {
                *pool_items.entry(names("items", item)).or_default() += 1;
            }
        }
    }
    println!(
        "pool_blockers: distinct_moves={} distinct_abilities={} distinct_items={}",
        pool_moves.len(),
        pool_abilities.len(),
        pool_items.len()
    );
    let dump = |label: &str, entries: BTreeMap<String, usize>| {
        let mut list: Vec<_> = entries.into_iter().collect();
        list.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        println!(
            "{label}={}",
            list.iter()
                .map(|(name, count)| format!("{name}:{count}"))
                .collect::<Vec<_>>()
                .join(",")
        );
    };
    dump("pool_blocked_moves", pool_moves);
    dump("pool_blocked_abilities", pool_abilities);
    dump("pool_blocked_items", pool_items);
}
