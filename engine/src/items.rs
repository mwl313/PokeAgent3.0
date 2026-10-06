//! Data-driven item classification for the pinned Champions catalogue.
//!
//! The explicit `Item::compile` table covers items whose behaviour is a ported
//! native handler. Items whose entire reference declaration is expressed by
//! ported primitives are recognised here instead, under the same rule as
//! `assets::classify_move`: every declared field and callback must be handled
//! exactly, otherwise the item stays `Item::Unimplemented` (an explicit
//! operational error) rather than a silent approximation.
//!
//! Cold path: classification runs once when the shared immutable Dex loads.
use crate::{
    assets::{Dex, Id},
    effects::Item,
};
use serde_json::Value;

/// Catalogue metadata and inert flags that never drive a battle transition.
/// Any other declared field keeps the item unsupported until it is ported.
const INERT_ITEM_FIELDS: &[&str] = &[
    "affectsFainted",
    "effectType",
    "exists",
    "fling",
    "fullname",
    "gen",
    "id",
    "ignoreKlutz",
    "isBerry",
    "isGem",
    "isNonstandard",
    "isPokeball",
    "isPrimalOrb",
    "itemUser",
    "name",
    "noCopy",
    "num",
    "sourceEffect",
    "spritenum",
];

/// Behavioural keys the native item families below implement exactly.
const HANDLED_ITEM_CALLBACK_FIELDS: &[&str] = &["megaStone", "onTakeItem"];

/// Callback keys whose native port is exactly the item family below.
fn callback_ported(key: &str) -> bool {
    key.ends_with(".onTakeItem")
}

fn collect_callback_keys(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            if let Some(owner) = map.get("callback").and_then(|v| v.as_str()) {
                out.push(owner.to_string());
            }
            for nested in map.values() {
                collect_callback_keys(nested, out);
            }
        }
        Value::Array(items) => items.iter().for_each(|v| collect_callback_keys(v, out)),
        _ => {}
    }
}

/// A Mega Stone is a data-only item once its `onTakeItem` refusal is native:
/// the item's only battle behaviour is the base-form -> Mega mapping that the
/// Dex already stores, plus the rule that it cannot be taken by Knock Off,
/// Thief, Covet, Trick, Switcheroo, Corrosive Gas or Fling-adjacent effects.
fn mega_stone_handled(data: &Value) -> bool {
    let Some(map) = data["megaStone"].as_object() else {
        return false;
    };
    if map.is_empty() {
        return false;
    }
    let Some(fields) = data.as_object() else {
        return false;
    };
    if fields
        .keys()
        .any(|key| {
            !INERT_ITEM_FIELDS.contains(&key.as_str())
                && !HANDLED_ITEM_CALLBACK_FIELDS.contains(&key.as_str())
        })
    {
        return false;
    }
    let mut callbacks = Vec::new();
    collect_callback_keys(data, &mut callbacks);
    callbacks.iter().all(|key| callback_ported(key))
}

/// Cold classifier: explicit table first, then data-only families.
pub fn classify(id: &str, data: &Value) -> Item {
    let explicit = Item::compile(id);
    if explicit != Item::Unimplemented {
        return explicit;
    }
    if mega_stone_handled(data) {
        return Item::MegaStone;
    }
    Item::Unimplemented
}

impl Dex {
    /// Reference `item.megaStone` membership for the *holder's* base form.
    pub fn mega_stone_matches(&self, item: Id, base_species: Id) -> bool {
        self.effects.mega_stones[item as usize]
            .iter()
            .any(|(base, _)| *base == base_species)
    }

    /// Reference `onTakeItem` result for the currently ported item set.
    /// Returns true when the item cannot be taken from this holder at all.
    pub fn item_take_refused(&self, item: Id, base_species: Id) -> bool {
        self.effects.items[item as usize] == Item::MegaStone
            && self.mega_stone_matches(item, base_species)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn mega_stones_classify_only_with_the_exact_declaration() {
        let stone = json!({
            "exists": true,
            "id": "charizarditey",
            "name": "Charizardite Y",
            "megaStone": {"Charizard": "Charizard-Mega-Y"},
            "onTakeItem": {"callback": "items:charizarditey.onTakeItem"},
        });
        assert_eq!(classify("charizarditey", &stone), Item::MegaStone);
        // A stone with any additional behavioural field stays an explicit
        // operational error until that field is ported.
        let extra = json!({
            "megaStone": {"Charizard": "Charizard-Mega-Y"},
            "onTakeItem": {"callback": "items:charizarditey.onTakeItem"},
            "onEat": {"callback": "items:charizarditey.onEat"},
        });
        assert_eq!(classify("charizarditey", &extra), Item::Unimplemented);
        let no_mapping = json!({"onTakeItem": {"callback": "items:x.onTakeItem"}});
        assert_eq!(classify("x", &no_mapping), Item::Unimplemented);
    }
}
