//! Native submission validator for the pinned Champions M-C format.
//!
//! This mirrors the reference `TeamValidator` for the rules the pinned format
//! actually enforces (Flat Rules: Obtainable, Species Clause, Item Clause = 1,
//! Adjust Level = 50, Min Team Size = 6, Picked Team Size = 4, 66 Stat Points,
//! 32 per stat, 31 IVs, no Mythical or Restricted Legendary). Rule constants
//! come from the pinned format's `ruleTable` (`evLimit = 66`, `minTeamSize = 6`,
//! `pickedTeamSize = 4`, `adjustLevel = 50`); `engine/tests/legality.rs` proves
//! the behaviour against the reference on the frozen pool and crafted cases.
//!
//! Cold path: this is submission validation, never a battle transition.
use crate::{
    assets::{Dex, Id},
    state::{Team, TeamSet},
};

/// Stat Point budget from the pinned format's `evLimit`.
pub const STAT_POINT_TOTAL: u16 = 66;
/// Per-stat Stat Point bound.
pub const STAT_POINT_MAX: u8 = 32;
/// Pinned `adjustLevel`; higher source levels are adjusted down at battle start.
pub const ADJUSTED_LEVEL: u16 = 50;
/// Reference `maxLevel` for submission.
pub const MAX_SOURCE_LEVEL: u16 = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Legal,
    SpeciesClause,
    ItemClause,
    SpeciesNotAllowed,
    IllegalAbility,
    IllegalMove,
    DuplicateMove,
    IllegalItem,
    IllegalNature,
    Level,
    Ivs,
    StatPoints,
    ZeroStatPoints,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegalityReport {
    pub verdict: Verdict,
    /// Index of the first offending member, when the problem is member-scoped.
    pub member: Option<usize>,
}

impl LegalityReport {
    pub fn legal(&self) -> bool {
        self.verdict == Verdict::Legal
    }
}

fn member_report(verdict: Verdict, member: usize) -> LegalityReport {
    LegalityReport {
        verdict,
        member: Some(member),
    }
}

/// Validates one submitted six-member team exactly as the pinned reference
/// does for the rules this format enforces.
pub fn validate_team(dex: &Dex, team: &Team) -> LegalityReport {
    let mut species_seen: Vec<Id> = Vec::with_capacity(6);
    let mut items_seen: Vec<Id> = Vec::with_capacity(6);
    for (index, member) in team.members.iter().enumerate() {
        if member.species == 0 || usize::from(member.species) >= dex.species.len() {
            return member_report(Verdict::SpeciesNotAllowed, index);
        }
        let species = &dex.species[member.species as usize];
        // A submitted Mega form is accepted when its base form is a legal
        // starting species: the reference validates it and the battle starts
        // from the base forme. Legal alternate formes (for example Hisuian
        // Goodra) are starting species in their own right; their base species
        // is not required to be legal.
        let starting = dex.legal_starting_species[member.species as usize];
        let mega_form =
            !starting && species.is_mega && dex.legal_starting_species[species.base_species as usize];
        let species_legal = starting || mega_form;
        if !species_legal {
            return member_report(Verdict::SpeciesNotAllowed, index);
        }
        let ability_ok = if starting {
            member.ability != 0
                && dex.legal_abilities_by_species[member.species as usize].contains(&member.ability)
        } else {
            species.abilities.contains(&member.ability)
        };
        if !ability_ok {
            return member_report(Verdict::IllegalAbility, index);
        }
        if member.item != 0 && !dex.legal_items[member.item as usize] {
            return member_report(Verdict::IllegalItem, index);
        }
        if member.nature == 0 || usize::from(member.nature) >= dex.natures.len() {
            return member_report(Verdict::IllegalNature, index);
        }
        if member.level < ADJUSTED_LEVEL || member.level > MAX_SOURCE_LEVEL {
            return member_report(Verdict::Level, index);
        }
        if member.ivs.iter().any(|value| *value != 31) {
            return member_report(Verdict::Ivs, index);
        }
        if member.points.iter().any(|value| *value > STAT_POINT_MAX) {
            return member_report(Verdict::StatPoints, index);
        }
        let total: u16 = member.points.iter().map(|value| u16::from(*value)).sum();
        if total > STAT_POINT_TOTAL {
            return member_report(Verdict::StatPoints, index);
        }
        // The reference rejects an uninvested Serious set unless the nature is
        // changed to a different neutral nature as an explicit acknowledgement.
        if total == 0 && dex.natures[member.nature as usize].plus.is_none()
            && dex.natures[member.nature as usize].minus.is_none()
            && is_serious(dex, member)
        {
            return member_report(Verdict::ZeroStatPoints, index);
        }
        if has_duplicate_move(member) {
            return member_report(Verdict::DuplicateMove, index);
        }
        if !validate_moves(dex, member, starting) {
            return member_report(Verdict::IllegalMove, index);
        }
        if species_seen.contains(&species.base_species) {
            return LegalityReport {
                verdict: Verdict::SpeciesClause,
                member: Some(index),
            };
        }
        if member.item != 0 && items_seen.contains(&member.item) {
            return LegalityReport {
                verdict: Verdict::ItemClause,
                member: Some(index),
            };
        }
        species_seen.push(species.base_species);
        items_seen.push(member.item);
    }
    LegalityReport {
        verdict: Verdict::Legal,
        member: None,
    }
}

fn is_serious(dex: &Dex, member: &TeamSet) -> bool {
    dex.names["natures"][member.nature as usize] == "serious"
}

fn has_duplicate_move(member: &TeamSet) -> bool {
    member
        .moves
        .iter()
        .enumerate()
        .any(|(index, move_id)| member.moves[..index].contains(move_id))
}

fn validate_moves(dex: &Dex, member: &TeamSet, starting: bool) -> bool {
    if member.moves.is_empty() || member.moves.len() > 4 {
        return false;
    }
    let learnset = if starting {
        &dex.legal_moves_by_species[member.species as usize]
    } else {
        &dex.legal_moves_by_species[dex.species[member.species as usize].base_species as usize]
    };
    member
        .moves
        .iter()
        .all(|move_id| *move_id != 0 && learnset.contains(move_id))
}
