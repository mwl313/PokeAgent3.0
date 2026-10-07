//! Player-safe numeric observation schema `pa3-observation-v1`.
//!
//! Inputs are exclusively a PlayerView and a borrowed immutable Dex. Outcomes,
//! team provenance, hidden sets, RNG and pending opponent actions are not read.
//! 96 fixed rows use knowledge::token_layout; the eight trailing rows and unused
//! event rows are masked zero padding. IDs are u16 embedding categories, floats
//! are normalized without clipping, and flags have independent known masks.
//! Unknown payloads are always zero, even if Known.value contains stale data.
//!
//! Species stats/actual stats/HP/levels use /65535, allocations /32, IVs /31,
//! boosts /6, PP /255, power /65535, accuracy /100, priority /128, turn /65535,
//! event changes /2147483648. Exact battle state is never modified or quantized.
//! All effect entries, type IDs, and persistent revealed moves use uncapped
//! ragged buffers with row ranges; they are not truncated to the token count.
//! Caller-owned buffers retain capacity between calls. Catalogue names are
//! resolved only during Encoder construction, never during encode_into.
//!
//! This schema is a foundation, not full observation readiness: PlayerView does
//! not expose own private effect metadata, and typed Dex lacks arbitrary item /
//! ability callback features. Such unavailable features remain unknown; no
//! hidden-world lookup or invented inference is used to fill them.

use crate::{
    EngineError, Result,
    actions::RequestKind,
    assets::{Category, Dex, Id, Target},
    effects::HitEffect,
    knowledge::{EffectKind, EffectKnowledge, EventKind, Known, OBSERVATION_TOKENS},
    state::PlayerView,
};
use std::collections::BTreeMap;

pub const SCHEMA_VERSION: u16 = 1;
pub const CATEGORY_COUNT: usize = 32;
pub const FLOAT_COUNT: usize = 50;
pub const FLAG_COUNT: usize = 40;

/// Explicit indices are part of the versioned wire schema, not enum casts from
/// engine implementation details. Catalogue IDs remain in separate columns.
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub enum CategoryFeature {
    Role = 0,
    Entity = 1,
    Slot = 2,
    Species = 3,
    BaseSpecies = 4,
    Gender = 5,
    Status = 6,
    Ability = 7,
    Item = 8,
    PreviousItem = 9,
    Nature = 10,
    Move = 11,
    MoveType = 12,
    MoveCategory = 13,
    MoveTarget = 14,
    EventKind = 15,
    EffectKind = 16,
    EffectId = 17,
    EventTarget = 18,
    RequestKind = 19,
    Side = 20,
    Regulation = 21,
    ActiveSlot = 22,
    BoundaryColor = 23,
    PrimaryStatus = 24,
    PrimaryVolatile = 25,
    SideCondition = 26,
    Weather = 27,
    Terrain = 28,
    SelectionOrder = 29,
}
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub enum FloatFeature {
    /// Six fields in HP/Atk/Def/SpA/SpD/Spe order.
    Stats = 0,
    Points = 6,
    Ivs = 12,
    /// Seven fields in Atk/Def/SpA/SpD/Spe/accuracy/evasion order.
    Boosts = 18,
    BaseStats = 25,
    Hp = 31,
    MaxHp = 32,
    HpFraction = 33,
    PublicHpNumerator = 34,
    PublicHpDenominator = 35,
    Level = 36,
    Pp = 37,
    MaxPp = 38,
    BasePp = 39,
    Power = 40,
    Accuracy = 41,
    Priority = 42,
    EventChange = 43,
    Turn = 44,
    CritRatio = 45,
    Weight = 46,
    Recoil = 47,
    Drain = 48,
}
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub enum FlagFeature {
    Selected = 0,
    Active = 1,
    Fainted = 2,
    Transformed = 3,
    Own = 4,
    HpExact = 5,
    AlwaysHit = 6,
    MoveDisabled = 7,
    MoveUsed = 8,
    Contact = 9,
    Protect = 10,
    Sound = 11,
    Powder = 12,
    Pulse = 13,
    Punch = 14,
    Slicing = 15,
    Bite = 16,
    NoPpBoosts = 17,
    IgnoreImmunity = 18,
    Defrost = 19,
    ThawsTarget = 20,
    MegaUsed = 21,
    Slot0Present = 22,
    Slot1Present = 23,
    Slot0Replacement = 24,
    Slot1Replacement = 25,
    Slot0CanMega = 26,
    Slot1CanMega = 27,
    Slot0Trapped = 28,
    Slot1Trapped = 29,
    Slot0MaybeTrapped = 30,
    Slot1MaybeTrapped = 31,
    BenchEligible = 32,
    PreviewEligible = 33,
    MegaForm = 34,
    PrivateEffectsAvailable = 35,
    ItemEffectMetadataAvailable = 36,
    AbilityEffectMetadataAvailable = 37,
    HasRecoil = 38,
    HasDrain = 39,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TokenFeatures {
    pub categories: [u16; CATEGORY_COUNT],
    pub category_known: [bool; CATEGORY_COUNT],
    pub floats: [f32; FLOAT_COUNT],
    pub float_known: [bool; FLOAT_COUNT],
    pub flags: [bool; FLAG_COUNT],
    pub flag_known: [bool; FLAG_COUNT],
}
impl Default for TokenFeatures {
    fn default() -> Self {
        Self {
            categories: [0; CATEGORY_COUNT],
            category_known: [false; CATEGORY_COUNT],
            floats: [0.0; FLOAT_COUNT],
            float_known: [false; FLOAT_COUNT],
            flags: [false; FLAG_COUNT],
            flag_known: [false; FLAG_COUNT],
        }
    }
}
impl TokenFeatures {
    fn cat(&mut self, field: CategoryFeature, value: u16) {
        self.categories[field as usize] = value;
        self.category_known[field as usize] = true;
    }
    fn number(&mut self, field: FloatFeature, value: f32) {
        self.float_at(field as usize, value);
    }
    fn float_at(&mut self, field: usize, value: f32) {
        self.floats[field] = value;
        self.float_known[field] = true;
    }
    fn flag(&mut self, field: FlagFeature, value: bool) {
        self.flags[field as usize] = value;
        self.flag_known[field as usize] = true;
    }
    fn known_cat(&mut self, field: CategoryFeature, value: Known<Id>) {
        if value.known {
            self.cat(field, value.value);
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BufferRange {
    pub start: usize,
    pub len: usize,
}

/// Source uses public entity indices 0..11; source_known distinguishes entity 0
/// from unknown. Absent/unknown durations and stacks are zero, not guessed.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedEffect {
    pub id: Id,
    pub present: bool,
    pub duration: f32,
    pub duration_known: bool,
    pub stacks: f32,
    pub stacks_known: bool,
    pub source: u8,
    pub source_known: bool,
}

/// Complete typed move effect metadata for each encoded known move. Kinds:
/// 1 primary target, 2 primary self, 3 secondary target, 4 secondary self.
/// Effects beyond these typed Dex fields remain explicitly unavailable.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedMoveEffect {
    pub kind: u8,
    pub chance: f32,
    pub status: Id,
    pub volatile: Id,
    pub boosts: [f32; 7],
    pub heal: f32,
    pub heal_known: bool,
}

/// Original own move slots remain separate from current slots and public history.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedBaseMove {
    pub id: Id,
    pub pp: f32,
    pub max_pp: f32,
    pub disabled: bool,
    pub used: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObservationBuffers {
    pub schema_version: u16,
    pub tokens: [TokenFeatures; OBSERVATION_TOKENS],
    pub token_mask: [bool; OBSERVATION_TOKENS],
    pub effects: Vec<EncodedEffect>,
    pub effect_ranges: [BufferRange; OBSERVATION_TOKENS],
    pub repertoire: Vec<Id>,
    pub repertoire_ranges: [BufferRange; OBSERVATION_TOKENS],
    pub types: Vec<Id>,
    pub type_ranges: [BufferRange; OBSERVATION_TOKENS],
    pub base_moves: Vec<EncodedBaseMove>,
    pub base_move_ranges: [BufferRange; OBSERVATION_TOKENS],
    pub move_effects: Vec<EncodedMoveEffect>,
    pub move_effect_ranges: [BufferRange; OBSERVATION_TOKENS],
}
impl Default for ObservationBuffers {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            tokens: std::array::from_fn(|_| TokenFeatures::default()),
            token_mask: [false; OBSERVATION_TOKENS],
            effects: Vec::new(),
            effect_ranges: [BufferRange::default(); OBSERVATION_TOKENS],
            repertoire: Vec::new(),
            repertoire_ranges: [BufferRange::default(); OBSERVATION_TOKENS],
            types: Vec::new(),
            type_ranges: [BufferRange::default(); OBSERVATION_TOKENS],
            base_moves: Vec::new(),
            base_move_ranges: [BufferRange::default(); OBSERVATION_TOKENS],
            move_effects: Vec::new(),
            move_effect_ranges: [BufferRange::default(); OBSERVATION_TOKENS],
        }
    }
}
impl ObservationBuffers {
    fn clear(&mut self) {
        self.schema_version = SCHEMA_VERSION;
        self.tokens.fill(TokenFeatures::default());
        self.token_mask.fill(false);
        self.effect_ranges.fill(BufferRange::default());
        self.repertoire_ranges.fill(BufferRange::default());
        self.type_ranges.fill(BufferRange::default());
        self.move_effect_ranges.fill(BufferRange::default());
        self.base_move_ranges.fill(BufferRange::default());
        self.effects.clear();
        self.repertoire.clear();
        self.types.clear();
        self.move_effects.clear();
        self.base_moves.clear();
    }
    fn row(&mut self, index: usize, role: u16) -> &mut TokenFeatures {
        self.token_mask[index] = true;
        self.tokens[index].cat(CategoryFeature::Role, role);
        &mut self.tokens[index]
    }
}

/// Caller-owned batch output with a retained high-water pool. Sparse and empty
/// requests hide inactive rows without discarding their ragged allocations.
/// Only the active slice is public, compared, or printed; previous inactive
/// player views are never exposed through Debug or Deref.
#[derive(Clone, Default)]
pub struct ObservationBatchBuffers {
    storage: Vec<ObservationBuffers>,
    active_len: usize,
}

impl ObservationBatchBuffers {
    pub fn as_slice(&self) -> &[ObservationBuffers] {
        &self.storage[..self.active_len]
    }

    pub fn len(&self) -> usize {
        self.active_len
    }

    pub fn is_empty(&self) -> bool {
        self.active_len == 0
    }

    pub(crate) fn prepare(&mut self, len: usize) {
        if self.storage.len() < len {
            self.storage.resize_with(len, ObservationBuffers::default);
        }
        self.active_len = len;
    }

    pub(crate) fn active_mut(&mut self) -> &mut [ObservationBuffers] {
        &mut self.storage[..self.active_len]
    }
}

impl std::ops::Deref for ObservationBatchBuffers {
    type Target = [ObservationBuffers];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl std::fmt::Debug for ObservationBatchBuffers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ObservationBatchBuffers")
            .field(&self.as_slice())
            .finish()
    }
}

impl PartialEq for ObservationBatchBuffers {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

/// Construct once beside an immutable shared Dex; encoding only visits local
/// view entries and directly indexed metadata, never full catalogue scans.
pub struct Encoder<'a> {
    dex: &'a Dex,
    limits: [usize; 7], // species, moves, abilities, items, conditions, types, natures
}
impl<'a> Encoder<'a> {
    /// Batch integration must use the exact immutable Dex whose catalogue was
    /// validated at construction, even if a different object has equal lengths.
    pub fn matches_dex(&self, dex: &Dex) -> bool {
        std::ptr::eq(self.dex, dex)
    }

    pub fn new(dex: &'a Dex) -> Result<Self> {
        let encoder = Self::from_validated_dex(dex)?;
        // Cold catalogue validation; never rerun in the observation hot path.
        for (i, species) in dex.species.iter().enumerate().skip(1) {
            if usize::from(species.id) != i {
                return Err(EngineError::AssetMismatch("species observation ID".into()));
            }
            for &id in &species.types {
                encoder.id(id, 5, false)?;
            }
        }
        for (i, m) in dex.moves.iter().enumerate().skip(1) {
            if usize::from(m.id) != i {
                return Err(EngineError::AssetMismatch("move observation ID".into()));
            }
            encoder.id(m.move_type, 5, false)?;
            for id in [m.side_condition, m.weather, m.terrain] {
                encoder.id(id, 4, true)?;
            }
            encoder.hit_valid(&m.hit)?;
            if let Some(effect) = &m.self_effect {
                encoder.hit_valid(effect)?;
            }
            for secondary in &m.secondaries {
                if secondary.chance > 100 {
                    return Err(EngineError::AssetMismatch(
                        "secondary observation chance".into(),
                    ));
                }
                encoder.hit_valid(&secondary.target)?;
                if let Some(effect) = &secondary.own {
                    encoder.hit_valid(effect)?;
                }
            }
            for ratio in [m.recoil, m.drain].into_iter().flatten() {
                if ratio[1] == 0 {
                    return Err(EngineError::AssetMismatch("observation move ratio".into()));
                }
            }
        }
        Ok(encoder)
    }

    /// Construct an encoder for an immutable Dex whose catalogue already passed
    /// [`Encoder::new`]. Only catalogue lengths are re-read; the cold
    /// full-catalogue validation scan is skipped. The caller must have validated
    /// this exact Dex (same identity for `matches_dex`) at startup; the PyO3
    /// binding does so once per engine and then reuses this cheap constructor.
    pub fn from_validated_dex(dex: &'a Dex) -> Result<Self> {
        let mut limits = [0; 7];
        for (index, name) in [
            "species",
            "moves",
            "abilities",
            "items",
            "conditions",
            "types",
            "natures",
        ]
        .into_iter()
        .enumerate()
        {
            limits[index] = dex
                .names
                .get(name)
                .ok_or_else(|| {
                    EngineError::AssetMismatch(format!("missing observation catalogue {name}"))
                })?
                .len();
        }
        if dex.species.len() != limits[0] || dex.moves.len() != limits[1] {
            return Err(EngineError::AssetMismatch(
                "observation catalogue length".into(),
            ));
        }
        Ok(Self { dex, limits })
    }
    fn id(&self, id: Id, catalogue: usize, zero: bool) -> Result<()> {
        if (!zero && id == 0) || usize::from(id) >= self.limits[catalogue] {
            return Err(EngineError::InvalidInput("observation catalogue ID".into()));
        }
        Ok(())
    }
    fn hit_valid(&self, effect: &HitEffect) -> Result<()> {
        self.id(effect.status, 4, true)?;
        self.id(effect.volatile, 4, true)?;
        if effect.heal.is_some_and(|ratio| ratio[1] == 0) {
            return Err(EngineError::AssetMismatch(
                "observation heal denominator".into(),
            ));
        }
        Ok(())
    }
    fn effects_valid(&self, effects: &BTreeMap<Id, EffectKnowledge>) -> Result<()> {
        for (&id, effect) in effects {
            self.id(id, 4, false)?;
            if effect.source.known && effect.source.value >= 12 {
                return Err(EngineError::InvalidInput(
                    "observation effect source".into(),
                ));
            }
        }
        Ok(())
    }

    /// Compile-time completeness audit for the player-knowledge contract.
    ///
    /// Every field of `Knowledge`, `PublicPokemon`, `EffectKnowledge` and
    /// `SemanticEvent` is destructured exhaustively (no `..`), so adding a
    /// knowledge field fails compilation until its author classifies it here:
    /// encoded by `encode_into`, or deliberately excluded with a reason.
    ///
    /// ENCODED: `mega_used`, `pokemon`, `events`, `field`, `sides`; every
    /// `PublicPokemon` field except `revealed_move_repertoire`; every
    /// `EffectKnowledge` field; every `SemanticEvent` field.
    ///
    /// DELIBERATELY NOT ENCODED: `PublicPokemon::revealed_move_repertoire`
    /// (folded into the fixed repertoire rows by `encode_into`).
    fn knowledge_field_audit(view: &crate::knowledge::Knowledge) {
        let crate::knowledge::Knowledge {
            ref mega_used,
            ref pokemon,
            ref events,
            ref field,
            ref sides,
        } = *view;
        let _ = (mega_used, pokemon, events, field, sides);
        for mon in pokemon {
            let crate::knowledge::PublicPokemon {
                ref species,
                ref types,
                ref gender,
                ref health,
                ref status,
                ref boosts,
                ref ability,
                ref item,
                ref previous_item,
                ref selected,
                ref active_slot,
                ref fainted,
                ref current_moves,
                ref revealed_move_repertoire,
                ref effects,
            } = *mon;
            let _ = (
                species,
                types,
                gender,
                health,
                status,
                boosts,
                ability,
                item,
                previous_item,
                selected,
                active_slot,
                fainted,
                current_moves,
                revealed_move_repertoire,
                effects,
            );
        }
        for effect in field.values().chain(sides.iter().flat_map(|side| side.values())) {
            let crate::knowledge::EffectKnowledge {
                present,
                ref duration,
                ref stacks,
                ref source,
            } = *effect;
            let _ = (present, duration, stacks, source);
        }
        for event in events {
            let crate::knowledge::SemanticEvent {
                kind,
                subject,
                target,
                effect,
                effect_kind,
                value,
                ref health,
            } = *event;
            let _ = (kind, subject, target, effect, effect_kind, value, health);
        }
    }
    fn validate(&self, view: &PlayerView) -> Result<()> {
        Self::knowledge_field_audit(&view.knowledge);
        view.request.validate()?;
        if view.knowledge.events.len() > 24 {
            return Err(EngineError::InvalidInput(
                "observation recent event window exceeds 24".into(),
            ));
        }
        self.effects_valid(&view.knowledge.field)?;
        for side in &view.knowledge.sides {
            self.effects_valid(side)?;
        }
        for mon in &view.own.pokemon {
            self.id(mon.species, 0, false)?;
            self.id(mon.base_species, 0, false)?;
            self.id(mon.ability, 2, true)?;
            self.id(mon.item, 3, true)?;
            self.id(mon.previous_item, 3, true)?;
            self.id(mon.status, 4, true)?;
            self.id(mon.nature, 6, false)?;
            if mon.stats[0] == 0
                || mon.hp > mon.stats[0]
                || mon.moves.len() > 4
                || mon.active_slot.is_some_and(|s| s >= 2)
                || mon.gender > 2
            {
                return Err(EngineError::InvalidInput("observation own Pokémon".into()));
            }
            for &kind in &mon.types {
                // Type id 0 is the `'???'` placeholder (Double Shock).
                self.id(kind, 5, true)?;
            }
            for m in mon.moves.iter().chain(&mon.base_moves) {
                self.id(m.id, 1, false)?;
                if m.pp > m.max_pp {
                    return Err(EngineError::InvalidInput("observation PP".into()));
                }
            }
        }
        for mon in &view.knowledge.pokemon {
            self.id(mon.species, 0, true)?;
            for &kind in &mon.types {
                // Type id 0 is the `'???'` placeholder (Double Shock).
                self.id(kind, 5, true)?;
            }
            for (known, cat) in [
                (mon.status, 4),
                (mon.ability, 2),
                (mon.item, 3),
                (mon.previous_item, 3),
            ] {
                if known.known {
                    self.id(known.value, cat, true)?;
                }
            }
            if mon.active_slot.is_some_and(|s| s >= 2) || mon.gender.known && mon.gender.value > 2 {
                return Err(EngineError::InvalidInput(
                    "observation public Pokémon".into(),
                ));
            }
            if mon.health.known {
                validate_health(mon.health.value)?;
            }
            for m in &mon.current_moves {
                if m.known {
                    self.id(m.value, 1, false)?;
                }
            }
            for &m in &mon.revealed_move_repertoire {
                self.id(m, 1, false)?;
            }
            self.effects_valid(&mon.effects)?;
        }
        for event in &view.knowledge.events {
            if event.subject >= 12 || event.target.is_some_and(|t| t >= 12) {
                return Err(EngineError::InvalidInput("observation event entity".into()));
            }
            match event.effect_kind {
                EffectKind::None if event.effect != 0 => {
                    return Err(EngineError::InvalidInput(
                        "observation none effect ID".into(),
                    ));
                }
                EffectKind::None => (),
                EffectKind::Move => self.id(event.effect, 1, false)?,
                EffectKind::Ability => self.id(event.effect, 2, false)?,
                EffectKind::Item => self.id(event.effect, 3, false)?,
                EffectKind::Condition => self.id(event.effect, 4, false)?,
                EffectKind::Species => self.id(event.effect, 0, false)?,
                EffectKind::Stat if event.effect >= 7 => {
                    return Err(EngineError::InvalidInput("observation stat ID".into()));
                }
                EffectKind::Stat => (),
            }
            if let Some(health) = event.health {
                validate_health(health)?;
            }
        }
        Ok(())
    }

    /// On error the existing output is untouched. On success all prior rows and
    /// ragged lengths are replaced, retaining vector allocation capacity.
    pub fn encode_into(&self, view: &PlayerView, out: &mut ObservationBuffers) -> Result<()> {
        self.validate(view)?;
        out.clear();
        {
            let row = out.row(0, 1);
            row.cat(CategoryFeature::RequestKind, request_id(view.request.kind));
            row.cat(CategoryFeature::Side, view.side.index() as u16 + 1);
            row.cat(CategoryFeature::Regulation, 1); // pinned Champions M-C
            row.number(FloatFeature::Turn, f32::from(view.turn) / 65535.0);
            row.flag(FlagFeature::MegaUsed, view.own.mega_used);
            for (i, slot) in view.request.slots.iter().enumerate() {
                for (index, value) in [
                    (22 + i, slot.present),
                    (24 + i, slot.requires_replacement),
                    (26 + i, slot.can_mega),
                    (28 + i, slot.trapped),
                    (30 + i, slot.maybe_trapped),
                ] {
                    row.flags[index] = value;
                    row.flag_known[index] = true;
                }
            }
        }
        out.row(1, 2);
        append_effects(out, 1, &view.knowledge.field);
        for side in 0..2 {
            let row = out.row(2 + side, 3);
            row.cat(CategoryFeature::Side, side as u16 + 1); // viewer relative
            row.flag(
                FlagFeature::MegaUsed,
                if side == 0 {
                    view.own.mega_used
                } else {
                    view.knowledge.mega_used[1]
                },
            );
            append_effects(out, 2 + side, &view.knowledge.sides[side]);
        }
        for entity in 0..12 {
            let index = 4 + entity;
            let public = &view.knowledge.pokemon[entity];
            let species = if entity < 6 {
                view.own.pokemon[entity].species
            } else {
                public.species
            };
            let row = out.row(index, 4);
            row.cat(CategoryFeature::Entity, entity as u16 + 1);
            row.flag(FlagFeature::Own, entity < 6);
            row.flag(FlagFeature::PrivateEffectsAvailable, false);
            row.flag(FlagFeature::ItemEffectMetadataAvailable, false);
            row.flag(FlagFeature::AbilityEffectMetadataAvailable, false);
            if species != 0 {
                row.cat(CategoryFeature::Species, species);
                let metadata = &self.dex.species[species as usize];
                for (i, &value) in metadata.base_stats.iter().enumerate() {
                    row.float_at(
                        FloatFeature::BaseStats as usize + i,
                        f32::from(value) / 65535.0,
                    );
                }
                row.number(
                    FloatFeature::Weight,
                    metadata.weight_hg as f32 / u32::MAX as f32,
                );
                row.flag(FlagFeature::MegaForm, metadata.is_mega);
            }
            if entity < 6 {
                let own = &view.own.pokemon[entity];
                row.cat(CategoryFeature::BaseSpecies, own.base_species);
                row.cat(CategoryFeature::Gender, u16::from(own.gender));
                row.cat(CategoryFeature::Ability, own.ability);
                row.cat(CategoryFeature::Item, own.item);
                row.cat(CategoryFeature::PreviousItem, own.previous_item);
                row.cat(CategoryFeature::Nature, own.nature);
                row.cat(CategoryFeature::Status, own.status);
                row.number(FloatFeature::Level, f32::from(own.level) / 65535.0);
                row.number(FloatFeature::Hp, f32::from(own.hp) / 65535.0);
                row.number(FloatFeature::MaxHp, f32::from(own.stats[0]) / 65535.0);
                row.number(
                    FloatFeature::HpFraction,
                    f32::from(own.hp) / f32::from(own.stats[0]),
                );
                for i in 0..6 {
                    row.float_at(
                        FloatFeature::Stats as usize + i,
                        f32::from(own.stats[i]) / 65535.0,
                    );
                    row.float_at(
                        FloatFeature::Points as usize + i,
                        f32::from(own.points[i]) / 32.0,
                    );
                    row.float_at(FloatFeature::Ivs as usize + i, f32::from(own.ivs[i]) / 31.0);
                }
                for i in 0..7 {
                    row.float_at(
                        FloatFeature::Boosts as usize + i,
                        f32::from(own.boosts[i]) / 6.0,
                    );
                }
                row.flag(FlagFeature::HpExact, true);
                row.flag(FlagFeature::Selected, own.selected);
                row.flag(FlagFeature::Active, own.active_slot.is_some());
                row.flag(FlagFeature::Fainted, own.fainted);
                row.flag(FlagFeature::Transformed, own.transformed);
                row.flag(
                    FlagFeature::BenchEligible,
                    view.request.bench.contains(&(entity as u8)),
                );
                row.flag(
                    FlagFeature::PreviewEligible,
                    view.request.preview_roster.contains(&(entity as u8)),
                );
                if let Some(slot) = own.active_slot {
                    row.cat(CategoryFeature::ActiveSlot, u16::from(slot) + 1);
                } else {
                    row.cat(CategoryFeature::ActiveSlot, 0);
                }
                if let Some(order) = view.own.selected_order {
                    row.cat(
                        CategoryFeature::SelectionOrder,
                        order
                            .iter()
                            .position(|&r| usize::from(r) == entity)
                            .map_or(0, |i| i as u16 + 1),
                    );
                }
                append_types(out, index, &own.types);
                out.base_move_ranges[index] = BufferRange {
                    start: out.base_moves.len(),
                    len: own.base_moves.len(),
                };
                out.base_moves
                    .extend(own.base_moves.iter().map(|m| EncodedBaseMove {
                        id: m.id,
                        pp: f32::from(m.pp) / 255.0,
                        max_pp: f32::from(m.max_pp) / 255.0,
                        disabled: m.disabled,
                        used: m.used,
                    }));
            } else {
                if public.gender.known {
                    row.cat(CategoryFeature::Gender, u16::from(public.gender.value));
                }
                row.known_cat(CategoryFeature::Ability, public.ability);
                row.known_cat(CategoryFeature::Item, public.item);
                row.known_cat(CategoryFeature::PreviousItem, public.previous_item);
                row.known_cat(CategoryFeature::Status, public.status);
                if public.health.known {
                    put_health(row, public.health.value);
                }
                for i in 0..7 {
                    row.float_at(
                        FloatFeature::Boosts as usize + i,
                        f32::from(public.boosts[i]) / 6.0,
                    );
                }
                row.flag(FlagFeature::HpExact, false);
                if public.selected.known {
                    row.flag(FlagFeature::Selected, public.selected.value);
                }
                row.flag(FlagFeature::Active, public.active_slot.is_some());
                row.flag(FlagFeature::Fainted, public.fainted);
                row.cat(
                    CategoryFeature::ActiveSlot,
                    public.active_slot.map_or(0, |s| u16::from(s) + 1),
                );
                append_types(out, index, &public.types);
            }
            append_effects(out, index, &public.effects);
            out.repertoire_ranges[index] = BufferRange {
                start: out.repertoire.len(),
                len: public.revealed_move_repertoire.len(),
            };
            out.repertoire
                .extend_from_slice(&public.revealed_move_repertoire);
            for slot in 0..4 {
                let move_index = 16 + entity * 4 + slot;
                let row = out.row(move_index, 5);
                row.cat(CategoryFeature::Entity, entity as u16 + 1);
                row.cat(CategoryFeature::Slot, slot as u16 + 1);
                let id = if entity < 6 {
                    view.own.pokemon[entity].moves.get(slot).map(|m| {
                        row.number(FloatFeature::Pp, f32::from(m.pp) / 255.0);
                        row.number(FloatFeature::MaxPp, f32::from(m.max_pp) / 255.0);
                        row.flag(FlagFeature::MoveDisabled, m.disabled);
                        row.flag(FlagFeature::MoveUsed, m.used);
                        m.id
                    })
                } else {
                    let m = public.current_moves[slot];
                    m.known.then_some(m.value)
                };
                if let Some(id) = id {
                    self.move_metadata(out, move_index, id);
                }
            }
        }
        for (i, event) in view.knowledge.events.iter().enumerate() {
            let row = out.row(64 + i, 6);
            row.cat(CategoryFeature::Slot, i as u16 + 1); // oldest to newest
            row.cat(CategoryFeature::Entity, u16::from(event.subject) + 1);
            row.cat(CategoryFeature::EventKind, event_id(event.kind));
            row.cat(
                CategoryFeature::EffectKind,
                effect_kind_id(event.effect_kind),
            );
            row.cat(CategoryFeature::EffectId, event.effect);
            row.cat(
                CategoryFeature::EventTarget,
                event.target.map_or(0, |t| u16::from(t) + 1),
            );
            // Audience filtering replaces private opponent integers with zero.
            // Those placeholders are unknown, not observed changes of zero.
            let filtered = event.subject >= 6
                && matches!(
                    event.kind,
                    EventKind::Damage
                        | EventKind::Heal
                        | EventKind::SideEffectStart
                        | EventKind::FieldEffectStart
                );
            if !filtered {
                row.number(FloatFeature::EventChange, event.value as f32 / 2147483648.0);
            }
            if let Some(health) = event.health {
                put_health(row, health);
            }
        }
        Ok(())
    }
    fn move_metadata(&self, out: &mut ObservationBuffers, index: usize, id: Id) {
        let m = &self.dex.moves[id as usize];
        let row = &mut out.tokens[index];
        row.cat(CategoryFeature::Move, id);
        row.cat(CategoryFeature::MoveType, m.move_type);
        row.cat(
            CategoryFeature::MoveCategory,
            match m.category {
                Category::Physical => 1,
                Category::Special => 2,
                Category::Status => 3,
            },
        );
        row.cat(CategoryFeature::MoveTarget, target_id(m.target));
        row.cat(CategoryFeature::PrimaryStatus, m.hit.status);
        row.cat(CategoryFeature::PrimaryVolatile, m.hit.volatile);
        row.cat(CategoryFeature::SideCondition, m.side_condition);
        row.cat(CategoryFeature::Weather, m.weather);
        row.cat(CategoryFeature::Terrain, m.terrain);
        row.number(FloatFeature::BasePp, f32::from(m.pp) / 255.0);
        row.number(FloatFeature::Power, f32::from(m.power) / 65535.0);
        row.number(FloatFeature::Priority, f32::from(m.priority) / 128.0);
        row.number(FloatFeature::CritRatio, f32::from(m.crit_ratio) / 255.0);
        if let Some([n, d]) = m.recoil {
            row.number(FloatFeature::Recoil, f32::from(n) / f32::from(d));
        }
        if let Some([n, d]) = m.drain {
            row.number(FloatFeature::Drain, f32::from(n) / f32::from(d));
        }
        row.flag(FlagFeature::AlwaysHit, m.accuracy.is_none());
        row.flag(FlagFeature::HasRecoil, m.recoil.is_some());
        row.flag(FlagFeature::HasDrain, m.drain.is_some());
        if let Some(accuracy) = m.accuracy {
            row.number(FloatFeature::Accuracy, f32::from(accuracy) / 100.0);
        }
        for (field, flag) in [
            (FlagFeature::Contact, m.contact),
            (FlagFeature::Protect, m.protect),
            (FlagFeature::Sound, m.sound),
            (FlagFeature::Powder, m.powder),
            (FlagFeature::Pulse, m.pulse),
            (FlagFeature::Punch, m.punch),
            (FlagFeature::Slicing, m.slicing),
            (FlagFeature::Bite, m.bite),
            (FlagFeature::NoPpBoosts, m.no_pp_boosts),
            (FlagFeature::IgnoreImmunity, m.ignore_immunity),
            (FlagFeature::Defrost, m.defrost),
            (FlagFeature::ThawsTarget, m.thaws_target),
        ] {
            row.flag(field, flag);
        }
        let start = out.move_effects.len();
        append_move_effect(&mut out.move_effects, 1, 100, &m.hit);
        if let Some(effect) = &m.self_effect {
            append_move_effect(&mut out.move_effects, 2, 100, effect);
        }
        for secondary in &m.secondaries {
            append_move_effect(
                &mut out.move_effects,
                3,
                secondary.chance,
                &secondary.target,
            );
            if let Some(effect) = &secondary.own {
                append_move_effect(&mut out.move_effects, 4, secondary.chance, effect);
            }
        }
        out.move_effect_ranges[index] = BufferRange {
            start,
            len: out.move_effects.len() - start,
        };
    }
}

fn append_types(out: &mut ObservationBuffers, index: usize, types: &[Id]) {
    out.type_ranges[index] = BufferRange {
        start: out.types.len(),
        len: types.len(),
    };
    out.types.extend_from_slice(types);
}
fn append_effects(
    out: &mut ObservationBuffers,
    index: usize,
    effects: &BTreeMap<Id, EffectKnowledge>,
) {
    let start = out.effects.len();
    for (&id, effect) in effects {
        out.effects.push(EncodedEffect {
            id,
            present: effect.present,
            duration: if effect.duration.known {
                f32::from(effect.duration.value) / 32768.0
            } else {
                0.0
            },
            duration_known: effect.duration.known,
            stacks: if effect.stacks.known {
                f32::from(effect.stacks.value) / 32768.0
            } else {
                0.0
            },
            stacks_known: effect.stacks.known,
            source: if effect.source.known {
                effect.source.value
            } else {
                0
            },
            source_known: effect.source.known,
        });
    }
    out.effect_ranges[index] = BufferRange {
        start,
        len: out.effects.len() - start,
    };
}
fn append_move_effect(out: &mut Vec<EncodedMoveEffect>, kind: u8, chance: u8, effect: &HitEffect) {
    out.push(EncodedMoveEffect {
        kind,
        chance: f32::from(chance) / 100.0,
        status: effect.status,
        volatile: effect.volatile,
        boosts: effect.boosts.map(|v| f32::from(v) / 6.0),
        heal: effect
            .heal
            .map_or(0.0, |[n, d]| f32::from(n) / f32::from(d)),
        heal_known: effect.heal.is_some(),
    });
}
fn validate_health(health: crate::knowledge::HealthDisplay) -> Result<()> {
    if health.denominator == 0 || health.numerator > health.denominator || health.boundary_color > 3
    {
        return Err(EngineError::InvalidInput(
            "observation public health".into(),
        ));
    }
    Ok(())
}
fn put_health(row: &mut TokenFeatures, health: crate::knowledge::HealthDisplay) {
    row.number(
        FloatFeature::PublicHpNumerator,
        f32::from(health.numerator) / 65535.0,
    );
    row.number(
        FloatFeature::PublicHpDenominator,
        f32::from(health.denominator) / 65535.0,
    );
    row.number(
        FloatFeature::HpFraction,
        f32::from(health.numerator) / f32::from(health.denominator),
    );
    row.cat(
        CategoryFeature::BoundaryColor,
        u16::from(health.boundary_color),
    );
}
fn request_id(kind: RequestKind) -> u16 {
    match kind {
        RequestKind::Preview => 1,
        RequestKind::Normal => 2,
        RequestKind::Replacement => 3,
        RequestKind::Wait => 4,
        RequestKind::Finished => 5,
    }
}
fn target_id(target: Target) -> u16 {
    match target {
        Target::Normal => 1,
        Target::AdjacentFoe => 2,
        Target::AdjacentAlly => 3,
        Target::AdjacentAllyOrSelf => 4,
        Target::Any => 5,
        Target::RandomNormal => 6,
        Target::SelfOnly => 7,
        Target::AllAdjacent => 8,
        Target::AllAdjacentFoes => 9,
        Target::All => 10,
        Target::AllySide => 11,
        Target::FoeSide => 12,
        Target::AllyTeam => 13,
        Target::Allies => 14,
        Target::Scripted => 15,
    }
}
fn effect_kind_id(kind: EffectKind) -> u16 {
    match kind {
        EffectKind::None => 0,
        EffectKind::Move => 1,
        EffectKind::Ability => 2,
        EffectKind::Item => 3,
        EffectKind::Condition => 4,
        EffectKind::Species => 5,
        EffectKind::Stat => 6,
    }
}
fn event_id(kind: EventKind) -> u16 {
    match kind {
        EventKind::Switch => 1,
        EventKind::Move => 2,
        EventKind::Damage => 3,
        EventKind::Heal => 4,
        EventKind::Faint => 5,
        EventKind::Status => 6,
        EventKind::CureStatus => 7,
        EventKind::Ability => 8,
        EventKind::Item => 9,
        EventKind::EndItem => 10,
        EventKind::Boost => 11,
        EventKind::Forme => 12,
        EventKind::Mega => 13,
        EventKind::EffectStart => 14,
        EventKind::EffectEnd => 15,
        EventKind::SideEffectStart => 16,
        EventKind::SideEffectEnd => 17,
        EventKind::FieldEffectStart => 18,
        EventKind::FieldEffectEnd => 19,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_high_water_pool_hides_inactive_rows_and_retains_ragged_storage() {
        let mut batch = ObservationBatchBuffers::default();
        batch.prepare(2);
        batch.active_mut()[0].types.extend_from_slice(&[1, 2]);
        batch.active_mut()[1].types.extend_from_slice(&[3, 4, 5]);
        batch.active_mut()[1].tokens[0].categories[0] = 55555;
        let rows = batch.as_ptr();
        let types = batch[1].types.as_ptr();
        batch.prepare(1);
        assert_eq!(batch.len(), 1);
        assert!(!format!("{batch:?}").contains("55555"));
        batch.prepare(0);
        assert!(batch.is_empty());
        assert_eq!(format!("{batch:?}"), "ObservationBatchBuffers([])");
        assert_eq!(batch, ObservationBatchBuffers::default());
        assert!(batch.clone().as_slice().is_empty());
        batch.prepare(2);
        assert_eq!(batch.as_ptr(), rows);
        assert_eq!(batch[1].types.as_ptr(), types);
        assert_eq!(batch[1].types, [3, 4, 5]);
    }
}
