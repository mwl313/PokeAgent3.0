//! Player knowledge is updated from audience-filtered numeric semantic events.
//! Observation construction never accepts an opponent's private battle state.
use crate::{EngineError, Result, assets::Id};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const OBSERVATION_TOKENS: usize = 96;
pub const ACTIVE_TOKENS: usize = 88;
pub const EVENT_CAPACITY: usize = 24;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Known<T> {
    pub value: T,
    pub known: bool,
}
impl<T> Known<T> {
    pub fn new(value: T) -> Self {
        Self { value, known: true }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealthDisplay {
    pub numerator: u16,
    pub denominator: u16,
    /// Additional reference colour suffix at 20%/50%: 0 none, 1 red, 2 yellow, 3 green.
    pub boundary_color: u8,
}

pub fn public_health(hp: u16, max_hp: u16) -> HealthDisplay {
    assert!(max_hp > 0 && hp <= max_hp);
    let percent = if hp == 0 {
        0
    } else {
        (100 * u32::from(hp) / u32::from(max_hp)).max(1) as u16
    };
    let boundary_color = match percent {
        20 => {
            if u32::from(hp) * 5 > u32::from(max_hp) {
                2
            } else {
                1
            }
        }
        50 => {
            if u32::from(hp) * 2 > u32::from(max_hp) {
                3
            } else {
                2
            }
        }
        _ => 0,
    };
    HealthDisplay {
        numerator: percent,
        denominator: 100,
        boundary_color,
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectKnowledge {
    pub present: bool,
    pub duration: Known<i16>,
    pub stacks: Known<i16>,
    /// Public observation entity index, never a hidden world-state index.
    pub source: Known<u8>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicPokemon {
    pub species: Id,
    pub types: Vec<Id>,
    pub gender: Known<u8>,
    pub health: Known<HealthDisplay>,
    pub status: Known<Id>,
    pub boosts: [i8; 7],
    pub ability: Known<Id>,
    pub item: Known<Id>,
    pub previous_item: Known<Id>,
    pub selected: Known<bool>,
    pub active_slot: Option<u8>,
    pub fainted: bool,
    pub current_moves: [Known<Id>; 4],
    /// Grows independently of four current move slots (Transform/Mimic).
    pub revealed_move_repertoire: Vec<Id>,
    pub effects: BTreeMap<Id, EffectKnowledge>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventKind {
    Switch,
    Move,
    Damage,
    Heal,
    Faint,
    Status,
    CureStatus,
    Ability,
    Item,
    EndItem,
    Boost,
    Forme,
    Mega,
    EffectStart,
    EffectEnd,
    SideEffectStart,
    SideEffectEnd,
    FieldEffectStart,
    FieldEffectEnd,
}

/// IDs belong to separate compact catalogues; a move ID and item ID can have
/// the same numeric value. Events retain the catalogue explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum EffectKind {
    None,
    Move,
    Ability,
    Item,
    Condition,
    Species,
    Stat,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum EffectRef {
    None,
    Move(Id),
    Ability(Id),
    Item(Id),
    Condition(Id),
    Species(Id),
    Stat(Id),
}
impl EffectRef {
    pub fn id(self) -> Id {
        match self {
            Self::None => 0,
            Self::Move(id)
            | Self::Ability(id)
            | Self::Item(id)
            | Self::Condition(id)
            | Self::Species(id)
            | Self::Stat(id) => id,
        }
    }
    pub fn kind(self) -> EffectKind {
        match self {
            Self::None => EffectKind::None,
            Self::Move(_) => EffectKind::Move,
            Self::Ability(_) => EffectKind::Ability,
            Self::Item(_) => EffectKind::Item,
            Self::Condition(_) => EffectKind::Condition,
            Self::Species(_) => EffectKind::Species,
            Self::Stat(_) => EffectKind::Stat,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticEvent {
    pub kind: EventKind,
    /// Indices 0..5 = own public entities, 6..11 = opposing preview entities.
    pub subject: u8,
    /// Public target for move/other events. For EffectStart, Some explicitly
    /// identifies the publicly observed effect source; None adds no source
    /// knowledge. Producers must never fill this from hidden provenance.
    pub target: Option<u8>,
    pub effect: Id,
    pub effect_kind: EffectKind,
    pub value: i32,
    pub health: Option<HealthDisplay>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Knowledge {
    /// Public resource use, relative to this viewer: own side then opponent.
    pub mega_used: [bool; 2],
    pub pokemon: [PublicPokemon; 12],
    pub events: Vec<SemanticEvent>,
    pub field: BTreeMap<Id, EffectKnowledge>,
    pub sides: [BTreeMap<Id, EffectKnowledge>; 2],
}

impl Default for Knowledge {
    fn default() -> Self {
        Self {
            mega_used: [false; 2],
            pokemon: std::array::from_fn(|_| PublicPokemon::default()),
            events: Vec::with_capacity(EVENT_CAPACITY),
            field: BTreeMap::new(),
            sides: Default::default(),
        }
    }
}

impl Knowledge {
    pub fn preview(
        &mut self,
        entity: u8,
        species: Id,
        types: Vec<Id>,
        gender: Known<u8>,
    ) -> Result<()> {
        let mon = self
            .pokemon
            .get_mut(entity as usize)
            .ok_or_else(|| EngineError::InvalidInput("public entity".into()))?;
        *mon = PublicPokemon {
            species,
            types,
            gender,
            ..Default::default()
        };
        Ok(())
    }

    pub fn reveal_move(&mut self, entity: u8, move_id: Id) -> Result<()> {
        let mon = self
            .pokemon
            .get_mut(entity as usize)
            .ok_or_else(|| EngineError::InvalidInput("public entity".into()))?;
        if !mon.revealed_move_repertoire.contains(&move_id) {
            mon.revealed_move_repertoire.push(move_id);
        }
        if !mon
            .current_moves
            .iter()
            .any(|m| m.known && m.value == move_id)
            && let Some(slot) = mon.current_moves.iter_mut().find(|m| !m.known)
        {
            *slot = Known::new(move_id);
        }
        Ok(())
    }

    /// A visible moveset replacement does not erase historical reveals.
    pub fn replace_current_moves(&mut self, entity: u8, moves: [Known<Id>; 4]) -> Result<()> {
        let mon = self
            .pokemon
            .get_mut(entity as usize)
            .ok_or_else(|| EngineError::InvalidInput("public entity".into()))?;
        mon.current_moves = moves;
        for m in moves {
            if m.known && !mon.revealed_move_repertoire.contains(&m.value) {
                mon.revealed_move_repertoire.push(m.value);
            }
        }
        Ok(())
    }

    pub fn apply(&mut self, event: SemanticEvent) -> Result<()> {
        if event.subject >= 12 || event.target.is_some_and(|x| x >= 12) {
            return Err(EngineError::InvalidInput("public event entity".into()));
        }
        if event.kind == EventKind::Switch && !(0..=1).contains(&event.value)
            || event.kind == EventKind::Boost && event.effect >= 7
            || matches!(
                event.kind,
                EventKind::SideEffectStart | EventKind::FieldEffectStart
            ) && !(0..=i32::from(i16::MAX)).contains(&event.value)
            || event.health.is_some_and(|h| {
                h.denominator == 0 || h.numerator > h.denominator || h.boundary_color > 3
            })
        {
            return Err(EngineError::InvalidInput("public event payload".into()));
        }
        let expected_kind = match event.kind {
            EventKind::Move => Some(EffectKind::Move),
            EventKind::Ability => Some(EffectKind::Ability),
            EventKind::Item | EventKind::EndItem => Some(EffectKind::Item),
            EventKind::Status
            | EventKind::CureStatus
            | EventKind::EffectStart
            | EventKind::EffectEnd
            | EventKind::SideEffectStart
            | EventKind::SideEffectEnd
            | EventKind::FieldEffectStart
            | EventKind::FieldEffectEnd => Some(EffectKind::Condition),
            EventKind::Forme | EventKind::Mega => Some(EffectKind::Species),
            EventKind::Boost => Some(EffectKind::Stat),
            EventKind::Switch | EventKind::Faint => Some(EffectKind::None),
            EventKind::Damage | EventKind::Heal => None,
        };
        if expected_kind.is_some_and(|kind| kind != event.effect_kind) {
            return Err(EngineError::InvalidInput(
                "public event effect catalogue".into(),
            ));
        }
        if event.kind == EventKind::Mega {
            self.mega_used[usize::from(event.subject >= 6)] = true;
        }
        if event.kind == EventKind::Move {
            self.reveal_move(event.subject, event.effect)?;
        }
        if event.kind == EventKind::FieldEffectStart {
            let effect = self.field.entry(event.effect).or_default();
            effect.present = true;
            effect.source = Known::new(event.subject);
            if event.value > 0 {
                effect.duration = Known::new(event.value as i16);
            }
        } else if event.kind == EventKind::FieldEffectEnd {
            self.field.remove(&event.effect);
        }
        let side = usize::from(event.subject >= 6);
        if event.kind == EventKind::SideEffectStart {
            let effect = self.sides[side].entry(event.effect).or_default();
            effect.present = true;
            if event.value > 0 {
                effect.duration = Known::new(event.value as i16);
            }
        } else if event.kind == EventKind::SideEffectEnd {
            self.sides[side].remove(&event.effect);
        }
        let mon = &mut self.pokemon[event.subject as usize];
        if let Some(health) = event.health {
            mon.health = Known::new(health);
        }
        match event.kind {
            EventKind::Switch => {
                mon.selected = Known::new(true);
                mon.active_slot = Some(event.value as u8);
                mon.boosts = [0; 7];
            }
            EventKind::Faint => {
                mon.fainted = true;
                mon.status = Known::new(0);
                mon.boosts = [0; 7];
                mon.effects.clear();
                mon.health = Known::new(HealthDisplay {
                    numerator: 0,
                    denominator: 100,
                    boundary_color: 0,
                });
            }
            EventKind::Status => mon.status = Known::new(event.effect),
            EventKind::CureStatus => mon.status = Known::new(0),
            EventKind::Ability => mon.ability = Known::new(event.effect),
            EventKind::Item => mon.item = Known::new(event.effect),
            EventKind::EndItem => {
                mon.previous_item = Known::new(event.effect);
                mon.item = Known::new(0);
            }
            EventKind::Boost => {
                let boost = mon
                    .boosts
                    .get_mut(event.effect as usize)
                    .ok_or_else(|| EngineError::InvalidInput("boost stat".into()))?;
                *boost = (i32::from(*boost) + event.value).clamp(-6, 6) as i8;
            }
            EventKind::Forme | EventKind::Mega => mon.species = event.effect,
            EventKind::EffectStart => {
                let effect = mon.effects.entry(event.effect).or_default();
                effect.present = true;
                if let Some(source) = event.target {
                    effect.source = Known::new(source);
                }
            }
            EventKind::EffectEnd => {
                mon.effects.remove(&event.effect);
            }
            _ => (),
        }
        if event.kind == EventKind::Switch {
            let side_start = if event.subject < 6 { 0 } else { 6 };
            for i in side_start..side_start + 6 {
                if i != event.subject as usize
                    && self.pokemon[i].active_slot == Some(event.value as u8)
                {
                    self.pokemon[i].active_slot = None;
                    self.pokemon[i].boosts = [0; 7];
                    self.pokemon[i].effects.clear();
                }
            }
        }
        if self.events.len() == EVENT_CAPACITY {
            self.events.remove(0);
        }
        self.events.push(event);
        Ok(())
    }
}

/// Fixed token roles, with complete effect/repertoire storage attached separately.
/// No effect is discarded to meet the 96-token encoder padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenRole {
    Global,
    Field,
    Side(u8),
    Pokemon(u8),
    Move(u8, u8),
    Event(u8),
    Padding,
}

pub fn token_layout() -> [TokenRole; OBSERVATION_TOKENS] {
    std::array::from_fn(|i| match i {
        0 => TokenRole::Global,
        1 => TokenRole::Field,
        2..=3 => TokenRole::Side((i - 2) as u8),
        4..=15 => TokenRole::Pokemon((i - 4) as u8),
        16..=63 => TokenRole::Move(((i - 16) / 4) as u8, ((i - 16) % 4) as u8),
        64..=87 => TokenRole::Event((i - 64) as u8),
        _ => TokenRole::Padding,
    })
}
