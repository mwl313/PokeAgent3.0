//! Complete serializable state containers. World state is private to the engine;
//! player views are constructed from own state and persistent public knowledge.
use crate::{
    EngineError, Result,
    actions::{AtomicAction, Request},
    assets::{Dex, Id},
    knowledge::{Knowledge, Known, SemanticEvent},
    rng::BattleRng,
    stats,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SideId {
    P1,
    P2,
}
impl SideId {
    pub fn index(self) -> usize {
        if self == Self::P1 { 0 } else { 1 }
    }
    pub fn other(self) -> Self {
        if self == Self::P1 { Self::P2 } else { Self::P1 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamSet {
    pub species: Id,
    pub ability: Id,
    pub item: Id,
    pub nature: Id,
    pub moves: Vec<Id>,
    pub points: [u8; 6],
    pub ivs: [u8; 6],
    pub gender: String,
    pub level: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Team {
    pub id: String,
    pub members: [TeamSet; 6],
}

impl Team {
    pub fn validate_structure(&self, dex: &Dex) -> Result<()> {
        let mut species_clause = Vec::with_capacity(6);
        let mut item_clause = Vec::with_capacity(6);
        for m in &self.members {
            let valid = m.species > 0
                && usize::from(m.species) < dex.species.len()
                && m.ability > 0
                && usize::from(m.ability) < dex.names["abilities"].len()
                && usize::from(m.item) < dex.names["items"].len()
                && m.nature > 0
                && usize::from(m.nature) < dex.natures.len()
                && (1..=4).contains(&m.moves.len())
                && m.moves
                    .iter()
                    .all(|x| *x > 0 && usize::from(*x) < dex.moves.len())
                && m.points.iter().all(|x| *x <= 32)
                && m.points.iter().map(|x| u16::from(*x)).sum::<u16>() <= 66
                && m.ivs.iter().all(|x| *x <= 31)
                && m.level == 50
                && ["", "N", "M", "F"].contains(&m.gender.as_str());
            if !valid {
                return Err(EngineError::InvalidInput(
                    "invalid compiled team set".into(),
                ));
            }
            if !dex.legal_starting_species[m.species as usize]
                || !dex.legal_items[m.item as usize]
                || !dex.legal_abilities_by_species[m.species as usize].contains(&m.ability)
                || !m
                    .moves
                    .iter()
                    .all(|mv| dex.legal_moves_by_species[m.species as usize].contains(mv))
            {
                return Err(EngineError::InvalidInput(
                    "set outside pinned M-C eligibility/learnset".into(),
                ));
            }
            let base = dex.species[m.species as usize].base_species;
            if species_clause.contains(&base) || (m.item != 0 && item_clause.contains(&m.item)) {
                return Err(EngineError::InvalidInput("species/item clause".into()));
            }
            if m.moves
                .iter()
                .enumerate()
                .any(|(i, mv)| m.moves[..i].contains(mv))
            {
                return Err(EngineError::InvalidInput("duplicate move".into()));
            }
            species_clause.push(base);
            item_clause.push(m.item);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectState {
    pub id: Id,
    pub duration: Option<u16>,
    pub source: Option<(SideId, u8)>,
    pub effect_order: u32,
    pub effect_order_assigned: bool,
    /// Variable-length numeric state: dynamic mechanics cannot silently overflow
    /// an arbitrary fixed volatile capacity.
    pub values: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoveState {
    pub id: Id,
    pub pp: u8,
    pub max_pp: u8,
    pub disabled: bool,
    /// `disabled` came from Imprison's `'hidden'` disable, which the served
    /// request only applies to the side's last active Pokemon.
    #[serde(default)]
    pub hidden: bool,
    pub used: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PokemonState {
    pub active_turns: u16,
    /// Reference `activeMoveActions`: reset on entry and incremented for every
    /// move action the Pokémon actually runs while active.
    pub active_move_actions: u16,
    pub cached_speed: i32,
    pub base_species: Id,
    pub species: Id,
    pub types: Vec<Id>,
    pub base_ability: Id,
    pub ability: Id,
    pub ability_ending: bool,
    /// `abilities:protean|libero.onPrepareHit` one-shot flag: the holder only
    /// changes type once per switch-in. Reset when it enters the field.
    #[serde(default)]
    pub protean_used: bool,
    pub ability_effect_order: Option<u32>,
    pub item_effect_order: Option<u32>,
    pub item: Id,
    pub previous_item: Id,
    pub nature: Id,
    pub level: u16,
    pub gender: u8,
    pub points: [u8; 6],
    pub ivs: [u8; 6],
    pub stats: [u16; 6],
    pub hp: u16,
    pub boosts: [i8; 7],
    pub status: Id,
    pub status_state: EffectState,
    pub moves: Vec<MoveState>,
    pub base_moves: Vec<MoveState>,
    pub volatiles: BTreeMap<Id, EffectState>,
    pub selected: bool,
    pub active_slot: Option<u8>,
    pub fainted: bool,
    pub transformed: bool,
    /// Reference `switchFlag`: the move that forces this active Pokémon to
    /// leave the field as soon as the current action finishes (`selfSwitch`
    /// pivots). Cleared when the switch is resolved.
    pub switch_flag: Option<Id>,
    /// Reference `switchFlag = true`: a pending switch that is not tied to a
    /// pivot move (Emergency Exit / Wimp Out). The request boundary treats it
    /// exactly like `switch_flag`.
    #[serde(default)]
    pub plain_switch_flag: bool,
    /// Reference `forceSwitchFlag`: `roar`/`whirlwind`/`dragontail`/
    /// `circlethrow` mark the target and the post-action phazing step drags a
    /// random reserve in.
    pub force_switch_flag: bool,
    /// Reference `lastMove`: the move this Pokémon most recently used while
    /// active (0 = none). Encore, Disable, Torment and Cursed Body read it.
    pub last_move: Id,
    /// Reference `timesAttacked`: landed hits this Pokémon has taken since it
    /// last entered the field. Rage Fist's `basePowerCallback` reads it.
    #[serde(default)]
    pub times_attacked: u16,
    /// Reference `moveThisTurnResult`: the outcome of the most recent move
    /// attempt this turn. Rolled into `move_last_turn_result` at turn start.
    #[serde(default)]
    pub move_this_turn_result: MoveResult,
    /// Reference `moveLastTurnResult` (see `MoveResult`).
    #[serde(default)]
    pub move_last_turn_result: MoveResult,
}

/// Reference `moveThisTurnResult` / `moveLastTurnResult`. `Undefined` is the
/// reference's `undefined` (no attempt yet), `Skipped` is `null` (a skipped
/// action: recharge, a charge turn, or an unresolved request), `Failed` is
/// `false` (a move that did not connect or was refused) and `Success` is
/// `true`. Stomping Tantrum doubles only on `Failed`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoveResult {
    #[default]
    Undefined,
    Skipped,
    Failed,
    Success,
}

impl PokemonState {
    /// Reference `getLockedMove()`: the charging move of a two-turn move
    /// (`twoturnmove.onLockMove`), the forced Recharge pseudo-move
    /// (`mustrecharge.onLockMove`) and the location recorded on the charge
    /// volatile. A locked slot refuses switches.
    pub fn locked_state(&self, dex: &Dex) -> (Option<Id>, bool, i8) {
        if self.volatiles.contains_key(&dex.effects.must_recharge) {
            return (None, true, 0);
        }
        match self.volatiles.get(&dex.effects.two_turn_move) {
            Some(state) => {
                let id = state.values.first().copied().unwrap_or(0);
                let loc = state.values.get(1).copied().unwrap_or(0);
                if id <= 0 || id > i64::from(u16::MAX) {
                    (None, false, 0)
                } else {
                    (Some(id as Id), false, loc as i8)
                }
            }
            None => (None, false, 0),
        }
    }

    fn initialize(set: &TeamSet, dex: &Dex, rng: &mut BattleRng) -> Self {
        let species = &dex.species[set.species as usize];
        let gender = match set.gender.as_str() {
            "N" => 0,
            "M" => 1,
            "F" => 2,
            _ => species
                .fixed_gender
                .unwrap_or_else(|| 1 + rng.below(2) as u8),
        };
        let stats = stats::champions_stats(
            species.base_stats,
            set.points,
            dex.natures[set.nature as usize],
            species.max_hp,
        );
        let moves: Vec<_> = set
            .moves
            .iter()
            .map(|id| {
                let m = &dex.moves[*id as usize];
                let pp = stats::champions_pp(m.pp, m.no_pp_boosts);
                MoveState {
                    id: *id,
                    pp,
                    max_pp: pp,
                    disabled: false,
                    hidden: false,
                    used: false,
                }
            })
            .collect();
        Self {
            cached_speed: i32::from(stats[5]),
            active_turns: 0,
            base_species: set.species,
            species: set.species,
            types: species.types.clone(),
            base_ability: set.ability,
            ability: set.ability,
            ability_ending: false,
            protean_used: false,
            ability_effect_order: None,
            item_effect_order: None,
            item: set.item,
            previous_item: 0,
            nature: set.nature,
            level: 50,
            gender,
            points: set.points,
            ivs: set.ivs,
            stats,
            hp: stats[0],
            boosts: [0; 7],
            status: 0,
            status_state: Default::default(),
            base_moves: moves.clone(),
            moves,
            volatiles: BTreeMap::new(),
            selected: false,
            active_slot: None,
            fainted: false,
            transformed: false,
            active_move_actions: 0,
            switch_flag: None,
            plain_switch_flag: false,
            force_switch_flag: false,
            last_move: 0,
            times_attacked: 0,
            move_this_turn_result: MoveResult::Undefined,
            move_last_turn_result: MoveResult::Undefined,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SideState {
    pub pokemon: [PokemonState; 6],
    pub selected_order: Option<[u8; 4]>,
    /// Reference `side.pokemon` array order for the four selected members:
    /// positions[0..2] are the active slots and positions[2..4] the reserves.
    /// A switch-in swaps the incoming reserve's slot with the active slot it
    /// takes, exactly like `switchIn` does in the reference.
    pub positions: [u8; 4],
    pub active: [Option<u8>; 2],
    pub conditions: BTreeMap<Id, EffectState>,
    pub slot_conditions: [BTreeMap<Id, EffectState>; 2],
    pub mega_used: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EndReason {
    LastPokemon,
    RuleTurnLimit,
    RuleTiebreak,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    pub terminated: bool,
    pub truncated: bool,
    pub operational_error: Option<String>,
    pub winner: Option<SideId>,
    pub reason: Option<EndReason>,
}

impl Outcome {
    pub fn reward(&self, side: SideId) -> Option<i8> {
        if !self.terminated || self.truncated || self.operational_error.is_some() {
            return None;
        }
        Some(match self.winner {
            None => 0,
            Some(winner) if winner == side => 1,
            _ => -1,
        })
    }
}

/// Current snapshot schema. Bump when the persisted world shape changes; the
/// restore path rejects every other value, and tests read this constant so a
/// bump cannot leave a stale hard-coded expectation behind.
pub const SNAPSHOT_SCHEMA: u32 = 10;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BattleState {
    schema: u32,
    pub(crate) next_effect_order: u32,
    asset_digest: String,
    pub(crate) sides: [SideState; 2],
    pub(crate) knowledge: [Knowledge; 2],
    pub(crate) requests: [Request; 2],
    pub(crate) pending: [Option<Vec<AtomicAction>>; 2],
    pub(crate) field: BTreeMap<Id, EffectState>,
    pub(crate) rng: BattleRng,
    pub(crate) role_map: [u8; 2],
    pub(crate) turn: u16,
    pub(crate) outcome: Outcome,
    pub(crate) queue: Vec<crate::effects::QueuedAction>,
    pub(crate) mid_turn: bool,
    pub(crate) faint_queue: Vec<FaintData>,
    pub(crate) trace: Option<NativeTrace>,
}

/// One queued faint with the reference `faintData` provenance: the fainted
/// Pokémon, the source that caused it (when any) and whether a move caused it.
/// `faintMessages` reads the same triple for its `AfterFaint` event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaintData {
    pub target: crate::effects::Entity,
    #[serde(default)]
    pub source: Option<crate::effects::Entity>,
    #[serde(default)]
    pub from_move: bool,
}

/// Opt-in development/evaluation trace. The starting snapshot contains complete
/// world/knowledge/RNG state, while event streams retain their player audiences.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeTrace {
    pub oracle_commit: String,
    pub format: String,
    pub initial_state: Vec<u8>,
    pub actions: Vec<TraceEntry>,
    pub events: [Vec<TraceEvent>; 2],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceEvent {
    pub turn: u16,
    pub event: SemanticEvent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceEntry {
    pub side: SideId,
    pub actions: Vec<AtomicAction>,
    pub rng_before: [u16; 4],
    pub turn: u16,
}

/// Deliberately excludes the opponent's true sets, selected reserves, RNG,
/// pending actions, role mapping, provenance and team identifiers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlayerView {
    pub side: SideId,
    pub turn: u16,
    pub own: OwnSideView,
    pub knowledge: Knowledge,
    pub request: Request,
    pub outcome: Outcome,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnPokemonView {
    pub base_species: Id,
    pub species: Id,
    pub types: Vec<Id>,
    pub ability: Id,
    pub item: Id,
    pub previous_item: Id,
    pub nature: Id,
    pub level: u16,
    pub gender: u8,
    pub points: [u8; 6],
    pub ivs: [u8; 6],
    pub stats: [u16; 6],
    pub hp: u16,
    pub boosts: [i8; 7],
    pub status: Id,
    pub moves: Vec<MoveState>,
    pub base_moves: Vec<MoveState>,
    pub selected: bool,
    pub active_slot: Option<u8>,
    pub fainted: bool,
    pub transformed: bool,
}

impl From<&PokemonState> for OwnPokemonView {
    fn from(m: &PokemonState) -> Self {
        Self {
            base_species: m.base_species,
            species: m.species,
            types: m.types.clone(),
            ability: m.ability,
            item: m.item,
            previous_item: m.previous_item,
            nature: m.nature,
            level: m.level,
            gender: m.gender,
            points: m.points,
            ivs: m.ivs,
            stats: m.stats,
            hp: m.hp,
            boosts: m.boosts,
            status: m.status,
            moves: m.moves.clone(),
            base_moves: m.base_moves.clone(),
            selected: m.selected,
            active_slot: m.active_slot,
            fainted: m.fainted,
            transformed: m.transformed,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnSideView {
    pub pokemon: [OwnPokemonView; 6],
    pub selected_order: Option<[u8; 4]>,
    pub active: [Option<u8>; 2],
    pub mega_used: bool,
}

impl BattleState {
    pub fn reset(dex: &Dex, teams: [&Team; 2], seed: [u16; 4], role_map: [u8; 2]) -> Result<Self> {
        if role_map != [0, 1] && role_map != [1, 0] {
            return Err(EngineError::InvalidInput(
                "role_map must map the two seats bijectively".into(),
            ));
        }
        for team in teams {
            team.validate_structure(dex)?;
        }
        let mut rng = BattleRng::new(seed);
        let sides = std::array::from_fn(|i| SideState {
            pokemon: std::array::from_fn(|j| {
                PokemonState::initialize(&teams[i].members[j], dex, &mut rng)
            }),
            selected_order: None,
            positions: [0, 1, 2, 3],
            active: [None; 2],
            conditions: BTreeMap::new(),
            slot_conditions: Default::default(),
            mega_used: false,
        });
        let mut state = Self {
            schema: SNAPSHOT_SCHEMA,
            next_effect_order: 0,
            asset_digest: dex.asset_digest.clone(),
            sides,
            knowledge: Default::default(),
            requests: [Request::preview(), Request::preview()],
            pending: [None, None],
            field: BTreeMap::new(),
            rng,
            role_map,
            turn: 0,
            outcome: Outcome::default(),
            queue: vec![],
            mid_turn: false,
            faint_queue: vec![],
            trace: None,
        };
        for viewer in 0..2 {
            for owner in 0..2 {
                for j in 0..6 {
                    let entity = if owner == viewer { j } else { j + 6 };
                    let mon = &state.sides[owner].pokemon[j];
                    state.knowledge[viewer].preview(
                        entity as u8,
                        mon.species,
                        mon.types.clone(),
                        Known::new(mon.gender),
                    )?;
                }
            }
        }
        Ok(state)
    }

    pub fn observe(&self, side: SideId) -> PlayerView {
        let own = &self.sides[side.index()];
        let own = OwnSideView {
            pokemon: std::array::from_fn(|i| OwnPokemonView::from(&own.pokemon[i])),
            selected_order: own.selected_order,
            active: own.active,
            mega_used: own.mega_used,
        };
        let mut request = self.requests[side.index()].clone();
        if self.pending[side.index()].is_some() {
            request.kind = crate::actions::RequestKind::Wait;
        }
        PlayerView {
            side,
            turn: self.turn,
            own,
            knowledge: self.knowledge[side.index()].clone(),
            request,
            outcome: self.outcome.clone(),
        }
    }

    pub(crate) fn allocate_effect_order(&mut self) -> Result<u32> {
        let order = self.next_effect_order;
        self.next_effect_order = order
            .checked_add(1)
            .ok_or_else(|| EngineError::InvalidInput("effect creation order exhausted".into()))?;
        Ok(order)
    }

    pub fn snapshot(&self) -> Result<Vec<u8>> {
        let payload = serde_json::to_string(self)?;
        let envelope = serde_json::json!({"payload": payload, "sha256": format!("{:x}", Sha256::digest(payload.as_bytes()))});
        Ok(serde_json::to_vec(&envelope)?)
    }

    pub fn restore(dex: &Dex, bytes: &[u8]) -> Result<Self> {
        #[derive(Deserialize)]
        struct Envelope {
            payload: String,
            sha256: String,
        }
        let envelope: Envelope = serde_json::from_slice(bytes)?;
        if format!("{:x}", Sha256::digest(envelope.payload.as_bytes())) != envelope.sha256 {
            return Err(EngineError::AssetMismatch("snapshot integrity".into()));
        }
        let state: Self = serde_json::from_str(&envelope.payload)?;
        if state.schema != SNAPSHOT_SCHEMA || state.asset_digest != dex.asset_digest {
            return Err(EngineError::AssetMismatch("snapshot schema/dex".into()));
        }
        if state.role_map != [0, 1] && state.role_map != [1, 0] {
            return Err(EngineError::InvalidInput("snapshot role map".into()));
        }
        // The pinned rule ends on turn 1001; no playable boundary may exceed
        // that value, even if an externally re-signed payload has a valid hash.
        if state.turn > 1001
            || state.turn == 1001
                && !(state.outcome.terminated
                    && state.outcome.reason == Some(EndReason::RuleTurnLimit))
        {
            return Err(EngineError::InvalidInput("snapshot turn boundary".into()));
        }
        let valid_entity = |e: crate::effects::Entity| e.side < 2 && e.roster < 6;
        let valid_effect = |e: &EffectState| {
            // Condition ids name most volatiles; a two-turn charge marker is
            // keyed by its move id in the reference, so a move with a ported
            // charge recipe is also a legal volatile identity.
            let condition = usize::from(e.id) < dex.names["conditions"].len();
            let charge_marker = usize::from(e.id) < dex.moves.len()
                && dex.moves[e.id as usize].charge.is_some();
            (condition || charge_marker) && e.source.is_none_or(|(_, roster)| roster < 6)
        };
        let valid_moves = |moves: &[MoveState]| {
            (1..=4).contains(&moves.len())
                && moves.iter().all(|mv| {
                    mv.id > 0 && usize::from(mv.id) < dex.moves.len() && mv.pp <= mv.max_pp
                })
        };
        for (side_index, side) in state.sides.iter().enumerate() {
            if let Some(order) = side.selected_order {
                if order.iter().any(|r| *r >= 6)
                    || order
                        .iter()
                        .enumerate()
                        .any(|(i, r)| order[..i].contains(r))
                    || side
                        .pokemon
                        .iter()
                        .enumerate()
                        .any(|(i, m)| m.selected != order.contains(&(i as u8)))
                {
                    return Err(EngineError::InvalidInput("snapshot selected roster".into()));
                }
            } else if side.pokemon.iter().any(|m| m.selected) {
                return Err(EngineError::InvalidInput(
                    "snapshot selection without order".into(),
                ));
            }
            if side.active.iter().flatten().any(|r| *r >= 6)
                || side.active[0].is_some() && side.active[0] == side.active[1]
            {
                return Err(EngineError::InvalidInput("snapshot active roster".into()));
            }
            // The reference array order is a permutation of the four selected
            // members; its first two entries are the active slots.
            if side.positions.iter().any(|r| *r >= 6)
                || side
                    .positions
                    .iter()
                    .enumerate()
                    .any(|(i, r)| side.positions[..i].contains(r))
                || side.positions[..2] != [side.active[0].unwrap_or(255), side.active[1].unwrap_or(255)]
                    && side.active.iter().all(|slot| slot.is_some())
            {
                return Err(EngineError::InvalidInput("snapshot side order".into()));
            }
            for (roster, m) in side.pokemon.iter().enumerate() {
                let entity = crate::effects::Entity {
                    side: side_index as u8,
                    roster: roster as u8,
                };
                if m.species == 0
                    || usize::from(m.species) >= dex.species.len()
                    || m.base_species == 0
                    || usize::from(m.base_species) >= dex.species.len()
                    || usize::from(m.ability) >= dex.names["abilities"].len()
                    || usize::from(m.base_ability) >= dex.names["abilities"].len()
                    || usize::from(m.item) >= dex.names["items"].len()
                    || usize::from(m.previous_item) >= dex.names["items"].len()
                    || m.nature == 0
                    || usize::from(m.nature) >= dex.natures.len()
                    || m.types.is_empty()
                    || m.types.len() > 3
                    // Type id 0 is the `'???'` placeholder Double Shock maps
                    // Electric slots to, mirroring the reference's `'???'`.
                    || m.types
                        .iter()
                        .any(|t| usize::from(*t) >= dex.names["types"].len())
                    || m.stats[0] == 0
                    || m.level != 50
                    || m.gender > 2
                    || m.boosts.iter().any(|b| !(-6..=6).contains(b))
                    || m.hp > m.stats[0]
                    || m.fainted && m.hp != 0
                    || m.hp == 0 && !m.fainted && !state.faint_queue.iter().any(|f| f.target == entity)
                    || !valid_moves(&m.moves)
                    || !valid_moves(&m.base_moves)
                    || m.active_slot.is_some_and(|s| s > 1)
                    || usize::from(m.status) >= dex.names["conditions"].len()
                    || !valid_effect(&m.status_state)
                    || m.status != 0 && m.status != m.status_state.id
                    || [dex.effects.sleep, dex.effects.freeze, dex.effects.toxic]
                        .contains(&m.status)
                        && m.status_state.values.is_empty()
                    || m.volatiles
                        .iter()
                        .any(|(id, effect)| *id != effect.id || !valid_effect(effect))
                    || m.switch_flag.is_some_and(|id| {
                        id == 0
                            || usize::from(id) >= dex.moves.len()
                            || dex.moves[id as usize].self_switch
                                != crate::assets::SelfSwitch::Switch
                    })
                    || m.switch_flag.is_some() && (m.fainted || m.active_slot.is_none())
                    || m.plain_switch_flag && (m.fainted || m.active_slot.is_none())
                    || m.force_switch_flag && (m.fainted || m.active_slot.is_none())
                    || m.last_move != 0
                        && (usize::from(m.last_move) >= dex.moves.len()
                            || (m.last_move != dex.effects.struggle
                                && !m
                                    .base_moves
                                    .iter()
                                    .chain(m.moves.iter())
                                    .any(|mv| mv.id == m.last_move)))
                {
                    return Err(EngineError::InvalidInput("invalid snapshot Pokémon".into()));
                }
                if m.active_turns > state.turn {
                    return Err(EngineError::InvalidInput(
                        "snapshot active turn count".into(),
                    ));
                }
                let status_ids = [
                    dex.effects.burn,
                    dex.effects.paralysis,
                    dex.effects.sleep,
                    dex.effects.freeze,
                    dex.effects.poison,
                    dex.effects.toxic,
                ];
                let status_effect = &m.status_state;
                if m.status != 0 && !status_ids.contains(&m.status)
                    || status_effect.id != 0 && !status_ids.contains(&status_effect.id)
                {
                    return Err(EngineError::Unsupported("snapshot major status".into()));
                }
                let valid_status_values = if status_effect.id == dex.effects.sleep
                    || status_effect.id == dex.effects.freeze
                {
                    status_effect.values.len() == 1 && (1..=3).contains(&status_effect.values[0])
                } else if status_effect.id == dex.effects.toxic {
                    status_effect.values.len() == 1 && (0..=15).contains(&status_effect.values[0])
                } else {
                    status_effect.values.is_empty()
                };
                if !valid_status_values
                    || status_effect.duration.is_some()
                    || status_effect.id != 0 && status_effect.source.is_none()
                    || status_effect.id == 0 && status_effect.source.is_some()
                    || m.status == 0 && status_effect.id != 0 && !m.fainted
                {
                    return Err(EngineError::InvalidInput(
                        "snapshot major status payload".into(),
                    ));
                }
                for (&id, effect) in &m.volatiles {
                    let valid = if id == dex.effects.protect
                        || id == dex.effects.flinch
                        || id == dex.effects.endure
                        || id == dex.effects.spiky_shield
                        || id == dex.effects.baneful_bunker
                        || id == dex.effects.kings_shield
                    {
                        effect.duration == Some(1) && effect.values.is_empty()
                    } else if id == dex.effects.must_recharge {
                        effect
                            .duration
                            .is_some_and(|duration| (1..=2).contains(&duration))
                            && effect.values.is_empty()
                    } else if id == dex.effects.confusion {
                        // Reference `onStart` rolls `random(2, 6)` -> 2..5 and
                        // `onBeforeMove` decrements to zero, so any retained
                        // timer is in 1..=5.
                        effect.duration.is_none()
                            && effect.values.len() == 1
                            && (1..=5).contains(&effect.values[0])
                    } else if id == dex.effects.stall {
                        effect
                            .duration
                            .is_some_and(|duration| (1..=2).contains(&duration))
                            && effect.values.len() == 1
                            && [3, 9, 27, 81, 243, 729].contains(&effect.values[0])
                    } else if id == dex.effects.choice_lock {
                        effect.duration.is_none()
                            && effect.values.len() == 1
                            && effect.values[0] > 0
                            && effect.values[0] < dex.moves.len() as i64
                            && m.moves
                                .iter()
                                .any(|mv| i64::from(mv.id) == effect.values[0])
                    } else if id == dex.effects.flash_fire {
                        effect.duration.is_none()
                            && effect.values.is_empty()
                            && effect.source.is_some()
                    } else if id == dex.effects.throat_chop {
                        // Two-turn sound lock: duration ticks at residual order
                        // 22, no numeric payload, always attacker-sourced.
                        effect
                            .duration
                            .is_some_and(|duration| (1..=2).contains(&duration))
                            && effect.values.is_empty()
                            && effect.source.is_some()
                    } else if id == dex.effects.encore {
                        // Locks the holder into a move still in its repertoire;
                        // base duration 3, +1 when no action was queued yet.
                        effect
                            .duration
                            .is_some_and(|duration| (1..=4).contains(&duration))
                            && effect.values.len() == 1
                            && effect.values[0] > 0
                            && effect.values[0] < dex.moves.len() as i64
                            && m.moves.iter().any(|mv| i64::from(mv.id) == effect.values[0])
                    } else if id == dex.effects.disable {
                        // Base duration 5, one decrement when the target had not
                        // yet acted (or the ability fired mid-move).
                        effect
                            .duration
                            .is_some_and(|duration| (1..=5).contains(&duration))
                            && effect.values.len() == 1
                            && effect.values[0] > 0
                            && effect.values[0] < dex.moves.len() as i64
                            && m.moves.iter().any(|mv| i64::from(mv.id) == effect.values[0])
                    } else if id == dex.effects.taunt {
                        effect
                            .duration
                            .is_some_and(|duration| (1..=4).contains(&duration))
                            && effect.values.is_empty()
                    } else if id == dex.effects.torment {
                        effect.duration.is_none() && effect.values.is_empty()
                    } else if id == dex.effects.imprison {
                        effect.duration.is_none()
                            && effect.values.is_empty()
                            && effect.source == Some((if side_index == 0 { SideId::P1 } else { SideId::P2 }, roster as u8))
                    } else if id == dex.effects.yawn {
                        // `moves:yawn.condition`: a two-turn countdown that
                        // ends in sleep. The residual ticks it at order 23, so
                        // any retained snapshot sees 1 or 2. The source is the
                        // Yawn user recorded at `onStart`.
                        effect
                            .duration
                            .is_some_and(|duration| (1..=2).contains(&duration))
                            && effect.values.is_empty()
                            && effect.source.is_some()
                    } else if id == dex.effects.roost {
                        // `moves:roost.condition`: a single casting-turn
                        // volatile with the caster as its source.
                        effect.duration == Some(1)
                            && effect.values.is_empty()
                            && effect.source.is_some()
                    } else if id == dex.effects.glaive_rush {
                        // `moves:glaiverush.condition`: a duration-less
                        // drawback volatile that `onBeforeMove` consumes.
                        effect.duration.is_none()
                            && effect.values.is_empty()
                            && effect.source.is_some()
                    } else if id == dex.effects.perish_song {
                        // `moves:perishsong.condition`: a four-tick countdown
                        // that ends in a faint.
                        effect
                            .duration
                            .is_some_and(|duration| (1..=4).contains(&duration))
                            && effect.values.is_empty()
                            && effect.source.is_some()
                    } else if id == dex.effects.leech_seed {
                        // `moves:leechseed.condition`: a duration-less drain
                        // volatile that records the seeding slot.
                        effect.duration.is_none()
                            && effect.values.is_empty()
                            && effect.source.is_some()
                    } else if id == dex.effects.substitute {
                        // `moves:substitute.condition`: the decoy's remaining
                        // HP starts at floor(maxHP/4) and only shrinks; a decoy
                        // that reaches zero is removed, so any retained
                        // snapshot sees 1..floor(maxHP/4). No duration, no
                        // source.
                        effect.duration.is_none()
                            && effect.values.len() == 1
                            && effect.values[0] > 0
                            && effect.values[0] <= i64::from(m.stats[0] / 4)
                    } else if id == dex.effects.partially_trapped {
                        // `partiallytrapped`: a 5-or-6 turn bind that stores its
                        // damage divisor and keeps the binding source.
                        effect
                            .duration
                            .is_some_and(|duration| (1..=6).contains(&duration))
                            && effect.values.as_slice() == [8]
                            && effect.source.is_some()
                    } else if id == dex.effects.two_turn_move {
                        // `twoturnmove.onStart` records the charging move and
                        // the player's chosen location; the duration is 2 and
                        // `onEnd` removes the marker volatile. The residual
                        // ticks it at the end of each turn, so any retained
                        // snapshot sees 1 or 2.
                        effect
                            .duration
                            .is_some_and(|duration| (1..=2).contains(&duration))
                            && effect.values.len() == 2
                            && effect.values[0] > 0
                            && effect.values[0] < dex.moves.len() as i64
                            && dex.moves[effect.values[0] as usize].charge.is_some()
                            && (-2..=2).contains(&effect.values[1])
                    } else if usize::from(id) < dex.moves.len()
                        && dex.moves[id as usize].charge.is_some()
                    {
                        // The charge marker: `attacker.addVolatile(move.id)`
                        // with the move's own condition duration and the
                        // recorded target location.
                        let spec = dex.moves[id as usize].charge.as_ref().unwrap();
                        let duration_ok = match spec.volatile_duration {
                            Some(declared) => effect
                                .duration
                                .is_some_and(|ticked| (1..=declared).contains(&ticked)),
                            None => effect.duration.is_none(),
                        };
                        duration_ok
                            && effect.values.len() == 1
                            && (-2..=2).contains(&effect.values[0])
                    } else {
                        return Err(EngineError::Unsupported(format!("snapshot volatile {id}")));
                    };
                    if !valid
                        || (m.active_slot.is_none() || m.fainted)
                            && state.outcome.operational_error.is_none()
                    {
                        return Err(EngineError::InvalidInput(
                            "snapshot volatile payload/lifecycle".into(),
                        ));
                    }
                }
                if let Some(effect) = m.volatiles.get(&dex.effects.flash_fire)
                    && (effect.duration.is_some()
                        || !effect.values.is_empty()
                        || effect.source.is_none()
                        || m.active_slot.is_none()
                        || m.fainted)
                {
                    return Err(EngineError::InvalidInput(
                        "snapshot Flash Fire state".into(),
                    ));
                }
                if m.ability_ending
                    && (!matches!(
                        dex.effects.abilities[m.ability as usize],
                        crate::effects::Ability::CloudNine | crate::effects::Ability::AirLock
                    ) || m.active_slot.is_some() && !m.fainted)
                {
                    return Err(EngineError::InvalidInput("snapshot ability ending".into()));
                }
                if m.active_slot.is_some_and(|slot| {
                    !m.selected || side.active[slot as usize] != Some(roster as u8)
                }) || side.active.iter().enumerate().any(|(slot, active)| {
                    *active == Some(roster as u8) && m.active_slot != Some(slot as u8)
                }) {
                    return Err(EngineError::InvalidInput(
                        "snapshot active-slot mapping".into(),
                    ));
                }
            }
            if side
                .conditions
                .iter()
                .chain(side.slot_conditions.iter().flat_map(|c| c.iter()))
                .any(|(id, effect)| *id != effect.id || !valid_effect(effect))
            {
                return Err(EngineError::InvalidInput("snapshot side conditions".into()));
            }
            if side
                .slot_conditions
                .iter()
                .any(|conditions| !conditions.is_empty())
            {
                return Err(EngineError::Unsupported("snapshot slot condition".into()));
            }
            for (&id, effect) in &side.conditions {
                if ![
                    dex.effects.tailwind,
                    dex.effects.reflect,
                    dex.effects.light_screen,
                    dex.effects.aurora_veil,
                    dex.effects.wide_guard,
                    dex.effects.quick_guard,
                    dex.effects.toxic_spikes,
                ]
                .contains(&id)
                {
                    return Err(EngineError::Unsupported(format!(
                        "snapshot side condition {id}"
                    )));
                }
                if id == dex.effects.toxic_spikes {
                    // Entry hazards store their layer count in `values` and
                    // mirror it in the duration slot for the fixture contract.
                    // Their source is the Pokémon that scattered them (the
                    // Toxic Debris holder), which sits on either side.
                    if !matches!(effect.values.as_slice(), [1] | [2])
                        || effect.duration != effect.values.first().map(|layers| *layers as u16)
                        || effect.source.is_none()
                    {
                        return Err(EngineError::InvalidInput("snapshot hazard layers".into()));
                    }
                    continue;
                }
                if !effect.values.is_empty() {
                    return Err(EngineError::InvalidInput(
                        "snapshot side condition payload".into(),
                    ));
                }
                let maximum = if id == dex.effects.tailwind {
                    Some(4)
                } else if id == dex.effects.reflect
                    || id == dex.effects.light_screen
                    || id == dex.effects.aurora_veil
                {
                    Some(8)
                } else {
                    Some(1)
                };
                if maximum.is_some_and(|maximum| {
                    effect.duration.is_none_or(|d| d == 0 || d > maximum)
                        || effect
                            .source
                            .is_none_or(|(owner, _)| owner.index() != side_index)
                }) {
                    return Err(EngineError::InvalidInput(
                        "snapshot side-effect duration/source".into(),
                    ));
                }
            }
        }
        if state
            .faint_queue
            .iter()
            .any(|f| !valid_entity(f.target) || f.source.is_some_and(|s| !valid_entity(s)))
            || state
                .field
                .iter()
                .any(|(id, effect)| *id != effect.id || !valid_effect(effect))
        {
            return Err(EngineError::InvalidInput(
                "snapshot field/faint queue".into(),
            ));
        }
        for (&id, effect) in &state.field {
            if ![
                dex.effects.rain,
                dex.effects.sun,
                dex.effects.sand,
                dex.effects.snow,
                dex.effects.trick_room,
                dex.effects.electric_terrain,
                dex.effects.grassy_terrain,
                dex.effects.misty_terrain,
                dex.effects.psychic_terrain,
            ]
            .contains(&id)
            {
                return Err(EngineError::Unsupported(format!(
                    "snapshot field condition {id}"
                )));
            }
            if !effect.values.is_empty() {
                return Err(EngineError::InvalidInput(
                    "snapshot field condition payload".into(),
                ));
            }
        }
        let weather_ids = [
            dex.effects.rain,
            dex.effects.sun,
            dex.effects.sand,
            dex.effects.snow,
        ];
        let mut weather_count = 0;
        for (&id, effect) in &state.field {
            if weather_ids.contains(&id) {
                weather_count += 1;
                if effect
                    .duration
                    .is_none_or(|duration| !(1..=8).contains(&duration))
                    || effect.source.is_none()
                {
                    return Err(EngineError::InvalidInput(
                        "snapshot weather duration/source".into(),
                    ));
                }
            }
        }
        if weather_count > 1 {
            return Err(EngineError::InvalidInput(
                "snapshot simultaneous weather".into(),
            ));
        }
        if state
            .field
            .get(&dex.effects.trick_room)
            .is_some_and(|effect| {
                effect
                    .duration
                    .is_none_or(|duration| !(1..=5).contains(&duration))
                    || effect.source.is_none()
            })
        {
            return Err(EngineError::InvalidInput(
                "snapshot Trick Room duration/source".into(),
            ));
        }
        let terrain_ids = [
            dex.effects.electric_terrain,
            dex.effects.grassy_terrain,
            dex.effects.misty_terrain,
            dex.effects.psychic_terrain,
        ];
        let mut terrain_count = 0;
        for (&id, effect) in &state.field {
            if terrain_ids.contains(&id) {
                terrain_count += 1;
                if effect
                    .duration
                    .is_none_or(|duration| !(1..=8).contains(&duration))
                    || effect.source.is_none()
                {
                    return Err(EngineError::InvalidInput(
                        "snapshot terrain duration/source".into(),
                    ));
                }
            }
        }
        if terrain_count > 1 {
            return Err(EngineError::InvalidInput(
                "snapshot simultaneous terrain".into(),
            ));
        }
        let mut assigned_orders = std::collections::BTreeSet::new();
        let mut validate_order = |order: Option<u32>| -> Result<()> {
            if let Some(order) = order
                && (order >= state.next_effect_order || !assigned_orders.insert(order))
            {
                return Err(EngineError::InvalidInput(
                    "snapshot effect creation order".into(),
                ));
            }
            Ok(())
        };
        for side in &state.sides {
            for mon in &side.pokemon {
                validate_order(mon.ability_effect_order)?;
                validate_order(mon.item_effect_order)?;
                if mon.item == 0 && mon.item_effect_order.is_some()
                    || mon.active_slot.is_some()
                        && !mon.fainted
                        && state.outcome.operational_error.is_none()
                        && (mon.ability != 0 && mon.ability_effect_order.is_none()
                            || mon.item != 0 && mon.item_effect_order.is_none())
                {
                    return Err(EngineError::InvalidInput(
                        "snapshot ability/item order assignment".into(),
                    ));
                }
                for effect in std::iter::once(&mon.status_state).chain(mon.volatiles.values()) {
                    if effect.effect_order_assigned != (effect.id != 0)
                        || !effect.effect_order_assigned && effect.effect_order != 0
                    {
                        return Err(EngineError::InvalidInput(
                            "snapshot Pokemon effect order assignment".into(),
                        ));
                    }
                    validate_order(effect.effect_order_assigned.then_some(effect.effect_order))?;
                }
            }
            for effect in side.conditions.values().chain(
                side.slot_conditions
                    .iter()
                    .flat_map(|conditions| conditions.values()),
            ) {
                if !effect.effect_order_assigned {
                    return Err(EngineError::InvalidInput(
                        "snapshot side effect order assignment".into(),
                    ));
                }
                validate_order(Some(effect.effect_order))?;
            }
        }
        if state
            .field
            .values()
            .any(|effect| effect.effect_order_assigned || effect.effect_order != 0)
        {
            return Err(EngineError::InvalidInput(
                "snapshot field effect order assignment".into(),
            ));
        }
        use crate::effects::QueuedKind;
        for action in &state.queue {
            if action.actor.is_some_and(|e| !valid_entity(e))
                || !matches!(action.kind, QueuedKind::BeforeTurn | QueuedKind::Residual)
                    && action.actor.is_none()
                || action.kind == QueuedKind::Switch && action.destination >= 6
                || action.kind == QueuedKind::Move
                    && (action.move_id == 0
                        || usize::from(action.move_id) >= dex.moves.len()
                        || !(action.move_slot < 4 || action.move_slot == crate::actions::NO_SLOT))
                || !(-2..=2).contains(&action.target_location)
            {
                return Err(EngineError::InvalidInput("snapshot queued action".into()));
            }
        }
        use crate::actions::{MoveChoice, RequestKind, SlotRequest};
        let closed = state.outcome.terminated || state.outcome.operational_error.is_some();
        if state.outcome.truncated
            || state.outcome.terminated != state.outcome.reason.is_some()
            || !state.outcome.terminated && state.outcome.winner.is_some()
            || state.outcome.reason == Some(EndReason::RuleTurnLimit)
                && (state.turn != 1001 || state.outcome.winner.is_some())
            || state.outcome.reason == Some(EndReason::RuleTiebreak)
            || closed
                && (state.pending.iter().any(Option::is_some)
                    || state
                        .requests
                        .iter()
                        .any(|r| r.kind != RequestKind::Finished))
            || !closed
                && state
                    .requests
                    .iter()
                    .any(|r| r.kind == RequestKind::Finished)
        {
            return Err(EngineError::InvalidInput(
                "snapshot outcome/request phase".into(),
            ));
        }
        if state.outcome.terminated
            && (state.turn == 0 || state.sides.iter().any(|s| s.selected_order.is_none()))
        {
            return Err(EngineError::InvalidInput(
                "snapshot terminal roster phase".into(),
            ));
        }
        if state.outcome.reason == Some(EndReason::LastPokemon) {
            let alive: [bool; 2] = std::array::from_fn(|i| {
                state.sides[i]
                    .pokemon
                    .iter()
                    .any(|p| p.selected && !p.fainted)
            });
            let expected = match alive {
                [true, false] => Some(SideId::P1),
                [false, true] => Some(SideId::P2),
                _ => None,
            };
            if alive == [true, true]
                || state.outcome.winner.is_none()
                || expected.is_some() && state.outcome.winner != expected
            {
                return Err(EngineError::InvalidInput(
                    "snapshot last Pokémon outcome".into(),
                ));
            }
        }
        if !closed
            && state
                .pending
                .iter()
                .enumerate()
                .all(|(i, p)| p.is_some() || state.requests[i].kind == RequestKind::Wait)
        {
            return Err(EngineError::InvalidInput(
                "snapshot already committed choices".into(),
            ));
        }
        if !closed {
            let preview = state
                .requests
                .iter()
                .any(|r| r.kind == RequestKind::Preview);
            let normal = state.requests.iter().any(|r| r.kind == RequestKind::Normal);
            if preview {
                if state.turn != 0
                    || state.mid_turn
                    || !state.queue.is_empty()
                    || !state.faint_queue.is_empty()
                    || state.requests.iter().any(|r| *r != Request::preview())
                    || state.sides.iter().any(|s| {
                        s.selected_order.is_some() || s.active != [None, None] || s.mega_used
                    })
                {
                    return Err(EngineError::InvalidInput("snapshot preview phase".into()));
                }
            } else {
                if state.turn == 0
                    || state.sides.iter().any(|s| s.selected_order.is_none())
                    || normal
                        && (state.mid_turn
                            || !state.queue.is_empty()
                            || state.requests.iter().any(|r| r.kind != RequestKind::Normal))
                    || !normal
                        && !state
                            .requests
                            .iter()
                            .any(|r| r.kind == RequestKind::Replacement)
                    || state
                        .pending
                        .iter()
                        .enumerate()
                        .all(|(i, p)| p.is_some() || state.requests[i].kind == RequestKind::Wait)
                {
                    return Err(EngineError::InvalidInput("snapshot playable phase".into()));
                }
                for (side_index, side) in state.sides.iter().enumerate() {
                    let request = &state.requests[side_index];
                    let bench: Vec<u8> = side
                        .selected_order
                        .unwrap()
                        .into_iter()
                        .filter(|r| {
                            let p = &side.pokemon[*r as usize];
                            !p.fainted && p.active_slot.is_none()
                        })
                        .collect();
                    let slots = std::array::from_fn(|slot| {
                        if normal {
                            let Some(roster) = side.active[slot] else {
                                return SlotRequest::default();
                            };
                            let p = &side.pokemon[roster as usize];
                            let (locked_move, locked_recharge, locked_target_location) =
                                p.locked_state(dex);
                            let locked = locked_move.is_some() || locked_recharge;
                            let last_active = (slot + 1..2).all(|later| {
                                side.active[later]
                                    .is_none_or(|r| side.pokemon[r as usize].fainted)
                            });
                            let moves = if locked_recharge {
                                Vec::new()
                            } else if let Some(id) = locked_move {
                                p.moves
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, m)| m.id == id)
                                    .map(|(slot, m)| MoveChoice {
                                        id: m.id,
                                        slot: slot as u8,
                                        target: dex.moves[m.id as usize].target,
                                        disabled: false,
                                        hidden: false,
                                        pp: m.pp,
                                    })
                                    .collect()
                            } else {
                                p.moves
                                    .iter()
                                    .enumerate()
                                    .map(|(slot, m)| MoveChoice {
                                        id: m.id,
                                        slot: slot as u8,
                                        target: dex.moves[m.id as usize].target,
                                        disabled: m.disabled,
                                        hidden: m.hidden,
                                        pp: m.pp,
                                    })
                                    .collect()
                            };
                            let (trap_state, trap_maybe) = state.trap_flags(
                                dex,
                                crate::effects::Entity {
                                    side: side_index as u8,
                                    roster,
                                },
                            );
                            let can_switch_in = !bench.is_empty();
                            let (trapped, maybe_trapped) = if locked {
                                (true, false)
                            } else if last_active {
                                (
                                    can_switch_in && trap_state == Some(false),
                                    can_switch_in && trap_state != Some(false) && trap_maybe,
                                )
                            } else {
                                (can_switch_in && trap_state.is_some(), false)
                            };
                            SlotRequest {
                                present: !p.fainted,
                                can_mega: !locked
                                    && !p.fainted
                                    && !side.mega_used
                                    && dex.effects.mega_stones[p.item as usize]
                                        .iter()
                                        .any(|(base, _)| *base == p.base_species),
                                moves,
                                trapped,
                                maybe_trapped,
                                locked_move,
                                locked_recharge,
                                locked_target_location,
                                last_active,
                                ..Default::default()
                            }
                        } else {
                            SlotRequest {
                                requires_replacement: side.active[slot]
                                    .is_some_and(|r| side.pokemon[r as usize].fainted),
                                ..Default::default()
                            }
                        }
                    });
                    let kind = if normal {
                        RequestKind::Normal
                    } else if !bench.is_empty() && slots.iter().any(|s| s.requires_replacement) {
                        RequestKind::Replacement
                    } else {
                        RequestKind::Wait
                    };
                    let expected = Request {
                        kind,
                        slots,
                        bench,
                        preview_roster: vec![],
                    };
                    if *request != expected
                        || request.kind == RequestKind::Wait && state.pending[side_index].is_some()
                    {
                        return Err(EngineError::InvalidInput(
                            "snapshot request/world relationship".into(),
                        ));
                    }
                }
            }
        }
        for (i, request) in state.requests.iter().enumerate() {
            request.validate()?;
            if request
                .slots
                .iter()
                .flat_map(|slot| &slot.moves)
                .any(|mv| usize::from(mv.id) >= dex.moves.len())
            {
                return Err(EngineError::InvalidInput("snapshot request move".into()));
            }
            if let Some(actions) = &state.pending[i] {
                request.validate_joint(actions)?;
            }
        }
        Ok(state)
    }

    pub fn rng_seed(&self) -> [u16; 4] {
        self.rng.seed()
    }

    /// Development probe: total RNG draws consumed since reset.
    pub fn rng_draws(&self) -> u64 {
        self.rng.draws
    }
}
