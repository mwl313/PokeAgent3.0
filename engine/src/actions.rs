//! Candidate construction accepts only the player's request, never opponent
//! world state. Hidden trapping belongs to the rule-defined re-request protocol.
use crate::{
    EngineError, Result,
    assets::{Id, Target},
};
use serde::{Deserialize, Serialize};
use smallvec::{SmallVec, smallvec};

pub const CANDIDATE_PADDING: usize = 64;
pub const NO_SLOT: u8 = u8::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestKind {
    Preview,
    Normal,
    Replacement,
    Wait,
    Finished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionKind {
    Pick,
    Move,
    Switch,
    Pass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Resource {
    None,
    Mega,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AtomicAction {
    pub kind: ActionKind,
    pub own_slot: u8,
    pub move_slot: u8,
    pub target_location: i8,
    pub switch_destination: u8,
    pub resource: Resource,
}

impl AtomicAction {
    pub fn select(kind: ActionKind, own_slot: u8, destination: u8) -> Self {
        Self {
            kind,
            own_slot,
            move_slot: NO_SLOT,
            target_location: 0,
            switch_destination: destination,
            resource: Resource::None,
        }
    }

    pub fn pass(own_slot: u8) -> Self {
        Self::select(ActionKind::Pass, own_slot, NO_SLOT)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoveChoice {
    pub id: Id,
    pub slot: u8,
    pub target: Target,
    pub disabled: bool,
    /// The world flag came from Imprison's `'hidden'` disable. The served
    /// request turns it into `!restrictData` (`Pokemon#getMoves`), so only the
    /// side's last active slot actually loses the move.
    #[serde(default)]
    pub hidden: bool,
    pub pp: u8,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotRequest {
    pub present: bool,
    pub requires_replacement: bool,
    pub moves: Vec<MoveChoice>,
    /// Only the trapping information the actual player request exposes.
    pub trapped: bool,
    pub maybe_trapped: bool,
    pub can_mega: bool,
    /// Reference `getLockedMove()` from the `twoturnmove` condition: this
    /// slot's only legal move, used with `locked_target_location` (the
    /// location recorded by `twoturnmove.onStart`). The reference request
    /// lists exactly one entry for it and refuses switches.
    pub locked_move: Option<Id>,
    /// `mustrecharge.onLockMove: 'recharge'`: the only legal action is the
    /// no-op Recharge pseudo-move, which the BeforeMove gate consumes.
    pub locked_recharge: bool,
    pub locked_target_location: i8,
    /// Reference `getMoves(..., restrictData = isLastActive())`: true when this
    /// slot is the side's last non-fainted active Pokemon. Only that slot has
    /// Imprison's `'hidden'` disables served as enabled (`disabled =
    /// !restrictData`); the execution-time `onFoeBeforeMove` gate then refuses
    /// the move.
    #[serde(default)]
    pub last_active: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    pub kind: RequestKind,
    pub slots: [SlotRequest; 2],
    /// Eligible destinations in the player's stable roster indexing.
    pub bench: Vec<u8>,
    pub preview_roster: Vec<u8>,
}

impl Request {
    pub fn preview() -> Self {
        Self {
            kind: RequestKind::Preview,
            slots: Default::default(),
            bench: vec![],
            preview_roster: (0..6).collect(),
        }
    }

    pub fn branch_slots(&self) -> Vec<u8> {
        self.branch_slots_inline().into_vec()
    }

    fn branch_slots_inline(&self) -> SmallVec<[u8; 4]> {
        match self.kind {
            RequestKind::Preview => smallvec![0, 1, 2, 3],
            RequestKind::Normal => smallvec![0, 1],
            RequestKind::Replacement => (0..2)
                .filter(|s| self.slots[*s].requires_replacement)
                .map(|s| s as u8)
                .collect(),
            RequestKind::Wait | RequestKind::Finished => SmallVec::new(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        for values in [&self.bench, &self.preview_roster] {
            let mut mask = 0u8;
            for &x in values {
                if x >= 6 || mask & (1 << x) != 0 {
                    return Err(EngineError::InvalidInput("request roster index".into()));
                }
                mask |= 1 << x;
            }
        }
        if self.kind == RequestKind::Preview && self.preview_roster.len() != 6 {
            return Err(EngineError::InvalidInput(
                "six-member preview required".into(),
            ));
        }
        for slot in &self.slots {
            let mut mask = 0u8;
            if slot.moves.len() > 4 {
                return Err(EngineError::InvalidInput(
                    "more than four current moves".into(),
                ));
            }
            for m in &slot.moves {
                if m.id == 0 || m.slot >= 4 || mask & (1 << m.slot) != 0 {
                    return Err(EngineError::InvalidInput("invalid move slot".into()));
                }
                mask |= 1 << m.slot;
            }
        }
        Ok(())
    }

    fn raw_candidates(
        &self,
        prefix: &[AtomicAction],
        branches: &[u8],
    ) -> SmallVec<[AtomicAction; 64]> {
        if prefix.len() >= branches.len() {
            return SmallVec::new();
        }
        let slot_id = branches[prefix.len()];
        let used = |dest| {
            prefix.iter().any(|a| {
                matches!(a.kind, ActionKind::Pick | ActionKind::Switch)
                    && a.switch_destination == dest
            })
        };
        if self.kind == RequestKind::Preview {
            return self
                .preview_roster
                .iter()
                .filter(|x| !used(**x))
                .map(|x| AtomicAction::select(ActionKind::Pick, slot_id, *x))
                .collect();
        }
        let slot = &self.slots[slot_id as usize];
        let available: SmallVec<[u8; 6]> =
            self.bench.iter().copied().filter(|x| !used(*x)).collect();
        if self.kind == RequestKind::Replacement {
            let mut out: SmallVec<[AtomicAction; 64]> = available
                .iter()
                .map(|x| AtomicAction::select(ActionKind::Switch, slot_id, *x))
                .collect();
            // With two empty slots and one reserve, either slot can receive it.
            if branches.len() - prefix.len() > available.len() {
                out.push(AtomicAction::pass(slot_id));
            }
            return out;
        }
        if !slot.present {
            return smallvec![AtomicAction::pass(slot_id)];
        }
        // Reference `chooseMove`: a locked Pokémon ignores the submitted move
        // entirely. The action is the locked move at the location recorded by
        // the charge volatile, and switching is refused (`trapped`).
        if slot.locked_recharge {
            return smallvec![AtomicAction {
                kind: ActionKind::Move,
                own_slot: slot_id,
                move_slot: NO_SLOT,
                target_location: 0,
                switch_destination: NO_SLOT,
                resource: Resource::None,
            }];
        }
        if let Some(locked) = slot.locked_move
            && let Some(choice) = slot.moves.iter().find(|m| m.id == locked)
        {
            return smallvec![AtomicAction {
                kind: ActionKind::Move,
                own_slot: slot_id,
                move_slot: choice.slot,
                target_location: slot.locked_target_location,
                switch_destination: NO_SLOT,
                resource: Resource::None,
            }];
        }
        let mut out: SmallVec<[AtomicAction; 64]> = SmallVec::new();
        let mega = slot.can_mega && !prefix.iter().any(|a| a.resource == Resource::Mega);
        let usable: SmallVec<[&MoveChoice; 4]> = slot
            .moves
            .iter()
            .filter(|m| {
                (!m.disabled || m.hidden && slot.last_active) && m.pp > 0
            })
            .collect();
        for m in &usable {
            for target in [-2, -1, 0, 1, 2] {
                if !m.target.valid_location(slot_id, target) {
                    continue;
                }
                for resource in [Resource::None, Resource::Mega] {
                    if resource == Resource::Mega && !mega {
                        continue;
                    }
                    out.push(AtomicAction {
                        kind: ActionKind::Move,
                        own_slot: slot_id,
                        move_slot: m.slot,
                        target_location: target,
                        switch_destination: NO_SLOT,
                        resource,
                    });
                }
            }
        }
        if usable.is_empty() {
            for resource in [Resource::None, Resource::Mega] {
                if resource == Resource::Mega && !mega {
                    continue;
                }
                // NO_SLOT is Struggle, whose random target is chosen during resolution.
                out.push(AtomicAction {
                    kind: ActionKind::Move,
                    own_slot: slot_id,
                    move_slot: NO_SLOT,
                    target_location: 0,
                    switch_destination: NO_SLOT,
                    resource,
                });
            }
        }
        if !slot.trapped {
            out.extend(
                available
                    .iter()
                    .map(|x| AtomicAction::select(ActionKind::Switch, slot_id, *x)),
            );
        }
        out
    }

    fn has_completion(&self, prefix: &mut SmallVec<[AtomicAction; 4]>, branches: &[u8]) -> bool {
        if prefix.len() == branches.len() {
            return true;
        }
        for candidate in self.raw_candidates(prefix, branches) {
            prefix.push(candidate);
            let result = self.has_completion(prefix, branches);
            prefix.pop();
            if result {
                return true;
            }
        }
        false
    }

    pub fn candidates(&self, prefix: &[AtomicAction]) -> Result<Vec<AtomicAction>> {
        let mut out = Vec::new();
        self.candidates_into(prefix, &mut out)?;
        Ok(out)
    }

    /// Reuse a caller-owned candidate buffer; ordering and prefix feasibility
    /// are identical to the owned convenience API, with no truncation.
    pub fn candidates_into(
        &self,
        prefix: &[AtomicAction],
        out: &mut Vec<AtomicAction>,
    ) -> Result<()> {
        out.clear();
        self.validate()?;
        let branches = self.branch_slots_inline();
        if prefix.len() > branches.len() {
            return Err(EngineError::InvalidInput("overlong action prefix".into()));
        }
        for i in 0..prefix.len() {
            if !self
                .raw_candidates(&prefix[..i], &branches)
                .contains(&prefix[i])
            {
                return Err(EngineError::InvalidInput("invalid action prefix".into()));
            }
        }
        let mut working = SmallVec::<[AtomicAction; 4]>::from_slice(prefix);
        out.extend(
            self.raw_candidates(prefix, &branches)
                .into_iter()
                .filter(|candidate| {
                    working.push(*candidate);
                    let feasible = self.has_completion(&mut working, &branches);
                    working.pop();
                    feasible
                }),
        );
        if out.len() > CANDIDATE_PADDING {
            return Err(EngineError::Unsupported(format!(
                "candidate capacity {} exceeds {CANDIDATE_PADDING}; no truncation allowed",
                out.len()
            )));
        }
        Ok(())
    }

    pub fn validate_joint(&self, joint: &[AtomicAction]) -> Result<()> {
        self.validate()?;
        let branches = self.branch_slots_inline();
        if joint.len() != branches.len() {
            return Err(EngineError::InvalidInput("incomplete joint action".into()));
        }
        // A fully valid sequence is its own completion witness. Avoid building
        // every alternative feasible prefix just to validate the chosen joint.
        for i in 0..joint.len() {
            if !self
                .raw_candidates(&joint[..i], &branches)
                .contains(&joint[i])
            {
                return Err(EngineError::InvalidInput("illegal joint action".into()));
            }
        }
        Ok(())
    }
}
