//! Per-actor local environment group. No process, socket or reference engine is
//! used by reset/step/observation/snapshot operations.
use crate::{
    EngineError, Result,
    actions::AtomicAction,
    assets::Dex,
    battle::StepResult,
    observation::{Encoder, ObservationBatchBuffers},
    state::{BattleState, PlayerView, SideId, Team},
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Handle {
    pub slot: u32,
    pub generation: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct ResetSpec {
    pub team_a: usize,
    pub team_b: usize,
    pub seed: [u16; 4],
    pub role_map: [u8; 2],
}

#[derive(Debug, Clone)]
pub struct SideChoice {
    pub side: SideId,
    pub actions: Vec<AtomicAction>,
}

#[derive(Debug, Clone)]
pub struct StepSpec {
    pub handle: Handle,
    /// One or both currently requested sides; each side may appear once.
    pub choices: Vec<SideChoice>,
}

pub struct BattleBatch {
    pub dex: Arc<Dex>,
    pub teams: Arc<Vec<Team>>,
    workers: rayon::ThreadPool,
    states: Vec<BattleState>,
    generation: u32,
    work_slots: Vec<Option<usize>>,
    step_results: Vec<Option<StepResult>>,
}

impl BattleBatch {
    pub fn new(dex: Arc<Dex>, teams: Arc<Vec<Team>>, workers: usize) -> Result<Self> {
        if workers == 0 || workers > 80 {
            return Err(EngineError::InvalidInput(
                "worker count must be 1..80".into(),
            ));
        }
        let workers = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .map_err(|e| EngineError::InvalidInput(e.to_string()))?;
        Ok(Self {
            dex,
            teams,
            workers,
            states: vec![],
            generation: 0,
            work_slots: vec![],
            step_results: vec![],
        })
    }

    pub fn reset_batch(&mut self, specs: &[ResetSpec]) -> Result<Vec<Handle>> {
        if specs.is_empty() || specs.len() > u32::MAX as usize {
            return Err(EngineError::InvalidInput("batch length".into()));
        }
        if specs
            .iter()
            .any(|s| s.team_a >= self.teams.len() || s.team_b >= self.teams.len())
        {
            return Err(EngineError::InvalidInput("team index".into()));
        }
        let generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| EngineError::InvalidInput("handle generation exhausted".into()))?;
        // All work must succeed before the old group is replaced.
        let states: Result<Vec<_>> = self.workers.install(|| {
            specs
                .par_iter()
                .map(|s| {
                    BattleState::reset(
                        &self.dex,
                        [&self.teams[s.team_a], &self.teams[s.team_b]],
                        s.seed,
                        s.role_map,
                    )
                })
                .collect()
        });
        self.states = states?;
        self.generation = generation;
        Ok((0..specs.len())
            .map(|i| Handle {
                slot: i as u32,
                generation,
            })
            .collect())
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Read-only access to one side's current request for action-mask and
    /// candidate generation. Unknown/stale handles are rejected.
    pub fn request(&self, handle: Handle, side: SideId) -> Result<&crate::actions::Request> {
        Ok(&self.state(handle)?.requests[side.index()])
    }

    fn state(&self, handle: Handle) -> Result<&BattleState> {
        if handle.generation != self.generation {
            return Err(EngineError::InvalidInput("stale environment handle".into()));
        }
        self.states
            .get(handle.slot as usize)
            .ok_or_else(|| EngineError::InvalidInput("environment handle".into()))
    }

    pub fn observe_batch(&self, requests: &[(Handle, SideId)]) -> Result<Vec<PlayerView>> {
        self.workers.install(|| {
            requests
                .par_iter()
                .map(|(handle, side)| self.state(*handle).map(|s| s.observe(*side)))
                .collect()
        })
    }

    /// Encode many player views in one native call. The encoder is constructed
    /// once against this group's shared Dex; output rows retain caller capacity
    /// and preserve request order, including repeated handles for both players.
    /// Invalid handles or a different Dex leave the existing output untouched.
    pub fn observe_encoded_batch_into(
        &self,
        requests: &[(Handle, SideId)],
        encoder: &Encoder<'_>,
        out: &mut ObservationBatchBuffers,
    ) -> Result<()> {
        if !encoder.matches_dex(&self.dex) {
            return Err(EngineError::InvalidInput(
                "observation encoder belongs to another Dex".into(),
            ));
        }
        for &(handle, _) in requests {
            self.state(handle)?;
        }
        out.prepare(requests.len());
        self.workers.install(|| {
            requests
                .par_iter()
                .zip(out.active_mut().par_iter_mut())
                .try_for_each(|(&(handle, side), buffer)| {
                    let view = self.state(handle)?.observe(side);
                    encoder.encode_into(&view, buffer)
                })
        })
    }

    /// Validate the entire submission before mutating any environment. Execution
    /// is native and parallel by environment, with no cloning or serialization in
    /// this path. Results preserve input order even for sparse/reordered handles.
    pub fn step_batch(&mut self, specs: &[StepSpec]) -> Result<Vec<StepResult>> {
        let mut results = Vec::new();
        self.step_batch_into(specs, &mut results)?;
        Ok(results)
    }

    /// Reuse scheduling and caller-owned result buffers across submissions.
    pub fn step_batch_into(&mut self, specs: &[StepSpec], out: &mut Vec<StepResult>) -> Result<()> {
        self.work_slots.resize(self.states.len(), None);
        self.work_slots.fill(None);
        for (index, spec) in specs.iter().enumerate() {
            let state = self.state(spec.handle)?;
            let slot = spec.handle.slot as usize;
            if self.work_slots[slot].is_some() || spec.choices.is_empty() || spec.choices.len() > 2
            {
                return Err(EngineError::InvalidInput(
                    "duplicate environment or invalid choice count".into(),
                ));
            }
            let mut sides = 0;
            for choice in &spec.choices {
                let bit = 1 << choice.side.index();
                if sides & bit != 0 {
                    return Err(EngineError::InvalidInput("duplicate player choice".into()));
                }
                sides |= bit;
                state.validate_choice(choice.side, &choice.actions)?;
            }
            self.work_slots[slot] = Some(index);
        }
        self.step_results.resize_with(self.states.len(), || None);
        self.step_results.fill(None);
        let dex = &self.dex;
        let work = &self.work_slots;
        let results = &mut self.step_results;
        self.workers.install(|| {
            self.states
                .par_iter_mut()
                .zip(results.par_iter_mut())
                .zip(work.par_iter())
                .try_for_each(|((state, result), work)| -> Result<()> {
                    if let Some(index) = *work {
                        let mut last = None;
                        for choice in &specs[index].choices {
                            last = Some(state.step(dex, choice.side, &choice.actions)?);
                        }
                        *result = last;
                    }
                    Ok(())
                })
        })?;
        out.clear();
        out.reserve(specs.len());
        for spec in specs {
            out.push(
                self.step_results[spec.handle.slot as usize]
                    .take()
                    .expect("preflight requires a player choice"),
            );
        }
        Ok(())
    }

    pub fn snapshot(&self, handle: Handle) -> Result<Vec<u8>> {
        self.state(handle)?.snapshot()
    }

    pub fn enable_trace(&mut self, handle: Handle) -> Result<()> {
        self.state(handle)?;
        self.states[handle.slot as usize].enable_trace()
    }

    pub fn export_trace(&self, handle: Handle) -> Result<Option<&crate::state::NativeTrace>> {
        Ok(self.state(handle)?.export_trace())
    }

    pub fn restore(&mut self, handle: Handle, bytes: &[u8]) -> Result<()> {
        self.state(handle)?;
        let restored = BattleState::restore(&self.dex, bytes)?;
        self.states[handle.slot as usize] = restored;
        Ok(())
    }
}
