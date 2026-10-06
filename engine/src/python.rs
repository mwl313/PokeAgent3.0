//! PyO3 batch binding. Python submits whole batches; each call advances many
//! environments with one native transition pass and one observation pass.
//! There is no per-effect, per-Pokémon or per-environment Python crossing, and
//! no reference (Pokémon Showdown) execution anywhere in this module.
//!
//! Handles are opaque `(slot, generation)` pairs supplied by `reset_batch`.
//! Observation payloads are one little-endian packed blob per view; the exact
//! layout is versioned by `observation::SCHEMA_VERSION` and mirrored by
//! `engine/python/pa3_observation.py`.
use crate::{
    EngineError, Result,
    actions::{ActionKind, AtomicAction, RequestKind, Resource},
    assets::Dex,
    batch::{BattleBatch, Handle, ResetSpec, SideChoice, StepSpec},
    knowledge::OBSERVATION_TOKENS,
    observation::{
        CATEGORY_COUNT, EncodedBaseMove, EncodedEffect, EncodedMoveEffect,
        Encoder, FLAG_COUNT, FLOAT_COUNT, ObservationBatchBuffers, ObservationBuffers,
        SCHEMA_VERSION, TokenFeatures,
    },
    state::{SideId, Team},
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::Path;
use std::sync::Arc;

const OBSERVATION_FIXED_BYTES: usize = 2
    + OBSERVATION_TOKENS
    + OBSERVATION_TOKENS * CATEGORY_COUNT * 2
    + OBSERVATION_TOKENS * CATEGORY_COUNT
    + OBSERVATION_TOKENS * FLOAT_COUNT * 4
    + OBSERVATION_TOKENS * FLOAT_COUNT
    + OBSERVATION_TOKENS * FLAG_COUNT
    + OBSERVATION_TOKENS * FLAG_COUNT
    + OBSERVATION_TOKENS * 5 * 2;


/// Packed action tuple: kind, slot, move slot, target location, destination, resource.
pub type ActionTuple = (u8, u8, u8, i8, u8, u8);
/// One candidate branch: action prefix of packed tuples.
pub type Branch = Vec<ActionTuple>;
/// One environment submission: league/step ids plus the branches for both sides.
pub type EnvSpec = (u32, u32, Vec<(u8, Branch)>);
/// One stepped environment result: terminated, truncated, request pending,
/// winner side, operational error, packed observation length, branch count.
pub type StepResult = (bool, bool, bool, Option<i8>, Option<String>, u8, u8);
/// One request move row: move id, remaining PP, slot, disabled, already used.
pub type MoveRow = (u16, u8, u8, bool, bool);

fn put_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_f32(out: &mut Vec<u8>, value: f32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn pack_tokens(out: &mut Vec<u8>, tokens: &[TokenFeatures; OBSERVATION_TOKENS]) {
    for token in tokens {
        for value in token.categories {
            put_u16(out, value);
        }
    }
    for token in tokens {
        for known in token.category_known {
            out.push(u8::from(known));
        }
    }
    for token in tokens {
        for value in token.floats {
            put_f32(out, value);
        }
    }
    for token in tokens {
        for known in token.float_known {
            out.push(u8::from(known));
        }
    }
    for token in tokens {
        for flag in token.flags {
            out.push(u8::from(flag));
        }
    }
    for token in tokens {
        for known in token.flag_known {
            out.push(u8::from(known));
        }
    }
}

fn pack_effect(out: &mut Vec<u8>, effect: &EncodedEffect) {
    put_u16(out, effect.id);
    out.push(u8::from(effect.present));
    put_f32(out, effect.duration);
    out.push(u8::from(effect.duration_known));
    put_f32(out, effect.stacks);
    out.push(u8::from(effect.stacks_known));
    out.push(effect.source);
    out.push(u8::from(effect.source_known));
}

fn pack_base_move(out: &mut Vec<u8>, base: &EncodedBaseMove) {
    put_u16(out, base.id);
    put_f32(out, base.pp);
    put_f32(out, base.max_pp);
    out.push(u8::from(base.disabled));
    out.push(u8::from(base.used));
}

fn pack_move_effect(out: &mut Vec<u8>, effect: &EncodedMoveEffect) {
    out.push(effect.kind);
    put_f32(out, effect.chance);
    put_u16(out, effect.status);
    put_u16(out, effect.volatile);
    for boost in effect.boosts {
        put_f32(out, boost);
    }
    put_f32(out, effect.heal);
    out.push(u8::from(effect.heal_known));
}

/// One view as a single little-endian blob: fixed token block, then the five
/// ragged sections in effect/repertoire/type/base-move/move-effect order, each
/// preceded by its per-token lengths.
pub fn pack_observation(view: &ObservationBuffers) -> Vec<u8> {
    let mut out = Vec::new();
    pack_observation_into(view, &mut out);
    out
}

/// Fixed-size token block only (schema version, mask, token features and the
/// five ragged length rows). A batch of these is one fixed-stride buffer that
/// Python can wrap with a single numpy structured view, so the model input does
/// not need a per-view copy.
pub fn pack_fixed_into(view: &ObservationBuffers, out: &mut Vec<u8>) {
    debug_assert_eq!(view.tokens.len(), OBSERVATION_TOKENS);
    out.reserve(OBSERVATION_FIXED_BYTES);
    put_u16(out, view.schema_version);
    for present in view.token_mask {
        out.push(u8::from(present));
    }
    pack_tokens(out, &view.tokens);
    for range in view.effect_ranges {
        put_u16(out, u16::try_from(range.len).unwrap_or(u16::MAX));
    }
    for range in view.repertoire_ranges {
        put_u16(out, u16::try_from(range.len).unwrap_or(u16::MAX));
    }
    for range in view.type_ranges {
        put_u16(out, u16::try_from(range.len).unwrap_or(u16::MAX));
    }
    for range in view.base_move_ranges {
        put_u16(out, u16::try_from(range.len).unwrap_or(u16::MAX));
    }
    for range in view.move_effect_ranges {
        put_u16(out, u16::try_from(range.len).unwrap_or(u16::MAX));
    }
}

/// The five ragged sections in effect/repertoire/type/base-move/move-effect
/// order, concatenated. Lengths live in the matching fixed block.
pub fn pack_ragged_into(view: &ObservationBuffers, out: &mut Vec<u8>) {
    for effect in &view.effects {
        pack_effect(out, effect);
    }
    for id in &view.repertoire {
        put_u16(out, *id);
    }
    for id in &view.types {
        put_u16(out, *id);
    }
    for base in &view.base_moves {
        pack_base_move(out, base);
    }
    for effect in &view.move_effects {
        pack_move_effect(out, effect);
    }
}

/// Reuse a caller-owned buffer for the packed view. The buffer is cleared first;
/// its capacity is retained across calls.
pub fn pack_observation_into(view: &ObservationBuffers, out: &mut Vec<u8>) {
    debug_assert_eq!(view.tokens.len(), OBSERVATION_TOKENS);
    let entry_bytes = view.effects.len() * 18
        + view.repertoire.len() * 2
        + view.types.len() * 2
        + view.base_moves.len() * 12
        + view.move_effects.len() * 42;
    out.clear();
    out.reserve(OBSERVATION_FIXED_BYTES + entry_bytes);
    pack_fixed_into(view, out);
    pack_ragged_into(view, out);
}

fn parse_action(kind: u8, own_slot: u8, move_slot: u8, target: i8, destination: u8, resource: u8) -> Result<AtomicAction> {
    let kind = match kind {
        0 => ActionKind::Pick,
        1 => ActionKind::Move,
        2 => ActionKind::Switch,
        3 => ActionKind::Pass,
        _ => return Err(EngineError::InvalidInput("action kind".into())),
    };
    let resource = match resource {
        0 => Resource::None,
        1 => Resource::Mega,
        _ => return Err(EngineError::InvalidInput("action resource".into())),
    };
    Ok(AtomicAction {
        kind,
        own_slot,
        move_slot,
        target_location: target,
        switch_destination: destination,
        resource,
    })
}

fn parse_side(side: u8) -> Result<SideId> {
    match side {
        0 => Ok(SideId::P1),
        1 => Ok(SideId::P2),
        _ => Err(EngineError::InvalidInput("side".into())),
    }
}

fn request_kind_id(kind: RequestKind) -> u8 {
    match kind {
        RequestKind::Preview => 0,
        RequestKind::Normal => 1,
        RequestKind::Replacement => 2,
        RequestKind::Wait => 3,
        RequestKind::Finished => 4,
    }
}

fn action_tuple(action: &AtomicAction) -> (u8, u8, u8, i8, u8, u8) {
    let kind = match action.kind {
        ActionKind::Pick => 0,
        ActionKind::Move => 1,
        ActionKind::Switch => 2,
        ActionKind::Pass => 3,
    };
    let resource = match action.resource {
        Resource::None => 0,
        Resource::Mega => 1,
    };
    (
        kind,
        action.own_slot,
        action.move_slot,
        action.target_location,
        action.switch_destination,
        resource,
    )
}

/// One native environment group: a shared immutable Dex/team pool plus the
/// battle states for this actor process (documented sizing: two groups of 1,024
/// environments with 16 workers each).
#[pyclass]
pub struct NativeEngine {
    batch: BattleBatch,
    /// Retained high-water storage: one batch call never reallocates these.
    buffers: ObservationBatchBuffers,
    scratch: Vec<u8>,
    scratch_ragged: Vec<u8>,
    step_results: Vec<crate::battle::StepResult>,
}

#[pymethods]
impl NativeEngine {
    /// `data_dir` holds the exported dex/scope assets; `teams_json` is the
    /// frozen team pool. Validation of the full observation catalogue happens
    /// once here, never in a batch call.
    #[new]
    #[pyo3(signature = (data_dir, teams_json, workers = 16))]
    fn new(data_dir: &str, teams_json: &str, workers: usize) -> PyResult<Self> {
        let dex = Arc::new(Dex::load(Path::new(data_dir)).map_err(to_py)?);
        let bytes =
            std::fs::read(teams_json).map_err(|e| to_py(EngineError::Io(e)))?;
        let teams: Vec<Team> =
            serde_json::from_slice(&bytes).map_err(|e| to_py(EngineError::Json(e)))?;
        Encoder::new(&dex).map_err(to_py)?;
        let batch = BattleBatch::new(dex, Arc::new(teams), workers).map_err(to_py)?;
        Ok(Self {
            batch,
            buffers: ObservationBatchBuffers::default(),
            scratch: Vec::new(),
            scratch_ragged: Vec::new(),
            step_results: Vec::new(),
        })
    }

    fn env_count(&self) -> usize {
        self.batch.len()
    }

    fn team_count(&self) -> usize {
        self.batch.teams.len()
    }

    /// Catalogue ID -> name for cold integration code (never a hot path).
    fn catalogue(&self, kind: &str) -> PyResult<Vec<String>> {
        self.batch
            .dex
            .names
            .get(kind)
            .cloned()
            .ok_or_else(|| to_py(EngineError::InvalidInput(format!("catalogue {kind}"))))
    }

    fn reset_batch(
        &mut self,
        team_a: Vec<usize>,
        team_b: Vec<usize>,
        seeds: Vec<(u16, u16, u16, u16)>,
        role_map: Vec<(u8, u8)>,
    ) -> PyResult<Vec<(u32, u32)>> {
        if team_a.len() != team_b.len()
            || team_a.len() != seeds.len()
            || team_a.len() != role_map.len()
        {
            return Err(to_py(EngineError::InvalidInput("reset batch length".into())));
        }
        let specs: Vec<ResetSpec> = team_a
            .into_iter()
            .zip(team_b)
            .zip(seeds)
            .zip(role_map)
            .map(|(((team_a, team_b), seed), role_map)| ResetSpec {
                team_a,
                team_b,
                seed: [seed.0, seed.1, seed.2, seed.3],
                role_map: [role_map.0, role_map.1],
            })
            .collect();
        self.batch
            .reset_batch(&specs)
            .map(|handles| handles.into_iter().map(|h| (h.slot, h.generation)).collect())
            .map_err(to_py)
    }

    /// One entry per environment: `(slot, generation, [(side, [action, ...]), ...])`.
    /// The whole submission is validated before any environment mutates.
    fn step_batch(
        &mut self,
        specs: Vec<EnvSpec>,
    ) -> PyResult<Vec<StepResult>> {
        let mut native: Vec<StepSpec> = Vec::with_capacity(specs.len());
        for (slot, generation, choices) in specs {
            let mut parsed = Vec::with_capacity(choices.len());
            for (side, actions) in choices {
                let side = parse_side(side).map_err(to_py)?;
                let mut list = Vec::with_capacity(actions.len());
                for (kind, own_slot, move_slot, target, destination, resource) in actions {
                    list.push(
                        parse_action(kind, own_slot, move_slot, target, destination, resource)
                            .map_err(to_py)?,
                    );
                }
                parsed.push(SideChoice { side, actions: list });
            }
            native.push(StepSpec {
                handle: Handle { slot, generation },
                choices: parsed,
            });
        }
        // Retained result buffer: the caller-owned vector keeps its capacity
        // across batches, so repeated stepping does not reallocate it.
        self.batch
            .step_batch_into(&native, &mut self.step_results)
            .map_err(to_py)?;
        Ok(self
            .step_results
            .iter()
            .map(|result| {
                (
                    result.accepted,
                    result.outcome.terminated,
                    result.outcome.truncated,
                    result.outcome.winner.map(|w| w.index() as i8),
                    result.outcome.operational_error.clone(),
                    request_kind_id(result.request_kinds[0]),
                    request_kind_id(result.request_kinds[1]),
                )
            })
            .collect())
    }

    /// Encoded numeric observations, one packed `bytes` blob per request in
    /// submitted order (both players may request the same handle).
    fn observe_encoded_batch(
        &mut self,
        py: Python<'_>,
        handles: Vec<(u32, u32)>,
        sides: Vec<u8>,
    ) -> PyResult<Vec<Py<PyBytes>>> {
        if handles.len() != sides.len() {
            return Err(to_py(EngineError::InvalidInput(
                "observation batch length".into(),
            )));
        }
        let mut requests = Vec::with_capacity(handles.len());
        for ((slot, generation), side) in handles.into_iter().zip(sides) {
            requests.push((
                Handle { slot, generation },
                parse_side(side).map_err(to_py)?,
            ));
        }
        // Split the borrows so the immutable Dex (via the group) and the
        // retained observation/scratch buffers can be used in one call.
        let NativeEngine {
            batch,
            buffers,
            scratch,
            ..
        } = self;
        let encoder = Encoder::from_validated_dex(&batch.dex).map_err(to_py)?;
        // Retained storage: the previous batch's pool is reused without a
        // fresh allocation when the request count does not grow.
        batch
            .observe_encoded_batch_into(&requests, &encoder, buffers)
            .map_err(to_py)?;
        Ok(buffers
            .as_slice()
            .iter()
            .map(|view| {
                scratch.clear();
                pack_observation_into(view, scratch);
                PyBytes::new(py, scratch).unbind()
            })
            .collect())
    }

    /// Batch observation payload with a fixed stride: `fixed` holds exactly
    /// `OBSERVATION_FIXED_BYTES` per request and `ragged` concatenates the five
    /// ragged sections in request order. Python can wrap `fixed` in a single
    /// numpy structured view (zero per-view copy) and slice `ragged` with the
    /// per-view counts. Semantics are identical to `observe_encoded_batch`.
    fn observe_fixed_batch(
        &mut self,
        py: Python<'_>,
        handles: Vec<(u32, u32)>,
        sides: Vec<u8>,
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>)> {
        if handles.len() != sides.len() {
            return Err(to_py(EngineError::InvalidInput(
                "observation batch length".into(),
            )));
        }
        let mut requests = Vec::with_capacity(handles.len());
        for ((slot, generation), side) in handles.into_iter().zip(sides) {
            requests.push((
                Handle { slot, generation },
                parse_side(side).map_err(to_py)?,
            ));
        }
        let NativeEngine {
            batch,
            buffers,
            scratch,
            scratch_ragged,
            ..
        } = self;
        let encoder = Encoder::from_validated_dex(&batch.dex).map_err(to_py)?;
        batch
            .observe_encoded_batch_into(&requests, &encoder, buffers)
            .map_err(to_py)?;
        scratch.clear();
        scratch_ragged.clear();
        scratch.reserve(buffers.len() * OBSERVATION_FIXED_BYTES);
        for view in buffers.as_slice() {
            pack_fixed_into(view, scratch);
            pack_ragged_into(view, scratch_ragged);
        }
        Ok((
            PyBytes::new(py, scratch).unbind(),
            PyBytes::new(py, scratch_ragged).unbind(),
        ))
    }

    /// Legal completions of a joint action prefix, as action tuples. The prefix
    /// is the actions already chosen for this request's decision branches.
    fn candidates(
        &self,
        slot: u32,
        generation: u32,
        side: u8,
        prefix: Vec<(u8, u8, u8, i8, u8, u8)>,
    ) -> PyResult<Vec<ActionTuple>> {
        let side = parse_side(side).map_err(to_py)?;
        let request = self
            .batch
            .request(Handle { slot, generation }, side)
            .map_err(to_py)?;
        let mut parsed = Vec::with_capacity(prefix.len());
        for (kind, own_slot, move_slot, target, destination, resource) in prefix {
            parsed.push(
                parse_action(kind, own_slot, move_slot, target, destination, resource)
                    .map_err(to_py)?,
            );
        }
        let candidates = request.candidates(&parsed).map_err(to_py)?;
        Ok(candidates.iter().map(action_tuple).collect())
    }

    fn request_kind(&self, slot: u32, generation: u32, side: u8) -> PyResult<u8> {
        let side = parse_side(side).map_err(to_py)?;
        let request = self
            .batch
            .request(Handle { slot, generation }, side)
            .map_err(to_py)?;
        Ok(request_kind_id(request.kind))
    }

    fn request_branches(&self, slot: u32, generation: u32, side: u8) -> PyResult<Vec<u8>> {
        let side = parse_side(side).map_err(to_py)?;
        let request = self
            .batch
            .request(Handle { slot, generation }, side)
            .map_err(to_py)?;
        Ok(request.branch_slots())
    }

    fn request_moves(
        &self,
        slot: u32,
        generation: u32,
        side: u8,
        own_slot: u8,
    ) -> PyResult<Vec<MoveRow>> {
        let side = parse_side(side).map_err(to_py)?;
        let request = self
            .batch
            .request(Handle { slot, generation }, side)
            .map_err(to_py)?;
        let slot_request = request
            .slots
            .get(own_slot as usize)
            .ok_or_else(|| to_py(EngineError::InvalidInput("slot".into())))?;
        Ok(slot_request
            .moves
            .iter()
            .map(|move_choice| {
                (
                    move_choice.id,
                    move_choice.slot,
                    move_choice.pp,
                    move_choice.disabled,
                    slot_request.can_mega,
                )
            })
            .collect())
    }

    fn snapshot(&self, slot: u32, generation: u32) -> PyResult<Vec<u8>> {
        self.batch
            .snapshot(Handle { slot, generation })
            .map_err(to_py)
    }

    fn restore(&mut self, slot: u32, generation: u32, bytes: Vec<u8>) -> PyResult<()> {
        self.batch
            .restore(Handle { slot, generation }, &bytes)
            .map_err(to_py)
    }

    fn enable_trace(&mut self, slot: u32, generation: u32) -> PyResult<()> {
        self.batch
            .enable_trace(Handle { slot, generation })
            .map_err(to_py)
    }

    /// Development-only trace export as JSON bytes; never part of a step path.
    fn export_trace(&self, slot: u32, generation: u32) -> PyResult<Option<Vec<u8>>> {
        let trace = self
            .batch
            .export_trace(Handle { slot, generation })
            .map_err(to_py)?;
        match trace {
            Some(trace) => serde_json::to_vec(trace)
                .map(Some)
                .map_err(|e| to_py(EngineError::Json(e))),
            None => Ok(None),
        }
    }
}

fn to_py(error: EngineError) -> PyErr {
    match error {
        EngineError::InvalidInput(message) => pyo3::exceptions::PyValueError::new_err(message),
        EngineError::Unsupported(message) => {
            pyo3::exceptions::PyNotImplementedError::new_err(message)
        }
        EngineError::Io(error) => pyo3::exceptions::PyOSError::new_err(error.to_string()),
        other => pyo3::exceptions::PyRuntimeError::new_err(other.to_string()),
    }
}

#[pymodule]
fn pa3_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeEngine>()?;
    m.add("SCHEMA_VERSION", SCHEMA_VERSION)?;
    m.add("OBSERVATION_TOKENS", OBSERVATION_TOKENS)?;
    m.add("CATEGORY_COUNT", CATEGORY_COUNT)?;
    m.add("FLOAT_COUNT", FLOAT_COUNT)?;
    m.add("FLAG_COUNT", FLAG_COUNT)?;
    m.add("OBSERVATION_FIXED_BYTES", OBSERVATION_FIXED_BYTES)?;
    Ok(())
}
