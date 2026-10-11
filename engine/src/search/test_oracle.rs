//! Fully specified known-world branch harness for tests.
//!
//! `KnownWorldTestOracle` holds and restores complete engine snapshots. Those
//! snapshots contain private state and RNG, so this type is **TEST ORACLE ONLY**:
//! it is not player-information safe, not a CTS teacher, and must never create
//! labels or be wired into Python/training.

use crate::{
    actions::{AtomicAction, Request},
    batch::{BattleBatch, Handle, SideChoice, StepSpec},
    battle::StepResult,
    state::SideId,
    EngineError, Result,
};

/// Result of one isolated, fully specified joint-action branch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnownWorldBranch {
    pub result: StepResult,
    /// Complete post-resolution snapshot for deterministic test assertions.
    pub snapshot: Vec<u8>,
}

/// Snapshot/restore branch runner for one owned `BattleBatch` slot.
///
/// The baseline is captured at construction and restored before *and* after
/// every branch. This is intentionally a small correctness utility, not a
/// reusable simulator pool or a production search interface.
pub struct KnownWorldTestOracle {
    batch: BattleBatch,
    handle: Handle,
    baseline: Vec<u8>,
}

impl KnownWorldTestOracle {
    pub fn new(batch: BattleBatch, handle: Handle) -> Result<Self> {
        let baseline = batch.snapshot(handle)?;
        Ok(Self {
            batch,
            handle,
            baseline,
        })
    }

    /// Enumerate complete legal action sequences for one currently requested
    /// side at the saved baseline. Conditional candidates are expanded only
    /// through legal prefixes.
    pub fn legal_joint_actions(&mut self, side: SideId) -> Result<Vec<Vec<AtomicAction>>> {
        self.batch.restore(self.handle, &self.baseline)?;
        let request = self.batch.request(self.handle, side)?.clone();
        let mut out = Vec::new();
        enumerate_complete(&request, &mut Vec::new(), &mut out)?;
        for actions in &out {
            request.validate_joint(actions)?;
        }
        self.batch.restore(self.handle, &self.baseline)?;
        Ok(out)
    }

    /// Return the currently legal next-prefix candidates from the saved
    /// fixture. This is test-only access to the engine request tree; it does
    /// not expose a hidden opponent state to a planner.
    pub fn legal_actions_for_prefix(
        &mut self,
        side: SideId,
        prefix: &[AtomicAction],
    ) -> Result<Vec<AtomicAction>> {
        self.batch.restore(self.handle, &self.baseline)?;
        let request = self.batch.request(self.handle, side)?.clone();
        let candidates = request.candidates(prefix)?;
        self.batch.restore(self.handle, &self.baseline)?;
        Ok(candidates)
    }

    /// Resolve both submitted complete choices as one simultaneous decision.
    /// Neither choice is selected after observing a resolved opponent action.
    /// The original baseline is restored even if stepping fails.
    pub fn branch(
        &mut self,
        p1_actions: Vec<AtomicAction>,
        p2_actions: Vec<AtomicAction>,
    ) -> Result<KnownWorldBranch> {
        self.batch.restore(self.handle, &self.baseline)?;
        let stepped = self.batch.step_batch(&[StepSpec {
            handle: self.handle,
            choices: vec![
                SideChoice {
                    side: SideId::P1,
                    actions: p1_actions,
                },
                SideChoice {
                    side: SideId::P2,
                    actions: p2_actions,
                },
            ],
        }]);
        let branch = match stepped {
            Ok(mut results) => {
                let result = results.pop().ok_or_else(|| {
                    EngineError::InvalidInput("missing test-oracle branch result".into())
                })?;
                let snapshot = self.batch.snapshot(self.handle)?;
                Ok(KnownWorldBranch { result, snapshot })
            }
            Err(error) => Err(error),
        };
        let restore = self.batch.restore(self.handle, &self.baseline);
        match (branch, restore) {
            (_, Err(error)) => Err(error),
            (result, Ok(())) => result,
        }
    }

    /// Snapshot of the unchanged fixture baseline, for isolation assertions.
    pub fn baseline_snapshot(&self) -> &[u8] {
        &self.baseline
    }

    /// Reset the owned slot to its captured baseline after an external test
    /// assertion. This never changes the captured baseline itself.
    pub fn restore_baseline(&mut self) -> Result<()> {
        self.batch.restore(self.handle, &self.baseline)
    }

    pub fn snapshot(&self) -> Result<Vec<u8>> {
        self.batch.snapshot(self.handle)
    }
}

fn enumerate_complete(
    request: &Request,
    prefix: &mut Vec<AtomicAction>,
    out: &mut Vec<Vec<AtomicAction>>,
) -> Result<()> {
    let candidates = request.candidates(prefix)?;
    if candidates.is_empty() {
        request.validate_joint(prefix)?;
        out.push(prefix.clone());
        return Ok(());
    }
    for action in candidates {
        prefix.push(action);
        enumerate_complete(request, prefix, out)?;
        prefix.pop();
    }
    Ok(())
}
