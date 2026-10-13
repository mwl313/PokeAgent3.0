# PPO, DDP and native-engine validation contract

## 1. Engine mechanics and player-safe observation validation

A battle that terminates is not necessarily mechanically correct. Compare reachable mechanics from the **actual 1,220-team corpus and dynamic effect closure** against pinned Showdown reference at recorded commit `14546894d86f9589ac11130c510bbe73b6968665` (verify the pin from manifest before use). Use original fixture hashes and additional hand-authored adversarial fixtures. Scope.json's `training_ready=false`, `closure_certified=false` must not be silently reinterpreted as proof of full-Reg coverage.

Suggested witness matrix:

- Target and redirection: spread, ally, opponent left/right, random target, Follow Me/Rage Powder, protection.
- Speed and priority: Trick Room, Tailwind, status priority, ties with fixed RNG.
- Dynamic types and identity: Soak, Protean/Libero, Magic Powder, Double Shock, Burn Up, Mega, Mimicry, Illusion, Transform and switch re-entry.
- Switching and resources: leading pairs, force switch, Revival Blessing, two replacements, pivoting, Mega resource once per side.
- Interactions: weather/terrain/screen/volatile duration, switch cleanup, PP/Encore/Disable/Imprison, hidden data restriction.

For each fixture retain seed, moves, exact input teams, rank/order, Showdown results vs native results, public events, actual damage, public health display, reason for divergence. Tag `PASS`, `BLOCKED_UNSUPPORTED`, `FAIL`, `NOT_REACHABLE`; do not treat unimplemented callbacks as neutral.

## 2. PPO mathematical independent reference

Keep a small independent, readable Python reference separate from the optimized learner implementation. Verify:

- Terminal rewards per side of **one natural match**: winner side +1, other -1; draw both 0. Valid natural completion only, not engine error/truncation. Each recorded side's final request row receives terminal reward exactly once.
- GAE by `(match_id, side)` and `request_index` only; no opponent requests or next match cross-contamination. `gamma=1`, `lambda=0.95` by baseline configuration, with explicit terminal zero bootstrap and nonterminal bootstrap scenario.
- Singleton requests and Wait/Finished generate no erroneous policy-gradient rows; critic value rows follow explicit design.
- Global advantage normalization over actor rows with identical behavior single-rank and multi-rank; zero/singleton count edge cases.
- PPO ratios from recomputed conditional logprob sums, correct clipping, policy coefficient, entropy normalized over active branches, uniform KL, value MSE coefficient, target KL gate, gradient clip.
- No `NaN`, `Inf`, silent clamping or saturation for invalid row. Correct masking of padded microbatches.

Existing reference tests to retain and expand:

`tests/agent/test_gae.py`, `test_full_ppo_objective_reference.py`, `test_ddp_global_gradient.py`, `test_manual_allreduce_parity.py`, `test_logprob_recompute.py`, `tests/integration/test_collect_both_current_sides.py`, `test_ddp_collective_protocol.py`.

Do not rewrite the math because earlier audits suspected it: specifically log which tests demonstrated correctness and which fail on V2 changes.

## 3. Actor-Learner semantic parity

The old recompute gate can pass if Actor and Learner both consume the same **incorrect** observation. Therefore use two complementary gates:

**Gate A: reference semantic integrity** — independent fixture code asserts the emitted player-visible observation, typed tokens, ragged features and candidate IDs actually represent intended battle facts.

**Gate B: numerical recomputation** — behavior Actor's stored selected conditional logprobs compared with learner's recomputed results under the same model/Mode D, with explicit treatment of AMP batch-shape sensitivity. Tolerance defaults to historical `1e-3`; do not widen globally to suppress failures. If bounded original-batch audit is used, require hashes and evidence of contextual precision sensitivity.

Hash observation model-input features + action candidate IDs + selected branches + model SHA at rollout capture; use same contract in recompute validation. Test batch regrouping, microbatch recomputation, padding and two V100 ranks.

## 4. DDP invariants

- One global iteration is N natural committed matches across rank0+rank1; do not double-count both sides as separate matches.
- Per-rank gradients are properly weighted by **global actor rows** for actor loss, **global value rows** for critic, accounting for DDP average gradients.
- Both ranks share identical model, optimizer, GradScaler and scheduler state hashes after each verified iteration. Every rank must see identical collective sequence counts even when rows differ.
- Failed recompute or divergence triggers safe collective halt before checkpoint publication and before team stats commit.
- No skipped optimizer step accepted silently. If optimizer step skips due to GradScaler, report gate failure/decide permissible policy explicitly.
- Include time, games/sec, microbatch GPU memory, cumulative storage and data transfer profiling. Quantify V2 attention overhead without hiding it in baseline changes.

## 5. Checkpoint and resume integrity

A generation receipt must include:

- run ID / generation ID / iteration ID / source SHA / data corpus SHA / model architecture hash;
- Observation, Action, Ragged, Precision and Candidate version IDs;
- Model state, Adam moments, GradScaler, RNG, learning-rate scheduler;
- exact committed natural matches (global and per rank), rollout row counts and two-rank consistency digests;
- per-team stats receipt identity and per-rank stats payload hashes.

All writes must be staged and verified then published atomically. A crash may leave tentative artifacts that are never interpreted as committed. Resume from latest fully verified generation, reconstruct counters from the committed checkpoint watermark, and replay unfinished games **without counting old tentative results**.

## 6. Evaluation integrity

V1 and V2 policies have different schema and perhaps architecture, so a mixed-checkpoint V1/V2 league is **not scientifically comparable by default**. Create separate V2 league with pinned fixture manifest/seed, seat-swaps, scoring, checkpoint hashing, and independent evaluation data. The 1,220 train teams do not provide out-of-pool generalization. Reserve a separate future fixture panel of unseen roster groups to measure generalization, without modifying frozen train teams.

Fix numerical CDF padding guard in evaluator if reproduced; never choose padded candidates, preserve fixture-keyed random uniforms. League startup must be independently controlled; do not kill an existing unrelated evaluation league while repairing PPO.

## 7. Minimum gate outputs

Place under new run `validation/` directory:

- `engine_oracle_matrix.jsonl` + concise `ENGINE_PARITY_REPORT.md`
- `semantic_obs_action_cases.jsonl` + feature hashes
- `ppo_reference_results.json`
- `recompute_gate_results.jsonl`
- `ddp_parity_receipts.jsonl`
- `checkpoint_resume_replay_test.json`
- `performance_v2.json`

All evidence carries git SHA, exact command, dependency versions, dataset SHA and test ID from `09_TEST_MATRIX.csv`. A textual `PASS` without test logs is not a passing gate.
