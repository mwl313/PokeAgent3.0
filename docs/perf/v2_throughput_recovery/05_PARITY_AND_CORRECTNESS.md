# Correctness evidence

Final closure on executed optimization source
`b3c189ba9e61ffa002a3875b0848ac592f6450a0`: **28 passed in2.33s**:

```bash
OMP_NUM_THREADS=2 PYTHONPATH=engine/python:. \
 /home/minidc/Desktop/PokeAgent3.0/PokeAgent3.0-agent1-pa3/.venv/bin/python \
 -m pytest -q tests/agent/test_v2_source_migration.py \
 tests/agent/test_v2_operational_contract.py \
 tests/agent/test_a_checkpoint_envelope.py \
 tests/agent/test_v2_throughput_paths.py
```

Persistent log: `evidence/final_closure_tests.log`. Earlier focused47-test
run passed in9.81s (`evidence/focused_tests_final.log`); the counts below
describe separate runs and must not be summed into a unique test count.

The original-workspace historical replay completed13 iterations/53,248 games.
Both rank artifacts were independently loaded on CPU: full model and Adam
digests match historical telemetry at57,344, ranks have the same source,
iteration14, dataset/action/opponent contract and rank digests. Scheduler,
GradScaler, owned RNG and collector restart state exist. Both collector drain
receipts report complete with0 in-flight battles. All28 team-stat receipts
passed the codec's canonical-content hash validation and clock/W/L/D
reconciliation. Manifest/rank SHA256 and a separate local full-state backup
are verified in `evidence/CLOSEOUT_RECEIPT.json`.

The raw recovery utility reports HISTORICAL_REPLAY_DIFFERS whenever its fixed
73,728 target is not reached, even when every digest matches. Its raw output
is retained unchanged. The audited closeout classification is
**USER_STOPPED_VERIFIED_PARTIAL**, not a digest failure or full recovery.

Exploratory source8a743ad full-stack3×5 comparisons match rollouts/model/Adam/
work; same-source uninterrupted-versus-restored test was BIT_EXACT at12,288.
Both ran without the original cuBLAS workspace setting. Their results prove
only that tested runtime and are not production-runtime acceptance. The
corrected-environment panel and continuation were cancelled before launch.
No all-green production gate claim is made.

Focused first run:53 passed in13.65s. After block store:61 passed in12.32s.
Migration/ownership/envelope/operational tests:27 passed in2.00s.
Commands use `OMP_NUM_THREADS=2 PYTHONPATH=engine/python:.` and the existing
`PokeAgent3.0-agent1-pa3/.venv/bin/python -m pytest -q`.

Focused files: `test_v2_throughput_paths`, `test_v2_action_contract`,
`test_v2_observation_contract`, `test_cached_rollout`, `test_ddp_global_gradient`,
`test_real_gradient_parity`, `test_v2_team_stats`, `test_v2_source_migration`,
`test_a_checkpoint_envelope`, `test_v2_operational_contract` under tests/agent.

Captured input CPU parity: observations, candidate tables/order/masks/refs,
stored actions/selected indices/team IDs, every PPO loss term, all parameter
gradient digest, one Adam step and resulting model digest match bitwise across
independent source processes. Synthetic advantage/return targets are declared;
this does not replace a real two-rank PPO or continuation test. Integer and
ownership tests include signed targets, arbitrary active roster, switch/pass,
Struggle sentinel, Mega resources, preview, empty/ragged/repeated selections,
five-family payload preservation, source mutation and returned-tensor mutation.

Broad inherited suite:261 passed,8 failed,1 skipped (49.16s). Seven failures
were reproduced on untouched5ccc51b (38 pass/7 fail in selected old files):
old precision-v1 expectations (2), invalid native/compact V2 fixture assumptions
(2), invalid move-slot4 fixtures (2), differing seed in strict checkpoint
config (1). An eighth near-zero candidate-padding scalar comparison failed
in broad-suite order (3.58e-7 versus2e-7 absolute tolerance); isolated original
padding suite passed. No gate or tolerance was relaxed. This is recorded as
an unresolved order-sensitive legacy test, not silently marked green.

Paths and artifact hashes are enumerated in RECEIPT.json when final experiments
complete. Runtime tests must be read alongside their source revisions and dirty
probe hashes. Controlled repetitions overlapping unit/probe work are excluded.
