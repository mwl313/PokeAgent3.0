# A2-R end-to-end continuation validation

Date: 2026-10-11. Source branch: `research/a-ppo-efficiency`.

The bounded GPU run was executed while the A2-R changes were present but
before the final commit (`git_sha=166b6fe`, dirty worktree). The clean final
implementation is `41f4e74`; rerun the same bounded gate from that clean SHA
before treating this as an integration receipt.

The experimental mode is opt-in through `scripts/run_ddp_ppo.py`:

```text
--experimental-checkpoint PREFIX
--experimental-resume PREFIX
--dataset-manifest PATH
```

Each rank writes/reads `PREFIX.rankN`. Normal `--checkpoint` behavior is
unchanged. The mode uses static collector slots, saves only after collection
has drained all battles, the PPO update has completed, and all rank model and
optimizer digests agree. DDP is initialized again on every process; no
communicator or reducer is serialized.

## Bounded GPU comparison

Both runs used two Tesla V100s, the same source checkout, seed `20261009`,
`games=4`, `envs=4`, `workers=1`, `minibatch=8`, `microbatch=4`, `epochs=2`,
static slots, no recompute gate, and the same dataset manifest.

Run A was two uninterrupted iterations. Run B was one iteration, process
exit, then one restored iteration from the two rank envelopes.

| Metric | Uninterrupted iteration 2 | Restored iteration |
|---|---:|---:|
| rank 0 games / rows | 4 / 109 | 4 / 109 |
| rank 1 games / rows | 4 / 98 | 4 / 98 |
| committed matches | 16 | 16 |
| optimizer steps cumulative | 102 | 102 |
| operational errors | 0 | 0 |
| policy/value report | equal | equal |
| model digest | equal | equal |
| owned shuffle RNG state | equal | equal |
| optimizer digest | differs only in one `exp_avg_sq` value by `1.3878e-17` | same result on both ranks |

The model tensors were byte-identical and the second-iteration row counts,
decisions, reports and outcomes matched. The optimizer digest is not
byte-identical because CUDA execution produced a sub-ULP Adam second-moment
difference after process restart. This is not claimed as bit exact.

**Classification: `BOUNDARY_EQUIVALENT`.** The validated claim is that a
completed iteration boundary can be restored with identical contract state,
RNG streams, data counters and model trajectory at this bounded workload,
within the measured optimizer tolerance. It does not establish bit-exact
floating-point continuation across a fresh CUDA process.

## Remaining limitations

The collector restart contract is deterministic only at a drained static-slot
boundary. In-flight native handles are rejected; rolling slots are rejected in
experimental mode. Historical opponents are still required to be empty and
their manifest is checked. A future exactness pass would need deterministic
CUDA/collective controls or a documented optimizer-state tolerance.

No 4-vs-2 learning comparison was started.
