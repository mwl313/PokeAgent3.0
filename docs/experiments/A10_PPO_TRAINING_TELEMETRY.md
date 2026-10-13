# A10 PPO iteration telemetry

## Scope
Additive metrics writing only. No changes to PPO loss, gradients, optimizer, DDP collectives, LR schedule, checkpoint format, or Evaluation League.

The existing `PPOLearner.UpdateReport` already records policy loss, value loss, normalized entropy, uniform reverse KL, sampled PPO approximate KL, ratio, clip fraction, pre-clip gradient norm, learning rate, skipped steps, and AMP scaler. The launcher previously retained per-iteration reports in memory until the long-running worker ended.

The updated launcher appends one record per completed iteration and rank to:

```
<--report path>.rank0.ppo.jsonl
<--report path>.rank1.ppo.jsonl
```

Append occurs after rank digest agreement and the final iteration barrier. One small fsynced JSON object is written per iteration, and ordinary logging failures are nonfatal.

Each record includes absolute iteration, committed match clock, performance timings and calculated games/sec, existing PPO metrics, collector statistics, and recompute gate output.

The loss, KL, entropy, ratio and clip values were already DDP-aggregated by the learner. Gradient norm is rank-local. Do not sum two ranks or interpret uniform reverse KL as old-policy/new-policy KL.

## Activation

**Git edits cannot hot-patch the running Python process.** The production A10 campaign should not be restarted just for observability.

Deploy during an already planned, verified checkpoint-based continuation (after confirming the branch is compatible with the exact checkpoint's source/version identity), or implement a separately validated opt-in live telemetry mechanism. Do not use forced process suspension, signals, ptrace, or arbitrary injection into the current learner. The initial ~1.16 million game interval cannot be backfilled with unrecorded losses.

The exact launcher's `--report` argument determines destination. Preserve and inspect the deployed launch arguments.

Run the focused regression tests in the MiniDC checkout:

```bash
python -m pytest -q tests/agent/test_a10_training_telemetry.py
```

After normal activation, check latest lines:

```bash
tail -n 2 '<REPORT_PATH>.rank0.ppo.jsonl'
tail -n 2 '<REPORT_PATH>.rank1.ppo.jsonl'
```

If an iteration emits a nonfinite number or disk write fails, it will generate a nonfatal warning rather than kill the PPO worker.

## Future optional metrics

Explained variance requires access to paired value predictions and returns before the buffer is discarded, and should be weighted by valid value rows. Advantage mean/std, preclip gradient norm distribution (not only max/last), and per-step GradScaler skip reason likewise require additional instrumentation. Add them in a subsequent opt-in patch after measuring logging overhead.

Do not treat training loss trends as an independent measure of policy strength; evaluate real games in the A10 league.
