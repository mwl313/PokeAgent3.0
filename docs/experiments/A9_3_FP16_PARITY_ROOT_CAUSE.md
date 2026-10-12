# Track A9.3 — FP16 parity root cause

## Finding

The original Mode D contract (FP16-rounded observations, AMP encoder, FP32
scorer/GRU) passed the short A9.2 arm but failed during a long run at iteration
14: sampled-versus-recomputed logprob maximum absolute error was `1.438e-3`
against the unchanged `1e-3` fail-closed gate. The failure was on rank 1;
there were no operational errors and the preceding 13 boundaries had passed.

The failure is consistent with V100 FP16 matrix reductions accumulating
shape-dependent rounding error in the AMP encoder. It is not an action legality,
mask, candidate ordering, PPO-objective, or tolerance issue.

## D1 intervention

The minimal intervention sets
`torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False`
before model construction. Mode D, the gate, and all PPO defaults remain
unchanged. This is opt-in as `--precision-stability fp16-reduction-off` and is
recorded in the experimental envelope.

The bounded D1 arm completed 8/8 gates over 16,384 games at 85.76 games/s.
Layer-by-layer causal attribution beyond this bounded evidence remains open;
the result does establish a reproducible, fail-closed stability candidate.
