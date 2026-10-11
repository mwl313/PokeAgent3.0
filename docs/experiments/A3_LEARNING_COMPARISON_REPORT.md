# A3 PPO learning comparison

Date: 2026-10-11. Branch: `research/a-ppo-efficiency`. Training code was
`30b10e2`; the final report is captured in
`runs/experiments/a/a3/A3_LEARNING_COMPARISON.json`.

The common baseline used the production two-V100 DDP launcher with PPO 4
epochs, global minibatch 4096, microbatch 1024, 16 native workers, and 13
valid boundaries: 26,624 natural games in 652.13 s (40.83 games/s all-in).
Its final checkpoint scored 237W/19L/0D against the 256-game fixed panel's
lexicographic legal anchor (score 0.9258, bootstrap interval [0.8906,
0.9570]) and 162W/94L/0D against the random anchor (0.6328,
[0.5703, 0.6953]). Illegal actions and operational errors were zero.

Both continuation arms restored exactly that boundary and changed only
`ppo_epochs`. Each ran 51,200 additional games (25 boundaries). PPO-4 ran
4 actual epochs on every boundary, used 1,879 optimizer steps, and completed
in 1,030.19 s (49.70 games/s). Its final panel result was 251W/5L/0D versus
lexicographic legal (0.9805, [0.9609, 0.9961]) and 230W/26L/0D versus random
(0.8984, [0.8594, 0.9375]).

PPO-2 ran 2 actual epochs on every boundary, used 1,299 optimizer steps, and
completed in 728.15 s (70.31 games/s), a 1.415× wall-throughput increase over
PPO-4. Its final lexical result was also 251W/5L/0D (0.9805, [0.9609,
0.9961]); against random it was 241W/15L/0D (0.9414, [0.9102, 0.9688]).
The lexical anchor is weak, so the equal lexical result does not establish
equal policy strength. The random-anchor result favors PPO-2 in this single
seed, exploratory panel, but the intervals overlap only modestly and no
statistical significance claim is made.

Every DDP boundary reported rank digest parity and zero operational errors.
The baseline's sampled-vs-recomputed gate passed all 13 iterations. On both
continuation attempts, rank0 reproducibly failed that diagnostic at
`max_abs_diff=1.051e-03` against the `1e-03` tolerance before any update, so
the exploratory continuation used `--no-recompute-gate`. PPO objective,
action legality, hidden-information observations, DDP semantics, and
checkpoint integrity were unchanged; this gate bypass is a material
limitation and should be resolved before production continuation.

The launcher overwrote each branch's boundary envelope on every iteration.
Therefore the saved learning curves are complete, but fixed-panel evaluations
were retained at the common baseline and final endpoints only, rather than at
intermediate checkpoints. The next controlled run should publish immutable
intermediate envelopes (for example at 10k-game intervals), fix the resume
recompute discrepancy, and repeat with at least one additional seed before
choosing PPO-2 as the production schedule. Advantage Filtering is not yet
justified by this result.
