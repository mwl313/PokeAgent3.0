# Track A4: experimental learning-rate comparison

Date: 2026-10-11. Branch: `research/a-ppo-efficiency`. The original PPO-2
249,856-match checkpoint was preserved before any run:
`runs/experiments/a/a3_ppo2_to250k/original_backup/boundary.rank{0,1}`.
SHA-256 hashes are recorded in the accompanying JSON. Both arms explicitly
migrated the envelope provenance from training source
`30b10e28d608f869811a4525afaa8101613c7adc` to the current implementation; no
source check was bypassed implicitly.

Both arms resumed the same model, Adam moments, GradScaler, scheduler clock,
RNG streams, team pool, and PPO-2 configuration, adding 51,200 natural matches
to 301,056. Arm LR-1.0 used effective LR approximately `3.0e-4`, completed in
552.0 s, used 441 optimizer steps, and ended with entropy 0.4834, KL 0.00628,
and value loss 0.2294. Arm LR-0.5 used effective LR approximately `1.5e-4`,
completed in 558.0 s, used 450 optimizer steps, and ended with entropy 0.4804,
KL 0.00532, and value loss 0.2309. Rank digests matched throughout and both
arms reported zero operational errors. Early stopping occurred once in LR-1.0
and twice in LR-0.5.

The balanced 1,024-pair / 2,048-game fixed panel used the full training team
pool, deterministic seat-swapped fixtures, the Rust engine, and the frozen
PPO-4 77,824-match checkpoint as opponent. LR-1.0 scored 1,200W/848L/0D
(score 0.58594, paired bootstrap interval [0.56592, 0.60742]); LR-0.5 scored
1,264W/784L/0D (0.61719, [0.59814, 0.63672]). Both had zero illegal actions
and operational errors. On this single seed, LR-0.5 is the more promising
continuation: it is modestly slower but stronger and has a slightly lower KL.
The result is exploratory and does not establish production superiority.

The evaluator now derives independent deterministic RNG streams per fixture and
seat and computes uncertainty from paired fixture means rather than treating
the two seats as unrelated samples. The known sampled-vs-recomputed logprob
issue remains unresolved (`1.0507107e-3` versus `1e-3`); these runs therefore
used the explicit `--no-recompute-gate` flag and must not be promoted to
production evidence until the narrow AMP/packed-row discrepancy is fixed.

Historical self-play was not started in this milestone. The current collector
has a single model execution path and current-vs-current row attribution; a
history pool needs an explicit per-environment frozen model/role path before
it can safely exclude historical actions from PPO rows. Implementing that
without changing row provenance is the next code milestone. No historical
opponent training was silently substituted for the LR comparison.
