# Track A5 Historical Self-Play comparison

Date: 2026-10-11. Branch: `research/a-ppo-efficiency`. Both arms resumed the
same A4 PPO-2 LR-0.5 checkpoint at 301,056 committed matches. The source
migration was explicit and the original A4 checkpoint was never overwritten.
The recompute gate remained explicitly disabled because the known
`1.0507107e-3` versus `1e-3` discrepancy is unresolved.

The control arm ran 51,200 natural matches to 352,256 using current/current
self-play only. Across both ranks it produced 875,177 learner rows (34.19 global rows/game; rank-0 local rows were 438,248), 441
additional optimizer steps, 549.3 s wall time, and 0 operational errors. The
historical arm ran the same 51,200-match budget with 70% current/current and
30% current/historical matches. Across both ranks it produced 748,530 learner rows (29.24 global rows/game; rank-0 local rows were 374,116), 400 optimizer steps, 550.8 s wall time, and 0 operational errors.
The historical arm completed 17,963 current/current and 7,637
current/historical games; historical rows were excluded from PPO. Rank digests
matched in every iteration. Approximate rows/sec were 798 for control and 679
for historical on rank 0, so the historical mixture reduced PPO sample work
without changing the natural-game budget.

Final rank-0 metrics were:

| arm | effective LR | entropy | approx KL | value loss | actor rows |
|---|---:|---:|---:|---:|---:|
| control | 1.499996e-4 | 0.43698 | 0.00504 | 0.23084 | 17,241 |
| historical | 1.499996e-4 | 0.43570 | 0.00548 | 0.22236 | 14,863 |

The primary 1,024-pair / 2,048-game direct panel scored control versus
historical at 1,004W/1,044L/0D (0.4902, paired bootstrap [0.4731, 0.5088]).
This is inconclusive and does not support a claim that historical training is
better. Against the common 301,056-match A4 starting checkpoint on 512-game
panels, control scored 266W/246L (0.5195, [0.4844, 0.5547]) and historical
scored 265W/247L (0.5176, [0.4824, 0.5547]); both intervals overlap heavily.
All evaluations had zero illegal actions and operational errors.

The collector now routes requests by `(policy_id, branch_count)`, freezes
historical models in inference mode, keeps current/current two-sided rows,
records only the current seat for current/historical matches, and records the
actual opponent policy ID in both typed and packed rows. Match assignments are
deterministic and fixed for the natural match. Rolling-slot historical mode is
rejected explicitly; the experiment used static slots. The existing focused
suite passed (9 tests), plus native mixed-policy smoke tests verified both
orientations, row provenance, reward completion, and zero engine errors.

Historical Self-Play is therefore functional and operationally safe, but this
single seed shows no measurable strength advantage over conventional self-play.
LR 0.5 remains the provisional experimental schedule because A4 favored it
against PPO-4. The strongest A5 endpoint by direct matchup is not established;
Track C should use the historical endpoint only as an experimental reference,
with the control endpoint as the simpler baseline.
