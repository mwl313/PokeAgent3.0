# Fixed-policy evaluation harness design

Before any 4-vs-2 PPO comparison, run both branches against the same frozen
evaluation panel. The harness must load a policy checkpoint without updating
weights, use the pinned native reference engine, and count one complete battle
as one match.

Each case is identified by a stable `(team_a, team_b, seed, seat_assignment)`
record. Use a fixed hash of the selected training-team panel, mirror each case
with swapped seats, and keep the logical seed list identical across branches.
The first bounded panel should include 256 games per anchor/condition, then
expand only if the clustered confidence interval cannot resolve the practical
margin.

Required anchors are: a frozen random-initial policy, a deterministic legal
heuristic that uses only public information, and a frozen independent policy
snapshot. Self-play 50% is not a strength result. Record wins, losses, draws,
engine aborts, illegal actions, failed actions, and operational errors by team
group and seat.

The result record must include source/checkpoint SHA, panel hash, seed list,
matches, full wall seconds, policy decisions, value calibration, entropy,
approximate KL, clip fraction, and action-branch coverage. Use paired
per-case outcomes and seed-clustered bootstrap intervals; report inconclusive
when the interval crosses the practical margin. Evaluation time is included in
the matched wall-time comparison but evaluation matches are excluded from
training counters.
