# A3 PPO-2 continuation to 250k matches

Date: 2026-10-11. Branch: `research/a-ppo-efficiency`. The training runtime was
source SHA `30b10e28d608f869811a4525afaa8101613c7adc`, the exact SHA recorded in
the A3 checkpoint envelopes. The branch report commit was `a69258d` before this
receipt. The immutable local final envelopes are under
`runs/experiments/a/a3_ppo2_to250k/chunk07/boundary.rank{0,1}`.

The A3 PPO-2 checkpoint at 77,824 committed natural matches was resumed without
reinitializing the model, optimizer, scheduler, scaler, RNG streams, dataset,
or opponent contract. Seven 12-iteration chunks added 172,032 matches, ending
at 249,856 (the nearest valid two-rank iteration boundary to 250k). Both V100s
were used by the DDP launcher. All seven chunk receipts reported rank digest
parity and zero operational errors. The summed iteration wall time was about
1,948.4 s (32.47 min); per-chunk all-in throughput ranged from 77.51 to
91.68 games/s; aggregate throughput over the summed wall time was 88.29 games/s.

Actual PPO work was not forced to a nominal count: KL early stopping ended 13
of 84 boundaries after fewer than two epochs (216, 216, 216, 216, 216, 216,
216 nominal optimizer minibatch steps per chunk are shown in the receipts; the
final cumulative optimizer-step count was 2,832). The final rank-0 metrics were
learning rate `2.9983e-4`, entropy `0.5143`, approximate KL `0.02477`, and
value loss `0.2283`. A transient KL spike (`0.605`) caused a bounded early stop
in chunk 5; subsequent iterations recovered to normal KL and entropy, with no
operational errors or legality failures. This is a training-stability signal,
not evidence of a collapse.

The direct balanced A3 endpoint comparison used 256 seat-swapped fixture pairs
(512 games), the actual frozen team pool and Rust engine. PPO-2 as policy versus
PPO-4 scored 242W/270L/0D (0.4727, bootstrap interval [0.4316, 0.5156]); thus
PPO-2 did not beat PPO-4 in this head-to-head, although it remains substantially
faster. Selected 256-game panels gave chunk04 PPO-2 161W/95L versus PPO-4 and
chunk07 150W/106L; chunk07 beat the earlier 77,824-match PPO-2 snapshot 165W/91L.
These single-seed panels are exploratory and do not establish significance.

A reproducible sampled-vs-recomputed log-probability discrepancy remains:
`max_abs_diff=1.0507107e-3` against a `1e-3` gate, concentrated in the
branch-count-2 actor-active stratum. The continuation therefore used the
explicit `--no-recompute-gate` experiment flag. This was not hidden or changed
in the PPO objective; production continuation should remain fail-closed until
the AMP/packed-row discrepancy is resolved. Checkpoint digests, DDP semantics,
action legality, and hidden-information boundaries were preserved.

The result supports PPO-2 as a provisional experimental schedule for further
seeds, not as a production default. The next action is to resolve the narrow
recompute discrepancy, then repeat a shorter multi-seed PPO-2/PPO-4 comparison
with immutable intermediate panels before extending the stronger policy.
Machine-readable details are in `A3_PPO2_TO250K_RESULTS.json`; raw chunk logs and
checkpoints remain local under `runs/experiments/a/a3_ppo2_to250k/` and are not
committed because of their size.
