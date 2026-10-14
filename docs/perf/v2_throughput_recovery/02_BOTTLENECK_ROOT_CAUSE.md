# V2 root cause and measurement boundaries

Published first-five continuation snapshot: 31.35 games/s; collection/clock
wait66.514s, update49.537s, prepare12.003s, residual2.599s per4096 games.
Fresh read of all18 completed production iterations gives **33.4166 games/s**
(iteration time only). Last completed iteration: 73,728 total matches,
98.715s,41.493 games/s,84,464 learner rows,42 optimizer steps. Reduced rows
and work explain part of this increase without a code change.

Production failed before iteration19's PPO update at15:51:25 KST with
rank errors0.00115824/0.00170612. Existing saved-model replay exactly reproduced
them. Normal two-branch actor requests are affected. AMP is batch-shape
sensitive; FP32 on compared rows is much less sensitive. Original actor
groups were not recorded, so input mutation cannot be fully excluded. Do not
raise the1e-3 threshold, accept unexplained outliers, or change only gate
precision. This throughput work retains the numerical contract and therefore
cannot claim that the long-run recompute failure is repaired.

Independently measured CPU data-path probe uses the actual576 sampled V2
observations and4,233 prefix-conditioned candidates from the saved failure,
five passes in each source process. It compares representations plus a real
48-row full-model objective/gradient/Adam calculation with declared synthetic
advantage/return targets. JSON hashes and results accompany the final report.
This is not a self-play throughput benchmark and not a learning experiment.

Source-confirmed avoidable costs: candidate decode/validation/token mapping
repeated per request and again while recording; per-row fixed observation
copying/dictionaries; object→array materialization; ragged gather index creation
through one np.arange per selected row. New implementations change data
organization while retaining every model/observation/action semantic field.

The original first iteration collector profile had record30–33s/table14–15s
versus native battle step0.35s. These host timers include synchronization.
They do not justify claims about exact attention, backward or NCCL percentages.
The new CUDA profiler uses events, NVTX/CPU ranges, kernel interval unions and
runtime-sync events; its isolated workload and limitations are reported with
actual output, rather than extrapolated as end-to-end gains.

The benchmark runner restores the valid4,096-match learner state, with all
PPO/precision/collector settings preserved. It omits campaign team-statistics
writes in **both arms** to protect the path-bound production ledger. Production
team-stage first-iteration cost was about0.16s; experiment rates are not
labelled production-with-ledger rates. Startup and iteration-only rates are
reported separately. Any repetition overlapping tests/probes is excluded
from the controlled performance panel.

Missing per-iteration historical candidate counts, GPU peak and CPU RSS are
null in the recalculated telemetry artifact; they are not inferred from
another iteration. Full launcher reports supply those metrics for new runs.
