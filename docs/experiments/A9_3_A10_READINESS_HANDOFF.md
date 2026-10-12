# A9.3 to A10 handoff

A10 may use Mode D plus `--precision-stability fp16-reduction-off` as the
numerical baseline. The gate remains fail-closed at `1e-3`; no tolerance
relaxation is permitted. Preserve the D1 flag in experiment metadata and
require `pa3-precision-D-v1` when loading frozen checkpoints.

Before production-scale training, run one independent-seed 25-iteration
confirmation and a fixed-panel evaluation against the archived A5/A9 anchors.
The current A9.3 result is exploratory and establishes stable throughput, not
statistical policy superiority or a production hyperparameter decision.
