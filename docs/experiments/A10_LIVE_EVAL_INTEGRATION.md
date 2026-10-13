# A10 live evaluation integration

Status: bounded and non-disruptive. Production source `0b5d930`, dataset `a10-p0-proposed-20261013` (manifest `1aa64507…fca309bb`). The active PPO launcher was not stopped, signaled, reconfigured, or reloaded.

The A10 adapter validates only complete `rolling/generations/<generation>` directories: `LATEST` is resolved, both rank files and manifest byte sizes are checked, world size/clock/source/dataset contracts are checked, and incomplete generations are rejected. The first verified generation is `iter-000000102400-1791864154353943584` at 102,400 matches, iteration 25.

A10 scheduling uses anchor 0, light interval 50,000, and full interval 200,000 in an isolated durable schedule. Missed 50k and 100k boundaries are represented as delayed jobs using the true 102,400-match checkpoint identity; no fictional 50k model is claimed. The scheduler runs independently through a user timer every 10 minutes and only archives complete checkpoints/queues jobs. It does not launch GPU evaluation.

A model-only immutable snapshot was archived under the production run's `evaluation/policies/` directory. It excludes optimizer state and retains source generation, true match clock, source SHA, dataset hash, and source checkpoint SHA. The scheduler uses a lock and atomic JSON writes.

The active trainer remains the sole GPU workload. Full 256-game evaluation is queued/deferred until a separately authorized CPU budget window; the CPU smoke path is proven below.
