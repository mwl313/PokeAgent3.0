# A10 Evaluation League design

The A10 league is an independent sidecar. It reuses `agent.evaluation.fixed_policy`, `batched_policy`, `frozen_policy`, `league`, `policy_archive`, `identity`, and the durable schedule semantics, while adding an A10 generation adapter and inference-only archive. It never imports the PPO learner into the health watchdog and never changes training opponents (`historical_probability=0`).

The schedule is anchored at zero with light milestones every 50,000 matches and full milestones every 200,000. Since recovery checkpoints are approximately 100,000 matches, unavailable boundaries remain delayed bookkeeping records with the actual checkpoint clock preserved.

The sidecar discovers complete generations, verifies both rank files and manifest identity, extracts a content-addressed model-only snapshot, registers it in `a10_registry.json`, and atomically queues jobs. It uses an exclusive lock and bounded state. Queue-only mode is the deployed default; actual evaluation is opt-in and CPU-only. No V100 is selected by the service.
