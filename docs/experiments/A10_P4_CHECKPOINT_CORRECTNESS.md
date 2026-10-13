# A10-P4 checkpoint correctness

Production scheduling is now absolute-clock based: the first safe completed iteration at or after each 100,000 committed-match boundary is saved, including after resume. The scheduler catches up crossed boundaries without restarting at zero. Full envelopes are no longer serialized every 2,048-game iteration.

`--resume-latest-from ROOT` resolves the newest verified generation and falls back through `CheckpointGenerationStore.latest()` when the newest generation is corrupt. Rank files can be loaded directly from a generation directory; no manual copying is required. Publication still requires both drained rank envelopes, matching global clock/optimizer steps, provenance, and rank digests.

`StorageBudgetError` is propagated out of the polling loop; transient peer timing and incomplete local reads are retried, while unexpected serialization failures terminate the launcher after stopping workers. Existing storage tests cover capacity refusal, corruption fallback, three-generation retention, protected artifacts, and real Torch payload round-trip. The full production long-run protocol was not started because the interrupted/resumed parity gate failed below.
