# A10 storage and checkpoint policy

Full recovery checkpoints are written at the next completed safe iteration boundary near every 100,000 committed natural games. A complete generation contains both rank files plus a verified manifest with learner, optimizer, scheduler, GradScaler, RNG, collector, dataset, configuration, and DDP identity. Three complete generations are retained by default. Model-only archives are evaluation artifacts and are never used for exact continuation.

Publication is generation-based: write a unique temporary directory, fsync both rank files, verify hashes/schema/clock/rank consistency, write and fsync the manifest, rename the generation, then atomically replace `LATEST`. Failures before publication leave the previous pointer usable. Cleanup runs only on explicit, dependency-aware paths and respects `protected_artifacts.json` pins. Active evaluation policies and A5/A9.4 anchors must be pinned.

The disk guard measures actual bytes before serialization, counts a temporary plus final generation peak, and requires a 15 GiB filesystem reserve. It refuses an unsafe write and permits cleanup only for abandoned incomplete generations or explicitly registered disposable artifacts. Evaluation metrics, fixture hashes, policy hashes, and failure reports are retained; full battle traces and redundant optimizer copies are not.

## Adaptive capacity and launcher integration

Filesystem free capacity is the primary guard. `project_quota_bytes` is optional and unset by default; an expected post-install free-space target is informational. Before a generation, the launcher measures free bytes, estimates temporary plus final generation size, adds the configured concurrent-writer allowance, and requires the minimum reserve. `--rolling-checkpoint-root` publishes the two experimental rank envelopes as one verified generation; the flag is opt-in and leaves normal production behavior unchanged.

## A10-P3 publication rule

When enabled with `--rolling-checkpoint-root`, the launcher parent polls the atomically replaced rank envelopes while workers remain active. A generation is published only when both ranks report the same iteration, committed-match clock, optimizer-step boundary, source/dataset/opponent provenance, complete collector drain, and identical model/optimizer digests. DDP communicators are recreated on restore. Three complete generations are retained; the filesystem reserve and concurrent-write allowance remain enforced.

## A10-P4 final scheduling rule

Production full envelopes are scheduled from the absolute global committed-match clock at the first safe iteration boundary after each 100,000-match threshold. The scheduler state is reconstructed from the saved clock on resume. `--resume-latest-from` selects the latest complete verified generation and falls back on corruption. Capacity refusals propagate as fatal errors; the launcher stops workers and preserves the last verified generation.

## A10-P5 continuity evidence

The 100,000-match scheduler and latest-generation resume path passed bounded regression tests. A full rank envelope is approximately 105,266,129 bytes; observed publication stage cost was 0.69–0.85 seconds per rank in bounded runs. The deterministic diagnostic continuation matched uninterrupted state exactly; normal Mode D variation is baseline CUDA/AMP numerical nondeterminism and is recorded separately.
