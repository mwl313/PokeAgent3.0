# A10-P2 adaptive storage integration

`CheckpointGenerationStore` is now wired into the actual DDP launcher through the opt-in `--rolling-checkpoint-root` flag. The launcher requires both existing experimental rank envelopes, then publishes them as one generation with dataset, precision, iteration, and committed-match metadata. Normal production behavior is unchanged unless the flag is supplied.

The storage budget now treats filesystem capacity as authoritative. A project quota is optional and defaults to unset; the expected 50 GB after the external installation is recorded as an operational expectation, not an invented lifetime directory limit. Preflight accounts for temporary plus final serialization and a configurable concurrent-writer allowance. The default 15 GiB reserve remains fail-closed.

The external installation shares the filesystem and is treated as protected. Its process identity was not inferred or modified. Free space was observed falling during the campaign; no unrelated path was cleaned or deleted. The final measured space and project-owned usage are in `A10_P2_STORAGE_VALIDATION.json`.

The storage and checkpoint test campaign passed 25 tests, including real `torch.save`/`torch.load` rank payload round trips, complete two-rank publication, generation rotation, corrupted-latest fallback, missing rank, interrupted generation cleanup, insufficient-space refusal, protected generations, and launcher flag parsing. No production optimizer update or checkpoint was created.
