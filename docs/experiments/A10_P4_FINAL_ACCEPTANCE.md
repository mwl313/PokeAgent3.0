# A10-P4 final acceptance

The frozen dataset is verified and the pool-scoped engine audit passes with bounded dynamic-universe evidence. Global Regulation M-C coverage remains incomplete by design and is recorded separately.

Checkpoint scheduling and recovery were corrected: production interval is 100,000 committed matches, rank envelopes are not written every PPO iteration, generation resume is available through `--resume-latest-from`, three-generation retention remains active, and storage-capacity failures are fail-closed.

The mandatory exact-configuration two-V100 continuity test did **not** pass. The uninterrupted and interrupted/resumed arms reached the same 4,096 committed matches, 56 optimizer steps, LR clock, recompute gate, and per-arm DDP rank parity, but final model and optimizer digests differed (`arm A` model `2b2123…`, `arm B` model `4dbc69…`). This is classified `NOT_REPRODUCIBLE`, not equivalent. A10 production launch is therefore blocked. No 100-million-game process was started.
