# A10-P1 storage budget report

The relevant filesystem has its capacity measured in the companion JSON. At capture time it had approximately 127.49 GB decimal free (118.75 GiB). The PokeAgent-A worktree occupied approximately 7.04 GB (6.56 GiB), including `runs` 4.45 GB, `data` 176 MB, `target` 1.54 GB, and the restored local Rust toolchain/cache `.tools` 664 MB. No project quota is enforced by the filesystem; the user's 50 GB is treated as a configured PokeAgent budget, not as total machine capacity.

A reusable generation store now writes both DDP rank files into a unique directory, fsyncs files and directories, verifies hashes and schema metadata, atomically publishes a `LATEST` pointer, and retains three complete generations. Incomplete generations are ignored and removable only through an explicit cleanup call. Corrupt latest data falls back to the newest older verified generation. Protected archive and active-evaluation pins are recorded in an explicit registry.

`check_storage_budget.py` reports available capacity, project-owned usage, checkpoint usage, projected peak (temporary generation plus final generation), and refuses writes when the 15 GiB reserve would be crossed. It never deletes the last valid generation or ambiguous project files. The existing PPO mathematical contract is untouched; the launcher has not been changed to start training.

A bounded synthetic two-rank 16 MiB publication took 0.973 s (17.25 decimal MB/s on this filesystem). This measures the storage path only, not a production learner checkpoint size or PPO throughput.

The initial P1 measurement recorded 162.42 GB decimal free before toolchain restoration. The post-validation measurement is 124.99 GB decimal free. The difference is reported as an observed filesystem change and is not attributed solely to this worktree; the measured PokeAgent-owned usage is 7.63 GB at the later capture.

The final post-commit measurement is 122.80 GB decimal free (114.37 GiB).
