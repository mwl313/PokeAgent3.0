# Runtime/profile comparison

Training hyperparameters are unchanged between original immutable runtime_profile.json and proposed versioned profile. Precision gains explicit recompute_secondary=true (original absent and checkpoint false): this is a **BLOCKED migration**, not a silent resume change. Dataset path strings become worktree-relative; bytes/hashes/count remain identical. The profile hash changes and must be recorded by a future release.

Actual launcher defaults are PPOConfig defaults with explicit profile epochs=2; raw PPOConfig default4 remains unchanged. Drift checker compares numerical objective/schedule/batch values to executable PPOConfig, data/team hashes/count, semantic contract file hash and SoT facts. Global4096/perrank2048/micro1024 implies two accumulation slices/rank. It is configuration consistency proof, not DDP execution proof.

Evaluation configuration50k/200k/1024pairs/batch128 is normative intended; original sidecar command256pairs/batch32 and checkpoint polling do not implement it. This remains a release deviation, not a parameter silently corrected in an old run.

Machine difference: ../operations/evidence/profile_diff.json. No active run changed.
