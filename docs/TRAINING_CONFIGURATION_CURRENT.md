# Current training configuration

The active machine-readable profile is [`configs/validated_a94_runtime.yaml`](../configs/validated_a94_runtime.yaml). Resolve it with `scripts/resolve_training_config.py`; the launcher accepts the same profile with `--profile` and records its hash.

| Category | Effective value | Meaning | Canonical source | Revalidation |
|---|---|---|---|---|
| PPO epochs | 2 | Full PPO update repeats | profile | yes |
| LR scale | 0.5 | Multiplies match-clock schedule | profile/scheduler | yes |
| LR | scheduled, peak 0.00015 effective | 250k warmup, cosine to 100M matches | profile + `agent/ppo/schedule.py` | yes |
| Batch | 4096 global; 2048/rank; 1024 micro | Two-rank accumulation | profile | yes |
| Self-play | current/current; historical 0% | Both seats produce rows | profile | yes |
| Precision | D + `fp16-reduction-off` | FP16-rounded inputs, AMP encoder, FP32 scorer/GRU | profile + precision contract | mandatory |
| Gate | enabled, tolerance 1e-3 | Fail-closed actor/learner parity | profile/launcher | mandatory |
| Dataset | `mb-mc-v3-userteam-all-train`, 1,137 teams | Uniform unique-team sampling; immutable v3 | profile/manifest | mandatory |
| Evaluation | 50k lightweight, 200k full; batched 128 | Frozen-policy only | profile/league | yes |

A9.4 evidence is 25 iterations/51,200 games at 91.41 all-in games/s on seed
20261010. This is measured evidence, not a guarantee for another dataset.
PPO-4, historical self-play, new teams, and 100M-match authorization remain
unselected decisions. A10 is pending and non-runnable.
