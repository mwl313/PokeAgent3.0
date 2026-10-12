# A9.4.1 Finalization Report

## Canonical configuration

`configs/validated_a94_runtime.yaml` is the versioned A9.4 profile. The
launcher consumes it through `--profile`; `--print-effective-config` and
`scripts/resolve_training_config.py` emit the same resolved values and semantic
hash. The profile pins PPO-2, LR scale 0.5 over the committed-match warmup and
cosine schedule, global minibatch 4096, microbatch 1024, 1024 environments,
16 workers, current/current self-play, historical probability 0, and D1.

The v3 manifest remains unchanged: 1,137 teams and the recorded manifest hash.
The broad `configs/train.yaml` is retained as historical design/source inventory
and now points active execution to the validated profile.

## Evaluation repair and evidence

All frozen loaders now restore precision mode and stability, apply the process
global D1 backend setting, and prequantize continuous observations per policy.
The evaluation identity and archive include precision metadata. A fresh 2,048
game panel at batch size 128 completed at 93.63 games/s with 1,042/1,006/0
W/L/D, 0 illegal actions, and 0 operational errors; paired 95% CI was
[0.4907, 0.5269].

## Validation and remaining blocker

The independent PPO arm passed 25/25 gates at 91.41 all-in and 92.67 steady
games/s. Seventeen relevant tests passed, archive verification passed for six
historical policies, and scheduler persistence/restart coverage passed.

The authoritative Rust readiness command could not run because `cargo` is not
available on this host. Therefore full regulation-wide engine readiness remains
unverified and blocks A10 full-production authorization. No new teams were
imported and A10 training was not started.
