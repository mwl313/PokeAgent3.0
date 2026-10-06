# Batch execution report: documented two-actor, 2,048-environment topology

2026-10-07, primary agent. Development evidence, not a readiness claim. The
run below covers the **currently ported native subset**; it does not measure
the full regulation, because unported mechanics still stop most training-pool
battles with explicit operational errors.

## Topology under test

`scripts/run_actor_pair.py` starts **two OS processes**, each owning one
`pa3_engine.NativeEngine` group of **1,024 environments / 16 Rayon workers**,
for 2,048 environments and 32 workers total. Each rank is bound with
`numactl --cpunodebind=N --membind=N` and the actor additionally applies
`sched_setaffinity` for its worker threads. Pokémon Showdown is never executed.

Startup validation cross-checks the live kernel topology against the documented
machine profile before any worker starts (`/sys/bus/pci/devices/*/numa_node`,
`/sys/devices/system/node/nodeN/cpulist`, `/proc/cpuinfo`, `nvidia-smi`):

```
hardware: numactl: /usr/bin/numactl
hardware: gpu0: pci 05:00.0 numa 0
hardware: gpu1: pci 84:00.0 numa 1
hardware: rank 0: gpu 05:00.0 (expected 05:00.0) pci-numa 0 rank-numa 0 workers 0-15
hardware: rank 1: gpu 84:00.0 (expected 84:00.0) pci-numa 1 rank-numa 1 workers 20-35
```

The runner aborts on any GPU bus / NUMA / CPU-map disagreement unless
`--allow-hardware-mismatch` is passed explicitly.

## Cohort and command

Throughput is measured on the differential-fixture cohort, which is provably
inside the ported subset: `scripts/export_fixture_cohort.py` extracts the
reference-validated fixture teams from `engine/data/turn-fixtures.json` (587
unique teams at the 2026-10-07 evening refresh). Random pairings of that cohort
can still include environments whose teams hold an *unported* mechanic that the
recorded fixture battle never selected; those stop as counted operational
errors and are excluded from completed games, exactly as the engine contract
requires.

```
python3 scripts/export_fixture_cohort.py
.venv/bin/python scripts/run_actor_pair.py --envs 1024 --games 4096 --workers 16 \
    --teams engine/benchmarks/fixture_teams.json \
    --report engine/benchmarks/batch_pair_2x1024_fixture_cohort.json
```

## Result: 2 × 1,024 environments, 16 workers per rank

| Metric | Rank 0 (NUMA0) | Rank 1 (NUMA1) |
| --- | ---: | ---: |
| Natural complete battles in one cohort | 1,024 | 1,024 |
| Steady-state battles/s (rank wall) | 359.5 | 373.6 |
| Engine transitions/s | 9,739 | 10,132 |
| Observation encoding (per-view payload) | 31.0 ms/round | 27.1 ms/round |
| `step_batch` | 2.34 ms/round | 2.22 ms/round |
| Peak RSS | 228.9 MB | 228.7 MB |
| Operational errors (excluded from games) | 0 | 0 |
| Combined end-to-end wall (incl. startup) | 3.16 s | |

Combined steady state is ≈ **733 natural battles/s** on this cohort; the
end-to-end figure including process startup and catalogue loading is
≈ 649 battles/s. These are measured on short fixture battles with a placeholder
policy; a real policy and the full regulation will change both numbers and must
be re-measured when coverage is complete.

## Worker scaling (2 × 1,024 envs, one 1,024-env cohort per rank)

| Workers/rank | Steady battles/s (rank 0) | Rank transitions/s | Observation ms/round | `step_batch` ms/round |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 460.3 | 12,479 | 39.7 | 5.42 |
| 2 | 515.5 | 13,973 | 34.8 | 4.53 |
| 4 | 589.2 | 15,972 | 30.4 | 2.77 |
| 8 | 633.1 | 17,162 | 28.7 | 2.07 |
| 16 | 645.4 | 17,496 | 28.0 | 2.19 |

Observation encoding dominates the round cost at every width (~28–40 ms/round
for ~1,000 views, i.e. ≈42 kB of packed tokens per decision). Scaling flattens
after 4–8 workers, which is the main measured headroom left in the bridge:
either the encoder or the batching/scheduling path, not the battle transition
itself (`step_batch` is ~2.2 ms/round for the whole 1,024-env cohort).

## Observation payload path A/B (same cohort, 2 × 1,024 envs, 16 workers)

`NativeEngine.observe_fixed_batch` returns one fixed-stride token buffer plus
one ragged buffer per batch. Python then builds zero-copy numpy views with
`pa3_engine.parse_batch` (≈4 ms/round for ~1,000 views) instead of copying every
view into dense arrays (≈26 ms/round). However, the producer side currently
costs more because the whole ~42 MB batch is copied into a single fresh
`bytes` object every round:

| Path | Steady battles/s (rank 0/1) | Observation | `step_batch` | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| per-view blobs (`observe_encoded_batch`) | 361.9 / 371.9 | 30.6 / 27.3 ms/round | 2.34 / 2.22 ms/round | 228.6 MB |
| fixed-stride batch (`observe_fixed_batch`) | 238.5 / 249.8 | 45.5 / 39.7 ms/round | 2.45 / 2.14 ms/round | 353 MB |

The actor therefore defaults to per-view payloads; the fixed-stride API remains
available for consumers that want the whole batch as one zero-copy view. Its
producer cost is a known optimization target (a retained Python buffer filled
in place would remove the per-round 42 MB allocation/copy).

## Limitations (explicit)

* Fixture cohort only. Training-pool teams still stop on unported moves,
  abilities, volatiles and interactions; `engine/examples/pool_run_report.rs`
  is the authoritative blocker list and it is not exhausted.
* The attached policy is the placeholder `first`/`random` Python picker, not a
  torch policy; GPU inference, observation tensor transfer and PPO row
  handling are not part of this measurement.
* `observation encoding` measures the native pack step plus the sparse batch
  call, not NumPy/Torch consumption of the blob.
* Operational errors are reported separately and never counted as draws, wins
  or completed games.
