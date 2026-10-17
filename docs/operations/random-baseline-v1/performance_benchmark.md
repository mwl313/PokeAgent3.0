# Throughput calibration

Calibration used a disjoint fixture set (512 games) and real frozen-model CPU inference.
Steady-state excludes process start and model loading; wall-inclusive includes them.

| config | workers | engine w | batch | games | steady games/s | wall games/s | CPU avg % | RAM avg GiB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| w08 | 8 | 1 | 32 | 512 | 29.12 | 25.41 | 10.3 | 9.14 | complete |
| w12 | 12 | 1 | 32 | 512 | 40.80 | 25.41 | 11.4 | 9.34 | complete |
| w16 | 16 | 1 | 32 | 512 | 49.15 | 33.83 | 16.8 | 10.77 | complete |
| w16-b64 | 16 | 1 | 64 | 512 | 49.59 | 33.83 | 16.8 | 10.81 | complete |
| w16-ew2 | 16 | 2 | 32 | 512 | 49.52 | 33.82 | 16.8 | 10.81 | complete |
| w18 | 18 | 1 | 32 | 512 | 56.17 | 33.81 | 17.9 | 10.91 | complete |
| w20 | 20 | 1 | 32 | 512 | 57.51 | 33.80 | 18.8 | 11.05 | complete |

Selected configuration: **w20** (highest steady-state valid games/s with zero errors and zero illegal actions).

## Full evaluation runs

| test | games | wall s | steady games/s | wall games/s | CPU avg % | CPU eq. threads | RAM avg GiB | RAM max GiB | load1 max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test_a | 8192 | 131.1 | 65.35 | 62.47 | 24.7 | 19.77 | 13.73 | 14.58 | 20.20 |
| test_b | 2048 | 35.3 | 74.66 | 58.02 | 21.2 | 16.93 | 12.45 | 14.44 | 20.05 |
| control_random_random | 512 | 5.1 | 1195.22 | 101.33 | 16.7 | 13.35 | 9.50 | 12.34 | 18.58 |
| control_lexicographic_random | 512 | 5.1 | 1037.54 | 101.34 | 16.8 | 13.41 | 9.46 | 12.34 | 20.13 |

CPU percentages are machine-wide (80 logical threads). The 20-way configuration used 20 distinct physical cores, balanced 10/10 across both NUMA nodes.
