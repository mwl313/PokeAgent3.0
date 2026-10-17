# Random baseline evaluation v1

## Identity

* Candidate: `v2-000002764800-da35700f` (2,764,800 matches, iteration 675).
* Checkpoint SHA-256: `434be431de0e25a9c7730a7a0f38eda66d5db30397835983ebde9f4b8f1d9b36`.
* Candidate source SHA: `da35700f75c95833b0e8250d7b4b04e11320f8b7`.
* Evaluation worktree HEAD at measurement: `7b3b4cea23e92af17493f775e5161feb489215fd`.
* Frozen generation: `iter-000002764800-1792212738467610757`.
* Engine SHA-256: `8f3643a9f7b2febfa567f039b911d7af2f88bbf7bfc912849ada9e6a690d1c62` (read-only external build).
* Team pool: `a10-training-teams.json` (1,220 teams), SHA-256 `ecabd47a8d000ba8dd9f3010c09992505fe21e95efe244e2c8a3d75ac3f70b1f`.
* Dataset manifest SHA-256: `1aa64507e17f4a1f819d59834dbd314e96a1a1b6d13f94a791bd3f51fca309bb`.
* Evaluator: `pa3-native-batched-evaluator-v3`; sampling `fixture-seat-decision-branch-keyed-valid-cdf-v3`.
* Seed domain: `pa3-random-baseline-v1`.
* Fixture manifest hash: `5fe97fe15be9874f2461ac89516957ece515761d9cf770732fb70333eaacd84e`.
* Execution host: miniDC, 2x Xeon E5-2673 v4, CPU-only, CUDA hidden.

## Tests

| test | games | opponent | seed domain |
|---|---:|---|---|
| Test A | 8,192 | branch-wise uniform legal random | `pa3-random-baseline-v1:test-a` |
| Test B | 2,048 | lexicographic legal | `pa3-random-baseline-v1:test-b` |
| Control RR | 512 | random | `pa3-random-baseline-v1:control-random-random` |
| Control LR | 512 | random | `pa3-random-baseline-v1:control-lexicographic-random` |

Fixtures are generated once, frozen, and sharded by complete seat-swapped pair. Every global fixture ID is
stable. No `(team_a, team_b, seed)` identity is duplicated within a suite.

## Reproduction

```bash
/home/minidc/Desktop/PokeAgent3.0/PokeAgent3.0-agent1-pa3/.venv/bin/python scripts/random_baseline_eval.py --run-dir /home/minidc/eval-league/random-baseline/v1 all
```

Worker configuration: **w20** (highest steady-state valid games/s with zero errors and zero illegal actions).

## Statistical interpretation

* Draws count 0.5.
* The 95% confidence interval is a paired bootstrap over fixture pairs (both seats), not over individual games.
* Opponents are weak baselines. A high score here does not establish competitive ladder strength.
* The fixture pool is the frozen training-team pool, so team-composition generalisation is limited to it.
