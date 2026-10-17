# Random baseline evaluation v1

## Headline

* Test A (random opponent): **W 8099 / L 93 / D 0**, score **0.9886** (paired 95% CI 0.9863–0.9910).
* Test B (lexicographic legal): **W 2046 / L 2 / D 0**, score **0.9990** (paired 95% CI 0.9976–1.0000).

## Controls

* `control_random_random`: W 243 / L 269 / D 0, score 0.4746 (paired 95% CI 0.4355–0.5137).
* `control_lexicographic_random`: W 49 / L 463 / D 0, score 0.0957 (paired 95% CI 0.0723–0.1211).

The lexicographic-vs-random control is only 0.0957, so lexicographic legal is a much weaker baseline than random. Test B is therefore a sanity check, not evidence of strategy beyond beating random.

## Artifacts

* `performance.csv` — per-test totals, paired CI, hard gates, throughput.
* `seat_results.csv` — seat-specific results.
* `team_results.csv` — policy-team and opponent-team breakdowns.
* `behavior_report.md` — trace sample and observed action patterns.
* `performance_benchmark.md` — worker-count calibration.
* `readiness_assessment.md` — evidence-based recommendation.
* `manifest.json` — all hashes, commands, host and fixture identity.
* `traces/` — 30 representative decision-level traces.
