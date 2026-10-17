# Behaviour diagnostics (trace sample)

* Traces captured: **30** (target 10 wins, up to 10 losses, remainder unusual/longest).
* Trace replays matching the frozen outcome and round count: **30/30**.
* Metrics below are computed from this trace sample and from the full aggregate separately; they are not a full-population action census.

## Observed action classes (selected levels)

* `move`: 854 (72.0%)
* `pass`: 127 (10.7%)
* `pick`: 120 (10.1%)
* `switch`: 51 (4.3%)
* `move+mega`: 23 (1.9%)
* `move-no-slot`: 11 (0.9%)

## Most selected moves

* Protect: 119
* Flare Blitz: 46
* Close Combat: 30
* Stockpile: 30
* Minimize: 27
* Yawn: 25
* Soak: 25
* Hyper Voice: 24
* Sucker Punch: 23
* Tailwind: 19
* Fake Out: 19
* Earth Power: 19
* Shadow Ball: 18
* Fissure: 17
* Encore: 16

## Targeting and resources

* Policy decisions inspected: 582.
* Move selections: 877; switch selections: 51; target locations: 0:376, 1:177, 2:324.
* Mega candidates offered: 176; Mega selected: 23.
* Protect-family selections: 123.
* Consecutive identical policy decision tuples: 64.

## Notable observations

* The trace set is intentionally enriched with the longest remaining games, so Protect/Stockpile/Minimize rates are not population estimates.
* Long games (40+ rounds) show repeated Protect, Stockpile or Minimize sequences and `move-no-slot`/pass cycles; this is the main observed gameplay concern.
* No illegal action or operational error appeared in the trace replays.

## Full-suite duration

* `test_a`: rounds median 9.0, p90 13, p99 19, max 60; histogram {'1-5': 122, '11-20': 2746, '21-40': 44, '41-80': 4, '6-10': 5276}.
* `test_b`: rounds median 8.0, p90 12, p99 18, max 28; histogram {'1-5': 69, '11-20': 308, '21-40': 11, '6-10': 1660}.
* `control_random_random`: rounds median 14.0, p90 21, p99 29, max 60; histogram {'11-20': 382, '21-40': 52, '41-80': 2, '6-10': 76}.
* `control_lexicographic_random`: rounds median 11.0, p90 20, p99 35, max 59; histogram {'1-5': 2, '11-20': 254, '21-40': 45, '41-80': 3, '6-10': 208}.

## Limitations

* Only the 30 selected games carry decision-level records; all other games retain compact receipts.
* Legal candidate sets are branch-level tuples, matching the evaluator contract; they are not a uniform joint-action distribution.
* Subjective tactical quality is not inferred beyond the recorded actions and outcomes.
