# A10-P2 exact-pool engine validation

## Result

The exact pool remains **NOT VERIFIED** for a pool-scoped PASS, although the bounded natural probes completed cleanly. The manifest is unchanged at 1,220 teams (1,137 predecessor plus 83 additions), with the expected manifest SHA in the JSON receipt.

Project-local Rust/Cargo 1.90 remained available. Release compilation of `pool_run_report` succeeded. Against a temporary simulator-ready copy of the proposed 1,220-team data, the deterministic pool probe instantiated and completed 1,220/1,220 games with zero blockers, unsupported errors, operational errors, or truncations. This probe uses each team against itself and is evidence of initialization and one natural trajectory only.

A targeted probe covered all 83 new teams against two varied opponents, one seed, both seat assignments: 332/332 games completed with zero unsupported, truncated, or engine errors. Static inventory coverage found 194 species, 346 moves, 139 abilities, and 137 items in the native catalogue. The conservative closure derived 346 moves, 233 abilities, 137 items, and 267 species/forms across six dynamic rules (call, transform, copy/swap, transfer, suppression, and item-end semantics).

Existing differential suites passed: 16 contract tests, 2 fixture-coverage tests, legality, ability-contact, ability-interactions, move hooks, and 3 Poltergeist regressions. These receipts do not prove every closure interaction for the exact pool. Resource-form executable verification, exhaustive pool action/request coverage, and a fully varied all-team natural campaign remain unverified; no engine source was changed.
