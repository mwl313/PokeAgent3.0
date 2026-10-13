# A10-P3 engine final acceptance

The frozen 1,220-team pool completed a varied native campaign of 19,520 natural games (two seeds, four opponents, both seats, 16 games/team): 19,520 completed, zero operational errors, truncations, or unsupported results. Existing targeted differential suites remain green (contracts 16, fixture coverage 2, legality 1, ability contact 1, ability interactions 2, move hooks 1, Poltergeist 3).

The readiness checker is still **NOT READY** for global launch: 486/515 regulation moves and 200/223 abilities are executable; dynamic closure has 11 blocked callers and incomplete move/ability universes. Static natural execution of all frozen teams passes, as do player-safe observations, action masks, deterministic replay, snapshot/restore, and PyO3 batch gates. This is a genuine coverage blocker, not a varied-campaign failure.
