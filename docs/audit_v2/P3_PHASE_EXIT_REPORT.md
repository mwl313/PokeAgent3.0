# P3 Action V2 exit report

Source `f3060f4368eb0ec98c1a4fbabc008d82ae60258c`; evidence `p3/RECEIPT.json`.

Confirmed and repaired: signed -2/-1 were aliased by scorer clamping; packed u8 targets were not decoded before neural scoring/storage; normal own_slot was treated as roster index. Both actor and learner could agree on these wrong features, so old recompute success was insufficient. All five native signed locations now map to separate IDs. Player-known own active-slot category supplies roster identity for MOVE/SWITCH/PASS; PICK uses destination. Missing/duplicate maps, malformed candidates/references and capacity overflow fail closed. Null actor is an explicit learned reference. Architecture hash changes via target_vocab=5/null_entity_reference=True, so earlier model configs cannot silently load.

Actual Rust battle fixture: leads (5,3), switches (2,4), snapshot ground truth and emitted public mapping agree; typed/packed ordered candidates, masks, actor/move refs and logits agree at 1e-6. Stored compact rows retain signed targets and references. Existing native pinned-reference legality and phase/snapshot suites pass. Optional direct target/destination neural projections remain deferred as specified by frozen contract.

Rolling partial refill previously replaced every policy assignment with the new subset. It now replaces only requested indices; default rolling remains off. One pre-existing reward test assumed learner always seat1 even though roles alternate; corrected test captures actual reset roles without consuming RNG. Production reward math unchanged. Updated an old buffer-schema assertion to include V2 identity/ragged fields.

Tests: 18 focused/integration, 37 observation/math/reference, 52 storage/sampling, 2 Rust integration passes (overlapping Python suites). See logs and exact commands. P4 real two-V100 updates, replay-safe per-team stats, evaluation padding and strict V2 league still block production. No PPO campaign started.
