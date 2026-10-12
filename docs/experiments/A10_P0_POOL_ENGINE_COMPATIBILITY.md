# A10-P0 pool/engine compatibility

Status: **NOT VERIFIED** for the exact 1,220-team proposed pool.

The proposed pool was normalized against the pinned Showdown reference (`14546894d86f9589ac11130c510bbe73b6968665`) and has complete six-member records for every accepted team. Static inventory and manifest generation completed; predecessor preservation is PASS (1,137/1,137 unchanged), and source-level legality is PASS for the 85 accepted new candidates.

The authoritative Rust toolchain (`cargo`, `rustc`, and `rustup`) is unavailable on this machine. Therefore the following gates were not claimed: pool-scoped Rust handler coverage, reachable dependency closure execution, native action/request differential fixtures, and whole-pool natural completion/no-error validation. Existing v3 engine artifacts were not overwritten and cannot certify the added 83 teams. A10 must remain blocked until those checks run with the pinned engine/toolchain.
