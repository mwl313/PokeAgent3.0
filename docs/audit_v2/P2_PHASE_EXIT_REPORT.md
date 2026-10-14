# P2 — Observation V2 gate PASS; production still blocked by P3/P4

Source `9c03ee63e18e47ef2a707e5c1722af7d3b1520f5`, isolated V2 worktree.
Evidence, source/binary hashes and commands: `p2/RECEIPT.json`.
The final semantic contract is
`dd723dc687a60b0568302fa0b09cfb110cbe67eac2c7022f2b4ffe4dff373e35`.
Earlier preliminary hashes were refined before any V2 model/run release;
metadata enumerates fixed fields, namespaces, ranges and public reveal-order
semantics. Old checkpoints and the preserved incomplete stash remain historical.

All five ragged groups now survive native wire, typed adaptation, select/narrow/
cat, inline and columnar rollout storage, precision preparation and neural
embedding. CSR rows are dynamic; wire count overflow is rejected, not clipped.
Each family contributes typed categorical and numeric row features, ordered
segment sums and counts. Empty families remain exactly zero but graph-connected
for DDP. Compact content hashing handles ragged contents rather than object
pointers; stores own mutable row data. The default model now has 9,889,459
parameters, so old ~8M performance numbers are not a new throughput measurement.

Known numeric/boolean masks enter the network explicitly. Known categorical
zero is distinct from unknown. Invalid namespace IDs, nonfinite known values,
reserved known fields, wrong versions/hashes and malformed CSR are rejected.
Opponent fixed move entries retain public reveal order with unknown physical
Slot; the uncapped repertoire carries all public reveals. Own physical move
slots remain known. No private opponent move-slot identity is fabricated.

Event changes use signed asinh normalization and weight uses hectograms/10000.
The 50-feature CSV audits domain endpoints, inverse mappings and actual ranges
from every frozen team's preview observation. There were zero nonzero FP16
underflows and no distinct-value collisions in those preview values. Targeted
tests verify signed ±1/±6 and small weights. Larger mathematical i32/u16 domains
are not claimed lossless in FP16; preview ranges are not full dynamic ranges.
Ragged integer IDs are never half-quantized; their numeric fields and known
masks follow the declared V2 contract. Battle physics is unchanged.

Wire version 2 includes the semantic SHA256. Typed/compact adapters, model
config, learner checkpoints and frozen evaluators reject missing/old/mismatched
identities. Architecture and precision checks run before loading learner state.
Source/dataset checks in the existing envelope are retained. No legacy weights
were loaded into a V2 model, and no production training was started.

Verification: 44 Python tests (including bounded V100 Mode D nonempty-ragged
forward/backward), 30 relevant Rust tests, and native bridge layout, packed
parity, deterministic replay and snapshot checks all pass. Controlled family
perturbations change neural embeddings and produce gradients; hidden unknown
payload changes do not. Bounded current-self-play collection tests complete
natural games with both-side rewards. Streaming/recompute tests preserve the
existing objective and tolerance. This is not the full P4 semantic/DDP gate.

P3 remains mandatory: signed target decoding and active-to-roster mapping are
still incorrect in the inherited scorer/collector and must be fixed before
interpreting actual learning results. P4 must then prove two-rank gradients,
telemetry, generation-bound per-team statistics and safe resumption. Only after
those gates may P5 initialize a fresh campaign. Main, frozen teams and old
historical runs remain unchanged; old evaluation remains retired as authorized.
