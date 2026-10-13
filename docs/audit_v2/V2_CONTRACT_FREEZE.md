# Phase 0.5 — frozen V2 contract

Normative source: `configs/pa3_v2_contract.json`, canonical JSON SHA256
`8a0e225e3f657f1016b1dfb39d0e683425185bf14b5a6c2a540930da55f6db43`.
P1 prerequisite: `548902547a5d5235f532193a47a77a89abfb1cdd`.

The 96/88 token layout and fixed 32/50/40 feature shapes remain. Wire version
2 adds a 32-byte semantic digest after its u16 version. All five ragged families
use dynamic per-token CSR without truncation, survive compact/columnar storage,
and contribute family-specific row embeddings and counts before the encoder.
Numeric/boolean known masks enter the network explicitly. Known categorical
zero (including typeless type ID 0) is distinct from unknown.

Event change uses signed asinh normalization; weight uses hectograms / 10000.
Other numeric features require the complete P2 audit before acceptance. Mode D
remains encoder autocast with fixed-input FP16 roundtrip at both actor and
learner; ragged categorical IDs are never FP16-quantized. This is precision V2,
not an assertion that the old precision identity is compatible.

Action V2 preserves the six-field wire; target is signed [-2,-1,0,1,2] and maps
to target + 2. Normal actor identity comes from the actual active-to-roster map
(own category 22: active slot + 1, zero for inactive). PICK uses its destination
roster. Direct target/destination neural projections are deferred; they are
optional architecture changes rather than required correctness repairs.

V1 weights, compact observations and checkpoints are incompatible. P2/P3/P4
must prove the actual native→policy→buffer→learner path and strict checkpoint
identities. Freezing this file does not mark those implementation gates PASS
or authorize premature production activation.
