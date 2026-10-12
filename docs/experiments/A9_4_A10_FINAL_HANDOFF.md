# A9.4 to A10 handoff

The production candidate is Mode D with the explicit D1 backend contract:
`--precision-mode D --precision-stability fp16-reduction-off`. Resume now
rejects mismatched or unknown stability settings and only permits missing
metadata through the trusted, explicit A5/A9 migration allowlist.

Frozen-policy loading restores the precision mode and stability identity and
prequantizes continuous observations per policy before encoding. Archive
verification includes precision metadata, preventing result reuse across
execution contracts.

A10 remains unstarted. Before the A10 dataset launch, complete the independent
25-iteration receipt and a common-fixture 2,048-game evaluation panel at the
best correctness-passing batch size. Preserve separate experiment-scoped
schedule state, archive namespace, and fixture manifest.
