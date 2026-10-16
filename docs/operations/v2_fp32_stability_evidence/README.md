# FP32 policy repair evidence

This directory contains compact copies of immutable local evidence. `COPIED_EVIDENCE_MANIFEST.json` records each original path and checksum. Large checkpoint tensors, original failure captures and traces remain local.

`benchmark_comparison.json` records three independently restored repetitions per FP32 arm, including all-in and complete-iteration rates. E_exact is selected. The two additional E_actor transfer changes were not promoted; the marginal gain was within observed variation. All arms retain all96 token slots, no observation trimming and the same PPO/data settings. A brief small-tensor CUDA ownership test overlapped the startup window of E_exact repeat3; the three-sample comparison is descriptive, with no significance claim. Learner update time improves consistently across the three exact repetitions.

The independent GPU probe uses an audited Mode-D implementation with independently changed encoder AMP arithmetic and matches full loss/gradient/Adam/model results. Probe Adam is diagnostic; the genuine PPO continuation restores complete original learner state. Legacy migration proofs remain in the prior evidence directory and in their original local paths.

The sustained run was interrupted by the Codex usage limit after its third completed iteration. Both third-boundary rank envelopes survived and were verified/published by the existing checkpoint publisher. The resumed run completes two further iterations, followed by a five-iteration continuation to finish the original ten-iteration target. Original post-barrier telemetry is retained; no final parent/rank JSON was produced for the interrupted segment. Verification derives only recorded metrics from that telemetry and does not invent missing communication or rollout-digest fields. Full saved learner, RNG, counters and collector state provide the checkpoint restore comparison.

The final report records the validation outcome and scope. All ten requested sustained E iterations are complete after the user's instruction to keep going. Diagnostic matches and isolated evaluation never advance production counters. Production remains stopped.

Work was paused by the user before preparing a new receipt for the final documentation snapshot. The committed E_exact proof and migration receipt pin tested executable9d3d605. They do not authorize a different Git HEAD. No training/evaluation processes remain active.
