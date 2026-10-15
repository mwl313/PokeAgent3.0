# Source of Truth validation

Proposed candidate configuration consistency: PASS (evidence/source_of_truth_validation.json). All1220 frozen teams and hashes match. Numerical PPO values are compared against executable PPOConfig; profile epochs2 is explicit, base dataclass4 unchanged. The Markdown embeds the canonical profile config hash; JSON facts must match training,precision,dataset,collector,evaluation sections exactly. Semantic contract file hash is pinned. Negative drift test rejects modified epoch facts.

This PASS does not authorize production, source migration, or canonical promotion. README and PROJECT_STATUS point at the candidate Source of Truth and preserve prior prose marked historical. The canonical correctness branch is deliberately unchanged until gates pass.
