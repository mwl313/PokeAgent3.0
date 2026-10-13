# P0 — source reconciliation and historical preservation

Integration worktree: `/home/minidc/Desktop/PokeAgent3.0-V2`, branch
`codex/pa3-v2-correctness`, based on `266bb65c3cb47a7e44512f9049338efdd35333cf`.
The supplied package was extracted under `package/`; every supplied SHA256SUMS
entry verified. Its documents remain unmodified. Updated execution ledgers
live beside the package, not inside it.

## Source reconciliation

The audited baseline is `d10ac692009566686fd32285c4456309a5f383a7`.
The actual last legacy checkpoint records source
`266bb65c3cb47a7e44512f9049338efdd35333cf`, with a recorded migration from
`0b5d930743298720e073f65264464f0d953ca7d3`. This preserves the bounded recompute
secondary checks, strict source migration and production PPO telemetry.
`8997cb4` is an ancestor; the Poltergeist fix is inherited. No existing engine
code was replaced by the remote audit revision.

The independent V1 evaluation worktree is at
`3c5f252338038525c762c8cde084ff50a4192169`, incorporating `3df5abd`, the GPU
round-robin runner and both round-robin reports. It is not the PPO launcher
source. It remains separate and untouched. All other worktree HEADs, dirty
status, diffs, remotes, actual systemd commands and environment are captured
in `P0_INVENTORY.json`.

## Historical run preservation

PPO was already inactive; this task did not signal or restart it. The active
independent Evaluation League was not stopped or modified.

Historical run and both round-robin directories were independently copied to
`/home/minidc/Desktop/pa3-legacy-evidence-20261013/` (approximately 2.1 GiB).
All copied checkpoint-generation rank files/manifests and frozen evaluation
policy files were SHA256-compared with their originals. This is a same-disk
copy, not an off-machine backup. Mutable league queue files are point-in-time
snapshots; the live queue continues on its original path.

Last durable legacy restore generation:
`/home/minidc/Desktop/PokeAgent3.0-A/runs/a10_production_20261013_v2/rolling/generations/iter-000001400832-1791883571874674126`.
It contains both ranks, model/Adam/scaler/scheduler/RNG/accounting data.
V1 models and optimizer state are historical evidence and prohibited as V2
initialization. Nothing was deleted or relabelled as V2.

## Exact dataset and native binary

Actual file:
`/home/minidc/Desktop/PokeAgent3.0-A/engine/data/a10-training-teams.json`.
SHA256: `ecabd47a8d000ba8dd9f3010c09992505fe21e95efe244e2c8a3d75ac3f70b1f`.

Manifest:
`/home/minidc/Desktop/PokeAgent3.0-A/data/teams/a10-p0-proposed-20261013/manifest.json`.
SHA256: `1aa64507e17f4a1f819d59834dbd314e96a1a1b6d13f94a791bd3f51fca309bb`.

Two independently instantiated native loaders both returned 1,220 teams from
identical bytes and array index order. These are loader witnesses, not a new
two-rank DDP run. The old launch arguments explicitly selected this file and
manifest; the stale 1,137-team default does not establish that old production
used the wrong dataset.

Executed legacy extension SHA256:
`105cb6140a205279948ba741f40951219dbe0c8b2b21c7ba9d826df8ea12e8a3`.
The new Rust source was compiled only in the isolated worktree; the deployed
legacy Python extension was not overwritten.

Inventory command:
`/home/minidc/Desktop/PokeAgent3.0/PokeAgent3.0-agent1-pa3/.venv/bin/python scripts/v2_source_inventory.py --output docs/audit_v2/P0_INVENTORY.json`.

P0 inventory/isolation/archive work is complete. V2 dataset profile and rank
identity are still later release requirements, not implied by this report.
