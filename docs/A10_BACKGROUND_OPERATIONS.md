# A10 Background Operations

Status recorded 2026-10-13 for `a10-production-20261013-v2` on branch `research/a-ppo-efficiency` (source `0b5d930`). The validated production launcher is running from `/home/minidc/Desktop/PokeAgent3.0-A` with the A10 1,220-team manifest, PPO-2, Mode D, fp16 reduction off, and 100,000-match rolling checkpoint interval. The launcher command and manifest are preserved in `A10_P4_PRODUCTION_RUN_MANIFEST.json`.

## Current supervision

- `scripts/a10_status.py --json` is read-only and reports launcher/worker identity, log-derived committed matches, rank agreement, recompute gate, latest checkpoint generation, disk and GPU telemetry.
- `scripts/a10_watchdog.py --once` performs one low-overhead check. The watchdog uses `operations/watchdog.lock`, records deduplicated incidents in `operations/incidents.jsonl`, and persists state in `operations/watchdog_state.json`. It never imports PyTorch, mutates model state, or restarts training.
- The preferred scheduler is the per-user systemd timer `a10-watchdog.timer`, every 180 seconds. If systemd user services are unavailable, run `nohup .../scripts/a10_watchdog.py --interval 180` under the same singleton lock.
- The watchdog checks process identity (full command line, not a stale PID), exactly two workers, progress from rank logs, recompute failures, rank mismatch, operational errors, stale progress (warning 15 minutes / critical 20 minutes), duplicate launchers, and disk free space (warning below 25 GiB, critical below the 15 GiB reserve).
- The local `codex` CLI is installed and authenticated (`codex login status` reports ChatGPT), but no continuous Codex call is made. No user webhook, Slack, email, or other notification destination is configured. Incidents are durable local artifacts. `scripts/a10_incident_dispatch.py` can create a bounded local Codex request file; Codex is only considered for incident-only use when explicitly enabled and authenticated, never as a continuous monitor.

## Operations

```bash
scripts/a10_status.py
scripts/a10_status.py --json > /tmp/a10-status.json
scripts/a10_watchdog.py --once
scripts/a10_safe_stop.py   # writes a boundary-stop request; sends no signal
```

A safe stop is intentionally request-only because the current production launcher does not consume a stop sentinel. To stop, inspect the latest completed iteration and send an explicit operator signal, preserving the latest complete rolling checkpoint. Resume only from a verified `LATEST` generation with the matching source SHA, dataset manifest, action schema, model configuration, opponent manifest, world size, and checkpoint generation.

## Current limitations

The final `report.json` is written by the launcher at run completion, so during training the watchdog uses bounded incremental reads of rank logs. Checkpoint health is checked from the rolling `LATEST` pointer and manifest metadata; full checkpoint hashing is deliberately not performed every three minutes. Evaluation profiles remain configured separately and are reported as configured versus triggered; the watchdog does not launch evaluation or alter training cadence.

The Codex desktop session should exit after this setup. Monitoring is performed by the user-level timer or fallback process, not by a continuously running Codex conversation.
