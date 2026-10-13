# A10.2 evaluation operations

Canonical code: `/home/minidc/Desktop/PokeAgent3.0-A-evaluation-league`, branch `codex/a10-evaluation-league`. Runtime artifacts: `/home/minidc/Desktop/PokeAgent3.0-A/runs/a10_production_20261013_v2/evaluation/league_v2/`.

```bash
/home/minidc/Desktop/PokeAgent3.0/PokeAgent3.0-agent1-pa3/.venv/bin/python /home/minidc/Desktop/PokeAgent3.0-A-evaluation-league/scripts/a10_evaluation_status.py --json
systemctl --user status a10-evaluation-worker.service a10-evaluation.timer
journalctl --user -u a10-evaluation-worker.service -n 20 --no-pager
```

Discovery runs every ten minutes. The persistent worker automatically executes native CPU batched battles, extends close contests through 256/1,024/2,048 games, records receipts and promotes directly superior policies. Both services are independent of PPO and hide CUDA. Evaluation failures/restarts never signal PPO.

The canonical inference pointer is `league_v2/best_policy.json`; the old top-level `evaluation/best_policy.json` and old milestone `schedule.json` are historical and are not used by A10.2. Consumers can call `agent.evaluation.best_policy.BestPolicyHandle.refresh()` between requests. PPO continues from its own current learner regardless of which policy wins evaluation.

Inspect `jobs.json`, `active.json`, `worker_state.json`, `resource_state.json`, `registry.json`, `milestones.json`, `decisions/` and `receipts/` for exact clocks, opponents, resource throttling and outcome evidence. No exact 50k model is invented. Old inference archives with incomplete original hash evidence are explicitly labelled; new generations are checked against both manifest hashes and actual serialized model/Adam digests. Archive cleanup is restricted to the owned v2 inference directory with dependency protection and a 2 GiB budget.

If necessary, stop only evaluation with `systemctl --user stop a10-evaluation-worker.service`; this does not stop PPO. Resume evaluation with `systemctl --user start a10-evaluation-worker.service`. A saved chunk receipt is reused after a worker restart. No Codex session is required for normal operation.
