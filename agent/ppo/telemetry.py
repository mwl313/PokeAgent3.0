"""Optional structured PPO telemetry for a future validated trainer resume.
This module is intentionally not imported by the active production launcher."""
from __future__ import annotations
import json, os, tempfile
from pathlib import Path

def append_iteration(path, *, iteration, committed_matches, rank, metrics):
    record={'schema':'pa3-ppo-telemetry-v1','iteration':int(iteration),'committed_matches':int(committed_matches),'rank':int(rank),'metrics':dict(metrics)}
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.telemetry-',dir=path.parent)
    with os.fdopen(fd,'w') as f: json.dump(record,f,sort_keys=True); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path); return record

def read_iteration(path):
    with open(path) as f: row=json.load(f)
    if row.get('schema')!='pa3-ppo-telemetry-v1': raise ValueError('unsupported telemetry schema')
    for key in ('iteration','committed_matches','rank','metrics'):
        if key not in row: raise ValueError(f'missing telemetry field: {key}')
    return row

def append_training_update(path, *, iteration, committed_matches, rank,
                           update, collector, iteration_matches,
                           iteration_wall_s, timing=None, recompute=None, verification=None):
    """Append one completed PPO iteration without changing learner state.

    This is called after the rank-digest and DDP iteration barrier succeed.
    The caller should treat telemetry I/O as best-effort: a disk/logging
    failure must never cause a resumed training campaign to fail.
    """
    import datetime
    import math

    if iteration < 1 or committed_matches < 0 or rank < 0:
        raise ValueError("invalid PPO iteration identity")
    if iteration_matches < 0 or iteration_wall_s <= 0:
        raise ValueError("invalid PPO iteration performance counters")

    data = update.as_dict() if hasattr(update, "as_dict") else dict(update)
    # The existing learner already computes all these scalars; no new model
    # forward, backward, CUDA sync, or distributed collective is required.
    keys = (
        "policy_loss", "value_loss", "entropy", "uniform_kl",
        "approx_kl", "epoch_approx_kl", "ratio_mean", "clip_fraction",
        "grad_norm", "grad_norm_max", "learning_rate",
        "optimizer_steps", "optimizer_steps_skipped", "epochs_run",
        "stopped_early", "scaler_scale", "rows", "actor_rows",
    )
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"learner report missing telemetry fields: {missing}")
    ppo = {key: data[key] for key in keys}
    record = {
        "schema": "pa3-ppo-training-metrics-v1",
        "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "identity": {
            "absolute_iteration": int(iteration),
            "committed_matches": int(committed_matches),
            "rank": int(rank),
        },
        "ppo": ppo,
        "collector": {
            key: collector.get(key) for key in (
                "games", "decisions", "learner_rows", "operational_errors",
                "reward_sum", "wins", "losses", "draws",
            ) if key in collector
        },
        "performance": {
            "committed_matches_this_iteration": int(iteration_matches),
            "iteration_wall_s": float(iteration_wall_s),
            "training_games_per_second": float(iteration_matches / iteration_wall_s),
            "timings": dict(timing or {}),
        },
        "safety": {
            "recompute": dict(recompute or {}),
            "rank_verification": dict(verification or {}),
            "skipped_optimizer_steps": int(data["optimizer_steps_skipped"]),
        },
        "semantics": {
            "policy_value_entropy_kl_ratio_clip": "already aggregated by learner over valid rows across DDP ranks",
            "gradient_norm": "rank-local final preclip L2 norm; grad_norm_max is rank-local maximum preclip norm",
            "entropy": "log(K)-normalized request entropy",
            "uniform_kl": "log(K)-normalized KL(U||policy), not old-vs-new policy KL",
            "approx_kl": "sampled PPO old-vs-new approximate KL",
        },
    }
    # Strict JSON rejects NaN and Inf instead of silently introducing bogus
    # points into a training dashboard. One append per rank per iteration.
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode("utf-8")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # fdopen owns fd after construction and closes it on context exit.
        raise
    return record
