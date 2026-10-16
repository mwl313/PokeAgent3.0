"""Canonical pre-update sampled log-probability recompute gate."""

from __future__ import annotations

import torch


def recompute_check(learner, buffer, device, limit=1024, amp=False, per_stratum=96, audit=False,
                    secondary=False, evidence_path=None):
    """Keep malformed primary inputs fail-closed with a durable error receipt."""
    try:
        return _recompute_check_impl(learner,buffer,device,limit,amp,per_stratum,audit,
                                     secondary,evidence_path)
    except (ValueError, TypeError, IndexError, KeyError, RuntimeError) as error:
        result={"rows":0,"total_rollout_rows":len(buffer.rows),"within_gate":False,
                "primary_within_gate":False,"max_abs_diff":float("inf"),
                "measurement_available":False,"tolerance":1e-3 if amp else 1e-4,
                "reason":"malformed primary input or failed diagnostic",
                "error_type":type(error).__name__,"error":str(error)}
        if evidence_path:
            from agent.ppo.experimental_checkpoint import save_envelope
            observations=[]
            for row in buffer.rows[:8]:
                try:
                    observations.append(buffer.observation_store.get(row.observation_ref).to_compact_numpy())
                except (ValueError, TypeError, IndexError, KeyError, RuntimeError) as export_error:
                    observations.append({"export_error":str(export_error)})
            save_envelope(evidence_path,{"kind":"ppo_recompute_input_error_v1",
                "result":result,"source_file":__file__,"sample_rows":buffer.rows[:8],
                "sample_observations":observations,
                "actor_group_metadata":getattr(buffer,"actor_groups",[])[:8],
                "model_config":learner.model.config.to_dict()})
        return result


def _recompute_check_impl(learner, buffer, device, limit=1024, amp=False, per_stratum=96, audit=False,
                    secondary=False, evidence_path=None):
    """Stratified sampled/recomputed log-probability parity gate.

    A first-N-rows check cannot be extrapolated to the whole iteration, so rows
    are sampled per (request kind, branch count, actor active) stratum and every
    stratum's row count and max absolute difference is reported.
    """
    import time as _time
    started = _time.perf_counter()
    strata: dict[tuple, list] = {}
    for row in buffer.rows:
        key = (int(row.request_kind), int(row.branch_count), bool(row.actor_active))
        strata.setdefault(key, []).append(row)
    sampled = []
    for _key, group in sorted(strata.items()):
        if len(group) <= per_stratum:
            sampled.extend(group)
        else:
            step = len(group) / per_stratum
            sampled.extend(group[int(index * step)] for index in range(per_stratum))
    sampled = sampled[: max(limit, 1)]
    selection_wall = _time.perf_counter() - started
    started = _time.perf_counter()
    batch = buffer.to_batch(sampled, device=device)
    materialize_wall = _time.perf_counter() - started
    # Recompute under the same forward precision that sampled the rollout, as
    # the plan requires for mixed-precision parity.
    context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if amp and device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
    )
    started = _time.perf_counter()
    with torch.no_grad(), context:
        encoded = learner.model.encode(batch.observation)
        evaluation = learner.model.evaluate_encoded(encoded, batch.candidates, selected=batch.candidates.selected)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    batched_wall = _time.perf_counter() - started
    stored = torch.tensor([row.old_logprob for row in sampled], dtype=torch.float32, device=device)
    difference = (evaluation.request_logprob.float() - stored).abs()
    # Isolate batch-shape sensitivity: the same preserved row is evaluated
    # alone, without changing its observation, mask, or selected prefix.
    isolated = []
    isolated_logprobs = []
    isolated_wall = 0.0
    if audit:
        started = _time.perf_counter()
        with torch.no_grad():
            for row in sampled:
                one = buffer.to_batch([row], device=device)
                with context:
                    one_eval = learner.model.evaluate(one.observation, one.candidates,
                                                      selected=one.candidates.selected)[1]
                isolated.append(float(abs(one_eval.request_logprob.float().item() - row.old_logprob)))
                isolated_logprobs.append(float(one_eval.request_logprob.float().item()))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        isolated_wall = _time.perf_counter() - started
    per_row = difference.detach().cpu().tolist()
    stratum_report: dict[str, dict] = {}
    for index, row in enumerate(sampled):
        key = (f"kind{int(row.request_kind)}_branches{int(row.branch_count)}"
               f"_actor{int(bool(row.actor_active))}")
        entry = stratum_report.setdefault(key, {"rows": 0, "max_abs_diff": 0.0})
        entry["rows"] += 1
        entry["max_abs_diff"] = max(entry["max_abs_diff"], float(per_row[index]))
    grouped_errors = {}
    for index,row in enumerate(sampled):
        key=f"kind{int(row.request_kind)}_branches{int(row.branch_count)}_actor{int(bool(row.actor_active))}"
        grouped_errors.setdefault(key,[]).append(per_row[index])
    for key,values in grouped_errors.items():
        errors=torch.tensor(values,dtype=torch.float32)
        stratum_report[key]['mean_abs_diff']=float(errors.mean())
        stratum_report[key]['p99_abs_diff']=float(torch.quantile(errors,.99))
    tolerance = 1e-3 if amp else 1e-4
    result = {
        "rows": len(sampled),
        "total_rollout_rows": len(buffer.rows),
        "sampled_fraction": len(sampled) / max(len(buffer.rows), 1),
        "strata": stratum_report,
        "max_abs_diff": float(difference.max().item()) if len(sampled) else 0.0,
        "mean_abs_diff": float(difference.mean().item()) if len(sampled) else 0.0,
        "median_abs_diff": float(difference.median().item()) if len(sampled) else 0.0,
        "p95_abs_diff": float(torch.quantile(difference.float(), 0.95).item()) if len(sampled) else 0.0,
        "p99_abs_diff": float(torch.quantile(difference.float(), 0.99).item()) if len(sampled) else 0.0,
        "audit_enabled": bool(audit),
        "isolated_max_abs_diff": max(isolated) if isolated else None,
        "isolated_mean_abs_diff": sum(isolated) / len(isolated) if isolated else None,
        "timing_seconds": {
            "row_selection": selection_wall,
            "rollout_materialization": materialize_wall,
            "batched_recompute": batched_wall,
            "isolated_audit": isolated_wall,
            "total_gate": selection_wall + materialize_wall + batched_wall + isolated_wall,
        },
        "tolerance": tolerance,
        "within_gate": bool(float(difference.max().item()) <= tolerance) if len(sampled) else True,
    }
    result["primary_within_gate"] = result["within_gate"]
    if secondary and (audit or not result["within_gate"]):
        from agent.ppo.recompute_safety import secondary_audit
        secondary_started = _time.perf_counter()
        try:
            result["secondary"] = secondary_audit(
                learner.model, buffer, sampled, evaluation.request_logprob.float().cpu().tolist(),
                device, tolerance, force=audit,
            )
        except (ValueError, IndexError, KeyError, RuntimeError) as error:
            # A diagnostic error is a rejected proof, never permission to
            # update. Continue to durable failure publication below.
            result["secondary"] = {"accepted": False, "required": True,
                                   "reason": "secondary replay failed", "error": str(error)}
        result["timing_seconds"]["secondary_audit"] = _time.perf_counter() - secondary_started
        if not result["within_gate"]:
            result["within_gate"] = result["secondary"]["accepted"]
        elif audit and not result["secondary"]["accepted"]:
            result["within_gate"] = False
        if audit and isolated:
            worst = sorted(range(len(isolated)), key=lambda i: isolated[i], reverse=True)[:4]
            result["isolated_outlier_diagnosis"] = secondary_audit(
                learner.model, buffer, [sampled[i] for i in worst],
                [isolated_logprobs[i] for i in worst], device, tolerance, force=True)
            if not result["isolated_outlier_diagnosis"]["accepted"]:
                result["within_gate"] = False
    # Persist detailed diagnostics immediately, including rejected results.
    # Failure evidence contains immutable CPU copies of sampled rows and their
    # original actor groups plus model weights, enough for offline replay.
    if evidence_path and (not result["primary_within_gate"] or not result["within_gate"]):
        from agent.ppo.experimental_checkpoint import save_envelope
        groups = getattr(buffer, "actor_groups", [])
        flagged = [row for row, diff in zip(sampled, per_row) if not math_isfinite(diff) or diff > tolerance]
        ids = {id(row): i for i, row in enumerate(buffer.rows)}
        flagged_indices = {ids[id(row)] for row in flagged}
        # A forced secondary proof can reject while the primary passes. Keep
        # the original groups of those actual rejected rows, including the
        # isolated replay diagnosis, rather than unrelated large logprobs.
        for diagnosis in ("secondary", "isolated_outlier_diagnosis"):
            for evidence in result.get(diagnosis, {}).get("rows", []):
                index = evidence.get("row_index")
                if (not evidence.get("accepted", False) and type(index) is int
                        and 0 <= index < len(buffer.rows)):
                    flagged_indices.add(index)
        if audit and not flagged_indices:
            flagged = sorted(sampled,key=lambda row: abs(row.old_logprob),reverse=True)[:8]
            flagged_indices.update(ids[id(row)] for row in flagged)
        chosen_groups = [g for g in groups if any(g["start"] <= index < g["stop"] for index in flagged_indices)]
        group_rows = [buffer.rows[g["start"]:g["stop"]] for g in chosen_groups[:8]]
        payload = {"kind": "ppo_recompute_failure_evidence_v1", "result": result,
                   "model": {k: v.detach().cpu().clone() for k, v in learner.model.state_dict().items()},
                   "model_config": learner.model.config.to_dict(),
                   "precision_mode": learner.model.precision_contract.mode,
                   "sampled_rows": sampled, "actor_groups": chosen_groups[:8], "group_rows": group_rows,
                   "sampled_observations": [buffer.observation_store.get(r.observation_ref) for r in sampled],
                   "group_observations": [buffer.observation_store.stacked_indices([r.observation_ref for r in rows]) for rows in group_rows]}
        save_envelope(evidence_path, payload)
    return result


def math_isfinite(value):
    import math
    return math.isfinite(value)
