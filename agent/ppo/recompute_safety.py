"""Bounded, fail-closed investigation of mixed-precision parity outliers.

This never edits old_logprob, model weights, PPO config, or RNG streams. A
secondary pass requires the recorded actor group to replay tightly, unchanged
inputs/actions, and both AMP results within the original tolerance of a stable
FP32 reference. Missing provenance and every unexplained discrepancy fail.
"""
from __future__ import annotations

import hashlib
import json
import math

import torch


def compact_hash(fields):
    digest = hashlib.sha256()
    for key, value in sorted(fields.items()):
        digest.update(key.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        if value.dtype.hasobject:
            # Object arrays contain dynamic ragged rows; pointer bytes are not
            # a content identity and change across process/restoration.
            for row in value:
                digest.update(str((row.shape,row.dtype)).encode())
                digest.update(row.tobytes())
        else:
            digest.update(value.tobytes())
    return digest.hexdigest()


def row_contract_hash(rows):
    fields = ("action_ids", "candidate_mask", "entity_token", "move_token",
              "selected", "selected_actions", "branch_slots", "old_logprob")
    return hashlib.sha256(json.dumps(
        [[getattr(row, field) for field in fields] for row in rows],
        separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def evidence_passes(row, tolerance):
    """Pure decision rule, independently tested with corrupted evidence."""
    required = ("actor_replay", "actor_repeat", "isolated_amp", "fp32_group",
                "fp32_isolated", "batched_amp", "stored")
    if not row.get("inputs_verified") or any(
        key not in row or not math.isfinite(row[key]) for key in required
    ):
        return False
    tight = 1e-4
    return bool(
        abs(row["actor_replay"] - row["stored"]) <= tight
        and abs(row["actor_repeat"] - row["actor_replay"]) <= tight
        and abs(row["fp32_group"] - row["fp32_isolated"]) <= tight
        and abs(row["actor_replay"] - row["fp32_group"]) <= tolerance
        and abs(row["batched_amp"] - row["fp32_isolated"]) <= tolerance
        and abs(row["isolated_amp"] - row["fp32_isolated"]) <= tolerance
    )


def secondary_audit(model, buffer, sampled, batched_logprobs, device,
                    tolerance, *, force=False, max_outliers=8):
    """Replay at most eight flagged rows; budget exhaustion fails closed."""
    by_identity = {id(row): index for index, row in enumerate(buffer.rows)}
    flagged = [index for index, row in enumerate(sampled)
               if not math.isfinite(batched_logprobs[index])
               or not math.isfinite(row.old_logprob)
               or abs(batched_logprobs[index] - row.old_logprob) > tolerance]
    indices = flagged if flagged else (sorted(range(len(sampled)),
               key=lambda i: abs(batched_logprobs[i] - sampled[i].old_logprob), reverse=True)[:4] if force else [])
    result = {"required": bool(flagged), "outliers": len(flagged),
              "max_outliers": max_outliers, "rows": [], "accepted": False}
    if len(indices) > max_outliers:
        result["reason"] = "secondary audit budget exceeded"
        return result
    groups = getattr(buffer, "actor_groups", [])
    original_precision = model.precision_contract

    def evaluate(batch, fp32=False):
        model.configure_precision("A" if fp32 else original_precision.mode)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=not fp32 and device.type == "cuda",
        ):
            return model.evaluate(batch.observation, batch.candidates,
                                  selected=batch.candidates.selected)[1].request_logprob.float().cpu().tolist()

    # Explicit selected prefixes; no multinomial calls. No state restore is
    # needed because the audit does not consume any RNG.
    try:
        cache = {}
        for index in indices:
            row = sampled[index]
            absolute_index = by_identity[id(row)]
            group = next((g for g in groups if g["start"] <= absolute_index < g["stop"]), None)
            evidence = {"row_index": absolute_index, "match_id": row.match_id,
                        "request_index": row.request_index, "side": row.side,
                        "kind": int(row.request_kind), "branches": row.branch_count,
                        "selected": list(row.selected), "legal_counts": list(row.legal_counts),
                        "stored": row.old_logprob, "batched_amp": batched_logprobs[index],
                        "inputs_verified": False}
            if group is None:
                evidence["reason"] = "original actor group unavailable"
                result["rows"].append(evidence)
                continue
            key = group["start"]
            if key not in cache:
                rows = buffer.rows[group["start"]:group["stop"]]
                observations = buffer.observation_store.stacked_indices([r.observation_ref for r in rows])
                verified = (compact_hash(observations.to_compact_numpy()) == group["observation_hash"]
                            and row_contract_hash(rows) == group["row_contract_hash"])
                legal = all(all(0 <= pick < len(mask) and mask[pick]
                                and r.selected_actions[level] == r.action_ids[level][pick]
                                for level, (pick, mask) in enumerate(zip(r.selected, r.candidate_mask)))
                            for r in rows)
                if not verified or not legal:
                    cache[key] = None
                else:
                    batch = buffer.to_batch(rows, device=device)
                    cache[key] = (evaluate(batch), evaluate(batch), evaluate(batch, fp32=True))
                    del batch
            values = cache[key]
            if values is None:
                evidence["reason"] = "actor observations, actions or stored logprobs changed"
            else:
                offset = absolute_index - group["start"]
                one = buffer.to_batch([row], device=device)
                evidence.update(inputs_verified=True, actor_group_size=group["stop"]-group["start"],
                                actor_replay=values[0][offset], actor_repeat=values[1][offset],
                                fp32_group=values[2][offset],
                                isolated_amp=evaluate(one)[0], fp32_isolated=evaluate(one, fp32=True)[0])
                evidence["accepted"] = evidence_passes(evidence, tolerance)
            result["rows"].append(evidence)
    finally:
        model.configure_precision(original_precision.mode)
    result["accepted"] = bool(indices) and all(row.get("accepted", False) for row in result["rows"])
    result["reason"] = "verified shape/AMP variation" if result["accepted"] else "unexplained discrepancy"
    return result
