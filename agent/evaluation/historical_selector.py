"""PFSP-inspired historical-opponent weighting for SHADOW plans only.

This module is a dependency-free reference implementation.  It consumes
freeze-frame, verified paired evaluation W/D/L counts and returns a normalized
opponent distribution.  It never reads or mutates PPO state, training
checkpoints, formal evaluation receipts or the promotion alpha ledger.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

SELECTOR_SCHEMA = "pa3-history-sampler-reference-v1"


@dataclass(frozen=True)
class Record:
    policy_id: str
    wins: int
    draws: int
    losses: int
    verified: bool = True


def _capped_distribution(raw: list[float], cap: float) -> list[float]:
    n = len(raw)
    if n == 0:
        return []
    cap = max(cap, 1.0 / n)
    remaining = set(range(n))
    out = [0.0] * n
    mass = 1.0
    while remaining:
        total = sum(raw[i] for i in remaining)
        trial = {
            i: mass * (raw[i] / total if total > 0 else 1 / len(remaining))
            for i in remaining
        }
        exceeded = [i for i in sorted(remaining) if trial[i] > cap + 1e-12]
        if not exceeded:
            for i, value in trial.items():
                out[i] = value
            break
        for i in exceeded:
            out[i] = cap
            mass -= cap
            remaining.remove(i)
    return out


def calculate(records: list[Record], hard_mix: float = 0.60,
              close_mix: float = 0.30, uniform_mix: float = 0.10,
              prior_games: float = 128, max_weight: float = 0.40) -> dict:
    if not records:
        raise ValueError("history selector requires a nonempty pool")
    for value in (hard_mix, close_mix, uniform_mix):
        if not math.isfinite(value) or value < 0:
            raise ValueError("component mixtures must be finite nonnegative")
    if not math.isclose(sum((hard_mix, close_mix, uniform_mix)), 1.0, abs_tol=1e-12):
        raise ValueError("component mixture must sum to 1")
    if prior_games < 0 or not math.isfinite(prior_games):
        raise ValueError("invalid smoothing prior")
    items = sorted(records, key=lambda row: row.policy_id)
    if len({row.policy_id for row in items}) != len(items):
        raise ValueError("duplicate opponent ID")
    for row in items:
        if not row.policy_id or any(not isinstance(n, int) or n < 0 for n in
                                    (row.wins, row.draws, row.losses)):
            raise ValueError("invalid WDL/ID")
    k = len(items)
    active = [row.verified and (row.wins + row.draws + row.losses) > 0 for row in items]
    scores, hard, close = [], [], []
    for row, eligible in zip(items, active):
        games = row.wins + row.draws + row.losses
        smoothed = ((row.wins + 0.5 * row.draws + 0.5 * prior_games)
                    / (games + prior_games))
        scores.append(smoothed)
        hard.append((1.0 - smoothed) ** 2 if eligible else 0.0)
        close.append(4.0 * smoothed * (1.0 - smoothed) if eligible else 0.0)

    def normalized(values):
        total = sum(values)
        return [value / total for value in values] if total > 0 else [1 / k] * k

    if not any(active):
        raw = [1 / k] * k
    else:
        hard_n, close_n = normalized(hard), normalized(close)
        raw = [hard_mix * hard_n[i] + close_mix * close_n[i] + uniform_mix / k
               for i in range(k)]
    weights = _capped_distribution(raw, max_weight)
    opponents = [
        {
            "policy_id": row.policy_id,
            "win_rate_smoothed": round(scores[index], 8),
            "verified": bool(active[index]),
            "weight": weights[index],
            "games": row.wins + row.draws + row.losses,
        }
        for index, row in enumerate(items)
    ]
    payload = {
        "schema": SELECTOR_SCHEMA,
        "opponents": opponents,
        "components": {"hard": hard_mix, "close": close_mix, "uniform": uniform_mix},
        "prior_games": prior_games,
        "max_weight": max_weight,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload
