"""Learning-rate schedule clocked by committed natural training matches.

The clock is *not* wall time, optimizer steps or checkpoints: it advances only
when natural training matches are committed to the experience counters.  First
250,000 matches warm up linearly from 1e-5 to 3e-4, then cosine decay reaches
3e-5 at the 100,000,000-match horizon.
"""

from __future__ import annotations

import math
from typing import Optional

from agent.ppo.config import PPOConfig


class MatchClockScheduler:
    def __init__(self, optimizer, config: Optional[PPOConfig] = None) -> None:
        self.optimizer = optimizer
        self.config = config or PPOConfig()
        self.config.validate()
        self.matches = 0
        for group in self.optimizer.param_groups:
            group.setdefault("initial_lr", self.config.warmup_start_lr)
            group["lr"] = self.config.warmup_start_lr

    # -- pure function of the clock -------------------------------------
    def lr_for_matches(self, matches: float) -> float:
        cfg = self.config
        matches = max(0.0, float(matches))
        if matches < cfg.warmup_matches:
            progress = matches / max(1.0, float(cfg.warmup_matches))
            return cfg.warmup_start_lr + (cfg.peak_lr - cfg.warmup_start_lr) * progress
        span = max(1.0, float(cfg.horizon_matches - cfg.warmup_matches))
        progress = min(1.0, (matches - cfg.warmup_matches) / span)
        return cfg.final_lr + 0.5 * (cfg.peak_lr - cfg.final_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    # -- clock ------------------------------------------------------------
    def advance(self, committed_matches: int) -> float:
        """Advance the clock by ``committed_matches`` and apply the new LR."""
        if committed_matches < 0:
            raise ValueError("committed matches cannot decrease")
        self.matches += int(committed_matches)
        return self.step_to(self.matches)

    def step_to(self, matches: int) -> float:
        """Set the clock to an absolute committed-match count."""
        self.matches = max(0, int(matches))
        lr = self.lr_for_matches(self.matches)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    @property
    def learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    # -- checkpointing -----------------------------------------------------
    def state_dict(self) -> dict:
        return {"matches": self.matches, "lr": self.learning_rate}

    def load_state_dict(self, state: dict) -> None:
        self.step_to(int(state.get("matches", 0)))
