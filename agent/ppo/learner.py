"""PPO update loop (scaffolding; no real training is started).

Implements exactly the numeric contract of Full Spec 1.1 §8:

* Adam with peak LR 3e-4, betas (0.9, 0.999), eps 1e-5, weight decay 0,
* four epochs over the iteration's rows, global minibatch 4096 built from
  256-row microbatches with gradient accumulation,
* clip epsilon 0.2, gamma 1.0, GAE lambda 0.95, value coefficient 0.5 with
  ``0.5 * MSE`` and no value clipping, max grad norm 0.5,
* advantages normalized once over the iteration's actor rows (std floor 1e-8),
* entropy coefficient 0.01 on the ``log K``-normalized branch mean,
* uniform reverse-KL (``KL(U || pi)``) coefficient 0.001, also ``log K``
  normalized,
* target approximate KL 0.03: when an epoch exceeds it, the remaining epochs of
  that iteration only are skipped and a new rollout is collected,
* FP32 probability/loss math, optional FP16 autocast and a dynamic grad scaler,
* the LR schedule is clocked by committed natural training matches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from agent.buffer.rollout_buffer import RolloutBatch, RolloutBuffer
from agent.model.pa3_model import PA3Model
from agent.ppo.config import PPOConfig
from agent.ppo.losses import (
    approx_kl,
    normalize_advantages,
    policy_loss,
    value_loss,
)
from agent.ppo.schedule import MatchClockScheduler


@dataclass
class UpdateReport:
    """Per-iteration PPO bookkeeping (recorded to ``metrics.jsonl`` later)."""

    epochs_run: int = 0
    stopped_early: bool = False
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    uniform_kl: float = 0.0
    approx_kl: float = 0.0
    epoch_approx_kl: list[float] = field(default_factory=list)
    ratio_mean: float = 0.0
    clip_fraction: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    rows: int = 0
    actor_rows: int = 0
    optimizer_steps: int = 0
    scaler_scale: float = 1.0
    committed_matches: int = 0

    def as_dict(self) -> dict:
        data = self.__dict__.copy()
        data["epoch_approx_kl"] = list(self.epoch_approx_kl)
        return data


class PPOLearner:
    def __init__(
        self,
        model: PA3Model,
        config: Optional[PPOConfig] = None,
        device: Optional[torch.device | str] = None,
        amp: Optional[bool] = None,
    ) -> None:
        self.config = config or PPOConfig()
        self.config.validate()
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.warmup_start_lr,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = MatchClockScheduler(self.optimizer, self.config)
        self.amp = bool(amp) if amp is not None else (
            self.config.fp16_autocast and self.device.type == "cuda"
        )
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.amp and self.config.amp_grad_scaler
        )
        self.optimizer_steps = 0

    # -- batch preparation -------------------------------------------------
    def prepare_batch(
        self, buffer: RolloutBuffer, rows: Optional[list] = None
    ) -> RolloutBatch:
        """GAE + advantage normalization over the iteration's actor rows."""
        rows = list(buffer.rows if rows is None else rows)
        buffer.compute_gae(
            gamma=self.config.gamma, gae_lambda=self.config.gae_lambda, rows=rows
        )
        # Keep the full iteration on the host: a 10k-match rollout holds hundreds
        # of thousands of 96-token observations, and materializing all of them on
        # the GPU at once exhausts a 32 GiB card. Minibatches are moved to the
        # device one at a time in update() instead.
        batch = buffer.to_batch(rows, device="cpu")
        actor_mask = batch.actor_mask & batch.row_valid
        normalized = normalize_advantages(
            batch.raw_advantages,
            actor_mask=actor_mask,
            std_floor=self.config.advantage_std_floor,
        )
        return batch.with_advantages(normalized)

    # -- forward -----------------------------------------------------------
    def _row_masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(values.dtype)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def _forward(self, batch: RolloutBatch) -> dict:
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp,
        ):
            encoded = self.model.encode(batch.observation)
            values = self.model.value(encoded)
            evaluation = self.model.evaluate_encoded(
                encoded, batch.candidates, selected=batch.candidates.selected
            )
        # FP32 probability/loss math.
        values = values.float()
        logprob = evaluation.request_logprob.float()
        advantages = (
            batch.advantages.float()
            if batch.advantages is not None
            else batch.raw_advantages.float()
        )
        ratio = torch.exp((logprob - batch.old_logprob.float()).clamp(-20.0, 20.0))
        valid = batch.row_valid
        actor_mask = batch.actor_mask & valid
        policy = policy_loss(
            ratio,
            advantages,
            actor_mask=actor_mask,
            clip_epsilon=self.config.clip_epsilon,
        )
        value = value_loss(values[valid], batch.returns[valid])
        entropy = self._row_masked_mean(
            evaluation.request_entropy.float(), actor_mask
        )
        uniform_kl = self._row_masked_mean(
            evaluation.request_uniform_kl.float(), actor_mask
        )
        loss = (
            policy
            + self.config.value_coefficient * value
            - self.config.entropy_coefficient * entropy
            + self.config.uniform_kl_coefficient * uniform_kl
        )
        with torch.no_grad():
            row_kl = ((ratio - 1.0) - torch.log(ratio.clamp_min(1e-12)))
            actor_count = actor_mask.sum().clamp_min(1)
            stats = {
                "loss": loss,
                "policy_loss": policy.detach(),
                "value_loss": value.detach(),
                "entropy": entropy.detach(),
                "uniform_kl": uniform_kl.detach(),
                "approx_kl": (row_kl * actor_mask).sum().detach() / actor_count,
                "ratio_mean": (ratio * actor_mask).sum().detach() / actor_count,
                "clip_fraction": (
                    ((ratio - 1.0).abs() > self.config.clip_epsilon).float() * actor_mask
                ).sum().detach()
                / actor_count,
                "actor_rows": actor_mask.sum().detach(),
                "valid_rows": valid.sum().detach(),
            }
        return stats

    # -- update -------------------------------------------------------------
    def update(
        self,
        batch: RolloutBatch,
        committed_matches: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> UpdateReport:
        """Run at most ``ppo_epochs`` epochs; stop early on target KL."""
        if batch.advantages is None:
            raise ValueError("call prepare_batch() before update()")
        if committed_matches is not None:
            self.scheduler.step_to(int(committed_matches))
        report = UpdateReport(
            rows=len(batch),
            actor_rows=int((batch.actor_mask & batch.row_valid).sum().item()),
            learning_rate=self.scheduler.learning_rate,
            committed_matches=self.scheduler.matches,
        )
        self.model.train()
        accumulation = self.config.effective_accumulation
        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "uniform_kl": 0.0,
            "approx_kl": 0.0,
            "ratio_mean": 0.0,
            "clip_fraction": 0.0,
        }
        batches = 0

        for epoch in range(self.config.ppo_epochs):
            epoch_kl = 0.0
            epoch_batches = 0
            for minibatch in batch.iter_minibatches(
                self.config.global_minibatch_size,
                shuffle=True,
                generator=generator,
                drop_last=self.config.drop_last_minibatch,
            ):
                minibatch = minibatch.to(self.device)
                weight = minibatch.sample_weight if self.config.sample_weighted_ddp_reduction else 1.0
                self.optimizer.zero_grad(set_to_none=True)
                microbatches = list(minibatch.iter_microbatches(self.config.microbatch_size))
                used = 0
                for micro in microbatches:
                    if int(micro.row_valid.sum()) == 0:
                        continue  # padding-only chunk carries no experience
                    stats = self._forward(micro)
                    scale = micro.sample_weight / max(1, len(microbatches))
                    self.scaler.scale(stats["loss"] * scale).backward()
                    used += 1
                if used == 0:
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer_steps += 1

                report.grad_norm = float(grad_norm)
                report.scaler_scale = float(self.scaler.get_scale())
                report.optimizer_steps = self.optimizer_steps
                for key in totals:
                    totals[key] += float(stats[key]) * weight
                batches += 1
                epoch_kl += float(stats["approx_kl"]) * weight
                epoch_batches += 1

            if epoch_batches == 0:
                break
            epoch_kl /= max(1e-9, epoch_batches)
            report.epoch_approx_kl.append(epoch_kl)
            report.epochs_run = epoch + 1
            if epoch_kl > self.config.target_approx_kl:
                report.stopped_early = True
                break

        if batches:
            denominator = float(batches)
            report.policy_loss = totals["policy_loss"] / denominator
            report.value_loss = totals["value_loss"] / denominator
            report.entropy = totals["entropy"] / denominator
            report.uniform_kl = totals["uniform_kl"] / denominator
            report.approx_kl = totals["approx_kl"] / denominator
            report.ratio_mean = totals["ratio_mean"] / denominator
            report.clip_fraction = totals["clip_fraction"] / denominator
        report.learning_rate = self.scheduler.learning_rate
        return report

    # -- checkpointing --------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "optimizer_steps": self.optimizer_steps,
            "config": {
                "ppo": self.config.__dict__,
                "model": self.model.config.to_dict(),
            },
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state.get("scheduler", {}))
        if "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])
        self.optimizer_steps = int(state.get("optimizer_steps", 0))
