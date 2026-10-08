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

from dataclasses import dataclass, field, replace
from typing import Optional
import time

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
    grad_norm_max: float = 0.0
    learning_rate: float = 0.0
    rows: int = 0
    actor_rows: int = 0
    optimizer_steps: int = 0
    optimizer_steps_skipped: int = 0
    scaler_scale: float = 1.0
    committed_matches: int = 0

    def as_dict(self) -> dict:
        data = self.__dict__.copy()
        data["epoch_approx_kl"] = list(self.epoch_approx_kl)
        return data


@dataclass
class StreamingPlan:
    """Whole-iteration plan without materializing the iteration.

    ``rows`` stay as the compact per-row records; ``advantages`` holds the
    row-normalized advantages, and each minibatch is materialized on demand
    from the buffer's compact store.
    """

    buffer: RolloutBuffer
    rows: list
    advantages: torch.Tensor
    actor_rows: int


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
        # Set by `attach_ddp`; when present the learner routes the forward
        # through the DDP entry and uses the global-denominator objective.
        self.ddp_model = None
        self.world_size = 1
        # Stage accounting for the M0 breakdown (always collected; ~12
        # perf_counter calls per minibatch, no GPU synchronization).
        self.profile: dict[str, float] = {}
        self.prepare_profile: dict[str, float] = {}

    @staticmethod
    def _blank_profile() -> dict:
        return {
            "prepare_total": 0.0, "prepare_build": 0.0, "prepare_normalize": 0.0,
            "minibatch_select": 0.0, "h2d": 0.0, "forward": 0.0, "backward": 0.0,
            "optimizer": 0.0, "metrics": 0.0, "epoch_sync": 0.0,
        }

    # -- batch preparation -------------------------------------------------
    def prepare_batch(
        self, buffer: RolloutBuffer, rows: Optional[list] = None
    ) -> RolloutBatch:
        """GAE + advantage normalization over the iteration's actor rows."""
        profile = {"gae": 0.0, "build": 0.0, "normalize": 0.0, "total": 0.0}
        started = time.perf_counter()
        rows = list(buffer.rows if rows is None else rows)
        start = time.perf_counter()
        buffer.compute_gae(
            gamma=self.config.gamma, gae_lambda=self.config.gae_lambda, rows=rows
        )
        profile["gae"] = time.perf_counter() - start
        start = time.perf_counter()
        # Keep the full iteration on the host: a 10k-match rollout holds hundreds
        # of thousands of 96-token observations, and materializing all of them on
        # the GPU at once exhausts a 32 GiB card. Minibatches are moved to the
        # device one at a time in update() instead.
        batch = buffer.to_batch(rows, device="cpu")
        profile["build"] = time.perf_counter() - start
        profile["build_observations"] = buffer.profile.get("observations", 0.0)
        profile["build_candidates"] = buffer.profile.get("candidates", 0.0)
        profile["build_columns"] = buffer.profile.get("columns", 0.0)
        profile["build_move"] = buffer.profile.get("move", 0.0)
        start = time.perf_counter()
        actor_mask = batch.actor_mask & batch.row_valid
        normalized = normalize_advantages(
            batch.raw_advantages,
            actor_mask=actor_mask,
            std_floor=self.config.advantage_std_floor,
        )
        profile["normalize"] = time.perf_counter() - start
        profile["total"] = time.perf_counter() - started
        self.prepare_profile = profile
        return batch.with_advantages(normalized)

    def prepare_streaming(
        self, buffer: RolloutBuffer, rows: Optional[list] = None
    ) -> StreamingPlan:
        """GAE + advantage normalization without materializing the iteration."""
        profile = {"gae": 0.0, "normalize": 0.0, "total": 0.0}
        started = time.perf_counter()
        rows = list(buffer.rows if rows is None else rows)
        start = time.perf_counter()
        buffer.compute_gae(
            gamma=self.config.gamma, gae_lambda=self.config.gae_lambda, rows=rows
        )
        profile["gae"] = time.perf_counter() - start
        start = time.perf_counter()
        raw = torch.tensor([row.advantage for row in rows], dtype=torch.float32)
        actor = torch.tensor([row.actor_active for row in rows], dtype=torch.bool)
        normalized = normalize_advantages(
            raw, actor_mask=actor, std_floor=self.config.advantage_std_floor
        )
        profile["normalize"] = time.perf_counter() - start
        profile["total"] = time.perf_counter() - started
        self.prepare_profile = profile
        return StreamingPlan(
            buffer=buffer, rows=rows, advantages=normalized,
            actor_rows=int(actor.sum().item()),
        )

    # -- forward -----------------------------------------------------------
    def _row_masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(values.dtype)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def _forward(self, batch: RolloutBatch) -> dict:
        terms = self._forward_terms(batch)
        return {
            "loss": terms["loss_unscaled"] + self.config.value_coefficient * terms["value_mean"],
            "policy_loss": terms["policy_mean"],
            "value_loss": terms["value_mean_detached"],
            "entropy": terms["entropy_mean"],
            "uniform_kl": terms["uniform_kl_mean"],
            "approx_kl": terms["approx_kl_mean"],
            "ratio_mean": terms["ratio_mean"],
            "clip_fraction": terms["clip_fraction"],
            "actor_rows": terms["actor_count"],
            "valid_rows": terms["value_count"],
        }

    def _forward_terms(self, batch: RolloutBatch) -> dict:
        """Raw sufficient statistics of one micro/minibatch.

        Every metric is returned as a *sum plus its denominator* instead of a
        mean, so the caller can aggregate exactly over microbatches that hold
        different numbers of valid actor rows (padding, uneven actor masks, the
        last partial minibatch). The loss terms stay attached to the graph; the
        reported scalars are detached.
        """
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp,
        ):
            if self.ddp_model is not None:
                # DDP requires its own forward entry to own the graph and the
                # `no_sync` context to cover forward *and* backward.
                encoded, values, evaluation = self.ddp_model(
                    batch.observation, batch.candidates, selected=batch.candidates.selected
                )
            else:
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
        actor_f = actor_mask.to(ratio.dtype)
        actor_count = actor_mask.sum()
        value_count = valid.sum()
        actor_denom = actor_count.clamp_min(1)
        entropy_sum = (evaluation.request_entropy.float() * actor_f).sum()
        uniform_kl_sum = (evaluation.request_uniform_kl.float() * actor_f).sum()
        entropy = entropy_sum / actor_denom
        uniform_kl = uniform_kl_sum / actor_denom
        loss = (
            policy
            - self.config.entropy_coefficient * entropy
            + self.config.uniform_kl_coefficient * uniform_kl
        )
        with torch.no_grad():
            row_kl = ((ratio - 1.0) - torch.log(ratio.clamp_min(1e-12)))
            kl_sum = (row_kl * actor_f).sum()
            ratio_sum = (ratio * actor_f).sum()
            clip_sum = (
                ((ratio - 1.0).abs() > self.config.clip_epsilon).to(ratio.dtype) * actor_f
            ).sum()
            stats = {
                "loss_unscaled": loss,
                "value_mean": value,
                "policy_mean": policy.detach(),
                "value_mean_detached": value.detach(),
                "entropy_mean": entropy.detach(),
                "uniform_kl_mean": uniform_kl.detach(),
                "approx_kl_mean": (kl_sum / actor_denom).detach(),
                "ratio_mean": (ratio_sum / actor_denom).detach(),
                "clip_fraction": (clip_sum / actor_denom).detach(),
                "actor_count": actor_count.detach(),
                "value_count": value_count.detach(),
                "policy_sum": policy.detach() * actor_count.detach(),
                "value_sum": value.detach() * value_count.detach(),
                "entropy_sum": entropy_sum.detach(),
                "uniform_kl_sum": uniform_kl_sum.detach(),
                "kl_sum": kl_sum.detach(),
                "ratio_sum": ratio_sum.detach(),
                "clip_sum": clip_sum.detach(),
            }
        return stats

    # -- update -------------------------------------------------------------
    def _process_minibatch(self, minibatch, sum_keys, epoch_sums, exact, profile, report) -> int:
        """Micro loop + optimizer step for one minibatch; returns micro count."""
        weight = minibatch.sample_weight if self.config.sample_weighted_ddp_reduction else 1.0
        # Denominators of this minibatch's objective: the actor terms average
        # over valid actor rows, the value term over valid rows. Counted on the
        # CPU copy so no GPU scalar is read per minibatch.
        minibatch_actor = float((minibatch.actor_mask & minibatch.row_valid).sum().item())
        minibatch_valid = float(minibatch.row_valid.sum().item())
        self.optimizer.zero_grad(set_to_none=True)
        # Drop padding-only microbatches on the CPU (a GPU-side
        # `int(micro.row_valid.sum())` check forces one device sync per micro).
        micro_ranges = []
        for begin in range(0, len(minibatch), self.config.microbatch_size):
            end = min(begin + self.config.microbatch_size, len(minibatch))
            if bool(minibatch.row_valid[begin:end].any().item()):
                micro_ranges.append((begin, end))
        h2d_start = time.perf_counter()
        minibatch = minibatch.to(self.device)
        profile["h2d"] += time.perf_counter() - h2d_start
        used = 0
        for begin, end in micro_ranges:
            forward_start = time.perf_counter()
            micro = minibatch.select(
                torch.arange(begin, end, dtype=torch.long,
                             device=minibatch.old_logprob.device)
            )
            terms = self._forward_terms(micro)
            profile["forward"] += time.perf_counter() - forward_start
            forward_start = time.perf_counter()
            if exact:
                # Exact full-minibatch objective: each micro contributes its
                # share of the minibatch's valid actor/value rows.
                actor_scale = terms["actor_count"] / max(minibatch_actor, 1.0)
                value_scale = terms["value_count"] / max(minibatch_valid, 1.0)
                loss = weight * (
                    terms["loss_unscaled"] * actor_scale
                    + self.config.value_coefficient * terms["value_mean"] * value_scale
                )
            else:  # legacy sample_weight/len(microbatches) accumulation
                scale = micro.sample_weight / max(1, len(micro_ranges))
                loss = weight * scale * (
                    terms["loss_unscaled"]
                    + self.config.value_coefficient * terms["value_mean"]
                )
            self.scaler.scale(loss).backward()
            profile["backward"] += time.perf_counter() - forward_start
            forward_start = time.perf_counter()
            for key in sum_keys:
                epoch_sums[key] = epoch_sums[key] + terms[key].float()
            profile["metrics"] += time.perf_counter() - forward_start
            used += 1
        if used == 0:
            self.optimizer.zero_grad(set_to_none=True)
            return 0
        optimizer_start = time.perf_counter()
        scale_before = self.scaler.get_scale()
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        profile["optimizer"] += time.perf_counter() - optimizer_start
        stepped = self.scaler.get_scale() >= scale_before
        if stepped:
            self.optimizer_steps += 1
        else:
            report.optimizer_steps_skipped += 1
        report.grad_norm = float(grad_norm.detach())
        report.grad_norm_max = max(report.grad_norm_max, report.grad_norm)
        report.scaler_scale = float(self.scaler.get_scale())
        report.optimizer_steps = self.optimizer_steps
        return used

    def _new_report(self, rows, actor_rows, committed_matches) -> UpdateReport:
        if committed_matches is not None:
            self.scheduler.step_to(int(committed_matches))
        return UpdateReport(
            rows=rows,
            actor_rows=int(actor_rows),
            learning_rate=self.scheduler.learning_rate,
            committed_matches=self.scheduler.matches,
        )

    def _finish_report(self, report, totals, sum_keys) -> None:
        if report.epochs_run:
            actor_total = float(totals["actor_count"].clamp_min(1).item())
            value_total = float(totals["value_count"].clamp_min(1).item())
            report.policy_loss = float(totals["policy_sum"].item()) / actor_total
            report.value_loss = float(totals["value_sum"].item()) / value_total
            report.entropy = float(totals["entropy_sum"].item()) / actor_total
            report.uniform_kl = float(totals["uniform_kl_sum"].item()) / actor_total
            report.approx_kl = float(totals["kl_sum"].item()) / actor_total
            report.ratio_mean = float(totals["ratio_sum"].item()) / actor_total
            report.clip_fraction = float(totals["clip_sum"].item()) / actor_total
        report.learning_rate = self.scheduler.learning_rate

    def attach_ddp(self, ddp_model, world_size: int) -> None:
        """Route forwards through a DDP wrapper and enable the global objective."""
        self.ddp_model = ddp_model
        self.world_size = int(world_size)

    def detach_ddp(self) -> None:
        self.ddp_model = None
        self.world_size = 1

    def update_ddp(
        self,
        plan: StreamingPlan,
        committed_matches: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        communication=None,
    ) -> UpdateReport:
        """Two-rank DDP update with global actor/value denominators.

        Per minibatch the local valid actoralue counts are all-reduced once
        (`all_reduce_sums`), and each rank's loss uses
        ``world_size * (sum_local/A + value_coef * value_sum_local/V)`` so DDP's
        gradient averaging reproduces the single-process global objective. The
        `no_sync` context wraps the forward *and* backward of every non-final
        microbatch.
        """
        from agent.ppo.ddp import all_reduce_sums, ddp_rank_loss

        rows = plan.rows
        report = self._new_report(len(rows), plan.actor_rows, committed_matches)
        self.model.train()
        sum_keys = (
            "policy_sum", "value_sum", "entropy_sum", "uniform_kl_sum",
            "kl_sum", "ratio_sum", "clip_sum", "actor_count", "value_count",
        )
        totals = {key: torch.zeros((), dtype=torch.float32, device=self.device) for key in sum_keys}
        profile = self._blank_profile()
        # Per-rank chunk: the global minibatch is split across ranks; using the
        # global size here would double the padded batch on every rank.
        batch_size = max(
            1,
            self.config.per_rank_minibatch_size
            if self.config.per_rank_minibatch_size > 0
            else self.config.global_minibatch_size // max(self.world_size, 1),
        )
        world_size = max(self.world_size, 1)
        update_started = time.perf_counter()
        # Collective alignment: ranks have different local row counts, so the
        # number of minibatches (and therefore all-reduce calls and optimizer
        # steps) must be agreed before the epoch loop or DDP deadlocks. Every
        # rank runs the same count; a rank that runs out processes an
        # all-padding minibatch whose loss is zero.
        local_batches = (len(rows) + batch_size - 1) // batch_size
        batch_count = local_batches
        if world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            tensor = torch.tensor([local_batches], dtype=torch.long, device=self.device)
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
            batch_count = int(tensor.item())

        for epoch in range(self.config.ppo_epochs):
            epoch_sums = {key: torch.zeros((), dtype=torch.float32, device=self.device) for key in sum_keys}
            epoch_batches = 0
            start = time.perf_counter()
            order = torch.randperm(len(rows), generator=generator)
            for batch_index in range(batch_count):
                begin = batch_index * batch_size
                chunk = order[begin:begin + batch_size] if begin < len(order) else order[:1]
                real = len(chunk)
                padding = batch_size - real
                if begin >= len(order):
                    # Aligned padding-only minibatch: zero loss, no rows.
                    real = 0
                    padding = batch_size
                    chunk = order[:1]
                valid = torch.cat([
                    torch.ones(real, dtype=torch.bool),
                    torch.zeros(padding, dtype=torch.bool),
                ])
                indices = chunk if padding == 0 else torch.cat([chunk[:real] if real else chunk[:0], chunk[:1].repeat(padding)])
                minibatch_rows = [rows[int(index)] for index in indices.tolist()]
                minibatch = plan.buffer.to_batch(minibatch_rows, device="cpu")
                minibatch = replace(
                    minibatch, row_valid=valid, advantages=plan.advantages[indices]
                )
                profile["minibatch_select"] += time.perf_counter() - start
                start = time.perf_counter()

                local_actor = (minibatch.actor_mask & minibatch.row_valid).sum().float()
                local_value = minibatch.row_valid.sum().float()
                minibatch = minibatch.to(self.device)
                profile["h2d"] += time.perf_counter() - start
                start = time.perf_counter()
                actor_total, value_total = all_reduce_sums(
                    torch.zeros((), device=self.device),
                    torch.zeros((), device=self.device),
                    local_actor.to(self.device),
                    local_value.to(self.device),
                    world_size,
                )
                profile["epoch_sync"] += time.perf_counter() - start

                tiny = 1e-8
                self.optimizer.zero_grad(set_to_none=True)
                micro_ranges = []
                for micro_begin in range(0, len(minibatch), self.config.microbatch_size):
                    micro_end = min(micro_begin + self.config.microbatch_size, len(minibatch))
                    if bool(minibatch.row_valid[micro_begin:micro_end].any().item()):
                        micro_ranges.append((micro_begin, micro_end))
                used = 0
                if communication is not None:
                    communication.reset()
                for micro_index, (micro_begin, micro_end) in enumerate(micro_ranges):
                    context = communication.context() if communication is not None else None
                    from contextlib import nullcontext

                    with (context if context is not None else nullcontext()):
                        forward_start = time.perf_counter()
                        micro = minibatch.select(
                            torch.arange(micro_begin, micro_end, dtype=torch.long,
                                         device=minibatch.old_logprob.device)
                        )
                        terms = self._forward_terms(micro)
                        profile["forward"] += time.perf_counter() - forward_start
                        actor_sum = terms["loss_unscaled"] * terms["actor_count"]
                        value_sum = terms["value_mean"] * terms["value_count"]
                        loss = ddp_rank_loss(
                            actor_sum, value_sum, actor_total, value_total,
                            world_size, self.config.value_coefficient,
                        )
                        self.scaler.scale(loss).backward()
                    for key in sum_keys:
                        epoch_sums[key] = epoch_sums[key] + terms[key].float()
                    used += 1
                if used == 0:
                    continue
                optimizer_start = time.perf_counter()
                scale_before = self.scaler.get_scale()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                profile["optimizer"] += time.perf_counter() - optimizer_start
                stepped = self.scaler.get_scale() >= scale_before
                if stepped:
                    self.optimizer_steps += 1
                else:
                    report.optimizer_steps_skipped += 1
                report.grad_norm = float(grad_norm.detach())
                report.grad_norm_max = max(report.grad_norm_max, report.grad_norm)
                report.scaler_scale = float(self.scaler.get_scale())
                report.optimizer_steps = self.optimizer_steps
                epoch_batches += 1
                start = time.perf_counter()

            if epoch_batches == 0:
                break
            for key in sum_keys:
                totals[key] = totals[key] + epoch_sums[key]
            actor_total = float(epoch_sums["actor_count"].clamp_min(1).item())
            epoch_kl = float(epoch_sums["kl_sum"].item()) / actor_total
            report.epoch_approx_kl.append(epoch_kl)
            report.epochs_run = epoch + 1
            if epoch_kl > self.config.target_approx_kl:
                report.stopped_early = True
                break

        self._finish_report(report, totals, sum_keys)
        profile["total"] = time.perf_counter() - update_started
        self.profile = profile
        return report

    def update(
        self,
        batch: RolloutBatch,
        committed_matches: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> UpdateReport:
        """Run at most ``ppo_epochs`` epochs; stop early on target KL."""
        if batch.advantages is None:
            raise ValueError("call prepare_batch() before update()")
        report = self._new_report(
            len(batch), int((batch.actor_mask & batch.row_valid).sum().item()), committed_matches
        )
        self.model.train()
        sum_keys = (
            "policy_sum", "value_sum", "entropy_sum", "uniform_kl_sum",
            "kl_sum", "ratio_sum", "clip_sum", "actor_count", "value_count",
        )
        totals = {key: torch.zeros((), dtype=torch.float32, device=self.device) for key in sum_keys}
        exact = self.config.exact_row_weighted_accumulation
        profile = self._blank_profile()
        update_started = time.perf_counter()

        for epoch in range(self.config.ppo_epochs):
            epoch_sums = {key: torch.zeros((), dtype=torch.float32, device=self.device) for key in sum_keys}
            epoch_batches = 0
            start = time.perf_counter()
            minibatch_iter = batch.iter_minibatches(
                self.config.global_minibatch_size,
                shuffle=True,
                generator=generator,
                drop_last=self.config.drop_last_minibatch,
            )
            for minibatch in minibatch_iter:
                profile["minibatch_select"] += time.perf_counter() - start
                start = time.perf_counter()
                used = self._process_minibatch(
                    minibatch, sum_keys, epoch_sums, exact, profile, report
                )
                if used:
                    epoch_batches += 1
                start = time.perf_counter()

            if epoch_batches == 0:
                break
            for key in sum_keys:
                totals[key] = totals[key] + epoch_sums[key]
            actor_total = float(epoch_sums["actor_count"].clamp_min(1).item())
            value_total = float(epoch_sums["value_count"].clamp_min(1).item())
            epoch_kl = float(epoch_sums["kl_sum"].item()) / actor_total
            profile["epoch_sync"] += time.perf_counter() - start
            report.epoch_approx_kl.append(epoch_kl)
            report.epochs_run = epoch + 1
            start = time.perf_counter()
            if epoch_kl > self.config.target_approx_kl:
                report.stopped_early = True
                break

        self._finish_report(report, totals, sum_keys)
        profile["total"] = time.perf_counter() - update_started
        self.profile = profile
        return report

    def update_streaming(
        self,
        plan: StreamingPlan,
        committed_matches: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> UpdateReport:
        """Update from a `StreamingPlan`, materializing one minibatch at a time.

        The global minibatch size, the micro split, the exact row-weighted
        objective, the padded final minibatch and every reported statistic are
        identical to `update()` over the same rows; only the host memory
        footprint differs (the whole iteration is never expanded at once).
        """
        rows = plan.rows
        report = self._new_report(len(rows), plan.actor_rows, committed_matches)
        self.model.train()
        sum_keys = (
            "policy_sum", "value_sum", "entropy_sum", "uniform_kl_sum",
            "kl_sum", "ratio_sum", "clip_sum", "actor_count", "value_count",
        )
        totals = {key: torch.zeros((), dtype=torch.float32, device=self.device) for key in sum_keys}
        exact = self.config.exact_row_weighted_accumulation
        profile = self._blank_profile()
        batch_size = self.config.global_minibatch_size
        update_started = time.perf_counter()

        for epoch in range(self.config.ppo_epochs):
            epoch_sums = {key: torch.zeros((), dtype=torch.float32, device=self.device) for key in sum_keys}
            epoch_batches = 0
            start = time.perf_counter()
            order = torch.randperm(len(rows), generator=generator)
            for begin in range(0, len(rows), batch_size):
                chunk = order[begin:begin + batch_size]
                real = len(chunk)
                padding = batch_size - real
                valid = torch.cat([
                    torch.ones(real, dtype=torch.bool),
                    torch.zeros(padding, dtype=torch.bool),
                ])
                indices = chunk if padding == 0 else torch.cat([chunk, chunk[:1].repeat(padding)])
                minibatch_rows = [rows[int(index)] for index in indices.tolist()]
                minibatch = plan.buffer.to_batch(minibatch_rows, device="cpu")
                minibatch = replace(
                    minibatch,
                    row_valid=valid,
                    advantages=plan.advantages[indices],
                )
                profile["minibatch_select"] += time.perf_counter() - start
                start = time.perf_counter()
                used = self._process_minibatch(
                    minibatch, sum_keys, epoch_sums, exact, profile, report
                )
                if used:
                    epoch_batches += 1
                start = time.perf_counter()

            if epoch_batches == 0:
                break
            for key in sum_keys:
                totals[key] = totals[key] + epoch_sums[key]
            actor_total = float(epoch_sums["actor_count"].clamp_min(1).item())
            epoch_kl = float(epoch_sums["kl_sum"].item()) / actor_total
            profile["epoch_sync"] += time.perf_counter() - start
            report.epoch_approx_kl.append(epoch_kl)
            report.epochs_run = epoch + 1
            start = time.perf_counter()
            if epoch_kl > self.config.target_approx_kl:
                report.stopped_early = True
                break

        self._finish_report(report, totals, sum_keys)
        profile["total"] = time.perf_counter() - update_started
        self.profile = profile
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
