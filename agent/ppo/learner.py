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

from dataclasses import dataclass, field, fields, is_dataclass, replace
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
    cached_batch: Optional[RolloutBatch] = None
    compact_candidates: bool = False


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
        self, buffer: RolloutBuffer, rows: Optional[list] = None, *,
        cache_device: Optional[str] = None, cache_max_bytes: int = 8 << 30,
        compact_candidates: bool = False,
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
        plan = StreamingPlan(
            buffer=buffer, rows=rows, advantages=normalized,
            actor_rows=int(actor.sum().item()), compact_candidates=compact_candidates,
        )
        return self._cache_plan(plan, cache_device, cache_max_bytes)

    def prepare_streaming_ddp(
        self,
        buffer: RolloutBuffer,
        rows: Optional[list] = None,
        group=None,
        *,
        cache_device: Optional[str] = None,
        cache_max_bytes: int = 8 << 30,
        compact_candidates: bool = False,
    ) -> StreamingPlan:
        """Streaming plan whose advantages are normalised over **all ranks**.

        Rank-local GAE is fine (each rank owns its on-policy rows), but the
        advantage statistics must be global: the plan's §5.1 requires the
        current iteration's actor rows to share one mean/std. Two all-reduces
        (count/sum, then squared deviations) keep the variance numerically
        stable, matching ``normalize_advantages``'s unbiased=False definition.
        """
        from agent.ppo.ddp import all_reduce_tensor, normalize_advantages_global

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
        def _reduce(value: torch.Tensor) -> torch.Tensor:
            return all_reduce_tensor(value.to(self.device), group=group).cpu()

        normalized = normalize_advantages_global(
            raw,
            actor,
            reduce=_reduce,
            std_floor=self.config.advantage_std_floor,
        )
        profile["normalize"] = time.perf_counter() - start
        profile["total"] = time.perf_counter() - started
        self.prepare_profile = profile
        plan = StreamingPlan(
            buffer=buffer, rows=rows, advantages=normalized.float(),
            actor_rows=int(actor.sum().item()), compact_candidates=compact_candidates,
        )
        return self._cache_plan(plan, cache_device, cache_max_bytes)

    @staticmethod
    def _tensor_bytes(value) -> int:
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if is_dataclass(value):
            return sum(PPOLearner._tensor_bytes(getattr(value, f.name)) for f in fields(value))
        return 0

    def _cache_plan(self, plan, cache_device, cache_max_bytes):
        """Expand immutable rollout tensors once, within a bounded budget.

        Large production rollouts retain streaming admission. CUDA caching
        reserves activation space scaled to the microbatch and respects the 28 GiB
        per-card soft budget; it never relies on an OOM to select a path.
        """
        if cache_device not in (None, "none", "cpu", "cuda"):
            raise ValueError("cache_device must be none, cpu or cuda")
        if cache_device in (None, "none") or not plan.rows:
            return plan
        if cache_device == "cuda" and self.device.type != "cuda":
            raise ValueError("CUDA rollout cache requires a CUDA learner")
        started = time.perf_counter()
        probe = plan.buffer.to_batch(plan.rows[:1], device="cpu")
        estimate = (self._tensor_bytes(probe) + 4) * len(plan.rows)
        budget = max(0, int(cache_max_bytes))
        if cache_device == "cuda":
            free, total = torch.cuda.mem_get_info(self.device)
            # Freed tensors in PyTorch's allocator are reusable by this cache.
            # Counting reserved memory as live would disable admission after a
            # warm update even though the previous cache has been released.
            reusable = torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
            available = free + reusable
            live = total - available
            reserve = (2 << 30) + int((10 << 30) * self.config.microbatch_size / 1024)
            budget = min(budget, max(0, min(available, (28 << 30) - live) - reserve))
        self.prepare_profile["cache_estimated_bytes"] = estimate
        self.prepare_profile["cache_budget_bytes"] = budget
        self.prepare_profile["cache_enabled"] = 0.0
        if estimate <= budget:
            batch = plan.buffer.to_batch(plan.rows, device="cpu").with_advantages(plan.advantages)
            if plan.compact_candidates:
                batch = self._compact_candidate_batch(batch, plan.rows)
            self.prepare_profile["candidate_padding"] = batch.candidates.shape[2]
            plan.cached_batch = batch.to(self.device if cache_device == "cuda" else "cpu")
            self.prepare_profile["cache_enabled"] = 1.0
        self.prepare_profile["cache_build"] = time.perf_counter() - started
        self.prepare_profile["total"] += self.prepare_profile["cache_build"]
        return plan

    def _plan_minibatch(self, plan, indices, valid):
        if plan.cached_batch is not None:
            return plan.cached_batch.select(indices, row_valid=valid)
        rows = [plan.rows[int(index)] for index in indices.tolist()]
        batch = replace(
            plan.buffer.to_batch(rows, device="cpu"),
            row_valid=valid,
            advantages=plan.advantages[indices],
        )
        return self._compact_candidate_batch(batch, rows) if plan.compact_candidates else batch

    @staticmethod
    def _compact_candidate_batch(batch, rows):
        # Round up for tensor-core-friendly widths, retaining every stored
        # candidate (legal or illegal), its order and its selected prefix.
        required = max((len(branch) for row in rows for branch in row.action_ids), default=1)
        width = min(batch.candidates.shape[2], max(8, ((required + 7) // 8) * 8))
        return replace(batch, candidates=batch.candidates.trim_padding(width))

    def _update_distributed_scaler(self, globally_finite: bool) -> None:
        # Manual SUM happens after unscale_ has recorded *local* overflow.
        # All ranks must back off together, including those with finite local
        # gradients. Public state APIs also reset the growth tracker exactly.
        if self.scaler.is_enabled() and not globally_finite:
            state = self.scaler.state_dict()
            state["scale"] *= state["backoff_factor"]
            state["_growth_tracker"] = 0
            self.scaler.update(new_scale=state["scale"])
            self.scaler.load_state_dict(state)
        else:
            self.scaler.update()

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
        # Boolean indexing invokes CUDA nonzero (a host synchronization) twice
        # per micro. A fixed-shape masked reduction has the same objective and
        # keeps padding-only micros connected to the critic graph.
        residual = torch.where(valid, values - batch.returns.float(), 0.0)
        value = 0.5 * residual.square().sum() / valid.sum().clamp_min(1)
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
            micro = minibatch.narrow(begin, end)
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
        fallback_batch: Optional[RolloutBatch] = None,
    ) -> UpdateReport:
        """Fixed-step two-rank DDP update with global denominators.

        Protocol (plan §5.1 / §6.2):

        * every rank runs the same number of minibatches (all-reduce MAX of the
          local counts) and exactly ``micro_steps`` micro steps inside each of
          them; a rank without real rows in a step runs a graph-connected zero
          micro so every rank executes the identical DDP collective sequence;
        * ``no_sync`` wraps the forward *and* backward of every non-final micro
          step, and the final micro step always synchronises;
        * the loss is ``world_size * (sum_actor_local/A + value_coef *
          sum_value_local/V)`` with ``A``/``V`` the global valid counts, so
          DDP's post-backward average reproduces the single-process global
          objective;
        * gradients are unscaled, checked for finiteness on every rank, clipped
          and applied by exactly one shared Adam step, or skipped together when
          the global finite flag is false or the minibatch holds no valid rows;
        * epoch KL is aggregated across ranks and the target-KL early stop is a
          single global decision, so ranks never diverge on epoch count.
        """
        from contextlib import nullcontext

        from agent.ppo.ddp import all_reduce_flag, all_reduce_tensor, ddp_rank_loss

        world_size = max(self.world_size, 1)
        distributed = (
            world_size > 1
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        rows = list(plan.rows)
        update_started = time.perf_counter()
        report = self._new_report(len(rows), plan.actor_rows, committed_matches)
        self.model.train()
        sum_keys = (
            "policy_sum", "value_sum", "entropy_sum", "uniform_kl_sum",
            "kl_sum", "ratio_sum", "clip_sum", "actor_count", "value_count",
        )
        totals = {key: torch.zeros((), dtype=torch.float32, device=self.device) for key in sum_keys}
        profile = self._blank_profile()
        profile["ddp_sync_steps"] = 0.0
        batch_size = max(
            1,
            self.config.per_rank_minibatch_size
            if self.config.per_rank_minibatch_size > 0
            else self.config.global_minibatch_size // world_size,
        )
        microbatch = max(1, self.config.microbatch_size)
        # Fixed micro-step count per minibatch: the plan's requirement. The old
        # implementation derived the micro list from *valid* rows, so a rank
        # that ran out of rows skipped its backward (and the DDP reduction)
        # while its peer kept going -- the v3 collective mismatch.
        micro_steps = max(1, (batch_size + microbatch - 1) // microbatch)
        local_batches = (len(rows) + batch_size - 1) // batch_size
        batch_count = local_batches
        if distributed:
            count_tensor = torch.tensor([local_batches], dtype=torch.long, device=self.device)
            torch.distributed.all_reduce(count_tensor, op=torch.distributed.ReduceOp.MAX)
            batch_count = int(count_tensor.item())
        if batch_count == 0:
            self._finish_report(report, totals, sum_keys)
            profile["total"] = time.perf_counter() - update_started
            self.profile = profile
            return report
        if not rows and fallback_batch is None:
            raise RuntimeError(
                "rank has no local rows; pass fallback_batch (masked placeholder "
                "rows) so the fixed-step DDP protocol keeps its collective "
                "sequence without duplicating training data"
            )

        for epoch in range(self.config.ppo_epochs):
            epoch_sums = {key: torch.zeros((), dtype=torch.float32, device=self.device) for key in sum_keys}
            epoch_batches = 0
            start = time.perf_counter()
            order = (
                torch.randperm(len(rows), generator=generator)
                if rows
                else torch.zeros(0, dtype=torch.long)
            )
            for batch_index in range(batch_count):
                if rows:
                    begin = batch_index * batch_size
                    if begin < len(order):
                        chunk = order[begin:begin + batch_size]
                        real = int(chunk.numel())
                    else:
                        chunk = order[:1]
                        real = 0
                    if real == batch_size:
                        indices = chunk
                        valid = torch.ones(batch_size, dtype=torch.bool)
                    else:
                        indices = torch.cat(
                            [chunk[:real], chunk[:1].repeat(batch_size - real)]
                        )
                        valid = torch.cat(
                            [
                                torch.ones(real, dtype=torch.bool),
                                torch.zeros(batch_size - real, dtype=torch.bool),
                            ]
                        )
                    minibatch = self._plan_minibatch(plan, indices, valid)
                else:
                    take = min(len(fallback_batch), batch_size)
                    repeats = (batch_size + take - 1) // take
                    selection = torch.arange(take, dtype=torch.long).repeat(repeats)[:batch_size]
                    minibatch = fallback_batch.select(selection)
                    minibatch = replace(
                        minibatch,
                        row_valid=torch.zeros(batch_size, dtype=torch.bool),
                        advantages=torch.zeros(batch_size, dtype=torch.float32),
                    )
                profile["minibatch_select"] += time.perf_counter() - start
                start = time.perf_counter()
                minibatch = minibatch.to(self.device)
                counts = torch.stack(
                    [
                        (minibatch.actor_mask & minibatch.row_valid).sum().float(),
                        minibatch.row_valid.sum().float(),
                    ]
                )
                all_reduce_tensor(counts)
                actor_total = float(counts[0].item())
                value_total = float(counts[1].item())
                profile["h2d"] += time.perf_counter() - start
                start = time.perf_counter()
                self.optimizer.zero_grad(set_to_none=True)
                if communication is not None:
                    communication.reset()
                for micro_index in range(micro_steps):
                    micro_begin = micro_index * microbatch
                    micro_end = min(micro_begin + microbatch, batch_size)
                    micro = minibatch.narrow(micro_begin, micro_end)
                    is_last = micro_index == micro_steps - 1
                    context = (
                        communication.context(synchronize=is_last)
                        if communication is not None
                        else nullcontext()
                    )
                    with context:
                        forward_start = time.perf_counter()
                        terms = self._forward_terms(micro)
                        profile["forward"] += time.perf_counter() - forward_start
                        backward_start = time.perf_counter()
                        loss = ddp_rank_loss(
                            terms["loss_unscaled"] * terms["actor_count"],
                            terms["value_mean"] * terms["value_count"],
                            actor_total,
                            value_total,
                            world_size,
                            self.config.value_coefficient,
                        )
                        if not loss.requires_grad:  # pragma: no cover - defensive
                            raise RuntimeError(
                                "DDP micro loss lost its graph; padding-only micros "
                                "must stay graph-connected for the reducer"
                            )
                        self.scaler.scale(loss).backward()
                        profile["backward"] += time.perf_counter() - backward_start
                    metrics_start = time.perf_counter()
                    for key in sum_keys:
                        epoch_sums[key] = epoch_sums[key] + terms[key].float()
                    profile["metrics"] += time.perf_counter() - metrics_start
                if communication is not None:
                    profile["ddp_sync_steps"] = float(communication.sync_calls)

                optimizer_start = time.perf_counter()
                self.scaler.unscale_(self.optimizer)
                finite = torch.ones((), dtype=torch.float32, device=self.device)
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        finite = finite * torch.isfinite(parameter.grad).all().float()
                all_reduce_flag(finite)
                has_rows = actor_total > 0.0 or value_total > 0.0
                globally_finite = bool(finite.item())
                do_step = globally_finite and has_rows
                if do_step:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                else:
                    grad_norm = torch.zeros((), device=self.device)
                    self.optimizer.zero_grad(set_to_none=True)
                self._update_distributed_scaler(globally_finite)
                profile["optimizer"] += time.perf_counter() - optimizer_start
                if do_step:
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
            stacked = torch.stack([epoch_sums[key].detach().float() for key in sum_keys])
            all_reduce_tensor(stacked)
            for key, value in zip(sum_keys, stacked):
                epoch_sums[key] = value
                totals[key] = totals[key] + value
            epoch_actor = float(epoch_sums["actor_count"].clamp_min(1).item())
            epoch_kl = float(epoch_sums["kl_sum"].item()) / epoch_actor
            report.epoch_approx_kl.append(epoch_kl)
            report.epochs_run = epoch + 1
            if epoch_kl > self.config.target_approx_kl:
                report.stopped_early = True
                break

        self._finish_report(report, totals, sum_keys)
        profile["total"] = time.perf_counter() - update_started
        self.profile = profile
        return report

    def update_manual_allreduce(
        self,
        plan: StreamingPlan,
        committed_matches: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        fallback_batch: Optional[RolloutBatch] = None,
    ) -> UpdateReport:
        """Manual FP32 gradient-SUM executor (plan §7.1).

        Same fixed-step micro protocol as :meth:`update_ddp`, but the model is
        not wrapped in DDP: each rank's local loss is
        ``S_actor_local/A + value_coef*S_value_local/V`` (no world-size factor),
        gradients are unscaled, flattened in a fixed parameter order (exact
        zeros for missing grads), summed with exactly one ``all_reduce(SUM)``,
        written back, clipped globally and applied by one shared Adam step.
        """
        from contextlib import nullcontext

        from agent.ppo.ddp import (
            all_reduce_flag,
            all_reduce_tensor,
            assign_flat_gradients,
            flatten_gradients,
        )

        world_size = max(self.world_size, 1)
        distributed = (
            world_size > 1
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        rows = list(plan.rows)
        update_started = time.perf_counter()
        report = self._new_report(len(rows), plan.actor_rows, committed_matches)
        self.model.train()
        sum_keys = (
            "policy_sum", "value_sum", "entropy_sum", "uniform_kl_sum",
            "kl_sum", "ratio_sum", "clip_sum", "actor_count", "value_count",
        )
        totals = {key: torch.zeros((), dtype=torch.float32, device=self.device) for key in sum_keys}
        profile = self._blank_profile()
        profile["manual_allreduce_seconds"] = 0.0
        profile["manual_flat_bytes"] = 0.0
        batch_size = max(
            1,
            self.config.per_rank_minibatch_size
            if self.config.per_rank_minibatch_size > 0
            else self.config.global_minibatch_size // world_size,
        )
        microbatch = max(1, self.config.microbatch_size)
        micro_steps = max(1, (batch_size + microbatch - 1) // microbatch)
        local_batches = (len(rows) + batch_size - 1) // batch_size
        batch_count = local_batches
        if distributed:
            count_tensor = torch.tensor([local_batches], dtype=torch.long, device=self.device)
            torch.distributed.all_reduce(count_tensor, op=torch.distributed.ReduceOp.MAX)
            batch_count = int(count_tensor.item())
        if batch_count == 0:
            self._finish_report(report, totals, sum_keys)
            profile["total"] = time.perf_counter() - update_started
            self.profile = profile
            return report
        if not rows and fallback_batch is None:
            raise RuntimeError(
                "rank has no local rows; pass fallback_batch (masked placeholder "
                "rows) so the manual executor keeps its collective sequence"
            )

        for epoch in range(self.config.ppo_epochs):
            epoch_sums = {key: torch.zeros((), dtype=torch.float32, device=self.device) for key in sum_keys}
            epoch_batches = 0
            order = (
                torch.randperm(len(rows), generator=generator)
                if rows
                else torch.zeros(0, dtype=torch.long)
            )
            for batch_index in range(batch_count):
                if rows:
                    begin = batch_index * batch_size
                    if begin < len(order):
                        chunk = order[begin:begin + batch_size]
                        real = int(chunk.numel())
                    else:
                        chunk = order[:1]
                        real = 0
                    if real == batch_size:
                        indices = chunk
                        valid = torch.ones(batch_size, dtype=torch.bool)
                    else:
                        indices = torch.cat(
                            [chunk[:real], chunk[:1].repeat(batch_size - real)]
                        )
                        valid = torch.cat(
                            [
                                torch.ones(real, dtype=torch.bool),
                                torch.zeros(batch_size - real, dtype=torch.bool),
                            ]
                        )
                    minibatch = self._plan_minibatch(plan, indices, valid)
                else:
                    take = min(len(fallback_batch), batch_size)
                    repeats = (batch_size + take - 1) // take
                    selection = torch.arange(take, dtype=torch.long).repeat(repeats)[:batch_size]
                    minibatch = fallback_batch.select(selection)
                    minibatch = replace(
                        minibatch,
                        row_valid=torch.zeros(batch_size, dtype=torch.bool),
                        advantages=torch.zeros(batch_size, dtype=torch.float32),
                    )
                minibatch = minibatch.to(self.device)
                counts = torch.stack(
                    [
                        (minibatch.actor_mask & minibatch.row_valid).sum().float(),
                        minibatch.row_valid.sum().float(),
                    ]
                )
                all_reduce_tensor(counts)
                actor_total = float(counts[0].item())
                value_total = float(counts[1].item())

                self.optimizer.zero_grad(set_to_none=True)
                for micro_index in range(micro_steps):
                    micro_begin = micro_index * microbatch
                    micro_end = min(micro_begin + microbatch, batch_size)
                    micro = minibatch.narrow(micro_begin, micro_end)
                    forward_start = time.perf_counter()
                    terms = self._forward_terms(micro)
                    profile["forward"] += time.perf_counter() - forward_start
                    backward_start = time.perf_counter()
                    loss = (
                        terms["loss_unscaled"] * terms["actor_count"] / max(actor_total, 1.0)
                        + self.config.value_coefficient
                        * terms["value_mean"]
                        * terms["value_count"]
                        / max(value_total, 1.0)
                    )
                    self.scaler.scale(loss).backward()
                    profile["backward"] += time.perf_counter() - backward_start
                    metrics_start = time.perf_counter()
                    for key in sum_keys:
                        epoch_sums[key] = epoch_sums[key] + terms[key].float()
                    profile["metrics"] += time.perf_counter() - metrics_start

                optimizer_start = time.perf_counter()
                self.scaler.unscale_(self.optimizer)
                flat = flatten_gradients(self.model)
                if distributed:
                    allreduce_start = time.perf_counter()
                    torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
                    profile["manual_allreduce_seconds"] += (
                        time.perf_counter() - allreduce_start
                    )
                    profile["manual_flat_bytes"] = float(flat.numel() * flat.element_size())
                assign_flat_gradients(self.model, flat)
                finite = torch.ones((), dtype=torch.float32, device=self.device)
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        finite = finite * torch.isfinite(parameter.grad).all().float()
                all_reduce_flag(finite)
                has_rows = actor_total > 0.0 or value_total > 0.0
                globally_finite = bool(finite.item())
                do_step = globally_finite and has_rows
                if do_step:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                else:
                    grad_norm = torch.zeros((), device=self.device)
                    self.optimizer.zero_grad(set_to_none=True)
                self._update_distributed_scaler(globally_finite)
                profile["optimizer"] += time.perf_counter() - optimizer_start
                if do_step:
                    self.optimizer_steps += 1
                else:
                    report.optimizer_steps_skipped += 1
                report.grad_norm = float(grad_norm.detach())
                report.grad_norm_max = max(report.grad_norm_max, report.grad_norm)
                report.scaler_scale = float(self.scaler.get_scale())
                report.optimizer_steps = self.optimizer_steps
                epoch_batches += 1

            if epoch_batches == 0:
                break
            stacked = torch.stack([epoch_sums[key].detach().float() for key in sum_keys])
            all_reduce_tensor(stacked)
            for key, value in zip(sum_keys, stacked):
                epoch_sums[key] = value
                totals[key] = totals[key] + value
            epoch_actor = float(epoch_sums["actor_count"].clamp_min(1).item())
            epoch_kl = float(epoch_sums["kl_sum"].item()) / epoch_actor
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
                minibatch = self._plan_minibatch(plan, indices, valid)
                profile["minibatch_select"] += time.perf_counter() - start
                for key, value in getattr(plan.buffer, "profile", {}).items():
                    profile[f"materialize_{key}"] = (
                        profile.get(f"materialize_{key}", 0.0) + float(value)
                    )
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
