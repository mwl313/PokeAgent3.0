"""PA3-8M: the single randomly initialized Entity Transformer policy/value net.

Contract highlights (Full Spec 1.1 §6, ``configs/train.yaml::model``):

* the observation state is encoded **once** per request; branches reuse the
  cached token representations and never re-run the encoder,
* the conditional prefix scorer receives the previously selected actions of the
  *same request* through a 320-unit ``GRUCell``; there is no recurrent match
  memory,
* the value head reads the observation/global representation only and never
  sees the selected action prefix,
* illegal candidates are masked before sampling and before any log-probability
  computation,
* all probability/log-probability math is FP32 even when the encoder runs under
  FP16 autocast
* no external LLM or search is part of the battle loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import nn

from agent.model.config import PA3Config
from agent.model.encoder import TokenEncoder
from agent.model.scorer import ConditionalPrefixScorer, masked_log_softmax, probabilities
from agent.types.observation import ObservationBatch
from agent.types.requests import BranchCandidatesBatch


@dataclass
class EncodedState:
    """Cached per-request encoding (produced exactly once per request)."""

    tokens: torch.Tensor  # [B, T, D]
    global_repr: torch.Tensor  # [B, D]
    token_mask: torch.Tensor  # [B, T] bool
    layout: object


@dataclass
class BranchEvaluation:
    """Per-branch and per-request statistics of a candidate tree."""

    logits: torch.Tensor  # [B, K, P]
    logprob_selected: torch.Tensor  # [B, K]
    entropy_normalized: torch.Tensor  # [B, K]  (0 for K < 2)
    uniform_kl_normalized: torch.Tensor  # [B, K]  (0 for K < 2)
    branch_k: torch.Tensor  # [B, K]
    branch_valid: torch.Tensor  # [B, K] bool (the request has this branch)
    branch_active: torch.Tensor  # [B, K] bool (K >= 2)
    request_logprob: torch.Tensor  # [B]
    request_entropy: torch.Tensor  # [B]
    request_uniform_kl: torch.Tensor  # [B]
    actor_active: torch.Tensor  # [B] bool


@dataclass
class SamplingResult:
    """A sampled joint action plus everything the buffer needs to record."""

    selected: torch.Tensor  # [B, K] long
    logprob_selected: torch.Tensor  # [B, K]
    request_logprob: torch.Tensor  # [B]
    entropy_normalized: torch.Tensor  # [B, K]
    uniform_kl_normalized: torch.Tensor  # [B, K]
    branch_k: torch.Tensor  # [B, K]
    actor_active: torch.Tensor  # [B] bool
    probabilities: torch.Tensor  # [B, K, P]

    def prefix(self, branch_count: int) -> torch.Tensor:
        """Selected indices for a request with ``branch_count`` real branches.

        ``selected`` is padded to the branch capacity; a collector stores only
        the real prefix of the request it sampled.
        """
        return self.selected[:, : int(branch_count)]


@dataclass
class LevelSamplingResult:
    """Sampling result when each branch level carries its own candidate table.

    The native engine's mask for branch ``j`` is computed *after* the branches
    selected before it (preview picks cannot repeat a member, a used Mega flag
    disappears from the second slot, and so on), so a faithful collector cannot
    pre-build all branch tables in one pass. ``sample_levels`` walks the levels
    in order with the prefix-dependent table for each one.
    """

    selected: torch.Tensor  # [B, L] long
    logprob_selected: torch.Tensor  # [B, L]
    entropy_normalized: torch.Tensor  # [B, L]
    uniform_kl_normalized: torch.Tensor  # [B, L]
    branch_k: torch.Tensor  # [B, L] long
    request_logprob: torch.Tensor  # [B]
    actor_active: torch.Tensor  # [B] bool


@dataclass
class LevelStepResult:
    """One branch level of a sequential sample."""

    pick: torch.Tensor  # [B] long
    logprob: torch.Tensor  # [B]
    entropy_normalized: torch.Tensor  # [B]
    uniform_kl_normalized: torch.Tensor  # [B]
    branch_k: torch.Tensor  # [B] long
    hidden: torch.Tensor  # [B, D]


def assemble_level_result(steps: Sequence[LevelStepResult]) -> LevelSamplingResult:
    """Combine per-level sampling steps into one request-level result."""
    if not steps:
        raise ValueError("no sampled levels")
    import torch as _torch

    selected = _torch.stack([step.pick for step in steps], dim=1)
    logprob = _torch.stack([step.logprob for step in steps], dim=1)
    entropy = _torch.stack([step.entropy_normalized for step in steps], dim=1)
    uniform_kl = _torch.stack([step.uniform_kl_normalized for step in steps], dim=1)
    branch_k = _torch.stack([step.branch_k for step in steps], dim=1)
    return LevelSamplingResult(
        selected=selected,
        logprob_selected=logprob,
        entropy_normalized=entropy,
        uniform_kl_normalized=uniform_kl,
        branch_k=branch_k,
        request_logprob=logprob.sum(dim=-1),
        actor_active=(branch_k >= 2).any(dim=-1),
    )


class PA3Model(nn.Module):
    """The one PA3-8M network (policy and value share the encoder)."""

    def __init__(self, config: Optional[PA3Config] = None) -> None:
        super().__init__()
        self.config = config or PA3Config()
        self.config.validate()
        self.encoder = TokenEncoder(self.config)
        self.scorer = ConditionalPrefixScorer(self.config)
        self.value_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.critic_hidden),
            nn.GELU(),
            nn.Linear(self.config.critic_hidden, 1),
        )
        self.reset_parameters()

    # -- initialization ---------------------------------------------------
    def reset_parameters(self) -> None:
        self.encoder.embedding.reset_parameters()
        for block in self.encoder.blocks:
            block.reset_parameters()
        nn.init.ones_(self.encoder.final_norm.weight)
        nn.init.zeros_(self.encoder.final_norm.bias)
        self.scorer.reset_parameters()
        nn.init.xavier_uniform_(self.value_head[0].weight)
        nn.init.zeros_(self.value_head[0].bias)
        nn.init.xavier_uniform_(self.value_head[2].weight)
        nn.init.zeros_(self.value_head[2].bias)
        # Small final policy/value gains keep the initial policy near uniform.
        with torch.no_grad():
            self.value_head[2].weight.mul_(self.config.policy_output_init_gain)

    # -- introspection ----------------------------------------------------
    def parameter_count(self, trainable_only: bool = True) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (p for p in parameters if p.requires_grad)
        return sum(p.numel() for p in parameters)

    # -- encode -----------------------------------------------------------
    def forward(self, observation: ObservationBatch, candidates: Optional[BranchCandidatesBatch] = None,
                selected: Optional[torch.Tensor] = None, mode: str = "learner"):
        """DDP-visible forward entry.

        `DistributedDataParallel` requires the wrapped module's own ``forward``
        to own the *whole* graph; calling ``encode``/``value``/``evaluate``
        directly on the inner module uses parameters outside that entry and
        makes DDP's reducer mark them ready twice. When a candidate batch is
        given the learner forward returns the full ``(encoded, values,
        evaluation)`` triple; without it only the encoder/value path runs.
        """
        encoded = self.encode(observation)
        values = self.value(encoded)
        if candidates is not None:
            evaluation = self.evaluate_encoded(encoded, candidates, selected=selected)
            return encoded, values, evaluation
        return encoded, values

    def encode(self, observation: ObservationBatch) -> EncodedState:
        tokens = self.encoder(observation)
        global_index = observation.layout.GLOBAL
        token_mask = observation.token_mask
        weights = token_mask.unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        if tokens.shape[1] > global_index:
            # Keep the defensive batch-wide fallback on the device. Converting
            # this reduction to bool forced a CUDA synchronization for every
            # learner microbatch and collector encoding.
            global_repr = torch.where(
                token_mask[:, global_index].any(), tokens[:, global_index], pooled
            )
        else:  # pragma: no cover - defensive contract fallback
            global_repr = pooled
        return EncodedState(
            tokens=tokens,
            global_repr=global_repr,
            token_mask=token_mask,
            layout=observation.layout,
        )

    def value(self, encoded: EncodedState) -> torch.Tensor:
        return self.value_head(encoded.global_repr).squeeze(-1)

    # -- branch scoring ----------------------------------------------------
    def _branch_stats(
        self, logits: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Probability/log-probability math is always FP32, even if the encoder
        # ran under FP16 autocast.
        logits = logits.float()
        log_prob = masked_log_softmax(logits, mask)
        probs = probabilities(log_prob, mask)
        # The PPO learner adds ``-entropy_coefficient * entropy`` and
        # ``uniform_kl_coefficient * uniform_kl`` to the loss, so both terms
        # must stay attached to the scorer graph. Detaching either factor (or
        # both) silently removed the regularizer gradient while still
        # reporting the correct scalar values; see
        # tests/agent/test_policy_regularizer_gradients.py.
        entropy = -(probs * log_prob).sum(dim=-1)
        k = mask.sum(dim=-1)
        safe_k = k.clamp_min(1).to(log_prob.dtype)
        log_k = torch.log(safe_k)
        normalizer = torch.where(k >= 2, log_k, torch.ones_like(log_k))
        entropy_normalized = torch.where(k >= 2, entropy / normalizer, torch.zeros_like(entropy))
        mean_log_prob = (log_prob * mask).sum(dim=-1) / safe_k
        # KL(U || pi) = sum_u (1/K)(log(1/K) - log pi(u)) = -log K - mean log pi.
        # The direction (uniform -> policy) and the log K normalization are
        # pinned by the config; the term is 0 for a uniform policy and grows as
        # the policy collapses onto a single candidate.
        uniform_kl = (-log_k - mean_log_prob) / normalizer
        uniform_kl = torch.where(
            k >= 2, uniform_kl.expand_as(entropy), torch.zeros_like(entropy)
        )
        return log_prob, probs, entropy_normalized, uniform_kl

    def _run_branches(
        self,
        encoded: EncodedState,
        candidates: BranchCandidatesBatch,
        selected: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
        deterministic: bool = False,
        return_final_hidden: bool = True,
    ) -> tuple[BranchEvaluation, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared branch loop for evaluation and sampling.

        The encoder is *not* called here: the caller passes the cached
        :class:`EncodedState` so a multi-branch request encodes the state once.
        """
        tokens = encoded.tokens
        batch = tokens.shape[0]
        branch_capacity = candidates.action_ids.shape[1]
        padding = candidates.action_ids.shape[2]
        device = tokens.device
        dtype = tokens.dtype if tokens.dtype.is_floating_point else torch.float32

        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        hidden = self.scorer.initial_state(batch, device=device, dtype=dtype)
        logits_out = torch.full(
            (batch, branch_capacity, padding), float("-inf"), device=device, dtype=dtype
        )
        logprob_selected = torch.zeros((batch, branch_capacity), device=device, dtype=dtype)
        entropy_out = torch.zeros((batch, branch_capacity), device=device, dtype=dtype)
        uniform_kl_out = torch.zeros((batch, branch_capacity), device=device, dtype=dtype)
        probs_out = torch.zeros((batch, branch_capacity, padding), device=device, dtype=dtype)
        selected_out = torch.zeros((batch, branch_capacity), device=device, dtype=torch.long)

        for branch in range(branch_capacity):
            branch_valid = candidates.branch_valid[:, branch]
            action_ids = candidates.action_ids[:, branch]
            mask = candidates.mask[:, branch]
            # Illegal candidates must be exactly zero-probability: the mask is
            # applied inside score() before any sampling or log-prob math.
            keys = self.scorer.build_keys(
                tokens,
                action_ids,
                candidates.entity_token[:, branch],
                candidates.move_token[:, branch],
            )
            logits = self.scorer.score_keys(keys, hidden, mask)
            if temperature != 1.0:
                logits = logits / temperature
            log_prob, probs, entropy, uniform_kl = self._branch_stats(logits, mask)
            logits_out[:, branch] = logits
            probs_out[:, branch] = probs

            if selected is None:
                any_legal = mask.any(dim=-1, keepdim=True)
                safe_probs = probs.clone()
                # Rows without a legal candidate (padded branches) must not
                # reach multinomial; the branch-valid mask discards them.
                no_legal = (~any_legal).squeeze(-1)
                safe_probs[:, 0] += no_legal.to(safe_probs.dtype)
                if deterministic:
                    pick = torch.argmax(
                        torch.where(mask, logits, torch.full_like(logits, float("-inf"))),
                        dim=-1,
                    )
                    pick = torch.where(
                        any_legal.squeeze(-1), pick, torch.zeros_like(pick)
                    )
                else:
                    pick = torch.multinomial(
                        safe_probs, num_samples=1, generator=generator
                    ).squeeze(-1)
                pick = torch.where(branch_valid, pick, torch.zeros_like(pick))
            else:
                pick = selected[:, branch].clamp(0, padding - 1)

            gathered = torch.gather(log_prob, 1, pick.unsqueeze(-1)).squeeze(-1)
            logprob_selected[:, branch] = torch.where(
                branch_valid, gathered, torch.zeros_like(gathered)
            )
            entropy_out[:, branch] = torch.where(
                branch_valid, entropy, torch.zeros_like(entropy)
            )
            uniform_kl_out[:, branch] = torch.where(
                branch_valid, uniform_kl, torch.zeros_like(uniform_kl)
            )
            selected_out[:, branch] = pick

            # Public evaluate/sample consume no final hidden state. Their last
            # GRU update cannot affect a score; direct callers requesting the
            # private helper's hidden state retain its complete prefix.
            if return_final_hidden or branch + 1 < branch_capacity:
                safe_pick = pick.clamp(0, padding - 1)
                selected_keys = torch.gather(
                    keys, 1, safe_pick[:, None, None].expand(-1, 1, keys.shape[-1])
                ).squeeze(1)
                updated = self.scorer.advance_keys(selected_keys, hidden)
                hidden = torch.where(branch_valid.unsqueeze(-1), updated, hidden)

        k = candidates.branch_k
        branch_active = candidates.branch_valid & (k >= 2)
        active_count = branch_active.sum(dim=-1).clamp_min(1)
        request_entropy = (entropy_out * branch_active).sum(dim=-1) / active_count
        request_uniform_kl = (uniform_kl_out * branch_active).sum(dim=-1) / active_count
        request_logprob = logprob_selected.sum(dim=-1)
        actor_active = branch_active.any(dim=-1)

        evaluation = BranchEvaluation(
            logits=logits_out,
            logprob_selected=logprob_selected,
            entropy_normalized=entropy_out,
            uniform_kl_normalized=uniform_kl_out,
            branch_k=k,
            branch_valid=candidates.branch_valid,
            branch_active=branch_active,
            request_logprob=request_logprob,
            request_entropy=request_entropy,
            request_uniform_kl=request_uniform_kl,
            actor_active=actor_active,
        )
        return evaluation, hidden, probs_out, selected_out

    # -- public API ---------------------------------------------------------
    def evaluate(
        self,
        observation: ObservationBatch,
        candidates: BranchCandidatesBatch,
        selected: Optional[torch.Tensor] = None,
    ) -> tuple[EncodedState, BranchEvaluation]:
        """Recompute branch log-probabilities/entropy under the current weights.

        ``selected`` defaults to the indices stored on ``candidates`` (the
        rollout's preserved prefix).  The state is encoded once.
        """
        if len(observation) != candidates.action_ids.shape[0]:
            raise ValueError("observation and candidate batch sizes differ")
        encoded = self.encode(observation)
        return encoded, self.evaluate_encoded(encoded, candidates, selected)

    def evaluate_encoded(
        self,
        encoded: EncodedState,
        candidates: BranchCandidatesBatch,
        selected: Optional[torch.Tensor] = None,
    ) -> BranchEvaluation:
        """Recompute branch statistics from an already-cached encoding."""
        selected_tensor = selected if selected is not None else candidates.selected
        if selected_tensor is None:
            raise ValueError("evaluate() needs a selected prefix")
        evaluation, _, _, _ = self._run_branches(
            encoded, candidates, selected=selected_tensor, deterministic=True,
            return_final_hidden=False,
        )
        return evaluation

    @torch.no_grad()
    def sample(
        self,
        observation: ObservationBatch,
        candidates: BranchCandidatesBatch,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
        deterministic: bool = False,
    ) -> SamplingResult:
        """Categorical sampling at temperature 1.0 (spec default)."""
        if len(observation) != candidates.action_ids.shape[0]:
            raise ValueError("observation and candidate batch sizes differ")
        encoded = self.encode(observation)
        evaluation, _, probs, selected = self._run_branches(
            encoded,
            candidates,
            selected=None,
            temperature=temperature,
            generator=generator,
            deterministic=deterministic,
            return_final_hidden=False,
        )
        return SamplingResult(
            selected=selected,
            logprob_selected=evaluation.logprob_selected,
            request_logprob=evaluation.request_logprob,
            entropy_normalized=evaluation.entropy_normalized,
            uniform_kl_normalized=evaluation.uniform_kl_normalized,
            branch_k=evaluation.branch_k,
            actor_active=evaluation.actor_active,
            probabilities=probs,
        )

    @torch.no_grad()
    def sample_levels(
        self,
        encoded: EncodedState,
        tables: Sequence[BranchCandidatesBatch],
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> LevelSamplingResult:
        """Sequential sampling with one candidate table per level.

        ``tables[j]`` must be the engine mask for branch ``j`` computed after
        the branches already selected, as a single-branch batch (``[B, 1, P]``).
        The reference recomputation path (``evaluate_encoded`` over the stored
        per-level tables and the stored selected prefix) produces exactly the
        log-probabilities this method sampled, so the PPO ratio is 1 at the
        start of the first epoch.
        """
        hidden = self.scorer.initial_state(
            encoded.tokens.shape[0],
            device=encoded.tokens.device,
            dtype=encoded.tokens.dtype if encoded.tokens.dtype.is_floating_point else torch.float32,
        )
        steps = []
        for table in tables:
            # The GRU prefix state must advance with each selected candidate;
            # a list comprehension here would silently reuse the initial state.
            step = self.sample_level_step(
                encoded, hidden, table, temperature=temperature, generator=generator
            )
            steps.append(step)
            hidden = step.hidden
        return assemble_level_result(steps)

    @torch.no_grad()
    def sample_level_step(
        self,
        encoded: EncodedState,
        hidden: torch.Tensor,
        table: BranchCandidatesBatch,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> LevelStepResult:
        """Score and sample one branch level from its prefix-dependent table.

        The collector interleaves this with the engine's ``candidates_batch``:
        level ``j+1``'s table needs the action sampled at level ``j``.
        """
        batch = encoded.tokens.shape[0]
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if table.action_ids.shape[0] != batch or table.action_ids.shape[1] != 1:
            raise ValueError(
                f"level table must be [B, 1, P], got {tuple(table.action_ids.shape)}"
            )
        action_ids = table.action_ids[:, 0]
        mask = table.mask[:, 0]
        entity = table.entity_token[:, 0]
        move = table.move_token[:, 0]
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("level has a request without a legal candidate")
        keys = self.scorer.build_keys(encoded.tokens, action_ids, entity, move)
        logits = self.scorer.score_keys(keys, hidden, mask)
        if temperature != 1.0:
            logits = logits / temperature
        log_prob, probs, entropy, uniform_kl = self._branch_stats(logits, mask)
        pick = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
        gathered = torch.gather(log_prob, 1, pick.unsqueeze(-1)).squeeze(-1)
        selected_keys = torch.gather(
            keys, 1, pick[:, None, None].expand(-1, 1, keys.shape[-1])
        ).squeeze(1)
        next_hidden = self.scorer.advance_keys(selected_keys, hidden)
        return LevelStepResult(
            pick=pick,
            logprob=gathered.detach(),
            entropy_normalized=entropy.detach(),
            uniform_kl_normalized=uniform_kl.detach(),
            branch_k=mask.sum(dim=-1).to(torch.long),
            hidden=next_hidden,
        )
