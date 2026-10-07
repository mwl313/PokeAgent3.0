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
from typing import Optional

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
    def encode(self, observation: ObservationBatch) -> EncodedState:
        tokens = self.encoder(observation)
        global_index = observation.layout.GLOBAL
        token_mask = observation.token_mask
        if tokens.shape[1] > global_index and bool(token_mask[:, global_index].any()):
            global_repr = tokens[:, global_index]
        else:  # pragma: no cover - defensive contract fallback
            weights = token_mask.unsqueeze(-1).to(tokens.dtype)
            global_repr = (tokens * weights).sum(1) / weights.sum(1).clamp_min(1.0)
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
        entropy = -(probs.detach() * log_prob.detach()).sum(dim=-1)
        k = mask.sum(dim=-1)
        safe_k = k.clamp_min(1).to(log_prob.dtype)
        log_k = torch.log(safe_k)
        normalizer = torch.where(k >= 2, log_k, torch.ones_like(log_k))
        entropy_normalized = torch.where(k >= 2, entropy / normalizer, torch.zeros_like(entropy))
        mean_log_prob = (log_prob.detach() * mask).sum(dim=-1) / safe_k
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
            logits = self.scorer.score(
                tokens,
                hidden,
                action_ids,
                mask,
                candidates.entity_token[:, branch],
                candidates.move_token[:, branch],
            )
            if temperature != 1.0:
                logits = logits / temperature
            log_prob, probs, entropy, uniform_kl = self._branch_stats(logits, mask)
            logits_out[:, branch] = logits
            probs_out[:, branch] = probs

            if selected is None:
                any_legal = mask.any(dim=-1, keepdim=True)
                safe_probs = torch.where(any_legal, probs, torch.zeros_like(probs))
                # Rows without a legal candidate (padded branches) must not
                # reach multinomial; the branch-valid mask discards them.
                no_legal = (~any_legal).squeeze(-1)
                if bool(no_legal.any()):
                    safe_probs = safe_probs.clone()
                    safe_probs[no_legal, 0] = 1.0
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

            # GRU prefix update from the selected candidate only.
            safe_pick = pick.clamp(0, padding - 1)
            step_action = torch.gather(
                action_ids,
                1,
                safe_pick.view(-1, 1, 1).expand(-1, 1, action_ids.shape[-1]),
            ).squeeze(1)
            step_entity = torch.gather(candidates.entity_token[:, branch], 1, safe_pick.unsqueeze(-1)).squeeze(-1)
            step_move = torch.gather(candidates.move_token[:, branch], 1, safe_pick.unsqueeze(-1)).squeeze(-1)
            updated = self.scorer.advance(tokens, hidden, step_action, step_entity, step_move)
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
            encoded, candidates, selected=selected_tensor, deterministic=True
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
