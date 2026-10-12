"""Versioned, inference-only PPO adapter for Track C coordinators."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from agent.model import PA3Config, build_model
from agent.evaluation.identity import canonical_model_digest, sha256_file
from agent.model.pa3_model import PA3Model
from agent.model.precision import get_precision_contract
from agent.train.native_collector import NativeCollector, NativeCollectorConfig
from agent.types.observation import ObservationLayout


INTERFACE_VERSION = "pa3-frozen-policy-v1"
OBSERVATION_SCHEMA = "pa3-entity-observation-v1"
ACTION_SCHEMA = "atomic-action-v1"


@dataclass(frozen=True)
class PolicyIdentity:
    interface_version: str
    model_digest: str
    checkpoint_sha256: str | None
    observation_schema: str
    action_schema: str
    precision_contract: str = "pa3-precision-A-v1"
    precision_stability: str = "default"


class FrozenPPOPolicy:
    """Expose legal-action priors and values without training state."""

    def __init__(self, model: PA3Model, *, device="cpu", identity: PolicyIdentity | None = None):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.identity = identity or PolicyIdentity(
            INTERFACE_VERSION, canonical_model_digest(self.model.state_dict()), None, OBSERVATION_SCHEMA, ACTION_SCHEMA,
            getattr(self.model, "precision_contract", get_precision_contract("A")).identity, "default"
        )
        self._helper = NativeCollector.__new__(NativeCollector)
        self._helper.config = NativeCollectorConfig(record_rows=False)
        self._helper.layout = ObservationLayout()
        self._helper.device = self.device

    @classmethod
    def from_checkpoint(cls, path, *, device="cpu", config: PA3Config | None = None):
        path = os.path.abspath(os.fspath(path))
        if "://" in path or not os.path.isfile(path):
            raise ValueError("frozen PPO checkpoints must be trusted local files")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("learner", {}).get("model") if isinstance(payload, dict) else None
        if state is None:
            raise ValueError("checkpoint does not contain an experimental learner model")
        model = build_model(config or PA3Config(), device=device)
        experiment = payload.get("experiment", {}) if isinstance(payload, dict) else {}
        precision_mode = experiment.get("precision_mode", "A")
        precision_stability = experiment.get("precision_stability", "default")
        if precision_stability not in ("default", "fp16-reduction-off"):
            raise ValueError(f"unsupported precision stability contract: {precision_stability!r}")
        if precision_stability == "fp16-reduction-off":
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        model.configure_precision(precision_mode)
        model.load_state_dict(state, strict=True)
        file_hash = sha256_file(path)
        adapter = cls(model, device=device)
        adapter.identity = PolicyIdentity(INTERFACE_VERSION, adapter.identity.model_digest,
                                          file_hash, OBSERVATION_SCHEMA, ACTION_SCHEMA,
                                          get_precision_contract(precision_mode).identity,
                                          precision_stability)
        return adapter

    def priors_and_value(self, observation, candidate_lists, selected_prefix=None):
        """Return legal-prefix log-probabilities and value for one observation.

        ``candidate_lists`` contains only native legal candidates for each
        sequential branch. Hidden state and optimizer/checkpoint metadata are
        never accepted by this interface.
        """
        if not candidate_lists:
            raise ValueError("at least one legal action branch is required")
        with torch.inference_mode():
            encoded = self.model.encode(observation.to(self.device))
            hidden = self.model.scorer.initial_state(1, device=self.device,
                                                      dtype=encoded.tokens.dtype)
            logprobs = []
            if len(candidate_lists) > 1 and selected_prefix is None:
                raise ValueError("multi-branch scoring requires an explicit selected prefix")
            selected_prefix = list(selected_prefix or [])
            if len(selected_prefix) > len(candidate_lists):
                raise ValueError("selected prefix is longer than the candidate path")
            for level, candidates in enumerate(candidate_lists):
                if not candidates:
                    raise ValueError("candidate branch is empty")
                table = self._helper._level_table([candidates])
                keys = self.model.scorer.build_keys(encoded.tokens, table.action_ids[:, 0],
                                                    table.entity_token[:, 0], table.move_token[:, 0])
                logits = self.model.scorer.score_keys(keys, hidden, table.mask[:, 0])
                logprob, probs, _entropy, _kl = self.model._branch_stats(logits, table.mask[:, 0])
                logprobs.append(logprob.detach().float().cpu())
                if level < len(selected_prefix):
                    pick = int(selected_prefix[level])
                    if pick < 0 or pick >= len(candidates):
                        raise ValueError("selected action index is outside the legal candidate list")
                elif len(candidate_lists) == 1:
                    pick = int(torch.argmax(logits, dim=-1).item())
                else:
                    raise ValueError("selected prefix does not cover every branch")
                selected = keys[:, pick]
                hidden = self.model.scorer.advance_keys(selected, hidden)
            value = self.model.value(encoded).detach().float().cpu()
        return {"identity": self.identity, "branch_logprobs": logprobs, "value": value}
