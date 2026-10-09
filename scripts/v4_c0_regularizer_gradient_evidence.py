"""C0 evidence: entropy / uniform-KL gradient before and after the fix.

Runs the real PA3-8M architecture on a mock rollout fixture twice with the same
initial weights:

* ``prefix_detached`` reproduces the pre-fix ``_branch_stats`` (both
  regularizer terms detached). Its scalar losses are the reference values.
* ``fixed`` uses the current, graph-attached ``_branch_stats``.

The script records scalar parity plus the parameter-group gradient norms and
writes a compact JSON artifact under ``runs/perf/v4/`` (no large tensors).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import torch

from agent.mock_engine.mock_engine import MockNativeEngine, collect_mock_rollout
from agent.model import PA3Config, build_model
from agent.model.pa3_model import PA3Model
from agent.model.scorer import masked_log_softmax, probabilities
from agent.ppo import PPOConfig, PPOLearner


def _pre_fix_branch_stats(self, logits: torch.Tensor, mask: torch.Tensor):
    """The exact pre-fix implementation (detached regularizers)."""
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
    uniform_kl = (-log_k - mean_log_prob) / normalizer
    uniform_kl = torch.where(
        k >= 2, uniform_kl.expand_as(entropy), torch.zeros_like(entropy)
    )
    return log_prob, probs, entropy_normalized, uniform_kl


def _small_config(seed: int) -> PA3Config:
    return PA3Config(
        seed=seed,
        encoder_layers=2,
        d_model=64,
        attention_heads=2,
        head_dim=32,
        ffn_dim=128,
        category_vocab=256,
        category_embed_dim=8,
        role_vocab=12,
        prefix_hidden=64,
        critic_hidden=64,
    )


def _group_gradients(model: PA3Model) -> dict[str, torch.Tensor]:
    groups = {"encoder": [], "scorer": [], "value_head": []}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        target = (
            "encoder"
            if name.startswith("encoder")
            else "scorer"
            if name.startswith("scorer")
            else "value_head"
        )
        groups[target].append(parameter.grad.detach().float().reshape(-1))
    return {
        group: torch.cat(chunks) if chunks else torch.zeros(0)
        for group, chunks in groups.items()
    }


def _group_norms(gradients: dict[str, torch.Tensor]) -> dict[str, float]:
    return {group: float(tensor.norm()) for group, tensor in gradients.items()}


def _single_term_grad_norm(learner: PPOLearner, batch, attribute: str) -> float:
    learner.optimizer.zero_grad(set_to_none=True)
    encoded = learner.model.encode(batch.observation)
    evaluation = learner.model.evaluate_encoded(
        encoded, batch.candidates, selected=batch.candidates.selected
    )
    term = getattr(evaluation, f"request_{attribute}")
    actor = (batch.actor_mask & batch.row_valid).to(term.dtype)
    (term * actor).sum().backward()
    norms = [
        parameter.grad.detach().float().norm()
        for name, parameter in learner.model.named_parameters()
        if name.startswith("scorer") and parameter.grad is not None
    ]
    learner.optimizer.zero_grad(set_to_none=True)
    return float(torch.stack(norms).norm()) if norms else 0.0


def _run_once(learner: PPOLearner, batch, detached: bool) -> dict:
    original = PA3Model._branch_stats
    if detached:
        PA3Model._branch_stats = _pre_fix_branch_stats
    try:
        learner.optimizer.zero_grad(set_to_none=True)
        terms = learner._forward_terms(batch)
        loss = terms["loss_unscaled"]
        loss.backward()
        gradients = _group_gradients(learner.model)
        scalar_keys = (
            "policy_mean",
            "value_mean_detached",
            "entropy_mean",
            "uniform_kl_mean",
            "approx_kl_mean",
            "ratio_mean",
            "clip_fraction",
        )
        return {
            "loss": float(loss.detach()),
            "scalars": {key: float(terms[key]) for key in scalar_keys},
            "actor_count": float(terms["actor_count"]),
            "value_count": float(terms["value_count"]),
            "gradients": gradients,
        }
    finally:
        PA3Model._branch_stats = original


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("runs/perf/v4/c0_regularizer_gradient_evidence.json"),
    )
    args = parser.parse_args()

    seed = 20261009
    torch.manual_seed(seed)
    model = build_model(_small_config(seed))
    engine = MockNativeEngine(num_teams=8, requests_per_match=4)
    buffer = collect_mock_rollout(
        engine, model, envs=4, target_matches=8, policy_id="current"
    )
    config = PPOConfig(
        global_minibatch_size=16,
        microbatch_size=16,
        entropy_coefficient=0.01,
        uniform_kl_coefficient=0.001,
    )
    learner = PPOLearner(model, config, amp=False)
    prepared = learner.prepare_batch(buffer)
    batch = next(iter(prepared.iter_minibatches(len(prepared), shuffle=False)))

    initial = {
        name: parameter.detach().clone()
        for name, parameter in learner.model.named_parameters()
    }
    detached = _run_once(learner, batch, detached=True)
    fixed = _run_once(learner, batch, detached=False)
    entropy_only_grad = _single_term_grad_norm(learner, batch, "entropy")
    kl_only_grad = _single_term_grad_norm(learner, batch, "uniform_kl")

    detached_norms = _group_norms(detached["gradients"])
    fixed_norms = _group_norms(fixed["gradients"])
    delta_norms = {
        group: float((fixed["gradients"][group] - detached["gradients"][group]).norm())
        for group in fixed["gradients"]
    }
    scalar_parity = {
        key: {
            "detached": detached["scalars"][key],
            "fixed": fixed["scalars"][key],
            "abs_diff": abs(detached["scalars"][key] - fixed["scalars"][key]),
        }
        for key in detached["scalars"]
    }
    payload = {
        "gate": "C0",
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "working_tree_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        ),
        "seed": seed,
        "model_parameter_count": int(model.parameter_count()),
        "fixture_rows": int(len(buffer)),
        "minibatch_rows": int(len(batch)),
        "detached_prefix_loss": detached["loss"],
        "fixed_prefix_loss": fixed["loss"],
        "loss_abs_diff": abs(detached["loss"] - fixed["loss"]),
        "scalar_parity": scalar_parity,
        "gradient_norms": {
            "prefix_detached": detached_norms,
            "fixed": fixed_norms,
            "delta_norm": delta_norms,
            "delta_relative": {
                group: (
                    delta_norms[group] / detached_norms[group]
                    if detached_norms[group] > 0
                    else None
                )
                for group in delta_norms
            },
        },
        "regularizer_only_scorer_grad_norm": {
            "entropy_term": entropy_only_grad,
            "uniform_kl_term": kl_only_grad,
        },
        "precondition": {
            "detached_scorer_grad_from_regularizers": 0.0,
            "note": "pre-fix regularizer gradient is exactly zero because both factors were detached",
        },
        "model_state_restored": all(
            torch.equal(initial[name], parameter.detach())
            for name, parameter in learner.model.named_parameters()
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
