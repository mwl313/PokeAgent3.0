"""M1: the manual FP32 gradient-SUM executor reproduces the global objective.

No process group is needed: the executor's formula is
``S_actor_local/A + value_coef*S_value_local/V`` with *global* A/V, and
``all_reduce(SUM)`` is exactly the sum of the per-rank gradients. The test
therefore accumulates two uneven shards sequentially on one model and compares
the flattened gradient against the single-process objective over the union.
"""

from __future__ import annotations

import torch

from agent.mock_engine import MockNativeEngine, collect_mock_rollout
from agent.model import PA3Config, build_model
from agent.ppo import PPOLearner, PPOConfig
from agent.ppo.ddp import assign_flat_gradients, flatten_gradients


def _config() -> PPOConfig:
    return PPOConfig(
        global_minibatch_size=64,
        microbatch_size=16,
        per_rank_minibatch_size=32,
        grad_accumulation_per_rank=2,
        ppo_epochs=1,
        sample_weighted_ddp_reduction=False,
    )


def _fixture():
    config = PA3Config(
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
    torch.manual_seed(20261009)
    model = build_model(config)
    engine = MockNativeEngine(num_teams=4, requests_per_match=3)
    buffer = collect_mock_rollout(engine, model, envs=12, target_matches=24)
    if len(buffer.rows) < 40:
        raise RuntimeError("fixture too small")
    return model, buffer


def _trajectory_shards(rows):
    groups: dict[tuple, list] = {}
    for row in rows:
        groups.setdefault((row.match_id, row.side), []).append(row)
    ordered = [groups[key] for key in sorted(groups)]
    shard0: list = []
    shard1: list = []
    for group in ordered:
        if len(shard0) + len(group) <= 28:
            shard0.extend(group)
        elif len(shard1) + len(group) <= 12:
            shard1.extend(group)
    return shard0, shard1


def test_flatten_assign_round_trip_covers_missing_grads():
    model = build_model(
        PA3Config(
            encoder_layers=1,
            d_model=32,
            attention_heads=2,
            head_dim=16,
            ffn_dim=64,
            prefix_hidden=32,
            critic_hidden=32,
        )
    )
    parameters = list(model.parameters())
    for parameter in parameters[:-1]:
        parameter.grad = torch.full_like(parameter, 3.0)
    # The last parameter deliberately has no gradient.
    flat = flatten_gradients(model)
    assert flat.shape[0] == sum(parameter.numel() for parameter in parameters)
    missing = flat[sum(p.numel() for p in parameters[:-1]):]
    assert torch.equal(missing, torch.zeros_like(missing))
    assign_flat_gradients(model, torch.ones_like(flat))
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.equal(parameter.grad, torch.ones_like(parameter))


def test_manual_sum_of_rank_gradients_matches_single_process_objective():
    model, buffer = _fixture()
    rows = list(buffer.rows)
    shard0, shard1 = _trajectory_shards(rows)
    reference_rows = shard0 + shard1
    learner = PPOLearner(model, _config(), device="cpu", amp=False)
    full = learner.prepare_batch(buffer, rows=reference_rows)
    full = full.to("cpu")

    # Single-process reference: one objective over the union of both shards.
    learner.optimizer.zero_grad(set_to_none=True)
    terms = learner._forward_terms(full)
    loss = terms["loss_unscaled"] + learner.config.value_coefficient * terms["value_mean"]
    loss.backward()
    reference = flatten_gradients(model)

    # Manual executor: two local sums over global A/V, accumulated (SUM). The
    # advantages must be the *global* ones (the executor uses
    # prepare_streaming_ddp), not a per-shard normalization.
    actor_total = float(terms["actor_count"].item())
    value_total = float(terms["value_count"].item())
    plan = learner.prepare_streaming_ddp(buffer, rows=reference_rows)
    learner.optimizer.zero_grad(set_to_none=True)
    offset = 0
    for shard in (shard0, shard1):
        local = buffer.to_batch(shard, device="cpu")
        local.advantages = plan.advantages[offset:offset + len(shard)]
        offset += len(shard)
        local_terms = learner._forward_terms(local)
        local_loss = (
            local_terms["loss_unscaled"] * local_terms["actor_count"] / max(actor_total, 1.0)
            + learner.config.value_coefficient
            * local_terms["value_mean"]
            * local_terms["value_count"]
            / max(value_total, 1.0)
        )
        local_loss.backward()
    manual = flatten_gradients(model)

    difference = (manual - reference).abs()
    scale = max(float(reference.abs().max()), 1e-6)
    assert float(difference.max()) <= 1e-5 * scale + 1e-6


def test_manual_executor_runs_single_process_and_matches_the_standard_update():
    """The full executor path (fixed steps, one shared Adam step) is exercised."""
    import copy

    model, buffer = _fixture()
    rows = list(buffer.rows)
    shard0, shard1 = _trajectory_shards(rows)
    reference_rows = shard0 + shard1

    reference_model = copy.deepcopy(model)
    reference = PPOLearner(reference_model, _config(), device="cpu", amp=False)
    batch = reference.prepare_batch(buffer, rows=reference_rows)
    reference_report = reference.update(
        batch, committed_matches=1_000, generator=torch.Generator().manual_seed(3)
    )

    manual_model = copy.deepcopy(model)
    manual_config = PPOConfig(
        global_minibatch_size=64,
        microbatch_size=16,
        per_rank_minibatch_size=64,
        grad_accumulation_per_rank=4,
        ppo_epochs=1,
        sample_weighted_ddp_reduction=False,
    )
    learner = PPOLearner(manual_model, manual_config, device="cpu", amp=False)
    plan = learner.prepare_streaming_ddp(buffer, rows=reference_rows)
    report = learner.update_manual_allreduce(
        plan,
        committed_matches=1_000,
        generator=torch.Generator().manual_seed(3),
    )
    assert report.optimizer_steps == reference.optimizer_steps
    assert report.optimizer_steps_skipped == 0
    assert report.epochs_run == reference_report.epochs_run
    for name, value in reference.model.state_dict().items():
        assert torch.allclose(
            value, manual_model.state_dict()[name], atol=1e-6, rtol=1e-5
        ), name
