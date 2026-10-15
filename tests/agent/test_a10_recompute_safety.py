import copy
import math
import pytest
import torch
from agent.buffer.rollout_buffer import RolloutBuffer
from agent.types.observation import ObservationBatch
from agent.ppo.recompute_safety import compact_hash, row_contract_hash, evidence_passes, secondary_audit
from pa3_test_util import make_request, preview_branches, make_candidate_batch


def evidence():
    return dict(inputs_verified=True, stored=-2.0007, actor_replay=-2.0007,
                actor_repeat=-2.0007, batched_amp=-1.9993, isolated_amp=-2.0001,
                fp32_group=-2., fp32_isolated=-2.)


def test_secondary_accepts_only_explained_bounded_variation():
    row = evidence()
    assert abs(row['stored']-row['batched_amp']) > 1e-3
    assert evidence_passes(row, 1e-3)
    for key in ('stored', 'actor_repeat', 'fp32_group', 'isolated_amp', 'batched_amp'):
        bad = dict(row); bad[key] += 0.02
        assert not evidence_passes(bad, 1e-3)
    for key in ('stored', 'actor_replay', 'fp32_isolated', 'batched_amp'):
        bad = dict(row); bad[key] = float('nan')
        assert not evidence_passes(bad, 1e-3)
    assert not evidence_passes({**row, 'inputs_verified': False}, 1e-3)


def recorded(model):
    obs = ObservationBatch.dummy(3, seed=23)
    reqs = [make_request(obs.select([i]), preview_branches()) for i in range(3)]
    with torch.inference_mode():
        sampled = model.sample(obs, make_candidate_batch(reqs), generator=torch.Generator().manual_seed(91))
    buffer = RolloutBuffer()
    for i, req in enumerate(reqs):
        buffer.record(obs.select([i]), req, selected=sampled.selected[i, :req.branch_count].tolist(),
                      old_logprob=sampled.request_logprob[i].item(), value=0., match_id=i,
                      side=0, policy_id='current', opponent_policy_id='current', team_ids=(0, 1),
                      request_index=0, turn=0)
    buffer.actor_groups = [dict(start=0, stop=3, observation_hash=compact_hash(obs.to_compact_numpy()),
                               row_contract_hash=row_contract_hash(buffer.rows))]
    return buffer


@pytest.mark.parametrize('corruption', [None, 'logprob', 'observation', 'mask', 'prefix', 'missing'])
def test_actor_replay_integrity_and_rng(model_factory, corruption):
    model = model_factory(seed=11)
    buffer = recorded(model)
    if corruption == 'logprob': buffer.rows[0].old_logprob += .02
    if corruption == 'observation': buffer.observation_store._rows[0]['floats'][0, 0, 0] += .25
    if corruption == 'mask':
        row = buffer.rows[0]; mask = [list(m) for m in row.candidate_mask]
        mask[0][row.selected[0]] = False; row.candidate_mask = tuple(tuple(m) for m in mask)
    if corruption == 'prefix':
        row = buffer.rows[0]; row.selected = ((row.selected[0]+1) % len(row.candidate_mask[0]),) + row.selected[1:]
    if corruption == 'missing': buffer.actor_groups = []
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    logs = [r.old_logprob for r in buffer.rows]
    result = secondary_audit(model, buffer, buffer.rows, logs, torch.device('cpu'), 1e-3, force=True)
    assert result['accepted'] is (corruption is None)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert [r.old_logprob for r in buffer.rows] == logs


def test_secondary_budget_fails_closed(model_factory):
    model = model_factory(seed=11); buffer = recorded(model)
    result = secondary_audit(model, buffer, buffer.rows, [r.old_logprob+.02 for r in buffer.rows],
                             torch.device('cpu'), 1e-3, max_outliers=1)
    assert not result['accepted']
    assert result['reason'] == 'secondary audit budget exceeded'

@pytest.mark.parametrize('logs,budget', [([], 8), ([0., 0., 0.], 0)])
def test_secondary_input_shape_and_budget_fail_closed(model_factory, logs, budget):
    model = model_factory(seed=11); buffer = recorded(model)
    result = secondary_audit(model, buffer, buffer.rows, logs, torch.device('cpu'), 1e-3, max_outliers=budget)
    assert not result['accepted']
    assert result['reason'] == 'invalid secondary audit input or budget'


def test_missing_group_on_real_outlier_never_accepted(model_factory):
    model = model_factory(seed=11); buffer = recorded(model); buffer.actor_groups = []
    logs = [r.old_logprob + .0015 for r in buffer.rows]
    result = secondary_audit(model, buffer, buffer.rows, logs, torch.device('cpu'), 1e-3)
    assert result['required'] and not result['accepted']
    assert all(r['reason'] == 'original actor group unavailable' for r in result['rows'])
