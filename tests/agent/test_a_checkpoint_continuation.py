import copy
import random

import torch

from agent.ppo.experimental_checkpoint import capture_rng_state, restore_rng_state


def test_experimental_rng_checkpoint_restores_owned_streams():
    random.seed(20261011)
    torch.manual_seed(20261011)
    generator = torch.Generator(device="cpu").manual_seed(77)
    saved = capture_rng_state([generator])
    expected = (random.random(), torch.rand(3), torch.randperm(8, generator=generator))

    random.random()
    torch.rand(5)
    torch.randperm(8, generator=generator)
    restore_rng_state(copy.deepcopy(saved), [generator])
    actual = (random.random(), torch.rand(3), torch.randperm(8, generator=generator))
    assert actual[0] == expected[0]
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(actual[2], expected[2])
