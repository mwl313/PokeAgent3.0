import pytest,torch
from agent.types.observation import ObservationBatch

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_canonical_amp_rows_and_learner_gradient_identity(model_factory):
    model=model_factory(seed=19).cuda();model.configure_precision('D')
    obs=ObservationBatch.dummy(9,seed=41).to('cuda')
    model.encoder.canonical_batch_rows=8
    before=torch.cuda.get_rng_state().clone()
    with torch.inference_mode():
        full=model.encode(obs).tokens
        for index in [0,4,8]:
            one=model.encode(obs.select([index])).tokens
            torch.testing.assert_close(full[index:index+1],one,rtol=0,atol=0)
    assert torch.equal(before,torch.cuda.get_rng_state())
    batch=obs.narrow(0,8)
    model.encoder.canonical_batch_rows=0
    loss=model.encode(batch).tokens.square().mean();old=torch.autograd.grad(loss,tuple(model.parameters()),allow_unused=True)
    model.encoder.canonical_batch_rows=8
    loss=model.encode(batch).tokens.square().mean();new=torch.autograd.grad(loss,tuple(model.parameters()),allow_unused=True)
    assert all((a is None and b is None) or torch.equal(a,b) for a,b in zip(old,new))
