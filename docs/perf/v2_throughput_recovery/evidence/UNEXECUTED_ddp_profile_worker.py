"""Diagnostic-only wrapper: scoped profiler hook; learner math unchanged."""
import pathlib,sys,torch
root=pathlib.Path('/home/minidc/Desktop/PokeAgent3.0-V2-throughput');sys.path[:0]=[str(root),str(root/'engine/python')]
from agent.ppo.learner import PPOLearner
original=PPOLearner.update_ddp
def traced(self,*args,**kwargs):
    rank=__import__('os').environ['RANK'];out=root/'runs/perf/v2_recovery/ddp_profile_contract';out.mkdir(exist_ok=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
      schedule=torch.profiler.schedule(skip_first=2,wait=0,warmup=1,active=1,repeat=1),
      on_trace_ready=lambda p:p.export_chrome_trace(str(out/f'trace.rank{rank}.json'))) as profiler:
        hook=self.optimizer.register_step_post_hook(lambda *unused:profiler.step())
        try:return original(self,*args,**kwargs)
        finally:hook.remove()
PPOLearner.update_ddp=traced
from scripts.run_ddp_ppo import parse_args,worker
worker(parse_args())
