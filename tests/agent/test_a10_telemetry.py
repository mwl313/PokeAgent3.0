from agent.ppo.telemetry import append_iteration, read_iteration

def test_structured_telemetry_roundtrip(tmp_path):
 p=tmp_path/'metrics.json'; append_iteration(p,iteration=4,committed_matches=16384,rank=1,metrics={'policy_loss':.1,'entropy':.2,'approx_kl':.003})
 x=read_iteration(p); assert x['committed_matches']==16384 and x['metrics']['approx_kl']==.003
